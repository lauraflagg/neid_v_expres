"""Compute coadded spectra and save outputs in a Julia-readable HDF5 file.

CLI version that reads overlap windows from a CSV file.

Expected overlaps CSV format
----------------------------
A header row with at least two columns (defaults shown):
  start_jd,end_jd
followed by rows of float values.

Memory usage
------------
Optimized for low RAM:
- Coaddition is done in a streaming fashion (no stacking all exposures into a 3-D cube).
- Results are written to HDF5 incrementally per overlap window.

Parallel processing
-------------------
Supports multiprocessing to speed up coaddition across overlap windows.
- Computation (FITS reading + resampling + coadding) is done in worker processes.
- HDF5 writing is done in the main process (safe with h5py).

Resampling (SpectRes)
---------------------
Before coadding, each exposure is resampled onto a common wavelength grid using SpectRes.
The common grid for each overlap window is defined as the wavelength array of the first exposure
in that window ("first wavelength grid of each day").

NaN/inf handling
----------------
During coaddition, any pixel with NaN/inf flux or NaN/inf uncertainty is ignored for that exposure.
Pixels with uncertainty <= 0 are ignored. If no valid exposures contribute to a pixel, the
coadded flux and uncertainty are set to NaN for that pixel.

Outputs
-------
An HDF5 file with groups for NEID and EXPRES, each containing:
  - wls   : group of wavelength arrays (datasets 0000, 0001, ...)
  - coadds: group of coadded flux arrays (datasets 0000, 0001, ...)
  - uncs  : group of uncertainty arrays (datasets 0000, 0001, ...)
  - date_sets: per-window lists of (JD, file_loc)

Run
---
  python get_coadds_julia_cli_csv.py \
      --overlaps-csv window_overlaps.csv \
      --dates-and-dirs dates_and_dirs.pkl \
      --out coadds_outputs.h5 \
      --workers 8

Dependencies
------------
  pip install spectres
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Sequence, Tuple

from csv import DictReader

import numpy as np
import pickle
import h5py
from astropy.io import fits
from astropy.time import Time

# SpectRes: resample flux + uncertainties onto a new wavelength grid.
# See: spectres.spectres(new_wavs, spec_wavs, spec_fluxes, spec_errs=None, fill=None, verbose=True)
# Returns (new_fluxes, new_errs) if spec_errs is provided.
import spectres


# -----------------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------------

JD = float
FileLoc = str
Obs = Tuple[JD, FileLoc]
DateSet = List[Obs]
DateSets = List[DateSet]


# -----------------------------------------------------------------------------
# Input: overlaps CSV
# -----------------------------------------------------------------------------

def load_overlaps_csv(path: str, start_col: str = "start_jd", end_col: str = "end_jd"):
    """Load overlap windows from a CSV with columns `start_col` and `end_col`."""

    overlaps: List[Tuple[float, float]] = []
    with open(path, "r", newline="") as f:
        reader = DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV appears to have no header row.")
        if start_col not in reader.fieldnames or end_col not in reader.fieldnames:
            raise ValueError(
                f"CSV must contain columns '{start_col}' and '{end_col}'. Found: {reader.fieldnames}"
            )
        for row in reader:
            st = float(row[start_col])
            en = float(row[end_col])
            overlaps.append((st, en))
    return overlaps


# -----------------------------------------------------------------------------
# FITS readers
# -----------------------------------------------------------------------------

def read_expr_fits(hdul: fits.HDUList):
    """Return (flux, wavelength, uncertainty) from an EXPRES reduced FITS."""
    flux = hdul[1].data["spectrum"]
    wave = hdul[1].data["bary_wavelength"]
    unc = hdul[1].data["uncertainty"]
    return flux, wave, unc


def read_neid_fits(hdul: fits.HDUList):
    """Return (wavelength, flux, uncertainty) from a NEID reduced FITS."""
    flux = hdul["SCIFLUX"].data / hdul["SCIBLAZE"].data
    uncert = np.sqrt(hdul["SCIVAR"].data) / hdul["SCIBLAZE"].data
    wave = hdul["SCIWAVE"].data
    return wave, flux, uncert


# -----------------------------------------------------------------------------
# Date-set construction
# -----------------------------------------------------------------------------

def build_date_sets(
    overlaps: Sequence[Tuple[JD, JD]],
    dates_and_dirs: Dict[str, Sequence],
) -> Tuple[DateSets, DateSets]:
    """Build NEID and EXPRES date_sets for each overlap window."""

    neid_date_sets: DateSets = []
    expres_date_sets: DateSets = []

    neid_dates = dates_and_dirs["NEID_dates"]
    neid_locs = dates_and_dirs["NEID_file_locs"]
    expres_dates = dates_and_dirs["EXPRES_dates"]
    expres_locs = dates_and_dirs["EXPRES_file_locs"]

    for st, end in overlaps:
        t_start = Time(st, format="jd")
        t_end = Time(end, format="jd")

        neids: DateSet = []
        for jd, floc in zip(neid_dates, neid_locs):
            t_date = Time(jd, format="jd")
            if (t_date >= t_start) & (t_date <= t_end):
                neids.append((float(jd), str(floc)))

        expreses: DateSet = []
        for jd, floc in zip(expres_dates, expres_locs):
            t_date = Time(jd, format="jd")
            if (t_date >= t_start) & (t_date <= t_end):
                expreses.append((float(jd), str(floc)))

        neid_date_sets.append(neids)
        expres_date_sets.append(expreses)

    return neid_date_sets, expres_date_sets


# -----------------------------------------------------------------------------
# SpectRes resampling helpers
# -----------------------------------------------------------------------------

def _ensure_increasing(w: np.ndarray, *arrs: np.ndarray):
    """Ensure wavelength axis is increasing. If decreasing, reverse wavelength and matching arrays.

    Returns (w_fixed, arrs_fixed, reversed_flag)
    """
    w = np.asarray(w)
    if w.size < 2:
        return w, arrs, False

    # Determine monotonicity along last axis
    if np.nanmedian(np.diff(w)) < 0:
        w2 = np.flip(w, axis=-1)
        arrs2 = tuple(np.flip(a, axis=-1) for a in arrs)
        return w2, arrs2, True
    return w, arrs, False


def resample_to_grid(
    new_wave: np.ndarray,
    old_wave: np.ndarray,
    flux: np.ndarray,
    unc: np.ndarray,
    *,
    fill_value: float = np.nan,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample (flux, unc) from old_wave to new_wave using SpectRes.

    Supports 1D or 2D echelle-style arrays (orders x pixels).
    For 2D, resampling is applied order-by-order.
    """

    new_wave = np.asarray(new_wave)
    old_wave = np.asarray(old_wave)
    flux = np.asarray(flux, dtype=float)
    unc = np.asarray(unc, dtype=float)

    if new_wave.ndim == 1:
        # Ensure increasing wavelength for SpectRes; then reverse outputs back if needed
        new_w, (old_w, f, e), rev_new = _ensure_increasing(new_wave, old_wave, flux, unc)
        old_w2, (f2, e2), rev_old = _ensure_increasing(old_w, f, e)

        out = spectres.spectres(new_w, old_w2, f2, spec_errs=e2, fill=fill_value, verbose=False)
        new_f, new_e = out

        # Undo reversals to match original new_wave orientation
        if rev_old:
            new_f = np.flip(new_f, axis=-1)
            new_e = np.flip(new_e, axis=-1)
        if rev_new:
            new_f = np.flip(new_f, axis=-1)
            new_e = np.flip(new_e, axis=-1)
        return np.asarray(new_f), np.asarray(new_e)

    if new_wave.ndim == 2:
        # Order-by-order
        n_orders = new_wave.shape[0]
        new_flux = np.empty_like(flux, dtype=float)
        new_unc = np.empty_like(unc, dtype=float)

        for o in range(n_orders):
            nf, ne = resample_to_grid(new_wave[o], old_wave[o], flux[o], unc[o], fill_value=fill_value)
            new_flux[o] = nf
            new_unc[o] = ne

        return new_flux, new_unc

    raise ValueError(f"Unsupported wavelength grid dimension: new_wave.ndim={new_wave.ndim}")


# -----------------------------------------------------------------------------
# Coaddition (low-memory, NaN-safe, with SpectRes resampling)
# -----------------------------------------------------------------------------

def coadd_one_window(window_obs: DateSet, instrument: str):
    """Coadd a single overlap window in a streaming fashion.

    Common wavelength grid = wavelength array of the first exposure in the window.
    All subsequent exposures are resampled onto this grid using SpectRes before accumulating.
    """

    if len(window_obs) == 0:
        return np.array([]), np.array([]), np.array([])

    sum_w = None
    sum_wflux = None
    wave_ref = None

    # Reusable weight array (initialized after first exposure)
    w = None

    for j, (jd, floc) in enumerate(window_obs):
        with fits.open(floc, memmap=True) as hdul:
            if instrument.upper() == "NEID":
                wave, flux, unc = read_neid_fits(hdul)
            elif instrument.upper() == "EXPRES":
                flux, wave, unc = read_expr_fits(hdul)
            else:
                raise ValueError(f"Unknown instrument: {instrument}")

        flux = np.asarray(flux, dtype=float)
        unc = np.asarray(unc, dtype=float)
        wave = np.asarray(wave)

        if wave_ref is None:
            # Common grid is the first wavelength grid of the window
            wave_ref = wave
            sum_w = np.zeros_like(unc, dtype=float)
            sum_wflux = np.zeros_like(flux, dtype=float)
            w = np.zeros_like(unc, dtype=float)
        else:
            # Resample to common grid
            flux, unc = resample_to_grid(wave_ref, wave, flux, unc, fill_value=np.nan)

        # Shape consistency check
        if flux.shape != sum_wflux.shape or unc.shape != sum_w.shape:
            raise ValueError(
                f"Shape mismatch after resampling: expected {sum_w.shape}, got flux {flux.shape}, unc {unc.shape} for {floc}"
            )

        # NaN-safe weighting
        valid = np.isfinite(flux) & np.isfinite(unc) & (unc > 0)

        w.fill(0.0)
        np.square(unc, out=w, where=valid)        # w = unc^2
        np.divide(1.0, w, out=w, where=valid)     # w = 1/unc^2

        sum_w += w
        sum_wflux[valid] += flux[valid] * w[valid]

    ok = sum_w > 0

    coadd_flux = np.full_like(sum_wflux, np.nan, dtype=float)
    np.divide(sum_wflux, sum_w, out=coadd_flux, where=ok)

    coadd_unc = np.full_like(sum_w, np.nan, dtype=float)
    np.sqrt(1.0 / sum_w, out=coadd_unc, where=ok)

    return np.asarray(wave_ref), coadd_flux, coadd_unc


# --- multiprocessing worker must be top-level for pickling ---

def _worker_task(args):
    """Worker: coadd one window.

    args = (index, instrument, window_obs)
    returns (index, wave_ref, coadd_flux, coadd_unc)
    """
    i, instrument, window_obs = args
    wave_ref, coadd_flux, coadd_unc = coadd_one_window(window_obs, instrument=instrument)
    return i, wave_ref, coadd_flux, coadd_unc


# -----------------------------------------------------------------------------
# HDF5 writer helpers
# -----------------------------------------------------------------------------

def _vlen_str_dtype():
    return h5py.string_dtype(encoding="utf-8")


def _init_list_group(parent: h5py.Group, name: str, length: int) -> h5py.Group:
    g = parent.create_group(name)
    g.attrs["layout"] = "list"
    g.attrs["length"] = int(length)
    return g


def _write_date_sets(parent: h5py.Group, date_sets: DateSets):
    ds_group = parent.create_group("date_sets")
    ds_group.attrs["layout"] = "list_of_lists"
    ds_group.attrs["length"] = len(date_sets)

    for i, window in enumerate(date_sets):
        wgrp = ds_group.create_group(f"{i:04d}")
        jds = np.array([obs[0] for obs in window], dtype=float)
        wgrp.create_dataset("jds", data=jds)
        flocs = np.array([obs[1] for obs in window], dtype=_vlen_str_dtype())
        wgrp.create_dataset("file_locs", data=flocs)


# -----------------------------------------------------------------------------
# Main save routine (incremental + optional parallel)
# -----------------------------------------------------------------------------

def save_coadds_hdf5_incremental(
    out_path: str,
    *,
    overlaps: Sequence[Tuple[JD, JD]] | None,
    neid_date_sets: DateSets,
    expres_date_sets: DateSets,
    source_dates_and_dirs: str | None = None,
    source_overlaps_csv: str | None = None,
    workers: int = 1,
    chunksize: int = 1,
    mp_start_method: str = "spawn",
) -> str:
    """Compute coadds and write HDF5 incrementally."""

    import multiprocessing as mp

    with h5py.File(out_path, "w") as f:
        f.attrs["created_utc"] = datetime.utcnow().isoformat() + "Z"
        f.attrs["format"] = "coadds_hdf5_v1"
        if source_dates_and_dirs is not None:
            f.attrs["dates_and_dirs_source"] = str(source_dates_and_dirs)
        if source_overlaps_csv is not None:
            f.attrs["overlaps_csv_source"] = str(source_overlaps_csv)

        if overlaps is not None:
            f.create_dataset("overlaps", data=np.asarray(overlaps, dtype=float))

        for inst_name, date_sets in (("NEID", neid_date_sets), ("EXPRES", expres_date_sets)):
            ginst = f.create_group(inst_name)
            _write_date_sets(ginst, date_sets)

            nwin = len(date_sets)
            gwls = _init_list_group(ginst, "wls", nwin)
            gco = _init_list_group(ginst, "coadds", nwin)
            gun = _init_list_group(ginst, "uncs", nwin)

            tasks = [(i, inst_name, date_sets[i]) for i in range(nwin)]

            if workers and workers > 1:
                ctx = mp.get_context(mp_start_method)
                with ctx.Pool(processes=workers, maxtasksperchild=50) as pool:
                    for i, wave_ref, coadd_flux, coadd_unc in pool.imap(_worker_task, tasks, chunksize=chunksize):
                        gwls.create_dataset(f"{i:04d}", data=np.asarray(wave_ref), compression="gzip", shuffle=True)
                        gco.create_dataset(f"{i:04d}", data=np.asarray(coadd_flux), compression="gzip", shuffle=True)
                        gun.create_dataset(f"{i:04d}", data=np.asarray(coadd_unc), compression="gzip", shuffle=True)
            else:
                for i, instrument, window_obs in tasks:
                    wave_ref, coadd_flux, coadd_unc = coadd_one_window(window_obs, instrument=instrument)
                    gwls.create_dataset(f"{i:04d}", data=np.asarray(wave_ref), compression="gzip", shuffle=True)
                    gco.create_dataset(f"{i:04d}", data=np.asarray(coadd_flux), compression="gzip", shuffle=True)
                    gun.create_dataset(f"{i:04d}", data=np.asarray(coadd_unc), compression="gzip", shuffle=True)

    return out_path


# -----------------------------------------------------------------------------
# Run everything
# -----------------------------------------------------------------------------

def run_coadds_and_save(
    overlaps: Sequence[Tuple[JD, JD]],
    dates_and_dirs_pkl: str = "dates_and_dirs.pkl",
    out_h5: str = "coadds_outputs.h5",
    overlaps_csv: str | None = None,
    workers: int = 1,
    chunksize: int = 1,
    mp_start_method: str = "spawn",
) -> str:
    """Load dates_and_dirs.pkl, build date_sets, compute coadds, and write HDF5."""

    with open(dates_and_dirs_pkl, "rb") as f:
        dates_and_dirs = pickle.load(f)

    neid_date_sets, expres_date_sets = build_date_sets(overlaps, dates_and_dirs)

    return save_coadds_hdf5_incremental(
        out_h5,
        overlaps=overlaps,
        neid_date_sets=neid_date_sets,
        expres_date_sets=expres_date_sets,
        source_dates_and_dirs=dates_and_dirs_pkl,
        source_overlaps_csv=overlaps_csv,
        workers=workers,
        chunksize=chunksize,
        mp_start_method=mp_start_method,
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Compute NEID/EXPRES coadded spectra for overlap windows and save outputs "
            "(wavelengths, coadds, uncertainties, and date_sets) to a Julia-readable HDF5 file."
        )
    )

    parser.add_argument(
        "--overlaps-csv",
        default="window_overlaps.csv",
        help=(
            "CSV file containing overlap windows with headers 'start_jd' and 'end_jd' "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--start-col",
        default="start_jd",
        help="Column name for window start (default: %(default)s)",
    )
    parser.add_argument(
        "--end-col",
        default="end_jd",
        help="Column name for window end (default: %(default)s)",
    )
    parser.add_argument(
        "--dates-and-dirs",
        default="dates_and_dirs.pkl",
        help="Path to dates_and_dirs.pkl (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        default="coadds_outputs.h5",
        help="Output HDF5 path (default: %(default)s)",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for parallel coaddition (default: %(default)s).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1,
        help="Chunksize for multiprocessing imap (default: %(default)s).",
    )
    parser.add_argument(
        "--mp-start-method",
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="Multiprocessing start method (default: %(default)s).",
    )

    args = parser.parse_args()

    overlaps = load_overlaps_csv(args.overlaps_csv, start_col=args.start_col, end_col=args.end_col)

    out_path = run_coadds_and_save(
        overlaps,
        dates_and_dirs_pkl=args.dates_and_dirs,
        out_h5=args.out,
        overlaps_csv=args.overlaps_csv,
        workers=args.workers,
        chunksize=args.chunksize,
        mp_start_method=args.mp_start_method,
    )

    print(out_path)
