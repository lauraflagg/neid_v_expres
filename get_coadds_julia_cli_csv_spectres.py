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

Test mode
---------
Use `--test-mode` to process only the first five overlap windows. Test outputs
are written separately with a `_test5` suffix, so the normal output is not
overwritten. Example commands:

  # Use the default input/output filenames; creates coadds_outputs_test5.h5
  python get_coadds_julia_cli_csv_spectres.py --test-mode

  # Specify test inputs, output, and worker count
  python get_coadds_julia_cli_csv_spectres.py \\
      --overlaps-csv window_overlaps.csv \\
      --dates-and-dirs dates_and_dirs.pkl \\
      --out coadds_outputs.h5 \\
      --workers 2 \\
      --test-mode

Dependencies
------------
  pip install spectres
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Sequence, Tuple

from csv import DictReader, writer

import numpy as np
import pickle
import h5py
from astropy.io import fits
from astropy.time import Time

# SpectRes: resample flux + uncertainties onto a new wavelength grid.
# See: spectres.spectres(new_wavs, spec_wavs, spec_fluxes, spec_errs=None, fill=None, verbose=True)
# Returns (new_fluxes, new_errs) if spec_errs is provided.
import spectres

from astropy import constants, units

c_kms=constants.c.value/10**3


# -----------------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------------
# One observation is (Julian date scalar, FITS file-location string). A DateSet
# is the list of observations belonging to one overlap window; DateSets contains
# one such list per window.

JD = float
FileLoc = str
Obs = Tuple[JD, FileLoc]
DateSet = List[Obs]
DateSets = List[DateSet]


# -----------------------------------------------------------------------------
# Input: overlaps CSV
# -----------------------------------------------------------------------------
# Reads the time boundaries that define which observations are grouped together.

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
# These functions normalize the instrument-specific FITS layouts to the common
# return order (wavelength, flux, uncertainty), generally with order x pixel
# arrays. They do not coadd; coadd_one_window handles the combination.

def read_expr_fits(hdul: fits.HDUList):
    """Return (flux, wavelength, uncertainty) from an EXPRES reduced FITS."""
    flux = hdul[1].data["spectrum"]
    wave = hdul[1].data["bary_wavelength"]
    unc = hdul[1].data["uncertainty"]
    return wave, flux,  unc


def read_neid_fits(hdul: fits.HDUList):
    order_start=4
    order_end=-5
    """Return (wavelength, flux, uncertainty) from a NEID reduced FITS."""
    flux = (hdul["SCIFLUX"].data / hdul["SCIBLAZE"].data)[order_start:order_end]
    uncert = (np.sqrt(hdul["SCIVAR"].data) / hdul["SCIBLAZE"].data)[order_start:order_end]
    wave0 = hdul["SCIWAVE"].data[order_start:order_end]
    head=hdul['PRIMARY'].header
    rvcors=[]
    for i in range(82,143):
        strfmt=str(i).zfill(3)
        #print(strfmt)
        rvcors.append(head['SSBRV'+strfmt]) #in km/s
    
    barycor = np.nanmedian(rvcors)
    wave=wave0*(1+barycor/c_kms) #apply barycentric correction
    return wave, flux, uncert


# -----------------------------------------------------------------------------
# Date-set construction
# -----------------------------------------------------------------------------
# For each overlap interval, select the NEID and EXPRES files whose Julian dates
# fall inclusively between the start and end times.

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
# These helpers make wavelength order monotonic and resample each order
# independently. They preserve the distinction between orders; resampling is
# along the pixel/wavelength axis, not across orders.

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

    For 2D, resampling is applied order-by-order.
    """

    new_wave = np.asarray(new_wave)
    old_wave = np.asarray(old_wave)
    flux = np.asarray(flux, dtype=float)
    unc = np.asarray(unc, dtype=float)

    #check shape is 2d
    if old_wave.ndim == 1:
        print("old_wave is 1d")
        print(old_wave.shape)


    if new_wave.ndim == 2:
        # Order-by-order
        n_orders = new_wave.shape[0]
        new_flux = np.empty_like(flux, dtype=float)
        new_unc = np.empty_like(unc, dtype=float)

        for o in range(n_orders):
            #print(o, n_orders)
            nf, ne = spectres.spectres(np.array(new_wave[o]), np.array(old_wave[o]), np.array(flux[o]), np.array(unc[o]), fill=fill_value)
            new_flux[o] = nf
            new_unc[o] = ne

        return new_flux, new_unc

    raise ValueError(f"Unsupported wavelength grid dimension: new_wave.ndim={new_wave.ndim}")


# -----------------------------------------------------------------------------
# Coaddition (low-memory, NaN-safe, with SpectRes resampling)
# -----------------------------------------------------------------------------

def coadd_one_window(window_obs: DateSet, instrument: str):
    """Coadd all observations in one overlap window, one array element at a time.

    Shapes:
      * ``window_obs`` has N entries: (JD scalar, file-location string).
      * Each loaded wavelength, flux, and uncertainty array is expected to have
        shape S. For an echelle spectrum, S is typically (n_orders, n_pixels).
      * Every accumulator and output array has shape S. Thus [order, column]
        is accumulated only with the corresponding [order, column] from other
        exposures; no weights are summed across different pixels.
      * The resampling helper in this script supports a 2D wavelength grid.

    At a given array position p, the inverse-variance weight of exposure k is
    w[k,p] = 1 / unc[k,p]**2. The coadd is
        flux_coadd[p] = sum_k(w[k,p] * flux[k,p]) / sum_k(w[k,p])
    and, assuming independent measurement errors,
        unc_coadd[p] = sqrt(1 / sum_k(w[k,p])).
    Flux uncertainties are assumed to use the same units as flux; weights
    therefore have units 1/(flux unit)**2.
    """

    # No observations means there is no reference wavelength grid or spectrum.
    # Keep the return convention: three empty 1D arrays.
    if len(window_obs) == 0:
        return np.array([]), np.array([]), np.array([]), []

    # Per-pixel running totals, all with spectrum shape S:
    #   sum_w[p]     = sum over exposures k of 1 / unc[k,p]**2
    #   sum_wflux[p] = sum over exposures k of flux[k,p] / unc[k,p]**2
    # They are initialized after the first file reveals the array shape.
    sum_w = None
    sum_wflux = None

    # The first exposure defines the common wavelength grid. Its shape is S.
    wave_ref = None

    # Track files with at least one valid pixel in the final coadd.
    contributing_files: List[str] = []

    # Reused scratch array of shape S for the current exposure's weights.
    w = None

    # Stream over exposures rather than loading all N spectra into a 3D cube.
    for j, (jd, floc) in enumerate(window_obs):
        # Open one FITS file at a time; the context manager closes it after read.
        with fits.open(floc, memmap=True) as hdul:
            # Instrument-specific readers return (wavelength, flux, uncertainty).
            if instrument.upper() == "NEID":
                wave, flux, unc = read_neid_fits(hdul)
            elif instrument.upper() == "EXPRES":
                wave, flux, unc = read_expr_fits(hdul)
            else:
                raise ValueError(f"Unknown instrument: {instrument}")

        # Ensure arithmetic arrays are floating point; their shapes remain S.
        flux = np.asarray(flux, dtype=float)
        unc = np.asarray(unc, dtype=float)
        wave = np.asarray(wave)

        if wave_ref is None:
            # First exposure: adopt its wavelength array (shape S) as the grid.
            wave_ref = wave

            # Allocate zero-valued per-pixel accumulators and scratch weights,
            # each shape S. Float dtype is needed for sums/division.
            sum_w = np.zeros_like(unc, dtype=float)
            sum_wflux = np.zeros_like(flux, dtype=float)
            w = np.zeros_like(unc, dtype=float)
        else:
            # Later exposures: interpolate/resample their flux and uncertainty
            # onto wave_ref. The returned flux and unc should both have shape S.
            # Locations outside spectral coverage are filled with NaN.
            flux, unc = resample_to_grid(wave_ref, wave, flux, unc, fill_value=np.nan)

        # Require resampled arrays to match accumulator shape S. This check
        # prevents accidental broadcasting or combining incompatible grids.
        if flux.shape != sum_wflux.shape or unc.shape != sum_w.shape:
            raise ValueError(
                f"Shape mismatch after resampling: expected {sum_w.shape}, got flux {flux.shape}, unc {unc.shape} for {floc}"
            )

        # Boolean mask with shape S: only finite flux, finite uncertainty,
        # and strictly positive uncertainty are allowed to contribute.
        valid = np.isfinite(flux) & np.isfinite(unc) & (unc > 0)
        if np.any(valid):
            contributing_files.append(str(floc))

        # Reset scratch weights for this exposure. Invalid pixels stay at zero.
        w.fill(0.0)

        # At valid pixels, first calculate unc**2, then replace it with
        # 1/unc**2. Both operations are elementwise over shape S.
        np.square(unc, out=w, where=valid)        # w[p] = unc[p]**2
        np.divide(1.0, w, out=w, where=valid)     # w[p] = 1 / unc[p]**2

        # Elementwise accumulation: each pixel's total receives only the
        # weight at that same pixel from this exposure.
        sum_w += w
        # Add weighted flux only at valid pixels; invalid values (e.g. NaN)
        # are not multiplied into or added to the flux accumulator.
        sum_wflux[valid] += flux[valid] * w[valid]

    # Boolean array, shape S. True where at least one valid exposure
    # contributed positive weight at that particular pixel.
    ok = sum_w > 0

    # Initialize outputs to NaN so pixels with no valid exposures remain NaN.
    coadd_flux = np.full_like(sum_wflux, np.nan, dtype=float)

    # Weighted mean, elementwise:
    #   coadd_flux[p] = sum_wflux[p] / sum_w[p]
    #                 = sum_k(flux[k,p] / unc[k,p]**2)
    #                   / sum_k(1 / unc[k,p]**2)
    # ``where=ok`` avoids division by zero where no exposure contributed.
    np.divide(sum_wflux, sum_w, out=coadd_flux, where=ok)

    # Initialize uncertainty output to NaN for pixels without contributors.
    coadd_unc = np.full_like(sum_w, np.nan, dtype=float)

    # Standard uncertainty of the inverse-variance weighted mean, assuming
    # independent input errors:
    #   Var(coadd_flux[p]) = 1 / sum_k(1 / unc[k,p]**2)
    #   coadd_unc[p]       = sqrt(Var(coadd_flux[p]))
    np.sqrt(1.0 / sum_w, out=coadd_unc, where=ok)

    # Return wavelength grid, weighted-mean flux and its 1-sigma uncertainty.
    # All three have shape S for a non-empty window.
    return np.asarray(wave_ref), coadd_flux, coadd_unc, contributing_files

# --- multiprocessing worker must be top-level for pickling ---

def _worker_task(args):
    """Worker: coadd one window.

    args = (index, instrument, window_obs)
    returns (index, wave_ref, coadd_flux, coadd_unc, contributing_files)
    """
    i, instrument, window_obs = args
    wave_ref, coadd_flux, coadd_unc, contributing_files = coadd_one_window(window_obs, instrument=instrument)
    return i, wave_ref, coadd_flux, coadd_unc, contributing_files


# -----------------------------------------------------------------------------
# HDF5 writer helpers
# -----------------------------------------------------------------------------
# Store lists as numbered groups and retain the date/file provenance for each
# observation in each overlap window.

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
# Each window is coadded independently. Workers return NumPy arrays; only the
# main process writes HDF5 datasets, avoiding concurrent writes through h5py.

def save_coadds_hdf5_incremental(
    out_path: str,
    *,
    overlaps: Sequence[Tuple[JD, JD]] | None,
    neid_date_sets: DateSets,
    expres_date_sets: DateSets,
    source_dates_and_dirs: str | None = None,
    source_overlaps_csv: str | None = None,
    manifest_csv: str | None = None,
    workers: int = 1,
    chunksize: int = 1,
    mp_start_method: str = "spawn",
) -> str:
    """Compute coadds and write HDF5 incrementally."""

    import multiprocessing as mp

    manifest_rows = []
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
                    for i, wave_ref, coadd_flux, coadd_unc, contributing_files in pool.imap(_worker_task, tasks, chunksize=chunksize):
                        window_bounds = overlaps[i] if overlaps is not None else ("", "")
                        jd_by_file = {floc: jd for jd, floc in date_sets[i]}
                        for floc in contributing_files:
                            manifest_rows.append([inst_name, f"{i:04d}", *window_bounds, jd_by_file.get(floc, ""), floc])
                        gwls.create_dataset(f"{i:04d}", data=np.asarray(wave_ref), compression="gzip", shuffle=True)
                        gco.create_dataset(f"{i:04d}", data=np.asarray(coadd_flux), compression="gzip", shuffle=True)
                        gun.create_dataset(f"{i:04d}", data=np.asarray(coadd_unc), compression="gzip", shuffle=True)
            else:
                for i, instrument, window_obs in tasks:
                    wave_ref, coadd_flux, coadd_unc, contributing_files = coadd_one_window(window_obs, instrument=instrument)
                    window_bounds = overlaps[i] if overlaps is not None else ("", "")
                    jd_by_file = {floc: jd for jd, floc in window_obs}
                    for floc in contributing_files:
                        manifest_rows.append([inst_name, f"{i:04d}", *window_bounds, jd_by_file.get(floc, ""), floc])
                    gwls.create_dataset(f"{i:04d}", data=np.asarray(wave_ref), compression="gzip", shuffle=True)
                    gco.create_dataset(f"{i:04d}", data=np.asarray(coadd_flux), compression="gzip", shuffle=True)
                    gun.create_dataset(f"{i:04d}", data=np.asarray(coadd_unc), compression="gzip", shuffle=True)

    if manifest_csv is not None:
        with open(manifest_csv, "w", newline="", encoding="utf-8") as csvfile:
            out = writer(csvfile)
            out.writerow(
                ["instrument", "window_id", "start_jd", "end_jd", "input_jd", "input_file"]
            )
            out.writerows(manifest_rows)

    return out_path


# -----------------------------------------------------------------------------
# Run everything
# -----------------------------------------------------------------------------
# High-level entry point: load the observation index, group files by window,
# compute each instrument's coadds, and save the results.

def run_coadds_and_save(
    overlaps: Sequence[Tuple[JD, JD]],
    dates_and_dirs_pkl: str = "dates_and_dirs.pkl",
    out_h5: str = "coadds_outputs.h5",
    overlaps_csv: str | None = None,
    manifest_csv: str | None = None,
    workers: int = 1,
    chunksize: int = 1,
    mp_start_method: str = "spawn",
) -> str:
    """Load dates_and_dirs.pkl, build date_sets, compute coadds, and write HDF5."""

    with open(dates_and_dirs_pkl, "rb") as f:
        dates_and_dirs = pickle.load(f)

    neid_date_sets, expres_date_sets = build_date_sets(overlaps, dates_and_dirs)
    if manifest_csv is None:
        manifest_csv = str(__import__("pathlib").Path(out_h5).with_name(__import__("pathlib").Path(out_h5).stem + "_inputs.csv"))

    return save_coadds_hdf5_incremental(
        out_h5,
        overlaps=overlaps,
        neid_date_sets=neid_date_sets,
        expres_date_sets=expres_date_sets,
        source_dates_and_dirs=dates_and_dirs_pkl,
        source_overlaps_csv=overlaps_csv,
        manifest_csv=manifest_csv,
        workers=workers,
        chunksize=chunksize,
        mp_start_method=mp_start_method,
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
# Parse command-line options, read overlap windows, then call the high-level
# pipeline and print the output HDF5 path.

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
        "--manifest-csv",
        default=None,
        help="CSV provenance output (default: alongside --out as <stem>_inputs.csv)",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Process only the first five overlap windows and write outputs with a _test5 suffix.",
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

    out_h5 = args.out
    manifest_csv = args.manifest_csv
    if args.test_mode:
        # Limit both instruments to the same first five windows from the CSV.
        overlaps = overlaps[:5]
        print(f"Test mode: processing {len(overlaps)} overlap window(s).")

        # Keep test outputs separate from production outputs. Apply the suffix
        # to explicitly supplied paths too; the default manifest is derived
        # from the already-suffixed HDF5 path inside run_coadds_and_save().
        def add_test_suffix(path: str) -> str:
            from pathlib import Path

            p = Path(path)
            if p.stem.endswith("_test5"):
                return str(p)
            return str(p.with_name(f"{p.stem}_test5{p.suffix}"))

        out_h5 = add_test_suffix(out_h5)
        if manifest_csv is not None:
            manifest_csv = add_test_suffix(manifest_csv)

    out_path = run_coadds_and_save(
        overlaps,
        dates_and_dirs_pkl=args.dates_and_dirs,
        out_h5=out_h5,
        overlaps_csv=args.overlaps_csv,
        manifest_csv=manifest_csv,
        workers=args.workers,
        chunksize=args.chunksize,
        mp_start_method=args.mp_start_method,
    )

    print(out_path)
