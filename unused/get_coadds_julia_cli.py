"""Compute coadded spectra and save outputs in a Julia-readable HDF5 file.

This is a CLI-friendly refactor of the original `get_coadds.py`.

What it produces
----------------
An HDF5 file with groups for NEID and EXPRES, each containing:
  - wls   : list of wavelength arrays (one per overlap window)
  - coadds: list of coadded flux arrays (one per overlap window)
  - uncs  : list of uncertainty arrays (one per overlap window)
  - date_sets: per-window lists of (JD, file_loc)

Julia can read the output using HDF5.jl.

Run
---
Typical (reuses overlaps from your original get_coadds.py in the same folder):
  python get_coadds_julia_cli.py --dates-and-dirs dates_and_dirs.pkl --out coadds_outputs.h5

Or provide overlaps externally:
  python get_coadds_julia_cli.py --overlaps-npy overlaps.npy --out coadds_outputs.h5
  python get_coadds_julia_cli.py --overlaps-pkl overlaps.pkl --out coadds_outputs.h5

"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pickle
import h5py
from astropy.io import fits
from astropy.time import Time


# -----------------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------------

JD = float
FileLoc = str
Obs = Tuple[JD, FileLoc]
DateSet = List[Obs]               # observations within one overlap window
DateSets = List[DateSet]          # list over overlap windows


@dataclass
class CoaddResult:
    """Container for one instrument's outputs."""

    wls: List[np.ndarray]         # one element per overlap window
    coadds: List[np.ndarray]      # one element per overlap window
    uncs: List[np.ndarray]        # one element per overlap window
    date_sets: DateSets           # matching date/file lists per window


# -----------------------------------------------------------------------------
# FITS readers (same logic as your script)
# -----------------------------------------------------------------------------

def read_expr_fits(hdul: fits.HDUList):
    flux = hdul[1].data["spectrum"]
    wave = hdul[1].data["bary_wavelength"]
    unc = hdul[1].data["uncertainty"]
    return flux, wave, unc


def read_neid_fits(hdul: fits.HDUList):
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
# Coaddition
# -----------------------------------------------------------------------------

def coadd_spectra(date_sets: DateSets, instrument: str):
    """Coadd spectra within each date_set using inverse-variance weighting."""

    all_waves: List[np.ndarray] = []
    all_fluxes: List[np.ndarray] = []
    all_uncerts: List[np.ndarray] = []

    for window_obs in date_sets:
        if len(window_obs) == 0:
            all_waves.append(np.array([]))
            all_fluxes.append(np.array([]))
            all_uncerts.append(np.array([]))
            continue

        flux_list = []
        unc_list = []
        wave_ref = None

        for jd, floc in window_obs:
            with fits.open(floc) as hdul:
                if instrument.upper() == "NEID":
                    wave, flux, unc = read_neid_fits(hdul)
                elif instrument.upper() == "EXPRES":
                    flux, wave, unc = read_expr_fits(hdul)
                else:
                    raise ValueError(f"Unknown instrument: {instrument}")

            if wave_ref is None:
                wave_ref = wave
            flux_list.append(flux)
            unc_list.append(unc)

        fluxes = np.asarray(flux_list)
        uncs = np.asarray(unc_list)

        weights = 1.0 / (uncs ** 2)
        weighted_flux = fluxes * weights
        coadd_flux = np.sum(weighted_flux, axis=0) / np.sum(weights, axis=0)
        coadd_unc = np.sqrt(1.0 / np.sum(weights, axis=0))

        all_waves.append(np.asarray(wave_ref))
        all_fluxes.append(np.asarray(coadd_flux))
        all_uncerts.append(np.asarray(coadd_unc))

    return all_waves, all_fluxes, all_uncerts


# -----------------------------------------------------------------------------
# HDF5 writer (Julia-friendly)
# -----------------------------------------------------------------------------

def _vlen_str_dtype():
    return h5py.string_dtype(encoding="utf-8")


def _write_list_of_arrays(g: h5py.Group, name: str, arrs: List[np.ndarray]):
    sub = g.create_group(name)
    sub.attrs["layout"] = "list"
    sub.attrs["length"] = len(arrs)
    for i, a in enumerate(arrs):
        a = np.asarray(a)
        sub.create_dataset(f"{i:04d}", data=a, compression="gzip", shuffle=True)


def _write_date_sets(g: h5py.Group, date_sets: DateSets):
    ds_group = g.create_group("date_sets")
    ds_group.attrs["layout"] = "list_of_lists"
    ds_group.attrs["length"] = len(date_sets)

    for i, window in enumerate(date_sets):
        wgrp = ds_group.create_group(f"{i:04d}")
        jds = np.array([obs[0] for obs in window], dtype=float)
        wgrp.create_dataset("jds", data=jds)
        flocs = np.array([obs[1] for obs in window], dtype=_vlen_str_dtype())
        wgrp.create_dataset("file_locs", data=flocs)


def save_coadds_hdf5(
    out_path: str,
    *,
    neid: CoaddResult,
    expres: CoaddResult,
    overlaps: Sequence[Tuple[JD, JD]] | None = None,
    source_dates_and_dirs: str | None = None,
) -> str:

    with h5py.File(out_path, "w") as f:
        f.attrs["created_utc"] = datetime.utcnow().isoformat() + "Z"
        f.attrs["format"] = "coadds_hdf5_v1"
        if source_dates_and_dirs is not None:
            f.attrs["dates_and_dirs_source"] = str(source_dates_and_dirs)

        if overlaps is not None:
            f.create_dataset("overlaps", data=np.asarray(overlaps, dtype=float))

        for inst_name, res in ("NEID", neid), ("EXPRES", expres):
            g = f.create_group(inst_name)
            _write_list_of_arrays(g, "wls", res.wls)
            _write_list_of_arrays(g, "coadds", res.coadds)
            _write_list_of_arrays(g, "uncs", res.uncs)
            _write_date_sets(g, res.date_sets)

    return out_path


# -----------------------------------------------------------------------------
# Convenience: run everything and save
# -----------------------------------------------------------------------------

def run_coadds_and_save(
    overlaps: Sequence[Tuple[JD, JD]],
    dates_and_dirs_pkl: str = "dates_and_dirs.pkl",
    out_h5: str = "coadds_outputs.h5",
) -> str:

    with open(dates_and_dirs_pkl, "rb") as f:
        dates_and_dirs = pickle.load(f)

    neid_date_sets, expres_date_sets = build_date_sets(overlaps, dates_and_dirs)

    neid_wls, neid_coadds, neid_uncs = coadd_spectra(neid_date_sets, instrument="NEID")
    expr_wls, expr_coadds, expr_uncs = coadd_spectra(expres_date_sets, instrument="EXPRES")

    neid_res = CoaddResult(neid_wls, neid_coadds, neid_uncs, neid_date_sets)
    expr_res = CoaddResult(expr_wls, expr_coadds, expr_uncs, expres_date_sets)

    return save_coadds_hdf5(
        out_h5,
        neid=neid_res,
        expres=expr_res,
        overlaps=overlaps,
        source_dates_and_dirs=dates_and_dirs_pkl,
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import importlib
    import sys

    parser = argparse.ArgumentParser(
        description=(
            "Compute NEID/EXPRES coadded spectra for overlap windows and save outputs "
            "(wavelengths, coadds, uncertainties, and date_sets) to a Julia-readable HDF5 file."
        )
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

    overlaps_src = parser.add_mutually_exclusive_group(required=False)
    overlaps_src.add_argument(
        "--overlaps-module",
        default="get_coadds",
        help=(
            "Python module to import that contains a global variable named 'overlaps' "
            "(default: %(default)s)."
        ),
    )
    overlaps_src.add_argument(
        "--overlaps-npy",
        help="Path to a .npy file containing an (N,2) float array of overlap windows.",
    )
    overlaps_src.add_argument(
        "--overlaps-pkl",
        help="Path to a pickle file containing a list of (start_jd, end_jd) tuples.",
    )

    args = parser.parse_args()

    # Load overlaps
    if args.overlaps_npy:
        overlaps_arr = np.load(args.overlaps_npy)
        overlaps = [tuple(map(float, row)) for row in np.asarray(overlaps_arr)]
    elif args.overlaps_pkl:
        with open(args.overlaps_pkl, "rb") as f:
            overlaps = pickle.load(f)
    else:
        try:
            mod = importlib.import_module(args.overlaps_module)
        except Exception as e:
            print(
                "ERROR: Could not import overlaps module. Provide --overlaps-npy or --overlaps-pkl, "
                "or ensure the module is on PYTHONPATH.\n"
                f"  module: {args.overlaps_module}\n  error: {e}",
                file=sys.stderr,
            )
            raise SystemExit(2)

        if not hasattr(mod, "overlaps"):
            print(
                "ERROR: overlaps module does not define a global variable named 'overlaps'.\n"
                f"  module: {args.overlaps_module}",
                file=sys.stderr,
            )
            raise SystemExit(2)

        overlaps = getattr(mod, "overlaps")

    out_path = run_coadds_and_save(
        overlaps,
        dates_and_dirs_pkl=args.dates_and_dirs,
        out_h5=args.out,
    )

    print(out_path)
