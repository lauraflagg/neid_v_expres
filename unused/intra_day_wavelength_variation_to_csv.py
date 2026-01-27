#!/usr/bin/env python3
"""
Intra-day wavelength solution variation for EXPRES or NEID -> export per-day summary to CSV.

For each "day" folder under a top-level data directory:
  - Load wavelength solution from every FITS file in that folder
  - Compare each file to a chosen intra-day reference (default: first file)
  - Compute Δλ and convert to velocity: v = c * Δλ / λ_ref
  - Summarize per-file velocity stats into per-day summary
  - Write one row per day to a CSV

EXPRES wavelength:
  hdul[1].data['bary_wavelength']

NEID wavelength:
  hdul['SCIWAVE'].data

Usage:
  python intra_day_wavelength_variation_to_csv.py --instrument EXPRES --data-dir /path/to/expres --out-csv expres_summary.csv
  python intra_day_wavelength_variation_to_csv.py --instrument NEID   --data-dir /path/to/neid   --out-csv neid_summary.csv

Options:
  --pattern "*.fits*"                 Glob pattern for FITS files (default: *.fits*)
  --reference first|median            Reference choice within each day (default: first)
  --max-files N                       Limit number of files per day (for speed/testing)
  --recursive                         If day folders contain nested structure, search recursively
  --print                             Also print per-day summary lines to screen
"""

import argparse
import fnmatch
import os
import csv
from typing import Dict, List, Tuple, Any

import numpy as np
from astropy.io import fits

C_MPS = 299_792_458.0  # speed of light [m/s]


def list_day_directories(data_dir: str, recursive: bool, pattern: str) -> List[str]:
    """Return a list of directories considered "day folders"."""
    day_dirs: List[str] = []

    if not recursive:
        for name in sorted(os.listdir(data_dir)):
            path = os.path.join(data_dir, name)
            if os.path.isdir(path):
                files = [f for f in os.listdir(path) if fnmatch.fnmatch(f, pattern)]
                if files:
                    day_dirs.append(path)
        return day_dirs

    for root, dirs, files in os.walk(data_dir):
        if len(dirs) == 0:
            matches = [f for f in files if fnmatch.fnmatch(f, pattern)]
            if matches:
                day_dirs.append(root)

    day_dirs.sort()
    return day_dirs


def list_fits_files(day_dir: str, pattern: str, recursive: bool) -> List[str]:
    """List FITS files in a day directory (optionally recursively)."""
    fits_files: List[str] = []

    if not recursive:
        for f in os.listdir(day_dir):
            if fnmatch.fnmatch(f, pattern):
                fits_files.append(os.path.join(day_dir, f))
    else:
        for root, _, files in os.walk(day_dir):
            for f in files:
                if fnmatch.fnmatch(f, pattern):
                    fits_files.append(os.path.join(root, f))

    fits_files.sort()
    return fits_files


def day_label(day_dir: str) -> str:
    """Use folder name as day label."""
    return os.path.basename(os.path.normpath(day_dir))


def extract_wavelength_solution(instrument: str, fits_path: str) -> np.ndarray:
    """Extract wavelength solution array for EXPRES or NEID."""
    inst = instrument.upper()

    with fits.open(fits_path, memmap=True) as hdul:
        if inst == "EXPRES":
            data = hdul[1].data
            if data is None or "bary_wavelength" not in data.columns.names:
                raise KeyError(f"EXPRES: missing hdul[1].data['bary_wavelength'] in {fits_path}")
            wave = data["bary_wavelength"]

        elif inst == "NEID":
            if "SCIWAVE" not in hdul:
                raise KeyError(f"NEID: missing extension 'SCIWAVE' in {fits_path}")

            wave = hdul["SCIWAVE"].data

            # If SCIWAVE is a table, try to pull a wavelength column
            if hasattr(wave, "columns"):
                names_lower = [n.lower() for n in wave.columns.names]
                for candidate in ("wavelength", "wave", "sciwave"):
                    if candidate in names_lower:
                        colname = wave.columns.names[names_lower.index(candidate)]
                        wave = wave[colname]
                        break
                else:
                    if len(wave.columns.names) == 1:
                        wave = wave[wave.columns.names[0]]
                    else:
                        raise KeyError(
                            f"NEID: SCIWAVE table columns {wave.columns.names} unrecognized in {fits_path}"
                        )
        else:
            raise ValueError("instrument must be EXPRES or NEID")

    wave = np.asarray(wave, dtype=np.float64)

    if not np.isfinite(wave).any():
        raise ValueError(f"No finite wavelength values in {fits_path}")

    return wave


def summarize(x: np.ndarray) -> Dict[str, float]:
    """Summary stats: mean, std, rms, maxabs."""
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "rms": float(np.sqrt(np.mean(x * x))),
        "maxabs": float(np.max(np.abs(x))),
    }


def compute_velocity(lambda_ref: np.ndarray, lambda_i: np.ndarray) -> np.ndarray:
    """Compute velocity array v = c * (λ_i - λ_ref) / λ_ref, with masking applied."""
    if lambda_i.shape != lambda_ref.shape:
        raise ValueError(f"Shape mismatch: {lambda_i.shape} vs reference {lambda_ref.shape}")

    dl = lambda_i - lambda_ref
    mask = np.isfinite(dl) & np.isfinite(lambda_ref) & (lambda_ref != 0.0)

    v = C_MPS * (dl[mask] / lambda_ref[mask])
    if v.size == 0:
        raise ValueError("No valid elements after masking (finite & nonzero reference wavelengths).")
    return v


def median_wavelength_solution(waves: List[np.ndarray]) -> np.ndarray:
    """Element-wise median wavelength solution across a list of arrays."""
    stack = np.stack(waves, axis=0)
    return np.nanmedian(stack, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Intra-day wavelength solution variations -> per-day CSV summary"
    )
    parser.add_argument("--instrument", required=True, choices=["EXPRES", "NEID"])
    parser.add_argument("--data-dir", required=True, help="Top-level directory containing day folders")
    parser.add_argument("--out-csv", required=True, help="Output CSV filename (e.g., summary.csv)")
    parser.add_argument("--pattern", default="*.fits*", help="Glob for FITS files (default: *.fits*)")
    parser.add_argument(
        "--reference",
        default="first",
        choices=["first", "median"],
        help="Reference within each day: first file or median of all files (default: first)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on number of files per day (after sorting)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for day folders and/or files (leaf directories)",
    )
    parser.add_argument(
        "--print",
        dest="do_print",
        action="store_true",
        help="Also print a compact per-day summary line to screen",
    )
    args = parser.parse_args()

    day_dirs = list_day_directories(args.data_dir, recursive=args.recursive, pattern=args.pattern)
    if not day_dirs:
        raise SystemExit(f"No day folders found under {args.data_dir} with pattern {args.pattern}")

    fieldnames = [
        "day",
        "day_path",
        "instrument",
        "reference",
        "pattern",
        "recursive",
        "n_files_found",
        "n_files_used",
        "n_files_failed",
        "ref_desc",
        "v_rms_mean_mps",
        "v_rms_std_mps",
        "v_rms_rms_mps",
        "v_rms_maxabs_mps",
        "v_maxabs_mean_mps",
        "v_maxabs_std_mps",
        "v_maxabs_rms_mps",
        "v_maxabs_maxabs_mps",
    ]

    rows: List[Dict[str, Any]] = []

    for ddir in day_dirs:
        label = day_label(ddir)
        files = list_fits_files(ddir, args.pattern, recursive=args.recursive)
        n_found = len(files)

        if args.max_files is not None:
            files = files[: args.max_files]

        good_files: List[str] = []
        waves: List[np.ndarray] = []
        bad_files: List[Tuple[str, str]] = []

        for f in files:
            try:
                w = extract_wavelength_solution(args.instrument, f)
                good_files.append(f)
                waves.append(w)
            except Exception as e:
                bad_files.append((f, str(e)))

        if len(waves) < 2:
            row = {
                "day": label,
                "day_path": ddir,
                "instrument": args.instrument,
                "reference": args.reference,
                "pattern": args.pattern,
                "recursive": bool(args.recursive),
                "n_files_found": n_found,
                "n_files_used": len(waves),
                "n_files_failed": len(bad_files),
                "ref_desc": "",
                "v_rms_mean_mps": np.nan,
                "v_rms_std_mps": np.nan,
                "v_rms_rms_mps": np.nan,
                "v_rms_maxabs_mps": np.nan,
                "v_maxabs_mean_mps": np.nan,
                "v_maxabs_std_mps": np.nan,
                "v_maxabs_rms_mps": np.nan,
                "v_maxabs_maxabs_mps": np.nan,
            }
            rows.append(row)
            if args.do_print:
                print(f"{label}: insufficient files (used {len(waves)} of {n_found})")
            continue

        if args.reference == "first":
            ref_wave = waves[0]
            ref_desc = os.path.basename(good_files[0])
        else:
            ref_wave = median_wavelength_solution(waves)
            ref_desc = "median(wavelength solutions)"

        per_file_v_rms: List[float] = []
        per_file_v_maxabs: List[float] = []

        for w in waves:
            v = compute_velocity(ref_wave, w)
            v_stats = summarize(v)
            per_file_v_rms.append(v_stats["rms"])
            per_file_v_maxabs.append(v_stats["maxabs"])

        v_rms_arr = np.asarray(per_file_v_rms, dtype=float)
        v_max_arr = np.asarray(per_file_v_maxabs, dtype=float)

        v_rms_stats = summarize(v_rms_arr)
        v_max_stats = summarize(v_max_arr)

        row = {
            "day": label,
            "day_path": ddir,
            "instrument": args.instrument,
            "reference": args.reference,
            "pattern": args.pattern,
            "recursive": bool(args.recursive),
            "n_files_found": n_found,
            "n_files_used": len(waves),
            "n_files_failed": len(bad_files),
            "ref_desc": ref_desc,
            "v_rms_mean_mps": v_rms_stats["mean"],
            "v_rms_std_mps": v_rms_stats["std"],
            "v_rms_rms_mps": v_rms_stats["rms"],
            "v_rms_maxabs_mps": v_rms_stats["maxabs"],
            "v_maxabs_mean_mps": v_max_stats["mean"],
            "v_maxabs_std_mps": v_max_stats["std"],
            "v_maxabs_rms_mps": v_max_stats["rms"],
            "v_maxabs_maxabs_mps": v_max_stats["maxabs"],
        }
        rows.append(row)

        if args.do_print:
            print(
                f"{label}: N_used={len(waves)} | "
                f"v_RMS mean/std/max={v_rms_stats['mean']:.3f}/{v_rms_stats['std']:.3f}/{v_rms_stats['maxabs']:.3f} m/s | "
                f"v_max max={v_max_stats['maxabs']:.3f} m/s"
            )

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} day rows to: {args.out_csv}\n")


if __name__ == "__main__":
    main()
