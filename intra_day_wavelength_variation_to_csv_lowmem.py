#!/usr/bin/env python3
"""intra_day_wavelength_variation_to_csv_lowmem.py

Low-memory intra-day wavelength solution variation for EXPRES or NEID.

What it does
------------
For each day folder under a top-level data directory:
  1) Select a reference wavelength solution for that day:
      - reference=first  : uses the first readable FITS in the day (ONE PASS, very low RAM)
      - reference=median : computes the per-pixel median across files using an on-disk memmap
                           (still low RAM; uses disk space ~ Nfiles * Npix * 4 bytes)
  2) For each file, compute velocity residuals relative to the day's reference:
         v = c * (lambda_i - lambda_ref) / lambda_ref
     then compute per-file v_RMS and v_maxabs.
  3) Summarize those per-file values into per-day distribution stats and write ONE row per day
     to a CSV.

Wavelength extraction
---------------------
EXPRES: hdul[1].data['bary_wavelength']
NEID  : hdul['SCIWAVE'].data

Examples
--------
python intra_day_wavelength_variation_to_csv_lowmem.py --instrument NEID --data-dir /path/to/neid --out-csv neid.csv --reference first
python intra_day_wavelength_variation_to_csv_lowmem.py --instrument NEID --data-dir /path/to/neid --out-csv neid.csv --reference median --tmp-dir /scratch

Notes
-----
- With reference=first, peak RAM is ~ a few arrays (tens of MB for 122x9200 float64).
- With reference=median, peak RAM is still modest (chunked), but requires temporary disk.
"""

import argparse
import csv
import fnmatch
import os
import tempfile
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from astropy.io import fits

C_MPS = 299_792_458.0  # m/s


# ----------------------------
# Directory + file discovery
# ----------------------------

def list_day_directories(data_dir: str, recursive: bool, pattern: str) -> List[str]:
    """Return directories considered "day folders"."""
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
    return os.path.basename(os.path.normpath(day_dir))


# ----------------------------
# Wavelength extraction
# ----------------------------

def extract_wavelength_solution(instrument: str, fits_path: str) -> np.ndarray:
    """Extract wavelength solution for EXPRES or NEID. Returns float64 array."""
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


# ----------------------------
# Stats helpers
# ----------------------------

def summarize(x: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "rms": float(np.sqrt(np.mean(x * x))),
        "maxabs": float(np.max(np.abs(x))),
    }


def velocity_from_waves(lambda_ref: np.ndarray, lambda_i: np.ndarray) -> np.ndarray:
    """Compute velocity residuals in m/s with masking."""
    if lambda_i.shape != lambda_ref.shape:
        raise ValueError(f"Shape mismatch: {lambda_i.shape} vs reference {lambda_ref.shape}")

    dl = lambda_i - lambda_ref
    mask = np.isfinite(dl) & np.isfinite(lambda_ref) & (lambda_ref != 0.0)
    v = C_MPS * (dl[mask] / lambda_ref[mask])
    if v.size == 0:
        raise ValueError("No valid elements after masking.")
    return v


# ----------------------------
# Low-memory reference builders
# ----------------------------

def build_reference_first(instrument: str, files: List[str]) -> Tuple[np.ndarray, str, List[Tuple[str, str]]]:
    """Pick the first readable file as reference."""
    bad: List[Tuple[str, str]] = []
    for f in files:
        try:
            ref = extract_wavelength_solution(instrument, f)
            return ref, os.path.basename(f), bad
        except Exception as e:
            bad.append((f, str(e)))
    raise RuntimeError("No readable files to build reference.")


def build_reference_median_memmap(
    instrument: str,
    files: List[str],
    tmp_dir: Optional[str],
    chunk_size: int,
    dtype: np.dtype = np.float32,
) -> Tuple[np.ndarray, str, List[str], List[Tuple[str, str]]]:
    """Build a per-pixel median reference using an on-disk memmap.

    Returns:
      ref_wave (float64), ref_desc, good_files, bad_files

    Disk usage ~ n_files * n_pix * dtype.itemsize.
    """
    bad: List[Tuple[str, str]] = []
    good_files: List[str] = []

    # Find first readable to get shape
    first_wave = None
    for f in files:
        try:
            first_wave = extract_wavelength_solution(instrument, f)
            good_files.append(f)
            break
        except Exception as e:
            bad.append((f, str(e)))

    if first_wave is None:
        raise RuntimeError("No readable files to determine shape for median reference.")

    shape = first_wave.shape
    n_pix = int(np.prod(shape))

    # We'll create a temporary memmap file
    if tmp_dir is not None:
        os.makedirs(tmp_dir, exist_ok=True)

    fd, mmap_path = tempfile.mkstemp(prefix="wave_stack_", suffix=".dat", dir=tmp_dir)
    os.close(fd)  # numpy will manage file

    try:
        # We don't know good file count yet; build list first by reading each file once and writing row-wise.
        # Since memmap needs known shape, we first identify good files and write into a growing list,
        # then create the memmap with final size and fill it.

        # Identify remaining good files (we already have first)
        for f in files[len(bad) + 1:]:  # after the first readable attempt (approx)
            try:
                w = extract_wavelength_solution(instrument, f)
                if w.shape != shape:
                    raise ValueError(f"Shape mismatch {w.shape} != {shape}")
                good_files.append(f)
            except Exception as e:
                bad.append((f, str(e)))

        n_files = len(good_files)
        if n_files < 2:
            raise RuntimeError("Need at least 2 readable files to compute median reference.")

        # Create memmap stack [n_files, n_pix]
        stack = np.memmap(mmap_path, dtype=dtype, mode="w+", shape=(n_files, n_pix))

        # Fill memmap: read each file once
        for i, f in enumerate(good_files):
            w = extract_wavelength_solution(instrument, f)
            stack[i, :] = np.asarray(w, dtype=dtype).reshape(-1)

        # Compute median per pixel in chunks to keep RAM low
        ref_flat = np.empty(n_pix, dtype=np.float64)
        for start in range(0, n_pix, chunk_size):
            end = min(start + chunk_size, n_pix)
            # This slice is a memmap view; nanmedian will load the chunk, but size is bounded
            block = stack[:, start:end]
            ref_flat[start:end] = np.nanmedian(block, axis=0).astype(np.float64)

        ref_wave = ref_flat.reshape(shape)
        return ref_wave, "median(wavelength solutions)", good_files, bad

    finally:
        # Clean up the temporary memmap file
        try:
            os.remove(mmap_path)
        except OSError:
            pass


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Low-memory intra-day wavelength variations -> per-day CSV")
    parser.add_argument("--instrument", required=True, choices=["EXPRES", "NEID"])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--pattern", default="*.fits*")
    parser.add_argument("--reference", default="first", choices=["first", "median"])
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--print", dest="do_print", action="store_true")

    # Median options
    parser.add_argument("--tmp-dir", default=None, help="Directory for temporary memmap files (median only)")
    parser.add_argument("--chunk-size", type=int, default=50_000, help="Pixels per chunk when computing median")
    parser.add_argument("--median-dtype", default="float32", choices=["float32", "float64"],
                        help="Datatype for on-disk stack when reference=median")

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

    dtype = np.float32 if args.median_dtype == "float32" else np.float64

    for ddir in day_dirs:
        label = day_label(ddir)
        files = list_fits_files(ddir, args.pattern, recursive=args.recursive)
        n_found = len(files)

        if args.max_files is not None:
            files = files[: args.max_files]

        if n_found < 2:
            rows.append({
                "day": label,
                "day_path": ddir,
                "instrument": args.instrument,
                "reference": args.reference,
                "pattern": args.pattern,
                "recursive": bool(args.recursive),
                "n_files_found": n_found,
                "n_files_used": 0,
                "n_files_failed": 0,
                "ref_desc": "",
                "v_rms_mean_mps": np.nan,
                "v_rms_std_mps": np.nan,
                "v_rms_rms_mps": np.nan,
                "v_rms_maxabs_mps": np.nan,
                "v_maxabs_mean_mps": np.nan,
                "v_maxabs_std_mps": np.nan,
                "v_maxabs_rms_mps": np.nan,
                "v_maxabs_maxabs_mps": np.nan,
            })
            if args.do_print:
                print(f"{label}: insufficient files found ({n_found})")
            continue

        bad_files: List[Tuple[str, str]] = []

        # Build reference
        if args.reference == "first":
            try:
                ref_wave, ref_desc, bad_files = build_reference_first(args.instrument, files)
                # Determine good files (all files, but exclude those that failed before reference found? We'll re-check on the fly)
                good_files = files  # We'll handle failures during stats pass.
            except Exception as e:
                # Can't build reference
                rows.append({
                    "day": label,
                    "day_path": ddir,
                    "instrument": args.instrument,
                    "reference": args.reference,
                    "pattern": args.pattern,
                    "recursive": bool(args.recursive),
                    "n_files_found": n_found,
                    "n_files_used": 0,
                    "n_files_failed": n_found,
                    "ref_desc": "",
                    "v_rms_mean_mps": np.nan,
                    "v_rms_std_mps": np.nan,
                    "v_rms_rms_mps": np.nan,
                    "v_rms_maxabs_mps": np.nan,
                    "v_maxabs_mean_mps": np.nan,
                    "v_maxabs_std_mps": np.nan,
                    "v_maxabs_rms_mps": np.nan,
                    "v_maxabs_maxabs_mps": np.nan,
                })
                if args.do_print:
                    print(f"{label}: failed to build reference ({e})")
                continue

        else:
            # median reference via memmap
            try:
                ref_wave, ref_desc, good_files, bad_files = build_reference_median_memmap(
                    args.instrument, files, tmp_dir=args.tmp_dir, chunk_size=args.chunk_size, dtype=dtype
                )
            except Exception as e:
                rows.append({
                    "day": label,
                    "day_path": ddir,
                    "instrument": args.instrument,
                    "reference": args.reference,
                    "pattern": args.pattern,
                    "recursive": bool(args.recursive),
                    "n_files_found": n_found,
                    "n_files_used": 0,
                    "n_files_failed": n_found,
                    "ref_desc": "",
                    "v_rms_mean_mps": np.nan,
                    "v_rms_std_mps": np.nan,
                    "v_rms_rms_mps": np.nan,
                    "v_rms_maxabs_mps": np.nan,
                    "v_maxabs_mean_mps": np.nan,
                    "v_maxabs_std_mps": np.nan,
                    "v_maxabs_rms_mps": np.nan,
                    "v_maxabs_maxabs_mps": np.nan,
                })
                if args.do_print:
                    print(f"{label}: failed to build median reference ({e})")
                continue

        # Stats pass (streaming: one file in memory at a time)
        per_file_v_rms: List[float] = []
        per_file_v_maxabs: List[float] = []
        n_used = 0
        n_failed = 0

        for f in good_files:
            try:
                w = extract_wavelength_solution(args.instrument, f)
                v = velocity_from_waves(ref_wave, w)
                v_stats = summarize(v)
                per_file_v_rms.append(v_stats["rms"])
                per_file_v_maxabs.append(v_stats["maxabs"])
                n_used += 1
            except Exception as e:
                n_failed += 1
                bad_files.append((f, str(e)))

        if n_used < 2:
            row = {
                "day": label,
                "day_path": ddir,
                "instrument": args.instrument,
                "reference": args.reference,
                "pattern": args.pattern,
                "recursive": bool(args.recursive),
                "n_files_found": n_found,
                "n_files_used": n_used,
                "n_files_failed": len(bad_files),
                "ref_desc": ref_desc,
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
                print(f"{label}: insufficient usable files after reading (used {n_used})")
            continue

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
            "n_files_used": n_used,
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
                f"{label}: N_used={n_used} | "
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
