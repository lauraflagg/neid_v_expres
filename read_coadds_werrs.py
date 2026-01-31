import argparse
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams.update({'errorbar.capsize': 3})
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams.update({'axes.labelsize': 16})

from astropy.time import Time
from astropy.table import Table
from astropy import constants

import h5py
from astropy.io import ascii
import pickle

from lmfit import Model


# Speed of light in km/s
c_kms = constants.c.value / 10**3


def gaussian_w_slope(x, amp, cen, wid, bas, slope=0):
    """1-d gaussian with baseline + slope."""
    return (amp / (np.sqrt(2 * np.pi) * wid)) * np.exp(-(x - cen) ** 2 / (2 * wid ** 2)) + bas + x * slope


def wl2vel(wl_arr, central_wl=None):
    """returns vel in km/s; inputs are wl_arr and central_wl"""
    if central_wl is None:
        central_wl = np.average(wl_arr)
    vel_arr = (wl_arr - central_wl) * c_kms / central_wl
    return vel_arr


gmod = Model(gaussian_w_slope)


def build_overlap_indices(data):
    """
    Find indices (keys) where there are data for both NEID and EXPRES coadds.
    Returns (neid_indices, expres_indices, overlap_mids).
    """
    neid_indices = []
    expres_indices = []
    overlap_mids = []

    # Note: original code enumerates NEID coadds keys and indexes into data['overlaps'][i]
    # We keep that behavior but add guards.
    neid_keys = list(data['NEID']['coadds'].keys())

    for i, key in enumerate(neid_keys):
        # check if NEID has non-empty data
        if data['NEID']['coadds'][key] is not None and len(data['NEID']['coadds'][key]) > 0:
            # check if EXPRES also has this key with non-empty data
            if key in data['EXPRES']['coadds'].keys():
                if data['EXPRES']['coadds'][key] is not None and len(data['EXPRES']['coadds'][key]) > 0:
                    neid_indices.append(key)
                    expres_indices.append(key)

                    # overlaps are stored by position i in the original script
                    overlap = data['overlaps'][i]
                    mid = (overlap[0] + overlap[1]) / 2
                    overlap_mids.append(Time(mid, format='jd').isot)

    # Consistency check
    if neid_indices == expres_indices and len(neid_indices) == len(overlap_mids):
        print('everything matches:', neid_indices)
    else:
        print('NEID and EXPRES indices do not match')

    return neid_indices, expres_indices, overlap_mids


def fit_lines(
    w, s, e, line_list,
    plot=False,
    wl_min=6000,
    wl_max=6030,
    fit_range=10.0,
    dist_from_edge=500,
    plotbad=False,
    detrend=True
):
    """
    Fits lines in line_list to the spectrum s with wavelengths w.

    Inputs:
      w: 2D array of wavelengths (orders x pixels)
      s: 2D array of spectrum (orders x pixels)
      e: 2D array of uncertainties (orders x pixels)
      line_list: astropy table with columns 'lambda' and 'depth' (and 'line_id' if you want it saved)
      fit_range: half-width of fit window in km/s (fit in [-fit_range, +fit_range])
    """
    line_fits = []

    for i, l in enumerate(line_list['lambda'].value):
        if not (wl_min < l < wl_max):
            continue

        # Find orders containing wavelength l and sufficiently far from edge
        orders_with_line = []
        for o in range(w.shape[0]):
            dw = (w[o][1] - w[o][0])
            if (l > w[o].min() + dist_from_edge * dw) and (l < w[o].max() - dist_from_edge * dw):
                orders_with_line.append(o)

        for o in orders_with_line:
            wl_order = w[o]
            s_order = s[o]
            e_order = e[o]  # FIX: was incorrectly set to s[o] in original script

            vel_order = wl2vel(wl_order, central_wl=l)

            mask = (
                (vel_order > (-fit_range)) &
                (vel_order < (fit_range)) &
                (~np.isnan(s_order)) &
                (~np.isnan(e_order))
            )

            # Skip if insufficient data points
            if np.count_nonzero(mask) < 5:
                continue

            wl_fit = wl_order[mask]
            s_fit = s_order[mask]
            vel_fit = vel_order[mask]
            e_fit = e_order[mask]

            # avoid division by zero in weights
            safe_e = np.where(e_fit <= 0, np.nanmedian(e_fit[e_fit > 0]) if np.any(e_fit > 0) else 1.0, e_fit)
            weights_fit = 1.0 / (safe_e ** 2)

            # pixel index of line center
            line_center_index = int(np.argmin(np.abs(wl_order - l)))

            # Initial guesses
            amp_init = line_list['depth'][i] * np.sum(s_fit) * -1
            cen_init = 0.0
            wid_init = 3.0
            bas_init = float(np.median(s_fit))
            slope_init = 0.0

            params0 = gmod.make_params(
                amp=dict(value=amp_init, max=0),
                cen=dict(value=cen_init, min=-2, max=2),
                wid=dict(value=wid_init, min=0),
                bas=bas_init,
                slope=slope_init
            )

            try:
                result0 = gmod.fit(s_fit, params0, x=vel_fit, weights=weights_fit)

                if detrend:
                    baseline = (result0.params['bas'].value + vel_fit * result0.params['slope'].value)

                    # protect against zero / negative baseline
                    baseline = np.where(baseline == 0, 1.0, baseline)

                    s_fit_detrended = s_fit / baseline
                    e_fit_detrended = safe_e / np.abs(baseline)
                    weights_fit_detrended = 1.0 / (e_fit_detrended ** 2)

                    params1 = gmod.make_params(
                        amp=dict(value=amp_init, max=0),
                        cen=dict(value=cen_init, min=-2, max=2),
                        wid=dict(value=result0.params['wid'].value, min=0),
                        bas=0,
                        slope=0
                    )

                    # FIX: use detrended weights
                    result = gmod.fit(s_fit_detrended, params1, x=vel_fit, weights=weights_fit_detrended)
                else:
                    result = result0

            except Exception as exc:
                print(f"Fit failed for line at {l} Angstroms in order {o}: {exc}")
                continue

            # plotting handling whether detrend is on or off
            if plot or plotbad:
                s_plot = s_fit_detrended if detrend else s_fit
                if plot or (plotbad and (result.rsquared is None or result.rsquared < 0.8)):
                    plt.figure()
                    plt.plot(vel_fit, s_plot, label='data')
                    plt.plot(vel_fit, result.best_fit, label='fit')
                    plt.xlabel('Velocity (km/s)')
                    plt.ylabel('Flux')
                    plt.title(f'Order {o} Line at {l} Angstroms')
                    plt.legend()
                    plt.show()

            # Save the fit results
            line_fits.append({
                'line_wl': float(l),
                'order': int(o),
                'amp': float(result.params['amp'].value),
                'amp_err': None if result.params['amp'].stderr is None else float(result.params['amp'].stderr),
                'cen': float(result.params['cen'].value),
                'cen_err': None if result.params['cen'].stderr is None else float(result.params['cen'].stderr),
                'wid': float(result.params['wid'].value),
                'wid_err': None if result.params['wid'].stderr is None else float(result.params['wid'].stderr),
                'bas': float(result.params['bas'].value),
                'bas_err': None if result.params['bas'].stderr is None else float(result.params['bas'].stderr),
                'slope': float(result.params['slope'].value),
                'chi2': float(result.chisqr),
                'bic': float(result.bic),
                'r': None if result.rsquared is None else float(result.rsquared),
                'line_center_index': line_center_index,
                'line_depth': float(line_list['depth'][i]),
                'line_id': line_list['line_id'][i] if 'line_id' in line_list.colnames else None
            })

    # FIX: return after processing all lines, not after first append
    return line_fits


def get_fit_results(
    data, neid_indices, overlap_mids, line_list, fit_range,
    plot=False, plotbad=False, test=False, testnum=0, dist_from_edge=500
):
    """
    Loop through overlapping indices and fit lines to NEID and EXPRES coadds.
    """
    fit_results = []

    # Loop over (possibly sliced) indices
    sliced_indices = neid_indices[testnum:]
    for j, idx in enumerate(sliced_indices):
        w_NEID = np.array(data['NEID']['wls'][idx])
        s_NEID = np.array(data['NEID']['coadds'][idx])
        e_NEID = np.array(data['NEID']['uncs'][idx])

        w_EXPRES = np.array(data['EXPRES']['wls'][idx])
        s_EXPRES = np.array(data['EXPRES']['coadds'][idx])
        e_EXPRES = np.array(data['EXPRES']['uncs'][idx])

        upper = np.min((np.max(w_NEID), np.max(w_EXPRES)))
        lower = np.max((np.min(w_NEID), np.min(w_EXPRES)))

        # Find line fits in both spectra that are at least dist_from_edge pixels from edge in both instruments
        lines_to_use = {col: [] for col in ['lambda', 'depth', 'line_id'] if col in line_list.colnames}
        if 'lambda' not in lines_to_use:
            lines_to_use['lambda'] = []
        if 'depth' not in lines_to_use:
            lines_to_use['depth'] = []
        if 'line_id' not in lines_to_use:
            lines_to_use['line_id'] = []

        for row in line_list:
            l = row['lambda']
            if not (lower < l < upper):
                continue

            neid_ok = False
            expres_ok = False

            for o in range(w_NEID.shape[0]):
                dw = (w_NEID[o][1] - w_NEID[o][0])
                if (l > w_NEID[o].min() + dist_from_edge * dw) and (l < w_NEID[o].max() - dist_from_edge * dw):
                    neid_ok = True
                    break

            for o in range(w_EXPRES.shape[0]):
                dw = (w_EXPRES[o][1] - w_EXPRES[o][0])
                if (l > w_EXPRES[o].min() + dist_from_edge * dw) and (l < w_EXPRES[o].max() - dist_from_edge * dw):
                    expres_ok = True
                    break

            if neid_ok and expres_ok:
                lines_to_use['lambda'].append(row['lambda'])
                lines_to_use['depth'].append(row['depth'])
                lines_to_use['line_id'].append(row['line_id'] if 'line_id' in line_list.colnames else None)

        lines_to_use = Table(lines_to_use)

        print(f"Fitting {len(lines_to_use)} lines for index {idx} between {lower:.1f} and {upper:.1f} Angstroms")
        print("Now fitting for NEID...")
        line_fits_NEID = fit_lines(
            w_NEID, s_NEID, e_NEID, lines_to_use,
            plot=plot, wl_min=lower, wl_max=upper,
            fit_range=fit_range, dist_from_edge=dist_from_edge,
            plotbad=plotbad
        )

        if not test:
            print("Now fitting for EXPRES...")
            line_fits_EXPRES = fit_lines(
                w_EXPRES, s_EXPRES, e_EXPRES, lines_to_use,
                plot=plot, wl_min=lower, wl_max=upper,
                fit_range=fit_range, dist_from_edge=dist_from_edge,
                plotbad=plotbad
            )
            # FIX: correct overlap_mids indexing when using testnum slicing
            fit_results.append({
                'index': idx,
                'NEID_fits': line_fits_NEID,
                'EXPRES_fits': line_fits_EXPRES,
                'overlap_mid': overlap_mids[testnum + j] if (testnum + j) < len(overlap_mids) else None
            })
        else:
            fit_results.append(line_fits_NEID)

    return fit_results


def main():
    parser = argparse.ArgumentParser(
        description="Fit spectral lines in NEID/EXPRES coadds and save fit results."
    )
    parser.add_argument(
        "--fit-range",
        type=float,
        default=10.0,
        help="Half-width of velocity fitting window in km/s (default: 10.0)."
    )
    parser.add_argument(
        "--dist-from-edge",
        type=int,
        default=200,
        help="Minimum distance from order edge in pixels (default: 200)."
    )
    parser.add_argument(
        "--coadds-h5",
        type=str,
        default="coadds_outputs.h5",
        help="Path to HDF5 file containing coadds (default: coadds_outputs.h5)."
    )
    parser.add_argument(
        "--line-list",
        type=str,
        default="RvLineList_allowBlends=0_overlapCutoff=5.0e-6_blend_RV_cutoff=2.0_depthCutoff=0.02_"
                "rejectTelluricSlope=2000.0_badLineFilter=none_quant=95_nbin=1_DP=true_binParam=depth_n=1_"
                "long_output=true_VACUUM.csv",
        help="Path to CSV line list (default: your current VACUUM.csv filename)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output pickle filename. If not provided, one is generated using fit-range."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (only NEID fits per get_fit_results implementation)."
    )
    parser.add_argument(
        "--testnum",
        type=int,
        default=0,
        help="Starting index for test mode slicing (default: 0)."
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show plots for each fit."
    )
    parser.add_argument(
        "--plotbad",
        action="store_true",
        help="Plot only fits with poor r^2 (per fit_lines plotbad behavior)."
    )

    args = parser.parse_args()

    # Load data
    data = h5py.File(args.coadds_h5, 'r')

    # Find overlapping indices
    neid_indices, expres_indices, overlap_mids = build_overlap_indices(data)

    # Load line list
    line_list = ascii.read(args.line_list)

    # Run fits
    fit_results = get_fit_results(
        data=data,
        neid_indices=neid_indices,
        overlap_mids=overlap_mids,
        line_list=line_list,
        fit_range=args.fit_range,
        plot=args.plot,
        plotbad=args.plotbad,
        test=args.test,
        testnum=args.testnum,
        dist_from_edge=args.dist_from_edge
    )

    # Output filename
    outname = args.output
    if outname is None:
        outname = f"line_fit_results_neid_expres_overlap_2024_w_err_range{args.fit_range:g}.pkl"

    with open(outname, 'wb') as f:
        pickle.dump(fit_results, f)

    print(f"Saved results to: {outname}")


if __name__ == "__main__":
    main()
