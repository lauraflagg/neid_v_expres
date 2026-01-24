
from astropy.time import Time, TimeDelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def merge_events_astropy(events, gap_minutes=15):
    """
    Merge events into continuous blocks using astropy Time and TimeDelta.
    
    events: list of dicts like:
        {"midpoint": Time(...), "duration": <seconds>}

    Event start = midpoint - duration/2
    Event end   = midpoint + duration/2
    """
    gap = TimeDelta(gap_minutes * 60, format="sec")

    intervals = []
    for e in events:
        mid = e["midpoint"]
        half = TimeDelta(e["duration"] / 2, format="sec")  # duration is now in SECONDS
        start = mid - half
        end = mid + half
        intervals.append((start, end))

    # Sort by start time
    intervals.sort(key=lambda x: x[0].unix)

    merged = []
    current_start, current_end = intervals[0]

    for start, end in intervals[1:]:
        if start <= current_end + gap:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end

    merged.append((current_start, current_end))
    return merged




def block_overlaps_astropy(blocksA, blocksB, min_overlap_minutes=120):
    """
    Find overlapping blocks of continuous coverage.
    """
    min_overlap = TimeDelta(min_overlap_minutes * 60, format="sec")

    overlaps = []
    i = j = 0

    while i < len(blocksA) and j < len(blocksB):
        A_start, A_end = blocksA[i]
        B_start, B_end = blocksB[j]
        #convert all times to astropy objects in MJD format
        A_start = Time(A_start, format="mjd")
        A_end = Time(A_end, format="mjd")
        B_start = Time(B_start, format="mjd")
        B_end = Time(B_end, format="mjd")

        overlap_start = max(A_start, B_start)
        overlap_end   = min(A_end, B_end)

        if overlap_start < overlap_end:
            if (overlap_end - overlap_start) >= min_overlap:
                overlaps.append((overlap_start, overlap_end))

        # Advance whichever block ends first
        if A_end < B_end:
            i += 1
        else:
            j += 1

    return overlaps


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from astropy.time import Time, TimeDelta

def plot_blocks(A_blocks, B_blocks, overlaps, months_per_plot=6):
    """
    Plot A/B/Overlap blocks, but create a separate figure for each N‑month window.
    Preserves original plot appearance.

    Parameters
    ----------
    A_blocks, B_blocks, overlaps : list of (Time, Time)
        Start/end pairs of astropy Time objects.
    months_per_plot : int
        Size of each plotting interval (default: 6 months).
    """

    # Helper: convert Time → matplotlib float
    def t2m(t):
        return mdates.date2num(t.to_datetime())

    # Figure out the global min/max time from all blocks
    all_times = []
    for arr in (A_blocks, B_blocks, overlaps):
        for start, end in arr:
            all_times.append(start)
            all_times.append(end)

    tmin = min(all_times)
    tmax = max(all_times)

    # 6 months ≈ 182.625 days
    days = months_per_plot * (365.25 / 12)
    dt = TimeDelta(days, format="jd")

    # Slide window from tmin → tmax in 6‑month increments
    window_start = tmin
    fig_idx = 1

    while window_start < tmax:
        window_end = window_start + dt

        # --- Start new figure for this window ---
        fig, ax = plt.subplots(figsize=(12, 4))

        # Select blocks that intersect the window
        def window_filter(blocks):
            out = []
            for start, end in blocks:
                if end > window_start and start < window_end:
                    # Clip block to window bounds
                    s = max(start, window_start)
                    e = min(end, window_end)
                    out.append((s, e))
            return out

        A_clip = window_filter(A_blocks)
        B_clip = window_filter(B_blocks)
        O_clip = window_filter(overlaps)

        # Plot all A/B/Overlap bars exactly as before
        # --- A blocks ---
        for i, (start, end) in enumerate(A_clip):
            ax.barh(
                y=1,
                width=t2m(end) - t2m(start),
                left=t2m(start),
                height=0.3,
                color="tab:blue",
                alpha=0.6,
                label="A blocks" if i == 0 else ""
            )

        # --- B blocks ---
        for i, (start, end) in enumerate(B_clip):
            ax.barh(
                y=0,
                width=t2m(end) - t2m(start),
                left=t2m(start),
                height=0.3,
                color="tab:orange",
                alpha=0.6,
                label="B blocks" if i == 0 else ""
            )

        # --- Overlaps ---
        for i, (start, end) in enumerate(O_clip):
            ax.barh(
                y=0.5,
                width=t2m(end) - t2m(start),
                left=t2m(start),
                height=0.3,
                color="tab:green",
                alpha=0.9,
                label="Overlap" if i == 0 else ""
            )

        # --- Formatting ---
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S"))
        plt.xticks(rotation=30)

        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(["List B", "Overlap", "List A"])

        ax.set_title(f"Event Coverage Blocks\n{window_start.iso[:10]} → {window_end.iso[:10]}")
        ax.set_xlabel("Time")

        ax.legend()
        plt.tight_layout()
        plt.show()

        # move window forward
        window_start = window_end
        fig_idx += 1
