"""Small, user-facing view of observed and forecast hourly rainfall."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


def save_forecast_view(observed_mm, prediction_labels, land_mask, grid, directory,
                       source_start, source_end, target_end) -> str:
    """Save two maps with a shared rainfall scale and explicit hourly windows.

    Forecast labels represent integer millimeters, with label 100 meaning
    100 mm or more. Missing observations remain distinct from dry land/sea.
    """
    local_times = []
    taipei = timezone(timedelta(hours=8))
    for value in (source_start, source_end, target_end):
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Forecast view times must be timezone-aware")
        local_times.append(value.astimezone(taipei))
    start, end, future = local_times
    if end - start != timedelta(hours=1) or future - end != timedelta(hours=1):
        raise ValueError("Forecast view requires consecutive one-hour intervals")
    grid.coordinates()
    observed = np.asarray(observed_mm, dtype=float)
    predicted = np.asarray(prediction_labels, dtype=float)
    mask = np.asarray(land_mask, dtype=bool)
    shape = (grid.height, grid.width)
    if any(value.shape != shape for value in (observed, predicted, mask)) or not np.any(mask):
        raise ValueError("Rainfall arrays and land mask must match a nonempty shared grid")
    if np.any(np.isfinite(observed[mask]) & (observed[mask] < 0)):
        raise ValueError("Observed rainfall cannot be negative")
    labels = predicted[mask]
    if (not np.all(np.isfinite(labels)) or np.any((labels < 0) | (labels > 100))
            or np.any(labels != np.floor(labels))):
        raise ValueError("Forecast labels must be integers in 0..100 on land")

    # Import plotting only after configuring its project-local cache.
    from typhoon_rainfall_realtime.pipeline import configure_plotting
    configure_plotting()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from typhoon_rainfall.visualization.plotting import Plotting

    plotter = Plotting()
    sea_color = "#dcecf3"
    missing_color = "#777777"
    sea_rgba = matplotlib.colors.to_rgba(sea_color)
    missing_rgba = matplotlib.colors.to_rgba(missing_color)

    def colors(values):
        rgba = plotter.precip_colormap(plotter.norm(np.nan_to_num(values)))
        rgba[~mask] = sea_rgba
        rgba[mask & ~np.isfinite(values)] = missing_rgba
        return rgba

    def window(left, right):
        # Include both dates so the label remains clear across midnight/year.
        return left.strftime("%Y-%m-%d %H:%M") + " to " + right.strftime("%Y-%m-%d %H:%M")

    figure, axes = plt.subplots(1, 2, figsize=(11, 6), constrained_layout=True)
    try:
        extent = (grid.west, grid.east, grid.south, grid.north)
        titles = ("Observed | " + window(start, end), "Forecast | " + window(end, future))
        for axis, values, title in zip(axes, (observed, predicted), titles):
            axis.imshow(colors(values), extent=extent, origin="upper", interpolation="nearest")
            axis.set(title=title, xlabel="Longitude", ylabel="Latitude")
            axis.set_aspect(1 / np.cos(np.deg2rad((grid.south + grid.north) / 2)))
            axis.tick_params(labelsize=8)
            axis.title.set_fontsize(9)
        scalar = plt.cm.ScalarMappable(norm=plotter.norm, cmap=plotter.precip_colormap)
        colorbar = figure.colorbar(scalar, ax=list(axes), fraction=0.035, pad=0.025,
                                  ticks=[0, 1, 6, 15, 30, 50, 90, 130, 200, 500])
        colorbar.set_label("Hourly accumulation (mm)")
        legend = [Patch(facecolor=sea_color, label="Sea / outside mainland")]
        if np.any(mask & ~np.isfinite(observed)):
            legend.append(Patch(facecolor=missing_color, label="Missing observation"))
        axes[0].legend(handles=legend, loc="lower left", fontsize=7, framealpha=0.95)
        figure.suptitle("Next-hour rainfall | Taiwan time (UTC+08:00)\n"
                       "Forecast display: 100 mm denotes 100 mm or more", fontsize=11)
        path = Path(directory) / "forecast.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=150)
        return path.name
    finally:
        plt.close(figure)
