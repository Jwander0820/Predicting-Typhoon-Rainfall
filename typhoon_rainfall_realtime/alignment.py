"""Offline alignment preview against the archived thesis land footprint."""

import json
from pathlib import Path

import numpy as np

from typhoon_rainfall_realtime.geometry import GridSpec

ASSETS = Path(__file__).resolve().parent / "assets"


def training_land_mask():
    payload = json.loads((ASSETS / "training_land_mask.json").read_text(encoding="utf-8"))
    result = np.zeros((payload["height"], payload["width"]), dtype=bool)
    for y, intervals in enumerate(payload["row_intervals_start_inclusive_end_exclusive"]):
        for start, end in intervals:
            result[y, start:end] = True
    if int(result.sum()) != payload["land_pixels"]:
        raise ValueError("Archived training footprint is corrupt")
    return result


def save_alignment_preview(paths, directory):
    from typhoon_rainfall_realtime.pipeline import configure_plotting
    configure_plotting()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    with Image.open(paths["RD"]) as image:
        rd = np.array(image)
    with Image.open(paths["IR"]) as image:
        ir = np.array(image)
    mask = training_land_mask()
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.7))
    try:
        axes[0].imshow(rd)
        axes[1].imshow(ir, cmap="gray", vmin=0, vmax=255)
        axes[2].imshow(rd)
        axes[2].imshow(ir, cmap="gray", vmin=0, vmax=255, alpha=0.45)
        for axis, title in zip(axes, ("RD on common grid", "IR TM -> common grid", "RD / IR overlay")):
            axis.contour(mask.astype(float), levels=[0.5], colors=["#ff00ff"], linewidths=0.75)
            axis.set(xlim=(-0.5, 127.5), ylim=(127.5, -0.5), title=title, xlabel="Column (west to east)")
        axes[0].set_ylabel("Row (north to south)")
        figure.suptitle("Shared Taiwan mainland 128x128 grid\nMagenta: archived thesis land footprint (4700 pixels)")
        figure.tight_layout()
        figure.savefig(Path(directory) / "alignment.png", dpi=150)
    finally:
        plt.close(figure)
    return "alignment.png"
