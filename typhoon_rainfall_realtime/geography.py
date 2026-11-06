"""Reuse the original static longitude/latitude images without rescaling them."""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from typing import Optional
from zipfile import ZipFile

import numpy as np
from PIL import Image

from typhoon_rainfall_realtime.geometry import GridSpec


DEFAULT_SOURCE = Path(__file__).resolve().parents[1] / "database.zip"
ARCHIVE_MEMBERS = {
    "LON": "database/train_data_GI/lon.png",
    "LAT": "database/train_data_GI/lat.png",
}


def export_geography(directory: Path, source: Optional[Path] = None,
                     grid: Optional[GridSpec] = None):
    """Copy verified historical GI PNGs into a run's input directory.

    ``source`` is either the original ``database.zip`` or a directory containing
    ``lon.png`` and ``lat.png``. The default is the fixed project database ZIP;
    no model-folder search, resizing, normalization or coordinate re-encoding is
    performed. The core's existing modality merger applies division by 255.
    """
    source = Path(source if source is not None else DEFAULT_SOURCE).resolve()
    grid = grid or GridSpec()
    grid.coordinates()  # Validate the shared grid before creating any output.
    if (grid.width, grid.height) != (128, 128):
        raise ValueError("Archived GI images require the shared 128x128 model grid")
    content, locations = {}, {}
    if source.is_dir():
        for kind, filename in (("LON", "lon.png"), ("LAT", "lat.png")):
            path = source / filename
            content[kind] = path.read_bytes()
            locations[kind] = str(path)
    elif source.is_file():
        with ZipFile(source) as archive:
            for kind, member in ARCHIVE_MEMBERS.items():
                # Read these exact entries only; never extract the archive.
                if archive.getinfo(member).file_size > 1024 * 1024:
                    raise ValueError("Archived GI image is unexpectedly large: " + member)
                content[kind] = archive.read(member)
                locations[kind] = member
    else:
        raise FileNotFoundError("GI source not found: " + str(source))

    arrays = {}
    for kind in ("LON", "LAT"):
        with Image.open(BytesIO(content[kind])) as image:
            if image.format != "PNG" or image.mode != "L" or image.size != (128, 128):
                raise ValueError("GI " + kind + " must be a 128x128 grayscale PNG")
            arrays[kind] = np.array(image)
    lon, lat = arrays["LON"], arrays["LAT"]
    if (not np.array_equal(lon, np.broadcast_to(lon[:1, :], lon.shape))
            or not np.all(np.diff(lon[0].astype(np.int16)) > 0)):
        raise ValueError("GI longitude must be constant down columns and increase west to east")
    if (not np.array_equal(lat, np.broadcast_to(lat[:, :1], lat.shape))
            or not np.all(np.diff(lat[:, 0].astype(np.int16)) < 0)):
        raise ValueError("GI latitude must be constant across rows and decrease north to south")

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    paths, files = {}, {}
    for kind in ("LON", "LAT"):
        path = directory / (kind + ".png")
        path.write_bytes(content[kind])
        paths[kind] = path
        files[kind] = {
            "source_image": locations[kind], "output_file": path.name,
            "sha256": hashlib.sha256(content[kind]).hexdigest(),
            "mode": "L", "size": [128, 128],
            "minimum": int(arrays[kind].min()), "maximum": int(arrays[kind].max()),
        }
    return paths, {
        "source": str(source), "static": True, "grid": grid.to_dict(),
        "encoding": "Original rank-coordinate grayscale images, copied byte-for-byte",
        "orientation": "LON increases west to east; LAT decreases north to south",
        "normalization": "Unchanged core merge_modal_arrays divides GI grayscale values by 255.0",
        "files": files,
    }
