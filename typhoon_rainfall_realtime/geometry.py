"""Shared north-up target grid and versioned CWA image georeferencing."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    """WGS84 outer edges; samples are pixel centers, with row zero at north."""

    west: float = 119.34666666666666
    south: float = 21.8936824807
    east: float = 122.76
    north: float = 25.3085641108
    width: int = 128
    height: int = 128

    def coordinates(self):
        if (not all(np.isfinite(v) for v in (self.west, self.south, self.east, self.north))
                or not -180 <= self.west < self.east <= 180
                or not -90 <= self.south < self.north <= 90
                or not isinstance(self.width, int) or not isinstance(self.height, int)
                or not 2 <= self.width <= 1024 or not 2 <= self.height <= 1024):
            raise ValueError("Invalid shared grid bounds or dimensions")
        longitude = self.west + (np.arange(self.width) + 0.5) * (self.east - self.west) / self.width
        latitude = self.north - (np.arange(self.height) + 0.5) * (self.north - self.south) / self.height
        return np.meshgrid(longitude, latitude)

    def to_dict(self):
        return dict(asdict(self), crs="EPSG:4326", row_order="north_to_south",
                    bbox_convention="outer_edges", sampling="pixel_centers")


@dataclass(frozen=True)
class SourceImageGrid:
    kind: str
    product_id: str
    width: int
    height: int
    crs: str
    # x=ax*projected_easting+bx, y=ay*projected_northing+by; integer indices are centers.
    affine: tuple
    pixel_roi: tuple
    basis: str

    def world_to_pixel(self, longitude, latitude):
        from pyproj import Transformer

        transformer = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        easting, northing = transformer.transform(longitude, latitude)
        ax, bx, ay, by = self.affine
        return ax * np.asarray(easting) + bx, ay * np.asarray(northing) + by

    def pixel_to_world(self, x, y):
        from pyproj import Transformer

        ax, bx, ay, by = self.affine
        transformer = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        return transformer.transform((np.asarray(x) - bx) / ax, (np.asarray(y) - by) / ay)


# IR official projection family/center: CWA 3-7-2.pdf. Pixel affine is calibrated
# against labeled graticules; the XML geographic envelope is not an affine image CRS.
IR_CRS = "+proj=tmerc +lat_0=23.7 +lon_0=121 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
SOURCE_GRIDS = {
    "RD": SourceImageGrid("RD", "O-A0058-003", 3600, 3600, "EPSG:4326",
                          (600.0, -70800.5, -600.0, 15899.5), (0, 0, 3600, 3600),
                          "CWA 118-124E,20.5-26.5N image envelope; outer-edge pixel convention"),
    "IR": SourceImageGrid("IR", "O-C0042-004", 800, 800, IR_CRS,
                          (0.0007823941450716034, 398.9671146696392,
                           -0.000783703003258268, 399.32570925509634), (4, 35, 795, 795),
                          "CWA Transverse Mercator center121E/23.7N; labeled-graticule pixel calibration v1"),
}


def reproject_image(image, source, target=None, supersample=8):
    """Inverse warp with subpixel integration, then AREA reduction to model size."""
    import cv2

    target = target or GridSpec()
    if image.shape[:2] != (source.height, source.width):
        raise ValueError("Unexpected " + source.kind + " image dimensions for calibrated georeferencing")
    if not isinstance(supersample, int) or not 1 <= supersample <= 8:
        raise ValueError("supersample must be an integer in 1..8")
    # Construct each final pixel's subcenters without changing its geographic footprint.
    fine_width, fine_height = target.width * supersample, target.height * supersample
    target.coordinates()  # validate final grid independently of the temporary fine resolution
    lon = target.west + (np.arange(fine_width) + 0.5) * (target.east - target.west) / fine_width
    lat = target.north - (np.arange(fine_height) + 0.5) * (target.north - target.south) / fine_height
    longitude, latitude = np.meshgrid(lon, lat)
    x, y = source.world_to_pixel(longitude, latitude)
    left, top, right, bottom = source.pixel_roi
    if (not np.all(np.isfinite(x)) or not np.all(np.isfinite(y))
            or x.min() < left - 0.5 or y.min() < top - 0.5
            or x.max() > right - 0.5 or y.max() > bottom - 0.5):
        raise ValueError("Target grid falls outside usable " + source.kind + " imagery; refusing border fill")
    # The outer half pixels belong to the edge cells, rather than an absent black border.
    # Clamp to the *usable ROI* centers as well, so bilinear taps cannot read annotations.
    x = np.clip(x, left, right - 1)
    y = np.clip(y, top, bottom - 1)
    warped = cv2.remap(image, x.astype(np.float32), y.astype(np.float32), interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
    result = cv2.resize(warped, (target.width, target.height), interpolation=cv2.INTER_AREA)
    return result, {"product_id": source.product_id, "source_size": [source.width, source.height],
                    "source_crs": source.crs, "source_affine": list(source.affine),
                    "source_pixel_roi_xyxy": list(source.pixel_roi), "georeferencing_basis": source.basis,
                    "sampled_pixel_range_xyxy": [float(x.min()), float(y.min()), float(x.max()), float(y.max())],
                    "resampling": "8x bilinear subpixel integration then OpenCV INTER_AREA" if supersample == 8
                                  else f"{supersample}x bilinear subpixel integration then OpenCV INTER_AREA"}
