"""IDW using the thesis-confirmed N=5 neighbours and u=2 power defaults.

Great-circle (Haversine) distance is a choice of this implementation, not a
distance definition established by the thesis. The caller owns the grid's
geography, north/south order, land mask, label conversion and rendering. No
maximum distance is imposed unless the caller explicitly requests one.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np


EARTH_RADIUS_KM = 6371.0088
_BATCH_SIZE = 256
_COINCIDENT_KM = 1e-9  # Numerical tolerance for identical spherical coordinates.


def _positive_real(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(name + " must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(name + " must be a positive finite number")
    return result


def idw_grid(
    station_lon,
    station_lat,
    rainfall_mm,
    grid_lon,
    grid_lat,
    power=2.0,
    neighbors=5,
    max_distance_km=None,
) -> np.ndarray:
    """Interpolate valid rain gauges onto an existing, unchanged lon/lat grid.

    Station arrays must be one-dimensional and have equal lengths. Invalid
    station rows (non-finite values, negative rainfall or out-of-range lon/lat)
    are excluded; zero rainfall remains valid. With no valid station, raise
    ``ValueError``. Grid arrays must be finite, nonempty, equally shaped arrays
    in degrees, with longitude in [-180, 180] and latitude in [-90, 90].

    The thesis confirms N=5 and u=2, so these are the default neighbour count
    and power. Each cell uses at most ``neighbors`` nearest valid gauges. The
    default ``max_distance_km=None`` imposes no cutoff; an explicit positive
    radius restricts selection inclusively. No gauge within an explicit radius
    yields NaN. At a coincident station location, return the mean of *all* coincident
    valid gauges, even if there are more than ``neighbors``. The numerical
    coincidence tolerance is 1e-9 km. Other cells use inverse-distance weights.

    Results are float64 with the input grid's exact shape and ordering. Only
    256 grid points are processed at once, avoiding a full grid-by-station
    distance matrix. Haversine is the current implementation choice; the
    thesis does not establish this distance formula or a maximum radius.
    These defaults do not establish bit-for-bit recovery of the old RA script.
    """
    power = _positive_real(power, "power")
    radius = None if max_distance_km is None else _positive_real(max_distance_km, "max_distance_km")
    if isinstance(neighbors, bool) or not isinstance(neighbors, Integral) or neighbors <= 0:
        raise ValueError("neighbors must be a positive integer")

    lon = np.asarray(station_lon, dtype=np.float64)
    lat = np.asarray(station_lat, dtype=np.float64)
    rain = np.asarray(rainfall_mm, dtype=np.float64)
    if lon.ndim != 1 or lat.ndim != 1 or rain.ndim != 1 or lon.shape != lat.shape or lon.shape != rain.shape:
        raise ValueError("Station longitude, latitude and rainfall must be equally sized 1D arrays")
    valid = (
        np.isfinite(lon) & np.isfinite(lat) & np.isfinite(rain)
        & (lon >= -180) & (lon <= 180)
        & (lat >= -90) & (lat <= 90) & (rain >= 0)
    )
    if not np.any(valid):
        raise ValueError("No valid rain-gauge stations available for interpolation")
    lon, lat, rain = lon[valid], lat[valid], rain[valid]

    output_lon = np.asarray(grid_lon, dtype=np.float64)
    output_lat = np.asarray(grid_lat, dtype=np.float64)
    if output_lon.shape != output_lat.shape or output_lon.size == 0 or output_lon.ndim == 0:
        raise ValueError("Grid longitude and latitude must be nonempty arrays with identical shapes")
    if not (
        np.all(np.isfinite(output_lon)) and np.all(np.isfinite(output_lat))
        and np.all(np.abs(output_lon) <= 180) and np.all(np.abs(output_lat) <= 90)
    ):
        raise ValueError("Grid coordinates must be finite, valid longitude/latitude degrees")

    station_lon_rad = np.deg2rad(lon)[None, :]
    station_lat_rad = np.deg2rad(lat)[None, :]
    station_cos_lat = np.cos(station_lat_rad)
    points_lon = np.deg2rad(output_lon.ravel())
    points_lat = np.deg2rad(output_lat.ravel())
    result = np.full(points_lon.size, np.nan, dtype=np.float64)
    k = min(int(neighbors), len(rain))

    for first in range(0, points_lon.size, _BATCH_SIZE):
        last = min(first + _BATCH_SIZE, points_lon.size)
        point_lat = points_lat[first:last, None]
        delta_lat = station_lat_rad - point_lat
        delta_lon = station_lon_rad - points_lon[first:last, None]
        haversine = (
            np.sin(delta_lat / 2) ** 2
            + np.cos(point_lat) * station_cos_lat * np.sin(delta_lon / 2) ** 2
        )
        distances = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))
        coincident = distances <= _COINCIDENT_KM
        exact_rows = np.any(coincident, axis=1)
        for row in np.flatnonzero(exact_rows):
            values = rain[coincident[row]]
            # Divide before summing to avoid overflow of finite gauge values.
            result[first + row] = np.sum(values / len(values))

        rows = np.flatnonzero(~exact_rows)
        if rows.size == 0:
            continue
        remaining_distances = distances[rows]
        nearest = np.argpartition(remaining_distances, k - 1, axis=1)[:, :k]
        selected_distances = np.take_along_axis(remaining_distances, nearest, axis=1)
        covered = np.ones_like(selected_distances, dtype=bool) if radius is None else selected_distances <= radius
        # Scaling by the minimum distance is algebraically equivalent to
        # 1/d**power while preventing tiny distances or large powers overflowing.
        minimum_distance = np.min(selected_distances, axis=1, keepdims=True)
        weights = (minimum_distance / selected_distances) ** power
        weights[~covered] = 0
        totals = weights.sum(axis=1)
        valid_rows = totals > 0
        if np.any(valid_rows):
            weights = weights[valid_rows] / totals[valid_rows, None]
            interpolated = np.sum(weights * rain[nearest[valid_rows]], axis=1)
            result[first + rows[valid_rows]] = interpolated

    return result.reshape(output_lon.shape)
