"""Observed RA: thesis-confirmed N=5/u=2, with explicitly recorded modern georeferencing."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

from typhoon_rainfall_realtime.interpolation import idw_grid
from typhoon_rainfall_realtime.pipeline import PROJECT_ROOT, configure_plotting
from typhoon_rainfall_realtime.geometry import GridSpec

BOUNDARY_URL = (
    "https://trms.tycg.gov.tw/server/rest/services/Hosted/COUNTY_MOI_1090820/"
    "FeatureServer/0/query?where=1%3D1&outFields=countycode%2Ccountyname"
    "&returnGeometry=true&outSR=4326&maxAllowableOffset=0.003&geometryPrecision=5&f=geojson"
)
BOUNDARY_ATTRIBUTION = "NLSC COUNTY_MOI_1090820, via Taoyuan City GIS; simplified to 0.003 degrees"


def load_boundary(client, boundary_file=None):
    """Cache the small official GeoJSON; retain its attribution and fingerprint."""
    if boundary_file is not None:
        path = Path(boundary_file).resolve()
        content = path.read_bytes()
        source = str(path)
    else:
        path = PROJECT_ROOT / ".cache" / "realtime-boundary" / "counties.geojson"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            content = client.get_bytes(BOUNDARY_URL)
            # Do not cache server errors or a truncated service response.
            boundary_polygons(json.loads(content.decode("utf-8-sig")))
            path.write_bytes(content)
        content = path.read_bytes()
        source = BOUNDARY_URL
    payload = json.loads(content.decode("utf-8-sig"))
    boundary_polygons(payload)
    return payload, {"source": source, "sha256": hashlib.sha256(content).hexdigest(),
                     "attribution": BOUNDARY_ATTRIBUTION if boundary_file is None else "User-supplied GeoJSON",
                     "license": "https://data.gov.tw/dataset/7442" if boundary_file is None else None}


def boundary_polygons(boundary):
    if boundary.get("type") != "FeatureCollection" or not boundary.get("features"):
        raise ValueError("RA boundary must be a nonempty GeoJSON FeatureCollection")
    if boundary.get("exceededTransferLimit"):
        raise ValueError("RA boundary service returned an incomplete feature collection")
    polygons = []
    for feature in boundary["features"]:
        geometry = feature.get("geometry") or {}
        kind, coordinates = geometry.get("type"), geometry.get("coordinates")
        if kind not in ("Polygon", "MultiPolygon"):
            raise ValueError("RA boundary contains a non-polygon geometry")
        for polygon in ([coordinates] if kind == "Polygon" else coordinates):
            rings = []
            for ring in polygon:
                array = np.asarray(ring, dtype=np.float64)
                if (array.ndim != 2 or array.shape[0] < 4 or array.shape[1] < 2
                        or not np.all(np.isfinite(array[:, :2]))):
                    raise ValueError("RA boundary has an invalid polygon ring")
                if np.any(np.abs(array[:, 0]) > 180) or np.any(np.abs(array[:, 1]) > 90):
                    raise ValueError("RA boundary must use WGS84 longitude/latitude")
                rings.append(array[:, :2])
            if not rings:
                raise ValueError("RA boundary has an empty polygon")
            polygons.append(rings)
    return polygons


def land_mask(longitude, latitude, boundary):
    configure_plotting()
    from matplotlib.path import Path as PolygonPath

    points = np.column_stack((longitude.ravel(), latitude.ravel()))
    result = np.zeros(len(points), dtype=bool)
    for polygon in boundary_polygons(boundary):
        outer = polygon[0]
        selected = np.flatnonzero((points[:, 0] >= outer[:, 0].min()) & (points[:, 0] <= outer[:, 0].max())
                                  & (points[:, 1] >= outer[:, 1].min()) & (points[:, 1] <= outer[:, 1].max()))
        if not len(selected):
            continue
        inside = PolygonPath(outer).contains_points(points[selected])
        for hole in polygon[1:]:
            inside &= ~PolygonPath(hole).contains_points(points[selected])
        result[selected] |= inside
    return result.reshape(longitude.shape)


def mainland_boundary(boundary):
    """Keep each mainland county's largest component, excluding outlying counties/islands."""
    features = []
    for feature in boundary["features"]:
        properties = {key.lower(): value for key, value in feature.get("properties", {}).items()}
        name = str(properties.get("countyname", "")).replace("台", "臺")
        if name in ("澎湖縣", "金門縣", "連江縣"):
            continue
        polygons = boundary_polygons({"type": "FeatureCollection", "features": [feature]})
        def area(polygon):
            ring = polygon[0]
            return abs(np.dot(ring[:-1, 0], ring[1:, 1]) - np.dot(ring[1:, 0], ring[:-1, 1]))
        rings = max(polygons, key=area)
        features.append({"type": "Feature", "properties": properties,
                         "geometry": {"type": "Polygon", "coordinates": [ring.tolist() for ring in rings]}})
    if not features:
        raise ValueError("No mainland polygons in supplied boundary")
    return {"type": "FeatureCollection", "features": features}


def encode_labels(rainfall):
    """Thesis EWB01: half-up rounded, capped at 100 (101 classes); 255=missing."""
    values = np.asarray(rainfall, dtype=float)
    valid = np.isfinite(values) & (values >= 0)
    labels = np.full(values.shape, 255, dtype=np.uint8)
    labels[valid] = np.floor(np.minimum(values[valid], 100.0) + 0.5).astype(np.uint8)
    return labels


def render_ra(rainfall, mask, boundary, grid, directory, start, end, power, neighbors):
    configure_plotting()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from typhoon_rainfall.visualization.plotting import Plotting

    plotter = Plotting()
    rgba = plotter.precip_colormap(plotter.norm(np.nan_to_num(rainfall)))
    rgba[~mask] = (0.86, 0.93, 0.97, 1)
    rgba[mask & ~np.isfinite(rainfall)] = (0.65, 0.65, 0.65, 1)
    Image.fromarray(np.rint(rgba * 255).astype(np.uint8), "RGBA").save(directory / "RA_color.png")
    figure, axis = plt.subplots(figsize=(7, 7))
    try:
        axis.imshow(rgba, extent=(grid.west, grid.east, grid.south, grid.north), origin="upper")
        for polygon in boundary_polygons(boundary):
            for ring in polygon:
                axis.plot(ring[:, 0], ring[:, 1], color="#444444", linewidth=0.35)
        axis.set(xlim=(grid.west, grid.east), ylim=(grid.south, grid.north),
                 xlabel="Longitude (WGS84)", ylabel="Latitude (WGS84)",
                 title=f"Observed hourly RA | IDW N={neighbors}, u={power:g}\n" + start.isoformat() + " to " + end.isoformat())
        axis.set_aspect(1 / np.cos(np.deg2rad((grid.south + grid.north) / 2)))
        scalar = plt.cm.ScalarMappable(norm=plotter.norm, cmap=plotter.precip_colormap)
        figure.colorbar(scalar, ax=axis, fraction=0.04, pad=0.03, label="Hourly accumulation (mm)")
        axis.legend(handles=[Patch(color=(0.65, 0.65, 0.65), label="No station coverage"),
                             Patch(color=(0.86, 0.93, 0.97), label="Sea / outside county boundary")],
                    loc="lower left", fontsize=7)
        figure.tight_layout()
        figure.savefig(directory / "RA_map.png", dpi=150)
    finally:
        plt.close(figure)


def generate_ra(observations_file, output_dir, boundary, boundary_info, *, grid=None,
                power=2.0, neighbors=5, max_distance_km=None):
    """Generate one independently georeferenced RA map per observed hour."""
    observations_file = Path(observations_file)
    source = observations_file.read_bytes()
    records = json.loads(source.decode("utf-8-sig"))
    if not isinstance(records, list) or not records:
        raise ValueError("No hourly station observations to interpolate")
    grid = grid or GridSpec()
    longitude, latitude = grid.coordinates()
    boundary = mainland_boundary(boundary)
    mask = land_mask(longitude, latitude, boundary)
    if not np.any(mask):
        raise ValueError("RA grid does not intersect the supplied land boundary")
    groups = {}
    for record in records:
        start = datetime.fromisoformat(record["interval_start"])
        end = datetime.fromisoformat(record["interval_end"])
        if (start.tzinfo is None or end.tzinfo is None or end - start != timedelta(hours=1)):
            raise ValueError("RA observations must have timezone-aware, separate one-hour intervals")
        groups.setdefault((start, end), []).append(record)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "grid.npz", longitude_wgs84=longitude, latitude_wgs84=latitude, land_mask=mask)
    summary = {"method": "IDW_with_thesis_confirmed_N5_u2_defaults", "power": power,
               "neighbors": neighbors, "max_distance_km": max_distance_km,
               "distance": "haversine_km", "grid": grid.to_dict(),
               "boundary": boundary_info, "observations_source": str(observations_file.resolve()),
               "observations_sha256": hashlib.sha256(source).hexdigest(),
               "label_encoding": "EWB01 floor(min(mm,100)+0.5); 101 classes; outside mainland=0; missing mainland=255", "thesis_grid_compatibility_verified": False,
               "thesis_source": "Final thesis section 4.5, printed p.29 / PDF p.42; EWB01 p.28 / PDF p.41",
               "georeferencing_note": "Shared RD/IR/RA target calibrated from archived training RD graticules; exact original RA rasterizer remains unverified.",
               "scope": "Taiwan mainland only; current valid mainland stations, not the historical fixed 556-station list",
               "note": "Observed hourly RA, not a forecast; not fed into the existing RD+IR weights.", "maps": []}
    for (start, end), rows in sorted(groups.items(), key=lambda item: item[0][1], reverse=True):
        station_lon = np.array([row.get("longitude_wgs84") for row in rows], dtype=float)
        station_lat = np.array([row.get("latitude_wgs84") for row in rows], dtype=float)
        values = np.array([row.get("rainfall_mm") for row in rows], dtype=float)
        eligible = land_mask(station_lon, station_lat, boundary) & np.isfinite(values) & (values >= 0)
        station_lon, station_lat, values = station_lon[eligible], station_lat[eligible], values[eligible]
        field = idw_grid(station_lon, station_lat, values, longitude, latitude,
                         power=power, neighbors=neighbors, max_distance_km=max_distance_km)
        field[~mask] = np.nan
        if not np.any(np.isfinite(field)):
            raise ValueError("No valid station coverage on RA land pixels for " + end.isoformat())
        # Include offset in the folder name so distinct timezone intervals never overwrite.
        name = "RA_" + end.strftime("%Y%m%d_%H%M%z")
        directory = output_dir / name
        directory.mkdir(exist_ok=False)
        np.savetxt(directory / "RA_mm.csv", field, delimiter=",", fmt="%.6f")
        labels = encode_labels(field)
        labels[~mask] = 0  # Thesis background convention, with a separate validity mask.
        Image.fromarray(labels, "L").save(directory / "RA_labels.png")
        Image.fromarray((np.isfinite(field) * 255).astype(np.uint8), "L").save(directory / "valid_mask.png")
        render_ra(field, mask, boundary, grid, directory, start, end, power, neighbors)
        item = {"interval_start": start.isoformat(), "interval_end": end.isoformat(), "station_rows": len(rows),
                "valid_mainland_stations": int(eligible.sum()),
                "valid_land_pixels": int(np.count_nonzero(np.isfinite(field))), "land_pixels": int(mask.sum()),
                "maximum_mm": float(np.nanmax(field)), "directory": str(directory.resolve())}
        (directory / "metadata.json").write_text(json.dumps(dict(item, method=summary["method"],
                   grid=summary["grid"], label_encoding=summary["label_encoding"]), ensure_ascii=False, indent=2), encoding="utf-8")
        summary["maps"].append(item)
    (output_dir / "ra_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return summary
