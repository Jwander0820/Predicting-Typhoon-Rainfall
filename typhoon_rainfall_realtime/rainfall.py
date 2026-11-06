"""Optional, observation-only rain-gauge snapshots for the realtime demo.

The hourly table is an official website product, not a versioned JSON API.
Keep its raw snapshot so a later RA preprocessing script can be audited.
This module does not interpolate rain gauges or feed rainfall into the model.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse


HOURLY_URL = "https://www.cwa.gov.tw/V8/C/P/Rainfall/MOD/22.html"
STATIONS_URL = "https://cwaopendata.s3.ap-northeast-1.amazonaws.com/Observation/O-A0002-001.json"
RAINFALL_IMAGE_URL = "https://www.cwa.gov.tw/Data/rainfall/QZT.jpg"
TAIPEI = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class HourWindow:
    start: datetime
    end: datetime


@dataclass(frozen=True)
class HourlyStation:
    station_id: str
    name: str
    location: str
    values: Tuple[Optional[float], ...]
    raw_values: Tuple[str, ...]


@dataclass(frozen=True)
class HourlyTable:
    windows: Tuple[HourWindow, ...]
    stations: Tuple[HourlyStation, ...]
    updated_at: datetime


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.active = False
        self.heading = False
        self.headers = []
        self.rows = []
        self.row = None
        self.parts = None
        self.links = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "table" and attributes.get("id") == "sortTable":
            self.active = True
        if not self.active:
            return
        if tag == "thead":
            self.heading = True
        elif tag == "tr":
            self.row = []
        elif tag in ("th", "td"):
            self.parts, self.links = [], []
        elif tag == "br" and self.parts is not None:
            self.parts.append("\n")
        elif tag == "a" and self.parts is not None:
            self.links.append(attributes.get("href", ""))

    def handle_data(self, data):
        if self.active and self.parts is not None:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if not self.active:
            return
        if tag in ("td", "th") and self.parts is not None:
            self.row.append(("".join(self.parts).strip(), tuple(self.links)))
            self.parts = None
        elif tag == "tr" and self.row is not None:
            (self.headers if self.heading else self.rows).append(self.row)
            self.row = None
        elif tag == "thead":
            self.heading = False
        elif tag == "table":
            self.active = False


def _source_date(month: int, day: int, reference: datetime) -> datetime:
    """Resolve the source's omitted year from source metadata, including Jan 1."""
    if reference.tzinfo is None or reference.utcoffset() is None:
        raise ValueError("Rainfall source reference must include a timezone")
    local = reference.astimezone(TAIPEI)
    candidates = []
    for year in (local.year - 1, local.year, local.year + 1):
        try:
            candidates.append(datetime(year, month, day, tzinfo=TAIPEI))
        except ValueError:
            continue
    if not candidates:
        raise ValueError("Invalid rainfall source calendar date")
    return min(candidates, key=lambda date: abs((date - local).total_seconds()))


def _rain_value(value: str) -> Optional[float]:
    if value == "-":
        return 0.0
    if value.upper() in ("X", "", "T", "NA", "N/A"):
        return None
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError("Unrecognized hourly rainfall value: {!r}".format(value)) from exc
    return number if math.isfinite(number) and number >= 0 else None


def parse_hourly_table(html: str, reference: datetime) -> HourlyTable:
    """Read actual dated headers; never reconstruct hours from the host clock."""
    parser = _TableParser()
    parser.feed(html)
    hour_pattern = re.compile(r"(?:(\d{1,2})/(\d{1,2})\s*)?(\d{1,2})\s*↓\s*(\d{1,2})")
    header = next((row for row in parser.headers if any("↓" in cell[0] for cell in row)), None)
    if header is None:
        raise ValueError("Official rainfall table has no hourly headers")
    windows, columns = [], []
    for column, (text, _) in enumerate(header):
        match = hour_pattern.fullmatch(text.strip())
        if match is None:
            continue
        month, day, start_hour, end_hour = match.groups()
        start_hour, end_hour = int(start_hour), int(end_hour)
        if not 0 <= start_hour <= 23 or not 0 <= end_hour <= 23 or (end_hour - start_hour) % 24 != 1:
            raise ValueError("Official rainfall header is not a one-hour interval")
        if not windows:
            if month is None:
                raise ValueError("First rainfall interval has no source date")
            start = _source_date(int(month), int(day), reference).replace(hour=start_hour)
            end = start + timedelta(hours=1)
        else:
            end = windows[-1].start
            start = end - timedelta(hours=1)
            if start.hour != start_hour or end.hour != end_hour:
                raise ValueError("Official rainfall intervals are not contiguous")
            if month is not None and (start.month, start.day) != (int(month), int(day)):
                raise ValueError("Official rainfall header date disagrees with its hours")
        windows.append(HourWindow(start, end))
        columns.append(column)
    if not 1 <= len(windows) <= 24:
        raise ValueError("Unexpected number of hourly rainfall columns")
    stations = []
    for row in parser.rows:
        if not row:
            continue
        if len(row) <= max(columns):
            raise ValueError("Official rainfall row does not match the hourly headers")
        ids = {parse_qs(urlparse(link).query).get("ID", [""])[0] for link in row[0][1]}
        ids.discard("")
        if len(ids) != 1:
            raise ValueError("Official rainfall row has no unambiguous station ID")
        raw = tuple(" ".join(row[index][0].split()) for index in columns)
        stations.append(HourlyStation(ids.pop(), " ".join(row[0][0].split()), " ".join(row[1][0].split()), tuple(_rain_value(value) for value in raw), raw))
    if not stations or len({station.station_id for station in stations}) != len(stations):
        raise ValueError("Official rainfall table has missing or duplicate stations")
    update = re.search(r"\bUpdateTime\s*=\s*['\"](\d{1,2})/(\d{1,2})\s+(\d{2}):(\d{2}):(\d{2})['\"]", html)
    if update is None:
        raise ValueError("Official rainfall table has no source update time")
    month, day, hour, minute, second = map(int, update.groups())
    updated = _source_date(month, day, reference).replace(hour=hour, minute=minute, second=second)
    if updated < windows[0].end:
        raise ValueError("Rainfall table update precedes its latest interval")
    return HourlyTable(tuple(windows), tuple(stations), updated)


def parse_station_metadata(payload: dict) -> Tuple[datetime, Dict[str, dict]]:
    """Use WGS84 only; do not silently substitute TWD67 coordinates."""
    root = payload["cwaopendata"]
    sent = datetime.fromisoformat(root["sent"].replace("Z", "+00:00"))
    if sent.tzinfo is None or sent.utcoffset() is None:
        raise ValueError("Rainfall JSON sent time has no timezone")
    stations = {}
    for station in root["dataset"]["Station"]:
        station_id = station["StationId"].strip().upper()
        if not station_id or station_id in stations:
            raise ValueError("Rainfall station JSON contains an empty or duplicate ID")
        coordinates = station.get("GeoInfo", {}).get("Coordinates", [])
        wgs84 = [item for item in coordinates if item.get("CoordinateName", "").upper() == "WGS84"]
        latitude = longitude = None
        if len(wgs84) == 1:
            try:
                latitude = float(wgs84[0]["StationLatitude"])
                longitude = float(wgs84[0]["StationLongitude"])
                if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
                    latitude = longitude = None
            except (KeyError, TypeError, ValueError):
                latitude = longitude = None
        stations[station_id] = {"station_id": station_id, "latitude": latitude, "longitude": longitude, "observation_time": station.get("ObsTime", {}).get("DateTime")}
    if not stations:
        raise ValueError("Rainfall station JSON is empty")
    return sent, stations


def hourly_records(table: HourlyTable, metadata: Dict[str, dict], hours: int) -> List[dict]:
    if not 1 <= hours <= min(24, len(table.windows)):
        raise ValueError("hours must be between 1 and the available hourly columns (at most 24)")
    records = []
    for station in table.stations:
        station_id = station.station_id.upper()
        match = metadata.get(station_id)
        status = "exact" if match else "unmatched"
        if match is None:
            candidates = [item for key, item in metadata.items() if len(key) == len(station_id) + 1 and key.startswith(station_id)]
            if len(candidates) == 1:
                match, status = candidates[0], "normalized"
            elif len(candidates) > 1:
                status = "ambiguous"
        for index, window in enumerate(table.windows[:hours]):
            records.append({"station_id": station.station_id, "station_name": station.name, "location": station.location, "metadata_station_id": match["station_id"] if match else None, "coordinate_match": status, "latitude_wgs84": match["latitude"] if match else None, "longitude_wgs84": match["longitude"] if match else None, "interval_start": window.start.isoformat(), "interval_end": window.end.isoformat(), "rainfall_mm": station.values[index], "raw_value": station.raw_values[index]})
    return records


def collect_rainfall(
    client,
    output_dir: Path,
    hours: int = 6,
    now: Optional[datetime] = None,
    max_age_minutes: float = 90,
) -> dict:
    """Export observations separately; QZT is a reference image, never model RA."""
    if not isinstance(hours, int) or not 1 <= hours <= 24:
        raise ValueError("hours must be an integer between 1 and 24")
    if not math.isfinite(max_age_minutes) or max_age_minutes <= 0:
        raise ValueError("max_age_minutes must be positive and finite")
    acquired_at = now if now is not None else datetime.now(timezone.utc)
    if acquired_at.tzinfo is None or acquired_at.utcoffset() is None:
        raise ValueError("Rainfall acquisition time must include a timezone")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"raw_hourly_html": output_dir / "rainfall_hourly.html", "raw_stations_json": output_dir / "rainfall_stations.json", "hourly_json": output_dir / "rainfall_hourly.json", "hourly_csv": output_dir / "rainfall_hourly.csv"}
    html_bytes = client.get_bytes(HOURLY_URL)
    paths["raw_hourly_html"].write_bytes(html_bytes)
    station_bytes = client.get_bytes(STATIONS_URL)
    paths["raw_stations_json"].write_bytes(station_bytes)
    sent, metadata = parse_station_metadata(json.loads(station_bytes.decode("utf-8-sig")))
    table = parse_hourly_table(html_bytes.decode("utf-8-sig"), sent)
    for source_name, source_time in (("station metadata", sent), ("hourly table", table.updated_at)):
        age = (acquired_at - source_time).total_seconds() / 60
        if age < -5:
            raise ValueError("Rainfall {} time is in the future".format(source_name))
        if age > max_age_minutes:
            raise ValueError("Rainfall {} is stale".format(source_name))
    records = hourly_records(table, metadata, hours)
    paths["hourly_json"].write_text(json.dumps(records, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with paths["hourly_csv"].open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    warning = None
    try:
        image_bytes = client.get_bytes(RAINFALL_IMAGE_URL)
        if not image_bytes.startswith(b"\xff\xd8"):
            raise ValueError("Official rainfall image is not a JPEG")
        paths["reference_image"] = output_dir / "observed_hourly_rainfall.jpg"
        paths["reference_image"].write_bytes(image_bytes)
    except Exception as exc:
        warning = "Reference rain image unavailable ({})".format(type(exc).__name__)
        paths.pop("reference_image", None)
    station_rows = records[::hours]
    return {"status": "collected", "hours": hours, "station_count": len(table.stations), "metadata_station_count": len(metadata), "record_count": len(records), "missing_rainfall_count": sum(row["rainfall_mm"] is None for row in records), "unmatched_stations": sum(row["coordinate_match"] == "unmatched" for row in station_rows), "ambiguous_stations": sum(row["coordinate_match"] == "ambiguous" for row in station_rows), "stations_without_wgs84": sum(row["latitude_wgs84"] is None for row in station_rows), "table_updated_at": table.updated_at.isoformat(), "station_metadata_sent_at": sent.isoformat(), "latest_interval_end": table.windows[0].end.isoformat(), "sources": {"hourly": HOURLY_URL, "stations": STATIONS_URL, "reference_image": RAINFALL_IMAGE_URL}, "files": {key: str(path) for key, path in paths.items()}, "warning": warning, "note": "Hourly observations only; no spatial interpolation or model RA reconstruction. QZT image acquisition time does not establish its observation interval."}
