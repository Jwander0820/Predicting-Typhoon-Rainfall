import csv
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from typhoon_rainfall_realtime.rainfall import (
    HOURLY_URL,
    RAINFALL_IMAGE_URL,
    STATIONS_URL,
    collect_rainfall,
    hourly_records,
    parse_hourly_table,
    parse_station_metadata,
)


# Excerpts use the live CWA table/schema observed on 2026-10-31. Keep only
# dated headers and a few stations; tests do not need a network connection.
HTML = """
<table id="sortTable" class="table table-striped version_5"><thead><tr>
<th scope="col">測站名稱</th><th>所在<br>鄉鎮</th>
<th>10/31<BR>00<BR>↓<BR>01</th><th><BR>23<BR>↓<BR>00</th>
<th><BR>22<BR>↓<BR>23</th><th>24<BR>小時</th>
</tr></thead><tbody id="Rainfall_MOD"><tr>
<th scope="row"><span><a href="/V8/C/P/Rainfall/Rainfall_PlotImg.html?ID=46692">臺北</a></span></th>
<td>臺北市<br>中正區</td><td><span>-</span></td><td><span>X</span></td>
<td><span>1.5</span></td><td>20.0</td></tr></tbody></table>
<script> var UpdateTime='10/31 01:36:02'; </script>
"""


def metadata_station(station_id="466920", name="臺北"):
    return {
        "StationId": station_id,
        "StationName": name,
        "ObsTime": {"DateTime": "2026-10-31T01:40:00+08:00"},
        "GeoInfo": {"Coordinates": [
            {"CoordinateName": "TWD67", "StationLatitude": "25.039", "StationLongitude": "121.506"},
            {"CoordinateName": "WGS84", "StationLatitude": "25.037658", "StationLongitude": "121.514853"},
        ]},
        "RainfallElement": {"Past1hr": {"Precipitation": "9.0"}},
    }


def station_payload(stations=None):
    return {"cwaopendata": {
        "sent": "2026-10-31T01:45:00+08:00",
        "dataset": {"Station": stations if stations is not None else [metadata_station()]},
    }}


class FakeClient:
    def __init__(self, no_image=False):
        self.no_image = no_image
        self.urls = []

    def get_bytes(self, url):
        self.urls.append(url)
        if url == HOURLY_URL:
            return HTML.encode("utf-8")
        if url == STATIONS_URL:
            return json.dumps(station_payload()).encode("utf-8")
        if url == RAINFALL_IMAGE_URL:
            if self.no_image:
                raise RuntimeError("image temporarily missing")
            return b"\xff\xd8\xff\xd9"
        raise AssertionError(url)


class RainfallTests(unittest.TestCase):
    def setUp(self):
        self.reference = datetime.fromisoformat("2026-10-31T01:45:00+08:00")

    def test_dated_hourly_headers_cross_midnight_and_keep_missing_values(self):
        table = parse_hourly_table(HTML, self.reference)
        self.assertEqual(table.windows[0].start.isoformat(), "2026-10-31T00:00:00+08:00")
        self.assertEqual(table.windows[1].start.isoformat(), "2026-10-30T23:00:00+08:00")
        self.assertEqual(table.windows[1].end.isoformat(), "2026-10-31T00:00:00+08:00")
        self.assertEqual(table.stations[0].values, (0.0, None, 1.5))
        self.assertEqual(table.stations[0].raw_values, ("-", "X", "1.5"))
        self.assertEqual(table.updated_at.isoformat(), "2026-10-31T01:36:02+08:00")

    def test_omitted_year_uses_source_metadata_across_new_year(self):
        html = HTML.replace("10/31", "12/31")
        table = parse_hourly_table(html, datetime.fromisoformat("2027-01-01T00:15:00+08:00"))
        self.assertEqual(table.windows[0].start.year, 2026)
        self.assertEqual(table.windows[0].start.month, 12)

    def test_metadata_selects_wgs84_and_does_not_substitute_twd67(self):
        sent, stations = parse_station_metadata(station_payload())
        self.assertEqual(sent, self.reference)
        self.assertEqual(stations["466920"]["latitude"], 25.037658)
        station = metadata_station()
        station["GeoInfo"]["Coordinates"].pop()
        _, without_wgs84 = parse_station_metadata(station_payload([station]))
        self.assertIsNone(without_wgs84["466920"]["latitude"])

    def test_unique_station_id_mapping_and_hourly_rather_than_latest_window(self):
        table = parse_hourly_table(HTML, self.reference)
        _, metadata = parse_station_metadata(station_payload())
        records = hourly_records(table, metadata, 3)
        self.assertEqual(records[0]["metadata_station_id"], "466920")
        self.assertEqual(records[0]["coordinate_match"], "normalized")
        self.assertEqual([row["rainfall_mm"] for row in records], [0.0, None, 1.5])
        self.assertEqual(records[2]["longitude_wgs84"], 121.514853)

    def test_ambiguous_or_unmatched_stations_are_retained_without_coordinates(self):
        table = parse_hourly_table(HTML, self.reference)
        _, metadata = parse_station_metadata(station_payload([metadata_station("466920"), metadata_station("466921")]))
        record = hourly_records(table, metadata, 1)[0]
        self.assertEqual(record["coordinate_match"], "ambiguous")
        self.assertIsNone(record["latitude_wgs84"])
        self.assertEqual(record["rainfall_mm"], 0.0)
        record = hourly_records(table, {}, 1)[0]
        self.assertEqual(record["coordinate_match"], "unmatched")
        self.assertIsNone(record["metadata_station_id"])

    def test_duplicate_station_metadata_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_station_metadata(station_payload([metadata_station(), metadata_station()]))

    def test_missing_date_or_changed_hour_order_fails_instead_of_guessing(self):
        with self.assertRaisesRegex(ValueError, "source date"):
            parse_hourly_table(HTML.replace("10/31<BR>", ""), self.reference)
        with self.assertRaisesRegex(ValueError, "not contiguous"):
            parse_hourly_table(HTML.replace("23<BR>↓<BR>00", "21<BR>↓<BR>22"), self.reference)

    def test_collect_exports_zero_missing_and_sources_without_model_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            result = collect_rainfall(FakeClient(), Path(directory), hours=3, now=self.reference)
            self.assertEqual(result["record_count"], 3)
            self.assertEqual(result["missing_rainfall_count"], 1)
            self.assertEqual(result["stations_without_wgs84"], 0)
            self.assertIsNone(result["warning"])
            records = json.loads(Path(result["files"]["hourly_json"]).read_text(encoding="utf-8"))
            self.assertEqual([row["rainfall_mm"] for row in records], [0.0, None, 1.5])
            with Path(result["files"]["hourly_csv"]).open(encoding="utf-8-sig", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(csv_rows[0]["rainfall_mm"], "0.0")
            self.assertEqual(csv_rows[1]["rainfall_mm"], "")
            self.assertEqual(Path(result["files"]["raw_hourly_html"]).read_bytes(), HTML.encode("utf-8"))

    def test_reference_image_failure_is_reported_without_discarding_gauges(self):
        with tempfile.TemporaryDirectory() as directory:
            result = collect_rainfall(FakeClient(no_image=True), Path(directory), hours=1, now=self.reference)
            self.assertIn("RuntimeError", result["warning"])
            self.assertNotIn("reference_image", result["files"])
            self.assertTrue(Path(result["files"]["hourly_csv"]).exists())

    def test_hour_limit_is_validated_before_network_access(self):
        client = FakeClient()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                collect_rainfall(client, Path(directory), hours=25)
        self.assertEqual(client.urls, [])

    def test_stale_table_and_future_metadata_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "hourly table is stale"):
                collect_rainfall(FakeClient(), Path(directory), hours=1,
                                 now=datetime.fromisoformat("2026-10-31T02:00:00+08:00"),
                                 max_age_minutes=20)
            self.assertTrue((Path(directory) / "rainfall_hourly.html").exists())
            self.assertTrue((Path(directory) / "rainfall_stations.json").exists())
            self.assertFalse((Path(directory) / "rainfall_hourly.csv").exists())
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "metadata time is in the future"):
                collect_rainfall(FakeClient(), Path(directory), hours=1,
                                 now=datetime.fromisoformat("2026-10-31T01:30:00+08:00"))

    def test_schema_failure_keeps_raw_evidence_before_parsing(self):
        client = FakeClient()
        original_get_bytes = client.get_bytes
        changed_html = b"<html>official table format changed</html>"
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(client, "get_bytes", side_effect=lambda url: changed_html if url == HOURLY_URL else original_get_bytes(url)):
                with self.assertRaisesRegex(ValueError, "no hourly headers"):
                    collect_rainfall(client, Path(directory), hours=1, now=self.reference)
            self.assertEqual((Path(directory) / "rainfall_hourly.html").read_bytes(), changed_html)
            self.assertTrue((Path(directory) / "rainfall_stations.json").exists())

    def test_naive_acquisition_time_is_rejected_before_network_access(self):
        client = FakeClient()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "timezone"):
                collect_rainfall(client, Path(directory), now=datetime(2026, 10, 31))
        self.assertEqual(client.urls, [])


if __name__ == "__main__":
    unittest.main()
