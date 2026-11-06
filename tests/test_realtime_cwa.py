from datetime import datetime, timedelta, timezone
import unittest

import requests

from typhoon_rainfall_realtime.cwa import (
    CwaClient,
    CwaDataError,
    DataConsistencyError,
    StaleDataError,
)


BASE = "https://cwaopendata.s3.ap-northeast-1.amazonaws.com/Observation"
RD_META = f"{BASE}/O-A0058-003.xml"
IR_META = f"{BASE}/O-C0042-004.xml"
RD_IMAGE = f"{BASE}/O-A0058-003.png"
IR_IMAGE = f"{BASE}/O-C0042-004.jpg"
TZ = timezone(timedelta(hours=8))
NOW = datetime(2026, 10, 31, 9, 0, tzinfo=TZ)
PNG = b"\x89PNG\r\n\x1a\nimage-content"
JPG = b"\xff\xd8\xffimage-content\xff\xd9"

# Reduced copies of the real anonymous XML schemas inspected on 2026-10-31.
# RD and IR differ in both nesting and capitalization; these are deliberately
# literal external-format fixtures rather than a shared XML builder.
RD_XML = b"""<?xml version='1.0' encoding='UTF-8'?>
<cwaopendata xmlns="urn:cwa:gov:tw:cwacommon:0.1">
  <identifier>rd-0840</identifier><sent>2026-10-31T08:46:32+08:00</sent>
  <dataid>O-A0058-003</dataid>
  <dataset>
    <datasetInfo><parameterSet><ImageDimension>3600x3600</ImageDimension></parameterSet></datasetInfo>
    <resource><mimeType>image/png</mimeType>
      <ProductURL>https://cwaopendata.s3.ap-northeast-1.amazonaws.com/Observation/O-A0058-003.png</ProductURL>
    </resource>
    <DateTime>2026-10-31T08:40:00+08:00</DateTime>
  </dataset>
</cwaopendata>"""

IR_XML = b"""<?xml version='1.0' encoding='UTF-8'?>
<cwaopendata xmlns="urn:cwa:gov:tw:cwacommon:0.1">
  <identifier>ir-0830</identifier><sent>2026-10-31T08:46:33+08:00</sent>
  <dataid>O-C0042-004</dataid>
  <dataset>
    <GeoInfo><LongitudeRange>115.976888855-126.02300114</LongitudeRange></GeoInfo>
    <ObsTime><Datetime>2026-10-31T08:30:00+08:00</Datetime></ObsTime>
    <Resource>
      <ProductURL>https://cwaopendata.s3.ap-northeast-1.amazonaws.com/Observation/O-C0042-004.jpg</ProductURL>
    </Resource>
  </dataset>
</cwaopendata>"""


class FakeResponse:
    def __init__(self, body, status=200, headers=None, read_error=None):
        self.body = body
        self.status = status
        self.headers = headers or {}
        self.read_error = read_error
        self.closed = False

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(f"HTTP {self.status}")

    def iter_content(self, chunk_size):
        for offset in range(0, len(self.body), chunk_size):
            yield self.body[offset : offset + chunk_size]
        if self.read_error:
            raise self.read_error

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, responses):
        self.responses = {url: list(items) for url, items in responses.items()}
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if url not in self.responses or not self.responses[url]:
            raise AssertionError(f"Unexpected request: {url}")
        item = self.responses[url].pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def stable_session(rd_xml=RD_XML, ir_xml=IR_XML, rd_image=PNG):
    return FakeSession(
        {
            RD_META: [FakeResponse(rd_xml), FakeResponse(rd_xml)],
            IR_META: [FakeResponse(ir_xml), FakeResponse(ir_xml)],
            RD_IMAGE: [FakeResponse(rd_image)],
            IR_IMAGE: [FakeResponse(JPG)],
        }
    )


class CwaClientTests(unittest.TestCase):
    def test_real_rd_and_ir_schemas_preserve_observation_times(self):
        session = stable_session()
        result = CwaClient(session=session).get_latest_images(now=NOW)
        self.assertEqual(set(result), {"RD", "IR"})
        self.assertEqual(result["RD"].observed_at, datetime(2026, 10, 31, 8, 40, tzinfo=TZ))
        self.assertEqual(result["IR"].observed_at, datetime(2026, 10, 31, 8, 30, tzinfo=TZ))
        self.assertEqual(result["IR"].published_at, datetime(2026, 10, 31, 8, 46, 33, tzinfo=TZ))
        self.assertEqual(result["RD"].acquired_at, NOW)
        self.assertEqual(result["RD"].content, PNG)
        self.assertEqual(result["IR"].product_id, "O-C0042-004")
        self.assertEqual(len(session.calls), 6)
        for _, kwargs in session.calls:
            self.assertEqual(kwargs["timeout"], (5.0, 20.0))
            self.assertTrue(kwargs["stream"])
            self.assertNotIn("Authorization", kwargs["headers"])
            self.assertNotIn("Cookie", kwargs["headers"])

    def test_utc_source_times_are_normalized_to_taipei(self):
        rd = RD_XML.replace(b"2026-10-31T08:40:00+08:00", b"2026-10-31T00:40:00Z")
        result = CwaClient(session=stable_session(rd_xml=rd)).get_latest_images(now=NOW)
        self.assertEqual(result["RD"].observed_at.hour, 8)
        self.assertEqual(result["RD"].observed_at.utcoffset(), timedelta(hours=8))

    def test_missing_optional_publication_time_does_not_replace_observation_time(self):
        rd = RD_XML.replace(b"<sent>2026-10-31T08:46:32+08:00</sent>", b"")
        result = CwaClient(session=stable_session(rd_xml=rd)).get_latest_images(now=NOW)
        self.assertIsNone(result["RD"].published_at)
        self.assertEqual(result["RD"].observed_at.minute, 40)

    def test_rejects_old_data_before_downloading_images(self):
        session = stable_session()
        with self.assertRaises(StaleDataError):
            CwaClient(session=session).get_latest_images(now=NOW + timedelta(hours=2))
        self.assertTrue(all(url.endswith(".xml") for url, _ in session.calls))

    def test_rejects_future_observation(self):
        with self.assertRaisesRegex(CwaDataError, "observation time is in the future"):
            CwaClient(session=stable_session()).get_latest_images(
                now=datetime(2026, 10, 31, 8, 39, tzinfo=TZ)
            )

    def test_rejects_future_publication(self):
        with self.assertRaisesRegex(CwaDataError, "publication time is in the future"):
            CwaClient(session=stable_session()).get_latest_images(
                now=datetime(2026, 10, 31, 8, 45, tzinfo=TZ)
            )

    def test_rejects_publication_before_observation(self):
        rd = RD_XML.replace(b"08:46:32", b"08:39:32")
        with self.assertRaisesRegex(CwaDataError, "precedes"):
            CwaClient(session=stable_session(rd_xml=rd)).get_latest_images(now=NOW)

    def test_requires_dated_timezone_aware_observations(self):
        variants = [
            IR_XML.replace(b"<Datetime>2026-10-31T08:30:00+08:00</Datetime>", b""),
            IR_XML.replace(b"08:30:00+08:00", b"08:30:00"),
            IR_XML.replace(b"2026-10-31T08:30:00+08:00", b"not-a-date"),
        ]
        for xml in variants:
            with self.subTest(xml=xml):
                with self.assertRaises(CwaDataError):
                    CwaClient(session=stable_session(ir_xml=xml)).get_latest_images(now=NOW)

    def test_rejects_malformed_xml_and_changed_product(self):
        variants = [
            b"<not-closed>",
            RD_XML.replace(b"<dataid>O-A0058-003</dataid>", b"<dataid>O-A0058-001</dataid>"),
            RD_XML.replace(b"O-A0058-003.png", b"O-A0058-004.png"),
        ]
        for xml in variants:
            with self.subTest(xml=xml):
                with self.assertRaises(CwaDataError):
                    CwaClient(session=stable_session(rd_xml=xml)).get_latest_images(now=NOW)

    def test_rejects_excessive_rd_ir_time_difference(self):
        ir = IR_XML.replace(b"08:30:00", b"08:00:00")
        session = stable_session(ir_xml=ir)
        with self.assertRaises(DataConsistencyError):
            CwaClient(session=session).get_latest_images(now=NOW, max_skew_minutes=20)
        self.assertTrue(all(url.endswith(".xml") for url, _ in session.calls))

    def test_retries_entire_pair_if_metadata_changes_during_download(self):
        newer = RD_XML.replace(b"rd-0840", b"rd-0850").replace(b"08:40:00", b"08:50:00").replace(b"08:46:32", b"08:56:32")
        session = FakeSession(
            {
                RD_META: [FakeResponse(xml) for xml in (RD_XML, newer, newer, newer)],
                IR_META: [FakeResponse(IR_XML) for _ in range(4)],
                RD_IMAGE: [FakeResponse(PNG + b"old"), FakeResponse(PNG + b"new")],
                IR_IMAGE: [FakeResponse(JPG), FakeResponse(JPG)],
            }
        )
        result = CwaClient(session=session).get_latest_images(now=NOW)
        self.assertEqual(result["RD"].observed_at.minute, 50)
        self.assertEqual(result["RD"].content, PNG + b"new")
        self.assertEqual(len(session.calls), 12)

    def test_race_retries_are_bounded(self):
        newer = RD_XML.replace(b"rd-0840", b"rd-new")
        session = FakeSession(
            {
                RD_META: [FakeResponse(xml) for xml in (RD_XML, newer, RD_XML, newer)],
                IR_META: [FakeResponse(IR_XML) for _ in range(4)],
                RD_IMAGE: [FakeResponse(PNG), FakeResponse(PNG)],
                IR_IMAGE: [FakeResponse(JPG), FakeResponse(JPG)],
            }
        )
        with self.assertRaisesRegex(DataConsistencyError, "2 acquisition attempts"):
            CwaClient(session=session, max_attempts=2).get_latest_images(now=NOW)
        self.assertEqual(len(session.calls), 12)

    def test_rejects_html_returned_as_an_image(self):
        with self.assertRaisesRegex(CwaDataError, "expected image format"):
            CwaClient(session=stable_session(rd_image=b"<html>maintenance</html>")).get_latest_images(now=NOW)

    def test_download_checks_declared_and_streamed_size_and_closes_response(self):
        for headers in ({"Content-Length": "20"}, {}):
            with self.subTest(headers=headers):
                response = FakeResponse(b"1234567890", headers=headers)
                session = FakeSession({RD_IMAGE: [response]})
                with self.assertRaisesRegex(CwaDataError, "size limit"):
                    CwaClient(session=session, max_download_bytes=5).get_bytes(RD_IMAGE)
                self.assertTrue(response.closed)

    def test_download_rejects_empty_response(self):
        response = FakeResponse(b"")
        with self.assertRaisesRegex(CwaDataError, "empty"):
            CwaClient(session=FakeSession({RD_META: [response]})).get_bytes(RD_META)
        self.assertTrue(response.closed)

    def test_download_errors_are_source_errors_and_responses_are_closed(self):
        responses = [
            FakeResponse(b"not found", status=404),
            FakeResponse(b"partial", read_error=requests.ReadTimeout("slow source")),
        ]
        for response in responses:
            with self.subTest(response=response):
                with self.assertRaises(CwaDataError):
                    CwaClient(session=FakeSession({RD_META: [response]})).get_bytes(RD_META)
                self.assertTrue(response.closed)
        session = FakeSession({RD_META: [requests.ConnectTimeout("unreachable")]})
        with self.assertRaises(CwaDataError):
            CwaClient(session=session).get_bytes(RD_META)

    def test_rejects_naive_reference_time_without_network_calls(self):
        session = FakeSession({})
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            CwaClient(session=session).get_latest_images(now=datetime(2026, 10, 31, 9))
        self.assertFalse(session.calls)


if __name__ == "__main__":
    unittest.main()
