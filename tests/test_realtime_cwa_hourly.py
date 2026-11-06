from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import Mock, call

from typhoon_rainfall_realtime.cwa import (
    CwaClient, CwaDataError, DataConsistencyError, StaleDataError,
)


TZ = timezone(timedelta(hours=8))
START = datetime(2026, 11, 6, 17, tzinfo=TZ)
END = START + timedelta(hours=1)
NOW = END + timedelta(minutes=23)
PNG = b"\x89PNG\r\n\x1a\nimage"
JPG = b"\xff\xd8\xffimage"
RD_URL = "https://www.cwa.gov.tw/Data/radar/CV1_TW_3600_202611061730.png"
IR_URL = "https://www.cwa.gov.tw/Data/satellite/TWI_IR1_Gray_800/TWI_IR1_Gray_800-2026-11-06-17-30.jpg"


def client_with_images():
    client = CwaClient(session=Mock())
    client.get_bytes = Mock(side_effect=[PNG, JPG])
    return client


class HourlyCwaClientTests(unittest.TestCase):
    def test_pairs_both_images_at_completed_rain_hours_midpoint(self):
        client = client_with_images()
        images = client.get_images_for_hour(START, END, now=NOW)
        self.assertEqual(client.get_bytes.call_args_list, [call(RD_URL), call(IR_URL)])
        for kind, product, content in (("RD", "O-A0058-003", PNG), ("IR", "O-C0042-004", JPG)):
            image = images[kind]
            self.assertEqual(image.kind, kind)
            self.assertEqual(image.product_id, product)
            self.assertEqual(image.observed_at, START + timedelta(minutes=30))
            self.assertEqual(image.acquired_at, NOW)
            self.assertIsNone(image.published_at)
            self.assertEqual(image.content, content)

    def test_taiwan_midnight_and_new_year_use_previous_days_midpoint(self):
        client = client_with_images()
        end = datetime(2027, 1, 1, tzinfo=TZ)
        start = end - timedelta(hours=1)
        images = client.get_images_for_hour(start, end, now=end + timedelta(minutes=10))
        self.assertEqual(images["RD"].image_url,
                         "https://www.cwa.gov.tw/Data/radar/CV1_TW_3600_202612312330.png")
        self.assertEqual(images["IR"].image_url,
                         "https://www.cwa.gov.tw/Data/satellite/TWI_IR1_Gray_800/TWI_IR1_Gray_800-2026-12-31-23-30.jpg")

    def test_utc_inputs_are_converted_before_constructing_local_filenames(self):
        client = client_with_images()
        images = client.get_images_for_hour(START.astimezone(timezone.utc),
                                            END.astimezone(timezone.utc),
                                            now=NOW.astimezone(timezone.utc))
        self.assertEqual(images["RD"].image_url, RD_URL)
        self.assertEqual(images["RD"].observed_at.utcoffset(), timedelta(hours=8))

    def test_requires_aware_times_before_network(self):
        variants = ((START.replace(tzinfo=None), END, NOW),
                    (START, END.replace(tzinfo=None), NOW),
                    (START, END, NOW.replace(tzinfo=None)))
        for start, end, now in variants:
            with self.subTest(start=start, end=end, now=now):
                client = client_with_images()
                with self.assertRaisesRegex(ValueError, "timezone-aware"):
                    client.get_images_for_hour(start, end, now=now)
                client.get_bytes.assert_not_called()

    def test_requires_exactly_one_whole_hour(self):
        variants = ((START, END + timedelta(hours=1)),
                    (START + timedelta(minutes=10), END + timedelta(minutes=10)),
                    (START + timedelta(microseconds=1), END + timedelta(microseconds=1)),
                    (END, START))
        for start, end in variants:
            with self.subTest(start=start, end=end):
                client = client_with_images()
                with self.assertRaisesRegex(ValueError, "whole hour"):
                    client.get_images_for_hour(start, end, now=NOW)
                client.get_bytes.assert_not_called()

    def test_rejects_incomplete_rain_interval(self):
        client = client_with_images()
        with self.assertRaisesRegex(DataConsistencyError, "not complete"):
            client.get_images_for_hour(START, END, now=END - timedelta(seconds=1))
        client.get_bytes.assert_not_called()

    def test_rejects_finished_target_interval_without_fetching(self):
        client = client_with_images()
        with self.assertRaisesRegex(StaleDataError, "target rain interval"):
            client.get_images_for_hour(START, END, now=END + timedelta(hours=1))
        client.get_bytes.assert_not_called()

    def test_explicit_replay_still_enforces_maximum_rain_age(self):
        client = client_with_images()
        images = client.get_images_for_hour(START, END, now=END + timedelta(minutes=70),
                                            require_future_target=False)
        self.assertEqual(images["RD"].observed_at, START + timedelta(minutes=30))
        stale = client_with_images()
        with self.assertRaisesRegex(StaleDataError, "too old"):
            stale.get_images_for_hour(START, END, now=END + timedelta(minutes=91),
                                      require_future_target=False)
        stale.get_bytes.assert_not_called()

    def test_invalid_maximum_age_is_rejected_before_network(self):
        for age in (0, -1, float("inf"), float("nan")):
            with self.subTest(age=age):
                client = client_with_images()
                with self.assertRaisesRegex(ValueError, "max_age_minutes"):
                    client.get_images_for_hour(START, END, now=NOW, max_age_minutes=age)
                client.get_bytes.assert_not_called()

    def test_dated_download_failure_never_falls_back_to_latest(self):
        client = client_with_images()
        client.get_bytes.side_effect = CwaDataError("HTTP 404")
        with self.assertRaises(CwaDataError):
            client.get_images_for_hour(START, END, now=NOW)
        self.assertEqual(client.get_bytes.call_args_list, [call(RD_URL)])

    def test_wrong_signature_for_either_product_fails(self):
        for responses in ([b"<html>not found</html>"], [PNG, PNG]):
            with self.subTest(responses=responses):
                client = client_with_images()
                client.get_bytes.side_effect = responses
                with self.assertRaisesRegex(CwaDataError, "expected image format"):
                    client.get_images_for_hour(START, END, now=NOW)


if __name__ == "__main__":
    unittest.main()
