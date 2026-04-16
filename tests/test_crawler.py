from datetime import datetime
import unittest

from typhoon_rainfall.experimental.crawler import WeatherCrawler


class CrawlerTests(unittest.TestCase):
    def test_time_alignment_for_early_minutes(self):
        crawler = WeatherCrawler()
        formatted = crawler.format_time(datetime(2023, 3, 12, 0, 20))
        self.assertEqual(formatted.as_tuple(), ("2023", "03", "11", "23", "30"))

    def test_time_alignment_for_late_minutes(self):
        crawler = WeatherCrawler()
        formatted = crawler.format_time(datetime(2023, 3, 12, 0, 50))
        self.assertEqual(formatted.as_tuple(), ("2023", "03", "12", "00", "00"))

    def test_build_urls(self):
        crawler = WeatherCrawler()
        ir_url, rd_url = crawler.build_urls(datetime(2023, 3, 12, 0, 50))
        self.assertIn("TWI_IR1_Gray_800-2023-03-12-00-00.jpg", ir_url)
        self.assertIn("CV1_TW_3600_202303120000.png", rd_url)


if __name__ == "__main__":
    unittest.main()
