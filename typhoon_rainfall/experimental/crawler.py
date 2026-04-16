"""Experimental weather image crawler.

這個模組保留原本即時抓圖的研究實驗功能。它不是核心訓練資料來源，
因此測試以時間對齊與 URL 規則為主；實際下載仍可能受外部網站格式
與網路狀態影響。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class FormattedTime:
    """Formatted timestamp components used by weather image URLs."""

    year: str
    month: str
    day: str
    hour: str
    minute: str

    def as_tuple(self) -> Tuple[str, str, str, str, str]:
        """Return legacy tuple format used by old scripts."""
        return (self.year, self.month, self.day, self.hour, self.minute)

    @property
    def compact(self) -> str:
        """Return YYYYMMDDHHMM format used by radar filenames."""
        return f"{self.year}{self.month}{self.day}{self.hour}{self.minute}"


class WeatherCrawler:
    """Download and crop IR/RD weather images for experimental prediction."""

    def __init__(self, output_dir: Path = Path("img")) -> None:
        self.my_headers = {"user-agent": "typhoon-rainfall-research/1.0"}
        self.output_dir = output_dir

    def get_format_time(self, specified_time: Optional[datetime] = None):
        """Compatibility wrapper returning the old five-value tuple."""
        return self.format_time(specified_time).as_tuple()

    def format_time(self, specified_time: Optional[datetime] = None) -> FormattedTime:
        """Align arbitrary time to the nearest available CWB image timestamp.

        Original rule:
        - minute 00-29: use previous hour at 30 minutes
        - minute 30-59: use current hour at 00 minutes
        """

        dt = specified_time or datetime.now()
        if 0 <= dt.minute < 30:
            dt = dt.replace(minute=30, second=0, microsecond=0) - timedelta(hours=1)
        else:
            dt = dt.replace(minute=0, second=0, microsecond=0)
        return FormattedTime(
            year=str(dt.year).zfill(2),
            month=str(dt.month).zfill(2),
            day=str(dt.day).zfill(2),
            hour=str(dt.hour).zfill(2),
            minute=str(dt.minute).zfill(2),
        )

    def build_urls(self, specified_time: Optional[datetime] = None) -> Tuple[str, str]:
        """Build IR and RD image URLs for an aligned timestamp."""
        formatted = self.format_time(specified_time)
        ir_url = (
            "https://www.cwb.gov.tw/Data/satellite/TWI_IR1_Gray_800/"
            f"TWI_IR1_Gray_800-{formatted.year}-{formatted.month}-{formatted.day}-"
            f"{formatted.hour}-{formatted.minute}.jpg"
        )
        rd_url = (
            "https://www.cwb.gov.tw/Data/radar/"
            f"CV1_TW_3600_{formatted.compact}.png"
        )
        return ir_url, rd_url

    def _get_image(self, url: str, filename: Path, crop_grayscale: bool = False) -> bool:
        """Download one image, crop Taiwan area, resize to 128x128, and save."""
        import cv2
        import requests

        try:
            response = requests.get(url, headers=self.my_headers, timeout=5)
            response.raise_for_status()
            filename.parent.mkdir(parents=True, exist_ok=True)
            filename.write_bytes(response.content)
            if crop_grayscale:
                image = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
                cropped = image[261:558, 247:544]
            else:
                image = cv2.imread(str(filename), cv2.IMREAD_COLOR)
                cropped = image[716:2762, 830:2876]
            resized = cv2.resize(cropped, (128, 128), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(filename.with_name(f"{filename.stem}_crop.png")), resized)
            return True
        except Exception as exc:  # pragma: no cover - integration-oriented path
            print(f"沒有爬取到{filename.name}")
            print(exc)
            return False

    def get_weather_images(self, specified_time: Optional[datetime] = None):
        """Download both IR and RD images and return success flags."""
        formatted = self.format_time(specified_time)
        ir_url, rd_url = self.build_urls(specified_time)
        ir = self._get_image(ir_url, self.output_dir / f"{formatted.compact}_IR.png", crop_grayscale=True)
        rd = self._get_image(rd_url, self.output_dir / f"{formatted.compact}_RD.png", crop_grayscale=False)
        return ir, rd
