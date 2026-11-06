"""Small anonymous client for the CWA images used by the existing model.

The XML schemas were checked against the public products on 2026-10-31:
RD uses ``dataset/DateTime``; IR uses ``dataset/ObsTime/Datetime``.
The fixed S3 objects are current snapshots, not a historical-data API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import math
from typing import Dict, Optional, Tuple, Union
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET

import requests


TAIPEI = timezone(timedelta(hours=8))
_BASE_URL = "https://cwaopendata.s3.ap-northeast-1.amazonaws.com/Observation"


class CwaDataError(RuntimeError):
    """The source could not provide a usable, dated weather snapshot."""


class StaleDataError(CwaDataError):
    """An observation is older than the caller permits."""


class DataConsistencyError(CwaDataError):
    """Products disagree in time, or changed while they were downloaded."""


@dataclass(frozen=True)
class ImageSnapshot:
    kind: str
    product_id: str
    image_url: str
    observed_at: datetime
    published_at: Optional[datetime]
    acquired_at: datetime
    content: bytes = field(repr=False)


@dataclass(frozen=True)
class _Product:
    product_id: str
    extension: str
    time_path: Tuple[str, ...]

    @property
    def metadata_url(self) -> str:
        return f"{_BASE_URL}/{self.product_id}.xml"

    @property
    def image_url(self) -> str:
        return f"{_BASE_URL}/{self.product_id}.{self.extension}"


_PRODUCTS = {
    "RD": _Product("O-A0058-003", "png", ("dataset", "DateTime")),
    "IR": _Product("O-C0042-004", "jpg", ("dataset", "ObsTime", "Datetime")),
}


@dataclass(frozen=True)
class _Manifest:
    observed_at: datetime
    published_at: Optional[datetime]
    identifier: Optional[str]


def _text_at(element: ET.Element, path: Tuple[str, ...]) -> Optional[str]:
    """Read an exact path, tolerating the official XML default namespace."""
    for name in path:
        element = next(
            (child for child in element if child.tag.rsplit("}", 1)[-1] == name),
            None,
        )
        if element is None:
            return None
    return element.text.strip() if element.text and element.text.strip() else None


def _parse_time(value: Optional[str], name: str) -> datetime:
    if not value:
        raise CwaDataError(f"CWA metadata is missing {name}")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CwaDataError(f"CWA metadata has an invalid {name}: {value!r}") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise CwaDataError(f"CWA metadata {name} has no timezone: {value!r}")
    return result.astimezone(TAIPEI)


def _parse_manifest(payload: bytes, product: _Product) -> _Manifest:
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise CwaDataError(f"Invalid XML for {product.product_id}") from exc

    if _text_at(root, ("dataid",)) != product.product_id:
        raise CwaDataError(f"CWA metadata product does not match {product.product_id}")
    resource_name = "resource" if product.extension == "png" else "Resource"
    image_url = _text_at(root, ("dataset", resource_name, "ProductURL"))
    if image_url != product.image_url:
        raise CwaDataError(f"Unexpected image URL in {product.product_id} metadata")

    published = _text_at(root, ("sent",))
    return _Manifest(
        observed_at=_parse_time(_text_at(root, product.time_path), "observation time"),
        published_at=_parse_time(published, "publication time") if published else None,
        identifier=_text_at(root, ("identifier",)),
    )


class CwaClient:
    """Fetch dated RD/IR snapshots without an account, cookies or API key.

    ``session`` can be supplied for tests or connection reuse. No files are
    written. An unchanged metadata pair around the image downloads guards
    against an update during acquisition; it cannot prove upstream file
    publication is atomic.
    """

    def __init__(
        self,
        session: Optional[requests.Session] = None,
        timeout: Union[float, Tuple[float, float]] = (5.0, 20.0),
        max_download_bytes: int = 8 * 1024 * 1024,
        max_attempts: int = 3,
    ) -> None:
        if max_download_bytes <= 0:
            raise ValueError("max_download_bytes must be positive")
        if not 1 <= max_attempts <= 5:
            raise ValueError("max_attempts must be between 1 and 5")
        timeouts = timeout if isinstance(timeout, tuple) else (timeout,)
        if len(timeouts) not in (1, 2) or any(
            not math.isfinite(value) or value <= 0 for value in timeouts
        ):
            raise ValueError("timeout must contain positive, finite seconds")
        self.session = session if session is not None else requests.Session()
        self._owns_session = session is None
        self.timeout = timeout
        self.max_download_bytes = max_download_bytes
        self.max_attempts = max_attempts

    def close(self) -> None:
        if self._owns_session:
            self.session.close()

    def get_bytes(self, url: str) -> bytes:
        """GET one public resource with timeouts and a streamed size limit."""
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("CWA resource URLs must use HTTPS")
        response = None
        try:
            response = self.session.get(
                url,
                timeout=self.timeout,
                stream=True,
                headers={"User-Agent": "typhoon-rainfall-realtime/1.0"},
            )
            response.raise_for_status()
            declared_length = response.headers.get("Content-Length")
            if declared_length:
                try:
                    too_large = int(declared_length) > self.max_download_bytes
                except (TypeError, ValueError) as exc:
                    raise CwaDataError("CWA response has invalid Content-Length") from exc
                if too_large:
                    raise CwaDataError("CWA response exceeds the download size limit")
            chunks = []
            size = 0
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                size += len(chunk)
                if size > self.max_download_bytes:
                    raise CwaDataError("CWA response exceeds the download size limit")
                chunks.append(chunk)
            if not chunks:
                raise CwaDataError("CWA returned an empty resource")
            return b"".join(chunks)
        except requests.RequestException as exc:
            # Keep query parameters out of the public error message.
            target = f"{parsed.hostname}{parsed.path}"
            raise CwaDataError(f"Download failed for {target}: {type(exc).__name__}") from exc
        finally:
            if response is not None:
                response.close()

    @staticmethod
    def _validate_time(manifest: _Manifest, now: datetime, max_age: timedelta) -> None:
        if manifest.observed_at > now:
            raise CwaDataError("CWA observation time is in the future")
        if now - manifest.observed_at > max_age:
            raise StaleDataError(
                f"CWA observation is too old: {manifest.observed_at.isoformat()}"
            )
        if manifest.published_at is not None:
            if manifest.published_at > now:
                raise CwaDataError("CWA publication time is in the future")
            if manifest.published_at < manifest.observed_at:
                raise CwaDataError("CWA publication precedes the observation")

    def get_images_for_hour(
        self,
        interval_start: datetime,
        interval_end: datetime,
        *,
        now: Optional[datetime] = None,
        max_age_minutes: float = 90,
        require_future_target: bool = True,
    ) -> Dict[str, ImageSnapshot]:
        """Fetch the RD/IR images at a completed rain hour's midpoint.

        The official dated website filenames declare the observation time in
        Taiwan time. These are distinct from the mutable latest S3 snapshots;
        no fallback to newer imagery is permitted. Publication time is unknown
        for dated files and remains ``None``. Decoding/dimension validation is
        performed by the shared image preparation pipeline.

        ``max_age_minutes`` applies to the rain interval's end. By default its
        successor hour must not have ended yet; the caller must still display
        the actual target interval rather than claim execution-time + 1 hour.
        """
        local_times = []
        for name, value in (("interval_start", interval_start), ("interval_end", interval_end),
                            ("now", now)):
            if value is None and name == "now":
                continue
            if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(name + " must be timezone-aware")
            local_times.append(value.astimezone(TAIPEI))
        start, end = local_times[:2]
        reference = local_times[2] if now is not None else None
        if end - start != timedelta(hours=1) or any(
            value.minute or value.second or value.microsecond for value in (start, end)
        ):
            raise ValueError("Rain interval must be exactly one whole hour in Taiwan time")
        if not math.isfinite(max_age_minutes) or max_age_minutes <= 0:
            raise ValueError("max_age_minutes must be positive and finite")

        def validate_reference(value: datetime) -> None:
            if end > value:
                raise DataConsistencyError("Rain interval is not complete yet")
            if value - end > timedelta(minutes=max_age_minutes):
                raise StaleDataError("Rain interval is too old for paired imagery")
            if require_future_target and end + timedelta(hours=1) <= value:
                raise StaleDataError("The target rain interval has already ended")

        validate_reference(reference if reference is not None else datetime.now(TAIPEI))
        midpoint = start + timedelta(minutes=30)
        urls = {
            "RD": "https://www.cwa.gov.tw/Data/radar/CV1_TW_3600_"
                  + midpoint.strftime("%Y%m%d%H%M") + ".png",
            "IR": "https://www.cwa.gov.tw/Data/satellite/TWI_IR1_Gray_800/"
                  "TWI_IR1_Gray_800-" + midpoint.strftime("%Y-%m-%d-%H-%M") + ".jpg",
        }
        snapshots = {}
        for kind, url in urls.items():
            content = self.get_bytes(url)
            signature = b"\x89PNG\r\n\x1a\n" if kind == "RD" else b"\xff\xd8\xff"
            if not content.startswith(signature):
                raise CwaDataError(f"CWA dated {kind} resource is not the expected image format")
            snapshots[kind] = ImageSnapshot(
                kind=kind, product_id=_PRODUCTS[kind].product_id, image_url=url,
                observed_at=midpoint, published_at=None,
                acquired_at=reference if reference is not None else datetime.now(TAIPEI),
                content=content,
            )
        validate_reference(reference if reference is not None else datetime.now(TAIPEI))
        return snapshots

    def get_latest_images(
        self,
        max_age_minutes: float = 90,
        max_skew_minutes: float = 20,
        now: Optional[datetime] = None,
    ) -> Dict[str, ImageSnapshot]:
        """Return ``RD`` and ``IR`` snapshots, or fail with a source error.

        Observation time is copied from metadata, never from the wall clock.
        ``now`` must be timezone-aware when supplied, primarily for testing.
        No timestamp rounding or historical filename guessing is performed.
        """
        if not math.isfinite(max_age_minutes) or max_age_minutes <= 0:
            raise ValueError("max_age_minutes must be positive and finite")
        if not math.isfinite(max_skew_minutes) or max_skew_minutes < 0:
            raise ValueError("max_skew_minutes must be nonnegative and finite")
        if now is not None:
            if now.tzinfo is None or now.utcoffset() is None:
                raise ValueError("now must be timezone-aware")
            now = now.astimezone(TAIPEI)
        max_age = timedelta(minutes=max_age_minutes)
        max_skew = timedelta(minutes=max_skew_minutes)

        for _ in range(self.max_attempts):
            before = {
                kind: _parse_manifest(self.get_bytes(product.metadata_url), product)
                for kind, product in _PRODUCTS.items()
            }
            reference_time = now if now is not None else datetime.now(TAIPEI)
            for manifest in before.values():
                self._validate_time(manifest, reference_time, max_age)
            if abs(before["RD"].observed_at - before["IR"].observed_at) > max_skew:
                raise DataConsistencyError("CWA radar and infrared times are too far apart")

            images = {}
            acquired = {}
            for kind, product in _PRODUCTS.items():
                images[kind] = self.get_bytes(product.image_url)
                signature = b"\x89PNG\r\n\x1a\n" if kind == "RD" else b"\xff\xd8\xff"
                if not images[kind].startswith(signature):
                    raise CwaDataError(f"CWA {kind} resource is not the expected image format")
                acquired[kind] = now if now is not None else datetime.now(TAIPEI)
            after = {
                kind: _parse_manifest(self.get_bytes(product.metadata_url), product)
                for kind, product in _PRODUCTS.items()
            }
            if before != after:
                continue

            reference_time = now if now is not None else datetime.now(TAIPEI)
            for manifest in after.values():
                self._validate_time(manifest, reference_time, max_age)
            return {
                kind: ImageSnapshot(
                    kind=kind,
                    product_id=product.product_id,
                    image_url=product.image_url,
                    observed_at=after[kind].observed_at,
                    published_at=after[kind].published_at,
                    acquired_at=acquired[kind],
                    content=images[kind],
                )
                for kind, product in _PRODUCTS.items()
            }

        raise DataConsistencyError(
            f"CWA metadata kept changing during {self.max_attempts} acquisition attempts"
        )
