"""Dataset contracts shared by training, evaluation, and prediction.

這裡只描述資料「應該長什麼樣子」與路徑規則，不負責真正讀圖。
這樣可以把資料契約和影像處理分開，日後若資料夾結構改變，優先
檢查本檔案即可。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from typhoon_rainfall.config import DataConfig


@dataclass(frozen=True)
class InputModes:
    """Flags describing which image modalities are enabled."""

    use_rd: bool = True
    use_ir: bool = True
    use_ra: bool = False
    use_gi: bool = False

    @classmethod
    def from_data_config(cls, config: DataConfig) -> "InputModes":
        """Create modality flags from the top-level data config."""
        return cls(
            use_rd=config.use_rd,
            use_ir=config.use_ir,
            use_ra=config.use_ra,
            use_gi=config.use_gi,
        )


@dataclass(frozen=True)
class SamplePair:
    """One source-target pair from `train.txt` or `val.txt`.

    `source_name` 是輸入時間點，`target_name` 是要預測的未來時間點。
    例如 `201308271130.png;201308271230.png` 表示用 11:30 的影像
    預測 12:30 的雨量標籤。
    """

    source_name: str
    target_name: str

    @property
    def source_stem(self) -> str:
        """Filename without extension for source image outputs."""
        return Path(self.source_name).stem

    @property
    def target_stem(self) -> str:
        """Filename without extension for target image outputs."""
        return Path(self.target_name).stem

    def prediction_stem(self, shift: int) -> str:
        """Return the conventional prediction filename stem, e.g. `xxx_t+1`."""
        return f"{self.source_stem}_t+{shift}"


@dataclass(frozen=True)
class DatasetPaths:
    """Centralized path builder for the fixed research dataset layout."""

    root: Path
    interval: str

    def rd_path(self, split: str, filename: str) -> Path:
        """Path to a radar reflectivity image."""
        return self.root / "train_data_RD" / split / filename

    def ir_path(self, split: str, filename: str) -> Path:
        """Path to an infrared satellite image."""
        return self.root / "train_data_IR" / split / filename

    def ra_path(self, split: str, filename: str) -> Path:
        """Path to a rainfall label image under the selected interval."""
        return self.root / "train_data_RA" / self.interval / split / filename

    def lon_path(self) -> Path:
        """Path to the longitude helper image used when GI is enabled."""
        return self.root / "train_data_GI" / "lon.png"

    def lat_path(self) -> Path:
        """Path to the latitude helper image used when GI is enabled."""
        return self.root / "train_data_GI" / "lat.png"

    def val_true_csv_path(self, filename_stem: str) -> Path:
        """Path to validation true-value CSV used for extra analysis."""
        return self.root / "val_label_csv_true_value" / f"{filename_stem}.csv"

    def val_color_png_path(self, filename_stem: str) -> Path:
        """Path to validation color rainfall map used for visualization."""
        return self.root / "val_label_RA_colorimage" / f"{filename_stem}.png"


def parse_annotation_file(list_path: Path) -> List[SamplePair]:
    """Parse a `source;target` annotation list into SamplePair objects."""
    pairs: List[SamplePair] = []
    for raw_line in list_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        source_name, target_name = line.split(";")
        pairs.append(SamplePair(source_name=source_name, target_name=target_name))
    return pairs


def compute_input_shape(
    image_size: Sequence[int],
    input_modes: InputModes,
) -> Tuple[int, int, int]:
    """Compute model input shape from image size and enabled modalities.

    Channel convention follows the original research code:
    RD contributes 3 channels, IR contributes 1 channel when combined with
    other modalities but is expanded to 3 channels when used alone, RA as input
    contributes 3 channels, and GI contributes lon/lat 2 channels.
    """

    if not any([input_modes.use_rd, input_modes.use_ir, input_modes.use_ra]):
        raise ValueError("At least one of RD, IR, or RA must be enabled.")

    channels = 0
    if input_modes.use_rd:
        channels += 3
    if input_modes.use_ir:
        channels += 1 if input_modes.use_rd or input_modes.use_ra else 3
    if input_modes.use_ra:
        channels += 3
    if input_modes.use_gi:
        channels += 2
    return int(image_size[0]), int(image_size[1]), channels


def ensure_dataset_contract(
    dataset_paths: DatasetPaths,
    split: str,
    pairs: Iterable[SamplePair],
) -> List[Path]:
    """Return missing files for a list of pairs without raising immediately."""
    missing: List[Path] = []
    for pair in pairs:
        for path in (
            dataset_paths.rd_path(split, pair.source_name),
            dataset_paths.ir_path(split, pair.source_name),
            dataset_paths.ra_path(split, pair.source_name),
            dataset_paths.ra_path(split, pair.target_name),
        ):
            if not path.exists():
                missing.append(path)
    return missing
