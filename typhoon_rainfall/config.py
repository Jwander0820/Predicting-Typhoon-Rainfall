"""Project-wide configuration objects.

這個檔案集中保存「研究設定」而不是模型邏輯，目的是讓訓練、
驗證、推論都讀同一份設定來源，避免 notebook、CLI、舊腳本各自
硬編碼一套參數。設定類別都使用 frozen dataclass，降低執行過程中
被意外修改的風險，對論文研究的可重現性比較友善。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple


ImageSize = Tuple[int, int]


@dataclass(frozen=True)
class DataConfig:
    """Dataset and input-modality settings.

    控制資料根目錄、影像大小、預測時間位移，以及本次實驗要使用
    哪些輸入模態。預設值對應目前論文研究中最常用的 RD + IR、
    EWB01、t+1 設定。
    """

    dataset_root: Path = Path("database")
    image_size: ImageSize = (128, 128)
    interval: str = "EWB01"
    shift: int = 1
    use_rd: bool = True
    use_ir: bool = True
    use_ra: bool = False
    use_gi: bool = False

    def train_list_dir(self) -> Path:
        """Return the directory that stores train/validation pair lists."""
        return self.dataset_root / "train_list" / f"t2t+{self.shift}"


@dataclass(frozen=True)
class ModelConfig:
    """Model architecture and checkpoint settings.

    `name` 對應既有模型工廠中的名稱，例如 Unet3、Unet5、FCN8。
    `checkpoint_path` 若有指定，會優先載入該權重；否則使用
    `checkpoint_dir` 中符合模型名稱的預設權重。
    """

    name: str = "Unet5"
    num_classes: int = 100
    checkpoint_dir: Path = Path("checkpoint")
    checkpoint_path: Optional[Path] = None


@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters and output locations."""

    epochs: int = 100
    batch_size: int = 8
    validation_batch_size: int = 1
    learning_rate: float = 1e-4
    use_focal_loss: bool = False
    checkpoint_dir: Path = Path("checkpoint")
    logs_dir: Path = Path("logs")
    output_dir: Path = Path("output")


@dataclass(frozen=True)
class PredictConfig:
    """Prediction and evaluation output settings."""

    output_dir: Path = Path("output")
    prediction_array_dir: Path = Path("output/predict_array")
    prediction_plot_dir: Path = Path("output/predict")
    blend: bool = False
    source_split: str = "val"
    limit: Optional[int] = None


@dataclass(frozen=True)
class ProjectConfig:
    """Top-level config that groups data, model, training, and prediction."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    predict: PredictConfig = field(default_factory=PredictConfig)

    @classmethod
    def defaults(cls) -> "ProjectConfig":
        """Build the default research configuration."""
        return cls()

    @classmethod
    def from_json(cls, config_path: Path) -> "ProjectConfig":
        """Load a project config from a UTF-8 JSON file."""
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ProjectConfig":
        """Build a config from dictionaries loaded from JSON or tests."""
        sections = {
            "data": _build_dataclass(DataConfig, payload.get("data", {})),
            "model": _build_dataclass(ModelConfig, payload.get("model", {})),
            "train": _build_dataclass(TrainConfig, payload.get("train", {})),
            "predict": _build_dataclass(PredictConfig, payload.get("predict", {})),
        }
        return cls(**sections)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the config into JSON-compatible primitive values."""
        return {
            "data": _serialize_paths(asdict(self.data)),
            "model": _serialize_paths(asdict(self.model)),
            "train": _serialize_paths(asdict(self.train)),
            "predict": _serialize_paths(asdict(self.predict)),
        }


def _build_dataclass(dataclass_type: Any, payload: Dict[str, Any]) -> Any:
    """Convert a JSON section into one of the config dataclasses."""
    converted: Dict[str, Any] = {}
    for key, value in payload.items():
        if value is None:
            converted[key] = value
        elif key.endswith("_dir") or key.endswith("_root") or key.endswith("_path"):
            converted[key] = Path(value)
        elif key == "image_size":
            converted[key] = tuple(value)
        else:
            converted[key] = value
    return dataclass_type(**converted)


def _serialize_paths(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Path and tuple values into JSON-friendly representations."""
    converted: Dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, Path):
            converted[key] = str(value)
        elif isinstance(value, tuple):
            converted[key] = list(value)
        else:
            converted[key] = value
    return converted


def save_project_config(config: ProjectConfig, output_path: Path) -> None:
    """Write a config to disk for reproducible reruns."""
    output_path.write_text(
        json.dumps(config.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def apply_overrides(config: ProjectConfig, args: Any) -> ProjectConfig:
    """Apply command-line overrides without mutating the original config.

    CLI 參數只覆蓋使用者明確傳入的值；沒有傳入的欄位會保留 JSON
    或預設設定，避免一次命令不小心改掉整個研究設定。
    """

    data = dict(config.to_dict()["data"])
    model = dict(config.to_dict()["model"])
    train = dict(config.to_dict()["train"])
    predict = dict(config.to_dict()["predict"])

    _override_if_present(data, "interval", args.interval)
    _override_if_present(data, "shift", args.shift)
    _override_if_present(model, "name", args.model_name)
    _override_if_present(model, "checkpoint_path", args.checkpoint)
    _override_if_present(train, "epochs", args.epochs)
    _override_if_present(train, "batch_size", args.batch_size)
    _override_if_present(train, "learning_rate", args.learning_rate)
    _override_if_present(predict, "limit", args.limit)

    return ProjectConfig.from_dict(
        {
            "data": data,
            "model": model,
            "train": train,
            "predict": predict,
        }
    )


def _override_if_present(target: Dict[str, Any], key: str, value: Any) -> None:
    """Set a config field only when the CLI provided a non-None value."""
    if value is not None:
        target[key] = value
