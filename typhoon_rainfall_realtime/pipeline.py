"""Adapt downloaded images to the existing, unmodified inference API."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict

from typhoon_rainfall.config import ProjectConfig
from typhoon_rainfall_realtime.cwa import CwaClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TAIPEI = timezone(timedelta(hours=8))
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "realtime"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.json"


def new_run_dir(output_root: Path, command: str) -> Path:
    directory = Path(output_root).resolve() / (
        datetime.now(TAIPEI).strftime("%Y%m%d_%H%M%S_%f") + "_" + command
    )
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def write_manifest(directory: Path, manifest: dict) -> None:
    path = directory / "manifest.json"
    temporary = directory / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_prediction_config(path: Path) -> ProjectConfig:
    config = ProjectConfig.from_json(Path(path))
    data = config.data
    if (not data.use_rd or not data.use_ir or data.use_ra or data.use_gi
            or tuple(data.image_size) != (128, 128) or data.shift != 1):
        raise ValueError("Live plugin requires the thesis RD+IR, 128x128, t+1 configuration (RA/GI off).")
    if config.model.num_classes != 100:
        raise ValueError("Live plugin preserves the thesis 100-class rainfall labels.")
    checkpoint_dir = config.model.checkpoint_dir
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = PROJECT_ROOT / checkpoint_dir
    checkpoint = config.model.checkpoint_path
    if checkpoint is None:
        checkpoint = checkpoint_dir / (config.model.name + "_val_min_RMSE.h5")
    elif not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError("Checkpoint not found: " + str(checkpoint))
    return replace(config, model=replace(config.model, checkpoint_dir=checkpoint_dir, checkpoint_path=checkpoint))


def prepare_images(snapshots: dict, directory: Path) -> Dict[str, Path]:
    """Map both sources to the same thesis-region grid before model fusion."""
    import cv2
    import numpy as np
    from PIL import Image
    from io import BytesIO
    from typhoon_rainfall_realtime.geometry import GridSpec, SOURCE_GRIDS, reproject_image

    raw_dir, input_dir = directory / "raw", directory / "inputs"
    raw_dir.mkdir()
    input_dir.mkdir()
    paths = {}
    target = GridSpec()
    alignment = {"profile": "cwa-thesis-region-v1", "target_grid": target.to_dict(), "sources": {},
                 "scope": "Taiwan mainland region, 128x128; source projections handled independently",
                 "target_basis": "Archived training RD graticules, fitted on 176 train frames and checked on 56 held-out val frames",
                 "historical_grid_exact_reproduction_verified": False}
    for kind, extension, flag in (
        ("RD", ".png", cv2.IMREAD_COLOR), ("IR", ".jpg", cv2.IMREAD_GRAYSCALE),
    ):
        content = snapshots[kind].content
        (raw_dir / (kind + extension)).write_bytes(content)
        source = SOURCE_GRIDS[kind]
        if snapshots[kind].product_id != source.product_id:
            raise ValueError("Unexpected " + kind + " product for calibrated georeferencing")
        if kind == "RD" and (len(content) < 26 or content[:8] != b"\x89PNG\r\n\x1a\n"
                             or content[24] != 8 or content[25] != 2):
            raise ValueError("Unexpected RD image encoding; calibrated source requires 8-bit RGB PNG")
        with Image.open(BytesIO(content)) as image:
            if ((kind == "RD" and (image.format != "PNG" or image.mode != "RGB"))
                    or (kind == "IR" and (image.format != "JPEG" or image.mode not in ("L", "RGB")))):
                raise ValueError("Unexpected " + kind + " image encoding for calibrated georeferencing")
        decoded = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), flag)
        if decoded is None:
            raise ValueError("Unable to decode " + kind + " imagery")
        resized, alignment["sources"][kind] = reproject_image(decoded, source, target)
        path = input_dir / (kind + ".png")
        if not cv2.imwrite(str(path), resized):
            raise OSError("Unable to save preprocessed input: " + str(path))
        paths[kind] = path
    longitude, latitude = target.coordinates()
    np.savez_compressed(directory / "grid.npz", longitude_wgs84=longitude, latitude_wgs84=latitude)
    from typhoon_rainfall_realtime.alignment import ASSETS
    for asset in ("georeference_validation.json", "training_grid_validation.json"):
        (directory / asset).write_bytes((ASSETS / asset).read_bytes())
    alignment["validation_files"] = ["georeference_validation.json", "training_grid_validation.json"]
    (directory / "alignment.json").write_text(json.dumps(alignment, indent=2) + "\n", encoding="utf-8")
    return paths


def save_overview(paths: dict, prediction: object, directory: Path, base_time: datetime) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from typhoon_rainfall.visualization.plotting import Plotting

    plotter = Plotting()
    figure, axes = plt.subplots(1, 3, figsize=(10, 3.8))
    try:
        for axis, kind in zip(axes[:2], ("RD", "IR")):
            with Image.open(paths[kind]) as image:
                axis.imshow(image, cmap="gray" if kind == "IR" else None)
            axis.set_title(kind + " input")
        axes[2].imshow(prediction, cmap=plotter.precip_colormap, norm=plotter.norm)
        axes[2].set_title("t+1h prediction (labels 0-99)")
        for axis in axes:
            axis.axis("off")
        figure.suptitle("Thesis model demo | RD base: " + base_time.isoformat())
        figure.tight_layout()
        path = directory / "overview.png"
        figure.savefig(path, dpi=150)
        return path
    finally:
        plt.close(figure)


def configure_plotting() -> None:
    # Keep the optional plugin's cache inside the project, including sandbox runs.
    cache = PROJECT_ROOT / ".cache" / "matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))


def run_images(*, output_root: Path = DEFAULT_OUTPUT, config_path: Path = DEFAULT_CONFIG,
               fetch_only: bool = False, max_age_minutes: float = 90,
               max_skew_minutes: float = 20, client=None, predictor_factory=None) -> Path:
    # Validate the model contract before downloading, but fetch works without TF/checkpoints.
    config = None if fetch_only else load_prediction_config(config_path)
    directory = new_run_dir(output_root, "fetch" if fetch_only else "predict")
    manifest = {"schema_version": 1, "status": "fetching", "created_at": datetime.now(TAIPEI).isoformat(),
                "operation": "fetch" if fetch_only else "predict", "sources": {}}
    write_manifest(directory, manifest)
    source_client = client if client is not None else CwaClient()
    try:
        snapshots = source_client.get_latest_images(
            max_age_minutes=max_age_minutes, max_skew_minutes=max_skew_minutes)
        for kind, snapshot in snapshots.items():
            manifest["sources"][kind] = {
                "product_id": snapshot.product_id, "image_url": snapshot.image_url,
                "observed_at": snapshot.observed_at.isoformat(),
                "published_at": snapshot.published_at.isoformat() if snapshot.published_at else None,
                "acquired_at": snapshot.acquired_at.isoformat(),
                "sha256": hashlib.sha256(snapshot.content).hexdigest(),
                "raw_path": "raw/" + kind + (".png" if kind == "RD" else ".jpg"),
                "input_path": "inputs/" + kind + ".png",
            }
        rd_time = snapshots["RD"].observed_at
        manifest["prediction"] = {
            "base_time": rd_time.isoformat(), "target_time": (rd_time + timedelta(hours=1)).isoformat(),
            "lead_hours": 1, "input_shape": [128, 128, 4],
            "output_semantics": "Original thesis rainfall class indices 0-99; not a new calibration.",
            "time_basis": "RD observation time, not wall-clock execution time. IR time recorded separately.",
        }
        paths = prepare_images(snapshots, directory)
        manifest["alignment"] = json.loads((directory / "alignment.json").read_text(encoding="utf-8"))
        manifest["alignment"]["grid_file"] = "grid.npz"
        from typhoon_rainfall_realtime.alignment import save_alignment_preview
        manifest["alignment"]["preview"] = save_alignment_preview(paths, directory)
        if fetch_only:
            manifest["status"] = "fetched"
        else:
            configure_plotting()
            import matplotlib
            matplotlib.use("Agg")
            if predictor_factory is None:
                from typhoon_rainfall.inference.predictor import RainfallPredictor
                predictor_factory = RainfallPredictor
            predict_config = replace(config.predict, output_dir=directory,
                                     prediction_array_dir=directory, prediction_plot_dir=directory)
            config = replace(config, predict=predict_config)
            manifest["model"] = {"name": config.model.name, "checkpoint": str(config.model.checkpoint_path),
                                 "config_path": str(Path(config_path).resolve())}
            manifest["status"] = "predicting"
            write_manifest(directory, manifest)
            predictor = predictor_factory(config.data, config.model, config.predict)
            from PIL import Image
            with Image.open(paths["RD"]) as rd, Image.open(paths["IR"]) as ir:
                prediction = predictor.predict_modal_images(rd_image=rd, ir_image=ir)
            import numpy as np
            manifest["prediction"]["label_summary"] = {
                "minimum": int(np.min(prediction)), "maximum": int(np.max(prediction)),
                "nonzero_pixels": int(np.count_nonzero(prediction)), "pixel_count": int(np.size(prediction)),
            }
            artifacts = predictor.save_prediction(prediction, "prediction")
            overview = save_overview(paths, prediction, directory, rd_time)
            manifest["artifacts"] = {"csv": Path(artifacts.csv_path).name,
                                     "png": Path(artifacts.png_path).name, "overview": overview.name}
            manifest["status"] = "completed"
        manifest["finished_at"] = datetime.now(TAIPEI).isoformat()
        write_manifest(directory, manifest)
        return directory
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        write_manifest(directory, manifest)
        raise
    finally:
        if client is None:
            source_client.close()
