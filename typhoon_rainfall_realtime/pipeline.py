"""Adapt downloaded images to the existing, unmodified inference API."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict

from typhoon_rainfall.config import ProjectConfig
from typhoon_rainfall_realtime.cwa import CwaClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TAIPEI = timezone(timedelta(hours=8))
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "realtime"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "realtime.json"


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


def compact_prediction_output(directory: Path, manifest: dict) -> None:
    """Keep presentation artifacts from this newly created, completed run only."""
    directory = directory.resolve()
    if manifest.get("status") != "completed" or manifest.get("operation") != "predict":
        raise ValueError("Only completed predictions can be compacted")
    maps = manifest.get("ra", {}).get("maps", [])
    if maps:
        source = Path(maps[0]["directory"]).resolve() / "RA_map.png"
        source.relative_to(directory)
        shutil.copyfile(source, directory / "observed_ra.png")
    keep = {"inputs", "alignment.png", "forecast.png", "prediction_mm.csv", "manifest.json",
            "observed_ra.png", "overview.png"}
    # Legacy configurations have no hourly forecast or millimeter export.
    if "forecast" in manifest["artifacts"]:
        keep.discard("overview.png")
    else:
        keep.update({"prediction.png", "prediction.csv"})
    summary = {key: manifest[key] for key in
               ("schema_version", "operation", "status", "created_at", "finished_at", "prediction")}
    summary["output_mode"] = "compact"
    summary["model"] = {key: manifest["model"][key] for key in ("name", "modalities")}
    summary["model"]["sha256"] = manifest["model"]["checkpoint_contract"]["sha256"]
    summary["sources"] = {kind: {key: value[key] for key in
                           ("image_url", "observed_at", "sha256")} for kind, value in manifest["sources"].items()}
    summary["artifacts"] = {key: value for key, value in manifest["artifacts"].items() if value in keep}
    summary["artifacts"].update(inputs="inputs", alignment="alignment.png")
    if maps:
        summary["artifacts"]["observed_ra"] = "observed_ra.png"
    for child in directory.iterdir():
        if child.name in keep:
            continue
        # Never follow a link outside this run when removing intermediate files.
        child.resolve().relative_to(directory)
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
    write_manifest(directory, summary)


def load_prediction_config(path: Path) -> ProjectConfig:
    config = ProjectConfig.from_json(Path(path))
    data = config.data
    from typhoon_rainfall.data.contracts import InputModes, compute_input_shape
    if tuple(data.image_size) != (128, 128) or data.shift != 1:
        raise ValueError("Live plugin requires a 128x128, t+1 configuration.")
    compute_input_shape(data.image_size, InputModes.from_data_config(data))
    if config.model.num_classes not in (100, 101):
        raise ValueError("Live plugin supports EWB01 models with 100 or 101 classes.")
    if data.use_ra and config.model.num_classes != 101:
        raise ValueError("Generated RA uses thesis EWB01 101-class encoding; use matching trained weights.")
    if data.interval != "EWB01":
        raise ValueError("Live RA preprocessing requires EWB01 rainfall bins.")
    checkpoint_dir = config.model.checkpoint_dir
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = PROJECT_ROOT / checkpoint_dir
    checkpoint = config.model.checkpoint_path
    if checkpoint is None:
        checkpoint = checkpoint_dir / (config.model.name + "_val_min_RMSE.h5")
    elif not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(
            "Checkpoint not found: " + str(checkpoint)
            + ". Restore the pretrained checkpoint required by this configuration before predicting. "
              "The bundled demo does not require retraining."
        )
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
        try:
            with Image.open(BytesIO(content)) as image:
                if ((kind == "RD" and (image.format != "PNG" or image.mode != "RGB"))
                        or (kind == "IR" and (image.format != "JPEG" or image.mode not in ("L", "RGB")))):
                    raise ValueError("Unexpected " + kind + " image encoding for calibrated georeferencing")
                if image.size != (source.width, source.height):
                    raise ValueError("Unexpected " + kind + " image dimensions for calibrated georeferencing")
                # OpenCV may successfully decode a truncated JPEG with blank/corrupt
                # regions. Force the stricter full decode before accepting model data.
                image.load()
        except (OSError, SyntaxError) as exc:
            raise ValueError("Incomplete or invalid " + kind + " image payload") from exc
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
        axes[2].set_title("Next-hour rainfall prediction")
        for axis in axes:
            axis.axis("off")
        figure.suptitle("Thesis model demo | Forecast base: " + base_time.isoformat())
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


def _prepare_hourly_inputs(client, directory, config, max_age_minutes, boundary_file, now):
    """Use a complete observed hour and the RD/IR at that hour's midpoint."""
    import numpy as np
    from PIL import Image
    from typhoon_rainfall_realtime.rainfall import collect_rainfall
    from typhoon_rainfall_realtime.ra import generate_ra, load_boundary

    rainfall = collect_rainfall(client, directory / "observations", hours=1,
                               max_age_minutes=max_age_minutes, now=now)
    observations = Path(rainfall["files"]["hourly_json"])
    records = json.loads(observations.read_text(encoding="utf-8"))
    windows = {(row["interval_start"], row["interval_end"]) for row in records}
    if len(windows) != 1:
        raise ValueError("Multimodal prediction requires one shared observed rainfall hour")
    start_text, end_text = next(iter(windows))
    start, end = datetime.fromisoformat(start_text), datetime.fromisoformat(end_text)
    # get_images_for_hour rejects future/incomplete, stale, or already elapsed targets.
    snapshots = client.get_images_for_hour(start, end, max_age_minutes=max_age_minutes, now=now)
    boundary, boundary_info = load_boundary(client, boundary_file)
    ra = generate_ra(observations, directory / "ra", boundary, boundary_info)
    selected = Path(ra["maps"][0]["directory"])
    observed_mm = np.loadtxt(selected / "RA_mm.csv", delimiter=",")
    with np.load(directory / "ra" / "grid.npz") as grid_data:
        mask = grid_data["land_mask"].copy()
    if not np.all(np.isfinite(observed_mm[mask])):
        raise ValueError("Observed RA has uncovered mainland pixels; refusing missing model input")
    with Image.open(selected / "RA_labels.png") as image:
        labels = np.asarray(image)
    if np.any(labels > 100):
        raise ValueError("Observed RA has missing/unknown labels; refusing invalid model input")
    return snapshots, rainfall, ra, selected / "RA_labels.png", observed_mm, mask, start, end


def _predict_from_paths(predictor, config, paths):
    """Use the core's existing fusion rules, including explicit longitude/latitude."""
    from PIL import Image
    from typhoon_rainfall.data.contracts import InputModes
    from typhoon_rainfall.data.dataset import merge_modal_arrays

    with ExitStack() as stack:
        images = {key: stack.enter_context(Image.open(path)) for key, path in paths.items()}
        merged = merge_modal_arrays(
            InputModes.from_data_config(config.data), config.model.num_classes,
            rd_image=images.get("RD"), ir_image=images.get("IR"), ra_image=images.get("RA"),
            lon_image=images.get("LON"), lat_image=images.get("LAT"))
    return predictor.predict_array(merged), list(merged.shape)


def run_images(*, output_root: Path = DEFAULT_OUTPUT, config_path: Path = DEFAULT_CONFIG,
               fetch_only: bool = False, max_age_minutes: float = 90,
               max_skew_minutes: float = 20, client=None, predictor_factory=None,
               geography_source=None, boundary_file=None, now=None, detailed_output=True) -> Path:
    # Validate the model contract before downloading, but fetch works without TF/checkpoints.
    config = None if fetch_only else load_prediction_config(config_path)
    if now is not None and (now.tzinfo is None or now.utcoffset() is None):
        raise ValueError("now must be timezone-aware")
    if config is not None:
        from typhoon_rainfall.data.contracts import InputModes, compute_input_shape
        from typhoon_rainfall_realtime.model_contract import validate_checkpoint
        input_shape = compute_input_shape(config.data.image_size, InputModes.from_data_config(config.data))
        checkpoint_info = validate_checkpoint(config.model.checkpoint_path, input_shape,
                                              config.model.num_classes)
        metadata_path = config.model.checkpoint_path.with_suffix(".metadata.json")
        training_metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else None
        if (training_metadata is not None
                and training_metadata.get("checkpoint_sha256") != checkpoint_info["sha256"]):
            raise ValueError("Checkpoint fingerprint does not match its training metadata")
    directory = new_run_dir(output_root, "fetch" if fetch_only else "predict")
    manifest = {"schema_version": 1, "status": "fetching", "created_at": datetime.now(TAIPEI).isoformat(),
                "operation": "fetch" if fetch_only else "predict", "sources": {}}
    write_manifest(directory, manifest)
    source_client = client if client is not None else CwaClient()
    try:
        hourly = config is not None and config.data.use_ra
        observed_mm, land = None, None
        if hourly:
            (snapshots, rainfall, ra, ra_path, observed_mm, land,
             observed_start, base_time) = _prepare_hourly_inputs(
                source_client, directory, config, max_age_minutes, boundary_file, now)
            manifest["rainfall"] = rainfall
            manifest["ra"] = ra
        else:
            snapshots = source_client.get_latest_images(
                max_age_minutes=max_age_minutes, max_skew_minutes=max_skew_minutes)
            base_time = snapshots["RD"].observed_at
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
        target_time = base_time + timedelta(hours=1)
        manifest["prediction"] = {
            "base_time": base_time.isoformat(), "target_time": target_time.isoformat(),
            "lead_hours": 1, "input_shape": [128, 128, 4] if fetch_only else list(input_shape),
            "output_semantics": "EWB01 discrete hourly rainfall bins; class 100 means >=100 mm when enabled.",
            "time_basis": "Complete RA hour end; next hourly accumulation interval" if hourly
                          else "RD observation time; legacy RD/IR demonstration",
        }
        if hourly:
            manifest["prediction"].update(
                observed_interval=[observed_start.isoformat(), base_time.isoformat()],
                target_interval=[base_time.isoformat(), target_time.isoformat()],
                source_center=snapshots["RD"].observed_at.isoformat(),
                target_center=(snapshots["RD"].observed_at + timedelta(hours=1)).isoformat(),
                image_time_basis="Official dated filenames; RA-hour midpoint, not acquisition time")
        paths = prepare_images(snapshots, directory)
        if hourly:
            paths["RA"] = directory / "inputs" / "RA.png"
            shutil.copyfile(ra_path, paths["RA"])
        if config is not None and config.data.use_gi:
            from typhoon_rainfall_realtime.geography import export_geography
            gi_paths, gi_info = export_geography(directory / "inputs", geography_source)
            paths.update(gi_paths)
            manifest["geography"] = gi_info
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
                from typhoon_rainfall_realtime.runtime import configure_runtime
                configure_runtime()
                from typhoon_rainfall.inference.predictor import RainfallPredictor
                predictor_factory = RainfallPredictor
            predict_config = replace(config.predict, output_dir=directory,
                                     prediction_array_dir=directory, prediction_plot_dir=directory)
            config = replace(config, predict=predict_config)
            manifest["model"] = {"name": config.model.name, "checkpoint": str(config.model.checkpoint_path),
                                 "config_path": str(Path(config_path).resolve()),
                                 "checkpoint_contract": checkpoint_info,
                                 "modalities": [key for key, enabled in (("RD", config.data.use_rd),
                                     ("IR", config.data.use_ir), ("RA", config.data.use_ra),
                                     ("GI", config.data.use_gi)) if enabled],
                                 "normalization": {"RD_IR_GI": "/255", "RA": f"/{config.model.num_classes}"}}
            if training_metadata is not None:
                manifest["model"]["training"] = training_metadata
            manifest["status"] = "predicting"
            write_manifest(directory, manifest)
            predictor = predictor_factory(config.data, config.model, config.predict)
            prediction, actual_shape = _predict_from_paths(predictor, config, paths)
            manifest["prediction"]["input_shape"] = actual_shape
            import numpy as np
            prediction = np.asarray(prediction).copy()
            if prediction.shape != (128, 128) or not np.all(np.isfinite(prediction)):
                raise ValueError("Predictor returned an invalid rainfall grid")
            if np.any(prediction < 0) or np.any(prediction >= config.model.num_classes):
                raise ValueError("Predictor returned labels outside the trained rainfall bins")
            if hourly:
                prediction[~land] = 0
            manifest["prediction"]["label_summary"] = {
                "minimum": int(np.min(prediction)), "maximum": int(np.max(prediction)),
                "nonzero_pixels": int(np.count_nonzero(prediction)), "pixel_count": int(np.size(prediction)),
            }
            artifacts = predictor.save_prediction(prediction, "prediction")
            overview = save_overview(paths, prediction, directory, base_time)
            manifest["artifacts"] = {"csv": Path(artifacts.csv_path).name,
                                     "png": Path(artifacts.png_path).name, "overview": overview.name}
            if hourly:
                from typhoon_rainfall_realtime.geometry import GridSpec
                from typhoon_rainfall_realtime.presentation import save_forecast_view
                field = prediction.astype(float)
                field[~land] = np.nan
                np.savetxt(directory / "prediction_mm.csv", field, delimiter=",", fmt="%.0f")
                manifest["artifacts"]["rainfall_mm_csv"] = "prediction_mm.csv"
                manifest["artifacts"]["forecast"] = save_forecast_view(
                    observed_mm, prediction, land, GridSpec(), directory,
                    observed_start, base_time, target_time)
                issued_at = now if now is not None else datetime.now(TAIPEI)
                if issued_at >= target_time:
                    raise ValueError("Target hour elapsed during prediction; refusing an outdated live forecast")
                manifest["prediction"]["issued_at"] = issued_at.isoformat()
                manifest["prediction"]["target_elapsed_minutes_at_issue"] = max(
                    0.0, (issued_at - base_time).total_seconds() / 60)
            manifest["status"] = "completed"
        manifest["finished_at"] = datetime.now(TAIPEI).isoformat()
        write_manifest(directory, manifest)
        if not fetch_only and not detailed_output:
            compact_prediction_output(directory, manifest)
        return directory
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        write_manifest(directory, manifest)
        raise
    finally:
        if client is None:
            source_client.close()
