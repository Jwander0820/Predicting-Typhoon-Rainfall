"""Command-line entrypoint for train/evaluate/predict workflows.

PyCharm users can run the helper scripts in the project root. CLI users can run
`python -m typhoon_rainfall ...`. This module keeps argument parsing separate
from the actual training/evaluation implementation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from typhoon_rainfall.config import ProjectConfig, apply_overrides


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser and supported subcommands."""
    parser = argparse.ArgumentParser(description="Typhoon rainfall research CLI")
    parser.add_argument(
        "--config",
        dest="global_config",
        type=Path,
        default=None,
        help="Path to a JSON config file.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("train", "evaluate", "predict"):
        subparser = subparsers.add_parser(name)
        _add_common_arguments(subparser)
        if name == "predict":
            subparser.add_argument("--rd-image", type=Path, default=None)
            subparser.add_argument("--ir-image", type=Path, default=None)
            subparser.add_argument("--ra-image", type=Path, default=None)
            subparser.add_argument("--stem", default="prediction")
    return parser


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach options shared by train/evaluate/predict subcommands."""
    parser.add_argument(
        "--config",
        dest="command_config",
        type=Path,
        default=None,
        help="Path to a JSON config file.",
    )
    parser.add_argument("--model-name", dest="model_name", default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--interval", default=None)
    parser.add_argument("--shift", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)


def load_config(config_path: Optional[Path]) -> ProjectConfig:
    """Load a JSON config when provided, otherwise use research defaults."""
    if config_path is None:
        return ProjectConfig.defaults()
    return ProjectConfig.from_json(config_path)


def main(argv: Optional[list[str]] = None) -> int:
    """Parse arguments and dispatch to the selected workflow."""
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = args.command_config or args.global_config
    config = apply_overrides(load_config(config_path), args)
    _print_run_context(args.command, config_path, config)

    if args.command == "train":
        from typhoon_rainfall.training.pipeline import run_training

        result = run_training(config)
        _print_result("Training finished", result)
    elif args.command == "evaluate":
        from typhoon_rainfall.evaluation.runner import run_evaluation

        result = run_evaluation(config)
        _print_result("Evaluation finished", result)
    elif args.command == "predict":
        from typhoon_rainfall.inference.runner import run_prediction, run_single_prediction

        if args.rd_image or args.ir_image or args.ra_image:
            result = run_single_prediction(
                config=config,
                stem=args.stem,
                rd_image_path=args.rd_image,
                ir_image_path=args.ir_image,
                ra_image_path=args.ra_image,
            )
        else:
            result = run_prediction(config)
        _print_result("Prediction finished", result)
    else:  # pragma: no cover
        parser.error(f"Unknown command: {args.command}")
    return 0


def _print_run_context(command: str, config_path: Optional[Path], config: ProjectConfig) -> None:
    """Print the effective model/data settings before running a workflow.

    PyCharm 直接 RUN 時不容易知道實際採用哪個模型與 checkpoint。
    這裡集中列出「本次有效設定」，方便確認 config / CLI override
    是否符合預期。
    """

    enabled_modes = []
    if config.data.use_rd:
        enabled_modes.append("RD")
    if config.data.use_ir:
        enabled_modes.append("IR")
    if config.data.use_ra:
        enabled_modes.append("RA")
    if config.data.use_gi:
        enabled_modes.append("GI")

    print("\nRun context")
    print(f"- command: {command}")
    print(f"- config: {config_path or 'ProjectConfig.defaults()'}")
    print(f"- model: {config.model.name}")
    print(f"- num_classes: {config.model.num_classes}")
    print(f"- checkpoint_dir: {config.model.checkpoint_dir}")
    print(f"- checkpoint_path: {config.model.checkpoint_path or '(auto)'}")
    print(f"- dataset_root: {config.data.dataset_root}")
    print(f"- interval: {config.data.interval}")
    print(f"- shift: t+{config.data.shift}")
    print(f"- input_modes: {' + '.join(enabled_modes)}")
    print(f"- image_size: {config.data.image_size[0]}x{config.data.image_size[1]}")
    if command == "predict":
        print(f"- source_split: {config.predict.source_split}")
        print(f"- limit: {config.predict.limit if config.predict.limit is not None else '(all)'}")
        print(f"- prediction_array_dir: {config.predict.prediction_array_dir}")
        print(f"- prediction_plot_dir: {config.predict.prediction_plot_dir}")


def _print_result(title: str, result) -> None:
    """Print a short human-readable summary after CLI workflows finish."""
    print(f"\n{title}")
    if not isinstance(result, dict):
        return
    for key, value in result.items():
        if isinstance(value, (Path, int, float, str)):
            print(f"- {key}: {value}")
