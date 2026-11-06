"""Separate CLI for optional live-data collection and thesis-model inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from typhoon_rainfall_realtime.cwa import CwaClient
from typhoon_rainfall_realtime.pipeline import (
    DEFAULT_CONFIG, DEFAULT_OUTPUT, new_run_dir, run_images, write_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CWA live-data plugin for the existing thesis model")
    commands = parser.add_subparsers(dest="command", required=True)
    for command, description in (("predict", "Download latest RD/IR and run existing weights"),
                                 ("fetch", "Download and preprocess RD/IR without loading TensorFlow")):
        child = commands.add_parser(command, help=description)
        child.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
        child.add_argument("--max-age-minutes", type=float, default=90)
        child.add_argument("--max-skew-minutes", type=float, default=20)
        if command == "predict":
            child.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    rain = commands.add_parser("rainfall", help="Collect observations and generate hourly IDW RA maps")
    rain.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    rain.add_argument("--hours", type=int, choices=range(1, 25), default=6)
    rain.add_argument("--observations-only", action="store_true", help="Skip RA interpolation")
    ra = commands.add_parser("ra", help="Generate IDW RA maps from a saved rainfall_hourly.json")
    ra.add_argument("--observations", type=Path, required=True)
    ra.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    for child in (rain, ra):
        child.add_argument("--boundary-file", type=Path, help="Use local WGS84 GeoJSON instead of official cached boundaries")
        child.add_argument("--grid-size", type=int, choices=(128, 256, 512), default=128)
        child.add_argument("--idw-power", type=float, default=2.0)
        child.add_argument("--neighbors", type=int, default=5)
        child.add_argument("--radius-km", type=float, default=None, help="Optional extra cutoff; thesis does not specify one")
    return parser


def _generate_ra(args, client, observations, directory):
    from typhoon_rainfall_realtime.ra import GridSpec, generate_ra, load_boundary
    boundary, info = load_boundary(client, args.boundary_file)
    return generate_ra(observations, directory / "ra", boundary, info,
                       grid=GridSpec(width=args.grid_size, height=args.grid_size),
                       power=args.idw_power, neighbors=args.neighbors, max_distance_km=args.radius_km)


def main(argv=None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(arguments or ["predict"])
    try:
        if args.command == "rainfall":
            from typhoon_rainfall_realtime.rainfall import collect_rainfall
            directory = new_run_dir(args.output_root, "rainfall")
            manifest = {"schema_version": 1, "operation": "rainfall", "status": "fetching"}
            write_manifest(directory, manifest)
            client = CwaClient()
            try:
                manifest["rainfall"] = collect_rainfall(client, directory, hours=args.hours)
                if not args.observations_only:
                    manifest["status"] = "interpolating"
                    write_manifest(directory, manifest)
                    manifest["ra"] = _generate_ra(args, client, manifest["rainfall"]["files"]["hourly_json"], directory)
                manifest["status"] = "completed"
            except Exception as exc:
                manifest.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
                raise
            finally:
                client.close()
                write_manifest(directory, manifest)
        elif args.command == "ra":
            directory = new_run_dir(args.output_root, "ra")
            manifest = {"schema_version": 1, "operation": "ra", "status": "interpolating", "mode": "saved_observations"}
            write_manifest(directory, manifest)
            client = CwaClient()
            try:
                manifest["ra"] = _generate_ra(args, client, args.observations, directory)
                manifest["status"] = "completed"
            except Exception as exc:
                manifest.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
                raise
            finally:
                client.close()
                write_manifest(directory, manifest)
        else:
            if args.max_age_minutes <= 0 or args.max_skew_minutes < 0:
                raise ValueError("max-age-minutes must be positive; max-skew-minutes must be nonnegative.")
            directory = run_images(output_root=args.output_root,
                                   config_path=getattr(args, "config", DEFAULT_CONFIG),
                                   fetch_only=args.command == "fetch",
                                   max_age_minutes=args.max_age_minutes,
                                   max_skew_minutes=args.max_skew_minutes)
        print(json.dumps({"output_dir": str(directory), "manifest": str(directory / "manifest.json")},
                         ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print("Live plugin failed (" + type(exc).__name__ + "): " + str(exc), file=sys.stderr)
        return 1
