import json
import tempfile
import unittest
from pathlib import Path
from typhoon_rainfall_realtime.pipeline import compact_prediction_output
from typhoon_rainfall_realtime.cli import build_parser


class CompactOutputTests(unittest.TestCase):
    def test_default_and_detailed_cli(self):
        parser = build_parser()
        self.assertFalse(parser.parse_args(["predict"]).detailed_output)
        self.assertTrue(parser.parse_args(["predict", "--detailed-output"]).detailed_output)

    def test_retains_inputs_and_visuals_without_dangling_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "run"
            run.mkdir()
            sibling = root / "previous.txt"
            sibling.write_text("keep")
            (run / "inputs").mkdir()
            (run / "inputs" / "RA.png").write_bytes(b"original labels")
            (run / "ra").mkdir()
            (run / "ra" / "RA_map.png").write_bytes(b"colored map")
            for name in ("forecast.png", "prediction_mm.csv", "alignment.png", "overview.png", "prediction.csv"):
                (run / name).write_bytes(b"result")
            manifest = dict(schema_version=1, operation="predict", status="completed",
                            created_at="start", finished_at="end", prediction={}, sources={},
                            model=dict(name="Unet3", modalities=["RA"], checkpoint_contract=dict(sha256="hash")),
                            ra=dict(maps=[dict(directory=str(run / "ra"))]),
                            artifacts=dict(forecast="forecast.png", rainfall_mm_csv="prediction_mm.csv",
                                           overview="overview.png", csv="prediction.csv"))
            compact_prediction_output(run, manifest)
            self.assertEqual((run / "inputs" / "RA.png").read_bytes(), b"original labels")
            self.assertEqual((run / "observed_ra.png").read_bytes(), b"colored map")
            self.assertFalse((run / "ra").exists())
            self.assertFalse((run / "overview.png").exists())
            self.assertEqual(sibling.read_text(), "keep")
            summary = json.loads((run / "manifest.json").read_text())
            for value in summary["artifacts"].values():
                self.assertTrue((run / value).exists())
