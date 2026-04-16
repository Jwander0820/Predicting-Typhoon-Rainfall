import tempfile
import unittest
from pathlib import Path

from typhoon_rainfall.data.contracts import InputModes, compute_input_shape, parse_annotation_file


class ContractTests(unittest.TestCase):
    def test_compute_input_shape_for_rd_ir(self):
        shape = compute_input_shape((128, 128), InputModes(use_rd=True, use_ir=True, use_ra=False, use_gi=False))
        self.assertEqual(shape, (128, 128, 4))

    def test_compute_input_shape_for_ir_only(self):
        shape = compute_input_shape((128, 128), InputModes(use_rd=False, use_ir=True, use_ra=False, use_gi=False))
        self.assertEqual(shape, (128, 128, 3))

    def test_parse_annotation_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            list_path = Path(tmp_dir) / "train.txt"
            list_path.write_text("a.png;b.png\nc.png;d.png\n", encoding="utf-8")
            pairs = parse_annotation_file(list_path)
            self.assertEqual(len(pairs), 2)
            self.assertEqual(pairs[0].prediction_stem(1), "a_t+1")


if __name__ == "__main__":
    unittest.main()
