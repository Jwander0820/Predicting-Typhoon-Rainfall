from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from typhoon_rainfall_realtime.geometry import GridSpec
from typhoon_rainfall_realtime.presentation import save_forecast_view


TZ = timezone(timedelta(hours=8))
START = datetime(2026, 12, 31, 23, tzinfo=TZ)
END = START + timedelta(hours=1)
FUTURE = END + timedelta(hours=1)


class ForecastViewTests(unittest.TestCase):
    def test_saves_view_with_missing_observation_and_capped_forecast_across_new_year(self):
        grid = GridSpec(width=2, height=2)
        observed = np.array([[np.nan, 1.4], [0, np.nan]])
        predicted = np.array([[100, 2], [0, 0]])
        mask = np.array([[True, True], [True, False]])
        with tempfile.TemporaryDirectory() as temporary:
            name = save_forecast_view(observed, predicted, mask, grid, temporary,
                                      START, END, FUTURE)
            with Image.open(Path(temporary) / name) as image:
                self.assertEqual(image.format, "PNG")
                self.assertGreater(image.width, image.height)
                self.assertGreater(image.height, 500)
        import matplotlib.pyplot as plt
        self.assertFalse(plt.get_fignums())

    def test_refuses_nonconsecutive_or_naive_intervals(self):
        grid = GridSpec(width=2, height=2)
        field = np.zeros((2, 2))
        mask = np.ones((2, 2), dtype=bool)
        for start, end, future in ((START, END, END),
                                    (START, END, FUTURE + timedelta(hours=1)),
                                    (START.replace(tzinfo=None), END, FUTURE)):
            with self.subTest(start=start, end=end, future=future):
                with self.assertRaises(ValueError):
                    save_forecast_view(field, field, mask, grid, ".", start, end, future)

    def test_refuses_invalid_prediction_labels_instead_of_rendering_them_as_rain(self):
        grid = GridSpec(width=2, height=2)
        field = np.zeros((2, 2))
        mask = np.ones((2, 2), dtype=bool)
        for value in (-1, 0.5, 101, 255, np.nan):
            with self.subTest(value=value):
                predicted = field.copy()
                predicted[0, 0] = value
                with self.assertRaisesRegex(ValueError, "integers in 0..100"):
                    save_forecast_view(field, predicted, mask, grid, ".", START, END, FUTURE)


if __name__ == "__main__":
    unittest.main()
