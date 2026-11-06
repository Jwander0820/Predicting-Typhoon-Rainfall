import unittest

import numpy as np

from typhoon_rainfall_realtime.interpolation import idw_grid


class IdwGridTests(unittest.TestCase):
    def test_exact_station_and_all_colocated_gauges_override_neighbor_limit(self):
        result = idw_grid([121, 121, 122], [23, 23, 24], [0, 10, 100],
                          np.array([[121.0]]), np.array([[23.0]]), neighbors=1)
        np.testing.assert_array_equal(result, [[5.0]])
        self.assertEqual(result.dtype, np.dtype("float64"))

    def test_symmetric_gauges_have_arithmetic_mean(self):
        result = idw_grid([120.9, 121.1], [23, 23], [0, 20], [121], [23])
        np.testing.assert_allclose(result, [10.0], atol=1e-10)

    def test_equatorial_distances_give_hand_computed_inverse_square_weights(self):
        # Distances along the equator are in the ratio 1:2; weights are 4:1.
        result = idw_grid([-1, 2], [0, 0], [0, 30], [0], [0],
                          neighbors=2, max_distance_km=300)
        np.testing.assert_allclose(result, [6.0], atol=1e-12)

    def test_neighbor_count_changes_the_interpolation(self):
        nearest = idw_grid([-1, 2], [0, 0], [0, 30], [0], [0],
                           neighbors=1, max_distance_km=300)
        both = idw_grid([-1, 2], [0, 0], [0, 30], [0], [0],
                        neighbors=12, max_distance_km=300)
        np.testing.assert_array_equal(nearest, [0.0])
        np.testing.assert_allclose(both, [6.0])

    def test_zero_rain_is_valid_but_missing_negative_and_invalid_stations_are_excluded(self):
        result = idw_grid(
            [121, 121.1, np.nan, 121.2, 121.3, 400, 121.4],
            [23, 23, 23, 23, 23, 23, 91],
            [0, np.nan, 80, -1, np.inf, 100, 100],
            [121, 121.05], [23, 23],
        )
        np.testing.assert_array_equal(result, [0.0, 0.0])

    def test_radius_excludes_distant_stations_and_leaves_uncovered_cells_nan(self):
        result = idw_grid([121], [23], [5], [121.1, 122], [23, 23], max_distance_km=50)
        self.assertEqual(result[0], 5.0)
        self.assertTrue(np.isnan(result[1]))
        # The far wet station is beyond the radius; it must not dilute a dry gauge.
        dry = idw_grid([121, 122], [23, 23], [0, 200], [121.01], [23], max_distance_km=50)
        np.testing.assert_array_equal(dry, [0.0])

    def test_default_and_explicit_none_allow_distant_stations(self):
        default = idw_grid([0], [0], [7], [1], [0])
        unlimited = idw_grid([0], [0], [7], [1], [0], max_distance_km=None)
        limited = idw_grid([0], [0], [7], [1], [0], max_distance_km=50)
        np.testing.assert_array_equal(default, [7.0])
        np.testing.assert_array_equal(unlimited, [7.0])
        self.assertTrue(np.isnan(limited[0]))

    def test_default_five_neighbors_excludes_the_sixth_gauge(self):
        default = idw_grid([1, 2, 3, 4, 5, 6], [0] * 6, [0, 0, 0, 0, 0, 600], [0], [0])
        six = idw_grid([1, 2, 3, 4, 5, 6], [0] * 6, [0, 0, 0, 0, 0, 600], [0], [0], neighbors=6)
        np.testing.assert_array_equal(default, [0.0])
        self.assertGreater(six[0], 0)

    def test_haversine_handles_dateline_and_high_latitude(self):
        dateline = idw_grid([179.9, 170], [0, 0], [10, 50], [-179.9], [0], max_distance_km=50)
        np.testing.assert_array_equal(dateline, [10.0])
        polar = idw_grid([-1, 1], [89, 89], [10, 20], [0], [89], max_distance_km=5)
        np.testing.assert_allclose(polar, [15.0], atol=1e-10)

    def test_grid_shape_and_north_south_order_are_preserved(self):
        grid_lon, grid_lat = np.meshgrid([121, 121.1], [24, 23, 22])
        result = idw_grid([121, 121], [24, 22], [10, 30], grid_lon, grid_lat,
                          max_distance_km=300)
        self.assertEqual(result.shape, (3, 2))
        np.testing.assert_allclose(result[:, 0], [10, 20, 30], atol=1e-10)
        reversed_result = idw_grid([121, 121], [24, 22], [10, 30],
                                   grid_lon[::-1], grid_lat[::-1], max_distance_km=300)
        np.testing.assert_allclose(reversed_result, result[::-1])

    def test_large_grid_is_processed_without_reordering_between_batches(self):
        grid_lon = np.linspace(120.0, 120.6, 900).reshape(30, 30)
        grid_lat = np.full_like(grid_lon, 23.0)
        result = idw_grid([120.3], [23], [7], grid_lon, grid_lat)
        np.testing.assert_allclose(result, np.full((30, 30), 7.0))

    def test_finite_large_rainfall_does_not_overflow_weighted_mean(self):
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            exact = idw_grid([121, 121], [23, 23], [1e308, 1e308], [121], [23])
            weighted = idw_grid([120.9, 121.1], [23, 23], [1e308, 1e308], [121], [23])
        np.testing.assert_allclose(exact, [1e308])
        np.testing.assert_allclose(weighted, [1e308])

    def test_large_power_uses_stable_weights(self):
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            result = idw_grid([120.99, 121.1], [23, 23], [5, 30], [121], [23], power=1e10)
        np.testing.assert_array_equal(result, [5.0])

    def test_no_valid_stations_raises_instead_of_succeeding_with_all_nan(self):
        for lon, lat, rain in (([], [], []), ([121], [23], [np.nan]),
                               ([181], [23], [10]), ([121], [23], [-99])):
            with self.subTest(lon=lon, lat=lat, rain=rain):
                with self.assertRaisesRegex(ValueError, "No valid"):
                    idw_grid(lon, lat, rain, [121], [23])

    def test_invalid_parameter_values_are_rejected(self):
        for parameter, values in (
            ("power", [0, -1, np.nan, np.inf, True, "2"]),
            ("max_distance_km", [0, -1, np.nan, np.inf, False]),
            ("neighbors", [0, -1, 2.0, np.nan, True]),
        ):
            for value in values:
                with self.subTest(parameter=parameter, value=value):
                    with self.assertRaises(ValueError):
                        idw_grid([121], [23], [1], [121], [23], **{parameter: value})

    def test_station_and_grid_shapes_and_grid_coordinates_are_checked(self):
        with self.assertRaisesRegex(ValueError, "1D"):
            idw_grid([[121]], [23], [1], [121], [23])
        with self.assertRaisesRegex(ValueError, "1D"):
            idw_grid([121, 122], [23], [1], [121], [23])
        for grid_lon, grid_lat in (([], []), ([121, 122], [23]), (121, 23),
                                  ([np.nan], [23]), ([121], [np.inf]),
                                  ([181], [23]), ([121], [91])):
            with self.subTest(grid_lon=grid_lon, grid_lat=grid_lat):
                with self.assertRaises(ValueError):
                    idw_grid([121], [23], [1], grid_lon, grid_lat)


if __name__ == "__main__":
    unittest.main()
