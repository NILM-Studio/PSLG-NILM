import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.prepare_ukdale_nilm_pair import load_power_series, nearest_alignment, prepare


class PrepareUkdaleNilmPairTests(unittest.TestCase):
    def test_nearest_alignment_uses_offset_grid_and_rejects_gap(self):
        mains = np.array([95, 101, 107, 125], dtype=np.int64)
        appliance = np.array([98, 104, 110, 116], dtype=np.int64)
        nearest, valid = nearest_alignment(mains, appliance, 3.1)
        self.assertEqual(nearest.tolist(), [0, 1, 2, 2])
        self.assertEqual(valid.tolist(), [True, True, True, False])

    def test_prepare_writes_aligned_csv_and_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mains = root / "channel_1.dat"
            appliance = root / "appliance.csv"
            output = root / "pair.csv"
            mains.write_text("95 599\n101 582\n107 600\n", encoding="ascii")
            pd.DataFrame({
                "timestamp": [98, 104, 110, 120],
                "power": [0, 20, 100, 0],
            }).to_csv(appliance, index=False)

            audit = prepare(mains, appliance, output, 3.1, chunksize=2)
            pair = pd.read_csv(output)
            self.assertEqual(pair["timestamp"].tolist(), [98, 104, 110])
            self.assertEqual(pair["mains"].tolist(), [599, 582, 600])
            self.assertEqual(audit["matched_rows"], 3)
            self.assertEqual(audit["unmatched_rows"], 1)
            with open(str(output) + ".audit.json", encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual(saved["alignment_method"],
                             "nearest_mains_to_appliance_grid")
            self.assertEqual(saved["mains_power_type"], "unknown")
            self.assertFalse(saved["measurement_compatible_for_additive_synthesis"])
            self.assertEqual(saved["interpolated_rows"], 0)

    def test_four_column_mains_preserves_fractional_timestamp_and_active_power(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mains, appliance, output = root / "mains.dat", root / "channel_5.dat", root / "pair.csv"
            mains.write_text("100.4 450.25 600.50 241.00\n101.4 460.50 610.75 242.00\n")
            appliance.write_text("100 100\n101 110\n")
            audit = prepare(mains, appliance, output, 0.5, chunksize=1,
                            mains_format="ukdale-mains", appliance_power_type="active")
            pair = pd.read_csv(output)
            np.testing.assert_allclose(pair["mains"], [450.25, 460.50])
            self.assertAlmostEqual(audit["timestamp_offset_seconds"]["min"], 0.4)
            self.assertEqual(audit["mains_input"]["timestamp_column"], 0)
            self.assertEqual(audit["mains_input"]["power_column"], 1)
            self.assertEqual(audit["mains_power_type"], "active")
            self.assertTrue(audit["measurement_compatible_for_additive_synthesis"])
            self.assertEqual(audit["mains_input"]["valid_rows"], 2)
            timestamps, _ = load_power_series(mains)
            np.testing.assert_allclose(timestamps, [100.4, 101.4])

    def test_apparent_selection_and_unknown_appliance_do_not_allow_additive_synthesis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mains, appliance, output = root / "mains.dat", root / "channel_5.dat", root / "pair.csv"
            mains.write_text("100.4 450.25 600.50 241.00\n")
            appliance.write_text("100 100\n")
            audit = prepare(mains, appliance, output, 0.5,
                            mains_power_type="apparent", appliance_power_type="active")
            self.assertEqual(pd.read_csv(output)["mains"].tolist(), [600.5])
            self.assertEqual(audit["mains_input"]["power_unit"], "VA")
            self.assertFalse(audit["measurement_compatible_for_additive_synthesis"])
            unknown_appliance = prepare(mains, appliance, output, 0.5)
            self.assertEqual(unknown_appliance["mains_power_type"], "active")
            self.assertFalse(unknown_appliance["measurement_compatible_for_additive_synthesis"])

    def test_two_column_explicit_active_declaration_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mains, appliance = root / "active.dat", root / "channel_5.dat"
            mains.write_text("100 500\n")
            appliance.write_text("100 100\n")
            audit = prepare(mains, appliance, root / "pair.csv", 0,
                            mains_power_type="active", appliance_power_type="active")
            self.assertTrue(audit["measurement_compatible_for_additive_synthesis"])
            self.assertEqual(audit["mains_input"]["measurement_type_source"], "explicit_argument")

    def test_requested_interval_is_half_open_and_audits_machine_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mains, appliance, output = root / "mains.dat", root / "appliance.csv", root / "pair.csv"
            stamps = [int(pd.Timestamp(day, tz="UTC").timestamp())
                      for day in ["2015-09-06", "2015-09-07", "2015-09-08", "2015-09-09"]]
            mains.write_text("".join(f"{stamp} 500 600 240\n" for stamp in stamps))
            pd.DataFrame({"timestamp": stamps, "power": [0, 100, 200, 0]}).to_csv(appliance, index=False)
            audit = prepare(mains, appliance, output, 0, chunksize=1,
                            start="2015-09-07", end="2015-09-09",
                            instance_boundary="2015-09-08", appliance_power_type="active")
            self.assertEqual(pd.read_csv(output)["timestamp"].tolist(), stamps[1:3])
            self.assertEqual(audit["mains_input"]["rows_read"], 4)
            self.assertEqual(audit["mains_input"]["range_excluded_rows"], 2)
            self.assertEqual(audit["output_time_range"]["start_unix"], stamps[1])
            self.assertEqual(audit["output_time_range"]["end_unix"], stamps[2])
            self.assertEqual(audit["instance_periods"]["before_boundary_rows"], 1)
            self.assertEqual(audit["instance_periods"]["at_or_after_boundary_rows"], 1)
            self.assertEqual(audit["requested_time_range"]["timezone_for_naive_dates"], "UTC")

    def test_invalid_format_nonfinite_and_duplicate_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.dat"
            for contents, message in [
                    ("100 500 240\n", "expected 2 channel or 4"),
                    ("100 500\n100 550\n", "strictly increasing"),
                    ("100 500\n101 nan\n", "NaN or Inf"),
                    ("100 500\ninf 550\n", "NaN or Inf")]:
                with self.subTest(contents=contents):
                    path.write_text(contents)
                    # chunksize=1 checks monotonicity at chunk boundaries too.
                    appliance = Path(directory) / "appliance.dat"
                    appliance.write_text("100 100\n")
                    with self.assertRaisesRegex(ValueError, message):
                        prepare(path, appliance, Path(directory) / "out.csv", 3.1, chunksize=1)

    def test_empty_or_reversed_time_interval_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mains, appliance = root / "mains.dat", root / "appliance.dat"
            mains.write_text("100 500\n")
            appliance.write_text("100 100\n")
            with self.assertRaisesRegex(ValueError, "start must be earlier"):
                prepare(mains, appliance, root / "out.csv", 3.1,
                        start="2015-09-09", end="2015-09-08")
            with self.assertRaisesRegex(ValueError, "empty power series in requested"):
                prepare(mains, appliance, root / "out.csv", 3.1, start="2015-09-08")
            self.assertFalse((root / "out.csv").exists())

    def test_nonfinite_tolerance_is_rejected(self):
        for tolerance in [float("nan"), float("inf"), -1]:
            with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                nearest_alignment(np.array([100.0]), np.array([100.4]), tolerance)


if __name__ == "__main__":
    unittest.main()
