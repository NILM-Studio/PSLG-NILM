import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.prepare_ukdale_nilm_pair import nearest_alignment, prepare


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


if __name__ == "__main__":
    unittest.main()
