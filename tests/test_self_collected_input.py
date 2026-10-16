import numpy as np
import pandas as pd
from src.steps.extract_active_data_step import ExtractActiveDataStep


def test_self_collected_uses_power_and_seconds(tmp_path):
    path = tmp_path / 'appliance.csv'
    pd.DataFrame({'datetime': ['2025-08-04 00:00:00', '2025-08-04 00:00:01'],
                  'timestamp': [1754236800026000, 1754236801026000],
                  'active power': [28.0, 18.0]}).to_csv(path, index=False)
    timestamps, power = ExtractActiveDataStep()._read_data(str(path))
    np.testing.assert_allclose(np.diff(timestamps), [1.0])
    np.testing.assert_allclose(power, [28.0, 18.0])
    assert 1e9 < timestamps[0] < 2e9
