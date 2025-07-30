import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from pharmacoepi.data import simulate_cohort


def test_simulate_cohort_shapes() -> None:
    df = simulate_cohort(n=100, seed=1)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 100
    required_cols = {"age", "sex", "treatment", "time", "event"}
    assert required_cols.issubset(df.columns)
