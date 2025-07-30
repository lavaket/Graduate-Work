"""Data generation utilities for the pharmacoepidemiology project."""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_cohort(
    n: int = 1000,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate a cohort dataset for a pharmacoepidemiology study.

    Parameters
    ----------
    n : int
        Number of subjects in the cohort.
    seed : int | None
        Optional random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Simulated cohort with baseline covariates, treatment indicator,
        follow-up time, and event indicator.
    """
    rng = np.random.default_rng(seed)

    age = rng.normal(loc=50, scale=10, size=n)
    sex = rng.integers(0, 2, size=n)  # 0 = female, 1 = male

    # baseline probability of treatment
    p_treat = 1 / (1 + np.exp(-(-3 + 0.05 * age + 0.3 * sex)))
    treatment = rng.binomial(1, p_treat, size=n)

    # baseline hazard components
    base_hazard = np.exp(-7 + 0.06 * age + 0.5 * sex)
    treat_effect = 0.7  # hazard ratio <1 indicates protective effect
    hazard = base_hazard * np.where(treatment == 1, treat_effect, 1.0)

    # simulate event times using exponential distribution
    time_to_event = rng.exponential(1 / hazard)
    censor_time = rng.uniform(0.5, 3.0, size=n)

    time = np.minimum(time_to_event, censor_time)
    event = (time_to_event <= censor_time).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "sex": sex,
            "treatment": treatment,
            "time": time,
            "event": event,
        }
    )
    return df
