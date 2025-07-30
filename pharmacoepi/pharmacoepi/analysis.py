"""Analysis helpers for pharmacoepidemiology studies."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.linear_model import LogisticRegression


def fit_propensity_score_model(
    df: pd.DataFrame,
    covariates: Iterable[str],
    treatment_col: str = "treatment",
) -> pd.Series:
    """Fit a logistic regression model for the propensity score.

    Parameters
    ----------
    df : pd.DataFrame
        Cohort dataset.
    covariates : Iterable[str]
        Column names to use as predictors.
    treatment_col : str
        Name of the treatment column.

    Returns
    -------
    pd.Series
        Estimated propensity scores.
    """
    model = LogisticRegression(max_iter=1000)
    X = df[list(covariates)]
    y = df[treatment_col]
    model.fit(X, y)
    ps = model.predict_proba(X)[:, 1]
    return pd.Series(ps, index=df.index, name="propensity_score")


def calculate_ip_weights(
    treatment: pd.Series,
    propensity: pd.Series,
) -> pd.Series:
    """Calculate inverse probability of treatment weights."""
    treated = treatment == 1
    weights = np.where(treated, 1 / propensity, 1 / (1 - propensity))
    return pd.Series(weights, index=treatment.index, name="ip_weight")


def fit_weighted_cox_model(
    df: pd.DataFrame,
    duration_col: str = "time",
    event_col: str = "event",
    treatment_col: str = "treatment",
    weight_col: str = "ip_weight",
    covariates: Iterable[str] | None = None,
) -> CoxPHFitter:
    """Fit a weighted Cox proportional hazards model."""
    cph = CoxPHFitter()
    covariate_cols = [treatment_col]
    if covariates:
        covariate_cols.extend(covariates)
    cph.fit(
        df[covariate_cols + [duration_col, event_col, weight_col]],
        duration_col=duration_col,
        event_col=event_col,
        weights_col=weight_col,
    )
    return cph
