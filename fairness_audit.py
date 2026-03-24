from typing import Dict, Tuple

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
)
from sklearn.metrics import precision_score, recall_score


def compute_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_feature: pd.Series,
) -> MetricFrame:
    return MetricFrame(
        metrics={
            "precision": precision_score,
            "recall": recall_score,
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_feature,
    )


def fairness_summary(
    df_test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_column: str,
) -> Tuple[Dict, Dict]:
    sensitive = df_test[sensitive_column]

    frame = compute_group_metrics(y_true=y_true, y_pred=y_pred, sensitive_feature=sensitive)

    dp_diff = demographic_parity_difference(y_true=y_true, y_pred=y_pred, sensitive_features=sensitive)
    eo_diff = equalized_odds_difference(y_true=y_true, y_pred=y_pred, sensitive_features=sensitive)

    group_metrics = frame.by_group.to_dict()

    fairness = {
        "demographic_parity_difference": float(dp_diff),
        "equalized_odds_difference": float(eo_diff),
    }

    return group_metrics, fairness

