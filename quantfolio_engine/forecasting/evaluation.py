"""
Evaluation utilities for forecasting models.

Provides train/test splitting, cross-validation, and forecast metrics.
"""

from typing import Dict, List, Optional, Tuple

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


class TimeSeriesSplit:
    """
    Time series cross-validation splitter.

    Provides train/test splits that respect temporal ordering.
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        gap: int = 0,
    ):
        """
        Initialize time series splitter.

        Args:
            n_splits: Number of splits
            test_size: Size of test set (if None, uses 1/n_splits of data)
            gap: Gap between train and test sets
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap

    def split(self, X: pd.Series) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits.

        Args:
            X: Time series (index used for splitting)

        Returns:
            List of (train_indices, test_indices) tuples
        """
        n = len(X)
        if self.test_size is None:
            test_size = max(1, n // (self.n_splits + 1))
        else:
            test_size = self.test_size

        splits = []
        for i in range(self.n_splits):
            # Calculate split point
            test_start = n - test_size * (self.n_splits - i) - self.gap
            test_end = test_start + test_size

            if test_start < 0:
                continue

            train_indices = np.arange(0, test_start)
            test_indices = np.arange(test_start, min(test_end, n))

            if len(train_indices) > 0 and len(test_indices) > 0:
                splits.append((train_indices, test_indices))

        return splits


class ForecastEvaluator:
    """
    Evaluator for forecasting models.

    Calculates metrics and performs diagnostics on forecasts.
    """

    def __init__(self):
        """Initialize evaluator."""
        self.metrics_history: List[Dict] = []

    def calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        y_lower: Optional[pd.Series] = None,
        y_upper: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """
        Calculate forecast accuracy metrics.

        Args:
            y_true: Actual values
            y_pred: Predicted values
            y_lower: Lower bound of prediction interval
            y_upper: Upper bound of prediction interval

        Returns:
            Dictionary with metrics
        """
        # Align indices
        common_idx = y_true.index.intersection(y_pred.index)
        y_true_aligned = y_true.loc[common_idx]
        y_pred_aligned = y_pred.loc[common_idx]

        metrics = {}

        # RMSE
        rmse = np.sqrt(mean_squared_error(y_true_aligned, y_pred_aligned))
        metrics["rmse"] = float(rmse)

        # MAE
        mae = mean_absolute_error(y_true_aligned, y_pred_aligned)
        metrics["mae"] = float(mae)

        # MAPE (Mean Absolute Percentage Error)
        non_zero_mask = y_true_aligned != 0
        if non_zero_mask.sum() > 0:
            mape = np.mean(
                np.abs(
                    (y_true_aligned[non_zero_mask] - y_pred_aligned[non_zero_mask])
                    / y_true_aligned[non_zero_mask]
                )
            )
            metrics["mape"] = float(mape * 100)  # As percentage
        else:
            metrics["mape"] = np.nan

        # Coverage (if intervals provided)
        if y_lower is not None and y_upper is not None:
            y_lower_aligned = y_lower.reindex(common_idx)
            y_upper_aligned = y_upper.reindex(common_idx)
            coverage = (
                (y_true_aligned >= y_lower_aligned)
                & (y_true_aligned <= y_upper_aligned)
            ).mean()
            metrics["coverage"] = float(coverage)

        # Directional accuracy
        if len(y_true_aligned) > 1:
            true_direction = np.sign(y_true_aligned.diff().dropna())
            pred_direction = np.sign(y_pred_aligned.diff().dropna())
            directional_accuracy = (true_direction == pred_direction).mean()
            metrics["directional_accuracy"] = float(directional_accuracy)
        else:
            metrics["directional_accuracy"] = np.nan

        return metrics

    def evaluate_model(
        self,
        model_results: Dict,
        y_test: pd.Series,
        forecast_df: pd.DataFrame,
    ) -> Dict[str, any]:
        """
        Evaluate a fitted model on test data.

        Args:
            model_results: Results from model fitting
            y_test: Test set actual values
            forecast_df: Forecast DataFrame with intervals

        Returns:
            Dictionary with evaluation results
        """
        # Calculate metrics
        metrics = self.calculate_metrics(
            y_test,
            forecast_df["forecast"],
            forecast_df.get("lower"),
            forecast_df.get("upper"),
        )

        # Add model diagnostics
        diagnostics = model_results.get("diagnostics", {})

        evaluation = {
            "metrics": metrics,
            "diagnostics": diagnostics,
            "model_info": {
                "aic": getattr(model_results.get("model"), "aic", None),
                "bic": getattr(model_results.get("model"), "bic", None),
            },
        }

        self.metrics_history.append(evaluation)
        return evaluation


def calculate_forecast_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_lower: Optional[pd.Series] = None,
    y_upper: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """
    Convenience function to calculate forecast metrics.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        y_lower: Lower bound of prediction interval
        y_upper: Upper bound of prediction interval

    Returns:
        Dictionary with metrics
    """
    evaluator = ForecastEvaluator()
    return evaluator.calculate_metrics(y_true, y_pred, y_lower, y_upper)


def train_test_split_time_series(
    y: pd.Series,
    test_size: float = 0.2,
    exog: Optional[pd.DataFrame] = None,
) -> Tuple[pd.Series, pd.Series, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Split time series into train and test sets.

    Args:
        y: Target time series
        test_size: Proportion of data for test set
        exog: Exogenous variables (optional)

    Returns:
        Tuple of (y_train, y_test, exog_train, exog_test)
    """
    n = len(y)
    split_idx = int(n * (1 - test_size))

    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    if exog is not None:
        exog_train = exog.iloc[:split_idx]
        exog_test = exog.iloc[split_idx:]
    else:
        exog_train = None
        exog_test = None

    return y_train, y_test, exog_train, exog_test
