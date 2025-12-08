"""
Forecasting module for QuantFolio Engine.

This module implements time series forecasting models for portfolio returns:
- TSLM (Time Series Linear Model) with covariates
- ARIMA (AutoRegressive Integrated Moving Average)
- ARIMA-with-errors (Regression with ARIMA errors)

Designed for DS 5740 Advanced Stats for DS forecasting assignment.
"""

from quantfolio_engine.forecasting.evaluation import (
    ForecastEvaluator,
    TimeSeriesSplit,
    calculate_forecast_metrics,
)
from quantfolio_engine.forecasting.models import ForecastingEngine
from quantfolio_engine.forecasting.visualization import (
    plot_diagnostics,
    plot_forecasts,
    plot_model_comparison,
)

__all__ = [
    "ForecastingEngine",
    "ForecastEvaluator",
    "TimeSeriesSplit",
    "calculate_forecast_metrics",
    "plot_forecasts",
    "plot_diagnostics",
    "plot_model_comparison",
]
