"""
Forecasting models for time series analysis.

Implements TSLM, ARIMA, and ARIMA-with-errors models for portfolio return forecasting.
"""

from typing import Dict, Optional, Tuple, Union

from loguru import logger
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

from quantfolio_engine.config import PROCESSED_DATA_DIR


class ForecastingEngine:
    """
    Main forecasting engine for time series analysis.

    Supports three model types:
    1. TSLM: Time Series Linear Model with trend, seasonality, and covariates
    2. ARIMA: Standard AutoRegressive Integrated Moving Average
    3. ARIMA-with-errors: Regression with ARIMA errors (covariates + ARIMA residuals)

    Args:
        debug: If True, sets logger to DEBUG level
    """

    def __init__(self, debug: bool = False):
        """Initialize forecasting engine."""
        if debug:
            from loguru import logger

            logger.remove()
            logger.add(lambda msg: print(msg, end=""), level="DEBUG", colorize=True)

        self.models: Dict[str, any] = {}
        self.fitted_models: Dict[str, any] = {}
        self.forecast_results: Dict[str, pd.DataFrame] = {}

    def fit_tslm(
        self,
        y: pd.Series,
        exog: Optional[pd.DataFrame] = None,
        trend: str = "c",
        seasonal: bool = True,
        period: Optional[int] = None,
    ) -> Dict[str, any]:
        """
        Fit Time Series Linear Model (TSLM).

        TSLM models the time series as:
        y_t = trend + seasonality + β * covariates + ε_t

        Args:
            y: Target time series (portfolio returns)
            exog: Exogenous variables (covariates) as DataFrame
            trend: Trend component ('c' for constant, 't' for linear, 'ct' for both)
            seasonal: Whether to include seasonal components
            period: Seasonal period (auto-detected if None)

        Returns:
            Dictionary with fitted model and diagnostics
        """
        logger.info("Fitting TSLM model...")

        # Prepare data
        y_clean = y.dropna()
        if len(y_clean) < 20:
            raise ValueError(f"Insufficient data for TSLM: {len(y_clean)} < 20")

        # Auto-detect seasonal period if not provided
        if seasonal and period is None:
            # Try to detect period from data frequency
            freq = pd.infer_freq(y_clean.index)
            if freq == "M" or freq.startswith("M"):
                period = 12  # Monthly data -> annual seasonality
            elif freq == "Q" or freq.startswith("Q"):
                period = 4  # Quarterly data -> annual seasonality
            else:
                period = 12  # Default to 12

        # Create design matrix
        n = len(y_clean)
        X_list = []

        # Add trend components
        if "c" in trend:
            X_list.append(pd.Series(1.0, index=y_clean.index, name="const"))
        if "t" in trend:
            X_list.append(pd.Series(range(n), index=y_clean.index, name="trend"))

        # Add seasonal components (Fourier terms)
        if seasonal and period > 1:
            for k in range(1, min(period // 2 + 1, 6)):  # Limit to 6 harmonics
                t = np.arange(n)
                X_list.append(
                    pd.Series(
                        np.sin(2 * np.pi * k * t / period),
                        index=y_clean.index,
                        name=f"sin_{k}",
                    )
                )
                X_list.append(
                    pd.Series(
                        np.cos(2 * np.pi * k * t / period),
                        index=y_clean.index,
                        name=f"cos_{k}",
                    )
                )

        # Add exogenous variables
        if exog is not None:
            exog_aligned = exog.reindex(y_clean.index).fillna(method="ffill").fillna(0)
            X_list.append(exog_aligned)

        # Combine design matrix
        if X_list:
            X = pd.concat(X_list, axis=1)
        else:
            X = pd.DataFrame({"const": 1.0}, index=y_clean.index)

        # Fit OLS model
        model = sm.OLS(y_clean, X, missing="drop")
        results = model.fit()

        # Store model
        model_key = "tslm"
        self.fitted_models[model_key] = {
            "model": results,
            "y": y_clean,
            "X": X,
            "exog": exog,
            "trend": trend,
            "seasonal": seasonal,
            "period": period,
        }

        # Calculate diagnostics
        residuals = results.resid
        diagnostics = self._calculate_diagnostics(residuals, model_key)

        logger.success(f"TSLM fitted: R² = {results.rsquared:.4f}")

        return {
            "model": results,
            "diagnostics": diagnostics,
            "residuals": residuals,
            "fitted_values": results.fittedvalues,
        }

    def fit_arima(
        self,
        y: pd.Series,
        order: Optional[Tuple[int, int, int]] = None,
        auto_select: bool = True,
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5,
        ic: str = "aic",
    ) -> Dict[str, any]:
        """
        Fit ARIMA model.

        ARIMA(p,d,q) models the time series as:
        (1 - φ₁B - ... - φₚBᵖ)(1 - B)ᵈy_t = (1 + θ₁B + ... + θₑBᵑ)ε_t

        Args:
            y: Target time series (portfolio returns)
            order: ARIMA order (p, d, q). If None, auto-selects.
            auto_select: Whether to auto-select order using information criterion
            max_p: Maximum AR order for auto-selection
            max_d: Maximum differencing order for auto-selection
            max_q: Maximum MA order for auto-selection
            ic: Information criterion for auto-selection ('aic', 'bic', 'hqic')

        Returns:
            Dictionary with fitted model and diagnostics
        """
        logger.info("Fitting ARIMA model...")

        y_clean = y.dropna()
        if len(y_clean) < 20:
            raise ValueError(f"Insufficient data for ARIMA: {len(y_clean)} < 20")

        # Auto-select order if requested
        if order is None and auto_select:
            logger.info("Auto-selecting ARIMA order...")
            order = self._auto_select_arima_order(y_clean, max_p, max_d, max_q, ic)
            logger.info(f"Selected ARIMA order: {order}")

        if order is None:
            order = (1, 0, 1)  # Default fallback

        # Fit ARIMA model
        try:
            model = ARIMA(y_clean, order=order)
            results = model.fit()

            # Store model
            model_key = "arima"
            self.fitted_models[model_key] = {
                "model": results,
                "y": y_clean,
                "order": order,
            }

            # Calculate diagnostics
            residuals = results.resid
            diagnostics = self._calculate_diagnostics(residuals, model_key)

            logger.success(f"ARIMA{order} fitted: AIC = {results.aic:.2f}")

            return {
                "model": results,
                "diagnostics": diagnostics,
                "residuals": residuals,
                "fitted_values": results.fittedvalues,
                "order": order,
            }

        except Exception as e:
            logger.error(f"ARIMA fitting failed: {e}")
            raise

    def fit_arima_errors(
        self,
        y: pd.Series,
        exog: pd.DataFrame,
        arima_order: Optional[Tuple[int, int, int]] = None,
        auto_select: bool = True,
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5,
        ic: str = "aic",
    ) -> Dict[str, any]:
        """
        Fit ARIMA-with-errors model (Regression with ARIMA errors).

        This model combines regression with ARIMA:
        y_t = β * covariates_t + u_t
        where u_t follows ARIMA(p,d,q)

        Args:
            y: Target time series (portfolio returns)
            exog: Exogenous variables (covariates) as DataFrame
            arima_order: ARIMA order for error term. If None, auto-selects.
            auto_select: Whether to auto-select ARIMA order
            max_p: Maximum AR order for auto-selection
            max_d: Maximum differencing order for auto-selection
            max_q: Maximum MA order for auto-selection
            ic: Information criterion for auto-selection

        Returns:
            Dictionary with fitted model and diagnostics
        """
        logger.info("Fitting ARIMA-with-errors model...")

        y_clean = y.dropna()
        exog_aligned = exog.reindex(y_clean.index).fillna(method="ffill").fillna(0)

        if len(y_clean) < 20:
            raise ValueError(
                f"Insufficient data for ARIMA-with-errors: {len(y_clean)} < 20"
            )

        if exog_aligned.empty:
            raise ValueError("No exogenous variables provided for ARIMA-with-errors")

        # Fit regression first to get residuals
        X_with_const = sm.add_constant(exog_aligned)
        reg_model = sm.OLS(y_clean, X_with_const, missing="drop")
        reg_results = reg_model.fit()
        residuals = reg_results.resid

        # Auto-select ARIMA order for residuals if requested
        if arima_order is None and auto_select:
            logger.info("Auto-selecting ARIMA order for residuals...")
            arima_order = self._auto_select_arima_order(
                residuals, max_p, max_d, max_q, ic
            )
            logger.info(f"Selected ARIMA order for residuals: {arima_order}")

        if arima_order is None:
            arima_order = (1, 0, 1)  # Default fallback

        # Fit ARIMA model with exogenous variables
        try:
            model = ARIMA(y_clean, exog=exog_aligned, order=arima_order)
            results = model.fit()

            # Store model
            model_key = "arima_errors"
            self.fitted_models[model_key] = {
                "model": results,
                "y": y_clean,
                "exog": exog_aligned,
                "order": arima_order,
            }

            # Calculate diagnostics
            model_residuals = results.resid
            diagnostics = self._calculate_diagnostics(model_residuals, model_key)

            logger.success(
                f"ARIMA-with-errors{arima_order} fitted: AIC = {results.aic:.2f}"
            )

            return {
                "model": results,
                "diagnostics": diagnostics,
                "residuals": model_residuals,
                "fitted_values": results.fittedvalues,
                "order": arima_order,
            }

        except Exception as e:
            logger.error(f"ARIMA-with-errors fitting failed: {e}")
            raise

    def forecast(
        self,
        model_key: str,
        steps: int,
        exog: Optional[pd.DataFrame] = None,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """
        Generate forecasts with prediction intervals.

        Args:
            model_key: Key of fitted model ('tslm', 'arima', 'arima_errors')
            steps: Number of steps ahead to forecast
            exog: Exogenous variables for future periods (required for TSLM/ARIMA-errors)
            alpha: Significance level for prediction intervals (default 0.05 = 95% CI)

        Returns:
            DataFrame with columns: ['forecast', 'lower', 'upper', 'lower_80', 'upper_80']
        """
        if model_key not in self.fitted_models:
            raise ValueError(
                f"Model '{model_key}' not fitted. Available: {list(self.fitted_models.keys())}"
            )

        logger.info(f"Generating {steps}-step forecast using {model_key}...")

        model_info = self.fitted_models[model_key]
        model = model_info["model"]

        # Generate forecast dates
        last_date = model_info["y"].index[-1]
        if isinstance(last_date, pd.Timestamp):
            freq = pd.infer_freq(model_info["y"].index) or "M"
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1), periods=steps, freq=freq
            )
        else:
            forecast_dates = pd.RangeIndex(
                start=len(model_info["y"]), stop=len(model_info["y"]) + steps
            )

        # Generate forecasts based on model type
        if model_key == "tslm":
            forecast_result = self._forecast_tslm(
                model, model_info, steps, exog, alpha, forecast_dates
            )
        elif model_key == "arima":
            forecast_result = self._forecast_arima(
                model, model_info, steps, alpha, forecast_dates
            )
        elif model_key == "arima_errors":
            forecast_result = self._forecast_arima_errors(
                model, model_info, steps, exog, alpha, forecast_dates
            )
        else:
            raise ValueError(f"Unknown model key: {model_key}")

        # Store forecast results
        self.forecast_results[model_key] = forecast_result

        return forecast_result

    def _forecast_tslm(
        self,
        model,
        model_info: Dict,
        steps: int,
        exog: Optional[pd.DataFrame],
        alpha: float,
        forecast_dates: pd.Index,
    ) -> pd.DataFrame:
        """Generate TSLM forecast."""
        # Build future X matrix
        n_historical = len(model_info["y"])
        X_future_list = []

        # Trend components
        if "c" in model_info["trend"]:
            X_future_list.append(pd.Series(1.0, index=forecast_dates, name="const"))
        if "t" in model_info["trend"]:
            X_future_list.append(
                pd.Series(
                    range(n_historical, n_historical + steps),
                    index=forecast_dates,
                    name="trend",
                )
            )

        # Seasonal components
        if model_info["seasonal"] and model_info["period"]:
            period = model_info["period"]
            for k in range(1, min(period // 2 + 1, 6)):
                t = np.arange(n_historical, n_historical + steps)
                X_future_list.append(
                    pd.Series(
                        np.sin(2 * np.pi * k * t / period),
                        index=forecast_dates,
                        name=f"sin_{k}",
                    )
                )
                X_future_list.append(
                    pd.Series(
                        np.cos(2 * np.pi * k * t / period),
                        index=forecast_dates,
                        name=f"cos_{k}",
                    )
                )

        # Exogenous variables
        if exog is not None:
            exog_future = exog.reindex(forecast_dates).fillna(method="ffill").fillna(0)
            X_future_list.append(exog_future)
        elif model_info["exog"] is not None:
            # Use last values if exog not provided
            last_exog = model_info["exog"].iloc[-1:]
            exog_future = pd.concat([last_exog] * steps, ignore_index=True)
            exog_future.index = forecast_dates
            X_future_list.append(exog_future)

        X_future = pd.concat(X_future_list, axis=1)

        # Generate forecast
        forecast = model.predict(X_future)
        forecast_se = model.get_prediction(X_future).se_mean

        # Calculate prediction intervals
        from scipy import stats

        z_score = stats.norm.ppf(1 - alpha / 2)
        z_score_80 = stats.norm.ppf(0.9)  # 80% interval

        lower = forecast - z_score * forecast_se
        upper = forecast + z_score * forecast_se
        lower_80 = forecast - z_score_80 * forecast_se
        upper_80 = forecast + z_score_80 * forecast_se

        return pd.DataFrame(
            {
                "forecast": forecast,
                "lower": lower,
                "upper": upper,
                "lower_80": lower_80,
                "upper_80": upper_80,
            },
            index=forecast_dates,
        )

    def _forecast_arima(
        self,
        model,
        model_info: Dict,
        steps: int,
        alpha: float,
        forecast_dates: pd.Index,
    ) -> pd.DataFrame:
        """Generate ARIMA forecast."""
        # Use get_forecast() to get ForecastResults with confidence intervals
        forecast_result = model.get_forecast(steps=steps)
        forecast = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=alpha)

        # Calculate 80% intervals
        forecast_result_80 = model.get_forecast(steps=steps)
        conf_int_80 = forecast_result_80.conf_int(alpha=0.2)

        return pd.DataFrame(
            {
                "forecast": forecast.values,
                "lower": conf_int.iloc[:, 0].values,
                "upper": conf_int.iloc[:, 1].values,
                "lower_80": conf_int_80.iloc[:, 0].values,
                "upper_80": conf_int_80.iloc[:, 1].values,
            },
            index=forecast_dates,
        )

    def _forecast_arima_errors(
        self,
        model,
        model_info: Dict,
        steps: int,
        exog: Optional[pd.DataFrame],
        alpha: float,
        forecast_dates: pd.Index,
    ) -> pd.DataFrame:
        """Generate ARIMA-with-errors forecast."""
        if exog is None:
            raise ValueError(
                "Exogenous variables required for ARIMA-with-errors forecast"
            )

        exog_future = exog.reindex(forecast_dates).fillna(method="ffill").fillna(0)

        # Get forecast with confidence intervals
        forecast_result = model.get_forecast(steps=steps, exog=exog_future)
        forecast = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=alpha)

        # Calculate 80% intervals
        forecast_result_80 = model.get_forecast(steps=steps, exog=exog_future)
        conf_int_80 = forecast_result_80.conf_int(alpha=0.2)

        return pd.DataFrame(
            {
                "forecast": forecast.values,
                "lower": conf_int.iloc[:, 0].values,
                "upper": conf_int.iloc[:, 1].values,
                "lower_80": conf_int_80.iloc[:, 0].values,
                "upper_80": conf_int_80.iloc[:, 1].values,
            },
            index=forecast_dates,
        )

    def _auto_select_arima_order(
        self,
        y: pd.Series,
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5,
        ic: str = "aic",
    ) -> Tuple[int, int, int]:
        """
        Auto-select ARIMA order using information criterion.

        Prefers non-trivial models (excludes (0,0,0) unless significantly better)
        to avoid constant forecasts.

        Args:
            y: Time series
            max_p: Maximum AR order
            max_d: Maximum differencing order
            max_q: Maximum MA order
            ic: Information criterion ('aic', 'bic', 'hqic')

        Returns:
            Best (p, d, q) order
        """
        best_ic_non_trivial = np.inf
        best_order_non_trivial = (1, 0, 1)
        best_ic_000 = np.inf  # Track (0,0,0) separately

        # First pass: find best non-trivial model and (0,0,0)
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(y, order=(p, d, q))
                        results = model.fit()

                        if ic == "aic":
                            current_ic = results.aic
                        elif ic == "bic":
                            current_ic = results.bic
                        elif ic == "hqic":
                            current_ic = results.hqic
                        else:
                            current_ic = results.aic

                        # Track (0,0,0) separately
                        if (p, d, q) == (0, 0, 0):
                            best_ic_000 = current_ic
                        else:
                            # Track best non-trivial model
                            if current_ic < best_ic_non_trivial:
                                best_ic_non_trivial = current_ic
                                best_order_non_trivial = (p, d, q)

                    except Exception:
                        continue

        # Decision: use (0,0,0) only if it's significantly better (2+ IC units)
        if best_ic_000 < np.inf and best_ic_non_trivial < np.inf:
            ic_diff = best_ic_non_trivial - best_ic_000
            if ic_diff >= 2:
                logger.info(
                    f"ARIMA(0,0,0) is significantly better ({ic_diff:.2f} IC units). "
                    f"Using (0,0,0) - forecasts will be constant (mean)."
                )
                return (0, 0, 0)
            else:
                logger.info(
                    f"ARIMA(0,0,0) has better {ic.upper()} ({best_ic_000:.2f} vs {best_ic_non_trivial:.2f}), "
                    f"but difference ({ic_diff:.2f}) < 2. Using {best_order_non_trivial} to avoid constant forecasts."
                )
                return best_order_non_trivial
        elif best_ic_000 < np.inf:
            logger.warning(
                "Only ARIMA(0,0,0) converged. Using (0,0,0) but forecasts will be constant."
            )
            return (0, 0, 0)
        else:
            return best_order_non_trivial

    def _calculate_diagnostics(
        self, residuals: pd.Series, model_key: str
    ) -> Dict[str, any]:
        """
        Calculate diagnostic statistics for residuals.

        Args:
            residuals: Model residuals
            model_key: Model identifier

        Returns:
            Dictionary with diagnostic statistics
        """
        diagnostics = {}

        # Ljung-Box test for autocorrelation
        try:
            lb_result = acorr_ljungbox(residuals, lags=10, return_df=True)
            diagnostics["ljung_box_pvalue"] = lb_result["lb_pvalue"].iloc[-1]
            diagnostics["ljung_box_statistic"] = lb_result["lb_stat"].iloc[-1]
        except Exception as e:
            logger.warning(f"Ljung-Box test failed: {e}")
            diagnostics["ljung_box_pvalue"] = None

        # Jarque-Bera test for normality
        try:
            jb_result = jarque_bera(residuals)
            # Handle both tuple and named tuple returns
            if isinstance(jb_result, tuple):
                diagnostics["jarque_bera_statistic"] = float(jb_result[0])
                diagnostics["jarque_bera_pvalue"] = float(jb_result[1])
            else:
                diagnostics["jarque_bera_statistic"] = float(jb_result.statistic)
                diagnostics["jarque_bera_pvalue"] = float(jb_result.pvalue)
        except Exception as e:
            logger.warning(f"Jarque-Bera test failed: {e}")
            diagnostics["jarque_bera_statistic"] = None
            diagnostics["jarque_bera_pvalue"] = None

        # Basic statistics
        diagnostics["residual_mean"] = float(residuals.mean())
        diagnostics["residual_std"] = float(residuals.std())
        diagnostics["residual_skew"] = float(residuals.skew())
        diagnostics["residual_kurtosis"] = float(residuals.kurtosis())

        return diagnostics
