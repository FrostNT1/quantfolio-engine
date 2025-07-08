"""
Black-Litterman portfolio optimization model for QuantFolio Engine.

This module implements the Black-Litterman model with:
- Empirical covariance matrix estimation
- Factor-timing based views
- Sentiment-adjusted priors
- Customizable constraints
"""

from typing import Dict, Optional, Tuple, Union

import cvxpy as cp
from loguru import logger
import numpy as np
import pandas as pd

from quantfolio_engine.config import ASSET_UNIVERSE


class BlackLittermanOptimizer:
    """
    Black-Litterman portfolio optimization model.

    Combines market equilibrium with investor views and quantitative signals
    to generate optimal portfolio weights.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        market_cap_weight: Optional[Dict[str, float]] = None,
        tau: float = 0.05,
        lambda_mkt: float = 0.25,  # Market risk aversion parameter (scaled for monthly Σ)
        time_basis: str = "monthly",  # Time basis for all calculations
    ):
        """
        Initialize Black-Litterman optimizer.

        Args:
            risk_free_rate: Annual risk-free rate
            market_cap_weight: Market capitalization weights (if None, equal weights)
            tau: Prior uncertainty parameter (typically 0.05)
            lambda_mkt: Market risk aversion parameter (default 3.07)
        """
        self.risk_free_rate = risk_free_rate
        self.tau = tau
        self.lambda_mkt = lambda_mkt
        self.grand_view_gamma = 0.0  # Grand view blend parameter
        self.time_basis = time_basis

        # Standardize risk-free rate to time basis
        if time_basis == "monthly":
            self.rf_monthly = risk_free_rate / 12
            self.rf_annual = risk_free_rate
        elif time_basis == "annual":
            self.rf_monthly = risk_free_rate / 12
            self.rf_annual = risk_free_rate
        else:
            raise ValueError("time_basis must be 'monthly' or 'annual'")

        # Set default market cap weights if not provided
        if market_cap_weight is None:
            n_assets = len(ASSET_UNIVERSE)
            self.market_cap_weight = {
                asset: 1.0 / n_assets for asset in ASSET_UNIVERSE.keys()
            }
        else:
            self.market_cap_weight = market_cap_weight

        # Normalize weights
        total_weight = sum(self.market_cap_weight.values())
        self.market_cap_weight = {
            k: v / total_weight for k, v in self.market_cap_weight.items()
        }

    def estimate_covariance_matrix(
        self, returns_df: pd.DataFrame, method: str = "sample", shrinkage: float = 0.1
    ) -> pd.DataFrame:
        """
        Estimate covariance matrix from historical returns.

        Args:
            returns_df: Asset returns DataFrame
            method: Estimation method ("sample", "lw", "oas")
            shrinkage: Shrinkage parameter for Ledoit-Wolf

        Returns:
            Covariance matrix DataFrame

        Note:
            The τ parameter in Black-Litterman applies to this estimated covariance matrix.
            When using shrinkage methods (lw, oas), τ scales the shrunk covariance.
        """
        logger.info(f"Estimating covariance matrix using {method} method...")

        # Clean data
        returns_clean = returns_df.dropna()

        if method == "sample":
            cov_matrix = returns_clean.cov()
        elif method == "lw":
            # Ledoit-Wolf shrinkage
            from sklearn.covariance import LedoitWolf

            lw = LedoitWolf(assume_centered=True)
            cov_matrix = pd.DataFrame(
                lw.fit(returns_clean).covariance_,
                index=returns_clean.columns,
                columns=returns_clean.columns,
            )
        elif method == "oas":
            # Oracle Approximating Shrinkage
            from sklearn.covariance import OAS

            oas = OAS()
            cov_matrix = pd.DataFrame(
                oas.fit(returns_clean).covariance_,
                index=returns_clean.columns,
                columns=returns_clean.columns,
            )
        else:
            raise ValueError(f"Unknown covariance method: {method}")

        logger.info(f"Covariance matrix shape: {cov_matrix.shape}")
        return cov_matrix

    def calculate_equilibrium_returns(
        self,
        cov_matrix: pd.DataFrame,
        market_cap_weight: Optional[Dict[str, float]] = None,
        grand_view_gamma: float = 0.0,
    ) -> pd.Series:
        """
        Calculate equilibrium returns using reverse optimization.

        Args:
            cov_matrix: Asset covariance matrix
            market_cap_weight: Market cap weights (uses instance default if None)
            grand_view_gamma: Grand view blend parameter (0.0 = pure π, 1.0 = pure μ̄)

        Returns:
            Equilibrium returns Series
        """
        if market_cap_weight is None:
            market_cap_weight = self.market_cap_weight

        # Convert to numpy arrays
        w_mkt = np.array([market_cap_weight[asset] for asset in cov_matrix.columns])
        sigma = cov_matrix.values

        # Reverse optimization: π = λΣw_mkt
        # Note: λ = 0.25 is calibrated for monthly covariance matrix Σ
        # This gives reasonable equilibrium returns in monthly units
        pi = self.lambda_mkt * sigma @ w_mkt

        # Apply grand view blend if γ > 0
        if grand_view_gamma > 0:
            # Calculate grand mean (equal-weighted historical mean)
            # This would typically come from a broader market index or CAPE-implied premium
            # For now, use the historical mean of the assets in our universe
            grand_mean = np.mean(pi)  # Simple average of equilibrium returns
            pi_blended = (1 - grand_view_gamma) * pi + grand_view_gamma * grand_mean
            pi = pi_blended
            logger.info(f"Applied grand view blend (γ={grand_view_gamma:.2f})")

        equilibrium_returns = pd.Series(pi, index=cov_matrix.columns)

        # Diagnostic logging
        logger.info(f"Market cap weights: {dict(zip(cov_matrix.columns, w_mkt))}")
        logger.info(f"Lambda market: {self.lambda_mkt}")
        logger.info(f"Equilibrium returns (monthly): {equilibrium_returns.to_dict()}")
        logger.info(f"Mean equilibrium return: {equilibrium_returns.mean():.6f}")
        logger.info(f"Risk-free rate (monthly): {self.risk_free_rate / 12:.6f}")

        return equilibrium_returns

    def _calculate_information_coefficients(
        self,
        factor_exposures: pd.DataFrame,
        returns_df: pd.DataFrame,
        lookback_periods: int = 12,
    ) -> Dict[str, float]:
        """
        Calculate Information Coefficients (IC) for factor timing views.

        IC measures the correlation between factor exposures and forward returns.
        Higher IC indicates more predictive power for factor timing.

        Args:
            factor_exposures: Asset factor exposures (long format with date, asset columns)
            returns_df: Asset returns
            lookback_periods: Number of periods for IC calculation

        Returns:
            Dictionary mapping factor names to their IC values
        """
        ic_values = {}

        # Handle long format factor exposures (with date, asset columns)
        if "date" in factor_exposures.columns and "asset" in factor_exposures.columns:
            # Factor exposures is in long format - pivot to wide format for each factor
            factor_columns = [
                col for col in factor_exposures.columns if col not in ["date", "asset"]
            ]

            for factor_col in factor_columns:
                try:
                    # Pivot the data for this specific factor
                    exposures_pivot = factor_exposures.pivot(
                        index="date", columns="asset", values=factor_col
                    )

                    # Align with returns data
                    common_dates = exposures_pivot.index.intersection(returns_df.index)
                    if len(common_dates) < lookback_periods + 1:
                        logger.debug(
                            f"Insufficient data for IC calculation of {factor_col}: {len(common_dates)} periods"
                        )
                        continue

                    exposures_aligned = exposures_pivot.loc[common_dates]
                    returns_aligned = returns_df.loc[common_dates]

                    # Calculate rolling correlation between factor exposure and forward returns
                    correlations = []
                    for i in range(lookback_periods, len(common_dates)):
                        # Current factor exposure
                        current_exposure = exposures_aligned.iloc[i]

                        # Forward returns (next period)
                        if i + 1 < len(common_dates):
                            forward_returns = returns_aligned.iloc[i + 1]

                            # Ensure data is numeric
                            try:
                                current_exposure_numeric = pd.to_numeric(
                                    current_exposure, errors="coerce"
                                )
                                forward_returns_numeric = pd.to_numeric(
                                    forward_returns, errors="coerce"
                                )

                                # Check for valid data
                                if (
                                    not current_exposure_numeric.isna().all()
                                    and not forward_returns_numeric.isna().all()
                                ):
                                    # Remove NaN values
                                    valid_mask = ~(
                                        current_exposure_numeric.isna()
                                        | forward_returns_numeric.isna()
                                    )
                                    if (
                                        valid_mask.sum() > 1
                                    ):  # Need at least 2 points for correlation
                                        # Extract values for correlation calculation
                                        exposure_values = current_exposure_numeric[
                                            valid_mask
                                        ].values
                                        returns_values = forward_returns_numeric[
                                            valid_mask
                                        ].values

                                        if (
                                            len(exposure_values) > 1
                                            and len(returns_values) > 1
                                        ):
                                            corr = np.corrcoef(
                                                exposure_values, returns_values
                                            )[0, 1]
                                            if not np.isnan(corr):
                                                correlations.append(corr)
                            except (ValueError, TypeError) as e:
                                logger.debug(
                                    f"Error calculating correlation for {factor_col}: {e}"
                                )
                                continue

                    if correlations:
                        ic_values[factor_col] = np.mean(correlations)
                        logger.debug(
                            f"IC for {factor_col}: {ic_values[factor_col]:.4f}"
                        )

                except Exception as e:
                    logger.debug(f"Error processing factor {factor_col}: {e}")
                    continue
        elif isinstance(factor_exposures.index, pd.MultiIndex):
            # Handle MultiIndex structure (legacy support)
            factor_columns = [
                col for col in factor_exposures.columns if col not in ["date", "asset"]
            ]

            for factor_col in factor_columns:
                try:
                    # Pivot the data for this specific factor
                    factor_data = factor_exposures.reset_index()
                    exposures_pivot = factor_data.pivot(
                        index="date", columns="asset", values=factor_col
                    )

                    # Align with returns data
                    common_dates = exposures_pivot.index.intersection(returns_df.index)
                    if len(common_dates) < lookback_periods + 1:
                        logger.debug(
                            f"Insufficient data for IC calculation of {factor_col}: {len(common_dates)} periods"
                        )
                        continue

                    exposures_aligned = exposures_pivot.loc[common_dates]
                    returns_aligned = returns_df.loc[common_dates]

                    # Calculate rolling correlation between factor exposure and forward returns
                    correlations = []
                    for i in range(lookback_periods, len(common_dates)):
                        # Current factor exposure
                        current_exposure = exposures_aligned.iloc[i]

                        # Forward returns (next period)
                        if i + 1 < len(common_dates):
                            forward_returns = returns_aligned.iloc[i + 1]

                            # Ensure data is numeric
                            try:
                                current_exposure_numeric = pd.to_numeric(
                                    current_exposure, errors="coerce"
                                )
                                forward_returns_numeric = pd.to_numeric(
                                    forward_returns, errors="coerce"
                                )

                                # Check for valid data
                                if (
                                    not current_exposure_numeric.isna().all()
                                    and not forward_returns_numeric.isna().all()
                                ):
                                    # Remove NaN values
                                    valid_mask = ~(
                                        current_exposure_numeric.isna()
                                        | forward_returns_numeric.isna()
                                    )
                                    if (
                                        valid_mask.sum() > 1
                                    ):  # Need at least 2 points for correlation
                                        # Extract values for correlation calculation
                                        exposure_values = current_exposure_numeric[
                                            valid_mask
                                        ].values
                                        returns_values = forward_returns_numeric[
                                            valid_mask
                                        ].values

                                        if (
                                            len(exposure_values) > 1
                                            and len(returns_values) > 1
                                        ):
                                            corr = np.corrcoef(
                                                exposure_values, returns_values
                                            )[0, 1]
                                            if not np.isnan(corr):
                                                correlations.append(corr)
                            except (ValueError, TypeError) as e:
                                logger.debug(
                                    f"Error calculating correlation for {factor_col}: {e}"
                                )
                                continue

                    if correlations:
                        ic_values[factor_col] = np.mean(correlations)
                        logger.debug(
                            f"IC for {factor_col}: {ic_values[factor_col]:.4f}"
                        )

                except Exception as e:
                    logger.debug(f"Error processing factor {factor_col}: {e}")
                    continue
        else:
            # Handle other formats (fallback)
            logger.warning(
                "Factor exposures format not recognized - using default IC values"
            )
            # Provide default IC values for common factors
            default_ic_values = {
                "UNRATE": 0.15,  # Unemployment rate
                "CPIAUCSL": 0.12,  # CPI
                "INDPRO": 0.18,  # Industrial production
                "FEDFUNDS": 0.20,  # Fed funds rate
                "GDPC1": 0.10,  # GDP
                "UMCSENT": 0.08,  # Consumer sentiment
                "GS10": 0.25,  # 10-year Treasury
                "M2SL": 0.05,  # Money supply
                "DCOILWTICO": 0.30,  # Oil price
                "^VIX": 0.35,  # VIX volatility
            }
            ic_values.update(default_ic_values)

        return ic_values

    def create_factor_timing_views(
        self,
        factor_exposures: pd.DataFrame,
        factor_regimes: pd.DataFrame,
        returns_df: pd.DataFrame,
        sentiment_scores: Optional[pd.DataFrame] = None,
        view_strength: float = 1.5,  # Stronger view strength for meaningful factor timing signals
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create views based on factor timing signals.

        Args:
            factor_exposures: Asset factor exposures (long format with date, asset columns)
            factor_regimes: Factor regime classifications
            sentiment_scores: Sentiment scores (optional)
            view_strength: Strength of factor timing views

        Returns:
            Tuple of (P, Q, Omega) matrices for Black-Litterman
        """
        logger.info("Creating factor timing views...")

        # Calculate Information Coefficients for factor timing
        ic_values = self._calculate_information_coefficients(
            factor_exposures, returns_df
        )

        # Handle long format factor exposures data
        if "date" in factor_exposures.columns and "asset" in factor_exposures.columns:
            # Convert to wide format for the latest date
            latest_date = factor_exposures["date"].max()
            latest_exposures = factor_exposures[factor_exposures["date"] == latest_date]

            # Pivot to get asset exposures for the latest date
            exposures_wide = latest_exposures.pivot(
                index="date", columns="asset", values="UNRATE"  # Use any factor column
            )

            # Get all factor columns except date and asset
            factor_columns = [
                col for col in factor_exposures.columns if col not in ["date", "asset"]
            ]

            # Create a combined exposure for view formation
            latest_exposures_combined = {}
            for asset in exposures_wide.columns:
                asset_data = latest_exposures[latest_exposures["asset"] == asset]
                if not asset_data.empty:
                    # Use the first factor as representative (or average across factors)
                    latest_exposures_combined[asset] = asset_data.iloc[0]

            latest_exposures = pd.Series(latest_exposures_combined)

        else:
            # Handle wide format or MultiIndex format (legacy)
            # Align data
            common_dates = factor_exposures.index.intersection(factor_regimes.index)
            if len(common_dates) == 0:
                logger.warning("No common dates between factor exposures and regimes")
                return np.array([]), np.array([]), np.array([])

            exposures_aligned = factor_exposures.loc[common_dates]
            # Get latest data for view formation
            latest_exposures = exposures_aligned.iloc[-1]

        # Debug logging only when appropriate level is set
        logger.debug(f"Latest exposures: {latest_exposures}")
        logger.debug(f"Latest exposures index: {latest_exposures.index}")

        # Create views based on factor exposures and regimes
        views = []
        view_returns = []
        view_uncertainties = []
        view_pairs = set()  # Track view pairs to prevent duplicates

        assets = list(ASSET_UNIVERSE.keys())

        # Check for missing assets in universe
        missing_assets = set(returns_df.columns) - set(ASSET_UNIVERSE.keys())
        if missing_assets:
            logger.warning(f"Assets not in universe: {missing_assets}")

        # Enhanced factor timing views with stronger signals and regime awareness

        # Helper function to extract asset name from factor column
        def extract_asset(col):
            return col.split("_")[0] if "_" in col else col

        # Get current regime for regime-aware view strength
        current_regime = None
        if factor_regimes is not None and not factor_regimes.empty:
            latest_regime = factor_regimes.iloc[-1]
            try:
                if isinstance(latest_regime, (pd.Series, pd.DataFrame)):
                    # Try multiple ways to get regime
                    if "regime" in latest_regime.index:
                        current_regime = latest_regime["regime"]
                    elif (
                        hasattr(latest_regime, "get")
                        and latest_regime.get("regime") is not None
                    ):
                        current_regime = latest_regime.get("regime")
                    elif len(latest_regime) == 1:  # Single value series
                        current_regime = latest_regime.iloc[0]
            except (KeyError, AttributeError, IndexError):
                # Handle case where factor_regimes doesn't have expected structure
                current_regime = None

        # Enhanced regime-aware view strength multipliers
        regime_multipliers = {
            0: 2.0,  # Bull market - much stronger views (momentum works well)
            1: 1.2,  # Neutral market - moderate views
            2: 1.5,  # Bear market - stronger views (value and defensive factors)
        }
        # Only use current_regime as key if it's an int
        if isinstance(current_regime, int):
            regime_multiplier = regime_multipliers.get(current_regime, 1.0)
        else:
            regime_multiplier = 1.0

        # Base view strength with regime adjustment
        adjusted_view_strength = view_strength * regime_multiplier

        logger.info(
            f"Regime {current_regime}: Using view strength multiplier {regime_multiplier:.2f}"
        )
        logger.info(f"Adjusted view strength: {adjusted_view_strength:.3f}")

        # For long format data, we need to create views differently
        if "date" in factor_exposures.columns and "asset" in factor_exposures.columns:
            # Create views based on factor exposures for the latest date
            latest_date = factor_exposures["date"].max()
            latest_data = factor_exposures[factor_exposures["date"] == latest_date]

            # Group by asset and create views based on factor exposures
            for factor_col in factor_columns:
                if factor_col in latest_data.columns:
                    # Get exposures for this factor
                    factor_exposures_series = latest_data.set_index("asset")[factor_col]
                    factor_exposures_series = pd.to_numeric(
                        factor_exposures_series, errors="coerce"
                    ).dropna()

                    if len(factor_exposures_series) >= 2:
                        # Find assets with highest and lowest exposures
                        high_exposure = factor_exposures_series.nlargest(2)
                        low_exposure = factor_exposures_series.nsmallest(2)

                        for high_asset in high_exposure.index:
                            for low_asset in low_exposure.index:
                                if (
                                    high_asset != low_asset
                                    and high_asset in assets
                                    and low_asset in assets
                                ):
                                    # Prevent duplicate views
                                    view_pair = tuple(sorted([high_asset, low_asset]))
                                    if view_pair in view_pairs:
                                        continue
                                    view_pairs.add(view_pair)

                                    p_row = np.zeros(len(assets))
                                    p_row[assets.index(high_asset)] = 1
                                    p_row[assets.index(low_asset)] = -1

                                    views.append(p_row)

                                    # Scale return using Information Coefficient
                                    ic_value = ic_values.get(factor_col, 0.1)
                                    avg_exposure = abs(factor_exposures_series.mean())

                                    # IC-based scaling: higher IC = stronger views
                                    base_return = (
                                        0.01 * abs(ic_value) * regime_multiplier
                                    )
                                    scaled_return = base_return * max(
                                        avg_exposure, 0.01
                                    )

                                    view_returns.append(
                                        adjusted_view_strength * scaled_return
                                    )
                                    view_uncertainties.append(
                                        0.5
                                        * abs(adjusted_view_strength * scaled_return)
                                    )

        else:
            # Original wide format logic (legacy)
            # View 1: Inflation Factor Timing (CPI)
            # Assets with high CPI exposure may outperform in inflationary environments
            cpi_exposures = latest_exposures.filter(like="CPIAUCSL")
            if not cpi_exposures.empty:
                cpi_exposures = pd.to_numeric(cpi_exposures, errors="coerce").dropna()
                if not cpi_exposures.empty:
                    high_cpi = cpi_exposures.nlargest(3).index
                    low_cpi = cpi_exposures.nsmallest(3).index

                    for high_asset_col in high_cpi:
                        for low_asset_col in low_cpi:
                            high_asset = extract_asset(high_asset_col)
                            low_asset = extract_asset(low_asset_col)
                            if (
                                high_asset != low_asset
                                and high_asset in assets
                                and low_asset in assets
                            ):
                                # Prevent duplicate views
                                view_pair = tuple(sorted([high_asset, low_asset]))
                                if view_pair in view_pairs:
                                    continue
                                view_pairs.add(view_pair)

                                p_row = np.zeros(len(assets))
                                p_row[assets.index(high_asset)] = 1
                                p_row[assets.index(low_asset)] = -1

                                views.append(p_row)
                                # Scale return using Information Coefficient and factor volatility
                                factor_name = "CPIAUCSL"
                                ic_value = ic_values.get(
                                    factor_name, 0.1
                                )  # Default IC if not available
                                avg_exposure = abs(cpi_exposures.mean())

                                # IC-based scaling: higher IC = stronger views
                                # Base monthly return: 1% with IC adjustment
                                base_return = 0.01 * abs(ic_value) * regime_multiplier
                                scaled_return = base_return * max(avg_exposure, 0.01)

                                view_returns.append(
                                    adjusted_view_strength * scaled_return
                                )
                                view_uncertainties.append(
                                    0.5 * abs(adjusted_view_strength * scaled_return)
                                )

        if not views:
            logger.warning("No factor timing views created")
            return np.array([]), np.array([]), np.array([])

        # Convert to numpy arrays
        P = np.array(views)
        Q = np.array(view_returns)
        Omega = np.diag(view_uncertainties)

        logger.info(f"Created {len(views)} factor timing views")

        return P, Q, Omega

    def optimize_portfolio(
        self,
        returns_df: pd.DataFrame,
        factor_exposures: Optional[pd.DataFrame] = None,
        factor_regimes: Optional[pd.DataFrame] = None,
        sentiment_scores: Optional[pd.DataFrame] = None,
        constraints: Optional[Dict] = None,
        target_return: Optional[float] = None,
        max_volatility: Optional[float] = None,
    ) -> Dict[str, Union[pd.Series, float]]:
        """
        Optimize portfolio using Black-Litterman model.

        Args:
            returns_df: Asset returns DataFrame
            factor_exposures: Factor exposures (optional)
            factor_regimes: Factor regimes (optional)
            sentiment_scores: Sentiment scores (optional)
            constraints: Portfolio constraints
            target_return: Target annual return
            max_volatility: Maximum annual volatility

        Returns:
            Dictionary with optimal weights and metrics
        """
        logger.info("Starting Black-Litterman portfolio optimization...")

        # Estimate covariance matrix
        cov_matrix = self.estimate_covariance_matrix(returns_df)

        # Calculate equilibrium returns
        pi = self.calculate_equilibrium_returns(
            cov_matrix, grand_view_gamma=self.grand_view_gamma
        )

        # Create views if factor data is available
        P, Q, Omega = (
            self.create_factor_timing_views(
                factor_exposures,
                factor_regimes,
                returns_df,
                sentiment_scores,
                view_strength=getattr(
                    self, "view_strength", 1.5
                ),  # Use instance view_strength if set
            )
            if factor_exposures is not None
            else (np.array([]), np.array([]), np.array([]))
        )

        # Black-Litterman posterior calculation
        if len(P) > 0:
            logger.info("Calculating posterior with factor timing views...")
            # With views
            tau_sigma = self.tau * cov_matrix.values

            # Posterior precision matrix
            M1 = np.linalg.inv(tau_sigma)
            M2 = P.T @ np.linalg.inv(Omega) @ P
            M = M1 + M2

            # Posterior mean
            m1 = M1 @ pi.values
            m2 = P.T @ np.linalg.inv(Omega) @ Q
            mu_bl = np.linalg.inv(M) @ (m1 + m2)

            # Posterior covariance: Σ_bl = Σ + inv(M) (He & Litterman, 1999)
            # Add minimum variance floor to prevent singular matrices
            M_inv = np.linalg.inv(M)
            sigma_bl = cov_matrix.values + M_inv

            # Ensure minimum variance floor to prevent numerical issues
            min_variance = 1e-6
            for i in range(sigma_bl.shape[0]):
                if sigma_bl[i, i] < min_variance:
                    sigma_bl[i, i] = min_variance

        else:
            # No views - use equilibrium returns with prior covariance
            logger.info(
                "No views available - using equilibrium returns with prior covariance"
            )
            mu_bl = pi.values
            sigma_bl = cov_matrix.values  # Prior Σ, not τΣ

        # Convert to pandas
        mu_bl_series = pd.Series(mu_bl, index=cov_matrix.columns)
        sigma_bl_df = pd.DataFrame(
            sigma_bl, index=cov_matrix.columns, columns=cov_matrix.columns
        )

        # Optimize weights with constraints
        optimal_weights = self._optimize_weights(
            mu_bl_series, sigma_bl_df, constraints, target_return, max_volatility
        )

        # Calculate portfolio metrics (monthly)
        portfolio_return_monthly = (optimal_weights * mu_bl_series).sum()
        portfolio_vol_monthly = np.sqrt(
            optimal_weights @ sigma_bl_df.values @ optimal_weights
        )

        # Annualize the metrics
        portfolio_return_annual = portfolio_return_monthly * 12
        portfolio_vol_annual = portfolio_vol_monthly * np.sqrt(12)

        # Calculate Sharpe ratio using annualized metrics
        sharpe_ratio = (
            portfolio_return_annual - self.risk_free_rate
        ) / portfolio_vol_annual

        # Calculate additional risk metrics using Monte Carlo simulation
        # Generate portfolio returns using the optimal weights and historical data
        portfolio_returns = (returns_df * optimal_weights).sum(axis=1)

        # Calculate max drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calculate VaR (95%)
        var_95 = np.percentile(portfolio_returns, 5)  # 5th percentile for 95% VaR

        # Debug logging
        logger.info("Portfolio metrics calculation:")
        logger.info(f"  Monthly return: {portfolio_return_monthly:.6f}")
        logger.info(f"  Monthly volatility: {portfolio_vol_monthly:.6f}")
        logger.info(f"  Annual return: {portfolio_return_annual:.6f}")
        logger.info(f"  Annual volatility: {portfolio_vol_annual:.6f}")
        logger.info(f"  Risk-free rate: {self.risk_free_rate:.6f}")
        logger.info(f"  Sharpe ratio: {sharpe_ratio:.6f}")
        logger.info(f"  Max drawdown: {max_drawdown:.6f}")
        logger.info(f"  VaR (95%): {var_95:.6f}")

        results = {
            "weights": pd.Series(optimal_weights, index=cov_matrix.columns),
            "expected_return": portfolio_return_annual,  # Return annualized value
            "volatility": portfolio_vol_annual,  # Return annualized value
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "var_95": var_95,
            "covariance_matrix": sigma_bl_df,
            "equilibrium_returns": pi,
            "posterior_returns": mu_bl_series,
        }

        logger.success(f"Portfolio optimization completed. Sharpe: {sharpe_ratio:.3f}")
        return results

    def _optimize_weights(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        constraints: Optional[Dict] = None,
        target_return: Optional[float] = None,
        max_volatility: Optional[float] = None,
    ) -> np.ndarray:
        """
        Optimize portfolio weights using CVXPY.

        Args:
            expected_returns: Expected returns
            covariance_matrix: Covariance matrix
            constraints: Portfolio constraints
            target_return: Target return constraint
            max_volatility: Maximum volatility constraint

        Returns:
            Optimal weights array
        """
        n_assets = len(expected_returns)

        # Variables
        w = cp.Variable(n_assets)

        # Objective: maximize excess return (Sharpe-optimal)
        # Use same risk aversion as market equilibrium calculation for consistency

        # Use excess returns for better economic logic
        # Note: Sharpe ratio calculation uses annualized returns minus annual RF rate
        # Since we're working in monthly space, convert annual RF to monthly
        monthly_rf = self.risk_free_rate / 12
        excess_returns = expected_returns.values - monthly_rf

        # Keep both returns and covariance in monthly units for consistency
        # Note: max_volatility constraint should also be in monthly units
        # λ = 0.25 is the same risk aversion parameter used in equilibrium returns calculation
        objective = cp.Maximize(
            excess_returns @ w
            - 0.5 * self.lambda_mkt * cp.quad_form(w, covariance_matrix.values)
        )

        # Constraints
        constraints_list = [
            cp.sum(w) == 1,  # Weights sum to 1
            w >= 0,  # Long-only constraint
        ]

        logger.info("Applying long-only portfolio constraints (no short selling)")

        # Add target return constraint
        if target_return is not None:
            constraints_list.append(expected_returns.values @ w >= target_return)

        # Add maximum volatility constraint
        if max_volatility is not None:
            # Convert annual max_volatility to monthly for consistency with covariance matrix
            monthly_max_vol = max_volatility / np.sqrt(12)
            constraints_list.append(
                cp.quad_form(w, covariance_matrix.values) <= monthly_max_vol**2
            )

        # Add custom constraints
        if constraints:
            if "max_weight" in constraints:
                constraints_list.append(w <= constraints["max_weight"])
            if "min_weight" in constraints:
                constraints_list.append(w >= constraints["min_weight"])
            if "sector_limits" in constraints:
                for sector, limit in constraints["sector_limits"].items():
                    sector_assets = [
                        i
                        for i, asset in enumerate(expected_returns.index)
                        if ASSET_UNIVERSE[asset]["type"] == sector
                    ]
                    if sector_assets:
                        constraints_list.append(cp.sum(w[sector_assets]) <= limit)

        # Solve optimization problem
        problem = cp.Problem(objective, constraints_list)
        # Use ECOS_BB solver for long-only quadratic programs, fallback to OSQP
        try:
            problem.solve(solver=cp.ECOS_BB)
        except cp.error.SolverError:
            logger.warning("ECOS_BB solver not available, trying OSQP...")
            problem.solve(solver=cp.OSQP)

        if problem.status != "optimal":
            logger.warning(f"Optimization failed with status: {problem.status}")
            # Fallback to equal weights
            return np.ones(n_assets) / n_assets

        # Clip negative weights due to numerical precision issues
        weights = np.clip(w.value, 0, None)
        weights /= weights.sum()  # Renormalize to sum to 1

        return weights

    def calibrate_market_risk_aversion(
        self,
        returns_df: pd.DataFrame,
        target_sharpe: Optional[float] = None,
        lambda_range: tuple = (0.5, 0.75),
        n_points: int = 10,
    ) -> float:
        """
        Calibrate market risk aversion λ to achieve realistic equilibrium returns.

        Args:
            returns_df: Asset returns DataFrame
            target_sharpe: Target Sharpe ratio (if None, uses historical market Sharpe)
            lambda_range: Range of λ values to test (min, max)
            n_points: Number of λ values to test

        Returns:
            Calibrated λ value
        """
        logger.info("Calibrating market risk aversion λ...")

        # Validate parameters
        if lambda_range[0] >= lambda_range[1]:
            raise ValueError(
                f"Invalid lambda_range: min ({lambda_range[0]}) must be less than max ({lambda_range[1]})"
            )
        if n_points <= 0:
            raise ValueError(f"n_points must be positive, got {n_points}")

        # Calculate market cap weights (equal weight for now)
        market_cap_weight = {
            asset: 1.0 / len(returns_df.columns) for asset in returns_df.columns
        }
        w_mkt = np.array([market_cap_weight[asset] for asset in returns_df.columns])

        # Calculate historical market Sharpe if target not provided
        if target_sharpe is None:
            market_returns = (returns_df * w_mkt).sum(axis=1)
            market_sharpe = (market_returns.mean() * 12 - self.risk_free_rate) / (
                market_returns.std() * np.sqrt(12)
            )
            target_sharpe = market_sharpe
            logger.info(f"Historical market Sharpe: {market_sharpe:.3f}")

        # Estimate covariance matrix
        cov_matrix = self.estimate_covariance_matrix(returns_df)
        sigma = cov_matrix.values

        # Test different λ values
        lambda_values = np.linspace(lambda_range[0], lambda_range[1], n_points)
        equilibrium_sharpes = []

        for lam in lambda_values:
            # Calculate equilibrium returns
            pi = lam * sigma @ w_mkt

            # Calculate equilibrium portfolio Sharpe
            eq_return = (pi * w_mkt).sum() * 12  # Annualized
            eq_vol = np.sqrt(w_mkt @ sigma @ w_mkt) * np.sqrt(12)  # Annualized
            eq_sharpe = (eq_return - self.risk_free_rate) / eq_vol
            equilibrium_sharpes.append(eq_sharpe)

        # Find λ that gives closest Sharpe to target
        sharpe_diffs = np.abs(np.array(equilibrium_sharpes) - target_sharpe)
        best_idx = np.argmin(sharpe_diffs)
        best_lambda = lambda_values[best_idx]

        logger.info(
            f"Calibrated λ: {best_lambda:.3f} (target Sharpe: {target_sharpe:.3f})"
        )
        logger.info(
            f"Equilibrium Sharpe with calibrated λ: {equilibrium_sharpes[best_idx]:.3f}"
        )

        return best_lambda
