"""
Factor timing signal generation for QuantFolio Engine.

This module implements factor exposure calculations and regime detection:
- Fama-French 3/5 factor model regression
- Rolling factor statistics
- Regime classification using clustering and HMM
"""

from pathlib import Path
from typing import Dict, Optional, Union

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from quantfolio_engine.config import PROCESSED_DATA_DIR


class FactorExposureCalculator:
    """Calculate factor exposures using Fama-French factor model regression."""

    def __init__(self, lookback_period: int = 60):
        """
        Initialize factor exposure calculator.

        Args:
            lookback_period: Number of months for rolling regression window
        """
        self.lookback_period = lookback_period
        self.factor_exposures: Dict[str, pd.Series] = {}

    def calculate_rolling_factor_exposures(
        self, returns_df: pd.DataFrame, factors_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate rolling factor exposures using Fama-French model.

        Args:
            returns_df: Asset returns (assets as columns, dates as index)
            factors_df: Factor returns (factors as columns, dates as index)

        Returns:
            DataFrame with rolling factor exposures for each asset
        """
        logger.info("Calculating rolling factor exposures...")

        # Clean and prepare factors data
        factors_clean = self._prepare_factors_data(factors_df)
        if factors_clean.empty:
            logger.error("No valid factor data available")
            return pd.DataFrame()

        # Align data
        common_dates = returns_df.index.intersection(factors_clean.index)
        if len(common_dates) == 0:
            logger.error("No common dates between returns and factors data")
            return pd.DataFrame()

        returns_aligned = returns_df.loc[common_dates]
        factors_aligned = factors_clean.loc[common_dates]

        logger.info(
            f"Aligned data: {len(returns_aligned)} returns rows, {len(factors_aligned)} factors rows"
        )
        logger.info(
            f"Date range: {returns_aligned.index.min()} to {returns_aligned.index.max()}"
        )

        if len(returns_aligned) < self.lookback_period + 10:
            logger.warning(
                f"Insufficient data for rolling regression (need at least {self.lookback_period + 10} points)"
            )
            return pd.DataFrame()

        # Calculate rolling factor exposures for each asset
        exposures_data = {}

        for asset in returns_aligned.columns:
            logger.debug(f"Calculating exposures for {asset}...")
            asset_returns = returns_aligned[asset].dropna()

            if len(asset_returns) < self.lookback_period + 5:
                logger.warning(f"Insufficient data for {asset}, skipping")
                continue

            # Rolling regression
            rolling_betas = self._rolling_regression(asset_returns, factors_aligned)
            if not rolling_betas.empty:
                exposures_data[asset] = rolling_betas

        if not exposures_data:
            logger.error("No factor exposures calculated for any asset")
            return pd.DataFrame()

        # Combine into DataFrame
        exposures_df = pd.DataFrame(exposures_data)
        exposures_df.index.name = "date"

        logger.success(
            f"Calculated factor exposures for {len(exposures_df.columns)} assets"
        )
        return exposures_df

    def _prepare_factors_data(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare factors data by handling missing values and converting to returns.

        Args:
            factors_df: Raw factors DataFrame

        Returns:
            Cleaned factors DataFrame
        """
        # Remove completely empty rows
        factors_clean = factors_df.dropna(how="all")

        # Convert levels to returns (monthly changes)
        factors_returns = factors_clean.pct_change().dropna()

        # Handle infinite values
        factors_returns = factors_returns.replace([np.inf, -np.inf], np.nan)

        # Forward fill limited missing values (for quarterly data like GDP)
        factors_returns = factors_returns.fillna(method="ffill", limit=3)

        # Drop rows with too many missing values
        min_factors_required = max(3, len(factors_returns.columns) // 2)
        factors_returns = factors_returns.dropna(thresh=min_factors_required)

        logger.info(f"Factors data prepared: {factors_returns.shape}")
        logger.info(f"Factors available: {list(factors_returns.columns)}")

        return factors_returns

    def _rolling_regression(
        self, asset_returns: pd.Series, factors: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate rolling factor exposures using regression.

        Args:
            asset_returns: Asset return series
            factors: Factor return DataFrame

        Returns:
            Series with rolling factor exposures
        """
        betas = []
        dates = []

        # Ensure we have enough data
        min_required = max(self.lookback_period, len(factors.columns) + 1)

        for i in range(min_required, len(asset_returns)):
            # Get rolling window
            y = asset_returns.iloc[i - self.lookback_period : i]
            X = factors.iloc[i - self.lookback_period : i]

            # Align data within the window
            common_dates = y.index.intersection(X.index)
            if len(common_dates) < min_required:
                continue

            y_window = y.loc[common_dates]
            X_window = X.loc[common_dates]

            # Check for sufficient non-missing data
            if (
                len(y_window.dropna()) < min_required
                or X_window.isnull().sum().sum() > len(X_window) * 0.5
            ):
                continue

            # Drop rows with any missing values
            complete_data = pd.concat([y_window, X_window], axis=1).dropna()
            if len(complete_data) < min_required:
                continue

            y_clean = complete_data.iloc[:, 0]
            X_clean = complete_data.iloc[:, 1:]  # Skip the asset column

            try:
                # Use Ridge regression for better stability
                from sklearn.linear_model import Ridge

                reg = Ridge(alpha=0.1, fit_intercept=True)
                reg.fit(X_clean, y_clean)
                beta = reg.coef_

                # Store factor betas (all factors)
                betas.append(beta.tolist())
                dates.append(asset_returns.index[i])

            except Exception:
                # If regression fails, try correlation-based approach
                try:
                    correlations = X_window.corrwith(y_window)
                    # Use individual correlations for each factor, not mean
                    if not correlations.isna().all():
                        betas.append(correlations.values.tolist())
                        dates.append(asset_returns.index[i])
                except Exception:
                    # If everything fails, skip this window
                    continue

        if not betas:
            logger.warning("No valid regressions for asset, using simple correlation")
            return self._calculate_correlation_exposure(asset_returns, factors)

        # Convert to DataFrame with factor names as columns
        betas_df = pd.DataFrame(betas, index=dates, columns=factors.columns)

        # Return average exposure across factors (excluding the first factor which is often market)
        if len(betas_df.columns) > 1:
            return betas_df.iloc[:, 1:].mean(axis=1)  # Exclude first factor
        else:
            return betas_df.mean(axis=1)

    def _calculate_correlation_exposure(
        self, asset_returns: pd.Series, factors: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate factor exposure using correlation when rolling regression fails.

        Args:
            asset_returns: Asset return series
            factors: Factor return DataFrame

        Returns:
            Series with correlation-based factor exposures (sparse, matching regression pattern)
        """
        # Align data
        common_dates = asset_returns.index.intersection(factors.index)
        if len(common_dates) < 10:
            return pd.Series(dtype=float)

        asset_aligned = asset_returns.loc[common_dates]
        factors_aligned = factors.loc[common_dates]

        # Calculate correlations
        correlations = factors_aligned.corrwith(asset_aligned)

        # Return correlation as exposure - create a sparse Series with same pattern as regression
        # Only return values at the end of each rolling window to match regression behavior
        exposure_value = correlations.mean()

        # Create sparse Series with same dates as rolling regression would produce
        min_required = max(self.lookback_period, len(factors.columns) + 1)
        sparse_dates = asset_returns.index[min_required:]

        return pd.Series(exposure_value, index=sparse_dates)


class RegimeDetector:
    """Detect factor regimes using various methods."""

    def __init__(self, n_regimes: int = 3):
        """
        Initialize regime detector.

        Args:
            n_regimes: Number of regimes to detect
        """
        self.n_regimes = n_regimes
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_regimes, random_state=42)

    def detect_regimes_rolling_stats(
        self, factor_exposures: pd.DataFrame, window: int = 12
    ) -> pd.DataFrame:
        """
        Detect regimes using rolling statistics.

        Args:
            factor_exposures: Factor exposure DataFrame
            window: Rolling window size in months

        Returns:
            DataFrame with regime classifications
        """
        logger.info("Detecting regimes using rolling statistics...")

        # Calculate rolling statistics
        rolling_mean = factor_exposures.rolling(window=window).mean()
        rolling_std = factor_exposures.rolling(window=window).std()
        rolling_skew = factor_exposures.rolling(window=window).skew()

        # Combine features
        features = pd.concat(
            [
                rolling_mean.add_suffix("_mean"),
                rolling_std.add_suffix("_std"),
                rolling_skew.add_suffix("_skew"),
            ],
            axis=1,
        )

        # Remove NaN values
        features_clean = features.dropna()

        if len(features_clean) == 0:
            logger.warning("No data available for regime detection")
            return pd.DataFrame()

        # Apply dimensionality reduction to avoid curse of dimensionality
        # Standardize features first
        features_scaled = self.scaler.fit_transform(features_clean)

        # Use PCA to reduce dimensions (explain 95% of variance)
        pca = PCA(n_components=0.95, random_state=42)
        features_reduced = pca.fit_transform(features_scaled)

        logger.debug(
            f"Reduced features from {features_scaled.shape[1]} to {features_reduced.shape[1]} dimensions"
        )

        # Cluster
        regimes = self.kmeans.fit_predict(features_reduced)

        # Create result DataFrame
        regime_df = pd.DataFrame({"regime": regimes}, index=features_clean.index)

        logger.success(f"Detected {self.n_regimes} regimes using rolling statistics")
        return regime_df

    def _get_hmm_module(self):
        """
        Helper to import and return hmmlearn.hmm. Allows for easier mocking in tests.
        """
        import hmmlearn.hmm

        return hmmlearn.hmm

    def detect_regimes_hmm(self, factor_exposures: pd.DataFrame) -> pd.DataFrame:
        """
        Detect regimes using Hidden Markov Model.

        Args:
            factor_exposures: Factor exposure DataFrame

        Returns:
            DataFrame with regime probabilities
        """
        logger.info("Detecting regimes using HMM...")

        try:
            hmm_mod = self._get_hmm_module()

            # Prepare data
            data_clean = factor_exposures.dropna()

            if len(data_clean) == 0:
                logger.warning("No data available for HMM regime detection")
                return pd.DataFrame()

            # Standardize data
            data_scaled = self.scaler.fit_transform(data_clean)

            # Fit HMM
            model = hmm_mod.GaussianHMM(n_components=self.n_regimes, random_state=42)
            model.fit(data_scaled)

            # Get regime probabilities
            regime_probs = model.predict_proba(data_scaled)
            predicted_regimes = model.predict(data_scaled)

            # Create result DataFrame
            regime_df = pd.DataFrame(
                regime_probs,
                index=data_clean.index,
                columns=[f"regime_{i}_prob" for i in range(self.n_regimes)],
            )
            regime_df["regime"] = predicted_regimes

            logger.success(f"Detected {self.n_regimes} regimes using HMM")
            return regime_df

        except ImportError:
            logger.warning(
                "hmmlearn not available, falling back to clustering-based regime detection"
            )
            logger.info(
                "To use HMM regime detection, install hmmlearn: pip install hmmlearn"
            )

            # Fallback to clustering with clear indication
            return self._fallback_clustering_regimes(factor_exposures)

        except Exception as e:
            logger.error(f"HMM regime detection failed: {e}")
            logger.info("Falling back to clustering-based regime detection")
            return self._fallback_clustering_regimes(factor_exposures)

    def _fallback_clustering_regimes(
        self, factor_exposures: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Fallback regime detection using clustering when HMM fails.

        Args:
            factor_exposures: Factor exposure DataFrame

        Returns:
            DataFrame with regime classifications (simplified format)
        """
        # Prepare data
        data_clean = factor_exposures.dropna()

        if len(data_clean) == 0:
            logger.warning("No data available for fallback regime detection")
            return pd.DataFrame()

        # Standardize data
        data_scaled = self.scaler.fit_transform(data_clean)

        # Cluster
        regimes = self.kmeans.fit_predict(data_scaled)

        # Create result DataFrame with simplified format
        regime_df = pd.DataFrame({"regime": regimes}, index=data_clean.index)

        logger.info(
            f"Detected {self.n_regimes} regimes using clustering (HMM fallback)"
        )
        return regime_df


class FactorTimingEngine:
    """Main engine for factor timing signal generation."""

    def __init__(self, lookback_period: int = 60, n_regimes: int = 3):
        """
        Initialize factor timing engine.

        Args:
            lookback_period: Rolling window for factor exposure calculation
            n_regimes: Number of regimes to detect
        """
        self.exposure_calculator = FactorExposureCalculator(lookback_period)
        self.regime_detector = RegimeDetector(n_regimes)

    def generate_factor_timing_signals(
        self,
        returns_file: Optional[Union[str, Path]] = None,
        factors_file: Optional[Union[str, Path]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate factor timing signals from data files.

        Args:
            returns_file: Path to returns CSV file
            factors_file: Path to factors CSV file

        Returns:
            Dictionary with factor exposures and regime classifications
        """
        # Load data
        if returns_file is None:
            returns_file = PROCESSED_DATA_DIR / "returns_monthly.csv"
        if factors_file is None:
            factors_file = PROCESSED_DATA_DIR / "macro_monthly.csv"

        logger.info(f"Loading returns from {returns_file}")
        logger.info(f"Loading factors from {factors_file}")

        try:
            returns_df = pd.read_csv(returns_file, index_col=0, parse_dates=True)
            factors_df = pd.read_csv(factors_file, index_col=0, parse_dates=True)
        except FileNotFoundError as e:
            logger.error(f"Data file not found: {e}")
            return {}

        # Calculate factor exposures
        exposures_df = self.exposure_calculator.calculate_rolling_factor_exposures(
            returns_df, factors_df
        )

        # Detect regimes
        rolling_regimes = self.regime_detector.detect_regimes_rolling_stats(
            exposures_df
        )
        hmm_regimes = self.regime_detector.detect_regimes_hmm(exposures_df)

        # Save results
        self._save_results(exposures_df, rolling_regimes, hmm_regimes)

        return {
            "factor_exposures": exposures_df,
            "rolling_regimes": rolling_regimes,
            "hmm_regimes": hmm_regimes,
        }

    def _save_results(
        self,
        exposures_df: pd.DataFrame,
        rolling_regimes: pd.DataFrame,
        hmm_regimes: pd.DataFrame,
    ):
        """Save factor timing results to files."""
        # Save factor exposures
        exposures_file = PROCESSED_DATA_DIR / "factor_exposures.csv"
        exposures_df.to_csv(exposures_file)
        logger.info(f"Saved factor exposures to {exposures_file}")

        # Save regime classifications
        if not rolling_regimes.empty:
            rolling_file = PROCESSED_DATA_DIR / "factor_regimes_rolling.csv"
            rolling_regimes.to_csv(rolling_file)
            logger.info(f"Saved rolling regimes to {rolling_file}")

        if not hmm_regimes.empty:
            hmm_file = PROCESSED_DATA_DIR / "factor_regimes_hmm.csv"
            hmm_regimes.to_csv(hmm_file)
            logger.info(f"Saved HMM regimes to {hmm_file}")
