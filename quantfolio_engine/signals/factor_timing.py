"""
Factor timing signal generation for QuantFolio Engine.

This module implements factor exposure calculations and regime detection using:
- Fama-French factor model regression for calculating factor exposures
- Rolling statistics and Hidden Markov Models for regime detection
- Multiple factor generation methods including macro, Fama-French, and simple style factors

The main components are:
- FactorExposureCalculator: Calculates rolling factor exposures using regression
- RegimeDetector: Detects market regimes using statistical methods
- FactorTimingEngine: Main engine that combines exposure calculation and regime detection
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
    """
    Calculates factor exposures using Fama-French factor model regression.

    Uses rolling window regression to estimate time-varying factor exposures
    for each asset-factor pair. Falls back to correlation-based exposures
    if regression fails. Returns exposures for all factors, not just the first.
    """

    def __init__(self, lookback_period: int = 60):
        """
        Initialize calculator with specified lookback period.

        Args:
            lookback_period: Number of months for rolling regression window
        """
        self.lookback_period = lookback_period
        self.factor_exposures: Dict[str, pd.Series] = {}

    def calculate_rolling_factor_exposures(
        self, returns_df: pd.DataFrame, factors_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate rolling factor exposures using Fama-French model regression.

        Performs rolling window regression of asset returns on factor returns
        to estimate time-varying factor exposures.

        Args:
            returns_df: Asset returns with assets as columns, dates as index
            factors_df: Factor returns with factors as columns, dates as index

        Returns:
            DataFrame with rolling factor exposures for each asset-factor pair
        """
        logger.info("Calculating rolling factor exposures...")

        factors_clean = self._prepare_factors_data(factors_df)
        if factors_clean.empty:
            logger.error("No valid factor data available")
            return pd.DataFrame()

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

        all_exposures = {}

        for asset in returns_aligned.columns:
            logger.debug(f"Calculating exposures for {asset}...")
            asset_returns = returns_aligned[asset].dropna()

            if len(asset_returns) < self.lookback_period + 5:
                logger.warning(f"Insufficient data for {asset}, skipping")
                continue

            rolling_betas = self._rolling_regression(asset_returns, factors_aligned)
            if not rolling_betas.empty:
                if isinstance(rolling_betas, pd.DataFrame):
                    for factor in rolling_betas.columns:
                        col_name = f"{asset}_{factor}"
                        all_exposures[col_name] = rolling_betas[factor]
                else:
                    all_exposures[f"{asset}_factor"] = rolling_betas

        if not all_exposures:
            logger.error("No factor exposures calculated for any asset")
            return pd.DataFrame()

        exposures_df = pd.DataFrame(all_exposures)
        exposures_df.index.name = "date"

        logger.success(
            f"Calculated factor exposures for {len(exposures_df.columns)} asset-factor pairs"
        )
        return exposures_df

    def _prepare_factors_data(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare factors data by handling missing values and converting to returns.

        Args:
            factors_df: Raw factors DataFrame

        Returns:
            Cleaned factors DataFrame with returns calculated and missing values handled
        """
        factors_clean = factors_df.dropna(how="all")
        factors_returns = factors_clean.pct_change().dropna()
        factors_returns = factors_returns.replace([np.inf, -np.inf], np.nan)
        factors_returns = factors_returns.fillna(method="ffill", limit=3)

        min_factors_required = max(3, len(factors_returns.columns) // 2)
        factors_returns = factors_returns.dropna(thresh=min_factors_required)

        logger.info(f"Factors data prepared: {factors_returns.shape}")
        logger.info(f"Factors available: {list(factors_returns.columns)}")

        return factors_returns

    def _rolling_regression(
        self, asset_returns: pd.Series, factors: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate rolling factor exposures using Ridge regression.

        Uses Ridge regression for stability and falls back to correlation
        if regression fails.

        Args:
            asset_returns: Asset return series
            factors: Factor return DataFrame

        Returns:
            DataFrame with rolling factor exposures per factor
        """
        betas = []
        dates = []

        min_required = max(self.lookback_period, len(factors.columns) + 1)

        # Start from min_required (inclusive) so the first window ends at index min_required
        for i in range(min_required, len(asset_returns)):
            y = asset_returns.iloc[i - self.lookback_period : i]
            X = factors.iloc[i - self.lookback_period : i]

            common_dates = y.index.intersection(X.index)
            if len(common_dates) < min_required:
                continue

            y_window = y.loc[common_dates]
            X_window = X.loc[common_dates]

            if (
                len(y_window.dropna()) < min_required
                or X_window.isnull().sum().sum() > len(X_window) * 0.5
            ):
                continue

            complete_data = pd.concat([y_window, X_window], axis=1).dropna()
            if len(complete_data) < min_required:
                continue

            y_clean = complete_data.iloc[:, 0]
            X_clean = complete_data.iloc[:, 1:]

            try:
                from sklearn.linear_model import Ridge

                reg = Ridge(alpha=0.1, fit_intercept=True)
                reg.fit(X_clean, y_clean)
                beta = reg.coef_
                betas.append(beta.tolist())
                dates.append(
                    asset_returns.index[i]
                )  # Use the current date, not the end of the window

            except Exception:
                try:
                    correlations = X_window.corrwith(y_window)
                    if not correlations.isna().all():
                        betas.append(correlations.values.tolist())
                        dates.append(asset_returns.index[i])
                except Exception:
                    continue

        if not betas:
            logger.warning("No valid regressions for asset, using simple correlation")
            return self._calculate_correlation_exposure(asset_returns, factors)

        betas_df = pd.DataFrame(betas, index=dates, columns=factors.columns)
        return betas_df

    def _calculate_correlation_exposure(
        self, asset_returns: pd.Series, factors: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate factor exposure using correlation when regression fails.

        Args:
            asset_returns: Asset return series
            factors: Factor return DataFrame

        Returns:
            DataFrame with correlation-based factor exposures
        """
        common_dates = asset_returns.index.intersection(factors.index)
        if len(common_dates) < 10:
            return pd.DataFrame()

        asset_aligned = asset_returns.loc[common_dates]
        factors_aligned = factors.loc[common_dates]

        # Calculate rolling correlations using same window as regression
        correlations_df = pd.DataFrame(
            index=asset_aligned.index, columns=factors_aligned.columns
        )

        min_required = max(self.lookback_period, len(factors.columns) + 1)

        for i in range(min_required - 1, len(asset_aligned)):
            # Get rolling window
            y = asset_aligned.iloc[i - self.lookback_period : i]
            X = factors_aligned.iloc[i - self.lookback_period : i]

            # Calculate correlations for this window
            window_correlations = X.corrwith(y)
            correlations_df.iloc[i] = window_correlations

        # Remove NaN rows and return
        correlations_df = correlations_df.dropna()
        logger.warning(
            "Using rolling correlation fallback. This provides time-varying exposures."
        )
        return correlations_df


class RegimeDetector:
    """
    Detects market regimes using statistical methods.

    Implements both rolling statistics and Hidden Markov Model approaches
    for regime detection.
    """

    def __init__(self, n_regimes: int = 3):
        """
        Initialize detector with specified number of regimes.

        Args:
            n_regimes: Number of distinct regimes to detect
        """
        self.n_regimes = n_regimes
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_regimes, n_init="auto", random_state=42)

    def detect_regimes_rolling_stats(
        self, factor_exposures: pd.DataFrame, window: int = 12
    ) -> pd.DataFrame:
        """
        Detect regimes using rolling statistical features.

        Uses rolling mean, standard deviation and skewness to identify
        distinct market regimes via clustering.

        Args:
            factor_exposures: Factor exposure DataFrame
            window: Rolling window size in months

        Returns:
            DataFrame with regime classifications
        """
        logger.info("Detecting regimes using rolling statistics...")

        rolling_mean = factor_exposures.rolling(window=window).mean()
        rolling_std = factor_exposures.rolling(window=window).std()
        rolling_skew = factor_exposures.rolling(window=window).skew()

        features = pd.concat(
            [
                rolling_mean.add_suffix("_mean"),
                rolling_std.add_suffix("_std"),
                rolling_skew.add_suffix("_skew"),
            ],
            axis=1,
        )

        features_clean = features.dropna()

        if len(features_clean) < self.kmeans.n_clusters:
            logger.warning(
                "Not enough samples for KMeans regime detection. Returning empty DataFrame."
            )
            return pd.DataFrame()

        features_scaled = self.scaler.fit_transform(features_clean)
        pca = PCA(n_components=0.95, random_state=42)
        features_reduced = pca.fit_transform(features_scaled)

        logger.debug(
            f"Reduced features from {features_scaled.shape[1]} to {features_reduced.shape[1]} dimensions"
        )

        regimes = self.kmeans.fit_predict(features_reduced)
        regime_df = pd.DataFrame({"regime": regimes}, index=features_clean.index)
        logger.success(f"Detected {self.n_regimes} regimes using rolling statistics")
        return regime_df

    def _get_hmm_module(self):
        """Helper to import hmmlearn.hmm for testing."""
        import hmmlearn.hmm

        return hmmlearn.hmm

    def detect_regimes_hmm(self, factor_exposures: pd.DataFrame) -> pd.DataFrame:
        """
        Detect regimes using Hidden Markov Model.

        Uses HMM to identify latent market regimes and their transition probabilities.
        Falls back to clustering if HMM fails.

        Args:
            factor_exposures: Factor exposure DataFrame

        Returns:
            DataFrame with regime probabilities and classifications
        """
        logger.info("Detecting regimes using HMM...")

        try:
            hmm_mod = self._get_hmm_module()
            data_clean = factor_exposures.dropna()
            if len(data_clean) < self.n_regimes:
                logger.warning(
                    "Not enough samples for HMM regime detection. Returning empty DataFrame."
                )
                return pd.DataFrame()
            data_scaled = self.scaler.fit_transform(data_clean)
            model = hmm_mod.GaussianHMM(n_components=self.n_regimes, random_state=42)
            model.fit(data_scaled)
            regime_probs = model.predict_proba(data_scaled)
            predicted_regimes = model.predict(data_scaled)
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
            return self._fallback_clustering_regimes(factor_exposures)
        except Exception as e:
            logger.error(f"HMM regime detection failed: {e}")
            logger.info("Falling back to clustering-based regime detection")
            return self._fallback_clustering_regimes(factor_exposures)

    def _fallback_clustering_regimes(
        self, factor_exposures: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Fallback regime detection using K-means clustering.

        Args:
            factor_exposures: Factor exposure DataFrame

        Returns:
            DataFrame with regime classifications
        """
        data_clean = factor_exposures.dropna()

        if len(data_clean) == 0:
            logger.warning("No data available for fallback regime detection")
            return pd.DataFrame()

        data_scaled = self.scaler.fit_transform(data_clean)
        regimes = self.kmeans.fit_predict(data_scaled)
        regime_df = pd.DataFrame({"regime": regimes}, index=data_clean.index)

        logger.info(
            f"Detected {self.n_regimes} regimes using clustering (HMM fallback)"
        )
        return regime_df


class FactorTimingEngine:
    """
    Main engine for generating factor timing signals.

    Combines factor exposure calculation and regime detection to generate
    timing signals. Supports multiple factor generation methods.
    """

    def __init__(
        self,
        lookback_period: int = 60,
        n_regimes: int = 3,
        factor_method: str = "macro",
    ):
        """
        Initialize engine with specified parameters.

        Args:
            lookback_period: Rolling window for factor exposure calculation
            n_regimes: Number of regimes to detect
            factor_method: Factor generation method ("macro", "fama_french", "simple")
        """
        self.exposure_calculator = FactorExposureCalculator(lookback_period)
        self.regime_detector = RegimeDetector(n_regimes)
        self.factor_method = factor_method

    def generate_factor_timing_signals(
        self,
        returns_file: Optional[Union[str, Path]] = None,
        factors_file: Optional[Union[str, Path]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate factor timing signals from return and factor data.

        Loads data, calculates exposures using specified method, and detects
        regimes using multiple approaches.

        Args:
            returns_file: Path to returns CSV file
            factors_file: Path to factors CSV file

        Returns:
            Dictionary containing factor exposures and regime classifications
        """
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

        if self.factor_method == "macro":
            exposures_df = self.exposure_calculator.calculate_rolling_factor_exposures(
                returns_df, factors_df
            )
        elif self.factor_method == "fama_french":
            style_factors_df = self._generate_fama_french_factors(returns_df)
            exposures_df = self.exposure_calculator.calculate_rolling_factor_exposures(
                returns_df, style_factors_df
            )
        elif self.factor_method == "simple":
            style_factors_df = self._generate_simple_factors(returns_df)
            exposures_df = self.exposure_calculator.calculate_rolling_factor_exposures(
                returns_df, style_factors_df
            )
        else:
            logger.error(f"Unknown factor method: {self.factor_method}")
            return {}

        rolling_regimes = self.regime_detector.detect_regimes_rolling_stats(
            exposures_df
        )
        hmm_regimes = self.regime_detector.detect_regimes_hmm(exposures_df)

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
        """Save factor timing results to CSV files."""
        exposures_file = PROCESSED_DATA_DIR / "factor_exposures.csv"
        exposures_df.to_csv(exposures_file)
        logger.info(f"Saved factor exposures to {exposures_file}")

        if not rolling_regimes.empty:
            rolling_file = PROCESSED_DATA_DIR / "factor_regimes_rolling.csv"
            rolling_regimes.to_csv(rolling_file)
            logger.info(f"Saved rolling regimes to {rolling_file}")

        if not hmm_regimes.empty:
            hmm_file = PROCESSED_DATA_DIR / "factor_regimes_hmm.csv"
            hmm_regimes.to_csv(hmm_file)
            logger.info(f"Saved HMM regimes to {hmm_file}")

    def _generate_fama_french_factors(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate Fama-French style factors from returns data.

        Creates market, size, value and momentum factors using ETF proxies
        when available, otherwise uses statistical approximations.

        Args:
            returns_df: Asset returns DataFrame

        Returns:
            DataFrame with Fama-French factors
        """
        logger.info("Generating Fama-French style factors...")

        market_factor = (
            returns_df["SPY"]
            if "SPY" in returns_df.columns
            else returns_df.mean(axis=1)
        )

        if "IWM" in returns_df.columns and "SPY" in returns_df.columns:
            size_factor = returns_df["IWM"] - returns_df["SPY"]
        else:
            rolling_vol = returns_df.rolling(12).std()
            size_factor = (
                rolling_vol.rank(axis=1, pct=True)
                .apply(lambda x: (x > 0.7).astype(float) - (x < 0.3).astype(float))
                .mean(axis=1)
            )

        value_assets = []
        growth_assets = []

        if "XLE" in returns_df.columns:
            value_assets.append(returns_df["XLE"])
        if "XLI" in returns_df.columns:
            value_assets.append(returns_df["XLI"])
        if "XLK" in returns_df.columns:
            growth_assets.append(returns_df["XLK"])
        if "XLV" in returns_df.columns:
            growth_assets.append(returns_df["XLV"])

        if value_assets and growth_assets:
            value_factor = pd.concat(value_assets, axis=1).mean(axis=1)
            growth_factor = pd.concat(growth_assets, axis=1).mean(axis=1)
            value_factor = value_factor - growth_factor
        else:
            value_factor = -returns_df.rolling(6).mean().mean(axis=1)

        momentum_12m = returns_df.rolling(12).mean().fillna(0)
        winners = momentum_12m.rank(axis=1, pct=True).apply(
            lambda x: (x > 0.7).astype(float)
        )
        losers = momentum_12m.rank(axis=1, pct=True).apply(
            lambda x: (x < 0.3).astype(float)
        )

        winner_count = winners.sum(axis=1).replace(0, 1)
        loser_count = losers.sum(axis=1).replace(0, 1)
        winner_returns = (returns_df * winners).sum(axis=1) / winner_count
        loser_returns = (returns_df * losers).sum(axis=1) / loser_count
        momentum_factor = (winner_returns - loser_returns).fillna(0)

        factors_df = pd.DataFrame(
            {
                "market": market_factor.fillna(0),
                "size": size_factor.fillna(0),
                "value": value_factor.fillna(0),
                "momentum": momentum_factor,
            },
            index=returns_df.index,
        )

        logger.info(f"Generated Fama-French factors: {list(factors_df.columns)}")
        return factors_df

    def _generate_simple_factors(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate simple style factors from returns data.

        Creates momentum, value and size factors using basic statistical measures.

        Args:
            returns_df: Asset returns DataFrame

        Returns:
            DataFrame with simple style factors
        """
        logger.info("Generating simple style factors...")

        momentum = returns_df.rolling(12).mean().mean(axis=1).fillna(0)
        value = -returns_df.rolling(6).mean().mean(axis=1).fillna(0)
        size = returns_df.rolling(24).std().mean(axis=1).fillna(0)

        factors_df = pd.DataFrame(
            {"momentum": momentum, "value": value, "size": size}, index=returns_df.index
        )

        logger.info(f"Generated simple factors: {list(factors_df.columns)}")
        return factors_df
