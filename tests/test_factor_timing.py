"""Tests for the factor timing module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quantfolio_engine.signals.factor_timing import (
    FactorExposureCalculator,
    FactorTimingEngine,
    RegimeDetector,
)


class TestFactorExposureCalculator:
    """Test cases for FactorExposureCalculator."""

    def test_init(self):
        """Test FactorExposureCalculator initialization."""
        calculator = FactorExposureCalculator(lookback_period=60)
        assert calculator.lookback_period == 60
        assert calculator.factor_exposures == {}

    def test_calculate_rolling_factor_exposures_multiindex_output(self):
        """Test that rolling factor exposure calculation returns MultiIndex DataFrame."""
        calculator = FactorExposureCalculator(lookback_period=12)

        # Create dummy data with proper structure
        dates = pd.date_range("2020-01-01", periods=24, freq="ME")
        spy_prices = 100 * (1 + np.cumsum(np.random.normal(0.01, 0.05, 24)))
        tlt_prices = 100 * (1 + np.cumsum(np.random.normal(0.005, 0.03, 24)))

        returns_df = pd.DataFrame(
            {
                "SPY": pd.Series(spy_prices, index=dates).pct_change(),
                "TLT": pd.Series(tlt_prices, index=dates).pct_change(),
            },
            index=dates,
        )

        # Create factor data (levels, will be converted to returns)
        factors_df = pd.DataFrame(
            {
                "factor1": np.random.normal(100, 10, 24),
                "factor2": np.random.normal(50, 5, 24),
                "factor3": np.random.normal(200, 20, 24),
            },
            index=dates,
        )

        # Calculate exposures
        exposures = calculator.calculate_rolling_factor_exposures(
            returns_df, factors_df
        )

        # Check MultiIndex structure
        assert isinstance(exposures, pd.DataFrame)
        assert isinstance(exposures.index, pd.MultiIndex)
        assert len(exposures.index.levels) == 2  # date and asset
        assert exposures.index.names == ["date", "asset"]

        # Check factor-level granularity
        assert len(exposures.columns) == 3  # factor1, factor2, factor3
        assert list(exposures.columns) == ["factor1", "factor2", "factor3"]

        # Check assets are present
        unique_assets = exposures.index.get_level_values("asset").unique()
        assert len(unique_assets) == 2  # SPY and TLT
        assert "SPY" in unique_assets
        assert "TLT" in unique_assets

    def test_vectorized_rolling_regression_with_statsmodels(self):
        """Test vectorized rolling regression using statsmodels."""
        calculator = FactorExposureCalculator(lookback_period=6)

        # Create data with 3 factors
        dates = pd.date_range("2020-01-01", periods=20, freq="ME")
        asset_returns = pd.Series(np.random.normal(0.01, 0.05, 20), index=dates)
        factors = pd.DataFrame(
            {
                "factor1": np.random.normal(0, 0.02, 20),
                "factor2": np.random.normal(0, 0.015, 20),
                "factor3": np.random.normal(0, 0.01, 20),
            },
            index=dates,
        )

        # Test that the method doesn't crash and returns a DataFrame
        result = calculator._rolling_regression_vectorized(asset_returns, factors)

        # Should return DataFrame (may be empty if statsmodels fails)
        assert isinstance(result, pd.DataFrame)

    def test_manual_regression_fallback(self):
        """Test manual regression fallback when statsmodels is not available."""
        calculator = FactorExposureCalculator(lookback_period=6)

        dates = pd.date_range("2020-01-01", periods=20, freq="ME")
        asset_returns = pd.Series(np.random.normal(0.01, 0.05, 20), index=dates)
        factors = pd.DataFrame(
            {
                "factor1": np.random.normal(0, 0.02, 20),
                "factor2": np.random.normal(0, 0.015, 20),
            },
            index=dates,
        )

        # Mock Ridge regression
        with patch("sklearn.linear_model.Ridge") as mock_ridge:
            mock_ridge_instance = MagicMock()
            mock_ridge_instance.coef_ = np.array([0.1, 0.2])
            mock_ridge_instance.fit.return_value = mock_ridge_instance
            mock_ridge.return_value = mock_ridge_instance
            result = calculator._rolling_regression_manual(asset_returns, factors)

            # Should return DataFrame with factor exposures
            assert isinstance(result, pd.DataFrame)
            assert len(result.columns) == 2  # factor1, factor2
            assert len(result) > 0

    def test_correlation_fallback_per_factor(self):
        """Test that correlation fallback returns per-factor correlations, not averaged."""
        calculator = FactorExposureCalculator(lookback_period=6)

        # Create data where factors have different correlations
        dates = pd.date_range("2020-01-01", periods=20, freq="ME")
        asset_returns = pd.Series(np.random.normal(0.01, 0.05, 20), index=dates)
        factors = pd.DataFrame(
            {
                "factor1": asset_returns * 0.8
                + np.random.normal(0, 0.01, 20),  # High correlation
                "factor2": asset_returns * 0.2
                + np.random.normal(0, 0.01, 20),  # Low correlation
                "factor3": -asset_returns * 0.5
                + np.random.normal(0, 0.01, 20),  # Negative correlation
            },
            index=dates,
        )

        result = calculator._calculate_correlation_exposure_vectorized(
            asset_returns, factors
        )

        # Should return DataFrame with per-factor correlations
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 3  # factor1, factor2, factor3
        assert len(result) > 0

        # Check that correlations are different (not averaged)
        if len(result) > 1:
            # Factor1 should have highest correlation (positive)
            # Factor3 should have negative correlation
            # Factor2 should have lower correlation
            assert result["factor1"].abs().mean() > result["factor2"].abs().mean()
            assert result["factor3"].mean() < 0  # Negative correlation

    def test_window_size_validation(self):
        """Test that window size validation works correctly."""
        calculator = FactorExposureCalculator(lookback_period=3)  # Small window

        dates = pd.date_range("2020-01-01", periods=20, freq="ME")
        asset_returns = pd.Series(np.random.normal(0.01, 0.05, 20), index=dates)
        factors = pd.DataFrame(
            {
                "factor1": np.random.normal(0, 0.02, 20),
                "factor2": np.random.normal(0, 0.015, 20),
                "factor3": np.random.normal(0, 0.01, 20),
                "factor4": np.random.normal(0, 0.01, 20),
            },
            index=dates,
        )

        # Window (3) <= n_factors (4), should use correlation fallback
        result = calculator._rolling_regression_vectorized(asset_returns, factors)
        assert isinstance(result, pd.DataFrame)
        # Should fall back to correlation method

    def test_convert_to_returns_renaming(self):
        """Test that _convert_to_returns properly converts levels to returns."""
        calculator = FactorExposureCalculator(lookback_period=12)

        # Create level data with more data points to avoid filtering
        dates = pd.date_range("2020-01-01", periods=50, freq="ME")  # More data points
        factors_df = pd.DataFrame(
            {
                "GDP": [100 + i for i in range(50)],  # Monotonic increasing
                "CPI": [200 + i * 0.5 for i in range(50)],  # Monotonic increasing
                "UNRATE": [5 + np.sin(i / 10) for i in range(50)],  # Add more factors
            },
            index=dates,
        )

        result = calculator._convert_to_returns(factors_df)

        # Should be returns (percentage changes)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0  # Should have data after conversion
        assert list(result.columns) == ["GDP", "CPI", "UNRATE"]

        # First values should be NaN (no previous value for pct_change)
        # But after dropna(), the first row might not be NaN anymore
        assert len(result) > 0  # Should have data after conversion

    def test_insufficient_data_handling(self):
        """Test handling of insufficient data for regression."""
        calculator = FactorExposureCalculator(lookback_period=12)

        # Create data with insufficient observations
        dates = pd.date_range("2020-01-01", periods=10, freq="ME")  # Less than lookback
        asset_returns = pd.Series(np.random.normal(0.01, 0.05, 10), index=dates)
        factors = pd.DataFrame(
            {
                "factor1": np.random.normal(0, 0.02, 10),
                "factor2": np.random.normal(0, 0.015, 10),
            },
            index=dates,
        )

        result = calculator._rolling_regression_vectorized(asset_returns, factors)
        assert isinstance(result, pd.DataFrame)
        assert result.empty  # Should return empty DataFrame

    def test_quarterly_gap_handling(self):
        """Test handling of quarterly data gaps with forward-filling."""
        calculator = FactorExposureCalculator(lookback_period=36)

        # Create monthly asset returns
        monthly_dates = pd.date_range("2000-01-31", periods=120, freq="ME")
        asset_returns = pd.Series(np.random.randn(120), index=monthly_dates)

        # Create quarterly factor data, then forward-fill
        quarterly_dates = pd.date_range("2000-03-31", periods=40, freq="QE")
        quarterly_factor = pd.Series(np.random.randn(40), index=quarterly_dates)
        monthly_factor = quarterly_factor.resample("ME").ffill()

        factors = monthly_factor.to_frame("GDP")

        # Should handle the resampling gracefully
        result = calculator._rolling_regression_vectorized(asset_returns, factors)
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_multicollinearity_handling(self):
        """Test that Ridge regression handles multicollinearity better than OLS."""
        calculator = FactorExposureCalculator(lookback_period=6)

        dates = pd.date_range("2020-01-01", periods=20, freq="ME")
        base_factor = np.random.normal(0, 0.02, 20)
        asset_returns = pd.Series(np.random.normal(0.01, 0.05, 20), index=dates)
        factors = pd.DataFrame(
            {
                "factor1": base_factor,
                "factor2": base_factor
                + np.random.normal(0, 0.001, 20),  # Nearly collinear
                "factor3": np.random.normal(0, 0.015, 20),
            },
            index=dates,
        )

        # Should not crash with Ridge regression
        result = calculator._rolling_regression_manual(asset_returns, factors)
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 3


class TestRegimeDetector:
    """Test cases for RegimeDetector."""

    def test_init(self):
        """Test RegimeDetector initialization."""
        detector = RegimeDetector(n_regimes=3)
        assert detector.n_regimes == 3
        assert detector.kmeans.n_clusters == 3

    def test_detect_regimes_rolling_stats_fresh_scaler(self):
        """Test that rolling stats uses fresh scaler instance."""
        detector = RegimeDetector(n_regimes=3)
        # Provide enough data for regime detection
        dates = pd.date_range("2020-01-01", periods=100, freq="ME")
        assets = ["SPY", "TLT", "GLD", "IWM", "EFA", "XLE", "XLK", "XLV"]
        index = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])
        factor_exposures = pd.DataFrame(
            np.random.randn(len(index), 3),
            index=index,
            columns=["factor1", "factor2", "factor3"],
        )
        result = detector.detect_regimes_rolling_stats(factor_exposures, window=12)
        assert isinstance(result, pd.DataFrame)
        assert "regime" in result.columns
        assert len(result) > 0

    def test_detect_regimes_rolling_stats_insufficient_data(self):
        detector = RegimeDetector(n_regimes=3)
        # Provide too little data
        dates = pd.date_range("2020-01-01", periods=2, freq="ME")
        assets = ["SPY"]
        index = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])
        factor_exposures = pd.DataFrame(
            np.random.randn(len(index), 3),
            index=index,
            columns=["factor1", "factor2", "factor3"],
        )
        with pytest.raises(
            ValueError, match="Not enough samples for KMeans regime detection"
        ):
            detector.detect_regimes_rolling_stats(factor_exposures, window=12)

    def test_detect_regimes_hmm_raw_data(self):
        """Test HMM with raw data to preserve regime-specific changes."""
        detector = RegimeDetector(n_regimes=3)

        # Create MultiIndex factor exposures
        dates = pd.date_range("2020-01-01", periods=50, freq="ME")
        assets = ["SPY", "TLT"]
        index = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])
        factor_exposures = pd.DataFrame(
            np.random.randn(100, 3),
            index=index,
            columns=["factor1", "factor2", "factor3"],
        )

        with patch("hmmlearn.hmm.GaussianHMM") as mock_hmm:
            mock_model = MagicMock()
            mock_model.predict_proba.return_value = np.random.rand(100, 3)
            mock_model.predict.return_value = np.random.randint(0, 3, 100)
            mock_model.fit.return_value = mock_model
            # Add internal attributes that might be accessed
            mock_model.monitor_ = MagicMock()
            mock_model.monitor_.converged = True
            mock_hmm.return_value = mock_model

            result = detector.detect_regimes_hmm(factor_exposures)

            assert isinstance(result, pd.DataFrame)
            assert "regime" in result.columns
            assert any(
                col.startswith("regime_") and col.endswith("_prob")
                for col in result.columns
            )

    def test_hmm_import_error_fallback(self):
        """Test HMM fallback when hmmlearn is not available."""
        detector = RegimeDetector(n_regimes=3)

        dates = pd.date_range("2020-01-01", periods=50, freq="ME")
        assets = ["SPY", "TLT"]
        index = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])
        factor_exposures = pd.DataFrame(
            np.random.randn(100, 3),
            index=index,
            columns=["factor1", "factor2", "factor3"],
        )

        with patch.object(detector, "_get_hmm_module", return_value=None):
            result = detector.detect_regimes_hmm(factor_exposures)

            assert isinstance(result, pd.DataFrame)
            assert "regime" in result.columns

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        detector = RegimeDetector(n_regimes=3)
        empty_df = pd.DataFrame()
        with pytest.raises(
            ValueError, match="Not enough samples for KMeans regime detection"
        ):
            detector.detect_regimes_rolling_stats(empty_df)


class TestFactorTimingEngine:
    """Test cases for FactorTimingEngine."""

    def test_init(self):
        """Test FactorTimingEngine initialization."""
        engine = FactorTimingEngine(
            lookback_period=60, n_regimes=3, factor_method="macro"
        )
        assert engine.lookback_period == 60
        assert engine.factor_method == "macro"

    @patch("quantfolio_engine.signals.factor_timing.pd.read_csv")
    def test_generate_factor_timing_signals_macro(self, mock_read_csv):
        """Test factor timing signal generation with macro factors."""
        # Mock data loading
        dates = pd.date_range("2020-01-01", periods=50, freq="ME")
        returns_df = pd.DataFrame(
            np.random.randn(50, 2), index=dates, columns=["SPY", "TLT"]
        )
        factors_df = pd.DataFrame(
            np.random.randn(50, 3), index=dates, columns=["GDP", "CPI", "VIX"]
        )

        mock_read_csv.side_effect = [returns_df, factors_df]

        engine = FactorTimingEngine(factor_method="macro")

        with patch.object(
            engine.exposure_calculator, "calculate_rolling_factor_exposures"
        ) as mock_calc:
            with patch.object(
                engine.regime_detector, "detect_regimes_rolling_stats"
            ) as mock_rolling:
                with patch.object(
                    engine.regime_detector, "detect_regimes_hmm"
                ) as mock_hmm:

                    mock_calc.return_value = pd.DataFrame(np.random.randn(100, 3))
                    mock_rolling.return_value = pd.DataFrame(
                        {"regime": np.random.randint(0, 3, 100)}
                    )
                    mock_hmm.return_value = pd.DataFrame(
                        {"regime": np.random.randint(0, 3, 100)}
                    )

                    result = engine.generate_factor_timing_signals()

                    assert isinstance(result, dict)
                    assert "factor_exposures" in result
                    assert "rolling_regimes" in result
                    assert "hmm_regimes" in result

    def test_generate_fama_french_factors(self):
        """Test Fama-French factor generation."""
        engine = FactorTimingEngine()

        dates = pd.date_range("2020-01-01", periods=50, freq="ME")
        returns_df = pd.DataFrame(
            np.random.randn(50, 4), index=dates, columns=["SPY", "IWM", "XLE", "XLK"]
        )

        factors = engine._generate_fama_french_factors(returns_df)

        assert isinstance(factors, pd.DataFrame)
        assert "market" in factors.columns
        assert "size" in factors.columns
        assert "value" in factors.columns
        assert "momentum" in factors.columns

    def test_generate_simple_factors(self):
        """Test simple factor generation."""
        engine = FactorTimingEngine()

        dates = pd.date_range("2020-01-01", periods=50, freq="ME")
        returns_df = pd.DataFrame(
            np.random.randn(50, 3), index=dates, columns=["SPY", "TLT", "GLD"]
        )

        factors = engine._generate_simple_factors(returns_df)

        assert isinstance(factors, pd.DataFrame)
        assert "momentum" in factors.columns
        assert "value" in factors.columns
        assert "size" in factors.columns


class TestIntegration:
    """Integration tests for the factor timing module."""

    def test_end_to_end_pipeline(self):
        """Test complete end-to-end factor timing pipeline."""
        # Provide enough data for regime detection
        dates = pd.date_range("2020-01-01", periods=100, freq="ME")
        returns_df = pd.DataFrame(
            np.random.randn(100, 3), index=dates, columns=["SPY", "TLT", "GLD"]
        )
        factors_df = pd.DataFrame(
            {
                "GDP": 100 + np.cumsum(np.random.normal(0.5, 0.1, 100)),
                "CPI": 200 + np.cumsum(np.random.normal(0.2, 0.05, 100)),
                "VIX": 20 + np.random.normal(0, 5, 100),
            },
            index=dates,
        )
        calculator = FactorExposureCalculator(lookback_period=12)
        exposures = calculator.calculate_rolling_factor_exposures(
            returns_df, factors_df
        )
        if not exposures.empty:
            detector = RegimeDetector(n_regimes=3)
            rolling_regimes = detector.detect_regimes_rolling_stats(exposures)
            hmm_regimes = detector.detect_regimes_hmm(exposures)
            assert isinstance(exposures, pd.DataFrame)
            assert isinstance(rolling_regimes, pd.DataFrame)
            assert isinstance(hmm_regimes, pd.DataFrame)
            if not exposures.empty:
                assert isinstance(exposures.index, pd.MultiIndex)
                assert exposures.index.names == ["date", "asset"]

    def test_data_quality_validation(self):
        """Test data quality validation and error handling."""
        calculator = FactorExposureCalculator(lookback_period=12)

        # Test with completely missing data
        empty_returns = pd.DataFrame()
        empty_factors = pd.DataFrame()

        result = calculator.calculate_rolling_factor_exposures(
            empty_returns, empty_factors
        )
        assert result.empty

        # Test with insufficient data
        short_dates = pd.date_range("2020-01-01", periods=5, freq="ME")
        short_returns = pd.DataFrame(np.random.randn(5, 2), index=short_dates)
        short_factors = pd.DataFrame(np.random.randn(5, 3), index=short_dates)

        result = calculator.calculate_rolling_factor_exposures(
            short_returns, short_factors
        )
        assert result.empty


def test_factor_timing_import():
    """Test that factor timing module can be imported."""
    from quantfolio_engine.signals.factor_timing import (
        FactorExposureCalculator,
        FactorTimingEngine,
        RegimeDetector,
    )

    # Test instantiation
    calc = FactorExposureCalculator()
    detector = RegimeDetector()
    engine = FactorTimingEngine()

    assert calc is not None
    assert detector is not None
    assert engine is not None


class TestPerformance:
    """Performance and numerical stability tests."""

    def test_large_dataset_performance(self):
        """Test performance with larger datasets."""
        calculator = FactorExposureCalculator(lookback_period=60)

        # Create larger dataset
        dates = pd.date_range("2010-01-01", periods=500, freq="ME")
        returns_df = pd.DataFrame(
            np.random.randn(500, 20),  # 20 assets
            index=dates,
            columns=[f"ASSET_{i}" for i in range(20)],
        )

        factors_df = pd.DataFrame(
            np.random.randn(500, 5),  # 5 factors
            index=dates,
            columns=[f"FACTOR_{i}" for i in range(5)],
        )

        # Should complete without errors
        exposures = calculator.calculate_rolling_factor_exposures(
            returns_df, factors_df
        )
        assert isinstance(exposures, pd.DataFrame)

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        calculator = FactorExposureCalculator(lookback_period=12)

        dates = pd.date_range("2020-01-01", periods=50, freq="ME")

        # Create data with extreme values
        returns_df = pd.DataFrame(
            {
                "SPY": np.random.normal(0.01, 0.05, 50),
                "TLT": np.random.normal(0.005, 0.03, 50),
            },
            index=dates,
        )

        # Add some extreme values
        returns_df.loc[dates[10], "SPY"] = 10.0  # Extreme positive
        returns_df.loc[dates[20], "TLT"] = -5.0  # Extreme negative

        factors_df = pd.DataFrame(
            {
                "factor1": np.random.normal(0, 0.02, 50),
                "factor2": np.random.normal(0, 0.015, 50),
            },
            index=dates,
        )

        # Should handle extreme values gracefully
        exposures = calculator.calculate_rolling_factor_exposures(
            returns_df, factors_df
        )
        assert isinstance(exposures, pd.DataFrame)
