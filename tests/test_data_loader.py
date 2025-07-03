"""Tests for the data loader module."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from quantfolio_engine.config import ASSET_UNIVERSE, MACRO_INDICATORS
from quantfolio_engine.data.data_loader import DataLoader


class TestDataLoader:
    """Test cases for DataLoader class."""

    def test_init(self):
        """Test DataLoader initialization."""
        loader = DataLoader()
        assert loader is not None
        # Should create directories
        from quantfolio_engine.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

        assert RAW_DATA_DIR.exists()
        assert PROCESSED_DATA_DIR.exists()

    def test_asset_universe_config(self):
        """Test that asset universe is properly configured."""
        assert len(ASSET_UNIVERSE) > 0
        assert "SPY" in ASSET_UNIVERSE
        assert "AAPL" in ASSET_UNIVERSE
        assert ASSET_UNIVERSE["SPY"]["type"] == "Broad Index"

    def test_macro_indicators_config(self):
        """Test that macro indicators are properly configured."""
        assert len(MACRO_INDICATORS) > 0
        assert "CPIAUCSL" in MACRO_INDICATORS
        assert MACRO_INDICATORS["CPIAUCSL"]["source"] == "FRED"

    @patch("quantfolio_engine.data.data_loader.yf.download")
    def test_fetch_asset_returns_mock(self, mock_download):
        """Test asset returns fetching with mocked data."""
        # Mock yfinance response
        mock_data = pd.DataFrame(
            {
                "Adj Close": [
                    100,
                    101,
                    102,
                    103,
                    104,
                    105,
                    106,
                    107,
                    108,
                    109,
                    110,
                    111,
                ],
                "Open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                "High": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                "Low": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                "Close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                "Volume": [1000] * 12,
            },
            index=pd.date_range("2023-01-01", periods=12, freq="D"),
        )

        mock_download.return_value = mock_data

        loader = DataLoader()
        result = loader.fetch_asset_returns(
            start_date="2023-01-01", end_date="2023-01-12", save_raw=False
        )

        # Should return a DataFrame with returns
        assert isinstance(result, pd.DataFrame)
        # With monthly resampling and pct_change, we should have at least some data
        # (the exact number depends on how pandas handles the resampling)
        assert len(result.columns) > 0  # Check that we have columns (assets)

    @patch("quantfolio_engine.data.data_loader.Fred")
    def test_fetch_macro_indicators_mock(self, mock_fred):
        """Test macro indicators fetching with mocked data."""
        # Mock FRED response
        mock_series = pd.Series(
            [100, 101, 102, 103, 104],
            index=pd.date_range("2023-01-01", periods=5, freq="D"),
        )

        mock_fred_instance = Mock()
        mock_fred_instance.get_series.return_value = mock_series
        mock_fred.return_value = mock_fred_instance

        loader = DataLoader()
        # Mock the fred attribute
        loader.fred = mock_fred_instance

        result = loader.fetch_macro_indicators(
            start_date="2023-01-01", end_date="2023-01-05", save_raw=False
        )

        # Should return a DataFrame with macro data
        assert isinstance(result, pd.DataFrame)

    def test_fetch_sentiment_data_no_api_key(self):
        """Test sentiment data fetching without API key."""
        loader = DataLoader()
        result = loader.fetch_sentiment_data(save_raw=False)

        # Should return DataFrame with placeholder data when no API key
        assert isinstance(result, pd.DataFrame)
        assert not result.empty  # Should have placeholder data
        assert len(result.columns) > 0  # Should have sentiment columns

    def test_load_all_data(self):
        """Test loading all data sources."""
        with (
            patch.object(DataLoader, "fetch_asset_returns") as mock_returns,
            patch.object(DataLoader, "fetch_macro_indicators") as mock_macro,
            patch.object(DataLoader, "fetch_sentiment_data") as mock_sentiment,
        ):

            # Mock return values
            mock_returns.return_value = pd.DataFrame({"SPY": [0.01, 0.02]})
            mock_macro.return_value = pd.DataFrame({"CPI": [100, 101]})
            mock_sentiment.return_value = pd.DataFrame({"sentiment": [0.5, 0.6]})

            loader = DataLoader()
            returns, macro, sentiment = loader.load_all_data(save_raw=False)

            assert isinstance(returns, pd.DataFrame)
            assert isinstance(macro, pd.DataFrame)
            assert isinstance(sentiment, pd.DataFrame)

            # Verify methods were called
            mock_returns.assert_called_once()
            mock_macro.assert_called_once()
            mock_sentiment.assert_called_once()

    def test_normalization(self):
        """Test normalization utilities for returns, macro, and sentiment."""
        loader = DataLoader()
        # Create dummy data
        df = pd.DataFrame(
            {"A": np.random.normal(10, 2, 100), "B": np.random.normal(-5, 5, 100)},
            index=pd.date_range("2020-01-01", periods=100),
        )
        # Returns normalization
        norm = loader.normalize_returns(df)
        assert np.allclose(norm.mean(skipna=True), 0, atol=1e-1)
        assert np.allclose(norm.std(skipna=True, ddof=0), 1, atol=1e-1)
        # Macro normalization
        norm = loader.normalize_macro(df)
        assert np.allclose(norm.mean(skipna=True), 0, atol=1e-1)
        assert np.allclose(norm.std(skipna=True, ddof=0), 1, atol=1e-1)
        # Sentiment normalization
        norm = loader.normalize_sentiment(df)
        assert np.allclose(norm.mean(skipna=True), 0, atol=1e-1)
        assert np.allclose(norm.std(skipna=True, ddof=0), 1, atol=1e-1)


def test_data_loader_import():
    """Test that DataLoader can be imported."""
    from quantfolio_engine.data.data_loader import DataLoader

    assert DataLoader is not None


def test_config_imports():
    """Test that config values can be imported."""
    from quantfolio_engine.config import (
        ASSET_UNIVERSE,
        MACRO_INDICATORS,
        SENTIMENT_ENTITIES,
        SENTIMENT_TOPICS,
    )

    assert ASSET_UNIVERSE is not None
    assert MACRO_INDICATORS is not None
    assert SENTIMENT_ENTITIES is not None
    assert SENTIMENT_TOPICS is not None
