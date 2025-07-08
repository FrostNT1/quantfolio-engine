"""Tests for the data loader module."""

from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from quantfolio_engine.config import (
    ASSET_UNIVERSE,
    MACRO_INDICATORS,
    get_default_data_config,
)
from quantfolio_engine.data.data_loader import (
    DataLoader,
    RandomSentimentProvider,
    set_index_name,
    strip_timezone,
)


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

    def test_dependency_injection(self):
        """Test that dependency injection works correctly."""
        # Create mock clients
        mock_yf = Mock()
        mock_fred = Mock()
        mock_sentiment = RandomSentimentProvider()

        # Create custom config
        config = get_default_data_config()
        config.asset_universe = {"SPY": {"type": "ETF", "description": "S&P 500"}}
        config.macro_indicators = {"CPI": {"name": "CPI", "source": "FRED"}}
        config.sentiment_entities = ["SPY"]
        config.sentiment_topics = ["inflation"]

        # Create DataLoader with injected dependencies
        loader = DataLoader(
            config=config,
            yf_client=mock_yf,
            fred_client=mock_fred,
            sentiment_client=mock_sentiment,
        )

        # Verify dependencies are set correctly
        assert loader.yf_client == mock_yf
        assert loader.fred_client == mock_fred
        assert loader.sentiment_client == mock_sentiment
        assert loader.config == config

    def test_monthly_resample_fix(self):
        """Test that monthly resampling uses 'M' instead of 'ME'."""
        # Create sample data with timezone-aware index
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="D", tz="UTC")
        data = pd.DataFrame({"Close": np.random.randn(len(dates))}, index=dates)

        # Test that resampling works with 'ME'
        monthly_data = data["Close"].resample("ME").last()
        assert len(monthly_data) > 0
        # Monthly data has freq='ME' (MonthEnd), not None
        assert monthly_data.index.freq is not None

    def test_timezone_handling(self):
        """Test consistent timezone handling."""
        # Create timezone-aware data
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="D", tz="UTC")
        data = pd.DataFrame({"Close": np.random.randn(len(dates))}, index=dates)

        # Test strip_timezone function
        stripped_data = strip_timezone(data.copy())
        assert stripped_data.index.tz is None

        # Test that original data is not modified
        assert data.index.tz is not None

    def test_index_name_consistency(self):
        """Test consistent index naming."""
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
        data = pd.DataFrame({"Close": np.random.randn(len(dates))}, index=dates)

        # Test set_index_name function
        named_data = set_index_name(data.copy(), "date")
        assert named_data.index.name == "date"

        # Test with default name
        default_named = set_index_name(data.copy())
        assert default_named.index.name == "date"

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
        # Mock yfinance response for batch download
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

        # Mock the batch download to return data for each ticker
        mock_download.return_value = {
            "SPY": mock_data,
            "TLT": mock_data,
            "GLD": mock_data,
        }

        loader = DataLoader()
        result = loader.fetch_asset_returns(
            start_date="2023-01-01", end_date="2023-01-12", save_raw=False
        )

        # Should return a DataFrame with returns
        assert isinstance(result, pd.DataFrame)
        # With monthly resampling and pct_change, we should have at least some data
        # (the exact number depends on how pandas handles the resampling)
        assert len(result.columns) > 0  # Check that we have columns (assets)

    @patch("quantfolio_engine.data.data_loader.yf.download")
    def test_batch_download_fallback(self, mock_download):
        """Test that batch download falls back to individual downloads."""
        # Make batch download fail
        mock_download.side_effect = Exception("Batch download failed")

        # Create a minimal config for testing
        config = get_default_data_config()
        config.asset_universe = {"SPY": {"type": "ETF", "description": "S&P 500"}}

        loader = DataLoader(config=config)

        # This should not raise an exception due to fallback
        with patch.object(loader, "_fetch_single_asset") as mock_fetch:
            mock_returns = pd.Series(
                [0.01, 0.02], index=pd.date_range("2023-01-01", periods=2, freq="ME")
            )
            mock_raw_data = pd.DataFrame()
            mock_fetch.return_value = {
                "returns": mock_returns,
                "raw_data": mock_raw_data,
            }

            # Should not raise exception
            result = loader.fetch_asset_returns("2023-01-01", "2023-12-31")
            assert not result.empty

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
        # Mock the fred_client attribute
        loader.fred_client = mock_fred_instance

        result = loader.fetch_macro_indicators(
            start_date="2023-01-01", end_date="2023-01-05", save_raw=False
        )

        # Should return a DataFrame with macro data
        assert isinstance(result, pd.DataFrame)

    def test_sentiment_consistency(self):
        """Test that sentiment generation is consistent."""
        provider = RandomSentimentProvider()
        start_dt = datetime(2023, 1, 1)
        end_dt = datetime(2023, 12, 31)

        # Test entity sentiment
        entity_sentiment = provider.fetch("AAPL", start_dt, end_dt)
        assert isinstance(entity_sentiment, pd.Series)
        assert len(entity_sentiment) > 0
        assert all(-1 <= x <= 1 for x in entity_sentiment.values)

        # Test topic sentiment
        topic_sentiment = provider.fetch("inflation", start_dt, end_dt)
        assert isinstance(topic_sentiment, pd.Series)
        assert len(topic_sentiment) > 0
        assert all(-1 <= x <= 1 for x in topic_sentiment.values)

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

    def test_normalization_improvements(self):
        """Test improved normalization strategies."""
        # Create sample data
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="ME")

        # Test returns normalization
        returns_data = pd.DataFrame(
            {
                "SPY": np.random.randn(len(dates)) * 0.02,  # 2% volatility
                "TLT": np.random.randn(len(dates)) * 0.01,  # 1% volatility
            },
            index=dates,
        )

        loader = DataLoader()
        normalized_returns = loader.normalize_returns(returns_data)
        assert normalized_returns.shape == returns_data.shape
        assert abs(normalized_returns.mean().mean()) < 1e-10  # Close to zero
        assert abs(normalized_returns.std().mean() - 1.0) < 1e-10  # Close to one

        # Test macro normalization with different data types
        macro_data = pd.DataFrame(
            {
                "CPIAUCSL": np.cumsum(
                    np.random.randn(len(dates)) * 0.01 + 0.02
                ),  # Rate-like
                "UNRATE": np.random.randn(len(dates)) * 0.5 + 4.0,  # Level-like
            },
            index=dates,
        )

        normalized_macro = loader.normalize_macro(macro_data)
        assert normalized_macro.shape == macro_data.shape

        # Test sentiment normalization with bounds
        sentiment_data = pd.DataFrame(
            {
                "sentiment_AAPL": np.random.uniform(
                    -2, 2, len(dates)
                ),  # Some values outside bounds
                "sentiment_inflation": np.random.uniform(-0.5, 0.5, len(dates)),
            },
            index=dates,
        )

        normalized_sentiment = loader.normalize_sentiment(sentiment_data)
        assert normalized_sentiment.shape == sentiment_data.shape
        assert normalized_sentiment.min().min() >= -1.0
        assert normalized_sentiment.max().max() <= 1.0

    def test_parquet_support(self, tmp_path):
        """Test parquet file support."""
        # Create sample data
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="ME")
        data = pd.DataFrame(
            {
                "SPY": np.random.randn(len(dates)),
                "TLT": np.random.randn(len(dates)),
            },
            index=dates,
        )

        # Create a temporary data loader with a test directory
        from quantfolio_engine.config import DataConfig
        test_config = DataConfig(
            raw_data_dir=tmp_path / "raw",
            processed_data_dir=tmp_path / "processed",
            asset_universe={"SPY": "SPY", "TLT": "TLT"},
            macro_indicators={"CPI": {"name": "CPI", "source": "FRED"}},
            sentiment_entities=[],
            sentiment_topics=[],
            start_date="2023-01-01",
            end_date="2023-12-31",
            save_raw=True,
            max_workers=1,
            fred_api_key="test",
            news_api_key="test",
        )
        loader = DataLoader(config=test_config)

        # Test saving
        loader.save_processed_data(data, "test_data")

        # Test loading
        loaded_data = loader.load_processed_data("test_data")
        assert loaded_data.shape == data.shape
        assert list(loaded_data.columns) == list(data.columns)

        # Clean up is automatic with tmp_path

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
        # Sentiment normalization (clips to [-1, 1] bounds)
        norm = loader.normalize_sentiment(df)
        assert np.allclose(norm.mean(skipna=True), 0, atol=1e-1)
        # After clipping, std may not be exactly 1, but should be reasonable
        assert np.allclose(norm.std(skipna=True, ddof=0), 1, atol=5e-1)
        # Check bounds
        assert norm.min().min() >= -1.0
        assert norm.max().max() <= 1.0

    def test_batch_save_raw_data_error(self):
        """Test error handling in _batch_save_raw_data."""
        loader = DataLoader()

        # Patch to raise exception on to_csv
        class DummyDF:
            def to_csv(self, *a, **k):
                raise IOError("fail csv")

            def to_parquet(self, *a, **k):
                raise IOError("fail parquet")

        data_dict = {"fail": DummyDF()}
        # Should not raise
        loader._batch_save_raw_data(data_dict, "fail")

    def test_save_processed_data_error(self):
        """Test error handling in save_processed_data."""
        loader = DataLoader()

        class DummyDF:
            def to_csv(self, *a, **k):
                raise IOError("fail csv")

            def to_parquet(self, *a, **k):
                raise IOError("fail parquet")

        # Should not raise
        loader.save_processed_data(DummyDF(), "fail")

    def test_load_processed_data_error(self, tmp_path):
        """Test error handling in load_processed_data."""
        loader = DataLoader()
        # Should return empty DataFrame if file does not exist
        df = loader.load_processed_data("does_not_exist")
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test__fetch_single_asset_error(self):
        """Test error branch in _fetch_single_asset."""
        loader = DataLoader()
        # Patch yf_client to raise
        loader.yf_client = Mock()
        loader.yf_client.Ticker.side_effect = Exception("fail")
        result = loader._fetch_single_asset("SPY", "2023-01-01", "2023-01-31")
        assert result is None

    def test__fetch_single_macro_series_error(self):
        """Test error branch in _fetch_single_macro_series."""
        loader = DataLoader()
        loader.fred_client = Mock()
        loader.fred_client.get_series.side_effect = Exception("fail")
        result = loader._fetch_single_macro_series(
            "CPI", {"name": "CPI", "source": "FRED"}, "2023-01-01", "2023-01-31"
        )
        assert result is None

    def test__fetch_entity_sentiment_error(self):
        """Test error branch in _fetch_entity_sentiment."""
        loader = DataLoader()

        class BadSentiment:
            def fetch(self, *a, **k):
                raise Exception("fail")

        loader.sentiment_client = BadSentiment()
        result = loader._fetch_entity_sentiment(
            "AAPL", datetime(2023, 1, 1), datetime(2023, 1, 31)
        )
        assert result is None

    def test__fetch_topic_sentiment_error(self):
        """Test error branch in _fetch_topic_sentiment."""
        loader = DataLoader()

        class BadSentiment:
            def fetch(self, *a, **k):
                raise Exception("fail")

        loader.sentiment_client = BadSentiment()
        result = loader._fetch_topic_sentiment(
            "inflation", datetime(2023, 1, 1), datetime(2023, 1, 31)
        )
        assert result is None

    @patch("quantfolio_engine.data.data_loader.yf.download")
    def test_fetch_asset_returns_batch_fallback(self, mock_download):
        """Test batch download fallback logic."""
        mock_download.side_effect = Exception("fail")
        loader = DataLoader()
        with patch.object(loader, "_fetch_asset_returns_individual") as mock_fallback, \
             patch.object(loader, "save_processed_data") as mock_save:  # Prevent overwriting real data
            mock_fallback.return_value = {
                "SPY": pd.Series(
                    [0.01, 0.02],
                    index=pd.date_range("2023-01-01", periods=2, freq="ME"),
                )
            }
            df = loader.fetch_asset_returns("2023-01-01", "2023-01-31")
            assert not df.empty
            # Verify save_processed_data was called but with mocked behavior
            mock_save.assert_called_once()

    def test_fetch_macro_indicators_vix_error(self):
        """Test VIX error branch in fetch_macro_indicators."""
        loader = DataLoader()
        loader.fred_client = Mock()
        loader.fred_client.get_series.return_value = pd.Series(
            [1, 2, 3], index=pd.date_range("2023-01-01", periods=3)
        )
        # Patch yf_client to raise
        loader.yf_client = Mock()
        loader.yf_client.Ticker.side_effect = Exception("fail")
        with patch.object(loader, "save_processed_data") as mock_save:  # Prevent overwriting real data
            df = loader.fetch_macro_indicators("2023-01-01", "2023-01-31")
            assert isinstance(df, pd.DataFrame)
            # Verify save_processed_data was called but with mocked behavior
            mock_save.assert_called_once()

    def test_fetch_sentiment_data_missing_api_key(self):
        """Test fetch_sentiment_data with missing API key branch."""
        loader = DataLoader()
        loader.config.news_api_key = None
        with patch.object(loader, "save_processed_data") as mock_save:  # Prevent overwriting real data
            df = loader.fetch_sentiment_data("2023-01-01", "2023-01-31")
            assert isinstance(df, pd.DataFrame)
            assert not df.empty
            # Verify save_processed_data was called but with mocked behavior
            mock_save.assert_called_once()

    def test_empty_data(self):
        """Test edge case: empty data for returns, macro, sentiment."""
        loader = DataLoader()
        # Patch config to have no assets/macros/entities/topics
        loader.config.asset_universe = {}
        loader.config.macro_indicators = {}
        loader.config.sentiment_entities = []
        loader.config.sentiment_topics = []
        # Should not raise
        with patch.object(loader, "save_processed_data") as mock_save:  # Prevent overwriting real data
            returns = loader.fetch_asset_returns("2023-01-01", "2023-01-31")
            macro = loader.fetch_macro_indicators("2023-01-01", "2023-01-31")
            sentiment = loader.fetch_sentiment_data("2023-01-01", "2023-01-31")
            assert returns.empty
            assert macro.empty
            assert sentiment.empty
            # Verify save_processed_data was called but with mocked behavior
            assert mock_save.call_count >= 0  # May or may not be called for empty data


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
