"""
Data loader module for QuantFolio Engine.

This module handles data ingestion from various sources:
- Asset returns from Yahoo Finance
- Macroeconomic indicators from FRED
- Sentiment data from News API
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional, Protocol, Tuple

from fredapi import Fred
from loguru import logger
import numpy as np
import pandas as pd
import yfinance as yf

from quantfolio_engine.config import (
    DataConfig,
    get_default_data_config,
)


def get_default_end_date() -> str:
    """Get default end date as string."""
    return datetime.now().strftime("%Y-%m-%d")


def set_index_name(df: pd.DataFrame, name: str = "date") -> pd.DataFrame:
    """Set index name consistently."""
    df.index.name = name
    return df


def strip_timezone(df: pd.DataFrame) -> pd.DataFrame:
    """Strip timezone from DataFrame index consistently."""
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


class BaseSentimentProvider(Protocol):
    """Protocol for sentiment data providers."""

    def fetch(self, query: str, start: datetime, end: datetime) -> pd.Series:
        """Fetch sentiment data for a query."""
        ...


class NewsAPISentimentProvider:
    """News API sentiment provider implementation."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def fetch(self, query: str, start: datetime, end: datetime) -> pd.Series:
        """Fetch sentiment data from News API."""
        # Placeholder implementation - replace with actual News API call
        date_range = pd.date_range(start=start, end=end, freq="ME")
        sentiment_scores = pd.Series(
            index=date_range, data=np.random.uniform(-1, 1, len(date_range))
        )
        return sentiment_scores


class RandomSentimentProvider:
    """Random sentiment provider for testing."""

    def fetch(self, query: str, start: datetime, end: datetime) -> pd.Series:
        """Generate random sentiment data."""
        date_range = pd.date_range(start=start, end=end, freq="ME")
        sentiment_scores = pd.Series(
            index=date_range, data=np.random.uniform(-1, 1, len(date_range))
        )
        return sentiment_scores


class DataLoader:
    """Data loader for fetching and processing financial data."""

    def __init__(
        self,
        config: Optional[DataConfig] = None,
        yf_client=None,
        fred_client=None,
        sentiment_client: Optional[BaseSentimentProvider] = None,
    ):
        """
        Initialize the data loader with optional dependency injection.

        Args:
            config: Data configuration
            yf_client: Yahoo Finance client (defaults to yfinance)
            fred_client: FRED API client (defaults to Fred)
            sentiment_client: Sentiment provider (defaults to RandomSentimentProvider)
        """
        self.config = config or get_default_data_config()
        self.yf_client = yf_client or yf
        self.fred_client = fred_client or Fred(api_key=self.config.fred_api_key)
        self.sentiment_client = sentiment_client or RandomSentimentProvider()

        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure data directories exist."""
        self.config.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.config.processed_data_dir.mkdir(parents=True, exist_ok=True)

    def fetch_asset_returns(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        save_raw: Optional[bool] = None,
    ) -> pd.DataFrame:
        """
        Fetch asset returns from Yahoo Finance using batch download.

        Args:
            start_date: Start date for data fetch (YYYY-MM-DD)
            end_date: End date for data fetch (YYYY-MM-DD), None for current date
            save_raw: Whether to save raw data to files

        Returns:
            DataFrame with monthly returns for all assets
        """
        start_date = start_date or self.config.start_date
        end_date = end_date or get_default_end_date()
        save_raw = save_raw if save_raw is not None else self.config.save_raw

        # Early return if asset universe is empty
        if not self.config.asset_universe:
            return pd.DataFrame()

        logger.info("Fetching asset returns from Yahoo Finance...")

        # Use batch download for better performance
        try:
            logger.info("Using batch download for better performance...")
            tickers = list(self.config.asset_universe.keys())
            prices = self.yf_client.download(
                tickers,
                start=start_date,
                end=end_date,
                group_by="ticker",
                auto_adjust=True,
                threads=True,
            )

            all_data = {}
            raw_data_dict = {}

            # Process each ticker
            for ticker in tickers:
                try:
                    if len(tickers) == 1:
                        # Single ticker case
                        ticker_data = prices
                        ticker_name = tickers[0]
                    else:
                        # Multi-ticker case
                        ticker_data = prices[ticker]
                        ticker_name = ticker

                    if not ticker_data.empty:
                        # Strip timezone consistently
                        ticker_data = strip_timezone(ticker_data)

                        # Use Close price for returns calculation
                        monthly_data = ticker_data["Close"].resample("ME").last()
                        returns = monthly_data.pct_change().dropna()
                        all_data[ticker_name] = returns

                        # Store raw data for batch saving
                        if save_raw:
                            raw_data_dict[ticker_name] = ticker_data

                except Exception as e:
                    logger.error(f"Error processing data for {ticker}: {e}")
                    continue

            # Batch save raw data
            if save_raw and raw_data_dict:
                self._batch_save_raw_data(raw_data_dict, "prices")

        except Exception as e:
            logger.warning(
                f"Batch download failed: {e}, falling back to individual downloads..."
            )
            # Fallback to individual downloads
            all_data = self._fetch_asset_returns_individual(
                start_date, end_date, save_raw
            )

        # Combine all returns into a single DataFrame
        returns_df = pd.DataFrame(all_data)
        returns_df = set_index_name(returns_df)

        # Save processed data
        returns_df = strip_timezone(returns_df)
        self.save_processed_data(returns_df, "returns_monthly")
        logger.info("Saved processed returns data")

        return returns_df

    def _fetch_asset_returns_individual(
        self,
        start_date: str,
        end_date: str,
        save_raw: bool = True,
    ) -> dict:
        """Fallback method for individual asset downloads."""
        all_data = {}
        raw_data_dict = {}

        # Use ThreadPoolExecutor for concurrent downloads
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(
                    self._fetch_single_asset, ticker, start_date, end_date
                ): ticker
                for ticker in self.config.asset_universe.keys()
            }

            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        all_data[ticker] = result["returns"]
                        if save_raw and result["raw_data"] is not None:
                            raw_data_dict[ticker] = result["raw_data"]
                except Exception as e:
                    logger.error(f"Error fetching data for {ticker}: {e}")
                    continue

        # Batch save raw data
        if save_raw and raw_data_dict:
            self._batch_save_raw_data(raw_data_dict, "prices")

        return all_data

    def _fetch_single_asset(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> Optional[dict]:
        """Fetch data for a single asset."""
        try:
            logger.debug(f"Fetching data for {ticker}...")
            ticker_obj = self.yf_client.Ticker(ticker)
            ticker_data = ticker_obj.history(
                start=start_date, end=end_date, auto_adjust=True
            )

            if not ticker_data.empty:
                # Strip timezone consistently
                ticker_data = strip_timezone(ticker_data)

                # Use Close price for returns calculation
                monthly_data = ticker_data["Close"].resample("ME").last()
                returns = monthly_data.pct_change().dropna()

                return {"returns": returns, "raw_data": ticker_data}

        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return None
        return None

    def _batch_save_raw_data(self, data_dict: dict, prefix: str):
        """Batch save raw data files."""
        for name, data in data_dict.items():
            try:
                # Save as CSV (legacy)
                csv_file = self.config.raw_data_dir / f"{prefix}_{name.lower()}.csv"
                data.to_csv(csv_file)
                logger.debug(f"Saved raw data to {csv_file}")

                # Save as parquet (more efficient)
                parquet_file = (
                    self.config.raw_data_dir / f"{prefix}_{name.lower()}.parquet"
                )

                # Handle both Series and DataFrame for parquet saving
                if isinstance(data, pd.Series):
                    # Convert Series to DataFrame for parquet saving
                    data_df = data.to_frame(name=name)
                    data_df.to_parquet(parquet_file)
                else:
                    # DataFrame can be saved directly
                    data.to_parquet(parquet_file)

                logger.debug(f"Saved raw data to {parquet_file}")
            except Exception as e:
                logger.error(f"Error saving raw data for {name}: {e}")

    def fetch_macro_indicators(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        save_raw: Optional[bool] = None,
    ) -> pd.DataFrame:
        """
        Fetch macroeconomic indicators from FRED.

        Args:
            start_date: Start date for data fetch (YYYY-MM-DD)
            end_date: End date for data fetch (YYYY-MM-DD), None for current date
            save_raw: Whether to save raw data to files

        Returns:
            DataFrame with monthly macro indicators
        """
        start_date = start_date or self.config.start_date
        end_date = end_date or get_default_end_date()
        save_raw = save_raw if save_raw is not None else self.config.save_raw

        # Early return if macro indicators is empty
        if not self.config.macro_indicators:
            return pd.DataFrame()

        if not self.fred_client:
            logger.error("FRED API key not available")
            return pd.DataFrame()

        logger.info("Fetching macroeconomic indicators from FRED...")

        all_data = {}
        raw_data_dict = {}

        # Use ThreadPoolExecutor for concurrent FRED downloads
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {}

            for series_id, info in self.config.macro_indicators.items():
                if info["source"] == "FRED":
                    future = executor.submit(
                        self._fetch_single_macro_series,
                        series_id,
                        info,
                        start_date,
                        end_date,
                    )
                    futures[future] = series_id

            for future in as_completed(futures):
                series_id = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        all_data[series_id] = result["monthly_data"]
                        if save_raw and result["raw_data"] is not None:
                            raw_data_dict[series_id] = result["raw_data"]
                except Exception as e:
                    logger.error(f"Error fetching {series_id}: {e}")
                    continue

        # Handle VIX separately (from Yahoo Finance)
        try:
            logger.info("Fetching VIX from Yahoo Finance...")
            vix_obj = self.yf_client.Ticker("^VIX")
            vix_data = vix_obj.history(start=start_date, end=end_date, auto_adjust=True)
            if not vix_data.empty:
                # Strip timezone consistently
                vix_data = strip_timezone(vix_data)
                monthly_vix = vix_data["Close"].resample("ME").mean()
                all_data["^VIX"] = monthly_vix

                if save_raw:
                    raw_data_dict["^VIX"] = vix_data
        except Exception as e:
            logger.error(f"Error fetching VIX: {e}")

        # Batch save raw data
        if save_raw and raw_data_dict:
            self._batch_save_raw_data(raw_data_dict, "macro")

        # Combine all macro data into a single DataFrame
        macro_df = pd.DataFrame(all_data)
        macro_df = set_index_name(macro_df)

        # Save processed data
        macro_df = strip_timezone(macro_df)
        self.save_processed_data(macro_df, "macro_monthly")
        logger.info("Saved processed macro data")

        return macro_df

    def _fetch_single_macro_series(
        self,
        series_id: str,
        info: dict,
        start_date: str,
        end_date: str,
    ) -> Optional[dict]:
        """Fetch a single macro series from FRED."""
        try:
            logger.debug(f"Fetching {series_id} ({info['name']})...")
            data = self.fred_client.get_series(
                series_id,
                observation_start=start_date,
                observation_end=end_date,
            )

            if not data.empty:
                # Strip timezone consistently
                data = strip_timezone(data)

                # Resample to monthly frequency
                if info["name"] == "Real GDP (Quarterly)":
                    # GDP is quarterly, forward-fill within quarter
                    monthly_data = data.resample("ME").ffill()
                else:
                    # Resample to monthly average
                    monthly_data = data.resample("ME").mean()

                return {"monthly_data": monthly_data, "raw_data": data}

        except Exception as e:
            logger.error(f"Error fetching {series_id}: {e}")
            return None
        return None

    def fetch_sentiment_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        save_raw: Optional[bool] = None,
    ) -> pd.DataFrame:
        """
        Fetch sentiment data from sentiment provider.

        Args:
            start_date: Start date for data fetch (YYYY-MM-DD)
            end_date: End date for data fetch (YYYY-MM-DD), None for current date
            save_raw: Whether to save raw data to files

        Returns:
            DataFrame with monthly sentiment scores
        """
        start_date = start_date or self.config.start_date
        end_date = end_date or get_default_end_date()
        save_raw = save_raw if save_raw is not None else self.config.save_raw

        # Early return if no sentiment entities or topics
        if not self.config.sentiment_entities and not self.config.sentiment_topics:
            return pd.DataFrame()

        if not self.config.news_api_key:
            logger.warning("News API key not available, using sentiment provider")
            return self._generate_placeholder_sentiment(start_date, end_date, save_raw)

        logger.info("Fetching sentiment data from sentiment provider...")

        # Convert dates to datetime for processing
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        all_sentiment = {}
        raw_data_dict = {}

        # Use ThreadPoolExecutor for concurrent sentiment downloads
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {}

            # Fetch sentiment for entities
            for entity in self.config.sentiment_entities:
                future = executor.submit(
                    self._fetch_entity_sentiment, entity, start_dt, end_dt
                )
                futures[future] = f"entity_{entity}"

            # Fetch sentiment for topics
            for topic in self.config.sentiment_topics:
                future = executor.submit(
                    self._fetch_topic_sentiment, topic, start_dt, end_dt
                )
                futures[future] = f"topic_{topic}"

            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        if name.startswith("entity_"):
                            entity = name.replace("entity_", "")
                            all_sentiment[f"sentiment_{entity}"] = result
                            if save_raw:
                                raw_data_dict[f"sentiment_{entity}"] = result
                        elif name.startswith("topic_"):
                            topic = name.replace("topic_", "")
                            all_sentiment[f"sentiment_{topic.replace(' ', '_')}"] = (
                                result
                            )
                            if save_raw:
                                raw_data_dict[
                                    f"sentiment_{topic.replace(' ', '_')}"
                                ] = result
                except Exception as e:
                    logger.error(f"Error fetching sentiment for {name}: {e}")
                    continue

        # Batch save raw data
        if save_raw and raw_data_dict:
            self._batch_save_raw_data(raw_data_dict, "sentiment")

        # Combine all sentiment data into a single DataFrame
        sentiment_df = pd.DataFrame(all_sentiment)
        sentiment_df = set_index_name(sentiment_df)

        # Save processed data
        sentiment_df = strip_timezone(sentiment_df)
        self.save_processed_data(sentiment_df, "sentiment_monthly")
        logger.info("Saved processed sentiment data")

        return sentiment_df

    def _generate_placeholder_sentiment(
        self,
        start_date: str,
        end_date: str,
        save_raw: bool = True,
    ) -> pd.DataFrame:
        """
        Generate placeholder sentiment data when API is not available.

        Args:
            start_date: Start date for data generation (YYYY-MM-DD)
            end_date: End date for data generation (YYYY-MM-DD), None for current date
            save_raw: Whether to save raw data to files

        Returns:
            DataFrame with monthly placeholder sentiment scores
        """
        logger.info("Generating placeholder sentiment data...")

        # Convert dates to datetime for processing
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Generate monthly date range
        date_range = pd.date_range(start=start_dt, end=end_dt, freq="ME")

        all_sentiment = {}
        raw_data_dict = {}

        # Generate placeholder sentiment for entities
        for entity in self.config.sentiment_entities:
            # Generate random sentiment scores between -1 and 1
            sentiment_scores = pd.Series(
                index=date_range, data=np.random.uniform(-1, 1, len(date_range))
            )
            all_sentiment[f"sentiment_{entity}"] = sentiment_scores
            if save_raw:
                raw_data_dict[f"sentiment_{entity}"] = sentiment_scores

        # Generate placeholder sentiment for topics
        for topic in self.config.sentiment_topics:
            # Generate random sentiment scores between -1 and 1
            sentiment_scores = pd.Series(
                index=date_range, data=np.random.uniform(-1, 1, len(date_range))
            )
            all_sentiment[f"sentiment_{topic.replace(' ', '_')}"] = sentiment_scores
            if save_raw:
                raw_data_dict[f"sentiment_{topic.replace(' ', '_')}"] = sentiment_scores

        # Batch save raw data
        if save_raw and raw_data_dict:
            self._batch_save_raw_data(raw_data_dict, "sentiment")

        # Combine all sentiment data into a single DataFrame
        sentiment_df = pd.DataFrame(all_sentiment)
        sentiment_df = set_index_name(sentiment_df)

        # Save processed data
        self.save_processed_data(sentiment_df, "sentiment_monthly")
        logger.info("Saved processed placeholder sentiment data")

        return sentiment_df

    def _fetch_entity_sentiment(
        self, entity: str, start_dt: datetime, end_dt: datetime
    ) -> Optional[pd.Series]:
        """
        Fetch sentiment data for a specific entity.

        Args:
            entity: Entity name to fetch sentiment for
            start_dt: Start datetime
            end_dt: End datetime

        Returns:
            Sentiment series or None if failed
        """
        try:
            return self.sentiment_client.fetch(entity, start_dt, end_dt)
        except Exception as e:
            logger.error(f"Error fetching entity sentiment for {entity}: {e}")
            return None

    def _fetch_topic_sentiment(
        self, topic: str, start_dt: datetime, end_dt: datetime
    ) -> Optional[pd.Series]:
        """
        Fetch sentiment data for a specific topic.

        Args:
            topic: Topic name to fetch sentiment for
            start_dt: Start datetime
            end_dt: End datetime

        Returns:
            Sentiment series or None if failed
        """
        try:
            return self.sentiment_client.fetch(topic, start_dt, end_dt)
        except Exception as e:
            logger.error(f"Error fetching topic sentiment for {topic}: {e}")
            return None

    def load_all_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        save_raw: Optional[bool] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all data types (returns, macro, sentiment).

        Args:
            start_date: Start date for data fetch (YYYY-MM-DD)
            end_date: End date for data fetch (YYYY-MM-DD), None for current date
            save_raw: Whether to save raw data to files

        Returns:
            Tuple of (returns_df, macro_df, sentiment_df)
        """
        logger.info("Loading all data types...")

        # Fetch all data types
        returns_df = self.fetch_asset_returns(start_date, end_date, save_raw)
        macro_df = self.fetch_macro_indicators(start_date, end_date, save_raw)
        sentiment_df = self.fetch_sentiment_data(start_date, end_date, save_raw)

        return returns_df, macro_df, sentiment_df

    def normalize_returns(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize returns data using z-score normalization.

        Args:
            returns_df: Returns DataFrame

        Returns:
            Normalized returns DataFrame
        """
        # Z-score normalization for returns
        normalized = (returns_df - returns_df.mean()) / returns_df.std()
        return normalized

    def normalize_macro(self, macro_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize macro data with different strategies for different data types.

        Args:
            macro_df: Macro DataFrame

        Returns:
            Normalized macro DataFrame
        """
        # Separate rate-like and level-like series
        rate_like_patterns = ["CPI", "GDPDEF", "CPIAUCSL"]
        rate_like_cols = [
            col
            for col in macro_df.columns
            if any(pattern in col for pattern in rate_like_patterns)
        ]

        level_like_cols = [col for col in macro_df.columns if col not in rate_like_cols]

        normalized_df = pd.DataFrame(index=macro_df.index)

        # Handle rate-like series (convert to percentage change)
        if rate_like_cols:
            rate_like_data = macro_df[rate_like_cols].pct_change()
            normalized_df[rate_like_cols] = rate_like_data

        # Handle level-like series (z-score normalization)
        if level_like_cols:
            level_like_data = macro_df[level_like_cols]
            normalized_level = (
                level_like_data - level_like_data.mean()
            ) / level_like_data.std()
            normalized_df[level_like_cols] = normalized_level

        return normalized_df

    def normalize_sentiment(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize sentiment data with bounds to (-1, 1).

        Args:
            sentiment_df: Sentiment DataFrame

        Returns:
            Normalized sentiment DataFrame bounded to (-1, 1)
        """
        # Z-score normalization first
        normalized = (sentiment_df - sentiment_df.mean()) / sentiment_df.std()

        # Clip to sentiment bounds (-1, 1)
        bounded = normalized.clip(-1, 1)

        return bounded

    def save_processed_data(
        self, df: pd.DataFrame, filename: str, use_parquet: bool = True
    ):
        """
        Save processed data in CSV and/or parquet format.

        Args:
            df: DataFrame to save
            filename: Base filename without extension
            use_parquet: Whether to save as parquet (default True)
        """
        try:
            # Always save as CSV for compatibility
            csv_file = self.config.processed_data_dir / f"{filename}.csv"
            df.to_csv(csv_file)
            logger.debug(f"Saved processed data to {csv_file}")

            # Save as parquet if requested
            if use_parquet:
                parquet_file = self.config.processed_data_dir / f"{filename}.parquet"
                df.to_parquet(parquet_file)
                logger.debug(f"Saved processed data to {parquet_file}")

        except Exception as e:
            logger.error(f"Error saving processed data {filename}: {e}")

    def load_processed_data(
        self, filename: str, use_parquet: bool = True
    ) -> pd.DataFrame:
        """
        Load processed data from CSV or parquet format.

        Args:
            filename: Base filename without extension
            use_parquet: Whether to try loading parquet first (default True)

        Returns:
            Loaded DataFrame
        """
        try:
            if use_parquet:
                parquet_file = self.config.processed_data_dir / f"{filename}.parquet"
                if parquet_file.exists():
                    df = pd.read_parquet(parquet_file)
                    logger.debug(f"Loaded data from {parquet_file}")
                    return df

            # Fallback to CSV
            csv_file = self.config.processed_data_dir / f"{filename}.csv"
            if csv_file.exists():
                df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
                logger.debug(f"Loaded data from {csv_file}")
                return df
            else:
                logger.warning(f"No data file found for {filename}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error loading processed data {filename}: {e}")
            return pd.DataFrame()


def main():
    """Main function for testing the data loader."""
    loader = DataLoader()
    returns_df, macro_df, sentiment_df = loader.load_all_data()
    logger.info(
        f"Loaded {len(returns_df)} returns, {len(macro_df)} macro, {len(sentiment_df)} sentiment records"
    )


if __name__ == "__main__":
    main()
