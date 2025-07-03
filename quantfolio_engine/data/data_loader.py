"""
Data loader module for QuantFolio Engine.

This module handles data ingestion from various sources:
- Asset returns from Yahoo Finance
- Macroeconomic indicators from FRED
- Sentiment data from News API
"""

from datetime import datetime
from typing import Optional, Tuple

from fredapi import Fred
from loguru import logger
import numpy as np
import pandas as pd
import yfinance as yf

from quantfolio_engine.config import (
    ASSET_UNIVERSE,
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    FRED_API_KEY,
    MACRO_INDICATORS,
    NEWS_API_KEY,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SENTIMENT_ENTITIES,
    SENTIMENT_TOPICS,
)


class DataLoader:
    """Data loader for fetching and processing financial data."""

    def __init__(self):
        """Initialize the data loader with API clients."""
        self.fred = Fred(api_key=FRED_API_KEY) if FRED_API_KEY else None
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure data directories exist."""
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    def fetch_asset_returns(
        self,
        start_date: str = DEFAULT_START_DATE,
        end_date: Optional[str] = DEFAULT_END_DATE,
        save_raw: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch asset returns from Yahoo Finance.

        Args:
            start_date: Start date for data fetch (YYYY-MM-DD)
            end_date: End date for data fetch (YYYY-MM-DD), None for current date
            save_raw: Whether to save raw data to files

        Returns:
            DataFrame with monthly returns for all assets
        """
        logger.info("Fetching asset returns from Yahoo Finance...")

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        all_data = {}

        for ticker in ASSET_UNIVERSE.keys():
            try:
                logger.info(f"Fetching data for {ticker}...")
                ticker_obj = yf.Ticker(ticker)
                ticker_data = ticker_obj.history(
                    start=start_date, end=end_date, auto_adjust=True
                )

                if not ticker_data.empty:
                    # Use Close price for returns calculation
                    monthly_data = ticker_data["Close"].resample("ME").last()
                    returns = monthly_data.pct_change().dropna()
                    all_data[ticker] = returns

                    # Save raw data if requested
                    if save_raw:
                        raw_file = RAW_DATA_DIR / f"prices_{ticker.lower()}.csv"
                        ticker_data.to_csv(raw_file)
                        logger.info(f"Saved raw data to {raw_file}")

            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}")
                continue

        # Combine all returns into a single DataFrame
        returns_df = pd.DataFrame(all_data)
        returns_df.index.name = "date"

        # Save processed data
        processed_file = PROCESSED_DATA_DIR / "returns_monthly.csv"
        returns_df.to_csv(processed_file)
        logger.info(f"Saved processed returns to {processed_file}")

        return returns_df

    def fetch_macro_indicators(
        self,
        start_date: str = DEFAULT_START_DATE,
        end_date: Optional[str] = DEFAULT_END_DATE,
        save_raw: bool = True,
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
        if not self.fred:
            logger.error("FRED API key not available")
            return pd.DataFrame()

        logger.info("Fetching macroeconomic indicators from FRED...")

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        all_data = {}

        for series_id, info in MACRO_INDICATORS.items():
            if info["source"] == "FRED":
                try:
                    logger.info(f"Fetching {series_id} ({info['name']})...")
                    data = self.fred.get_series(
                        series_id,
                        observation_start=start_date,
                        observation_end=end_date,
                    )

                    if not data.empty:
                        # Resample to monthly frequency
                        if info["name"] == "Real GDP (Quarterly)":
                            # GDP is quarterly, keep as is
                            monthly_data = data
                        else:
                            # Resample to monthly average
                            monthly_data = data.resample("ME").mean()

                        all_data[series_id] = monthly_data

                        # Save raw data if requested
                        if save_raw:
                            raw_file = RAW_DATA_DIR / f"macro_{series_id.lower()}.csv"
                            data.to_csv(raw_file)
                            logger.info(f"Saved raw data to {raw_file}")

                except Exception as e:
                    logger.error(f"Error fetching {series_id}: {e}")
                    continue

        # Handle VIX separately (from Yahoo Finance)
        try:
            logger.info("Fetching VIX from Yahoo Finance...")
            vix_obj = yf.Ticker("^VIX")
            vix_data = vix_obj.history(start=start_date, end=end_date, auto_adjust=True)
            if not vix_data.empty:
                monthly_vix = vix_data["Close"].resample("ME").mean()
                # Convert timezone-aware index to timezone-naive to match other data
                monthly_vix.index = monthly_vix.index.tz_localize(None)
                all_data["^VIX"] = monthly_vix

                if save_raw:
                    raw_file = RAW_DATA_DIR / "macro_vix.csv"
                    vix_data.to_csv(raw_file)
                    logger.info(f"Saved raw VIX data to {raw_file}")
        except Exception as e:
            logger.error(f"Error fetching VIX: {e}")

        # Combine all macro data into a single DataFrame
        macro_df = pd.DataFrame(all_data)
        macro_df.index.name = "date"

        # Save processed data
        processed_file = PROCESSED_DATA_DIR / "macro_monthly.csv"
        macro_df.to_csv(processed_file)
        logger.info(f"Saved processed macro data to {processed_file}")

        return macro_df

    def fetch_sentiment_data(
        self,
        start_date: str = DEFAULT_START_DATE,
        end_date: Optional[str] = DEFAULT_END_DATE,
        save_raw: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch sentiment data from News API.

        Args:
            start_date: Start date for data fetch (YYYY-MM-DD)
            end_date: End date for data fetch (YYYY-MM-DD), None for current date
            save_raw: Whether to save raw data to files

        Returns:
            DataFrame with monthly sentiment scores
        """
        if not NEWS_API_KEY:
            logger.warning(
                "News API key not available, generating placeholder sentiment data"
            )
            return self._generate_placeholder_sentiment(start_date, end_date, save_raw)

        logger.info("Fetching sentiment data from News API...")

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # Convert dates to datetime for processing
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        all_sentiment = {}

        # Fetch sentiment for entities
        for entity in SENTIMENT_ENTITIES:
            try:
                logger.info(f"Fetching sentiment for {entity}...")
                entity_sentiment = self._fetch_entity_sentiment(
                    entity, start_dt, end_dt
                )
                if entity_sentiment is not None:
                    all_sentiment[f"sentiment_{entity}"] = entity_sentiment

                    if save_raw:
                        raw_file = RAW_DATA_DIR / f"sentiment_{entity.lower()}.csv"
                        entity_sentiment.to_csv(raw_file)
                        logger.info(f"Saved raw sentiment data to {raw_file}")

            except Exception as e:
                logger.error(f"Error fetching sentiment for {entity}: {e}")
                continue

        # Fetch sentiment for topics
        for topic in SENTIMENT_TOPICS:
            try:
                logger.info(f"Fetching sentiment for topic: {topic}...")
                topic_sentiment = self._fetch_topic_sentiment(topic, start_dt, end_dt)
                if topic_sentiment is not None:
                    all_sentiment[f"sentiment_{topic.replace(' ', '_')}"] = (
                        topic_sentiment
                    )

                    if save_raw:
                        raw_file = (
                            RAW_DATA_DIR
                            / f"sentiment_{topic.replace(' ', '_').lower()}.csv"
                        )
                        topic_sentiment.to_csv(raw_file)
                        logger.info(f"Saved raw sentiment data to {raw_file}")

            except Exception as e:
                logger.error(f"Error fetching sentiment for topic {topic}: {e}")
                continue

        # Combine all sentiment data into a single DataFrame
        sentiment_df = pd.DataFrame(all_sentiment)
        sentiment_df.index.name = "date"

        # Save processed data
        processed_file = PROCESSED_DATA_DIR / "sentiment_monthly.csv"
        sentiment_df.to_csv(processed_file)
        logger.info(f"Saved processed sentiment data to {processed_file}")

        return sentiment_df

    def _generate_placeholder_sentiment(
        self,
        start_date: str = DEFAULT_START_DATE,
        end_date: Optional[str] = DEFAULT_END_DATE,
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

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # Convert dates to datetime for processing
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Generate monthly date range
        date_range = pd.date_range(start=start_dt, end=end_dt, freq="ME")

        all_sentiment = {}

        # Generate placeholder sentiment for entities
        for entity in SENTIMENT_ENTITIES:
            # Generate random sentiment scores between -1 and 1
            sentiment_scores = pd.Series(
                index=date_range, data=np.random.uniform(-1, 1, len(date_range))
            )
            all_sentiment[f"sentiment_{entity}"] = sentiment_scores

            if save_raw:
                raw_file = RAW_DATA_DIR / f"sentiment_{entity.lower()}.csv"
                sentiment_scores.to_csv(raw_file)
                logger.info(f"Saved placeholder sentiment data to {raw_file}")

        # Generate placeholder sentiment for topics
        for topic in SENTIMENT_TOPICS:
            # Generate random sentiment scores between -1 and 1
            sentiment_scores = pd.Series(
                index=date_range, data=np.random.uniform(-1, 1, len(date_range))
            )
            all_sentiment[f"sentiment_{topic.replace(' ', '_')}"] = sentiment_scores

            if save_raw:
                raw_file = (
                    RAW_DATA_DIR / f"sentiment_{topic.replace(' ', '_').lower()}.csv"
                )
                sentiment_scores.to_csv(raw_file)
                logger.info(f"Saved placeholder sentiment data to {raw_file}")

        # Combine all sentiment data into a single DataFrame
        sentiment_df = pd.DataFrame(all_sentiment)
        sentiment_df.index.name = "date"

        # Save processed data
        processed_file = PROCESSED_DATA_DIR / "sentiment_monthly.csv"
        sentiment_df.to_csv(processed_file)
        logger.info(f"Saved processed placeholder sentiment data to {processed_file}")

        return sentiment_df

    def _fetch_entity_sentiment(
        self, entity: str, start_dt: datetime, end_dt: datetime
    ) -> Optional[pd.Series]:
        """Fetch sentiment data for a specific entity."""
        # For now, implement basic sentiment scoring
        # This is a placeholder - in production, you'd use a proper sentiment analysis model

        # Generate monthly sentiment scores (placeholder)
        date_range = pd.date_range(start=start_dt, end=end_dt, freq="ME")
        sentiment_scores = pd.Series(
            index=date_range,
            data=[0.0] * len(date_range),  # Placeholder: neutral sentiment
        )

        return sentiment_scores

    def _fetch_topic_sentiment(
        self, topic: str, start_dt: datetime, end_dt: datetime
    ) -> Optional[pd.Series]:
        """Fetch sentiment data for a specific topic."""
        # Similar placeholder implementation
        date_range = pd.date_range(start=start_dt, end=end_dt, freq="ME")
        sentiment_scores = pd.Series(
            index=date_range,
            data=[0.0] * len(date_range),  # Placeholder: neutral sentiment
        )

        return sentiment_scores

    def load_all_data(
        self,
        start_date: str = DEFAULT_START_DATE,
        end_date: Optional[str] = DEFAULT_END_DATE,
        save_raw: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all data sources.

        Args:
            start_date: Start date for data fetch
            end_date: End date for data fetch
            save_raw: Whether to save raw data

        Returns:
            Tuple of (returns_df, macro_df, sentiment_df)
        """
        logger.info("Loading all data sources...")

        returns_df = self.fetch_asset_returns(start_date, end_date, save_raw)
        macro_df = self.fetch_macro_indicators(start_date, end_date, save_raw)
        sentiment_df = self.fetch_sentiment_data(start_date, end_date, save_raw)

        logger.info("Data loading completed!")

        return returns_df, macro_df, sentiment_df


def main():
    """Main function for testing the data loader."""
    loader = DataLoader()

    # Load all data
    returns, macro, sentiment = loader.load_all_data()

    print(f"Returns data shape: {returns.shape}")
    print(f"Macro data shape: {macro.shape}")
    print(f"Sentiment data shape: {sentiment.shape}")

    if not returns.empty:
        print("\nReturns data head:")
        print(returns.head())

    if not macro.empty:
        print("\nMacro data head:")
        print(macro.head())


if __name__ == "__main__":
    main()
