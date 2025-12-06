"""
Data loader module for QuantFolio Engine.

This module handles data ingestion from various sources:
- Asset returns from Yahoo Finance
- Macroeconomic indicators from FRED
- Sentiment data from News API
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import os
from pathlib import Path
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
    """News API sentiment provider with VADER sentiment scoring."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/everything"

        # Initialize VADER sentiment analyzer
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            self.analyzer = SentimentIntensityAnalyzer()
            self.vader_available = True
        except ImportError:
            logger.warning(
                "vaderSentiment not available. Install with: pip install vaderSentiment"
            )
            self.analyzer = None
            self.vader_available = False

    def _score_text(self, text: str) -> float:
        """Score text using VADER sentiment analysis."""
        if not text or not self.vader_available:
            return 0.0

        scores = self.analyzer.polarity_scores(text)
        # Return compound score (normalized between -1 and 1)
        return float(scores["compound"])

    def fetch(self, query: str, start: datetime, end: datetime) -> pd.Series:
        """Fetch sentiment data from News API and score with VADER."""
        if not self.api_key:
            logger.warning("No News API key provided, returning neutral sentiment")
            date_range = pd.date_range(start=start, end=end, freq="ME")
            return pd.Series(index=date_range, data=0.0)

        if not self.vader_available:
            logger.warning("VADER not available, returning neutral sentiment")
            date_range = pd.date_range(start=start, end=end, freq="ME")
            return pd.Series(index=date_range, data=0.0)

        # Generate monthly buckets
        months = pd.date_range(start=start, end=end, freq="ME")
        scores = []

        logger.debug(f"Fetching sentiment for '{query}' across {len(months)} months")

        for i in range(len(months)):
            # Determine month boundaries
            if i > 0:
                month_start = months[i - 1] + timedelta(days=1)
            else:
                month_start = start
            month_end = months[i]

            params = {
                "q": query,
                "from": month_start.date().isoformat(),
                "to": month_end.date().isoformat(),
                "language": "en",
                "pageSize": 100,
                "sortBy": "relevancy",
                "apiKey": self.api_key,
            }

            try:
                import requests

                response = requests.get(self.base_url, params=params, timeout=15)
                response.raise_for_status()

                data = response.json()
                articles = data.get("articles", [])

                if not articles:
                    scores.append(0.0)
                    logger.debug(
                        f"No articles for {query} in {month_end.strftime('%Y-%m')}"
                    )
                    continue

                # Score each article
                article_scores = []
                seen_urls = set()

                for article in articles:
                    # Deduplicate by URL
                    url = article.get("url")
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)

                    # Combine title and description for scoring
                    title = article.get("title") or ""
                    description = article.get("description") or ""
                    text = f"{title}. {description}"

                    score = self._score_text(text)
                    article_scores.append(score)

                if article_scores:
                    # Winsorize to handle outliers, then average
                    article_series = pd.Series(article_scores).clip(
                        lower=-0.9, upper=0.9
                    )
                    monthly_sentiment = float(article_series.mean())
                    scores.append(monthly_sentiment)
                    logger.debug(
                        f"{query} {month_end.strftime('%Y-%m')}: {len(article_scores)} articles, "
                        f"sentiment={monthly_sentiment:.3f}"
                    )
                else:
                    scores.append(0.0)

            except ImportError:
                logger.error(
                    "requests library not available. Install with: pip install requests"
                )
                scores.append(0.0)
            except Exception as e:
                logger.warning(
                    f"Error fetching sentiment for '{query}' in {month_end.strftime('%Y-%m')}: {e}"
                )
                scores.append(0.0)

        # Create series with monthly index
        sentiment_series = pd.Series(scores, index=months)
        sentiment_series.index.name = "date"

        logger.debug(
            f"Fetched sentiment for '{query}': mean={sentiment_series.mean():.3f}, "
            f"std={sentiment_series.std():.3f}"
        )

        return sentiment_series


class YahooFinanceSentimentProvider:
    """Yahoo Finance news sentiment provider with VADER scoring."""

    def __init__(self):
        # Initialize VADER sentiment analyzer
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            self.analyzer = SentimentIntensityAnalyzer()
            self.vader_available = True
        except ImportError:
            logger.warning(
                "vaderSentiment not available. Install with: pip install vaderSentiment"
            )
            self.analyzer = None
            self.vader_available = False

    def _score_text(self, text: str) -> float:
        """Score text using VADER sentiment analysis."""
        if not text or not self.vader_available:
            return 0.0

        scores = self.analyzer.polarity_scores(text)
        return float(scores["compound"])

    def fetch(self, query: str, start: datetime, end: datetime) -> pd.Series:
        """
        Fetch sentiment data from Yahoo Finance news.

        Note: Yahoo Finance only provides recent news (~10-50 articles per ticker).
        Historical data is limited to the last 1-2 months.

        Args:
            query: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            start: Start date
            end: End date

        Returns:
            pd.Series with monthly sentiment scores
        """
        if not self.vader_available:
            logger.warning("VADER not available, returning neutral sentiment")
            date_range = pd.date_range(start=start, end=end, freq="ME")
            return pd.Series(index=date_range, data=0.0)

        # Generate monthly buckets
        months = pd.date_range(start=start, end=end, freq="ME")

        try:
            # Fetch news from Yahoo Finance
            ticker = yf.Ticker(query)
            news = ticker.news

            if not news or len(news) == 0:
                logger.warning(
                    f"No news found for {query}, returning neutral sentiment"
                )
                return pd.Series(index=months, data=0.0)

            # Extract articles with dates
            articles_by_month = {month: [] for month in months}

            for article in news:
                # Extract publish date from nested content structure
                content = article.get("content", {})
                pub_date_str = content.get("pubDate")

                if not pub_date_str:
                    # Fallback to top-level providerPublishTime if available
                    pub_time = article.get("providerPublishTime")
                    if pub_time:
                        article_date = pd.Timestamp(pub_time, unit="s").tz_localize(
                            None
                        )
                    else:
                        continue
                else:
                    # Parse ISO format date and convert to timezone-naive
                    article_date = pd.Timestamp(pub_date_str).tz_localize(None)

                # Find the month bucket
                for month in months:
                    if article_date <= month:
                        # Extract text for scoring from nested content
                        title = content.get("title", "")
                        summary = content.get("summary", "")
                        text = f"{title}. {summary}" if summary else title

                        # Score the text
                        score = self._score_text(text)
                        articles_by_month[month].append(score)
                        break

            # Aggregate scores by month
            scores = []
            for month in months:
                month_scores = articles_by_month[month]
                if month_scores:
                    # Winsorize and average
                    monthly_sentiment = float(
                        pd.Series(month_scores).clip(lower=-0.9, upper=0.9).mean()
                    )
                    scores.append(monthly_sentiment)
                    logger.debug(
                        f"{query} {month.strftime('%Y-%m')}: {len(month_scores)} articles, "
                        f"sentiment={monthly_sentiment:.3f}"
                    )
                else:
                    # No news for this month - return neutral
                    scores.append(0.0)

            sentiment_series = pd.Series(scores, index=months)
            sentiment_series.index.name = "date"

            logger.info(
                f"Fetched Yahoo Finance sentiment for '{query}': "
                f"{len([s for s in scores if s != 0.0])}/{len(months)} months with data"
            )

            return sentiment_series

        except Exception as e:
            logger.warning(f"Error fetching Yahoo Finance sentiment for '{query}': {e}")
            return pd.Series(index=months, data=0.0)


class LLMSentimentProvider:
    """
    LLM-based sentiment provider using OpenAI GPT-5-nano with Batch API support.

    Uses OpenAI's Batch API for cost-effective processing of large volumes
    of articles. Falls back to VADER if OpenAI API is unavailable.
    """

    # Financial sentiment prompt template (simplified for GPT-5-nano)
    FINANCIAL_SENTIMENT_PROMPT = """Analyze financial sentiment of this article about {entity}:

{article_text}

Score from -1.0 (bearish) to 1.0 (bullish). Respond with JSON only:
{{
    "sentiment": <float -1.0 to 1.0>,
    "confidence": <float 0.0 to 1.0>
}}"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5-nano",
        use_batch_api: bool = True,
        batch_size: int = 100,
        temperature: float = 0.0,
        cache_dir: Optional[Path] = None,
        vader_fallback: Optional["NewsAPISentimentProvider"] = None,
    ):
        """
        Initialize LLM sentiment provider.

        Args:
            api_key: OpenAI API key. If None, will try to load from environment.
            model: OpenAI model to use (default: gpt-5-nano)
            use_batch_api: Whether to use Batch API for cost efficiency
            batch_size: Number of articles per batch
            temperature: Model temperature (0.0 for deterministic)
            cache_dir: Directory for caching responses
            vader_fallback: VADER provider for fallback
        """
        self.api_key = api_key
        self.model = model
        self.use_batch_api = use_batch_api
        self.batch_size = batch_size
        self.temperature = temperature

        # Initialize OpenAI client
        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=self.api_key) if self.api_key else OpenAI()
            self.openai_available = True
        except ImportError:
            logger.warning(
                "openai library not available. Install with: pip install openai>=1.12.0"
            )
            self.client = None
            self.openai_available = False
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI client: {e}")
            self.client = None
            self.openai_available = False

        # Setup cache
        if cache_dir is None:
            from quantfolio_engine.config import DATA_DIR

            cache_dir = DATA_DIR / "cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "llm_sentiment_cache.json"
        self._load_cache()

        # VADER fallback
        self.vader_fallback = vader_fallback

        if not self.openai_available and not self.vader_fallback:
            logger.warning(
                "⚠️ OpenAI not available and no VADER fallback provided. "
                "Sentiment extraction will return neutral scores."
            )

    def _load_cache(self) -> dict:
        """Load response cache from disk."""
        if self.cache_file.exists():
            try:
                import json

                with open(self.cache_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return {}

    def _save_cache(self, cache: dict):
        """Save response cache to disk."""
        try:
            import json

            with open(self.cache_file, "w") as f:
                json.dump(cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from article text."""
        import hashlib

        return hashlib.sha256(text.encode()).hexdigest()

    def _score_text_with_llm(self, text: str, entity: str) -> float:
        """
        Score text using LLM.

        Args:
            text: Article text to score
            entity: Entity name (for context)

        Returns:
            Sentiment score between -1.0 and 1.0
        """
        if not text or not self.openai_available:
            return 0.0

        # Check cache
        cache = self._load_cache()
        cache_key = self._get_cache_key(text)
        if cache_key in cache:
            cached_score = cache[cache_key].get("sentiment", 0.0)
            logger.debug(f"Using cached sentiment score: {cached_score:.3f}")
            return float(cached_score)

        # Format prompt
        prompt = self.FINANCIAL_SENTIMENT_PROMPT.format(
            entity=entity, article_text=text[:2000]  # Limit to avoid token limits
        )

        try:
            # GPT-5-nano has specific parameter requirements
            api_params = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a financial sentiment analyst. Respond only with valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
            }

            # GPT-5-nano specific parameters
            if "gpt-5" in self.model.lower():
                # GPT-5-nano uses reasoning tokens, needs more completion tokens for actual output
                # Also, response_format can cause issues with reasoning models, so we'll parse JSON manually
                api_params["max_completion_tokens"] = 1000
                # GPT-5-nano only supports default temperature (1.0), not custom values
                # Don't set temperature parameter for GPT-5-nano
            else:
                api_params["response_format"] = {"type": "json_object"}
                api_params["max_tokens"] = 150
                api_params["temperature"] = self.temperature

            response = self.client.chat.completions.create(**api_params)

            # Parse response
            import json
            import re

            content = response.choices[0].message.content

            # Debug logging
            logger.debug(f"LLM raw response: {content[:200]}")

            if not content or not content.strip():
                raise ValueError("Empty response from LLM")

            # Try to parse JSON, handle cases where response might not be valid JSON
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from the response (might be wrapped in text)
                # Look for JSON object pattern
                json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content)
                if json_match:
                    try:
                        result = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        # Try to find sentiment value directly if JSON parsing fails
                        sentiment_match = re.search(
                            r'"sentiment"\s*:\s*([-+]?\d*\.?\d+)', content
                        )
                        if sentiment_match:
                            sentiment_val = float(sentiment_match.group(1))
                            result = {"sentiment": sentiment_val, "confidence": 0.5}
                        else:
                            logger.warning(
                                f"Could not parse JSON. Raw response: {content[:500]}"
                            )
                            raise ValueError(
                                f"Could not parse JSON from response: {content[:200]}"
                            )
                else:
                    logger.warning(f"Could not find JSON in response: {content[:500]}")
                    raise ValueError(
                        f"Could not parse JSON from response: {content[:200]}"
                    )

            sentiment = float(result.get("sentiment", 0.0))

            # Validate and clip to [-1, 1]
            sentiment = max(-1.0, min(1.0, sentiment))

            # Cache result
            cache[cache_key] = {
                "sentiment": sentiment,
                "confidence": result.get("confidence", 0.5),
                "reasoning": result.get("reasoning", ""),
                "timestamp": datetime.now().isoformat(),
            }
            self._save_cache(cache)

            return sentiment

        except Exception as e:
            logger.warning(f"LLM sentiment scoring failed: {e}")
            # Fallback to VADER if available
            if self.vader_fallback:
                logger.info("Falling back to VADER sentiment analysis")
                return self.vader_fallback._score_text(text)
            return 0.0

    def fetch(self, query: str, start: datetime, end: datetime) -> pd.Series:
        """
        Fetch sentiment data using LLM analysis.

        For now, uses synchronous API calls. Batch API support will be added
        in a separate method for historical backfill.

        Args:
            query: Entity or topic name
            start: Start date
            end: End date

        Returns:
            pd.Series with monthly sentiment scores
        """
        if not self.openai_available:
            if self.vader_fallback:
                logger.warning(
                    f"⚠️ OpenAI API unavailable for {query}. "
                    "Falling back to VADER sentiment analysis."
                )
                return self.vader_fallback.fetch(query, start, end)
            else:
                logger.warning(
                    f"⚠️ OpenAI API unavailable and no fallback. "
                    f"Returning neutral sentiment for {query}."
                )
                date_range = pd.date_range(start=start, end=end, freq="ME")
                return pd.Series(index=date_range, data=0.0)

        # Fetch articles using News API (reuse existing logic)
        # For now, use News API to get articles, then score with LLM
        try:
            import requests

            news_api_key = os.getenv("NEWS_API_KEY")
            if not news_api_key:
                logger.warning("News API key not available for article fetching")
                date_range = pd.date_range(start=start, end=end, freq="ME")
                return pd.Series(index=date_range, data=0.0)

            base_url = "https://newsapi.org/v2/everything"
            months = pd.date_range(start=start, end=end, freq="ME")
            scores = []

            logger.info(
                f"Fetching LLM sentiment for '{query}' across {len(months)} months"
            )

            for i in range(len(months)):
                if i > 0:
                    month_start = months[i - 1] + timedelta(days=1)
                else:
                    month_start = start
                month_end = months[i]

                params = {
                    "q": query,
                    "from": month_start.date().isoformat(),
                    "to": month_end.date().isoformat(),
                    "language": "en",
                    "pageSize": 50,  # Reduced for cost efficiency
                    "sortBy": "relevancy",
                    "apiKey": news_api_key,
                }

                try:
                    response = requests.get(base_url, params=params, timeout=15)
                    response.raise_for_status()
                    data = response.json()
                    articles = data.get("articles", [])

                    if not articles:
                        scores.append(0.0)
                        logger.debug(
                            f"No articles for {query} in {month_end.strftime('%Y-%m')}"
                        )
                        continue

                    # Score articles with LLM
                    article_scores = []
                    seen_urls = set()

                    for article in articles:
                        url = article.get("url")
                        if url in seen_urls:
                            continue
                        seen_urls.add(url)

                        title = article.get("title") or ""
                        description = article.get("description") or ""
                        text = f"{title}. {description}"

                        if text.strip():
                            score = self._score_text_with_llm(text, query)
                            article_scores.append(score)

                    if article_scores:
                        article_series = pd.Series(article_scores).clip(
                            lower=-0.9, upper=0.9
                        )
                        monthly_sentiment = float(article_series.mean())
                        scores.append(monthly_sentiment)
                        logger.debug(
                            f"{query} {month_end.strftime('%Y-%m')}: {len(article_scores)} articles, "
                            f"sentiment={monthly_sentiment:.3f}"
                        )
                    else:
                        scores.append(0.0)

                except Exception as e:
                    logger.warning(
                        f"Error fetching articles for '{query}' in {month_end.strftime('%Y-%m')}: {e}"
                    )
                    scores.append(0.0)

            sentiment_series = pd.Series(scores, index=months)
            sentiment_series.index.name = "date"

            logger.info(
                f"Fetched LLM sentiment for '{query}': mean={sentiment_series.mean():.3f}, "
                f"std={sentiment_series.std():.3f}"
            )

            return sentiment_series

        except Exception as e:
            logger.error(f"Error in LLM sentiment fetch: {e}")
            if self.vader_fallback:
                logger.warning(f"⚠️ Falling back to VADER for {query}")
                return self.vader_fallback.fetch(query, start, end)
            date_range = pd.date_range(start=start, end=end, freq="ME")
            return pd.Series(index=date_range, data=0.0)

    def create_batch_for_backfill(
        self,
        articles_by_month: dict,
        entity: str,
        output_file: Path,
    ) -> Path:
        """
        Create a batch file for OpenAI Batch API processing.

        Args:
            articles_by_month: Dict mapping month (str) to list of article dicts
            entity: Entity name for context
            output_file: Path to save batch JSONL file

        Returns:
            Path to created batch file
        """
        import json

        batch_items = []
        for month, articles in articles_by_month.items():
            for idx, article in enumerate(articles):
                title = article.get("title", "")
                description = article.get("description", "")
                text = f"{title}. {description}".strip()

                if not text:
                    continue

                prompt = self.FINANCIAL_SENTIMENT_PROMPT.format(
                    entity=entity, article_text=text[:2000]
                )

                batch_item = {
                    "custom_id": f"{entity}_{month}_{idx}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a financial sentiment analyst. Respond only with valid JSON.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "response_format": {"type": "json_object"},
                    },
                }

                # GPT-5-nano specific parameters
                if "gpt-5" in self.model.lower():
                    # GPT-5-nano uses reasoning tokens, needs more completion tokens for actual output
                    batch_item["body"]["max_completion_tokens"] = 1000
                    # Don't use response_format with GPT-5-nano (causes issues with reasoning)
                    # GPT-5-nano only supports default temperature (1.0)
                    # Don't set temperature parameter for GPT-5-nano
                else:
                    batch_item["body"]["response_format"] = {"type": "json_object"}
                    batch_item["body"]["max_tokens"] = 150
                    batch_item["body"]["temperature"] = self.temperature

                batch_items.append(batch_item)

        # Write JSONL file
        with open(output_file, "w") as f:
            for item in batch_items:
                f.write(json.dumps(item) + "\n")

        logger.info(f"Created batch file with {len(batch_items)} items: {output_file}")
        return output_file

    def submit_batch_job(self, batch_file_path: Path) -> str:
        """
        Submit batch file to OpenAI Batch API.

        Args:
            batch_file_path: Path to batch JSONL file

        Returns:
            Batch job ID
        """
        if not self.openai_available:
            raise RuntimeError("OpenAI client not available")

        # Upload batch file
        with open(batch_file_path, "rb") as f:
            batch_file = self.client.files.create(file=f, purpose="batch")

        # Create batch job
        batch = self.client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        logger.info(f"Submitted batch job {batch.id} with {batch_file.id}")
        return batch.id

    def poll_batch_status(self, batch_id: str) -> dict:
        """
        Poll batch job status.

        Args:
            batch_id: Batch job ID

        Returns:
            Batch status dict
        """
        if not self.openai_available:
            raise RuntimeError("OpenAI client not available")

        batch = self.client.batches.retrieve(batch_id)
        return {
            "id": batch.id,
            "status": batch.status,
            "created_at": batch.created_at,
            "completed_at": batch.completed_at,
            "request_counts": (
                batch.request_counts.__dict__ if batch.request_counts else None
            ),
        }

    def retrieve_batch_results(self, batch_id: str) -> dict:
        """
        Retrieve batch results and parse sentiment scores.

        Args:
            batch_id: Batch job ID

        Returns:
            Dict mapping custom_id to sentiment score
        """
        if not self.openai_available:
            raise RuntimeError("OpenAI client not available")

        batch = self.client.batches.retrieve(batch_id)

        if batch.status != "completed":
            raise RuntimeError(f"Batch not completed. Status: {batch.status}")

        # Download results file
        results_file = self.client.files.content(batch.output_file_id)

        # Parse results
        import json

        results = {}
        for line in results_file.text.split("\n"):
            if not line.strip():
                continue
            try:
                result = json.loads(line)
                custom_id = result.get("custom_id")
                response_body = result.get("response", {}).get("body", {})
                choices = response_body.get("choices", [])

                if choices:
                    content = choices[0].get("message", {}).get("content", "{}")
                    parsed = json.loads(content)
                    sentiment = float(parsed.get("sentiment", 0.0))
                    sentiment = max(-1.0, min(1.0, sentiment))  # Clip to [-1, 1]
                    results[custom_id] = sentiment
            except Exception as e:
                logger.warning(f"Failed to parse batch result line: {e}")

        logger.info(f"Retrieved {len(results)} sentiment scores from batch {batch_id}")
        return results


class HybridSentimentProvider:
    """
    Hybrid sentiment provider combining LLM (primary) with VADER fallback.

    Uses LLM for sentiment analysis by default, falls back to VADER if LLM is unavailable.
    Can also combine LLM for recent data with VADER for historical data.
    """

    def __init__(
        self,
        llm_provider: Optional[LLMSentimentProvider] = None,
        vader_provider: Optional[NewsAPISentimentProvider] = None,
        cutoff_date: Optional[datetime] = None,
        use_llm_for_recent: bool = True,
    ):
        """
        Initialize hybrid provider.

        Args:
            llm_provider: LLM sentiment provider (primary)
            vader_provider: VADER sentiment provider (fallback)
            cutoff_date: Date before which to use VADER. If None, uses LLM for all dates.
            use_llm_for_recent: If True, use LLM for recent data (after cutoff)
        """
        self.llm_provider = llm_provider
        self.vader_provider = vader_provider

        # Default cutoff: 2 months ago (for historical data)
        if cutoff_date is None:
            self.cutoff_date = datetime.now() - timedelta(days=60)
        else:
            self.cutoff_date = cutoff_date

        self.use_llm_for_recent = use_llm_for_recent

        # Initialize providers if not provided
        if self.llm_provider is None:
            try:
                from quantfolio_engine.config import OPENAI_API_KEY

                vader_fallback = self.vader_provider or NewsAPISentimentProvider()
                self.llm_provider = LLMSentimentProvider(
                    api_key=OPENAI_API_KEY,
                    vader_fallback=vader_fallback,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize LLM provider: {e}")
                self.llm_provider = None

        if self.vader_provider is None:
            from quantfolio_engine.config import NEWS_API_KEY

            self.vader_provider = NewsAPISentimentProvider(api_key=NEWS_API_KEY)

        logger.info(
            f"Hybrid sentiment provider initialized: "
            f"LLM (primary) with VADER fallback"
        )

    def fetch(self, query: str, start: datetime, end: datetime) -> pd.Series:
        """Fetch sentiment data using hybrid approach (LLM primary, VADER fallback)."""

        # Try LLM first if available
        if self.llm_provider and self.llm_provider.openai_available:
            try:
                if (
                    self.cutoff_date
                    and start < self.cutoff_date
                    and not self.use_llm_for_recent
                ):
                    # Historical data: use VADER
                    logger.debug(
                        f"{query}: Using VADER for historical data (before {self.cutoff_date.date()})"
                    )
                    return self.vader_provider.fetch(
                        query, start, min(end, self.cutoff_date)
                    )
                else:
                    # Use LLM
                    logger.debug(f"{query}: Using LLM sentiment analysis")
                    return self.llm_provider.fetch(query, start, end)
            except Exception as e:
                logger.warning(
                    f"⚠️ LLM sentiment failed for {query}: {e}. "
                    "Falling back to VADER sentiment analysis."
                )
                return self.vader_provider.fetch(query, start, end)
        else:
            # LLM not available, use VADER
            logger.warning(
                f"⚠️ LLM provider unavailable for {query}. "
                "Using VADER sentiment analysis."
            )
            return self.vader_provider.fetch(query, start, end)


class RandomSentimentProvider:
    """Random sentiment provider for testing."""

    def fetch(self, query: str, start: datetime, end: datetime) -> pd.Series:
        """Generate random sentiment data."""
        date_range = pd.date_range(start=start, end=end, freq="ME")
        sentiment_scores = pd.Series(
            index=date_range, data=np.random.uniform(-1, 1, len(date_range))
        )
        return sentiment_scores


def set_log_level(debug: bool):
    from loguru import logger

    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        level="DEBUG" if debug else "INFO",
        colorize=True,
    )


class DataLoader:
    """
    Data loader for fetching and processing financial data.

    Args:
        debug (bool): If True, sets logger to DEBUG level for verbose output. Default is False.
    """

    def __init__(
        self,
        config: Optional[DataConfig] = None,
        yf_client=None,
        fred_client=None,
        sentiment_client: Optional[BaseSentimentProvider] = None,
        debug: bool = False,
    ):
        set_log_level(debug)
        self.config = config or get_default_data_config()
        self.yf_client = yf_client or yf
        self.fred_client = fred_client or Fred(api_key=self.config.fred_api_key)

        # Use Hybrid sentiment provider (LLM primary, VADER fallback)
        if sentiment_client:
            self.sentiment_client = sentiment_client
        else:
            # Try to initialize LLM provider with VADER fallback
            try:
                from quantfolio_engine.config import NEWS_API_KEY, OPENAI_API_KEY

                # Initialize VADER provider for fallback
                vader_provider = NewsAPISentimentProvider(api_key=NEWS_API_KEY)

                # Initialize LLM provider with VADER fallback
                llm_provider = LLMSentimentProvider(
                    api_key=OPENAI_API_KEY,
                    vader_fallback=vader_provider,
                )

                # Create hybrid provider
                self.sentiment_client = HybridSentimentProvider(
                    llm_provider=llm_provider,
                    vader_provider=vader_provider,
                )
                logger.info(
                    "Using LLM sentiment provider (OpenAI GPT-5-nano) with VADER fallback"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to initialize LLM provider: {e}. "
                    "Falling back to VADER-only sentiment provider."
                )
                # Fallback to VADER-only
                from quantfolio_engine.config import NEWS_API_KEY

                vader_provider = NewsAPISentimentProvider(api_key=NEWS_API_KEY)
                self.sentiment_client = HybridSentimentProvider(
                    llm_provider=None,
                    vader_provider=vader_provider,
                )
                logger.info("Using VADER sentiment provider (LLM unavailable)")

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

                except Exception:
                    logger.exception(f"Error processing data for {ticker}")
                    continue

            # Batch save raw data
            if save_raw and raw_data_dict:
                self._batch_save_raw_data(raw_data_dict, "prices")

        except Exception:
            logger.warning(
                "Batch download failed, falling back to individual downloads..."
            )
            logger.exception("Batch download error details")
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
                except Exception:
                    logger.exception(f"Error fetching data for {ticker}")
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

        except Exception:
            logger.exception(f"Error fetching data for {ticker}")
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
            except Exception:
                logger.exception(f"Error saving raw data for {name}")

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
                except Exception:
                    logger.exception(f"Error fetching {series_id}")
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
                all_data["VIX"] = monthly_vix

                if save_raw:
                    raw_data_dict["VIX"] = vix_data
        except Exception:
            logger.exception("Error fetching VIX from Yahoo Finance")

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
                            # Use ticker name directly (no prefix) for entity sentiment
                            entity = name.replace("entity_", "")
                            all_sentiment[entity] = result
                            if save_raw:
                                raw_data_dict[entity] = result
                        elif name.startswith("topic_"):
                            # Use "topic_" prefix for topics to distinguish them from assets
                            topic = name.replace("topic_", "")
                            all_sentiment[f"topic_{topic.replace(' ', '_')}"] = result
                            if save_raw:
                                raw_data_dict[f"topic_{topic.replace(' ', '_')}"] = (
                                    result
                                )
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
        # Use ticker name directly (no prefix) for entity sentiment to align with asset names
        for entity in self.config.sentiment_entities:
            # Generate random sentiment scores between -1 and 1
            sentiment_scores = pd.Series(
                index=date_range, data=np.random.uniform(-1, 1, len(date_range))
            )
            all_sentiment[entity] = sentiment_scores
            if save_raw:
                raw_data_dict[entity] = sentiment_scores

        # Generate placeholder sentiment for topics
        # Use "topic_" prefix for topics to distinguish them from assets
        for topic in self.config.sentiment_topics:
            # Generate random sentiment scores between -1 and 1
            sentiment_scores = pd.Series(
                index=date_range, data=np.random.uniform(-1, 1, len(date_range))
            )
            all_sentiment[f"topic_{topic.replace(' ', '_')}"] = sentiment_scores
            if save_raw:
                raw_data_dict[f"topic_{topic.replace(' ', '_')}"] = sentiment_scores

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
