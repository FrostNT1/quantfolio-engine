#!/usr/bin/env python3
"""
Backfill historical sentiment data using OpenAI Batch API.

This script:
1. Fetches articles for all entities from start_date to end_date
2. Creates batch files for OpenAI Batch API
3. Submits batches and monitors progress
4. Retrieves results and saves sentiment data
"""
from datetime import datetime
import json
from pathlib import Path

# Add parent directory to path
import sys
import time
from typing import Dict, List

from loguru import logger
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantfolio_engine.config import (
    DATA_DIR,
    NEWS_API_KEY,
    OPENAI_API_KEY,
    PROCESSED_DATA_DIR,
    SENTIMENT_ENTITIES,
)
from quantfolio_engine.data.data_loader import (
    LLMSentimentProvider,
    NewsAPISentimentProvider,
)


def fetch_articles_for_period(
    entity: str,
    start_date: datetime,
    end_date: datetime,
    news_api_key: str,
) -> Dict[str, List[dict]]:
    """
    Fetch articles for entity across date range, grouped by month.

    Returns:
        Dict mapping month string (YYYY-MM) to list of article dicts
    """
    import requests

    base_url = "https://newsapi.org/v2/everything"
    months = pd.date_range(start=start_date, end=end_date, freq="ME")
    articles_by_month = {}

    logger.info(f"Fetching articles for {entity} across {len(months)} months...")

    for month_end in tqdm(months, desc=f"Fetching {entity}"):
        month_key = month_end.strftime("%Y-%m")

        # Determine month boundaries
        month_start = month_end.replace(day=1)
        if month_end == months[0]:
            month_start = start_date

        params = {
            "q": entity,
            "from": month_start.date().isoformat(),
            "to": month_end.date().isoformat(),
            "language": "en",
            "pageSize": 100,
            "sortBy": "relevancy",
            "apiKey": news_api_key,
        }

        try:
            response = requests.get(base_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            articles = data.get("articles", [])

            # Deduplicate by URL
            seen_urls = set()
            unique_articles = []
            for article in articles:
                url = article.get("url")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_articles.append(article)

            articles_by_month[month_key] = unique_articles
            logger.debug(f"{entity} {month_key}: {len(unique_articles)} articles")

        except Exception as e:
            logger.warning(f"Error fetching articles for {entity} {month_key}: {e}")
            articles_by_month[month_key] = []

    total_articles = sum(len(articles) for articles in articles_by_month.values())
    logger.info(f"Fetched {total_articles} total articles for {entity}")

    return articles_by_month


def backfill_entity_sentiment(
    entity: str,
    start_date: datetime,
    end_date: datetime,
    llm_provider: LLMSentimentProvider,
    news_api_key: str,
    batch_dir: Path,
) -> pd.Series:
    """
    Backfill sentiment for a single entity using Batch API.

    Returns:
        pd.Series with monthly sentiment scores
    """
    # Fetch articles
    articles_by_month = fetch_articles_for_period(
        entity, start_date, end_date, news_api_key
    )

    # Create batch file
    batch_file = (
        batch_dir
        / f"batch_{entity}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.jsonl"
    )
    llm_provider.create_batch_for_backfill(articles_by_month, entity, batch_file)

    # Submit batch
    logger.info(f"Submitting batch for {entity}...")
    batch_id = llm_provider.submit_batch_job(batch_file)

    # Poll for completion
    logger.info(f"Polling batch {batch_id} for {entity}...")
    max_wait_time = 3600 * 24  # 24 hours
    start_time = time.time()

    with tqdm(desc=f"Processing {entity}") as pbar:
        while True:
            status = llm_provider.poll_batch_status(batch_id)
            pbar.set_postfix({"status": status["status"]})

            if status["status"] == "completed":
                break
            elif status["status"] in ["failed", "expired", "cancelled"]:
                raise RuntimeError(
                    f"Batch {batch_id} failed with status: {status['status']}"
                )

            if time.time() - start_time > max_wait_time:
                raise RuntimeError(f"Batch {batch_id} timed out after {max_wait_time}s")

            time.sleep(60)  # Check every minute
            pbar.update(1)

    # Retrieve results
    logger.info(f"Retrieving results for {entity}...")
    results = llm_provider.retrieve_batch_results(batch_id)

    # Aggregate results by month
    months = pd.date_range(start=start_date, end=end_date, freq="ME")
    monthly_scores = {}

    for month in months:
        month_key = month.strftime("%Y-%m")
        month_scores = []

        for custom_id, score in results.items():
            if custom_id.startswith(f"{entity}_{month_key}_"):
                month_scores.append(score)

        if month_scores:
            monthly_scores[month] = pd.Series(month_scores).clip(-0.9, 0.9).mean()
        else:
            monthly_scores[month] = 0.0

    sentiment_series = pd.Series(monthly_scores)
    sentiment_series.index.name = "date"

    logger.info(
        f"✅ {entity}: mean={sentiment_series.mean():.3f}, "
        f"std={sentiment_series.std():.3f}"
    )

    return sentiment_series


def main():
    """Main backfill function."""
    import argparse

    parser = argparse.ArgumentParser(description="Backfill sentiment data using LLM")
    parser.add_argument(
        "--start-date",
        type=str,
        default="2010-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--entities",
        type=str,
        nargs="+",
        default=None,
        help="Entities to process. Defaults to all in config.",
    )
    parser.add_argument(
        "--batch-dir",
        type=str,
        default=None,
        help="Directory for batch files. Defaults to data/cache/batches/",
    )

    args = parser.parse_args()

    # Parse dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = (
        datetime.strptime(args.end_date, "%Y-%m-%d")
        if args.end_date
        else datetime.now()
    )

    # Entities to process
    entities = args.entities or SENTIMENT_ENTITIES

    # Batch directory
    if args.batch_dir:
        batch_dir = Path(args.batch_dir)
    else:
        batch_dir = DATA_DIR / "cache" / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Initialize providers
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment")
    if not NEWS_API_KEY:
        raise ValueError("NEWS_API_KEY not found in environment")

    vader_provider = NewsAPISentimentProvider(api_key=NEWS_API_KEY)
    llm_provider = LLMSentimentProvider(
        api_key=OPENAI_API_KEY,
        model="gpt-5-nano",
        vader_fallback=vader_provider,
    )

    if not llm_provider.openai_available:
        raise RuntimeError("OpenAI client not available. Check API key.")

    logger.info(f"Starting backfill for {len(entities)} entities")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")

    # Process each entity
    all_sentiment = {}
    checkpoint_file = batch_dir / "backfill_checkpoint.json"

    # Load checkpoint if exists
    completed_entities = set()
    if checkpoint_file.exists():
        with open(checkpoint_file, "r") as f:
            checkpoint = json.load(f)
            completed_entities = set(checkpoint.get("completed_entities", []))
            logger.info(f"Resuming from checkpoint. Completed: {completed_entities}")

    for entity in tqdm(entities, desc="Processing entities"):
        if entity in completed_entities:
            logger.info(f"Skipping {entity} (already completed)")
            continue

        try:
            sentiment_series = backfill_entity_sentiment(
                entity,
                start_date,
                end_date,
                llm_provider,
                NEWS_API_KEY,
                batch_dir,
            )
            all_sentiment[entity] = sentiment_series

            # Update checkpoint
            completed_entities.add(entity)
            checkpoint = {
                "completed_entities": list(completed_entities),
                "last_updated": datetime.now().isoformat(),
            }
            with open(checkpoint_file, "w") as f:
                json.dump(checkpoint, f, indent=2)

        except Exception as e:
            logger.error(f"Error processing {entity}: {e}")
            continue

    # Combine all sentiment data
    if all_sentiment:
        sentiment_df = pd.DataFrame(all_sentiment)
        sentiment_df = sentiment_df.sort_index()

        # Save results
        output_file = PROCESSED_DATA_DIR / "sentiment_monthly_llm.csv"
        sentiment_df.to_csv(output_file)
        logger.info(f"✅ Saved sentiment data to {output_file}")
        logger.info(f"Shape: {sentiment_df.shape}")
    else:
        logger.warning("No sentiment data generated")


if __name__ == "__main__":
    main()
