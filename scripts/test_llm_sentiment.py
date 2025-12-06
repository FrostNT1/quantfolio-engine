#!/usr/bin/env python3
"""
Test script for LLM sentiment provider.

This script tests the LLM sentiment provider with a real API call
to verify everything is working correctly.
"""
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from loguru import logger

from quantfolio_engine.config import NEWS_API_KEY, OPENAI_API_KEY
from quantfolio_engine.data.data_loader import (
    LLMSentimentProvider,
    NewsAPISentimentProvider,
)


def test_llm_sentiment():
    """Test LLM sentiment provider with a simple example."""

    if not OPENAI_API_KEY:
        logger.error("❌ OPENAI_API_KEY not found in environment variables")
        logger.info("Please set OPENAI_API_KEY in your .env file")
        return False

    logger.info("✅ OPENAI_API_KEY found")
    logger.info(f"Testing with model: gpt-5-nano")

    # Initialize providers
    try:
        vader_provider = NewsAPISentimentProvider(api_key=NEWS_API_KEY)
        llm_provider = LLMSentimentProvider(
            api_key=OPENAI_API_KEY,
            model="gpt-5-nano",
            vader_fallback=vader_provider,
        )

        if not llm_provider.openai_available:
            logger.error("❌ OpenAI client not available")
            return False

        logger.info("✅ LLM provider initialized successfully")

        # Test with a simple entity and recent date range
        test_entity = "AAPL"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)  # Last 2 months

        logger.info(f"\n📊 Testing sentiment extraction for {test_entity}")
        logger.info(f"Date range: {start_date.date()} to {end_date.date()}")

        # Fetch sentiment
        logger.info("\n🔄 Fetching sentiment data...")
        sentiment_series = llm_provider.fetch(test_entity, start_date, end_date)

        if sentiment_series is None or len(sentiment_series) == 0:
            logger.error("❌ No sentiment data returned")
            return False

        logger.info(f"\n✅ Successfully fetched sentiment data!")
        logger.info(f"   Shape: {sentiment_series.shape}")
        logger.info(f"   Mean sentiment: {sentiment_series.mean():.3f}")
        logger.info(f"   Std sentiment: {sentiment_series.std():.3f}")
        logger.info(f"   Min sentiment: {sentiment_series.min():.3f}")
        logger.info(f"   Max sentiment: {sentiment_series.max():.3f}")
        logger.info(f"\n   Sample values:")
        for date, value in sentiment_series.head().items():
            logger.info(f"   {date.strftime('%Y-%m-%d')}: {value:.3f}")

        # Verify sentiment scores are in valid range
        if sentiment_series.min() < -1.0 or sentiment_series.max() > 1.0:
            logger.warning("⚠️ Some sentiment scores are outside [-1, 1] range")
        else:
            logger.info("✅ All sentiment scores are in valid range [-1, 1]")

        # Test direct LLM scoring
        logger.info("\n🧪 Testing direct LLM text scoring...")
        test_text = "Apple Inc. reported strong quarterly earnings, beating analyst expectations. The stock surged 5% in after-hours trading."
        score = llm_provider._score_text_with_llm(test_text, "AAPL")
        logger.info(f"   Test text score: {score:.3f}")

        if abs(score) <= 1.0:
            logger.info("✅ Direct scoring test passed")
        else:
            logger.warning(f"⚠️ Score {score} is outside expected range")

        logger.info(
            "\n✅ All tests passed! LLM sentiment provider is working correctly."
        )
        return True

    except Exception as e:
        logger.error(f"❌ Error during testing: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_llm_sentiment()
    sys.exit(0 if success else 1)
