#!/usr/bin/env python3
"""
Direct test of LLM sentiment scoring without News API dependency.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from loguru import logger

from quantfolio_engine.config import OPENAI_API_KEY
from quantfolio_engine.data.data_loader import LLMSentimentProvider


def test_direct_llm():
    """Test LLM directly with sample text."""

    if not OPENAI_API_KEY:
        logger.error("❌ OPENAI_API_KEY not found")
        return False

    logger.info("✅ Testing GPT-5-nano directly")

    try:
        provider = LLMSentimentProvider(
            api_key=OPENAI_API_KEY,
            model="gpt-5-nano",
        )

        if not provider.openai_available:
            logger.error("❌ OpenAI client not available")
            return False

        # Test cases
        test_cases = [
            {
                "text": "Apple Inc. reported strong quarterly earnings, beating analyst expectations. The stock surged 5% in after-hours trading.",
                "entity": "AAPL",
                "expected": "positive",
            },
            {
                "text": "Microsoft faces regulatory scrutiny over cloud computing practices. Shares decline 3% on concerns about potential antitrust action.",
                "entity": "MSFT",
                "expected": "negative",
            },
            {
                "text": "JPMorgan Chase announces quarterly dividend increase. The bank's capital position remains strong.",
                "entity": "JPM",
                "expected": "positive",
            },
        ]

        logger.info(f"\n🧪 Testing {len(test_cases)} sample texts...\n")

        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"Test {i}/{len(test_cases)}: {test_case['entity']}")
            logger.info(f"  Text: {test_case['text'][:80]}...")

            try:
                score = provider._score_text_with_llm(
                    test_case["text"], test_case["entity"]
                )

                logger.info(f"  ✅ Sentiment score: {score:.3f}")

                # Validate score
                if abs(score) > 1.0:
                    logger.warning(f"  ⚠️ Score {score} outside [-1, 1] range")
                else:
                    logger.info(f"  ✅ Score in valid range")

                # Check if direction matches expectation
                if test_case["expected"] == "positive" and score > 0:
                    logger.info(f"  ✅ Direction matches (positive)")
                elif test_case["expected"] == "negative" and score < 0:
                    logger.info(f"  ✅ Direction matches (negative)")
                else:
                    logger.warning(f"  ⚠️ Direction doesn't match expectation")

            except Exception as e:
                logger.error(f"  ❌ Error: {e}")
                import traceback

                traceback.print_exc()

            logger.info("")

        logger.info("✅ Direct LLM test completed!")
        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_direct_llm()
    sys.exit(0 if success else 1)
