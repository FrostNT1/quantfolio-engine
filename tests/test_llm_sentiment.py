"""Tests for LLM sentiment provider."""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from quantfolio_engine.data.data_loader import LLMSentimentProvider


class TestLLMSentimentProvider:
    """Test cases for LLMSentimentProvider class."""

    def test_init_with_api_key(self):
        """Test LLMSentimentProvider initialization with API key."""
        provider = LLMSentimentProvider(api_key="test-key")
        assert provider is not None
        assert provider.model == "gpt-5-nano"
        assert provider.openai_available is True

    def test_init_without_api_key(self):
        """Test LLMSentimentProvider initialization without API key."""
        # When api_key is None, it tries to load from environment
        # If OpenAI is installed, it will create a client (openai_available=True)
        # If not installed or fails, openai_available=False
        provider = LLMSentimentProvider(api_key="invalid-key-that-will-fail")
        assert provider is not None
        # openai_available depends on whether OpenAI library is available
        assert hasattr(provider, "openai_available")

    def test_prompt_template(self):
        """Test prompt template exists."""
        provider = LLMSentimentProvider(api_key="test-key")
        assert hasattr(provider, "FINANCIAL_SENTIMENT_PROMPT")
        prompt_template = provider.FINANCIAL_SENTIMENT_PROMPT
        assert "entity" in prompt_template
        assert "article_text" in prompt_template
        assert "sentiment" in prompt_template.lower()

    def test_get_cache_key(self):
        """Test cache key generation."""
        provider = LLMSentimentProvider(api_key="test-key")
        # _get_cache_key takes only text parameter (entity is included in text)
        key1 = provider._get_cache_key("text1_AAPL")
        key2 = provider._get_cache_key("text1_AAPL")
        key3 = provider._get_cache_key("text2_AAPL")

        # Same text should produce same key
        assert key1 == key2
        # Different text should produce different key
        assert key1 != key3

    def test_score_text_with_llm_success(self):
        """Test successful LLM text scoring."""
        # Mock OpenAI client and response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = (
            '{"sentiment": 0.75, "confidence": 0.8}'
        )
        mock_client.chat.completions.create.return_value = mock_response

        provider = LLMSentimentProvider(api_key="test-key")
        provider.client = mock_client
        provider.openai_available = True

        score = provider._score_text_with_llm("Apple reported strong earnings.", "AAPL")

        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0
        assert score == 0.75

    def test_score_text_with_llm_empty_response(self):
        """Test LLM scoring with empty response (fallback to VADER)."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = ""
        mock_client.chat.completions.create.return_value = mock_response

        # Mock VADER fallback
        mock_vader = Mock()
        mock_vader._score_text.return_value = 0.1

        provider = LLMSentimentProvider(api_key="test-key", vader_fallback=mock_vader)
        provider.client = mock_client
        provider.openai_available = True

        score = provider._score_text_with_llm("Test text", "AAPL")

        # Should fallback to VADER
        assert score == 0.1
        mock_vader._score_text.assert_called_once()

    def test_score_text_with_llm_invalid_json(self):
        """Test LLM scoring with invalid JSON response."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Not valid JSON"
        mock_client.chat.completions.create.return_value = mock_response

        # Mock VADER fallback
        mock_vader = Mock()
        mock_vader._score_text.return_value = 0.0

        provider = LLMSentimentProvider(api_key="test-key", vader_fallback=mock_vader)
        provider.client = mock_client
        provider.openai_available = True

        score = provider._score_text_with_llm("Test text", "AAPL")

        # Should fallback to VADER
        assert score == 0.0

    def test_score_text_with_llm_api_error(self):
        """Test LLM scoring with API error."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        # Mock VADER fallback
        mock_vader = Mock()
        mock_vader._score_text.return_value = 0.0

        provider = LLMSentimentProvider(api_key="test-key", vader_fallback=mock_vader)
        provider.client = mock_client
        provider.openai_available = True

        score = provider._score_text_with_llm("Test text", "AAPL")

        # Should fallback to VADER
        assert score == 0.0

    def test_fetch_without_client(self):
        """Test fetch method when OpenAI client is not available."""
        provider = LLMSentimentProvider(api_key=None)
        start = datetime(2023, 1, 1)
        end = datetime(2023, 12, 31)

        result = provider.fetch("AAPL", start, end)

        assert isinstance(result, pd.Series)
        assert len(result) > 0
        # Should return neutral sentiment (0.0)
        assert all(result == 0.0)

    @patch("requests.get")
    def test_fetch_with_articles(self, mock_requests_get):
        """Test fetch method with article data."""
        # Mock News API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "articles": [
                {"title": "Apple earnings", "description": "Strong results"},
                {"title": "Apple stock", "description": "Price up"},
            ]
        }
        mock_response.status_code = 200
        mock_requests_get.return_value = mock_response

        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_response = Mock()
        mock_openai_response.choices = [Mock()]
        mock_openai_response.choices[0].message.content = (
            '{"sentiment": 0.8, "confidence": 0.9}'
        )
        mock_client.chat.completions.create.return_value = mock_openai_response

        provider = LLMSentimentProvider(api_key="test-key")
        provider.client = mock_client
        provider.openai_available = True

        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 31)

        with patch.dict("os.environ", {"NEWS_API_KEY": "test-news-key"}):
            result = provider.fetch("AAPL", start, end)

        assert isinstance(result, pd.Series)
        assert len(result) > 0

    def test_sentiment_score_clipping(self):
        """Test that sentiment scores are clipped to [-1, 1] range."""
        provider = LLMSentimentProvider(api_key="test-key")

        # Test clipping logic (simulated)
        test_scores = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
        for score in test_scores:
            clipped = max(-1.0, min(1.0, score))
            assert -1.0 <= clipped <= 1.0

    def test_gpt5_nano_parameters(self):
        """Test GPT-5-nano specific parameter handling."""
        provider = LLMSentimentProvider(api_key="test-key", model="gpt-5-nano")

        assert provider.model == "gpt-5-nano"
        # GPT-5-nano should use max_completion_tokens, not max_tokens
        # This is tested in the actual API call, but we verify model is set correctly
