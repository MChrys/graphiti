"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from graphiti_core.llm_client.config import LLMConfig, ModelSize
from graphiti_core.llm_client.gemini_client import (
    DEFAULT_GEMINI_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_SMALL_MODEL,
    GEMINI_MODEL_MAX_TOKENS,
    GeminiClient,
)
from graphiti_core.prompts.models import Message


class ResponseModel(BaseModel):
    """Test model for response testing."""

    test_field: str
    optional_field: int = 0


class TestGeminiClientInitialization:
    """Tests for GeminiClient initialization."""

    @patch('google.genai.Client')
    def test_init_with_config(self, mock_genai_client):
        """Test initialization with a config object."""
        config = LLMConfig(api_key='test_api_key', model='gemini-2.5-flash', temperature=0.5, max_tokens=1000)
        client = GeminiClient(config=config, cache=False)

        assert client.config == config
        assert client.model == 'gemini-2.5-flash'
        assert client.temperature == 0.5
        mock_genai_client.assert_called_once_with(api_key='test_api_key')

    @patch('google.genai.Client')
    def test_init_with_default_model(self, mock_genai_client):
        """Test initialization with default model when none is provided."""
        config = LLMConfig(api_key='test_api_key')
        client = GeminiClient(config=config, cache=False)

        assert client.model == DEFAULT_MODEL

    @patch('google.genai.Client')
    def test_init_without_config(self, mock_genai_client):
        """Test initialization without a config uses defaults."""
        client = GeminiClient(cache=False)

        assert client.config is not None
        assert client.model is None  # Config model is None when not provided
        mock_genai_client.assert_called_once_with(api_key=None)

    @patch('google.genai.Client')
    def test_init_with_thinking_config(self, mock_genai_client):
        """Test initialization with thinking config."""
        with patch('google.genai.types.ThinkingConfig') as mock_thinking_config:
            thinking_config = mock_thinking_config.return_value
            client = GeminiClient(thinking_config=thinking_config)
            assert client.thinking_config == thinking_config

    @patch('google.genai.Client')
    def test_init_with_custom_client(self, mock_genai_client):
        """Test initialization with a custom client."""
        mock_client = MagicMock()
        client = GeminiClient(client=mock_client)

        assert client.client == mock_client
        # Should not create a new client
        mock_genai_client.assert_not_called()

    @patch('google.genai.Client')
    def test_init_with_max_tokens(self, mock_genai_client):
        """Test initialization with max_tokens parameter."""
        config = LLMConfig(api_key='test_api_key', model='gemini-2.5-flash')
        client = GeminiClient(config=config, cache=False, max_tokens=5000)

        assert client.max_tokens == 5000


class TestGeminiClientHelperMethods:
    """Tests for GeminiClient helper methods."""

    def test_get_model_for_size_small(self):
        """Test small model size selection."""
        client = GeminiClient(config=LLMConfig(api_key='test'), cache=False)

        small_model = client._get_model_for_size(ModelSize.small)
        assert small_model == DEFAULT_SMALL_MODEL

    def test_get_model_for_size_medium(self):
        """Test medium model size selection."""
        client = GeminiClient(config=LLMConfig(api_key='test'), cache=False)

        medium_model = client._get_model_for_size(ModelSize.medium)
        assert medium_model == client.model

    def test_get_model_for_size_large(self):
        """Test large model size selection."""
        client = GeminiClient(config=LLMConfig(api_key='test'), cache=False)

        large_model = client._get_model_for_size(ModelSize.large)
        assert large_model == client.model

    def test_get_max_tokens_for_known_model(self):
        """Test max tokens for known Gemini models."""
        client = GeminiClient(config=LLMConfig(api_key='test'), cache=False)

        # Test various known models
        test_cases = [
            ('gemini-2.5-pro', 65536),
            ('gemini-2.5-flash', 65536),
            ('gemini-2.5-flash-lite', 64000),
            ('gemini-2.0-flash', 8192),
            ('gemini-1.5-pro', 8192),
        ]

        for model, expected_tokens in test_cases:
            result = client._get_max_tokens_for_model(model)
            assert result == expected_tokens

    def test_get_max_tokens_for_unknown_model(self):
        """Test max tokens for unknown model uses default."""
        client = GeminiClient(config=LLMConfig(api_key='test'), cache=False)

        result = client._get_max_tokens_for_model('unknown-model')
        assert result == DEFAULT_GEMINI_MAX_TOKENS

    def test_resolve_max_tokens_with_explicit_value(self):
        """Test max_tokens resolution with explicit value takes precedence."""
        client = GeminiClient(config=LLMConfig(api_key='test'), cache=False)

        result = client._resolve_max_tokens(5000, 'gemini-2.5-flash')
        assert result == 5000  # Explicit value should be used

    def test_resolve_max_tokens_with_instance_value(self):
        """Test max_tokens resolution with instance max_tokens."""
        client = GeminiClient(config=LLMConfig(api_key='test'), cache=False, max_tokens=3000)

        result = client._resolve_max_tokens(None, 'gemini-2.5-flash')
        assert result == 3000  # Instance value should be used

    def test_resolve_max_tokens_fallback_to_model_mapping(self):
        """Test max_tokens resolution falls back to model mapping."""
        client = GeminiClient(config=LLMConfig(api_key='test'), cache=False)

        result = client._resolve_max_tokens(None, 'gemini-2.5-flash')
        assert result == GEMINI_MODEL_MAX_TOKENS['gemini-2.5-flash']

    def test_resolve_max_tokens_fallback_to_default(self):
        """Test max_tokens resolution falls back to default for unknown model."""
        client = GeminiClient(config=LLMConfig(api_key='test'), cache=False)

        result = client._resolve_max_tokens(None, 'unknown-model')
        assert result == DEFAULT_GEMINI_MAX_TOKENS


class TestGeminiClientSafetyChecks:
    """Tests for GeminiClient safety check methods."""

    def test_check_safety_blocks_no_candidates(self):
        """Test safety check when response has no candidates."""
        client = GeminiClient(config=LLMConfig(api_key='test'), cache=False)

        # Response without candidates should not raise
        mock_response = MagicMock()
        del mock_response.candidates  # Remove candidates attribute

        # Should not raise exception
        client._check_safety_blocks(mock_response)

    def test_check_safety_blocks_no_finish_reason(self):
        """Test safety check when candidate has no finish_reason."""
        client = GeminiClient(config=LLMConfig(api_key='test'), cache=False)

        mock_candidate = MagicMock()
        del mock_candidate.finish_reason  # Remove finish_reason attribute

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        # Should not raise exception
        client._check_safety_blocks(mock_response)

    def test_check_safety_blocks_non_safety_finish(self):
        """Test safety check when finish reason is not SAFETY."""
        client = GeminiClient(config=LLMConfig(api_key='test'), cache=False)

        mock_candidate = MagicMock()
        mock_candidate.finish_reason = 'STOP'

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        # Should not raise exception
        client._check_safety_blocks(mock_response)

    def test_check_safety_blocks_with_safety_violation(self):
        """Test safety check when response is blocked for safety."""
        client = GeminiClient(config=LLMConfig(api_key='test'), cache=False)

        mock_safety_rating = MagicMock()
        mock_safety_rating.blocked = True
        mock_safety_rating.category = 'HARM_CATEGORY_HARASSMENT'
        mock_safety_rating.probability = 'HIGH'

        mock_candidate = MagicMock()
        mock_candidate.finish_reason = 'SAFETY'
        mock_candidate.safety_ratings = [mock_safety_rating]

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        # Should raise exception with safety details
        with pytest.raises(Exception, match='Response blocked by Gemini safety filters'):
            client._check_safety_blocks(mock_response)

    def test_check_safety_blocks_with_no_safety_ratings(self):
        """Test safety check when blocked but no safety ratings provided."""
        client = GeminiClient(config=LLMConfig(api_key='test'), cache=False)

        mock_candidate = MagicMock()
        mock_candidate.finish_reason = 'SAFETY'
        mock_candidate.safety_ratings = None

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        # Should raise exception with generic message
        with pytest.raises(Exception, match='Response blocked by Gemini safety filters'):
            client._check_safety_blocks(mock_response)

    def test_check_prompt_blocks_no_block_reason(self):
        """Test prompt block check when no block reason."""
        client = GeminiClient(config=LLMConfig(api_key='test'), cache=False)

        mock_response = MagicMock()
        mock_response.prompt_feedback = None

        # Should not raise exception
        client._check_prompt_blocks(mock_response)

    def test_check_prompt_blocks_with_block_reason(self):
        """Test prompt block check when prompt is blocked."""
        client = GeminiClient(config=LLMConfig(api_key='test'), cache=False)

        mock_prompt_feedback = MagicMock()
        mock_prompt_feedback.block_reason = 'BLOCKED_REASON_OTHER'

        mock_response = MagicMock()
        mock_response.prompt_feedback = mock_prompt_feedback

        # Should raise exception
        with pytest.raises(Exception, match='Prompt blocked by Gemini safety filters'):
            client._check_prompt_blocks(mock_response)


class TestGeminiClientMessageProcessing:
    """Tests for GeminiClient message processing methods."""

    def test_clean_input_method(self):
        """Test the _clean_input method inherited from LLMClient."""
        client = GeminiClient(config=LLMConfig(api_key='test'), cache=False)

        # Test basic functionality - this method should work the same as in base LLMClient
        test_cases = [
            # Basic text should remain unchanged
            ('Hello World', 'Hello World'),
            # Control characters should be removed
            ('Hello\x00World', 'HelloWorld'),
            # Newlines should be preserved
            ('Hello\nWorld', 'Hello\nWorld'),
            # Invalid Unicode should be removed
            ('Hello\udcdeWorld', 'HelloWorld'),
        ]

        for input_str, expected in test_cases:
            assert client._clean_input(input_str) == expected, f'Failed for input: {repr(input_str)}'

    def test_config_property_preservation(self):
        """Test that config property is preserved exactly."""
        config = LLMConfig(
            api_key='test_key',
            model='gemini-2.5-flash',
            temperature=0.5,
            max_tokens=1000,
            base_url='https://api.google.com'
        )
        client = GeminiClient(config=config, cache=False)

        assert client.config is config
        assert client.config.api_key == 'test_key'
        assert client.config.model == 'gemini-2.5-flash'

    def test_model_property_from_config(self):
        """Test model property from config."""
        config = LLMConfig(api_key='test', model='gemini-2.5-pro')
        client = GeminiClient(config=config, cache=False)

        assert client.model == 'gemini-2.5-pro'

    def test_temperature_property_inheritance(self):
        """Test temperature property from LLMConfig."""
        config = LLMConfig(api_key='test', temperature=0.7)
        client = GeminiClient(config=config, cache=False)

        assert client.temperature == 0.7

    def test_max_tokens_property_inheritance(self):
        """Test max_tokens property from constructor parameter."""
        config = LLMConfig(api_key='test', max_tokens=2000)
        client = GeminiClient(config=config, cache=False, max_tokens=3000)

        # Constructor parameter should override config
        assert client.max_tokens == 3000


class TestGeminiClientConstants:
    """Tests for GeminiClient constants and mappings."""

    def test_max_retries_constant(self):
        """Test that MAX_RETRIES constant is set correctly."""
        assert GeminiClient.MAX_RETRIES == 2

    def test_model_max_tokens_mapping_completeness(self):
        """Test that model max tokens mapping includes expected models."""
        expected_models = [
            'gemini-2.5-pro',
            'gemini-2.5-flash',
            'gemini-2.5-flash-lite',
            'gemini-2.0-flash',
            'gemini-2.0-flash-lite',
            'gemini-1.5-pro',
            'gemini-1.5-flash',
            'gemini-1.5-flash-8b',
        ]

        for model in expected_models:
            assert model in GEMINI_MODEL_MAX_TOKENS, f'Missing model in mapping: {model}'
            assert GEMINI_MODEL_MAX_TOKENS[model] > 0, f'Invalid max tokens for {model}'

    def test_default_constants(self):
        """Test that default constants are set correctly."""
        assert DEFAULT_MODEL == 'gemini-2.5-flash'
        assert DEFAULT_SMALL_MODEL == 'gemini-2.5-flash-lite'
        assert DEFAULT_GEMINI_MAX_TOKENS == 8192