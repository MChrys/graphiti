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

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from graphiti_core.llm_client.anthropic_client import (
    ANTHROPIC_MODEL_MAX_TOKENS,
    DEFAULT_ANTHROPIC_MAX_TOKENS,
    AnthropicClient,
)
from graphiti_core.llm_client.config import LLMConfig, ModelSize
from graphiti_core.prompts.models import Message


class ResponseModel(BaseModel):
    """Test model for response testing."""

    test_field: str
    optional_field: int = 0


class TestAnthropicClientInitialization:
    """Tests for AnthropicClient initialization."""

    @patch('anthropic.AsyncAnthropic')
    def test_init_with_config(self, mock_async_anthropic):
        """Test initialization with a config object."""
        config = LLMConfig(
            api_key='test_api_key', model='claude-3-5-sonnet-latest', temperature=0.5, max_tokens=1000
        )
        client = AnthropicClient(config=config, cache=False)

        assert client.config == config
        assert client.model == 'claude-3-5-sonnet-latest'
        assert client.temperature == 0.5
        assert client.max_tokens == 1000
        mock_async_anthropic.assert_called_once_with(api_key='test_api_key', max_retries=1)

    @patch('anthropic.AsyncAnthropic')
    def test_init_with_default_model(self, mock_async_anthropic):
        """Test initialization with default model when none is provided."""
        config = LLMConfig(api_key='test_api_key')
        client = AnthropicClient(config=config, cache=False)

        assert client.model == 'claude-haiku-4-5-latest'

    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'env_api_key'})
    @patch('anthropic.AsyncAnthropic')
    def test_init_without_config(self, mock_async_anthropic):
        """Test initialization without a config, using environment variable."""
        client = AnthropicClient(cache=False)

        assert client.config.api_key == 'env_api_key'
        assert client.model == 'claude-haiku-4-5-latest'
        mock_async_anthropic.assert_called_once_with(api_key='env_api_key', max_retries=1)

    @patch('anthropic.AsyncAnthropic')
    def test_init_with_custom_client(self, mock_async_anthropic):
        """Test initialization with a custom AsyncAnthropic client."""
        mock_client = MagicMock()
        client = AnthropicClient(client=mock_client)

        assert client.client == mock_client
        # Should not create a new client
        mock_async_anthropic.assert_not_called()


class TestAnthropicClientHelperMethods:
    """Tests for AnthropicClient helper methods."""

    def test_extract_json_from_text_valid_json(self):
        """Test JSON extraction from valid JSON embedded in text."""
        client = AnthropicClient(config=LLMConfig(api_key='test'), cache=False)

        # Valid JSON embedded in text
        text = 'Some text before {"test_field": "value", "optional_field": 42} and after'
        result = client._extract_json_from_text(text)

        assert result == {'test_field': 'value', 'optional_field': 42}

    def test_extract_json_from_text_invalid_json(self):
        """Test JSON extraction with invalid JSON raises ValueError."""
        client = AnthropicClient(config=LLMConfig(api_key='test'), cache=False)

        # Invalid JSON
        with pytest.raises(ValueError, match='Could not extract JSON from model response'):
            client._extract_json_from_text('Not JSON at all')

    def test_extract_json_from_text_no_braces(self):
        """Test JSON extraction when no braces are found raises ValueError."""
        client = AnthropicClient(config=LLMConfig(api_key='test'), cache=False)

        # No JSON braces
        with pytest.raises(ValueError, match='Could not extract JSON from model response'):
            client._extract_json_from_text('No JSON structure here')

    def test_extract_json_from_text_malformed_json(self):
        """Test JSON extraction with malformed JSON raises ValueError."""
        client = AnthropicClient(config=LLMConfig(api_key='test'), cache=False)

        # Malformed JSON
        with pytest.raises(ValueError, match='Could not extract JSON from model response'):
            client._extract_json_from_text('{"incomplete": "json"')

    def test_create_tool_with_response_model(self):
        """Test tool creation with a response model."""
        client = AnthropicClient(config=LLMConfig(api_key='test'), cache=False)

        tools, tool_choice = client._create_tool(ResponseModel)

        assert len(tools) == 1
        assert tools[0]['name'] == 'ResponseModel'
        assert 'properties' in tools[0]['input_schema']
        assert 'test_field' in tools[0]['input_schema']['properties']
        assert tool_choice['type'] == 'tool'
        assert tool_choice['name'] == 'ResponseModel'

    def test_create_tool_without_response_model(self):
        """Test tool creation without a response model (generic JSON)."""
        client = AnthropicClient(config=LLMConfig(api_key='test'), cache=False)

        tools, tool_choice = client._create_tool()

        assert len(tools) == 1
        assert tools[0]['name'] == 'generic_json_output'
        assert tools[0]['input_schema']['type'] == 'object'
        assert tool_choice['type'] == 'tool'
        assert tool_choice['name'] == 'generic_json_output'

    def test_create_tool_with_model_description(self):
        """Test tool creation respects model schema description."""

        class DescribedModel(BaseModel):
            """A test model with description."""
            test_field: str

            model_config = {
                "json_schema_extra": {
                    "description": "Custom model description"
                }
            }

        client = AnthropicClient(config=LLMConfig(api_key='test'), cache=False)

        tools, _ = client._create_tool(DescribedModel)

        assert tools[0]['description'] == 'Custom model description'

    def test_clean_input_method(self):
        """Test the _clean_input method inherited from LLMClient."""
        client = AnthropicClient(config=LLMConfig(api_key='test'), cache=False)

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


class TestAnthropicClientModelSizeLogic:
    """Tests for model size selection logic."""

    def test_get_model_for_size_small(self):
        """Test small model size selection."""
        client = AnthropicClient(config=LLMConfig(api_key='test'), cache=False)

        small_model = client._get_model_for_size(ModelSize.small)
        # AnthropicClient uses the same model for all sizes since it doesn't have model size mapping
        assert small_model == client.model

    def test_get_model_for_size_medium(self):
        """Test medium model size selection."""
        client = AnthropicClient(config=LLMConfig(api_key='test'), cache=False)

        medium_model = client._get_model_for_size(ModelSize.medium)
        assert medium_model == client.model

    def test_get_model_for_size_large(self):
        """Test large model size selection."""
        client = AnthropicClient(config=LLMConfig(api_key='test'), cache=False)

        large_model = client._get_model_for_size(ModelSize.large)
        assert large_model == client.model

    def test_max_tokens_for_known_model(self):
        """Test max tokens mapping for known Anthropic models."""
        config = LLMConfig(api_key='test', model='claude-3-5-sonnet-latest')
        client = AnthropicClient(config=config, cache=False)

        # Should use the mapped value for claude-3-5-sonnet-latest
        assert client.max_tokens == ANTHROPIC_MODEL_MAX_TOKENS['claude-3-5-sonnet-latest']

    def test_max_tokens_for_unknown_model(self):
        """Test max tokens for unknown model uses default."""
        config = LLMConfig(api_key='test', model='unknown-model')
        client = AnthropicClient(config=config, cache=False)

        # Should use default max tokens
        assert client.max_tokens == DEFAULT_ANTHROPIC_MAX_TOKENS

    def test_explicit_max_tokens_takes_precedence(self):
        """Test that explicit max_tokens parameter takes precedence over model mapping."""
        config = LLMConfig(api_key='test', model='claude-3-5-sonnet-latest')
        client = AnthropicClient(config=config, cache=False, max_tokens=5000)

        # Should use explicit value, not model mapping
        assert client.max_tokens == 5000


class TestAnthropicClientMessageProcessing:
    """Tests for message processing methods."""

    def test_convert_messages_to_anthropic_format(self):
        """Test converting Message objects to Anthropic format."""
        client = AnthropicClient(config=LLMConfig(api_key='test'), cache=False)

        messages = [
            Message(role='system', content='System message'),
            Message(role='user', content='User message'),
            Message(role='assistant', content='Assistant message'),
        ]

        anthropic_messages = client._convert_messages(messages)

        # System message should be included
        assert len(anthropic_messages) == 3
        assert anthropic_messages[0]['role'] == 'user'  # System message converted to user
        assert 'System message: ' in anthropic_messages[0]['content']
        assert anthropic_messages[1]['role'] == 'user'
        assert anthropic_messages[1]['content'] == 'User message'
        assert anthropic_messages[2]['role'] == 'assistant'
        assert anthropic_messages[2]['content'] == 'Assistant message'

    def test_convert_messages_without_system(self):
        """Test converting messages without system message."""
        client = AnthropicClient(config=LLMConfig(api_key='test'), cache=False)

        messages = [
            Message(role='user', content='User message'),
            Message(role='assistant', content='Assistant message'),
        ]

        anthropic_messages = client._convert_messages(messages)

        assert len(anthropic_messages) == 2
        assert anthropic_messages[0]['role'] == 'user'
        assert anthropic_messages[1]['role'] == 'assistant'

    def test_convert_messages_empty_list(self):
        """Test converting empty message list."""
        client = AnthropicClient(config=LLMConfig(api_key='test'), cache=False)

        messages = []
        anthropic_messages = client._convert_messages(messages)

        assert anthropic_messages == []


class TestAnthropicClientConfigProperties:
    """Tests for client configuration properties."""

    def test_model_property_validation(self):
        """Test that model property accepts valid Anthropic models."""
        valid_models = [
            'claude-3-5-sonnet-latest',
            'claude-3-haiku-20240307',
            'claude-2.1',
        ]

        for model in valid_models:
            config = LLMConfig(api_key='test', model=model)
            client = AnthropicClient(config=config, cache=False)
            assert client.model == model

    def test_temperature_property_inheritance(self):
        """Test temperature property from LLMConfig."""
        config = LLMConfig(api_key='test', temperature=0.7)
        client = AnthropicClient(config=config, cache=False)

        assert client.temperature == 0.7

    def test_max_tokens_property_inheritance(self):
        """Test max_tokens property from LLMConfig."""
        config = LLMConfig(api_key='test', max_tokens=2000)
        client = AnthropicClient(config=config, cache=False)

        assert client.max_tokens == 2000

    def test_config_property_preservation(self):
        """Test that config property is preserved exactly."""
        config = LLMConfig(
            api_key='test_key',
            model='claude-3-5-sonnet-latest',
            temperature=0.5,
            max_tokens=1000,
            base_url='https://api.anthropic.com'
        )
        client = AnthropicClient(config=config, cache=False)

        assert client.config is config
        assert client.config.api_key == 'test_key'
        assert client.config.base_url == 'https://api.anthropic.com'