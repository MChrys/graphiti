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

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.prompts.models import Message


class ResponseModel(BaseModel):
    """Test model for response testing."""

    test_field: str
    optional_field: int = 0


@pytest.fixture
def mock_openai_client():
    """Fixture to mock the AsyncOpenAI client."""
    with patch('openai.AsyncOpenAI') as mock_client:
        # Setup mock instance and its methods
        mock_instance = mock_client.return_value
        mock_instance.responses = MagicMock()
        mock_instance.responses.parse = AsyncMock()
        mock_instance.chat = MagicMock()
        mock_instance.chat.completions = MagicMock()
        mock_instance.chat.completions.create = AsyncMock()
        yield mock_instance


@pytest.fixture
def openai_client(mock_openai_client):
    """Fixture to create an OpenAIClient with a mocked client."""
    config = LLMConfig(api_key='test_api_key', model='test-model', temperature=0.5, max_tokens=1000)
    client = OpenAIClient(config=config, cache=False)
    # Replace the client's client with our mock to ensure we're using the mock
    client.client = mock_openai_client
    return client


class TestOpenAIClientInitialization:
    """Tests for OpenAIClient initialization."""

    @patch('openai.AsyncOpenAI')
    def test_init_with_config(self, mock_async_openai):
        """Test initialization with a config object."""
        config = LLMConfig(api_key='test_api_key', model='test-model', temperature=0.5, max_tokens=1000)
        client = OpenAIClient(config=config, cache=False)

        assert client.config == config
        assert client.model == 'test-model'
        assert client.temperature == 0.5
        assert client.max_tokens == 1000
        mock_async_openai.assert_called_once_with(api_key='test_api_key', base_url=None)

    @patch('openai.AsyncOpenAI')
    def test_init_with_base_url(self, mock_async_openai):
        """Test initialization with base URL."""
        config = LLMConfig(
            api_key='test_api_key',
            model='test-model',
            temperature=0.5,
            max_tokens=1000,
            base_url='https://api.openai.com/v1'
        )
        client = OpenAIClient(config=config, cache=False)

        mock_async_openai.assert_called_once_with(api_key='test_api_key', base_url='https://api.openai.com/v1')

    @patch('openai.AsyncOpenAI')
    def test_init_without_config(self, mock_async_openai):
        """Test initialization without a config, using defaults."""
        client = OpenAIClient(cache=False)

        assert client.config is not None
        assert client.config.api_key is None
        mock_async_openai.assert_called_once_with(api_key=None, base_url=None)

    def test_init_with_custom_client(self):
        """Test initialization with a custom AsyncOpenAI client."""
        mock_client = MagicMock()
        client = OpenAIClient(client=mock_client)

        assert client.client == mock_client


class TestOpenAIClientStructuredCompletion:
    """Tests for OpenAIClient structured completion methods."""

    @pytest.mark.asyncio
    async def test_create_structured_completion_standard_model(self, openai_client, mock_openai_client):
        """Test structured completion with standard model (supports temperature)."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.output_text = '{"test_field": "test_value", "optional_field": 42}'
        mock_openai_client.responses.parse.return_value = mock_response

        # Call method
        messages = [
            Message(role='system', content='System message'),
            Message(role='user', content='User message'),
        ]
        result = await openai_client._create_structured_completion(
            model='gpt-4',
            messages=[msg.to_dict() for msg in messages],
            temperature=0.5,
            max_tokens=1000,
            response_model=ResponseModel
        )

        # Assertions
        assert result == mock_response
        mock_openai_client.responses.parse.assert_called_once_with(
            model='gpt-4',
            input=[msg.to_dict() for msg in messages],
            temperature=0.5,
            max_output_tokens=1000,
            text_format=ResponseModel,
            reasoning=None,
            text=None
        )

    @pytest.mark.asyncio
    async def test_create_structured_completion_reasoning_model(self, openai_client, mock_openai_client):
        """Test structured completion with reasoning model (no temperature)."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.output_text = '{"test_field": "test_value"}'
        mock_openai_client.responses.parse.return_value = mock_response

        # Call method
        messages = [Message(role='user', content='Test message')]
        await openai_client._create_structured_completion(
            model='o1-preview',
            messages=[msg.to_dict() for msg in messages],
            temperature=0.5,  # Should be ignored for reasoning models
            max_tokens=1000,
            response_model=ResponseModel
        )

        # Verify temperature is None for reasoning models
        mock_openai_client.responses.parse.assert_called_once_with(
            model='o1-preview',
            input=[msg.to_dict() for msg in messages],
            temperature=None,  # Should be None for reasoning models
            max_output_tokens=1000,
            text_format=ResponseModel,
            reasoning=None,
            text=None
        )

    @pytest.mark.asyncio
    async def test_create_structured_completion_with_reasoning_params(self, openai_client, mock_openai_client):
        """Test structured completion with reasoning and verbosity parameters."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.output_text = '{"test_field": "test_value"}'
        mock_openai_client.responses.parse.return_value = mock_response

        # Call method with reasoning and verbosity
        messages = [Message(role='user', content='Test message')]
        await openai_client._create_structured_completion(
            model='o1-custom',
            messages=[msg.to_dict() for msg in messages],
            temperature=0.7,
            max_tokens=1000,
            response_model=ResponseModel,
            reasoning='intense',
            verbosity='high'
        )

        # Verify reasoning and verbosity are passed correctly
        mock_openai_client.responses.parse.assert_called_once_with(
            model='o1-custom',
            input=[msg.to_dict() for msg in messages],
            temperature=None,  # Should be None for reasoning models
            max_output_tokens=1000,
            text_format=ResponseModel,
            reasoning={'effort': 'intense'},
            text={'verbosity': 'high'}
        )

    @pytest.mark.asyncio
    async def test_create_completion_standard_model(self, openai_client, mock_openai_client):
        """Test regular completion with standard model."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"test_field": "test_value"}'))
        ]
        mock_openai_client.chat.completions.create.return_value = mock_response

        # Call method
        messages = [Message(role='user', content='Test message')]
        await openai_client._create_completion(
            model='gpt-4',
            messages=[msg.to_dict() for msg in messages],
            temperature=0.5,
            max_tokens=1000
        )

        # Verify call parameters
        mock_openai_client.chat.completions.create.assert_called_once_with(
            model='gpt-4',
            messages=[msg.to_dict() for msg in messages],
            temperature=0.5,
            max_tokens=1000,
            response_format={'type': 'json_object'}
        )

    @pytest.mark.asyncio
    async def test_create_completion_reasoning_model(self, openai_client, mock_openai_client):
        """Test regular completion with reasoning model."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"test_field": "test_value"}'))
        ]
        mock_openai_client.chat.completions.create.return_value = mock_response

        # Call method
        messages = [Message(role='user', content='Test message')]
        await openai_client._create_completion(
            model='o1-preview',
            messages=[msg.to_dict() for msg in messages],
            temperature=0.5,  # Should be ignored for reasoning models
            max_tokens=1000
        )

        # Verify temperature is None for reasoning models
        mock_openai_client.chat.completions.create.assert_called_once_with(
            model='o1-preview',
            messages=[msg.to_dict() for msg in messages],
            temperature=None,  # Should be None for reasoning models
            max_tokens=1000,
            response_format={'type': 'json_object'}
        )

    @pytest.mark.asyncio
    async def test_reasoning_model_detection_gpt5(self, openai_client, mock_openai_client):
        """Test reasoning model detection for gpt-5 family."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.output_text = '{"test_field": "test_value"}'
        mock_openai_client.responses.parse.return_value = mock_response

        # Call method with gpt-5 model
        messages = [Message(role='user', content='Test message')]
        await openai_client._create_structured_completion(
            model='gpt-5-turbo',
            messages=[msg.to_dict() for msg in messages],
            temperature=0.5,
            max_tokens=1000,
            response_model=ResponseModel
        )

        # Verify temperature is None for gpt-5 reasoning models
        call_args = mock_openai_client.responses.parse.call_args
        assert call_args[1]['temperature'] is None

    @pytest.mark.asyncio
    async def test_reasoning_model_detection_o3(self, openai_client, mock_openai_client):
        """Test reasoning model detection for o3 family."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.output_text = '{"test_field": "test_value"}'
        mock_openai_client.responses.parse.return_value = mock_response

        # Call method with o3 model
        messages = [Message(role='user', content='Test message')]
        await openai_client._create_structured_completion(
            model='o3-mini',
            messages=[msg.to_dict() for msg in messages],
            temperature=0.5,
            max_tokens=1000,
            response_model=ResponseModel
        )

        # Verify temperature is None for o3 reasoning models
        call_args = mock_openai_client.responses.parse.call_args
        assert call_args[1]['temperature'] is None

    @pytest.mark.asyncio
    async def test_non_reasoning_model_temperature_preserved(self, openai_client, mock_openai_client):
        """Test that temperature is preserved for non-reasoning models."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.output_text = '{"test_field": "test_value"}'
        mock_openai_client.responses.parse.return_value = mock_response

        # Call method with gpt-4 model (non-reasoning)
        messages = [Message(role='user', content='Test message')]
        await openai_client._create_structured_completion(
            model='gpt-4',
            messages=[msg.to_dict() for msg in messages],
            temperature=0.7,
            max_tokens=1000,
            response_model=ResponseModel
        )

        # Verify temperature is preserved for non-reasoning models
        call_args = mock_openai_client.responses.parse.call_args
        assert call_args[1]['temperature'] == 0.7