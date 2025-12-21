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

import hashlib
import json
import logging
import threading
import time
import typing
from abc import ABC, abstractmethod
from enum import Enum

import httpx
from diskcache import Cache
from pydantic import BaseModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

from ..prompts.models import Message
from ..tracer import NoOpTracer, Tracer
from .config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from .errors import RateLimitError

DEFAULT_TEMPERATURE = 0
DEFAULT_CACHE_DIR = './llm_cache'


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""

    def __init__(self, circuit_name: str, failure_count: int = 0):
        self.circuit_name = circuit_name
        self.failure_count = failure_count
        super().__init__(f"Circuit breaker '{circuit_name}' is open after {failure_count} failures")


class CircuitBreaker:
    """Simple circuit breaker implementation for LLM calls.

    Prevents cascading failures by temporarily stopping calls to failing services.
    Circuit opens after failure threshold and stays open for reset timeout.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: int = 30,
        success_threshold: int = 3,
        name: str = "llm_circuit"
    ):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.success_threshold = success_threshold
        self.name = name

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0
        self._lock = threading.RLock()

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt circuit reset."""
        return time.time() - self._last_failure_time >= self.reset_timeout

    def _call_succeeded(self):
        """Handle successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info(f"Circuit breaker '{self.name}' closed after successful recovery")
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success in closed state
                self._failure_count = 0

    def _call_failed(self):
        """Handle failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.warning(
                        f"Circuit breaker '{self.name}' opened after {self._failure_count} failures"
                    )
            elif self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning(f"Circuit breaker '{self.name}' re-opened after failure in half-open state")

    def call(self, func: typing.Callable, *args, **kwargs):
        """Execute function through circuit breaker."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    logger.info(f"Circuit breaker '{self.name}' entering half-open state")
                else:
                    raise CircuitBreakerError(self.name, self._failure_count)

        try:
            result = func(*args, **kwargs)
            self._call_succeeded()
            return result
        except Exception as e:
            self._call_failed()
            raise

    async def call_async(self, coro: typing.Awaitable):
        """Execute async function through circuit breaker."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    logger.info(f"Circuit breaker '{self.name}' entering half-open state")
                else:
                    raise CircuitBreakerError(self.name, self._failure_count)

        try:
            result = await coro
            self._call_succeeded()
            return result
        except Exception as e:
            self._call_failed()
            raise

    def get_state(self) -> typing.Dict[str, typing.Any]:
        """Get current circuit breaker state."""
        with self._lock:
            return {
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "last_failure_time": self._last_failure_time,
                "failure_threshold": self.failure_threshold,
                "success_threshold": self.success_threshold,
                "reset_timeout": self.reset_timeout
            }


def get_extraction_language_instruction(group_id: str | None = None) -> str:
    """Returns instruction for language extraction behavior.

    Override this function to customize language extraction:
    - Return empty string to disable multilingual instructions
    - Return custom instructions for specific language requirements
    - Use group_id to provide different instructions per group/partition

    Args:
        group_id: Optional partition identifier for the graph

    Returns:
        str: Language instruction to append to system messages
    """
    return (
        '\n\nAny extracted information should be returned in the same language as it was written in. '
        'Only output non-English text when the user has written full sentences or phrases in that non-English language. '
        'Otherwise, output English.'
    )


logger = logging.getLogger(__name__)


def is_server_or_retry_error(exception):
    if isinstance(exception, RateLimitError | json.decoder.JSONDecodeError):
        return True

    return (
        isinstance(exception, httpx.HTTPStatusError) and 500 <= exception.response.status_code < 600
    )


class LLMClient(ABC):
    def __init__(self, config: LLMConfig | None, cache: bool = False):
        if config is None:
            config = LLMConfig()

        self.config = config
        self.model = config.model
        self.small_model = config.small_model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.cache_enabled = cache
        self.cache_dir = None
        self.tracer: Tracer = NoOpTracer()

        # Initialize circuit breaker with configurable parameters
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=getattr(config, 'circuit_failure_threshold', 5),
            reset_timeout=getattr(config, 'circuit_reset_timeout', 30),
            success_threshold=getattr(config, 'circuit_success_threshold', 3),
            name=f"llm_client_{self.__class__.__name__.lower()}"
        )

        # Only create the cache directory if caching is enabled
        if self.cache_enabled:
            self.cache_dir = Cache(DEFAULT_CACHE_DIR)

    def set_tracer(self, tracer: Tracer) -> None:
        """Set the tracer for this LLM client."""
        self.tracer = tracer

    def get_circuit_breaker_state(self) -> typing.Dict[str, typing.Any]:
        """Get the current state of the circuit breaker."""
        return self.circuit_breaker.get_state()

    def _clean_input(self, input: str) -> str:
        """Clean input string of invalid unicode and control characters.

        Args:
            input: Raw input string to be cleaned

        Returns:
            Cleaned string safe for LLM processing
        """
        # Clean any invalid Unicode
        cleaned = input.encode('utf-8', errors='ignore').decode('utf-8')

        # Remove zero-width characters and other invisible unicode
        zero_width = '\u200b\u200c\u200d\ufeff\u2060'
        for char in zero_width:
            cleaned = cleaned.replace(char, '')

        # Remove control characters except newlines, returns, and tabs
        cleaned = ''.join(char for char in cleaned if ord(char) >= 32 or char in '\n\r\t')

        return cleaned

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_random_exponential(multiplier=10, min=5, max=120),
        retry=retry_if_exception(is_server_or_retry_error),
        after=lambda retry_state: logger.warning(
            f'Retrying {retry_state.fn.__name__ if retry_state.fn else "function"} after {retry_state.attempt_number} attempts...'
        )
        if retry_state.attempt_number > 1
        else None,
        reraise=True,
    )
    async def _generate_response_with_retry(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        """Generate response with retry logic and circuit breaker protection."""
        async def _generate_response_internal():
            try:
                return await self._generate_response(messages, response_model, max_tokens, model_size)
            except (httpx.HTTPStatusError, RateLimitError) as e:
                raise e

        # Wrap the actual call with circuit breaker
        try:
            return await self.circuit_breaker.call_async(_generate_response_internal())
        except CircuitBreakerError as e:
            # Log circuit breaker events for observability
            logger.error(
                f"Circuit breaker '{e.circuit_name}' is open after {e.failure_count} failures. "
                f"LLM calls are temporarily disabled."
            )
            raise
        except Exception as e:
            # Other exceptions will be handled by the retry decorator
            raise

    @abstractmethod
    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        pass

    def _get_cache_key(self, messages: list[Message]) -> str:
        # Create a unique cache key based on the messages and model
        message_str = json.dumps([m.model_dump() for m in messages], sort_keys=True)
        key_str = f'{self.model}:{message_str}'
        return hashlib.md5(key_str.encode()).hexdigest()

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None,
        prompt_name: str | None = None,
    ) -> dict[str, typing.Any]:
        if max_tokens is None:
            max_tokens = self.max_tokens

        if response_model is not None:
            serialized_model = json.dumps(response_model.model_json_schema())
            messages[
                -1
            ].content += (
                f'\n\nRespond with a JSON object in the following format:\n\n{serialized_model}'
            )

        # Add multilingual extraction instructions
        messages[0].content += get_extraction_language_instruction(group_id)

        for message in messages:
            message.content = self._clean_input(message.content)

        # Wrap entire operation in tracing span
        with self.tracer.start_span('llm.generate') as span:
            # Get circuit breaker state for observability
            cb_state = self.circuit_breaker.get_state()

            attributes = {
                'llm.provider': self._get_provider_type(),
                'model.size': model_size.value,
                'max_tokens': max_tokens,
                'cache.enabled': self.cache_enabled,
                'circuit_breaker.state': cb_state['state'],
                'circuit_breaker.failure_count': cb_state['failure_count'],
                'circuit_breaker.name': self.circuit_breaker.name,
            }
            if prompt_name:
                attributes['prompt.name'] = prompt_name
            span.add_attributes(attributes)

            # Check cache first
            if self.cache_enabled and self.cache_dir is not None:
                cache_key = self._get_cache_key(messages)
                cached_response = self.cache_dir.get(cache_key)
                if cached_response is not None:
                    logger.debug(f'Cache hit for {cache_key}')
                    span.add_attributes({'cache.hit': True})
                    return cached_response

            span.add_attributes({'cache.hit': False})

            # Execute LLM call
            try:
                response = await self._generate_response_with_retry(
                    messages, response_model, max_tokens, model_size
                )

                # Add circuit breaker state after successful call
                final_cb_state = self.circuit_breaker.get_state()
                span.add_attribute('circuit_breaker.final_state', final_cb_state['state'])

            except CircuitBreakerError as e:
                # Record circuit breaker specific events
                span.add_event('circuit_breaker_open', {
                    'circuit_name': e.circuit_name,
                    'failure_count': e.failure_count,
                    'action': 'llm_call_blocked'
                })
                span.set_status('error', f'Circuit breaker {e.circuit_name} is open')
                span.record_exception(e)
                raise
            except Exception as e:
                # Add circuit breaker state after failed call
                final_cb_state = self.circuit_breaker.get_state()
                span.add_attribute('circuit_breaker.final_state', final_cb_state['state'])

                span.set_status('error', str(e))
                span.record_exception(e)
                raise

            # Cache response if enabled
            if self.cache_enabled and self.cache_dir is not None:
                cache_key = self._get_cache_key(messages)
                self.cache_dir.set(cache_key, response)

            return response

    def _get_provider_type(self) -> str:
        """Get provider type from class name."""
        class_name = self.__class__.__name__.lower()
        if 'openai' in class_name:
            return 'openai'
        elif 'anthropic' in class_name:
            return 'anthropic'
        elif 'gemini' in class_name:
            return 'gemini'
        elif 'groq' in class_name:
            return 'groq'
        else:
            return 'unknown'

    def _get_failed_generation_log(self, messages: list[Message], output: str | None) -> str:
        """
        Log the full input messages, the raw output (if any), and the exception for debugging failed generations.
        """
        log = ''
        log += f'Input messages: {json.dumps([m.model_dump() for m in messages], indent=2)}\n'
        if output is not None:
            if len(output) > 4000:
                log += f'Raw output: {output[:2000]}... (truncated) ...{output[-2000:]}\n'
            else:
                log += f'Raw output: {output}\n'
        else:
            log += 'No raw output available'
        return log
