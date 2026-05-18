import os
from unittest.mock import Mock, patch

import pytest

from mem0.configs.llms.base import BaseLlmConfig
from mem0.configs.llms.minimax import MinimaxConfig
from mem0.llms.minimax import MiniMaxLLM
from mem0.utils.factory import LlmFactory


def clear_minimax_env():
    for name in ("MINIMAX_API_BASE", "MINIMAX_BASE_URL", "MINIMAX_MODEL", "MINIMAX_API_KEY"):
        os.environ.pop(name, None)


@pytest.fixture
def mock_minimax_client():
    with patch("mem0.llms.minimax.OpenAI") as mock_openai:
        mock_client = Mock()
        mock_openai.return_value = mock_client
        yield mock_client


def test_minimax_llm_default_base_url():
    """Default config uses MiniMax official base URL."""
    with patch.dict(os.environ, {}, clear=False):
        clear_minimax_env()
        config = BaseLlmConfig(model="MiniMax-M2.7", temperature=0.7, max_tokens=100, top_p=1.0, api_key="api_key")
        llm = MiniMaxLLM(config)
        # OpenAI client may normalize URL with trailing slash
        assert str(llm.client.base_url).rstrip("/") == "https://api.minimax.io/v1"


def test_minimax_llm_env_base_url():
    """Config uses MINIMAX_API_BASE env variable when set."""
    with patch.dict(os.environ, {}, clear=False):
        clear_minimax_env()
        provider_base_url = "https://api.provider.com/v1/"
        os.environ["MINIMAX_API_BASE"] = provider_base_url
        config = MinimaxConfig(
            model="MiniMax-M2.7",
            temperature=0.7,
            max_tokens=100,
            top_p=1.0,
            api_key="api_key",
        )
        llm = MiniMaxLLM(config)
        assert str(llm.client.base_url).rstrip("/") == provider_base_url.rstrip("/")


def test_minimax_llm_env_base_url_alias():
    """Config also supports MINIMAX_BASE_URL env variable."""
    with patch.dict(os.environ, {}, clear=False):
        clear_minimax_env()
        provider_base_url = "https://api.minimaxi.com/v1/"
        os.environ["MINIMAX_BASE_URL"] = provider_base_url
        config = MinimaxConfig(
            model="MiniMax-M2.7",
            temperature=0.7,
            max_tokens=100,
            top_p=1.0,
            api_key="api_key",
        )
        llm = MiniMaxLLM(config)
        assert str(llm.client.base_url).rstrip("/") == provider_base_url.rstrip("/")


def test_minimax_llm_config_base_url():
    """Config uses minimax_base_url when provided."""
    config_base_url = "https://api.config.com/v1/"
    config = MinimaxConfig(
        model="MiniMax-M2.7",
        temperature=0.7,
        max_tokens=100,
        top_p=1.0,
        api_key="api_key",
        minimax_base_url=config_base_url,
    )
    llm = MiniMaxLLM(config)
    assert str(llm.client.base_url).rstrip("/") == config_base_url.rstrip("/")


def test_minimax_llm_default_model(mock_minimax_client):
    """Default model is MiniMax-M2.7 when not specified."""
    with patch.dict(os.environ, {}, clear=False):
        clear_minimax_env()
        config = MinimaxConfig(temperature=0.7, max_tokens=100, api_key="api_key")
        llm = MiniMaxLLM(config)
        assert llm.config.model == "MiniMax-M2.7"


def test_minimax_llm_env_model(mock_minimax_client):
    """Default model can be supplied via MINIMAX_MODEL."""
    with patch.dict(os.environ, {}, clear=False):
        clear_minimax_env()
        os.environ["MINIMAX_MODEL"] = "MiniMax-M2.7"
        config = MinimaxConfig(temperature=0.7, max_tokens=100, api_key="api_key")
        llm = MiniMaxLLM(config)
        assert llm.config.model == "MiniMax-M2.7"


def test_minimax_llm_env_api_key():
    """Uses MINIMAX_API_KEY env when api_key not in config."""
    with patch.dict(os.environ, {}, clear=False):
        clear_minimax_env()
        os.environ["MINIMAX_API_KEY"] = "env-api-key"
        with patch("mem0.llms.minimax.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            config = MinimaxConfig(model="MiniMax-M2.7", api_key=None)
            MiniMaxLLM(config)
            mock_openai.assert_called_once_with(
                api_key="env-api-key",
                base_url="https://api.minimax.io/v1",
            )


def test_generate_response_without_tools(mock_minimax_client):
    """generate_response returns text when no tools provided."""
    config = BaseLlmConfig(model="MiniMax-M2.7", temperature=0.7, max_tokens=100, top_p=1.0, api_key="api_key")
    llm = MiniMaxLLM(config)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]

    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="I'm doing well, thank you for asking!"))]
    mock_minimax_client.chat.completions.create.return_value = mock_response

    response = llm.generate_response(messages)

    mock_minimax_client.chat.completions.create.assert_called_once_with(
        model="MiniMax-M2.7",
        messages=messages,
        temperature=0.7,
        max_completion_tokens=100,
        top_p=1.0,
        extra_body={"reasoning_split": True},
    )
    assert response == "I'm doing well, thank you for asking!"


def test_generate_response_with_tools(mock_minimax_client):
    """generate_response returns tool_calls when tools provided."""
    config = BaseLlmConfig(model="MiniMax-M2.7", temperature=0.7, max_tokens=100, top_p=1.0, api_key="api_key")
    llm = MiniMaxLLM(config)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Add a new memory: Today is a sunny day."},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "add_memory",
                "description": "Add a memory",
                "parameters": {
                    "type": "object",
                    "properties": {"data": {"type": "string", "description": "Data to add to memory"}},
                    "required": ["data"],
                },
            },
        }
    ]

    mock_response = Mock()
    mock_message = Mock()
    mock_message.content = "I've added the memory for you."

    mock_tool_call = Mock()
    mock_tool_call.function.name = "add_memory"
    mock_tool_call.function.arguments = '{"data": "Today is a sunny day."}'

    mock_message.tool_calls = [mock_tool_call]
    mock_response.choices = [Mock(message=mock_message)]
    mock_minimax_client.chat.completions.create.return_value = mock_response

    response = llm.generate_response(messages, tools=tools)

    mock_minimax_client.chat.completions.create.assert_called_once_with(
        model="MiniMax-M2.7",
        messages=messages,
        temperature=0.7,
        max_completion_tokens=100,
        top_p=1.0,
        extra_body={"reasoning_split": True},
        tools=tools,
        tool_choice="auto",
    )

    assert response["content"] == "I've added the memory for you."
    assert len(response["tool_calls"]) == 1
    assert response["tool_calls"][0]["name"] == "add_memory"
    assert response["tool_calls"][0]["arguments"] == {"data": "Today is a sunny day."}


def test_generate_response_with_response_format(mock_minimax_client):
    """generate_response passes response_format to the API."""
    config = BaseLlmConfig(model="MiniMax-M2.7", temperature=0.7, max_tokens=100, top_p=1.0, api_key="api_key")
    llm = MiniMaxLLM(config)
    messages = [{"role": "user", "content": "Return JSON."}]
    response_format = {"type": "json_object"}

    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content='{"key": "value"}'))]
    mock_minimax_client.chat.completions.create.return_value = mock_response

    llm.generate_response(messages, response_format=response_format)

    mock_minimax_client.chat.completions.create.assert_called_once_with(
        model="MiniMax-M2.7",
        messages=messages,
        temperature=0.7,
        max_completion_tokens=100,
        top_p=1.0,
        extra_body={"reasoning_split": True},
        response_format={"type": "json_object"},
    )


def test_generate_response_allows_disabling_reasoning_split(mock_minimax_client):
    """reasoning_split can be disabled for MiniMax-compatible gateways that do not support it."""
    config = MinimaxConfig(
        model="MiniMax-M2.7",
        temperature=0.7,
        max_tokens=100,
        top_p=1.0,
        api_key="api_key",
        reasoning_split=False,
    )
    llm = MiniMaxLLM(config)
    messages = [{"role": "user", "content": "Return plain text."}]

    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="ok"))]
    mock_minimax_client.chat.completions.create.return_value = mock_response

    llm.generate_response(messages)

    mock_minimax_client.chat.completions.create.assert_called_once_with(
        model="MiniMax-M2.7",
        messages=messages,
        temperature=0.7,
        max_completion_tokens=100,
        top_p=1.0,
        extra_body={"reasoning_split": False},
    )


def test_factory_creates_minimax_llm(mock_minimax_client):
    """LlmFactory.create returns MiniMaxLLM for provider 'minimax'."""
    llm = LlmFactory.create("minimax", {"model": "MiniMax-M2.7", "api_key": "test-key"})
    assert isinstance(llm, MiniMaxLLM)
    assert llm.config.model == "MiniMax-M2.7"
