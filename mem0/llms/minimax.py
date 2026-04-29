import json
import os
from typing import Dict, List, Optional, Union

from openai import OpenAI

from mem0.configs.llms.base import BaseLlmConfig
from mem0.configs.llms.minimax import MinimaxConfig
from mem0.llms.base import LLMBase
from mem0.memory.utils import extract_json


class MiniMaxLLM(LLMBase):
    def __init__(self, config: Optional[Union[BaseLlmConfig, MinimaxConfig, Dict]] = None):
        # Convert to MinimaxConfig if needed
        if config is None:
            config = MinimaxConfig()
        elif isinstance(config, dict):
            config = MinimaxConfig(**config)
        elif isinstance(config, BaseLlmConfig) and not isinstance(config, MinimaxConfig):
            # Convert BaseLlmConfig to MinimaxConfig
            config = MinimaxConfig(
                model=config.model,
                temperature=config.temperature,
                api_key=config.api_key,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                top_k=config.top_k,
                enable_vision=config.enable_vision,
                vision_details=config.vision_details,
                http_client_proxies=config.http_client,
            )

        super().__init__(config)

        if not self.config.model:
            self.config.model = os.getenv("MINIMAX_MODEL") or "MiniMax-M2.7"

        api_key = self.config.api_key or os.getenv("MINIMAX_API_KEY")
        base_url = (
            self.config.minimax_base_url
            or os.getenv("MINIMAX_BASE_URL")
            or os.getenv("MINIMAX_API_BASE")
            or "https://api.minimax.io/v1"
        )
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _get_minimax_params(self, messages: List[Dict[str, str]], **kwargs) -> Dict:
        params = self._get_supported_params(messages=messages, **kwargs)
        max_tokens = params.pop("max_tokens", None)
        if max_tokens is not None:
            params["max_completion_tokens"] = max_tokens

        extra_body = dict(getattr(self.config, "extra_body", {}) or {})
        reasoning_split = getattr(self.config, "reasoning_split", None)
        if reasoning_split is not None:
            extra_body.setdefault("reasoning_split", reasoning_split)
        if extra_body:
            params["extra_body"] = extra_body
        return params

    def _parse_response(self, response, tools):
        """
        Process the response based on whether tools are used or not.

        Args:
            response: The raw response from API.
            tools: The list of tools provided in the request.

        Returns:
            str or dict: The processed response.
        """
        if tools:
            processed_response = {
                "content": response.choices[0].message.content,
                "tool_calls": [],
            }

            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    processed_response["tool_calls"].append(
                        {
                            "name": tool_call.function.name,
                            "arguments": json.loads(extract_json(tool_call.function.arguments)),
                        }
                    )

            return processed_response
        else:
            return response.choices[0].message.content

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ):
        """
        Generate a response based on the given messages using MiniMax.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            response_format (str or object, optional): Format of the response. Defaults to None.
            tools (list, optional): List of tools that the model can call. Defaults to None.
            tool_choice (str, optional): Tool choice method. Defaults to "auto".
            **kwargs: Additional MiniMax-specific parameters.

        Returns:
            str: The generated response.
        """
        params = self._get_minimax_params(messages=messages, **kwargs)
        params.update(
            {
                "model": self.config.model,
                "messages": messages,
            }
        )

        if response_format:
            params["response_format"] = response_format

        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        response = self.client.chat.completions.create(**params)
        return self._parse_response(response, tools)
