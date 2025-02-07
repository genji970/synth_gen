# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
from typing import Any, List, Optional

import requests

from langchain.schema import ChatGeneration
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs.chat_result import ChatResult


def call_local_server(prompt):
    endpoint = "http://localhost:8000/generate"
    headers = {"Content-Type": "application/json"}
    # Make the request
    response = requests.get(endpoint, headers=headers, params={"prompt": prompt})

    # Check if the response was successful
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to retrieve data: {response.text}")


class LocalServerChatModel(BaseChatModel):
    def __init__(self):
        super().__init__()

    def _llm_type(self) -> str:
        return "local_server"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = messages[-1].content
        response = call_local_server(message)
        message = AIMessage(content=response["text"])
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])


local_server_chat_model = LocalServerChatModel()


if __name__ == "__main__":
    response = local_server_chat_model.invoke("1+3?")
    print(response.content)
