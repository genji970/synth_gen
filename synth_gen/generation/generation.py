# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from re import A
from types import SimpleNamespace
from typing import Any, Generic, Optional, TypeVar

from langchain.schema import BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel as LLM
from langchain_core.prompts import chat
from synth_gen.generation.schema import (
    AIMessage,
    ChatPromptValue,
    HumanMessage,
    SystemMessage,
)


@dataclass
class DataGeneration(SimpleNamespace):
    seed: Optional[Any] = None
    chat: Optional[ChatPromptValue] = None
    is_correct: Optional[bool] = None

    def to_json(self):
        if self.chat:
            messages_json = []
            for message in self.chat.messages:
                role = None
                if isinstance(message, HumanMessage):
                    role = "human"
                elif isinstance(message, AIMessage):
                    role = "ai"
                elif isinstance(message, SystemMessage):
                    role = "system"
                messages_json.append({"role": role, "content": message.content})
            chat_json = {"messages": messages_json}
        else:
            chat_json = None
        return {
            "chat": chat_json,
            "is_correct": self.is_correct,
        }


GenerationType = TypeVar(
    "GenerationType",
    bound=DataGeneration,
)


@dataclass
class PromptAnswerGeneration(DataGeneration):
    prompt: Optional[ChatPromptValue] = None
    answer: Optional[BaseMessage] = None


if __name__ == "__main__":
    chat = ChatPromptValue(
        messages=[
            HumanMessage(content="What is the capital of France?"),
            AIMessage(content="Paris"),
            HumanMessage(content="What is the capital of Italy?"),
            AIMessage(content="Rome"),
            HumanMessage(content="What is the capital of Spain?"),
            AIMessage(content="Madrid"),
        ]
    )
    generation = DataGeneration(chat=chat, is_correct=True)
    print(generation.to_json())
