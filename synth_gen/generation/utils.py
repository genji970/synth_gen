# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, List, Union

from synth_gen.schema import AIMessage, BaseMessage, ChatPromptValue, HumanMessage


def convert_prompt(
    self, prompt: Union[str, BaseMessage, ChatPromptValue]
) -> ChatPromptValue:
    if isinstance(prompt, str):
        chat_history = ChatPromptValue(messages=[HumanMessage(content=prompt)])
        return chat_history
    elif isinstance(prompt, BaseMessage):
        chat_history = ChatPromptValue(messages=[prompt])
        return chat_history
    elif isinstance(prompt, ChatPromptValue):
        return prompt


def convert_response(
    self, response: Union[str, BaseMessage, List[str], List[BaseMessage]]
) -> List[BaseMessage]:
    if isinstance(response, str):
        return [AIMessage(content=response)]
    elif isinstance(response, BaseMessage):
        return [response]
    elif isinstance(response, list):
        if isinstance(response[0], str):
            return [AIMessage(content=response[0])]
        elif all(isinstance(ans, BaseMessage) for ans in response):
            response = cast(List[BaseMessage], response)
            return response
        else:
            raise ValueError("Unknown type of response.")
    else:
        raise ValueError("Unknown type of response.")
