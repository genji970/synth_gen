# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import tempfile
from typing import Any, Optional

import dotenv
import langchain
from diskcache import Cache
from langchain.schema import ChatGeneration
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs.chat_result import ChatResult
from langchain_together import ChatTogether
from synth_gen.generation.parsing import (
    extract_list_from_string,
    extract_source_code as extract_source_code_fct,
    extract_triple_backticked_string,
)
from synth_gen.generation.schema import LLM

# Load environment variables from .env file
script_folder = os.path.dirname(os.path.realpath(__file__))
dotenv.load_dotenv(os.path.join(script_folder, "..", ".env"))

# Check environment variable to set the default LLM model
if "LLM" in os.environ:
    if os.environ["LLM"] == "together_llama3_1_8B":
        default_llm = ChatTogether(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            max_tokens=8192,
        )
    elif os.environ["LLM"] == "together_llama3_1_70B":
        default_llm = ChatTogether(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            max_tokens=8192,
        )
    elif os.environ["LLM"] == "together_llama3_1_405B":
        default_llm = ChatTogether(
            model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            max_tokens=8192,
        )
    elif os.environ["LLM"] == "local_server":
        from synth_gen.generation.local_LLM_server import local_server_chat_model

        default_llm = local_server_chat_model
else:

    class DummyLLM(BaseChatModel):
        def __init__(self):
            super().__init__()

        def _llm_type(self) -> str:
            return "dummy_llm"

        def _generate(
            self,
            messages: list[BaseMessage],
            stop: Optional[list[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> ChatResult:
            raise NotImplementedError(
                "Please choose a LLM model. You can either set the LLM environment variable or use the set_default_llm function which is compatiable with any langchain chat model."
            )

    default_llm = DummyLLM()


if (
    "USE_LLM_CACHE" in os.environ
    and os.environ["USE_LLM_CACHE"] != "0"
    and os.environ["USE_LLM_CACHE"].lower() != "false"
):
    default_use_llm_cache = True
else:
    default_use_llm_cache = False


def set_default_llm(llm: LLM):
    """
    Set the default language model.

    Args:
        llm (LLM): The language model to set as default. Must be a langchain chat model.
    """
    global default_llm
    default_llm = llm


def get_default_llm():
    """
    Get the default language model.

    Returns:
        LLM: The default language model.
    """
    return default_llm


# Initialize cache for storing LLM responses
cache = Cache(os.path.join(tempfile.gettempdir(), "synth_gen/LLM_cache"))


def call_llm(
    prompt: LanguageModelInput,
    llm: LLM = default_llm,
    stream=False,
    parse_json=False,
    extract_triple_backtick=False,
    extract_source_code=False,
    extract_list=False,
    use_cache=default_use_llm_cache,
):
    """
    Call the language model with a given prompt and options.

    Args:
        prompt (str): The input prompt for the language model.
        llm (LLM, optional): The language model to use. Defaults to default_llm.
        stream (bool, optional): Whether to stream the response. Defaults to False.
        parse_json (bool, optional): Whether to parse the response as JSON. Defaults to False.
        extract_triple_backtick (bool, optional): Whether to extract content within triple backticks. Defaults to False.
        extract_source_code (bool, optional): Whether to extract source code from the response. Defaults to False.
        use_cache (bool, optional): Whether to use cached responses. Defaults to False.

    Returns:
        str or dict: The response from the language model, parsed as JSON if specified.
    """
    if isinstance(llm, ChatTogether):
        if not os.environ.get("TOGETHER_API_KEY"):
            print("TOGETHER_API_KEY is missing. Please set it in your environment.")
            raise Exception("TOGETHER_API_KEY not set")
    if stream:
        raise NotImplementedError("Streaming is not implemented yet.")
    else:
        message = None
        if use_cache:
            message = cache.get(prompt)
        if message is None:
            response = llm.invoke(prompt)
            message = str(response.content)
            cache.set(prompt, message)
    if parse_json:
        try:
            return json.loads(message)
        except json.JSONDecodeError:
            return message
    if extract_triple_backtick:
        return extract_triple_backticked_string(message)
    if extract_source_code:
        return extract_source_code_fct(message)
    if extract_list:
        return extract_list_from_string(message)
    return message
