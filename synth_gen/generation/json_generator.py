import json
import re
from typing import Any, Dict, Iterator, Optional, Tuple, Union

import jsonschema

from datasets.utils import extract
from langchain.schema import AIMessage, HumanMessage
from langchain_core.prompt_values import ChatPromptValue

from synth_gen.generation.parsing import (
    extract_source_code,
    extract_triple_backticked_string,
)
from synth_gen.generation.seed_source import (
    EnglishVocabularySource,
    SeedSource,
    zipSeedSources,
)

from synth_gen.generators import VerifiedGeneration, VerifiedGenerator
from synth_gen.schema import BaseMessage
from synth_gen.verification.schema import Verification, Verifier


class ToolCallSeedSource(SeedSource):
    tools = [
        {
            "name": "calculator",
            "description": "A tool for performing mathematical calculations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "formula": {
                        "type": "string",
                        "description": "The mathematical formula to be evaluated.",
                    }
                },
            },
        },
        {
            "name": "weather",
            "description": "A tool for retrieving weather information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location for which weather information is requested.",
                    }
                },
            },
        },
        {
            "name": "time",
            "description": "A tool for retrieving the current time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location for which the current time is requested.",
                    }
                },
            },
        },
    ]
    iterator: Optional[Iterator[Dict[str, Any]]] = None

    def iter_next_seed(self) -> Iterator[Dict[str, Any]]:
        current_tool_index = -1
        while True:
            current_tool_index += 1
            if current_tool_index >= len(self.tools):
                current_tool_index = 0
            for use_tool in [True, False]:
                yield {"use_tool": use_tool, "tool": self.tools[current_tool_index]}

    def get_next_seed(self) -> Dict[str, Any]:
        if self.iterator is None:
            self.iterator = self.iter_next_seed()
        return next(self.iterator)


tool_and_inspiration_seed_source = zipSeedSources(
    ToolCallSeedSource(), EnglishVocabularySource()
)


class ToolCallGeneration(VerifiedGeneration):
    tool_name: str
    tool_description: str
    tool_parameters: Dict[str, Any]


def format_json_error_message(error: json.JSONDecodeError, json_str: str) -> str:
    return f"Failed to decode JSON:\nError: {error.msg}\nLine: {error.lineno}, Column: {error.colno}\nPosition: {error.pos}\nContext: {json_str[max(0, error.pos-20):error.pos+20]}"


class JsonSchemaVerifier(Verifier):
    def check(self, generation: VerifiedGeneration) -> Verification:
        json_str = extract_source_code(generation.response.content)
        json_schema_str = generation.json_schema
        verification = Verification(
            generation=generation, verifier_name=self.__class__.__name__
        )
        if json_str is None:
            verification.is_verified = False
            verification.error_message = "The response does not contain a JSON."
            return verification
        try:
            json_data = json.loads(json_str)
        except json.JSONDecodeError as e:
            verification.is_verified = False
            error_message = format_json_error_message(e, json_str)
            verification.error_message = f"This is not a valid JSON: {error_message}"
            return verification
        if type(json_schema_str) is str:
            try:
                json_schema = json.loads(json_schema_str)
            except json.JSONDecodeError as e:
                verification.is_verified = False
                error_message = format_json_error_message(e, json_schema_str)
                verification.error_message = (
                    f"The JSON schema is not a valid JSON: {error_message}"
                )
                verification.is_unfixable = True
                return verification
        else:
            json_schema = json_schema_str

        try:
            jsonschema.validate(instance=json_data, schema=json_schema)
            verification.is_verified = True
            return verification
        except jsonschema.ValidationError as e:
            verification.is_verified = False
            verification.error_message = f"This JSON does not match the provided JSON schema: {json.dumps(json_schema)}\n{e.message}"
            return verification
        except json.JSONDecodeError as e:
            verification.is_verified = False
            verification.error_message = f"While verifying that the provided JSON schema matches the provided JSON schema, the following error happened: {e.msg}"
            return verification
        except jsonschema.SchemaError as e:
            verification.is_verified = False
            verification.error_message = (
                f"The provided JSON schema is not a valid JSON schema: {e.message}"
            )
            verification.is_unfixable = True
            return verification


class ToolCallGenerator(VerifiedGenerator[ToolCallGeneration]):
    """
    A class for generating tool calls for a given tool and input data.
    """

    def __init__(self):
        verifiers = [JsonSchemaVerifier()]
        super().__init__(
            seed_source=tool_and_inspiration_seed_source, verifiers=verifiers
        )

    def _generate_prompt(self, seed: Tuple[Dict[str, Any], str]) -> ToolCallGeneration:
        tool, inspiration = seed
        prompt_prompt = f"""
Make a prompt that should be sent to an AI.
The prompt should require the AI to use the following tool:
- tool name: {tool['tool']['name']}
- tool description: {tool['tool']['description']}

To get started, think about a prompt that would be related to the following topic (use it as an inspiration):
```
{inspiration}
```

Rules:
- You can think first before writing the prompt.
- The prompt should be surrounded by ``` (triple backticks).
- Don't explicitly request to use the tool.
"""
        prompt = self.call_llm(prompt_prompt, extract_triple_backtick=True)

        generation = ToolCallGeneration(
            prompt=ChatPromptValue(messages=[HumanMessage(content=prompt)]),
        )
        generation.tool_name = tool["tool"]["name"]
        generation.tool_description = tool["tool"]["description"]
        generation.tool_parameters = tool["tool"]["parameters"]

        return generation

    def _generate_response(
        self, generation: ToolCallGeneration
    ) -> Union[str, BaseMessage, ToolCallGeneration]:
        tool_call_prompt = f"""
You are an AI and try to answer the following prompt:
```{generation.prompt.messages[0].content}```

You have to use the following tool:
- tool name: {generation.tool_name}
- tool description: {generation.tool_description}

Rules:
- Write the tool call arguments as a JSON object surrounded by ``` (triple backticks).
- Your response should have the following format {{"type":"tool_call", "tool":"{generation.tool_name}", "arguments":<ENTER THE ARGUMENTS HERE>"}}
"""
        arguments_str = self.call_llm(tool_call_prompt, extract_triple_backtick=True)

        response = f"""{arguments_str}"""
        generation.json_schema = {
            "type": "object",
            "properties": {
                "type": {"const": "tool_call"},
                "tool": {"const": generation.tool_name},
                "arguments": generation.tool_parameters,
            },
        }
        generation.response = AIMessage(content=response)
        return generation

    def _fix_response(
        self, generation: ToolCallGeneration, verification: Verification
    ) -> Optional[Union[ToolCallGeneration, str]]:
        return super()._fix_response(generation, verification)


if __name__ == "__main__":
    generator = ToolCallGenerator()
    generation = generator.generate_training_data()
    print(generation)
    generation = generation[0]
    print("TOOL_NAME:", generation.tool_name)
    print("TOOL_DESCRIPTION:", generation.tool_description)
    print("TOOL_PARAMETERS:", generation.tool_parameters)
    print("PROMPT:", generation.prompt.messages[0].content)
    print("GENERATED_TOOL_CALL:", generation.response.content)
    pass
