from copy import deepcopy

import pytest
from langchain_core.callbacks.manager import H
from synth_gen.generation.json_generator import JsonSchemaVerifier
from synth_gen.generators import DataGeneration
from synth_gen.schema import AIMessage, ChatPromptValue, HumanMessage
from synth_gen.verification.schema import VerifiedGeneration


def test_json_schema_verifier():
    verifier = JsonSchemaVerifier()
    generation = VerifiedGeneration()
    generation.response = HumanMessage(content="""{"a":1}""")
    generation.json_schema = """{"type": "object", "properties": {"a": {"type": "number"}}, "required": ["a"]}"""
    verification = verifier.check(generation)
    assert (
        verification.is_verified
    ), "JsonSchemaVerifier failed on a correct json, and json schema pair."

    incorrect_generation = deepcopy(generation)
    incorrect_generation.response = HumanMessage(content="""{"b":1}""")
    verification = verifier.check(incorrect_generation)
    assert (
        verification.is_verified is False
    ), "JsonSchemaVerifier failed on a incorrect json, and json schema pair."

    incorrect_generation_2 = deepcopy(generation)
    incorrect_generation_2.json_schema = """sdfdsfds"""
    verification = verifier.check(incorrect_generation_2)
    assert (
        verification.is_verified is False
    ), "JsonSchemaVerifier failed on a incorrect json, and json schema pair."
    assert (
        verification.is_unfixable
    ), "When the json schema is not a valid json, it is unfixable."
    pass


if __name__ == "__main__":
    test_json_schema_verifier()
