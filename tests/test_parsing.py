# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from synth_gen.generation.parsing import (
    extract_multiple_source_codes,
    extract_python_function_signature,
    extract_source_code,
    extract_titled_list,
)


def test_extract_source_code():
    # Test with a valid input
    input_code = (
        "aaa\n```python\ndef example_function():\n    return 'Hello, World!'\n```"
    )
    expected_output = "def example_function():\n    return 'Hello, World!'"
    print(extract_source_code(input_code))
    print(expected_output)
    code = extract_source_code(input_code)
    assert code is not None
    assert code.strip() == expected_output.strip()


def test_extract_multiple_source_codes():
    # Test with multiple code blocks
    input_code = "aaa\n```python\ndef example_function():\n    return 'Hello, World!'\n```\nbbb\n```python\ndef example_function2():\n    return 'Hello, World!'\n```"
    expected_output = [
        "def example_function():\n    return 'Hello, World!'",
        "def example_function2():\n    return 'Hello, World!'",
    ]
    assert extract_multiple_source_codes(input_code) == expected_output


def test_extract_titled_list():
    raw = """
[THOUGHT 1]
thought 1


  [THOUGHT 2]
thought 2


**[THOUGHT 3]**
thought 3
"""

    titled_list = extract_titled_list(raw=raw, tag="THOUGHT")
    assert len(titled_list) == 3


def test_extract_python_function_signature():
    raw = """
# comment
def test_function(self, arg1: int, arg2: str) -> None: # comment
    print("Hello, World!")
# comment
"""
    signature = extract_python_function_signature(raw=raw)
    assert signature == "def test_function(self, arg1: int, arg2: str) -> None"

    async_raw = """
# comment
async def test_function(self, arg1: int, arg2: str) -> None: # comment
    print("Hello, World!")
# comment
"""
    signature = extract_python_function_signature(raw=async_raw)
    assert signature == "async def test_function(self, arg1: int, arg2: str) -> None"

    no_args_raw = """
async def test_function():
    print("Hello, World!")
"""
    signature = extract_python_function_signature(raw=no_args_raw)
    assert signature == "async def test_function()"


if __name__ == "__main__":
    pytest.main(["-q", __file__])
