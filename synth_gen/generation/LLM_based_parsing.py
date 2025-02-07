# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from subprocess import call
from typing import cast, List, Optional, Union

from synth_gen.generation.LLM import call_llm, default_llm
from synth_gen.generation.parsing import (
    extract_python_function_signature,
    extract_source_code,
)


def extract_programming_function_signature(
    code: str, programming_language, llm=default_llm
) -> Optional[str]:
    if programming_language == "python":
        return extract_python_function_signature(code)
    prompt = f"""
Analyze the following {programming_language} code.
And extract the signature of the following function.
Surround the signature by triple back ticks.

If you cannot extract the signature, return NO_SIGNATURE.

```{programming_language}
{code}
```
"""
    signature = call_llm(
        prompt, llm=llm, extract_triple_backtick=True, use_cache=True)

    if type(signature) is str and "NO_SIGNATURE" in signature:
        return None

    return str(signature)


def extract_math_expressions(text: str, llm=default_llm) -> Optional[str]:
    """
    Extracts mathematical expressions from a given text using a language model.

    Args:
        text (str): The input text from which to extract mathematical expressions.
        llm: The language model to use for extraction. Defaults to `default_llm`.

    Returns:
        Optional[str]: A string containing the extracted mathematical expressions and a test code,
        or None if no expressions can be extracted.
    """

    # Note: Llama fails to extract math expressions. Just keep this code in case, it works better in the future.
    # Note: We could use this library to teach llama how to do that better.
    prompt = f"""
Read this message:
```
{text}
```

From the above message:
1. Extract all mathematical expressions from it,including the result of the expression.
Use this format:
left_side: 1+1
right_side: 3
operator: =
2. Filter in only mathematical expressions that do not contain any variables. For example, do not extract expressions that contain variables like "x" or "y". Also, filter in only expressions that are present in the original message.
3. End your message with a code that tests wether the expressions are correct. The code should be in Python. The code should be executable and should not contain any syntax errors. Each assert statement should include an assertion message.
Use this format:

def test_math_expressions():
    left_side = 1+1
    right_side = 3
    assert left_side == right_side, "1+1 should be equal to 3"

4. If you cannot extract any mathematical expressions, return NO_EXPRESSIONS.

Rules:
- Do not correct the expressions. Only extract the expressions.
- Do not invent expressions. Only extract expressions that are already in the message.
- Do not compute the expressions. Only extract the expressions.
- Copy the results of the expressions exactly as they are in the message.
- Do not calculate the resulr of the expressions. Only extract the expressions.
"""
    expression_test = call_llm(prompt, llm=llm, parse_json=True)
    print(expression_test)
    if (
        expression_test is None
        or type(expression_test) is str
        and "NO_EXPRESSIONS" in expression_test
    ):
        return None
    expression_test = extract_source_code(str(expression_test))
    return expression_test
