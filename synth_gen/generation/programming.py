# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Tuple, Union

from synth_gen.generation.LLM import call_llm, default_llm
from synth_gen.generation.parsing import (
    extract_multiple_source_codes,
    extract_source_code,
)
from synth_gen.generation.schema import LLM

from synth_gen.schema import BaseMessage, ChatPromptValue


def generate_programming_problem_from_seed(
    seed: str, llm=default_llm, programming_language="python", verbose=False
) -> str:
    """
    Generates a programming problem based on an seed snippet.

    Args:
        seed (str): The seed snippet to draw from.
        llm: The language model to use for generation.
        programming_language (str): The target programming language for the problem.
        verbose (bool): If True, prints the prompt and response for debugging.

    Returns:
        str: A generated programming problem.
    """
    # Create a prompt for the language model to generate a problem
    prompt = f"""
Read the following snippet:
{seed}
Please draw seed from the previous random snippet to create a high-quality and challenging programming problem for {programming_language}.
Rules:
- NEVER mention the "snippet".
- Do not write the solution.
- Do not specify constraints or example inputs/outputs.
- The seed is just that, an seed. You are free to deviate from it.
- The solution to the problem should be contained within a single function, not an entire program.
- The problem should be self-contained.
- Feel free to brainstorm first and then write your proposed problem.
- The proposed problem should be surrounded by triple backticks. Nothing else should be surrounded by triple backticks.

""".strip()
    if verbose:
        print(f"Prompt: {prompt}")
    # Call the language model to generate a response based on the prompt
    response = call_llm(prompt, extract_triple_backtick=True)
    if verbose:
        print(f"Response: {response}")
    return str(response)


def generate_programming_solution(
    problem: str, llm=default_llm, programming_language="python", verbose=False
) -> str:
    prompt = f"""
Solve the following programming problem in {programming_language}:

```
{problem}
```

Write a doc string.
Write comments before each line to explain your thought process.
""".strip()
    if verbose:
        print(f"Prompt: {prompt}")
    response = call_llm(prompt, llm=llm, use_cache=False)
    if verbose:
        print(f"Response: {response}")
    return str(response)


def fix_programming_solution(
    problem: str,
    solution: str,
    error_message: str,
    llm=default_llm,
    programming_language="python",
    verbose=False,
):
    """
    Fixes a programming solution based on a problem and an error message.

    Args:
        problem (str): The problem statement that the solution is trying to solve.
        solution (str): The current solution code that needs fixing.
        error_message (str): The error message encountered when running the solution.
        llm: The language model to use for generating the fix.
        programming_language (str): The target programming language for the solution.
        verbose (bool): If True, prints the prompt and response for debugging.

    Returns:
        str: A fixed version of the solution code.
    """

    prompt = f"""
I am trying to solve the following problem in {programming_language}:
```
{problem}
```

I made this code:
```{programming_language}
{solution}
```

But this error happens:
```
{str(error_message)}
```

Fix the code.

Rules:
- Do not write unit tests.
""".strip()
    if verbose:
        print(f"Prompt: {prompt}")
    response = call_llm(prompt, extract_source_code=True, llm=llm)
    if verbose:
        print(f"Response: {response}")
    return response


def generate_programming_test(
    problem,
    signature,
    llm=default_llm,
    programming_language="python",
    extract_only_code=True,
) -> str:
    """
    Generates a programming test based on a problem and function signature.

    Args:
        problem (str): The problem statement to generate tests for.
        signature (str): The function signature to be tested.
        llm: The language model to use for generation.
        programming_language (str): The target programming language for the test.

    Returns:
        str: A generated programming test.
    """

    prompt = f"""
Consider the following problem in {programming_language}:

```
{problem}
```

The signature of the function you are testing is: {signature}

Plan:
- Frist, rewrite the problem by adding a lot of details, including all relevant definitions, principles, and hypothesis.
- Second, write a list of all expectations and principles about the solution function that are explicitly described in the problem or the extended problem.
- Finally, write in a single code block a programming test. Within comments, write the reference to the constraint that the test is testing.

Rules:
- Do not write the solution.
- Do not write the problem.
- Do not test the error cases.
- Do not re-implement the solution anywhere in the test.
- If the constraint is about a specific input, write a test case by calling the function with that input.
- If the constraint does not mention a specific input, write a test case by calling the function with a random input.
- Avoid calling the function on fixed inputs as much as possible.
- Test the function on random inputs as much as possible.
- Try to test the principles rather than using specific values.
- Do not test for type errors on inputs.
- Write messages for each test case.

""".strip()
    if programming_language == "python":
        prompt += "- Use pytest.\n"
    response = call_llm(prompt, llm=llm, use_cache=True,
                        extract_source_code=True)
    return str(response)


def fix_programming_test_or_solution(
    test: str,
    solution: str,
    problem: str,
    error_message: str,
    llm=default_llm,
    programming_language="python",
    verbose=False,
) -> Tuple[bool, Optional[str]]:
    """
    Fix a programming test based on a problem, solution, and error message.

    Args:
        test (str): The test code to check.
        solution (str): The solution code that the test is trying to test.
        problem (str): The problem statement that the test is trying to test.
        error_message (str): The error message encountered when running the test.
        llm: The language model to use for generation.
        programming_language (str): The target programming language for the test.
        verbose (bool): If True, prints the prompt and response for debugging.

    Returns:
        Tuple[bool, Optional[str]]: A tuple containing a boolean indicating if the test has been fixed and the new test code if applicable.



    """
    prompt = f"""
We are trying to solve the following the following problem in {programming_language}:
```
{problem}
```

Here is our current solution:
```{programming_language}
{solution}
```

We are testing the solution with the following test:
```{programming_language}
{test}
```

Unfortunately, the test fails with the following error:
```
{error_message}
```

First, express your thoughts.
Then, try to evaluate if the problem comes from the test or the solution.
Then, write verbatim TEST_IS_WRONG or SOLUTION_IS_WRONG.
If the test fails because of unsupported edge cases like running with unexpected inputs or unexpected input types, write TEST_ABOUT_UNEXPECTED_INPUT, and remove the tests about these edge cases.

Finally, if the test is wrong, fix the tests.
The fixed test code should contain all the original tests if possible.
Surround the new test with triple backticks.

Rules:
- Do not write the solution.
- If you ttebirtdeetkhekicjdvcvbbkejetlkv the test, write it entirely within one triple backtick block.
- Do not test for type errors on inputs.


"""
    response = str(call_llm(prompt, llm=llm, use_cache=True))
    if "TEST_IS_WRONG" in response or "TEST_ABOUT_UNEXPECTED_INPUT" in response:
        sources = extract_multiple_source_codes(response)
        new_test_source = sources[-1]
        return True, new_test_source
    if True or "SOLUTION_IS_WRONG" in response:
        return False, None


def fix_programming_test(
    test: str,
    solution: str,
    problem: str,
    error_message: str,
    llm=default_llm,
    programming_language="python",
    verbose=False,
) -> Optional[str]:
    """
    Fix a programming test based on a problem, solution, and error message.

    Args:
        test (str): The test code to check.
        solution (str): The solution code that the test is trying to test.
        problem (str): The problem statement that the test is trying to test.
        error_message (str): The error message encountered when running the test.
        llm: The language model to use for generation.
        programming_language (str): The target programming language for the test.
        verbose (bool): If True, prints the prompt and response for debugging.

    """
    prompt = f"""
We are trying to solve the following the following problem in {programming_language}:
```
{problem}
```

Here is our current solution:
```{programming_language}
{solution}
```

We are testing the solution with the following test:
```{programming_language}
{test}
```

Unfortunately, the test fails with the following error:
```
{error_message}
```

First, express your thoughts.
Then, try to evaluate if the problem comes from the test or the solution.
Then, if you thinkg that the test is wrong, write TEST_IS_WRONG. Otherwise, if you think that the solution is wrong, write SOLUTION_IS_WRONG.
If the test fails because of unsupported edge cases like running with unexpected inputs or unexpected input types, write TEST_ABOUT_UNEXPECTED_INPUT, and remove the tests about these edge cases.

Finally, if the test is wrong, fix the tests.
The fixed test code should contain all the original tests if possible.
Surround the new test with triple backticks.


"""
    fixed_test = str(call_llm(prompt, llm=llm, use_cache=True))
    if (
        ("TEST_IS_WRONG" in fixed_test)
        or ("TEST_ABOUT_UNEXPECTED_INPUT" in fixed_test)
        or (not "SOLUTION_IS_WRONG" in fixed_test)
    ):
        return extract_source_code(str(fixed_test))
    else:
        return None


def generate_multi_strategy_solution(
    problem: ChatPromptValue,
    llm=default_llm,
    programming_language="python",
    verbose=False,
) -> List[str]:
    """
    Generates multiple solutions following different strategies based on a problem.

    Args:
        problem (str): The problem statement to generate a solution for.
        llm: The language model to use for generation.
        programming_language (str): The target programming language for the solution.
        verbose (bool): If True, prints the prompt and response for debugging.

    Returns:
        BaseMessage
    """

    solution = generate_programming_solution(problem=problem, llm=llm)

    prompt = f"""
I am trying to solve the following problem in {programming_language}:
```
{problem}
```

List several strategies to solve this problem. For each strategy, write a few lines of description.

Rules:
- Do not write the solution.
- Write the list in markdown format. Each strategy should start with a "- " an be written in a single line.

"""
    strategies = call_llm(prompt, llm=llm, extract_list=True, use_cache=False)
    for strategy in strategies:
        prompt = f"""

Solve the following problem in {programming_language}:
```
{problem}
```

But you must follow the following strategy:
```{strategy}```

Write a doc string.
Write comments before each line to explain your thought process.
"""
        solution = call_llm(prompt, llm=llm, use_cache=False)


# def generate_inline_programming_test(
#     problem: str,
#     solution: str,
#     llm=default_llm,
#     programming_language="python",
# ) -> str:
#     prompt = f"""
# I am trying to solve the following problem in {programming_language}:
# ```
# {problem}
# ```

# Here is my solution:
# ```{programming_language}
# {solution}
# ```

# Goal:
# Modify this solution to add inline tests.
# The original solution should be preserved.
# You can only add assert statements within the solution.
# Also, add some code at the end to run the solution with several interesting inputs.

# Plan:
# - Frist, rewrite the problem by adding a lot of details, including all relevant definitions, principles, and hypothesis.
# - Second, write a list of all expectations and principles about the solution function that are explicitly described in the problem or the extended problem.
# - Finally, write in a single code block the original solution with inline tests. Within comments, write the reference to the constraint that the test is testing.

# Rules:
# - Avoid creating infinite loops. So don't call the solution function in the inline tests.
# """.strip()

#     response = call_llm(prompt, llm=llm, extract_source_code=True, use_cache=False)

#     return response


def build_prompt_for_defensive_programming_test(
    problem: str,
    solution: str,
    programming_language="python",
) -> Tuple[str, Dict[str, Any]]:
    prompt = f"""
I am trying to solve the following problem in {programming_language}:
```
{problem}
```

Here is my solution:
```{programming_language}
{solution}
```

Goal:
Insert defensive programming assertions into the following code snippet to ensure its correctness and robustness. The assertions should check for invalid input, potential null pointer dereferences, out-of-range values, and any other conditions that could lead to runtime errors or unexpected behavior.
In addition to checking function inputs and outputs, please also add assertions to verify intermediate values and calculations within the code. This should include checks for variables being assigned expected values, loop counters and indices being within valid ranges, and any other internal state that could impact the code's correctness.
Also, add some code at the end to run the solution with several interesting inputs.

Plan:
- Frist, rewrite the problem by adding a lot of details, including all relevant definitions, principles, and hypothesis.
- Second, write a list of all expectations and principles about the solution function that are explicitly described in the problem or the extended problem.
- Finally, write in a single code block the original solution with defensive programming assertions. Within comments, write the reference to the constraint that the test is testing.

Rules:
- Avoid creating infinite loops. So don't call the solution function in the inline tests.
""".strip()
    data = {
        "problem": problem,
        "solution": solution,
    }
    return prompt, data


def generate_defensive_programming_test(
    problem: str,
    solution: str,
    llm=default_llm,
    programming_language="python",
) -> str:
    prompt = build_prompt_for_defensive_programming_test(
        problem, solution, programming_language
    )
    response = call_llm(
        prompt, llm=llm, extract_source_code=True, use_cache=False)

    return str(response)
