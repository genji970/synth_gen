# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, cast, Dict, List, Optional, Sequence, Tuple, Union

import langchain
from langchain.schema import AIMessage, BaseMessage, BaseMessage, HumanMessage
from langchain_core.prompts import prompt
from synth_gen.execution.code_execution import (
    execute_code,
    execute_test,
    get_persistent_container,
    get_persistent_jupyter_notebook,
)
from synth_gen.execution.containerized_execution import PersistentContainer
from synth_gen.execution.python_execution import PersistentPythonContainer
from synth_gen.execution.shell_execution import PersistentJupyterContainer

from synth_gen.generation.generation import DataGeneration, PromptAnswerGeneration
from synth_gen.generation.generator import (
    VerifiedGenerator,
    VerifiedProgrammingGeneration,
)

from synth_gen.generation.LLM import call_llm, get_default_llm, LLM

from synth_gen.generation.LLM_based_parsing import (
    extract_programming_function_signature,
)
from synth_gen.generation.parsing import (
    extract_python_function_signature,
    extract_source_code,
)
from synth_gen.generation.programming import (
    build_prompt_for_defensive_programming_test,
    generate_programming_problem_from_seed,
    generate_programming_solution,
    generate_programming_test,
)
from synth_gen.generation.seed_source import SeedSource, SmallPythonSeedSource
from synth_gen.schema import ChatPromptValue

from synth_gen.verification.programming_verifiers import (
    ProgrammingExecutionVerifier,
    ProgrammingGeneratedTestVerifier,
)
from synth_gen.verification.python_verifiers import (
    default_python_verifiers,
    MypyVerifier,
    NonEmptyVerifier,
    PythonGeneratedTestCoverageVerifier,
    PythonGeneratedTestVerifier,
    PythonSyntaxVerifier,
)
from synth_gen.verification.schema import VerifiedGeneration, Verifier


class ProgramminPiece(Enum):
    code = "code"
    test_code = "test_code"
    problem = "problem"
    solution = "solution"


class ProgrammingProblemSynthDataGenerator(VerifiedGenerator):
    def __init__(self, language="python", use_stdio_for_input_output=False):
        verifiers = [NonEmptyVerifier()]
        if language == "python":
            verifiers += [
                PythonSyntaxVerifier(),
                MypyVerifier(),
                PythonGeneratedTestVerifier(),
                PythonGeneratedTestCoverageVerifier(),
            ]
        super().__init__(
            verifiers=[],
        )
        self.programming_language = language
        self.use_stdio_for_input_output = use_stdio_for_input_output

    def build_prompt(self, problem) -> VerifiedProgrammingGeneration:
        prompt = f"""
Solve the following programming problem in {self.programming_language}:

```
{problem}
```

Write a doc string.
Write comments before each line to explain your thought process.
""".strip()
        if self.use_stdio_for_input_output:
            prompt += f"""
Use the standard input and output for input and output.
"""
        data = {
            "problem": problem,
            "language": self.programming_language,
            "use_stdio_for_input_output": self.use_stdio_for_input_output,
        }
        return prompt, data


class ProgrammingVerifiedTestGenerator(VerifiedGenerator):
    def __init__(self, programming_language):
        self.programming_language = programming_language
        verifiers = [
            NonEmptyVerifier(),
            ProgrammingGeneratedTestVerifier(
                programming_language=self.programming_language
            ),
        ]
        # verifiers = [PythonGeneratedTestVerifier()]
        super().__init__(
            verifiers=verifiers,
        )

    def build_prompt(
        self, problem: str, solution: str
    ) -> Tuple[ChatPromptValue, Dict[str, Any]]:
        prompt_str = f"""
Here is a programming problem that we need to solve in python.
```
{problem}
```

Here is the solution to the problem.
```{self.programming_language}
{solution}
```

Now, write a test for the function in the solution. The test should be in the format of a pytest test case.
""".strip()
        prompt = ChatPromptValue(
            messages=[
                HumanMessage(
                    content=prompt_str,
                )
            ]
        )
        data = {
            "problem": problem,
            "solution": solution,
        }
        return prompt, data

    def generate_response(
        self, prompt: ChatPromptValue, data: Optional[Dict[str, Any]]
    ) -> Tuple[
        Union[str, List[str], BaseMessage,
              List[BaseMessage]], Optional[Dict[str, Any]]
    ]:
        if data is None or "problem" not in data or "solution" not in data:
            raise ValueError(
                "Data is required, with problem and solution fields")
        problem = data["problem"]
        solution = data["solution"]
        signature = extract_programming_function_signature(
            solution, programming_language=self.programming_language, llm=self.llm
        )
        programming_test = generate_programming_test(
            signature=signature,
            problem=problem,
            llm=self.llm,
            programming_language=self.programming_language,
        )
        return str(programming_test), None
