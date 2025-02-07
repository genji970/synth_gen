# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, cast, Dict, List, Optional, Sequence, Tuple, Union

import langchain
from langchain.schema import AIMessage, BaseMessage, BaseMessage, HumanMessage
from langchain_core.prompts import prompt
from synth_gen import generation
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
from synth_gen.generation.generator import VerifiedGenerator

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
    fix_programming_solution,
    fix_programming_test,
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
from synth_gen.verification.schema import (
    Verification,
    VerifiedGeneration,
    VerifiedProgrammingGeneration,
    Verifier,
)


class PythonVerifiedSynthDataGenerator(
    VerifiedGenerator[VerifiedProgrammingGeneration]
):
    def __init__(self):
        super().__init__(
            seed_source=SmallPythonSeedSource(),
            verifiers=[verifier() for verifier in default_python_verifiers],
        )
        self.programming_language = "python"
        self.data_generation_cls = VerifiedProgrammingGeneration

    def _generate_prompt(self, seed: Optional[str] = None) -> str:
        if seed is None:
            seed = str(self.get_next_seed())
        problem = generate_programming_problem_from_seed(
            seed=seed, llm=self.llm, programming_language="python"
        )
        return problem

    def _generate_response(
        self, generation: VerifiedProgrammingGeneration
    ) -> VerifiedProgrammingGeneration:
        solution = generate_programming_solution(
            str(generation.prompt), llm=self.llm, programming_language="python"
        )
        generation.code = extract_source_code(solution)
        generation.response = AIMessage(content=solution)
        return generation

    def _fix_response(
        self, generation: VerifiedProgrammingGeneration, verification: Verification
    ) -> Optional[VerifiedProgrammingGeneration]:
        if verification.error_message is None:
            return None
        fixed_solution = fix_programming_solution(
            problem=str(generation.prompt.messages[-1].content),
            solution=str(generation.response.content),
            error_message=verification.error_message,
            llm=self.llm,
            verbose=self.verbose,
            programming_language=self.programming_language,
        )
        if fixed_solution is None:
            return None
        fixed_solution = str(fixed_solution)
        generation.code = extract_source_code(fixed_solution)
        generation.response = AIMessage(content=fixed_solution)
        return generation


class PythonVerifiedTestGenerator(VerifiedGenerator[VerifiedProgrammingGeneration]):
    def __init__(self):
        verifiers = [
            NonEmptyVerifier(),
            PythonSyntaxVerifier(),
            # MypyVerifier(),
            PythonGeneratedTestVerifier(),
            PythonGeneratedTestCoverageVerifier(),
        ]
        # verifiers = [PythonGeneratedTestVerifier()]
        super().__init__(
            verifiers=verifiers,
        )
        self.data_generation_cls = VerifiedProgrammingGeneration

    def build_prompt(
        self, *args, problem: str, solution: str, **kwargs
    ) -> VerifiedProgrammingGeneration:
        prompt_str = f"""
Here is a programming problem that we need to solve in python.
```
{problem}
```

Here is the solution to the problem.
```python
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
        generation = VerifiedProgrammingGeneration()
        generation.prompt = prompt
        generation.problem = problem
        generation.code = solution
        return generation

    def _generate_response(self, generation: VerifiedProgrammingGeneration) -> str:
        if generation.problem is None or generation.prompt is None:
            raise ValueError("Problem and prompt are required")
        if generation.signature is None:
            generation.signature = extract_python_function_signature(
                str(generation.prompt.messages[-1].content)
            )
        programming_test = generate_programming_test(
            signature=generation.signature,
            problem=generation.problem,
            llm=self.llm,
            programming_language="python",
        )
        return str(programming_test)

    def _fix_response(
        self, generation: VerifiedProgrammingGeneration, verification: Verification
    ) -> Optional[Union[VerifiedProgrammingGeneration, str]]:
        if (
            generation.problem is None
            or generation.prompt is None
            or generation.response is None
        ):
            raise ValueError("Problem, response and prompt are required")
        test = extract_source_code(str(generation.response.content))
        solution = generation.code
        if test is None:
            return None
        if solution is None:
            return None
        if verification.error_message is None:
            return None
        fixed_test = fix_programming_test(
            test=test,
            solution=solution,
            problem=generation.problem,
            error_message=verification.error_message,
            llm=self.llm,
            verbose=self.verbose,
        )
        return fixed_test
