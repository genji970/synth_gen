# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

from synth_gen.execution.code_execution import execute_code

from synth_gen.generation.LLM import default_llm, LLM
from synth_gen.generation.parsing import (
    extract_python_function_signature,
    extract_source_code,
)
from synth_gen.generation.programming import (
    fix_programming_solution,
    fix_programming_test,
    fix_programming_test_or_solution,
    generate_defensive_programming_test,
    generate_programming_test,
)
from synth_gen.verification.schema import VerifiedGeneration, Verifier


class ProgrammingVerifier(Verifier):
    def __init__(self, programming_language):
        self.programming_language = programming_language

    def fix(
        self, generation: VerifiedGeneration, error_message: str, llm: LLM = default_llm
    ):
        fixed_solution = fix_programming_solution(
            str(generation.prompt.messages[-1].content),
            str(generation.response.content),
            error_message,
            llm=llm,
            programming_language=self.programming_language,
            verbose=False,
        )
        return fixed_solution


class ProgrammingExecutionVerifier(ProgrammingVerifier):
    def __init__(self, programming_language):
        self.programming_language = programming_language

    def check(self, generation: VerifiedGeneration) -> Tuple[bool, Optional[str]]:
        if generation.response is None:
            return False, "No response provided."
        code = extract_source_code(str(generation.response.content))
        stdout, stderr, return_code = execute_code(
            code, self.programming_language)
        if return_code != 0:
            return (
                False,
                f"The code failed to execute:\nreturn code: {return_code}\nstdout: {stdout}\nstderr: {stderr}",
            )
        return True, None


class ProgrammingInputOutputVerifier(ProgrammingVerifier):
    def __init__(self, programming_language):
        self.programming_language = programming_language

    def check(self, generation: VerifiedGeneration) -> Tuple[bool, Optional[str]]:
        if generation.response is None:
            return False, "No response provided."
        code = extract_source_code(str(generation.response.content))
        stdout, stderr, return_code = execute_code(
            code, self.programming_language)
        if return_code != 0:
            pass  # TODO


class ProgrammingGeneratedTestVerifier(ProgrammingVerifier):

    def check(self, generation: VerifiedGeneration):
        test = str(self.message_to_code(generation.response))
        full_code = f"{test}"
        stdout, stderr, return_code = execute_code(
            full_code, programming_language=self.programming_language
        )
        is_success = return_code == 0
        error_message = f"stdout:\n{stdout}\nstderr:\n{stderr}"
        return is_success, error_message


class DefensiveProgrammingTestVerifier(ProgrammingVerifier):
    def check(self, generation: VerifiedGeneration) -> Tuple[bool, Optional[str]]:
        if generation.response is None:
            return False, "No response provided."
        inline_test = generate_defensive_programming_test(
            problem=generation.data["problem"],
            solution=generation.data["solution"],
            llm=default_llm,
        )
        stdout, stderr, return_code = execute_code(
            inline_test, self.programming_language
        )
        if return_code != 0:
            error_message = f"stdout:\n{stdout}\nstderr:\n{stderr}"
            return (False, error_message)
        return True, None
