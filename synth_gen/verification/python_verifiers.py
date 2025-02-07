# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import ast
import os
import re
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

from synth_gen.execution.execution_utils import run_command_sync
from synth_gen.execution.python_execution import (
    execute_python,
    pytest_report_to_error_message,
    run_pytest,
    run_test_coverage,
)
from synth_gen.generation.LLM import default_llm, LLM
from synth_gen.generation.LLM_based_parsing import (
    extract_programming_function_signature,
)
from synth_gen.generation.parsing import (
    extract_python_function_signature,
    extract_source_code,
)
from synth_gen.generation.programming import (
    fix_programming_solution,
    fix_programming_test,
    fix_programming_test_or_solution,
    generate_programming_test,
)

from synth_gen.schema import BaseMessage, ChatPromptValue, LLM

from synth_gen.verification.programming_verifiers import ProgrammingVerifier

from synth_gen.verification.schema import (
    Verification,
    VerifiedGeneration,
    VerifiedProgrammingGeneration,
    Verifier,
)


class PythonVerifier(ProgrammingVerifier):
    def __init__(self):
        super().__init__(programming_language="python")


class NonEmptyVerifier(Verifier):
    def check(self, generation: VerifiedGeneration):
        if (
            type(generation.response) is None
            or type(generation.response.content) is None
        ):
            return False, "Response is None."
        if type(generation.response.content) is not str:
            return (
                False,
                f"Response is not a string. It is a {type(generation.response)}",
            )
        if len(generation.response.content) == 0:
            return False, "Response is empty."
        return True, None


class NonEmptyCodeVerifier(ProgrammingVerifier):
    def __init__(self):
        super().__init__(programming_language="code")

    def check(self, generation: VerifiedGeneration):
        if generation.response is None:
            return False, "No response provided."
        code = self.message_to_code(generation.response)
        if type(code) is None:
            return False, "The response contains no code."
        if type(code) is not str:
            return False, f"Response's code is not a string. It is a {type(code)}."
        if len(code) == 0:
            return False, "The code in the response is empty."
        return True, None


class PythonSyntaxVerifier(PythonVerifier):
    def check(self, generation: VerifiedGeneration):
        if generation.response is None:
            return False, "No response provided."
        code = str(self.message_to_code(generation.response))
        try:
            ast.parse(code)
        except SyntaxError as err:
            return (
                False,
                f"Syntax Error({err.msg}, {err.lineno}, {err.offset}, {err.text})",
            )
        return True, None


class MypyVerifier(PythonVerifier):
    def check(self, generation: VerifiedGeneration):
        if generation.response is None:
            return False, "No response provided."
        temp_dir = None
        code = str(self.message_to_code(generation.response))
        try:
            folder_path = temp_dir = tempfile.mkdtemp()
            solution_file_path = os.path.join(temp_dir, "solution.py")
            with open(solution_file_path, "w") as solution_file:
                solution_file.write(code)
            config_file_path = os.path.join(folder_path, "mypy.ini")
            with open(config_file_path, "w") as config_file:
                config_file.write(
                    "[mypy]\nignore_missing_imports = True\ncheck_untyped_defs = True"
                )
            stdout, stderr, return_code = run_command_sync(
                ["mypy", "solution.py"], cwd=folder_path
            )
            if return_code > 0:
                return False, stdout
            return True, None
        finally:
            if temp_dir is not None:
                shutil.rmtree(temp_dir)


class RuffVerifier(PythonVerifier):
    def check(self, generation: VerifiedGeneration):
        if generation.response is None:
            return False, "No response provided."
        temp_dir = None
        code = str(self.message_to_code(generation.response))
        try:
            folder_path = temp_dir = tempfile.mkdtemp()
            solution_file_path = os.path.join(temp_dir, "solution.py")
            with open(solution_file_path, "w") as solution_file:
                solution_file.write(code)
            stdout, stderr, return_code = run_command_sync(
                ["ruff", "check", "solution.py"], cwd=folder_path
            )
            if return_code > 0:
                return False, stdout
            return True, None
        finally:
            if temp_dir is not None:
                shutil.rmtree(temp_dir)


class PythonTestBasedVerifier(PythonVerifier):
    def __init__(
        self,
    ):
        super().__init__()
        self.response_contains_code = True
        self.response_contains_test = False

    def check(self, generation: VerifiedProgrammingGeneration, llm: LLM = default_llm):
        if generation.response is None:
            return False, "No response provided."
        if self.response_contains_code:
            code = str(self.message_to_code(generation.response))
        else:
            code = generation.code
        if self.response_contains_test:
            test_code = str(self.message_to_code(generation.response))
        else:
            if not hasattr(generation, "test_code"):
                generation.test_code = None
            test_code = generation.test_code
        if not hasattr(generation, "signature") or generation.signature is None:
            generation.signature = extract_programming_function_signature(
                code, programming_language=self.programming_language
            )
        if generation.test_code is None and not self.response_contains_test:
            generation.test_code = generate_programming_test(
                generation.problem,
                generation.signature,
                llm=llm,
                programming_language=self.programming_language,
            )
            test_code = generation.test_code
        try:
            error_message = ""
            for attempt in range(3):
                full_code = f"{code}\n\n\n{test_code}"

                print(full_code)
                report = run_pytest(full_code)
                print(report)
                is_success, error_message = pytest_report_to_error_message(
                    report)
                if is_success:
                    return True, None
                has_test_been_fixed, new_test = fix_programming_test_or_solution(
                    solution=code,
                    problem=self.message_to_string(generation.prompt),
                    test=str(test_code),
                    error_message=str(error_message),
                    llm=llm,
                )
                if has_test_been_fixed:
                    if self.response_contains_test:
                        verification = Verification(
                            verifier_name="PythonTestBasedVerifier",
                            generation=generation,
                        )
                        verification.is_unfixable = True
                        verification.is_verified = False
                        return verification
                    generation.test_code = new_test
                    continue
                if has_test_been_fixed is False:
                    return False, error_message
            return False, error_message
        except Exception as e:
            return False, str(e)


class PythonGeneratedTestVerifier(PythonTestBasedVerifier):

    def __init__(self):
        super().__init__()
        self.response_contains_code = False
        self.response_contains_test = True

    # def fix(
    #     self,
    #     generation: VerifiedGeneration,
    #     error_message: str,
    #     llm: LLM = default_llm,
    # ) -> Optional[str]:
    #     code = generation.data["solution"]
    #     test = str(self.message_to_code(generation.response))
    #     problem = generation.data["problem"]
    #     fixed_test = fix_programming_test(
    #         test=test,
    #         solution=code,
    #         problem=problem,
    #         error_message=error_message,
    #         llm=llm,
    #     )
    #     return fixed_test


class PythonGeneratedTestCoverageVerifier(PythonVerifier):
    def __init__(self):
        super().__init__()
        self.response_contains_code = False
        self.response_contains_test = True

    def check(
        self,
        generation: VerifiedProgrammingGeneration,
        llm: LLM = default_llm,
    ) -> Tuple[bool, Optional[str]]:
        if self.response_contains_code:
            solution = str(self.message_to_code(generation.response))
        else:
            solution = generation.code
        if generation.response is None:
            return False, "No response provided."
        if self.response_contains_test:
            test = str(self.message_to_code(generation.response))
        else:
            test = generation.test_code
        full_code = f"{solution}\n\n\n{test}"
        report = run_test_coverage(full_code, timeout=60)
        if report is None or len(report) == 0 or report[0] is None:
            return False, "No coverage report was found."
        print(report)
        missing_lines_count = report[0]["totals"]["missing_lines"]
        if missing_lines_count == 0:
            return True, None
        return False, f"The test does not cover all lines. {missing_lines_count}"

    # def fix(
    #     self,
    #     generation: VerifiedGeneration,
    #     error_message: str,
    #     llm: LLM = default_llm,
    # ) -> str:
    #     code = generation.data["solution"]
    #     test = str(self.message_to_code(generation.response))
    #     problem = data["problem"]
    #     fixed_test = fix_programming_test(
    #         test=test,
    #         solution=code,
    #         problem=problem,
    #         error_message=error_message,
    #         llm=llm,
    #     )
    #     return fixed_test


class PythonSpeedUpVerifier(PythonVerifier):
    def check(
        self,
        generation: VerifiedGeneration,
        llm: LLM = default_llm,
    ) -> Tuple[bool, Optional[str]]:
        slow_solution = generation.prompt.messages[0].additional_kwargs["slow_solution"]
        fast_solution = str(self.message_to_code(generation.response))

        full_code = f"""
import time

slow_code_t0 = time.time()
{slow_solution}
slow_code_t1 = time.time()

fast_code_t0 = time.time()
{fast_solution}
fast_code_t1 = time.time()

"""

        report = execute_python(full_code, timeout=60)
        print(report)
        missing_lines_count = report[0]["totals"]["missing_lines"]
        if missing_lines_count == 0:
            return True, None
        return False, f"The test does not cover all lines. {missing_lines_count}"

    def fix(
        self,
        generation: VerifiedGeneration,
        error_message: str,
        llm: LLM = default_llm,
    ) -> str:
        code = generation.prompt.messages[0].additional_kwargs["solution"]
        test = str(self.message_to_code(generation.response))
        problem = generation.prompt.messages[0].additional_kwargs["problem"]
        fixed_test = fix_programming_test(
            test=test,
            solution=code,
            problem=problem,
            error_message=error_message,
            llm=llm,
        )
        return fixed_test


class PythonDefensiveProgrammingVerifier(PythonVerifier):
    def check(self, generation: VerifiedGeneration) -> Tuple[bool, Optional[str]]:
        # TODO
        return super().check(generation)


default_python_verifiers = [
    NonEmptyCodeVerifier,
    PythonSyntaxVerifier,
    MypyVerifier,
    PythonTestBasedVerifier,
]
