# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Tuple

from diskcache.recipes import math
from synth_gen.execution.python_execution import (
    execute_python,
    pytest_report_to_error_message,
    run_pytest,
)
from synth_gen.generation.LLM import call_llm, default_llm, LLM

from synth_gen.generation.LLM_based_parsing import extract_math_expressions

from synth_gen.verification.schema import VerifiedGeneration, Verifier


class CalculationVerifier(Verifier):
    def check(
        self, generation: VerifiedGeneration, llm: LLM = default_llm
    ) -> Tuple[bool, Optional[str]]:
        response_str = self.message_to_string(generation.response)
        math_test = extract_math_expressions(response_str)
        print(math_test)
        if math_test is None:
            return True, "No math expressions found."
        report, stdout, stderr, return_code = run_pytest(math_test)
        print(report)
        is_success, error_message = pytest_report_to_error_message(report)
        return is_success, error_message
