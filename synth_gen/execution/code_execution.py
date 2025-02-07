# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, Tuple, Union

from langchain_core.load.dump import default
from synth_gen.execution.containerized_execution import PersistentContainer
from synth_gen.execution.cpp_execution import run_cpp, run_cpp_code_test_with_stdio
from synth_gen.execution.python_execution import (
    execute_python,
    run_pytest,
    run_test_coverage,
)
from synth_gen.execution.shell_execution import PersistentJupyterContainer

default_timeout = 60


def execute_code(code, programming_language, timeout=default_timeout):
    if programming_language == "python":
        return execute_python(code, timeout)
    elif programming_language == "cpp":
        return run_cpp(code, timeout)
    else:
        raise NotImplementedError(
            f"Running code for {programming_language} is not supported yet."
        )


def execute_test(code, test_code, programming_language, timeout=default_timeout):
    if programming_language == "python":
        full_code = f"""
{code}

{test_code}
"""
        return run_pytest(full_code, timeout=timeout)
    else:
        raise NotImplementedError(
            f"Running code for {programming_language} is not supported yet."
        )


def run_input_output_test(
    code, inputs, outputs, programming_language, timeout=default_timeout
):
    if programming_language == "cpp":
        return run_cpp_code_test_with_stdio(code, inputs, outputs, timeout=timeout)
    else:
        raise NotImplementedError(
            f"Running code for {programming_language} is not supported yet."
        )


def get_persistent_container(
    memory_limit="500m", volumes=None, image_name=None, docker_file_content=None
):
    return PersistentContainer(
        memory_limit=memory_limit,
        volumes=volumes,
        image_name=image_name,
        docker_file_content=docker_file_content,
    )


def get_persistent_jupyter_notebook(
    programming_language="python", memory_limit="500m", volumes=None
):
    return PersistentJupyterContainer(
        kernel_name=programming_language, memory_limit=memory_limit
    )
