# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
from ast import Dict
from os import error
from typing import Any, Dict as TDict, List, Optional, Tuple

from synth_gen.execution.containerized_execution import PersistentContainer

default_timeout = 60


def run_cpp(code: str, timeout: int = default_timeout) -> Tuple[str, str, int]:
    """
    Compiles and runs the provided C++ code.

    Args:
        code (str): The C++ source code to compile and run.
        timeout (int, optional): The maximum time in seconds to allow the process to run. Defaults to `default_timeout`.

    Returns:
        Tuple[str, str, int]: A tuple containing the standard output, standard error, and the exit code of the process.
    """
    docker_file_content = """
FROM ubuntu:latest
RUN apt-get update
RUN apt-get install -y build-essential g++
"""
    with tempfile.TemporaryDirectory() as code_folder:
        with open(f"{code_folder}/user_source_code.cpp", "w") as f:
            f.write(code)
            f.flush()
            env = {}
            compile_command = f"g++ -o {code_folder}/user_executable {code_folder}/user_source_code.cpp"
            run_cpp_command = f"{code_folder}/user_executable"
            container = PersistentContainer(
                volumes={
                    code_folder: code_folder,
                },
                docker_file_content=docker_file_content,
            )
            try:
                compile_stdout, compile_stderr, compile_return_code = (
                    container.run_command(
                        command=compile_command,
                        timeout=timeout,
                        env=env,
                        cwd=code_folder,
                    )
                )
                if compile_return_code > 0:
                    return compile_stdout, compile_stderr, compile_return_code
                run_cpp_stdout, run_cpp_stderr, run_cpp_return_code = (
                    container.run_command(
                        command=run_cpp_command,
                        timeout=timeout,
                        env=env,
                        cwd=code_folder,
                    )
                )
                return run_cpp_stdout, run_cpp_stderr, run_cpp_return_code
            finally:
                container.clean_up()


def run_cpp_google_test(
    test_code: str, code: Optional[str] = None, timeout: int = default_timeout
) -> Tuple[str, str, int]:

    docker_file_content = """
FROM ubuntu:latest
RUN apt-get update
RUN apt-get install -y build-essential g++
"""
    pass
    # with tempfile.TemporaryDirectory() as code_folder:
    #     with open(f"{code_folder}/user_source_code.cpp", "w") as f:
    #         f.write(code)
    #         f.flush()
    #         env = {}
    #         compile_command = f"g++ -o {code_folder}/user_executable {code_folder}/user_source_code.cpp"
    #         run_cpp_command = f"{code_folder}/user_executable"
    #         container = PersistentContainer(
    #             volumes={
    #                 code_folder: code_folder,
    #             },
    #             docker_file_content=docker_file_content,
    #         )
    #         try:
    #             compile_stdout, compile_stderr, compile_return_code = (
    #                 container.run_command(
    #                     command=compile_command,
    #                     timeout=timeout,
    #                     env=env,
    #                     cwd=code_folder,
    #                 )
    #             )
    #             if compile_return_code > 0:
    #                 return compile_stdout, compile_stderr, compile_return_code
    #             run_cpp_stdout, run_cpp_stderr, run_cpp_return_code = (
    #                 container.run_command(
    #                     command=run_cpp_command,
    #                     timeout=timeout,
    #                     env=env,
    #                     cwd=code_folder,
    #                 )
    #             )
    #             return run_cpp_stdout, run_cpp_stderr, run_cpp_return_code
    #         finally:
    #             container.clean_up()


def compile_cpp_code(code: str, timeout: int = default_timeout) -> Tuple[str, str, int]:
    pass


def run_cpp_with_stdio(code, inputs):
    pass


def run_cpp_code_test_with_stdio(
    code: str,
    inputs: List[str],
    outputs: List[str],
    timeout: int = default_timeout,
):
    assert len(inputs) == len(
        outputs), "Number of inputs and outputs must match"
    docker_file_content = """
FROM ubuntu:latest
RUN apt-get update
RUN apt-get install -y build-essential g++
"""
    report: Any = {"tests": [], "summary": {
        "total": 0, "passed": 0, "failed": 0}}
    with tempfile.TemporaryDirectory() as code_folder:
        with open(f"{code_folder}/user_source_code.cpp", "w") as f:
            f.write(code)
            f.flush()
            env = {}
            compile_command = f"g++ -o {code_folder}/user_executable {code_folder}/user_source_code.cpp"
            container = PersistentContainer(
                volumes={
                    code_folder: code_folder,
                },
                docker_file_content=docker_file_content,
            )
            try:
                compile_stdout, compile_stderr, compile_return_code = (
                    container.run_command(
                        command=compile_command,
                        timeout=timeout,
                        env=env,
                        cwd=code_folder,
                    )
                )
                report["compilation"] = {
                    "return_code": compile_return_code,
                    "compile_stdout": compile_stdout,
                    "compile_stderr": compile_stderr,
                }
                if compile_return_code > 0:
                    report["compile_success"] = False
                    error_message = compile_stdout
                    return report, False, error_message
                else:
                    for input, output in zip(inputs, outputs):
                        test: TDict[str, Any] = {
                            "input": input, "output": output}
                        report["tests"].append(test)
                        with open(f"{code_folder}/input.txt", "w") as f:
                            f.write(input)
                            f.flush()
                        run_cpp_command = f"sh -c 'cat input.txt | ./user_executable'"
                        run_cpp_stdout, run_cpp_stderr, run_cpp_return_code = (
                            container.run_command(
                                command=run_cpp_command,
                                timeout=timeout,
                                env=env,
                                cwd=code_folder,
                            )
                        )
                        report["summary"]["total"] += 1
                        test["return_code"] = run_cpp_return_code
                        test["stdout"] = run_cpp_stdout
                        test["stderr"] = run_cpp_stderr

                        if run_cpp_return_code > 0:
                            test["outcome"] = "failed"
                            report["summary"]["failed"] += 1
                            test["error_message"] = "Runtime Error\n\n" + \
                                run_cpp_stderr
                        elif run_cpp_stdout.strip() != output.strip():
                            report["summary"]["failed"] += 1
                            test["outcome"] = "failed"
                            test["error_message"] = (
                                f"Expected output: {output.strip()}\nActual output: {run_cpp_stdout.strip()}"
                            )
                        else:
                            report["summary"]["passed"] += 1
                            test["outcome"] = "passed"
                is_success = (
                    report["summary"]["failed"] == 0 and report["summary"]["total"] > 0
                )
                error_message = ""
                if report["summary"]["total"] == 0:
                    error_message = "No tests were run."
                    if report["compile_success"] is False:
                        error_message = report["compilation"]["compile_stderr"]
                else:
                    error_messages = []
                    for test in report["tests"]:
                        if test["outcome"] == "failed":
                            error_messages.append(test["error_message"])
                    error_message = "\n\n\n".join(error_messages)
                return report, is_success, error_message
            finally:
                container.clean_up()
