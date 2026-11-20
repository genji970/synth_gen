# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import tempfile
import time
from re import error
from typing import Optional, Tuple

from pydantic import Json

from synth_gen.execution.containerized_execution import (
    PersistentContainer,
    run_command_in_container,
)

from synth_gen.execution.python_dependencies import (
    dependency_folder,
    install_dependencies,
)

script_folder = os.path.dirname(os.path.realpath(__file__))

default_timeout = 60


def execute_python(
    code: str, timeout: int = default_timeout, parse_json=False
) -> Tuple[str, str, int]:
    """
    Executes a given Python code string within a containerized environment.

    Args:
        code (str): The Python code to execute.
        timeout (int, optional): The maximum time in seconds to allow the code to run. Defaults to `default_timeout`.

    Returns:
        Tuple[str, str, int]: A tuple containing the standard output, standard error, and the return code of the execution.
    """

    with tempfile.TemporaryDirectory() as code_folder:
        with open(f"{code_folder}/user_source_code.py", "w") as f:
            f.write(code)
            f.flush()
            python_paths = install_dependencies(code_folder)
            env = {"PYTHONPATH": ":".join(python_paths)}
            return run_command_in_container(
                command=f"python3 {code_folder}/user_source_code.py",
                timeout=timeout,
                env=env,
                volumes={
                    dependency_folder: dependency_folder,
                    code_folder: code_folder,
                },
                cwd=code_folder,
                image_name="python:3.10-slim",
                parse_json=parse_json,
            )


def run_pytest(
    code: str, timeout: int = default_timeout
) -> Tuple[Optional[dict], str, str, int]:
    """
    Executes pytest on a given Python code string within a containerized environment and returns the json test report.

    Args:
        code (str): The Python code to test.
        timeout (int, optional): The maximum time in seconds to allow the tests to run. Defaults to `default_timeout`.

    Returns:
        Tuple[dict, str, str, int]: A tuple containing the JSON test report, standard output, standard error, and the return code of the execution.
    """

    docker_file_content = """
FROM python:3.10-slim

RUN apt update
RUN apt install -y python3 python3-pip
RUN pip3 install pytest pytest-json-report
"""
    with tempfile.TemporaryDirectory() as code_folder:
        with open(f"{code_folder}/user_source_code.py", "w") as f:
            f.write(code)
            f.flush()
            python_paths = install_dependencies(code_folder)
            env = {"PYTHONPATH": ":".join(python_paths)}
            command = f"pytest {code_folder}/user_source_code.py --json-report"
            container = PersistentContainer(
                volumes={
                    dependency_folder: dependency_folder,
                    code_folder: code_folder,
                },
                docker_file_content=docker_file_content,
            )
            try:
                stdout, stderr, return_code = container.run_command(
                    command=command,
                    timeout=timeout,
                    env=env,
                    cwd=code_folder,
                )
                raw_report, _, raw_report_return_code = container.run_command(
                    command=f"cat .report.json",
                    timeout=10,
                    cwd=code_folder,
                )
                if raw_report_return_code > 0:
                    return None, stdout, stderr, return_code
                report = json.loads(raw_report)
                return report, stdout, stderr, return_code
            finally:
                container.clean_up()


def pytest_report_to_error_message(report):
    if report[0] is None:
        return False, f"No pytest report was found. {report}"
    summary = report[0]["summary"]
    if ("failed" not in summary or summary["failed"] == 0) and summary["total"] > 0:
        return True, None
    error_message = ""
    if report[0]["summary"]["total"] == 0:
        error_message = "No tests were run."
    else:
        error_messages = []
        for test in report[0]["tests"]:
            if test["outcome"] == "failed":
                error_messages.append(test["call"]["longrepr"])
        error_message = "\n\n\n".join(error_messages)
    return False, error_message


class PersistentPythonContainer:
    """
    A class that manages a persistent Python container for executing Python code.

    Attributes:
        persistent_container (PersistentContainer): An instance of PersistentContainer to manage the container lifecycle.

    Methods:
        __init__(memory_limit=None):
            Initializes the PersistentPythonContainer with optional memory limits and starts a personal server inside the container.

        run_command(code: str, timeout: Optional[int] = None):
            Executes a given Python code string within the persistent container.
    """

    def __init__(self, memory_limit=None):
        app_folder = os.path.join(
            script_folder, "containers/persistent_python")
        volumes = {
            dependency_folder: dependency_folder,
            app_folder: "/app/",
        }
        self.persistent_container = PersistentContainer(
            volumes=volumes,
            memory_limit=memory_limit,
            docker_file_folder=os.path.join(
                script_folder, "containers/persistent_python/Dockerfile"
            ),
        )
        self.persistent_container.run_command("python3 -m venv venv")
        # self.persistent_container.run_command(
        #     "venv/bin/pip install -r requirements.txt"
        # )
        self.persistent_container.run_command(
            "python3 -m pip config set global.break-system-packages true"
        )
        server_command = "python3 /app/python_persistent_server.py"
        # server_command = "bash -c 'source venv/bin/activate && python3 /app/python_persistent_server.py'"
        self.persistent_container.run_command(
            server_command,
            dont_wait=True,
            timeout=10 * 3600,
            env={"PYTHONPATH": "/home/ubuntu/.local/lib/python3.12/site-packages"},
        )
        time.sleep(0.5)

    def run_command(self, code: str, timeout: Optional[int] = None):
        escaped_code = code.replace("\\", "\\\\").replace("'", "'\"'\"'")

        stdout, stderr, return_code = self.persistent_container.run_command(
            f"python3 /app/python_persistent_client.py '{escaped_code}'"
        )
        try:
            stdout_json = json.loads(stdout)
        except json.JSONDecodeError:
            return stdout, stderr, True
        result = stdout_json["result"]
        is_failure = stdout_json["is_failure"]
        error_message = stdout_json["error_message"]
        return result, error_message, is_failure

    def clean_up(self):
        self.persistent_container.clean_up()


def run_test_coverage(code: str, timeout: int = default_timeout):
    docker_file_content = """
FROM python:3.10-slim

RUN apt update
RUN apt install -y python3 python3-pip
RUN pip3 install pytest pytest-json-report coverage
"""
    with tempfile.TemporaryDirectory() as code_folder:
        with open(f"{code_folder}/user_source_code.py", "w") as f:
            f.write(code)
            f.flush()
            python_paths = install_dependencies(code_folder)
            env = {"PYTHONPATH": ":".join(python_paths)}
            command = f"coverage run --omit '*/dependencies/*' -m pytest {code_folder}/user_source_code.py"
            container = PersistentContainer(
                volumes={
                    dependency_folder: dependency_folder,
                    code_folder: code_folder,
                },
                docker_file_content=docker_file_content,
            )
            try:
                stdout, stderr, return_code = container.run_command(
                    command=command,
                    timeout=timeout,
                    env=env,
                    cwd=code_folder,
                )
                _, _, _ = container.run_command(
                    command=f"coverage json",
                    timeout=10,
                    cwd=code_folder,
                )
                raw_report, _, raw_report_return_code = container.run_command(
                    command=f"cat coverage.json",
                    timeout=10,
                    cwd=code_folder,
                )

                if raw_report_return_code > 0:
                    return None, stdout, stderr, return_code
                report = json.loads(raw_report)
                return report, stdout, stderr, return_code
            finally:
                container.clean_up()


# Remote Podman execution
if os.environ.get("SYNTHGEN_USE_REMOTE_PODMAN") == "1":
    from custome.podman_generating import (
        execute_python,
        run_pytest,
        run_test_coverage,
    )

