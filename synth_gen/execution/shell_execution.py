# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import time
from typing import Optional

from synth_gen.execution.containerized_execution import PersistentContainer

script_folder = os.path.dirname(os.path.realpath(__file__))


class PersistentJupyterContainer:
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

    def __init__(
        self,
        kernel_name,
        memory_limit=None,
        docker_file_content=None,
    ):
        kernel_module = kernel_name + "_kernel"
        app_folder = os.path.join(script_folder, f"containers/persistent_{kernel_name}")
        os.makedirs(app_folder, exist_ok=True)
        volumes = {
            app_folder: "/app/",
        }
        if docker_file_content is not None:
            docker_file_folder = None
        else:
            docker_file_folder = os.path.join(
                script_folder, f"containers/persistent_{kernel_name}/Dockerfile"
            )
        self.persistent_container = PersistentContainer(
            volumes=volumes,
            memory_limit=memory_limit,
            docker_file_folder=docker_file_folder,
            docker_file_content=docker_file_content,
        )
        # self.persistent_container.run_command("python3 -m venv venv")
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


if __name__ == "__main__":
    container = PersistentJupyterContainer("bash", "bash_kernel")
    stdout, stderr, return_code = container.run_command("pwd")
    print(stdout)
    print(stderr)
    print(return_code)
