# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from asyncio.timeouts import timeout
from tempfile import tempdir
from typing import Dict, Optional, Tuple

import mmh3
from langchain_core.utils import image

from synth_gen.execution.execution_utils import run_command_sync

script_folder = os.path.dirname(os.path.realpath(__file__))

default_image_name = "ubuntu:latest"
default_memory_limit = "512m"
default_timeout = 60


class PodmanPool:
    def __init__(self):
        self.max_container = 10
        self.max_run_per_container = 1
        # self.image_name = "ubuntu_container"
        self.user_id = run_command_sync("id -u")[0].strip()
        self.group_id = run_command_sync("id -g")[0].strip()
        self.check_if_podman_is_installed()

    @staticmethod
    def check_if_podman_is_installed():
        # Check that podman is installed
        process = subprocess.run(
            "podman --help", shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE
        )
        if process.returncode != 0:
            print(process.stderr)
            print("It seems like 'podman' is not installed. Please install 'podman'")
            sys.exit(1)

    def build_container_image_if_necessary(self, docker_file_folder):
        os.makedirs(docker_file_folder, exist_ok=True)
        docker_file_content = open(
            os.path.join(docker_file_folder, "Dockerfile"), "r"
        ).read()
        docker_file_hash = str(abs(mmh3.hash(docker_file_content)))
        image_name = f"image-{docker_file_hash}"
        # Check that the podman image is built.
        image_list, _, _ = run_command_sync(
            "podman image list --format json", parse_json=True
        )
        is_image_name_within_image_list = any(
            "Names" in image and image_name in image["Names"][0] for image in image_list
        )
        # Build the image if necessary
        if not is_image_name_within_image_list:
            cmd = (
                f"""podman build --network=host -t {image_name} {docker_file_folder}"""
            )
            print(cmd)
            subprocess.run(cmd, shell=True, check=True)
        return image_name

    def run_command_in_container(
        self,
        command,
        volumes={},
        timeout=default_timeout,
        cwd=None,
        env=None,
        memory_limit=default_memory_limit,
        name=None,
        ports: Optional[Dict[str, str]] = None,
        persistent_start=False,
        persistent_continue=False,
        image_name=default_image_name,
        dont_wait=False,
        docker_file_folder=None,
        docker_file_content=None,
        parse_json=False,
    ) -> Tuple[str, str, int]:
        if docker_file_content:
            if docker_file_folder:
                raise Exception(
                    "If you provide docker file content, you should not provide a docker file folder."
                )
            docker_file_folder = tempfile.mkdtemp()
            with open(os.path.join(docker_file_folder, "Dockerfile"), "w") as file:
                file.write(docker_file_content)

        if (
            docker_file_folder
            and image_name != default_image_name
            and image_name is not None
        ):
            raise Exception(
                "If you provide a docker file path, you should not provide an image name."
            )
        if docker_file_folder is not None:
            if (
                not os.path.isdir(docker_file_folder)
                and os.path.exists(docker_file_folder)
                and docker_file_folder.lower().endswith("dockerfile")
            ):
                docker_file_folder = os.path.dirname(docker_file_folder)
            image_name = self.build_container_image_if_necessary(docker_file_folder)
        # pick up a random container name
        if name is None:
            name = str(uuid.uuid4())
        # default image name
        if image_name is None:
            image_name = default_image_name
        # set up volumes
        volume_command = ""
        if volumes is not None and volumes:
            for host_folder, container_folder in volumes.items():
                volume_command += f""" -v {host_folder}:{container_folder}"""
            volume_command += (
                f" --userns=keep-id:uid={self.user_id},gid={self.group_id}"
            )
        workdir_command = ""
        # set up ports
        port_command = ""
        if ports is not None:
            for port_1, port_2 in ports.items():
                port_command += f""" -p {port_1}:{port_2}"""
        # set up working dir
        if cwd:
            workdir_command = f"""--workdir {cwd}"""
        timeout_command = ""
        # set up time out
        if timeout and not persistent_start:
            #     timeout_command = f""" --timeout {timeout} """
            # if timeout and persistent_continue:
            timeout_command = ""
            command = f"""timeout -v {timeout} {command}"""
        # set up environment variables
        env_command = ""
        if env is not None:
            env_command = " ".join(
                f"""-e {variable}={value}""" for variable, value in env.items()
            )
        # set up memory limit
        memory_limit_command = ""
        if memory_limit:
            memory_limit_command = f"""--memory {memory_limit}"""
        # set up peristent option
        action = "run -it "
        if dont_wait:
            action = "run -itd "
        if persistent_start:
            action = "run -d "
            command = "tail -f /dev/null"

        # run the command in the container
        podman_command = f"""podman {action} {timeout_command} {env_command} --replace {memory_limit_command} --name {name} {volume_command} {port_command} {workdir_command} {image_name} {command}"""
        if persistent_continue:
            dont_wait_command = ""
            if dont_wait:
                dont_wait_command = "d"
            podman_command = f"""podman exec -it{dont_wait_command} {workdir_command} {env_command} {name} {command}"""
        print(podman_command)
        stdout, stderr, return_code = run_command_sync(podman_command)
        if timeout and stdout == "" and stderr == "" and return_code == 255:
            stderr = "Time out. The process took too long."

        # async clean up, kill and remove container if necessary
        if not persistent_start and not persistent_continue:
            self.clean_up(name, dont_wait=True)

        if parse_json:
            # Parse the last line of stdout
            if stdout:
                last_line = stdout.strip().split("\n")[-1]
                try:
                    parsed_json = json.loads(last_line)
                    stdout = parsed_json
                except json.JSONDecodeError:
                    pass
        return stdout, stderr, return_code

    def clean_up(self, name: str, dont_wait=False):
        if dont_wait:
            t = threading.Thread(target=self.clean_up, args=(name,))
            t.start()
        else:
            timeout = 30
            try:
                run_command_sync(f"podman kill {name}", timeout=timeout)
            except Exception as e:
                print(e)
                pass
            try:
                run_command_sync(f"podman rm {name}", timeout=timeout)
            except Exception as e:
                print(e)


podman_pool = None


def run_command_in_container(
    command,
    volumes={},
    timeout=default_timeout,
    cwd=None,
    env=None,
    memory_limit=default_memory_limit,
    image_name=default_image_name,
    docker_file_folder=None,
    docker_file_content=None,
    parse_json=False,
) -> Tuple[str, str, int]:
    """
    Run a command inside a container using Podman.

    Args:
        command (str): The command to run inside the container.
        volumes (dict, optional): A dictionary mapping host paths to container paths for volume mounting.
        timeout (int, optional): The maximum time in seconds to allow the command to run.
        cwd (str, optional): The working directory inside the container.
        env (dict, optional): Environment variables to set inside the container.
        memory_limit (str, optional): The memory limit for the container.
        image_name (str, optional): The name of the container image to use.
        docker_file_folder (str, optional): The folder containing the Dockerfile to build the image.
        docker_file_content (str, optional): The content of the Dockerfile to build the image.

    Returns:
        Tuple[str, str, int]: A tuple containing the standard output, standard error, and return code of the command.
    """

    global podman_pool
    if podman_pool is None:
        podman_pool = PodmanPool()
    return podman_pool.run_command_in_container(
        command,
        volumes=volumes,
        timeout=timeout,
        cwd=cwd,
        env=env,
        memory_limit=memory_limit,
        image_name=image_name,
        docker_file_folder=docker_file_folder,
        docker_file_content=docker_file_content,
        parse_json=parse_json,
    )


class PersistentContainer:
    """
    A class to manage a persistent container using Podman.

    This class allows for the creation and management of a persistent container
    that can run multiple commands over its lifetime without being destroyed
    after each command execution.

    Attributes:
        name (str): The name of the persistent container.
        podman_pool (PodmanPool): An instance of PodmanPool to manage container operations.

    Methods:
        __init__: Initializes a new persistent container with the specified parameters.
        run_command: Runs a command inside the persistent container.
        clean_up: Cleans up the persistent container by stopping and removing it.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        volumes: Optional[Dict[str, str]] = None,
        memory_limit: Optional[str] = None,
        dont_wait: bool = False,
        ports: Optional[Dict[str, str]] = None,
        image_name: Optional[str] = None,
        docker_file_folder: Optional[str] = None,
        docker_file_content: Optional[str] = None,
    ):
        if image_name is None:
            image_name = default_image_name
        if name is None:
            name = "persitent-" + str(uuid.uuid4())
        self.name = name
        global podman_pool
        if podman_pool is None:
            podman_pool = PodmanPool()
        self.podman_pool = podman_pool
        podman_pool.run_command_in_container(
            "",
            volumes=volumes,
            ports=ports,
            memory_limit=default_memory_limit,
            name=self.name,
            image_name=image_name,
            persistent_start=True,
            dont_wait=dont_wait,
            docker_file_folder=docker_file_folder,
            docker_file_content=docker_file_content,
        )

    def run_command(
        self,
        command,
        timeout=default_timeout,
        cwd=None,
        env=None,
        dont_wait=False,
    ):
        return self.podman_pool.run_command_in_container(
            command,
            timeout=timeout,
            cwd=cwd,
            env=env,
            name=self.name,
            persistent_continue=True,
            dont_wait=dont_wait,
        )

    def clean_up(self, dont_wait=False):
        self.podman_pool.clean_up(self.name, dont_wait=dont_wait)


def clean_up_persistent_container(name):
    global podman_pool
    if podman_pool is None:
        podman_pool = PodmanPool()
    podman_pool.clean_up(name, dont_wait=False)
