# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import tempfile
import time
from sys import stdout

import pytest

from synth_gen.execution.code_execution import execute_code
from synth_gen.execution.containerized_execution import (
    clean_up_persistent_container,
    PersistentContainer,
    run_command_in_container,
)

from synth_gen.execution.cpp_execution import (
    run_cpp,
    run_cpp_code_test_with_stdio,
    run_cpp_with_stdio,
)
from synth_gen.execution.python_dependencies import (
    get_installed_packages,
    install_dependencies,
)

from synth_gen.execution.python_execution import (
    execute_python,
    PersistentPythonContainer,
    run_pytest,
    run_test_coverage,
)
from synth_gen.execution.shell_execution import PersistentJupyterContainer

from synth_gen.generation import DataGenerator

script_folder = os.path.dirname(os.path.realpath(__file__))


def test_run_command_in_container():
    command = """sh -c 'echo $A'"""
    volumes = {"script_folder": "/home/user/lib"}
    expected_output = "aaa\n"
    output, _, returncode = run_command_in_container(
        command, timeout=2, volumes=volumes, env={"A": "aaa"}
    )
    assert (
        output.strip() == expected_output.strip()
    ), f"Expected {expected_output}, but got {output}"


def test_run_from_data_generator():
    generator = DataGenerator(programming_language="python")
    stdout, stderr, return_code = generator.execute_code(
        "print('hello world')", timeout=2
    )
    assert return_code == 0, f"Expected return code 0, but got {return_code}"


def test_run_command_in_container_with_dockerfile():
    command = """ls /app/"""
    docker_file_folder = os.path.join(script_folder, "podman_image/")
    output, _, returncode = run_command_in_container(
        command, timeout=2, docker_file_folder=docker_file_folder
    )
    assert "canary" in output.strip(), f"Expected to find canary file, but got {output}"


def test_run_command_in_container_with_dockerfile_content():
    command = """ls /app/"""
    docker_file_content = """
FROM ubuntu:latest

WORKDIR /app

RUN touch /app/canary
"""
    output, _, returncode = run_command_in_container(
        command, timeout=2, docker_file_content=docker_file_content
    )
    assert "canary" in output.strip(), f"Expected to find canary file, but got {output}"


def test_persistent_container():
    persistent_container = PersistentContainer()
    _, _, _ = persistent_container.run_command("touch abcd")
    print(persistent_container.name)
    assert persistent_container.name is not None
    stdout, stderr, returncode = persistent_container.run_command("ls")
    print(stdout)
    print(stderr)
    assert returncode == 0
    assert "abcd" in stdout
    persistent_container.clean_up()


def test_python_depencies():
    get_installed_packages()
    test_folder = os.path.join(
        tempfile.gettempdir(), "synth_gen", "python_dependencies_test"
    )
    os.makedirs(test_folder, exist_ok=True)
    with open(
        os.path.join(test_folder, "temp_file.py"),
        "w",
    ) as file:
        file.write("import faker\n")

    t0 = time.time()
    python_paths = install_dependencies(test_folder)
    t1 = time.time()
    assert any(
        "faker" in path.lower() for path in python_paths
    ), "Faker package should be insalled by the python dependency tool."
    print(python_paths)
    print(t1 - t0)


def test_run_python():
    stdout, stderr, return_code = execute_python(
        """
import faker
print('hello world')
""",
        timeout=2,
    )
    assert return_code == 0, stderr
    assert stdout.strip() == "hello world"


def test_timeout():
    stdout, stderr, return_code = execute_python(
        """
import time
time.sleep(10)
""",
        timeout=2,
    )
    assert (
        "timeout" in stdout or "timeout" in stderr
    ), "The standard output should mention timeout."
    assert return_code > 0, "The process should fail due to timeout."


def test_run_pytest():
    report, stdout, stderr, return_code = run_pytest(
        """
def test_true():
    assert True, "This is a true test"
"""
    )
    print([stdout, stderr, return_code])
    assert (
        return_code == 0
    ), f"the true test should pass. {return_code} / {stdout} / {stderr}"
    assert isinstance(
        report, dict
    ), f"Expected stdout to be a dict, but got {type(report)}. {report}"

    report, stdout, stderr, return_code = run_pytest(
        """
def false_test():
    assert False, "This is a false test"
"""
    )
    assert (
        return_code > 0
    ), f"the false test should fail. {return_code} / {stdout} / {stderr}"
    assert isinstance(
        report, dict
    ), f"Expected stdout to be a dict, but got {type(report)}. {report}"


def test_run_test_coverage():
    code = """
def foo(x):
    if x>0:
        return x
    else:
        return -x

def test_positive():
    assert foo(1) == 1
"""
    report, stdout, stderr, return_code = run_test_coverage(code)
    assert (
        report is not None
    ), f"Coverage verifier failed to produce a report. {stdout} {stderr}"
    assert return_code == 0
    assert report["totals"]["missing_lines"] == 1


def test_persistent_bash():
    container = PersistentJupyterContainer("bash")
    container.run_command("export A=123")
    result, error_message, is_failure = container.run_command("echo $A")
    assert result.strip() == "123", f"Expected a=123, but got {result}"
    container.run_command("touch tmp_abc")
    result, error_message, is_failure = container.run_command("ls")
    assert "tmp_abc" in result, f"Expected to find tmp_abc, but got {result}"


def test_persistent_python():
    container = PersistentPythonContainer()
    container.run_command("a=123")
    result, error_message, is_failure = container.run_command("a")
    assert result == "123", f"Expected a=123, but got {result}"
    result, error_message, is_failure = container.run_command("+**#")
    assert len(error_message) > 0, f"Expected to fail, but got {result}"
    assert is_failure is True, f"Expected to fail, but got {result}"
    result, error_message, is_failure = container.run_command("print(123)")
    assert not is_failure
    assert (
        "123" in result
    ), "Persistent python container did not properly capture print output."
    result, error_message, is_failure = container.run_command("import tqdm;tqdm.tqdm()")
    print(result, error_message, is_failure)
    assert (
        is_failure or "AssertionError" in result
    ), f"Importing a non-installed package in a persistent python container should fail.\n{result}\n{error_message}"
    # result, error_message, is_failure = container.run_command("!pip install tqdm")
    # result, error_message, is_failure = container.run_command("import tqdm;tqdm.tqdm()")
    # print(result, error_message, is_failure)
    # assert (not is_failure) and (
    #     not "AssertionError" in result
    # ), f"Could not import tqdm after it was installed in the persistent python container.\n{result}\n{error_message}"


def test_cpp_execution():
    code = """
#include <iostream>
int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}"""
    stdout, stderr, return_code = run_cpp(code)
    assert return_code == 0
    assert stdout.strip() == "Hello, World!"


def test_cpp_code_forces_test():
    code = """
#include <iostream>

int doubleValue(int x) {
    return 2 * x;
}

int main() {    
    int input;
    std::cin >> input;
    
    std::cout << doubleValue(input) << std::endl;
}
"""
    report, is_success, error_message = run_cpp_code_test_with_stdio(
        code, inputs=["5", "7", "12"], outputs=["10", "14", "24"]
    )
    assert is_success, json.dumps(report, indent=2)
    report, is_success, error_message = run_cpp_code_test_with_stdio(
        code, inputs=["5"], outputs=["11"]
    )
    assert not is_success, "This test should fail."
    bad_code = """
$53$#$##$#$#$
"""
    report, is_success, error_message = run_cpp_code_test_with_stdio(
        bad_code, inputs=["5"], outputs=["10"]
    )
    assert not is_success, "This test should fail."
    assert "error" in error_message, json.dumps(report, indent=2)


if __name__ == "__main__":
    test_persistent_python()
