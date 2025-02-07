# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import ast
import glob
import json
import os
import subprocess
import tempfile
import threading
import time
from functools import wraps
from typing import List, Set, Tuple

import pipreqs
from dpath import get

script_folder = os.path.dirname(os.path.realpath(__file__))


def get_packages_mapping():
    with open(os.path.join(list(pipreqs.__path__)[0], "mapping"), "r") as f:
        mapping = dict(x.strip().split(":") for x in f)
    return mapping


package_mapping = get_packages_mapping()


def thread_safe_memoize_with_expiration(lifespan=300):
    def thread_safe_memoize_with_expiration_decorator(func):
        cache = {}
        lock = threading.Lock()

        @wraps(func)
        def memoized_func(*args, **kwargs):
            key = (args, tuple(kwargs.items()))
            current_time = time.time()
            with lock:
                # Check if the result is already in the cache and is not expired
                if key in cache and current_time - cache[key]["time"] <= lifespan:
                    return cache[key]["result"]
                # Call the function and save the result along with the current time in the cache
                result = func(*args, **kwargs)
                cache[key] = {"result": result, "time": current_time}
                return result

        return memoized_func

    return thread_safe_memoize_with_expiration_decorator


@thread_safe_memoize_with_expiration(500)
def get_installed_packages() -> Set[str]:
    pip_list_output = subprocess.check_output(["pip", "list", "--format=json"])
    packages = [p["name"] for p in json.loads(pip_list_output.decode())]
    return set(packages)


def get_required_packages(folder_path: str) -> List[str]:
    # subprocess.run(["pipreqs", "--force", "--use-local", "--mode", "no-pin", "."], check=True, cwd=folder_path)
    subprocess.run(
        ["pipreqs", "--force", "--mode", "no-pin", "."], check=True, cwd=folder_path
    )
    requirements_path = os.path.join(folder_path, "requirements.txt")
    required_packages = []
    with open(requirements_path, "r") as file:
        for line in file.readlines():
            package_name = line.split("==")[0].strip()
            required_packages.append(package_name)
    return required_packages


def get_required_packages_fast(folder_path: str) -> List[str]:
    # print(folder_path)
    installed_packages = get_installed_packages()
    required_packages = set()
    for python_file in glob.glob(f"{folder_path}/**/*.py", recursive=True):
        with open(python_file, "r") as file:
            code = file.read()
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        required_packages.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    required_packages.add(node.module)
    # print(required_packages)
    missing_packages = required_packages - installed_packages
    missing_packages = [
        str(get(package_mapping, name, default=name)) for name in missing_packages
    ]
    return list(missing_packages)


dependency_folder = os.path.join(
    tempfile.gettempdir(), "containers/python/dependencies"
)
os.makedirs(dependency_folder, exist_ok=True)


def get_dependency_paths(
    to_be_installed_packages: List[str], do_install=True
) -> Tuple[bool, List[str]]:
    is_success = True
    library_folder_paths = []
    for package in to_be_installed_packages:
        library_folder_name = f"lib-{package}"
        library_folder_path = os.path.join(
            dependency_folder, library_folder_name)
        library_folder_paths.append(library_folder_path)
        if not os.path.exists(library_folder_path):
            if do_install:
                os.makedirs(library_folder_path, exist_ok=True)
                try:
                    subprocess.run(
                        ["pip", "install", package, "--target", library_folder_path],
                        check=True,
                    )
                except subprocess.CalledProcessError as e:
                    print(f"Failed to install {package}: {e}")
                    is_success = False
                    library_folder_paths = []
            else:
                is_success = False
    return is_success, library_folder_paths


def install_dependencies(folder_path: str) -> List[str]:
    required_packages = get_required_packages_fast(folder_path)
    is_success, python_paths = get_dependency_paths(
        required_packages, do_install=False)
    """
    Installs the necessary dependencies for the given folder path.

    This function first attempts to get the required packages quickly using
    the `get_required_packages_fast` method, which checks for missing packages
    that are not installed. If successful, it returns the paths to the installed
    packages. If not, it falls back to using the `get_required_packages` method
    to determine the required packages and installs them.

    Args:
        folder_path (str): The path to the folder for which dependencies need to be installed.

    Returns:
        List[str]: A list of paths to the installed package directories.
    """

    if is_success:
        return python_paths
    required_packages = get_required_packages(folder_path)
    is_success, python_paths = get_dependency_paths(
        required_packages, do_install=True)
    return python_paths
