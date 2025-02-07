# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union


def split_command(command: str) -> List[str]:
    """
    Splits a command string into a list of arguments, taking into account quoted substrings.

    Args:
        command (str): The command string to split.

    Returns:
        List[str]: A list of command arguments.

    Example:
        >>> split_command("ls -l 'My Documents'")
        ['ls', '-l', 'My Documents']
    """

    pattern = r"[^\s']+|'[^']*'"
    return [part.strip("'") for part in re.findall(pattern, command)]


def run_command_sync(
    cmd: Union[str, List[str]],
    timeout: Optional[int] = None,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    parse_json: bool = False,
) -> Tuple[Any, str, int]:
    """
    Execute a command and return its output, error message, and exit code.

    Args:
        cmd (Union[str, List[str]]): The command to execute. Can be a string or a list of strings.
        timeout (Optional[int]): The maximum time in seconds to allow the command to run.
        cwd (Optional[str]): The working directory in which to execute the command.
        env (Optional[dict]): The environment variables to use when executing the command.
        parse_json (bool): If True, parse the command's stdout as JSON.

    Returns:
        Tuple[str, str, int]: A tuple containing the stdout, stderr, and exit code of the command.
    """

    if isinstance(cmd, str):
        cmd = split_command(cmd)
    cmd = [c for c in cmd if c]
    # Start the subprocess
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, env=env
    )

    stdout = b""
    stderr = b""
    try:
        # Wait for the process to complete, and capture stdout and stderr
        stdout, stderr = process.communicate(timeout=timeout)  # timeout
        # stdout, stderr = process.communicate()
    except subprocess.TimeoutExpired:
        # If the process does not finish within the timeout, terminate it
        process.kill()
        # stdout, stderr = process.communicate(timeout=timeout)  # Ensure you read any output even after killing

    def stream2str(stream):
        string = stream.decode(errors="ignore")
        # string = re.sub(r"[\u0000-\u001F\u007F-\u009F]", "", string)
        return string

    stdout = stream2str(stdout)
    stderr = stream2str(stderr)

    # parse stdout as json, if requested.
    if parse_json:
        stdout = json.loads(stdout)

    return stdout, stderr, process.returncode
