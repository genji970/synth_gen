# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import time
from json.decoder import JSONDecodeError

import requests


def run_command(command):
    url = "http://localhost:8000/run"
    payload = {"command": command}
    response = requests.post(url, data=payload, timeout=60)
    for _ in range(50):  # Retry 50 times (every 100ms for 5 seconds)
        if response.ok:
            break
        time.sleep(0.1)
        response = requests.post(url, data=payload, timeout=5)
    return response.text


if __name__ == "__main__":
    # result = run_command("1+2")
    # print(result)
    # result = run_command("import numpy as np; np.asarray([1,2,3])")
    # print(result)
    # result = run_command("1./3.")
    # print(result)
    # result = run_command("[1./3. for i in range(1000_000)]")
    # print(result)
    command = sys.argv[1]
    result = run_command(command)
    print(result)
