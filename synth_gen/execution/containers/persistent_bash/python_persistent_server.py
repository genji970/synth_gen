# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time

import jupyter_client

from flask import Flask, jsonify, request

app = Flask(__name__)

km = jupyter_client.KernelManager(ip="127.0.0.1", kernel_name="bash")
km.start_kernel()
kc = km.client()


@app.route("/run", methods=["GET", "POST"])
def run_python_command():
    command = (
        request.args.get("command")
        if request.method == "GET"
        else request.form.get("command")
    )
    kc.execute(command)
    results = []
    messages = []
    is_failure = False
    error_messages = []
    while True:
        try:
            msg = kc.get_iopub_msg(timeout=100)
        except Exception as e:
            print(e)
            break
        messages.append(msg)
        if msg["msg_type"] == "execute_result":
            result = msg["content"]["data"]["text/plain"]
            results.append(result)
        if msg["msg_type"] == "error":
            error_message = (
                msg["content"]["evalue"]
                + "\n\n"
                + "\n\n".join(msg["content"]["traceback"])
            ).strip()
            error_messages.append(error_message)
            is_failure = True
        if msg["msg_type"] == "stream":
            if msg["content"]["name"] == "stdout":
                results.append(msg["content"]["text"])
            else:
                error_messages.append(msg["content"]["text"])
        if msg["msg_type"] == "status":
            if "execution_state" in msg["content"]:
                if msg["content"]["execution_state"] == "idle":
                    return jsonify(
                        {
                            "result": "\n\n".join(results),
                            "error_message": "\n\n\n".join(error_messages),
                            "is_failure": is_failure,
                            "raw": messages,
                        }
                    )


if __name__ == "__main__":
    app.run(port=8000)
