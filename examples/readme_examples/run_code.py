# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from synth_gen.execution.code_execution import PersistentJupyterContainer

container = PersistentJupyterContainer(kernel_name="bash")
container.run_command("export A=123")
result, error_message, is_failure = container.run_command("echo $A")
print(result, error_message, is_failure)
