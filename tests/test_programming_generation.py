# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from synth_gen.generation.LLM import llama3_1_405B

from synth_gen.generation.programming import (
    generate_programming_problem_from_seed,
    generate_programming_test,
    # generate_verified_programming_dialogue,
)

# print(
#     generate_correct_programming_dialogue(
#         "fibo... fibo... fibofibo... fibofibofibo... fibofibofibofibofibo... fibofibofibofibofibofibofibofibo..."
#     )
# )

print(
    generate_programming_test(
        "Implement Fibonacci function.", "fibo(n:int)->int", llm=llama3_1_405B
    )
)
