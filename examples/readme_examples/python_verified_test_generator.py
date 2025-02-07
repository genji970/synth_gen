# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from synth_gen.generators import PythonVerifiedTestGenerator
from synth_gen.verification.schema import VerifiedProgrammingGeneration

generator = PythonVerifiedTestGenerator()
generator.verbose = True
generation = generator.build_prompt(
    problem="Implement the Fibonnacci algorithm.",
    solution="""
cache = {}
def fibo(n):
    if n in cache:
        return cache[n]
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        res = fibo(n-1) + fibo(n-2)
        cache[n] = res
        return res
""",
)
generations = generator.generate_verified_responses(generation)
print(generations[0].response.content)
