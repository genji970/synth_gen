# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time

import pytest
from synth_gen.generation.LLM import call_llm


def test_call_llm():
    response = call_llm("Jus say AH", use_cache=True)
    assert "AH" in response, f"Expected 'AH' in response, but got {response}"
    t0 = time.time()
    response = call_llm("Jus say AH", use_cache=True)
    t1 = time.time()
    assert (
        t1 - t0 < 0.1
    ), f"Expected cached response time to be less than 0.1 seconds, but got {t1-t0}s"
