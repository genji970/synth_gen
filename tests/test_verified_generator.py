# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from typing import Optional, Tuple, Union

import pytest

from synth_gen.generation.generator import VerifiedGenerator
from synth_gen.verification.schema import Verification, VerifiedGeneration, Verifier


class MyVerifiedGenerator(VerifiedGenerator):
    def _generate_prompt(self, seed: Optional[str] = None) -> str:
        return "my problem"

    def _generate_response(self, generation: VerifiedGeneration) -> str:
        return "my response"

    def _fix_response(
        self, generation: VerifiedGeneration, verification: Verification
    ) -> str:
        return "my fixed response"


class MyVerifier(Verifier):
    def __init__(self):
        super().__init__()
        self.count = 0

    def check(self, generation: VerifiedGeneration) -> Tuple[bool, Optional[str]]:
        if self.count == 0:
            self.count += 1
            return False, "my error"
        else:
            return True, None


class FailingVerifier(Verifier):
    def check(self, generation: VerifiedGeneration) -> Tuple[bool, Optional[str]]:
        return False, "my error"


class TestMyVerifiedGenerator:
    def test_generate_training_data(self):
        generator = MyVerifiedGenerator()
        data = generator.generate_training_data()
        assert data[0].prompt.messages[0].content == "my problem"
        assert data[0].response.content == "my response"
        assert data[0].is_correct is True

        generator.verifiers = [MyVerifier()]
        data = generator.generate_training_data()
        assert data[0].response.content == "my fixed response"
        assert data[0].is_correct is True

        generator.verifiers = [FailingVerifier()]
        data = generator.generate_training_data()
        assert data[0].is_correct is False


if __name__ == "__main__":
    pytest.main(["-q", __file__])
