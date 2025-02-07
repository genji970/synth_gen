# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from typing import Optional, Sequence, Tuple

from synth_gen.generators import DataGeneration, DataGenerator
from synth_gen.schema import AIMessage, ChatPromptValue, HumanMessage
from synth_gen.verification.schema import Verification, Verifier


class MyGenerator(DataGenerator):
    def _generate_training_data(
        self, seed: Optional[str] = None
    ) -> Sequence[DataGeneration]:
        generation = DataGeneration()
        generation.chat = ChatPromptValue(
            messages=[HumanMessage(content="my prompt"), AIMessage(content="my answer")]
        )
        generation.is_correct = True
        return [generation]


from synth_gen.generators import VerifiedGeneration, VerifiedGenerator


class MyVerifiedGenerator(VerifiedGenerator):
    def __init__(self):
        super().__init__()
        self.verifiers = [MyVerifier1(), MyVerifier2()]

    def _generate_prompt(self, seed: Optional[str]) -> str:
        return "my problem"

    def _generate_response(self, generation: VerifiedGeneration) -> str:
        return "my response"

    def _fix_response(
        self, generation: VerifiedGeneration, verification: Verification
    ) -> str:
        return "my fixed response"


from synth_gen.verification.schema import Verification, Verifier


class MyVerifier(Verifier):
    def check(
        self,
        generation: VerifiedGeneration,
    ) -> Tuple[bool, Optional[str]]:
        return False, "my error message"
