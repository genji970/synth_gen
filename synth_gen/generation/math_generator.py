# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from synth_gen.generation import generator
from synth_gen.generation.generation import DataGeneration
from synth_gen.generation.generator import DataGenerator
from synth_gen.generation.seed_source import SeedSource
from synth_gen.schema import AIMessage, ChatPromptValue, HumanMessage


@dataclass
class SimpleMathDataGeneration(DataGeneration):
    formula: str = ""
    response: str = ""


class EnumerationStrategy(Enum):
    diagonal = "diagonal"
    diamand = "diamond"


class DoubleIntegerSeedSource(SeedSource):
    def __init__(self, enumeration_strategy: EnumerationStrategy):
        self.enumeration_strategy = enumeration_strategy
        self.int_1 = 0
        self.int_2 = 0

    def get_next_seed(self) -> Union[str, Dict[str, Any]]:
        if self.enumeration_strategy == EnumerationStrategy.diagonal:
            if self.int_1 == 0:
                self.int_1 = self.int_2 + 1
                self.int_2 = 0
            else:
                self.int_1 -= 1
                self.int_2 += 1
        elif self.enumeration_strategy == EnumerationStrategy.diamand:
            if self.int_1 == 0 and self.int_2 == 0:
                self.int_1 = 1
                self.int_2 = 0
            elif self.int_1 > 0 and self.int_2 >= 0:
                self.int_1 -= 1
                self.int_2 += 1
            elif self.int_1 <= 0 and self.int_2 > 0:
                self.int_1 -= 1
                self.int_2 -= 1
            elif self.int_1 < 0 and self.int_2 <= 0:
                self.int_1 += 1
                self.int_2 -= 1
            elif self.int_1 >= 0 and self.int_2 < -1:
                self.int_1 += 1
                self.int_2 += 1
            elif self.int_1 >= 0 and self.int_2 == -1:
                self.int_1 += 2
                self.int_2 += 1
        return {"int_1": self.int_1, "int_2": self.int_2}


class SimpleMathPlusGenerator(DataGenerator):
    def __init__(self):
        self.seed_source = DoubleIntegerSeedSource(EnumerationStrategy.diamand)
        self.operator = "+"

    def _generate_training_data(
        self, seed: Optional[str] = None
    ) -> Sequence[SimpleMathDataGeneration]:
        arg_1 = seed["int_1"]
        arg_2 = seed["int_2"]
        prompt = f"""{arg_1}{self.operator}{arg_2}"""
        solution = str(eval(f"""{arg_1}{self.operator}{arg_2}"""))
        chat = ChatPromptValue(
            messages=[HumanMessage(content=prompt),
                      AIMessage(content=solution)]
        )
        return [
            SimpleMathDataGeneration(
                formula=prompt, response=solution, chat=chat, is_correct=True
            )
        ]


class SimpleMathMinusGenerator(SimpleMathPlusGenerator):
    def __init__(self):
        super().__init__()
        self.operator = "-"


class SimpleMathTimesGenerator(SimpleMathPlusGenerator):
    def __init__(self):
        super().__init__()
        self.operator = "*"


if __name__ == "__main__":
    generator = SimpleMathTimesGenerator()
    for t in range(50):
        print(generator.generate_training_data()[0].to_json())
