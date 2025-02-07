# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from synth_gen.generation import generator

from synth_gen.generation.generation import DataGeneration

from synth_gen.generation.generator import DataGenerator

from synth_gen.generation.parsing import (
    extract_list_from_string,
    extract_triple_backticked_string,
)

from synth_gen.schema import AIMessage, ChatPromptValue, HumanMessage


class QAGenerator(DataGenerator):
    def _generate_training_data(
        self, seed: Optional[str] = None
    ) -> Sequence[DataGeneration]:
        generations = []
        facts_prompt = f""" 
Extract a list of questions and answers from the following text from wikipedia:
{seed}

Use this format:
- QUESTION: <question>
- ANSWER: <answer>
"""
        raw_QAs = self.call_llm(facts_prompt)
        QA_list = extract_list_from_string(raw_QAs)
        QAs = []
        for n in range(0, len(QA_list) - 1, 2):
            raw_question = QA_list[n]
            raw_answer = QA_list[n + 1]
            if raw_question.startswith("QUESTION:") and raw_answer.startswith(
                "ANSWER:"
            ):
                question = raw_question[len("QUESTION:"):].strip()
                answer = raw_answer[len("ANSWER:"):].strip()
                QAs.append((question, answer))
                if self.verbose:
                    print("QUESTION:")
                    print(question)
                    print("ANSWER:")
                    print(answer)
                generation = DataGeneration()
                generation.chat = ChatPromptValue(
                    messages=[
                        HumanMessage(content=question),
                        AIMessage(content=answer),
                    ]
                )
                generations.append(generation)
        return generations
