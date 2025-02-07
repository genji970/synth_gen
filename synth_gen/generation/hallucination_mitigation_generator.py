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


@dataclass
class FalsePromiseGeneration(DataGeneration):
    fact: Optional[str] = None
    false_promise: Optional[str] = None


class FalsePromiseGenerator(DataGenerator):
    def _generate_training_data(
        self, seed: Optional[str] = None
    ) -> Sequence[FalsePromiseGeneration]:
        facts_prompt = f"""
Have a look at the following text from wikipedia:
{seed}

First, extract a list of facts from this text.
Format them as a list of strings, one string per fact. Write "- " before each fact.

"""
        raw_facts = self.call_llm(facts_prompt)
        generations = []
        facts = extract_list_from_string(raw_facts)
        for fact in facts:
            if self.verbose:
                print("FACT:")
                print(fact)
            false_promise_prompt = f"""
Have a look at the following fact from wikipedia:
{fact}

Modify this fact to make it wrong.
Surround the modified fact with triple backticks.
"""
            false_promise = self.call_llm(
                false_promise_prompt, extract_triple_backtick=True
            ).strip()
            if self.verbose:
                print("FALSE PROMISE:")
                print(false_promise)
            question_prompt = f"""
Write a question that is based on the following false assumption (It should assume that the following statement is true, even though it is wrong):
{false_promise}

Surround the question with triple backticks.
"""
            question = self.call_llm(
                question_prompt, extract_triple_backtick=True)
            if self.verbose:
                print("QUESTION:")
                print(question)
            answer_prompt = f"""
Write an answer to the following question:
{question}

But, be careful, this question assumes the followinf false promise:
{false_promise}

However, the following fact is true:
{fact}

So, when answering you should first gently state the correct fact and then try your best to answer the question based on the true fact if possible.
"""
            answer = self.call_llm(answer_prompt)
            if self.verbose:
                print("ANSWER:")
                print(answer)

            generation = UnkownAnswerGeneration()
            generation.fact = fact
            generation.false_promise = false_promise
            generation.chat = ChatPromptValue(
                messages=[
                    HumanMessage(content=question),
                    AIMessage(content=answer),
                ]
            )
            generations.append(generation)
        return generations


@dataclass
class UnkownAnswerGeneration(DataGeneration):
    pass


class UnkownAnswerGenerator(DataGenerator):
    def _generate_training_data(
        self, seed: Optional[str] = None
    ) -> Sequence[UnkownAnswerGeneration]:
        questions_prompt = f"""
Have a look at the following text from wikipedia:
{seed}

List one or two questions questions related to this text that are not answerable. They may ask about an extremely small detail that is probably unkown. Make sure that the question is not answerable.
Format them as a list of strings, one string per fact. Write "- " before each fact.

"""
        question_facts = self.call_llm(questions_prompt)
        generations = []
        questions = extract_list_from_string(question_facts)
        for question in questions:

            if self.verbose:
                print("QUESTION:")
                print(question)

            answer_prompt = f"""
Write an answer to the following question:
{question}

But be careful, the answer to this question is probably unknown. It is okay to answer that you do not know.
"""

            answer = self.call_llm(answer_prompt)
            if self.verbose:
                print("ANSWER:")
                print(answer)

            generation = UnkownAnswerGeneration()
            generation.chat = ChatPromptValue(
                messages=[
                    HumanMessage(content=question),
                    AIMessage(content=answer),
                ]
            )
            generations.append(generation)
        return generations
