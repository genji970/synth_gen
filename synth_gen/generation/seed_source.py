# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import ast
import random
from typing import Any, Callable, Dict, Generic, Optional, Tuple, TypeVar, Union

from datasets import load_dataset

SeedType = Optional[Union[str, Dict[str, Any]]]

SeedType = TypeVar("SeedType", bound=Union[str, Dict[str, Any], Any])


class SeedSource:
    def get_next_seed(self) -> Any:
        """
        Get seed that can be used to generate a prompt.
        Seeds help to generate diverse prompts on the long tail distribution of topics.
        """
        raise NotImplemented("get_seed is not implemented.")


class HuggingFaceSeedSource(SeedSource):
    def __init__(
        self,
        hugging_face_dataset_name: str,
        field_name: str,
        post_processing: Optional[Callable[[Any], Optional[str]]] = None,
        filter: Optional[Callable[[Any], bool]] = None,
    ):
        self.dataset = load_dataset(hugging_face_dataset_name)
        if "train" in self.dataset:
            self.dataset = self.dataset["train"]
        self.dataset_iterator = iter(self.dataset)
        self.field_name = field_name
        self.post_processing = post_processing
        self.filter = filter
        # self.sample_iterator = None

    def get_next_seed(self) -> str:
        while True:
            # if self.sample_iterator is not None:
            #     try:
            #         sub_sample = next(self.sample_iterator)
            #     except StopIteration:
            #         raise StopIteration("No more samples in the dataset.")
            sample = next(self.dataset_iterator)
            if self.filter is not None:
                while not self.filter(sample):
                    sample = next(self.dataset_iterator)
            if self.field_name is not None:
                if self.field_name in sample:
                    # sample = getattr(sample, self.field_name)
                    sample = sample[self.field_name]
                else:
                    raise ValueError(
                        f"Field {self.field_name} not found in sample {sample}.\n{sample.keys()}.\n{sample}"
                    )
            if self.post_processing is not None:
                sample = self.post_processing(sample)
            if sample is not None:
                return sample


def extract_snippet(code: str) -> str:
    # Pick up to 30 random continuous lines of code
    sample_size = 30
    max_characters = 1000
    lines = code.split("\n")
    line_count = len(lines)
    if line_count < sample_size:
        return code
    sample_ind = random.randint(0, line_count - sample_size)
    sample = "\n".join(lines[sample_ind : sample_ind + sample_size])
    if len(sample) > max_characters:
        sample = sample[: max_characters - 15] + " # [truncated]"
    return sample


class SmallPythonSeedSource(HuggingFaceSeedSource):
    def __init__(
        self, hugging_face_dataset_name="calum/the-stack-smol-python-docstrings"
    ):
        super().__init__(
            hugging_face_dataset_name,
            field_name="body",
            post_processing=extract_snippet,
        )


def extract_function_from_code(code: str) -> Optional[str]:
    # Parse the code into an Abstract Syntax Tree (AST)
    tree = ast.parse(code)

    # Find all function definitions in the AST
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

    # If there are no functions, return an empty string
    if not functions:
        return ""

    # Convert the first function definition back to source code
    function_code = ast.get_source_segment(code, functions[0])

    return function_code


class SelfContainedPythonFunctionSource(HuggingFaceSeedSource):
    def __init__(
        self, hugging_face_dataset_name="calum/the-stack-smol-python-docstrings"
    ):
        super().__init__(
            hugging_face_dataset_name,
            field_name="code",
            post_processing=extract_function_from_code,
        )


class EnglishVocabularySource(HuggingFaceSeedSource):
    def __init__(self):
        super().__init__(
            hugging_face_dataset_name="yukiarimo/english-vocabulary",
            field_name="text",
            filter=lambda data: data["text"].isalpha() and data["text"].islower(),
        )


def zipSeedSources(source_1: SeedSource, source_2: SeedSource):
    class CombinedSeedSource(SeedSource):
        def __init__(self):
            super().__init__()
            self.source_1 = source_1
            self.source_2 = source_2

        def get_next_seed(self) -> Tuple[Any, Any]:
            return (self.source_1.get_next_seed(), self.source_2.get_next_seed())

    combined_source = CombinedSeedSource()
    return combined_source
