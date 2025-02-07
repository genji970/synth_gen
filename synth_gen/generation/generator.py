# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from re import S
from types import NoneType
from typing import (
    Any,
    cast,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import langchain
from langchain.schema import (
    AIMessage,
    BaseMessage,
    BaseMessage,
    Generation,
    HumanMessage,
)
from langchain_core.outputs import generation
from langchain_core.prompts import prompt
from synth_gen.execution.code_execution import (
    execute_code,
    execute_test,
    get_persistent_container,
    get_persistent_jupyter_notebook,
)
from synth_gen.execution.containerized_execution import PersistentContainer
from synth_gen.execution.python_execution import PersistentPythonContainer
from synth_gen.execution.shell_execution import PersistentJupyterContainer

from synth_gen.generation.generation import (
    DataGeneration,
    GenerationType,
    PromptAnswerGeneration,
)

from synth_gen.generation.LLM import call_llm, get_default_llm, LLM

from synth_gen.generation.LLM_based_parsing import (
    extract_programming_function_signature,
)
from synth_gen.generation.parsing import (
    extract_python_function_signature,
    extract_source_code,
)
from synth_gen.generation.programming import (
    build_prompt_for_defensive_programming_test,
    generate_programming_problem_from_seed,
    generate_programming_solution,
    generate_programming_test,
)
from synth_gen.generation.seed_source import SeedSource, SeedType, SmallPythonSeedSource
from synth_gen.schema import ChatPromptValue

from synth_gen.verification.programming_verifiers import (
    ProgrammingExecutionVerifier,
    ProgrammingGeneratedTestVerifier,
)
from synth_gen.verification.python_verifiers import (
    default_python_verifiers,
    MypyVerifier,
    NonEmptyVerifier,
    PythonGeneratedTestCoverageVerifier,
    PythonGeneratedTestVerifier,
    PythonSyntaxVerifier,
)
from synth_gen.verification.schema import (
    Verification,
    VerificationStatus,
    VerifiedGeneration,
    VerifiedProgrammingGeneration,
    Verifier,
)


class DataGenerator(Generic[GenerationType]):
    """
    Base class for all data generators.
    If you make your own data generator, derived it from DataGenerator, and implement the `_generate_training_data` method.
    You might also want to implement you own DataGenration (-ion not -or) class to store additional information in the generated data.
    """

    def __init__(
        self,
        seed_source: Optional[SeedSource] = None,
        llm: Optional[LLM] = None,
        code_executor: Optional[Any] = None,
        programming_language: Optional[str] = "python",
    ) -> None:
        self.seed_source = seed_source
        if llm is None:
            self.llm: LLM = get_default_llm()
        else:
            self.llm = llm
        self.programming_language = programming_language
        if code_executor:
            self.code_executor = code_executor
        else:
            self.code_executor = None
        self.verbose = False
        self.data_generation_cls: Type[GenerationType] = DataGeneration

    def set_llm(self, llm: LLM):
        self.llm = llm

    def call_llm(self, prompt, extract_triple_backtick=False) -> str:
        return str(
            call_llm(
                prompt, llm=self.llm, extract_triple_backtick=extract_triple_backtick
            )
        )

    def new_generation(self) -> GenerationType:
        return self.data_generation_cls()

    def set_code_executor(self, code_executor):
        self.code_executor = code_executor

    def execute_code(self, code, programming_language: str = "python", timeout=60):
        return execute_code(
            code, programming_language=programming_language, timeout=timeout
        )

    def get_persistent_jupyter_notebook(
        self,
        programming_language: str = "python",
        memory_limit: str = "500m",
        volumes: Optional[Dict[str, str]] = None,
    ):
        return get_persistent_jupyter_notebook(
            programming_language=programming_language,
            memory_limit=memory_limit,
            volumes=volumes,
        )

    def get_persistent_container(
        self,
        memory_limit="500m",
        volumes=None,
        image_name=None,
        docker_file_content=None,
    ):
        return get_persistent_container(
            memory_limit=memory_limit,
            volumes=volumes,
            image_name=image_name,
            docker_file_content=docker_file_content,
        )

    def execute_code_test(
        self, test_code, code="", programming_language: str = "python", timeout=60
    ):
        return execute_test(
            code, test_code, programming_language=programming_language, timeout=timeout
        )

    def generate_training_data(
        self, seed: Optional[Any] = None
    ) -> Sequence[GenerationType]:
        if seed is None:
            seed = self.get_next_seed()
        return self._generate_training_data(seed)

    def get_next_seed(self) -> Optional[Any]:
        if hasattr(self, "seed_source") and self.seed_source is not None:
            return self.seed_source.get_next_seed()
        return None

    def _generate_training_data(self, seed: Optional[Any]) -> Sequence[GenerationType]:
        raise NotImplementedError("Generate training data is not implemented.")

    def generate_dataset(self, n_sampe: int, output_path: str):
        """
        Generates a dataset by creating multiple samples of training data and writing them to a specified output file.

        Parameters:
        ----------
        n_sampe : int
            The number of samples to generate.
        output_path : str
            The file path where the generated dataset will be saved.
        """
        with open(output_path, "w") as output_file:
            for _ in range(n_sampe):
                training_datas = self.generate_training_data()
                for training_data in training_datas:
                    print(training_data.to_json(), file=output_file)


PromptAnswerGenerationType = TypeVar(
    "PromptAnswerGenerationType", bound=PromptAnswerGeneration
)


class PromptAnswerGenerator(DataGenerator[PromptAnswerGenerationType]):
    def _generate_training_data(
        self, seed: Optional[Any]
    ) -> Sequence[PromptAnswerGenerationType]:
        generation = self.new_generation()
        generation = self.generate_prompt(seed, generation)
        generations = self.generate_responses(generation)
        for generation in generations:
            generation.chat = ChatPromptValue(
                messages=[*generation.prompt.messages, generation.response]
            )
        return generations

    def generate_prompt(
        self, seed: Optional[SeedType], generation: PromptAnswerGenerationType
    ):
        prompt = self._generate_prompt(seed)
        generation = self.convert_prompt(prompt, generation)
        return generation

    def generate_responses(
        self, generation: PromptAnswerGenerationType
    ) -> Sequence[PromptAnswerGenerationType]:
        response = self._generate_response(generation)
        generations = self.convert_responses(response, generation)
        return generations

    def generate_response(
        self, generation: PromptAnswerGenerationType
    ) -> PromptAnswerGenerationType:
        response = self._generate_response(generation)
        generations = self.convert_response(response, generation)
        return generations

    def _generate_prompt(self, seed: Optional[Any]) -> str:
        raise NotImplementedError("Generate prompt is not implemented.")

    def _generate_response(
        self, generation: PromptAnswerGenerationType
    ) -> Union[str, BaseMessage, PromptAnswerGenerationType]:
        raw_response = self.call_llm(generation.prompt)
        return raw_response

    def convert_prompt(
        self, prompt: Union[str, VerifiedGeneration], generation
    ) -> PromptAnswerGenerationType:
        if isinstance(prompt, VerifiedGeneration):
            return prompt
        generation = deepcopy(generation)
        if isinstance(prompt, str):
            chat = ChatPromptValue(messages=[HumanMessage(content=prompt)])
            generation.prompt = chat
            return generation
        elif isinstance(prompt, BaseMessage):
            chat = ChatPromptValue(messages=[prompt])
            generation.prompt = chat
            return generation
        elif isinstance(prompt, ChatPromptValue):
            generation.prompt = prompt
            return generation

    def convert_responses(
        self,
        response: Union[
            str,
            BaseMessage,
            Sequence[str],
            Sequence[BaseMessage],
            PromptAnswerGenerationType,
            Sequence[PromptAnswerGenerationType],
        ],
        generation: Optional[PromptAnswerGenerationType] = None,
    ) -> Sequence[PromptAnswerGenerationType]:
        if generation is None:
            generation = self.data_generation_cls()
        if isinstance(response, str):
            generation.response = AIMessage(content=response)
            return [generation]
        elif isinstance(response, BaseMessage):
            generation.response = response
            return [generation]
        elif isinstance(response, self.data_generation_cls):
            return [response]
        elif isinstance(response, list):
            generations = []
            for r in response:
                content: str = ""
                if isinstance(r, str):
                    content = r
                elif isinstance(r, BaseMessage):
                    content = str(r.content)
                elif (
                    isinstance(r, Generation)
                    or isinstance(r, self.data_generation_cls)
                    or isinstance(r, VerifiedProgrammingGeneration)
                ):
                    generations.append(r)
                    continue
                else:
                    raise ValueError(f"Unknown type of response: {type(r)}")
                generation = deepcopy(generation)
                generation.response = AIMessage(content=content)
                generations.append(generation)
            return generations
        else:
            raise ValueError(f"Unknown type of response: {type(response)}")

    def convert_response(
        self,
        response: Union[
            str,
            BaseMessage,
            PromptAnswerGenerationType,
        ],
        generation: Optional[PromptAnswerGenerationType] = None,
    ) -> PromptAnswerGenerationType:
        return self.convert_responses(response, generation)[0]


VerifiedGenerationType = TypeVar("VerifiedGenerationType", bound=VerifiedGeneration)


class VerifiedGenerator(PromptAnswerGenerator[VerifiedGenerationType]):
    def __init__(
        self,
        verifiers: Optional[List[Verifier]] = None,
        seed_source: Optional[SeedSource] = None,
        verbose=False,
    ):
        super().__init__(seed_source=seed_source)
        self.llm: LLM = get_default_llm()
        if verifiers is None:
            verifiers = []
        self.verifiers: List[Verifier] = verifiers
        self.seed_source: Optional[SeedSource] = seed_source
        self.max_fixing = 5
        self.verbose = verbose
        self.data_generation_cls = VerifiedGeneration

    def add_verifier(self, verifier: Verifier):
        self.verifiers.append(verifier)

    def convert_verification(
        self,
        verification: Union[Verification, Tuple[bool, Optional[str]]],
        verifier_name: str,
        generation: VerifiedGenerationType,
    ):
        if isinstance(verification, Verification):
            verification.generation = deepcopy(generation)
            verification.verifier_name = verifier_name
            return verification
        elif isinstance(verification, tuple):
            is_verified, error_message = verification
            return Verification(
                verifier_name=verifier_name,
                generation=generation,
                is_verified=is_verified,
                error_message=error_message,
            )
        else:
            raise ValueError(f"Unknown type of verification: {type(verification)}")

    def build_prompt(self, *args, **kwargs) -> VerifiedProgrammingGeneration:
        raise NotImplementedError("Build prompt is not implemented.")

    def fix_response(
        self, generation: VerifiedGenerationType, verification: Verification
    ) -> Optional[VerifiedGenerationType]:
        fixed_response = self._fix_response(generation, verification)
        if fixed_response is None:
            return None
        generation = self.convert_responses(fixed_response, generation)[0]
        return generation

    def _fix_response(
        self, generation: VerifiedGenerationType, verification: Verification
    ) -> Optional[Union[VerifiedGenerationType, str]]:
        fixed_response = self.call_llm(
            f"""
I am trying toanswer the following prompt:
```
{generation.prompt}
```

So far, here is my answer:
```
{generation.response}
```

However, the following error was found:
```
{verification.error_message}
```

Can you fix my response so that it works? Surround your new fixed solution with triple backticks.
""",
            extract_triple_backtick=True,
        )
        return fixed_response

    def _generate_training_data(
        self, seed: Optional[Any]
    ) -> Sequence[VerifiedGenerationType]:
        generation = self.new_generation()
        generation = self.generate_prompt(seed, generation)
        return self.generate_verified_responses(generation)

    def generate_verified_responses(
        self,
        generation: VerifiedGenerationType,
    ) -> Sequence[VerifiedGenerationType]:
        raw_reponse = self.generate_responses(generation)
        generations = self.convert_responses(raw_reponse, generation)
        verified_generations = self.verify_and_fix_responses(generations)
        return verified_generations

    def verify_and_fix_responses(
        self,
        generations: Sequence[VerifiedGenerationType],
        do_not_fix: bool = False,
    ) -> List[VerifiedGenerationType]:
        generations = self.convert_responses(generations)
        output_generations = []
        for generation in generations:
            prompt = generation.prompt
            response = generation.response
            if response is None:
                continue
            if len(self.verifiers) == 0:
                generation.verification_success = True
                generation.is_correct = True
                generation.failed_verifier_name = None
                generation.failed_verifier_error_message = None
                output_generations.insert(0, deepcopy(generation))
                continue
            verification = None
            for _ in range(self.max_fixing):
                for verifier in self.verifiers:
                    if self.verbose:
                        print(verifier.__class__.__name__)
                    verification = verifier.check(generation)
                    verification = self.convert_verification(
                        verification, verifier.__class__.__name__, generation
                    )
                    generation.verification_success = verification.is_verified
                    if not verification.is_verified:
                        if self.verbose:
                            print(f"Wrong response:\n{generation.response.content}")
                            print(f"Error: {verification.error_message}")
                        generation.failed_verifier_name = verifier.__class__.__name__
                        generation.failed_verifier_error_message = (
                            verification.error_message
                        )
                        # bad_generations.append(generation)
                        if do_not_fix:
                            break
                        fixed_response = self.fix_response(
                            generation,
                            verification=verification,
                        )
                        if fixed_response is None:
                            did_fixing_fail = True
                            break
                        generation = self.convert_responses(fixed_response, generation)[
                            0
                        ]
                        verification.fixed_generation = generation
                        break
                if verification is not None and verification.is_verified:
                    verification.generation.verification_success = True
                    verification.generation.is_correct = True
                    verification.generation.failed_verifier_name = None
                    verification.generation.failed_verifier_error_message = None
                    output_generations.insert(0, deepcopy(verification.generation))
                    break
                else:
                    output_generations.append(deepcopy(generation))
                if (
                    verification is not None
                    and verification.is_verified is False
                    and verification.fixed_generation is None
                ):
                    break
                # prev_generation.verification_success = False
                # prev_generation.is_correct = False

                # output_generations.append(deepcopy(prev_generation))
        return output_generations


class PythonVerifiedFasterCodeGenerator(VerifiedGenerator):
    def __init__(self):
        verifiers: List[Verifier] = [
            NonEmptyVerifier(),
            PythonSyntaxVerifier(),
            # MypyVerifier(),
            PythonGeneratedTestVerifier(),
            PythonGeneratedTestCoverageVerifier(),
        ]
        # verifiers = [PythonGeneratedTestVerifier()]
        super().__init__(
            verifiers=verifiers,
        )
        self.language = "python"

    def generate_prompt(self, seed: Optional[SeedType] = None):
        if seed is None:
            seed = self.get_next_seed()
        pass

    def build_prompt(self, slow_function: str) -> ChatPromptValue:
        prompt_str = f"""
Here is a python function.
```{self.language}
{slow_function}
```

This function is slow. Rewrite it to make it faster.
The new function should be functionally equivalent to the original function.
""".strip()
        code = extract_source_code(slow_function)
        if code is None:
            raise ValueError("Could not extract source code from slow function.")
        signature = extract_python_function_signature(code)
        prompt = ChatPromptValue(
            messages=[
                HumanMessage(
                    content=prompt_str,
                    additional_kwargs={
                        "slow_function": slow_function,
                        "signature": signature,
                    },
                )
            ]
        )
        return prompt


#     def prompt_for_fixing(self, prompt: ChatPromptValue) -> str:
#         prompt_str = f"""
# We are trying to speeed up the following function:
# ```{self.language}
# """


class DefensiveProgrammingTestGenerator(VerifiedGenerator):
    def __init__(self, programming_language):
        self.programming_language = programming_language
        verifiers = [
            NonEmptyVerifier(),
            ProgrammingGeneratedTestVerifier(
                programming_language=self.programming_language
            ),
        ]
        super().__init__(
            verifiers=verifiers,
        )

    def build_prompt(
        self, problem: str, solution: str
    ) -> Tuple[ChatPromptValue, Dict[str, Any]]:
        prompt, data = build_prompt_for_defensive_programming_test(
            problem=problem,
            solution=solution,
            programming_language=self.programming_language,
        )
        return ChatPromptValue(messages=[HumanMessage(content=prompt)]), data


def verify_and_fix_programming_response(
    prompt: str, response: str, data: Optional[Dict] = None, verifiers=[], verbose=False
) -> Tuple[List[VerifiedGeneration], List[VerifiedGeneration]]:
    class CustomProgrammingGenerator(VerifiedGenerator):
        def __init__(self):
            super().__init__(verifiers, seed_source=None, verbose=verbose)

    generator = CustomProgrammingGenerator()
    # generation = VerifiedGeneration(prompt=None, response=response, data=data)
    return generator.verify_and_fix_responses(
        ChatPromptValue(messages=[HumanMessage(content=prompt)]),
        [AIMessage(content=response)],
        data,
    )
