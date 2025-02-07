# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from langchain.schema import BaseMessage
from synth_gen.generation.generation import DataGeneration, PromptAnswerGeneration
from synth_gen.generation.LLM import call_llm, default_llm, LLM
from synth_gen.generation.parsing import extract_source_code

from synth_gen.schema import ChatPromptValue


@dataclass
class VerifiedGeneration(PromptAnswerGeneration):
    """
    A class to represent the result of a generation process with verification.

    Attributes:
    ----------
    prompt : Optional[ChatPromptValue]
        The prompt used for generation.
    response : Optional[BaseMessage]
        The response generated from the prompt.
    verification_success : Optional[bool]
        Indicates if the response has sucessfully been verified. None, if the verification has not been performed.
    data : Dict[str, Any]
        Additional formatted data related which can be used by the verifiers, and depends on the application.
    failed_verifier_name : Optional[str]
        The name of the verifier on which the response failed.
    failed_verifier_error_message : Optional[str]
        The error message from the failed verifier.
    """

    prompt: Optional[ChatPromptValue] = None
    response: Optional[BaseMessage] = None
    verification_success: Optional[bool] = None
    data: Dict[str, Any] = field(default_factory=dict)
    failed_verifier_name: Optional[str] = None
    failed_verifier_error_message: Optional[str] = None
    verifications: Optional[Sequence["Verification"]] = None

    @property
    def is_correct(self):
        return self.verification_success

    @property
    def chat(self):
        return ChatPromptValue(messages=[*self.prompt.messages, self.response])

    @chat.setter
    def chat(self, value: ChatPromptValue):
        pass

    @is_correct.setter
    def is_correct(self, value: bool):
        pass


class VerificationStatus(Enum):
    verified = "verfied"
    fixed = "fixed"
    unfixable = "unfixable"
    not_fixed = "not_fixed"


@dataclass
class Verification:
    generation: VerifiedGeneration
    verifier_name: str
    is_verified: Optional[bool] = None
    is_unfixable: Optional[bool] = False
    error_message: Optional[str] = None
    fixed_generation: Optional[VerifiedGeneration] = None


class Verifier:
    verbose: bool = False

    def check_and_fix(
        self,
        generation,
        verbose: Optional[bool] = None,
        llm: LLM = default_llm,
        do_not_fix=False,
    ) -> Verification:
        if verbose is not None:
            self.verbose = verbose
        if self.verbose:
            print(self.__class__.__name__)
        verification = Verification(
            generation=generation, verifier_name=self.__class__.__name__
        )
        is_correct, error_message = self.check(generation)
        verification.is_verified = is_correct
        if not is_correct:
            if self.verbose:
                print(f"Wrong response:\n{generation.response.content}")
                print(f"Error: {error_message}")
            verification.error_message = error_message
            # bad_generations.append(generation)
            if do_not_fix:
                return verification
            fixed_response = self.fix(
                generation,
                error_message=str(error_message),
                llm=llm,
            )
            if fixed_response is None:
                did_fixing_fail = True

                return verification
            fixed_generation = deepcopy(generation)
            fixed_generation.response = fixed_response
            verification.fixed_generation = fixed_generation
            return verification
        else:
            return verification

    def check(
        self,
        generation: VerifiedGeneration,
    ) -> Union[Tuple[bool, Optional[str]], Verification]:
        raise NotImplementedError("Verifier check is not implemented.")

    def fix(
        self,
        generation: VerifiedGeneration,
        error_message: str,
        llm: LLM = default_llm,
    ) -> Union[str, VerifiedGeneration]:
        if generation.prompt is None:
            raise ValueError("No prompt provided.")
        response_str = self.message_to_string(generation.response)
        prompt_str = self.message_to_code(generation.prompt)
        fix_prompt = f"""
I am trying to respond to the following prompt:
```
{prompt_str}
```

For now, I have the following response:
```
{response_str}
```

Unfortunately, it is not working. I encoutered the following error:
```
{error_message}
```

Can you fix my response so that it works? Surround your new fixed solution with triple backticks.


"""
        response = call_llm(
            fix_prompt, llm, use_cache=True, extract_triple_backtick=True
        )
        return str(response)

    def message_to_string(
        self, messages: Optional[Union[BaseMessage, ChatPromptValue]]
    ) -> str:
        if messages is None:
            return ""
        elif isinstance(messages, BaseMessage):
            return str(messages.content)
        elif isinstance(messages, ChatPromptValue):
            if len(messages.messages) == 1:
                return str(messages.messages[0].content)
            else:
                return str(messages)
        else:
            raise ValueError(f"Unkown messages type: {type(messages)}, {messages}")

    def message_to_code(
        self, messages: Union[BaseMessage, ChatPromptValue]
    ) -> Optional[str]:
        message_str = self.message_to_string(messages)
        code = extract_source_code(message_str)
        return code


class VerifiedProgrammingGeneration(VerifiedGeneration):
    programming_language: str = "python"
    code: Optional[str] = None
    test_code: Optional[str] = None
    problem: Optional[str] = None
    signature: Optional[str] = None
