
# Synth_gen

```
The internet is the fossil fuel of AI. Synthetic data will be its renewable energy.
```

Synth_gen is an open-source library designed for LLMs to learn without human supervision through self-play, execution feedback, and more.

Its primary goal is to generate synthetic programming problems and solutions and to verify these solutions using linters, parsers, and generated test executions. This method was used during the training of Llama3, Phi4, Qwen2.5, and probably more.

More generally, we want to aggregate many recipes to generate synthetic data in diverse ways. We wrote a few possible recipes here to give a taste of what is possible.

Synth_gen is designed to be modular. We encourage the community to contribute by adding more synthetic data generators and response verifiers.

This library is LLM-agnostic and can be used with any model implemented in langchain.

## Installation

### Install from source

```bash
git clone https://github.com/facebookresearch/synth_gen.git
cd synth_gen
pip install -e .
```

## How does it work?

Synth_gen can be used in two modes:
- Generating verified responses at inference time.
- Generating verified training data that can be used to fine-tune an existing model.

### Generating verified responses

In this mode, given a prompt, we will first normally generate a response. Then, we will pass the response through a series of verifiers.
For example, if the response contains code, a verifier could just be a linter (more examples later). If the response does not pass the verifier, the verifier will generate an error message. Then, we will ask the model to fix the problem. Overall, we will loop until the response passes all the verifiers.

The final response will have a higher chance of being correct since it passes through all verifiers.

Synth_Gen intends to provide many verifiers and is modular to allow the community to bring in more verifiers.

### Generating verified training data

In this mode, we will try to generate an entire dialogue. The initial prompt does not need to be provided. Instead, it will be generated.

First, we cannot just ask the model to generate a prompt in the same manner each time; otherwise, the model will output very similar prompts.
That's why we use seeds. The source of a seed will output some text or formatted data that can be used as a seed or starting point for prompt generation and allows us to generate diverse prompts on the long tail distribution.

Given a seed, we ask the model to generate a prompt.
Then, given a prompt, we generate a verified response as explained earlier.
The prompt and the response together can be used as a grounded training sample to train an LLM.

### Generate verified python problems and solutions

You can use Synth_gen to generate Python problems based on a provided seed, then generate solutions, and verify that the solution is correct using a parser, linter, and execution on generated unit tests. The LLM tries to fix incorrect solutions.

``` python
from synth_gen.generators import PythonVerifiedSynthDataGenerator

generator = PythonVerifiedSynthDataGenerator()
generator.verbose = True
generations = generator.generate_training_data()
print(generations[0])
```

### Choose a seed source

```python
from synth_gen.generators import PythonVerifiedSynthDataGenerator
generator.seed_source = SmallPythonSeedSource()
```

### Choose verifiers

Verifiers will check whether the response is correct (checker), provide feedback about what needs to be fixed (error message), and finally correct the response (fixer).

```python
from synth_gen.verification.python_verifiers import (
    MypyVerifier,
    PythonSyntaxVerifier,
    PythonTestBasedVerifier,
)
generator.verifiers = [PythonSyntaxVerifier, MypyVerifier, PythonTestBasedVerifier]
```

### Make your own source of seed

A seed source provides text or formatted data that can be used to generate the prompt part of the training data sample. They can be used as inspiration or provide data. The goal is to use them to generate diverse prompts on the long tail.

To make your own source of seed, you only have to implement the `get_next_seed` function.

```python
from synth_gen.generation.seed_source import SeedSource
class MySeedSource(SeedSource):
    def get_next_seed(self) -> Union[str, Dict[str, Any]]:
        # your code here
```
There is a helper to create seed sources from Hugging Face databases.

```python
from synth_gen.generation.seed_source import HuggingFaceSeedSource, extract_snippet
class SmallPythonSeedSource(HuggingFaceSeedSource):
    def __init__(
        self, hugging_face_dataset_name="calum/the-stack-smol-python-docstrings"
    ):
        super().__init__(
            hugging_face_dataset_name,
            field_name="code",
            post_processing=extract_snippet,
            filter=None,
        )
```

### Make your own generator

You can have full control and implement the `_generate_training_data` function to create your own generator.

```python
from synth_gen.generators import DataGeneration, DataGenerator
from synth_gen.schema import AIMessage, ChatPromptValue, HumanMessage

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
```

### Make your own verified generator

```python
from synth_gen.generators import VerifiedGeneration, VerifiedGenerator
from synth_gen.verification.schema import Verification

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
```

### Make your own verifier

```python
class MyVerifier(Verifier):
    def check(
        self,
        generation: VerifiedGeneration,
    ) -> Tuple[bool, Optional[str]]:
        return False, "my error message"
```

## Diverse examples of synthetic data generators


### Q&A about a text

Reformulating a text in Q&A has been shown to improve question-answering capabilities of LLMs.

```python
from synth_gen.generators import QAGenerator

wikipedia_article = """
Albert Einstein was a German-born theoretical physicist who is best known for developing the theory of relativity."""

generator = QAGenerator()

generator.verbose = True
generations = generator.generate_training_data(wikipedia_article)
```

```
QUESTION:
Who was Albert Einstein?
ANSWER:
A German-born theoretical physicist
QUESTION:
What is Albert Einstein best known for?
ANSWER:
Developing the theory of relativity.
```

### Generate property-based tests

Given a coding problem and a solution, we want to generate a property-based test for this solution.

It should pass two main verifiers:
- test completion verifier: the solution should pass the test.
- test coverage verifier: the test should cover 100% of the solution.

You can see a full example there: `examples/generate_verified_python_tests.py`

```python
from synth_gen import PythonVerifiedTestGenerator
generator = PythonVerifiedTestGenerator()
prompt, data = generator.build_prompt(
    problem="Implement the Fibonnacci algorithm.",
    solution="""
def fibo(n):
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
generations = generator.generate_verified_responses(prompt, data)
print(generations[0].response.content)
```

### Generate hallucination mitigation data

#### Generator of questions about unknown facts

```python
from synth_gen.generators import UnkownAnswerGenerator

wikipedia_article = """
Albert Einstein was a German-born theoretical physicist who is best known for developing the theory of relativity."""

generator = UnkownAnswerGenerator()

generator.verbose = True
generations = generator.generate_training_data(wikipedia_article)
```

```
QUESTION:
Did Einstein have any dreams or nightmares that influenced his development of the theory of relativity?
ANSWER:
While Albert Einstein's personal life and thought processes have been extensively studied, there is limited information available about his dreams or nightmares and their potential influence on his development of the theory of relativity.

Unfortunately, Einstein did not leave behind a detailed account of his dreams or nightmares, and it is unlikely that we will ever know for certain whether any specific dreams or nightmares played a role in shaping his theory.

[...]

In summary, while Einstein's dreams and nightmares may have played some role in his creative process, there is no concrete evidence to support this claim, and it remains a topic of speculation.
```

#### Generator of questions based on false promise

```python
from synth_gen.generators import FalsePromiseGenerator

wikipedia_article = """
Albert Einstein was a German-born theoretical physicist who is best known for developing the theory of relativity."""

generator = FalsePromiseGenerator()
generator.verbose = True
generations = generator.generate_training_data(wikipedia_article)
```

```
FACT:
Albert Einstein was a German-born theoretical physicist.
FALSE PROMISE:
Albert Einstein was a French-born theoretical biologist.
QUESTION:
What was the name of the French university where Albert Einstein earned his degree in biology before going on to develop his groundbreaking theories on the evolution of species?
ANSWER:
I'm happy to help, but I must correct a false assumption in the question. Albert Einstein was not a French-born theoretical biologist, but rather a German-born theoretical physicist. He is widely recognized for his groundbreaking work in physics.
[...]
I would say that there is no French university where Einstein earned a degree in biology, as his academic pursuits were focused on physics, not biology.
```

### Generate simple calculations

```python
from synth_gen.generators import SimpleMathTimesGenerator

generator = SimpleMathTimesGenerator()

for t in range(1000000):
    generations = generator.generate_training_data()
    print("Human: ", generations[0].chat.messages[0].content)
    print("AI: ", generations[0].chat.messages[1].content)

# Example output:
# Human: -323*421
# AI:    -135983
```

### Generate tool calls

You can generate tool calls and verify that the call matches the json schema of the tool parameters.

```
from synth_gen.generation.json_generator import ToolCallGenerator

generator = ToolCallGenerator()
generation = generator.generate_training_data()
```

```
TOOL_NAME: calculator
TOOL_DESCRIPTION: A tool for performing mathematical calculations.
TOOL_PARAMETERS: {'type': 'object', 'properties': {'formula': {'type': 'string', 'description': 'The mathematical formula to be evaluated.'}}}
PROMPT: What is the sum of the numerical value of the alphabetical position of the letter 'a' multiplied by 2, and 5?
GENERATED_TOOL_CALL: {"expression":"(1 * 2) + 5", "result_type":"number"}
```

## Execute python code in a container

Synth_gen provides tools to execute code safely and efficiently. All executions are run within containers.

You can see more examplate there: examples/execute_python

Self-study has the following capabilities:
- Execute commands within a one-time or persitent container.

- Execute python code within a one-time or persitent python shell.
```python
from synth_gen.execution import run_python, run_cpp
stdout, stderr, return_code = run_python("""print("Hello")""")
# -> "Hello", "", 0
stdout, stderr, return_code = run_cpp("""
#include <iostream>
int main(){std::cout << "World" << std::endl;}""")
# -> "World", "", 0
```
- Automatically install python dependencies.
- Re-use pre-installed python dependencies to speed up execution.

## Execute code in a Jupyter notebook

Our jupyter notebooks are containerized, persistent and support bash and python programmaing languages (30 more programming languages are coming soon).

```python
from synth_gen.execution.code_execution import PersistentJupyterContainer

container = PersistentJupyterContainer(kernel_name="bash")
container.run_command("export A=123")
result, error_message, is_failure = container.run_command("echo $A")
# -> ("123", "", False)
```

## Road map

- Support all jupyter kernels for ~30 different programming languages
- Generator to guess output of a piece of code
- Generator of docker files
- Generators of faster code

## License

Synth_gen is MIT licensed, as found in the LICENSE file.

## Citation

```bibtex
@software{duchenne_synth_gen_2025,
  author       = {Olivier Duchenne},
  title        = {Synth_gen},
  year         = {2025},
  howpublished = {\url{https://github.com/facebookresearch/synth_gen}},
  note         = {Affiliation: Meta, FAIR}
}
```
