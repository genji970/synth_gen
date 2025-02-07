# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import ast
import re

from typing import Any, List, Optional


def extract_python_function_signature(raw: str) -> Optional[str]:
    """
    Extracts the first python function signature from the provided raw string output of an LLM.

    If found, it returns the function signature; otherwise, it returns None.

    Args:
        raw (str): The raw string output from an LLM which potentially contains function definitions.

    Returns:
        Optional[str]: The extracted function signature if found, otherwise None.
        Example:
    """

    # class FunctionSignatureExtractor(ast.NodeVisitor):
    #     def __init__(self):
    #         self.signature = None

    #     def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
    #         start_lineno = node.lineno
    #         start_col_offset = node.col_offset
    #         end_lineno = (
    #             node.returns.end_lineno if node.returns else node.args.args[-1].end_lineno
    #         )
    #         end_col_offset = (
    #             node.returns.end_col_offset
    #             if node.returns
    #             else node.args.args[-1].end_col_offset
    #         )
    #         self.signature = extract_code_segment(
    #             raw, start_lineno, start_col_offset, end_lineno, end_col_offset
    #         )

    #         # Stop after the first function definition
    #         return

    #     def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
    #         start_lineno = node.lineno
    #         start_col_offset = node.col_offset
    #         end_lineno = (
    #             node.returns.end_lineno if node.returns else node.args.args[-1].end_lineno
    #         )
    #         end_col_offset = (
    #             node.returns.end_col_offset
    #             if node.returns
    #             else node.args.args[-1].end_col_offset
    #         )
    #         self.signature = extract_code_segment(
    #             raw, start_lineno, start_col_offset, end_lineno, end_col_offset
    #         )

    #         # Stop after the first function definition
    #         return

    # try:
    #     tree = ast.parse(raw)
    #     extractor = FunctionSignatureExtractor()
    #     extractor.visit(tree)
    #     return extractor.signature
    # except SyntaxError as e:
    #     print(e)
    #     return None

    raw = re.sub(r"^\d+\.\s+", "", raw)
    function_call_match = re.search(
        r"^\s*((?:async\s+)?def \w+\(.*?\).*:)", raw, re.MULTILINE
    )
    if function_call_match:
        signature_code = function_call_match.group(1)[:-1]
        return signature_code
    else:
        return None


def extract_code_segment(
    code, start_lineno, start_col_offset, end_lineno, end_col_offset
):
    """
    Extracts the code between two line numbers and column offsets.
    Args:
        code (str): The source code.
        start_lineno (int): The starting line number.
        start_col_offset (int): The starting column offset.
        end_lineno (int): The ending line number.
        end_col_offset (int): The ending column offset.
    Returns:
        str: The extracted code.
    """
    # Get the source code lines
    lines = code.splitlines()
    # Initialize the extracted code
    extracted_code = ""
    # Iterate over the lines
    for i, line in enumerate(lines, start=1):
        if start_lineno <= i <= end_lineno:
            # If the line is within the range, extract the relevant part
            if i == start_lineno and i == end_lineno:
                extracted_code += line[start_col_offset:end_col_offset]
            elif i == start_lineno:
                # For the first line, extract from the start column offset
                extracted_code += line[start_col_offset:] + "\n"
            elif i == end_lineno:
                # For the last line, extract up to the end column offset
                extracted_code += line[:end_col_offset]
            else:
                # For other lines, extract the entire line
                extracted_code += line + "\n"
        elif i > end_lineno:
            break
    return extracted_code


def extract_source_code(raw: str) -> Optional[str]:
    """
    Extracts the first block of code enclosed in triple backticks from the provided raw string output of an LLM.

    If a block of code is found within triple backticks, it returns the first such block. If no such block is found,
    it is assumed that the entire raw is already code, and it returns the entire raw string.

    Args:
        raw (str): The raw string output from an LLM which potentially contains blocks of code.

    Returns:
        str: The extracted block of code or the entire raw string stripped of whitespace.
    """

    if type(raw) is not str:
        return None
    sources = re.findall(r"```\w*\n(.*?)```", raw, re.DOTALL)
    if len(sources) >= 1:
        source = sources[0]
    else:
        source = raw.strip()
    return source


def extract_multiple_source_codes(raw: str) -> List[str]:
    """
    Extracts all blocks of code enclosed in triple backticks from the provided raw string output of an LLM.

    Returns a list of all such blocks. If no blocks are found, it returns an empty list.

    Args:
        raw (str): The raw string output from an LLM which potentially contains multiple blocks of code.

    Returns:
        List[str]: A list containing all extracted blocks of code.
    """
    if type(raw) is not str:
        return []
    sources = re.findall(r"```\w*\n(.*?)```", raw, re.DOTALL)
    for source_index, source in enumerate(sources):
        sources[source_index] = source.strip()
    return sources


def extract_triple_backticked_string(raw: str) -> Optional[str]:
    """
    Extracts the first block of code enclosed in triple backticks from the provided raw string output of an LLM.

    If a block of code is found within triple backticks, it returns the first such block. If no such block is found,
    it is assumed that the entire raw is already a quote, and it returns the entire raw string.

    Args:
        raw (str): The raw string output from an LLM which potentially contains blocks of code.

    Returns:
        Optional[str]: The extracted block of code or None if no such block is found.
    """
    if type(raw) is not str:
        return None
    sources = re.findall(r"```(.*?)```", raw, re.DOTALL)
    if len(sources) >= 1:
        source = sources[-1].strip()
    else:
        source = None
    return source


def extract_titled_list(raw: str, tag) -> List[str]:
    tag_regex = rf"(^\s*[\*]*\[{tag}.*\][\*\)]*\s*)"
    tagged_parts = re.split(tag_regex, raw, flags=re.MULTILINE)
    if len(tagged_parts) > 1:
        tagged_parts = [tagged_parts[0]] + [
            x + y for x, y in zip(tagged_parts[1::2], tagged_parts[2::2])
        ]
    elif len(tagged_parts) == 1:
        tagged_parts = [tagged_parts[0]]
    else:
        tagged_parts = []
    tagged_parts = [x.strip() for x in tagged_parts if x.strip() != ""]
    return tagged_parts


def extract_list_from_string(raw: str) -> List[str]:
    """
    Extracts a list of items from the provided raw string output of an LLM.

    Args:
        raw (str): The raw string output from an LLM which potentially contains a list of items.

    Returns:
        List[str]: A list containing all extracted items.
    """
    if type(raw) is not str:
        return []
    items = re.findall(r"\s*-\s(.*)", raw, re.MULTILINE)
    return items
