# Copyright (c) Meta Platforms, Inc. and affiliates.

from src.tasks.human_eval import extract_docstring
from .filter import clean_comment, remove_print

def rindex(lst, value):
    return len(lst) - lst[::-1].index(value) - 1


def postprocess_func_only(code, tokens):
    lines = []
    for line in code.split("\n"):
        if len(line.strip()) > 0 and not line.startswith(" "):
            continue
        else:
            lines.append(line)

    code = "\n".join(lines)
    code = code.rstrip()

    curr = ""
    for i, tok in enumerate(tokens):
        curr += tok
        if len(curr) >= len(code):
            break

    return code, tokens[: i + 1]


def make_new_context(prompt, generation, func_name, canonicalize=False, clean_print=False):
    if canonicalize:
        try:
            generation = clean_comment(generation)
        except:
            # static error
            generation = generation
    if clean_print:
        generation = remove_print(generation)
    docstring, func_header, func_context, doc_start = extract_docstring(prompt)
    if canonicalize:
        func_header = func_header.replace(f"{func_name}(", "f(")
        docstring = docstring.replace(f"{func_name}(", "f(")
        generation = generation.replace(f"{func_name}(", "f(")
    reverse_prompt = "\n\n# write the docstring for the above function\n"
    without_ref = (
        func_context
        + "\n"
        + func_header.strip()
        + "\n"
        + generation
        + reverse_prompt
        + func_header.strip()
        + "\n"
        + f"    {doc_start}"
    )
    with_ref = without_ref + docstring.strip()[3:]
    return with_ref.rstrip(), without_ref


def rindex(lst, value):
    return len(lst) - lst[::-1].index(value) - 1


def find_start(tokens):
    tokens = tokens[:-2]  # remove last docstring marker
    for marker in [' """', " '''", ' ""', "''"]:
        if marker in tokens:
            return rindex(tokens[:-1], marker) + 1
    raise ValueError("not found")
