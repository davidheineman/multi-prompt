# Copyright (c) Meta Platforms, Inc. and affiliates.
# Taken from: https://github.com/facebookresearch/coder_reviewer_reranking

import keyword, re
from pyminifier import analyze
from pyminifier.minification import remove_comments_and_docstrings, remove_blank_lines

RESERVED_WORDS = keyword.kwlist + analyze.builtins


def filter_empty(code, remove_function_header=False):
    if remove_function_header:
        code = "\n".join(
            [l for l in code.split("\n") if not l.strip().startswith("def")]
        )
    try:
        code = clean_comment(code)
    except:
        code = ""
    return code.strip() not in ["", "pass", "return"]


def filter_repeat(x, threshold=0.25):
    import zlib

    bytes_x = bytes(x, encoding="utf-8")
    comp_x = zlib.compress(bytes_x)
    return len(comp_x) / len(bytes_x) > threshold


def reject_degenerate(pred):
    return filter_empty(pred, remove_function_header=False) and filter_repeat(pred)


def is_degenerate(pred):
    return not reject_degenerate(pred)


def clean_comment(code):
    code = remove_comments_and_docstrings(code)
    code = remove_blank_lines(code)
    return code


def remove_print(code):
    code = re.sub("print(.+)", "print('')", code)
    code = re.sub("Error(.+)", "Error('')", code)
    code = re.sub("Exception(.+)", "Exception('')", code)
    code = re.sub("assert (.+), +['\"].+['\"]", "assert \\1", code)
    return code


def clean_code(code):
    code = clean_comment(code)
    code = remove_print(code)
    return code