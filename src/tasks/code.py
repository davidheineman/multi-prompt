from src.metrics.human_eval.execution import check_correctness

COMPILE_ERRORS = ['NameError', 'UnboundLocalError']

# Taken from https://github.com/openai/simple-evals/blob/main/humaneval_eval.py
GPT_4_INSTRUCTION = "Read the following function signature and docstring, and fully implement the function described. Your response should ONLY contain the code for this function. Do NOT print the provided docstring or function header or ANYTHING but the code.\n"

EXAMPLE_PREFIXES = [
    ">>>",
    "Example:",
    "For example:",
    "For Example:",
    "Examples",
    "Examples:",
    "Example",
    "for example:",
    "Example 1:",
    "It must be implemented like this:",
    "[input/output] samples:",
    "Example :",
    "example:",
    "Note:",
]

DOCSTRING_FORMATS = [
    '"""',
    "'''"
]

# These examples use the function itself to denote examples (e.g., next_smallest([1, 2, 3, 4, 5]) == 2)
EXAMPLE_NAME_MARKERS = [
    'next_smallest',            # 90
    'is_nested',                # 132
    'fix_spaces',               # 140
    'simplify',                 # 144
    'double_the_difference',    # 151
    'cycpattern_check',         # 154
    'find_max',                 # 158
]


def _get_instruction(source):
    # Find beginning of docstring. E.g, ''' or """
    parsed_ds = None
    for ds in DOCSTRING_FORMATS:
        if ds in source:
            parsed_ds = source.split(ds)[1]
    if parsed_ds == None: raise ValueError(f'Could not parse source: {source}.')

    # Find beginning of exampes. E.g., >>> or "For Example:"
    parsed_ep = None
    for ep in EXAMPLE_PREFIXES:
        if ep in parsed_ds:
            parsed_ep = parsed_ds.split(ep)[0]
            break
    # if parsed_ep == None: raise ValueError(f'Could not parse source: {source}.')
    if parsed_ep == None: parsed_ep = parsed_ds

    # Get exampe marker exceptions
    parsed_mk = None
    for mk in EXAMPLE_NAME_MARKERS:
        if mk in parsed_ds:
            parsed_mk = parsed_ds.split(f'    {mk}')[0] + '    '
            break
    if parsed_mk is not None: parsed_ep = parsed_mk
    
    return parsed_ep


def parse_human_eval_instruction(source):
    parsed = _get_instruction(source)
    parsed = parsed.replace('\n', '').lstrip('\n ').replace('    ', '')
    return parsed


def prepare_human_eval_template(source):
    prefix = None
    if source.count('def ') > 1:
        prefix, source = source[:source.rfind('def ')], source[source.rfind('def '):]

    parsed = _get_instruction(source)
    template = source.replace(parsed, '\n    {instructions}\n    ')

    if prefix is not None:
        template = prefix + template

    return template


def truncate_multiline(text):
    """
    Truncates any string at the first non-indented line. Helpful heuristic for preventing
    tests, comments or additional functions after code
    """
    out = []

    # If the model responds in Markdown, remove the scaffolding
    if text.startswith("```python\n"): text = text[len("```python\n"):]
    if text.endswith("\n```"): text = text[:-len("\n```")]

    # (Newly added for GPT-4) Find the first indented line of code
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("  "):
            break
    text = "\n".join(lines[i:])

    # Deletes all text after first non-indented line of code
    for l in text.splitlines():
        if l.startswith("  ") or l.isspace() or len(l.strip()) == 0: out += [l]
        else: break

    # Delete trailing spaces or newlines
    i = len(out) - 1
    while i >= 0 and (out[i] == "" or out[i].isspace()):
        out.pop()
        i -= 1

    # Fix indentation on first line
    out = "\n".join(out)
    out = "    " + out.lstrip(" ")
    
    return out


def check_compiles(sent, timeout=1):
    problem = {
        'task_id': None,
        'prompt': sent['source'],
        'test': sent['metadata']['test'],
        'entry_point': sent['metadata']['entry_point'],
    }
    output = check_correctness(problem, sent['target'], timeout)
    has_error = any([o in output['result'] for o in COMPILE_ERRORS])
    return not has_error

