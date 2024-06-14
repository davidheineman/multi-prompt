import os
from . import get_resource_paths

PROMPT_DIR, DATA_DIR = get_resource_paths('code')

HUMAN_EVAL_PATH = os.path.join(DATA_DIR, 'HumanEval.jsonl')

HUMAN_PROMPT_PATH = os.path.join(PROMPT_DIR, 'human.json')
GPT_PROMPT_PATH = os.path.join(PROMPT_DIR, 'gpt-4.json')