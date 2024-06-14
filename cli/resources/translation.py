import os
from . import get_resource_paths


PROMPT_DIR, DATA_DIR = get_resource_paths('translation')

NEWSTEST19_PATH = os.path.join(DATA_DIR, 'newstest19')
ALMA_TEST_PATH = os.path.join(DATA_DIR, 'alma')
NTREX_PATH = os.path.join(DATA_DIR, 'ntrex')

HUMAN_PROMPT_PATH = os.path.join(PROMPT_DIR, 'human.json')
GPT_PROMPT_PATH = os.path.join(PROMPT_DIR, 'gpt-4.json')
GPT_1000_PROMPT_PATH = os.path.join(PROMPT_DIR, 'gpt-4-1000.json')