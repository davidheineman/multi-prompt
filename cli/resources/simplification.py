import os
from . import get_resource_paths

PROMPT_DIR, DATA_DIR = get_resource_paths('simplification')

SIMPEVAL_PATH = os.path.join(DATA_DIR, 'lens-data', 'simpeval_2022.csv')
ASSET_PATH = os.path.join(DATA_DIR, 'asset-data')

HUMAN_PROMPT_PATH = os.path.join(PROMPT_DIR, 'human.json')
GPT_PROMPT_PATH = os.path.join(PROMPT_DIR, 'gpt-4.json')
GPT_1000_PROMPT_PATH = os.path.join(PROMPT_DIR, 'gpt-4.json')
GPT_WEIGHTED_PROMPT_PATH = os.path.join(PROMPT_DIR, 'weighted', 'gpt-4.json')
GPT_1000_WEIGHTED_PROMPT_PATH = os.path.join(PROMPT_DIR, 'weighted', 'gpt-4-1000.json')
HUMAN_WEIGHTED_PROMPT_PATH = os.path.join(PROMPT_DIR, 'weighted', 'human.json')