import os

from utils.constants import ROOT_DIR

def get_resource_paths(task):
    PROMPT_DIR = os.path.join(ROOT_DIR, 'prompts', task)
    DATA_DIR = os.path.join(ROOT_DIR, 'data', task)
    return PROMPT_DIR, DATA_DIR