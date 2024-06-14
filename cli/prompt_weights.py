import os, json
from collections import Counter

from utils.constants import RESULTS_DIR

from src.tasks.translation import get_translate_labels
from src.tasks.code import parse_human_eval_instruction
from src.tasks.simplification import SIMPLIFICATION_INSTRUCTION_SUFFIX
from src.util import get_time


def get_prompt_dist(results, instruction_suffix=None, parse_method=None):
    """
    For a list of generations, return the prompt distribution.
    """
    # Get the prompt of the selected candidate
    selected_prompts, all_prompts = [], []
    for sent in results:
        cand = max(sent['candidates'], key=lambda x: x['score'])
        selected_prompts += [cand['prompt']]
        all_prompts += [cand['prompt'] for cand in sent['candidates']]

    # Extract the prompt
    if instruction_suffix is not None:
        selected_prompts = [instruction_suffix.join(i.split(instruction_suffix)[:-1]) for i in selected_prompts]
        all_prompts = [instruction_suffix.join(i.split(instruction_suffix)[:-1]) for i in all_prompts]
    elif parse_method == 'code':
        selected_prompts = [parse_human_eval_instruction(i) for i in selected_prompts]
        all_prompts = [parse_human_eval_instruction(i) for i in all_prompts]
    else:
        raise RuntimeError('Need to specify an insturction suffix or special parse method.')
    
    print(f'Found {len(set(selected_prompts))} chosen prompts out of {len(set(all_prompts))} unique prompts')

    # Get value counts
    value_counts = sorted(dict(Counter(selected_prompts)).items(), key=lambda i: i[1], reverse=True)
    for prompt in set(all_prompts):
        if prompt not in [p[0] for p in value_counts]:
            value_counts += [(prompt, 0)]

    return [{
        'prompt': i[0],
        'weight': round(i[1] / len(selected_prompts), 6)
    } for i in value_counts]


def calculate_prompt_weights(results_files, output_json, instruction_suffix=None, parse_method=None):
    """
    Given simplification results, get weights for each prompt depending on usage.
    """
    weights_output = {}
    for result_file in results_files:
        print(f"Getting weights for {result_file}...")

        with open(result_file, 'r', encoding='utf-8') as f:
            results = json.load(f)

        weights = get_prompt_dist(results, instruction_suffix, parse_method)
        weights_output[result_file] = weights
        with open(output_json, "w", encoding='utf-8') as f:
            json.dump(weights_output, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    # Simplification prompt weights
    RESULTS_FILES = [
        ''
    ]
    RESULTS_FILES = [os.path.join(RESULTS_DIR, f) for f in RESULTS_FILES]
    OUTPUT_PATH = os.path.join(RESULTS_DIR, f'weights-simplification-{get_time()}.json')
    calculate_prompt_weights(RESULTS_FILES, OUTPUT_PATH, instruction_suffix=SIMPLIFICATION_INSTRUCTION_SUFFIX)

    # Translation prompt weights
    RESULTS_FILES = [
        ''
    ]
    RESULTS_FILES = [os.path.join(RESULTS_DIR, f) for f in RESULTS_FILES]
    OUTPUT_PATH = os.path.join(RESULTS_DIR, f'weights-translation-{get_time()}.json')
    translation_suffix = get_translate_labels('en', 'cs', return_suffix=True)
    calculate_prompt_weights(RESULTS_FILES, OUTPUT_PATH, instruction_suffix=translation_suffix)

    # Code generation prompt weights
    RESULTS_FILES = [
        ''
    ]
    RESULTS_FILES = [os.path.join(RESULTS_DIR, f) for f in RESULTS_FILES]
    OUTPUT_PATH = os.path.join(RESULTS_DIR, f'weights-code-{get_time()}.json')
    calculate_prompt_weights(RESULTS_FILES, OUTPUT_PATH, parse_method='code')
