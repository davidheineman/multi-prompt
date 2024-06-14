import os, json, random

from cli.utils.constants import ROOT_DIR, OPENAI_SECRET_PATH
from cli.resources.code import HUMAN_EVAL_PATH

from src.tasks.code import parse_human_eval_instruction
from src.dataloader import load_human_eval
from src.generate import openai_init, generate_gpt

WRITE_PROMPT = """{write_instructions} Please do not write any examples.

Prompt:"""

HUMAN_EVAL_PROMPT = """Please write a variation of the following instruction for a coding task. You may be creative in proposing potential solutions, or explaining the nature of the task. Please do not write any examples. 

Instruction: {instruction}

Variation of Instruction:"""

WRITE_PROMPT_FEW_SHOT = """{write_instructions}

Example: {example_1}

Example: {example_2}

Prompt:"""


PROMPT_GENERATOR = 'gpt-4-turbo-2024-04-09'


def write_zero_shot(write_prompt, folder_path, n_prompts=100):
    path = os.path.join(folder_path, f'{PROMPT_GENERATOR}_zero.json')
    
    generation = generate_gpt([write_prompt for _ in range(n_prompts)])
    zero_shot = [gen.lstrip(' \n') for gen in generation]

    with open(path, 'w') as f:
        json.dump(zero_shot, f, indent=4)


def write_few_shot(write_instructions, human_prompts, folder_path, n_prompts=100):
    path = os.path.join(folder_path, f'{PROMPT_GENERATOR}_few.json')
    
    write_prompts = []
    for _ in range(n_prompts):
        example_1, example_2 = random.sample(human_prompts, 2)

        write_prompts += [WRITE_PROMPT_FEW_SHOT.format(
            write_instructions=write_instructions,
            example_1=example_1,
            example_2=example_2
        )]

    generation = generate_gpt([prompt for prompt in write_prompts])
    few_shot_prompts = [gen.lstrip(' \n') for gen in generation]

    with open(path, 'w') as f:
        json.dump(few_shot_prompts, f, indent=4)


def write_human_eval_prompts(instruction, folder_path, n_prompts=100):
    """
    Write prompts for HumanEval coding dataset. HumanEval is structured as such:

    def has_close_elements(numbers: List[float], threshold: float) -> bool:
        # Check if in given list of numbers, are any two numbers closer to each other than given threshold.
        >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
        False
        >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
        True
    
    We will extract only the prompt and use GPT to write variations of that prompt.
    """
    path = os.path.join(folder_path, f'{PROMPT_GENERATOR}.json')
    human_eval = load_human_eval(HUMAN_EVAL_PATH)
    zero_shot = []

    # Continue progress if applicable
    if os.path.exists(path) and os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as f:
            zero_shot = json.load(f)
        task_ids = [e['task_id'] for e in zero_shot]
        human_eval = [h for h in human_eval if h['metadata']['task_id'] not in task_ids]
        print(f'Found existing prompts! Only generating for {len(human_eval)} tasks...')

    # Sanity check the instruction parsing
    for entry in human_eval:
        assert parse_human_eval_instruction(entry['source']) != ''
    
    for entry in human_eval:
        print(f"Writing prompts for HumanEval task \'{entry['metadata']['task_id']}\'...")

        parsed_instruction = parse_human_eval_instruction(entry['source'])

        write_prompt = HUMAN_EVAL_PROMPT.format(instruction=parsed_instruction)

        # Generate with instruction here
        generation = generate_gpt([write_prompt for _ in range(n_prompts)])
        prompts = [gen.lstrip(' \n') for gen in generation]

        zero_shot += [{
            'task_id': entry['metadata']['task_id'],
            'prompts': prompts
        }]
        
        with open(path, 'w') as f:
            json.dump(zero_shot, f, indent=4)


def write_prompts(task, n_prompts=100):
    print(f'Writing {n_prompts} prompts for {task}...')

    # Get instruction for task
    tasks_path = os.path.join(ROOT_DIR, 'prompts', f'prompt_writing.json')
    with open(tasks_path, 'r') as f:
        task_instructions = json.load(f)
    if task not in task_instructions.keys():
        raise ValueError(f"Task {task} not found! Must be one of: {task_instructions.keys()}")
    instruction = task_instructions[task]

    # Create task folder
    folder_path = os.path.abspath(os.path.join(ROOT_DIR, 'prompts', task))
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Get prompt for task
    prompt = WRITE_PROMPT.format(write_instructions=instruction)
        
    if task == 'code':
        write_human_eval_prompts(prompt, folder_path, n_prompts=n_prompts)
        return

    # Write with zero-shot prompts
    # write_zero_shot(prompt, folder_path, n_prompts=n_prompts)

    # Write with few-shot prompts
    if os.path.exists(os.path.join(folder_path, f'human.json')):
        with open(os.path.join(folder_path, f'human.json'), 'r', encoding='utf-8') as f:
            human_prompts = json.load(f)
    else:
        raise RuntimeError('No human prompts found!')

    write_few_shot(instruction, human_prompts, folder_path, n_prompts=n_prompts)


if __name__ == '__main__':
    openai_init(OPENAI_SECRET_PATH, model_name=PROMPT_GENERATOR)

    write_prompts('code', n_prompts=1000)

    # write_prompts('simplification', n_prompts=1000)
