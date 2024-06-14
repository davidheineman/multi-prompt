from src.tasks.code import truncate_multiline
from src.tasks.simplification import SIMPLIFY_PROMPT


def get_task_setup(task, instruction, model_name):
    text_postprocessing_func, all_problem_prompts, template = None, None, None
    match task:
        case 'simplification':
            template = SIMPLIFY_PROMPT
            if 'T5' in model_name:
                # For T5, we use the control token model. We will ignore the task instruction
                # input and use the control tokens instead. See: github.com/Yao-Dou/LENS/tree/master/generation
                template = '{instructions} {source}'
                instruction = ['<NC_0.95> <LS_0.75> <DR_0.75> <WR_0.75>' for _ in instruction]
        case 'code':
            text_postprocessing_func = truncate_multiline
            all_problem_prompts = instruction
        case 'translation':
            pass
        case _:
            raise RuntimeError(f'Task "{task}" does not have a specific configuration...')

    return instruction, template, text_postprocessing_func, all_problem_prompts