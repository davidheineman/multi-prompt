import os, json
from typing_extensions import Literal

from utils.constants import MODEL_NAME, METRIC_NAME, MODEL_PORT, RESULTS_DIR, OPENAI_SECRET_PATH, IS_OVERCAP, MODEL_TEMPLATES
from utils.interrupt import overcap_handle_iter, claim_data_ids

from src.endpoint import get_metric, METRIC_SCORING_FUNCTIONS
from src.util import get_time
from src.mbr import candidate_argmax
from src.generate import generate_endpoint, generate_gpt, openai_init, get_generate_params
from src.prompt import sample_prompts
from src.metrics.human_eval.filter import is_degenerate
from src.tasks import get_task_setup
from src.tasks.code import prepare_human_eval_template, check_compiles, GPT_4_INSTRUCTION


def run_experiment(
        run_label: str, 
        task: str, 
        data: list[dict], 
        instruction: list[str], 
        k: int, 
        p: int, 
        n: float = 0.6, 
        temp: float = 0.9, 
        ranking_method: Literal['mbr', 'reranker', 'reranker-mbr', 'multi-turn-mbr'] = 'mbr', 
        decoding_method: Literal['top-p', 'epsilon', 'beam'] = 'top-p', 
        weight_method: Literal['uniform', 'best', 'top-n', 'ensemble', 'top-n-ensemble'] = 'uniform', 
        mbr_type: Literal['efficient', 'full_matrix', 'no_matrix'] = 'full_matrix',
        prompt_weights = None, 
        prompt_template: str = None,
    ) -> dict:
    """Run a multi-prompt MBR experiment.

    Args:
        instruction: An individual prompt, or bank of instructions to sample from
        k: Number of MBR candidates to generate per prompt
        p: Number of prompts to sample from to generate each set of k candidates (total number of candidates is p X k)

        ranking_method: mbr, reranker, reranker-mbr, multi-turn-mbr
        decoding_method: top-p, epsilon, beam
        weight_method: uniform, best, top-n, ensemble, top-n-ensemble

    Return:
        A json with predictions
    """
    run_prefix = f'{task}-{run_label}-{MODEL_NAME}-{METRIC_NAME}-{decoding_method}-{ranking_method}-{weight_method}-p={p}-k={k}'
    run_name = f'{run_prefix}-{get_time()}'
    results_json = os.path.join(RESULTS_DIR, f'{run_name}.json')

    print(f"Beginning run: {run_name}")
    print(f"Using params: k={k}, p={p}, decoding_method={decoding_method}, ranking_method={ranking_method}, weight_method={weight_method}")

    generate_params = get_generate_params(decoding_method, n_beams=k, temp=temp)
    
    generator = generate_endpoint
    if 'gpt' in MODEL_NAME:
        openai_init(model_name=MODEL_NAME, secret_path=OPENAI_SECRET_PATH) 
        generator = generate_gpt

    instruction, template, postprocessor, all_problem_prompts = get_task_setup(task, instruction, MODEL_NAME)
    metrics = get_metric(task, ranking_method)

    # Override the scoring function if the METRIC_NAME is given by a slurm argument
    if ranking_method == 'mbr' and METRIC_NAME is not None and METRIC_NAME in METRIC_SCORING_FUNCTIONS:
        metrics = METRIC_SCORING_FUNCTIONS.get(METRIC_NAME)

    print(f'Using metric {metrics}...')

    # Get chat template for chat models
    if MODEL_NAME in MODEL_TEMPLATES.keys():
        template = MODEL_TEMPLATES[MODEL_NAME]

    # Manual overrides
    if prompt_template: template = prompt_template

    # Run generation
    batch = claim_data_ids(run_prefix, data) if IS_OVERCAP else data
    results, i = [], 0

    while i < len(batch):
        sent = batch[i]
        source, refs, metadata = sent['source'], sent['references'], sent['metadata']

        instruction_weights, cand_filter_func = prompt_weights, None
        if task == 'code':
            def code_filter_f(cand):
                return check_compiles({'target': cand, **sent}) and not is_degenerate(cand)
            cand_filter_func = code_filter_f
            if isinstance(all_problem_prompts, list):
                # Get prompts for specific HumanEval task
                instruction = [e for e in all_problem_prompts if e['task_id'] == metadata['task_id']][0]['prompts']
                if prompt_weights:
                    instruction_weights = [e for e in prompt_weights if e['task_id'] == metadata['task_id']][0]['weights']
            else:
                # If we are given a single string, use this for all prompts
                instruction, instruction_weights = all_problem_prompts, prompt_weights

        prompts = sample_prompts(instruction, k, p, weight_method, instruction_weights, n=n)

        if task == 'code':
            template = prepare_human_eval_template(source)
            if 'gpt-4' in MODEL_NAME: template = GPT_4_INSTRUCTION + template
            prompts_with_instructions = [
                template.replace('{instructions}', f' {prompt}')
                for prompt in prompts
            ]
        else:
            prompts_with_instructions = [
                template.format(instructions=prompt, source=source)
                for prompt in prompts
            ]

        try:
            target, candidates, score_matrix, unfiltered = candidate_argmax(
                source = source, 
                prompt = prompts_with_instructions, 
                generate_sentence_func = generator, 
                metrics = metrics, 
                text_postprocessing_func = postprocessor,
                cand_filter_func = cand_filter_func,
                return_scores = True, 
                port = MODEL_PORT, 
                metadata = metadata,
                mbr_type = mbr_type,
                **generate_params
            )

            result = {
                'source': source,
                'target': target,
                'references': refs
            }

            if candidates is not None: result['candidates'] = candidates
            if score_matrix is not None: result['score_matrix'] = score_matrix
            if unfiltered is not None: result['unfiltered_candidates'] = unfiltered
            if metadata is not None: result['metadata'] = metadata
            
            results += [result]
        except RuntimeError as e:
            print(f'Failed to generate on: {source} -- Recieved: {e}')
            i += 1
            continue

        if IS_OVERCAP:
            batch, results, i = overcap_handle_iter(run_prefix, data, batch, i, results)
            
        else:
            if len(results) == 0:
                raise ValueError("Failed to generate any sentences...")

            with open(results_json, "w", encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
        
        i += 1

    return results_json
