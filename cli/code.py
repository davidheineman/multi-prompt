import os

from resources.code import *
from cli.candidate_generation import run_experiment
from cli.utils.interrupt import init_handlers
from cli.utils.subprocesses import init_server
from cli.utils.constants import IS_OVERCAP

from src.tasks.code import parse_human_eval_instruction
from src.dataloader import load_prompts, load_human_eval
from src.prompt import embed_prompts, get_closest_points, get_furthest_points, cluster_prompts


def main():
    if IS_OVERCAP:
        init_handlers()
        init_server()

    human_eval = load_human_eval(HUMAN_EVAL_PATH)
    gpt_zero_prompts = load_prompts(GPT_PROMPT_PATH)

    default_prompts = []
    for example in human_eval:
        default_prompts += [{
            'task_id': example['metadata']['task_id'],
            'prompts': [parse_human_eval_instruction(example['source'])]
        }]

    test_mbr(human_eval, gpt_zero_prompts, n_prompts=5)
    test_mbr(human_eval, gpt_zero_prompts, n_prompts=20)
    test_mbr(human_eval, gpt_zero_prompts, n_prompts=100)
    test_reranker(human_eval, gpt_zero_prompts)
    test_beam(human_eval, gpt_zero_prompts)  
    test_prompt_sampling(human_eval, gpt_zero_prompts, subset_size=10)
    test_prompt_sampling(human_eval, gpt_zero_prompts, subset_size=20)
    

def test_mbr(ds, prompts, n_prompts):
    for e in range(1):
        k, p = 1, n_prompts
        run_experiment(f'e={e}-gpt-zero', 'code', ds, prompts, k=k, p=p)
        k, p = n_prompts, 1
        run_experiment(f'e={e}-gpt-zero', 'code', ds, prompts, k=k, p=p)


def test_reranker(ds, prompts):
    """ Re-ranking Experiments """
    for ranking_method in ['mbr', 'reranker', 'reranker-mbr', 'multi-turn-mbr']:
        k, p = 1, 20
        run_experiment(f'gpt-zero', 'code', ds, prompts, k=k, p=p, ranking_method=ranking_method)
        k, p = 20, 1
        run_experiment(f'gpt-zero', 'code', ds, prompts, k=k, p=p, ranking_method=ranking_method)


def test_beam(ds, prompts):
    """ Beam decoding experiments """
    for e in range(3):
        for p, k in reversed([(1, 20), (1, 18), (1, 16), (1, 14), (1, 12), (1, 10), (1, 8), (1, 6), (1, 4), (1, 2), (1, 1)]):
            run_experiment(
                f'gpt-zero-e={e}', 'code', ds, prompts, k=k, p=p, decoding_method='beam'
            )


def test_prompt_sampling(ds, prompts, subset_size=5):
    SUBSET_SIZE = subset_size

    # Baselines
    k, p = SUBSET_SIZE, 1
    run_experiment(f'gpt-zero', 'code', ds, prompts, k=k, p=p)

    import random
    
    p, k = SUBSET_SIZE, 1
    
    gpt_zero_random, gpt_zero_closest, gpt_zero_farthest, gpt_zero_cluster_random = [], [], [], []
    
    for prompt_entry in prompts:
        task_id, prompt_set = prompt_entry['task_id'], prompt_entry['prompts']
        embedding = embed_prompts(prompt_set)
        (gpt_zero_closest_idx, min_dist), (gpt_zero_farthest_idx, max_dist) = get_closest_points(embedding), get_furthest_points(embedding)
    
        gpt_zero_random         += [{'task_id': task_id, 'prompts': random.sample(prompt_set, SUBSET_SIZE)}]
        gpt_zero_closest        += [{'task_id': task_id, 'prompts': [prompt_set[i] for i in gpt_zero_closest_idx]}]
        gpt_zero_farthest       += [{'task_id': task_id, 'prompts': [prompt_set[i] for i in gpt_zero_farthest_idx]}]
        gpt_zero_cluster_random += [{'task_id': task_id, 'prompts': cluster_prompts(prompt_set, embedding, 'random')}]
    
    run_experiment('gpt-zero-random', 'code', ds, gpt_zero_random, k=k, p=p)
    run_experiment('gpt-zero-closest-prompts', 'code', ds, gpt_zero_closest, k=k, p=p)
    run_experiment('gpt-zero-farthest-prompts', 'code', ds, gpt_zero_farthest, k=k, p=p)
    run_experiment('gpt-zero-cluster-random', 'code', ds, gpt_zero_cluster_random, k=k, p=p)


if __name__ == "__main__":
    main()
