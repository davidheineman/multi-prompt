from resources.simplification import *
from cli.candidate_generation import run_experiment
from cli.utils.interrupt import init_handlers
from cli.utils.subprocesses import init_server
from cli.utils.constants import IS_OVERCAP

from src.dataloader import load_prompts, load_simp_eval
from src.prompt import embed_prompts, get_closest_points, get_furthest_points, cluster_prompts
from src.tasks.simplification import construct_few_shot, SIMPLIFICATION_INSTRUCTION


def main():
    if IS_OVERCAP:
        init_handlers()
        init_server()

    simp_eval_diag = load_simp_eval(SIMPEVAL_PATH, split='diagnostic')
    simp_eval      = load_simp_eval(SIMPEVAL_PATH)
    human_prompts  = load_prompts(HUMAN_PROMPT_PATH)

    test_mbr_basic(simp_eval_diag)
    test_mbr_basic(simp_eval)
    test_mbr(simp_eval)
    test_reranker(simp_eval)
    test_hypothesis_size(simp_eval)
    test_single_prompt(simp_eval)
    text_top_n(simp_eval)
    test_beam(simp_eval)
    test_icl(simp_eval)
    test_temperature(simp_eval)
    test_prompt_sampling(simp_eval)


def test_mbr_basic(ds):
    gpt_few_prompts = load_prompts(GPT_PROMPT_PATH)
    gpt_few_prompts_1000 = load_prompts(GPT_1000_PROMPT_PATH)

    # Basic experiment
    k, p = 1, 10
    run_experiment('gpt-few', 'simplification', ds, gpt_few_prompts, k=k, p=p)
    k, p = 10, 1
    run_experiment('gpt-few', 'simplification', ds, gpt_few_prompts, k=k, p=p)

    k, p = 1, 100
    run_experiment('gpt-few', 'simplification', ds, gpt_few_prompts, k=k, p=p)
    k, p = 100, 1
    run_experiment('gpt-few', 'simplification', ds, gpt_few_prompts, k=k, p=p)

    # 500 candidate experiment
    k, p = 1, 500
    run_experiment('gpt-few', 'simplification', ds, gpt_few_prompts_1000, k=k, p=p)
    k, p = 500, 1
    run_experiment('gpt-few', 'simplification', ds, gpt_few_prompts_1000, k=k, p=p)


def test_mbr(ds):
    # MBR experiment
    gpt_few_prompts, gpt_few_prompts_weights = load_prompts(GPT_WEIGHTED_PROMPT_PATH, return_weights=True)
    k, p = 1, 100
    run_experiment(
        f'gpt-few', 'simplification', ds, gpt_few_prompts, k=k, p=p, 
        weight_method='top-n-ensemble', prompt_weights=gpt_few_prompts_weights, temp=0.6
    )
    k, p = 100, 1
    run_experiment(
        f'gpt-few', 'simplification', ds, gpt_few_prompts[0], k=k, p=p, temp=0.6
    )


def test_reranker(ds):
    # Re-ranker experiment
    gpt_few_prompts, gpt_few_prompts_weights = load_prompts(GPT_WEIGHTED_PROMPT_PATH, return_weights=True)
    for e in range(3):
        for ranking_method in ['reranker', 'mbr', 'reranker-mbr', 'multi-turn-mbr']:
            k, p = 1, 100
            run_experiment(
                f'gpt-few-e={e}', 'simplification', ds, gpt_few_prompts, k=k, p=p, 
                weight_method='top-n-ensemble', prompt_weights=gpt_few_prompts_weights, ranking_method=ranking_method
            )
    k, p = 100, 1
    run_experiment(
        f'gpt-few', 'simplification', ds, gpt_few_prompts, k=k, p=p
    )


def test_hypothesis_size(ds):
    # Constant candidate set size experiment (temp=0.3)
    gpt_few_prompts, gpt_few_prompts_weights = load_prompts(GPT_WEIGHTED_PROMPT_PATH, return_weights=True)
    for e in range(3):
        for p, k in [[100, 1], [50, 2], [25, 4], [20, 5], [10, 10], [5, 20], [4, 25], [2, 50], [1, 100]]:
            run_experiment(
                f'gpt-few-e={e}', 'simplification', ds, gpt_few_prompts, 
                k=k, p=p, weight_method='top-n-ensemble', prompt_weights=gpt_few_prompts_weights, temp=0.3
            )


def test_single_prompt(ds):
    """ Single prompt using each prompt experiment """
    gpt_few_prompts, gpt_few_prompts_weights = load_prompts(GPT_WEIGHTED_PROMPT_PATH, return_weights=True)
    k, p = 1, 100
    run_experiment(
        'gpt-few', 'simplification', ds, gpt_few_prompts, 
        k=k, p=p, weight_method='top-n-ensemble', prompt_weights=gpt_few_prompts_weights
    )
    for prompt_idx, prompt in enumerate(gpt_few_prompts):
        k, p = 100, 1
        run_experiment(f'gpt-few-idx={prompt_idx}', 'simplification', ds, prompt, k=k, p=p)
    
    human_prompts, human_prompts_weights = load_prompts(HUMAN_WEIGHTED_PROMPT_PATH, return_weights=True)
    k, p = 1, 10
    run_experiment(
        'human', 'simplification', ds, human_prompts, k=k, p=p, 
        weight_method='top-n-ensemble', prompt_weights=human_prompts_weights
    )
    for prompt_idx, prompt in enumerate(human_prompts):
        k, p = 10, 1
        run_experiment(f'human-idx={prompt_idx}', 'simplification', ds, prompt, k=k, p=p)


def text_top_n(ds):
    # Ablate the prompt set size in top-n prompt sampling
    gpt_few_prompts, gpt_few_prompts_weights = load_prompts(GPT_WEIGHTED_PROMPT_PATH, return_weights=True)
    k, p = 1, 100
    for subset_size in [1, 2, 5, 10, 20, 40, 60, 80, 100]:
        prompt_set, prompt_weights = gpt_few_prompts[:subset_size], gpt_few_prompts_weights[:subset_size]
        for weight_method in ['top-n-ensemble']:    
            run_experiment(
                f'gpt-few-top-n={subset_size}-llama-7b', 'simplification', ds, prompt_set,
                k=k, p=p, weight_method=weight_method, prompt_weights=prompt_weights
            )

    # Ablate the n in top-n prompt sampling
    gpt_few_prompts, gpt_few_prompts_weights = load_prompts(GPT_WEIGHTED_PROMPT_PATH, return_weights=True)
    k, p = 1, 100
    for n in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        for weight_method in ['top-n-ensemble']:    
            run_experiment(
                f'gpt-few-top-n={n}-llama-7b', 'simplification', ds, gpt_few_prompts,
                k=k, p=p, weight_method=weight_method, prompt_weights=gpt_few_prompts_weights
            )


def test_beam(ds):
    # Beam decoding experiments
    gpt_few_prompts = load_prompts(GPT_WEIGHTED_PROMPT_PATH)
    for p, k in reversed([(1, 100), (1, 80), (1, 60), (1, 40), (1, 20), (1, 10), (1, 5), (1, 2), (1, 1)]):
        run_experiment(
            'gpt-few', 'simplification', ds, gpt_few_prompts, 
            k=k, p=p, decoding_method='beam'
        )


def test_icl(ds):
    # In-context example simplification experiment
    n_prompts = 100
    few_shot_prompts = construct_few_shot(ds, n_prompts, n_icl=3)
    k, p = 1, n_prompts
    run_experiment('few-shot', 'simplification', ds, few_shot_prompts, k=k, p=p)
    k, p = n_prompts, 1
    run_experiment('few-shot', 'simplification', ds, few_shot_prompts, k=k, p=p)


def test_temperature(ds):
    # Different temperature decoding
    import numpy as np
    single_prompt = SIMPLIFICATION_INSTRUCTION
    human_prompts, human_prompt_weights = load_prompts(HUMAN_WEIGHTED_PROMPT_PATH, return_weights=True)
    
    for e in range(10):
        for temp in np.arange(0.1, 2, 0.1):
            k, p = 10, 1
            run_experiment(
                f'human-e={e}-t={temp}', 'simplification', ds, single_prompt, k=k, p=p, temp=temp
            )
            k, p = 1, 10
            run_experiment(
                f'human-e={e}-t={temp}', 'simplification', ds, human_prompts, 
                k=k, p=p, temp=temp, weight_method='top-n-ensemble', prompt_weights=human_prompt_weights
            )


def test_prompt_sampling(ds):
    SUBSET_SIZE = 10

    gpt_few_prompts, gpt_few_prompts_weights = load_prompts(GPT_WEIGHTED_PROMPT_PATH, return_weights=True)

    # Baselines
    k, p = SUBSET_SIZE, 1
    run_experiment('gpt-few', 'simplification', ds, gpt_few_prompts, k=k, p=p, temp=0.6)
    k, p = 100, 1
    run_experiment('gpt-few', 'simplification', ds, gpt_few_prompts, k=k, p=p, temp=0.6)

    # More prompt examples
    k, p = SUBSET_SIZE, 1
    run_experiment('gpt-few-best-prompt', 'simplification', ds, gpt_few_prompts[0], k=k, p=p, temp=0.6)
    k, p = 100, 1
    run_experiment('gpt-few-best-prompt', 'simplification', ds, gpt_few_prompts[0], k=k, p=p, temp=0.6)

    # Experiment with prompt semantic similarity. Note this will select a subset (e.g., 10 of 100 prompts)
    embedding = embed_prompts(gpt_few_prompts)
    (gpt_few_closest_idx, min_dist), (gpt_few_farthest_idx, max_dist) = get_closest_points(embedding), get_furthest_points(embedding)
    
    import random
    gpt_few_random = random.sample(gpt_few_prompts, SUBSET_SIZE)
    gpt_few_highest_performing = [p for p, _ in sorted(list(zip(gpt_few_prompts, gpt_few_prompts_weights)), key=lambda x: x[1], reverse=True)][:SUBSET_SIZE]
    
    gpt_few_closest = [gpt_few_prompts[i] for i in gpt_few_closest_idx]
    gpt_few_farthest = [gpt_few_prompts[i] for i in gpt_few_farthest_idx]
    gpt_few_cluster_random = cluster_prompts(gpt_few_prompts, embedding, 'random')
    gpt_few_cluster_performance = cluster_prompts(gpt_few_prompts, embedding, 'performance', prompt_weights=gpt_few_prompts_weights)

    k, p = 1, SUBSET_SIZE
    
    run_experiment(f'gpt-few-random', 'simplification', ds, gpt_few_closest, k=k, p=p, temp=0.6)
    run_experiment(f'gpt-few-highest-performance', 'simplification', ds, gpt_few_closest, k=k, p=p, temp=0.6)
    run_experiment(f'gpt-few-closest-prompts', 'simplification', ds, gpt_few_closest, k=k, p=p, temp=0.6)
    run_experiment(f'gpt-few-farthest-prompts', 'simplification', ds, gpt_few_farthest, k=k, p=p, temp=0.6)
    run_experiment(f'gpt-few-cluster-random', 'simplification', ds, gpt_few_cluster_random, k=k, p=p, temp=0.6)
    run_experiment(f'gpt-few-cluster-performance', 'simplification', ds, gpt_few_cluster_performance, k=k, p=p, temp=0.6)

    # Prompt weighting experiments
    gpt_few_prompts, gpt_few_prompts_weights = load_prompts(GPT_WEIGHTED_PROMPT_PATH, return_weights=True)
    k, p = 1, 100
    for weight_method in ['uniform', 'best', 'top-n', 'top-n-ensemble', 'ensemble']:    
        run_experiment(
            'gpt-few', 'simplification', ds, gpt_few_prompts,
            k=k, p=p, weight_method=weight_method, prompt_weights=gpt_few_prompts_weights
        )


if __name__ == "__main__":
    main()