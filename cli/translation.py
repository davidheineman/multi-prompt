import os

from resources.translation import *
from cli.candidate_generation import run_experiment
from cli.utils.interrupt import init_handlers
from cli.utils.subprocesses import init_server
from cli.utils.constants import IS_OVERCAP

from src.tasks.translation import construct_few_shot, get_translate_prompt_template
from src.dataloader import load_prompts, load_newstest19, load_alma_test, load_ntrex
from src.prompt import embed_prompts, get_closest_points, get_furthest_points, cluster_prompts


def main():
    if IS_OVERCAP:
        init_handlers()
        init_server()

    src_lang, tgt_lang = 'en', 'cs'

    newstest19_all = load_newstest19(NEWSTEST19_PATH, src_lang=src_lang, tgt_lang=tgt_lang)
    
    alma_test      = load_alma_test(ALMA_TEST_PATH, src_lang=src_lang, tgt_lang=tgt_lang)
    alma_test_diag = load_alma_test(ALMA_TEST_PATH, src_lang=src_lang, tgt_lang=tgt_lang, split='diagnostic')

    # en_cs_prompts_100 = construct_few_shot(newstest19_all, src_lang, tgt_lang, 100)
    # en_cs_prompts_1000 = construct_few_shot(newstest19_all, src_lang, tgt_lang, 1000)

    en_cs_prompts, en_cs_weights = load_prompts(os.path.join(PROMPT_DIR, 'en-cs.json'), return_weights=True)
    en_cs_prompts_1000, en_cs_weights_1000 = load_prompts(os.path.join(PROMPT_DIR, 'en-cs-1000.json'), return_weights=True)

    test_mbr(alma_test_diag, src_lang, tgt_lang, en_cs_prompts, en_cs_weights, n_prompts=100)
    test_mbr(alma_test, src_lang, tgt_lang, en_cs_prompts, en_cs_weights, n_prompts=100)
    test_mbr(alma_test, src_lang, tgt_lang, en_cs_prompts_1000, en_cs_weights_1000, n_prompts=500)
    test_reranker(alma_test, src_lang, tgt_lang, en_cs_prompts, en_cs_weights, n_prompts=100)
    test_prompt_sampling(alma_test, src_lang, tgt_lang, en_cs_prompts, en_cs_weights)
    test_language_pairs(src_lang)


def test_mbr(ds, src_lang, tgt_lang, prompts, weights, n_prompts):
    template = get_translate_prompt_template(src_lang, tgt_lang)

    k, p = 1, n_prompts
    run_experiment(
        f'few-shot-new-setup-{src_lang}-{tgt_lang}-alma', 'translation', ds, prompts, k=k, p=p, 
        weight_method='top-n-ensemble', prompt_weights=weights, prompt_template=template, 
        temp=0.3
    )
    k, p = n_prompts, 1
    run_experiment(
        f'few-shot-new-setup-{src_lang}-{tgt_lang}-alma', 'translation', ds, prompts[0], 
        k=k, p=p, prompt_template=template, 
        temp=0.3
    )


def test_reranker(ds, src_lang, tgt_lang, prompts, weights, n_prompts):
    template = get_translate_prompt_template(src_lang, tgt_lang)

    for ranking_method in ['mbr', 'reranker', 'reranker-mbr', 'multi-turn-mbr']:
        k, p = 1, n_prompts
        run_experiment(
            f'few-shot-{src_lang}-{tgt_lang}-alma', 'translation', ds, prompts, k=k, p=p, 
            weight_method='top-n-ensemble', prompt_weights=weights, ranking_method=ranking_method, prompt_template=template
        )
    k, p = n_prompts, 1
    run_experiment(
        f'few-shot-{src_lang}-{tgt_lang}-alma', 'translation', ds, prompts, 
        k=k, p=p, prompt_template=template
    )


def test_beam(ds, src_lang, tgt_lang, prompts):
    template = get_translate_prompt_template(src_lang, tgt_lang)

    for p, k in reversed([(1, 100), (1, 80), (1, 60), (1, 40), (1, 20), (1, 10), (1, 5), (1, 2), (1, 1)]):
        run_experiment(
            f'few-shot-{src_lang}-{tgt_lang}-alma', 'translation', ds, prompts, 
            k=k, p=p, prompt_template=template, decoding_method='beam'
        )


def test_prompt_sampling(ds, src_lang, tgt_lang, prompts, weights):
    SUBSET_SIZE = 10
    template = get_translate_prompt_template(src_lang, tgt_lang)

    # Baselines
    k, p = SUBSET_SIZE, 1
    run_experiment(f'few-shot-{src_lang}-{tgt_lang}-alma', 'translation', ds, prompts, k=k, p=p, prompt_template=template)
    k, p = 100, 1
    run_experiment(f'few-shot-{src_lang}-{tgt_lang}-alma', 'translation', ds, prompts, k=k, p=p, prompt_template=template)

    k, p = SUBSET_SIZE, 1
    run_experiment(f'few-shot-{src_lang}-{tgt_lang}-alma', 'translation', ds, prompts[0], k=k, p=p, prompt_template=template)
    k, p = 100, 1
    run_experiment(f'few-shot-{src_lang}-{tgt_lang}-alma', 'translation', ds, prompts[0], k=k, p=p, prompt_template=template)

    # Experiment with prompt semantic similarity. Note this will select a subset (e.g., 10 of 100 prompts)
    import random
    
    p, k = SUBSET_SIZE, 1
    embedding = embed_prompts(prompts)
    (gpt_few_closest_idx, min_dist), (gpt_few_farthest_idx, max_dist) = get_closest_points(embedding), get_furthest_points(embedding)
    
    gpt_few_random              = random.sample(prompts, SUBSET_SIZE)
    gpt_few_highest_performing  = [p for p, _ in sorted(list(zip(prompts, weights)), key=lambda x: x[1], reverse=True)][:SUBSET_SIZE]
    gpt_few_closest             = [prompts[i] for i in gpt_few_closest_idx]
    gpt_few_farthest            = [prompts[i] for i in gpt_few_farthest_idx]
    gpt_few_cluster_random      = cluster_prompts(prompts, embedding, 'random')
    gpt_few_cluster_performance = cluster_prompts(prompts, embedding, 'performance', prompt_weights=weights)
    
    run_experiment('icl-random', 'translation', ds, gpt_few_random, k=k, p=p, prompt_template=template)
    run_experiment('icl-highest-performance', 'translation', ds, gpt_few_highest_performing, k=k, p=p, prompt_template=template)
    run_experiment('icl-closest-prompts', 'translation', ds, gpt_few_closest, k=k, p=p, prompt_template=template)
    run_experiment('icl-farthest-prompts', 'translation', ds, gpt_few_farthest, k=k, p=p, prompt_template=template)
    run_experiment('icl-cluster-random', 'translation', ds, gpt_few_cluster_random, k=k, p=p, prompt_template=template)
    run_experiment('icl-cluster-performance', 'translation', ds, gpt_few_cluster_performance, k=k, p=p, prompt_template=template)

    # Prompt weighting experiments
    k, p = 1, 100
    for weight_method in ['uniform', 'best', 'top-n', 'top-n-ensemble', 'ensemble']:
        run_experiment(
            'alma-en-cs', 'translation', ds, prompts, k=k, p=p, 
            weight_method=weight_method, prompt_weights=weights, prompt_template=template, temp=0.3
        )


def test_language_pairs(src_lang):
    for tgt_lang in ['de', 'cs', 'is', 'zh', 'ru']:
        alma_test = load_alma_test(ALMA_TEST_PATH, src_lang=src_lang, tgt_lang=tgt_lang, limit=300, sorted_by_length=True)
        alma_test_all = load_alma_test(ALMA_TEST_PATH, src_lang=src_lang, tgt_lang=tgt_lang)

        N_PROMPTS = 100
        few_shot_prompts = construct_few_shot(alma_test_all, src_lang, tgt_lang, N_PROMPTS)
        template = get_translate_prompt_template(src_lang, tgt_lang)
    
        k, p = 1, 100
        run_experiment(
            f'few-shot-{src_lang}-{tgt_lang}-alma', 'translation', alma_test, few_shot_prompts, 
            k=k, p=p, prompt_template=template
        )
    
        k, p = 100, 1
        run_experiment(
            f'few-shot-{src_lang}-{tgt_lang}-alma', 'translation', alma_test, few_shot_prompts, 
            k=k, p=p, prompt_template=template
        )

    for tgt_lang in ['fra', 'tur', 'cat', 'ind', 'jpn', 'urd']:
        ntrex = load_ntrex(NTREX_PATH, src_lang=src_lang, tgt_lang=tgt_lang, limit=300, sorted_by_length=True)
        ntrex_all = load_ntrex(NTREX_PATH, src_lang=src_lang, tgt_lang=tgt_lang)
    
        N_PROMPTS = 100
        few_shot_prompts = construct_few_shot(ntrex_all, src_lang, tgt_lang, N_PROMPTS)
        template = get_translate_prompt_template(src_lang, tgt_lang)
    
        k, p = 1, 100
        run_experiment(
            f'few-shot-{src_lang}-{tgt_lang}-ntrex', 'translation', ntrex, few_shot_prompts, 
            k=k, p=p, prompt_template=template
        )
    
        k, p = 100, 1
        run_experiment(
            f'few-shot-{src_lang}-{tgt_lang}-ntrex', 'translation', ntrex, few_shot_prompts, 
            k=k, p=p, prompt_template=template
        )


if __name__ == "__main__":
    main()