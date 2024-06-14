import statistics, json, os, random, math
import torch
import numpy as np

from metrics.misc import unique_bigrams, candidate_agreement, oracle, count_filtered
from mbr import reduce_score_matrices

DEVICES = [int(gpu.strip()) for gpu in os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')]
assert len(DEVICES) == 1, 'Multi-GPU evaluation currently not working'

def get_task_metrics(task):
    match task:
        case 'simplification': 
            from metrics.simplification import Sari, Lens, SLE
            from metrics.translation import BertScore
            return [
                Sari(),
                BertScore(devices=DEVICES), 
                Lens(variation='lens', devices=DEVICES),
                Lens(variation='lens_salsa', devices=DEVICES),
                SLE(devices=DEVICES),
            ]
        case 'translation': 
            from metrics.translation import BLEU, BertScore, BartScore, Comet, XComet, MetricX
            return [
                BLEU(), 
                BertScore(devices=DEVICES), 
                Comet(variation='comet_22', devices=DEVICES), 
                Comet(variation='comet_kiwi_23', size='xl', devices=DEVICES), 
                XComet(size='xl', devices=DEVICES), 
                MetricX(variation='metricx', size='xl', devices=DEVICES),
                MetricX(variation='metricx_qe', size='xl', devices=DEVICES)
            ]
        case 'code':
            from metrics.code import CodeBertScore, FuncCorrect
            return [
                FuncCorrect(variation='pass@1', n_workers=164),
                FuncCorrect(variation='pass@k', n_workers=164),
            ]
        case _: raise NotImplementedError(f'Task metrics for "{task}" not supported!')


def evaluate(src, pred, ref, metrics, cands, cand_scores, include_oracle=False):    
    # Run and print results on automatic metrics
    scores = {}
    for metric in metrics:
        dataset_scores = metric(src=src, pred=pred, ref=ref)
        scores[f'{metric.name}_per_source'] = dataset_scores
        scores[metric.name] = statistics.mean(dataset_scores)
        torch.cuda.empty_cache()
    
    # Calculate candidate set metrics
    scores['unique_bigrams'] = unique_bigrams(
        list_of_candidates=cands, candidate_scores=cand_scores, src=src, ref=ref
    )
    scores['candidate_agreement'] = candidate_agreement(
        list_of_candidates=cands, candidate_scores=cand_scores, src=src, ref=ref
    )

    # Calculate oracle performance (slow!)
    if include_oracle:
        for metric in metrics:
            oracle_scores = oracle(list_of_candidates=cands, src=src, ref=ref, metric=metric)
            scores[f'{metric.name}_oracle_per_source'] = oracle_scores
            scores[f'{metric.name}_oracle'] = statistics.mean(oracle_scores)

    return scores


def code_evaluate(pred, metadata, metrics, unfiltered_cands=None):
    scores = {}
    
    # if unfiltered_cands:
    #     scores['%_no_compile'] = statistics.mean(count_filtered(unfiltered_cands))

    for metric in metrics:
        if metric.name == 'pass@k' and unfiltered_cands:
            dataset_scores = metric(unfiltered_cands=unfiltered_cands, metadata=metadata)
        else:
            dataset_scores = metric(pred=pred, metadata=metadata)
        scores[f'{metric.name}_per_source'] = dataset_scores
        scores[metric.name] = statistics.mean(dataset_scores)
        torch.cuda.empty_cache()

    return scores


def subset_selection(score_matrices_list, cands_list, s, n_filtered=None):
    subset_pred, subset_cands, subset_scores = [], [], []
    for sent_id, score_matrices in enumerate(score_matrices_list):
        cands = cands_list[sent_id]
        subset_size = s
        if n_filtered is not None:
            subset_size = max(1, math.floor(s*(1-n_filtered[sent_id])))
            # print(f'Running subset eval with subset_size={s}, but sent_id={sent_id} has score matrix shape {matrix.shape[0]} and n_filtered={n_filtered[sent_id]}. This is likely because more than subset_size canidates were filtered. Setting subset_size={subset_size}...')

        # Get random selection of indices
        ind = sorted(random.sample(range(len(cands)), k=subset_size))
        cands = [cands[i] for i in ind]

        is_multi_metric = len(score_matrices) < 3

        # Re-compute the score matrix based on the subset selection
        if not is_multi_metric:
            matrix = np.array(score_matrices)
            is_2d = matrix.ndim == 2
            matrix = matrix[ind][:, ind] if is_2d else matrix[ind]
            scores = np.mean(matrix, axis=1).tolist() if is_2d else matrix
            generation = cands[np.argmax(scores)]
        else:
            selected_matrices = []
            for matrix in score_matrices:
                matrix = np.array(matrix)
                is_2d = matrix.ndim == 2
                matrix = matrix[ind][:, ind] if is_2d else matrix[ind]
                selected_matrices += [matrix]
            generation, cands, scores = reduce_score_matrices(selected_matrices, cands)
        
        subset_pred   += [generation]
        subset_cands  += [cands]
        subset_scores += [scores]

    return subset_pred, subset_cands, subset_scores


def evaluate_results_file(results_json, task, metrics, include_oracle=False, subset_eval=False):
    """
    Load and evaluate a results file
    """
    # Load the results file
    with open(results_json, 'r', encoding='utf-8') as f:
        results = json.load(f)

    src      = [s['source'] for s in results]
    pred     = [s['target'] for s in results]
    ref      = [s['references'] if 'references' in s.keys() else s['reference'] for s in results]
    metadata = [s['metadata'] for s in results]
    cands    = [[c['candidate'] for c in sent['candidates']] for sent in results]
    cand_scores    = [[c['score'] for c in sent['candidates']] for sent in results]
    
    has_filter = 'unfiltered_candidates' in results[0].keys()
    if has_filter:
        unfiltered_cands = [sent['unfiltered_candidates'] for sent in results]
        n_cands = len(unfiltered_cands[0])
    else:
        n_cands = len(cands[0])

    code_metrics = [m for m in metrics if 'pass@' in m.name]
    metrics      = [m for m in metrics if 'pass@' not in m.name]

    scores = evaluate(
        src, pred, ref, metrics, 
        cands=cands, cand_scores=cand_scores, include_oracle=include_oracle
    )
    
    if task == 'code': 
        assert has_filter
        code_scores = code_evaluate(pred, metadata, code_metrics, unfiltered_cands=None)
        scores.update(code_scores)

    all_scores = {f's={n_cands}': scores}

    # Re-compute MBR on a subset of generations (slow!)
    if subset_eval:
        N_SUBSET_EVAL = (20 if task == 'code' else 5) # Number of times to repeat subset eval
        n_filtered = count_filtered(unfiltered_cands) if has_filter else None

        score_matrices = [sent['score_matrix'] for sent in results]

        subset_sizes = [n_cands]
        if n_cands >= 10:  subset_sizes = [1, 2, 4, 6, 8]
        if n_cands >= 20:  subset_sizes = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        if n_cands >= 100:  subset_sizes = [1, 2, 5, 10, 20, 40, 60, 80]
        if n_cands >= 500:  subset_sizes = [1, 2, 5, 10, 20, 40, 60, 80, 100, 200, 300, 400]

        for s in subset_sizes:
            print(f'Evaluating at subset size {s}...')
            for e in range(N_SUBSET_EVAL):
                subset_pred, subset_cands, subset_cands_scores = subset_selection(score_matrices, cands, s, n_filtered)
                scores = evaluate(
                    src, subset_pred, ref, metrics, 
                    cands=subset_cands, cand_scores=subset_cands_scores, include_oracle=include_oracle
                )
                if task == 'code': 
                    # unfiltered_cands = [sent['unfiltered_candidates'] for sent in results] # <- needs to use subset
                    code_scores = code_evaluate(subset_pred, metadata, code_metrics, unfiltered_cands=None)
                    scores.update(code_scores)
                all_scores[f's={s}-e={e}'] = scores

    return all_scores


def clean_metrics(metrics):
    """ Clean up metric memory usage """
    for metric in metrics:
        if hasattr(metric, 'metric'): 
            del metric.metric
        del metric
    torch.cuda.empty_cache()
