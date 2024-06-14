import random
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer


def embed_prompts(prompts):
    model = SentenceTransformer('all-mpnet-base-v2')
    return model.encode(prompts)


def get_closest_points(embedding, p=10):
    """
    Given an (n x d) embedding matrix, return the indices of the k closest points
    """
    kd = KDTree(embedding)
    min_dist, min_idx = float('inf'), None
    for query_point in embedding:
        d, idx = kd.query([query_point], k=p)
        total_dist = np.sum(d)
        if total_dist < min_dist:
            min_dist, min_idx = total_dist, idx
    return min_idx[0].tolist(), min_dist


def get_furthest_points(embedding, p=10):
    """
    Given an (n x d) embedding matrix, return the indices of the k furthest away points
    """
    max_dist, max_idx = float('-inf'), None
    for query_point in embedding:
        max_idx_pt, max_dist_pt = [], []
        for i, pt in enumerate(embedding):
            max_dist_pt += [np.linalg.norm(pt - query_point)]
            max_idx_pt += [i]
            if len(max_idx_pt) > p:
                min_dist_idx = np.argmin(max_dist_pt)
                max_idx_pt.pop(min_dist_idx)
                max_dist_pt.pop(min_dist_idx)
        total_dist = np.sum(max_dist_pt)
        if total_dist > max_dist:
            max_dist, max_idx = total_dist, max_idx_pt
    return max_idx, max_dist


def cluster_prompts(prompts, embedding, selection_method, p=10, prompt_weights=None):
    """
    Fits a k-NN classifier to prompt embeddings and selects from each cluster. Use selection_method
    to specify the cluster selection method.
    """
    kmeans = KMeans(n_clusters=p)
    kmeans.fit(embedding)

    selected_prompts, prompt_clusters = [], []
    for i in set(kmeans.labels_):
        prompt_cluster = [(j, p) for j, p in enumerate(prompts) if kmeans.labels_[j] == i]
        prompt_clusters += [prompt_cluster]

        if selection_method == 'random':
            selected_prompts += [random.choice(prompt_cluster)]
        elif selection_method == 'performance':
            if prompt_weights is None:
                raise ValueError('You must have prompt weights to use the performance cluster selection method.')
            selected_prompts += [max(prompt_cluster, key=lambda x: prompt_weights[x[0]])]
        elif selection_method == 'furthest' or selection_method == 'closest':
            continue
        else:
            raise ValueError(f'Invalid cluster selection method: {selection_method}')

    # I'll admit I don't have an efficient method to compute this. It almost feels like a mix of
    # subset sum (NP-complete) and minimum distance (O(nlogn))
    # "given 10 sets with 10 points, select 1 point from each set such that the selection of points is the closest together"
    if selection_method == 'furthest':
        raise NotImplementedError()
    elif selection_method == 'closest':
        raise NotImplementedError()

    return [prompt for _, prompt in selected_prompts]


def sample_prompts(instruction, k, p, weight_method='uniform', prompt_weights=None, n=0.6):
    """
    Given a set of instructions and hyperparameters, sample a set of p*k prompts
    """
    # Sample p*k prompts from prompt bank
    if weight_method != 'uniform':
        assert prompt_weights is not None, f'To use a non-uniform prompt weighting, you must pass not None. Seeing "{prompt_weights}"'
        assert len(instruction) == len(prompt_weights), 'You must have an equal number of prompts and prompt weights!'

    if isinstance(instruction, list):
        if len(instruction) == 1:
            prompts = [instruction[0] for _ in range(p*k)]
        if weight_method == 'uniform':
            prompts = random.sample(instruction, p) # Select without replacement
            prompts = [prompt for prompt in prompts for _ in range(k)]
        elif weight_method == 'best':
            # Get highest scoring prompt and use for all exemplars
            best_prompt = instruction[prompt_weights.index(max(prompt_weights))]
            prompts = [best_prompt for _ in range(p*k)]
        elif 'top-n' in weight_method:
            # Order both lists by prompt weights
            sorted_instruction, sorted_prompt_weights = zip(*sorted(list(zip(instruction, prompt_weights)), key=lambda x: x[1], reverse=True))

            # Get top instructions using CDF -- i.e. Instructions with top-n of probability mass
            cdf = np.cumsum(sorted_prompt_weights) / np.sum(sorted_prompt_weights)
            top_instructions = [sorted_instruction[i] for i in range(0, np.argmax(cdf >= n))]
            top_prompt_weights = [sorted_prompt_weights[i] for i in range(0, np.argmax(cdf >= n))]

            # Sample from the restricted instruction set
            if weight_method == 'top-n':
                prompts = random.choices(top_instructions, k=p*k) # Select with replacement
            elif weight_method == 'top-n-ensemble':
                prompts = random.choices(top_instructions, top_prompt_weights, k=p*k)
            else:
                raise ValueError(f'Weight method "{weight_method}" not supported!')
        elif weight_method == 'ensemble':
            prompts = random.choices(instruction, prompt_weights, k=p*k)
        else:
            raise ValueError(f'Weight method "{weight_method}" not supported!')
    else:
        prompts = [instruction for _ in range(p*k)]

    assert p * k == len(prompts), f"Using p={p}, k={k}, but the number of prompts are {len(prompts)}"

    return prompts