import numpy as np
from typing import Union, Callable, Optional, List

from src.util import extract_prefixes
from src.metrics import AbstractMetric


def mbr(metric: AbstractMetric, cands: List[str], source: List[str]):
    """
    Calculate MBR using a reference-based value metric.
    """
    if len(cands) == 1: return [1]

    # Create a pairwise list of candidates for evaluation
    candidate_matrix = []
    for i in range(len(cands)):
        for j in range(len(cands)):
            candidate_matrix += [(cands[i], cands[j])]
    assert len(candidate_matrix) == len(cands)**2

    # Score list of candidates
    sources    = [source for _ in candidate_matrix]
    targets    = [i for (i, j) in candidate_matrix]
    references = [[j] for (i, j) in candidate_matrix]

    candidate_matrix_scores = metric(src=sources, pred=targets, ref=references)

    # Restore MBR as a 2D matrix
    score_matrix = np.zeros((len(cands), len(cands)))
    cnt = 0
    for i in range(len(cands)):
        for j in range(len(cands)):
            score_matrix[i][j] = candidate_matrix_scores[cnt]
            cnt += 1

    return np.mean(score_matrix, axis=1).tolist(), score_matrix


def rerank(metric: AbstractMetric, cands: List[str], source: List[str]):
    """
    Score a list of candidates using reference-free re-ranking
    """
    if len(cands) == 1: return [1]
    
    sources = [source for _ in cands]
    targets = cands

    assert len(sources) == len(targets)

    candidate_scores = metric(src=sources, pred=targets)

    return candidate_scores, candidate_scores


def mbr_exec(cands, source, metadata, first_assert_only=False):
    """
    Execution-based MBR. This is slightly different from mbr() because results are 
    computed prior to the MBR calcuation, unlike pairwise interaction.

    first_assert_only = Only use the first assert case in the docstring
    """
    from src.tasks.human_eval import extract_docstring, extract_test 
    from metrics.human_eval.mbr_exec import humaneval_postprocess, humaneval_execute_one_assertion, humaneval_execute_multiple_assertion

    pid, entry_point = metadata['task_id'], metadata['entry_point']

    # Extract the docstring and test case for a HumanEval example
    docstring, _, _, _ = extract_docstring(source)
    tests = extract_test(pid, entry_point, docstring)

    # Given a generation, run the test case to get a solution, or return an error type
    # status: 0 = successfully ran, 1 = could not complete
    statuses, results = [], []
    for generation in cands:
        completion = humaneval_postprocess(generation)
        if first_assert_only:
            status, result = humaneval_execute_one_assertion(source, completion, pid, tests[0])
            statuses += [status]
            results += [result]
        else:
            out = humaneval_execute_multiple_assertion(source, completion, pid, tests)
            statuses += [[o[0] for o in out]]
            results += [[o[1] for o in out]]

    # MBR calculation
    score_matrix = np.zeros((len(cands), len(cands)))
    for i in range(len(cands)):
        for j in range(len(cands)):
            if first_assert_only:
                # For only the first assertion
                if i < j and statuses[i] == 0 and statuses[j] == 0 and results[i] == results[j]:
                    score_matrix[i][j] = score_matrix[j][i] = 1
            else:
                # For multiple assertions
                if i < j:
                    for k in range(len(results[i])):
                        if statuses[i][k] == 0 and statuses[j][k] == 0 and results[i][k] == results[j][k]:
                            score_matrix[i][j] += 1
                            score_matrix[j][i] = score_matrix[i][j]

    return np.sum(score_matrix, axis=1).tolist(), score_matrix


def rerank_code_review(generate_f, prompts, cands, source, cand_scores, metadata, normalized=True, use_source=True):
    """
    Code re-ranking by generating a docstring using the generated code as a prior, then
    calculates a final score by comparing cum_log_probs. Different from rerank() as it
    requires a second call to the generation endpoint.

    Our implementation is slightly different from Zhang et al., 2022, specifically that we
    calculate both the generation and docstring log probs on the second pass, and calculate
    the final score using the probs before and after the docstring.

    Additionally `use_source` will use the original HumanEval docstring, rather than using
    the prompt written for each example.
    """
    from metrics.human_eval.filter import clean_code
    from metrics.human_eval.reranker import make_new_context, postprocess_func_only

    entry_point = metadata['entry_point']

    assert len(prompts) == len(cands), prompts

    docstring_prompts = []
    for prompt, generation in zip(prompts, cands):
        # Given a generation, clean up using minifcation
        try:
            cleaned = clean_code(generation)
        except Exception as e:
            print(f'Recieved {e}. Could not clean candidate: \n{generation}\nSkipping...')
            cleaned = 'bad generation'

        # cleaned = postprocess_func_only(generation, tokens)
        # logprobs = logprobs[:len(tokens)]

        # Given a generation, create the new prompt to generate a docstring
        if use_source:
            with_ref_prompt, _ = make_new_context(
                source, cleaned, entry_point, canonicalize=True, clean_print=True
            )
        else:
            with_ref_prompt, _ = make_new_context(
                prompt, cleaned, entry_point, canonicalize=True, clean_print=True
            )
        docstring_prompts += [with_ref_prompt]

    assert len(docstring_prompts) == len(cands), docstring_prompts
        
    # Generate conditioned on the program
    _, with_ref_response_scores = generate_f(
        _prompt=docstring_prompts, max_new_tokens=1, temperature=1, 
        return_candidates_override=True, output_scores=True
    )

    # We use the prompt itself for segementation, so we need to extract the prompt token IDs
    if not hasattr(rerank_code_review, "prompt_toks"):
        print(f'Calculating the prompt tokens for later use...')
        REVERSE_PROMPT = "\n\n# write the docstring for the above function\n"
        _, prompt_scores = generate_f(
            _prompt=[REVERSE_PROMPT], max_new_tokens=1, temperature=1, 
            return_candidates_override=True, output_scores=True
        )
        rerank_code_review.prompt_toks = prompt_scores[0]['prompt_token_ids'][2:-1]
    prompt_toks = rerank_code_review.prompt_toks

    final_scores = []
    for scores in with_ref_response_scores:
        # Segment the sections before and after the prompt
        all_toks, all_probs = scores['prompt_token_ids'], scores['prompt_logprobs']
        assert len(all_toks) == len(all_probs)
        all_probs = [list(p.values())[0]['logprob'] if p is not None else 0 for p in all_probs]
        prompt_idx = next((i for i in range(len(all_toks) - len(prompt_toks) + 1) if all_toks[i:i + len(prompt_toks)] == prompt_toks), -1)
        if prompt_idx == -1:
            raise RuntimeError(f'Could not find prompt within docstring generation tokens: {scores}')
        func_toks, docstring_toks = all_toks[:prompt_idx], all_toks[prompt_idx+len(prompt_toks):]
        func_probs, docstring_probs = all_probs[:prompt_idx], all_probs[prompt_idx+len(prompt_toks):]
        
        # Coder-Reviewer Reranking score
        if normalized:
            score = (sum(func_probs) / len(func_toks)) + (sum(docstring_probs) / len(docstring_toks))
        else:
            score = sum(func_probs) + sum(docstring_probs)

        final_scores += [score]

    assert len(final_scores) == len(cands), final_scores

    return final_scores, final_scores


def generate_beam(prompt, generate_sentence_func, **kwargs):
    """
    Perform beam search runs for each unique value of k. 
    
    Sends an individual beam search request for each unique number of beams. For example, if each prompt occurs equally,
    only one request is given.
    """
    results = []
    instructions, input_text = extract_prefixes(prompt)
    value_counts = {s: instructions.count(s) for s in instructions}
    for inst_count in set(value_counts.values()):
        inst = [k for k, v in value_counts.items() if v == inst_count]
        kwargs.update({
            'num_beams': inst_count,
            'num_return_sequences': inst_count,
        })
        results += generate_sentence_func(prompt=[i + input_text for i in inst], **kwargs)
    return results


def reduce_score_matrices(score_matrices, cands):
    """ 
    Uses a list of score matrices to narrow down the candidate set according to a threshold.

    In practice, it would be more efficient to cut the candidate set and evaluate on the red-
    uced size, but this setup allows saving all metric scores for subset evaluation. 
    """
    score_matrix = np.array(score_matrices[0])
    for i in range(len(score_matrices)):
        # Cutoff top candidates in list
        THRESHOLD = 0.4  # i.e. Top 40% scoring candidates

        # Calculate matrix scores
        is_2d = score_matrix.ndim == 2
        scores = np.mean(score_matrix, axis=1) if is_2d else score_matrix

        # Get top scores according to threshold cutoff
        sorted = np.argsort(scores)[::-1]
        ind = sorted[:int(max(1, len(sorted) * THRESHOLD))]

        # Get new metatdata with only top scores
        cands  = [cands[i_] for i_ in ind]
        scores = [scores[i_] for i_ in ind]
        generation = cands[0]

        # Prepare next metric narrowing
        if i + 1 < len(score_matrices):
            next_score_matrix = np.array(score_matrices[i+1])
            is_2d = next_score_matrix.ndim == 2
            next_score_matrix = next_score_matrix[ind][:, ind] if is_2d else next_score_matrix[ind]

            print(f'Narrowing {score_matrix.shape} -> {next_score_matrix.shape}')

            score_matrix = next_score_matrix

        print(scores)
    
    return generation, cands, scores


def candidate_argmax(
        source: List[str], 
        prompt: List[str], 
        generate_sentence_func: Callable, 
        metrics: Union[AbstractMetric, List[AbstractMetric]], 
        return_scores: bool = False, 
        text_postprocessing_func: Optional[Callable] = None, 
        cand_filter_func: Optional[Callable] = None, 
        metadata: Optional[dict] = None, 
        mbr_type: str = 'efficient',
        **kwargs
    ):
    """
    Run all candidates through a scoring function and get the highest scoring candidate. If metrics is a list
    it will iteratively narrow down the candidate space using the list
    """
    scores = []

    if not isinstance(metrics, list): metrics = [metrics]
    
    def generate_f(_prompt, **additional_kwargs):
        gen_params = kwargs.copy()
        gen_params.update(additional_kwargs)

        # For code reranker, in cases where we want to use the generation function during candidate subset re-ranking
        if 'return_candidates' in gen_params and 'return_candidates_override' in gen_params:
            del gen_params['return_candidates']
        if 'return_candidates_override' in gen_params:
            del gen_params['return_candidates_override']

        if 'return_candidates' in gen_params:
            # This argument will override generation and directly provide candidates, such as if we have
            # already generated candidates
            return gen_params['return_candidates'], None
        elif gen_params['do_sample']:
            cands = generate_sentence_func(prompt=_prompt, **gen_params)
        else:
            cands = generate_beam(prompt=_prompt, generate_sentence_func=generate_sentence_func, **gen_params)

        if 'output_scores' in gen_params:
            return [c[0] for c in cands], [c[1] for c in cands]
        
        return cands, None
    
    cands, cand_scores = generate_f(prompt)

    print(cands)

    # Custom text formatting code to apply to each candidate (e.g., cutting off after a newline)
    if text_postprocessing_func is not None:
        cands = [text_postprocessing_func(c) for c in cands]

    print(cands)

    # Custom candidate filtering (e.g., verifying whether a piece of code compiles)
    if cand_filter_func is not None:
        unfiltered_prompts, unfiltered_cands = prompt, cands
        is_filtered = [cand_filter_func(c) for c in unfiltered_cands]
        cands, prompt = [c for c, f in zip(unfiltered_cands, is_filtered) if f], [p for p, f in zip(unfiltered_prompts, is_filtered) if f]
        if len(cands) <= 0:
            print('All candidates were filered, reverting to original candidate set...')
            cands, prompt = unfiltered_cands, unfiltered_prompts
        assert len(cands) == len(prompt)

    score_matrices = []
    for metric in metrics:
        if metric.name == 'mbr_exec':
            scores, score_matrix = mbr_exec(cands, source, metadata)
        elif metric.name == 'code_reranker':
            scores, score_matrix = rerank_code_review(generate_f, prompt, cands, source, cand_scores, metadata)
        elif metric.requires_references:
            scores, score_matrix = mbr(metric, cands, source)
        else:
            scores, score_matrix = rerank(metric, cands, source)

        assert len(prompt) == len(cands) == len(scores)

        results = list(zip(prompt, cands, scores))

        # Sort results based on the MBR score
        # results = sorted(results, key=lambda x: x[2], reverse=True)
        # score_matrix = score_matrix[np.argsort(-np.sum(score_matrix, axis=1))].tolist()

        score_matrix = score_matrix.tolist() if isinstance(score_matrix, np.ndarray) else score_matrix
        score_matrices += [score_matrix]

    # Get highest scoring generation
    generation = max(results, key=lambda x: x[2])[1]

    # If multiple metrics were used, calculate the best candidate by reducing among candidates
    if len(score_matrices) > 1:
        generation, _, _ = reduce_score_matrices(score_matrices, cands)
    elif len(score_matrices) == 1: 
        score_matrices = score_matrices[0]
    
    if return_scores:
        candidates = [{
            'prompt': p,
            'candidate': c,
            'score': s
        } for p, c, s in results]

        unfiltered_candidates = None
        if cand_filter_func is not None:
            unfiltered_candidates = [{
                'prompt': p,
                'candidate': c,
                'rejected': f
            } for p, c, f in zip(unfiltered_prompts, unfiltered_cands, is_filtered)]
        return generation, candidates, score_matrices, unfiltered_candidates
    
    return generation, None, None, None
