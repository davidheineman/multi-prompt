import statistics

BIGRAM_TOKENIZER = "bert-base-multilingual-uncased"

def unique_bigrams(list_of_candidates, candidate_scores=None, src=None, ref=None):
    """
    Calculate the average unique bigrams across a set of candidates
    """
    from transformers import AutoTokenizer
    from collections import Counter

    tokenizer = AutoTokenizer.from_pretrained(BIGRAM_TOKENIZER)

    total_unique_bigrams = []
    for cand_set in list_of_candidates:
        total_bigrams = []
        for cand in cand_set:
            tokens = tokenizer.tokenize(cand)
            total_bigrams += [(tokens[i], tokens[i+1]) for i in range(len(tokens) - 1)]
        total_unique_bigrams += [len(Counter(total_bigrams))]
    return statistics.mean(total_unique_bigrams)


def candidate_agreement(candidate_scores, list_of_candidates=None, src=None, ref=None):
    """
    Calculate the average MBR metric scores, averaged across all sentences. A higher total value
    metric score meant candidates were scored higher according to all other candidates.
    """
    return statistics.mean(statistics.mean(scores) for scores in candidate_scores)


def oracle(list_of_candidates, src, ref, metric):
    """
    Calculate the maximum possible score given all candidates 
    """
    scores = []
    for src, cands, ref in zip(src, list_of_candidates, ref):
        sources = [src for _ in cands]
        candidates = cands
        references = [ref for _ in cands]
        scores += [max(metric(src=sources, pred=candidates, ref=references))]
    return scores


def count_filtered(unfiltered_candidates):
    filt = []
    for cands in unfiltered_candidates:
        if any('rejected' not in c for c in cands):
            return None
        filt += [sum(c['rejected'] == False for c in cands) / len(cands)]
    return filt