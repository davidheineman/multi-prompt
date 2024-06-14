import statistics

import code_bert_score

from .human_eval.evaluation import evaluate_functional_correctness
from . import AbstractMetric


class CodeBertScore(AbstractMetric):
    def __init__(self, **kwargs):
        super().__init__(name='code_bert_score', requires_references=True, **kwargs)

    def __call__(self, src, pred, ref):
        _, _, f1, _ = code_bert_score.score(sources=src, cands=pred, refs=ref, lang='python')
        return [s*100 for s in f1.tolist()]
    

class FuncCorrect(AbstractMetric):
    def __init__(self, n_workers, variation='pass@1', **kwargs):
        super().__init__(name=variation, **kwargs)
        self.n_workers = n_workers

    def human_eval_acc(self, pred, metadata, k=[1]):
        results = [{'task_id': m['task_id'], 'completion': p} for p, m in zip(pred, metadata)]

        evaluation, metadata = evaluate_functional_correctness(
            model_outputs=results,
            n_workers=self.n_workers,
            k=k
        )

        # Get the reasons for failure among all candidates
        no_pass_avg = None
        if k[0] > 1:
            no_pass_cands = []
            COMPILE_ERRORS = ['NameError', 'UnboundLocalError']
            for task in metadata.values():
                passed, no_pass, no_compile = 0, 0, 0
                for cand in task:
                    output = cand[1]['result']
                    has_compile_error = any([o in output for o in COMPILE_ERRORS])
                    if not has_compile_error and 'passed' not in output: 
                        no_pass += 1
                    elif has_compile_error:
                        no_compile += 1
                    else:
                        passed += 1
                no_pass_cands += [no_pass]
            no_pass_avg = statistics.mean(no_pass_cands)
        
        return evaluation, no_pass_avg

    def human_eval_pass_k(self, unfiltered_pred, metadata):
        # Packages candidates into HumanEval evalaution input
        all_candidates = []
        for pred, meta in zip(unfiltered_pred, metadata):
            k = len(pred)
            for p in pred:
                all_candidates += [{
                    'target': p['candidate'],
                    'metadata': meta
                }]
        results = all_candidates

        print(f'Seeing k={k}...')
        evaluation, no_pass_avg = self.human_eval_acc(results, k=[k])

        if f'pass@{k}' not in evaluation: 
            raise RuntimeError(f'Cannot evaluate at k={k}. Perhaps you did not generate 164*k candidates? Received evalaution result: {evaluation}')
        
        evaluation = (evaluation[f'pass@{k}'] * 100).tolist()
        return evaluation, no_pass_avg

    def __call__(self, metadata, pred=None, unfiltered_pred=None):
        match self.name:
            case 'pass@1': 
                assert pred is not None, f'Need to pass predictions to cacluate pass@1, seeing pred={pred}'
                acc, _ = self.human_eval_acc(pred, metadata)
                acc = (acc['pass@1'] * 100).tolist()
                return acc
            case 'pass@k': 
                assert unfiltered_pred is not None, f'Need to pass unfiltered predictions to cacluate pass@1, seeing unfiltered_pred={unfiltered_pred}'
                pass_at_k, _ = self.human_eval_pass_k(unfiltered_pred, metadata)
                return pass_at_k
            case _: raise NotImplementedError(f'FuncCorrect variation {self.name} not supported!')
