import os, json

from utils.constants import RESULTS_DIR
from src.eval import evaluate_results_file, get_task_metrics, clean_metrics
from src.util import get_time, render_table


def evaluate_folder(folder_path, task, include_oracle=False, subset_eval=False):
    print(f'Loading metrics for {task}...')
    metrics = get_task_metrics(task)

    output_path = os.path.join(RESULTS_DIR, f'eval-{get_time()}.json')
    evaluation = {}

    json_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json') and not f.startswith('eval')])

    for f in json_files:
        if task not in f: continue

        print(f"Evaluating {f} with metrics {metrics}...")

        results = evaluate_results_file(os.path.join(folder_path, f), task, metrics, include_oracle=include_oracle, subset_eval=subset_eval)
        for result_label, result in results.items():
            result_label = f'{os.path.splitext(f)[0]}-{result_label}' if len(results) >= 1 else os.path.splitext(f)[0]
            evaluation.update({result_label: result})

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation, f, indent=4, ensure_ascii=False)

        print(render_table(evaluation))

    clean_metrics(metrics)
    

if __name__ == "__main__":
    EVAL_PATH = os.path.join(RESULTS_DIR)

    result_paths = [
        '',
    ]
    for path in result_paths:
        evaluate_folder(os.path.join(EVAL_PATH, path), 'simplification', subset_eval=True)

    result_paths = [
        '',
    ]
    for path in result_paths:
        evaluate_folder(os.path.join(EVAL_PATH, path), 'translation', subset_eval=True)

    result_paths = [
       ''
    ]
    for path in result_paths:
        evaluate_folder(os.path.join(EVAL_PATH, path), 'code', subset_eval=True)
