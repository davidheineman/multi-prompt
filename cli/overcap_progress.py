import os, json, sys

from utils.constants import OVERCAP_DIR

def overcap_progress():
    for root, _, _ in os.walk(OVERCAP_DIR):
        root_name = root.split("/")[-1]
        pad = max(100, len(root_name))
        print('='*pad + f'\n=== Results in {root_name} ' + '='*(pad - 16 - len(root_name)) + '\n' + '='*pad)
        
        results_files = sorted([f for f in os.listdir(root) if f.endswith(".json")])
        for result_file in results_files:
            run_name = '-'.join(result_file.split('-')[:-5])
            result_path = os.path.join(root, result_file)
            
            results = None
            with open(result_path, 'r', encoding='utf-8') as f:
                try:
                    results = json.load(f)
                except UnicodeDecodeError as e:
                    raise RuntimeError(f'Unicode decoding error: {result_path}: {e}')
            
                result_ids = [s['metadata']['_id'] for s in results]

            if not os.path.exists(os.path.join(root, f'{run_name}.progress')):
                continue
            with open(os.path.join(root, f'{run_name}.progress'), 'r', encoding='utf-8') as f:
                claimed_ids = set([int(i) for i in f.read().splitlines()])

            completed_ids = [_id for _id in claimed_ids if _id in result_ids]

            print(f"{len(completed_ids)} / {len(claimed_ids)}: {run_name}")
            if len(completed_ids) != len(claimed_ids):
                print(f' - {sorted(list(set(claimed_ids) - set(completed_ids)))}')

                # Flush all non-completed ids
                if "--flush" in sys.argv:
                    print(f'Flushing ids for {run_name}...')
                    with open(os.path.join(root, f'{run_name}.progress'), "r+", encoding='utf-8') as f:
                        f.seek(0)
                        f.write('\n'.join([str(id_) for id_ in sorted(completed_ids)]))
                        f.truncate()

            if "--finalize" in sys.argv:
                if os.path.exists(os.path.join(root, f'{run_name}.progress')):
                    print(f'Deleting progress file for {run_name}...')
                    os.remove(os.path.join(root, f'{run_name}.progress'))
                if os.path.exists(os.path.join(root, f'{result_file[:-5]}.lock')):
                    print(f'Deleting lock file for {run_name}...')
                    os.remove(os.path.join(root, f'{result_file[:-5]}.lock'))
        print('\n')

if __name__ == "__main__":
    overcap_progress()
