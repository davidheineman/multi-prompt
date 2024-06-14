import os, json, signal, threading
from filelock import FileLock, Timeout

from utils.constants import OVERCAP_DIR
from src.util import get_time

SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", 0)
OVERCAP_BATCH_SIZE = 10

REQUEUE = threading.Event()
REQUEUE.clear()
EXIT = threading.Event()
EXIT.clear()


def _requeue_job():
    print(f"Requeuing job {SLURM_JOB_ID}...")
    os.system(f"scontrol requeue {SLURM_JOB_ID}")


def _requeue_handler(signum, frame):
    print("REQUEUE SIGNAL recieved, attempting to requeue...")
    EXIT.set()
    REQUEUE.set()


def _clean_exit_handler(signum, frame):
    EXIT.set()
    print("EXIT SIGNAL recieved, exiting cleanly...", flush=True)


def init_handlers():
    print('Initializing overcap interrupt signals...')
    signal.signal(signal.SIGUSR1, _requeue_handler)
    signal.signal(signal.SIGINT, _clean_exit_handler)
    signal.signal(signal.SIGTERM, _clean_exit_handler)
    signal.signal(signal.SIGUSR2, _clean_exit_handler)


def flush_data_ids(run_prefix, results, batch, parent_dir=OVERCAP_DIR):
    """
    On a graceful exit, remove all ids of sentence that could not be added from the
    [RUN_NAME].progress file.
    """
    print(f'Flushing data ids...')
    
    # Get claimed IDs at progress path
    existing_jobs = [f for f in os.listdir(parent_dir) if '.progress' in f and run_prefix == os.path.splitext(f)[0]]
    assert len(existing_jobs) == 1 # Make this more elegant
    progress_path = existing_jobs[0]
    progress_path = os.path.join(parent_dir, progress_path)

    with open(progress_path, "r+", encoding='utf-8') as f:
        all_ids = [int(i) for i in f.read().splitlines()]
        claimed_ids = [s['metadata']['_id'] for s in batch]
        completed_ids = [s['metadata']['_id'] for s in results]

        all_completed_ids = sorted(list(set(all_ids) - (set(claimed_ids) - set(completed_ids))))

        f.seek(0)
        f.write('\n'.join([str(id_) for id_ in sorted(all_completed_ids)]))
        f.truncate()


def add_results(run_prefix, results, parent_dir=OVERCAP_DIR):
    """
    Load an existing results file and add all IDs which don't exist in the results file.
    """
    existing_jobs = [f for f in os.listdir(parent_dir) if '.json' in f and run_prefix in f]

    if len(existing_jobs) > 0:
        job_path = existing_jobs[0]
    else: 
        job_path = f'{run_prefix}-{get_time()}.json'
    job_path = os.path.join(parent_dir, job_path)

    # Create results file if it does not exist
    if not os.path.exists(job_path):
        with open(job_path, 'w', encoding='utf-8') as f: 
            json.dump([], f)
    
    print(f'Found {len(existing_jobs)} existing jobs, saving to {job_path}...')

    try:
        with FileLock(job_path.replace('.json', '.lock'), timeout=10):
            with open(job_path, 'r+', encoding='utf-8') as f:
                # Load results file
                existing_results = json.load(f)
                existing_ids = [s['metadata']['_id'] for s in existing_results]

                # Paste in existing results (ids which don't currently exist)
                new_results = [s for s in results if s['metadata']['_id'] not in existing_ids]
                results = sorted(existing_results + new_results, key=lambda s: s['metadata']['_id'])

                f.seek(0)
                json.dump(results, f, ensure_ascii=False) # indent=4, 

                # Reload results to ensure overwiting has not occured
                f.seek(0)
                try:
                    results_reloaded = f.read()
                except (OSError, json.decoder.JSONDecodeError) as e:
                    print(f'Corrupted JSON file detected! Rewriting...')
                    f.seek(0)
                    json.dump(results, f, ensure_ascii=False)
    except Timeout:
        print(f"Lock timeout. Could not edit: {job_path.replace('.json', '.lock')}")


def claim_data_ids(run_prefix, data, parent_dir=OVERCAP_DIR):
    """
    Given a prefix, claims a subset of data ids

    Updates/creates a [RUN_NAME].progress file of any claimed or completed task ids
    """
    # Get all task ids within dataset
    task_ids = [s['metadata']['_id'] for s in data]

    # Check if the prefix exists in OVERCAP_DIR
    existing_jobs = [f for f in os.listdir(parent_dir) if '.progress' in f and run_prefix == os.path.splitext(f)[0]]

    # If it does, then create or load a [RUN_NAME].progress file, which is a text file in utf-8 which contains claimed IDs
    claimed_ids = []
    if len(existing_jobs) > 0:
        progress_path = os.path.splitext(os.path.basename(existing_jobs[0]))[0]
    else:
        progress_path = run_prefix
    progress_path = os.path.join(parent_dir, f'{progress_path}.progress')
    if not os.path.exists(progress_path):
        with open(progress_path, 'w', encoding='utf-8') as f: 
            pass
    with open(progress_path, "r+", encoding='utf-8') as f:
        claimed_ids = [int(i) for i in f.read().splitlines()]

        # Get a set of task_ids from the dataset, "claim" the lowest task ids which don't exist in the progress file
        unclaimed_ids = list(set(task_ids) - set(claimed_ids))
        ids_to_claim = unclaimed_ids[:OVERCAP_BATCH_SIZE]

        print(f'Claiming task ids: {ids_to_claim}...')
        claimed_ids += ids_to_claim

        f.seek(0)
        f.write('\n'.join([str(id_) for id_ in sorted(claimed_ids)]))
        f.truncate()

    batch = [s for s in data if s['metadata']['_id'] in ids_to_claim]
    return batch


def check_results_complete(run_prefix, results, data, parent_dir=OVERCAP_DIR):
    return False


def overcap_handle_iter(run_prefix, data, batch, i, results, parent_dir=OVERCAP_DIR):
    try:
        if REQUEUE.is_set():
            # Save progress and delete any IDs which couldn't be run
            add_results(run_prefix, results, parent_dir=parent_dir)
            flush_data_ids(run_prefix, results, batch, parent_dir=parent_dir)
            _requeue_job()

        if EXIT.is_set():
            add_results(run_prefix, results, parent_dir=parent_dir)
            flush_data_ids(run_prefix, results, batch, parent_dir=parent_dir)
            raise RuntimeError(f'Exit signal recieved, exiting...')

        if i == len(batch) - 1:
            if len(batch) != len(results):
                print(f'Warning! Not all sentences may have been generated (seeing {len(results)}/{len(batch)})')
            
            # Save progress and delete any IDs which couldn't be run
            add_results(run_prefix, results, parent_dir=parent_dir)
            flush_data_ids(run_prefix, results, batch, parent_dir=parent_dir)

            # Try to request another batch
            batch = claim_data_ids(run_prefix, data, parent_dir=parent_dir)
            results, i = [], -1

            # If another batch is not available, check if the results file is complete
            if len(batch) == 0:
                print('No more batches! Checking if results are complete...')

                if check_results_complete(run_prefix, results, data, parent_dir=parent_dir):
                    print('All results are complete')

                    # If the results file is complete, move it to the results folder
                    # Ideally we can queue a job here to do subset eval on the file

                    raise NotImplementedError()
                else:
                    print('Not all results are complete. Moving to next job')
        return batch, results, i
    except json.decoder.JSONDecodeError as e:
        print(f'Ran into JSON decoding error for {run_prefix}: {e}. Moving to next job')
        return [], [], -1
