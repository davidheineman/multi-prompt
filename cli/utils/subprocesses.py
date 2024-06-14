import requests, time, subprocess, atexit, os

from utils.constants import MODEL_PORT, METRIC_PORT, ROOT_DIR, SUBPROCESS_DIR
from utils.interrupt import SLURM_JOB_ID

PING_ENDPOINT = 'http://localhost:{port}/ping'
NO_GPU_METRICS = ['mbr_exec', 'code_reranker']

def wait_for_server(port, max_retries=50, delay=20): # max_retries=15
    retries = 0
    while retries < max_retries:
        try:
            requests.get(PING_ENDPOINT.format(port=port)).raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            print(f'Waiting for server at port {port} (retry {retries})...')
            pass

        retries += 1
        time.sleep(delay)
    
    raise RuntimeError(f"Max retries ({max_retries}) reached. Could not connect to the server.")


def init_server(load_model=True, load_metric=True):
    """
    Will spin up a model and metric endpoints as subprocesses and wait for them to load
    """
    processes = []      

    # Extract the torch port by appending the unique model port
    torch_port      = 29600 + int(str(MODEL_PORT)[-2:])
    full_model_name = str(os.environ.get("MODEL_NAME", ''))
    metric_name     = str(os.environ.get("METRIC_NAME"))
    use_vllm        = bool(os.environ.get("USE_VLLM", 'false').lower() == "true")
    root_log_dir    = os.path.join(ROOT_DIR, 'src', 'server', 'log')
    load_model      = load_model and full_model_name != ''

    if metric_name in NO_GPU_METRICS:
        print(f'Using a metric-free re-ranker: {metric_name}...')
        load_metric = False

    if 'openai' in full_model_name:
        print(f'Using an OpenAI API for generation, skipping model loading...')
        load_model = False

    # Get GPU IDs
    model_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if model_devices is not None:
        model_devices = [int(d) for d in model_devices.split(',')]
        if load_metric:
            model_devices = model_devices[1:]

    if load_model and use_vllm:
        print(f'Initalizing vLLM with devices: {model_devices}...')
        log_path = os.path.join(root_log_dir, 'overcap', f'vllm_endpoint_{SLURM_JOB_ID}.log') if SLURM_JOB_ID != 0 else os.path.join(root_log_dir, 'log', 'vllm_endpoint.log')
        processes += [subprocess.Popen(
            ['bash', '-c', f"{os.path.join(SUBPROCESS_DIR, 'run_vllm.sh')} {str(MODEL_PORT)} {full_model_name} {log_path} {','.join([str(d) for d in model_devices])}"],
            preexec_fn=os.setpgrp
        )]
    elif load_model:
        log_path = os.path.join(root_log_dir, 'overcap', f'model_endpoint_{SLURM_JOB_ID}.log') if SLURM_JOB_ID != 0 else os.path.join(root_log_dir, 'log', 'model_endpoint.log')
        processes += [subprocess.Popen(
            ['bash', '-c', f"{os.path.join(SUBPROCESS_DIR, 'run_model.sh')} {str(MODEL_PORT)} {full_model_name} {str(torch_port)} {log_path} {','.join([str(d) for d in model_devices])}"],
            preexec_fn=os.setpgrp
        )]

    if load_metric:
        log_path = os.path.join(root_log_dir, 'overcap', f'metric_endpoint_{SLURM_JOB_ID}.log') if SLURM_JOB_ID != 0 else os.path.join(root_log_dir, 'log', 'metric_endpoint.log')
        processes += [subprocess.Popen(
            ['bash', '-c', f"{os.path.join(SUBPROCESS_DIR, 'run_metric.sh')} {str(METRIC_PORT)} {metric_name} {str(torch_port+1)} {log_path}"],
            preexec_fn=os.setpgrp
        )]

    def exit_handler():
        print('Terminating processes...')
        for p in processes:
            if p: p.terminate()

    atexit.register(exit_handler)

    # Ping an endpoints before continuing
    if load_model: wait_for_server(MODEL_PORT)
    if load_metric: wait_for_server(METRIC_PORT)
