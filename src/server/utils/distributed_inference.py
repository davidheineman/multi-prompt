import datetime, copy, os, sys
import torch
import torch.distributed as dist
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

os.environ['NCCL_P2P_DISABLE'] = '1'

def setup_model_parallel():
    global local_rank, world_size, device

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    device = f'cuda:{local_rank}'

    dist.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # Disable torch gradient (this way we don't have to wrap our code in torch.no_grad)
    torch.set_grad_enabled(False)

    # Silence outputs of non-head processes
    # if local_rank > 0:
    #     sys.stdout = open(os.devnull, "w")

    # This is a super interesting option. In theory we should use this because all the generators
    # will be using the same seed, but because the code would be executed the EXACT same way on all
    # GPUs (except GPU 0), it will actually give the same output for all GPUs (even if sampling params
    # are stochastic, which we don't want).
    # torch.manual_seed(1)

    return local_rank, world_size


def run_metric(data, metric_func):
    """
    Evaluate a metric function and report the evaluation time.
    """
    start_time = datetime.datetime.now()

    print(f'Evaluating {len(data)} examples on cuda:{local_rank}.')

    scores = metric_func(data)

    torch.cuda.empty_cache()

    # Measure evaluation time
    duration = (datetime.datetime.now() - start_time).total_seconds()
    print(f"Evaluated {len(data)} examples in {duration:.2f}s at {len(data)/duration:.2f} sent/s on cuda:{local_rank}.")
    
    return scores


def generate_distributed(data, generate_func):
    """
    Initalize a root process for distributed inference. This will orchestrate distributed inference across child
    processes on each GPU.
    """
    # Create a new '_dist' field, which contains the assignment for each GPU to generate
    if isinstance(data, dict):
        data['_dist'] = []
        # Get a list input. E.g., "complex", "source"
        list_key = [key for key, value in data.items() if isinstance(value, list)][0]
        print(f'Splitting distributed inference on key "{list_key}" between world size {world_size}')
        for i, d in enumerate(data[list_key]): 
            data['_dist'] += [{
                'id': i,
                'output': None,
                'device': f'cuda:{i % world_size}'
            }]
    elif isinstance(data, list):
        assert isinstance(data[0], dict), f"List input must be a list of dicts! Recieved: {data}"
        for i, d in enumerate(data): 
            d['_dist'] = {
                'id': i,
                'output': None,
                'device': f'cuda:{i % world_size}'
            }
    else:
        raise ValueError(f'Invalid data format: {type(data_subset)}')
        
    data = [data]

    # Send data to all GPUs
    dist.broadcast_object_list(data, src=0)

    # Generate for GPU 0
    data_subset = copy.deepcopy(data[0])
    if isinstance(data_subset, dict):
        data_subset_ids = [d['id'] for d in data_subset['_dist'] if d['device'] == device]
        iter_entries = [k for k in data[0].keys() if not k.startswith('_')] if 'input_text' not in data[0].keys() else ['input_text']
        for entry_name in iter_entries:
            data_subset[entry_name] = [sent for i, sent in enumerate(data[0][entry_name]) if data[0]['_dist'][i]['device'] == device]
        scores = generate_func(**{k: v for k, v in data_subset.items() if not k.startswith('_')})
        for idx, g in zip(data_subset_ids, scores): 
            data[0]['_dist'][idx]['output'] = g
    elif isinstance(data_subset, list):
        data_subset_ids = [data_subset[i]['_dist']['id'] for i in range(len(data_subset)) if data_subset[i]['_dist']['device'] == device]
        data_subset = [sent for i, sent in enumerate(data[0]) if data[0][i]['_dist']['device'] == device]
        scores = generate_func(data_subset)
        for idx, g in zip(data_subset_ids, scores): 
            data[0][idx]['_dist']['output'] = g
    
    # Wait for a response from all GPUs
    ack = [[None] for _ in range(world_size)]
    ack[local_rank] = data
    for lr in range(1, world_size):
        # print(f'{device} is waiting for ack from {lr}')
        dist.broadcast_object_list(ack[lr], src=lr)

    # Reset data
    data = [None]
    dist.broadcast_object_list(data, src=0)

    # Process the data from each GPU into a list of strings
    if isinstance(ack[0][0], dict):
        full_results = [gen for gen in [i for j in [d[0]['_dist'] for d in ack] for i in j] if gen['output'] is not None]
        final_scores = [g['output'] for g in sorted(full_results, key=lambda x: x['id'])]
    elif isinstance(ack[0][0], list):
        full_results = [gen for gen in [i['_dist'] for j in [d[0] for d in ack] for i in j] if gen['output'] is not None]
        final_scores = [g['output'] for g in sorted(full_results, key=lambda x: x['id'])]

    return final_scores
    

def generate_distributed_child(generate_func):
    """
    Initalize a child process for distributed inference. This will run on all GPUs other than the head node.
    """
    data = [None]

    # Wait for data from GPU 0s
    dist.broadcast_object_list(data, src=0)

    if data[0] is None:
        return

    # Create a copy of the data object for generation on the GPU (may not be necessary)
    data_subset = copy.deepcopy(data[0])
    if isinstance(data_subset, dict):
        data_subset_ids = [d['id'] for d in data_subset['_dist'] if d['device'] == device]
        iter_entries = [k for k in data[0].keys() if not k.startswith('_')] if 'input_text' not in data[0].keys() else ['input_text']
        for entry_name in iter_entries:
            data_subset[entry_name] = [sent for i, sent in enumerate(data[0][entry_name]) if data[0]['_dist'][i]['device'] == device]
        if len(iter_entries[0]) > 0: # all([len(e) > 0 for e in iter_entries[0]]):
            scores = generate_func(**{k: v for k, v in data_subset.items() if not k.startswith('_')})
            for idx, g in zip(data_subset_ids, scores): 
                data[0]['_dist'][idx]['output'] = g
    elif isinstance(data_subset, list):
        data_subset_ids = [data_subset[i]['_dist']['id'] for i in range(len(data_subset)) if data_subset[i]['_dist']['device'] == device]
        data_subset = [sent for i, sent in enumerate(data[0]) if data[0][i]['_dist']['device'] == device]
        scores = generate_func(data_subset)
        for idx, g in zip(data_subset_ids, scores): 
            data[0][idx]['_dist']['output'] = g

    # Wait for GPUs 1 ... local_rank to finish generating
    # i.e., Generation goes in order from 1, 2, 3, ..., 8, 0
    ack = [[None] for _ in range(world_size)]
    for lr in range(1, local_rank):
        # print(f'{device} is waiting for ack from {lr}')
        dist.broadcast_object_list(ack[lr], src=lr)

    # Push results for this GPU to others
    ack[local_rank] = data
    dist.broadcast_object_list(ack[local_rank], src=local_rank)

    # Wait for later GPUs to finish generating
    for lr in range(local_rank+1, world_size):
        # print(f'{device} is waiting for ack from {lr}')
        dist.broadcast_object_list(ack[lr], src=lr)