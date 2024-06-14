import datetime, os

from flask import Flask, request, jsonify
from vllm import LLM, SamplingParams

DEFAULT_KWARGS = {
    'do_sample': True,
    'top_p': 0.9,
    'temperature': 0.9,
    'epsilon_cutoff': 0,
    'num_beams': 1,
    'num_return_sequences': 1,
    'max_new_tokens': 256,
    'return_dict_in_generate': False, 
    'output_scores': False
}

MODEL_NAME = str(os.environ.get("MODEL_NAME", 'meta-llama/Llama-2-7b-chat-hf'))
MODEL_PORT = int(os.environ.get("MODEL_PORT", 8500))
VLLM_DIR = str(os.environ.get("HF_HOME"))
DEVICES = [int(d) for d in os.environ.get("CUDA_VISIBLE_DEVICES", -1).split(',')]

FOUR_BIT_MODEL_MAP = {
    # "meta-llama/Llama-2-70b-chat-hf": "TheBloke/Llama-2-70b-Chat-AWQ",
    # "codellama/CodeLlama-70b-Python-hf": "TheBloke/CodeLlama-70B-Python-AWQ",
    # "codellama/CodeLlama-70b-Instruct-hf": "TheBloke/CodeLlama-70B-Instruct-AWQ",
}

app = Flask(__name__)

def quantize_model(model_name, output_path):
    """
    Save an HF model using AWQ quantization, which is compatible with vLLM
    """
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }
    model = AutoAWQForCausalLM.from_pretrained(model_name, **{"low_cpu_mem_usage": True})
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model.quantize(tokenizer, quant_config=quant_config)
    model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)


def setup():
    """
    Note, rather than use torchrun to start multiple endpoints in parallel and distribute across them,
    we use 1 endpoint which manages GPUs within the vLLM abstraction.
    """
    global MODEL, DEVICES

    if DEVICES == -1:
        raise RuntimeError(f'Ensure CUDA_VISIBLE_DEVICES is set! Seeing {DEVICES}')

    print(f'Loading model: {MODEL_NAME} to devices {DEVICES}...')

    # For code reranking, the vLLM's feature returning logprobs is not memory safe
    # so we need to reduce the GPU utilization of vLLM.
    is_code_override = 'CodeLlama-13b' in MODEL_NAME
    if is_code_override: print(f'Using safe GPU utilization!')

    is_olmo = 'OLMo' in MODEL_NAME
    if is_olmo: print(f'Using smaller model length.')

    if MODEL_NAME in FOUR_BIT_MODEL_MAP.keys():
        MODEL = LLM(
            model=FOUR_BIT_MODEL_MAP[MODEL_NAME], 
            dtype='float16',
            quantization="AWQ",
            tensor_parallel_size=len(DEVICES),
            max_model_len=(2048 if is_olmo else 4096),
            download_dir=VLLM_DIR,
            trust_remote_code=True
        )
    else:
        MODEL = LLM(
            model=MODEL_NAME, 
            tensor_parallel_size=len(DEVICES),
            max_model_len=(2048 if is_olmo else 4096),
            download_dir=VLLM_DIR,
            gpu_memory_utilization=(0.7 if is_code_override else 0.9),
            trust_remote_code=True
        )


def _fix_params(params):
    """
    Our original params format was for the HF generator, so we have to convert them
    to be usable with vLLM. 
    """
    do_sample = params['do_sample']
    output_scores = params['output_scores']
    del params['do_sample']
    del params['epsilon_cutoff'] # Epsilon sampling not supported
    del params['output_scores']

    params['max_tokens'] = params['max_new_tokens']
    del params['max_new_tokens']
    del params['return_dict_in_generate']

    if do_sample:
        del params['num_beams']
        del params['num_return_sequences']
    else:
        params['top_p'] = 1
        params['temperature'] = 0
        params['n'] = params['num_beams']
        params['best_of'] = params['num_beams']
        params['use_beam_search'] = True
        del params['num_beams']
        del params['num_return_sequences']

    if output_scores:
        params['logprobs'] = 1
        params['prompt_logprobs'] = 1

    return params


def model_generate(input_text, **kwargs):
    start_time = datetime.datetime.now()

    if not isinstance(input_text, list): input_text = [input_text]

    # Manual fix for code completion with starcoder models
    if 'starcoder' in MODEL_NAME: 
        input_text = [f'<fim_prefix>{p}<fim_suffix><fim_middle>' for p in input_text]

    params = DEFAULT_KWARGS.copy()
    for k, kwarg in kwargs.items():
        if k in params.keys():
            params[k] = kwarg

    output_scores = params['output_scores']
    params = _fix_params(params)

    print(f'Generating {len(input_text)} examples on {DEVICES} with params {params}')
    
    # See: https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
    params = SamplingParams(**params)

    generation = MODEL.generate(
        prompts=input_text, 
        sampling_params=params,
        use_tqdm=True
    )

    # Measure generation time
    duration = (datetime.datetime.now() - start_time).total_seconds()
    gen_length = sum(sum(len(out.token_ids) for out in o.outputs) for o in generation)
    print(f"Generated {gen_length} tokens in {duration:.2f}s at {gen_length/duration:.2f} tok/s on {DEVICES}.")

    # Flatten nested outputs, helpful for beam search
    generation_text = [i for j in [[o.text for o in out.outputs] for out in generation] for i in j]

    if output_scores:
        if params.use_beam_search:
            raise NotImplementedError('We currently do not support returning logprobs with beam search!')
        seq_prob = [{
            'prompt_logprobs': o.prompt_logprobs,
            'prompt_token_ids': o.prompt_token_ids,
            'gen_logprobs': o.outputs[0].logprobs,
            'gen_cumulative_logprob': o.outputs[0].cumulative_logprob,
            'gen_token_ids': o.outputs[0].token_ids,
        } for o in generation]
        return [(g, s) for g, s in zip(generation_text, seq_prob)]    
    return generation_text


@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    generation = model_generate(**data)
    return jsonify({'generation': generation})


@app.route('/ping')
def ping():
    return 'Pong!'


if __name__ == '__main__':
    setup()

    print(model_generate("What is 1+1?"))
    # print(model_generate(input_text=["What is 1+1?", "What is the airspeed velocity of a swallow?", "A Royale with "]))
    # print(model_generate(input_text="What is 1+1?", do_sample=False, num_beams=50))
    app.run(host='0.0.0.0', port=MODEL_PORT)