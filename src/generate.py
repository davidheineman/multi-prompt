import requests, os, json, datetime, time, concurrent.futures
import openai
from openai import OpenAI


DEFAULT_KWARGS_ENDPOINT = {
    'do_sample': True,
    'top_p': 0.9,
    'epsilon_cutoff': 0,
    'temperature': 0.9,
    'max_new_tokens': 256,
    'num_beams': 1,
    'num_return_sequences': 1,
    'return_dict_in_generate': False, 
    'output_scores': False
}

DEFAULT_KWARGS_OPENAI = {
    'top_p': 0.9,
    'temperature': 0.9,
    'max_new_tokens': 256, 
    'output_scores': False
}

MODEL_ENDPOINT = 'http://localhost:{port}/generate'


def openai_init(secret_path, model_name="gpt-3.5-turbo-instruct"):
    global OPENAI_MODEL, OPENAI_CLIENT
    OPENAI_MODEL = model_name
    if not os.path.exists(secret_path): 
        raise RuntimeError(f'Need an OpenAI Key! Did not find a key at {secret_path}')
    with open(secret_path, 'r') as f: 
        api_key = f.read().strip()
    OPENAI_CLIENT = OpenAI(api_key=api_key)


def _call_openai(p, params, sentence_level=False, max_retries=7, base_delay=2):
    retries = 0
    while retries < max_retries:
        try:
            response = OPENAI_CLIENT.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{ "role": "user", "content": p }],
                temperature=params['temperature'],
                max_tokens=params['max_new_tokens'],
                top_p=params['top_p'],
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=['.'] if sentence_level else None
            )
            return response
        except (openai.RateLimitError, openai.InternalServerError) as e:
            retries += 1
            print(f"OpenAI API request exceeded rate limit: {e}. Retrying ({retries}/{max_retries})...")
            if retries < max_retries:
                delay = base_delay * (2 ** retries)
                time.sleep(delay)
            else:
                raise RuntimeError('OpenAI API failed to respond')


def generate_gpt(prompt, sentence_level=False, concurrent=False, **kwargs):
    """
    Generate with GPT. 
    
    sentence_level: use '.' as a stop condition
    """
    start_time = datetime.datetime.now()
    if not isinstance(prompt, list): prompt = [prompt]
    
    params = DEFAULT_KWARGS_OPENAI.copy()
    for k, kwarg in kwargs.items():
        if k in params.keys():
            params[k] = kwarg

    print(f'Generating {len(prompt)} examples on {OPENAI_MODEL} with params {params}')
    
    if concurrent:
        # Query OpenAI using threading
        with concurrent.futures.ThreadPoolExecutor() as exec:
            futures = [exec.submit(_call_openai, p, params, sentence_level=sentence_level) for p in prompt]
            cands = [f.result() for f in concurrent.futures.as_completed(futures)]
        cands = [c.choices[0].message.content for c in cands]
    else:
        # Query OpenAI sequentially
        cands = []
        for p in prompt:
            resp = _call_openai(p, params, sentence_level=sentence_level)
            cands += [resp.choices[0].message.content]

    duration = (datetime.datetime.now() - start_time).total_seconds()
    print(f"Generated {len(prompt)} queries in {duration:.2f}s at {len(prompt)/duration:.2f} prompt/s.")

    return cands


def generate_endpoint(prompt, sentence_level=False, port=8500, max_retries=3, base_delay=2, **kwargs):
    """
    Use an API endpoint to generate
    """
    params = DEFAULT_KWARGS_ENDPOINT.copy()
    for k, kwarg in kwargs.items():
        if k in params.keys():
            params[k] = kwarg

    data = { 
        'input_text': prompt
    }
    data.update(params)

    if sentence_level:
        data.update({'stopping_criteria': sentence_level})

    endpoint = MODEL_ENDPOINT.format(port=port)

    # Exponential backoff
    retries = 0
    while retries < max_retries:
        try:
            response = requests.post(endpoint, json=data)
            response = response.json()
            return response['generation']
        except json.decoder.JSONDecodeError:
            retries += 1
            print(f"Endpoint ({endpoint}) failed to respond. Returned: {response}. Retrying ({retries}/{max_retries})...")
            if retries < max_retries:
                delay = base_delay * (2 ** retries)
                time.sleep(delay)
            else:
                raise RuntimeError('Endpoint failed to respond')


def get_generate_params(decoding_method, n_beams=None, temp=0.9):
    if decoding_method == 'top-p':
        generate_params = {
            'do_sample': True,
            'top_p': 1,
            'temperature': temp,
            'max_new_tokens': 256
        }
    elif decoding_method == 'epsilon':
        generate_params = {
            'do_sample': True,
            'top_p': 1,
            'epsilon_cutoff': 0.02,
            'temperature': temp,
            'max_new_tokens': 256
        }
    elif decoding_method == 'greedy':
        generate_params = {
            'do_sample': True,
            'top_p': 1,
            'temperature': 0.01,
            'max_new_tokens': 256
        }
    elif decoding_method == 'beam':
        assert n_beams is not None, "Must provide a number of beams!"
        generate_params = {
            'do_sample': False,
            'top_p': None,
            'temperature': None,
            'num_beams': n_beams,
            'num_return_sequences': n_beams,
            'max_new_tokens': 256
        }
    elif decoding_method == 'multi-prompt-beam':
        # Custom beam search where each prompt is set as a start node
        raise NotImplementedError()
    else:
        raise ValueError(f'Decoding method {decoding_method} not supported!')
    
    return generate_params
