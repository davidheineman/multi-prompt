import datetime, os

os.environ['NCCL_P2P_DISABLE'] = '1'

import torch
import torch.distributed as dist
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, BitsAndBytesConfig, AutoConfig
from transformers.generation.utils import GenerationConfig

from utils.distributed_inference import setup_model_parallel, generate_distributed, generate_distributed_child
from utils.chat_context import prepare_chat_model_inputs

os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['NCCL_P2P_DISABLE'] = '1'

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

FOUR_BIT_MODELS = [
    'baichuan', 
    'Llama-2-13b',
    'Llama-2-70b',
    'Qwen',
    'falcon-40b',
    'guanaco-65b',
    'guanaco-13b',
    'Yi-34B',
    'CodeLlama-13b',
    'CodeLlama-34b',
    'CodeLlama-70b',
    'deepseek-coder-33b',
    'T5-11B',
    'ALMA-13B-R',
    'TowerInstruct-13B',
    'aya-101'
]

CHAT_MODELS = [
    'baichuan',
    'Qwen',
    'Tower',
    'OLMo'
]

SEQ_TO_SEQ_MODELS = [
    'T5',
    'aya'
]

# After splitting batches among GPUs, this will split into inidividuals runs on each GPU
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 5))

MODEL_NAME = str(os.environ.get("MODEL_NAME", 'meta-llama/Llama-2-7b-chat-hf'))
MODEL_PORT = int(os.environ.get("MODEL_PORT", 8500))

app = Flask(__name__)

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1, start_idx=0):
        super().__init__()
        self.stops = stops
        self.start_idx = start_idx
        self.encounters = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        """
        Stop generation when all sentences have at least one stop token, starting the counter
        after self.start_idx.
        """
        stop_generation = True
        for input_idx in range(input_ids.shape[0]):
            stop_count = 0
            for stop in self.stops:
                stop_count += (stop == input_ids[input_idx][self.start_idx:]).sum().item()
            if (stop_count < self.encounters):
                stop_generation = False
        return stop_generation

def setup():
    global tokenizer, model, stop_words_ids, seq_to_seq, device, local_rank, world_size, multi_gpu_inference

    multi_gpu_inference = torch.distributed.is_available()

    # Got this error?: "RuntimeError: CUDA unknown error - this may be due to an incorrectly set up environment, e.g. changing env variable CUDA_VISIBLE_DEVICES after program start. Setting the available devices to be zero."
    # This is a problem with the node itself, CUDA env needs to be reset (requires sudo)

    # Got this error?: "RuntimeError: probability tensor contains either inf, nan or element < 0"
    # This is caused sometimes when loading the model in 16-bit: "torch_dtype=torch.float16"

    device = 'auto'
    if "LOCAL_RANK" in os.environ and multi_gpu_inference:
        local_rank, world_size = setup_model_parallel()
        device = f'cuda:{local_rank}'

    print(f'Loading model: {MODEL_NAME} to device {device}...')

    quantize = None
    if any([name in MODEL_NAME for name in FOUR_BIT_MODELS]):
        quantize = '4bit'

    seq_to_seq = False
    if any([name in MODEL_NAME for name in SEQ_TO_SEQ_MODELS]):
        seq_to_seq = True

    if quantize == '4bit':
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )
        if seq_to_seq:
            config = AutoConfig.from_pretrained(MODEL_NAME)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_NAME, config=config, device_map=device, quantization_config=bnb_config
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, trust_remote_code=True, device_map=device, quantization_config=bnb_config,
            )
    elif quantize == '8bit':
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, trust_remote_code=True, device_map=device, torch_dtype=torch.float16, load_in_8bit=True
        )
    else:
        if seq_to_seq:
            config = AutoConfig.from_pretrained(MODEL_NAME)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_NAME, config=config, device_map=device
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, trust_remote_code=True, device_map=device
            )

    # Load custom configuration for baichuan chat model
    if 'baichuan' in MODEL_NAME and 'Chat' in MODEL_NAME:
        # Got this error? AttributeError: 'BaichuanTokenizer' object has no attribute 'sp_model'
        # You need to go into tokenization_baichuan.py and move the "self." attributes before super()
        model.generation_config = GenerationConfig.from_pretrained(MODEL_NAME)

    if 'baichuan' in MODEL_NAME or 'Qwen' in MODEL_NAME or 'OLMo' in MODEL_NAME:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side='left')
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    if 'OLMo' in MODEL_NAME:
        model.config.bos_token_id = model.config.pad_token_id

    if 'llama' in MODEL_NAME or 'falcon' in MODEL_NAME or 'Mistral' in MODEL_NAME or 'incoder' in MODEL_NAME or 'Tower' in MODEL_NAME:
        model.config.pad_token_id = model.config.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        if 'incoder' in MODEL_NAME:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))

    if 'Qwen' in MODEL_NAME:
        print("Manually adding [PAD] to tokenizer!")
        tokenizer.pad_token = '<|extra_0|>'
        model.config.bos_token_id = tokenizer._convert_token_to_id('<|im_start|>')

    # Initialize sentence-level stopping criteria
    # The LLaMA tokenizer treats '.' alone and '[word].' as different tokens, so we use this workaround.
    stop_words_ids = None
    if 'llama' in MODEL_NAME:
        stop_words = ['word.',  'word!', 'word?'] # 'word\n'
        stop_words_ids = [tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze()[2] for stop_word in stop_words]
        print(f'Looking for stop token ids: {stop_words_ids}')

def split_list(input_list, batch_size):
    return [input_list[i:i + batch_size] for i in range(0, len(input_list), batch_size)]

def model_batch_generate(input_text, **kwargs):
    if kwargs['do_sample']:
        batches = split_list(input_text, BATCH_SIZE)
    else:
        # For beam search, run each sentence individually, with the batch size
        # used for the different beams
        batches = split_list(input_text, 1)

    print(f"Split input into {len(batches)} batches on {device}")
    generation = []
    for idx, batch in enumerate(batches):
        print(f"Generating batch {idx}/{len(batches)} on {device}")
        torch.cuda.empty_cache()
        generation += model_generate(batch, **kwargs)

    return generation

def model_generate(input_text, **kwargs):
    start_time = datetime.datetime.now()

    params = DEFAULT_KWARGS.copy()
    for k, kwarg in kwargs.items():
        if k in params.keys():
            params[k] = kwarg

    if not params['do_sample']:
        del params['top_p']
        del params['temperature']
        del params['epsilon_cutoff']
        raise NotImplementedError('Beam search no longer working!')

    if params['do_sample']:
        del params['num_beams']
        del params['num_return_sequences']

    sentence_level = 'stopping_criteria' in kwargs and kwargs['stopping_criteria']
    if sentence_level:
        input_length = inputs['input_ids'].shape[1] # This may become a big problem if prompts are different
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids, start_idx=input_length)])
        params.update({'stopping_criteria': stopping_criteria})

    print(f'Generating {len(input_text)} examples on {device} with params {params}')

    if any([name in MODEL_NAME for name in CHAT_MODELS]):
        inputs = prepare_chat_model_inputs(model, tokenizer, MODEL_NAME, input_text)
    else:
        inputs = tokenizer(
            input_text, 
            padding=True,
            return_tensors="pt"
        ).to(model.device)

    if 'incoder' in MODEL_NAME or 'OLMo' in MODEL_NAME:
        del inputs['token_type_ids']
    
    # For beam search see: ../transformers/generation/utils.py
    generation = model.generate(
        **inputs, 
        **params,
        use_cache=True,
    )
    
    # If 'output_scores': True, return those scores 
    return_scores = not torch.is_tensor(generation)
    if return_scores:
        output = generation.sequences
    else:
        output = generation

    # Delete input text from output so it isn't returned to user
    if not seq_to_seq:
        if params['do_sample']:
            for i in range(inputs['input_ids'].size(0)):
                input_length = inputs['input_ids'][i].size(0)
                output[i, :input_length] = model.config.bos_token_id
        else:
            # NOTE: For beam search we only use one candidate and return n responses
            input_length = inputs['input_ids'][0].size(0)
            output[:, :input_length] = model.config.bos_token_id

        if return_scores:
            if params['do_sample']:
                transition_scores = model.compute_transition_scores(
                    generation.sequences, generation.scores, normalize_logits=False
                )
            else:
                transition_scores = model.compute_transition_scores(
                    generation.sequences, generation.scores, generation.beam_indices, normalize_logits=False
                )
            transition_scores[~torch.isfinite(transition_scores)] = 0
            seq_prob = torch.sum(transition_scores, axis=1)

            # Beam search will return -log(p(y|x)), sampling return p(y|x)
            if params['do_sample']: seq_prob = -torch.log(seq_prob)
            
            seq_prob = seq_prob.tolist()

    # Measure generation time
    duration = (datetime.datetime.now() - start_time).total_seconds()
    gen_length = output.shape[1] - inputs['input_ids'].shape[1]
    print(f"Generated {gen_length} tokens in {duration:.2f}s at {gen_length/duration:.2f} tok/s on {device}.")

    if sentence_level:
        # Set all tokens after stop token to EOS
        for row_idx, row in enumerate(output):
            for col_idx, value in enumerate(row[input_length:]):
                if value in stop_words_ids:
                    output[row_idx, input_length+col_idx+1:] = model.config.eos_token_id
    
    if output.shape[0] == 1:
        generation = tokenizer.decode(
            output[0], 
            skip_special_tokens=True
        )
        generation = [generation]
    else:
        generation = tokenizer.batch_decode(
            output, 
            skip_special_tokens=True
        )

    if return_scores:
        return [(g, s) for g, s in zip(generation, seq_prob)]
    
    if not params['do_sample']:
        # Package multiple beam outputs into single arrays
        generation = [generation]
            
    return generation

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json

    if not multi_gpu_inference:
        generation = model_generate(**data)
        return jsonify({'generation': generation})

    final_generation = generate_distributed(request.json, model_batch_generate)

    if not data['do_sample']:
        # For beam search: number of inputs != number of outputs
        # Decompress the generation arrays
        final_generation = [i for j in final_generation for i in j]

    return jsonify({'generation': final_generation})


@app.route('/ping')
def ping():
    return 'Pong!'


if __name__ == '__main__':
    setup()

    if multi_gpu_inference:
        try:
            dist.get_rank()
        except RuntimeError as e:
            raise RuntimeError(f'Failed to initialize distributed setup. Did you mean to run `torchrun --nproc_per_node 8 model_endpoint.py` or `python model_endpoint.py --no_dist`? Threw: {e}')

    if not multi_gpu_inference or dist.get_rank() == 0:
        print(model_generate("What is 1+1?")) # Sanity check
        # print(model_generate(input_text=["What is 1+1?", "What is the airspeed velocity of a swallow?", "A Royale with "]))
        app.run(host='0.0.0.0', port=MODEL_PORT)
    else:
        while True:
            try:
                generate_distributed_child(model_batch_generate)
            except Exception as e:
                print(f'Device {device} threw exception: {e}')
                pass
