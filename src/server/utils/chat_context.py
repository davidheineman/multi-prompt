import torch

TOWER_TEMPLATE = """<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"""
OLMO_TEMPLATE = """<|user|>\n{prompt}\n<|assistant|>\n"""

def prepare_chat_model_inputs(model, tokenizer, model_name, input_text):
    """
    Wrap chat models with various [USER] and [ASSISTANT] tokens
    """
    if not isinstance(input_text, list): 
        input_text = [input_text]
        
    if 'Qwen' in model_name:
        if not isinstance(input_text, list): 
            input_text = [input_text]
        
        preprocessed_text = []
        for sent in input_text:
            raw_text, context_tokens = qwen_make_context(tokenizer, sent)
            preprocessed_text += [raw_text]
    
        inputs = tokenizer(
            preprocessed_text, 
            padding=True,
            return_tensors="pt"
        ).to(model.device)
    elif 'baichuan' in model_name:
        # Wrap input with [USER] ... [ASSISTANT] tokens
        # First add [ASSISTANT] token and [EOS] at the end of sequence
        assistant_tokens = (model.generation_config.assistant_token_id * torch.ones(inputs['input_ids'].shape[0], 1, dtype=inputs['input_ids'].dtype)).to(model.device)
        pad_tokens = torch.zeros(inputs['input_ids'].shape[0], 1, dtype=inputs['input_ids'].dtype).to(model.device)
        inputs['input_ids'] = torch.cat((pad_tokens, inputs['input_ids'], assistant_tokens), dim=1)

        # Add [USER] whenever attention mask begins
        attention_lengths = inputs['attention_mask'].sum(dim=1)
        for i, length in enumerate(attention_lengths):
            inputs['input_ids'][i, inputs['attention_mask'].shape[1]-length] = model.generation_config.user_token_id

        # Add [USER] token to attention mask
        ones_mask = torch.ones(inputs['attention_mask'].shape[0], 1, dtype=inputs['attention_mask'].dtype).to(model.device)
        zeros_mask = torch.zeros(inputs['attention_mask'].shape[0], 1, dtype=inputs['attention_mask'].dtype).to(model.device)
        inputs['attention_mask'] = torch.cat((zeros_mask, inputs['attention_mask'], ones_mask), dim=1)

        attention_lengths = inputs['attention_mask'].sum(dim=1)
        for i, length in enumerate(attention_lengths):
            inputs['attention_mask'][i, inputs['attention_mask'].shape[1]-length-1] = 1
    elif 'Tower' in model_name:
        input_text = [TOWER_TEMPLATE.format(prompt=in_) for in_ in input_text]
        inputs = tokenizer(
            input_text, 
            padding=True,
            return_tensors="pt"
        ).to(model.device)
    elif 'OLMo' in model_name:
        input_text = [OLMO_TEMPLATE.format(prompt=in_) for in_ in input_text]
        inputs = tokenizer(
            input_text, 
            padding=True,
            return_tensors="pt"
        ).to(model.device)
    else:
        raise ValueError(f'Chat model {model_name} not supported!')
    
    return inputs


def qwen_make_context(tokenizer, query, history = None, system = "", max_window_size = 6144, chat_format = "chatml"):
    """
    Uses exact code from QWEN model to create the context for an individual sentence
    """
    if history is None:
        history = []

    if chat_format == "chatml":
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [tokenizer.im_start_id]
        im_end_tokens = [tokenizer.im_end_id]
        nl_tokens = tokenizer.encode("\n")

        def _tokenize_str(role, content):
            return f"{role}\n{content}", tokenizer.encode(
                role, allowed_special=set()
            ) + nl_tokens + tokenizer.encode(content, allowed_special=set())

        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        raw_text = ""
        context_tokens = []

        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
            response_text, response_tokens_part = _tokenize_str(
                "assistant", turn_response
            )
            response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

            next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
            prev_chat = (
                f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
            )

            current_context_size = (
                len(system_tokens) + len(next_context_tokens) + len(context_tokens)
            )
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break

        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        context_tokens += (
            nl_tokens
            + im_start_tokens
            + _tokenize_str("user", query)[1]
            + im_end_tokens
            + nl_tokens
            + im_start_tokens
            + tokenizer.encode("assistant")
            + nl_tokens
        )
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"
    
    return raw_text, context_tokens
