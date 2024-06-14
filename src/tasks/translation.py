import random

PROMPT_GENERATION_SEED = 42

# Default prompts
TRANSLATE_INSTRUCTION = """Translate."""
TRANSLATE_PROMPT = """{instructions}

[src_label]: {source}

[tgt_label]: """
TRANSLATE_EXAMPLE = """[src_label]: {source}

[tgt_label]: {target}"""


# Prompt construction utils
def get_translate_labels(src_lang, tgt_lang, return_suffix=False):
    label_pairs = {
        ('en', 'zh'): ('英文句子', '中文句子'),
        ('en', 'cs'): ('Anglická věta', 'Česká věta'),
        ('en', 'de'): ('Englischer Satz', 'Deutscher Satz'),
        ('en', 'ru'): ('Английское предложение', 'русское предложение'),
        ('en', 'is'): ('Ensk setning', 'íslensk setning'),
        ('en', 'fra'): ('Phrase anglaise', 'Phrase française'),
        ('en', 'tur'): ('İngilizce cümle', 'Türkçe cümle'),
        ('en', 'ind'): ('kalimat bahasa inggris', 'Kalimat yang diterjemahkan'),
        ('en', 'jpn'): ('英語の文章', '日本語の文章'),
        ('en', 'urd'): ('انگریزی جملہ', 'اردو جملہ'),
        ('en', 'cat'): ('frase anglesa', 'frase catalana'),
    }
    
    if (src_lang, tgt_lang) not in label_pairs:
        raise ValueError(f'Language pair not supported: {src_lang}-{tgt_lang}')
    
    src_label, tgt_label = label_pairs[(src_lang, tgt_lang)]
    
    if return_suffix:
        return f'\n\n{tgt_label}: '

    return src_label, tgt_label


def get_translate_example_template(src_lang, tgt_lang):
    src_label, tgt_label = get_translate_labels(src_lang, tgt_lang)
    return TRANSLATE_EXAMPLE.replace("[src_label]", src_label).replace("[tgt_label]", tgt_label)


def get_translate_prompt_template(src_lang, tgt_lang):
    src_label, tgt_label = get_translate_labels(src_lang, tgt_lang)
    return TRANSLATE_PROMPT.replace("[src_label]", src_label).replace("[tgt_label]", tgt_label)


def construct_few_shot(examples, src_lang, tgt_lang, n_prompts, n_icl=5):
    # When performing distributed generation, we want to make sure to use the same set of
    # prompts, so we will set a seed only for this selection code
    orig_state = random.getstate()
    random.seed(PROMPT_GENERATION_SEED)

    few_shot_prompts = []
    for _ in range(n_prompts):
        icl = random.choices(examples, k=n_icl)
        translation_prompt = ""
        for i, example in enumerate(icl):
            translation_prompt += get_translate_example_template(src_lang, tgt_lang).format(source=example['source'], target=example['references'][0])
            if i < len(icl) - 1: translation_prompt += "\n\n"
        few_shot_prompts += [translation_prompt]

    random.setstate(orig_state)

    return few_shot_prompts