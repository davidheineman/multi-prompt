import random

# Default prompts
SIMPEVAL_INSTRUCTION = """Rewrite the following complex sentence in order to make it easier to understand by non-native speakers of English. You can do so by replacing complex words with simpler synonyms (i.e. paraphrasing), deleting unimportant information (i.e. compression), and/or splitting a long complex sentence into several simpler ones. The final simplified sentence needs to be grammatical, fluent, and retain the main ideas of its original counterpart without altering its meaning."""
SIMPLIFICATION_INSTRUCTION = "Simplify the sentence please."

SIMPLIFY_PROMPT = """{instructions}

Complex sentence: {source}

Simple sentence: """
SIMPLIFICATION_INSTRUCTION_SUFFIX = '\n\nComplex sentence: '

SIMPLIFICATION_EXAMPLE = """Complex sentence: {source}

Simple sentence: {target}"""


def construct_few_shot(examples, n_prompts, n_icl=5):
    few_shot_prompts = []
    for _ in range(n_prompts):
        icl = random.choices(examples, k=n_icl)
        prompt = ""
        for i, example in enumerate(icl):
            prompt += SIMPLIFICATION_EXAMPLE.format(source=example['source'], target=example['references'][0])
            if i < len(icl) - 1: prompt += "\n\n"
        few_shot_prompts += [prompt]
    return few_shot_prompts