import os, sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # Obscure fix for CUDA >= 10.2 w/ RoBERTa: see github.com/pytorch/pytorch/issues/47672

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path = sys.path[1:] + [os.path.join(ROOT_DIR), os.path.join(ROOT_DIR, 'src')]

IS_OVERCAP = bool(os.environ.get("OVERCAP", "False") == "True")
LIMIT = None

OPENAI_SECRET_PATH = os.path.join(ROOT_DIR, '.openai-secret') # '.OPENAI-SECRET-GPT-4'

# Configuration for endpoint servers
MODEL_NAME = str(os.environ.get("MODEL_NAME", '/debug')).split('/')[1]
METRIC_NAME = str(os.environ.get("METRIC_NAME", 'default'))

MODEL_PORT = int(os.environ.get("MODEL_PORT", 8500))
METRIC_PORT = int(os.environ.get("METRIC_PORT", 8501))

# Load resource paths
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
OVERCAP_DIR = os.path.join(ROOT_DIR, 'results', '.overcap')
SUBPROCESS_DIR = os.path.join(ROOT_DIR, 'cli', 'slurm', 'subprocess')

# Create results folders
for d in [RESULTS_DIR, OVERCAP_DIR]: 
    if not os.path.exists(d): os.makedirs(d)

# Custom model chat templates
MODEL_TEMPLATES = {
    "Yi-6B": """<|im_start|>\n{instructions}<|im_end|>\n<|im_start|>user\n{source}<|im_end|>\n<|im_start|>""",
    "Yi-34B": """<|im_start|>\n{instructions}<|im_end|>\n<|im_start|>user\n{source}<|im_end|>\n<|im_start|>""",
    "Mistral-7B-v0.1": """<s> [INST] {instructions}\n\n{source} [/INST] """,
    "zephyr-7b-beta": """<|system|>\n{instructions}</s>\n<|user|>\n{source}</s>\n<|assistant|>\n""",
    "tulu-2-dpo-7b": """<|user|>\n{instructions}\n\n{source}\n<|assistant|>\n""",
    "tulu-2-dpo-70b": """<|user|>\n{instructions}\n\n{source}\n<|assistant|>\n""",
}
