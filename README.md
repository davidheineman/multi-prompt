<div align="center">
    <h1>Multi-prompt MBR Decoding</h1>

<!-- [**Quick Start Demo**](https://colab.research.google.com/drive/14d9ZitBSzbJ1iH13dZEzBzmzoyAhVq9l?usp=sharing) |  -->
[**View Paper**](https://aclanthology.org/2024.emnlp-main.1255/)
</div>

Code and data for *Improving Minimum Bayes Risk Decoding with Multi-Prompt*, appearing at EMNLP Main 2024.

<!-- ## Quick Start
To see an example of multi-prompt MBR, please see our [quick start demo](https://colab.research.google.com/drive/14d9ZitBSzbJ1iH13dZEzBzmzoyAhVq9l?usp=sharing). -->

## Running Experiments
To replicate the experiments in our paper, please follow the below instructions:

### Setup
```sh
# (Optional) Create Conda env
conda create -y -n mbr
conda install -y -n mbr python=3.10
conda activate mbr

# Install dependencies
pip install -r requirements.txt

# Download data and install metrics
chmod +x setup.sh
./setup.sh

# If you want to use OpenAI models, add your OpenAI API token
echo API_KEY >> .openai-secret
```

### CLI Usage
Our code is designed to run in **three terminal windows simultaneously**. This allows debugging the core MBR code without having to restart the model and metric endpoints. Once you have built the experiment code in the CLI, the Slurm utilities (see [below](#slurm-usage)) can run experiments with a single command.

```sh
# In each terminal, export the port so all endpoints are correctly synced
export MODEL_PORT=8500
export METRIC_PORT=8501
```

```sh
# In terminal 1: run the model endpoint
CUDA_VISIBLE_DEVICES=0 MODEL_NAME=meta-llama/Llama-2-7b-chat-hf python src/server/vllm_endpoint.py

# In terminal 2: Run the metric endpoint
CUDA_VISIBLE_DEVICES=1 METRIC_NAME=lens torchrun --nproc_per_node 1 --master_port 29501 src/server/metric_endpoint.py
```

A list of supported models and metrics are included in [#supported-models](#supported-models).

For models not supported by vLLM, we implement a distributed endpoint using HuggingFace inference:
```sh
# Run the model endpoint (slow HuggingFace inference, high compatibility)
CUDA_VISIBLE_DEVICES=0 MODEL_NAME=meta-llama/Llama-2-7b-chat-hf torchrun --nproc_per_node 1 --master_port 29500 src/server/model_endpoint.py
```

To use additional GPUs, simply swap out `CUDA_VISIBLE_DEVICES` with additional GPU IDs.

In the third terminal, these commands will load the data and execute the multi-prompt MBR using the endpoints for inference. They will create a `results` folder containing the generations:
```sh
# In terminal 3: Run the MBR code to run the experiment
python cli/simplification.py
```

**Debugging.** To summarize, here are the three terminal commands to debug each task:
```sh
# Text Simplification
CUDA_VISIBLE_DEVICES=0 MODEL_NAME=meta-llama/Llama-2-7b-chat-hf python src/server/vllm_endpoint.py
CUDA_VISIBLE_DEVICES=1 METRIC_NAME=lens torchrun --nproc_per_node 1 --master_port 29501 src/server/metric_endpoint.py
python cli/simplification.py
```

```sh
# Translation
CUDA_VISIBLE_DEVICES=0 MODEL_NAME=haoranxu/ALMA-7B-R python src/server/vllm_endpoint.py
CUDA_VISIBLE_DEVICES=1 METRIC_NAME=comet torchrun --nproc_per_node 1 --master_port 29501 src/server/metric_endpoint.py
python cli/translation.py
```

```sh
# Code Generation
CUDA_VISIBLE_DEVICES=0 MODEL_NAME=codellama/CodeLlama-13b-Instruct-hf python src/server/vllm_endpoint.py
python cli/code.py # (MBR Exec does not require a GPU!)
```


**Prompt Weight Calculation.** To calculate prompt usage weights for a result file:
```sh
python cli/prompt_weights.py
```

**Evaluation.** Once you have generated a set of results, this command will evaluate a folder, giving the reported metrics in the paper:
```sh
python cli/evaluate_folder.py
```

### Slurm Usage
Slurm scripts are included in [`cli/slurm`](./cli/slurm/). We primarily used overcap scripts as they can be interrupted and can use any number of GPUs for inference:
```sh
cd cli/slurm/overcap
sbatch simplification.sh
sbatch code.sh
sbatch translation.sh
sbatch eval.sh
```

We also include a few utilies for managing the jobs and result files created by slurm.
```sh
python cli/overcap_progress.py # Add --flush flag to remove any in-progress entries, Add --patch to fix corrupted result files
python cli/squash_evals.py
```

### Supported Models
A complete lists of supported models in this project are as follows:

**Models:**
- Translation ([`translation.sh`](./cli/slurm/overcap/translation.sh)): `openai/gpt-3.5-turbo-0125`, `openai/gpt-4-turbo-2024-04-09`, `baichuan-inc/Baichuan2-13B-Chat (no vLLM)`, `Qwen/Qwen-14B-Chat (no vLLM)`, `haoranxu/ALMA-7B-R`, `haoranxu/ALMA-13B-R`, `Unbabel/TowerInstruct-13B-v0.1 (no vLLM)`, `Unbabel/TowerInstruct-7B-v0.2 (no vLLM)`, `CohereForAI/aya-101 (no vLLM)`
- Text Simplification ([`simplification.sh`](./cli/slurm/overcap/simplification.sh)): `douy/T5-3B-Ctrl-Simplification`, `douy/T5-11B-Ctrl-Simplification`, `01-ai/Yi-6B`, `01-ai/Yi-34B`, `tiiuae/falcon-7b-instruct`, `tiiuae/falcon-40b-instruct`, `HuggingFaceH4/zephyr-7b-beta`,  `meta-llama/Llama-2-7b-chat-hf`, `meta-llama/Llama-2-13b-chat-hf`, `meta-llama/Llama-2-70b-chat-hf`, `mistralai/Mistral-7B-v0.1`, `allenai/tulu-2-dpo-7b`, `allenai/tulu-2-dpo-70b`, `google/gemma-2b-it`, `google/gemma-7b-it`, `allenai/OLMo-1B (no vLLM)`, `allenai/OLMo-7B (no vLLM)`,  `openai/gpt-3.5-turbo-0125`, `openai/gpt-4-turbo-2024-04-09`
- Code Generation ([`code.sh`](./cli/slurm/overcap/code.sh)): `replit/replit-code-v1-3b`, `codellama/CodeLlama-7b-Instruct-hf`, `codellama/CodeLlama-13b-Instruct-hf`, `codellama/CodeLlama-34b-Instruct-hf (4 A40)`, `codellama/CodeLlama-70b-Instruct-hf (4 A40)`, `deepseek-ai/deepseek-coder-1.3b-instruct`, `deepseek-ai/deepseek-coder-6.7b-instruct`, `deepseek-ai/deepseek-coder-33b-instruct (4 A40)`, `openai/gpt-3.5-turbo-0125`, `openai/gpt-4-turbo-2024-04-09`, `facebook/incoder-1B (no vLLM)`, `facebook/incoder-6B (no vLLM)`, `bigcode/starcoder2-15b`

**Metrics:**
- Translation ([`translation.sh`](./cli/slurm/overcap/translation.sh)): `bleu`, `bert_score`, `comet`, `comet_kiwi_xxl`, `xcomet`, `metricx`, `metricx_qe`
- Text Simplification ([`simplification.sh`](./cli/slurm/overcap/simplification.sh)): `bert_score`, `sari`, `lens`, `lens_salsa`, `sle`
- Code Generation ([`code.sh`](./cli/slurm/overcap/code.sh)): `code_bert_score`, `mbr_exec`, `code_reranker`