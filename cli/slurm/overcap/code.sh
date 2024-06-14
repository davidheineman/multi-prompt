#!/bin/bash
#SBATCH -G a40:1
#SBATCH -c 14
#SBATCH -p overcap
#SBATCH --nodes=1
#SBATCH --job-name=code
#SBATCH --output=../../log/overcap/llama_overcap_%j.log
#SBATCH --signal=USR1@300
#SBATCH --requeue
#SBATCH --exclude=../../../.bad-nodes

# replit/replit-code-v1-3b
# codellama/CodeLlama-7b-Instruct-hf
# codellama/CodeLlama-13b-Instruct-hf (2 A40)
# codellama/CodeLlama-34b-Instruct-hf (4 A40)
# codellama/CodeLlama-70b-Instruct-hf (4 A40)
# deepseek-ai/deepseek-coder-1.3b-instruct
# deepseek-ai/deepseek-coder-6.7b-instruct
# deepseek-ai/deepseek-coder-33b-instruct (4 A40)

# openai/gpt-3.5-turbo-0125
# openai/gpt-4-turbo-2024-04-09
# openai/gpt-4o-2024-05-13

# facebook/incoder-1B (no vLLM)
# facebook/incoder-6B (no vLLM)
# bigcode/starcoder2-15b

export MODEL_NAME="codellama/CodeLlama-13b-Instruct-hf"
export USE_VLLM="True"

# code_bert_score
# mbr_exec
# code_reranker

export METRIC_NAME="mbr_exec"
export OVERCAP="True"

export PYTHONUNBUFFERED=TRUE
cd mbr/cli/slurm

# Remove vLLM lockfile if it exists
lockfile="/tmp/${MODEL_NAME/\//-}.lock"
rm -f "$lockfile" && echo "Deleting '$lockfile'..."

# Set server ports for model and metric
OPEN_PORTS=($(./subprocess/find_port.sh 2))
export MODEL_PORT=${OPEN_PORTS[0]}
export METRIC_PORT=${OPEN_PORTS[1]}

conda activate mbr

cd mbr/cli
echo "Starting code job..."

SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}
LOG_FILE="log/overcap/code_${SLURM_JOB_ID}.log"

srun python code.py
