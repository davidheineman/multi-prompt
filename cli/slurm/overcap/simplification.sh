#!/bin/bash
#SBATCH -G a40:2
#SBATCH -c 14
#SBATCH -p overcap
#SBATCH --nodes=1
#SBATCH --job-name=simplification
#SBATCH --output=../../log/overcap/llama_overcap_%j.log
#SBATCH --signal=USR1@300
#SBATCH --requeue
#SBATCH --exclude=../../../.bad-nodes

# douy/T5-3B-Ctrl-Simplification
# douy/T5-11B-Ctrl-Simplification
# 01-ai/Yi-6B
# 01-ai/Yi-34B
# tiiuae/falcon-7b-instruct
# tiiuae/falcon-40b-instruct
# HuggingFaceH4/zephyr-7b-beta
# allenai/tulu-2-dpo-7b
# allenai/tulu-2-dpo-70b

# meta-llama/Llama-2-7b-chat-hf
# meta-llama/Llama-2-13b-chat-hf
# meta-llama/Llama-2-70b-chat-hf (4 A40)
# meta-llama/Meta-Llama-3-8B-Instruct
# meta-llama/Meta-Llama-3-70B-Instruct (4 A40)
# mistralai/Mistral-7B-v0.1
# google/gemma-2b-it
# google/gemma-7b-it
# allenai/OLMo-1B
# allenai/OLMo-7B-Instruct

# openai/gpt-3.5-turbo-0125
# openai/gpt-4-turbo-2024-04-09
# openai/gpt-4o-2024-05-13

export MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
export USE_VLLM="True"

# bert_score
# sari
# lens
# lens_salsa
# sle

export METRIC_NAME="lens"
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
echo "Starting simplification job..."

SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}
LOG_FILE="log/overcap/simplification_${SLURM_JOB_ID}.log"

srun python simplification.py
