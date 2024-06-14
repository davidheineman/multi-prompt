#!/bin/bash
#SBATCH -G a40:2
#SBATCH -c 7
#SBATCH -p overcap
#SBATCH --nodes=1
#SBATCH --job-name=translation
#SBATCH --output=../../log/overcap/llama_overcap_%j.log
#SBATCH --signal=USR1@300
#SBATCH --requeue
#SBATCH --exclude=../../../.bad-nodes

# openai/gpt-3.5-turbo-0125
# openai/gpt-4-turbo-2024-04-09
# openai/gpt-4o-2024-05-13

# baichuan-inc/Baichuan2-13B-Chat
# Qwen/Qwen-14B-Chat
# haoranxu/ALMA-7B-R
# haoranxu/ALMA-13B-R

# Unbabel/TowerInstruct-7B-v0.2 (no vLLM)
# Unbabel/TowerInstruct-13B-v0.1 (no vLLM)
# CohereForAI/aya-101 (no vLLM)

export MODEL_NAME="haoranxu/ALMA-7B-R"
export USE_VLLM="True"

# bleu
# bert_score
# comet
# comet_kiwi_xxl
# xcomet
# metricx
# metricx_qe

export METRIC_NAME="comet"
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
echo "Starting translation job..."

SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}
LOG_FILE="log/overcap/translation_${SLURM_JOB_ID}.log"

srun python translation.py
