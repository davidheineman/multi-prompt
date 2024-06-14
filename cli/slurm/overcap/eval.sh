#!/bin/bash
#SBATCH -G a40:1
#SBATCH -c 14
#SBATCH -p nlprx-lab
#SBATCH --qos short
#SBATCH --nodes=1
#SBATCH --job-name=eval
#SBATCH --output=../../log/overcap/llama_overcap_%j.log

export OVERCAP="True"
export PYTHONUNBUFFERED=TRUE

cd mbr/cli/slurm

conda activate mbr

cd mbr/cli
echo "Starting evaluation job..."

SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}
LOG_FILE="log/overcap/eval_${SLURM_JOB_ID}.log"

srun python evaluate_folder.py
