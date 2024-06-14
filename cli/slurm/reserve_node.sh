#!/bin/bash
#SBATCH -G a40:2
#SBATCH -c 40
#SBATCH -p nlprx-lab
#SBATCH --qos short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=../log/reserve_node.log
#SBATCH -J mbr

echo "Hello from node $SLURM_NODELIST"

while true; do
    sleep 60
done
