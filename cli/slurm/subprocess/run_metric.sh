echo "Starting metric server..."
cd mbr/cli/slurm
cd mbr/src/server
export METRIC_PORT=${1}
export METRIC_NAME=${2}
conda activate mbr
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 nohup torchrun --nproc_per_node 1 --master_port ${3} metric_endpoint.py > "${4}" 2>&1 &
