echo "Starting model server..."
cd mbr/cli/slurm
cd mbr/src/server
export MODEL_PORT=${1}
export MODEL_NAME="$2"
conda activate mbr
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=$5 nohup torchrun --nproc_per_node 1 --master_port ${3} model_endpoint.py > "$4" 2>&1 &
