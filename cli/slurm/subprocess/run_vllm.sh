echo "Starting model server..."
cd mbr/cli/slurm
cd mbr/src/server
export MODEL_PORT=${1}
export MODEL_NAME="$2"
conda activate mbr
CUDA_VISIBLE_DEVICES=$4 python vllm_endpoint.py > "$3" 2>&1 &
