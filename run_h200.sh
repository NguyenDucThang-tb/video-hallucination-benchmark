#!/bin/bash

#PBS -P hn98 
#PBS -q gpuhopper

# 1 H200 141GB
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=64GB

# local temporary disk
#PBS -l jobfs=100GB

# thời gian tối đa
#PBS -l walltime=3:00:00

# project storage
#PBS -l storage=scratch/jp09

# chạy tại thư mục qsub
#PBS -l wd

# tên job
#PBS -N positive_h200


echo "========================================"
echo "JOB ID   : $PBS_JOBID"
echo "HOST     : $(hostname)"
echo "START    : $(date)"
echo "========================================"

nvidia-smi

# ==============================
# Environment
# ==============================

# sửa dòng này theo env của bạn
source .venv/bin/activate

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=12

# giảm fragmentation CUDA memory
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# lazy CUDA loading
export CUDA_MODULE_LOADING=LAZY
MODEL_DIR=/scratch/jp09/dd9648/huggingface/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5
export HF_HOME=/scratch/jp09/dd9648/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QWEN25_VL_MAX_PIXELS=262144



# ==============================
# Output
# ==============================

mkdir -p results/gadi

LOG="results/gadi/positive_feature_${PBS_JOBID}.log"

echo "Log: $LOG"


# ==============================
# Run benchmark
# ==============================

python -u scripts/run_benchmark.py \
    --config configs/experiment_positive_feature.yaml \
    2>&1 | tee "$LOG"


echo "========================================"
echo "END      : $(date)"
echo "========================================"