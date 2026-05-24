#!/bin/bash

#SBATCH -J PSLG-NILM
#SBATCH -p A6000
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -o /home/scnu2023024258/data/code/PSLG-NILM/slurm_log/job-%x-%j.out
#SBATCH -e /home/scnu2023024258/data/code/PSLG-NILM/slurm_log/job-%x-%j.err
#SBATCH --time=24:00:00

# --- 1. 作业开始，打印基本信息 ---
echo "Job started on: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"

# --- 2. 创建日志和输出目录 ---
mkdir -p /home/scnu2023024258/data/code/PSLG-NILM/slurm_log
mkdir -p /home/scnu2023024258/data/code/PSLG-NILM/output
mkdir -p /home/scnu2023024258/data/code/PSLG-NILM/log
echo "Created necessary directories"

# --- 3. 设置软件环境 ---
module purge
echo "Environment purged."
module load miniconda3/latest cuda-toolkit/12.1
echo "Modules loaded."

# --- 4. 激活 Conda 环境 ---
source $(conda info --base)/bin/activate
conda activate PSLG-NILM
echo "Conda environment activated: $CONDA_DEFAULT_ENV"

# --- 5. 执行数据预处理与训练脚本 ---
echo "Starting main.py..."
cd /home/scnu2023024258/data/code/PSLG-NILM

python main.py --config config/recommand_config/config_dishwasher.yaml

if [ $? -eq 0 ]; then
    echo "Workflow executed successfully!"
else
    echo "Workflow execution failed!"
fi

# --- 6. 作业结束 ---
echo "Job finished on: $(date)"
