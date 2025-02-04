#!/bin/bash
#SBATCH --verbose
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --mem=32GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --output=../../logs/iql_%j.out
#SBATCH --error=../../logs/iql_%j.err
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# --------------------- Your Paths ---------------------
export CORL_ROOT="/scratch/yd2247/CORL"
export CODE_DIR="${CORL_ROOT}/algorithms/iql"
export DATASET_DIR="/scratch/yd2247/rl_paradigm/gym/data/.d4rl/datasets"
#export WANDB_ENTITY="yd2247"
# ------------------------------------------------------

echo "SLURM_JOBID: $SLURM_JOBID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# Experiment config - MODIFY THESE FOR EACH RUN
export ENV_NAME="halfcheetah-medium-expert-v2"
export REWARD_APPROACH="original"  # "original", "naive", or "difference"
export EXPERT_PCT=1.0  # For naive/difference approaches
export SEED=0


# Automatic output directory
export JOB_NAME="IQL_${REWARD_APPROACH}_${ENV_NAME}_${SEED}"
export OUTPUT_DIR="${CORL_ROOT}/results/iql_experiments/${JOB_NAME}/"

# GPU test
singularity exec --nv /$SCRATCH/corl-sandbox python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')
"

# Run experiment with proper dataset binding
singularity exec --nv \
-B ${CORL_ROOT}:/CORL \
-B ${DATASET_DIR}:/d4rl_datasets \
-B /$SCRATCH/corl-sandbox/opt/conda/lib/python3.8/site-packages/mujoco_py/:/opt/conda/lib/python3.8/site-packages/mujoco_py/ \
-B /$SCRATCH/corl-sandbox//workspace/.mujoco/mujoco210/:/$HOME/.mujoco/mujoco210/ \
/$SCRATCH/corl-sandbox bash -c '

# Set Python and library paths
export PYTHONPATH=$PYTHONPATH:/CORL
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/yd2247/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export D4RL_DATASET_DIR=/d4rl_datasets

# Run training script
python /CORL/algorithms/iql/train_iql.py \
    --env ${ENV_NAME} \
    --reward_approach ${REWARD_APPROACH} \
    --expert_pct ${EXPERT_PCT} \
    --seed ${SEED} \
    --checkpoints_path ${OUTPUT_DIR}
'