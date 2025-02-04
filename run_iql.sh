#!/bin/bash
#SBATCH --verbose
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --mem=32GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your.email@nyu.edu
#SBATCH --array=0-23%4  # 8 envs * 3 seeds = 24 jobs, running 4 at a time
#SBATCH --output=../../logs/%A_%a.out
#SBATCH --error=../../logs/%A_%a.err
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# Avoid job start collisions
sleep $(( (RANDOM%10) + 1 ))

echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

# Calculate environment and seed index from array task ID
ENV_IDX=$((SLURM_ARRAY_TASK_ID / 3))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 3))

# Arrays of environments and corresponding seeds
ENVS=(
    "halfcheetah/medium_v2.yaml"
    "halfcheetah/medium_expert_v2.yaml"
    "halfcheetah/medium_replay_v2.yaml"
    "halfcheetah/expert_v2.yaml"
    "hopper/full_replay_v2.yaml"
    "pen/expert_v1.yaml"
    "pen/cloned_v1.yaml"
    "relocate/expert_v1.yaml"
)

SEEDS=(0 1 2)

# Get the environment and seed for this job
ENV=${ENVS[$ENV_IDX]}
SEED=${SEEDS[$SEED_IDX]}

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# Run GPU detection test
echo "Running GPU detection test..."
singularity exec --nv /$SCRATCH/corl-sandbox python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
print('Current device:', torch.cuda.current_device())
print('Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')
"

# Use Singularity to execute the job in an isolated environment
singularity exec --nv \
-B /$SCRATCH/CORL:/CORL \
-B /$SCRATCH/robosuite:/robosuite \
-B /$SCRATCH/corl-sandbox/opt/conda/lib/python3.8/site-packages/mujoco_py/:/opt/conda/lib/python3.8/site-packages/mujoco_py/ \
-B /$SCRATCH/corl-sandbox//workspace/.mujoco/mujoco210/:/$HOME/.mujoco/mujoco210/ \
/$SCRATCH/corl-sandbox bash -c '

# Set up Python paths
export PYTHONPATH=$PYTHONPATH:/CORL:/robosuite
export PYOPENGL_PLATFORM="egl"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$USER/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# Run the IQL training
python /CORL/algorithms/offline/iql.py \
    --config-path /CORL/configs/offline/iql/'$ENV' \
    --seed '$SEED'
' 