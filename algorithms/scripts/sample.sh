#!/bin/bash
#SBATCH --verbose
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --mem=32GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yd2247@nyu.edu
#SBATCH --array=0-5 # Adjust the range depending on the number of tasks 0-5
#SBATCH --output=../../logs/%A_%a.out
#SBATCH --error=../../logs/%A_%a.err
#SBATCH --partition=nvidia  # Use the nvidia or condo partition
#SBATCH --gres=gpu:1        # Request 1 GPU
#SBATCH --cpus-per-task=8   # CPU cores per task
########################################################
sleep $(( (RANDOM%10) + 1 )) # Avoid job start collisions

echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: ${SLURM_ARRAY_TASK_ID}"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
# Set environment variables for the experiment
export ALGO_NAME="bc"
export DATASET_NAME="can"
export FILTER_SUCCESS=True # filter out the trajectories that are not successful, fbc here
export RANDOM_TIMESTEPS=0
export DATA_TYPE="mg"
export MODE="sparse"
#export SEQ_LENGTH=5

export QUALITY="low_dim"
# export BATCH_SIZE=256 # it does not matter for bc as well
# export NUM_EPOCHS=1200 # it does not matter for bc
export ROLLOUT_N=100 # number of rollouts during evaluation
export LR=1e-4
export WEIGHT_DECAY=0.1
export DROPOUT=0.1


export NUM_EPISODES_DURING_EVAL=50 
export PCT_TRAJ=1

#[256,256]
export CONFIG=/rl_paradigm/robomimic/robomimic/exps/templates/${ALGO_NAME}.json

# Set up job name and checkpoint path
export JOB_NAME=${DATASET_NAME}_${DATA_TYPE}_256_256_qual_${QUALITY}_bc_random_${RANDOM_TIMESTEPS}_NEW
export OUTPUT_DIR=/scratch/yd2247/rl_paradigm/robomimic/output_oct30/${ALGO_NAME}_${FILTER_SUCCESS}/${JOB_NAME}/${SLURM_ARRAY_TASK_ID}/


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
-B /$SCRATCH/rl_paradigm:/rl_paradigm \
-B /$SCRATCH/robosuite:/robosuite \
-B /$SCRATCH/rl_paradigm/gym:/gym \
-B /$SCRATCH/rl_paradigm/robomimic:/robomimic \
-B /$SCRATCH/corl-sandbox/opt/conda/lib/python3.8/site-packages/mujoco_py/:/opt/conda/lib/python3.8/site-packages/mujoco_py/ \
-B /$SCRATCH/corl-sandbox//workspace/.mujoco/mujoco210/:/$HOME/.mujoco/mujoco210/ \
/$SCRATCH/corl-sandbox bash -c '


cd /rl_paradigm/robomimic
#cd /rl_paradigm
# Add project folders to Python paths so that 'import' functions well
export PYTHONPATH=$PYTHONPATH:/rl_paradigm:/gym:/robomimic:/robosuite
export PYOPENGL_PLATFORM='egl'
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/yd2247/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# Run the experiment
python3 /rl_paradigm/robomimic/robomimic/scripts/train_draft.py \
    --config $CONFIG \
    --dataset /scratch/yd2247/rl_paradigm/robomimic/datasets/${DATASET_NAME}/${DATA_TYPE}/${QUALITY}_${MODE}.hdf5 \
    --experiment_name $ALGO_NAME \
    --output_dir $OUTPUT_DIR \
    --rollout_n $ROLLOUT_N \
    --lr $LR \
    --filter_success $FILTER_SUCCESS \
    --weight_decay $WEIGHT_DECAY \
    --dropout $DROPOUT \
    --pct_traj $PCT_TRAJ \
    --num_episodes_during_eval $NUM_EPISODES_DURING_EVAL \
    --random_timesteps $RANDOM_TIMESTEPS \
'
