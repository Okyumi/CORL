# In your terminal
sbatch --export=REWARD_APPROACH="original",ENV_NAME="halfcheetah-medium-expert-v2",EXPERT_PCT=1.0,SEED=0 run_iql.sh

sbatch --export=REWARD_APPROACH="naive",ENV_NAME="halfcheetah-medium-expert-v2",EXPERT_PCT=0.1,SEED=0 run_iql.sh

sbatch --export=REWARD_APPROACH="difference",ENV_NAME="halfcheetah-medium-expert-v2",EXPERT_PCT=0.1,SEED=0 run_iql.sh