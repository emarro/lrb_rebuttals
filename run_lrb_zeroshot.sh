#!/bin/bash
#SBATCH --get-user-env                   # Retrieve the users login environment
#SBATCH -t 96:00:00                      # Time limit (hh:mm:ss)
#SBATCH --mem=96000M                     # RAM
#SBATCH --constraint="[h100|a100|3090|a6000|a5000]" 
#SBATCH --gres=gpu:1                     # Number of GPUs
# SBATCH --partition=kuleshov
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -N 1                             # Number of nodes
#SBATCH --requeue                        # Requeue job if it fails
#SBATCH --open-mode=append               # Do not overwrite logs
#SBATCH --array=[9-61]

# source ~/.bashrc
# Setup environment
source setup_env.sh

# Expected args:
# - MODEL_NAME
# - TASK
# - SEQ_LEN



# Run script
echo "*****************************************************"
cd /share/kuleshov/emm392/rebuttal_eval
echo "Running zero_shot model: ${MODEL_NAME}, task: ${TASK},  sequence_length: ${SEQ_LEN}" 
# shellcheck disable=SC2086
NCCL_P2P_DISABLE=1 accelerate launch --num_processes=1 --main_process_port $((29500 + $OFFSET + $SLURM_ARRAY_TASK_ID)) caduceus_zeroshot_script.py \
  --task_name "${TASK}" \
  --sequence_length ${SEQ_LEN} \
  --model_name "${MODEL_NAME}" \
  --shards 61 \
  --rank $SLURM_ARRAY_TASK_ID
echo "*****************************************************"
