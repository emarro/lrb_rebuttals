#!/bin/bash

# Choose one from below
## 131k-ps
model=caduceus-ps_seqlen-131k_d_model-256_n_layer-16
MODEL_NAME=kuleshov-group/$model
LOG_DIR="./watch_folder/${model}"
SEQ_LEN=131072
OFFSET=1

## 131k-ph
#model=caduceus-ph_seqlen-131k_d_model-256_n_layer-16
#MODEL_NAME="kuleshov-group/${model}"
#LOG_DIR="./watch_folder/${model}"
#SEQ_LEN=131072
#OFFSET=4


## 1k-ps
#model=caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3
#MODEL_NAME=kuleshov-group/$model
#LOG_DIR="./watch_folder/${model}"
#SEQ_LEN=1024
#OFFSET=7

## 1k-ph
#model=caduceus-ph_seqlen-1k_d_model-256_n_layer-4_lr-8e-3
#MODEL_NAME=kuleshov-group/$model
#LOG_DIR="./watch_folder/${model}"
#SEQ_LEN=1024
#OFFSET=10


#mkdir -p "${LOG_DIR}"
#export_str="ALL,MODEL_NAME=${MODEL_NAME},SEQ_LEN=${SEQ_LEN}"
#for TASK in "variant_effect_causal_eqtl" "variant_effect_pathogenic_clinvar" "variant_effect_pathogenic_omim"; do
for TASK in  "variant_effect_pathogenic_omim"; do
  export_str="ALL,MODEL_NAME=${MODEL_NAME},SEQ_LEN=${SEQ_LEN},TASK=${TASK},OFFSET=${OFFSET}"
  OFFSET=$(($OFFSET+1))
  NEW_LOG_DIR=$LOG_DIR/$TASK
  echo $NEW_LOG_DIR
  mkdir -p "${NEW_LOG_DIR}"
  #export_str="${export_str} TASK=${TASK}"
  job_name="${TASK}_${MODEL_NAME}"
  echo $export_str
  echo $NEW_LOG_DIR
  sbatch \
    --job-name="${job_name}" \
    --export="$export_str" \
    "run_lrb_zeroshot.sh"
done
echo 'Done'
