#!/bin/bash

PROJECT_LIST=(
  'HealthApp'
  'OpenStack'
  'OpenSSH'
  'Proxifier'
  'HPC'
  'Zookeeper'
  'Mac'
  'Hadoop'
  'Linux'
  'Android'
  'HDFS'
  'BGL'
  'Windows'
  'Apache'
  'Thunderbird'
  'Spark'
)

percentage="$1"

# Checks if command line arguments are provided
if [ -z "$percentage" ]; then
  echo "Please provide a percentage parameter"
  exit 1
fi

# Loop through the parameter list
for param in "${PROJECT_LIST[@]}"; do
    data_path="../logs/$param/$percentage/train.json"
    output_dir="../trained/$param/$percentage"

    # Create the result directory if it doesn't exist
    mkdir -p "$result_dir"

    # Execute the Python command line, using $param as a parameter
    python finetune.py \
    --base_model '../LLMs/llama-7b-hf' \
    --data_path "$data_path" \
    --output_dir "$output_dir" \
    --batch_size 5 \
    --micro_batch_size 5 \
    --num_epochs 30 \
    --learning_rate 5e-4 \
    --cutoff_len 1024 \
    --val_set_size 0 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length

    python eval.py \
    --base_model './llama-7b-hf' \
    --project "$param" \
    --percentage "$1"
    
done
