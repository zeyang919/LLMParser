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
    
    # Execute the Python command line, using $param as a parameter
    python train.py \
    --batch_size 5 \
    --num_epochs 30 \
    --model "../LLMs/chatglm6b-dddd" \
    --learning_rate 5e-4 \
    --project "$param" \
    --train_percentage "$percentage"
    
    python eval.py \
    --base_model "../LLMs/chatglm6b-dddd" \
    --project "$param" \
    --percentage "$percentage" \
    
done
