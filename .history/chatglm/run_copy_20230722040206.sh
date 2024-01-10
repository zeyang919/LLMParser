#!/bin/bash
# 0.5

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

percentage="0.05"

# Checks if command line arguments are provided
if [ -z "$percentage" ]; then
    echo "Please provide a percentage parameter"
    exit 1
fi

# Loop through the parameter list
for param in "${PROJECT_LIST[@]}"; do

    result_dir="./result/${param}/${percentage}"
    
    # Create the result directory if it doesn't exist
    mkdir -p "$result_dir"
    
    # Execute the Python command line, using $param as a parameter
    # python train.py \
    # --batch_size 5 \
    # --num_epochs 15 \
    # --model "chatglm6b-dddd" \
    # --learning_rate 5e-4 \
    # --project "$param" \
    # --train_percentage "$percentage" > "$result_dir/result.txt"
    
    python eval_copy.py \
    --base_model "chatglm6b-dddd" \
    --project "$param" \
    --percentage "$percentage"
    
done
