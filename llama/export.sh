#!/bin/bash
# cross
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
    python export_hf_checkpoint.py \
    --base_model './llama-7b-hf' \
    --project "$param" \
    --percentage "$1"
    
done
