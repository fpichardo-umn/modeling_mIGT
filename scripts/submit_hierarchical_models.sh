#!/bin/bash

# Function to print usage
print_usage() {
  echo "Usage: $0 -m <model_names> -f <fit_config> -d <data_config> -e <email>"
  echo "Example: $0 -m \"ev_ddm,pvl_ddm\" -f default -d full -e your@email.edu"
  echo "Options:"
  echo "  -m    Comma-separated list of model names"
  echo "  -f    Fit parameters config name (default: default)"
  echo "  -d    Data parameters config name (default: default)"
  echo "  -e    Your email address (required)"
  exit 1
}

# Parse command line arguments
while getopts ":m:f:d:e:" opt; do
  case $opt in
    m) MODEL_NAMES=$OPTARG ;;
    f) FIT_CONFIG=$OPTARG ;;
    d) DATA_CONFIG=$OPTARG ;;
    e) USER_EMAIL=$OPTARG ;;
    \?) echo "Invalid option -$OPTARG" >&2; print_usage ;;
  esac
done

# Check if required arguments are provided
if [ -z "$MODEL_NAMES" ] || [ -z "$USER_EMAIL" ]; then
  echo "Error: Model names and email address are required."
  print_usage
fi

# Set default values if not provided
FIT_CONFIG=${FIT_CONFIG:-default}
DATA_CONFIG=${DATA_CONFIG:-default}

# Convert comma-separated model names to array
IFS=',' read -ra MODEL_ARRAY <<< "$MODEL_NAMES"

# Submit a job for each model
for MODEL_NAME in "${MODEL_ARRAY[@]}"; do
  job_id=$(sbatch --parsable \
    --export=ALL,MODEL_NAME=$MODEL_NAME,FIT_CONFIG=$FIT_CONFIG,DATA_CONFIG=$DATA_CONFIG \
    --mail-user=$USER_EMAIL \
    ./scripts/submit_scripts/resources_fit_hierarchical_models.sbatch)
  echo "Submitted job for model $MODEL_NAME with ID: $job_id"
done

echo "Use 'squeue -u $USER' to monitor your jobs."