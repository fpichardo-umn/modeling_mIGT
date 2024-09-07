#!/bin/bash

# Load modules
module load R/4.2.0-rocky8

# Function to print usage
print_usage() {
  echo "Usage: $0 -m <model_names> -f <fit_config> -d <data_config> -e <email> -t <model_type> -k <task> -g <group_type> [-n]"
  echo "Example: $0 -m \"ev_ddm,pvl_ddm\" -f default -d full -e your@email.edu -t fit -k igt_mod -g group_hier"
  echo "Options:"
  echo "  -m    Comma-separated list of model names"
  echo "  -f    Fit parameters config name (default: default)"
  echo "  -d    Data parameters config name (default: default)"
  echo "  -t    Type of stan code to run (fit, postpc, prepc) (default: fit)"
  echo "  -k    Task name (e.g., igt_mod)"
  echo "  -g    Group type (sing, group, group_hier)"
  echo "  -e    Your email address (required)"
  echo "  -n    Dry run (optional)"
  exit 1
}

# Parse command line arguments
DRY_RUN=false
while getopts ":m:f:d:e:t:k:g:n" opt; do
  case $opt in
    m) MODEL_NAMES=$OPTARG ;;
    f) FIT_CONFIG=$OPTARG ;;
    d) DATA_CONFIG=$OPTARG ;;
    t) MODEL_TYPE=$OPTARG ;;
    k) TASK=$OPTARG ;;
    g) GROUP_TYPE=$OPTARG ;;
    e) USER_EMAIL=$OPTARG ;;
    n) DRY_RUN=true ;;
    \?) echo "Invalid option -$OPTARG" >&2; print_usage ;;
  esac
done

# Check if required arguments are provided
if [ -z "$MODEL_NAMES" ] || [ -z "$USER_EMAIL" ] || [ -z "$TASK" ] || [ -z "$GROUP_TYPE" ]; then
  echo "Error: Model names, email address, task, and group type are required."
  print_usage
fi

# Set default values if not provided
FIT_CONFIG=${FIT_CONFIG:-default}
DATA_CONFIG=${DATA_CONFIG:-default}
MODEL_TYPE=${MODEL_TYPE:-fit}

# Directory checks
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
CONFIG_DIR="${SCRIPT_DIR}/configs"
R_SCRIPT="${SCRIPT_DIR}/fit_hierarchical_models_cmdSR.R"
SUBMIT_SCRIPTS_DIR="${SCRIPT_DIR}/submit_scripts"
SBATCH_SCRIPT="${SUBMIT_SCRIPTS_DIR}/resources_fit_hierarchical_models_cmdSR.sbatch"

PROJ_DIR="${SCRIPT_DIR}/.."
MODEL_DIR="${PROJ_DIR}/models/bin"
DATA_FILE="${PROJ_DIR}/Data/AHRB/modigt_data_Wave1.sav"
OUTPUT_DIR="${PROJ_DIR}/log_files"

# Function to check and print status
check_and_print() {
  if [ $? -eq 0 ]; then
    echo "[OK] $1"
  else
    echo "[FAIL] $1"
    exit 1
  fi
}

# Check config files
[ -f "${CONFIG_DIR}/fit_params_${FIT_CONFIG}.conf" ]
check_and_print "Fit config file: ${CONFIG_DIR}/fit_params_${FIT_CONFIG}.conf"

[ -f "${CONFIG_DIR}/data_params_${DATA_CONFIG}.conf" ]
check_and_print "Data config file: ${CONFIG_DIR}/data_params_${DATA_CONFIG}.conf"

# Check R script
[ -f "$R_SCRIPT" ]
check_and_print "R script: $R_SCRIPT"

# Check R script
[ -f "$SBATCH_SCRIPT" ]
check_and_print "SBATCH script: $SBATCH_SCRIPT"

# Check data file
[ -f "$DATA_FILE" ]
check_and_print "Data file: $DATA_FILE"

# Check data file
check_and_print "Model type: $MODEL_TYPE"

# Check output directory
[ -d "$OUTPUT_DIR" ] && [ -w "$OUTPUT_DIR" ]
check_and_print "Output directory (exists and writable): $OUTPUT_DIR"

# Check for required R packages
Rscript -e "if (!all(c('rstan', 'optparse') %in% installed.packages()[,'Package'])) quit(status=1)"
check_and_print "Required R packages installed"

# Convert comma-separated model names to array
IFS=',' read -ra MODEL_ARRAY <<< "$MODEL_NAMES"

# Function to generate R script call
generate_r_call() {
  local model=$1
  local dry_run_flag=$2
  echo "Rscript $R_SCRIPT -m $model -t $MODEL_TYPE -k $TASK -g $GROUP_TYPE --n_subs \${n_subs} --n_trials \${n_trials} --RTbound_ms \${RTbound_ms} --rt_method \${rt_method} --n_warmup \${n_warmup} --n_iter \${n_iter} --n_chains \${n_chains} --adapt_delta \${adapt_delta} --max_treedepth \${max_treedepth} $dry_run_flag"
}

# Check model files and submit jobs
for MODEL_NAME in "${MODEL_ARRAY[@]}"; do
  MODEL_FILE="${MODEL_DIR}/${MODEL_TYPE}/${TASK}_${GROUP_TYPE}_${MODEL_NAME}_${MODEL_TYPE}.rds"
  echo "[OK] Model file: $MODEL_FILE"

  if $DRY_RUN; then
    echo "Dry run for model ${TASK}_${GROUP_TYPE}_${MODEL_NAME}_${MODEL_TYPE}:"
    echo "  SLURM job would be submitted with:"
    echo "    MODEL_NAME=$MODEL_NAME"
    echo "    FIT_CONFIG=$FIT_CONFIG"
    echo "    DATA_CONFIG=$DATA_CONFIG"
    echo "    USER_EMAIL=$USER_EMAIL"
    echo "    MODEL_TYPE=$MODEL_TYPE"
    echo "    TASK=$TASK"
    echo "    GROUP_TYPE=$GROUP_TYPE"
    echo "  R script call would be:"
    generate_r_call $MODEL_NAME "--dry_run"
    
    # Actually run the R script in dry-run mode
    source "${CONFIG_DIR}/fit_params_${FIT_CONFIG}.conf"
    source "${CONFIG_DIR}/data_params_${DATA_CONFIG}.conf"
    eval $(generate_r_call $MODEL_NAME "--dry_run")
  else
    JOB_NAME="${TASK}_${GROUP_TYPE}_${MODEL_NAME}_${MODEL_TYPE}"
    job_id=$(sbatch --parsable \
      --job-name=$JOB_NAME \
      --export=ALL,JOB_NAME=$JOB_NAME,MODEL_NAME=$MODEL_NAME,FIT_CONFIG=$FIT_CONFIG,DATA_CONFIG=$DATA_CONFIG,USER_EMAIL=$USER_EMAIL,MODEL_TYPE=$MODEL_TYPE,TASK=$TASK,GROUP_TYPE=$GROUP_TYPE \
      $SBATCH_SCRIPT)
    if [ $? -eq 0 ]; then
      echo "Submitted job for model $MODEL_NAME with ID: $job_id"
    else
      echo "Failed to submit job for model $MODEL_NAME"
    fi
  fi
  echo
done

if $DRY_RUN; then
  echo "This was a dry run. No jobs were actually submitted."
else
  echo "Use 'squeue -u $USER' to monitor your jobs."
  echo "Use 'squeue -l --me' to monitor your jobs."
fi