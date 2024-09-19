#!/bin/bash

# Load modules
module load R/4.2.0-rocky8

# Function to print usage
print_usage() {
  echo "Usage: $0 -m <model_names> -f <fit_config> -d <data_config> -e <email> -k <task> -g <group_type> -p <param_space_exp> [-n]"
  echo "Example: $0 -m \"ev_ddm,pvl_ddm\" -f default -d full -e your@email.edu -k igt_mod -g group_hier -p \"ssFPSE,rndFPSE\""
  echo "Options:"
  echo "  -m    Comma-separated list of model names"
  echo "  -f    Fit parameters config name (default: default)"
  echo "  -d    Data parameters config name (default: default)"
  echo "  -k    Task name (e.g., igt_mod)"
  echo "  -g    Group type (sing, group, group_hier)"
  echo "  -p    Comma-separated list of parameter space exploration types"
  echo "  -e    Your email address (required)"
  echo "  -n    Dry run (optional)"
  exit 1
}

# Parse command line arguments
DRY_RUN=false
while getopts ":m:f:d:e:k:g:p:n" opt; do
  case $opt in
    m) MODEL_NAMES=$OPTARG ;;
    f) FIT_CONFIG=$OPTARG ;;
    d) DATA_CONFIG=$OPTARG ;;
    k) TASK=$OPTARG ;;
    g) GROUP_TYPE=$OPTARG ;;
    p) PARAM_SPACE_EXP=$OPTARG ;;
    e) USER_EMAIL=$OPTARG ;;
    n) DRY_RUN=true ;;
    \?) echo "Invalid option -$OPTARG" >&2; print_usage ;;
  esac
done

# Check if required arguments are provided
if [ -z "$MODEL_NAMES" ] || [ -z "$USER_EMAIL" ] || [ -z "$TASK" ] || [ -z "$GROUP_TYPE" ] || [ -z "$PARAM_SPACE_EXP" ]; then
  echo "Error: Model names, email address, task, group type, and parameter space exploration types are required."
  print_usage
fi

# Set default values if not provided
FIT_CONFIG=${FIT_CONFIG:-default}
DATA_CONFIG=${DATA_CONFIG:-default}

# Directory checks
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
CONFIG_DIR="${SCRIPT_DIR}/configs"
SUBMIT_SCRIPTS_DIR="${SCRIPT_DIR}/submit_scripts"
SBATCH_SCRIPT="${SUBMIT_SCRIPTS_DIR}/param_recovery_igt_mod_cmdSR.sbatch"

PROJ_DIR="${SCRIPT_DIR}/.."
DATA_DIR="${PROJ_DIR}/Data"
DATA_SIM_DIR="${DATA_DIR}/sim"
DATA_SIM_TXT_DIR="${DATA_SIM_DIR}/txt"
OUTPUT_DIR="${PROJ_DIR}/Data/sim/rds"

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

# Check SBATCH script
[ -f "$SBATCH_SCRIPT" ]
check_and_print "SBATCH script: $SBATCH_SCRIPT"

# Check output directory
[ -d "$OUTPUT_DIR" ] || mkdir -p "$OUTPUT_DIR"
check_and_print "Output directory (exists and writable): $OUTPUT_DIR"

# Convert comma-separated model names and param space exp to arrays
IFS=',' read -ra MODEL_ARRAY <<< "$MODEL_NAMES"
IFS=',' read -ra PARAM_SPACE_ARRAY <<< "$PARAM_SPACE_EXP"

# Check model files and submit jobs
for MODEL_NAME in "${MODEL_ARRAY[@]}"; do
  for PARAM_SPACE in "${PARAM_SPACE_ARRAY[@]}"; do
    JOB_NAME="${TASK}_${GROUP_TYPE}_${MODEL_NAME}_${PARAM_SPACE}"
    FULL_MODEL_NAME="${TASK}_${GROUP_TYPE}_${MODEL_NAME}"
    
    # Check for required input files
    SIM_DATA_FILE="${DATA_SIM_TXT_DIR}/sim_${FULL_MODEL_NAME}_desc-data_${PARAM_SPACE}.csv"
    SIM_PARAMS_FILE="${DATA_SIM_TXT_DIR}/sim_${FULL_MODEL_NAME}_desc-params_${PARAM_SPACE}.csv"
    
    if [ ! -f "$SIM_DATA_FILE" ]; then
      echo "[FAIL] Simulated data file not found: $SIM_DATA_FILE"
      continue
    fi
    
    if [ ! -f "$SIM_PARAMS_FILE" ]; then
      echo "[FAIL] Simulated parameters file not found: $SIM_PARAMS_FILE"
      continue
    fi
    
    if $DRY_RUN; then
      echo "Dry run for model ${JOB_NAME}:"
      echo "  SLURM job would be submitted with:"
      echo "    MODEL_NAME=$MODEL_NAME"
      echo "    FIT_CONFIG=$FIT_CONFIG"
      echo "    DATA_CONFIG=$DATA_CONFIG"
      echo "    USER_EMAIL=$USER_EMAIL"
      echo "    TASK=$TASK"
      echo "    GROUP_TYPE=$GROUP_TYPE"
      echo "    PARAM_SPACE_EXP=$PARAM_SPACE"
      echo "    OUTPUT_DIR=$OUTPUT_DIR"
      echo "  Input files:"
      echo "    Simulated data: $SIM_DATA_FILE"
      echo "    Simulated parameters: $SIM_PARAMS_FILE"
    else
      job_id=$(sbatch --parsable \
        --job-name=$JOB_NAME \
        --mail-user=$USER_EMAIL \
        --export=ALL,JOB_NAME=$JOB_NAME,MODEL_NAME=$MODEL_NAME,FIT_CONFIG=$FIT_CONFIG,DATA_CONFIG=$DATA_CONFIG,TASK=$TASK,GROUP_TYPE=$GROUP_TYPE,PARAM_SPACE_EXP=$PARAM_SPACE,OUTPUT_DIR=$OUTPUT_DIR \
        $SBATCH_SCRIPT)
      if [ $? -eq 0 ]; then
        echo "Submitted job for model $MODEL_NAME, param space $PARAM_SPACE with ID: $job_id"
      else
        echo "Failed to submit job for model $MODEL_NAME, param space $PARAM_SPACE"
      fi
    fi
    echo
  done
done

if $DRY_RUN; then
  echo "This was a dry run. No jobs were actually submitted."
else
  echo "Use 'squeue -u $USER' to monitor your jobs."
fi