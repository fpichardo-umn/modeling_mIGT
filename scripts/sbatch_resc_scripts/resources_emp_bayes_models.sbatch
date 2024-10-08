#!/bin/bash -l
#SBATCH -J ${JOB_NAME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=32gb
#SBATCH --tmp=32gb
#SBATCH -t 96:00:00
#SBATCH --mail-type=ALL
#SBATCH -o log_files/empbayes/%x_%A_%a.out
#SBATCH -e log_files/empbayes/%x_%A_%a.err

# Check if email is provided
if [ -z "$USER_EMAIL" ]; then
  echo "Error: USER_EMAIL is not set. Please provide an email address."
  exit 1
fi

#SBATCH --mail-user=$USER_EMAIL

# Convert STEPS string to array
IFS=',' read -ra STEP_ARRAY <<< "$STEPS"

# Function to check if a step should be run
should_run_step() {
  for step in "${STEP_ARRAY[@]}"; do
    if [ "$step" = "$1" ]; then
      return 0  # Return true (0 in bash) if the step is found
    fi
  done
  return 1  # Return false (1 in bash) if the step is not found
}

# Load necessary modules
module load R/4.2.0-rocky8
module load gcc/13.1.0-mptekim
module load cmake/3.29.2-rocky8 
module load boost/1.82.0-gcc-13.1.0-sox6zlc

# Change to the submit directory
cd ${SLURM_SUBMIT_DIR}

# Source the config files
source ./scripts/configs/fit_params_emp_hier_${FIT_CONFIG}.conf
source ./scripts/configs/data_params_${DATA_CONFIG}.conf

# Function to check exit status and log
check_status() {
    if [ $1 -ne 0 ]; then
        echo "Error: $2 failed with exit code $1"
        exit $1
    else
        echo "$2 completed successfully"
    fi
}

# Step 1: Run initial hierarchical model
if should_run_step 1; then
  echo "Starting initial hierarchical model fit..."
  Rscript ./scripts/empbayes/fit_emp_bayes_hierarchical_models.R \
    -m ${MODEL_NAME} \
    -k ${TASK} \
    --n_trials ${n_trials} \
    --RTbound_ms ${RTbound_ms} \
    --rt_method ${rt_method} \
    --n_warmup ${n_warmup} \
    --n_iter ${n_iter} \
    --n_chains ${n_chains} \
    --adapt_delta ${adapt_delta} \
    --max_treedepth ${max_treedepth} \
    --check_iter ${CHECK_ITER} \
    --seed ${SLURM_JOB_ID}
  check_status $? "Initial hierarchical model fit"
fi

# Step 2: Generate informative priors
if should_run_step 2; then
  echo "Generating informative priors..."
  Rscript ./scripts/empbayes/generate_informative_priors.R \
    -m ${MODEL_NAME} \
    -k ${TASK} \
    --seed ${SLURM_JOB_ID} \
    --verbose
  check_status $? "Generating informative priors"
fi

# Step 3: Run full empirical Bayes model
if should_run_step 3; then
  echo "Starting full empirical Bayes model fit..."
  # Source the config files
  source ./scripts/configs/fit_params_emp_indiv_${FIT_CONFIG}.conf
  
  Rscript ./scripts/empbayes/fit_emp_bayes_group_models.R \
    -m ${MODEL_NAME} \
    -k ${TASK} \
    --n_trials ${n_trials} \
    --RTbound_ms ${RTbound_ms} \
    --rt_method ${rt_method} \
    --n_warmup ${n_warmup} \
    --n_iter ${n_iter} \
    --n_chains ${n_chains} \
    --adapt_delta ${adapt_delta} \
    --max_treedepth ${max_treedepth} \
    --check_iter ${CHECK_ITER} \
    --seed ${SLURM_JOB_ID}
  check_status $? "Full empirical Bayes model fit"
fi
