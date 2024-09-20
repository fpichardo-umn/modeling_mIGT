#!/bin/bash -l
#SBATCH -J ${JOB_NAME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=32gb
#SBATCH --tmp=32gb
#SBATCH -t 96:00:00
#SBATCH --mail-type=ALL
#SBATCH -o log_files/fit/%x_%A_%a.out
#SBATCH -e log_files/fit/%x_%A_%a.err

# Check if email is provided
if [ -z "$USER_EMAIL" ]; then
  echo "Error: USER_EMAIL is not set. Please provide an email address."
  exit 1
fi

#SBATCH --mail-user=$USER_EMAIL

# Load necessary modules
module load R/4.2.0-rocky8
module load gcc/13.1.0-mptekim
module load cmake/3.29.2-rocky8 
module load boost/1.82.0-gcc-13.1.0-sox6zlc

# Change to the submit directory
cd ${SLURM_SUBMIT_DIR}

# Source the config files
source ./scripts/configs/fit_params_${FIT_CONFIG}.conf
source ./scripts/configs/data_params_${DATA_CONFIG}.conf

# Run the R script with the configuration
Rscript ./scripts/fit/fit_hierarchical_models_cmdSR.R \
  -m ${MODEL_NAME} \
  -t ${MODEL_TYPE} \
  -k ${TASK} \
  -g ${GROUP_TYPE} \
  --n_subs ${n_subs} \
  --n_trials ${n_trials} \
  --RTbound_ms ${RTbound_ms} \
  --rt_method ${rt_method} \
  --n_warmup ${n_warmup} \
  --n_iter ${n_iter} \
  --n_chains ${n_chains} \
  --adapt_delta ${adapt_delta} \
  --max_treedepth ${max_treedepth} \
  --check_iter ${CHECK_ITER}