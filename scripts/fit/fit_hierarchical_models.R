#!/usr/bin/env Rscript

# Load required libraries
suppressPackageStartupMessages({
  library(optparse)
  library(here)
  library(rstan)
  library(posterior)
  library(foreign)
  library(dplyr)
  library(tidyr)
})

# Function to fit and save a model
fit_and_save_model <- function(task, group_type, model_name, model_type, data_list, n_subs, n_trials, n_warmup, n_iter, n_chains, adapt_delta, max_treedepth, model_params, dry_run = FALSE, checkpoint_interval = 1000) {
  model_str <- paste(task, group_type, model_name, sep="_")
  rds_pathname <- file.path(file.path(MODELS_RDS_DIR, model_type), paste0(model_str, "_", model_type, ".rds"))
  stanmodel_arg <- readRDS(rds_pathname)
  
  output_file <- sub('/models/', '/Data/', paste0(tools::file_path_sans_ext(rds_pathname), "_output.rds"))
  checkpoint_file <- paste0(tools::file_path_sans_ext(output_file), "_checkpoint.rds")
  if (!file.exists(dirname(output_file))) {
    stop("Output folder does not exist. Expected path: ", output_file)
  }

  if (dry_run) {
    cat("Dry run for model:", model_str, "\n")
    cat("Stan code:\n")
    cat(rstan::get_stancode(stanmodel_arg), "\n")
    cat("Data list:\n")
    print(str(data_list))
    cat("Parameters:\n")
    cat("n_warmup =", n_warmup, "\n")
    cat("n_iter =", n_iter, "\n")
    cat("n_chains =", n_chains, "\n")
    cat("adapt_delta =", adapt_delta, "\n")
    cat("max_treedepth =", max_treedepth, "\n")
    cat("\n", "File to be save as:", output_file, "\n")
    cat("\n", "Output folder exists:", dirname(output_file), "\n")
    return(NULL)
  }

  cat("Fitting model:", model_str, "\n")
  
  # Create a temporary file for Stan output
  temp_files <- lapply(1:n_chains, function(chain) generate_temp_filename(model_str, chain))
  
  # Check if there's a checkpoint to resume from
  if (file.exists(checkpoint_file)) {
    cat("Resuming from checkpoint\n")
    checkpoint <- readRDS(checkpoint_file)
    current_iter <- checkpoint$current_iter
    accumulated_samples <- checkpoint$accumulated_samples
    step_size <- checkpoint$step_size
    inv_metric <- checkpoint$inv_metrics
    warmup_done <- checkpoint$warmup_done
  } else {
    current_iter <- 0
    accumulated_samples <- list()
    step_size <- NULL
    inv_metric <- NULL
    warmup_done <- FALSE
  }
  
  total_iter <- n_warmup + n_iter
  
  while (current_iter < total_iter) {
    remaining_iter <- min(checkpoint_interval, total_iter - current_iter)
    
    if (!warmup_done) {
      # Initial run or warmup not complete
      iter_to_run <- max(n_warmup + 1, remaining_iter)  # Ensure iter > warmup
      fit <- sampling(stanmodel_arg, data = data_list,
                      iter = iter_to_run, warmup = n_warmup,
                      chains = n_chains, cores = n_chains,
                      control = list(adapt_delta = adapt_delta, max_treedepth = max_treedepth),
                      sample_file = temp_files, refresh = ceiling(iter_to_run/10))
      
      # Extract step size and inverse metric after warmup
      step_size <- get_stepsize(fit)
      inv_metrics <- sapply(temp_files, extract_inv_metric)
      warmup_done <- TRUE
      
      new_samples <- rstan::extract(fit, permuted = FALSE, inc_warmup = FALSE)
      current_iter <- iter_to_run
    } else {
      # Continue sampling post-warmup
      last_draws <- accumulated_samples[[length(accumulated_samples)]]
      last_draws <- last_draws[dim(last_draws)[1],,]  # Get the last iteration for all chains
      init_values <- lapply(1:n_chains, function(i) as.list(last_draws[i,]))
      
      fit <- sampling(stanmodel_arg, data = data_list,
                      iter = remaining_iter, warmup = 0,
                      chains = n_chains, cores = n_chains,
                      control = list(adapt_delta = adapt_delta, max_treedepth = max_treedepth,
                                     adapt_engaged = FALSE, stepsize = step_size, inv_metric = inv_metrics),
                      init = init_values, refresh = ceiling(remaining_iter/10))
      
      new_samples <- rstan::extract(fit, permuted = FALSE, inc_warmup = FALSE)
      current_iter <- current_iter + dim(new_samples)[1]
    }
    
    # Accumulate new samples
    accumulated_samples <- c(accumulated_samples, list(new_samples))
    
    # Save checkpoint
    saveRDS(list(current_iter = current_iter, 
                 accumulated_samples = accumulated_samples, 
                 step_size = step_size, 
                 inv_metric = inv_metric,
                 warmup_done = warmup_done), 
            file = checkpoint_file)
    cat("Checkpoint saved at iteration", current_iter, "\n")
  }
  
  # Combine all accumulated samples into a single array
  all_samples <- do.call(abind::abind, c(accumulated_samples, list(along = 1)))
  
  # Create a stanfit-like object from accumulated samples
  final_fit <- create_stanfit_like_object(all_samples, stanmodel_arg, n_warmup, n_chains)
  
  # Add additional relevant info
  fit = list(
    fit = final_fit,
    n_warmup = n_warmup,
    n_iter = n_iter,
    n_chains = n_chains,
    adapt_delta = adapt_delta,
    max_treedepth = max_treedepth,
    tss = (n_iter - n_warmup) * n_chains,
    all_params = names(final_fit),
    list_params = unique(gsub("\\[.*?\\]", "", names(final_fit))),
    samples = rstan::extract(final_fit)
  )
  
  cat("Extracting parameters\n")
  fit$params <- extract_params(rstan::extract(fit$fit), n_subs, main_params_vec = model_params)
  fit$params <- unname(fit$params)
  
  cat("Saving fitted model to:", output_file, "\n")
  saveRDS(fit, file = output_file)
  
  # Remove checkpoint file after successful completion
  if (file.exists(checkpoint_file) && file.exists(output_file)) {
    file.remove(checkpoint_file)
  }
  
  # Clean up the temporary file
  if (file.exists(temp_files[1])) {
    sapply(temp_files, file.remove)
  }

  return(fit)
}

# Parse command line arguments
option_list = list(
  make_option(c("-m", "--model"), type="character", default=NULL, help="Model name"),
  make_option(c("-t", "--type"), type="character", default="fit", help="Model type (fit, postpc, prepc)"),
  make_option(c("-k", "--task"), type="character", default=NULL, help="Task name"),
  make_option(c("-g", "--group"), type="character", default=NULL, help="Group type (sing, group, group_hier)"),
  make_option(c("-d", "--data"), type="character", default=NULL, 
              help="Comma-separated list of data to extract"),
  make_option(c("-p", "--params"), type="character", default=NULL, 
              help="Comma-separated list of model parameters"),
  make_option(c("--n_subs"), type="integer", default=1000, help="Number of subjects"),
  make_option(c("--n_trials"), type="integer", default=120, help="Number of trials"),
  make_option(c("--RTbound_ms"), type="integer", default=50, help="RT bound in milliseconds"),
  make_option(c("--rt_method"), type="character", default="remove", help="RT method"),
  make_option(c("--n_warmup"), type="integer", default=3000, help="Number of warmup iterations"),
  make_option(c("--n_iter"), type="integer", default=15000, help="Number of iterations"),
  make_option(c("--n_chains"), type="integer", default=4, help="Number of chains"),
  make_option(c("--adapt_delta"), type="double", default=0.95, help="Adapt delta"),
  make_option(c("--max_treedepth"), type="integer", default=12, help="Max tree depth"),
  make_option(c("--seed"), type="integer", default=29518, help="Set seed. Default: 29518"),
  make_option(c("--dry_run"), action="store_true", default=FALSE, help="Perform a dry run")
)

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

# Set up directories
PROJ_DIR <- here::here()
DATA_DIR <- file.path(PROJ_DIR, "Data")
SAFE_DATA_DIR <- file.path(DATA_DIR, "AHRB")
MODELS_DIR <- file.path(PROJ_DIR, "models")
SCRIPT_DIR <- file.path(PROJ_DIR, "scripts")
MODELS_RDS_DIR <- file.path(MODELS_DIR, "rds")
MODELS_RDS_FIT_DIR <- file.path(MODELS_RDS_DIR, "fit")

# Load helper functions
helper_functions_path <- file.path(SCRIPT_DIR, "helper_functions.R")
if (!file.exists(helper_functions_path)) {
  stop("helper_functions.R not found. Expected path: ", helper_functions_path)
}
source(helper_functions_path)

# Set random seed for reproducibility
set.seed(opt$seed)

# Default data and parameters for each model
model_defaults <- list(
  "igt_mod_group_hier_ddm" = list(
    data = c("N", "Nplay", "Npass", "Nplay_max", "Npass_max", "T", "Tsubj", "RTbound", "minRT", "RTpass", "RTplay"),
    params = c("boundary", "tau", "beta", "drift")
  ),
  "igt_mod_group_hier_ev_ddm" = list(
    data = c("N", "T", "Tsubj", "RTbound", "minRT", "RT", "choice", "shown", "outcome"),
    params = c("boundary", "tau", "beta", "drift_con", "wgt_pun", "wgt_rew", "update")
  ),
  "igt_mod_group_hier_ev" = list(
    data = c("N", "T", "Tsubj", "choice", "shown", "outcome"),
    params = c("con", "wgt_pun", "wgt_rew", "update")
  ),
  "igt_mod_group_hier_new_pvl_ddm" = list(
    data = c("N", "T", "Tsubj", "RTbound", "minRT", "RT", "choice", "shown", "outcome"),
    params = c("boundary", "tau", "beta", "drift_con", "exp_upd", "lambda", "alpha", "A", "update_pe", "exp_max")
  ),
  "igt_mod_group_hier_new_pvl" = list(
    data = c("N", "T", "Tsubj", "choice", "shown", "outcome"),
    params = c("con", "exp_upd", "lambda", "alpha", "A", "update_pe", "exp_max")
  )
)

# Main execution
if (is.null(opt$model) || is.null(opt$task) || is.null(opt$group)) {
  stop("Please specify a model, task, and group type using the -m, -k, and -g options.")
}

model_name <- opt$model
task <- opt$task
group_type <- opt$group
full_model_name = paste(task, group_type, model_name, sep="_")

if (!full_model_name %in% names(model_defaults)) {
  cat(full_model_name,"\n")
  cat(names(model_defaults))
  stop("Unrecognized model. Please check the model name.")
}

data_to_extract <- if (!is.null(opt$data)) strsplit(opt$data, ",")[[1]] else model_defaults[[full_model_name]]$data
model_params <- if (!is.null(opt$params)) strsplit(opt$params, ",")[[1]] else model_defaults[[full_model_name]]$params

cat("Preparing data for", full_model_name, "\n")

# Load data
# Check for data file existence
wave1.sav.file <- file.path(SAFE_DATA_DIR, "modigt_data_Wave1.sav")
if (!file.exists(wave1.sav.file)) {
  stop("Data file not found. Expected path: ", wave1.sav.file)
}

if (!opt$dry_run) {
  cat("Preparing data for", full_model_name, "\n")
  # Load data
  wave1.raw <- read.spss(wave1.sav.file, to.data.frame = TRUE)
  
  data_list <- extract_sample_data(wave1.raw, data_to_extract, 
    n_subs = opt$n_subs, n_trials = opt$n_trials, 
    RTbound_ms = opt$RTbound_ms, RTbound_reject_ms = 100, 
    rt_method = opt$rt_method, minrt_ep_ms = 0)
} else {
  cat("Dry run: Data would be loaded from", wave1.sav.file, "\n")
  cat("Dry run: Data to extract:", paste(data_to_extract, collapse=", "), "\n")
  data_list <- NULL  # or you could create a dummy data structure here if needed for dry run
}

fit <- fit_and_save_model(task, group_type, model_name, opt$type, data_list, 
  n_subs = opt$n_subs, n_trials = opt$n_trials,
  n_warmup = opt$n_warmup, n_iter = opt$n_iter, n_chains = opt$n_chains,
  adapt_delta = opt$adapt_delta, max_treedepth = opt$max_treedepth,
  model_params = model_params, dry_run = opt$dry_run)

if (!opt$dry_run) {
  cat("Model fitted and saved successfully.\n")
} else {
  cat("Dry run completed successfully.\n")
}