#!/usr/bin/env Rscript

# Load required libraries
suppressPackageStartupMessages({
  library(optparse)
  library(here)
  library(cmdstanr)
  library(posterior)
  library(foreign)
  library(dplyr)
  library(tidyr)
})

# Function to fit and save a model
fit_and_save_model <- function(task, group_type, model_name, model_type, data_list, n_subs, n_trials, n_warmup, n_iter, n_chains, adapt_delta, max_treedepth, model_params, dry_run = FALSE, checkpoint_interval = 1000) {
  model_str <- paste(task, group_type, model_name, sep="_")
  model_path <- file.path(file.path(MODELS_BIN_DIR, model_type), paste0(model_str, "_", model_type, ".stan"))
  stanmodel_arg <- cmdstan_model(exe_file = model_path)
  
  output_file <- sub('/models/bin', '/Data/rds', paste0(tools::file_path_sans_ext(model_path), "_output.rds"))
  checkpoint_file <- paste0(tools::file_path_sans_ext(output_file), "_checkpoint.rds")
  if (!file.exists(dirname(output_file))) {
    stop("Output folder does not exist. Expected path: ", output_file)
  }
  
  if (dry_run) {
    cat("Dry run for model:", model_str, "\n")
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
  
  # Check if there's a checkpoint to resume from
  if (file.exists(checkpoint_file)) {
    cat("Resuming from checkpoint\n")
    checkpoint <- readRDS(checkpoint_file)
    current_iter <- checkpoint$current_iter
    accumulated_samples <- checkpoint$accumulated_samples
    accumulated_diagnostics <- checkpoint$accumulated_diagnostics
    step_size <- checkpoint$step_size
    inv_metric <- checkpoint$inv_metric
    warmup_done <- checkpoint$warmup_done
  } else {
    current_iter <- 0
    accumulated_samples <- list()
    accumulated_diagnostics <- list()
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
      fit <- stanmodel_arg$sample(
        data = data_list,
        iter_sampling = iter_to_run - n_warmup,
        iter_warmup = n_warmup,
        chains = n_chains,
        parallel_chains = n_chains,
        adapt_delta = adapt_delta,
        max_treedepth = max_treedepth,
        refresh = ceiling(iter_to_run/10)
      )
      
      # Extract step size and inverse metric after warmup
      step_size <- fit$metadata()$step_size_adaptation
      inv_metric <- fit$inv_metric()
      warmup_done <- TRUE
      
      new_samples <- fit$draws()
      current_iter <- iter_to_run
    } else {
      # Continue sampling post-warmup
      last_draws <- accumulated_samples[[length(accumulated_samples)]]
      last_draws <- last_draws[dim(last_draws)[1],,]  # Get the last iteration for all chains
      init_list <- lapply(1:n_chains, function(chain_idx) create_init_list(last_draws, chain_idx))
      
      fit <- stanmodel_arg$sample(
        data = data_list,
        iter_sampling = remaining_iter,
        iter_warmup = 0,
        chains = n_chains,
        parallel_chains = n_chains,
        adapt_delta = adapt_delta,
        metric='dense_e',
        max_treedepth = max_treedepth,
        init = init_list,
        inv_metric = inv_metric,
        adapt_engaged = FALSE,
        step_size = step_size,
        refresh = ceiling(remaining_iter/10)
      )
      
      new_samples <- fit$draws()
      current_iter <- current_iter + dim(new_samples)[1]
    }
    
    # Accumulate new samples
    accumulated_samples <- c(accumulated_samples, list(new_samples))
    
    # Extract and accumulate diagnostics
    new_diagnostics <- fit$sampler_diagnostics()
    accumulated_diagnostics <- c(accumulated_diagnostics, list(new_diagnostics))
    
    # Save checkpoint
    saveRDS(list(current_iter = current_iter, 
                 accumulated_samples = accumulated_samples,
                 accumulated_diagnostics = accumulated_diagnostics,
                 step_size = step_size, 
                 inv_metric = inv_metric,
                 warmup_done = warmup_done), 
            file = checkpoint_file)
    cat("Checkpoint saved at iteration", current_iter, "\n")
  }
  
  # Combine all accumulated samples into a single draws object
  all_samples <- do.call(posterior::bind_draws, c(accumulated_samples, along = "iteration"))
  
  # Combine all accumulated diagnostics
  all_diagnostics <- do.call(posterior::bind_draws, c(accumulated_diagnostics, along = "iteration"))
  
  # Calculate diagnostic summaries
  num_divergent <- colSums(all_diagnostics[, , "divergent__"])
  num_max_treedepth <- colSums(all_diagnostics[, , "treedepth__"] == max_treedepth)
  ebfmi <- sapply(1:n_chains, function(chain) {
    energy <- all_diagnostics[, chain, "energy__"]
    sum(diff(energy)^2) / (length(energy) - 1) / var(energy)
  })
  
  # Calculate summary statistics based on all accumulated samples
  summary_stats <- posterior::summarize_draws(all_samples)
  
  # Apply the function to each parameter
  param_names <- dimnames(all_samples)[[3]]
  diagnostics <- lapply(param_names, function(param) {
    param_draws <- all_samples[, , param]
    calculate_param_diagnostics(param_draws)
  })
  
  # Combine the results into a data frame
  diagnostics_df <- do.call(rbind, diagnostics)
  rownames(diagnostics_df) <- param_names
  
  # Create a fit object with all necessary information
  fit = list(
    draws = all_samples,
    sampler_diagnostics = all_diagnostics,
    n_warmup = n_warmup,
    n_iter = n_iter,
    n_params = dim(all_samples)[3],
    n_chains = n_chains,
    adapt_delta = adapt_delta,
    max_treedepth = max_treedepth,
    tss = dim(all_samples)[1],
    all_params = param_names,
    list_params = unique(gsub("\\[.*?\\]", "", param_names)),
    summary_stats = summary_stats,
    diagnostic_summary = list(
      num_divergent = num_divergent,
      num_max_treedepth = num_max_treedepth,
      ebfmi = ebfmi
    ),
    diagnostics = diagnostics_df,
    cmdstan_version = cmdstan_version(),
    model_name = model_str,
    n_runs = length(accumulated_samples)  # Number of sampling runs
  )
  
  # Print diagnostic warnings
  total_iterations <- n_chains * (n_iter - n_warmup)
  total_divergent <- sum(num_divergent)
  total_max_treedepth <- sum(num_max_treedepth)
  
  if (total_divergent > 0) {
    cat(sprintf("\nWarning: %d of %d (%.1f%%) transitions ended with a divergence.\n", 
                total_divergent, total_iterations, 100 * total_divergent / total_iterations))
    cat("See https://mc-stan.org/misc/warnings for details.\n")
  }
  
  if (total_max_treedepth > 0) {
    cat(sprintf("\nWarning: %d of %d (%.1f%%) transitions hit the maximum treedepth limit of %d.\n", 
                total_max_treedepth, total_iterations, 100 * total_max_treedepth / total_iterations, max_treedepth))
    cat("See https://mc-stan.org/misc/warnings for details.\n")
  }
  
  cat("Extracting parameters\n")
  fit$params <- extract_params(fit$all_params, n_subs, main_params_vec = model_params)
  fit$params <- unname(fit$params)
  
  cat("Saving fitted model to:", output_file, "\n")
  saveRDS(fit, file = output_file)
  
  # Remove checkpoint file after successful completion
  if (file.exists(checkpoint_file) && file.exists(output_file)) {
    file.remove(checkpoint_file)
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
  make_option(c("--dry_run"), action="store_true", default=FALSE, help="Perform a dry run"),
  make_option(c("--check_iter"), type="integer", default=1000, help="Iteration interval for checkpoint runs. Default: 1000")
)

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

# Set up directories
PROJ_DIR <- here::here()
DATA_DIR <- file.path(PROJ_DIR, "Data")
SAFE_DATA_DIR <- file.path(DATA_DIR, "AHRB")
MODELS_DIR <- file.path(PROJ_DIR, "models")
SCRIPT_DIR <- file.path(PROJ_DIR, "scripts")
MODELS_BIN_DIR <- file.path(MODELS_DIR, "bin")

# Load helper functions
helper_functions_path <- file.path(SCRIPT_DIR, "helper_functions_cmdSR.R")
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
                          model_params = model_params, dry_run = opt$dry_run, checkpoint_interval = opt$check_iter)

if (!opt$dry_run) {
  cat("Model fitted and saved successfully.\n")
} else {
  cat("Dry run completed successfully.\n")
}