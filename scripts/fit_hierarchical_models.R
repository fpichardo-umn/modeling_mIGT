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
fit_and_save_model <- function(task, group_type, model_name, model_type, data_list, n_subs, n_trials, n_warmup, n_iter, n_chains, adapt_delta, max_treedepth, model_params, dry_run = FALSE) {
  model_str <- paste(task, group_type, model_name, sep="_")
  rds_pathname <- file.path(file.path(MODELS_RDS_DIR, model_type), paste0(model_str, "_", model_type, ".rds"))
  stanmodel_arg <- readRDS(rds_pathname)

  output_file <- sub('/models/', '/Data/', paste0(tools::file_path_sans_ext(rds_pathname), "_output.rds"))
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
  fit <- fit_stan_model(stanmodel_arg, data_list,
                        n_warmup = n_warmup, n_iter = n_iter + n_warmup,
                        n_chains = n_chains, adapt_delta = adapt_delta,
                        max_treedepth = max_treedepth, parallel = TRUE)

  cat("Extracting parameters\n")
  fit$params <- extract_params(fit$all_params, n_subs, main_params_vec = model_params)
  fit$params <- unname(fit$params)
  
  cat("Saving fitted model to:", output_file, "\n")
  saveRDS(fit, file = output_file)

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
    data = c("N", "Nplay", "Npass", "T", "Tsubj", "RTbound", "minRT", "RTpass", "RTplay"),
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