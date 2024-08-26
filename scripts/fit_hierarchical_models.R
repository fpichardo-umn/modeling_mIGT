#!/usr/bin/env Rscript

# Function to check and install packages
install_if_missing <- function(packages) {
  new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
  if(length(new_packages)) {
    cat("Installing missing packages:", paste(new_packages, collapse=", "), "\n")
    install.packages(new_packages, repos="https://cran.rstudio.com/")
  }
}

# Function to fit and save a model
fit_and_save_model <- function(model_str, data_list, n_subs, n_trials, n_warmup, n_iter, n_chains, adapt_delta, max_treedepth, model_params) {
  rds_pathname <- file.path(MODELS_RDS_FIT_DIR, paste0(model_str, "_fit.rds"))
  stanmodel_arg <- readRDS(rds_pathname)
  
  cat("Fitting model:", model_str, "\n")
  fit <- fit_stan_model(stanmodel_arg, data_list,
                        n_warmup = n_warmup, n_iter = n_iter + n_warmup,
                        n_chains = n_chains, adapt_delta = adapt_delta,
                        max_treedepth = max_treedepth, parallel = TRUE)
  
  cat("Extracting parameters\n")
  fit$params <- extract_params(fit$all_params, n_subs, main_params_vec = model_params)
  fit$params <- unname(fit$params)
  
  output_file <- sub('/models/', '/data/', paste0(tools::file_path_sans_ext(rds_pathname), "_fit_output.rds"))
  cat("Saving fitted model to:", output_file, "\n")
  saveRDS(fit, file = output_file)
  
  return(fit)
}

# List of required packages
required_packages <- c("plyr", "dplyr", "ggplot2", "gridExtra", "grid", "kableExtra", 
                       "tidyr", "here", "foreign", "bayesplot", "posterior", "rstan", "optparse")

# Check and install missing packages
install_if_missing(required_packages)

# Load required packages
lapply(required_packages, library, character.only = TRUE)

# Parse command line arguments
option_list <- list(
  make_option(c("-m", "--model"), type="character", default=NULL, 
              help="Model string (e.g., 'igt_mod_group_hier_new_pvl')"),
  make_option(c("-d", "--data"), type="character", default=NULL, 
              help="Comma-separated list of data to extract"),
  make_option(c("-p", "--params"), type="character", default=NULL, 
              help="Comma-separated list of model parameters"),
  make_option(c("--n_subs"), type="integer", default=1000, help="Number of subjects"),
  make_option(c("--n_trials"), type="integer", default=120, help="Number of trials"),
  make_option(c("--RTbound_ms"), type="integer", default=50, help="RT bound in milliseconds"),
  make_option(c("--rt_method"), type="character", default="remove", help="RT method"),
  make_option(c("--n_warmup"), type="integer", default=3000, help="Number of warmup iterations"),
  make_option(c("--n_iter"), type="integer", default=15000, help="Number of iterations (not including warmups)"),
  make_option(c("--n_chains"), type="integer", default=4, help="Number of chains"),
  make_option(c("--adapt_delta"), type="double", default=0.95, help="Adapt delta"),
  make_option(c("--max_treedepth"), type="integer", default=12, help="Max tree depth"),
  make_option(c("--seed"), type="integer", default=29518, help="Set seed. Default: 29518")
)

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

# Set up directories
PROJ_DIR <- here::here()
DATA_DIR <- file.path(PROJ_DIR, "Data")
SAFE_DATA_DIR <- file.path(DATA_DIR, "AHRB")
MODELS_DIR <- file.path(PROJ_DIR, "models")
MODELS_RDS_DIR <- file.path(MODELS_DIR, "rds")
MODELS_RDS_FIT_DIR <- file.path(MODELS_RDS_DIR, "fit")

# Load helper functions
script_dir <- dirname(sys.frame(1)$ofile)
source(file.path(script_dir, "helper_functions.R"))

# Load data
wave1.sav.file <- file.path(SAFE_DATA_DIR, "modigt_data_Wave1.sav")
wave1.raw <- read.spss(wave1.sav.file, to.data.frame = TRUE)

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
if (is.null(opt$model)) {
  stop("Please specify a model using the -m or --model option.")
}

model_str <- opt$model
if (!model_str %in% names(model_defaults)) {
  stop("Unrecognized model. Please check the model name.")
}

data_to_extract <- if (!is.null(opt$data)) strsplit(opt$data, ",")[[1]] else model_defaults[[model_str]]$data
model_params <- if (!is.null(opt$params)) strsplit(opt$params, ",")[[1]] else model_defaults[[model_str]]$params

cat("Preparing data for", model_str, "\n")
data_list <- extract_sample_data(wave1.raw, data_to_extract, 
                                 n_subs = opt$n_subs, n_trials = opt$n_trials, 
                                 RTbound_ms = opt$RTbound_ms, RTbound_reject_ms = 100, 
                                 rt_method = opt$rt_method, minrt_ep_ms = 0)

fit <- fit_and_save_model(model_str, data_list, 
                          n_subs = opt$n_subs, n_trials = opt$n_trials,
                          n_warmup = opt$n_warmup, n_iter = opt$n_iter, n_chains = opt$n_chains,
                          adapt_delta = opt$adapt_delta, max_treedepth = opt$max_treedepth,
                          model_params = model_params)

cat("Model fitted and saved successfully.\n")