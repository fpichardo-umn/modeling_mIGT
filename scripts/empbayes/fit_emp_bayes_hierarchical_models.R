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
  library(ggplot2)
  library(bayesplot)
  library(jsonlite)
})


# Source helper functions
source(here("scripts", "helper_functions_cmdSR.R"))

# Define command line options
option_list = list(
  make_option(c("-m", "--model"), type="character", default=NULL, help="Model name"),
  make_option(c("-k", "--task"), type="character", default=NULL, help="Task name"),
  make_option(c("-d", "--data"), type="character", default=NULL, help="Comma-separated list of data to extract"),
  make_option(c("-p", "--params"), type="character", default=NULL, help="Comma-separated list of model parameters"),
  make_option(c("--n_trials"), type="integer", default=120, help="Number of trials"),
  make_option(c("--RTbound_ms"), type="integer", default=50, help="RT bound in milliseconds"),
  make_option(c("--rt_method"), type="character", default="remove", help="RT method"),
  make_option(c("--n_warmup"), type="integer", default=1000, help="Number of warmup iterations"),
  make_option(c("--n_iter"), type="integer", default=2000, help="Number of sampling iterations"),
  make_option(c("--n_chains"), type="integer", default=4, help="Number of chains"),
  make_option(c("--adapt_delta"), type="double", default=0.95, help="Adapt delta"),
  make_option(c("--max_treedepth"), type="integer", default=12, help="Max tree depth"),
  make_option(c("--seed"), type="integer", default=29518, help="Set seed"),
  make_option(c("--subs_file"), type="character", default=NULL, help="Path to subject IDs file"),
  make_option(c("--debug"), action="store_true", default=FALSE, help="Run in debug mode with reduced dataset and iterations"),
  make_option(c("--dry_run"), action="store_true", default=FALSE, help="Perform a dry run without data processing or model fitting"),
  make_option(c("--check_iter"), type="integer", default=1000, help="Iteration interval for checkpoint runs. Default: 1000")
)

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

# Check for required options
if (is.null(opt$model) || is.null(opt$task)) {
  stop("Please specify a model and task using the -m and -k options.")
}

full_model_name <- paste(opt$task, "group_hier", opt$model, sep="_")

# Set up directories
PROJ_DIR <- here::here()
DATA_DIR <- file.path(PROJ_DIR, "Data")
SAFE_DATA_DIR <- file.path(DATA_DIR, "AHRB")
MODELS_DIR <- file.path(PROJ_DIR, "models")
MODELS_BIN_DIR <- file.path(MODELS_DIR, "bin")
DATA_RDS_DIR <- file.path(DATA_DIR, "rds")
DATA_RDS_eB_DIR <- file.path(DATA_RDS_DIR, "empbayes")
DATA_TXT_DIR <- file.path(DATA_DIR, "txt")
DATA_SUBS_DIR <- file.path(DATA_TXT_DIR, "subs")

# Ensure directories exist
dir.create(DATA_RDS_eB_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(DATA_SUBS_DIR, recursive = TRUE, showWarnings = FALSE)

# Set random seed for reproducibility
set.seed(opt$seed)

# Load model defaults
model_defaults_file <- here::here("models", "params", paste0(opt$task, "_defaults.json"))

if (!file.exists(model_defaults_file)) {
  stop("Model defaults file not found: ", model_defaults_file)
}

model_defaults <- fromJSON(model_defaults_file)

if (!full_model_name %in% names(model_defaults)) {
  cat(full_model_name, "\n")
  cat(names(model_defaults))
  stop("Unrecognized model. Please check the model name.")
}
# Use defaults if data or params are not provided
if (is.null(opt$data)) {
  opt$data <- paste(model_defaults[[full_model_name]]$data, collapse = ",")
}
if (is.null(opt$params)) {
  if (is.null(model_defaults[[full_model_name]]$params)) {
    stop("No default parameters found for model: ", full_model_name)
  }
  opt$params <- model_defaults[[full_model_name]]$params
}

# Adjust parameters if in debug mode
if (opt$debug) {
  opt$n_trials <- min(opt$n_trials, 60)
  opt$n_warmup <- 500
  opt$n_iter <- 1000
  opt$n_chains <- 2
  opt$adapt_delta <- 0.8  # Reduced adapt_delta for faster (but potentially less stable) sampling
  opt$max_treedepth <- 8  # Reduced max_treedepth
}


if (!opt$dry_run) {
  # Load data
  wave1.igt.raw <- read.spss(file.path(SAFE_DATA_DIR, "modigt_data_Wave1.sav"), to.data.frame = TRUE)
  wave1.risk.data <- read.spss(file.path(SAFE_DATA_DIR, "AHRB.P1W1_v11_AW_v2.sav"), to.data.frame = TRUE)
  
  # Load or generate subject IDs
  subs_df <- load_or_generate_subs(opt$subs_file, wave1.igt.raw, wave1.risk.data, opt$n_trials, opt$debug)
  
  # Prepare data for Stan (using hierarchical subset)
  hier_data <- wave1.igt.raw %>% 
    filter(sid %in% subs_df$sid[subs_df$set == "hier"]) %>%
    dplyr::group_by(sid) %>%
    slice(1:opt$n_trials) %>%
    ungroup()
  
  data_list <- extract_sample_data(hier_data, strsplit(opt$data, ",")[[1]], 
                                   n_trials = opt$n_trials,
                                   RTbound_ms = opt$RTbound_ms,
                                   rt_method = opt$rt_method)
} else {
  cat("Dry run: Data would be loaded from", file.path(SAFE_DATA_DIR, "modigt_data_Wave1.sav"), 
      "and", file.path(SAFE_DATA_DIR, "AHRB.P1W1_v11_AW_v2.sav"), "\n")
  cat("Dry run: Subject IDs would be loaded or generated\n")
  cat("Dry run: Data to extract:", opt$data, "\n")
  data_list <- NULL
  subs_df <- data.frame(set = c("hier", "train", "test"), 
                        count = c(10, 10, 10))  # Dummy data for dry run
}

# Fit the initial hierarchical model

if (!opt$dry_run) {
  # Fit the initial hierarchical model
  fit <- fit_and_save_model(opt$task, "group_hier", opt$model, "fit", data_list, 
                            n_subs = nrow(subs_df[subs_df$set == "hier",]), 
                            n_trials = opt$n_trials,
                            n_warmup = opt$n_warmup, n_iter = opt$n_iter, n_chains = opt$n_chains,
                            adapt_delta = opt$adapt_delta, max_treedepth = opt$max_treedepth,
                            model_params = opt$params, checkpoint_interval = min(opt$check_iter, opt$n_warmup),
                            output_dir = DATA_RDS_eB_DIR, emp_bayes = T)
  
  # Check model diagnostics
  fit$empBayesdiagnostics <- check_model_diagnostics(fit)
  
  # Save the output with a descriptor
  saveRDS(fit, file = file.path(DATA_RDS_eB_DIR, paste0(full_model_name, "_desc-emp_hier_output.rds")))
  
  # Print diagnostic results
  cat("\nDiagnostic Results:\n")
  cat("High R-hat parameters:", paste(fit$empBayesdiagnostics$high_rhat, collapse = ", "), "\n")
  cat("Low ESS parameters:", paste(fit$empBayesdiagnostics$low_ess, collapse = ", "), "\n")
  cat("High MCSE parameters:", paste(fit$empBayesdiagnostics$high_mcse, collapse = ", "), "\n")
  
  cat("\nInitial hierarchical model fit completed and saved.\n")
} else {
  fit_and_save_model(opt$task, "group_hier", opt$model, "fit", c(), 
                     n_subs = "TBD", 
                     n_trials = opt$n_trials,
                     n_warmup = opt$n_warmup, n_iter = opt$n_iter, n_chains = opt$n_chains,
                     adapt_delta = opt$adapt_delta, max_treedepth = opt$max_treedepth,
                     model_params = opt$params, checkpoint_interval = opt$check_iter,
                     output_dir = DATA_RDS_eB_DIR, emp_bayes = T,
                     dry_run = T)
  cat("\nDry run: Model diagnostics would be checked and results saved.\n")
}


cat("\nScript execution ", ifelse(opt$dry_run, "dry run ", ""), "completed.\n")