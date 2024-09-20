#!/usr/bin/env Rscript

# Load required libraries
suppressPackageStartupMessages({
  library(optparse)
  library(here)
  library(cmdstanr)
  library(posterior)
  library(dplyr)
  library(tidyr)
  library(foreign)
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
  make_option(c("-g", "--group"), type="character", default=NULL, help="Group type (sing, group, group_hier)"),
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
  make_option(c("--debug"), action="store_true", default=FALSE, help="Run in debug mode with reduced dataset and iterations")
)

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

# Check for required options
if (is.null(opt$model) || is.null(opt$task) || is.null(opt$group)) {
  stop("Please specify a model, task, and group type using the -m, -k, and -g options.")
}

full_model_name <- paste(opt$task, opt$group, opt$model, sep="_")

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

model_name_for_defaults = sub("group_emp", "group_hier", full_model_name)
if (!model_name_for_defaults %in% names(model_defaults)) {
  cat(full_model_name, "\n")
  cat(paste(names(model_defaults), "\n"))
  stop("Unrecognized model. Please check the model name.")
}

# Use defaults if data or params are not provided
if (is.null(opt$data)) {
  opt$data <- paste(model_defaults[[model_name_for_defaults]]$data, collapse = ",")
}
if (is.null(opt$params)) {
  if (is.null(model_defaults[[model_name_for_defaults]]$params)) {
    stop("No default parameters found for model: ", full_model_name)
  }
  opt$params <- model_defaults[[model_name_for_defaults]]$params
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

# Load data
wave1.igt.raw <- read.spss(file.path(SAFE_DATA_DIR, "modigt_data_Wave1.sav"), to.data.frame = TRUE)
wave1.risk.data <- read.spss(file.path(SAFE_DATA_DIR, "AHRB.P1W1_v11_AW_v2.sav"), to.data.frame = TRUE)

# Load subject IDs
subs_df <- load_or_generate_subs(opt$subs_file, wave1.igt.raw, wave1.risk.data, opt$n_trials, opt$debug)

# Load informative priors
priors_file <- file.path(DATA_RDS_eB_DIR, paste0(model_name_for_defaults, "_informative_priors.rds"))
if (!file.exists(priors_file)) {
  stop("Informative priors file not found: ", priors_file)
}
informative_priors <- readRDS(priors_file)

# Prepare data for Stan (using training subset)
train_data <- wave1.igt.raw %>% 
  filter(sid %in% subs_df$sid[subs_df$use == "training"]) %>%
  dplyr::group_by(sid) %>%
  slice(1:opt$n_trials) %>%
  ungroup()

data_list <- extract_sample_data(train_data, strsplit(opt$data, ",")[[1]], 
                                 n_trials = opt$n_trials,
                                 RTbound_ms = opt$RTbound_ms,
                                 rt_method = opt$rt_method)

# Fit the empirical Bayes model
emp_fit <- fit_and_save_model(opt$task, opt$group, opt$model, "fit", data_list, 
                              n_subs = nrow(subs_df[subs_df$use == "training",]), 
                              n_trials = opt$n_trials,
                              n_warmup = opt$n_warmup, n_iter = opt$n_iter, n_chains = opt$n_chains,
                              adapt_delta = opt$adapt_delta, max_treedepth = opt$max_treedepth,
                              model_params = opt$params,
                              output_dir = DATA_RDS_eB_DIR, emp_bayes = TRUE,
                              informative_priors = informative_priors)

# Check model diagnostics
emp_fit$empBayesdiagnostics <- check_model_diagnostics(emp_fit)

# Load the original hierarchical fit for comparison
hier_fit_file <- file.path(DATA_RDS_eB_DIR, paste0(model_name_for_defaults, "_desc-emp_hier_output.rds"))
hier_fit <- readRDS(hier_fit_file)

# Validate empirical Bayes results
validation_results <- validate_empirical_bayes(hier_fit, emp_fit, emp_fit$model_params, subs_df)

# Save validation results
saveRDS(validation_results, file = file.path(DATA_RDS_eB_DIR, paste0(full_model_name, "_validation_results.rds")))

# Print summary of validation results
cat("\nValidation Results Summary:\n")
for (param in names(validation_results)) {
  cat(param, ":\n")
  cat("  Mean difference (across subjects):", mean(validation_results[[param]]$mean_diff), "\n")
  cat("  Median difference (across subjects):", median(validation_results[[param]]$median_diff), "\n")
  cat("  Mean SD of difference (across subjects):", mean(validation_results[[param]]$sd_diff), "\n")
  cat("  95% CI of mean difference:", 
      quantile(validation_results[[param]]$mean_diff, 0.025), "to", 
      quantile(validation_results[[param]]$mean_diff, 0.975), "\n\n")
}

# Save plots
pdf(file.path(DATA_RDS_eB_DIR, paste0(full_model_name, "_validation_plots.pdf")))
for (param in names(validation_results)) {
  print(validation_results[[param]]$plot)
}
dev.off()

# Save detailed subject-level results
subject_results <- data.frame()
for (param in names(validation_results)) {
  param_results <- data.frame(
    parameter = param,
    subject = validation_results[[param]]$subjects,
    mean_diff = validation_results[[param]]$mean_diff,
    median_diff = validation_results[[param]]$median_diff,
    sd_diff = validation_results[[param]]$sd_diff,
    lower_ci = validation_results[[param]]$quantiles[,1],
    upper_ci = validation_results[[param]]$quantiles[,4]
  )
  subject_results <- rbind(subject_results, param_results)
}
write.csv(subject_results, file = file.path(DATA_RDS_eB_DIR, paste0(full_model_name, "_subject_level_validation.csv")), row.names = FALSE)

cat("Empirical Bayes model fit, diagnostics, and validation completed.\n")