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
  library(truncnorm)
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
  opt$params <- paste(model_defaults[[full_model_name]]$params, collapse = ",")
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

# Function to load or generate subject IDs
load_or_generate_subs <- function(subs_file, igt_data, risk_data, n_trials, debug = FALSE) {
  if (!is.null(subs_file) && file.exists(subs_file)) {
    # Load existing subject IDs
    subs_df <- read.table(subs_file, header = TRUE, stringsAsFactors = FALSE)
    cat("Loaded subject IDs from file:", subs_file, "\n")
  } else {
    # Identify shared subjects and merge relevant information
    shared_subs <- intersect(unique(igt_data$sid), unique(risk_data$sid))
    risk_shared_df <- risk_data[risk_data$sid %in% shared_subs,]
    sid_to_grpid <- unique(igt_data[, c("sid", "grpid")])
    risk_shared_df$grpid <- sid_to_grpid$grpid[match(risk_shared_df$sid, sid_to_grpid$sid)]
    
    # Create balanced strata
    stratified_data <- create_balanced_strata(risk_shared_df)
    
    # Filter data for required number of trials
    filtered_data <- igt_data %>%
      dplyr::group_by(sid) %>%
      filter(n() >= n_trials) %>%
      slice(1:n_trials) %>%
      ungroup()
    
    # Calculate proportions for splitting
    n_total <- length(unique(filtered_data$sid))
    p_hier <- 200 / n_total
    p_train <- 0.8 - p_hier
    
    # Split the data
    split_data <- stratified_split(stratified_data, c(p_hier, p_train, 0.2))
    
    subs_df <- split_data %>%
      mutate(
        set = case_when(
          split == 1 ~ "hier",
          split == 2 ~ "train",
          split == 3 ~ "test"
        ),
        use = ifelse(set %in% c("hier", "train"), "training", "testing")
      ) %>%
      select(sid, set, use)
    
    # Save the generated subject IDs
    if (is.null(subs_file)) {
      subs_file <- file.path(DATA_SUBS_DIR, "subject_ids.txt")
    }
    write.table(subs_df, file = subs_file, row.names = FALSE, sep = "\t", quote = FALSE)
    cat("Generated and saved subject IDs to file:", subs_file, "\n")
  }
  
  if (debug) {
    # Limit to a small number of subjects for debugging
    subs_df <- subs_df %>% dplyr::group_by(set) %>% slice_head(n = 10) %>% ungroup()
  }
  
  return(subs_df)
}

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

# Fit the initial hierarchical model
fit <- fit_and_save_model(opt$task, opt$group, opt$model, "fit", data_list, 
                          n_subs = nrow(subs_df[subs_df$set == "hier",]), 
                          n_trials = opt$n_trials,
                          n_warmup = opt$n_warmup, n_iter = opt$n_iter, n_chains = opt$n_chains,
                          adapt_delta = opt$adapt_delta, max_treedepth = opt$max_treedepth,
                          model_params = strsplit(opt$params, ",")[[1]],
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