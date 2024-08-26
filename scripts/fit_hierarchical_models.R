#!/usr/bin/env Rscript

# Load required packages
library(plyr)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(grid)
library(kableExtra)
library(tidyr)
library(here)
library(foreign)
library(bayesplot)
library(posterior)
library(rstan)

# Set up directories
PROJ_DIR <- here::here()
DATA_DIR <- file.path(PROJ_DIR, "Data")
SAFE_DATA_DIR <- file.path(DATA_DIR, "AHRB")
MODELS_DIR <- file.path(PROJ_DIR, "models")
MODELS_RDS_DIR <- file.path(MODELS_DIR, "rds")
MODELS_RDS_FIT_DIR <- file.path(MODELS_RDS_DIR, "fit")


# Data extraction Params
n_subs = 1000
n_trials = 120
RTbound_ms = 50
rt_method = "remove"

# Fit parameters
n_warmup = 3000
n_iter = 15000
n_chains = 4
adapt_delta = 0.95
max_treedepth = 12


# Load helper functions
source("helper_functions.R")  # Assume all functions are moved to this file

# Load data
wave1.sav.file <- file.path(SAFE_DATA_DIR, "modigt_data_Wave1.sav")
wave1.raw <- read.spss(wave1.sav.file, to.data.frame = TRUE)

# Set random seed for reproducibility
set.seed(12345)

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


# DDM Hierarchical Model
cat("Preparing data for DDM Hierarchical Model\n")
ddm_data_to_extract <- c("N", "Nplay", "Npass", "T", "Tsubj", "RTbound", "minRT", "RTpass", "RTplay")
ddm_model_params <- c("boundary", "tau", "beta", "drift")

ddm_data_list <- extract_sample_data(wave1.raw, ddm_data_to_extract, 
                                        n_subs = n_subs, n_trials = n_trials, 
                                        RTbound_ms = RTbound_ms, RTbound_reject_ms = 100, 
                                        rt_method = rt_method, minrt_ep_ms = 0)

ddm_fit <- fit_and_save_model("igt_mod_group_hier_ddm", ddm_data_list, 
                                 n_subs = n_subs, n_trials = n_trials,
                                 n_warmup = n_warmup, n_iter = n_iter, n_chains = n_chains,
                                 adapt_delta = adapt_delta, max_treedepth = max_treedepth,
                                 model_params = ddm_model_params)


# EV-DDM Hierarchical Model
cat("Preparing data for EV-DDM Hierarchical Model\n")
ev_ddm_data_to_extract <- c("N", "T", "Tsubj", "RTbound", "minRT", "RT", "choice", "shown", "outcome")
ev_ddm_model_params <- c("boundary", "tau", "beta", "drift_con", "wgt_pun", "wgt_rew", "update")

ev_ddm_data_list <- extract_sample_data(wave1.raw, ev_ddm_data_to_extract, 
                                        n_subs = n_subs, n_trials = n_trials, 
                                        RTbound_ms = RTbound_ms, RTbound_reject_ms = 100, 
                                        rt_method = rt_method, minrt_ep_ms = 0)

ev_ddm_fit <- fit_and_save_model("igt_mod_group_hier_ev_ddm", ev_ddm_data_list, 
                                 n_subs = n_subs, n_trials = n_trials,
                                 n_warmup = n_warmup, n_iter = n_iter, n_chains = n_chains,
                                 adapt_delta = adapt_delta, max_treedepth = max_treedepth,
                                 model_params = ev_ddm_model_params)


# EV Hierarchical Model
cat("Preparing data for EV-DDM Hierarchical Model\n")
ev_data_to_extract <- c("N", "T", "Tsubj", "choice", "shown", "outcome")
ev_model_params <- c("con", "wgt_pun", "wgt_rew", "update")

ev_data_list <- extract_sample_data(wave1.raw, ev_data_to_extract, 
                                        n_subs = n_subs, n_trials = n_trials, 
                                        RTbound_ms = RTbound_ms, RTbound_reject_ms = 100, 
                                        rt_method = rt_method, minrt_ep_ms = 0)

ev_fit <- fit_and_save_model("igt_mod_group_hier_ev", ev_data_list, 
                                 n_subs = n_subs, n_trials = n_trials,
                                 n_warmup = n_warmup, n_iter = n_iter, n_chains = n_chains,
                                 adapt_delta = adapt_delta, max_treedepth = max_treedepth,
                                 model_params = ev_model_params)


# PVL-DDM Hierarchical Model
cat("Preparing data for PVL-DDM Hierarchical Model\n")
pvl_ddm_data_to_extract <- c("N", "T", "Tsubj", "RTbound", "minRT", "RT", "choice", "shown", "outcome")
pvl_ddm_model_params <- c("boundary", "tau", "beta", "drift_con", "exp_upd", "lambda", "alpha", "A", "update_pe", "exp_max")

pvl_ddm_data_list <- extract_sample_data(wave1.raw, pvl_ddm_data_to_extract, 
                                        n_subs = n_subs, n_trials = n_trials, 
                                        RTbound_ms = RTbound_ms, RTbound_reject_ms = 100, 
                                        rt_method = rt_method, minrt_ep_ms = 0)

pvl_ddm_fit <- fit_and_save_model("igt_mod_group_hier_new_pvl_ddm", pvl_ddm_data_list, 
                                 n_subs = n_subs, n_trials = n_trials,
                                 n_warmup = n_warmup, n_iter = n_iter, n_chains = n_chains,
                                 adapt_delta = adapt_delta, max_treedepth = max_treedepth,
                                 model_params = ev_ddm_model_params)


# PVL Hierarchical Model
cat("Preparing data for PVL-DDM Hierarchical Model\n")
pvl_data_to_extract <- c("N", "T", "Tsubj", "choice", "shown", "outcome")
pvl_model_params <- c("con", "exp_upd", "lambda", "alpha", "A", "update_pe", "exp_max")

pvl_data_list <- extract_sample_data(wave1.raw, pvl_ddm_data_to_extract, 
                                         n_subs = n_subs, n_trials = n_trials, 
                                         RTbound_ms = RTbound_ms, RTbound_reject_ms = 100, 
                                         rt_method = rt_method, minrt_ep_ms = 0)

pvl_fit <- fit_and_save_model("igt_mod_group_hier_new_pvl", pvl_data_list, 
                                  n_subs = n_subs, n_trials = n_trials,
                                  n_warmup = n_warmup, n_iter = n_iter, n_chains = n_chains,
                                  adapt_delta = adapt_delta, max_treedepth = max_treedepth,
                                  model_params = pvl_model_params)


cat("All models fitted and saved successfully.\n")