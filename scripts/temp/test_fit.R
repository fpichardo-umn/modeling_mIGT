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
  "igt_mod_group_hier_ev_ddm_tic" = list(
    data = c("N", "T", "Tsubj", "RTbound", "minRT", "RT", "choice", "shown", "outcome"),
    params = c("boundary", "tau", "beta", "drift_con", "wgt_pun", "wgt_rew", "update")
  ),
  "igt_mod_group_hier_ev_ddm_tdc" = list(
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


model_name <- "ev_ddm_tic"
task <- "igt_mod"
group_type <- "group_hier"
model_type = "fit"
full_model_name = paste(task, group_type, model_name, sep="_")

data_to_extract <- model_defaults[[full_model_name]]$data
model_params <- model_defaults[[full_model_name]]$params


wave1.sav.file <- file.path(SAFE_DATA_DIR, "modigt_data_Wave1.sav")

# Load data
wave1.raw <- read.spss(wave1.sav.file, to.data.frame = TRUE)

opt = list(n_subs = 3, n_trials = 50, RTbound_ms = 50, rt_method = "remove")

data_list <- extract_sample_data(wave1.raw, data_to_extract, 
                                 n_subs = opt$n_subs, n_trials = opt$n_trials, 
                                 RTbound_ms = opt$RTbound_ms, RTbound_reject_ms = 100, 
                                 rt_method = opt$rt_method, minrt_ep_ms = 0)

model_str <- paste(task, group_type, model_name, sep="_")
model_path <- file.path(file.path(MODELS_BIN_DIR, model_type), paste0(model_str, "_", model_type, ".stan"))

current_iter <- 0
accumulated_samples <- list()
accumulated_diagnostics <- list()
step_size <- NULL
inv_metric <- NULL
warmup_done <- FALSE

n_chains = 8

stanmodel_arg <- cmdstan_model(exe_file = model_path)
fit <- stanmodel_arg$sample(
  data = data_list,
  iter_sampling = 10,
  iter_warmup = 5,
  chains = n_chains,
  parallel_chains = 4,
  adapt_delta = 0.9,
  max_treedepth = 10
)

output = c()
for (idx in 1:n_chains){
  output[[length(output) + 1]] = capture.output(fit$output(idx))
}

writeLines(unlist(output), "stan_output.txt")

