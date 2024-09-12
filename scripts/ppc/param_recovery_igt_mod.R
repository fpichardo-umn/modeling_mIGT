#!/usr/bin/env Rscript

# Load required libraries
suppressPackageStartupMessages({
  library(optparse)
  library(here)
  library(cmdstanr)
  library(posterior)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
})

# Set up directories
PROJ_DIR <- here::here()
DATA_DIR <- file.path(PROJ_DIR, "Data")
DATA_RDS_DIR <- file.path(DATA_DIR, "rds")
DATA_RDS_SIM_DIR <- file.path(DATA_RDS_DIR, "sim")
MODELS_DIR <- file.path(PROJ_DIR, "models")
MODELS_BIN_DIR <- file.path(MODELS_DIR, "bin")
SCRIPT_DIR <- file.path(PROJ_DIR, "scripts")

# Source helper functions
source(file.path(SCRIPT_DIR, "helper_functions_cmdSR.R"))

# Parse command line arguments
option_list = list(
  make_option(c("-n", "--n_subjects"), type="integer", default=100, help="Number of subjects"),
  make_option(c("-w", "--n_warmup"), type="integer", default=1000, help="Number of warmup iterations"),
  make_option(c("-i", "--n_iter"), type="integer", default=2000, help="Number of sampling iterations"),
  make_option(c("-c", "--n_chains"), type="integer", default=4, help="Number of chains"),
  make_option(c("-a", "--adapt_delta"), type="double", default=0.95, help="Adapt delta"),
  make_option(c("-d", "--max_treedepth"), type="integer", default=12, help="Max tree depth"),
  make_option(c("-s", "--seed"), type="integer", default=123, help="Random seed")
)

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

set.seed(opt$seed)

# Load simulated data
sim_data_file <- file.path(DATA_RDS_SIM_DIR, "sim_igt_mod_ev_desc-data.csv")
sim_params_file <- file.path(DATA_RDS_SIM_DIR, "sim_igt_mod_ev_desc-sub_params.csv")

sim_data <- read.csv(sim_data_file)
true_params <- read.csv(sim_params_file)

# Determine number of subjects to use
total_subjects <- length(unique(sim_data$subject_id))
n_subjects <- if(is.null(opt$n_subjects)) total_subjects else min(opt$n_subjects, total_subjects)

# Randomly select subjects if n_subjects is less than total
selected_subjects <- sample(unique(sim_data$subject_id), n_subjects)
sim_data_filtered <- sim_data %>% filter(subject_id %in% selected_subjects)
true_params_filtered <- true_params %>% filter(subject_id %in% selected_subjects)

# Prepare data for Stan
data_list <- list(
  N = n_subjects,
  T = max(sim_data_filtered$trial),
  Tsubj = as.vector(table(sim_data_filtered$subject_id)),
  choice = matrix(sim_data_filtered$choice, nrow = n_subjects, byrow = TRUE),
  shown = matrix(sim_data_filtered$deck_shown, nrow = n_subjects, byrow = TRUE),
  outcome = matrix(sim_data_filtered$net_outcome, nrow = n_subjects, byrow = TRUE)
)

# Fit the model
model_file <- file.path(MODELS_BIN_DIR, "fit", "igt_mod_group_hier_ev_fit.stan")
model <- cmdstan_model(exe_file = model_file)

fit <- model$sample(
  data = data_list,
  seed = 123,
  chains = 2,
  parallel_chains = 2,
  iter_warmup = 500,
  iter_sampling = 5000,
  adapt_delta = 0.95,
  max_treedepth = 12
)

# Save results
saveRDS(fit, file = file.path(DATA_RDS_SIM_DIR, "parameter_recovery_fit.rds"))


# Extract posterior samples
posterior_samples <- as_draws_df(fit$draws())


# Function to extract parameter summaries
extract_param_summary <- function(samples, param_name) {
  samples %>%
    select(all_of(param_name)) %>%
    summarise(
      mean = mean(!!sym(param_name)),
      sd = sd(!!sym(param_name))
    ) %>%
    mutate(parameter = param_name, value = mean) %>%
    select(-mean)
}

# Extract group-level parameter summaries
group_params <- bind_rows(
  extract_param_summary(posterior_samples, "mu_con"),
  extract_param_summary(posterior_samples, "mu_wgt_pun"),
  extract_param_summary(posterior_samples, "mu_wgt_rew"),
  extract_param_summary(posterior_samples, "mu_update")
)

# Function to extract individual-level parameter summaries
extract_individual_param_summary <- function(samples, param_name, method = "mean") {
  param_cols <- grep(paste0("^", param_name, "\\["), names(samples), value = TRUE)
  
  if(length(param_cols) == 0) {
    warning(paste("No columns found for parameter:", param_name))
    return(NULL)
  }
  
  result <- samples %>%
    select(all_of(param_cols)) %>%
    summarise(across(everything(), list(
      avg = ~ get(method)(.),
      sd = sd
    ))) %>%
    pivot_longer(everything(), names_to = "full_name", values_to = "value") %>%
    mutate(
      stat = sub(".*_(.*)$", "\\1", full_name),
      full_name = sub("_[^_]+$", "", full_name),
      subject = as.integer(sub(paste0(param_name, "\\[(\\d+)\\]"), "\\1", full_name)),
      parameter = param_name
    ) %>%
    select(-full_name)
  
  return(result)
}

# Update the call to this function
individual_params <- bind_rows(
  extract_individual_param_summary(posterior_samples, "con", method = "mean"),
  extract_individual_param_summary(posterior_samples, "wgt_pun", method = "mean"),
  extract_individual_param_summary(posterior_samples, "wgt_rew", method = "mean"),
  extract_individual_param_summary(posterior_samples, "update", method = "mean")
)

# Print summary of extracted individual parameters
cat("\nSummary of extracted individual parameters:\n")
print(table(individual_params$parameter))


# Create a mapping between subject index and subject_id
subject_mapping <- true_params_filtered %>%
  mutate(subject_index = row_number()) %>%
  select(subject_index, subject_id)

# Prepare true parameters
true_group_params <- true_params_filtered %>%
  summarise(across(c(con, wgt_pun, wgt_rew, update), mean)) %>%
  pivot_longer(everything(), names_to = "parameter", values_to = "true_value") %>%
  mutate(parameter = paste0("mu_", parameter))

true_individual_params <- true_params_filtered %>%
  pivot_longer(c(con, wgt_pun, wgt_rew, update), names_to = "parameter", values_to = "true_value")

# Combine estimated and true parameters
group_recovery_data <- left_join(group_params, true_group_params, by = "parameter")

# Join individual_params with subject_mapping
individual_recovery_data <- individual_params %>%
  filter(stat == "avg") %>%
  select(-stat) %>%
  rename(subject_index = subject) %>%
  left_join(subject_mapping, by = "subject_index")

individual_recovery_data <- individual_recovery_data %>%
  left_join(true_individual_params, by = c("subject_id", "parameter"))


# Create recovery plots
overall_recovery_plot = function(data, title) {
  ggplot(data, aes(x = true_value, y = value)) +
    geom_point(alpha = 0.5) +
    geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
    geom_smooth(method = "lm", se = FALSE, color = "blue", linetype = "solid") +
    labs(x = "True Parameter Value", y = paste("Estimated Parameter Value"), title = title) +
    theme_minimal() +
    theme(aspect.ratio = 1)  # Make plots square
}

plot_recovery <- function(data, title) {
  ggplot(data, aes(x = true_value, y = value)) +
    geom_point(alpha = 0.5) +
    facet_wrap(~parameter, scales = "free") +
    geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
    labs(x = "True Parameter Value", y = "Estimated Parameter Value", title = title) +
    theme_minimal()
}

group_plot <- overall_recovery_plot(group_recovery_data, "Group-Level Parameter Recovery")
indiv_overall_plot <- overall_recovery_plot(individual_recovery_data, "Individual-Level Parameter Recovery")

individual_plot <- plot_recovery(individual_recovery_data, "Individual-Level Parameter Recovery")

ggsave(file.path(DATA_RDS_SIM_DIR, "group_parameter_recovery_plot.png"), group_plot, width = 12, height = 8)
ggsave(file.path(DATA_RDS_SIM_DIR, "individual_overall_parameter_recovery_plot.png"), indiv_overall_plot, width = 12, height = 8)
ggsave(file.path(DATA_RDS_SIM_DIR, "individual_parameter_recovery_plot.png"), individual_plot, width = 12, height = 8)

# Calculate recovery metrics
# Function for overall metrics (group and individual)
calculate_overall_metrics <- function(data) {
  data %>%
    summarise(
      correlation = cor(true_value, value),
      rmse = sqrt(mean((true_value - value)^2)),
      mae = mean(abs(true_value - value)),
      bias = mean(value - true_value),
      relative_bias = mean((value - true_value) / true_value),
      )
}

# Function for overall metrics (group and individual)
calculate_param_metrics <- function(data) {
  data %>%
    group_by(parameter) %>%
    summarise(
      correlation = cor(true_value, value),
      rmse = sqrt(mean((true_value - value)^2)),
      mae = mean(abs(true_value - value)),
      bias = mean(value - true_value),
      relative_bias = mean((value - true_value) / true_value),
    )
}

# Function for within-individual metrics
calculate_within_individual_metrics <- function(data) {
  data %>%
    group_by(subject_id) %>%
    summarise(
      correlation = cor(true_value, value),
      rmse = sqrt(mean((true_value - value)^2)),
      mae = mean(abs(true_value - value)),
      bias = mean(value - true_value),
      relative_bias = mean((value - true_value) / true_value)
    )}

# Calculate metrics
group_overall_metrics <- calculate_overall_metrics(group_recovery_data)
group_param_metrics <- calculate_param_metrics(group_recovery_data)

individual_overall_metrics <- calculate_overall_metrics(individual_recovery_data)
individual_param_metrics <- calculate_param_metrics(individual_recovery_data)
individual_within_metrics <- calculate_within_individual_metrics(individual_recovery_data)

# Print recovery metrics
cat("Group-level recovery metrics:\n")
print(group_overall_metrics)
cat("\nIndividual-level recovery metrics:\n")
print(individual_overall_metrics)

# Save results
saveRDS(list(
  fit = fit,
  group_recovery_data = group_recovery_data,
  individual_recovery_data = individual_recovery_data,
  group_overall_metrics = group_overall_metrics,
  group_param_metrics = group_param_metrics,
  individual_overall_metrics = individual_overall_metrics,
  individual_param_metrics = individual_param_metrics,
  individual_within_metrics = individual_within_metrics,
  n_subjects = n_subjects
), file = file.path(DATA_RDS_SIM_DIR, "parameter_recovery_results.rds"))

cat("Parameter recovery analysis completed for", n_subjects, "subjects.\n")
