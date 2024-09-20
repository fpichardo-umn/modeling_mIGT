suppressPackageStartupMessages({
  library(R6)
  library(data.table)
  library(optparse)
  library(here)
})

# Define parameter ranges
PARAM_RANGES <- list(
  con = list(range = c(-2, 2), min = -2, max = 2, low = c(-2, -1), medium = c(-1, 1), high = c(1, 2)),
  wgt_pun = list(range = c(0, 1), min = 0, max = 1, low = c(0, 0.3), medium = c(0.3, 0.7), high = c(0.7, 1)),
  wgt_rew = list(range = c(0, 1), min = 0, max = 1, low = c(0, 0.3), medium = c(0.3, 0.7), high = c(0.7, 1)),
  update = list(range = c(0, 1), min = 0, max = 1, low = c(0, 0.3), medium = c(0.3, 0.7), high = c(0.7, 1))
)

# Function to generate parameter values
## Full Parameter Space Explorations (FPSE) Methods
### Grid-based FPSE (gbFPSE)

### Random Sampling FPSE (rsFPSE)

### Stratified Sampling FPSE (ssFPSE)
stratified_fpse_sampling <- function(n_sets) {
  param_sets <- vector("list", n_sets + 3)
  
  # Ensure we have at least one set with all low, all medium, and all high values
  param_sets[[1]] <- lapply(PARAM_RANGES, function(range) round(runif(1, range$low[1], range$low[2]), 2))
  param_sets[[2]] <- lapply(PARAM_RANGES, function(range) round(runif(1, range$medium[1], range$medium[2]), 2))
  param_sets[[3]] <- lapply(PARAM_RANGES, function(range) round(runif(1, range$high[1], range$high[2]), 2))
  
  # Create sets with interesting combinations
  for (i in 4:(length(param_sets) + 1)) {
    param_set <- list()
    for (param in names(PARAM_RANGES)) {
      range_choice <- sample(c("low", "medium", "high"), 1)
      range <- PARAM_RANGES[[param]][[range_choice]]
      param_set[[param]] <- round(runif(1, range[1], range[2]),2)
    }
    param_sets[[i]] <- param_set
  }
  
  return(param_sets)
}

## Empirical Parameter Space Explorations (EPSE) Methods
### Subject Posterior Sampling (SPS) EPSE
#### Median-based SPS (mbSPSepse)
mb_sps_epse_sampling <- function(n_sets, model_fit_obj, params) {
  n_subjects <- length(model_fit_obj$sid)
  param_sets <- vector("list", n_sets)
  
  for (i in 1:n_sets) {
    subject <- sample(1:n_subjects, 1)
    con <- median(extract_posterior_samples(model_fit_obj, "con", subject))
    wgt_pun <- median(extract_posterior_samples(model_fit_obj, "wgt_pun", subject))
    wgt_rew <- median(extract_posterior_samples(model_fit_obj, "wgt_rew", subject))
    update <- median(extract_posterior_samples(model_fit_obj, "update", subject))
    
    param_sets[[i]] <- list(con = con, wgt_pun = wgt_pun, wgt_rew = wgt_rew, update = update)
  }
  
  return(param_sets)
}

#### Simulation-based SPS (sbSPSepse)
sb_sps_epse_sampling <- function(n_sets, model_fit_obj, params) {
  n_subjects <- length(model_fit_obj$sid)
  param_sets <- vector("list", n_sets)
  
  for (i in 1:n_sets) {
    subject <- sample(1:n_subjects, 1)
    con <- sample(extract_posterior_samples(model_fit_obj, "con", subject), 1)
    wgt_pun <- sample(extract_posterior_samples(model_fit_obj, "wgt_pun", subject), 1)
    wgt_rew <- sample(extract_posterior_samples(model_fit_obj, "wgt_rew", subject), 1)
    update <- sample(extract_posterior_samples(model_fit_obj, "update", subject), 1)
    
    param_sets[[i]] <- list(con = con, wgt_pun = wgt_pun, wgt_rew = wgt_rew, update = update)
  }
  
  return(param_sets)
}

### Tuple SPS (tSPSepse)
sub_tuple_sampling <- function(model_fit_obj, params, n_sets = 1,
                               min_iterations = 10, min_percentile = 50, 
                               max_recursion = 3, param_reduc = 1) {
  n_subjects <- length(model_fit_obj$sid)
  param_sets <- vector("list", n_sets)
  
  for (i in 1:n_sets) {
    subject_index <- sample(1:n_subjects, 1)
    result <- tuple_sampling(model_fit_obj, params, subject_index = subject_index, n_sets = 1, 
                             min_iterations = min_iterations, min_percentile = min_percentile, 
                             max_recursion = max_recursion, param_reduc = param_reduc)
    
    # Explicitly extract each parameter value
    param_sets[[i]] <- setNames(
      lapply(params, function(param) result$param_sets[[1]][[param]]),
      params
    )
  }
  
  return(param_sets)
}

### Hierarchical Posterior Simulation EPSE (hpsEPSE)
hps_epse_sampling <- function(n_sets, model_fit_obj, params) {
  param_sets <- vector("list", n_sets)
  
  mu_pr <- sapply(params, function(param) {
    median(sample(extract_posterior_samples(model_fit_obj, paste0("mu_pr[", which(names(PARAM_RANGES) == param), "]")), 10))
  })
  
  sigma <- sapply(1:4, function(j) {
    median(sample(extract_posterior_samples(model_fit_obj, paste0("sigma[", j, "]")), 10))
  })
  
  cat("Means for", params, "\n")
  cat(mu_pr, "\n", "\n")
  cat("Sigmas for", params, "\n")
  cat(sigma, "\n")
  
  for (i in 1:n_sets){
    con_pr <- rnorm(1, mu_pr[1], sigma[1])
    wgt_pun_pr <- rnorm(1, mu_pr[2], sigma[2])
    wgt_rew_pr <- rnorm(1, mu_pr[3], sigma[3])
    update_pr <- rnorm(1, mu_pr[4], sigma[4])
    
    param_sets[[i]] <- transform_parameters(con_pr, wgt_pun_pr, wgt_rew_pr, update_pr, mu_pr, sigma)
  }
  return(param_sets)
}

### Tuple Hierarchical Posterior Simulation EPSE (thpsEPSE)
group_tuple_sampling <- function(model_fit_obj, params, n_sets = 1,
                                 min_iterations = 10, min_percentile = 50, 
                                 max_recursion = 3, param_reduc = 1){
  
  result <- tuple_sampling(model_fit_obj, params, n_sets = 1, 
                           min_iterations = min_iterations, min_percentile = min_percentile, 
                           max_recursion = max_recursion, param_reduc = param_reduc)
  
  group_params <- result$param_sets[[1]]
  mu_pr <- sapply(params, function(param) group_params[[paste0("mu_", param)]])
  sigma <- sapply(params, function(param) group_params[[paste0("sigma_", param)]])
  
  cat("Means for", params, "\n")
  cat(mu_pr, "\n", "\n")
  cat("Sigmas for", params, "\n")
  cat(sigma, "\n")
  
  param_sets <- vector("list", n_sets)
  for (i in 1:n_sets) {
    param_pr <- sapply(1:length(params), function(j) rnorm(1, mu_pr[j], sigma[j]))
    param_sets[[i]] <- transform_parameters(param_pr[1], param_pr[2], param_pr[3], param_pr[4], mu_pr, sigma)
  }
  return(param_sets)
}

# Tuple Sampling
tuple_sampling <- function(model_fit_obj, params, subject_index = NULL, n_sets = 1, 
                           initial_percentile = 90, min_iterations = 10, min_percentile = 50, 
                           max_recursion = 3, param_reduc = 1) {
  
  is_subject_level <- !is.null(subject_index)
  
  # Extract posterior samples
  if (is_subject_level) {
    posterior_samples <- lapply(params, function(param) {
      extract_posterior_samples(model_fit_obj, param, subject_index)
    })
    names(posterior_samples) <- params
  } else {
    mu_pr_samples <- lapply(params, function(param) {
      extract_posterior_samples(model_fit_obj, paste0("mu_pr[", which(names(PARAM_RANGES) == param), "]"))
    })
    sigma_samples <- lapply(seq_along(params), function(j) {
      extract_posterior_samples(model_fit_obj, paste0("sigma[", j, "]"))
    })
    posterior_samples <- c(mu_pr_samples, sigma_samples)
    names(posterior_samples) <- c(paste0("mu_", params), paste0("sigma_", params))
  }
  
  # Convert to data frame
  posterior_df <- as.data.frame(posterior_samples)
  
  # Find high-probability iterations
  high_prob_result <- find_high_prob_iterations(posterior_df, names(posterior_samples), min_iterations, 
                                                initial_percentile, min_percentile, 
                                                max_recursion, param_reduc)
  
  high_prob_iterations <- high_prob_result$iterations
  
  # Sample n_sets from high probability iterations
  sampled_iterations <- sample(high_prob_iterations, n_sets, replace = TRUE)
  
  # Extract parameter values for sampled iterations
  param_sets <- lapply(sampled_iterations, function(i) {
    as.list(posterior_df[i,])
  })
  
  return(list(param_sets = param_sets, 
              final_percentile = high_prob_result$final_percentile, 
              params_used = high_prob_result$params_used))
}


# Extract posterior samples for a specific parameter
extract_posterior_samples <- function(model_fit_obj, param_name, subject_index = NULL) {
  if (is.null(subject_index)) {
    # Group-level parameter
    samples <- model_fit_obj$draws[, , param_name]
  } else {
    # Individual-level parameter
    samples <- model_fit_obj$draws[, , paste0(param_name, "[", subject_index, "]")]
  }
  return(as.vector(samples))
}

# Get iteration tuples
find_high_prob_iterations <- function(posterior_df, params, min_iterations = 10, 
                                      initial_percentile = 90, min_percentile = 50, 
                                      max_recursion = 3, param_reduc = 1) {
  
  find_iterations <- function(percentile) {
    thresholds <- sapply(posterior_df[params], function(x) quantile(x, probs = percentile/100))
    high_prob <- apply(posterior_df[params], 1, function(row) all(row >= thresholds))
    which(high_prob)
  }
  
  current_percentile <- initial_percentile
  high_prob_iterations <- find_iterations(current_percentile)
  
  while (length(high_prob_iterations) < min_iterations && current_percentile > min_percentile) {
    current_percentile <- current_percentile - 5
    high_prob_iterations <- find_iterations(current_percentile)
  }
  
  if (length(high_prob_iterations) >= min_iterations || max_recursion == 0) {
    return(list(iterations = high_prob_iterations, final_percentile = current_percentile, params_used = params))
  } else {
    if (length(params) > 1) {
      # Remove parameters and recurse
      num_to_drop = min(length(params), param_reduc)
      remaining_params <- params[-num_to_drop]  # Remove the first parameter by default
      remaining_iterations <- min_iterations - length(high_prob_iterations)
      if (length(high_prob_iterations) > 0) {
        posterior_df = posterior_df[-high_prob_iterations, ]
      }
      
      recursion_result <- find_high_prob_iterations(
        posterior_df, remaining_params, 
        min_iterations = remaining_iterations,
        initial_percentile = initial_percentile,
        min_percentile = min_percentile,
        max_recursion = max_recursion - 1,
        param_reduc = param_reduc
      )
      
      # Combine the results
      all_iterations <- unique(c(high_prob_iterations, recursion_result$iterations))
      
      return(list(iterations = all_iterations, 
                  final_percentile = min(current_percentile, recursion_result$final_percentile),
                  params_used = recursion_result$params_used))
    } else {
      # If only one parameter left, return the top min_iterations
      warning("Using top available iterations for the last parameter.")
      top_iterations <- order(posterior_df[[params]], decreasing = TRUE)[1:min_iterations]
      return(list(iterations = top_iterations, final_percentile = NA, params_used = params))
    }
  }
}


# Apply parameter transforms
transform_parameters <- function(con_pr, wgt_pun_pr, wgt_rew_pr, update_pr, mu_pr, sigma) {
  con <- plogis(mu_pr[1] + sigma[1] * con_pr) * 4 - 2
  wgt_pun <- plogis(mu_pr[2] + sigma[2] * wgt_pun_pr)
  wgt_rew <- plogis(mu_pr[3] + sigma[3] * wgt_rew_pr)
  update <- plogis(mu_pr[4] + sigma[4] * update_pr)
  return(list(con = con, wgt_pun = wgt_pun, wgt_rew = wgt_rew, update = update))
}


### Load data
load_relevant_data = function(model_str){
  #model_str = "igt_mod_group_hier_ev"
  model.out.path = file.path(DATA_RDS_DIR, 'fit', paste0(model_str, "_fit_output.rds"))
  
  readRDS(file = model.out.path)
}

# Parse command line arguments
option_list <- list(
  make_option(c("-n", "--n_subjects"), type="integer", default=100, help="Number of subjects (def: 100)"),
  make_option(c("-o", "--output_dir"), type="character", default=NULL, help="Output directory for simulated data (def: Data/sim/params"),
  make_option(c("-m", "--model"), type="character", default=NULL, help="Model name"),
  make_option(c("-k", "--task"), type="character", default=NULL, help="Task name"),
  make_option(c("-g", "--group"), type="character", default=NULL, help="Group type (sing, group, group_hier)"),
  make_option(c("-s", "--seed"), type="integer", default=775625534, help="Seed for reproducibility (def: 775625534)")
)

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

# Set up directories
PROJ_DIR <- here::here()
DATA_DIR <- file.path(PROJ_DIR, "Data")
DATA_SIM_DIR <- file.path(DATA_DIR, "sim")
DATA_SIM_PARAMS_DIR <- file.path(DATA_SIM_DIR, "params")

if (is.null(opt$output_dir)) {
  opt$output_dir <- DATA_SIM_PARAMS_DIR
}

if (!dir.exists(opt$output_dir)) {
  dir.create(opt$output_dir, recursive = TRUE)
}

# Main execution
set.seed(opt$seed)

# Define the model
if (is.null(opt$model) || is.null(opt$task) || is.null(opt$group)) {
  stop("Please specify a model, task, and group type using the -m, -k, and -g options.")
}

model_name <- opt$model
task <- opt$task
group_type <- opt$group
full_model_name = paste(task, group_type, model_name, sep="_")

cat("Genearting parameter estimates for", opt$n_subjects, "simulated subjects.", "\n")

# Function to generate parameter values
## Full Parameter Space Explorations (FPSE) Methods
### Stratified Sampling FPSE (ssFPSE)
# Generate params
sim_params <- stratified_fpse_sampling(opt$n_subjects)

# Save params
filename = file.path(opt$output_dir, paste0(full_model_name, "_desc-sim_params_ssFPSE.rds"))
saveRDS(sim_params, filename)
cat("Stratified Sampling FPSE simulated parameters saved to", filename, "\n")

## Empirical Parameter Space Explorations (EPSE) Methods
### Subject Posterior Sampling (SPS) EPSE
#### Median-based SPS (mbSPSepse)
# Load data
model_fit_obj = load_relevant_data(full_model_name)

# Generate params
sim_params <- mb_sps_epse_sampling(opt$n_subjects, model_fit_obj, c("con", "wgt_pun", "wgt_rew", "update"))

# Save params
filename = file.path(opt$output_dir, paste0(full_model_name, "_desc-sim_params_mbSPSepse.rds"))
saveRDS(sim_params, filename)
cat("Median-based Subject Posterior Sampling EPSE simulated parameters saved to", filename, "\n")

#### Simulation-based SPS (sbSPSepse)
# Generate params
sim_params <- sb_sps_epse_sampling(opt$n_subjects, model_fit_obj, c("con", "wgt_pun", "wgt_rew", "update"))

# Save params
filename = file.path(opt$output_dir, paste0(full_model_name, "_desc-sim_params_sbSPSepse.rds"))
saveRDS(sim_params, filename)
cat("Simulation-based Subject Posterior Sampling EPSE simulated parameters saved to", filename, "\n")

### Tuple SPS (tSPSepse)
# Generate params
sim_params <- sub_tuple_sampling(model_fit_obj, c("con", "wgt_pun", "wgt_rew", "update"), n_sets = opt$n_subjects,
                                 min_iterations = 10, min_percentile = 50, 
                                 max_recursion = 3, param_reduc = 1)

# Save params
filename = file.path(opt$output_dir, paste0(full_model_name, "_desc-sim_params_tSPSepse.rds"))
saveRDS(sim_params, filename)
cat("Tuple Subject Posterior Sampling EPSE simulated parameters saved to", filename, "\n")

### Hierarchical Posterior Simulation EPSE (hpsEPSE)
#### Simulation based
# Generate params
sim_params <- hps_epse_sampling(opt$n_subjects, model_fit_obj, c("con", "wgt_pun", "wgt_rew", "update"))

# Save params
filename = file.path(opt$output_dir, paste0(full_model_name, "_desc-sim_params_hpsEPSE.rds"))
saveRDS(sim_params, filename)
cat("Hierarchical Posterior Sampling EPSE simulated parameters saved to", filename, "\n")

### Tuple Hierarchical Posterior Simulation EPSE (thpsEPSE)
# Generate params
sim_params <- group_tuple_sampling(model_fit_obj, c("con", "wgt_pun", "wgt_rew", "update"), n_sets = opt$n_subjects,
                                   min_iterations = 10, min_percentile = 50, 
                                   max_recursion = 3, param_reduc = 1)

# Save params
filename = file.path(opt$output_dir, paste0(full_model_name, "_desc-sim_params_thpsEPSE.rds"))
saveRDS(sim_params, filename)
cat("Tuple Hierarchical Posterior Sampling EPSE simulated parameters saved to", filename, "\n")


cat("All parameters generated. Values saved to", opt$output_dir, "\n")
