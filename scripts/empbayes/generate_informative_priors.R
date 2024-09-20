#!/usr/bin/env Rscript

# Load required libraries
suppressPackageStartupMessages({
  library(optparse)
  library(here)
  library(posterior)
  library(dplyr)
  library(tidyr)
  library(truncnorm)
  library(jsonlite)
})

# Source helper functions
source(here("scripts", "helper_functions_cmdSR.R"))

# Define command line options
option_list = list(
  make_option(c("-m", "--model"), type="character", default=NULL, help="Model name"),
  make_option(c("-k", "--task"), type="character", default=NULL, help="Task name"),
  make_option(c("-g", "--group"), type="character", default=NULL, help="Group type (sing, group, group_hier)"),
  make_option(c("--seed"), type="integer", default=29518, help="Set seed"),
  make_option(c("--debug"), action="store_true", default=FALSE, help="Run in debug mode"),
  make_option(c("--verbose"), action="store_true", default=FALSE, help="Run in verbose mode"),
  make_option(c("--dry_run"), action="store_true", default=FALSE, help="Perform a dry run without processing data or saving files")
)

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

# Check for required options
if (is.null(opt$model) || is.null(opt$task) || is.null(opt$group)) {
  stop("Please specify a model, task, and group type using the -m, -k, and -g options.")
}

# Set up directories
PROJ_DIR <- here::here()
DATA_DIR <- file.path(PROJ_DIR, "Data")
DATA_RDS_DIR <- file.path(DATA_DIR, "rds")
DATA_RDS_eB_DIR <- file.path(DATA_RDS_DIR, "empbayes")
DATA_TXT_DIR <- file.path(DATA_DIR, "txt")
DATA_TXT_eB_DIR <- file.path(DATA_TXT_DIR, "empbayes")

# Ensure directories exist
dir.create(DATA_TXT_eB_DIR, recursive = TRUE, showWarnings = FALSE)

# Set random seed for reproducibility
set.seed(opt$seed)

# Construct full model name
full_model_name <- paste(opt$task, opt$group, opt$model, sep="_")

# Functions to generate informative priors
fit_truncated_normal <- function(param_draws, init_lower = NULL, init_upper = NULL, verbose = FALSE) {
  # Initial estimates
  init_sd <- sd(param_draws)**(1/runif(1, min = 0.5, max = 2))
  init_mean <- mean(param_draws) + runif(1, min = -2, max = 2)*init_sd
  init_lower <- runif(1, min = min(param_draws), max = quantile(param_draws, .1))
  init_upper <- runif(1, min = quantile(param_draws, .9), max = max(param_draws))
  
  # Optimization function with optional verbose output
  nll <- function(par) {
    mean <- par[1]
    sd <- par[2]
    lower <- par[3]
    upper <- par[4]
    
    if (lower >= upper || sd <= 0) return(Inf)  # Invalid bounds
    
    # Compute truncated normal density
    density <- dtruncnorm(param_draws, a = lower, b = upper, mean = mean, sd = sd)
    # Avoid log(0) by replacing zeros with a small value
    log_density <- log(pmax(density, .Machine$double.eps))
    
    return(-sum(log_density))
  }
  
  init_params = c(init_mean, init_sd, init_lower, init_upper)
  cat("\nInit values (mean, sd, lower, upper):", init_params, "\n")
  
  # Optimize SANN
  fit_sann <- optim(init_params, nll, method = "SANN", 
                    control = list(trace = as.integer(verbose),
                                   temp = 150,  # Initial temperature
                                   tmax = 500,   # Number of function evaluations at initial temperature
                                   maxit = 5000))  # Increase maximum iterations
  
  print(fit_sann$par)
  
  cat("\nFollowup fit with BFGS...\n")
  # Run a final optimization with BFGS starting from the SANN result
  fit_final <- tryCatch({
    optim(fit_sann$par, nll, method = "BFGS",
                       control = list(trace = as.integer(verbose),
                                      maxit = 2000,
                                      reltol = 1e-10,
                                      abstol = 1e-10))
  }, error = function(e) {
    cat("Error with BFGS followup:", conditionMessage(e), "\n")
    cat("Using SANN results\n")
    fit_sann
  })
  
  
  # Convergence Check
  if (fit_final$convergence > 0){
    warning("\nFinal results did not converge: ", fit_final$convergence, "\n")
  }
  
  print(fit_final$par)
  
  return(list(mean = fit_final$par[1], 
              sd = fit_final$par[2], 
              lower = fit_final$par[3], 
              upper = fit_final$par[4])
  )
}

generate_informative_priors <- function(fit, debug = FALSE, verbose = FALSE) {
  # Extract individual-level parameters
  draws <- fit$draws
  params <- fit$model_params
  
  if (debug) {
    # Limit to a subset of parameters for debugging
    params <- params[1:min(length(params), 5)]
  }
  
  priors_list <- list()
  
  # Extract raw variable names
  raw_sub_all_params = fit$all_params[grepl("_pr\\[", fit$all_params)]
  raw_sub_all_params = raw_sub_all_params[!grepl("mu", raw_sub_all_params)]
  
  for (param in params) {
    raw_sub_params = raw_sub_all_params[grepl(param, raw_sub_all_params)]
    param_draws <- as.vector(draws[,, raw_sub_params])
    
    # Fit truncated normal distribution
    fit_result <- tryCatch({
      cat("\nFitting truncnorm to", param, "\n")
      fit_truncated_normal(param_draws, verbose = verbose)
    }, error = function(e) {
      cat("Error fitting truncated normal for", param, ":", conditionMessage(e), "\n")
      cat("Using mean, sd, min/max...\n\n")
      # If fitting fails, use simple mean and sd
      list(mean = mean(param_draws), sd = sd(param_draws),
           lower = min(param_draws), upper = max(param_draws))
    })
    
    priors_list[[param]] <- c(
      pr_mu = fit_result$mean,
      pr_sigma = fit_result$sd,
      lower_bound = fit_result$lower,
      upper_bound = fit_result$upper
    )
  }
  
  return(priors_list)
}

# Files
hier_fit_file <- file.path(DATA_RDS_eB_DIR, paste0(full_model_name, "_desc-emp_hier_output.rds"))
priors_file <- file.path(DATA_TXT_eB_DIR, paste0(full_model_name, "_informative_priors.txt"))


if (opt$dry_run) {
  cat("Dry run: Would attempt to load initial hierarchical fit from:", hier_fit_file, "\n")
  cat("Dry run: Generating dummy data for priors...\n")
  
  # Create dummy data for dry run
  dummy_draws <- matrix(rnorm(1000), ncol = 10)
  colnames(dummy_draws) <- paste0("param", 1:10)
  dummy_fit <- list(
    draws = dummy_draws,
    model_params = paste0("param", 1:5),
    all_params = c(paste0("param", 1:10, "_pr[1]"), paste0("param", 1:10, "_pr[2]"))
  )
  
  cat("Dry run: Would generate informative priors...\n")
  cat("Dry run: Would save informative priors to:", priors_file, "\n")
  cat("Dry run: Would also save priors as RDS to:", 
      file.path(DATA_RDS_eB_DIR, paste0(full_model_name, "_informative_priors.rds")), "\n")
} else {
  
} else {
  # Load the initial hierarchical fit
  if (!file.exists(hier_fit_file)) {
    stop("Initial hierarchical fit file not found: ", hier_fit_file)
  }
  hier_fit <- readRDS(hier_fit_file)
  
  # Generate informative priors
  cat("Generating informative priors...\n")
  priors <- generate_informative_priors(hier_fit, debug = opt$debug, verbose = opt$verbose)
  
  # Save priors to txt file
  priors_file <- file.path(DATA_TXT_eB_DIR, paste0(full_model_name, "_informative_priors.txt"))
  priors_df <- do.call(rbind, lapply(names(priors), function(param) {
    data.frame(parameter = param, t(priors[[param]]))
  }))
  
  write.table(priors_df, file = priors_file, row.names = FALSE, sep = ",", quote = FALSE)
  cat("Informative priors saved to:", priors_file, "\n")
  
  # Also save as RDS for easier R loading
  saveRDS(priors, file = file.path(DATA_RDS_eB_DIR, paste0(full_model_name, "_informative_priors.rds")))
  
  cat("Informative priors generation completed.\n")
}


cat("Informative priors generation ", ifelse(opt$dry_run, "dry run ", ""), "completed.\n")
