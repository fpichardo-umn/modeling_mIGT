# Load required libraries
suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(bayesplot)
  library(posterior)
  library(gridExtra)
})

## Functions
# Fit functions
# Function to fit and save a model
fit_and_save_model <- function(task, group_type, model_name, model_type, data_list, n_subs, n_trials, n_warmup, n_iter, n_chains, adapt_delta, max_treedepth, model_params, dry_run = FALSE, checkpoint_interval = 1000, output_dir = NULL, emp_bayes = FALSE) {
  model_str <- paste(task, group_type, model_name, sep="_")
  model_path <- file.path(file.path(MODELS_BIN_DIR, model_type), paste0(model_str, "_", model_type, ".stan"))
  stanmodel_arg <- cmdstan_model(exe_file = model_path)
  
  # Determine output directory
  if (is.null(output_dir)) {
    if (emp_bayes) {
      output_dir <- DATA_RDS_eB_DIR
    } else {
      output_dir <- file.path(DATA_DIR, "rds")
    }
  }
  
  output_file <- file.path(output_dir, paste0(model_str, "_", model_type, if(emp_bayes) "_desc-emp_hier" else "", "_output.rds"))
  checkpoint_file <- paste0(tools::file_path_sans_ext(output_file), "_checkpoint.rds")
  
  if (!file.exists(dirname(output_file))) {
    stop("Output folder does not exist. Expected path: ", output_file)
  }
  
  if (dry_run) {
    cat("Dry run for model:", model_str, "\n")
    cat("Data list:\n")
    print(str(data_list))
    cat("Parameters:\n")
    print(str(model_params))
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
  
  # Check if there's a checkpoint to resume from
  if (file.exists(checkpoint_file)) {
    cat("Resuming from checkpoint\n")
    checkpoint <- readRDS(checkpoint_file)
    current_iter <- checkpoint$current_iter
    accumulated_samples <- checkpoint$accumulated_samples
    accumulated_diagnostics <- checkpoint$accumulated_diagnostics
    step_size <- checkpoint$step_size
    inv_metric <- checkpoint$inv_metric
    warmup_done <- checkpoint$warmup_done
  } else {
    current_iter <- 0
    accumulated_samples <- list()
    accumulated_diagnostics <- list()
    step_size <- NULL
    inv_metric <- NULL
    warmup_done <- FALSE
  }
  
  total_iter <- n_warmup + n_iter
  
  while (current_iter < total_iter) {
    remaining_iter <- min(checkpoint_interval, total_iter - current_iter)
    
    if (!warmup_done) {
      # Initial run or warmup not complete
      iter_to_run <- max(n_warmup + 1, remaining_iter)  # Ensure iter > warmup
      fit <- stanmodel_arg$sample(
        data = data_list,
        iter_sampling = iter_to_run - n_warmup,
        iter_warmup = n_warmup,
        chains = n_chains,
        parallel_chains = n_chains,
        adapt_delta = adapt_delta,
        max_treedepth = max_treedepth,
        refresh = ceiling(iter_to_run/10)
      )
      
      # Extract step size and inverse metric after warmup
      step_size <- fit$metadata()$step_size_adaptation
      inv_metric <- fit$inv_metric()
      warmup_done <- TRUE
      
      new_samples <- fit$draws()
      current_iter <- iter_to_run
    } else {
      # Continue sampling post-warmup
      last_draws <- accumulated_samples[[length(accumulated_samples)]]
      last_draws <- last_draws[dim(last_draws)[1],,]  # Get the last iteration for all chains
      init_list <- lapply(1:n_chains, function(chain_idx) create_init_list(last_draws, chain_idx))
      
      fit <- stanmodel_arg$sample(
        data = data_list,
        iter_sampling = remaining_iter,
        iter_warmup = 0,
        chains = n_chains,
        parallel_chains = n_chains,
        adapt_delta = adapt_delta,
        metric='dense_e',
        max_treedepth = max_treedepth,
        init = init_list,
        inv_metric = inv_metric,
        adapt_engaged = FALSE,
        step_size = step_size,
        refresh = ceiling(remaining_iter/10)
      )
      
      new_samples <- fit$draws()
      current_iter <- current_iter + dim(new_samples)[1]
    }
    
    # Accumulate new samples
    accumulated_samples <- c(accumulated_samples, list(new_samples))
    
    # Extract and accumulate diagnostics
    new_diagnostics <- fit$sampler_diagnostics()
    accumulated_diagnostics <- c(accumulated_diagnostics, list(new_diagnostics))
    
    # Save checkpoint
    saveRDS(list(current_iter = current_iter, 
                 accumulated_samples = accumulated_samples,
                 accumulated_diagnostics = accumulated_diagnostics,
                 step_size = step_size, 
                 inv_metric = inv_metric,
                 warmup_done = warmup_done), 
            file = checkpoint_file)
    cat("Checkpoint saved at iteration", current_iter, "\n")
  }
  
  # Combine all accumulated samples into a single draws object
  all_samples <- do.call(posterior::bind_draws, c(accumulated_samples, along = "iteration"))
  
  # Combine all accumulated diagnostics
  all_diagnostics <- do.call(posterior::bind_draws, c(accumulated_diagnostics, along = "iteration"))
  
  # Calculate diagnostic summaries
  num_divergent <- colSums(all_diagnostics[, , "divergent__"])
  num_max_treedepth <- colSums(all_diagnostics[, , "treedepth__"] == max_treedepth)
  ebfmi <- sapply(1:n_chains, function(chain) {
    energy <- all_diagnostics[, chain, "energy__"]
    sum(diff(energy)^2) / (length(energy) - 1) / var(energy)
  })
  
  # Calculate summary statistics based on all accumulated samples
  summary_stats <- posterior::summarize_draws(all_samples)
  
  # Apply the function to each parameter
  param_names <- dimnames(all_samples)[[3]]
  diagnostics <- lapply(param_names, function(param) {
    param_draws <- all_samples[, , param]
    calculate_param_diagnostics(param_draws)
  })
  
  # Combine the results into a data frame
  diagnostics_df <- do.call(rbind, diagnostics)
  rownames(diagnostics_df) <- param_names
  
  # Create a fit object with all necessary information
  fit = list(
    draws = all_samples,
    sampler_diagnostics = all_diagnostics,
    n_warmup = n_warmup,
    n_iter = n_iter,
    n_params = dim(all_samples)[3],
    n_chains = n_chains,
    adapt_delta = adapt_delta,
    max_treedepth = max_treedepth,
    tss = dim(all_samples)[1],
    all_params = param_names,
    list_params = unique(gsub("\\[.*?\\]", "", param_names)),
    summary_stats = summary_stats,
    diagnostic_summary = list(
      num_divergent = num_divergent,
      num_max_treedepth = num_max_treedepth,
      ebfmi = ebfmi
    ),
    diagnostics = diagnostics_df,
    cmdstan_version = cmdstan_version(),
    model_name = model_str,
    n_runs = length(accumulated_samples)  # Number of sampling runs
  )
  
  # Print diagnostic warnings
  total_iterations <- n_chains * (n_iter - n_warmup)
  total_divergent <- sum(num_divergent)
  total_max_treedepth <- sum(num_max_treedepth)
  
  if (total_divergent > 0) {
    cat(sprintf("\nWarning: %d of %d (%.1f%%) transitions ended with a divergence.\n", 
                total_divergent, total_iterations, 100 * total_divergent / total_iterations))
    cat("See https://mc-stan.org/misc/warnings for details.\n")
  }
  
  if (total_max_treedepth > 0) {
    cat(sprintf("\nWarning: %d of %d (%.1f%%) transitions hit the maximum treedepth limit of %d.\n", 
                total_max_treedepth, total_iterations, 100 * total_max_treedepth / total_iterations, max_treedepth))
    cat("See https://mc-stan.org/misc/warnings for details.\n")
  }
  
  cat("Extracting parameters\n")
  fit$params <- extract_params(fit$all_params, n_subs, main_params_vec = model_params)
  fit$params <- unname(fit$params)
  
  cat("Saving fitted model to:", output_file, "\n")
  saveRDS(fit, file = output_file)
  
  # Remove checkpoint file after successful completion
  if (file.exists(checkpoint_file) && file.exists(output_file)) {
    file.remove(checkpoint_file)
  }
  
  return(fit)
}

create_init_list = function(last_draws, chain_idx) {
  
  # Extract the last draw for this chain
  last_draw_chain <- last_draws[1, chain_idx, ]
  
  # Get all the parameter names
  param_names <- dimnames(last_draw_chain)[[3]]
  
  # Separate parameters with and without square brackets
  grouped_params <- gsub("\\[.*\\]", "", param_names)  # Remove everything in square brackets
  unique_grouped_params <- unique(grouped_params)      # Get unique base parameter names
  
  # Create the list dynamically
  init_vals <- list()
  
  # Loop through each unique base parameter name
  for (param in unique_grouped_params) {
    # Find all elements corresponding to this parameter
    matching_indices <- grep(paste0("^", param), param_names)
    
    if (length(matching_indices) > 1) {
      # If there are multiple entries, it's an array/vector: group them
      init_vals[[param]] <- last_draw_chain[matching_indices]
    } else {
      # If there's only one entry, it's a scalar
      init_vals[[param]] <- last_draw_chain[matching_indices]
    }
  }
  
  return(init_vals)
}

# Function to create balanced strata
create_balanced_strata <- function(data, min_size = 10) {
  data %>%
    dplyr::group_by(grpid) %>%
    dplyr::mutate(strata = dplyr::case_when(
      dplyr::n() >= min_size ~ as.character(grpid),
      TRUE ~ "Other"
    )) %>%
    dplyr::group_by(strata, grade10) %>%
    dplyr::mutate(strata = paste(strata, grade10, sep = "")) %>%
    dplyr::group_by(strata, male) %>%
    dplyr::mutate(strata = paste(strata, male, sep = "")) %>%
    dplyr::group_by(strata, race_ethnicity) %>%
    dplyr::mutate(strata = paste(strata, race_ethnicity, sep = "")) %>%
    dplyr::group_by(strata, parent_edu4) %>%
    dplyr::mutate(strata = paste(strata, parent_edu4, sep = "")) %>%
    dplyr::ungroup()
}

# Function to split data maintaining strata proportions
stratified_split <- function(data, props) {
  cumulative_prop <- cumsum(props)
  
  data %>%
    dplyr::group_by(strata) %>%
    dplyr::mutate(
      random_number = runif(dplyr::n()),
      split = cut(random_number, 
                  breaks = c(0, cumulative_prop), 
                  labels = seq_along(props),
                  include.lowest = TRUE)
    ) %>%
    dplyr::select(-random_number) %>%
    dplyr::ungroup()
}

# Diagnostics
## Empirical Bayes diagnostics check
check_model_diagnostics <- function(fit, rhat_threshold = 0.93, ess_threshold = 0.75, mcse_threshold = 0.1) {
  # Check divergences
  sampler_diagnostics <- fit$sampler_diagnostics
  divergences <- sum(sampler_diagnostics[,,"divergent__"])
  n_iter <- dim(sampler_diagnostics)[1]
  n_chains <- dim(sampler_diagnostics)[2]
  div_rate <- divergences / (n_iter * n_chains)
  
  if (div_rate > 0.01) {
    cat("WARNING: Divergence rate is high (", div_rate * 100, "%)\n\n")
  } else if (div_rate > 0.001) {
    cat("CAUTION: Divergence rate is moderate (", div_rate * 100, "%)\n\n")
  } else {
    cat("Divergence rate is acceptable (", div_rate * 100, "%)\n\n")
  }
  
  # Check normality of energy distribution
  energy <- as.vector(sampler_diagnostics[,, "energy__"])
  energy_normality <- shapiro.test(energy)
  if (energy_normality$p.value < 0.05) {
    cat("WARNING: Energy distribution may not be normal (p-value ", if (round(energy_normality$p.value, 4) == 0) "< 0.001" else round(energy_normality$p.value, 4), ")\n\n")
  } else {
    cat("Energy distribution appears to be normal (p-value =", round(energy_normality$p.value, 4), ")\n\n")
  }
  
  # Check R-hat values
  rhat_values <- fit$diagnostics[,'rhat']
  max_rhat <- round(max(rhat_values, na.rm = TRUE), 4)
  q90_rhat <- round(quantile(rhat_values, 0.9, na.rm = TRUE), 4)
  
  if (max_rhat / 1.1 > rhat_threshold || q90_rhat / 1.1 > rhat_threshold) {
    cat("WARNING: R-hat values are high. \nMax R-hat/1.1 =", round(max_rhat/1.1, 4), "(", max_rhat, ")", ", \n90th percentile R-hat/1.1 =", round(q90_rhat/1.1, 4), "(", q90_rhat, ")", "\n\n")
  } else {
    cat("R-hat values are acceptable. \nMax R-hat/1.1 =", round(max_rhat/1.1, 4),  "(", max_rhat, ")", ",  \n90th percentile R-hat/1.1 =", round(q90_rhat/1.1, 4), "(", q90_rhat, ")",, "\n\n")
  }
  
  # Check Effective Sample Size (ESS)
  ess_bulk_values <- fit$diagnostics[,'ess_bulk']
  n_eff <- ess_bulk_values / prod(dim(fit$draws)[1:2])
  q10_neff <- round(quantile(n_eff, 0.1, na.rm = TRUE), 4)
  
  if (q10_neff < ess_threshold) {
    cat("WARNING: Low effective sample size. 10th percentile n_eff =", q10_neff, "\n\n")
    low_ess_params <- names(which(n_eff < ess_threshold))
    cat("Parameters with n_eff <", ess_threshold, ":", paste(low_ess_params, collapse = ", "), "\n\n")
  } else {
    cat("Effective sample sizes are acceptable. 10th percentile n_eff =", q10_neff, "\n\n")
  }
  
  # Check Monte Carlo Standard Error (MCSE)
  draws_matrix <- posterior::as_draws_matrix(fit$draws)
  mcse_values <- apply(draws_matrix, 2, posterior::mcse_mean)
  posterior_sd <- apply(draws_matrix, 2, sd)
  mcse_ratio <- mcse_values / posterior_sd
  q10_mcse_ratio <- round(quantile(mcse_ratio, 0.9, na.rm = TRUE), 4)
  
  high_mcse_params <- names(which(mcse_ratio > mcse_threshold))
  if (length(high_mcse_params) > 0) {
    cat("WARNING: High MCSE. 90th percentile MCSE ration =", q10_mcse_ratio, "\n\n")
    cat("Parameters with MCSE ratio >", mcse_threshold, ":", paste(high_mcse_params, collapse = ", "), "\n\n")
  } else {
    cat("MCSE values are acceptable for all parameters. 10th percentile ratio =", q10_mcse_ratio, "\n\n")
  }
  
  # Return a list of problematic parameters
  return(list(
    high_rhat = names(which(rhat_values / 1.1 > rhat_threshold)),
    low_ess = low_ess_params,
    high_mcse = high_mcse_params
  ))
}

# Function to calculate diagnostics for a single parameter
calculate_param_diagnostics <- function(param_draws) {
  # Calculate diagnostics
  rhat <- posterior::rhat(param_draws)
  ess_bulk <- posterior::ess_bulk(param_draws)
  ess_tail <- posterior::ess_tail(param_draws)
  
  return(c(rhat = rhat, ess_bulk = ess_bulk, ess_tail = ess_tail))
}


# Divergence check
check_divergences <- function(fit) {
  sampler_diagnostics <- fit$sampler_diagnostics
  divergences <- sum(sampler_diagnostics[,,"divergent__"])
  n_iter <- dim(sampler_diagnostics)[1]
  n_chains <- dim(sampler_diagnostics)[2]
  div_rate <- divergences / (n_iter * n_chains)
  
  cat("Number of divergent transitions:", divergences, "\n")
  cat("Rate of divergent transitions:", div_rate, "\n")
  
  if (div_rate < 0.001) {
    cat("Rate of divergent acceptable: rate < 0.001 \n")
  } else if (div_rate < 0.01) {
    cat("Rate of divergent borderline: rate < 0.01 \n")
  } else {
    cat("Rate of divergent problematic: rate > 0.01 \n")
  }
  
  # NUTS Energy Diagnostic
  energy <- sampler_diagnostics[,, "energy__"]
  energy_df <- data.frame(energy = as.vector(energy))
  p <- ggplot(energy_df, aes(x = energy)) +
    geom_histogram(bins = 30) +
    ggtitle("NUTS Energy Distribution (roughly bell-shaped?)") +
    xlab("Energy") +
    theme_minimal()
  print(p)
}

# Display density plots by chain
display_density_plots_by_chain <- function(fit, params, plots_per_page = 10) {
  plot_list <- list()
  
  for (param in params) {
    tryCatch({
      # Extract the parameter values
      param_values <- fit$draws[,, param]
      
      # Check if all values are finite
      if (all(is.finite(param_values))) {
        plot <- mcmc_dens_chains(param_values)
        plot_list[[param]] <- plot + ggtitle(param)
      } else {
        # Create a text plot for non-finite parameters
        plot <- ggplot() + 
          annotate("text", x = 0.5, y = 0.5, 
                   label = paste("Parameter", param, "contains non-finite values"),
                   size = 5) +
          theme_void() +
          labs(title = param)
        plot_list[[param]] <- plot
      }
    }, error = function(e) {
      # Create an error plot if there's any other error
      plot <- ggplot() + 
        annotate("text", x = 0.5, y = 0.5, 
                 label = paste("Error plotting parameter", param, ":", e$message),
                 size = 4, hjust = 0.5, vjust = 0.5) +
        theme_void() +
        labs(title = param)
      plot_list[[param]] <- plot
    })
  }
  
  # Display plots
  num_plots <- length(plot_list)
  num_pages <- ceiling(num_plots / plots_per_page)
  
  for (page in 1:num_pages) {
    start_idx <- (page - 1) * plots_per_page + 1
    end_idx <- min(page * plots_per_page, num_plots)
    plots_to_display <- plot_list[start_idx:end_idx]
    do.call(grid.arrange, c(plots_to_display, ncol = 2, 
                            top = paste("Density Plots by Chain (Page", page, "of", num_pages, ")")))
  }
}

# Display overall density plots
display_overall_density_plots <- function(fit, params, plots_per_page = 10) {
  plot_list <- list()
  
  for (param in params) {
    tryCatch({
      # Extract the parameter values
      param_values <- fit$draws[,, param]
      
      # Check if all values are finite
      if (all(is.finite(param_values))) {
        plot <- mcmc_dens(param_values)
        plot_list[[param]] <- plot + ggtitle(param)
      } else {
        # Create a text plot for non-finite parameters
        plot <- ggplot() + 
          annotate("text", x = 0.5, y = 0.5, 
                   label = paste("Parameter", param, "contains non-finite values"),
                   size = 5) +
          theme_void() +
          labs(title = param)
        plot_list[[param]] <- plot
      }
    }, error = function(e) {
      # Create an error plot if there's any other error
      plot <- ggplot() + 
        annotate("text", x = 0.5, y = 0.5, 
                 label = paste("Error plotting parameter", param, ":", e$message),
                 size = 4, hjust = 0.5, vjust = 0.5) +
        theme_void() +
        labs(title = param)
      plot_list[[param]] <- plot
    })
  }
  
  num_plots <- length(plot_list)
  num_pages <- ceiling(num_plots / plots_per_page)
  
  for (page in 1:num_pages) {
    start_idx <- (page - 1) * plots_per_page + 1
    end_idx <- min(page * plots_per_page, num_plots)
    plots_to_display <- plot_list[start_idx:end_idx]
    do.call(grid.arrange, c(plots_to_display, ncol = 2, top = paste("Overall Density Plots (Page", page, "of", num_pages, ")")))
  }
}

# R-hat analysis
analyze_rhat <- function(fit, params, lower_than_q = 0.1, higher_than_q = 0.9) {
  rhat_values <- fit$diagnostics[params,'rhat']
  print(summary(rhat_values))
  
  rhat_values_valid <- rhat_values[!is.na(rhat_values)]
  rhat_vals_to_print <- c(rhat_values_valid[rhat_values_valid < quantile(rhat_values_valid, lower_than_q)],
                          rhat_values_valid[rhat_values_valid > quantile(rhat_values_valid, higher_than_q)])
  
  mcmc_rhat(rhat_vals_to_print) +
    ggtitle("Highest/Lowest R-hat Values (should be < 1.1)")
}

# Effective Sample Size analysis
analyze_ess <- function(fit, params, lower_than_q = 0.25) {
  ess_bulk_values <- fit$diagnostics[params,'ess_bulk']
  n_eff <- ess_bulk_values / prod(dim(fit$draws)[1:2])  # Divide by total number of draws
  print(summary(n_eff))
  
  hist(n_eff, main = "Effective Sample Size Ratio (higher is better)", xlab = "Neff/N")
  mcmc_neff(n_eff[n_eff < quantile(n_eff, lower_than_q, na.rm = TRUE)])
}

# Monte Carlo Standard Error analysis
analyze_mcse <- function(fit, params) {
  # Extract draws for the specified parameters
  draws_matrix <- posterior::as_draws_matrix(fit$draws[,, params])
  
  mcse_values <- apply(draws_matrix, 2, posterior::mcse_mean)
  posterior_sd <- apply(draws_matrix, 2, sd)
  mcse_ratio <- mcse_values / posterior_sd
  
  print(summary(mcse_ratio))
  
  hist(mcse_values, main = "Monte Carlo Standard Error (lower is better)", xlab = "MCSE")
  hist(mcse_ratio, main = "MCSE / Posterior SD Ratio", xlab = "Ratio (should be < 0.1)")
  
  params_to_check <- names(which(mcse_ratio > 0.1))
  if (length(params_to_check) > 0) {
    cat("Parameters with MCSE > 10% of posterior SD:", paste(params_to_check, collapse = ", "), "\n")
  } else {
    cat("No Parameters with MCSE > 10% of posterior SD")
  }
}

extract_sample_data <- function(data, data_params, n_trials = NULL, n_subs = NULL, 
                                RTbound_ms = 50, RTbound_reject_ms = 100, rt_method = "remove", 
                                use_percentile = FALSE, minrt_ep_ms = 0, SID = F) {
  # Preprocess data
  data <- preprocess_data(data, RTbound_reject_ms, rt_method)
  
  # Determine if it's a group or single subject data
  is_group <- length(unique(data$sid)) > 1
  
  # Process n_trials and n_subs
  if (is.null(n_trials) || n_trials == "Full") {
    n_trials <- max(data$trial)
  } else {
    n_trials <- as.integer(n_trials)
  }
  
  if (is.null(n_subs) || n_subs == "Full") {
    n_subs <- if (is_group) length(unique(data$sid)) else 1
  } else {
    n_subs <- as.integer(n_subs)
  }
  
  # Subset data based on n_trials and n_subs
  if (is_group) {
    data <- data %>%
      dplyr::group_by(sid) %>%
      filter(trial <= n_trials) %>%
      ungroup() %>%
      dplyr::group_by(sid) %>%
      filter(dplyr::n() == n_trials) %>%
      ungroup()
    
    if (nrow(data) / n_trials > n_subs) {
      selected_sids <- sample(unique(data$sid), n_subs)
      data <- data %>% filter(sid %in% selected_sids)
    }
  } else {
    data <- data %>% filter(trial <= n_trials)
  }
  
  # Initialize the data list
  data_list <- list()
  
  if (SID) {
    data_list$sid = unique(data$sid)
  }
  
  # Helper function to create a matrix from a data frame
  create_matrix <- function(df, value_var, n_trials) {
    df %>%
      select(sid, trial, !!sym(value_var)) %>%
      pivot_wider(names_from = trial, values_from = !!sym(value_var), names_prefix = "trial_") %>%
      select(-sid) %>%
      as.matrix() %>%
      unname()
  }
  
  # Process requested parameters
  for (param in data_params) {
    switch(param,
           "N" = if (is_group) data_list$N <- as.integer(length(unique(data$sid))),
           "T" = {
             data_list$T <- as.integer(n_trials)
             if (is_group) data_list$Tsubj <- as.integer(rep(n_trials, length(unique(data$sid))))
           },
           "choice" = data_list$choice <- if (is_group) as.matrix(create_matrix(data, "v_response", n_trials) - 1) else as.vector(as.integer(data$v_response - 1)),
           "shown" = data_list$shown <- if (is_group) as.matrix(create_matrix(data, "v_targetdeck", n_trials)) else as.vector(as.integer(data$v_targetdeck)),
           "outcome" = data_list$outcome <- if (is_group) as.matrix(create_matrix(data, "v_netchange", n_trials)) else as.vector(as.numeric(data$v_netchange)),
           "Nplay" = data_list$Nplay <- if (is_group) as.integer(rowSums(data_list$choice == 1, na.rm = TRUE)) else as.integer(sum(data$v_response == 2, na.rm = TRUE)),
           "Npass" = data_list$Npass <- if (is_group) as.integer(rowSums(data_list$choice == 0, na.rm = TRUE)) else as.integer(sum(data$v_response == 1, na.rm = TRUE)),
           "Nplay_max" = data_list$Nplay_max <- if (is_group) max(as.integer(rowSums(data_list$choice == 1, na.rm = TRUE))),
           "Npass_max" = data_list$Npass_max <- if (is_group) max(as.integer(rowSums(data_list$choice == 0, na.rm = TRUE))),
           "RT" = {
             data_list$RT <- if (is_group) as.matrix(create_matrix(data, "RT", n_trials)) else as.vector(as.numeric(data$RT))
             
             if (use_percentile) {
               all_RTs <- as.vector(data_list$RT)
               RTbound <- as.numeric(quantile(head(sort(all_RTs), 100), 0.01))
             } else if (rt_method == "adaptive") {
               RTbound <- as.numeric(min(data_list$RT, na.rm = TRUE) - 1e-5)  # small epsilon
             } else {
               RTbound <- as.numeric(RTbound_ms) / 1000
             }
             
             data_list$RTbound <- as.numeric(RTbound)
             data_list$minRT <- if (is_group) as.numeric(apply(data_list$RT, 1, min, na.rm = TRUE)) else as.numeric(min(data_list$RT, na.rm = TRUE))
             data_list$minRT = data_list$minRT + pmax(minrt_ep_ms/1000, 0)
           },
           "RTplay" = {
             RT_mat <- if (is_group) as.matrix(create_matrix(data, "RT", n_trials)) else matrix(as.numeric(data$RT), nrow = 1)
             choice_mat <- if (is_group) data_list$choice else matrix(as.integer(data$v_response - 1), nrow = 1)
             data_list$RTplay <- RT_mat
             data_list$RTplay[choice_mat != 1] <- NA
             if (is_group) {
               data_list$RTplay <- t(apply(data_list$RTplay, 1, function(x) c(na.omit(x), rep(NA, sum(is.na(x))))))
             } else {
               data_list$RTplay <- c(na.omit(as.vector(data_list$RTplay)), rep(NA, sum(is.na(data_list$RTplay))))
             }
             data_list$RTplay <- as.matrix(data_list$RTplay)
           },
           "RTpass" = {
             RT_mat <- if (is_group) as.matrix(create_matrix(data, "RT", n_trials)) else matrix(as.numeric(data$RT), nrow = 1)
             choice_mat <- if (is_group) data_list$choice else matrix(as.integer(data$v_response - 1), nrow = 1)
             data_list$RTpass <- RT_mat
             data_list$RTpass[choice_mat != 0] <- NA
             if (is_group) {
               data_list$RTpass <- t(apply(data_list$RTpass, 1, function(x) c(na.omit(x), rep(NA, sum(is.na(x))))))
             } else {
               data_list$RTpass <- c(na.omit(as.vector(data_list$RTpass)), rep(NA, sum(is.na(data_list$RTpass))))
             }
             data_list$RTpass <- as.matrix(data_list$RTpass)
           }
    )
  }
  return(data_list)
}

preprocess_data <- function(data, RTbound_ms, rt_method = "remove") {
  # Ensure data is a data frame
  data <- as.data.frame(data)
  
  # Convert columns to appropriate types
  data <- data %>%
    dplyr::mutate(
      sid = as.factor(sid),
      v_response = as.integer(v_response),
      v_targetdeck = as.integer(v_targetdeck),
      v_netchange = as.numeric(v_netchange),
      latency = as.numeric(latency)
    )
  
  RTbound <- as.numeric(RTbound_ms) / 1000  # Convert to seconds
  
  # Convert latency to seconds
  data$RT <- data$latency / 1000
  
  if (rt_method == "remove") {
    # Remove trials with RT < RTbound
    data <- data[data$RT >= RTbound, ]
  } else if (rt_method == "force") {
    # Force RTs below bound to be equal to bound
    data$RT[data$RT < RTbound] <- RTbound
  }
  # For "adaptive" method, we'll handle it in the main function
  
  # Reset trial numbers
  data <- data %>%
    dplyr::group_by(sid) %>%
    dplyr::mutate(trial = row_number()) %>%
    ungroup()
  
  return(data)
}

# EXTRACT PARAMETERS
# Extract base name of a parameter
get_base_name <- function(param) {
  gsub("_pr$", "", gsub("\\[.*\\]", "", param))  # Remove indices and _pr suffix
}

# Check if a parameter has indices
has_indices <- function(param) {
  grepl("\\[", param)
}

# Categorize parameters
categorize_params <- function(params, hier_params_vec, main_params_vec, output_params_vec) {
  base_names <- sapply(params, get_base_name)
  
  hier_params <- base_names[sapply(base_names, function(p) any(sapply(hier_params_vec, function(h) grepl(paste0("^", h), p))))]
  main_params <- base_names[base_names %in% main_params_vec]
  output_params <- base_names[base_names %in% output_params_vec]
  
  other_params <- setdiff(base_names, c(hier_params, main_params, output_params))
  
  list(hier = hier_params, main = main_params, output = output_params, other = other_params)
}

sample_params <- function(params, sampled_indices) {
  if (length(params) < 1){
    return(c())
  } else if (length(names(params)) < 1) {
    return(params)
  }
  
  # If there's only one subject, return all parameters without sampling
  if (length(sampled_indices) == 1) {
    return(names(params))
  }
  
  sampled_params <- c()
  
  # Split parameters into base names and indices
  param_split <- strsplit(names(params), "\\[")
  base_names <- sapply(param_split, `[`, 1)
  
  # Group parameters by their base name
  param_families <- split(names(params), base_names)
  
  for (family in param_families) {
    indexed_params <- family[grepl("\\[", family)]
    non_indexed_params <- unname(family[!grepl("\\[", family)])
    
    if (length(indexed_params) > 0) {
      # For parameters with indices
      sampled_indexed <- indexed_params[sampled_indices]
      
      sampled_params <- c(sampled_params, non_indexed_params, sampled_indexed)
    } else {
      # For parameters without indices, include all
      sampled_params <- c(sampled_params, unname(family)[[1]])
    }
  }
  
  return(sampled_params)
}

sort_params <- function(params, hier_params_vec, main_params_vec, output_params_vec) {
  param_order <- function(param) {
    base_param <- get_base_name(param)
    
    hier_index <- which(sapply(hier_params_vec, function(h) grepl(paste0("^", h), base_param)))
    if (length(hier_index) > 0) return(hier_index[1])
    
    main_index <- which(main_params_vec == base_param)
    if (length(main_index) > 0) {
      return(length(hier_params_vec) + 2 * main_index[1] - (if(grepl("_pr$", param)) 1 else 0))
    }
    
    output_index <- which(output_params_vec == base_param)
    if (length(output_index) > 0) {
      return(length(hier_params_vec) + 2 * length(main_params_vec) + output_index[1])
    }
    
    return(1000)  # If not found in any category, put at the end
  }
  
  params[order(sapply(params, param_order))]
}

# Main function
extract_params <- function(param_names, n_subs = 1, num_to_view = 10, drop_lp = TRUE, 
                           hier_params_vec = c("mu", "sigma", "mu_"),
                           main_params_vec = NULL,
                           output_params_vec = NULL) {
  
  # Remove "lp__" if drop_lp is TRUE
  if (drop_lp) {
    param_names <- param_names[param_names != "lp__"]
  }
  
  categorized_params <- categorize_params(param_names, hier_params_vec, main_params_vec, output_params_vec)
  
  # Handle single subject case
  if (n_subs == 1) {
    sampled_indices_other <- 1
    sampled_indices_subs <- 1
    sampled_indices_trials <- 1:num_to_view  # Keep trial sampling for output params
  } else if (n_subs < num_to_view) {
    sampled_indices_other <- sort(sample(1:num_to_view, num_to_view))
    sampled_indices_subs <- 1:n_subs
    sampled_indices_trials <- sort(sample(1:num_to_view, num_to_view))
  }else {
    sampled_indices_other <- sort(sample(1:num_to_view, num_to_view))
    sampled_indices_subs <- sort(sample(1:n_subs, num_to_view))
    sampled_indices_trials <- sort(sample(1:num_to_view, num_to_view))
  }
  
  # Sample and combine parameters
  sampled_params <- c(
    sample_params(categorized_params$hier, 1:length(main_params_vec)),
    sample_params(categorized_params$main, sampled_indices_subs),
    sample_params(categorized_params$output, sampled_indices_trials),
    sample_params(categorized_params$other, sampled_indices_other)
  )
  
  # Sort parameters
  sorted_params <- sort_params(sampled_params, hier_params_vec, main_params_vec, output_params_vec)
  
  # Remove duplicates while preserving order
  sorted_params <- sorted_params[!duplicated(sorted_params)]
  
  return(sorted_params)
}

# Diagnostics
run_selected_diagnostics <- function(fit, steps_to_run = NULL, params = NULL, plots_pp = 10, lower_than_q = 0.1, higher_than_q = 0.9) {
  available_steps <- c(
    "divergences", "traceplots", "density_plots_by_chain", "overall_density_plots",
    "rhat", "rhat_all", "ess", "ess_all", "mcse", "mcse_all", "autocorrelation", "parallel_coordinates", "pairs_plot"
  )
  
  if (is.null(steps_to_run)) {
    steps_to_run <- available_steps
  } else {
    invalid_steps <- setdiff(steps_to_run, available_steps)
    if (length(invalid_steps) > 0) {
      stop(paste("Invalid step(s):", paste(invalid_steps, collapse = ", ")))
    }
  }
  
  params = fit$params
  for (step in steps_to_run) {
    cat("\n\n### Running:", step, "\n")
    switch(step,
           divergences = {
             check_divergences(fit)
           },
           traceplots = {
             print(mcmc_trace(fit$draws[,, sample(params, 10)]) +
                     ggtitle("Trace Plots (should resemble white noise)"))
           },
           density_plots_by_chain = {
             display_density_plots_by_chain(fit, params, plots_per_page = plots_pp)
           },
           overall_density_plots = {
             display_overall_density_plots(fit, params, plots_per_page = plots_pp)
           },
           rhat = {
             analyze_rhat(fit, params, lower_than_q = lower_than_q, higher_than_q = higher_than_q)
           },
           rhat_all = {
             analyze_rhat(fit, variables(fit$draws), lower_than_q = lower_than_q, higher_than_q = higher_than_q)
           },
           ess = {
             analyze_ess(fit, params, lower_than_q = lower_than_q)
           },
           ess_all = {
             analyze_ess(fit, variables(fit$draws), lower_than_q = lower_than_q)
           },
           mcse = {
             analyze_mcse(fit, params)
           },
           mcse_all = {
             analyze_mcse(fit, variables(fit$draws))
           },
           autocorrelation = {
             print(mcmc_acf(fit$draws[,, params]) +
                     ggtitle("Autocorrelation (Should decay quickly)"))
           },
           parallel_coordinates = {
             print(mcmc_parcoord(fit$draws[,, params]) +
                     ggtitle("Parallel Coordinates Plot"))
           },
           pairs_plot = {
             bayesplot::mcmc_pairs(fit$draws[,, params])
           }
    )
  }
}