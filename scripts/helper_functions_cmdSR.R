# Load required libraries
suppressPackageStartupMessages({
  library(abind)
  library(coda)
  library(cmdstanr)
})

## Functions
# Fit function
fit_stan_model <- function(stanmodel_arg, data_list, n_warmup = 1000, n_iter = 10000, n_chains = 2, adapt_delta = 0.95, max_treedepth = 12, init_opt = "random", parallel = TRUE) {
  if (parallel){
    options(mc.cores = parallel::detectCores())
  }
  
  fit <- rstan::sampling(
    object = stanmodel_arg,
    data = data_list,
    init = init_opt,
    init_r = 0.1,
    chains = n_chains,
    iter = n_iter,
    warmup = n_warmup,
    thin = 1,
    control = list(
      adapt_delta = adapt_delta,
      stepsize = 1,
      max_treedepth = max_treedepth
    )
  )
  
  return(list(
    fit = fit,
    n_warmup = n_warmup,
    n_iter = n_iter,
    n_chains = n_chains,
    adapt_delta = adapt_delta,
    max_treedepth = max_treedepth,
    tss = (n_iter - n_warmup) * n_chains,
    all_params = names(fit),
    list_params = unique(gsub("\\[.*?\\]", "", names(fit))),
    samples = rstan::extract(fit)
  ))
}

# Divergence check
check_divergences <- function(fit) {
  sampler_params <- rstan::get_sampler_params(fit, inc_warmup = FALSE)
  divergences <- sum(sapply(sampler_params, function(x) sum(x[, "divergent__"])))
  n_iter <- dim(sampler_params[[1]])[1]
  n_chains <- length(sampler_params)
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
  if ("energy__" %in% colnames(sampler_params[[1]])) {
    energy <- do.call(rbind, lapply(sampler_params, function(x) x[, "energy__"]))
    energy_df <- data.frame(energy = as.vector(energy))
    p <- ggplot(energy_df, aes(x = energy)) +
      geom_histogram(bins = 30) +
      ggtitle("NUTS Energy Distribution (roughly bell-shaped?)") +
      xlab("Energy") +
      theme_minimal()
    print(p)
  }
}

# Display density plots by chain
display_density_plots_by_chain <- function(fit, params, plots_per_page = 10) {
  plot_list <- list()
  
  for (param in params) {
    tryCatch({
      # Extract the parameter values
      param_values <- rstan::extract(fit, pars = param)[[1]]
      
      # Check if all values are finite
      if (all(is.finite(param_values))) {
        plot <- mcmc_dens_chains(fit, pars = param)
        plot_list[[param]] <- plot
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
      param_values <- rstan::extract(fit, pars = param)[[1]]
      
      # Check if all values are finite
      if (all(is.finite(param_values))) {
        plot <- mcmc_dens(fit, pars = param)
        plot_list[[param]] <- plot
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
    do.call(grid.arrange, c(plots_to_display, ncol = 2, top =paste("Overall Density Plots (Page", page, "of", num_pages, ")")))
  }
}

# R-hat analysis
analyze_rhat <- function(fit, params, lower_than_q = 0.1, higher_than_q = 0.9) {
  rhat_values <- bayesplot::rhat(fit, pars = params)
  print(summary(rhat_values))
  
  rhat_values_valid <- rhat_values[!is.na(rhat_values)]
  rhat_vals_to_print <- c(rhat_values_valid[rhat_values_valid < quantile(rhat_values_valid, lower_than_q)],
                          rhat_values_valid[rhat_values_valid > quantile(rhat_values_valid, higher_than_q)])
  
  mcmc_rhat(rhat_vals_to_print) +
    ggtitle("Highest/Lowest R-hat Values (should be < 1.1)")
}

# Effective Sample Size analysis
analyze_ess <- function(fit, params, lower_than_q = 0.25) {
  n_eff <- neff_ratio(fit, pars = params)
  print(summary(n_eff))
  
  hist(n_eff, main = "Effective Sample Size Ratio (higher is better)", xlab = "Neff/N")
  mcmc_neff(n_eff[n_eff < quantile(n_eff, lower_than_q, na.rm = TRUE)])
}

# Monte Carlo Standard Error analysis
analyze_mcse <- function(fit, params) {
  mcse_values <- apply(as_draws_matrix(fit), 2, mcse_mean)
  posterior_sd <- apply(as_draws_matrix(fit), 2, sd)
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
                                use_percentile = FALSE, minrt_ep_ms = 0) {
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
      group_by(sid) %>%
      filter(trial <= n_trials) %>%
      ungroup() %>%
      group_by(sid) %>%
      filter(n() == n_trials) %>%
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
    mutate(
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
    group_by(sid) %>%
    mutate(trial = row_number()) %>%
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
  } else {
    sampled_indices_other <- sort(sample(1:num_to_view, num_to_view))
    sampled_indices_subs <- sort(sample(1:n_subs, num_to_view))
    sampled_indices_trials <- sort(sample(1:num_to_view, num_to_view))
  }
  
  # Sample and combine parameters
  sampled_params <- c(
    sample_params(categorized_params$hier, sampled_indices_other),
    sample_params(categorized_params$main, sampled_indices_subs),
    sample_params(categorized_params$output, sampled_indices_trials),
    sample_params(categorized_params$other, sampled_indices_other)
  )
  
  # Sort parameters
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
  
  sorted_params <- sort_params(sampled_params, hier_params_vec, main_params_vec, output_params_vec)
  
  # Remove duplicates while preserving order
  sorted_params <- sorted_params[!duplicated(sorted_params)]
  
  return(sorted_params)
}

# Diagnostics
run_selected_diagnostics <- function(model_fit_obj, steps_to_run = NULL, params = NULL, plots_pp = 10, lower_than_q = 0.1, higher_than_q = 0.9) {
  available_steps <- c(
    "divergences", "traceplots", "density_plots_by_chain", "overall_density_plots",
    "rhat", "ess", "mcse", "autocorrelation", "parallel_coordinates", "pairs_plot"
  )
  
  if (is.null(steps_to_run)) {
    steps_to_run <- available_steps
  } else {
    invalid_steps <- setdiff(steps_to_run, available_steps)
    if (length(invalid_steps) > 0) {
      stop(paste("Invalid step(s):", paste(invalid_steps, collapse = ", ")))
    }
  }
  
  for (step in steps_to_run) {
    cat("\n\n### Running:", step, "\n")
    switch(step,
           divergences = {
             check_divergences(model_fit_obj$fit)
           },
           traceplots = {
             print(mcmc_trace(model_fit_obj$fit, params, iter1 = model_fit_obj$n_warmup + 1) +
                     ggtitle("Trace Plots (should resemble white noise)"))
             # Should be white noise (hairy caterpillars with out any obvious trends or patterns)
           },
           density_plots_by_chain = {
             display_density_plots_by_chain(model_fit_obj$fit, params, plots_per_page = plots_pp)
           },
           overall_density_plots = {
             display_overall_density_plots(model_fit_obj$fit, params, plots_per_page = plots_pp)
           },
           rhat = {
             analyze_rhat(model_fit_obj$fit, params, lower_than_q = lower_than_q, higher_than_q = higher_than_q)
           },
           ess = {
             analyze_ess(model_fit_obj$fit, params, lower_than_q = lower_than_q)
           },
           mcse = {
             analyze_mcse(model_fit_obj$fit, params)
           },
           autocorrelation = {
             print(mcmc_acf(model_fit_obj$fit, pars = params) +
                     ggtitle("Autocorrelation (Should decay quickly)")) # High AC at high lags suggests poor mixing
           },
           parallel_coordinates = {
             print(mcmc_parcoord(model_fit_obj$fit, pars =params) +
                     ggtitle("Parallel Coordinates Plot"))
             # Should be a mix of criss-crossing lines without any strong patterns
             # If lines are very tangled or show clear patterns, might indicate high correlation between params
           },
           pairs_plot = {
             pairs(model_fit_obj$fit, pars = params) # Should not have red dots (divergences)
           }
    )
  }
}