suppressPackageStartupMessages({
  library(R6)
  library(data.table)
  library(optparse)
  library(here)
})

# Define deck properties based on Table 2
DECK_PROPERTIES <- list(
  A = list(gain = 100, loss = -250, prob_gain = 0.50, prob_loss = 0.50, prob_zero = 0.00, first_loss = 3, expected_value = 25),
  B = list(gain = 100, loss = -1150, prob_gain = 0.90, prob_loss = 0.10, prob_zero = 0.00, first_loss = 10, expected_value = 25),
  C = list(gain = 50, loss = -25, prob_gain = 0.50, prob_loss = 0.25, prob_zero = 0.25, first_loss = 13, expected_value = 18.75),
  D = list(gain = 50, loss = -200, prob_gain = 0.90, prob_loss = 0.10, prob_zero = 0.00, first_loss = 10, expected_value = 25)
)

# Define parameter ranges for suggestion
PARAM_RANGES <- list(
  con = list(range = c(-2, 2), min = -2, max = 2, low = c(-2, -1), medium = c(-1, 1), high = c(1, 2)),
  wgt_pun = list(range = c(0, 1), min = 0, max = 1, low = c(0, 0.3), medium = c(0.3, 0.7), high = c(0.7, 1)),
  wgt_rew = list(range = c(0, 1), min = 0, max = 1, low = c(0, 0.3), medium = c(0.3, 0.7), high = c(0.7, 1)),
  update = list(range = c(0, 1), min = 0, max = 1, low = c(0, 0.3), medium = c(0.3, 0.7), high = c(0.7, 1))
)

# Subject class (Model 1 - EVL)
Subject <- R6Class("Subject",
                   public = list(
                     id = NULL,
                     ev = NULL,
                     con = NULL,
                     sensitivity = NULL,
                     update = NULL,
                     wgt_pun = NULL,
                     wgt_rew = NULL,
                     deck_plays = NULL,
                     
                     initialize = function(id, params = NULL) {
                       self$id <- id
                       self$ev <- rep(0, 4)  # Initial expected values for 4 decks
                       if (!is.null(params)) {
                         self$con <- params$con
                         self$update <- params$update
                         self$wgt_pun <- params$wgt_pun
                         self$wgt_rew <- params$wgt_rew
                       } else {
                         self$con <- runif(1, -2, 2)
                         self$update <- runif(1, 0, 1)
                         self$wgt_pun <- runif(1, 0, 1)
                         self$wgt_rew <- runif(1, 0, 1)
                       }
                       self$sensitivity <- function(t) t^self$con / 10  # Sensitivity function
                       self$deck_plays <- rep(0, 4)  # Track number of plays for each deck
                     },
                     
                     make_decision = function(shown_deck, t) {
                       info <- self$sensitivity(t) * self$ev[shown_deck]
                       prob_play <- 1 / (1 + exp(-info))  # Logistic function
                       decision <- sample(c(0, 1), 1, prob = c(1 - prob_play, prob_play))
                       return(decision)
                     },
                     
                     update_knowledge = function(deck, outcome, choice) {
                       if (choice == 1) {  # Only update if the deck was played
                         self$deck_plays[deck] <- self$deck_plays[deck] + 1
                         utility <- ifelse(outcome > 0, self$wgt_rew, self$wgt_pun) * outcome
                         self$ev[deck] <- self$ev[deck] + (utility - 2 * self$ev[deck]) * self$update
                       }
                     },
                     
                     get_ev = function() {
                       return(self$ev)
                     }
                   )
)

# Session class
Session <- R6Class("Session",
                   public = list(
                     data = NULL,
                     
                     initialize = function() {
                       self$data <- data.table()
                     },
                     
                     add_trial = function(subject_id, trial, block, deck_shown, choice, gain, loss, net_outcome, ev_A, ev_B, ev_C, ev_D, is_training = FALSE) {
                       new_row <- data.table(
                         subject_id = subject_id,
                         trial = trial,
                         block = block,
                         deck_shown = deck_shown,
                         choice = choice,
                         gain = gain,
                         loss = loss,
                         net_outcome = net_outcome,
                         ev_A = ev_A,
                         ev_B = ev_B,
                         ev_C = ev_C,
                         ev_D = ev_D,
                         is_training = is_training
                       )
                       self$data <- rbindlist(list(self$data, new_row))
                     }
                   )
)

# Helper function to generate outcome
generate_outcome <- function(deck, trial) {
  props <- DECK_PROPERTIES[[deck]]
  
  if (trial < props$first_loss) {
    # Force a win if loss is not possible yet
    return(list(gain = props$gain, loss = 0, net_outcome = props$gain))
  }
  
  rand <- runif(1)
  if (rand < props$prob_zero) {
    return(list(gain = 0, loss = 0, net_outcome = 0))
  } else if (rand < props$prob_zero + props$prob_gain) {
    return(list(gain = props$gain, loss = 0, net_outcome = props$gain))
  } else {
    return(list(gain = 0, loss = props$loss, net_outcome = props$loss))
  }
}


# Simulation functions
generate_balanced_deck_sequence <- function(n_blocks, trials_per_block) {
  total_trials <- n_blocks * trials_per_block
  trials_per_deck <- total_trials / 4
  
  balanced_sequence <- rep(1:4, each = trials_per_deck)
  shuffled_sequence <- sample(balanced_sequence)
  
  return(shuffled_sequence)
}

simulate_igt_trial <- function(subject, session, trial, block, deck_shown, forced_choice = NULL) {
  if (is.null(forced_choice)) {
    choice <- subject$make_decision(deck_shown, trial)
  } else {
    choice <- forced_choice
  }
  
  if (choice == 1) {
    outcome <- generate_outcome(LETTERS[deck_shown], trial)
    gain <- outcome$gain
    loss <- outcome$loss
    net_outcome <- outcome$net_outcome
  } else {
    gain <- 0
    loss <- 0
    net_outcome <- 0
  }
  
  subject$update_knowledge(deck_shown, net_outcome, choice)
  ev <- subject$get_ev()
  session$add_trial(subject$id, trial, block, deck_shown, choice, gain, loss, net_outcome, 
                    ev[1], ev[2], ev[3], ev[4], is_training = !is.null(forced_choice))
  
  return(list(subject = subject, session = session))
}

simulate_igt_session <- function(subject, n_blocks = 6, trials_per_block = 20, training_decks = NULL, training_choices = NULL) {
  session <- Session$new()
  total_trials <- n_blocks * trials_per_block
  
  deck_sequence <- generate_balanced_deck_sequence(n_blocks, trials_per_block)
  
  if (!is.null(training_decks) && !is.null(training_choices)) {
    stopifnot(length(training_decks) == length(training_choices))
    n_training_trials <- length(training_decks)
    
    for (t in 1:n_training_trials) {
      result <- simulate_igt_trial(subject, session, t, 1, deck_shown = training_decks[t], forced_choice = training_choices[t])
      subject <- result$subject
      session <- result$session
    }
    
    start_trial <- n_training_trials + 1
  } else {
    start_trial <- 1
  }
  
  for (trial in start_trial:total_trials) {
    block <- ceiling(trial / trials_per_block)
    deck_shown <- deck_sequence[trial]
    result <- simulate_igt_trial(subject, session, trial, block, deck_shown)
    subject <- result$subject
    session <- result$session
  }
  
  return(session)
}

# Modified function to generate IGT data for multiple subjects
generate_igt_data <- function(n_subjects, param_sets, training_decks = NULL, training_choices = NULL) {
  all_sessions <- list()
  subject_params <- data.table()
  
  for (i in 1:n_subjects) {
    params <- param_sets[[i]]
    subject <- Subject$new(id = i, params = params)
    session <- simulate_igt_session(subject, training_decks = training_decks, training_choices = training_choices)
    all_sessions[[i]] <- session$data
    
    subject_params <- rbindlist(list(subject_params, data.table(
      subject_id = i,
      con = subject$con,
      update = subject$update,
      wgt_pun = subject$wgt_pun,
      wgt_rew = subject$wgt_rew
    )))
  }
  
  simulated_data <- rbindlist(all_sessions)
  
  result <- list(data = simulated_data, subject_params = subject_params)
  
  return(result)
}

# Parse command line arguments
option_list <- list(
  make_option(c("-n", "--n_subjects"), type="integer", default=100, help="Number of subjects (def: 100)"),
  make_option(c("-b", "--n_blocks"), type="integer", default=6, help="Number of blocks (def: 6)"),
  make_option(c("-t", "--trials_per_block"), type="integer", default=20, help="Trials per block (def: 20)"),
  make_option(c("-p", "--param_space_exp"), type="character", default="ssFPSE", help="Parameter Space Exploration type to generate parameter values (def: ssFPSE) [mbSPSepse, sbSPSepse, tSPSepse, hpsEPSE, thpsEPSE]"),
  make_option(c("-o", "--output_dir"), type="character", default=NULL, help="Output directory for simulated data (def: Data/sim/txt"),
  make_option(c("-d", "--params_dir"), type="character", default=NULL, help="Directory for simulated parameters (def: Data/sim/params"),
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
DATA_SIM_TXT_DIR <- file.path(DATA_SIM_DIR, "txt")
DATA_SIM_PARAMS_DIR <- file.path(DATA_SIM_DIR, "params")

if (is.null(opt$output_dir)) {
  opt$output_dir <- DATA_SIM_TXT_DIR
}

if (is.null(opt$params_dir)) {
  opt$params_dir <- DATA_SIM_PARAMS_DIR
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

switch(
  opt$param_space_exp,
  "ssFPSE" = {
    filepath = file.path(opt$params_dir, paste0(full_model_name, "_desc-sim_params_", opt$param_space_exp, ".rds"))
    sim_params <- readRDS(filepath)
    cat("Loaded Stratified Sampling FPSE simulated parameters from", filepath, "\n")
  },
  "mbSPSepse" = {
    filepath = file.path(opt$params_dir, paste0(full_model_name, "_desc-sim_params_", opt$param_space_exp, ".rds"))
    sim_params <- readRDS(filepath)
    cat("Loaded Median-based Subject Posterior Sampling EPSE simulated parameters from", filepath, "\n")
  },
  "sbSPSepse" = {
    filepath = file.path(opt$params_dir, paste0(full_model_name, "_desc-sim_params_", opt$param_space_exp, ".rds"))
    sim_params <- readRDS(filepath)
    cat("Loaded Simulation-based Subject Posterior Sampling EPSE simulated parameters from", filepath, "\n")
  },
  "tSPSepse" = {
    filepath = file.path(opt$params_dir, paste0(full_model_name, "_desc-sim_params_", opt$param_space_exp, ".rds"))
    sim_params <- readRDS(filepath)
    cat("Loaded Tuple Subject Posterior Sampling EPSE simulated parameters from", filepath, "\n")
  },
  "hpsEPSE" = {
    filepath = file.path(opt$params_dir, paste0(full_model_name, "_desc-sim_params_", opt$param_space_exp, ".rds"))
    sim_params <- readRDS(filepath)
    cat("Loaded Hierarchical Posterior Sampling EPSE simulated parameters from", filepath, "\n")
  },
  "thpsEPSE" = {
    filepath = file.path(opt$params_dir, paste0(full_model_name, "_desc-sim_params_", opt$param_space_exp, ".rds"))
    sim_params <- readRDS(filepath)
    cat("Loaded Tuple Hierarchical Posterior Sampling EPSE simulated parameters from", filepath, "\n")
  },
  stop(paste0("Please a valid parameter exploration type. This is invalid: ", opt$param_space_exp))
)

# Define training decks and choices
training_decks <- c(4, 3, 2, 3, 3, 2, 2, 4, 2, 1, 4, 1, 4, 3, 2, 4, 3, 1, 1, 1)
training_choices <- c(1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0)

simulation_result <- generate_igt_data(
  n_subjects = opt$n_subjects,
  param_sets = sim_params,
  training_decks = training_decks,
  training_choices = training_choice
)

# Save data
fwrite(simulation_result$data, file.path(opt$output_dir, paste0("sim_", full_model_name, "_desc-data_", opt$param_space_exp, ".csv")))
fwrite(simulation_result$subject_params, file.path(opt$output_dir, paste0("sim_", full_model_name, "_desc-params_", opt$param_space_exp, ".csv")))

cat("Simulation completed. Data saved to", opt$output_dir, "\n")

# Print summary
print(summary(simulation_result$data))
print(summary(simulation_result$subject_params))