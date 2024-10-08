#!/usr/bin/env Rscript

# Load required libraries
suppressPackageStartupMessages({
  library(optparse)
  library(here)
  library(cmdstanr)
  library(digest)
})

# Set up command line options
option_list <- list(
  make_option(c("-t", "--type"), type="character", default="all",
              help="Model type(s) to compile: fit, postpc, prepc, or all [default= %default]"),
  make_option(c("-m", "--models"), type="character", default="all",
              help="String to match model names, or 'all' [default= %default]"),
  make_option(c("-v", "--verbose"), action="store_true", default=FALSE,
              help="Print verbose output [default= %default]"),
  make_option(c("--dry-run"), action="store_true", default=FALSE,
              help="Show what would be compiled without actually compiling [default= %default]"),
  make_option(c("-y", "--yes"), action="store_true", default=FALSE,
              help="Automatically compile all matching models without prompting [default= %default]")
)

# Parse command line arguments
opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

# Set up directory paths
PROJ_DIR <- here::here()
MODELS_DIR <- file.path(PROJ_DIR, "models")
MODELS_TXT_DIR <- file.path(MODELS_DIR, "txt")
MODELS_BIN_DIR <- file.path(MODELS_DIR, "bin")

# Function to compile Stan model
compile_stan_model <- function(stan_file, bin_dir, verbose = FALSE) {
  tryCatch({
    stan_filename <- basename(stan_file)
    exe_file <- file.path(bin_dir, paste0(tools::file_path_sans_ext(stan_filename), ".stan"))
    hash_file <- paste0(tools::file_path_sans_ext(exe_file), ".hash")
    
    # Calculate hash of the Stan file
    current_hash <- digest::digest(stan_file, file = TRUE)
    
    # Check if compilation is necessary
    need_compile <- TRUE
    if (file.exists(exe_file) && file.exists(hash_file)) {
      old_hash <- readLines(hash_file)
      if (current_hash == old_hash) {
        need_compile <- FALSE
        if (verbose) cat(paste0("No changes detected, skipping compilation: ", stan_filename, "\n"))
      }
    }
    
    if (need_compile) {
      if (verbose) cat(paste0("Compiling: ", stan_filename, "\n"))
      model <- cmdstan_model(stan_file, compile = TRUE, cpp_options = list(stan_threads = TRUE),
                             force_recompile = TRUE, exe_file = exe_file,
                             quiet = !verbose)
      # Save the new hash
      writeLines(current_hash, hash_file)
      if (verbose) cat(paste0("Completed: ", stan_filename, "\n\n"))
    }
  }, error = function(e) {
    cat(paste0("Error compiling ", stan_filename, ": ", e$message, "\n"))
    return(NULL)
  })
}

# Function to find matching models
find_matching_models <- function(model_types, model_pattern) {
  matching_models <- list()
  
  for (type in model_types) {
    txt_dir <- file.path(MODELS_TXT_DIR, type)
    if (!dir.exists(txt_dir)) {
      warning(paste("Directory does not exist:", txt_dir))
      next
    }
    
    stan_files <- list.files(txt_dir, pattern = "\\.stan$", full.names = TRUE)
    if (model_pattern != "all") {
      stan_files <- stan_files[grepl(model_pattern, basename(stan_files), ignore.case = TRUE)]
    }
    
    matching_models[[type]] <- stan_files
  }
  
  return(matching_models)
}

# Main execution
main <- function() {
  # Determine model types to compile
  model_types <- if (opt$type == "all") c("fit", "postpc", "prepc") else strsplit(opt$type, ",")[[1]]
  
  # Find matching models
  matching_models <- find_matching_models(model_types, opt$models)
  
  # Check if any models were found
  total_models <- sum(sapply(matching_models, length))
  if (total_models == 0) {
    stop("No matching models found.")
  }
  
  # If multiple models found, show them and check if we should compile all
  if (total_models > 1 && opt$models != "all") {
    cat("Multiple matching models found:\n")
    for (type in names(matching_models)) {
      for (model in matching_models[[type]]) {
        cat(paste0("- ", type, ": ", basename(model), "\n"))
      }
    }
    if (!opt$yes && !opt[["dry-run"]]) {
      cat("To compile these models, re-run with the --yes option.\n")
      return(invisible(NULL))
    } else if (!opt$yes && opt[["dry-run"]]) {
      cat("To compile these models, would require running with the --yes option. Otherwise, it will exit unless you use a more precise model str.\n")
      return(invisible(NULL))
    }
  }
  
  # Compile models or show what would be compiled
  for (type in names(matching_models)) {
    for (stan_file in matching_models[[type]]) {
      type_bin_dir <- file.path(MODELS_BIN_DIR, type)
      
      if (isTRUE(opt[["dry-run"]])) {
        cat(paste("Would compile:", basename(stan_file), "to", type_bin_dir, "\n"))
      } else {
        compile_stan_model(stan_file, type_bin_dir, verbose = opt$verbose)
      }
    }
  }
}

# Run the main function
main()