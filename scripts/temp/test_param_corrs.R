# Load necessary library
library(posterior)
library(corrplot)

# Extract the draws
draws <- as_draws_matrix(model_fit_obj$draws)

# Define the number of subjects
n_subjects <- 500

# Initialize a matrix to store the parameter values
mean_params_matrix <- matrix(nrow = n_subjects, ncol = 4)
colnames(mean_params_matrix) <- c("wgt_pun", "wgt_rew", "con", "update")

# Extract parameter values for each subject
for (subject in 1:n_subjects) {
  mean_params_matrix[subject, 1] <- mean(draws[, paste0("wgt_pun[", subject, "]")])
  mean_params_matrix[subject, 2] <- mean(draws[, paste0("wgt_rew[", subject, "]")])
  mean_params_matrix[subject, 3] <- mean(draws[, paste0("con[", subject, "]")])
  mean_params_matrix[subject, 4] <- mean(draws[, paste0("update[", subject, "]")])
}

# Calculate correlations
mean_correlations <- cor(mean_params_matrix)

# Print correlations
print(round(mean_correlations, 3))

# Optional: Create a correlation plot
corrplot(mean_correlations, method = "color", type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45)


# Define the number of subjects
n_subjects <- 500

# Initialize a matrix to store the parameter values
mdn_params_matrix <- matrix(nrow = n_subjects, ncol = 4)
colnames(mdn_params_matrix) <- c("wgt_pun", "wgt_rew", "con", "update")

# Extract parameter values for each subject
for (subject in 1:n_subjects) {
  mdn_params_matrix[subject, 1] <- median(draws[, paste0("wgt_pun[", subject, "]")])
  mdn_params_matrix[subject, 2] <- median(draws[, paste0("wgt_rew[", subject, "]")])
  mdn_params_matrix[subject, 3] <- median(draws[, paste0("con[", subject, "]")])
  mdn_params_matrix[subject, 4] <- median(draws[, paste0("update[", subject, "]")])
}

# Calculate correlations
mdn_correlations <- cor(mdn_params_matrix)

# Print correlations
print(round(mdn_correlations, 3))

# Optional: Create a correlation plot
corrplot(correlations, method = "color", type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45)



# Initialize matrices to store the parameter values
mean_params_matrix <- matrix(nrow = n_subjects, ncol = 4)
mdn_params_matrix <- matrix(nrow = n_subjects, ncol = 4)
colnames(mean_params_matrix) <- colnames(mdn_params_matrix) <- c("wgt_pun", "wgt_rew", "con", "update")

# Extract parameter values for each subject
for (subject in 1:n_subjects) {
  for (param in c("wgt_pun", "wgt_rew", "con", "update")) {
    col <- draws[, paste0(param, "[", subject, "]")]
    mean_params_matrix[subject, param] <- mean(col)
    mdn_params_matrix[subject, param] <- median(col)
  }
}


# Function to format summary statistics
format_summary <- function(x, digits = 3) {
  sprintf("Mean: %.*f, Median: %.*f, SD: %.*f\nMin: %.*f, Q1: %.*f, Q3: %.*f, Max: %.*f",
          digits, mean(x), digits, median(x), digits, sd(x),
          digits, min(x), digits, quantile(x, 0.25), digits, quantile(x, 0.75), digits, max(x))
}

# Subject-level summaries
cat("Subject-Level Parameter Summaries:\n")
cat("==================================\n\n")

for (param in c("wgt_pun", "wgt_rew", "con", "update")) {
  cat(paste0(param, " (based on means):\n"))
  cat(format_summary(mean_params_matrix[,param]), "\n\n")
  
  cat(paste0(param, " (based on medians):\n"))
  cat(format_summary(mdn_params_matrix[,param]), "\n\n")
}

# Group-level summaries
cat("Group-Level Parameter Summaries:\n")
cat("================================\n\n")

group_params <- c("mu_con", "mu_update", "mu_wgt_pun", "mu_wgt_rew")
for (param in group_params) {
  cat(paste0(param, ":\n"))
  summary_data <- summary(draws[, param])
  cat(sprintf("Mean: %.3f, Median: %.3f, SD: %.3f\n", 
              summary_data$mean, summary_data$median, summary_data$sd))
  cat(sprintf("Q5: %.3f, Q95: %.3f\n", summary_data$q5, summary_data$q95))
  cat(sprintf("Rhat: %.3f, ESS bulk: %.0f, ESS tail: %.0f\n\n",
              summary_data$rhat, summary_data$ess_bulk, summary_data$ess_tail))
}
