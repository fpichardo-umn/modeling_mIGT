# Load required libraries
suppressPackageStartupMessages({
  library(here)
  library(foreign)
  library(dplyr) #for %>%
  library(tidyr)
  library(plyr)   #for revalue
  library(ggplot2)
  library(gridExtra)
  library(grid)   #for textGrob
  library(kableExtra) #for knitr tables
  library(caret)
})

# Directories
PROJ_DIR            = file.path(here::here())
DATA_DIR            = file.path(PROJ_DIR, "Data")
SAFE_DATA_DIR       = file.path(DATA_DIR, "AHRB")
MODELS_DIR          = file.path(PROJ_DIR, "models")
MODELS_RDS_DIR      = file.path(MODELS_DIR, "rds")
MODELS_RDS_FIT_DIR  = file.path(MODELS_RDS_DIR, "fit")
MODELS_RDS_PoPC_DIR = file.path(MODELS_RDS_DIR, "postpc")


# Load Data
## Trial-level data
wave1.sav.file = file.path(SAFE_DATA_DIR, "modigt_data_Wave1.sav")
risk.sav.file  = file.path(SAFE_DATA_DIR, "AHRB.P1W1_v11_AW_v2.sav")

wave1.igt.raw   = read.spss(wave1.sav.file,to.data.frame = TRUE)
wave1.risk.data = read.spss(risk.sav.file,to.data.frame = TRUE)

# Subs
igt.subs  = unique(wave1.igt.raw$sid)
risk.subs = unique(wave1.risk.data$sid)

shared_subs = intersect(igt.subs, risk.subs)

wave1.risk.shared.df = wave1.risk.data[wave1.risk.data$sid %in% shared_subs,]

sid_to_grpid = unique(wave1.igt.raw[, c("sid", "grpid")])
wave1.risk.shared.df$grpid = NA  # Initialize the column with NA
for (sid in shared_subs) {
  wave1.risk.shared.df$grpid[wave1.risk.shared.df$sid == sid] =  sid_to_grpid$grpid[sid_to_grpid$sid == sid]
}

# Sample for EFA
print(length(shared_subs))
print(table(wave1.risk.shared.df$grpid))
print(table(wave1.risk.shared.df$grade10))
print(table(wave1.risk.shared.df$race_ethnicity))
print(table(wave1.risk.shared.df$male))
print(table(wave1.risk.shared.df$parent_edu4))

## Set seed: 1645689915
set.seed(1645689915)


# Function to create balanced strata
create_balanced_strata <- function(data, min_size = 10) {
  data %>%
    dplyr::group_by(grpid) %>%
    dplyr::mutate(strata = dplyr::case_when(
      dplyr::n() >= min_size ~ as.character(grpid),
      TRUE ~ "Other"
    )) %>%
    dplyr::group_by(strata, grade10) %>%
    dplyr::mutate(strata = paste(strata, grade10, sep = "_")) %>%
    dplyr::group_by(strata, male) %>%
    dplyr::mutate(strata = paste(strata, male, sep = "_")) %>%
    dplyr::group_by(strata, race_ethnicity) %>%
    dplyr::mutate(strata = paste(strata, race_ethnicity, sep = "_")) %>%
    dplyr::group_by(strata, parent_edu4) %>%
    dplyr::mutate(strata = paste(strata, parent_edu4, sep = "_")) %>%
    dplyr::ungroup()
}

# Create stratified data
stratified_data <- create_balanced_strata(wave1.risk.shared.df)

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

# Split the full data
split_data <- stratified_split(stratified_data, c(0.6, 0.2, 0.2))

# Create subsets (including all variables)
training_data <- dplyr::filter(split_data, split == 1)
validation_data <- dplyr::filter(split_data, split == 2)
testing_data <- dplyr::filter(split_data, split == 3)

# Check sizes
print(nrow(training_data))
print(nrow(validation_data))
print(nrow(testing_data))

# Function to calculate proportions
calc_props <- function(data) {
  list(
    grp = prop.table(table(data$grpid)),
    grade = prop.table(table(data$grade10)),
    gender = prop.table(table(data$male)),
    race = prop.table(table(data$race_ethnicity)),
    parent_edu = prop.table(table(data$parent_edu4))
  )
}

# Compare proportions
full_props <- calc_props(wave1.risk.shared.df)
train_props <- calc_props(training_data)
valid_props <- calc_props(validation_data)
test_props <- calc_props(testing_data)

# Print proportions
for (var in c("grp", "grade", "gender", "race", "parent_edu")) {
  cat("\n", var, ":\n")
  print(full_props[[var]])
  print(train_props[[var]])
  print(valid_props[[var]])
  print(test_props[[var]])
}

# Save validation_data as a CSV file in the specified directory
write.csv(validation_data, file = file.path(SAFE_DATA_DIR, "efa_sample.csv"), row.names = FALSE)
