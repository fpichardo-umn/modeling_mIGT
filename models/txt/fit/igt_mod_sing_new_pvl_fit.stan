functions {
  vector igt_model_lp(
			int[] choice, int[] shown, real[] outcome,
			int T, vector exploit, vector explore,
			vector sensitivity, real A, real update_pe, real lambda,
			real alpha, real explore_upd, real exp_max
			) {
    // Define values
    real curUtil; // Current utility
    int curDeck; // Current deck
    real curInfo;
    vector[4] local_exploit = exploit;
    vector[4] local_explore = explore;

    // For each deck shown
    for (t in 1: T) {
      // Deck presented to sub
      curDeck = shown[t];

      // Bernoulli distribution to decide whether to gamble on the current deck or not
      // Use bernoulli_logit_lpmf for numerical stability
      curInfo = (local_exploit[curDeck] + local_explore[curDeck]) * sensitivity[t];
      target += bernoulli_logit_lpmf(choice[t] | curInfo);

      // Compute utility
      curUtil = pow(abs(outcome[t]), alpha) * (outcome[t] > 0 ? 1 : lambda) * choice[t];

      // Decay-RL
      local_exploit *= pow(e(), -A);

      // Update expected values
      real pe = curUtil - local_exploit[curDeck];
      local_exploit[curDeck] += pe * (pe > 0 ? update_pe : 1 - update_pe) * choice[t];

      // Update exploration values
      local_explore += (exp_max - local_explore)*explore_upd;

      if (choice[t] == 1) {
        local_explore[curDeck] = 0;
      }
    }
    return append_row(local_exploit, local_explore);
  }
}

data {
  int<lower=1> T; // Number of trials
  array[T] int<lower=0, upper=1> choice; // Binary choices made at each trial
  array[T] int<lower=0, upper=4> shown; // Deck shown at each trial
  array[T] real outcome; // Outcome at each trial
}

parameters {
  // Subject-level raw parameters
  real con_pr; // Consistency parameter
  real lambda_pr; // Loss aversion
  real alpha_pr; // Attention weight for rewards
  real A_pr; // Decay rate
  real update_pe_pr; // Updating PE rate
  real exp_max_pr; // Max exploration rate
}

transformed parameters {
  // Transform subject-level raw parameters
  real<lower=-2, upper=2> con;
  real<lower=0, upper=1>  exp_upd;
  real<lower=0, upper=1>  lambda;
  real<lower=0, upper=2>  alpha;
  real<lower=0, upper=1>  A;
  real<lower=0, upper=1>  update_pe;
  real<lower=0>  	  exp_max;
  
  con       = inv_logit(con_pr) * 4 - 2;
  exp_upd   = 1 - inv_logit(con_pr); // Related to the inverse of the consistency
  lambda    = inv_logit(lambda_pr);
  alpha     = inv_logit(alpha_pr);
  A         = inv_logit(A_pr);
  update_pe = inv_logit(update_pe_pr);
  exp_max   = exp(inv_logit(exp_max_pr) * 4);
}

model {
  // Individual parameters
  con_pr       ~ normal(0, 1);
  lambda_pr    ~ normal(0, 1);
  alpha_pr     ~ normal(0, 1);
  A_pr         ~ normal(0, 1);
  update_pe_pr ~ normal(0, 1);
  exp_max_pr   ~ normal(0, 1);

  // Initial subject-level deck expectations
  vector[4] exploit = rep_vector(0., 4);
  vector[4] explore = rep_vector(0, 4);

  // Dynamic sensitivity
  vector[T] sensitivity = pow(to_vector(linspaced_array(T, 1, T)) / 10.0, con);

  vector[8] result = igt_model_lp(
			choice, shown, outcome,
			T, exploit, explore,
			sensitivity, A, update_pe, -1*lambda,
			alpha, exp_upd, exp_max
			);

  // final_exploit = result[1:4];
  // final_explore = result[5:8];
}
