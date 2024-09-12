functions {
  vector igt_model_lp(
			array[] int choice, array[] int shown, array[] real outcome,
			int T, vector exploit, vector explore,
			vector sensitivity, real A, real update_pe, real lambda,
			real alpha, real explore_upd, real exp_max
			) {
    // Define values
    real curUtil; // Current utility
    int curDeck; // Current deck
    vector[T] Info;
    vector[4] local_exploit = exploit;
    vector[4] local_explore = explore;
    real decay_factor = exp(-A);

    // For each deck shown
    for (t in 1:T) {
      // Deck presented to sub
      curDeck = shown[t];
      
      // EV and sensitivity
      Info[t] = (local_exploit[curDeck] + local_explore[curDeck]) * sensitivity[t];
      
      // Update exploration values
      local_explore += (exp_max - local_explore)*explore_upd;

      // Compute utility
      curUtil = pow(abs(outcome[t]), alpha) * (outcome[t] > 0 ? 1 : lambda) * choice[t];

      // Decay-RL
      local_exploit *= decay_factor;

      // Update expected values
      real pe = curUtil - local_exploit[curDeck];
      local_exploit[curDeck] += pe * (pe > 0 ? update_pe : 1 - update_pe) * choice[t];

      if (choice[t] == 1) {
        local_explore[curDeck] = 0;
      }
    }
    // Bernoulli distribution to decide whether to gamble on the current deck or not
    // Use bernoulli_logit_lpmf for numerical stability
    target += bernoulli_logit_lpmf(choice | Info);
    
    return append_row(local_exploit, local_explore);
  }
}

data {
  int<lower=1> 			    N; 	      // Number of subjects
  int<lower=1> 			    T;        // Number of trials
  array[N] int<lower=1> 			    Tsubj; // Number of trials for a subject
  array[N, T] int<lower=0, upper=1> choice;   // Binary choices made at each trial
  array[N, T] int<lower=0, upper=4> shown;    // Deck shown at each trial
  array[N, T] real 		    outcome;  // Outcome at each trial
}

parameters {
  // Group hyper-parameters
  vector[6] 	     mu_pr;
  vector<lower=0>[6] sigma;

  // Subject-level raw parameters
  vector[N] con_pr; // Consistency parameter
  vector[N] lambda_pr; // Loss aversion
  vector[N] alpha_pr; // Attention weight for rewards
  vector[N] A_pr; // Decay rate
  vector[N] update_pe_pr; // Updating PE rate
  vector[N] exp_max_pr; // Updating rate
}

transformed parameters {
  // Transform subject-level raw parameters
  vector<lower=-2, upper=2>[N] con;
  vector<lower=0, upper=1>[N]  exp_upd;
  vector<lower=0, upper=1>[N]  lambda;
  vector<lower=0, upper=2>[N]  alpha;
  vector<lower=0, upper=3>[N]  A; // Decay as high as only 5% of a trace remaining
  vector<lower=0, upper=1>[N]  update_pe;
  vector<lower=0>[N]           exp_max;
  
  con       = inv_logit(mu_pr[1] + sigma[1]*con_pr) * 4 - 2;
  exp_upd   = 1 - inv_logit(mu_pr[1] + sigma[1]*con_pr);
  lambda    = inv_logit(mu_pr[2] + sigma[2]*lambda_pr);
  alpha     = inv_logit(mu_pr[3] + sigma[3]*alpha_pr);
  A         = inv_logit(mu_pr[4] + sigma[4]*A_pr) * 3;
  update_pe = inv_logit(mu_pr[5] + sigma[5]*update_pe_pr);
  exp_max   = exp(inv_logit(mu_pr[6] + sigma[6]*exp_max_pr) * 4);
}

model {
  // Hyperparameters
  mu_pr ~ normal(0, 5);
  sigma ~ cauchy(0, 2.5);

  // Individual parameters
  con_pr       ~ normal(0, 2.5);
  lambda_pr    ~ normal(0, 2.5);
  alpha_pr     ~ normal(0, 2.5);
  A_pr         ~ normal(0, 2.5);
  update_pe_pr ~ normal(0, 2.5);
  exp_max_pr   ~ normal(0, 5);

  // Initial subject-level deck expectations
  array[N] vector[4] exploit;
  array[N] vector[4] explore;
  for (n in 1:N) {
    exploit[n] = rep_vector(0., 4);
    explore[n] = rep_vector(0., 4);
  }

  // Initial trial data for theta
  vector[T] theta_ts = to_vector(linspaced_array(T, 1, T)) / 10.0;

  // For each subject
  for (n in 1:N) {
    vector[Tsubj[n]] sensitivity = pow(theta_ts[:Tsubj[n]], con[n]);

    vector[8] result = igt_model_lp(
			choice[n][:Tsubj[n]], shown[n][:Tsubj[n]], outcome[n][:Tsubj[n]],
			Tsubj[n], exploit[n], explore[n],
			sensitivity, A[n], update_pe[n], -1*lambda[n],
			alpha[n], exp_upd[n], exp_max[n]
			);
  }
}

generated quantities {
  // Init
  real<lower=-2, upper=2> mu_con;
  real<lower=0, upper=1>  mu_exp_upd;
  real<lower=0, upper=1>  mu_lambda;
  real<lower=0, upper=1>  mu_alpha;
  real<lower=0, upper=1>  mu_A;
  real<lower=0, upper=1>  mu_update_pe;
  real<lower=0>           mu_exp_max;

  {
    // Pre-transformed mu
    vector[6] mu_transformed = inv_logit(mu_pr);

    // Compute interpretable group-level parameters
    mu_con       = mu_transformed[1] * 4 - 2;
    mu_exp_upd   = 1 - mu_transformed[1];
    mu_lambda    = mu_transformed[2];
    mu_alpha     = mu_transformed[3];
    mu_A         = mu_transformed[4];
    mu_update_pe = mu_transformed[5];
    mu_exp_max   = exp(mu_transformed[6] * 4);
  }
}
