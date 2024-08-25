functions {
  vector igt_model_lp(
			int[] choice, int[] shown, real[] outcome,
			array[] real RT, int T, vector exploit, vector explore,
			vector sensitivity, real A, real update_pe, real lambda,
			real alpha, real explore_upd, real exp_max,
			real boundary, real tau, real beta
			){
    // Define values
    real curUtil; // Current utility
    int curDeck; // Current deck
    real curDrift; // Current drift
    vector[4] local_exploit = exploit;
    vector[4] local_explore = explore;

    for (t in 1:T){
      // Deck presented to sub
      curDeck = shown[t];

      // Drift diffusion process
      curDrift = (local_exploit[curDeck] + local_explore[curDeck]) * sensitivity[t]; // Drift scaling

      // Update exploration values
      local_explore += (exp_max - local_explore)*explore_upd;

      // Model both RT and choice
      if (choice[t] == 1) {
        target += wiener_lpdf(RT[t] | boundary, tau, beta, curDrift);
        local_explore[curDeck] = 0;
      } else {
        target += wiener_lpdf(RT[t] | boundary, tau, 1-beta, -curDrift);
      }

      target += bernoulli_logit_lpmf(choice[t] | curDrift);

      // Compute utility
      curUtil = pow(abs(outcome[t]), alpha) * (outcome[t] > 0 ? 1 : lambda) * choice[t];

      // Decay-RL
      local_exploit *= pow(e(), -A);

      // Update expected values
      real pe = curUtil - local_exploit[curDeck];
      local_exploit[curDeck] += pe * (pe > 0 ? update_pe : 1 - update_pe) * choice[t];
    }
    return append_row(local_exploit, local_explore);
  }
}

data {
  int<lower=1> T; // Number of trials
  real<lower=0> minRT;  // Minimum RT + small value to restrict tau
  real RTbound; // Lower bound or RT across all subjects (e.g., 0.1 second)
  array[T] real<lower=0> RT;  // Reaction times
  array[T] int<lower=0, upper=1> choice; // Binary choices made at each trial
  array[T] int<lower=0, upper=4> shown; // Deck shown at each trial
  array[T] real outcome; // Outcome at each trial
}

parameters {
  real boundary_pr;  // Boundary separation (a)
  real tau_pr;  // Non-decision time (tau)
  real beta_pr;  // Starting point
  real drift_con_pr; // Drift consistency parameter
  real lambda_pr; // Loss aversion
  real alpha_pr; // Attention weight for rewards
  real A_pr; // Decay rate
  real update_pe_pr; // Updating PE rate
  real exp_max_pr; // Max exploration rate
}

transformed parameters {
  real<lower=0> 		   boundary;
  real<lower=RTbound, upper=minRT> tau;
  real<lower=0, upper=1> 	   beta;
  real<lower=-2, upper=2> 	   drift_con;
  real<lower=0, upper=1>  	   exp_upd;
  real<lower=0, upper=1>  	   lambda;
  real<lower=0, upper=2>  	   alpha;
  real<lower=0, upper=1>  	   A;
  real<lower=0, upper=1>  	   update_pe;
  real<lower=0>  	  	   exp_max;

  boundary  = exp(boundary_pr);
  tau 	    = inv_logit(tau_pr) * (minRT - RTbound) + RTbound;
  beta 	    = inv_logit(beta_pr);
  drift_con = inv_logit(drift_con_pr) * 4 - 2;
  exp_upd   = 1 - inv_logit(drift_con_pr); // Related to the inverse of the consistency
  lambda    = inv_logit(lambda_pr);
  alpha     = inv_logit(alpha_pr);
  A         = inv_logit(A_pr);
  update_pe = inv_logit(update_pe_pr);
  exp_max   = exp(inv_logit(exp_max_pr) * 4);
}

model {
  // Priors
  boundary_pr  ~ normal(0, 1);
  tau_pr       ~ normal(0, 1);
  beta_pr      ~ normal(0, 1);
  drift_con_pr ~ normal(0, 1);
  lambda_pr    ~ normal(0, 1);
  alpha_pr     ~ normal(0, 1);
  A_pr         ~ normal(0, 1);
  update_pe_pr ~ normal(0, 1);
  exp_max_pr   ~ normal(0, 1);

  // Initial subject-level deck variables
  vector[4] exploit = rep_vector(0., 4);
  vector[4] explore = rep_vector(0, 4);

  vector[T] theta_ts = to_vector(linspaced_array(T, 1, T)) / 10.0;

  {
    // Initial trial data for theta
    vector[T] sensitivity = pow(theta_ts, drift_con);
    vector[8] result = igt_model_lp(
			choice, shown, outcome,
			RT, T, exploit, explore,
			sensitivity, A, update_pe, -1*lambda,
			alpha, exp_upd, exp_max,
			boundary, tau, beta
			);
    
  }
}
