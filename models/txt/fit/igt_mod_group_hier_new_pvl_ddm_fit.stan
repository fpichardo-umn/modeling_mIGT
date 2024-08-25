functions {
  vector igt_model_lp(
			int[] choice, int[] shown, real[] outcome,
			real[] RT, int T, vector exploit, vector explore,
			vector sensitivity, real A, real update_pe,
			real lambda, real alpha, real explore_upd,
			real exp_max, real boundary, real tau, real beta
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
  int<lower=1> 			    N;	      // Number of subjects
  int<lower=1> 			    T; 	      // Max overall number of trials
  int<lower=1>	 		    Tsubj[N]; // Number of trials for a subject
  real<lower=0> 		    RTbound;  // Lower bound or RT across all (e.g., 0.1 second)
  array[N] real 		    minRT;    // Minimum RT for each sub
  array[N, T] real<lower=0> 	    RT;       // Reaction times
  array[N, T] int<lower=0, upper=1> choice;   // Binary choices made at each trial
  array[N, T] int<lower=0, upper=4> shown;    // Deck shown at each trial
  array[N, T] real 		    outcome;  // Outcome at each trial
}

transformed data{
  vector[N] minRTdiff = to_vector(minRT) - RTbound;
  real      RTmax     = max(minRT);
}

parameters {
  // Group hyper-parameters
  vector[9] 	     mu_pr;
  vector<lower=0>[9] sigma;

  // Subject-level raw parameters
  vector[N] boundary_pr;  // Boundary separation (a)
  vector[N] tau_pr;  // Non-decision time (tau)
  vector[N] beta_pr;  // Starting point
  vector[N] drift_con_pr; // Drift consistency parameter
  vector[N] lambda_pr; // Loss aversion
  vector[N] alpha_pr; // Attention weight for rewards
  vector[N] A_pr; // Decay rate
  vector[N] update_pe_pr; // Attend to underestimate: 0 -> only care to learn from overestimates
  vector[N] exp_max_pr; // Updating rate
}

transformed parameters {
  vector<lower=0>[N] 		        boundary;
  vector<lower=RTbound, upper=RTmax>[N] tau;
  vector<lower=0, upper=1>[N] 	   	beta;
  vector<lower=-2, upper=2>[N] 	   	drift_con;
  vector<lower=0, upper=1>[N]  	   	exp_upd;
  vector<lower=0, upper=1>[N]  	   	lambda;
  vector<lower=0, upper=2>[N]  	   	alpha;
  vector<lower=0, upper=1>[N]  	   	A;
  vector<lower=0, upper=1>[N]  	   	update_pe;
  vector<lower=0>[N]  	  	   	exp_max;
  
  boundary  = exp(mu_pr[1] + sigma[1]*boundary_pr);
  tau 	    = inv_logit(mu_pr[2] + sigma[2]*tau_pr) .* minRTdiff + RTbound;
  beta 	    = inv_logit(mu_pr[3] + sigma[3]*beta_pr);
  drift_con = inv_logit(mu_pr[4] + sigma[4]*drift_con_pr) * 4 - 2;
  exp_upd   = 1 - inv_logit(mu_pr[4] + sigma[4]*drift_con_pr);
  lambda    = inv_logit(mu_pr[5] + sigma[5]*lambda_pr);
  alpha     = inv_logit(mu_pr[6] + sigma[6]*alpha_pr);
  A         = inv_logit(mu_pr[7] + sigma[7]*A_pr);
  update_pe = inv_logit(mu_pr[8] + sigma[8]*update_pe_pr);
  exp_max   = exp(inv_logit(mu_pr[9] + sigma[9]*exp_max_pr) * 4);
}

model {
  // Hyperparameters
  mu_pr ~ normal(0, 5);
  sigma ~ normal(0, 5);

  // Individual parameters
  boundary_pr  ~ normal(0, 5);
  tau_pr       ~ normal(0, 5);
  beta_pr      ~ normal(0, 5);
  drift_con_pr ~ normal(0, 5);
  lambda_pr    ~ normal(0, 2);
  alpha_pr     ~ normal(0, 2);
  A_pr         ~ normal(0, 2);
  update_pe_pr ~ normal(0, 5);
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
    vector[Tsubj[n]] sensitivity = pow(theta_ts[:Tsubj[n]], drift_con[n]);

    vector[8] result = igt_model_lp(
			choice[n][:Tsubj[n]], shown[n][:Tsubj[n]], outcome[n][:Tsubj[n]],
			RT[n][:Tsubj[n]], Tsubj[n], exploit[n], explore[n],
			sensitivity, A[n], update_pe[n],
			-1*lambda[n], alpha[n], exp_upd[n],
			exp_max[n], boundary[n], tau[n], beta[n]
			);
  }
}

generated quantities {
  // Init
  real<lower=0> 		   mu_boundary;
  real<lower=RTbound, upper=RTmax> mu_tau;
  real<lower=0, upper=1> 	   mu_beta;
  real<lower=-2, upper=2> 	   mu_drift_con;
  real<lower=0, upper=1>  	   mu_exp_upd;
  real<lower=0, upper=1>  	   mu_lambda;
  real<lower=0, upper=1>  	   mu_alpha;
  real<lower=0, upper=1>  	   mu_A;
  real<lower=0, upper=1>  	   mu_update_pe;
  real<lower=0>           	   mu_exp_max;

  {
    // Pre-transformed mu
    vector[9] mu_transformed = inv_logit(mu_pr);

    // Compute interpretable group-level parameters
    mu_boundary  = exp(mu_transformed[1]);
    mu_tau 	 = inv_logit(mu_transformed[2]) * (mean(minRT) - RTbound) + RTbound;
    mu_beta 	 = inv_logit(mu_transformed[3]);
    mu_drift_con = mu_transformed[4] * 4 - 2;
    mu_exp_upd   = 1 - mu_transformed[4];
    mu_lambda    = mu_transformed[5];
    mu_alpha     = mu_transformed[6];
    mu_A         = mu_transformed[7];
    mu_update_pe = mu_transformed[8];
    mu_exp_max   = exp(mu_transformed[9] * 4);
  }
}
