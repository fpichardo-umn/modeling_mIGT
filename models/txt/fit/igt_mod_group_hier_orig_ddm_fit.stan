functions {
  vector igt_model_lp(
    array[] int choice, array[] int shown, array[] real outcome,
    array[] real RT, int T, vector deck_means, vector deck_vars,
    vector sensitivity, real A, real update_pe,
    real lambda, real alpha, real inf_var,
    real update_var, real boundary, real tau, real beta
    ) {
    vector[T] drift_rates;
    vector[4] local_means = deck_means;
    vector[4] local_vars = deck_vars;
    array[T] int play_indices;
    array[T] int pass_indices;
    int play_count = 0;
    int pass_count = 0;
    real decay_factor = exp(-A);
    
    for (t in 1:T) {
      int curDeck = shown[t];
      
      // Compute drift rate
      drift_rates[t] = (local_means[curDeck] * (1 - abs(inf_var)) + sqrt(local_vars[curDeck]) * inf_var) * sensitivity[t];
      
      // Store indices for play and pass
      if (choice[t] == 1) {
        play_count += 1;
        play_indices[play_count] = t;
        local_explore[curDeck] = 0;
      } else {
        pass_count += 1;
        pass_indices[pass_count] = t;
      }
      
      // Compute utility
      real curUtil = pow(abs(outcome[t]), alpha) * (outcome[t] > 0 ? 1 : -1*lambda) * choice[t];
      
      // Decay-RL for all decks
      local_means *= decay_factor;

      // Update internal models
      curPE  = curUtil - local_means[curDeck];

      // Deck means
      local_means[curDeck] += curPE * (curUtil > 0 ? update_pe : 1 - update_pe) * choice[t];

      // Deck vars
      local_vars[curDeck] += sqrt((curPE**2 - local_vars[curDeck]) * update_var) * choice[t];
    }
    
    // Compute log probability for RTs/choice
    target += wiener_lpdf(RT[play_indices[:play_count]] | boundary, tau, beta, drift_rates[play_indices[:play_count]]);
    target += wiener_lpdf(RT[pass_indices[:pass_count]] | boundary, tau, 1-beta, -drift_rates[pass_indices[:pass_count]]);
    
    return append_row(local_means, local_vars);
  }
}

data {
  int<lower=1> 			    N;	      // Number of subjects
  int<lower=1> 			    T; 	      // Max overall number of trials
  array[N] int<lower=1> 			    Tsubj; // Number of trials for a subject
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
  real buffer = 0.001;  // 1 ms buffer
}

parameters {
  // Group hyper-parameters
  vector[10] 	     mu_pr;
  vector<lower=0>[10] sigma;

  // Subject-level raw parameters
  vector[N] boundary_pr;   // Boundary separation (a)
  vector[N] tau_pr;        // Non-decision time (tau)
  vector[N] beta_pr;       // Starting point
  vector[N] drift_con_pr;  // Drift consistency parameter
  vector[N] inf_var_pr;    // Uncertainty tolerance
  vector[N] lambda_pr;     // Loss aversion
  vector[N] alpha_pr;      // Attention weight for rewards
  vector[N] A_pr;          // Decay rate
  vector[N] update_pe_pr;  // Updating PE rate
  vector[N] update_var_pr; // Updating variance
}

transformed parameters {
  vector<lower=1e-6>[N] 		       boundary;
  vector<lower=RTbound - 1e-5, upper=RTmax>[N] tau;
  vector<lower=0, upper=1>[N] 	   	       beta;
  vector<lower=-2, upper=2>[N] 	   	       drift_con;
  vector<lower=-1, upper=1>[N]                 inf_var; // <0: unc averse; >0:unc driven
  vector<lower=0, upper=1>[N]                  lambda;
  vector<lower=-2, upper=2>[N]                 alpha;
  vector<lower=0, upper=3>[N]                  A; // Decay as high as only 5% of a trace remaining
  vector<lower=0, upper=1>[N]                  update_pe;
  vector<lower=0, upper=1>[N]                  update_var;
  
  boundary   = exp(inv_logit(mu_pr[1] + sigma[1]*boundary_pr) * 10 - 5);
  tau 	     = inv_logit(mu_pr[2] + sigma[2]*tau_pr) .* (minRTdiff - buffer) + RTbound;
  beta 	     = inv_logit(mu_pr[3] + sigma[3]*beta_pr);
  drift_con  = inv_logit(mu_pr[4] + sigma[4]*drift_con_pr) * 4 - 2;
  inf_var    = inv_logit(mu_pr[5] + sigma[5]*inf_var_pr) * 2 - 1;
  lambda     = inv_logit(mu_pr[6] + sigma[6]*lambda_pr);
  alpha      = inv_logit(mu_pr[7] + sigma[7]*alpha_pr) * 4 - 2;
  A          = inv_logit(mu_pr[8] + sigma[8]*A_pr) * 3;
  update_pe  = inv_logit(mu_pr[9] + sigma[9]*update_pe_pr);
  update_var = inv_logit(mu_pr[10] + sigma[10]*update_var_pr);
}

model {
  // Hyperparameters
  mu_pr ~ normal(0, 2.5);
  sigma ~ normal(0, 2);

  // Individual parameters
  boundary_pr   ~ normal(0, 2);
  tau_pr        ~ normal(0, 2);
  beta_pr       ~ normal(0, 2);
  drift_con_pr  ~ normal(0, 2);
  inf_var_pr    ~ normal(0, 2);
  lambda_pr     ~ normal(0, 2);
  alpha_pr      ~ normal(0, 2);
  A_pr          ~ normal(0, 2);
  update_pe_pr  ~ normal(0, 2);
  update_var_pr ~ normal(0, 2);


  // Initial subject-level deck expectations
  array[N] vector[4] deck_means;
  array[N] vector[4] deck_vars;

  // Begin with uninformative priors for each
  for (n in 1:N) {
    deck_means[n] = rep_vector(0., 4);
    deck_vars[n]  = rep_vector(1, 4);
  }

  // Initial trial data for theta
  vector[T] theta_ts = to_vector(linspaced_array(T, 1, T)) / 10.0;

  // For each subject
  for (n in 1:N) {
    vector[Tsubj[n]] sensitivity = pow(theta_ts[:Tsubj[n]], drift_con[n]);

    vector[8] result = igt_model_lp(
			choice[n][:Tsubj[n]], shown[n][:Tsubj[n]], outcome[n][:Tsubj[n]],
			RT[n][:Tsubj[n]], Tsubj[n], deck_means[n], deck_vars[n],
			sensitivity, A[n], update_pe[n],
			lambda[n], alpha[n], inf_var[n],
			update_var[n], boundary[n], tau[n], beta[n]
			);
  }
}

generated quantities {
  // Init
  real<lower=1e-6> 		   mu_boundary;
  real<lower=RTbound, upper=RTmax> mu_tau;
  real<lower=0, upper=1> 	   mu_beta;
  real<lower=-2, upper=2> 	   mu_drift_con;
  real<lower=-1, upper=1> 	   mu_inf_var;
  real<lower=0, upper=1>  	   mu_lambda;
  real<lower=-2, upper=2> 	   mu_alpha;
  real<lower=0, upper=3>  	   mu_A;
  real<lower=0, upper=1>  	   mu_update_pe;
  real<lower=0, upper=1>  	   mu_update_var;

  {
    // Pre-transformed mu
    vector[9] mu_transformed = inv_logit(mu_pr);

    // Compute interpretable group-level parameters
    mu_boundary   = exp(mu_transformed[1] * 10 - 5);
    mu_tau 	  = mu_transformed[2] * (mean(minRT) - RTbound) + RTbound;
    mu_beta 	  = mu_transformed[3];
    mu_drift_con  = mu_transformed[4] * 4 - 2;
    mu_inf_var    = mu_transformed[5] * 2 - 1;
    mu_lambda     = mu_transformed[6];
    mu_alpha      = mu_transformed[7] * 4 - 2;
    mu_A          = mu_transformed[8] * 3;
    mu_update_pe  = mu_transformed[9];
    mu_update_var = mu_transformed[10];
  }
}
