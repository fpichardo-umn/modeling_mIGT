functions {
  vector igt_model_lp(
			array[] int choice, array[] int shown, array[] real outcome,
			int T, vector deck_means, vector deck_vars,
			vector sensitivity, real A, real update_pe, real lambda,
			real alpha, real inf_var, real update_var
			) {
    // Define values
    real curUtil; // Current utility
    int curDeck; // Current deck
    real curPE; // Prediction error 
    real curVar; //Current deck var
    vector[T] Info;
    vector[4] local_means  = deck_means;
    vector[4] local_vars = deck_vars;
    real decay_factor  = exp(-A);

    // For each deck shown
    for (t in 1:T) {
      // Deck presented to sub
      curDeck = shown[t];
      
      // Info integrated for choices
      Info[t] = (local_means[curDeck] * (1 - abs(inf_var)) + sqrt(local_stds[curDeck]) * inf_var) * sensitivity[t];

      // Compute utility
      curUtil = pow(abs(outcome[t]), alpha) * (outcome[t] > 0 ? 1 : -1*lambda) * choice[t];

      // Decay-RL
      local_means *= decay_factor;

      // Update internal models
      curPE = curUtil - local_means[curDeck];

      // Deck means
      local_means[curDeck] += curPE * (curUtil > 0 ? update_pe : 1 - update_pe) * choice[t];

      // Deck vars
      local_vars[curDeck] += (curPE**2 - local_vars[curDeck]) * update_var * choice[t];
    }
    // Bernoulli distribution to decide whether to gamble on the current deck or not
    // Use bernoulli_logit_lpmf for numerical stability
    target += bernoulli_logit_lpmf(choice | Info);
    
    return append_row(local_means, local_vars);
  }
}

data {
  int<lower=1> 			    N; 	      // Number of subjects
  int<lower=1> 			    T;        // Number of trials
  array[N] int<lower=1> 	    Tsubj;    // Number of trials for a subject
  array[N, T] int<lower=0, upper=1> choice;   // Binary choices made at each trial
  array[N, T] int<lower=0, upper=4> shown;    // Deck shown at each trial
  array[N, T] real 		    outcome;  // Outcome at each trial
}

parameters {
  // Group hyper-parameters
  vector[7] 	     mu_pr;
  vector<lower=0>[7] sigma;

  // Subject-level raw parameters
  vector[N] con_pr;        // Consistency parameter
  vector[N] inf_var_pr;    // Uncertainty tolerance
  vector[N] lambda_pr;     // Loss aversion
  vector[N] alpha_pr;      // Attention weight for rewards
  vector[N] A_pr;          // Decay rate
  vector[N] update_pe_pr;  // Updating PE rate
  vector[N] update_var_pr; // Updating variance
}

transformed parameters {
  // Transform subject-level raw parameters
  vector<lower=-2, upper=2>[N] con;
  vector<lower=-1, upper=1>[N] inf_var; // <0: uncertainty averse; >0: uncertainty driven
  vector<lower=0, upper=1>[N]  lambda;
  vector<lower=-2, upper=2>[N] alpha;
  vector<lower=0, upper=3>[N]  A; // Decay as high as only 5% of a trace remaining
  vector<lower=0, upper=1>[N]  update_pe;
  vector<lower=0, upper=1>[N]  update_var;
  
  con        = inv_logit(mu_pr[1] + sigma[1]*con_pr) * 4 - 2;
  inf_var    = inv_logit(mu_pr[2] + sigma[2]*inf_var_pr) * 2 - 1;
  lambda     = inv_logit(mu_pr[3] + sigma[3]*lambda_pr);
  alpha      = inv_logit(mu_pr[4] + sigma[4]*alpha_pr) * 4 - 2;
  A          = inv_logit(mu_pr[5] + sigma[5]*A_pr) * 3;
  update_pe  = inv_logit(mu_pr[6] + sigma[6]*update_pe_pr);
  update_var = inv_logit(mu_pr[7] + sigma[7]*update_var_pr);
}

model {
  // Hyperparameters
  mu_pr ~ normal(0, 2.5);
  sigma ~ normal(0, 2);

  // Individual parameters
  con_pr        ~ normal(0, 2);
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
    deck_vars[n] = rep_vector(1, 4);
  }

  // Initial trial data for theta
  vector[T] theta_ts = to_vector(linspaced_array(T, 1, T)) / 10.0;

  // For each subject
  for (n in 1:N) {
    vector[Tsubj[n]] sensitivity = pow(theta_ts[:Tsubj[n]], con[n]);

    vector[8] result = igt_model_lp(
			choice[n][:Tsubj[n]], shown[n][:Tsubj[n]], outcome[n][:Tsubj[n]],
			Tsubj[n], deck_means[n], deck_vars[n],
			sensitivity, A[n], update_pe[n], lambda[n],
			alpha[n], inf_var[n], update_var[n]
			);
  }
}

generated quantities {
  // Init
  real<lower=-2, upper=2> mu_con;
  real<lower=-1, upper=1> mu_inf_var;
  real<lower=0, upper=1>  mu_lambda;
  real<lower=-2, upper=2> mu_alpha;
  real<lower=0, upper=3>  mu_A;
  real<lower=0, upper=1>  mu_update_pe;
  real<lower=0, upper=1>  mu_update_var;

  {
    // Pre-transformed mu
    vector[6] mu_transformed = inv_logit(mu_pr);

    // Compute interpretable group-level parameters
    mu_con        = mu_transformed[1] * 4 - 2;
    mu_inf_var    = mu_transformed[2] * 2 - 1;
    mu_lambda     = mu_transformed[3];
    mu_alpha      = mu_transformed[4] * 4 - 2;
    mu_A          = mu_transformed[5] * 3;
    mu_update_pe  = mu_transformed[6];
    mu_update_var = mu_transformed[7];
  }
}
