functions {
  // Fitting EV DDM model
  vector igt_model_lp(
    array[] int choice, array[] int shown, array[] real outcome,
    array[] real RT, vector ev, int Tsub,
    vector sensitivity, real update, real wgt_pun,
    real wgt_rew, real boundary, real tau, real beta
    ) {
    vector[4] local_ev = ev;
    vector[Tsub] drift_rates;
    array[Tsub] int play_indices;
    array[Tsub] int pass_indices;
    int play_count = 0;
    int pass_count = 0;
    
    // Compute drift rates and update local_ev
    for (t in 1:Tsub) {
      int curDeck = shown[t];
      drift_rates[t] = local_ev[curDeck] * sensitivity[t];
      
      // Update local_ev
      real curUtil = ((outcome[t] > 0 ? wgt_rew : wgt_pun)) * outcome[t] * choice[t];
      local_ev[curDeck] += (curUtil - 2 * local_ev[curDeck]) * update * choice[t];
      
      // Store indices for play and pass
      if (choice[t] == 1) {
        play_count += 1;
        play_indices[play_count] = t;
      } else {
        pass_count += 1;
        pass_indices[pass_count] = t;
      }
    }
    
    // Compute log probability for RTs/choice
    target += wiener_lpdf(RT[play_indices[:play_count]] | boundary, tau, beta, drift_rates[play_indices[:play_count]]);
    target += wiener_lpdf(RT[pass_indices[:pass_count]] | boundary, tau, 1-beta, -drift_rates[pass_indices[:pass_count]]);
    
    return local_ev;
  }
}

data {
  int<lower=1> 			    N;	      // Number of subjects
  int<lower=1> 			    T; 	      // Max overall number of trials
  array[N] int<lower=1> 	    Tsubj;    // Number of trials for a subject
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
  // Hyper-parameters
  vector[7] 	     mu_pr;
  vector<lower=0>[7] sigma;

  // Subject-level raw parameters
  vector[N] boundary_pr;  // Boundary separation (a)
  vector[N] tau_pr; 	  // Non-decision time (tau)
  vector[N] beta_pr;	  // Starting point
  vector[N] drift_con_pr; // Drift consistency parameter
  vector[N] wgt_pun_pr;   // Attention weight for punishments
  vector[N] wgt_rew_pr;   // Attention weight for rewards
  vector[N] update_pr;    // Updating rate
}

transformed parameters {
  vector<lower=1e-6>[N] 			boundary;
  vector<lower=RTbound - 1e-5, upper=RTmax>[N] tau;
  vector<lower=0, upper=1>[N] 		beta;
  vector<lower=-2, upper=2>[N] 		drift_con;
  vector<lower=0, upper=1>[N] 		wgt_pun;
  vector<lower=0, upper=1>[N] 		wgt_rew;
  vector<lower=0, upper=1>[N] 		update;

  boundary  = exp(inv_logit(mu_pr[1] + sigma[1]*boundary_pr) * 5 - 2); // Range: -2,3 -> 0.135,20.08
  tau       = inv_logit(mu_pr[2] + sigma[2] * tau_pr) .* (minRTdiff - buffer) + RTbound;
  beta      = inv_logit(to_vector(mu_pr[3] + sigma[3] * beta_pr));
  drift_con = inv_logit(to_vector(mu_pr[4] + sigma[4] * drift_con_pr)) * 4 - 2;
  wgt_pun   = inv_logit(to_vector(mu_pr[5] + sigma[5] * wgt_pun_pr));
  wgt_rew   = inv_logit(to_vector(mu_pr[6] + sigma[6] * wgt_rew_pr));
  update    = inv_logit(to_vector(mu_pr[7] + sigma[7] * update_pr));
}

model {
  // Hyperparameters
  to_vector(mu_pr) ~ normal(0, 2.5);
  to_vector(sigma) ~ normal(0, 2);

  // Priors
  to_vector(boundary_pr)  ~ normal(0, 2);
  to_vector(tau_pr)       ~ normal(0, 2);
  to_vector(beta_pr)      ~ normal(0, 2);
  to_vector(drift_con_pr) ~ normal(0, 2);
  to_vector(wgt_pun_pr)   ~ normal(0, 2);
  to_vector(wgt_rew_pr)   ~ normal(0, 2);
  to_vector(update_pr)    ~ normal(0, 2);

  // Initial subject-level deck expectations
  array[N] vector[4] ev;
  for (n in 1:N) {
    ev[n] = rep_vector(0., 4);
  }

  // Initial trial data for theta
  vector[T] theta_ts = to_vector(linspaced_array(T, 1, T)) / 10.0;

  // For each subject
  for (n in 1:N) {
    vector[Tsubj[n]] sensitivity = pow(theta_ts[:Tsubj[n]], drift_con[n]);

    ev[n] = igt_model_lp(
			choice[n][:Tsubj[n]], shown[n][:Tsubj[n]], outcome[n][:Tsubj[n]],
			RT[n][:Tsubj[n]], ev[n], Tsubj[n],
			sensitivity, update[n], wgt_pun[n],
			wgt_rew[n], boundary[n], tau[n], beta[n]
			);
  }
}

generated quantities {
  // Init
  real<lower=RTbound, upper=RTmax> mu_tau;
  real<lower=1e-6>  		   mu_boundary;
  real<lower=0, upper=1> 	   mu_beta;
  real<lower=-2, upper=2> 	   mu_drift_con;
  real<lower=0, upper=1> 	   mu_wgt_pun;
  real<lower=0, upper=1> 	   mu_wgt_rew;
  real<lower=0, upper=1> 	   mu_update;

  {
    real RTlowerbound = (mean(minRT) - RTbound) + RTbound;

    // Pre-transformed mu
    vector[7] mu_transformed = inv_logit(mu_pr);

    // Compute interpretable group-level parameters
    mu_boundary = exp(mu_transformed[1] * 10 - 5);
    mu_tau 	 = mu_transformed[2] * RTlowerbound;
    mu_beta	 = mu_transformed[3];
    mu_drift_con = mu_transformed[4] * 4 - 2;
    mu_wgt_pun   = mu_transformed[5];
    mu_wgt_rew   = mu_transformed[6];
    mu_update    = mu_transformed[7];
  }
}
