functions {
  vector igt_model_rng(
			array[] int choice, array[] int shown, array[] real outcome,
			vector ev, int Tsub, vector sensitivity,
			real update, real wgt_pun, real wgt_rew
			) {
    // Define values
    real curUtil;   // Current utility
    real curInfo;   // Current Info
    int  curDeck;   // Current deck
    real EV2update; // Current EV to update

    // Outputs
    vector[4]    local_ev = ev;
    vector[Tsub] log_lik;
    vector[Tsub] choice_pred;

    // For each deck shown
    for (t in 1: Tsub) {
      // Deck presented to sub
      curDeck = shown[t];

      // Info to update
      EV2update = local_ev[curDeck];
      curInfo   = sensitivity[t] * EV2update;

      // Calculate log likelihood
      log_lik[t] = bernoulli_logit_lpmf(choice[t] | curInfo);

      // Generate prediction
      choice_pred[t] = bernoulli_rng(inv_logit(curInfo));

      // Compute utility
      curUtil = ((outcome[t] > 0 ? wgt_rew : wgt_pun)) * outcome[t] * choice[t];

      // Update expected values
      local_ev[curDeck] += (curUtil - 2 * EV2update) * update * choice[t];
    }
    return append_row(append_row(local_ev, log_lik), choice_pred);
  }
}

data {
  int<lower=1> 			    N; 	      // Number of subjects
  int<lower=1> 			    T;        // Number of trials
  int<lower=1> 			    Tsubj[N]; // Number of trials for a subject
  array[N, T] int<lower=0, upper=1> choice;   // Binary choices made at each trial
  array[N, T] int<lower=0, upper=4> shown;    // Deck shown at each trial
  array[N, T] real 		    outcome;  // Outcome at each trial
}

parameters {
  // Group hyper-parameters
  vector[4] 	     mu_pr;
  vector<lower=0>[4] sigma;

  // Subject-level raw parameters
  vector[N] con_pr; 	// Consistency parameter
  vector[N] wgt_pun_pr; // Attention weight for punishments
  vector[N] wgt_rew_pr; // Attention weight for rewards
  vector[N] update_pr;  // Updating rate
}

transformed parameters {
  // Transform subject-level raw parameters
  vector<lower=-2, upper=2>[N] con;
  vector<lower=0, upper=1>[N]  wgt_pun;
  vector<lower=0, upper=1>[N]  wgt_rew;
  vector<lower=0, upper=1>[N]  update;
  
  con     = inv_logit(mu_pr[1] + sigma[1]*con_pr) * 4 - 2;
  wgt_pun = inv_logit(mu_pr[2] + sigma[2]*wgt_pun_pr);
  wgt_rew = inv_logit(mu_pr[3] + sigma[3]*wgt_rew_pr);
  update  = inv_logit(mu_pr[4] + sigma[4]*update_pr);
}

generated quantities {
  // For group level parameters
  real<lower=-2, upper=2> mu_con;
  real<lower=0, upper=1>  mu_wgt_pun;
  real<lower=0, upper=1>  mu_wgt_rew;
  real<lower=0, upper=1>  mu_update;

  array[N, T] int<lower=0, upper=1> choice_pred;
  array[N] vector[4] 		    final_ev;
  real 				    total_log_lik;
  
  {
    // Init local variables
    vector[N] log_lik;

    // Initial subject-level deck expectations
    array[N] vector[4] ev;
    for (n in 1:N) {
      ev[n] = rep_vector(0., 4);
    }

    // Initial trial data for theta
    vector[T] main_theta_ts = to_vector(linspaced_array(T, 1, T)) / 10.0;

    total_log_lik = 0;
    
    for (n in 1:N) {
      vector[Tsubj[n]] sensitivity = pow(main_theta_ts[:Tsubj[n]], con);
      
      vector[4 + 2*Tsubj[n]] results = igt_model_rng(
				choice[n][:Tsubj[n]], shown[n][:Tsubj[n]], outcome[n][:Tsubj[n]],
				ev[n], Tsubj[n], sensitivity,
				update[n], wgt_pun[n], wgt_rew[n]
				);
      
      final_ev[n] 	        = results[1:4];
      log_lik[n]  	        = sum(results[5:(4+Tsubj[n])]);
      choice_pred[n][:Tsubj[n]] = to_int(to_array_1d(results[(5+Tsubj[n]):(4+2*Tsubj[n])]));
      
      total_log_lik += log_lik[n];
    }
  }
}
