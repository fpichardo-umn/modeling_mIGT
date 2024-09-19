functions {
  vector igt_model_lp(
			array[] int choice, array[] int shown, array[] real outcome,
			vector ev, int Tsub, vector sensitivity,
			real update, real wgt_pun, real wgt_rew
			) {
    // Define values
    real curUtil;   // Current utility
    int  curDeck;   // Current deck
    vector[Tsub] Info;

    // Accumulation
    vector[4] local_ev = ev;

    // For each deck shown
    for (t in 1:Tsub) {
      // Deck presented to sub
      curDeck = shown[t];

      // EV and sensitivity
      Info[t] = sensitivity[t] * local_ev[curDeck];

      // Compute utility
      curUtil = ((outcome[t] > 0 ? wgt_rew : wgt_pun)) * outcome[t] * choice[t]; // choice 0, util 0

      // Update expected values
      local_ev[curDeck] += (curUtil - 2 * local_ev[curDeck]) * update * choice[t]; // choice 0, update 0
    }
    
    // Bernoulli distribution to decide whether to play the current deck or not
    target += bernoulli_logit_lpmf(choice | Info);
    
    return local_ev;
  }
}

data {
  int<lower=1> 			    N; 	      // Number of subjects
  int<lower=1> 			    T;        // Number of trials
  array[N] int<lower=1> 	    Tsubj;    // Number of trials for a subject
  array[N, T] int<lower=0, upper=1> choice;   // Binary choices made at each trial
  array[N, T] int<lower=0, upper=4> shown;    // Deck shown at each trial
  array[N, T] real 		    outcome;  // Outcome at each trial
  vector[4] 	    	    	    pr_mu;    // Informative priors
  vector[4] 	    	    	    pr_sigma; // Informative priors
}

parameters {
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
  
  con     = inv_logit(con_pr) * 4 - 2;
  wgt_pun = inv_logit(wgt_pun_pr);
  wgt_rew = inv_logit(wgt_rew_pr);
  update  = inv_logit(update_pr);
}

model {
  // Individual parameters
  con_pr     ~ normal(pr_mu[1], pr_sigma[1]);
  wgt_pun_pr ~ normal(pr_mu[2], pr_sigma[2]);
  wgt_rew_pr ~ normal(pr_mu[3], pr_sigma[3]);
  update_pr  ~ normal(pr_mu[4], pr_sigma[4]);

  // Initial subject-level deck expectations
  array[N] vector[4] ev;
  for (n in 1:N) {
    ev[n] = rep_vector(0., 4);
  }

  // Initial trial data for theta
  vector[T] theta_ts = to_vector(linspaced_array(T, 1, T)) / 10.0;

  // For each subject
  for (n in 1:N) {
    vector[Tsubj[n]] sensitivity = pow(theta_ts[:Tsubj[n]], con[n]);

    ev[n] = igt_model_lp(
			choice[n][:Tsubj[n]], shown[n][:Tsubj[n]], outcome[n][:Tsubj[n]],
			ev[n], Tsubj[n], sensitivity,
			update[n], wgt_pun[n], wgt_rew[n]
			);
  }
}

