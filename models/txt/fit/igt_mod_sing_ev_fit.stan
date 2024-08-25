functions {
  vector igt_model_lp(
			int[] choice, int[] shown, real[] outcome,
			int T, vector ev, vector sensitivity, 
			real update, real wgt_pun, real wgt_rew
			) {
    // Define values
    real curUtil; // Current utility
    int curDeck; // Current deck
    real EV2update;  // Current EV to update
    vector[4] local_ev = ev;

    // For each deck shown
    for (t in 1: T) {
      // Deck presented to sub
      curDeck = shown[t];

      // EV to update
      EV2update = local_ev[curDeck];

      // Bernoulli distribution to decide whether to gamble on the current deck or not
      // Use bernoulli_logit_lpmf for numerical stability
      target += bernoulli_logit_lpmf(choice[t] | sensitivity[t] * EV2update);

      // Compute utility
      curUtil = (outcome[t] * (outcome[t] > 0 ? wgt_rew : wgt_pun)) * choice[t];

      // Update expected values
      local_ev[curDeck] += (curUtil - 2 * EV2update) * update * choice[t];
    }
    return local_ev;
  }
}

data {
  int<lower=1> 			 T; 	  // Number of trials
  array[T] int<lower=0, upper=1> choice;  // Binary choices made at each trial
  array[T] int<lower=0, upper=4> shown;   // Deck shown at each trial
  array[T] real 		 outcome; // Outcome at each trial
}


parameters {
  // Subject-level raw parameters
  real con_pr; 	   // Consistency parameter
  real wgt_pun_pr; // Attention weight for punishments
  real wgt_rew_pr; // Attention weight for rewards
  real update_pr;  // Updating rate
}

transformed parameters {
  // Transform subject-level raw parameters
  real<lower=-2, upper=2> con;
  real<lower=0, upper=1>  wgt_pun;
  real<lower=0, upper=1>  wgt_rew;
  real<lower=0, upper=1>  update;
  
  con     = inv_logit(con_pr) * 4 - 2;
  wgt_pun = inv_logit(wgt_pun_pr);
  wgt_rew = inv_logit(wgt_rew_pr);
  update  = inv_logit(update_pr);
}

model {
  // Individual parameters
  con_pr     ~ normal(0, 1);
  wgt_pun_pr ~ normal(0, 1);
  wgt_rew_pr ~ normal(0, 1);
  update_pr  ~ normal(0, 1);

  // Initial subject-level deck expectations
  vector[4] ev = rep_vector(0., 4);
  

  // Initial trial data for theta
  vector[T] sensitivity = pow(to_vector(linspaced_array(T, 1, T)) / 10.0, con);

  ev = igt_model_lp(
		choice, shown, outcome,
		T, ev, sensitivity,
		update, wgt_pun, wgt_rew
		);
}
