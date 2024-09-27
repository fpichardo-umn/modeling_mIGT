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
  int<lower=1> sid;  // Subject ID
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

generated quantities {
  int subid = sid;  // Include the subject ID in the output
}
