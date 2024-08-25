functions {
  vector igt_model_rng(
			int[] choice, int[] shown, real[] outcome,
			int T, vector ev, vector sensitivity, 
			real update, real wgt_pun, real wgt_rew
			) {
    // Define values
    real curUtil;   // Current utility
    real curInfo;   // Current utility
    int curDeck;    // Current deck
    real EV2update; // Current EV to update

    // Output
    vector[4] local_ev = ev;
    vector[T] log_lik;
    vector[T] y_pred;

    // For each deck shown
    for (t in 1:T) {
      // Deck presented to sub
      curDeck = shown[t];

      // EV to update
      EV2update = local_ev[curDeck];
      curInfo = sensitivity[t] * EV2update;

      // Calculate log likelihood
      log_lik[t] = bernoulli_logit_lpmf(choice[t] | curInfo);

      // Generate prediction
      y_pred[t] = bernoulli_rng(inv_logit(curInfo));

      // Compute utility
      curUtil = (outcome[t] * (outcome[t] > 0 ? wgt_rew : wgt_pun)) * choice[t];

      // Update expected values
      local_ev[curDeck] += (curUtil - 2 * EV2update) * update * choice[t];
    }

    // Concatenate all results into a single vector
    return append_row(append_row(local_ev, log_lik), y_pred);
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

generated quantities {
  real      log_lik;
  vector[T] y_pred;
  vector[T] log_lik_array;
  vector[4] final_ev = rep_vector(0., 4);
  
  {
    vector[T] sensitivity = pow(to_vector(linspaced_array(T, 1, T)) / 10.0, con);
    
    vector[4 + 2*T] results = igt_model_rng(
				choice, shown, outcome,
				T, ev, sensitivity,
				update, wgt_pun, wgt_rew
				);
    
    final_ev 	  = results[1:4];
    log_lik_array = results[5:(4+T)];
    y_pred 	  = results[(5+T):(4+2*T)];
    
    // Sum log_lik_array to get total log likelihood
    log_lik = sum(log_lik_array);
  }
}
