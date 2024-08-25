functions {
  // Modified Random number generator based on Shahar et al. (2019) https://doi.org/10.1371/journal.pcbi.1006803
  vector wiener_rng(real a, real tau, real z, real d) {
    real dt    = .0001;
    real sigma = 1;
    real y     = z * a;  // starting point
    real p     = .5 * (1 + ((d * sqrt(dt)) / sigma));
    int i      = 0;
    vector[3] ret;
    
    while (y < a && y > 0) {
      if (uniform_rng(0, 1) <= p) {
        y += sigma * sqrt(dt);
      } else {
        y -= sigma * sqrt(dt);
      }
      i += 1;
    }
    
    ret[1] = (y >= a) ? 1 : 0;  // Upper boundary choice -> 1 (play), lower boundary choice -> 0 (pass)
    ret[2] = i * dt + tau;
    ret[3] = wiener_log_likelihood(a, tau, z, d, ret[2], ret[1]);
    return ret;
  }

  real wiener_log_likelihood(real a, real tau, real z, real d, real t_obs, real y_obs) {
    real sigma = 1;  // Assuming constant noise, can be parameterized if needed
    real t;
    real p_upper;
    real log_f_upper;
    real log_f_lower;
    real log_p_upper;
    real log_p_lower;
  
    // Parameter validation
    if (a <= 0 || tau < 0 || z < 0 || z > a) {
      return negative_infinity();  // Invalid parameters
    }
  
    // Adjust observed time to remove non-decision time tau
    t = t_obs - tau;
  
    // Ensure adjusted time is positive
    if (t <= 0) {
      return negative_infinity();  // Invalid, non-decision time should be less than observed time
    }
    
    // Probability of hitting the upper boundary (in log space)
    p_upper = Phi((z + a * d) / (sigma * sqrt(a)));
    log_p_upper = log(p_upper);
    log_p_lower = log1m(p_upper);  // log(1 - p) more accurately
  
    // First-passage time densities (in log space)
    log_f_upper = -log(2 * pi()) / 2 - 1.5 * log(t) - square(a - z - d * t) / (2 * square(sigma) * t);
    log_f_lower = -log(2 * pi()) / 2 - 1.5 * log(t) - square(z + d * t) / (2 * square(sigma) * t);
  
    // Add log of prefactors
    log_f_upper += log(a - z);
    log_f_lower += log(z);
  
    // Return log-likelihood based on observed response
    if (y_obs == 1) {
      // Upper boundary
      return log_sum_exp(log_p_upper, log_f_upper);
    } else if (y_obs == 0) {
      // Lower boundary
      return log_sum_exp(log_p_lower, log_f_lower);
    } else {
      // Invalid observation
      return negative_infinity();
    }
  }

  // Generate EV DDM model
  vector igt_model_rng(
			array[] int choice, array[] int shown, array[] real outcome,
			array[] real RT, vector ev, int Tsub, vector sensitivity,
			real update, real wgt_pun, real wgt_rew,
			real boundary, real tau, real beta
			) {
    // Define values
    real curUtil;   // Current utility
    int  curDeck;   // Current deck
    real EV2update; // Current EV to update
    real curDrift;  // Current drift

    // Outputs
    real         log_lik = 0;
    vector[4]    local_ev = ev;
    vector[Tsub] rt_pred;
    vector[Tsub] choice_pred;

    // For each deck shown
    for (t in 1:Tsub) {
      // Deck presented to sub
      curDeck = shown[t];

      // EV to update
      EV2update = local_ev[curDeck];

      // Drift diffusion process
      curDrift = EV2update * sensitivity[t]; // Drift scaling

      // Generate responses
      vector[3] result = wiener_rng(boundary, tau, beta, curDrift);

      choice_pred[t] = result[1];  // 1 (play), 0 (pass)
      rt_pred[t]     = result[2];

      log_lik += result[3];

      if (choice[t] == 1) {
         log_lik += wiener_lpdf(rt_pred[t] | boundary, tau, beta, curDrift);
      } else {
         log_lik += wiener_lpdf(rt_pred[t] | boundary, tau, 1-beta, -curDrift);
      }

      // Compute utility
      curUtil = outcome[t] * (outcome[t] > 0 ? wgt_rew : wgt_pun) * choice[t];

      // Update expected values
      local_ev[curDeck] += (curUtil - 2 * EV2update) * update * choice[t];
    }

    // Concatenate all results into a single vector
    return append_row(append_row(append_row(local_ev, log_lik), choice_pred), rt_pred);
  }
}

data {
  int<lower=1> 			 T; 	  // Number of trials
  real<lower=0> 		 minRT;   // Minimum RT + small value to restrict tau
  real 				 RTbound; // Lower bound or RT across all subjects (e.g., 0.1 second)
  array[T] real<lower=0> 	 RT;  	  // Reaction times
  array[T] int<lower=0, upper=1> choice;  // Binary choices made at each trial
  array[T] int<lower=0, upper=4> shown;   // Deck shown at each trial
  array[T] real 		 outcome; // Outcome at each trial
}

parameters {
  real boundary_pr;  // Boundary separation (a)
  real tau_pr;       // Non-decision time (tau)
  real beta_pr;      // Starting point
  real drift_con_pr; // Drift consistency parameter
  real wgt_pun_pr;   // Attention weight for punishments
  real wgt_rew_pr;   // Attention weight for rewards
  real update_pr;    // Updating rate
}

transformed parameters {
  real<lower=0> 		   boundary;
  real<lower=RTbound, upper=minRT> tau;
  real<lower=0, upper=1> 	   beta;
  real<lower=-2, upper=2> 	   drift_con;
  real<lower=0, upper=1> 	   wgt_pun;
  real<lower=0, upper=1> 	   wgt_rew;
  real<lower=0, upper=1> 	   update;

  boundary  = exp(boundary_pr);
  tau       = inv_logit(tau_pr) * (minRT - RTbound) + RTbound;
  beta      = inv_logit(beta_pr);
  drift_con = inv_logit(drift_con_pr) * 4 - 2;
  wgt_pun   = inv_logit(wgt_pun_pr);
  wgt_rew   = inv_logit(wgt_rew_pr);
  update    = inv_logit(update_pr);
}

generated quantities {
  real 	    log_lik = 0;
  vector[T] choice_pred;
  vector[T] rt_pred;
  vector[4] final_ev = rep_vector(0., 4);

  {  
    vector[T] log_lik_array;
    vector[T] sensitivity = pow(to_vector(linspaced_array(T, 1, T)) / 10.0, drift_con);
      
    vector[4 + 1 + 3*T] results = igt_model_rng(
					choice, shown, outcome,
					RT, ev, T, sensitivity,
			 		update, wgt_pun, wgt_rew, 			 							boundary, tau, beta
					);

    final_ev 	  = results[1:4];
    log_lik_array = results[5:(4+T)];
    choice_pred	  = results[(5+T):(4+2*T)];
    rt_pred       = to_array_1d(results[(6+T):(5+2*T)]);
      
    log_lik = sum(log_lik_array);
    }
  }
}
