functions {
  // Modified Random number generator based on Shahar et al. (2019) https://doi.org/10.1371/journal.pcbi.1006803
  vector wiener_rng(real a, real tau, real z, real d) {
    real dt = .0001;
    real sigma = 1;
    real y = z * a;  // starting point
    real p = .5 * (1 + ((d * sqrt(dt)) / sigma));
    int i = 0;
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

  // Fitting EV DDM model
  vector igt_model_lp(array[] int choice, array[] int shown, array[] real outcome, 
                       array[] real RT, vector ev, int Tsub, vector theta_ts, 
                       real update, real wgt_pun, real wgt_rew, real drift_con, 
                       real boundary, real tau, real beta) {
    // Define values
    real curUtil; // Current utility
    real theta;   // Sensitivity parameter
    int  curDeck; // Current deck
    real EV2update;  // Current EV to update
    real curDrift; // Current drift
    vector[4] local_ev = ev;

    // For each deck shown
    for (t in 1: Tsub) {
      // Deck presented to sub
      curDeck = shown[t];

      // EV to update
      EV2update = local_ev[curDeck];

      // Drift diffusion process
      theta = pow((theta_ts[t]), drift_con);
      curDrift = EV2update * theta; // Drift scaling

      if (choice[t] == 1) {
        RT[t] ~ wiener(boundary, tau, beta, curDrift);
      } else {
        RT[t] ~ wiener(boundary, tau, 1-beta, -curDrift);
      }

      // Compute utility
      curUtil = (outcome[t] * (outcome[t] > 0 ? wgt_rew : wgt_pun)) * choice[t]; // if choice == 0, curUtil is 0

      // Update expected values
      local_ev[curDeck] += choice[t] * update * (curUtil - 2 * EV2update); // if choice == 0, update is just EV2update
    }
    return local_ev;
  }

  // Generate EV DDM model
  vector igt_model_rng(array[] int choice, array[] int shown, array[] real outcome, 
                       array[] real RT, vector ev, int Tsub, vector theta_ts, 
                       real update, real wgt_pun, real wgt_rew, real drift_con, 
                       real boundary, real tau, real beta) {
    // Define values
    real curUtil; // Current utility
    real theta;   // Sensitivity parameter
    int curDeck; // Current deck
    real EV2update;  // Current EV to update
    real curDrift; // Current drift
    real log_lik = 0;
    vector[4] local_ev = ev;
    vector[Tsub] rt_pred;
    vector[Tsub] choice_pred;

    // For each deck shown
    for (t in 1:Tsub) {
      // Deck presented to sub
      curDeck = shown[t];

      // EV to update
      EV2update = local_ev[curDeck];

      // Drift diffusion process
      theta = pow((theta_ts[t]), drift_con);
      curDrift = EV2update * theta; // Drift scaling

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
      curUtil = (outcome[t] * (outcome[t] > 0 ? wgt_rew : wgt_pun)) * choice[t];

      // Update expected values
      local_ev[curDeck] += choice[t] * update * (curUtil - 2 * EV2update);
    }

    // Concatenate all results into a single vector
    return append_row(append_row(append_row(local_ev, log_lik), choice_pred), rt_pred);
  }
}

data {
  int<lower=1> N; // Number of subjects
  int<lower=1> T; // Max overall number of trials
  int<lower=1> Tsubj[N]; // Number of trials for a subject
  real RTbound; // Lower bound or RT across all subjects (e.g., 0.1 second)
  array[N] real minRT;  // Minimum RT for each sub
  array[N, T] real<lower=0> RT;  // Reaction times
  array[N, T] int<lower=0, upper=1> choice; // Binary choices made at each trial
  array[N, T] int<lower=0, upper=4> shown; // Deck shown at each trial
  array[N, T] real outcome; // Outcome at each trial
}

transformed data{
  vector[N] minRTdiff = to_vector(minRT) - RTbound;
}

parameters {
  vector[N] boundary_pr;  // Boundary separation (a)
  vector[N] tau_pr;  // Non-decision time (tau)
  vector[N] beta_pr;  // Starting point
  vector[N] drift_con_pr; // Drift consistency parameter
  vector[N] wgt_pun_pr; // Attention weight for punishments
  vector[N] wgt_rew_pr; // Attention weight for rewards
  vector[N] update_pr; // Updating rate
}

transformed parameters {
  vector<lower=0>[N] 			boundary;
  vector<lower=RTbound, upper=max(minRT)>[N] tau;
  vector<lower=0, upper=1>[N] 		beta;
  vector<lower=-2, upper=2>[N] 		drift_con;
  vector<lower=0, upper=1>[N] 		wgt_pun;
  vector<lower=0, upper=1>[N] 		wgt_rew;
  vector<lower=0, upper=1>[N] 		update;

  boundary  = exp(to_vector(boundary_pr));
  tau       = inv_logit(tau_pr) .* (minRTdiff) + RTbound;
  beta      = inv_logit(to_vector(beta_pr));
  drift_con = inv_logit(to_vector(drift_con_pr)) * 4 - 2;
  wgt_pun   = inv_logit(to_vector(wgt_pun_pr));
  wgt_rew   = inv_logit(to_vector(wgt_rew_pr));
  update    = inv_logit(to_vector(update_pr));
}

model {
  // Priors
  to_vector(boundary_pr)  ~ normal(0, 1);
  to_vector(tau_pr)       ~ normal(0, 1);
  to_vector(beta_pr)      ~ normal(0, 1);
  to_vector(drift_con_pr) ~ normal(0, 1);
  to_vector(wgt_pun_pr)   ~ normal(0, 1);
  to_vector(wgt_rew_pr)   ~ normal(0, 1);
  to_vector(update_pr)    ~ normal(0, 1);

  // Initial subject-level deck expectations
  array[N] vector[4] ev;
  for (n in 1:N) {
    ev[n] = rep_vector(0., 4);
  }

  // Initial trial data for theta
  vector[T] theta_ts = to_vector(linspaced_array(T, 1, T)) / 10.0;

  // For each subject
  for (n in 1:N) {
    ev[n] = igt_model_lp(choice[n][:Tsubj[n]], shown[n][:Tsubj[n]], outcome[n][:Tsubj[n]],
			 RT[n][:Tsubj[n]], ev[n], Tsubj[n], theta_ts[:Tsubj[n]],
			 update[n], wgt_pun[n], wgt_rew[n], drift_con[n],
			 boundary[n], tau[n], beta[n]);
  }
}

generated quantities {
  real total_log_lik = 0;
  array[N, T] real choice_pred;
  array[N, T] real rt_pred;
  array[N] vector[4] final_ev;

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

    for (n in 1:N) {
      vector[Tsubj[n]] theta_ts = main_theta_ts[:Tsubj[n]];
      
      vector[4 + 1 + 2*Tsubj[n]] results = igt_model_rng(choice[n][:Tsubj[n]], shown[n][:Tsubj[n]], 
			 		outcome[n][:Tsubj[n]], RT[n][:Tsubj[n]], ev[n], Tsubj[n],
			 		theta_ts[:Tsubj[n]], update[n], wgt_pun[n], wgt_rew[n], 			 			drift_con[n], boundary[n], tau[n], beta[n]);
      
      final_ev[n]               = results[1:4];
      log_lik[n]                = results[5];
      choice_pred[n][:Tsubj[n]] = to_array_1d(results[6:(5+Tsubj[n])]);
      rt_pred[n][:Tsubj[n]]   = to_array_1d(results[(6+Tsubj[n]):(5+2*Tsubj[n])]);
      
      total_log_lik += log_lik[n];
    }
  }
}
