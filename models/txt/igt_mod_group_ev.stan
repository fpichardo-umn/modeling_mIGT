functions {
  vector igt_model_lp(array[] int choice, array[] int shown, array[] real outcome, vector ev, 
                  int Tsub, vector theta_ts, real update, real wgt_pun, real wgt_rew, real con) {
    // Define values
    real curUtil; // Current utility
    real theta;   // Sensitivity parameter
    int curDeck; // Current deck
    real EV2update;  // Current EV to update
    vector[4] local_ev = ev;

    // For each deck shown
    for (t in 1: Tsub) {
      // Deck presented to sub
      curDeck = shown[t];

      // EV to update
      EV2update = local_ev[curDeck];

      // Dynamic theta
      theta = pow((theta_ts[t]), con);

      // Bernoulli distribution to decide whether to gamble on the current deck or not
      // Use bernoulli_logit_lpmf for numerical stability
      target += bernoulli_logit_lpmf(choice[t] | theta * EV2update);

      // Compute utility
      curUtil = (outcome[t] * (outcome[t] > 0 ? wgt_rew : wgt_pun)) * choice[t]; // if choice == 0, curUtil is 0

      // Update expected values
      local_ev[curDeck] += choice[t] * update * (curUtil - 2 * EV2update); // if choice == 0, update is just EV2update
    }
    return local_ev;
  }

  vector igt_model_rng(array[] int choice, array[] int shown, array[] real outcome, vector ev, 
                       int Tsub, vector theta_ts, real update, real wgt_pun, real wgt_rew, real con) {
    // Define values
    real curUtil; // Current utility
    real theta;   // Sensitivity parameter
    int curDeck; // Current deck
    real EV2update;  // Current EV to update
    vector[4] local_ev = ev;
    vector[Tsub] log_lik;
    vector[Tsub] y_pred;

    // For each deck shown
    for (t in 1:Tsub) {
      // Deck presented to sub
      curDeck = shown[t];

      // EV to update
      EV2update = local_ev[curDeck];

      // Dynamic theta
      theta = pow((theta_ts[t]), con);

      // Calculate log likelihood
      log_lik[t] = bernoulli_logit_lpmf(choice[t] | theta * EV2update);

      // Generate prediction
      y_pred[t] = bernoulli_rng(inv_logit(theta * EV2update));

      // Compute utility
      curUtil = (outcome[t] * (outcome[t] > 0 ? wgt_rew : wgt_pun)) * choice[t];

      // Update expected values
      local_ev[curDeck] += choice[t] * update * (curUtil - 2 * EV2update);
    }

    // Concatenate all results into a single vector
    return append_row(append_row(local_ev, log_lik), y_pred);
  }
}

data {
  int<lower=1> N; // Number of subjects
  int<lower=1> T; // Number of trials
  int<lower=1> Tsubj[N]; // Number of trials for a subject
  array[N, T] int<lower=0, upper=1> choice; // Binary choices made at each trial
  array[N, T] int<lower=0, upper=4> shown; // Deck shown at each trial
  array[N, T] real outcome; // Outcome at each trial
}

parameters {
  // Subject-level raw parameters
  array[N] real con_pr; // Consistency parameter
  array[N] real wgt_pun_pr; // Attention weight for punishments
  array[N] real wgt_rew_pr; // Attention weight for rewards
  array[N] real update_pr; // Updating rate
}

transformed parameters {
  // Transform subject-level raw parameters
  vector<lower=-2, upper=2>[N] con;
  vector<lower=0, upper=1>[N] wgt_pun;
  vector<lower=0, upper=1>[N] wgt_rew;
  vector<lower=0, upper=1>[N] update;
  
  con = inv_logit(to_vector(con_pr)) * 4 - 2;
  wgt_pun = inv_logit(to_vector(wgt_pun_pr));
  wgt_rew = inv_logit(to_vector(wgt_rew_pr));
  update = inv_logit(to_vector(update_pr));
}

model {
  // Individual parameters
  to_vector(con_pr)  ~ normal(0, 1);
  to_vector(wgt_pun_pr) ~ normal(0, 1);
  to_vector(wgt_rew_pr) ~ normal(0, 1);
  to_vector(update_pr)  ~ normal(0, 1);

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
			 ev[n], Tsubj[n], theta_ts[:Tsubj[n]],
			 update[n], wgt_pun[n], wgt_rew[n], con[n]);
  }
}

generated quantities {
  array[N] real log_lik;
  array[N, T] int<lower=0, upper=1> y_pred;
  array[N] vector[4] final_ev;
  real total_log_lik;

  vector[T] main_theta_ts = to_vector(linspaced_array(T, 1, T)) / 10.0;
  
  {
    total_log_lik = 0;
    
    for (n in 1:N) {
      vector[4] ev = rep_vector(0., 4);
      vector[Tsubj[n]] theta_ts = main_theta_ts[:Tsubj[n]];
      
      vector[4 + 2*Tsubj[n]] results = igt_model_rng(choice[n][:Tsubj[n]], shown[n][:Tsubj[n]], 
                                                     outcome[n][:Tsubj[n]], ev, Tsubj[n], theta_ts,
                                                     update[n], wgt_pun[n], wgt_rew[n], con[n]);
      
      final_ev[n] = results[1:4];
      log_lik[n] = sum(results[5:(4+Tsubj[n])]);
      y_pred[n][:Tsubj[n]] = to_int(to_array_1d(results[(5+Tsubj[n]):(4+2*Tsubj[n])]));
      
      total_log_lik += log_lik[n];
    }
  }
}
