data {
  int<lower=1> N; // Number of subjects
  int<lower=1> T; // Number of trials
  int<lower=1> Tsubj[N]; // Number of trials for a subject
  int<lower=0, upper=1> choice[N, T]; // Binary choices made at each trial
  int<lower=0, upper=4> shown[N, T]; // Deck shown at each trial
  real outcome[N, T]; // Outcome (O) at each trial
}

parameters {
  // Subject-level raw parameters
  vector[N] update_pr; // Updating rate
  vector[N] wgt_pr; // Attention weight
  vector[N] con_pr; // Consistency parameter
}

transformed parameters {
  // Transform subject-level raw parameters
  vector<lower=-2, upper=2>[N] con;
  vector<lower=0, upper=1>[N] wgt;
  vector<lower=0, upper=1>[N] update;

  for (n in 1:N) {
    con[n]     = Phi_approx(con_pr[n]) * 4 - 2;
    wgt[n] = Phi_approx(wgt_pr[n]);
    update[n]  = Phi_approx(update_pr[n]);
  }
}

model {
  // Individual parameters
  update_pr  ~ normal(0, 1);
  wgt_pr ~ normal(0, 1);
  con_pr  ~ normal(0, 1);

  // For each subject
  for (n in 1:N) {
    // Define values
    vector[4] ev = rep_vector(0, 4); // Initial subject-level deck expectations
    real curUtil; // Current utility
    real theta;   // Sensitivity parameter

    int curDeck; // Current deck
    real prob_choose_deck;         // Current probability of choosing the deck shown

    // For each deck shown
    for (t in 1:Tsubj[n]) {
      // Deck presented to sub
      curDeck = shown[n, t];

      // Dynamic theta
      theta = pow((t / 10.0), con[n]);

      // Softmax choice rule: logistic based on the expected value of the current deck - inv_logit == logistic
      prob_choose_deck = inv_logit(theta * ev[curDeck]); // Probability of choosing the current deck

      // Bernoulli distribution to decide whether to gamble on the current deck or not
      choice[n, t] ~ bernoulli(prob_choose_deck);

      if (choice[n, t] != 0) {  // if the deck is played, use feedback to learn
        // Compute utility
        curUtil = outcome[n, t] * (outcome[n, t] > 0 ? wgt[n] :(1 - wgt[n]));

        // Update expected values
        ev[curDeck] = (ev[curDeck] * (1 - update[n])) + (update[n] * (curUtil - ev[curDeck]));
      }
    }
  }
}
