data {
  int<lower=1> T; // Number of trials
  int<lower=0, upper=1> choice[T]; // Binary choices made at each trial
  int<lower=1, upper=4> shown[T]; // Deck shown at each trial
  real outcome[T]; // Outcome at each trial
}

parameters {
  // Subject-level raw parameters
  real update_pr;  // Updating rate
  real wgt_pr; // Attention weight - closer to 0 is more punishment and closer to 1 is more reward
  real<lower=0> con_pr;     // Consistency parameter
}

transformed parameters {
  // Transform subject-level raw parameters
  real<lower=-2, upper=2> con;
  real<lower=0, upper=1> wgt;
  real<lower=0, upper=1> update;

  con     = Phi_approx(con_pr) * 4 - 2;
  wgt     = Phi_approx(wgt_pr);
  update  = Phi_approx(update_pr);
}

model {
  // Individual parameters
  update_pr  ~ normal(0, 0.1);
  wgt_pr ~ normal(0, 0.1);
  con_pr  ~ lognormal(-3, 0.1);

  // Define values
  vector[4] ev; // Expected values for decks
  real curUtil; // Current utility
  real theta;   // Sensitivity parameter

  int curDeck; // Current deck
  real prob_choose_deck;         // Current probability of choosing the deck shown

  // Initialize values
  ev = rep_vector(0.0, 4); // Initial subject-level deck expectations

  // For each deck shown
  for (t in 1:T) {
    // Deck presented to sub
    curDeck = shown[t];

    // Dynamic theta
    theta = pow((t / 10), con);

    // Softmax choice rule: logistic based on the expected value of the current deck - inv_logit == logistic
    prob_choose_deck = inv_logit(theta * ev[curDeck]); // Probability of choosing the current deck

    // Bernoulli distribution to decide whether to gamble on the current deck or not
    choice[t] ~ bernoulli(prob_choose_deck);

    if (choice[t] != 0) {  // if the deck is played, use feedback to learn
      // Compute utility
      curUtil = outcome[t] * (outcome[t] > 0 ? wgt :(1-wgt));

      // Update expected values
      ev[curDeck] = (ev[curDeck] * (1 - update)) + (update * (curUtil - ev[curDeck]));
    }
  }
}
