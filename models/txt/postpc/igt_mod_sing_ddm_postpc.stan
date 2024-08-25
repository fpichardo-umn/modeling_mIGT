functions {
  // Modified Random number generator based on Shahar et al. (2019) https://doi.org/10.1371/journal.pcbi.1006803
  vector wiener_rng(real a, real tau, real z, real d) {
    real dt    = .0001;
    real sigma = 1;
    real y     = z * a;  // starting point
    real p     = .5 * (1 + ((d * sqrt(dt)) / sigma));
    int  i     = 0;
    vector[2] ret;
    
    while (y < a && y > 0) {
      if (uniform_rng(0, 1) <= p) {
        y += sigma * sqrt(dt);
      } else {
        y -= sigma * sqrt(dt);
      }
      i += 1;
    }
    
    ret[1] = (y >= a) ? 1.0 : 0.0;  // Upper boundary choice -> 1 (play), lower boundary choice -> 0 (pass)
    ret[2] = i * dt + tau;
    return ret;
  }
}

data {
  int<lower=1>  	     Nplay;   // Number of play trials
  int<lower=1>  	     Npass;   // Number of pass trials
  real<lower=0> 	     minRT;   // Minimum RT + small value to restrict tau
  real<lower=0>		     RTbound; // Lower bound or RT across all subjects (e.g., 0.1 second)
  array[Nplay] real<lower=0> RTplay;  // Reaction times for play trials
  array[Npass] real<lower=0> RTpass;  // Reaction times for play trials
}

parameters {
  real boundary_pr; // Boundary separation (a)
  real tau_pr;      // Non-decision time (tau)
  real beta_pr;     // Starting point
  real drift;       // Drift rate
}

transformed parameters {
  real<lower=0> 		   boundary;
  real<lower=RTbound, upper=minRT> tau;
  real<lower=0, upper=1> 	   beta;


  boundary = exp(boundary_pr);
  tau 	   = inv_logit(tau_pr) * (minRT - RTbound) + RTbound;
  beta     = inv_logit(beta_pr);
}


generated quantities {
  real 		            log_lik;
  array[Nplay + Npass] real rt_pred;
  array[Nplay + Npass] int  choice_pred;
  array[Nplay + Npass] real prob_play;
  
  {
    for (t in 1:(Nplay + Npass)) {
      vector[2] result = wiener_rng(boundary, tau, beta, drift);
    
      choice_pred[t] = to_int(result[1]);  // 1 (play), 0 (pass)
      rt_pred[t]     = result[2];
      prob_play[t]   = Phi((drift / boundary) * (boundary - (boundary/2)));
    }
    
    log_lik  = wiener_lpdf(RTplay | boundary, tau, beta, drift);
    log_lik += wiener_lpdf(RTpass | boundary, tau, 1-beta, -drift);
    
  }
}
