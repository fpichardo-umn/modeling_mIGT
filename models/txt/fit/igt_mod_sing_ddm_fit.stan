
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

model {
  // Priors
  boundary_pr ~ normal(0, 1);
  tau_pr      ~ normal(0, 1);
  beta_pr     ~ normal(0, 1);
  drift	      ~ normal(0, 1);
  
  // Likelihood
  RTplay ~ wiener(boundary, tau, beta, drift);
  RTpass ~ wiener(boundary, tau, 1-beta, -drift);
}
