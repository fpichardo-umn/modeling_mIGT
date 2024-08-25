
data {
  int<lower=1> 		   N;         // Number of subjects
  int<lower=1> 		   T;         // Max overall trials
  real<lower=0> 	   RTbound;   // Lower bound RT across all subjects (e.g., 0.1 second)
  int<lower=0> 		   Nplay_max; // Max (across subjects) number of play trials
  int<lower=0> 		   Npass_max; // Max (across subjects) number of pass trials
  int<lower=0> 		   Nplay[N];  // Number of play trials for each sub
  int<lower=0> 		   Npass[N];  // Number of pass trials for each sub
  array[N, Nplay_max] real RTplay;    // Reaction times for play trials
  array[N, Npass_max] real RTpass;    // Reaction times for sub trials
  vector[N] 		   minRT;     // Minimum RT for each sub
}

transformed data{
  vector[N] minRTdiff    = minRT - RTbound;
  real      RTmax        = max(minRT);
} 

parameters {
  // Hyper-parameters
  vector[4] 	     mu_pr;
  vector<lower=0>[4] sigma;

  // Subject-level raw parameters
  vector[N] boundary_pr;  // Boundary separation (a)
  vector[N] tau_pr;  	  // Non-decision time (tau)
  vector[N] beta_pr;  	  // Starting point
  vector[N] drift_pr;  	  // Drift rate
}

transformed parameters {
  vector<lower=0>[N] 			boundary;
  vector<lower=RTbound, upper=RTmax>[N] tau;
  vector<lower=0, upper=1>[N] 		beta;
  vector[N] 				drift;

  boundary = exp(mu_pr[1] + sigma[1] * boundary_pr);
  tau      = inv_logit(mu_pr[2] + sigma[2] * tau_pr) .* minRTdiff + RTbound;
  beta     = inv_logit(mu_pr[3] + sigma[3] * beta_pr);
  drift    = mu_pr[4] + sigma[4] * drift_pr;
}

model {
  // Hyperparameters
  mu_pr ~ normal(0, 1);
  sigma ~ normal(0, 1);

  // Priors
  to_vector(boundary_pr) ~ normal(0, 1);
  to_vector(tau_pr)      ~ normal(0, 1);
  to_vector(beta_pr)     ~ normal(0, 1);
  to_vector(drift)       ~ normal(0, 1);
  
  // For each sub
  for (n in 1:N) { 
    RTplay[n, 1:Nplay[n]] ~ wiener(boundary[n], tau[n],   beta[n],  drift[n]);
    RTpass[n, 1:Npass[n]] ~ wiener(boundary[n], tau[n], 1-beta[n], -drift[n]);
  }
}

generated quantities {
  // Init
  real<lower=RTbound, upper=RTmax> mu_tau;
  real<lower=0, upper=1>           mu_beta;
  real<lower=0> 		   mu_boundary;
  real 				   mu_drift = mu_pr[4];

  {
    // Pre-transformed mu
    vector[4] mu_transformed = inv_logit(mu_pr);

    real RTlowerbound = (mean(minRT) - RTbound) + RTbound;

    // Compute interpretable group-level parameters
    mu_boundary = exp(mu_pr[1]);

    mu_tau  = mu_transformed[2] * RTlowerbound;
    mu_beta = mu_transformed[3];
  }
}
