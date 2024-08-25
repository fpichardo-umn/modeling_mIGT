functions {
  // Modified Random number generator based on Shahar et al. (2019) https://doi.org/10.1371/journal.pcbi.1006803
  vector wiener_rng(real a, real tau, real z, real d) {
    real      dt    = .0001;
    real      sigma = 1;
    real      y     = z * a;  // starting point
    real      p     = .5 * (1 + ((d * sqrt(dt)) / sigma));
    int       i     = 0;
    vector[2] ret;
    
    while (y < a && y > 0) {
      if (uniform_rng(0, 1) <= p) {
        y += sigma * sqrt(dt);
      } else {
        y -= sigma * sqrt(dt);
      }
      i += 1;
    }
    
    ret[1] = (y >= a) ? 1 : 0;  // Upper: choice, 1 (play), lower: choice, 0 (pass)
    ret[2] = i * dt + tau;
    return ret;
  }
}

data {
  int<lower=1> 		   N;         // Number of subjects
  int<lower=1> 		   T;         // Max overall trials
  real<lower=0> 	   RTbound;   // Lower bound RT across all subjects (e.g., 0.1 second)
  int<lower=0> 		   Nplay_max; // Max (across subjects) number of play trials
  int<lower=0> 		   Npass_max; // Max (across subjects) number of pass trials
  int<lower=0> 		   Nplay[N];  // Number of play trials for each sub
  int<lower=0> 		   Npass[N];  // Number of pass trials for each sub
  array[N, Nplay_max] real RTplay;    // Reaction times for play trials
  array[N, Npass_max] real RTpass;    // Reaction times for pass trials
  vector[N] 		   minRT;     // Minimum RT for each sub
}

transformed data{
  vector[N] minRTdiff = minRT - RTbound;
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
  vector<lower=0>[N] 			     boundary;
  vector<lower=RTbound, upper=max(minRT)>[N] tau;
  vector<lower=0, upper=1>[N] 		     beta;
  vector[N] 				     drift;

  boundary = exp(mu_pr[1] + sigma[1] * boundary_pr);
  tau      = inv_logit(mu_pr[2] + sigma[2] * tau_pr) .* (minRTdiff) + RTbound;
  beta     = inv_logit(mu_pr[3] + sigma[3] * beta_pr);
  drift    = mu_pr[4] + sigma[4] * drift_pr;
}

generated quantities {
  // Output params
  real 	           total_log_lik = 0;
  array[N, T] int  choice_pred	 = rep_array(-1, N, T);  // Initialize with -1
  array[N, T] real rt_pred	 = rep_array(-1, N, T);  // Initialize with -1
  
  {
    vector[N] log_lik;

    for (n in 1:N) {
      int Tsubj = Nplay[n] + Npass[n];
      for (t in 1:Tsubj) {
        vector[2] result = wiener_rng(boundary[n], tau[n], beta[n], drift[n]);

        // Unpack and accumulate results
        choice_pred[n,t] = to_int(result[1]);  // 1 (play), 0 (pass)
        rt_pred[n,t]     = result[2];
      }
      
      log_lik[n]  = wiener_lpdf(RTplay | boundary[n], tau[n], beta[n], drift[n]);
      log_lik[n] += wiener_lpdf(RTpass | boundary[n], tau[n], 1-beta[n], -drift[n]);
      
      total_log_lik += log_lik[n];
    }
  }
}
