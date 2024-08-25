functions {
  // Modified Random number generator based on Shahar et al. (2019) https://doi.org/10.1371/journal.pcbi.1006803
  vector wiener_rng(real a, real tau, real z, real d) {
    real dt = .0001;
    real sigma = 1;
    real y = z * a;  // starting point
    real p = .5 * (1 + ((d * sqrt(dt)) / sigma));
    int i = 0;
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
  int<lower=1> N;      // Number of subjects
  int<lower=1> T;      // Max overall trials
  int<lower=0> Nplay_max; // Max (across subjects) number of play trials
  int<lower=0> Npass_max; // Max (across subjects) number of pass trials
  int<lower=0> Nplay[N];  // Number of play trials for each sub
  int<lower=0> Npass[N];  // Number of pass trials for each sub
  array[N, Nplay_max] real RTplay;  // Reaction times for play trials
  array[N, Npass_max] real RTpass;  // Reaction times for play trials
  array[N] real minRT;  // Minimum RT + small value to restrict tau for each sub
  real<lower=0> RTbound; // Lower bound or RT across all subjects (e.g., 0.1 second)
}

parameters {
  array[N] real boundary_pr;  // Boundary separation (a)
  array[N] real tau_pr;  // Non-decision time (tau)
  array[N] real beta_pr;  // Starting point
  vector[N] drift;  // Drift rate
}

transformed parameters {
  vector<lower=0>[N] boundary;
  vector<lower=RTbound, upper=max(minRT)>[N] tau;
  vector<lower=0, upper=1>[N] beta;

  boundary = exp(to_vector(boundary_pr));
  for (n in 1:N) {
    tau[n] = inv_logit(tau_pr[n]) * (minRT[n] - RTbound) + RTbound;
  }
  beta = inv_logit(to_vector(beta_pr));
}

model {
  // Priors
  to_vector(boundary_pr) ~ normal(0, 1);
  to_vector(tau_pr) ~ normal(0, 1);
  to_vector(beta_pr) ~ normal(0, 1);
  to_vector(drift) ~ normal(0, 1);

  // print("Init: boundary_pr=", boundary_pr, " tau_pr=", tau_pr, " beta_pr=", beta_pr);
  // print("Init: boundary=", boundary, " tau=", tau, " beta=", beta, " drift=", drift);
  
  // For each sub
  for (n in 1:N) { 
    RTplay[n, 1:Nplay[n]] ~ wiener(boundary[n], tau[n], beta[n], drift[n]);
    RTpass[n, 1:Npass[n]] ~ wiener(boundary[n], tau[n], 1-beta[n], -drift[n]);
  }
}

generated quantities {
  real total_log_lik = 0;
  array[N, T] real rt_pred = rep_array(-1, N, T);  // Initialize with -1
  array[N, T] int choice_pred = rep_array(-1, N, T);  // Initialize with -1
  
  {
    vector[N] log_lik;

    for (n in 1:N) {
      int Tsubj = Nplay[n] + Npass[n];
      for (t in 1:Tsubj) {
        vector[2] result = wiener_rng(boundary[n], tau[n], beta[n], drift[n]);
    
        choice_pred[n,t] = to_int(result[1]);  // 1 (play), 0 (pass)
        rt_pred[n,t] = result[2];
      }
      
      log_lik[n] = wiener_lpdf(RTplay | boundary[n], tau[n], beta[n], drift[n]);
      log_lik[n] += wiener_lpdf(RTpass | boundary[n], tau[n], 1-beta[n], -drift[n]);
      
      total_log_lik += log_lik[n];
    }
  }
}
