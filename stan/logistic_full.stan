data {
  int<lower=1> N;
  int<lower=1> P;
  array[N] int<lower=1, upper=P> province;
  array[N] int<lower=0, upper=1> y;

  // Numeric predictors
  vector[N] log_fire_size;
  vector[N] log_dist_to_fn_km;
  array[N] int<lower=0, upper=1> fn_indicator;

  int<lower=1> K_prot;
  array[N] int<lower=1,upper=K_prot> protection_zone;
}

parameters {
  real alpha;
  vector[P] a_prov;

  vector[K_prot] a_prot;

  real beta_log_fire_size;
  real beta_dist;
  real beta_fn;
}

model {
  vector[N] theta;

  // Priors
  alpha ~ normal(0, 2);
  a_prov ~ normal(0, 1);
  a_prot ~ normal(0, 1);

  beta_log_fire_size ~ normal(0, 1);
  beta_dist ~ normal(0, 1);
  beta_fn ~ normal(0, 1);

  for (n in 1:N) {
    theta[n] = alpha
         + a_prov[province[n]]
         + a_prot[protection_zone[n]]
             + beta_log_fire_size * log_fire_size[n]
             + beta_dist * log_dist_to_fn_km[n]
             + beta_fn * fn_indicator[n];
  }

  y ~ bernoulli_logit(theta);
}

generated quantities {
  vector[N] log_lik;
  array[N] int y_rep;

  for (n in 1:N) {
      real theta_n = alpha
       + a_prov[province[n]]
       + a_prot[protection_zone[n]]
         + beta_log_fire_size * log_fire_size[n]
         + beta_dist * log_dist_to_fn_km[n]
         + beta_fn * fn_indicator[n];
    log_lik[n] = bernoulli_logit_lpmf(y[n] | theta_n);
    y_rep[n] = bernoulli_logit_rng(theta_n);
  }
}
