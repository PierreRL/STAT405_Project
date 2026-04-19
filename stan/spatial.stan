data {
  int<lower=1> N;
  int<lower=1> P;
  int<lower=1> R;
  int<lower=1> K_prot;
  int<lower=0> E;

  array[N] int<lower=1, upper=P> province;
  array[N] int<lower=1, upper=R> region;
  array[N] int<lower=1, upper=K_prot> protection_zone;
  array[N] int<lower=0, upper=1> y;

  // Numeric predictors
  vector[N] log_fire_size;
  vector[N] log_dist_to_fn_km;
  array[N] int<lower=0, upper=1> fn_indicator;

  // Spatial adjacency
  array[E] int<lower=1, upper=R> node1;
  array[E] int<lower=1, upper=R> node2;
}

parameters {
  real alpha;

  vector[P] a_prov;
  vector[K_prot] a_prot;
  vector[R] u_region;

  real beta_log_fire_size;
  real beta_dist;
  real beta_fn;

  real<lower=0> sigma_region;
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

  // Spatial effects
  u_region ~ normal(0, 1);
  mean(u_region) ~ normal(0, 0.001); // Sum to zero
  sigma_region ~ normal(0, 1);

  // Spatial structure
  if (E > 0) {
    target += -0.5 / square(sigma_region)
              * dot_self(u_region[node1] - u_region[node2]);
  }

  for (n in 1:N) {
    theta[n] = alpha
             + a_prov[province[n]]
             + a_prot[protection_zone[n]]
             + u_region[region[n]]
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
                 + u_region[region[n]]
                 + beta_log_fire_size * log_fire_size[n]
                 + beta_dist * log_dist_to_fn_km[n]
                 + beta_fn * fn_indicator[n];

    log_lik[n] = bernoulli_logit_lpmf(y[n] | theta_n);
    y_rep[n] = bernoulli_logit_rng(theta_n);
  }
}