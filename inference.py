"""
inference.py
============
Bayesian inference engine for the post-AGB binary mass-loss model.

Posterior (per the user's formulation):
    P(theta | D)  ∝  L(D | theta) * P(theta)

  * P(theta)  : priors informed by Gaia DR3 NSS (P, e, q, a) and the KU Leuven
                catalog (disc class, luminosity, R_in from VLTI).
  * L(D|theta): Gaussian chi^2 between the synthetic surface-density map /
                line profile and the (resolved ALMA) data.

Two samplers are provided:
  1. HMC / NUTS  (numpyro)  — fast, gradient-based, good for unimodal posteriors.
  2. Nested Sampling (dynesty) — robust to multimodality, also gives the evidence
     for model comparison (e.g. disc-dominated vs outflow-dominated).

The forward model is the JAX surrogate in forward_model.py, so gradients flow
through the chi^2 for HMC.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from scipy.ndimage import gaussian_filter

from forward_model import project_to_sky, line_profile, PARAM_NAMES, N_PARAMS

numpyro.set_host_device_count(1)


# ----------------------------------------------------------------------------
# Priors (Gaia DR3 NSS + KU Leuven informed)
# ----------------------------------------------------------------------------
# Each entry: (dist_name, *args).  These are deliberately *informative but wide*
# for the orbital parameters (Gaia/KU Leuven) and *weakly informative* for the
# mass-loss rates (which is what we want to infer).
PRIOR_SPEC = {
    "log10_Mdot_disc": ("uniform", -8.5, -4.0),
    "log10_Mdot_wind": ("uniform", -9.0, -4.0),
    "alpha_deg":       ("truncnorm", 5.0, 30.0, 14.0, 6.0),   # KU Leuven discs
    "q":               ("truncnorm", 0.1, 1.0, 0.6, 0.25),    # Gaia NSS mass ratio
    "log10_a_AU":      ("truncnorm", np.log10(2.0), np.log10(40.0),
                        np.log10(8.0), 0.25),                 # Gaia/KU Leuven a
    "e":               ("truncnorm", 0.0, 0.7, 0.2, 0.15),    # Gaia NSS eccentricity
    "inc_deg":         ("uniform", 0.0, 90.0),
    "log10_Rin_AU":    ("truncnorm", np.log10(5.0), np.log10(80.0),
                        np.log10(25.0), 0.15),                # VLTI inner rim
    "log10_Rout_AU":   ("truncnorm", np.log10(300.0), np.log10(1500.0),
                        np.log10(900.0), 0.2),                # ALMA disc extent
    "v_wind_kms":      ("truncnorm", 30.0, 400.0, 130.0, 60.0),
    "M1_msun":         ("truncnorm", 0.5, 0.9, 0.62, 0.08),   # post-AGB core mass
    "log10_d_kpc":     ("truncnorm", np.log10(0.5), np.log10(4.0),
                        np.log10(1.3), 0.2),                  # Gaia distance
}


def _sample_prior(name, spec):
    kind = spec[0]
    if kind == "uniform":
        return numpyro.sample(name, dist.Uniform(spec[1], spec[2]))
    if kind == "truncnorm":
        lo, hi, loc, scale = spec[1], spec[2], spec[3], spec[4]
        return numpyro.sample(
            name,
            dist.TruncatedNormal(loc=loc, scale=scale, low=lo, high=hi),
        )
    raise ValueError(kind)


def prior_sample_all():
    return [ _sample_prior(n, PRIOR_SPEC[n]) for n in PARAM_NAMES ]


# ----------------------------------------------------------------------------
# Likelihood
# ----------------------------------------------------------------------------
def _model_map(theta, obs):
    theta = jnp.asarray(theta)
    Sigma, _, _ = project_to_sky(
        theta, nx=obs["Sigma_obs"].shape[0], ny=obs["Sigma_obs"].shape[1],
        half_width_au=obs["half_width_au"],
    )
    # apply the synthesised beam
    pix_au = obs["pix_au"]
    sigma_pix = (obs["beam_fwhm_au"] / 2.3548) / pix_au
    # differentiable Gaussian conv via FFT-free approach: use jax.scipy
    from jax.scipy.ndimage import map_coordinates  # not used; fallback below
    # jax.scipy.ndimage has no gaussian_filter; approximate with separable conv
    import jax.scipy.signal as jss
    k = int(6 * sigma_pix) | 1
    x = jnp.arange(k) - k // 2
    g = jnp.exp(-0.5 * (x / sigma_pix) ** 2)
    g = g / g.sum()
    Sigma = jss.convolve(Sigma, g[:, None], mode="same")
    Sigma = jss.convolve(Sigma, g[None, :], mode="same")
    return Sigma


def numpyro_model(obs):
    theta = prior_sample_all()
    theta = jnp.stack(theta)
    Sigma_mod = _model_map(theta, obs)
    numpyro.sample(
        "obs_map",
        dist.Normal(Sigma_mod, obs["Sigma_err"]),
        obs=obs["Sigma_obs"],
    )


# ----------------------------------------------------------------------------
# HMC / NUTS
# ----------------------------------------------------------------------------
def run_hmc(obs, num_warmup=500, num_samples=1000, seed=0):
    kernel = NUTS(numpyro_model, target_accept_prob=0.9)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(rng_key, obs=obs)
    samples = mcmc.get_samples()
    return {name: np.asarray(samples[name]) for name in PARAM_NAMES}, mcmc


# ----------------------------------------------------------------------------
# Nested Sampling (dynesty) — likelihood in numpy space
# ----------------------------------------------------------------------------
def _loglikelihood_dynesty(theta_vec, obs):
    try:
        Sigma_mod = _model_map(jnp.asarray(theta_vec), obs)
        resid = (np.asarray(Sigma_mod) - obs["Sigma_obs"]) / obs["Sigma_err"]
        chi2 = np.sum(resid**2)
        return -0.5 * chi2
    except Exception:
        return -1e30


def _prior_transform_dynesty(u):
    """Map unit cube -> physical parameters using the PRIOR_SPEC."""
    from scipy.stats import norm, uniform
    out = np.empty(N_PARAMS)
    for i, name in enumerate(PARAM_NAMES):
        spec = PRIOR_SPEC[name]
        if spec[0] == "uniform":
            out[i] = uniform.ppf(u[i], loc=spec[1], scale=spec[2] - spec[1])
        else:
            lo, hi, loc, scale = spec[1], spec[2], spec[3], spec[4]
            a, b = (lo - loc) / scale, (hi - loc) / scale
            from scipy.stats import truncnorm
            out[i] = truncnorm.ppf(u[i], a, b, loc=loc, scale=scale)
    return out


def run_nested(obs, nlive=200, seed=0):
    import dynesty
    rstate = np.random.default_rng(seed)
    sampler = dynesty.NestedSampler(
        lambda t: _loglikelihood_dynesty(t, obs),
        _prior_transform_dynesty,
        ndim=N_PARAMS,
        nlive=nlive,
        sample="rwalk",
        rstate=rstate,
    )
    sampler.run_nested(dlogz=0.5, print_progress=False)
    res = sampler.results
    # equal-weight posterior samples
    from dynesty.utils import resample_equal
    try:
        w = res.importance_weights()
    except Exception:
        w = np.exp(res.logwt - res.logwt.max())
        w = w / w.sum()
    samples = resample_equal(res.samples, w)
    return {name: samples[:, i] for i, name in enumerate(PARAM_NAMES)}, res
