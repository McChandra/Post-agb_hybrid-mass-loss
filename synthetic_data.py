"""
synthetic_data.py
=================
Generate a synthetic resolved observation (surrogate ALMA CO cube) from a known
"truth" parameter vector, then degrade it with realistic noise and a synthesised
beam.  This lets us validate the inference pipeline end-to-end before pointing it
at real ALMA archive cubes.

Also provided: a loader stub for real ALMA FITS cubes via astropy, so the same
pipeline can ingest archive data once a target (e.g. IRAS 08544-4431) is chosen.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from scipy.ndimage import gaussian_filter

from forward_model import project_to_sky, line_profile, PARAM_NAMES


# ----------------------------------------------------------------------------
# A fiducial "truth" system modelled on IRAS 08544-4431 / the Gallardo Cava sample
# ----------------------------------------------------------------------------
TRUTH = {
    "log10_Mdot_disc": -6.3,     # Msun/yr  (disc-dominated system)
    "log10_Mdot_wind": -6.9,     # Msun/yr
    "alpha_deg": 12.0,           # disc half-opening angle
    "q": 0.55,                   # mass ratio
    "log10_a_AU": np.log10(8.0), # 8 AU semi-major axis
    "e": 0.22,                   # eccentricity
    "inc_deg": 68.0,             # near edge-on-ish
    "log10_Rin_AU": np.log10(25.0),   # 25 AU inner rim (VLTI)
    "log10_Rout_AU": np.log10(900.0), # 900 AU outer radius (ALMA)
    "v_wind_kms": 130.0,         # fast bipolar outflow
    "M1_msun": 0.62,             # post-AGB core mass
    "log10_d_kpc": np.log10(1.3),     # 1.3 kpc
}


def truth_vector() -> np.ndarray:
    return np.array([TRUTH[k] for k in PARAM_NAMES], dtype=float)


def synthesize_observation(
    theta=None,
    nx: int = 64,
    ny: int = 64,
    half_width_au: float = 1200.0,
    beam_fwhm_au: float = 120.0,
    noise_frac: float = 0.05,
    seed: int = 0,
):
    """
    Produce a noisy, beam-convolved surface-density map + integrated line profile.

    Returns a dict with:
      Sigma_obs  : (nx, ny) observed surface density (Msun/AU^2)
      Sigma_err  : (nx, ny) 1-sigma noise map
      v_los_true : (nx, ny) noiseless LOS velocity (for reference)
      v_cent     : (nv,) line-profile velocity centres
      F_obs      : (nv,) observed line profile
      F_err      : (nv,) 1-sigma line-profile noise
      aux_true   : derived scalars of the truth model
    """
    rng = np.random.default_rng(seed)
    if theta is None:
        theta = truth_vector()
    theta = jnp.asarray(theta, dtype=jnp.float64)

    Sigma_true, v_los_true, aux_true = project_to_sky(theta, nx, ny, half_width_au)
    Sigma_true = np.asarray(Sigma_true)
    v_los_true = np.asarray(v_los_true)

    # Beam convolution (Gaussian synthesised beam)
    pix_au = 2.0 * half_width_au / nx
    sigma_pix = (beam_fwhm_au / 2.3548) / pix_au
    Sigma_beam = gaussian_filter(Sigma_true, sigma=sigma_pix)

    # Noise: fractional + floor
    floor = 0.02 * Sigma_beam.max()
    Sigma_err = noise_frac * np.abs(Sigma_beam) + floor
    Sigma_obs = Sigma_beam + rng.normal(0.0, Sigma_err)

    # Line profile
    v_cent, F_true, _ = line_profile(theta, nx=nx, ny=ny, half_width_au=half_width_au)
    F_true = np.asarray(F_true)
    F_beam = gaussian_filter(F_true, sigma=1.0)
    F_err = 0.05 * np.abs(F_beam) + 0.02 * F_beam.max()
    F_obs = F_beam + rng.normal(0.0, F_err)

    return {
        "Sigma_obs": Sigma_obs,
        "Sigma_err": Sigma_err,
        "Sigma_true": Sigma_true,
        "v_los_true": v_los_true,
        "v_cent": np.asarray(v_cent),
        "F_obs": F_obs,
        "F_err": F_err,
        "aux_true": {k: float(np.asarray(v)) for k, v in aux_true.items()},
        "pix_au": pix_au,
        "half_width_au": half_width_au,
        "beam_fwhm_au": beam_fwhm_au,
    }


def load_alma_cube(fits_path: str):
    """
    Stub loader for a real ALMA CO cube from the Science Archive.

    In practice you would:
      1. query:  astroquery.alma.Alma.query_object('IRAS 08544-4431')
      2. stage & download the 12CO / 13CO J=3-2 or J=2-1 cubes (FITS)
      3. collapse over velocity -> moment-0 (integrated intensity) map
      4. convert to surface density with an X_CO / excitation model
      5. return in the same dict format as synthesize_observation()

    Here we simply read a FITS image and return it; the X_CO conversion is left
    as a user-supplied calibration because it is source-dependent.
    """
    from astropy.io import fits

    with fits.open(fits_path) as hdul:
        data = np.squeeze(hdul[0].data)
        hdr = hdul[0].header
    mom0 = np.nansum(data, axis=0) if data.ndim == 3 else data
    return {"Sigma_obs": mom0, "header": hdr}
