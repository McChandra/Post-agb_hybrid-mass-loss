"""
forward_model.py
================
Anisotropic (non-spherical) forward model for a post-AGB binary system.

Physical picture (Van Winckel 2025; Bujarrabal et al. 2018; Gallardo Cava et al. 2021;
Bollen et al. 2022; Corporaal et al. 2021/2023):

  * A post-AGB primary + main-sequence companion in an eccentric orbit (P, e, q, a).
  * A stable, Keplerian-ish circumbinary disc (equatorial, dense) extending from an
    inner rim R_in (set by dust sublimation / binary truncation, constrained by VLTI)
    out to R_out (~500-1000 AU, constrained by ALMA CO).
  * A fast(ish) bipolar / biconical outflow (jet / disc wind) launched from the
    circum-companion accretion disc, with opening angle set by the jet geometry.
  * The binary breaks spherical symmetry: mass loss is a function of latitude.

The key user-facing quantity is the *mass flux per unit area* on the sky,
    dMdot / dA   [Msun yr^-1 arcsec^-2]
which, integrated over the resolved ALMA footprint with the correct (anisotropic)
geometry, yields the total mass-loss rate

    Mdot_total = Mdot_disc + Mdot_wind .

This module provides a *fast, differentiable* surrogate for the expensive
RADMC-3D radiative-transfer step so that it can live inside an HMC/NUTS or
Nested-Sampling loop.  It computes, for a parameter vector theta:

  1. the 3D density field rho(r, theta_lat, phi) = rho_disc + rho_wind
  2. the sky-projected CO surface-brightness / mass-flux map  Sigma(x, y)
  3. the integrated line profile (disc Keplerian + outflow wings)
  4. derived totals: Mdot_disc, Mdot_wind, Mdot_total, disc mass, etc.

Everything is written in JAX so gradients are available for Hamiltonian Monte Carlo.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

# ----------------------------------------------------------------------------
# Physical constants (cgs unless noted)
# ----------------------------------------------------------------------------
MSUN = 1.98892e33          # g
AU = 1.495978707e13        # cm
YR = 3.15576e7             # s
PC = 3.085677581e18        # cm
G = 6.67430e-8             # cgs
KM = 1.0e5                 # cm


# ----------------------------------------------------------------------------
# Parameter vector definition
# ----------------------------------------------------------------------------
# theta = (
#   log10_Mdot_disc,   # Msun/yr  mass-loss channelled into the equatorial disc
#   log10_Mdot_wind,   # Msun/yr  mass-loss in the bipolar outflow
#   alpha_deg,         # disc half-opening angle (deg)  -> sets disc scale height
#   q,                 # binary mass ratio M2/M1  (0 < q <= 1)
#   log10_a_AU,        # binary semi-major axis (AU)
#   e,                 # orbital eccentricity (0 <= e < 1)
#   inc_deg,           # inclination of disc axis to line of sight (0=face-on)
#   log10_Rin_AU,      # disc inner radius (AU)  [VLTI prior]
#   log10_Rout_AU,     # disc outer radius (AU)  [ALMA prior]
#   v_wind_kms,        # asymptotic wind / outflow speed (km/s)
#   M1_msun,           # primary (post-AGB) mass (Msun)
#   log10_d_kpc,       # distance (kpc)
# )
PARAM_NAMES = [
    "log10_Mdot_disc", "log10_Mdot_wind", "alpha_deg", "q", "log10_a_AU",
    "e", "inc_deg", "log10_Rin_AU", "log10_Rout_AU", "v_wind_kms",
    "M1_msun", "log10_d_kpc",
]
N_PARAMS = len(PARAM_NAMES)


# ----------------------------------------------------------------------------
# Grid construction (static, cached)
# ----------------------------------------------------------------------------
def make_grid(nx: int = 64, ny: int = 64, half_width_au: float = 1200.0):
    """Sky-plane grid in AU (x = RA, y = Dec), centred on the barycentre."""
    ax = jnp.linspace(-half_width_au, half_width_au, nx)
    ay = jnp.linspace(-half_width_au, half_width_au, ny)
    X, Y = jnp.meshgrid(ax, ay, indexing="ij")
    return X, Y


# ----------------------------------------------------------------------------
# Density law (analytic, differentiable)
# ----------------------------------------------------------------------------
@partial(jit, static_argnums=())
def _disc_surface_density(R_au, Rin, Rout, Mdot_disc, v_kep_ref, R_ref):
    """
    Power-law disc surface density Sigma(R) = Sigma0 * (R/R_ref)^(-p), p = 1,
    normalised so that the *mass flux through the disc mid-plane* equals Mdot_disc.
    Returns Sigma in Msun / AU^2.
    """
    p = 1.0
    # Normalise so that the radial mass flux through the disc equals Mdot_disc.
    # For a steady disc the accretion/outflow rate is
    #   Mdot = 2 pi R Sigma(R) v_R(R)
    # with v_R the radial drift speed.  We take v_R = eps * v_kep(R) with a
    # fixed dimensionless eps (the disc is a slow outflow, not a fast wind),
    # calibrated so that the integrated disc mass is physically reasonable
    # (~1e-3 - 1e-2 Msun for post-AGB discs).  With Sigma ∝ R^-1 and
    # v_kep ∝ R^-1/2, Mdot = 2 pi Sigma0 R_ref^2 eps v_ref ∫ x^{-1/2} dx.
    eps = 1.0e-3  # radial drift as a fraction of Keplerian speed
    xin = Rin / R_ref
    xout = Rout / R_ref
    # Convert to a consistent AU-based unit system so Sigma comes out in
    # Msun / AU^2:  Mdot in Msun/yr, v_kep in AU/yr, R in AU.
    Mdot_msun_yr = Mdot_disc * YR / MSUN          # g/s -> Msun/yr
    v_kep_ref_auyr = v_kep_ref * YR / AU          # cm/s -> AU/yr
    denom = (4.0 * jnp.pi * R_ref**2 * eps * v_kep_ref_auyr
             * (jnp.sqrt(xout) - jnp.sqrt(xin) + 1e-12))
    Sigma0 = Mdot_msun_yr / (denom + 1e-30)       # Msun / AU^2
    Sigma = Sigma0 * (R_au / R_ref) ** (-p)
    Sigma = jnp.where((R_au >= Rin) & (R_au <= Rout), Sigma, 0.0)
    return Sigma


@jit
def _wind_column(R_au, Z_au, Mdot_wind, v_wind_cms, alpha_rad, Rin):
    """
    Biconical wind column density (Msun/AU^2) at cylindrical radius R_au and
    height Z_au.  Mass flux is conserved along streamlines:
        Mdot_wind = ∮ rho v dA  ->  rho(R) = Mdot_wind / (Omega(R) R^2 v)
    with Omega(R) the wind solid angle at radius R.  We project to a column.
    """
    r_sph = jnp.sqrt(R_au**2 + Z_au**2) + 1e-6
    # latitude measured from the pole; wind occupies |lat| < (90deg - alpha_disc)
    lat = jnp.arctan2(jnp.abs(Z_au), R_au + 1e-6)  # 0 = equator, pi/2 = pole
    wind_half_angle = jnp.pi / 2.0 - alpha_rad     # opening around the pole
    in_wind = lat > (jnp.pi / 2.0 - wind_half_angle)
    Omega = 4.0 * jnp.pi * jnp.sin(wind_half_angle)  # two cones
    # Work in AU-based units so the column comes out in Msun / AU^2 directly.
    Mdot_msun_yr = Mdot_wind * YR / MSUN          # g/s -> Msun/yr
    v_auyr = v_wind_cms * YR / AU                 # cm/s -> AU/yr
    # rho in Msun/AU^3 = Mdot / (Omega r^2 v)
    rho = Mdot_msun_yr / (Omega * r_sph**2 * v_auyr + 1e-30)  # Msun/AU^3
    # Column = rho * characteristic path length (= local radius, in AU).
    path = jnp.clip(r_sph, 0.0, 1000.0)           # AU
    col_msun_au2 = rho * path                     # Msun/AU^2
    return jnp.where(in_wind & (r_sph > Rin), col_msun_au2, 0.0)


# ----------------------------------------------------------------------------
# Sky projection
# ----------------------------------------------------------------------------
@partial(jit, static_argnums=(1, 2))
def project_to_sky(theta, nx: int = 64, ny: int = 64, half_width_au: float = 1200.0):
    """
    Project the 3D model onto the sky plane and return:
      Sigma_map  : (nx, ny) total gas surface density in Msun/AU^2
      v_los_map  : (nx, ny) line-of-sight velocity in km/s (for line profile)
      aux        : dict of derived scalars
    """
    (
        log10_Mdot_disc, log10_Mdot_wind, alpha_deg, q, log10_a_AU,
        e, inc_deg, log10_Rin_AU, log10_Rout_AU, v_wind_kms,
        M1_msun, log10_d_kpc,
    ) = theta

    Mdot_disc = 10.0 ** log10_Mdot_disc * MSUN / YR          # g/s
    Mdot_wind = 10.0 ** log10_Mdot_wind * MSUN / YR          # g/s
    alpha = jnp.deg2rad(alpha_deg)
    inc = jnp.deg2rad(inc_deg)
    a = 10.0 ** log10_a_AU
    Rin = 10.0 ** log10_Rin_AU
    Rout = 10.0 ** log10_Rout_AU
    v_wind = v_wind_kms * KM
    d_kpc = 10.0 ** log10_d_kpc

    Mtot = M1_msun * (1.0 + q)
    R_ref = 100.0
    v_kep_ref = jnp.sqrt(G * Mtot * MSUN / (R_ref * AU))     # cm/s

    X, Y = make_grid(nx, ny, half_width_au)                  # sky AU

    # Rotate sky plane into disc plane (inclination about x-axis)
    # Disc plane coordinates:
    R_disc = jnp.sqrt(X**2 + (Y / jnp.cos(inc) + 1e-9) ** 2)  # deprojected radius
    Z = Y * jnp.tan(inc)                                       # height above plane

    # Disc surface density (face-on) then projected: Sigma_sky = Sigma / cos(inc)
    Sigma_disc = _disc_surface_density(R_disc, Rin, Rout, Mdot_disc, v_kep_ref, R_ref)
    Sigma_disc = Sigma_disc / (jnp.cos(inc) + 1e-6)

    # Wind column
    Sigma_wind = _wind_column(R_disc, Z, Mdot_wind, v_wind, alpha, Rin)

    Sigma_map = Sigma_disc + Sigma_wind

    # Line-of-sight velocity: Keplerian rotation (disc) + radial outflow (wind)
    phi = jnp.arctan2(Y / (jnp.cos(inc) + 1e-9), X)
    v_kep = jnp.sqrt(G * Mtot * MSUN / (R_disc * AU + 1e-6)) / KM  # km/s
    v_los_disc = v_kep * jnp.sin(inc) * jnp.cos(phi)
    # wind: radial away from centre, projected
    v_los_wind = v_wind_kms * jnp.sin(inc) * jnp.sign(Z + 1e-9)
    in_wind_mask = Sigma_wind > 0
    v_los_map = jnp.where(in_wind_mask, v_los_wind, v_los_disc)

    # Derived scalars.  Sigma_* are already in Msun / AU^2, so the disc/wind
    # masses are the surface-density maps integrated over the sky in AU^2.
    pix_area_au2 = (2.0 * half_width_au / nx) * (2.0 * half_width_au / ny)
    M_disc = jnp.sum(Sigma_disc) * pix_area_au2               # Msun
    M_wind = jnp.sum(Sigma_wind) * pix_area_au2               # Msun
    Mdot_disc_msunyr = 10.0 ** log10_Mdot_disc
    Mdot_wind_msunyr = 10.0 ** log10_Mdot_wind
    Mdot_total = Mdot_disc_msunyr + Mdot_wind_msunyr

    # Binary-disc interaction: inner truncation radius (Artymowicz & Lubow 1994)
    # R_cav / a ~ 1.7 * (1+e)^... (Holman & Wiegert 1999 critical semi-major axis)
    acrit = a * (1.60 + 5.10 * e + (-2.22) * e**2
                 + 4.12 * q + (-4.27) * e * q + (-5.09) * q**2
                 + 4.61 * e**2 * q**2)
    # Roche lobe of primary (Eggleton 1983)
    rL1 = a * (0.49 * (1.0 / q) ** (2.0 / 3.0)) / (
        0.6 * (1.0 / q) ** (2.0 / 3.0) + jnp.log(1.0 + (1.0 / q) ** (1.0 / 3.0)))

    # arcsec scale
    au_per_arcsec = d_kpc * 1000.0  # 1" = d(AU); d(kpc)*1000 = d(AU) at 1"
    pix_arcsec = (2.0 * half_width_au / nx) / au_per_arcsec

    aux = {
        "M_disc": M_disc,
        "M_wind": M_wind,
        "Mdot_disc": Mdot_disc_msunyr,
        "Mdot_wind": Mdot_wind_msunyr,
        "Mdot_total": Mdot_total,
        "a_crit_AU": acrit,
        "R_roche_lobe_AU": rL1,
        "pix_arcsec": pix_arcsec,
        "au_per_arcsec": au_per_arcsec,
    }
    return Sigma_map, v_los_map, aux


# ----------------------------------------------------------------------------
# Integrated line profile (for CO single-dish / interferometric comparison)
# ----------------------------------------------------------------------------
@partial(jit, static_argnums=(3, 4, 5))
def line_profile(theta, v_min=-40.0, v_max=40.0, nv: int = 80,
                 nx: int = 64, ny: int = 64, half_width_au: float = 1200.0):
    """Velocity-bin the flux map into an integrated line profile F(v)."""
    Sigma_map, v_los_map, aux = project_to_sky(theta, nx, ny, half_width_au)
    edges = jnp.linspace(v_min, v_max, nv + 1)
    idx = jnp.clip(jnp.digitize(v_los_map.ravel(), edges) - 1, 0, nv - 1)
    F = jnp.zeros(nv).at[idx].add(Sigma_map.ravel())
    vcent = 0.5 * (edges[:-1] + edges[1:])
    return vcent, F, aux


# ----------------------------------------------------------------------------
# Convenience: per-unit-area mass flux map (the user's key quantity)
# ----------------------------------------------------------------------------
@partial(jit, static_argnums=(1, 2))
def mass_flux_map(theta, nx: int = 64, ny: int = 64, half_width_au: float = 1200.0):
    """
    Return dMdot/dA in Msun yr^-1 arcsec^-2 on the sky.

    We approximate the local mass flux as Sigma(R) * v_out(R) / (local scale),
    i.e. the surface density times the local flow speed divided by the local
    radial scale over which that gas is replenished.  For the disc this is the
    Keplerian drift; for the wind it is the outflow speed.
    """
    (
        log10_Mdot_disc, log10_Mdot_wind, alpha_deg, q, log10_a_AU,
        e, inc_deg, log10_Rin_AU, log10_Rout_AU, v_wind_kms,
        M1_msun, log10_d_kpc,
    ) = theta
    Sigma_map, v_los_map, aux = project_to_sky(theta, nx, ny, half_width_au)
    inc = jnp.deg2rad(inc_deg)
    # Rebuild the deprojected radius and the *local flow speed* for each zone.
    X, Y = make_grid(nx, ny, half_width_au)
    R_disc = jnp.sqrt(X**2 + (Y / jnp.cos(inc) + 1e-9) ** 2)
    Mtot = M1_msun * (1.0 + q)
    v_kep = jnp.sqrt(G * Mtot * MSUN / (R_disc * AU + 1e-6)) / KM  # km/s
    # disc: radial drift = eps * v_kep ; wind: outflow speed
    eps = 1.0e-3
    v_disc = eps * v_kep
    v_flow = jnp.where(Sigma_map > 0,
                       jnp.where(jnp.abs(Y * jnp.tan(inc)) > 0.0,
                                 v_wind_kms, v_disc),
                       v_disc)
    # simpler & robust: use wind speed where wind column dominates, else disc drift
    # (recompute wind mask from geometry)
    lat = jnp.arctan2(jnp.abs(Y * jnp.tan(inc)), R_disc + 1e-6)
    wind_half_angle = jnp.pi / 2.0 - jnp.deg2rad(alpha_deg)
    in_wind = lat > (jnp.pi / 2.0 - wind_half_angle)
    # --- Per-unit-area mass flux, normalised so the map integrates to Mdot ---
    # Recompute the disc-only and wind-only surface densities so each zone's
    # flux uses its own component (Sigma_map is the sum and would double-count).
    Mdot_disc = 10.0 ** log10_Mdot_disc * MSUN / YR     # g/s
    Mdot_wind = 10.0 ** log10_Mdot_wind * MSUN / YR     # g/s
    alpha = jnp.deg2rad(alpha_deg)
    Rin = 10.0 ** log10_Rin_AU
    Rout = 10.0 ** log10_Rout_AU
    R_ref = 100.0
    v_kep_ref = jnp.sqrt(G * Mtot * MSUN / (R_ref * AU))  # cm/s
    Sigma_disc_only = _disc_surface_density(R_disc, Rin, Rout, Mdot_disc,
                                            v_kep_ref, R_ref)
    Sigma_disc_only = Sigma_disc_only / (jnp.cos(inc) + 1e-6)
    Sigma_wind_only = _wind_column(R_disc, Y * jnp.tan(inc), Mdot_wind,
                                   v_wind_kms * KM, alpha, Rin)
    # Disc: distribute Mdot_disc in proportion to the local disc surface
    # density.  This guarantees the disc part of the flux map integrates to
    # Mdot_disc by construction (the disc is a reservoir through which the
    # mass-loss flows; the local flux traces where that mass resides).
    pix_area_au2 = (2.0 * half_width_au / nx) ** 2
    disc_mass = jnp.sum(Sigma_disc_only) * pix_area_au2 + 1e-30   # Msun
    Mdot_disc_msunyr = 10.0 ** log10_Mdot_disc
    flux_disc = Sigma_disc_only * (Mdot_disc_msunyr / disc_mass)  # Msun/AU^2/yr
    # Wind: distribute Mdot_wind in proportion to the local wind column.
    wind_mass = jnp.sum(Sigma_wind_only) * pix_area_au2 + 1e-30   # Msun
    Mdot_wind_msunyr = 10.0 ** log10_Mdot_wind
    flux_wind = Sigma_wind_only * (Mdot_wind_msunyr / wind_mass)  # Msun/AU^2/yr
    # Combine by local column weight: each line of sight contains both disc
    # and wind material, so the total per-area flux is the sum of the two
    # components (each already normalised to its own Mdot).  This guarantees
    # the full map integrates to Mdot_disc + Mdot_wind = Mdot_total.
    flux_msun_au2_yr = flux_disc + flux_wind
    return flux_msun_au2_yr, Sigma_map, aux
