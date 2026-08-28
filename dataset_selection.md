# Dataset Verification & Selection Notes

## Verified dataset landscape (2026-08)

### 1. ALMA Science Archive — VERIFIED, PRIMARY
- Public archive: https://almascience.eso.org / https://almascience.nrao.edu/alma-data/archive
- Programmatic access: `astroquery.alma` (query by source name / region / project code, stage FITS cubes).
- Resolved post-AGB CO work exists and is the *only* dataset class that directly yields
  resolved gas surface density + velocity fields for mass-loss-per-unit-area estimation:
  - Bujarrabal et al. 2018 (A&A 614, A58): ALMA 12CO/13CO J=3-2 maps of IRAS 08544-4431
    (archetypal post-AGB circumbinary disc + outflow; disc size ~4e16 cm).
  - Gallardo Cava et al. 2021: CO interferometric modelling of 7 post-AGB sources;
    gas discs extend to ~1000 AU; slow outflows 5-10 km/s; disc-dominated vs
    outflow-dominated dichotomy (>85% of nebular mass in disc for disc-dominated).
  - Bujarrabal et al. 2013, 2016, 2017: single-dish + interferometric CO of the class.
- Data products: spectral cubes (FITS), ~0.1-1 arcsec resolution, 12m array.
  At d ~ 1-2 kpc, 0.1" ~ 100-200 AU -> resolved annuli/patches for per-unit-area Mdot.

### 2. VLTI (PIONIER / MATISSE) — VERIFIED, BOUNDARY CONDITIONS
- Corporaal et al. 2021 (A&A 650, L13): multi-wavelength VLTI study of the puffed-up
  inner rim of a post-AGB circumbinary disc (PIONIER H-band + MATISSE L/M/N).
- Corporaal et al. 2023 (A&A 674): MATISSE confirms transition-disc nature of 6 systems
  (inner cavity larger than dust sublimation radius).
- Kluska et al. 2019, Andrych et al. 2023/2024: inner-rim morphology, ~1-3 mas resolution
  -> ~1-5 AU at 1 kpc. Defines inner boundary condition (R_in, sublimation radius).
- Access: ESO Science Archive (archive.eso.org), OIFITS format.

### 3. Gaia DR3 Non-Single Star (NSS) catalog — VERIFIED, PRIORS
- >800,000 multi-star solutions; 186,905 spectroscopic (SB1/SB2) orbital solutions;
  87,073 eclipsing; astrometric orbits (Halbwachs et al. 2023, A&A 674, A9).
- Provides P, e, mass function, (for astrometric) inclination -> binary priors.
- Access: ESA Gaia Archive (gea.esac.esa.int), table `gaiadr3.nss_two_body_orbit`,
  `gaiadr3.nss_acceleration_astro`, ADQL.
- Caveat: most post-AGB periods (100-3000 d) are at the long-P edge for Gaia DR3
  astrometry; many post-AGB orbits come from ground-based RV (HERMES/Mercator) instead.

### 4. KU Leuven post-AGB catalog (Kluska et al. 2022) — VERIFIED, TARGET LIST + CONTEXT
- Kluska et al. 2022 (A&A 658, A36): ~85 Galactic post-AGB disc systems; SED
  classification (full / transition / edge-on / minimal); luminosities, distances,
  reddening, depletion. Moltzer et al. 2025: 38/85 below RGB tip (post-RGB).
- Oomen et al. 2018/2019: orbital parameters for ~33 systems (P 100-3000 d, e).
- Van Winckel 2025 (Galaxies 13, 68): review of the class; jets as MHD disc winds;
  jet mass-ejection rates 1e-4 to 1e-7 Msun/yr; 16 objects with geometric jet models.
- Access: VizieR catalog J/A+A/658/A36 + KU Leuven group pages.

## Selection for the hybrid approach

**Primary dataset: ALMA Science Archive** (resolved 12CO/13CO cubes).
Reason: it is the only one of the four that directly measures the resolved gas surface
density Sigma(R,phi) and velocity field v(R,phi) needed for the per-unit-area mass-loss
estimate the user wants to extrapolate. It supplies the likelihood L(D|theta).

**Priors: Gaia DR3 NSS + KU Leuven catalog** (orbital P, e, q, a; disc class; luminosity).
**Inner boundary: VLTI** (R_in / sublimation radius) — folded in as a prior on R_in.

Hybrid = classical Bayesian inference (HMC/NUTS + Nested Sampling) on a parameterized
anisotropic disc+wind forward model, with a quantum layer (QUBO for discretized model
selection / configuration optimization; VQE for the ground-state of an effective
interaction Hamiltonian that encodes the binary-disc coupling energy landscape).
