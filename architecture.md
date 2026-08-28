# Architecture of the Hybrid Classical–Quantum Post-AGB Mass-Loss Model

**Author:** Manus AI
**Date:** 2026-08-15

---

## 1. Design goal

The model infers the **total mass-loss rate** of a post-AGB binary and its
partition into an equatorial **circumbinary disc** and a polar **bipolar
outflow**, starting from a *resolved* patch of the envelope and extrapolating to
the full envelope **without assuming spherical symmetry**.  Binary interactions
are built into the forward model through the orbital parameters
$(q, a, e)$, which set the inner truncation radius and the Roche-lobe geometry.

## 2. High-level data flow

```
        ┌──────────────────────────────────────────────────────────┐
        │                     DATA LAYER                           │
        │  ALMA (primary, likelihood)   Gaia DR3 NSS + KU Leuven   │
        │  VLTI (inner boundary R_in)   (priors on q, a, e, α)     │
        └───────────────┬──────────────────────────┬───────────────┘
                        │ resolved Σ(R,φ), v(R,φ)  │ priors
                        ▼                          ▼
        ┌──────────────────────────────────────────────────────────┐
        │              CLASSICAL PROBABILISTIC MODEL               │
        │                                                          │
        │  Forward model (JAX, differentiable)                     │
        │    ρ(r,θ,φ) = ρ_disc + ρ_wind                            │
        │    → Σ(x,y), v_los(x,y), line profile, dMdot/dA map      │
        │                                                          │
        │  Inference engine                                        │
        │    P(θ|D) ∝ L(D|θ) P(θ)                                  │
        │    • HMC / NUTS   (numpyro)                              │
        │    • Nested Sampling (dynesty)  → posterior + ln Z       │
        └───────────────┬──────────────────────────────────────────┘
                        │ posterior samples / means
                        ▼
        ┌──────────────────────────────────────────────────────────┐
        │                   QUANTUM LAYER                          │
        │  QUBO  : discrete model selection (one-hot groups)       │
        │          solved exactly, by simulated annealing, and by  │
        │          Ising-mapped VQE                                │
        │  VQE   : ground state of effective interaction           │
        │          Hamiltonian H = Σ h_i Z_i + Σ J_ij Z_iZ_j       │
        │          + Σ g_i X_i  built from posterior means         │
        └───────────────┬──────────────────────────────────────────┘
                        ▼
        ┌──────────────────────────────────────────────────────────┐
        │              DERIVED PRODUCTS                            │
        │  Mdot_disc, Mdot_wind, Mdot_total, dMdot/dA map,         │
        │  disc mass, a_crit, R_roche, anisotropy ratio,           │
        │  selected configuration, quantum cross-check             │
        └──────────────────────────────────────────────────────────┘
```

## 3. Module map

| Module | Responsibility | Key public functions |
|--------|----------------|----------------------|
| `src/forward_model.py` | Differentiable anisotropic forward model | `project_to_sky`, `line_profile`, `mass_flux_map` |
| `src/synthetic_data.py` | Surrogate ALMA observation + real-cube loader | `synthesize_observation`, `load_alma_cube`, `truth_vector` |
| `src/inference.py` | Bayesian inference (priors, likelihood, samplers) | `run_hmc`, `run_nested`, `PRIOR_SPEC` |
| `src/quantum_layer.py` | QUBO + VQE | `build_qubo_from_chi2`, `qubo_bruteforce`, `qubo_simulated_annealing`, `solve_qubo_with_vqe`, `build_interaction_hamiltonian`, `run_vqe` |
| `src/run_pipeline.py` | End-to-end driver | `main` |
| `notebook.ipynb` | Executable walkthrough | — |

## 4. Parameter vector

The 12-dimensional state vector $\theta$ is

| # | Parameter | Meaning | Prior source |
|---|-----------|---------|--------------|
| 1 | $\log_{10}\dot M_{\rm disc}$ | disc mass-loss rate | weakly informative |
| 2 | $\log_{10}\dot M_{\rm wind}$ | wind mass-loss rate | weakly informative |
| 3 | $\alpha$ | disc half-opening angle | KU Leuven disc geometry |
| 4 | $q$ | binary mass ratio $M_2/M_1$ | Gaia DR3 NSS |
| 5 | $\log_{10} a$ | semi-major axis | Gaia DR3 NSS / KU Leuven |
| 6 | $e$ | eccentricity | Gaia DR3 NSS |
| 7 | $i$ | inclination | geometric |
| 8 | $\log_{10} R_{\rm in}$ | disc inner radius | VLTI |
| 9 | $\log_{10} R_{\rm out}$ | disc outer radius | ALMA |
| 10 | $v_{\rm wind}$ | outflow speed | ALMA / jet modelling |
| 11 | $M_1$ | post-AGB core mass | stellar evolution |
| 12 | $\log_{10} d$ | distance | Gaia |

## 5. Forward model

The density field is the sum of two components:

* **Disc.**  A power-law surface density $\Sigma_{\rm disc}(R) = \Sigma_0 (R/R_{\rm ref})^{-1}$,
  truncated to $[R_{\rm in}, R_{\rm out}]$ and normalised so that the radial mass
  flux $2\pi R\,\Sigma\,v_R$ (with $v_R = \epsilon v_{\rm kep}$, $\epsilon = 10^{-3}$)
  equals $\dot M_{\rm disc}$.
* **Wind.**  A biconical outflow with mass-conserving density
  $\rho_{\rm wind}(r) = \dot M_{\rm wind}/(\Omega r^2 v_{\rm wind})$,
  occupying the polar cones outside the disc opening angle.

Binary interactions enter through the **critical semi-major axis** for disc
truncation (Holman & Wiegert 1999) and the **Roche-lobe radius** (Eggleton
1983), both computed from $(q, a, e)$.

The model is projected onto the sky at inclination $i$ and convolved with the
synthesised beam to produce the synthetic surface-density map that enters the
$\chi^2$ likelihood.

## 6. Inference engine

The posterior is

$$P(\theta\,|\,D) \propto \mathcal{L}(D\,|\,\theta)\,P(\theta),$$

with a Gaussian likelihood
$\ln\mathcal{L} = -\tfrac12\sum_{\rm pix} [(\Sigma_{\rm mod}-\Sigma_{\rm obs})/\sigma]^2$.
Two samplers are used:

* **HMC/NUTS** exploits the JAX gradients of the forward model for fast,
  efficient exploration of the main posterior mode.
* **Nested Sampling** explores the full (possibly multimodal) posterior and
  returns the evidence $\ln Z$, enabling Bayesian model comparison
  (e.g. disc-dominated vs outflow-dominated).

## 7. Quantum layer

### 7.1 QUBO model selection

Discrete modelling choices (disc- vs outflow-dominated, jet on/off, full vs
transition disc, high vs low eccentricity) are encoded as binary variables.
The QUBO matrix has the per-choice $\chi^2$ on the diagonal and one-hot group
penalties $P(\sum_{i\in g} x_i - 1)^2$ on the off-diagonals to enforce mutual
exclusivity.  It is solved three ways: exact enumeration, simulated annealing,
and an Ising-mapped VQE.

### 7.2 VQE interaction Hamiltonian

An effective qubit Hamiltonian

$$H = \sum_i h_i Z_i + \sum_{i<j} J_{ij} Z_i Z_j + \sum_i g_i X_i$$

is built with coefficients derived from the posterior means.  Its ground state,
found with a hardware-efficient `EfficientSU2` ansatz and the COBYLA optimiser
on a statevector simulator, encodes the most probable large-scale configuration
and provides a quantum-consistency cross-check on the classical posterior.

## 8. Validation

On a synthetic observation of a fiducial IRAS 08544-4431-like system
($\dot M_{\rm total} = 6.3\times10^{-7}\,M_\odot\,{\rm yr}^{-1}$), the pipeline
recovers $\dot M_{\rm total} = (6.9\pm4.9)\times10^{-7}\,M_\odot\,{\rm yr}^{-1}$
(Nested Sampling), correctly identifies the system as disc-dominated with a jet
via the QUBO, and returns a per-unit-area mass-flux map that integrates to
$\dot M_{\rm total}$ by construction.
