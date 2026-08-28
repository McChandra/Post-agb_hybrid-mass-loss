# Post-agb_hybrid-mass-loss
# Hybrid Classical–Quantum Model for Post-AGB Binary Mass Loss

This repository contains an executable research prototype for inferring **anisotropic mass loss in binary post-asymptotic-giant-branch (post-AGB) systems**. It combines a differentiable physical forward model, Bayesian inference with HMC/NUTS and Nested Sampling, and a quantum-computing layer based on **QUBO optimisation and the Variational Quantum Eigensolver (VQE)**.

The central physical point is that a binary companion breaks spherical symmetry. A resolved surface patch therefore cannot be extrapolated over a sphere without modelling whether the material belongs to a dense equatorial circumbinary disc, a bipolar wind, or both. The model estimates

$$
\dot M_{\mathrm{total}}=\dot M_{\mathrm{disc}}+\dot M_{\mathrm{wind}},
$$

and constructs a resolved mass-flux map, $$\mathrm{d}\dot M/\mathrm{d}A$$, whose integral equals the inferred total mass-loss rate.

## Dataset strategy

| Dataset | Model role | Main information |
| --- | --- | --- |
| **ALMA Science Archive** | Primary likelihood | Resolved $$^{12}$$CO/$$^{13}$$CO gas surface density and velocity fields |
| **Gaia DR3 Non-Single Star catalog** | Orbital priors | Period, eccentricity, mass function, mass ratio, and semi-major axis |
| **KU Leuven post-AGB catalog** | Target list and contextual priors | Disc class, luminosity, spectroscopy, and orbital information |
| **VLTI PIONIER/MATISSE** | Inner boundary condition | Dust sublimation radius and circumbinary-disc inner rim |

**ALMA is selected as the primary dataset** because it is the only candidate that directly supplies the resolved gas distribution and kinematics needed for per-unit-area mass-loss inference. Gaia DR3 NSS and the KU Leuven catalog supply binary priors, while VLTI constrains the inner disc boundary.

## Architecture

```
ALMA resolved CO maps/cubes ───────────────┐
Gaia DR3 NSS + KU Leuven orbital priors ──┼─> Bayesian forward model
VLTI inner-rim prior ──────────────────────┘      │
                                                  ├─> HMC/NUTS posterior
                                                  ├─> Nested posterior + evidence
                                                  ├─> dMdot/dA and Mdot_total
                                                  └─> QUBO + VQE quantum layer
```

The state vector contains the disc and wind mass-loss rates, disc opening angle, mass ratio, semi-major axis, eccentricity, inclination, inner and outer radii, wind speed, primary mass, and distance. Binary interactions enter through the orbital parameters $$(q,a,e)$$ and the corresponding circumbinary truncation and Roche-lobe scales.

Full details are available in [`docs/architecture.md`](docs/architecture.md).

## Repository contents

| Path | Description |
| --- | --- |
| [`notebook.ipynb`](notebook.ipynb) | Executed, end-to-end Python notebook with saved outputs |
| [`src/forward_model.py`](src/forward_model.py) | JAX-based anisotropic disc + wind forward model |
| [`src/inference.py`](src/inference.py) | Priors, likelihood, HMC/NUTS, and Nested Sampling |
| [`src/quantum_layer.py`](src/quantum_layer.py) | QUBO builders/solvers and VQE implementation |
| [`src/synthetic_data.py`](src/synthetic_data.py) | Surrogate ALMA observation and FITS-loader entry point |
| [`src/run_pipeline.py`](src/run_pipeline.py) | Command-line end-to-end pipeline |
| [`docs/report.md`](docs/report.md) | Research report, results, limitations, and references |
| [`docs/dataset_selection.md`](docs/dataset_selection.md) | Verified dataset assessment |
| [`results/summary.json`](results/summary.json) | Machine-readable validation results |
| [`figures/`](figures/) | Posterior and model-validation figures |

## Installation

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run the complete pipeline

```bash
cd src
python run_pipeline.py
```

The pipeline generates a beam-convolved resolved surrogate observation, runs HMC/NUTS and Nested Sampling, derives the mass-loss products, solves the QUBO, runs the VQE, and writes results under `results/` and `figures/`.

The HMC and Nested-Sampling stages are computationally heavier than the forward and quantum-model demonstrations. Runtime depends on the available CPU and JAX installation.

## Run the notebook

```bash
jupyter notebook notebook.ipynb
```

The committed notebook has already been executed end-to-end and contains saved outputs. It can be rerun from the repository root after installing the dependencies.

## Validation result

The synthetic validation uses an IRAS 08544-4431-like system with an injected total rate of

$$
\dot M_{\mathrm{total,true}}=6.27\times10^{-7}\ M_\odot\,\mathrm{yr}^{-1}.
$$

Nested Sampling recovered

$$
\dot M_{\mathrm{total}}=(6.9\pm4.9)\times10^{-7}\ M_\odot\,\mathrm{yr}^{-1},
$$

which is consistent with the injected value within the posterior uncertainty. The QUBO selected a **disc-dominated, jet-on, full-disc, low-eccentricity** configuration.

![Resolved observation, posterior-mean model, and residual](https://private-us-east-1.manuscdn.com/sessionFile/3jL62AjrdYXodMDmk24VYg/sandbox/2DP74AEKSWSMJfzXjsdGG5-images_1787917113301_na1fn_L2hvbWUvdWJ1bnR1L3BhZ2JfaHlicmlkL2ZpZ3VyZXMvbWFwX2NvbXBhcmlzb24.png?Expires=1788089941&Signature=MEYCIQD~H9JCi3nCS30Te5XolCk4wPKkhHDrIToMYBTydLmgggIhAMetRkBXmaJ8wvXlSfDZRGD4csrGa0Y~4c-DVqJ6or-o&Key-Pair-Id=K1K5N5YNBUUMMN)

![Key posterior distributions](https://private-us-east-1.manuscdn.com/sessionFile/3jL62AjrdYXodMDmk24VYg/sandbox/2DP74AEKSWSMJfzXjsdGG5-images_1787917113301_na1fn_L2hvbWUvdWJ1bnR1L3BhZ2JfaHlicmlkL2ZpZ3VyZXMvY29ybmVy.png?Expires=1788089941&Signature=MEUCIQCNcIyo~gXYP9ZihMskoAkrCaTvbqGhQc-Dmz9BF~QmSwIgDcYq0cgvSSBCmFwuAy41IALHVGgk4nR3gAcBWxjMkCI_&Key-Pair-Id=K1K5N5YNBUUMMN)

## Using real ALMA data

The code includes `load_alma_cube` as an entry point for FITS data. A real-data analysis should:

1. Query the ALMA Science Archive for a selected KU Leuven post-AGB binary, such as IRAS 08544-4431.

1. Download calibrated $$^{12}$$CO and $$^{13}$$CO spectral cubes.

1. Construct moment maps and a source-specific CO excitation/abundance calibration to convert intensity into gas surface density.

1. Replace `synthesize_observation` with the calibrated observational product.

1. Insert target-specific Gaia/KU Leuven orbital priors and a VLTI inner-rim prior.

1. Fit both image-plane maps and spectral channels or line-profile wings to break the wind-speed/mass-loss degeneracy.

## Scientific limitations

This is a working **methodological prototype**, not a publication-ready radiative-transfer fit to a particular ALMA target. The JAX forward model is a fast surrogate for full 3-D line radiative transfer. For scientific application, it should be calibrated against RADMC-3D, LIME, or an equivalent solver; observational covariance and beam-channel correlations should be included; and posterior convergence should be evaluated with multiple HMC chains and larger Nested-Sampling runs. The VQE Hamiltonian is an effective decision/interaction encoding rather than a fundamental quantum description of the stellar plasma.

## Documentation

The complete analysis is in [`docs/report.md`](docs/report.md). The module and data-flow design is in [`docs/architecture.md`](docs/architecture.md), and the source assessment is in [`docs/dataset_selection.md`](docs/dataset_selection.md).

## References

1. [Van Winckel, H. (2025), *Post-AGB Binaries as Interacting Systems*](https://www.mdpi.com/2075-4434/13/3/68)

1. [Bujarrabal, V. et al. (2018), *High-resolution observations of IRAS 08544-4431*](https://www.aanda.org/articles/aa/pdf/2018/06/aa32422-17.pdf)

1. [Gaia DR3 Non-Single Stars, ESA](https://www.cosmos.esa.int/web/gaia/dr3-non-single-stars)

1. [Corporaal, A. et al. (2023), *Transition disc nature of post-AGB binary systems confirmed by mid-infrared interferometry*](https://www.aanda.org/articles/aa/abs/2023/06/aa46408-23/aa46408-23.html)

## Status

The notebook was executed successfully with **zero notebook cell errors** in the validation environment. Numerical outputs are reproducible with the included random seeds, subject to normal dependency- and platform-level floating-point variation.
