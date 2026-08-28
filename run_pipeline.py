"""
run_pipeline.py
===============
End-to-end driver for the hybrid classical-quantum post-AGB mass-loss model.

Steps
-----
1. Build the synthetic "truth" observation (surrogate ALMA CO map + line profile).
2. Run classical Bayesian inference:
     a. HMC / NUTS (numpyro)  -> posterior samples
     b. Nested Sampling (dynesty) -> posterior + evidence
3. Compute derived quantities: Mdot_disc, Mdot_wind, Mdot_total, per-unit-area
   mass-flux map, disc mass, binary interaction radii.
4. Quantum layer:
     a. QUBO model selection over a small configuration space.
     b. VQE ground-state of the effective interaction Hamiltonian built from the
        posterior means.
5. Save posterior summaries, figures, and a JSON results file.
"""

from __future__ import annotations

import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from forward_model import (
    PARAM_NAMES, project_to_sky, mass_flux_map, line_profile,
)
from synthetic_data import synthesize_observation, truth_vector, TRUTH
from inference import run_hmc, run_nested
from quantum_layer import (
    build_qubo_from_chi2, qubo_bruteforce, qubo_simulated_annealing,
    build_interaction_hamiltonian, run_vqe, most_probable_bitstring,
    solve_qubo_with_vqe,
)

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

NX = NY = 48          # keep modest for speed
HW = 1200.0           # AU half-width


def banner(txt):
    print("\n" + "=" * 70 + f"\n{txt}\n" + "=" * 70)


# ---------------------------------------------------------------------------
def step1_data():
    banner("STEP 1: synthetic resolved observation (surrogate ALMA CO)")
    obs = synthesize_observation(nx=NX, ny=NY, half_width_au=HW)
    print(f"  map shape      : {obs['Sigma_obs'].shape}")
    print(f"  beam FWHM      : {obs['beam_fwhm_au']:.0f} AU")
    print(f"  truth Mdot_disc: {obs['aux_true']['Mdot_disc']:.2e} Msun/yr")
    print(f"  truth Mdot_wind: {obs['aux_true']['Mdot_wind']:.2e} Msun/yr")
    print(f"  truth Mdot_tot : {obs['aux_true']['Mdot_total']:.2e} Msun/yr")
    print(f"  truth M_disc   : {obs['aux_true']['M_disc']:.3e} Msun")
    print(f"  truth a_crit   : {obs['aux_true']['a_crit_AU']:.1f} AU")
    print(f"  truth R_lobe   : {obs['aux_true']['R_roche_lobe_AU']:.2f} AU")
    return obs


# ---------------------------------------------------------------------------
def step2_hmc(obs):
    banner("STEP 2a: HMC / NUTS inference")
    t0 = time.time()
    samples, mcmc = run_hmc(obs, num_warmup=300, num_samples=600, seed=0)
    dt = time.time() - t0
    print(f"  HMC finished in {dt:.1f} s")
    mcmc.print_summary(exclude_deterministic=True)
    return samples


def step2_nested(obs):
    banner("STEP 2b: Nested Sampling (dynesty)")
    t0 = time.time()
    samples, res = run_nested(obs, nlive=60, seed=0)
    dt = time.time() - t0
    print(f"  Nested sampling finished in {dt:.1f} s")
    print(f"  log Z = {res.logz[-1]:.2f} +/- {res.logzerr[-1]:.2f}")
    return samples, res


# ---------------------------------------------------------------------------
def step3_derived(obs, post_samples):
    banner("STEP 3: derived mass-loss quantities from posterior")
    means = {n: float(np.mean(post_samples[n])) for n in PARAM_NAMES}
    stds = {n: float(np.std(post_samples[n])) for n in PARAM_NAMES}

    md_disc = 10 ** post_samples["log10_Mdot_disc"]
    md_wind = 10 ** post_samples["log10_Mdot_wind"]
    md_tot = md_disc + md_wind
    print(f"  Mdot_disc = {np.mean(md_disc):.2e} +/- {np.std(md_disc):.1e} Msun/yr")
    print(f"  Mdot_wind = {np.mean(md_wind):.2e} +/- {np.std(md_wind):.1e} Msun/yr")
    print(f"  Mdot_tot  = {np.mean(md_tot):.2e} +/- {np.std(md_tot):.1e} Msun/yr")
    print(f"  truth     = {obs['aux_true']['Mdot_total']:.2e} Msun/yr")

    # per-unit-area mass-flux map at the posterior mean (Msun/yr/AU^2)
    theta_mean = np.array([means[n] for n in PARAM_NAMES])
    flux_map_au2, Sigma_map, aux = mass_flux_map(theta_mean, nx=NX, ny=NY,
                                                 half_width_au=HW)
    flux_map_au2 = np.asarray(flux_map_au2)
    au_per_arcsec = aux["au_per_arcsec"]
    flux_map = flux_map_au2 * au_per_arcsec**2   # -> Msun/yr/arcsec^2
    pix_au = 2.0 * HW / NX
    pix_arcsec = pix_au / au_per_arcsec
    integ = np.nansum(flux_map_au2) * pix_au**2   # Msun/yr
    print(f"  integrated flux map = {integ:.2e} Msun/yr "
          f"(target Mdot_total ~ {np.mean(10**post_samples['log10_Mdot_disc'] + 10**post_samples['log10_Mdot_wind']):.2e})")
    print(f"  median dMdot/dA = {np.median(flux_map[flux_map>0]):.2e} Msun/yr/arcsec^2")

    # anisotropy: equatorial vs polar extrapolation
    yy, xx = np.mgrid[0:NX, 0:NY]
    cx = cy = NX / 2
    ang = np.arctan2(yy - cy, xx - cx)
    eq = (np.abs(ang) < np.pi / 6) | (np.abs(np.abs(ang) - np.pi) < np.pi / 6)
    pol = np.abs(np.abs(ang) - np.pi / 2) < np.pi / 6
    eq_flux = np.nansum(flux_map_au2 * eq) * pix_au**2
    pol_flux = np.nansum(flux_map_au2 * pol) * pix_au**2
    print(f"  equatorial-wedge Mdot proxy = {eq_flux:.2e} Msun/yr")
    print(f"  polar-wedge     Mdot proxy = {pol_flux:.2e} Msun/yr")
    print(f"  anisotropy ratio eq/pol    = {eq_flux/(pol_flux+1e-30):.2f}")

    return means, stds, flux_map, Sigma_map, aux


# ---------------------------------------------------------------------------
def step4_quantum(post_samples, means):
    banner("STEP 4: quantum layer (QUBO + VQE)")

    # ---- QUBO model selection ---------------------------------------------
    # Candidate configurations: 8 modelling choices
    labels = [
        "disc-dominated", "outflow-dominated", "jet-on", "jet-off",
        "full-disc", "transition-disc", "high-e", "low-e",
    ]
    # chi^2 proxy: distance of each choice from the posterior mean
    md_disc = np.mean(10 ** post_samples["log10_Mdot_disc"])
    md_wind = np.mean(10 ** post_samples["log10_Mdot_wind"])
    e_mean = np.mean(post_samples["e"])
    Rin_mean = 10 ** np.mean(post_samples["log10_Rin_AU"])

    chi2 = np.array([
        0.0 if md_disc > md_wind else 2.0,     # disc-dominated
        0.0 if md_wind >= md_disc else 2.0,    # outflow-dominated
        0.0 if md_wind > 3e-8 else 1.5,        # jet-on
        0.0 if md_wind <= 3e-8 else 1.5,       # jet-off
        0.0 if Rin_mean < 40 else 1.0,         # full-disc
        0.0 if Rin_mean >= 40 else 1.0,        # transition-disc
        0.0 if e_mean > 0.25 else 0.8,         # high-e
        0.0 if e_mean <= 0.25 else 0.8,        # low-e
    ])
    incompat = [(0, 1), (2, 3), (4, 5), (6, 7)]
    groups = [[0, 1], [2, 3], [4, 5], [6, 7]]  # mutually-exclusive one-hot pairs
    Q = build_qubo_from_chi2(chi2, penalty=6.0, incompat_pairs=incompat,
                             groups=groups)

    x_exact, E_exact = qubo_bruteforce(Q)
    x_sa, E_sa = qubo_simulated_annealing(Q, seed=0)
    print("  QUBO exact solution :", x_exact.astype(int), f"E={E_exact:.3f}")
    print("  QUBO annealed sol.  :", x_sa.astype(int), f"E={E_sa:.3f}")
    chosen = [labels[i] for i, b in enumerate(x_exact) if b > 0.5]
    print("  selected configuration:", chosen)

    # VQE-on-QUBO (Ising mapped)
    bitstr_q, E_q, prob_q = solve_qubo_with_vqe(Q, reps=1, maxiter=120, seed=0)
    print(f"  VQE-on-QUBO bitstring: {bitstr_q}  E={E_q:.3f}  p={prob_q:.2f}")

    # ---- VQE interaction Hamiltonian --------------------------------------
    h, J, g = build_interaction_hamiltonian(means, n_qubits=6)
    E0, params, sv, ansatz, op = run_vqe(h, J, g, reps=2, maxiter=200, seed=0)
    bitstr, prob = most_probable_bitstring(sv, 6)
    print(f"  VQE ground energy = {E0:.4f}")
    print(f"  most probable config |{bitstr}>  (p={prob:.2f})")
    meaning = ["disc-dominated", "jet-on", "transition-disc",
               "high-e", "high-q", "strong-coupling"]
    active = [meaning[i] for i, b in enumerate(bitstr[::-1]) if b == "1"]
    print("  active physical attributes:", active)

    return {
        "qubo_labels": labels,
        "qubo_exact": x_exact.astype(int).tolist(),
        "qubo_selected": chosen,
        "qubo_vqe_bitstring": bitstr_q,
        "qubo_vqe_prob": prob_q,
        "vqe_ground_energy": E0,
        "vqe_bitstring": bitstr,
        "vqe_prob": prob,
        "vqe_active": active,
    }


# ---------------------------------------------------------------------------
def step5_figures(obs, post_samples, flux_map, Sigma_map, means):
    banner("STEP 5: figures")

    # 1. data vs posterior-mean model
    theta_mean = np.array([means[n] for n in PARAM_NAMES])
    Sigma_mod, v_mod, _ = project_to_sky(theta_mean, nx=NX, ny=NY, half_width_au=HW)
    Sigma_mod = np.asarray(Sigma_mod)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, img, ttl in zip(
        axes,
        [obs["Sigma_obs"], Sigma_mod, obs["Sigma_obs"] - Sigma_mod],
        ["Observed Sigma (surrogate ALMA)", "Posterior-mean model", "Residual"],
    ):
        im = ax.imshow(np.log10(np.abs(img) + 1e-12), origin="lower", cmap="inferno")
        ax.set_title(ttl)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "map_comparison.png"), dpi=130)
    plt.close(fig)

    # 2. line profile
    v_cent, F_mod, _ = line_profile(theta_mean, nx=NX, ny=NY, half_width_au=HW)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(obs["v_cent"], obs["F_obs"], yerr=obs["F_err"], fmt=".",
                label="observed", alpha=0.7)
    ax.plot(v_cent, F_mod, "-", label="posterior-mean model")
    ax.set_xlabel("v (km/s)"); ax.set_ylabel("F (arb.)")
    ax.legend(); ax.set_title("Integrated CO line profile")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "line_profile.png"), dpi=130)
    plt.close(fig)

    # 3. corner plot (subset of parameters)
    try:
        import corner
        labs = [r"$\log\dot M_{\rm disc}$", r"$\log\dot M_{\rm wind}$",
                r"$\alpha$", r"$q$", r"$e$", r"$i$"]
        keys = ["log10_Mdot_disc", "log10_Mdot_wind", "alpha_deg", "q", "e", "inc_deg"]
        arr = np.column_stack([post_samples[k] for k in keys])
        truths = [TRUTH[k] for k in keys]
        fig = corner.corner(arr, labels=labs, truths=truths, show_titles=True,
                            title_fmt=".2f")
        fig.savefig(os.path.join(FIG, "corner.png"), dpi=130)
        plt.close(fig)
    except Exception as ex:
        print("  corner plot skipped:", ex)

    # 4. mass-flux map
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(np.log10(flux_map + 1e-16), origin="lower", cmap="viridis")
    ax.set_title(r"$\mathrm{d}\dot M/\mathrm{d}A$  "
                 r"[$M_\odot\,\mathrm{yr}^{-1}\,\mathrm{arcsec}^{-2}$]")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "mass_flux_map.png"), dpi=130)
    plt.close(fig)

    print("  figures written to", FIG)


# ---------------------------------------------------------------------------
def main():
    obs = step1_data()

    post_hmc = step2_hmc(obs)
    post_ns, ns_res = step2_nested(obs)

    # Report both posteriors; use the Nested-Sampling posterior for the derived
    # mass-loss quantities because it explores the full (possibly multimodal)
    # posterior and is less sensitive to gradient-driven mode-seeking bias.
    banner("Posterior comparison (HMC vs Nested)")
    for n in PARAM_NAMES:
        print(f"  {n:16s}  HMC {np.mean(post_hmc[n]):+.3f}±{np.std(post_hmc[n]):.3f}"
              f"   NS {np.mean(post_ns[n]):+.3f}±{np.std(post_ns[n]):.3f}"
              f"   truth {truth_vector()[PARAM_NAMES.index(n)]:+.3f}")

    means, stds, flux_map, Sigma_map, aux = step3_derived(obs, post_ns)

    quantum_results = step4_quantum(post_ns, means)

    step5_figures(obs, post_ns, flux_map, Sigma_map, means)

    # ---- save summary ------------------------------------------------------
    summary = {
        "truth": TRUTH,
        "posterior_mean": means,
        "posterior_std": stds,
        "posterior_mean_hmc": {n: float(np.mean(post_hmc[n])) for n in PARAM_NAMES},
        "derived": {
            "Mdot_disc_mean": float(np.mean(10 ** post_ns["log10_Mdot_disc"])),
            "Mdot_disc_std": float(np.std(10 ** post_ns["log10_Mdot_disc"])),
            "Mdot_wind_mean": float(np.mean(10 ** post_ns["log10_Mdot_wind"])),
            "Mdot_wind_std": float(np.std(10 ** post_ns["log10_Mdot_wind"])),
            "Mdot_total_mean": float(
                np.mean(10 ** post_ns["log10_Mdot_disc"]
                        + 10 ** post_ns["log10_Mdot_wind"])),
            "Mdot_total_truth": obs["aux_true"]["Mdot_total"],
        },
        "nested_logZ": float(ns_res.logz[-1]),
        "nested_logZ_err": float(ns_res.logzerr[-1]),
        "quantum": quantum_results,
    }
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    banner("DONE — results written to results/summary.json")


if __name__ == "__main__":
    main()
