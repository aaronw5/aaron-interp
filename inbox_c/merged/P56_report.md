# P56: invariant mass vs spread about L-GATr's own momentum axis (claims C6 v2, C22; ParT references C3 / C18 v3)

Status: done (round C13). Script: `interp/p56_lgatr_mass_vs_spread.py`. Jobs: c-p56-run (GPU: L-GATr stage + ParT/ParticleNet stage), c-p56-ana3 (CPU: analysis + figure).
Results: `results/lgatr/p56_lgatr_mass_vs_spread.json`, `p56_lg.json`, `p56_part.json`, figure `results/lgatr/p56_lgatr_mass_vs_spread.png`; arrays `/j-jepa-vol/aaron-interp/results/lgatr/p56_lg.npz`, `p56_part.npz`.

## Conclusion

We made L-GATr's jets heavier in two ways. In one, every particle was given a little extra mass and its direction was left alone. In the other, particles were spread further apart. When the jet got heavier because the particles were heavier, L-GATr moved its W-vs-Z score only about a third as much (R = 0.36) as when the same mass change came from opening the angles. When the spread grew and the jet mass was held fixed, the score moved about a fifth as much (0.18). So L-GATr follows neither the invariant mass alone nor the spread about its own axis alone. The spread reaches it only through its per-particle angle inputs. Most of the rest of the response comes through the four-vectors, and that route also reacts to the masses of individual particles even when the jet mass does not change. The particle-mass edits produce particles unlike any in the data (classifier AUC 0.99). So this test cannot show whether the four-vector route reads the invariant mass specifically.

## Procedure

- **Sample.** JetClass test_20M WToQQ_117 / ZToQQ_117 (W/Z arm, N = 4,000) and TTBar_117 (top arm, N = 2,000): 2,000 random rows per file, numpy default_rng(5656), never first-k, jets with ≥ 3 particles (none dropped). Deviation from plan: the planned val_5M *_125 files do not exist (val_5M has _120 to _124, all used by earlier nodes). test_20M _117 is not used by any earlier node.
- **Models.** L-GATr final (model_run0.pt); untrained L-GATr (same config, torch seed 5656); ParT_full, ParT_kin (released weights); ParticleNet_full (released weights).
- **Notation.** Own axis = (η_J, φ_J) of Σ p_i, using pseudorapidity as in the L-GATr embedding. M2_own = Σ pT_i ΔR_i² about it. m = jet invariant mass. The stored axis (jet_eta, jet_phi) is never changed. Four-vectors are built in float64 and cast to float32.
- **Edits** (jet-mass doses f = +1, +2, +5, +10 %):
  - **M (particle-mass edit).** Each p_i is kept. Each particle's mass is raised to m_i'² = m_i² + μ², with one μ per jet found by bisection so that m' = (1+f)·m. Deviation from plan: μ is added in quadrature, not as a replacement, because JetClass charged hadrons already carry 0.1396 GeV and neutral hadrons up to about 0.5 GeV. Median μ is 0.19 / 0.27 / 0.43 / 0.62 GeV, above the pion mass in 89–100 % of jets. M2_own and directions are unchanged exactly. The achieved Δlog m matches the target to 2·10⁻⁴ in float32.
  - **E4 (calibration).** Every particle's (Δη, Δφ) about the own axis is scaled by k, solved per jet for the same m' (median k = 1.051 at +5 %). Particle masses are fixed.
  - **N (energy-shuffle null).** Same |ΔE_i| as M, with signs set by greedy balance (a minus sign only where E_i − |ΔE_i| > 0). The larger group is scaled by c (median 0.999, 5th percentile 0.996) so that ΣΔE = 0. This leaves m and Σ p unchanged; 46 % of particles end with E < |p|.
  - **X.** The two hardest particles' offsets from the own axis are scaled by (1+a), with a per jet so that m' = (1+f)·m (median a = 0.39 at +5 %). M2_own rises by 10 % at +5 %.
  - **S (spread up at fixed m), f = 1, 2, 5 %.** Apply X, then scale the other particles' masses m_j → √λ·m_j until m' = m (float64 residual ≤ 3·10⁻¹³). This is feasible only where those masses can absorb the rise: 28.5 % / 8.7 % / 0.6 % of W/Z jets (N = 1,140 / 349 / 23).
  - **X5 − M5 (all-jet spread contrast).** The response to X at +5 % minus the response to M at +5 %. The two jets have the same m; X5 has the larger M2_own and M5 the heavier particles.
  - **References.** ZERO (all particles massless; Δlog m = −0.9 %) and PION (all particles given 0.1396 GeV; Δlog m = −0.26 %), each with its own matched E4.
- **Read-out.** d = ℓ_W − ℓ_Z (top arm: d_t = ℓ_Tbqq − ℓ_QCD). R = Σ_jets Δd(edit) / Σ_jets Δd(E4 with the same Δm), summed over jets valid for both. R = 1 means the network responds as to a real mass change; R = 0 means no response.
- **Routes (L-GATr final).** A = full edit. D = edited four-vectors with the 7 scalar features (log pT, log E, log pT/pT_J, log E/E_J, Δφ, Δη, ΔR about the own axis) taken from the unedited jet. F = unedited four-vectors with features from the edited jet.
- **Statistics.** 2,000 bootstrap resamples over jets, shared across edits and models. Bootstrap p for R_M = 1, R_M = R_N, R_S = 0, R_S = 1. Exact McNemar on per-jet correctness with Holm over 225 arm × model tests. C2ST: a histogram likelihood-ratio classifier on per-particle (log|p|, log E) and on (log|p|, particle mass √|E² − p²|), comparing edited jets with the other half of the unedited jets. AUC 0.5 means the two are indistinguishable.

## Numbers (W/Z arm unless stated; 95 % bootstrap intervals)

**Unedited accuracy.**

| model | P1 sample, 10-class (N 4,000) | this W/Z sample | top sample (called top) |
|---|---|---|---|
| L-GATr | 83.6 % (known 83.7 %) | 71.7 % | 94.0 % |
| ParT_full | 86.1 % | 74.9 % | 95.0 % |
| ParT_kin | 74.1 % | 62.9 % | 88.6 % |
| ParticleNet_full | 84.7 % | 72.2 % | 94.1 % |

Untrained L-GATr: 10 %.

Rerun floor (L-GATr final, same input twice): max |Δd| 0.30 on W/Z (P51: 0.25). Untrained: 1·10⁻⁵.

**R by edit.**

| arm | L-GATr A | L-GATr D (four-vectors) | L-GATr F (features) | L-GATr untrained | ParT_full | ParT_kin | PN_full |
|---|---|---|---|---|---|---|---|
| M pooled 1–10 % | 0.356 [0.283, 0.430] | 0.361 | −0.003 | 0.382 | 0.196 [0.135, 0.259] | 2.66 | −0.007 |
| M by dose 1/2/5/10 % | 0.17 / 0.43 / 0.51 / 0.28 | | | | 0.13 / 0.13 / 0.19 / 0.22 | 16.6 / 7.2 / 1.9 / 0.60 | ≈ −0.007 |
| N pooled (null) | −0.447 [−0.570, −0.330] | −0.452 | −0.001 | 0.125 | −0.088 [−0.160, −0.017] | 1.75 | −0.000 |
| X5 − M5 (all jets) | 0.177 [0.081, 0.267] | 0.048 [−0.046, 0.137] | 0.131 [0.124, 0.137] | −0.039 | 0.567 [0.470, 0.657] | −0.89 | 0.594 [0.512, 0.670] |
| S pooled (feasible jets) | 0.240 [−0.190, 0.661] | 0.105 | 0.141 [0.120, 0.165] | 0.186 | 0.506 [0.048, 0.954] | 11.9 | 0.544 [0.167, 0.877] |
| X +5 % | 0.683 [0.602, 0.754] | 0.558 | 0.128 | 0.336 | 0.752 | 0.98 | 0.588 |
| ZERO | 1.55 [1.23, 1.88] | 1.56 | −0.002 | 0.197 | 1.09 | −5.1 | −0.005 |
| PION | 4.98 [3.69, 6.61] | 4.95 | 0.032 | 0.075 | 4.47 | −49 | −0.005 |

- **Route shares of a real mass change (L-GATr E4).** D 0.78, F 0.23 at every dose (P51v: R_F 0.14–0.22).
- **Hypothesis tests (L-GATr final).**
  - R_M = 1: p 0.0005.
  - R_M − R_N = 0.80 [0.71, 0.90], p 0.0005.
  - R_S (exact S) = 0: p 0.30; R_S = 1: p 0.0005.
  - X5 − M5 = 0: p 0.0005; X5 − M5 = 1: p 0.0005.
- **Calls at +5 % (L-GATr).**
  - M: 29.1 % of calls changed; true-W → Z 11.3 → 15.8 %; McNemar lost / gained correct 721 / 190, Holm p < 10⁻⁴.
  - E4: 16.6 % changed; W → Z 11.3 → 23.9 %; 423 / 184.
  - X: 22.7 % changed; W → Z 11.3 → 25.8 %.
  - N: 34.8 % changed; W → Z 11.3 → 15.3 %; 898 / 238.
  - S +2 %: 15.5 % changed; 21 / 22, Holm p 1.0.
  - PION: 9.8 % changed; Holm p 1.0.
- **Calls at +5 % (ParT_full and PN_full).** ParT_full M: 17.4 % changed, W → Z 9.7 → 12.3 %. PN_full M: 0.1 % changed; PN_full X: 21 % changed.
- **Top arm (d_t).** E4 moves d_t by only −0.19 at +5 %, so R values there are large and not comparable to the W/Z arm.
  - L-GATr M +5 %: mean Δd_t −4.3 (top calls 94 → 83 %); N +5 %: −6.7 (94 → 61 %); X +5 %: −1.55 (94 → 89 %); E4: 94 → 94.5 %. All of this goes through route D.
  - ParT_kin M +5 %: top calls 89 → 1 %.
  - PN_full M: no change.
- **C2ST AUC.**
  - On (log|p|, log E): ≤ 0.54 for every edit.
  - On (log|p|, particle mass): base 0.505, E4 0.55, X 0.51, M 0.99–1.00, N 0.99, S 0.82–0.83, ZERO 0.78, PION 0.94.

## Reading (scoped to these edits)

- **Comparison with the predictions.**
  - Invariant-mass reading (R_M in [0.8, 1.2], R_S in [−0.15, 0.15]): R_M = 0.36 is outside its range. S overlaps the range, but X5 − M5 = 0.18 does not.
  - Own-axis-spread reading (R_M ≈ R_N, R_S ≈ 1, carried by route F): excluded on both counts.
  - The measured pair is intermediate. The spread about the own axis acts only through route F (0.13–0.14 under X5 − M5, S and X; 0.23 under E4); route D gives 0.05 [−0.05, 0.14] for X5 − M5.
- **What the four-vector route (D) responds to.**
  - It responds to angle-made mass changes (E4: 0.78) and to particle-mass-made mass changes (M: 0.36). The ratio 0.36 / 0.78 = 0.46.
  - It also responds to changes of individual particle masses at fixed jet mass: N gives −0.45 and PION gives 5.0 times its matched dilation. Its response to particle masses is therefore not proportional to the jet-mass change they cause.
  - Particle masses in JetClass are tied to particle type (0 for photons, 0.1396 GeV for charged hadrons). The M and N edits make the per-particle masses distinguishable from real ones (AUC 0.99), so R_M mixes an invariant-mass response with a response to off-manifold particle masses.
- **References.** ParticleNet_full ignores energies (R_M −0.007, 0.1 % of calls) and responds to spread at fixed mass with 0.59: the positive control for the spread contrast works. ParT_full: R_M 0.20, X5 − M5 0.57. ParT_kin is dominated by its log E inputs (R_M 2.7, top calls collapse under M), so its R values do not measure a mass reading.
- **Untrained L-GATr.** R_M = 0.38 with tiny absolute Δd (mean −0.02 vs E4 −0.06), and X5 − M5 = −0.04. A random equivariant function of the four-vectors also responds to particle masses.
- **Exact S edit.** It can only be built in jets with enough particle mass (feasible fraction 0.6 % at +5 %), and it is itself off-manifold (AUC 0.83). Its interval is wide.

## Issues / uncertain

- val_5M _125 does not exist; test_20M _117 was used instead.
- The plan's E' = √(|p|² + μ²) was replaced by quadrature addition, so heavy hadrons do not become lighter.
- The N null makes 46 % of particles spacelike. A null that stays physical cannot balance the energies with the particle masses available.
- The L-GATr GPU rerun floor (max |Δd| 0.30 per jet) is averaged out in the sums but adds to the bootstrap width.
- In figure panel (c), L-GATr's and ParT_full's PION values (5.0, 4.5) are clipped at the right edge (x = 2.9), as are ParT_kin's S value (11.9) and PION value (−49). The tables above give the full values.
- A next test of whether route D reads the invariant mass should change m at on-manifold particle masses. For example: wide-angle top-jet edits that move m² and M0·M2_own apart, or swapping particle identities consistently with their masses.

## Explainer

Added `P56_EXPLAINER` and glossary entries (own axis / M2_own, M and N, X / S / X5 − M5, E4 own-axis dilation, R_M / R_S, ZERO / PION) to `site/explain.py`. C2ST, McNemar and Holm were already defined there.
