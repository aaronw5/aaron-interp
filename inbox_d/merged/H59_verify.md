# H59 verify (skeptic, loop D): jet-charge seed null and the Q_κ weighting of the QG particle-ID networks

Verifier run: job `d-h59-vf3` (GPU, 1,492 s; `d-h59-verify` was already taken by the worker's own rerun, and a first launch `d-h59-vf2` stopped at a Python syntax error before any computation and was deleted). Script `interp/h59v_kappa_verify.py`; output `$AI_RESULTS/physics/h59_verify_kappa.json` (copied to `results/physics/h59_verify_kappa.json`), log `logs/d-h59-vf3.log`. The worker's own seed-5911 rerun, which this file previously held, is kept unchanged in the appendix at the end.

## Conclusion

On 20,000 new jets, the second-seed networks again reproduce the seed-0 jet-charge response to within 7 %. A new test flips the charge of one particle at a time. Both networks weight a particle's charge as roughly (its momentum fraction)^κ, with κ between 0.3 and 0.4 on quark jets, so a plain charge count and the leading particle alone are excluded. This confirms the main H59 result by a second method. ParticleNet adds the contributions of several flipped particles almost linearly (0.88–0.94 of the sum). ParT does not: flipping the two hardest particles together moves its score by only 0.48 of the sum of the two single-flip moves, and its response depends on which way the charge is flipped. So "the score is proportional to the change in Q_κ" holds for ParticleNet but not for ParT, whose response saturates. The κ range is also wider than stated: 0.32–0.41 on quark jets, depending on seed and method, and 0.35–0.48 on gluon jets.

## Terms

- **Q_κ (jet charge):** Q_κ = Σ_i q_i z_i^κ.
  - q_i is the particle's electric charge (from its PDG id).
  - z_i = p_T,i / p_T,J, where p_T,J is the magnitude of the vector sum of the particle p_T.
  - κ = 0 counts + minus − particles; larger κ gives more weight to the hard particles.
- **s:** the network score, log p(quark) − log p(gluon), in log-odds. Δs is its change under an edit.
- **Single flip S_r:** replace only the r-th hardest charged particle by its antiparticle (PDG id → −id). Its momentum, every other particle, the multiplicity N and the charged count n_ch are unchanged; this is asserted in the code for every jet.
- **u (signed single-flip response):** u = Δs × (−q)/2, where q is the flipped particle's charge before the flip. If the score were s = a·Q_κ + (terms that do not depend on charge), every single flip would give exactly u = a z^κ. A fit of u against z therefore measures κ directly, one particle at a time.
- **Additivity slope:** the least-squares slope through the origin of Δs(flip the k hardest together) against Σ_{r≤k} Δs(S_r), the sum of the k single-flip responses. It is 1 for a score that is linear in the charges and below 1 for a response that saturates.
- **κ\* (slope ratio, H59's method):** the κ at which the slope of Δs against ΔQ_κ for soft-half flips (Fs) equals the slope for hard flips (Fh1–3).
- **Permuted network:** a seed-0 network with every weight tensor's entries randomly permuted (H40v `add_perm`, seeds 1–3). It has the same architecture and weight values but no learned function.
- **LR_κ:** logistic regression on (p_T,J, N, Q_κ), fitted on 189,998 Pythia test jets outside the sample (`h59.fit_lrs`). It is a positive control with a known κ.

## Code review of `interp/h59_charge_kappa.py`

Checks that passed:
- **Edits change PDG ids only.** p_T, y, φ, N and n_ch are asserted unchanged for every edited jet.
- **Score sign.** `-Models.logit` gives log p(q) − log p(g) (the weaver label order is [isQ, isG]). It has the same sign as the LR controls.
- **Padding.** Q_κ is computed on the particle lists, which have no padding. In `fit_lrs` the padded entries get z = 1 but q = 0, so they add nothing.
- **Sampling.** Pythia jets come from `data_qg.sample(5000, seed)`, which draws random rows per class, not the first k. The LR fit excludes exactly these rows.
- **Hardest-particle selection (Fh_k).** This assumes the lists are p_T-sorted. The verifier checked: 1 of 10,000 Pythia lists and 0 of 10,000 Herwig lists were not sorted. The effect is negligible, and the verifier's script sorts the lists explicitly.
- **128-particle input length (worker issue 6).** κ\* computed with ΔQ over only the first 128 particles equals κ\* over all particles to 3 decimals for all four networks (for example ParT s0: 0.337 and 0.337). The issue does not affect the result.

Limits of the method (not code bugs):
- **The LR "recovers its own κ exactly" check validates the code, not the method.** An LR is exactly linear in ΔQ_κ, so it cannot show whether κ\* is biased for a network whose response saturates. The single-flip fit below is independent of that assumption, because each row moves the jet charge by one particle only.
- **The worker's figure, left panel.** ParT's held-out R² (0.26–0.41) lies inside the range of the 12 permuted networks (0–0.74) at every κ. That panel therefore does not separate ParT from random-weight networks, which the worker's issue 4 states. The right panel (slope ratio) is drawn correctly. Neither panel shows an edit on a real jet.

## Procedure (verifier run)

**Samples** (fresh): Pythia `data_qg.sample(5000, 5931)` and Herwig 7.1 files 18–19 with seed 5932. Each has 5,000 quark and 5,000 gluon jets. The main analyses use quark jets from both generators pooled (≈ 10,000 jets); gluon jets are reported separately.

**Models:**
- `part_kinpid` and `pnet_kinpid` (seed 0); `part_kinpid_s1` and `pnet_kinpid_s1` (seed 1).
- 3 permuted copies of each seed-0 network.
- LR_0.3, LR_0.5 and LR_1.

**Unedited networks on this sample (AUC / accuracy):**

| | Pythia | Herwig |
|---|---|---|
| ParT s0 | 0.914 / 84.1 % | 0.815 / 72.5 % |
| PN s0 | 0.910 / 83.5 % | 0.816 / 71.7 % |
| ParT s1 | 0.914 / 84.0 % | 0.815 / 72.0 % |
| PN s1 | 0.911 / 83.5 % | 0.816 / 72.0 % |

The permuted networks have AUC 0.44–0.58.

**Edits** (PDG id sign only; four-momenta, N and n_ch fixed):
- S1, S2, S3, S4, S6, S8: single flips of the r-th hardest charged particle.
- Sx: a single flip of one random charged particle with z below the jet's median charged z.
- Fh2, Fh3: flip the 2 or 3 hardest charged particles.
- Fs, FrS: as in H59.
- E1: charge conjugation of the whole jet.

**Analyses:**
- **A (single-flip power law).** Least-squares fit of u = a z^κ over all single-flip rows, on a κ grid 0–1.5 in steps of 0.01, with a jet bootstrap (B = 500).
  - Four variants: all rows; without the leader (S1); and both again in an antisymmetrised form, where + and − flips get equal weight in each of 15 log z bins, which removes any part of the response that does not depend on the flip direction.
  - Per edit: the mean u divided by the prediction of the no-leader fit.
- **B (additivity).** For Fh2 and Fh3, the additivity slope defined above, with a bootstrap CI over jets (B = 500).
- **C (slope ratio).** κ\* as in H59, on a finer grid (0–1 in steps of 0.025), with ΔQ over all particles and over the first 128 only. Bootstrap B = 300.
- **D (Part 1 check).** E1 × sign(Q_0.3): the mean change of s when every charged particle is replaced by its antiparticle, multiplied by the sign of the unedited jet's Q_0.3.

**Hypotheses:**
- H0a: the network reads a charge count (κ = 0). H0b: it reads a hard-weighted charge (κ ≥ 0.5). H0c: it reads the leader's charge only (u = 0 for every particle except the leader).
- H0d: the response is linear in Q_κ (additivity slope = 1, and Δs for a + flip equals −Δs for a − flip at the same z).
- Tests: two-sided bootstrap p (fraction of bootstrap draws beyond the null value, floor 1/500), Holm over the 4 trained networks × 4 hypotheses.

## Results

**A. Single-flip κ (quark jets).** The table gives the κ that best fits u = a z^κ, with its 95 % jet-bootstrap CI.

| network | κ, all single flips | κ, leader excluded | κ, antisymmetrised | a (log-odds) | variance explained by a z^κ |
|---|---|---|---|---|---|
| ParT s0 | 0.32 [0.31, 0.33] | 0.33 [0.32, 0.34] | 0.33 | 0.45 | 0.44 |
| ParT s1 | 0.34 [0.33, 0.35] | 0.35 [0.34, 0.36] | 0.35 | 0.47 | 0.44 |
| PN s0 | 0.35 [0.34, 0.35] | 0.37 [0.37, 0.38] | 0.35 | 0.48 | 0.82 |
| PN s1 | 0.39 [0.38, 0.39] | 0.41 [0.40, 0.41] | 0.39 | 0.54 | 0.84 |
| LR_0.3 / LR_0.5 / LR_1 (controls) | 0.30 / 0.50 / 1.00 | 0.30 / 0.50 / 1.00 | 0.30 / 0.50 | 0.35 / 0.50 / – | 1.00 |
| permuted (6) | 0.00 (×4), 0.10, 1.17 | – | – | 0.004–0.017; one PN copy −0.12 | – |

Per-particle response, mean u by rank for ParT s0 (PN s0 in parentheses), with the ratio to the no-leader power-law prediction:

| flip | mean z | mean u | ratio to fit |
|---|---|---|---|
| S1 | 0.235 | 0.266 (0.267) | 0.95 (0.92) |
| S2 | 0.118 | 0.214 (0.218) | 0.96 (0.96) |
| S3 | 0.072 | 0.187 (0.190) | 0.99 (1.01) |
| S6 | 0.026 | 0.137 (0.129) | 1.03 (1.02) |
| Sx | 0.006 | 0.077 (0.064) | 1.00 (0.94) |

- The z^κ law holds within ±8 % from z = 0.006 to z = 0.23. The leader sits slightly below the law (0.92–0.95), not above it.
- S2/S1 = 0.80 for both networks. Q_0.3 predicts 0.82 (LR_0.3) and Q_0.5 predicts 0.71.
- Sx/S1 = 0.29 for ParT and 0.24 for PN. LR_0.3 gives 0.31 and LR_0.5 gives 0.15.
- Hypothesis tests: κ = 0 is excluded for all four networks (bootstrap p < 0.002, Holm p < 0.02). κ ≥ 0.5 is excluded on quark jets (upper CI ≤ 0.41). Leader-only is excluded, because the soft single flip Sx gives u = 0.06–0.08 with CI far from 0.
- **Gluon jets** give higher κ: single-flip fit ParT s0 0.44 [0.42, 0.47], ParT s1 0.48 [0.46, 0.51], PN s0 0.41 [0.40, 0.42], PN s1 0.45 [0.44, 0.46]. The slope-ratio κ\* is 0.35–0.40.

**B. Additivity.** Additivity slope of Δs(Fh_k) against the sum of the single-flip responses, with 95 % CI:

| network | Fh2 | Fh3 | mean \|Σ singles\| → mean \|Fh2\| |
|---|---|---|---|
| ParT s0 | 0.48 [0.47, 0.50] | 0.43 [0.42, 0.44] | 0.77 → 0.46 |
| ParT s1 | 0.48 [0.47, 0.50] | 0.44 [0.43, 0.45] | 0.77 → 0.46 |
| PN s0 | 0.93 [0.92, 0.93] | 0.88 [0.87, 0.89] | 0.45 → 0.41 |
| PN s1 | 0.94 [0.93, 0.95] | 0.90 [0.89, 0.90] | 0.48 → 0.44 |

- H0d (additive, slope 1) is excluded for every network (p < 0.002). The departure is large for ParT and small for ParticleNet.
- ParT's single flips also depend on the flip direction. Flipping the leader + → − gives Δs = −0.38; flipping it − → + gives Δs = +0.72. The direction-independent part is +0.17 log-odds. For PN the values are −0.52 and +0.56, and for LR_0.3 −0.44 and +0.43.
- So ParT's response depends on the jet's starting charge and saturates, rather than being proportional to ΔQ_κ. This is consistent with H59's low R² for ParT (0.40) and with ParT's smaller Fh2 − Fh1 increment (0.32 against 0.43 for PN).
- The antisymmetrised fit removes the direction-independent part and gives the same κ (0.33), so the κ estimate is not driven by the asymmetry.

**C. Slope-ratio κ\*** (fine grid; fresh sample), quark jets, with 95 % CI:
- ParT s0 0.337 [0.319, 0.354]; ParT s1 0.351 [0.337, 0.365]; PN s0 0.346 [0.340, 0.352]; PN s1 0.379 [0.373, 0.385].
- The same values result with ΔQ over the first 128 particles only.
- Compared with H59 (run / verify): ParT s0 0.32 / 0.33, ParT s1 0.34 / 0.35, PN s0 0.34 / 0.34, PN s1 0.36 / 0.37. The shift between samples is up to 0.015, which is sometimes outside the per-sample CI.
- Single-flip κ and slope-ratio κ\* agree to within 0.02 for each network on quark jets.

**D. Part 1 seed null.** E1 × sign(Q_0.3), Pythia / Herwig:

| | seed 0 | seed 1 | s1/s0 |
|---|---|---|---|
| ParT | −0.255 / −0.248 | −0.256 / −0.251 | 1.01 / 1.01 |
| PN | −0.240 / −0.238 | −0.254 / −0.254 | 1.06 / 1.07 |

- ParT s0 CI on Pythia: [−0.264, −0.246].
- Permuted networks: |value| ≤ 0.034, except one PN copy at +0.116 / +0.114.
- This reproduces H59 Part 1, including the s1/s0 ratio within 7 %.

## Assessment of the H59 reading

- **Reproduced:**
  - The seed null on both architectures (within 7 %).
  - κ ≈ 0.3–0.4 on quark jets, by an independent method (single flips) that agrees with κ\* to within 0.02.
  - Exclusion of the count (κ = 0), of leader-only, and of κ ≥ 0.5 on quark jets.
  - Fs/Fh per unit net charge ≈ 0.22.
  - R² cannot separate κ = 0.3 from 0.5.
  - The 128-particle truncation is negligible.
- **Not supported as stated:**
  1. "Both architectures respond … in proportion to the change in Q_κ." This holds for ParticleNet (additivity 0.88–0.94, R² 0.8). It does not hold for ParT: the per-particle weights follow z^κ, but they combine to 0.43–0.48 of the sum, and the response depends on the flip direction (−0.38 against +0.72 for the leader).
  2. "The 2nd and 3rd hardest particles each add about as much as the first." A single flip of the 2nd particle moves the score 0.80 times as much as a flip of the 1st, as Q_0.3 predicts (0.82). For ParT the increment Fh2 − Fh1 is only 0.60 of Fh1, because of the sub-additivity.
  3. "κ about 0.32–0.37." Across seeds, samples and methods, quark jets give 0.32–0.41 (PN s1 reaches 0.39–0.41), and gluon jets give 0.35–0.48. On gluon jets, κ ≥ 0.5 is not excluded for ParT s1 (CI upper 0.51).
- **Controls.**
  - The permuted networks are an adequate null for the size of the charge response. Their slopes are 0.004–0.017 against 0.45–0.54 for the trained networks, except one PN copy at −0.12. They do not serve as a null for the κ value, which is undefined for a network with no response.
  - The seed comparison is the informative null for κ. The label-noise (ln30) networks were still training and were not tested.

## Corrected reading

At fixed four-momenta, multiplicity and charged count, both quark–gluon particle-ID networks, at both training seeds, weight each charged particle's charge as about z^κ with κ = 0.32–0.41 on quark jets. Two methods give this: single-particle flips and the soft/hard slope ratio, which agree to within 0.02. On gluon jets κ is 0.35–0.48. A charge count, a leader-only charge, and κ ≥ 0.5 on quark jets are excluded.

ParticleNet's response to multi-particle flips is close to linear in ΔQ_κ (additivity 0.88–0.94). ParT's is not: its response to flipping two or three hard particles together is 0.43–0.48 of the sum of the single-flip responses, and it depends on the flip direction. The seed-1 networks reproduce the seed-0 charge-conjugation response within 7 % on new jets. The shuffled-label networks are constant, and the label-noise networks were not yet available.

## Issues

1. The task named the job `d-h59-verify`, but that name was already used by the worker's rerun, so this run is `d-h59-vf3`.
2. No verifier figure was made: local computation is not allowed, and the plot was not added to the job. The binned u(z) values are in the JSON (`A.<net>.binned`).
3. The single-flip fit uses quark jets from Pythia and Herwig pooled. Per-generator κ values were not computed.
4. On the 0–1 grid, κ\* for LR_1 returns NaN, because its crossing lies at the grid edge. LR_0.3 and LR_0.5 return 0.300 and 0.500 exactly.

---

## Appendix: worker's verify run (seed 5911), as previously in this file

(See `results/physics/h59_charge_kappa_v.json`; command `./part_job.sh d-h59-verify 'python -m interp.h59_charge_kappa --step run --seed 5911 --tag _v'`.)

| quantity | ParT s0 | ParT s1 | PN s0 | PN s1 |
|---|---|---|---|---|
| Pythia AUC (unedited) | 0.909 → 0.907 | 0.909 → 0.907 | 0.904 → 0.902 | 0.905 → 0.903 |
| Herwig AUC (unedited) | 0.813 → 0.804 | 0.812 → 0.805 | 0.815 → 0.805 | 0.815 → 0.807 |
| E1 × sign Q_0.3, Pythia | −0.246 → −0.249 | −0.249 → −0.254 | −0.238 → −0.237 | −0.254 → −0.249 |
| E1 × sign Q_0.3, Herwig | −0.246 → −0.243 | −0.252 → −0.248 | −0.237 → −0.243 | −0.249 → −0.256 |
| held-out R² at κ = 0.3 | 0.400 → 0.400 | 0.406 → 0.412 | 0.792 → 0.790 | 0.806 → 0.803 |
| best κ by R² (best − second, Holm p) | 0.3 → 0.3 (+0.003, 0.08) | 0.5 → 0.5 (+0.005, 0.008) | 0.3 → 0.3 (+0.016, 0.008) | 0.3 → 0.5 (+0.002, 0.16) |
| κ* (slope ratio Fs/Fh = 1) [CI] | 0.32 → 0.33 [0.32, 0.35] | 0.34 → 0.35 [0.34, 0.36] | 0.34 → 0.34 [0.34, 0.35] | 0.36 → 0.37 [0.36, 0.38] |
| Fh2 − Fh1, signed | +0.318 → +0.319 | +0.317 → +0.322 | +0.426 → +0.428 | +0.447 → +0.444 |
| Δs per unit net charge, Fs / Fh1 | 0.22 → 0.22 | 0.21 → 0.22 | 0.22 → 0.22 | 0.20 → 0.20 |

- LR controls on that sample: each LR recovers its own κ.
- Part 1 seed ratios s1/s0: E1 0.98–1.05; E2 0.98–1.04; E3 0.88–1.02. Holm over the 32 tests of the non-constant trained networks: all p < 10⁻¹⁰⁰.
- That run used the code version before the Holm fix for constant networks.
