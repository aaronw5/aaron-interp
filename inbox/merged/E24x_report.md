# E24x: what the context-only objects S58 and S31 compute (claims A2, A3z, A13; exploratory) (2026-10-20)

## Conclusion
Two of JEDI-linear's second-round summary units (S58 and S31) do not read the individual particles at all. They read only the jet-level context vector, which is built from counts of particles on either side of fixed lines in the (Δη, Δφ) plane. Written as sums over the particles, S31's inputs are the signed particle-count offsets along Δη and Δφ (Σ Δη and Σ Δφ, not pT-weighted). S58's inputs are the count-weighted elongation terms Σ r cos 2α and Σ r sin 2α. These sums explain 93–97 % of the variance of each unit's context threshold on held-out jets, up from at most 19 % for the earlier fits. Still uncertain: an untrained network with the same sparsity pattern also gives context inputs that low-order sums describe well, so what training adds is the separation into pure dipole or pure quadrupole inputs. The per-slot rounding of the network also limits how much of the units' own values the named form reproduces (78 % for S58 and 63 % for S31, against ceilings of 81 % and 65 %).

Terms: the *context threshold* θ of a unit is the jet-level part of its pre-activation, θ = −(W3 · g + b3). A *count dipole* is a sum over particles of a direction (Σ Δη, Σ Δφ). A *count quadrupole* is a sum of an elongation term (Σ r cos 2α, Σ r sin 2α, with α the particle's azimuth in the (Δη, Δφ) plane). All sums run over real particles and are not pT-weighted.

## Procedure
- **Script and jobs:**
  - Script `interp/e24x_context_jedi.py`; figure `interp/e24x_plot.py` (rendered locally).
  - Job `e24x-run`: 1 CPU / 2 GiB, wall 1,088 s. The trained arms 0–3 completed and saved `results/jedi/e24x_context.json` and `e24x_arrays.npz`. It then exited during the untrained control (bug: an untrained unit with no live g input gave an empty fit).
  - Job `e24x-untr` (fix + `--untrained-only`): wall 545 s. Saved `results/jedi/e24x_untrained.json`.
  - Smoke runs `e24x-smoke` (first draft, failed in the untrained control at a 7-unit index) and `e24x-smoke2` (N 1,500).
  - Logs: `logs/e24x-run.log`, `logs/e24x-untr.log`, `logs/e24x-smoke2.log`.
  - Peak pod memory was 2,047 MiB in every job. That is the cgroup peak, which includes the file cache from reading the test archive, as in E24w.
- **Model:** official JEDI-linear 3-feature n64 via `interp.jedi_core.Jedi(64)` (no default changes). A probe replica (l1 → P → g → s → add → l3 → qs1) reproduces jedi_core exactly: max |ΔS| = 0 and max |Δg| = 0 on 2,000 jets.
- **Samples:**
  - Fit: `sample(3000, 3101)`, N 15,000. Accuracy 80.79 %.
  - Test: `sample(1500, 3102)` with the 466 jets also in the fit sample removed, N 7,034. Accuracy 81.65 %, above the 80.6–81.3 % band; binomial 95 % ±0.9 points at this N.
- **Structure check from the weights:**
  - W3[:, 58] and W3[:, 31] each have 4 non-zero entries. All 4 sit on s channels that are 0 for every particle, padding included.
  - So the per-particle term W3 · s_i is identically 0 for both units: the "own" standard deviation is 0 over real particles in the window.
  - u_i = ReLU(W3 · q(g) + b3), up to the per-slot quantizers. S = (1/16) Σ_{slots in window} q1(u_i).
  - Live g inputs: S58 ← g36 (W3 −0.25), g63 (−0.25), g44 (−0.125), g47 (−0.125), b3 = +3. S31 ← g5 (+1.0), g29 (+0.75), g55 (+0.5), g39 (+0.375), b3 = 0.
  - Slot windows: S58 60 of 64 slots, S31 47. These are set by the per-slot quantizer masks, not by particle content.
- **Arm 0 (does S read anything but g):**
  - Every particle of 3,000 test jets was replaced by a random donor's particles, with g overridden to the recipient's g (`run_jetwise`).
  - Metric: max |ΔS| against the recipient's S.
- **Arm 1 (naming, fit on fit jets, R² on test jets, 300-jet bootstrap CI on S):**
  - (i) Structural forms: exact pre-activation (only the qs1 rounding left out); additive gate S = (n_win/16)·ReLU(−θ) with the exact θ; θ frozen at its mean.
  - (ii) The planned per-particle form Σ (f(pT, r, rank) − θ)₊, with f = MARS on per-particle features. It is degenerate here because f ≡ 0.
  - (iii) θ named by MARS (≤ 8 hinges) on rotation-invariant jet observables (ΣpT, N, pT_lead, M2, M2/M0, Σ pT r, Σr, Σr², m, m/pT, ΣpT of the top 14, N(pT > 1)). Also named on these plus fixed-frame angular count moments and half-plane counts ("extended"). GBT on both sets as an upper reference.
  - (iv) **Multipole chain:** each g input is ReLU(preg_k), and preg_k is linear in the pooled l1 statements P (checked: test R² 0.998–0.9997). preg_k is therefore a sum over particles of a per-particle field. preg_k was fitted by least squares on count-weighted sums over real particles, in nested blocks:
    - m0: N, N(pT < 11 GeV), Σ ln pT, ΣpT, Σr, Σr².
    - + m1: Σ cos α, Σ sin α, Σ Δη, Σ Δφ, Σ r² cos α, Σ r² sin α.
    - + m2: Σ cos 2α, Σ sin 2α, Σ r cos 2α, Σ r sin 2α, Σ (Δη² − Δφ²), Σ 2ΔηΔφ.
    - + m3.
    - The fitted preg_k was then passed through the ReLU, W3 and b3 to give θ and S.
  - (v) References for S: E24w-style MARS on the nine pooled half-planes; MARS on all 64 P; ridge on the live g units; GBT on the top-7 g units and on all live g units; ridge on the top-7 g units plus their pairwise products.
  - (vi) Per-particle field: the exact a_i of 1,000 fit jets (44,831 real particles), passed through the dd kernel and the pool scale, minus the padding particle's contribution. The resulting θ field and each g field were regressed on the per-particle basis by block.
- **Arm 2 (causal, threshold):**
  - g of 3,000 test recipients was overridden by a donor's g, from a matched donor (nearest neighbour in standardised slot-mean q(s)) or a random donor.
  - Metric: slope of the actual ΔS on the predicted ΔS. Predictions come from the exact θ (structural), the multipole-named θ (named_mp), and the MARS-named θ (inv, ext). 1,000-jet bootstrap CI.
  - Path call test: only S58 and S31 were set to the actual or the predicted values, with the other 62 S units at their originals. IIA = % of jets whose calls agree between the two; also the % of actual call changes that the prediction reproduces, and McNemar on predicted vs actual changes.
  - Control: g + a random unit direction in the 11-dimensional live-g subspace, scaled to the matched-donor |Δg|; 200 draws × 500 jets, 50 draws for calls.
- **Arm 3 (causal, input):**
  - The hardest real particle outside S58's slot window (slot 0 if none; 12.8 % of jets) was moved outward by Δr = 0.1 at fixed pT and azimuth. Control edit: the softest real particle moved by the same amount.
  - Each edit was run with the full network and with g frozen at its unedited value. Gate effect = ΔS(full) − ΔS(g frozen).
  - H0: S58 and S31 change only through particles in their own window (gate effect 0). H1: they change through θ(g) as predicted.
- **Untrained control:** `Jedi(64, untrained=True, seed=0/1/2)` (the redraw keeps the trained zero-bit masks), with the same arm-1 fits on units 58 and 31 where alive. Seed 1 had S60 as a second unit and seed 2 had S30; both have θ ≡ constant (no live g input), so no θ naming applies.
- **Multiple testing:** Holm over 2 units × 2 causal arms. The tests are the arm-2 matched named_mp slope and the arm-3 hard-edit named_mp slope (H0 slope = 0).

## Numbers
**Arm 0: particles replaced, g held (N 3,000)**
- S58 and S31: max |ΔS| = 0.0 in 3,000 / 3,000 jets.
- The other 17 live S units change in 51–100 % of jets (median 98 %); median mean |ΔS| is 1.05 sd.

**Arm 1: held-out R² of S (test N 7,034, 95 % CI)**

| form | S58 | S31 |
|---|---|---|
| E24w-style MARS, 9 pooled half-planes | 0.166 | 0.201 |
| MARS, all 64 P | 0.190 | 0.240 |
| MARS on invariant jet observables | 0.244 | 0.221 |
| MARS on extended (fixed-frame) observables | 0.491 | 0.832 |
| GBT on extended observables | 0.792 | 0.943 |
| ridge, live g (linear) | 0.862 | 0.981 |
| ridge, top-7 g + pairwise products | 0.924 | 0.982 |
| GBT, top-7 / all live g | 0.995 / 0.995 | 0.985 / 0.986 |
| θ frozen at its mean (d) | −0.174 [−0.198, −0.151] | −0.096 [−0.116, −0.081] |
| additive gate with exact θ (ceiling of any θ naming) | 0.851 [0.844, 0.856] | 0.657 [0.642, 0.673] |
| exact pre-activation (only qs1 rounding left out) | 0.927 [0.924, 0.929] | 0.997 [0.997, 0.997] |
| chain ReLU(preg), no add/l3 quantizers | 0.809 [0.800, 0.816] | 0.648 [0.631, 0.664] |
| **multipole chain m0 / +m1 / +m2 / +m3** | −0.80 / −0.80 / **0.784 [0.774, 0.792]** / 0.785 | −0.69 / **0.658 [0.640, 0.676]** / 0.632 / 0.632 |
| planned per-particle form, MARS θ inv / ext | 0.034 / 0.408 | 0.065 / 0.573 |

**Arm 1: held-out R² of the threshold θ** (sd θ: S58 1.20, S31 1.86)

| θ named by | S58 | S31 |
|---|---|---|
| MARS, invariant observables | 0.217 | 0.227 |
| GBT, invariant observables | 0.221 | 0.268 |
| MARS, extended | 0.604 | 0.845 |
| GBT, extended | 0.842 | 0.946 |
| multipole chain m0 | < 0 | < 0 |
| multipole chain m0 + m1 | < 0 | **0.951** |
| multipole chain m0 + m1 + m2 | **0.929** | 0.967 |
| exact chain ReLU(preg) | 0.972 | 0.988 |

**Per g input: preg R² by block** (test)
- S58 inputs:
  - m0: 0.04–0.22. m0 + m1: 0.04–0.23. m0 + m1 + m2: 0.984–0.991.
  - Largest standardised terms: g36 −Σ r sin 2α (−12.2) and −Σr (−6.5); g47 +Σ r sin 2α (+10.4) and −Σr; g44 −Σ r cos 2α (−7.7) and −Σr; g63 +Σ r cos 2α (+7.0) and −Σr.
- S31 inputs:
  - m0: 0.14–0.19. m0 + m1: 0.97–0.99.
  - Largest terms: g5 −Σ Δη (−3.0) and −Σr (−1.4); g29 +Σ Δη (+4.3) and −Σr; g39 +Σ Δφ (+8.8) and −Σr; g55 −Σ Δφ (−6.9) and −Σr.
- Fraction of jets with g > 0: S58 inputs 0.29–0.42; S31 inputs 0.21–0.29.
- Resulting forms:
  - S31 = (47/16)·ReLU(Σ_k w_k g_k), with b3 = 0. This is a weighted sum of rectified ±Σ Δη and ±Σ Δφ counts, each above a Σr-dependent threshold.
  - S58 = (60/16)·ReLU(3 − 0.25(g36 + g63) − 0.125(g44 + g47)). It falls as the rectified count quadrupole grows. S58 > 0 in 96 % of jets.

**Per-particle θ field (44,831 particles):** R² by block.
- S58: m0 0.593, + m1 0.602, + m2 0.975.
- S31: m0 0.930, + m1 0.981, + m2 0.986. The summed field is mostly radial; the dipole parts cancel between the ± pairs and appear only after each g input's ReLU.
- Single g fields: S58's 0.02–0.11 at m0 and 0.96–0.97 with m2; S31's 0.04–0.06 at m0 and 0.98–0.99 with m1.

**Arm 2: g interchange (N 3,000)**

| donor | mean L1 \|Δg\| | full-net calls changed | path (S58, S31 only) calls changed | S58 slope [CI], R² (structural; named_mp) | S31 slope [CI], R² (structural; named_mp) |
|---|---|---|---|---|---|
| matched | 16.9 | 7.50 [6.57, 8.43] % | 4.40 [3.67, 5.17] % (132 jets) | 0.991 [0.980, 1.002], 0.93; **0.976 [0.960, 0.992], 0.84** | 0.684 [0.677, 0.691], 0.75; **0.677 [0.669, 0.686], 0.70** |
| random | 26.5 | 28.2 [26.6, 29.9] % | 5.37 [4.57, 6.23] % (161) | 1.001, 0.94; 0.996 [0.983, 1.008], 0.88 | 0.685, 0.76; 0.683 [0.675, 0.692], 0.73 |

- **MARS-named θ, matched donor:**
  - invariant: S58 slope 0.66, R² 0.01; S31 0.76, R² 0.04.
  - extended: S58 0.87, R² 0.42; S31 0.75, R² 0.71.
- **Path calls, matched donor (named_mp):**
  - IIA 89.0 [87.9, 90.1] %; actual changes reproduced 59.1 [50.8, 67.4] %.
  - The prediction changes 12.4 % of calls against an actual 4.4 % (McNemar 276/37, p 2e-46).
  - Structural: IIA 88.8, reproduced 60.6 [52.3, 68.9] %, predicted 12.8 %.
  - MARS invariant: reproduced 20.5 [13.6, 28.0] %.
- **Random-direction control:**
  - |ΔS31| 6.98 [6.78, 7.16] sd vs 0.88 [0.80, 0.96] sd for the matched donor; |ΔS58| 0.98 vs 0.91 sd.
  - Path calls 22.7 [20.2, 25.6] %; full-net calls 42.2 [39.6, 44.8] %.
  - The structural additive prediction fails off the data's g range: R² over all draws is 0.16 for S58 and −26 for S31; first-draw slope 0.61 and 0.14.
- **Holm** (4 tests: arm-2 matched named_mp slope and arm-3 named_mp slope, S58 and S31): all p = 0.001 (the bootstrap floor), Holm 0.004.

**Arm 3: input edit (N 3,000; S58 window = 60 slots; 87.2 % of jets have a real particle outside it)**

| edit | \|ΔS58\|/sd full | g frozen | \|ΔS31\|/sd full | g frozen | jets with ΔS58 ≠ 0 | calls full / g frozen | McNemar |
|---|---|---|---|---|---|---|---|
| hardest particle outside window, Δr +0.1 | 0.10 [0.09, 0.11] | 0.000 (exact) | 0.07 [0.06, 0.08] | 0.000 | 20.4 % | 9.30 / 8.97 % (diff +0.33 [−0.03, 0.70]) | 21/11, p 0.11 |
| softest particle, Δr +0.1 (control) | 0.11 [0.10, 0.12] | 0.000 | 0.07 [0.06, 0.08] | 0.000 | 21.9 % | 1.37 / 1.13 % | 14/7, p 0.19 |

- **Gate-effect prediction (hard edit), slope [CI], R²:**
  - structural: S58 0.943 [0.89, 1.00], 0.68; S31 0.696 [0.68, 0.72], 0.71.
  - named_mp: S58 0.953 [0.75, 1.17], R² 0.05 (corr 0.23); S31 0.643 [0.59, 0.70], R² 0.15 (corr 0.46).
  - MARS invariant: S58 0.007 [−0.07, 0.08]; S31 0.14.
- **Signed mean gate effect:** S58 −0.006 sd [−0.01, 0.00], p 0.26; S31 −0.012 sd, p 0.001.

**Untrained control (θ R², multipole chain m0 / +m1 / +m2; exact chain ceiling)**
- Seed 0, S58: 0.10 / 0.28 / 0.30 (chain 0.34).
- Seed 1, S58: −0.59 / 0.21 / 0.29 (chain 0.39).
- Seed 2, S31: 0.30 / 0.57 / 0.74 (chain 0.74).
- Untrained per-g preg R² at m0 / +m1 / +m2:
  - Seed 0: g36 0.64 / 0.90 / 0.98; g44 0.04 / 0.62 / 0.97; g47 0.57 / 0.89 / 0.99.
  - Seed 1: g47 0.53 / 0.86 / 0.98; g63 0.26 / 0.87 / 0.96.
  - Seed 2: g39 0.72 / 0.82 / 0.98; g55 0.36 / 0.75 / 0.99.
- Untrained accuracy 20.2–20.6 %.

## Reading (scoped to what was measured)
- **Per-particle hinge with a jet-dependent threshold (the planned form):**
  - For S58 and S31 the per-particle part is identically 0: the W3 weights sit only on s channels that are always 0.
  - Arm 0: replacing every particle while holding g leaves both units unchanged in all 3,000 jets. The other 17 live S units change in a median 98 % of jets.
  - The planned form Σ_i (f(pT_i, r_i, rank_i) − θ)₊ reduces to (n_win/16)·ReLU(−θ(g)). The slot window comes from quantizer masks, not from particle content.
- **H0 of arm 3** ("S58 changes only through particles in its window") is excluded. With g frozen, ΔS58 and ΔS31 are exactly 0 under both edits. With g free, |ΔS58| = 0.10 sd and |ΔS31| = 0.07 sd. All of the change passes through θ(g).
  - The call-level difference between g free and g frozen is +0.33 [−0.03, 0.70] points (McNemar p 0.11). This g freeze also holds the context paths of S0 and S30.
- **Naming θ:**
  - The thesis prediction "θ is a named rotation-invariant jet quantity such as ΣpT or the hardest particle's pT" is not supported. Invariant observables reach R² 0.22–0.27 for θ (MARS and GBT).
  - θ follows count-weighted fixed-frame multipole sums: S31's from dipoles (Σ Δη, Σ Δφ; θ R² 0.95), S58's from quadrupoles (Σ r cos 2α, Σ r sin 2α; θ R² 0.93). For S, this lifts R² from ≤ 0.24 (half-plane and invariant fits) to 0.78 (S58) and 0.66 (S31).
  - These S values are 97–102 % of what the exact-θ additive form reaches (0.85, 0.66). The rest of S31's variance (exact pre-activation 0.997) is per-slot quantization of the l3 input, which no jet-level θ captures.
  - The "≥ 0.8" target is met for θ in both units. For S it is not met in S31 (0.66, quantizer-limited) and is marginal in S58 (0.78).
- **What is specific to training:**
  - Low-order multipole sums also describe the untrained redraws' g inputs (m ≤ 2: R² 0.96–0.99). This follows from preg being a linear combination of half-plane counts.
  - The difference measured: in the trained network each input of S58 is a pure quadrupole and each input of S31 a pure dipole. Their m0 + m1 share of the variance is ≤ 0.23 for S58's inputs, and m0 alone is ≤ 0.19 for S31's. Untrained inputs mix orders (m0 0.04–0.72, m1 adds 0.1–0.6).
  - Untrained θ is quantizer-dominated: the exact chain reaches only 0.34–0.74.
- **Arm 2 (g interchange):**
  - The multipole-named θ predicts ΔS58 with slope 0.976 [0.960, 0.992] (R² 0.84) and ΔS31 with slope 0.677 [0.669, 0.686] (R² 0.70).
  - The S31 slope below 1 is shared by the exact-θ additive form (0.684). It comes from the per-slot quantizers, not from the naming.
  - On calls, the named and structural forms both predict about 3× more changes than occur (12.4 vs 4.4 %). This follows from the same S31 over-prediction. 59 % of the actual changes are reproduced.
  - Random g directions of the same size push S31 about 7 sd, against 0.9 sd for real donors. The g values that occur in jets cover a narrow part of g space.
- **Arm 3 (single-particle edits):** the named θ predicts the sign and scale of the gate effect on average (S58 slope 0.95). It explains little of its per-jet variance (R² 0.05–0.15), because a 0.1 rad move of one particle changes θ by less than the fit residual.

## Statistics summary
- **H0/H1:**
  - Arm 3: H0 gate effect = 0. Excluded: the g-frozen change is exactly 0 and the g-free |ΔS| intervals exclude 0.
  - Arm 2: H0 slope = 0 for the named θ. Holm p 0.004 in both units.
- **Competing hypotheses:**
  - (a) Additive hinges on pooled half-planes (E24w): excluded as sufficient, R² ≤ 0.24.
  - (b) A rotation-invariant scalar threshold (ΣpT, pT_lead, M2, ...): excluded as sufficient, θ R² ≤ 0.27.
  - (c) Count dipole (S31) and count quadrupole (S58) sums through the network's ReLU chain: θ R² 0.95 / 0.93.
  - (d) θ constant (frozen at its mean): S R² < 0.
- **Controls:**
  - Random donors and random g directions: arm 2.
  - Softest-particle edit: arm 3. Its |ΔS| is the same size as the hard edit's, while it changes far fewer calls (1.4 vs 9.3 %).
  - Untrained redraws, 3 seeds.

## Issues / debug history
- Debug 1 of 2: the first smoke draft failed in the untrained control at a 7-unit index. The main run then failed at an untrained unit with no live g input. Both are fixed; the untrained control was rerun alone (`--untrained-only`). The trained results come from the first full run and were not rerun.
- Test accuracy 81.65 % is 0.35 points above the stated band. E24w also measured 81.60 % on its sample, and the binomial ±0.9 points at N 7,034 covers it. The probe identity is exact.
- The arm-2 "matched" donor matches the slot-mean of q(s). Since S58 and S31 do not read s, the matching affects only the other S units in the full-network calls.
- The arm-3 g freeze holds all 64 g units, so the call difference (+0.33) includes S0 and S30's context paths and is not specific to S58 and S31.
- Not run: a separate verification job (e24x-verify). The planned "jet-matched s" interchange is degenerate for these units (arm 0). A verifier should rerun arms 1–2 on a fresh seed, and test the dipole/quadrupole name causally with a rigid azimuthal rotation by 90° (which swaps cos 2α ↔ −cos 2α) and a mirror Δη → −Δη (which flips Σ Δη).

## Explainer entry (for site/explain.py)
```python
E24X_EXPLAINER = [
 ('Question',
  'JEDI-linear\'s context-only second-round objects S58 and S31 carry most of the call change when a soft prong is moved (E24w), but fits on the pooled half-plane counts explained at most 19 % of them. What jet quantity do they compute?', None),
 ('Background',
  'In JEDI-linear, $u_i=\\mathrm{ReLU}(W_3(s_i+g)+b_3)$ and $S=\\frac1{16}\\sum_i u_i$. For S58 and S31 the non-zero entries of $W_3$ sit only on $s$ channels that are 0 for every particle, so $u_i=\\mathrm{ReLU}(-\\theta)$ with the jet-level threshold $\\theta=-(W_3\\cdot g+b_3)$. Each context unit is $g_k=\\mathrm{ReLU}(W_d P+b_d)_k$, and $P$ is a sum over particles of half-plane ramps, so $g_k$ is a rectified sum over particles of a per-particle field $F_k(\\Delta\\eta,\\Delta\\phi,p_T)$. Count multipoles: $\\sum_i \\Delta\\eta_i$, $\\sum_i \\Delta\\phi_i$ (dipole, $m=1$) and $\\sum_i r_i\\cos2\\alpha_i$, $\\sum_i r_i\\sin2\\alpha_i$ (quadrupole, $m=2$), with $\\alpha_i$ the particle\'s azimuth in the $(\\Delta\\eta,\\Delta\\phi)$ plane and no $p_T$ weight.', None),
 ('Method',
  'Fit sample(3000, 3101) ($N$ 15,000), test sample(1500, 3102) minus overlap ($N$ 7,034). (0) All particles replaced by a donor\'s with $g$ held. (1) $g_k$ pre-activations fitted on nested multipole blocks $m\\le0,1,2,3$, passed through ReLU, $W_3$, $b_3$; compared with MARS / GBT on rotation-invariant observables, trees on $g$, and the E24w half-plane fit. '
  '(2) $g$ swapped between jets (matched and random donors) and predicted vs actual $\\Delta S$; random $g$ directions as control. (3) The hardest particle outside S58\'s slot window moved by $\\Delta r=0.1$, with $g$ free vs frozen. Untrained redraws, 3 seeds. Holm over 2 units × 2 arms.', None),
 ('Result',
  'Replacing every particle with $g$ held leaves S58 and S31 unchanged in 3,000/3,000 jets. S31\'s four inputs are rectified $\\pm\\sum\\Delta\\eta$, $\\pm\\sum\\Delta\\phi$ (preg $R^2$ 0.97–0.99 with $m\\le1$); S58\'s are rectified $\\pm\\sum r\\cos2\\alpha$, $\\pm\\sum r\\sin2\\alpha$ ($R^2$ 0.98–0.99 with $m\\le2$, $\\le0.23$ without). Held-out $R^2$ of $\\theta$: 0.95 (S31) and 0.93 (S58), against $\\le0.27$ for rotation-invariant observables. Under $g$ interchange the named $\\theta$ predicts $\\Delta S58$ with slope 0.98 [0.96, 0.99].', 'results/jedi/e24x_context.png'),
 ('Checks',
  'Probe identical to jedi_core ($\\max|\\Delta S|=0$); preg linear in $P$ ($R^2\\ge0.998$). The S31 slope of 0.68 and the remaining S variance come from per-slot quantizers (exact pre-activation $R^2$ 0.997, any jet-level $\\theta$ $\\le0.66$). With $g$ frozen, single-particle edits change S58, S31 by exactly 0. Untrained redraws: inputs also fit by $m\\le2$ ($R^2$ 0.96–0.99) but mix all orders; $\\theta$ $R^2$ 0.29–0.74.', None),
 ('Why it matters',
  'This names the last two of the 19 second-round objects (claim A3z). They are built from jet statements only, with no particle statements. S31 measures how far the particle-count centroid sits from the pT-weighted axis along $\\Delta\\eta$ and $\\Delta\\phi$; S58 falls as the count distribution elongates. These are fixed-frame, not rotation-invariant, which matches the azimuth-dependent response of E24/E24w (A13). Open: a causal test of the names by 90° rotations and mirrors.', None),
]
```
