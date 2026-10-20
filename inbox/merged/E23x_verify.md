# E23x verify (2026-10-20)

Node E23x: which centre carries JEDI-linear's direction-dependent (odd) shift response, and does it pass through the context-only units S31 (count dipole) and S58 (count quadrupole).

## Conclusion
On two new jet samples, the E23x numbers reproduce to within 0.01. Holding the two context units S31 and S58 at their unshifted values removes about a quarter of the direction-dependent shift response in the later layers (h4, h5, z), and none of it in the W-versus-Z score. All of the context round's contribution at h4–z goes through these two units. A size-matched control was added: freezing other second-round units that together carry the same amount of the shift response at their own layer. That control removes about twice as much downstream (h4 0.54 against 0.25). So S31 and S58 pass on less of their shift response than other units do; E23x's comparison with random pairs made them look stronger only because their own response is large. Still uncertain: the "hard 14 moves the W−Z score 2× more than soft particles" result was measured on jets that are 63 % top and 27 % gluon (2.5 % W/Z). It also needs $M_2$ to be held: moving the particles without holding $M_2$ gives equal soft and hard responses.

## Code review of `interp/e23x_centre_jedi.py`
Checks that found no defect:
- **Stage and head.** The head check is exact: max |Δlogit| = 0 on both new samples, and 100 % call agreement under the shift.
- **Padding.** Shift, 5a and 5b leave padded slots untouched (checked on the data: `shift_padding_untouched`, `soft_`, `hard_padding_untouched` all True).
- **Rank windows.** Padding sorts after the real particles (−pT = 0).
- **Freezes.** The S freeze broadcasts S0 over the cells. The g freeze leaves S31 and S58 exactly at S0 (max change 0.0).
- **Odd/even pairs.** Pairs are (k, k+4) within a ring.
- **Fit/test split.** Shared rows are removed (224 on the new pair).
- **5a/5b.** $M_2$ is held to < 1e−7.
- **Statistics.** Holm covers 24 tests.

Points that affect the reading (not code bugs):
1. **Unmatched null.** The arm-4 null is not matched in size, as the worker reported. Its direction matters: see the matched control below.
2. **5a vs 5b jet set.** The comparison uses jets feasible for both edits (640–673 of 3,000). On sample 5311 these are 62.9 % top, 27.2 % gluon, 7.4 % quark, 0.9 % W and 1.6 % Z, with mean multiplicity 61 against 46 overall. The "W−Z margin" ratio on this set is measured almost entirely on top and gluon jets.
3. **The pT moment in the regression.** $q_{p_T}$ is zero by construction in two of the three pooled edits (5a and 5b hold $M_2$). On the rigid edit it varies 9× less than $q_{\rm count}$. The worker states the size effect. The pooled $R^2$ comparison therefore cannot give the pT moment a large $R^2$, except through its gain on the rigid edit.
4. **5a/5b freeze shares.** E23x computed the S31 + S58 freeze shares for 5a and for 5b on different jet sets (2,409 vs 685). Here they are recomputed on the joint set (below); the values are unchanged.
5. **Memory.** The new rerun's pod peak was 2,047 MiB at the 2 GiB request (no OOM). The E23x original peaked at 1,812 MiB.

## Procedure
- **Pass 1 (`e23x-verify`).**
  - When this step started, the job `anrunw-ai-e23x-verify` already existed and had completed (17:05–17:16 UTC). It came from an earlier attempt of this verification step, with script `interp/e23x_verify.py` and seed 4313.
  - I read its code (no defects found except the V3 control, see below) and copied its results back: `results/jedi/e23x_verify.json`, `logs/e23x-verify.log`.
  - Sample `sample(1500, 4313)`, N 7,500, 214 rows shared with 2311.
- **Pass 2 (`e23x-verify2`, 1 CPU / 2 GiB).**
  - Wall 838 s; CPU 0.51 cores; peak 2,047 MiB.
  - Script `interp/e23x_verify_jedi.py`. `--part extra` writes `results/jedi/e23x_verify_extra.json`; `--part rerun` is the unchanged E23x `main()` on test `sample(1500, 5311)` and fit `sample(1500, 5310)` (224 shared rows removed), writing `results/jedi/e23x_verify_rerun.json`. Log: `logs/e23x-verify2.log`.
- **Edits.** As in E23x: rigid shifts of 30 and 60 mrad at 8 azimuths, giving 8 mirror pairs. Odd response $O=\tfrac12[f(\vec c)-f(-\vec c)]$ per sd-standardised unit. Share removed $=1-\sum O_{\rm frz}^2/\sum O^2$.
- **Added arms.**
  - (a) **Decomposition of the g freeze.** g is held, but S31 and S58 are restored to their shifted values. This measures g's contribution through the other 17 S units. Also the path difference $\sum(O_g-O_{31,58})^2/\sum O^2$.
  - (b) **Size-matched, on-manifold null.** Freeze random subsets of the 17 other live S units, k = 2…17 (210 subsets; at most 25 per k). The S-level odd power removed then spans 0.01–0.50, which brackets S31 + S58's 0.495. Fit the downstream share against the S-level share (linear) and compare S31 + S58 with the prediction at their S-level share.
    - H0: S31 + S58 remove the same downstream share as other-unit freezes that remove the same S-level odd power.
    - H1: they differ.
  - (c) **Random-subspace null.** Remove the shift-induced change of S along random 2-D subspaces of the 19 live S units. Pass 1: 300 + 400 draws. Pass 2: 120 draws. The coordinate case (e31, e58) reproduces the freeze exactly (difference 0.0).
  - (d) **5a/5b on one jet set.** Freeze shares of S31 + S58 and of all g on the jointly feasible jets. Pass 1 also ran two other $M_2$ treatments: rescale only the moved set to hold its own $M_2$ ("self"), and no $M_2$ hold ("free").

## Numbers
**Reproduction of E23x.** Odd power removed at h4 / h5 / z / margin:

| sample | S31 + S58 | all g | S31 | S58 |
|---|---|---|---|---|
| 2311 (E23x) | 0.255 / 0.249 / 0.208 / −0.009 | 0.259 / 0.244 / 0.208 / 0.038 | 0.208 / 0.229 / 0.200 / 0.005 | 0.040 / 0.015 / 0.017 / −0.012 |
| 4313 (pass 1) | 0.255 [0.249, 0.260] / 0.247 / 0.212 / −0.003 [−0.010, 0.004] | 0.259 / 0.240 / 0.212 / 0.045 | 0.208 / 0.227 / 0.203 / 0.010 | 0.040 / 0.013 / 0.018 / −0.009 |
| 5311 (pass 2) | 0.251 [0.246, 0.257] / 0.244 / 0.210 / −0.002 [−0.008, 0.005] | 0.256 / 0.238 / 0.210 / 0.045 | 0.206 / 0.223 / 0.202 / 0.012 | 0.039 / 0.014 / 0.017 / −0.011 |

- **Against the 136-pair null (5311).** Rank p 0.007 / 0.015 / 0.051 / 0.365; Holm 0.18 / 0.87 / 1 / 1. Same as E23x.
- **H1 (> 50 %).** P(share > 0.5) = 0 at every stage.
- **Calls at 60 mrad (5311).**
  - Calls changed 58.8 → 55.7 % with S31 + S58 frozen (diff [2.7, 3.4] points; McNemar 3,163 / 1,319, p 8e−172).
  - W → Z 60.1 → 62.5 % with S31 + S58 frozen; 69.5 % with g frozen.
- **5a / 5b (5311).**
  - Margin ratio soft/hard 0.50 [0.45, 0.55] on 640 jets. Pass 1: 0.47 [0.42, 0.52] on 659.
  - Held-out $R^2$, count moment vs pT moment: S 0.366 vs 0.001, h4 0.175 vs 0.001, z 0.146 vs 0.005, margin 0.002 vs 0.001.
  - Gain ratio q_count/q_pT: h4 0.87 [0.74, 0.98], z 0.51, margin 0.12.
  - Count-moment slope relative to rigid at h4–z: 0.38–0.61.
- **180° rotation, share removed at h4 (5311).** S58 0.147 (single-unit rank p 0.11); S31 0.21; both 0.357; all g 0.365.
- **Untrained redraws.** Live S sets are identical to E23x ([58], [58, 60], [30, 31]).

**(a) Decomposition of the g freeze.** g frozen with S31 / S58 left free removes:

| sample | h4 | h5 | z | margin |
|---|---|---|---|---|
| 5311 | 0.001 [0.000, 0.001] | −0.018 | −0.016 | 0.043 [0.036, 0.049] |
| 4313 | 0.000 | −0.018 | −0.015 | 0.039 |

- Path difference all-g vs S31 + S58: 0.004 (h4), 0.009 (h5), 0.012 (z), 0.038 (margin) of the odd power.
- At h4–z, the context round's contribution goes through S31 + S58. At the margin, g has a 4 % contribution that does not go through S31 + S58.

**(b) Size-matched coordinate null (5311, 210 subsets of the 17 other units).**
- **Mean shares by subset size** (S-level / h4 / margin): k 8: 0.24 / 0.20 / −0.15; k 12: 0.35 / 0.36 / 0.06; k 16: 0.48 / 0.56 / 0.66; all 17: 0.505 / 0.599 / 0.928. All 17 also give h5 0.754 and z 0.779.
- **Linear fit of the downstream share on the S-level share, evaluated at S31 + S58's 0.495:**

| stage | predicted | S31 + S58 | residual sd | z | p (two-sided) | Holm (4) |
|---|---|---|---|---|---|---|
| h4 | 0.540 | 0.251 | 0.088 | −3.26 | 0.0011 | 0.0034 |
| h5 | 0.632 | 0.244 | 0.158 | −2.45 | 0.014 | 0.028 |
| z | 0.660 | 0.210 | 0.129 | −3.48 | 0.0005 | 0.0020 |
| margin | 0.273 | −0.002 | 0.338 | −0.81 | 0.42 | 0.42 |

- **Downstream / S-level share ratio** (rank-based, no linear model): S31 + S58 0.51 at h4. Subsets with S-level share ≥ 0.2 (n 113): mean 0.98, 2.5–97.5 % range 0.23–1.48, and 91 % have a higher ratio (one-sided p 0.09).
- Competing predictions:
  - "S31 + S58 are a privileged route" predicts a share at or above the matched-null prediction. That is excluded at h4 and z (Holm ≤ 0.003).
  - "S31 + S58 carry a response that the head weights less than other S units" predicts a share below it. That matches the data.

**(c) Random-subspace null.**
- Removing the shift change along random 2-D directions gives downstream shares of −50 to −115 at h4, which means the odd response grows 7–11× in amplitude, at matched S-level share (both passes).
- These edits leave the set of S values the network produces and are not a usable null. The pass-1 z-scores of 4–5.7 from this control are not reported as evidence. The cause was not diagnosed. The change injected into the non-responding units S0 and S19 has a median of only 2e−5 sd.

**(d) 5a / 5b on the jointly feasible jets (5311, 673 jets).**
- S31 + S58 share at h4: soft 0.53 [0.51, 0.56], hard −0.02 [−0.03, −0.01]. All g: 0.53 / −0.01. At the margin: soft 0.00 [−0.11, 0.09], hard −0.05.
- Pass 1, other $M_2$ treatments (h4 share; margin soft/hard ratio):
  - self: soft 0.745, hard 0.088; margin ratio 0.49 [0.47, 0.51] on 2,199 jets.
  - free: soft 0.54, hard 0.07; margin ratio **1.03 [0.98, 1.08]** on 2,982 jets.

## Assessment of the reading
- **Reproduced on two new samples:**
  - S31 + S58 share 0.21–0.26 at h4–z and ≈ 0 at the margin.
  - All-g share equal at h4–z, now shown directly (g through other units: ≤ 0.001 at h4).
  - Calls, the 180° shares, the count-vs-pT $R^2$ and its size explanation, and the soft-move 0.5 / hard-move 0 freeze shares.
- **Needs correction:**
  - "The S31 + S58 share exceeds every random-pair share at h4 (raw p 0.007)". At matched S-level odd power, other S units remove about twice as much downstream (h4 0.54 predicted vs 0.25; z 0.66 vs 0.21; Holm ≤ 0.003). The raw-p excess comes from S31's large own response, not from a larger downstream weight.
  - "Moving only the 14 hardest at fixed M2 changes the margin 2.2× more". This holds with $M_2$ held, either way it is held (0.45–0.50; 0.49 with the self hold on 73 % of jets). Without the hold the ratio is 1.03. The cmp-hold comparison set is 63 % top and 27 % gluon, with < 3 % W/Z.
- **Thesis effect "weakens".** Kept, and the matched control strengthens it: S31 / S58 are not a preferential route for the odd response at h4–z.

## Corrected reading
Under 30–60 mrad rigid shifts, freezing S31 + S58 removes 0.21–0.26 of the odd response power at h4–z and 0.00 at the W−Z margin. This was measured on three samples (2311, 4313, 5311). Freezing all of g removes the same at h4–z. g acting through the other 17 S units removes ≤ 0.001 at h4 and 0.04 at the margin.

Freezing other live S units that remove the same S-level odd power (0.50) removes 0.54 at h4, 0.63 at h5, 0.66 at z and 0.27 at the margin (linear fit over 210 subsets; S31 + S58 below it at h4 and z, Holm p ≤ 0.003). All 17 others together remove 0.60 / 0.75 / 0.78 / 0.93. S31 + S58 thus pass on a smaller share of their S-level shift response to h4–z than the other S units do. Most of the odd response, including the margin's, goes through the 17 particle-reading S units.

With $M_2$ held, moving only the 14 hardest particles changes the margin's odd response about 2× more than moving only soft particles (ratio 0.45–0.50). On the jointly feasible set, these jets are 63 % top and 27 % gluon. Without the $M_2$ hold the two moves give equal margin responses (1.03). Half of the soft-move response at h4 passes through S31 + S58 (0.49–0.53), and none of the hard-move response does.

The odd response at S–z follows the unweighted moment $\sum r^2$ (held-out $R^2$ 0.15–0.37) more than $M_2$ (≤ 0.005). This mainly reflects the 9× larger change of $\sum r^2$ under a rigid shift. The slope depends on which particles move.

## Figure
`results/jedi/e23x_centre.png`: readable; axes in % of odd power; one message.
- The grey band is the unmatched 136-pair null. Read against it, S31 + S58 look stronger than other units.
- Adding the size-matched prediction (h4 0.54, h5 0.63, z 0.66, margin 0.27) as markers would show the opposite comparison. Not redrawn here (outputs restricted to e23x_verify*).

## Explainer addition (Checks step, site/explain.py E23X_EXPLAINER)
"Verified on samples 4313 and 5311 (shares within 0.01). Size-matched control: freezing other second-round units that remove the same share of the shift response at their own layer ($\approx 0.50$) removes $0.54$ at h4 and $0.66$ at z, against $0.25$ and $0.21$ for S31 + S58. The comparison of soft-only and hard-only moves holds with $M_2$ held (ratio $\approx 0.5$) and not without the hold (ratio $1.03$). The jets where both edits are feasible are mostly top and gluon jets."

## Files
- `interp/e23x_verify_jedi.py` (pass 2); `interp/e23x_verify.py` (pass 1, written by an earlier attempt of this step).
- `results/jedi/e23x_verify.json`, `results/jedi/e23x_verify_extra.json`, `results/jedi/e23x_verify_rerun.json`.
- `logs/e23x-verify.log`, `logs/e23x-verify2.log`.
