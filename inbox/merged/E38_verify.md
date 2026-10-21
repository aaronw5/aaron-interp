# E38 verification (job e38-verify)

Script `interp/e38_verify.py`; results `results/jedi/e38_verify.json` (volume: `$AI_RESULTS/jedi/e38_verify.json`); log `logs/e38-verify.log`.
Job: 1 CPU / 2 GiB, wall 1,672 s, 0.46 cores on average, pod peak memory 2,047 MiB (at the 2 GiB limit; a larger sample would need 3 GiB).
First submission landed on node fiona-1.famu.edu and stalled in `ceph_mdsc_wait_request` (no output after 10 min); I deleted that job and resubmitted
(landed on chi-dgx-node02.csuchico.edu). Adding fiona-1.famu.edu to the job.sh node exclusion list was refused by the permission system; the ledger
agent may want to add it.

## Conclusion
The basic E38 map reproduces on new jets: with E38's way of holding the jet mass fixed, almost every pT move between the hardest particles pushes
Z jets toward W, whichever way the pT is moved, and a random soft-particle reshuffle of the same GeV pushes harder. But this "same sign in both
directions" pattern depends on how the mass is held fixed. When the particle angles are left untouched and the soft particles' pT absorbs the
change instead, moving pT out of the hardest particle lowers the W-vs-Z score and moving pT into it raises it, by small amounts (about 0.05 logit).
In E36's minimal call flips, the three leading-particle internal objects carry about half (0.47) of the gluon-to-quark flips and none of the W/Z
flips; the pooled 3 % in E38 is an average of these opposite group results. Still uncertain: which way of holding the mass fixed is the
physically relevant one, and whether the group result survives correction for the four flip groups tested.

## 1. What was rerun and what was added
Sample: `sample(600, 3821)` minus jets of E38's main sample(1000, 3801): N = 2,944 jets (about 589 per class), 2,943 with >= 8 particles.
Exact accuracy 80.91 % (E38 main 81.86 %). Edit fraction f = 0.2, all 42 E38 cells (k, j in {0..5, rest}; rest = pT ranks 6–63 pooled).
Statistics: 1,000-sample jet bootstrap (95 % CI); McNemar with Holm over 42 cells for paired call changes.

Edits (all compared to the unedited jet; margins: W−Z = logit(W) − logit(Z) on true W + Z jets, q−g = logit(q) − logit(g) on true q + g jets):
- **T** — E38's edit, imported unchanged (`transfer_u` + `make_edit`): a fraction f of rank k's pT goes to rank j; then every particle's angles
  (Δη, Δφ) move by the minimum Euclidean-norm amount that restores the summed four-momentum (jet m, pT, E-scheme axis).
  Because a particle's four-momentum sensitivity to its angle is proportional to its pT, this choice moves the hardest particles' angles most.
- **null** — E38's soft-sector null (random zero-sum pT reshuffle among ranks 6–63, same GeV L2 as T, same angle projection).
- **pt_only / angle_only** — E38's decomposition (the pT transfer without the angle correction; the angle correction without the transfer).
- **EQ (new control: projection metric)** — the same pT transfer, the same four-momentum restored by angle moves, but with a weighted minimum norm
  $\min \sum_i (\delta\theta_i / w_i)^2$, $w_i \propto p_{T,i}^{-1/2}$, so that every particle's angle moves by a similar amount instead of the hardest particles
  moving most. Tests whether E38's map depends on the arbitrary choice of projection metric. Pt-weighted median angle move at 0->k: 13–21 mrad (T: 24–31).
- **PS (new control: angles fixed)** — single-rank cells only (k, j in 0..5; 30 cells). The same pT transfer; no particle angle is changed; the
  jet four-momentum is restored by the minimum-norm change of log pT of the rank 6–63 particles only. This is a pure pT-redistribution edit
  at fixed four-momentum. Compliance (|Δm|/m < 1e-3, axis < 1 mrad, |ΔpT|/pT < 1e-3) 96.7–99.9 % per cell. The compensation changes the soft
  particles' pT by a pT-weighted median 2–7 % (0.06–0.13 of the transfer's GeV L2). M2 = Σ pT r² changes < 0.003 sd.
- **PS_comp** — the PS compensation alone (the rank 6–63 pT changes, without the transfer), the matched control for PS.

E36 hold (Part B): E36's stored minimal flips (`e36_edits.npz`, 1,142 primary flips, identical to E38). Margin removed
= mean(m_edit − m_hold)/mean(m_edit − m_0), m = logit(target) − logit(unedited call), with S units held at unedited values via `run_jetwise`.
New: per flip group (W→Z, Z→W, q→g, g→q), every single live S unit, and all 968 live-S triples (not only the 500 nearest in displacement norm).

## 2. Code review of `interp/e38_hardness_channel_jedi.py` and `interp/e38_plot.py`
- Stages: S holds act on the pooled S layer after the quantizer (`run_jetwise` override 'S'); logits depend only on S, so holding all 19 live
  units at edited values reproduces the edit exactly (E38's identity check). No wrong stage, mask or quantizer found.
- Padding: `transfer_u` and the projection mask non-real slots (pT = 0); all used jets have >= 8 particles, so ranks 0–5 are real. No padded-slot leakage.
- Splits: naming fit / test halves are disjoint jets; reference MLPs train on a disjoint pool. No leakage found.
- The soft-sector null's two receivers get a·G1·w of pT (G1 = pT of about half of ranks 6–63, a up to 0.9). This can exceed the rank-5 pT, so after
  re-sorting the "soft" null can also change the identities of ranks 0–5. Not quantified here; it affects the label "soft-sector", not the arithmetic.
- `flips undone` is ~0.5 for any triple because the minimal flips sit at the decision boundary (E38 says this); margin removed is the usable metric.
- Report text: the Conclusion says "99 of 126" cells with Holm < 0.05 while the numbers say 100/126.
- Figure `results/jedi/e38_hardness.png` (read): panels are readable and match e38.json (map values, hold bars 0.03 / 0.38 / 0.11 / 0.36).
  Panel (c) shows only pooled hold values; the group split below is not visible there.

## 3. Results

### 3.1 E38's map reproduces (T vs null)
| quantity (f = 0.2, W−Z on W + Z jets) | E38 main | verify (N 2,944) |
|---|---|---|
| cells with positive W−Z change | 42/42 (≥ 0) | 41/42 (5→2: −0.007); 37 with CI > 0 |
| 0→1 | +0.32 [0.25, 0.38] | +0.29 [0.20, 0.37] |
| 1→0 | +0.11 [0.07, 0.16] | +0.13 [0.06, 0.19] |
| rest→0 | +0.37 | +0.26 [0.17, 0.35] |
| Z jets / W jets, mean over cells | +0.35 / −0.04 | +0.31 / −0.05 |
| T − null, cells < 0 | 126/126 | 42/42 (−0.32 to −0.06) |
| McNemar, null-only > T-only calls | 126/126 | 42/42; 38 Holm < 0.05 (0→1 94 : 168; 1→0 65 : 197) |
| q−g positive cells | 41/42 | 41/42 |

H0 (T map = soft null) is rejected in the same direction as E38 (null larger).

### 3.2 The same-sign pattern depends on the projection (EQ, PS)
| W−Z on W + Z jets | T (E38 projection) | EQ (equal-angle projection) | PS (angles fixed, soft pT absorbs) | PS_comp (compensation alone) |
|---|---|---|---|---|
| 0→1 | +0.286 [+0.199, +0.369] | +0.048 [−0.031, +0.121] | **−0.040 [−0.078, −0.000]** | +0.158 [+0.088, +0.231] |
| 1→0 | +0.125 [+0.063, +0.187] | +0.051 [+0.011, +0.092] | **+0.068 [+0.041, +0.098]** | −0.063 [−0.097, −0.028] |
| 0→2 / 2→0 | +0.18 / +0.13 | +0.04 / +0.03 | **−0.067 [−0.113, −0.022] / +0.050 [+0.024, +0.080]** | +0.21 / −0.08 |
| 0→5 / 5→0 | +0.25 / +0.03 | +0.23 / −0.03 | −0.058 [−0.106, −0.005] / +0.001 | +0.33 / −0.06 |
| positive cells | 41/42 | 26/42 (17 CI > 0, 7 CI < 0) | 15/30 (5 CI > 0, 6 CI < 0) | 18/30 |
| Z-jet mean / W-jet mean | +0.31 / −0.05 | +0.13 / +0.02 | +0.015 / −0.019 | +0.19 / −0.08 |
| cell correlation of W−Z with ΔpT^D (sd) | 0.15 | 0.48 | 0.61 | 0.88 |
| map correlation with T (single-rank cells) | 1 | 0.79 | −0.19 | |

- H0 (the W−Z response at fixed four-momentum has the same sign for k→j and j→k) vs H1 (opposite signs). Under PS, 0→1 and 1→0 have opposite
  signs with both CIs excluding 0 (0→1 upper bound −0.000); 0→2 / 2→0 likewise. Sum test 0↔1: +0.028 [−0.020, +0.081] (consistent with
  antisymmetry); under T the sum is +0.41 [+0.28, +0.53]. With angles fixed, pT into rank 0 moves the call toward W, pT out of rank 0 toward Z,
  as in the pT-only edit, but 3–6× smaller than T's direction-independent term.
- Under EQ, 0→1 drops from +0.29 to +0.05 and rest→5 rises from +0.20 to +0.48: the size and rank pattern of the T map depend on how the angle
  correction is distributed over particles.
- PS is not the sum of its parts: the compensation alone gives +0.16 at 0→1, PS gives −0.04.
- Competing hypotheses: (a) "direction-independent perturbation response" predicts T ≈ EQ ≈ PS, all positive — excluded by PS (opposite signs,
  CIs exclude 0 in 0↔1 and 0↔2). (b) "signed hardness response at fixed four-momentum" predicts antisymmetry under every mass-preserving edit —
  excluded by T and EQ (sums > 0). The data fit (c): a small signed pT-rank response plus an angle-correction response whose size and sign depend
  on the projection.

### 3.3 E36 hold by flip group (N 1,142; deterministic, pooled values reproduce E38 exactly)
| held at unedited values | pooled | W→Z (348) | Z→W (247) | q→g (34) | g→q (513) |
|---|---|---|---|---|---|
| S20 + S51 + S43 | 0.034 [0.010, 0.059] | −0.149 [−0.185, −0.113] | −0.143 [−0.181, −0.104] | −0.05 [−0.20, 0.14] | **0.467 [0.447, 0.487]** |
| S1 + S29 + S30 | 0.384 [0.348, 0.421] | 0.294 [0.235, 0.356] | 0.812 [0.753, 0.868] | 0.31 [0.15, 0.52] | 0.031 [0.018, 0.045] |
| all 968 other triples, median / 95th pct | 0.122 / 0.397 | 0.038 / 0.328 | 0.196 / 0.804 | 0.152 / 0.469 | 0.137 / 0.468 |
| rank p of S20 + S51 + S43 among 968 | 0.81 | 0.87 | 0.74 | 0.97 | 0.052 (Holm over 4 groups: 0.21) |

- The pooled 0.034 is an average of +0.47 on g→q flips (45 % of the flips) and −0.15 on W/Z flips (holding the leading-particle triple makes
  the W/Z flips stronger). The statement "the units that move most under the flips carry little of them" holds for W/Z flips, not for g→q flips.
- The radial triple's pooled 0.38 comes mostly from S29 alone (single-unit margin removed 0.337; S1 0.091, S30 0.096); among all 968 triples
  its rank p is 0.062 (E38: 0.032 among the 500 norm-matched).
- Single units (pooled margin removed): S20 0.010, S51 0.049, S43 0.094; largest S29 0.337, S50 0.132, S60 0.114.

## 4. Corrections to E38's reading
1. "Both directions of a transfer push the same way" holds for E38's minimum-norm angle projection only. With angles fixed (pT-space
   compensation in ranks 6–63), 0↔1 and 0↔2 transfers give opposite-signed W−Z changes (0→1 −0.040 [−0.078, −0.000], 1→0 +0.068 [+0.041, +0.098]).
2. "Only the pT-transfer-alone edit gives a signed, rank-ordered response ... that edit changes the jet mass": a signed response also exists at
   fixed jet four-momentum when the angles are not moved; it is small (≤ 0.07 logit; W + Z margin sd 4.74).
3. The T map's magnitude and rank pattern depend on the projection metric (EQ: 0→1 +0.05 vs +0.29; 26/42 positive cells).
4. The E36 hold: S20 + S51 + S43 remove 0.47 of the g→q flip margin (near the 95th percentile of all triples, rank p 0.052, Holm 0.21) and a
   negative share of W/Z flip margins. The pooled 0.03 should not be read as "these units carry little of the flips".
5. Unchanged: T map values, T < soft null (all cells), McNemar, q−g pattern, references, naming (not rerun).

## 5. Explainer draft (site/explain.py)
- **Question**: does E38's "direction-independent" response to pT moved between the hardest particles reflect the network, or the way the
  jet mass is held fixed?
- **Background**: moving pT between particles at different angles changes the jet mass $m^2 = (\sum_i p_i)^2$. To hold $m$, $p_T^J$ and the axis
  fixed, something else must change. E38 changed all angles by the minimum-norm amount $\delta\theta = -A^\top (A A^\top)^{-1} r$, where
  $A = \partial (\sum_i p_i^\mu)/\partial\theta$ and $r$ the four-momentum residual; since $A_i \propto p_{T,i}$, the hardest particles move most.
- **Method**: same transfers, two other ways to restore $\sum_i p_i^\mu$: (EQ) weighted minimum norm $\min \sum_i (\delta\theta_i/w_i)^2$ with
  $w_i \propto p_{T,i}^{-1/2}$ (equal angle moves); (PS) no angle change, minimum-norm change of $\log p_T$ of ranks 6–63.
- **Result**: under PS, $\Delta(\mathrm{logit}_W - \mathrm{logit}_Z)$ is −0.04 for 0→1 and +0.07 for 1→0 (opposite signs); under E38's projection both
  are positive (+0.29, +0.13). Figure suggestion: bar pairs (k→j, j→k) for T, EQ, PS.
- **Checks**: compliance ≥ 96.7 % per cell; M2 change < 0.003 sd; compensation-alone control.
- **Why it matters**: the sign of the network's response to "hardness" at fixed mass depends on what else is changed to keep the mass fixed;
  a claim about a hardness channel needs a stated choice of compensating edit.
- Glossary additions: *minimum-norm projection* (the smallest change, in a chosen metric, that restores a constraint); *pT-space compensation*
  (restoring the four-momentum by rescaling soft-particle pT with angles fixed); *margin removed* (share of an edit's change of
  logit(target) − logit(original call) that disappears when chosen internal units are held at their unedited values).
