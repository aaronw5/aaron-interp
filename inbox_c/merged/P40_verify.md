# P40v: verification of P40 (class-token MLP neuron atlas of ParT) on a fresh sample, with the missing controls

## Conclusion
On a new set of jets, the main P40 measurements repeat. Taken as a whole, the 1,024 hidden neurons in ParT's two final class-token layers are about as often tied to a single physics quantity or class as random rotated directions in the same space are (about half in both cases). A pre-registered set of high-impact neurons again pushes its labelled class the predicted way in 75–86 % of cases. Three statements in the P40 reading do not hold up:
- In ParT_full, nudging a random neuron now moves its labelled class the predicted way more often than a random direction does (58 % vs 38 %). P40 had found no difference.
- The lepton-class neurons named as examples are easy to read but have almost no individual effect on the output. P40 had not tested them causally.
- The 18-neuron "mass" set's effect comes mostly from two units, and on the new jets those two are labelled as W-class rather than mass.

Still uncertain: whether the 'selective' measure is a fair test of the neuron basis. Among units that pass the threshold, neurons are more selective than every rotation seed, so the "kin neurons below rotations" result comes from the threshold, not from selectivity.

## What was checked
**Code review of `interp/p40_neuron_atlas.py` and `interp/p40_plot.py`.**
- **Hooks.** They sit on `cls_blocks[b].fc1` (pre-GELU) and `cls_blocks[b].act` (post-GELU) of both class blocks. They take `o[0]`, the single class token of shape (1, N, 512). In weaver 0.4.17 a 512-d `post_fc_norm` LayerNorm follows the GELU and precedes `fc2`, so the edits are applied before that LayerNorm, as the script's docstring says. The class-block re-run (`tail`) reproduces the logits exactly (max |ΔL| = 0, both runs). No layer or index error.
- **Rows.** These are random `default_rng` rows, never first-k, with random halves, and the jet order is shuffled. Jets are rebuilt with the sign-aware η build (`part_core.build`: η = jet_eta + sign(jet_eta)·Δη). Padded particles are excluded from every observable (mask, z > 0 in D2, charged & mask for d0).
- **Edits.** The dilation scales Δη and Δφ by 1.05 at fixed pT and particle mass, giving a median Δlog m of 0.048. Only post-GELU activations are patched or steered.
- **Problems found (not code bugs):**
  1. The rotated steering control is stepped by its own projection s.d., which is 5.1× (full) and 3.1× (kin) the neuron s.d. in median. P40 noted this but did not run the equal-step control. It is run here.
  2. The 'selective' fraction requires 'clean' first. ParT_kin neurons pass 'clean' less often than rotated directions, so their lower selective fraction can come from the threshold alone. Two measures are added here: the selective share among clean units, and the unthresholded family gap.
  3. The lepton-class units b2n153, b2n249 and b1n138 are the most readable units. The reading lists them as examples of the "visible-effect" causal minority, but P40 never steered them, so that minority was never tested for them.
  4. The calibration ratio R is a net quantity. A subset of units can carry more than 100 % or less than 0 % of the response (see the oracle set below). "Carry 17–30 % of the response" should therefore be read as "patching them reproduces 17–30 % of the net change".
  5. Per-class circuits: p = 0.048 is the smallest value possible with 20 random sets (1/21), and it is reached in 3 of 20 uncorrected tests (10 classes × 2 models). After Holm correction over the 20, none is below 0.05. These circuits keep attention and the particle blocks, so they are not literal circuits in the user's definition (P40 states this).
- **Figure `results/part/p40_neuron_atlas.png`:**
  - In panel (a), the legend covers the ParT_kin neuron 'clean' bar.
  - Panel (b) shows the post-hoc |Δl| ≥ 0.05 subset without saying it was post hoc, and its rotated bars use the larger own-s.d. step.
  - In panel (c), the grey error bar is the min–max over 10 random draws, while the blue one is a 95 % bootstrap interval; the legend does not say so.
  - The panel titles state findings ("neurons are not more readable…").

## Procedure (job `anrunw-ai-c-p40-verify`, GPU, 786 s incl. smoke; script `interp/p40v_verify.py`)
- **Sample.** JetClass test_20M `<sample>_111.root` (P40 used `_110`). 2,000 random rows per class (`default_rng(4141)`), 10,000 labelling and 10,000 held-out jets, all rebuilt through `part_core.build`. The 'clean' threshold is recomputed from the untrained ParT (seed 0) on this sample:
  - post-GELU: 0.280 (full) and 0.374 (kin), against 0.276 and 0.359 in P40.
- **Unedited accuracy.**
  - P1 sample: ParT_full 85.93 %, ParT_kin 74.47 % (reference ≈ 85.9 / 74.5).
  - Fresh held-out half: 86.61 % / 75.18 %.
- **Atlas.** Same 22 labels, BH correction and 10 Haar rotations (raw and standardised) as P40. Added measures:
  - *sel|clean* = the selective fraction divided by the clean fraction;
  - *family gap* = the best label family's held-out |S| minus the second best (S = Spearman ρ for an observable, 2(AUC − 0.5) for a class), taken over all 1,024 post-GELU units with no threshold.
- **Steering.**
  - *What is steered:* 300 fresh random clean post-GELU units. To each one, ±2 s.d. is added (s.d. from the labelling half).
  - *Read-out:* Δl, the change in the target class's log-odds l = log p − log(1 − p), averaged over the 10,000 held-out jets.
  - *Direction-correct:* s·Δl(+2) > 0 with its 95 % bootstrap interval excluding 0, and s·Δl(−2) < 0. Here s is the predicted sign.
  - *Controls:* the same-index column of the Haar rotation Q0, steered by its own projection s.d. ("rot own", the P40 control) and by the neuron's s.d. ("rot eq": the same L2 size of edit, the missing control). For a random direction, the neuron's target class is an arbitrary class, so these controls give the chance rate of the direction-correct rule.
- **Pre-registered subset.** These are P40's units with |Δl| ≥ 0.05 at +2 s.d. (24 full, 36 kin), kept with P40's target class and sign and re-steered on the fresh jets. Also run: P40's 20 most readable units (top v), steered and then mean-ablated one at a time. Mean ablation sets the unit to its labelling-half mean; the read-out is the class recall, with McNemar on correct calls and BH over the 20.
- **Calibration.**
  - *Edit:* all particle angles (Δη, Δφ) × 1.05 at fixed pT. This is a real change of about 5 % in both the invariant mass and m_A (the axis-moment mass √(M0·M2)).
  - *R:* R_W = (mean change in W log-odds when only the patched units take their dilated-jet values) / (mean change in W log-odds under the full dilation).
  - *Sets:* (a) P40's 18 mass units, a fixed list; (b) the freshly labelled mass units; (c) 20 random neuron sets with the same per-block counts; (d) 20 random sets of clean units with non-mass labels; (e) 10 random rotated direction sets; (f) "oracle", the 18 units whose activation shifts most under the dilation, selected on the labelling half.
- **Rigid φ shift.** Every particle's Δφ is increased by 0.06 rad. This rotates the jet in φ, so the invariant mass is unchanged (max |Δlog m| 4·10⁻⁴); only the coordinates relative to the stored axis move. It is applied to the 1,000 held-out true-W jets. The same sets are patched with their shifted-jet values, and the ratio is taken against the full W log-odds change.

## Numbers (fresh sample; P40 value in brackets)
**Atlas, post-GELU, 1,024 units.** For the rotations, the mean is given with the [min, max] over 10 seeds.

| | ParT_full neurons | rot raw | rot std | ParT_kin neurons | rot raw | rot std |
|---|---|---|---|---|---|---|
| clean | 0.862 (0.868) | 1.000 | 0.984 | 0.529 (0.574) | 0.976 | 0.673 |
| selective | 0.485 [0.455, 0.516] (0.488) | 0.502 [0.464, 0.522] | 0.503 [0.484, 0.525] | 0.304 [0.276, 0.333] (0.312) | 0.520 [0.499, 0.535] | 0.332 [0.311, 0.348] |
| selective among clean | 0.563 | 0.502 [0.464, 0.522] | 0.511 [0.492, 0.533] | 0.574 | 0.533 [0.509, 0.547] | 0.493 [0.468, 0.512] |
| median family gap | 0.098 | 0.100 | 0.101 | 0.069 | 0.107 | 0.070 |

- **H0:** neurons are exchangeable with rotation seeds. For "selective among clean", neurons exceed all 10 seeds in both bases and both models; p = 1/11 = 0.09, the floor with 10 seeds, in each of 4 comparisons.
- **Selective and family gap:** neurons lie inside or below the rotation range.
- **Label stability:** P40's 300 steered units keep the same label on fresh jets in 89 % (full) and 93 % (kin) of cases.

**Steering at ±2 s.d.**

| set | model | neuron | rot own | rot eq | test (neuron vs rot eq) |
|---|---|---|---|---|---|
| 300 fresh random clean units | full | 175/300 = 0.583 [0.527, 0.638] (P40 0.533) | 0.380 | 0.383 | Fisher p 1.3e-6; Wilcoxon on s·Δl p 9e-6 |
| | kin | 154/300 = 0.513 [0.457, 0.569] (P40 0.537) | 0.513 | 0.517 | Fisher p 1.0; Wilcoxon p 0.04 |
| P40 |Δl| ≥ 0.05 subset, pre-registered | full | 18/24 = 0.75 [0.55, 0.88] | 11/24 | 11/24 | binomial vs 0.5: p 0.023; Fisher p 0.075; Wilcoxon p 0.013 |
| | kin | 31/36 = 0.86 [0.71, 0.94] | 19/36 | 19/36 | binomial p 1.3e-5; Fisher p 0.004; Wilcoxon p 6e-4 |
| fresh |Δl| ≥ 0.05 rule (sign only) | full | 25/32 (p 0.002) | 27/69 | 10/27 | |
| | kin | 28/32 (p 2e-5) | 20/44 | 13/22 | |
| P40 top-20 most readable | full | 16/20 = 0.80 | 7/20 | 7/20 | Fisher p 0.0095 |
| | kin | 13/20 = 0.65 | 7/20 | 7/20 | Fisher p 0.11 |

- **Effect sizes:** the median |Δl| at +2 s.d. is 0.0015 for neurons in both models. Rotated directions give 0.0049 (full) and 0.0042 (kin) at the equal step, and 0.023 / 0.022 at their own step. The equal-step control gives the same direction-correct rate as P40's own-step control (0.383 vs 0.380 full, 0.517 vs 0.513 kin), so the step size does not change the comparison.
- **Lepton-class units (ParT_full), +2 s.d.:**
  - b2n153 (fires for t→bℓν): Δl = +0.030;
  - b2n249 (fires for H→ℓνqq′): Δl = +0.299;
  - b1n138 (fires for H→ℓνqq′): Δl = +0.0007.
- **Mean ablation of any one top-20 unit** changes 0–1 calls of its class in ParT_full (recall 0.991 → 0.990–0.991) and 0–11 in ParT_kin. Only kin b2n496 (t→bqq′) reaches BH q 0.02: 11 calls lost, recall 0.889 → 0.878.

**Calibration** (angles × 1.05). The change in W log-odds is −0.163 [−0.179, −0.148] (full) and −0.173 [−0.191, −0.156] (kin).

| set | ParT_full R_W | ParT_kin R_W |
|---|---|---|
| P40's 18 mass units, fixed list | 0.284 [0.263, 0.308] (P40 0.303) | 0.175 [0.160, 0.193] (P40 0.173) |
| b1n92 + b1n108 alone (full) / b1n8 alone (kin) | 0.188 [0.174, 0.204] | 0.055 [0.049, 0.061] |
| fresh mass-labelled units | 0.084 [0.076, 0.092] (11 units; b1n92 and b1n108 now labelled W-class) | 0.163 [0.150, 0.180] (17 units) |
| 20 random neuron sets | mean −0.03, 95 % range [−0.23, 0.15], max 0.16; p 0.048 | mean 0.00, max 0.06; p 0.048 |
| 20 random clean non-mass sets | mean −0.02, max 0.14; p 0.048 | mean −0.04, max 0.05; p 0.048 |
| 10 random rotated sets | mean −0.01, max 0.08 | mean −0.01, max 0.01 |
| oracle: 18 most-shifted units | −0.73 [−0.83, −0.65] | −0.44 [−0.50, −0.39] |

The oracle result means the units whose activation moves most under the dilation push the W score the other way. The net response is therefore a sum of opposing unit contributions, and R for a subset is not bounded to [0, 1].

**Rigid φ shift (+0.06 rad, true-W jets, N = 1,000).**
- W → Z calls: 11.5 % → 31.9 % (full) and 16.5 % → 36.1 % (kin); McNemar p 5e-47 and 5e-38. This agrees with C3 (12 → 31 %, 16 → 37 %).
- Share of the W log-odds change (−0.97 full, −0.84 kin) reproduced by patching:
  - P40's 18 mass units: 0.261 [0.250, 0.273] (full), 0.211 [0.198, 0.223] (kin);
  - b1n92 + b1n108: 0.107; b1n8 (kin): 0.057;
  - the dilation oracle set: about 0;
  - 20 random neuron sets: mean 0.015 (max 0.18) and 0.005 (max 0.08); p 0.048.

## Hypotheses
- **H1 (C19 as stated):** neurons are at least 2× as readable as rotations, and more than 70 % of steered neurons are direction-correct. **Excluded on both samples.**
- **H0 (no privileged basis):** neurons are equal to rotations in readability and in steering. **Readability:** consistent on the selective fraction and the family gap; contradicted, weakly, on "selective among clean" (p = 0.09 floor, 4/4 in the same direction). **Steering:** contradicted for ParT_full on the fresh sample (p 1e-6); consistent for ParT_kin.
- **H2 (small causal minority):** a minority of neurons with |Δl| ≥ 0.05 move their class the predicted way well above chance. **Supported out of sample** (pre-registered: 18/24 and 31/36).

## Corrected reading
- **Clean and selective.** The 'clean' criterion saturates (86–100 % in every trained basis), and the replication confirms it.
- **Selective fraction (fresh sample):**
  - ParT_full neurons lie inside the rotation range (0.485 vs 0.46–0.53);
  - ParT_kin neurons lie below both rotation controls (0.304 vs 0.31–0.54);
  - among units that pass 'clean', neurons are more selective than all 10 rotation seeds in both models (0.56–0.57 vs 0.49–0.55, p = 0.09 per comparison). The ParT_kin deficit therefore comes from fewer ParT_kin neurons passing the threshold, not from less selective ones.
- **Steering a random clean neuron by 2 s.d.:**
  - direction-correct in 53–58 % (ParT_full) and 51–54 % (ParT_kin) across the two samples;
  - median |Δ log-odds| about 0.001;
  - same-index rotated directions give 38–47 % (full) and 51–52 % (kin), at both their own step and the neuron's step;
  - in ParT_full the neuron advantage is 6 points on the P40 sample (Wilcoxon p 0.10) and 20 points on the fresh sample (Fisher p 1e-6), so the claim "does not differ in ParT_full" does not hold. ParT_kin shows no difference.
- **High-impact minority.** P40's post-hoc set of units with |Δl| ≥ 0.05 was re-tested as a pre-registered set on fresh jets. It is direction-correct in 18/24 (full) and 31/36 (kin), against 11/24 and 19/36 for equal-step rotated directions.
- **Lepton-class units.**
  - b2n153 and b1n138 are among the most readable units (held-out AUC score v ≈ 0.98–1.0), but steering them changes their class log-odds by only 0.03 and 0.0007.
  - Ablating any one of them changes no calls.
  - They are readable but individually weak, not part of the visible-effect minority.
- **Mass units.** P40's 18 axis-mass-labelled units are a fixed list. Patching them reproduces:
  - 28 % (full) and 17 % (kin) of the net W log-odds change under a real 5 % mass increase;
  - 26 % / 21 % of the change under a rigid φ shift that leaves the invariant mass unchanged.

  Random neuron sets give 0 ± 0.2. In ParT_full, two units (b1n92, b1n108) give 0.19 of the 0.28, and on fresh jets these two are labelled W-class rather than mass. The set chosen by the mass label alone on the fresh sample gives 0.08 (full) and 0.16 (kin). Because the units that move most under the dilation give R = −0.73 / −0.44, R is a net partial contribution, not a share of a fixed total.
- **Mean-hidden-layer and per-class circuits.** These were not re-run. The per-class p = 0.048 values do not survive Holm correction over 20 tests.

## Corrected Conclusion (for P40)
The neurons inside ParT's final class-token layers are, as a set, about as readable as random rotated directions in the same space: roughly half respond selectively to one class or physics quantity in either case. Nudging a single neuron moves its labelled class the predicted way only slightly more often than a random direction does (58 % vs 38 % in the full-input model on fresh jets, no difference in the kinematics-only model), and usually by a negligible amount. A small, reproducible minority of neurons have clear causal roles, among them a few that follow the axis-based jet mass and, patched together, reproduce about a quarter of the W-versus-other response to either a real mass change or a rigid φ shift of the jet. Still uncertain: which units count as "mass units" depends on the sample, and much of the effect sits in two units.

## Explainer entry (for site/explain.py; ledger agent to integrate)
- **Question.** Is the neuron basis of ParT's class-token MLP special, in the sense that single neurons are easier to read and to steer than arbitrary directions?
- **Background.** If training aligned features with neurons, a neuron would correlate with one quantity and steering it would move one class. A random orthogonal rotation $Q$ of the 512-d hidden space describes the same computation, since $W_2 Q^\top (Q h) = W_2 h$, so its coordinates are the fair null.
- **Method.**
  - Steering: $h_i \to h_i + k\,\sigma_i$ with $k = \pm 2$. Read-out: $\Delta l_c$, the change of $l_c = \log p_c - \log(1-p_c)$.
  - Control: $h \to h + k\,\sigma_i\,q_i$, with $q_i$ a column of $Q$ (equal L2 step).
  - Calibration: $R_W = \overline{\Delta l_W^{\text{patch}}}/\overline{\Delta l_W^{\text{edit}}}$.
- **Result.**
  - Selective fraction: 0.49 (neurons) vs 0.50 (rotations) in ParT_full.
  - Pre-registered high-impact neurons: direction-correct in 75–86 % vs 46–53 % for rotations.
  - Mass units: $R_W$ = 0.28 / 0.17 under dilation and 0.26 / 0.21 under a φ shift.
- **Checks.** Fresh file, seed and halves; equal-step control; random neuron, non-mass and rotated sets; an oracle set.
- **Why it matters.** Most of the class-token MLP is not organised neuron by neuron, so a full map needs feature-level tools (transcoders, P13). The few causal mass units respond to the axis-based mass, which agrees with C3.
- **Glossary additions:**
  - **equal-step control:** a rotated direction steered by the same absolute amount as the neuron;
  - **selective among clean:** the selective fraction divided by the clean fraction;
  - **family gap:** the best minus the second-best label family's held-out |score|;
  - **oracle set:** the units whose activation moves most under the edit.

## Files
- `/Users/anrunw/Documents/aaron-interp/interp/p40v_verify.py`
- `/Users/anrunw/Documents/aaron-interp/results/part/p40_verify.json` (and `_smoke`); on the volume `/j-jepa-vol/aaron-interp/results/part/p40_verify*.json`
- `/Users/anrunw/Documents/aaron-interp/logs/c-p40-verify.log`
