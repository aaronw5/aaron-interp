# P50: what ParticleNet's EdgeConv-1/2 edge function computes, and which part of it sets ParticleNet's smaller axis response

Node P50 (loop C, round C11, exploratory; takes up P47; linked to C21, C5, C9, C1). Script `interp/p50_pn_edge_function.py`.
Jobs: `anrunw-ai-c-p50-run` (smoke, then a full run with 3,000 MLP steps; exit 0, 29 min) and `anrunw-ai-c-p50-long` (the same
run with 12,000 MLP steps; exit 0, 26 min). Both ran on a GTX 1080 Ti, then `anrunw-ai-c-p50-plot` (CPU, figure only).
Results: `results/particlenet/p50_pn_edge_s12000.json` (primary), `p50_pn_edge.json` (3,000-step repeat) and
`p50_pn_edge_smoke.json`. Figures: `p50_pn_edge_s12000.png` (primary) and `p50_pn_edge.png`. Logs: `logs/c-p50-run.log`,
`logs/c-p50-long.log`, `logs/c-p50-plot.log`. Every number below comes from the 12,000-step run unless marked otherwise.

## Conclusion

ParticleNet's first graph layer does not compute a fixed formula of rotation-invariant pair quantities in the way ParT's pair
bias does. Its edge messages also depend on the signed direction from a particle to its neighbour, and they are not additive in
their inputs. In the kinematics-only ParticleNet, a version of layer 1 that sees only invariant pair quantities keeps the
accuracy within 1 point. It lowers the network's response to the mass-preserving axis shift from 0.38 to 0.27 of a real mass
change, so the response falls further away from ParT's 0.8. Putting the signed direction back restores the response. The small
response is therefore not produced by the oriented part of the edges. Two questions stay open. In the full-input ParticleNet the
invariant-only layer costs about 4 points of accuracy, so its response cannot be compared. The φ-versus-η difference in the
response (the direction dependence) is not located in the oriented part.

## Definitions

- **ParticleNet (PN)**: the released JetClass tagger. Its 3 EdgeConv layers are EC1 (64 numbers per particle), EC2 (128) and
  EC3 (256). They are followed by a mean over particles and a 2-layer head. PN_full uses all 17 per-particle inputs. PN_kin uses
  the 7 kinematic ones: log pT, log E, log(pT/pT_jet), log(E/E_jet), ΔR to the axis, Δη, Δφ.
- **Edge message**: in EdgeConv layer l, each particle i takes its k = 16 nearest neighbours j. In EC1 these are nearest in
  (Δη, Δφ). In EC2 they are nearest in the 64 EC1 numbers. For each pair the layer computes
  $e_{ij} = h_l(x_i,\, x_j - x_i)$, where $h_l$ is 3 × [1×1 linear map, BatchNorm, ReLU], so $e_{ij} \ge 0$. The particle's
  new features are $\mathrm{ReLU}(\mathrm{shortcut}(x_i) + \frac{1}{16}\sum_j e_{ij})$. $x$ is the BatchNorm'd input in EC1 and
  the EC1 output in EC2. Only the mean over the 16 neighbours reaches the rest of the network.
- **Pair invariants**: ParT's four pair features, ln kT, ln z, ln ΔR and ln m²_ij. Here ΔR is measured in (rapidity, φ),
  kT = min(pT_i, pT_j)·ΔR, z = min(pT_i, pT_j)/(pT_i + pT_j), and m²_ij is the pair's invariant mass squared. A rigid rotation
  of the jet about the beam leaves all four unchanged. ParT's attention is mostly a fixed function of them (claim C1).
- **Feature sets for the fits**:
  - A = the layer's centre input $x_i$ plus the 4 pair invariants (and a flag for a padded neighbour).
  - B = A plus the signed pair offsets Δη_ij = Δη_j − Δη_i and Δφ_ij = Δφ_j − Δφ_i. This is the orientation of the pair:
    it says in which direction j lies from i.
  - C = A plus the neighbour's non-angular input differences: (x_j − x_i) of every input except Δη, Δφ and ΔR (5 columns
    in PN_kin's EC1, 14 in PN_full's).
  - null = A with the pair invariants taken from another of i's 16 neighbours, chosen at random.
  - ceiling = the layer's exact inputs (x_i, x_j − x_i).
  Note on $x_i$: in EC1, $x_i$ contains particle i's own signed Δη, Δφ and ΔR about the stored jet axis. A therefore excludes
  only the orientation of the *pair*, not the particle's position about the axis.
- **Fit forms**:
  - hinge = additive piecewise-linear model, "C1 style": every input column enters linearly, plus 7 hinges ReLU(u − q_k)
    at its 1/8 … 7/8 quantiles. There are no products of inputs. Fitted by ridge (λ = 1e-3 × mean diagonal), and the output
    is clamped at 0.
  - mlp = a multilayer perceptron D → 256 → 256 → C (ReLU), output clamped at 0. Adam 2e-3, cosine decay, 12,000 steps of
    4,096 edges.
- **Held-out R²** per channel = 1 − Σ(e − ê)²/Σ(e − ē)² on edges of held-out jets. The table gives the mean over live
  channels; channels with variance < 1e-6 (dead) are excluded: PN_full 1 of 64 in EC1 and 1 of 128 in EC2, PN_kin 6 and 9.
  Two levels: "edge" (each e_ij) and "mean over 16" (the per-particle mean over the 16 neighbours, which is what the network
  uses).
- **Formula substitution ("kept alone")**: all EC1 messages (arm ec1), or all EC1 and EC2 messages (arm ec12), are replaced by
  the fit's prediction from the same feature set. The fit is evaluated on the substituted network's own features. The rest of
  the trained network is unchanged, and the kNN graphs of EC2 and EC3 are recomputed from the substituted features.
- **d** = logit_W − logit_Z.
- **R** = per-jet slope through 0 of Δd(φ + 0.06) against Δd(matched dilation), over true W and Z jets (main, as planned). The
  value over W, Z and QCD jets (the P45 definition) is also stored. In words: R is the network's W-vs-Z response to a rigid
  0.06 rad φ shift of every particle, with the invariant mass fixed, divided by its response to a real angular dilation that
  raises the axis-moment mass √(M0·M2) by the same amount. M0 = Σ pT and M2 = Σ pT r² about the stored axis. R = 1 means the
  shift counts like a real mass change; R = 0 means the shift is ignored. **R_η** is the same with an η + 0.06 shift and its own
  matched dilation. **Direction dependence** = R_φ − R_η.
- **McNemar**: exact binomial test on the jets whose correctness differs between the substituted and the unedited network.
  **Paired permutation p for ΔR**: for each jet, the two arms' (Δd shift, Δd dilation) pairs are swapped with probability 1/2,
  2,000 times. The smallest possible p is 5e-4. **Holm**: step-down correction over the arms of a model (both forms and both
  layer sets).
- **Comparability rule (plan)**: the R of two arms is compared only when their accuracies differ by ≤ 1 point. Otherwise the
  pair is reported as not comparable.

## Procedure

- **Models and code check**:
  - PN_full and PN_kin with the official weights.
  - Own message-level forward vs weaver-core 0.4.17 `ParticleNet.forward`: max |Δlogit| = 0 on 334 jets for both models
    (required ≤ 1e-5).
  - Controls on PN_full: an untrained PN (PyTorch default initialisation, torch seed 5050) and a weight-permuted PN (the
    entries of every trained weight tensor permuted, seed 5053, BatchNorm running statistics kept).
- **Samples**: val_5M `*_123` of each class.
  - Evaluation: 1,000 random rows per class (default_rng(5050)), N 10,000.
  - Fit: 500 random rows per class from the remaining rows (default_rng(5051)), N 5,000, split by jet 80/20 into fit-train
    and held-out.
  - In each fit jet, 35 % of the real particles (torch seed 5052) are kept as centres, each with all 16 edges. This gives
    884,096 training and 219,680 held-out edges per layer.
  - The jets are rebuilt with `part_core.build` (η = jet_eta + sign(jet_eta)·Δη).
- **Unedited accuracy**:
  - PN_full: 84.66 % [83.9, 85.4] on the evaluation sample; 84.56 % on P1 (reference 84.56 %).
  - PN_kin: 69.87 % [68.9, 70.7] on the evaluation sample; 70.36 % on P1 (reference 70.36 %).
- **Edits** (stored axis fixed, masks unchanged, checked):
  - φ + 0.06 rad on every real particle. This is a rigid rotation about the beam: max |Δ log m| = 0.0006, and every pair
    invariant, the signed pair offsets Δη_ij, Δφ_ij and the EC1 kNN graph are unchanged. The shift reaches EC1 only through
    each particle's own Δφ and ΔR in $x_i$, and through the ΔR difference in x_j − x_i.
  - Matched dilation: (Δη, Δφ) × k_j with k_j = √(M0M2 shifted / M0M2 unedited). Mean k 1.083, mean Δ log m +0.073.
  - η + 0.06: max |Δ log m| 0.003. Its matched dilation has mean k 1.069.
- **Step 1**: record e_ij per channel before the mean over neighbours, on unedited fit jets.
- **Step 2**: fit A, B, C, null and ceiling × {hinge, mlp} for EC1 and EC2 of each model, and for EC1 of the two controls.
- **Step 3**: 2 forms × 2 layer sets × 5 feature sets = 20 arms per model. For each arm: accuracy (95 % bootstrap, 2,000
  resamples over jets), the 10-class confusion (stored in the json), McNemar vs unedited, calls changed, R_φ and R_η with
  bootstrap intervals, and ΔR vs unedited (paired bootstrap interval and permutation p). Between-arm comparisons are B − A,
  C − A and null − A within each form and layer set.
- **Repeat**: the 3,000-step run (same samples and seeds, shorter MLP training) is a check that the arm results do not depend
  on how long the fits are trained.

## Hypotheses and predictions

- **H_inv** ("shared invariant pair program", as ParT's pair bias):
  - fit A reaches R² ≥ 0.7;
  - the network with A substituted stays within about 2 points of unedited accuracy;
  - its R stays at PN's value (H0: ΔR = 0).
- **H_or** (oriented edges):
  - B fits much better than A;
  - A substitution moves R (toward ParT's 0.8 or toward 0);
  - the direction dependence R_φ − R_η sits in the oriented part, so A substitution removes it.
- **Null for the fits**: the shuffled-neighbour fit, which says how much the per-particle terms explain without the right
  neighbour. Training controls: the untrained and permuted networks, which say how much of B − A is a generic property of an
  EdgeConv layer.

## Numbers

### Unedited

| model | acc % [95 % CI] | R_φ (W, Z) [CI] | R_φ (W, Z, QCD) | R_η (W, Z) [CI] | R_φ − R_η [CI] | W → Z before → after φ |
|---|---|---|---|---|---|---|
| PN_full | 84.66 [83.9, 85.4] | 0.507 [0.478, 0.534] | 0.498 | 0.437 [0.406, 0.469] | +0.070 [+0.029, +0.112] | 10.6 → 23.6 % |
| PN_kin | 69.87 [68.9, 70.7] | 0.378 [0.351, 0.405] | 0.376 | 0.322 [0.290, 0.356] | +0.056 [+0.012, +0.097] | 17.4 → 28.1 % |
| untrained PN | 10.0 | 2.32 [2.18, 2.45] | | 0.30 | | |
| permuted PN | 10.2 | −0.12 [−0.27, +0.03] | | −0.11 | | |

### Step 2: held-out R² of the edge messages (mean over live channels; edge level / mean over 16 neighbours)

| layer, form | A: x_i + invariants | B: A + signed Δη, Δφ | C: A + non-angular diffs | null (shuffled neighbour) | ceiling (exact inputs) |
|---|---|---|---|---|---|
| PN_full EC1, mlp | 0.53 / 0.70 | 0.61 / 0.80 | 0.76 / 0.81 | 0.16 / 0.55 | 0.86 / 0.90 |
| PN_full EC2, mlp | 0.45 / 0.63 | 0.52 / 0.67 | 0.57 / 0.67 | 0.12 / 0.57 | 0.86 / 0.89 |
| PN_kin EC1, mlp | 0.80 / 0.81 | **0.93 / 0.94** | 0.83 / 0.84 | 0.10 / 0.51 | 0.93 / 0.94 |
| PN_kin EC2, mlp | 0.57 / 0.69 | 0.65 / 0.73 | 0.59 / 0.69 | 0.14 / 0.61 | 0.87 / 0.90 |
| PN_full EC1, hinge | 0.31 / 0.46 | 0.34 / 0.49 | 0.43 / 0.50 | 0.11 / 0.39 | 0.45 / 0.55 |
| PN_full EC2, hinge | 0.27 / 0.49 | 0.27 / 0.49 | 0.32 / 0.50 | 0.11 / 0.48 | 0.48 / 0.61 |
| PN_kin EC1, hinge | 0.49 / 0.52 | 0.50 / 0.54 | 0.51 / 0.53 | 0.08 / 0.38 | 0.48 / 0.56 |
| PN_kin EC2, hinge | 0.34 / 0.55 | 0.35 / 0.55 | 0.35 / 0.55 | 0.13 / 0.53 | 0.55 / 0.67 |
| untrained PN EC1, mlp | 0.78 / 0.95 | 0.79 / 0.95 | 0.93 / 0.97 | 0.47 / 0.89 | 0.96 / 0.99 |
| permuted PN EC1, mlp | 0.57 / 0.85 | 0.58 / 0.85 | 0.85 / 0.89 | 0.49 / 0.84 | 0.86 / 0.89 |

Gain from the signed orientation (B − A, edge R², mlp, EC1):

| network | B − A |
|---|---|
| PN_full | +0.09 |
| PN_kin | +0.13 |
| untrained PN | +0.01 |
| permuted PN | +0.00 |

Fraction of EC1 channels with edge R² ≥ 0.7 under A / B:

| network | A | B |
|---|---|---|
| PN_full | 0.27 | 0.38 |
| PN_kin | 0.78 | 0.95 |

### Step 3: formula substitution, MLP form, EC1 only (ec1)

| model, arm | acc % | Δacc points [CI] | McNemar Holm p | calls changed | R_φ (W, Z) [CI] | ΔR vs unedited [CI] | perm. p (Holm) | comparable | R_η | Δ(R_φ − R_η) vs unedited [CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| PN_full A | 80.83 | −3.83 [−4.42, −3.28] | 6e-40 | 10.3 % | 0.444 [0.415, 0.473] | −0.063 [−0.086, −0.040] | 5e-4 (n.c.) | no | 0.374 | +0.000 [−0.028, +0.030] |
| PN_full B | 82.94 | −1.72 [−2.14, −1.31] | 2e-15 | 5.7 % | 0.487 [0.460, 0.516] | −0.019 [−0.035, −0.003] | 0.020 (n.c.) | no | 0.420 | −0.002 [−0.024, +0.020] |
| PN_full C | 82.70 | −1.96 [−2.44, −1.52] | 1e-16 | 6.8 % | 0.468 [0.438, 0.498] | −0.038 [−0.056, −0.020] | 1e-3 (n.c.) | no | 0.399 | −0.000 [−0.022, +0.022] |
| PN_full null | 76.54 | −8.12 | 5e-106 | 16.9 % | 0.612 [0.588, 0.638] | +0.105 [+0.072, +0.140] | 5e-4 (n.c.) | no | 0.624 | −0.082 [−0.132, −0.032] |
| PN_full ceiling | 84.20 | −0.46 [−0.75, −0.19] | 0.0016 | 2.7 % | 0.494 [0.465, 0.522] | −0.013 [−0.024, −0.003] | 0.012 (0.012) | yes | 0.421 | +0.003 [−0.012, +0.016] |
| PN_kin A | 68.95 | −0.92 [−1.40, −0.45] | 8e-4 | 8.1 % | **0.274 [0.246, 0.301]** | **−0.104 [−0.117, −0.092]** | 5e-4 (**0.0025**) | yes | 0.203 | +0.015 [−0.002, +0.031] |
| PN_kin B | 70.05 | +0.18 [−0.05, +0.42] | 0.33 | 2.3 % | 0.365 [0.338, 0.392] | −0.013 [−0.017, −0.009] | 5e-4 (0.0025) | yes | 0.306 | +0.003 [−0.004, +0.009] |
| PN_kin C | 69.10 | −0.77 [−1.21, −0.33] | 0.0024 | 7.2 % | 0.299 [0.273, 0.324] | −0.079 [−0.090, −0.069] | 5e-4 (0.0025) | yes | 0.254 | −0.011 [−0.025, +0.003] |
| PN_kin null | 61.79 | −8.08 | 2e-80 | 24.7 % | 0.419 [0.393, 0.444] | +0.041 [+0.009, +0.071] | 0.013 (n.c.) | no | 0.376 | −0.013 [−0.055, +0.030] |
| PN_kin ceiling | 69.78 | −0.09 [−0.35, +0.17] | 0.54 | 2.5 % | 0.372 [0.346, 0.399] | −0.006 [−0.011, −0.001] | 0.013 (0.027) | yes | 0.309 | +0.007 [0.000, +0.014] |

n.c. = not comparable (accuracy more than 1 point from unedited), so the arm is excluded from the Holm family for ΔR. The
Holm family for ΔR has 5 arms for PN_kin and 2 for PN_full.

Between arms (MLP, EC1; ΔR_φ (W, Z), permutation p, Holm over the comparable pairs):

| comparison | ΔR | p | comparable |
|---|---|---|---|
| PN_kin B − A | +0.091 [+0.079, +0.104] | 5e-4 | no (accuracies 1.10 points apart) |
| PN_kin C − A | +0.025 [+0.018, +0.032] | 5e-4, Holm 0.003 | yes |
| PN_full B − A | +0.044 [+0.027, +0.062] | 5e-4 | no (2.11 points apart) |
| PN_full C − A | +0.024 [+0.009, +0.041] | 0.004 | no (1.87 points apart) |

Per-class accuracy of PN_kin, unedited → A-substituted (class order QCD, Hbb, Hcc, Hgg, H4q, Hqql, Z, W, t→bqq′, t→bℓν):

| class | QCD | Hbb | Hcc | Hgg | H4q | Hqql | Z | W | t→bqq′ | t→bℓν |
|---|---|---|---|---|---|---|---|---|---|---|
| A-substituted | 0.687 | 0.449 | 0.506 | 0.728 | 0.736 | 0.807 | 0.539 | 0.638 | 0.889 | 0.916 |
| B-substituted | 0.702 | 0.509 | 0.507 | 0.661 | 0.768 | 0.828 | 0.526 | 0.675 | 0.907 | 0.922 |

The A arm loses mostly Hbb (−6 points against B) and W (−3.7 points against B); full 10 × 10 confusions are in the json.

### Step 3: other arms (short)

**EC1 + EC2 replaced (ec12), MLP:**

| arm | PN_full: Δacc, R | PN_kin: Δacc, R |
|---|---|---|
| A | −9.0 points, R 0.607 | −3.9 points, R 0.367 |
| B | −6.1, 0.657 | −2.2, 0.498 |
| C | −4.6, 0.610 | −3.8, 0.403 |
| null | −20.6, 0.925 | −23.0, 0.905 |
| ceiling | −0.84, 0.530 (ΔR +0.023 [+0.008, +0.038], Holm 0.007) | −0.78, 0.365 (ΔR −0.013, Holm 0.073) |

Only the ceiling arms are within 1 point. The A, B and C arms are not comparable.

**Hinge form:**

| arm | PN_full | PN_kin |
|---|---|---|
| EC1 replaced (A, B, C, ceiling) | −9.9 to −12.2 points, R 0.48–0.57 | −4.0 to −4.8 points, R 0.25–0.32 |
| EC1 + EC2 replaced | −20 to −32 points | −16 to −29 points |

No hinge arm is comparable.

**3,000-step repeat** (`p50_pn_edge.json`; MLP, EC1, Δacc / R):

| arm | PN_full | PN_kin |
|---|---|---|
| A | −4.3 / 0.451 | −1.02 / 0.264 |
| B | −2.1 / 0.496 | +0.14 / 0.355 |
| C | −2.8 / 0.457 | −1.00 / 0.266 |
| ceiling | −1.26 / 0.487 | −0.15 / 0.364 |

The pattern is the same: in PN_kin, A lowers R by 0.11 and B keeps it.

## Reading

**Step 2 (what the edge function depends on).**
- The trained EC1 messages are predicted better when the signed pair offsets (Δη_ij, Δφ_ij) are added to the centre input and
  the four invariants: edge R² +0.09 (PN_full) and +0.13 (PN_kin).
- In PN_kin, A + orientation (B) reaches the ceiling set by the exact inputs (0.93 vs 0.93). A alone stays at 0.80.
- In the untrained and permuted networks the same addition gains ≤ 0.01. So the dependence on orientation comes from training
  and is not a property of any EdgeConv layer.
- The shuffled-neighbour null gives edge R² 0.10–0.16, so the pair invariants carry edge-specific information beyond the
  centre particle.
- The additive hinge form reaches at most 0.45–0.51 at edge level on EC1 even with the exact inputs (MLP 0.86–0.93). So the EC1
  messages are not a sum of one-variable functions of their inputs. This is unlike the ParT pair-bias hinge of C1.
- EC2 is fitted less well by every restricted set: A 0.45 / 0.57, ceiling 0.86 / 0.87.

**Step 3 (EC1 replaced by the fit).**
- In PN_kin the invariant-only arm A, the non-angular arm C and the oriented arm B are all within 1 point of the unedited
  accuracy.
- A lowers R_φ from 0.378 to 0.274 (Holm p 0.0025), and C lowers it to 0.299. B and the ceiling stay within 0.013 of the
  unedited R.
- So, at matched accuracy, an EC1 that does not see the pair orientation gives a smaller response to the mass-preserving
  shift, not a larger one.
- In PN_full the same ordering appears (A 0.444 < C 0.468 < B 0.487 < unedited 0.507). The A, B and C arms lose 1.7–3.8 points,
  so by the plan's rule they are not comparable with the unedited R. Only the ceiling arm (−0.46 points) is, and it changes R by
  −0.013.
- Replacing EC2 as well raises R in both models (PN_full A 0.607, null 0.93) but costs 4–23 points. These arms are not
  comparable. That the null arms have R near 1 indicates that degraded networks drift toward a larger response, and it is why
  the 1-point rule is needed.

**Direction dependence.**
- Unedited PN responds more to the φ shift than to the η shift: R_φ − R_η = +0.070 (full), +0.056 (kin).
- No comparable arm changes this difference: PN_kin A +0.015 [−0.002, +0.031], B +0.003.
- The φ/η difference measured here is therefore not located in the oriented part of EC1. This is a φ-versus-η check on
  unedited jets. It does not repeat P43v's offset-direction edit (E6e/E6r).

**Predictions.**
- **H_inv**:
  - A R² ≥ 0.7 holds for PN_kin (0.80) but not PN_full (0.53 per edge, 0.70 for the mean over 16 neighbours).
  - "Within 2 points" holds for PN_kin (−0.9) but not PN_full (−3.8).
  - "R unchanged" is excluded for PN_kin (ΔR −0.104, Holm p 0.0025) and untested for PN_full (not comparable).
- **H_or**:
  - "B better than A" holds (training-specific).
  - "A moves R" holds, toward 0 and not toward ParT's 0.8, in PN_kin.
  - "Direction dependence in the oriented part" is not supported.

**Physics meaning.** ParticleNet's first layer uses where a neighbour sits relative to a particle (above/below, left/right), not
only how far apart the two are and their pair mass. Removing that directional information does not make ParticleNet respond to
an axis shift more like ParT. It makes it respond less.

**Limits.**
- The fits are trained on 5,000 jets of one file per class. The MLP ceiling is 0.86–0.93, not 1.
- R² is a mean over channels and does not weight channels by how much the downstream network uses them.
- The rigid φ shift leaves Δη_ij and Δφ_ij unchanged. The B − A difference in R therefore measures how the fitted function
  routes the per-particle terms (Δφ_i, ΔR_i) when orientation is or is not available. It does not measure a response of the
  orientation inputs themselves.
- PN_kin B vs A is just outside the 1-point rule (1.10 points), so their direct comparison is flagged not comparable. Each of
  them vs unedited is comparable.

## Checks

- Own forward vs weaver: |Δlogit| = 0 (PN_full, PN_kin).
- Unedited accuracy on P1 reproduces the references: 84.56 % and 70.36 %.
- Ceiling substitution (the exact EC1 inputs through the MLP fit): −0.46 points (PN_full) and −0.09 points (PN_kin). This
  shows that the substitution machinery itself (kNN recomputation, masking, mean over neighbours) does not change the
  network's calls beyond the fit error.
- Edit checks: the φ shift changes log m by ≤ 0.0006; masks are equal.
- The 3,000-step repeat gives the same signs and ordering. The smoke run completed.
- Permutation p-values are at their floor (1/2001) for most effects.

## Explainer entry (for site/explain.py, Paper C tab)

- **Question**: Is ParticleNet's first pairwise operation a fixed formula of rotation-invariant pair quantities, as ParT's pair
  bias is? And is ParticleNet's smaller response to a mass-preserving axis shift set by the invariant or by the oriented part of
  that operation?
- **Background**:
  - ParT adds to its attention a learned function $U(\ln k_T, \ln z, \ln\Delta R, \ln m^2_{ij})$ of four pair invariants
    (C1).
  - ParticleNet instead computes, for each particle $i$ and each of its 16 nearest neighbours $j$, a message
    $e_{ij} = h(x_i, x_j - x_i)$ with a small network $h$, and averages the 16 messages. Its inputs $x_j - x_i$ contain the
    signed offsets $\Delta\eta_{ij}, \Delta\phi_{ij}$, i.e. the direction from $i$ to $j$.
  - P45v found that ParticleNet's first two layers are associated with its smaller axis response
    $R \approx 0.51$ (ParT $\approx 0.8$), where
    $R = \Delta d(\phi + 0.06) / \Delta d(\text{matched dilation})$ and $d = \ell_W - \ell_Z$.
- **Method**:
  1. Record $e_{ij}$ per channel on 5,000 jets.
  2. Fit it from (A) $x_i$ + the four invariants, (B) A + signed $\Delta\eta_{ij}, \Delta\phi_{ij}$, (C) A + non-angular
     $x_j - x_i$, a shuffled-neighbour null and the exact inputs. Each fit is made both as an additive hinge formula and as a
     small MLP. Report held-out $R^2$.
  3. Replace every EC1 message by the fit and run the rest of the trained network.
  4. Measure accuracy and $R$ on 10,000 other jets.
- **Result (figure `p50_pn_edge_s12000.png`)**:
  - Orientation adds 0.09–0.13 to $R^2$ in trained networks and 0.00–0.01 in untrained/permuted ones.
  - With PN_kin's EC1 replaced by the invariant-only fit (accuracy −0.9 points), $R$ falls from 0.38 to 0.27. With orientation
    included (+0.2 points), $R$ stays at 0.37.
- **Checks**: weaver identity; ceiling substitution within 0.5 points; 3,000-step repeat; Holm over arms; 1-point
  comparability rule.
- **Why it matters**: ParticleNet's first layer is not the invariant pair formula ParT uses, so the "first pairwise operation =
  fixed invariant formula" description does not carry over to ParticleNet. Its smaller axis response is not caused by
  directional pair information. Removing that information lowers the response further.
- **Glossary additions**:
  - *edge message (P50)*: the vector $e_{ij} = h(x_i, x_j - x_i)$ an EdgeConv layer computes for a particle and one of its
    16 nearest neighbours, before the mean over neighbours.
  - *signed pair offset / pair orientation (P50)*: $(\Delta\eta_j - \Delta\eta_i, \Delta\phi_j - \Delta\phi_i)$, the direction
    from particle $i$ to particle $j$. It is unchanged by a rigid rotation of the jet but distinguishes "above" from "below".
  - *formula substitution (P50)*: every message of a layer is replaced by a fitted function of a chosen input set, with the
    rest of the network intact.
  - *comparability rule (P50)*: two arms' $R$ are compared only when their accuracies are within 1 point.
  - *direction dependence (P50)*: $R_\phi - R_\eta$, the difference between the responses to a $\phi$ and an $\eta$ shift of
    the same size.
