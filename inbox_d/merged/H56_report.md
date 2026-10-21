# H56: seed and shuffled-label nulls for the axis-direction (H39), charge-relabel (H42) and count (H24v / H26) results (loop D, round D11)

Status: (1) axis direction and (3) count response done. (2) charge relabel was not run because none of the four kinpid trainings has a checkpoint yet. Those trainings ran out of memory at 40 GiB; all four were relaunched at 48 GiB and are running (jobs listed at the end).

## Conclusion

**Axis direction (D4m).** We moved each jet's coordinates onto the winner-take-all axis (WTA) or onto the hardest particle's direction. The "direction part" is the change in tagger calls caused by that move, minus the change caused by a shift of the same length in a random direction. A second training seed puts this direction part inside the same band as untrained networks, as seed 0 did. Its sign agrees across seeds in 38 of 42 cells, but its size moves between seeds by up to about 1 to 2 percentage points (for example quark–gluon PFN at 50 % quark efficiency: +1.16 → +0.08 points). So the two quark–gluon PFN cells that H39 found closest to the edge of the untrained band do not repeat with seed 1.

**Count response (D7).** Splitting a soft particle exactly in two keeps the energy flow the same and adds one particle. The trained quark–gluon ParT_kin and ParticleNet_kin respond to this split by +0.006 to +0.008 and +0.004 log-odds per added particle. A seed-to-seed difference of at most 0.0016 changes neither number by much. In the reviewer's form: beyond IRC-safe energy flow, the networks' response to N is about 4–8 % of what a likelihood ratio in N alone gives (+0.10 per particle). The shift between Herwig-trained and Pythia-trained networks measured in H24v (−0.013 to −0.015 per particle) is about 8 times this seed spread for ParT and far larger than it for ParticleNet.

**Shuffled-label nulls.** Every network trained on class-independent labels ended as a constant function: the same output for every jet, 50 % accuracy. Its responses are therefore zero by construction. This control rules out an effect that comes from architecture plus input geometry alone after training. It does not provide a spread to test against. The untrained (epoch-0) networks remain the informative null.

**Still uncertain.** With one extra seed, the seed spread is a single difference per cell. The QG ParT seed-1 axis result and the QG ParticleNet shuffled-label result use checkpoints from runs that were still training. The charge test (D16) has no seed or shuffled-label result yet.

## Procedure

- **Script.** `interp/h56_seed_nulls.py`. It uses `interp/h39_axis_null.py` for the edits, scoring and statistics of the axis test, and `interp/h24_qg_generator_models.py` / `h24_verify.py` for the count edits.
- **Jobs.**
  - GPU job `d-h56-smoke` (100 jets per class, 3 directions; exit 0).
  - GPU job `d-h56-run` (RTX 2080 Ti, 4,766 s, exit 0).
  - CPU job `d-h56-fig` (figure).
- **Networks per architecture.** All were trained on the same training data with the authors' settings.
  - **s0:** the networks used so far (H2 / H4 / H36 / H39).
  - **s1:** second seed, torch / energyflow seed 1, built before training (`h39_seeded_net.py`, `h39_train_ef.py`).
  - **shuf:** labels independent of the class, same number of steps.
    - weaver: label = hash(jet pT, η, φ) mod 2. This agrees with the true class for 0.500 of test jets.
    - energyflow: permuted labels.
  - **init:** six epoch-0 networks per architecture, the same ones as in H39.
    - ParT / ParticleNet: built after `torch.manual_seed(1000+k)`.
    - EFN / PFN: energyflow seed k.
  - **Provisional (still training at run start), labelled "prov.":**
    - QG ParT_kin s1: epoch 16 of 0–19 (`s1p`). It is compared with the s0 checkpoint of the same epoch 16 (`s0e`), so the training stage is matched.
    - QG ParticleNet_kin shuf: epoch 10 (`shufp`).
    - QG ParT_kin s1 finished (best epoch, test accuracy 0.827) while the run was in the count step, so the count analysis uses its final best checkpoint.
  - Top ParT was not retrained: there is no Top ParT s1 or shuf.
- **(1) Axis direction (H39 protocol, unchanged).**
  - Samples: Top Landscape test, `data_top.sample(5000/class, seed 3901)`; QG Pythia test, `data_qg.sample(5000/class, seed 3911)`. 10,000 jets each, random rows.
  - Edits:
    - WTA: every particle's (Δη, Δφ) is recomputed about the WTA axis.
    - HP: the same about the hardest particle.
    - Both are a rigid translation of all particles by the axis offset δ_j.
    - Control R_k (k = 1…20): the same per-jet length |δ_j| in a uniformly random direction.
  - Calls: sign of the log-odds; and the working points where the unedited signal efficiency is ε_S = 0.3 and 0.5. Signal = top / quark. The threshold is fixed from the unedited scores.
  - **D** (percentage points, pp) = mean over jets of 1[call changes about the axis] − (1/20)Σ_k 1[call changes under R_k]. This is the part of the call change that comes from the shift pointing at the hardest particle rather than in a random direction.
  - Per network: 95 % jet bootstrap CI (B = 2000).
  - Against the init band: z and two-sided t prediction p (df = n − 1), as in H39. Only non-degenerate inits count: score SD > 10⁻⁶ and ≥ 0.5 % of calls moved by R.
  - **Seed difference:** D(s1) − D(s0) per jet, with a paired jet bootstrap CI and a sign-flip permutation p (P = 20,000).
  - **McNemar test:** an exact binomial test on the jets where only one of two networks changes its call about the axis, for s1 vs s0 and shuf vs s0.
- **(3) Count response (H24v protocol).**
  - Sample: QG Pythia, `h14.load_sample('py', 2000/class, seed 7243)`. 4,000 jets; edits applied to the 3,998 with N ≥ 10.
  - **sp_k:** k (1–5) particles from the softer half of the jet are each split into two daughters with half the pT at the same (y, φ). N rises by k. Energy flow, pT^D, girth and mass are unchanged, so every IRC-safe observable is unchanged.
  - **ct_k:** control with ΔN = 0 and pT^D matched (H24).
  - **Metric:** per jet, (slope over k of Δlog-odds toward gluon under sp_k) − (slope under ct_k), in log-odds per added particle.
  - **References:**
    - LR(N): negative-binomial likelihood ratio in N alone (`h24_base.json`), the response of a pure counter.
    - EFN: IRC-safe by construction, so exactly 0 on sp_k.
    - part_kin with permuted weights (H24v null).
  - Statistics: 95 % jet bootstrap (B = 2000) and sign-flip p. The seed spread is |s1 − s0| with a paired bootstrap CI.
- **Multiple testing.** Holm over all 283 tests in this run: axis vs init, paired s1 − s0, shuf − s0, McNemar, and count tests excluding the init networks.
- **H0 (training artefact absent).** Shuffled-label response inside the epoch-0 band, and seed difference inside the jet-bootstrap CI.
- **H1 (effect from architecture / input geometry, not learning).** Shuffled-label networks reproduce the effect.
- **D4m formula-driven reading.** D(s0) and D(s1) outside the init band and of the same sign.
- **Generic-translation reading.** D inside the band, sign not fixed across seeds.

## Implementation checks and accuracies on the samples used

- **Axis edit vs shift of the stored arrays (TN):** max |Δlogit| = 0 on both samples.
- **Torch EF port vs numpy energyflow:** ≤ 5.7 × 10⁻⁶.
- **Seed-0 reproduction:** the s0 rows reproduce H39 exactly (same sample and code), e.g. Top ParT WTA wp50 D −2.12, QG EFN +3.88.
- **Count reproduction:** the count numbers reproduce H24v on the same seed-7243 sample (ParT_kin +0.0079, PN_kin +0.0042; H24v +0.0079, +0.0042).

| sample | network | s0 acc | s1 acc | shuf acc (score SD) |
|---|---|---|---|---|
| Top (10k) | ParT_kin | 94.18 % | – | – |
| Top | ParticleNet_kin | 94.05 % | 94.05 % | 50.0 % (0) |
| Top | PFN | 93.13 % | 93.20 % | 50.0 % (0) |
| Top | EFN | 92.43 % | 92.20 % | 50.0 % (0) |
| QG Py (10k) | ParT_kin | 82.45 % | 82.53 % (prov. ep 16; s0 ep 16: 82.65 %) | – |
| QG Py | ParticleNet_kin | 82.36 % | 82.38 % | 50.0 % (0; prov. ep 10) |
| QG Py | PFN | 81.70 % | 81.15 % | 50.0 % (0) |
| QG Py | EFN | 80.74 % | 80.83 % | 50.0 % (0) |
| QG Py count (4k) | ParT_kin / PN_kin | 82.20 / 82.43 % | 82.37 (final) / 82.20 % | – / 50.0 % |

- **LorentzNet:** 94.30 % (Top), 84.15 % (QG). Its axis D is 0 by construction (four-vector inputs).
- **Init networks:** 34–64 % accuracy.
- **Degenerate shuffled networks:** "score SD 0" means the network outputs the same log-odds for every jet. The small nonzero D of the energyflow shuf networks at the working points (+0.4 to +0.55 pp) comes from float-level ties at the threshold. They are flagged degenerate.

## Numbers

### (1) Axis direction, calls at ε_S = 0.5, D in pp [95 % CI]; init band = [min, max] of 6 epoch-0 networks

| sample | arch | axis | s0 | s1 | s1 − s0 [CI], Holm p | shuf | init band |
|---|---|---|---|---|---|---|---|
| Top | ParticleNet | WTA | −0.58 [−0.88, −0.29] | −0.56 [−0.87, −0.25] | +0.03 [−0.31, +0.39], 1 | 0 (const.) | [−0.86, +1.96] |
| Top | PFN | WTA | −1.12 [−1.58, −0.65] | −1.34 [−1.85, −0.83] | −0.21 [−0.79, +0.38], 1 | 0 (const.) | [−3.27, −0.07] |
| Top | EFN | WTA | −1.13 [−1.73, −0.56] | −1.35 [−1.97, −0.75] | −0.22 [−0.90, +0.49], 1 | +0.49 (const.) | [−3.02, +0.49] |
| Top | ParT | WTA | −2.12 [−2.57, −1.69] | – | – | – | [−1.82, +1.86] |
| QG | ParT (s1 prov. ep 16 vs s0 ep 16) | WTA | +0.29 [+0.10, +0.49] | +0.35 [+0.13, +0.56] | +0.06 [−0.24, +0.34], 1 | – | [−1.17, +0.37] |
| QG | ParticleNet | WTA | +0.35 [+0.10, +0.61] | +0.77 [+0.50, +1.04] | +0.42 [+0.09, +0.74], 1 | +0.03 (const., prov.) | [−0.21, +0.92] |
| QG | PFN | WTA | +0.68 [+0.30, +1.05] | −0.02 [−0.31, +0.26] | −0.70 [−1.15, −0.25], 0.27 | +0.04 (const.) | [−1.03, +0.23] |
| QG | PFN | HP | +1.16 [+0.79, +1.53] | +0.08 [−0.21, +0.38] | −1.08 [−1.52, −0.63], 0.012 | +0.07 (const.) | [−1.42, +0.24] |
| QG | EFN | WTA | +3.88 [+3.34, +4.44] | +3.61 [+3.08, +4.15] | −0.27 [−0.90, +0.35], 1 | +0.55 (const.) | [−0.31, +6.37] |

- **Against the init band:** no s0, s1 or provisional cell is outside it after Holm, in any of the 3 call definitions × 2 axes × 2 datasets.
  - Smallest uncorrected prediction-interval p: QG ParticleNet s1, ε_S 0.3, WTA, +0.54 pp vs init [−0.10, +0.30], z 2.8, p 0.047. Holm p = 1.
- **Sign agreement s0 vs s1:** same sign in 38 of 42 network × call × axis cells. The four opposite-sign cells are:
  - Top ParticleNet at the sign call, WTA;
  - Top ParticleNet at ε_S 0.3, HP;
  - QG ParticleNet at the sign call, WTA;
  - QG PFN at ε_S 0.5, WTA.

  In each of these, one of the two |D| values is ≤ 0.41 pp.
- **Seed differences with Holm p < 0.05:**
  - Top EFN at the sign call: WTA −1.66 [−2.17, −1.15]; HP −1.97.
  - Top ParticleNet HP at the sign call: −0.94.
  - QG PFN at the sign call: +1.05 / +1.09.
  - QG PFN HP at ε_S 0.5: −1.08.
- **McNemar s1 vs s0:** significant after Holm in Top PFN at ε_S 0.3 / 0.5 and in QG PFN at ε_S 0.5. There the two seeds change different numbers of calls (e.g. QG PFN ε_S 0.5 WTA: 364 jets change only for s0, 231 only for s1). The jet-level sets of moved calls differ even where D agrees.

### (3) Count response, exact split − ΔN = 0 control, log-odds toward gluon per added particle (3,998 Pythia jets)

| network | s0 | s1 | |s1 − s0| [CI of s1 − s0], sign-flip p | shuf | init band (6) |
|---|---|---|---|---|---|
| ParT_kin | +0.0079 [0.0069, 0.0089] | +0.0063 [0.0053, 0.0073] | 0.0016 [−0.0024, −0.0008], 2 × 10⁻⁴ | – (training not done) | [−0.077, +0.036] |
| ParticleNet_kin | +0.0042 [0.0030, 0.0054] | +0.0042 [0.0030, 0.0055] | 0.00004 [−0.0010, +0.0012], 0.95 | 0 (const., prov. ep 10) | [−0.0005, +0.0006] |
| PFN | +0.0207 [0.0200, 0.0215] | +0.0252 [0.0246, 0.0259] | 0.0045 [+0.0038, +0.0051], 5 × 10⁻⁵ | 0 (const.) | [−0.44, +0.57] |
| EFN | −0.0033 (split part 0.0000; control part +0.0033) | −0.0032 | 0.0001 [−0.0001, +0.0003], 0.43 | 0 (const.) | [−0.0001, +0.0001] |
| LR(N) | +0.1006 [0.0995, 0.1017] | | | | |
| ParT_kin, weights permuted | −0.0057 [−0.0060, −0.0053] | | | | |

- **Raw parts:**
  - ParT_kin s0 / s1: split +0.0106 / +0.0089 per particle, control +0.0027 / +0.0025.
  - ParticleNet_kin: split +0.0071 / +0.0071, control +0.0029 / +0.0028.
- **EFN:** its split response is 0 to float precision, as an IRC-safe network must give. Its nonzero sp − ct comes from its response to the ΔN = 0 control edit.
- **Against the init band:**
  - Trained ParticleNet_kin lies far outside its init band (z ≈ 10). Untrained ParticleNet barely responds to the split.
  - Trained ParT_kin and PFN lie inside very wide init bands. Untrained ParT responds to a split by −0.08 to +0.04 per particle.
- **H26 comparison:** Herwig-trained − Pythia-trained shift from H24v, on Pythia jets −0.0127 (ParT) and −0.0145 (ParticleNet) per particle. That is 8× (ParT) and > 100× (ParticleNet) the seed spread measured here.

## Reading (scoped to what was measured)

- **Axis direction.**
  - H39's one-seed result (trained D inside the untrained band) holds for the second seed on Top (ParticleNet, PFN, EFN) and QG (ParticleNet, PFN, EFN, provisional ParT).
  - The D4m formula-driven prediction (outside the band, same sign across seeds) is not observed in any cell.
  - Seed-to-seed differences reach 1–2 pp in some cells (QG PFN, Top EFN at the sign call). That is the same size as the init-band SD (0.4–2.5 pp). So a single-seed D of this size cannot be attributed to training.
  - The cells H39 flagged as nearest the edge (QG PFN ε_S 0.5, z +2.1 / +2.3) move to the band centre with seed 1. This fits the generic-translation reading.
- **Shuffled labels.**
  - All shuffled-label trainings (Top ParticleNet, Top / QG PFN and EFN, QG ParticleNet provisional) converge to a constant output. Their D and count response are therefore 0 and lie inside every epoch-0 band that contains 0.
  - This is consistent with H0 for every test. It does not test whether a non-constant network trained without class information would reproduce the effects (H1).
  - The McNemar "shuf vs s0" tests (Holm p ≈ 0) only show that a constant network changes no calls.
- **Count (D7).**
  - The trained ParT_kin / ParticleNet_kin response to N at fixed IRC-safe energy flow is +0.004 to +0.008 log-odds per particle. Both seeds give the same value to within 0.0016.
  - LR(N) gives 0.10, so the response is 4–8 % of what a pure counter would give.
  - In the reviewer's form: the networks show almost no response to N beyond IRC-safe energy flow, and the residual is reproducible across seeds (ParticleNet: identical; ParT: 20 % lower in seed 1).
  - PFN (not IRC-safe by construction) responds 3× more, also seed-stable (+0.021 / +0.025).
  - The Herwig-vs-Pythia training shift of H24v is 8× (ParT) to > 100× (ParticleNet) larger than the seed spread. The H26 question is answered for the Pythia side: the generator shift is not a seed effect.
- **Charge (D16).** Not tested. The H40v weight-permuted band (|E1 × sign Q| ≤ 0.019 ParT, ≤ 0.112 ParticleNet) remains the only null.

## Training diagnosis (d-h42-qg-pn-kp-shuf and the other kinpid runs)

- **What happened:** the pod `anrunw-ai-d-h42-qg-pn-kp-shuf-zqc5w` was terminated with Reason OOMKilled, exit 137, after 8 min at limit = request = 40 GiB.
  - Its log showed GPU 0 % and 22.6 GiB host memory at 5 min: it was still in weaver's `--in-memory` loading of the 1.6 M training jets.
  - The other three kinpid jobs (`d-h42-qg-part-kp-s1`, `-part-kp-shuf`, `-pn-kp-s1`) had waited for a 40 GiB GPU node, then ran and were also OOMKilled (status at 04:45 UTC).
- **Why:** the kinpid inputs have 13 instead of 7 features per particle.
  - The original H2 kinpid trainings reached 36.0 GiB during loading (`d-h2-qg-pnet.log`) and were run at 44 Gi.
  - The kin shuffled run (`d-h39-qg-pn-shuf`) uses 24.9 GiB.
  - The hash-label formula is not the cause: the s1 runs without it failed the same way.
- **Fix:** relaunched at 48 Gi, 4 CPU, same script and settings (authors' in-memory loading and 20 epochs, so seed 1 differs from seed 0 only in its seed).
  - `d-h56-qg-pn-kp-shuf` passed loading in < 5 min, then GPU 99 %, host memory 25.5 → 32.7 GiB (68 % of request), about 18 min per epoch. It had 4 epochs validated at 04:48 UTC.
  - After the four 40 GiB jobs had failed, the other three were relaunched the same way:
    - `d-h56-qg-part-kp-s1`: 2 epochs, 32.0 GiB;
    - `d-h56-qg-part-kp-shuf`: 1 epoch, 36.5 GiB;
    - `d-h56-qg-pn-kp-s1`: loading, 32.2 GiB at 25 min.
  - A first attempt to delete the old pending 40 GiB jobs before they ran was blocked by the permission system. The jobs were left alone and failed by themselves.
- **Next:** when they finish (about 6–7 h), run `./part_job.sh d-h56-charge 'python -m interp.h42_qg_charge_null --step run --models part_kinpid_s1,pnet_kinpid_s1,part_kinpid_shuf,pnet_kinpid_shuf'` (H42 already loads these names), then `--step merge`.

## Job status

| job | state at report time |
|---|---|
| d-h39-qg-ef, d-h39-top-pn-s1, d-h39-top-pn-shuf, d-h39-qg-pn-s1, d-h39-qg-part-s1 | complete; analysed (QG ParT s1 axis at epoch 16 provisional; count with the final best checkpoint) |
| d-h39-qg-pn-shuf | running, 12 of 20 epochs; analysed at epoch 10 (provisional, constant output) |
| d-h56-qg-pn-kp-shuf, d-h56-qg-part-kp-s1, d-h56-qg-part-kp-shuf, d-h56-qg-pn-kp-s1 | running (relaunches of the OOM-killed d-h42 kinpid trainings) |

## Issues

- **One extra seed:** the seed spread is a single paired difference. The jet-bootstrap CI measures only how well that difference is estimated on these jets, not how widely seeds vary.
- **Degenerate shuffled controls:** the shuffled-label controls give no spread. A non-degenerate "no class information" control would be needed, e.g. a regression on a class-independent jet observable with matched loss, or early-stopped shuffled training.
- **ParT seed 1 axis result is provisional (epoch 16).** Rerun the axis step with `H56_STEPS=axis` for the final checkpoint and the finished QG ParticleNet shuffled-label net.
- **Top ParT:** no second seed or shuffled run.
- **Thin init nulls at the sign call:** ParticleNet and QG PFN have no non-degenerate init network at the sign call, so those cells have no band.

## Explainer entry (for site/explain.py, Paper D)

```
H56_EXPLAINER = [
 ('Question', 'Are three earlier results properties of training on quark/gluon or top labels, and how much do they move between two trainings that differ only in their random seed? (i) the direction part of the axis-convention response (H36/H39), (ii) the jet-charge response (H40/H42), (iii) the residual response to particle count at fixed IRC-safe energy flow (H24v, claim D7).', None),
 ('Physics background', 'Recomputing particle coordinates about the winner-take-all (WTA) or hardest-particle (HP) axis is a rigid translation by the axis offset $\\delta_j$. The direction part is $D=\\langle \\mathbf 1[\\text{call changes about the axis}]-\\tfrac1{20}\\sum_k \\mathbf 1[\\text{call changes under a shift of length }|\\delta_j|\\text{ in direction }k]\\rangle$, in percentage points. An exact collinear split replaces one soft particle by two with half its $p_T$ each at the same $(y,\\phi)$: $N\\to N+1$ while every IRC-safe observable (energy flow, $p_T^D$, girth, mass) is unchanged, so a response to it is a response to $N$ beyond energy flow. A likelihood ratio in $N$ alone, LR(N), gives $+0.10$ log-odds per particle.', None),
 ('Method', 'Per architecture (ParT, ParticleNet, PFN, EFN): seed 0, a second seed (torch/energyflow seed 1, same data and settings), a shuffled-label training (label = hash of jet $(p_T,\\eta,\\phi)$ mod 2, or permuted labels), and 6 untrained (epoch-0) networks. Axis: Top Landscape and quark-gluon Pythia test jets, 10,000 each, calls at the sign of the log-odds and at signal efficiency 0.3 and 0.5. Count: 4,000 Pythia jets (seed 7243), slope over $k=1..5$ of the logit change under $k$ exact splits minus that under a $\\Delta N=0$ control. 95 % jet bootstrap; paired sign-flip and McNemar (paired call changes) tests for seed 1 vs seed 0; Holm over 283 tests.', None),
 ('Result', 'Axis: seed 1 lies inside the untrained band like seed 0 in every cell; the sign agrees in 38 of 42 cells, but $D$ moves by up to 1-2 pp between seeds (QG PFN, 50 % efficiency, HP: $+1.16\\to+0.08$, Holm $p=0.012$). Count: ParT_kin $+0.0079$ / $+0.0063$, ParticleNet_kin $+0.0042$ / $+0.0042$ log-odds per particle (seed 0 / seed 1), against LR(N) $+0.10$; EFN split response $0$. Every shuffled-label network ended with a constant output (50 % accuracy), so its responses are 0.', 'results/physics/h56_seed_nulls.png'),
 ('Checks', 'Seed-0 rows reproduce H39 and H24v exactly on the same samples; axis edit equals a shift of the stored arrays to 0; unedited accuracies of seed-1 networks within 0.6 pp of seed 0. The charge test is not run: the four kinpid seed-1 / shuffled trainings were killed for memory at 40 GiB and are rerunning at 48 GiB.', None),
 ('Physical significance', 'The extra response of trained taggers to shifting the jet toward its hardest particle is no larger than the difference between two trainings or between untrained networks, so it is not a learned use of the axis direction. The small residual response to particle count at fixed energy flow (4-8 % of a pure counter) is reproducible across seeds, and the Herwig-vs-Pythia training shift of that response is 8 to over 100 times the seed spread.', None),
]
```

Conclusion for site/conclusions.py: "Retraining the taggers with a second random seed leaves the axis-direction effect inside the range of untrained networks, and its size changes by up to 1–2 percentage points between seeds. The small response to particle number at fixed energy flow (about 0.004–0.008 log-odds per particle, against 0.10 for a pure particle counter) repeats across seeds within 0.002. Networks trained on labels unrelated to the class learn a constant output, so they cannot serve as a spread-giving null. The jet-charge test still waits for its retrained networks."

Glossary additions (explain.GLOSSARY):
- **seed spread:** difference of a measured response between two trainings that differ only in the random seed.
- **shuffled-label training:** training on labels independent of the jet class (hash of jet kinematics or permuted labels), same number of steps.
- **epoch-0 network:** a network built with the training code but not trained.
- **direction part D:** call change under the axis shift minus the mean call change under shifts of the same length in random directions, in percentage points.
- **exact collinear split:** one particle replaced by two with half its pT each at the same (y, φ); N + 1, IRC-safe observables unchanged.
- **McNemar test:** exact binomial test on paired yes/no outcomes using only the jets where the two outcomes differ.

## Files

- `interp/h56_seed_nulls.py`
- `results/physics/h56_seed_nulls.json`, `h56_seed_nulls.png`, `h56_count_perjet.npz`; per-jet axis arrays on the volume: `h56_axis_perjet_{top,py}.npz`
- smoke: `h56_smoke_*`
- logs: `logs/d-h56-run.log`; `/j-jepa-vol/aaron-interp/logs/d-h56-qg-*-kp-*.log`
