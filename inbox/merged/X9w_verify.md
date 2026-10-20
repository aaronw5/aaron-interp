# X9w verify: re-run on a fresh sample, plus a translation outside the augmentation range

Node X9w (claims A10m / A4 / D4). Verifier pass, 2026-10-20. Script `interp/x9w_verify.py` (imports the X9w model loaders,
edit constructions and statistics from `interp/x9w_repr_2x2.py`); figure `interp/x9w_verify_plot.py`.
Job x9w-verify (1 CPU / 2 GiB, wall 1,089 s, pod peak memory 2,047 MiB, i.e. at the 2 GiB limit; 0.54 cores).
Results `results/jedi/x9w_verify.json`, log `logs/x9w-verify.log`, figure `results/jedi/x9w_verify.png`.

## Conclusion
On a fresh set of jets the X9w numbers reproduce: PFN, EFN and Deep Sets trained with random sideways shifts of up to 0.08 rad
no longer change W calls to Z when the jet is slid by 0.06 rad, while the same networks trained without shifts send about two
thirds of W jets to Z. Sliding by 0.12 rad, larger than any shift seen in training, sends 51–84 % of W jets to Z in these
augmented networks, so they did not learn to measure the jet's spread about its own centre; they learned to ignore offsets
inside the range they were trained on. The X9w statement that augmented networks read mass "from the spread about the jet's own
centroid" therefore holds only for the recentred networks (where it is true by construction) and, for augmented networks, only
for offsets up to 0.08 rad. Whether retrained JEDI-linear keeps a translation response because of its architecture or because
its augmented runs are under-trained is still open.

## Code review (interp/x9w_repr_2x2.py and the opt-in training flags)
- Edits: `translate`, `rotate_matched`, `scale_angles` (interp/x9_efn_pfn.py) move only particles with pT > 0; padding is never
  moved (edit hygiene `padding_changed` false for every edit). T leaves log m unchanged (mean 9e-9) and raises log M2 by 0.61;
  rotation leaves both unchanged; M2held lowers log m by 0.305; dilM2 raises it by 0.305.
- Recentring: the same pT-weighted-centroid formula in x9_train.recentre_np, train_small.recentre_centroid,
  train_hgq_jedi.recentre16 and x9w_repr_2x2.recentre; applied before normalisation for train, dev and test jets; the eval
  applies edits before recentring, so the recentred-cell zeros are a property of the input pipeline (exact up to float32).
- Augmentation: training jets only (dev / test GPUData / Xva / Xte built without it), real particles only, redrawn every epoch
  (every batch for Deep Sets), after recentring in rcaug. No split leakage found (dev = splits.npz['dev'], test = val archive).
- Model flags are asserted against each checkpoint's metadata (recentre / aug) in load_model.
- `git diff` of x9_train.py / train_small.py / train_hgq_jedi.py: the aug = 0, no-recentre code path is unchanged, so the
  reused stored-cell checkpoints (X9, small-model, X12) come from the same training code as the new runs; the aug path of
  x9_train (PFN/EFN) uses hand-written early stopping (patience 5, best weights restored), same rule as the Keras callback.
- No bug found that changes a reported number. The 216-relabelling stratified test quoted in the X9w report is not in the
  script or JSON (the script stores a 20,000-draw pooled test); it is recomputed below.
- Figure x9w_repr_2x2.png: matches the JSON. Suptitle "bars = 95 % CI" mislabels: bars are cell means, whiskers are the CIs.
  The official JEDI-linear reference row is not plotted.

## Procedure
Sample `jedi_core.sample(2000, 2610)`, N = 10,000 (2,000 true W); 384 of the 10,000 rows are also in the X9w sample (909).
Same 33 networks + official JEDI-linear, same edits T, Tr, rot, M2held, dilM2 (X9w constructions and rng seeds). Added edits:
T12 (+0.12 rad in Δη, outside the training augmentation range |c| ≤ 0.08), Tr12 (0.12 rad, random direction per jet, rng 9094),
Tneg (−0.06 rad in Δη), Tphi (+0.06 rad in Δφ). All are rigid translations: invariant mass unchanged.
Metrics: W→Z flow (edited − unedited, true W), W→g/q flow (edited − unedited; X9w quoted absolute fractions), calls changed
(all jets), mean total-variation distance of the softmax TV = ½ Σ_c |p_c(edit) − p_c(0)| minus the same under rotation.
Statistics: 2,000-draw jet bootstrap CIs; exact McNemar (edit vs rot, T vs Tr) with Holm over the 34 networks; seed level:
stored vs aug, exact stratified permutation over PFN/EFN/Deep Sets (6 relabellings per architecture, 216 total, minimum
two-sided p 0.0093) and JEDI exact 3 vs 3 (20 relabellings, minimum p 0.10).

## Re-run of the X9w numbers (fresh sample)
| network | flow(T) X9w | flow(T) verify [95 % CI] | flow(dilM2) verify | ratio T/dilM2 |
|---|---|---|---|---|
| official JEDI-linear | 0.721 | 0.744 [0.724, 0.765] | 0.579 | 1.29 |
| PFN stored s0 / s1 | 0.522 / 0.697 | 0.527 [0.502, 0.552] / 0.708 [0.686, 0.728] | 0.502 / 0.699 | 1.05 / 1.01 |
| EFN stored s0 / s1 | 0.356 / 0.667 | 0.346 [0.324, 0.369] / 0.710 [0.688, 0.732] | 0.223 / 0.242 | 1.56 / 2.93 |
| Deep Sets stored s0 / s1 | 0.638 / 0.683 | 0.661 [0.639, 0.683] / 0.708 [0.686, 0.730] | 0.445 / 0.488 | 1.49 / 1.45 |
| JEDI HGQ-50 stored s0 / s1 / s2 | 0.577 / 0.538 / 0.718 | 0.596 / 0.564 / 0.749 | 0.543 / 0.537 / 0.520 | 1.10 / 1.05 / 1.44 |
| PFN aug s0 / s1 | 0.012 / 0.007 | 0.009 [0.002, 0.017] / 0.007 [0.002, 0.013] | 0.587 / 0.580 | 0.02 / 0.01 |
| EFN aug s0 / s1 | 0.004 / 0.003 | 0.004 [−0.003, 0.009] / 0.002 [−0.002, 0.005] | 0.356 / 0.320 | 0.01 / 0.00 |
| Deep Sets aug s0 / s1 | −0.001 / −0.001 | −0.001 [−0.006, 0.004] / −0.005 [−0.011, 0.001] | 0.476 / 0.451 | 0.00 / −0.01 |
| JEDI HGQ-50 aug s0 / s1 / s2 | 0.325 / −0.019 / 0.467 | 0.375 [0.351, 0.399] / −0.010 [−0.022, 0.002] / 0.496 [0.473, 0.519] | 0.482 / 0.713 / 0.735 | 0.78 / −0.01 / 0.68 |
| recentred, recentred + aug (13 nets) | 0 | 0 (calls changed 0) | 0.19–0.74 | 0 |

Rotation null flow −0.003 to +0.018 in every network. Seed level, stored − aug flow(T), PFN/EFN/Deep Sets: 0.608, exact
stratified p 0.0093 (X9w 0.590, p 0.0093). JEDI stored − aug: 0.349, p 0.10 (X9w 0.353, p 0.10). dilM2 stored − aug: −0.028,
p 0.49 (PFN/EFN/Deep Sets); −0.110, p 0.40 (JEDI). M2held flow: stored 0.008–0.149 (JEDI s1 0.149; X9w 0.128), aug / recentred
−0.142 to −0.010. JEDI aug +Δη excess over a random direction (T − Tr): s0 0.053 [0.030, 0.076], s2 0.276 [0.250, 0.299]
(X9w 0.05 / 0.24); T − Tneg: s2 0.529. A +Δη vs random-direction difference also appears in stored networks (Deep Sets 0.14 /
0.16, official 0.16, JEDI stored s2 0.14, EFN stored s0 −0.18), so a direction-specific term is not specific to augmented JEDI.

## Added control 1: translation outside the augmentation range (T12 = 0.12 rad)
H0: an augmented network is translation-invariant (reads mass about the jet's own centroid), so T12 moves no more W jets to Z
than rotation. H1 (invariance only within the trained offsets): T12 moves W jets to Z.
| network | flow(T12) [95 % CI] | flow(Tr12) | calls changed T12 / rotation | McNemar T12 vs rot, p_Holm |
|---|---|---|---|---|
| PFN aug s0 / s1 | 0.514 [0.491, 0.535] / 0.837 [0.821, 0.853] | 0.818 / 0.784 | 0.233 / 0.056; 0.352 / 0.052 | < 1e-290 / 0 |
| EFN aug s0 / s1 | 0.830 [0.813, 0.846] / 0.626 [0.604, 0.648] | 0.830 / 0.654 | 0.459 / 0.064; 0.497 / 0.063 | 0 / 0 |
| Deep Sets aug s0 / s1 | 0.808 [0.790, 0.827] / 0.668 [0.646, 0.692] | 0.762 / 0.778 | 0.432 / 0.044; 0.578 / 0.041 | 0 / 0 |
| JEDI aug s0 / s1 / s2 | 0.219 / 0.641 / 0.760 | 0.346 / 0.675 / 0.708 | 0.609 / 0.399 / 0.610 vs 0.094–0.125 | 4e-38 / 0 / 0 |
| stored networks (11) | −0.073 to −0.009 | −0.071 to +0.026 | 0.65–0.85 vs 0.036–0.088 | |
| recentred networks (13) | 0 | 0 | 0 | |

H0 rejected in all 9 augmented networks. Stored networks change 65–85 % of calls under T12 but their W jets go past Z (W→Z flow
slightly negative; the receiving class was not recorded). Output movement under T (0.06 rad) in augmented PFN/EFN/Deep Sets:
TV − TV(rot) 0.006–0.028 (all CIs above 0) with calls changed 5.1–9.3 % vs rotation 4.1–6.4 %: the outputs are not exactly
invariant inside the trained range either, but the W→Z calls are.

## Added control 2: M2held as a flow
W→g/q flow under M2held (edited − unedited): stored −0.070 to −0.009 (all 11 networks incl. official); aug PFN 0.580 / 0.812,
EFN 0.186 / 0.216, Deep Sets 0.265 / 0.320, JEDI 0.212 / 0.416 / 0.356; recentred / recentred + aug 0.18–0.85. X9w's
"30–95 % vs 5–10 %" were absolute W→g/q fractions (unedited 0.084–0.128, JEDI aug s0 0.282); as flows the contrast is
0.19–0.85 vs −0.07 to −0.01 and holds in direction.

## Competing hypotheses for the augmented cells
| hypothesis | prediction | outcome |
|---|---|---|
| (i) aug networks read mass about the jet's own centroid | flow(T) ≈ flow(T12) ≈ rotation | flow(T) ≤ 0.009, flow(T12) 0.51–0.84: excluded |
| (ii) aug networks ignore axis offsets up to the trained 0.08 rad, still read axis-relative spread beyond | flow(T) ≈ 0, flow(T12) large | observed in 6/6 PFN/EFN/Deep Sets |
| (iii) no axis-relative reading at all after aug | M2held flow ≈ 0 at 0.06 and no T12 response | excluded by T12 |

## Assessment of the X9w reading
- Reproduced: stored-cell flows 0.35–0.71 (PFN/EFN/Deep Sets); aug flows at rotation level; recentred exactly 0; dilM2 response in
  every cell; JEDI aug 1 of 3 seeds at null, 2 of 3 at 0.38 / 0.50; seed-level p 0.0093.
- Not supported: "in those cells mass is read from the spread about the jet's own centroid". True for recentred networks by
  construction; for augmented networks only for offsets ≤ 0.08 rad (T12 control).
- Number corrections: "5–10 % for stored-axis networks" is an absolute fraction; the stored-network M2held W→g/q flow is −0.07 to
  −0.01. "lose ≤ 0.9 points" holds for architecture means; per network the largest aug loss is Deep Sets s1 1.13 points
  (80.81 → 79.68 %) and EFN s1 0.93.
- Thesis note overreach: "the axis-moment mass reading in JEDI-linear is a property of stored-axis training data shared by generic
  set networks" is inferred from PFN/EFN/Deep Sets; in JEDI-linear itself augmentation did not remove the response in 2 of 3
  seeds (p 0.10, minimum attainable).
- Design limits: the recentred-cell zeros, and hence the "representation" and "interaction" effects, are fixed by construction;
  only stored vs aug is an empirical test. JEDI recentred cells have one seed.

## Corrected reading
PFN, EFN and Deep Sets trained on stored-axis inputs send 0.35–0.71 of W jets to Z under a 0.06 rad rigid translation
(reproduced on sample(2000, 2610)). The same architectures trained with translation augmentation (|c| ≤ 0.08 rad) show 0.06 rad
flows at the rotation-null level (≤ 0.012) at a mean accuracy cost ≤ 0.9 points (≤ 1.13 per network), and networks on recentred
inputs give exactly 0 by construction. The augmented networks are not translation-invariant beyond the trained range: a 0.12 rad
translation sends 0.51–0.84 of their W jets to Z (rotation null ≤ 0.01). All networks respond to a matched real widening
(dilM2). Under the M2-held edit (invariant mass −26 %), recentred and augmented networks move 0.19–0.85 of W jets to g/q,
stored-axis networks −0.07 to −0.01. In retrained JEDI-linear HGQ-50, augmentation reduced the 0.06 rad flow to the null level in
1 of 3 seeds; the other two keep 0.38–0.50, with a +Δη excess over a random direction of 0.05–0.28; these runs lost 6–14 accuracy
points, so architecture vs under-training is not separated.

## Explainer entry (for site/explain.py, Paper A, claim A10m, verify)
**Question.** Do set networks trained with random sideways shifts of the jet learn to measure the jet's spread about its own
centre, or only to ignore the shifts they were shown?

**Background.** Translating every particle by $\vec c$ leaves the invariant mass unchanged but changes
$M_2=\sum_i p_{T,i} r_i^2$ about the stored axis by $2\vec c\cdot\sum_i p_{T,i}\vec r_i+|\vec c|^2 M_0$. The augmented networks of
X9w saw $|\vec c|\sim U(0,0.08)$ rad in training.

**Method.** Same 33 networks, 10,000 fresh test jets. Translate by $0.06$ rad (inside the training range) and $0.12$ rad
(outside); count W jets newly called Z; null: rotation about the axis; McNemar with Holm over 34 networks.

**Result.** (figure x9w_verify.png) At $0.06$ rad augmented PFN/EFN/Deep Sets: flow $\le 0.009$; at $0.12$ rad: $0.51$–$0.84$.
Stored-axis networks at $0.06$ rad: $0.35$–$0.71$.

**Checks.** Random-direction $0.12$ rad translation gives the same ($0.65$–$0.83$); rotation null $\le 0.018$; the X9w numbers
reproduce within the bootstrap intervals on the new jets.

**Why it matters.** Augmentation removes the axis-moment response only inside the trained offset range; it does not make these
networks read mass about the jet centre, so "translation invariant" applies to recentred networks only.
