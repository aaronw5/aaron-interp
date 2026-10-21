# P34: lepton circuit follow-up (fresh P32v rerun, literal lepton circuit); claim C15 (agent report, 2026-10-20)

## Conclusion
In ParT, removing the electron/muon flag from one particle still moves almost every H→ℓνqq′ jet to another class (98.7 % of calls on a fresh sample). The same-size flag change on a hadron of similar pT moves 1.3 %, and swapping electron for muon moves 0.07 %, so the network reads whether a lepton flag is present, not which lepton it is.

When ParT is cut down to a literal circuit (only the lepton-reading attention heads kept; every other head and every MLP, the per-token two-layer feed-forward networks, replaced by its average output), its H→ℓνqq′ score still separates H→ℓνqq′ jets from the other classes (AUC 0.92, above most random head sets). However, its call is the same class for every jet, so H→ℓνqq′ recall is 0. Keeping only the two MLPs of the final class-token blocks restores most of the call (recall 0.88). Those same circuits then also call 94 % of leptonic-top jets H→ℓνqq′.

So the heads carry the lepton signal, and the class-block MLPs turn it into the decision. The "head-only circuit" reading of P32 is withdrawn under the user's circuit definition. Still uncertain:
- the AUC advantage over random sets is significant only for 1–3 heads;
- which MLP neurons do the conversion is not yet located.

## Runs
| job | command | node / GPU | wall | output |
|---|---|---|---|---|
| `anrunw-ai-c-p34-p32v3` | `python -m interp.p32v_verify --smoke && python -m interp.p32v_verify` (script unchanged; its pre-registered seed 9157) | gpn-fiona-mizzou-7 | 335 s | `results/part/p32_verify.json`, `logs/c-p34-p32v3.log` |
| `anrunw-ai-c-p34-run4` | `python -m interp.p34_lepton_circuit --smoke && python -m interp.p34_lepton_circuit && … --plot` | GPU | 1,523 s | `results/part/p34_lepton_circuit.json`, `logs/c-p34-run4.log` |
| `anrunw-ai-c-p34-plot2` | `python -m interp.p34_lepton_circuit --plot` (CPU, job.sh) | – | 4 s | `results/part/p34_lepton_circuit.png` |

Cluster notes:
- The first submissions failed:
  - `c-p34-run`: node k8s-haosu-09 rejected the pod with "GPU is lost". This is the same fault as k8s-haosu-22 in P32v; k8s-haosu-09 is not in part_job.sh's exclusion list.
  - `c-p34-p32v` / `c-p34-p32v2`: the image pull timed out on two fresh nodes.
- `part_job.sh` is outside this loop's write scope, so I used a copy, `interp/p34_job.sh`, which differs only in:
  - k8s-haosu-09 is added to the NotIn list;
  - a soft preference (preferredDuringScheduling) for nodes that already hold the weaver image.
- **Suggestion for the ledger agent:** add k8s-haosu-09.sdsc.optiputer.net to part_job.sh. The P37 smoke jobs of this loop were also rejected there.
- `c-p34-run3` was stopped after its smoke step, to add the per-block MLP variants; its results are not used.

## Unedited accuracy (reference check)
- P1 sample (rows 3,000–4,000 of each class's _100 file), 10-class accuracy:
  - ParT_full 0.8593 and ParT_kin 0.7447, as in P1/P32;
  - ParT_kinpid 0.7902, PN_full 0.8456, PN_kinpid 0.7720.
- P34 evaluation sample (15,000 jets, class-balanced), 10-class accuracy:
  - ParT_full 0.862 [0.856, 0.868];
  - ParT_kinpid 0.796, ParT_kin 0.742, PN_full 0.845, PN_kinpid 0.769.
- ParT_full on this sample: H→ℓνqq′ recall 0.983; t→bℓν jets called H→ℓνqq′ 1.1 %.

## Procedure
**Sample (P34).**
- JetClass test_20M `<class>_116.root`, all 10 classes, numpy `default_rng(3416)`.
- 1,800 random rows per class, never first-k, split at random into:
  - 1,500 per class for evaluation (N = 15,000);
  - 300 per class for the reference set used to compute mean outputs (N = 3,000; disjoint from the evaluation rows).
- A class-balanced subsample of 300 per class (3,000 jets) of the evaluation sample is used for the random-head-set controls.
- Every jet passes through `part_core.build`. Edited jets are compared with `rebuild(base)`, so both have the same number of build passes.
- The P32v rerun used its own pre-registered sample: the same _116 files, `default_rng(9157)`, 3,000 H→ℓνqq′ + 400 per other class. Its rows are an independent draw, so the two samples can overlap.

**Lepton candidate and control tokens.**
- Lepton candidate: the highest-pT particle with isElectron or isMuon.
- 1,480 of 1,500 H→ℓνqq′ jets have a lepton candidate and a charged hadron: 48 % electrons, 52 % muons. The lepton is the leading particle in 71 %.
- Matched hadron: a random charged hadron with pT within a factor 1.5 of the lepton's, else the charged hadron nearest in pT. Median |ln(pT ratio)| = 1.11, so in many jets no close-pT hadron exists.

**Edits** (on the 1,480 jets; nothing else in the jet changes):
- **e1:** lepton flag removed: isElectron = isMuon = 0, isChargedHadron = 1. Four-momentum, charge and d0/dz are unchanged (checked: these columns are bit-identical).
- **e2n (control):** the matched charged hadron's flag is changed to isNeutralHadron. This is a one-hot change of the same size, on a token of similar pT, with no lepton flag involved. It replaces P32's e2, which created a two-lepton jet.
- **e0 (no-op check):** isElectron = isMuon = 0 written on the neutral hadron nearest in pT to the lepton, where both are already 0. The edited jet must be bit-identical to the base (checked; 1,416 jets).
- **e1x:** electron ↔ muon swap on the lepton. This tests whether the network reads the identity of the flag or only its presence.

**Metrics.**
- Share of calls (argmax) that change.
- Δ log p(H→ℓνqq′).
- Δ log-odds(H→ℓνqq′) = Δ[logit_ℓνqq − logsumexp(other 9 logits)].
- McNemar exact test on paired outcomes.
- 95 % bootstrap intervals over jets, 1,000 resamples.

**Knock-outs (all three ParT models).** For the dominant class-attention head (ParT_full b2h1; ParT_kinpid and ParT_kin b2h4):
- Key mask: the lepton's attention logit is set to −10⁹, so the head renormalises its weight over the other particles.
- Value zeroing: the lepton's value vector is set to 0 in that head. The attention weights are unchanged, so the head output loses the lepton's share.
- Control: the same on the matched hadron.
- Joint key masks: the 16 class heads, the 64 particle-block heads, all 80 heads.
- Redundancy test: "dominant + k random heads" vs "the same k random heads" (k = 1, 3, 7, 15, 31; 20 random sets each, drawn from the other 79 heads). Paired McNemar per set, Holm over the 20 sets.

**Literal circuit (ParT_full, ParT_kinpid).**
- *Kept:*
  - the per-particle input embedding, which is the only place the inputs enter;
  - the kept heads, including the pair-embedding bias into those heads;
  - all LayerNorms, the class-token parameter and the final linear layer.
- *Mean-ablated:* every other head and all 10 MLPs (8 particle blocks + 2 class blocks). Mean ablation means a unit's output is replaced, for every jet and token, by its average over the reference set computed with the unablated network. For a head this is its 128-dim contribution after the output projection, averaged over real tokens (class heads: over the class token). For an MLP it is its fc2 output.
- *Head choice:* the top-k heads (k = 1, 3, 5, 10) of P32's single-head lepton-key knock-out scan, restricted to the Holm-significant heads and ordered by effect. Three sets:
  - mixed: ParT_full b2h1, p7h8, p8h6, p8h2, b2h6, p3h2, p6h7, p5h4, p7h7, b2h8;
  - class-block heads only;
  - particle-block heads only.
- *Variants of each circuit:*
  - only the 2 class-block MLPs kept;
  - only the 8 particle-block MLPs kept;
  - all MLPs kept (P32's head-only ablation);
  - e1 / e2n applied while the circuit is in place.
- *Controls:*
  - k = 1: all 80 single-head circuits (exhaustive);
  - k = 3, 5, 10: 100 random sets with the same number of class and particle heads;
  - class-only / particle-only circuits: 30 random sets of that block type.
  - Empirical p = (1 + #random ≥ selected)/(1 + N_random). Random sets are evaluated on the 3,000-jet subsample; the selected circuits on both the subsample and the full sample.
- *Metrics:*
  - H→ℓνqq′ recall;
  - class-balanced accuracy (the mean of the 10 per-class recalls; 0.1 = one class for every jet);
  - 10-class accuracy and the 10×10 confusion matrix (in the JSON);
  - number of distinct calls;
  - t→bℓν jets called H→ℓνqq′;
  - AUC of the H→ℓνqq′ log-odds for H→ℓνqq′ vs the other 9 classes, and vs t→bℓν only. AUC is the probability that a random H→ℓνqq′ jet scores higher than a random other jet; 0.5 = no separation.

## Numbers

### Part 1: flag edits (1,480 H→ℓνqq′ jets with a lepton; P34 sample)
| model | e1 calls changed | e2n (control) | e1x (e ↔ μ) | e0 (no-op) | e1 Δlog p | McNemar e1 vs e2n (changed only under e1 / only under e2n) |
|---|---|---|---|---|---|---|
| ParT_full | 98.7 % [98.1, 99.3] | 1.28 % [0.74, 1.89] | 0.07 % [0, 0.2] | 0 (max abs. logit difference 0) | −7.23 [−7.34, −7.12] | 1,443 / 1, p < 10⁻³⁰⁰ |
| ParT_kinpid | 98.6 % [98.0, 99.1] | 2.36 % [1.62, 3.18] | 0.20 % [0, 0.47] | 0 | −7.33 | 1,424 / 0 |
| ParT_kin (no flags) | 0 | 0 | 0 | 0 | 0 | – |
| PN_full | 99.1 % [98.5, 99.5] | 0.74 % [0.34, 1.22] | 0.14 % | 0 | −8.01 | 1,455 / 0 |
| PN_kinpid | 98.4 % [97.6, 99.0] | 1.49 % [0.88, 2.10] | 0.27 % | 0 | −8.28 | 1,435 / 1 |

- Under e1, ParT_full's H→ℓνqq′ calls go to H→4q (633), Z (313), W (302) and QCD (155).
- e2n calls in ParT_full go mostly to t→bℓν (13 of 19).
- Electron and muon jets respond alike to e1: 98.5 % vs 99.0 % of calls changed (ParT_full).
- After e1, 13.5 % of jets still contain another lepton-flagged particle.
- The P32v rerun (seed 9157, 2,954 jets) gives the same pattern:
  - ParT_full: e1 98.7 %, e2 14.3 %, e2n 1.0 %, e1x 0.1 %;
  - ParT_kinpid: e1 98.3 %, e2n 2.8 %;
  - PN_full: e1 98.5 %, e2n 1.1 %.

### Part 1: knock-outs (1,480 jets)
| ParT_full, lepton key masked in | Δ log-odds | calls changed | matched-hadron control: calls changed | McNemar (lepton vs control) |
|---|---|---|---|---|
| b2h1 only | −5.42 [−5.53, −5.32] | 2.9 % [2.0, 3.9] | 0.0 % | 43 / 0, p 2·10⁻¹³ |
| b2h1 only, value zeroed | −4.26 [−4.35, −4.16] | 2.0 % [1.3, 2.8] | 0.0 % | 30 / 0 |
| 16 class heads | −6.83 | 12.6 % [10.9, 14.3] | 0.1 % | 184 / 0 |
| 64 particle-block heads | −5.50 | 5.8 % [4.7, 7.0] | 1.3 % | 80 / 13 |
| all 80 heads | −13.78 [−14.00, −13.58] | 95.1 % [94.0, 96.1] | 1.3 % | 1,388 / 0 |

**b2h1 + k random heads vs the same k random heads** (ParT_full; 20 sets per k; mean calls changed, increment with 95 % interval over sets):

| k | k random | b2h1 + k random | increment | sets with a Holm-significant increment |
|---|---|---|---|---|
| 1 | 0.0 % | 3.1 % | +3.0 [2.9, 3.2] | 20 / 20 |
| 3 | 0.1 % | 3.3 % | +3.2 [3.0, 3.4] | 20 / 20 |
| 7 | 0.3 % | 4.3 % | +4.1 [3.4, 5.3] | 20 / 20 |
| 15 | 0.6 % | 6.5 % | +5.9 [4.6, 7.5] | 20 / 20 |
| 31 | 1.3 % | 23.6 % | +22.4 [16.1, 28.8] (range 4.9–61.4) | 20 / 20 |

- ParT_kinpid (b2h4): class heads 3.4 %, particle-block heads 25.9 %, all 80 heads 92.4 %. b2h4 + 31 random heads adds 9.8 % [6.6, 13.5].
- ParT_kin (b2h4, reading the lepton by kinematics): b2h4 alone 9.1 %, class heads 12.2 %, particle-block heads 39.4 %, all 80 heads 71.4 %.

### Part 2: literal circuit (ParT_full; evaluation sample, 15,000 jets)
| circuit (everything else at mean) | distinct calls | H→ℓνqq′ recall | balanced acc. | AUC H→ℓνqq′ vs rest | AUC vs t→bℓν | e1 Δ log-odds under circuit (e2n) |
|---|---|---|---|---|---|---|
| nothing kept | 1 (all Z) | 0 | 0.100 | 0.46 | 0.44 | – |
| top-1: b2h1 | 1 (all Z) | 0 | 0.100 | 0.921 | 0.409 | −1.61 [−1.63, −1.59] (0.00) |
| top-3: b2h1, p7h8, p8h6 | 1 | 0 | 0.100 | 0.923 | 0.417 | −1.64 (−0.01) |
| top-5 mixed | 1 | 0 | 0.100 | 0.923 | 0.403 | −2.08 (−0.02) |
| top-10 mixed | 1 | 0 | 0.100 | 0.926 | 0.415 | −2.36 (−0.01) |
| class heads b2h1, b2h6, b2h8 | 1 | 0 | 0.100 | 0.920 | 0.387 | −2.28 (0.00) |
| top-3 particle heads p7h8, p8h6, p8h2 | 1 | 0 | 0.100 | 0.479 | 0.474 | 0.00 (0.00) |
| all 80 heads, MLPs at mean | 10 | 0.876 | 0.608 | 0.988 | 0.933 | – |
| full network | 10 | 0.983 | 0.862 | 0.999 | 0.999 | – |

**Selected vs random head sets** (AUC H→ℓνqq′ vs rest on the 3,000-jet subsample; random sets matched in size and class/particle mix):

| k | selected | random mean | random 95th pct | random max | p (raw) | p (Holm over 4 k) |
|---|---|---|---|---|---|---|
| 1 (all 80 heads) | 0.912 (b2h1, highest of 80) | 0.514 | 0.771 | 0.912 | 0.025 (smallest possible, 2/81) | 0.075 |
| 3 | 0.915 | 0.700 | 0.909 | 0.915 | 0.010 | 0.040 |
| 5 | 0.915 | 0.780 | 0.917 | 0.929 | 0.079 | 0.158 |
| 10 | 0.921 | 0.845 | 0.926 | 0.950 | 0.119 | 0.158 |

- Random sets that contain b2h1 reach the same AUC: 0.913 at k = 3 (4 sets), 0.918 at k = 10 (15 sets).
- Other single class heads come close: b2h6 0.907, b2h8 0.898, b2h3 0.851.
- Balanced accuracy and recall are 0.100 and 0 for almost all selected and random sets, so they cannot separate selected from random heads (p = 0.99–1.0).

**Which MLPs turn the score into a call** (ParT_full; recall / t→bℓν called H→ℓνqq′):

| heads kept | all MLPs at mean | only the 8 particle-block MLPs kept | only the 2 class-block MLPs kept | all MLPs kept |
|---|---|---|---|---|
| b2h1 | 0 / 0 | 0 / 0 | 0.856 / 0.914 | 0.978 / 0.991 |
| top-3 mixed | 0 / 0 | 0.009 / 0.002 | 0.884 / 0.940 | 0.977 / 0.991 |
| class heads b2h1, b2h6, b2h8 | 0 / 0 | 0.725 / 0.660 | 0.948 / 0.984 | 0.981 / 0.991 |
| top-10 mixed | 0 / 0 | 0.815 / 0.568 | 0.965 / 0.989 | 0.982 / 0.992 |
| all 80 heads | 0.876 / 0.130 | 0.973 / 0.008 | 0.955 / 0.133 | 0.983 / 0.011 (full) |

- With all 80 heads kept, the particle-block MLPs separate t→bℓν from H→ℓνqq′: t→bℓν called H→ℓνqq′ is 0.8 % with them and 13 % without them.
- **ParT_kinpid:**
  - b2h4 literal circuit: AUC 0.914 (highest of 80 single heads, p 0.025), recall 0. With the class-block MLPs kept: recall 0.643, t→bℓν called H→ℓνqq′ 0.791.
  - top-3 vs 100 random sets: p 0.040; top-5 p 0.050; top-10 p 0.257.
  - all 80 heads with MLPs at mean: recall 0.903, balanced 0.614.

## Hypotheses and tests
| test | H0 | H1 | result |
|---|---|---|---|
| e1 vs e2n (ParT_full) | removing the lepton flag changes calls as often as a same-size flag change on a matched hadron | e1 changes more | 1,443 vs 1 jets, McNemar p < 10⁻³⁰⁰; H0 rejected |
| e1 vs e1x | flag identity (e vs μ) matters as much as flag presence | presence matters more | 1,460 vs 0; H0 rejected; e1x changes 0.07 % |
| b2h1 adds to k random heads | adding b2h1's lepton knock-out to k random knock-outs changes no extra calls | it does | 20/20 sets Holm-significant for every k; increment 3–22 % |
| literal circuit call | top-k literal circuit has H→ℓνqq′ recall ≤ random sets | recall > random | recall 0 for every selected set (p = 1); H1 not supported |
| literal circuit score | AUC(selected) ≤ AUC(random) | AUC(selected) > AUC(random) | k = 3 p 0.010 (Holm 0.040); k = 1 p 0.025 (Holm 0.075); k = 5, 10 not significant |
| flag dependence inside the circuit | e1 leaves the circuit's H→ℓνqq′ log-odds unchanged | e1 lowers it | b2h1 circuit −1.61 [−1.63, −1.59] vs e2n 0.00 |

**Competing hypotheses.**
- *(a) A literal head circuit carries the H→ℓνqq′ call (C15 as stated):* predicts recall above random sets with MLPs at mean. Excluded: recall is 0 for every selected circuit.
- *(b) Heads carry the lepton signal, MLPs make the call (the plan's alternative):* predicts that the score separates while the call does not, and that keeping the MLPs restores the call. Consistent: AUC 0.92, recall 0, recall 0.88 with the class-block MLPs.
- *(c) The circuit is a lepton detector, not a 10-class tagger:* predicts t→bℓν absorbed into H→ℓνqq′ once a call is possible. Consistent: AUC vs t→bℓν is 0.41 under the literal circuit, and 94 % of t→bℓν jets are called H→ℓνqq′ with the class-block MLPs.
- *(d) Two redundant routes (class heads, particle-block heads):* predicts that either route alone changes few calls and both together change most. Consistent on the fresh sample: 12.6 % + 5.8 % alone vs 95.1 % together.
  - In literal circuits the particle-block route cannot be isolated: a particle-head-only circuit has no path to the class token, because the class heads are the only path from particles to the class token. Its AUC is 0.48 by construction.

## Reading
- **Flag presence, not identity.** On a fresh sample, ParT_full, ParT_kinpid, PN_full and PN_kinpid change 98–99 % of H→ℓνqq′ calls when the leading lepton candidate loses its electron/muon flag. A matched-pT charged → neutral flag change changes 0.7–2.4 %, and an electron ↔ muon swap 0.07–0.27 %. The response therefore depends on the presence of a lepton flag on that particle, not on its flavour.
- **b2h1 is not interchangeable.** b2h1 (ParT_full's final class-attention head 1, which puts 0.99 of the class token's attention on the lepton) contributes a small, consistent share of calls: +3.0 % on top of one random head. That share grows to +22 % when 31 other heads are also blocked. The class-head route and the particle-block route each change few calls alone (12.6 % and 5.8 %) and 95 % together.
- **The heads alone do not make the call.** In the literal circuit, b2h1 alone (or with up to 9 more lepton heads) gives an H→ℓνqq′ score that separates H→ℓνqq′ from the other 9 classes (AUC 0.92) and moves with the flag (e1 lowers it by 1.6 log-odds; e2n by 0.00). But the call is the same class for every jet.
- **The class-block MLPs make the call.** The two class-token MLPs alone convert this score into H→ℓνqq′ calls (recall 0.86–0.97). The resulting circuits call 91–99 % of t→bℓν jets H→ℓνqq′.
- **Separating H→ℓνqq′ from t→bℓν** requires the particle-block MLPs together with many heads.

## Figure
`results/part/p34_lepton_circuit.png` (checked readable).
- **Left panel:** AUC of the H→ℓνqq′ score for the selected top-k heads vs 100 random sets (k = 1: all 80 single heads), ParT_full, all MLPs at mean.
- **Right panel:** top-3 lepton heads kept. H→ℓνqq′ recall and the share of t→bℓν jets called H→ℓνqq′ when no MLP, only the particle-block MLPs, only the class-block MLPs, or all MLPs are kept, next to the full network.

## Explainer entry (for site/explain.py)
```python
P34_EXPLAINER = [
 ('Question', 'P32 found that ParT recognises $H\\to\\ell\\nu q\\bar q\'$ jets through the electron/muon flag of one particle, read by a class-attention head (b2h1 in ParT_full puts 99 % of its attention on that particle). Does this hold on a fresh sample with a clean control, and is there a circuit made literally of a few attention heads (everything else replaced by its average) that still makes the $H\\to\\ell\\nu qq\'$ call?', None),
 ('Background',
  'Each JetClass particle carries five particle-ID flags (charged hadron, neutral hadron, photon, electron, muon). ParT has 8 particle blocks and 2 class blocks; each block has 8 attention heads and one MLP (a two-layer feed-forward network applied to each token). In the class blocks a class token attends to the particles; its final state gives the 10 class scores. '
  'A literal circuit (project definition) keeps only the named heads; every other head and every MLP is mean-ablated: its output is replaced, for every jet, by its average over a reference set of 3,000 jets (300 per class), so it carries no information about the jet. Inputs still enter through the per-particle embedding.', None),
 ('Method',
  'Fresh JetClass test jets (files _116, seed 3416): 1,500 random jets per class. Edits on the 1,480 $H\\to\\ell\\nu qq\'$ jets with a lepton candidate (highest-$p_T$ particle flagged electron or muon): (e1) its flag is set to charged hadron, momentum and charge unchanged; (e2n, control) a charged hadron of similar $p_T$ is set to neutral hadron, a flag change of the same size without a lepton; (e1x) electron and muon swapped; (e0) a no-op write, which must give exactly zero. '
  'Knock-outs: the lepton is hidden from b2h1 (attention logit $\\to-10^9$) alone, together with $k$ random heads, or in all heads; control: the same on the matched hadron. '
  'Literal circuits: the top $k=1,3,5,10$ heads of P32\'s knock-out scan, compared with 100 random head sets of the same size and block mix. Read-outs: $H\\to\\ell\\nu qq\'$ recall, class-balanced accuracy (mean of the 10 per-class recalls), and the AUC of the score $\\log\\frac{p_{\\ell\\nu qq}}{1-p_{\\ell\\nu qq}}$ for $H\\to\\ell\\nu qq\'$ against the other classes (probability that a random $H\\to\\ell\\nu qq\'$ jet scores higher than a random other jet; 0.5 = no separation). Statistics: bootstrap over jets, McNemar for paired calls, empirical p against random sets, Holm over $k$.', None),
 ('Result',
  'Removing the lepton flag changes 98.7 % of ParT_full\'s $H\\to\\ell\\nu qq\'$ calls; the matched control changes 1.3 % and the electron/muon swap 0.07 % (McNemar 1,443 vs 1 jets). ParticleNet behaves the same (99.1 % vs 0.7 %). Hiding the lepton from b2h1 adds 3 % changed calls on top of any random head, and 22 % on top of 31 random heads; class heads alone change 12.6 %, particle-block heads alone 5.8 %, all heads 95 %. '
  'The literal circuit with b2h1 (or the top 3–10 heads) gives a score that separates $H\\to\\ell\\nu qq\'$ jets (AUC 0.92; random 3-head sets 0.70 on average, $p=0.01$) and drops by 1.6 when the flag is removed, but its call is Z for every jet (recall 0). Keeping the two class-block MLPs restores recall 0.88, and those circuits then call 94 % of leptonic-top ($t\\to b\\ell\\nu$) jets $H\\to\\ell\\nu qq\'$.', 'results/part/p34_lepton_circuit.png'),
 ('Checks',
  'Unedited accuracy reproduces: ParT_full 85.9 % on the P1 reference sample, 86.2 % on this sample; ParT_kin 74.5 %. The no-op edit gives exactly zero logit change; ParT_kin, which has no flag inputs, gives exactly zero response to all flag edits. With no head kept the circuit outputs one class for all jets (check of the ablation). The P32v verify script, rerun on its own fresh sample (seed 9157), gives the same flag numbers (98.7 % vs 1.0 % for e2n). '
  'Limits: the AUC advantage over random head sets is significant after Holm correction only for $k=3$ ($p=0.04$); random sets that contain b2h1 reach the same AUC, and other class heads (b2h6, b2h8) reach 0.90–0.91 alone.', None),
 ('Why it matters',
  'The lepton signal is carried by a few class-attention heads, but the decision is made by the class-token MLPs, so a heads-only circuit of ParT\'s $H\\to\\ell\\nu qq\'$ call does not exist under the literal-circuit definition. The heads detect "a lepton is present", which semileptonic top jets share; telling the two apart needs the particle-block MLPs and many heads. This refines claim C15 and withdraws P32\'s "head circuit" reading.', None),
]
```

## Glossary additions (explain.GLOSSARY)
- **McNemar test:** test for paired yes/no outcomes on the same jets (e.g. call changed under edit A vs under edit B). It uses only the jets where the two outcomes differ, and asks whether one direction is more frequent than 50/50 (exact binomial p).
- **Literal circuit:** a set of units (attention heads, neurons) kept intact while every other head and MLP of the network is mean-ablated. The project's definition of a circuit: it must report its own accuracy and 10-class confusion.
- **Mean ablation:** replacing a unit's output, for every jet and token, by its average over a reference set of jets computed with the intact network, so it carries no jet-specific information.
- **Balanced accuracy:** the mean over the 10 classes of the fraction of each class's jets that are called correctly. 0.1 = the same call for every jet; insensitive to class proportions in the sample.
- **AUC (of a score):** the probability that a randomly chosen jet of the target class has a higher score than a randomly chosen jet of the other classes. 0.5 = no separation, 1 = perfect.
- **Value zeroing:** setting one particle's value vector to zero in one attention head. The attention weights are unchanged, so that head's output loses that particle's contribution. Compare the key mask (logit −10⁹), which also renormalises the weights over the other particles.

## Files
- Scripts:
  - `interp/p34_lepton_circuit.py` (new);
  - `interp/p34_job.sh` (copy of part_job.sh with k8s-haosu-09 excluded and a cached-image preference);
  - `interp/p32v_verify.py` (unchanged).
- Results:
  - `results/part/p34_lepton_circuit.json` (+ `_smoke`);
  - `results/part/p32_verify.json` (+ `_smoke`);
  - `results/part/p34_lepton_circuit.png`.
- Logs: `logs/c-p34-run4.log`, `logs/c-p34-p32v3.log`, `logs/c-p34-plot2.log`.
