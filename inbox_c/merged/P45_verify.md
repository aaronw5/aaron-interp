# P45v: verification of P45 (model stitching ParticleNet_full → ParT_full)

Script `interp/p45v_verify.py` (imports data loading, the two networks, the edits and the statistics helpers unchanged from
`interp/p45_stitch_pn_part.py`). Job `anrunw-ai-c-p45-verify`: smoke run, then a full run. The first full run stopped on a GPU
fault on node k8s-haosu-12 ("CUDA error: unspecified launch failure" while computing teacher logits; log
`logs/c-p45-verify.try1.log`). Two relaunches were rejected at admission ("GPU is lost") on k8s-chase-ci-01. The fourth
launch completed (exit 0, 48 min). Results: `results/part/p45_verify.json`, figure `results/part/p45_verify.png`, log
`logs/c-p45-verify.log`.

## Conclusion

On fresh jets the accuracy half of P45 holds. ParticleNet's per-particle features after its first or second graph layer
replace ParT's own particle features at ParT's first attention block through one linear-plus-offset map, at a cost of about
1 point of 10-class accuracy, against about 9 points for an untrained ParticleNet. The axis-response half does not hold as
P45 states it. ParT's internal features fed into ParticleNet's later layers keep ParT's larger response to a mass-preserving
shift of the jet axis. ParticleNet's smaller response appears only when its second- or third-layer features are read out
through ParT's last block. Untrained or raw-input features fed into ParT's first block give a smaller response than trained
ParticleNet features, so ParT's later blocks do not rebuild ParT's response from any input that carries the particle angles.

## Definitions

- **Stitching**: the frozen networks are cut at one point, and the internal features of one network (the *donor*) are fed
  into the rest of the other network (the *recipient*) through a learned affine map t_i = W x_i + b, applied to each
  particle i (or to the jet-level vector). Only W and b are trained.
- **ParT_full / ParticleNet_full (PN)**: the released JetClass taggers, both using all 17 per-particle inputs. ParT: an
  embedding MLP, 8 particle attention blocks with a pair bias U computed from pairwise (k_T, z, ΔR, m²), 2 class-token
  blocks and a linear head. PN: 3 EdgeConv layers (EC1 64, EC2 128, EC3 256 numbers per particle; each one updates a
  particle from its 16 nearest neighbours, using (Δη, Δφ) to find them in EC1 and the previous features in EC2/EC3), a mean
  over particles (256), then a 2-layer MLP head (fc).
- **Stitch names**: `pn2>b1` = PN EC2 output → input of ParT block 1 (ParT's embedding skipped). `pn2>b8` = EC2 output →
  input of ParT block 8 (blocks 1–7 skipped). `pnpool>cls` = PN's pooled 256-vector → ParT's final class-token vector.
  `u2` = the same from an untrained PN (PyTorch default weights, torch seed 4545). `raw` = ParT's own 17 scaled inputs.
  `id` = ParT's own block input (code check). New in this verification, the reverse direction (ParT → PN): `pt1>ec2` =
  ParT token after block 1 → input of PN EC2 (EC1 skipped); `pt8>ec2` = ParT token after block 8 → input of EC2;
  `pt1>ec3`, `pt8>ec3` = the same into EC3 (EC1 and EC2 skipped); `ptcls>fc` = ParT's class-token vector (128) → PN's pooled
  vector (256), so that only PN's MLP head follows.
- **Training of a map**: least-squares start (ridge 1e-4 × mean diagonal, 20,224 jets) to the recipient's own activation at
  the stitch point, then Adam on KL(p_recipient ‖ p_stitched) = Σ_c p_c log(p_c / q_c) over the 10 class probabilities,
  where p is the recipient network's own output on the same jet.
- **Stitching gap** = accuracy(recipient alone) − accuracy(stitched), on the same jets, in percentage points.
- **d** = logit_W − logit_Z. **R** = the per-jet slope through 0 of the change of d under the φ shift against the change of d
  under the matched dilation, over true W, Z and QCD jets. In words: the network's W-vs-Z response to the axis shift divided
  by its response to a real angular dilation that raises the axis-moment mass √(M0·M2) by the same amount (M0 = Σ pT,
  M2 = Σ pT r² about the stored jet axis). R = 1: the shift counts like a real mass change; R = 0: it is ignored.
- **W → Z**: the fraction of true-W jets called Z, before and after the shift.
- **McNemar test**: exact binomial test on the jets whose correctness differs between two networks. **Holm**: step-down
  correction of p-values over the stitches of one test family. Bootstrap: 2,000 resamples of jets (p resolution 5e-4).

## Procedure

- Edits (as P45, `part_core.build`, η = jet_eta + sign(jet_eta)·Δη): (1) φ + 0.06 rad added to every real particle with the
  stored axis fixed. This is a rigid rotation about the beam: every pairwise quantity, so ParT's pair bias, is unchanged,
  and the invariant mass changes by at most 0.0007 in log m (val sample; 0.005 on P1). In a stitched network the shift reaches
  the recipient only through the donor's features. (2) Matched dilation: each particle's (Δη, Δφ) × k_j with
  k_j = √(M0M2(shifted)/M0M2(unedited)) per jet (mean k 1.08, mean Δlog m +0.071). Masks unchanged in both edits (checked).
- Training jets (fresh): train_100M, 2 random files per class (default_rng(4747); e.g. ZJetsToNuNu_042/_049, WToQQ_076/_015),
  15,000 random rows per file, N 300,000 (P45: default_rng(4545)). ParT 86.12 %, PN 84.38 % on them. Batch order seed 7.
- Training length: pass 1 = one pass with P45's schedule (Adam 5e-4, 3 % warm-up, cosine decay), then evaluation; pass 2 =
  a second pass (Adam 2.5e-4, cosine, other batch order), then evaluation again.
- Evaluation samples: v122 = val_5M *_122, 2,000 random rows per class (default_rng(4748)), N 20,000 (P45 used *_121, N
  10,000). P1 = test *_100 rows 3,000–4,000 per class, N 10,000 (the reference sample).
- Code checks: own ParT forward vs weaver |Δlogit| ≤ 3.8e-6; own PN forward vs weaver 0; PN restarted from its own EC1
  output at EC2, from its own EC2 output at EC3, and from its own pooled vector at fc: |Δlogit| = 0 in all three. Identity
  stitch id>b1: gap −0.01 [−0.06, +0.03] points, R 0.791 (ParT 0.791).
- Code review of `p45_stitch_pn_part.py`: the block indices (hs[k−1] = input of block k), the class-block path, the padding
  mask, the PN forward (bn → EC1–EC3 with the 1e9 coordinate shift on padded slots, counts-normalised mean), the
  least-squares fit on real particles only, the McNemar, Holm and bootstrap code and the confusion matrices were checked; no
  error found. Samples use random rows (P1 is the project's fixed reference sample). The build uses the corrected η sign.

## Hypotheses

- H_rec (P45's reading: the recipient's remaining blocks set R): forward stitches into early ParT blocks give ParT's R from any
  donor; reverse stitches into PN's later layers give PN's R (~0.5).
- H_don (the donor's features set R): forward stitches give PN's R at every point; reverse stitches give ParT's R.
- H0 for each pair of networks: equal R (two-sided paired bootstrap over jets).

## Numbers (v122, N 20,000; pass 1 → pass 2 where they differ by more than 0.02 in R or 0.5 points)

Unedited: ParT_full 86.42 % [85.91, 86.88] (P1: 85.93 %, reference 85.9 %); ParticleNet_full 84.50 % [83.99, 84.97] (P1 84.56 %).
ParT R 0.791 [0.776, 0.807], W → Z 10.4 → 31.2 % (McNemar p 4e-97). PN R 0.505 [0.484, 0.524], W → Z 11.3 → 26.0 %.

| stitch | recipient | acc % | gap [95 % CI] (points) | R [95 % CI] | P45 (_121) R | Holm p R vs PN | Holm p R vs ParT |
|---|---|---|---|---|---|---|---|
| pn1>b1 | ParT | 85.34 | +1.08 [+0.80, +1.38] | 0.774 [0.758, 0.789] | 0.776 | 0.008 | 0.088 |
| pn2>b1 | ParT | 85.08 | +1.34 [+1.05, +1.64] | 0.814 [0.798, 0.831] | 0.809 | 0.008 | 0.024 |
| pn3>b1 | ParT | 78.22 → 79.00 | +8.20 → +7.42 | 0.761 → 0.773 | 0.771 | 0.008 | 0.21 |
| pn1>b8 | ParT | 77.20 | +9.22 [+8.69, +9.75] | 0.773 → 0.790 | 0.738 | 0.008 | 0.29 |
| pn2>b8 | ParT | 81.72 | +4.70 [+4.28, +5.13] | 0.511 → 0.537 | 0.487 | 0.63 | 0.008 |
| pn3>b8 | ParT | 80.46 | +5.96 [+5.52, +6.44] | 0.492 [0.466, 0.516] | 0.482 | 0.63 | 0.008 |
| u2>b1 | ParT | 77.18 | +9.24 [+8.75, +9.75] | 0.677 [0.660, 0.694] | 0.678 | 0.008 | 0.008 |
| u2>b8 | ParT | 46.77 → 50.67 | +39.65 → +35.75 | 0.949 [0.924, 0.971] | 0.971 | 0.008 | 0.008 |
| raw>b1 | ParT | 72.56 | +13.86 [+13.30, +14.43] | 0.476 [0.454, 0.498] | 0.486 | 0.12 | 0.008 |
| raw>b8 | ParT | 30.37 | +56.05 [+55.29, +56.78] | 0.721 [0.696, 0.746] | 0.738 | 0.008 | 0.008 |
| pnpool>cls | ParT | 82.86 | +3.56 [+3.15, +3.95] | 0.297 [0.275, 0.317] | 0.353 | 0.008 | 0.008 |
| **pt1>ec2** (new) | PN | 81.54 | +2.96 [+2.57, +3.37] | 0.751 [0.728, 0.773] | – | 0.008 | 0.008 |
| **pt8>ec2** (new) | PN | 84.55 | −0.05 [−0.37, +0.27] | 0.770 [0.745, 0.794] | – | 0.008 | 0.085 |
| **pt1>ec3** (new) | PN | 82.01 | +2.49 [+2.12, +2.89] | 0.832 [0.810, 0.853] | – | 0.008 | 0.008 |
| **pt8>ec3** (new) | PN | 85.11 | −0.61 [−0.92, −0.32] | 0.702 [0.676, 0.731] | – | 0.008 | 0.008 |
| **ptcls>fc** (new) | PN | 86.03 | −1.53 [−1.84, −1.21] | 0.796 [0.779, 0.813] | – | 0.008 | 0.29 |

Gap for the reverse stitches is measured against PN (negative = more accurate than PN). Holm over the 16 non-identity
stitches; with 2,000 resamples the smallest Holm value is 0.008. Pass 2 changes every R by ≤ 0.03 (largest: pn2>b8
+0.026, pn1>b8 +0.017, pt8>ec3 −0.018) and every gap by ≤ 0.8 points except u2>b8 (−3.9 points), so the P45 numbers are not
an artefact of stopping after one pass. The P1 sample gives the same pattern (pn2>b1 R 0.833, pn2>b8 0.536, pt8>ec2 0.785,
ptcls>fc 0.790, PN 0.532, ParT 0.798).

Paired R differences (v122, pass 1; Holm over 8 pairs):

| pair | ΔR [95 % CI] | Holm p |
|---|---|---|
| pn2>b8 − u2>b8 (trained vs untrained EC2, late) | −0.438 [−0.463, −0.410] | 0.004 |
| pn2>b1 − u2>b1 (trained vs untrained EC2, early) | +0.137 [+0.115, +0.157] | 0.004 |
| raw>b1 − pn2>b1 (raw inputs vs trained EC2, early) | −0.338 [−0.364, −0.311] | 0.004 |
| pn1>b8 − pn2>b8 (EC1 vs EC2, late) | +0.262 [+0.238, +0.286] | 0.004 |
| pn2>b1 − pn2>b8 | +0.303 [+0.284, +0.321] | 0.004 |
| pn3>b1 − pn3>b8 | +0.269 [+0.237, +0.300] | 0.004 |
| pt1>ec2 − pt8>ec2 | −0.019 [−0.042, +0.003] | 0.094 |
| pt1>ec3 − pt8>ec3 | +0.129 [+0.104, +0.153] | 0.004 |

## What reproduces

- Accuracy: ec1/ec2 → block 1 cost 1.1–1.3 points (P45: 0.85–1.7); ec3 → block 1 costs 7.4–8.2 points and is not better than
  the untrained EC2 donor (9.2); untrained EC2 at block 1 costs 9.2 points; raw inputs 13.9 points; pooled → class token 3.6
  points. These match P45 within 0.5 points.
- R values of every forward stitch match P45's _121 values within 0.06 (largest: pnpool>cls 0.297 vs 0.353, pn2>b8 after
  pass 2 0.537 vs 0.487, pn1>b8 0.773 vs 0.738); P45's two seeds already differed by up to 0.07 at these points.
- Untrained-donor and raw-input stitches give R that rises from block 1 to block 8 (0.68 → 0.95; 0.48 → 0.72).

## What does not hold as stated

1. "Untrained features also give a ParT-sized response when fed into ParT's early blocks" and "ParT's later blocks set the
   size of the response from any features that carry the particle angles." At block 1, with all 8 ParT blocks following, R
   depends on the donor: trained EC2 0.814, untrained EC2 0.677, ParT's own 17 inputs through an affine map 0.476 (the raw
   inputs contain Δη, Δφ and ΔR). The two differences from trained EC2 are −0.14 and −0.34, Holm p 0.004 each; untrained
   and raw differ from ParT's 0.791 at Holm p 0.008. Under H_rec all three would be about 0.79.
2. "It matches ParT when most of ParT's blocks follow and ParticleNet when only ParT's last block follows." This holds for
   EC2 and EC3 features only. EC1 → block 8 gives 0.77–0.79 (ParT-sized, Holm p vs ParT 0.29), untrained EC2 → block 8 gives
   0.95, raw → block 8 gives 0.72. At block 8 the trained-vs-untrained EC2 difference is −0.44 (Holm p 0.004), so at that
   point R is set by the donor's features, not by the recipient.
3. Thesis note on C5: "ParticleNet's smaller axis response comes from its own later layers (EdgeConv-3, pooling, fc)." The
   reverse stitches test this directly and contradict it. ParT tokens after block 8, passed through PN's EC2, EC3, pooling
   and MLP head, give R 0.770 at PN's own accuracy (gap −0.05 points); ParT's class vector passed through PN's MLP head
   alone gives 0.796; ParT tokens after block 1 into EC2 or EC3 give 0.75–0.83. All are above PN's 0.505 (Holm p 0.008).
   So PN's EC2, EC3, pooling and head do not reduce R when they receive ParT features. In the forward direction the
   PN-sized R appears with EC2 or EC3 features read out by ParT block 8 (0.49–0.54) and not with EC1 features (0.77–0.79).
   The measured location of the smaller response is PN's first two EdgeConv layers together, read out by a short head; the
   pooled stitch (0.30, below PN) shows that the read-out also changes R.
4. Neither H_rec nor H_don describes all points. H_rec is excluded by the reverse stitches (predicted ~0.5, measured
   0.70–0.83) and by the donor dependence at block 1. H_don is excluded by the early forward stitches (predicted ~0.5,
   measured 0.76–0.81 for pn1/pn2/pn3 > b1). Description that fits all 16 points: R is ParT-sized whenever ParT's own
   embedding plus at least one block, or at least six ParT blocks after trained PN features, are in the path; it is
   PN-sized only when trained PN EC1+EC2(+EC3) features pass through no more than ParT's last block and class blocks.

## Checks on adequacy of the P45 controls

- The untrained-donor control (a) and raw-input floor (b) are informative, but they lower accuracy by 9–56 points, so their R
  values at blocks 5–8 (0.72–1.03) come from networks at 30–68 % accuracy; R of such networks is not comparable to R of an
  86 % network. P45 used them to argue that a ParT-sized R does not need trained PN features; at block 1, where accuracy is
  77 % and 73 %, they give R below ParT's.
- The missing control was the reverse direction (ParT features into PN). It is the direct test of "recipient vs donor" and is
  added here (5 stitches). Its maps reach PN's accuracy (gap −1.5 to +3.0 points), so these R values come from working
  networks.
- Training length: checked with a second pass (above).
- Not checked: other read-outs than d = logit_W − logit_Z; the kin models; other stitch points of the reverse direction
  (e.g. ParT block 3 or 5 into EC2).

## Figure

`results/part/p45_stitch.png` (P45): two panels (accuracy and R vs stitch point, _121 sample); the plotted values match the
JSON; the y-axis label "d-response" is shorthand for the change of d = logit_W − logit_Z. `results/part/p45_verify.png`: R
with 95 % bootstrap interval for every network after pass 1 and pass 2 (blue = into ParT, red = ParT features into PN; bar
text = 10-class accuracy; dashed lines = ParT and PN alone).

## Corrected reading

ParticleNet_full's EC1 and EC2 per-particle features replace ParT_full's tokens at blocks 1–3 through one affine map at a
cost of 1–2 points of 10-class accuracy (reproduced on val_5M _122, N 20,000: 1.1 / 1.3 points at block 1; untrained EC2 9.2
points); EC3 features lose 5–9 points. The axis-response ratio R of a stitched network is not set by the recipient alone:
ParT features fed into PN's EC2/EC3/pooling/head keep ParT-sized R (0.70–0.83 vs PN 0.51), and at ParT's block 1 the donor
changes R (trained EC2 0.81, untrained EC2 0.68, raw inputs 0.48). PN-sized R (0.49–0.54) appears only for trained EC2/EC3
features read out through ParT's block 8 and class blocks. PN's smaller response is therefore associated with its first two
EdgeConv layers plus a short read-out, not with its EC3, pooling or MLP head.
