# P65 verify: flavour (H→bb vs H→cc) read-out in ParT_full (verifier report, 2026-10-22)

## Conclusion
On a fresh file, ParT_full's H→bb vs H→cc decision again depends on the charged tracks' impact parameters (how far each track misses the collision point), and four attention heads of the first class-attention block (the stage where one summary token reads all particles) pass this difference to the output. Swapping their outputs with those from another jet of the same class changes nothing, so the effect is flavour-specific, not a side effect of the swap. Two parts of P65's reading do not hold as stated. First, keeping a jet's own impact-parameter values but shuffling them among its tracks removes 90 % of the effect, so the "same-class resample" control is not neutral: the network reads each track's impact parameter together with that track's own kinematics. Second, the per-particle feed-forward layers (MLPs) are not shown to be where the flavour feature is computed: swapping all 64 particle-block heads carries at least as much flavour-specific information as swapping all 8 particle-block MLPs.

## Code review of interp/p65_flavour_circuit_part.py and interp/p65_plot.py
- Head and block indices are correct: modules 0–7 are the particle blocks (p1–p8) and modules 8–9 are class blocks b1–b2 (`HEADS`, `hlab` from p32). Per-head outputs are taken after `out_proj` and before the block's `c_attn` scale and LayerNorm. This is a linear decomposition of the attention output, and its sum matches torch's attention to within 1.2·10⁻⁴.
- Padding: padded query positions are excluded from means (`realq`), and the key padding mask is applied. Donor and recipient batches share one padded length, and the donor's padded positions get the donor's mean. Checks: donor logits under common padding match the clean logits to within 6·10⁻⁶; everything ablated reproduces the donor exactly; everything kept reproduces the clean output to within 8·10⁻⁶.
- Rows are drawn at random with `default_rng`, never first-k. The P1 accuracy check uses the fixed P1 sample (rows 3,000–4,000 of `_100`), as intended.
- Build: `rebuild` uses `part_core.build` with the sign(η) fix. Impact-parameter edits leave the four-momenta bit-identical (checked).
- Edits: only the four IP columns of charged tracks change; neutral particles are untouched (checked).
- Caveat on the class-MLP "neurons": in weaver's ParT Block a LayerNorm over all 512 units (`post_fc_norm`, scale_fc = True) follows the GELU. Mean-ablating one GELU output therefore also shifts the normalisation seen by the other 511. The single-neuron effects are "unit plus its LayerNorm side effect".
- P65's random controls for the joint ablation (k ≤ 8) and for the literal circuit at k = 8 are block-matched inside class block 1, which has only 8 heads. The random sets therefore share most heads with the selected set: at k = 8, 6 of the 8 heads come from b1. The comparison at k = 8 only tests which 6 of the 8 b1 heads are chosen.
- Figure `results/part/p65_flavour_circuit.png`:
  - Panel (c) plots the subsample values (k = 32: 0.643), while the report table quotes the full evaluation set (0.658). Both are correct; the panel title should name the subsample.
  - The "everything ablated" label overlaps the random-set bars.
  - Panels (a) and (b) agree with the JSON (b1h8: 0.21 of attention on the 3 largest-S tracks, 0.03 on the 3 smallest).
- No bug was found that changes P65's numbers.

## Rerun (job `anrunw-ai-c-p65-verify`, 8 min including smoke)
- Script: `interp/p65_verify_part.py`. It reuses P65's Run, edits and builders. The head ranking and the 32 class-MLP neurons are frozen from P65 (not re-ranked).
- Outputs: `results/part/p65_verify.json`, `results/part/p65_verify.png`, `logs/c-p65-verify.log`.

**Sample**
- JetClass test_20M `<sample>_118.root` (P65 used `_117`), seed 6599, random rows.
- Evaluation jets: 2,000 each of H→bb, H→cc, Z→qq and QCD, plus 300 each of the other six classes (N = 9,800).
- Donor pool: 300 per class.
- Same-class impact-parameter pool: 1,000 per main class.
- Light-track pool: 2,000 QCD jets.
- Unedited ParT_full:
  - P1-sample accuracy 0.8593 (as in P1);
  - 10-class accuracy on this sample 0.821;
  - b-vs-c accuracy 0.9815 [0.9775, 0.9855].

**Definitions**
- s_bc = logit(H→bb) − logit(H→cc), the network's b-vs-c log-odds.
- Margin u = +s_bc on H→bb jets and −s_bc on H→cc jets. Its mean is 8.48 here.
- Δu = the mean change of u under an edit; "share of margin" = Δu / 8.48.
- S = |d0|/σ(d0), the transverse impact-parameter significance.
- Intervals are 95 % bootstrap intervals over jets (1,000 resamples).

### 1. Input edits (P65's a–d reproduced, plus new control e)
| edit (what changes in the jet) | Δu_bc [CI] | share of edit a | b-vs-c acc. after |
|---|---|---|---|
| a: every charged track's (d0, σ(d0), dz, σ(dz)) ← a QCD track of the same charge and nearest pT | −6.39 [−6.60, −6.18] | 1 | 0.605 |
| b: the same, only for the 3 largest-S tracks | −1.70 [−1.81, −1.60] | 0.27 | 0.923 |
| c: the same, only for the 3 smallest-S tracks (control for b) | −0.29 [−0.32, −0.26] | 0.05 | 0.978 |
| d: every track's IP ← a pT- and charge-matched track from another jet of the same class | −4.81 [−4.96, −4.67] | 0.75 [0.74, 0.77] | 0.847 |
| **e (new): the jet's own IP 4-tuples permuted among its own charged tracks** | −5.72 [−5.84, −5.59] | **0.90 [0.87, 0.92]** | 0.806 |

- Edit e keeps each jet's set of IP values exactly (checked per jet). Only which track carries which value changes.
- H0 (e is a neutral edit): Δu = 0. Result: e loses 732 correct b-vs-c calls and gains 31 (McNemar p 6·10⁻¹⁷⁵). H0 is rejected.
- a − e = −0.67 [−0.84, −0.49]. a − d = −1.58 [−1.72, −1.42] (P65: −1.66).
- Competing readings of control d:
  - (H_generic) the network responds to any change in the IP values. It predicts that e, which keeps the jet's values, should give less than d. Excluded: e gives more (0.90 vs 0.75 of a).
  - (H_track) the network reads each IP value together with the kinematics of the track that carries it. Predicts that d and e are both large. Observed.
- So d and e are not flavour-neutral controls. They destroy the track-level b-jet signature (which tracks are displaced, consistent with their pT and direction).
- A light-specific comparison that keeps the track-level structure, for example replacing H→cc tracks' IP values with pT-matched H→bb values, was not run.
- Edit a takes b-vs-c accuracy to 0.605, below ParT_kin's 0.79 without any IP inputs. Light-like IP values therefore push H→bb jets toward "not b" (Δs_bc on H→bb −12.1, on H→cc +0.6); they do not only remove information.

### 2. Head and MLP swaps: other-flavour donor (P65) vs same-class donor (new null)
**Procedure**
- Jets: the 4,000 H→bb + H→cc evaluation jets.
- A unit's output (one head's 128-dim contribution, or a particle block's MLP output, token by token in pT order) is replaced by the same unit's output on a donor jet:
  - "other": a pT-matched jet of the other flavour, as in P65;
  - "same" (null): a pT-matched different jet of the same class.
- pT matching: the median |log(pT ratio)| is 0.0007 for both donor types.
- H0: Δu(other) = Δu(same). H1: the other-flavour swap removes more margin.
- Test: per-jet paired difference, bootstrap CI, normal-approximation p, Holm within each family (16 single heads; 6 joint sets; 9 MLP sets).

| unit (what it is) | other: share of margin [CI] | same: share of margin | other − same Δu [CI] | b-vs-c acc. other / same |
|---|---|---|---|---|
| b1h8 (class block 1 head 8) | −0.625 [−0.631, −0.618] | +0.006 | −5.35 [−5.40, −5.30] | 0.854 / 0.997 |
| b1h2 | −0.532 | +0.013 | −4.62 | 0.922 / 0.995 |
| b1h3 | −0.253 | +0.013 | −2.26 | 0.966 / 0.991 |
| b1h7 | −0.194 | +0.007 | −1.71 | 0.970 / 0.990 |
| p7h5, b1h1, p8h4, b2h1 | −0.07 to −0.05 | −0.015 to 0 | −0.48 to −0.62 | ≥ 0.978 |
| 8 random other heads | −0.043 to −0.002 | −0.024 to 0 | −0.16 to +0.04 | ≥ 0.978 |
| top 2 heads jointly | −1.165 | +0.014 | −10.0 | 0.268 / 0.998 |
| all 8 heads of class block 1 | −1.870 | +0.018 | −16.0 | 0.020 / 0.989 |
| all 64 particle-block heads | −1.367 [−1.379, −1.355] | −0.310 [−0.321, −0.298] | −8.97 [−9.07, −8.86] | 0.122 / 0.978 |
| one particle-block MLP (each of 8) | −0.05 to −0.13 | −0.06 to +0.005 | −0.14 to −1.01 | ≥ 0.975 |
| all 8 particle-block MLPs | −1.152 [−1.162, −1.142] | −0.310 [−0.321, −0.299] | −7.14 [−7.23, −7.05] | 0.298 / 0.981 |

- All listed top-head, joint and MLP differences have Holm p < 10⁻²⁰.
- P65's single-head numbers reproduce on the fresh file: b1h8 −5.30 (P65 −5.33); mean ablation −2.66 (P65 −2.67).
- The same-class null for the class-block heads is at most 1.8 % of the margin. The P65 resample shares are therefore flavour-specific and not caused by the swap itself. This supports P65's head read-out claim.
- Competing readings of "the particle MLPs compute the flavour feature":
  - (H_MLP) the MLPs compute it and the heads only move it. Predicts that the flavour-specific part of an MLP swap far exceeds that of a particle-head swap.
  - (H_joint) both carry it. Predicts comparable parts.
  - Observed: the flavour-specific part (other − same) is −8.97 for the 64 particle heads and −7.14 for the 8 MLPs. H_MLP is not supported; H_joint is not excluded.
  - Caveat: these are swaps on the full network. They show where flavour-dependent content sits in the per-particle stream, not where it is first computed.
- Single MLP blocks each carry 2–12 % of the margin (most in blocks 5, 6, 8). No single block dominates.

### 3. Key masking (reproduced)
One head's keys are computed from the edit-a jet; queries and values from the unedited jet. Shares of the edit-a effect:

| keys masked | this run | P65 |
|---|---|---|
| b1h8 | 0.183 [0.178, 0.188] | 0.185 |
| b1h2 | 0.023 | 0.022 |
| top 8 | 0.303 | 0.30 |
| all 80 | 0.541 [0.532, 0.550] | 0.55 |

### 4. Literal circuit
**Procedure**
- Kept: the frozen top-k heads plus 32 class-MLP neurons.
- Everything else is replaced by its value on a pT-matched random-class donor jet.
- Random controls: 20 block-matched random head sets per variant on 1,000 H→bb + 1,000 H→cc jets; the minimum possible p is 0.048.

| variant | b-vs-c acc. [CI] (9,800 jets) | 10-class acc. | P65 | selected vs 20 random sets (subsample) |
|---|---|---|---|---|
| k = 8 | 0.520 [0.505, 0.537] | 0.112 | 0.514 | 0.516 vs mean 0.507 (max 0.514), p 0.048 |
| k = 32 | 0.654 [0.640, 0.669] | 0.150 | 0.658 | 0.644 vs 0.520 (max 0.550), p 0.048 |
| k = 8 + particle MLPs kept | 0.633 | 0.133 | 0.647 | 0.633 vs 0.542 (max 0.591), p 0.048 |
| k = 32 + particle MLPs kept | 0.843 [0.832, 0.855] | 0.326 | 0.850 | 0.837 vs 0.660 (max 0.790), p 0.048 |
| 0 heads + particle MLPs + 32 neurons | 0.505 | 0.094 | – (new) | – |
| all 80 heads + all neurons, particle MLPs ablated | 0.730 | 0.429 | 0.718 | – |
| nothing kept (floor) / everything kept | 0.505 / 0.9815 | 0.094 / 0.821 | same | – |

- The literal-circuit numbers reproduce within 0.02.
- At k = 8 the circuit is 1 to 1.5 points above chance and above random sets by under 1 point. The "above random" result at k = 8 holds but is small.
- Random head sets also gain from keeping the particle MLPs: k = 32 random rises from 0.52 to 0.66. The selected heads still add about 0.18 over random.
- Particle MLPs kept without heads give the floor. This is expected: only attention heads move particle information into the class token.

## Verdict on each part of P65's reading
- Input edits a–d (75 %, 20 %, 4 %, 56 % of the margin): reproduced.
- "26 % of edit a is specific to the light values" / "the input-level response is mostly pattern-generic" (thesis note): the arithmetic reproduces (a − d = 25 % of a). The interpretation does not: control e shows the network reads each track's IP with that track's kinematics. d is not a neutral control, so how much of edit a is light-specific is still unmeasured.
- Class-block-1 heads b1h8 and b1h2 carry more than half of the margin under resample ablation: reproduced, and now backed by a same-class null (≈ 0). Under mean ablation the shares are 31 % and 15 %.
- Key masking (18 % for b1h8, 55 % for all 80): reproduced.
- The literal circuit stays near chance up to k = 8 and reaches 0.66 at k = 32; 0.85 with the particle MLPs: reproduced.
- "The flavour feature is computed in the per-particle MLPs": not supported. Swapping all 64 particle-block heads carries at least as much flavour-specific information (other − same −8.97) as swapping all 8 MLPs (−7.14). Random head sets also gain about 0.14 when the MLPs are kept.

## Corrected reading
- Light-matching all tracks' impact parameters removes 75 % of ParT_full's b-vs-c margin (fresh file: 0.982 → 0.605 accuracy).
- Shuffling a jet's own impact-parameter values among its tracks removes 90 %, so the network uses which tracks are displaced together with their kinematics, not the jet's set of values alone.
- In class block 1, heads b1h8 and b1h2 carry 63 % and 53 % of the margin when their outputs are swapped with those of a pT-matched jet of the other flavour; the same swap from a jet of the same class changes the margin by less than 2 %.
- Flavour-dependent content in the per-particle stream sits in both the particle-block heads and the particle-block MLPs: swapping all of either with other-flavour outputs removes 0.84–1.06 of the margin beyond the same-class swap.
- A literal circuit of the top 32 heads plus 32 class-MLP neurons reaches 0.65 b-vs-c accuracy; keeping the particle MLPs raises it to 0.84 (random head sets with MLPs: 0.66).

## Suggested follow-ups
- A light-specific control that keeps the track-level structure: swap H→cc tracks' IPs with pT-matched H→bb values, and the reverse.
- Edit e restricted to the 3 largest-S tracks: move their IP values onto 3 other tracks of the same jet.
- Value-path masking for the 45 % that does not enter through keys.
- A particle-MLP unit scan with the same-class null.

## Explainer additions (for site/explain.py; ledger agent)
- **Checks** (append to the P65 explainer):
  - On a fresh file (`_118`, seed 6599) every main number reproduces within 0.02.
  - Replacing a head's output by its output on a $p_T$-matched jet of the *same* class changes the margin by under 2 %. Only an *other-flavour* donor moves it ($-0.63$ of the margin for b1h8), so the head carries flavour, not swap noise.
  - Shuffling a jet's own impact-parameter values among its tracks removes 90 % of the effect: $\Delta\bar u_e/\Delta\bar u_a = 0.90\,[0.87, 0.92]$.
- **Glossary**:
  - *Same-class donor null*: resample ablation with a donor jet of the recipient's own class. It measures the effect of the swap itself, without any change of flavour.
  - *Within-jet IP permutation*: the jet's (d0, σ(d0), dz, σ(dz)) values are permuted among its own charged tracks. The jet keeps its set of displacement values, but which track carries which value is scrambled.
