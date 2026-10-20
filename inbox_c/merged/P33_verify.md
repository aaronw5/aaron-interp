# P33 verify: what ParT's pair features read in the axis response (verifier report, 2026-10-20)

## Conclusion
When one particle in a W or Z jet is moved by 0.01 rad, ParT's response through its pair features comes mostly from particles that have a neighbour closer than about 0.01 rad, and in ParT_full it passes through two of the eight pair-bias heads; both points reproduce on a fresh sample. The claim that the response's uneven dependence on step size comes from the logarithm of the pair distance alone, with no network nonlinearity needed, does not hold: a first-order (linearised) copy of the network fed the same input changes gives only about a third of the even response, so the network's own nonlinearity produces most of it. Carrying the moved particle's neighbours closer than 0.01 rad along with it removes only 20–35 % of the response (neighbours out to 0.05 rad must be carried to remove about 85 %), so "pairs closer than the step" describes which particles respond, not which pairs carry the response. The float32 rounding found in ParT_kin changes about 1 % of its 10-class calls on real jets; the rounding term also accounts for one of the three ParT_kin heads named in P33.

## Status of the rerun
- Script `interp/p33_verify.py` (imports the P33 helpers `load_rows`, `load_model`, `inputs_fixed`, `v64`, `Runner`; no change to P33 code).
- GPU job `anrunw-ai-c-p33-verify` was submitted twice; both pods were rejected with `UnexpectedAdmissionError` and never started (same node fault as reported in P32_verify).
- The rerun therefore ran on CPU: `anrunw-ai-c-p33-verify-cpu` (job.sh, AI_PY=conda + env-part on PATH, 4 CPU / 12 Gi; 34 min; 3.86 cores, peak 3.7 GiB) at a reduced size, plus `anrunw-ai-c-p33-verify-chk` (60 jets per class, 9 min) that adds the odd-part check of the linearisation.
- Results: `results/part/p33_verify_cpu.json` (main), `results/part/p33_verify_cpu_chk.json` (linearisation check), `results/part/p33_verify_smoke.json` on the volume.

## Code review of interp/p33_pair_channels.py, p33b_input_geometry.py, p33_plot.py
No errors found in these checks:
- **Rows.** `load_rows` draws random sorted indices over the whole file (`default_rng(3316)`), never first-k. Controls use random per-class indices.
- **Build.** `part_core.build` and `v64` both use η = jet_eta + sign(jet_eta)·Δη; edited and unedited rows go through the same build, floor exactly 0.
- **Edits.** `plan` moves only particle i (single) or the set `dist < RCL` (cluster) along i's radial unit vector; padded slots have `dist = inf` and are never moved. pT is kept, so the pT order and particle indices are unchanged (asserted on the mask and pT).
- **Heads.** The pair embedding outputs 8 channels shared by all 8 particle blocks; `U.reshape(-1, P, P)` matches `nn.MultiheadAttention`'s (batch·head) ordering, so `U[:, h]` is head h in every block. The own forward agrees with weaver's to 3·10⁻⁶.
- **Padding.** Pair counts and hinge counts are restricted to real×real pairs (`rpb`); the moved particles are real particles (first n stored slots).
- "10 highest-pT particles" in the docstring means the 10 leading stored particles (stored order); `build` re-sorts by pT and the edit follows the particle.

Points that affect the reading (not code errors):
1. **ParT_kin head shares were measured on the float32 pair route**, which contains the rounding term (issue listed by the worker, but the reading and thesis note still name "heads 2–4"). With float64 pair features, head 4's share is 0.000 (see below).
2. **The untrained network is not a no-nonlinearity control.** It has the same GELU hinges and softmax. Its partial correlation (0.11) being close to ParT_full's (0.13) does not test whether a nonlinearity is needed. The linearised network below is that control.
3. **"Non-smooth only where the input is"** is read from the bin pattern, but the P33 numbers already show deviations: ParT_kin (float64) 0.29 [0.14, 0.54] in the 0.01–0.03 bin where the input gives 0.081, and ParT_full 0.21 vs input 0.38 below 0.003.
4. **"Closer than about 0.01 rad"** comes from binning by nearest-neighbour distance; the only intervention used radius 0.05.

## Procedure of the rerun
- **Sample.** JetClass test_20M `WToQQ_117.root` + `ZToQQ_117.root` (P33 used `_116`), 300 random rows each (`numpy.random.default_rng(7117)`, sorted, never first-k); N = 600 jets, 6,000 moved particles (10 leading stored particles per jet). 45.5 % of moved particles have ΔR_nn < 0.01; median ΔR_nn = 0.011 rad (P33: 44 %, 0.0113).
- **Models.** ParT_full, ParT_kin (released weights). Pair features always in float64, cast to float32 before the pair embedding (P33's `pair64`), so the float32 rounding term is absent.
- **Edits per moved particle.** s± (±0.01 rad radial), s3± (±0.003), c± (particle + all particles within ΔR < 0.05 moved by the same 0.01 rad vector, = P33), k± (same with radius 0.01: only neighbours closer than the step carried; new).
- **Conditions.**
  - `pair`: unedited per-particle inputs, edited pair features (P33 pair64).
  - `lin` (new control): first-order network, d_lin = Σ_{c,i,j} ∂S/∂F_cij |unedited · (F_cij(edited) − F_cij(unedited)), S = logit_Z − logit_W, gradient taken once per jet. Its even part is produced only by the nonlinearity of the pair features in the particle coordinates (ln ΔR, ln kT, ln m²); the network enters through its fixed gradient.
  - On s± rows: KOh (head h's pair bias reset to unedited), ONLY (only the P33 top heads get the edited bias: full 0, 2; kin 2, 3, 4), KOTOP (those heads reset).
- **Input alone.** E_in = Σ_j even(Δ ln ΔR_ij)² for each move (P33b definition), extended to the k move.
- **Float32 calls (new).** 10-class argmax with float32 pair features (weaver as deployed) vs float64 pair features on the unedited jets: the W/Z sample, and a 10-class sample of 150 random rows per class from the first 25,000 rows of each `_117` file (`default_rng(7118)`, N = 1,500).
- **Read-out.** d = change of logit_Z − logit_W vs the unedited jet in the same condition; even (d₊ + d₋)/2, odd (d₊ − d₋)/2. Smoothness ratio (E/O at 0.003)/(E/O at 0.01); smooth response = 0.09.
- **Statistics.** 1,000 bootstrap resamples over jets; head shares one-sided bootstrap p (H0: share ≤ 1/8), Holm over 8 heads; McNemar exact test on paired correctness changes.

## Numbers
**Unedited accuracy.**

| model | W/Z sample (float32 / float64 pair features) | 10-class sample (float32 / float64) |
|---|---|---|
| ParT_full | 77.5 % / 77.5 % | 86.6 % / 86.6 % |
| ParT_kin | 63.5 % / 63.8 % | 74.1 % / 74.8 % |

(P1 sample: 85.9 % / 74.5 %; P33 W+Z `_116`: 74.3 % / 61.5 %.)

**Linearised-network control** (H0, P33 reading: the even response is the input's own nonlinearity passed through the network's first-order response, so even mean square lin/net ≈ 1 and R² ≈ 1; H1: the network's own nonlinearity contributes, ratio < 1).

| quantity (single 0.01 moves) | ParT_full | ParT_kin |
|---|---|---|
| even mean square, lin / net | 0.36 [0.15, 0.65] | 0.40 [0.26, 0.59] |
| R² of lin even vs net even | 0.07 | 0.09 |
| check: odd part at 0.003 rad, R² lin vs net (60-jet/class run) | 0.95 | 0.64 |
| check: odd part, ΔR_nn > 0.03, R² at 0.01 / 0.003 | 0.99 / 1.00 | 1.00 / 1.00 |

The gradient reproduces the odd response where the pair-feature change is small (ΔR_nn > 0.03, or 0.003 rad steps), so the linearisation is correct; for close pairs the network is outside its linear range. H0 is excluded for both models (CI upper < 1): about 60–65 % of the even mean square comes from the network's own curvature.

**Smoothness ratio by ΔR_nn bin** (pair route, float64 pair features; smooth = 0.09).

| ΔR_nn [rad] | n | input ln ΔR | ParT_full net | full lin | ParT_kin net | kin lin | full / kin share of even mean square |
|---|---|---|---|---|---|---|---|
| all | 6,000 | 0.64 [0.52, 0.77] | 0.119 [0.080, 0.181] | 0.079 [0.025, 0.39] | 0.240 [0.168, 0.365] | 0.184 [0.107, 0.263] | 1 / 1 |
| < 0.003 | 453 | 0.49 [0.38, 0.61] | 0.034 [0.017, 0.148] | 0.041 | 0.26 [0.10, 0.70] | 0.15 | 0.33 / 0.35 |
| 0.003–0.01 | 2,274 | 0.19 [0.17, 0.21] | 0.164 [0.108, 0.260] | 0.075 | 0.214 [0.168, 0.299] | 0.127 | 0.64 / 0.51 |
| 0.01–0.03 | 2,356 | 0.083 [0.080, 0.087] | 0.156 [0.084, 0.215] | 0.070 | 0.149 [0.106, 0.203] | 0.089 | 0.03 / 0.14 |
| > 0.03 | 917 | 0.090 | 0.118 [0.095, 0.130] | 0.087 | 0.098 [0.091, 0.109] | 0.092 | 0.002 / 0.003 |

- net − lin, all particles: full 0.04 [−0.24, 0.09] (p = 0.52), kin 0.06 [−0.04, 0.20] (p = 0.21): not separated overall.
- net − lin, 0.01–0.03 bin: kin 0.059 [0.015, 0.116] (p = 0.001), full 0.086 [−0.007, 0.162] (p = 0.11); > 0.03 bin, full 0.031 [0.007, 0.043] (p = 0.004).
- P33 (`_116`): ParT_full all 0.24 [0.17, 0.34]; this sample 0.119 [0.080, 0.181], whose interval contains 0.09.
- The network ratio does not follow the input ratio bin by bin: below 0.003 the input gives 0.49 and ParT_full 0.034; in 0.01–0.03 the input gives 0.083 and both models 0.15–0.16.

**Where the even response sits.** Particles with ΔR_nn < 0.01 hold 0.967 [0.937, 0.981] (full) and 0.857 [0.775, 0.913] (kin) of the even mean square (P33: 0.93 / 0.74). Reproduced.

**Neighbour-holding moves** (even mean square, cluster move / single move; H0 ratio = 1).

| move | n | ParT_full | ParT_kin | full lin | kin lin | input ln ΔR |
|---|---|---|---|---|---|---|
| carry neighbours within 0.05 (P33 move) | 5,620 | 0.162 [0.053, 0.367], p = 0.001 | 0.147 [0.086, 0.224], p = 0.001 | 0.054 | 0.052 | 0.077 [0.066, 0.090] |
| carry neighbours within 0.01 (new) | 2,727 | 0.648 [0.487, 0.807], p = 0.001 | 0.818 [0.553, 1.198], p = 0.34 | 0.58 | 0.44 | 0.394 [0.356, 0.436] |

- The 0.05 move reproduces P33's direction (P33: 0.087 / 0.103); on this sample the network ratio is about twice the input ratio, with overlapping intervals.
- Carrying only the neighbours closer than the step leaves 65 % (full) and 82 % (kin, not different from 1) of the even response.

**Heads** (float64 pair features; projection share of the even pair response; Holm over 8).

| head | ParT_full share | p_Holm | ParT_kin share | p_Holm |
|---|---|---|---|---|
| 0 | 0.475 [0.29, 0.64] | 0.008 | 0.002 | 1 |
| 1 | 0.010 | 1 | 0.136 [0.12, 0.16] | 0.64 |
| 2 | 0.554 [0.41, 0.69] | 0.008 | 0.360 [0.29, 0.43] | 0.008 |
| 3 | 0.001 | 1 | 0.477 [0.39, 0.56] | 0.008 |
| 4 | 0.098 | 1 | 0.000 [−0.001, 0.001] | 1 |
| 5 | 0.081 | 1 | 0.271 [0.22, 0.33] | 0.008 |
| 6 | 0.025 | 1 | 0.030 | 1 |
| 7 | 0.004 | 1 | 0.011 | 1 |

- ParT_full, only heads 0 and 2 edited: projection share 0.77 [0.69, 0.83], even mean square 0.68 of the all-head edit; heads 0 and 2 reset: 0.075 [0.047, 0.129] remains. P33's heads 0, 2 reproduce.
- The same two heads carry 0.63 [0.56, 0.68] of the odd part, so they carry the pair-route change in general, not specifically the close-neighbour even part.
- ParT_kin with float64 pair features: heads 3, 2, 5. Head 4 (P33: 0.446 on the float32 route) has share 0.000, so P33's head 4 carried the float32 rounding term. P33's set {2, 3, 4} edited alone gives 0.57 [0.53, 0.61].

**Float32 rounding and calls** (unedited jets; float32 vs float64 pair features).
- ParT_full: 0 of 1,500 10-class calls change; 0 of 600 W/Z calls.
- ParT_kin: 17 of 1,500 10-class calls change (1.1 %); correct only with float64 11, correct only with float32 1; McNemar exact p = 0.006. W/Z sample: 11 of 600 change (6 vs 4, p = 0.75).

## Hypotheses and outcome

| hypothesis | prediction | outcome |
|---|---|---|
| P33: even response = input log nonlinearity, no network nonlinearity needed | lin/net even ≈ 1, R² ≈ 1 | excluded: 0.36 / 0.40, R² < 0.1 |
| P33: non-smooth only where input ln ΔR is | net ratio tracks input ratio per bin | not reproduced (full < 0.003: 0.03 vs 0.49; both models 0.15 vs 0.08 in 0.01–0.03) |
| even response from particles with a neighbour < 0.01 | share of even mean square | reproduced (0.97 / 0.86) |
| carrying close (< 0.01) neighbours removes it | k-move ratio ≪ 1 | not observed (0.65 / 0.82) |
| carrying neighbours < 0.05 removes most of it | c-move ratio ≪ 1 | reproduced (0.16 / 0.15) |
| ParT_full: heads 0, 2 | ONLY ≈ 1, KOTOP ≈ 0 | reproduced (0.77, 0.075) |
| ParT_kin: heads 2–4 | head 4 share > 1/8 at float64 | excluded: head 4 = 0.000; heads 2, 3, 5 |
| float32 term does not change calls | McNemar p > 0.05 | excluded for ParT_kin (17/1,500, p = 0.006) |

Not retested: channel shares (no single channel ≥ 0.70 in ParT_full); the code for them was reviewed and found correct.

## Corrected reading
- The even pair-route response to a single 0.01 rad move sits in particles with a neighbour closer than 0.01 rad (97 % full, 86 % kin of the even mean square). Moving the particle together with every neighbour within 0.05 rad leaves 15–16 % of it; carrying only the neighbours within 0.01 rad leaves 65–82 %. The response is therefore carried by pairs out to about 0.05 rad, and is large for particles that also have a very close neighbour.
- The first-order network applied to the exact pair-feature changes gives 36–40 % of the even mean square (R² < 0.1). The input's log nonlinearity sets which particles respond, but most of the even response comes from the network's own nonlinear response to the large pair-feature changes of close pairs. The step-size ratio of the network does not separate from that of the linearised network overall, and it does not follow the input ratio bin by bin; the statement that the non-smoothness is inherited from ln ΔR alone is not supported.
- ParT_full: pair-bias heads 0 and 2 carry 77 % of the even response and 63 % of the odd response; resetting them leaves 7.5 %.
- ParT_kin, without float32 rounding: heads 3, 2 and 5. P33's head 4 carried the rounding term.
- Float32 rounding of the pair features changes 1.1 % of ParT_kin's 10-class calls (11 of 12 changed-correctness jets become correct with float64) and none of ParT_full's.

## Explainer entry (for site/explain.py)
```python
P33V_EXPLAINER = [
 ('Question', 'P33 found that the even part of ParT\'s pair-route response to moving one particle comes from close neighbours and attributed its non-smooth step dependence to the input $\\ln\\Delta R$ alone. Does that hold on a fresh sample, and does the network\'s own nonlinearity contribute?', None),
 ('Background', 'If the network were linear in the pair features $F$, its response would be $d_{\\rm lin}=\\sum_{c,i,j}\\partial S/\\partial F_{cij}\\,\\Delta F_{cij}$, and the even part $(d_++d_-)/2$ would come only from the nonlinearity of $F$ (e.g. $\\ln\\Delta R$) in the particle coordinates. Comparing $d$ with $d_{\\rm lin}$ separates the input nonlinearity from the network\'s.', None),
 ('Method', '600 random W/Z jets (JetClass _117), 10 leading particles moved by $\\pm0.01$ and $\\pm0.003$ rad, alone or with all neighbours within 0.05 or 0.01 rad; pair features in float64. Read-out $S=\\ell_Z-\\ell_W$. Controls: the linearised network, the input $\\ln\\Delta R$ alone, per-head pair-bias resets. Float32 vs float64 pair features on 1,500 jets of all 10 classes.', None),
 ('Result', 'The linearised network gives 36 % (ParT_full) and 40 % (ParT_kin) of the even mean square, so most of it is the network\'s own nonlinearity. Carrying neighbours within 0.05 rad leaves 16 % of the response, within 0.01 rad 65 %. ParT_full heads 0 and 2 carry 77 %; ParT_kin heads 3, 2, 5. Float32 rounding changes 1.1 % of ParT_kin\'s calls.', None),
 ('Checks', 'The gradient reproduces the odd response for small feature changes ($R^2$ 0.99–1.00 for $\\Delta R_{nn}>0.03$). Unedited accuracy 86.6 % (full) and 74.1 % (kin) on the 10-class sample. Bootstrap over jets; Holm over 8 heads; McNemar for call changes.', None),
 ('Why it matters', 'The close-pair geometry selects which particles move the pair route, but the size of the even response is produced by the trained pair embedding and attention, not by $\\ln\\Delta R$ alone; ParT_kin\'s deployed float32 pair features add a numerical dependence that changes some calls.', None),
]
```
