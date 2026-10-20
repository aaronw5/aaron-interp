# P33: what ParT's pair features read in the axis response (C14, C12, C1; exploratory; takes up proposed P30) (agent report, 2026-10-20)

Jobs (all complete):
- `anrunw-ai-c-p33-smoke`, `c-p33-smoke2` (smoke, 8 jets per class); `c-p33-full` failed in its smoke-analysis tail (shape mismatch from an older condition list; no result used) and was replaced by `c-p33-full2` (ParT_full, GTX 1080 Ti, 20 min).
- `c-p33-kin` (ParT_kin, 13 min), `c-p33-ctrl` (ParT_untr + ParT_kin_perm, 7 min), `c-p33-ana` (CPU analysis, 26 s).
- Added in this round: `c-p33b-geo` (CPU, 41 s; input-level geometry of the same moves, `interp/p33b_input_geometry.py`) and `c-p33-plot` (figure).
- Scripts: `interp/p33_pair_channels.py`, `interp/p33b_input_geometry.py`, `interp/p33_plot.py`.
- Results: `results/part/p33_pair_channels.json`, `results/part/p33b_input_geometry.json`; per-row logits in `$AI_RESULTS/part/p33_pair_channels_<model>.npz` (volume only, 17–94 MB, not for commit).
- Figures: `results/part/p33_smoothness.png` (summary), `results/part/p33_pair_channels.png` (four-panel detail).
- Logs: `logs/c-p33-*.log`, `logs/c-p33b-geo.log`.

## Conclusion
When one particle in a W or Z jet is moved by 0.01 rad, ParT's response through its pair features comes almost entirely from that particle's pairs with close neighbours (closer than about 0.01 rad). The response is not smooth in the step size only for those particles. The cause is the input itself: the logarithm of a pair's angular distance changes sharply when the step is comparable to that distance. Applying ln ΔR alone, with no network, gives the same non-smooth pattern, and no specific nonlinearity (hinge) in the pair-embedding network is needed to explain it. The response is spread over the three distance-dependent pair channels (ln kT, ln ΔR, ln m²) rather than held by one, and in ParT_full it passes mainly through two of the eight pair-bias heads (attention heads 0 and 2 in every block). Also, about 40 % of ParT_kin's pair response to such moves (and part of the non-smoothness P27v reported) is float32 rounding noise in the self-pair mass feature, not a physical dependence. Still uncertain: what heads 0 and 2 compute from close pairs, and whether the float32 effect changes ParT_kin's calls on real jets.

## Procedure
**Models.**
- ParT_full and ParT_kin (released JetClass weights).
- Controls (300 jets per class each, no head knock-outs):
  - ParT_untr: ParT_full architecture, weaver initialisation, torch seed 3316;
  - ParT_kin_perm: ParT_kin with the output rows of every weight tensor permuted, seed 3316.

**Sample.**
- JetClass test_20M `WToQQ_116.root` and `ZToQQ_116.root`, 1,500 random rows each (`numpy.random.default_rng(3316)`, sorted indices, never first-k). N = 3,000 jets; none dropped.
- Moved particles: the 10 leading stored particles of each jet, 30,000 particles in total.
- Profile of the moved particles:
  - median nearest-neighbour distance ΔR_nn = 0.0113 rad;
  - 44 % have ΔR_nn < 0.01 rad and 7.9 % have ΔR_nn < 0.003 rad;
  - 94 % have at least one other particle within ΔR < 0.05.

**Build.** We used `part_core.build` (η = jet_eta + sign(jet_eta)·Δη) with pT kept, so the particle order is unchanged. Pair features come from weaver's own `pairwise_lv_fts`. The forward pass is the model's own, with the pair bias U passed explicitly; it agrees with the weaver forward to within 2.9·10⁻⁶ (full) and 1.9·10⁻⁶ (kin). Both the edited and the unedited rows are rebuilt the same way, and the numerical floor on the unedited rows is exactly 0 in every condition.

**Edits.**
- (A) Whole-jet moves:
  - φ + 0.06 rad;
  - the matched dilation: (Δη, Δφ) scaled by √(M2'/M2), where M2 = Σ pT r² is the axis moment and M2' its value after the φ shift.
- (B) Single-particle radial moves, outward and inward, by 0.01 and by 0.003 rad, plus 10⁻⁶ rad as a numerical-smoothness probe.
- (C) Neighbour-holding (rigid cluster) move: the particle and every particle within ΔR < 0.05 of it are moved by the same 0.01 rad radial vector, so the distances inside that cluster stay fixed.

**Conditions** (per edited row):
- `full`: everything edited.
- `part`: per-particle inputs edited, pair inputs unedited.
- `pair`: all four pair channels edited, per-particle inputs unedited.
- Channel-only `kT`, `z`, `dR`, `m2`: only that channel taken from the edited jet.
- `no_c`: all channels edited except c.
- `m2diag` / `m2off`: only the diagonal (i = j) or only the off-diagonal entries of ln m² edited.
- `pair64`: as `pair`, with the pair features computed in float64 and then cast to float32.
- `KOh` (single moves, trained models only): as `pair`, with the pair bias of head h set to its unedited value.

**Read-out.**
- d is the change of (logit_Z − logit_W) relative to the unedited jet run in the same condition.
- Even part: (d₊ + d₋)/2. Odd part: (d₊ − d₋)/2.
- Shares are projection shares Σ e_c·e_pair / Σ e_pair², computed over particles.

**Smoothness.** The even/odd variance ratio is (E/O at 0.003) / (E/O at 0.01). A smooth (twice-differentiable) response gives (0.003/0.01)² = 0.09.

**Input geometry (P33b).** No network is involved. For the same moved particles we compute D_j = Δ ln ΔR_ij for every other particle j; ln kT and ln m² change by the same amount and by twice it, respectively. From these we form E_in = Σ_j even(D_j)² and O_in = Σ_j odd(D_j)², and then the input smoothness ratio and the input cluster/single ratio. The network's even response is related to the layer-1 near-collinear hinge-crossing count using a rank partial correlation at fixed √E_in.

**Statistics.**
- 95 % bootstrap intervals over jets (1,000 resamples; 200 for the partial correlations).
- Head shares: one-sided bootstrap p for H0: share ≤ 1/8, Holm-adjusted over the 8 heads.
- Top-3 heads: chosen on a random half of the jets and evaluated on the other half, against all 56 three-head sets.
- Channel differences: bootstrap of the share difference, H0: 0.
- The sign-flip permutation test "channel vs none" turned out degenerate. The unedited floor is exactly 0, so any nonzero rounding passes, and even `z`, whose even mean square is 10⁻¹⁴, gets p = 0.00025. Its p-values (all 0.001 after Holm over 4 channels) are listed but not used.

## Numbers
**Unedited accuracy** (10-class argmax, this W+Z sample):

| model | all | W | Z | reference |
|---|---|---|---|---|
| ParT_full | 74.3 % | 79.3 % | 69.3 % | P27 W+Z sample (_106): 75.1 %; P1 sample: 85.9 % |
| ParT_kin | 61.5 % | 67.5 % | 55.5 % | P27 W+Z: 62.5 %; P1: 74.8 % |
| ParT_untr | 0.3 % | | | |
| ParT_kin_perm | 0.0 % | | | |

**Route split** (single 0.01 rad moves, even mean square).

| quantity | ParT_full | ParT_kin |
|---|---|---|
| pair / full | 0.961 [0.933, 0.990] | 0.977 [0.965, 0.986] |
| per-particle-only even share of its own variance | 0.06 | 0.01 |
| route additivity R² (full = part + pair) | 0.978 | 0.992 |

The pair share reproduces P27v's 96–98 %.

**Channels** (projection share of the even pair response; 95 % CI):

| channel | full: only this edited | full: lost when only this is not edited | kin: only this edited | kin: lost |
|---|---|---|---|---|
| ln kT | 0.435 [0.30, 0.56] | 0.39 [0.28, 0.49] | 0.107 [0.07, 0.14] | 0.13 [0.09, 0.17] |
| ln z | 0.000 | 0.000 | 0.000 | 0.000 |
| ln ΔR | 0.174 [0.14, 0.22] | 0.51 [0.43, 0.58] | 0.201 [0.16, 0.25] | 0.35 [0.28, 0.41] |
| ln m² | 0.393 [0.31, 0.48] | 0.39 [0.26, 0.52] | 0.656 [0.59, 0.72] | 0.74 [0.69, 0.79] |
| — ln m² diagonal only | 0.000 | | 0.418 [0.32, 0.51] | |
| — ln m² off-diagonal only | 0.392 | | 0.233 [0.19, 0.28] | |

- ln z = min(pT)/(pT_i + pT_j) does not depend on angles, so its zero is a built-in null.
- Channel additivity R² (sum of channel-only responses vs the all-four response):
  - ParT_full −3.9 [−5.1, −3.0]. In full, editing ln kT alone gives an even mean square 2.3 times that of the all-four edit.
  - ParT_kin 0.35 [0.18, 0.48].
  - controls 0.97–0.99.
- Full, share differences: kT − m² = 0.04 [−0.09, 0.15], p = 0.49; kT − ΔR = 0.26 [0.11, 0.38], p = 0.001.
- Bootstrap P(top share ≥ 0.70): 0.000 (full, top = kT); 0.079 (kin, top = m², of which 0.42 is the diagonal).
- Controls (only-this-edited share): m² takes 0.86 (untr) and 0.97 (perm), carried by the diagonal (0.66 / 0.90).

**Float32 rounding in the self-pair mass** (ParT_kin).

| quantity | ParT_kin | ParT_full |
|---|---|---|
| pair64 / pair even mean square | 0.52 | 1.00 |
| even mean square, 10⁻⁶ rad move / 0.01 rad move (smooth expectation 10⁻⁸; 0 with pair64) | 0.42 [0.31, 0.52] | 0.001 |
| score shift on unedited jets, float32 → float64 pair features: mean | 0.041 | 0.003 |
| score shift on unedited jets: max | 5.45 | 0.41 |

**Smoothness ratio** (smooth = 0.09).

| condition | ParT_full | ParT_kin |
|---|---|---|
| pair | 0.242 [0.17, 0.34] | 1.64 [1.25, 2.08] |
| pair64 | 0.237 [0.16, 0.33] | 0.44 [0.24, 0.75] |
| per-particle route | 0.111 | 0.125 |

- The input ln ΔR alone (no network) gives 0.588 [0.54, 0.64] on the same moves.
- By ΔR_nn bin (pair64): see the table below.

| ΔR_nn [rad] | n | input ln ΔR | ParT_full | ParT_kin (pair64) | ParT_kin (float32) | full: share of even mean square |
|---|---|---|---|---|---|---|
| < 0.003 | 2,359 | 0.38 [0.35, 0.43] | 0.21 [0.10, 0.40] | 0.63 [0.26, 1.22] | 0.74 | 0.40 |
| 0.003–0.01 | 10,947 | 0.19 [0.18, 0.20] | 0.23 [0.17, 0.30] | 0.19 [0.15, 0.26] | 1.64 | 0.53 |
| 0.01–0.03 | 12,203 | 0.081 | 0.10 [0.07, 0.14] | 0.29 [0.14, 0.54] | 1.97 | 0.069 |
| 0.03–0.05 | 2,618 | 0.090 | 0.11 [0.09, 0.13] | 0.10 | 13.0 | 0.001 |
| > 0.05 | 1,873 | 0.090 | 0.10 [0.09, 0.11] | 0.12 | 6.5 | 0.000 |

**Neighbour-holding (cluster) move vs single move.** Even mean square ratio, particles with ≥ 1 neighbour within 0.05 (n = 28,127). The cluster move changes M2 4.6 times more than the single move.

| condition | ParT_full | ParT_kin | ParT_untr | input ln ΔR |
|---|---|---|---|---|
| pair | 0.087 [0.05, 0.14] | 1.80 [1.38, 2.21] | 4.2 | 0.081 [0.075, 0.086] |
| pair64 | 0.085 [0.05, 0.13] | 0.103 [0.08, 0.13] | 0.071 [0.04, 0.12] | |
| ln kT only | 0.091 | 0.086 | 0.080 | |
| ln ΔR only | 0.063 | 0.069 | 0.063 | |
| per-particle route | 15.6 | 93 | | |
| full | 0.68 [0.53, 0.86] | 2.24 | | |

**Hinge map** (layer 1 of the pair-embedding MLP, sign changes of the pre-activation; near-collinear = unedited ΔR < 0.05).
- Mean crossings per moved particle (out + in): 18.4 (full) and 29.8 (kin). 88 % (full) and 91 % (kin) of particles have ≥ 1 near-collinear crossing, and those particles hold 99.9 % / 99.7 % of the even mean square.
- Excess (share of even mean square minus share of particles): full 0.115 [0.110, 0.120], kin 0.088; untr −0.03 [−0.10, 0.03].
- Correlations (Spearman, |even pair response| vs measure):

| measure | full | kin (pair64) | untr (pair64) | perm (pair64) |
|---|---|---|---|---|
| near-collinear crossings | 0.47 | 0.51 | 0.54 | 0.20 |
| √E_in (input change) | 0.50 | 0.49 | 0.73 | 0.20 |
| crossings vs √E_in (between the two measures) | 0.79 | 0.79 | 0.67 | 0.67 |
| partial: crossings at fixed √E_in | 0.133 [0.118, 0.146] | 0.223 [0.207, 0.237] | 0.109 [0.075, 0.139] | 0.082 [0.050, 0.113] |

**Heads** (pair bias of head h set to its unedited value in all 8 particle-attention blocks; projection share of the even pair response; Holm over 8 heads, H0: share ≤ 1/8).

| head | full share | full remaining even after KO | full p_Holm | kin share | kin p_Holm |
|---|---|---|---|---|---|
| 0 | 0.493 [0.42, 0.56] | 0.40 | 0.008 | 0.000 | 1 |
| 1 | 0.009 | 0.98 | 1 | 0.076 | 1 |
| 2 | 0.605 [0.53, 0.68] | 0.32 | 0.008 | 0.228 [0.18, 0.28] | 0.008 |
| 3 | 0.001 | 1.00 | 1 | 0.268 [0.21, 0.33] | 0.008 |
| 4 | 0.148 [0.11, 0.19] | 0.77 | 0.73 | 0.446 [0.35, 0.53] | 0.008 |
| 5 | 0.092 | 0.86 | 1 | 0.118 | 1 |
| 6 | 0.032 | 0.95 | 1 | 0.009 | 1 |
| 7 | 0.003 | 1.00 | 1 | 0.013 | 1 |

- Head additivity R²: 0.57 (full), 0.55 (kin). The single shares sum to 1.38 and 1.16.
- Top 3 heads, chosen on half A and evaluated on half B:
  - full: heads 2, 0, 4 hold 1.22 [1.12, 1.32] of the even response; all 56 three-head sets give mean 0.51 (2.5–97.5 %: 0.05–1.14); p = 1/56 = 0.018.
  - kin: heads 4, 3, 2 hold 0.96 [0.89, 1.03] against a mean of 0.45; p = 0.018. The kin head shares are measured on the float32 pair route, which includes the rounding part.
- Mean |ΔU| per head under the full pair edit (ParT_full): heads 0 (0.033) and 2 (0.025) are the largest. Under the ln kT-only and ln m²-only edits, head 6 has the largest |ΔU| (0.058, 0.048), yet its knock-out share is 0.03.

**Whole-jet edits** (mean d with 95 % CI; share of the full response).

| edit | quantity | ParT_full | ParT_kin |
|---|---|---|---|
| φ + 0.06 | full | 0.723 [0.68, 0.77] | 0.540 [0.51, 0.58] |
| φ + 0.06 | pair route | 0.0000 | −0.009 [−0.015, −0.003] (all from the ln m² diagonal, float32) |
| dilation | full | 0.897 | 0.744 |
| dilation | pair route | 0.147 [0.137, 0.157] (16 % of full) | 0.082 (11 %) |
| dilation | ln kT only | −0.142 | |
| dilation | ln ΔR only | +0.072 | |
| dilation | ln m² only | +0.200 | |

The pair features are exactly invariant under a common φ shift, so the φ route is 100 % per-particle (as in P2).

## Reading
- **Where the even pair response sits.** In both trained models it comes from the moved particle's pairs with close neighbours:
  - Particles with ΔR_nn < 0.01 (44 % of moved particles) hold 93 % (full) and 74 % (kin) of the even mean square.
  - The rigid cluster move, which fixes those short distances, leaves 0.085–0.10 of the single-move even pair response while it changes M2 4.6 times more.
  - The same cluster/single ratio holds for Σ_j even(Δ ln ΔR_ij)² of the inputs alone (0.081) and for the untrained network with float64 pair features (0.071). The reduction is therefore set by which pair distances the move changes, not by a trained mechanism.
- **Non-smoothness.**
  - The input ln ΔR is itself non-smooth under these steps: 0.59 overall, 0.38 for ΔR_nn < 0.003, and 0.08–0.09 above 0.01. This is because ln ΔR_ij(±δ) has its expansion radius at ΔR_ij.
  - ParT_full's pair route (0.24) and ParT_kin's float64 pair route (0.44) are non-smooth in the same bins (ΔR_nn < 0.01) and close to 0.09 above 0.03.
  - Layer-1 hinge crossings track the size of the input change (ρ = 0.79). At fixed input change they add a rank partial correlation of 0.13 (full) and 0.22 (kin), against 0.11 (untrained) and 0.08 (permuted). The full model is inside the untrained interval; kin is above it.
  - The thesis prediction that the non-smoothness is concentrated in hinge crossings is not separable from the input's own log singularity in this measurement. The alternative's prediction, a geometric input effect, accounts for the bin pattern.
- **Float32 rounding in ParT_kin.**
  - ParT_kin's float32 pair route has an additional component. It is present under a 10⁻⁶ rad move at 42 % of the 0.01 rad even mean square and disappears when the pair features are computed in float64. It is carried by the diagonal ln m²_ii, the self-pair invariant mass squared, computed as E² − |p|² in float32.
  - This component accounts for kin's smoothness ratio 1.64 (0.44 without it) and for kin's cluster/single ratio 1.8 (0.10 without it). It also accounts for the untrained and permuted controls' m²-dominated channel shares.
  - P27v's "non-smooth, ratio 0.38–1.72" for ParT therefore mixes the log-distance geometry (full) with this rounding term (kin).
- **Channels.**
  - ParT_full: no single channel reaches 0.70 (top: ln kT 0.44, CI upper 0.56). ln kT, ln ΔR and off-diagonal ln m² each remove 0.39–0.51 when left unedited.
  - ParT_full: channel-only edits are not additive (R² −3.9). Each one alone produces a pair-feature combination that does not occur in real jets (ln kT − ln ΔR = ln min pT fixed, ln m² ≈ ln pT_i pT_j + 2 ln ΔR), and the shares are therefore not a decomposition.
  - ParT_kin: off-diagonal ln m² 0.23, ln ΔR 0.20 and ln kT 0.11, plus the 0.42 diagonal rounding term.
  - This matches the alternative (the response spread over the ΔR-dependent channels) and excludes the thesis's single-channel ≥ 70 % for ParT_full.
- **Heads.** The even pair response is concentrated:
  - ParT_full: heads 0 and 2 (shares 0.49 and 0.61; knocking out either leaves 0.40 / 0.32); the top 3 hold 1.22 against 0.51 for a random three-head set.
  - ParT_kin: heads 2, 3 and 4.
  - This supports the "≤ 3 heads" part of the thesis and excludes the alternative's spread heads. The head with the largest bias change (head 6 under channel-only edits) is not the one that carries the response. The head shares are not additive (R² 0.55–0.57).
- **Whole-jet edits.** The φ shift does not use the pair route (exact invariance). Under the matched dilation the pair route carries 11–16 %, with ln kT-only and ln m²-only of opposite signs in ParT_full.
- **Scope.** Statements are scoped to W/Z jets of one test file, the 10 leading stored particles, and radial moves of 0.003–0.01 rad.

**Thesis vs alternative.**

| prediction | outcome |
|---|---|
| one channel ≥ 70 % | excluded (full) / not met after removing rounding (kin) |
| non-smoothness from pair-MLP hinges | not separated from the input log singularity; the input alone reproduces the bin pattern |
| neighbour-holding removes most of the even part | observed (0.085–0.10 remains), equally in inputs and untrained network, so geometric |
| ≤ 3 heads | observed (full: heads 0, 2; kin: 2, 3, 4) |
| alternative "spread across ΔR channels" | supported |
| alternative "neighbour-holding does not remove it" | excluded |
| alternative "heads spread" | excluded |

## Explainer entry (for site/explain.py)
```python
P33_EXPLAINER = [
 ('Question', 'ParT responds to moving a single particle by 0.01 rad, and almost all of the even part of that response (same sign for outward and inward moves) goes through the pair features $U_{ij}=\\mathrm{MLP}(\\ln k_T,\\ln z,\\ln\\Delta R,\\ln m^2)$ (P27v). Which pair channel, which heads, and why is it not smooth in the step size?', None),
 ('Background', 'For a smooth response $d(\\delta)=a\\delta+b\\delta^2+\\dots$ the even part scales as $\\delta^2$ and the odd part as $\\delta$, so the ratio $(E/O)_{0.003}/(E/O)_{0.01}$ equals $(0.003/0.01)^2=0.09$. A larger value means non-smooth. The pair input $\\ln\\Delta R_{ij}$ has a log singularity: if the step $\\delta$ is comparable to $\\Delta R_{ij}$, $\\ln|\\Delta R_{ij}\\pm\\delta|$ is not well approximated by its Taylor expansion.', None),
 ('Method', '3,000 random W/Z jets (JetClass _116). The 10 leading particles are moved one at a time by $\\pm0.01$ and $\\pm0.003$ rad radially, or together with every particle within $\\Delta R<0.05$ (rigid cluster move). The network is fed edited pair features for one channel at a time (others unedited), all four, or all four in float64; each head\'s pair bias is reset to its unedited value in turn. The same moves are applied to $\\ln\\Delta R$ alone (no network). Read-out: change of $\\ell_Z-\\ell_W$. Controls: untrained and weight-permuted ParT.', None),
 ('Result', 'The non-smooth even response sits in particles with a neighbour closer than the step ($\\Delta R_{nn}<0.01$: 93 % of the even response in ParT_full), and $\\ln\\Delta R$ alone shows the same pattern (figure). The cluster move leaves 9 % of the even pair response, as for the inputs alone (8 %). ParT_full routes it mainly through pair-bias heads 0 and 2; no single channel carries more than 44 %.', 'results/part/p33_smoothness.png'),
 ('Checks', 'Numerical floor 0; own forward = weaver forward within $3\\cdot10^{-6}$; unedited accuracy 74.3 % (full) and 61.5 % (kin) on this W+Z sample, matching P27. In ParT_kin, 42 % of the even response is float32 rounding in the self-pair $\\ln m^2_{ii}$ (present at a $10^{-6}$ rad move, gone with float64 pair features). Hinge crossings add a partial correlation of 0.13 at fixed input change, inside the untrained network\'s 0.08–0.14.', None),
 ('Why it matters', 'The pair route of ParT\'s per-particle response reads the moved particle\'s distances to its closest neighbours, through the three distance-dependent channels together and two attention heads; its non-smoothness is inherited from $\\ln\\Delta R$, not created by the pair-embedding nonlinearities. The float32 term is a numerical sensitivity of ParT_kin, not a physics dependence.', None),
]
```
Conclusion (for site/conclusions.py): the text of the Conclusion section above.
