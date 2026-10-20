# X13 — what sets the size of the stored-axis translation response across networks (A4 / A10m, exploratory)

## Conclusion
Every trained network tested (56: JEDI-linear, PFN, EFN, Deep Sets, Linformer, transformers) responds to a "radial-only" edit, which moves each particle's distance from the jet axis exactly as a 0.06 rad shift of the jet would, about as strongly as to a real mass change of the same size. The networks differ in how much of that response survives when the whole jet is shifted instead, which keeps every particle-to-particle distance fixed: in JEDI-linear (HGQ), PFN and EFN almost all of it survives, while in many transformers and Linformers only a third or less does. So the size of the axis response is set by how much the network also uses particle-to-particle distances, which cancel part of the axis-distance signal. This spread is not an architecture constant in these data: two seeds of the same small transformer can differ by a factor of 6 (ratio 2.07 against 0.35), and the architecture labels explain no more of the spread than shuffled labels (permutation p = 0.38).

## Question
The calibrated translation ratio R (read-out response to a rigid 0.06 rad translation of the constituents, divided by the response to a matched dilation) ranged 0.07–2.0 across networks trained on stored-axis inputs. Does the per-particle radial route ($r_i$ = distance from the stored axis) or pairwise structure set R, and is R an architecture property or a seed property?

## Procedure
- Script `interp/x13_axis_size.py` (groups `jedi`, `flow`, `s64`, `s32`; `--analyze`), figure `interp/x13_plot.py`. Jobs x13-run-jedi, x13-run-flow, x13-run-s64, x13-run-s32 (each 1 CPU / 2 GiB; peak 0.5–1.8 GiB, 0.87–0.96 cores), x13-ana (1 CPU / 2 GiB; 1.06 GiB, 0.44 cores). No training.
- Sample: `jedi_core.sample(per_class=1000, seed=1313)`, N = 5,000 test jets, identical jets for every network (n = 128 networks read the same rows of `val_ptetaphi_f16.npy`; first 64 particles checked equal). Fixed split rng(1313): 2,500 fit jets (read-out fit), 2,500 test jets (all read-out numbers).
- Networks: 143 = 56 trained + 87 untrained redraws.
  - Trained: official JEDI-linear 3f n32, n64, n128 and 3f-perminv n64; X12 retrains (HGQ s0–2, HGQ-r150 s0–2, float s0–1); X9 PFN/EFN (9; exact numpy forward pass, `interp.x9_efn_pfn.Flow`, runs in the jax env); X0/X1 torch small models N 64 (Deep Sets s0–2, JEDI-style Deep Sets, two Linformers, six transformers incl. e100 ones; s0–1) and N 32 (Deep Sets, two Linformers, four transformers; s0–1).
  - Untrained: 3 redraws per architecture (JEDI: `redraw_untrained` seeds 0–2; torch default init, `torch.manual_seed(5000+k)`; PFN/EFN he-uniform, zero biases).
- Read-out: ridge regression of $\log m$ (invariant mass of the n input particles) on the last hidden layer (standardised; α chosen on the last quarter of the fit half; E6v/X2 procedure), fitted on unedited fit-half jets, applied before/after each edit on test jets. Secondary: same on the pooled latent; native calls (W→Z flow on true W jets). Untrained networks: logistic head refit on the last hidden layer (native output constant for JEDI redraws).
- Edits (real particles only, padding untouched, c = 0.06 rad):
  - $T_k$: rigid translation by $c$ at azimuth $2\pi k/8$, k = 0..7 (invariant mass unchanged; mean $\tfrac12\Delta\log M_0M_2$ = +0.302 at n = 64; pairwise distances unchanged).
  - $R_k$ (radial-only): each particle keeps its azimuth about the stored axis; $r_i \to |p_i + c_k|$, the same per-particle $r$ as $T_k$ (max $|\Delta r|$ difference 1.2e-7). Changes pairwise distances (mean 27.9 mrad) and the invariant mass ($\Delta\log m$ = +0.241).
  - dil: per-jet dilation of all angles with $\log s = \mathrm{mean}_k\,\tfrac12\Delta\log M_2(T_k)$, so $\tfrac12\Delta\log M_0M_2$ equals the translation's azimuth mean per jet (+0.302; $\Delta\log m$ +0.302).
  - $A'_d$ (pair route, E24 A′): per-particle azimuth scramble at fixed $r_i$, then shift+rescale so $M_0$, vector $M_1$, $M_2$ are held exactly (3 draws). Pairwise distances change by 63.8 mrad; $\Delta\log m$ = 0.000 on average.
  - rot (null): rigid rotation about the stored axis by a random angle per jet; $r_i$, pairwise distances, $m$, $M_0$, $M_2$ unchanged.
- Quantities (test jets, read-out on last hidden layer): $R = \overline{d(T)}/\overline{d(\mathrm{dil})}$; $R_\mathrm{rad} = \overline{d(R_k)}/\overline{d(\mathrm{dil})}$; $Q_\mathrm{rad} = \overline{d(R_k)}/\overline{d(T)} = R_\mathrm{rad}/R$; $Q_\mathrm{pair} = \overline{|d(A')|}/\overline{|d(T)|}$; $R_\mathrm{pair} = \overline{|d(A')|}/\overline{d(\mathrm{dil})}$ (pair sensitivity without the translation in the denominator); odd/even = first azimuthal harmonic of $d(T_k)$ / mean.
- Statistics: jet bootstrap B = 1000 (same jet indices for every network); across-network Spearman with a bootstrap over networks and jets (1000), permutation p (10,000), partial Spearman given test accuracy and read-out $R^2$; variance decomposition of $\log R$ into between-family (same architecture, size and N, different seeds; 21 families with ≥ 2 seeds, 45 networks) and within-family parts, 10,000 label permutations; paired seed differences with jet-bootstrap CIs; Holm over networks / pairs.

## Checks
- Official JEDI-linear 3f n64 on this sample: 80.46 % (N = 5,000; binomial SE 0.56 %; 260k-jet value 80.94 %, difference −0.48 % = −0.9 SE). n128 80.94 %, perminv 81.18 %, n32 77.14 %.
- Edit checks (n = 64): T leaves $m$ unchanged ($\Delta\log m$ 0.0000) and pairwise distances unchanged (0.0 mrad); R reproduces T's $r_i$ to 1.2e-7; A′ holds $M_0$, $M_1$, $M_2$ ($\tfrac12\Delta\log M_0M_2$ −0.0000); rot changes no $r_i$ (0.0 mrad).
- Rotation null: $|d(\mathrm{rot})|/d(\mathrm{dil})$ = 0.05–0.38 across trained networks (mean signed rotation response near 0). For every trained network the 95 % CI of $d(T) - d(\mathrm{rot})$ excludes 0 (bootstrap p = 0.002 each, the B = 1000 floor; Holm over 56 networks = 0.11 for all, limited by the floor, not by the effect).
- One network with degenerate calls (x12-hgq-r150_s1, 56.8 %, one class 39 % of calls) included; dropping it changes no Spearman by more than 0.003.

## Results

### 1. Radial-only response vs translation response
| class (trained nets) | R median [range] | $R_\mathrm{rad}$ median [range] | $Q_\mathrm{rad}$ range |
|---|---|---|---|
| JEDI-linear HGQ (10: 4 official + 6 X12) | 1.03 [0.65, 1.91] | 1.12 [0.92, 1.89] | 0.97–1.43 |
| JEDI-linear float (2) | 0.43, 0.88 | 0.95, 1.01 | 2.21, 1.15 |
| PFN (5) | 1.05 [0.99, 1.20] | 1.09 [1.02, 1.22] | 1.01–1.07 |
| EFN (4) | 1.11 [1.03, 1.22] | 1.12 [1.02, 1.20] | 0.99–1.01 |
| Deep Sets (5) | 0.75 [0.64, 0.83] | 0.89 [0.83, 0.98] | 1.07–1.36 |
| JEDI-style Deep Sets (2) | 0.81, 0.95 | 0.99, 1.16 | 1.22, 1.23 |
| Linformer (8) | 1.09 [0.26, 1.49] | 1.13 [0.94, 1.31] | 0.88–3.55 |
| transformer (20) | 0.54 [−1.11, 2.07] | 1.05 [0.64, 1.38] | 0.67–4.85 (two negative) |

- Across all 56: $R_\mathrm{rad}$ median 1.05 (IQR 0.99–1.18), SD of $\log R_\mathrm{rad}$ 0.15; $R$ median 0.94 (IQR 0.62–1.13), SD of $\log R$ (54 positive) 0.48. Per-network jet-bootstrap CIs of R and $R_\mathrm{rad}$ are ±0.01–0.2.
- Spearman($R$, $R_\mathrm{rad}$) = +0.77 [0.61, 0.87], perm p 1e-4; partial given accuracy and read-out $R^2$: +0.80 [0.65, 0.88].
- Spearman($R$, $Q_\mathrm{rad}$) = −0.75 [−0.96, −0.46], perm p 1e-4; partial −0.70 [−0.95, −0.42]. (Equivalently +0.75 for the radial share $1/Q_\mathrm{rad} = R/R_\mathrm{rad}$.) $R$ and $Q_\mathrm{rad}$ share $\overline{d(T)}$; since $R_\mathrm{rad}$ varies little, this correlation mostly restates that the spread of $R$ is the spread of $\overline{d(T)}/\overline{d(R_k)}$.
- Same on the pooled latent: Spearman($R$, $Q_\mathrm{rad}$) −0.61 [−0.87, −0.34]; on native W→Z flow −0.95 [−0.98, −0.81].
- The low-R networks also have large odd/even (first azimuthal harmonic of the translation response over its mean): 1.3–4.7 for transformers with R < 0.55, 0.15–0.40 for PFN/EFN/X12 HGQ.

### 2. Pair route
- $Q_\mathrm{pair}$ (A′ response / translation response): 0.14–0.21 for X12 HGQ, PFN, EFN; 0.34–0.44 for transformers with R < 0.55. Spearman($R$, $Q_\mathrm{pair}$) −0.68 [−0.81, −0.51] (shares $d(T)$ with R).
- Normalised by the dilation instead ($R_\mathrm{pair}$, independent of $d(T)$): Spearman($R$, $R_\mathrm{pair}$) −0.25 [−0.52, +0.05], perm p 0.069; partial given accuracy and $R^2$ +0.32 [−0.02, +0.53]. Rotation null, same normalisation: Spearman(R, $|d(\mathrm{rot})|/d(\mathrm{dil})$) −0.16 [−0.45, +0.12].
- Calls changed by A′: 6–9 % (PFN, X12, JEDI official), 12–26 % (transformers); rotation 2–12 %.

### 3. Architecture vs seed
- Between-family share of var($\log R$): 0.48 (45 nets, 21 families); permutation 95th percentile 0.64, p = 0.38. Linear R: 0.56, q95 0.62, p = 0.14. By architecture class (8 classes, 54 nets): 0.19, q95 0.25, p = 0.19. Within-family SD of $\log R$ 0.52 vs total 0.51.
- On native calls ($R_\mathrm{fl}$, W→Z flow ratio): between-family share 0.83, q95 0.64, p = 2e-4; class share 0.41, p = 0.008. $R$ (read-out) and $R_\mathrm{fl}$ are not rank-correlated: Spearman +0.16 [−0.15, +0.44].
- Paired seed differences (same jets; 29 pairs): 28 of 29 CIs exclude 0 (bootstrap p 0.002–0.02; Holm 0.058 for all, the B = 1000 floor). Small within-family spread: X12 HGQ ΔR −0.04 to −0.13, PFN +0.12/+0.17, EFN +0.12, Deep Sets +0.02 to +0.10. Large: tf-1b-d32-n64 +1.72 [1.69, 1.76] (2.07 vs 0.35), tf-1b-d32-n32 +1.66, Linformer-1b n32 −1.04, tf-2b-d32-n64-e100 +0.77, X12 float −0.45 [−0.48, −0.42] (0.43 vs 0.88).
- Untrained redraws (refit read-out): R from −0.86 to 4.6 across all; transformers 0.72–2.18, Linformers 0.99–1.39, Deep Sets 0.20–0.80, PFN 0.49–0.91; read-out $R^2$ 0.01–0.81 (JEDI redraws ≤ 0.54, output constant).

### 4. Accuracy / read-out quality (competing hypothesis c)
- Spearman($R$, test accuracy) +0.24, perm p 0.073; Spearman($R$, read-out $R^2$) +0.50, perm p 2e-4. The $R$–$R_\mathrm{rad}$ relation holds with both partialled out (+0.80).

## Hypotheses
- H0: R not related to the radial route (Spearman CI includes 0) and architecture labels explain no more variance than permuted labels. H1: Spearman > 0.6 and between-architecture share > permutation 95th percentile.
  - Radial part: the CI excludes 0 (|ρ| 0.75–0.77), but the sign for $Q_\mathrm{rad}$ as defined in the plan is negative: low-R networks respond more to the radial-only edit than to the translation.
  - Architecture part: H0 not rejected (0.48 vs q95 0.64, p 0.38).
- (a) radial-route share sets R. Predicted: $Q_\mathrm{rad}$ ≈ 1 for networks with R ≈ 1, $Q_\mathrm{rad}$ < 1 and $Q_\mathrm{pair}$ large for low-R networks.
  - First prediction: matches (PFN, EFN, X12 HGQ: 0.97–1.08).
  - Second: $Q_\mathrm{rad}$ < 1 is excluded for low-R networks (2.0–4.9). $Q_\mathrm{pair}$ is larger there, but not after normalising by the dilation.
  - What is measured: $R_\mathrm{rad}$ ≈ 1 for all networks, and R falls below $R_\mathrm{rad}$ by the part of the radial response that the translation does not produce. The two edits have the same $r_i$ and differ only in the pairwise distances, which the translation keeps.
- (b) seed / optimisation sets R. Predicted: within-family spread as large as between. Not excluded: within-family SD of log R 0.52 vs total 0.51, p 0.38. The seed spread is small for JEDI-linear HGQ, PFN, EFN and Deep Sets, and large for transformers, Linformers and JEDI-linear float.
- (c) accuracy or read-out quality sets R. R vs read-out $R^2$ +0.50 (p 2e-4) and vs accuracy +0.24 (p 0.07); the $R$–$R_\mathrm{rad}$ relation is unchanged with both partialled out. This excludes (c) as the only factor, but not as a contributor.

## Reading (scoped)
- On these 2,500 test jets and with this read-out, every network's log m read-out moves under the radial-only edit by 0.8–1.4× the matched dilation (one transformer 0.64, JEDI n128 1.89). The spread of R comes from the difference between the translation and the radial-only edit. That difference is a change of pairwise distances at fixed $r_i$; for $\log m$ read-outs it is a positive response in the low-R networks.
- The size of that difference is close to 0 for JEDI-linear HGQ retrains, PFN and EFN in every seed. It varies between seeds in attention models and in the float JEDI-linear. The official JEDI-linear n64 (R 0.84, $Q_\mathrm{rad}$ 1.43) and perminv n64 (0.65, 1.42) lie between the two groups.
- The independent pair edit A′ (azimuth scramble at fixed $M_0$, $M_1$, $M_2$) shows no significant rank relation to R once it is normalised by the dilation. A network's general sensitivity to pairwise rearrangement does not track R in this test; what tracks R is the response to the specific change of pairwise distances that separates the radial-only edit from the translation.
- Not measured here: which units carry the cancellation in the low-R transformers; whether $R$ on the native calls (which is architecture-structured, p 2e-4) reflects mass-window placement rather than the read-out.

## Files
- `interp/x13_axis_size.py`, `interp/x13_plot.py`
- `results/jedi/x13_jedi.json`, `x13_flow.json`, `x13_s64.json`, `x13_s32.json` (per-network rows), `x13_summary.json` (all statistics), `x13_axis_size.png`; per-network arrays at `/j-jepa-vol/aaron-interp/results/jedi/x13_nets/*.npz`
- logs: `logs/x13-run-{jedi,flow,s64,s32}.log`, `logs/x13-ana.log`

## Explainer draft (for site/explain.py; Paper A tab, A4/A10m)
- **Question.** Why does the axis (translation) response differ so much between networks (R from below 0.3 to 2)?
- **Background.** Inputs are angles relative to a stored jet axis. A rigid shift of all constituents by $c$ leaves the invariant mass unchanged. It does change $M_2 = \sum_i p_{T,i} r_i^2$ about the stored axis, by $\tfrac12\Delta\log M_0M_2 \approx 0.30$ on average for $c$ = 0.06 rad. A network that reads mass through the $r_i$ responds as if the mass changed. We measure that response relative to a real mass change of the same $\tfrac12\Delta\log M_0M_2$: a dilation $\Delta\eta_i, \Delta\phi_i \to s\,\Delta\eta_i, s\,\Delta\phi_i$. This gives $R = \overline{\Delta \hat y(T)}/\overline{\Delta \hat y(\mathrm{dil})}$, with $\hat y$ a ridge read-out of $\log m$.
- **Method.** Radial-only edit: $r_i \to |\vec p_i + \vec c|$ at fixed azimuth $\theta_i$. It has the same $r_i$ as the shift but different pairwise distances $|\vec p_i - \vec p_j|$, which the shift keeps. $R_\mathrm{rad}$ is the same ratio for this edit. If a network used only the $r_i$, then $R = R_\mathrm{rad}$. Pair edit A′: scramble $\theta_i$ at fixed $r_i$, then hold $M_0$, $\vec M_1$, $M_2$. Null: rigid rotation about the axis. This was run on 56 trained networks of 8 architecture classes, with 2–3 seeds per architecture, on the same 2,500 test jets.
- **Result.** Figure `x13_axis_size.png`. $R_\mathrm{rad} \approx 1$ for all (median 1.05). $R \le R_\mathrm{rad}$ in most networks, and $R \ll R_\mathrm{rad}$ in many transformers. Spearman($R$, $R_\mathrm{rad}$) = 0.77. Seeds of the same transformer give R = 2.07 and 0.35. Architecture labels explain 48 % of var log R vs 64 % at the permutation 95th percentile.
- **Checks.** Rotation null below the shift response in all 56 networks. Shift leaves $m$ and pairwise distances exactly unchanged; the radial edit reproduces the shift's $r_i$ to $10^{-7}$. Partial correlations given accuracy and read-out $R^2$. Official n64 accuracy on the sample 80.46 %.
- **Why it matters.** The un-recentred mass response is present in every trained network ($R_\mathrm{rad}$ ≈ 1). Its visible size under a jet shift depends on how much the network also uses the shift-invariant pairwise distances, and in attention models that varies with the seed. A single R value is therefore a property of one trained network, not of the architecture.
