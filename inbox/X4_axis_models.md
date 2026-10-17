# X4: invariant mass or $\sqrt{M_0 M_2}$ about the input axis in the hardest-particle-axis models (agent report, 2026-10-17)

**Question.** The models of E18 (`dsets-d64`, `linf-2b-d32-k8`, N = 64, `-axhard`) were trained with every particle's
$(\Delta\eta, \Delta\phi)$ measured from the hardest particle, so $\vec M_1 \neq 0$ on the training jets and
$\sqrt{M_0 M_2}$ about the axis differs from $m$ ($M_0 M_2 = m^2 + |\vec M_1|^2$; for a two-prong jet with the hardest
particle in a prong of momentum fraction $z$, $\approx m/\sqrt z$). Do their internal mass and their decisions follow $m$
(H_a) or $\sqrt{M_0M_2}$ about their own axis (H_b)? Stored-axis versions of the same architectures are the control.

**Code / outputs.** `interp/x4_axis_models.py` (stages `--prep`, `--model/--axis/--seed`, `--combine`); cluster jobs
`x4-prep`, `x4-{dsets,linf}-{h,s}{0,1}`, `x4-combine` (1 CPU / 2 GiB; peak 0.26–1.75 GiB). Results
`/j-jepa-vol/aaron-interp/results/small/x4_*.json|npz`, `x4_summary.json`, `x4_prep.json`; PNGs
`results/site/T15_axhard_mass.png`, `results/site/F8_axis_jump.png` (local copies under `results/`).

**Inputs.** Val archive (= the training code's test set), first 64 slots, float16 → float32, pT-ordered, axhard models
recentred with `train_small.recentre_hardest`, shift/scale from each `arch.json`. Exact check: softmax vs `test_probs.npy`
max |Δp| ≤ 1e-5 on the 20,000-jet sample and on all 104,533 W/Z jets, argmax agreement 1.000 for all 8 models.

## Test 1: read-outs on natural jets
Sample: 4,000 jets per class, seed 71; rng(71) halves, ridge (E6v procedure) from the last hidden layer (Deep Sets h3,
16 units; Linformer pool, 32), held-out $R^2$ with bootstrap CIs (1,000).
Correlation of $\log m$ and $\log\sqrt{M_0M_2}$ about the hardest particle: 0.979 (all classes), 0.909 (W/Z). About the
stored axis: 0.99999.

| model | $R^2(\log m)$ | $R^2(\log\sqrt{M_0M_2}_{\rm axis})$ | $R^2(\rho)$, $\rho = \log\sqrt{M_0M_2}_{\rm axis} - \log m$ |
|---|---|---|---|
| Deep Sets axhard s0 / s1 | 0.920 [0.917, 0.923] / 0.938 [0.935, 0.940] | 0.922 / 0.941 | 0.711 [0.697, 0.724] / 0.819 [0.810, 0.828] |
| Linformer axhard s0 / s1 | 0.901 [0.896, 0.905] / 0.883 [0.878, 0.887] | 0.895 / 0.892 | 0.747 [0.737, 0.756] / 0.691 [0.677, 0.705] |
| Deep Sets stored s0 / s1 | 0.950 / 0.919 | 0.950 / 0.919 | 0.037 / 0.052 |
| Linformer stored s0 / s1 | 0.908 / 0.915 | 0.908 / 0.915 | 0.117 / 0.090 |

$R^2(\log m) - R^2(\log\sqrt{M_0M_2})$ on axhard models: −0.002, −0.004, +0.006, −0.010 (CIs exclude 0). Both targets are
decodable at nearly equal $R^2$; the axhard last layers also encode their difference $\rho$ ($R^2$ 0.69–0.82, stored-axis
models 0.04–0.12).

## Test 2: axis jump (decisive test)
**Sample.** W/Z jets of the val archive outside the seed-71 sample; exclusive kT clustering to two subjets (E-scheme, no
beam distance), $z_A$ = pT fraction of the subjet holding the hardest particle, $0.2 < z_A < 0.8$ (66 % of W/Z jets);
$j$ = hardest particle of the other subjet; $p_{T,j} \to 1.005\,p_{T,\rm hardest}$, kept when this raise is ≤ 5 % of
$p_{T,j}$ (3.3 % of two-prong jets). N = 2,092 (1,075 W, 1,017 Z); raise median 2.7 % (0.5–5 %), ΔpT median 3.6 GeV;
Δlog m median +0.0033 (99 % < 0.013). $z_A \ge 0.5$ in 1,095 jets, $< 0.5$ in 997. Inputs recomputed (re-sorted,
recentred about the new hardest particle).
**Predictions.** H_a: change of the log m read-out = Δlog m (≈ 0.004). H_b: = ½Δlog(M0M2) about the model's axis; on
axhard models per jet −0.69…+0.69, mean +0.197 (hardest in harder prong) / −0.186 (softer); idealised ½log(z_old/z_new)
correlates with it. Matched calibration: on the unedited jet the offsets about the model's axis scaled by
$k = e^{\rm prediction}$ per jet (a real change of the predicted size), giving the read-out's own response.
**Null.** A random other particle of the same subjet raised by the same ΔpT (axis unchanged; Δlog m mean 0.0077).
**Control.** The same jets through the stored-axis models (their axis does not move; both predictions ≈ 0.004).

Per-jet slope of the log m read-out change on ½Δlog(M0M2) about the hardest particle (bootstrap 2,000 over jets):

| model | observed slope [CI] | slope of matched real change | obs / real [CI] | p(H0a: slope = real-change slope under H_a, 0.003–0.010) | p(H0b) |
|---|---|---|---|---|---|
| Deep Sets axhard s0 | +0.20 [0.18, 0.22] | 1.42 | 0.14 [0.13, 0.16] | 2.5e-63 | < 1e-300 |
| Deep Sets axhard s1 | −0.17 [−0.18, −0.16] | 1.36 | −0.13 [−0.14, −0.11] | 1.1e-145 | < 1e-300 |
| Linformer axhard s0 | −0.38 [−0.41, −0.35] | 0.93 | −0.41 [−0.44, −0.37] | 3.1e-165 | < 1e-300 |
| Linformer axhard s1 | +0.15 [0.13, 0.18] | 1.13 | 0.14 [0.12, 0.16] | 8.8e-28 | < 1e-300 |
| stored-axis models | +0.006, +0.006 (DS); +0.038, +0.039 (Linf) | ≈ 0.005 | — | — | — |

Read-out fitted to $\log\sqrt{M_0M_2}$ about the axis, same edit: slopes 0.92, 0.40, 0.32, 0.74 (axhard), 0.006–0.039
(stored). Null edit slopes 0.002–0.012. Mean change by subset (axhard, harder / softer prong): DS s0 +0.044 / −0.037,
DS s1 −0.032 / +0.036, Linf s0 −0.072 / +0.078, Linf s1 +0.043 / −0.033, against H_b +0.197 / −0.186 (matched real
change +0.10…+0.26 / −0.26…−0.32) and H_a +0.005 / +0.003.

**Calls.** Calls changed: jump 6.0–9.4 %, null 1.8–1.9 %, matched real change 58–64 % (axhard); stored 0.4–1.2 % vs
null 0.9–1.0 %. W jets with the hardest particle in the harder prong (n = 552), W → Z nat / jump / null / matched real
change: DS s0 2.0 / 8.2 / 3.6 / 44.4 % (McNemar jump vs null p = 6.2e-4), DS s1 1.6 / 7.4 / 3.6 / 44.6 % (1.9e-3),
Linf s0 0.4 / 3.8 / 1.4 / 39.5 % (0.011), Linf s1 0.9 / 5.6 / 1.6 / 41.5 % (2.0e-4); jump vs matched real change
p < 1e-38 in all four. Softer prong (n = 523): W → Z drops, DS s0 9.0 → 3.6 % (null 11.5 %, p = 2.7e-8), DS s1 5.4 → 4.8
(7.1, p = 0.058), Linf s0 3.6 → 1.5 (5.9, p = 1.2e-4), Linf s1 6.1 → 2.1 (7.6, p = 1.1e-6). Z jets, softer prong, Z → W
under the matched real change 49–66 %, under the jump 3.6–4.4 % (p < 1e-58). Stored-axis models: W → Z 0.4–1.4 % →
0.9–1.6 % (jump), McNemar jump vs null p = 0.03–1.0.

## Test 3: rigid shift
For the axhard models a rigid shift of all particles followed by recentring about the hardest particle leaves the inputs
unchanged: measured max |Δinput| 3e-8 (Δη + 0.03) and 6e-8 (+ 0.06) rad, 0 calls changed, max |Δlogit| < 8e-6. The test
is therefore uninformative for them; test 2 replaces it. Stored-axis models at Δη + 0.06: 54–62 % of calls change,
log m read-out +0.19…+0.27 against ½Δlog M2 = 0.306 (same behaviour as X2).

## Test 4: W → Z vs $z_A$ on natural jets
All 52,000 W (and 52,000 Z) test jets, kT subjets. W → Z rate by $z_A$ bin:

| $z_A$ | n | DS axhard s0 / s1 | Linf axhard s0 / s1 | DS stored s0 / s1 | Linf stored s0 / s1 |
|---|---|---|---|---|---|
| < 0.2 | 50 | 56 / 58 % | 22 / 24 % | 2 / 0 % | 4 / 12 % |
| 0.2–0.3 | 533 | 19.1 / 12.2 | 5.4 / 7.5 | 2.3 / 1.3 | 2.3 / 3.0 |
| 0.3–0.4 | 1,999 | 5.5 / 3.7 | 3.3 / 4.1 | 1.2 / 1.4 | 0.9 / 1.6 |
| 0.5–0.6 | 6,970 | 2.0 / 2.2 | 1.2 / 1.1 | 1.1 / 1.4 | 1.2 / 1.2 |
| 0.9–1.0 | 6,636 | 5.6 / 4.2 | 7.6 / 7.8 | 4.8 / 4.8 | 7.4 / 8.4 |

$z_A < 0.3$ vs $\ge 0.7$: axhard DS 22.3 / 3.5 and 16.1 / 2.9 %, Linf 6.9 / 3.9 and 8.9 / 3.9 %; stored DS 2.2 / 2.8 and
1.2 / 2.8 %, Linf 2.4 / 3.5 and 3.8 / 4.2 %. All models rise at $z_A > 0.8$ (one-prong-like W jets), so the linear
Cochran–Armitage trend is positive for stored models (p < 1e-41) and mixed for axhard (DS: p = 0.095, 0.042; Linf
p < 1e-41). Logistic slope in $z_A$, axhard − stored (paired bootstrap, 300): DS −2.97 [−3.37, −2.56], −3.09
[−3.53, −2.72]; Linf −1.60 [−2.08, −1.15], −1.91 [−2.35, −1.44] (p < 0.0067 each). Z → W shows no excess at small $z_A$
for any model.

## Answer (scoped to these measurements)
The last hidden layers of the hardest-particle-axis models carry both $\log m$ and $\log\sqrt{M_0M_2}$ about the hardest
particle ($R^2$ 0.88–0.94 each) and their difference ($R^2$ 0.69–0.82). Under the axis jump, which changes
$\sqrt{M_0M_2}$ about the axis by ±0.2 in log and $m$ by 0.003, the log m read-out moves by 0.13–0.41 of the matched real
change, with a sign that differs between seeds; both H_a (exact invariance) and H_b (full axis-moment response) are
rejected for all four models (p ≤ 1e-27). The decisions move in the H_b direction (W → Z up when the axis moves to the
softer prong, down when it moves to the harder prong; McNemar vs null p = 3e-8…0.06) at 6–9 % of calls changed against
58–64 % for a real change of the same size. On natural jets, W jets whose hardest particle sits in a prong with
$z_A < 0.3$ are called Z at 7–22 % against 1–4 % for the stored-axis models. Stored-axis controls respond to the jump
at the null level. Caveats: edit set limited to 2,092 jets with near-equal leading particles in the two prongs; C/A
exclusive clustering without a beam distance agrees with kT on 23 % of these jets (soft wide-angle particles end as a
separate subjet), so the C/A subset results are not used.
