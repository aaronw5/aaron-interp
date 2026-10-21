# H67 — Mass reader or width + pT reader (loop D, claims D25 / D9; links A4, D4m)

Script `interp/h67_mass_vs_width.py`. Jobs: `d-h67-ed-hls`, `d-h67-ed-jc`, `d-h67-ed-top` (edits, CPU), `d-h67-ev-jedi`, `d-h67-ev-small` (CPU), `d-h67-ev-gpu` (JetClass + Top networks), `d-h67-sum` (references, statistics, figures).
Results: `results/physics/h67_summary.json`, `h67_edits_{hls,jetclass,top}.json`, figures `h67_jet.png`, `h67_phi.png`. Per-jet tables, logits and features are in `/j-jepa-vol/aaron-interp/results/physics/h67_work/` (not copied back).

## Conclusion

We edited jets in two ways. One edit narrows or widens the jet's energy-weighted angular spread w while holding the invariant mass m and the jet pT exactly fixed. The other changes m by 10 % while holding w and pT fixed. On JetClass and Top Landscape, every tagger (ParT, ParticleNet, LorentzNet, PFN, EFN) responds to both edits. The share of its response that goes to the width edit is φ ≈ 0.33–0.47 (ParT_full on W jets: 0.35, 95 % CI 0.34–0.36). So none of these taggers reads mass alone (φ = 0) or width plus pT alone (φ = 1). The hls4ml networks, JEDI-linear included, respond to the width edit on W and Z jets hardly more than to a control that leaves both quantities fixed. There they behave close to mass readers, but on top jets they too are mixed (φ ≈ 0.43). One caveat matters for interpretation: at fixed mass, the width edit works by changing how the jet's pT is shared between its prongs. It therefore also moves D2 (by about +0.3) and τ21 (by about +0.03). The measurement cannot say whether the networks read w itself or the prong sharing, D2 or τ21. A gradient-boosted classifier on energy-flow polynomials (GBDT(EFP)) has φ values closest to the networks'.

## Question and hypotheses

H64v could not separate two readings of the boost response. A boost along the jet axis moves w and pT together at fixed m, and an MLP on (w, pT) without mass reproduced ParT_full's boost ratio. H67 builds edits that move one of (m, w) at a time with the jet four-momentum otherwise fixed.

- H_m (pure invariant-mass reader, as MLP(m, pT)): no response to the width edit E_w, so φ = 0.
- H_w (pure width + pT reader, as MLP(w, pT)): no response to the mass edit E_m, so φ = 1.
- H_mix (reads both, as MLP(m, w, pT)): 0 < φ < 1.
- JEDI-linear (A4: it reads m_A = √(M0·M2) about the stored axis) was predicted to follow MLP(m_A, pT).

## Definitions

- **m**: invariant mass of the summed constituent four-vectors. **pT**: jet transverse momentum |Σ p⃗_T|.
- **w** = Σ_i pT_i ΔR_i / Σ_i pT_i, where ΔR_i is particle i's (η, φ) distance to the jet's E-scheme axis. The **E-scheme axis** is the direction of the summed constituent three-momentum.
- **m_A** = √(M0·M2) with M0 = Σ pT_i and M2 = Σ pT_i ΔR_i² about the stored axis (JEDI-linear's mass variable, A4).
- **Prongs**: exclusive-kT subjets (p = 1, E-scheme recombination): 3 for jets of the top class, 2 for every other jet. **z** is the pT share of a prong.
- **D2** (β = 1) = e3/e2³, from energy correlation functions. **τ21** = τ2/τ1 (N-subjettiness, β = 1, kT winner-take-all axes). Both are larger for one-prong, QCD-like jets.
- **Δs**: change of the own-class one-vs-rest log-odds, l_k = log p_k − log(1 − p_k), from the unedited jet to the edited jet, for jets of true class k.
- **φ** = Σ|Δs(E_w)| / (Σ|Δs(E_w)| + Σ|Δs(E_m)|) over the jets of one class, at the same 10 % fractional size. It is 0 when the network responds only to the mass edit and 1 when it responds only to the width edit. **ρ** = Σ|Δs(E_w)| / Σ|Δs(E_m)| = φ/(1 − φ) is also stored in the JSON.
- **φ_excess**: the same ratio after subtracting the crot response from each sum (crot is defined below).
- **McNemar**: paired test on calls flipped (at the 50 %-efficiency working point of the class) by an edit versus by crot. **Holm**: step-down family-wise correction.

## Procedure

**Samples.**
- hls4ml: `jedi_core.sample(2000, 6701)`, i.e. 2,000 jets per class × 5 classes; 3,000 disjoint jets per class as reference training set (rng 6711).
- JetClass: 5,000 random rows per class of Wqq, Zqq, Tbqq and QCD from the first test file (rng 6702), never the first k rows. These are split at random (rng 6712) into 2,000 evaluated + 3,000 reference jets per class.
- Top Landscape test: 5,000 top + 5,000 QCD random rows (rng 6703), plus 5,000 + 5,000 disjoint reference rows (rng 6713).

**Edit map.** The map is exact and per jet, built about the jet's own E-scheme axis, and it keeps the number of particles. It has two parameters:
1. **Prong sharing f.** Every particle of prong k has its pT (and its mass) scaled by 1 + f·(P/K − P_k)/P_k. Here P_k is the prong's pT, P is the jet's Σ pT and K is the number of prongs. Σ pT and each prong's centroid are unchanged. f > 0 makes the prongs' pT more equal; f < 0 makes them less equal.
2. **Opening angle u.** Each prong is translated rigidly so that its centroid moves from c_k to u·c_k, measured from the jet axis.
3. **Restore the jet.** All particles are translated in (η, φ) until the summed three-momentum points along the original E-scheme axis again. Then every four-vector is scaled so the jet pT equals the unedited value.

(f, u) are solved per jet by a damped 2-D Newton iteration to hit targets in (log m, log w). A jet counts as converged when the residual is below 1e-5 in log. Achieved precision on converged jets: |Δm/m| < 1e-14, |ΔpT/pT| < 1e-15, jet η and φ within 6e-8. The JetClass inputs are rebuilt with η = jet_eta + sign(jet_eta)·Δη.

Departure from the plan. The plan's lever moves pT between the core and the edge of each prong. Measured on the real jets (`lever_diag` in `h67_edits_*.json`), it changes log m and log w by almost the same amount, as the opening angle u also does. Examples at f = +0.3: hls W Δlog m +0.0094, Δlog w +0.0079; JetClass and Top similar. The two-target system is then singular. The prong sharing f instead moves log w about twice as much as log m, as expected from m ≈ pT·Δ·√(z(1−z)) and w ≈ 2z(1−z)·Δ for two prongs at separation Δ. Examples at f = −0.3: hls W Δlog m −0.087, Δlog w −0.163; JetClass Wqq −0.074, −0.144. The core/edge lever is kept only as a diagnostic and in the crand control.

**Edits.**
- **E_w ±10 %, ±20 %**: w × (1 ± 0.1 or 0.2) at fixed m and pT.
- **E_m ±10 %**: m × (1 ± 0.1) at fixed w and pT.
- E_w−10 and E_m+10 are reached by making the sharing less equal and opening the prongs (pair "uneq", u ≈ 1.15–1.34). E_w+10 and E_m−10 are reached by making it more equal and closing the prongs (pair "eq"). The "eq" pair converges on fewer jets because z(1−z) ≤ 1/4.
- Convergence (hls / JetClass / Top):

  | edit | hls | JetClass | Top |
  |---|---|---|---|
  | E_w−10 | 0.71 | 0.71 | 0.53 |
  | E_m+10 | 0.68 | 0.69 | 0.49 |
  | E_w+10 | 0.45 | 0.46 | 0.37 |
  | E_m−10 | 0.33 | 0.35 | 0.25 |

  On top-class jets: 0.34–0.38 for E_w−10 and E_m+10.

**Controls.**
- **crot**: each prong is rotated about its own centroid by a random angle, then (f, u) are solved for Δm = Δw = 0. N, m, w and pT are fixed, and D2 and τ21 change by ≤ 0.05 and ≤ 0.01. Its median displacement is larger than E_w's (JetClass 0.048 vs 0.026 rad), so it is a conservative floor.
- **crand_free**: random pT redistribution inside each prong, with the same moved pT as E_w−10 (about 7 % of Σ pT). The prong is recentred and u is solved for Δm = 0; w changes by less than 0.1 %.
- **crand_match**: the same random redistribution, then solved to Δm = 0 and Δw = −10 %.
- **TN**: translation of the whole jet relative to the stored axis, by E_w−10's displacement, in a random direction.
- Every edit's ΔD2, Δτ21, Δm_A and Δ(Σ pT) is listed in `h67_edits_*.json`.

**Models.**
- hls4ml: JEDI-linear n64 3-feature, Deep Sets (dsets-d64-n64), transformer (tf-2b-d64-n64-e100), JEDI-small (jedis-d64-n64).
- JetClass: ParT_full, ParT_kin, PN_full, PN_kin.
- Top: ParT, PN, LorentzNet, PFN, EFN.
- Nulls: 6 weight-permuted JEDI-linear copies (hls), 3 permuted ParT_full + 3 permuted ParT_kin (JetClass), 6 permuted Top ParT copies.
- References, trained on each dataset's reference jets: MLP(32, 32) on [log m, log pT], [log w, log pT], [log m, log w, log pT] and [log m_A, log pT]; HistGradientBoosting on 32 energy-flow polynomials (GBDT(EFP)).

**Statistics.**
- 1,000 bootstrap resamples over jets within each class.
- H0 "|Δs(edit)| = |Δs(crot)|" (paired normal test on per-jet differences, on the same jets).
- McNemar on call flips, edit vs crot.
- Holm over trained models × edits × classes: 414 tests in each family.

## Accuracy of the unedited models on these samples

| model | this sample | known (other sample) |
|---|---|---|
| JEDI-linear | 0.813 [0.805, 0.820] | 0.807 (H64) |
| dsets | 0.809 | 0.802 |
| tf_d64 | 0.816 | 0.810 |
| jedis | 0.798 | – |
| ParT_full (W / Z / top / QCD) | 0.78 / 0.69 / 0.96 / 0.79 | 0.79 / 0.69 / 0.96 / 0.77 |
| ParT_kin | 0.68 / 0.57 / 0.91 / 0.73 | 0.69 / 0.56 / 0.91 / 0.71 |
| PN_full | 0.77 / 0.66 / 0.94 / 0.78 | 0.77 / 0.64 / 0.95 / 0.75 |
| PN_kin | 0.66 / 0.53 / 0.91 / 0.71 | 0.67 / 0.53 / 0.91 / 0.69 |
| Top ParT / PN / LorentzNet / PFN / EFN | 0.943 / 0.944 / 0.945 / 0.931 / 0.921 | 0.937 / 0.940 / 0.943 / 0.929 / 0.923 |

All are within sampling error of the known values. The permuted nulls are at chance or constant output.

## Numbers (pair "uneq": E_w−10 vs E_m+10; mean |Δs| in log-odds; φ with 95 % CI)

| dataset / model / class | n | \|Δs\| E_w | \|Δs\| E_m | \|Δs\| crot | φ [95 % CI] | φ_excess | φ (pair "eq") |
|---|---|---|---|---|---|---|---|
| hls JEDI-linear W | 1640 | 0.69 | 5.24 | 0.60 | 0.116 [0.111, 0.121] | 0.018 | 0.254 |
| hls JEDI-linear Z | 1635 | 0.94 | 3.22 | 0.83 | 0.225 [0.217, 0.234] | 0.042 | 0.128 |
| hls JEDI-linear t | 695 | 0.97 | 1.31 | 0.58 | 0.425 [0.412, 0.439] | 0.346 | 0.420 |
| hls dsets / tf_d64 / jedis W | 1640 | 0.49 / 0.65 / 0.38 | 5.7 / 6.7 / 5.1 | 0.46 / 0.59 / 0.39 | 0.079 / 0.088 / 0.069 | 0.005 / 0.009 / 0.000 | 0.21–0.24 |
| hls dsets / tf_d64 / jedis t | 695 | 1.15 / 1.14 / 0.77 | 1.43 / 1.49 / 0.90 | 0.49 / 0.55 / 0.48 | 0.446 / 0.434 / 0.461 | 0.39–0.41 | 0.47–0.49 |
| JetClass ParT_full Wqq | 1607 | 1.62 | 3.01 | 1.10 | 0.350 [0.342, 0.357] | 0.214 | 0.335 |
| JetClass ParT_full Zqq | 1677 | 1.69 | 3.06 | 1.20 | 0.355 [0.349, 0.362] | 0.208 | 0.310 |
| JetClass ParT_full Tbqq | 637 | 3.79 | 5.32 | 1.87 | 0.416 [0.409, 0.423] | 0.358 | 0.403 |
| JetClass ParT_full QCD | 1129 | 0.94 | 1.49 | 0.90 | 0.387 [0.378, 0.395] | 0.068 | 0.339 |
| JetClass ParT_kin W / Z / top / QCD | | 1.86 / 1.45 / 3.39 / 0.89 | 3.06 / 2.55 / 4.41 / 1.39 | 1.10 / 0.88 / 1.37 / 0.93 | 0.378 / 0.363 / 0.435 / 0.390 | 0.28 / 0.26 / 0.40 / 0.00 | 0.35 / 0.32 / 0.44 / 0.36 |
| JetClass PN_full W / Z / top / QCD | | 1.31 / 1.38 / 2.85 / 0.80 | 2.70 / 2.73 / 4.05 / 1.32 | 1.07 / 1.16 / 2.35 / 0.82 | 0.327 / 0.335 / 0.413 / 0.378 | 0.13 / 0.12 / 0.23 / 0.00 | 0.32 / 0.29 / 0.37 / 0.33 |
| JetClass PN_kin W / Z / top / QCD | | 0.55 / 0.50 / 2.24 / 0.68 | 1.84 / 1.64 / 3.29 / 1.15 | 0.64 / 0.58 / 0.88 / 0.68 | 0.230 / 0.232 / 0.405 / 0.371 | 0.00 / 0.00 / 0.36 / 0.01 | 0.27 / 0.22 / 0.35 / 0.35 |
| Top ParT top / QCD | 1594 / 2689 | 2.45 / 0.62 | 3.20 / 0.85 | 1.01 / 0.70 | 0.434 / 0.420 | 0.397 / 0.000 | 0.410 / 0.425 |
| Top PN, LorentzNet, PFN, EFN (top class) | 1594 | 2.43, 2.67, 1.90, 1.60 | 3.29, 3.48, 2.17, 1.94 | 1.00, 1.09, 0.55, 0.57 | 0.425, 0.434, 0.466, 0.451 | 0.39, 0.40, 0.45, 0.43 | 0.43, 0.43, 0.47, 0.42 |
| ref MLP(m, pT) (all datasets) | | 0 | 0.26–2.8 | 0 | 0.000 | – | 0.000 |
| ref MLP(w, pT) | | 0.22–0.62 | 0 | 0 | 1.000 | – | 1.000 |
| ref MLP(m, w, pT) hls W / JetClass W / Top top | | | | | 0.071 / 0.249 / 0.338 | | 0.17 / 0.29 / 0.23 |
| ref MLP(m_A, pT) | | ≤ 0.025 | 0.18–2.8 | | 0.002–0.049 | | ≤ 0.027 |
| ref GBDT(EFP) hls W / JetClass W / JetClass top / Top top | | | | | 0.089 / 0.348 / 0.458 / 0.445 | | 0.25 / 0.35 / 0.46 / 0.45 |
| null permuted ParT_full (JetClass, perm1) W / Z / top / QCD | | 0.28 / 0.38 / 0.25 / 0.57 | 0.28 / 0.36 / 0.24 / 0.60 | | 0.49–0.52 | | 0.47–0.50 |

**Signed responses.** The mean Δs under E_w−10 (more unequal prong sharing at fixed m) is negative for every two- and three-prong signal class, i.e. the edited jet is called less like its true class:

| model / class | mean Δs under E_w−10 |
|---|---|
| ParT_full Wqq | −1.44 |
| ParT_full Tbqq | −3.55 |
| Top ParT top | −2.27 |
| JEDI-linear W | −0.32 |
| JEDI-linear t | −0.67 |

On QCD-class jets the sign is positive and |Δs| is close to the crot floor.

**Edit side effects at fixed m and pT.** E_w−10 raises D2 by +0.28 to +0.57 (median per class) and τ21 by +0.03 on two-prong jets (≈ 0 on top). E_m+10 raises D2 by +0.2 to +0.3 and leaves τ21 within ±0.016. E_w leaves m_A within 0.1 %; E_m moves m_A by +10 to +10.3 %. TN moves m_A by +0.7 to +1.8 % and leaves m, w, D2 and τ21 fixed. TN's mean |Δs|:

| model | TN mean \|Δs\| |
|---|---|
| hls JEDI-linear W | 1.07 |
| hls tf_d64 W | 1.80 |
| ParT_full W | 0.34 |
| Top ParT top | 0.07 |

**Tests.**
- |Δs(edit)| vs |Δs(crot)|: 348 of 414 tests have Holm p < 0.05. Per-class p values are in the JSON (`p_Ew_eq_crot`, `p_Em_eq_crot`, `holm.abs_vs_crot`).
- McNemar on call flips, edit vs crot: 261 of 414 have Holm p < 0.05.
- Tests that do not reject H0 for E_w (Holm-uncorrected p > 0.1): hls dsets, tf_d64 (Z only) and jedis on W and Z; JetClass ParT_full, ParT_kin and PN_full on QCD; PN_kin on QCD; Top PFN on QCD.

**Reference assignment** (references whose φ lies inside the network's 95 % CI). No network contains MLP(m, pT) (φ = 0), MLP(w, pT) (φ = 1) or MLP(m_A, pT) in any class or pair. GBDT(EFP) is inside the CI for:
- JetClass ParT_full Wqq (uneq), ParT_kin Zqq (uneq) and Wqq (eq);
- ParT_full and ParT_kin QCD;
- PN_kin QCD (eq);
- hls JEDI-linear q (uneq) and W (eq), dsets q (uneq) and W (eq), tf_d64 W (uneq), jedis W (eq).

MLP(m, w, pT) is inside for:
- JetClass QCD (ParT_full, ParT_kin, PN_full);
- hls top (eq) for dsets, tf_d64 and jedis;
- hls jedis W (uneq);
- Top EFN QCD (uneq).

All other network × class × pair cells lie inside no reference interval.

## Reading

1. **H_m (pure mass reader).** It is excluded for every JetClass and Top network: φ CIs lie between 0.22 and 0.47, and E_w responses exceed the crot floor for the signal classes (Holm p < 0.05). The hls4ml networks on W and Z have φ = 0.07–0.23, but their E_w response is within 0–4 % of the crot floor (φ_excess ≤ 0.04). On these classes their response to the width edit is indistinguishable from their response to an edit that keeps m, w, pT, D2 and τ21 fixed.
2. **H_w (pure width + pT reader).** It is excluded for every network: all φ ≤ 0.49, and E_m responses are larger than E_w responses in every class.
3. **Where the networks sit.** JetClass and Top networks have φ = 0.33–0.47 on signal classes, and φ_excess 0.13–0.45 on W, Z and top (0 on QCD). This range overlaps GBDT(EFP) (JetClass 0.35–0.46, Top 0.45) more than MLP(m, w, pT) (0.22–0.34). The permuted nulls give φ ≈ 0.5 with absolute responses 5–20 times smaller than the trained ParT's on signal classes.
4. **What E_w changes.** At fixed m and pT, E_w changes the prong pT sharing. For two prongs, w ∝ √(z(1−z)) at fixed m and pT, so w and z are not separable by any edit that holds m and pT fixed and keeps two point-like prongs. E_w therefore also changes D2 (+0.3 to +0.6) and τ21 (+0.03). The response to E_w is a response to "w or prong sharing or D2 / τ21", not to w alone. The sign (more unequal sharing → less W/Z/top-like) agrees with the physics of signal decays (more symmetric) versus QCD splittings (more asymmetric).
5. **JEDI-linear and A4.** The prediction "both edits move m_A" does not hold. E_w leaves m_A unchanged within 0.1 % (about the E-scheme axis, m_A ≈ m for these jets), so MLP(m_A, pT) has φ ≈ 0.002–0.007. JEDI-linear's W and Z response to E_w sits at the crot floor, consistent with a mass-type reader on W and Z. Its top-class φ of 0.43 (φ_excess 0.35) is not explained by m_A. JEDI-linear's TN response on W (1.07 log-odds at 0.013 rad displacement) is larger than its E_w response (0.69), consistent with D4m (axis-relative inputs).
6. **Caveats.**
   - The pair "eq" (more equal sharing) converges on fewer jets: 25–46 % overall and 22–38 % on top jets. Those jets are a selected subset with unequal sharing.
   - crot moves particles about twice as far as E_w, so φ_excess underestimates the width response.
   - The top class uses 3 prongs; its E_w acts through the sharing among three subjets.
   - Diverged solves (non-converged jets) are excluded from every comparison; 1–4 JetClass jets per edit had non-finite edited inputs and were replaced by the unedited jet before evaluation.

## Effect on claims

- **D25**: refines. The JetClass and Top taggers' response at fixed pT is not mass-only. For W, Z and top they respond to a width / prong-sharing change at fixed mass with 0.5–0.8 times the strength of a 10 % mass change (ρ = φ/(1 − φ)). The H64v ambiguity is resolved against "pure (w, pT)". It is not resolved between w, z, D2 and τ21.
- **D9**: hls4ml networks on W and Z behave as mass readers under this edit; on top they do not.
- **A4**: JEDI-linear on W and Z is consistent with a mass-type reader. The m_A-specific prediction is not tested by these edits, because both m and m_A move together here.

## Proposed next steps

1. Separate prong sharing from D2 and τ21 at fixed m and w: change in-prong radiation (D2 changes through e3) at fixed z. Reference MLPs on (m, z, pT) and (m, D2, pT).
2. An m-vs-m_A edit: translate the stored axis at fixed jet four-momentum, using JetClass's stored jet_eta.
3. A displacement-matched crot (rotation angle matched to E_w's displacement).

## Explainer entry (for site/explain.py; not written to site/ by this worker)

**Question.** Does a jet tagger read the jet's invariant mass, or its angular width together with its pT?

**Physics background.** For a jet of pT and invariant mass m, the angular size scales like m/pT. A two-prong decay with prong momentum fractions z and 1 − z and opening angle Δ has m² ≈ z(1−z)·pT²·Δ² and width w ≈ 2z(1−z)·Δ. So at fixed m and pT, w can still change if z changes:

$$ w = \frac{\sum_i p_{T,i}\,\Delta R_i}{\sum_i p_{T,i}}, \qquad m \simeq p_T\,\Delta\sqrt{z(1-z)}, \qquad w \simeq 2z(1-z)\,\Delta \;\Rightarrow\; \frac{w}{m/p_T} \simeq 2\sqrt{z(1-z)}. $$

**Method.** Each jet is split into 2 (or 3, for top) kT subjets. Two exact knobs are applied:
- move pT between the prongs, with each prong's position fixed;
- scale the prong-prong opening angle.

The jet is then re-centred on its E-scheme axis and rescaled to its original pT. A Newton solve finds, per jet, the knob settings giving (i) w ±10 % at fixed m and pT, or (ii) m ±10 % at fixed w and pT. The tagger's own-class log-odds change Δs is measured, and

$$ \varphi = \frac{\sum |\Delta s(E_w)|}{\sum |\Delta s(E_w)| + \sum |\Delta s(E_m)|} $$

is 0 for a mass reader and 1 for a width + pT reader.

**Result (figure `h67_phi.png`; the edit on a real W jet is shown in `h67_jet.png`).** JetClass and Top taggers have φ ≈ 0.33–0.47, close to a boosted-decision-tree on energy-flow polynomials. hls4ml networks on W and Z jets have φ ≈ 0.07–0.23, with a width response no larger than the response to rotating each prong.

**Checks.** m, pT, η and φ of the jet are held to < 1e-7. Reference MLPs give exactly φ = 0 (mass + pT) and φ = 1 (width + pT). Permuted networks give φ ≈ 0.5 with small responses. The prong-rotation control (N, m, w, pT fixed) sets the floor.

**Physical significance.** These taggers use more than the jet mass: at fixed mass they also respond to how the jet's energy is shared between its prongs. That quantity is tied to the width at fixed mass, and also to D2 and τ21.

**Glossary additions.** w (pT-weighted angular width), φ (width-response share), ρ, φ_excess, prong sharing f, crot, crand, E_w, E_m.
