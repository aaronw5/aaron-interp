# H1 — IRC-safety panel: collinear split and soft emission across nine taggers (claim D5)

## Conclusion
Every trained tagger we tested (JEDI-linear, Deep Sets, Linformer and a transformer on hls4ml; ParT and ParticleNet on JetClass) changes some of its calls when one particle is replaced by two exactly collinear particles with the same total momentum. The changes grow with the number of particles split, are the same at zero and at finite opening angle, and move hls4ml jets toward the gluon class. Splitting the hardest particle into 5 % and 95 % pieces changes 1.1–5.7 % of calls, against 0–0.25 % for an energy-flow network (EFN) that is infrared- and collinear-safe (IRC-safe: its output cannot depend on how momentum is shared between collinear particles or on vanishingly soft ones). All of the EFN's changes come from jets where the extra particle pushes another out of the 64-particle input window. Adding one particle with 10⁻⁴ of the jet pT changes 0.7–7.4 % of calls for the trained taggers, against 0.5 % for the EFN, whose response falls about ten-fold per decade of particle pT. The taggers therefore react to the number of particles, which the IRC-safe quantities they are often described with (mass, τ21, D2) do not depend on. Two points are still open. For ParT and ParticleNet the half-and-half split is confounded, because the matched control (moving half the hardest particle's pT to its neighbour) changes calls even more often. ParT_full's and ParticleNet_full's soft response falls only two-fold between 10⁻³ and 10⁻⁴, so these runs do not show whether it levels off at a constant rate.

## Question and hypotheses
Do the taggers use the number of particles (IRC-unsafe) or only IRC-safe shapes?
- H0: % calls changed under the split equals % changed under a count-fixed control of matched size (C1: reshuffle; for soft emission, the recoil alone). H1: split > control.
- Predictions. Counting model: the change does not depend on ΔR down to 0, grows with the number of splits, and flows toward classes with many particles (g; in JetClass, QCD and Hgg). IRC-safe-shape model: zero change at ΔR = 0 (as for the EFN), and a soft response that falls in proportion to ε. Generic input sensitivity: a weight-scrambled network responds like the trained one.

## Procedure
Script `interp/h1_irc_panel.py` has four parts, all run as cluster jobs: `d-h1-jedi3` (jax), `d-h1-small` (conda, which also trains the EFN), `d-h1-jc5` (GPU) and `d-h1-sum2` (Holm correction and figure).
- Samples. hls4ml: the jets of `jedi_core.sample(per_class=2000, seed=401)`, 10,000 jets, 64 slots (22.6 % of jets fill all 64). JetClass: 1,000 random rows per class (seed 402) from each class's first test file, 10,000 jets, 128 slots. The jets are built with η = jet_η + sign(jet_η)·Δη; rebuilding the table changes the four-vectors by at most 3e-5.
- Edits, applied on the real particles:
  - Split: the particle of pT rank 0, 1 or 4, or a random particle, becomes two daughters, p₁ = z p + k û and p₂ = (1−z) p − k û. Here û ⊥ p at a random azimuth and k = ΔR·z(1−z)·pT. The jet 3-momentum is kept to 1e-15 relative; at ΔR = 0 the four-momentum is kept exactly. z ∈ {0.5, 0.2, 0.05} and ΔR ∈ {0, 1e-3, 1e-2}. Both daughters keep the parent's PID, charge and impact parameters.
  - Multi-split: the k ∈ {2, 4, 8} hardest particles are each split (z 0.5, ΔR 0).
  - Soft emission: one particle with pT = ε·pT_jet, ε ∈ {1e-4, 1e-3, 1e-2}, at a uniform point of the disc ΔR < 0.8, with its 3-momentum taken from the hardest particle. In JetClass it is a massless neutral hadron.
  - Truncation: after each edit the particles are re-sorted and cut to the slot limit. Overflow is recorded, and every rate is also given on the jets without overflow.
- Controls:
  - C1 reshuffle: the particle of rank r gives the fraction f = z of its 3-momentum to its nearest real neighbour. The neighbour is within ΔR < 1e-2 for 95 % of hls4ml jets and 49 % of JetClass jets (rank 0); results are reported on all jets and on that subset.
  - Recoil alone, with no particle added.
  - C2: merge of the closest pair.
  - C3: an EFN, L = Σᵢ zᵢ Φ(Δηᵢ, Δφᵢ), with Φ a 2-128-64 ReLU network and F 64-128-128-5, trained on the hls4ml training archive minus the dev split for 12 epochs. Test accuracy on the 260k-jet val archive is 72.06 %.
  - C4: weight-scrambled JEDI (redrawn), Deep Sets (entries permuted) and ParT_full (redrawn).
- Metrics: % calls changed with a 95 % bootstrap interval over jets (1,000 resamples); paired McNemar test of edit against control; Holm correction over all 521 edit-vs-control tests; class flow matrix and net flow; Δlog-softmax per class; relative logit response; signal efficiency and background rejection at the 50 %-efficiency threshold frozen on unedited jets.
- Classical observables, computed on 1,000 jets (rows [::10]): N; pT^D = √Σpᵢ²/Σpᵢ; m; m_SD (C/A on all particles, z_cut 0.1, β 0); τ21 and τ32 (exclusive-kT axes); D2 (β 1, all particles).

## Unedited accuracy on the samples (known values in brackets)
- hls4ml: JEDI-linear 80.51 % [79.68, 81.34] (80.92 % on the 260k test set); Deep Sets 80.41 % (80.64 %); Linformer 81.13 % (81.07 %); transformer tf-2b-d64-e100 81.22 % (81.44 %); EFN 71.61 % (72.06 %); scrambled networks 20.0 % (one constant call).
- JetClass: ParT_full 85.59 % [84.87, 86.29] (85.9 % in D4 on rows 8000–9000; 86.1 % published); ParT_kin 73.63 % (74.5 % in D4); ParticleNet_full 83.90 % (84.4 % published); ParticleNet_kin 69.52 % (no reference value); scrambled ParT 10.05 %.
- Every model is within 1 point of its reference value.

## Numbers (% calls changed [95 % CI]; control; edit − control [CI], Holm p)
**Split of the hardest particle, z 0.05, ΔR 0 (edit / C1 f 0.05):**

| model | edit | control | edit − control | Holm p |
|---|---|---|---|---|
| JEDI | 2.60 [2.31, 2.93] | 1.22 | +1.38 [+1.10, +1.67] | 1e-16 |
| Deep Sets | 1.06 | 0.43 | +0.63 | 5e-9 |
| Linformer | 1.91 | 0.74 | +1.17 | 1e-15 |
| Transformer | 2.53 | 2.15 | +0.38 [+0.10, +0.65] | 0.69 |
| EFN | 0.25 (0.00 without overflow) | 0.00 | +0.25 | 1e-5 |
| ParT_full | 2.11 | 0.72 | +1.39 [+1.10, +1.69] | 6e-18 |
| ParT_kin | 5.02 | 5.43 | −0.41 [−0.94, +0.08] | 1.0 |
| PN_full | 3.12 | 0.97 | +2.15 | 1e-30 |
| PN_kin | 5.68 | 2.95 | +2.73 | 5e-29 |

- Restricting to the jets whose nearest neighbour is within ΔR < 1e-2, ParT_kin gives 5.29 vs 2.45 (McNemar p 2e-22) and the transformer gives 2.57 vs 2.18 (p 5e-3).
- **Split of the hardest particle, z 0.5, ΔR 0.** JEDI 3.98 vs 2.14; Deep Sets 2.32 vs 1.12; Linformer 3.76 vs 2.04; transformer 8.61 vs 5.25 (all Holm p ≤ 1e-18). ParT_full 3.43 vs 5.82, ParT_kin 8.98 vs 13.15 and PN_full 3.36 vs 4.77: the control is larger in these three. PN_kin 10.04 vs 8.14.
- **Splits at all ranks, z and ΔR, Holm < 0.05 with split > control:** JEDI, Deep Sets, Linformer and PN_kin 36/36; PN_full 31; ParT_full 29; transformer 21; ParT_kin 19. Split < control: ParT_full 3, ParT_kin 3, PN_full 2, all at z = 0.5. The EFN's 28 significant tests come only from overflow jets.
- **ΔR dependence (hardest particle, z 0.5; ΔR 0 / 1e-3 / 1e-2):**
  - JEDI 3.98 / 3.98 / 4.02 and Deep Sets 2.32 / 2.32 / 2.31: no dependence on ΔR.
  - ParT_full 3.43 / 3.94 / 4.60: rises with ΔR.
  - PN_full 3.36 / 3.40 / 3.90.
- **Number of splits (k = 1, 2, 4, 8):**
  - JEDI 3.98 / 5.55 / 7.80 / 11.62; Deep Sets 2.32 / 3.64 / 5.82 / 10.60; Linformer 3.76 / 4.97 / 7.55 / 13.33; transformer 8.61 / 10.77 / 13.09 / 15.37.
  - ParT_full 3.43 / 4.92 / 8.75 / 17.85; ParT_kin 8.98 / 13.73 / 19.27 / 26.44; PN_full 3.36 / 5.33 / 9.78 / 21.55; PN_kin 10.04 / 13.51 / 17.74 / 24.43.
  - EFN 0.26 / 0.41 / 0.64 / 1.46, and 0.00 at every k on jets without overflow.
- **Class flow, k = 8:**
  - hls4ml net flow into g: JEDI +814, Deep Sets +921, Linformer +1063, transformer +680, EFN −47. The q→g against g→q McNemar counts are 446/17 for JEDI and 483/0 for Deep Sets.
  - ParT_full: Wqq −466, QCD +273, Tbqq +117.
  - PN_full: Wqq −517, H4q −346, Hbb +281, Hgg +228, QCD +205, Tbqq +223.
  - Kinematic-only models: Hbb +1107 (ParT_kin) and +667 (PN_kin), Hcc −427 and −293.
- **Working point (k = 8, threshold frozen at 50 % signal efficiency on unedited jets).** Signal efficiency falls to 0.37–0.38 (JEDI W, Z, t) and 0.25–0.33 (ParT_full and PN_full for Wqq, Zqq, Tbqq). Only 0–25 background jets pass the threshold, so the rejection values are not resolved.
- **Soft emission (ε = 1e-4 / 1e-3 / 1e-2; recoil-only control at ε = 1e-4):**
  - JEDI 2.53 / 3.62 / 31.93 (0.13); Deep Sets 7.38 / 9.01 / 39.65 (0.02); Linformer 1.98 / 2.39 / 16.67 (0.02); transformer 2.04 / 2.81 / 18.63 (0.04).
  - EFN 0.50 / 7.01 / 50.90 (0.04).
  - ParT_full 0.66 / 1.34 / 6.75 (0.03); ParT_kin 3.32 / 3.68 / 12.90 (1.41); PN_full 0.72 / 1.31 / 6.78 (0.10); PN_kin 2.03 / 2.72 / 10.68 (0.17).
  - Ratio of the 1e-3 rate to the 1e-4 rate: EFN 14.0; Deep Sets 1.2, JEDI 1.4, Linformer 1.2, transformer 1.4, ParT_kin 1.1, PN_kin 1.3; ParT_full 2.0, PN_full 1.8.
  - At ε = 1e-2 the edit is not soft: the jet mass rises by +21.5 GeV on average (hls4ml) and +6.7 GeV (JetClass), and W→Z flows dominate.
- **Charged split, ParT_full (subset where the hardest particle is a charged hadron):** daughter 2 relabelled neutral, 5.26 %, against both daughters charged, 5.63 % (McNemar p 0.32). PN_full: 3.95 against 3.85.
- **Merge of the closest pair (C2):** JEDI 4.35, Deep Sets 1.82, Linformer 3.91, transformer 2.73, EFN 0.00, ParT_full 4.00, ParT_kin 7.48, PN_full 2.97, PN_kin 3.95. The pair is within ΔR < 1e-2 in 96–100 % of jets.
- **Weight-scrambled networks (C4).** Their calls are constant (JEDI, Deep Sets), or nearly constant for ParT (0.2–0.5 % changed under every edit, controls included), so call rates cannot separate the hypotheses. Relative logit response, split (z 0.5) against reshuffle: scrambled JEDI 0.209 against 0.046, trained JEDI 0.108 against 0.057; scrambled Deep Sets 0.076 against 0.033.
- **Classical observables, split ΔR 0:**
  - JetClass: m, m_SD, τ21, τ32 and D2 change by ≤ 1e-13 relative; N rises by +1.
  - pT^D: −0.052 at z 0.5, with 17.7 % of jets crossing the sample median; −0.009 at z 0.05, 2.1 % crossing. N crosses its median in 1.8 % of jets.
  - hls4ml: the same, except that the 22.6 % of jets that overflow lose their 64th particle, so m changes by −0.09 GeV on average.
  - At ΔR = 1e-2, τ32 crosses its median in 2.7 % of JetClass jets and 9.8 % of hls4ml jets. The other IRC-safe observables cross in ≤ 0.4 %.

## Reading
- Splitting a particle into exactly collinear pieces leaves m, m_SD, τ21, τ32 and D2 unchanged to 1e-13 and leaves the EFN's calls unchanged on every jet without overflow. On the same jets it changes 1–9 % of the calls of every trained JEDI, Deep Sets, Linformer, transformer, ParT and ParticleNet model.
- H0 (split = count-fixed control) is rejected after Holm correction in all eight trained models for at least 19 of 36 split settings.
- The counting predictions hold for the hls4ml models: no ΔR dependence, monotonic growth with the number of splits, and flow into g. For JetClass the flow goes out of Wqq and into QCD, Hgg and Hbb (full) or Hbb (kin).
- ParT_full's response rises with ΔR (3.4 → 4.6 %), so it also reads the pair angle of the split. The plan's guess that its pair features damp the collinear response is not supported: its rate at z 0.05 (2.1 %) is in the range of the other models.
- At z = 0.5 the reshuffle control changes ParT's and PN_full's calls more often than the split does. Both edits halve the hardest particle's pT; the reshuffle also changes which particle, with which PID, becomes the hardest. So at z 0.5 these models' count response cannot be separated from their response to the leading particle. At z 0.05 it can.
- In the soft limit the EFN's rate falls 14-fold per decade of ε, as an IRC-safe model's should. The rates of the hls4ml models and of the kinematic-only JetClass models change by at most 1.4-fold, so a nonzero rate remains as ε → 0. ParT_full and PN_full fall 2-fold, a case between the two.
- The generic-sensitivity prediction (C4) cannot be tested from call rates, because the scrambled networks give constant calls. Their relative logit responses show that the sum-pooling architecture alone already responds to the count.

## Issues and limits
- `soft_eps*_fix` (add the soft particle, then remove the softest) removes the added particle itself whenever it is the softest, which is nearly always at ε ≤ 1e-3. That condition then equals the recoil-only edit and is not used.
- In hls4ml, 22.6 % of jets fill all 64 slots, so a split pushes one particle out. Rates without overflow are in the JSON (for example, JEDI split z 0.5: 3.84 % without overflow against 3.98 % on all jets).
- The EFN is 9 points less accurate than JEDI (72.1 % against 80.9 %), and at ε = 1e-3 it responds more often than most of the taggers (7.0 %). IRC safety only guarantees a vanishing response, not a small one at finite ε.
- Rejection at the 50 % working point is not resolved with 2,000 (hls4ml) or 1,000 (JetClass) background jets.
- The classical-observable crossing rates use a median cut; they are not each observable's published working point.
- Debug fixes during the run: a module-name clash (`compute`); out-of-memory errors from oversized D2 chunks; a ParticleNet kNN error when a batch had fewer than 17 particle slots (P is now ≥ 17, which matches weaver's padding to 128 because padded points are masked).

## Files
- Script: `interp/h1_irc_panel.py`.
- Results: `results/physics/h1_{jedi,small,jetclass,summary}.json` and `h1_*_calls.npz` (per-jet calls for every model and condition).
- EFN: `results/physics/h1_efn/{best.pt,info.json}`.
- Figure: `results/physics/h1_panel.png`.

## Explainer entry (for site/explain.py; ledger agent to merge)
- **Question.** Do jet taggers change their decision when one particle is split into two collinear pieces, or when a very soft particle is added? Neither change alters any IRC-safe property of the jet.
- **Physics background.** An observable $O$ is IRC safe if
$$O(p_1,\dots,p_i,\dots,p_n)=O(p_1,\dots,z p_i,(1-z)p_i,\dots,p_n)\quad\text{(collinear)}$$
$$\lim_{\epsilon\to 0} O(p_1,\dots,p_n,\epsilon q)=O(p_1,\dots,p_n)\quad\text{(infrared)}$$
  Only such observables have finite perturbative QCD predictions and do not depend on how the detector or the hadronisation splits energy. Mass, soft-drop mass, $\tau_{21}=\tau_2/\tau_1$ and $D_2=e_3/e_2^3$ are IRC safe. The multiplicity $N$ and $p_T^D=\sqrt{\sum_i p_{T,i}^2}/\sum_i p_{T,i}$ are not. An energy flow network $L=\sum_i z_i\,\Phi(\Delta\eta_i,\Delta\phi_i)$ with $z_i=p_{T,i}/\sum_j p_{T,j}$ is IRC safe by construction.
- **Method.**
  - Split: replace the particle $p$ by $p_1=z\,p+k\hat u$ and $p_2=(1-z)\,p-k\hat u$, with $\hat u\perp p$ and $k=\Delta R\,z(1-z)\,p_T$. The 3-momentum is conserved exactly, and at $\Delta R=0$ the four-momentum is too.
  - Soft emission: add a particle with $p_T=\epsilon\,p_{T,\rm jet}$ in $\Delta R<0.8$, its momentum taken from the hardest particle.
  - Compare with a control of the same size that keeps $N$ fixed (moving the same pT fraction to the nearest neighbour), with the EFN, and with weight-scrambled networks.
- **Result (figure h1_panel.png).** Splitting the hardest particle into 5 % and 95 % pieces changes 1.1–5.7 % of the calls of the eight trained taggers, against 0 % for the EFN on jets without truncation. Splitting more particles changes more calls (up to 11–26 % for eight splits). A particle with $10^{-4}$ of the jet pT changes 0.7–7.4 % of calls; for most taggers this rate barely falls with $\epsilon$, while the EFN's falls about tenfold per decade.
- **Checks.** IRC-safe observables are unchanged to $10^{-13}$ at $\Delta R=0$; McNemar tests against the matched control, Holm-corrected over 521 tests; rates are also given without truncated jets. Each model's unedited accuracy is within 1 point of its reference value.
- **Physical significance.** The taggers read particle multiplicity, which is IRC unsafe and depends on hadronisation and detector granularity. This bears on how they transfer to data and to other simulators, and on reading their decisions as IRC-safe jet-shape cuts.
