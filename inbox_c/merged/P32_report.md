# P32: lepton route for H→ℓνqq′ in ParT and ParticleNet (new claim C15; exploratory) (agent report, 2026-10-20)

Job `anrunw-ai-c-p32-run2` (RTX 2080 Ti, 1,268 s): `python -m interp.p32_lepton_route && python -m interp.p32_lepton_route --plot`. Figure re-rendered by CPU job `c-p32-plot` with corrected labels and legend.
- Script: `interp/p32_lepton_route.py`.
- Results: `results/part/p32_lepton_route.json`.
- Figure: `results/part/p32_lepton_route.png`.
- Logs: `logs/c-p32-run2.log`; first attempt `logs/c-p32-run.attempt1.log` with `results/part/p32_lepton_route.attempt1.json`.

**Attempt 1 (`c-p32-run`, not used).** It completed, but had two bugs:
- **Rebuild mismatch.** The unedited jets went through `part_core.build` once and every edited jet twice. A second build is not exactly idempotent: the float32 particle mass is recomputed from E² − p². ParT_kin has no PID input, so it should not respond to e1 or e5 at all. Instead it showed Δ log p −0.03 under e1 and the same +0.025 for all three e5 variants.
  - Fix: the base jets now get the same number of build passes. After the fix, ParT_kin gives exactly 0 for e1, e2 and e5.
- **p-value floor.** The bootstrap p had a floor of 1/1,000, so the smallest possible Holm p over 80 heads was 0.08 and no head could pass.
  - Fix: the head scans now use a normal approximation with the bootstrap standard error (field `p0`). The percentile-bootstrap p is kept as `p0_boot`.
- A top-3-head literal circuit was also added.

## Conclusion
- In ParT and ParticleNet with particle-ID inputs, removing the electron/muon flag from the lepton candidate (all else unchanged) moves almost every H→ℓνqq′ jet out of that class: about 99 % of calls change. The same flag change on a hadron of similar pT changes 12–25 %.
- One class-attention head per ParT model puts nearly all of its attention on the lepton. In ParT_full, blocking only that head's view of the lepton gives the largest single-head drop. But at the level of the 10-class call:
  - the lepton information also reaches the output through many other heads (37 of 80 heads give significant drops);
  - a circuit of the top three heads, with all other heads replaced by their mean, still calls 97 % of H→ℓνqq′ jets correctly, but it also labels many other jets as H→ℓνqq′, so it is not a specific lepton circuit.
- ParT_kin (no flag inputs) reads the same particle through its own class-attention head, identified by kinematics. It responds to removing or rescaling the lepton.
- Still open: what the remaining particle-block heads compute, and whether some smaller head set keeps the 10-class performance.

## Procedure
**Models.**
- ParT_full, ParT_kinpid, ParT_kin: released JetClass weights, `models/part`.
- PN_full, PN_kinpid: released ParticleNet weights, `models/particlenet`.
- kinpid input list: JetClass_kinpid.yaml, i.e. the kinematic features plus charge and five PID flags, without impact parameters.

**Sample.**
- JetClass test_20M `<sample>_115.root`, numpy `default_rng(3215)`, random rows (not first-k): 5,000 HToWW2Q1L rows plus 556 random rows of each of the other 9 classes. N = 10,004.
- Every jet, unedited included, is passed through `part_core.build` once more than the jets it is compared with, so that edited and unedited jets have the same number of build passes.

**Lepton candidate.**
- Definition: the highest-pT particle with isElectron or isMuon.
- Jets with no candidate: 1.5 % of H→ℓνqq′ jets; 38–84 % of the other hadronic classes; 0.2 % of t→bℓν.
- 4,926 H→ℓνqq′ jets have a candidate and a charged hadron to use as a control: 49.7 % electrons, 50.3 % muons.
- The lepton is the leading particle in 71 % of these jets. Median lepton pT share is 0.32 (10th–90th percentile 0.08–0.66).

**Control token.**
- A random charged hadron with pT within a factor 1.5 of the lepton's pT, otherwise the charged hadron nearest in pT.
- Median |ln(pT ratio)| = 1.09: in many jets no hadron is close in pT.

**Edits** (on the 4,926 H→ℓνqq′ jets):
- **e1:** lepton flag → isChargedHadron. Kinematics, charge and d0/dz are kept. 13.9 % of edited jets still contain another lepton-flagged particle.
- **e2 (control):** the matched hadron is given the lepton's flag.
- **e3:** the lepton is removed and its four-vector is added to the nearest charged hadron in (Δη, Δφ). Jet mass change: median 3·10⁻⁴ GeV, max 0.009 GeV. N = 4,921.
  - **e3c (control):** the matched hadron is merged into its nearest charged hadron.
- **e4h / e4d:** lepton pT ×0.5 / ×2, with the other particles rescaled so that Σ pT is kept. Median |Δ jet pT|/pT = 0.09 %. e4d uses jets with lepton share < 0.5, N = 3,626.
  - **e4hc / e4dc (control):** the same on the matched hadron.
- **e5 (reverse):** on 1,668 QCD, H→4q and t→bqq′ jets, the leading charged hadron is set to isMuon (e5mu) or isElectron (e5el).
  - **Control e5mu_r:** a random non-leading charged hadron is set to isMuon.

**Metric.**
- Δ log p(H→ℓνqq′) per jet: edited minus unedited.
- Share of calls (argmax) changed.
- McNemar test on paired outcomes: (a) H→ℓνqq′ calls, unedited vs edited; (b) call changed under the edit vs under its control.
- 1,000 bootstrap resamples over jets give the 95 % intervals.

**ParT internals** (2,000 random jets from the 4,926):
- **Attention.** The attention weight of the class token on the lepton key per class-attention head, vs on the matched hadron. Notation: b1h1–b2h8 = class-attention block 1 or 2, heads 1–8; p1h1–p8h8 = particle blocks.
- **Knock-out (KO).** The lepton key's attention logit is set to −10⁹ for every query in one head.
  - Control: the same on the matched hadron's key.
  - H0: Δ log-odds(lepton KO) = Δ log-odds(matched KO). Paired bootstrap, normal-approximation p, Holm over 80 heads.
- **Joint KOs:** all 80 heads, the 16 class heads, the 64 particle heads, each block, and the Holm-significant set S. S is compared with 100 random head sets of the same size and stage composition. Greedy top-k for k = 1–10.
- **e1 head patching.** One class head's output is taken from the e1 run; its share of the e1 effect is reported.
- **e1 residual patching.** At layer l (0 = embedding, l = output of particle block l), either the lepton token's residual or every other token's residual is taken from the e1 run.
- Log-odds here means logit_ℓνqq − logsumexp(other 9 logits); unlike log p it does not saturate near p = 1.

**Literal circuit** (ParT_full, ParT_kinpid; whole 10-class sample):
- Heads outside the circuit are replaced by their mean output over a class-balanced pool: 556 jets per class, all real tokens. MLPs are kept.
- **Scope "class":** only class-attention heads are ablated; the circuit is S ∩ class heads.
- **Scope "all":** all 80 heads; the circuit is S.
- **Scope "top3":** all 80 heads; the circuit is the three heads with the largest single-head lepton KO.
- Reported for each: H→ℓνqq′ recall (95 % CI), share of other-class jets called H→ℓνqq′, 10-class accuracy, balanced accuracy and per-class recall (full confusion in the JSON).
- Control: 100 random head sets of the same size and composition, with a permutation p.

**ParticleNet internals:**
- The share of the lepton's k = 16 nearest neighbours that change at EdgeConv 1–3 under e1, compared with the matched hadron under e2 and with all tokens under e1.
- e1 with the kNN graphs held at the unedited ones.

## Numbers
**Unedited accuracy.**
- P1 sample (part_core.sample, rows 3,000–4,000 of `_100`), reproduced: ParT_full 0.8593 (known 85.9 %), ParT_kinpid 0.7902, ParT_kin 0.7447 (known 74.5 %), PN_full 0.8456, PN_kinpid 0.7720.
- 10-class accuracy on the P32 sample [95 % CI], dominated by H→ℓνqq′ jets: ParT_full 0.916 [0.910, 0.921], ParT_kinpid 0.867, ParT_kin 0.799, PN_full 0.906, PN_kinpid 0.847.
- Balanced accuracy on the P32 sample: 0.865, 0.799, 0.739, 0.849, 0.770.
- H→ℓνqq′ recall: 0.979, 0.951, 0.873, 0.977, 0.944.
- Recall on H→ℓνqq′ jets with no lepton candidate (N = 73): 0.53 (ParT_full), 0.64 (ParT_kin).

**Input edits on H→ℓνqq′ jets.** Each cell: share of calls changed, then mean Δ log p [95 % CI]. N = 4,926; e3 N = 4,921; e4d N = 3,626.

| edit | ParT_full | ParT_kinpid | ParT_kin | PN_full | PN_kinpid |
|---|---|---|---|---|---|
| e1 lepton flag → hadron | 0.987; −7.23 [−7.29, −7.17] | 0.985; −7.23 | 0 (no flag input) | 0.986; −7.96 | 0.983; −8.10 |
| e2 control | 0.145; −0.40 [−0.43, −0.37] | 0.228; −0.72 | 0 | 0.116; −0.31 | 0.247; −0.74 |
| e3 lepton merged | 0.968; −6.23 [−6.29, −6.16] | 0.984; −7.26 | 0.879; −5.01 [−5.08, −4.93] | 0.952; −5.92 | 0.981; −7.86 |
| e3c control | 0.023; −0.06 | 0.032; −0.04 | 0.050; −0.04 | 0.013; −0.02 | 0.017; −0.01 |
| e4h lepton pT ×0.5 | 0.035; −0.09 [−0.10, −0.07] | 0.093; −0.19 | 0.278; −0.82 [−0.87, −0.78] | 0.037; −0.10 | 0.099; −0.19 |
| e4hc control | 0.004; −0.00 | 0.010; −0.00 | 0.043; +0.03 | 0.004; −0.00 | 0.009; −0.00 |
| e4d lepton pT ×2 | 0.059; −0.13 | 0.193; −0.40 | 0.323; −0.60 | 0.082; −0.19 | 0.228; −0.87 |
| e4dc control | 0.013; −0.04 | 0.028; −0.04 | 0.082; −0.09 | 0.013; −0.03 | 0.027; −0.05 |

- **e1 vs e2 (ParT_full).** Calls changed: McNemar 4,160 jets changed under e1 only vs 9 under e2 only, p < 10⁻³⁰⁰. Δ(Δ log p) = −6.83 [−6.90, −6.77].
  - In all four PID models, e1 vs e2 has McNemar p ≈ 0.
- **Where e1 sends the ParT_full calls:** H→4q 2,073, Z→qq 1,111, W→qq 1,005, QCD 516, other classes ≤ 71.
- **e1 by flavour (ParT_full):** electrons −7.20 (98.3 % changed), muons −7.26 (99.2 %).
- **e1 by lepton pT-share tercile (ParT_full):** −7.62, −7.62, −6.44.
- **e2 changes go mostly to t→bℓν** (626 of the ParT_full changes): two lepton-flagged particles.
- **e4 vs control (ParT_full):** e4h ΔΔ −0.086 [−0.100, −0.072], McNemar p 6·10⁻³⁷. ParT_kin e4h ΔΔ −0.85 [−0.90, −0.80].

**Reverse edit e5** (1,668 QCD/H→4q/t→bqq′ jets). Each cell: Δ log p(H→ℓνqq′); then share of calls moved to H→ℓνqq′ [95 % CI].

| | ParT_full | ParT_kinpid | PN_full | PN_kinpid |
|---|---|---|---|---|
| e5mu leading hadron → μ | +4.91; 0.092 [0.080, 0.108] | +3.20; 0.026 | +5.02; 0.069 | +3.54; 0.022 |
| e5el leading hadron → e | +4.70; 0.088 | +2.97; 0.025 | +4.97; 0.071 | +3.64; 0.024 |
| e5mu_r random hadron → μ (control) | +2.15; 0.012 | +1.09; 0.002 | +1.91; 0.004 | +1.12; 0.001 |

- e5mu − e5mu_r in ParT_full: +2.76 [2.60, 2.93].
- Share moved to H→ℓνqq′ under e5mu, by class (ParT_full): QCD 0.160, H→4q 0.103, t→bqq′ 0.014.
- ParT_kin: exactly 0 for all three.

**ParT class attention on the lepton.** Mean weight lepton / matched hadron; uniform-attention reference (`uniform` in the JSON) 0.039. Holm p < 10⁻⁴⁵ for every head listed.
- ParT_full: b2h1 0.993 / 0.000, b2h6 0.598 / 0.074, b2h8 0.529 / 0.001, b2h2 0.33, b2h3 0.32.
- ParT_kinpid: b2h4 0.993 / 0.000, b2h7 0.974 / 0.001.
- ParT_kin: b2h4 0.955 / 0.002, b2h5 0.393 / 0.009.

**Single-head lepton KO.** Δ log-odds [95 % CI]; the matched-hadron control is in parentheses. N = 2,000.
- **ParT_full:**
  - b2h1 −5.38 [−5.47, −5.29] (control 0.000);
  - p7h8 −0.46 (0.000), p8h6 −0.37 (+0.002), p8h2 −0.23, b2h6 −0.23 (−0.013), p3h2 −0.19.
  - b2h1 KO as Δ log p: −0.14 (p(H→ℓνqq′) is near 1 on these jets).
  - 37 of 80 heads Holm-significant (lepton effect more negative than control): 9 class heads, 28 particle-block heads.
- **ParT_kinpid:** b2h4 −2.39 [−2.48, −2.29], p8h5 −0.57, p4h5 −0.52; 36 heads Holm-significant.
- **ParT_kin:** b2h4 −1.80 [−1.86, −1.73], p8h3 −0.37, p8h1 −0.13; 40 heads Holm-significant.

**Joint lepton KO** (ParT_full): Δ log-odds; share of calls changed (matched-hadron control in parentheses); H→ℓνqq′ recall after the KO.

| heads masked | Δ log-odds [95 % CI] | control | calls changed | H→ℓνqq′ recall |
|---|---|---|---|---|
| all 80 | −13.74 [−13.91, −13.56] | −1.11 | 0.947 (0.013) | 0.985 → 0.049 |
| 16 class heads | −6.78 [−6.90, −6.65] | −0.19 | 0.110 (0.002) | 0.880 |
| 64 particle heads | −5.47 [−5.58, −5.36] | −0.85 | 0.060 (0.010) | 0.930 |
| block b2 (8) | −6.76 | −0.02 | 0.100 (0.000) | 0.888 |
| S (37 Holm heads) | −10.45 [−10.58, −10.31] | −0.05 | 0.784 (0.004) | 0.204 |

- S vs 100 random 37-head sets: random mean −5.44, minimum −9.59, permutation p 0.0099.
- Greedy top-k share of the all-80 KO: k = 1: 0.39, 3: 0.43, 5: 0.53, 10: 0.64.
- ParT_kinpid: all 80 −11.43 (0.923 changed), class 16 −3.82 (0.032 changed); top-1 / top-3 / top-10 shares 0.21 / 0.28 / 0.46.
- ParT_kin: all 80 −6.36 (0.745 changed), class 16 −1.95 (0.128 changed); top-1 / top-3 / top-10 shares 0.28 / 0.45 / 0.80.

**e1 patching** (ParT_full, e1 on the 2,000 KO jets: Δ log-odds −15.59).
- Class-head patch shares: b2h1 0.34 (Δ −5.25 [−5.34, −5.16]); every other class head ≤ 0.06; all 16 together 1.00.
- ParT_kinpid: b2h4 0.16.
- Residual patch shares, lepton token / all other tokens:

| layer | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 |
|---|---|---|---|---|---|---|---|---|---|
| ParT_full | 1.00 / 0.00 | 0.98 / 0.00 | 0.96 / 0.00 | 0.92 / 0.01 | 0.87 / 0.03 | 0.78 / 0.06 | 0.69 / 0.09 | 0.58 / 0.19 | 0.50 / 0.31 |
| ParT_kinpid | 1.00 / 0.00 | 0.96 / 0.00 | 0.90 / 0.01 | 0.80 / 0.03 | 0.66 / 0.08 | 0.60 / 0.09 | 0.53 / 0.10 | 0.41 / 0.19 | 0.30 / 0.34 |

**Literal circuits** (whole sample, N = 10,004; mean-ablation pool 5,560 jets, 556 per class).
- Columns: H→ℓνqq′ recall [95 % CI]; other-class jets called H→ℓνqq′; 10-class accuracy; balanced accuracy.
- Random head sets (100): mean (max), permutation p.
- Reference: unedited ParT_full has recall 0.979, 10-class accuracy 0.916 and balanced accuracy 0.865 on this sample.

| model, scope, heads | recall | others → ℓνqq | acc10 | balanced | random recall | random acc10 |
|---|---|---|---|---|---|---|
| ParT_full class, 9 (b2h1, b2h6, b2h8, b1h7, b1h4, b2h2, b2h3, b1h1, b2h5) | 0.989 [0.986, 0.992] | 0.010 | 0.855 | 0.748 | 0.972 (0.986), p 0.010 | 0.887 (0.913), p 0.90 |
| ParT_full all, 37 (S) | 0.984 [0.980, 0.987] | 0.048 | 0.698 | 0.470 | 0.941 (0.985), p 0.020 | 0.809 (0.867), p 1.0 |
| ParT_full top3 (b2h1, p7h8, p8h6) | 0.973 [0.969, 0.977] | 0.136 | 0.550 | 0.212 | 0.060 (0.977), p 0.050 | 0.107 (0.553), p 0.050 |
| ParT_kinpid class, 5 (b2h4, b1h8, b1h1, b1h5, b1h3) | 0.945 | 0.007 | 0.816 | 0.712 | 0.840 (0.974), p 0.39 | 0.695 (0.849), p 0.10 |
| ParT_kinpid all, 36 | 0.943 | 0.053 | 0.715 | 0.533 | 0.756 (0.975), p 0.28 | 0.582 (0.788), p 0.13 |
| ParT_kinpid top3 (b2h4, p8h5, p4h5) | 0.820 | 0.136 | 0.464 | 0.179 | 0.050 (0.853), p 0.050 | 0.092 (0.481), p 0.050 |

- Per-class recall, ParT_full top3: H→ℓνqq′ 0.97, Z→qq 0.83, H→cc 0.32, all other classes 0.00.
- Per-class recall, ParT_full class circuit: QCD 0.38, W→qq 0.10, the other classes 0.81–0.99.
- With no head kept, every jet gets the same non-ℓνqq call: acc10 0.056 = 556/10,004.

**ParticleNet kNN.** Share of the lepton's 16 neighbours changed at EdgeConv 1 / 2 / 3.

| | under e1 | matched hadron under e2 | all tokens under e1 |
|---|---|---|---|
| PN_full | 0 / 0.023 / 0.043 | 0 / 0.039 / 0.071 | 0 / 0.012 / 0.027 |
| PN_kinpid | 0 / 0.034 / 0.079 | 0 / 0.059 / 0.095 | – |

- e1 with the graphs held fixed: PN_full −15.54 vs −15.60 free. The graph change carries −0.06 [−0.07, −0.04], 0.4 % of the effect.
- PN_kinpid: 0.1 %.

## Reading
**Hypotheses.**
- **H0 (edit level):** e1 changes calls no more often than the matched control e2. Rejected in all four PID models (McNemar p ≈ 0; 98–99 % vs 12–25 %).
- **Thesis, part by part:**
  - "e1 drops p(ℓνqq) on most jets, ≫ e2": observed.
  - "the effect concentrates in ≤ 3 class-attention heads reading the lepton token": partly.
    - One class head per model attends almost only to the lepton: ParT_full b2h1, ParT_kinpid b2h4, both 0.99.
    - That head gives the largest single-head KO: 39 % and 21 % of the all-80 KO in log-odds; 34 % of the e1 log-odds change in ParT_full by output patching.
    - Masking all 16 class heads changes only 11 % of ParT_full calls, against 95 % for all 80 heads.
    - 37 heads are Holm-significant, and the top 3 carry 43 % of the all-80 KO.
    - Residual patching shows the e1 information on the other tokens' residuals from block 6 on (31 % at block 8).
    - So at the call level the lepton is read through particle-block attention as well as the class heads.
  - "the literal circuit keeps most ℓνqq accuracy": the top-3 circuit keeps H→ℓνqq′ recall at 0.97, but it calls 14 % of the other jets H→ℓνqq′ and 10-class accuracy falls from 0.92 to 0.55.
    - The random 3-head controls have mean recall 0.06, but 4 of 100 random sets reach 0.97 or more.
    - The per-set head lists were not stored. A set contains b2h1 with probability 1/16, so b2h1 being in those sets would fit, but this is not tested.
  - "ParT_kin unaffected by e1 but affected by e3/e4": observed (e1 exactly 0; e3 88 %, e4h 28 %, e4d 32 % of calls changed).
- **Alternative, part by part:**
  - "e1 changes few calls, e3/e4 dominate in all models": excluded for the four PID models.
    - e4 changes 4–23 % of their calls, against 98–99 % for e1.
    - It holds only for ParT_kin.
  - "knock-out effects spread over many heads": partly supported.
    - Many heads are significant.
    - One class head carries the largest single share.
    - 10 heads carry 64 % (ParT_full).
- **Reverse edit.** A muon or electron flag on the leading charged hadron of QCD/H→4q/t→bqq′ jets raises log p(ℓνqq) by about 5 in ParT_full and PN_full, but moves only 7–9 % of calls (QCD 16 %, t→bqq′ 1.4 %). The flag alone does not make the call; the rest of the jet also enters.

**ParT_kin.**
- Without flag inputs it still concentrates one class head (b2h4, weight 0.955) on the lepton candidate.
- Masking the lepton key in all heads changes 75 % of its calls.
- So it reads the same particle, selected by its kinematics (in 71 % of jets the lepton is the leading particle), not by a separate jet-level route.

**ParticleNet.**
- e1 has the same size as in ParT.
- It acts through the EdgeConv features, not the kNN graph: the lepton's neighbours change by ≤ 8 %, less than under the control flag change, and holding the graph fixed keeps 99.6 % of the effect.

**Scope.**
- One file (_115), one seed; the matched-pT control is loose (median pT ratio about 3).
- KO masks keys for all queries in a head, including the lepton's own query.
- Δ log-odds on jets with p ≈ 1 overstates call-level effects. Calls changed are reported next to it.
- Literal circuits ablate heads only: MLPs and the embedding are kept.
- Random-set p values are limited to ≥ 0.0099 by the 100 sets.

## Explainer entry (for site/explain.py)
```python
P32_EXPLAINER = [
 ('Question', 'Does ParT recognise $H\\to WW^*\\to \\ell\\nu q\\bar q\'$ jets through the one particle flagged as an electron or muon, and is that read by a small set of attention heads?', None),
 ('Background',
  'JetClass gives each particle five particle-ID flags (charged hadron, neutral hadron, photon, electron, muon). A semileptonic $H\\to WW^*$ jet contains a hard lepton; hadronic jets rarely do. ParT ends with two class-attention blocks in which a class token attends over the particles; each block has 8 heads. ParT_kin is trained without the flags.', None),
 ('Method',
  '5,000 random $H\\to\\ell\\nu qq\'$ jets and 556 random jets of each other class (JetClass test file _115, seed 3215). Lepton candidate: the highest-$p_T$ particle with an electron or muon flag. Edits: (e1) change its flag to charged hadron; (e2, control) give a hadron of similar $p_T$ the lepton flag; (e3) merge the lepton into its nearest charged hadron (jet four-vector kept); (e4) scale the lepton $p_T$ by 0.5 or 2 at fixed $\\sum p_T$; (e5) put a muon flag on the leading hadron of QCD, $H\\to4q$ and $t\\to bqq\'$ jets. '
  'Inside ParT: for each head, block attention to the lepton (its attention logit set to $-10^9$) and compare with blocking a matched hadron; Holm correction over 80 heads. Literal circuit: keep only the chosen heads and replace every other head by its mean output over a class-balanced pool; report $H\\to\\ell\\nu qq\'$ recall and 10-class accuracy against 100 random head sets of the same size.', None),
 ('Result',
  'Changing only the lepton flag moves 98.7 % of ParT_full\'s $H\\to\\ell\\nu qq\'$ calls (mostly to $H\\to4q$, $Z$, $W$); the control flag change moves 14.5 %. ParticleNet behaves the same (98.6 % vs 11.6 %). One class-attention head (b2h1 in ParT_full) puts 99 % of its attention on the lepton and gives the largest single-head drop, but 37 of 80 heads have significant effects; masking the lepton in all 16 class heads changes 11 % of calls, in all 80 heads 95 %. Keeping only the top three heads keeps 97 % recall but calls 14 % of other jets $H\\to\\ell\\nu qq\'$ (10-class accuracy 0.55).', 'results/part/p32_lepton_route.png'),
 ('Checks',
  'ParT_kin (no flags) gives exactly zero response to the flag edits and responds to removing (88 % of calls) or rescaling (28–32 %) the lepton; it also has one class head with 96 % attention on the same particle. Merging a matched hadron changes 1–5 % of calls. Holding ParticleNet\'s neighbour graph fixed keeps 99.6 % of the e1 effect. Unedited accuracy on the reference sample: ParT_full 85.9 %, ParT_kin 74.5 %.', None),
 ('Why it matters',
  'This is the first ParT class outside the $W/Z$ mass question with a single-particle causal edit: the $H\\to\\ell\\nu qq\'$ decision depends almost entirely on one input flag of one particle. The flag is read through one dominant class head plus many particle-block heads, so the literal three-head circuit reproduces the $H\\to\\ell\\nu qq\'$ call but not the rest of the classifier.', None),
]
```
Conclusion (for site/conclusions.py): the text of the Conclusion section above.
