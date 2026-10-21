# P39v: skeptic's verification of P39 (axis dependence on unedited jets; φ-mirror input split; P31 final)

Status: done. Job `anrunw-ai-c-p39-verify` (V100, 16 min; smoke then full in one command). Script `interp/p39v_verify.py`. Results `results/part/p39_verify.json` (smoke: `p39_verify_smoke.json`). Log `logs/c-p39-verify.log`. No figure was made: the task allowed only `p39_verify*.json` outputs.

## Conclusion
Conclusion: on a fresh sample the main P39 results reproduce. (1) Comparing unedited off-centre jets with centred jets of the same mass does not test ParT's axis dependence. The pT-weighted centre of a real jet sits off its axis almost entirely because pseudorapidity is not linear in the longitudinal momentum. The offset therefore follows from the jet's |η| and width (96 % of its variance), and moving the off-centre jet to its partner's |η| removes all of the score gap (87–172 % depending on the network). (2) Shifting a jet so that its momentum direction moves off the axis while its spread about the axis (M2, defined below) stays fixed gives almost no response: 0.06 of the recentring response on this sample, against 0.19 in P39. A shift of the same length that raises M2 instead gives a response of opposite sign. This supports the spread reading over the momentum-direction reading, but the networks follow increases in M2 at only 0.7–0.85 of the size of a real mass change. (3) The full models' sensitivity to a φ mirror nearly disappears when charge and d0 signs flip with it. Still uncertain: under rigid shifts, M2 and the centroid offset cannot be told apart, and ParT_kin's response to these small shifts is not resolved.

## Definitions
- **Stored axis**: JetClass (jet_η, jet_φ); the networks receive each particle's Δη, Δφ and ΔR relative to it.
- **Centroid offset** $\vec c=\sum_i p_{T,i}(\Delta\eta_i,\Delta\phi_i)/\sum_i p_{T,i}$, the pT-weighted centre of the particles relative to the stored axis.
- **Moments.** $M_0=\sum p_T$; $M_2=\sum p_T r^2$ about the stored axis; $M_{2c}=M_2-|\vec c|^2M_0$, the same moment about the centroid.
- **Axis-moment mass** $m_A=\sqrt{M_0M_2}$.
- **Read-out** $d=\ell_Z-\ell_W$, the Z logit minus the W logit.
- **Local slope** $s_j$: the per-jet change in d per unit log-mass under a ±3 % angular dilation about the stored axis (a real mass change).
- **R**: mean Δd under an edit divided by the mean $s_j\,\Delta\log m_A$. R = 1 means the network responds as it would to a real mass change of the same size in $m_A$.
- **$R_{nat}$**: P39's natural-pair ratio $\sum(y-P_I)/\sum(P_A-P_I)$, where y is the score gap within a pair, $P_A=\bar s\,\Delta\log m_A$ and $P_I=\bar s\,\Delta\log m_{inv}$.
- **iso shift**: a rigid shift of length $|\vec c|$ at ±120° to $\vec c$. It keeps M2 exactly and moves the momentum direction about $|\vec c|$ off the axis.
- **perp shift** (new): a rigid shift of length $|\vec c|$ at ±90° to $\vec c$. It moves the momentum direction as far as recentring does but raises M2 by $|\vec c|^2M_0$. That is about the same size as the drop under recentring, with the opposite sign.
- **η transport** (new): the off-centre jet of a natural pair is moved to its partner's jet_η, with every particle's Δη, Δφ, pT and mass kept. Only the particle energies (and the stored axis) change.
- **η law** (new): a second-order expansion of $\sum_ip_{T,i}\sinh\eta_i=p_{T,jet}\sinh\eta_{jet}$ gives $c_\eta\approx-\tanh|\eta_{jet}|\,M_2/(2M_0)$, in the signed Δη convention of JetClass. $c_\phi$ is third order.
- **McNemar**: an exact binomial test on jets whose call changes under one edit but not under the other.
- **Holm**: step-down family-wise correction of p-values.

## Code review of `interp/p39_natural_axis.py`
- **Sample.** W/Z rows are a permutation of the whole file (`default_rng(3939)`). Other classes use `rng.choice` without replacement. No first-k subsampling.
- **Build and η sign.** `build` uses η = jet_eta + sign(jet_eta)·Δη. `momentum_offset` and `rebuild_axis` use the same sign convention. The measured median momentum-axis offset of unedited jets is 0.0002 rad, which would be about 0.01 rad or more if the sign were wrong. Correct.
- **Padded particles.** All moments are weighted by pT, which is 0 in padded slots. Shifts are multiplied by the mask. `npart` comes from the mask. Correct.
- **Edits.**
  - iso: s = R(±120°)c gives $2\vec s\cdot\vec c+|\vec s|^2=0$, so M2 is unchanged. Measured Δlog m_A ≈ 1e-11 here.
  - dil_rec: $\sqrt{M_2^{rec}/M_2}$ is $\sqrt{M_{2c}/M_2}$. Correct.
  - Mirror variants flip only the named branches.
  - In L-GATr, rec_rb is copied from rec. That is valid: the rebuilt jet has identical absolute particle four-vectors.
- **Kin fill.** For flip edits, kin logits are copied from mirp or base. Valid, because the kin inputs contain no charge, d0 or dz.
- **No bug found that changes a reported number.** Minor points:
  1. The script docstring says the unedited momentum-axis offset is 0.0007 rad (P36v); measured here and in P39 it is 0.0002.
  2. The P39 report's mirror accuracies (78.6 → 74.1 → 78.0 %) are raw, not class-balanced. W/Z make up 20,000 of the 28,000 jets, so these differ from the class-balanced 85.6 % given elsewhere. The call-change rates are class-balanced, as stated.
  3. P39 says the iso edit separates "M2 (equivalently the centroid offset)" from momentum-axis displacement, but its Conclusion reads it as the spread about the axis. Under any rigid shift at fixed pT, $M_2=M_{2c}+M_0|\vec c|^2$ with $M_{2c}$ fixed. So "reads M2" and "reads the centroid offset" make identical predictions for every edit in P39 and here. Only the magnitude (R ≈ 1 against a real dilation) favours the M2 wording.

## Re-run on a fresh sample
**Sample.**
- JetClass test_20M `<class>_116.root` (P39 used `_109`), numpy `default_rng(3995)`, random rows.
- W/Z: 30,000 Wqq and 30,000 Zqq. The top 20 % of |c| per class gives 12,000 jets, 3× P39's top-20 set.
- Mirror test: 800 random jets per class × 10 classes (N = 8,000).

**Models.** Released ParT_full, ParT_kin, ParticleNet_full and ParticleNet_kin. L-GATr was not re-run.

**Unedited accuracy** (P1 sample, N = 10,000): ParT_full 85.93 %, ParT_kin 74.47 %, PN_full 84.56 %, PN_kin 70.36 %. These are the known values. Class-balanced 10-class accuracy on the mirror sample: 85.7 / 73.5 / 83.9 / 69.7 %.

**Statistics.** 2,000 bootstrap resamples over jets or pairs; 5,000 sign-flip permutations; exact McNemar; Holm within each family.

### (a) Natural pairs: reproduced, plus a causal control and the η law
**Matching.** P39's primary nn design gives 8,932 pairs. They match in m_inv (mean Δlog m_inv = −0.0001) and differ in |c| by +0.011 rad. The |η_jet| standardised mean difference is **+4.65** (P39: 4.45).

**η law.** $c_\eta\approx-\tanh|\eta_{jet}|\,M_2/(2M_0)$:
- explains 95.4 % of the variance of $c_\eta$ (fitted slope 0.94) and 96.2 % of the variance of |c|;
- median |c_φ| is 0.00017, against 0.0055 for |c_η|.

So the natural offset is a coordinate effect: it is set by |η_jet| and the jet's angular width ⟨r²⟩ = M2/M0. Once m, pT, N and M2c are matched, |c| can differ only through |η_jet|. This is why P39's |η|-matched design kept only 718 pairs and gave unstable values.

The median offset term $\tfrac12\log(1+|\vec c|^2M_0/M_{2c})$ is 0.0008, i.e. m_A exceeds m_c by 0.08 %.

| model | score gap y [95 % CI] | P_A (H1) | R_nat [95 % CI] | y after moving the off-centre jet to the partner's η | fraction of gap removed | y after moving the centred jet to the off-centre jet's η |
|---|---|---|---|---|---|---|
| ParT_full | −0.338 [−0.418, −0.262] | +0.031 | −10.9 [−14.0, −8.1] | +0.106 [0.028, 0.181] | 1.32 [1.07, 1.69] | −0.170 [−0.250, −0.093] |
| ParT_kin | −0.243 [−0.281, −0.205] | +0.027 | −8.8 [−10.7, −7.2] | +0.174 [0.137, 0.212] | 1.72 [1.49, 2.01] | +0.011 [−0.029, 0.050] |
| PN_full | −0.325 [−0.401, −0.252] | +0.029 | −11.1 [−14.1, −8.4] | −0.041 [−0.117, 0.032] | 0.87 [0.71, 1.13] | −0.038 [−0.116, 0.037] |
| PN_kin | −0.254 [−0.283, −0.226] | +0.023 | −10.7 [−12.5, −9.1] | +0.052 [0.027, 0.079] | 1.20 [1.09, 1.35] | +0.033 [0.004, 0.063] |

**Tests.**
- H0 "y = 0" and H0 "the transport leaves y unchanged" are each rejected for every model: sign-flip p_Holm 0.0016, family of 8, the floor at 5,000 permutations.
- Competing predictions:
  - If the gap came from the centroid offset, the transport would leave y unchanged. Excluded.
  - If it came from |η_jet|, the transport would bring y to about P_A. Not excluded for PN_full (p 0.06) or ParT_full (p 0.06). For ParT_kin (p 2e-4) and PN_kin (p 0.03) y overshoots P_A.
- Caveat on the transport: the moved jet's particles keep the offset belonging to its old |η|. Its momentum axis is therefore no longer on the stored axis: the median offset rises from 0.0004 to 0.0104 rad. The overshoot in ParT may come from this mismatch. This control establishes that the sign and size of the natural gap come from |η_jet| through the energy inputs. It does not give a clean residual R_nat.

### (b) Edits on W/Z jets: reproduced, with one number smaller than in P39
Top 20 % of |c| per class, N = 12,000. Mean Δd [95 % CI]; R in parentheses.

| model | rec | iso | perp (new) | rs | ref | dil_rec | iso / rec | perp / rec |
|---|---|---|---|---|---|---|---|---|
| ParT_full | −0.0280 [−0.0304, −0.0257] (R 0.94 [0.86, 1.01]) | −0.0016 [−0.0041, 0.0010] | +0.0211 (R 0.71 [0.62, 0.80]) | +0.0232 (R 0.82) | +0.0744 (R 0.84) | −0.0302 (R 1.01) | **0.06 [−0.04, 0.14]** | −0.75 [−0.88, −0.64] |
| PN_full | −0.0250 (R 0.93 [0.82, 1.04]) | −0.0026 [−0.0058, 0.0005] | +0.0211 (R 0.79) | +0.0167 (R 0.65) | +0.0616 (R 0.78) | R 1.01 | 0.10 [−0.02, 0.22] | −0.85 [−1.02, −0.69] |
| PN_kin | −0.0189 (R 0.87) | −0.0020 | +0.0147 (R 0.68) | +0.0125 (R 0.59) | +0.0469 (R 0.73) | R 1.00 | 0.11 [−0.00, 0.21] | −0.78 [−0.93, −0.63] |
| ParT_kin | −0.0047 [−0.0113, 0.0018] (R 0.19) | −0.0031 | −0.0022 | −0.0013 | +0.0043 | R 1.72 | not resolved [−3.2, 3.6] | not resolved |

**Edit geometry** (top 20 %, medians). Momentum-axis offset after the edit: iso 0.0126 rad, perp 0.0123, rec 0.0130 (P39 value; here also ≈ 0.013). Δlog m_A: iso 0, perp +0.0029, rec −0.0028. The invariant mass is unchanged (|Δlog m| ≤ 1e-6).

**Predictions for (iso/rec, perp/rec)** under each hypothesis:
- the network reads M2 (axis moment): (0, −1);
- it reads how far the momentum direction is displaced: (+1, +1);
- it responds linearly to displacement along c: (0.5, 0).

**Results** (ParT_full, top 20 %):
- iso = rec, iso = rec/2, perp = rec and perp = 0 are each rejected (p_Holm 0.0032, family of 16 across the 4 models; permutation floor).
- iso = 0 is not rejected (p 0.22).
- perp = −rec is rejected on the top 20 % (p 6e-4) but not on all 60,000 jets (perp/rec −0.94 [−1.05, −0.85], p 0.26).
- PN_full and PN_kin show the same pattern.

**Joint fit.** Per jet over rec, iso, perp and ref: Δd = a·s·Δlog m_A + b·s·(signed displacement of the momentum direction along −c).
- ParT_full: a = 0.82 [0.78, 0.85]; b = 0.005 [−0.016, 0.028] per rad, consistent with no displacement term.
- PN_full: a = 0.69, b ≈ 0.

**Difference from P39.**
- P39 found iso/rec 0.19 [0.07, 0.32] for ParT_full (N = 4,000). This sample gives 0.06 [−0.04, 0.14] (N = 12,000). The intervals overlap, so the "residual of about 0.2" in P39's reading does not reproduce as a distinct effect.
- R(rec) is 0.94 here (P39: 1.04). Edits that raise M2 (perp, rs, ref) give R 0.71–0.84, so the response is not symmetric around R = 1.
- ParT_kin remains unresolved, as in P36, P36v and P39. Each rigid edit changes 6 % of its calls with no resolvable mean W/Z shift.

### (c) φ mirror: reproduced
Class-balanced share of 10-class calls changed [95 % CI].

| model | mirp (φ only) | + charge | + d0 sign | + charge and d0 | McNemar mirp vs + charge, d0 (mirp-only / variant-only), p_Holm (m = 6) |
|---|---|---|---|---|---|
| ParT_full | 7.74 % [7.18, 8.34] | 10.91 % | 8.91 % | **2.78 % [2.45, 3.13]** | 536 / 139, 4e-55 |
| PN_full | 11.62 % [10.97, 12.31] | 11.50 % | 5.70 % | **3.15 % [2.81, 3.54]** | 814 / 136, 1.5e-117 |
| ParT_kin / PN_kin (reference) | 1.58 / 1.45 % | = | = | = | – |

P39 had 7.82 → 2.80 % and 11.83 → 3.36 %. The one difference: PN_full with charge flipped alone is not distinguishable from mirp here (p_Holm 0.65), whereas P39 had 9e-10 at N = 28,000.

### (d) P31 final
Not re-run. Spot check of `results/part/p31_classes_steps_final.json`: per-jet R φ+ is m10 0.350 [0.340, 0.361], m10s 0.316, L 0.293, t 0.215 and t2 0.254. Released ParT_kin and ParT_full reproduce 74.47 / 85.93 % on the P1 sample. These match the report.

## Adequacy of the controls
- **(a)** P39 inferred the |η| confound from covariate balance and from L-GATr showing the same gap. The η transport here is a causal control: moving only the jet's |η| removes the gap. The η law shows why no observational design can separate the offset from |η|.
- **(b)** The iso edit tests only one competitor, "displacement magnitude". The perp edit adds a test with a sign: the M2 reading predicts a response opposite to recentring, momentum displacement predicts the same sign, and the observed sign is opposite.
  - Not covered by any edit: M2 versus the centroid offset |c| as the quantity read. They are interchangeable under rigid shifts.
  - Separating them needs an edit that changes M2c at fixed |c|, for example a dilation about the centroid. At natural offsets the offset term is 0.08 % of m_A, so the two predictions would differ by about 3 %. Not tested.
- **(c)** The kin models are a valid reference, since they read none of charge, d0 or dz.

## Corrected reading
- **Unedited jets.** Off-centre and centred W/Z jets matched in m, pT, N and M2c differ by 4.5–4.65 SD in |η_jet|. This follows from $c_\eta\approx-\tanh|\eta_{jet}|M_2/(2M_0)$ (96 % of the variance of |c|). Every network gives a score gap about 10× the m_A prediction and of opposite sign. Moving the off-centre jet to its partner's |η| removes all of it (87–172 %). The natural comparison therefore cannot identify the axis-moment effect.
- **Edits.** A shift that moves the momentum direction as far as recentring does, but keeps M2 about the stored axis fixed, gives 0.06 [−0.04, 0.14] (this sample) to 0.19 [0.07, 0.32] (P39) of the recentring response in ParT_full. A shift that raises M2 by the same amount gives −0.75 of it. This excludes momentum-axis displacement as the source of P36's response.
  - Recentring gives R 0.94–1.04. M2-raising shifts give R 0.71–0.84.
  - Rigid edits cannot tell M2 from the centroid offset.
- **Mirror.** As in P39: flipping charge and d0 signs with the mirror lowers the full models' call changes to within 1.2–1.7 points of the kin models.

## Glossary additions
- **perp shift**: a rigid shift of length |c| at 90° to c. It moves the momentum direction as far as recentring does while raising M2 by |c|²M0.
- **η transport**: a jet moved to another jet_η with its particles' Δη, Δφ, pT and masses kept, so that only the particle energies change.
- **η law of the centroid offset**: $c_\eta\approx-\tanh|\eta_{jet}|\,M_2/(2M_0)$. The pT centroid of a real jet sits off its momentum direction because pseudorapidity is not linear in p_z.
