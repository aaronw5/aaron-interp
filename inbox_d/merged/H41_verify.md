# H41 verification: colour flow (soft-radiation mirror) in W/Z/H→bb vs QCD

Jobs: d-h41-verify-smoke (150 jets per class, 255 s) and d-h41-verify (V100, 3,000 jets per class, 649 s); both exit 0. Script `interp/h41_verify.py`. It imports the geometry, soft-particle selection and M/C1/C2 edits from `interp/h41_colour_flow.py` without changing them. Outputs: `results/physics/h41_verify.json` (all cells) and `h41_verify_summary.json` (key cells). Log: `logs/d-h41-verify.log` on the volume.

## Conclusion

On a fresh sample, ParT and ParticleNet (full and four-momentum-only) again lower their W, Z and H→bb scores when soft particles near a prong are mirrored from the side facing the other prong to the far side. The edit that mirrors the other way raises those scores. The response remains when jet mass, τ21, girth and the energy-correlation ratio D2 are all held within 5 %. A classifier built on mass, τ21, multiplicity, girth, D2, C2 and τ32 shows no response (|Δ| ≤ 0.06 in log-odds), while one that also sees the pull angles and the between-prong energy fraction responds in the same direction. The networks therefore respond to where soft radiation sits relative to the two prongs (the colour-flow pattern), beyond the standard shape observables tested here.

Still uncertain:
- About a third to a half of the network response to the mirror also appears for the control mirror across the prong line.
- The predicted opposite response on QCD jets is not reproduced: its sign differs from H41's, and both are consistent with zero.
- With girth and D2 held within 2 %, the W-jet response is no longer resolved.

## What was checked in the code (interp/h41_colour_flow.py)

| check | finding |
|---|---|
| padded particles | `pkin` masks slots ≥ nn and pT = 0; soft selection and every observable use that mask |
| row sampling | random rows (`default_rng(seed).choice`, sorted), not first-k; Top arm via `h4_top_wmass._sample` (random per class) |
| axis | every variant scored about its own E-scheme axis; `jc_dict`/`part_core.build` use η = jet_eta + sign(jet_eta)·Δη and recompute jet_pt and jet_energy from the edited particles |
| constituent masses | taken from `base['pmass']`; for Mx/C1x/C2x scaled by the same factor as the soft four-vectors (consistent with the mass-exact claim) |
| observables vs model inputs | observables and network inputs use the same edited P4 arrays |
| working point | ε_S = 0.5 threshold = median s_c over unedited class-c jets, per model |
| Holm family | 6 (sample, score) cells × 4 nets × {M−C1, M−C2} = 48, as stated |

No coding error that changes the reported numbers was found. Design points that affect how the numbers read:

1. **C1 is not a null for the networks.** On their own, C1 and C2 lower the signal score.
   - In the H41 matched set, ParT_full W: M −0.67, C1 −0.33, C2 −0.45.
   - In this rerun, matched ParT_full: C1 −0.37 / −0.27 / −0.51 against M −0.62 / −0.91 / −1.04 (W / Z / H).
   - So 30–50 % of the M response is shared with an edit that changes f_btw and the pull angles by < 0.05. The reading reports only the differences M − C1 and M − C2.
2. **C2 is undersized and is not neutral to colour flow.**
   - Mean displacement: C2 0.167 against M 0.227 and C1 0.268, because the rotation is clipped at 2ρ.
   - C2 also changes f_btw by −0.10 on W (half of M's −0.19).
   - M − C2 is therefore a weaker contrast than M − C1. It does not on its own exclude a generic radial/shape reading.
3. **The mass-exact variant rescales soft pT.** The Mx scale factor spans 0.65–1.51 (5–95 %), so the soft-pT spectrum differs between Mx and C1x (C1x: 0.98–1.06).
4. **Response ratios.** The network-to-GB5 ratio (1.1–7) compares log-odds responses of classifiers with different AUC (0.97–0.98 vs 0.92–0.94). It is not a measure of how much each classifier uses the feature.

## Rerun and added controls

**Sample.** JetClass test WToQQ_100, ZToQQ_100, HToBB_100 and ZJetsToNuNu_100; 500 < pT < 1000 GeV, |η| < 2, ≥ 3 particles. 3,000 random rows per class (default_rng 4301–4304, disjoint from H41's 4101–4104); N = 12,000.

**Unedited 10-class accuracy on this sample:**

| model | accuracy | AUC W / Z / H vs QCD |
|---|---|---|
| ParT_full | 79.8 % | 0.979 / 0.978 / 0.999 |
| ParT_kin | 65.4 % | 0.971 / 0.965 / 0.984 |
| PN_full | 77.6 % | 0.976 / 0.974 / 0.999 |
| PN_kin | 60.4 % | 0.967 / 0.959 / 0.972 |
| ParT_perm | 0.1 % | 0.51 / 0.48 / 0.45 |

**Added control 1: shape confound.**
- Observables added for every variant: girth (Σ z_i ΔR_i about the E-scheme axis) and D2 and C2 (β = 1). τ32 was already computed.
- New matched sets: the loose m/τ21 match plus |Δgirth|/girth < r and |ΔD2|/D2 < r for each of M, C1 and C2, with r = 0.10, 0.05 and 0.02. One version (r = 0.05) also requires |Δτ32| < 0.03.
- New references, HistGradientBoosting on train_100M, 10,000 jets per class (rng 4311–4314):
  - GBS: (m, τ21, N, girth, D2, C2, τ32);
  - GBS5: GBS plus (θ_p1, θ_p2, f_btw);
  - GB3 and GB5 refit as in H41.
- Held-out AUC GB3 / GB5 / GBS / GBS5: W 0.917 / 0.925 / 0.939 / 0.944; H→bb 0.933 / 0.937 / 0.942 / 0.943.

**Added control 2: signed response vs generic disruption.**
- M_in mirrors only the soft particles with u_rel > 0 (on the side facing the other prong) to the far side. C1_in applies the across-line mirror to the same particles.
- M_out and C1_out do the same for the particles with u_rel < 0. These carry 47 % of the soft pT on average.
- Predictions on signal jets:
  - colour-flow or radial reading: M_in − C1_in < 0 and M_out − C1_out > 0;
  - generic sensitivity to moved particles: both < 0.

**Statistics.** Per-jet paired contrasts; 95 % bootstrap CI over jets (B = 2,000); two-sided sign-flip permutation p (5,000). Holm within each family (signal cells × 4 networks × contrasts).

## Numbers (Δ log p(c)/p(QCD), mean [95 % CI])

### Effect of M on observables (all jets, mean change; unedited mean in brackets)

| sample | f_btw | θ_p1 [rad] | D2 | girth | C2 | τ32 | m [GeV] |
|---|---|---|---|---|---|---|---|
| W | −0.190 (0.511) | +0.13 | −0.096 (1.94) | +0.0012 (0.113) | −0.006 (0.119) | +0.014 | −0.41 |
| QCD | −0.057 (0.283) | −0.03 | −0.252 (4.60) | −0.0003 (0.092) | −0.014 (0.168) | +0.013 | −2.85 |

On W, C1 changes each of these by ≤ 0.5 %, except τ32, which rises by +0.015 (about 2 %). M and C2 raise τ32 by about the same amount. On signal jets M lowers D2 by about 5 %, which is the direction a D2-based tagger scores as more two-prong (more signal-like). The observed score drop therefore has the opposite sign to the D2 change.

### Signal jets, M − C1 (signal score, own class)

| set (n W/Z/H) | ParT_full | ParT_kin | PN_full | PN_kin | ParT_perm | GBS | GB5 | GBS5 |
|---|---|---|---|---|---|---|---|---|
| matched, Δm<1, Δτ21<0.01 (188/146/94) | W −0.25 [−0.42,−0.08]; Z −0.64; H −0.53 | −0.26 / −0.48 / −0.45 | −0.21 / −0.61 / −0.17 [−0.41,+0.07] | −0.19 / −0.30 / −0.24 | +0.02 / +0.02 / +0.04 | +0.00 / +0.01 / −0.02 | −0.14 / −0.13 / −0.11 | −0.16 / −0.11 / −0.08 |
| loose, Δm<3, Δτ21<0.03 (1152/1101/862) | −0.54 / −0.77 / −0.94 | −0.42 / −0.47 / −0.44 | −0.40 / −0.52 / −0.57 | −0.32 / −0.35 / −0.22 | +0.01 / −0.00 / +0.02 | +0.06 / +0.05 / +0.04 | −0.12 / −0.17 / −0.13 | −0.06 / −0.08 / −0.03 |
| loose + girth, D2 within 5 % (590/540/492) | −0.27 [−0.36,−0.17] / −0.66 / −0.72 | −0.25 / −0.34 / −0.32 | −0.20 / −0.39 / −0.38 | −0.20 / −0.22 / −0.15 | +0.02 / +0.00 / +0.02 | +0.02 / −0.00 / +0.01 | −0.12 / −0.17 / −0.11 | −0.08 / −0.07 / −0.06 |
| same + τ32 within 0.03 (324/297/246) | −0.37 / −0.71 / −0.88 | −0.37 / −0.41 / −0.40 | −0.30 / −0.48 / −0.46 | −0.25 / −0.28 / −0.22 | +0.01 / −0.01 / +0.03 | +0.02 / −0.03 / −0.00 | −0.16 / −0.16 / −0.09 | −0.12 / −0.09 / −0.05 |
| loose + girth, D2 within 2 % (126/108/121) | −0.16 [−0.34,+0.03] / −0.49 [−0.69,−0.28] / −0.37 [−0.66,−0.09] | −0.10 / −0.19 / −0.18 | −0.03 / −0.24 / −0.17 | −0.08 / −0.09 / −0.01 | +0.02 / +0.02 / +0.04 | −0.03 / +0.01 / +0.00 | −0.10 / −0.09 / −0.11 | −0.12 / −0.06 / −0.03 |

Holm results:
- matched: 21 of 24 tests have p < 0.05. The exceptions are PN_full H→bb (M − C1 and M − C2) and PN_kin H→bb M − C2 (Holm 0.06).
- loose: 24 of 24 (Holm 0.0048).
- girth/D2 within 5 %: 23 of 24 (PN_kin H→bb M − C2, Holm 0.08).
- Girth/D2 within 2 % (not in a Holm family): the four W-jet CIs all include 0; 3 of 4 Z cells and 2 of 4 H cells exclude 0.

M − C2 with girth/D2 within 5 %: −0.07 to −0.44 for the networks, against GBS −0.00 to +0.03.

The H41 numbers in the matched set (−0.15 to −0.69) and the loose set (−0.19 to −0.96) fall in the same ranges.

### Signed test (girth and D2 within 5 %, loose m/τ21 match for the two edits)

| sample | M_in − C1_in, nets (n) | M_out − C1_out, nets (n) | ParT_perm | GBS | GB5 |
|---|---|---|---|---|---|
| W | −0.49 / −0.45 / −0.39 / −0.38 (839) | +0.35 / +0.36 / +0.37 / +0.31 (769) | +0.03 / −0.00 | +0.01 / +0.00 | −0.67 / +0.38 |
| Z | −0.70 / −0.51 / −0.47 / −0.40 (671) | +0.40 / +0.36 / +0.34 / +0.29 (671) | +0.02 / −0.00 | +0.01 / −0.02 | −0.57 / +0.31 |
| H→bb | −1.07 / −0.57 / −0.61 / −0.30 (349) | +0.54 / +0.38 / +0.42 / +0.26 (544) | +0.01 / +0.00 | −0.01 / +0.02 | −0.22 / +0.37 |
| QCD (W score) | +0.04 / +0.04 / +0.03 / +0.03 (1273) | +0.06 [−0.02,+0.14] / +0.13 / +0.12 / +0.12 (208) | +0.01 / −0.08 | +0.04 / −0.03 | −0.08 / +0.29 |

- Network order: ParT_full / ParT_kin / PN_full / PN_kin.
- Signal cells: 24 of 24 Holm p < 0.05 (Holm 0.0024 for each).
- The generic-disruption prediction (both contrasts negative) is excluded on signal jets. The sign of each half follows GB5.

### QCD jets, M − C1 (fresh sample)

- matched (n 125): −0.03 to −0.08 for every network and score; every CI includes 0. H41 gave +0.06 to +0.17.
- loose (n 723): −0.06 to +0.03; GB5 −0.02 to −0.06.
- In the signed test, M_out − C1_out is positive on QCD (+0.06 to +0.13), as the colour-flow prediction says. M_in − C1_in is also positive (+0.03 to +0.04), where the prediction (and GB5, −0.08) is negative.

### All-jet regression

- Intercept of M − C1 at zero (Δm, Δτ21, Δgirth, ΔD2) on signal jets: networks −0.44 to −1.26; GBS −0.03 to −0.05; GB5 −0.22 to −0.26.
- On QCD jets the m/τ21/N references give −0.09 to −0.13 under the same regression. Because the edit lowers the QCD mass by 2.9 GeV, the regression intercept on QCD jets is not a clean null (as H41 reported).

## Hypotheses and outcome

H0: the network M − C1 response equals that of a classifier on standard shape observables (GBS: m, τ21, N, girth, D2, C2, τ32) and lies within the permuted-network band.

H1 (colour-flow use): signal score falls when near-side soft radiation moves outside and rises for the reverse, beyond GBS.

Competing readings:

| reading | prediction | measured |
|---|---|---|
| axis artefact | M ≈ TN | H41: TN and M_st − M |Δ| ≤ 0.03; not rerun |
| generic disruption | M_in − C1_in and M_out − C1_out both < 0 | M_out − C1_out > 0 on signal (+0.26 to +0.54); excluded |
| D2/girth/C2/τ32 shape reading | response tracks GBS and vanishes at fixed girth and D2 | GBS ≈ 0; networks −0.15 to −0.72 at the 5 % match; excluded at that tolerance. At 2 %, W unresolved (n 126) |
| colour-flow reading | sign of GB5 on signal and QCD | signal: matches in every cell. QCD: not resolved; one half (M_in) has the opposite sign |

## Corrected reading

On W, Z and H→bb jets, mirroring soft particles (pT < 2 % of the jet pT) about their prong along the prong–prong line lowers the signal-vs-QCD log-odds of all four JetClass networks, relative to the same mirror across the line (C1).
- Size: −0.17 to −0.64 in the strict m/τ21 match (n 94–188, fresh sample, 21 of 24 Holm p < 0.05) and −0.15 to −0.72 with girth and D2 also held within 5 % (n 492–590).
- The two halves of the mirror have opposite signs: near-side to far-side −0.30 to −1.07, far-side to near-side +0.26 to +0.54. This excludes generic sensitivity to moved particles as the source of the difference.
- A boosted-tree classifier on (m, τ21, N, girth, D2, C2, τ32) does not respond (|Δ| ≤ 0.06); the permuted network stays within ±0.04. A classifier that adds the pull angles and f_btw responds with the same sign.
- C1 and C2 alone also lower the network scores (ParT_full −0.27 to −0.51), so roughly half of the raw M response is shared with direction-neutral moves.
- With girth and D2 within 2 %, the W-jet difference is not resolved (n 126).
- The predicted rise on QCD jets is not resolved: −0.03 to −0.08 here against +0.06 to +0.17 in H41, every CI including 0.
- The comparison of response sizes with the pull/f_btw classifier is not a measure of relative use, because the classifiers differ in AUC.
- The Top arm was not rerun.

## Explainer entry (verification step, for site/explain.py under H41)

- **Question.** Is the networks' response to moving soft radiation between and outside the prongs explained by standard shape variables or by generic sensitivity to moved particles?
- **Background.** Moving soft particles away from the other prong also changes girth $g=\sum_i z_i\Delta R_{i,J}$ and the energy-correlation ratios $D_2=e_3/e_2^3$ and $C_2=e_3/e_2^2$, with $e_2=\sum_{i<j}z_iz_j\theta_{ij}$ and $e_3=\sum_{i<j<k}z_iz_jz_k\theta_{ij}\theta_{ik}\theta_{jk}$.
- **Method.** Fresh JetClass sample, 3,000 per class.
  - Jets are required to keep $m$, $\tau_{21}$, $g$ and $D_2$ within tolerance under every edit.
  - A tree classifier on $(m,\tau_{21},N,g,D_2,C_2,\tau_{32})$ serves as the shape reference.
  - The mirror is split into a near-to-far half and a far-to-near half, each compared with the across-the-line mirror of the same particles.
- **Result.** With $g$ and $D_2$ within 5 %:
  - $M-C1=-0.15$ to $-0.72$ for the networks and $\approx0$ for the shape classifier;
  - near-to-far gives $-0.30$ to $-1.07$ and far-to-near gives $+0.26$ to $+0.54$.
  - At 2 %, the W-jet difference is not resolved.
- **Checks.** Permuted-weight ParT within $\pm0.04$. Unedited accuracies 79.8 / 65.4 / 77.6 / 60.4 %.
- **Physical significance.** The taggers' score depends on the side of each prong on which soft radiation sits, at fixed mass, $\tau_{21}$, girth and $D_2$. This is the colour-connection pattern of a colour-singlet decay. On QCD jets the corresponding rise was not measured with enough precision.
