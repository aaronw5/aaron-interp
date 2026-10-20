# H24 verify — Pythia- vs Herwig-trained quark–gluon taggers: fresh-sample rerun, near-collinear split, on-manifold multi-particle addition, epoch spread

Script `interp/h24_verify.py` (imports `interp/h24_qg_generator_models.py` unchanged; sets its module SEED to 7243 so sample rows
and edit draws are new; adds three controls). Jobs: `anrunw-ai-d-h24-verify` (GPU, GTX 1080 Ti, 40 min, exit 0),
`anrunw-ai-d-h24-verify-k`, `anrunw-ai-d-h24-verify-k2` (1 CPU / 2 GiB, per-k summaries with bootstrap CIs, exit 0).
Results: `results/physics/h24_verify.json` (all metrics, paired contrasts, epoch tests, Holm), `h24_verify_k.json` (per-k means,
addk / adsw slopes with CIs, observables per k), `h24_verify_models.json` (checkpoint paths, C2ST), `h24_verify.png`;
per-jet logits `h24_verify_logits_{py,hw}.npz` on the volume only. Logs `logs/d-h24-verify*.log`.

## Conclusion
ParT and ParticleNet barely move toward "gluon" when soft particles are added in ways that change only the particle count (exact
or near-collinear splits, or extra soft particles at typical positions), whether they were trained on Pythia or on Herwig: at most
about a tenth of the move made by a classifier that uses only the count, and the Herwig-trained networks move slightly toward
"quark". Their different outputs on Pythia and Herwig gluon jets therefore do not come mainly from the generators' different
multiplicities. The Herwig-trained networks respond 0.01–0.03 log-odds per added particle lower than the Pythia-trained ones,
which is more than the change between saved checkpoints of one training; whether it is more than the difference between two
independent trainings is not measured. Removing a jet's softest particles while keeping the count fixed moves all four networks
toward "quark", so the networks do use the soft particles' momenta and positions, not their number.

## Code check (`interp/h24_qg_generator_models.py`)
- Rows: Pythia `data_qg.sample` (random rows per class, sorted); Herwig `H14.load_sample` (random rows per class from files 18–19).
  No first-k subsampling. Lists are pT-sorted, so `a[-1]` is the softest particle as the edits assume.
- Inputs: every edited list goes through `from_list` → `kinematics` (E-scheme axis of the edited jet, η sign convention as the
  QG conversion) → `weaver_inputs`. Observables (`basic_obs`) and the references (EFPs, N) are computed from the same edited lists.
  Padded slots are masked (`real = pT > 0`); N counts real particles only.
- Edits: `split_set` conserves every four-momentum exactly (z = ½, ΔR = 0; Δgirth −4e-12, Δmass 0 in the log). `ctrl_set` keeps
  N, ΣpT and positions and matches ΣpT² to the split; it moves momentum between particles at different positions, so mass
  changes by +0.025 GeV per k (reported in H24). `restore_s2` and `merge_soft` behave as described. No conservation error found.
- `weaver_inputs` keeps the 128 hardest particles (L_IN = 128); a particle added to a jet with N ≥ 128 is dropped. Gluon N is
  53 ± 16, so this affects a negligible number of jets.
- Statistics: the 20 000-vector sign-flip test and the Holm list are as described. The H1 "seed spread" uses the ParT − PN
  architecture spread and the PFN two-run spread; neither is a same-architecture replicate (stated in H24 Issues).
- No bug found that changes a number.

## Fresh-sample rerun (H24 edits unchanged)
Sample: 2000 quark + 2000 gluon jets per generator, seed 7243 (H24: 4000 + 4000, seed 9641); 147 of the 4000 Pythia rows are also
in the H24 sample. Split family on 3992 (Py) and 3996 (Hw) jets with N ≥ 10.

Unedited AUC / accuracy, Pythia jets ; Herwig jets (H24 in brackets):
- Py ParT_kin 0.900 / 0.822 ; 0.815 / 0.725 (0.904 / 0.833 ; 0.815 / 0.721)
- Py PN_kin 0.899 / 0.824 ; 0.815 / 0.720 (0.905 / 0.832 ; 0.817 / 0.720)
- Hw ParT_kin 0.881 / 0.793 ; 0.842 / 0.767 (0.888 / 0.803 ; 0.836 / 0.760)
- Hw PN_kin 0.876 / 0.784 ; 0.840 / 0.765 (0.884 / 0.796 ; 0.838 / 0.762)
- permuted-weight ParT_kin 0.544 / 0.499 ; 0.521 / 0.497

Δlogit per added particle (logit = log p_g − log p_q), mean [95 % bootstrap CI over jets], Pythia jets ; Herwig jets (H24):
| scorer | sp − ct slope | add_soft − add_swap |
|---|---|---|
| Py ParT_kin | +0.0079 [0.0069, 0.0088] ; +0.0096 [0.0086, 0.0107] (+0.0083 ; +0.0091) | +0.0041 ; +0.0075 (+0.0045 ; +0.0074) |
| Py PN_kin | +0.0042 ; +0.0071 (+0.0045 ; +0.0058) | −0.0031 ; +0.0022 (−0.0026 ; +0.0022) |
| Hw ParT_kin | −0.0048 [−0.0056, −0.0040] ; +0.0009 [0.0001, 0.0018] (−0.0041 ; +0.0013) | −0.0174 ; −0.0100 (−0.0168 ; −0.0102) |
| Hw PN_kin | −0.0082 ; −0.0019 (−0.0083 ; −0.0014) | −0.0312 ; −0.0224 (−0.0304 ; −0.0212) |
| ParT_kin, permuted | −0.0056 ; −0.0058 (−0.0055 ; −0.0056) | −0.0103 ; −0.0112 (−0.011 ; −0.011) |
| LR(N) Pythia / Herwig | +0.101 / +0.078 ; +0.104 / +0.081 | +0.108 / +0.085 ; +0.111 / +0.088 |

Herwig-trained − Pythia-trained, paired per jet (sign-flip p = 5 × 10⁻⁵ for all): ParT sp − ct −0.0127 [−0.0137, −0.0117] ;
−0.0088 (H24 −0.0124 ; −0.0077); PN −0.0124 ; −0.0090 (−0.0128 ; −0.0072); ParT add − swap −0.0215 ; −0.0175 (−0.0213 ; −0.0176);
PN −0.0282 ; −0.0246 (−0.0279 ; −0.0234).

Swap on Herwig gluon jets (N +4.48 at fixed pT^D): Py ParT +0.063 (H24 +0.065), Hw ParT −0.0024 (−0.003); fraction of each model's
Pythia − Herwig gluon logit gap: Py ParT 6.0 %, Py PN 4.5 %, Hw ParT −0.5 %, Hw PN −4.5 % (H24: 6, 4, −1, −4 %). GBDT(EFP+N) and
PFN were not rerun.

All H24 network numbers that were rerun come back within the bootstrap intervals or within 0.002.

## Added controls
**1. Near-collinear split (nsp_k).** The same k particles as sp_k, split at z = ½ with the daughters at ±0.005 from the parent
position in a random direction (ΔR = 0.01, no zero-distance pair). C2ST AUC for k = 1: 0.527 [0.503, 0.551] (Py), 0.517 (Hw);
exact split 1.000. Δmass +0.002 GeV and Δgirth +2.5 × 10⁻⁶ at k = 5.
- nsp − ct slope: Py ParT +0.0079 [0.0070, 0.0088] ; +0.0096; Py PN +0.0045 ; +0.0074; Hw ParT −0.0050 ; +0.0007; Hw PN −0.0082 ;
  −0.0016; LR(N) unchanged. Equal to the exact-split values to ≤ 0.0003 for every model.
- H0 (the sp response is an artefact of ΔR = 0 pairs, which ParT's ln ΔR pair feature puts far off its training range) predicts
  nsp ≠ sp. Observed nsp = sp. H0 excluded.

**2. On-manifold addition of k soft particles (addk_k, k = 1…5)**, each with the jet's softest pT at a random particle's position
+ 0.05, with no particle removed. ΔN = +k; per particle Δmass +0.033 (Py) / +0.036 (Hw) GeV, ΔpT^D −1 × 10⁻⁴, Δgirth +3 × 10⁻⁵.
C2ST 0.53 (k = 1), 0.65 (k = 5; N itself is a C2ST feature). Per-jet least-squares slope over k, [95 % CI]:
| scorer | Pythia jets | Herwig jets |
|---|---|---|
| Py ParT_kin | +0.0018 [0.0010, 0.0025] | +0.0042 [0.0034, 0.0050] |
| Py PN_kin | −0.0056 [−0.0069, −0.0043] | +0.0009 [−0.0004, 0.0022] |
| Hw ParT_kin | −0.0139 [−0.0147, −0.0131] | −0.0080 [−0.0088, −0.0073] |
| Hw PN_kin | −0.0345 [−0.0361, −0.0327] | −0.0245 [−0.0260, −0.0229] |
| ParT_kin, permuted | −0.0175 | −0.0178 |
| LR(N) Pythia / Herwig | +0.101 / +0.078 | +0.104 / +0.081 |
Subtracting the ct slope (matched ΔpT^D, ΔN = 0) changes these by ≤ 0.003. Interpolated to the generator gap (4.48 particles), the
Py-trained ParT moves Herwig gluon jets by about +0.02, about 2 % of its 1.05 gluon logit gap. The H0 of H24 (no counting) holds on
an on-manifold multi-particle edit.

**3. The adsw_k control (k added particles, k softest original particles removed; ΔN = 0)** moves all four networks toward quark
by −0.045 to −0.056 per particle on both generators (permuted null −0.020; LR(N) 0). It also changes the jet: per particle at k = 5,
Δmass −0.19 GeV, ΔpT^D +2 × 10⁻⁴, Δgirth −1.1 × 10⁻⁴. So addk − adsw (slope +0.020 to +0.061, 25–60 % of LR(N)) is not a count
response: it is the response to removing the softest particles, whose mass and girth contribution is not fixed. The same mixture is
present at k = 1 in H24's add_soft − add_swap (Δmass −0.05 GeV from the removal); there the removal effect is small (Py ParT
add_swap −0.009).

**4. Epoch spread.** Each of the four trainings evaluated at its saved checkpoints of epochs 12–19 (best epochs 14, 17, 12, 19 are
in this range), on the same jets, k = 1, 3, 5. Within-run SD over the 8 checkpoints: 0.001–0.004 per particle (sp − ct, add − swap,
addk − adsw). Herwig-trained − Pythia-trained, mean over checkpoints:
| contrast | Pythia jets | Herwig jets |
|---|---|---|
| ParT sp − ct | −0.0136 (SD Hw 0.0020, Py 0.0011) | −0.0096 |
| PN sp − ct | −0.0145 (SD 0.0023, 0.0043) | −0.0110 |
| ParT add − swap | −0.0245 | −0.0200 |
| PN add − swap | −0.0306 | −0.0266 |
The Herwig-trained and Pythia-trained checkpoint ranges do not overlap in any of the 12 contrasts (2 architectures × 3 metrics × 2
generators). Exact permutation test over the 16 checkpoint values: p = 1.6 × 10⁻⁴ (the minimum, 2/12 870) in all 12. Checkpoints of
one run are correlated, so this spread is a lower bound on the spread between independent trainings (different seeds), which is
still not measured.

**Holm** over 98 tests (5 metrics × 7 scorers × 2 generators, 8 paired contrasts × 2, 6 epoch tests × 2): 93 below 0.05. Not below:
Hw ParT sp − ct and nsp − ct on Herwig jets (Holm 0.13, 0.22), Hw PN nsp − ct on Herwig jets (0.059), Py PN add − swap on Herwig
jets (0.11), Hw ParT on-manifold swap on Pythia jets (0.34).

## Hypotheses
| hypothesis | prediction | observed | status |
|---|---|---|---|
| H0: ParT/PN of either generator do not count (D7) | count response ≪ LR(N) for exact split, near-collinear split, single and k-fold on-manifold addition | toward-gluon response ≤ 0.010 per particle against LR(N) 0.078–0.104 for all four edits | consistent |
| ΔR = 0 artefact | nsp ≠ sp | equal to ≤ 0.0003 | excluded |
| Generator shift is within-run checkpoint noise | Hw − Py within checkpoint spread | 3–10 × the within-run SD, ranges disjoint, p = 1.6e-4 | excluded (seed spread untested) |
| Networks ignore soft particles altogether | adsw ≈ 0 | −0.045 to −0.056 per particle | excluded |

## Assessment of the H24 reading
- Reproduced on a fresh sample: unedited AUCs within 0.008, every rerun network response within its CI or 0.002, all eight paired
  Herwig − Pythia contrasts with the same size and sign.
- Two of the H24 open issues are resolved: the exact-split response is not a ΔR = 0 artefact (near-collinear split gives the same
  numbers, C2ST 0.52–0.53), and an on-manifold k-particle addition also gives ≤ 0.004 per particle toward gluon.
- The generator shift exceeds within-run checkpoint variation; the seed spread is still open, as H24 states.
- Wording: "at most a tenth" of LR(N) holds in the gluon direction. In the quark direction Hw PN_kin moves up to −0.031 (add − swap)
  and −0.035 (addk), about a third of LR(N) in size and opposite in sign; the permuted null moves −0.010 to −0.018. "Barely react"
  should be read as "no gluon-ward count response", not "no response".
- add − swap is not at fixed IRC-safe content: removing the softest particle moves the networks, and with 5 removals the move is
  −0.05 per particle. The H24 phrase "at fixed IRC-safe content" applies strictly to sp − ct and nsp − ct.
- Not rerun: PFN/EFN, GBDT(EFP+N) (65–69 % swap fraction), the Pythia-jet merge swap.
