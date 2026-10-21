# H55 verify (skeptic): classical-tagger fingerprint panel

## Conclusion
On JetClass (W, Z and top versus QCD) and on the Top Tagging Landscape, every trained network changes its calls under the nine edits in a pattern closer to a groomed-mass-plus-substructure tagger than to an ungroomed mass window. This is not true on hls4ml: all five hls4ml networks are closest to the plain mass window for Z versus QCD, and JEDI-linear and the Linformer are closest to it for top versus QCD. The panel has no groomed-mass-only tagger, so it cannot tell "uses the soft-drop mass" apart from "uses τ21, D2 or τ32 as well". In JetClass the networks change about 10 % of their calls when the axis their inputs are measured about moves. One classical tagger in the panel (the girth likelihood ratio) also responds to this axis change, so it is not a property of the networks alone.

## Status of the rerun
- Fresh-sample rerun script: `interp/h55_verify.py`. It reruns H55's own code on hls4ml `jedi_core.sample(2000, 5591)` and on JetClass with 2,000 random rows per class (`default_rng 5592`), then adds the controls listed below.
- Job `anrunw-ai-d-h55-verify` (GPU) was created twice on node k8s-chase-ci-01. Both times the kubelet refused it with `UnexpectedAdmissionError ... GPU is lost`, a node fault.
- It was then created a third time with that node excluded, from a scratchpad copy of part_job.sh: `/private/tmp/claude-501/-Users-anrunw-Documents-distill-part/97edd5f4-0297-498c-b0c4-c84a470a6b9f/scratchpad/part_job_x.sh`.
- After that, the session's permission classifier blocked all further cluster commands and local scripts. The job could not be monitored, and the follow-up jobs were not launched: `d-h55-verify-jedi` (JEDI-linear in the jax env) and `d-h55-verify-stats` (`--part stats` and `--part extra`).
- **No fresh-sample numbers exist yet.** The job ends by itself (12 h deadline at most) and writes to `/j-jepa-vol/aaron-interp/results/physics/h55_verify_work/`.
- To finish:
  - `./job.sh d-h55-verify-jedi 1 4 'python -m interp.h55_verify --part nets --ds hls_jedi'`
  - `AI_PY=conda ./job.sh d-h55-verify-stats 2 8 'python -m interp.h55_verify --part stats && python -m interp.h55_verify --part extra'`
  - These produce `results/physics/h55_verify.json`.
- Everything below comes from reading the code, `results/physics/h55_fingerprint.json`, the `h55_*_tab.json` files, the job logs and the two figures. It describes the original sample (hls4ml seed 5501, JetClass 5502, Top 5503, QG 5504).

## Definitions
- **chg** = fraction of jets whose call (signal or background) flips under an edit.
- **flow** = (calls gained − calls lost) / jets.
- **Fingerprint** = the vector of (chg, flow) over the edits. Each component is divided by its root-mean-square over the trained networks and classical taggers of that task.
- **cos** = cosine similarity between two fingerprints.
- **Permuted copy** = the same architecture with each weight tensor's entries randomly shuffled. It is the null model.
- **m_SD** = soft-drop groomed mass (β = 0, z_cut = 0.1).
- **τ21, τ32** = N-subjettiness ratios.
- **D2** = energy-correlation ratio e3/e2³.
- **WTA** = winner-take-all axis.
- **Holm** = Holm–Bonferroni correction for multiple tests.
- **Rank p** = (1 + number of permuted copies with cos ≥ the network's cos) / (number of copies + 1).

## Code check (interp/h55_classical_fingerprint.py)
| item | result |
|---|---|
| Row sampling | Random indices in every dataset (`rng.choice`, `_rows`). No first-k rows. |
| Exact edits | Stored max \|Δ\| over ok jets is 0 for m, m_SD, τ21, τ32 and D2 under split and ksplit3, and 0 for N under wta/ctrl. Under decay, m is held to ≤ 0.0065 GeV (JetClass) and N to 0. Under zboost, m ≤ 0.022 GeV and pT 0. These match the claims. |
| Observables on the same jets the networks see | Yes: one finalized table per condition, shared by the observables and the networks. Exceptions: Top PFN/EFN see only the first 128 of 200 slots, and hls4ml jets with 64 particles are excluded from the count edits (22–25 %). Both are stated in the report. |
| Working point | Networks are cut at the median log-odds of unedited signal (50 % signal efficiency). Classical taggers use fixed cuts with signal efficiency 0.35–0.94 (mass_V 0.84–0.94). The fingerprint shapes are not compared at matched efficiency; `h55_verify.py --part extra (c)` adds that comparison. |
| Sign of Δη in the JetClass `prep` | `sg` is recomputed from the edited axis. For jets with \|η_jet\| smaller than the axis offset (median offset 0.085), wta/ctrl flip the sign of the network's Δη input, which reflects the jet in η. This affects wta and ctrl alike, for a small fraction of jets. Not fixed. |
| soft edit | Conserves the jet three-momentum, not its energy (the added particle is massless and the hardest particle keeps its mass). The docstring says so. |
| zboost on hls4ml | τ21 changes by up to 0.078 on a few jets: exclusive-kT axis ties at float32 precision. The classical calls are unchanged (0.0 %). |

## Problems with the reading as stated
1. **"On W/Z/top tasks every network's nearest tagger is an m_SD + shape tagger, not the plain mass window" is false on hls4ml.** Nearest tagger by cosine, with the fraction of bootstrap draws in which it is nearest:
   - hls4ml Z versus QCD: mass_V for all five networks. JEDI-linear 0.43 vs V_DDT 0.38 (bootstrap 0.62); dsets 0.42 vs 0.39; linf 0.44 vs 0.41; tf 0.39 vs 0.34; jedis 0.45 vs 0.44.
   - hls4ml t versus QCD: mass_top for JEDI-linear (0.28 vs τ32 0.10, bootstrap 0.66) and for linf (0.16 vs 0.11).
   - hls4ml W versus QCD: V_DDT is nearest, but V_DDT and V_tau21 are tied within the CI (0.35 vs 0.35, bootstrap 0.46–0.61).
   - The worker's own thesis note says JEDI-linear is nearest the mass window for Z, which contradicts the reading.
   - The statement holds on JetClass W, Z and top (all four networks, bootstrap ≥ 0.97) and on the Top Landscape (all five networks, bootstrap 1.0).
2. **Groomed mass and shape are not separated.**
   - Every "shape" tagger contains the m_SD window. No m_SD-only tagger is in the panel.
   - Classical taggers have chg = 0 on split, ksplit3, wta, ctrl and zboost by construction. So the shape-versus-mass ranking rests on the soft, gin, gout and decay columns.
   - Under decay, the ungroomed mass is fixed (mass_V chg = 0), but m_SD falls by 17 GeV on average on JetClass. In the edit figure, one W jet's m_SD goes from 83 to 40 GeV. An m_SD window alone would therefore also respond to decay and to gin but not to gout.
   - So the measured preference is "groomed-mass-like over ungroomed-mass-like". τ21, D2 or τ32 use is not shown.
   - The missing control is an m_SD-only tagger, 65–105 GeV and 105–210 GeV. It is cheap to add from the stored `h55_*_obs.npz`.
3. **"Shape over mass" is never tested directly.** Holm is applied to cos(network, tagger) against the permuted-copy null, one tagger at a time. Ten of the 36 Holm-significant cells are mass_V cells: JetClass Z for all four networks, and hls4ml Z for JEDI-linear, dsets, linf and jedis. On JetClass Z the plain mass window is also above the null (cos 0.45–0.57 vs permuted 0.23). The direct test is a paired bootstrap of cos(network, best shape tagger) − cos(network, mass window). It is in `h55_verify.py (b)`; no numbers yet.
4. **The p-values come from a normal approximation fitted to 6–30 permuted copies.**
   - The rank p (exact, no normal approximation) floors at 1/13 = 0.077 for JetClass (12 copies), 1/25 for Top and QG, and 1/31 for hls4ml. None of these passes Holm over 130 cells.
   - Values such as p = 1.6e-14 are extrapolations of a normal tail fitted to 12 points.
   - On JetClass W, V_tau21 is still far outside the permuted range: network cos 0.40–0.43 vs permuted 0.09–0.22, rank 12/12 for every network. The qualitative separation from the null holds there; the quoted p-values do not.
5. **Degenerate permuted copies are included in the null.**
   - All six JEDI-linear permuted copies have accuracy 0.2000. In five or six of them, depending on the task, the call never changes, so the fingerprint is all zeros and the cosine is 0 by construction.
   - The hls4ml null pool has 30 copies: 5–6 are degenerate.
   - Removing them raises the null mean by about 0.02–0.06. Examples: hls4ml W V_tau21 0.13 → 0.16; hls4ml q/g LR_N 0.29 → 0.35, LR_girth 0.38 → 0.46.
   - On JetClass Hbb, all 12 copies are degenerate. That task has no classical tagger, so it does not enter.
6. **"8–20 % under input-axis shifts" does not hold on QG.**
   - QG networks change 1.1 % (part), 2.3 % (pnet), 4.4 % (pfn) and 0 % (LorentzNet) under wta; only efn reaches 18 %.
   - **"WTA response mostly matches a random shift":** 14 of 46 McNemar tests differ after Holm, in both directions. JetClass W ParT_full changes 9.6 % under wta vs 12.5 % under ctrl (31 vs 145 discordant jets, p = 8e-19), and ParT_kin 10.7 vs 12.5 % (p = 6e-8). QG efn changes 18.0 vs 13.4 % (p = 5e-15).
7. **"Components no classical tagger in the panel has" is false.**
   - LR(girth) changes 12 % (QG) and 7 % (hls4ml) of calls under wta, and 11 % and 7 % under ctrl, because girth is measured about the axis.
   - LR(N) and LR(N, pT^D) change 4–10 % under split and ksplit3.
   - The statement holds only for the IRC-safe mass and shape taggers, where it is true by construction.
8. **Permuted copies change more calls than trained networks under every edit.** JetClass W permuted median: split 10 %, ksplit3 17 %, decay 25 %, zboost 12 %; trained networks: 2–5 %, 3–5 %, 13–15 %, 3–5 %. The cosine removes the scale, so the null controls the pattern, not the magnitude. A 1–5 % split response in a trained network is therefore smaller than in a random network with the same inputs.
9. **zboost.** "These networks read particle energies" follows from the input design: ParT and PN take log E and log E/E_jet as inputs. The JetClass permuted copies also change 12 %. Top LorentzNet changes 2.7 %, consistent with the beam reference particles that break boost invariance. The result shows the input design, not something the networks learned.

## Statistics summary (original sample)
- **Effect:** cos(network, tagger), with 95 % bootstrap CI over jets, B = 1000.
- **Null:** permuted copies, mean and range.
- **H0:** the network's cos ≤ the permuted mean.
- **Valid exact p:** the rank p, floored at 0.077 for JetClass.
- **Competing hypotheses for the nearest tagger, W/Z tasks:**
  - H_mass (ungroomed mass) predicts mass_V nearest: seen on hls4ml Z.
  - H_groomed (m_SD-like, with or without shape) predicts a V_* tagger nearest: seen on JetClass W/Z, hls4ml W (tied) and Top.
  - H_shape versus H_groomed-only cannot be separated: no m_SD-only tagger.

## Figure check
- `h55_heatmap.png` matches the JSON chg values. It omits the Z tasks, where hls4ml networks are nearest the mass window.
- `h55_edits.png` shows the edits on one JetClass W jet as described. Its decay panel shows the m_SD drop of 43 GeV discussed in point 2.

## Corrected reading
- Classical calls computed on the edited jets reproduce the expected invariances:
  - IRC-safe taggers: 0 % changed under exact collinear splits.
  - Mass window: 0 % under the fixed-four-momentum decay rotation.
  - LR(N): 0 % under axis changes.
  - hls4ml networks: 0 % under the beam boost (true by construction).
- **JetClass (W, Z, top) and Top Landscape:**
  - Every network's nearest tagger is the m_SD + τ21 / D2 / τ32 tagger.
  - On JetClass W the plain mass window is at the permuted-copy null (rank 4–11 of 12). On JetClass Z it is above the null but below the m_SD taggers.
  - The panel has no m_SD-only tagger, so the preference shows groomed-mass-like behaviour; it does not show use of τ21, D2 or τ32.
- **hls4ml:**
  - All five networks are nearest the plain mass window for Z.
  - JEDI-linear and Linformer are nearest it for top.
  - For W, V_DDT and V_tau21 tie.
- **QG:** part, pnet, pfn and LorentzNet are nearest LR(N, pT^D); only pfn is above the null after Holm. efn is nearest LR(girth), at the null.
- **Collinear splits:** networks change 1–5 % of calls (EFN 0); permuted copies change about 3–10 % (medians).
- **Axis changes:**
  - Networks change about 8–20 % on JetClass, Top and hls4ml, but 0–4 % on QG (efn 18 %).
  - WTA versus a random shift of equal length differs in 14 of 46 cells, in both directions.
  - LR(girth) also responds to axis changes.
- **Significance:** the reported Holm p-values rest on a normal approximation to 6–30 permuted copies. Exact rank p-values cannot pass Holm at this number of copies.

## Files
- `/Users/anrunw/Documents/aaron-interp/interp/h55_verify.py` (rerun and extra controls; not yet run to completion)
- Cluster job `anrunw-ai-d-h55-verify` (outputs: `/j-jepa-vol/aaron-interp/results/physics/h55_verify_work/`)
