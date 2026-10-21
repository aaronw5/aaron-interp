# H58 verify (skeptic): matched-50 % classical-tagger fingerprint, fresh sample and missing controls

## Conclusion
The H58 numbers reproduce on fresh jets: on top-quark tasks the networks' call changes are closer to a soft-drop (groomed) mass window than to an ungroomed mass window. That preference comes almost entirely from one edit, the decay-angle rotation. This edit keeps the jet mass fixed and changes how momentum is shared between the two prongs. With it removed, the groomed-versus-ungroomed difference is about 0 (−0.01 to +0.03 in cosine). The direct test of grooming adds soft particles that soft drop either keeps or removes. Top networks change their calls about half as often when the particles are removed (ratio 0.40–0.60), so they do not ignore groomed-away radiation. On JetClass W/Z they change calls 0.74–0.99 times as often, and on hls4ml W/Z about equally (1.00). When the τ32 tagger is compared with a tagger that has the same mass window and the same efficiency but no shape cut, the τ32 cut does add similarity. H58's "τ32 adds nothing" result is a cancellation between the τ cut and the wider mass window. Still uncertain: whether the networks compute τ32 itself or another prong-sharing quantity that responds to the same edits.

## Definitions (used below)
- **Edits** (H55, unchanged): split (hardest particle split into two collinear halves); ksplit3 (three soft particles split); soft (one 0.1 %-pT particle added at R/2–R); wta (inputs measured about the winner-take-all axis); ctrl (axis moved by the same distance in a random direction); **gin / gout** (three soft particles with 2 % of jet pT that soft drop keeps / removes, matched in ungroomed-mass change); **decay** (jet rest-frame rotation raising |cos θ*| of the two-prong axis by 0.3; jet four-momentum, mass and N fixed); zboost (beam boost Δy = 0.5).
- **chg** = fraction of jets whose call flips; **flow** = (gained − lost)/jets. **Fingerprint** = (chg, flow) over edits, each component divided by its RMS over the trained networks and the @50 classical taggers (H58 scaling, kept here). **cos** = cosine similarity of two fingerprints.
- **@50** = cut set to keep 50 % of unedited signal jets. **m_SD** = soft-drop mass (β = 0, z_cut = 0.1). **τ21, τ32** = N-subjettiness ratios.
- **d_shape** = cos(net, shape@50) − cos(net, msd@50); **d_groom** = cos(net, msd@50) − cos(net, mass@50); **d_sm** = their sum (H58 definitions).
- **New control tagger msdpub_rand@50**: the same published m_SD window as the shape@50 tagger (65–105 / 105–210 GeV) and a per-jet random number u_j < q. The random number is fixed across all edits, and q is set for 50 % signal efficiency. It has the shape tagger's window and efficiency, but its second cut cannot be moved by any edit.
  - **d_tau** = cos(net, shape@50) − cos(net, msdpub_rand@50): the part of the similarity due to the τ cut at fixed window and efficiency.
  - **d_width** = cos(net, msdpub_rand@50) − cos(net, msd@50): the part due to the wider window.
  - d_shape = d_tau + d_width.
- **r_g = chg(gout)/chg(gin)** on the matched gin/gout jets. An m_SD window gives r_g = 0 by construction, an ungroomed-m window r_g = 1.
- **Holm** = Holm–Bonferroni correction. **Leave-one-edit-out** = the contrast recomputed without that edit's two columns, with the scaling recomputed.

## Code check (interp/h58_fingerprint_rerun.py)
| item | finding |
|---|---|
| Row selection | Random indices everywhere (`rng.choice`, explicit exclusion of H55/H55v rows); no first-k. JetClass from `*_102.root`. |
| Axis / coordinates | JetClass `jc_prep`: η = ax_eta + sign·Δη via part_core.build; inherited from H55 (checked in H55_verify; the small sign flip under wta/ctrl for jets with \|η\| < offset remains). |
| Same jets for networks and observables | Yes: one finalized table per edit, read by both (H55 code). Top PFN/EFN see 128 of 200 slots (as H55). |
| Conservation | Unchanged H55 edits; split/ksplit3 exact, decay fixes m, zboost fixes m and pT (verified in H55_verify). |
| Working point | Networks and every @50 tagger at 50 % signal efficiency on the task sample. **But shape@50 and msd@50 differ in two ways**: shape@50 = published wide window (holding 69–87 % of signal) + τ cut, msd@50 = narrow 50 % window. d_shape therefore mixes the τ cut with the window width. This is the main design problem (see below). |
| Snap | Post hoc, sub-0.05 GeV / 1e-4 only; changes are as stated. |
| Leave-one-edit-out reporting | Correct as sign counts, but the report omits that dropping **decay** reduces every top-task d_groom from +0.08–0.18 to −0.02–+0.02 (H58's own JSON). |
| Bugs | None found that change the reported numbers. |

## Rerun
- **Script:** `interp/h58_verify.py`.
- **Job:** `d-h58-verify` (GPU; launched from the node-excluding copy of part_job.sh, as H58).
- **Fresh samples:**
  - JetClass test_20M `*_103.root` (H55/H55v used `_100`, H58 used `_102`): Hbb, Wqq, Zqq, Tbqq, QCD, 3,000 random rows each, `default_rng 5892`.
  - Top Landscape test split: 3,000 top + 3,000 QCD, `default_rng 5893`, with the H55 and H58 rows excluded (asserted).
  - Edits, observables, taggers, snap and network calls: unchanged H55/H58 code. Matched gin/gout fraction: 0.737 (JetClass), 0.635 (Top).
- **Extra analysis** on the fresh sample and on H58's stored sample (hls4ml, JetClass, Top), trained networks only:
  - paired jet bootstrap, B = 20,000, with the scaling recomputed per draw;
  - two-sided bootstrap p;
  - Holm within each contrast family over 49 network × task cells (p floor 1e-4, Holm floor 0.0049).
- **Not rerun:** permuted copies (not needed for these contrasts); hls4ml on fresh rows (controls run on H58's hls4ml sample only).

Unedited accuracy on the fresh sample:
- **JetClass ParT_full** per class QCD / Hbb / Zqq / Wqq / Tbqq: 0.785 / 0.924 / 0.690 / 0.799 / 0.957. Overall 10-class argmax over the five classes: ParT_full 0.831, ParT_kin 0.704, PN_full 0.813, PN_kin 0.667.
- **Top** accuracy (AUC): ParT_kin 0.939 (0.986), PN_kin 0.941 (0.985), PFN 0.929 (0.980), EFN 0.926 (0.976), LorentzNet 0.941 (0.987).
- All are within 0.01 of H58.

## Numbers
### 1. Reproduction of H58 main contrasts (fresh sample, d [Holm p])
- **JetClass top, d_groom:**
  - ParT_full +0.168 (0.005), PN_full +0.133 (0.005);
  - ParT_kin +0.098 (0.066), PN_kin +0.085 (0.14).
  - H58 values: +0.183, +0.151, +0.103, +0.107.
- **Top Landscape, d_groom:**
  - ParT_kin +0.125 (0.010), PN_kin +0.117 (0.005), LorentzNet +0.120 (0.005), PFN +0.151 (0.015);
  - EFN −0.132 (0.005).
- **EFN d_shape:** +0.269 (H58 +0.274).
- **JetClass Z, d_shape:** +0.068 to +0.086, all Holm 0.005 (H58 +0.053 to +0.074).
- **JetClass W full nets:** all three contrasts within ±0.03, CIs include 0 (as H58).

### 2. d_groom on top tasks without the decay edit (the main finding)
d_groom with the decay columns removed, 95 % CI:

| sample | network | d_groom (all edits) | d_groom without decay |
|---|---|---|---|
| fresh JetClass top | ParT_full | +0.168 | +0.002 [−0.057, +0.054] |
| fresh JetClass top | PN_full | +0.133 | +0.004 [−0.060, +0.062] |
| fresh JetClass top | ParT_kin / PN_kin | +0.098 / +0.085 | −0.007 / −0.010 (CIs include 0) |
| fresh Top | ParT_kin / PN_kin / LorentzNet / PFN | +0.125 / +0.117 / +0.120 / +0.151 | +0.032 / +0.016 / +0.026 / −0.004 (CIs include 0) |
| H58 JetClass top | four nets | +0.103 to +0.183 | +0.009 to +0.017 |
| H58 Top | four nets | +0.083 to +0.140 | −0.044 to +0.020 |

Dropping gout or gin instead leaves d_groom positive: fresh JetClass ParT_full drops to +0.154 without gout. Under decay, the networks change 8–15 % of calls, the m_SD and τ32 windows change 9–14 %, and the ungroomed-m window changes 0 % because m is fixed. So d_groom > 0 on top shows that the networks respond to a change of prong momentum sharing at fixed mass, which m_SD and τ32 also respond to. It does not show a groomed-mass preference specifically.

### 3. Direct grooming test r_g = chg(gout)/chg(gin) [95 % CI]
- **Top tasks:**
  - JetClass top: 0.40–0.46 (fresh), 0.42–0.50 (H58).
  - Top ParT_kin 0.60, PN_kin 0.44, LorentzNet 0.45 (fresh); PFN 0.85 [0.76, 0.95]; EFN 1.62 [1.45, 1.80].
  - Both 0 (m_SD window) and 1 (ungroomed window) are excluded for every JetClass/Top top network except PFN on H58 (0.95 [0.83, 1.09]) and the EFN (> 1).
- **JetClass W/Z:** 0.74–0.99. ParT_kin W is 0.99 [0.95, 1.03], indistinguishable from an ungroomed window.
- **hls4ml W/Z:** all five networks 0.98–1.02, including the Transformer and the Linformer. **hls4ml top:** 0.82–1.03.

On the edit built to separate groomed from ungroomed mass, the hls4ml W/Z networks do not split by architecture: all respond like an ungroomed mass. H58's Transformer d_groom > 0 comes from the decay column (without decay: −0.039 W, −0.052 Z).

### 4. Window-matched control for the τ cut (d_tau, d_width)
- **Top tasks, d_tau (τ32 cut versus same-window random thinning, both at 50 %):**
  - fresh JetClass: ParT_kin +0.204, PN_kin +0.220 (Holm 0.005); PN_full +0.092 (0.008); ParT_full +0.057 [−0.003, +0.114] (0.19).
  - fresh Top: ParT_kin +0.200, PN_kin +0.120, LorentzNet +0.195, PFN +0.185, EFN +0.353 (all Holm 0.005).
- **Top tasks, d_width:** −0.04 to −0.16. So H58's d_shape ≈ 0 is a cancellation: the τ32 cut raises similarity and the wider window lowers it.
- **JetClass/hls4ml W/Z:** d_tau +0.08 to +0.16 (Holm 0.005).
  - With the gout columns removed, d_tau falls to +0.01 to +0.03 (not significant) on JetClass W/Z.
  - JetClass Z d_shape without gout is −0.01 to +0.01.
  - The JetClass Z τ21 preference therefore rests on the gout column. The networks change calls when soft drop-removed particles are added (r_g 0.74–0.90); the τ21 tagger does too (chg 0.04–0.05); the m_SD window does not (0).

## Competing hypotheses (top tasks)
| hypothesis | prediction | result |
|---|---|---|
| H_groom (responds like an m_SD window: ignores groomed-away particles) | r_g ≈ 0; d_groom > 0 without decay | r_g 0.40–0.60, excluded; d_groom without decay ≈ 0 |
| H_mass (ungroomed mass window) | r_g ≈ 1; no response to decay | excluded (r_g < 1, decay chg 8–15 %) |
| H_sharing (uses prong momentum sharing / substructure, partial grooming) | response to decay at fixed m; 0 < r_g < 1; τ cut adds similarity at matched window | consistent |

## Control adequacy
- H58's permuted-copy null tests the network against random weights. It does not test the classical-tagger comparison itself.
- The missing control was a tagger with the shape tagger's window and efficiency but no shape cut. It is added here (msdpub_rand@50), and it changes the τ conclusion.
- The cosine metric weights edits by an RMS scale, and the top-task d_groom depends on a single edit (decay). Leave-one-edit-out with CIs should be part of any claim built on these contrasts.

## Figure check
`h58_contrasts.png` matches the JSON values (contrast points, CIs, permuted-copy medians, Holm stars). Its panel titles describe d_shape as "m_SD window + τ cut − m_SD window only". The two windows differ in width (published versus 50 % interval), which the title does not state.

## Corrected reading
- **Reproduction:** at matched 50 % signal efficiency, the H58 contrasts reproduce on fresh JetClass (`*_103`) and Top rows.
- **Top tasks, d_groom > 0:** the preference for the groomed over the ungroomed mass window rests on the decay-angle rotation at fixed mass. Without it, d_groom ≈ 0 for every top network.
- **Direct grooming test:**
  - Top networks change calls under soft-drop-removed particles at 0.40–0.60 of the rate for kept particles (PFN 0.85–0.95, EFN 1.6).
  - JetClass W/Z networks: 0.74–0.99.
  - All five hls4ml W/Z networks: 1.00. The architecture split does not appear on this test.
- **τ cut:** at a fixed m_SD window and efficiency, the τ32 cut increases similarity for 8 of 9 top networks (fresh sample). H58's d_shape ≈ 0 combines this with a negative window-width term.
- **JetClass Z τ21 preference:** it rests on the response to soft drop-removed particles.

## Proposed claim handling
- **Do not adopt** "top taggers follow the groomed, not the ungroomed, mass" (H58's 0.55).
- **Supported narrower statement (0.6):** under edits at fixed jet mass that change prong momentum sharing, JetClass and Top top taggers change 8–15 % of calls, as m_SD and τ32 taggers do and an ungroomed-mass window cannot. Soft particles that soft drop removes still change about half as many calls as soft particles it keeps (r_g 0.40–0.60).
- **hls4ml W/Z (all five architectures):** r_g = 1.00, as for an ungroomed mass window (consistent with H49 / A4).

## Files
- `/Users/anrunw/Documents/aaron-interp/interp/h58_verify.py`
- `/Users/anrunw/Documents/aaron-interp/results/physics/h58_verify.json` (copied from `/j-jepa-vol/aaron-interp/results/physics/`; fresh tables and logits in `/j-jepa-vol/aaron-interp/results/physics/h58_verify_work/`)
- Log `/j-jepa-vol/aaron-interp/logs/d-h58-verify.log`
