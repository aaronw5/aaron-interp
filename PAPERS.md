# Papers: working theses and evidence

Each paper has one working thesis. Every experiment in EXPERIMENTS.md names the paper and thesis claim it tests and is
logged as supports / weakens / refines. When the evidence changes a thesis, the old wording stays below the new one,
with the date and the experiment that changed it.

---

## Narrative (user 2026-10-16: "what's most important is a compelling narrative")

Paper A, as it stands after E1–E6 (E6 pending verification): *a trigger tagger's decision, written out.* The reader follows
one jet through the network and sees, at each stage, a quantity with a name:
1. particles pass 13 cuts (soft-particle ramps, half-planes through the axis);
2. the cuts become objects: counts and sums over the passing particles inside pT-rank windows;
3. the context round carries ΣpT (M0) into every particle's second cut;
4. the second-round objects are radial hinges summed over rank windows: ΣpT·r² (S29, S50), Σ(r − 0.2)₊ beyond the leading
   particle (S1), soft multiplicity (S19), a core count (S30); the half-planes of round one have become radial functions;
5. the dense layers multiply them: the network's "mass" is √(ΣpT · Σ pT r²) about the given axis, not the invariant mass
   (a rigid 0.06 rad shift of the jet, which leaves m unchanged, turns 76 % of W calls into Z);
6. the 16 neurons are windows on that variable (W window peaks at 75 GeV, Z at 94–98, top at 135–148);
7. the classes are conjunctions of the windows.
The worked variable (mass) is the proof of the layer-ordering claim (A9): sum → context → product → window. Each step
has a substitution test and a causal edit.
Status of the steps after round 2 (2026-10-17):
- step 2 (rank windows): causal edit with matched control for six angular objects (E20/E20v): an angle swap across the
  0–13 edge changes S43/S52/S63 by 0.056–0.084 per jet, the adjacent inside swap by 0.0014–0.0043, and the peak moves with
  shifted masks; S0, S19 untested by this edit; S1, S20, S51, S56 show no edge peak.
- step 5 ("the dense layers multiply them"): E15 withdrawn (E15v); E21's iso-product/iso-sum ratio does not locate a product
  stage (arm A inside the random-pair band). E21v's per-jet gradient-ratio test gives product-like dependence on (S0, S50) at
  z7 (slope 0.94), z0 (0.80) and the W − Z margin (0.64), above every random-pair draw (≤ 0.66) and random ReLU layer (≤ 0.01),
  with per-jet Spearman 0.21–0.43; the stage where it forms is not measured. Step 5 now reads "the read-out depends on S0 and
  S50 approximately as on their product; stage open".
- step 6 (neurons on the objects): K = 32 hinge formulas on the 19 S objects change 9.2–10.4 % of calls and follow the
  network's edit responses at 0.42–0.58 of their size (E7/E7v); stated as approximate. ParT/ParticleNet then show the same stages in a different arithmetic.

Status after round 3 (2026-10-17):
- step 2 across checkpoints (X6/X6v): the hard-particle window acts causally in all three sorted-input models (2.4–5.4×); the
  permutation-invariant checkpoint has no rank windows, the same cut types, radial objects and shift response, and +0.9 points accuracy.
- whole program (E22/E22v): per-unit interchange agreement 0.19–0.88 on call-changing swaps; the 94 % same-call figure is not a
  per-variable causal statement.
- Paper B baselines (B4/B4v): the program is not more compact; its shift lead over boosted trees is not shift-specific.

Status after round 4 (2026-10-17; tests of the proposed thesis v3 against review weaknesses 1–3):
- step 5 on natural jets (E23/E23v, two samples of 100,000 jets): on unedited off-centre jets, recentring with m fixed lowers the z
  read-out of log m by 1.5–2.5 times a matched real mass change; the reflected move raises it (1.06–1.26); a perpendicular translation of
  equal displacement raises it (0.46–0.80); rotation leaves it unchanged. The read-out follows the sign of ½Δlog(M0M2) about the stored
  axis, with gain 0.6–0.8 on the part quadratic in the shift and 1.3–1.8 on the part linear in M1. Natural centroid offsets are 1–13 mrad;
  at those sizes the call changes on tail jets follow the translation size, not the predicted mass change.
- axis convention (X7/X7v): W/Z call flow follows the sign of Δ about each model's training axis in both directions. For stored-trained
  models a translation with Δ held at 0 changes 0.30–0.42 as many calls, with no W → Z flow; for hardest-particle-trained models it
  changes 23–31 % of calls with a W → Z flow, so those models also read a translation-invariant spread.
- compact program (B5/B5v): no ≤ 50-term program reproduces the shift response; the network's shift and re-axis responses depend on
  direction, which no radial program reproduces.

Status after round 5 (2026-10-19; functional nulls, training-regime and architecture comparisons; X8/X8v, X5/X5v, X9/X9v):
- step 5 (mass) against nulls: the 0.06 rad translation W → Z flow is 0.744–0.752 in JEDI-linear and 0.66–0.73 in trained Deep Sets /
  JEDI-style float models, against ≤ 0.135 in every shuffled-label, weight-permuted and redrawn copy (X8, X8v). The same response
  appears in float, L1 and HGQ Deep Sets (X5, 12 models, ratio 0.71–1.29 of the √(M0M2) prediction), and in PFN and EFN (X9). A
  translation followed by an angle rescale that holds M0 and M2 about the stored axis fixed leaves 4–11 % W → Z in all five X9v networks
  (plain translation 39–81 %, rotation 2–3 %).
- step 2 (rank windows): the 13|14 across/inside ratio (5.19–5.22) lies inside the range of weight-permuted copies that keep the zero-bit
  masks (2.1–11.3, non-degenerate); the trained network's own live units without a 13|14 mask edge give 1.45 (X8v). The window
  asymmetry measured by this edit is set by the mask layout. With units chosen as for the flow models (top-10 by cost), JEDI-linear's
  ratio is 2.96, PFN/EFN 0.84–0.87 (X9v). No Deep Sets regime (float, L1, 10-epoch HGQ) forms rank windows (X5).
- step 1 (pure cuts): L1 at JEDI-linear's sparsity gives 100 % pure units, as does its permutation null; pure cuts above the null appear
  only in Deep Sets trained with the HGQ script (2 of 3 seeds), which also differs from the float pipeline in optimizer, batch-norm
  folding, epochs and initialisation (X5v). PFN: pure fraction equals its null (X9).
- ablation method: resample ablation ranks live units as mean ablation does (Spearman 0.965 on two samples); g under resample changes
  27.8–27.9 % of calls (mean ablation 14.6–14.9 %) (X8, X8v).

Scale requirement (user 2026-10-16: "it can't JUST be a trigger tagger"): the small model is where the program is complete
and exact; the paper's claim is about taggers in general, so ParT (and ParticleNet) are a main section, not an appendix. For
every JEDI stage there is a ParT counterpart, run with the same edits:
| stage | JEDI-linear | ParT |
|---|---|---|
| statements | l1 cuts on (pT, Δη, Δφ) | per-particle embedding MLP's first layer on 17 inputs; pair-bias hinge formulas on (ln kT, ln z, ln ΔR, ln m²) [G108] |
| objects | cut ∧ rank window, pooled by sum | the particle sets a head pools: prongs, displaced-track sets [G201]; head roles fixed by the pair bias [G101, G157] |
| context | g (ΣpT) added to every particle | class token; messages (1–2 numbers per particle late) [G121/G122] |
| variables over layers | P → S → h4/h5 → z | block-by-block read-out of the same observable list (E10 on ParT) |
| mass | √(M0·M2) about the given axis, no recentring | pair features are translation invariant by construction, so ParT's mass should be the invariant mass: the SAME translation edit on both models is the cross-model table |
| neurons | 16 windows | 13 active class-token neurons [G5b, G197] |
| classes | conjunctions of windows [J33b] | conjunctions of neurons [G206] |

Two shapes for Paper A (user 2026-10-16: "if we can get a FULL STRONG PAPER off the toy or small models that's also very
valuable"), decided after P1 and the first small-transformer results:
- **A-small: a complete account on small models.** Precedent at ICML/ICLR: complete mechanistic accounts of small networks
  (modular-addition circuits, toy models of superposition/universality) were accepted because the account was complete and
  the phenomenon general. Here the model is deployed (a trigger tagger) and the inputs are physical. Requirements for
  "full strong": (i) the whole program written out and verified stage by stage (substitution + causal edits) for JEDI-linear;
  (ii) the same program structure (statements → objects → moments → products → windows → conjunctions) recovered in ≥2 other
  small architectures trained on the same data (native transformer, Linformer; a Deep Sets / EFN baseline if cheap), with
  an untrained control each; (iii) the layer-ordering result (A9) and the mass result (A4) shown in all of them; (iv) the
  points where the architectures differ stated as measurements (e.g. does a small transformer's mass also lack recentring?).
  ParT then goes to Paper C (circuits at scale) with a one-paragraph bridge in A.
- **A-scale: small model + ParT as a main section** (the stage table above).
The small-transformer training set is therefore needed for either shape; it is launched now with the proposed set
(QUESTIONS.md), adjustable later.

## What to look for (from Cammarata, "Neural Networks Are Already Clean", Goodfire; user 2026-10-16)
Argument for our framing: most units have one main job once the tooling is right; superposition is the last hypothesis, not the
first. Checklist applied to every unit we name:
- describe the unit by the pattern it responds to (a cut, a radial hinge, a window), not by the class labels it fires on;
- weak / offshoot activations: are they neighbours of the main pattern ("recursive generalisation")?
- negative poles: for z neurons (pre-ReLU) and for the S-level hinges, what does the off side distinguish?
- upstream and downstream context (which P/S units feed it; which logits read it), plus a steering test;
- formation over training: per-epoch checkpoints of the small transformers (JEDI has no checkpoints);
- unexplained remainder → suspect the tool (quantiser masks were exactly this: 43.7 % → 80.68 %).

## Reviewer lens (keep current; every objection gets an experiment)
| objection | answer / experiment |
|---|---|
| "A 16-neuron FPGA model is a toy" | same stages shown on ParT (2M-parameter transformer) and ParticleNet; the points where exactness stops are stated |
| "Objects is a relabelling of sum pooling / EFN filter plots" | rank-window masks and cut ∧ window membership are not in EFN; causal edits that move one particle across a cut change the call as the identity predicts; untrained control. E20/E20v (2026-10-17): swapping the angles of ranks 13 and 14 changes S43/S52/S63 by 0.056–0.084 per jet, 2–5 % of that with the window masks removed, 0 with the window shifted to 0–10; the peak follows shifted masks (13 → 10 / 16); masks permuted across slots: 32–36 % accuracy vs 80.7 %. Open: S0/S19 windows (pT-only objects; node E20b). X8/X8v (2026-10-19): weight-permuted copies with the masks kept give the same across-edge asymmetry (2.1–11.3 vs 5.19), so the edit evidences the mask layout, not trained weights; X9: no rank-window response in PFN/EFN (0.84–0.87, permutation invariant by construction). X11 (2026-10-19): native S units named above every random rotation (0.79 vs ≤ 0.63), so the unit basis is not arbitrary; Deep Sets also shows a preference (0.29). X12: the 13|14 window effect is not reproduced by retrained JEDI-linear networks (masks form, effect absent). E24/E24v (2026-10-20, round 7): a per-particle azimuth scramble at fixed M0, M1, M2 moves the named radial S objects ≤ 0.19 sd, but changes 7.1–7.2 % of JEDI-linear's calls vs 7.3–8.7 % for Deep Sets and PFN, so call-level near-invariance under this edit does not separate the object description from a generic pooled latent; a latent-level difference was measured (JEDI S moves 0.47 of its spread vs 0.21 under rotation; Deep Sets / PFN 0.50–0.54 vs 0.58–0.62) |
| "Probes show readability, not use" | every probe claim paired with an intervention (translation, iso-mass, freeze S0/g) |
| "One dataset" | hls4ml + JetClass; top tagging / QG when weights exist |
| "Off-manifold edits" (weakness 1) | on-manifold check: |M1|²/(M0M2) median 3.5e-4, so the two mass definitions coincide on data; the edit isolates which one is computed. E23/E23v (2026-10-17): on unedited off-centre jets, recentring (m fixed, toward the centred configuration) lowers the read-out at 1.5–2.5 of a matched real change, a perpendicular translation of equal displacement raises it (0.46–0.80); sign of ½Δlog(M0M2) followed, two gains (even 0.6–0.8, M1-linear 1.3–1.8); call changes at natural offsets follow translation size. Open: E23w (fit both gains per stage) |
| "Axis dependence is covariate shift any non-invariant model fails" (weakness 2) | X7/X7v: W/Z flow follows the sign of Δ in both directions (stored-trained 865–2230 W → Z vs 1–45; axhard 337–1997 Z → W vs 0–233); the M2-preserving translation (Δ = 0) changes 0.30–0.42 as many calls for stored-trained models with no W → Z flow, and 23–31 % for axhard models with a W → Z flow (886–1207 vs 1–68). Rotation is not an adequate control (radial objects). Open: X7w (name the axhard models' invariant spread) |
| "Where is the compact formula" (weakness 3) | B5/B5v: axis-relative programs of 20–50 terms bracket the network's marginal call-change rates; invariant programs change none at equal same call; no ≤ 50-term program meets the strict criterion; the network's direction dependence (±Δη, rotation 3.7 %) is absent from every fitted program. Open: B5w (add odd, half-plane observables) |
| "Responses could come from architecture/inputs, not training; mean ablation is off-distribution" (round-4 weakness 12) | X8/X8v (2026-10-19): translation W → Z 0.744–0.752 vs ≤ 0.135 in 15 weight-permuted, 3 redrawn JEDI-linear copies and shuffled-label / permuted Deep Sets and JEDI-style float models (Holm p at floor); the rank-window across/inside ratio is reproduced by permuted weights with the masks kept (2.1–11.3 vs 5.19), so that edit measures the mask layout; the radial S50 response exceeds the nulls through its correlation with Σ pT r² (0.93 vs ≤ 0.70), not per unit of correlation; WTA accuracy loss 0.218 vs 0.250 for a random-direction translation of the same length. Resample ablation: Spearman 0.965 with mean ablation; g 27.9 % (mean 14.9 %). Open: X8w (null with above-chance output) |
| "Compare with the standard set architectures (EFN/PFN)" (round-4 weakness 1) | X9/X9v (2026-10-19): PFN (80.4–80.9 %) and EFN (73.4–74.3 %) on the same inputs: translation W → Z 37–82 % vs rotation ≤ 3 %, calibrated read-out 0.96–1.30; an M2-held translation leaves 4–11 % W → Z in all five networks. Rank-14 swap response only in JEDI-linear (2.96 with matched unit selection vs 0.84–0.87); PFN pure cut fraction = shuffled null; PFN/EFN latents named at lower R² (median 0.73–0.87 vs 0.98). X9w/X9wv (2026-10-20, round 7): PFN, EFN and Deep Sets trained with random rigid translations (|c| ≤ 0.08 rad) of the training jets give 0.06 rad W → Z ≤ 0.012 against 0.35–0.71 on stored-axis training (seed-level p 0.0093), at ≤ 1.13 points accuracy; a 0.12 rad translation outside that range still sends 0.51–0.84 of their W jets to Z; recentred-input networks give 0 by construction. So in generic set networks the response follows the training data; JEDI-linear HGQ-50 with augmentation keeps it in 2 of 3 seeds (under-trained, −6 to −14 points). Open: X9x (augmentation range scan; JEDI aug on the official schedule) |
| "Pure cuts and rank windows are an HGQ artefact" (round-4 weakness 7) | X5/X5v (2026-10-19): Deep Sets float / L1 / HGQ, 3 seeds: shift response in all regimes; L1 purity = its permutation null; purity above null only with the HGQ training script (seeds 0, 2; pipeline differs in optimizer, BN, epochs); no rank windows in any regime; 10-epoch HGQ formed no slot masks. X12/X12v (2026-10-19): JEDI-linear retrained with HGQ (6 runs) and with quantizers off (2 runs, same pipeline): every quantized first layer is 100 % pure, as are its sparsity-matched nulls, so purity follows from the 13–20-entry sparsity HGQ produces; float layers 53–58 % (near null). All HGQ retrainings form zero-bit masks; none reproduces the official 13|14 swap response (0.20–0.90 vs 5.18). Open: X12w (official-length HGQ schedule) |
| "Prior work: Rai & Ganguly, Wang et al." | they trace a top-tagging circuit / attention sparsity; here: stage-wise formulas, exact identity, cross-architecture comparison |
| "Single model / seed" (round-2 weakness 1) | X6/X6v: four official checkpoints; cut types, radial objects, shift response recur; window strength model dependent (2.4–5.4× at the hard-particle edge, ≤ 1 at five other edges). X12/X12v (2026-10-19, round-5 weakness 6): 8 retrained n64 networks (HGQ 6, float 2): the M2-carried translation response recurs (4/5 usable HGQ by preregistered thresholds; float at matched calibration 0.69–1.30 vs HGQ 1.02–1.38); the 13|14 rank-window response does not (0.20–0.90 vs 5.18) |
| "Program fidelity is correlational (no causal abstraction)" (weakness 4) | E22/E22v: effective IIA per unit 0.19–0.88; above permuted alignment; above a swap-free predictor at z only. X10/X10v (2026-10-19, DAS): learned 1–2-dim subspaces reproduce M0 / M2 input edits on 0.66–0.87 of call-changing pairs (random ≤ 0.14; full-swap baseline M0 0.35, M2 0.61); Δ = ½log(M0M2) not found as one variable (the "Δ" subspace follows a pT rescale only). Untrained networks also beat random subspaces; the full-swap baseline separates them for M0/M2. Open: X10w |
| "'Units are already clean' is unsupported; an SAE might find better features" (round-5 weakness 4) | X11/X11v (2026-10-19, preregistered): JEDI-linear S units named by ≤ 8-term rank-window formulas at 0.79 on three samples vs 0 of 250 random rotations reaching it (max 0.63); SAE 2×/4× features 0.16–0.21; Deep Sets units also above their rotations (0.29 vs ≤ 0.12), PFN not; untrained architectures 0–2 % named. Open: X11w (learned vs architectural basis preference; top-k SAE; non-additive scorer) |
| "No baselines" (weakness 5) | B4/B4v: not more compact; shift lead over GBDT equals its lead on a random mass-raising displacement; direct MARS fails the shift. B6/B6v (2026-10-20, round 7; review weaknesses 3, 6): Pareto comparison on JEDI-linear, Deep Sets, PFN and a transformer: at about 13 kbit the blind rank-window program's agreement with the networks' call changes under the 0.06 rad translation equals that of dev-tuned trees / GBDT / MLP of at most the same bits (−1.3 to +1.7 points over three samples); B6's +2.8 to +5.0 lead was against one under-fitted GBDT setting; larger surrogates (4–124× bits) are 5–12 points higher; M2-held translation lead in JEDI-linear only (+1.7 to +2.0). Symbolic regression not run (PySR / gplearn not installed). Open: B6w |
| "Quantisation specifics" | float + masks reproduces 95.5 % of calls; results reported on the float program with the fixed-point network as the reference |
| Paper D: "Physics edits were on 400 jets per class, first-k rows and a mirrored η build" | H3/H3v (2026-10-19): 20,000 hls4ml and 40,000 JetClass jets, random rows, corrected η; boost, decay collapse, remove-b, displaced-track, isolation edits return with the same sign, mostly within the old intervals; fresh-sample rerun of the boost and H→bb rows within ~2 points |
| Paper D: "Controls are not matched to the tested quantity" | H3v: H3's W-pair control moved the W-pair mass by 10–28 GeV; with a W-fixed control no model responds to the W-pair mass beyond rearranging the subjet-pair masses (Holm p ≥ 0.13) (2026-10-20, H23/H23v: the ParT/PN part withdrawn; it came from a fixed-axis build, and rebuilt about each jet's own axis ParT_full responds at 1.93× MLPx). H→bb: S capped at 3.5 (count kept) −3.1 vs keep-2 −61.0 points. Open: H6 (subjet-pair mass set) |
| Paper D: "Taggers being IRC unsafe is already known / soft-particle edits are off-manifold" | H1/H1v: quantified per architecture against an IRC-safe EFN (0 calls on exact collinear splits) and a count-fixed control; the soft-particle probe is below the data pT floor and position dependent, so the multiplicity statement rests on in-distribution edits (k-splits, removing the softest particle, count restore 0.14–0.67 vs EFN 0.013) |
| Paper D: "Only multi-class W/Z/top benchmarks; no rejection at fixed efficiency" | H2: Top Landscape and QG classical references at eps_S 0.3 / 0.5 with CIs (LR(mSD, tau32, mW) R30 305 vs published ParT 1602; QG N AUC 0.840); H3 rejection at the classical working point (JEDI W vs QCD 174 vs 12.4); trained one-signal models running (H4, H5 next). Round D2 (2026-10-19): edits run on trained Top Landscape and QG models (H4, H5) |
| Paper D: "The W-mass null (D1) has no positive control; a W-pair edit may just break the jet" (2026-10-19) | H4/H4v: at fixed m and tau32, LR(mSD, tau32, mW) and an MLP on the same inputs respond (−1.5 to −2.0 log-odds at 60 GeV), LR(mSD, tau32) does not (+0.03 [−0.02, +0.07]); a substructure MLP without any pair mass (MLPx) gives the side-effect level (−0.63 [−0.71, −0.55]); ParT, ParticleNet and LorentzNet respond at 2.1–4.1× MLPx and equal to MLPx + mW, on two samples. PFN/EFN at MLPx level. The 60 GeV edit is off-manifold (C2ST 0.98); on the natural-looking half of the 100 GeV edits ParT still responds (−0.66 [−0.91, −0.41]). Open: H9 (side-effect-matched edit for PFN/EFN) |
| Paper D: "Replace the 9/4 Casimir test; measure the trained QG response to a collinear split" (2026-10-19) | H5/H5v: Casimir-meets-Poisson predicts 0.463 per added particle, the data LR(N) slope is 0.095 (multiplicity sd about twice Poisson), the network random-split slopes are 0.024–0.062; an on-manifold added soft particle (C2ST 0.50) moves ParT/PN/LorentzNet −0.024 to −0.004 vs LR(N) +0.097; the hardest collinear split moves them +0.08 to +0.21 with or without a count change. EFN 0. Open: H10 (which hard-core function) |
| Paper D: "Novelty relative to mass + N-subjettiness / EFP guided search (Faucett et al., Datta–Larkoski)" (2026-10-19) | H8/H8v: Faucett-style EFP/ADO surrogates reach ADO 0.92–0.98 on natural jets (ParT Z/W 0.66–0.76) and change 0 % of calls under WTA / translation (by construction) vs 0.7–54 % for the networks; with m replaced by the axis-moment mass m_A (ADO unchanged) the surrogate reproduces JEDI-linear's WTA Z/W response (37.7 vs 37.6 %), so the axis gap is mostly one known variable (A4); not reproduced by any surrogate: the collinear-split and pT-scaling responses and ParT's Z/W ranking. Open: H11 |
| Paper D: "One-signal results rest on single training runs and interim checkpoints" (2026-10-19) | H4 uses final ParT/PN top checkpoints (best epochs 16/19; interim epochs 11/7 give the same sign, −1.18/−1.09); H5 epoch scan: every ParT/PN QG checkpoint gives k = 1 response 0.013–0.078 and remove-softest within ±0.022; PN QG final checkpoints pending (d-h2-qg-pnet, -kin running); no second seed yet. Round D3: H14 uses the final PN QG checkpoints (d-h2-qg-pnet, -kin done); still one seed per model |
| Paper D: "Does a tagger read the invariant mass or a pT-and-angle proxy fixed by its training pT?" (2026-10-19, round D3) | H12/H12v: an exact boost along the jet axis (m and m_A fixed) changes 21–38 % of calls of all 11 hls4ml-trained nets at pT × 0.9 vs 0.3–4.2 % for a matched rotation (JEDI-linear 32.5 / 32.9 % on two samples); JetClass ParT/PN 1.5–4×, Top ParT/PN/LorentzNet 3–4× fewer calls under the boost than under pT-only or angle-only scaling. Open: H12 cannot separate m from m_A; why hls4ml and Top nets differ (both narrow pT) is untested (H16) |
| Paper D: "The ΣpT shortcut (D1) may be an hls4ml sampling artefact" (weakness 7) | H12: JEDI-style retrains on pT-normalised inputs or with pT-flattened weights lose most of the pT-only response (9.6 → 0.0 / 5.8 %) but keep a 21–31 % boost response. H12v: the pT-normalised and EFN results follow from their inputs; only the flat-pT retrain bears on weakness 7, and it reweights within the same narrow pT spectrum. Partly addressed |
| Paper D: "Do the axis-convention results (D4) matter at realistic offsets?" (weakness 10) | H13/H13v: pileup-like soft particles (ρ 5–40 GeV per unit area) move the E-scheme axis by 2.5–13 mrad (median), against 40–95 mrad for WTA; translating the jet by that offset changes 0–3.3 % of calls and recomputing vs freezing the axis differs by ≤ 0.8 pp. The D4 decision-level statement applies to WTA / hardest-particle-sized offsets |
| Paper D: "Pileup robustness: do taggers act like groomed observables?" (2026-10-19, round D3) | H13/H13v: no; every network loses signal efficiency faster than soft-drop mass windows at ρ ≥ 10 (JetClass W: ParT_full 80 → 26 % at ρ 40, W(m_SD) 72 → 61 %). Content controls (H13v): JEDI-linear follows the added ungroomed mass; ParT_full/PN_full and Top ParT/LorentzNet/PFN follow the number of added soft particles. Toy pileup, models trained without pileup. Open: H17 |
| Paper D: "The QG no-counting result (D7) may be generator specific" (weakness 2) and "Casimir counting is a straw man" (weakness 9) | H14 (not yet verified): on Herwig 7.1 jets the Pythia-trained ParT/PN/LorentzNet respond −0.016 to −0.002 to an added soft particle vs +0.089 for a Herwig count-only LR(N); a GBDT on IRC-safe EFPs + N (the stronger counting alternative) responds to ΔN and not to the ΔN = 0 control, unlike the networks. Herwig-trained ParT_kin / PN_kin running (d-h14-hw-part3, -pnet3) |
| Paper D: "D7 (QG nets do not count) and D10 (JetClass/Top nets follow the added count) conflict, or differ only because the edits differ" (2026-10-20, round D4) | H18/H18v: one edit family on QG, JetClass, Top and hls4ml models. Soft particles that add energy flow move QG nets −0.068 to −0.081 per particle, about an IRC-safe EFN (−0.060/−0.067). An exact collinear split of soft particles (count +k, ΣpT and centroid fixed, EFN 0) moves QG nets −0.006 to −0.020 per particle vs LR(N) −0.068 and JetClass W-vs-QCD nets −0.055 to −0.102 vs −0.044, so the difference is model/dataset, not edit. H18's single-particle contrast is not matched (added pT 0.13 vs 0.99 GeV). Open: JetClass split halves (~0.8 GeV) may lie below the JetClass soft range; top taggers move against LR(N) (H21) |
| Paper D: "JetClass vs Top Landscape top taggers: is the W-pair-mass difference (D1/H3v vs D6) training or sample, and are edited jets built consistently?" (2026-10-20, round D4) | H19/H19v: Top-trained ParT, PN, LorentzNet respond at 2.4–6.0× MLPx on JetClass and Top jets. H19 built JetClass inputs of edited jets about the unedited axis; the non-W control shifts the axis 0.13–0.15 and a pure translation of that size moves JetClass nets −1.6 to −2.1 log-odds. Rebuilt about each edited jet's own axis, JetClass ParT/PN respond at 3.0–4.2× (T60) and 4.1–5.7× (T100) on both samples (D11). Open: H3v's ParT null with the same rebuild; residual dN of JetClass full models (H20) |
| Paper D: "Boost sensitivity of hls4ml taggers (D9) is a property of the hls4ml sample" (2026-10-20, round D4) | H16/H16v: identical small binary taggers trained on hls4ml (full or ±8 % pT window) or Top Landscape change 3.7–7.5 % of calls under B0.9 vs 0.3–0.9 % for a rotation; training sample ≤ 1.2 pp (seed spread 0.85 pp); with absolute-pT inputs the direction flips with the training sample. 67–77 % of H12's five-class changes are on W/Z jets. Open: large models trained on hls4ml; five-class task (H22) |
| Paper D: "Edits that move the jet axis were scored on a fixed axis, so earlier W-pair and pileup results mix the edit with a translation" (2026-10-20, round D5) | H23/H23v: every earlier edit rebuilt both ways. W-pair (H4 edit) on JetClass tops: fixed axis 0.7–1.6× MLPx, recentred 2.9–4.5× MLPx = 0.8–1.25× MLPxW on two independent samples, dose ordered in |mW − 80| (Holm ≤ 4e-8). H3v's ParT null withdrawn (rebuilt 1.93× MLPx, Holm 1.9e-5). H13 pu arms and Top-trained nets on Top data unchanged (≤ 1.5 pp; already recentred). Open: full-input models' non-translation residual (+0.5 to +1.2 log-odds; H28) |
| Paper D: "The QG no-counting result (D7) is a property of Pythia training jets" (2026-10-20, round D5; weakness 2 continued) | H24/H24v: ParT_kin and PN_kin trained on Herwig 7.1 give split − ΔN = 0 control −0.008 to +0.001 per particle vs LR(N) +0.08 to +0.105 on both test generators; Pythia-trained +0.005 to +0.009; the N gap between generators reproduces ≤ 6 % of the networks' Py-vs-Hw gluon output gap. Open: seed replicate (H26; checkpoint SD 0.001–0.004 is a lower bound); what removing the softest particles at fixed N (−0.05 per particle) reads |
| Paper D: "Top taggers' use of the prong shape τ32 at fixed m and mW is untested; a width edit changes several observables at once" (2026-10-20, round D5) | H25/H25v: a subjet-width edit at fixed m and mW moves all 7 networks with the τ32 sign at 0.4–2.7× MLP(m, τ32, mW) on both datasets (fresh rows reproduce within 0.2); but a side-effect MLP without τ predicts the same sign, and a profile edit that changes τ32 at fixed subjet masses gives 26 cells with the τ32 sign and 20 opposite. Claim D13 states use of the prong-internal pattern, not τ32. Open: on-manifold separation of τ32, subjet width, subjet mass and core profile (H27) |
| Paper C: "The ParT circuit (C2, thesis steps 2–3) rests on 400 jets/class and the η-mirror build" | P3/P3v (2026-10-19, loop C round C1): 2,000 jets/class + 20,000 W/Z jets, random rows, corrected η, two fresh verifier samples: the b2h8 switch reproduces (m50 87.1 / 86.6 GeV, the only head with span > 20 pt); the G21b attention-dominated split does not (attention 0.34, value 0.68); b2h8 removes 3–5 % of n98's response to a real mass edit, block-1 heads 98–99 % (b1h3 43–53 %). Open: P7 (block-1 → n98 path) |
| Paper C: "No circuit is stated as a literal set with its own accuracy" | P3/P3v: literal {b2h8, b1h3, n98} with the rest of the class stage resampled: 50–54 % W/Z (random equal-size circuits 49.7 %, model 83 %), 10-class 10–13 %; the strict definition caps all 16 heads + n98 at 63 % W/Z; best head pairs (MLP and read-out intact) contain b1h2 (65–66 %). Open: P7 (other MLP selections, scope) |
| Paper C: "Resample slopes are not shares" | P3v: the resample slope of all 16 heads is 0.12–0.15 of the real-edit slope although they carry the whole input; shares are reported as direct patch removal under the real edit (R_all16 = 1 by construction) |
| Paper C: "Only one big model has causal tests" | P4/P4v (2026-10-19): ParticleNet_full/kin (official weights, 84.6 / 70.4 %): φ + 0.06 W → Z 12 → 26–28 % (full), via ΔR and per-particle layer-1 terms; kNN graphs fixed keep 0.96–1.01; ΔR-keeping rotations change nothing (Holm p 1) |
| Paper C: "The axis response could be a shared learned solution, not the input format" | open: P4 shows two architectures on the same inputs agree; the input-format test is L-GATr (P5 training running, anrunw-ai-c-p5-lgatr2; P6 pre-registered iso-mass M2 edits) or a model retrained without ΔR |
| Paper C: "Mass-invariant mirror controls in the full models are not clean" | P4v: dφ → −dφ alone moves PN_full W → Z +4.7 points because charge and d0 are signed; flipping charge and d0 with it gives +1.7 (PN_full), +1.9 (ParT_full); rotation by 0.3 rad is the clean ΔR-keeping control |
| Paper C: "A circuit paper needs a circuit: is there a small set of heads/neurons that reproduces the W/Z decision?" (2026-10-19, round C2) | P9/P9v: whole-network search (433 nodes, 21,504 edges, opposite-class pT-matched resample): top 5 % of edges F 0.51, W/Z 55 %; a trained 10 % mask F 0.98 but W/Z 63 % (model 83 %); ACDC literal circuits 29–59 % W/Z vs random 17 %. Reported as accuracy and F against size (C8). Open: coarser units (whole blocks, SAE features), other corruptions |
| Paper C: "Faithfulness F can be inflated without reproducing calls" (2026-10-19) | P9v: F is a sum of logit differences over jets; the trained mask raises F 0.61 → 0.98 at 10 % of edges with W/Z unchanged (63.0 → 63.1 %). Accuracy is the primary metric, F secondary |
| Paper C: "Random-circuit controls that omit the read-out are too weak" (2026-10-19) | P9v: random circuits that keep E, the class stage and z coordinates score 0.38 (89-node size) against ACDC 0.56 and the read-out alone 0.25; P9's unanchored random control (0.014) is not used for claims |
| Paper C: "Where does ParT form its mass? Patching shares near the noise floor" (2026-10-19) | P8/P8v: single-sublayer restored shares (≤ 0.084) are within the mass-matched donor-resample control (MLP b3 0.055 own vs 0.130 donor); only cumulative patches (0.42 for all 16 sublayers) are used for claims |
| Paper C: "A substitution null may test the wrong variable" (2026-10-19) | P8v: P8's equal-invariant-mass donors also fixed √(S0·S2) (\|dlog mom\| 0.022), so its null (ratio 0.004) tested normalised M2 at fixed mom; donors chosen by the candidate variable give 0.42 [0.38, 0.46]. Donor selection is stated per candidate variable |
| Paper C: "Head indices and route shares come from one checkpoint" (weakness 3, 2026-10-19) | P10 running: ParT_full / ParT_kin × 3 seeds on the 2M L-GATr subset; S1–S4 pre-registered; interim e3 full_s0 R 0.94, kin seeds 0.17–0.35. 2026-10-20 (round C3): all six seeds trained 50 epochs (checkpoints in models/part_seeds/); S1–S4 analysis is P11. Round C4: done, see the "Head- and block-level circuit statements come from one checkpoint" row below |
| Paper C: "Heads and neurons may be the wrong units; with learned sparse features a circuit could be small" (weaknesses 1 and 4, 2026-10-20) | P13/P13v (0.75): transcoders on all 10 MLPs + SAEs on the class-token residual (splice drop ≤ 0.3 pt); attribution-ranked feature circuits kept alone: 50 features 50–51 % W/Z, 1,000–2,000 features 74–79 % (model 83–84 %), random 16–22 %, with attention and skip maps on the real jet; ≤ 43 % when those are also resampled. Open: trained-mask feature selection, joint feature + head circuits |
| Paper C: "A one-direction DAS match may not need a trained variable" (2026-10-20) | P15/P15v (ParticleNet): weight-shuffled and freshly initialised networks, and an unfitted mean-difference direction, reach the same summed share as the trained network's EdgeConv-1/2 direction. ParT's P8 had an untrained control (0.02–0.31 vs 0.58–0.69); a weight-shuffled ParT control is proposed (P17) |
| Paper C: "Directions fitted on one edit sign may record the sign of the edit, not the variable" (2026-10-20) | P15v: φ − 0.06 changes ParticleNet's W/Z score as much as φ + 0.06; the EdgeConv-1 direction fitted on + shifts reproduces −0.29 of the − response (Holm p 0.004); the EdgeConv-2 direction 0.73–0.80 of each sign. Mirror-sign tests are added to direction claims; ParT's C7 direction has not had this test (P17) |
| Paper C: "Atlas / monosemanticity comparisons depend on the rotation control" (2026-10-20) | P15v: rotating centred channels (P15) gives channels < rotated (0.311 vs 0.394); rotating standardised channels gives channels > rotated (0.311 vs 0.266, p 1e-4). P13's conditional-mean R² penalises sparse features and does not separate m_A from m (corr 0.998–0.999 on natural jets). No basis-alignment claim is made from either |
| Paper C: "Compare with standard attribution and probe baselines" (weakness 7, 2026-10-20) | P15: on ParticleNet, EFP (degree ≤ 4) and primary Lund-plane linear probes predict the W−Z logit with R² 0.03–0.19 and give zero response to a φ shift by construction; gradient and LRP-ε predict the per-jet φ response at r 0.31–0.45 (LRP conservation median −0.38). Same comparison on ParT not run |
| Paper C: "The methods were never checked on a known answer" (weaknesses 2 and 8, 2026-10-20) | P14: ParT trained on planted mass-window labels (invariant m, m_A, invariant m with pair route only, shuffled); training done; arm analysis pending (P16). Released-model control on fresh rows reproduces C3 (R 0.79 / 0.72, per-particle share 0.83); the pair-route-only arm learns the invariant-mass window to 97 %. Round C4 (P16/P16v): the translation test and route split recover the planted structure where it differs (arm c, pair route only: R 0, pair share 1.00; arms a/b: R 0.18–0.22, pair share 0.80–0.83 vs released 0.15). 1-d DAS and EAP-IG did not separate the m_inv arm from the m_A arm, whose labels agree on 97.7 % of jets, so this check has no discriminating power yet; a decorrelated label is proposed (P19) |
| Paper C: "Is the axis response forced by ParT's architecture and axis-relative inputs?" (round C4, 2026-10-20) | P16/P16v: ParT with the kin inputs trained from scratch on a mass-window label gives R 0.18–0.22 vs released 0.72–0.79 (fresh rows, two read-outs) and routes 80 % of a real mass change through the pair features. Not forced. Which part of the released training produces it is open: a planted ParT trained on the true class labels with the same data and epochs is the missing control (P19) |
| Paper C: "Head- and block-level circuit statements come from one checkpoint" (weakness 3, round C4, 2026-10-20) | P11/P11v: in six retrained seeds the call-level axis response reproduces (W → Z +13.6 to +19.8 pt; untrained ≤ 4 pt) but the class-block-1 share (0.63–0.84) lies inside the range of untrained networks (0.32–0.87); only the released 0.97–0.98 exceeds it. Switch heads like b2h8 appear in 4/6 seeds at other indices and masses. C2a/C2b are scoped to the released checkpoints; seed vs data scale is open (P20) |
| Paper C: "Untrained networks can score as well on probes and one-direction DAS" (round C4, 2026-10-20) | P18v: an untrained ParT reaches ridge R² 0.99 for N and pT and 0.80–0.89 for the masses, and 1-d DAS IIA 0.45–0.95, above ParT_full in 44/55 cells. Claims use the training-specific part only: calibrated gains (prong 0.77–0.80 vs 0.00–0.05) and depth profiles as difference-in-differences against two untrained seeds (scramble +0.44). The weight-permuted control (accuracy 10–12 %) is a weak null and is not used alone. Applies to C7 (P17 proposed) |
| Paper C: "Does ParT build variables in JEDI-linear's order (cuts → objects → axis-moment mass last)?" (claim A8, round C4) | P18/P18v: no. Count and pT at the embedding, mass at block 1 (matched by untrained nets), prong shape at blocks 2–4 and training-dependent; mass − prong t90 −0.38 to −0.66 sites, H1's order p = 1.0 |

## Claim confidence (0–1; from all evidence and the verifiers; updated by the ledger step each round)
Last update (A, B rows): 2026-10-20, round 7 (X9w/X9wv, E24/E24v, B6/B6v); round-6 values: B4 0.25 (A10m, A3 unchanged; A10r, A3z, A13, B6t new). Round 6: 2026-10-19 (X10/X10v, X11/X11v, X12/X12v); round-5 values: A1q 0.25, A3 0.8, A4-product 0.35, A10m 0.8, A11 0.65, B2m 0.35. D rows 2026-10-20, loop D round D5 (H23, H24, H25; round D4 values: D1 0.85, D2 0.75, D3 0.55, D4 0.8, D4m 0.45, D5 0.65, D6 0.75, D7 0.75, D8 0.5, D9 0.55, D10 0.4, D11 0.45, D12 0.55); round D3 2026-10-19 (H12, H13, H14; round D2 values: D1 0.85, D2 0.75, D3 0.55, D4 0.8, D4m 0.5, D5 0.55, D6 0.7, D7 0.7, D8 0.5; round D1 values: D1 0.85, D2 0.75, D3 0.55, D4 0.8, D4m 0.45, D5 0.55, D6 0.0, D7 0.0); C rows 2026-10-20, loop C round C4 (P11/P11v, P16/P16v, P18/P18v; round C3 values: C1 0.8, C2 0.2, C2a 0.85, C2b 0.75, C3 0.85, C3s 0.3, C4 0.0, C5 0.75, C6 0.0, C7 0.45, C8 0.55, C9 0.45, C10 0.0); round C3 2026-10-20 (P13–P15; round C2 values: C1 0.8, C2 0.2, C2a 0.85, C2b 0.75, C3 0.85, C3s 0.3, C4 0.0, C5 0.7, C6 0.0, C7 0.5, C8 0.45); round C2 2026-10-19 (P8–P10; round C1 values: C1 0.8, C2 0.2, C2a 0.85, C2b 0.7, C3 0.85, C4 0.0, C5 0.7, C6 0.0). Change from the previous value in parentheses.
Round 4 values: A1 0.85, A1x 0.55, A3 0.8, A4 0.9, A10 0.6, B2m 0.4, D4 0.8, D4m 0.45 (others below unchanged unless marked).
Round 3 values: A12 0.6, B3 0.6 (others below unchanged unless marked). Round 2 values: A1 0.9, A3 0.8, A4 0.85, A5 0.5, B2 0.85, B3 0.6, B4 0.0.
| claim | confidence | basis |
|---|---|---|
| A1 statements = cuts; objects = cut ∧ rank window (n64) | 0.8 (=) | round 6, X12/X12v (0.55): scope — the 13|14 window effect belongs to the official n64 checkpoint; 8 retrained networks form zero-bit masks but give 0.20–0.90 at 13|14 (official 5.18); X8/X8v (0.65): the across-edge window asymmetry is reproduced by weight-permuted copies with the masks kept (2.1–11.3 vs 5.19) and is absent on the trained network's no-edge units (1.45): the window is set by the zero-bit masks, the trained weights are not needed for it; X9v: matched-selection ratio 2.96 vs PFN/EFN 0.84–0.87; X5: no Deep Sets regime forms windows; E1–E3m, E9, R2; E20/E20v causal window test (0.75) for six angular objects; X6v: in n32/n128 the window effect holds at the hard-particle edge only (2.4–5.4×), 0.27–1.00 or ≈ 0 at five other edges; half or more of the n64 mask-removal cost is padded slots (real slots only: 18.4 points, not 37.7) |
| A1q (reworded 2026-10-19, round 6, X12) HGQ training produces the first-layer sparsity (13–20 nonzero weights); pure pT-only / angle-only cuts follow from that sparsity (old wording, round 5: "pure pT-only / angle-only cuts come from HGQ training", 0.25) | 0.6 (+0.35, new wording) | X12/X12v (0.55): every quantized JEDI-linear first layer (official + 6 retrained) 100 % pure = column-permutation null median and sparsity-matched float null; zero fraction 0.896–0.932 vs 0 for the pipeline-matched float arm, whose purity 0.53–0.58 is near its null (Holm 0.14); X5/X5v (0.72): purity above the permutation null only in Deep Sets trained with train_hgq_dsets.py (seeds 0, 2, p ≤ 0.0015; 30–50 % of units vs JEDI-linear 13/13), a pipeline that also differs in optimizer, BN folding, epochs and init; L1 at JEDI sparsity = null; X9: PFN = null; X6v: n32/n128/perminv at the sparsity null (p 0.16–0.48) |
| A1x program elements across official JEDI-linear checkpoints (X6) | 0.55 (=) | X6/X6v (0.45): cut types, ≥ 0.977 within-pT radial share (11/12 units), shift W → Z 0.45–0.76 in 4 checkpoints; pure cuts at the level random placement gives at the observed sparsity (p 0.16–0.48); perminv has no rank windows |
| A1r retrained JEDI-linear n64 (HGQ, 200–300 epochs) forms zero-bit slot masks but not the official 13\|14 rank-window swap response; the M2-carried translation response recurs (new, 2026-10-19, round 6) | 0.65 (new) | X12/X12v (0.55): masks 0.84–0.94 zero-bit; 13\|14 vs 11\|12 ratio 0.20–0.90 (sign-flip p ≥ 0.60) vs official 5.18–5.35 on three samples; own-edge peaks near ranks 40–42 in HGQ-50 s0, s1; translation W→Z 0.61–0.79, M2-held 0.08–0.19 in the usable HGQ nets; one primary seed collapsed (57 %), retrained copies 1–4 points below official, 10× fewer epochs |
| A2 context through four objects | 0.85 (=) | E4, E5v; R2: S58 replaceable; X8/X8v: g under resample ablation changes 27.8–27.9 % of calls (mean ablation 14.6–14.9 %); unit ranking mean vs resample Spearman 0.965 |
| A3 second-round objects = (pT, r) hinges over rank windows | 0.85 (=, round 7) | round 7, E24/E24v (0.3 for the E24 reading; the latent part reproduced): under a per-particle azimuth scramble at fixed M0, M1, M2 the named radial objects move S0, S19 0, S29/S30 ≤ 0.09 sd, S1 0.18–0.19 sd (largest per-particle unit S56 0.27 sd), on three samples; round 6: X11/X11v (0.85): 15/19 S units named by ≤ 8-term rank-window (pT, r) formulas (R² ≥ 0.9, median 0.955) on three samples, above all 250 random rotations; 7–8 of the 15 also pass the edit test (slope CI in [0.8, 1.2]); X12/X12v: radial hinge objects more frequent with quantization in every retrained network (network-level p 0.048 / 0.17); X8/X8v: the S unit most correlated with Σ pT r² (S50, ρ 0.93) responds 0.20 s.d. to angles × 1.1, above all nulls (≤ 0.142, ρ ≤ 0.70); per unit of correlation null networks respond as much (0.95–1.13 vs 0.98); X9: PFN/EFN Φ radial share 0.33–0.56 vs JEDI ≥ 0.98; X6v: radial objects recur in 4 checkpoints above a combination-of-cuts null (11/12); perminv shows they do not require rank windows; E5, E5v, E14/E14v, R2 (5 radial under every edit); E20 windows |
| A3z under a per-particle azimuth scramble about the stored axis at fixed pT, r, rank and fixed M0, vector M1, M2, JEDI-linear's named radial S objects change ≤ 0.19 sd, and the call change above rigid rotation (2.8–3.2 points) passes mostly through the context-only units S58, S31 (fed by g); Deep Sets and PFN change as many calls, so the call-level result does not separate JEDI-linear from them (new, 2026-10-20, round 7; E24's planned prediction "B ≈ A′, residual in the odd direction units, vanishing under A′" not supported) | 0.6 (new) | E24 (sample 2401), E24v passes 1, 2 (samples 2501, 2601): JEDI A′ 7.2 / 7.07 % vs rot 4.0 / 4.28 %; freezing S58, S31 lowers the excess 2.99 → 0.54 points; Deep Sets 7.3–8.5 %, PFN 7.7–8.7 %; latent change / spread JEDI 0.21 → 0.47 vs Deep Sets / PFN 0.58–0.62 → 0.50–0.54; cut-aligned scramble B − Br +0.46 points (Holm p 0.017); the quantity S58/S31 compute is not named (E24w) |
| A13 JEDI-linear's calls depend on azimuthal structure at fixed M0, M1, M2 beyond what a mass-and-radial classifier shows; whether that structure is prong shape (τ21-type) is not determined (new, 2026-10-20, round 7, E24v; E24's reading "no evidence for τ21-like information beyond the radial profile" and "the 20–40 % prong-use alternative is excluded" not supported) | 0.5 (new) | E24v pass 2 (sample 2601, N 10,000): A′ − rot excess JEDI +2.79 [2.28, 3.31] vs boosted tree on mass + radial (F0) +1.12 and with τ21 (F1) +6.38; accuracy drop under A′ JEDI 1.45 [0.97, 1.90], F1 1.14, F0 0.00; JEDI vs F1 calls McNemar p 0.60; τ21 and D2 add only 0.22 / 0.42 points of accuracy over mass and radial features, so this edit has little power to separate prong use; rotation-draw spread ±0.5 points; radial-matched real pairs N 351, mostly top |
| A4 un-recentred M2-like mass variable | 0.92 (+0.02) | X8/X8v: translation W → Z 0.744–0.752 vs ≤ 0.135 in all permuted / redrawn / shuffled-label nulls (Holm p ≤ 0.012); X9v: a translation with M0, M2 about the stored axis held fixed gives W → Z 4.0 % (JEDI-linear) vs 76.6 % for the plain +Δη translation, read-out ratio 0.00 [−0.01, 0.01]; E6, E6v, R1, E14v, E17; separate S units for M0, M2; X6v: 0.06 rad shift W → Z 0.45–0.76 vs rotation 0.02–0.03 in all four official checkpoints (log10 p ≤ −147); E23/E23v: on unedited off-centre jets recentring with m fixed lowers the read-out (1.5–2.5 of a matched real change, Holm p ≤ 1e-10 to 1e-27 on two samples) |
| A4n on natural off-centre jets the read-out follows ½Δlog(M0M2) about the stored axis (sign; quadratic part at 0.6–0.8, M1-linear part at 1.3–1.8 of prediction) | 0.6 (new) | E23 (seed 101), E23v (seed 211, 0.6): recentre / reflect / perpendicular translation give the predicted sign at matched displacement; single-gain √(M0M2) excluded (odd-part CIs exclude 1); arm A (observational matching) underpowered |
| A4d the read-out carries a direction-dependent term (odd in the shift) | 0.6 (new) | E6v odd term 12–20 %; E14/E14v per-particle maps 0.96–1.05; E23v odd part 1.34–1.82 of the M1-linear prediction; B5v: +Δη vs −Δη shift W → Z 77.9 vs 43.3 % (p 5e-54), rotation 3.7 % of calls |
| A4 product of M0 and M2 / stage where it forms | 0.25 (−0.1) | round 6, X10/X10v (0.45): DAS finds M0 (S, k 1: 0.85–0.87) and M2 (S, k 2: 0.72–0.75) subspaces, but the subspace for Δ = ½log(M0M2) reproduces only a global pT rescale; an angle-only edit of the same Δ is reproduced at 0.19–0.43 (below the 0.61 full-swap baseline, 9/9 cells) and follows the M2 subspace; Δ not shown as one variable at S, h4 or h5; E15 withdrawn (0.2); E21 R test uninformative; E21v gradient ratio product-like at the read-out, not reviewed by the verifier, stage open |
| A4c M0 and M2 about the stored axis are causal variables held in 1–2-dim linear subspaces of JEDI-linear from S on (new, 2026-10-19, round 6) | 0.6 (new) | X10/X10v (0.45): DAS effective IIA M0 0.83–0.87 (S, h4), M2 0.72–0.78 (S, h4, h5) vs random subspaces ≤ 0.14, on two disjoint samples; above the full-swap baseline by +0.5 (M0) and +0.05 to +0.17 (M2); untrained networks below the full-swap baseline in 36/36 M0/M2 cells; the swap needs the range clip Π at h4/h5; M0 calibration: carries 0.00–0.02 of a translation's read-out shift (translation leaves M0 unchanged) |
| A5 neurons = formulas on the S objects | 0.45 (−0.05) | E7/E7v (0.55): 9.2–10.4 % calls, edit slopes 0.41–0.58; E22/E22v (0.55): effective IIA z 0.37–0.88 (mean 0.66, swap-free 0.52, jet CI of the difference [0.07, 0.23]); S, h4, h5 not separated from a swap-free predictor at stage level |
| A6 classes = window rules on z, effect via additive margin | 0.6 (=) | E8, E8v |
| A8 ParT follows the same stages | 0.35 (new) | C3, P1v, P2v cover the mass row only |
| A9 depth ordering of readability | 0.55 (−0.05) | E10, R2 (ρ 0.51), X2 (ρ 0.85–0.90); D2 not causal; mass product step not located |
| A10 v2 generality across small architectures | 0.65 (+0.05) | X2 (not skeptic-rerun); X5/X5v (0.72): shift response in all 12 float / L1 / HGQ Deep Sets (W → Z 0.43–0.89, ratio 0.71–1.29); X9/X9v (0.7): PFN, EFN same response; rank windows, pure cuts and tight radial objects are not general (X5, X9) |
| A10m the un-recentred M2 response is shared by set architectures trained on stored-axis inputs (new, 2026-10-19) | 0.85 (+0.05) | round 6, X12/X12v (0.55): recurs in 4 of 5 usable retrained HGQ JEDI-linear networks and in the pipeline-matched float arm; matched real-change calibration flow(T)/flow(dilM2) HGQ 1.02–1.38, float 0.69–1.30 (network-level p 0.24), official 1.22; X12's "2–3× weaker without quantization" not supported (float networks place the Z boundary differently for ~105 GeV-equivalent jets); M1-held leaves 0.19–0.24 W→Z vs M2-held 0.03–0.19 (p ≤ 1e-11), so Σ pT r alone is excluded; earlier: X8 (dsets, jedis: W → Z 0.66–0.73 vs nulls ≤ 0.016), X5/X5v (12 Deep Sets, 3 training regimes), X9/X9v (PFN, EFN; M2-held translation removes most of it in all five networks, W → Z 4–11 %; PFN read-out keeps −0.13 to −0.18 of an invariant-like part; 13–21 points of extra call changes remain); round 7, X9w/X9wv (0.5): in PFN, EFN and Deep Sets the response is removed by translation-augmented training for offsets inside the trained range (see A10r), so the "stored-axis inputs" scope is confirmed as training-data dependence for these architectures; JEDI-linear HGQ-50 augmented runs keep it in 2 of 3 seeds (architecture vs under-training open) |
| A10r in PFN, EFN and Deep Sets the un-recentred translation response is set by the training data: training on jets randomly translated by up to 0.08 rad removes the response for offsets inside that range at ≤ 1.13 points accuracy, but not beyond it (0.12 rad); recentred inputs remove it by construction; for JEDI-linear HGQ-50 augmentation removed it in 1 of 3 seeds (new, 2026-10-20, round 7; X9w's proposed wording "augmented networks read mass from the spread about the jet's own centroid" refuted by X9wv) | 0.55 (new) | X9w (sample 909) / X9wv (sample 2610, 0.5): stored 0.35–0.71 vs aug ≤ 0.012 W→Z under 0.06 rad (rotation null ≤ 0.018), seed-level stratified p 0.0093 (minimum attainable, 2 seeds per architecture); T12 (0.12 rad) aug 0.51–0.84; dilM2 responded to in every cell (stored − aug p 0.49); M2-held W→g/q 0.19–0.85 in aug / recentred vs −0.07 to −0.01 stored; aug outputs not exactly invariant at 0.06 rad (TV excess 0.006–0.028); JEDI aug 0.375 / −0.010 / 0.496 with 4–14 points accuracy loss (p 0.10, minimum attainable); held at 0.55: two seeds per architecture, one augmentation range |
| A11 one main job per unit | 0.7 (+0.05) | E16 (not skeptic-rerun); round 6 X11/X11v (0.85) supports the named-unit part for S and h4 by a preregistered, blind measure (A11b) |
| A11b JEDI-linear's pooled S units (and h4) are a preferred basis: named by short rank-window pT/angle formulas more often than any random rotation of the same activations, and an SAE does not raise the rate (new, 2026-10-19, round 6) | 0.75 (new) | X11/X11v (0.85): S 0.79 on three samples vs 0/250 rotations reaching it (max 0.63; p 0.005, 0.020); h4 0.55–0.56 vs ≤ 0.36; generic 380-observable library 0.74 vs ≤ 0.58; SAE 2×/4× 0.16–0.21, PCA 0.11; not unique: Deep Sets 0.29 vs ≤ 0.12; PFN inside its rotation range; untrained architectures 0–2 % named (control at floor: learned vs architectural preference open); named native and rotated directions pass the edit test at the same rate (0.47 vs 0.50); SAE comparison specific to additive hinge naming |
| A12 mass formula follows the training axis | 0.5 (−0.1) | E18, X4 (not skeptic-rerun); X7/X7v (0.4): hardest-trained models evaluated on the stored axis move Z → W (337–1997 vs 0–233) as ½Δlog(M0M2) about their own axis predicts, with read-out 0.16–0.62 of a matched real change (not "mostly recentred" outside the X4 axis-jump edit); a translation with that quantity held fixed changes 23–31 % of their calls with a W → Z flow, so they also read a translation-invariant spread |
| B1 exact pooled identity | 0.9 (=) | E3, E3m, E9 |
| B2 nested if-statement program reproduces the network's calls (94 %) | 0.85 (=) | E11, E11v, R2, B4v (94.06 % fresh) |
| B2m the program's variables are the network's causal variables (IIA) | 0.4 (+0.05) | round 6, X10/X10v (0.45): X9's null was a basis effect; learned 1–2-dim subspaces for the program-level variables M0 and M2 reach effective IIA 0.66–0.87 (random ≤ 0.14), above the full-swap baseline (M0 +0.5, M2 +0.05 to +0.17); Δ not found as one variable; the named per-unit formulas remain at the random-rotation level (X9); X9's random-rotation baseline lacked the range clip (wrap untested); X9: JEDI-linear S-formula effective IIA 0.701 vs random-rotation alignment 0.705 (p_rot 0.44–0.95 for every model); above permuted alignment (0.251, p 0.005); E22/E22v: effective IIA stage means 0.52–0.66 vs permuted 0.22–0.35; above swap-free at z only; no named unit ≥ 0.8; random multi-stage swaps 0.78–0.83 vs 0.944 without swap |
| B3 same call on data ≠ same mechanism (direct surrogate fails edits) | 0.65 (+0.05) | B5/B5v (0.5): axis vs invariant programs of equal size differ by ≤ 1.2 points of same call and by 51–68 vs 0 % of calls under the shift (the invariant zero holds by construction); every fitted program, mars30 included, misses the network's direction dependence (−Δη W → Z +29.8 points);  E11v; E7/E7v basis-dependent; B4v: direct physics-to-logit MARS (92.9 % same call) reproduces 18–23 % of the shift's call changes with 0 % W → Z, 56–57 % of a random displacement's; GBDT on logits follows the shift (W → Z 74.5–76.3 %); E22v: the probe-vs-formula IIA contrast tracks fit quality (not matched) |
| B5c a program of ≤ 50 terms on axis-relative observables reproduces the shift and re-axis responses (thesis v3 compact form) | 0.15 (new) | B5/B5v (0.5): marginal rates bracketed at T 20–50 (51–68 % vs 56 %; 27–33 % vs 30 %), strict criterion not met, W → Z 23–64 % vs 75 %, per-jet agreement at or below the never-flip reference at T 20, no direction dependence |
| B4 the nested program beats compact baselines at equal DL | 0.15 (−0.1, round 7; was 0.0, not started, before round 3) | round 7, B6/B6v (0.7): blind rank-window program at ~13 kbit vs dev-tuned trees / GBDT / MLP of ≤ equal bits: −1.3 to +1.7 points translation agreement over three samples (JEDI, PFN), +0.2 (Deep Sets), +1.6 (transformer); oracle-selected baselines 0–2.4 points higher in all four; the remaining lead is on the M2-held translation in JEDI-linear (+1.7 to +2.0, Holm ≤ 6e-5) where every ≤ 14 kbit surrogate is within ~3 points of never-flip; frontier AUC below GBDT in 3 of 4 networks; earlier: B4/B4v (0.55): not more compact (baselines 4.6–19× lower DL for K ≤ 16); same call within 1 point at K 16, 30; shift lead over GBDT +2.8–3.3 points equals its lead on a random mass-raising displacement (difference +0.13 [−1.29, 1.52], −1.95 [−3.41, −0.41]) |
| B6t the blind pipeline (MARS on rank-window (pT, r) sums, fitted input-to-logit without unit access) transfers to Deep Sets, PFN and a small transformer at the edit fidelity per bit of dev-tuned trees / GBDT / MLP (new, 2026-10-20, round 7; B6's proposed "the program's frontier dominates GBDT/trees at equal DL" not supported) | 0.7 (new) | B6 (N 8,152) and B6v (three samples): translation agreement prog30 − dev-selected baseline of ≤ equal bits −1.3 to +1.7 points (JEDI, PFN), +0.2 (Deep Sets), +1.6 [0.78, 2.47] (transformer); same call 87.3–92.5 %; scr invisible to radial observables (Jaccard 0); symbolic regression not run; the 32-bit cost convention favours the program |
| C1 / C2 / C3 / C4 | 0.8 (=) / 0.2 (−0.5; v1 wording "the mass enters through the attention; b2h8 thresholds near 85 GeV", 2026-10-16, split by P3/P3v into C2a and C2b) / 0.9 (+0.05, round C4: reproduced on two more fresh samples by P11/P11v and P16/P16v, and with a second read-out) / see C4 row below (loop C round C4, 2026-10-20) | round C4 (C3): released R (φ + 0.06 vs matched dilation) 0.787 / 0.722 (P11, *_100, rng 11), 0.785 / 0.725 (P11v, *_101), 0.790 / 0.723 (P16), 0.793 / 0.719 (P16v, *_101, seed 16); W-or-Z vs QCD read-out 0.654 / 0.628 (P16v); invariant mass (R = 0) excluded in all; matched-dilation route per-particle 0.82–0.84, pair 0.15–0.18. Earlier basis: | G107/G108; C2: G21–G22, P3, P3v (attention re-weighting 0.34 of n98's response, value change 0.68; b2h8 route 3–5 %); P1, P1v, P2, P2v, R1, P4 (ParT numbers reproduced on the same rows); round C2: P10 battery reproduces the released models on 10,000 fresh random rows (φ + 0.06 W → Z 9.4 → 31.4 %, R 0.76; WTA 11.1 / 34.3 %); P8v: the mass-evidence direction follows √(S0·S2), not m; round C3: P14 control on 20,000 fresh W/Z/QCD rows (seed 14): R (φ + 0.06 vs a matched real dilation with equal Δlog m_A) 0.790 [0.781, 0.799] (full), 0.723 (kin), rotation null ≈ 0; per-particle route 0.83, pair route 0.15–0.17 of the matched dilation; under φ the pair-only route equals the rotation null |
| C2a ParT_full's class-attention head b2h8 switches its class-token share at 87 GeV (4 GeV wide); no other class-attention head has a mass switch (new, 2026-10-19; scope added round C4, 2026-10-20, P11/P11v: a property of the released checkpoint; retrained seeds have switch heads at other indices and masses) | 0.85 (=) | round C4: P11 released b2h8 m50 87.0 [86.5, 87.0] GeV, w 4.8, 0.87 → 0.08 (reproduced); P11v: strong class-share switches in 4 of 6 retrained seeds at different heads (c2h2 105 GeV, c2h4 68 GeV, c2h1 64 GeV, c2h3 85 GeV), kin s2 c2h3 0.85 → 0.11 (amplitude 0.74 vs 0.79); none in 8 untrained networks. Earlier: | P3 (m50 87.10 [86.79, 87.39], span 82.5 pt vs ≤ 16.4); P3v fresh samples (86.64 [86.31, 87.01], span 80.6 vs 15.7); the share also moves under the mass-fixed φ shift (−0.17) |
| C2b in ParT_full the response of the W/Z class-token coordinate n98 (z[97]) to a real mass change enters through the first class-attention block (about half through head b1h3), not through b2h8 (new, 2026-10-19; round C2: the same holds under the mass-fixed φ shift; scope added round C4, 2026-10-20, P11/P11v: holds for the released checkpoints only) | 0.75 (=, scoped) | round C4: P11/P11v class-block-1 share of the W/Z unit's angles × 1.1 response: released full 0.981 / 0.982, kin 0.968 / 0.966 (above every untrained value); retrained seeds 0.63–0.84 (S3 > 0.9 fails 6/6), inside the untrained range 0.32–0.87; class block 2 in seeds 0.17–0.36; largest single block-1 head released 0.45–0.46 (c1h3 full, c1h7 kin), seeds ≤ 0.23; continued training on 2M jets leaves the seed share flat (e9–e49). Earlier: | P8 path patching under φ + 0.06: block-1 class heads 0.991 [0.988, 0.993], b1h3 0.532, b1h6 0.152, block 2 directly 0.012 (not re-run by P8v); P9: the largest attributions sit at the n98 direct path, b1h2/b1h3 and class MLP group 0; P3v direct patch removal (angles × 1.1): block 1 0.987, b1h3 0.466 (0.43–0.53 over edits), b2h8 0.046, block 2 0.061; literal circuit {b2h8, b1h3, n98} 50–54 % W/Z (strict definition caps all 16 heads at 63 %); particle-block heads 4/8 0.26 vs null 0.13 on W only |
| C5 ParticleNet shares ParT's axis-relative, non-invariant mass response, entering through each particle's ΔR and per-particle terms, not the kNN graph (new, 2026-10-19) | 0.75 (+0.05, round C3: P15/P15v graph freezing and φ response on three more samples) | round C3: P15 (_120) PN_full W → Z 10.8 → 25.4 %, PN_kin 17.4 → 28.0 %; all kNN graphs frozen keep 0.993 [0.975, 1.014]; P15v: freezing removes 0.7–4.0 % (full), −1.3 to 0.1 % (kin) over three samples; layer-1 graph unchanged by construction. P4/P4v (0.72): φ + 0.06 W → Z 12 → 26–28 % (full; ParT 31–33 %), 18 → 27–29 % (kin); calibrated ratio 0.62–0.71; ΔR alone 0.94–1.41; graphs fixed 0.96–1.01; ΔR-keeping rotations Holm p 1. Input format as the cause: not tested (C6) |
| C6 a Lorentz-equivariant tagger (L-GATr) responds to iso-invariant-mass edits that change M2 with read-out ratio ≈ 0 (new, setup) | 0.0 (not started) | P5: no released weights; training anrunw-ai-c-p5-lgatr2 running (ETA 2026-10-21); interim it-4999 (79.2 %): rigid φ rotation 0.01 % calls (by construction), angles × 1.1 W → Z 17.9 → 42.6 %. Test P6 pre-registered |
| C7 ParT_full carries, from the input embedding onward, one direction of its jet-averaged particle representation that follows the axis-moment mass √(ΣpT · ΣpT ΔR²) and carries about two thirds of the response to a mass-fixed φ shift; no particle block or sublayer forms it (new, 2026-10-19, P8/P8v) | 0.45 (−0.05, round C3) | round C3 (lowered): P13v: in natural jets the W/Z read-out changes with m_A at a slope that accounts for 0.152 of the φ + 0.06 logit change 1.077 (14 %), so most of the φ response is not explained by natural m_A variation; the single class-level SAE feature most correlated with m_A (C0#831, r −0.76) does not carry the φ response; P15/P15v (ParticleNet): one-direction DAS matches of this size also appear in weight-shuffled and freshly initialised networks, and a direction fitted on one shift sign can record the sign (ParT's direction has not had the weight-shuffled or mirror-sign test, P17). P8: 1-dim jet-mean DAS direction recovers 0.58–0.69 of the φ logit change at emb..b8 (random ≤ 0.06, untrained 0.02–0.31; held-out η 0.45–0.71); single sublayers ≤ 0.084, all 16 sublayers 0.418 [0.406, 0.428]. P8v (fresh _102): direction follows dlog mom (r 0.87); substitution from donors differing in mom 0.42 [0.38, 0.46] of the edit-calibrated slope (random ≈ 0); natural β_S0/(2β_M2) 1.02 [1.00, 1.05] at b1 (mom 1, normalised M2 0). Held below 0.7: not an exact log-mom variable (pT/φ per dlog mom 0.71 at b1, 0.40 at emb); real-jet substitution under half the edit slope; nonlinear / per-particle forms untested; single-sublayer shares within the donor-resample control |
| C8 (v2, 2026-10-20, P13/P13v) no circuit found inside ParT_full reproduces the W-vs-Z calls when every other unit carries a pT-matched jet of the other class: neither head/MLP-group circuits with up to 10 % of the edges nor attribution-ranked circuits of up to 50 learned sparse features (transcoder / SAE latents). (v1 wording, 2026-10-19, P9/P9v: "no circuit found inside ParT_full with up to 10 % of its edges reproduces the W-vs-Z calls when every other edge carries a pT-matched jet of the other class") | 0.55 (+0.10, round C3) | round C3: P13/P13v (0.75): dictionaries spliced into all 12 sites change 10-class accuracy by 0.03–0.28 pt; IG-ranked feature circuits kept alone (attention, skip maps, embeddings live): 50 features 50.0 / 51.4 % W/Z, 1,000 74.1 / 73.8 %, 2,000 79.0 / 78.3 %, all 81.1–81.6 % (model 83.5–84.2 %, random equal-size 16–22 %, p 0.01–0.02); with particle attention and skip maps also resampled ≤ 43.4 % even with all features kept; R1 latents alone with upstream resampled 17.0 % (= donor). Limits: ranked selection only (no trained mask over features); no joint feature + head circuit. P9: EAP-IG-ranked top 5 % of edges F 0.513 [0.506, 0.520], W/Z 54.7 % (model 83.25 %, everything resampled 16.8 %); ACDC literal 10/27/89 nodes W/Z 29/36/59 %. P9v: trained top-K mask 10 % of edges F 0.977, W/Z 63.1 %; 5 % F 0.715, W/Z 58.2 %; F = 0.8 by ranking at 6,318 edges (29 %). Existence claim limited to the searches run (IG ranking, ACDC from top 2,000, 800-step trained mask) and to one corruption; coarser units (blocks) untested; learned features tested in P13 |
| C4 (v2, 2026-10-20, round C4, P18/P18v) in ParT the readable order is count and pT at the input embedding, mass and tau21 from block 1, tau32 at blocks 2–4; only the prong-shape read-outs (tau21, tau32) depend on training, and the share of the response to an azimuth scramble about the axis carried by one jet-level direction rises with depth only in trained networks; there is no mass-last stage. (v1 wording, 2026-10-16: "per-block readability of observables follows algebraic depth", 0.0 not started) | 0.55 (new wording; v1 as worded with mass last: not supported) | P18 (N 50,000, *_100, seed 18; 85.84 / 73.88 %) and P18v (_101 seed 1018, N 20,000; _102 seed 2018, N 5,000): ridge R² ParT_full emb → b1: N 0.989 → 0.986, log pT 0.989 → 0.981, log m_A 0.827 → 0.937 (no later rise; untrained 0.80–0.89, margin 0.04–0.13 by init seed), tau21 0.537 → 0.729 (b4 0.799; untrained 0.377), tau32 0.055 → 0.373 (b4 0.505; untrained 0.106); mass − prong t90 −0.66 (P18) / −0.38 to −0.42 sites (P18v), H1 order p = 1.0; scramble calibration gain 0.77–0.80 trained vs 0.00–0.05 untrained; 1-d DAS levels not training evidence (untrained IIA 0.45–0.95); full-jet scramble DAS 0.02 → 0.68 at b6–b7 vs flat 0.45–0.54 untrained (difference-in-differences +0.44 [0.42, 0.47], both untrained seeds, both models); pT / dilation / φ DAS falls with depth (DiD −0.15 to −0.20); soft-particle rise training-specific only in ParT_kin. Open: which prong observable (tau21, tau32, D2 change together under the edit) |
| C3s (v2, 2026-10-20, round C4, P11/P11v) the call-level axis response holds in six ParTs retrained from scratch on 2M jets (φ + 0.06 W → Z +13.6 to +19.8 points on two samples; untrained −0.3 to +4.0); the class-block-1 route (S3) and a single dominant head are not reproduced. (v1 wording, 2026-10-19, P10: "C3's axis response holds across retrained ParT seeds and at L-GATr's 2M-jet training scale (pre-registered S1–S4)", 0.3 interim) | 0.7 (+0.4, v2 wording) | round C4: P11 (*_100, rng 11) / P11v (*_101, rng 21): W → Z full seeds +19.1 / +15.0 / +17.8 (+19.8 / +15.2 / +19.1), kin +15.5 / +14.7 / +13.6 (+15.5 / +15.0 / +14.5); Holm McNemar p ≤ 5e-245; R full 0.65–0.81, kin 0.53–0.59 (released 0.79 / 0.72), but untrained R −0.23 to 0.78, so R's value is not a training test; pre-registered S1 holds 3/6 (untrained full 0.93), S2 structural, S3 fails 6/6 (0.63–0.84 vs untrained 0.32–0.87), S4 holds in every trained model (strong switches 4/6 at different heads). Seed, distinct data (2M vs 100M) and training length (51M vs 512M samples) confounded. Earlier (v1 basis): | P10 epoch 3 of 50 (N 10,000): full_s0 φ + 0.06 W → Z Δ +22.6 [19.7, 25.4] (released +22.0), R 0.94 [0.89, 0.99] (√(M0M2) prediction 0.97, invariant 0); kin seeds Δ +8.5 / +4.2 / +4.4, R 0.35 / 0.17 / 0.20 (released kin 0.69), still changing between epochs (two kin seeds negative at e0). Round C3 (2026-10-20): all six seeds trained 50 epochs; best checkpoints in /j-jepa-vol/aaron-interp/models/part_seeds/ (four jobs show Failed from a wrapper syntax error after the final test step; final test metric full_s0 0.757, full_s1 0.760, kin_s0 0.715); S1–S4 analysis is P11 (0.3 = until then) |
| C9 in ParticleNet the W/Z response to a mass-fixed φ shift is the same for +0.06 and −0.06 rad; after the second EdgeConv layer one jet-level direction reproduces 0.73–0.80 of the response to each sign and is not an axis-moment-mass copy; the rebuilt kNN graphs carry ≤ 4 %; 16–32 pooled channels kept alone match the W/Z accuracy but not the 10-class accuracy (new, 2026-10-20, P15/P15v) | 0.45 (new) | P15 (_120, N 20,000, seed 15; PN_full 84.50 %, PN_kin 70.14 %) and P15v (_124, seed 45; 83.95 / 69.74 %): φ − 0.06 Δd −0.487 vs φ + 0.06 −0.483 (full); ec2 direction 0.75 / 0.80 (full), 0.73 / 0.74 (kin) of the ± responses; ec1 direction fitted on + shifts 1.13 / −0.29 (Holm p 0.004), both-sign fit 0.82–0.98; per dlog m_A relative to φ: angle scaling 0.11–0.39, pT scaling −0.05 to 0.36 (m_A copy predicts 1); weight-shuffled network 0.90–2.5 and fresh-init / mean-difference directions reach the same summed share, so the direction does not show a trained variable; graphs frozen remove 0.7–4.0 %; pooled full K 32 70.9 / 69.3 % vs model 71.9 / 70.9 % (McNemar p 0.29 / 0.099), kin K 16 60.0 / 59.9 % vs 60.1 / 58.2 %, 10-class 20–45 % vs 69–85 % (channels ranked by W-vs-Z attribution); post-pooling one-direction share sample-dependent (0.11–0.57) |
| C10 (v2, 2026-10-20, round C4, P16/P16v) ParT with the kin inputs does not force the axis-relative response: trained from scratch on a mass-window label (invariant mass or axis-moment mass; the two labels agree on 97.7 % of jets) it responds to the mass-fixed φ shift at R 0.18–0.22 against 0.72–0.79 for the released models, and takes 80–83 % of a real mass change through the pair features (released: 82–84 % through the per-particle angles); which part of the released training (10 classes, 100M jets, training length, seed) produces the larger response is not identified. (v1 wording, pre-registered 2026-10-20, P14: "the axis-relative response is a learned choice: a ParT trained on an exact invariant-mass-window label does not respond to the mass-fixed φ shift (R < 0.2) while one trained on an m_A window does (R ≈ 1); the alternative (input / architecture prior) predicts R ≥ 0.5 for both", 0.0; outcome: R_a < 0.2 holds, R_b ≈ 1 not observed, the architecture-prior prediction excluded) | 0.55 (new wording; v1 was 0.0 untested) | round C4: P16 (*_100, N 20,000, rng 14) / P16v (*_101, rng 16): R_a 0.185 [0.181, 0.189] / 0.184, R_b 0.215 / 0.215, R_c 0.000, released full 0.790 / 0.793, kin 0.723 / 0.719 (W-or-Z vs QCD read-out 0.654 / 0.628); permutation p 0.0005 for R_a vs R_b, c, s, released; calls changed under φ a 5.9 %, b 6.3–6.5 %, released 18.5–31.4 %; arm a R 0.12–0.24 over epochs 0–9; arm a misses only near window edges (0.81–0.90 at 8–12 GeV from 85). Arm s collapsed to a constant. Arms a and b share torch seed, split and batch order. Method check: 1-d DAS and EAP-IG did not separate arms a and b (same function on 97.7 % of jets); EAP literal 256-node circuits 0.92–0.94 own-label accuracy vs random F 0.21–0.23 (20 circuits, p 0.048). Earlier (v1 setup basis): | P14: arms a (m_inv window), b (m_A window, same positive rate 0.4137), c (m_inv window, pair route only), s (shuffled) trained (anrunw-ai-c-p14-a/b/c Complete); arm analysis pending (P16). Labels a and b agree on 97.5 % of jets (corr(m_inv, m_A) 0.999), so the φ edit is the separating test. Arm c reaches 0.971 held-out label accuracy at epoch 2: ParT with only the pair features learns the invariant-mass window |
| D1 / D2 / D3 / D4 | 0.85 (+0.05) / 0.75 (+0.05) / 0.55 (new; was 0.0, not started) / 0.8 (=) (round D5, 2026-10-20: values unchanged. D1's ParT/JetClass extension from H3v is withdrawn (H23/H23v): rebuilt about each edited jet's own axis, H3v's k-means edit gives ParT_full 1.93× [1.51, 2.39] MLPx at 60 GeV (Holm 1.9e-5, = 1.07× MLPxW) and −0.42 [−0.63, −0.22] beyond MLPxW at 100 GeV; D1's value now refers to the JEDI-linear statement only. D4 methods instance: fixed-axis builds of edits that move the axis mix the edit with a translation. Round D4, 2026-10-20: values unchanged. D1's JEDI-linear statement unchanged; its ParT extension from H3v ("ParT does not respond to the W-pair mass beyond a W-fixed control") is contradicted by H19v on 550–650 GeV JetClass tops with each edited jet rebuilt about its own axis (ParT_full W − N −1.80 [−2.03, −1.59], 4.06× MLPx), see D11; whether H3v's build held the axis fixed is not yet checked (H20). D4 gains an instance: holding the unedited axis for edited jets produced H19's JetClass T60 null. 2026-10-19, round D1; D4 unchanged after round 5: X8v shows the WTA accuracy loss, 0.218, is matched by a random-direction translation of the same length, 0.250, so the loss reflects displacement size) | D1: J26–J30; H3/H3v: JEDI W call 87.4–87.5 → 26.9–27.0 % under pT × 0.9 on two samples, rotation 87.2–87.7; no W-pair-mass response beyond a W-fixed control (JEDI, ParT_full, ParT_kin, Holm p ≥ 0.13). D2: G194, G202–G204; H3/H3v: H→bb follows the count of tracks with S ≳ 2–3 (capped −3.1 vs keep-2 −61.0); isolation, charge flip, π⁰, dead cone not re-verified. D3: H3 (verifier 0.45; the proposed ParT_full W-pair refinement refuted). D4: P1v, E17, E19, R1; X7/X7v: JEDI-linear about the hardest particle 30.98 % / 31.66 % of calls on two fresh samples (E17 30.3 %). Round D2 (2026-10-19): values unchanged. D1's W-mass null is scoped to hls4ml / JetClass models (H4/H4v: Top-Landscape-trained ParT, ParticleNet, LorentzNet do use the W-pair mass, D6). D4 supported by H8/H8v: axis-free IRC-safe surrogates with ADO 0.92–0.98 change no calls under WTA while the networks change 0.7–38 %. Round D3 (2026-10-19): values unchanged. D1 refined by H12/H12v: H3's "boost × 0.9" was a first-order mass-preserving boost; the exact boost reproduces it (W 87.3 → 26.8 %), an exact pT-only scaling gives 87.3 → 47.9 %; 62 % of JEDI-linear's boost-changed calls are W/Z → t, the ΣpT shortcut. D4 scoped by H13/H13v: pileup-sized axis offsets (2.6–13 mrad) change 0–3.3 % of calls |
| D5 trained taggers are not IRC safe and use constituent multiplicity (new, 2026-10-19; round D2: multiplicity part scoped to hls4ml/JetClass, see D7) | 0.65 (+0.05, round D4, 2026-10-20: H18v exact collinear split of soft particles, count changed with ΣpT and centroid fixed and EFN response 0: JetClass W-vs-QCD networks −0.055 to −0.102 per particle vs LR(N) −0.044; QG networks −0.006 to −0.020 vs −0.068; earlier 0.6, +0.05 round D3: H13v count control) | Round D3: H13v c4 (the same pileup-like soft particles placed at r/20, count and ΣpT kept, Δm ≈ 0) lowers W efficiency of JetClass ParT_full 80.0 → 64.2 % and PN_full 78.0 → 62.8 % (full edit 72.0 / 70.2 %) and Top ParT retention to 0.71 (= full edit), LorentzNet 0.50, PFN 0.70, while one particle carrying the same mass and ΣpT (c5) gives little; so ParT/PN_full and Top-trained ParT/LorentzNet/PFN respond to the number of soft particles; JEDI-linear does not (c4 87.5 vs 88.2 %). Earlier: H1/H1v (0.55): exact collinear split of the hardest particle (z 0.05, dR 0) changes 1–5 % of calls of 8 trained taggers on hls4ml and JetClass, 0 % for an IRC-safe EFN without truncation; above the count-fixed control in 7 of 8 (not ParT_kin); k splits, removing the softest particle and count restore move calls along the particle count; soft-particle probe below the data pT floor and position dependent. Round D2: H5/H5v extend the IRC-unsafe half to QG-trained ParT, ParticleNet, LorentzNet (hardest collinear split +0.08 to +0.21 logit, EFN 0); the multiplicity half does not hold for these QG models (on-manifold added soft particle −0.024 to −0.004 vs LR(N) +0.097) |
| D6 top taggers trained on Top Landscape (ParT, ParticleNet, LorentzNet) use the W-pair mass inside top beyond total mass and tau32 (reworded 2026-10-19, H4/H4v; setup wording "top taggers do / do not use the W-pair mass inside top", H2) | 0.75 (= round D5, 2026-10-20: H23 rc Top ParT T60 −1.73 [−2.03, −1.43] reproduces H4v −1.79, the original H4 build was already recentred; not re-run by the verifier. Prong-internal radiation pattern also used beyond m and mW, see D13; +0.05 round D4, 2026-10-20: H19/H19v Top-trained ParT, PN, LorentzNet respond to the W-pair edit at 2.4–4.7× MLPx on JetClass tops 550–650 GeV as on Top jets, 2.9–6.0× on fresh JetClass rows; earlier 0.7, +0.7 round D2; was 0.0, not started) | H4 (verifier 0.55 for H4 as stated): W-candidate pair 80 → 60 / 100 GeV at \|dm\| < 0.5 GeV, \|dtau32\| < 0.02; W − N dscore ParT −1.73 / −1.20 (H4v −1.79 / −1.14), PN, LNet larger; LR2 0; MLPx (side effects, no pair mass) −0.63 / −0.54, MLPxW −1.89 / −1.11; QCD pairs closed from 90–115 onto 80.4: ParT +0.50 [+0.25, +0.75]. Held below 0.8: the 60 GeV edit is off-manifold (C2ST 0.98), one side-effect reference. PFN / EFN: open (response at MLPx level) |
| D7 quark–gluon taggers (ParT, ParticleNet, LorentzNet) do not count constituents; they are collinear unsafe through the energy sharing of the hardest particles (reworded 2026-10-19, H5/H5v; setup wording "quark–gluon taggers count constituents", H2, and H5's pre-registered "Δlogit per ΔN = 0.46 at fixed IRC-safe shapes" not supported) | 0.8 (+0.05, round D5, 2026-10-20: H24/H24v Herwig-trained ParT_kin and PN_kin also do not count: split − ΔN = 0 control per particle Hw ParT −0.0041 [−0.0047, −0.0035] / +0.0013, Hw PN −0.0083 / −0.0014 (Pythia / Herwig jets) vs LR(N) +0.079 to +0.105; fresh seed 7243 reproduces; near-collinear split (C2ST 0.52) equals the exact split; on-manifold additions without removal −0.035 to +0.004 per particle. N-swap by the generator gap reproduces ≤ 6 % of the networks' Py-vs-Hw gluon logit gap (GBDT(EFP+N) 65–69 %). Herwig training shifts the response by −0.01 to −0.03 per particle, 3–10× the checkpoint SD, seed spread unmeasured. Removing the softest particles at fixed N moves all four nets toward quark (−0.045 to −0.056 per particle), so soft-particle momenta and positions are used. Earlier 0.75, +0.05 round D4, 2026-10-20; scope stated: "do not count constituents" holds for count at fixed IRC-safe content. H18v exact collinear split of soft particles: QG ParT, PN, LorentzNet −0.006 to −0.020 per particle vs LR(N) −0.068 (PFN −0.047). H18: soft particles that add energy flow move QG nets −0.068 to −0.081 per particle, about an IRC-safe EFN (−0.060/−0.067). H18's e1 contrast not used: added particle 0.13 GeV in QG vs 0.99 GeV in JetClass; earlier 0.7, +0.7 round D2; was 0.0) | H5/H5v (0.75): add one soft particle at a realistic position (C2ST 0.50) −0.024 to −0.004 logit (LR(N) +0.097, PFN +0.057); remove softest ≤ 0.011 (LR(N) −0.10); random-split slope 0.024–0.062 vs LR(N) 0.095, Poisson 0.463; hardest split +0.08 to +0.21 = ΔN-0 control. Which hard-core function (pT^D or another) is not identified; split jets off-manifold (C2ST ≈ 1); PN QG checkpoints interim. Round D3 (=, H14 not yet verified): on Herwig 7.1 test jets (N 40,000) the Pythia-trained ParT, PN (final checkpoints), LorentzNet respond −0.016 to −0.002 to an added soft particle vs Herwig LR(N) +0.089; split − ΔN = 0 control −0.019 to +0.011 on both generators while an EFP+N GBDT gives +0.058 to +0.084 (counting-plus-IRC-safe-shapes excluded); z1 at fixed pT^D ≤ 0.006 per +0.01; local pT^D response of the size of the split but the IRC-safe EFN responds too. Herwig-trained models running |
| D8 an IRC-safe, axis-free description (EFP/ADO guided search) reproduces the taggers' rankings on natural jets but not their axis, translation, collinear-split and pT-scaling responses; the axis part is reproduced by replacing m with the axis-moment mass m_A about the given axis (new, 2026-10-19, H8/H8v) | 0.5 (new) | H8/H8v (0.5): ADO 0.92–0.98 for W/QCD, top/QCD, JEDI Z/W (ParT Z/W 0.66–0.76); axis-free surrogates 0 % under WTA / translation (by construction) vs networks 0.7–54 %; m_A surrogate: JEDI Z/W WTA 37.7 vs 37.6 % (q 0.96, r 0.88), W/QCD 22.5 vs 17.1 %; ParT same order, larger (r 0.3–0.6); split (0.1–3.4 %) and JEDI boost (Z/W 18.0 vs 2.3 %) not reproduced; selected sets beyond the first observable sample dependent |
| D4m the axis-convention response follows each model's own mass formula, not generic sensitivity to shifted inputs | 0.45 (= round D5, 2026-10-20: H23 adds, not verified, that Top ParT/PN fed edited jets on a fixed axis respond to a pure translation, dTN −1.1 to −2.9 log-odds, as the JetClass nets do; −0.05 round D4, 2026-10-20: H16v TC control, a translation that keeps m, pT_J and m_A fixed and moves the centroid, changes 4.4–13.8 % of the calls of 30 small binary top taggers vs 1.3–2.8 % for a matched rotation, so in these nets a translation response exists with the model's mass formula unchanged; H19v: JetClass ParT/PN respond to a 0.13–0.15 axis shift alone, dTN −1.6 to −2.1 log-odds; earlier 0.5, +0.05 round D2: H8v m_A surrogate reproduces JEDI-linear's WTA Z/W call changes, 37.7 vs 37.6 %, per-jet r 0.88; ParT only partly, r 0.3–0.6) | X9v: M2-held translation removes the W → Z flow (4–11 % vs 39–81 %) in five networks but 17–31 % of calls still change (rotation 4–9 %); X8v: under re-axis vs a random-direction translation of equal length, the random translation changes more calls (p 7e-25); X7/X7v (0.4): W/Z flow follows the sign of Δ in both directions; stored-trained models: a Δ = 0 translation of equal size changes 0.30–0.42 as many calls with no W → Z flow, a random-direction translation with the same Δ the same calls (JEDI 35.4 vs 31.7 %); hardest-trained models: Δ = 0 translation 1.1–3.0× the edit; E23v: at natural offsets (6–15 mrad) recentring and a perpendicular translation change calls at the same rate (−0.11 [−0.49, +0.24] points). Round D3 (=): H12's "supports D4m/A4" not adopted (H12v: the boost fixes both m and m_A, so it cannot tell them apart); H13: the axis-only slope fitted on pure-axis edits does not predict the pileup response (obs/pred 0.13–0.67 or opposite sign; Top ParT 1.15 [1.05, 1.27]), which H13v attributes to count and mass content rather than the axis, so D4m's scope (axis-convention changes) is unchanged |
| D9 hls4ml-trained taggers (JEDI-linear and 10 other nets) are not invariant under a boost along the jet axis that keeps the invariant mass and m_A fixed; JetClass and Top Landscape ParT, ParticleNet (and Top LorentzNet) change several times fewer calls under that boost than under pT-only or angle-only scaling (new, 2026-10-19, H12/H12v; H12's proposed wording "hls4ml nets read mass through opening angles and pT at their training pT; wide-pT or four-vector training gives a mostly invariant-mass reading" not adopted: the cause of the difference is untested) | 0.55 (= round D4, 2026-10-20; scope: H16v splits H12's five-class 21–38 % by class: 67–77 % of changed calls are on W/Z jets, top-call changes on t/q/g jets 3.6–10.7 %; H16/H16v: identical small binary taggers trained on hls4ml or Top Landscape change 3.7–7.5 % of calls under B0.9 vs 0.3–0.9 % for a rotation, so the hls4ml-vs-Top contrast is not caused by the training sample in small nets, see D12; large models trained on hls4ml untested) | H12 (seed 301) / H12v (seed 917): % calls changed at pT × 0.9, boost B / rotation R: JEDI-linear n64 32.5 / 2.6 and 32.9 [32.2, 33.7] / 2.4 [2.2, 2.6]; all 11 hls4ml nets 21–38 vs 0.3–4.2; Holm p ≤ 3e-3 (H12v ≤ 2e-7). JetClass B / P / A: ParT_full 4.5 / 6.9 / 14.4 (H12v 4.36 / 6.89 / 14.5), PN_full 4.3 / 8.1 / 14.5, kin 5.2–5.8 / 17.5–20 / 21–22; Top ParT λ 0.95 0.6 / 1.8 / 1.9 (H12v λ 0.9 0.94 / 3.34 / 4.05). Independent boost implementation gives identical calls. Held at 0.55: m vs m_A not separated; Top/QG edited jets mostly off the training pT window (in-window B0.9: Top 0.32, QG 0); C2ST 0.69–0.93; absolute rates not comparable across datasets |
| D10 under diffuse pileup-like soft contamination no tagger tested acts like a groomed observable; JEDI-linear responds to the added ungroomed mass, the particle-ID JetClass ParT/ParticleNet and the Top Landscape ParT, LorentzNet and PFN mainly to the number of added soft particles; the induced axis offset is not the cause (new, 2026-10-19, H13/H13v; H13's proposed wording "JEDI-linear acts like the ungroomed m window; JetClass and Top networks fall between ungroomed and groomed; the response comes from the added soft momentum and mass" refuted by H13v except for JEDI-linear) | 0.4 (= round D5, 2026-10-20: H23 pu arms, calls changed differ ≤ 1.5 pp between fixed-axis and recentred builds, e.g. ParT_full pu_40 70.0 / 70.2 %, JEDI-linear 68.8 / 69.0 %; not re-run by the verifier; = round D4, 2026-10-20; H18/H18v: the count response is model- and dataset-specific, not caused by the diffuse edit: under an exact collinear split of soft particles JetClass W-vs-QCD networks move 1.2–2.3× LR(N), QG networks 0.1–0.3×; c4_20 − c4m_20 (same added four-momentum, n particles vs one) JetClass W −0.023 to −0.185 per extra particle, QG −0.022 to +0.026. Corrections: c4 adds 10–40 GeV of core pT; about 40 % of Top LorentzNet's c4 response comes from the added four-momentum (exact-split slope −0.018 per particle); Top networks and JetClass ParT_full move toward QCD under exact splits while LR(N) moves toward top) | H13 (N 10k per dataset, seed 302) / H13v (fresh seeds 3131–3133): W efficiency at ρ 5: JEDI-linear 87 → 42 / 88.2 → 43.1 %, W(m) 93 → 36 / 93.6 → 36.9, W(m_SD) 77 → 78; JetClass ρ 40 ParT_full 80 → 26, W(m_SD) 72 → 61 (65). H13v content controls at ρ 5 (edit / same particles at r/20 / one particle with kept m and ΣpT): JEDI 43.3 / 87.5 / 39.0; ParT_full 72.0 / 64.2 / 76.8; Top ParT 0.71 / 0.71 / 0.92. Axis offset 2.5–13 mrad, translation by it 0–3.3 % of calls. Held at 0.4: toy pileup (uniform disc, massless), no pileup in training, r/20 control also moves soft particles relative to the prongs; Top working-point comparison changes sign at ρ 40 (ParT +0.09 [0.06, 0.12] above top(m, τ32)); d-h13-verify (hls4ml m_SD on fresh rows) still running |
| D11 JetClass-trained ParT and ParticleNet (full and kinematic inputs) use the W-candidate pair mass inside top jets beyond total mass and τ32, at the level of Top-Landscape-trained models, when each edited jet's inputs are computed about its own axis (new, 2026-10-20, round D4, H19v; H19's proposed reading "JetClass nets respond only when the W pair is raised to 100 GeV; the JetClass-vs-Top contrast depends on edit direction" refuted by H19v: produced by building edited jets about the unedited axis; quantitative clause refined 2026-10-20, round D5, H23/H23v: "about as much as a reference given the W-pair mass (0.8–1.25× MLPxW at 60 GeV, 2.9–4.5× MLPx), ordered in |mW − 80|") | 0.6 (+0.15, round D5, 2026-10-20: H23 on fresh rows (seed 9523) and H23v on independent rows (rng 6623/6624): rc T60 ParT_full −1.92 [−2.25, −1.57] (4.39× MLPx) / −1.59 [−1.93, −1.25] (3.59× MLPx, 0.99× MLPxW); all four nets 2.9–4.5× MLPx (Holm ≤ 9e-6), 0.80–1.15× MLPxW (Holm ≥ 0.6); T100 4.3–6.3× MLPx; dose ordered 60 > 70, 100 > 90 GeV (Holm ≤ 4e-8). H3v's k-means edit rebuilt: ParT_kin, PN_kin 2.7–3.0× at 60 GeV. Held at 0.6: full-model non-translation residual (ParT_full +0.48, PN_full +1.20) unexplained; edit off-manifold at 60 GeV; earlier 0.45 new) | H19v (two passes, fresh rows): recentred W − N on JetClass tops 550–650 GeV, T60: ParT_full −1.80 [−2.03, −1.59] (4.06× [3.48, 4.76] MLPx), ParT_kin 4.21×, PN_full 3.01×, PN_kin 3.25× (Holm 0.0195); T100 4.1–5.7×; T80 ≈ 0 (MLPxW +0.07). On fresh Top Landscape tops: JC ParT_kin 3.76× [3.35, 4.25], PN_kin 3.11× (T60), 6.0× / 4.7× (T100); calls at ε_S 0.5 ParT_kin 73.6 → 24.8 % (W) vs 62.5 % (N), McNemar p 4e-43. Held at 0.45: rests on the verifier's passes only (not independently re-verified); JetClass full models keep dN −1.1 to −1.4 to the non-W control after recentring (source not identified); edit off-manifold at 60 GeV (H4v C2ST 0.98); Q80 n = 20 / 17 |
| D12 for small top taggers (JEDI-style, transformer, PFN) the sensitivity to a mass-preserving boost along the jet axis comes from the architecture, inputs and task, not the training sample: trained on hls4ml (full pT or a ±8 % pT window) or on Top Landscape they change a similar share of binary calls; with absolute-pT inputs the training sample sets the direction of the change (new, 2026-10-20, round D4, H16/H16v; H16's "every net responds to moments about the given axis rather than invariant mass" not adopted: H16v TC control) | 0.55 (new) | H16 (seed 4161) / H16v (seed 9917): B0.9 % binary calls changed, 30 nets: frac hlsA 4.7–5.3, hlsB 4.1–5.8, top 3.9–5.1; abs hlsA 4.3–7.5, top 3.7–5.5; rotation 0.3–0.9; B vs R Holm ≤ 3e-18; frac dataset contrasts −0.5 to +1.2 pp (seed spread up to 0.85 pp); frac B = A within 0.3 pp. Direction (→ top / → bkg): hlsA-abs 4.0–6.6 / 0.3–0.9 %, top-abs 1.0–1.5 / 2.5–4.0 %; abs-PFN +3.0 to +3.6 pp. Held at 0.55: two seeds; hlsA vs hlsB a weak pT-width test (65 % of hlsA jets already in the window); binary task only; large models untested |
| D13 top taggers (Top Landscape ParT, ParticleNet, LorentzNet, PFN, EFN; JetClass ParT_kin, PN_kin) use the radiation pattern inside the three prongs beyond jet mass and the W-pair mass, but do not follow τ32 as a single number (new, 2026-10-20, round D5, H25/H25v; H25's proposed reading "these taggers use prong-shape information (τ32 / subjet width) beyond m and mW, comparable to a τ32 classifier" adopted only for the existence of the response; the τ32 attribution not supported by H25v) | 0.5 (new) | H25 (seed 9733, 3,000 + 3,000 per sample) / H25v (fresh rows, 2,000 + 2,000): subjet-width edit S at fixed m and mW, S − c1 on Top tops at s = 1.4: Top ParT −1.14 [−1.24, −1.05] / −1.15 [−1.26, −1.04], PN −1.48 / −1.39, LorentzNet −1.23 / −1.19, EFN −1.43 / −1.47, PFN −0.67 / −0.67, JC ParT_kin −1.16 / −1.11; MLP3 (m, τ32, mW) −0.76 / −0.85, MLPx (no τ) −0.62 / −0.65, LR(m, mW) −0.01; Holm < 0.05 109/112 and 55/56, all with the τ32 sign. Profile edit P at fixed subjet masses (H25v): dτ32 +0.063 / −0.086, MLP3 −0.48 / +0.60, MLPx ≈ 0; nets at P1.6 (sharper cores) ParT +0.72 [+0.63, +0.82], PN +0.98, LorentzNet +1.05, JC PN_kin +0.83, PFN −0.37; at P0.5 −0.11 to +0.28; 26 of 46 significant cells with the τ32 sign. Held at 0.5: S changes subjet masses, mSD and τ21 with τ32; MLPx predicts the S sign; edited jets off-manifold (C2ST 0.71–1.00); s ≥ 1.4 and P1.6 sets selected; P not pre-registered, two exponents only |

## Proposed thesis revision after the round-2 review (2026-10-17; pending the user's decision)
The external-style review (round 2: soundness 6, presentation 4, contribution 5, overall 5) finds the statistically strongest and most
novel line is the mass/axis result, not the "statements → objects" framing (whose identity is definitional and whose sparse cuts do not
generalise). Proposed Paper A thesis v3: "Jet taggers compute the simplest formula that is correct on their training data. Causal
edits identify which one: JEDI-linear, ParT and small taggers read the jet mass as the momentum-weighted spread about the given axis,
not the invariant mass; this makes their decisions depend on the jet-axis convention by an amount the formula predicts; training with
an off-centre axis largely removes it." Statements/objects, the neuron atlas and the if-statement program become the method and
supporting sections (Paper B keeps the faithfulness result). Keep v1/v2 above until the user decides.

Round-4 tests of thesis v3 (2026-10-17; E23/E23v, X7/X7v, B5/B5v). Clause by clause, as measured:
- "read the jet mass as the momentum-weighted spread about the given axis, not the invariant mass": JEDI-linear on unedited off-centre
  jets, recentring with m fixed lowers the read-out (H0 invariant mass excluded, Holm p ≤ 1e-10); the read-out follows the sign of
  ½Δlog(M0M2) for three edits (recentre, reflect, perpendicular translation) but with two gains (0.6–0.8 and 1.3–1.8), so it is not a
  single function of √(M0M2) (E23v). ParT and small taggers: unchanged from P1/X2.
- "decisions depend on the axis convention by an amount the formula predicts": stored-trained models, direction of the W/Z flow follows
  the sign of Δ (X7, X7v), and the share of calls changed is 0.96–1.21 of a matched real change; 0.30–0.42 of the calls also change under
  a Δ = 0 translation (X7v m2t), so the size is not set by the formula alone. At natural centroid offsets (1–13 mrad) call changes follow
  translation size (E23v).
- "training with an off-centre axis largely removes it": hardest-trained models change 10–22 % of calls on the stored axis (read-out
  0.16–0.62 of a matched real change) and respond to a translation-invariant spread at Δ = 0 (23–31 % of calls, W → Z) (X7v).
- compact form (review weakness 3): no ≤ 50-term program reproduces the shift response (B5/B5v).
Proposed v3.1 wording (2026-10-17, pending the user; v3 above kept): "Jet taggers compute a formula that is correct on their training
data, and causal edits identify which one. JEDI-linear, ParT and small taggers trained with the centred axis read the jet mass through
pT-weighted moments about the given axis, not through the invariant mass: on unedited off-centre jets an edit that keeps the invariant
mass fixed moves JEDI-linear's mass read-out with the sign of ½Δlog(M0M2), and under a change of axis convention the W/Z calls move in the
direction this quantity predicts. Models trained about the hardest particle follow the same quantity about their own axis together with a
translation-invariant spread." 

Round-5 tests of thesis v3 (2026-10-19; X8/X8v, X5/X5v, X9/X9v). Clause by clause, as measured:
- "JEDI-linear, ParT and small taggers read the jet mass as the momentum-weighted spread about the given axis": the translation W → Z
  flow requires trained weights in JEDI-linear, Deep Sets and a JEDI-style float model (X8); it appears in float, L1 and HGQ Deep Sets
  (X5) and in PFN and EFN (X9). In all five X9v networks a translation that holds M0 and M2 about the stored axis fixed leaves 4–11 %
  W → Z and a read-out ratio of −0.18 to 0.02 (invariant-mass reader −1). Measured scope: set architectures trained on stored-axis
  inputs; whether the input representation alone produces it (recentred-input or augmented training) is not tested (X9w).
- "decisions depend on the jet-axis convention": the WTA accuracy loss (0.218) is smaller than that of a random-direction translation
  of the same per-jet length (0.250) (X8v), so its size follows the displacement, not the WTA direction.
- Supporting sections: the rank-window asymmetry is produced by the zero-bit mask layout with permuted weights (X8v); pure first-layer
  cuts are at the sparsity null for L1 and PFN and above it only with the HGQ training script (X5v).
No change to the v3 / v3.1 wording (pending the user); the clause "JEDI-linear, ParT and small taggers" can be extended to "set
architectures trained on stored-axis inputs (Deep Sets float / L1 / HGQ, PFN, EFN)" if the user adopts v3.1.

Round-6 tests of thesis v3 (2026-10-19; X10/X10v, X11/X11v, X12/X12v). Clause by clause, as measured:
- "read the jet mass through pT-weighted moments about the given axis": in 8 retrained JEDI-linear networks (HGQ and pipeline-matched
  float) the 0.06 rad translation response is removed by holding Σ pT r² fixed; relative to a real widening with the same M0 and M2,
  float and HGQ networks respond alike (0.69–1.30 vs 1.02–1.38) (X12v). Inside the official network, M0 and M2 are each held in a
  1–2-dim subspace from S on (DAS effective IIA 0.72–0.87) (X10).
- "the formula ½log(M0M2)": not found as one internal variable; the subspace learned for it reproduces a global pT rescale, and an
  angle-only change of the same quantity runs through the M2 subspace (X10v). The thesis statements that name the product as the
  network's variable are not supported at the representation level; the call-level statements (sign of ½Δlog(M0M2)) are unchanged.
- Supporting sections: the official 13|14 rank-window effect is not reproduced by 200–300-epoch retrainings (X12v); "pure cuts" reduce
  to HGQ sparsity (X12); the S units are a preferred, nameable basis against random rotations and SAE features (X11v), as is Deep Sets
  at a lower rate.
No change to the v3 / v3.1 wording (pending the user). If v3.1 is adopted, "pT-weighted moments about the given axis" is the measured
form; "½log(M0M2)" should be stated as the call-level predictor, not as a variable the network forms.

Round-7 tests of thesis v3 (2026-10-20; X9w/X9wv, E24/E24v, B6/B6v). Clause by clause, as measured:
- "read the jet mass through pT-weighted moments about the given axis" (scope): in PFN, EFN and Deep Sets the 0.06 rad translation
  response is present after stored-axis training (0.35–0.71 W → Z) and at the rotation-null level after training with random rigid
  translations up to 0.08 rad (≤ 0.012; seed-level p 0.0093), at ≤ 1.13 points accuracy. Beyond the trained offsets (0.12 rad) the
  augmented networks respond again (0.51–0.84), so augmentation does not produce a centroid-relative mass; recentred inputs give 0 by
  construction. In JEDI-linear HGQ-50 augmentation removed the response in 1 of 3 seeds, with 4–14 points accuracy loss (X9wv).
- "training with an off-centre axis largely removes it" (v3 clause): for these three architectures, training with offsets removes the
  response inside the trained offset range only; the v3 clause would need that scope.
- Supporting sections: the radial S objects are nearly unchanged by an azimuth scramble at fixed moments (≤ 0.19 sd), and the call
  change above rotation passes through the context-only units S58/S31; the call-level size equals that of a τ21-using classifier and
  of Deep Sets / PFN, so neither prong use nor an object-specific call signature is shown (E24v). The blind rank-window program
  reaches the edit fidelity per bit of tuned trees and GBDT on four architectures, not more (B6v).
No change to the v3 / v3.1 wording (pending the user). If v3.1 is adopted, "set architectures trained on stored-axis inputs" can
carry the qualifier "without translation augmentation over the offsets considered" (X9wv).

## Main results, A vs B (and the merge rule)

- **A's main result:** the complete program of a deployed tagger. For JEDI-linear: ~13 particle cuts → ~11 first objects
  and a jet context → 19 second-round objects (named energy-flow quantities with pT-rank windows) → 16 neurons as statements
  on them → 5 classes as combinations, each stage checked by substitution and by causal edits; the order in which
  high-level variables appear across the layers; and the same stage structure found in ParT / ParticleNet / small
  transformers.
- **B's main result:** the procedure that produces such a program from any tagger, and the test that tells when it is wrong.
  Content that is about the method and not about one model: the exact identity for sum-pooled layers; the MARS
  if-statement fit for the rest; the validation suite (stage substitution, per-particle attribution, causal tuning curves);
  the negative results that decision agreement does not imply mechanism fidelity (G160, G180–G190), measured on several
  architectures; and comparisons with tree extraction, symbolic regression and probes.
- **Merge rule (user, 2026-10-16):** if the papers become too similar or either fails to stand alone, combine them. B stands
  alone only if it has (i) at least one result about the method measured on ≥3 architectures or 2 datasets, and (ii) the
  baseline comparison. Decision point: after the first cross-architecture results (ParticleNet, small transformers, JEDI).
  Default if B fails the test: B becomes A's method section and A's appendix carries the failure modes.

## Paper A: mechanistic interpretability (flagship), JEDI-linear first, then ParT

**Thesis (v1, 2026-10-16):** a jet tagger builds its class boundaries in four steps, each written as an explicit formula:
1. **Statements:** cuts on single particles.
2. **Objects:** groups of particles passing a cut, read through their count and summed momenta and positions.
3. **Neurons:** statements on objects.
4. **Classes:** combinations of neuron objects.

For JEDI-linear the whole chain from input to logits can be written as such a formula and checked stage by stage against
the network.

Claims to establish:
- A1. Each first-layer unit is one cut on (pT, Δη, Δφ). Each pooled channel equals count × offset + weights · (sums over
  the passing particles), exactly. [E2, E3]
- A2 (v2, 2026-10-16, E4; pending verification). The context g shifts the second-round cuts of 9 of 19 live units, and the
  decision needs it through four objects: S0 (ΣpT, via g49 ← P15), S30, and S31/S58 (functions of g alone). g → mean changes
  14.9 % of calls; those four objects alone 14.4 %. v1: "adds jet context; size and role to be measured".
  [2026-10-19, X8/X8v: with g taken from a random other jet (resample ablation, 10–20 derangements) instead of its mean, 27.8 %
  [27.3, 28.3] / 27.9 % [27.3, 28.4] of calls change on two samples (mean ablation 14.6 / 14.85 %, McNemar p ≤ 2e-167). Wording kept;
  the 14.9 % figure is the mean-ablation cost.]
- A3 (v2, 2026-10-16, E5; pending verification). 17 of the 19 second-round objects are each a hinge in two variables
  (pT or ln pT, and r = ΔR to the axis) summed over a fixed pT-rank window; azimuth adds ≤ 0.017 R². Named: S29 ≈ ΣpT·r² +
  Σr² + soft count (R² 0.976; cost 40.9 %), S1 = Σ_{ranks ≥ 1}(r − 0.2)₊, S0 = ΣpT of the 14 hardest, S50 = ΣpT·r² over ranks
  0–38, S19 = N(pT > 1 GeV) among ranks 13–63, S30 = Σ r for r < 0.165. Radial edits on S29/S1 match the hinge prediction
  (slope 1.03–1.04). The radial symmetry is built in round two from round one's directional half-planes [superseded by A3 v4]. v1: "nameable
  energy-flow quantities (counts and pT-weighted sums over regions)".
- A4 (v3, 2026-10-16, E6 + E6v, verified on a fresh sample; verifier confidence 0.55 for the exact form, > 0.9 for
  non-recentring). The network's mass variable is an un-recentred M2-like quantity: M0 = ΣpT and M2 = Σ pT r² are read
  separately at S and multiplied in h4/h5 (S0 carries M0 through the context g). A rigid translation of all particles by
  δ (invariant mass unchanged to 1e-7) raises the read-out mass as δ² with 0.74–0.84 of the ½Δlog M2 prediction at every
  δ ≥ 0.02 and flips 77 % of W calls to Z; the alternatives Σ pT r and Σ r² are excluded by magnitude (×1.7, ×2.1) and
  exponent. It is not an exact function of r² about the axis: a direction-dependent odd term of 12–20 % of the even term
  remains, consistent with its construction from nine asymmetric half-plane cuts (the only angular inputs); the pT·r²
  products form piecewise-linearly in l2/l3 (the response travels a → s → u → S, not through P/g). [2026-10-17, E14/E14v: the odd term is
  the sum of the per-particle second-round maps: with g frozen the maps give 0.96–1.05 of the measured odd part (95 % CIs within
  [0.93, 1.08]); on the point-reflected sample the sample-asymmetry term is −0.0005 to +0.0003 (p ≥ 0.20); symmetrized maps give
  odd 0.0000 [−0.0005, +0.0005] with the even part unchanged; 15–24 % (seed 31) / 9–21 % (seed 8) of the odd part passes through g.]
  [2026-10-17, E15/E15v: "multiplied in h4/h5" is not established. E15 measured a read-out interaction β 0.6–0.7 at h4/h5 (sum 0,
  product 1); E15v showed the same read-out gives −0.75 for ReLU windows on a sum, 0.9–1.15 for ReLU units on separate M0, M2 and
  −0.01 to 1.14 for random ReLU layers on S, so β does not separate the rules. Holds: M0 (S0) and M2 (S50) are separate S units; the
  read-out's gain-normalised M0 response matches the product within 0.06 for M2 > 8.1 GeV from S onward (random ReLU layers on S
  do the same). Open: where, or whether, a product forms. Wording before this date: "M0 = ΣpT and M2 = Σ pT r² are read separately at
  S and multiplied in h4/h5".]
  [2026-10-17, E21/E21v: S-level patches of S0 (M0) with S50 (M2), seed 47 and seed 91, N 10,000 each, no input edit. (i) Ratio
  R = Σ|ΔU| along iso-product / along iso-sum: W − Z margin 0.44 / 0.43, z7 0.40 / 0.39; random S-unit pairs with the same
  displacements 0.36–1.38 (5–95 %), and the two directions are only ≈ 32° apart, so this ratio does not separate product from
  additive dependence (verifier geometry). Per-unit R at h4, h5, z matches random ReLU layers on S; no product stage is located.
  An exact function of S0·S50 or of M0·M2 at the read-out is excluded (margin R 0.39–0.47 in three θ bins). (ii) Per-jet gradient
  ratio, slope of log(g50/g0) on log(S0/S50) (product 1, fixed additive rule 0): margin 0.64 [0.59, 0.68], z7 0.94 [0.90, 0.99], z0
  0.80 [0.75, 0.85], z10 0.59; with the affine moments M0 = 21·S0 + 658 GeV, M2 = 2.6·S50 + 2.8 the margin slope is 1.02 [0.95,
  1.08]; random S pairs (20 draws) −0.59 to 0.66, random ReLU layers on S −0.41 to +0.01, synthetic sum 0, product 1; per-jet
  Spearman 0.21–0.43. (iii) With S29 as a second M2 carrier: margin R 0.275, below all 20 three-unit control draws (0.34–1.75);
  iso-sum changes 23.5 % of calls vs 7.0 % iso-product (controls −0.5 to +2.8 points). Rank p with 20 draws is floored at 0.095.
  Reading: the read-out depends on S0 and S50 approximately as on a product (stronger for the affine moments), with large per-jet
  scatter; where this forms (h4, h5 or z) is not measured; the untrained control is degenerate (one live S unit). The E21v rerun
  numbers come from a job that completed after the verifier's session and were not reviewed by the verifier.]
  [2026-10-17, E23/E23v (verifier confidence 0.6): natural off-centre jets, no off-manifold edit. Samples sample(20000, 101) and
  sample(20000, 211), N 100,000 each; read-out = ridge of log m on z (held-out R² 0.942 / 0.944). Tail sets = top 1–5 % of q =
  |M1|²/(M0M2) or of d = |M1|/M0 (d p50 1.46, p99 13.1 mrad); 72 % top, 0.4 % W/Z in the top 5 %. Recentring (particles moved by −M1/M0;
  |Δlog m| ≤ 4e-8) changes the read-out by 1.50–2.01 (seed 101) and 2.03–2.50 (seed 211) times a matched real change of the predicted
  size (Tq5 −4.2e-3 [−5.1, −3.4] vs prediction −2.4e-3); the reflected move +M1/M0 by 1.06–1.26; a perpendicular translation of equal
  displacement (+, raises M2) by +0.46–0.80; a rotation of equal displacement by 0 within CI; bulk recentring by 0. Per-jet even/odd
  split: the part quadratic in the shift is 0.62–0.71 of its prediction (equal to the perpendicular response), the part linear in M1
  is 1.34–1.82 (CIs exclude 1). Freezing the angle-only P units keeps 57–80 % of the response. Calls: recentring changes 1.6–3.4 % of
  tail calls, the same as the perpendicular translation (Tq5 −0.11 [−0.49, +0.24] points), no Z → W excess (8–10 vs 5–6). Reading:
  on natural jets the read-out follows the sign of ½Δlog(M0M2) about the stored axis; H0 (invariant mass) and H2 (generic translation
  lowers it) are excluded for the read-out; a single gain on √(M0M2) is excluded; the M1-linear term is over-weighted. Observational
  matching of tail and bulk jets is underpowered (half-widths 0.005–0.013 vs predicted 0.002–0.004).] v2: "√(M0·M2) about
  the given axis" (exact form overstated). v1 read "m² ≈ 2·ΣpT·Σ pT r² − 2|Σ pT r⃗|²": the factor 2 was an error.
- A5 (v2, 2026-10-17, E7 + E7v; verifier confidence 0.55). Each of the 16 z neurons written as K = 32 hinge terms on the 19 live
  S objects, substituted all at once (the network's own last layer makes the call), changes 9.2–10.4 % of calls on fresh jets
  (accuracy −3.2 points); random hinge terms on S change 21.6–23.0 %, a random-init S basis 38.7 %. Under three edits without refit
  the S formulas' neuron changes have the network's sign and ordering at 0.57–0.58 (0.06 rad translation; 0.65 after read-out
  calibration; the same inside and outside the fit range), 0.42 (45° rotation) and 0.41 (a32 crossing) of its size; the slope rises
  with K (0.29, 0.39, 0.49, 0.58 for K 4–32). The rotation flips of the S formula cover 20–24 % of the network's flips (random S terms
  12–19 %). Post-translation W → Z (S 87.6–88.5 %, network 76–77 %) is also reproduced by random terms on S (72–86 %), so it reflects
  the S inputs, not the fitted formula. Negative poles (most negative 1 %): z1 W 67–69 %, z6 W 85 %, z8 W 98 %; z0, z2, z3, z11, z15
  t 85–91 %; z4, z14 q 84–86 %. The neuron stage is approximately, not exactly, a function of the objects at this term budget.
  [2026-10-17, E22 + E22v; verifier confidence 0.55: interchange interventions. Setting one unit to its value on a source jet, the program
  (same variable swapped, downstream formulas recomputed) gives the network's new call in 0.19–0.88 of the pairs where the network's
  call changes (three samples; S 0.33–0.77, h4 0.20–0.75, h5 0.25–0.79, z 0.37–0.88). Named: S0 0.76 [jet CI 0.71, 0.80], S29
  0.55–0.57, S1 0.38–0.41 (inside its permuted range), z0 0.75–0.77, z7 0.63–0.72. Swap-free predictor (program's runner-up class
  on the base): stage-level difference z +0.15 [0.07, 0.23], S +0.03 [−0.04, 0.10], h4/h5 ≈ 0; 17/118 units higher after Holm
  (pairs treated as independent). The network's own runner-up class scores 0.75–0.90. Most of the S29/S1 shortfall sits in the
  h4/h5/z formulas: S formulas with the network downstream give S29 0.83, S1 0.74. So the z formulas carry swap-specific agreement;
  per-unit IIA is the measure to state, not the 94 % same call.]
  v1 (2026-10-16): "The 16 neurons are statements on A3's objects. Replacing each stage by its formula keeps the network's calls. [E7]"
- A6 (v2, 2026-10-17, E8 + E8v; verifier confidence 0.6). Each class's calls are described by a rule of 2–3 neuron windows on z
  (precision 96–98 %, coverage 70–82 % on fresh jets; rules of similar quality exist in random rotations of z: 87–99 %, 46–82 %).
  Setting the rule's neurons to the class's in-rule medians moves 26–62 % of jets called otherwise into the class (g 54, q 32, W 41,
  Z 61, t 26 %); the rule's set ranks 6th–78th of the 560 neuron triples with the same values, and triples without a rule neuron move
  4–8 % on average. Moving the neurons just inside their windows moves 3–25 %, while 96–98 % of unedited in-rule jets are called the
  class; the logit change is the exact sum of single-neuron changes through d7. So the windows describe the calls; the causal
  effect is the distance moved along an additive logit margin, not window membership. Windows: z0 (78–81 GeV) and z7 (90–92 GeV) are
  E6's W and Z neurons; the t rule's z8 (124–170 GeV) is not E6's top neuron z10. At S: g 48, t 34, q 21, Z 16, W 6 % (W at the
  level of pairs without a rule unit, 4.7 %).
  v1 (2026-10-16): "Classes are combinations of neuron objects, and imposing a combination causes the class (J33b on the network's
  own objects). [E8]"
- A7 (scope v3, user 2026-10-16: "just use the best version of JEDI"): one JEDI model, 3-feature n64 (the model of all
  earlier J-results), with the untrained n64 network as the control. Cross-model generality comes from the other
  architectures (A10), not from JEDI variants. [E9 → control only]
- A9 (user, 2026-10-16). High-level variables are built over the layers: sums of particles first (P, S), then products
  and ratios (h4, h5), then clustering-type variables, if at all. The layer where a variable first becomes linearly readable
  rises with its algebraic depth. [E10; E6 for mass]
  [2026-10-17, E15v: the product step for mass (sum at S → product in h4/h5) is not located by E15's test; the mass example of A9
  stays open until a read-out-free test separates the combination rule from the pT-threshold features.]
  [2026-10-17, E21/E21v: the S-level patch test (no input edit, so no pT-threshold features move) does not locate a product stage:
  per-unit iso-product/iso-sum ratios at h4, h5, z match random ReLU layers on S. The per-jet gradient-ratio test gives a
  product-like dependence of z0, z7 and the W − Z margin on (S0, S50) (slopes 0.64–0.94; random pairs ≤ 0.66), measured on the
  read-outs only. The mass example of A9 therefore has "sum at S" and "product-like dependence by z", without the stage of the product.]
- A8. ParT follows the same scheme: pair-bias statements, head routing, messages, class-token neurons, and combinations. [P-series]
- A11 (user, 2026-10-16: "similar to the Goodfire post, look at neurons too"). Most live units have one main job that can be
  stated as a physical pattern (jet images + observables), with weak activations that are neighbours of the main pattern and a
  readable negative pole; units without one are listed. Tally per stage, analogous to Goodfire's 90 / 7 / 2–4 %. [E16 neuron atlas;
  later the same on the small transformers' MLP neurons and heads across training checkpoints]
  [2026-10-19, X11/X11v (verifier 0.85): new sub-claim A11b. Preregistered blind naming (≤ 8-term MARS on 97 rank-window observables,
  held-out R² ≥ 0.9): JEDI-linear S units 0.79 (15/19, same units on three samples) vs 0/250 random rotations reaching it (max 0.63;
  p 0.005, 0.020); h4 0.55–0.56 vs ≤ 0.36; SAE 2×/4× features 0.16–0.21 and PCA 0.11 are named less often. Deep Sets pooled units
  0.29 vs ≤ 0.12 (rotations); PFN latent 0.10, inside its rotation range. Untrained JEDI/Deep Sets architectures: 0/56, 1/57, 0/46.
  A11 wording kept.]
- A12 (2026-10-17, E18/X4): the un-recentred mass follows from the training data: models retrained with off-centre (hardest-particle)
  axes mostly recentre (axis-jump response 0.13–0.41 of a matched real change; both pure hypotheses rejected); residual W → Z 7–22 % for
  soft-prong-axis jets. Narrative: a network learns the simplest formula that is correct on its training distribution.
  [2026-10-17, X7/X7v (verifier confidence 0.4): reverse direction. Deep Sets / Linformer trained about the hardest particle, evaluated
  on stored-axis inputs (20,000 and 10,000 jets, two seeds): 10.1–11.1 % / 17.6–21.6 % of calls change, Z → W 337–1997 vs W → Z 0–233,
  true Z called W 5 → 19 % (Deep Sets), 4.5–4.8 → 45–50 % (Linformer); read-out 0.16–0.62 of a matched real change. A translation of the
  same size with ½log(M0M2) about the hardest particle held fixed changes 23–31 % of their calls with W → Z 886–1207 vs 1–68 (true W
  called Z 2.5–2.8 → 42.6–56.5 %). Reading: these models read ½log(M0M2) about their own axis and also a translation-invariant spread;
  the reverse response is the net of the two. "Mostly recentre (0.13–0.41)" holds for the X4 axis-jump edit; under the stored-axis edit
  the read-out ratio is 0.16–0.62. Wording before this date: "models retrained with off-centre (hardest-particle) axes mostly recentre".]
- A10 v3 (2026-10-19, X5/X5v, X9/X9v, X8; verifier confidences 0.72, 0.7, 0.65). General across the set architectures tested on
  the same stored-axis inputs (JEDI-linear, Deep Sets float / L1 / HGQ, JEDI-style float, PFN, EFN): the un-recentred mass response
  (0.06 rad translation W → Z 0.43–0.89, rotation ≤ 0.09; calibrated read-out 0.71–1.30 of the √(M0M2) prediction), carried mainly by
  M2 about the stored axis (M2-held translation W → Z 4–11 %). Measured only in JEDI-linear: the rank-window response (2.96 with
  matched unit selection vs PFN/EFN 0.84–0.87; no Deep Sets regime forms windows), the pure cut types above their null (L1 and PFN at
  the null; HGQ-script Deep Sets 30–50 % of units, 2 of 3 seeds, pipeline-confounded), first-layer sparsity of 13 units, and radial Φ
  maps with share ≥ 0.98 (PFN 0.48–0.56, EFN 0.33–0.37). Hypothesis "the sparse cuts come from HGQ fixed-point training" (A10 v2):
  not established; in Deep Sets, HGQ-script training gives purity above null at low sparsity, the quantization itself not isolated.
  A10 v2 wording kept below.
- A10 note (2026-10-17, X6/X6v): across four official JEDI-linear checkpoints (n32, n64, n128, perminv n64) the pure pT-only /
  angle-only cuts are not separated from random placement at HGQ sparsity (p 0.16–0.48), so the "sparse cuts come from HGQ training"
  hypothesis gains a sparsity explanation for purity; X5 (Deep Sets with HGQ / L1) still tests it across architectures.
- A10 v2 (2026-10-17, X2): general across small architectures: axis-relative mass (invariant mass rejected) and layer ordering by
  complexity (ρ 0.85–0.90). Not general: JEDI-linear's sparse 13-cut first layer (Deep Sets uses 63–64 mixed units) and the size of the
  axis-moment response (0.07–2.0 of √(M0M2) across architectures and seeds). Hypothesis to test: the sparse cuts come from HGQ fixed-point
  training. Thesis A impact: "statements and objects" are exact for sum-pooled networks, but their readability depends on training.
- A10 (user, 2026-10-16). The same stages exist in other architectures: ParticleNet (EdgeConv: statements on particle pairs
  in k-NN graphs, pooled), L-GATr (Lorentz-equivariant tokens), and small transformers / Linformers trained on hls4ml. [X-series]

Refinement (2026-10-16, E3/E3q): the identity is exact in floating point. With the network's fixed-point rounding, the
float version of the particle and object stages changes 3.8 % of calls. The paper states the float program, and the rounding
separately as a perturbation.
E2 detail for the narrative: the angular statements are half-planes through the jet axis and carry no pT weight, so the first
objects are particle-count-weighted angular spreads, not energy-weighted ones.

Refinement 2 (2026-10-16, E3w/E3m): the right float program keeps HGQ's zero-bit entries as fixed masks (80–84 % of the
slot × channel entries in the second round). Float + masks scores 80.68 % (fixed point 81.29 %, 4.49 % calls changed); fully
float 43.7 %. So an object is a cut AND a pT-rank window (e.g. u0: the 14 hardest particles; u19: ranks 13–63). Claim A1 now
reads: statement = cut on (pT, Δη, Δφ); object = particles passing the cut within a fixed rank window, read by count and sums.

Refinement 3 (2026-10-17, E20 + E20v; verifier confidence 0.75): the rank windows act causally on the network's own objects.
Edit: swap only (Δη, Δφ) of two adjacent pT ranks (pT order kept). Across the 13|14 edge S43, S52, S63 change by 0.0841 [0.0813,
0.0871], 0.0560, 0.0838 per jet (seed 77); at the adjacent inside pair 12|13 by 0.0014–0.0043; at 14|15 by 0 (DiD +0.055 to +0.081,
Holm p ≤ 4e-162). With the masks shifted by −3 / +3 the peak moves to 10|11 / 16|17 (0.052–0.088). The same edge peak appears at
each unit's own qs1 edge for S13 (10|11), S17 (9|10) and S62 (4|5); not for S1 at 0|1, S20, S51, S56. The 0–13 window is written
three times (add-layer zero-bit mask on s, l3 input mask, qs1); any one alone keeps the peak; qs and l2 do not set it. Removing l3 and
qs1 masks (with the add masks also absent) removes 95–98 % of the response and costs 26.6 points. Masks permuted across slots at
fixed density: 32–36 % accuracy. Inside-window swaps are not null (above 1 LSB for up to 27 % of jets): in the float program from
the non-separable (pT, r) hinge, plus a part absent from the float program (rounding, not separately tested). S0 and S19 (pT-only)
are not tested by angle swaps. Wording before this date: "an object is a cut AND a pT-rank window" (from masks, no causal test).

Refinement 4 (2026-10-17, X6 + X6v; verifier confidence 0.45): other official checkpoints. In 3-feature n32 / n128 and
3-feature-perminv n64, every l1 unit with a nonzero weight is pT-only or angle-only (57/57); at the observed kernel sparsity random
placement of the nonzero entries gives this with p = 0.16–0.48 per model (product 0.009, Fisher 0.31), so the pure-cut pattern is not
separated from HGQ sparsity. Rank windows: the qs1 masks of the sorted-input models contain a hard-particle window (0–9, 0–13, 0–19 at
n = 32, 64, 128; other prefix windows exist); an angle swap across its edge changes the windowed S units 2.4–5.4× more than an inside swap
(V: 5.37, 2.36, 4.59; the n128 edge chosen post hoc), 0.71–0.82 with masks removed. At five other tested edges the ratio is 0.27–1.00 or
≈ 0 (n64 0|1 S1; n128 61|62 0.32 [0.27, 0.38] with radial units S21, S48, S54). Removing slot masks at real-particle slots costs 2.7 / 18.4
/ 20.7 points (n32 / n64 / n128); E3m's "fully float 43.7 %" also admits padded slots (n64 real slots only: 62.3 %). perminv has no
slot-dependent masks and is +0.90 [0.58, 1.22] points more accurate than n64 on M (paired, p 3e-8; V +0.49 [−0.04, 1.02], p 0.07), with
the same cut types, radial objects (0.989–0.995) and shift response (W → Z 0.56). Reading: in this model family an object does not
require a rank window; where windows exist (sorted inputs) the hard-particle window acts causally. Wording kept: "object = particles
passing the cut within a fixed rank window" holds for the n64 model as measured by E20/E20v; its generality is limited to sorted-input models.

Refinement 5 (2026-10-19, X8 + X8v, X9v; verifier confidences 0.65, 0.7): what the rank-window edit measures. On fresh seed 7307 the
13|14 vs 5|6 across/inside ratio of the edge units (S0, S43, S51, S52, S63) is 5.19 [4.96, 5.42]; the trained network's 14 live S
units without a qs1 change at 13|14 give 1.45; weight-permuted copies with the masks kept give 2.07–11.34 (+ 2 × ∞) on their edge units.
The edge asymmetry is therefore produced by the zero-bit masks with any weights. With units selected as for PFN/EFN (top 10 by
resample cost) JEDI-linear's ratio is 2.96 [2.88, 3.05]; at 0|1 vs 3|4 it is 1.93 (PFN 1.97–2.20). Wording kept: "object = particles
passing the cut within a fixed rank window" (the masks define the window); added scope: the window is part of the mask layout fixed
by HGQ training; E20/E20v's edit does not show that the trained weights use it beyond what the masks impose. Masks-vs-weights split
with non-constant mask-permuted networks: not available (all variant-b copies constant-call; X8w).

Refinement 6 (2026-10-19, round 6: X12 + X12v, X10 + X10v, X11 + X11v; verifier confidences 0.55, 0.45, 0.85). Retrained JEDI-linear
n64 (same architecture; HGQ-150 s0–2 primary, HGQ-50 s0–2, pipeline-matched float s0–1; 200–300 epochs vs the official 2,868):
- A1 window clause: all quantized retrainings form zero-bit qs1 masks (0.84–0.94), but none reproduces the official 13|14 swap
  response (0.20–0.90 at the same edge vs 5.18; 0–1.5 at their own top edges, ranks 24–45; HGQ-50 s0, s1 show a localized peak at
  their own edges near ranks 40–42). Scope added: the rank-window effect at the hard-particle edge is a property of the official
  checkpoint (or of its long training), not reproduced by shorter retrainings. Wording of A1 kept for the official n64 network.
- A1q reworded (old, 2026-10-19 round 5: "pure pT-only / angle-only cuts come from HGQ training"): HGQ training produces the
  first-layer sparsity (13–20 nonzero weights of 192; zero fraction 0.896–0.932); at that sparsity every quantized first layer and
  both sparsity-matched nulls are 100 % pure, so purity adds nothing beyond sparsity. Dense float first layers are 53–58 % pure, near
  their null (Holm 0.14).
- A3: radial (pT, r) hinge objects are more frequent with quantization in every network (HGQ-150 16/20 vs float 7/40), network-level
  permutation p 0.048 (all usable HGQ vs float) / 0.17 (primary arm). X11: 15/19 official S units named by rank-window formulas,
  above every one of 250 random rotations.
- A10m: the translation W→Z response, removed by holding Σ pT r² fixed, recurs in 4 of 5 usable HGQ retrainings and in both float
  ones. X12's "2–3× weaker without quantization" does not hold after a matched real-change calibration (X12v): flow(T)/flow(dilM2)
  float 0.69–1.30 vs HGQ 1.02–1.38 (p 0.24); the float networks call fewer ~105 GeV-equivalent W jets Z.
- A4 product / stage: X10's DAS subspace for Δ = ½log(M0·M2) reproduces a global pT rescale, not an angle-only edit of the same Δ
  (0.19–0.43 vs full-swap baseline 0.61; the angle edit follows the M2 subspace, overlap 0.68–0.99). Δ is not shown to exist as one
  variable at S, h4 or h5. M0 (S, k 1: 0.85–0.87) and M2 (S, k 2: 0.72–0.75) are held in low-dimensional subspaces from S on.

Refinement 7 (2026-10-20, round 7: X9w + X9wv, E24 + E24v; verifier confidences 0.5, 0.3).
- A10m scope (X9w/X9wv): PFN, EFN and Deep Sets (2 seeds each) trained with random rigid translations of the training jets
  (|c| ~ U(0, 0.08) rad) give 0.06 rad W → Z ≤ 0.012 vs 0.35–0.71 for stored-axis training on two samples (stratified seed-level
  p 0.0093), at ≤ 1.13 points accuracy per network; a 0.12 rad translation sends 0.51–0.84 of their W jets to Z (rotation ≤ 0.01).
  Recentred-input networks give 0 (by construction). New claim A10r (0.55). X9w's proposed reading "augmented networks read mass from
  the spread about the jet's own centroid" refuted by the 0.12 rad control (X9wv). JEDI-linear HGQ-50: augmentation reduced the flow
  to the null level in 1 of 3 seeds; the other two keep 0.38–0.50 with 6–14 points accuracy loss (architecture vs under-training
  open; X9x). A10m wording kept; A10m 0.85 (=).
- A3 (E24/E24v): under a per-particle azimuth scramble at fixed pT, r, rank and fixed M0, vector M1, M2 (W τ21 0.26 → 0.72), S0 and
  S19 change by 0, S29/S30 ≤ 0.09 sd, S1 0.18–0.19 sd. The excess of calls changed over rigid rotation (2.8–3.2 points on three
  samples) passes mostly through the context-only units S58 and S31 (freeze: 2.99 → 0.54 points), which read g, formed from the pooled
  half-plane units. A3 wording kept (0.85, =); new claim A3z (0.6). E24's planned predictions that failed: B ≈ A′ (B 2.5 % vs A′ 7.2 %),
  residual in the odd direction units, residual vanishing under A′.
- New claim A13 (0.5): JEDI-linear's call change under the scramble (7.07 %) equals that of a boosted tree using τ21 (6.89 %,
  McNemar p 0.60); its excess over rotation (+2.79) lies between a mass-and-radial tree (+1.12) and the τ21 tree (+6.38), and its
  accuracy drop (1.45 points) equals the τ21 tree's (1.14). E24's reading "no evidence for τ21-like information beyond the radial
  profile" (2026-10-19) not kept: on these jets τ21 adds 0.22 points of accuracy over mass and radial features, so the edit has
  little power to separate prong use.
- A10 / weakness 2 (objects vs generic Deep Sets): call-level near-invariance under the scramble is shared by Deep Sets and PFN
  (7.3–8.7 %); the only JEDI-specific signature measured is latent (S moves 0.47 of its spread under scramble vs 0.21 under rotation;
  Deep Sets / PFN 0.50–0.54 vs 0.58–0.62).

A3 v3 (2026-10-16, E5v, verified on a fresh sample): rigid rotations and reflections of whole jets change 3.7–4.1 % of calls. Ten of
the 17 particle objects (S0, S19, S20, S29, S30, S40, S43, S50, S51, S60 — the five costliest among them) move ≤ 0.1 SD; S13, S17,
S63, S1 move 0.1–0.2 SD; S52, S56, S62 and the context-only S31, S58 (built from η- and φ-cut g units) are directional. S29's causal
object is the hinge sum itself (edit slope 0.99); the 3-term name is a correlational proxy (slope 0.83). A2 verified (E5v: 14.61 %,
four objects 13.91 %, random four 0–0.9 %).

A3 v4 (2026-10-17, E14 + E14v; verifier confidence 0.7). How round two builds the near-radial objects: 15 particle-level u units
take half-plane input through five s units (s43: a29, a6, a36, a40, a43 → u1, u29, u56, u60; s60: a6, a43, a46, a3 → u52, u62, u43,
u13, u30; s53: a29, a40, a30 → u51, u63; s32 → u17, u20; s52 → u50, u40). 11 units use one s unit, so their on-contour is the level
set of one weighted sum of 3–5 half-planes; u13, u43, u40, u60 combine 2–3 s units. The contours are polygons in the 3–9 directions
with nonzero weight (all nine only for u40), max/min 1.14–1.47, 3–9 % rms from a circle; random-direction polygons with fitted
weights fit about as well as a circle (2–8 %), so the shape test is weaker than "nine-direction polygon" implied. The context g adds no
per-particle angular term; the half-plane P units reach round two through g for u30 (32 %) and u31, u58 (100 %). Old wording (A3 v2,
2026-10-16, E5): "The radial symmetry is built in round two from round one's directional half-planes."

Narrative order: architecture identity (A1) → objects (A2, A3) → how mass is computed (A4) → neurons and classes (A5, A6)
→ universality (A7) → a larger model (A8).

R2 corrections (2026-10-17, statistics on fresh jets): A3 radial objects reduced to 5 significant under every rotation/reflection (10 only at 180°); A2 four-object set not unique (S58 replaceable by S17/S13/S43); A1 float+masks cost −0.29 [−0.55, −0.03] points; Linformer vs JEDI accuracy not distinguishable with two seeds.

## Paper B: method — turning a model into if-statements, and the statement–object decomposition

**Thesis (v2, 2026-10-16; user: "write about the MARS fit method … turning model into if statements. Or explore it"):**
a trained tagger can be rewritten as if-statements at two levels. (i) MARS-style fitting (jetdistill/mars.py): each neuron
as a sum of hinge terms max(0, Q − t), max(0, t − Q) and their products on jet observables Q, chosen by forward selection,
read through the network's own last layer; validated by stage substitution (same-call rate) and causal tuning curves
(J18, G80/G81). (ii) The exact statement–object identity for sum-pooled ReLU layers: each pooled channel = b·count +
w·(sums over the particles passing the cut, inside its rank window), which needs no fit. The method paper states both,
their fidelity (same call, per-particle attribution, causal responses) and their failure modes (G160: accuracy kept but
rejection lost; G180–G190: the fit cannot get both the jet and its particles right), and compares them with decision-tree
extraction, symbolic regression and probes on the same networks.

v1 (2026-10-16, superseded): decomposition identity + stage substitution only.

Models (user 2026-10-16): JEDI-linear family (20 official models); small native transformers and Linformers trained on the
hls4ml dataset (training approved by the user for this purpose); ParT, ParticleNet, L-GATr on JetClass (pretrained weights
on the volume; L-GATr from other groups' runs, copied read-only).

User idea (2026-10-16), node E11: MARS-fit the entire trigger model, every unit of every stage, on physics observables and
compositionally on the previous stage's formulas; check whether the all-if-statement model reproduces the network's
performance; read off how features evolve over the layers and what each neuron looks at, against physics.

E11 result (2026-10-16; verification E11v running): the nested if-statement program (144 units, 3,824 hinge terms, 6,064
if-statements, 211 physics inputs; each stage fitted on the previous stage's fitted values) reproduces 94.0 % of the
network's calls on 8,848 held-out jets (94.6 % with half-plane sums) at 80.72 % accuracy vs 80.42 %. Fitted on physics inputs,
mass-family terms first appear at h4 and carry 0 → 3 → 3 → 13 → 21 → 28 % of formula weight from P to z (independent of E10).

E11v (verified, 2026-10-17): no leakage; standalone program identical (94.02 %); fresh 17,011 jets 94.23 % same call, accuracy
−0.31 ± 0.17 points; W rejection 0.70×. Under the 0.06 rad shift the nested program reproduces the network (W → Z 77.6 % vs 74.5 %);
a formula fitted directly on physics observables (92.5 % agreement on ordinary jets) does not (0.1 %). This is Paper B's central
result: compositional fits are faithful under mechanism-separating edits, direct surrogates are not.

R2: program W rejection 1.13× [0.96, 1.42] on fresh jets (E11v's 0.70× not reproduced; only 60 background jets pass): rejection not resolved at these counts; accuracy difference −0.04 [−0.35, 0.26].

Open questions for B: does the MARS fit on hand observables match the network on substitution tests as well as the object
formula does? If yes, B is one section of A. Which if-statement form transfers across architectures?
[2026-10-17, E7 + E7v, answer for z at K = 32 terms: on ordinary jets the hand-observable fit (87 observables) changes fewer calls
than the fit on the 19 S objects (7.4–7.5 % vs 9.2–10.4 %). Under the 0.06 rad translation its neuron changes are 0.12–0.16 of the
network's and it sends 1.7–4.3 % of W jets to Z (network 76–77 %); under the 45° rotation its slope is ≤ 0.011. The 87-observable
formulas put 37 % of their weight on the invariant mass. With the mass family removed (77 observables) the hand fit changes 9.05 % of
calls (S 9.23 %, difference −0.17 [−0.71, 0.37]), sends 75.7 % of W to Z, and its translation slope is 0.37 (S 0.57, difference +0.20,
p < 0.001, 16/16 neurons after Holm); under the rotation it does not respond (0.006). So whether a direct surrogate follows a
mechanism-separating edit depends on which correlated observable the fit selects; the same-call rate on ordinary jets does not detect
this, the edit does. This qualifies B3 (E11v's direct surrogate) as basis-dependent. Merge rule: no change yet; B's stand-alone test
(≥ 3 architectures or 2 datasets) is still open.]

Round 3 (2026-10-17), E22/E22v, B4/B4v (verifier confidence 0.55 each):
- B2 split into B2 (calls: 94 %, unchanged) and B2m (variables as causal variables: effective IIA 0.52–0.66 per stage; above a swap-free
  predictor at z only). Old B2 wording ("reproduces the whole network … it also reproduces the network's responses") kept on the
  results page with this date.
- B4 v1 (2026-10-17, B4/B4v): the nested program is not more compact than direct MARS / GBDT (they reach mars2–16's same call at 4.6–19×
  lower DL); at K = 16 and 30 it is within 1 point on same call and leads the strongest boosted-tree model of equal or smaller DL by
  2.0–3.7 points of agreement under the coherent 0.06 rad shift (recall 61–69 % vs 50–52 %), and by 2.7–5.2 under a random 0.06 rad
  displacement that raises Σ pT r² by the same expected amount, so the lead is not specific to the shift. Specific to the shift: direct
  physics-to-logit MARS fits reproduce 18–23 % of its call changes with 0 % W → Z (56–57 % under the random displacement). Rank swap,
  rotation and the a32 crossing do not separate surrogates (network flips 1–4 %). Prior B4 entry: "Baselines (tree extraction,
  symbolic regression, probes)", not started. PySR and energyflow are not in the job env.
- B3: the E22 probe-vs-formula IIA contrast at z is confounded by fit quality (post-ReLU R² 0.36–0.89 probe vs 0.83–0.99 formula; gap
  tracks R² gap, Spearman 0.43); at matched R² (S29, S50) the difference is 0.01–0.02.
- Merge rule: B's stand-alone case after round 3 rests on (i) the 94 % call reproduction, (ii) the direct-MARS shift failure, (iii)
  per-variable IIA above permuted/swap-free baselines at z. Compactness and a mechanism-specific lead over GBDT are not supported.
  The ≥ 3-architecture condition is still open. No merge decision taken (user gate).
- Round 5 (2026-10-19), X9 (verifier 0.7): the E22-style effective IIA of JEDI-linear's S formulas is 0.701 against 0.705 for random
  rotations of the aligned subspace (p_rot 0.44–0.95 for JEDI, PFN and EFN); above permuted alignment (0.251, p 0.005). B2m lowered
  0.4 → 0.35. PFN / EFN latents fitted on the same library: IIA 0.52–0.58, paired difference to JEDI −0.12 to −0.19.
- Round 6 (2026-10-19), X10/X10v (verifier 0.45): with learned subspaces (DAS) instead of the named per-unit formulas, M0 and M2
  reach effective IIA 0.66–0.87 against ≤ 0.14 for random subspaces and above the full-swap baseline (M0 +0.5, M2 +0.05 to +0.17);
  the W−Z margin +0.15 to +0.30 at h4/h5. So X9's null reflects the formula basis, not the absence of causal variables. B2m
  0.35 → 0.4. Method notes for B: (i) a fixed-point network needs a range clip in any interchange (unsigned quantizers wrap);
  (ii) random subspaces do not separate trained from untrained networks, the full-swap (source-call) baseline does for M0/M2;
  (iii) a variable can be "found" for one edit and not for an equivalent one (Δ by pT vs by angles), so each variable needs at least
  two edits that set it. X11/X11v (0.85): a preregistered naming score separates the unit basis from rotations and SAE features for
  JEDI-linear (0.79 vs ≤ 0.63 vs 0.16–0.21); usable as a basis test on other architectures (Deep Sets 0.29, PFN at the rotation level).
- Round 7 (2026-10-20), B6/B6v (verifier confidence 0.7; review weaknesses 3, 6): one Pareto comparison of edit agreement against DL
  (32 bits per number + structure bits) for the blind rank-window MARS program (98 observables, fitted input-to-logit, no unit access),
  CART, HistGBDT, Lasso / ridge, MLP and EFP-linear, on JEDI-linear, Deep Sets, PFN and tf-2b-d32, under five edits. Against dev-tuned
  baselines of ≤ equal bits the program's translation agreement differs by −1.3 to +1.7 points (three samples, JEDI and PFN), +0.2
  (Deep Sets), +1.6 (transformer); oracle-selected baselines are 0–2.4 points higher. B6's +2.8 to +5.0 lead was against one GBDT
  setting (lr 0.2) and depends on requiring the same new call (that GBDT has higher flipped-set Jaccard, 0.72 vs 0.63). Under the
  M2-held translation the program leads in JEDI-linear (+1.7 to +2.0, Holm ≤ 6e-5) and the transformer (+1.7), with every ≤ 14 kbit
  surrogate within ~3 points of never-flip. Larger GBDT / MLP (4–124× bits) are 5–12 points higher. Symbolic regression not run (PySR
  and gplearn not in the job env).
  B4 v2 (2026-10-20, B6/B6v): "at about 13 kbit the blind rank-window program follows the networks' call changes under translation as
  well as tuned trees / GBDT / MLP of equal bits, not better, on four architectures; its only measured lead is on the M2-held
  translation in JEDI-linear and the transformer." B4 v1 (2026-10-17) kept above. B4 0.25 → 0.15; new claim B6t (0.7).
  Merge rule: the ≥ 3-architecture condition for B is now met as "tree-level fidelity per bit on four architectures"; the method
  claim rests on the exact pooled identity (B1), the 94 % call reproduction (B2), naming, and this comparative figure. No merge
  decision taken (user gate).
- Round 4 (2026-10-17), B5/B5v (verifier confidence 0.5): shared-budget MARS programs (T = 5–50 terms over the 5 logits) on three
  pools. Axis-relative pool: same call 83.2 % (T 20) / 89.5 % (T 50); calls changed under the +0.06 Δη shift 51 / 68 % and under the
  WTA re-axis 27 / 33 % (network 56 / 30 %). Invariant pool (m, EFPs, pT sums): same call 88.8 % at T 50 and 0 % calls changed under
  shift and re-axis (by construction). No program of ≤ 50 terms meets the strict reproduction criterion on the shift; mars30 also fails
  it (+3.8 [3.0, 4.6] points). B5v (seed 23): the network's shift response depends on its sign and direction (W → Z 77.9 / 43.3 /
  62.7 % for +Δη / −Δη / random; −Δη vs +Δη calls changed +6.6 points, p 5e-54) and it changes 3.7 % of calls under a 45° rotation;
  every fitted program (mars30 included) gives the same response for every direction and 0–1.8 % under the rotation. The direction
  match of mars30 and P_all T 50 (W → Z within 3 points) holds for +Δη only (mars30 under −Δη +29.8 points). For B3 this is a
  further case where same call on unedited jets does not separate two programs (invariant vs axis pools ≤ 1.2 points apart) and the
  edits do. For thesis v3's compact form: not met at ≤ 50 terms with this fitter.

## Datasets
- hls4ml jets (5 classes g, q, W, Z, t): JEDI-linear family, small transformers / Linformers.
- JetClass (10 classes): ParT, ParticleNet, L-GATr.
- Top Tagging Landscape (Kasieczka et al. 2019; top vs QCD; user 2026-10-16: one signal class, so a decision boundary is one
  object set vs background): candidates for pretrained weights online (ParT/ParticleNet fine-tunes, L-GATr `toptagging`
  config, PELICAN, LorentzNet); the mass / three-prong / W-inside-top story of Paper C has a direct test here. Weights check
  running (scout agent) before any training.
- Quark–Gluon (Komiske, Metodiev & Thaler; 2M Pythia jets, with PID; user 2026-10-16): also one signal class; the
  multiplicity / Casimir-scaling story (Frye et al. 1704.06266) has a direct test here. Same weights check first.
- Status 2026-10-19 (H2): both added. Loaders interp/data_top.py and interp/data_qg.py (sample(per_class, seed, split),
  random rows); converted arrays in /j-jepa-vol/aaron-interp/data/{top,qg} (128 hardest constituents; QG validation =
  last 11.11 % of each train file). Released weights found: LorentzNet top and QG only (models/{top,qg}/lorentznet); no
  ParT/ParticleNet fine-tunes, PELICAN, L-GATr or EFN/PFN weights; no Herwig QG sample. Own trainings with the authors'
  code (ext/particle_transformer 2925bdb, weaver-core 98ca934; energyflow): ParT and ParticleNet on Top (kin) and QG
  (kin, kinpid), EFN and PFN on both, running. Test sets: top 403,999 jets, QG 199,998.
- Status 2026-10-19 (round D2, H4/H5): Top: ParT_kin and PN_kin finished (best epochs 16 / 19; AUC 0.9860 / 0.9862 on a 10k
  sample, R30 1493 vs published ParT 1602); PFN / EFN top finished (AUC 0.9806 / 0.9756); LorentzNet released input path
  (two beam particles) reproduces AUC 0.9871 (released scores 0.9870). QG: ParT kin / kinpid finished (best epochs 14 / 17;
  kinpid full test 0.9121 = published); PN kin / kinpid still training (d-h2-qg-pnet, d-h2-qg-pnet-kin); d-h2-pfn failed,
  so QG PFN / EFN are torch stand-ins trained inside H5 (400k jets); d-h2-efn still running. Herwig 7.1 QG (Zenodo 3066475,
  energyflow 'herwig') downloaded to /j-jepa-vol/aaron-interp/data/qg/herwig: AUC Pythia 0.90–0.92 → Herwig 0.80–0.81.
- Status 2026-10-19 (round D3, H12–H14): QG PN kin / kinpid final checkpoints done (d-h2-qg-pnet, -kin); QG EFN (d-h2-efn)
  done, ported to torch for H14 (test acc 0.809, AUC 0.883). Herwig 7.1 QG split 80/10/10 by file (files 18–19 = test);
  Herwig-trained ParT_kin / PN_kin running (d-h14-hw-part3, d-h14-hw-pnet3, 48 GiB; first launch OOMKilled at 40 GiB);
  QG PFN rerun d-h14-pfn2 running (H2 PFN OOMKilled). New hls4ml models: JEDI-style n64 retrains with pT-normalised inputs,
  pT-flattened weights and raw inputs, 2 seeds each (H12; models/small/jedis-d64-n64-h12{ptnorm,ptflat,raw}/seed{0,1};
  test 76.2–79.0 %). Top Landscape training pT window 550–650 GeV (sample IQR 570–619), hls4ml IQR 993–1061 GeV.
- Status 2026-10-20 (round D5, H24): no new dataset. Herwig-trained QG ParT_kin (d-h14-hw-part3, best epoch 12; Herwig test
  AUC 0.836, acc 0.760) and PN_kin (d-h14-hw-pnet3, best epoch 19; 0.838 / 0.762) finished; on Pythia jets they give
  0.888 / 0.884 (Pythia-trained 0.904 / 0.905 on Pythia, 0.815 / 0.817 on Herwig). QG PFN rerun d-h14-pfn2 done (AUC 0.896).
  One training run per generator; no seed replicate yet (H26).

Cross-model mass row (2026-10-16, P1; verification P1v running): ParT_full and ParT_kin are not translation invariant either.
A rigid φ shift by 0.06 rad (invariant mass unchanged; only the per-particle dphi and deltaR inputs change) moves true-W → Z
from 12 to 31 % (full) and 16 to 37 % (kin); the class-token read-out mass rises by 0.7–0.8 (full) and 1.1–1.4 (kin) of the
½Δlog(M0·M2) prediction, against 0.78–0.84 for JEDI. On data the two mass definitions coincide (|M1|²/(M0·M2) median 2e-3 on
JetClass, 3.5e-4 on hls4ml) because the stored axis is the pT centroid; the edit identifies which formula each network computes.
Physical consequence under test (P1v): the taggers' dependence on the axis convention (winner-take-all vs centroid axis).
C3 v1 ("ParT's mass is the invariant mass, unlike JEDI's") refuted by P1.
P1v (verified, confidence 0.9): ParT_full's response enters only through the per-particle deltaR input (fed unedited: 92–100 %
of the response disappears); ParT_kin's through dphi (0.65) and deltaR (0.35); the pair features are invariant to float32 noise.
P2 (2026-10-17; statistics pending P2v): under real mass changes the per-particle axis-relative route carries 93–101 % of ParT's
mass read-out change, the pair route 1–6 % (and 10–21 % of the W → Z change for mass increases).
Physical consequence (Paper D, D4): inputs about the winner-take-all axis change 11.7 % (ParT_full) and 34.1 % (ParT_kin) of calls.
D4 note (2026-10-17, X7/X7v, verifier confidence 0.4): JEDI-linear evaluated about the hardest particle changes 30.98 [30.39, 31.63] % /
31.66 [30.89, 32.35] % of calls on two fresh samples (W → Z 865–2230 vs Z → W 1–45). Against covariate shift: a rotation of equal mean
displacement changes 0.9–3.3 % (not adequate: the models' objects are radial); a translation of equal displacement with Δ = ½Δlog(M0M2)
held at 0 changes 9.8–18.0 % (JEDI 11.4 %) with no W → Z flow; a random-direction translation of the same length (same Δ) changes the same
share as the edit (35.4 vs 31.7 %). So the W/Z direction and about 2/3 of the call changes of stored-trained models require the change
of Δ. E23v: at natural centroid offsets (6–15 mrad) the call changes follow translation size; a W → Z excess appears for random shifts of
5.8–13 mrad (p90–p99 of the natural offset; 76 vs 37, 175 vs 22). D4 wording kept; its decision-level statement is scoped to axis changes
of this size or larger (WTA, hardest particle).

## Paper C: circuits (ParT)

**Thesis (v1, 2026-10-16):** ParT computes jet mass through geometric routing, as three linked steps:
1. **Geometric routing:** the pair-bias formula fixes the head roles.
2. **Threshold head:** head b2h8 switches near 85 GeV.
3. **Readout:** far-side heads write the mass into the W/Z neuron n98.

Higher-point correlators may need more depth. Existing evidence: G21b/c, G22, G101, G107, G108, G121–G123, G131, G157.

**Thesis (v2, 2026-10-19, loop C round C1, nodes P3/P3v, P4/P4v, P5; v1 above kept):** ParT computes its W/Z mass sense
from per-particle inputs measured from the jet axis (C3), as ParticleNet does (C5), in three steps:
1. **Geometric routing:** the pair-bias formula fixes the head roles (C1, unchanged).
2. **Threshold head:** head b2h8 switches its class-token share at 87 GeV, 4 GeV wide; no other class-attention head has a
   mass switch (C2a). It also moves under a φ shift that keeps the invariant mass fixed.
3. **Read-out:** the mass change reaches the W/Z coordinate n98 through the first class-attention block (98–99 % of the
   response; head b1h3 43–53 %), mostly as changed particle embeddings (value change 0.68, attention re-weighting 0.34);
   b2h8 carries 3–5 % (C2b). The literal circuit {b2h8, b1h3, n98} kept alone gives 50–54 % W/Z (model 83 %).

**Thesis log (round C1, 2026-10-19).**
- Step 2 (v1 "b2h8 switches near 85 GeV"): reproduced at full statistics with the corrected build (P3, P3v): m50 87.1 / 86.6 GeV.
- Step 3 (v1 "far-side heads write the mass into n98"): not supported as stated. P3v: patching b2h8 removes 0.046 of n98's
  response to angles × 1.1, block-1 heads 0.987. Particle-block heads 4/8 (the "far-side" heads) remove 0.26 vs a stratified
  random 0.13 on W jets only (not on Z, not under the φ shift), mostly head 4. Rewritten as above.
- C2 (v1 "the mass enters through the attention; head b2h8 is a threshold near 85 GeV", 2026-10-16, G21–G22): split into C2a
  (switch, 0.85) and C2b (route through block-1 heads, 0.7); the G21b attention-dominated decomposition (−0.458 of −0.478)
  came from the formula fit and is not reproduced in ParT itself (attention 0.34). C2 v1 confidence 0.7 → 0.2.
- C3 wording kept; P4 reproduces ParT's numbers on the same rows. New C5 (ParticleNet, 0.7). "Input format causes it" is not
  claimed: P4v notes that a shared learned solution predicts the same result; C6 (L-GATr, P5 running, P6 pre-registered) tests it.
- Circuit definition: P3v shows the strict literal definition (resample all other class-stage units and the other 127
  class-token coordinates) caps even all 16 heads + n98 at 63 % W/Z, so literal-circuit scores are reported with that ceiling.

**Thesis (v3, 2026-10-19, loop C round C2, nodes P8/P8v, P9/P9v, P10; v1 and v2 above kept):** ParT computes its W/Z mass
sense from per-particle inputs measured from the jet axis (C3), as ParticleNet does (C5), in three steps:
1. **Mass-evidence direction from the embedding onward (C7):** one direction of the jet-averaged particle representation,
   readable from the input embedding on, follows the axis-moment mass √(ΣpT · ΣpT ΔR²) (r 0.87 between donor and recipient
   differences) and carries 0.6–0.7 of the φ-shift response. No particle block forms it: each block adds 0.03–0.10, and
   0.58 of the response reaches the class stage through the residual path of the edited embedding.
2. **Read-out (C2b):** the first class-attention block carries the response into n98 (0.99 under the mass-fixed φ shift;
   b1h3 0.53, b1h6 0.15). b2h8 switches at 87 GeV (C2a) but carries 3–5 %.
3. **Circuit size (C8):** no circuit found with up to 10 % of the edges reproduces the W/Z calls under opposite-class resample
   ablation (≤ 63 % vs 83 %); circuits are reported as accuracy and faithfulness against size.
Pair-bias head roles (C1) are kept as earlier evidence.

**Thesis log (round C2, 2026-10-19).**
- Planner step 3 of v2 ("ParT turns ΔR into an M2-like quantity at an identifiable block 1–3, read by n98 through block-1 heads",
  P8 plan): the "identifiable block" part is not supported (P8: no sublayer > 0.084, single shares within the donor-resample
  control per P8v; restoration accumulates 0.03–0.10 per block). P8's further reading "no linear jet-level axis-moment variable"
  is refuted by P8v: P8's donors held √(S0·S2) fixed; with donors that differ in it, the direction follows it (r 0.87) and
  substitution gives 0.42 [0.38, 0.46] of the edit-calibrated slope. New C7 (0.5). Not an exact copy: pT weighted 0.71 of φ at b1.
- C2b wording kept, extended: the φ path through block-1 class heads is 0.991 (P8), as under real mass changes (P3v 0.987).
  Confidence 0.7 → 0.75.
- Localized-circuit expectation (P9 plan: F > 0.8 at < 5 % of edges): excluded (F 0.51 [0.506, 0.520] at 5 %). P9's reading
  "F ≥ 0.8 needs over a third of the network" is not supported (P9v: 29 % by interpolation; a trained mask reaches F 0.98 at
  10 %), but the trained mask's W/Z accuracy is 63 % (model 83 %). New C8 (0.45), worded on accuracy.
- C3 wording kept (0.85). New C3s (seed and data-scale robustness, interim 0.3): P10 at epoch 3 of 50, full_s0 R 0.94 (released
  0.76), kin seeds R 0.17–0.35 (released 0.69); S1–S4 pre-registered for the final checkpoints.
- P7 (proposed in round C1) absorbed: its path question by P8, its circuit question by P9.

**Thesis (v4, 2026-10-20, loop C round C3, nodes P13/P13v, P14, P15/P15v; v1–v3 above kept):** ParT computes its W/Z mass
sense from per-particle inputs measured from the jet axis (C3; reproduced on fresh rows by the P14 control, R 0.79 / 0.72),
as ParticleNet does (C5, 0.75), in three steps:
1. **Mass-evidence direction from the embedding onward (C7, 0.45):** as in v3, with two open checks added: the natural
   m_A slope of the read-out explains about 1/7 of the φ-shift logit change (P13v), and one-direction matches of similar
   size appear in weight-shuffled ParticleNet copies (P15v); ParT's direction has had an untrained control but not a
   weight-shuffled or mirror-sign control (P17).
2. **Read-out (C2b):** unchanged from v3.
3. **Circuit size (C8 v2, 0.55):** no small circuit found with heads / MLP groups (P9) or with attribution-ranked learned
   sparse features (P13: 50 features 51 % W/Z, 75 % needs 1,000–2,000 features with attention live; ≤ 43 % when attention
   and skip maps are also resampled; model 83 %).
ParticleNet internals (C9, 0.45): the φ response is sign-symmetric; a sign-symmetric one-direction signal exists after
EdgeConv-2 but is not an m_A copy and is matched in untrained networks; graphs carry ≤ 4 %; small pooled-channel circuits
match W/Z but not 10-class accuracy. Whether the axis response is a learned choice or an input/architecture prior is
pre-registered as C10 (P14 trained, analysis P16).

**Thesis log (round C3, 2026-10-20).**
- P13 thesis ("C8 reverses with learned sparse features: ≤ 50 features reach ≥ 75 % W/Z; steering the m_A feature gives
  ≥ 0.6 of the φ flow; features more monosemantic than rotated-basis controls"): all three predictions excluded (50 features
  50–51 %; steering ratio −0.001 [−0.005, +0.003]; features below rotated coordinates on the conditional-mean R² score).
  C8 v1 wording kept above, extended to C8 v2 (features); 0.45 → 0.55. P13v: P13's steering criterion could not be met
  under its own calibration (whole read-out moves 14 % of the φ change per natural m_A step); only ablation was a valid
  single-feature test.
- C7 wording kept; 0.5 → 0.45 for the two open checks above.
- P15 thesis ("one m_A direction from EdgeConv-1 on, as ParT's C7; graphs < 10 %; literal circuits cap well below the
  model; probes predict calls but not the φ response"): EdgeConv-1 direction refuted as stated by the mirror-sign test
  (P15v: −0.29 of the − response); EdgeConv-2 direction sign-symmetric (0.73–0.80) but not an m_A copy and matched by
  shuffled networks; graphs ≤ 4 % (supported); circuit cap holds for 10-class accuracy, not W/Z accuracy; probe
  prediction as expected (EFP / Lund respond 0 by construction). New C9 (0.45). C5 0.7 → 0.75.
- P14: training of the planted arms finished; only the released-model control is analysed (supports C3). New C10
  pre-registered (0.0). Arm c (pair route only) learns the invariant-mass window to 97 %.
- P10: all six seeds finished 50 epochs (wrapper error after training; checkpoints complete); C3s stays 0.3 until P11.

**Thesis (v5, 2026-10-20, loop C round C4, nodes P11/P11v, P16/P16v, P18/P18v; v1–v4 above kept):** ParT computes its W/Z
mass sense from per-particle inputs measured from the jet axis (C3, 0.9). This is a property of training on JetClass, not
of the architecture: six ParTs retrained from scratch on 2M jets keep it at the call level (C3s v2, 0.7), while ParTs with
the same inputs trained on a mass-window label respond to the mass-fixed φ shift at R 0.18–0.22 (released 0.72–0.79) and
compute the mass mostly from pair features (C10 v2, 0.55). Inside the released checkpoints:
1. **Stage order (C4 v2, 0.55):** count and pT are readable at the embedding and the mass from block 1, at levels an
   untrained ParT also reaches; prong shape (tau21, tau32) forms over blocks 1–4 and is the training-dependent stage. No
   mass-last stage as in JEDI-linear (A8).
2. **Mass-evidence direction from the embedding onward (C7, 0.45):** unchanged from v4; P18v adds that untrained ParTs
   reach 1-d DAS IIA 0.45–0.95, so the direction's training evidence must come from a difference against untrained nets (P17).
3. **Read-out (C2a, C2b, scoped to the released checkpoints):** block-1 class heads carry 0.97–0.98 of the W/Z unit's
   mass response and b2h8 switches at 87 GeV. Retrained seeds split the response between the two class blocks (block 1
   0.63–0.84, inside the untrained range) and have switch heads at other indices and masses.
4. **Circuit size (C8 v2, 0.55):** unchanged from v4.
ParticleNet (C5 0.75, C9 0.45) unchanged.

**Thesis log (round C4, 2026-10-20).**
- C3 wording kept; 0.85 → 0.9 (P11/P11v and P16/P16v reproduce R 0.79 / 0.72 on two more fresh samples; second read-out
  W-or-Z vs QCD gives 0.65 / 0.63).
- C3s v1 ("axis response holds across retrained seeds and at L-GATr's 2M scale, pre-registered S1–S4", 0.3) kept above;
  reworded v2 to the call level (0.7). Pre-registered outcome: S1 3/6 (not a training test: untrained full 0.93), S2
  structural, S3 (block-1 share > 0.9) fails 6/6, S4 holds. P11's "no seed has a head as dominant as b2h8" and "3 of 6
  switches" corrected by P11v (4 of 6; kin s2 c2h3 as large as b2h8).
- C2a, C2b wording kept with the scope "released checkpoint" added (confidence unchanged): the seeds' block-1 shares lie
  inside the untrained range; only the released 0.97–0.98 exceeds it.
- C10 v1 pre-registration (R_a < 0.2 and R_b ≈ 1 for a learned choice; R ≥ 0.5 for both under an architecture prior) kept
  above; outcome R_a 0.185, R_b 0.215: architecture prior excluded, R_b ≈ 1 not observed. P16's sentence "H_data strict
  excluded" withdrawn (P16v: labels a and b agree on 97.7 %, shared seed and data order). Reworded v2 (0.55).
- C4 v1 ("per-block readability follows algebraic depth", 0.0) kept above; P18's H1 (count → prong → mass last with a
  causal m_A step) not observed; reworded v2 (0.55). Raw R² is matched by an untrained ParT for count, pT and mass
  (P18v), so C4 rests on calibrated gains and depth profiles relative to untrained nets.
- Method: untrained-network DAS is now a required control for one-direction claims (P18v: untrained IIA 0.45–0.95).
- Proposed: P19 (planted ParT on the true class labels, decorrelated m_A label, second seed), P20 (seed vs data scale for
  the block route). Still running: anrunw-ai-c-p5-lgatr2 (L-GATr, P6 after it ends).

## Paper D: physics journal

**Thesis (v1, 2026-10-16):** physics-motivated input edits on JEDI-linear and ParT, compared with classical taggers at
their published working points, show which physical quantities each model uses and which it does not. Existing
evidence: J25–J33, G192–G206.

**Thesis log (round D1, 2026-10-19, nodes H1–H3).** v1 wording kept. Scope extended: the edits now cover nine models on two
datasets (H1), with one-signal datasets (Top Landscape, QG) set up and models in training (H2).
- D1 (v1 wording kept: "JEDI: mass windows plus a ΣpT shortcut; no W-mass constraint inside top"). H3/H3v: reproduced at 10×
  statistics; extended: against a control that holds the W pair at its mass, neither JEDI-linear nor ParT_full nor ParT_kin
  responds to the W-pair mass inside top (Holm p ≥ 0.13); ParT's top call drops under any rearrangement of the subjet-pair
  masses at fixed jet mass (up to −16 points at 120 GeV). H3's proposed refinement "ParT_full depends a little on the W-pair
  mass (±2.7 points)" was refuted by H3v and is not adopted.
- D2 (v1 wording kept: "ParT: displaced-track multiplicity, not lifetime; isolation used; π⁰, dead cone, lepton–b charge link
  not used"). H3v sharpens the first clause: ParT_full's H→bb call follows the number of tracks with impact-parameter
  significance S ≳ 2–3 (capping every S at 3.5 keeps the count and costs 3.1 points; keeping 2 displaced tracks costs 61). The
  old "lifetime × 0.35" evidence did not hold the count fixed (S > 2 count −26 %). Small H3 effects not re-verified: lepton
  charge in H→ℓνqq′ (−2.6 points), H→cc dead cone (−2.1).
- D3 (was "not started"): done by H3 (verifier confidence 0.45).
- D5 (new, H1/H1v, 0.55): trained taggers (JEDI-linear, Deep Sets, Linformer, transformer, ParT, ParticleNet) are not IRC safe:
  an exact collinear split changes 1–5 % of their calls and none of an IRC-safe EFN's; in-distribution edits (more splits,
  removing the softest particle, count restore) move calls along the particle count. The soft-particle (IR) probe is below
  the data pT floor and position dependent; for ParT_kin / PN_kin it grows as the particle gets softer.
- D6, D7 (new, setup in H2, 0.0): top W-pair mass use; QG constituent counting (Casimir ratio measured 1.59).

**Thesis log (round D2, 2026-10-19, nodes H4, H5, H8).** v1 thesis wording kept; scope now three datasets (hls4ml,
JetClass, Top Landscape) plus quark–gluon, with LorentzNet, PFN and EFN added to the model set.
- D1 (wording kept). Scope stated: the "no W-mass constraint inside top" result holds for JEDI-linear (and ParT, H3v) on
  hls4ml / JetClass. It does not extend to taggers trained on Top Landscape (H4/H4v, D6).
- D6 (setup wording, H2: "top taggers do / do not use the W-pair mass inside top"). New wording (H4/H4v, 0.7): ParT,
  ParticleNet and LorentzNet trained on Top Landscape use the W-pair mass inside top beyond total mass and τ32: at fixed m and
  τ32 they respond 2.1–4.1× a side-effect classifier without the pair mass and as much as the same classifier with it, on two
  samples. H4's thesis note also listed PFN and EFN and called the EFN "not a zero control"; H4v found both at the
  side-effect level, so that part is not adopted (open, H9).
- D7 (setup wording, H2: "quark–gluon taggers count constituents"; H5 pre-registration: "Δlogit per ΔN equals the
  multiplicity-LR prediction 0.46 at fixed IRC-safe shapes, EFN zero"). Not supported except the EFN zero. New wording
  (H5/H5v, 0.7): QG-trained ParT, ParticleNet and LorentzNet do not count constituents (on-manifold added soft particle
  −0.024 to −0.004 vs LR(N) +0.097); their collinear-split response follows the hard-particle energy sharing (ΔN = 0
  control equal). H5's suggested attribution to pT^D is not adopted (H5v: not isolated). Casimir-meets-Poisson
  overpredicts the data LR(N) slope 4.8× because measured multiplicity is about twice as wide as Poisson.
- D5 (wording kept, scope stated): the IRC-unsafe half extends to QG models (H5); the multiplicity half is measured on
  hls4ml / JetClass only, and QG-trained models do not show it.
- D8 (new, H8/H8v, 0.5): an EFP/ADO-chosen IRC-safe surrogate matches the taggers' rankings on natural jets but not their
  edit responses; for JEDI-linear the axis response is reproduced by the axis-moment mass m_A (A4), so the axis gap is mostly
  one known variable. H8's thesis note "the edit responses go beyond what the IRC-safe shape literature describes" is
  narrowed by H8v to: the collinear-split and pT-scaling responses and ParT's Z/W ranking are not described by the pool.
- D4m 0.45 → 0.5 (H8v m_A surrogate, JEDI-linear).

**Thesis log (round D3, 2026-10-19, nodes H12, H13, H14).** v1 thesis wording kept. Two new physics edits (an exact boost
along the jet axis; pileup-like diffuse soft contamination) applied to 20+ models on hls4ml, JetClass, Top Landscape and QG,
and the QG edits repeated on Herwig 7.1 jets.
- D9 (new, H12/H12v, 0.55): hls4ml-trained taggers are not invariant under a boost along the jet axis that fixes m and m_A
  (21–38 % of calls at pT × 0.9 vs 0.3–4.2 % for a matched rotation); JetClass and Top ParT / ParticleNet (and Top
  LorentzNet) change 1.5–4× fewer calls under it than under pT-only or angle-only scaling. H12's proposed explanation
  ("mass read through opening angles at the training pT; wide-pT or four-vector training gives an invariant-mass reading")
  not adopted (H12v: Top models were also trained in a narrow pT window on equivalent inputs; cause untested, H16). H12's
  "Lowering pT moves JEDI calls to the heavier class, opposite of a mass reading" withdrawn (H12v: the flow is W/Z → t, the
  ΣpT shortcut; at the W/Z boundary Δ_B = Δ_P + Δ_A). H12's "supports D4m/A4" not adopted (B fixes m_A too).
- D1 (v1 wording kept). Refined: the D1/H3 "boost × 0.9" (W 87 → 27 %) was a first-order mass-preserving boost, not a
  pT-only edit; an exact pT-only scaling gives W 87.3 → 47.9 %. Weakness 7 (ΣpT shortcut as a sampling artefact): the
  flat-pT retrain lowers the pT-only response (9.6 → 5.8 %) and keeps a 21 % boost response; partly addressed.
- D10 (new, H13/H13v, 0.4): under diffuse soft contamination no network acts like a groomed observable. H13's proposed
  wording ("JEDI-linear acts like the ungroomed m window, JetClass/Top networks between ungroomed and groomed; the response
  comes from the added soft momentum and mass", 0.6) refuted except for JEDI-linear: H13v content controls show that
  JetClass ParT/PN_full and Top ParT/LorentzNet/PFN respond mainly to the number of added soft particles.
- D4 (wording kept, scope): pileup-sized axis offsets (2.5–13 mrad) change 0–3.3 % of calls; the decision-level statement
  applies to WTA / hardest-particle-sized offsets (40–95 mrad), as already scoped after E23v.
- D5 0.55 → 0.6: the multiplicity half gains an on-manifold-like count control on JetClass (ParT/PN_full) and extends to
  Top-trained ParT/LorentzNet/PFN (H13v c4); the QG exception (D7) stands.
- D7 (wording kept, 0.7, H14 not yet verified): the no-counting result holds on Herwig 7.1 test jets; counting plus
  IRC-safe shapes (EFP+N GBDT) excluded by the ΔN = 0 control; the leading-particle fraction z1 at fixed pT^D excluded as
  the hard-core function; pT^D not separated from angular energy flow. H10 absorbed into H14. Herwig-trained models running.
- D4m (0.5, =): no change from H12/H13 (neither isolates the axis-convention response).

**Thesis log (round D4, 2026-10-20, nodes H16, H18, H19).** v1 thesis wording kept. The three open questions of round D3
(a: boost sensitivity, dataset or architecture; b: soft-count response, edit or model; c: W-pair mass, JetClass vs Top)
were tested with matched edits across datasets.
- D7 (wording of 2026-10-19 kept: "QG-trained ParT, ParticleNet, LorentzNet do not count constituents; collinear unsafe
  through the hard-particle energy sharing"). Scope stated (H18/H18v): "do not count" means count at fixed IRC-safe content
  (exact collinear split of soft particles: −0.006 to −0.020 per particle vs LR(N) −0.068). Soft particles that add energy
  flow move them about as much as an IRC-safe EFN. 0.7 → 0.75.
- D10 (wording kept, 0.4): the count response is model/dataset specific (JetClass W-vs-QCD nets 1.2–2.3× LR(N) under exact
  splits), not an effect of the diffuse edit. H18's proposed wording ("D10's count response is not caused by the diffuse
  edit; the e1 contrast excludes an edit difference") is adopted only for the exact-split and c4 − c4m evidence; the e1 part
  is not (H18v: added pT differs 0.13 vs 0.99 GeV). Top LorentzNet's c4 response is about 40 % added four-momentum.
  New observation, unexplained: top taggers (JetClass, Top Landscape) move toward QCD under added particles while LR(N)
  moves toward top.
- D5 0.6 → 0.65 (exact-split count response of JetClass W-vs-QCD nets; QG exception stands).
- D6 (wording kept) 0.7 → 0.75: the Top-trained W-pair response holds on JetClass tops (H19, sample-independent).
- D11 (new, H19v, 0.45): JetClass-trained ParT and ParticleNet also use the W-pair mass when edited jets are rebuilt about
  their own axis. H19's proposed reading ("JetClass nets respond only at 100 GeV; the contrast depends on edit direction",
  confidence 0.15) refuted by H19v (axis-build artefact) and not adopted. D1's ParT extension (H3v null) is contradicted
  pending a rerun of H3v with the same rebuild (H20); D1's JEDI-linear statement unchanged.
- D9 (wording kept, 0.55): scope added: H12's five-class 21–38 % is 67–77 % W/Z jets; on t/q/g jets the top call changes on
  3.6–10.7 % of jets.
- D12 (new, H16/H16v, 0.55): in small taggers boost sensitivity comes from architecture, inputs and task, not the training
  sample; with absolute-pT inputs the sample sets the direction. H16's "every net reads moments about the axis rather than
  the invariant mass" not adopted (H16v TC control).
- D4m 0.5 → 0.45: H16v TC (m_A fixed, centroid moved) changes 4.4–13.8 % of small-net calls vs 1.3–2.8 % for a rotation;
  H19v: JetClass ParT/PN respond to an axis shift alone. D4 (0.8, =) gains an instance (H19's fixed-axis build).

**Thesis log (round D5, 2026-10-20, nodes H23, H24, H25).** v1 thesis wording kept. The earlier W-pair and soft-contamination
edits were audited with every edited jet rebuilt about its own axis (H23, absorbs H20); the Herwig-trained QG models were
evaluated (H24, absorbs H15); the prong-shape variable τ32 was tested at fixed m and mW (H25).
- D1 (v1 wording kept: "JEDI: mass windows plus a ΣpT shortcut; no W-mass constraint inside top"; JEDI-linear part
  unchanged, 0.85). The ParT/JetClass extension added in round D1 from H3v ("neither JEDI-linear nor ParT_full nor ParT_kin
  responds to the W-pair mass inside top beyond a W-fixed control", 2026-10-19) is withdrawn (2026-10-20, H23/H23v): with
  each edited jet rebuilt about its own axis, H3v's own k-means edit gives ParT_full 1.93× [1.51, 2.39] MLPx at 60 GeV
  (Holm 1.9e-5 on fresh rows; = 1.07× MLPxW), ParT_kin and PN_kin 2.7–3.0×, and at 100 GeV ParT_full −0.42 [−0.63, −0.22]
  beyond MLPxW. The H3v null came mostly from the fixed-axis build. H23's own wording "at 100 GeV no model responds beyond
  MLPx" and "ParT_full not separated from MLPx after Holm" were not reproduced by H23v and are not adopted.
- D11 (wording of 2026-10-20 kept: "JetClass-trained ParT and ParticleNet ... use the W-candidate pair mass ... at the level
  of Top-Landscape-trained models, when each edited jet's inputs are computed about its own axis"; its quantitative clause
  "3–6× a classifier that sees only the edit's side effects" refined): rc T60 is 2.9–4.5× MLPx and 0.8–1.25× MLPxW (a
  reference given the pair mass) on two independent samples; T100 4.3–6.3× MLPx (1.4–2.1× MLPxW); the response is ordered in
  |mW − 80| (60 > 70, 100 > 90 GeV, Holm ≤ 4e-8 for all four nets). 0.45 → 0.6. H23's "for ParT the translation response
  equals the fixed-minus-recentred difference within ±0.15" holds for ParT_kin only (H23v: ParT_full residual +0.48, PN
  +0.7 to +1.2); the source of the full models' extra non-W-control response is not identified (feature zeroing is
  off-distribution; H28).
- D6 (wording kept, 0.75): Top-trained networks on Top data are unchanged by the audit (their original builds already
  recomputed the axis; rc Top ParT T60 −1.73 vs H4v −1.79; not re-run by the verifier).
- D13 (new, H25/H25v, 0.5): top taggers (Top ParT, PN, LorentzNet, PFN, EFN; JetClass ParT_kin, PN_kin) use the radiation
  pattern inside the three prongs beyond m and mW, but do not follow τ32 as a single number. H25's proposed reading ("these
  taggers use prong-shape information beyond m and mW, comparable to a τ32 classifier", thesis note "top taggers use prong
  shape (τ32 / subjet width)") is adopted only for the S edit's existence of a response; the τ32 attribution is not
  supported (H25v profile edit at fixed subjet masses: 26 of 46 significant cells with the τ32 sign, 20 opposite). The
  pre-registered edit (subjet four-momenta kept) is infeasible.
- D7 (wording of 2026-10-19 kept; scope of 2026-10-20 round D4 kept) 0.75 → 0.8: the no-counting result holds for ParT_kin
  and PN_kin trained on Herwig 7.1 as well as Pythia (H24/H24v; split − ΔN = 0 control −0.008 to +0.009 per particle vs
  LR(N) +0.08 to +0.105 on both test generators); the Pythia-vs-Herwig gluon output gap is not carried by multiplicity
  (N-swap reproduces ≤ 6 % of it vs 65–69 % for GBDT(EFP+N)). H1 (training generator sets the strategy, sign following
  LR(N)) excluded. Refinement: Herwig training shifts the count response by −0.01 to −0.03 per particle (beyond the
  checkpoint spread; seed spread unmeasured, H26); removing the softest particles at fixed N moves all four nets toward quark
  (−0.045 to −0.056 per particle), so the soft particles' momenta and positions are used. H24's "at most a tenth of LR(N)"
  holds toward gluon only (Herwig PN moves toward quark by up to a third of LR(N)'s size).
- D4m (0.45, =): instance added, not verified: Top ParT/PN fed edited jets on a fixed axis respond to a pure translation
  (dTN −1.1 to −2.9 log-odds), as the JetClass nets do (H23).
- D10 (0.4, =): H13's pu-arm call changes differ ≤ 1.5 pp between fixed-axis and recentred builds (H23; not re-run by the
  verifier).
- Methods rule (H23): any edit that moves the E-scheme axis is rebuilt about the new axis before its effect is attributed to
  a physical quantity; results computed on a fixed axis are reported as such.
