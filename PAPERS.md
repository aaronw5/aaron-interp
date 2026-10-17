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
has a substitution test and a causal edit. ParT/ParticleNet then show the same stages in a different arithmetic.

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
| "Objects is a relabelling of sum pooling / EFN filter plots" | rank-window masks and cut ∧ window membership are not in EFN; causal edits that move one particle across a cut change the call as the identity predicts; untrained control |
| "Probes show readability, not use" | every probe claim paired with an intervention (translation, iso-mass, freeze S0/g) |
| "One dataset" | hls4ml + JetClass; top tagging / QG when weights exist |
| "Off-manifold edits" | on-manifold check: |M1|²/(M0M2) median 3.5e-4, so the two mass definitions coincide on data; the edit isolates which one is computed |
| "Prior work: Rai & Ganguly, Wang et al." | they trace a top-tagging circuit / attention sparsity; here: stage-wise formulas, exact identity, cross-architecture comparison |
| "Quantisation specifics" | float + masks reproduces 95.5 % of calls; results reported on the float program with the fixed-point network as the reference |

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
- A3 (v2, 2026-10-16, E5; pending verification). 17 of the 19 second-round objects are each a hinge in two variables
  (pT or ln pT, and r = ΔR to the axis) summed over a fixed pT-rank window; azimuth adds ≤ 0.017 R². Named: S29 ≈ ΣpT·r² +
  Σr² + soft count (R² 0.976; cost 40.9 %), S1 = Σ_{ranks ≥ 1}(r − 0.2)₊, S0 = ΣpT of the 14 hardest, S50 = ΣpT·r² over ranks
  0–38, S19 = N(pT > 1 GeV) among ranks 13–63, S30 = Σ r for r < 0.165. Radial edits on S29/S1 match the hinge prediction
  (slope 1.03–1.04). The radial symmetry is built in round two from round one's directional half-planes. v1: "nameable
  energy-flow quantities (counts and pT-weighted sums over regions)".
- A4 (v3, 2026-10-16, E6 + E6v, verified on a fresh sample; verifier confidence 0.55 for the exact form, > 0.9 for
  non-recentring). The network's mass variable is an un-recentred M2-like quantity: M0 = ΣpT and M2 = Σ pT r² are read
  separately at S and multiplied in h4/h5 (S0 carries M0 through the context g). A rigid translation of all particles by
  δ (invariant mass unchanged to 1e-7) raises the read-out mass as δ² with 0.74–0.84 of the ½Δlog M2 prediction at every
  δ ≥ 0.02 and flips 77 % of W calls to Z; the alternatives Σ pT r and Σ r² are excluded by magnitude (×1.7, ×2.1) and
  exponent. It is not an exact function of r² about the axis: a direction-dependent odd term of 12–20 % of the even term
  remains, consistent with its construction from nine asymmetric half-plane cuts (the only angular inputs); the pT·r²
  products form piecewise-linearly in l2/l3 (the response travels a → s → u → S, not through P/g). v2: "√(M0·M2) about
  the given axis" (exact form overstated). v1 read "m² ≈ 2·ΣpT·Σ pT r² − 2|Σ pT r⃗|²": the factor 2 was an error.
- A5. The 16 neurons are statements on A3's objects. Replacing each stage by its formula keeps the network's calls. [E7]
- A6. Classes are combinations of neuron objects, and imposing a combination causes the class (J33b on the network's own objects). [E8]
- A7 (scope v3, user 2026-10-16: "just use the best version of JEDI"): one JEDI model, 3-feature n64 (the model of all
  earlier J-results), with the untrained n64 network as the control. Cross-model generality comes from the other
  architectures (A10), not from JEDI variants. [E9 → control only]
- A9 (user, 2026-10-16). High-level variables are built over the layers: sums of particles first (P, S), then products
  and ratios (h4, h5), then clustering-type variables, if at all. The layer where a variable first becomes linearly readable
  rises with its algebraic depth. [E10; E6 for mass]
- A8. ParT follows the same scheme: pair-bias statements, head routing, messages, class-token neurons, and combinations. [P-series]
- A11 (user, 2026-10-16: "similar to the Goodfire post, look at neurons too"). Most live units have one main job that can be
  stated as a physical pattern (jet images + observables), with weak activations that are neighbours of the main pattern and a
  readable negative pole; units without one are listed. Tally per stage, analogous to Goodfire's 90 / 7 / 2–4 %. [E16 neuron atlas;
  later the same on the small transformers' MLP neurons and heads across training checkpoints]
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

A3 v3 (2026-10-16, E5v, verified on a fresh sample): rigid rotations and reflections of whole jets change 3.7–4.1 % of calls. Ten of
the 17 particle objects (S0, S19, S20, S29, S30, S40, S43, S50, S51, S60 — the five costliest among them) move ≤ 0.1 SD; S13, S17,
S63, S1 move 0.1–0.2 SD; S52, S56, S62 and the context-only S31, S58 (built from η- and φ-cut g units) are directional. S29's causal
object is the hinge sum itself (edit slope 0.99); the 3-term name is a correlational proxy (slope 0.83). A2 verified (E5v: 14.61 %,
four objects 13.91 %, random four 0–0.9 %).

Narrative order: architecture identity (A1) → objects (A2, A3) → how mass is computed (A4) → neurons and classes (A5, A6)
→ universality (A7) → a larger model (A8).

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

Open questions for B: does the MARS fit on hand observables match the network on substitution tests as well as the object
formula does? If yes, B is one section of A. Which if-statement form transfers across architectures?

## Datasets
- hls4ml jets (5 classes g, q, W, Z, t): JEDI-linear family, small transformers / Linformers.
- JetClass (10 classes): ParT, ParticleNet, L-GATr.
- Top Tagging Landscape (Kasieczka et al. 2019; top vs QCD; user 2026-10-16: one signal class, so a decision boundary is one
  object set vs background): candidates for pretrained weights online (ParT/ParticleNet fine-tunes, L-GATr `toptagging`
  config, PELICAN, LorentzNet); the mass / three-prong / W-inside-top story of Paper C has a direct test here. Weights check
  running (scout agent) before any training.
- Quark–Gluon (Komiske, Metodiev & Thaler; 2M Pythia jets, with PID; user 2026-10-16): also one signal class; the
  multiplicity / Casimir-scaling story (Frye et al. 1704.06266) has a direct test here. Same weights check first.

Cross-model mass row (2026-10-16, P1; verification P1v running): ParT_full and ParT_kin are not translation invariant either.
A rigid φ shift by 0.06 rad (invariant mass unchanged; only the per-particle dphi and deltaR inputs change) moves true-W → Z
from 12 to 31 % (full) and 16 to 37 % (kin); the class-token read-out mass rises by 0.7–0.8 (full) and 1.1–1.4 (kin) of the
½Δlog(M0·M2) prediction, against 0.78–0.84 for JEDI. On data the two mass definitions coincide (|M1|²/(M0·M2) median 2e-3 on
JetClass, 3.5e-4 on hls4ml) because the stored axis is the pT centroid; the edit identifies which formula each network computes.
Physical consequence under test (P1v): the taggers' dependence on the axis convention (winner-take-all vs centroid axis).
C3 v1 ("ParT's mass is the invariant mass, unlike JEDI's") refuted by P1.
P1v (verified, confidence 0.9): ParT_full's response enters only through the per-particle deltaR input (fed unedited: 92–100 %
of the response disappears); ParT_kin's through dphi (0.65) and deltaR (0.35); the pair features are invariant to float32 noise.
Physical consequence (Paper D, D4): inputs about the winner-take-all axis change 11.7 % (ParT_full) and 34.1 % (ParT_kin) of calls.

## Paper C: circuits (ParT)

**Thesis (v1, 2026-10-16):** ParT computes jet mass through geometric routing, as three linked steps:
1. **Geometric routing:** the pair-bias formula fixes the head roles.
2. **Threshold head:** head b2h8 switches near 85 GeV.
3. **Readout:** far-side heads write the mass into the W/Z neuron n98.

Higher-point correlators may need more depth. Existing evidence: G21b/c, G22, G101, G107, G108, G121–G123, G131, G157.

## Paper D: physics journal

**Thesis (v1, 2026-10-16):** physics-motivated input edits on JEDI-linear and ParT, compared with classical taggers at
their published working points, show which physical quantities each model uses and which it does not. Existing
evidence: J25–J33, G192–G206.
