# Papers: working theses and evidence

Each paper has one working thesis. Every experiment in EXPERIMENTS.md names the paper and thesis claim it tests and is
logged as supports / weakens / refines. When the evidence changes a thesis, the old wording stays below the new one,
with the date and the experiment that changed it.

---

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
- A2. The second round adds jet context that shifts the per-particle cuts. Its size and its role are to be measured. [E4]
- A3. The second-round objects are nameable energy-flow quantities (counts and pT-weighted sums over regions). [E5]
- A4. Jet mass is built from products of moments (m² ≈ 2·ΣpT·Σ pT r² − 2|Σ pT r⃗|²) in d4/d5. [E6]
- A5. The 16 neurons are statements on A3's objects. Replacing each stage by its formula keeps the network's calls. [E7]
- A6. Classes are combinations of neuron objects, and imposing a combination causes the class (J33b on the network's own objects). [E8]
- A7. The same objects appear across JEDI-linear sizes and input sets; an untrained network does not show them. Scope (user,
  2026-10-16: not all 20 official models): 3-feature n8/n32/n64/n128, 3-feature-perminv n64 (ordering control), 16-feature n64
  (input control), untrained n64. [E9]
- A9 (user, 2026-10-16). High-level variables are built over the layers: sums of particles first (P, S), then products
  and ratios (h4, h5), then clustering-type variables, if at all. The layer where a variable first becomes linearly readable
  rises with its algebraic depth. [E10; E6 for mass]
- A8. ParT follows the same scheme: pair-bias statements, head routing, messages, class-token neurons, and combinations. [P-series]
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

Open questions for B: does the MARS fit on hand observables match the network on substitution tests as well as the object
formula does? If yes, B is one section of A. Which if-statement form transfers across architectures?

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
