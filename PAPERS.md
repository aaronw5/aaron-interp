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
- A7. The same objects appear in all 20 official JEDI-linear models; an untrained network does not show them. [E9]
- A8. ParT follows the same scheme: pair-bias statements, head routing, messages, class-token neurons, and combinations. [P-series]

Narrative order: architecture identity (A1) → objects (A2, A3) → how mass is computed (A4) → neurons and classes (A5, A6)
→ universality (A7) → a larger model (A8).

## Paper B: method (statement–object decomposition)

**Thesis (v1, 2026-10-16):** for networks that pool ReLU particle features with a sum (Deep Sets, EFN, and JEDI-type
models), the pooled layer is exactly a set of particle cuts each read by count and summed inputs. This identity plus
stage-by-stage substitution gives a decomposition that is exact where the identity holds. It is tested by substitution
and by causal edits. It is compared with decision-tree extraction, symbolic regression, and probes on the same networks.

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
