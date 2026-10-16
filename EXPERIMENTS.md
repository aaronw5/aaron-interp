# Experiment ledger

One row per experiment. Columns:
- id
- paper / claim
- question
- procedure (sample, edit, control, metric)
- result (numbers)
- what it tells us
- effect on the thesis (supports / weakens / refines / neutral)

Earlier results (J1–J33, G1–G206) are in `distill-part/jld-features/research/FINDINGS.md` and are cited by id.

| id | paper/claim | question | status | result | tells us | thesis |
|---|---|---|---|---|---|---|
| E1 | A1, B | how many units per layer are dead, and how many change ≥1% of calls when set to their mean | done | 10,000 test jets (2,000/class), acc 81.29 %. Never firing / changing ≥1 % of calls: l1 49/64 · 13; P 50 · 11; g 49 · 7 (11 at ≥0.1 %); s 55 · 9; u/S 45 · 19; h4 9 · 42; h5 0 · 30; z 0 · 14. Largest single costs: a32 64.2 %, s14 64.9 %, u29 61.2 %, z1 21.6 % | the particle stage is about 13 cuts, the first objects about 11, the second-round objects 19; the post-pooling MLP is dense | supports A1 (small program) |
| E2 | A1 | each l1 unit as a cut in GeV and rad | done | live cuts: 6 pT ramps (a32: 3.86 − 0.352·pT, on for pT < 11.0 GeV, cost 64 %; a56: pT < 31.0 GeV; a15: 0.088·pT + 3.94, always on) and 9 angular half-planes through ≤0.01 rad of the axis in ~8 directions (a29: Δη > 0.007; a36: Δφ < −0.0024; a6: 0.62Δη + 0.78Δφ > 0.008 …), with zero pT coefficient | particle statements are "how soft" ramps and "which side of the axis" cuts; the angular objects are particle-count weighted, not pT weighted | supports A1; new: angular objects not IRC safe |
| E3 | A1, B | does Σ_i ReLU(w·x_i+b) = b·count + w·Σ_A x hold on the pooled layer P | done | exact (max error 0.0) on the network's own rounded inputs before the pooling quantizer; the pooling quantizer rounds each particle's value per slot (unit 32, slot 63: values snap to 2 or 4), mean |q(a)−a| 0.17–0.25. Variance split of P: P15 = ΣpT (share 1.00); P32 = 0.56 count + 0.44 ΣpT of particles below 11 GeV; angular P units = one-sided Σ Δη / Σ Δφ over particles on one side (shares 0.85–1.03) | each first-round object is one cut read by its count and summed (pT, Δη, Δφ); fixed-point rounding turns soft-particle ramps toward counts | supports A1/B (exact in float) |
| E3q | A1, B | does the decision depend on the fixed-point rounding of the first layers | done | same 10,000 jets; quantizers skipped: l1 input 2.62 % calls changed (acc 81.21 %); qs 1.48 % (81.43); qs1 2.76 % (80.88); l1+qs+qs1 3.83 % (81.02); all quantizers 51.6 % (43.71) | the particle and object stages can be written in float at a 3.8 % call cost; a later quantizer (likely the last layer's modulo wrap) is part of the computation | refines A1: "exact up to 3.8 % of calls" |
| E3w | A1, B | which quantizer does the all-float network (43.7 %) lack | done | 10,000 jets; float everywhere except one layer's quantizer: except l3 input 78.25 % (8.0 % calls changed), except qs1 78.83 % (8.1 %), every other single layer 39–54 %; float last layer d7 alone 0.30 % calls changed (its modulo wrap is not used) | the second-round quantizers carry something the network needs | led to E3m |
| E3m | A1, A3, B | are HGQ's zero-bit entries (fixed per slot × channel masks) the missing part | done | zero-bit fractions: qs 81.6 %, l2 79.4 %, dd 78.1 %, l3 80.1 %, qs1 84.2 %, d4 70.3 %, d5 12.5 %. Float + masks only: 80.68 % (fixed 81.29 %), 4.49 % calls changed; fully float 43.71 %. Slot windows of live S objects: u0, u43 = 14 hardest particles; u19 = slots 13–63; u1 = all but the leading particle; u29, u30, all 64 | the program is float arithmetic plus fixed membership masks: each object = cut ∧ pT-rank window; rounding and wrap are a 4.5 % perturbation | refines A1/A3: objects are defined by a cut AND a rank window |
| E10 | A9 | are high-level variables built over the layers, in order of algebraic depth | running (agent) | | | |
