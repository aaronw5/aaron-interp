"""Content of the results page (one tab per paper). Edited by the ledger step after each round; site/gen.py renders it.
Writing rule: plain statements of what was measured; no verdict words, no metaphor, no editorial or declarative language.
Figures: paths relative to the repo root (PNG); missing files are skipped with a note."""

UPDATED = '2026-10-16'

OVERVIEW = dict(
    title='Overview',
    intro='Interpretability of jet taggers through formulas: JEDI-linear (hls4ml trigger tagger) first, then small transformers trained '
          'on the same data, then ParT / ParticleNet / L-GATr on JetClass. One tab per paper. Tabs merge if papers merge.',
    status=[
        ('Science loop (Paper A + B)', 'running on the cluster: round 1 (plan → run → verify → ledger → review)'),
        ('E5v', 'rotation / reflection verification of the second-round objects'),
        ('E11', 'MARS fit of the entire trigger model, all stages (cluster jobs)'),
        ('P0 / P1', 'ParT setup and the translation test (cluster GPU)'),
        ('Small transformers', 'native transformer / Linformer / Deep Sets on hls4ml, per-epoch checkpoints (cluster GPU)'),
        ('Models and data', 'ParticleNet weights copied; L-GATr: no released weights anywhere, so trained in-house; top-tagging and QG data on the volume'),
    ],
    decisions=[
        ('2026-10-16', 'JEDI scope: the 3-feature 64-particle model only (user). Cross-model generality from other architectures.'),
        ('2026-10-16', 'Two shapes for Paper A: A-small (complete account on small models; ParT to Paper C) or A-scale (ParT as a main section). Decided after P1 and the first small-transformer results.'),
        ('2026-10-16', 'Papers that become too similar or cannot stand alone are combined (criteria in PAPERS.md). B stands alone only with a method result on ≥ 3 architectures or 2 datasets and the baseline comparison.'),
        ('2026-10-16', 'No paper writing until a paper reaches ~90 % confidence, and the user is asked first.'),
        ('2026-10-16', 'All compute on the NRP cluster (rules in CLUSTER.md).'),
    ],
)

PAPERS = [
dict(key='a', tab='Paper A', title='A — How a tagger builds its decisions',
    thesis='Class boundaries are built in four stages, each written as a formula in the network\'s own terms: statements (cuts on single '
           'particles); objects (the particles passing a cut within a fixed pT-rank window, read as count and summed pT, Δη, Δφ); neurons '
           '(statements on objects); classes (combinations of neurons). High-level variables appear over the layers in order of algebraic '
           'complexity. For JEDI-linear the whole chain from input to logits is written out and checked stage by stage.',
    thesis_log=[
        ('v1, 2026-10-16', 'Four stages; mass as products of moments (with an erroneous factor 2).'),
        ('refinement 1, E3/E3q', 'The identity is exact in floating point; the fixed-point rounding changes 3.8 % of calls.'),
        ('refinement 2, E3w/E3m', 'The zero-bit entries of the fixed-point format are masks: an object is a cut AND a pT-rank window (float + masks 80.68 % vs 81.29 %).'),
        ('A4 v3, E6/E6v', 'The mass variable is an un-recentred M2-like quantity (Σ pT r² times ΣpT, no −|Σ pT r⃗|² term); exact form open (direction-dependent odd term 12–20 %).'),
        ('A2/A3 v2, E4/E5', 'Context enters through four objects; 17 of 19 second-round objects are radial (pT, r) hinges over rank windows.'),
    ],
    claims=[
        ('A1', 'statements = cuts on (pT, Δη, Δφ); objects = cut ∧ rank window, read by count and sums (exact identity)', 'verified by rerun', 0.9, 'E1–E3, E3m'),
        ('A2', 'the context round carries ΣpT (and three other objects) into the second cuts; g → mean changes 14.9 % of calls', 'verification run pending', 0.6, 'E4, E5v'),
        ('A3', '17 of 19 second-round objects are (pT, r) hinges over rank windows; radial symmetry built from half-planes', 'verification script ready, run pending; code review found a slot-window bit-count error and octant means varying 2–2.5×', 0.5, 'E5, E5v'),
        ('A4', 'mass = un-recentred M2-like product formed in h4/h5; a rigid shift of the jet flips W → Z', 'verified (fresh sample, dose, controls)', 0.85, 'E6, E6v'),
        ('A5', 'the 16 neurons are statements on the network\'s own objects; stage substitution keeps the calls', 'next (loop)', 0.3, 'E7, E11'),
        ('A6', 'classes are conjunctions of neuron windows; imposing one causes the class', 'next (loop); J33b on the formula\'s objects', 0.4, 'E8'),
        ('A9', 'variables of higher algebraic depth become readable later (Spearman 0.51)', 'verification pending; D2 readable but not causally responsive', 0.6, 'E10'),
        ('A8/A10', 'the same stages in ParT, ParticleNet, small transformers', 'running (P1); training', 0.2, 'P-series, X-series'),
    ],
    figures=[
        ('results/jedi/F1_mass_translation.png', 'Mass under a rigid translation of all particles (E6v). Left: read-out of log m at S, h4, h5, z rises as δ², at 0.74–0.84 of the ½ Δlog M2 prediction; Σ pT r is excluded. Right: W jets become Z calls (77 % at δ = 0.06); freezing the nine angular first-round units does not remove the effect. 10,000 test jets, seed 17.'),
        ('results/site/T2_mass_and_windows.png', 'How the mass is formed and read (E6, E6v): M0 and M2 readable at S; units responding as m does in h4/h5/z; window knots of the W, Z and top neurons in GeV.'),
        ('results/site/T1_second_round_objects.png', 'The second-round objects (E5): per-particle response, rank window, name with held-out R², causal check. Verification by rotation and reflection running.'),
        ('results/jedi/F2_variables_over_layers.png', 'Held-out linear R² of 45 observables by stage, grouped by algebraic depth (E10). Depth 0–1 reach 0.9 at P or S; depth 2–3 peak at h4 and mostly stay below 0.9. Untrained control: nothing readable from S on. 20,000 jets.'),
    ],
    experiments=[
        ('E1', 'units per layer; mean ablation', '13 live l1 cuts, 11 P, 19 S; a32 changes 64 % of calls', 'done'),
        ('E2', 'cuts in GeV and rad', 'pT ramps (a32 on for pT < 11 GeV) and nine half-planes through the axis, zero pT weight', 'done'),
        ('E3 / E3m', 'count + sum identity; masks', 'exact before the pooling quantizer; float + masks 80.68 % (fixed 81.29 %), fully float 43.7 %', 'done'),
        ('E4', 'context round', 'g → mean 80.58 → 74.02 %; S0 (ΣpT via g49 ← P15), S30, S31, S58 carry 14.4 of the 14.9 %', 'verification running'),
        ('E5', 'second-round objects', '17/19 = (pT, r) hinge × rank window, R² 0.67–0.996; S29 radial edit +0.482 vs +0.469 predicted', 'verification running'),
        ('E6 / E6v', 'mass', 'translation δ = 0.06: Δlog m̂ 0.24–0.29 (pred 0.30), W → Z 77 %; odd term 12–20 %; windows z0 75, z7/z14 94–98, z10 135–148 GeV', 'verified'),
        ('E10', 'variables over layers', 'Spearman depth vs first stage 0.51; girth edit slope 0.91 at S; D2 edit slope 0.06 at h4', 'verification pending'),
    ],
    open=['Exact form of the mass variable (odd term).', 'Which u units combine which half-planes into radial functions.',
          'Neurons as statements on the network\'s own objects (E7) and classes as conjunctions imposed causally (E8).',
          'Small transformers: does the same stage structure appear; do their masses recentre?'],
),
dict(key='b', tab='Paper B', title='B — Turning a model into if-statements',
    thesis='A tagger can be rewritten as if-statements at two levels: (i) MARS-style hinge fits of each unit on observables (forward selection '
           'of max(0, Q − t), max(0, t − Q) and products), validated by stage substitution and causal tuning curves; (ii) the exact '
           'statement–object identity for sum-pooled ReLU layers, which needs no fit. The paper states both, their fidelity measures '
           '(same call, which particles, causal responses), their failure modes, and baselines (tree extraction, symbolic regression, probes) across architectures.',
    thesis_log=[('v1, 2026-10-16', 'Decomposition identity + stage substitution only.'), ('v2, 2026-10-16', 'MARS if-statement method added (user: write about or explore the MARS fit).')],
    claims=[
        ('B1', 'the pooled-layer identity holds exactly in float for JEDI-type layers', 'verified', 0.9, 'E3, E3m'),
        ('B2', 'MARS fits of every stage reproduce the network (same call, accuracy); feature evolution over layers readable from the fits', 'running (E11)', 0.3, 'E11'),
        ('B3', 'decision agreement does not imply mechanism fidelity (earlier: G160, G180–G190 on ParT)', 'earlier evidence; to redo on JEDI objects', 0.6, 'G160, G180–G190'),
        ('B4', 'baselines (tree extraction, symbolic regression, probes) on the same networks', 'not started', 0.0, '—'),
    ],
    figures=[],
    experiments=[('E11', 'MARS fit of the entire trigger model', 'running on the cluster', 'running')],
    open=['Does MARS on hand observables match the network on substitution tests as well as the object formula? If yes, B folds into A.'],
),
dict(key='c', tab='Paper C', title='C — Circuits in ParT',
    thesis='ParT computes jet mass through geometric routing: the pair-bias formula fixes head roles; head b2h8 switches near 85 GeV; far-side '
           'heads write into the W/Z neuron n98. Variables of higher algebraic depth may need more blocks.',
    thesis_log=[('v1, 2026-10-16', 'From G21b/c, G22, G101, G107, G108, G121–G123, G131, G157.')],
    claims=[
        ('C1', 'attention is mostly the fixed pair bias; the bias is a hinge formula on (ln kT, ln z, ln ΔR, ln m²)', 'earlier evidence', 0.8, 'G107, G108'),
        ('C2', 'the mass enters through the attention; b2h8 is a threshold near 85 GeV', 'earlier evidence', 0.7, 'G21b/c, G22'),
        ('C3', 'ParT\'s mass is the invariant mass (translation invariant), unlike JEDI\'s', 'running (P1)', 0.5, 'P1'),
        ('C4', 'per-block readability of observables follows algebraic depth', 'not started', 0.0, 'E10 on ParT'),
    ],
    figures=[], experiments=[('P1', 'translation test on ParT_full / ParT_kin', 'cluster GPU', 'running')],
    open=['Interchange interventions for the mass subspace.', 'm²_ij-only vs ΔR-only mass edits.'],
),
dict(key='d', tab='Paper D', title='D — Physics edits versus classical taggers',
    thesis='Physics-motivated input edits on JEDI-linear and ParT, compared with classical taggers at their published working points, '
           'show which physical quantities each model uses and which it does not.',
    thesis_log=[('v1, 2026-10-16', 'From J25–J33 and G192–G206 (physics checks page).')],
    claims=[('D1', 'JEDI: mass windows + a ΣpT shortcut; no W-mass constraint inside top', 'earlier evidence', 0.8, 'J26–J30'),
            ('D2', 'ParT: displaced-track multiplicity not lifetime; isolation used; π⁰, dead cone, lepton–b charge not used', 'earlier evidence (400 jets/class)', 0.7, 'G194, G202–G204'),
            ('D3', 'all edits at full statistics with bootstrap intervals', 'not started', 0.0, '—')],
    figures=[], experiments=[], open=['Rerun at 20,000+ jets per class on the cluster.'],
),
]
