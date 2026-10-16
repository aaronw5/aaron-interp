"""Content of the results page (one tab per paper). Edited by the ledger step after each round; site/gen.py renders it.
Writing rule: plain statements of what was measured; no verdict words, no metaphor, no editorial or declarative language.
Each claim carries a `detail` block (rendered as a drop-down): what the claim is, procedure (sample, edit, control, metric),
result (numbers), tables, figures, verification, files. Figure paths are relative to the repo root; missing files are skipped."""

UPDATED = '2026-10-16 (E5v, P1 interim)'

OVERVIEW = dict(
    title='Overview',
    intro='Interpretability of jet taggers through formulas: JEDI-linear (hls4ml trigger tagger) first, then small transformers trained '
          'on the same data, then ParT / ParticleNet / L-GATr on JetClass. One tab per paper; each claim opens to its procedure, numbers, '
          'tables and figures. Tabs merge if papers merge.',
    status=[
        ('Science loop (Paper A + B)', 'round 1 on the cluster: E8 (classes as conjunctions, imposed), E14, E15; then verify → ledger → review'),
        ('E5v', 'done: A2 verified; A3 verified in corrected form (10 of 17 objects radial, 3 directional)'),
        ('E11', 'MARS fit of the entire trigger model, all stages: 10 cluster jobs finished, analysis in progress'),
        ('P1', 'ParT translation test running on an A10 (job anrunw-ai-p1b); interim: at δ = 0.02 ParT_full changes 1.4 % of calls, W → Z 0.114 → 0.122 (JEDI at δ = 0.03: 0.03 → 0.26)'),
        ('Small transformers', 'first set trained (N = 32, seed 0, 30 epochs): tf-2b-d32 77.2 %, dsets-d64 77.8 %, linf-2b-d32-k8 78.4 % vs JEDI n32 78.0 %; the other three jobs queued; per-epoch checkpoints saved'),
        ('Models and data', 'ParticleNet JetClass weights copied; no released L-GATr weights exist, so it is trained in-house; LorentzNet ships top-tagging and QG checkpoints; top-tagging and QG data on the volume'),
    ],
    decisions=[
        ('2026-10-16', 'JEDI scope: the 3-feature 64-particle model (user: "the best version"); the best by accuracy is 16-feature-perminv n64 (82.36 % vs 80.92 %), question open whether to add or switch.'),
        ('2026-10-16', 'Two shapes for Paper A: A-small (complete account on small models; ParT to Paper C) or A-scale (ParT as a main section). Decided after P1 and the first small-transformer results.'),
        ('2026-10-16', 'Papers that become too similar or cannot stand alone are combined. B stands alone only with a method result on ≥ 3 architectures or 2 datasets and the baseline comparison.'),
        ('2026-10-16', 'No paper writing until a paper reaches ~90 % confidence, and the user is asked first.'),
        ('2026-10-16', 'All compute on the NRP cluster (rules in CLUSTER.md); code on github.com/aaronw5/aaron-interp, pushed after every commit.'),
        ('2026-10-16', 'Correction logged in FINDINGS.md: G192\'s jet rebuild mirrored η for jets with jet_eta < 0 (JetClass stores part_deta with the sign of jet_eta); G192–G206 edit effects stand as relative measurements with a 0.9 % systematic.'),
    ],
)

# ---------------------------------------------------------------------------------------------------------------------- Paper A
A_CLAIMS = [
 dict(id='A1', statement='Statements are cuts on single particles; objects are the particles passing a cut within a fixed pT-rank window, read by count and summed (pT, Δη, Δφ).',
      status='verified by rerun (cluster, 260,000 jets)', confidence=0.9, evidence='E1, E2, E3, E3m, E9',
      detail=dict(
        what='JEDI-linear applies 64 ReLU units to each particle\'s normalised (pT, Δη, Δφ), sums them over the 64 pT-ordered slots, and the sum feeds '
             'the rest of the network. For one unit with weights w and bias b, Σ_i ReLU(w·x_i + b) = b·|A| + w·Σ_{i∈A} x_i where A is the set of '
             'particles with w·x_i + b > 0. So each pooled channel is one cut, read through the count of passing particles and their summed inputs. '
             'The fixed-point format (HGQ) gives every (slot, channel) entry its own bit width; entries with 0 bits are always zero, so a channel sums '
             'only over a fixed set of pT ranks.',
        procedure='E1: 10,000 test jets (2,000 per class, seed 0); every unit of every stage set to its mean over the sample; metric = fraction of the '
                  'network\'s calls that change. E2: the first-layer weights converted to GeV and rad through the stored normalisation. E3: the identity '
                  'checked numerically on the network\'s own rounded inputs (before and after the pooling quantizer). E3q/E3w/E3m: the forward pass '
                  'rerun with each quantizer replaced by float arithmetic, then with only the zero-bit masks kept. E9: the same on the cluster for '
                  'all 260,000 test jets, trained and untrained.',
        result='Units changing ≥ 1 % of calls: 13 of 64 first-layer cuts (49 never fire), 11 first-round objects, 7 context units, 19 second-round '
               'objects; the dense layers are denser (h4 42, h5 30, z 14). The identity holds exactly (max error 0.0) before the pooling quantizer; '
               'with the pooling mask the residual is per-slot rounding (7.8 % relative, the same as in the untrained network). Fully float: 43.7 %. '
               'Float with the zero-bit masks kept: 80.68 % against 81.29 % fixed point, 4.49 % of calls different. Zero-bit fractions: qs 0.82, l2 0.79, '
               'dd 0.78, l3 0.80, qs1 0.84, d4 0.70, d5 0.13.',
        tables=[
          ('First-layer cuts (E2; top by E1 cost)', ['unit', 'cut (GeV, rad)', 'kind', 'calls changed when → mean'],
           [['a32', '3.86 − 0.352·pT > 0  (on for pT < 11.0 GeV)', 'soft-particle ramp', '64.2 %'],
            ['a29', 'Δη > 0.007', 'half-plane', '31.9 %'],
            ['a6', '0.62 Δη + 0.78 Δφ > 0.008', 'half-plane', '31.6 %'],
            ['a15', '0.088·pT + 3.94 (always on)', 'ΣpT', '29.4 %'],
            ['a36', 'Δφ < −0.002', 'half-plane', '27.3 %'],
            ['a40', '−0.65 Δη + 0.76 Δφ > 0.003', 'half-plane', '26.3 %'],
            ['a56', '9.36 − 0.302·pT > 0  (on for pT < 31.0 GeV)', 'soft-particle ramp', '22.0 %'],
            ['a43', '−0.91 Δη − 0.41 Δφ > 0.005', 'half-plane', '21.5 %']]),
          ('Rank windows of the second-round objects (zero-bit masks of the second pooling)', ['object', 'kept ranks', 'object', 'kept ranks'],
           [['u0', '0–13 (14 hardest)', 'u19', '13–63'], ['u1', '1–63 (all but the leading)', 'u29, u30', 'all 64'], ['u43, u51, u52, u63', '14 slots', 'u50', '42'], ['u20', '3', 'u60', '26']]),
          ('Quantizers removed (E3q/E3w/E3m; 10,000 jets)', ['forward pass', 'accuracy', 'calls changed vs fixed point'],
           [['fixed point (the network)', '81.29 %', '0'], ['float input quantizer only', '81.21 %', '2.62 %'], ['float first and second pooling', '81.02 %', '3.83 %'],
            ['float everywhere, zero-bit masks kept', '80.68 %', '4.49 %'], ['float everywhere, masks dropped', '43.71 %', '51.6 %'], ['float last layer only', '81.28 %', '0.30 %']]),
        ],
        figures=[],
        verification='E9 on the cluster: 260,000 jets, 80.94 % (official 80.92 %); per-stage counts 13/10/7/9/19/19/38/29/13 agree with E1. '
                     'Untrained control (trained bit widths, redrawn weights): 20.2 %, calls everything top, no unit changes any call (a degenerate control; '
                     'the small-transformer checkpoints will replace it).',
        files='interp/e1_e3_jedi.py, interp/e3w_quantizers.py, interp/e3m_masks.py, interp/e1_e3_sweep.py; results/jedi/e1_e3.json, results/sweep/summary.md')),
 dict(id='A2', statement='The context round carries ΣpT (and three other jet-level objects) into every particle\'s second cut; without it 14.9 % of calls change.',
      status='verified (fresh sample; random-object controls)', confidence=0.85, evidence='E4, E5v',
      detail=dict(
        what='After the first pooling, a 64-unit jet vector g = ReLU(Wd·P + bd) is added to every particle\'s second-round pre-activation. '
             'The question is how much of each second-round cut is the particle\'s own part and how much is this shared context, and whether the decision needs it.',
        procedure='20,000 test jets (4,000 per class, seed 0), accuracy 80.58 %. Each second-round pre-activation split exactly into own + context + '
                  'quantizer residual + bias (sum reproduces the unit with error 0.0). Edits: g set to its mean over jets, to 0, or to another jet\'s g; '
                  'each g unit alone set to its mean; each second-round object alone replaced by its mean-context value. Metric: accuracy and fraction of calls changed.',
        result='Context reaches 9 of 19 live second-round units; between-jet SD(context)/SD(own): u30 0.56, u17 0.29, u20 0.18, u0 0.14, rest ≤ 0.08; '
               'u31 and u58 have no own part. g → mean: 80.58 → 74.02 % (14.90 % of calls change); g → 0: 43.15 %; g → another jet\'s g: 63.64 %. '
               'Single g units: g49 10.38 %, g4 4.17 %, g55 2.20 %, g39 1.98 %. Through the objects: S0 alone 10.38 %, S30 4.17 %, S31 3.55 %, S58 1.92 %; '
               'the four together 14.44 %. g49 feeds only u0 (weight −3.25); 72 % of its variance comes from P15 (ΣpT), the rest from the soft counts P32/P56.',
        tables=[('Context paths', ['g unit', 'feeds', 'built from', 'best observable fit (R²)', 'calls changed when g unit → mean'],
                 [['g49', 'u0', 'P15 (ΣpT) 72 %, P32/P56 soft counts', 'log ΣpT 0.69; 3 terms 0.944', '10.38 %'],
                  ['g4', 'u30', 'three annulus counts', '0.963', '4.17 %'],
                  ['g55, g39, g5', 'u31', 'half-plane P units', '≤ 0.14', '2.20 / 1.98 / 1.57 %'],
                  ['g63', 'u58', '—', '0.46', '1.34 %']])],
        figures=[],
        verification='E5v, fresh 10,000 jets (seed 7): g → mean 80.66 → 74.24 % (14.61 % of calls); the four objects alone 13.91 %, with the same call as g → mean on 98.07 % of '
                     'jets and on 90.3 % of the 1,461 jets that g → mean changes; four random objects from the other 15: 0.00–0.88 %. Qualification: S31 and S58 are directional '
                     'objects (see A3) and S30\'s own term is not Σ r (its edit response comes through g). Verifier confidence 0.9.',
        files='interp/e4_e5_jedi.py; results/jedi/e4_e5.json; inbox/E4_E5_objects.md')),
 dict(id='A3', statement='The second-round objects are hinges in (pT or ln pT, r = ΔR to the axis) summed over fixed pT-rank windows; 10 of the 17 particle objects (the five costliest among them) are radial to ≤ 0.1 SD under rotations of the whole jet, 4 to 0.1–0.2 SD, and 3 (plus the two context-only objects) are directional.',
      status='verified in corrected form: 10 of 17 particle objects radial to ≤ 0.1 SD; 3 directional; rotations change 3.7–4.1 % of calls', confidence=0.8, evidence='E5, E5v',
      detail=dict(
        what='Each second-round object S_c = Σ_i u_{i,c}/16 with u_{i,c} = ReLU(W3 (s_i + g) + b3)_c. With g fixed at its mean, the per-particle '
             'response f_c(pT, Δη, Δφ) was measured directly and then named with jet-level quantities.',
        procedure='f_c from the masked-float forward (zero-bit masks kept, no rounding; 80.14 % on this sample, 4.18 % of calls differ from fixed point), '
                  'over real particles in the unit\'s kept slots, on 20,000 jets. Naming: greedy linear fits over 380 jet observables and ~290 hand-built '
                  'quantities (ΣpT·r², Σr², Σ(r − 0.2)₊, counts above pT thresholds in rank windows), fitted on a random half of the jets, R² on the other half. '
                  'Causal: 10,000 further jets (seed 1; 815 overlap the fit sample); a particle moved across the inferred radial boundary, or a 1 GeV particle '
                  'added at a given r; the measured change of S_c compared with the hinge\'s prediction; control = the same edit on a particle that does not cross.',
        result='Azimuth adds ≤ 0.017 R² to every object. S29 (40.9 % of calls): ReLU(4.44 ln pT + 93.8 r − 21.2), named ΣpT·r² + Σr² + N(pT<2, r>0.1) '
               '(R² 0.976); radial crossing +0.482 measured vs +0.469 predicted, control +0.008; random radial moves: correlation 0.97, slope 1.03. '
               'S1 (20.2 %, ranks 1–63): ReLU(89.3 r − 22.9 + 0.55 ln pT), Σ_{ranks≥1}(r − 0.2)₊ (0.947); +0.457 vs +0.446. S0 (12.0 %, 14 hardest): '
               '0.32 pT + 1.40, ΣpT (0.934; 61 % of its variance through g). S50: ΣpT·r² over ranks 0–38 (0.938). S19 (ranks 13–63): N(pT > 1 GeV) (0.991). '
               'S30: Σ r for r < 0.165 (0.957). S31, S58: functions of g alone.',
        tables=[],
        figures=[('results/site/T1_second_round_objects.png', 'The nine costliest second-round objects: response, rank window, name, causal check (E5).'),
                 ('results/site/T3_rotation_test.png', 'Rotation and reflection test of the objects (E5v).'),
                 ('results/site/T4_e5v_naming_context.png', 'Verification of the S29 name and of the context path (E5v).')],
        verification='E5v, fresh 10,000 jets (seed 7), nothing refit. Rigid rotations by 45°, 90°, 180° and the two reflections (every r unchanged to 1e-7) change '
                     '3.73–4.08 % of calls (float + masks 3.04–3.90 %); a per-particle azimuth scramble 6.85 %. rms ΔS/SD at 180°: S0, S19 0; S50, S40, S29, S43, S51, S30, S60, S20 '
                     '0.06–0.09; S13, S63, S17, S1 0.12–0.21; S62, S52, S56 0.25–0.30; S58 0.41, S31 0.44 (both built from directional g units: g5, g29 η cuts; g39, g55 φ cuts). '
                     'Carriers of the call change (one rotated object patched into the unrotated network): S1 2.27 %, S29 2.11 %, S56 1.58 %, S31 1.09 %, S17 1.05 %. '
                     'S29 name: R² 0.976 on fresh jets; r × 1.1 edit moves S29 by +1.749 vs hinge-sum prediction +1.799 (slope 0.99) and 3-term formula +1.979 (slope 0.83): '
                     'the causal object is Σ ReLU(4.44 ln pT + 93.8 r − 21.2)/16, the 3-term name a correlational proxy. Bit-count check: 0 disagreeing entries, the isolated '
                     'window slots are real. Verifier verdict: R1 refuted as first stated ("radial" for all), confidence 0.85 in the corrected reading; R2 reproduced (0.9).',
        files='interp/e4_e5_jedi.py, interp/e5_causal_jedi.py, interp/e5_verify.py; results/jedi/e4_e5.json, e5_causal.json, e5_verify.json')),
 dict(id='A4', statement='The network\'s mass variable is an un-recentred M2-like quantity: ΣpT and Σ pT r² are read separately at S and multiplied in h4/h5; a rigid shift of the whole jet, which leaves the invariant mass unchanged, flips 77 % of W calls to Z.',
      status='verified (fresh sample, dose–response, direction, input-range and first-round controls)', confidence=0.85, evidence='E6, E6v',
      detail=dict(
        what='For axis-centred particles, m² ≈ M0·M2 − |M1|² with M0 = ΣpT, M1 = Σ pT r⃗, M2 = Σ pT r² (checked: correlation with m 0.999986, median ratio 0.99998). '
             'A sum-pooling network can carry M0, M1, M2 linearly in its pooled layers; the product needs the dense layers. The −|M1|² term is what makes m '
             'independent of a rigid shift of all particles; on data |M1|²/(M0·M2) has median 3.5e-4, so the two definitions coincide unless the jet is shifted.',
        procedure='E6: 15,000 test jets (3,000 per class, seed 6). Ridge read-outs (held-out R², fixed 50/50 split; linear and linear + pairwise products of the '
                  'top-8 PCA components) of M0, |M1|, M2, M0·M2, m², m, log m at every stage. Edits: angles × k (0.8–1.25; M2 × k²), pT × c (M0 × c), rigid translation '
                  'r → r + δ (m unchanged), iso-mass (pT × c, angles × 1/c). Per unit the slopes e_k = dU/dlog k and e_c = dU/dlog c and their ratio ρ (m-like 1, M2-like 2, M0-like 0). '
                  'Controls: g frozen at each jet\'s own value; S0 frozen. E6v: fresh 10,000 jets (seed 17); dose–response δ = 0.01–0.09 along ±η, ±φ, diagonal; '
                  'padded slots checked unmoved; input-range control (jets whose particles stay inside the training 99 % range); the nine angular first-round units frozen.',
        result='Linear R² at S: M0 0.945, M2 0.995, m 0.977; at h4: 0.943, 0.995, 0.991. Units with ρ ≈ 1 (respond as m): S 2, h4 9, h5 8, z 3; with S0 frozen their '
               'ρ rises to 2–5 (M2-like). Translation δ = 0.06: Δlog m̂ at S/h4/h5/z = 0.288/0.275/0.235/0.251 against ½Δlog M2 = 0.300; W → Z 77 % (unedited 3 %). '
               'Dose: observed/predicted 0.46, 0.72, 0.78, 0.79, 0.78, 0.74 at δ = 0.01…0.09; log-log exponent 1.9–2.3 (δ²). Alternatives Σ pT r and √Σ r² predict 0.52 and 0.64 '
               '(excluded by magnitude, exponent 1.1). Direction: odd part 12–20 % of the even part (−η, −φ larger). Input range: not the cause (ratio 0.74 inside the training range vs 0.82 outside). '
               'Nine angular P units frozen: W → Z 0.83 vs 0.77 free, so the response travels a → s → u → S, not through P/g. Iso-mass pT × 0.8: 78 % of W jets → top (pT is used at fixed mass).',
        tables=[('Mass windows of the class neurons (3-knot fits with a per-jet offset)', ['neuron', 'angle-scan knots (GeV)', 'pT-scan knots (GeV)', 'peak mass angle / pT (GeV)'],
                 [['z0 (W)', '80, 110 (falls −0.19/GeV)', '75, 117.5', '74.7 / 76.6'], ['z7 (Z)', '62.5, 92.5, 110', '65, 92.5, 152.5', '94.0 / 96.0'],
                  ['z12', '65, 97.5, 177.5 (lowest near 97.5)', '62.5, 95, 207.5', '—'], ['z14 (Z)', '57.5, 97.5, 255', '42.5, 100, 210', '96.2 / 98.3 (ρ 5.2: not m alone)'],
                  ['z10 (top)', '87.5, 160', '80, 137.5', '148 / 135']])],
        figures=[('results/jedi/F1_mass_translation.png', 'E6v: read-out of log m and the W → Z fraction under a rigid translation along η (10,000 jets, seed 17).'),
                 ('results/site/T2_mass_and_windows.png', 'Summary table of the mass measurements.')],
        verification='E6v reproduced the E6 numbers within 0.01 on a fresh sample. Verdict: not refuted for non-invariance and the M2-sized response; "exactly √(M0·M2) about the axis" '
                     'overstated (odd term 12–20 %; verifier confidence 0.55 for the exact form, > 0.9 for non-recentring). The v2 formula with a factor 2 in PAPERS.md was an error, corrected.',
        files='interp/e6_mass_jedi.py, interp/e6_verify.py; results/jedi/e6_mass.json, e6_verify.json, e6_verify_translation.png')),
 dict(id='A5', statement='The 16 neurons are statements on the network\'s own objects; replacing the neuron stage by the formula keeps the calls.',
      status='next (science loop; E11 MARS fit of every stage finished, analysis in progress)', confidence=0.3, evidence='E7, E11',
      detail=dict(what='Each z neuron as a hinge sum of the second-round objects S (and the dense layers in between) rather than of hand observables.',
                  procedure='Planned: fit per neuron on S units (E11 compositional fits); substitute z by the formula and let the last layer finish; same-call % and accuracy.',
                  result='Pending.', tables=[], figures=[], verification='—', files='interp/e11_mars_all.py')),
 dict(id='A6', statement='Classes are conjunctions of neuron windows; imposing a conjunction on the network causes the class.',
      status='running (loop node E8)', confidence=0.4, evidence='E8; J33b on the formula\'s objects',
      detail=dict(what='J33b showed on the fitted formula that W = n12[W window] ∧ n4[boson] ∧ n0 etc.; here the same on the network\'s own z/S values with causal imposition.',
                  procedure='Loop node E8 (cluster). Metric: fraction of jets called otherwise that get the class when the conjunction\'s neurons are set to their in-object medians; random-neuron control.',
                  result='Pending.', tables=[], figures=[], verification='—', files='—')),
 dict(id='A9', statement='Variables of higher algebraic depth become linearly readable later in the network.',
      status='measured; verification pending; one causal check negative (D2)', confidence=0.6, evidence='E10',
      detail=dict(
        what='45 jet observables labelled by algebraic depth: 0 sums over particles (ΣpT, counts, Σ pT r), 1 ratios/products of two sums (girth, mass, e2, LHA), '
             '2 three-point or ratio-of-ratio quantities (e3, C2, D2, N2, planar flow), 3 clustering/grooming quantities (τN, τ21, τ32, soft-drop mass, subjets).',
        procedure='20,000 test jets, fixed 50/50 split; RidgeCV on standardised live units of P, g, S, h4, h5, z, and an MLP read-out (64 hidden) as the nonlinear ceiling; '
                  'controls: untrained network (same bit widths), ridge on raw flattened input, within-class R². First stage at R² ≥ 0.9 and ≥ 0.7 per observable; Spearman correlation with depth. '
                  'Causal: multiplicity edit (5 softest particles split collinearly: Δn = +5, nothing else changes), girth edit (soft particles moved outward, ΣpT and n fixed), '
                  'D2 edit at fixed mass (one of the three leading particles moved about the others\' centroid); control = whole jet rotated 90°.',
        result='Spearman depth vs first stage at 0.9: ρ 0.51 (p 4e-4; 0.56 without slot-order quantities). Depth 0–1 reach 0.9 at P or S; depth 2–3 mostly never (7/8, 7/9) and peak at h4 '
               '(median linear R² 0.80, 0.75). MLP at S minus best linear at any stage: +0.01 to +0.04. Causal: multiplicity probe at S slope 0.52 (control −0.004); girth probe at S slope 0.91, '
               'correlation 0.95 (control mean |Δ| 0.0018; side effects: mass +18 GeV); D2 probe at h4 slope 0.057, correlation 0.28 (control 0.30): D2 is readable in the data but does not respond to its own edit. '
               'Untrained: R² ≤ 0.006 at S and h4 for all 45 observables (S has 1 live unit).',
        tables=[('First stage at R² ≥ 0.9 / ≥ 0.7 (selection)', ['observable', 'depth', 'first stage ≥ 0.9', 'first stage ≥ 0.7', 'linear R² at P / S / h5'],
                 [['ΣpT', '0', 'P', 'P', '.98 / .96 / .87'], ['n particles', '0', 'S', 'P', '.89 / .99 / .94'], ['Σ pT ΔR²', '0', 'S', 'P', '.74 / .99 / .98'],
                  ['girth', '1', 'S', 'S', '.66 / .99 / .97'], ['mass', '1', 'S', 'P', '.74 / .97 / .99'], ['e2', '1', 'S', 'S', '.49 / .95 / .96'],
                  ['D2', '2', '–', 'h4', '.21 / .63 / .79'], ['C2', '2', '–', 'S', '.48 / .71 / .76'], ['planar flow', '2', '–', 'h4', '.08 / .67 / .67'],
                  ['τ21', '3', '–', 'h4', '.09 / .68 / .74'], ['τ32', '3', '–', '–', '.24 / .36 / .44'], ['soft-drop mass', '3', 'h4', 'S', '.38 / .88 / .88']])],
        figures=[('results/jedi/F2_variables_over_layers.png', 'Median held-out R² (bars = IQR) by stage and depth group; trained linear, trained MLP, untrained linear.')],
        verification='Not yet rerun by a skeptic. Caveats recorded: the untrained control is nearly empty from S on; 99.9 % of particles sit at ΔR < 0.44, so outward edits leave the data range.',
        files='interp/e10_layers_jedi.py; results/jedi/e10_layers.json, e10_causal.json')),
 dict(id='A8/A10', statement='The same stages appear in ParT, ParticleNet and small transformers trained on the same data.',
      status='running (P1 on GPU; training jobs)', confidence=0.2, evidence='P1, X-series',
      detail=dict(what='For each JEDI stage a counterpart in ParT (pair-bias statements, head-pooled sets as objects, class token as context, class-token neurons, conjunctions) and in the small models.',
                  procedure='P1: the same rigid-translation edit on ParT_full and ParT_kin (1,000 jets per class, δ = 0–0.09), with angle scaling as the positive control; metrics: calls changed, W → Z, the W/Z neuron shift, Δlog m̂ vs ½Δlog M2 and vs 0.',
                  result='Interim (ParT_full, 10,000 jets, accuracy 85.9 %; W/Z neuron = index 97, sd 1.756, last layer × sd W +2.06 / Z −0.67; ridge R² for log m from the class token 0.881): '
                         'δ = 0.01: 0.7 % of calls change, W → Z 0.114 (baseline 0.114), Δlog m̂ −0.001 (prediction +0.0003); δ = 0.02 along η / φ: 1.4 / 1.5 % change, W → Z 0.122 / 0.125, '
                         'Δlog m̂ +0.001 / +0.007 (prediction +0.005 / +0.010), slope on prediction 0.02–0.03. JEDI at δ = 0.03: W → Z 0.26–0.29, Δlog m̂ 0.07 (prediction 0.11). '
                         'Larger δ, ParT_kin and the angle-scaling controls pending. Cluster smoke test: ParT_full 86.2 %, ParT_kin 74.4 % (as G192); the corrected rebuild changes 0.05 % of calls.',
                  tables=[], figures=[], verification='—', files='interp/part_core.py, interp/p1_translation_part.py, part_job.sh')),
]

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
        ('A4 v3, E6/E6v', 'The mass variable is an un-recentred M2-like quantity; exact form open (direction-dependent odd term 12–20 %).'),
        ('A2/A3 v2, E4/E5', 'Context enters through four objects; 17 of 19 second-round objects are radial (pT, r) hinges over rank windows.'),
        ('A3 v3, E5v', 'Radial only for 10 of 17 particle objects (to ≤ 0.1 SD); S52, S56, S62 and the context-only S31, S58 are directional; rigid rotations change 3.7–4.1 % of calls.'),
    ],
    claims=A_CLAIMS,
    open=['Exact form of the mass variable (odd term).', 'Which u units combine which half-planes into radial functions (rotation test running).',
          'Neurons as statements on the network\'s own objects (E7/E11) and classes as conjunctions imposed causally (E8).',
          'Small transformers: does the same stage structure appear; do their masses recentre?'],
),
dict(key='b', tab='Paper B', title='B — Turning a model into if-statements',
    thesis='A tagger can be rewritten as if-statements at two levels: (i) MARS-style hinge fits of each unit on observables (forward selection '
           'of max(0, Q − t), max(0, t − Q) and products), validated by stage substitution and causal tuning curves; (ii) the exact '
           'statement–object identity for sum-pooled ReLU layers, which needs no fit. The paper states both, their fidelity measures '
           '(same call, which particles, causal responses), their failure modes, and baselines (tree extraction, symbolic regression, probes) across architectures.',
    thesis_log=[('v1, 2026-10-16', 'Decomposition identity + stage substitution only.'), ('v2, 2026-10-16', 'MARS if-statement method added (user: write about or explore the MARS fit).')],
    claims=[
        dict(id='B1', statement='The pooled-layer identity holds exactly in float for JEDI-type layers.', status='verified', confidence=0.9, evidence='E3, E3m, E9',
             detail=dict(what='Same identity as A1, stated as a property of any ReLU-then-sum layer.', procedure='See A1.', result='Max error 0.0 before the pooling quantizer; 7.8 % relative residual from fixed-point rounding on kept slots.', tables=[], figures=[], verification='E9 (260,000 jets).', files='interp/e1_e3_jedi.py')),
        dict(id='B2', statement='MARS fits of every stage reproduce the network (same call, accuracy), and feature evolution over layers can be read from the fits.', status='fits finished (10 cluster jobs), analysis in progress', confidence=0.3, evidence='E11',
             detail=dict(what='Every live unit of P, g, S, h4, h5, z fitted as hinge sums on (a) physics observables and (b) the previous stage\'s fitted formulas; stage-wise and cumulative substitution into the real network.',
                         procedure='Fit on 2/3 of a 30,000-jet sample (seed 0), validate on 1/3, evaluate on 10,000 further jets (seed 11); ≤ 30 terms per unit; controls: untrained network, shuffled labels.',
                         result='Pending.', tables=[], figures=[], verification='—', files='interp/e11_mars_all.py')),
        dict(id='B3', statement='Decision agreement does not imply mechanism fidelity.', status='earlier evidence on ParT; to redo on JEDI objects', confidence=0.6, evidence='G160, G180–G190',
             detail=dict(what='A formula can match the network\'s calls while attributing them to different particles.', procedure='Earlier: G160 (formula keeps 89 % of ParT\'s accuracy, 1–48 % of its rejection); G180–G190 (half of the key particles; refit cannot get both jet and particles right).', result='To be redone with the object formula on JEDI.', tables=[], figures=[], verification='—', files='distill-part/jld-features/research/FINDINGS.md')),
        dict(id='B4', statement='Baselines (tree extraction, symbolic regression, probes) on the same networks.', status='not started', confidence=0.0, evidence='—',
             detail=dict(what='—', procedure='—', result='—', tables=[], figures=[], verification='—', files='—')),
    ],
    open=['Does MARS on hand observables match the network on substitution tests as well as the object formula? If yes, B folds into A.'],
),
dict(key='c', tab='Paper C', title='C — Circuits in ParT',
    thesis='ParT computes jet mass through geometric routing: the pair-bias formula fixes head roles; head b2h8 switches near 85 GeV; far-side '
           'heads write into the W/Z neuron (index 97, "n98"). Variables of higher algebraic depth may need more blocks.',
    thesis_log=[('v1, 2026-10-16', 'From G21b/c, G22, G101, G107, G108, G121–G123, G131, G157.')],
    claims=[
        dict(id='C1', statement='Attention is mostly the fixed pair bias; the bias is a hinge formula on (ln kT, ln z, ln ΔR, ln m²).', status='earlier evidence', confidence=0.8, evidence='G107, G108',
             detail=dict(what='Each head\'s attention logit = content term + a pair bias shared by all 8 blocks.', procedure='G107: variance share of the bias; attention recomputed from the bias alone. G108: additive hinge formula fitted to the bias (R² 0.86–0.99 per head), ParT rerun with it.', result='Bias explains a median 79 % of logit variance; bias-only attention 69.4 % accuracy, content-only 40.2 %; formula bias 85.0 % vs 85.4 %.', tables=[], figures=[], verification='—', files='FINDINGS.md G107/G108')),
        dict(id='C2', statement='The mass enters through the attention; head b2h8 is a threshold near 85 GeV.', status='earlier evidence', confidence=0.7, evidence='G21b/c, G22',
             detail=dict(what='Decomposition of the W/Z neuron shift under angle scaling into attention and per-particle parts.', procedure='10,000 W/Z jets, angles × 1.1; exact decomposition; b2h8 class-token share vs mass.', result='Shift −0.478 s.d. = attention −0.458 + features −0.019; b2h8 share 87 % at 60–65 GeV → 7 % at 105+ GeV.', tables=[], figures=[], verification='—', files='FINDINGS.md G21b/G21c/G22')),
        dict(id='C3', statement='ParT\'s mass is the invariant mass (translation invariant), unlike JEDI\'s.', status='running (P1)', confidence=0.5, evidence='P1',
             detail=dict(what='ParT\'s pair features are translation invariant by construction; its per-particle inputs (Δη, Δφ, ΔR) are not.', procedure='Same translation edit as E6 on ParT_full and ParT_kin, 1,000 jets per class.', result='Pending.', tables=[], figures=[], verification='—', files='interp/p1_translation_part.py')),
        dict(id='C4', statement='Per-block readability of observables follows algebraic depth.', status='not started', confidence=0.0, evidence='—',
             detail=dict(what='E10 on ParT\'s blocks.', procedure='—', result='—', tables=[], figures=[], verification='—', files='—')),
    ],
    open=['Interchange interventions for the mass subspace.', 'm²_ij-only vs ΔR-only mass edits.'],
),
dict(key='d', tab='Paper D', title='D — Physics edits versus classical taggers',
    thesis='Physics-motivated input edits on JEDI-linear and ParT, compared with classical taggers at their published working points, '
           'show which physical quantities each model uses and which it does not.',
    thesis_log=[('v1, 2026-10-16', 'From J25–J33 and G192–G206 (physics checks page).')],
    claims=[
        dict(id='D1', statement='JEDI: mass windows plus a ΣpT shortcut; no W-mass constraint inside top.', status='earlier evidence', confidence=0.8, evidence='J26–J30',
             detail=dict(what='Physics edits on JEDI-linear.', procedure='J26 angular scaling; J27 boost, collinear split, soft emission; J30 W mass inside top.', result='W → 91.2 GeV: Z 93 %; boost × 0.9: W 87 → 28 %; W mass inside top 88 → 85–88 %.', tables=[], figures=[], verification='—', files='FINDINGS.md J26–J30')),
        dict(id='D2', statement='ParT: displaced-track multiplicity, not lifetime; isolation used; π⁰, dead cone, lepton–b charge link not used.', status='earlier evidence (400 jets per class; η-mirror systematic 0.9 %)', confidence=0.7, evidence='G194, G202–G204',
             detail=dict(what='Physics edits on ParT_full.', procedure='G194 prompt tracks, lifetime scaling, track count; G202 π⁰ pairing; G203 isolation; G204 vertex mass, dead cone, g → bb̄.', result='Prompt: Hbb 92 → 10 %; lifetime × 0.35: 86 %; keep 2 displaced tracks: 32 %; isolation: Tbl 99 → 71 %; π⁰: no peak; dead cone: no effect.', tables=[], figures=[], verification='Rebuild mirrored η for jet_eta < 0 (same on both sides of each comparison).', files='FINDINGS.md G192–G206')),
        dict(id='D3', statement='All edits at full statistics with bootstrap intervals and the corrected rebuild.', status='not started', confidence=0.0, evidence='—',
             detail=dict(what='—', procedure='—', result='—', tables=[], figures=[], verification='—', files='—')),
    ],
    open=['Rerun at 20,000+ jets per class on the cluster with the corrected build.'],
),
]
