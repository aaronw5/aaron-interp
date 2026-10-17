import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import os, pathlib
S = str(pathlib.Path(__file__).resolve().parents[1] / 'results' / 'site') + '/'; os.makedirs(S, exist_ok=True)
def table(fname, title, cols, rows, widths, note=None, h=0.5):
    fig, ax = plt.subplots(figsize=(sum(widths), 0.9 + h*len(rows) + (0.5 if note else 0)), dpi=160); ax.axis('off')
    t = ax.table(cellText=rows, colLabels=cols, colWidths=[w/sum(widths) for w in widths], loc='upper center', cellLoc='left', colLoc='left')
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.75)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor('#d8d4ca'); cell.PAD = 0.03
        if r == 0: cell.set_facecolor('#eeebe3'); cell.set_text_props(weight='bold')
        elif r % 2 == 0: cell.set_facecolor('#f8f7f3')
    ax.set_title(title, loc='left', fontsize=11, weight='bold', pad=8)
    if note: fig.text(0.01, 0.01, note, fontsize=8, color='#555')
    fig.tight_layout(); fig.savefig(S + fname, bbox_inches='tight'); print(fname)
table('T1_second_round_objects.png', 'JEDI-linear: the second-round objects (E5; 20,000 test jets; verification running)',
 ['object', 'cost (calls\nchanged)', 'pT ranks', 'per-particle response f(pT [GeV], r = ΔR to axis)', 'name (held-out R²)', 'causal check: measured vs predicted'],
 [['S29', '40.9 %', 'all', 'ReLU(4.44 ln pT + 93.8 r − 21.2)', 'ΣpT·r² + Σr² + N(pT<2, r>0.1)  (0.976)', 'r crossing +0.482 vs +0.469; control +0.008'],
  ['S1', '20.2 %', '1–63', 'ReLU(89.3 r − 22.9 + 0.55 ln pT):  r > 0.25', 'Σ ranks≥1 (r − 0.2)₊  (0.947)', '+0.457 vs +0.446; control +0.004'],
  ['S0', '12.0 %', '0–13', '0.32 pT + 1.40  (no r term)', 'ΣpT of the 14 hardest (0.934; 61 % via g)', 'all pT × 1.1: +5.43 (g frozen +1.71)'],
  ['S50', '11.9 %', '0–38', 'ReLU(7.07 ln pT + 85.3 r − 34.7)', 'ΣpT·r² over ranks 0–38  (0.938)', '—'],
  ['S19', '11.1 %', '13–63', 'ReLU(2.86 − 1.39 pT):  pT < 2.06 GeV', 'N(pT > 1 GeV) among ranks 13–63  (0.991)', '—'],
  ['S30', '8.0 %', 'all', 'ReLU(3.47 − 21.0 r):  r < 0.165', 'Σ r  (0.957)', '—'],
  ['S43', '7.8 %', '0–13', 'ReLU(0.105 pT + 74.3 r − 11.3)', 'm of the 15 hardest + 2 terms  (0.932)', '—'],
  ['S60', '7.0 %', '0–25', 'ReLU(5.14 ln pT − 142 r − 6.8):  narrow core', '3 terms  (0.969)', '—'],
  ['S31, S58', '3.6 %, 1.9 %', '—', 'no particle part', 'linear in the context g  (0.98, 0.85)', '—']],
 [0.7, 1.0, 0.8, 3.4, 3.6, 3.4],
 note='Azimuth adds ≤ 0.017 R² to every object: the directional half-planes of round one are radial functions by round two. 17 of 19 objects fit a (pT, r) hinge over a rank window.')
table('T2_mass_and_windows.png', 'JEDI-linear: how the mass is formed and read (E6 + verification; 15,000 + 10,000 test jets)',
 ['stage / unit', 'measurement', 'number'],
 [['S', 'linear R² for M0 = ΣpT  |  M2 = Σ pT r²  |  m', '0.945  |  0.995  |  0.977'],
  ['h4', 'same', '0.943  |  0.995  |  0.991'],
  ['h4, h5, z', 'units responding to angle and pT as m does (ratio of slopes ≈ 1)', '9, 8, 3 units; with S0 frozen their ratio → 2–5 (M2-like)'],
  ['S0 → g49 → P15', 'route of M0 into the product', 'freezing S0 halves the pT response of the mass read-out'],
  ['translation δ = 0.06 rad', 'Δ log m̂ at S / h4 / h5 / z  (invariant m unchanged to 1e-7)', '0.288 / 0.275 / 0.235 / 0.251  (½ Δlog M2 predicts 0.300)'],
  ['translation, dose', 'obs / pred at δ = .01 .02 .03 .045 .06 .09', '0.46  0.72  0.78  0.79  0.78  0.74;  exponent ≈ 2 (δ²)'],
  ['translation, calls', 'W jets called Z at δ = 0.03 / 0.06', '28 % / 77 %  (unedited 3 %)'],
  ['direction', 'odd part of the response (−η vs +η)', '12–20 % of the even part: not an exact r² function'],
  ['alternatives', 'Σ pT r (G1), √Σ r²', 'excluded: ×1.7, ×2.1 too large; exponent 1.1'],
  ['z0 (W)', 'window knots, angle scan | pT scan; peak mass', '80, 110 | 75, 117 GeV;  75 / 77 GeV'],
  ['z7 (Z)', 'same', '62, 92, 110 | 65, 92, 152;  94 / 96 GeV'],
  ['z14 (Z)', 'same', '57, 97, 255 | 42, 100, 210;  96 / 98 GeV  (not m alone)'],
  ['z10 (top)', 'same', '87, 160 | 80, 137;  148 / 135 GeV'],
  ['iso-mass edit', 'pT × 0.8, angles × 1.25 (m fixed)', '78 % of W jets → top: pT is used at fixed mass']],
 [1.6, 4.0, 4.2])
table('T3_rotation_test.png', 'JEDI-linear: are the second-round objects radial? Rigid rotations and reflections of whole jets (E5v; 10,000 fresh test jets, seed 7)',
 ['edit (every r unchanged)', 'calls changed\n(fixed point)', 'calls changed\n(float + masks)', 'objects moving > 0.25 SD (rms ΔS / SD)', 'carriers of the call change\n(one rotated object patched in)'],
 [['rotate 45°', '3.81 %', '3.13 %', 'S58 0.68, S31 0.44, S56 0.34, S52 0.33, S62 0.26', '—'],
  ['rotate 90°', '4.08 %', '3.90 %', 'same ordering', '—'],
  ['rotate 180°', '3.87 %', '3.17 %', 'S31 0.44, S58 0.41, S56 0.30, S52 0.29, S62 0.25', 'S1 2.27 %, S29 2.11 %, S56 1.58 %,\nS31 1.09 %, S17 1.05 %'],
  ['reflect Δφ → −Δφ', '3.73 %', '3.04 %', 'same ordering', '—'],
  ['reflect Δη → −Δη', '4.08 %', '3.62 %', 'same ordering', '—'],
  ['per-particle azimuth scramble', '6.85 %', '6.37 %', 'S58 1.52, S31 1.25', '—'],
  ['— radial to ≤ 0.1 SD under every edit', '', '', 'S0, S19, S20, S29, S30, S40, S43, S50, S51, S60\n(10 of 17 particle objects; the five costliest among them)', ''],
  ['— 0.1–0.2 SD', '', '', 'S13, S17, S63, S1', ''],
  ['— directional', '', '', 'S52, S56, S62; S31, S58\n(built from the η-cut units g5, g29 and φ-cut units g39, g55)', '']],
 [2.7, 1.4, 1.5, 5.2, 3.4],
 note='First round under the same edits: 9 of 13 live P units move 0.7–1.2 SD (the half-planes); P15, P21, P32, P56 (pT units) move 0. Accuracy stays 80.7–81.0 %.')
table('T4_e5v_naming_context.png', 'JEDI-linear: verification of the S29 name and of the context path (E5v, fresh sample)',
 ['check', 'measurement', 'result'],
 [['S29 name on fresh jets', 'stored 3-term formula ΣpT·r² + Σr² + N(pT<2, r>0.1), R² vs the network\'s S29', '0.976 (0.976 on the 9,233 jets outside the fit sample); ΣpT·r² alone 0.880; hinge sum 0.937'],
  ['S29 causal edit', 'r × 1.1 for particles with r > 0.15 (14.1 particles/jet): measured ΔS29 vs predicted', 'measured +1.749 ± 0.016; hinge sum +1.799 (slope 0.99, corr 0.95); 3-term formula +1.979 (slope 0.83, corr 0.94); ΣpT·r² alone +1.145'],
  ['same edit, other objects', 'slope of measured on predicted (3-term)', 'S1 1.05 (corr 0.93); S50 0.99 (0.92); S30 −0.301 measured but −0.048 with g frozen (its response comes through g)'],
  ['context g → mean', 'accuracy; calls changed; per class', '80.66 → 74.24 %; 14.61 % (g 21.3, q 14.4, W 16.4, Z 7.5, t 13.4 %)'],
  ['four objects S0, S30, S31, S58 → mean-context values', 'calls changed; agreement with g → mean\'s calls', '13.91 % (acc 74.81 %); same call as g → mean on 98.07 % of jets and 90.3 % of the 1,461 jets g → mean changes'],
  ['single objects', 'calls changed', 'S0 9.91, S30 4.22, S31 3.32, S58 1.75, S17 0.82, S13 0.31 %'],
  ['controls', '5 random draws of 4 from the 15 other live objects', '0.00, 0.82, 0.42, 0.17, 0.88 %'],
  ['code check', 'HGQ bit count b = i+f vs k+i+f', '0 disagreeing entries: the isolated slots in the windows are real']],
 [3.0, 4.2, 5.4])
table('T5_e11_program.png', 'JEDI-linear written as if-statements, stage by stage (E11; 8,848 held-out test jets; network 80.42 % on them)',
 ['stage replaced\n(with all stages before it)', 'units', 'held-out R²\nmedian (min)', 'same call as\nthe network', 'accuracy', 'control: units at\ntheir mean instead'],
 [['P  first objects', '12', '0.817 (0.785)\n+ half-plane sums: 0.996 (0.974)', '95.7 %  (98.8 %)', '80.22 %', '81.6 % same call, 71.91 %'],
  ['g  context', '11', '0.572 (0.390)\n+ half-plane sums: 0.950 (0.881)', '96.2 %  (98.4 %)', '80.72 %', '85.3 %, 74.28 %'],
  ['S  second objects', '19', '0.982 (0.555)', '94.2 %  (95.3 %)', '80.99 %', '18.7 %, 19.91 %'],
  ['h4', '54', '0.958 (0.610)', '94.6 %  (95.1 %)', '80.83 %', '18.7 %, 19.91 %'],
  ['h5', '32', '0.946 (0.653)', '94.2 %  (95.0 %)', '80.63 %', '19.9 %, 20.14 %'],
  ['z  class neurons\n→ network\'s last layer', '16', '0.941 (0.825)', '94.0 %  (94.6 %)', '80.72 %  (80.75 %)', '19.9 %, 20.14 %']],
 [2.6, 0.7, 3.0, 1.9, 1.9, 2.4],
 note='Nested program: each stage fitted on the previous stage\'s fitted values, starting from physics quantities computed from the particles (no network activation enters).\n'
      'Size: 144 units, 3,824 terms (2,344 products), 6,064 if-statements, 211 physics inputs. Per class g/q/W/Z/t: program 78.0 / 70.7 / 87.1 / 85.0 / 82.8, network 77.8 / 70.9 / 87.0 / 83.4 / 82.9.\n'
      'Controls: untrained network R² S 0.08, h4 0.17, h5 0.18, z 0.14; shuffled S targets ≈ 0. Parentheses: with one-sided half-plane sums added to the inputs. Verification running.')
table('T6_axis_convention.png', 'ParT under a change of jet-axis convention (P1v; 10,000 JetClass test jets; 4-vectors unchanged, only Δη, Δφ, ΔR recomputed)',
 ['axis the inputs are measured from', 'axis displacement\n(median, rad)', '|M1|² / (M0·M2)\n(median)', 'ParT_full: accuracy;\ncalls changed; W → Z', 'ParT_kin: accuracy;\ncalls changed; W → Z', 'Δlog m̂ full / kin\n(prediction)'],
 [['stored axis (the training convention)', '0', '0.0020', '85.9 %; 0; 0.117', '74.5 %; 0; 0.162', '0 / 0'],
  ['E-scheme axis recomputed from the stored particles', '3e-4', '≈ 0.002', '0.08 % changed', '0.18 % changed', '—'],
  ['winner-take-all (WTA) axis', '0.095 (W 0.075, QCD 0.030)', '0.22', '80.9 %; 11.7 %; 0.340', '56.0 %; 34.1 %; 0.377', '0.133 / 0.125 (0.138)'],
  ['hardest-particle axis', '—', '—', '14.0 % changed', '38.0 % changed', '—'],
  ['rigid φ shift by 0.06 (P1, for scale)', '0.06', '—', '8.1 % changed; 0.310', '19.6 % changed; 0.373', '0.052 / 0.095 (0.073)']],
 [3.6, 1.9, 1.5, 2.4, 2.4, 2.1],
 note='Input location (P1v): for ParT_full the response enters only through the per-particle deltaR input (fed unedited, 92–100 % of Δlog m̂ and 98–100 % of the W → Z change disappear);\n'
      'for ParT_kin through dphi (0.65) and deltaR (0.35). The pair features (ln kT, ln z, ln ΔR, ln m²) are unchanged to within float32 noise. Caveat: 2.6 % of jets change the sign of the axis η under WTA.')
table('T11_e11_faithfulness.png', 'If-statement programs vs JEDI-linear on ordinary and on edited jets (E11v; inputs recomputed from the edited particles)',
 ['program', 'same call as the network\n(17,011 fresh jets)', 'accuracy\n(network 80.74 %)', 'rigid shift 0.06 rad: W → Z\n(network 74.5 %)', 'rigid shift: same call\nas the network', 'pT × 0.8: Z → t\n(network 62.5 %)', 'pT × 0.8: same call\nas the network'],
 [['nested program (each layer fitted on the previous layer\'s formulas)', '94.23 %', '80.44 %', '77.6 %', '73.9 %', '71.2 %', '88.1 %'],
  ['nested program + half-plane sums as inputs', '94.84 %', '80.52 %', '81.5 %', '81.9 %', '63.8 %', '90.3 %'],
  ['class neurons fitted directly on physics observables', '92.50 %', '80.25 %', '0.1 %', '51.8 %', '61.3 %', '83.9 %']],
 [4.6, 2.2, 1.8, 2.4, 2.1, 2.1, 2.1],
 note='All three programs agree with the network on 92.5–94.8 % of ordinary jets. Under the rigid shift (invariant mass unchanged) only the nested programs reproduce the network\'s W → Z response:\n'
      'they compute the mass the way the network does (radial sums at P and S, then a product). The direct formula uses the invariant mass and does not respond. W rejection at 50 % efficiency: program 0.70× the network\'s [0.50, 0.95].')
table('T17_atlas_tally.png', 'Neuron atlas (E16): how many live units have one main job, related jobs, unrelated jobs, or no description (20,000 test jets)',
 ['layer', 'live units', 'one main job', 'related jobs\n(same observable, two jet populations)', 'unrelated jobs', 'unexplained\n(best description R² < 0.5)'],
 [['P  first-round objects', '12', '0', '12', '0', '0'],
  ['g  jet context', '11', '3', '5', '3', '0'],
  ['S  second-round objects', '19', '1', '17', '0', '1'],
  ['h4', '54', '13', '30', '7', '4'],
  ['h5', '32', '10', '12', '8', '2'],
  ['z  class neurons', '16', '4', '6', '6', '0'],
  ['all', '144', '31  (21.5 %)', '82  (56.9 %)', '24  (16.7 %)', '7  (4.9 %)'],
  ['all, weighted by decision cost', '', '13.6 %', '72.5 %', '12.5 %', '1.4 %']],
 [3.0, 1.1, 1.7, 3.0, 1.6, 2.2],
 note='One main job: one cluster of the jets where the unit is on holds ≥ 80 % of its activation. Related: every cluster with ≥ 10 % of the activation is best described by an observable with |Spearman ρ| ≥ 0.7 to the unit\'s own\n'
      '(mostly the same quantity over a lower- and a higher-mass population). Steering (setting the unit to its 99th percentile) changes a median 9.9 % of calls, the same as for a random unit of the same layer (9.8 %): steering is specific only for the class neurons z.')
table('T18_e8_conjunctions.png', 'Classes as conjunctions of neuron windows on z (E8 + verification; 10,000 evaluation jets each; rules fit on 20,000)',
 ['class', 'rule on z (tree, fit jets)', 'precision / coverage\nE8 → fresh', 'switched, rule units at\nin-rule medians: E8 → fresh', 'rank of rule set\nof 560 triples', 'highest triple\n(switched)', 'units moved just inside\nthe windows (edge)', 'still called c after rule\nunits → other medians'],
 [['g', 'z1 ≥ 4.33, z4 < 0.435, z8 < 3.87', '96.5/74.8 → 98.1/76.4 %', '54.4 → 54.0 %', '6', 'z1 z4 z9 (61.6 %)', '17.0 (12.8) %', '0.0 %'],
  ['q', 'z1 < 3.89, z3 > 0, z12 ≥ 1.38', '98.2/72.8 → 98.4/71.5 %', '31.2 → 31.8 %', '39', 'z1 z4 z9 (52.5 %)', '25.1 (4.7) %', '96.1 %'],
  ['W', 'z0 ≥ 1.35, z1 < 4.33, z4 > 0', '98.4/82.8 → 98.3/82.0 %', '42.0 → 40.6 %', '10', 'z0 z4 z7 (48.3 %)', '7.6 (3.4) %', '22.3 %'],
  ['Z', 'z0 < 1.35, z5 ≥ 0.44, z7 ≥ 1.04', '96.7/76.6 → 97.4/77.3 %', '62.0 → 61.4 %', '6', 'z0 z7 z8 (75.7 %)', '21.5 (9.2) %', '57.4 %'],
  ['t', 'z1 < 4.87, z4 > 0, z8 ≥ 2.59', '95.8/69.5 → 96.2/69.7 %', '26.0 → 26.2 %', '78', 'z8 z10 z13 (77.3 %)', '7.2 (5.7) %', '89.7 %'],
  ['controls', 'triples without a rule unit', '—', 'mean 3.8–7.7 %;\nmax 12.8–58.1 %', '—', 'all 16 neurons: 100 %', '—', '—'],
  ['basis', 'trees in 5 random rotations of z', 'precision 87–99 %;\ncoverage 46–82 %', 'switched 2–69 %', '—', '—', '—', '—']],
 [0.8, 2.7, 2.2, 2.4, 1.5, 1.7, 1.9, 1.9],
 note='Fresh sample: sample(7800, seed 23) minus the 4,545 jets shared with E8; 6,000 per class. The logit change of a 3-unit edit equals the sum of the single-unit changes exactly (max |difference| 0.0).\n'
      'Window masses (25/50/75 %): z0 ≥ 1.35: 78/80/81 GeV (E6 W window); z7 ≥ 1.04: 90/91/92 GeV (E6 Z window); z8 ≥ 2.59: 124/153/170 GeV (not E6\'s top window z10). At S: g 47.8, q 21.4, W 5.5, Z 15.5, t 33.7 % switched.')
table('T19_e14_halfplanes.png', 'How round two turns the nine half-planes into near-radial objects (E14 + verification; 10,000 jets each, seeds 8 and 31)',
 ['measurement', 'E14 (seed 8)', 'verification (seed 31)'],
 [['units with half-plane input / via one s unit / via 2–3 s units', '15 u units via s43, s60, s53, s32, s52', '11 single-path; u13, u43 (s32, s53, s60), u40 (s32, s52, s53), u60 (s43, s53)'],
  ['half-planes with nonzero weight per unit', '"nine-direction polygon"', '3 (u51, u63); 4 (u17, u20, u30, u52, u62); 5 (u1, u29, u50, u56);\n6 (u60); 7 (u13, u43); 9 (u40 only)'],
  ['on-contour max/min (78 contours)', '1.14–1.47', '1.142–1.465'],
  ['rms error of r_c(θ): Part-1 weight polygon', 'median 6e-8; ≤ 0.005 in 60/78; max 0.064', 'median 5.5e-8; ≤ 0.005 in 60/78; max 0.064'],
  ['   best circle about the axis / Fourier m ≤ 4', '—', '0.032–0.086 / 0.019–0.037'],
  ['   9 random directions, fitted weights (median)', '0.046–0.089', '0.046–0.083 (below the circle in 6/17 contours)'],
  ['   permuted Part-1 weights (median)', '0.21–0.72', '0.22–0.81; one draw reproduces u63 at 3 GeV (p = 0.022)'],
  ['context g: half-plane variance via g', 'u30 31.8 %; u31, u58 100 %', 'u30 32.3 %; u31, u58 100 %'],
  ['odd part, maps ÷ measured (g frozen), δ 0.03', '0.97 / 0.93 / 0.97 (η / φ / diag)', '0.99 [0.97, 1.01] / 0.96 [0.93, 0.99] / 0.98 [0.96, 0.99]'],
  ['odd part, maps ÷ measured (g frozen), δ 0.06', '1.01 / 1.02 / 1.03', '1.00 [0.99, 1.02] / 1.05 [1.03, 1.08] / 1.03 [1.02, 1.05]'],
  ['share of the odd part through g (1 − frozen/free)', '9–20 %', '15–24 % (e.g. δ 0.06 η: frozen/free 0.76 [0.74, 0.78])'],
  ['sample-asymmetry term B (reflected sample), δ 0.03', '—', '−0.0004 [−0.0011, +0.0002], p 0.20–0.22 (H0: B = 0 not rejected)'],
  ['odd part, point-symmetrized maps', '≤ 0.0008 (hybrid construction)', '+0.0000 [−0.0005, +0.0005]; even part 0.101 (unsymmetrized 0.101–0.103)']],
 [4.6, 3.4, 6.0],
 note='Brackets: 95 % bootstrap intervals over jets (1,000 resamples). E14\'s hybrid counterfactual (S_own + ½[ΔS(x′) + ΔS(−x′)]) gives negative S entries in 6–24 % of jets; through the fixed-point head (unsigned d4 inputs)\n'
      'it flips sign (+0.078/+0.052/+0.093); the construction with base and edited S both symmetrized gives 0 in both heads. Permutation p values: 25–163 complete draws of 200, so p resolution 0.006–0.04; no Holm correction over 51 tests.')
table('T20_e15_calibration.png', 'Does the read-out β separate a sum from a product of M0 and M2? Same ridge procedure on unit populations with a known rule (E15 verification; 9,639 jets)',
 ['population (64 units from the true moments, or the network)', 'A2 β of m̂ (GeV)\n(E15: sum 0, product 1)', 'M0-only response ÷ pT×c response\nlow / mid / high M2 third'],
 [['sum λM0 + M2, held linearly', '0.00', '0.754 / 0.496 / 0.294'],
  ['product √(M0·M2)', '1.00', '0.500 / 0.500 / 0.500'],
  ['ReLU windows on λM0 + M2', '−0.73, −0.76, −0.77 (3 seeds)', '0.783 / 0.487 / 0.296'],
  ['ReLU windows on √(M0·M2)', '1.00, 1.00, 1.00', '0.500 / 0.500 / 0.499'],
  ['ReLU on random directions in (z(M0), z(M2))', '1.15, 0.90, 1.04', '0.22–0.59 / 0.42–0.47 / 0.41–0.53'],
  ['random untrained ReLU layer on the trained S', '−0.01, 0.42, 1.14', '0.49–0.66 / 0.44–0.48 / 0.46–0.50'],
  ['network S / h4 / h5 / z', '0.42 / 0.70 / 0.53 / 0.32\n(E15: 0.33 / 0.70 / 0.60 / 0.32)', 'S 0.62/0.58/0.55; h4 0.63/0.44/0.47; z 0.65/0.54/0.53'],
  ['network h4: read-out of V_sum / of V_prod instead of log m', '0.03 / 0.71 (R² 0.995 / 0.991)', '—']],
 [4.6, 3.6, 3.8],
 note='The A2 β follows the read-out target and generic ReLU nonlinearity on S, so the 0.6–0.7 at h4/h5 does not place the combination between sum and product. The gain-normalised M0 response separates the known rules;\n'
      'the network matches the product within 0.06 in the high-M2 third (M2 > 8.1 GeV) from S onward. Matched M0×c vs M2×c steps (read-out-free, product d = 0, sum +0.47/0/−0.48): h4 median d +0.61/+0.15/+0.02, IQR width ≈ 1.5.')
