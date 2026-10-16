import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import os, pathlib
S = str(pathlib.Path(__file__).resolve().parents[1] / 'results' / 'site') + '/'; os.makedirs(S, exist_ok=True)
def table(fname, title, cols, rows, widths, note=None, h=0.42):
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
