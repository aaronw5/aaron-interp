"""E7v (skeptic of E7): rerun E7's code on a fresh sample (fit seed 7, test seed 71) for the S, hand and randS bases at K in {8, 32},
plus three checks E7 did not run:
  (1) hand_nomass: the hand basis without the invariant-mass family (every id containing 'mass', m01, m012, mratio_*), so the fit
      has to use the axis-relative observables it already contains (girth, girth2, LHA, width, lam1/2, n_dr_*, z_dr_*, psi_*).
      Question: is the hand formula's translation failure a property of observables in general, or of the fit picking the
      invariant mass among correlated alternatives?
  (2) extrapolation: share of translated jets whose S values leave the fit range, and the S formula's translation slope on jets
      inside vs outside that range.
  (3) slope restricted to the (jet, neuron) entries where the network's max(0, z) changes; W->Z of randS vs S (is W->Z inherited
      from the S basis whatever the formula?); identity check: substituting the network's own z must change 0 calls.
The E7 source is executed unchanged except for the constants below (asserted replacements), so the reproduction uses E7's code.
Outputs: $AI_RESULTS/jedi/e7_verify_*.npz/json/png and e7_verify.json.
  python -m interp.e7_verify [--smoke]
"""
import sys, json, re, time
from pathlib import Path
import numpy as np

SRC = Path(__file__).resolve().parent / 'e7_z_objects_jedi.py'
smoke = '--smoke' in sys.argv
sys.argv = [str(SRC)] + (['--smoke'] if smoke else [])
src = SRC.read_text()
REPL = [
    ("SEED, SEED_T = 2, 43", "SEED, SEED_T = 7, 71"),
    ("TAG = 'e7s' if SMOKE else 'e7'", "TAG = 'e7_verify_smoke' if SMOKE else 'e7_verify'"),
    ("KS = (4, 8, 16, 32)", "KS = (8, 32)"),
    ("N_RAND = 3 if SMOKE else 10; N_RAND_Z = 3 if SMOKE else 5", "N_RAND = 2 if SMOKE else 3; N_RAND_Z = N_RAND"),
    ("    if b in ('hand', 'hand_all'):\n        hk = hand_keys(b == 'hand_all'); res['observables'] = hk",
     "    if b in ('hand', 'hand_all', 'hand_nomass'):\n        hk = hand_keys(b == 'hand_all')\n"
     "        if b == 'hand_nomass': hk = [k for k in hk if not ('mass' in k or k in ('m01', 'm012') or k.startswith('mratio'))]\n"
     "        res['observables'] = hk"),
    ("for b in ('S', 'SP', 'hand', 'hand_all', 'initS', 'randS'):", "for b in ('S', 'SP', 'hand', 'hand_all', 'hand_nomass', 'initS', 'randS'):"),
    ("for other in ('hand', 'hand_all', 'SP', 'initS'):", "for other in ('hand', 'hand_all', 'hand_nomass', 'SP', 'initS'):"),
    ("hand_all='#f0a35e',", "hand_all='#f0a35e', hand_nomass='#a0522d',"),
    ("hand_all='all observables',", "hand_all='all observables', hand_nomass='hand observables without mass',"),
    ("for b in ('S', 'SP', 'hand', 'hand_all', 'initS') if b in out['substitution']]",
     "for b in ('S', 'SP', 'hand', 'hand_all', 'hand_nomass', 'initS') if b in out['substitution']]"),
]
for a, b in REPL:
    assert src.count(a) == 1, a
    src = src.replace(a, b)
main = "if ARGS.combine:\n    combine()\nelse:"
assert src.count(main) == 1
src = src[:src.index(main)]
G = {'__name__': 'e7_verify_exec', '__file__': str(SRC)}
exec(compile(src, str(SRC), 'exec'), G)
log, OUT, TAG = G['log'], G['OUT'], G['TAG']
log(f'E7v: seeds fit {G["SEED"]}, test {G["SEED_T"]}; KS {G["KS"]}; TAG {TAG}')

# ---- identity check: the network's own z through the substitution path changes no call
xt, Rt, ct = G['xt'], G['Rt'], G['ct']
cid = G['run_override'](xt, {'z': (np.ones(16), np.maximum(Rt['z'], 0))}).argmax(1)
ident = float((cid != ct).mean() * 100); log(f'identity check: own z substituted -> {ident:.4f} % calls changed')
# invariant mass before / after translation (hand observables recomputed)
from jetdistill.observables import compute
Je, xb, xa = G['EDITS']['trans']
mb = compute(np.asarray(xb[:2000], np.float64), 64, ids=['mass', 'girth2'])
ma = compute(np.asarray(xa[:2000], np.float64), 64, ids=['mass', 'girth2'])
dmass = float(np.nanmax(np.abs(ma['mass'] - mb['mass']) / np.maximum(mb['mass'], 1e-9)))
dg2 = float(np.median(ma['girth2'] / np.maximum(mb['girth2'], 1e-12)))
log(f'translation: max rel change of invariant mass {dmass:.2e}; median girth2 after/before {dg2:.3f}')

bases = ['S', 'hand', 'hand_nomass', 'randS']
for b in bases:
    if (OUT / f'{TAG}_{b}.npz').exists() and (OUT / f'{TAG}_{b}.json').exists(): log(f'{b}: saved arrays exist, not rerun'); continue
    G['run_basis'](b)
G['combine']()

# ---- extra checks on the saved arrays
V = dict(seeds=dict(fit=G['SEED'], test=G['SEED_T']), N_test=int(G['JT']), identity_calls_changed_pct=ident,
         translation_max_rel_dmass=dmass, translation_girth2_ratio_median=dg2)
SDH, liveS, fit, R, RN, yt = G['SDH'], G['liveS'], G['fit'], G['R'], G['RN'], G['yt']
rate, slope_corr, ci, p_boot = G['rate'], G['slope_corr'], G['ci'], G['p_boot']
A = {b: dict(np.load(OUT / f'{TAG}_{b}.npz')) for b in bases}
dn = (np.maximum(RN['trans'][1]['z'], 0) - np.maximum(Rt['z'], 0)) / SDH
# (2) extrapolation of the S basis under translation
Sf = R['S'][fit][:, liveS]; Sa = RN['trans'][1]['S'][:, liveS]; St = Rt['S'][:, liveS]
lo, hi = np.percentile(Sf, 0.5, 0), np.percentile(Sf, 99.5, 0); mn, mx = Sf.min(0), Sf.max(0)
out_t = ((St < lo) | (St > hi)).any(1); out_a = ((Sa < lo) | (Sa > hi)).any(1); out_a_mm = ((Sa < mn) | (Sa > mx)).any(1)
V['extrapolation'] = dict(share_test_outside_p0p5_p99p5=float(out_t.mean() * 100), share_trans_outside_p0p5_p99p5=float(out_a.mean() * 100),
                          share_trans_outside_minmax=float(out_a_mm.mean() * 100),
                          per_column_share_trans_outside=[float(v * 100) for v in ((Sa < lo) | (Sa > hi)).mean(0)], columns=[f'S{k}' for k in liveS])
for K in (8, 32):
    df = (np.maximum(A['S'][f'zh_K{K}_trans_a'].astype(float), 0) - np.maximum(A['S'][f'zh_K{K}_test'].astype(float), 0)) / SDH
    row = {}
    for nm, m in (('inside', ~(out_a | out_t)), ('outside', out_a | out_t)):
        s0, sb, r0, rb, _ = slope_corr(dn[m], df[m]); row[nm] = dict(N=int(m.sum()), slope=ci(s0, sb), r=float(r0))
    # (3) entries where the network's relu(z) changes
    ch = dn != 0; x_, y_ = dn[ch], df[ch]
    row['changed_entries'] = dict(N=int(ch.sum()), slope=float(np.cov(x_, y_)[0, 1] / x_.var()), r=float(np.corrcoef(x_, y_)[0, 1]))
    V[f'S_trans_slope_K{K}'] = row
    log(f'S K={K} translation slope inside fit range {row["inside"]["slope"]["v"]:.3f} (N {row["inside"]["N"]}), outside '
        f'{row["outside"]["slope"]["v"]:.3f} (N {row["outside"]["N"]}); changed entries only {row["changed_entries"]["slope"]:.3f}')
for b in ('hand', 'hand_nomass'):
    for K in (8, 32):
        df = (np.maximum(A[b][f'zh_K{K}_trans_a'].astype(float), 0) - np.maximum(A[b][f'zh_K{K}_test'].astype(float), 0)) / SDH
        ch = dn != 0; V[f'{b}_trans_slope_changed_entries_K{K}'] = float(np.cov(dn[ch], df[ch])[0, 1] / dn[ch].var())
# randS vs S W->Z (inheritance from the S basis)
isW = yt == G['Wc']; Zc = G['Zc']
V['WtoZ_trans'] = {'network_before': float((ct[isW] == Zc).mean() * 100),
                   'network_after': float((RN['trans'][1]['logits'].argmax(1)[isW] == Zc).mean() * 100)}
for b in bases:
    for K in (8, 32):
        if b == 'randS':
            V['WtoZ_trans'][f'randS_K{K}'] = [dict(before=float((A[b][f'call_d{d}_K{K}_test'][isW] == Zc).mean() * 100),
                                                   after=float((A[b][f'call_d{d}_K{K}_trans_a'][isW] == Zc).mean() * 100)) for d in range(G['N_RAND'])]
        else:
            V['WtoZ_trans'][f'{b}_K{K}'] = dict(before=float((A[b][f'call_K{K}_test'][isW] == Zc).mean() * 100),
                                               after=float((A[b][f'call_K{K}_trans_a'][isW] == Zc).mean() * 100))
# observables the hand formulas use (contribution-weighted)
for b in ('hand', 'hand_nomass'):
    fj = json.loads((OUT / f'{TAG}_{b}.json').read_text()); w = {}
    for f in fj['formulas']['32']:
        for t in f['terms']:
            for o in set(re.findall(r'[A-Za-z_][A-Za-z0-9_]*', t[0])) - {'max'}:
                w[o] = w.get(o, 0) + t[2]
    tot = sum(w.values()); V[f'{b}_K32_observable_weight_share'] = {k: round(v / tot, 4) for k, v in sorted(w.items(), key=lambda r: -r[1])[:20]}
    log(f'{b} K=32 observables by weight: ' + ', '.join(f'{k} {v:.2f}' for k, v in list(V[f"{b}_K32_observable_weight_share"].items())[:10]))
log('WtoZ: ' + json.dumps(V['WtoZ_trans']))
(OUT / f'{TAG}.json').write_text(json.dumps(V, indent=1))
log(f'wrote {OUT / f"{TAG}.json"}')
log('done')
