"""E7v add-on: do the formula's call changes under an edit fall on the same jets as the network's?  For each edit and basis (K = 32):
precision P(network call changes | formula call changes), recall P(formula changes | network changes), same new class among jets where
both change, with 95 % bootstrap intervals over jets; randS draws as the control. Reads E7's saved per-jet calls; reruns only the
network (E7's setup code executed with E7's or E7v's constants).
  python -m interp.e7_verify_flips --run e7          (E7's samples, seeds 2 / 43, arrays e7_*.npz)
  python -m interp.e7_verify_flips --run e7_verify   (E7v's samples, seeds 7 / 71, arrays e7_verify_*.npz)
Output: $AI_RESULTS/jedi/e7_verify_flips_<run>.json
"""
import sys, json
from pathlib import Path
import numpy as np

run = sys.argv[sys.argv.index('--run') + 1]
SRC = Path(__file__).resolve().parent / 'e7_z_objects_jedi.py'
sys.argv = [str(SRC)]
src = SRC.read_text()
if run == 'e7_verify':
    for a, b in (("SEED, SEED_T = 2, 43", "SEED, SEED_T = 7, 71"), ("TAG = 'e7s' if SMOKE else 'e7'", "TAG = 'e7s' if SMOKE else 'e7_verify'")):
        assert src.count(a) == 1; src = src.replace(a, b)
main = "if ARGS.combine:\n    combine()\nelse:"
src = src[:src.index(main)]
G = {'__name__': 'e7_flips_exec', '__file__': str(SRC)}
exec(compile(src, str(SRC), 'exec'), G)
log, OUT, TAG, RN, ct, bw = G['log'], G['OUT'], G['TAG'], G['RN'], G['ct'], G['bw']
NB = {'trans_b': ct, 'rot_b': ct}
for k in G['EDITS']:
    NB[f'{k}_a'] = RN[k][1]['logits'].argmax(1)
    if k not in ('trans', 'rot'): NB[f'{k}_b'] = RN[k][0]['logits'].argmax(1)


def stats(fn, ff, cn, cf):
    W = bw(len(fn)); fn, ff = fn.astype(float), ff.astype(float); both = fn * ff; same = both * (cn == cf)
    def q(num, den):
        return float(num.sum() / max(den.sum(), 1e-9)), W @ num / np.maximum(W @ den, 1e-9)
    out = {}
    for nm, (v, b) in dict(precision=q(both, ff), recall=q(both, fn), same_class_if_both=q(same, both)).items():
        out[nm] = dict(v=v * 100, lo=float(np.percentile(b, 2.5) * 100), hi=float(np.percentile(b, 97.5) * 100))
    out['n_net'] = int(fn.sum()); out['n_formula'] = int(ff.sum()); out['n_both'] = int(both.sum())
    out['chance_precision'] = float(fn.mean() * 100)
    return out


res = dict(run=run, seeds=dict(fit=G['SEED'], test=G['SEED_T']), N_test=int(G['JT']))
for b in ('S', 'hand', 'hand_nomass', 'randS'):
    f = OUT / f'{TAG}_{b}.npz'
    if not f.exists(): continue
    A = np.load(f); res[b] = {}
    for k in G['EDITS']:
        sb, sa = (('test', f'{k}_a') if k in ('trans', 'rot') else (f'{k}_b', f'{k}_a'))
        cnb, cna = NB[f'{k}_b'], NB[f'{k}_a']; fn = cna != cnb
        if b == 'randS':
            nd = len([key for key in A.files if key.startswith('call_d') and key.endswith('_K32_test')])
            res[b][k] = [stats(fn, A[f'call_d{d}_K32_{sa}'] != A[f'call_d{d}_K32_{sb}'], cna, A[f'call_d{d}_K32_{sa}']) for d in range(nd)]
            log(f'{b} {k}: precision ' + ', '.join(f'{r["precision"]["v"]:.1f}' for r in res[b][k]) + '; recall ' +
                ', '.join(f'{r["recall"]["v"]:.1f}' for r in res[b][k]))
        else:
            r = stats(fn, A[f'call_K32_{sa}'] != A[f'call_K32_{sb}'], cna, A[f'call_K32_{sa}']); res[b][k] = r
            log(f'{b} {k}: network flips {r["n_net"]}, formula {r["n_formula"]}, both {r["n_both"]}; precision {r["precision"]["v"]:.1f} '
                f'[{r["precision"]["lo"]:.1f}, {r["precision"]["hi"]:.1f}] (chance {r["chance_precision"]:.1f}); recall {r["recall"]["v"]:.1f} '
                f'[{r["recall"]["lo"]:.1f}, {r["recall"]["hi"]:.1f}]; same new class {r["same_class_if_both"]["v"]:.1f}')
(OUT / f'e7_verify_flips_{run}.json').write_text(json.dumps(res, indent=1))
log('done')
