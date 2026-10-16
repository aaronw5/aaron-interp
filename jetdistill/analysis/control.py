"""Untrained-network control: does step 1 describe any network this well, or only the trained one?

The control network has fresh random weights at each layer's trained scale (kernel, bias and batch-norm scale redrawn
from a Gaussian with that weight's trained mean and spread) and the trained number formats. Step 1 runs on it with the
same observables as the setup 'all'. Per neuron that varies:
  R² on the test jets, R² along the forward selection (validation jets), which kinds of observables carry it (term
  importance by category; the share on orientation-only observables, Δη or Δφ alone), how well it separates one class from the rest (best AUC), how much of it the jet mass alone
  explains (R² of its mean in 1-GeV mass bins).
Two further checks, both fitted on training jets and scored on test jets:
  learnable  boosted trees on the raw particle inputs (no observables): can anything describe the neuron?
  lowlevel   step 1 with only the raw inputs (pT, Δη, Δφ of the particles) as observables
Output: results/control/n<N>/{step1_untrained.json, step1_lowlevel[_untrained].json, compare.json, learnable.json}
Usage: python -m jetdistill.analysis.control <N> [step1|lowlevel|compare|learnable ...]"""
import json, re, sys
import numpy as np
from .. import data, mars
from ..config import RESULTS
from ..network import Network
from ..observables import compute

KS = [1, 2, 3, 5, 10, 18, 30, 45, 60, 100]
CATS = ['mass', 'prong structure', 'width and shape', 'pT and multiplicity', 'single particles']
ORIENTATION = re.compile(r'^(mean_eta2?|mean_phi2?|eta_\d+|phi_\d+|abseta_\d+|absphi_\d+|orientation_deg)$')   # depend on Δη or Δφ alone


def orientation_share(neurons):
    """share of the formula's term importance on observables that depend on the jet's orientation (Δη or Δφ alone), not on
    distances; a random network's neurons are mostly fitted by these"""
    tot = d = 0.0
    for nr in neurons:
        for t in nr['terms']:
            qs = [q for q in (t['q'], t.get('q2')) if q]; w = abs(t['importance']); tot += w; d += w * sum(bool(ORIENTATION.match(q)) for q in qs) / len(qs)
    return d / max(tot, 1e-12)


def category(q):
    if 'mass' in q or q in ('m01', 'm012') or q.startswith('mratio') or 'over_m' in q: return 'mass'
    if re.match(r'(tau|C2|D2|C3|D3|N2|N3|M2|M3|e2|e3|e4|sj[23]_|sd_zg|sd_rg|sd_nremoved|dr01|dr02|dr12|dr_m)', q): return 'prong structure'
    if re.match(r'(pt|eta|phi|dr|z|abseta|absphi|zdr|dr0|dr1|ptdr0)_\d+$', q) or q.startswith('soft'): return 'single particles'
    if re.match(r'(girth|width|lam|eccentricity|planar|LHA|max_dr|centroid|psi_|n_dr_|z_dr_|mean_eta|mean_phi|orientation)', q): return 'width and shape'
    return 'pT and multiplicity'


def out_dir(n): d = RESULTS / 'control' / f'n{n}'; d.mkdir(parents=True, exist_ok=True); return d


def step1(n, lowlevel=False):
    for untrained in ((False, True) if lowlevel else (True,)):
        name = 'step1' + ('_lowlevel' if lowlevel else '') + ('_untrained' if untrained else '')
        mars.run('all', n, untrained=untrained, lowlevel=lowlevel, out=out_dir(n) / f'{name}.json')


def summary(neurons):
    out = []
    for nr in neurons:
        if not nr['terms']: continue
        imp = {c: 0.0 for c in CATS}
        for t in nr['terms']:
            for q in (t['q'], t.get('q2')):
                if q: imp[category(q)] += abs(t['importance']) / (2 if t.get('q2') else 1)
        tot = sum(imp.values()) or 1; p = nr['dev_r2_path']
        out.append(dict(neuron=nr['neuron'], r2_test=nr['test_r2'], terms=len(nr['terms']), r2_at={k: p[min(k, len(p)) - 1] for k in KS}, cats={c: v / tot for c, v in imp.items()}))
    return out


def per_neuron(H, y, m):
    from sklearn.metrics import roc_auc_score
    out = {}; b = np.clip(m.astype(int), 0, 400)
    for j in range(H.shape[1]):
        h = H[:, j]
        if h.std() == 0: continue
        prof = np.bincount(b, h, 401) / np.maximum(np.bincount(b, minlength=401), 1)
        out[j] = dict(auc=float(max(max(a, 1 - a) for a in (roc_auc_score(y == c, h) for c in range(5)))),
                      mass_r2=float(1 - ((h - prof[b]) ** 2).sum() / ((h - h.mean()) ** 2).sum()))
    return out


def compare(n):
    x = data.particles('test', n); y = data.labels('test'); m = compute(x, n, ['mass'])['mass']; res = {}
    for lab, f, un in (('trained', RESULTS / 'all' / f'n{n}' / 'step1.json', False), ('untrained', out_dir(n) / 'step1_untrained.json', True)):
        N = summary(json.loads(f.read_text())['neurons']); P = per_neuron(np.maximum(Network(n, untrained=un).run(x)['z'], 0), y, m)
        for nr in N: nr.update(P.get(nr['neuron'], {}))
        res[lab] = N; res[f'{lab}_orientation_share'] = orientation_share(json.loads(f.read_text())['neurons'])
        print(lab, 'median R² on test jets', round(float(np.median([r['r2_test'] for r in N])), 3), flush=True)
    (out_dir(n) / 'compare.json').write_text(json.dumps(res, indent=1))


def learnable(n, n_fit=40000):
    from sklearn.ensemble import HistGradientBoostingRegressor
    feats = lambda x: np.concatenate([x.reshape(len(x), -1), np.log1p(x[..., 0]), np.hypot(x[..., 1], x[..., 2])], 1)
    xf, xt = data.particles('fit', n, stop=n_fit), data.particles('test', n); res = {}
    for lab, un in (('trained', False), ('untrained', True)):
        net = Network(n, untrained=un); Hf, Ht = (np.maximum(net.run(x)['z'], 0) for x in (xf, xt)); r = []
        for j in range(Ht.shape[1]):
            if Hf[:, j].std() == 0 or Ht[:, j].std() == 0: continue
            p = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.1, max_leaf_nodes=63, random_state=0).fit(feats(xf), Hf[:, j]).predict(feats(xt))
            r.append(dict(neuron=j, r2=float(1 - ((Ht[:, j] - p) ** 2).sum() / ((Ht[:, j] - Ht[:, j].mean()) ** 2).sum()), distinct_values=int(len(np.unique(Ht[:, j])))))
        res[lab] = r; print(lab, 'median R² (boosted trees, raw inputs)', round(float(np.median([q['r2'] for q in r])), 3), flush=True)
    (out_dir(n) / 'learnable.json').write_text(json.dumps(res, indent=1))


def numbers(n):
    """the control's summary for one network (None until every part exists): median over neurons of the test-jet R² of step 1
    on the physics observables, of step 1 on the raw inputs and of boosted trees on the raw inputs, and the orientation share
    of the step-1 formula; each as {'trained': x, 'untrained': x}"""
    d = RESULTS / 'control' / f'n{n}'
    if not all((d / f).exists() for f in ('compare.json', 'learnable.json', 'step1_lowlevel.json', 'step1_lowlevel_untrained.json')): return None
    C = json.loads((d / 'compare.json').read_text()); L = json.loads((d / 'learnable.json').read_text()); med = lambda v: float(np.median(v))
    low = lambda sfx: med([x['test_r2'] for x in json.loads((d / f'step1_lowlevel{sfx}.json').read_text())['neurons'] if x['terms']])
    return dict(physics={k: med([r['r2_test'] for r in C[k]]) for k in ('trained', 'untrained')},
                orientation={k: C.get(f'{k}_orientation_share') for k in ('trained', 'untrained')},
                raw_inputs=dict(trained=low(''), untrained=low('_untrained')), trees={k: med([r['r2'] for r in L[k]]) for k in ('trained', 'untrained')})


if __name__ == '__main__':
    n, *todo = sys.argv[1:]; n = int(n)
    for s in todo or ['step1', 'lowlevel', 'compare', 'learnable']:
        dict(step1=lambda: step1(n), lowlevel=lambda: step1(n, lowlevel=True), compare=lambda: compare(n), learnable=lambda: learnable(n))[s]()
