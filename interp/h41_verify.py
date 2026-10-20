"""H41 verification (loop D): fresh-sample rerun of the colour-flow mirror test plus two missing controls.

Fresh sample: JetClass test WToQQ_100, ZToQQ_100, HToBB_100, ZJetsToNuNu_100, 500 < pT < 1000 GeV, |eta| < 2, >= 3 particles;
  PER random rows per class (default_rng(4301 + k)); disjoint seeds from H41 (4101-4104). Geometry, soft definition and the
  M / C1 / C2 edits are imported unchanged from interp.h41_colour_flow.
Control 1 (shape confound). M moves between-prong soft particles away from the jet axis and the other prong, so it also changes
  girth (sum z_i dR_i,axis), D2 and C2 (beta 1 energy correlation ratios) and tau32. Added: (a) these observables for every
  variant; (b) sets matched on girth and D2 (relative tolerance r) on top of the loose m / tau21 match; (c) references
  GBS = HGB on (m, tau21, N, girth, D2, C2, tau32) and GBS5 = GBS + (theta1, theta2, f_btw), next to GB3 / GB5; (d) the all-jet
  intercept at zero (dm, dtau21, dgirth, dD2).
Control 2 (signed response vs generic disruption). M_in mirrors only the soft particles on the near side of their prong
  (u_rel > 0, toward the other prong) to the far side; M_out mirrors only those with u_rel < 0 onto the near side. C1_in / C1_out
  apply the across-the-line mirror to the same particles. Colour-flow and radial readings predict M_in - C1_in < 0 and
  M_out - C1_out > 0 on signal jets; generic sensitivity to moved particles predicts both < 0.
Run: ./part_job.sh d-h41-verify 'python -m interp.h41_verify'   (H41V_SMOKE=1: 150 per class)
Output: $AI_RESULTS/physics/h41_verify.json
"""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import h41_colour_flow as C     # noqa: E402
from interp import h4_top_wmass as H        # noqa: E402
from interp import h19_wpair_cross as X     # noqa: E402

try:
    from numba import njit, prange
except ImportError:
    njit = lambda *a, **k: (lambda f: f); prange = range

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
SMOKE = bool(os.environ.get('H41V_SMOKE'))
RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'physics'
TAG = 'h41_verify_smoke' if SMOKE else 'h41_verify'
PER = 150 if SMOKE else 3000
NREF = 800 if SMOKE else 10000
B = 300 if SMOKE else 2000
NPERM = 500 if SMOKE else 5000
SEED = 4301
SAMPLES, SIG, QCD = C.SAMPLES, C.SIG, C.QCD
NETS, NULLS = C.NETS, C.NULLS
REFN = ['GB3', 'GB5', 'GBS', 'GBS5']
FEAT = dict(GB3=['m', 'tau21', 'N'], GB5=['m', 'tau21', 'N', 'theta1', 'theta2', 'f_btw'],
            GBS=['m', 'tau21', 'N', 'girth', 'D2', 'C2', 'tau32'], GBS5=['m', 'tau21', 'N', 'girth', 'D2', 'C2', 'tau32', 'theta1', 'theta2', 'f_btw'])
OBSL = ['m', 'tau21', 'tau32', 'N', 'girth', 'D2', 'C2', 'theta1', 'theta2', 'f_btw']


@njit(parallel=True, cache=False)
def ecf_numba(F, nn):
    """e2, e3 (beta 1), rapidity-azimuth distances, z_i = pT_i / sum pT"""
    J = F.shape[0]; e2o = np.zeros(J); e3o = np.zeros(J)
    for j in prange(J):
        n = nn[j]
        if n < 3: continue
        pt = np.zeros(n); yy = np.zeros(n); ph = np.zeros(n)
        for i in range(n):
            px = F[j, i, 0]; py = F[j, i, 1]; pz = F[j, i, 2]; E = F[j, i, 3]
            pt[i] = np.hypot(px, py); ph[i] = np.arctan2(py, px); e = max(E, abs(pz) + 1e-9); yy[i] = 0.5 * np.log((e + pz) / (e - pz))
        z = pt / pt.sum(); D = np.zeros((n, n))
        for a in range(n):
            for b in range(a + 1, n):
                dphi = abs(ph[a] - ph[b])
                if dphi > np.pi: dphi = 2 * np.pi - dphi
                D[a, b] = np.sqrt((yy[a] - yy[b]) ** 2 + dphi ** 2); D[b, a] = D[a, b]
        e2 = 0.0; e3 = 0.0
        for a in range(n):
            for b in range(a + 1, n):
                w = z[a] * z[b] * D[a, b]; e2 += w
                for c in range(b + 1, n): e3 += w * z[c] * D[a, c] * D[b, c]
        e2o[j] = e2; e3o[j] = e3
    return e2o, e3o


def obs(P4, nn):
    from interp.h23_axis_audit import axis
    O, fr, G = C.observables(P4, nn, 'jc')
    F = np.ascontiguousarray(P4.astype(np.float64)); e2, e3 = ecf_numba(F, nn.astype(np.int64))
    O['D2'] = e3 / np.maximum(e2 ** 3, 1e-30); O['C2'] = e3 / np.maximum(e2 ** 2, 1e-30)
    k = C.pkin(P4, nn); ae, ap = axis(P4, nn)
    dr = np.hypot(k['eta'] - ae[:, None], C.wrap(k['phi'] - ap[:, None])); z = k['pt'] / np.maximum(k['pt'].sum(1, keepdims=True), 1e-12)
    O['girth'] = (z * dr * k['real']).sum(1)
    return O, fr, G


def variants(P4, nn, G, seed):
    V, info = C.make_variants(P4, nn, G, seed)
    k = G['k']; soft = G['soft']; e, u, v, ur, vr = G['e'], G['u'], G['v'], G['ur'], G['vr']
    eM = e - 2 * ur[..., None] * u; eC1 = e - 2 * vr[..., None] * v
    for side, sel in (('in', soft & (ur > 0)), ('out', soft & (ur < 0))):
        for nm, ee in (('M', eM), ('C1', eC1)):
            eta = np.where(sel, G['cen'][..., 0] + ee[..., 0], k['eta']); phi = np.where(sel, C.wrap(G['cen'][..., 1] + ee[..., 1]), k['phi'])
            V[f'{nm}_{side}'] = (eta, phi)
        info[f'frac_soft_pt_{side}'] = float(((k['pt'] * sel).sum(1) / np.maximum((k['pt'] * soft).sum(1), 1e-12)).mean())
    return V, info


def fit_refs(out):
    from interp.part_core import PRE
    from sklearn.ensemble import HistGradientBoostingClassifier
    OB = {}
    for kk, c in enumerate(SAMPLES):
        J, inf = C.read_window(C.TRAIN / f'{PRE[c]}_000.root', NREF, 4311 + kk); out.setdefault('ref_rows', {})[c] = inf
        P4, nn = X.jc_P4(J); P4 = H._identity(P4, nn).astype(np.float32); OB[c] = obs(P4, nn)[0]; log(f'ref obs {c} {len(nn)}')
    R, info = {}, {}
    for c in SIG:
        Os = {o: np.concatenate([OB[c][o], OB['QCD'][o]]) for o in OBSL}; y = np.r_[np.ones(len(OB[c]['m'])), np.zeros(len(OB['QCD']['m']))]
        tr = np.random.default_rng(4320).random(len(y)) < 0.8
        for nm in REFN:
            Xf = np.column_stack([Os[o] for o in FEAT[nm]]); mdl = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(Xf[tr], y[tr])
            pr = mdl.predict_proba(Xf[~tr])[:, 1]; R[(c, nm)] = mdl; info[f'{c}|{nm}'] = float(H._auc(pr, y[~tr]))
    log('ref held-out AUC', info)
    return R, info


def ref_scores(R, O):
    lo = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    return {(c, nm): lo(m.predict_proba(np.column_stack([O[o] for o in FEAT[nm]]))[:, 1]) for (c, nm), m in R.items()}


def main():
    from interp.part_core import cat, kin
    from interp.h23_axis_audit import axis
    RES.mkdir(parents=True, exist_ok=True)
    out = dict(node='H41_verify', smoke=SMOKE, per_class=PER, seed=SEED, B=B, n_perm=NPERM)
    Js = []
    for kk, c in enumerate(SAMPLES):
        J, inf = C.read_window(X.jc_file(c, 100), PER, SEED + kk); Js.append(J); out.setdefault('sample', {})[c] = inf
    J = cat(Js); lab = J['label'].astype(int); P4raw, nn = X.jc_P4(J); P4 = H._identity(P4raw, nn).astype(np.float32)
    J['pmass'] = kin(J)[3].astype(np.float32); cls = np.array([{QCD: 'QCD', C.HBB: 'Hbb', C.ZQQ: 'Zqq', C.WQQ: 'Wqq'}[l] for l in lab])
    O0, fr, G = obs(P4, nn); log('unedited obs')
    V, out['edit_info'] = variants(P4, nn, G, SEED + 10); k0 = G['k']; soft = G['soft']
    P = {'id': P4}
    for v in V: P[v] = C.to_p4(k0, *V[v])
    OBS = {'id': O0}
    for v in V: OBS[v] = obs(P[v], nn)[0]
    log('edited obs')
    out['obs_change'] = {c: {v: {o: float((OBS[v][o] - O0[o])[(cls == c)].mean()) for o in OBSL} for v in V} for c in SAMPLES}
    out['obs_id_means'] = {c: {o: float(O0[o][cls == c].mean()) for o in OBSL} for c in SAMPLES}
    for c in ('Wqq', 'QCD'):
        log(c, 'M:', {o: round(x, 4) for o, x in out['obs_change'][c]['M'].items()}); log(c, 'C1:', {o: round(x, 4) for o, x in out['obs_change'][c]['C1'].items()})
    has = soft.sum(1) > 0
    d = lambda v, o: np.abs(OBS[v][o] - O0[o])
    rel = lambda v, o: d(v, o) / np.maximum(np.abs(O0[o]), 1e-9)
    loose = lambda vs: has & np.all([(d(v, 'm') < 3) & (d(v, 'tau21') < 0.03) for v in vs], 0)
    strict = lambda vs: has & np.all([(d(v, 'm') < 1) & (d(v, 'tau21') < 0.01) for v in vs], 0)
    shape = lambda vs, r: loose(vs) & np.all([(rel(v, 'girth') < r) & (rel(v, 'D2') < r) for v in vs], 0)
    MC = ('M', 'C1', 'C2')
    SETS = {'matched': strict(MC), 'loose': loose(MC), 'all': has}
    for r in (0.02, 0.05, 0.10): SETS[f'loose_shape{r:g}'] = shape(MC, r)
    SETS['loose_shape0.05_tau32'] = SETS['loose_shape0.05'] & np.all([d(v, 'tau32') < 0.03 for v in MC], 0)
    for side in ('in', 'out'):
        SETS[f'loose_{side}'] = loose((f'M_{side}', f'C1_{side}')); SETS[f'loose_shape0.05_{side}'] = shape((f'M_{side}', f'C1_{side}'), 0.05)
        SETS[f'all_{side}'] = has
    out['sets'] = {s: {c: int((m & (cls == c)).sum()) for c in SAMPLES} for s, m in SETS.items()}; log('sets', out['sets'])
    # networks
    nets = C.JCNets(); LG = {v: nets.logits(Pv, nn, J, axis(Pv, nn)) for v, Pv in P.items()}; log('logits done')
    out['unedited'] = {}
    for m in NETS + NULLS:
        L = LG['id'][m]; dd = dict(acc_10class=float((L.argmax(1) == lab).mean()))
        for c, ci in SIG.items():
            sel = (cls == c) | (cls == 'QCD'); dd[f'AUC_{c}'] = float(H._auc(L[sel, ci] - L[sel, QCD], (cls[sel] == c).astype(int)))
        out['unedited'][m] = dd
    log('unedited', out['unedited'])
    REF, out['ref_auc'] = fit_refs(out); RS = {v: ref_scores(REF, OBS[v]) for v in OBS}

    def score(model, v, c):
        if model in NETS + NULLS: L = LG[v][model]; return L[:, SIG[c]] - L[:, QCD]
        return RS[v][(c, model)]

    rng = np.random.default_rng(SEED + 20); res = {}; fam = {}
    CON = {'M-C1': ('M', 'C1'), 'M-C2': ('M', 'C2'), 'M': ('M', None), 'C1': ('C1', None), 'C2': ('C2', None),
           'Min-C1in': ('M_in', 'C1_in'), 'Mout-C1out': ('M_out', 'C1_out'), 'M_in': ('M_in', None), 'M_out': ('M_out', None)}
    for sname, smask in SETS.items():
        cons = ['Min-C1in', 'M_in'] if sname.endswith('_in') else ['Mout-C1out', 'M_out'] if sname.endswith('_out') else ['M-C1', 'M-C2', 'M', 'C1', 'C2']
        for samp in SAMPLES:
            sel = smask & (cls == samp)
            if sel.sum() < 5: continue
            for c in ([samp] if samp != 'QCD' else list(SIG)):
                for model in NETS + NULLS + REFN:
                    for cn in cons:
                        a, b = CON[cn]; x = (score(model, a, c) - (score(model, b, c) if b else score(model, 'id', c)))[sel]
                        mu, ci = C.boot(x, rng, B); p = C.perm2(x, rng, NPERM); key = f'{sname}|{samp}|{c}|{model}|{cn}'
                        res[key] = dict(mean=mu, ci=ci, p_perm=p, n=int(sel.sum()))
                        if model in NETS and samp != 'QCD' and '-' in cn and sname in ('loose_shape0.05', 'loose_shape0.05_in', 'loose_shape0.05_out', 'matched', 'loose'):
                            fam.setdefault(sname, []).append(key)
                        if sname.startswith('all') and b is not None:
                            Z = np.column_stack([(OBS[a][o] - OBS[b][o])[sel] for o in ('m', 'tau21', 'girth', 'D2')])
                            res[key + '|intercept4'] = C.intercept(x, Z, rng, B)
                            Z2 = Z[:, :2]; res[key + '|intercept2'] = C.intercept(x, Z2, rng, B)
    out['holm'] = {}
    for f, lst in fam.items():
        adj = C.holm([res[t]['p_perm'] for t in lst])
        for t, a in zip(lst, adj): res[t]['p_holm'] = float(a)
        out['holm'][f] = dict(n_tests=len(lst), n_lt_0p05=int((adj < 0.05).sum()))
    out['results'] = res
    # concise log table
    for sname in ('matched', 'loose', 'loose_shape0.1', 'loose_shape0.05', 'loose_shape0.02', 'loose_shape0.05_tau32', 'all',
                  'loose_in', 'loose_out', 'loose_shape0.05_in', 'loose_shape0.05_out', 'all_in', 'all_out'):
        cn = 'Min-C1in' if sname.endswith('_in') else 'Mout-C1out' if sname.endswith('_out') else 'M-C1'
        for samp in SAMPLES:
            for c in ([samp] if samp != 'QCD' else ['Wqq']):
                row = []
                for model in NETS + NULLS + REFN:
                    r = res.get(f'{sname}|{samp}|{c}|{model}|{cn}')
                    if r: row.append(f"{model} {r['mean']:+.3f}[{r['ci'][0]:+.2f},{r['ci'][1]:+.2f}]" + (f" h{r['p_holm']:.2g}" if 'p_holm' in r else ''))
                    i4 = res.get(f'{sname}|{samp}|{c}|{model}|{cn}|intercept4')
                    if i4: row.append(f"int4 {i4['intercept']:+.3f}[{i4['ci'][0]:+.2f},{i4['ci'][1]:+.2f}]")
                if row: log(f'{sname:22s} {samp:4s} s_{c} {cn}: n {res[f"{sname}|{samp}|{c}|ParT_full|{cn}"]["n"]} | ' + ' | '.join(row))
    (RES / f'{TAG}.json').write_text(json.dumps(C.clean(out), indent=1)); log('saved', RES / f'{TAG}.json')


if __name__ == '__main__':
    main()
