"""H35 verify (loop D): fresh-sample rerun of the b-prong localisation edits plus three missing controls.

Sample: JetClass test TTBar_101 / ZJetsToNuNu_101 (H35 used _100), 550 < jet_pt < 650 GeV, |jet_eta| < 2, >= 3 particles,
  2,000 random rows per class (rng 3551 / 3552). Roles, counts and the H35 edits M, C1, C2 come from interp.h35_top_bprong.
New edits on top jets (only (d0val, d0err, dzval, dzerr) are exchanged; kinematics fixed; jet displaced count and S multiset kept):
  Mk   count-matched M: the same k1 sources as C1 (the k1 highest-pT displaced tracks of c) go to the k1 highest-pT prompt tracks
       of the W prong w (k1 = C1's swap size).
  Mpt  count- and pT-matched M: for each C1 target (prompt track of c, rank i) the prompt track of w with the nearest log pT
       (greedy, without replacement) receives source i. Mpt - C1 differs only in which prong the receiving tracks are on.
  RV   reverse move: wd = the W prong with more displaced tracks (tie: a); kr = min(n_disp(wd), n_prompt(c), n_prompt(wd)); the kr
       highest-pT displaced tracks of wd exchange with the kr highest-pT prompt tracks of c.
  CW   control for RV: the same kr displaced tracks of wd exchange with the kr highest-pT prompt tracks of wd.
Predictions: localisation to the mass-assigned b-prong (H1): Mpt - C1 < 0 and RV - CW > 0. Generic cross-prong sensitivity
  (H2: any move between prongs lowers the score): Mpt - C1 < 0 and RV - CW < 0. Count only / pT-tuple mismatch (H0): Mpt - C1 = 0.
Models: ParT_full, PN_full, ParT_perm (null), ParT_kin (zero check). Metric: change in top log-odds (10-class).
Run: ./part_job.sh d-h35-verify 'python -m interp.h35_verify'
Output: $AI_RESULTS/physics/h35_verify.json
"""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import h4_top_wmass as H          # noqa: E402
from interp import h19_wpair_cross as X       # noqa: E402
from interp import h35_top_bprong as T        # noqa: E402

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'physics'
PER = int(os.environ.get('H35V_PER', 2000))
B, NPERM = 2000, 5000
IPK = T.IPK
NEW = ('Mk', 'Mpt', 'RV', 'CW', 'RVpt')
TAG = os.environ.get('H35V_TAG', 'h35_verify')
NETS = ('ParT_full', 'PN_full', 'ParT_perm', 'ParT_kin')


def new_edits(J, LAB, R, y):
    N = len(y); ch, S = T.sig(J); pt = np.hypot(J['px'], J['py'])
    E = {v: {k: J[k].copy() for k in IPK} for v in NEW}
    info = {k: np.zeros(N, np.int64) for k in ('k1', 'kr')}; info['dlogpt_Mk'] = np.full(N, np.nan); info['dlogpt_Mpt'] = np.full(N, np.nan)
    nd, npr = T.counts(J, LAB)

    def swap(v, j, i1, i2):
        for k in IPK:
            A = E[v][k]; t = A[j, i1].copy(); A[j, i1] = A[j, i2]; A[j, i2] = t

    for j in range(N):
        if y[j] != 1: continue
        lab = LAB[j]; n = int((lab >= 0).sum())
        disp = np.flatnonzero(ch[j, :n] & (S[j, :n] > 2)); prm = np.flatnonzero(ch[j, :n] & (S[j, :n] <= 2))
        byp = lambda idx: idx[np.argsort(-pt[j, idx], kind='stable')]
        c = R['c'][j]; a, b = R['a'][j], R['b'][j]
        w = a if (npr[j, a] > npr[j, b] or (npr[j, a] == npr[j, b] and nd[j, a] <= nd[j, b])) else b   # H35's target rule
        Dc = byp(disp[lab[disp] == c]); Pc = byp(prm[lab[prm] == c]); Pw = byp(prm[lab[prm] == w])
        k = min(len(Dc), len(Pw)); k1 = min(k, len(Pc)); info['k1'][j] = k1
        if k1:
            swap('Mk', j, Dc[:k1], Pw[:k1])
            free = list(Pw); tg = []
            for i in range(k1):
                q = int(np.argmin([abs(np.log(pt[j, f]) - np.log(pt[j, Pc[i]])) for f in free])); tg.append(free.pop(q))
            tg = np.array(tg, np.int64); swap('Mpt', j, Dc[:k1], tg)
            info['dlogpt_Mk'][j] = np.mean(np.abs(np.log(pt[j, Pw[:k1]]) - np.log(pt[j, Pc[:k1]])))
            info['dlogpt_Mpt'][j] = np.mean(np.abs(np.log(pt[j, tg]) - np.log(pt[j, Pc[:k1]])))
        wd = a if nd[j, a] >= nd[j, b] else b
        Dw = byp(disp[lab[disp] == wd]); Pwd = byp(prm[lab[prm] == wd])
        kr = min(len(Dw), len(Pc), len(Pwd)); info['kr'][j] = kr
        if kr:
            swap('RV', j, Dw[:kr], Pc[:kr]); swap('CW', j, Dw[:kr], Pwd[:kr])
            free = list(Pc); tg = []
            for i in range(kr):
                q = int(np.argmin([abs(np.log(pt[j, f]) - np.log(pt[j, Pwd[i]])) for f in free])); tg.append(free.pop(q))
            swap('RVpt', j, Dw[:kr], np.array(tg, np.int64))
    return E, info


def main():
    from interp.part_core import cat
    RES.mkdir(parents=True, exist_ok=True); out = dict(node='H35_verify', per_class=PER, B=B, n_perm=NPERM)
    Jt, it = X.jc_read('Tbqq', 101, PER, 3551)[::2]; Jq, iq = X.jc_read('QCD', 101, PER, 3552)[::2]
    J = cat([Jt, Jq]); y = (J['label'] == T.TBQQ).astype(int); t = y == 1
    out['sample'] = dict(Tbqq=it, QCD=iq, n_top=int(t.sum()), n=int(len(y)))
    P4, nn = X.jc_P4(J); LAB = H.kt3_batch(np.ascontiguousarray(P4.astype(np.float64)), nn); R = T.roles(P4, nn, LAB)
    nd0, _ = T.counts(J, LAB); r = np.arange(len(y))
    ndc = nd0[r, R['c']]; agree = ndc > np.maximum(nd0[r, R['a']], nd0[r, R['b']])
    out['roles'] = dict(top_agree_frac=float(agree[t].mean()), top_ndisp_c=float(ndc[t].mean()), top_ndisp_jet=float(nd0[t].sum(1).mean()))
    log('roles', out['roles'])
    E0, info0 = T.make_edits(J, LAB, R, y); E1, info1 = new_edits(J, LAB, R, y)
    E = {v: E0[v] for v in ('M', 'C1', 'C2')}; E.update(E1)
    for v, Ev in E.items():
        Jv = dict(J); Jv.update(Ev); ndv, _ = T.counts(Jv, LAB)
        assert np.array_equal(ndv.sum(1), nd0.sum(1)), v
        assert np.allclose(np.sort(T.sig(Jv)[1], 1), np.sort(T.sig(J)[1], 1)), v
    out['edit_info'] = dict(k_M=float(info0['k'][t].mean()), k1_C1=float(info0['k1'][t].mean()), k1_Mk=float(info1['k1'][t].mean()),
                            kr=float(info1['kr'][t].mean()),
                            mean_abs_dlogpt_target_Mk_vs_C1=float(np.nanmean(info1['dlogpt_Mk'][t])),
                            mean_abs_dlogpt_target_Mpt_vs_C1=float(np.nanmean(info1['dlogpt_Mpt'][t])))
    assert np.array_equal(info0['k1'][t], info1['k1'][t])
    log('edits', out['edit_info'])
    nets = T.Nets(); Jid = T.variant_dict(J, {k: J[k] for k in IPK}); Jvar = {v: T.variant_dict(J, Ev) for v, Ev in E.items()}
    LT, CALL = {}, {}; out['unedited'] = {}
    for name in NETS:
        Lid = nets.logits(name, Jid); LT[(name, 'id')] = T.top_lo(Lid); CALL[(name, 'id')] = Lid.argmax(1) == T.TBQQ
        lab10 = np.where(t, T.TBQQ, T.QCD)
        out['unedited'][name] = dict(acc_10class=float((Lid.argmax(1) == lab10).mean()),
                                     acc_binary=float(((Lid[:, T.TBQQ] - Lid[:, T.QCD] > 0) == t).mean()),
                                     top_call_rate_tops=float(CALL[(name, 'id')][t].mean()))
        for v in E:
            L = nets.logits(name, Jvar[v]); LT[(name, v)] = T.top_lo(L); CALL[(name, v)] = L.argmax(1) == T.TBQQ
        log(name, out['unedited'][name])
    d = lambda m, v: LT[(m, v)] - LT[(m, 'id')]
    CON = {'M': lambda m: d(m, 'M'), 'C1': lambda m: d(m, 'C1'), 'M-C1': lambda m: d(m, 'M') - d(m, 'C1'),
           'Mk': lambda m: d(m, 'Mk'), 'Mk-C1': lambda m: d(m, 'Mk') - d(m, 'C1'), 'Mpt': lambda m: d(m, 'Mpt'),
           'Mpt-C1': lambda m: d(m, 'Mpt') - d(m, 'C1'), 'RV': lambda m: d(m, 'RV'), 'CW': lambda m: d(m, 'CW'),
           'RV-CW': lambda m: d(m, 'RV') - d(m, 'CW'), 'RVpt': lambda m: d(m, 'RVpt'), 'RVpt-CW': lambda m: d(m, 'RVpt') - d(m, 'CW')}
    TESTED = ('M-C1', 'Mk-C1', 'Mpt-C1', 'RV-CW', 'RVpt-CW')
    lt0 = LT[('ParT_full', 'id')]; low = lt0 < np.median(lt0[t])
    kM, k1, kr = info0['k'], info1['k1'], info1['kr']
    sets = {'top_agree_M': t & agree & (kM > 0), 'top_all_M': t & (kM > 0), 'qcd_M': ~t & (kM > 0),
            'top_agree_k1': t & agree & (k1 > 0), 'top_all_k1': t & (k1 > 0),
            'top_agree_kr': t & agree & (kr > 0), 'top_disagree_kr': t & ~agree & (kr > 0), 'top_all_kr': t & (kr > 0),
            'top_agree_k1_low': t & agree & (k1 > 0) & low, 'top_disagree_kr_low': t & ~agree & (kr > 0) & low}
    use = {'M': ('top_agree_M', 'top_all_M', 'qcd_M'), 'C1': ('top_agree_M', 'top_all_M', 'qcd_M', 'top_agree_k1', 'top_all_k1'),
           'M-C1': ('top_agree_M', 'top_all_M', 'qcd_M'), 'Mk': ('top_agree_k1', 'top_all_k1'), 'Mk-C1': ('top_agree_k1', 'top_all_k1'),
           'Mpt': ('top_agree_k1', 'top_all_k1'), 'Mpt-C1': ('top_agree_k1', 'top_all_k1'),
           'RV': ('top_agree_kr', 'top_disagree_kr', 'top_all_kr'), 'CW': ('top_agree_kr', 'top_disagree_kr', 'top_all_kr'),
           'RV-CW': ('top_agree_kr', 'top_disagree_kr', 'top_all_kr', 'top_disagree_kr_low'),
           'RVpt': ('top_agree_kr', 'top_disagree_kr', 'top_all_kr'), 'RVpt-CW': ('top_agree_kr', 'top_disagree_kr', 'top_all_kr', 'top_disagree_kr_low')}
    use['Mpt-C1'] = use['Mpt-C1'] + ('top_agree_k1_low',)
    out['sets'] = {k: int(v.sum()) for k, v in sets.items()}; log('sets', out['sets'])
    rng = np.random.default_rng(3599); res, cells = {}, []
    for m in NETS:
        for cn, f in CON.items():
            for sn in use[cn]:
                x = f(m)[sets[sn]]; mu, ci = T.boot(x, rng, B); p = T.perm2(x, rng, NPERM)
                res[f'{sn}|{m}|{cn}'] = dict(mean=mu, ci=ci, p_perm=p, n=int(len(x)))
                if cn in TESTED and m != 'ParT_kin': cells.append(f'{sn}|{m}|{cn}')
        for sn, (va, vb) in (('top_agree_M', ('C1', 'M')), ('top_agree_k1', ('C1', 'Mpt')), ('top_all_k1', ('C1', 'Mpt')),
                             ('top_disagree_kr', ('CW', 'RV')), ('top_all_kr', ('CW', 'RV')), ('top_disagree_kr', ('CW', 'RVpt'))):
            sm = sets[sn]; res[f'{sn}|{m}|mcnemar_{vb}_vs_{va}'] = H._mcnemar(CALL[(m, va)][sm], CALL[(m, vb)][sm])
            for v in ('id', va, vb): res[f'{sn}|{m}|topcall|{v}'] = float(CALL[(m, v)][sm].mean())
    adj = T.holm([res[c]['p_perm'] for c in cells])
    for c, a in zip(cells, adj): res[c]['p_holm'] = float(a)
    out['holm_family'] = dict(n_tests=len(cells), cells=cells); out['results'] = res
    out['kin_sanity'] = {v: float(np.abs(d('ParT_kin', v)).max()) for v in E}
    for c in cells: log(f'{c}: {res[c]["mean"]:+.3f} [{res[c]["ci"][0]:+.3f}, {res[c]["ci"][1]:+.3f}] holm {res[c]["p_holm"]:.2g}')
    log('kin sanity', out['kin_sanity'])

    def clean(o):
        if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [clean(v) for v in o]
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return clean(o.tolist())
        return o
    (RES / f'{TAG}.json').write_text(json.dumps(clean(out), indent=1))
    np.savez_compressed(RES / f'{TAG}_perjet.npz', y=y, agree=agree, k=kM, k1=k1, kr=kr,
                        **{f'lt|{m}|{v}': a for (m, v), a in LT.items()})
    log('saved')


if __name__ == '__main__':
    main()
