"""H44 verification (loop D, skeptical verifier). Fresh sample, re-run of the key H44 edits, and missing controls.

Sample: JetClass test_20M HToBB_101 / HToCC_101 / ZJetsToNuNu_101 (different files from H44's _100), PER random rows each
  (rng 4451 / 4452 / 4453; never first-k). Every variant, identity included, rebuilt with part_core.build (via h44.variant_dict).
Re-run (h44 definitions): L(0.4), Lc(0.4), Lc(2.0), Vm/C1m/Vp/C1p/Cr (h44.make_edits).
New edits / controls (all on IP tuples only; four-momenta never changed):
  LcF(lam)  count-preserving map as Lc but on 18 fine S bins (edges 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8.5, 10, 12, 15, 20, 30, 50, 100,
            200, 500, inf; tail S' = 500 + lam (S - 500)). Keeps n(S>s) fixed at every one of these 18 thresholds. A network that
            only counts tracks above a threshold located at any of these edges gives 0; a smooth function of S gives a response
            scaled by the (smaller) change in ln S.
  Lr        random-direction control of equal per-track size: each displaced track independently gets the coarse Lc map with
            lam = 0.4 or lam = 2.5 (rng 4460, p = 1/2). Mean change of ln S near 0; per-track |change| similar to Lc(0.4).
  LcP(0.4)  prompt-track control: the in-bin map applied to prompt tracks only (S in (0,1]: S' = S^(1/lam); (1,2]: the power map),
            displaced tracks untouched; counts fixed. Tests whether any in-bin d0 shrinkage moves s_bc (non-lifetime tracks).
  Ls(0.4)   d0val, d0err, dzval, dzerr of displaced tracks all x 0.4: S unchanged (all counts and every S fixed), absolute
            |d0| and its error shrink. Separates a response to the absolute displacement from a response to the significance.
Data check (no network): on JetClass train_100M HToBB_001 / HToCC_001 (NREF random rows each, rng 4461 / 4462) and on the test
  sample with truth labels: at fixed count tuple (n2, n3, n5, n10), is the in-bin position of displaced tracks larger for bb
  than for cc? u_bin = mean over displaced tracks in finite coarse bins of ln(S/lo)/ln(hi/lo); also median ln S and max ln S.
  Stratified bb - cc difference (strata with >= 5 jets of each class; weight = stratum size), 95 % bootstrap over jets.
Models: ParT_full, PN_full; ParT_perm1..NPN (h44.load_net, torch seeds 5..).
Statistics: per-jet response mean with 95 % bootstrap, sign-flip permutation p, Holm over the tested cells.
  Pre-registered here: V - C1 contrast at equal pT mismatch = intercept of the per-jet regression of (V - C1) on (mis_V - mis_C1).
Run: ./job.sh d-h44-verify 8 16 'export PATH=/j-jepa-vol/aaron-interp/env-part/bin:$PATH JETDISTILL_TAGGER=part ...;
     python -m interp.h44_verify'      (H44V_SMOKE=1: 200 per class)
Output: $AI_RESULTS/physics/h44_verify.json, h44_verify_perjet.npz
"""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import h44_bc_lifetime as H  # noqa: E402

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
SMOKE = bool(os.environ.get('H44V_SMOKE'))
RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'physics'
TAG = 'h44_verify_smoke' if SMOKE else 'h44_verify'
PER = 200 if SMOKE else 1500
NREF = 3000 if SMOKE else 20000
B = 300 if SMOKE else 2000
NPERM = 500 if SMOKE else 5000
NPN = 2 if SMOKE else 4
FINE = np.array([2.0, 2.5, 3, 3.5, 4, 5, 6, 7, 8.5, 10, 12, 15, 20, 30, 50, 100, 200, 500, np.inf])
COARSE = np.array([2.0, 3.0, 5.0, 10.0, np.inf])
QCD, HBB, HCC = 0, 1, 2
CLS = (('Hbb', 'HToBB', 4451, HBB), ('Hcc', 'HToCC', 4452, HCC), ('QCD', 'ZJetsToNuNu', 4453, QCD))
IPK = H.IPK


def binmap(S, sel, edges, lam):
    """factor S'/S of the in-bin power map on bins (edges[i], edges[i+1]] for tracks in sel (S > edges[0]); open tail scaled by lam"""
    b = np.clip(np.searchsorted(edges, S, side='left') - 1, 0, len(edges) - 2)
    lo = edges[b]; hi = edges[b + 1]; fin = sel & np.isfinite(hi); tail = sel & ~np.isfinite(hi)
    with np.errstate(all='ignore'):
        lo_ = np.where(fin, lo, 1.0); hi_ = np.where(fin, hi, 2.0); S_ = np.where(fin, S, 1.0)
        u = np.clip(np.log(S_ / lo_) / np.log(hi_ / lo_), 0, 1)
        S2 = np.where(fin, lo_ * (hi_ / lo_) ** (u ** (1.0 / lam)), S)
        S2 = np.where(tail, lo + lam * (S - lo), S2)
        f = np.where(sel & (S > 0), S2 / np.where(S > 0, S, 1), 1.0)
    return f, b


def apply_factor(J, f, sel, edges, keys=('d0val', 'dzval')):
    e = {k: J[k].copy() for k in IPK}
    for k in keys: e[k] = (J[k] * f).astype(np.float32)
    ch, S2 = H.sig(J, e); _, S = H.sig(J)
    b0 = np.searchsorted(edges, S, side='left'); b1 = np.searchsorted(edges, S2, side='left'); bad = sel & (b0 != b1)
    for k in keys: e[k] = np.where(bad, J[k], e[k]).astype(np.float32)
    return e, int(bad.sum())


def new_edits(J):
    ch, S = H.sig(J); d = ch & (S > 2); pr = ch & (S <= 2); E = {}; info = {}
    for lam in (0.4, 2.0):
        f, _ = binmap(S, d, FINE, lam); E[f'LcF{lam}'], info[f'LcF{lam}_reverted'] = apply_factor(J, f, d, FINE)
    rng = np.random.default_rng(4460); coin = rng.random(S.shape) < 0.5
    f1, _ = binmap(S, d, COARSE, 0.4); f2, _ = binmap(S, d, COARSE, 2.5); f = np.where(coin, f1, f2)
    E['Lr'], info['Lr_reverted'] = apply_factor(J, f, d, COARSE)
    # prompt control: (0,1]: S' = S^(1/lam); (1,2]: power map with lo = 1, hi = 2
    lam = 0.4
    with np.errstate(all='ignore'):
        a = pr & (S > 0) & (S <= 1); S2 = np.where(a, np.where(a, S, 1) ** (1 / lam), S)
        bsel = pr & (S > 1); u = np.clip(np.log(np.where(bsel, S, 1)) / np.log(2.0), 0, 1); S2 = np.where(bsel, 2.0 ** (u ** (1 / lam)), S2)
        f = np.where(pr & (S > 0), S2 / np.where(S > 0, S, 1), 1.0)
    E['LcP0.4'], info['LcP0.4_reverted'] = apply_factor(J, f, pr, np.array([0.0, 1.0, 2.0, np.inf]))
    E['Ls0.4'], info['Ls0.4_reverted'] = apply_factor(J, np.where(d, 0.4, 1.0), d, np.array([0.0, 1, 2, 3, 5, 10, np.inf]), keys=IPK)
    return E, info


def size_of(J, e):
    """per-jet sum over charged tracks of d ln S and of d tanh|d0val| (the networks read tanh(d0val))"""
    ch, S0 = H.sig(J); _, S1 = H.sig(J, e)
    with np.errstate(all='ignore'):
        dl = np.where(ch & (S0 > 0), np.log(np.maximum(S1, 1e-12)) - np.log(np.maximum(S0, 1e-12)), 0.0)
    dt = np.where(ch, np.tanh(np.abs(e['d0val'].astype(np.float64))) - np.tanh(np.abs(J['d0val'].astype(np.float64))), 0.0)
    return dl.sum(1), np.abs(dl).sum(1), dt.sum(1)


def inbin_obs(J):
    ch, S = H.sig(J); d = ch & (S > 2)
    b = np.clip(np.searchsorted(COARSE, S, side='left') - 1, 0, len(COARSE) - 2); lo = COARSE[b]; hi = COARSE[b + 1]; fin = d & np.isfinite(hi)
    with np.errstate(all='ignore'):
        u = np.where(fin, np.log(np.where(fin, S, 1) / lo) / np.log(np.where(fin, hi, 2) / lo), np.nan)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore'); ub = np.nanmean(u, 1)
        tl = np.where(d & ~np.isfinite(hi), np.log(np.where(d, S, 10) / 10.0), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore'); tm = np.nanmean(tl, 1)
    cnt = np.stack([(ch & (S > t)).sum(1) for t in (2, 3, 5, 10)], 1)
    return cnt, ub, tm


def strat_diff(cnt, x, isbb, rng):
    """stratified mean(x | bb) - mean(x | cc) at fixed count tuple; bootstrap over jets"""
    ok = np.isfinite(x)
    def one(ix):
        o = ok[ix]; c = cnt[ix][o]; xx = x[ix][o]; bb = isbb[ix][o].astype(float)
        keys = c[:, 0] * 10**6 + c[:, 1] * 10**4 + c[:, 2] * 10**2 + c[:, 3]
        _, inv = np.unique(keys, return_inverse=True)
        nb = np.bincount(inv, bb); nc = np.bincount(inv, 1 - bb); sb = np.bincount(inv, xx * bb); scc = np.bincount(inv, xx * (1 - bb))
        v = (nb >= 5) & (nc >= 5); w = (nb + nc)[v]
        if not v.any(): return np.nan, 0
        return float((w * (sb[v] / nb[v] - scc[v] / nc[v])).sum() / w.sum()), float(w.sum())
    est, w = one(np.arange(len(x)))
    bs = [one(rng.integers(0, len(x), len(x)))[0] for _ in range(300 if SMOKE else 1000)]
    return dict(diff_bb_minus_cc=float(est), ci=np.nanpercentile(bs, [2.5, 97.5]).tolist(), jets_in_strata=int(w))


def main():
    from interp.part_core import cat
    RES.mkdir(parents=True, exist_ok=True)
    out = dict(node='H44_verify', smoke=SMOKE, per_class=PER, B=B, n_perm=NPERM, fine_edges=FINE.tolist()); parts = []; out['sample'] = {}
    for c, pre, seed, lab in CLS:
        Jc, inf = H.load_rows(H.TEST / f'{pre}_101.root', PER, seed); parts.append(Jc); out['sample'][c] = inf
        assert np.all(Jc['label'] == lab)
    J = cat(parts); y = J['label'].astype(int); N = len(y)
    LAB, P4 = H.subjet_labels(J); F0 = H.feats(J, LAB, P4)
    E0, out['h44_edit_info'], bk, okm, okp = H.make_edits(J, LAB, P4, F0)
    E = {k: E0[k] for k in ('L0.4', 'Lc0.4', 'Lc2.0', 'Vm', 'C1m', 'Vp', 'C1p', 'Cr')}
    En, out['new_edit_info'] = new_edits(J); E.update(En); VARS = list(E)
    log('edits', out['h44_edit_info'], out['new_edit_info'])
    # assertions: counts fixed
    ch, S0 = H.sig(J)
    for v in VARS:
        if v == 'L0.4': continue
        _, S1 = H.sig(J, E[v]); thr = (1, 2, 3, 5, 10) + (tuple(FINE[1:-1]) if v.startswith('LcF') else ())
        for t in thr: assert np.array_equal((ch & (S1 > t)).sum(1), (ch & (S0 > t)).sum(1)), f'{v} changed n(S>{t})'
        if v == 'Ls0.4': assert np.allclose(S1, S0, rtol=1e-4), 'Ls changed S'
    SZ = {v: size_of(J, E[v]) for v in VARS}
    out['edit_size'] = {v: {c: dict(sum_dlnS=float(SZ[v][0][y == lab].mean()), sum_abs_dlnS=float(SZ[v][1][y == lab].mean()),
                                    sum_dtanh_d0=float(SZ[v][2][y == lab].mean())) for c, _, _, lab in CLS} for v in VARS}
    log('edit size', json.dumps(out['edit_size']['Lc0.4']), json.dumps(out['edit_size']['LcF0.4']), json.dumps(out['edit_size']['Lr']))
    # ---- data direction check (no network)
    rng = np.random.default_rng(4470); hb = np.isin(y, [HBB, HCC]); cnt, ub, tm = inbin_obs(J)
    out['data_check_test'] = {nm: strat_diff(cnt[hb], x[hb], y[hb] == HBB, rng) for nm, x in
                              (('u_bin', ub), ('tail_lnS_over_10', tm), ('medlogS', F0['medlogS']), ('maxlogS', F0['maxlogS']))}
    Jr = cat([H.load_rows(H.TRAIN / f'{p}_001.root', NREF, s)[0] for p, s in (('HToBB', 4461), ('HToCC', 4462))])
    yr = Jr['label'].astype(int); cr, ubr, tmr = inbin_obs(Jr); LABr, P4r = H.subjet_labels(Jr); Fr = H.feats(Jr, LABr, P4r)
    out['data_check_train'] = {nm: strat_diff(cr, x, yr == HBB, rng) for nm, x in
                               (('u_bin', ubr), ('tail_lnS_over_10', tmr), ('medlogS', Fr['medlogS']), ('maxlogS', Fr['maxlogS']))}
    log('data check test', json.dumps(out['data_check_test'])); log('data check train', json.dumps(out['data_check_train']))
    # ---- networks
    Jid = H.variant_dict(J, {k: J[k] for k in IPK}); Jv = {v: H.variant_dict(J, E[v]) for v in VARS}
    NETS = ('ParT_full', 'PN_full') + tuple(f'ParT_perm{i}' for i in range(1, NPN + 1)); SC = {}; out['unedited'] = {}
    for name in NETS:
        M = H.load_net(name); L = H.logits(M, Jid); SC[(name, 'id')] = L
        out['unedited'][name] = dict(acc_10class=float((L.argmax(1) == y).mean()),
                                     acc_bb_vs_cc=float(((L[hb, HBB] > L[hb, HCC]) == (y[hb] == HBB)).mean()),
                                     auc_bb_vs_cc=H.auc(L[hb, HBB] - L[hb, HCC], (y[hb] == HBB).astype(int)))
        log(name, out['unedited'][name])
        for v in VARS: SC[(name, v)] = H.logits(M, Jv[v])
        del M
    # ---- statistics
    rng = np.random.default_rng(4499)
    sc = dict(bc=lambda L: L[:, HBB] - L[:, HCC], bq=lambda L: L[:, HBB] - L[:, QCD])
    resp = lambda m, v, s: sc[s](SC[(m, v)]) - sc[s](SC[(m, 'id')])
    CON = {v: (lambda m, s, v=v: resp(m, v, s)) for v in VARS}
    for V, C in (('Vm', 'C1m'), ('Vp', 'C1p'), ('Cr', 'C1m')):
        CON[f'{V}-{C}'] = lambda m, s, V=V, C=C: resp(m, V, s) - resp(m, C, s)
    SETOF = lambda cn: okm if any(t in cn for t in ('Vm', 'C1m', 'Cr')) else (okp if any(t in cn for t in ('Vp', 'C1p')) else np.ones(N, bool))
    TESTED = ['L0.4', 'Lc0.4', 'Lc2.0', 'LcF0.4', 'LcF2.0', 'Lr', 'LcP0.4', 'Ls0.4', 'Vm-C1m', 'Vp-C1p', 'Cr-C1m']
    res, cells = {}, []
    for c, _, _, lab in CLS:
        for s in ('bc', 'bq'):
            for cn, fn in CON.items():
                sm = SETOF(cn) & (y == lab)
                for m in NETS:
                    x = fn(m, s)[sm]; mu, ci = H.boot(x, rng, B); cell = dict(mean=mu, ci=ci, n=int(sm.sum()))
                    if m in ('ParT_full', 'PN_full') and cn in TESTED and s == 'bc' and c != 'QCD':
                        cell['p_perm'] = H.perm2(x, rng, NPERM); cells.append(f'{c}|{s}|{cn}|{m}')
                    res[f'{c}|{s}|{cn}|{m}'] = cell
                pm = [res[f'{c}|{s}|{cn}|ParT_perm{i}']['mean'] for i in range(1, NPN + 1)]
                res[f'{c}|{s}|{cn}|perm_band'] = dict(min=float(min(pm)), max=float(max(pm)))
        # response per unit sum d ln S, and LcF / Lc ratio vs size ratio
        for m in ('ParT_full', 'PN_full'):
            ym = y == lab
            for v in ('Lc0.4', 'LcF0.4', 'Lc2.0', 'LcF2.0', 'L0.4'):
                r = resp(m, v, 'bc')[ym]; z = SZ[v][0][ym]; ii = rng.integers(0, ym.sum(), (B, ym.sum()))
                q = r[ii].mean(1) / z[ii].mean(1)
                res[f'{c}|per_unit_dlnS|{v}|{m}'] = dict(value=float(r.mean() / z.mean()), ci=np.percentile(q, [2.5, 97.5]).tolist())
            for a_, b_ in (('LcF0.4', 'Lc0.4'), ('LcF2.0', 'Lc2.0'), ('Lc0.4', 'L0.4')):
                ra = resp(m, a_, 'bc')[ym]; rb = resp(m, b_, 'bc')[ym]; ii = rng.integers(0, ym.sum(), (B, ym.sum()))
                q = ra[ii].mean(1) / rb[ii].mean(1)
                res[f'{c}|ratio_{a_}_to_{b_}|{m}'] = dict(ratio=float(ra.mean() / rb.mean()), ci=np.percentile(q, [2.5, 97.5]).tolist(),
                                                         size_ratio_dlnS=float(SZ[a_][0][ym].mean() / SZ[b_][0][ym].mean()))
            # pre-registered pT-mismatch-matched V - C1 (intercept of regression on mismatch difference)
            for V, C, ok in (('Vm', 'C1m', okm), ('Vp', 'C1p', okp)):
                s_ = ok & ym
                if s_.sum() < 20: continue
                dv = (resp(m, V, 'bc') - resp(m, C, 'bc'))[s_]; dx = (bk[f'mis{V}'] - bk[f'mis{C}'])[s_]
                X = np.column_stack([np.ones_like(dx), dx]); beta = np.linalg.lstsq(X, dv, rcond=None)[0]
                bs = []
                for _ in range(1000):
                    ii = rng.integers(0, len(dv), len(dv)); bs.append(np.linalg.lstsq(X[ii], dv[ii], rcond=None)[0])
                bs = np.array(bs)
                res[f'{c}|mismatch_adjusted|{V}-{C}|{m}'] = dict(intercept=float(beta[0]), ci=np.percentile(bs[:, 0], [2.5, 97.5]).tolist(),
                                                                slope=float(beta[1]), n=int(s_.sum()), mean_mis_diff=float(dx.mean()))
    adj = H.holm([res[c]['p_perm'] for c in cells])
    for c, a in zip(cells, adj): res[c]['p_holm'] = float(a)
    out['holm_family'] = dict(n_tests=len(cells)); out['results'] = res
    for c in ('Hbb', 'Hcc', 'QCD'):
        for cn in TESTED + ['C1m']:
            k = f'{c}|bc|{cn}'; pb = res[f'{k}|perm_band']
            log(f'{c} bc {cn}: ParT {res[k + "|ParT_full"]["mean"]:+.3f} {np.round(res[k + "|ParT_full"]["ci"], 3).tolist()}  '
                f'PN {res[k + "|PN_full"]["mean"]:+.3f} {np.round(res[k + "|PN_full"]["ci"], 3).tolist()}  perm [{pb["min"]:+.3f},{pb["max"]:+.3f}]  '
                f'holm ParT {res[k + "|ParT_full"].get("p_holm", float("nan")):.2g}  n {res[k + "|ParT_full"]["n"]}')
    (RES / f'{TAG}.json').write_text(json.dumps(H.clean(out), indent=1))
    keep = dict(y=y, okm=okm, okp=okp, **{f'sz_{v}': SZ[v][0] for v in VARS})
    for (m, v), L in SC.items():
        if m in ('ParT_full', 'PN_full'): keep[f'L|{m}|{v}'] = L.astype(np.float32)
    np.savez_compressed(RES / f'{TAG}_perjet.npz', **keep); log('saved')


if __name__ == '__main__':
    main()
