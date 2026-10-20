"""H25 (loop D, claim D6 new arm): do top taggers use the three-prong shape tau32 at fixed jet mass and W-pair mass?

H4 held tau32 fixed and moved the W-pair mass. Here the subjet four-momenta are held fixed (so the jet mass m, every pairwise
subjet mass and so the W candidate mW are fixed) and the angular width of each subjet is changed, which moves tau32.

Samples (random rows, never first-k)
  top  Top Landscape test split, 3,000 top + 3,000 QCD (interp.data_top.sample rows, seed 9733).
  jc   JetClass test TTBar_100 (Tbqq) and ZJetsToNuNu_100 (QCD), stored 550 < jet_pt < 650 GeV, |jet_eta| < 2, >= 3 particles;
       3,000 per class (rng 9733 / 9734 among eligible rows; interp.h19_wpair_cross.jc_read).
  Every jet is first rebuilt through (pT, eta, phi, m) (identity, h4_top_wmass._identity).
Edit S(s), s in {0.5, 0.7, 1.4, 2.0}: three exclusive-kT subjets (E-scheme, h4_top_wmass.kt3_batch). Every constituent's
  (eta, phi) offset from its subjet's pT-weighted centroid is multiplied by s (all intra-subjet pairwise angles scale by s;
  constituent pT and masses kept). The subjets are then moved rigidly to restore the W-candidate pair mass and the jet mass
  (h4_top_wmass inner_W with target = the jet's own mW: the W pair, = the subjet pair with mass closest to 80.4 GeV, opened
  about its pT-weighted centroid by f; the third subjet moved along the line from that centroid by g). So m and mW are kept;
  tau32 moves (s < 1 lowers it); the subjet masses scale with s (side effect, covered by the MLPx reference below).
  Pre-registered first version (debug attempt 0, smoke d-h25-smoke): keep each subjet four-momentum by reweighting the
  constituent pT inside the subjet (w_i = exp(q_i . lambda)); infeasible: 0/300 Top jets converged at s = 0.5, 1.4, 2.0
  (a subjet's mass at fixed pT fixes its pT-weighted angular second moment, so its width cannot be scaled by s).
  Accepted if, recomputed after re-clustering, |dm| < 0.5 GeV and |dmW| < 2 GeV (mW = pair mass closest to 80.4, H2).
Controls (same jets, same acceptance rule, same m / mW restore)
  c1  rotation of each subjet's constituents about its centroid (random sign per subjet) by the angle that gives the same
      pT-weighted mean constituent displacement as S: intra-subjet pairwise angles, subjet masses and so tau32 ~ fixed.
  c2  the same scaling by s about 3 random constituents (drawn with probability prop. to pT; each constituent assigned to the
      nearest), then the same restore.
  c3  translation-only TN: the whole jet translated rigidly in (eta, phi) by the S edit's mean displacement, random direction.
  Every edited jet is rebuilt about its own constituent-sum axis for every network (JetClass nets: recentred jet dict, H19v).
Models: Top Landscape ParT_kin / PN_kin (H4 checkpoint snapshots), LorentzNet (released), PFN, EFN (H2); JetClass ParT_kin,
  PN_kin (released; score L(Tbqq) - L(QCD)); every network on both samples (cross-dataset).
  References fitted per dataset on disjoint jets (top: Top val split 25k per class, rng 4194; jc: TTBar_101 / ZJetsToNuNu_101,
  25k per class, rng 4193/4194; h19_wpair_cross.ref_train_*): LR3 = binned LR(m, tau32, mW) and MLP3 = MLP(64, 64) on
  (m, tau32, mW) (positive controls), LR2 = LR(m, mW) (negative control), MLPx = MLP on the edit's side effects without any
  N-subjettiness (m, mW, mSD, pair masses, subjet masses, subjet pT fractions, pair dR, multiplicity), MLPxT = MLPx + tau1-3,
  tau21, tau32 (side-effect references, as H19).
Metrics per sample x class x s x model: d_v = score_v - score_id (top log-odds) for v = S, c1, c2, c3; S - c1 mean with 95 %
  bootstrap CI over jets (B = 2000), sign-flip permutation p (two-sided), Holm over networks x s x samples x classes;
  ratio (S - c1)_net / (S - c1)_LR3 (bootstrap CI); dtau32 achieved; % calls changed at eps_S 0.3 / 0.5 (thresholds from the
  sample's unedited tops) and McNemar S vs c1; C2ST per s (edited jets vs natural reference jets matched in (m, tau32, mW),
  HGB on 20 substructure features, 2-fold AUC; identity row = matching floor).
Run: ./sync.sh && ./part_job.sh d-h25-run 'python -m interp.h25_top_tau32'   (H25_SMOKE=1: 150 per class, 2,000 ref jets)
Outputs: $AI_RESULTS/physics/h25_top_tau32.json, h25_perjet.npz, h25_tau32.png
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

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
SMOKE = bool(os.environ.get('H25_SMOKE'))
RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'physics'
TAG = 'h25_smoke' if SMOKE else 'h25'
PER = 150 if SMOKE else 3000
X.NREF = 2000 if SMOKE else 25000
B = 300 if SMOKE else 2000
NPERM = 500 if SMOKE else 5000
SS = (0.5, 0.7, 1.4, 2.0)
VAR = ('S', 'c1', 'c2', 'c3')
NCPU = int(os.environ.get('OMP_NUM_THREADS', '4'))
NETS = ['ParT_top', 'PN_top', 'LNet_top', 'PFN_top', 'EFN_top', 'ParT_jc', 'PN_jc']
REFS = ['LR3', 'MLP3', 'LR2', 'MLPx', 'MLPxT']
WMIN, WMAX = 0.05, 20.0
TBQQ, QCD = 8, 0


# ================================================================================================================ edit
def _jet(P):
    P = np.asarray(P, np.float64); pt = np.hypot(P[:, 0], P[:, 1]); m = np.sqrt(np.maximum(P[:, 3] ** 2 - (P[:, :3] ** 2).sum(1), 0))
    eta = np.arcsinh(P[:, 2] / np.maximum(pt, 1e-12)); phi = np.arctan2(P[:, 1], P[:, 0]); S = P.sum(0)
    eJ = np.arcsinh(S[2] / np.hypot(S[0], S[1])); pJ = np.arctan2(S[1], S[0])
    return P, pt, m, eta - eJ, (phi - pJ + np.pi) % (2 * np.pi) - np.pi, eJ, pJ


def _p4(pt, m, eta, phi):
    return np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), np.sqrt((pt * np.cosh(eta)) ** 2 + m ** 2)], -1)


def _restore(pt, m, x2, y2, lab, eJ, pJ, w, mW0, m0):
    """rigid (eta, phi) shifts of the kT subjets after an internal operation: the W pair (a, b) opened about its pT-weighted
    centroid by f so that m_ab = mW0, then subjet c moved along the line from that centroid by g so that m_jet = m0
    (h4_top_wmass.edit_jet inner_W with target mW0); returns (x, y) or None"""
    a, b = w; c = 3 - a - b; sel = [lab == k for k in range(3)]
    ptk = np.array([pt[s].sum() for s in sel]); r = np.array([[(pt[s] * x2[s]).sum() / ptk[k], (pt[s] * y2[s]).sum() / ptk[k]] for k, s in enumerate(sel)])
    cW = (ptk[a] * r[a] + ptk[b] * r[b]) / (ptk[a] + ptk[b])
    V = lambda k, sh: _p4(pt[sel[k]], m[sel[k]], eJ + x2[sel[k]] + sh[0], pJ + y2[sel[k]] + sh[1]).sum(0)
    f = H._bisect(lambda f: H._minv(V(a, (f - 1) * (r[a] - cW)) + V(b, (f - 1) * (r[b] - cW))) - mW0, 0.05, 4.0, tol=0.005)
    if f is None: return None
    Vab = V(a, (f - 1) * (r[a] - cW)) + V(b, (f - 1) * (r[b] - cW))
    g = H._bisect(lambda g: H._minv(Vab + V(c, (g - 1) * (r[c] - cW))) - m0, 0.0, 6.0, tol=0.005)
    if g is None: return None
    sh = np.zeros((3, 2)); sh[a] = (f - 1) * (r[a] - cW); sh[b] = (f - 1) * (r[b] - cW); sh[c] = (g - 1) * (r[c] - cW)
    return x2 + sh[lab, 0], y2 + sh[lab, 1]


def edit_jet(P, lab, s, seed):
    """S, c1, c2, c3 on one jet for one s; returns {kind: (P_new or None, D, reason)}"""
    P, pt, m, x, y, eJ, pJ = _jet(P); rng = np.random.default_rng(seed); out = {}
    sel = [lab == k for k in range(3)]
    if min(s_.sum() for s_ in sel) < 1: return {k: (None, 0.0, 'empty_subjet') for k in VAR}
    Vk = [P[s_].sum(0) for s_ in sel]; pm = {(i, j): H._minv(Vk[i] + Vk[j]) for i, j in ((0, 1), (0, 2), (1, 2))}
    w = min(pm, key=lambda p: abs(pm[p] - H.MWREF)); mW0 = pm[w]; m0 = H._minv(P.sum(0))
    cen = np.array([[(pt[s_] * x[s_]).sum() / pt[s_].sum(), (pt[s_] * y[s_]).sum() / pt[s_].sum()] for s_ in sel])
    rho = np.hypot(x - cen[lab, 0], y - cen[lab, 1]); mrho = float((pt * rho).sum() / pt.sum())
    disp = lambda x2, y2: float((pt * np.hypot(x2 - x, y2 - y)).sum() / pt.sum())

    def finish(kind, x2, y2):
        xy = _restore(pt, m, x2, y2, lab, eJ, pJ, w, mW0, m0)
        if xy is None: out[kind] = (None, disp(x2, y2), 'restore_no_conv'); return
        out[kind] = (_p4(pt, m, eJ + xy[0], pJ + xy[1]), disp(*xy), 'ok')
    # S: every constituent's offset from its subjet centroid scaled by s, then m and mW restored by rigid subjet shifts
    finish('S', cen[lab, 0] + s * (x - cen[lab, 0]), cen[lab, 1] + s * (y - cen[lab, 1]))
    DS = out['S'][1] if out['S'][0] is not None else abs(s - 1) * mrho
    # c1: rotation about each subjet centroid (random sign per subjet), angle set so the mean displacement equals S's
    al = 2 * np.arcsin(min(1.0, DS / (2 * max(mrho, 1e-9)))); sg = rng.choice([-1.0, 1.0], 3)[lab]
    ca, sa = np.cos(al * sg), np.sin(al * sg); dx, dy = x - cen[lab, 0], y - cen[lab, 1]
    finish('c1', cen[lab, 0] + ca * dx - sa * dy, cen[lab, 1] + sa * dx + ca * dy)
    # c2: the same scaling about 3 random constituents (prob. prop. to pT), each constituent assigned to the nearest
    ci = rng.choice(len(pt), 3, replace=False, p=pt / pt.sum()) if len(pt) >= 3 else np.arange(len(pt))
    near = ci[np.argmin((x[:, None] - x[None, ci]) ** 2 + (y[:, None] - y[None, ci]) ** 2, 1)]
    finish('c2', x[near] + s * (x - x[near]), y[near] + s * (y - y[near]))
    # c3 (TN): rigid translation of the whole jet by S's mean displacement in a random direction
    th = rng.uniform(0, 2 * np.pi)
    out['c3'] = (_p4(pt, m, eJ + x + DS * np.cos(th), pJ + y + DS * np.sin(th)), DS, 'ok')
    return out


def _work(arg):
    j, P, lab = arg
    return {s: edit_jet(P, lab, s, 9733 * 1000 + j * 10 + i) for i, s in enumerate(SS)}


def run_edits(P4, nn):
    import multiprocessing as mp
    rows = np.flatnonzero(nn >= 3); F = np.zeros((len(rows), P4.shape[1], 4), np.float64); F[:] = P4[rows]
    LAB = H.kt3_batch(F, nn[rows]); log(f'  kT subjets for {len(rows)} jets')
    jobs = [(int(j), P4[j, :nn[j]].astype(np.float64), LAB[ii, :nn[j]]) for ii, j in enumerate(rows)]
    with mp.get_context('fork').Pool(NCPU) as pool:
        res = pool.map(_work, jobs, chunksize=16)
    E = {}
    for s in SS:
        for kind in VAR:
            Pv = P4.copy(); conv = np.zeros(len(nn), bool); D = np.zeros(len(nn)); lw = np.zeros(len(nn)); why = {}
            for j, r in zip(rows, res):
                Pn, d, reason = r[s][kind]; why[reason] = why.get(reason, 0) + 1; D[j] = d
                if Pn is not None:
                    Pv[j, :nn[j]] = Pn.astype(np.float32); conv[j] = True
            E[(s, kind)] = dict(P=Pv, conv=conv, D=D, logw=lw, reasons=why)
            log(f'  s={s} {kind}: converged {conv.sum()}/{len(nn)} {why}')
    return E


def sort_pt(P4, nn):
    """constituents re-sorted by pT (decreasing) within each jet (the Top nets take the first 128)"""
    pt = np.hypot(P4[..., 0], P4[..., 1]); L = P4.shape[1]; real = np.arange(L)[None, :] < nn[:, None]
    o = np.argsort(-np.where(real, pt, -1), 1, kind='stable'); return np.take_along_axis(P4, o[..., None], 1)


# ================================================================================================================ models
def axis(P4, nn):
    P = P4.astype(np.float64); real = np.arange(P.shape[1])[None, :] < nn[:, None]
    S = (P * real[..., None]).sum(1); pt = np.hypot(S[:, 0], S[:, 1])
    return np.arcsinh(S[:, 2] / pt), np.arctan2(S[:, 1], S[:, 0])


def score(M, ds, P4, nn, base, order):
    """all networks on (P4, nn), each jet rebuilt about its own constituent-sum axis"""
    r = {}
    e, p = axis(P4, nn); b = dict(base); b['jet_eta'] = e.astype(np.float32); b['jet_phi'] = p.astype(np.float32)
    Jd = X.jc_dict(b, P4, nn)
    for name in ('ParT_kin', 'PN_kin'):
        L = M.jc_logits(name, Jd); r[name.replace('_kin', '_jc')] = L[:, TBQQ] - L[:, QCD]
    if ds == 'jc' and M.massless:
        P4 = P4.copy(); P4[..., 3] = np.sqrt((P4[..., :3].astype(np.float64) ** 2).sum(-1)) * (P4[..., 3] != 0)
    for name, net in M.top.items():
        r[name] = net.score(P4, nn, order) if name == 'LNet_top' else net.score(P4, nn)
    k3 = np.concatenate([H.kin3_of(P4[s:s + 5000], nn[s:s + 5000]) for s in range(0, len(nn), 5000)])
    for name, f in M.flows.items(): r[name] = f.score(k3)
    return r


E3 = [np.linspace(0, 300, 16), np.linspace(0, 1, 21), np.linspace(0, 200, 11)]
E2 = [E3[0], E3[2]]
lo = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))


def xsel(Xf, O, kind):
    """feature sets (h4_top_wmass._features columns): MLP3 = (m, tau32, mW); MLPx = side effects of the edit without any
    N-subjettiness (m, mW, mSD, smallest pair mass, 3-subjet mass, multiplicity, sorted pair masses, subjet pT fractions,
    subjet masses, pair dR); MLPxT = MLPx + tau1, tau2, tau3, tau21, tau32"""
    if kind == 'MLP3': return np.stack([O['m'], O['tau32'], O['mW']], 1)
    base = np.column_stack([O['m'], O['mW'], Xf[:, 0], Xf[:, 5], Xf[:, 6], Xf[:, 7], Xf[:, 8:20]])
    return base if kind == 'MLPx' else np.column_stack([base, Xf[:, 1:5], O['tau32']])


def fit_refs(Xf, O, y):
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    X3 = np.stack([O['m'], O['tau32'], O['mW']], 1)
    R = dict(LR3=H._lr_fit(X3, y, E3), LR2=H._lr_fit(np.stack([O['m'], O['mW']], 1), y, E2)); info = dict(n=int(len(y)))
    for nm in ('MLP3', 'MLPx', 'MLPxT'):
        Z = xsel(Xf, O, nm); sc = StandardScaler().fit(Z)
        mlp = MLPClassifier(hidden_layer_sizes=(64, 64), early_stopping=True, random_state=0, max_iter=200).fit(sc.transform(Z), y)
        R[nm] = (mlp, sc); info[nm] = dict(iterations=int(mlp.n_iter_), internal_val_acc=float(mlp.best_validation_score_))
    return R, info


def eval_refs(R, Xf, O):
    X3 = np.stack([O['m'], O['tau32'], O['mW']], 1)
    out = dict(LR3=H._lr_eval(R['LR3'], X3, E3), LR2=H._lr_eval(R['LR2'], np.stack([O['m'], O['mW']], 1), E2))
    for nm in ('MLP3', 'MLPx', 'MLPxT'):
        mlp, sc = R[nm]; out[nm] = lo(mlp.predict_proba(sc.transform(xsel(Xf, O, nm)))[:, 1])
    return out


# ================================================================================================================ samples
def load_top():
    S = H._sample(PER, 9733); nn = np.minimum(S['jet'][:, 5].astype(np.int64), S['full'].shape[1])
    P4 = H._identity(S['full'][..., :4].astype(np.float32), nn).astype(np.float32); y = S['label'].astype(int)
    log(f'top sample: {len(nn)} jets ({y.sum()} top), seed 9733')
    return dict(P4=P4, nn=nn, y=y, base=X.top_base(P4, nn), info=dict(n=int(len(nn)), n_top=int(y.sum()), seed=9733, split='test'))


def load_jc():
    from interp.part_core import cat, kin
    Jt, _, it = X.jc_read('Tbqq', 100, PER, 9733); Jq, _, iq = X.jc_read('QCD', 100, PER, 9734)
    J = cat([Jt, Jq]); P4raw, nn = X.jc_P4(J); P4 = H._identity(P4raw, nn).astype(np.float32)
    J['pmass'] = kin(J)[3].astype(np.float32); y = (J['label'] == TBQQ).astype(int)
    return dict(P4=P4, nn=nn, y=y, base=J, info=dict(Tbqq=it, QCD=iq))


# ================================================================================================================ stats
def perm_p2(x, rng, n=NPERM):
    x = np.asarray(x, float); obs = abs(x.mean()); out = 0
    for s0 in range(0, n, 500):
        k = min(500, n - s0); sg = rng.choice([-1.0, 1.0], (k, len(x))); out += int(np.sum(np.abs((sg * x).mean(1)) >= obs))
    return float((out + 1) / (n + 1))


def c2st(Xa, Xb, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    Xc = np.concatenate([Xa, Xb]); yy = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]; p = np.zeros(len(yy))
    for tr, te in StratifiedKFold(2, shuffle=True, random_state=seed).split(Xc, yy):
        p[te] = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(Xc[tr], yy[tr]).predict_proba(Xc[te])[:, 1]
    return float(H._auc(p, yy.astype(int))), p[:len(Xa)]


def matched_c2st(Xe, Oe, Xp, Op):
    """edited jets vs nearest natural pool jets in standardized (m, tau32, mW) (with replacement); AUC on the 20 features"""
    from scipy.spatial import cKDTree
    k3 = lambda O: np.stack([O['m'], O['tau32'], O['mW']], 1)
    Zp = k3(Op); mu, sd = Zp.mean(0), Zp.std(0) + 1e-9; tree = cKDTree((Zp - mu) / sd)
    _, nb = tree.query((k3(Oe) - mu) / sd, k=1)
    auc, pe = c2st(Xe, Xp[nb])
    return auc, pe   # pe: out-of-fold probability that each edited jet is 'natural'-unlike (label 0 = edited, so low = edited-like)


# ================================================================================================================ main
def run_sample(ds, S, M, REF, POOL):
    nn, y = S['nn'], S['y']; P4id = S['P4']
    if ds == 'top': P4id = sort_pt(P4id, nn); S['P4'] = P4id
    X0, O0 = H._features(P4id, nn); log(f'[{ds}] unedited observables on {len(nn)} jets')
    order = np.random.default_rng(605).permutation(len(nn))
    sc = {'id': score(M, ds, P4id, nn, S['base'], order)}; sc['id'].update(eval_refs(REF, X0, O0))
    obs = {'id': (X0, O0)}
    perf = {}
    for k, s in sc['id'].items():
        st, sb = s[y == 1], s[y == 0]
        perf[k] = dict(AUC=float(H._auc(s, y)), acc_binary=float(((s > 0) == (y == 1)).mean()) if k != 'LR2' else None, wp={})
        for nm, q in (('eS30', 0.7), ('eS50', 0.5)):
            thr = float(np.quantile(st, q)); eb = float((sb >= thr).mean())
            perf[k]['wp'][nm] = dict(thr=thr, eps_B=eb, n_bkg_pass=int((sb >= thr).sum()), rejection=float(1 / eb) if eb > 0 else None)
    log(f'[{ds}] unedited: ' + ', '.join(f"{k} AUC {v['AUC']:.4f} acc {v['acc_binary']}" for k, v in perf.items()))
    log(f'[{ds}] editing')
    E = run_edits(P4id, nn)
    acc = {}; info = {}
    for (s, kind), e in E.items():
        Pv = sort_pt(e['P'], nn) if ds == 'top' else e['P']
        r = np.flatnonzero(e['conv']); Xv, Ov = X0.copy(), {k: v.copy() for k, v in O0.items()}
        if len(r):
            xr, orr = H._features(np.ascontiguousarray(Pv[r]), nn[r]); Xv[r] = xr
            for k in Ov: Ov[k][r] = orr[k]
        ok = e['conv'] & (np.abs(Ov['m'] - O0['m']) < 0.5) & (np.abs(Ov['mW'] - O0['mW']) < 2)
        acc[(s, kind)] = ok; obs[(s, kind)] = (Xv, Ov)
        v = sc[(s, kind)] = score(M, ds, Pv, nn, S['base'], order); v.update(eval_refs(REF, Xv, Ov))
        info[f'{s}|{kind}'] = dict(converged=int(e['conv'].sum()), accepted=int(ok.sum()), reasons=e['reasons'],
                                   D_median=float(np.median(e['D'][e['conv']])) if e['conv'].any() else None)
        log(f'[{ds}] s={s} {kind}: accepted {ok.sum()} / converged {e["conv"].sum()}; scored')
    # statistics
    rng = np.random.default_rng(7); out = dict(unedited=perf, edit_info=info, s={})
    for s in SS:
        base_ok = acc[(s, 'S')] & acc[(s, 'c1')]
        for cls, cname in ((1, 'top'), (0, 'qcd')):
            rows = np.flatnonzero(base_ok & (y == cls)); n = len(rows); key = f'{s}|{cname}'
            R = dict(n=int(n), n_class=int((y == cls).sum()), n_c2_acc=int((acc[(s, 'c2')][rows]).sum()) if n else 0)
            if n < 10: out['s'][key] = R; continue
            c2r = rows[acc[(s, 'c2')][rows]]
            R['observables'] = {}
            for q in ('m', 'mSD', 'tau32', 'tau21', 'tau3', 'tau2', 'mW', 'n'):
                R['observables'][q] = dict(id_median=float(np.median(O0[q][rows])))
                for kind in VAR:
                    rr = c2r if kind == 'c2' else rows; dq = obs[(s, kind)][1][q][rr] - O0[q][rr]
                    R['observables'][q][kind] = dict(median_after=float(np.median(obs[(s, kind)][1][q][rr])), mean_change=float(dq.mean()),
                                                    median_change=float(np.median(dq)), median_abs_change=float(np.median(np.abs(dq))))
            R['observables']['subjet_mass_sum'] = {kind: float(np.median(obs[(s, kind)][0][rows, 14:17].sum(1) - X0[rows, 14:17].sum(1))) for kind in VAR}
            R['D_median'] = {kind: float(np.median(E[(s, kind)]['D'][c2r if kind == 'c2' else rows])) for kind in VAR}
            bi = rng.integers(0, n, (B, n)); R['models'] = {}
            refx = {r_: (sc[(s, 'S')][r_] - sc[(s, 'c1')][r_])[rows] for r_ in ('LR3', 'MLP3', 'MLPx', 'MLPxT')}
            for mdl in NETS + REFS:
                if mdl not in sc['id']: continue
                d = {kind: (sc[(s, kind)][mdl] - sc['id'][mdl]) for kind in VAR}
                x = (d['S'] - d['c1'])[rows]
                Mm = dict(S_minus_c1=float(x.mean()), ci95=np.percentile(x[bi].mean(1), [2.5, 97.5]).tolist(), p_perm=perm_p2(x, rng))
                for kind in VAR:
                    rr = c2r if kind == 'c2' else rows; z = d[kind][rr]; bb = rng.integers(0, len(rr), (B, len(rr))) if len(rr) else None
                    Mm[f'd{kind}'] = float(z.mean()) if len(rr) else None
                    Mm[f'd{kind}_ci95'] = np.percentile(z[bb].mean(1), [2.5, 97.5]).tolist() if len(rr) else None
                for r_, rx in refx.items():
                    if mdl == r_: continue
                    den = rx[bi].mean(1)
                    Mm[f'ratio_{r_}'] = float(x.mean() / rx.mean()) if abs(rx.mean()) > 1e-9 else None
                    Mm[f'ratio_{r_}_ci95'] = np.percentile(x[bi].mean(1) / np.where(np.abs(den) > 1e-9, den, np.nan), [2.5, 97.5]).tolist()
                if mdl in NETS:   # H0_side: the network's S - c1 equals the side-effect reference's (per-jet sign-flip, two-sided)
                    Mm['p_perm_vs_MLPx'] = perm_p2(x - refx['MLPx'], rng)
                Mm['calls'] = {}
                if mdl in perf and perf[mdl]['wp']:
                    for nm in ('eS30', 'eS50'):
                        thr = perf[mdl]['wp'][nm]['thr']; c0 = sc['id'][mdl][rows] >= thr
                        cS = sc[(s, 'S')][mdl][rows] >= thr; c1 = sc[(s, 'c1')][mdl][rows] >= thr; c3 = sc[(s, 'c3')][mdl][rows] >= thr
                        Mm['calls'][nm] = dict(pass_id=float(c0.mean()), pass_S=float(cS.mean()), pass_c1=float(c1.mean()), pass_c3=float(c3.mean()),
                                               changed_S=float((cS != c0).mean()), changed_c1=float((c1 != c0).mean()), changed_c3=float((c3 != c0).mean()),
                                               mcnemar_S_vs_c1=H._mcnemar(c1, cS), mcnemar_S_vs_id=H._mcnemar(c0, cS))
                R['models'][mdl] = Mm
            # C2ST: edited jets vs natural pool jets of the same class matched in (m, tau32, mW)
            Xp, Op, yp = POOL; pm = yp == cls; Opc = {k: v[pm] for k, v in Op.items()}
            sub = rows if len(rows) <= 1500 else np.sort(rng.permutation(rows)[:1500])
            C = {kind: matched_c2st(obs[kv][0][sub], {k: v[sub] for k, v in obs[kv][1].items()}, Xp[pm], Opc)
                 for kind, kv in (('id', 'id'), ('S', (s, 'S')), ('c1', (s, 'c1')))}
            R['c2st'] = {kind: c[0] for kind, c in C.items()}
            # natural-looking half of the S jets: out-of-fold P(natural) of each edited jet above its median (edited = class 0)
            pS = C['S'][1]; nat = sub[pS >= np.median(pS)]
            if len(nat) >= 10:
                bh = rng.integers(0, len(nat), (B, len(nat))); R['natural_half'] = dict(n=int(len(nat)), models={})
                R['natural_half']['dtau32_mean'] = float((obs[(s, 'S')][1]['tau32'][nat] - O0['tau32'][nat]).mean())
                for mdl in NETS + ['LR3', 'MLP3', 'MLPx', 'MLPxT']:
                    z = (sc[(s, 'S')][mdl] - sc[(s, 'c1')][mdl])[nat]
                    R['natural_half']['models'][mdl] = dict(S_minus_c1=float(z.mean()), ci95=np.percentile(z[bh].mean(1), [2.5, 97.5]).tolist())
            out['s'][key] = R
            log(f'[{ds}] {key}: n {n}; dtau32 S {R["observables"]["tau32"]["S"]["mean_change"]:+.4f} c1 {R["observables"]["tau32"]["c1"]["mean_change"]:+.4f}; '
                f'C2ST {R["c2st"]}')
            for mdl, Mm in R['models'].items():
                log(f"    {mdl:9s} S-c1 {Mm['S_minus_c1']:+.3f} {np.round(Mm['ci95'], 3)} p {Mm['p_perm']:.3g} dS {Mm['dS']:+.3f} dc1 {Mm['dc1']:+.3f} "
                    f"dc2 {Mm['dc2']} dc3 {Mm['dc3']:+.3f} xLR3 {Mm.get('ratio_LR3')} xMLPx {Mm.get('ratio_MLPx')} xMLPxT {Mm.get('ratio_MLPxT')}")
    perjet = {f'{ds}|{"id" if k == "id" else f"{k[0]}|{k[1]}"}|{m}': np.asarray(v, np.float32) for k, d in sc.items() for m, v in d.items()}
    perjet[f'{ds}|y'] = y
    for (s, kind), ok in acc.items(): perjet[f'{ds}|acc|{s}|{kind}'] = ok
    for k, (Xo, Oo) in obs.items():
        nm = 'id' if k == 'id' else f'{k[0]}|{k[1]}'
        perjet[f'{ds}|tau32|{nm}'] = Oo['tau32'].astype(np.float32)
    return out, perjet


def main():
    RES.mkdir(parents=True, exist_ok=True)
    z = np.array([[1.0, 0, 5, 6], [0, 1.0, 2, 3], [0.5, 0.5, 1, 2]]); H._tau32(z[:, 0].copy(), z[:, 1].copy(), z[:, 2].copy(), z[:, 3].copy(), 3)
    log(f'H25 start; smoke={SMOKE}, per class {PER}, ref jets per class {X.NREF}, s {SS}')
    SAMP = {'top': load_top(), 'jc': load_jc()}
    REF, POOL, rinfo = {}, {}, {}
    for ds, fn in (('jc', X.ref_train_jc), ('top', X.ref_train_top)):
        Xr, Or, yr = fn(); REF[ds], rinfo[ds] = fit_refs(Xr, Or, yr); POOL[ds] = (Xr, Or, yr)
        log(f'refs {ds}: {rinfo[ds]}')
    M = X.Models()
    P = SAMP['top']['P4'].astype(np.float64); real = np.arange(P.shape[1])[None, :] < SAMP['top']['nn'][:, None]
    m2 = (P[..., 3] ** 2 - (P[..., :3] ** 2).sum(-1))[real]; M.massless = bool(np.mean(np.sqrt(np.maximum(m2, 0)) > 0.1) < 0.01)
    log(f'Top constituents massless: {M.massless}')
    R = dict(node='H25', smoke=SMOKE, s_values=SS, samples={ds: S['info'] for ds, S in SAMP.items()}, references=rinfo, sets={})
    PJ = {}
    for ds, S in SAMP.items():
        R['sets'][ds], pj = run_sample(ds, S, M, REF[ds], POOL[ds]); PJ.update(pj)
    tests = [(ds, k, m) for ds in R['sets'] for k, r in R['sets'][ds]['s'].items() for m in r.get('models', {}) if m in NETS]
    for pk, hk in (('p_perm', 'p_holm'), ('p_perm_vs_MLPx', 'p_holm_vs_MLPx')):
        adj = H._holm([R['sets'][ds]['s'][k]['models'][m][pk] for ds, k, m in tests]) if tests else []
        for (ds, k, m), pa in zip(tests, adj): R['sets'][ds]['s'][k]['models'][m][hk] = float(pa)
    R['holm_family'] = f'{len(tests)} tests: networks x s x samples x classes, S - c1 two-sided sign-flip permutation ({NPERM} perms)'
    (RES / f'{TAG}_top_tau32.json').write_text(json.dumps(R, indent=1, default=float)); log(f'wrote {TAG}_top_tau32.json')
    np.savez_compressed(RES / f'{TAG}_perjet.npz', **PJ)
    try: fig(R)
    except Exception as ex: log('figure failed:', repr(ex))


def fig(R):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    show = ['ParT_top', 'PN_top', 'LNet_top', 'PFN_top', 'EFN_top', 'ParT_jc', 'PN_jc', 'LR3', 'MLP3', 'MLPx', 'LR2']
    lab = {'ParT_top': 'Top ParT', 'PN_top': 'Top PN', 'LNet_top': 'Top LorentzNet', 'PFN_top': 'Top PFN', 'EFN_top': 'Top EFN',
           'ParT_jc': 'JetClass ParT_kin', 'PN_jc': 'JetClass PN_kin', 'LR3': 'LR(m, τ32, mW) (pos.)', 'MLP3': 'MLP(m, τ32, mW) (pos.)', 'MLPx': 'MLP side effects, no τ', 'LR2': 'LR(m, mW) (neg.)'}
    col = {'ParT_top': '#c05621', 'PN_top': '#dd6b20', 'LNet_top': '#975a16', 'PFN_top': '#b7791f', 'EFN_top': '#d69e2e',
           'ParT_jc': '#2b6cb0', 'PN_jc': '#4299e1', 'LR3': 'k', 'MLP3': '#2f855a', 'MLPx': '#805ad5', 'LR2': '0.55'}
    fig_, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for i, (ds, dsn) in enumerate((('top', 'Top Landscape jets'), ('jc', 'JetClass jets (550-650 GeV)'))):
        for j, (cn, cnn) in enumerate((('top', 'top jets'), ('qcd', 'QCD jets'))):
            ax = axes[i, j]; sv = [s for s in R['s_values'] if f'{s}|{cn}' in R['sets'][ds]['s'] and 'models' in R['sets'][ds]['s'][f'{s}|{cn}']]
            for k, m in enumerate(show):
                xs, ys, lo_, hi_ = [], [], [], []
                for s in sv:
                    Mm = R['sets'][ds]['s'][f'{s}|{cn}']['models'].get(m)
                    if Mm is None: continue
                    xs.append(s * (1 + 0.012 * (k - 4))); ys.append(Mm['S_minus_c1']); lo_.append(Mm['ci95'][0]); hi_.append(Mm['ci95'][1])
                if not xs: continue
                ys, lo_, hi_ = map(np.array, (ys, lo_, hi_))
                ax.errorbar(xs, ys, yerr=[ys - lo_, hi_ - ys], marker='s' if m.startswith(('LR', 'MLP')) else 'o', ms=4, lw=2.2 if m == 'LR3' else 1.2,
                            ls='--' if m.startswith(('LR', 'MLP')) else '-', color=col[m], capsize=2, label=lab[m])
            ax.axhline(0, color='0.6', lw=0.8); ax.axvline(1, color='0.85', lw=0.8); ax.set_xscale('log')
            ax.set_xticks(list(R['s_values'])); ax.set_xticklabels([str(s) for s in R['s_values']]); ax.minorticks_off()
            dt_ = [R['sets'][ds]['s'][f'{s}|{cn}']['observables']['tau32']['S']['mean_change'] for s in sv]
            n_ = [R['sets'][ds]['s'][f'{s}|{cn}']['n'] for s in sv]
            ax.set_title(f'{dsn}, {cnn}; mean Δτ32 ' + ', '.join(f'{d:+.2f}' for d in dt_) + f'\n(n = {min(n_) if n_ else 0}-{max(n_) if n_ else 0} jets per s)', fontsize=9, loc='left')
            if j == 0: ax.set_ylabel('Δ top log-odds, subjet-width edit\nminus rotation control (S − c1)')
            if i == 1: ax.set_xlabel('subjet angular scale factor s (s < 1: narrower subjets, lower τ32)')
    axes[0, 0].legend(fontsize=7.5, loc='best')
    fig_.suptitle('Top-score change when subjets are made narrower or wider at fixed jet mass and fixed W-pair mass', fontsize=10)
    fig_.tight_layout(); fig_.savefig(RES / f'{TAG}_tau32.png', dpi=130); log(f'wrote {TAG}_tau32.png')


if __name__ == '__main__':
    if '--fig' in sys.argv: fig(json.loads((RES / f'{TAG}_top_tau32.json').read_text()))
    else: main()
