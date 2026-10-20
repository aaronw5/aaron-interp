"""H31 (loop D, new D claim extending D6 / D11 / D13): do top taggers use the position of the three prongs on the top Dalitz
plane, i.e. the W decay angle cos theta* (equivalently the split of m_bq1^2 + m_bq2^2), at fixed m_top and m_W?

Samples (random rows, never first-k)
  top  Top Landscape test split, 2,000 top + 2,000 QCD (h4_top_wmass._sample rows, seed 3101); jets rebuilt through
       (pT, eta, phi, m) (identity path) and sorted by pT.
  jc   JetClass test TTBar_100 (Tbqq) and ZJetsToNuNu_100 (QCD), stored 550 < jet_pt < 650 GeV, |jet_eta| < 2, >= 3 particles,
       2,000 per class (rng 3101 / 3102 among eligible rows; h19_wpair_cross.jc_read; eta = jet_eta + sign(jet_eta) deta).
Dalitz variables (per jet): three exclusive-kT subjets (E-scheme, h4_top_wmass.kt3_batch); W pair (a, b) = the subjet pair with
  mass closest to 80.4 GeV; c = the third subjet ("b"). mW = m_ab; m_bq_hi / m_bq_lo = max / min(m_ac, m_bc);
  |cos theta*| = |n . u| with n = direction of subjet a and u = direction of subjet c, both in the W rest frame
  (u is the W flight direction in the top rest frame, up to sign). Quark flavour is unknown, so only |cos theta*| is defined.
Edit D(dc), dc in {-0.6, -0.3, +0.3, +0.6}: boost every constituent of a and b to the W rest frame; n+ = the W daughter with
  n+ . u >= 0, c0 = n+ . u; target c1 = clip(c0 + dc, 0, 0.995); rotate all constituents of a and b rigidly (one rotation R about
  k = n+ x u by alpha = arccos c0 - arccos c1); boost back. The W four-momentum, mW, m_top, each subjet's internal invariant
  structure (all intra-subjet invariant masses) and N are unchanged; |cos theta*| -> c1 and m_bq_hi - m_bq_lo changes.
  Accepted if, after re-clustering the edited jet, |dm| < 0.5 GeV and |dmW| < 2 GeV.
Controls (same jets, same rotation angle |alpha|)
  cphi  rotation of the W system about u (the W flight direction in the top rest frame) by |alpha| (random sign): |cos theta*|,
        m_ac and m_bc are unchanged (subjet c's four-momentum is invariant under it); constituent positions move.
  cr    rotation about a random axis (uniform on the sphere) by |alpha|.
  TN    rigid (eta, phi) translation of the whole jet by D's pT-weighted mean constituent displacement, random direction.
  Every edited jet is rebuilt about its own constituent-sum axis for every network (H25 score path).
Models: Top Landscape ParT_kin / PN_kin (H4 snapshots), LorentzNet (released), PFN, EFN (H2) on both samples; JetClass ParT_kin,
  PN_kin on both samples, ParT_full, PN_full on the JetClass sample (score = L(Tbqq) - L(QCD)).
References (MLP(64, 64), retrained per dataset on disjoint jets: Top val 25k per class rng 4194; JetClass TTBar_101 /
  ZJetsToNuNu_101 in-window 25k per class rng 4193 / 4194): MLP3 (m, mW, tau32); MLP5 (+ m_bq_hi, m_bq_lo); MLP5c (+ |cos theta*|);
  MLPx (edit side effects without pair masses: m, mW, mSD, tau1-3, tau21, tau32, multiplicity, sorted subjet pT fractions,
  sorted subjet masses).
Metrics per sample x class x dc x model: d_v = score_v - score_id (top log-odds); D - cphi mean, 95 % bootstrap CI (B = 2000),
  sign-flip permutation p (two-sided) for H0: D - cphi = 0 and per-jet against MLP3 and MLP5; Holm over networks x dc x samples
  x classes; ratio to MLP5, MLP5c, MLP3, MLPx (bootstrap CI); % calls changed at eps_S 0.5 / 0.3 (thresholds from the sample's
  unedited tops) and McNemar D vs cphi; dose = Spearman of the D - cphi means against dc; C2ST (edited vs natural pool jets of
  the same class matched in standardized (m, tau32, mW, |cos theta*|); HGB on 20 substructure features, 2-fold AUC).
Run: ./sync.sh && ./part_job.sh d-h31-run 'python -m interp.h31_top_dalitz'    (H31_SMOKE=1: 150 per class, 2,000 ref jets)
Outputs: $AI_RESULTS/physics/h31_top_dalitz.json, h31_perjet.npz, h31_dalitz.png
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
from interp import h25_top_tau32 as H25       # noqa: E402

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
SMOKE = bool(os.environ.get('H31_SMOKE'))
RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'physics'
TAG = 'h31_smoke' if SMOKE else 'h31'
PER = 150 if SMOKE else 2000
X.NREF = 2000 if SMOKE else 25000
B = 300 if SMOKE else 2000
NPERM = 500 if SMOKE else 5000
DCS = (-0.6, -0.3, 0.3, 0.6)
VAR = ('D', 'cphi', 'cr', 'TN')
NETS = ['ParT_top', 'PN_top', 'LNet_top', 'PFN_top', 'EFN_top', 'ParT_kin', 'PN_kin', 'ParT_full', 'PN_full']
REFS = ['MLP3', 'MLP5', 'MLP5c', 'MLPx']
TBQQ, QCD = 8, 0
CMAX = 0.995


# ================================================================================================================ kinematics
def boost(P, b):
    """four-vectors P (..., 4) = (px, py, pz, E) seen in a frame moving with velocity b (3,)"""
    b2 = float(b @ b)
    if b2 < 1e-16: return P.copy()
    g = 1.0 / np.sqrt(1.0 - b2); bp = P[..., :3] @ b
    out = np.empty_like(P)
    out[..., :3] = P[..., :3] + (((g - 1.0) * bp / b2) - g * P[..., 3])[..., None] * b
    out[..., 3] = g * (P[..., 3] - bp)
    return out


def rodrigues(k, a):
    k = k / np.linalg.norm(k); K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


def m2(v):
    return v[..., 3] ** 2 - (v[..., :3] ** 2).sum(-1)


def dalitz_from(F, nn, LAB):
    """vectorised Dalitz variables from kT-3 labels: mW, m_bq_hi, m_bq_lo, |cos theta*|, plus (a, b, c) indices"""
    J, L = LAB.shape; real = np.arange(L)[None, :] < nn[:, None]
    oh = np.stack([(LAB == k) & real for k in range(3)], -1).astype(np.float64)
    S = np.einsum('jlk,jld->jkd', oh, F[:, :L].astype(np.float64))                    # (J, 3, 4) subjet sums
    pairs = [(0, 1), (0, 2), (1, 2)]
    pm = np.stack([np.sqrt(np.maximum(m2(S[:, i] + S[:, j]), 0)) for i, j in pairs], 1)
    w = np.argmin(np.abs(pm - H.MWREF), 1); ab = np.array(pairs)[w]; a, b = ab[:, 0], ab[:, 1]; c = 3 - a - b
    r = np.arange(J); Va, Vb, Vc = S[r, a], S[r, b], S[r, c]; W = Va + Vb
    mW = pm[r, w]; mac = np.sqrt(np.maximum(m2(Va + Vc), 0)); mbc = np.sqrt(np.maximum(m2(Vb + Vc), 0))
    cos = np.zeros(J)
    for j in range(J):
        if nn[j] < 3 or W[j, 3] <= 0: continue
        be = W[j, :3] / W[j, 3]; na = boost(Va[j], be)[:3]; uc = boost(Vc[j], be)[:3]
        den = np.linalg.norm(na) * np.linalg.norm(uc); cos[j] = abs(na @ uc) / den if den > 0 else 0.0
    mt = np.sqrt(np.maximum(m2(S.sum(1)), 0))
    return dict(mW=mW, mbq_hi=np.maximum(mac, mbc), mbq_lo=np.minimum(mac, mbc), cos=cos, m3=mt)


def feats(P4, nn):
    """H4 substructure features (one kT-3 clustering shared with the Dalitz variables)"""
    O = H._obs(P4, nn); F = np.ascontiguousarray(P4.astype(np.float64)); LAB = H.kt3_batch(F, nn); sf = H._subfeats(F, nn, LAB)
    Xf = np.column_stack([O['mSD'], O['tau1'], O['tau2'], O['tau3'], O['tau21'], O['m12min'], O['m123'], O['n'], sf])
    Dz = dalitz_from(F, nn, LAB); O = dict(O); O['mWk'] = Dz['mW']; O['mbq_hi'] = Dz['mbq_hi']; O['mbq_lo'] = Dz['mbq_lo']; O['cos'] = Dz['cos']
    return Xf, O


# ================================================================================================================ edit
def _eta_phi(P):
    pt = np.hypot(P[:, 0], P[:, 1]); return np.arcsinh(P[:, 2] / np.maximum(pt, 1e-12)), np.arctan2(P[:, 1], P[:, 0]), pt


def edit_jet(P, lab, dc, seed):
    """D, cphi, cr, TN for one jet and one dc: {kind: (P_new or None, info)}"""
    P = np.asarray(P, np.float64); rng = np.random.default_rng(seed); out = {}
    sel = [lab == k for k in range(3)]
    if min(s.sum() for s in sel) < 1: return {k: (None, dict(reason='empty_subjet')) for k in VAR}
    V = [P[s].sum(0) for s in sel]; pairs = [(0, 1), (0, 2), (1, 2)]
    pm = [np.sqrt(max(m2(V[i] + V[j]), 0)) for i, j in pairs]; a, b = pairs[int(np.argmin(np.abs(np.array(pm) - H.MWREF)))]; c = 3 - a - b
    W = V[a] + V[b]; be = W[:3] / W[3]; inW = sel[a] | sel[b]
    Pr = boost(P[inW], be); na = boost(V[a], be)[:3]; u = boost(V[c], be)[:3]
    na = na / np.linalg.norm(na); u = u / np.linalg.norm(u); cs = float(na @ u); npl = na if cs >= 0 else -na; c0 = abs(cs)
    c1 = float(np.clip(c0 + dc, 0.0, CMAX)); al = float(np.arccos(min(c0, 1.0)) - np.arccos(c1))
    k = np.cross(npl, u)
    if np.linalg.norm(k) < 1e-9: return {kk: (None, dict(reason='collinear')) for kk in VAR}
    eta0, phi0, pt0 = _eta_phi(P)

    def apply(R):
        Q = Pr.copy(); Q[:, :3] = Q[:, :3] @ R.T; Pn = P.copy(); Pn[inW] = boost(Q, -be); return Pn
    Pd = apply(rodrigues(k, al)); out['D'] = (Pd, dict(reason='ok', c0=c0, c1=c1, alpha=al))
    s = rng.choice([-1.0, 1.0]); out['cphi'] = (apply(rodrigues(u, s * abs(al))), dict(reason='ok', alpha=abs(al)))
    ax = rng.normal(size=3); out['cr'] = (apply(rodrigues(ax, abs(al))), dict(reason='ok', alpha=abs(al)))
    e1, p1, _ = _eta_phi(Pd); dsp = np.hypot(e1 - eta0, (p1 - phi0 + np.pi) % (2 * np.pi) - np.pi)
    DS = float((pt0 * dsp).sum() / pt0.sum()); th = rng.uniform(0, 2 * np.pi)
    m = np.sqrt(np.maximum(m2(P), 0)); e2, p2 = eta0 + DS * np.cos(th), phi0 + DS * np.sin(th)
    out['TN'] = (np.stack([pt0 * np.cos(p2), pt0 * np.sin(p2), pt0 * np.sinh(e2), np.sqrt((pt0 * np.cosh(e2)) ** 2 + m ** 2)], -1), dict(reason='ok', D=DS))
    out['D'][1]['D'] = DS
    return out


def run_edits(P4, nn):
    rows = np.flatnonzero(nn >= 3); F = np.ascontiguousarray(P4[rows].astype(np.float64))
    LAB = H.kt3_batch(F, nn[rows]); log(f'  kT subjets for {len(rows)} jets')
    E = {(dc, kind): dict(P=P4.copy(), conv=np.zeros(len(nn), bool), c0=np.full(len(nn), np.nan), c1=np.full(len(nn), np.nan),
                          alpha=np.zeros(len(nn)), D=np.zeros(len(nn)), reasons={}) for dc in DCS for kind in VAR}
    for ii, j in enumerate(rows):
        n = nn[j]
        for i, dc in enumerate(DCS):
            r = edit_jet(P4[j, :n], LAB[ii, :n], dc, 3101 * 1000 + int(j) * 10 + i)
            for kind, (Pn, inf) in r.items():
                e = E[(dc, kind)]; e['reasons'][inf['reason']] = e['reasons'].get(inf['reason'], 0) + 1
                if Pn is None: continue
                e['P'][j, :n] = Pn.astype(np.float32); e['conv'][j] = True
                e['alpha'][j] = inf.get('alpha', 0.0); e['D'][j] = inf.get('D', 0.0)
                if kind == 'D': e['c0'][j] = inf['c0']; e['c1'][j] = inf['c1']
    for (dc, kind), e in E.items(): log(f'  dc={dc:+.1f} {kind}: edited {e["conv"].sum()}/{len(nn)} {e["reasons"]}')
    return E


# ================================================================================================================ models
def score(M, ds, P4, nn, base, order):
    """all networks, each jet rebuilt about its own constituent-sum axis (H25 path; full JetClass nets on the jc sample only)"""
    r = {}
    e, p = H25.axis(P4, nn); b = dict(base); b['jet_eta'] = e.astype(np.float32); b['jet_phi'] = p.astype(np.float32)
    Jd = X.jc_dict(b, P4, nn)
    for name in ('ParT_kin', 'PN_kin') + (('ParT_full', 'PN_full') if ds == 'jc' else ()):
        L = M.jc_logits(name, Jd); r[name] = L[:, TBQQ] - L[:, QCD]
    if ds == 'jc' and M.massless:
        P4 = P4.copy(); P4[..., 3] = np.sqrt((P4[..., :3].astype(np.float64) ** 2).sum(-1)) * (P4[..., 3] != 0)
    for name, net in M.top.items():
        r[name] = net.score(P4, nn, order) if name == 'LNet_top' else net.score(P4, nn)
    k3 = np.concatenate([H.kin3_of(P4[s:s + 5000], nn[s:s + 5000]) for s in range(0, len(nn), 5000)])
    for name, f in M.flows.items(): r[name] = f.score(k3)
    return r


lo = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))


def xsel(Xf, O, kind):
    base3 = [O['m'], O['mWk'], O['tau32']]
    if kind == 'MLP3': return np.column_stack(base3)
    if kind == 'MLP5': return np.column_stack(base3 + [O['mbq_hi'], O['mbq_lo']])
    if kind == 'MLP5c': return np.column_stack(base3 + [O['mbq_hi'], O['mbq_lo'], O['cos']])
    return np.column_stack(base3 + [Xf[:, 0], Xf[:, 1], Xf[:, 2], Xf[:, 3], Xf[:, 4], Xf[:, 7], Xf[:, 11:17]])   # MLPx


def fit_refs(Xf, O, y):
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    R, info = {}, dict(n=int(len(y)))
    for nm in REFS:
        Z = xsel(Xf, O, nm); sc = StandardScaler().fit(Z)
        mlp = MLPClassifier(hidden_layer_sizes=(64, 64), early_stopping=True, random_state=0, max_iter=200).fit(sc.transform(Z), y)
        R[nm] = (mlp, sc); info[nm] = dict(iterations=int(mlp.n_iter_), internal_val_acc=float(mlp.best_validation_score_))
    return R, info


def eval_refs(R, Xf, O):
    return {nm: lo(mlp.predict_proba(sc.transform(xsel(Xf, O, nm)))[:, 1]) for nm, (mlp, sc) in R.items()}


def ref_train(ds):
    from interp import data_top as dt
    if ds == 'jc':
        Xs, Os, ys = [], [], []
        for c, lab in (('Tbqq', 1), ('QCD', 0)):
            J, _, _ = X.jc_read(c, 101, X.NREF, 4193 + lab); P4, nn = X.jc_P4(J); Xf, O = feats(P4, nn)
            Xs.append(Xf); Os.append(O); ys.append(np.full(len(nn), lab))
        return np.concatenate(Xs), {k: np.concatenate([o[k] for o in Os]) for k in Os[0]}, np.concatenate(ys)
    yv = np.load(dt.DST / 'val' / 'label.npy'); rv = np.random.default_rng(4194)
    iv = np.sort(np.concatenate([rv.choice(np.flatnonzero(yv == c), X.NREF, replace=False) for c in (0, 1)]))
    fv = np.load(dt.DST / 'val' / 'full.npy'); Pv = np.asarray(fv[iv][..., :4], np.float32); del fv
    nv = np.minimum(np.load(dt.DST / 'val' / 'jet.npy')[iv, 5].astype(np.int64), Pv.shape[1])
    Xf, O = feats(Pv, nv); return Xf, O, yv[iv].astype(int)


# ================================================================================================================ samples
def load_top():
    S = H._sample(PER, 3101); nn = np.minimum(S['jet'][:, 5].astype(np.int64), S['full'].shape[1])
    P4 = H._identity(S['full'][..., :4].astype(np.float32), nn).astype(np.float32); y = S['label'].astype(int)
    log(f'top sample: {len(nn)} jets ({y.sum()} top), seed 3101')
    return dict(P4=P4, nn=nn, y=y, base=X.top_base(P4, nn), info=dict(n=int(len(nn)), n_top=int(y.sum()), seed=3101, split='test'))


def load_jc():
    from interp.part_core import cat, kin
    Jt, _, it = X.jc_read('Tbqq', 100, PER, 3101); Jq, _, iq = X.jc_read('QCD', 100, PER, 3102)
    J = cat([Jt, Jq]); P4raw, nn = X.jc_P4(J); P4 = H._identity(P4raw, nn).astype(np.float32)
    J['pmass'] = kin(J)[3].astype(np.float32); y = (J['label'] == TBQQ).astype(int)
    return dict(P4=P4, nn=nn, y=y, base=J, info=dict(Tbqq=it, QCD=iq))


# ================================================================================================================ stats
def perm_p2(x, rng, n=NPERM):
    return H25.perm_p2(x, rng, n)


def matched_c2st(Xe, Oe, Xp, Op):
    from scipy.spatial import cKDTree
    k4 = lambda O: np.stack([O['m'], O['tau32'], O['mWk'], O['cos']], 1)
    Zp = k4(Op); mu, sd = Zp.mean(0), Zp.std(0) + 1e-9; tree = cKDTree((Zp - mu) / sd)
    _, nb = tree.query((k4(Oe) - mu) / sd, k=1)
    return H25.c2st(Xe, Xp[nb])[0]


def spearman(xs, ys):
    from scipy.stats import spearmanr
    return float(spearmanr(xs, ys)[0]) if len(xs) > 2 else None


OBSQ = ('m', 'mWk', 'mbq_hi', 'mbq_lo', 'cos', 'tau32', 'tau21', 'mSD', 'n')


def run_sample(ds, S, M, REF, POOL):
    nn, y = S['nn'], S['y']; P4id = S['P4']
    if ds == 'top': P4id = H25.sort_pt(P4id, nn); S['P4'] = P4id
    X0, O0 = feats(P4id, nn); log(f'[{ds}] unedited observables on {len(nn)} jets')
    order = np.random.default_rng(605).permutation(len(nn))
    sc = {'id': score(M, ds, P4id, nn, S['base'], order)}; sc['id'].update(eval_refs(REF, X0, O0))
    perf = {}
    for k, s in sc['id'].items():
        st, sb = s[y == 1], s[y == 0]
        perf[k] = dict(AUC=float(H._auc(s, y)), acc_binary=float(((s > 0) == (y == 1)).mean()), wp={})
        for nm, q in (('eS30', 0.7), ('eS50', 0.5)):
            thr = float(np.quantile(st, q)); eb = float((sb >= thr).mean())
            perf[k]['wp'][nm] = dict(thr=thr, eps_B=eb, n_bkg_pass=int((sb >= thr).sum()))
    log(f'[{ds}] unedited: ' + ', '.join(f"{k} AUC {v['AUC']:.4f} acc {v['acc_binary']:.4f}" for k, v in perf.items()))
    hist = {cn: np.histogram(O0['cos'][y == cl], bins=10, range=(0, 1))[0].tolist() for cl, cn in ((1, 'top'), (0, 'qcd'))}
    win = (O0['m'] > 140) & (O0['m'] < 200) & (np.abs(O0['mWk'] - 80) < 15)
    hist_win = {cn: np.histogram(O0['cos'][(y == cl) & win], bins=10, range=(0, 1))[0].tolist() for cl, cn in ((1, 'top'), (0, 'qcd'))}
    log(f'[{ds}] |cos theta*| histograms (10 bins) all: {hist}; in 140<m<200, |mW-80|<15: {hist_win}')
    log(f'[{ds}] editing')
    E = run_edits(P4id, nn)
    acc = {}; obs = {'id': (X0, O0)}; info = {}
    for (dc, kind), e in E.items():
        Pv = H25.sort_pt(e['P'], nn) if ds == 'top' else e['P']
        r = np.flatnonzero(e['conv']); Xv, Ov = X0.copy(), {k: v.copy() for k, v in O0.items()}
        if len(r):
            xr, orr = feats(np.ascontiguousarray(Pv[r]), nn[r]); Xv[r] = xr
            for k in Ov: Ov[k][r] = orr[k]
        ok = e['conv'] & (np.abs(Ov['m'] - O0['m']) < 0.5) & (np.abs(Ov['mWk'] - O0['mWk']) < 2)
        acc[(dc, kind)] = ok; obs[(dc, kind)] = (Xv, Ov)
        v = sc[(dc, kind)] = score(M, ds, Pv, nn, S['base'], order); v.update(eval_refs(REF, Xv, Ov))
        info[f'{dc}|{kind}'] = dict(edited=int(e['conv'].sum()), accepted=int(ok.sum()), reasons=e['reasons'],
                                    alpha_median=float(np.median(np.abs(e['alpha'][e['conv']]))) if e['conv'].any() else None,
                                    D_median=float(np.median(E[(dc, 'D')]['D'][e['conv']])) if e['conv'].any() else None)
        log(f'[{ds}] dc={dc:+.1f} {kind}: accepted {ok.sum()} / edited {e["conv"].sum()}; scored')
    rng = np.random.default_rng(7); out = dict(unedited=perf, cos_hist=hist, cos_hist_window=hist_win, edit_info=info, dc={})
    for dc in DCS:
        base_ok = acc[(dc, 'D')] & acc[(dc, 'cphi')]
        for cls, cname in ((1, 'top'), (0, 'qcd')):
            rows = np.flatnonzero(base_ok & (y == cls)); n = len(rows); key = f'{dc}|{cname}'
            R = dict(n=int(n), n_class=int((y == cls).sum()))
            if n < 10: out['dc'][key] = R; continue
            R['requested_target_c1_mean'] = float(np.nanmean(E[(dc, 'D')]['c1'][rows])); R['c0_mean'] = float(np.nanmean(E[(dc, 'D')]['c0'][rows]))
            R['observables'] = {}
            for q in OBSQ:
                R['observables'][q] = dict(id_mean=float(O0[q][rows].mean()))
                for kind in VAR:
                    dq = obs[(dc, kind)][1][q][rows] - O0[q][rows]
                    R['observables'][q][kind] = dict(mean_change=float(dq.mean()), median_abs_change=float(np.median(np.abs(dq))))
            bi = rng.integers(0, n, (B, n)); R['models'] = {}
            refx = {r_: (sc[(dc, 'D')][r_] - sc[(dc, 'cphi')][r_])[rows] for r_ in REFS}
            for mdl in NETS + REFS:
                if mdl not in sc['id']: continue
                d = {kind: sc[(dc, kind)][mdl] - sc['id'][mdl] for kind in VAR}
                x = (d['D'] - d['cphi'])[rows]
                Mm = dict(D_minus_cphi=float(x.mean()), ci95=np.percentile(x[bi].mean(1), [2.5, 97.5]).tolist(), p_perm=perm_p2(x, rng))
                for kind in VAR:
                    z = d[kind][rows]; Mm[f'd{kind}'] = float(z.mean()); Mm[f'd{kind}_ci95'] = np.percentile(z[bi].mean(1), [2.5, 97.5]).tolist()
                for r_, rx in refx.items():
                    if mdl == r_: continue
                    den = rx[bi].mean(1)
                    Mm[f'ratio_{r_}'] = float(x.mean() / rx.mean()) if abs(rx.mean()) > 1e-9 else None
                    Mm[f'ratio_{r_}_ci95'] = np.percentile(x[bi].mean(1) / np.where(np.abs(den) > 1e-9, den, np.nan), [2.5, 97.5]).tolist()
                if mdl in NETS:
                    Mm['p_perm_vs_MLP3'] = perm_p2(x - refx['MLP3'], rng); Mm['p_perm_vs_MLP5'] = perm_p2(x - refx['MLP5'], rng)
                Mm['calls'] = {}
                for nm in ('eS30', 'eS50'):
                    thr = perf[mdl]['wp'][nm]['thr']; c0 = sc['id'][mdl][rows] >= thr
                    cD = sc[(dc, 'D')][mdl][rows] >= thr; cP = sc[(dc, 'cphi')][mdl][rows] >= thr
                    Mm['calls'][nm] = dict(pass_id=float(c0.mean()), pass_D=float(cD.mean()), pass_cphi=float(cP.mean()),
                                           changed_D=float((cD != c0).mean()), changed_cphi=float((cP != c0).mean()),
                                           mcnemar_D_vs_cphi=H._mcnemar(cP, cD))
                R['models'][mdl] = Mm
            Xp, Op, yp = POOL; pm = yp == cls; Opc = {k: v[pm] for k, v in Op.items()}
            sub = rows if len(rows) <= 1500 else np.sort(rng.permutation(rows)[:1500])
            R['c2st'] = {kind: matched_c2st(obs[kv][0][sub], {k: v[sub] for k, v in obs[kv][1].items()}, Xp[pm], Opc)
                         for kind, kv in (('id', 'id'), ('D', (dc, 'D')), ('cphi', (dc, 'cphi')))}
            out['dc'][key] = R
            ob = R['observables']
            log(f'[{ds}] {key}: n {n}; dcos D {ob["cos"]["D"]["mean_change"]:+.3f} cphi {ob["cos"]["cphi"]["mean_change"]:+.3f} cr {ob["cos"]["cr"]["mean_change"]:+.3f}; '
                f'dmbq_hi D {ob["mbq_hi"]["D"]["mean_change"]:+.1f} cphi {ob["mbq_hi"]["cphi"]["mean_change"]:+.1f}; dtau32 D {ob["tau32"]["D"]["mean_change"]:+.3f} '
                f'cphi {ob["tau32"]["cphi"]["mean_change"]:+.3f}; C2ST {R["c2st"]}')
            for mdl, Mm in R['models'].items():
                log(f"    {mdl:9s} D-cphi {Mm['D_minus_cphi']:+.3f} {np.round(Mm['ci95'], 3)} p {Mm['p_perm']:.3g} dD {Mm['dD']:+.3f} dcphi {Mm['dcphi']:+.3f} "
                    f"dcr {Mm['dcr']:+.3f} dTN {Mm['dTN']:+.3f} xMLP5 {Mm.get('ratio_MLP5')} xMLP3 {Mm.get('ratio_MLP3')}")
    out['dose'] = {}
    for cname in ('top', 'qcd'):
        for mdl in NETS + REFS:
            xs = [dc for dc in DCS if 'models' in out['dc'][f'{dc}|{cname}'] and mdl in out['dc'][f'{dc}|{cname}']['models']]
            ys = [out['dc'][f'{dc}|{cname}']['models'][mdl]['D_minus_cphi'] for dc in xs]
            if len(xs) >= 3: out['dose'][f'{cname}|{mdl}'] = dict(spearman=spearman(xs, ys), means=ys)
    perjet = {f'{ds}|{"id" if k == "id" else f"{k[0]}|{k[1]}"}|{m}': np.asarray(v, np.float32) for k, d in sc.items() for m, v in d.items()}
    perjet[f'{ds}|y'] = y
    for (dc, kind), ok in acc.items(): perjet[f'{ds}|acc|{dc}|{kind}'] = ok
    for k, (_, Oo) in obs.items():
        nm = 'id' if k == 'id' else f'{k[0]}|{k[1]}'
        for q in ('cos', 'mbq_hi', 'mbq_lo', 'tau32'): perjet[f'{ds}|{q}|{nm}'] = Oo[q].astype(np.float32)
    return out, perjet


def selftest():
    """edit invariants on a synthetic three-prong jet: W four-momentum, m_top, intra-subjet masses kept; |cos| -> target"""
    rng = np.random.default_rng(0); P = []
    for (e, p, pt) in ((0.0, 0.0, 300.0), (0.25, 0.1, 150.0), (-0.1, 0.3, 150.0)):
        for _ in range(8):
            ee, pp, q = e + rng.normal(0, 0.02), p + rng.normal(0, 0.02), pt / 8 * rng.uniform(0.5, 1.5)
            P.append([q * np.cos(pp), q * np.sin(pp), q * np.sinh(ee), q * np.cosh(ee)])
    P = np.array(P); lab = np.repeat(np.arange(3), 8)
    D0 = dalitz_from(P[None], np.array([24]), lab[None])
    for dc in DCS:
        r = edit_jet(P, lab, dc, 1)
        for kind in ('D', 'cphi', 'cr'):
            Pn = r[kind][0]; Dn = dalitz_from(Pn[None], np.array([24]), lab[None])
            sub = [np.sqrt(max(m2(Pn[lab == k].sum(0)), 0)) - np.sqrt(max(m2(P[lab == k].sum(0)), 0)) for k in range(3)]
            log(f'selftest dc={dc:+.1f} {kind}: dm {Dn["m3"][0] - D0["m3"][0]:+.2e} dmW {Dn["mW"][0] - D0["mW"][0]:+.2e} '
                f'dsubjet m {np.round(sub, 6)} cos {D0["cos"][0]:.3f} -> {Dn["cos"][0]:.3f} (target {r["D"][1]["c1"]:.3f}) '
                f'mbq {D0["mbq_hi"][0]:.1f}/{D0["mbq_lo"][0]:.1f} -> {Dn["mbq_hi"][0]:.1f}/{Dn["mbq_lo"][0]:.1f}')


def main():
    RES.mkdir(parents=True, exist_ok=True)
    log(f'H31 start; smoke={SMOKE}, per class {PER}, ref jets per class {X.NREF}, dc {DCS}')
    selftest()
    SAMP = {'top': load_top(), 'jc': load_jc()}
    REF, POOL, rinfo = {}, {}, {}
    for ds in ('jc', 'top'):
        Xr, Or, yr = ref_train(ds); REF[ds], rinfo[ds] = fit_refs(Xr, Or, yr); POOL[ds] = (Xr, Or, yr)
        rinfo[ds]['cos_hist_train'] = {cn: np.histogram(Or['cos'][yr == cl], bins=10, range=(0, 1))[0].tolist() for cl, cn in ((1, 'top'), (0, 'qcd'))}
        log(f'refs {ds}: {rinfo[ds]}')
    M = X.Models()
    P = SAMP['top']['P4'].astype(np.float64); real = np.arange(P.shape[1])[None, :] < SAMP['top']['nn'][:, None]
    mm = (P[..., 3] ** 2 - (P[..., :3] ** 2).sum(-1))[real]; M.massless = bool(np.mean(np.sqrt(np.maximum(mm, 0)) > 0.1) < 0.01)
    log(f'Top constituents massless: {M.massless}')
    R = dict(node='H31', smoke=SMOKE, dc_values=DCS, samples={ds: S['info'] for ds, S in SAMP.items()}, references=rinfo, sets={})
    PJ = {}
    for ds, S in SAMP.items():
        R['sets'][ds], pj = run_sample(ds, S, M, REF[ds], POOL[ds]); PJ.update(pj)
    tests = [(ds, k, m) for ds in R['sets'] for k, r in R['sets'][ds]['dc'].items() for m in r.get('models', {}) if m in NETS]
    for pk, hk in (('p_perm', 'p_holm'), ('p_perm_vs_MLP3', 'p_holm_vs_MLP3'), ('p_perm_vs_MLP5', 'p_holm_vs_MLP5')):
        adj = H._holm([R['sets'][ds]['dc'][k]['models'][m][pk] for ds, k, m in tests]) if tests else []
        for (ds, k, m), pa in zip(tests, adj): R['sets'][ds]['dc'][k]['models'][m][hk] = float(pa)
    R['holm_family'] = f'{len(tests)} tests: networks x dc x samples x classes, D - cphi two-sided sign-flip permutation ({NPERM} perms)'
    (RES / f'{TAG}_top_dalitz.json').write_text(json.dumps(R, indent=1, default=float)); log(f'wrote {TAG}_top_dalitz.json')
    np.savez_compressed(RES / f'{TAG}_perjet.npz', **PJ)
    try: fig(R)
    except Exception as ex: log('figure failed:', repr(ex))


def fig(R):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    show = NETS + ['MLP3', 'MLP5', 'MLP5c', 'MLPx']
    lab = {'ParT_top': 'Top ParT', 'PN_top': 'Top PN', 'LNet_top': 'Top LorentzNet', 'PFN_top': 'Top PFN', 'EFN_top': 'Top EFN',
           'ParT_kin': 'JetClass ParT_kin', 'PN_kin': 'JetClass PN_kin', 'ParT_full': 'JetClass ParT_full', 'PN_full': 'JetClass PN_full',
           'MLP3': 'MLP(m, mW, τ32)', 'MLP5': 'MLP(+ m_bq pair masses)', 'MLP5c': 'MLP(+ pair masses, |cos θ*|)', 'MLPx': 'MLP side effects, no pair masses'}
    col = {'ParT_top': '#c05621', 'PN_top': '#dd6b20', 'LNet_top': '#975a16', 'PFN_top': '#b7791f', 'EFN_top': '#d69e2e',
           'ParT_kin': '#2b6cb0', 'PN_kin': '#4299e1', 'ParT_full': '#553c9a', 'PN_full': '#9f7aea',
           'MLP3': '0.45', 'MLP5': '#2f855a', 'MLP5c': 'k', 'MLPx': '#e53e3e'}
    fig_, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for i, (ds, dsn) in enumerate((('top', 'Top Landscape jets'), ('jc', 'JetClass jets (550-650 GeV)'))):
        for j, (cn, cnn) in enumerate((('top', 'top jets'), ('qcd', 'QCD jets'))):
            ax = axes[i, j]; dv = [dc for dc in R['dc_values'] if 'models' in R['sets'][ds]['dc'][f'{dc}|{cn}']]
            for k, m in enumerate(show):
                xs, ys, lo_, hi_ = [], [], [], []
                for dc in dv:
                    Mm = R['sets'][ds]['dc'][f'{dc}|{cn}']['models'].get(m)
                    if Mm is None: continue
                    xs.append(dc + 0.012 * (k - 6)); ys.append(Mm['D_minus_cphi']); lo_.append(Mm['ci95'][0]); hi_.append(Mm['ci95'][1])
                if not xs: continue
                ys, lo_, hi_ = map(np.array, (ys, lo_, hi_)); ref = m.startswith('MLP')
                ax.errorbar(xs, ys, yerr=[ys - lo_, hi_ - ys], marker='s' if ref else 'o', ms=4, lw=2.2 if m == 'MLP5' else 1.2,
                            ls='--' if ref else '-', color=col[m], capsize=2, label=lab[m])
            ax.axhline(0, color='0.6', lw=0.8); ax.axvline(0, color='0.85', lw=0.8); ax.set_xticks(list(R['dc_values']))
            dcos = [R['sets'][ds]['dc'][f'{dc}|{cn}']['observables']['cos']['D']['mean_change'] for dc in dv]
            n_ = [R['sets'][ds]['dc'][f'{dc}|{cn}']['n'] for dc in dv]
            ax.set_title(f'{dsn}, {cnn}\nachieved mean Δ|cos θ*| ' + ', '.join(f'{d:+.2f}' for d in dcos) +
                         f'; n = {min(n_) if n_ else 0}-{max(n_) if n_ else 0} jets per Δc', fontsize=8.5, loc='left')
            if j == 0: ax.set_ylabel('Δ top log-odds: W-decay-angle edit\nminus azimuthal-rotation control (D − cφ)')
            ax.set_xlabel('requested change Δc of |cos θ*| (W rest frame)')
        ax = axes[i, 2]; cc = np.linspace(0.05, 0.95, 10)
        for cn, c_ in (('top', '#c05621'), ('qcd', '#2b6cb0')):
            h = np.array(R['sets'][ds]['cos_hist'][cn], float); ax.step(cc, h / h.sum() * 10, where='mid', color=c_, label=f'{cn} jets (all)')
            h = np.array(R['sets'][ds]['cos_hist_window'][cn], float)
            if h.sum() > 0: ax.step(cc, h / h.sum() * 10, where='mid', color=c_, ls=':', label=f'{cn}, 140<m<200, |mW−80|<15')
        g = np.linspace(0, 1, 50); ax.plot(g, 1.275 - 0.825 * g ** 2, color='k', lw=0.8, label='parton level, F0 = 0.7')
        ax.set_xlabel('|cos θ*| (kT-3 subjets, W rest frame)'); ax.set_ylabel('density'); ax.set_title(f'{dsn}: unedited |cos θ*|', fontsize=9, loc='left')
        ax.legend(fontsize=7)
    axes[1, 0].legend(fontsize=6.5, loc='lower left', ncol=2)
    fig_.suptitle('Top-score change when the W decay angle is rotated at fixed jet mass and fixed W-pair mass', fontsize=10)
    fig_.tight_layout(); fig_.savefig(RES / f'{TAG}_dalitz.png', dpi=120); log(f'wrote {TAG}_dalitz.png')


if __name__ == '__main__':
    if '--fig' in sys.argv: fig(json.loads((RES / f'{TAG}_top_dalitz.json').read_text()))
    elif '--selftest' in sys.argv: selftest()
    else: main()
