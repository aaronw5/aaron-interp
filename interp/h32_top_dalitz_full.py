"""H32 (loop D, claim D6 / D11 / D13 extension): W decay angle |cos theta*| inside top at fixed m_top and m_W. Full-statistics
run of the H31 edit plus a side-effect-matched control (c-side) that reproduces the edit's change in the prong pT fractions and
tau32 while keeping |cos theta*| (and every pairwise invariant mass) fixed.

Samples (random rows, never first-k; seed 3201, disjoint from the H31 smoke seed 3101)
  top  Top Landscape test split (converted parquet), 2,000 top + 2,000 QCD (h4_top_wmass._sample(2000, 3201)); rebuilt through
       (pT, eta, phi, m) and sorted by pT.
  jc   JetClass test TTBar_100 (Tbqq) / ZJetsToNuNu_100 (QCD), stored 550 < pT < 650 GeV, |eta| < 2, >= 3 particles, 2,000 per
       class (h19_wpair_cross.jc_read rng 3201 / 3202; eta = jet_eta + sign(jet_eta) deta).
Edits per jet and dc in {-0.6, -0.3, +0.3, +0.6} (h31_top_dalitz.edit_jet): D (rotate the W decay in the W rest frame toward /
  away from the b direction u, |cos theta*| -> c0 + dc), cphi (rotation about u by |alpha|), cr (random axis, |alpha|), TN (rigid
  translation by D's mean displacement), and new
  cside  a Lorentz transformation chosen per jet among 48 x 33 = 1,584 candidates
           Lambda = Lambda_T(Q) . Lambda_W(R) on the W-pair constituents, Lambda_T(Q) on the b-subjet constituents, where
           Lambda_W(R) = B(-beta_W) R B(beta_W) with R a rotation about u (24 azimuths) or the mirror theta* -> pi - theta*
           followed by a rotation about u (24 azimuths), and Lambda_T(Q) = B(-beta_T) Q B(beta_T) with Q = identity or a
           rotation of the top decay in the top rest frame tilting the W direction by rho in {0.05, 0.1, 0.2, 0.35} rad toward
           8 azimuths. Every candidate keeps the jet four-momentum, m_top, m_W, every intra-subjet invariant and |cos theta*|
           exactly fixed; m_ac, m_bc are exactly fixed for the non-mirrored candidates, while the mirror swaps them only up to
           a term proportional to the W-frame energy difference of the two W subjets (m_a != m_b), ~1 GeV in the self-test. Step 1: the 10 candidates closest to D in the three
           labelled-subjet pT fractions (W prongs sorted, b-subjet) with all subjets within dR 0.75 of the jet axis; step 2:
           among them the one minimising cost = sum (df - df_D)^2 / 0.02^2 + (dtau32 - dtau32_D)^2 / 0.01^2 (tau32 recomputed
           on the candidate jets, H4 top_obs, R = 0.8).
         Deviation from the plan: "within-prong rescaling" was not used, because rescaling constituent momenta inside a prong
           changes the prong mass and therefore m_W and m_top; the top-frame tilt Q supplies the extra freedom instead.
Acceptance (all variants): after re-clustering |dm| < 0.5 GeV and |dmW| < 2 GeV. Analysis set per (dc, class): jets accepted
  for D, cphi and cside. Every edited jet rebuilt about its own constituent-sum axis for every network (H25 path).
Models: as H31 (Top ParT / PN / LorentzNet / PFN / EFN; JetClass ParT_kin / PN_kin on both samples; ParT_full / PN_full on jc).
References: MLP3 (m, mW, tau32), MLP5 (+ m_bq_hi, m_bq_lo), MLP5c (+ |cos theta*|), MLPx (side effects without pair masses),
  MLP(64, 64) on 25k + 25k disjoint jets per dataset, 5 retrains each (random_state 0..4; init and early-stopping split).
Metrics per sample x class x dc x model: D - cside and D - cphi (top log-odds), 95 % bootstrap CI over jets (B = 2000),
  two-sided sign-flip permutation p (5,000), Holm over networks x dc x samples x classes (separately for each contrast);
  c-r null = cr - cphi with CI; McNemar D vs cside / D vs cphi at eps_S = 0.5 and 0.3; ratios to each MLP retrain
  (ranges over the 5 retrains); dose = Spearman of the D - cside means against dc; C2ST (HGB on 20 substructure features)
  for id, D, cphi, cside vs natural pool jets matched in (m, tau32, mW, |cos theta*|).
Run: ./sync.sh && ./part_job.sh d-h32-run 'python -m interp.h32_top_dalitz_full'   (H32_SMOKE=1: 150 per class, 2k ref jets)
Outputs: $AI_RESULTS/physics/h32_top_dalitz_full.json, h32_perjet.npz, h32_dalitz.png
"""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import h31_top_dalitz as D31      # noqa: E402
from interp import h4_top_wmass as H          # noqa: E402
from interp import h19_wpair_cross as X       # noqa: E402
from interp import h25_top_tau32 as H25       # noqa: E402

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
SMOKE = bool(os.environ.get('H32_SMOKE'))
RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'physics'
TAG = 'h32_smoke' if SMOKE else 'h32'
PER = 150 if SMOKE else 2000
X.NREF = 2000 if SMOKE else 25000
B = 300 if SMOKE else 2000
NPERM = 500 if SMOKE else 5000
NSEED = 2 if SMOKE else 5
SEED = 3201
DCS = D31.DCS
VAR = ('D', 'cphi', 'cr', 'TN', 'cside')
NETS = D31.NETS
REFS = D31.REFS
KCAND = 10
SF, ST = 0.02, 0.01
DRMAX = 0.75
boost, rodrigues, m2 = D31.boost, D31.rodrigues, D31.m2


# ================================================================================================================ c-side
def L4(b):
    """4x4 matrix of h31.boost(., b) acting on column (px, py, pz, E)"""
    b = np.asarray(b, np.float64); b2 = float(b @ b); M = np.eye(4)
    if b2 < 1e-16: return M
    g = 1.0 / np.sqrt(1.0 - b2)
    M[:3, :3] += (g - 1.0) * np.outer(b, b) / b2; M[:3, 3] = -g * b; M[3, :3] = -g * b; M[3, 3] = g
    return M


def R4(R):
    M = np.eye(4); M[:3, :3] = R; return M


def _perp(w):
    w = w / np.linalg.norm(w); t = np.array([1.0, 0, 0]) if abs(w[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(w, t); e1 /= np.linalg.norm(e1); return e1, np.cross(w, e1)


def _etaphi4(V):
    pt = np.hypot(V[..., 0], V[..., 1]); return np.arcsinh(V[..., 2] / np.maximum(pt, 1e-12)), np.arctan2(V[..., 1], V[..., 0]), pt


def _dr(e1, p1, e2, p2):
    return np.hypot(e1 - e2, (p1 - p2 + np.pi) % (2 * np.pi) - np.pi)


def pair_abc(P, lab):
    sel = [lab == k for k in range(3)]
    if min(s.sum() for s in sel) < 1: return None
    V = [P[s].sum(0) for s in sel]; pairs = [(0, 1), (0, 2), (1, 2)]
    pm = [np.sqrt(max(m2(V[i] + V[j]), 0)) for i, j in pairs]; a, b = pairs[int(np.argmin(np.abs(np.array(pm) - H.MWREF)))]
    return a, b, 3 - a - b, sel, V


def fracs(Va, Vb, Vc):
    """(f_W_hi, f_W_lo, f_b) of the labelled-subjet scalar pT sum; works on broadcast arrays (..., 4)"""
    pa, pb, pc = (np.hypot(V[..., 0], V[..., 1]) for V in (Va, Vb, Vc)); s = pa + pb + pc
    return np.stack([np.maximum(pa, pb) / s, np.minimum(pa, pb) / s, pc / s], -1)


def cside_transforms(P, lab):
    """candidate Lorentz transforms (LW on W constituents, LT on b constituents) at fixed |cos theta*| for one jet"""
    r = pair_abc(P, lab)
    if r is None: return None
    a, b, c, sel, V = r; Va, Vb, Vc = V[a], V[b], V[c]; W = Va + Vb; T = W + Vc
    bW = W[:3] / W[3]; LW = L4(bW); LWi = L4(-bW)
    na = (LW @ Va)[:3]; u = (LW @ Vc)[:3]; na /= np.linalg.norm(na); u /= np.linalg.norm(u)
    cs = float(na @ u); npl = na if cs >= 0 else -na; th = float(np.arccos(min(abs(cs), 1.0))); k = np.cross(npl, u)
    if np.linalg.norm(k) < 1e-9: return None
    Rs = []
    for s in (+1, -1):
        base = np.eye(3) if s > 0 else rodrigues(k, -(np.pi - 2 * th))
        for ph in np.linspace(0, 2 * np.pi, 24, endpoint=False): Rs.append(rodrigues(u, ph) @ base if ph > 0 else base)
    LWs = np.stack([LWi @ R4(R) @ LW for R in Rs])                                          # (48, 4, 4)
    bT = T[:3] / T[3]; LT = L4(bT); LTi = L4(-bT); wt = (LT @ W)[:3]; e1, e2 = _perp(wt)
    Qs = [np.eye(3)] + [rodrigues(np.cos(ps) * e1 + np.sin(ps) * e2, rho) for rho in (0.05, 0.1, 0.2, 0.35)
                        for ps in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
    LTs = np.stack([LTi @ R4(Q) @ LT for Q in Qs])                                          # (33, 4, 4)
    return dict(a=a, b=b, c=c, sel=sel, Va=Va, Vb=Vb, Vc=Vc, T=T, LWs=LWs, LTs=LTs)


def cside_rank(tr, fD):
    """fraction cost of every candidate (33 x 48) against the target fractions fD; inf if a subjet leaves dR 0.75"""
    LTs, LWs = tr['LTs'], tr['LWs']; M = np.einsum('qij,rjk->qrik', LTs, LWs)               # (33, 48, 4, 4)
    Va = np.einsum('qrij,j->qri', M, tr['Va']); Vb = np.einsum('qrij,j->qri', M, tr['Vb'])
    Vc = np.broadcast_to(np.einsum('qij,j->qi', LTs, tr['Vc'])[:, None], Va.shape)
    f = fracs(Va, Vb, Vc); cost = ((f - fD) ** 2).sum(-1)
    eT, pT_, _ = _etaphi4(tr['T'])
    dr = np.max(np.stack([_dr(*_etaphi4(V)[:2], eT, pT_) for V in (Va, Vb, Vc)]), 0)
    cost = np.where(dr <= DRMAX, cost, np.inf)
    return cost, f, M


def apply_cand(P, tr, M, q):
    Pn = P.copy(); inW = tr['sel'][tr['a']] | tr['sel'][tr['b']]; sc = tr['sel'][tr['c']]
    Pn[inW] = P[inW] @ M.T; Pn[sc] = P[sc] @ tr['LTs'][q].T; return Pn


def build_cside(P4, nn, LAB, rows, PD, convD, tau0, tauD):
    """per jet: rank candidates by labelled-subjet pT fractions against D, then pick by fractions + tau32 among the best KCAND"""
    J = len(nn); cands = [P4.copy() for _ in range(KCAND)]; ok = np.zeros((KCAND, J), bool); fc = np.full((KCAND, J), np.inf)
    fcand = np.full((KCAND, J, 3), np.nan); fD_all = np.full((J, 3), np.nan); f0_all = np.full((J, 3), np.nan)
    reasons = {}
    for ii, j in enumerate(rows):
        n = nn[j]
        if not convD[j]: reasons['D_not_edited'] = reasons.get('D_not_edited', 0) + 1; continue
        P = P4[j, :n].astype(np.float64); lab = LAB[ii, :n]; tr = cside_transforms(P, lab)
        if tr is None: reasons['no_transform'] = reasons.get('no_transform', 0) + 1; continue
        Pd = PD[j, :n].astype(np.float64); s = tr['sel']
        fD = fracs(Pd[s[tr['a']]].sum(0), Pd[s[tr['b']]].sum(0), Pd[s[tr['c']]].sum(0)); fD_all[j] = fD
        f0_all[j] = fracs(tr['Va'], tr['Vb'], tr['Vc'])
        cost, f, M = cside_rank(tr, fD); flat = np.argsort(cost, axis=None)[:KCAND]
        if not np.isfinite(cost.flat[flat[0]]): reasons['out_of_cone'] = reasons.get('out_of_cone', 0) + 1; continue
        reasons['ok'] = reasons.get('ok', 0) + 1
        for kk, fl in enumerate(flat):
            q, r_ = np.unravel_index(fl, cost.shape)
            if not np.isfinite(cost[q, r_]): continue
            cands[kk][j, :n] = apply_cand(P, tr, M[q, r_], q).astype(np.float32); ok[kk, j] = True; fc[kk, j] = cost[q, r_]
            fcand[kk, j] = f[q, r_]
    tc = np.stack([H._obs(cands[kk], nn)['tau32'] for kk in range(KCAND)])                 # (K, J)
    tot = fc / SF ** 2 + ((tc - tau0[None]) - (tauD - tau0)[None]) ** 2 / ST ** 2
    tot = np.where(ok, tot, np.inf); best = np.argmin(tot, 0); conv = np.isfinite(tot[best, np.arange(J)])
    Pc = P4.copy(); r = np.arange(J)
    for kk in range(KCAND):
        m = conv & (best == kk); Pc[m] = cands[kk][m]
    info = dict(reasons=reasons, f_id=f0_all, f_D=fD_all, f_cside=fcand[best, r], frac_cost=fc[best, r],
                tau32_cside_pred=tc[best, r], pick=best)
    return Pc, conv, info


# ================================================================================================================ samples
def load_top():
    S = H._sample(PER, SEED); nn = np.minimum(S['jet'][:, 5].astype(np.int64), S['full'].shape[1])
    P4 = H._identity(S['full'][..., :4].astype(np.float32), nn).astype(np.float32); y = S['label'].astype(int)
    log(f'top sample: {len(nn)} jets ({y.sum()} top), seed {SEED}')
    return dict(P4=P4, nn=nn, y=y, base=X.top_base(P4, nn), info=dict(n=int(len(nn)), n_top=int(y.sum()), seed=SEED, split='test'))


def load_jc():
    from interp.part_core import cat, kin
    Jt, _, it = X.jc_read('Tbqq', 100, PER, SEED); Jq, _, iq = X.jc_read('QCD', 100, PER, SEED + 1)
    J = cat([Jt, Jq]); P4raw, nn = X.jc_P4(J); P4 = H._identity(P4raw, nn).astype(np.float32)
    J['pmass'] = kin(J)[3].astype(np.float32); y = (J['label'] == D31.TBQQ).astype(int)
    log(f'jc sample: {len(nn)} jets ({y.sum()} Tbqq), seeds {SEED}/{SEED + 1}')
    return dict(P4=P4, nn=nn, y=y, base=J, info=dict(Tbqq=it, QCD=iq))


# ================================================================================================================ references
def fit_refs_seeds(Xf, O, y):
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    R, info = {}, dict(n=int(len(y)))
    for s in range(NSEED):
        for nm in REFS:
            Z = D31.xsel(Xf, O, nm); sc = StandardScaler().fit(Z)
            mlp = MLPClassifier(hidden_layer_sizes=(64, 64), early_stopping=True, random_state=s, max_iter=200).fit(sc.transform(Z), y)
            key = nm if s == 0 else f'{nm}@{s}'; R[key] = (mlp, sc)
            info[key] = dict(iterations=int(mlp.n_iter_), internal_val_acc=float(mlp.best_validation_score_))
        log(f'  ref retrain {s} done')
    return R, info


def eval_refs(R, Xf, O):
    return {nm: D31.lo(mlp.predict_proba(sc.transform(D31.xsel(Xf, O, nm.split('@')[0])))[:, 1]) for nm, (mlp, sc) in R.items()}


# ================================================================================================================ per sample
OBSQ = ('m', 'mWk', 'mbq_hi', 'mbq_lo', 'cos', 'tau32', 'tau21', 'mSD', 'n')
FR = ('f_W_hi', 'f_W_lo', 'f_b')


def run_sample(ds, S, M, REF, POOL):
    nn, y = S['nn'], S['y']; P4id = S['P4']
    if ds == 'top': P4id = H25.sort_pt(P4id, nn); S['P4'] = P4id
    X0, O0 = D31.feats(P4id, nn); log(f'[{ds}] unedited observables on {len(nn)} jets')
    order = np.random.default_rng(605).permutation(len(nn))
    sc = {'id': D31.score(M, ds, P4id, nn, S['base'], order)}; sc['id'].update(eval_refs(REF, X0, O0))
    perf = {}
    for k, s in sc['id'].items():
        st, sb = s[y == 1], s[y == 0]
        perf[k] = dict(AUC=float(H._auc(s, y)), acc_binary=float(((s > 0) == (y == 1)).mean()), wp={})
        for nm, q in (('eS30', 0.7), ('eS50', 0.5)):
            thr = float(np.quantile(st, q)); eb = float((sb >= thr).mean())
            perf[k]['wp'][nm] = dict(thr=thr, eps_B=eb, n_bkg_pass=int((sb >= thr).sum()))
    log(f'[{ds}] unedited: ' + ', '.join(f"{k} AUC {v['AUC']:.4f} acc {v['acc_binary']:.4f}" for k, v in perf.items() if '@' not in k))
    hist = {cn: np.histogram(O0['cos'][y == cl], bins=10, range=(0, 1))[0].tolist() for cl, cn in ((1, 'top'), (0, 'qcd'))}
    log(f'[{ds}] editing')
    D31.VAR = ('D', 'cphi', 'cr', 'TN'); E = D31.run_edits(P4id, nn)
    rows3 = np.flatnonzero(nn >= 3); LAB = H.kt3_batch(np.ascontiguousarray(P4id[rows3].astype(np.float64)), nn[rows3])
    acc, obs, info, side = {}, {'id': (X0, O0)}, {}, {}

    def do_variant(dc, kind, Pe, conv):
        Pv = H25.sort_pt(Pe, nn) if ds == 'top' else Pe
        r = np.flatnonzero(conv); Xv, Ov = X0.copy(), {k: v.copy() for k, v in O0.items()}
        if len(r):
            xr, orr = D31.feats(np.ascontiguousarray(Pv[r]), nn[r]); Xv[r] = xr
            for k in Ov: Ov[k][r] = orr[k]
        ok = conv & (np.abs(Ov['m'] - O0['m']) < 0.5) & (np.abs(Ov['mWk'] - O0['mWk']) < 2)
        acc[(dc, kind)] = ok; obs[(dc, kind)] = (Xv, Ov)
        v = sc[(dc, kind)] = D31.score(M, ds, Pv, nn, S['base'], order); v.update(eval_refs(REF, Xv, Ov))
        log(f'[{ds}] dc={dc:+.1f} {kind}: accepted {ok.sum()} / edited {conv.sum()}; scored')
        return ok

    for dc in DCS:
        for kind in ('D', 'cphi', 'cr', 'TN'):
            e = E[(dc, kind)]; ok = do_variant(dc, kind, e['P'], e['conv'])
            info[f'{dc}|{kind}'] = dict(edited=int(e['conv'].sum()), accepted=int(ok.sum()), reasons=e['reasons'])
        tauD = obs[(dc, 'D')][1]['tau32']
        Pc, convc, sinfo = build_cside(P4id, nn, LAB, rows3, E[(dc, 'D')]['P'], E[(dc, 'D')]['conv'], O0['tau32'], tauD)
        side[dc] = sinfo; ok = do_variant(dc, 'cside', Pc, convc)
        info[f'{dc}|cside'] = dict(edited=int(convc.sum()), accepted=int(ok.sum()), reasons=sinfo['reasons'],
                                   picked_rank_counts=np.bincount(sinfo['pick'][convc], minlength=KCAND).tolist())
        log(f'[{ds}] dc={dc:+.1f} cside reasons {sinfo["reasons"]}')

    rng = np.random.default_rng(11); out = dict(unedited=perf, cos_hist=hist, edit_info=info, dc={})
    allref = sorted(k for k in sc['id'] if k.split('@')[0] in REFS)
    for dc in DCS:
        base_ok = acc[(dc, 'D')] & acc[(dc, 'cphi')] & acc[(dc, 'cside')]
        for cls, cname in ((1, 'top'), (0, 'qcd')):
            rows = np.flatnonzero(base_ok & (y == cls)); n = len(rows); key = f'{dc}|{cname}'
            R = dict(n=int(n), n_class=int((y == cls).sum()), n_D_cphi=int((acc[(dc, 'D')] & acc[(dc, 'cphi')] & (y == cls)).sum()))
            if n < 10: out['dc'][key] = R; continue
            R['c0_mean'] = float(np.nanmean(E[(dc, 'D')]['c0'][rows])); R['c1_target_mean'] = float(np.nanmean(E[(dc, 'D')]['c1'][rows]))
            R['observables'] = {}
            for q in OBSQ:
                R['observables'][q] = dict(id_mean=float(O0[q][rows].mean()))
                for kind in VAR:
                    dq = obs[(dc, kind)][1][q][rows] - O0[q][rows]
                    R['observables'][q][kind] = dict(mean_change=float(dq.mean()), median_abs_change=float(np.median(np.abs(dq))),
                                                     ci95=np.percentile(dq[rng.integers(0, n, (B, n))].mean(1), [2.5, 97.5]).tolist())
            si = side[dc]; R['labelled_fractions'] = {}
            for i, fq in enumerate(FR):
                dD = si['f_D'][rows, i] - si['f_id'][rows, i]; dS = si['f_cside'][rows, i] - si['f_id'][rows, i]
                R['labelled_fractions'][fq] = dict(id_mean=float(si['f_id'][rows, i].mean()), D=float(dD.mean()), cside=float(dS.mean()),
                                                   mean_abs_mismatch=float(np.abs(dS - dD).mean()))
            # reclustered sorted subjet pT fractions (MLPx inputs, X cols 11:14) for D / cphi / cside
            R['reclustered_sorted_fractions'] = {kind: (obs[(dc, kind)][0][rows, 11:14] - X0[rows, 11:14]).mean(0).tolist() for kind in VAR}
            bi = rng.integers(0, n, (B, n)); R['models'] = {}
            refD = {r_: (sc[(dc, 'D')][r_] - sc[(dc, 'cside')][r_])[rows] for r_ in allref}
            refP = {r_: (sc[(dc, 'D')][r_] - sc[(dc, 'cphi')][r_])[rows] for r_ in allref}
            for mdl in NETS + allref:
                if mdl not in sc['id']: continue
                d = {kind: sc[(dc, kind)][mdl] - sc['id'][mdl] for kind in VAR}
                Mm = {}
                for nm_, xv in (('D_minus_cside', d['D'] - d['cside']), ('D_minus_cphi', d['D'] - d['cphi']),
                                ('cr_minus_cphi', d['cr'] - d['cphi']), ('cside_minus_cphi', d['cside'] - d['cphi'])):
                    x = xv[rows]; Mm[nm_] = float(x.mean()); Mm[f'{nm_}_ci95'] = np.percentile(x[bi].mean(1), [2.5, 97.5]).tolist()
                    if mdl in NETS or '@' not in mdl: Mm[f'{nm_}_p'] = D31.perm_p2(x, rng, NPERM) if nm_ in ('D_minus_cside', 'D_minus_cphi') else None
                for kind in VAR:
                    z = d[kind][rows]; Mm[f'd{kind}'] = float(z.mean()); Mm[f'd{kind}_ci95'] = np.percentile(z[bi].mean(1), [2.5, 97.5]).tolist()
                if mdl in NETS:
                    x = (d['D'] - d['cside'])[rows]; xp = (d['D'] - d['cphi'])[rows]
                    Mm['ratios'] = {}
                    for base_ in ('MLP3', 'MLP5', 'MLP5c', 'MLPx'):
                        rs, rsP, cis = [], [], []
                        for r_ in [k for k in allref if k.split('@')[0] == base_]:
                            rx = refD[r_]; den = rx[bi].mean(1)
                            rs.append(float(x.mean() / rx.mean()) if abs(rx.mean()) > 1e-9 else None)
                            rsP.append(float(xp.mean() / refP[r_].mean()) if abs(refP[r_].mean()) > 1e-9 else None)
                            cis.append(np.percentile(x[bi].mean(1) / np.where(np.abs(den) > 1e-9, den, np.nan), [2.5, 97.5]).tolist())
                        Mm['ratios'][base_] = dict(D_minus_cside=rs, D_minus_cside_ci95=cis, D_minus_cphi=rsP)
                    Mm['p_D_minus_cside_vs_MLP5c'] = D31.perm_p2(x - refD['MLP5c'], rng, NPERM)
                    Mm['p_D_minus_cside_vs_MLPx'] = D31.perm_p2(x - refD['MLPx'], rng, NPERM)
                Mm['calls'] = {}
                for nm in ('eS30', 'eS50'):
                    thr = perf[mdl]['wp'][nm]['thr']; c0 = sc['id'][mdl][rows] >= thr
                    cD, cP, cS = (sc[(dc, k)][mdl][rows] >= thr for k in ('D', 'cphi', 'cside'))
                    Mm['calls'][nm] = dict(pass_id=float(c0.mean()), pass_D=float(cD.mean()), pass_cphi=float(cP.mean()), pass_cside=float(cS.mean()),
                                           changed_D=float((cD != c0).mean()), changed_cphi=float((cP != c0).mean()), changed_cside=float((cS != c0).mean()),
                                           mcnemar_D_vs_cside=H._mcnemar(cS, cD), mcnemar_D_vs_cphi=H._mcnemar(cP, cD))
                R['models'][mdl] = Mm
            Xp, Op, yp = POOL; pm = yp == cls; Opc = {k: v[pm] for k, v in Op.items()}
            sub = rows if len(rows) <= 1500 else np.sort(rng.permutation(rows)[:1500])
            R['c2st'] = {kind: D31.matched_c2st(obs[kv][0][sub], {k: v[sub] for k, v in obs[kv][1].items()}, Xp[pm], Opc)
                         for kind, kv in (('id', 'id'), ('D', (dc, 'D')), ('cphi', (dc, 'cphi')), ('cside', (dc, 'cside')))}
            out['dc'][key] = R; ob = R['observables']; lf = R['labelled_fractions']
            log(f'[{ds}] {key}: n {n}; dcos D {ob["cos"]["D"]["mean_change"]:+.3f} cside {ob["cos"]["cside"]["mean_change"]:+.3f} cphi {ob["cos"]["cphi"]["mean_change"]:+.3f}; '
                f'dtau32 D {ob["tau32"]["D"]["mean_change"]:+.4f} cside {ob["tau32"]["cside"]["mean_change"]:+.4f} cphi {ob["tau32"]["cphi"]["mean_change"]:+.4f}; '
                f'df_W_hi D {lf["f_W_hi"]["D"]:+.4f} cside {lf["f_W_hi"]["cside"]:+.4f}; df_b D {lf["f_b"]["D"]:+.4f} cside {lf["f_b"]["cside"]:+.4f}; '
                f'dmbq_hi D {ob["mbq_hi"]["D"]["mean_change"]:+.1f} cside {ob["mbq_hi"]["cside"]["mean_change"]:+.1f}; C2ST {R["c2st"]}')
            for mdl, Mm in R['models'].items():
                if '@' in mdl: continue
                log(f"    {mdl:9s} D-cside {Mm['D_minus_cside']:+.3f} {np.round(Mm['D_minus_cside_ci95'], 3)} p {Mm.get('D_minus_cside_p')} | "
                    f"D-cphi {Mm['D_minus_cphi']:+.3f} | cr-cphi {Mm['cr_minus_cphi']:+.3f} {np.round(Mm['cr_minus_cphi_ci95'], 3)} | "
                    f"ratios D-cside xMLP5c {Mm.get('ratios', {}).get('MLP5c', {}).get('D_minus_cside')}")
    out['dose'] = {}
    for cname in ('top', 'qcd'):
        for mdl in NETS + REFS:
            xs = [dc for dc in DCS if 'models' in out['dc'][f'{dc}|{cname}'] and mdl in out['dc'][f'{dc}|{cname}']['models']]
            for nm_ in ('D_minus_cside', 'D_minus_cphi'):
                ys = [out['dc'][f'{dc}|{cname}']['models'][mdl][nm_] for dc in xs]
                if len(xs) >= 3: out['dose'][f'{cname}|{mdl}|{nm_}'] = dict(spearman=D31.spearman(xs, ys), means=ys)
    perjet = {f'{ds}|{"id" if k == "id" else f"{k[0]}|{k[1]}"}|{m}': np.asarray(v, np.float32) for k, d in sc.items() for m, v in d.items()}
    perjet[f'{ds}|y'] = y
    for (dc, kind), ok in acc.items(): perjet[f'{ds}|acc|{dc}|{kind}'] = ok
    for k, (Xo, Oo) in obs.items():
        nm = 'id' if k == 'id' else f'{k[0]}|{k[1]}'
        for q in ('cos', 'mbq_hi', 'mbq_lo', 'tau32', 'm', 'mWk'): perjet[f'{ds}|{q}|{nm}'] = Oo[q].astype(np.float32)
        perjet[f'{ds}|sortedfrac|{nm}'] = Xo[:, 11:14].astype(np.float32)
    for dc, si in side.items():
        for q in ('f_id', 'f_D', 'f_cside', 'frac_cost'): perjet[f'{ds}|side|{dc}|{q}'] = np.asarray(si[q], np.float32)
    return out, perjet


# ================================================================================================================ self-test
def selftest():
    """c-side invariants on a synthetic three-prong jet: m_top, m_W, pair masses (as a set), subjet masses, |cos| fixed"""
    rng = np.random.default_rng(0); P = []
    for (e, p, pt) in ((0.0, 0.0, 300.0), (0.25, 0.1, 150.0), (-0.1, 0.3, 150.0)):
        for _ in range(8):
            ee, pp, q = e + rng.normal(0, 0.02), p + rng.normal(0, 0.02), pt / 8 * rng.uniform(0.5, 1.5)
            P.append([q * np.cos(pp), q * np.sin(pp), q * np.sinh(ee), q * np.cosh(ee)])
    P = np.array(P); lab = np.repeat(np.arange(3), 8); D0 = D31.dalitz_from(P[None], np.array([24]), lab[None])
    tr = cside_transforms(P, lab); Pd = D31.edit_jet(P, lab, 0.3, 1)['D'][0]; s = tr['sel']
    fD = fracs(Pd[s[tr['a']]].sum(0), Pd[s[tr['b']]].sum(0), Pd[s[tr['c']]].sum(0)); f0 = fracs(tr['Va'], tr['Vb'], tr['Vc'])
    cost, f, M = cside_rank(tr, fD); worst = 0.0
    for q, r_ in [(0, 0), (0, 30), (5, 3), (20, 40), (32, 47)]:
        Pn = apply_cand(P, tr, M[q, r_], q); Dn = D31.dalitz_from(Pn[None], np.array([24]), lab[None])
        sub = max(abs(np.sqrt(max(m2(Pn[lab == k].sum(0)), 0)) - np.sqrt(max(m2(P[lab == k].sum(0)), 0))) for k in range(3))
        # the mirror candidates (r >= 24) keep |cos theta*| exactly but swap m_ac <-> m_bc only up to the subjet-mass
        # asymmetry term (E_a != E_b in the W frame when m_a != m_b), so pair masses are asserted only for r < 24
        dv = max(abs(Dn['m3'][0] - D0['m3'][0]), abs(Dn['mW'][0] - D0['mW'][0]), sub, abs(Dn['cos'][0] - D0['cos'][0]) * 100,
                 float(np.abs(Pn.sum(0) - P.sum(0)).max()),
                 (abs(Dn['mbq_hi'][0] - D0['mbq_hi'][0]) + abs(Dn['mbq_lo'][0] - D0['mbq_lo'][0])) if r_ < 24 else 0.0)
        worst = max(worst, dv)
        log(f'selftest cside cand ({q},{r_}): dm {Dn["m3"][0] - D0["m3"][0]:+.1e} dmW {Dn["mW"][0] - D0["mW"][0]:+.1e} '
            f'dmbq {Dn["mbq_hi"][0] - D0["mbq_hi"][0]:+.1e}/{Dn["mbq_lo"][0] - D0["mbq_lo"][0]:+.1e} dsub {sub:.1e} '
            f'cos {D0["cos"][0]:.4f}->{Dn["cos"][0]:.4f} fracs {np.round(f[q, r_], 3)} (id {np.round(f0, 3)})')
    q, r_ = np.unravel_index(np.argmin(cost), cost.shape)
    log(f'selftest: D(+0.3) fractions {np.round(fD, 4)} id {np.round(f0, 4)}; best cside ({q},{r_}) {np.round(f[q, r_], 4)} cost {cost[q, r_]:.2e}; '
        f'worst invariant change {worst:.2e}')
    assert worst < 1e-6, 'c-side invariants broken'


# ================================================================================================================ main
def main():
    RES.mkdir(parents=True, exist_ok=True)
    log(f'H32 start; smoke={SMOKE}, per class {PER}, ref jets per class {X.NREF}, retrains {NSEED}, dc {DCS}, seed {SEED}')
    D31.selftest(); selftest()
    SAMP = {'top': load_top(), 'jc': load_jc()}
    REF, POOL, rinfo = {}, {}, {}
    for ds in ('jc', 'top'):
        Xr, Or, yr = D31.ref_train(ds); REF[ds], rinfo[ds] = fit_refs_seeds(Xr, Or, yr); POOL[ds] = (Xr, Or, yr)
        log(f'refs {ds}: {rinfo[ds]}')
    M = X.Models()
    P = SAMP['top']['P4'].astype(np.float64); real = np.arange(P.shape[1])[None, :] < SAMP['top']['nn'][:, None]
    mm = (P[..., 3] ** 2 - (P[..., :3] ** 2).sum(-1))[real]; M.massless = bool(np.mean(np.sqrt(np.maximum(mm, 0)) > 0.1) < 0.01)
    log(f'Top constituents massless: {M.massless}')
    R = dict(node='H32', smoke=SMOKE, dc_values=DCS, samples={ds: S['info'] for ds, S in SAMP.items()}, references=rinfo, sets={},
             cside=dict(n_candidates=48 * 33, kcand=KCAND, sigma_frac=SF, sigma_tau32=ST, drmax=DRMAX))
    PJ = {}
    for ds, S in SAMP.items():
        R['sets'][ds], pj = run_sample(ds, S, M, REF[ds], POOL[ds]); PJ.update(pj)
        (RES / f'{TAG}_top_dalitz_full.json').write_text(json.dumps(R, indent=1, default=float))
        np.savez_compressed(RES / f'{TAG}_perjet_{ds}.npz', **pj); log(f'checkpoint after {ds}')
    tests = [(ds, k, m) for ds in R['sets'] for k, r in R['sets'][ds]['dc'].items() for m in r.get('models', {}) if m in NETS]
    for pk in ('D_minus_cside_p', 'D_minus_cphi_p', 'p_D_minus_cside_vs_MLP5c', 'p_D_minus_cside_vs_MLPx'):
        adj = H._holm([R['sets'][ds]['dc'][k]['models'][m][pk] for ds, k, m in tests]) if tests else []
        for (ds, k, m), pa in zip(tests, adj): R['sets'][ds]['dc'][k]['models'][m][pk + '_holm'] = float(pa)
    R['holm_family'] = f'{len(tests)} tests per contrast: networks x dc x samples x classes, two-sided sign-flip permutation ({NPERM})'
    (RES / f'{TAG}_top_dalitz_full.json').write_text(json.dumps(R, indent=1, default=float)); log(f'wrote {TAG}_top_dalitz_full.json')
    np.savez_compressed(RES / f'{TAG}_perjet.npz', **PJ)
    log('done')


if __name__ == '__main__':
    if '--selftest' in sys.argv: selftest()
    else: main()
