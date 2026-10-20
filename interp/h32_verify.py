"""H32 verification (loop D). Fresh-sample rerun of the H32 decay-angle edit D and the c-side control, plus
  (1) a tau32-priority control c-side2: the same Lorentz family as c-side with a wider top-frame tilt grid
      (rho in {0.05, 0.1, 0.2, 0.35, 0.5, 0.7} rad x 12 azimuths -> 73 x 48 = 3,504 candidates), step 1 = the 30 closest in
      labelled-subjet pT fractions, step 2 = cost sum(df - df_D)^2 / 0.05^2 + (dtau32 - dtau32_D)^2 / 0.005^2 (tau32 first).
      Every candidate keeps the jet four-momentum, m_top, m_W, intra-subjet invariants and |cos theta*| (as c-side).
  (2) robustness of the H32 residual regression y = s_D - s_ctrl = b0 + b_cos dcos + b_tau dtau32 + b_hi df_hi + b_lo df_lo:
      fitted on all four dc (as H32), on +dc only, on -dc only, and with extra side-effect regressors (dtau32^2, dtau21, dmSD,
      d sorted subjet masses x3, d middle fraction); paired bootstrap difference of b_cos against each MLPx retrain
      (angle-blind leak) and each MLP5 retrain (positive control: sees the pair masses that carry the angle).
  (3) a non-parametric check: jets whose residual D - ctrl mismatch is small (|dtau32| < 0.01, |df_hi| < 0.01, |df_lo| < 0.01)
      and the raw D - ctrl contrast there, networks vs MLPx / MLP5.
Samples: random rows, seed 3301 (Top Landscape test; rows of the H32 seed-3201 sample excluded) and JetClass TTBar_100 /
  ZJetsToNuNu_100 in 550-650 GeV (rng 3301 / 3302; rows drawn by H32's rng 3201 / 3202 excluded), 2,000 per class.
Run: ./sync.sh && ./part_job.sh d-h32-verify 'python -m interp.h32_verify'   (H32V_SMOKE=1: 150 per class, 2k ref jets)
Output (JSON only): $AI_RESULTS/physics/h32_verify.json (smoke: h32_verify_smoke.json)
"""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import h32_top_dalitz_full as P   # noqa: E402
from interp import h31_top_dalitz as D31      # noqa: E402
from interp import h4_top_wmass as H          # noqa: E402
from interp import h19_wpair_cross as X       # noqa: E402
from interp import h25_top_tau32 as H25       # noqa: E402

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
SMOKE = bool(os.environ.get('H32V_SMOKE'))
RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'physics'
TAG = 'h32_verify_smoke' if SMOKE else 'h32_verify'
PER = 150 if SMOKE else 2000
X.NREF = 2000 if SMOKE else 25000
P.NSEED = 2 if SMOKE else 5
B = 300 if SMOKE else 2000
NPERM = 500 if SMOKE else 5000
SEED, OLD = 3301, 3201
DCS = D31.DCS
NETS = D31.NETS
REFS = D31.REFS
K2, SF2, ST2 = 30, 0.05, 0.005
RHO2, NAZ2 = (0.05, 0.1, 0.2, 0.35, 0.5, 0.7), 12
VARS = ('D', 'cphi', 'cr', 'cside', 'cside2')
rodrigues, L4, R4 = D31.rodrigues, P.L4, P.R4


# ================================================================================================================ samples
def load_top():
    from interp import data_top as dt
    y = np.load(dt.DST / 'test' / 'label.npy')
    ro = np.random.default_rng(OLD); old = np.concatenate([ro.choice(np.flatnonzero(y == c), 2000, replace=False) for c in (0, 1)])
    rn = np.random.default_rng(SEED)
    idx = np.sort(np.concatenate([rn.choice(np.setdiff1d(np.flatnonzero(y == c), old), PER, replace=False) for c in (0, 1)]))
    full = H._full_test()[idx]; jet = np.load(dt.DST / 'test' / 'jet.npy')[idx]
    nn = np.minimum(jet[:, 5].astype(np.int64), full.shape[1])
    P4 = H._identity(full[..., :4].astype(np.float32), nn).astype(np.float32); yy = y[idx].astype(int)
    log(f'top sample: {len(nn)} jets ({yy.sum()} top), seed {SEED}, overlap with H32 rows {len(np.intersect1d(idx, old))}')
    return dict(P4=P4, nn=nn, y=yy, base=X.top_base(P4, nn), info=dict(n=int(len(nn)), n_top=int(yy.sum()), seed=SEED,
                                                                         overlap_with_h32=int(len(np.intersect1d(idx, old)))))


def jc_read_ex(c, per, seed, seed_old):
    import uproot
    from interp.part_core import read_root, cat, sub
    f = X.jc_file(c, 100); t = uproot.open(f)['tree']
    a = t.arrays(['jet_pt', 'jet_eta', 'jet_nparticles'], library='np')
    ok = (a['jet_pt'] > 550) & (a['jet_pt'] < 650) & (np.abs(a['jet_eta']) < 2) & (a['jet_nparticles'] >= 3)
    el = np.flatnonzero(ok); old = np.random.default_rng(seed_old).choice(el, min(2000, len(el)), replace=False)
    el2 = np.setdiff1d(el, old); n = min(per, len(el2))
    idx = np.sort(np.random.default_rng(seed).choice(el2, n, replace=False)); parts = []; N = t.num_entries
    for s in range(0, N, 20000):
        loc = idx[(idx >= s) & (idx < s + 20000)] - s
        if len(loc): parts.append(sub(read_root(f, start=s, stop=min(s + 20000, N)), loc))
    log(f'jc {c}: {len(el)} eligible, {len(old)} H32 rows excluded, drew {n} (rng {seed})')
    return cat(parts), dict(file=f.name, eligible=int(len(el)), drawn=int(n), overlap_with_h32=int(len(np.intersect1d(idx, old))))


def load_jc():
    from interp.part_core import cat, kin
    Jt, it = jc_read_ex('Tbqq', PER, SEED, OLD); Jq, iq = jc_read_ex('QCD', PER, SEED + 1, OLD + 1)
    J = cat([Jt, Jq]); P4raw, nn = X.jc_P4(J); P4 = H._identity(P4raw, nn).astype(np.float32)
    J['pmass'] = kin(J)[3].astype(np.float32); y = (J['label'] == D31.TBQQ).astype(int)
    return dict(P4=P4, nn=nn, y=y, base=J, info=dict(Tbqq=it, QCD=iq))


# ================================================================================================================ c-side2
def cside_transforms2(Pj, lab):
    """as h32 cside_transforms, wider top-frame tilt grid"""
    r = P.pair_abc(Pj, lab)
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
    LWs = np.stack([LWi @ R4(R) @ LW for R in Rs])
    bT = T[:3] / T[3]; LT = L4(bT); LTi = L4(-bT); wt = (LT @ W)[:3]; e1, e2 = P._perp(wt)
    Qs = [np.eye(3)] + [rodrigues(np.cos(ps) * e1 + np.sin(ps) * e2, rho) for rho in RHO2
                        for ps in np.linspace(0, 2 * np.pi, NAZ2, endpoint=False)]
    LTs = np.stack([LTi @ R4(Q) @ LT for Q in Qs])
    return dict(a=a, b=b, c=c, sel=sel, Va=Va, Vb=Vb, Vc=Vc, T=T, LWs=LWs, LTs=LTs)


def build_cside2(*args):
    saved = (P.cside_transforms, P.KCAND, P.SF, P.ST)
    P.cside_transforms, P.KCAND, P.SF, P.ST = cside_transforms2, K2, SF2, ST2
    try: return P.build_cside(*args)
    finally: P.cside_transforms, P.KCAND, P.SF, P.ST = saved


def selftest2():
    rng = np.random.default_rng(0); Pl = []
    for (e, p, pt) in ((0.0, 0.0, 300.0), (0.25, 0.1, 150.0), (-0.1, 0.3, 150.0)):
        for _ in range(8):
            ee, pp, q = e + rng.normal(0, 0.02), p + rng.normal(0, 0.02), pt / 8 * rng.uniform(0.5, 1.5)
            Pl.append([q * np.cos(pp), q * np.sin(pp), q * np.sinh(ee), q * np.cosh(ee)])
    Pj = np.array(Pl); lab = np.repeat(np.arange(3), 8); D0 = D31.dalitz_from(Pj[None], np.array([24]), lab[None])
    tr = cside_transforms2(Pj, lab); Pd = D31.edit_jet(Pj, lab, 0.3, 1)['D'][0]; s = tr['sel']
    fD = P.fracs(Pd[s[tr['a']]].sum(0), Pd[s[tr['b']]].sum(0), Pd[s[tr['c']]].sum(0)); cost, f, M = P.cside_rank(tr, fD); worst = 0.0
    for q, r_ in [(0, 0), (40, 7), (72, 20), (60, 30), (72, 47)]:
        Pn = P.apply_cand(Pj, tr, M[q, r_], q); Dn = D31.dalitz_from(Pn[None], np.array([24]), lab[None])
        worst = max(worst, abs(Dn['m3'][0] - D0['m3'][0]), abs(Dn['mW'][0] - D0['mW'][0]), abs(Dn['cos'][0] - D0['cos'][0]) * 100,
                    float(np.abs(Pn.sum(0) - Pj.sum(0)).max()))
    log(f'selftest c-side2: {M.shape[0]} x {M.shape[1]} candidates, worst invariant change {worst:.2e}')
    assert worst < 1e-6


# ================================================================================================================ per sample
def run_sample(ds, S, M, REF):
    nn, y = S['nn'], S['y']; P4id = S['P4']
    if ds == 'top': P4id = H25.sort_pt(P4id, nn); S['P4'] = P4id
    X0, O0 = D31.feats(P4id, nn)
    order = np.random.default_rng(605).permutation(len(nn))
    sc = {'id': D31.score(M, ds, P4id, nn, S['base'], order)}; sc['id'].update(P.eval_refs(REF, X0, O0))
    perf = {k: dict(AUC=float(H._auc(s, y)), acc=float(((s > 0) == (y == 1)).mean())) for k, s in sc['id'].items()}
    log(f'[{ds}] unedited: ' + ', '.join(f"{k} AUC {v['AUC']:.4f} acc {v['acc']:.4f}" for k, v in perf.items() if '@' not in k))
    D31.VAR = ('D', 'cphi', 'cr', 'TN'); E = D31.run_edits(P4id, nn)
    rows3 = np.flatnonzero(nn >= 3); LAB = H.kt3_batch(np.ascontiguousarray(P4id[rows3].astype(np.float64)), nn[rows3])
    acc, obs, side = {}, {'id': (X0, O0)}, {}

    def do_variant(dc, kind, Pe, conv):
        Pv = H25.sort_pt(Pe, nn) if ds == 'top' else Pe
        r = np.flatnonzero(conv); Xv, Ov = X0.copy(), {k: v.copy() for k, v in O0.items()}
        if len(r):
            xr, orr = D31.feats(np.ascontiguousarray(Pv[r]), nn[r]); Xv[r] = xr
            for k in Ov: Ov[k][r] = orr[k]
        ok = conv & (np.abs(Ov['m'] - O0['m']) < 0.5) & (np.abs(Ov['mWk'] - O0['mWk']) < 2)
        acc[(dc, kind)] = ok; obs[(dc, kind)] = (Xv, Ov)
        v = sc[(dc, kind)] = D31.score(M, ds, Pv, nn, S['base'], order); v.update(P.eval_refs(REF, Xv, Ov))
        log(f'[{ds}] dc={dc:+.1f} {kind}: accepted {ok.sum()} / edited {conv.sum()}')

    for dc in DCS:
        for kind in ('D', 'cphi', 'cr'):
            e = E[(dc, kind)]; do_variant(dc, kind, e['P'], e['conv'])
        tauD = obs[(dc, 'D')][1]['tau32']
        Pc, cc, si = P.build_cside(P4id, nn, LAB, rows3, E[(dc, 'D')]['P'], E[(dc, 'D')]['conv'], O0['tau32'], tauD)
        do_variant(dc, 'cside', Pc, cc)
        Pc2, cc2, si2 = build_cside2(P4id, nn, LAB, rows3, E[(dc, 'D')]['P'], E[(dc, 'D')]['conv'], O0['tau32'], tauD)
        do_variant(dc, 'cside2', Pc2, cc2); side[dc] = (si, si2)
    return dict(y=y, perf=perf, sc=sc, acc=acc, obs=obs, side=side, E=E)


# ================================================================================================================ statistics
def ci(x, bi):
    return np.percentile(x[bi].mean(1), [2.5, 97.5]).tolist()


def cell_stats(ds, Rr, POOL, rng):
    y, sc, acc, obs = Rr['y'], Rr['sc'], Rr['acc'], Rr['obs']; X0, O0 = obs['id']
    allref = sorted(k for k in sc['id'] if k.split('@')[0] in REFS); models = [m for m in NETS if m in sc['id']] + allref
    out = {}
    for dc in DCS:
        for ctrl in ('cside', 'cside2'):
            rows = np.flatnonzero(acc[(dc, 'D')] & acc[(dc, 'cphi')] & acc[(dc, ctrl)] & (y == 1)); n = len(rows)
            R = dict(n=int(n)); key = f'{dc}|{ctrl}'
            if n < 10: out[key] = R; continue
            bi = rng.integers(0, n, (B, n))
            R['obs'] = {}
            for q in ('cos', 'tau32', 'mbq_lo', 'mbq_hi', 'mSD', 'tau21'):
                R['obs'][q] = {k: float((obs[(dc, k)][1][q][rows] - O0[q][rows]).mean()) for k in ('D', 'cphi', ctrl)}
            R['obs']['sorted_frac'] = {k: (obs[(dc, k)][0][rows, 11:14] - X0[rows, 11:14]).mean(0).tolist() for k in ('D', 'cphi', ctrl)}
            R['tau32_match_frac'] = R['obs']['tau32'][ctrl] / R['obs']['tau32']['D'] if abs(R['obs']['tau32']['D']) > 1e-9 else None
            R['models'] = {}
            xs = {m: (sc[(dc, 'D')][m] - sc[(dc, ctrl)][m])[rows] for m in models}
            for m in models:
                x = xs[m]; Mm = dict(D_minus_ctrl=float(x.mean()), ci95=ci(x, bi))
                z = (sc[(dc, 'cr')][m] - sc[(dc, 'cphi')][m])[rows]; Mm['cr_minus_cphi'] = float(z.mean()); Mm['cr_minus_cphi_ci95'] = ci(z, bi)
                if m in NETS:
                    Mm['p'] = D31.perm_p2(x, rng, NPERM)
                    for base in ('MLPx', 'MLP5'):
                        dd = []
                        for r_ in [k for k in allref if k.split('@')[0] == base]:
                            d = x - xs[r_]; dd.append(dict(mean=float(d.mean()), ci95=ci(d, bi), p=D31.perm_p2(d, rng, NPERM)))
                        Mm[f'minus_{base}'] = dd; Mm[f'minus_{base}_pmax'] = max(v['p'] for v in dd)
                R['models'][m] = Mm
            if ctrl == 'cside2' or dc > 0:
                Xp, Op, yp = POOL; pm = yp == 1; Opc = {k: v[pm] for k, v in Op.items()}
                sub = rows if n <= 1500 else np.sort(rng.permutation(rows)[:1500])
                R['c2st'] = {k: D31.matched_c2st(obs[(dc, k)][0][sub], {kk: v[sub] for kk, v in obs[(dc, k)][1].items()}, Xp[pm], Opc)
                             for k in ('D', ctrl)}
            out[key] = R
            log(f"[{ds}] {key}: n {n}; dtau32 D {R['obs']['tau32']['D']:+.4f} {ctrl} {R['obs']['tau32'][ctrl]:+.4f}; dcos D "
                f"{R['obs']['cos']['D']:+.3f} {ctrl} {R['obs']['cos'][ctrl]:+.3f}; C2ST {R.get('c2st')}; "
                + ' '.join(f"{m} {R['models'][m]['D_minus_ctrl']:+.2f}" for m in models if '@' not in m))
    return out


def design(Rr, ctrl, dcs, ext):
    y, sc, acc, obs = Rr['y'], Rr['sc'], Rr['acc'], Rr['obs']
    Xs, rows_all, cells = [], [], []
    for dc in dcs:
        r = np.flatnonzero(acc[(dc, 'D')] & acc[(dc, 'cphi')] & acc[(dc, ctrl)] & (y == 1))
        XD, OD = obs[(dc, 'D')]; XC, OC = obs[(dc, ctrl)]
        cols = [np.ones(len(r)), (OD['cos'] - OC['cos'])[r], (OD['tau32'] - OC['tau32'])[r], (XD[:, 13] - XC[:, 13])[r], (XD[:, 11] - XC[:, 11])[r]]
        if ext:
            cols += [(OD['tau32'] - OC['tau32'])[r] ** 2, (OD['tau21'] - OC['tau21'])[r], (OD['mSD'] - OC['mSD'])[r] / 100.0,
                     (XD[:, 12] - XC[:, 12])[r]] + [(XD[:, k] - XC[:, k])[r] / 100.0 for k in (14, 15, 16)]
        Xs.append(np.column_stack(cols)); rows_all.append(r); cells += [dc] * len(r)
    return np.concatenate(Xs).astype(np.float64), rows_all, np.array(cells)


def regress(ds, Rr, rng):
    sc = Rr['sc']; allref = sorted(k for k in sc['id'] if k.split('@')[0] in REFS)
    models = [m for m in NETS if m in sc['id']] + allref; out = {}
    for ctrl in ('cside', 'cside2'):
        for vname, dcs, ext in (('all', DCS, False), ('pos', [d for d in DCS if d > 0], False), ('neg', [d for d in DCS if d < 0], False),
                                ('ext', DCS, True), ('ext_pos', [d for d in DCS if d > 0], True)):
            Xm, rows_all, cells = design(Rr, ctrl, dcs, ext); n = len(Xm)
            if n < 30: out[f'{ctrl}|{vname}'] = dict(n=int(n)); continue
            Y = np.column_stack([np.concatenate([(sc[(dc, 'D')][m] - sc[(dc, ctrl)][m])[r] for dc, r in zip(dcs, rows_all)]) for m in models]).astype(np.float64)
            coef = np.linalg.lstsq(Xm, Y, rcond=None)[0]                                  # (k, models)
            bi = rng.integers(0, n, (B, n)); bc = np.stack([np.linalg.lstsq(Xm[i], Y[i], rcond=None)[0][1] for i in bi])   # (B, models)
            S = dict(n=int(n), n_per_dc={str(dc): int((cells == dc).sum()) for dc in dcs}, mean_dcos=float(Xm[:, 1].mean()),
                     mean_dtau32=float(Xm[:, 2].mean()), corr_dcos_dtau32=float(np.corrcoef(Xm[:, 1], Xm[:, 2])[0, 1]), models={})
            idx = {m: i for i, m in enumerate(models)}
            for m in models:
                i = idx[m]; v = dict(b_cos=float(coef[1, i]), b_tau=float(coef[2, i]), ci95=np.percentile(bc[:, i], [2.5, 97.5]).tolist(),
                                     p=float(max(2 * min((bc[:, i] <= 0).mean(), (bc[:, i] >= 0).mean()), 1 / B)))
                if m in NETS:
                    for base in ('MLPx', 'MLP5'):
                        dd = []
                        for r_ in [k for k in allref if k.split('@')[0] == base]:
                            d = bc[:, i] - bc[:, idx[r_]]
                            dd.append(dict(diff=float(coef[1, i] - coef[1, idx[r_]]), ci95=np.percentile(d, [2.5, 97.5]).tolist(),
                                           p=float(max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / B))))
                        v[f'minus_{base}'] = dd; v[f'minus_{base}_pmax'] = max(x['p'] for x in dd)
                S['models'][m] = v
            for base in REFS:
                vv = [S['models'][m]['b_cos'] for m in allref if m.split('@')[0] == base]; S[f'{base}_range'] = [min(vv), max(vv)]
            out[f'{ctrl}|{vname}'] = S
            log(f'[{ds}] regression {ctrl}|{vname} n {n}: ' + ' '.join(f"{m} {S['models'][m]['b_cos']:+.2f}{np.round(S['models'][m]['ci95'], 2)}"
                                                                    for m in models if m in NETS)
                + f" | MLPx {np.round(S['MLPx_range'], 2)} MLP5 {np.round(S['MLP5_range'], 2)}")
    return out


def matched_subset(ds, Rr, rng, tol=0.01):
    """non-parametric: jets whose D - ctrl residual tau32 / f_hi / f_lo mismatch is below tol; raw D - ctrl there"""
    y, sc, acc, obs = Rr['y'], Rr['sc'], Rr['acc'], Rr['obs']; allref = sorted(k for k in sc['id'] if k.split('@')[0] in REFS)
    models = [m for m in NETS if m in sc['id']] + allref; out = {}
    for ctrl in ('cside', 'cside2'):
        for sgn, dcs in (('pos', [d for d in DCS if d > 0]), ('neg', [d for d in DCS if d < 0])):
            xs = {m: [] for m in models}; dcos = []
            for dc in dcs:
                XD, OD = obs[(dc, 'D')]; XC, OC = obs[(dc, ctrl)]
                ok = acc[(dc, 'D')] & acc[(dc, 'cphi')] & acc[(dc, ctrl)] & (y == 1) & (np.abs(OD['tau32'] - OC['tau32']) < tol) \
                    & (np.abs(XD[:, 13] - XC[:, 13]) < tol) & (np.abs(XD[:, 11] - XC[:, 11]) < tol)
                r = np.flatnonzero(ok); dcos.append((OD['cos'] - OC['cos'])[r])
                for m in models: xs[m].append((sc[(dc, 'D')][m] - sc[(dc, ctrl)][m])[r])
            dcos = np.concatenate(dcos); n = len(dcos); R = dict(n=int(n))
            if n < 15: out[f'{ctrl}|{sgn}'] = R; continue
            xs = {m: np.concatenate(v) for m, v in xs.items()}; bi = rng.integers(0, n, (B, n))
            R['mean_dcos'] = float(dcos.mean()); R['models'] = {}
            for m in models:
                Mm = dict(mean=float(xs[m].mean()), ci95=ci(xs[m], bi))
                if m in NETS:
                    Mm['p'] = D31.perm_p2(xs[m], rng, NPERM)
                    for base in ('MLPx', 'MLP5'):
                        dd = [dict(mean=float((xs[m] - xs[r_]).mean()), ci95=ci(xs[m] - xs[r_], bi), p=D31.perm_p2(xs[m] - xs[r_], rng, NPERM))
                              for r_ in allref if r_.split('@')[0] == base]
                        Mm[f'minus_{base}'] = dd; Mm[f'minus_{base}_pmax'] = max(v['p'] for v in dd)
                R['models'][m] = Mm
            for base in REFS:
                vv = [R['models'][m]['mean'] for m in allref if m.split('@')[0] == base]; R[f'{base}_range'] = [min(vv), max(vv)]
            out[f'{ctrl}|{sgn}'] = R
            log(f'[{ds}] matched subset {ctrl}|{sgn}: n {n}, mean dcos {dcos.mean():+.3f}: ' + ' '.join(
                f"{m} {R['models'][m]['mean']:+.2f}" for m in models if m in NETS) + f" | MLPx {np.round(R['MLPx_range'], 2)} MLP5 {np.round(R['MLP5_range'], 2)}")
    return out


def holm_all(V):
    """Holm across networks x samples within each family, on the conservative (max over retrains) p against MLPx"""
    fam = {}
    for ds, S in V['sets'].items():
        for blk in ('regression', 'matched', 'cells'):
            for key, R in S[blk].items():
                for m, Mm in R.get('models', {}).items():
                    if m in NETS and 'minus_MLPx_pmax' in Mm: fam.setdefault(f'{blk}|{key}', []).append((ds, m, Mm))
    for f, lst in fam.items():
        adj = H._holm([Mm['minus_MLPx_pmax'] for _, _, Mm in lst])
        for (_, _, Mm), a in zip(lst, adj): Mm['minus_MLPx_pmax_holm'] = float(a)


def main():
    RES.mkdir(parents=True, exist_ok=True)
    log(f'H32 verify start; smoke={SMOKE}, per class {PER}, seed {SEED}, retrains {P.NSEED}')
    P.selftest(); selftest2()
    SAMP = {'top': load_top(), 'jc': load_jc()}
    REF, POOL = {}, {}
    for ds in ('jc', 'top'):
        Xr, Or, yr = D31.ref_train(ds); REF[ds], _ = P.fit_refs_seeds(Xr, Or, yr); POOL[ds] = (Xr, Or, yr)
    M = X.Models()
    Pm = SAMP['top']['P4'].astype(np.float64); real = np.arange(Pm.shape[1])[None, :] < SAMP['top']['nn'][:, None]
    mm = (Pm[..., 3] ** 2 - (Pm[..., :3] ** 2).sum(-1))[real]; M.massless = bool(np.mean(np.sqrt(np.maximum(mm, 0)) > 0.1) < 0.01)
    V = dict(node='H32-verify', smoke=SMOKE, seed=SEED, samples={ds: S['info'] for ds, S in SAMP.items()},
             cside2=dict(n_candidates=48 * (1 + len(RHO2) * NAZ2), kcand=K2, sigma_frac=SF2, sigma_tau32=ST2, rho=RHO2, n_azimuth=NAZ2), sets={})
    rng = np.random.default_rng(3311)
    for ds, S in SAMP.items():
        Rr = run_sample(ds, S, M, REF[ds])
        V['sets'][ds] = dict(unedited=Rr['perf'], cells=cell_stats(ds, Rr, POOL[ds], rng), regression=regress(ds, Rr, rng),
                             matched=matched_subset(ds, Rr, rng),
                             cside_reasons={str(dc): [si['reasons'], si2['reasons']] for dc, (si, si2) in Rr['side'].items()})
        (RES / f'{TAG}.json').write_text(json.dumps(V, indent=1, default=float)); log(f'checkpoint after {ds}')
        del Rr
    holm_all(V)
    (RES / f'{TAG}.json').write_text(json.dumps(V, indent=1, default=float)); log(f'wrote {TAG}.json'); log('done')


if __name__ == '__main__':
    main()
