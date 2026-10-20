"""H25v (loop D, skeptic pass on H25): fresh-row replication of the subjet-width edit S(s) and a new control that changes
tau32 at FIXED subjet masses.

Why: H25's S(s) scales every constituent's offset from its subjet centroid by s. At fixed constituent pT this changes the
subjet masses together with tau32 (subjet mass^2 ~ pT^2 x pT-weighted angular variance), so S cannot tell whether a network
reads tau32 (linear in the constituent-to-axis distances) or the subjet masses / second moments (quadratic).

Samples (fresh rows, random, never first-k; the H25 rows are excluded)
  top  Top Landscape test split: the H25 rows (data_top.sample(3000, 9733) rng calls) removed, then 2,000 per class, rng 52025.
  jc   JetClass TTBar_100 / ZJetsToNuNu_100, stored 550 < pT < 650 GeV, |eta| < 2, >= 3 particles: the H25 rows (rng 9733 / 9734,
       3,000 each) removed, then 2,000 per class, rng 52025 / 52026. (TTBar_101 / ZJetsToNuNu_101 hold the reference-fit jets.)
Edits
  S(s), c1(s), s in {0.7, 1.4}: H25's own edit_jet (unchanged code, h25_top_tau32.edit_jet).
  P(k), k in {0.5, 1.6}: profile edit at fixed subjet mass. Inside each exclusive-kT subjet, every constituent's offset d_i
       from the subjet pT-weighted centroid keeps its direction; its length rho_i -> lam * rb * (rho_i / rb)^k (rb = largest
       rho in that subjet), lam in [0.02, 5] solved by bisection so that the subjet invariant mass equals its unedited value (|dm_k| <
       0.05 GeV). k < 1 flattens the radial profile (pT-weighted mean distance up at fixed second moment, so tau3 and tau32 up);
       k > 1 sharpens it (tau32 down). Jets with an edited constituent beyond dR = 1.0 of the jet axis (1.2 after the restore) are
       rejected. Then the H25 rigid restore (W pair opened, third subjet moved) brings back mW and m.
  cP(k): H25's c1 rotation (random sign per subjet about the subjet centroid) with the pT-weighted mean displacement of P(k)
       on the same jet, then the same restore.
  Acceptance as H25: |dm| < 0.5 GeV and |dmW| < 2 GeV after re-clustering. Analysis set per (k or s) = edit and its control
  both accepted. Every edited jet is rebuilt about its own constituent-sum axis (h25.score).
Predictions for P(k) - cP(k), at fixed m, mW and subjet masses:
  H_tau  (net reads tau32 / first-moment width): sign of MLP3's response, size ratio to MLP3 like that under S.
  H_mass (net reads subjet masses / second moments, as MLPx does): response ~ MLPx's (~0).
Statistics: mean difference over jets with 95 % bootstrap CI (B = 2000), sign-flip permutation p (5000, two-sided), Holm over
  networks x edits x samples x classes; paired net - MLPx and net - (ratio_S x MLP3) tests.
Run: ./sync.sh && ./part_job.sh d-h25-verify 'python -m interp.h25_verify'   (H25V_SMOKE=1: 150 per class, 2,000 ref jets)
Output: $AI_RESULTS/physics/h25_verify.json
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
from interp import h25_top_tau32 as T         # noqa: E402

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
SMOKE = bool(os.environ.get('H25V_SMOKE'))
RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'physics'
TAG = 'h25_verify_smoke' if SMOKE else 'h25_verify'
PER = 150 if SMOKE else 2000
X.NREF = 2000 if SMOKE else 25000
B = 300 if SMOKE else 2000
NPERM = 500 if SMOKE else 5000
SS = (0.7, 1.4)
KS = (0.5, 1.6)
NCPU = int(os.environ.get('OMP_NUM_THREADS', '4'))
NETS = T.NETS
REFS = ['MLP3', 'LR3', 'LR2', 'MLPx', 'MLPxT']
CONF = [(f'S{s}', f'c1_{s}') for s in SS] + [(f'P{k}', f'cP{k}') for k in KS]


# ================================================================================================================ edits
def profile_edit(P, lab, k, seed):
    """P(k) and its matched rotation control cP(k) on one jet; returns {kind: (P_new or None, D, reason, dmk_max)}"""
    P, pt, m, x, y, eJ, pJ = T._jet(P); rng = np.random.default_rng(seed); out = {}
    sel = [lab == q for q in range(3)]
    if min(s_.sum() for s_ in sel) < 1: return {kd: (None, 0.0, 'empty_subjet', None) for kd in ('P', 'cP')}
    Vk = [P[s_].sum(0) for s_ in sel]; mk0 = [H._minv(v) for v in Vk]
    pm = {(i, j): H._minv(Vk[i] + Vk[j]) for i, j in ((0, 1), (0, 2), (1, 2))}
    w = min(pm, key=lambda p: abs(pm[p] - H.MWREF)); mW0 = pm[w]; m0 = H._minv(P.sum(0))
    cen = np.array([[(pt[s_] * x[s_]).sum() / pt[s_].sum(), (pt[s_] * y[s_]).sum() / pt[s_].sum()] for s_ in sel])
    rho = np.hypot(x - cen[lab, 0], y - cen[lab, 1]); mrho = float((pt * rho).sum() / pt.sum())
    disp = lambda x2, y2: float((pt * np.hypot(x2 - x, y2 - y)).sum() / pt.sum())
    x2, y2 = x.copy(), y.copy()
    for q, s_ in enumerate(sel):
        r = rho[s_]
        if s_.sum() < 2 or r.max() <= 0: continue
        rb = float(r.max())   # normalisation by the subjet's largest offset: r^k <= r for k > 1, so lam stays O(1) (no blow-up of the tail)
        g = np.where(r > 0, rb * (np.maximum(r, 1e-12) / rb) ** k / np.maximum(r, 1e-12), 0.0)
        dx, dy = x[s_] - cen[q, 0], y[s_] - cen[q, 1]
        mk = lambda lam: H._minv(T._p4(pt[s_], m[s_], eJ + cen[q, 0] + lam * g * dx, pJ + cen[q, 1] + lam * g * dy).sum(0)) - mk0[q]
        lam = H._bisect(mk, 0.02, 5.0, tol=0.05)
        if lam is None: return {kd: (None, 0.0, 'subjet_mass_no_conv', None) for kd in ('P', 'cP')}
        x2[s_] = cen[q, 0] + lam * g * dx; y2[s_] = cen[q, 1] + lam * g * dy
    if not np.all(np.isfinite(x2) & np.isfinite(y2)) or np.hypot(x2, y2).max() > 1.0:
        return {kd: (None, 0.0, 'outside_dR1', None) for kd in ('P', 'cP')}

    def finish(kind, xa, ya):
        xy = T._restore(pt, m, xa, ya, lab, eJ, pJ, w, mW0, m0)
        if xy is None: out[kind] = (None, disp(xa, ya), 'restore_no_conv', None); return
        if not np.all(np.isfinite(xy[0]) & np.isfinite(xy[1])) or np.hypot(xy[0], xy[1]).max() > 1.2:
            out[kind] = (None, disp(xa, ya), 'outside_dR1.2', None); return
        Pn = T._p4(pt, m, eJ + xy[0], pJ + xy[1])
        dmk = max(abs(H._minv(Pn[s_].sum(0)) - mk0[q]) for q, s_ in enumerate(sel))
        out[kind] = (Pn, disp(*xy), 'ok', float(dmk))
    finish('P', x2, y2)
    DS = out['P'][1] if out['P'][0] is not None else float((pt * np.hypot(x2 - x, y2 - y)).sum() / pt.sum())
    al = 2 * np.arcsin(min(1.0, DS / (2 * max(mrho, 1e-9)))); sg = rng.choice([-1.0, 1.0], 3)[lab]
    ca, sa = np.cos(al * sg), np.sin(al * sg); dx, dy = x - cen[lab, 0], y - cen[lab, 1]
    finish('cP', cen[lab, 0] + ca * dx - sa * dy, cen[lab, 1] + sa * dx + ca * dy)
    return out


def _work(arg):
    j, P, lab = arg; r = {}
    for i, s in enumerate(SS):
        o = T.edit_jet(P, lab, s, 52025 * 1000 + j * 10 + i)
        r[f'S{s}'] = o['S'][:2] + (o['S'][2], None); r[f'c1_{s}'] = o['c1'][:2] + (o['c1'][2], None)
    for i, k in enumerate(KS):
        o = profile_edit(P, lab, k, 52026 * 1000 + j * 10 + i)
        r[f'P{k}'] = o['P']; r[f'cP{k}'] = o['cP']
    return r


def run_edits(P4, nn):
    import multiprocessing as mp
    rows = np.flatnonzero(nn >= 3); F = np.zeros((len(rows), P4.shape[1], 4), np.float64); F[:] = P4[rows]
    LAB = H.kt3_batch(F, nn[rows]); log(f'  kT subjets for {len(rows)} jets')
    jobs = [(int(j), P4[j, :nn[j]].astype(np.float64), LAB[ii, :nn[j]]) for ii, j in enumerate(rows)]
    with mp.get_context('fork').Pool(NCPU) as pool:
        res = pool.map(_work, jobs, chunksize=16)
    E = {}
    kinds = [a for c in CONF for a in c]
    for kind in kinds:
        Pv = P4.copy(); conv = np.zeros(len(nn), bool); D = np.zeros(len(nn)); dmk = np.full(len(nn), np.nan); why = {}
        for j, r in zip(rows, res):
            Pn, d, reason, dm = r[kind]; why[reason] = why.get(reason, 0) + 1; D[j] = d
            if dm is not None: dmk[j] = dm
            if Pn is not None: Pv[j, :nn[j]] = Pn.astype(np.float32); conv[j] = True
        E[kind] = dict(P=Pv, conv=conv, D=D, dmk=dmk, reasons=why)
        log(f'  {kind}: converged {conv.sum()}/{len(nn)} {why}')
    return E


# ================================================================================================================ samples
def load_top():
    from interp import data_top as dt
    d = dt.DST / 'test'; y = np.load(d / 'label.npy')
    r0 = np.random.default_rng(9733); old = np.concatenate([r0.choice(np.flatnonzero(y == c), 3000, replace=False) for c in (0, 1)])
    ok = np.ones(len(y), bool); ok[old] = False; rng = np.random.default_rng(52025)
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero((y == c) & ok), PER, replace=False) for c in (0, 1)]))
    assert not np.isin(idx, old).any()
    full = H._full_test(); jet = np.load(d / 'jet.npy')
    nn = np.minimum(jet[idx, 5].astype(np.int64), full.shape[1])
    P4 = H._identity(full[idx][..., :4].astype(np.float32), nn).astype(np.float32); yy = y[idx].astype(int)
    log(f'top fresh sample: {len(idx)} jets ({yy.sum()} top), rng 52025, H25 rows excluded ({len(old)})')
    return dict(P4=P4, nn=nn, y=yy, base=X.top_base(P4, nn), info=dict(n=int(len(idx)), n_top=int(yy.sum()), seed=52025, excluded_h25=int(len(old))))


def jc_read_fresh(c, per, old_seed, seed):
    import uproot
    from interp.part_core import read_root, cat, sub
    f = X.jc_file(c, 100); t = uproot.open(f)['tree']
    a = t.arrays(['jet_pt', 'jet_eta', 'jet_nparticles'], library='np')
    el = np.flatnonzero((a['jet_pt'] > 550) & (a['jet_pt'] < 650) & (np.abs(a['jet_eta']) < 2) & (a['jet_nparticles'] >= 3))
    old = np.random.default_rng(old_seed).choice(el, 3000, replace=False)      # the H25 draw (jc_read rng calls)
    el2 = np.setdiff1d(el, old); idx = np.sort(np.random.default_rng(seed).choice(el2, per, replace=False)); N = t.num_entries; parts = []
    for s in range(0, N, 20000):
        loc = idx[(idx >= s) & (idx < s + 20000)] - s
        if len(loc): parts.append(sub(read_root(f, start=s, stop=min(s + 20000, N)), loc))
    log(f'jc {c} {f.name}: {len(el)} eligible, {len(old)} H25 rows excluded, drew {len(idx)} (rng {seed})')
    return cat(parts), dict(file=f.name, eligible=int(len(el)), excluded_h25=int(len(old)), drawn=int(len(idx)), seed=seed)


def load_jc():
    from interp.part_core import cat, kin
    Jt, it = jc_read_fresh('Tbqq', PER, 9733, 52025); Jq, iq = jc_read_fresh('QCD', PER, 9734, 52026)
    J = cat([Jt, Jq]); P4raw, nn = X.jc_P4(J); P4 = H._identity(P4raw, nn).astype(np.float32)
    J['pmass'] = kin(J)[3].astype(np.float32); y = (J['label'] == T.TBQQ).astype(int)
    return dict(P4=P4, nn=nn, y=y, base=J, info=dict(Tbqq=it, QCD=iq))


# ================================================================================================================ stats
def perm_p2(x, rng, n=NPERM):
    x = np.asarray(x, float); obs = abs(x.mean()); out = 0
    for s0 in range(0, n, 500):
        k = min(500, n - s0); sg = rng.choice([-1.0, 1.0], (k, len(x))); out += int(np.sum(np.abs((sg * x).mean(1)) >= obs))
    return float((out + 1) / (n + 1))


def boot(x, bi):
    return [float(v) for v in np.percentile(x[bi].mean(1), [2.5, 97.5])]


def run_sample(ds, S, M, REF, POOL):
    nn, y = S['nn'], S['y']; P4id = S['P4']
    if ds == 'top': P4id = T.sort_pt(P4id, nn); S['P4'] = P4id
    X0, O0 = H._features(P4id, nn)
    order = np.random.default_rng(605).permutation(len(nn))
    sc = {'id': T.score(M, ds, P4id, nn, S['base'], order)}; sc['id'].update(T.eval_refs(REF, X0, O0))
    perf = {k: dict(AUC=float(H._auc(v, y)), acc=float(((v > 0) == (y == 1)).mean())) for k, v in sc['id'].items()}
    log(f'[{ds}] unedited: ' + ', '.join(f"{k} AUC {v['AUC']:.4f} acc {v['acc']:.3f}" for k, v in perf.items()))
    E = run_edits(P4id, nn); acc = {}; obs = {'id': (X0, O0)}; info = {}
    for kind, e in E.items():
        Pv = T.sort_pt(e['P'], nn) if ds == 'top' else e['P']
        r = np.flatnonzero(e['conv']); Xv, Ov = X0.copy(), {k: v.copy() for k, v in O0.items()}
        if len(r):
            xr, orr = H._features(np.ascontiguousarray(Pv[r]), nn[r]); Xv[r] = xr
            for k in Ov: Ov[k][r] = orr[k]
        ok = e['conv'] & (np.abs(Ov['m'] - O0['m']) < 0.5) & (np.abs(Ov['mW'] - O0['mW']) < 2)
        acc[kind] = ok; obs[kind] = (Xv, Ov)
        sc[kind] = T.score(M, ds, Pv, nn, S['base'], order); sc[kind].update(T.eval_refs(REF, Xv, Ov))
        info[kind] = dict(converged=int(e['conv'].sum()), accepted=int(ok.sum()), reasons=e['reasons'],
                          dmk_solver_max_median=float(np.nanmedian(e['dmk'][ok])) if np.isfinite(e['dmk'][ok]).any() else None)
        log(f'[{ds}] {kind}: accepted {ok.sum()} / converged {e["conv"].sum()}')
    rng = np.random.default_rng(11); out = dict(unedited=perf, edit_info=info, conf={})
    for ed, ct in CONF:
        both = acc[ed] & acc[ct]
        for cls, cname in ((1, 'top'), (0, 'qcd')):
            rows = np.flatnonzero(both & (y == cls)); n = len(rows); key = f'{ed}|{cname}'; R = dict(n=int(n), n_class=int((y == cls).sum()))
            if n < 10: out['conf'][key] = R; continue
            bi = rng.integers(0, n, (B, n)); R['obs'] = {}
            for q in ('tau32', 'tau3', 'tau21', 'mSD', 'mW', 'm', 'n'):
                de = obs[ed][1][q][rows] - O0[q][rows]; dc = obs[ct][1][q][rows] - O0[q][rows]
                R['obs'][q] = dict(id_median=float(np.median(O0[q][rows])), d_edit_mean=float(de.mean()), d_ctrl_mean=float(dc.mean()),
                                   edit_minus_ctrl=float((de - dc).mean()), ci95=boot(de - dc, bi))
            for nm, cols in (('subjet_mass_sum', slice(14, 17)), ('pair_dR_sum', slice(17, 20)), ('pair_mass_sum', slice(8, 11))):
                de = obs[ed][0][rows, cols].sum(1) - X0[rows, cols].sum(1); dc = obs[ct][0][rows, cols].sum(1) - X0[rows, cols].sum(1)
                R['obs'][nm] = dict(d_edit_median=float(np.median(de)), d_edit_mean=float(de.mean()), d_ctrl_mean=float(dc.mean()),
                                    d_edit_median_abs=float(np.median(np.abs(de))))
            R['D_median'] = {ed: float(np.median(E[ed]['D'][rows])), ct: float(np.median(E[ct]['D'][rows]))}
            diff = {mdl: (sc[ed][mdl] - sc[ct][mdl])[rows] for mdl in NETS + REFS if mdl in sc['id']}
            R['models'] = {}
            for mdl, x in diff.items():
                Mm = dict(edit_minus_ctrl=float(x.mean()), ci95=boot(x, bi), p_perm=perm_p2(x, rng),
                          d_edit=float((sc[ed][mdl] - sc['id'][mdl])[rows].mean()), d_ctrl=float((sc[ct][mdl] - sc['id'][mdl])[rows].mean()))
                if mdl in NETS:
                    Mm['p_perm_vs_MLPx'] = perm_p2(x - diff['MLPx'], rng)
                    den = diff['MLP3'][bi].mean(1)
                    Mm['ratio_MLP3'] = float(x.mean() / diff['MLP3'].mean()) if abs(diff['MLP3'].mean()) > 1e-6 else None
                    Mm['ratio_MLP3_ci95'] = [float(v) for v in np.nanpercentile(x[bi].mean(1) / np.where(np.abs(den) > 1e-6, den, np.nan), [2.5, 97.5])]
                R['models'][mdl] = Mm
            if cls == 1 and ed.startswith('P'):
                Xp, Op, yp = POOL; pm = yp == cls; Opc = {k: v[pm] for k, v in Op.items()}
                sub = rows if n <= 1500 else np.sort(rng.permutation(rows)[:1500])
                R['c2st'] = {kd: T.matched_c2st(obs[kv][0][sub], {k: v[sub] for k, v in obs[kv][1].items()}, Xp[pm], Opc)[0]
                             for kd, kv in (('id', 'id'), (ed, ed), (ct, ct))}
            out['conf'][key] = R
            o = R['obs']
            log(f"[{ds}] {key}: n {n}; d(tau32) {o['tau32']['edit_minus_ctrl']:+.4f} d(tau3) {o['tau3']['edit_minus_ctrl']:+.4f} "
                f"d(subjet m sum) med {o['subjet_mass_sum']['d_edit_median']:+.2f} |.| {o['subjet_mass_sum']['d_edit_median_abs']:.2f} "
                f"dmSD {o['mSD']['edit_minus_ctrl']:+.2f} c2st {R.get('c2st')}")
            for mdl, Mm in R['models'].items():
                log(f"    {mdl:9s} {Mm['edit_minus_ctrl']:+.3f} {np.round(Mm['ci95'], 3)} p {Mm['p_perm']:.3g} xMLP3 {Mm.get('ratio_MLP3')} "
                    f"pvsMLPx {Mm.get('p_perm_vs_MLPx')}")
    return out


def main():
    RES.mkdir(parents=True, exist_ok=True)
    z = np.array([[1.0, 0, 5, 6], [0, 1.0, 2, 3], [0.5, 0.5, 1, 2]]); H._tau32(z[:, 0].copy(), z[:, 1].copy(), z[:, 2].copy(), z[:, 3].copy(), 3)
    log(f'H25v start; smoke={SMOKE}, per class {PER}, ref jets {X.NREF}, S s {SS}, P k {KS}')
    SAMP = {'top': load_top(), 'jc': load_jc()}
    REF, POOL, rinfo = {}, {}, {}
    for ds, fn in (('jc', X.ref_train_jc), ('top', X.ref_train_top)):
        Xr, Or, yr = fn(); REF[ds], rinfo[ds] = T.fit_refs(Xr, Or, yr); POOL[ds] = (Xr, Or, yr); log(f'refs {ds}: {rinfo[ds]}')
    M = X.Models()
    P = SAMP['top']['P4'].astype(np.float64); real = np.arange(P.shape[1])[None, :] < SAMP['top']['nn'][:, None]
    m2 = (P[..., 3] ** 2 - (P[..., :3] ** 2).sum(-1))[real]; M.massless = bool(np.mean(np.sqrt(np.maximum(m2, 0)) > 0.1) < 0.01)
    R = dict(node='H25v', smoke=SMOKE, s_values=SS, k_values=KS, samples={ds: S['info'] for ds, S in SAMP.items()}, references=rinfo, sets={})
    for ds, S in SAMP.items():
        R['sets'][ds] = run_sample(ds, S, M, REF[ds], POOL[ds])
    tests = [(ds, k, m) for ds in R['sets'] for k, r in R['sets'][ds]['conf'].items() for m in r.get('models', {}) if m in NETS]
    for pk, hk in (('p_perm', 'p_holm'), ('p_perm_vs_MLPx', 'p_holm_vs_MLPx')):
        adj = H._holm([R['sets'][ds]['conf'][k]['models'][m][pk] for ds, k, m in tests]) if tests else []
        for (ds, k, m), pa in zip(tests, adj): R['sets'][ds]['conf'][k]['models'][m][hk] = float(pa)
    R['holm_family'] = f'{len(tests)} tests: networks x edits x samples x classes'
    (RES / f'{TAG}.json').write_text(json.dumps(R, indent=1, default=float)); log(f'wrote {TAG}.json')


if __name__ == '__main__':
    main()
