"""P21v -- verifier rerun of P21 (which prong observable ParT uses) on fresh jets, with added controls.

Sample: JetClass test_20M WToQQ_105, ZToQQ_105, TTBar_105, ZJetsToNuNu_105 (Wqq, Zqq, Tbqq, QCD); PER random rows per class
  (numpy default_rng(2121), never first-k); jets with >= 4 particles. Edits / controls / acceptance exactly as P21
  (interp/p21_prong_part.edit_positions, finalize) at doses 0.6 and 1.25.
Added arms (P21 did not have them):
  nodil = the edit E without the mass-restoring dilation (jet rebuilt about its own axis only; mass changes).
  dil   = the unedited jet dilated in angle by the per-jet total factor k that the mass restore applied to E (no other change).
  The mass restore of E is a dilation of every particle (hard prongs included) by k; nodil and dil separate the edit itself from
  that dilation (additivity check: E ~ nodil + dil).
Added observables: m_hard(z) = invariant mass of the particles with pT fraction >= z (z = 0.02, the P21 soft threshold, and 0.05),
  dR12 = distance between the two k-means 2-subjet axes.
Added references: HistGradientBoosting (as P21.fit_refs, 2-fold cross-fitted) on (log m, log m_hard02, tau21, tau32, log D2) and
  (log m, log m_hard05, tau21, tau32, log D2) next to P21's (log m, tau21) / (log m, tau32, log D2) / all four.
Location: head / block patching in blocks 5-8 of ParT_full, ParT_kin, init (P21.HeadPatch, resample from the unedited jet) for
  E1-E3 at 0.6, scrF, AND the rotation controls E1rot, E3rot and the dil arm of E3 (do non-prong moves give the same profile?);
  NB = 2000 bootstrap resamples so 64-head Holm can reach 0.05. 1-d DAS (P18 method) on ParT_full and init for E1, E3, scrF,
  E1rot, E3rot.
  ./sync.sh && ./part_job.sh c-p21-verify 'python -m interp.p21v_verify'
  -> $AI_RESULTS/part/p21_verify.json
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
SMOKE = '--smoke' in sys.argv
_argv = sys.argv; sys.argv = [_argv[0]] + (['--smoke'] if SMOKE else [])
import interp.p21_prong_part as Q  # noqa: E402  (imports interp.p18_stage_pipeline_part as Q.P)
sys.argv = _argv
P = Q.P

SEED = 2121
FILE = '_105.root'
PER = 100 if SMOKE else 2500
Q.SV = (0.6, 1.25)
NB = 300 if SMOKE else 2000
P.NB = NB
OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'
FN = OUT / ('p21_verify_smoke.json' if SMOKE else 'p21_verify.json')
log, ci, holm, strip = P.log, P.ci, P.holm, P.strip
iQ, iZ, iW, iT = Q.iQ, Q.iZ, Q.iW, Q.iT


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def finalize_k(Jb, pt, x, y, ms, m0, iters=6, kfix=None):
    """as P21.finalize, also returning the total dilation factor; iters=0 -> rebuild about own axis only; kfix -> dilate the
    given positions by kfix about the E-scheme axis once (no mass restore)"""
    from interp.part_core import build
    real = pt > 0; pt = pt.astype(np.float64); ms = ms.astype(np.float64)
    sg = np.where(Jb['jet_eta'] < 0, -1.0, 1.0)[:, None]
    eta = Jb['jet_eta'][:, None].astype(np.float64) + sg * x; phi = Jb['jet_phi'][:, None].astype(np.float64) + y

    def axis(eta, phi):
        px, py, pz = pt * np.cos(phi) * real, pt * np.sin(phi) * real, pt * np.sinh(eta) * real
        E = np.sqrt((pt * np.cosh(eta)) ** 2 + ms ** 2) * real; S = [a.sum(1) for a in (px, py, pz, E)]
        return np.arcsinh(S[2] / np.maximum(np.hypot(S[0], S[1]), 1e-9)), np.arctan2(S[1], S[0]), \
            np.sqrt(np.maximum(S[3] ** 2 - S[0] ** 2 - S[1] ** 2 - S[2] ** 2, 0))
    ktot = np.ones(len(pt))
    if kfix is not None:
        eJ, pJ, _ = axis(eta, phi); k = kfix[:, None]
        eta = eJ[:, None] + k * (eta - eJ[:, None]); phi = pJ[:, None] + k * wrap(phi - pJ[:, None]); ktot = kfix.copy()
    for it in range(iters):
        eJ, pJ, m = axis(eta, phi)
        k = np.clip(np.where(m > 0, m0 / np.maximum(m, 1e-9), 1.0), 0.5, 2.0)
        eta = eJ[:, None] + k[:, None] * (eta - eJ[:, None]); phi = pJ[:, None] + k[:, None] * wrap(phi - pJ[:, None]); ktot *= k
    eJ, pJ, m = axis(eta, phi); sgn = np.where(eJ < 0, -1.0, 1.0)[:, None]
    Jt = dict(Jb); Jt['jet_eta'] = eJ.astype(np.float32); Jt['jet_phi'] = wrap(pJ).astype(np.float32)
    de = np.where(real, sgn * (eta - eJ[:, None]), 0); dp = np.where(real, wrap(phi - pJ[:, None]), 0)
    return build(Jt, pt.astype(np.float32), de.astype(np.float32), dp.astype(np.float32), ms.astype(np.float32)), ktot


def extra_obs(torch, dev, J):
    """m_hard(0.02), m_hard(0.05) and the 2-subjet axis distance dR12 (k-means, as the tau axes)"""
    from interp.part_core import kin
    m = J['mask'] > 0; pt = np.hypot(J['px'], J['py']) * m; z = pt / np.maximum(pt.sum(1, keepdims=True), 1e-12); o = {}
    for nm, thr in (('m_hard02', 0.02), ('m_hard05', 0.05)):
        s = m & (z >= thr); E, px, py, pz = ((J[k] * s).sum(1) for k in ('energy', 'px', 'py', 'pz'))
        o[nm] = np.sqrt(np.maximum(E ** 2 - px ** 2 - py ** 2 - pz ** 2, 0)).astype(np.float64)
    p, de, dp, _ = kin(J); d12 = []
    for i in range(0, len(p), 4000):
        T = lambda a: torch.from_numpy(a[i:i + 4000].astype(np.float64)).to(dev)
        ax = Q.kmeans_t(torch, T(p), T(de), T(dp), 2)[0]; d12.append((ax[:, 0] - ax[:, 1]).norm(dim=-1).cpu().numpy())
    o['dR12'] = np.concatenate(d12); return o


def fit_refs2(OB, y, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier as HGB
    from sklearn.metrics import roc_auc_score
    lg = lambda a: np.log(np.clip(a, 1e-3, 1e4))
    feats = dict(m_t21=lambda o: np.stack([np.log(o['m'] + 1), o['tau21']], 1),
                 m_t32_D2=lambda o: np.stack([np.log(o['m'] + 1), o['tau32'], lg(o['D2'])], 1),
                 m_all=lambda o: np.stack([np.log(o['m'] + 1), o['tau21'], o['tau32'], lg(o['D2'])], 1),
                 m_mh02_all=lambda o: np.stack([np.log(o['m'] + 1), np.log(o['m_hard02'] + 1), o['tau21'], o['tau32'], lg(o['D2'])], 1),
                 m_mh05_all=lambda o: np.stack([np.log(o['m'] + 1), np.log(o['m_hard05'] + 1), o['tau21'], o['tau32'], lg(o['D2'])], 1))
    tasks = dict(WZ=(np.isin(y, [iW, iZ, iQ]), np.isin(y, [iW, iZ])), TQ=(np.isin(y, [iT, iQ]), y == iT))
    fold = np.random.default_rng(seed).integers(0, 2, len(y)); M = {}; auc = {}
    for tn, (sel, lab) in tasks.items():
        for fn, f in feats.items():
            X = f(OB); ms = [HGB(max_iter=200, learning_rate=0.1, random_state=seed).fit(X[sel & (fold == k)], lab[sel & (fold == k)]) for k in (0, 1)]
            M[(tn, fn)] = (f, ms)
            pr = np.where(fold == 0, ms[1].decision_function(X), ms[0].decision_function(X)); auc[f'{tn}|{fn}'] = float(roc_auc_score(lab[sel], pr[sel]))

    def pred(o):
        return {f'{tn}|{fn}': np.where(fold == 0, ms[1].decision_function(f(o)), ms[0].decision_function(f(o))) for (tn, fn), (f, ms) in M.items()}
    return pred, auc


def main():
    import torch
    from interp.part_core import CL, PRE, cat, kin, sub, jmass, sample, inputs, build
    from interp import pn_net
    from jetdistill.part.network import FEATURES
    OUT.mkdir(parents=True, exist_ok=True)
    R = dict(node='P21v', script='interp/p21v_verify.py', smoke=SMOKE, seed=SEED, files=FILE, per_class=PER, doses=Q.SV, NB=NB)
    save = lambda: FN.write_text(json.dumps(strip(R), indent=1))
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    J0 = cat([P.load_rows(P.TEST / f'{PRE[c]}{FILE}', PER, rng) for c in Q.CLS])
    keep = np.flatnonzero((J0['mask'] > 0).sum(1) >= 4); R['n_drawn'] = int(len(J0['label'])); J0 = sub(J0, keep)
    N = len(J0['label']); y = J0['label'].astype(int); R['N'] = N; log(f'N = {N}')
    pt, de, dp, ms = kin(J0); Jb = build(J0, pt, de, dp, ms); pt, de, dp, ms = kin(Jb); m0 = jmass(Jb).astype(np.float64)
    Jbase, _ = finalize_k(Jb, pt, de.astype(np.float64), dp.astype(np.float64), ms, m0)
    M = P.make_models(['full', 'kin']); dev = M['full'].device
    PNDIR = Path(os.environ.get('JETDISTILL_MODELS', '/j-jepa-vol/aaron-interp/models/part')).parent / 'particlenet'
    PN = {n: pn_net.load_particlenet(PNDIR / f'ParticleNet_{n}.pt', len(FEATURES[n]), dev) for n in ('full', 'kin')}
    NAMES = dict(full='ParT_full', kin='ParT_kin', pnfull='PN_full', pnkin='PN_kin', init='ParT_init')

    def logits(mn, J):
        if mn.startswith('pn'): return Q.pn_logits(torch, PN[mn[2:]], J, mn[2:])
        return M[mn].run(J)['logits']
    J1 = sample(per_class=1000, start=3000); p1, e1, f1, m1 = kin(J1); J1b = build(J1, p1, e1, f1, m1)
    R['accuracy_P1_rows'] = {NAMES[mn]: float((logits(mn, J1b).argmax(1) == J1['label']).mean()) for mn in NAMES}
    log('accuracy P1 rows:', R['accuracy_P1_rows']); del J1, J1b
    # ---- edits
    POS = {}; INFO = {}
    for i in range(0, N, Q.GCHUNK):
        sl = slice(i, min(N, i + Q.GCHUNK)); pos, info, _ = Q.edit_positions(torch, dev, pt[sl], de[sl], dp[sl], SEED + i)
        for k, v in pos.items(): POS.setdefault(k, []).append(v)
        for k, v in info.items():
            for a, b in v.items(): INFO.setdefault(k, {}).setdefault(a, []).append(b)
    POS = {k: (np.concatenate([a[0] for a in v]), np.concatenate([a[1] for a in v])) for k, v in POS.items()}
    INFO = {k: {a: np.concatenate(b) for a, b in v.items()} for k, v in INFO.items()}
    obs = lambda J: {**Q.obs_np(torch, dev, J), **extra_obs(torch, dev, J)}
    OB = {'base': obs(Jbase)}; ob0 = OB['base']; LG = {mn: {'base': logits(mn, Jbase)} for mn in NAMES}; JV = {}; KT = {}
    R['accuracy_sample'] = {NAMES[mn]: float((LG[mn]['base'].argmax(1) == y).mean()) for mn in NAMES}
    for kind in Q.KINDS:
        for s in Q.SV:
            for v in ('E', 'rot', 'raw', 'nodil', 'dil'):
                nm = f'{kind}_{s}_{v}'
                if v == 'nodil': Je, _ = finalize_k(Jb, pt, *POS[(kind, s, 'E')], ms, m0, iters=0)
                elif v == 'dil': Je, _ = finalize_k(Jb, pt, de.astype(np.float64), dp.astype(np.float64), ms, m0, iters=0, kfix=KT[(kind, s)])
                else:
                    Je, kt = finalize_k(Jb, pt, *POS[(kind, s, v)], ms, m0)
                    if v == 'E': KT[(kind, s)] = kt
                assert np.array_equal(Je['mask'], Jbase['mask']), nm
                OB[nm] = obs(Je)
                for mn in NAMES:
                    if mn == 'init' and s != 0.6: continue
                    LG[mn][nm] = logits(mn, Je)
                if s == 0.6 and v in ('E', 'rot', 'dil'): JV[f'{kind}{"" if v == "E" else v}'] = Je
            log(f'  {kind} {s}: k median {np.median(KT[(kind, s)]):.3f}')
    Je = P.make_edit(Jbase, 'scr', 1.0, np.random.default_rng(SEED + 5)); OB['scrF'] = obs(Je); JV['scrF'] = Je
    for mn in NAMES: LG[mn]['scrF'] = logits(mn, Je)
    rel = lambda nm, k: OB[nm][k] / np.maximum(ob0[k], 1e-12) - 1
    ACC = {}; R['edits'] = {}
    for kind in Q.KINDS:
        for s in Q.SV:
            a = (np.abs(rel(f'{kind}_{s}_E', Q.HOLD[kind][0])) <= Q.TOL) & (np.abs(rel(f'{kind}_{s}_E', Q.HOLD[kind][1])) <= Q.TOL) & \
                (np.abs(rel(f'{kind}_{s}_E', 'm')) <= Q.MTOL)
            ACC[(kind, s)] = a
            e = dict(accepted=int(a.sum()), accepted_by_class={c: float(a[y == CL.index(c)].mean()) for c in Q.CLS},
                     k_dilation=dict(median=float(np.median(KT[(kind, s)][a])), q10=float(np.quantile(KT[(kind, s)][a], 0.1)), q90=float(np.quantile(KT[(kind, s)][a], 0.9))),
                     comp_params_median=[float(np.median(INFO[(kind, s)]['p'][a])), float(np.median(INFO[(kind, s)]['q'][a]))])
            for v in ('E', 'rot', 'raw', 'nodil', 'dil'):
                e[v] = {k: float(np.mean(rel(f'{kind}_{s}_{v}', k)[a])) for k in ('tau21', 'tau32', 'D2', 'm', 'dR12')}
                e[v].update({f'{k}_median_rel': float(np.median(rel(f'{kind}_{s}_{v}', k)[a])) for k in ('m_hard02', 'm_hard05', 'dR12')})
                e[v].update({f'{k}_mean_GeV': float(np.mean((OB[f'{kind}_{s}_{v}'][k] - ob0[k])[a])) for k in ('m', 'm_hard02', 'm_hard05')})
            R['edits'][f'{kind}_{s}'] = e
            log(f'  {kind} {s}: accepted {a.sum()}; E rel', {k: round(x, 4) for k, x in e['E'].items()}, '; k', e['k_dilation'])
    save()
    # ---- behaviour
    boot = P.Boot(torch, dev, N, NB, SEED + 21)
    sc = lambda L: dict(dWQ=L[:, iW] - L[:, iQ], dTQ=L[:, iT] - L[:, iQ])
    R['behaviour'] = {}; R['edit_vs_control'] = {}; tests = []
    for mn in NAMES:
        B0 = sc(LG[mn]['base']); bh = {}; evc = {}
        for nm, L in LG[mn].items():
            if nm == 'base': continue
            sel = np.ones(N, bool) if nm == 'scrF' else ACC[(nm.split('_')[0], float(nm.split('_')[1]))]
            S = sc(L); o = dict(n=int(sel.sum()))
            for k in ('dWQ', 'dTQ'):
                est, cc, _ = Q.bmean(boot, S[k] - B0[k], sel); o[k] = est; o[k + '_ci'] = cc
                o[k + '_by_class'] = {c: float((S[k] - B0[k])[sel & (y == CL.index(c))].mean()) for c in Q.CLS}
            bh[nm] = o
        for kind in Q.KINDS:
            for s in Q.SV:
                if f'{kind}_{s}_E' not in LG[mn]: continue
                sel = ACC[(kind, s)]; LE = LG[mn][f'{kind}_{s}_E']; o = {}
                for ctl in ('rot', 'raw', 'nodil', 'dil'):
                    LC = LG[mn][f'{kind}_{s}_{ctl}']; oc = {}
                    for k in ('dWQ', 'dTQ'):
                        d = sc(LE)[k] - sc(LC)[k]; est, cc, _ = Q.bmean(boot, d, sel)
                        oc[k] = dict(diff=est, ci=cc, p_perm=Q.perm_p(torch, dev, d[sel], seed=SEED + 7))
                    o[ctl] = oc
                    if ctl == 'rot' and mn != 'init': tests.append((mn, f'{kind}_{s}', Q.PSCORE[kind], oc[Q.PSCORE[kind]]['p_perm']))
                # additivity: E vs nodil + dil (both relative to base)
                for k in ('dWQ', 'dTQ'):
                    d = (sc(LE)[k] - B0[k]) - ((sc(LG[mn][f'{kind}_{s}_nodil'])[k] - B0[k]) + (sc(LG[mn][f'{kind}_{s}_dil'])[k] - B0[k]))
                    o.setdefault('additivity_residual', {})[k] = dict(mean=Q.bmean(boot, d, sel)[0], ci=Q.bmean(boot, d, sel)[1])
                evc[f'{kind}_{s}'] = o
        R['behaviour'][NAMES[mn]] = bh; R['edit_vs_control'][NAMES[mn]] = evc
        log(f'  behaviour {NAMES[mn]}: ' + '; '.join(f'{n} dWQ {v["dWQ"]:+.3f} dTQ {v["dTQ"]:+.3f}' for n, v in bh.items()))
    for (mn, ks, k, p), h in zip(tests, holm([t[3] for t in tests])):
        R['edit_vs_control'][NAMES[mn]][ks]['rot'][k]['p_holm'] = h
    save()
    # ---- references
    pred, auc = fit_refs2(ob0, y, SEED + 3); R['ref_auc'] = auc; P0 = pred(ob0); R['reference'] = {}
    for kind in Q.KINDS:
        for s in Q.SV:
            sel = ACC[(kind, s)]; o = {}
            for v in ('E', 'rot', 'raw', 'nodil', 'dil'):
                Pv = pred(OB[f'{kind}_{s}_{v}']); o[v] = {nm: Q.bmean(boot, Pv[nm] - P0[nm], sel)[:2] for nm in Pv}
                o[v + '_by_class'] = {nm: {c: float((Pv[nm] - P0[nm])[sel & (y == CL.index(c))].mean()) for c in Q.CLS} for nm in ('WZ|m_all', 'WZ|m_mh02_all', 'WZ|m_mh05_all')}
            R['reference'][f'{kind}_{s}'] = o
    log('ref AUC', auc); save(); del LG
    # ---- head / block patching (blocks 5-8)
    pool = np.flatnonzero(ACC[('E1', 0.6)] & ACC[('E2', 0.6)] & ACC[('E3', 0.6)]); R['loc_pool'] = int(len(pool))
    n_ev = min(Q.N_EVAL, len(pool) // 4); n_tr = min(Q.N_DAS_TR, len(pool) - n_ev)
    pr = np.random.default_rng(SEED + 9).permutation(pool); didx = np.sort(pr[:n_tr + n_ev])
    loc = np.random.default_rng(SEED + 10).permutation(len(didx)); tr_i, ev_i = np.sort(loc[:n_tr]), np.sort(loc[n_tr:])
    cnt = (Jbase['mask'][didx] > 0).sum(1); EDS = ['E1', 'E2', 'E3', 'scrF', 'E1rot', 'E3rot', 'E3dil']
    ev_g = didx[ev_i]; order = ev_g[np.argsort(cnt[ev_i], kind='stable')]; nE = len(order)
    Bh = P.Boot(torch, dev, nE, NB, SEED + 23); R['heads'] = {}; hp = {ed: [] for ed in EDS}
    sets = [(bi, [h]) for bi in Q.HBLK for h in range(8)] + [(bi, list(range(8))) for bi in Q.HBLK]
    for mn in ('full', 'kin', 'init'):
        fk = P.FEAT_OF(mn); HP = Q.HeadPatch(torch, M[mn].model); model = M[mn].model
        Lb = np.zeros((nE, 10)); Le = {ed: np.zeros((nE, 10)) for ed in EDS}; Lp = {}
        T = lambda trip: tuple(torch.from_numpy(np.ascontiguousarray(a)).to(dev) for a in trip)
        for i in range(0, nE, 128):
            rows = np.arange(i, min(nE, i + 128)); idx = order[rows]
            with torch.no_grad():
                xb = T(P.slc(inputs(sub(Jbase, idx), fk), np.arange(len(idx)))); HP.mode = 'record'; Lb[rows] = model(*xb).double().cpu().numpy()
                for ed in EDS:
                    HP.mode = None; xe = T(P.slc(inputs(sub(JV[ed], idx), fk), np.arange(len(idx)))); Le[ed][rows] = model(*xe).double().cpu().numpy()
                    for (bi, hs) in sets:
                        HP.mode = 'patch'; HP.target = (bi, hs); Lp.setdefault((ed, bi, len(hs) == 8, hs[0]), np.zeros((nE, 10)))[rows] = model(*xe).double().cpu().numpy()
                HP.mode = None
        o = dict(decomposition_check=HP.chk)
        for ed in EDS:
            e0 = ((Le[ed] - Lb) ** 2).sum(1); oe = dict(mean_sq_logit_change=float(e0.mean()), heads={}, blocks={})
            for (ed2, bi, allh, h), L in Lp.items():
                if ed2 != ed: continue
                e = ((L - Lb) ** 2).sum(1); a, b = Bh.dot(e, e0); rb = 1 - a / np.maximum(b, 1e-30)
                cell = dict(restoration=float(1 - e.sum() / max(e0.sum(), 1e-30)), restoration_ci=ci(rb), p_one_sided=float(max(1 / NB, (rb <= 0).mean())), _rb=rb)
                if allh: oe['blocks'][f'b{bi + 1}'] = cell
                else:
                    oe['heads'][f'b{bi + 1}h{h + 1}'] = cell
                    if mn in ('full', 'kin'): hp[ed].append((mn, f'b{bi + 1}h{h + 1}', cell['p_one_sided']))
            hv = np.array([oe['heads'][f'b{bi + 1}h{h + 1}']['restoration'] for bi in Q.HBLK for h in range(8)]); oe['head_vector'] = hv.tolist()
            o[ed] = oe
        # similarity of the 32-head restoration vectors between edits and controls
        o['head_vector_corr'] = {f'{a}|{b}': float(np.corrcoef(o[a]['head_vector'], o[b]['head_vector'])[0, 1]) for i, a in enumerate(EDS) for b in EDS[i + 1:]}
        R['heads'][NAMES[mn]] = o
        for ed in EDS:
            log(f'  heads {NAMES[mn]} {ed}: blocks ' + ' '.join(f'{b}={v["restoration"]:.3f}' for b, v in o[ed]['blocks'].items()) + '; top ' +
                ', '.join(f'{h} {v["restoration"]:.3f}' for h, v in sorted(o[ed]['heads'].items(), key=lambda t: -t[1]['restoration'])[:3]))
        log('  head-vector corr', {k: round(v, 3) for k, v in o['head_vector_corr'].items()})
        for bi in Q.HBLK: del model.blocks[bi].attn.forward
    for ed, lst in hp.items():
        for (mn, h, p), ph in zip(lst, holm([t[2] for t in lst])): R['heads'][NAMES[mn]][ed]['heads'][h]['p_holm_64'] = ph
    save()
    # ---- DAS (ParT_full, init)
    DEDS = ['E1', 'E3', 'scrF', 'E1rot', 'E3rot']
    D = {'full': {'base': inputs(sub(Jbase, didx), 'full'), **{ed: inputs(sub(JV[ed], didx), 'full') for ed in DEDS}}}
    P.EDITS = DEDS; P.SITES = Q.SITES; P.VARS = {k: k for k in DEDS}
    nets = {mn: P.Net(M[mn]) for mn in ('full', 'init')}; boot_ev = P.Boot(torch, dev, len(ev_i), NB, SEED + 22); O = {}; R['das'] = {}
    for mn, net in nets.items():
        O[mn], _, _ = P.das_model(net, D['full'], cnt, tr_i, ev_i, np.random.default_rng(SEED + 32), None, NAMES[mn], boot_ev)
        R['das'][NAMES[mn]] = {ed: {s: dict(iia=O[mn][ed]['sites'][s]['das']['iia'], ci=O[mn][ed]['sites'][s]['das']['iia_ci'],
                                            lin_rand_max=O[mn][ed]['sites'][s]['lin']['rand_max']) for s in Q.SITES} for ed in DEDS}
        for ed in DEDS: log(f'  DAS {NAMES[mn]} {ed}: ' + ' '.join(f'{s}={O[mn][ed]["sites"][s]["das"]["iia"]:.2f}' for s in Q.SITES))
        save()
    R['das_did'] = {}
    for ed in DEDS:
        A, B = O['full'][ed]['sites'], O['init'][ed]['sites']
        dd = (A['b7']['das']['_bs'] - A['emb']['das']['_bs']) - (B['b7']['das']['_bs'] - B['emb']['das']['_bs'])
        R['das_did'][ed] = dict(did=float(np.mean(dd)), ci=ci(dd), p=P.p_two(dd))
    log('DAS DiD', R['das_did'])
    R['runtime_s'] = time.time() - P.T0; save(); log('wrote', FN)


if __name__ == '__main__':
    main()
