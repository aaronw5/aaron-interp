"""P21v -- verification of P21 (which prong observable ParT uses) on fresh rows.

Sample: JetClass test_20M WToQQ_104, ZToQQ_104, TTBar_104, ZJetsToNuNu_104 (files not used by P21); PER random rows per class
  (numpy default_rng(3121), never first-k); jets with >= 4 particles. Doses 0.6 and 1.25.
Arms (all rebuilt about their own E-scheme axis and mass-restored with p21.finalize, same as P21):
  E   = the P21 edit (primary component + compensation solved per jet);  rot = P21 matched rotation;  raw = primary alone;
  comp = the compensation alone (P21 parameters, primary component left at 1) -- separates what the compensation contributes;
  jit = matched isotropic jitter: every particle moved by exactly its own E displacement length |dx_i| in a uniformly random
        direction (seeded) -- same per-particle displacement as E, prong / radial structure removed (new control).
Behaviour (ParT_full, ParT_kin, PN_full, PN_kin): d_WQ, d_TQ change vs base, pooled and by true class; edit - rot, edit - jit,
  comp - rot, raw - rot; 95 % jet bootstrap CI (NB), sign-flip permutation p, Holm over the primary edit - jit tests.
  Odd / even parts over dose on jets accepted at both doses: odd = (D(0.6) - D(1.25)) / 2, even = (D(0.6) + D(1.25)) / 2.
References: HistGradientBoosting 2-fold cross-fitted on base jets: (m, tau21), (m, tau21, tau32, D2) and 'rich' =
  (m, tau21, tau32, D2, girth, soft-particle mean dR, 2-subjet masses / m, dR12, D2 over all particles); predicted change by class.
Location: ParT_full head / block patching in blocks 5-8 (P21 HeadPatch) on 1,000 random jets accepted for E1-E3 at 0.6, arms
  E1, E2, E3 and jit of E1, E3; NB_H resamples (Holm floor 32 / NB_H).
  ./sync.sh && ./part_job.sh c-p21-verify 'python -m interp.p21v_verify_part'
  -> $AI_RESULTS/part/p21_verify.json
"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
SMOKE = '--smoke' in sys.argv
_argv = sys.argv; sys.argv = [_argv[0]]
import interp.p21_prong_part as Q  # noqa: E402
sys.argv = _argv
P = Q.P

SEED = 3121
OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'
FN = OUT / ('p21_verify_smoke.json' if SMOKE else 'p21_verify.json')
PER = 120 if SMOKE else 3000
SV = (0.6, 1.25); Q.SV = SV
NB = 200 if SMOKE else 2000
NB_H = 200 if SMOKE else 2000
NPERM = 500 if SMOKE else 5000
N_H = 100 if SMOKE else 1000
CLS = Q.CLS; KINDS = Q.KINDS; HOLD, PRIM, PSCORE = Q.HOLD, Q.PRIM, Q.PSCORE
iQ, iZ, iW, iT = Q.iQ, Q.iZ, Q.iW, Q.iT
log, ci, holm, strip = P.log, P.ci, P.holm, P.strip
ARMS = ('E', 'rot', 'raw', 'comp', 'jit')


def extra_positions(torch, dev, pt_np, x_np, y_np, pos, info, seed):
    """comp (compensation only) and jit (matched isotropic jitter) for one chunk"""
    T = lambda a: torch.from_numpy(np.asarray(a, np.float64)).to(dev)
    pt, x0, y0 = T(pt_np), T(x_np), T(y_np); real = pt > 0; J = pt.shape[0]; one = torch.ones(J, dtype=torch.float64, device=dev)
    lab2 = Q.kmeans_t(torch, pt, x0, y0, 2)[1].argmin(-1).masked_fill(~real, -1)
    lab3 = Q.kmeans_t(torch, pt, x0, y0, 3)[1].argmin(-1).masked_fill(~real, -1)
    soft = (pt / pt.sum(1, keepdim=True) < Q.ZSOFT) & real
    g = np.random.default_rng(seed); out = {}; disp = {}

    def apply(s2, s3, u):
        x, y = Q.cl_scale(torch, pt, x0, y0, lab2, 2, s2); x, y = Q.cl_scale(torch, pt, x, y, lab3, 3, s3)
        return Q.soft_scale(torch, pt, x, y, soft, u)
    for kind in KINDS:
        for s in SV:
            p, q = T(info[(kind, s)]['p']), T(info[(kind, s)]['q'])
            par = (one, p, q) if kind == 'E1' else (p, one, q) if kind == 'E2' else (p, q, one)
            xc, yc = apply(*par); out[(kind, s, 'comp')] = (xc.cpu().numpy(), yc.cpu().numpy())
            xe, ye = pos[(kind, s, 'E')]; r = np.hypot(xe - x_np, ye - y_np) * (pt_np > 0); th = g.uniform(0, 2 * np.pi, r.shape)
            out[(kind, s, 'jit')] = (x_np + r * np.cos(th), y_np + r * np.sin(th))
            w = pt_np / pt_np.sum(1, keepdims=True)
            for v in ('E', 'rot', 'raw'):
                a, b = pos[(kind, s, v)]; disp[(kind, s, v)] = (w * np.hypot(a - x_np, b - y_np)).sum(1)
            for v in ('comp', 'jit'):
                a, b = out[(kind, s, v)]; disp[(kind, s, v)] = (w * np.hypot(a - x_np, b - y_np)).sum(1)
    return out, disp


def rich_obs(torch, dev, J, chunk=250):
    """girth, soft-particle mean dR, 2-subjet masses / m, dR12 (k-means 2 axes), D2 over all particles (beta 1)"""
    from interp.part_core import kin, jmass
    pt, de, dp, _ = kin(J); real = pt > 0; z = pt / pt.sum(1, keepdims=True); r = np.hypot(de, dp)
    soft = (z < Q.ZSOFT) & real; m = jmass(J).astype(np.float64)
    o = dict(girth=(z * r).sum(1), soft_r=np.where(soft.any(1), (z * r * soft).sum(1) / np.maximum((z * soft).sum(1), 1e-12), 0.0))
    m1, m2, d12, d2a = [], [], [], []
    for i in range(0, len(pt), chunk):
        Tt = lambda a: torch.from_numpy(a[i:i + chunk].astype(np.float64)).to(dev)
        tp, tx, ty = Tt(pt), Tt(de), Tt(dp); ax, dist = Q.kmeans_t(torch, tp, tx, ty, 2); lab = dist.argmin(-1).cpu().numpy()
        sl = slice(i, i + chunk); mk = real[sl]; ms = []
        for c in range(2):
            sel = (lab == c) & mk; E = (J['energy'][sl] * sel).sum(1); px, py, pz = ((J[k][sl] * sel).sum(1) for k in ('px', 'py', 'pz'))
            ms.append(np.sqrt(np.maximum(E.astype(np.float64) ** 2 - px ** 2 - py ** 2 - pz ** 2, 0)))
        ms = np.sort(np.stack(ms, 1), 1)[:, ::-1]; m1.append(ms[:, 0]); m2.append(ms[:, 1])
        d12.append((ax[:, 0] - ax[:, 1]).norm(dim=-1).cpu().numpy())
        zt = tp / tp.sum(1, keepdim=True)
        R = ((tx[:, :, None] - tx[:, None]) ** 2 + (ty[:, :, None] - ty[:, None]) ** 2).sqrt()
        W = zt[:, :, None] * zt[:, None, :] * R; e2 = W.sum((1, 2)) / 2
        e3 = (W * torch.einsum('jik,jk,jkl->jil', R, zt, R)).sum((1, 2)) / 6
        d2a.append((e3 / (e2 ** 3).clamp_min(1e-12)).cpu().numpy())
    mm = np.maximum(m, 1e-6)
    o.update(m1=np.concatenate(m1) / mm, m2=np.concatenate(m2) / mm, dR12=np.concatenate(d12), D2all=np.concatenate(d2a))
    return o


def fit_refs(OB, y, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier as HGB
    from sklearn.metrics import roc_auc_score
    L = lambda a: np.log(np.clip(a, 1e-4, 1e4))
    feats = dict(m_t21=lambda o: np.stack([np.log(o['m'] + 1), o['tau21']], 1),
                 m_all=lambda o: np.stack([np.log(o['m'] + 1), o['tau21'], o['tau32'], L(o['D2'])], 1),
                 rich=lambda o: np.stack([np.log(o['m'] + 1), o['tau21'], o['tau32'], L(o['D2']), L(o['girth']), L(o['soft_r'] + 1e-3),
                                          o['m1'], o['m2'], o['dR12'], L(o['D2all'])], 1))
    tasks = dict(WZ=(np.isin(y, [iW, iZ, iQ]), np.isin(y, [iW, iZ])), TQ=(np.isin(y, [iT, iQ]), y == iT))
    fold = np.random.default_rng(seed).integers(0, 2, len(y)); M = {}; auc = {}
    for tn, (sel, lab) in tasks.items():
        for fn, f in feats.items():
            X = f(OB); ms = [HGB(max_iter=300, learning_rate=0.1, random_state=seed).fit(X[sel & (fold == k)], lab[sel & (fold == k)]) for k in (0, 1)]
            M[(tn, fn)] = (f, ms); pr = np.where(fold == 0, ms[1].decision_function(X), ms[0].decision_function(X))
            auc[f'{tn}|{fn}'] = float(roc_auc_score(lab[sel], pr[sel]))

    def pred(o):
        return {f'{tn}|{fn}': np.where(fold == 0, ms[1].decision_function(f(o)), ms[0].decision_function(f(o))) for (tn, fn), (f, ms) in M.items()}
    return pred, auc


def main():
    import torch
    from interp.part_core import CL, PRE, cat, kin, sub, jmass, sample, inputs, build
    from interp import pn_net
    from jetdistill.part.network import FEATURES
    OUT.mkdir(parents=True, exist_ok=True)
    R = dict(node='P21v', script='interp/p21v_verify_part.py', smoke=SMOKE, seed=SEED, files='_104.root', per_class=PER, doses=SV, NB=NB,
             NB_heads=NB_H, NPERM=NPERM)
    save = lambda: FN.write_text(json.dumps(strip(R), indent=1))
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    Js = [P.load_rows(P.TEST / f'{PRE[c]}_104.root', PER, rng) for c in CLS]; J0 = cat(Js); del Js
    keep = np.flatnonzero((J0['mask'] > 0).sum(1) >= 4); R['n_drawn'] = int(len(J0['label'])); J0 = sub(J0, keep)
    N = len(J0['label']); y = J0['label'].astype(int); R['N'] = N; log(f'N = {N}')
    pt, de, dp, ms = kin(J0); Jb = build(J0, pt, de, dp, ms); pt, de, dp, ms = kin(Jb); m0 = jmass(Jb).astype(np.float64)
    Jbase = Q.finalize(Jb, pt, de.astype(np.float64), dp.astype(np.float64), ms, m0)
    M = P.make_models(['full', 'kin']); dev = M['full'].device
    PNDIR = Path(os.environ.get('JETDISTILL_MODELS', '/j-jepa-vol/aaron-interp/models/part')).parent / 'particlenet'
    PN = {n: pn_net.load_particlenet(PNDIR / f'ParticleNet_{n}.pt', len(FEATURES[n]), dev) for n in ('full', 'kin')}
    NAMES = dict(full='ParT_full', kin='ParT_kin', pnfull='PN_full', pnkin='PN_kin')
    logits = lambda mn, J: Q.pn_logits(torch, PN[mn[2:]], J, mn[2:]) if mn.startswith('pn') else M[mn].run(J)['logits']
    J1 = sample(per_class=1000, start=3000); p1, e1, f1, s1 = kin(J1); J1b = build(J1, p1, e1, f1, s1)
    R['accuracy_P1_rows'] = {NAMES[mn]: float((logits(mn, J1b).argmax(1) == J1['label']).mean()) for mn in NAMES}; del J1, J1b
    log('accuracy P1 rows:', R['accuracy_P1_rows'])
    # ---- positions
    POS, INFO, DISP = {}, {}, {}
    for i in range(0, N, Q.GCHUNK):
        sl = slice(i, min(N, i + Q.GCHUNK))
        pos, info, _ = Q.edit_positions(torch, dev, pt[sl], de[sl], dp[sl], SEED + i)
        ext, disp = extra_positions(torch, dev, pt[sl], de[sl], dp[sl], pos, info, SEED + 77 + i); pos.update(ext)
        for k, v in pos.items():
            if k[2] in ARMS: POS.setdefault(k, []).append((v[0].astype(np.float32), v[1].astype(np.float32)))
        for k, v in info.items():
            for a, b in v.items(): INFO.setdefault(k, {}).setdefault(a, []).append(b)
        for k, v in disp.items(): DISP.setdefault(k, []).append(v)
        log(f'  positions: {sl.stop} / {N}')
    POS = {k: (np.concatenate([a[0] for a in v]).astype(np.float64), np.concatenate([a[1] for a in v]).astype(np.float64)) for k, v in POS.items()}
    INFO = {k: {a: np.concatenate(b) for a, b in v.items()} for k, v in INFO.items()}; DISP = {k: np.concatenate(v) for k, v in DISP.items()}
    # ---- variants
    OB = {'base': {**Q.obs_np(torch, dev, Jbase), **rich_obs(torch, dev, Jbase)}}; LG = {mn: {'base': logits(mn, Jbase)} for mn in NAMES}
    R['accuracy_sample'] = {NAMES[mn]: float((LG[mn]['base'].argmax(1) == y).mean()) for mn in NAMES}
    JH = {}
    for kind in KINDS:
        for s in SV:
            for v in ARMS:
                nm = f'{kind}_{s}_{v}'; Je = Q.finalize(Jb, pt, *POS[(kind, s, v)], ms, m0); assert np.array_equal(Je['mask'], Jbase['mask'])
                OB[nm] = {**Q.obs_np(torch, dev, Je), **rich_obs(torch, dev, Je)}
                for mn in NAMES: LG[mn][nm] = logits(mn, Je)
                if s == 0.6 and (v == 'E' or (v == 'jit' and kind in ('E1', 'E3'))): JH[f'{kind}_{v}'] = Je
                log(f'  {nm}: d_WQ ParT_full {np.mean((LG["full"][nm][:, iW] - LG["full"][nm][:, iQ]) - (LG["full"]["base"][:, iW] - LG["full"]["base"][:, iQ])):+.3f}')
    del POS
    rel = lambda nm, k: OB[nm][k] / np.maximum(OB['base'][k], 1e-12) - 1
    ACC = {}; R['edits'] = {}
    for kind in KINDS:
        for s in SV:
            nm = f'{kind}_{s}_E'
            a = (np.abs(rel(nm, HOLD[kind][0])) <= Q.TOL) & (np.abs(rel(nm, HOLD[kind][1])) <= Q.TOL) & (np.abs(rel(nm, 'm')) <= Q.MTOL)
            ACC[(kind, s)] = a
            e = dict(accepted=int(a.sum()), accepted_by_class={c: float(a[y == CL.index(c)].mean()) for c in CLS},
                     comp_params_median=[float(np.median(INFO[(kind, s)]['p'][a])), float(np.median(INFO[(kind, s)]['q'][a]))])
            for v in ARMS:
                n2 = f'{kind}_{s}_{v}'
                e[v] = {k: float(np.mean(rel(n2, k)[a])) for k in ('tau21', 'tau32', 'D2', 'm', 'girth', 'soft_r', 'm1', 'm2', 'dR12', 'D2all')}
                e[v]['mean_disp'] = float(np.mean(DISP[(kind, s, v)][a]))
                e[v]['disp_ratio_to_E_median'] = float(np.median(DISP[(kind, s, v)][a] / np.maximum(DISP[(kind, s, 'E')][a], 1e-12)))
            R['edits'][f'{kind}_{s}'] = e
            log(f'  {kind} {s}: acc {a.sum()}; ' + '; '.join(f'{v}: ' + ' '.join(f'{k} {e[v][k]:+.3f}' for k in ('tau21', 'tau32', 'D2', 'soft_r', 'm1', 'dR12')) + f' disp {e[v]["mean_disp"]:.4f}' for v in ARMS))
    save()
    # ---- behaviour
    boot = P.Boot(torch, dev, N, NB, SEED + 21); sc = lambda L: dict(dWQ=L[:, iW] - L[:, iQ], dTQ=L[:, iT] - L[:, iQ])
    groups = {'all': np.ones(N, bool), **{c: y == CL.index(c) for c in CLS}}
    R['behaviour'] = {}; tests = []
    for mn in NAMES:
        B0 = sc(LG[mn]['base']); o = {}
        for kind in KINDS:
            for s in SV:
                acc = ACC[(kind, s)]; S = {v: sc(LG[mn][f'{kind}_{s}_{v}']) for v in ARMS}; cell = {}
                for k in ('dWQ', 'dTQ'):
                    ck = {}
                    for gname, gs in groups.items():
                        sel = acc & gs; gk = {}
                        for v in ARMS:
                            est, cc, _ = Q.bmean(boot, S[v][k] - B0[k], sel); gk[f'{v}_vs_base'] = [est, cc]
                        for a_, b_ in (('E', 'rot'), ('E', 'jit'), ('comp', 'rot'), ('raw', 'rot'), ('jit', 'rot'), ('E', 'comp')):
                            d = S[a_][k] - S[b_][k]; est, cc, _ = Q.bmean(boot, d, sel); gk[f'{a_}_minus_{b_}'] = [est, cc]
                            if gname == 'all' and k == PSCORE[kind] and (a_, b_) in (('E', 'rot'), ('E', 'jit')):
                                p = Q.perm_p(torch, dev, d[sel], n=NPERM, seed=SEED + 7); gk[f'{a_}_minus_{b_}_p'] = p
                                if (a_, b_) == ('E', 'jit'): tests.append((mn, kind, s, k, p))
                        ck[gname] = gk
                    cell[k] = ck
                o[f'{kind}_{s}'] = cell
        # odd / even over dose (jets accepted at both doses)
        oe = {}
        for kind in KINDS:
            both = ACC[(kind, 0.6)] & ACC[(kind, 1.25)]; oe[kind] = {'n': int(both.sum())}
            for k in ('dWQ', 'dTQ'):
                for ref in ('base', 'rot', 'jit'):
                    def D(s):
                        x = sc(LG[mn][f'{kind}_{s}_E'])[k]; return x - (B0[k] if ref == 'base' else sc(LG[mn][f'{kind}_{s}_{ref}'])[k])
                    odd, even = (D(0.6) - D(1.25)) / 2, (D(0.6) + D(1.25)) / 2
                    for gname, gs in groups.items():
                        sel = both & gs; a1 = Q.bmean(boot, odd, sel); a2 = Q.bmean(boot, even, sel)
                        oe[kind].setdefault(k, {}).setdefault(ref, {})[gname] = dict(odd=[a1[0], a1[1]], even=[a2[0], a2[1]])
        o['odd_even'] = oe
        R['behaviour'][NAMES[mn]] = o
        log(f'  behaviour {NAMES[mn]} done')
    for (mn, kind, s, k, p), h in zip(tests, holm([t[4] for t in tests])):
        R['behaviour'][NAMES[mn]][f'{kind}_{s}'][k]['all']['E_minus_jit_p_holm'] = h
    save()
    # ---- references
    pred, auc = fit_refs(OB['base'], y, SEED + 3); R['ref_auc'] = auc; P0 = pred(OB['base']); R['reference'] = {}
    for kind in KINDS:
        for s in SV:
            acc = ACC[(kind, s)]; PV = {v: pred(OB[f'{kind}_{s}_{v}']) for v in ARMS}; o = {}
            for nm in P0:
                o[nm] = {}
                for gname, gs in groups.items():
                    sel = acc & gs
                    o[nm][gname] = {f'{v}_vs_base': float(np.mean((PV[v][nm] - P0[nm])[sel])) for v in ARMS}
                    o[nm][gname]['E_minus_rot'] = float(np.mean((PV['E'][nm] - PV['rot'][nm])[sel]))
                    o[nm][gname]['E_minus_jit'] = float(np.mean((PV['E'][nm] - PV['jit'][nm])[sel]))
            R['reference'][f'{kind}_{s}'] = o
    log('ref AUC:', auc); save()
    # ---- head / block patching, ParT_full
    pool = np.flatnonzero(ACC[('E1', 0.6)] & ACC[('E2', 0.6)] & ACC[('E3', 0.6)]); R['head_pool'] = int(len(pool))
    hidx = np.sort(np.random.default_rng(SEED + 9).choice(pool, min(N_H, len(pool)), replace=False))
    cnt = (Jbase['mask'][hidx] > 0).sum(1); order = hidx[np.argsort(cnt, kind='stable')]; nE = len(order)
    Bh = P.Boot(torch, dev, nE, NB_H, SEED + 23); R['heads'] = {}
    for mn in ('full', 'kin'):
        model = M[mn].model; HP = Q.HeadPatch(torch, model); fk = P.FEAT_OF(mn)
        sets = [(bi, [h]) for bi in Q.HBLK for h in range(8)] + [(bi, list(range(8))) for bi in Q.HBLK]
        Lb = np.zeros((nE, 10)); Le = {ed: np.zeros((nE, 10)) for ed in JH}; Lp = {}
        for i in range(0, nE, 128):
            rows = np.arange(i, min(nE, i + 128)); idx = order[rows]
            T = lambda trip: tuple(torch.from_numpy(np.ascontiguousarray(a)).to(dev) for a in trip)
            with torch.no_grad():
                HP.mode = 'record'; Lb[rows] = model(*T(P.slc(inputs(sub(Jbase, idx), fk), np.arange(len(idx))))).double().cpu().numpy()
                for ed, Je in JH.items():
                    xe = T(P.slc(inputs(sub(Je, idx), fk), np.arange(len(idx)))); HP.mode = None; Le[ed][rows] = model(*xe).double().cpu().numpy()
                    for (bi, hs) in sets:
                        HP.mode = 'patch'; HP.target = (bi, hs)
                        Lp.setdefault((ed, bi, len(hs) == 8, hs[0]), np.zeros((nE, 10)))[rows] = model(*xe).double().cpu().numpy()
                HP.mode = None
        o = dict(decomposition_check=HP.chk)
        for ed in JH:
            e0 = ((Le[ed] - Lb) ** 2).sum(1); oe = dict(mean_sq_logit_change=float(e0.mean()), blocks={}, heads={})
            for (ed2, bi, allh, h), L in Lp.items():
                if ed2 != ed: continue
                e = ((L - Lb) ** 2).sum(1); a, b = Bh.dot(e, e0); rb = 1 - a / np.maximum(b, 1e-30)
                cellh = dict(restoration=float(1 - e.sum() / max(e0.sum(), 1e-30)), ci=ci(rb), p=float(max(1 / NB_H, (rb <= 0).mean())))
                if allh: oe['blocks'][f'b{bi + 1}'] = cellh
                else: oe['heads'][f'b{bi + 1}h{h + 1}'] = cellh
            hs_ = list(oe['heads']); ph = holm([oe['heads'][h]['p'] for h in hs_])
            for h, p in zip(hs_, ph): oe['heads'][h]['p_holm32'] = p
            oe['top_heads'] = sorted(((h, v['restoration']) for h, v in oe['heads'].items()), key=lambda t: -t[1])[:5]
            o[ed] = oe
            log(f'  heads {NAMES[mn]} {ed}: blocks ' + ' '.join(f'{b}={v["restoration"]:.3f}' for b, v in oe['blocks'].items()) + f'; top {oe["top_heads"][:3]}')
        R['heads'][NAMES[mn]] = o
        for bi in Q.HBLK: del model.blocks[bi].attn.forward
    R['runtime_s'] = time.time() - P.T0; save(); log('wrote', FN)


if __name__ == '__main__':
    main()
