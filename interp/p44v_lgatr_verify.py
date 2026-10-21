"""P44v (loop C): verification of P44 (L-GATr grades / spurion routes / mass read-out) on a fresh sample.

Sample: JetClass val_5M <class>_120.root (HToGG: _124, because _120 is its L-GATr validation file; selection.json), 500 random
rows per class (np.random.default_rng(7777)), N 5,000, shuffled. Checkpoint model_run0_it154999.pt (the one P44 used).
Checks:
 (1) unedited accuracy; identical-input rerun floor.
 (2) mean / resample ablation of s, g0, g1, g2, g3, g4, inv at blocks 0, 2, 3, 5, 8, 11 (reproduction).
     Missing control for "grade 0 carries little vs s": a type-matched control = 16 random channels out of the 32 auxiliary
     scalars s (same count as g0, same channel type as s), mean and resample, 8 draws per block. Also the across-jet variance
     share of each group (particle tokens) per block, to see whether g0 is benign because it barely varies.
 (3) mass probe split: ridge of log m from g0 only (global 16 + particle-mean 16) vs s only (32 + 32) vs inv (96), trained and
     untrained network; calibrated by the angular dilation x1.1 (true dlog m ~ +0.094).
 (4) boost-route linearity at y = +0.5 and -0.5: A (physical), D (spurion route), F (all features from boosted jet),
     F_logE (only the two log E features, input scalar columns 11, 13), F_rest (other feature columns), C (exact symmetry).
     Linear "two routes that cancel" predicts dL_A ~ dL_D + dL_F with dL_D, dL_F anti-aligned (cos < 0) and W->Z moving in
     opposite directions; "response to the disagreement between the two energy inputs" predicts cos > 0, same-sign W->Z, and
     an even response in y.
  ./sync.sh && ./part_job.sh c-p44-verify '/j-jepa-vol/aaron-interp/env-lgatr/bin/python -m interp.p44v_lgatr_verify'
-> $AI_RESULTS/part/p44_verify.json
"""
import os, sys, json, time
from pathlib import Path
import numpy as np

SMOKE = os.environ.get('P44V_SMOKE') == '1'
sys.argv = [sys.argv[0], '--stage', 'none']          # p44 parses its own args at import
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import interp.p44_lgatr_grades as P                  # noqa: E402

SEED = 7777; PER = 30 if SMOKE else 500
VALF = {'ZJetsToNuNu': 123, 'HToBB': 121, 'HToCC': 123, 'HToGG': 120, 'HToWW4Q': 121, 'HToWW2Q1L': 124, 'TTBar': 123,
        'TTBarLep': 124, 'WToQQ': 121, 'ZToQQ': 121}
FILE = {c: (120 if VALF[P.PRE[c]] != 120 else 124) for c in P.CL}
BLK = [0, 2, 3, 5, 8, 11]; GN = ['s', 'g0', 'g1', 'g2', 'g3', 'g4', 'inv']; NDR = 3 if SMOKE else 8
CKPT = f'{P.RUN}/model_run0_it154999.pt'
OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'part' / f'p44_verify{"_smoke" if SMOKE else ""}.json'
T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def read_sample():
    import uproot
    from interp.part_core import read_root, sub, cat
    rng = np.random.default_rng(SEED); parts = []
    for c in P.CL:
        path = P.VAL / f'{P.PRE[c]}_{FILE[c]}.root'; N = uproot.open(path)['tree'].num_entries
        rows = np.sort(rng.choice(N, PER, replace=False)); ch = []
        for s in range(0, N, 25000):
            r = rows[(rows >= s) & (rows < s + 25000)]
            if len(r): ch.append(sub(read_root(path, s, min(s + 25000, N)), r - s))
        J = cat(ch); assert (J['label'] == P.CL.index(c)).all(); parts.append(J)
    J = cat(parts); return sub(J, rng.permutation(len(J['label'])))


def boot(x, nb=2000, seed=0):
    r = np.random.default_rng(seed); x = np.asarray(x, float); n = len(x)
    b = np.array([x[r.integers(0, n, n)].mean() for _ in range(nb)])
    return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def mcn(c0, c1):
    from scipy.stats import binomtest
    b, c = int((c0 & ~c1).sum()), int((~c0 & c1).sum()); return dict(n01=b, n10=c, p=float(binomtest(b, b + c).pvalue) if b + c else 1.0)


def logits_x(lg, Jmv, Jfeat=None, spur_y=0.0, cols=None):
    """like LG.logits, but only input-scalar columns `cols` are taken from Jfeat (all columns if cols is None)"""
    import torch
    N = len(Jmv['label']); out = np.zeros((N, 10), np.float32)
    with torch.no_grad():
        for s in range(0, N, P.BS):
            sl = np.arange(s, min(s + P.BS, N)); e = lg.embed(Jmv, sl, None, spur_y)
            if Jfeat is not None:
                f = lg._embed1(Jfeat, sl); e['s'] = e['s'].clone()
                if cols is None: e['s'] = f['s']
                else: e['s'][:, cols] = f['s'][:, cols]
            hm, hs, mask, _ = lg.upto(e, P.NBLK - 1); out[sl] = lg.rest(e, mask, P.NBLK - 1, hm, hs).float().cpu().numpy()
    return out


def grades(lg, Jb, y, L0, res):
    import torch
    M = P.sample_means(lg, Jb); N = len(y); rng = np.random.default_rng(SEED + 1); crng = np.random.default_rng(SEED + 2)
    idx = {g: torch.tensor(P.GROUPS[g], device=lg.dev) for g in GN}
    ctl = {(k, d): torch.tensor(np.sort(crng.choice(P.GROUPS['s'], 16, replace=False)), device=lg.dev) for k in BLK for d in range(NDR)}
    pred = {}; var = {k: np.zeros(len(GN)) for k in BLK}; vtot = {k: 0.0 for k in BLK}
    with torch.no_grad():
        for s in range(0, N, P.BS):
            sl = np.arange(s, min(s + P.BS, N)); e = lg.embed(Jb, sl); _, _, mask, outs = lg.upto(e, P.NBLK - 1, cache=True)
            src = P.donor_index(e, rng); pt = e['tt'] == 3
            for k in BLK:
                X = P.flat(*outs[k]); Rm = M[k][e['tt']]; Rr = X[src]
                V = X[pt].double().var(0).cpu().numpy(); vtot[k] += V.sum()
                for gi, g in enumerate(GN): var[k][gi] += V[P.GROUPS[g]].sum()
                jobs = [(f'{g}_{mn}', idx[g]) for g in GN for mn in ('mean', 'resample')]
                jobs += [(f'shalf{d}_{mn}', ctl[(k, d)]) for d in range(NDR) for mn in ('mean', 'resample')]
                for name, ix in jobs:
                    R = Rm if name.endswith('mean') else Rr
                    X2 = X.clone(); X2[:, ix] = R[:, ix]
                    pred.setdefault((k, name), np.zeros(N, int))[sl] = lg.rest(e, mask, k, *P.unflat(X2)).argmax(1).cpu().numpy()
            log(f'grades batch {s // P.BS + 1}')
    p0 = L0.argmax(1); c0 = p0 == y; out = {}
    for k in BLK:
        for mn in ('mean', 'resample'):
            for g in GN:
                c1 = pred[(k, f'{g}_{mn}')] == y
                out[f'b{k}_{g}_{mn}'] = dict(acc=boot(c1), calls_changed=float((pred[(k, f'{g}_{mn}')] != p0).mean()), mcnemar_vs_base=mcn(c0, c1))
            cs = np.mean([pred[(k, f'shalf{d}_{mn}')] == y for d in range(NDR)], 0)          # per-jet control accuracy
            cg = (pred[(k, f'g0_{mn}')] == y).astype(float)
            accs = [float((pred[(k, f'shalf{d}_{mn}')] == y).mean()) for d in range(NDR)]
            d = cg - cs; sg = np.random.default_rng(SEED + 3).choice([-1.0, 1.0], (5000, len(d)))
            pperm = float((1 + (np.abs(sg @ d) >= abs(d.sum())).sum()) / 5001)
            out[f'b{k}_g0_vs_shalf16_{mn}'] = dict(acc_g0=float(cg.mean()), acc_shalf_draws=accs, acc_shalf_mean=float(np.mean(accs)),
                                                  g0_minus_shalf=boot(d), p_signflip=pperm)
        out[f'b{k}_variance_share'] = {g: float(var[k][gi] / vtot[k]) for gi, g in enumerate(GN)}
        log(f'block {k}: ' + ' '.join(f"{g} {100 * out[f'b{k}_{g}_mean']['acc'][0]:.1f}/{100 * out[f'b{k}_{g}_resample']['acc'][0]:.1f}" for g in GN)
            + f" | s-half16 {100 * out[f'b{k}_g0_vs_shalf16_mean']['acc_shalf_mean']:.1f}/{100 * out[f'b{k}_g0_vs_shalf16_resample']['acc_shalf_mean']:.1f}"
            + ' | var share ' + ' '.join(f'{g}:{v:.3f}' for g, v in out[f'b{k}_variance_share'].items()))
    ps = [out[f'b{k}_g0_vs_shalf16_{mn}']['p_signflip'] for k in BLK for mn in ('mean', 'resample')]
    for (k, mn), ph in zip([(k, mn) for k in BLK for mn in ('mean', 'resample')], P.holm(ps)):
        out[f'b{k}_g0_vs_shalf16_{mn}']['p_holm'] = float(ph)
    res['grades'] = out


def probe_split(lg, Jb, res, name):
    import torch
    from interp.part_core import kin, build, jmass
    pt, de, dp, ms = kin(Jb); Jd = build(Jb, pt, de * 1.1, dp * 1.1, ms)
    sets = {'g0': P.GROUPS['g0'], 's': P.GROUPS['s'], 'inv': P.INV}
    def feats(J):
        N = len(J['label']); F = {n: np.zeros((P.NBLK, N, 2 * len(ix)), np.float32) for n, ix in sets.items()}
        tix = {n: torch.tensor(ix, device=lg.dev) for n, ix in sets.items()}
        with torch.no_grad():
            for s in range(0, N, P.BS):
                sl = np.arange(s, min(s + P.BS, N)); e = lg.embed(J, sl); _, _, _, outs = lg.upto(e, P.NBLK - 1, cache=True)
                g = e['tt'] == 0; p = e['tt'] == 3; bp = e['batch'][p]
                for k, (hm, hs) in enumerate(outs):
                    X = P.flat(hm, hs)
                    for n, ix in tix.items():
                        U = X[:, ix]; sm = torch.zeros(e['nb'], len(ix), device=lg.dev).index_add_(0, bp, U[p])
                        sm /= (e['cnt'] - 3).clamp(min=1)[:, None].float(); F[n][k, sl] = torch.cat([U[g], sm], 1).cpu().numpy()
        return F
    Fb, Fd = feats(Jb), feats(Jd); m0 = jmass(Jb); ok = np.where(m0 > 1.0)[0]
    lm = np.log(np.maximum(m0, 1.0)); dl = np.log(np.maximum(jmass(Jd), 1.0)) - lm
    h = len(ok) // 2; tr, te = ok[:h], ok[h:]; out = {}
    for n in sets:
        for k in range(P.NBLK):
            f, a = P.ridge_fit(Fb[n][k][tr], lm[tr], np.random.default_rng(SEED + 4))
            pb = f(Fb[n][k][te]); pd = f(Fd[n][k][te]); r2 = 1 - np.mean((pb - lm[te]) ** 2) / np.var(lm[te])
            gb = [];  r = np.random.default_rng(SEED + 5)
            for _ in range(1000):
                i = r.integers(0, len(te), len(te)); gb.append((pd - pb)[i].mean() / dl[te][i].mean())
            out[f'{n}_b{k}'] = dict(R2=float(r2), gain=[float((pd - pb).mean() / dl[te].mean()), float(np.percentile(gb, 2.5)), float(np.percentile(gb, 97.5))])
        log(f'{name} {n}: R2 ' + ' '.join(f"{out[f'{n}_b{k}']['R2']:.3f}" for k in range(P.NBLK))
            + ' | gain ' + ' '.join(f"{out[f'{n}_b{k}']['gain'][0]:.2f}" for k in range(P.NBLK)))
    out['N_train'], out['N_test'], out['dlogm_dil'] = int(h), int(len(te)), float(dl[te].mean())
    res[name] = out


def spurion(lg, Jb, y, L0, res):
    import torch
    from interp.p5_lgatr_launch import boost_z, rotate_lab
    out = {}; p0 = L0.argmax(1); W, Z = y == P.iW, y == P.iZ
    with torch.no_grad():                                 # which input-scalar columns change under the boost
        sl = np.arange(min(P.BS, len(y))); a = lg._embed1(Jb, sl)['s']; b = lg._embed1(boost_z(Jb, 0.5), sl)['s']
        dcol = (a - b).abs().amax(0).cpu().numpy()
    out['input_scalar_col_maxabs_change_y0.5'] = dcol.tolist(); log('scalar columns changed by boost 0.5:', np.round(dcol, 4))
    LOGE, REST = [11, 13], [10, 12, 14, 15, 16]
    def st(L):
        p = L.argmax(1); return dict(calls_changed=boot((p != p0).astype(float)), acc=float((p == y).mean()),
                                     W_to_Z=float((p[W] == P.iZ).mean()), Z_to_W=float((p[Z] == P.iW).mean()),
                                     mean_abs_dlogit=float(np.abs(L - L0).mean()))
    out['base'] = dict(W_to_Z=float((p0[W] == P.iZ).mean()), Z_to_W=float((p0[Z] == P.iW).mean()))
    out['rot_phi_0.06'] = st(logits_x(lg, rotate_lab(Jb, 0.06)))
    dL = {}
    for yv in (0.5, -0.5):
        Jy = boost_z(Jb, yv)
        Ls = dict(A=logits_x(lg, Jy), C=logits_x(lg, Jy, Jb, yv), D=logits_x(lg, Jy, Jb, 0.0), F=logits_x(lg, Jb, Jy, 0.0),
                  F_logE=logits_x(lg, Jb, Jy, 0.0, LOGE), F_rest=logits_x(lg, Jb, Jy, 0.0, REST))
        for r, L in Ls.items():
            out[f'y{yv}_{r}'] = st(L); dL[(yv, r)] = L - L0
            log(f"y {yv} {r}: calls {100 * out[f'y{yv}_{r}']['calls_changed'][0]:.2f}% W->Z {out[f'y{yv}_{r}']['W_to_Z']:.3f}")
        dD, dF, dA = dL[(yv, 'D')], dL[(yv, 'F')], dL[(yv, 'A')]
        # centre logits per jet (softmax is shift invariant)
        cen = lambda x: x - x.mean(1, keepdims=True)
        dD, dF, dA = cen(dD), cen(dF), cen(dA)
        nrm = lambda x: np.linalg.norm(x, axis=1)
        big = (nrm(dD) > 0.05) & (nrm(dF) > 0.05)
        cosj = (dD * dF).sum(1)[big] / (nrm(dD) * nrm(dF))[big]
        lin = nrm(dD + dF - dA) / np.maximum(nrm(dA), 1e-6)
        out[f'y{yv}_linearity'] = dict(cos_DF_per_jet=boot(cosj), n_jets=int(big.sum()),
                                       frac_cos_positive=float((cosj > 0).mean()),
                                       norm_ratio_sum_vs_A=float(np.linalg.norm(dD + dF) / np.linalg.norm(dA)),
                                       norm_ratio_resid=float(np.linalg.norm(dD + dF - dA) / np.linalg.norm(dD)),
                                       mean_norm=dict(D=float(nrm(dD).mean()), F=float(nrm(dF).mean()), A=float(nrm(dA).mean()),
                                                      D_plus_F=float(nrm(dD + dF).mean())))
        log(f'y {yv} linearity', out[f'y{yv}_linearity'])
    for r in ('D', 'F'):
        a, b = dL[(0.5, r)], dL[(-0.5, r)]; a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
        out[f'{r}_even_odd'] = dict(cos_plus_minus=float((a * b).sum() / np.linalg.norm(a) / np.linalg.norm(b)))
    log('even/odd', out['D_even_odd'], out['F_even_odd'])
    res['spurion'] = out


def main():
    import torch
    from interp.part_core import kin, build
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; log('ckpt', CKPT, dev, 'files', FILE)
    lg = P.LG(CKPT, dev); lg.ckpt = CKPT; log('loaded')
    J = read_sample(); y = J['label'].astype(int); Jb = build(J, *kin(J)); log('sample', len(y))
    res = dict(node='P44v', ckpt=CKPT, sample=f'val_5M files {FILE}, {PER}/class, default_rng({SEED}), shuffled', N=int(len(y)))
    L0 = lg.logits(Jb); L0b = lg.logits(Jb); p0 = L0.argmax(1)
    res['base'] = dict(acc=boot(p0 == y), rerun_calls_changed=float((L0b.argmax(1) != p0).mean()), rerun_mean_abs_dlogit=float(np.abs(L0b - L0).mean()),
                       acc_stored=float((lg.logits(J).argmax(1) == y).mean()),
                       per_class_recall={c: float((p0[y == i] == i).mean()) for i, c in enumerate(P.CL)})
    log('base', res['base']); OUT.parent.mkdir(parents=True, exist_ok=True)
    dump = lambda: OUT.write_text(json.dumps(res, indent=1, default=float))
    for fn in (lambda: spurion(lg, Jb, y, L0, res), lambda: grades(lg, Jb, y, L0, res), lambda: probe_split(lg, Jb, res, 'probe_trained')):
        try: fn()
        except Exception:
            import traceback; traceback.print_exc(); res.setdefault('failed', []).append(str(fn))
        dump()
    try:
        lr = P.LG(None, dev); probe_split(lr, Jb, res, 'probe_untrained')
    except Exception:
        import traceback; traceback.print_exc(); res.setdefault('failed', []).append('untrained')
    dump(); log('wrote', OUT)


if __name__ == '__main__':
    main()
