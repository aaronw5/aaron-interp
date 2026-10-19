"""H5 verification (loop D): rerun the key H5 edits on a fresh sample (data_qg.sample(5000, seed 9137)) and add the missing
controls.

Reused from interp.h5_qg_counting: jet representation, edits, models (current net_best_epoch_state.pt of the weaver QG trainings,
released LorentzNet, the H5 torch EFN/PFN, permuted-weight ParT kin), LR(N) table, statistics.

Edits rerun: split hardest z 0.5 (Delta R = 5e-4), merge-2-softest + split hardest (Delta N 0), remove softest, merge 2 softest,
split softest, k = 1..3 random splits.
New controls:
  split_r0_dr{0.005,0.02}  split hardest at larger daughter separation (is the response tied to the 5e-4 pair, i.e. off-manifold
                           pair features?)
  ptd_match                Delta N = 0, no new particle, no new pair: hardest particle pT scaled by f, every other particle scaled
                           up to keep sum pT; f solved per jet so that sum pT^2 equals that after the z = 0.5 split of the hardest
                           (same pT^D change as split_r0). Positions unchanged.
  add_soft, add_soft3      Delta N = +1 / +3: add particle(s) with pT = the jet's softest pT at the position of a random existing
                           particle offset by 0.05 in a random direction (no collinear pair; IR edit).
  add_med_scaled           Delta N = +1: add a particle with the jet's median pT (same placement), then scale all pT to keep sum pT.
Output $AI_RESULTS/physics/h5_verify.json and h5_verify_perjet.npz.
"""
import argparse
import json
import time
import numpy as np
from interp import h5_qg_counting as H

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


def ptd_match(L):
    out, ok = [], np.ones(len(L), bool)
    for j, a in enumerate(L):
        pt = a[:, 0]; n = len(a)
        if n < 3: out.append(a); ok[j] = False; continue
        P, p0, S2 = pt.sum(), pt[0], (pt ** 2).sum(); T = S2 - p0 ** 2 / 2; R = S2 - p0 ** 2; Q = P - p0
        A = p0 ** 2 * (1 + R / Q ** 2); B = -2 * P * p0 * R / Q ** 2; C = P ** 2 * R / Q ** 2 - T
        disc = B * B - 4 * A * C
        if disc < 0: out.append(a); ok[j] = False; continue
        f = (-B + np.sqrt(disc)) / (2 * A)
        if not (0 < f < 1): out.append(a); ok[j] = False; continue
        g = (P - f * p0) / Q; b = a.copy(); b[0, 0] *= f; b[1:, 0] *= g
        out.append(b)
    return out, ok


def add_particles(L, rng, k=1, mode='soft', scale=False):
    out = []
    for a in L:
        b = a.copy()
        for _ in range(k):
            i = rng.integers(len(a)); th = rng.uniform(0, 2 * np.pi)
            p = a[i].copy(); p[1] += 0.05 * np.cos(th); p[2] += 0.05 * np.sin(th)
            p[0] = a[-1, 0] if mode == 'soft' else np.median(a[:, 0])
            b = np.concatenate([b, p[None]])
        if scale: b[:, 0] *= a[:, 0].sum() / b[:, 0].sum()
        out.append(b)
    return out, np.ones(len(L), bool)


def main(a):
    import torch
    from interp import data_qg
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    S = data_qg.sample(a.per_class, a.seed); y = S['label'].astype(int); J = len(y)
    log(f'sample {J} jets seed {a.seed}, quark frac {y.mean():.3f}, device {dev}')
    L0 = H.to_list(H.from_test(S)); Ju = H.from_list(L0, Lpad=160); O0 = H.shapes(Ju)
    lrN, lrn_info = H.lr_n_table()
    # N distribution check (train, non-val): mean, sd vs Poisson sd
    d = H.QG / 'train'; yt = np.load(d / 'label.npy'); isv = np.load(d / 'is_val.npy')
    nt = np.asarray(np.load(d / 'jet.npy', mmap_mode='r')[:, 5])[~isv].astype(int); yt = yt[~isv]
    ndist = {c: dict(mean=float(nt[yt == v].mean()), sd=float(nt[yt == v].std()), poisson_sd=float(np.sqrt(nt[yt == v].mean())))
             for c, v in (('q', 1), ('g', 0))}
    Nu = O0['N'].astype(int)
    lrn_slope_direct = float(np.mean(lrN[np.minimum(Nu + 1, 199)] - lrN[np.minimum(Nu, 199)]))
    log('N dist', ndist, 'LR(N) slope (mean lrN[N+1]-lrN[N] on sample)', lrn_slope_direct)

    M = H.load_models(dev); names = list(M.m)
    base = {}; unedited = {}
    for nm in names:
        base[nm] = M.logit(nm, Ju); unedited[nm] = H.roc_metrics(base[nm], y); log(nm, unedited[nm])
    base['LR_N'] = lrN[np.minimum(Nu, 199)]; names.append('LR_N'); unedited['LR_N'] = H.roc_metrics(base['LR_N'], y)

    rng = np.random.default_rng(a.seed + 1)
    specs = []
    for dr in (5e-4, 5e-3, 2e-2):
        specs.append((f'split_r0_dr{dr:g}', ('split', dict(rank=0, z=0.5)), dr, 1))
    specs += [('merge_split', ('merge_split', {}), 5e-4, 0), ('rmsoft', ('rmsoft', {}), None, -1),
              ('merge_soft', ('merge', {}), None, -1), ('split_soft', ('split', dict(rank='soft', z=0.5)), 5e-4, 1)]
    for k in (1, 2, 3): specs.append((f'ksplit_{k}', ('ksplit', dict(k=k)), 5e-4, k))
    specs += [('ptd_match', 'ptd', None, 0), ('add_soft', ('add', dict(k=1, mode='soft')), None, 1),
              ('add_soft3', ('add', dict(k=3, mode='soft')), None, 3),
              ('add_med_scaled', ('add', dict(k=1, mode='med', scale=True)), None, 1)]
    full, okm, rows, ptests, Ls = {}, {}, {}, [], {}
    for en, sp, dr, dn in specs:
        if dr is not None: H.DR_SPLIT = dr
        if sp == 'ptd': Le, ok = ptd_match(L0)
        elif sp[0] == 'add': Le, ok = add_particles(L0, rng, **sp[1])
        else: Le, ok = H.edit_list(L0, sp[0], rng, **sp[1])
        H.DR_SPLIT = 5e-4
        Ls[en] = Le
        Je = H.from_list(Le, Lpad=175); Oe = H.shapes(Je); okm[en] = ok
        dsh = {k: float(np.mean((Oe[k] - O0[k])[ok])) for k in ('N', 'ptd', 'girth', 'C1_1.0', 'mass')}
        R = dict(dN=dn, applied=int(ok.sum()), dshape=dsh, models={})
        for nm in names:
            le = lrN[np.minimum(Oe['N'].astype(int), 199)] if nm == 'LR_N' else M.logit(nm, Je)
            dd = le - base[nm]; full[(en, nm)] = dd; dv = dd[ok]
            r = dict(mean=float(dv.mean()), ci=H.boot_mean(dv), p_t=H._tp(dv),
                     by_class=dict(q=float(dv[y[ok] == 1].mean()), g=float(dv[y[ok] == 0].mean())))
            w = H.wp_effects(base[nm][ok], le[ok], y[ok]); r['calls_changed_pct'] = w['calls_changed_pct']; r['mcnemar'] = w['mcnemar']
            R['models'][nm] = r; ptests.append((en, nm, r['p_t']))
        rows[en] = R; log(en, 'applied', R['applied'], dsh, '; '.join(f'{nm} {R["models"][nm]["mean"]:+.3f}' for nm in names))

    # paired comparisons on common jets
    comps = {}
    for e1, e2 in (('split_r0_dr0.0005', 'merge_split'), ('split_r0_dr0.0005', 'ptd_match'),
                   ('split_r0_dr0.0005', 'split_r0_dr0.02'), ('add_soft', 'rmsoft')):
        jm = okm[e1] & okm[e2]; comps[f'{e1} vs {e2}'] = {}
        for nm in names:
            a1, c1 = full[(e1, nm)][jm], full[(e2, nm)][jm]; dd = a1 - c1
            comps[f'{e1} vs {e2}'][nm] = dict(n=int(jm.sum()), e1=float(a1.mean()), e2=float(c1.mean()), diff=float(dd.mean()),
                                               ci=H.boot_mean(dd), p_t=H._tp(dd), p_perm=H.signflip_p(dd, P=2000),
                                               ratio_e2_e1=float(c1.mean() / a1.mean()) if abs(a1.mean()) > 1e-9 else None)
            ptests.append((f'{e1} - {e2}', nm, comps[f'{e1} vs {e2}'][nm]['p_t']))
    # k slope
    lin = {}
    for nm in names:
        mu = np.array([rows[f'ksplit_{k}']['models'][nm]['mean'] for k in (1, 2, 3)])
        D = np.stack([full[(f'ksplit_{k}', nm)][okm['ksplit_1']] for k in (1, 2, 3)], 1); rb = np.random.default_rng(3); bs = []
        for _ in range(500):
            ix = rb.integers(0, len(D), len(D)); bs.append(np.polyfit([1, 2, 3], D[ix].mean(0), 1)[0])
        lin[nm] = dict(means=mu.tolist(), slope=float(np.polyfit([1, 2, 3], mu, 1)[0]),
                       slope_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
    adj = H.holm([p[2] for p in ptests])
    holm_tab = [dict(test=p[0], model=p[1], p=p[2], holm=float(h)) for p, h in zip(ptests, adj)]
    # off-manifold for the new edits
    off = {}
    for en in ('split_r0_dr0.0005', 'split_r0_dr0.02', 'ptd_match', 'add_soft', 'add_med_scaled', 'rmsoft'):
        off[en] = H.c2st(L0, Ls[en]); log('c2st', en, off[en])
    off['none'] = H.c2st(L0, L0)
    out = dict(sample=dict(per_class=a.per_class, seed=a.seed, J=J), models=M.info, unedited=unedited, N_dist_train=ndist,
               lr_n=lrn_info, lr_n_slope_direct=lrn_slope_direct, poisson_pred=H.POISSON, edits=rows, comparisons=comps,
               linearity=lin, holm=holm_tab, n_tests=len(ptests), offmanifold=off, wall_s=time.time() - T0)
    H.RES.mkdir(parents=True, exist_ok=True)
    (H.RES / 'h5_verify.json').write_text(json.dumps(out, indent=1, default=float))
    np.savez_compressed(H.RES / 'h5_verify_perjet.npz', y=y, idx=S['idx'], **{f'{e}|{m}': v for (e, m), v in full.items()},
                        **{f'base|{m}': v for m, v in base.items()})
    log('wrote', H.RES / 'h5_verify.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--per_class', type=int, default=5000); ap.add_argument('--seed', type=int, default=9137)
    main(ap.parse_args())
