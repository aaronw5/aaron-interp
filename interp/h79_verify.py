"""H79 skeptic verification (loop D). Fresh sample (seed 7931, 2000 q + 2000 g per generator, N >= 8, random rows), edits from
interp/h79_radial_variable_qg.py (imported, unchanged).

Checks
  1. Rerun of the r_1 +-10 % arm (with crand_big) on the fresh sample: odd part, even part, per-sign excess.
  2. Conditional data reference. H79 compares the networks' r_1 response with a tree trained on r_1 ALONE (marginal trend). The
     edit holds the angularities, so the data trend that applies is r_1 at fixed angularities. References trained here on
     20k + 20k train jets (seed 7932): s_r1 (r_1 only), g7 ((pT^D)^2 + lambda(1, beta), beta = 0.1 ... 3), g7r1 (g7 + r_1),
     gfull (g7 + r_1, r_2, z_1, z_2, N). If g7r1 / gfull give a negative odd part, the networks' sign agrees with the data's
     conditional trend.
  3. Decomposition of the r_1 edit delta = (delta of the hardest particle, delta of the rest): hard-only (only the hardest particle's
     (log pT, r/R) change applied) and rest-only (all other particles' change applied, hardest particle unedited). No restore.
  4. r_2 arm: the second-hardest particle's radius +-10 % at fixed r_1, D, lambda(1,0.1), LHA*width^2, lambda(1,2), (pT^D)^2,
     sum pT, axis (and N), with its crand_big. Tests whether the quark-ward response to "one hard particle outward at fixed
     angularities" is specific to the hardest particle.
  5. On-manifold D +-0.1 % and lambda(1,0.1) +-0.03 % arms with crand_big (as H79 S steps).
Output: $AI_RESULTS/physics/h79_verify.json (+ h79_verify_<gen>.npz per-jet)
"""
import argparse
import json
import time

import numpy as np

from interp import h76_angularity_qg as H76
from interp import h79_radial_variable_qg as H79

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
RES = H79.RES; R0 = H79.R0; BETAS = H79.BETAS
NETS = ['part_kin', 'part_kin_s1', 'pnet_kin', 'lorentznet', 'pfn_k', 'efn_k', 'hw_part_kin', 'hw_pnet_kin', 'part_kin_perm']
RFEAT = dict(s_r1=['r1'], g7=['ptd'] + [f'b{b}' for b in BETAS], g7r1=['ptd'] + [f'b{b}' for b in BETAS] + ['r1'],
             gfull=['ptd'] + [f'b{b}' for b in BETAS] + ['r1', 'r2', 'z1', 'z2', 'N'])
REFS = list(RFEAT)


def ext2(a):
    o = H79.ext(a); pt = a[:, 0]; o2 = np.argsort(-pt, kind='stable'); j = int(o2[1])
    dR = np.hypot(a[:, 1] - o['jy'], (a[:, 2] - o['jphi'] + np.pi) % (2 * np.pi) - np.pi) / R0
    o.update(r2=float(dR[j]), z2=float(pt[j] / pt.sum()), i2=float(j)); return o


def ext_table(L):
    O = [ext2(np.asarray(a, np.float64)) for a in L]
    return {k: np.array([o[k] for o in O]) for k in O[0]}


class Solver2(H79.Solver):
    """C = (log r_2, log r_1, log D, log lambda(1,0.1), log LHA*width^2, log lambda(1,2), log (pT^D)^2, log sum pT, axis y, axis phi)"""
    def pack(self, L):
        S = super().pack(L); t = self.t; n = S['n']; pt = t.exp(S['x'][:, :n]) * S['m']
        S['i2'] = t.argsort(pt, 1, descending=True)[:, 1]; return S

    def C(self, S, x):
        t = self.t; c = super().C(S, x); pt, eta, phi = self.coords(S, x); yJ, pJ = self._axis(pt, eta, phi)
        dR = t.sqrt((eta - yJ[:, None]) ** 2 + self._wrap(phi - pJ[:, None]) ** 2 + 1e-14) / R0
        return t.cat([t.log(dR.gather(1, S['i2'][:, None])), c], 1)


def tol_r2(X0, X, fr=0.10):
    keep = ['r1', 'D', 'b0.1', 'comb', 'b0.5', 'b1.0', 'b2.0', 'ptd', 'spt', 'jpt']
    rel = lambda k: np.abs(X[k] / X0[k] - 1)
    dev = np.max(np.stack([rel(k) for k in keep] + [np.abs(X['jy'] - X0['jy']) / R0,
                                                      np.abs((X['jphi'] - X0['jphi'] + np.pi) % (2 * np.pi) - np.pi) / R0], 1), 1)
    ach = X['r2'] / X0['r2'] - 1
    ok = (dev < H79.TOL) & (X['N'] == X0['N']) & (X['i1'] == X0['i1']) & (X['i2'] == X0['i2']) & (np.abs(np.abs(ach) - fr) < 0.05 * fr)
    return ok, dev, ach


def tol_restore(X0, X):
    keep = ['r1', 'D', 'b0.1', 'comb', 'b0.5', 'b1.0', 'b2.0', 'ptd', 'spt', 'jpt']
    dev = np.max(np.stack([np.abs(X[k] / X0[k] - 1) for k in keep] + [np.abs(X['jy'] - X0['jy']) / R0], 1), 1)
    return (dev < H79.TOL) & (X['N'] == X0['N']), dev


def make_edits(L0, seed, gen, dev, chunk=1000):
    import torch
    Sv = H79.Solver(dev); Sv2 = Solver2(dev); out = {}
    def put(cn, Ls, r):
        out.setdefault(cn, ([], [])); out[cn][0].extend(Ls); out[cn][1].append((r < 1e-6).cpu().numpy())
    for s in range(0, len(L0), chunk):
        Lc = L0[s:s + chunk]; rng = np.random.default_rng([seed, 11, s])
        # r_1 +-10 % arm, its decomposition and crand_big
        S = Sv.pack(Lc); x0 = S['x']; c0, _ = Sv.jac(S, x0); n = S['n']
        xs = {}
        for sgn, st in (('+', np.log(1.1)), ('-', np.log(0.9))):
            tg = c0.clone(); tg[:, 0] += st; x, r = Sv.solve(S, x0, tg); xs[sgn] = x; put(f'r1{sgn}', Sv.unpack(S, x, Lc), r)
            d = x - x0; mh = torch.zeros_like(d, dtype=torch.bool)
            mh.scatter_(1, S['i1'][:, None], True); mh.scatter_(1, S['i1'][:, None] + n, True)
            put(f'r1hard{sgn}', Sv.unpack(S, x0 + torch.where(mh, d, torch.zeros_like(d)), Lc), r)
            put(f'r1rest{sgn}', Sv.unpack(S, x0 + torch.where(mh, torch.zeros_like(d), d), Lc), r)
        dp, dm = xs['+'] - x0, xs['-'] - x0; big = (dp.norm(dim=1) >= dm.norm(dim=1))[:, None]; d = torch.where(big, dp, dm)
        v = Sv.null_rand(S, x0, torch.sqrt(d[:, :n] ** 2 + d[:, n:] ** 2), d.norm(dim=1), rng)
        for sgn, sg in (('+', 1.0), ('-', -1.0)):
            x, r = Sv.solve(S, x0 + sg * v, c0); put(f'r1_cb{sgn}', Sv.unpack(S, x, Lc), r)
        # D +-0.1 %, lambda(1,0.1) +-0.03 % (on-manifold steps), crand_big
        for ti, t, fr in ((1, 'D', 0.001), (2, 'l01', 0.0003)):
            xs = {}
            for sgn, st in (('+', np.log(1 + fr)), ('-', np.log(1 - fr))):
                tg = c0.clone(); tg[:, ti] += st; x, r = Sv.solve(S, x0, tg); xs[sgn] = x; put(f'{t}{sgn}', Sv.unpack(S, x, Lc), r)
            dp, dm = xs['+'] - x0, xs['-'] - x0; big = (dp.norm(dim=1) >= dm.norm(dim=1))[:, None]; d = torch.where(big, dp, dm)
            v = Sv.null_rand(S, x0, torch.sqrt(d[:, :n] ** 2 + d[:, n:] ** 2), d.norm(dim=1), rng)
            for sgn, sg in (('+', 1.0), ('-', -1.0)):
                x, r = Sv.solve(S, x0 + sg * v, c0); put(f'{t}_cb{sgn}', Sv.unpack(S, x, Lc), r)
        # r_2 +-10 % arm at fixed r_1 and the rest, crand_big
        S2 = Sv2.pack(Lc); x0 = S2['x']; c0, _ = Sv2.jac(S2, x0); xs = {}
        for sgn, st in (('+', np.log(1.1)), ('-', np.log(0.9))):
            tg = c0.clone(); tg[:, 0] += st; x, r = Sv2.solve(S2, x0, tg); xs[sgn] = x; put(f'r2{sgn}', Sv2.unpack(S2, x, Lc), r)
        dp, dm = xs['+'] - x0, xs['-'] - x0; big = (dp.norm(dim=1) >= dm.norm(dim=1))[:, None]; d = torch.where(big, dp, dm)
        v = Sv2.null_rand(S2, x0, torch.sqrt(d[:, :n] ** 2 + d[:, n:] ** 2), d.norm(dim=1), rng)
        for sgn, sg in (('+', 1.0), ('-', -1.0)):
            x, r = Sv2.solve(S2, x0 + sg * v, c0); put(f'r2_cb{sgn}', Sv2.unpack(S2, x, Lc), r)
        log(f'{gen}: edits {s + len(Lc)}/{len(L0)}')
    return {k: (v[0], np.concatenate(v[1])) for k, v in out.items()}


def train_refs(gen, n_per_class, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from interp import h14_qg_generator as H14
    Lt, yt, _ = H14.efp_train_lists(gen, n_per_class=n_per_class, seed=seed)
    keep = np.array([len(a) >= 8 for a in Lt]); Lt = [a for a, k in zip(Lt, keep) if k]; yt = yt[keep]
    O = ext_table(Lt); g = (yt == 0).astype(int)
    clf = {k: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, early_stopping=True, validation_fraction=0.15,
                                             random_state=0).fit(np.c_[tuple(O[c] for c in f)], g) for k, f in RFEAT.items()}
    log(f'refs {gen}: {len(yt)} train jets'); return clf


def accept(cn, X0, X, okS):
    if cn.startswith('r1hard') or cn.startswith('r1rest'): return okS & (X['N'] == X0['N'])
    if cn.startswith('r2'):
        return okS & (tol_r2(X0, X)[0] if '_cb' not in cn else tol_restore(X0, X)[0])
    if cn.startswith('r1') and '_cb' not in cn: return okS & H79.tolerance(X0, X, f'r1{cn[2]}L')[0]
    if cn.startswith(('D', 'l01')) and '_cb' not in cn:
        t = 'D' if cn.startswith('D') else 'l01'; return okS & H79.tolerance(X0, X, f'{t}{cn[len(t)]}S')[0]
    return okS & tol_restore(X0, X)[0]


def step_run(a):
    import torch
    from interp import h5_qg_counting as H
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; RES.mkdir(parents=True, exist_ok=True)
    if dev != 'cuda' and not a.allow_cpu: raise SystemExit('no usable GPU on this node (CUDA init failed); not holding the GPU idle')
    M = H79.load_models(dev); names = [n for n in NETS if n in M.m]; log('models', names, dev)
    gens = a.gens.split(','); refs = {g: train_refs(g, a.train_per_class, a.seed + 1) for g in gens}
    summ = dict(seed=a.seed, per_class=a.per_class, gens={})
    for gen in gens:
        L0, y, idx, f8 = H76.load_sample(gen, a.per_class, a.seed); J = len(y); X0 = ext_table(L0)
        log(f'{gen}: {J} jets; dropped N<8 {f8:.4f}; first idx {idx[:5].tolist()}; median r1 {np.median(X0["r1"]):.4f} r2 {np.median(X0["r2"]):.4f}')
        E = make_edits(L0, a.seed, gen, dev); out = dict(y=y, idx=idx); G = summ['gens'][gen] = dict(n=J, base={}, ok={}, shift={}, c2st={})
        rngh = np.random.default_rng(a.seed + 5); perm = rngh.permutation(J); A, B = np.sort(perm[: J // 2]), np.sort(perm[J // 2:])
        FB = H79.c2st_feats(L0, X0, B)
        for cn, (Le, okS) in [('base', (L0, np.ones(J, bool)))] + list(E.items()):
            Jt = H.from_list(Le, Lpad=max(len(x) for x in Le))
            for nm in names: out[f'{cn}|{nm}'] = M.logit(nm, Jt).astype(np.float64)
            X = X0 if cn == 'base' else ext_table(Le)
            for k, m in refs[gen].items(): out[f'{cn}|{k}'] = m.decision_function(np.c_[tuple(X[c] for c in RFEAT[k])])
            if cn == 'base':
                for nm in names + REFS: G['base'][nm] = H.roc_metrics(out[f'base|{nm}'], y)
                log(gen, 'unedited', {k: round(v['AUC'], 4) for k, v in G['base'].items()}); continue
            ok = accept(cn, X0, X, okS); out[f'ok|{cn}'] = ok; G['ok'][cn] = float(ok.mean())
            G['shift'][cn] = {k: float(np.median(X[k][ok] / X0[k][ok] - 1)) for k in ('r1', 'r2', 'D', 'b0.1', 'b0.25', 'b0.5', 'b1.0',
                                                                                   'b1.5', 'b2.0', 'b3.0', 'ptd', 'z1', 'mass')} if ok.any() else None
            if cn in ('r1+', 'r1-', 'r2+', 'r2-', 'r1hard+', 'r1rest+'):
                G['c2st'][cn], _ = H79.c2st(H79.c2st_feats(Le, X, A), FB)
            log(gen, cn, 'ok', round(G['ok'][cn], 3), 'c2st', G['c2st'].get(cn, {}).get('auc'), '|',
                ' '.join(f'{nm} {np.mean((out[f"{cn}|{nm}"] - out[f"base|{nm}"])[ok]):+.4f}' for nm in names[:3] + REFS))
        np.savez_compressed(RES / f'h79_verify_{gen}.npz', **out); log('wrote npz', gen)
    (RES / 'h79_verify_run.json').write_text(json.dumps(summ, indent=1, default=str))


def step_stats(a):
    from interp.h76_angularity_qg import _holm
    run = json.loads((RES / 'h79_verify_run.json').read_text())
    S = dict(run=run, gens={}); tests = []
    for gen in a.gens.split(','):
        Z = dict(np.load(RES / f'h79_verify_{gen}.npz')); models = [m for m in NETS + REFS if f'base|{m}' in Z]
        G = S['gens'][gen] = {}; rng = np.random.default_rng(7931)
        def arm(conds, cbs, label):
            ok = np.all([Z[f'ok|{c}'] for c in conds + cbs], 0); jj = np.flatnonzero(ok); idx = rng.integers(0, len(jj), (a.boot, len(jj)))
            R = G[label] = dict(n_ok=int(len(jj)))
            for m in models:
                s0 = Z[f'base|{m}']; ds = {c: (Z[f'{c}|{m}'] - s0)[jj] for c in conds + cbs}
                cbo = 0.5 * (ds[cbs[0]] - ds[cbs[1]]) if cbs else 0.0; cbe = 0.5 * (ds[cbs[0]] + ds[cbs[1]]) if cbs else 0.0
                od = 0.5 * (ds[conds[0]] - ds[conds[1]]) - cbo; ev = 0.5 * (ds[conds[0]] + ds[conds[1]]) - cbe
                bo = np.array([od[i].mean() for i in idx]); be = np.array([ev[i].mean() for i in idx])
                r = R[m] = dict(dS_plus=float(ds[conds[0]].mean()), dS_minus=float(ds[conds[1]].mean()),
                                odd=float(od.mean()), odd_ci=[float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))],
                                even=float(ev.mean()), even_ci=[float(np.percentile(be, 2.5)), float(np.percentile(be, 97.5))])
                if cbs: r['cb_even'] = float(np.mean(cbe))
                if m in NETS or m in REFS:
                    p = H79._signflip(od, a.perm, seed=len(tests)); r['p_odd'] = p; tests.append((gen, label, m, p))
            return R, jj, idx
        R1, jj, idx = arm(['r1+', 'r1-'], ['r1_cb+', 'r1_cb-'], 'r1')
        arm(['r1hard+', 'r1hard-'], [], 'r1_hard_only'); arm(['r1rest+', 'r1rest-'], [], 'r1_rest_only')
        arm(['r2+', 'r2-'], ['r2_cb+', 'r2_cb-'], 'r2'); arm(['D+', 'D-'], ['D_cb+', 'D_cb-'], 'D_0.1pct')
        arm(['l01+', 'l01-'], ['l01_cb+', 'l01_cb-'], 'l01_0.03pct')
        # per-unit ratios in the r_1 arm: network odd / reference odd (bootstrap over jets)
        rat = G['r1_ratios'] = {}
        for ref in ('s_r1', 'g7r1', 'gfull'):
            s0r = Z[f'base|{ref}']; odr = 0.5 * ((Z[f'r1+|{ref}'] - s0r) - (Z[f'r1-|{ref}'] - s0r))[jj]
            for m in models:
                s0 = Z[f'base|{m}']; od = 0.5 * ((Z[f'r1+|{m}'] - s0) - (Z[f'r1-|{m}'] - s0))[jj] - 0.5 * ((Z[f'r1_cb+|{m}'] - s0) - (Z[f'r1_cb-|{m}'] - s0))[jj]
                b = np.array([od[i].mean() / odr[i].mean() for i in idx])
                rat[f'{m}/{ref}'] = dict(v=float(od.mean() / odr.mean()), ci=[float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))])
        log(gen, 'r1 odd', {m: round(R1[m]['odd'], 4) for m in models})
        for lab in ('r1_hard_only', 'r1_rest_only', 'r2', 'D_0.1pct', 'l01_0.03pct'):
            log(gen, lab, {m: round(G[lab][m]['odd'], 4) for m in models})
    adj = _holm([p for *_, p in tests]); S['n_tests'] = len(tests)
    S['holm'] = [dict(gen=g, arm=l, model=m, p=p, p_holm=float(q)) for (g, l, m, p), q in zip(tests, adj)]
    for (g, l, m, p), q in zip(tests, adj): S['gens'][g][l][m]['p_holm'] = float(q)
    (RES / 'h79_verify.json').write_text(json.dumps(S, indent=1)); log('wrote', RES / 'h79_verify.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', required=True, choices=['run', 'stats'])
    ap.add_argument('--gens', default='py,hw'); ap.add_argument('--per_class', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=7931); ap.add_argument('--train_per_class', type=int, default=20000)
    ap.add_argument('--allow_cpu', action='store_true')
    ap.add_argument('--boot', type=int, default=1000); ap.add_argument('--perm', type=int, default=10000)
    a = ap.parse_args(); RES.mkdir(parents=True, exist_ok=True)
    dict(run=step_run, stats=step_stats)[a.step](a)
