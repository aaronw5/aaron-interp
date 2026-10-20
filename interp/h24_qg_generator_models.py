"""H24 (loop D; claim D7): generator dependence of quark-gluon taggers. Do Herwig-trained and Pythia-trained networks
respond differently to the particle count, at fixed IRC-safe content and fixed pT^D?

Samples (seed 9641, random rows, 4000 quark + 4000 gluon jets per generator)
  py: interp.data_qg test split (Komiske-Metodiev-Thaler Pythia 8.226)
  hw: Zenodo 3066475 Herwig 7.1 files 18-19 (= the Herwig test split of the retrains; interp.h14_qg_generator.load_sample)
Models (logit l = log p(gluon) - log p(quark); positive = toward gluon)
  Pythia-trained: part_kin, pnet_kin (H2 weaver, final), pfn_k (H2 energyflow PFN, d-h14-pfn2 rerun, ported to torch),
                  efn_k (H2 energyflow EFN, IRC-safe: exact-split response 0), pfn_t / efn_t (H5 torch EF on 400k jets: second
                  training run of the same architectures), part_kin_perm (permuted-weight null)
  Herwig-trained: hw_part_kin (d-h14-hw-part3, final), hw_pnet_kin (d-h14-hw-pnet3; final or interim, recorded)
  References per generator (CPU step): LR(N) negative binomial (lrn_<gen>), LR(pT^D), LR(girth) (histogram likelihood ratios,
                  lrp_<gen>, lrg_<gen>), GBDT(EFP d<=5 + N) and GBDT(EFP) (efpn_<gen>, efp_<gen>), trained on each generator's
                  train split (30k jets per class). Every input is rebuilt from the edited particle list (E-scheme axis of the
                  edited jet: interp.h5_qg_counting.kinematics).
Edits (deterministic from (generator, SEED); jets with N >= 10 for the split family)
  add_soft    Delta N = +1: one particle with the jet's softest pT at a random existing particle's position + 0.05 in a random
              direction (donor radius; H5v / H14 definition, C2ST checked here)
  add_swap    Delta N = 0 control: the same added particle, the original softest particle removed
  sp<k>       k = 1..5 distinct random particles of the softer half split exactly (z = 0.5, Delta R = 0): count +k, every IRC-safe
              observable (girth, mass, EFPs, EFN latent) unchanged; pT^D lowered (sum pT^2 - sum_i pT_i^2 / 2). Nested in k.
  ct<k>       Delta N = 0 control for sp<k>: for the same k particles, momentum moved into particle i from its nearest-in-angle
              particle h with pT_h - pT_i >= pT_i so that sum pT^2 drops by exactly pT_i^2 / 2: the same pT^D as sp<k>, N fixed.
  swap        observable swap: N moved by the Pythia - Herwig mean training multiplicity gap of the jet's class
              (gluon +4.48, quark +0.04; stochastic rounding per jet) at fixed pT^D:
                hw jets (toward Pythia): n exact collinear splits of soft particles, then sum pT^2 restored by one momentum
                  transfer between two existing particles chosen to minimise the girth change
                py jets (toward Herwig): n merges of the softest particle into its nearest-in-angle neighbour (pT summed,
                  pT-weighted position), then sum pT^2 restored the same way
              residual changes of girth and mass are recorded
Steps
  --step eval   (GPU, part_job.sh) -> physics/h24_logits_<gen>.npz (+ per-jet N, pT^D, girth, mass per condition), h24_models.json
  --step base   (CPU, venv-ef) references + C2ST -> physics/h24_base_<gen>.npz, h24_base.json
  --step stats  (CPU) -> physics/h24_run.json
  --step fig    (CPU) -> physics/h24_generator_models.png
"""
import argparse
import json
import os
import time
import zlib
from pathlib import Path

import numpy as np

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
V = Path('/j-jepa-vol/aaron-interp')
RES = Path(os.environ.get('AI_RESULTS', V / 'results')) / 'physics'
SEED = 9641
KS = (1, 2, 3, 4, 5)
CONDS = ['base', 'add_soft', 'add_swap'] + [f'sp{k}' for k in KS] + [f'ct{k}' for k in KS] + ['swap']
GEN_GAP_FALLBACK = dict(g=53.1333 - 48.6504, q=33.4340 - 33.3919)   # H14 train-split means (Pythia - Herwig)


def _rng(gen, name):
    return np.random.default_rng([SEED, zlib.crc32(gen.encode()), zlib.crc32(name.encode())])


def gen_gap():
    f = RES / 'h14_baselines.json'
    if f.exists():
        L = json.loads(f.read_text())['lrn']
        return dict(g=L['py']['mean_g'] - L['hw']['mean_g'], q=L['py']['mean_q'] - L['hw']['mean_q'])
    return GEN_GAP_FALLBACK


# ----------------------------------------------------------------------------------------------------------------- edits
def _axis_R(a):
    """distance of each particle from the E-scheme (massless) axis of list a (pT, eta, phi, pid)"""
    pt, eta, phi = a[:, 0], a[:, 1], a[:, 2]
    px, py, pz, E = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), pt * np.cosh(eta)
    jx, jy, jz = px.sum(), py.sum(), pz.sum(); jeta = np.arcsinh(jz / np.hypot(jx, jy)); jphi = np.arctan2(jy, jx)
    return np.hypot(eta - jeta, (phi - jphi + np.pi) % (2 * np.pi) - np.pi)


def _dang(a, i):
    return np.hypot(a[:, 1] - a[i, 1], (a[:, 2] - a[i, 2] + np.pi) % (2 * np.pi) - np.pi)


def split_set(a, S):
    """exact collinear split (z = 0.5, Delta R = 0) of the particles with indices S"""
    b = a.copy(); b[S, 0] *= 0.5
    return np.concatenate([b, b[S]])


def ctrl_set(a, S):
    """Delta N = 0 control: sum pT^2 lowered by sum_{i in S} pT_i^2 / 2 by moving momentum into each i from its nearest
    particle h with pT_h - pT_i >= pT_i(orig); positions, N and sum pT fixed. None if no donor exists."""
    pt = a[:, 0].copy()
    for i in S:
        r = a[i, 0] ** 2 / 2; g = pt - pt[i]
        cand = np.flatnonzero((g >= np.sqrt(2 * r)) & (np.arange(len(pt)) != i))
        if len(cand) == 0: return None
        h = cand[np.argmin(_dang(a, i)[cand])]; gh = pt[h] - pt[i]
        d = (gh - np.sqrt(gh * gh - 2 * r)) / 2; pt[h] -= d; pt[i] += d
    b = a.copy(); b[:, 0] = pt
    return b


def restore_s2(b, target):
    """one momentum transfer between two particles of b so that sum pT^2 = target (sum pT, positions, N fixed), the pair
    chosen to minimise |delta * (R_h - R_j)| (the girth change at fixed sum pT), donor keeps >= 10 % of its pT"""
    pt = b[:, 0]; r = target - (pt ** 2).sum()
    if abs(r) < 1e-12: return b
    R = _axis_R(b); n = len(pt)
    J, H = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')         # transfer delta from J to H (H harder)
    g = pt[H] - pt[J]; ok = g > 0
    if r > 0:     # raise sum pT^2: move from softer J to harder H: 2 d g + 2 d^2 = r
        d = (-g + np.sqrt(g * g + 2 * r)) / 2; ok &= d <= 0.9 * pt[J]
        src, dst = J, H
    else:         # lower: move from harder H to softer J: -2 d g + 2 d^2 = r
        disc = g * g + 2 * r; ok &= disc >= 0; d = (g - np.sqrt(np.maximum(disc, 0))) / 2; ok &= d <= 0.9 * pt[H]
        src, dst = H, J
    if not ok.any(): return None
    cost = np.where(ok, np.abs(d * (R[H] - R[J])), np.inf); k = np.unravel_index(np.argmin(cost), cost.shape)
    c = b.copy(); c[src[k], 0] -= d[k]; c[dst[k], 0] += d[k]
    return c


def merge_soft(a, n):
    """n times: the softest particle merged into its nearest-in-angle neighbour (pT summed, pT-weighted position, pid kept)"""
    b = a.copy()
    for _ in range(n):
        if len(b) < 3: return None
        s = len(b) - 1; dd = _dang(b, s); dd[s] = np.inf; t = int(np.argmin(dd))
        w = b[s, 0] + b[t, 0]; dphi = (b[s, 2] - b[t, 2] + np.pi) % (2 * np.pi) - np.pi
        b[t, 1] = (b[t, 0] * b[t, 1] + b[s, 0] * b[s, 1]) / w; b[t, 2] = b[t, 2] + b[s, 0] * dphi / w; b[t, 0] = w
        b = b[:-1]; b = b[np.argsort(-b[:, 0], kind='stable')]
    return b


def make_edits(L0, y, gen):
    """-> dict cond -> (edited list, applied mask, dN per jet)"""
    J = len(L0); out = {}
    rng = _rng(gen, 'add_soft'); A, B = [], []; okB = np.ones(J, bool)
    for j, a in enumerate(L0):
        i = rng.integers(len(a)); th = rng.uniform(0, 2 * np.pi)
        p = a[i].copy(); p[1] += 0.05 * np.cos(th); p[2] += 0.05 * np.sin(th); p[0] = a[-1, 0]
        A.append(np.concatenate([a, p[None]]))
        if len(a) >= 2: B.append(np.concatenate([a[:-1], p[None]]))
        else: B.append(a); okB[j] = False
    out['add_soft'] = (A, np.ones(J, bool), np.ones(J)); out['add_swap'] = (B, okB, np.zeros(J))
    rng = _rng(gen, 'split'); okS = np.array([len(a) >= 10 for a in L0]); order = []
    for a in L0:
        n = len(a); h = n // 2; order.append(rng.permutation(np.arange(n - h, n)))
    okC = okS.copy(); SP = {k: [] for k in KS}; CT = {k: [] for k in KS}
    for j, a in enumerate(L0):
        for k in KS:
            if not okS[j]: SP[k].append(a); CT[k].append(a); continue
            S = order[j][:k]; SP[k].append(split_set(a, S)); c = ctrl_set(a, S)
            if c is None: okC[j] = False; c = a
            CT[k].append(c)
    for k in KS:
        out[f'sp{k}'] = (SP[k], okS & okC, np.full(J, float(k))); out[f'ct{k}'] = (CT[k], okS & okC, np.zeros(J))
    # observable swap: N moved by the class's Pythia - Herwig gap at fixed pT^D
    gap = gen_gap(); rng = _rng(gen, 'swap'); W, okW, dN = [], np.ones(J, bool), np.zeros(J)
    for j, a in enumerate(L0):
        g = gap['g'] if y[j] == 0 else gap['q']; n = int(np.floor(abs(g)) + (rng.random() < abs(g) - np.floor(abs(g))))
        s2 = (a[:, 0] ** 2).sum()
        if n == 0: W.append(a); continue
        if gen == 'hw':      # toward Pythia: add n by exact splits of soft particles
            if len(a) < 10: W.append(a); okW[j] = False; continue
            h = len(a) // 2; S = rng.permutation(np.arange(len(a) - h, len(a)))[:n]; b = split_set(a, S); dN[j] = n
        else:                # toward Herwig: remove n by merging the softest into its nearest neighbour
            b = merge_soft(a, n); dN[j] = -n
            if b is None: W.append(a); okW[j] = False; dN[j] = 0; continue
        c = restore_s2(b, s2)
        if c is None: W.append(a); okW[j] = False; dN[j] = 0; continue
        W.append(c)
    out['swap'] = (W, okW, dN)
    for k, (Le, ok, d) in out.items():
        out[k] = ([b[np.argsort(-b[:, 0], kind='stable')] for b in Le], ok, d)
    return out


def basic_obs(L):
    """N, pT^D, girth (E-scheme axis), jet mass (massless constituents) per jet"""
    N = np.array([len(a) for a in L], float); ptd = np.zeros(len(L)); gir = np.zeros(len(L)); m = np.zeros(len(L))
    for j, a in enumerate(L):
        pt, eta, phi = a[:, 0], a[:, 1], a[:, 2]; P = pt.sum(); z = pt / P
        ptd[j] = np.sqrt((z ** 2).sum()); gir[j] = (z * _axis_R(a)).sum()
        px, py, pz, E = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), pt * np.cosh(eta)
        m[j] = np.sqrt(max(E.sum() ** 2 - px.sum() ** 2 - py.sum() ** 2 - pz.sum() ** 2, 0))
    return dict(N=N, ptd=ptd, girth=gir, mass=m)


def load(gen, per_class):
    from interp import h14_qg_generator as H14
    L0, y, info = H14.load_sample(gen, per_class, SEED)
    return L0, y, info


# ------------------------------------------------------------------------------------------------------------------- eval
def step_eval(a):
    import torch
    from interp import h5_qg_counting as H
    from interp import h14_qg_generator as H14
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; RES.mkdir(parents=True, exist_ok=True)
    M = H.Models(dev)
    for name, arch in (('part_kin', 'part'), ('pnet_kin', 'pn')): M.add_weaver(name, arch, False)
    M.add_permuted('part_kin_perm', 'part_kin')
    for kind in ('efn', 'pfn'):
        H14.add_keras_ef(M, kind)
        net = H.make_ef(kind); net.load_state_dict(torch.load(V / 'models' / 'qg' / f'h5_{kind}_torch.pt', map_location='cpu'))
        M.add_torch(f'{kind}_t', net, json.loads((V / 'models' / 'qg' / f'h5_{kind}_torch.json').read_text()))
    for name, arch in (('part_kin', 'part'), ('pnet_kin', 'pn')):
        d = V / 'models' / 'qg_herwig' / name
        if (d / 'net_best_epoch_state.pt').exists(): H14.add_weaver_dir(M, 'hw_' + name, arch, False, d)
        else: log(f'hw_{name}: no checkpoint')
    names = list(M.m); log('models', names)
    for gen in a.gens.split(','):
        L0, y, info = load(gen, a.per_class)
        log(f'{gen}: {len(y)} jets, quark frac {y.mean():.3f}, <N> q {np.mean([len(x) for x, c in zip(L0, y) if c == 1]):.2f} '
            f'g {np.mean([len(x) for x, c in zip(L0, y) if c == 0]):.2f}; device {dev}')
        E = make_edits(L0, y, gen); log(gen, 'edits applied:', {k: int(v[1].sum()) for k, v in E.items()})
        out = dict(y=y, idx=info['idx'])
        conds = [('base', L0, np.ones(len(L0), bool), np.zeros(len(L0)))] + [(k, *v) for k, v in E.items()]
        for cn, Le, ok, dN in conds:
            Jt = H.from_list(Le, Lpad=180)
            assert Jt['n_full'].max() <= 180
            for nm in names: out[f'{cn}|{nm}'] = M.logit(nm, Jt).astype(np.float32)
            O = basic_obs(Le); out.update({f'obs|{cn}|{k}': v.astype(np.float32) for k, v in O.items()})
            if cn != 'base': out[f'ok|{cn}'] = ok; out[f'dN|{cn}'] = dN.astype(np.float32)
            if cn == 'base':
                log(gen, 'unedited', '; '.join(f'{nm} {H.roc_metrics(out[f"base|{nm}"].astype(float), y)}' for nm in names))
            else:
                log(gen, cn, f'applied {int(ok.sum())}, mean dN {dN[ok].mean():+.3f}, dptd {np.mean((O["ptd"] - out["obs|base|ptd"])[ok]):+.5f}, '
                    f'dgirth {np.mean((O["girth"] - out["obs|base|girth"])[ok]):+.6f}, dmass {np.mean((O["mass"] - out["obs|base|mass"])[ok]):+.4f} | '
                    + ' '.join(f'{nm} {np.mean((out[f"{cn}|{nm}"] - out[f"base|{nm}"])[ok]):+.4f}' for nm in names))
        tag = '_smoke' if a.per_class < 4000 else ''
        np.savez_compressed(RES / f'h24_logits_{gen}{tag}.npz', **out); log('wrote', RES / f'h24_logits_{gen}{tag}.npz')
    (RES / f'h24_models{"_smoke" if a.per_class < 4000 else ""}.json').write_text(json.dumps(M.info, indent=1, default=str))


# ---------------------------------------------------------------------------------------------------------- references
def _lr_hist(x, y, bins):
    hq = np.histogram(x[y == 1], bins)[0] + 0.5; hg = np.histogram(x[y == 0], bins)[0] + 0.5
    return np.log(hg / hg.sum()) - np.log(hq / hq.sum())


def step_base(a):
    from multiprocessing import Pool
    from sklearn.ensemble import HistGradientBoostingClassifier
    from interp import h5_qg_counting as H
    from interp import h14_qg_generator as H14
    RES.mkdir(parents=True, exist_ok=True); smoke = a.per_class < 4000
    T, lrinfo = H14.lrn_tables(); pool = Pool(a.cpu)
    feat = lambda F, N, wn: np.c_[np.log(np.maximum(F, 1e-12)), N] if wn else np.log(np.maximum(F, 1e-12))
    clf, hist = {}, {}
    for gen in ('py', 'hw'):
        Lt, yt, Nt = H14.efp_train_lists(gen, n_per_class=2000 if smoke else 30000, seed=SEED + 1)
        Lt4 = [np.c_[x, np.zeros(len(x))] for x in Lt]; Ot = basic_obs(Lt4)
        bp = np.quantile(Ot['ptd'], np.linspace(0, 1, 41)); bp[0], bp[-1] = 0, 1.01
        bg = np.quantile(Ot['girth'], np.linspace(0, 1, 41)); bg[0], bg[-1] = 0, 10
        hist[gen] = dict(ptd=(bp, _lr_hist(Ot['ptd'], yt, bp)), girth=(bg, _lr_hist(Ot['girth'], yt, bg)))
        Ft = H14.efps(Lt, pool); g = (yt == 0).astype(int)
        for wn in (True, False):
            clf[('efpn' if wn else 'efp') + '_' + gen] = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.1, early_stopping=True, validation_fraction=0.15, random_state=0).fit(feat(Ft, Nt, wn), g)
        log(f'{gen}: references trained on {len(yt)} jets ({Ft.shape[1]} EFPs)')
    look = lambda gen, k, x: hist[gen][k][1][np.clip(np.searchsorted(hist[gen][k][0], x, side='right') - 1, 0, 39)]
    c2 = {}
    for gen in ('py', 'hw'):
        L0, y, info = load(gen, a.per_class); E = make_edits(L0, y, gen); out = dict(y=y)
        conds = [('base', L0, np.ones(len(L0), bool))] + [(k, v[0], v[1]) for k, v in E.items()]
        for cn, Lx, ok in conds:
            F = H14.efps([x[:, :3] for x in Lx], pool); O = basic_obs(Lx); N = O['N']
            for nm, m in clf.items(): out[f'{cn}|{nm}'] = m.decision_function(feat(F, N, nm.startswith('efpn'))).astype(np.float32)
            for tg in ('py', 'hw'):
                out[f'{cn}|lrn_{tg}'] = T[tg][np.minimum(N.astype(int), 299)].astype(np.float32)
                out[f'{cn}|lrp_{tg}'] = look(tg, 'ptd', O['ptd']).astype(np.float32)
                out[f'{cn}|lrg_{tg}'] = look(tg, 'girth', O['girth']).astype(np.float32)
            if cn != 'base':
                out[f'ok|{cn}'] = ok
                log(gen, cn, '; '.join(f'{k.split("|")[1]} {np.mean((out[k] - out["base|" + k.split("|")[1]])[ok]):+.4f}'
                                       for k in out if k.startswith(cn + '|')))
        np.savez_compressed(RES / f'h24_base_{gen}{"_smoke" if smoke else ""}.npz', **out)
        rs = np.random.default_rng(5); sub = np.sort(rs.choice(len(L0), min(8000, len(L0)), replace=False))
        Ls = [L0[i] for i in sub]; c2[gen] = {'none': H.c2st(Ls, Ls)}
        for en in ('add_soft', 'add_swap', 'sp1', 'sp5', 'ct1', 'ct5', 'swap'):
            c2[gen][en] = H.c2st(Ls, [E[en][0][i] for i in sub]); log('c2st', gen, en, c2[gen][en])
    (RES / f'h24_base{"_smoke" if smoke else ""}.json').write_text(json.dumps(dict(lrn=lrinfo, gap=gen_gap(), c2st=c2), indent=1))
    pool.close(); log('base step done')


# -------------------------------------------------------------------------------------------------------------- statistics
TRAINED = dict(part_kin='py', pnet_kin='py', pfn_k='py', efn_k='py', pfn_t='py', efn_t='py', part_kin_perm='py',
               hw_part_kin='hw', hw_pnet_kin='hw')
PAIRS = (('hw_part_kin', 'part_kin'), ('hw_pnet_kin', 'pnet_kin'))


def _load(gen, tag):
    Z = np.load(RES / f'h24_logits_{gen}{tag}.npz'); y = Z['y'].astype(int); S, ok, dN, obs = {}, {}, {}, {}
    for k in Z.files:
        p = k.split('|')
        if p[0] == 'ok': ok[p[1]] = Z[k].astype(bool)
        elif p[0] == 'dN': dN[p[1]] = Z[k].astype(float)
        elif p[0] == 'obs': obs[(p[1], p[2])] = Z[k].astype(float)
        elif len(p) == 2: S[(p[0], p[1])] = Z[k].astype(float)
    fb = RES / f'h24_base_{gen}{tag}.npz'
    if fb.exists():
        Zb = np.load(fb); assert (Zb['y'] == y).all()
        for k in Zb.files:
            p = k.split('|')
            if len(p) == 2 and p[0] != 'ok': S[(p[0], p[1])] = Zb[k].astype(float)
            elif p[0] == 'ok': assert (ok[p[1]] == Zb[k]).all(), k
    return y, S, ok, dN, obs


def _slope_rows(D):
    """per-row least-squares slope of D (J x 5) on k = 1..5"""
    k = np.array(KS, float); kc = k - k.mean()
    return D @ kc / (kc ** 2).sum()


def _boot_cls():
    from interp.h14_qg_generator import Boot as B0

    class Boot(B0):
        """H14 Boot with a finer sign-flip test: P = 20000 sign vectors generated in chunks from a fixed seed (minimum
        p = 1/20001, so Holm over ~120 tests can reach 0.05; with H14's P = 2000 the Holm floor was 0.061)"""
        def signflip(self, d, ok, P=20000, chunk=1000):
            dd = np.where(ok, d, 0.0).astype(np.float64); obs = abs(dd.sum()); r = np.random.default_rng(SEED + 7); cnt = 0
            for _ in range(P // chunk):
                sg = r.choice(np.array([-1.0, 1.0]), (chunk, self.J)); cnt += int((np.abs(sg @ dd) >= obs - 1e-9).sum())
            return (cnt + 1) / (P + 1)
    return Boot


def step_stats(a):
    from interp import h5_qg_counting as H
    from interp.h14_qg_generator import _tp
    Boot = _boot_cls()
    tag = '_smoke' if a.per_class < 4000 else ''
    BJ = json.loads((RES / f'h24_base{tag}.json').read_text()) if (RES / f'h24_base{tag}.json').exists() else {}
    R = dict(sample=dict(per_class=a.per_class, seed=SEED), models_info=json.loads((RES / f'h24_models{tag}.json').read_text()),
             lrn=BJ.get('lrn'), gap=BJ.get('gap'), c2st=BJ.get('c2st'), gens={}, cross={})
    tests = []; G = {}; per_jet = {}
    for gen in ('py', 'hw'):
        y, S, ok, dN, obs = _load(gen, tag); J = len(y); Bt = Boot(J, seed=SEED)
        models = sorted({m for (c, m) in S if c == 'base'}, key=lambda m: (m.startswith('l'), m.startswith('efp'), m))
        g = dict(J=J, unedited={}, obs={}, add_soft={}, ksplit={}, swap={}); per_jet[gen] = {}
        for m in models:
            g['unedited'][m] = dict(**H.roc_metrics(S[('base', m)], y), AUC_ci=H.boot_auc(S[('base', m)], y),
                                    mean_logit_q=float(S[('base', m)][y == 1].mean()), mean_logit_g=float(S[('base', m)][y == 0].mean()))
        for c in CONDS[1:]:
            o = ok[c]; g['obs'][c] = dict(applied=int(o.sum()), dN=float(dN[c][o].mean()),
                                         **{f'd{k}': float((obs[(c, k)] - obs[('base', k)])[o].mean()) for k in ('ptd', 'girth', 'mass')},
                                         **{f'abs_d{k}': float(np.abs(obs[(c, k)] - obs[('base', k)])[o].mean()) for k in ('girth', 'mass')})
        g['obs']['base'] = {k: dict(q=float(obs[('base', k)][y == 1].mean()), g=float(obs[('base', k)][y == 0].mean()))
                            for k in ('N', 'ptd', 'girth', 'mass')}
        for m in models:
            d0 = lambda c: S[(c, m)] - S[('base', m)]
            # add soft: edit, Delta N = 0 control, difference
            o = ok['add_swap']; e, c = d0('add_soft'), d0('add_swap'); df = e - c
            r = dict(n=int(o.sum()), edit=float(e[o].mean()), edit_ci=Bt.mean_ci(e, o)[0], control=float(c[o].mean()),
                     control_ci=Bt.mean_ci(c, o)[0], diff=float(df[o].mean()), diff_ci=Bt.mean_ci(df, o)[0], p_perm=Bt.signflip(df, o), p_t=_tp(df[o]))
            g['add_soft'][m] = r; tests.append((gen, 'add_soft - add_swap', m, r['p_perm']))
            # k-split slope, control slope, difference (per-jet slopes, same jets)
            o = ok['sp1']; Dsp = np.stack([d0(f'sp{k}') for k in KS], 1); Dct = np.stack([d0(f'ct{k}') for k in KS], 1)
            ssp, sct = _slope_rows(Dsp), _slope_rows(Dct); sdf = ssp - sct
            per_jet[gen][m] = dict(slope_diff=sdf, o_split=o, add=df, o_add=ok['add_swap'])
            r = dict(n=int(o.sum()), means_split=Dsp[o].mean(0).tolist(), means_ctrl=Dct[o].mean(0).tolist(),
                     slope_split=float(ssp[o].mean()), slope_split_ci=Bt.mean_ci(ssp, o)[0],
                     slope_ctrl=float(sct[o].mean()), slope_ctrl_ci=Bt.mean_ci(sct, o)[0],
                     slope_diff=float(sdf[o].mean()), slope_diff_ci=Bt.mean_ci(sdf, o)[0], p_perm=Bt.signflip(sdf, o), p_t=_tp(sdf[o]),
                     by_class=dict(q=float(sdf[o & (y == 1)].mean()), g=float(sdf[o & (y == 0)].mean())))
            g['ksplit'][m] = r; tests.append((gen, 'k-split slope - control slope', m, r['p_perm']))
            # observable swap (gluon jets carry the gap; quark jets mostly unedited)
            o = ok['swap'] & (dN['swap'] != 0); e = d0('swap'); pp = np.where(o, e / np.where(dN['swap'] == 0, 1, dN['swap']), 0)
            og = o & (y == 0)
            r = dict(n=int(o.sum()), n_gluon=int(og.sum()), dN_mean_gluon=float(dN['swap'][og].mean()) if og.any() else None,
                     edit_gluon=float(e[og].mean()), edit_gluon_ci=Bt.mean_ci(e, og)[0],
                     per_particle=float(pp[o].mean()), per_particle_ci=Bt.mean_ci(pp, o)[0], p_perm=Bt.signflip(e, o), p_t=_tp(e[o]))
            g['swap'][m] = r; tests.append((gen, 'swap', m, r['p_perm']))
            log(gen, m, f"AUC {g['unedited'][m]['AUC']:.4f} acc {g['unedited'][m]['acc']:.4f} | add-swap {g['add_soft'][m]['diff']:+.4f} "
                f"| slope sp {g['ksplit'][m]['slope_split']:+.4f} ct {g['ksplit'][m]['slope_ctrl']:+.4f} diff {g['ksplit'][m]['slope_diff']:+.4f} "
                f"| swap/particle {g['swap'][m]['per_particle']:+.4f}")
        # Herwig-trained minus Pythia-trained, paired per jet (sign-flip permutation)
        g['hw_minus_py'] = {}
        for mh, mp in PAIRS:
            if mh not in per_jet[gen] or mp not in per_jet[gen]: continue
            out = {}
            for key, ok_key, nm in (('slope_diff', 'o_split', 'k-split slope - control'), ('add', 'o_add', 'add_soft - add_swap')):
                d = per_jet[gen][mh][key] - per_jet[gen][mp][key]; o = per_jet[gen][mh][ok_key]
                out[nm] = dict(hw_trained=float(per_jet[gen][mh][key][o].mean()), py_trained=float(per_jet[gen][mp][key][o].mean()),
                               diff=float(d[o].mean()), ci=Bt.mean_ci(d, o)[0], p_perm=Bt.signflip(d, o))
                tests.append((gen, f'hw-trained - py-trained: {nm}', f'{mh} vs {mp}', out[nm]['p_perm']))
            g['hw_minus_py'][f'{mh} - {mp}'] = out
        # training-run spread reference: two trainings of the same PFN architecture (H2 keras vs H5 torch)
        g['replicate_spread'] = {}
        for m1, m2 in (('pfn_k', 'pfn_t'), ('efn_k', 'efn_t'), ('part_kin', 'pnet_kin')):
            if m1 in per_jet[gen] and m2 in per_jet[gen]:
                g['replicate_spread'][f'{m1} - {m2}'] = {nm: float(per_jet[gen][m1][k][per_jet[gen][m1][o]].mean() - per_jet[gen][m2][k][per_jet[gen][m1][o]].mean())
                                                        for k, o, nm in (('slope_diff', 'o_split', 'k-split slope - control'), ('add', 'o_add', 'add_soft - add_swap'))}
        G[gen] = g
    # generator logit gap on gluon (and quark) jets vs swap response (different jets: independent bootstrap)
    yp, Sp, *_ = _load('py', tag); yh, Sh, *_ = _load('hw', tag); rng = np.random.default_rng(SEED)
    for m in sorted({m for (c, m) in Sp if c == 'base'}):
        if ('base', m) not in Sh: continue
        r = {}
        for cl, cv in (('gluon', 0), ('quark', 1)):
            a1, a2 = Sp[('base', m)][yp == cv], Sh[('base', m)][yh == cv]
            bs = [a1[rng.integers(0, len(a1), len(a1))].mean() - a2[rng.integers(0, len(a2), len(a2))].mean() for _ in range(1000)]
            r[f'gap_{cl}'] = dict(py_minus_hw=float(a1.mean() - a2.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
        sw_h = G['hw']['swap'].get(m, {}).get('edit_gluon'); sw_p = G['py']['swap'].get(m, {}).get('edit_gluon')
        r['swap_hw_toward_py_gluon'] = sw_h; r['swap_py_toward_hw_gluon'] = sw_p
        gg = r['gap_gluon']['py_minus_hw']
        r['fraction_of_gluon_gap_hw_swap'] = float(sw_h / gg) if sw_h is not None and abs(gg) > 1e-6 else None
        r['fraction_of_gluon_gap_py_swap'] = float(-sw_p / gg) if sw_p is not None and abs(gg) > 1e-6 else None
        R['cross'][m] = r
    R['gens'] = G
    adj = H.holm([t[3] for t in tests])
    R['holm'] = dict(n_tests=len(tests), n_pass_005=int((adj < 0.05).sum()),
                     tests=[dict(gen=t[0], test=t[1], model=t[2], p=t[3], holm=float(h)) for t, h in zip(tests, adj)])
    (RES / f'h24_run{tag}.json').write_text(json.dumps(R, indent=1, default=float)); log('wrote', RES / f'h24_run{tag}.json')


# ----------------------------------------------------------------------------------------------------------------- figure
def step_fig(a):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    R = json.loads((RES / 'h24_run.json').read_text())
    lab = dict(part_kin='ParT_kin', pnet_kin='ParticleNet_kin', pfn_k='PFN', pfn_t='PFN (2nd run)', efn_k='EFN (IRC-safe)',
               hw_part_kin='ParT_kin', hw_pnet_kin='ParticleNet_kin', lrn_py='LR(N), Pythia', lrn_hw='LR(N), Herwig',
               efpn_py='GBDT(EFP+N), Pythia', efpn_hw='GBDT(EFP+N), Herwig', lrp_py='LR($p_T^D$), Pythia', lrp_hw='LR($p_T^D$), Herwig', part_kin_perm='ParT_kin, permuted weights')
    groups = [('Pythia-trained', ['part_kin', 'pnet_kin', 'pfn_k', 'pfn_t', 'efn_k', 'part_kin_perm']), ('Herwig-trained', ['hw_part_kin', 'hw_pnet_kin']),
              ('References', ['lrn_py', 'lrn_hw', 'efpn_py', 'efpn_hw', 'lrp_py', 'lrp_hw'])]
    rows, gl = [], []
    for gname, ms in groups:
        ms = [m for m in ms if m in R['gens']['py']['ksplit']]
        rows += ms; gl += [gname] * len(ms)
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 0.42 * len(rows) + 2.2), sharey=True)
    for k, (key, f, title) in enumerate((('ksplit', 'slope_diff', 'exact collinear split of k soft particles\nminus matched-p$_T^D$ control (ΔN = 0)'),
                                         ('add_soft', 'diff', 'one added soft particle at donor radius\nminus control (same particle, softest removed)'))):
        for i, (gen, c, nm) in enumerate((('py', '#1f77b4', 'Pythia test jets'), ('hw', '#d62728', 'Herwig test jets'))):
            M = R['gens'][gen][key]
            mu = np.array([M[m][f] for m in rows]); ci = np.array([M[m][f + '_ci'] if key == 'ksplit' else M[m]['diff_ci'] for m in rows])
            ax[k].errorbar(mu, np.arange(len(rows)) + (i - 0.5) * 0.3, xerr=[mu - ci[:, 0], ci[:, 1] - mu], fmt='o', ms=4, color=c, lw=1, label=nm)
        ax[k].axvline(0, color='k', lw=0.6); ax[k].set_title(title, fontsize=9.5)
        ax[k].set_xlabel('Δ logit per added particle, logit = log p(gluon) − log p(quark) (95 % CI)', fontsize=8.5)
    ax[0].set_yticks(range(len(rows))); ax[0].set_yticklabels([f'{lab[m]}  [{g}]' for m, g in zip(rows, gl)], fontsize=8.5)
    ax[0].invert_yaxis(); ax[0].legend(fontsize=8, loc='lower right')
    fig.suptitle(f'Quark–gluon taggers trained on Pythia or Herwig: logit change per added particle at fixed IRC-safe content '
                 f'({R["sample"]["per_class"]} quark + {R["sample"]["per_class"]} gluon jets per generator)', fontsize=10)
    fig.tight_layout(); fig.savefig(RES / 'h24_generator_models.png', dpi=130); log('wrote h24_generator_models.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', required=True, choices=['eval', 'base', 'stats', 'fig'])
    ap.add_argument('--gens', default='py,hw'); ap.add_argument('--per_class', type=int, default=4000)
    ap.add_argument('--cpu', type=int, default=4)
    a = ap.parse_args()
    dict(eval=step_eval, base=step_base, stats=step_stats, fig=step_fig)[a.step](a)
