"""H79 (loop D; claims D30, D7 v2): which radial variable do quark-gluon networks read: r_1, D or lambda(1,0.1)?

Observables (per jet; z_i = pT_i / sum_j pT_j, dR_i = distance in (y, phi) of particle i from the E-scheme jet axis, R = 0.4):
  lambda(1, beta) = sum_i z_i (dR_i / R)^beta     generalized angularity with kappa = 1 (IRC safe); LHA = lambda(1, 1/2),
                                                   width = lambda(1, 1), lambda(1, 2) ~ m^2 / pT^2
  (pT^D)^2 = sum_i z_i^2
  T1  r_1 = dR / R of the hardest particle (largest pT) from the E-scheme axis
  T2  D = lambda(1,1) / lambda(1,1/2)^2            radial dispersion (>= 1 by Cauchy-Schwarz; = 1 when all energy sits at one radius)
  T3  lambda(1, 0.1)                               a soft-insensitive but nearly logarithmic radial moment
Edits (per jet, N >= 8). Variables: a_i = log pT_i and b_i = r_i / R (signed distance from the unedited axis along each particle's
fixed azimuth about that axis), minimum-norm Gauss-Newton steps (H76 solver) to a target of the constraint vector
  C = (log r_1, log D, log lambda(1,0.1), log[LHA * width^2], log lambda(1,2), log (pT^D)^2, log sum pT, (y_J - y_J0)/R, (phi_J - phi_J0)/R).
  (log D, log LHA*width^2) is an invertible linear map of (log LHA, log width), so holding both holds LHA and width. D is a function
  of LHA and width, so D cannot move with both held: the D arm holds LHA * width^2 (the log-space combination orthogonal to log D)
  instead, i.e. D +1 % = width +0.2 %, LHA -0.4 %. The other two arms hold LHA, width (hence D) exactly.
  <t><s><p>   t in r1, D, l01; s in +, -; p in L, M, S = relative step of the target (r1: 10, 3, 1 %; D: 1, 0.3, 0.1 %;
           lambda(1,0.1): 0.3, 0.1, 0.03 %), the other 8 fixed. A smoke run showed D +-3 % and lambda(1,0.1) +-3 % at fixed others
           infeasible for most jets (solver success 7 % and 0 %), so the plan's common +-3 % / +-1 % steps were replaced.
  <t>_cb<s><p> crand_big: per jet the larger (by norm) of the two signed edits at that step gives the per-particle size profile and the
           total norm; a random 2D direction per particle with that profile, projected onto the null space of the constraint Jacobian,
           rescaled to the full norm, applied with both signs, then restored to C0 (all nine constraints at their unedited values).
  <t>_rot  crot: the 3 particles moved most by the larger L edit rotated about the axis (random sign) by the azimuth whose arc
           length equals their radial displacement, then C0 restored.
  N+       k = max(1, round(0.1 N)) softest particles split exactly into two halves (same position), then C0 restored (only (pT^D)^2
           changes in the split). N- : k times the softest particle merged into its nearest-in-angle neighbour, then C0 restored.
  Nsm      artefact control for the count arm: the same k exact splits, then k merges of the softest NOT-split particles into
           their nearest not-split neighbour (net Delta N = 0), then C0 restored. Under additive footprints
           Delta s(N+) = b k + a_split, Delta s(N-) = -b k + a_merge, Delta s(Nsm) = a_split + a_merge, so the even count part
           (Delta s(N+) + Delta s(N-))/2 equals Delta s(Nsm)/2 if it is an edit artefact.
  Nsp      N+ before the restore.
Models: Pythia-trained part_kin, part_kin_s1, pnet_kin, pnet_kin_s1, lorentznet, pfn_k, efn_k; Herwig-trained hw_part_kin,
  hw_pnet_kin; nulls part_kin_perm, pnet_kin_perm (weight-permuted), pnet_kin_shuf (trained on hash labels).
References (HistGB, trained per generator on 20k + 20k train jets disjoint from the test sample): single r_1 (s_r1), D (s_D),
  lambda(1,0.1) (s_l01); g7: (pT^D)^2 + lambda(1, beta) for beta in 0.1, 0.25, 0.5, 1, 1.5, 2, 3 (no N).
C2ST: HistGB edited (random half A of the jets) vs unedited (other half B); for the L edits and their crand_big also out-of-fold
  isotonic-calibrated probabilities -> density-ratio weights w = (1 - p) / p (clipped at 20) that reweight edited jets to real jets.
Steps
  --step run   (GPU) -> physics/h79_<gen>.npz, h79_example_<gen>.npz, h79_run.json
  --step stats (CPU) -> physics/h79_stats.json
  --step fig   (CPU) -> physics/h79_radial.png
"""
import argparse
import json
import os
import time
import zlib
from pathlib import Path

import numpy as np

from interp import h76_angularity_qg as H76

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
V = Path('/j-jepa-vol/aaron-interp')
RES = Path(os.environ.get('AI_RESULTS', V / 'results')) / 'physics'
R0 = 0.4
TG = ['r1', 'D', 'l01']                       # constraint index 0, 1, 2
STEPS = dict(r1={'L': 0.10, 'M': 0.03, 'S': 0.01},    # relative step of each target; L = main arm. D and lambda(1,0.1) are stiff at
             D={'L': 0.01, 'M': 0.003, 'S': 0.001},    # fixed others (a +-3 % move is infeasible for most jets), so their steps
             l01={'L': 0.003, 'M': 0.001, 'S': 0.0003})  # are smaller; responses are compared per unit of the reference GBDT
PKS = ['L', 'M', 'S']
BETAS = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
TOL = 0.001          # kept quantities within 0.1 %; the other two targets within 10 % of their own L step
NETS = ['part_kin', 'part_kin_s1', 'pnet_kin', 'pnet_kin_s1', 'lorentznet', 'pfn_k', 'efn_k', 'hw_part_kin', 'hw_pnet_kin',
        'part_kin_perm', 'pnet_kin_perm', 'pnet_kin_shuf']
NULLS = ['part_kin_perm', 'pnet_kin_perm', 'pnet_kin_shuf']
REFS = ['s_r1', 's_D', 's_l01', 'g7']
RFEAT = dict(s_r1=['r1'], s_D=['D'], s_l01=['b0.1'], g7=['ptd'] + [f'b{b}' for b in BETAS])
TVAR = dict(r1='r1', D='D', l01='b0.1')


def _rng(seed, gen, name):
    return np.random.default_rng([seed, zlib.crc32(gen.encode()), zlib.crc32(name.encode())])


# ----------------------------------------------------------------------------------------------- exact observables (numpy)
def ext(a):
    pt, eta, phi = a[:, 0], a[:, 1], a[:, 2]
    px, py, pz, E = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), pt * np.cosh(eta)
    X, Y, Z, EE = px.sum(), py.sum(), pz.sum(), E.sum()
    yJ = 0.5 * np.log((EE + Z) / (EE - Z)); pJ = np.arctan2(Y, X)
    dR = np.sqrt((eta - yJ) ** 2 + ((phi - pJ + np.pi) % (2 * np.pi) - np.pi) ** 2) / R0; z = pt / pt.sum()
    o = {f'b{b}': float((z * dR ** b).sum()) for b in BETAS}
    i = int(np.argmax(pt))
    o.update(D=o['b1.0'] / max(o['b0.5'] ** 2, 1e-30), comb=o['b0.5'] * o['b1.0'] ** 2, r1=float(dR[i]), z1=float(z[i]), i1=float(i),
             ptd=float((z ** 2).sum()), N=float(len(a)), spt=float(pt.sum()), jpt=float(np.hypot(X, Y)), jy=float(yJ), jphi=float(pJ),
             mass=float(np.sqrt(max(EE ** 2 - X ** 2 - Y ** 2 - Z ** 2, 0))))
    return o


def ext_table(L):
    O = [ext(a) for a in L]
    return {k: np.array([o[k] for o in O]) for k in O[0]}


# ---------------------------------------------------------------------------------------------------------- solver (torch)
class Solver(H76.Solver):
    """H76's batched Gauss-Newton minimum-norm solver in (log pT, r/R), azimuth fixed, with the H79 constraint vector"""
    def pack(self, L):
        S = super().pack(L); t = self.t
        n = S['n']; pt = t.exp(S['x'][:, :n]) * S['m']
        S['i1'] = pt.argmax(1); return S

    def C(self, S, x):
        t = self.t; pt, eta, phi = self.coords(S, x); m = S['m']
        yJ, pJ = self._axis(pt, eta, phi)
        dR = t.sqrt((eta - yJ[:, None]) ** 2 + self._wrap(phi - pJ[:, None]) ** 2 + 1e-14) / R0
        spt = pt.sum(1); z = pt / spt[:, None]
        lam = lambda b: (z * dR ** b * m).sum(1)
        l01, lha, wid, l2 = lam(0.1), lam(0.5), lam(1.0), lam(2.0); ptd = (z ** 2 * m).sum(1)
        r1 = dR.gather(1, S['i1'][:, None])[:, 0]
        out = [t.log(r1), t.log(wid) - 2 * t.log(lha), t.log(l01), t.log(lha) + 2 * t.log(wid), t.log(l2), t.log(ptd), t.log(spt),
               (yJ - S['y0']) / R0, self._wrap(pJ - S['p0']) / R0]
        return t.stack(out, 1)


def _split_soft(a, k):
    return H76._split_soft(a, k)


def _merge_soft(a, k):
    return H76._merge_soft(a, k)


def _split_then_merge(a, k):
    """k softest split into exact halves, then k merges among particles not involved in a split (softest into nearest)"""
    o = np.argsort(a[:, 0], kind='stable')[:k]; b = a.copy(); b[o, 0] *= 0.5; b = np.concatenate([b, b[o]])
    prot = np.zeros(len(b), bool); prot[o] = True; prot[len(a):] = True
    for _ in range(k):
        cand = np.flatnonzero(~prot); i = int(cand[np.argmin(b[cand, 0])])
        d = np.hypot(b[:, 1] - b[i, 1], (b[:, 2] - b[i, 2] + np.pi) % (2 * np.pi) - np.pi); d[i] = 9; d[prot] = 9
        h = int(np.argmin(d)); w = b[i, 0] + b[h, 0]
        b[h, 1] = (b[h, 0] * b[h, 1] + b[i, 0] * b[i, 1]) / w
        b[h, 2] = b[h, 2] + b[i, 0] * ((b[i, 2] - b[h, 2] + np.pi) % (2 * np.pi) - np.pi) / w; b[h, 0] = w
        b = np.delete(b, i, 0); prot = np.delete(prot, i)
    return b


def parse(cn):
    """-> (target, step fraction) for a target edit, else (None, None)"""
    for t in TG:
        if cn.startswith(t) and len(cn) > len(t) and cn[len(t)] in '+-':
            return t, STEPS[t][cn[len(t) + 1:]]
    return None, None


def make_edits(L0, seed, gen, dev, chunk=1000):
    import torch
    Sv = Solver(dev); J = len(L0); N = np.array([len(a) for a in L0]); kN = np.maximum(1, np.rint(0.1 * N)).astype(int)
    out = {}
    def put(cn, Ls, r, x=None, x0=None):
        if cn not in out: out[cn] = ([], [], [])
        out[cn][0].extend(Ls); out[cn][1].append((r < 1e-6).cpu().numpy())
        out[cn][2].append((x - x0).norm(dim=1).cpu().numpy() if x is not None else np.full(len(Ls), np.nan))
    for s in range(0, J, chunk):
        Lc = L0[s:s + chunk]; S = Sv.pack(Lc); x0 = S['x']; c0, _ = Sv.jac(S, x0); n = S['n']
        rng = _rng(seed, gen, f'chunk{s}')
        for ti, t in enumerate(TG):
            for pk, fr in STEPS[t].items():
                xs = {}
                for sgn, st in (('+', np.log(1 + fr)), ('-', np.log(1 - fr))):
                    tg = c0.clone(); tg[:, ti] += st; x, r = Sv.solve(S, x0, tg); xs[sgn] = x
                    put(f'{t}{sgn}{pk}', Sv.unpack(S, x, Lc), r, x, x0)
                dp, dm = xs['+'] - x0, xs['-'] - x0
                big = (dp.norm(dim=1) >= dm.norm(dim=1))[:, None]; d = torch.where(big, dp, dm)
                per = torch.sqrt(d[:, :n] ** 2 + d[:, n:] ** 2)
                v = Sv.null_rand(S, x0, per, d.norm(dim=1), rng)
                for sgn, sg in (('+', 1.0), ('-', -1.0)):
                    x, r = Sv.solve(S, x0 + sg * v, c0); put(f'{t}_cb{sgn}{pk}', Sv.unpack(S, x, Lc), r, x, x0)
                if pk == 'L':
                    top = torch.argsort(per, 1, descending=True)[:, :3]; rad = (x0[:, n:] * R0).abs().gather(1, top)
                    arc = (d[:, n:].abs() * R0).gather(1, top)
                    sgn_ = torch.from_numpy(rng.choice([-1.0, 1.0], top.shape)).to(dev)
                    rot = sgn_ * (arc / rad.clamp_min(0.01)).clamp(max=np.pi)
                    th = S['th'].clone(); th.scatter_(1, top, th.gather(1, top) + rot); S2 = dict(S, th=th)
                    x, r = Sv.solve(S2, x0, c0); put(f'{t}_rot', Sv.unpack(S2, x, Lc), r, x, x0)
        # count arm
        for cn, fn in (('N+', _split_soft), ('N-', _merge_soft), ('Nsm', _split_then_merge)):
            Le = [fn(a, int(k)) for a, k in zip(Lc, kN[s:s + chunk])]
            if cn == 'N+': put('Nsp', Le, torch.zeros(len(Le)))
            S2 = Sv.pack(Le); tg = c0.clone()
            tg[:, 7] = (S['y0'] - S2['y0']) / R0; tg[:, 8] = Sv._wrap(S['p0'] - S2['p0']) / R0
            x, r = Sv.solve(S2, S2['x'], tg); put(cn, Sv.unpack(S2, x, Le), r, x, S2['x'])
        log(f'{gen}: edits {s + len(Lc)}/{J}')
    return {k: (v[0], np.concatenate(v[1]), np.concatenate(v[2])) for k, v in out.items()}, kN


KEEP = ['r1', 'D', 'b0.1', 'comb', 'b0.5', 'b1.0', 'b2.0', 'ptd', 'spt', 'jpt']


def tolerance(X0, X, cond):
    tgt, fr = parse(cond); keep = list(KEEP)
    if tgt == 'r1': keep.remove('r1')
    if tgt == 'D': keep = [k for k in keep if k not in ('D', 'b0.5', 'b1.0')]
    if tgt == 'l01': keep.remove('b0.1')
    if cond == 'Nsp': keep.remove('ptd')
    rel = lambda k: np.abs(X[k] / X0[k] - 1)
    dev = np.max(np.stack([rel(k) for k in keep] + [np.abs(X['jy'] - X0['jy']) / R0,
                                                      np.abs((X['jphi'] - X0['jphi'] + np.pi) % (2 * np.pi) - np.pi) / R0], 1), 1)
    ok = dev < TOL
    for t2 in TG:
        if t2 != tgt: ok &= rel(TVAR[t2]) < 0.1 * STEPS[t2]['L']
    if not cond.startswith('N'): ok &= (X['N'] == X0['N']) & (X['i1'] == X0['i1'])
    ach = X[TVAR[tgt]] / X0[TVAR[tgt]] - 1 if tgt else X['N'] / X0['N'] - 1
    if tgt: ok &= np.abs(np.abs(ach) - fr) < 0.05 * fr
    return ok, dev, ach


# --------------------------------------------------------------------------------------------------------------- references
def train_refs(gen, n_per_class, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from interp import h14_qg_generator as H14
    Lt, yt, _ = H14.efp_train_lists(gen, n_per_class=n_per_class, seed=seed)
    keep = np.array([len(a) >= 8 for a in Lt]); Lt = [a for a, k in zip(Lt, keep) if k]; yt = yt[keep]   # N >= 8 as the test sample
    O = ext_table([np.asarray(a, np.float64) for a in Lt]); g = (yt == 0).astype(int)
    clf = {k: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, early_stopping=True, validation_fraction=0.15,
                                             random_state=0).fit(np.c_[tuple(O[c] for c in f)], g) for k, f in RFEAT.items()}
    log(f'refs {gen}: trained on {len(yt)} train jets (seed {seed})')
    return clf


def ref_scores(clf, X):
    return {k: m.decision_function(np.c_[tuple(X[c] for c in RFEAT[k])]) for k, m in clf.items()}


# ------------------------------------------------------------------------------------------------------------------- C2ST
def c2st_feats(L, X, ix):
    F = []
    for i in ix:
        a = L[i]; pt = a[:, 0]; z = pt / pt.sum(); n = len(a)
        d = np.hypot(a[:, None, 1] - a[None, :, 1], (a[:, None, 2] - a[None, :, 2] + np.pi) % (2 * np.pi) - np.pi)
        d[np.diag_indices(n)] = 9; m6 = min(n, 6); o = np.argsort(-pt)
        r = np.hypot(a[o[:3], 1] - X['jy'][i], (a[o[:3], 2] - X['jphi'][i] + np.pi) % (2 * np.pi) - np.pi)
        F.append(np.concatenate([[n, np.log(d.min() + 1e-9), np.log(d[:m6, :m6].min() + 1e-9), pt.min(),
                                  np.log(z[o[0]] / max(z[o[1]], 1e-9))], z[o[:5]], r,
                                 [X[k][i] for k in ('ptd', 'b0.1', 'b0.25', 'b0.5', 'b1.0', 'b2.0', 'b3.0', 'D', 'mass')]]))
    return np.array(F)


def c2st(FA, FB, weights=False, seed=0):
    """FA: features of edited jets (half A), FB: unedited (half B). AUC on a held-out half; optional cross-fitted density-ratio
    weights (real / edited) for the A jets"""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed); X = np.concatenate([FA, FB]); y = np.r_[np.ones(len(FA)), np.zeros(len(FB))]
    sp = rng.permutation(len(y)); tr, te = sp[: len(y) // 2], sp[len(y) // 2:]
    pr = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y[te], pr); bs = []
    for _ in range(200):
        ix = rng.integers(0, len(te), len(te)); bs.append(roc_auc_score(y[te][ix], pr[ix]))
    out = dict(auc=float(auc), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
    w = None
    if weights:
        from sklearn.calibration import CalibratedClassifierCV
        p = np.zeros(len(y))
        for f in (0, 1):
            trn, tst = (tr, te) if f == 0 else (te, tr)
            cal = CalibratedClassifierCV(HistGradientBoostingClassifier(max_iter=200, random_state=0), method='isotonic', cv=3)
            p[tst] = cal.fit(X[trn], y[trn]).predict_proba(X[tst])[:, 1]
        p = np.clip(p[: len(FA)], 1e-3, 1 - 1e-3); w = np.minimum((1 - p) / p, 20.0); w = w / w.mean()
        out['ess_frac'] = float(w.sum() ** 2 / (w ** 2).sum() / len(w))
    return out, w


# -------------------------------------------------------------------------------------------------------------------- run
def load_models(dev):
    import torch
    from interp import h5_qg_counting as H
    from interp import h14_qg_generator as H14
    M = H.Models(dev)
    for name, arch in (('part_kin', 'part'), ('part_kin_s1', 'part'), ('pnet_kin', 'pn'), ('pnet_kin_s1', 'pn'),
                       ('pnet_kin_shuf', 'pn')):
        try: M.add_weaver(name, arch, False)
        except Exception as e: log(f'{name}: not loaded ({e})')
    M.add_lnet()
    for kind in ('pfn', 'efn'): H14.add_keras_ef(M, kind)
    for name, arch in (('part_kin', 'part'), ('pnet_kin', 'pn')):
        H14.add_weaver_dir(M, 'hw_' + name, arch, False, V / 'models' / 'qg_herwig' / name)
    M.add_permuted('part_kin_perm', 'part_kin'); M.add_permuted('pnet_kin_perm', 'pnet_kin')
    return M


def step_run(a):
    import torch
    from interp import h5_qg_counting as H
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; RES.mkdir(parents=True, exist_ok=True)
    M = load_models(dev); names = [n for n in NETS if n in M.m]; log('models', names, 'device', dev)
    gens = a.gens.split(',')
    refs = {g: train_refs(g, a.train_per_class, a.seed + 1) for g in gens}
    summary = dict(models=M.info, seed=a.seed, per_class=a.per_class, gens={})
    for gen in gens:
        L0, y, idx, f_lt8 = H76.load_sample(gen, a.per_class, a.seed); J = len(y)
        log(f'{gen}: {J} jets (N >= 8; dropped share {f_lt8:.4f}); first rows {idx[:5].tolist()}')
        E, kN = make_edits(L0, a.seed, gen, dev); X0 = ext_table(L0)
        out = dict(y=y, idx=idx, kN=kN); G = summary['gens'][gen] = dict(n=J, dropped_lt8=f_lt8, conds={}, base={}, c2st={})
        rngh = np.random.default_rng(a.seed + 5); perm = rngh.permutation(J); A, B = np.sort(perm[: J // 2]), np.sort(perm[J // 2:])
        out['halfA'] = A; FB = c2st_feats(L0, X0, B)
        for cn, (Le, okS, dn) in [('base', (L0, np.ones(J, bool), np.zeros(J)))] + list(E.items()):
            Jt = H.from_list(Le, Lpad=max(len(x) for x in Le))
            for nm in names: out[f'{cn}|{nm}'] = M.logit(nm, Jt).astype(np.float32)
            X = X0 if cn == 'base' else ext_table(Le)
            for k, v in ref_scores(refs[gen], X).items(): out[f'{cn}|{k}'] = v.astype(np.float32)
            for k in ('N', 'r1', 'D', 'b0.1', 'b0.25', 'b0.5', 'b1.0', 'b1.5', 'b2.0', 'b3.0', 'ptd', 'z1', 'mass', 'jpt'):
                out[f'obs|{cn}|{k}'] = X[k].astype(np.float32)
            if cn == 'base':
                for nm in names + REFS: G['base'][nm] = H.roc_metrics(out[f'base|{nm}'].astype(float), y)
                log(gen, 'unedited', '; '.join(f'{nm} AUC {v["AUC"]:.4f} acc {v["acc"]:.4f}' for nm, v in G['base'].items()))
                FA0 = c2st_feats(L0, X0, A); G['c2st']['none'], _ = c2st(FA0, FB); log(gen, 'c2st none', G['c2st']['none'])
                continue
            ok2, dv, ach = tolerance(X0, X, cn); ok2 &= okS if cn != 'Nsp' else True
            out[f'ok|{cn}'] = ok2; out[f'dev|{cn}'] = dv.astype(np.float32); out[f'ach|{cn}'] = ach.astype(np.float32)
            out[f'dnorm|{cn}'] = dn.astype(np.float32)
            G['conds'][cn] = dict(solver_ok=float(okS.mean()), ok=float(ok2.mean()), dev_median=float(np.median(dv[ok2])) if ok2.any() else None,
                                  dev_p99=float(np.percentile(dv[ok2], 99)) if ok2.any() else None,
                                  achieved_median=float(np.median(ach[ok2])) if ok2.any() else None,
                                  dnorm_median=float(np.nanmedian(dn)) if np.isfinite(dn).any() else None,
                                  shift_median={k: float(np.median(X[k][ok2] / X0[k][ok2] - 1)) if ok2.any() else None
                                                for k in ('r1', 'D', 'b0.1', 'b0.25', 'b0.5', 'b1.0', 'b1.5', 'b2.0', 'b3.0', 'ptd', 'z1')})
            tg_, fr_ = parse(cn); pk = cn[len(tg_) + 1:] if tg_ else None
            do_c2st = cn in ('N+', 'N-', 'Nsm', 'Nsp') or cn.endswith('_rot') or tg_ is not None or ('_cb+' in cn) or cn.endswith('_cb-L')
            if do_c2st:
                wt = (tg_ is not None and pk == 'L') or cn.endswith('_cb+L') or cn.endswith('_cb-L')
                FA = c2st_feats(Le, X, A); G['c2st'][cn], w = c2st(FA, FB, weights=wt)
                if w is not None: out[f'w|{cn}'] = w.astype(np.float32)
            log(gen, cn, json.dumps({k: v for k, v in G['conds'][cn].items() if k != 'shift_median'}),
                'c2st', G['c2st'].get(cn, {}).get('auc'), '|',
                ' '.join(f'{nm} {np.mean((out[f"{cn}|{nm}"] - out[f"base|{nm}"])[ok2]):+.4f}' for nm in names[:5]))
        np.savez_compressed(RES / f'h79_{gen}{a.tag}.npz', **out); log('wrote', RES / f'h79_{gen}{a.tag}.npz')
        okq = (y == 1) & np.all([out[f'ok|{t}{s}L'] for t in TG for s in '+-'], 0)
        if okq.any():
            j = int(np.flatnonzero(okq)[0])
            np.savez_compressed(RES / f'h79_example_{gen}{a.tag}.npz', base=L0[j], j=j,
                                **{f'{t}{s}L': E[f'{t}{s}L'][0][j] for t in TG for s in '+-'})
    (RES / f'h79_run{a.tag}.json').write_text(json.dumps(summary, indent=1, default=str)); log('run done')


# ------------------------------------------------------------------------------------------------------------------ stats
def _bmean(d, idx, w=None):
    if w is None: return np.array([d[ix].mean() for ix in idx])
    return np.array([(d[ix] * w[ix]).sum() / w[ix].sum() for ix in idx])


def _ci(b):
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def _signflip(x, P, seed=0):
    r = np.random.default_rng(seed); obs = abs(x.mean()); cnt = 0
    for c in range(0, P, 500):
        sg = r.choice(np.array([-1.0, 1.0]), (min(500, P - c), len(x))); cnt += int((np.abs(sg @ x / len(x)) >= obs - 1e-15).sum())
    return (cnt + 1) / (P + 1)


def _mcnemar(fe, fc):
    from scipy.stats import binom
    b_, c_ = int((fe & ~fc).sum()), int((fc & ~fe).sum())
    return b_, c_, (float(min(1.0, 2 * binom.cdf(min(b_, c_), b_ + c_, 0.5))) if b_ + c_ else 1.0)


def step_stats(a):
    S = dict(B=a.boot, P=a.perm, gens={}); tests = []
    for gen in a.gens.split(','):
        Z = dict(np.load(RES / f'h79_{gen}{a.tag}.npz')); y = Z['y'].astype(int); J = len(y); A = Z['halfA']
        models = [m for m in NETS + REFS if f'base|{m}' in Z]
        G = S['gens'][gen] = dict(n=J, arms={}, count={}, models={}, ok={})
        rng = np.random.default_rng(7901)
        for t in TG:
            for pk in PKS:
                conds = [f'{t}+{pk}', f'{t}-{pk}', f'{t}_cb+{pk}', f'{t}_cb-{pk}']
                ok = np.all([Z[f'ok|{c}'] for c in conds], 0); G['ok'][f'{t}{pk}'] = float(ok.mean())
                jj = np.flatnonzero(ok); idx = rng.integers(0, len(jj), (a.boot, len(jj)))
                arm = G['arms'][f'{t}{pk}'] = dict(n_ok=int(len(jj)), models={})
                # unconstrained movement (median relative change, accepted jets)
                arm['shift'] = {s: {k: float(np.median(Z[f'obs|{t}{s}{pk}|{k}'][jj] / Z[f'obs|base|{k}'][jj] - 1))
                                    for k in ('r1', 'D', 'b0.1', 'b0.25', 'b0.5', 'b1.0', 'b1.5', 'b2.0', 'b3.0', 'ptd', 'z1')} for s in '+-'}
                arm['dnorm_median'] = {s: float(np.median(Z[f'dnorm|{t}{s}{pk}'][jj])) for s in '+-'}
                for m in models:
                    s0 = Z[f'base|{m}'].astype(float)
                    ds = {c: (Z[f'{c}|{m}'].astype(float) - s0)[jj] for c in conds}
                    cb = 0.5 * (ds[conds[2]] + ds[conds[3]]); R = arm['models'][m] = {}
                    for s, c in (('+', conds[0]), ('-', conds[1])):
                        e = ds[c] - cb
                        R[s] = dict(dS=float(ds[c].mean()), dS_ci=_ci(_bmean(ds[c], idx)), excess=float(e.mean()), excess_ci=_ci(_bmean(e, idx)))
                        if m in NETS and pk == 'L':
                            p = _signflip(e, a.perm, seed=len(tests)); R[s]['p'] = p; tests.append((gen, m, f'{t}{s}L', p))
                            thr = np.median(s0)                   # frozen working point: half the sample called gluon
                            c0_ = s0[jj] > thr; fe = (Z[f'{c}|{m}'].astype(float)[jj] > thr) != c0_
                            fc = (Z[f'{t}_cb{s}{pk}|{m}'].astype(float)[jj] > thr) != c0_
                            b_, c_, pm = _mcnemar(fe, fc); R[s].update(flip_edit=float(fe.mean()), flip_cb=float(fc.mean()), mc_b=b_, mc_c=c_, p_mcnemar=pm)
                    R['cb'] = float(cb.mean()); R['cb_ci'] = _ci(_bmean(cb, idx))
                    ev = 0.5 * (ds[conds[0]] + ds[conds[1]]) - cb
                    od = 0.5 * (ds[conds[0]] - ds[conds[1]]) - 0.5 * (ds[conds[2]] - ds[conds[3]])
                    R['even'] = float(ev.mean()); R['even_ci'] = _ci(_bmean(ev, idx))
                    R['odd'] = float(od.mean()); R['odd_ci'] = _ci(_bmean(od, idx)); R['_odd_jets'] = od
                    if pk == 'L':                                   # reweighted to real jets (density ratio, half A)
                        inA = np.isin(jj, A); pos = np.searchsorted(A, jj[inA]); ia = rng.integers(0, inA.sum(), (a.boot, inA.sum()))
                        for s, c in (('+', conds[0]), ('-', conds[1])):
                            wc = f'w|{c}'; wb = f'w|{t}_cb{s}{pk}'
                            if wc in Z and wb in Z:
                                we = Z[wc][pos].astype(float); wcb = Z[wb][pos].astype(float)
                                de = ds[c][inA]; dc = 0.5 * (ds[conds[2]] + ds[conds[3]])[inA]
                                val = (de * we).sum() / we.sum() - (dc * wcb).sum() / wcb.sum()
                                bb = _bmean(de, ia, we) - _bmean(dc, ia, wcb)
                                R[s]['rw_excess'] = float(val); R[s]['rw_excess_ci'] = _ci(bb)
                # per unit of the target's own reference (odd part of net / odd part of the single-variable GBDT on this target)
                ref_t = f's_{t}'; od_ref = arm['models'][ref_t]['_odd_jets']
                for m in models:
                    od = arm['models'][m]['_odd_jets']; num = _bmean(od, idx); den = _bmean(od_ref, idx)
                    arm['models'][m]['per_ref_unit'] = float(od.mean() / od_ref.mean()) if abs(od_ref.mean()) > 1e-9 else None
                    arm['models'][m]['per_ref_unit_ci'] = _ci(num / np.where(np.abs(den) > 1e-9, den, np.nan)) if abs(od_ref.mean()) > 1e-9 else None
                for m in models: arm['models'][m].pop('_odd_jets')
            log(gen, t, {pk: {m: (round(G['arms'][f'{t}{pk}']['models'][m]['+']['excess'], 3), round(G['arms'][f'{t}{pk}']['models'][m]['-']['excess'], 3))
                               for m in ('part_kin', 'efn_k', 's_r1', 's_D', 's_l01', 'g7') if m in models} for pk in PKS})
        # crot (3 %), vs zero
        for t in TG:
            ok = Z[f'ok|{t}_rot']; jj = np.flatnonzero(ok); idx = rng.integers(0, len(jj), (a.boot, len(jj)))
            G['arms'][f'{t}L']['crot'] = {m: dict(dS=float((Z[f'{t}_rot|{m}'] - Z[f'base|{m}'])[jj].mean()),
                                                 ci=_ci(_bmean((Z[f'{t}_rot|{m}'] - Z[f'base|{m}']).astype(float)[jj], idx))) for m in models}
        # count arm
        ok = Z['ok|N+'] & Z['ok|N-'] & Z['ok|Nsm']; jj = np.flatnonzero(ok); idx = rng.integers(0, len(jj), (a.boot, len(jj)))
        G['count']['n_ok'] = int(len(jj)); N0 = Z['obs|base|N'][jj]; kN = Z['kN'][jj]
        for m in models:
            s0 = Z[f'base|{m}'].astype(float)[jj]; d = {c: Z[f'{c}|{m}'].astype(float)[jj] - s0 for c in ('N+', 'N-', 'Nsm', 'Nsp')}
            ev = 0.5 * (d['N+'] + d['N-']); od = 0.5 * (d['N+'] - d['N-']) * 0.1 / (kN / N0); half = 0.5 * d['Nsm']; ex = ev - half
            R = G['count'][m] = {c: dict(dS=float(v.mean()), ci=_ci(_bmean(v, idx))) for c, v in d.items()}
            R['even'] = dict(v=float(ev.mean()), ci=_ci(_bmean(ev, idx))); R['odd_per10'] = dict(v=float(od.mean()), ci=_ci(_bmean(od, idx)))
            R['Nsm_half'] = dict(v=float(half.mean()), ci=_ci(_bmean(half, idx)))
            R['even_minus_Nsm_half'] = dict(v=float(ex.mean()), ci=_ci(_bmean(ex, idx)))
            if m in NETS:
                p = _signflip(ex, a.perm, seed=900 + len(tests)); R['even_minus_Nsm_half']['p'] = p; tests.append((gen, m, 'count_even-Nsm/2', p))
        log(gen, 'count', {m: (round(G['count'][m]['even']['v'], 4), round(G['count'][m]['Nsm_half']['v'], 4)) for m in models})
    from interp.h76_angularity_qg import _holm
    adj = _holm([p for *_, p in tests])
    S['holm'] = [dict(gen=g, model=m, test=t, p=p, p_holm=float(q)) for (g, m, t, p), q in zip(tests, adj)]
    S['n_tests'] = len(tests); S['n_holm_005'] = int((adj < 0.05).sum())
    hm = {(d['gen'], d['model'], d['test']): d['p_holm'] for d in S['holm']}
    for gen, G in S['gens'].items():
        for t in TG:
            for m, R in G['arms'][f'{t}L']['models'].items():
                for s in '+-':
                    if (gen, m, f'{t}{s}L') in hm: R[s]['p_holm'] = hm[(gen, m, f'{t}{s}L')]
        for m, R in G['count'].items():
            if isinstance(R, dict) and (gen, m, 'count_even-Nsm/2') in hm: R['even_minus_Nsm_half']['p_holm'] = hm[(gen, m, 'count_even-Nsm/2')]
    (RES / f'h79_stats{a.tag}.json').write_text(json.dumps(S, indent=1))
    log('wrote', RES / f'h79_stats{a.tag}.json', 'Holm<0.05:', S['n_holm_005'], '/', len(tests))


# -------------------------------------------------------------------------------------------------------------------- fig
def step_fig(a):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    S = json.loads((RES / f'h79_stats{a.tag}.json').read_text()); X = dict(np.load(RES / f'h79_example_py{a.tag}.npz'))
    run = json.loads((RES / f'h79_run{a.tag}.json').read_text())
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw=dict(width_ratios=[1, 1.25]))
    rg = np.geomspace(2e-3, 1.0, 600)
    def psi(L):
        e = ext(L); dR = np.hypot(L[:, 1] - e['jy'], (L[:, 2] - e['jphi'] + np.pi) % (2 * np.pi) - np.pi) / R0; z = L[:, 0] / L[:, 0].sum()
        return np.array([(z * (dR <= r)).sum() for r in rg]), e
    p0, e0 = psi(X['base']); col = dict(r1='C0', D='C3', l01='C2'); lab = dict(r1='r₁', D='D', l01='λ(1,0.1)')
    for t in TG:
        for s, ls in (('+', '-'), ('-', ':')):
            p1, e1 = psi(X[f'{t}{s}L'])
            ax[0].step(rg, 100 * (p1 - p0), ls, where='post', color=col[t], lw=1.3,
                       label=f'{lab[t]} {s}{100 * STEPS[t]["L"]:g} %: r₁ {e0["r1"]:.3f}→{e1["r1"]:.3f}, D {e0["D"]:.3f}→{e1["D"]:.3f}, λ(1,0.1) {e0["b0.1"]:.3f}→{e1["b0.1"]:.3f}')
    ax[0].axhline(0, color='0.6', lw=0.6); ax[0].axvline(e0['r1'], color='0.4', ls='--', lw=0.8)
    ax[0].set_xscale('log'); ax[0].set_xlim(2e-3, 1.0)
    ax[0].text(e0['r1'], ax[0].get_ylim()[1] * 0.9, ' r₁ (unedited)', fontsize=7)
    ax[0].set_xlabel('distance from the E-scheme axis, ΔR / R  (R = 0.4, log scale)')
    ax[0].set_ylabel('edited − unedited share of jet pT within ΔR  [% of jet pT]')
    ax[0].set_title(f'one Pythia quark jet (N = {int(e0["N"])}): radial pT profile change per edit\n'
                    f'(r₁ ±10 %, D ±1 %, λ(1,0.1) ±0.3 %; other targets, LHA·width², λ(1,2), (pT^D)², Σ pT, axis, N held)', fontsize=8.5)
    ax[0].legend(fontsize=6.3, loc='lower left')
    # per-sign excess over crand_big, 3 % edits, Pythia jets
    G = S['gens']['py']['arms']; ms = [m for m in ('part_kin', 'part_kin_s1', 'pnet_kin', 'lorentznet', 'efn_k', 'hw_part_kin', 'part_kin_perm',
                                                   's_r1', 's_D', 's_l01', 'g7') if m in G['r1L']['models']]
    nice = dict(part_kin='ParT_kin', part_kin_s1='ParT_kin s1', pnet_kin='PN_kin', lorentznet='LorentzNet', efn_k='EFN', hw_part_kin='ParT_kin (Herwig-trained)',
                part_kin_perm='ParT_kin permuted', s_r1='GBDT(r₁)', s_D='GBDT(D)', s_l01='GBDT(λ(1,0.1))', g7='GBDT(7 λ + pT^D)')
    arms = [(t, s) for t in TG for s in '+-']; w = 0.8 / len(ms)
    for i, m in enumerate(ms):
        v = [G[f'{t}L']['models'][m][s]['excess'] for t, s in arms]
        lo = [G[f'{t}L']['models'][m][s]['excess_ci'] for t, s in arms]
        err = np.abs(np.array(lo).T - np.array(v)[None])
        ax[1].bar(np.arange(6) + (i - len(ms) / 2) * w + w / 2, v, w, yerr=err, label=nice[m], capsize=1.5,
                  color=plt.cm.tab20(i), hatch='//' if m.startswith(('s_', 'g7')) else None, edgecolor='k' if m.startswith(('s_', 'g7')) else None, lw=0.3)
    ax[1].axhline(0, color='0.4', lw=0.6); ax[1].set_xticks(range(6)); ax[1].set_xticklabels([f'{lab[t]} {s}{100 * STEPS[t]["L"]:g} %' for t, s in arms])
    ax[1].set_ylabel('Δ gluon log-odds, edit minus crand_big  (95 % CI)')
    ax[1].set_title('Pythia jets: response to moving one radial variable at fixed others\n(hatched: trees trained on that variable / on all angularities)', fontsize=9)
    ax[1].legend(fontsize=6.5, ncol=2)
    fig.tight_layout(); p = RES / f'h79_radial{a.tag}.png'; fig.savefig(p, dpi=130); log('wrote', p)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', required=True, choices=['run', 'stats', 'fig'])
    ap.add_argument('--gens', default='py,hw'); ap.add_argument('--per_class', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=7901); ap.add_argument('--tag', default='')
    ap.add_argument('--train_per_class', type=int, default=20000)
    ap.add_argument('--boot', type=int, default=1000); ap.add_argument('--perm', type=int, default=10000)
    a = ap.parse_args(); RES.mkdir(parents=True, exist_ok=True)
    dict(run=step_run, stats=step_stats, fig=step_fig)[a.step](a)
