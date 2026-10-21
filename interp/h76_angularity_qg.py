"""H76 (loop D; claims D7, D22): which generalized angularity do quark-gluon taggers follow?

Generalized angularity (Larkoski, Thaler, Waalewijn 2014): lambda^kappa_beta = sum_i z_i^kappa (dR_i / R)^beta, z_i = pT_i / sum_j pT_j,
dR_i = distance in (rapidity, phi) of particle i from the E-scheme jet axis (four-vector sum), R = 0.4. Five reference directions:
  N    (0, 0)   particle multiplicity
  ptd  (2, 0)   sum z_i^2 = (pT^D)^2
  lha  (1, 0.5) Les Houches angularity
  wid  (1, 1)   width / girth (in units of R)
  mass (1, 2)   ~ m^2 / pT^2 (in units of R^2)

Edits (per jet; jets with N >= 8). Variables per particle: a_i = log pT_i and b_i = r_i / R, r_i = signed distance from the unedited
E-scheme axis along the particle's fixed azimuth theta_i about that axis (a "z move" changes a_i, a "dR move" changes b_i); the
minimum-norm step minimises sum_i (delta log pT_i)^2 + (delta r_i / R)^2.
Constraint vector C = (log lambda_ptd, log lambda_lha, log lambda_wid, log lambda_mass, log sum pT, (y_J - y_J0)/R, (phi_J - phi_J0)/R).
  t+ / t-  (t in lha, wid, ptd, mass): Gauss-Newton minimum-norm steps in (a, b) to C* = C0 + log(1 +- 0.03) on t, others
           fixed; N is unchanged (no particle added or removed). Reported per +10 % by linear scaling (x 10/3).
  t+10 / t-10  the same at +-10 % (side arm: LHA and width at +-10 % with the other three fixed are often infeasible, because
           log lambda^1_beta is convex in beta (Lyapunov), so the solver fails on part of the jets).
  N+       the k = max(1, round(0.1 N)) softest particles split exactly (two halves, same position; every IRC-safe angularity
           unchanged), then (pT^D)^2, which the split lowers, restored with the same solver (C* = C0).
  N-       k times: the softest particle merged into its nearest-in-angle neighbour (pT summed, pT-weighted position), then C* = C0.
  Nsp      N+ before the pT^D restore (exact split only): the EFN must give 0 here.
Controls
  t_r+ / t_r-  crand: per particle a random 2D direction in (a_i, b_i) with the per-particle size |delta_i| of the t+ edit,
           projected onto the null space of the constraint Jacobian, rescaled to the total size ||delta|| of the t+ edit, then
           restored to C0 (all five angularities, pT, axis at their unedited values); the minus edit uses the opposite vector.
  t_rot    crot: the 3 particles that the t+ edit moves most rotated in azimuth about the axis, random sign, by the angle whose
           arc length equals that particle's radial displacement in the t+ edit, then C0 restored (positions changed at fixed angularities).
Slope per jet (gluon log-odds s = log p(g) - log p(q)): d_t = (s(t+) - s(t-)) / 2 x 0.1 / 0.03 per +10 % move; for N x 0.1 / (k / N).
Models (part_job.sh, GPU): Pythia-trained part_kin, pnet_kin, lorentznet, pfn_k, efn_k (H2 energyflow ports), pfn_t, efn_t (H5 torch),
  Herwig-trained hw_part_kin, hw_pnet_kin (H14/H24), permuted part_kin_perm, pnet_kin_perm.
References (trained here on 30k + 30k train jets per generator, py and hw): GBDT on each single angularity (g1_<t>), GBDT on all
  five (g5), LR(N) negative binomial (lrn; interp.h14_qg_generator.lrn_tables).
Steps
  --step run    (GPU) sample, edits, model logits, references, C2ST -> physics/h76_<gen><tag>.npz, h76_run<tag>.json
  --step stats  (CPU) bootstrap / sign-flip / Holm / cosine -> physics/h76_stats<tag>.json
  --step fig    (CPU) physics/h76_angularity<tag>.png
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
R0 = 0.4
DIRS = ['N', 'lha', 'wid', 'ptd', 'mass']          # vector order (plan order: N, LHA, width, pT^D, mass)
IRCS = ['ptd', 'lha', 'wid', 'mass']                # constraint order inside C
KB = dict(ptd=(2, 0), lha=(1, 0.5), wid=(1, 1), mass=(1, 2))
FRAC = 0.03          # main step of the IRC-safe edits (+-3 %, reported per +10 % by linear scaling); +-10 % run as a side arm
NETS = ['part_kin', 'pnet_kin', 'lorentznet', 'pfn_k', 'efn_k', 'pfn_t', 'efn_t', 'hw_part_kin', 'hw_pnet_kin',
        'part_kin_perm', 'pnet_kin_perm']
TOL = 0.005


def _rng(seed, gen, name):
    return np.random.default_rng([seed, zlib.crc32(gen.encode()), zlib.crc32(name.encode())])


# ----------------------------------------------------------------------------------------------- exact observables (numpy)
def angs(a):
    """exact per-jet observables of a particle list (pt, eta(=y, massless), phi, pid): N, lambda's, jet pT / eta / phi / mass"""
    pt, eta, phi = a[:, 0], a[:, 1], a[:, 2]
    px, py, pz, E = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), pt * np.cosh(eta)
    X, Y, Z, EE = px.sum(), py.sum(), pz.sum(), E.sum()
    yJ = 0.5 * np.log((EE + Z) / (EE - Z)); pJ = np.arctan2(Y, X); jpt = np.hypot(X, Y)
    dR = np.sqrt((eta - yJ) ** 2 + ((phi - pJ + np.pi) % (2 * np.pi) - np.pi) ** 2); z = pt / pt.sum()
    out = dict(N=float(len(a)), spt=float(pt.sum()), jpt=float(jpt), jeta=float(np.arcsinh(Z / jpt)), jy=float(yJ), jphi=float(pJ),
               mass=float(np.sqrt(max(EE ** 2 - X ** 2 - Y ** 2 - Z ** 2, 0))))
    for t, (k, b) in KB.items(): out['l_' + t] = float((z ** k * (dR / R0) ** b).sum())
    return out


def obs_table(L):
    O = [angs(a) for a in L]
    return {k: np.array([o[k] for o in O]) for k in O[0]}


def feat5(O):
    return np.c_[O['N'], O['l_lha'], O['l_wid'], O['l_ptd'], O['l_mass']]


# ---------------------------------------------------------------------------------------------------------- solver (torch)
class Solver:
    """batched Gauss-Newton minimum-norm solver in (a = log pT, b = log r) about the unedited axis of each list"""
    def __init__(self, dev):
        import torch
        self.t = torch; self.dev = dev

    def pack(self, L):
        t = self.t; B = len(L); n = max(len(a) for a in L)
        P = np.zeros((B, n, 3)); m = np.zeros((B, n), bool)
        for j, a in enumerate(L): P[j, :len(a)] = a[:, :3]; m[j, :len(a)] = True
        P = t.from_numpy(P).to(self.dev); m = t.from_numpy(m).to(self.dev)
        pt, eta, phi = P[..., 0], P[..., 1], P[..., 2]
        y0, p0 = self._axis(pt * m, eta, phi)
        de = eta - y0[:, None]; dp = self._wrap(phi - p0[:, None])
        r = t.sqrt(de ** 2 + dp ** 2).clamp_min(1e-5); th = t.atan2(dp, de)
        a = t.where(m, t.log(pt.clamp_min(1e-12)), t.zeros_like(pt)); b = t.where(m, r / R0, t.zeros_like(r))
        return dict(m=m, y0=y0, p0=p0, th=th, x=t.cat([a, b], 1), n=n)

    def _wrap(self, d):
        return (d + np.pi) % (2 * np.pi) - np.pi

    def _axis(self, pt, eta, phi):
        t = self.t
        px, py, pz, E = (pt * t.cos(phi)).sum(1), (pt * t.sin(phi)).sum(1), (pt * t.sinh(eta)).sum(1), (pt * t.cosh(eta)).sum(1)
        return 0.5 * t.log((E + pz) / (E - pz)), t.atan2(py, px)

    def coords(self, S, x):
        t = self.t; n = S['n']; m = S['m']
        pt = t.exp(x[:, :n]) * m; r = x[:, n:] * R0          # r may pass through the axis (negative = opposite side)
        eta = S['y0'][:, None] + r * t.cos(S['th']); phi = S['p0'][:, None] + r * t.sin(S['th'])
        return pt, eta, phi

    def C(self, S, x):
        t = self.t; pt, eta, phi = self.coords(S, x)
        yJ, pJ = self._axis(pt, eta, phi)
        dR = t.sqrt((eta - yJ[:, None]) ** 2 + self._wrap(phi - pJ[:, None]) ** 2 + 1e-14)
        spt = pt.sum(1); z = pt / spt[:, None]; out = []
        for k in IRCS:
            kk, bb = KB[k]; out.append(t.log((z ** kk * (dR / R0) ** bb * S['m']).sum(1)))
        out += [t.log(spt), (yJ - S['y0']) / R0, self._wrap(pJ - S['p0']) / R0]
        return t.stack(out, 1)

    def jac(self, S, x):
        t = self.t; x = x.detach().requires_grad_(True); c = self.C(S, x); mv = t.cat([S['m'], S['m']], 1).double()
        J = t.stack([t.autograd.grad(c[:, k].sum(), x, retain_graph=k < c.shape[1] - 1)[0] * mv for k in range(c.shape[1])], 1)
        return c.detach(), J

    def solve(self, S, x, target, iters=100, maxstep=0.2):
        t = self.t; x = x.clone()
        for it in range(iters):
            c, J = self.jac(S, x); res = target - c
            if float(res.abs().max()) < 1e-10: break
            A = J @ J.transpose(1, 2) + 1e-12 * t.eye(J.shape[1], dtype=J.dtype, device=J.device)
            w = t.linalg.solve(A, res[..., None]); d = (J.transpose(1, 2) @ w)[..., 0]
            sc = (maxstep / d.abs().amax(1).clamp_min(1e-30)).clamp(max=1.0); x = x + d * sc[:, None]
        c = self.C(S, x).detach(); return x, (target - c).abs().amax(1)

    def null_rand(self, S, x0, per, norm, rng):
        """random vector with per-particle sizes `per` (B, n), random 2D direction per particle, projected onto the null space of
        the constraint Jacobian at x0, rescaled to total size `norm`"""
        t = self.t; n = S['n']; ang = t.from_numpy(rng.uniform(0, 2 * np.pi, per.shape)).to(self.dev)
        v = t.cat([per * t.cos(ang), per * t.sin(ang)], 1) * t.cat([S['m'], S['m']], 1)
        _, J = self.jac(S, x0); A = J @ J.transpose(1, 2) + 1e-12 * t.eye(J.shape[1], dtype=J.dtype, device=J.device)
        v = v - (J.transpose(1, 2) @ t.linalg.solve(A, (J @ v[..., None])))[..., 0]
        return v * (norm / v.norm(dim=1).clamp_min(1e-30))[:, None]

    def unpack(self, S, x, Lsrc):
        pt, eta, phi = [z.detach().cpu().numpy() for z in self.coords(S, x)]
        return [np.c_[pt[j, :len(a)], eta[j, :len(a)], phi[j, :len(a)], a[:, 3]] for j, a in enumerate(Lsrc)]


def _split_soft(a, k):
    o = np.argsort(a[:, 0], kind='stable')[:k]; b = a.copy(); b[o, 0] *= 0.5
    return np.concatenate([b, b[o]])


def _merge_soft(a, k):
    b = a.copy()
    for _ in range(k):
        i = int(np.argmin(b[:, 0])); d = np.hypot(b[:, 1] - b[i, 1], (b[:, 2] - b[i, 2] + np.pi) % (2 * np.pi) - np.pi); d[i] = 9
        h = int(np.argmin(d)); w = b[i, 0] + b[h, 0]
        b[h, 1] = (b[h, 0] * b[h, 1] + b[i, 0] * b[i, 1]) / w
        b[h, 2] = b[h, 2] + b[i, 0] * ((b[i, 2] - b[h, 2] + np.pi) % (2 * np.pi) - np.pi) / w; b[h, 0] = w
        b = np.delete(b, i, 0)
    return b


def parse(cn):
    """edit condition -> (target, |relative step|) or (None, None) for controls"""
    if cn in ('N+', 'N-'): return 'N', None
    if cn.endswith('10') and cn[:-3] in IRCS: return cn[:-3], 0.10
    if cn[:-1] in IRCS and cn[-1] in '+-': return cn[:-1], FRAC
    return None, None


def make_edits(L0, seed, gen, dev, chunk=1000):
    """-> dict cond -> (list, ok, info dict of per-jet arrays)"""
    import torch
    Sv = Solver(dev); J = len(L0); N = np.array([len(a) for a in L0]); kN = np.maximum(1, np.rint(0.1 * N)).astype(int)
    out = {}
    def put(cn, Ls, ok, **kw):
        if cn not in out: out[cn] = ([], [], {})
        out[cn][0].extend(Ls); out[cn][1].append(ok)
        for k, v in kw.items(): out[cn][2].setdefault(k, []).append(v)
    for s in range(0, J, chunk):
        Lc = L0[s:s + chunk]; S = Sv.pack(Lc); x0 = S['x']; c0, _ = Sv.jac(S, x0); n = S['n']
        rng = _rng(seed, gen, f'chunk{s}')
        for ti, t in enumerate(IRCS):
            xs = {}
            for fr, suf in ((FRAC, ''), (0.10, '10')):
                for sgn, st in (('+', np.log(1 + fr)), ('-', np.log(1 - fr))):
                    tg = c0.clone(); tg[:, ti] += st; x, r = Sv.solve(S, x0, tg)
                    if suf == '': xs[sgn] = x
                    put(t + sgn + suf, Sv.unpack(S, x, Lc), (r < 1e-6).cpu().numpy(), res=r.cpu().numpy(),
                        dnorm=(x - x0).norm(dim=1).cpu().numpy())
            d = xs['+'] - x0; per = torch.sqrt(d[:, :n] ** 2 + d[:, n:] ** 2); nrm = 0.5 * ((xs['+'] - x0).norm(dim=1) + (xs['-'] - x0).norm(dim=1))
            v = Sv.null_rand(S, x0, per, nrm, rng)
            for sgn, sg in (('+', 1.0), ('-', -1.0)):
                x, r = Sv.solve(S, x0 + sg * v, c0)
                put(f'{t}_r{sgn}', Sv.unpack(S, x, Lc), (r < 1e-6).cpu().numpy(), res=r.cpu().numpy(), dnorm=(x - x0).norm(dim=1).cpu().numpy())
            # crot: rotate the 3 most-moved particles in azimuth about the axis, then restore C0
            top = torch.argsort(per, 1, descending=True)[:, :3]; rad = (x0[:, n:] * R0).abs().gather(1, top)
            arc = (d[:, n:].abs() * R0).gather(1, top)                    # radial displacement of that particle in the t+ edit
            sgn_ = torch.from_numpy(rng.choice([-1.0, 1.0], top.shape)).to(dev)
            rot = sgn_ * (arc / rad.clamp_min(0.01)).clamp(max=np.pi)      # rotation whose arc length matches that displacement
            th = S['th'].clone()
            th.scatter_(1, top, th.gather(1, top) + rot); S2 = dict(S, th=th)
            x, r = Sv.solve(S2, x0, c0)
            put(f'{t}_rot', Sv.unpack(S2, x, Lc), (r < 1e-6).cpu().numpy(), res=r.cpu().numpy())
        # N direction
        for sgn, fn in (('+', _split_soft), ('-', _merge_soft)):
            Le = [fn(a, int(k)) for a, k in zip(Lc, kN[s:s + chunk])]
            if sgn == '+': put('Nsp', Le, np.ones(len(Le), bool))
            S2 = Sv.pack(Le); tg = c0.clone()
            tg[:, 5] = (S['y0'] - S2['y0']) / R0; tg[:, 6] = Sv._wrap(S['p0'] - S2['p0']) / R0   # axis back to the unedited one
            x, r = Sv.solve(S2, S2['x'], tg)
            put('N' + sgn, Sv.unpack(S2, x, Le), (r < 1e-6).cpu().numpy(), res=r.cpu().numpy(),
                dnorm=(x - S2['x']).norm(dim=1).cpu().numpy())
            if sgn == '+': S_np, x_np, xN, tgN = S2, S2['x'], x, tg
        # crand for N: size of the restoring move of N+, on the split jet, then restored
        d = xN - x_np; per = torch.sqrt(d[:, :S_np['n']] ** 2 + d[:, S_np['n']:] ** 2)
        v = Sv.null_rand(S_np, x_np, per, d.norm(dim=1), rng); Lsp = out['Nsp'][0][s:s + chunk]
        for sgn, sg in (('+', 1.0), ('-', -1.0)):
            x, r = Sv.solve(S_np, x_np + sg * v, tgN)
            put(f'N_r{sgn}', Sv.unpack(S_np, x, Lsp), (r < 1e-6).cpu().numpy(), res=r.cpu().numpy())
        log(f'{gen}: edits {s + len(Lc)}/{J}')
    return {k: (v[0], np.concatenate(v[1]), {kk: np.concatenate(vv) for kk, vv in v[2].items()}) for k, v in out.items()}, kN


def tolerance(O0, O, cond):
    """max relative deviation of every quantity that the edit should keep, and the achieved relative change of the target"""
    keep = ['l_' + t for t in IRCS]
    tgt = parse(cond)[0]
    if tgt in IRCS: keep.remove('l_' + tgt)
    if cond == 'Nsp': keep.remove('l_ptd')                     # exact split before the pT^D restore
    rel = lambda k: np.abs(O[k] / O0[k] - 1)
    dev = np.max(np.stack([rel(k) for k in keep] + [rel('jpt'), rel('spt'), np.abs(O['jeta'] - O0['jeta']) / R0,
                                                       np.abs((O['jphi'] - O0['jphi'] + np.pi) % (2 * np.pi) - np.pi) / R0], 1), 1)
    if not cond.startswith('N'): dev = np.maximum(dev, rel('N'))
    ach = O['l_' + tgt] / O0['l_' + tgt] - 1 if tgt in IRCS else O['N'] / O0['N'] - 1
    return dev, ach


# --------------------------------------------------------------------------------------------------------------- references
def train_refs(gen, n_per_class):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from interp import h14_qg_generator as H14
    Lt, yt, _ = H14.efp_train_lists(gen, n_per_class=n_per_class, seed=7601 + 1)
    O = obs_table([np.c_[a, np.zeros(len(a))] for a in Lt]); F = feat5(O); g = (yt == 0).astype(int)
    mk = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, early_stopping=True, validation_fraction=0.15,
                                                random_state=0)
    clf = {f'g1_{t}': mk().fit(F[:, [i]], g) for i, t in enumerate(DIRS)}; clf['g5'] = mk().fit(F, g)
    log(f'refs {gen}: trained on {len(yt)} jets; train-set means q/g N {O["N"][yt == 1].mean():.1f}/{O["N"][yt == 0].mean():.1f}')
    return clf


def ref_scores(clf, O, T, gen):
    F = feat5(O); out = {}
    for k, m in clf.items():
        out[k] = m.decision_function(F[:, [DIRS.index(k[3:])]] if k.startswith('g1_') else F)
    out['lrn'] = T[gen][np.minimum(O['N'].astype(int), 299)]
    return out


# ------------------------------------------------------------------------------------------------------------------- C2ST
def c2st(L0, Le, O0, Oe, seed=0):
    """GBDT two-sample test, edited jets (half A) vs unedited jets (half B): h5 low-level summaries + the five angularities,
    the radii of the 3 leading particles, jet mass"""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    def feats(L, O, ix):
        F = []
        for i in ix:
            a = L[i]; pt = a[:, 0]; z = pt / pt.sum(); n = len(a)
            d = np.hypot(a[:, None, 1] - a[None, :, 1], (a[:, None, 2] - a[None, :, 2] + np.pi) % (2 * np.pi) - np.pi)
            d[np.diag_indices(n)] = 9; m6 = min(n, 6); o = np.argsort(-pt)
            r = np.hypot(a[o[:3], 1] - O['jy'][i], (a[o[:3], 2] - O['jphi'][i] + np.pi) % (2 * np.pi) - np.pi)
            F.append(np.concatenate([[n, np.log(d.min() + 1e-9), np.log(d[:m6, :m6].min() + 1e-9), pt.min(),
                                      np.log(z[o[0]] / max(z[o[1]], 1e-9))], z[o[:5]], r,
                                     [O[k][i] for k in ('l_ptd', 'l_lha', 'l_wid', 'l_mass', 'mass')]]))
        return np.array(F)
    rng = np.random.default_rng(seed); J = len(L0); p = rng.permutation(J); A, B = p[: J // 2], p[J // 2:]
    X = np.concatenate([feats(Le, Oe, A), feats(L0, O0, B)]); y = np.r_[np.ones(len(A)), np.zeros(len(B))]
    sp = rng.permutation(len(y)); tr, te = sp[: len(y) // 2], sp[len(y) // 2:]
    pr = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y[te], pr); bs = []
    for _ in range(200):
        ix = rng.integers(0, len(te), len(te)); bs.append(roc_auc_score(y[te][ix], pr[ix]))
    return dict(auc=float(auc), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])


# -------------------------------------------------------------------------------------------------------------------- run
def load_sample(gen, per_class, seed):
    from interp import h14_qg_generator as H14
    L0, y, info = H14.load_sample(gen, int(per_class * 1.25) + 200, seed)
    N = np.array([len(a) for a in L0]); rng = np.random.default_rng([seed, 1])
    keep = np.sort(np.concatenate([rng.choice(np.flatnonzero((y == c) & (N >= 8)), per_class, replace=False) for c in (0, 1)]))
    return [L0[i] for i in keep], y[keep], info['idx'][keep], float((N < 8).mean())


def step_run(a):
    import torch
    from interp import h5_qg_counting as H
    from interp import h14_qg_generator as H14
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; RES.mkdir(parents=True, exist_ok=True); tag = a.tag
    M = H.Models(dev)
    for name, arch in (('part_kin', 'part'), ('pnet_kin', 'pn')): M.add_weaver(name, arch, False)
    M.add_lnet()
    for kind in ('pfn', 'efn'):
        H14.add_keras_ef(M, kind)
        net = H.make_ef(kind); net.load_state_dict(torch.load(V / 'models' / 'qg' / f'h5_{kind}_torch.pt', map_location='cpu'))
        M.add_torch(f'{kind}_t', net, json.loads((V / 'models' / 'qg' / f'h5_{kind}_torch.json').read_text()))
    for name, arch in (('part_kin', 'part'), ('pnet_kin', 'pn')):
        H14.add_weaver_dir(M, 'hw_' + name, arch, False, V / 'models' / 'qg_herwig' / name)
    M.add_permuted('part_kin_perm', 'part_kin'); M.add_permuted('pnet_kin_perm', 'pnet_kin')
    names = [n for n in NETS if n in M.m]; log('models', names, 'device', dev)
    T, lrinfo = H14.lrn_tables(); refs = {g: train_refs(g, a.train_per_class) for g in ('py', 'hw')}
    summary = dict(models=M.info, lrn=lrinfo, seed=a.seed, per_class=a.per_class, gens={})
    for gen in a.gens.split(','):
        L0, y, idx, f_lt8 = load_sample(gen, a.per_class, a.seed)
        log(f'{gen}: {len(y)} jets (N >= 8; dropped share {f_lt8:.4f}), <N> q {np.mean([len(x) for x, c in zip(L0, y) if c == 1]):.2f} '
            f'g {np.mean([len(x) for x, c in zip(L0, y) if c == 0]):.2f}')
        E, kN = make_edits(L0, a.seed, gen, dev); O0 = obs_table(L0)
        out = dict(y=y, idx=idx, kN=kN); G = summary['gens'][gen] = dict(n=int(len(y)), dropped_lt8=f_lt8, conds={}, base={}, c2st={})
        conds = [('base', L0, np.ones(len(L0), bool), {})] + [(k, *v) for k, v in E.items()]
        for cn, Le, ok, info in conds:
            Jt = H.from_list(Le, Lpad=max(len(x) for x in Le))
            for nm in names: out[f'{cn}|{nm}'] = M.logit(nm, Jt).astype(np.float32)
            O = O0 if cn == 'base' else obs_table(Le)
            for tg in ('py', 'hw'):
                for k, v in ref_scores(refs[tg], O, T, tg).items(): out[f'{cn}|{k}_{tg}'] = v.astype(np.float32)
            for k in ('N', 'l_ptd', 'l_lha', 'l_wid', 'l_mass', 'mass', 'jpt'): out[f'obs|{cn}|{k}'] = O[k].astype(np.float32)
            if cn == 'base':
                for nm in names + [k.split('|')[1] for k in out if k.startswith('base|') and k.split('|')[1] not in names]:
                    G['base'][nm] = H.roc_metrics(out[f'base|{nm}'].astype(float), y)
                log(gen, 'unedited', '; '.join(f'{nm} AUC {v["AUC"]:.4f} acc {v["acc"]:.4f}' for nm, v in G['base'].items()))
                continue
            dv, ach = tolerance(O0, O, cn); ok2 = ok & (dv < TOL)
            tg_, fr_ = parse(cn)
            if tg_ in IRCS: ok2 &= np.abs(np.abs(ach) - fr_) < TOL * fr_ / 0.1
            elif cn in ('N+', 'N-'): ok2 &= np.abs(np.abs(ach) - kN / O0['N']) < 1e-9
            out[f'ok|{cn}'] = ok2; out[f'dev|{cn}'] = dv.astype(np.float32); out[f'ach|{cn}'] = ach.astype(np.float32)
            for k, v in info.items(): out[f'{k}|{cn}'] = v.astype(np.float32)
            G['conds'][cn] = dict(solver_ok=float(ok.mean()), ok=float(ok2.mean()), dev_median=float(np.median(dv)),
                                  dev_p99=float(np.percentile(dv, 99)), achieved_mean=float(ach[ok2].mean()) if ok2.any() else None,
                                  dnorm_median=float(np.median(info['dnorm'])) if 'dnorm' in info else None,
                                  dmass_rel=float(np.mean(O['mass'][ok2] / O0['mass'][ok2] - 1)) if ok2.any() else None)
            log(gen, cn, json.dumps(G['conds'][cn]), '|', ' '.join(f'{nm} {np.mean((out[f"{cn}|{nm}"] - out[f"base|{nm}"])[ok2]):+.4f}'
                                                                     for nm in names))
        rs = np.random.default_rng(5); sub = np.sort(rs.choice(len(L0), min(a.c2st_n, len(L0)), replace=False))
        Ls0 = [L0[i] for i in sub]; Os0 = {k: v[sub] for k, v in O0.items()}; G['c2st']['none'] = c2st(Ls0, Ls0, Os0, Os0)
        for cn in ['lha+', 'wid+', 'ptd+', 'mass+', 'lha-', 'wid-', 'ptd-', 'mass-', 'lha+10', 'wid+10', 'ptd+10', 'mass+10', 'N+', 'N-', 'Nsp', 'wid_r+', 'ptd_r+', 'mass_rot', 'wid_rot']:
            Le = [E[cn][0][i] for i in sub]; G['c2st'][cn] = c2st(Ls0, Le, Os0, obs_table(Le)); log('c2st', gen, cn, G['c2st'][cn])
        np.savez_compressed(RES / f'h76_{gen}{tag}.npz', **out); log('wrote', RES / f'h76_{gen}{tag}.npz')
        # one gluon jet before / after (for the figure)
        j = int(np.flatnonzero((y == 0) & E['wid+10'][1] & E['ptd+10'][1])[0])
        np.savez_compressed(RES / f'h76_example_{gen}{tag}.npz', base=L0[j], **{c: E[c][0][j] for c in ('wid+10', 'ptd+10', 'wid+', 'N+', 'wid_r+')})
    (RES / f'h76_run{tag}.json').write_text(json.dumps(summary, indent=1, default=str)); log('run done')


# ------------------------------------------------------------------------------------------------------------------ stats
def _boot(d, idx):
    return np.array([d[ix].mean() for ix in idx]) if d.ndim == 1 else np.stack([d[ix].mean(0) for ix in idx])


def _signflip(D, P, seed=0):
    """sign-flip p for mean(D) = 0; D (J,) or (J, k) (statistic: norm of the mean)"""
    r = np.random.default_rng(seed); D = D if D.ndim == 2 else D[:, None]; obs = np.linalg.norm(D.mean(0)); cnt = 0
    for c in range(0, P, 500):
        sg = r.choice(np.array([-1.0, 1.0]), (min(500, P - c), len(D)))
        cnt += int((np.linalg.norm(sg @ D / len(D), axis=1) >= obs - 1e-15).sum())
    return (cnt + 1) / (P + 1)


def _holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, min(1, (m - r) * ps[i])); adj[i] = run
    return adj


def slopes(Z, model):
    """per jet: dict direction -> (effect d, control c, ok), units gluon log-odds per +10 % move"""
    s0 = Z[f'base|{model}']; out = {}
    N0 = Z['obs|base|N']; kN = Z['kN']
    for t in DIRS:
        sc = 0.1 / (kN / N0) if t == 'N' else 0.1 / FRAC
        d = (Z[f'{t}+|{model}'] - Z[f'{t}-|{model}']) / 2 * sc
        c = (Z[f'{t}_r+|{model}'] - Z[f'{t}_r-|{model}']) / 2 * sc
        ok = Z[f'ok|{t}+'] & Z[f'ok|{t}-'] & Z[f'ok|{t}_r+'] & Z[f'ok|{t}_r-']
        u = dict(abs_edit=np.abs(Z[f'{t}+|{model}'] - s0), abs_crand=np.abs(Z[f'{t}_r+|{model}'] - s0))
        if t != 'N':
            ok = ok & Z[f'ok|{t}_rot']; u['abs_crot'] = np.abs(Z[f'{t}_rot|{model}'] - s0); u['crot'] = Z[f'{t}_rot|{model}'] - s0
        if t != 'N':
            u['d10'] = (Z[f'{t}+10|{model}'] - Z[f'{t}-10|{model}']) / 2; u['ok10'] = Z[f'ok|{t}+10'] & Z[f'ok|{t}-10']
        out[t] = (d.astype(float), c.astype(float), ok, u)
    return out


def step_stats(a):
    tag = a.tag; S = dict(B=a.boot, P=a.perm, gens={}); allp = []
    for gen in a.gens.split(','):
        Z = dict(np.load(RES / f'h76_{gen}{tag}.npz')); y = Z['y'].astype(int); J = len(y)
        models = sorted({k.split('|')[1] for k in Z if k.startswith('base|')})
        rng = np.random.default_rng(7601); idx = rng.integers(0, J, (a.boot, J))
        okall = np.all([Z[k] for k in Z if k.startswith('ok|') and not k.endswith('10')], 0)
        G = S['gens'][gen] = dict(n=J, ok_all=float(okall.mean()), models={})
        vec = {}
        for m in models:
            sl = slopes(Z, m); R = G['models'][m] = {}
            E = np.stack([sl[t][0] - sl[t][1] for t in DIRS], 1)
            for ti, t in enumerate(DIRS):
                d, c, ok, u = sl[t]; ok = ok & okall
                e = d - c; bd = _boot(np.where(ok, d, 0), idx) / np.maximum(_boot(ok.astype(float), idx), 1e-9)
                bc = _boot(np.where(ok, c, 0), idx) / np.maximum(_boot(ok.astype(float), idx), 1e-9); be = bd - bc
                R[t] = dict(edit=float(d[ok].mean()), edit_ci=np.percentile(bd, [2.5, 97.5]).tolist(),
                            crand=float(c[ok].mean()), crand_ci=np.percentile(bc, [2.5, 97.5]).tolist(),
                            effect=float(e[ok].mean()), effect_ci=np.percentile(be, [2.5, 97.5]).tolist(),
                            p=_signflip(e[ok], a.perm, seed=ti) if m in NETS else None,
                            effect_q=float(e[ok & (y == 1)].mean()), effect_g=float(e[ok & (y == 0)].mean()),
                            abs_edit=float(u['abs_edit'][ok].mean()), abs_crand=float(u['abs_crand'][ok].mean()),
                            n_ok=int(ok.sum()))
                if 'd10' in u:
                    o10 = ok & u['ok10']
                    R[t].update(ok10_frac=float(u['ok10'][okall].mean()), edit10=float(u['d10'][o10].mean()) if o10.any() else None,
                                edit3_on10=float(d[o10].mean()) if o10.any() else None)
                if 'abs_crot' in u: R[t].update(abs_crot=float(u['abs_crot'][ok].mean()), crot=float(u['crot'][ok].mean()))
                if m in NETS: allp.append((gen, m, t, R[t]['p']))
            vec[m] = (E, okall)
            g = E[okall].mean(0); R['g'] = g.tolist(); R['g_norm'] = float(np.linalg.norm(g))
            R['g_unit'] = (g / max(np.linalg.norm(g), 1e-12)).tolist()
            R['g_unit_q'] = (lambda v: (v / max(np.linalg.norm(v), 1e-12)).tolist())(E[okall & (y == 1)].mean(0))
            R['g_unit_g'] = (lambda v: (v / max(np.linalg.norm(v), 1e-12)).tolist())(E[okall & (y == 0)].mean(0))
        # cosine similarity with references (bootstrap over jets, jointly), H0: unit vector = that of g5 (sign-flip on per-jet
        # differences of the vectors scaled by each model's own mean norm)
        okj = okall; Eix = lambda E, ix: E[ix][okj[ix]].mean(0)
        cos = lambda u, v: float(u @ v / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-12))
        refs = [f'{r}_{gen}' for r in ['g5'] + [f'g1_{t}' for t in DIRS] + ['lrn']] + ([f'g5_{"hw" if gen == "py" else "py"}'])
        G['cos'] = {}; G['h0_g5'] = {}
        for m in NETS:
            if m not in vec: continue
            E = vec[m][0]; G['cos'][m] = {}
            for r in refs:
                if r not in vec: continue
                Er = vec[r][0]; bs = [cos(Eix(E, ix), Eix(Er, ix)) for ix in idx[:a.boot_cos]]
                G['cos'][m][r] = dict(cos=cos(E[okj].mean(0), Er[okj].mean(0)), ci=np.percentile(bs, [2.5, 97.5]).tolist())
            for r in (f'g5_{gen}', f'g5_{"py" if gen == "hw" else "hw"}'):
                Er = vec[r][0]; un = E[okj] / max(np.linalg.norm(E[okj].mean(0)), 1e-12); ur = Er[okj] / max(np.linalg.norm(Er[okj].mean(0)), 1e-12)
                p = _signflip(un - ur, a.perm, seed=11); G['h0_g5'][f'{m}|{r}'] = dict(dist=float(np.linalg.norm((un - ur).mean(0))), p=p)
                if r == f'g5_{gen}': allp.append((gen, m, 'h0_g5', p))
        # generator dependence of the N slope: Herwig-trained minus Pythia-trained (paired over jets)
        G['gen_dep'] = {}
        for hw, py in (('hw_part_kin', 'part_kin'), ('hw_pnet_kin', 'pnet_kin')):
            if hw not in vec or py not in vec: continue
            for ti, t in enumerate(DIRS):
                D = vec[hw][0][okj, ti] - vec[py][0][okj, ti]; b = _boot(np.where(okj, vec[hw][0][:, ti] - vec[py][0][:, ti], 0), idx) / np.maximum(_boot(okj.astype(float), idx), 1e-9)
                G['gen_dep'][f'{hw}-{py}|{t}'] = dict(diff=float(D.mean()), ci=np.percentile(b, [2.5, 97.5]).tolist(), p=_signflip(D, a.perm, seed=21 + ti))
                allp.append((gen, f'{hw}-{py}', t, G['gen_dep'][f'{hw}-{py}|{t}']['p']))
        log(gen, 'done', {m: [round(v, 4) for v in G['models'][m]['g']] for m in NETS if m in G['models']})
    adj = _holm([p for *_, p in allp]); S['holm'] = [dict(gen=g, model=m, test=t, p=p, p_holm=float(q)) for (g, m, t, p), q in zip(allp, adj)]
    S['n_tests'] = len(allp); S['n_holm_005'] = int((adj < 0.05).sum())
    (RES / f'h76_stats{tag}.json').write_text(json.dumps(S, indent=1)); log('wrote', RES / f'h76_stats{tag}.json', 'Holm<0.05:', S['n_holm_005'], '/', len(allp))


# -------------------------------------------------------------------------------------------------------------------- fig
def step_fig(a):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    tag = a.tag; S = json.loads((RES / f'h76_stats{tag}.json').read_text()); X = dict(np.load(RES / f'h76_example_py{tag}.npz'))
    fig = plt.figure(figsize=(16, 7)); gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25])
    def rel(L, ax0):
        O = angs(L); return L[:, 1] - O['jy'], (L[:, 2] - O['jphi'] + np.pi) % (2 * np.pi) - np.pi, O
    for k, (c, ttl) in enumerate((('wid+10', 'width λ(1,1) +10 %'), ('ptd+10', '(pT^D)² = λ(2,0) +10 %'))):
        ax = fig.add_subplot(gs[0, k]); x0, y0, O0 = rel(X['base'], None); x1, y1, O1 = rel(X[c], None)
        ax.scatter(x0, y0, s=4 + 300 * X['base'][:, 0] / X['base'][:, 0].sum(), facecolors='none', edgecolors='0.5', lw=0.8, label='unedited')
        ax.scatter(x1, y1, s=4 + 300 * X[c][:, 0] / X[c][:, 0].sum(), color='C3', alpha=0.6, lw=0, label='edited')
        for i in range(len(x0)): ax.plot([x0[i], x1[i]], [y0[i], y1[i]], color='C3', lw=0.6)
        ax.add_patch(plt.Circle((0, 0), R0, fill=False, ls=':', color='0.6'))
        ax.set_xlim(-0.42, 0.42); ax.set_ylim(-0.42, 0.42); ax.set_aspect('equal')
        ax.set_xlabel('Δy from jet axis'); ax.set_ylabel('Δφ from jet axis')
        ax.set_title(f'gluon jet (Pythia), {ttl}\nN {int(O0["N"])}; width {O0["l_wid"]:.3f}→{O1["l_wid"]:.3f}; '
                     f'(pT^D)² {O0["l_ptd"]:.3f}→{O1["l_ptd"]:.3f}\nLHA {O0["l_lha"]:.3f}→{O1["l_lha"]:.3f}; '
                     f'λ(1,2) {O0["l_mass"]:.4f}→{O1["l_mass"]:.4f}', fontsize=8.5)
        ax.legend(fontsize=7, loc='lower right', title='marker area ∝ pT', title_fontsize=7)
    ax = fig.add_subplot(gs[0, 2]); rows = []
    lab = dict(part_kin='ParT_kin (Py)', pnet_kin='PN_kin (Py)', lorentznet='LorentzNet (Py)', pfn_k='PFN (Py)', efn_k='EFN (Py)',
               hw_part_kin='ParT_kin (Hw)', hw_pnet_kin='PN_kin (Hw)', part_kin_perm='ParT_kin permuted', g5_py='GBDT 5 angularities (Py)',
               g5_hw='GBDT 5 angularities (Hw)', lrn_py='LR(N) (Py)')
    for gen in ('py', 'hw'):
        G = S['gens'].get(gen)
        if not G: continue
        for m in lab:
            if m in G['models']:
                rows.append((f'{lab[m]} | {gen} jets', G['models'][m]['g_unit'], G['models'][m]['g_norm']))
    M = np.array([r[1] for r in rows]); im = ax.imshow(M, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([f'{r[0]}  |g|={r[2]:.2f}' for r in rows], fontsize=7)
    ax.set_xticks(range(5)); ax.set_xticklabels(['N (0,0)', 'LHA (1,½)', 'width (1,1)', '(pT^D)² (2,0)', 'λ(1,2)'], fontsize=8, rotation=20)
    for i in range(M.shape[0]):
        for j in range(5): ax.text(j, i, f'{M[i, j]:+.2f}', ha='center', va='center', fontsize=6)
    ax.set_title('unit response vector g/|g|: gluon log-odds per +10 % move\nof one angularity at fixed others, minus random null edit', fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout(); p = RES / f'h76_angularity{tag}.png'; fig.savefig(p, dpi=130); log('wrote', p)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', required=True, choices=['run', 'stats', 'fig'])
    ap.add_argument('--gens', default='py,hw'); ap.add_argument('--per_class', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=7601); ap.add_argument('--tag', default='')
    ap.add_argument('--train_per_class', type=int, default=30000); ap.add_argument('--c2st_n', type=int, default=8000)
    ap.add_argument('--boot', type=int, default=1000); ap.add_argument('--boot_cos', type=int, default=500)
    ap.add_argument('--perm', type=int, default=20000)
    a = ap.parse_args(); RES.mkdir(parents=True, exist_ok=True)
    dict(run=step_run, stats=step_stats, fig=step_fig)[a.step](a)
