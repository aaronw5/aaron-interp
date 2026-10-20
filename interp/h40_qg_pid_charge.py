"""H40 (loop D; new claim D16, exploratory): do QG-trained ParT and ParticleNet with particle-ID inputs (kinpid) use
particle-species information beyond kinematics: jet charge Q_kappa (with its sign), the charged-particle fraction, kaon/baryon
content? Every edit changes only PDG ids; (pT, y, phi) of every particle and N are unchanged, so every kinematic and IRC-safe
observable is fixed by construction.

Samples (random rows, both classes)
  py  interp.data_qg.sample(per_class=10000, seed=4001) (Komiske-Metodiev-Thaler Pythia 8 test split) via h14.load_sample('py')
  hw  Herwig 7.1 test jets (Zenodo 3066475 files 18-19), 10000 per class, seed 4001, via h14.load_sample('hw')
Models (score s = log p(q) - log p(g); positive = more quark-like)
  part_kinpid, pnet_kinpid   H2 weaver trainings (Pythia, final). Their PID inputs ('exp' set): charge, isChargedHadron
                             (|pid| in 211, 321, 2212), isNeutralHadron (130, 2112), isPhoton, isElectron, isMuon; massless
                             four-vectors. K+-/p -> pi+- at the same charge is therefore not visible to them (E4 zero by construction).
  lorentznet                 released QG LorentzNet: one-hot of |pid| over (11, 13, 22, 130, 211, 321, 2112, 2212) and PDG masses;
                             charge-blind (E1, E2 zero by construction), sees species (E4).
  part_kin, pnet_kin, efn_k  PID-blind (zero by construction for every edit; checked)
  part_kinpid_perm           part_kinpid with every weight tensor's entries permuted (null band)
References (trained on 50,000 + 50,000 random non-val rows of the KMT train split, scored on the same sample jets)
  LR_Q03, LR_Q05 (signed Q_kappa, kappa 0.3 / 0.5), LR_absQ03, LR_fch (n_ch / N), LR_N,
  GB_kin  = GBDT(N, log girth, log C1^{0.5,1,2}, pT^D, mass)        (stand-in for GBDT(EFP+N): IRC-safe shapes + N)
  GB_sp   = GB_kin + species counts (pi, K, p, gamma, K0L, n, e, mu, n+, n-)
  GB_spQ  = GB_sp + Q_0.3, Q_0.5
Q_kappa = sum_i q_i (pT_i / pT_J)^kappa, pT_J = |sum of particle pT vectors|.
Edits (per jet; rng per (generator, edit))
  E1      charge conjugation: pid -> -pid for every particle except photons (22) and K0L (130)
  E2k     k = 1, 2, 3 hardest charged particles: pid -> -pid (moves Q_kappa by -2 sum q_i z_i^kappa)
  C2      control: one + and one - charged particle flipped together, |Delta Q_kappa| < 0.01 for kappa 0.3 and 0.5, the pair
          with the highest softer pT among the 10 hardest of each sign
  E3g_m / E3n_m   m = 2, 4, 6 pi+- with pT below the jet's median particle pT (m/2 pi+, m/2 pi-, random) -> photon / K0L
  C3_m    control: the same particles pi+ <-> pi- (n_ch, total charge fixed)
  E3r_m   reverse: m photons below the median -> pi+ (m/2) and pi- (m/2);  C3r_m: the same photons -> K0L
  E4a     every K+- and p/pbar -> pi+- of the same charge;  E4b: m_j random pi+- -> K+- of the same charge (m_j = the jet's
          K+-/p count);  E4n: every n/nbar -> K0L;  C4: no-op relabel (pid unchanged, same code path)
  E5sh    PDG ids randomly permuted among the jet's particles (species counts and total charge fixed)
  gam2K0L_m  the C3r_m relabel (m soft photons -> K0L) read as its own edit against the unedited jet
Metrics: per-jet Delta s(edit) - Delta s(control), mean with 95 % bootstrap CI (B = 1000), sign-flip permutation p (P = 2000),
  Holm over contrasts x {part_kinpid, pnet_kinpid, lorentznet, part_kinpid_perm}; McNemar edit vs control calls at the threshold
  giving quark efficiency 0.5 on unedited jets (= the published R50 working point definition); E1 also as Delta s * sign(Q_0.3)
  and split by class and sign of Q; E2 slope dDelta s / dDelta Q_kappa vs the LR(Q_kappa) coefficient; E3 slope per unit
  charged fraction.
Steps
  --step run   (GPU, part_job.sh) references, models, edits, statistics -> $AI_RESULTS/physics/h40_qg_pid_charge.json,
               h40_perjet_<gen>.npz
  --step fig   (CPU, jax env) results/physics/h40_pid_edits.png
  --step pz    (CPU, jax env) normal-approximation p and Holm added to the json (permutation p floor 5e-4)
  --smoke N    N jets per class, files with suffix _smoke
"""
import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
V = Path('/j-jepa-vol/aaron-interp')
RES = Path(os.environ.get('AI_RESULTS', V / 'results')) / 'physics'
QG = V / 'data' / 'qg'
SEED = 4001
MS = (2, 4, 6)
NETS = ['part_kinpid', 'pnet_kinpid', 'lorentznet', 'part_kinpid_perm', 'part_kin', 'pnet_kin', 'efn_k']
TESTED = ['part_kinpid', 'pnet_kinpid', 'lorentznet', 'part_kinpid_perm']
ZEROS = ['part_kin', 'pnet_kin', 'efn_k']
REFS = ['LR_Q03', 'LR_Q05', 'LR_absQ03', 'LR_fch', 'LR_N', 'GB_kin', 'GB_sp', 'GB_spQ']
KNOWN = dict(py=dict(part_kinpid='AUC 0.9123 (test), 0.9121 published', pnet_kinpid='AUC 0.906 (H5v), 0.9116 published',
                     part_kin='AUC 0.9013', pnet_kin='AUC 0.9015', lorentznet='AUC 0.917 (H5v)', efn_k='AUC 0.8835 (H14)'),
             hw=dict(part_kinpid='AUC 0.8049 (H14)', part_kin='AUC 0.8079 (H14)', pnet_kin='AUC 0.8080 (H14)', efn_k='AUC 0.8023 (H14)'))

CHG = {211: 1, 321: 1, 2212: 1, -211: -1, -321: -1, -2212: -1, 11: -1, -11: 1, 13: -1, -13: 1}


def charge(pid):
    pid = np.asarray(pid).astype(np.int64); q = np.zeros(pid.shape)
    for k, v in CHG.items(): q[pid == k] = v
    return q


# ================================================================================================================ features
def species_feats(pt, pid, real):
    """(J, L) arrays -> dict per jet: N, counts per species, n+, n-, n_ch, f_ch, Q_0.3, Q_0.5"""
    ap = np.abs(pid); q = charge(pid) * real
    N = real.sum(1).astype(float)
    out = dict(N=N)
    for nm, codes in (('pi', (211,)), ('K', (321,)), ('p', (2212,)), ('gam', (22,)), ('K0L', (130,)), ('n', (2112,)),
                      ('e', (11,)), ('mu', (13,))):
        out['n_' + nm] = (np.isin(ap, codes) & real).sum(1).astype(float)
    out['n_plus'] = (q > 0).sum(1).astype(float); out['n_minus'] = (q < 0).sum(1).astype(float)
    out['n_ch'] = out['n_plus'] + out['n_minus']; out['fch'] = out['n_ch'] / np.maximum(N, 1)
    return out


def jet_pt(Jt):
    pt, phi = Jt['pt'], Jt['phi']
    return np.hypot((pt * np.cos(phi)).sum(1), (pt * np.sin(phi)).sum(1))


def qkappa(pt, pid, ptj, kappa):
    z = pt / ptj[:, None]
    return (charge(pid) * np.where(pt > 0, z, 0) ** kappa).sum(1)


def all_feats(Jt, shp=None):
    from interp import h5_qg_counting as H
    pt, pid = Jt['pt'], Jt['pid']; real = pt > 0
    F = species_feats(pt, pid, real); ptj = jet_pt(Jt)
    F['Q03'] = qkappa(pt, pid, ptj, 0.3); F['Q05'] = qkappa(pt, pid, ptj, 0.5)
    if shp is None:
        O = H.shapes(Jt); shp = dict(girth=O['girth'], C1_05=O['C1_0.5'], C1_1=O['C1_1.0'], C1_2=O['C1_2.0'], ptd=O['ptd'], mass=O['mass'])
    F.update(shp)
    return F, shp


SP = ['n_pi', 'n_K', 'n_p', 'n_gam', 'n_K0L', 'n_n', 'n_e', 'n_mu', 'n_plus', 'n_minus']


def ref_X(F, name):
    lg = lambda k: np.log(np.maximum(F[k], 1e-8))
    kin = [F['N'], lg('girth'), lg('C1_05'), lg('C1_1'), lg('C1_2'), F['ptd'], F['mass']]
    if name == 'LR_Q03': X = [F['Q03']]
    elif name == 'LR_Q05': X = [F['Q05']]
    elif name == 'LR_absQ03': X = [np.abs(F['Q03'])]
    elif name == 'LR_fch': X = [F['fch']]
    elif name == 'LR_N': X = [F['N']]
    elif name == 'GB_kin': X = kin
    elif name == 'GB_sp': X = kin + [F[k] for k in SP]
    elif name == 'GB_spQ': X = kin + [F[k] for k in SP] + [F['Q03'], F['Q05']]
    return np.stack(X, 1)


def train_refs(n_per_class=50000, seed=SEED + 7):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    d = QG / 'train'; y = np.load(d / 'label.npy'); isv = np.load(d / 'is_val.npy'); rng = np.random.default_rng(seed)
    ix = np.sort(np.concatenate([rng.choice(np.flatnonzero((y == c) & ~isv), n_per_class, replace=False) for c in (0, 1)]))
    k3 = np.load(d / 'kin3.npy')[ix].astype(np.float64)              # sequential full read (random-row mmap reads are slow on CephFS)
    pid = np.load(d / 'pid.npy')[ix][:, :k3.shape[1]].astype(np.int64)
    real = k3[..., 0] > 0
    # angles relative to the jet axis serve as (eta, phi): every feature is invariant under the longitudinal boost / rotation
    Jt = dict(pt=k3[..., 0] * real, eta=k3[..., 1] * real, phi=k3[..., 2] * real, pid=pid * real, n=real.sum(1))
    F, _ = all_feats(Jt); yy = y[ix].astype(int)
    log(f'refs: {len(ix)} train jets; <Q03> q {F["Q03"][yy == 1].mean():+.3f} g {F["Q03"][yy == 0].mean():+.3f}; '
        f'<fch> q {F["fch"][yy == 1].mean():.3f} g {F["fch"][yy == 0].mean():.3f}')
    R, info = {}, {}
    for nm in REFS:
        X = ref_X(F, nm)
        if nm.startswith('LR'):
            mu, sd = X.mean(0), X.std(0) + 1e-12
            m = LogisticRegression(max_iter=2000, C=1e3).fit((X - mu) / sd, yy)
            R[nm] = ('lr', m, mu, sd); info[nm] = dict(coef_per_unit=(m.coef_[0] / sd).tolist(), intercept=float(m.intercept_[0] - (m.coef_[0] * mu / sd).sum()))
        else:
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0).fit(X, yy)
            R[nm] = ('gb', m, None, None); info[nm] = dict(n_features=X.shape[1])
        log(f'  {nm} trained', info[nm])
    return R, dict(n_train=int(len(ix)), seed=seed, per_ref=info)


def ref_scores(R, F):
    out = {}
    for nm, (kind, m, mu, sd) in R.items():
        X = ref_X(F, nm)
        if kind == 'lr': out[nm] = m.decision_function((X - mu) / sd)
        else:
            p = np.clip(m.predict_proba(X)[:, 1], 1e-7, 1 - 1e-7); out[nm] = np.log(p) - np.log(1 - p)
    return out


# =================================================================================================================== edits
def _rng(gen, name):
    return np.random.default_rng([SEED, zlib.crc32(gen.encode()), zlib.crc32(name.encode())])


def _below_med(a):
    return a[:, 0] < np.median(a[:, 0])


def make_edits(L0, gen, full=True):
    """-> dict name -> (list of edited (n, 4) arrays, applied mask (J,), per-jet number of relabelled particles)"""
    J = len(L0); E = {}

    def new(name):
        return [a.copy() for a in L0], np.zeros(J, bool), np.zeros(J)

    # E1
    L, ok, nr = new('E1')
    for j, a in enumerate(L):
        pid = a[:, 3].astype(int); s = (pid != 22) & (pid != 130) & (pid != 0)
        a[s, 3] = -pid[s]; ok[j] = s.any(); nr[j] = s.sum()
    E['E1'] = (L, ok, nr)
    # E2k and C2
    ks = (1, 2, 3) if full else (1,)
    for k in ks:
        L, ok, nr = new(f'E2k{k}')
        for j, a in enumerate(L):
            ch = np.flatnonzero(charge(a[:, 3]) != 0)
            if len(ch) >= k: a[ch[:k], 3] *= -1; ok[j] = True; nr[j] = k
        E[f'E2k{k}'] = (L, ok, nr)
    L, ok, nr = new('C2')
    for j, a in enumerate(L):
        q = charge(a[:, 3]); z = a[:, 0] / np.hypot((a[:, 0] * np.cos(a[:, 2])).sum(), (a[:, 0] * np.sin(a[:, 2])).sum())
        P = np.flatnonzero(q > 0)[:10]; M = np.flatnonzero(q < 0)[:10]
        if len(P) == 0 or len(M) == 0: continue
        d3 = 2 * np.abs(z[M][None, :] ** 0.3 - z[P][:, None] ** 0.3); d5 = 2 * np.abs(z[M][None, :] ** 0.5 - z[P][:, None] ** 0.5)
        good = (d3 < 0.01) & (d5 < 0.01)
        if not good.any(): continue
        soft = np.minimum(a[P, 0][:, None], a[M, 0][None, :]); soft = np.where(good, soft, -1)
        i, m = np.unravel_index(np.argmax(soft), soft.shape)
        a[[P[i], M[m]], 3] *= -1; ok[j] = True; nr[j] = 2
    E['C2'] = (L, ok, nr)
    # E3 forward (charged -> neutral) and pair-flip control, same selected pions
    for m in (MS if full else (4,)):
        rng = _rng(gen, f'E3_{m}'); sel = []
        for a in L0:
            pid = a[:, 3].astype(int); bm = _below_med(a)
            pp = np.flatnonzero((pid == 211) & bm); pm = np.flatnonzero((pid == -211) & bm)
            sel.append(np.r_[rng.choice(pp, m // 2, replace=False), rng.choice(pm, m // 2, replace=False)].astype(int)
                       if len(pp) >= m // 2 and len(pm) >= m // 2 else None)
        for nm, fn in ((f'E3g_{m}', lambda p: np.full(len(p), 22)), (f'E3n_{m}', lambda p: np.full(len(p), 130)), (f'C3_{m}', lambda p: -p)):
            L, ok, nr = new(nm)
            for j, a in enumerate(L):
                if sel[j] is None: continue
                a[sel[j], 3] = fn(a[sel[j], 3]); ok[j] = True; nr[j] = m
            E[nm] = (L, ok, nr)
        rng = _rng(gen, f'E3r_{m}'); sel = []
        for a in L0:
            g = np.flatnonzero((a[:, 3].astype(int) == 22) & _below_med(a))
            sel.append(rng.choice(g, m, replace=False).astype(int) if len(g) >= m else None)
        for nm, fn in ((f'E3r_{m}', lambda p: np.r_[np.full(len(p) // 2, 211), np.full(len(p) - len(p) // 2, -211)]),
                       (f'C3r_{m}', lambda p: np.full(len(p), 130))):
            L, ok, nr = new(nm)
            for j, a in enumerate(L):
                if sel[j] is None: continue
                a[sel[j], 3] = fn(a[sel[j], 3]); ok[j] = True; nr[j] = m
            E[nm] = (L, ok, nr)
    # E4
    L, ok, nr = new('E4a')
    for j, a in enumerate(L):
        pid = a[:, 3].astype(int); s = np.isin(np.abs(pid), (321, 2212))
        a[s, 3] = np.sign(pid[s]) * 211; ok[j] = s.any(); nr[j] = s.sum()
    E['E4a'] = (L, ok, nr)
    rng = _rng(gen, 'E4b'); L, ok, nr = new('E4b')
    for j, a in enumerate(L):
        pid = a[:, 3].astype(int); m = int(np.isin(np.abs(pid), (321, 2212)).sum()); pi = np.flatnonzero(np.abs(pid) == 211)
        if m == 0 or len(pi) < m: continue
        s = rng.choice(pi, m, replace=False); a[s, 3] = np.sign(pid[s]) * 321; ok[j] = True; nr[j] = m
    E['E4b'] = (L, ok, nr)
    L, ok, nr = new('E4n')
    for j, a in enumerate(L):
        s = np.abs(a[:, 3].astype(int)) == 2112; a[s, 3] = 130; ok[j] = s.any(); nr[j] = s.sum()
    E['E4n'] = (L, ok, nr)
    L, ok, nr = new('C4')
    for j, a in enumerate(L):
        a[:, 3] = a[:, 3].astype(int).astype(float); ok[j] = True
    E['C4'] = (L, ok, nr)
    if full:
        rng = _rng(gen, 'E5sh'); L, ok, nr = new('E5sh')
        for j, a in enumerate(L):
            a[:, 3] = a[rng.permutation(len(a)), 3]; ok[j] = len(np.unique(a[:, 3])) > 1; nr[j] = len(a)
        E['E5sh'] = (L, ok, nr)
    return E


def contrasts(full=True):
    """(name, edit, control, kind); control None = compare with unedited (control Delta 0)"""
    C = [('E1', 'E1', None, 'mean'), ('E1_signQ', 'E1', None, 'signQ')]
    for k in ((1, 2, 3) if full else (1,)): C.append((f'E2k{k}-C2', f'E2k{k}', 'C2', 'mean'))
    for m in (MS if full else (4,)):
        C += [(f'E3g_{m}-C3', f'E3g_{m}', f'C3_{m}', 'mean'), (f'E3n_{m}-C3', f'E3n_{m}', f'C3_{m}', 'mean'),
              (f'E3r_{m}-C3r', f'E3r_{m}', f'C3r_{m}', 'mean'), (f'E3r_{m}', f'E3r_{m}', None, 'mean'),
              (f'gam2K0L_{m}', f'C3r_{m}', None, 'mean')]
    C += [('E4a-C4', 'E4a', 'C4', 'mean'), ('E4b-C4', 'E4b', 'C4', 'mean'), ('E4n-C4', 'E4n', 'C4', 'mean')]
    if full: C.append(('E5sh', 'E5sh', None, 'mean'))
    return C


# ============================================================================================================== statistics
class Boot:
    def __init__(self, B=1000, P=2000, seed=SEED):
        self.B, self.P, self.seed = B, P, seed

    def ci(self, d):
        rng = np.random.default_rng(self.seed); n = len(d); s = []
        for b in range(0, self.B, 100):
            s.append(d[rng.integers(0, n, (min(100, self.B - b), n))].mean(1))
        s = np.concatenate(s); return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]

    def p(self, d):
        if np.all(d == 0): return 1.0
        rng = np.random.default_rng(self.seed + 1); obs = abs(d.mean()); cnt = 0
        for s in range(0, self.P, 250):
            sg = rng.choice([-1.0, 1.0], (min(250, self.P - s), len(d)))
            cnt += int((np.abs(sg @ d / len(d)) >= obs - 1e-12).sum())
        return (cnt + 1) / (self.P + 1)


def mcnemar(a, b):
    from scipy.stats import binomtest
    n01 = int((~a & b).sum()); n10 = int((a & ~b).sum())
    return n01, n10, float(binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, min(1, (m - r) * ps[i])); adj[i] = run
    return adj


def roc(s, y):
    from sklearn.metrics import roc_auc_score
    thr = np.quantile(s[y == 1], 0.5); eg = np.mean(s[y == 0] >= thr)
    return dict(AUC=float(roc_auc_score(y, s)), acc=float(((s > 0).astype(int) == y).mean()), R50=float(1 / max(eg, 1e-9)), thr50=float(thr))


def slope_ci(x, d, B=1000, seed=SEED):
    """OLS slope through the origin of d on x, bootstrap over jets"""
    if np.sum(x * x) == 0: return dict(slope=0.0, ci=[0.0, 0.0])
    sl = float((x * d).sum() / (x * x).sum()); rng = np.random.default_rng(seed); n = len(x); v = []
    for _ in range(B):
        ix = rng.integers(0, n, n); xx = x[ix]; v.append((xx * d[ix]).sum() / max((xx * xx).sum(), 1e-12))
    return dict(slope=sl, ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])


# ===================================================================================================================== run
def load_models(dev):
    from interp import h5_qg_counting as H
    from interp import h14_qg_generator as H14
    M = H.Models(dev)
    for name, arch, pf in (('part_kinpid', 'part', True), ('pnet_kinpid', 'pn', True), ('part_kin', 'part', False), ('pnet_kin', 'pn', False)):
        M.add_weaver(name, arch, pf)
    M.add_lnet(); M.add_permuted('part_kinpid_perm', 'part_kinpid'); H14.add_keras_ef(M, 'efn')
    return M


def run_gen(gen, M, R, per_class, B):
    from interp import h5_qg_counting as H
    from interp import h14_qg_generator as H14
    full = gen == 'py'
    L0, y, info = H14.load_sample(gen, per_class, SEED)
    y = np.asarray(y).astype(int); J = len(y)
    log(f'{gen}: {J} jets ({info["source"]}), quark frac {y.mean():.3f}')
    E = make_edits(L0, gen, full=full)
    log(f'{gen}: edits made; applied ' + ', '.join(f'{k} {int(v[1].sum())}' for k, v in E.items()))
    out = dict(y=y, idx=np.asarray(info['idx']))
    J0 = H.from_list(L0, Lpad=180); F0, shp = all_feats(J0)
    for k, v in F0.items(): out[f'F|base|{k}'] = v.astype(np.float32)
    out['F|base|qlead'] = np.array([(lambda q: q[q != 0][0] if (q != 0).any() else 0.0)(charge(a[:, 3])) for a in L0])
    rs = ref_scores(R, F0)
    for nm in M.m: out[f'base|{nm}'] = -M.logit(nm, J0)
    for nm, v in rs.items(): out[f'base|{nm}'] = v
    unedited = {nm: roc(out[f'base|{nm}'], y) for nm in list(M.m) + REFS}
    for nm in list(M.m) + REFS: log(f'{gen} unedited {nm}: ' + ', '.join(f'{k} {v:.4f}' for k, v in unedited[nm].items()))
    zero_check = {}
    for en, (Le, ok, nr) in E.items():
        Je = H.from_list(Le, Lpad=180); Fe, _ = all_feats(Je, shp)
        out[f'ok|{en}'] = ok; out[f'nr|{en}'] = nr
        for k in ('Q03', 'Q05', 'fch', 'n_ch', 'n_K', 'n_p', 'n_gam', 'n_K0L', 'n_n', 'n_pi'): out[f'F|{en}|{k}'] = Fe[k].astype(np.float32)
        for nm in M.m: out[f'{en}|{nm}'] = -M.logit(nm, Je)
        for nm, v in ref_scores(R, Fe).items(): out[f'{en}|{nm}'] = v
        assert np.allclose(Je['pt'], J0['pt']) and np.allclose(Je['eta'], J0['eta']) and np.allclose(Je['phi'], J0['phi'])
        for nm in ZEROS: zero_check.setdefault(nm, []).append(float(np.abs(out[f'{en}|{nm}'] - out[f'base|{nm}']).max()))
        log(f'{gen} {en}: applied {int(ok.sum())}; mean dQ03 {np.mean((Fe["Q03"] - F0["Q03"])[ok]) if ok.any() else 0:+.4f}, '
            f'dfch {np.mean((Fe["fch"] - F0["fch"])[ok]) if ok.any() else 0:+.4f}; '
            + '; '.join(f'{nm} {np.mean((out[f"{en}|{nm}"] - out[f"base|{nm}"])[ok]) if ok.any() else 0:+.4f}'
                        for nm in TESTED + ['GB_spQ', 'LR_Q03', 'LR_fch']))
    zero_check = {k: float(max(v)) for k, v in zero_check.items()}; log(f'{gen} zero-model max |Delta s|:', zero_check)
    stats = statistics(out, y, unedited, gen, full, B)
    stats['zero_models_max_abs_delta'] = zero_check
    stats['unedited'] = unedited; stats['known'] = KNOWN.get(gen, {}); stats['sample'] = dict(J=J, per_class=per_class, seed=SEED, source=info['source'])
    stats['applied'] = {k: int(v[1].sum()) for k, v in E.items()}
    stats['data'] = data_checks(out, y, gen)
    return out, stats


def data_checks(out, y, gen):
    bt = Boot(B=500); D = {}
    for c, cn in ((1, 'quark'), (0, 'gluon')):
        s = y == c; d = {}
        for k in ('Q03', 'Q05'):
            v = out[f'F|base|{k}'][s].astype(float); d[f'mean_{k}'] = float(v.mean()); d[f'ci_{k}'] = bt.ci(v)
            d[f'frac_{k}_pos'] = float((v > 0).mean()); d[f'mean_abs_{k}'] = float(np.abs(v).mean())
        N = out['F|base|N'][s].astype(float); d['mean_N'] = float(N.mean())
        for k in ('fch',): v = out[f'F|base|{k}'][s].astype(float); d['mean_fch'] = float(v.mean()); d['ci_fch'] = bt.ci(v)
        for k in ('pi', 'K', 'p', 'gam', 'K0L', 'n', 'e', 'mu'):
            d[f'frac_{k}'] = float(out[f'F|base|n_{k}'][s].sum() / N.sum())
            d[f'mean_n_{k}'] = float(out[f'F|base|n_{k}'][s].mean())
        D[cn] = d
    log(f'{gen} data checks: ' + json.dumps({c: {k: round(v, 4) for k, v in d.items() if not isinstance(v, list)} for c, d in D.items()}))
    return D


def statistics(out, y, unedited, gen, full, B):
    bt = Boot(B=B); S = dict(contrasts={}); pl = []
    sq = np.sign(out['F|base|Q03'].astype(float))
    for cn, en, ctl, kind in contrasts(full):
        if f'ok|{en}' not in out or (ctl and f'ok|{ctl}' not in out): continue
        ok = out[f'ok|{en}'] & (out[f'ok|{ctl}'] if ctl else True)
        if kind == 'signQ': ok = ok & (sq != 0)
        row = dict(n_jets=int(ok.sum()), edit=en, control=ctl)
        if ctl:
            row['dQ03_edit'] = float(np.mean((out[f'F|{en}|Q03'] - out['F|base|Q03'])[ok])); row['dQ03_ctl'] = float(np.mean((out[f'F|{ctl}|Q03'] - out['F|base|Q03'])[ok]))
            row['dfch_edit'] = float(np.mean((out[f'F|{en}|fch'] - out['F|base|fch'])[ok])); row['dfch_ctl'] = float(np.mean((out[f'F|{ctl}|fch'] - out['F|base|fch'])[ok]))
        if ok.sum() < 20: S['contrasts'][cn] = row; continue
        for nm in NETS + REFS:
            base = out[f'base|{nm}']
            de = (out[f'{en}|{nm}'] - base)
            dc = (out[f'{ctl}|{nm}'] - base) if ctl else np.zeros_like(de)
            d = (de - dc)[ok]
            if kind == 'signQ': d = d * sq[ok]
            sd = float(np.std(base)); r = dict(mean=float(d.mean()), ci=bt.ci(d), mean_over_sd=float(d.mean() / max(sd, 1e-12)),
                                                edit_mean=float(de[ok].mean()), ctl_mean=float(dc[ok].mean()))
            if nm in TESTED or nm.startswith('GB') or nm.startswith('LR'):
                r['p'] = bt.p(d)
                thr = unedited[nm]['thr50']; a = (base + (dc if ctl else 0) >= thr)[ok]; b = (base + de >= thr)[ok]
                n01, n10, pm = mcnemar(a, b); r['mcnemar'] = dict(to_quark=n01, to_gluon=n10, p=pm, calls_changed_pct=float(100 * (a != b).mean()))
            if nm in TESTED: pl.append((cn, nm, r['p']))
            if kind == 'signQ' or cn == 'E1' or cn.startswith('E2k1'):   # class x sign split of the raw edit response
                r['split'] = {}; sv = sq if not cn.startswith('E2k1') else np.sign(out['F|base|qlead'].astype(float))
                for c, lab in ((1, 'q'), (0, 'g')):
                    for sg, sl in ((1, 'Qpos' if sv is sq else 'leadpos'), (-1, 'Qneg' if sv is sq else 'leadneg')):
                        mk = ok & (y == c) & (sv == sg)
                        if mk.sum() > 10: r['split'][f'{lab}_{sl}'] = dict(n=int(mk.sum()), mean=float(de[mk].mean()), ci=bt.ci(de[mk]))
            row[nm] = r
        S['contrasts'][cn] = row
    if pl:
        adj = holm([p for _, _, p in pl])
        for (cn, nm, _), a in zip(pl, adj): S['contrasts'][cn][nm]['p_holm'] = float(a)
        S['holm_family'] = dict(n_tests=len(pl), models=TESTED, n_reject_05=int((adj < 0.05).sum()))
    # per-unit responses
    S['per_unit'] = {}
    for nm in NETS + REFS:
        u = {}
        xs, ds = [], []
        for k in (1, 2, 3):
            en = f'E2k{k}'
            if f'ok|{en}' not in out: continue
            ok = out[f'ok|{en}']; xs.append((out[f'F|{en}|Q03'] - out['F|base|Q03'])[ok]); ds.append((out[f'{en}|{nm}'] - out[f'base|{nm}'])[ok])
        if xs: u['E2_per_dQ03'] = slope_ci(np.concatenate(xs).astype(float), np.concatenate(ds), B=300)
        ok = out['ok|E1']; u['E1_per_dQ03'] = slope_ci((out['F|E1|Q03'] - out['F|base|Q03'])[ok].astype(float), (out[f'E1|{nm}'] - out[f'base|{nm}'])[ok], B=300)
        for tgt, ctl in (('g', 'C3'), ('n', 'C3'), ('r', None)):
            xs, ds = [], []
            for m in MS:
                en = f'E3{tgt}_{m}'
                if f'ok|{en}' not in out: continue
                ok = out[f'ok|{en}'] & (out[f'ok|{ctl}_{m}'] if ctl else True)
                ref = out[f'{ctl}_{m}|{nm}'] if ctl else out[f'base|{nm}']
                xs.append((out[f'F|{en}|fch'] - out['F|base|fch'])[ok]); ds.append((out[f'{en}|{nm}'] - ref)[ok])
            if xs: u[f'E3{tgt}_per_dfch'] = slope_ci(np.concatenate(xs).astype(float), np.concatenate(ds), B=300)
        S['per_unit'][nm] = u
    return S


def generator_arm(Rpy, Rhw):
    """Pythia vs Herwig: observed mean score gap per class vs the PID-attributable estimate slope_E3 x Delta f_ch"""
    G = {}
    for c in ('quark', 'gluon'):
        dfch = Rhw['data'][c]['mean_fch'] - Rpy['data'][c]['mean_fch']; g = dict(delta_fch_hw_minus_py=dfch)
        for k in ('pi', 'K', 'p', 'gam', 'K0L', 'n'):
            g[f'delta_frac_{k}'] = Rhw['data'][c][f'frac_{k}'] - Rpy['data'][c][f'frac_{k}']
        G[c] = g
    return G


def step_run(a):
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; RES.mkdir(parents=True, exist_ok=True)
    sm = '_smoke' if a.smoke else ''; per = a.smoke or 10000; B = 200 if a.smoke else 1000
    log(f'device {dev}; per class {per}')
    R, rinfo = train_refs(n_per_class=5000 if a.smoke else 50000)
    M = load_models(dev); log('models:', list(M.m))
    res = dict(refs=rinfo, models=M.info, seed=SEED)
    gstats = {}
    for gen in ('py', 'hw'):
        out, st = run_gen(gen, M, R, per, B); gstats[gen] = st
        np.savez_compressed(RES / f'h40_perjet_{gen}{sm}.npz', **{k: (v.astype(np.float32) if v.dtype == np.float64 else v) for k, v in out.items()})
        # generator arm: mean score per class
        st['mean_score'] = {nm: {c: float(out[f'base|{nm}'][out['y'] == ci].mean()) for c, ci in (('quark', 1), ('gluon', 0))} for nm in NETS + REFS}
        res[gen] = st
        (RES / f'h40_qg_pid_charge{sm}.json').write_text(json.dumps(res, indent=1, default=float))
    G = generator_arm(gstats['py'], gstats['hw'])
    for nm in NETS + REFS:
        sl = gstats['py']['per_unit'][nm].get('E3g_per_dfch', {}).get('slope', np.nan)
        for c in ('quark', 'gluon'):
            G[c][nm] = dict(score_gap_hw_minus_py=gstats['hw']['mean_score'][nm][c] - gstats['py']['mean_score'][nm][c],
                            fch_attributable=float(sl * G[c]['delta_fch_hw_minus_py']))
    G['auc'] = {nm: dict(py=gstats['py']['unedited'][nm]['AUC'], hw=gstats['hw']['unedited'][nm]['AUC']) for nm in NETS + REFS}
    res['generator_arm'] = G
    (RES / f'h40_qg_pid_charge{sm}.json').write_text(json.dumps(res, indent=1, default=float))
    log('wrote', RES / f'h40_qg_pid_charge{sm}.json')


# ================================================================================================================== figure
def step_fig(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sm = '_smoke' if a.smoke else ''
    R = json.loads((RES / f'h40_qg_pid_charge{sm}.json').read_text())['py']['contrasts']
    rows = [('E1_signQ', 'E1 charge conjugation\n(response x sign Q$_{0.3}$)'), ('E2k1-C2', 'E2 flip hardest charged\nminus Q-matched pair flip'),
            ('E2k3-C2', 'E2 flip 3 hardest charged\nminus Q-matched pair flip'),
            ('E3g_6-C3', 'E3 6 soft $\\pi^\\pm\\to\\gamma$\nminus $\\pi^+\\leftrightarrow\\pi^-$'),
            ('E3n_6-C3', 'E3 6 soft $\\pi^\\pm\\to K^0_L$\nminus $\\pi^+\\leftrightarrow\\pi^-$'),
            ('E3r_6-C3r', 'E3 6 soft $\\gamma\\to\\pi^\\pm$\nminus $\\gamma\\to K^0_L$'),
            ('E4a-C4', 'E4 all $K^\\pm$, p $\\to\\pi^\\pm$'), ('E4b-C4', 'E4 $\\pi^\\pm\\to K^\\pm$ (count-matched)')]
    rows = [r for r in rows if r[0] in R and 'part_kinpid' in R[r[0]]]
    mods = [('part_kinpid', 'ParT kinpid', '#1f77b4'), ('pnet_kinpid', 'ParticleNet kinpid', '#ff7f0e'), ('lorentznet', 'LorentzNet', '#2ca02c'),
            ('part_kinpid_perm', 'ParT kinpid, permuted weights', '#7f7f7f'), ('GB_spQ', 'GBDT(shapes+N+species+Q)', '#d62728')]
    fig, ax = plt.subplots(figsize=(8.5, 0.75 * len(rows) + 1.6))
    for i, (cn, lab) in enumerate(rows):
        for k, (nm, ml, col) in enumerate(mods):
            r = R[cn].get(nm)
            if r is None: continue
            yv = i + (k - 2) * 0.14
            ax.errorbar(r['mean'], yv, xerr=[[r['mean'] - r['ci'][0]], [r['ci'][1] - r['mean']]], fmt='o', color=col, ms=4, capsize=2,
                        label=ml if i == 0 else None)
    ax.axvline(0, color='k', lw=0.6)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([l for _, l in rows], fontsize=8); ax.invert_yaxis()
    ax.set_xlabel('mean change of quark score  log p(q) - log p(g)  (edit minus control), 95 % CI')
    ax.set_title('Quark-gluon taggers: particle-ID edits at fixed four-momenta (Pythia test jets, 20,000)', fontsize=10)
    ax.legend(fontsize=7, loc='lower right'); ax.grid(axis='x', alpha=0.3)
    fig.tight_layout(); p = RES / f'h40_pid_edits{sm}.png'; fig.savefig(p, dpi=140); log('wrote', p)


def step_pz(a):
    """normal-approximation (one-sample t, n >= 9,000) two-sided p for every tested contrast, Holm per generator; the sign-flip
    permutation p of step run has a floor of 1/(P+1) = 5e-4, which puts every Holm-adjusted p at >= 0.022-0.048"""
    from scipy.stats import norm
    sm = '_smoke' if a.smoke else ''; jp = RES / f'h40_qg_pid_charge{sm}.json'; R = json.loads(jp.read_text())
    for gen in ('py', 'hw'):
        out = dict(np.load(RES / f'h40_perjet_{gen}{sm}.npz')); C = R[gen]['contrasts']; pl = []
        sq = np.sign(out['F|base|Q03'].astype(float))
        for cn, en, ctl, kind in contrasts(gen == 'py'):
            if cn not in C: continue
            ok = out[f'ok|{en}'] & (out[f'ok|{ctl}'] if ctl else True)
            if kind == 'signQ': ok = ok & (sq != 0)
            for nm in TESTED:
                if nm not in C[cn]: continue
                d = (out[f'{en}|{nm}'].astype(float) - out[f'base|{nm}'].astype(float)) - \
                    ((out[f'{ctl}|{nm}'].astype(float) - out[f'base|{nm}'].astype(float)) if ctl else 0)
                d = d[ok] * (sq[ok] if kind == 'signQ' else 1)
                se = d.std(ddof=1) / np.sqrt(len(d)); z = d.mean() / se if se > 0 else 0.0
                p = float(2 * norm.sf(abs(z))) if se > 0 else 1.0
                C[cn][nm]['z'] = float(z); C[cn][nm]['p_t'] = p; pl.append((cn, nm, p))
        adj = holm([p for _, _, p in pl])
        for (cn, nm, _), q in zip(pl, adj): C[cn][nm]['p_t_holm'] = float(q)
        R[gen]['holm_family_t'] = dict(n_tests=len(pl), n_reject_05=int((adj < 0.05).sum()),
                                       not_rejected=[f'{cn}|{nm}' for (cn, nm, _), q in zip(pl, adj) if q >= 0.05])
        log(gen, R[gen]['holm_family_t'])
    jp.write_text(json.dumps(R, indent=1, default=float)); log('wrote', jp)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', choices=['run', 'fig', 'pz'], required=True)
    ap.add_argument('--smoke', type=int, default=0)
    a = ap.parse_args()
    {'run': step_run, 'fig': step_fig, 'pz': step_pz}[a.step](a)
