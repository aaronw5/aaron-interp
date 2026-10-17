"""E7 (JEDI-linear 64 p, 3-feature; claim A5, Paper B open question): the 16 z neurons as hinge formulas on the network's
own second-round objects S vs on hand observables -- stage substitution on fresh jets, fidelity under mechanism-separating
input edits (no refit), and the negative poles.

Samples
  fit     sample(per_class=6000, seed=2) = 30,000 jets; random split (rng 0) 2/3 fit (20,000) / 1/3 'e12ev' (10,000), as E12
  test    sample(per_class=3000, seed=43) minus every jet also in the seed-2 sample (fresh jets; N logged)
Targets: the 16 pre-ReLU z (fixed-point network), clipped below at -0.2 sd of their positive values (as E12).
Bases (columns), equal term budget K in {4, 8, 16, 32} per neuron, hinge (MARS) family as E12 (X, max(0, X-t), max(0, t-X),
19 quantile knots, products of a chosen hinge with a hinge of the 10 columns most correlated with the residual; forward
selection on 12,000 fit rows (6,000 for hand_all), least-squares refit on all 20,000 fit jets):
  S        the 19 live S units (E1)
  SP       S + the 11 live P units
  hand     87 jet-level Q observables (E12's list)
  hand_all every Q observable in the library (finite, varying)
  initS    control: S of a He-normal random-init network (zero bias, float), 19 highest-variance units (E12)
  randS    control: K random hinge terms on the 19 live S units (10 draws), least-squares refit
Substitution: z -> max(0, formula) inside the jitted forward; the network's own d7 makes the call. All 16 at once and one
at a time. Edits (formulas not refit; hand observables recomputed from the edited particles with
jetdistill.observables.compute, S / P read off the network on the edited jets):
  trans  rigid translation of the real particles by +0.06 rad in eta (E6/E6v; invariant mass unchanged)
  rot    rotation of every jet about its axis by 45 deg (E5v)
  a32x   one particle with 9-13 GeV set to 10 GeV (before) and 12 GeV (after), pT rank kept (E13; a32 edge 11.0 GeV)
  a32c   matched control: one particle with 12-16 GeV, 13 -> 15 GeV (no a32 crossing)
Negative pole (S, hand): per neuron, hinge K=8 on the fit jets with z < 0 (raw pre-ReLU target), test R^2 on fresh z < 0 jets,
largest terms. Statistics (paired bootstrap over jets, McNemar, Holm over 16 neurons) in --combine.

  python -m interp.e7_z_objects_jedi --basis S,SP,initS,randS      (one job per group of bases)
  python -m interp.e7_z_objects_jedi --basis hand
  python -m interp.e7_z_objects_jedi --basis hand_all
  python -m interp.e7_z_objects_jedi --combine                     (statistics, figure; reads the npz files)
  add --smoke for a small run (300 / 150 jets per class, B = 100)
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp.jedi_core import Jedi, sample

ap = argparse.ArgumentParser()
ap.add_argument('--basis', default='')
ap.add_argument('--combine', action='store_true')
ap.add_argument('--smoke', action='store_true')
ARGS = ap.parse_args()
SMOKE = ARGS.smoke
OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'jedi'; OUT.mkdir(parents=True, exist_ok=True)
TAG = 'e7s' if SMOKE else 'e7'
PER, PER_T = (300, 150) if SMOKE else (6000, 3000)
SEED, SEED_T = 2, 43
KS = (4, 8, 16, 32); KMAX = max(KS)
N_SEL, N_SEL_ALL, N_KNOTS, TOP_PROD = 12000, 6000, 19, 10
N_RAND = 3 if SMOKE else 10; N_RAND_Z = 3 if SMOKE else 5
NBOOT = 100 if SMOKE else 1000
CL = ['g', 'q', 'W', 'Z', 't']; Wc, Zc = 2, 3
t0 = time.time()


def log(*a):
    print(f'[{time.time()-t0:6.0f} s] ' + ' '.join(str(v) for v in a), flush=True)


# ================================================================== common setup (network, samples, edits)
net = Jedi(64)


def run_override(xx, ov, batch=5000):
    """logits with per-jet overrides {stage: (mask (units,), values (J, units))} (z, S: internal shape (J, units))"""
    out = []
    for i in range(0, len(xx), batch):
        xb = xx[i:i + batch]
        O = {k: (np.asarray(m, np.float32), np.asarray(v[i:i + batch], np.float32).reshape(len(xb), -1)) for k, (m, v) in ov.items()}
        r = net._f(net.inputs(xb, net.n), O); out.append(np.asarray(r['logits'], np.float64).reshape(len(xb), -1))
    return np.concatenate(out)


def sample_idx(yall, per, seed):
    """the row indices jedi_core.sample(per, seed) draws (same rng calls)"""
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(yall == c), per, replace=False) for c in range(5)]))


def edit_shift(xx, de, dp):
    e = xx.copy(); rl = e[..., 0] > 0; e[..., 1] += np.float32(de) * rl; e[..., 2] += np.float32(dp) * rl; return e


def rotate(xx, deg):
    th = np.deg2rad(deg); c, s = np.cos(th), np.sin(th); e = xx.copy()
    e[..., 1] = c * xx[..., 1] - s * xx[..., 2]; e[..., 2] = s * xx[..., 1] + c * xx[..., 2]; return e.astype(np.float32)


def ramp(xx, lo, hi, win):
    """E13: one particle per jet with pT in win set to lo (before) and hi (after), only where the pT rank order is kept"""
    pt = xx[..., 0]; nreal = (pt > 0).sum(1); J, slots = [], []
    for j in range(len(xx)):
        n = nreal[j]; cand = np.flatnonzero((pt[j, :n] >= win[0]) & (pt[j, :n] <= win[1]))
        ok = [i for i in cand if (i == 0 or pt[j, i - 1] >= hi) and (pt[j, i + 1] <= lo if i + 1 < 64 else True)]
        if ok:
            i = min(ok, key=lambda i: abs(pt[j, i] - (lo + hi) / 2)); J.append(j); slots.append(i)
    J, slots = np.array(J, int), np.array(slots, int)
    xb, xa = xx[J].copy(), xx[J].copy(); xb[np.arange(len(J)), slots, 0] = lo; xa[np.arange(len(J)), slots, 0] = hi
    return J, xb, xa


def he_S(xx, seed=0, batch=5000):
    """E12's initS: S of a randomly initialised network of the same architecture (He-normal, zero biases, float)"""
    rg = np.random.default_rng(seed); he = lambda i, o: rg.normal(0, np.sqrt(2 / i), (i, o)).astype(np.float32)
    W1, Wd, W2, W3 = he(3, 64), he(64, 64), he(64, 64), he(64, 64); out = []
    for i in range(0, len(xx), batch):
        xn = np.asarray(net.inputs(xx[i:i + batch], 64), np.float32)
        a = np.maximum(xn @ W1, 0); P = a.sum(1) * net.pool_scale[0]; g = np.maximum(P @ Wd, 0)
        u = np.maximum((np.maximum(a @ W2, 0) + g[:, None, :]) @ W3, 0); out.append((u.sum(1) * net.pool_scale[1]).astype(np.float64))
    return np.concatenate(out)


x0, y0, _ = sample(per_class=400 if SMOKE else 2000, seed=0)
ACC0 = float((net.run(x0)['logits'].argmax(1) == y0).mean()); del x0, y0
log(f'check: seed-0 sample, network accuracy {ACC0*100:.2f} % (E1: 81.29 % on 10,000 jets)')
x, y, Q = sample(per_class=PER, seed=SEED); J = len(x)
R = net.run(x, keep=('P', 'S', 'z', 'logits')); call = R['logits'].argmax(1)
perm = np.random.default_rng(0).permutation(J); nf = 2 * J // 3
fit, ev = np.sort(perm[:nf]), np.sort(perm[nf:])
from jetdistill.config import RESULTS as _JR
yall = np.asarray(np.load(Path(_JR) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r')['y'])
i2, i43 = sample_idx(yall, PER, SEED), sample_idx(yall, PER_T, SEED_T)
xt, yt, Qt = sample(per_class=PER_T, seed=SEED_T)
assert len(i43) == len(xt) and (yall[i43] == yt).all()
fresh = ~np.isin(i43, i2)
xt, yt = xt[fresh], yt[fresh]; Qt = {k: v[fresh] for k, v in Qt.items()}; JT = len(xt)
Rt = net.run(xt, keep=('P', 'S', 'z', 'logits')); ct = Rt['logits'].argmax(1)
ACC_T = float((ct == yt).mean())
log(f'fit sample {J} jets (seed {SEED}): fit {len(fit)}, e12ev {len(ev)}; network accuracy e12ev {(call[ev] == y[ev]).mean()*100:.2f} %, '
    f'all {(call == y).mean()*100:.2f} %')
log(f'test sample: per_class {PER_T}, seed {SEED_T}: {len(i43)} jets, {int((~fresh).sum())} also in the seed-2 sample removed -> {JT} '
    f'fresh jets; network accuracy {ACC_T*100:.2f} %  (expected 80.6-81.3 %)')

EDITS = {}
EDITS['trans'] = (np.arange(JT), xt, edit_shift(xt, 0.06, 0.0))
EDITS['rot'] = (np.arange(JT), xt, rotate(xt, 45.0))
EDITS['a32x'] = ramp(xt, 10.0, 12.0, (9, 13))
EDITS['a32c'] = ramp(xt, 13.0, 15.0, (12, 16))
RN = {}
for k, (Je, xb, xa) in EDITS.items():
    rb = Rt if xb is xt else net.run(xb, keep=('P', 'S', 'z', 'logits'))
    RN[k] = (rb, net.run(xa, keep=('P', 'S', 'z', 'logits')))
    cb, ca = RN[k][0]['logits'].argmax(1), RN[k][1]['logits'].argmax(1)
    log(f'edit {k}: N {len(Je)}; network calls changed {(cb != ca).mean()*100:.2f} %; true W called Z before {(cb[yt[Je] == Wc] == Zc).mean()*100:.1f} %, '
        f'after {(ca[yt[Je] == Wc] == Zc).mean()*100:.1f} %')

Z = R['z']; Zf = Z[fit]
FLOOR = np.array([-0.2 * (Zf[Zf[:, j] > 0, j].std() if (Zf[:, j] > 0).sum() > 50 else Zf[:, j].std()) for j in range(16)])
T = np.maximum(Z, FLOOR)
SDH = np.maximum(np.maximum(Zf, 0).std(0), 1e-9)            # sd of relu(z) on the fit jets (standardises dz)
e1 = json.loads((OUT / 'e1_e3.json').read_text())['E1']
liveS = [int(k) for k in np.flatnonzero(np.array(e1['S']['changed']) > 0.01)]
liveP = [int(k) for k in np.flatnonzero(np.array(e1['P']['changed']) > 0.01)]
SETS = ['e12ev', 'test'] + [f'{k}_{s}' for k in EDITS for s in ('b', 'a') if not (k in ('trans', 'rot') and s == 'b')]
# set name -> (particles, network stages); 'trans_b' / 'rot_b' are the test jets
def set_data(name):
    if name == 'e12ev': return x[ev], {k: v[ev] for k, v in R.items()}
    if name == 'test': return xt, Rt
    k, s = name.rsplit('_', 1); Je, xb, xa = EDITS[k]
    return (xb, RN[k][0]) if s == 'b' else (xa, RN[k][1])


HAND = ['sum_pt', 'log_sum_pt', 'mass', 'mass_over_sum_pt', 'mass_over_sum_pt_sq', 'max_dr', 'pt_dispersion', 'z_1st', 'z_2nd',
        'z_3rd', 'z_top5', 'girth', 'girth2', 'mean_eta', 'mean_eta2', 'mean_phi', 'mean_phi2', 'n_particles', 'n_pt_above_1',
        'n_pt_above_5', 'n_pt_above_10', 'n_pt_above_50', 'tau21', 'tau32', 'e2', 'C2', 'D2', 'e2_sq', 'LHA', 'centroid_offset',
        'planar_flow', 'eccentricity', 'width', 'lam1', 'lam2', 'orientation_deg', 'dr01', 'pt1_over_pt0', 'pt_balance01',
        'pt1_dr01', 'm01', 'm012', 'min_pair_mass', 'max_pair_mass', 'dr02', 'dr12', 'dr_min_012', 'dr_max_012', 'pt2_over_pt0',
        'mratio_min_012', 'mratio_max_012', 'n_dr_0_0p05', 'z_dr_0_0p05', 'n_dr_0p05_0p1', 'z_dr_0p05_0p1', 'n_dr_0p1_0p2',
        'z_dr_0p1_0p2', 'n_dr_0p2_0p4', 'z_dr_0p2_0p4', 'n_dr_0p4_up', 'z_dr_0p4_up', 'psi_0p1', 'psi_0p2', 'psi_0p3',
        'n_for_50pct', 'n_for_90pct', 'pt_entropy', 'e3', 'e4', 'C3', 'D3', 'N2', 'N3', 'M2', 'M3', 'C2_b2', 'D2_b2', 'tau1',
        'tau2', 'tau3', 'tau4', 'tau43', 'tau21_b2', 'sd_mass', 'sd_zg', 'sd_rg', 'sd_nremoved']


def hand_keys(allq):
    from jetdistill.observables.library import library
    lib = library(64); keys = sorted(Q) if allq else HAND
    return [k for k in keys if k in Q and k in lib and np.isfinite(Q[k][fit]).all() and np.ptp(Q[k][fit]) > 0]


# ================================================================== hinge fitting (as E12; column norms computed in chunks)
def knots(v):
    return np.unique(np.round(np.quantile(v, np.linspace(.05, .95, N_KNOTS)), 12))


def one(v, kind, t):
    return v if kind == 'lin' else np.maximum(0, v - t) if kind == 'gt' else np.maximum(0, t - v)


def tval(tm, X):
    v = one(X[:, tm[1]], tm[0], tm[2])
    return v * one(X[:, tm[4]], tm[3], tm[5]) if len(tm) > 3 else v


def hinge_dictionary(Xs_):
    blocks, desc, rng_ = [], [], {}
    for f in range(Xs_.shape[1]):
        v = Xs_[:, f]; ks = knots(v); c = [v]; d = [('lin', f, None)]
        for t in ks: c.append(np.maximum(0, v - t)); d.append(('gt', f, float(t)))
        for t in ks: c.append(np.maximum(0, t - v)); d.append(('lt', f, float(t)))
        c = np.stack(c, 1); ok = c.std(0) > 1e-12
        idx = np.flatnonzero(ok); start = sum(b.shape[1] for b in blocks)
        blocks.append(c[:, idx].astype(np.float32)); desc += [d[i] for i in idx]; rng_[f] = (start, start + len(idx))
    return np.concatenate(blocks, 1), desc, rng_


def colstats(C, chunk=1000):
    s2, s1 = np.zeros(C.shape[1]), np.zeros(C.shape[1])
    for a in range(0, C.shape[1], chunk):
        c = C[:, a:a + chunk].astype(np.float64); s2[a:a + chunk] = (c * c).sum(0); s1[a:a + chunk] = c.sum(0)
    return s2, s1


class Selector:
    def __init__(self, y):
        n = len(y); self.Qb = np.zeros((n, KMAX + 1)); self.Qb[:, 0] = 1 / np.sqrt(n); self.k = 1; self.r = y - y.mean()

    def add(self, v):
        Qb = self.Qb[:, :self.k]
        for _ in range(2): v = v - Qb @ (Qb.T @ v)
        nv = np.linalg.norm(v)
        if nv < 1e-9: return None
        q = v / nv; self.Qb[:, self.k] = q; self.k += 1; self.r = self.r - q * (q @ self.r); return q


def select_columns(C, desc, y, kmax, products, X_sel, nC2, csum):
    S_ = Selector(y); nn = nC2 - csum ** 2 / len(y)
    chosen, used = [], np.zeros(C.shape[1], bool)
    Xc = (X_sel - X_sel.mean(0)) / np.maximum(X_sel.std(0), 1e-12)
    for _ in range(kmax):
        r32 = S_.r.astype(np.float32)
        g = np.where((nn > 1e-5 * nC2) & ~used, (C.T @ r32).astype(np.float64) ** 2 / np.maximum(nn, 1e-30), 0)
        i = int(g.argmax()); best = (g[i], desc[i], C[:, i].astype(np.float64), i)
        cr = np.abs(Xc.T @ S_.r); top = np.argsort(-cr)[:TOP_PROD]
        Qb32 = S_.Qb[:, :S_.k].astype(np.float32)
        for tm in chosen:
            if len(tm) > 3 or tm[0] == 'lin': continue
            pv = tval(tm, X_sel).astype(np.float32); cols, dd = [], []
            for f in top:
                if f == tm[1]: continue
                a, b = products[f]
                for c in range(a, b):
                    if desc[c][0] == 'lin': continue
                    cols.append(c); dd.append(tm + desc[c])
            if not cols: continue
            Pm = C[:, cols] * pv[:, None]; p2 = (Pm.astype(np.float64) ** 2).sum(0)
            npm = p2 - ((Qb32.T @ Pm).astype(np.float64) ** 2).sum(0)
            gp = np.where(npm > 1e-5 * p2, (Pm.T @ r32).astype(np.float64) ** 2 / np.maximum(npm, 1e-30), 0)
            j = int(gp.argmax())
            if gp[j] > best[0]: best = (gp[j], dd[j], Pm[:, j].astype(np.float64), None)
        if best[0] <= 0: break
        q = S_.add(best[2])
        if q is None: break
        chosen.append(best[1])
        if best[3] is not None: used[best[3]] = True
        nn = nn - (C.T @ q.astype(np.float32)).astype(np.float64) ** 2
    return chosen


def lsq(B, y):
    return np.linalg.lstsq(B, y, rcond=None)[0]


def design(tl, X):
    return np.stack([np.ones(len(X))] + [tval(t, X) for t in tl], 1)


def name_term(tm, names):
    def h(kind, f, t):
        return names[f] if kind == 'lin' else f'max(0,{names[f]}-{t:.4g})' if kind == 'gt' else f'max(0,{t:.4g}-{names[f]})'
    s = h(*tm[:3]); return s + '*' + h(*tm[3:]) if len(tm) > 3 else s


def fit_hinge(Xf, targets, kmax, nsel, seed=0):
    """forward selection on nsel rows (per target column), returns the term lists"""
    sel = np.sort(np.random.default_rng(seed).choice(len(Xf), min(nsel, len(Xf)), replace=False))
    Xsel = Xf[sel]; C, desc, rng_ = hinge_dictionary(Xsel); nC2, csum = colstats(C)
    log(f'    hinge dictionary: {C.shape[1]} columns x {C.shape[0]} selection rows ({C.nbytes/2**20:.0f} MiB)')
    terms = []
    for j in range(targets.shape[1]):
        terms.append(select_columns(C, desc, targets[sel, j], kmax, rng_, Xsel, nC2, csum))
    del C
    return terms


# ================================================================== per-basis job
def basis_matrix(b, xx, Rn, Qd, hk, uU):
    if b in ('S', 'randS'): return Rn['S'][:, liveS]
    if b == 'SP': return np.concatenate([Rn['S'][:, liveS], Rn['P'][:, liveP]], 1)
    if b == 'initS': return he_S(xx)[:, uU]
    if Qd is None:
        from jetdistill.observables import compute
        Qd = compute(np.asarray(xx, np.float64), 64, ids=hk)
    return np.nan_to_num(np.stack([np.asarray(Qd[k], np.float64) for k in hk], 1))


def evaluate(zh_sets, label):
    """substitution calls per set (all 16), single-neuron calls on test; quick log"""
    calls = {}
    for s, zh in zh_sets.items():
        xx, _ = set_data(s); calls[s] = run_override(xx, {'z': (np.ones(16), np.maximum(zh, 0))}).argmax(1).astype(np.int8)
    return calls


def quick(calls, zh, label):
    ch = (calls['test'] != ct).mean() * 100; acc = (calls['test'] == yt).mean() * 100
    r2 = 1 - ((np.maximum(Rt['z'], 0) - np.maximum(zh['test'], 0)) ** 2).mean(0) / np.maximum(np.maximum(Rt['z'], 0).var(0), 1e-12)
    msg = f'  {label}: test calls changed {ch:.2f} %, acc {acc:.2f} % (network {ACC_T*100:.2f} %), R2 mean {r2.mean():.3f}; e12ev changed {(calls["e12ev"] != call[ev]).mean()*100:.2f} %'
    Je = EDITS['trans'][0]; yW = yt[Je] == Wc
    msg += f'; trans W->Z formula {(calls["trans_a"][yW] == Zc).mean()*100:.1f} % (network {(RN["trans"][1]["logits"].argmax(1)[yW] == Zc).mean()*100:.1f} %)'
    dn = (np.maximum(RN['trans'][1]['z'], 0) - np.maximum(Rt['z'], 0)) / SDH; df = (np.maximum(zh['trans_a'], 0) - np.maximum(zh['test'], 0)) / SDH
    xn, yf = dn.ravel(), df.ravel(); msg += f'; trans dz slope {np.cov(xn, yf)[0, 1] / max(xn.var(), 1e-12):.3f}'
    log(msg)


def run_basis(b):
    t1 = time.time(); res = dict(basis=b, KS=list(KS)); arrs = {}
    hk, uU = None, None
    if b in ('hand', 'hand_all'):
        hk = hand_keys(b == 'hand_all'); res['observables'] = hk
        from jetdistill.observables import compute
        chk = compute(np.asarray(xt[:300], np.float64), 64, ids=hk)
        dq = max(float(np.abs(np.asarray(chk[k]) - Qt[k][:300]).max() / max(np.abs(Qt[k][:300]).max(), 1e-12)) for k in hk)
        res['recompute_vs_stored_max_rel'] = dq
        log(f'  {b}: {len(hk)} observables; recomputed vs stored on 300 test jets: max rel diff {dq:.2e}')
    if b == 'initS':
        SUf = he_S(x); uU = [int(k) for k in np.argsort(-SUf[fit].var(0))[:19]]; res['initS_units'] = uU
    # basis matrices for the fit jets and every evaluation set
    Xfit_all = basis_matrix(b, x, R, Q if hk else None, hk, uU)
    Xf = Xfit_all[fit]
    XS = {'e12ev': Xfit_all[ev], 'test': basis_matrix(b, xt, Rt, Qt if hk else None, hk, uU)}
    for s in SETS[2:]:
        xx, Rn = set_data(s); tq = time.time(); XS[s] = basis_matrix(b, xx, Rn, None, hk, uU)
        if hk: log(f'  {b}: observables recomputed for {s} ({len(xx)} jets) in {time.time()-tq:.0f} s')
    names = ([f'S{k}' for k in liveS] + ([f'P{k}' for k in liveP] if b == 'SP' else [])) if b in ('S', 'SP', 'randS') else \
        hk if hk else [f'iS{k}' for k in uU]
    res['columns'] = names
    if b == 'randS':
        sel = np.sort(np.random.default_rng(0).choice(len(Xf), min(N_SEL, len(Xf)), replace=False))
        _, descS, _ = hinge_dictionary(Xf[sel]); res['draws'] = {}
        for d in range(N_RAND):
            rg = np.random.default_rng(100 + d); picks = [rg.choice(len(descS), KMAX, replace=False) for _ in range(16)]
            for K in KS:
                zh = {s: np.zeros((len(XS[s]), 16)) for s in SETS}
                for j in range(16):
                    tl = [descS[i] for i in picks[j][:K]]; cf = lsq(design(tl, Xf), T[fit, j])
                    for s in SETS: zh[s][:, j] = design(tl, XS[s]) @ cf
                calls = evaluate(zh, '')
                for s in SETS:
                    arrs[f'call_d{d}_K{K}_{s}'] = calls[s]
                    if d < N_RAND_Z and K in (8, 32): arrs[f'zh_d{d}_K{K}_{s}'] = zh[s].astype(np.float32)
                quick(calls, zh, f'randS draw {d} K={K}')
        np.savez_compressed(OUT / f'{TAG}_{b}.npz', **arrs); (OUT / f'{TAG}_{b}.json').write_text(json.dumps(res, indent=1))
        return
    nsel = N_SEL_ALL if b == 'hand_all' else N_SEL
    log(f'--- basis {b}: {Xf.shape[1]} columns; selection rows {min(nsel, len(Xf))}')
    terms = fit_hinge(Xf, T[fit], KMAX, nsel); log(f'  {b}: selection done ({time.time()-t1:.0f} s)')
    res['formulas'] = {}
    for K in KS:
        zh = {s: np.zeros((len(XS[s]), 16)) for s in SETS}; forms = []
        for j in range(16):
            tl = terms[j][:K]; cf = lsq(design(tl, Xf), T[fit, j])
            for s in SETS: zh[s][:, j] = design(tl, XS[s]) @ cf
            Bt = design(tl, Xf)[:, 1:]; contrib = np.abs(cf[1:]) * Bt.std(0)
            forms.append(dict(neuron=j, intercept=float(cf[0]), terms=[[name_term(t, names), float(c), float(w)] for t, c, w in
                                                                       sorted(zip(tl, cf[1:], contrib), key=lambda r: -r[2])]))
        res['formulas'][str(K)] = forms
        calls = evaluate(zh, '')
        for s in SETS:
            arrs[f'zh_K{K}_{s}'] = zh[s].astype(np.float32); arrs[f'call_K{K}_{s}'] = calls[s]
        h = np.maximum(zh['test'], 0)
        for j in range(16):
            m = np.zeros(16); m[j] = 1
            arrs[f'single_K{K}_z{j}'] = run_override(xt, {'z': (m, h)}).argmax(1).astype(np.int8)
        quick(calls, zh, f'{b} K={K}')
    # negative pole: the off side of each neuron (raw pre-ReLU z < 0), hinge K=8
    if b in ('S', 'hand'):
        res['negative_pole'] = []
        for j in range(16):
            mf = Zf[:, j] < 0; mt = Rt['z'][:, j] < 0
            if mf.sum() < 200 or mt.sum() < 100:
                res['negative_pole'].append(dict(neuron=j, n_fit=int(mf.sum()), skipped=True)); continue
            tl = fit_hinge(Xf[mf], Zf[mf][:, [j]], 8, 6000)[0]; cf = lsq(design(tl, Xf[mf]), Zf[mf, j])
            pred = design(tl, XS['test'][mt]) @ cf; zt = Rt['z'][mt, j]
            r2 = float(1 - ((zt - pred) ** 2).mean() / max(zt.var(), 1e-12))
            Bt = design(tl, Xf[mf])[:, 1:]; contrib = np.abs(cf[1:]) * Bt.std(0)
            res['negative_pole'].append(dict(neuron=j, n_fit=int(mf.sum()), n_test=int(mt.sum()), r2_test=r2, intercept=float(cf[0]),
                                             terms=[[name_term(t, names), float(c), float(w)] for t, c, w in sorted(zip(tl, cf[1:], contrib), key=lambda r: -r[2])]))
            log(f'  negative pole z{j} ({b}): fit {int(mf.sum())} jets z<0, test R2 {r2:.3f}; largest terms ' +
                '; '.join(t[0] for t in res['negative_pole'][-1]['terms'][:3]))
    res['time_s'] = time.time() - t1
    np.savez_compressed(OUT / f'{TAG}_{b}.npz', **arrs); (OUT / f'{TAG}_{b}.json').write_text(json.dumps(res, indent=1))
    log(f'  {b}: saved {OUT / f"{TAG}_{b}.npz"} ({time.time()-t1:.0f} s)')


# ================================================================== statistics (combine)
_W = {}
def bw(n):
    """bootstrap count matrix (NBOOT, n) for a set of n jets (shared by every quantity on those jets: paired)"""
    if n not in _W:
        rng = np.random.default_rng(1000 + n); W = np.zeros((NBOOT, n))
        for i in range(NBOOT): W[i] = np.bincount(rng.integers(0, n, n), minlength=n)
        _W[n] = W
    return _W[n]


def ci(point, boot):
    return dict(v=float(point), lo=float(np.nanpercentile(boot, 2.5)), hi=float(np.nanpercentile(boot, 97.5)))


def p_boot(boot):
    b = boot[np.isfinite(boot)]
    return float(max(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean())), 1 / len(b)))


def rate(ind, den=None):
    """point and bootstrap of mean(ind | den)"""
    ind = np.asarray(ind, np.float32); den = np.ones_like(ind) if den is None else np.asarray(den, np.float32)
    W = bw(len(ind)); return (ind * den).sum() / den.sum(), (W @ (ind * den)) / np.maximum(W @ den, 1e-9)


def slope_corr(xn, yf):
    """OLS slope of yf on xn (with intercept) and Pearson r: pooled over all columns and per column; xn, yf (n, c);
    bootstrap over rows (the same count matrix for every quantity on these jets). Returns pooled (slope, boot, r, boot)
    and per column a list of (slope, boot, r)."""
    n, c = xn.shape
    agg = np.stack([np.ones((n, c)), xn, yf, xn * xn, xn * yf, yf * yf], 2)          # (n, c, 6)
    W = bw(n); A = np.concatenate([agg.sum(0)[None], (W @ agg.reshape(n, -1)).reshape(len(W), c, 6)], 0)   # (1 + B, c, 6)
    def st(A_):
        N, sx, sy, sxx, sxy, syy = np.moveaxis(A_, -1, 0)
        cxy = sxy - sx * sy / N; cxx = sxx - sx * sx / N; cyy = syy - sy * sy / N
        return cxy / np.maximum(cxx, 1e-30), cxy / np.sqrt(np.maximum(cxx * cyy, 1e-30))
    sl, r = st(A.sum(1)); slc, rc = st(A)
    return sl[0], sl[1:], r[0], r[1:], [(slc[0, j], slc[1:, j], rc[0, j]) for j in range(c)]


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.zeros(m); run = 0
    for i, k in enumerate(o):
        run = max(run, min(1, (m - i) * p[k])); adj[k] = run
    return adj.tolist()


def mcnemar(a_ok, b_ok):
    from scipy.stats import binomtest
    b = int((a_ok & ~b_ok).sum()); c = int((~a_ok & b_ok).sum())
    return dict(n01=b, n10=c, p=float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0)


def combine():
    from scipy.stats import rankdata
    out = dict(N_fit=int(len(fit)), N_e12ev=int(len(ev)), N_test=int(JT), seeds=dict(fit=SEED, test=SEED_T), n_boot=NBOOT,
               acc_check_seed0=ACC0, acc_net_test=ACC_T, acc_net_e12ev=float((call[ev] == y[ev]).mean()),
               edits={k: dict(N=int(len(v[0]))) for k, v in EDITS.items()}, liveS=liveS, liveP=liveP)
    B = {}
    for b in ('S', 'SP', 'hand', 'hand_all', 'initS', 'randS'):
        f = OUT / f'{TAG}_{b}.npz'
        if f.exists(): B[b] = dict(np.load(f)); log(f'loaded {f.name}')
        else: log(f'missing {f.name}')
    # network quantities per set
    NETZ = {'e12ev': R['z'][ev], 'test': Rt['z']}; NETC = {'e12ev': call[ev], 'test': ct}
    for k in EDITS:
        NETZ[f'{k}_b'], NETZ[f'{k}_a'] = RN[k][0]['z'], RN[k][1]['z']
        NETC[f'{k}_b'], NETC[f'{k}_a'] = RN[k][0]['logits'].argmax(1), RN[k][1]['logits'].argmax(1)
    NETZ['trans_b'] = NETZ['rot_b'] = Rt['z']; NETC['trans_b'] = NETC['rot_b'] = ct
    def get(b, kind, K, s, d=None):
        A = B[b]; s2 = 'test' if s in ('trans_b', 'rot_b') else s
        key = f'{kind}_K{K}_{s2}' if d is None else f'{kind}_d{d}_K{K}_{s2}'
        return A[key].astype(np.float64) if kind == 'zh' else A[key].astype(int)
    ylab = {'e12ev': y[ev], 'test': yt}

    # ---- 1. substitution on fresh jets
    sub = {}; boots = {}
    for b in B:
        sub[b] = {}
        for K in KS:
            if b == 'randS':
                chs = [(get(b, 'call', K, 'test', d) != ct) for d in range(N_RAND)]
                v, bt = rate(np.mean(chs, 0)); accs = [rate(get(b, 'call', K, 'test', d) == yt)[0] for d in range(N_RAND)]
                sub[b][str(K)] = dict(changed=ci(v * 100, bt * 100), changed_draws=[float(c.mean() * 100) for c in chs],
                                      acc_mean=float(np.mean(accs) * 100)); boots[(b, K)] = bt * 100; continue
            c = get(b, 'call', K, 'test'); v, bt = rate(c != ct); a, abt = rate(c == yt); _, nbt = rate(ct == yt)
            hz = np.maximum(get(b, 'zh', K, 'test'), 0); hn = np.maximum(Rt['z'], 0)
            r2 = 1 - ((hn - hz) ** 2).mean(0) / np.maximum(hn.var(0), 1e-12)
            single = [float((B[b][f'single_K{K}_z{j}'] != ct).mean() * 100) for j in range(16)]
            sub[b][str(K)] = dict(changed=ci(v * 100, bt * 100), acc=ci(a * 100, abt * 100), dacc_vs_net=ci((a - ACC_T) * 100, (abt - nbt) * 100),
                                  mcnemar_vs_net=mcnemar(ct == yt, c == yt), r2=[round(float(q), 4) for q in r2], r2_mean=float(r2.mean()),
                                  changed_e12ev=float((get(b, 'call', K, 'e12ev') != call[ev]).mean() * 100), single_changed=single)
            boots[(b, K)] = bt * 100
            log(f'substitution {b:8s} K={K:2d}: changed {v*100:5.2f} % [{np.percentile(bt*100, 2.5):.2f}, {np.percentile(bt*100, 97.5):.2f}], '
                f'acc {a*100:.2f} % (net {ACC_T*100:.2f}), McNemar p {sub[b][str(K)]["mcnemar_vs_net"]["p"]:.2g}, R2 {r2.mean():.3f}, '
                f'e12ev changed {sub[b][str(K)]["changed_e12ev"]:.2f} %')
    out['substitution'] = sub
    diffs = {}
    for K in KS:
        for b in B:
            if b == 'S' or ('S', K) not in boots: continue
            pt_ = sub[b][str(K)]['changed']['v'] - sub['S'][str(K)]['changed']['v']; bt = boots[(b, K)] - boots[('S', K)]
            diffs[f'{b}-S_K{K}'] = dict(ci(pt_, bt), p=p_boot(bt))
    out['substitution_diff_changed'] = diffs
    for k, v in diffs.items(): log(f'  calls changed difference {k}: {v["v"]:+.2f} points [{v["lo"]:+.2f}, {v["hi"]:+.2f}], p {v["p"]:.3g}')
    # single-neuron S vs hand: McNemar on 'call changed', Holm over 16
    if 'hand' in B and 'S' in B:
        sg = {}
        for K in (8, 32):
            ps, rows = [], []
            for j in range(16):
                a = B['S'][f'single_K{K}_z{j}'] != ct; h = B['hand'][f'single_K{K}_z{j}'] != ct
                m = mcnemar(~a, ~h); ps.append(m['p']); rows.append(dict(neuron=j, S=float(a.mean() * 100), hand=float(h.mean() * 100), **m))
            for r_, pa in zip(rows, holm(ps)): r_['p_holm'] = pa
            sg[str(K)] = rows
        out['single_S_vs_hand'] = sg

    # ---- 2. edits (no refit)
    ED = {}; EB = {}
    for k in EDITS:
        Je = EDITS[k][0]; sb, sa = f'{k}_b', f'{k}_a'
        dn = (np.maximum(NETZ[sa], 0) - np.maximum(NETZ[sb], 0)) / SDH
        cnb, cna = NETC[sb], NETC[sa]; yy = yt[Je]; isW = (yy == Wc)
        fn = cna != cnb
        ED[k] = dict(N=int(len(Je)), network=dict(flip=ci(*[q * 100 for q in rate(fn)]), WtoZ_before=ci(*[q * 100 for q in rate(cnb == Zc, isW)]),
                                                 WtoZ_after=ci(*[q * 100 for q in rate(cna == Zc, isW)]),
                                                 mean_abs_dz_sd=[float(v) for v in np.abs(dn).mean(0)]), bases={})
        _, bWZn = rate(cna == Zc, isW); _, bfn = rate(fn)
        for b in B:
            ED[k]['bases'][b] = {}
            for K in KS:
                if b == 'randS':
                    if K not in (8, 32): continue
                    vals = []
                    for d in range(N_RAND_Z):
                        df = (np.maximum(get(b, 'zh', K, sa, d), 0) - np.maximum(get(b, 'zh', K, sb, d), 0)) / SDH
                        s0, _, r0, _, _ = slope_corr(dn, df)
                        cfb, cfa = get(b, 'call', K, sb, d), get(b, 'call', K, sa, d)
                        vals.append(dict(slope=float(s0), r=float(r0), WtoZ_after=float((cfa[isW] == Zc).mean() * 100), flip=float((cfa != cfb).mean() * 100)))
                    ED[k]['bases'][b][str(K)] = dict(draws=vals, slope_mean=float(np.mean([v['slope'] for v in vals])))
                    continue
                df = (np.maximum(get(b, 'zh', K, sa), 0) - np.maximum(get(b, 'zh', K, sb), 0)) / SDH
                s0, sbt, r0, rbt, per = slope_corr(dn, df)
                # read-out calibration: the same regression across ordinary (unedited 'before') jets, i.e. the formula's
                # slope on the network for a real difference between jets; edit slope / this = calibrated slope
                c0, cbt, _, _, cper = slope_corr(np.maximum(NETZ[sb], 0) / SDH, np.maximum(get(b, 'zh', K, sb), 0) / SDH)
                cfb, cfa = get(b, 'call', K, sb), get(b, 'call', K, sa); ff = cfa != cfb
                agree = (ff == fn) & (~fn | (cfa == cna))
                wz, wzb = rate(cfa == Zc, isW); fl, flb = rate(ff); ag, agb = rate(agree)
                e = dict(slope=ci(s0, sbt), r=ci(r0, rbt), slope_minus_1_p=p_boot(sbt - 1),
                         calib_slope_ordinary_jets=ci(c0, cbt), slope_calibrated=ci(s0 / c0, sbt / cbt),
                         slope_calibrated_minus_1_p=p_boot(sbt / cbt - 1),
                         slope_calibrated_per_neuron=[float(p[0] / max(q[0], 1e-9)) for p, q in zip(per, cper)],
                         slope_per_neuron=[dict(ci(p[0], p[1]), r=float(p[2])) for p in per],
                         flip=ci(fl * 100, flb * 100), flip_minus_net=dict(ci((fl - rate(fn)[0]) * 100, (flb - bfn) * 100), p=p_boot(flb - bfn)),
                         agree_call_change=ci(ag * 100, agb * 100),
                         WtoZ_before=float((cfb[isW] == Zc).mean() * 100), WtoZ_after=ci(wz * 100, wzb * 100),
                         WtoZ_minus_net=dict(ci((wz - rate(cna == Zc, isW)[0]) * 100, (wzb - bWZn) * 100), p=p_boot(wzb - bWZn)))
                ED[k]['bases'][b][str(K)] = e; EB[(k, b, K)] = (sbt, per, wzb, flb, sbt / cbt)
                if K in (8, 32):
                    log(f'edit {k:5s} {b:8s} K={K:2d}: dz slope {s0:.3f} [{e["slope"]["lo"]:.3f}, {e["slope"]["hi"]:.3f}] r {r0:.3f}; '
                        f'ordinary-jet slope {c0:.3f}, calibrated {s0/c0:.3f} [{e["slope_calibrated"]["lo"]:.3f}, {e["slope_calibrated"]["hi"]:.3f}]; '
                        f'flip {fl*100:.2f} % (net {rate(fn)[0]*100:.2f}); call-change agreement {ag*100:.2f} %; W->Z after {wz*100:.1f} % '
                        f'(net {rate(cna == Zc, isW)[0]*100:.1f} %)')
    out['edits'] = ED
    # H0: S formula and hand formula have the same dz slope (paired bootstrap), per edit and K; per neuron with Holm
    H = {}
    for k in EDITS:
        for other in ('hand', 'hand_all', 'SP', 'initS'):
            if 'S' not in B or other not in B: continue
            for K in KS:
                sS, perS, wzS, flS, cS = EB[(k, 'S', K)]; sO, perO, wzO, flO, cO = EB[(k, other, K)]
                qS, qO = ED[k]['bases']['S'][str(K)]['slope_calibrated']['v'], ED[k]['bases'][other][str(K)]['slope_calibrated']['v']
                pS, pO = ED[k]['bases']['S'][str(K)]['slope']['v'], ED[k]['bases'][other][str(K)]['slope']['v']
                d = sS - sO
                pn = [p_boot(perS[j][1] - perO[j][1]) for j in range(16)]
                H[f'{k}_S-{other}_K{K}'] = dict(slope_diff=dict(ci(pS - pO, d), p=p_boot(d)),
                                                calibrated_slope_diff=dict(ci(qS - qO, cS - cO), p=p_boot(cS - cO)),
                                                abs_dev_from_1_diff=dict(ci(abs(pO - 1) - abs(pS - 1), np.abs(sO - 1) - np.abs(sS - 1)),
                                                                         p=p_boot(np.abs(sO - 1) - np.abs(sS - 1))),
                                                WtoZ_diff=dict(ci(ED[k]['bases']['S'][str(K)]['WtoZ_after']['v'] - ED[k]['bases'][other][str(K)]['WtoZ_after']['v'],
                                                                  (wzS - wzO) * 100), p=p_boot(wzS - wzO)),
                                                per_neuron_slope_diff=[dict(ci(perS[j][0] - perO[j][0], perS[j][1] - perO[j][1]), p=pn[j], p_holm=ph)
                                                                       for j, ph in zip(range(16), holm(pn))])
                if K == 32:
                    h_ = H[f'{k}_S-{other}_K{K}']
                    log(f'H0 {k} K=32 slope S - {other}: {h_["slope_diff"]["v"]:+.3f} [{h_["slope_diff"]["lo"]:+.3f}, {h_["slope_diff"]["hi"]:+.3f}] p {h_["slope_diff"]["p"]:.3g}; '
                        f'calibrated {h_["calibrated_slope_diff"]["v"]:+.3f} [{h_["calibrated_slope_diff"]["lo"]:+.3f}, {h_["calibrated_slope_diff"]["hi"]:+.3f}]; '
                        f'W->Z diff {h_["WtoZ_diff"]["v"]:+.1f} [{h_["WtoZ_diff"]["lo"]:+.1f}, {h_["WtoZ_diff"]["hi"]:+.1f}]; '
                        f'neurons with Holm p < 0.05: {[j for j, r_ in enumerate(h_["per_neuron_slope_diff"]) if r_["p_holm"] < 0.05]}')
    out['H0_S_vs_other'] = H
    # a32 crossing vs its matched control: flip-rate ratio, network and formulas
    if 'a32x' in EDITS:
        rr = {}
        for who in ['network'] + [b for b in B if b != 'randS']:
            for K in ((32,) if who == 'network' else KS):
                if who == 'network':
                    fx, fc = NETC['a32x_a'] != NETC['a32x_b'], NETC['a32c_a'] != NETC['a32c_b']
                else:
                    fx = get(who, 'call', K, 'a32x_a') != get(who, 'call', K, 'a32x_b'); fc = get(who, 'call', K, 'a32c_a') != get(who, 'call', K, 'a32c_b')
                vx, bx = rate(fx); vc, bc = rate(fc)
                rr[f'{who}_K{K}' if who != 'network' else who] = dict(cross=ci(vx * 100, bx * 100), control=ci(vc * 100, bc * 100),
                                                                     ratio=ci(vx / max(vc, 1e-9), bx / np.maximum(bc, 1e-9)))
        out['a32_cross_vs_control'] = rr
        for k_, v in rr.items():
            if k_ == 'network' or k_.endswith('K32'):
                log(f'a32 crossing / control flip rate {k_}: {v["cross"]["v"]:.2f} / {v["control"]["v"]:.2f} % = {v["ratio"]["v"]:.2f} [{v["ratio"]["lo"]:.2f}, {v["ratio"]["hi"]:.2f}]')

    # ---- 3. negative poles: class composition of the off side (network), formulas from the S and hand jobs
    neg = []
    zt = Rt['z']
    for j in range(16):
        o = np.argsort(zt[:, j])[:max(1, JT // 100)]; offm = zt[:, j] < 0
        comp = [float((yt[o] == c).mean() * 100) for c in range(5)]
        auc = []
        for c in range(5):
            if offm.sum() < 50 or (yt[offm] == c).sum() == 0 or (yt[offm] != c).sum() == 0: auc.append(None); continue
            rk = rankdata(-zt[offm, j]); pos = yt[offm] == c
            auc.append(float((rk[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())))
        row = dict(neuron=j, frac_off=float(offm.mean() * 100), off_by_class=[float((zt[yt == c, j] < 0).mean() * 100) for c in range(5)],
                   most_negative_1pct_class=comp, auc_minus_z_within_off=auc)
        for b in ('S', 'hand'):
            fj = OUT / f'{TAG}_{b}.json'
            if fj.exists():
                npj = json.loads(fj.read_text()).get('negative_pole', [])
                if j < len(npj) and not npj[j].get('skipped'):
                    row[f'{b}_r2'] = npj[j]['r2_test']; row[f'{b}_terms'] = [t[0] for t in npj[j]['terms'][:3]]
                pos_f = json.loads(fj.read_text())['formulas']['8'][j]['terms']; row[f'{b}_pos_terms'] = [t[0] for t in pos_f[:3]]
        neg.append(row)
        log(f'negative pole z{j}: off on {row["frac_off"]:.1f} % of jets; most negative 1 %: ' + ', '.join(f'{CL[c]} {comp[c]:.0f}' for c in range(5)) +
            f'; S R2 {row.get("S_r2", float("nan")):.3f} {row.get("S_terms")}; hand R2 {row.get("hand_r2", float("nan")):.3f} {row.get("hand_terms")}')
    out['negative_pole'] = neg
    (OUT / f'{TAG}_z_objects.json').write_text(json.dumps(out, indent=1))
    log(f'wrote {OUT / f"{TAG}_z_objects.json"}')
    figure(out)


def figure(out):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    col = dict(S='#1f6fb4', SP='#6aa6d8', hand='#d4651a', hand_all='#f0a35e', initS='#7a7a7a', randS='#b8b8b8')
    lab = dict(S='S objects (19)', SP='S + P (30)', hand='hand observables (87)', hand_all='all observables', initS='random-init S', randS='random terms on S')
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for b, d in out['substitution'].items():
        v = [d[str(K)]['changed']['v'] for K in KS]; lo = [d[str(K)]['changed']['lo'] for K in KS]; hi = [d[str(K)]['changed']['hi'] for K in KS]
        ax[0].errorbar(KS, v, yerr=[np.subtract(v, lo), np.subtract(hi, v)], color=col[b], marker='o', label=lab[b], capsize=3)
    ax[0].set_xscale('log', base=2); ax[0].set_xticks(KS, [str(k) for k in KS]); ax[0].set_xlabel('hinge terms per neuron K')
    ax[0].set_ylabel('calls changed vs network [%]'); ax[0].set_title(f'z replaced by formula, {out["N_test"]} fresh jets', loc='left', fontsize=11)
    ax[0].legend(fontsize=8, frameon=False)
    bs = [b for b in ('S', 'SP', 'hand', 'hand_all', 'initS') if b in out['substitution']]
    xs = np.arange(len(bs)); wdt = 0.27
    for i, (k, nm) in enumerate((('trans', '0.06 rad translation'), ('rot', '45° rotation'), ('a32x', 'a32 crossing 10→12 GeV'))):
        v = [out['edits'][k]['bases'][b]['32']['slope']['v'] for b in bs]
        lo = [out['edits'][k]['bases'][b]['32']['slope']['lo'] for b in bs]; hi = [out['edits'][k]['bases'][b]['32']['slope']['hi'] for b in bs]
        ax[1].bar(xs + (i - 1) * wdt, v, wdt, yerr=[np.subtract(v, lo), np.subtract(hi, v)], color=['#2b2b2b', '#8c8c8c', '#cfcfcf'][i], label=nm, capsize=2)
    ax[1].axhline(1, color='k', lw=0.8, ls='--'); ax[1].set_xticks(xs, [lab[b].split(' (')[0] for b in bs], fontsize=8, rotation=15)
    ax[1].set_ylabel('slope of formula Δz on network Δz'); ax[1].set_title('Edit response of the neurons, K = 32 (1 = network)', loc='left', fontsize=11)
    ax[1].legend(fontsize=8, frameon=False)
    tr = out['edits']['trans']; net_ = tr['network']['WtoZ_after']
    v = [tr['bases'][b]['32']['WtoZ_after']['v'] for b in bs]; lo = [tr['bases'][b]['32']['WtoZ_after']['lo'] for b in bs]
    hi = [tr['bases'][b]['32']['WtoZ_after']['hi'] for b in bs]
    ax[2].bar(xs, v, 0.6, yerr=[np.subtract(v, lo), np.subtract(hi, v)], color=[col[b] for b in bs], capsize=3)
    ax[2].axhspan(net_['lo'], net_['hi'], color='k', alpha=0.15); ax[2].axhline(net_['v'], color='k', lw=1)
    ax[2].text(len(bs) - 0.5, net_['v'] + 1.5, f'network {net_["v"]:.0f} %', ha='right', fontsize=9)
    ax[2].set_xticks(xs, [lab[b].split(' (')[0] for b in bs], fontsize=8, rotation=15); ax[2].set_ylabel('true W jets called Z after the edit [%]')
    ax[2].set_title('0.06 rad translation (mass unchanged), K = 32', loc='left', fontsize=11)
    fig.suptitle('JEDI-linear z neurons as hinge formulas: substitution and edit fidelity (95 % bootstrap intervals)', fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / f'{TAG}_z_objects.png', dpi=130); log(f'wrote {OUT / f"{TAG}_z_objects.png"}')


if ARGS.combine:
    combine()
else:
    for b in [s for s in ARGS.basis.split(',') if s]:
        run_basis(b)
log('done')
