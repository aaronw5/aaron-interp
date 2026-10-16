"""E12 (JEDI-linear 64 p, 3-feature; claims A5, B): the 16 neurons z written on the network's own second-round objects S.

Sample: sample(per_class=6000, seed=2) = 30,000 test jets; random split (rng 0) 2/3 fit (20,000) / 1/3 evaluation (10,000).
Targets: the 16 pre-ReLU neurons z (fixed-point network), clipped below at -0.2 * sd of their positive values on the fit
jets (as jetdistill/mars.py: the ReLU erases differences below 0).
Bases (columns of X):
  S       the 19 live S units (E1: mean ablation changes >= 1 % of calls)
  P       the 11 live P units (same rule)
  hand    the jet-level Q_ observables of full_test.npz (whole-jet block, shape block, soft drop; no per-particle,
          top-k-slot or subjet entries)
  hand_all  every Q_ observable (hinge family only; larger dictionary for the hand side)
  untrS   control: S of Jedi(64, untrained=True, seed=0), the 19 highest-variance units on the fit jets (only 1 unit varies:
          the redrawn weights with the trained number formats zero the rest, seeds 0-4 give 1-3)
          (the same redrawn weights in float arithmetic also give 2 varying units)
  initS   control (added, replaces the degenerate redraw as the working untrained control): S of a randomly initialised
          network of the same architecture (He-normal weights, zero biases, float arithmetic), 19 highest-variance units
  h4      diagnostic (added): the network's own h4 units = ReLU(W4 S + b4), cuts on combinations of S (ridge, hinge)
  raw     control: (pT, deta, dphi) of the 64 slots
  randS   control: K random terms on the 19 live S units instead of selected ones (10 draws)
Families per neuron, term budget K in {1, 2, 4, 8, 16, 32}:
  ridge   forward-selected linear terms (standardised), ridge refit (alpha 1)
  hinge   MARS-style: X, max(0, X - t), max(0, t - X) with t on the 5-95 % quantile grid (19 knots) and products of a chosen
          hinge with a hinge of one of the 10 columns most correlated with the residual; forward selection on 12,000
          fit jets, least-squares refit on all fit jets (as jetdistill/mars.py, without the validation stop)
  tree    regression tree with K leaves (best-first, sklearn max_leaf_nodes)
  poly    degree-2 polynomial, K forward-selected monomials x_i, x_i x_j (standardised), least-squares refit
Intervention: z replaced by max(0, formula) per jet inside the jitted forward (override of stage z, per-jet values
in the layer's internal shape (J, 16)); the network's own d7 makes the call. All 16 at once, and one at a time for z1, z0, z10, z9, z8, z13.
Metrics on the evaluation jets: % calls changed vs the network, accuracy, R^2 of relu(formula) vs relu(z) per neuron,
distinct units used over the 16 neurons.
Front-to-back: float + mask program (E3m) up to S, then the S formula, then the network's d7: % calls changed.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp.jedi_core import Jedi, sample

OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'jedi'; OUT.mkdir(parents=True, exist_ok=True)
LOGD = Path('logs'); LOGD.mkdir(exist_ok=True)
SMOKE = os.environ.get('E12_SMOKE') == '1'
PER = 600 if SMOKE else 6000
SEED = 2
KS = (1, 2, 4, 8, 16, 32)
KMAX = max(KS)
N_SEL, N_KNOTS, TOP_PROD = 12000, 19, 10
SINGLE = (1, 0, 10, 9, 8, 13)                 # largest E1 costs
N_RAND = 3 if SMOKE else 10
LOGF = open(LOGD / ('e12_smoke.log' if SMOKE else 'e12.log'), 'w')
t0 = time.time()


def log(*a):
    s = ' '.join(str(v) for v in a); print(s, flush=True); LOGF.write(s + '\n'); LOGF.flush()


# ------------------------------------------------------------------ network, checks
net = Jedi(64)
x0, y0, _ = sample(per_class=2000, seed=0)
acc0 = float((net.run(x0)['logits'].argmax(1) == y0).mean())
log(f'check: seed-0 sample 10,000 jets, accuracy {acc0*100:.2f} % (E1: 81.29 %)')

x, y, Q = sample(per_class=PER, seed=SEED); J = len(x)
R = net.run(x, keep=('P', 'S', 'z', 'logits'))
call = R['logits'].argmax(1)
perm = np.random.default_rng(0).permutation(J); nf = 2 * J // 3
fit, ev = np.sort(perm[:nf]), np.sort(perm[nf:])
xe, ye, ce = x[ev], y[ev], call[ev]
acc_net = float((ce == ye).mean())
log(f'sample: {J} jets (per_class {PER}, seed {SEED}); fit {len(fit)}, eval {len(ev)}; network accuracy on eval {acc_net*100:.2f} %, '
    f'on all {(call == y).mean()*100:.2f} %  [{time.time()-t0:.0f} s]')


def run_override(xx, ov, batch=5000):
    """logits with per-jet overrides {stage: (mask (units,), values (J, units))}; values reshaped to the layer's internal
    shape: (J, 1, units) for P, g (qs keeps the pooled axis); (J, units) for S, h4, h5, z (qs1 drops it; d4-d7 are 'bc,cC')"""
    out = []
    for i in range(0, len(xx), batch):
        xb = xx[i:i + batch]
        O = {k: (np.asarray(m, np.float32), np.asarray(v[i:i + batch], np.float32).reshape((len(xb), 1, -1) if k in ('P', 'g') else (len(xb), -1)))
             for k, (m, v) in ov.items()}
        r = net._f(net.inputs(xb, net.n), O); out.append(np.asarray(r['logits'], np.float64).reshape(len(xb), -1))
    return np.concatenate(out)


ONES16 = np.ones(16)
lo_id = run_override(xe, {'z': (ONES16, R['z'][ev])})
assert lo_id.shape == (len(ev), 5), lo_id.shape
d_id = float(np.abs(lo_id - R['logits'][ev]).max())
log(f'check: z override with the network\'s own z: max |d logit| {d_id:.3g}, calls changed {(lo_id.argmax(1) != ce).mean()*100:.3f} %')
lo_S = run_override(xe, {'S': (np.ones(64), R['S'][ev])})
log(f'check: S override with the network\'s own S: max |d logit| {np.abs(lo_S - R["logits"][ev]).max():.3g}')

import jax, jax.numpy as jnp


def bits(L, name):
    q = L[name].iq
    qq = q.quantizers[0] if hasattr(q, 'quantizers') else q
    qq = qq.quantizer if hasattr(qq, 'quantizer') else qq
    return np.asarray(qq.b) if hasattr(qq, 'b') else None


def mk(mode, nt):
    """the network nt's program: 'fixed' (HGQ quantizers), 'mask' (float + zero-bit masks, E3m), 'float' (no quantizers)"""
    L = nt.L
    KB = {k: (jnp.asarray(v[0], jnp.float32), jnp.asarray(v[1], jnp.float32)) for k, v in nt.KB.items()}
    MASK = {k: jnp.asarray(bits(L, k) > 0, jnp.float32) for k in ('l1', 'qs', 'l2', 'dd', 'l3', 'qs1', 'd4', 'd5', 'd6', 'd7') if bits(L, k) is not None}

    def q(name, t):
        if mode == 'fixed': return L[name].iq(t, training=False)
        if mode == 'mask' and name in MASK: return t * MASK[name]
        return t

    def dense(k, t, act=True):
        z = jnp.einsum(L[k].equation, q(k, t), KB[k][0]) + KB[k][1]
        return jax.nn.relu(z) if act and L[k].activation is not None else z

    def pool(k, a):
        return jnp.sum(q(k, a), axis=L[k].axes, keepdims=L[k].keepdims) * L[k].scale

    def f(xn):
        a = dense('l1', xn); g = dense('dd', pool('qs', a)); s = dense('l2', a)
        if mode == 'fixed':
            qs_, qd_ = L['add'].iq([s, g], training=False); t = qs_ + qd_
        else: t = s + g
        S = pool('qs1', dense('l3', t)); h = dense('d5', dense('d4', S))
        return S.reshape(len(xn), -1), dense('d7', jax.nn.relu(dense('d6', h, act=False))).reshape(len(xn), -1)
    return jax.jit(f)


def prog(mode, xx, nt=None, batch=5000):
    nt = nt or net; f = mk(mode, nt); Ss, Ls = [], []
    for i in range(0, len(xx), batch):
        s_, l_ = f(nt.inputs(xx[i:i + batch], 64)); Ss.append(np.asarray(s_, np.float64)); Ls.append(np.asarray(l_, np.float64))
    return np.concatenate(Ss), np.concatenate(Ls)


# ------------------------------------------------------------------ targets
Z = R['z']; Zf = Z[fit]
floor = np.array([-0.2 * (Zf[Zf[:, j] > 0, j].std() if (Zf[:, j] > 0).sum() > 50 else Zf[:, j].std()) for j in range(16)])
T = np.maximum(Z, floor)                 # clipped targets
Hev = np.maximum(Z[ev], 0)               # relu(z) on eval

# ------------------------------------------------------------------ bases
e1 = json.loads((OUT / 'e1_e3.json').read_text())['E1']
liveS = [int(k) for k in np.flatnonzero(np.array(e1['S']['changed']) > 0.01)]
liveP = [int(k) for k in np.flatnonzero(np.array(e1['P']['changed']) > 0.01)]
log(f'live S units ({len(liveS)}): {liveS}\nlive P units ({len(liveP)}): {liveP}')
netU = Jedi(64, untrained=True, seed=0)
SU = netU.run(x, keep=('S',))['S']
uU = [int(k) for k in np.argsort(-SU[fit].var(0))[:19]]
log(f'untrained S: units with nonzero variance {(SU[fit].std(0) > 1e-9).sum()}, 19 highest-variance {sorted(uU)}')
SUf, _ = prog('float', x, netU)
nUf = int((SUf[fit].std(0) > 1e-9).sum())
log(f'untrained S, same redrawn weights in float arithmetic (no quantizers): units with nonzero variance {nUf} (not used as a basis)')
del netU, SUf


def he_S(xx, seed=0, batch=5000):
    """S of a randomly initialised network of the same architecture: He-normal weights, zero biases, float arithmetic,
    the trained network's input scaling and pooling scales (l1 -> sum pool -> dd; l2; l3 on s + g; sum pool)"""
    rg = np.random.default_rng(seed); he = lambda i, o: rg.normal(0, np.sqrt(2 / i), (i, o)).astype(np.float32)
    W1, Wd, W2, W3 = he(3, 64), he(64, 64), he(64, 64), he(64, 64); out = []
    for i in range(0, len(xx), batch):
        xn = np.asarray(net.inputs(xx[i:i + batch], 64), np.float32)
        a = np.maximum(xn @ W1, 0); P = a.sum(1) * net.pool_scale[0]; g = np.maximum(P @ Wd, 0)
        u = np.maximum((np.maximum(a @ W2, 0) + g[:, None, :]) @ W3, 0); out.append((u.sum(1) * net.pool_scale[1]).astype(np.float64))
    return np.concatenate(out)


SUf = he_S(x)
uUf = [int(k) for k in np.argsort(-SUf[fit].var(0))[:19]]
log(f'random-init S (He-normal, zero bias, float): units with nonzero variance {(SUf[fit].std(0) > 1e-9).sum()}, 19 highest-variance {sorted(uUf)}')
H4 = net.run(x, keep=('h4',))['h4']; uh4 = [int(k) for k in np.flatnonzero(H4[fit].std(0) > 1e-9)]
log(f'h4 (diagnostic basis): {len(uh4)} units with nonzero variance')

HAND = ['sum_pt', 'log_sum_pt', 'mass', 'mass_over_sum_pt', 'mass_over_sum_pt_sq', 'max_dr', 'pt_dispersion', 'z_1st', 'z_2nd',
        'z_3rd', 'z_top5', 'girth', 'girth2', 'mean_eta', 'mean_eta2', 'mean_phi', 'mean_phi2', 'n_particles', 'n_pt_above_1',
        'n_pt_above_5', 'n_pt_above_10', 'n_pt_above_50', 'tau21', 'tau32', 'e2', 'C2', 'D2', 'e2_sq', 'LHA', 'centroid_offset',
        'planar_flow', 'eccentricity', 'width', 'lam1', 'lam2', 'orientation_deg', 'dr01', 'pt1_over_pt0', 'pt_balance01',
        'pt1_dr01', 'm01', 'm012', 'min_pair_mass', 'max_pair_mass', 'dr02', 'dr12', 'dr_min_012', 'dr_max_012', 'pt2_over_pt0',
        'mratio_min_012', 'mratio_max_012', 'n_dr_0_0p05', 'z_dr_0_0p05', 'n_dr_0p05_0p1', 'z_dr_0p05_0p1', 'n_dr_0p1_0p2',
        'z_dr_0p1_0p2', 'n_dr_0p2_0p4', 'z_dr_0p2_0p4', 'n_dr_0p4_up', 'z_dr_0p4_up', 'psi_0p1', 'psi_0p2', 'psi_0p3',
        'n_for_50pct', 'n_for_90pct', 'pt_entropy', 'e3', 'e4', 'C3', 'D3', 'N2', 'N3', 'M2', 'M3', 'C2_b2', 'D2_b2', 'tau1',
        'tau2', 'tau3', 'tau4', 'tau43', 'tau21_b2', 'sd_mass', 'sd_zg', 'sd_rg', 'sd_nremoved']


def obs_matrix(keys):
    keep = [k for k in keys if k in Q and np.isfinite(Q[k]).all() and np.ptp(Q[k][fit]) > 0]
    return np.stack([Q[k].astype(np.float64) for k in keep], 1), keep


Xh, hk = obs_matrix(HAND)
Xha, hka = obs_matrix(sorted(Q))
log(f'hand observables: {len(hk)} of {len(HAND)} listed (finite, varying); all Q_: {len(hka)} of {len(Q)}')
BASES = dict(
    S=(R['S'][:, liveS], [f'S{k}' for k in liveS]),
    P=(R['P'][:, liveP], [f'P{k}' for k in liveP]),
    hand=(Xh, hk),
    untrS=(SU[:, uU], [f'uS{k}' for k in uU]),
    initS=(SUf[:, uUf], [f'iS{k}' for k in uUf]),
    h4=(H4[:, uh4], [f'h4_{k}' for k in uh4]),
    raw=(x.reshape(J, -1).astype(np.float64), [f'{f}_{i}' for i in range(64) for f in ('pt', 'deta', 'dphi')]),
    hand_all=(Xha, hka),
)
FAMS = dict(S=('ridge', 'hinge', 'tree', 'poly'), P=('ridge', 'hinge', 'tree', 'poly'), hand=('ridge', 'hinge', 'tree', 'poly'),
            untrS=('ridge', 'hinge', 'tree', 'poly'), initS=('ridge', 'hinge', 'tree', 'poly'), h4=('ridge', 'hinge'), raw=('ridge', 'hinge', 'tree', 'poly'), hand_all=('hinge',))
sel = np.sort(np.random.default_rng(0).choice(len(fit), min(N_SEL, len(fit)), replace=False))   # selection rows (within fit)


# ------------------------------------------------------------------ term dictionaries
def knots(v):
    return np.unique(np.round(np.quantile(v, np.linspace(.05, .95, N_KNOTS)), 12))


def one(v, kind, t):
    return v if kind == 'lin' else np.maximum(0, v - t) if kind == 'gt' else np.maximum(0, t - v)


def tval(tm, X):
    """term value: tm = (kind, f, t) or (kind, f, t, kind2, f2, t2) on raw columns; ('std', f) / ('sq', f, g) use standardised"""
    if tm[0] == 'std': return X[:, tm[1]]
    if tm[0] == 'mono': return X[:, tm[1]] * X[:, tm[2]]
    v = one(X[:, tm[1]], tm[0], tm[2])
    return v * one(X[:, tm[4]], tm[3], tm[5]) if len(tm) > 3 else v


def hinge_dictionary(Xs_):
    """columns on the selection rows (float32) and their descriptors; per-feature column ranges (for products)"""
    blocks, desc, rng_ = [], [], {}
    for f in range(Xs_.shape[1]):
        v = Xs_[:, f]; ks = knots(v); c = [v]; d = [('lin', f, None)]
        for t in ks: c.append(np.maximum(0, v - t)); d.append(('gt', f, float(t)))
        for t in ks: c.append(np.maximum(0, t - v)); d.append(('lt', f, float(t)))
        c = np.stack(c, 1); ok = c.std(0) > 1e-12
        idx = np.flatnonzero(ok); start = sum(b.shape[1] for b in blocks)
        blocks.append(c[:, idx].astype(np.float32)); desc += [d[i] for i in idx]; rng_[f] = (start, start + len(idx))
    return np.concatenate(blocks, 1), desc, rng_


class Selector:
    """forward selection by squared-error decrease, orthogonalised against the chosen terms (Gram-Schmidt)"""
    def __init__(self, y):
        n = len(y); self.Qb = np.zeros((n, KMAX + 1)); self.Qb[:, 0] = 1 / np.sqrt(n); self.k = 1
        self.r = y - y.mean()

    def add(self, v):
        Qb = self.Qb[:, :self.k]
        for _ in range(2): v = v - Qb @ (Qb.T @ v)
        nv = np.linalg.norm(v)
        if nv < 1e-9: return None
        q = v / nv; self.Qb[:, self.k] = q; self.k += 1; self.r = self.r - q * (q @ self.r); return q


def select_columns(C, desc, y, kmax, products=None, X_sel=None):
    """C (n, M) float32 candidate columns; returns the chosen descriptors in order (nested prefixes = budgets)"""
    S_ = Selector(y); nC2 = (C.astype(np.float64) ** 2).sum(0); nn = nC2 - (C.sum(0, dtype=np.float64)) ** 2 / len(y)
    chosen, used = [], np.zeros(C.shape[1], bool)
    if products is not None: Xc = (X_sel - X_sel.mean(0)) / np.maximum(X_sel.std(0), 1e-12)
    for _ in range(kmax):
        r32 = S_.r.astype(np.float32)
        g = np.where((nn > 1e-5 * nC2) & ~used, (C.T @ r32).astype(np.float64) ** 2 / np.maximum(nn, 1e-30), 0)
        i = int(g.argmax()); best = (g[i], desc[i], C[:, i].astype(np.float64), i)
        if products is not None:
            cr = np.abs(Xc.T @ S_.r); top = np.argsort(-cr)[:TOP_PROD]
            Qb32 = S_.Qb[:, :S_.k].astype(np.float32)
            for tm in chosen:
                if len(tm) > 3 or tm[0] == 'lin': continue
                pv = tval(tm, X_sel).astype(np.float32)
                fs = [f for f in top if f != tm[1]]
                cols, dd = [], []
                for f in fs:
                    a, b = products[f]
                    for c in range(a, b):
                        if desc[c][0] == 'lin': continue
                        cols.append(c); dd.append(tm + desc[c])
                if not cols: continue
                Pm = C[:, cols] * pv[:, None]
                npm = (Pm.astype(np.float64) ** 2).sum(0) - ((Qb32.T @ Pm).astype(np.float64) ** 2).sum(0)
                gp = np.where(npm > 1e-5 * (Pm.astype(np.float64) ** 2).sum(0), (Pm.T @ r32).astype(np.float64) ** 2 / np.maximum(npm, 1e-30), 0)
                j = int(gp.argmax())
                if gp[j] > best[0]: best = (gp[j], dd[j], Pm[:, j].astype(np.float64), None)
        if best[0] <= 0: break
        q = S_.add(best[2])
        if q is None: break
        chosen.append(best[1])
        if best[3] is not None: used[best[3]] = True
        nn = nn - (C.T @ q.astype(np.float32)).astype(np.float64) ** 2
    return chosen


def select_poly(Xs_, y, kmax):
    """degree-2 monomials on standardised columns, gains from Gram matrices (no explicit monomial columns)"""
    n, d = Xs_.shape; iu, ju = np.triu_indices(d)
    S_ = Selector(y); X2 = Xs_ ** 2
    nl = (X2).sum(0) - Xs_.sum(0) ** 2 / n
    nq0 = (X2.T @ X2)[iu, ju]; nq = nq0 - ((Xs_.T @ Xs_)[iu, ju]) ** 2 / n
    nl0 = X2.sum(0)
    chosen, usedl, usedq = [], np.zeros(d, bool), np.zeros(len(iu), bool)
    for _ in range(kmax):
        r = S_.r
        gl = np.where((nl > 1e-7 * nl0) & ~usedl, (Xs_.T @ r) ** 2 / np.maximum(nl, 1e-30), 0)
        gq = np.where((nq > 1e-7 * nq0) & ~usedq, (((Xs_ * r[:, None]).T @ Xs_)[iu, ju]) ** 2 / np.maximum(nq, 1e-30), 0)
        if gl.max() >= gq.max():
            i = int(gl.argmax()); tm = ('std', i); v = Xs_[:, i]; usedl[i] = True; gbest = gl[i]
        else:
            i = int(gq.argmax()); tm = ('mono', int(iu[i]), int(ju[i])); v = Xs_[:, iu[i]] * Xs_[:, ju[i]]; usedq[i] = True; gbest = gq[i]
        if gbest <= 0: break
        q = S_.add(v)
        if q is None: continue
        chosen.append(tm)
        nl = nl - (Xs_.T @ q) ** 2; nq = nq - (((Xs_ * q[:, None]).T @ Xs_)[iu, ju]) ** 2
    return chosen


def lsq(B, y, alpha=0.0):
    if alpha == 0: return np.linalg.lstsq(B, y, rcond=None)[0]
    A = B.T @ B + alpha * np.diag([0] + [1] * (B.shape[1] - 1)); return np.linalg.solve(A, B.T @ y)


def feats_of(tm):
    if tm[0] in ('std',): return {tm[1]}
    if tm[0] == 'mono': return {tm[1], tm[2]}
    return {tm[1], tm[4]} if len(tm) > 3 else {tm[1]}


def name_term(tm, names):
    if tm[0] == 'std': return f'z({names[tm[1]]})'
    if tm[0] == 'mono': return f'z({names[tm[1]]})*z({names[tm[2]]})'
    def h(kind, f, t):
        return names[f] if kind == 'lin' else f'max(0,{names[f]}-{t:.4g})' if kind == 'gt' else f'max(0,{t:.4g}-{names[f]})'
    s = h(*tm[:3]); return s + '*' + h(*tm[3:]) if len(tm) > 3 else s


# ------------------------------------------------------------------ evaluation of a set of 16 formulas
def score(zhat_ev, units_used, label, single=True, extra=None):
    """zhat_ev (n_eval, 16) formula values (before max(0, .)); substitution through the network's own d7"""
    h = np.maximum(zhat_ev, 0)
    r2 = 1 - ((Hev - h) ** 2).mean(0) / np.maximum(Hev.var(0), 1e-12)
    lo = run_override(xe, {'z': (ONES16, h)}); c = lo.argmax(1)
    out = dict(changed=float((c != ce).mean() * 100), acc=float((c == ye).mean() * 100), r2=[round(float(v), 4) for v in r2],
               r2_mean=float(r2.mean()), units=len(units_used))
    if single:
        sg = {}
        for j in SINGLE:
            m = np.zeros(16); m[j] = 1
            cj = run_override(xe, {'z': (m, h)}).argmax(1); sg[f'z{j}'] = round(float((cj != ce).mean() * 100), 3)
        out['single'] = sg
    if extra: out.update(extra)
    return out


RES = dict(N=J, per_class=PER, seed=SEED, split='random 2/3 fit, 1/3 eval (rng 0)', n_fit=len(fit), n_eval=len(ev),
           acc_check_seed0=acc0, untrS_float_varying_units=nUf, acc_net_eval=acc_net, override_identity_maxdiff=d_id, liveS=liveS, liveP=liveP, untrS_units=uU,
           hand_observables=hk, n_hand_all=len(hka), floor=floor.tolist(), KS=list(KS), results={}, formulas={})
FORM = {}            # (basis, family) -> per neuron list of terms (descriptors)
COEF = {}            # (basis, family, K) -> per neuron coefficients


def fit_family(bname, fam):
    X, names = BASES[bname]; Xf, Xe = X[fit], X[ev]
    mu, sd = Xf.mean(0), np.maximum(Xf.std(0), 1e-12)
    res = {}; terms_all = []; t1 = time.time()
    if fam == 'tree':
        from sklearn.tree import DecisionTreeRegressor
        for K in KS:
            zh = np.zeros((len(ev), 16)); used = set(); nl = []
            for j in range(16):
                if K == 1: zh[:, j] = T[fit, j].mean(); nl.append(1); continue
                tr = DecisionTreeRegressor(max_leaf_nodes=K, random_state=0).fit(Xf, T[fit, j])
                zh[:, j] = tr.predict(Xe); used |= {int(f) for f in tr.tree_.feature if f >= 0}; nl.append(int(tr.get_n_leaves()))
            res[K] = score(zh, used, f'{bname}/{fam}/K{K}', extra=dict(leaves_mean=float(np.mean(nl)), units_list=sorted(names[u] for u in used)))
            log(f'  {bname:8s} {fam:5s} K={K:2d}: calls changed {res[K]["changed"]:6.2f} %, acc {res[K]["acc"]:.2f} %, '
                f'R2 mean {res[K]["r2_mean"]:.3f}, units {res[K]["units"]}, single {res[K]["single"]}  [{time.time()-t0:.0f} s]')
        return res
    Xsel = Xf[sel]
    if fam == 'hinge':
        C, desc, rng_ = hinge_dictionary(Xsel)
        for j in range(16):
            terms_all.append(select_columns(C, desc, T[fit[sel], j], KMAX, products=rng_, X_sel=Xsel))
        del C
    elif fam == 'ridge':
        Xs_ = ((Xsel - mu) / sd).astype(np.float32); desc = [('std', f) for f in range(X.shape[1])]
        for j in range(16):
            terms_all.append(select_columns(Xs_, desc, T[fit[sel], j], min(KMAX, X.shape[1])))
    elif fam == 'poly':
        Xs_ = (Xsel - mu) / sd
        for j in range(16):
            terms_all.append(select_poly(Xs_, T[fit[sel], j], KMAX))
    log(f'  {bname} {fam}: selection done in {time.time()-t1:.0f} s; terms per neuron {[len(t) for t in terms_all]}')
    Xfs, Xes = (Xf - mu) / sd, (Xe - mu) / sd
    FORM[(bname, fam)] = terms_all
    for K in KS:
        zh = np.zeros((len(ev), 16)); used = set(); nk = []; coefs = []
        for j in range(16):
            tl = terms_all[j][:K]; nk.append(len(tl))
            src_f, src_e = (Xfs, Xes) if fam in ('ridge', 'poly') else (Xf, Xe)
            Bf = np.stack([np.ones(len(fit))] + [tval(t, src_f) for t in tl], 1)
            Be = np.stack([np.ones(len(ev))] + [tval(t, src_e) for t in tl], 1)
            cf = lsq(Bf, T[fit, j], alpha=1.0 if fam == 'ridge' else 0.0)
            zh[:, j] = Be @ cf; coefs.append(cf.tolist())
            for t in tl: used |= feats_of(t)
        COEF[(bname, fam, K)] = coefs
        res[K] = score(zh, used, f'{bname}/{fam}/K{K}', extra=dict(terms_mean=float(np.mean(nk)), units_list=sorted(names[u] for u in used)))
        log(f'  {bname:8s} {fam:5s} K={K:2d}: calls changed {res[K]["changed"]:6.2f} %, acc {res[K]["acc"]:.2f} %, '
            f'R2 mean {res[K]["r2_mean"]:.3f}, units {res[K]["units"]}, terms/neuron {np.mean(nk):.1f}, single {res[K]["single"]}  [{time.time()-t0:.0f} s]')
    return res


for bname in ('S', 'hand', 'P', 'untrS', 'initS', 'raw', 'h4', 'hand_all'):
    for fam in FAMS[bname]:
        log(f'--- basis {bname}, family {fam}')
        RES['results'].setdefault(bname, {})[fam] = {str(k): v for k, v in fit_family(bname, fam).items()}
        json.dump(RES, open(OUT / ('e12_smoke.json' if SMOKE else 'e12_z_on_S.json'), 'w'), indent=1)

# ------------------------------------------------------------------ control: random terms on live S (10 draws)
log('--- control: random terms on the 19 live S units (hinge dictionary: X, max(0, X-t), max(0, t-X)) and random units (ridge)')
XS = BASES['S'][0]; XSf, XSe = XS[fit], XS[ev]
_, descS, _ = hinge_dictionary(XSf[sel])
rr = dict(hinge={str(K): [] for K in KS}, ridge={str(K): [] for K in KS})
for d in range(N_RAND):
    rg = np.random.default_rng(100 + d)
    picks = [rg.choice(len(descS), KMAX, replace=False) for _ in range(16)]
    upicks = [rg.permutation(len(liveS)) for _ in range(16)]
    for K in KS:
        zh = np.zeros((len(ev), 16)); zr = np.zeros((len(ev), 16)); used, usedr = set(), set()
        for j in range(16):
            tl = [descS[i] for i in picks[j][:K]]
            Bf = np.stack([np.ones(len(fit))] + [tval(t, XSf) for t in tl], 1); Be = np.stack([np.ones(len(ev))] + [tval(t, XSe) for t in tl], 1)
            zh[:, j] = Be @ lsq(Bf, T[fit, j]); used |= {t[1] for t in tl}
            uu = upicks[j][:min(K, len(liveS))]
            Bf = np.concatenate([np.ones((len(fit), 1)), XSf[:, uu]], 1); Be = np.concatenate([np.ones((len(ev), 1)), XSe[:, uu]], 1)
            zr[:, j] = Be @ lsq(Bf, T[fit, j]); usedr |= set(uu.tolist())
        a = score(zh, used, '', single=False); b = score(zr, usedr, '', single=False)
        rr['hinge'][str(K)].append(a); rr['ridge'][str(K)].append(b)
        log(f'  draw {d} K={K:2d}: random hinge terms {a["changed"]:6.2f} % (R2 {a["r2_mean"]:.3f}); random units linear {b["changed"]:6.2f} % (R2 {b["r2_mean"]:.3f})')
RES['results']['randS'] = {fam: {K: dict(changed_mean=float(np.mean([v['changed'] for v in L])), changed_min=float(np.min([v['changed'] for v in L])),
                                       changed_max=float(np.max([v['changed'] for v in L])), acc_mean=float(np.mean([v['acc'] for v in L])),
                                       r2_mean=float(np.mean([v['r2_mean'] for v in L])), draws=[round(v['changed'], 3) for v in L])
                                 for K, L in d_.items()} for fam, d_ in rr.items()}

# ------------------------------------------------------------------ front to back: float + mask program up to S, then the S formula
log('--- front to back: float + mask (E3m) up to S -> S formula -> network d7')
S_fx, L_fx = prog('fixed', xe); S_fm, L_fm = prog('mask', xe)
c_fx, c_fm = L_fx.argmax(1), L_fm.argmax(1)
lo_fmS = run_override(xe, {'S': (np.ones(64), S_fm)}); c_fmS = lo_fmS.argmax(1)
ftb = dict(fixed_rebuild_changed=float((c_fx != ce).mean() * 100), fixed_rebuild_S_maxdiff=float(np.abs(S_fx - R['S'][ev]).max()),
           float_mask_changed=float((c_fm != ce).mean() * 100), float_mask_acc=float((c_fm == ye).mean() * 100),
           float_mask_S_then_fixed_d4_d7_changed=float((c_fmS != ce).mean() * 100), float_mask_S_then_fixed_d4_d7_acc=float((c_fmS == ye).mean() * 100),
           S_float_mask_vs_fixed_rel_rms=float(np.sqrt(((S_fm - R['S'][ev]) ** 2).mean(0)[liveS].mean() / R['S'][ev][:, liveS].var(0).mean())))
log(f'  rebuilt fixed program vs network: {ftb["fixed_rebuild_changed"]:.3f} % calls changed (S max diff {ftb["fixed_rebuild_S_maxdiff"]:.3g})')
log(f'  float + mask whole program: {ftb["float_mask_changed"]:.2f} % calls changed, acc {ftb["float_mask_acc"]:.2f} %')
log(f'  float + mask S -> network d4..d7: {ftb["float_mask_S_then_fixed_d4_d7_changed"]:.2f} %, acc {ftb["float_mask_S_then_fixed_d4_d7_acc"]:.2f} %')
SfmL = S_fm[:, liveS]
for fam in ('hinge', 'ridge', 'poly', 'tree'):
    ftb[fam] = {}
    X = BASES['S'][0]; Xf = X[fit]; mu, sd = Xf.mean(0), np.maximum(Xf.std(0), 1e-12)
    for K in KS:
        zh = np.zeros((len(ev), 16))
        if fam == 'tree':
            from sklearn.tree import DecisionTreeRegressor
            for j in range(16):
                zh[:, j] = T[fit, j].mean() if K == 1 else DecisionTreeRegressor(max_leaf_nodes=K, random_state=0).fit(Xf, T[fit, j]).predict(SfmL)
        else:
            src = (SfmL - mu) / sd if fam in ('ridge', 'poly') else SfmL
            for j in range(16):
                tl = FORM[('S', fam)][j][:K]
                zh[:, j] = np.stack([np.ones(len(ev))] + [tval(t, src) for t in tl], 1) @ np.asarray(COEF[('S', fam, K)][j])
        c = run_override(xe, {'z': (ONES16, np.maximum(zh, 0))}).argmax(1)
        ftb[fam][str(K)] = dict(changed=float((c != ce).mean() * 100), acc=float((c == ye).mean() * 100))
        log(f'  float+mask S -> S {fam} K={K:2d} -> d7: {ftb[fam][str(K)]["changed"]:6.2f} % calls changed, acc {ftb[fam][str(K)]["acc"]:.2f} % '
            f'(network S: {RES["results"]["S"][fam][str(K)]["changed"]:.2f} %)')
RES['front_to_back'] = ftb

# ------------------------------------------------------------------ formulas (S hinge, S ridge, hand hinge) for later naming
for (b, fam) in (('S', 'hinge'), ('hand', 'hinge'), ('S', 'poly'), ('S', 'ridge')):
    names = BASES[b][1]
    RES['formulas'][f'{b}/{fam}'] = {str(K): [dict(neuron=j, intercept=COEF[(b, fam, K)][j][0],
                                                   terms=[[name_term(t, names), round(c, 6)] for t, c in zip(FORM[(b, fam)][j][:K], COEF[(b, fam, K)][j][1:])])
                                              for j in range(16)] for K in (4, 8, 16)}

# ------------------------------------------------------------------ summary: smallest K reaching a call-change level
summ = {}
for b, fd in RES['results'].items():
    for fam, kd in fd.items():
        ch = {int(K): (v['changed'] if 'changed' in v else v['changed_mean']) for K, v in kd.items()}
        summ[f'{b}/{fam}'] = dict(changed=ch, K_le3=min([K for K, c in ch.items() if c <= 3], default=None),
                                  K_le5=min([K for K, c in ch.items() if c <= 5], default=None),
                                  K_le10=min([K for K, c in ch.items() if c <= 10], default=None))
RES['summary'] = summ
log('--- summary: % calls changed (all 16 neurons substituted) by K; smallest K reaching <= 3 / 5 / 10 %')
for k, v in summ.items():
    log(f'  {k:16s} ' + ' '.join(f'{v["changed"][K]:6.2f}' for K in KS) + f'   K<=3%: {v["K_le3"]}, <=5%: {v["K_le5"]}, <=10%: {v["K_le10"]}')
RES['time_s'] = time.time() - t0
json.dump(RES, open(OUT / ('e12_smoke.json' if SMOKE else 'e12_z_on_S.json'), 'w'), indent=1)
log(f'done in {time.time()-t0:.0f} s')
