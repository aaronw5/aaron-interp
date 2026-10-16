"""E6 (JEDI-linear 64 p, 3-feature; claim A4): where and how the network computes the jet mass.

Moments (pT [GeV], r = (Δη, Δφ) [rad], padded slots pT = 0):
  M0 = Σ pT,  M1 = Σ pT r⃗,  M2 = Σ pT r²,   m² ≈ Σ_{i<j} pT_i pT_j ΔR_ij² = M0·M2 − |M1|²   (massless, small angle)
  control quantities: G1 = Σ pT |r| (degree 1 in angle, 1 in pT, like m), Q2 = Σ pT² r² (2, 2, like m²), R1 = Σ_real |r| (1, 0)

0  approximation check against the exact four-vector mass Q['mass'].
1  readout: ridge (held-out R², fixed 50/50 split, alpha on a validation part of the train half) of the targets from
   P, g, S, h4, h5, z; linear in the units, and linear + pairwise products of the top-8 PCA components.
2  angle edit Δη, Δφ × k (k = 0.8, 0.9, 1.1, 1.25): M0 fixed, M1 × k, M2 × k², m ≈ × k.
3  contrast edit pT × c (c = 0.8, 0.9, 1.1, 1.25): M0, M1, M2 × c, m × c.
   per unit: exponents p (angle) and q (pT) from ΔU_j = A_j (k^p − 1) = B_j (c^q − 1) (per-jet amplitude, shared exponent):
   m, G1: (1, 1); m², M0·M2, Q2: (2, 2); M2: (2, 1); M0: (0, 1); angle-only: (·, 0).
   per-jet fits "U follows Q" (ΔU = s_j Δlog Q + t_j Δlog Q², both edits jointly) for Q = m, M2, M0, M2/M0.
   control edit: translation r → r + δ (δ = 0.03, 0.06 rad along η or φ, real particles): m exactly unchanged
   (longitudinal boost / azimuthal rotation), M0·M2 → M0·M2 + 2M0 δ·M1 + M0²δ², M2, G1 change.
   g-freeze: the same edits with g held at the unedited jet's g (cuts the P → g → l3 jet-context route).
   stage readouts of log m (trained on unedited jets) applied to all edited jets.
4  dense scans k, c = 0.5 … 2.0: window / monotone fractions per unit, and the mass-window neurons z0, z7, z12, z14, z10
   as piecewise-linear functions of m with per-jet offsets (knots in GeV), on each scan and on the stage readouts m̂.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp.jedi_core import Jedi, sample

OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'jedi'; OUT.mkdir(parents=True, exist_ok=True)
SEED = 6
t0 = time.time()
PER = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
x, y, Q = sample(per_class=PER, seed=SEED)
x = x.astype(np.float32); J = len(x)
CL = ['g', 'q', 'W', 'Z', 't']
net = Jedi(64)
STG = ('P', 'g', 'S', 'h4', 'h5', 'z')
UST = ('P', 'S', 'h4', 'h5', 'z')
KEEP = STG + ('logits',)
res = dict(N=J, per_class=PER, seed=SEED)


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- moments and edits
def moments(xx):
    xx = np.asarray(xx, np.float64); pt, eta, phi = xx[..., 0], xx[..., 1], xx[..., 2]; rl = pt > 0
    M0 = pt.sum(1); M1 = np.stack([(pt * eta).sum(1), (pt * phi).sum(1)], 1); r2 = eta ** 2 + phi ** 2; r = np.sqrt(r2)
    M2 = (pt * r2).sum(1)
    p4 = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), pt * np.cosh(eta)], -1).sum(1)
    m = np.sqrt(np.maximum(p4[:, 3] ** 2 - (p4[:, :3] ** 2).sum(1), 0.0))
    return dict(M0=M0, M1e=M1[:, 0], M1p=M1[:, 1], M1=np.hypot(M1[:, 0], M1[:, 1]), M2=M2, M0M2=M0 * M2, m2=m ** 2, m=m,
                logm=np.log(np.maximum(m, 1e-3)), G1=(pt * r).sum(1), Q2=(pt ** 2 * r2).sum(1), R1=(r * rl).sum(1),
                R2u=(r2 * rl).sum(1), N=rl.sum(1).astype(float), girth2=M2 / M0, m2a=M0 * M2 - (M1 ** 2).sum(1))


def edit_angle(xx, k):
    e = xx.copy(); e[..., 1:] *= np.float32(k); return e


def edit_pt(xx, c):
    e = xx.copy(); e[..., 0] *= np.float32(c); return e


def edit_shift(xx, d_eta, d_phi):
    e = xx.copy(); rl = e[..., 0] > 0
    e[..., 1] += np.float32(d_eta) * rl; e[..., 2] += np.float32(d_phi) * rl; return e


M = moments(x)
# ---------------------------------------------------------------- 0. approximation
ma = np.sqrt(np.maximum(M['m2a'], 0)); mq = Q['mass'].astype(np.float64); ok = mq > 1.0
cr = lambda a, b: float(np.corrcoef(a, b)[0, 1])
res['approx'] = dict(n_ok=int(ok.sum()), corr_m=cr(ma[ok], mq[ok]), corr_m2=cr(M['m2a'][ok], mq[ok] ** 2),
                     median_ratio_m=float(np.median(ma[ok] / mq[ok])), p05_ratio=float(np.percentile(ma[ok] / mq[ok], 5)),
                     p95_ratio=float(np.percentile(ma[ok] / mq[ok], 95)), own_vs_Q_maxdiff=float(np.abs(M['m'] - mq).max()),
                     M1sq_over_M0M2_median=float(np.median((M['M1'] ** 2 / M['M0M2'])[ok])),
                     M1sq_over_M0M2_p95=float(np.percentile((M['M1'] ** 2 / M['M0M2'])[ok], 95)),
                     corr_m_G1=cr(M['G1'], mq), corr_m_sqrtM0M2=cr(np.sqrt(M['M0M2']), mq), corr_m_sqrtQ2=cr(np.sqrt(M['Q2']), mq),
                     corr_m_M2=cr(M['M2'], mq), corr_m_M0=cr(M['M0'], mq),
                     factor2_formula_median_ratio=float(np.median(np.sqrt(2 * np.maximum(M['m2a'], 0))[ok] / mq[ok])))
log(f'{J} jets ({PER}/class, seed {SEED}); jets with m > 1 GeV {ok.sum()}')
log('0 approx: sqrt(M0M2-|M1|^2) vs exact m: corr {corr_m:.6f}, median ratio {median_ratio_m:.5f} (5-95%: {p05_ratio:.4f}-{p95_ratio:.4f}); '
    'corr(M0M2-|M1|^2, m^2) {corr_m2:.6f}; |M1|^2/(M0M2) median {M1sq_over_M0M2_median:.2e} p95 {M1sq_over_M0M2_p95:.2e}; '
    'own m vs Q[mass] max diff {own_vs_Q_maxdiff:.3g} GeV; corr(m, .): G1 {corr_m_G1:.4f}, sqrt(M0M2) {corr_m_sqrtM0M2:.5f}, '
    'sqrt(Q2) {corr_m_sqrtQ2:.4f}, M2 {corr_m_M2:.4f}, M0 {corr_m_M0:.4f}; '
    'with factor 2 (PAPERS.md form) median ratio {factor2_formula_median_ratio:.4f}'.format(**res['approx']))

lm0, lm2 = np.log(M['M0']), np.log(M['M2'])
res['spread'] = dict(cv_M0=float(M['M0'].std() / M['M0'].mean()), sd_half_logM0=float(0.5 * lm0.std()), sd_half_logM2=float(0.5 * lm2.std()),
                     r2_logm_from_logM2=cr(M['logm'], lm2) ** 2,
                     r2_logm_from_logM0_logM2=float(1 - np.var(M['logm'] - np.c_[lm0, lm2, np.ones(J)] @ np.linalg.lstsq(np.c_[lm0, lm2, np.ones(J)], M['logm'], rcond=None)[0]) / np.var(M['logm'])))
log('   spread: cv(M0) {cv_M0:.3f}; sd(0.5 log M0) {sd_half_logM0:.3f} vs sd(0.5 log M2) {sd_half_logM2:.3f}; R²(log m ~ log M2) {r2_logm_from_logM2:.4f}, R²(log m ~ log M0 + log M2) {r2_logm_from_logM0_logM2:.4f}'.format(**res['spread']))
R0 = net.run(x, keep=KEEP)
call0 = R0['logits'].argmax(1)
res['acc'] = float((call0 == y).mean()); log(f'accuracy {res["acc"]:.4f}')
chk = net.run_jetwise(x, {'g': (np.ones(64), R0['g'])}, keep=KEEP)
res['jetwise_identity_maxdiff'] = {k: float(np.abs(chk[k] - R0[k]).max()) for k in KEEP}
log('run_jetwise with g := own g, max |diff| per stage:', res['jetwise_identity_maxdiff'])

# ---------------------------------------------------------------- 1. readout
rng = np.random.default_rng(SEED)
perm = rng.permutation(J); tr, te = np.sort(perm[:J // 2]), np.sort(perm[J // 2:])
trf, trv = tr[: len(tr) * 3 // 4], tr[len(tr) * 3 // 4:]
TGT = ('M0', 'M1', 'M2', 'M0M2', 'm2', 'm', 'logm', 'G1', 'Q2', 'R1')
ALPHAS = 10.0 ** np.arange(-4, 4.1, 0.5)


class Ridge:
    """standardised ridge; alpha chosen on a validation part of the training half, then refit on the whole half"""

    def __init__(self, X, t, idx_fit, idx_val, idx_all):
        self.mu = X[idx_all].mean(0); self.sd = X[idx_all].std(0); self.sd[self.sd < 1e-9] = np.inf
        Z = (X - self.mu) / self.sd; self.tm = t[idx_all].mean(); self.ts = t[idx_all].std()
        tt = (t - self.tm) / self.ts; best = None
        for a in ALPHAS:
            w = self._fit(Z[idx_fit], tt[idx_fit], a); e = ((Z[idx_val] @ w - tt[idx_val]) ** 2).mean()
            if best is None or e < best[0]: best = (e, a)
        self.alpha = best[1]; self.w = self._fit(Z[idx_all], tt[idx_all], self.alpha)

    @staticmethod
    def _fit(Z, t, a):
        Zc = Z - Z.mean(0); return np.linalg.solve(Zc.T @ Zc + a * len(Z) * np.eye(Z.shape[1]), Zc.T @ (t - t.mean()))

    def __call__(self, X):
        return ((X - self.mu) / self.sd) @ self.w * self.ts + self.tm


def r2(p, t):
    return float(1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum())


class PCAProd:
    """[units, products of the top-K PCA components (squares included)]; PCA fitted on the training half"""

    def __init__(self, X, idx, K=8):
        self.mu = X[idx].mean(0); sd = X[idx].std(0); sd[sd < 1e-9] = np.inf; self.sd = sd
        Z = (X[idx] - self.mu) / self.sd
        _, s, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
        self.K = min(K, int((s > 1e-8).sum())); self.V = Vt[:self.K].T; self.iu = np.triu_indices(self.K)

    def __call__(self, X):
        C = ((X - self.mu) / self.sd) @ self.V
        return np.concatenate([X, (C[:, :, None] * C[:, None, :])[:, self.iu[0], self.iu[1]]], 1)


res['readout'] = {}; readers = {}
log('1 readout held-out R^2 (linear | linear + PCA-pair products, K=8):')
log('   stage ' + ' '.join(f'{t:>13s}' for t in TGT))
for st in STG:
    X = R0[st]; Xn = PCAProd(X, tr)(X); row = {}
    for t in TGT:
        tv = M[t]; lin = Ridge(X, tv, trf, trv, tr); nl = Ridge(Xn, tv, trf, trv, tr)
        row[t] = dict(lin=r2(lin(X)[te], tv[te]), nonlin=r2(nl(Xn)[te], tv[te]), alpha_lin=lin.alpha, alpha_nl=nl.alpha)
        if t in ('logm', 'm'): readers[(st, t)] = lin
    res['readout'][st] = row
    log(f'   {st:5s} ' + ' '.join(f'{row[t]["lin"]:6.3f}|{row[t]["nonlin"]:6.3f}' for t in TGT))
F0 = np.stack([M[k] for k in ('M0', 'M1e', 'M1p', 'M2', 'N')], 1); iu5 = np.triu_indices(5)
F0n = np.concatenate([F0, (F0[:, :, None] * F0[:, None, :])[:, iu5[0], iu5[1]]], 1); row = {}
for t in TGT:
    row[t] = dict(lin=r2(Ridge(F0, M[t], trf, trv, tr)(F0)[te], M[t][te]), nonlin=r2(Ridge(F0n, M[t], trf, trv, tr)(F0n)[te], M[t][te]))
res['readout']['moments(M0,M1,M2,N)'] = row
log('   mom   ' + ' '.join(f'{row[t]["lin"]:6.3f}|{row[t]["nonlin"]:6.3f}' for t in TGT) + '   (inputs: M0, M1η, M1φ, M2, N | + their products)')

# ---------------------------------------------------------------- 2/3. edits
KS = (0.8, 0.9, 1.1, 1.25); CS = (0.8, 0.9, 1.1, 1.25)
SH = (('eta', 0.03), ('eta', 0.06), ('phi', 0.03), ('phi', 0.06))
ED = {}
for k in KS: ED[f'k{k}'] = edit_angle(x, k)
for c in CS: ED[f'c{c}'] = edit_pt(x, c)
for ax, d in SH: ED[f'{ax}{d}'] = edit_shift(x, d if ax == 'eta' else 0, d if ax == 'phi' else 0)
RE, RF, ME = {}, {}, {}
gfreeze = {'g': (np.ones(64), R0['g'])}
for name, xe in ED.items():
    RE[name] = net.run(xe, keep=KEEP); RF[name] = net.run_jetwise(xe, gfreeze, keep=KEEP); ME[name] = moments(xe)
log(f'edits run ({len(ED)}, free and g-frozen), {time.time()-t0:.0f}s')
res['edit_moments'] = {name: {q: float(np.median(ME[name][q] / np.maximum(M[q], 1e-9))) for q in ('M0', 'M2', 'M0M2', 'm', 'G1', 'Q2')}
                       for name in ED}
log('   median ratio edited/original: ' + '; '.join(f'{n}: m {v["m"]:.3f} M2 {v["M2"]:.3f} M0M2 {v["M0M2"]:.3f} G1 {v["G1"]:.3f}'
                                                    for n, v in res['edit_moments'].items()))
res['calls'] = {}
for tag, RR in (('free', RE), ('gfrozen', RF)):
    res['calls'][tag] = {}
    for name in ED:
        ce = RR[name]['logits'].argmax(1)
        res['calls'][tag][name] = {CL[a]: {CL[b]: float((ce[y == a] == b).mean()) for b in range(5)} for a in range(5)}
res['calls']['none'] = {CL[a]: {CL[b]: float((call0[y == a] == b).mean()) for b in range(5)} for a in range(5)}
for tag in ('free', 'gfrozen'):
    for name in ED:
        c = res['calls'][tag][name]
        log(f'   calls {tag:7s} {name:8s}: W->W {c["W"]["W"]:.3f} W->Z {c["W"]["Z"]:.3f} Z->Z {c["Z"]["Z"]:.3f} Z->W {c["Z"]["W"]:.3f} '
            f'Z->t {c["Z"]["t"]:.3f} t->t {c["t"]["t"]:.3f} q->q {c["q"]["q"]:.3f} g->g {c["g"]["g"]:.3f}')
c = res['calls']['none']
log(f'   calls unedited        : W->W {c["W"]["W"]:.3f} W->Z {c["W"]["Z"]:.3f} Z->Z {c["Z"]["Z"]:.3f} Z->W {c["Z"]["W"]:.3f} '
    f'Z->t {c["Z"]["t"]:.3f} t->t {c["t"]["t"]:.3f} q->q {c["q"]["q"]:.3f} g->g {c["g"]["g"]:.3f}')

PGRID = np.round(np.arange(-3, 8.001, 0.05), 3)


def exponent(d, s):
    """d (J, npts) responses at scale factors s; ΔU_j = A_j (s^p − 1) with shared p -> (p, R²)"""
    tss = (d ** 2).sum(); best = None
    for p in PGRID:
        b = np.log(s) if abs(p) < 1e-9 else s ** p - 1
        rss = tss - ((d @ b) ** 2).sum() / (b @ b)
        if best is None or rss < best[0]: best = (rss, p)
    return float(best[1]), float(1 - best[0] / max(tss, 1e-30))


lk = np.log(np.array(KS)); lc = np.log(np.array(CS)); FOLLOW = ('m', 'M2', 'M0', 'girth2')
xq = {q: np.stack([np.log(np.maximum(ME[f'k{k}'][q], 1e-9) / np.maximum(M[q], 1e-9)) for k in KS] +
                  [np.log(np.maximum(ME[f'c{c}'][q], 1e-9) / np.maximum(M[q], 1e-9)) for c in CS], 1) for q in FOLLOW}


def unit_rows(st, RR):
    U0 = R0[st]; sd = U0.std(0); out = {}
    dK = np.stack([RR[f'k{k}'][st] - U0 for k in KS], 1); dC = np.stack([RR[f'c{c}'][st] - U0 for c in CS], 1)
    dT = np.stack([RR[f'{a}{d}'][st] - U0 for a, d in SH], 1)
    eK = (dK * lk[None, :, None]).sum(1) / (lk ** 2).sum(); eC = (dC * lc[None, :, None]).sum(1) / (lc ** 2).sum()
    D = np.concatenate([dK, dC], 1)
    for u in np.flatnonzero(sd > 1e-6):
        d = D[:, :, u]; tss = (d ** 2).sum(); fits = {}
        for q in FOLLOW:
            A = np.stack([xq[q], xq[q] ** 2], 2)
            w = np.linalg.solve(np.einsum('jpa,jpb->jab', A, A) + 1e-12 * np.eye(2), np.einsum('jpa,jp->ja', A, d)[..., None])[..., 0]
            fits[q] = float(1 - ((d - np.einsum('jpa,ja->jp', A, w)) ** 2).sum() / max(tss, 1e-30))
        rssf = 0.0
        for blk, xv in ((slice(0, 4), lk), (slice(4, 8), lc)):
            A = np.stack([xv, xv ** 2], 1); rssf += ((d[:, blk] - d[:, blk] @ (A @ np.linalg.pinv(A)).T) ** 2).sum()
        fits['free'] = float(1 - rssf / max(tss, 1e-30))
        p, pr2 = exponent(dK[:, :, u], np.array(KS)); q, qr2 = exponent(dC[:, :, u], np.array(CS))
        mk_, mc_ = eK[:, u].mean(), eC[:, u].mean()
        out[int(u)] = dict(unit=int(u), sd=float(sd[u]), on=float((U0[:, u] > 0).mean()) if st != 'z' else None,
                           dk125=float(dK[:, 3, u].mean() / sd[u]), dk08=float(dK[:, 0, u].mean() / sd[u]),
                           dk125_abs=float(np.abs(dK[:, 3, u]).mean() / sd[u]),
                           dc125=float(dC[:, 3, u].mean() / sd[u]), dc08=float(dC[:, 0, u].mean() / sd[u]),
                           dc125_abs=float(np.abs(dC[:, 3, u]).mean() / sd[u]),
                           dshift={f'{a}{dd}': float(dT[:, i, u].mean() / sd[u]) for i, (a, dd) in enumerate(SH)},
                           dshift_abs={f'{a}{dd}': float(np.abs(dT[:, i, u]).mean() / sd[u]) for i, (a, dd) in enumerate(SH)},
                           e_k=float(mk_ / sd[u]), e_c=float(mc_ / sd[u]),
                           e_k_abs=float(np.abs(eK[:, u]).mean() / sd[u]), e_c_abs=float(np.abs(eC[:, u]).mean() / sd[u]),
                           rho=float(mk_ / mc_) if abs(mc_) > 1e-9 * sd[u] else None,
                           p=p, p_r2=pr2, q=q, q_r2=qr2, follow_r2=fits)
    return out


res['units'] = {st: unit_rows(st, RE) for st in UST}
res['units_gfrozen'] = {st: unit_rows(st, RF) for st in ('S', 'h4', 'h5', 'z')}
log(f'unit tables {time.time()-t0:.0f}s')

# P/S: per-unit linear models fitted on unedited jets (train half) -> predicted response to each edit (test half)
FSETS = {'M012N': ('M0', 'M1e', 'M1p', 'M2', 'N'), 'M012N+G1,Q2,R1,R2u': ('M0', 'M1e', 'M1p', 'M2', 'N', 'G1', 'Q2', 'R1', 'R2u')}
res['linmodel'] = {}
for st in ('P', 'S'):
    U0 = R0[st]; rows = []
    for u, e in res['units'][st].items():
        row = dict(unit=u)
        for nm, keys in FSETS.items():
            F = lambda MM: np.stack([MM[k] for k in keys], 1)
            X0 = F(M); w = np.linalg.lstsq(np.c_[X0[tr], np.ones(len(tr))], U0[tr, u], rcond=None)[0]
            pr = {}
            for name in ED:
                dpred = (F(ME[name])[te] - X0[te]) @ w[:-1]; dobs = RE[name][st][te, u] - U0[te, u]
                vo = ((dobs - dobs.mean()) ** 2).sum()
                pr[name] = dict(obs=float(dobs.mean() / e['sd']), pred=float(dpred.mean() / e['sd']),
                                obs_abs=float(np.abs(dobs).mean() / e['sd']),
                                slope=float((dpred @ dobs) / max(dpred @ dpred, 1e-30)) if dpred @ dpred > 1e-12 else None,
                                r2=float(1 - ((dobs - dpred) ** 2).sum() / vo) if vo > 1e-9 * len(te) * e['sd'] ** 2 else None)
            row[nm] = dict(fit_r2=r2(np.c_[X0[te], np.ones(len(te))] @ w, U0[te, u]), coef=w.tolist(), edits=pr)
        rows.append(row)
    res['linmodel'][st] = rows

# readout transfer: stage readouts of log m (trained on unedited jets, train half) applied to edited test jets
res['transfer'] = {}
for tag, RR in (('free', RE), ('gfrozen', RF)):
    res['transfer'][tag] = {}
    for st in STG:
        rd = readers[(st, 'logm')]; base = rd(R0[st][te])
        res['transfer'][tag][st] = {name: float(np.mean(rd(RR[name][st][te]) - base)) for name in ED}
res['transfer']['truth'] = {name: dict(dlogm=float(np.mean(ME[name]['logm'][te] - M['logm'][te])),
                                       dlog_sqrtM0M2=float(np.mean(0.5 * np.log(ME[name]['M0M2'][te] / M['M0M2'][te]))),
                                       dlog_G1=float(np.mean(np.log(ME[name]['G1'][te] / M['G1'][te]))),
                                       dlog_sqrtM2=float(np.mean(0.5 * np.log(ME[name]['M2'][te] / M['M2'][te]))),
                                       dlog_sqrtQ2=float(np.mean(0.5 * np.log(ME[name]['Q2'][te] / M['Q2'][te]))))
                             for name in ED}
log('3 readout transfer: mean Δ(log m̂) of each stage readout (free | g-frozen) vs quantities:')
for name in ED:
    t_ = res['transfer']['truth'][name]
    log(f'   {name:8s} Δlog m {t_["dlogm"]:+.3f} sqrt(M0M2) {t_["dlog_sqrtM0M2"]:+.3f} G1 {t_["dlog_G1"]:+.3f} sqrt(M2) {t_["dlog_sqrtM2"]:+.3f} '
        f'sqrt(Q2) {t_["dlog_sqrtQ2"]:+.3f} | ' + ' '.join(f'{st} {res["transfer"]["free"][st][name]:+.3f}/{res["transfer"]["gfrozen"][st][name]:+.3f}' for st in STG))

# ---------------------------------------------------------------- 4. dense scans
KD = np.round(np.arange(0.5, 2.001, 0.1), 2)
DST = ('S', 'h4', 'h5', 'z')
dense = {}
for ename, fn in (('k', edit_angle), ('c', edit_pt)):
    mm, UU, cc, hh = [], {st: [] for st in DST}, [], {st: [] for st in ('S', 'h4', 'h5')}
    for s in KD:
        xe = fn(x, s) if s != 1.0 else x
        r = net.run(xe, keep=DST + ('logits',)) if s != 1.0 else R0
        mm.append(moments(xe)['m']); cc.append(r['logits'].argmax(1))
        for st in DST: UU[st].append(r[st].astype(np.float32))
        for st in hh: hh[st].append(readers[(st, 'm')](r[st]))
    dense[ename] = dict(m=np.stack(mm, 1), U={st: np.stack(v, 1) for st, v in UU.items()}, call=np.stack(cc, 1),
                        mhat={st: np.stack(v, 1) for st, v in hh.items()})
log(f'dense scans done {time.time()-t0:.0f}s')
i1 = int(np.flatnonzero(KD == 1.0)[0])
res['dense_units'] = {}
for st in DST:
    rows = {}
    U = dense['k']['U'][st]; Uc = dense['c']['U'][st]; sd = R0[st].std(0); mk = dense['k']['m']
    for u in np.flatnonzero(sd > 1e-6):
        row = {}
        for ename, UUu, mmv in (('k', U[:, :, u], mk), ('c', Uc[:, :, u], dense['c']['m'])):
            seq = UUu.astype(np.float64); tol = 0.1 * sd[u]
            moving = (seq.max(1) - seq.min(1)) > tol
            am = seq.argmax(1); ends = np.maximum(seq[:, 0], seq[:, -1])
            peak = moving & (am > 0) & (am < len(KD) - 1) & (seq.max(1) > ends + tol)
            an = seq.argmin(1); trough = moving & (an > 0) & (an < len(KD) - 1) & (seq.min(1) < np.minimum(seq[:, 0], seq[:, -1]) - tol)
            dd = np.diff(seq, axis=1); t2 = 0.02 * sd[u]
            mono = moving & ((dd >= -t2).all(1) | (dd <= t2).all(1))
            mpk = mmv[np.arange(J), am]
            p_d, p_r2 = exponent(seq - seq[:, [i1]], KD)
            row[ename] = dict(moving=float(moving.mean()), peak=float(peak.mean()), trough=float(trough.mean()), mono=float(mono.mean()),
                              m_at_peak_median=float(np.median(mpk[peak])) if peak.sum() > 20 else None,
                              m_at_peak_q25=float(np.percentile(mpk[peak], 25)) if peak.sum() > 20 else None,
                              m_at_peak_q75=float(np.percentile(mpk[peak], 75)) if peak.sum() > 20 else None,
                              exp=p_d, exp_r2=p_r2)
        rows[int(u)] = row
    res['dense_units'][st] = rows

GRID = np.arange(20.0, 300.1, 2.5)


def pwl_fit(mv, zv, nk, grid=GRID, fixed=True):
    """z ≈ a_j + b·m + Σ_t w_t ReLU(m − t); mv, zv (groups, points); per-group offsets a_j if fixed, else one offset.
    Knots: greedy on GRID, then 3 rounds of coordinate refinement (exact least squares via the Gram matrix)."""
    A = np.concatenate([mv[..., None], np.maximum(mv[..., None] - grid, 0)], -1)
    if fixed: A = A - A.mean(1, keepdims=True); zd = zv - zv.mean(1, keepdims=True)
    else: A = A - A.reshape(-1, A.shape[-1]).mean(0); zd = zv - zv.mean()
    A = A.reshape(-1, A.shape[-1]); zd = zd.ravel(); sc = np.maximum(A.std(0), 1e-12); A = A / sc
    Gm = A.T @ A; bv = A.T @ zd; tss = zd @ zd

    def sse(ks):
        S_ = [0] + [1 + int(i) for i in ks]
        w = np.linalg.solve(Gm[np.ix_(S_, S_)] + 1e-9 * np.eye(len(S_)), bv[S_]); return tss - bv[S_] @ w, w / sc[S_]
    ks = []
    for _ in range(nk):
        ks = sorted(ks + [min((sse(ks + [i])[0], i) for i in range(len(grid)) if i not in ks)[1]])
    for _ in range(3):
        for j in range(len(ks)):
            rest = ks[:j] + ks[j + 1:]
            ks = sorted(rest + [min((sse(sorted(rest + [i]))[0], i) for i in range(len(grid)) if i not in rest)[1]])
    e, w = sse(ks)
    return dict(knots=[float(grid[i]) for i in ks], slopes=np.cumsum(w).tolist(), r2=float(1 - e / tss))


ZU = (0, 7, 12, 14, 10)
res['windows'] = {}
sub = np.arange(J); heavy = np.flatnonzero(y >= 2)
for u in ZU:
    row = {}
    for ename in ('k', 'c'):
        zz = dense[ename]['U']['z'][sub, :, u].astype(np.float64)
        for nk in (2, 3):
            row[f'{ename}scan_m_{nk}'] = pwl_fit(dense[ename]['m'][sub], zz, nk)
            for st in ('S', 'h4', 'h5'):
                row[f'{ename}scan_{st}hat_{nk}'] = pwl_fit(dense[ename]['mhat'][st][sub], zz, nk)
    for nk in (2, 3):
        row[f'natural_m_{nk}'] = pwl_fit(M['m'][None], R0['z'][None, :, u], nk, fixed=False)
        for ename in ('k', 'c'):
            row[f'{ename}scan_m_{nk}_WZt'] = pwl_fit(dense[ename]['m'][heavy], dense[ename]['U']['z'][heavy, :, u].astype(np.float64), nk)
    bins = np.arange(0, 305, 10.0); prof = {}
    for ename in ('k', 'c'):
        for cl in (2, 3, 4):
            mm_ = dense[ename]['m'][y == cl].ravel(); zv = dense[ename]['U']['z'][y == cl, :, u].ravel(); b = np.digitize(mm_, bins)
            prof[f'{ename}_{CL[cl]}'] = [float(zv[b == j].mean()) if (b == j).sum() > 30 else None for j in range(1, len(bins))]
    row['profile_bins'] = bins.tolist(); row['profile'] = prof
    res['windows'][f'z{u}'] = row
    for ename in ('k', 'c'):
        a = row[f'{ename}scan_m_3']
        log(f'4 z{u:<2d} {ename}-scan (per-jet offsets) knots {a["knots"]} GeV slopes/GeV {np.round(a["slopes"], 4).tolist()} R² {a["r2"]:.3f}; '
            f'2 knots {row[f"{ename}scan_m_2"]["knots"]} R² {row[f"{ename}scan_m_2"]["r2"]:.3f}; on h4 m̂: {row[f"{ename}scan_h4hat_3"]["knots"]} '
            f'R² {row[f"{ename}scan_h4hat_3"]["r2"]:.3f}; on S m̂: {row[f"{ename}scan_Shat_3"]["knots"]} R² {row[f"{ename}scan_Shat_3"]["r2"]:.3f}')
        a = row[f'{ename}scan_m_3_WZt']
        log(f'        {ename}-scan W/Z/t jets only: knots {a["knots"]} slopes {np.round(a["slopes"], 4).tolist()} R² {a["r2"]:.3f}')
    b_ = row['natural_m_3']
    log(f'        unedited jets: knots {b_["knots"]} slopes {np.round(b_["slopes"], 4).tolist()} R² {b_["r2"]:.3f}')
res['scan_calls'] = {}
for ename in ('k', 'c'):
    for cl in (2, 3):
        mm_ = dense[ename]['m'][y == cl].ravel(); cc_ = dense[ename]['call'][y == cl].ravel(); bins = np.arange(40, 165, 5.0); b = np.digitize(mm_, bins)
        res['scan_calls'][f'{ename}_{CL[cl]}'] = dict(bins=bins.tolist(), **{CL[t]: [float((cc_[b == j] == t).mean()) if (b == j).sum() > 30 else None
                                                                                   for j in range(1, len(bins))] for t in range(5)})

for cl in ('W', 'Z'):
    for ename in ('k', 'c'):
        sc_ = res['scan_calls'][f'{ename}_{cl}']; b0 = sc_['bins']
        log(f'   {cl} jets, {ename}-scan, fraction called W / Z by edited mass bin: ' + ' '.join(
            f'{b0[j]:.0f}:{sc_["W"][j]:.2f}/{sc_["Z"][j]:.2f}' for j in range(len(b0) - 1) if sc_['W'][j] is not None and 55 <= b0[j] <= 125))

# ---------------------------------------------------------------- printed summaries
def fmt(v, f='+6.2f'):
    return format(v, f) if v is not None else '   nan'


log('2/3 unit responses, sd units. dk/dc = mean ΔU at k or c = 1.25; p, q = angle / pT exponents (fit R²); '
    'follow R² m / M2 / M0 / girth2 / free; shift η0.06 φ0.06; dense-k peak / mono frac, m at peak; g-frozen p, q, dc')
res['followers'] = {}
for st in UST:
    fm, f2 = [], []
    for e in res['units'][st].values():
        mag = e['dk125_abs'] + e['dc125_abs']
        if mag < 0.2 or e['rho'] is None: continue
        f = e['follow_r2']
        if 0.75 <= e['rho'] <= 1.33 and f['m'] > f['M2']: fm.append(e['unit'])
        if 1.6 <= e['rho'] <= 2.5 and f['M2'] > f['m']: f2.append(e['unit'])
    res['followers'][st] = dict(m=fm, M2=f2)
    log(f'   {st}: units following m under both edits (0.75<=rho<=1.33, R²m > R²M2, |response|>0.2 sd): {fm}; following M2 (1.6<=rho<=2.5): {f2}')
for st in UST:
    rows = sorted(res['units'][st].values(), key=lambda e: -(e['dk125_abs'] + e['dc125_abs']))[:16 if st == 'z' else 14]
    for e in rows:
        f = e['follow_r2']; du = res['dense_units'].get(st, {}).get(e['unit'], {}).get('k', {})
        gf = res['units_gfrozen'].get(st, {}).get(e['unit'])
        log(f'   {st}{e["unit"]:<3d} dk {e["dk125"]:+.2f} dc {e["dc125"]:+.2f} rho {fmt(e["rho"])} | p {e["p"]:+.2f} ({e["p_r2"]:.2f}) q {e["q"]:+.2f} ({e["q_r2"]:.2f}) | '
            f'{f["m"]:.2f} {f["M2"]:.2f} {f["M0"]:.2f} {f["girth2"]:.2f} {f["free"]:.2f} | {e["dshift"]["eta0.06"]:+.2f} {e["dshift"]["phi0.06"]:+.2f} | '
            + (f'{du.get("peak", 0):.2f} {du.get("mono", 0):.2f} {fmt(du.get("m_at_peak_median"), "6.1f")} ' if du else '')
            + (f'| gf dk {gf["dk125"]:+.2f} dc {gf["dc125"]:+.2f} rho {fmt(gf["rho"])} p {gf["p"]:+.2f} q {gf["q"]:+.2f}' if gf else ''))
for st in ('P', 'S'):
    log(f'   {st} linear model (M0,M1,M2,N) fitted on unedited jets: fit R² | k1.25 obs/pred, per-jet R² | c1.25 obs/pred | eta0.06 obs/pred '
        f'|| extended model (+G1,Q2,R1,R2u) fit R² | k1.25 obs/pred R²')
    for row in sorted(res['linmodel'][st], key=lambda r: -r['M012N']['edits']['k1.25']['obs_abs'])[:12]:
        a = row['M012N']; e1, e2, e3 = a['edits']['k1.25'], a['edits']['c1.25'], a['edits']['eta0.06']
        b = row['M012N+G1,Q2,R1,R2u']; b1 = b['edits']['k1.25']
        log(f'   {st}{row["unit"]:<3d} {a["fit_r2"]:.3f} | {e1["obs"]:+.3f}/{e1["pred"]:+.3f} {fmt(e1["r2"])} | {e2["obs"]:+.3f}/{e2["pred"]:+.3f} | '
            f'{e3["obs"]:+.3f}/{e3["pred"]:+.3f} || {b["fit_r2"]:.3f} | {b1["obs"]:+.3f}/{b1["pred"]:+.3f} {fmt(b1["r2"])}')

# ---------------------------------------------------------------- 5. iso-mass edit and S0 freeze
# 5a iso-mass: pT × c and Δη, Δφ × 1/c -> m unchanged (m ∝ k·c), M0 × c, M2 × 1/c.  A unit that depends on m only stays flat;
#    a unit following M2 moves like −log c, one following M0 like +log c; a first-order linear mix of M0 and M2 is flat
#    to first order and moves at second order in (c − 1).
IC = (0.67, 0.8, 0.9, 1.1, 1.25, 1.5)
res['isomass'] = dict(c=IC, moments={}, transfer={}, units={}, calls={})
RI = {}
for c in IC:
    xe = edit_pt(edit_angle(x, 1 / c), c); MI = moments(xe); RI[c] = net.run(xe, keep=KEEP)
    res['isomass']['moments'][str(c)] = {q: float(np.median(MI[q] / np.maximum(M[q], 1e-9))) for q in ('m', 'M0', 'M2', 'M0M2', 'G1', 'Q2')}
    res['isomass']['transfer'][str(c)] = {st: float(np.mean(readers[(st, 'logm')](RI[c][st][te]) - readers[(st, 'logm')](R0[st][te]))) for st in STG}
    ce = RI[c]['logits'].argmax(1)
    res['isomass']['calls'][str(c)] = {CL[a]: {CL[b]: float((ce[y == a] == b).mean()) for b in range(5)} for a in range(5)}
for st in UST:
    sd = R0[st].std(0); res['isomass']['units'][st] = {}
    for u in np.flatnonzero(sd > 1e-6):
        res['isomass']['units'][st][int(u)] = {str(c): float((RI[c][st][:, u] - R0[st][:, u]).mean() / sd[u]) for c in IC}
log('5a iso-mass edit (pT × c, angles × 1/c): median m ratio ' + ' '.join(f'{c}:{res["isomass"]["moments"][str(c)]["m"]:.4f}' for c in IC))
log('   readout Δ(log m̂) per stage (truth 0; M2-only carrier would give −log c, M0-only +log c): ')
for st in STG:
    log(f'   {st:3s} ' + ' '.join(f'c{c}: {res["isomass"]["transfer"][str(c)][st]:+.3f}' for c in IC))
for cl in ('W', 'Z', 't'):
    log(f'   {cl} jets calls W/Z/t: ' + ' '.join(f'c{c}: {res["isomass"]["calls"][str(c)][cl]["W"]:.2f}/{res["isomass"]["calls"][str(c)][cl]["Z"]:.2f}/{res["isomass"]["calls"][str(c)][cl]["t"]:.2f}' for c in IC)
        + f' | unedited {res["calls"]["none"][cl]["W"]:.2f}/{res["calls"]["none"][cl]["Z"]:.2f}/{res["calls"]["none"][cl]["t"]:.2f}')
for st in ('S', 'h4', 'h5', 'z'):
    for kind in ('m', 'M2'):
        for u in res['followers'][st][kind]:
            v = res['isomass']['units'][st][u]
            log(f'   {st}{u:<3d} ({kind}-follower) iso-mass mean ΔU/sd: ' + ' '.join(f'{c}:{v[str(c)]:+.2f}' for c in IC)
                + f' | for scale: dk1.25 {res["units"][st][u]["dk125"]:+.2f} dc1.25 {res["units"][st][u]["dc125"]:+.2f}')
for st, u in (('S', 0), ('S', 50), ('S', 13), ('S', 43)):
    v = res['isomass']['units'][st][u]
    log(f'   {st}{u:<3d} iso-mass mean ΔU/sd: ' + ' '.join(f'{c}:{v[str(c)]:+.2f}' for c in IC))

# 5b S0 freeze: the pT edit with S0 (the S unit carrying M0; its pT response runs mostly through g) held at the unedited value
mask0 = np.zeros(64); mask0[0] = 1
RS0 = {f'c{c}': net.run_jetwise(ED[f'c{c}'], {'S': (mask0, R0['S'])}, keep=KEEP) for c in CS}
RS0.update({f'k{k}': net.run_jetwise(ED[f'k{k}'], {'S': (mask0, R0['S'])}, keep=KEEP) for k in KS})
res['S0freeze'] = dict(transfer={}, units={}, calls={})
for st in STG:
    res['S0freeze']['transfer'][st] = {n: float(np.mean(readers[(st, 'logm')](RS0[n][st][te]) - readers[(st, 'logm')](R0[st][te]))) for n in RS0}
for n in RS0:
    ce = RS0[n]['logits'].argmax(1)
    res['S0freeze']['calls'][n] = {CL[a]: {CL[b]: float((ce[y == a] == b).mean()) for b in range(5)} for a in range(5)}
log('5b S0 frozen at unedited value: readout Δ(log m̂) free / S0-frozen / g-frozen (c = 1.25; k = 1.25)')
for st in ('h4', 'h5', 'z'):
    log(f'   {st:3s} c1.25 {res["transfer"]["free"][st]["c1.25"]:+.3f} / {res["S0freeze"]["transfer"][st]["c1.25"]:+.3f} / {res["transfer"]["gfrozen"][st]["c1.25"]:+.3f}'
        f'   k1.25 {res["transfer"]["free"][st]["k1.25"]:+.3f} / {res["S0freeze"]["transfer"][st]["k1.25"]:+.3f} / {res["transfer"]["gfrozen"][st]["k1.25"]:+.3f}')
for st in ('h4', 'h5', 'z'):
    U0 = R0[st]; sd = U0.std(0); res['S0freeze']['units'][st] = {}
    dC = np.stack([RS0[f'c{c}'][st] - U0 for c in CS], 1); dK = np.stack([RS0[f'k{k}'][st] - U0 for k in KS], 1)
    eC = (dC * lc[None, :, None]).sum(1) / (lc ** 2).sum(); eK = (dK * lk[None, :, None]).sum(1) / (lk ** 2).sum()
    for u in res['followers'][st]['m'] + res['followers'][st]['M2']:
        e = res['units'][st][u]; ec, ek = eC[:, u].mean() / sd[u], eK[:, u].mean() / sd[u]
        res['S0freeze']['units'][st][u] = dict(e_c=float(ec), e_k=float(ek), rho=float(ek / ec) if abs(ec) > 1e-9 else None,
                                               dc125=float(dC[:, 3, u].mean() / sd[u]))
        log(f'   {st}{u:<3d} rho free {fmt(e["rho"])} | S0-frozen {fmt(res["S0freeze"]["units"][st][u]["rho"])} | g-frozen {fmt(res["units_gfrozen"][st][u]["rho"])}'
            f'   dc1.25 {e["dc125"]:+.2f} / {res["S0freeze"]["units"][st][u]["dc125"]:+.2f} / {res["units_gfrozen"][st][u]["dc125"]:+.2f}')
c_ = res['S0freeze']['calls']
log('   calls with S0 frozen: ' + ' '.join(f'{n}: W->W {c_[n]["W"]["W"]:.2f} W->Z {c_[n]["W"]["Z"]:.2f} Z->Z {c_[n]["Z"]["Z"]:.2f} Z->W {c_[n]["Z"]["W"]:.2f} Z->t {c_[n]["Z"]["t"]:.2f};' for n in RS0))
(OUT / 'e6_mass.json').write_text(json.dumps(res))
log(f'done {time.time()-t0:.0f}s -> {OUT / "e6_mass.json"}')
