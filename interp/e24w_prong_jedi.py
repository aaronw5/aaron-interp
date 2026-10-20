"""E24w (claims A13, A3z, A4d, A2; from E24v): does JEDI-linear use two-prong structure beyond M0, M2, the radial profile and
the jet-level anisotropy that its context-only units S58 / S31 read?

Sample: interp.jedi_core.sample(per_class=3000, seed=2801), N = 15,000 test jets. Two-prong subset: exclusive-kt reclustering
of the real particles to exactly 2 subjets (pairwise kt distance min(pT_i, pT_j)^2 dR_ij^2, no beam distance, pT-weighted
(deta, dphi) recombination), softer subjet pT fraction z_s >= 0.1 with >= 2 particles, and tau21 < 0.5 (beta 1, k-means
axes, as jetdistill.observables).
Step 1 (naming of the anisotropy a). The nine first-layer half-plane statements HP (E14) pooled over particles: P_HP (J, 9).
  S58, S31 (exact network) fitted on P_HP by least squares (half the jets; R^2 on the other half) and by a MARS-style
  forward selection of hinges (<= 8 terms); also on all 64 P values (reference). a = first principal component of the two
  standardised linear fits (b = second component, reported). Direct edit: rigid rotation of every jet by 45 deg: |da|/sd(a)
  and |dS|/sd for S58, S31 and the radial objects S29, S1, S0, S19, S30.
Step 2 (edits; every edit keeps each particle's pT and r about the stored axis, so M0, M2, sum pT r, pT rank and N are
  exact; each edit ends with a rigid rotation of the whole jet by an angle beta found by grid (2 deg) + bisection so that
  |a - a0| <= tol = 0.01 sd(a), a computed with the network's exact l1 + pooling arithmetic):
  R   negative control: rigid rotation only, |beta| >= 30 deg.
  D   prong-destroying: each particle of the softer subjet gets theta_i ~ U(0, 2 pi) (up to 12 redraws until a beta exists).
  P   prong-preserving: rigid rotation of the softer subjet's particles about the jet axis by an angle alpha (|alpha| >= 10 deg)
      chosen so the jet mass m equals D's mass of the same jet (grid + bisection, |m_P / m_D - 1| <= 1 %), then beta.
      (Rotating one prong about the axis changes m; matching m to D's makes P and D differ in prong shape, not in m.)
  D'  scramble of every particle (theta_i ~ U(0, 2 pi)), then beta (up to 12 redraws).
  Analysis set: jets for which all four edits reach the tolerance.
Read-outs (paired, same jets): % calls changed vs unedited; D - P, D - R, P - R, D' - R with 95 % jet bootstrap (1,000) and
  exact McNemar; W -> Z and W -> q call flows (true W jets); |d logits| (class-centred, L2) with CI; tau21, m, a, S58, S31,
  radial S changes. JEDI only: the same with g held at its unedited value (exact hold of every jet-level input of u) and
  with S58, S31 held.
Networks: official JEDI-linear n64, the 8 X12 retrained copies, Deep Sets d64 s0, PFN 64-64-32 s0. Controls: boosted trees
  (sklearn HistGradientBoosting, 300 iterations) fit on sample(4000, 2802) minus the main jets, features recomputed on the
  edited jets: Th (held: log M0, log M2/M0, sum pT r / M0, N; 0 change by construction), Thm (Th + log m), Thmt (Thm + tau21),
  Tht (Th + tau21), Tt (tau21 only).
H0: P(call changed | D) = P(call changed | P) (no prong use beyond a, M0, M2, radial moments and m); H1: > (prong use).
  Exact McNemar on discordant pairs; Holm over the 11 networks; trees reported as references.
  ./job.sh e24w-run 'python -m interp.e24w_prong_jedi'  ->  $AI_RESULTS/jedi/e24w_prong_jedi.json, e24w_calls.npz
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
T0 = time.time()
CL = ['g', 'q', 'W', 'Z', 't']
HP = [29, 6, 40, 57, 3, 43, 30, 36, 46]
RADIAL_S = [29, 1, 0, 19, 30]
EDITS = ['R', 'P', 'D', "D'"]
NETS = ['jedi'] + [f'x12:{m}' for m in ('hgq-r150_s0', 'hgq-r150_s1', 'hgq-r150_s2', 'hgq_s0', 'hgq_s1', 'hgq_s2',
                                         'float_s0', 'float_s1')] + ['dsets_s0', 'pfn_s0']
TREES = ['Th', 'Thm', 'Thmt', 'Tht', 'Tt']


def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


# ------------------------------------------------------------------ subjets
def kt2(x, chunk=500):
    """exclusive-kt to 2 subjets; -> label (J, n): 0 harder subjet, 1 softer, -1 padding; ok (exactly 2 subjets)"""
    J, n = x.shape[:2]; lab_out = np.full((J, n), -1, np.int8); ok = np.zeros(J, bool); iu = np.triu(np.ones((n, n), bool), 1)
    for c in range(0, J, chunk):
        X = np.asarray(x[c:c + chunk], np.float64); m = len(X)
        P, E, F = X[..., 0].copy(), X[..., 1].copy(), X[..., 2].copy(); alive = P > 0
        lab = np.tile(np.arange(n), (m, 1)); na = alive.sum(1)
        while True:
            act = np.flatnonzero(na > 2)
            if len(act) == 0: break
            p, e, f, al = P[act], E[act], F[act], alive[act]
            d = np.minimum(p[:, :, None], p[:, None]) ** 2 * ((e[:, :, None] - e[:, None]) ** 2 + (f[:, :, None] - f[:, None]) ** 2)
            d = np.where(al[:, :, None] & al[:, None] & iu, d, np.inf)
            k = d.reshape(len(act), -1).argmin(1); i, j = k // n, k % n; r = np.arange(len(act))
            w = p[r, i] + p[r, j]
            E[act, i] = (p[r, i] * e[r, i] + p[r, j] * e[r, j]) / w; F[act, i] = (p[r, i] * f[r, i] + p[r, j] * f[r, j]) / w
            P[act, i] = w; alive[act, j] = False; na[act] -= 1
            L = lab[act]; lab[act] = np.where(L == j[:, None], i[:, None], L)
        two = na == 2
        for q in np.flatnonzero(two):
            s = np.flatnonzero(alive[q]); h, so = (s[0], s[1]) if P[q, s[0]] >= P[q, s[1]] else (s[1], s[0])
            real = X[q, :, 0] > 0
            lab_out[c + q] = np.where(~real, -1, np.where(lab[q] == h, 0, 1))
        ok[c:c + m] = two
    return lab_out, ok


# ------------------------------------------------------------------ geometry
def polar(x):
    x = np.asarray(x, np.float64); return np.hypot(x[..., 1], x[..., 2]), np.arctan2(x[..., 2], x[..., 1]), x[..., 0] > 0


def rot(x, mask, ang):
    """rotate the particles in mask (J, n) by ang (J,) about the stored axis (pT, r exact)"""
    r, th, real = polar(x); t = th + mask * np.asarray(ang, np.float64)[:, None]
    e = np.asarray(x, np.float64).copy()
    e[..., 1] = np.where(mask, r * np.cos(t), e[..., 1]); e[..., 2] = np.where(mask, r * np.sin(t), e[..., 2]); return e


def rot_grid(x, mask, angs):
    """(J, n, 3), angles (B,) -> (J * B, n, 3)"""
    r, th, real = polar(x); J, n = r.shape; B = len(angs)
    t = th[:, None] + mask[:, None] * angs[None, :, None]
    e = np.broadcast_to(np.asarray(x, np.float64)[:, None], (J, B, n, 3)).copy()
    mm = np.broadcast_to(mask[:, None], (J, B, n))
    e[..., 1] = np.where(mm, r[:, None] * np.cos(t), e[..., 1]); e[..., 2] = np.where(mm, r[:, None] * np.sin(t), e[..., 2])
    return e.reshape(J * B, n, 3)


def scramble(x, mask, rng):
    r, th, real = polar(x); t = np.where(mask, rng.uniform(0, 2 * np.pi, th.shape), th)
    e = np.asarray(x, np.float64).copy()
    e[..., 1] = np.where(mask, r * np.cos(t), e[..., 1]); e[..., 2] = np.where(mask, r * np.sin(t), e[..., 2]); return e


def wrap(a): return np.mod(a + np.pi, 2 * np.pi) - np.pi


# ------------------------------------------------------------------ exact first-layer + pooling (P) forward
class PF:
    def __init__(self, j, rows=4096):
        import jax, jax.numpy as jnp
        L = j.L; K = jnp.asarray(j.KB['l1'][0], jnp.float32); b = jnp.asarray(j.KB['l1'][1], jnp.float32)

        def f(xn):
            a = L['l1'].activation(jnp.einsum(L['l1'].equation, L['l1'].iq(xn, training=False), K) + b)
            return jnp.sum(L['qs'].iq(a, training=False), axis=L['qs'].axes, keepdims=L['qs'].keepdims) * L['qs'].scale
        self.f = jax.jit(f); self.j = j; self.rows = rows

    def __call__(self, x):
        out = []; R = self.rows
        for i in range(0, len(x), R):
            xb = np.asarray(x[i:i + R], np.float32); m = len(xb)
            if m < R: xb = np.concatenate([xb, np.zeros((R - m,) + xb.shape[1:], np.float32)])
            out.append(np.asarray(self.f(self.j.inputs(xb, self.j.n)), np.float64).reshape(R, -1)[:m])
        return np.concatenate(out)


def moments(x):
    x = np.asarray(x, np.float64); pt, eta, phi = x[..., 0], x[..., 1], x[..., 2]; r = np.hypot(eta, phi)
    p4 = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), pt * np.cosh(eta)], -1).sum(1)
    m = np.sqrt(np.maximum(p4[:, 3] ** 2 - (p4[:, :3] ** 2).sum(1), 0.0))
    return dict(M0=pt.sum(1), M2=(pt * r ** 2).sum(1), Mr=(pt * r).sum(1), N=(pt > 0).sum(1).astype(float), m=m)


def mass_grid(x, mask, angs, chunk=200):
    out = []
    for c in range(0, len(x), chunk):
        xe = rot_grid(x[c:c + chunk], mask[c:c + chunk], angs); out.append(moments(xe)['m'].reshape(-1, len(angs)))
    return np.concatenate(out)


# ------------------------------------------------------------------ root search over one angle
def pick_crossing(d, valid_mid, rng, t=0):
    """d (J, B) signed distance on a cyclic grid; crossing k = sign change between k and k+1; random valid crossing (t-th)"""
    s = d >= 0; cr = (s != np.roll(s, -1, 1)) & valid_mid
    keys = np.where(cr, rng.random(cr.shape), np.inf); o = np.argsort(keys, 1)
    t = min(t, d.shape[1] - 1); k = o[:, t]; ok = np.isfinite(keys[np.arange(len(d)), k])
    return k, ok, s[np.arange(len(d)), k]


def search(x, mask, target, fval, grid_fn, rng, exclude=0.0, B=180, iters=14, t=0):
    """find ang with fval(rot(x, mask, ang)) = target; grid_fn(x, mask, angs) -> (J, B) values"""
    angs = np.linspace(-np.pi, np.pi, B, endpoint=False); dg = angs[1] - angs[0]
    G = grid_fn(x, mask, angs); d = G - target[:, None]
    valid = np.abs(wrap(angs + dg / 2)) >= exclude
    k, ok, slo = pick_crossing(d, np.broadcast_to(valid, d.shape), rng, t)
    lo = angs[k]; hi = lo + dg
    for _ in range(iters):
        mid = (lo + hi) / 2; sm = (fval(rot(x, mask, mid)) - target) >= 0
        lo = np.where(sm == slo, mid, lo); hi = np.where(sm == slo, hi, mid)
    vl = fval(rot(x, mask, lo)); vh = fval(rot(x, mask, hi))
    ang = np.where(np.abs(vl - target) <= np.abs(vh - target), lo, hi)
    return wrap(ang), ok


def p_upper(n10, n01):
    """one-sided exact binomial P(X >= n10 | n10 + n01, 1/2) (H1: edit a changes more calls than edit b)"""
    from math import lgamma, log as ln, exp
    n = n10 + n01
    if n == 0: return 1.0
    lp = [lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1) - n * ln(2) for k in range(n10, n + 1)]
    mx = max(lp); return float(min(1.0, exp(mx) * sum(exp(v - mx) for v in lp)))


# ------------------------------------------------------------------ naming fits
def lstsq_fit(Z, y, tr):
    A = np.column_stack([Z, np.ones(len(Z))]); w = np.linalg.lstsq(A[tr], y[tr], rcond=None)[0]; return w, A @ w


def r2(y, p): return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def mars(Z, y, tr, te, maxt=8):
    cands = {}
    for k in range(Z.shape[1]):
        cands[f'P{HP[k]}'] = Z[:, k]
        for q in (.1, .25, .5, .75, .9):
            t = np.quantile(Z[tr, k], q)
            cands[f'max(0,P{HP[k]}-{t:.3g})'] = np.maximum(0, Z[:, k] - t); cands[f'max(0,{t:.3g}-P{HP[k]})'] = np.maximum(0, t - Z[:, k])
    chosen = []; cols = [np.ones(len(y))]; path = []
    for _ in range(maxt):
        best = None
        for nm, c in cands.items():
            if nm in chosen: continue
            A = np.column_stack(cols + [c]); w = np.linalg.lstsq(A[tr], y[tr], rcond=None)[0]; rss = ((A[tr] @ w - y[tr]) ** 2).sum()
            if best is None or rss < best[0]: best = (rss, nm, c)
        chosen.append(best[1]); cols.append(best[2])
        A = np.column_stack(cols); w = np.linalg.lstsq(A[tr], y[tr], rcond=None)[0]
        path.append(dict(term=best[1], r2_test=r2(y[te], A[te] @ w)))
    return path


# ------------------------------------------------------------------ main
def main():
    import interp.e24_azimuth_prong as E24
    from interp.jedi_core import sample, Jedi
    from jetdistill.config import RESULTS
    from sklearn.ensemble import HistGradientBoostingClassifier
    ci, mcnemar, holm, clean, boot_p = E24.ci, E24.mcnemar, E24.holm, E24.clean, E24.boot_p
    R = dict(node='E24w', sample='interp.jedi_core.sample(per_class=3000, seed=2801)',
             pool='interp.jedi_core.sample(per_class=4000, seed=2802) minus main jets', models={}, trees={})
    save = lambda: (OUT / 'e24w_prong_jedi.json').write_text(json.dumps(clean(R), indent=1))
    rng = np.random.default_rng(2803)
    x, y, Q = sample(3000, 2801); N = len(x); R['N'] = N
    j = Jedi(64); pf = PF(j)
    base = j.run(x, keep=('P', 'g', 'S', 'logits'))
    acc = float((base['logits'].argmax(1) == y).mean()); R['jedi_acc_full_sample'] = acc
    Pchk = pf(x[:2000]); R['pf_exact_max_abs_diff'] = float(np.max(np.abs(Pchk - base['P'][:2000])))
    log(f'sample N {N}; JEDI accuracy {acc:.4f}; P replica max |dP| {R["pf_exact_max_abs_diff"]:.2e}')
    # ---------------- step 1: naming fit
    Z = base['P'][:, HP]; tr = np.zeros(N, bool); tr[rng.permutation(N)[:N // 2]] = True; te = ~tr
    fits = {}; Fhat = {}
    for u in (58, 31):
        yy = base['S'][:, u]; w, p = lstsq_fit(Z, yy, tr); _, p64 = lstsq_fit(base['P'], yy, tr)
        Fhat[u] = p
        fits[f'S{u}'] = dict(sd=float(yy.std()), lin_HP_r2_test=r2(yy[te], p[te]), lin_P64_r2_test=r2(yy[te], p64[te]),
                            lin_HP_coef={f'P{h}': float(c) for h, c in zip(HP, w[:-1])}, intercept=float(w[-1]),
                            mars_HP_path=mars(Z, yy, tr, te))
        log(f'S{u}: linear on P_HP R2 {fits[f"S{u}"]["lin_HP_r2_test"]:.3f}, on all P {fits[f"S{u}"]["lin_P64_r2_test"]:.3f}, '
            f'MARS {fits[f"S{u}"]["mars_HP_path"][-1]["r2_test"]:.3f}')
    W = {u: lstsq_fit(Z, base['S'][:, u], tr)[0] for u in (58, 31)}
    sd = {u: base['S'][:, u].std() for u in (58, 31)}
    zz = np.stack([Fhat[58] / sd[58], Fhat[31] / sd[31]], 1); mu = zz.mean(0)
    ev, V = np.linalg.eigh(np.cov((zz - mu).T)); v1, v2 = V[:, 1], V[:, 0]
    if v1[0] < 0: v1 = -v1
    wa = W[58][:-1] / sd[58] * v1[0] + W[31][:-1] / sd[31] * v1[1]; wb = W[58][:-1] / sd[58] * v2[0] + W[31][:-1] / sd[31] * v2[1]
    afun = lambda P: P[:, HP] @ wa; bfun = lambda P: P[:, HP] @ wb
    a_all = afun(base['P']); sda = float(a_all.std()); sdb = float(bfun(base['P']).std()); tol = 0.01 * sda
    R['naming'] = dict(fits=fits, corr_fitted_S58_S31=float(np.corrcoef(Fhat[58], Fhat[31])[0, 1]),
                       pc1_share=float(ev[1] / ev.sum()), pc1_weights_on_standardised_fits=v1.tolist(),
                       a_coef_on_P_HP={f'P{h}': float(c) for h, c in zip(HP, wa)}, sd_a=sda, sd_b=sdb, tol=tol,
                       corr_a_S58=float(np.corrcoef(a_all, base['S'][:, 58])[0, 1]),
                       corr_a_S31=float(np.corrcoef(a_all, base['S'][:, 31])[0, 1]))
    log('naming', json.dumps(clean({k: v for k, v in R['naming'].items() if k != 'fits'})))
    # ---------------- selection
    t21, _ = E24.substructure(x, with_d2=False)
    R['tau21_vs_stored_max_diff'] = float(np.max(np.abs(t21 - Q['tau21']))) if 'tau21' in Q else None
    lab, two = kt2(x); soft = lab == 1; real = x[..., 0] > 0
    ptp = x[..., 0].astype(np.float64); zs = (ptp * soft).sum(1) / ptp.sum(1); ns = soft.sum(1)
    sel = two & (zs >= 0.1) & (ns >= 2) & (t21 < 0.5); idx = np.flatnonzero(sel)
    R['selection'] = dict(n_two_subjets=int(two.sum()), n_selected=int(sel.sum()),
                          per_class={CL[c]: int((y[sel] == c).sum()) for c in range(5)},
                          zs_median=float(np.median(zs[sel])), n_soft_median=float(np.median(ns[sel])))
    log('selection', json.dumps(R['selection']))
    # direct naming edit: rotate the selected jets by 45 deg
    xs, ys = x[idx], y[idx]; J = len(xs); rs = real[idx]; ss = soft[idx]
    b0 = {k: v[idx] for k, v in base.items()}
    # within-jet fit (v2): the edits move only P_HP inside a jet, so a is defined from within-jet changes: 4 random rigid
    # rotations + 2 softer-subjet scrambles of 1,500 selected jets; dS58, dS31 regressed on dP_HP (no intercept); fit on
    # half the jets, R^2 on the other half; a = first principal direction of the two standardised fitted changes
    nw = min(1500, J); wj = rng.choice(J, nw, replace=False); dP, dS = [], []
    for k in range(6):
        xe = rot(xs[wj], rs[wj], rng.uniform(-np.pi, np.pi, nw)) if k < 4 else scramble(xs[wj], ss[wj], rng)
        rr = j.run(xe.astype(np.float32), keep=('P', 'S'))
        dP.append(rr['P'][:, HP] - b0['P'][wj][:, HP]); dS.append(rr['S'][:, [58, 31]] - b0['S'][wj][:, [58, 31]])
    dP = np.concatenate(dP); dS = np.concatenate(dS); jid = np.tile(np.arange(nw), 6); trw = jid < nw // 2; tew = ~trw
    within = {}; Wd = {}
    for q, u in enumerate((58, 31)):
        w = np.linalg.lstsq(dP[trw], dS[trw, q], rcond=None)[0]; pq = dP @ w; Wd[u] = w
        within[f'S{u}'] = dict(r2_test_about_zero=float(1 - ((dS[tew, q] - pq[tew]) ** 2).sum() / (dS[tew, q] ** 2).sum()),
                               coef={f'P{h}': float(c) for h, c in zip(HP, w)}, mars_path=mars(dP, dS[:, q], trw, tew),
                               sd_dS_over_sd=float(dS[:, q].std() / sd[u]))
        log(f'within-jet S{u}: linear R2 {within[f"S{u}"]["r2_test_about_zero"]:.3f}, MARS {within[f"S{u}"]["mars_path"][-1]["r2_test"]:.3f}')
    zw = np.stack([dP @ Wd[58] / sd[58], dP @ Wd[31] / sd[31]], 1)
    ev, V = np.linalg.eigh(zw.T @ zw / len(zw)); v1, v2 = V[:, 1], V[:, 0]
    if v1[0] < 0: v1 = -v1
    wa = Wd[58] / sd[58] * v1[0] + Wd[31] / sd[31] * v1[1]; wb = Wd[58] / sd[58] * v2[0] + Wd[31] / sd[31] * v2[1]
    a_all = afun(base['P']); sda = float(a_all.std()); sdb = float(bfun(base['P']).std()); tol = 0.01 * sda
    R['naming']['across_jet_definition_v1'] = {k: R['naming'].pop(k) for k in list(R['naming']) if k != 'fits'}
    R['naming'].update(definition='within-jet (v2)', within=within, corr_within_fitted=float(np.corrcoef(zw.T)[0, 1]),
                       pc1_share=float(ev[1] / ev.sum()), pc1_weights_on_standardised_fits=v1.tolist(),
                       a_coef_on_P_HP={f'P{h}': float(c) for h, c in zip(HP, wa)}, sd_a=sda, sd_b=sdb, tol=tol,
                       within_r2_of_a_for_dS58=float(np.corrcoef(dP @ wa, dS[:, 0])[0, 1] ** 2),
                       within_r2_of_a_for_dS31=float(np.corrcoef(dP @ wa, dS[:, 1])[0, 1] ** 2))
    log('naming v2', json.dumps(clean({k: v for k, v in R['naming'].items() if k not in ('fits', 'within', 'across_jet_definition_v1')})))
    x45 = rot(xs, rs, np.full(J, np.pi / 4)).astype(np.float32); r45 = j.run(x45, keep=('P', 'S'))
    sdS = base['S'].std(0) + 1e-12
    R['naming']['rot45'] = dict(med_abs_da_over_sd=float(np.median(np.abs(afun(r45['P']) - afun(b0['P'])) / sda)),
                               mean_abs_dS_over_sd={f'S{u}': float(np.mean(np.abs(r45['S'][:, u] - b0['S'][:, u])) / sdS[u])
                                                    for u in [58, 31] + RADIAL_S})
    log('rot45', json.dumps(clean(R['naming']['rot45'])))
    save()
    # ---------------- edits
    a0 = afun(b0['P'])
    fa = lambda xe: afun(pf(xe))
    agrid = lambda xe, mk, angs: np.concatenate([fa(rot_grid(xe[c:c + 20], mk[c:c + 20], angs)).reshape(-1, len(angs))
                                                 for c in range(0, len(xe), 20)])
    fm = lambda xe: moments(xe)['m']
    out = {e: np.zeros((J, 64, 3)) for e in EDITS}; okE = {e: np.zeros(J, bool) for e in EDITS}; info = {}
    # R
    beta, ok = search(xs, rs, a0, fa, agrid, rng, exclude=np.pi / 6)
    out['R'] = rot(xs, rs, beta); okE['R'] = ok & (np.abs(fa(out['R']) - a0) <= tol); info['R_beta'] = beta
    log('R ok', int(okE['R'].sum()), '/', J)
    # D and D'
    for e, mk in (('D', ss), ("D'", rs)):
        draws = np.zeros(J, int)
        for k in range(12):
            todo = np.flatnonzero(~okE[e])
            if len(todo) == 0: break
            xe = scramble(xs[todo], mk[todo], rng)
            bt, ok = search(xe, rs[todo], a0[todo], fa, agrid, rng)
            xe = rot(xe, rs[todo], bt); ok = ok & (np.abs(fa(xe) - a0[todo]) <= tol)
            out[e][todo[ok]] = xe[ok]; okE[e][todo[ok]] = True; draws[todo] += 1
        info[e + '_draws'] = draws
        log(e, 'ok', int(okE[e].sum()), '/', J, 'mean draws', float(draws[okE[e]].mean()) if okE[e].any() else None)
    # P: mass matched to D, then beta
    mD = np.where(okE['D'], fm(out['D']), np.nan); alpha = np.full(J, np.nan)
    for t in range(3):
        todo = np.flatnonzero(okE['D'] & ~okE['P'])
        if len(todo) == 0: break
        al, ok1 = search(xs[todo], ss[todo], mD[todo], fm, lambda xe, mk, angs: mass_grid(xe, mk, angs), rng,
                         exclude=np.pi / 18, iters=24, t=t)
        xe = rot(xs[todo], ss[todo], al); ok1 &= np.abs(fm(xe) / mD[todo] - 1) <= 0.01
        bt, ok2 = search(xe, rs[todo], a0[todo], fa, agrid, rng)
        xe = rot(xe, rs[todo], bt); ok = ok1 & ok2 & (np.abs(fa(xe) - a0[todo]) <= tol)
        out['P'][todo[ok]] = xe[ok]; okE['P'][todo[ok]] = True; alpha[todo[ok]] = al[ok]
    log('P ok', int(okE['P'].sum()), '/', int(okE['D'].sum()))
    # v2: analysis set = jets with P and D (the main contrast); R and D' contrasts on the subsets where those also reached tol
    for e in EDITS: out[e][~okE[e]] = xs[~okE[e]]          # placeholder (unedited) where an edit failed; excluded below
    K = np.flatnonzero(okE['P'] & okE['D']); n = len(K); SM = {e: okE[e][K] for e in EDITS}
    R['acceptance'] = dict(J=J, **{e: int(okE[e].sum()) for e in EDITS}, analysis=n, R_in_analysis=int(SM['R'].sum()),
                           Dp_in_analysis=int(SM["D'"].sum()), all_four=int((SM['R'] & SM["D'"]).sum()),
                           analysis_per_class={CL[c]: int((ys[K] == c).sum()) for c in range(5)},
                           D_mean_draws=float(info['D_draws'][K].mean()), Dp_mean_draws=float(info["D'_draws"][K][SM["D'"]].mean()),
                           P_alpha_deg_median_abs=float(np.median(np.abs(np.degrees(alpha[K])))),
                           R_beta_deg_median_abs=float(np.median(np.abs(np.degrees(info['R_beta'][K][SM['R']])))))
    log('analysis set', n, json.dumps(R['acceptance']['analysis_per_class']))
    X0 = xs[K].astype(np.float32); Y = ys[K]; XE = {e: out[e][K].astype(np.float32) for e in EDITS}
    # ---------------- edit checks
    chk = {}
    sub0, _ = E24.substructure(X0, with_d2=False); SUB = {'none': sub0}
    for e in EDITS:
        SUB[e], _ = E24.substructure(XE[e], with_d2=False); s_ = SM[e]
        mo = moments(XE[e][s_]); mo0 = moments(X0[s_]); r1, _, _ = polar(XE[e][s_]); r0, _, re0 = polar(X0[s_])
        chk[e] = dict(n=int(s_.sum()), max_rel_dM0=float(np.max(np.abs(mo['M0'] / mo0['M0'] - 1))), max_rel_dM2=float(np.max(np.abs(mo['M2'] / mo0['M2'] - 1))),
                      max_rel_dMr=float(np.max(np.abs(mo['Mr'] / mo0['Mr'] - 1))), max_abs_dr=float(np.max(np.abs(r1 - r0)[re0])),
                      max_dpt=float(np.max(np.abs(XE[e][s_][..., 0] - X0[s_][..., 0]))),
                      padding_changed=int((XE[e][s_][~re0] != X0[s_][~re0]).sum()),
                      med_rel_dm=float(np.median(np.abs(mo['m'] / np.maximum(mo0['m'], 1e-9) - 1))),
                      mean_rel_dm=ci(mo['m'] / np.maximum(mo0['m'], 1e-9) - 1),
                      dtau21=ci((SUB[e] - sub0)[s_]), dtau21_W=ci((SUB[e] - sub0)[s_ & (Y == 2)]),
                      tau21_median={CL[c]: float(np.median(SUB[e][s_ & (Y == c)])) for c in range(5)})
    chk['none'] = dict(tau21_median={CL[c]: float(np.median(sub0[Y == c])) for c in range(5)})
    mP, mDD = moments(XE['P'])['m'], moments(XE['D'])['m']
    chk['P_vs_D_mass'] = dict(med_abs_rel=float(np.median(np.abs(mP / mDD - 1))), max_abs_rel=float(np.max(np.abs(mP / mDD - 1))))
    chk['D_minus_P_tau21'] = ci(SUB['D'] - SUB['P'])
    R['edit_checks'] = chk
    log('edit checks', json.dumps(clean({e: dict(dm=chk[e]['med_rel_dm'], dt21=chk[e]['dtau21']['mean'], dr=chk[e]['max_abs_dr'])
                                         for e in EDITS})), json.dumps(clean(chk['P_vs_D_mass'])))
    save()
    # ---------------- networks
    calls, lgt = {}, {}
    full_acc = {}
    for name in NETS:
        if name == 'jedi':
            run = lambda xx: j.run(xx, keep=('P', 'g', 'S', 'logits')); full_acc[name] = acc
        else:
            run, _, _, _ = E24.get_model(name); full_acc[name] = float((run(x)['logits'].argmax(1) == y).mean())
        res = {'none': run(X0)}
        for e in EDITS: res[e] = run(XE[e])
        lgt[name] = {k: v['logits'] for k, v in res.items()}; calls[name] = {k: v.argmax(1) for k, v in lgt[name].items()}
        if name == 'jedi': JR = res
        log(name, 'full-sample acc', round(full_acc[name], 4), 'changed', {e: round(float((calls[name][e] != calls[name]['none'])[SM[e]].mean()), 4) for e in EDITS})
    R['full_sample_acc'] = full_acc
    # JEDI internals and freeze arms
    g0, S0 = JR['none']['g'], JR['none']['S']; aN = afun(JR['none']['P']); bN = bfun(JR['none']['P'])
    R['jedi_internal'] = {e: dict(abs_da_over_sd_max=float(np.max(np.abs(afun(JR[e]['P']) - aN)[SM[e]]) / sda),
                                 abs_db_over_sd_b=ci((np.abs(bfun(JR[e]['P']) - bN) / sdb)[SM[e]]),
                                 abs_dS_over_sd={f'S{u}': ci((np.abs(JR[e]['S'][:, u] - S0[:, u]) / sdS[u])[SM[e]]) for u in [58, 31] + RADIAL_S},
                                 abs_dg_mean=float(np.mean(np.abs(JR[e]['g'] - g0)[SM[e]])))
                         for e in EDITS}
    one = np.ones(64); mS = np.zeros(64); mS[[58, 31]] = 1
    for arm, ov in (('g_frozen', {'g': (one, g0)}), ('S58_S31_frozen', {'S': (mS, S0)})):
        calls['jedi|' + arm] = {'none': calls['jedi']['none']}
        lgt['jedi|' + arm] = {'none': lgt['jedi']['none']}
        for e in EDITS:
            lo = j.run_jetwise(XE[e], ov)['logits']; lgt['jedi|' + arm][e] = lo; calls['jedi|' + arm][e] = lo.argmax(1)
        log('jedi|' + arm, {e: round(float((calls['jedi|' + arm][e] != calls['jedi']['none'])[SM[e]].mean()), 4) for e in EDITS})
    save()
    # ---------------- trees
    d = np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r'); y_all = np.asarray(d['y'])
    im = E24.sample_idx(y_all, 3000, 2801); assert np.array_equal(y_all[im], y)
    xp, yp, _ = sample(4000, 2802); ip = E24.sample_idx(y_all, 4000, 2802); kp = ~np.isin(ip, im); xp, yp = xp[kp], yp[kp]
    R['pool_n'] = int(len(xp)); tp, _ = E24.substructure(xp, with_d2=False); log('pool', len(xp))

    def feats(xx, t, which):
        mo = moments(xx)
        h = [np.log(mo['M0']), np.log(mo['M2'] / mo['M0']), mo['Mr'] / mo['M0'], mo['N']]
        cols = {'Th': h, 'Thm': h + [np.log(np.maximum(mo['m'], 1e-6))], 'Thmt': h + [np.log(np.maximum(mo['m'], 1e-6)), t],
                'Tht': h + [t], 'Tt': [t]}[which]
        return np.stack(cols, 1)
    t_full = t21
    for T in TREES:
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0).fit(feats(xp, tp, T), yp)
        full_acc[T] = float((clf.predict(feats(x, t_full, T)) == y).mean())
        lgt[T] = {'none': clf.predict_proba(feats(X0, SUB['none'], T))}
        for e in EDITS: lgt[T][e] = clf.predict_proba(feats(XE[e], SUB[e], T))
        lgt[T] = {k: np.log(np.maximum(v, 1e-9)) for k, v in lgt[T].items()}; calls[T] = {k: v.argmax(1) for k, v in lgt[T].items()}
        log(T, 'full-sample acc', round(full_acc[T], 4), {e: round(float((calls[T][e] != calls[T]['none'])[SM[e]].mean()), 4) for e in EDITS})
    # ---------------- statistics
    W_ = Y == 2
    for name in list(calls):
        C, Lg = calls[name], lgt[name]; c0 = C['none']; ch = {e: C[e] != c0 for e in EDITS}
        cen = lambda L: L - L.mean(1, keepdims=True); dl = {e: np.linalg.norm(cen(Lg[e]) - cen(Lg['none']), axis=1) for e in EDITS}
        M = dict(acc_unedited_analysis=ci(c0 == Y), acc={e: ci((C[e] == Y)[SM[e]]) for e in EDITS},
                 calls_changed={e: ci(ch[e][SM[e]]) for e in EDITS},
                 W_to_Z={e: ci(((c0 == 2) & (C[e] == 3))[W_ & SM[e]]) for e in EDITS},
                 W_to_q={e: ci(((c0 == 2) & (C[e] == 1))[W_ & SM[e]]) for e in EDITS},
                 dlogit_l2={e: ci(dl[e][SM[e]]) for e in EDITS})
        for a_, b_ in (('D', 'P'), ('D', 'R'), ('P', 'R'), ("D'", 'R'), ("D'", 'D')):
            s_ = SM[a_] & SM[b_]
            mc = mcnemar(ch[a_][s_], ch[b_][s_])
            mc['p_one_sided'] = p_upper(mc['n10'], mc['n01'])
            M[f'{a_}_minus_{b_}'] = dict(n=int(s_.sum()), calls=ci((ch[a_].astype(float) - ch[b_])[s_]), mcnemar=mc,
                                         W_to_Z=ci((((c0 == 2) & (C[a_] == 3)).astype(float) - ((c0 == 2) & (C[b_] == 3)))[W_ & s_]),
                                         dlogit=ci((dl[a_] - dl[b_])[s_]), dlogit_p=boot_p((dl[a_] - dl[b_])[s_]))
        (R['trees'] if name in TREES else R['models'])[name] = M
    ps = [R['models'][m]['D_minus_P']['mcnemar']['p'] for m in NETS]
    for m, h in zip(NETS, holm(ps)): R['models'][m]['D_minus_P']['mcnemar']['p_holm_11'] = float(h)
    ps1 = [R['models'][m]['D_minus_P']['mcnemar']['p_one_sided'] for m in NETS]
    for m, h in zip(NETS, holm(ps1)): R['models'][m]['D_minus_P']['mcnemar']['p_one_sided_holm_11'] = float(h)
    # JEDI vs trees: paired difference of the D - P excess
    exj = (calls['jedi']['D'] != calls['jedi']['none']).astype(float) - (calls['jedi']['P'] != calls['jedi']['none'])
    R['jedi_vs_trees'] = {}
    for T in TREES:
        ext = (calls[T]['D'] != calls[T]['none']).astype(float) - (calls[T]['P'] != calls[T]['none'])
        R['jedi_vs_trees'][T] = dict(excess_diff_jedi_minus_tree=ci(exj - ext), p=boot_p(exj - ext))
    for name in NETS + ['jedi|g_frozen', 'jedi|S58_S31_frozen'] + TREES:
        M = (R['trees'] if name in TREES else R['models'])[name]
        log(f'{name:22s}', 'changed', {e: round(M['calls_changed'][e]['mean'] * 100, 2) for e in EDITS},
            'D-R', round(M['D_minus_R']['calls']['mean'] * 100, 2), 'P-R', round(M['P_minus_R']['calls']['mean'] * 100, 2),
            'D-P', round(M['D_minus_P']['calls']['mean'] * 100, 2), [round(v * 100, 2) for v in M['D_minus_P']['calls']['ci']],
            'p', f"{M['D_minus_P']['mcnemar']['p']:.2g}")
    save()
    np.savez_compressed(OUT / 'e24w_calls.npz', idx=idx[K], y=Y, **{f'{nm}|{e}'.replace("'", 'p'): C[e].astype(np.int8)
                                                                    for nm, C in calls.items() for e in ['none'] + EDITS},
                        tau21=np.stack([SUB[k] for k in ['none'] + EDITS]),
                        ok=np.stack([SM[e] for e in EDITS]))
    log('wrote', OUT / 'e24w_prong_jedi.json')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True); main()
