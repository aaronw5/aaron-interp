"""E21: where the mass product forms. S-level patches along iso-product and iso-sum curves of the M0 and M2 carriers.

Sample: sample(per_class=2000, seed=47), 10,000 jets, exact Jedi(64) (3-feature n64).  Unit interventions only (no input edit):
with run_jetwise on stage S, per jet j and signed step e in EPS (standardised length in (z0, z_k) space):
  iso-product   along m0 * m_k = const, m_u = a_u S_u + b_u (arms A, B: a = 1, b = 0, i.e. S0 * S_k; arm C: the affine fit of
                M0 = sum pT on S0 and M2 = sum pT r^2 on S50):  m0 -> m0 e^d, m_k -> m_k e^-d, d = e / |(m_u / (a_u sd_u))_u|
  iso-sum       along z0 + mean_k z_k = const:  z0 += e / sqrt(1 + K), z_k -= e / sqrt(1 + K)
  arm A: M2 carriers {S50};  arm B: {S50, S29};  arm C: {S50}, affine moments.  Jets with a patched S < 0 or m_u <= 0 in any
  step are flagged and left out (S >= 0 by construction).  Run 1 (results/jedi/e21_run1_cdesign.json) used S0 x c with
  S_k / c or S_k + (1 - c) S0 sd_k / sd0, c in {0.7 .. 1.4}: the iso-sum arm left S >= 0 for only 19 % of jets (6 W jets).
Every other S unit keeps its own value.  Per live unit U of h4, h5, z (and the W - Z logit margin, logit2 - logit3):
  |dU|_prod, |dU|_sum summed over the six c;  R = sum_j |dU|_prod / sum_j |dU|_sum,  F = |d|_prod / (|d|_prod + |d|_sum)
  (product rule: R = 0, F = 0; standardised-sum rule: R = inf, F = 1).  95 % bootstrap CI over jets (Poisson weights);
  H0 R = 1: paired z-test on D_j = |dU|_prod - |dU|_sum, Holm over units (per stage and over all stages).
Rules on the same patched S values: V_sum = z0 + mean_k z_k, V_prod = log S0 + mean_k log S_k, 64 ReLU windows on each,
  64 ReLU units on random directions in (z0, z_k) (separate carriers), 64 random ReLU units on the whole patched S (E15v rand1),
  then 32 more (rand2).
Controls: (a) the same standardised displacements on 2 random live S units (20 draws, arm A), (b) random ReLU layers on the
  trained S (3 seeds), (c) the untrained network (same patch formulas on its S0, S50).
Calls: % changed per c and edit, McNemar (exact binomial) iso-prod vs iso-sum.
Calibration (real input edits, the particles' pT and angles): pT x c, angles x 1/c (M0 x c, M2 / c: iso-product of the true
  moments) and pT x c, angles x c^-1/2 (M0 x c, M2 fixed); medians of S0'/S0, S50'/S50, S29'/S29 and the shift of the other S units.
Writes $AI_RESULTS/jedi/e21_isoproduct.json.   Run: python -m interp.e21_isoproduct_jedi
"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT_DIR = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
OUT_JSON = OUT_DIR / 'e21_isoproduct.json'
PER, SEED = 2000, 47
CS = (0.7, 0.8, 0.9, 1.1, 1.25, 1.4)            # calibration (real input edits)
EPS = (-0.5, -0.3, -0.15, 0.15, 0.3, 0.5)        # S-patch step length in standardised S units
ARMS = {'A': [50], 'B': [50, 29], 'C': [50]}     # C: iso-curves of the affine-fitted moments M0 = a0 S0 + b0, M2 = a50 S50 + b50
STG = ('h4', 'h5', 'z'); KEEP = ('S',) + STG + ('logits',)
NBOOT = 300; NDRAW = 20
NAMED_Z = [0, 7, 14, 10]
CL = ['g', 'q', 'W', 'Z', 't']
t0 = time.time()


def log(*a):
    print(f'[{time.time() - t0:6.0f}s]', *a, flush=True)


from interp.jedi_core import Jedi, sample  # noqa: E402

OUT_DIR.mkdir(parents=True, exist_ok=True)
x, y, Q = sample(per_class=PER, seed=SEED); del Q
x = np.ascontiguousarray(x).astype(np.float32); J = len(x); real = x[..., 0] > 0
SUBS = {'all': np.ones(J, bool), 'W': y == 2, 'Z': y == 3, 't': y == 4}
res = dict(N=J, per_class=PER, seed=SEED, cs=CS, eps=EPS, nboot=NBOOT, ndraw=NDRAW)
log(f'sample N {J}')
WB = np.random.default_rng(SEED + 1).poisson(1.0, size=(NBOOT, J)).astype(np.float64)


def pval_z(D, m):
    d = D[m]; s = d.std(ddof=1) / math.sqrt(len(d))
    if s <= 0: return 1.0
    return float(math.erfc(abs(d.mean() / s) / math.sqrt(2)))


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); n = len(ps); adj = np.empty(n); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (n - r) * ps[i])); adj[i] = run
    return adj


def mcnemar(a, b):
    """a, b bool arrays (changed under edit 1 / edit 2): exact two-sided binomial on discordant pairs"""
    n10 = int((a & ~b).sum()); n01 = int((~a & b).sum()); n = n10 + n01
    if n == 0: return dict(n10=0, n01=0, p=1.0)
    k = min(n10, n01)
    lp = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(k + 1)]
    mx = max(lp); p = min(1.0, 2 * math.exp(mx) * sum(math.exp(v - mx) for v in lp))
    return dict(n10=n10, n01=n01, p=float(p))


def in_range(net_, xx):
    q = net_.L['l1'].iq.quantizer
    k_, i_, f_ = (np.asarray(getattr(q, a)).reshape(64, 3) for a in ('k', 'i', 'f'))
    hi = 2.0 ** i_ - 2.0 ** -f_; lo = np.where(k_ > 0, -2.0 ** i_, 0.0)
    xn = net_.inputs(xx, 64); rl = xx[..., 0] > 0
    return ~((((xn > hi[None]) | (xn < lo[None])) & rl[..., None]).any((1, 2)))


def patch_values(S, sd, i0, ks, e, kind, disp=None, aff=None):
    """patched columns {unit: values} for a step of standardised length e (sign = direction).
    prod: along the iso-product curve of the moments m_u = a_u S_u + b_u (aff; default a = 1, b = 0, i.e. S0 * S_k):
          m0 -> m0 e^d, m_k -> m_k e^-d, d = e / sqrt(sum_u (m_u / (a_u sd_u))^2)  (first-order standardised length e)
    sum:  along the iso-sum line z0 + mean_k z_k = const: z0 += e / sqrt(1 + K), z_k -= e / sqrt(1 + K)
    disp=(dz0, dzk) per jet replaces the formulas (random-pair control).  Returns cols, flag (clipped below 0 or m_u <= 0)."""
    out = {}; flag = np.zeros(len(S), bool); us = [i0] + list(ks)
    if disp is not None:
        dz0, dzk = disp
        for u, dz in zip(us, [dz0] + [dzk] * len(ks)):
            v = S[:, u] + dz * sd[u]; flag |= v < 0; out[u] = np.maximum(v, 0)
        return out, flag
    if kind == 'prod':
        A = {u: (aff or {}).get(u, (1.0, 0.0)) for u in us}
        mom = {u: A[u][0] * S[:, u] + A[u][1] for u in us}
        for u in us: flag |= mom[u] <= 0
        den = np.sqrt(sum((np.maximum(mom[u], 0) / (A[u][0] * sd[u])) ** 2 for u in us))
        d = np.where(den > 0, e / np.maximum(den, 1e-30), 0.0)
        for u in us:
            mm = np.maximum(mom[u], 0) * np.exp(d if u == i0 else -d)
            v = (mm - A[u][1]) / A[u][0]; flag |= v < 0; out[u] = np.maximum(v, 0)
    else:
        a = e / np.sqrt(1 + len(ks))
        for u in us:
            v = S[:, u] + (a if u == i0 else -a) * sd[u]; flag |= v < 0; out[u] = np.maximum(v, 0)
    return out, flag


def make_tail(net_):
    """S (J, 64) -> h4, h5, z, logits with the network's own layers (same ops as Jedi.forward after the S override)"""
    import jax, jax.numpy as jnp
    L = net_.L; KB = {}
    for k in ('d4', 'd5', 'd6', 'd7'):
        K, b = L[k].get_fused_qkernel_and_qbias(False, L[k].moving_mean, L[k].moving_variance); KB[k] = (jnp.asarray(K), jnp.asarray(b))

    def dense(k, v, act=True):
        z = jnp.einsum(L[k].equation, L[k].iq(v, training=False), KB[k][0]) + KB[k][1]
        return L[k].activation(z) if act and L[k].activation is not None else z

    def f(S_):
        h4 = dense('d4', S_); h5 = dense('d5', h4); z = dense('d6', h5, act=False); lo = dense('d7', jax.nn.relu(z))
        Jn = S_.shape[0]
        return dict(h4=h4.reshape(Jn, -1), h5=h5.reshape(Jn, -1), z=z.reshape(Jn, -1), logits=lo.reshape(Jn, -1))
    return jax.jit(f)


TAILS = {}


def run_patch(net_, S, cols, exact=False):
    m = np.zeros(64); Sp = S.copy()
    for k, v in cols.items(): m[k] = 1; Sp[:, k] = v
    if exact or id(net_) not in TAILS:
        return net_.run_jetwise(x, {'S': (m, Sp)}, keep=KEEP[1:], batch=2500), Sp
    f, shp = TAILS[id(net_)]
    r = f(Sp.astype(np.float32).reshape((len(Sp),) + shp))
    return {k: np.asarray(v, np.float64) for k, v in r.items()}, Sp


def enable_tail(net_, R0_, S_, tag):
    """use the tail only if it reproduces run_jetwise exactly on the base S and on one patch"""
    for shp in ((1, 64), (64,)):
        try:
            TAILS[id(net_)] = (make_tail(net_), shp)
            o_ = np.argsort(-S_.std(0)); cols = patch_values(S_, S_.std(0), int(o_[0]), [int(o_[1])], 0.3, 'sum')[0]
            Re, _ = run_patch(net_, S_, cols, exact=True); Rt, _ = run_patch(net_, S_, cols)
            Rb, _ = run_patch(net_, S_, {})
            d = max(float(np.abs(Re[k] - Rt[k]).max()) for k in Re); d0 = max(float(np.abs(Rb[k] - R0_[k]).max()) for k in Re)
            log(f'{tag}: tail shape {shp}: max |diff| vs run_jetwise {d:.2e}, vs base {d0:.2e}')
            if d < 1e-6 and d0 < 1e-6: return dict(shape=list(shp), max_diff_patch=d, max_diff_base=d0)
        except Exception as e:  # noqa: BLE001
            log(f'{tag}: tail shape {shp} failed: {e!r}')
        TAILS.pop(id(net_), None)
    return dict(shape=None)


def margin(lo): return lo[:, 2] - lo[:, 3]


def live_units(R0):
    L = {}
    for st in STG:
        b = R0[st]; sd = b.std(0); on = (b > 0).mean(0)
        L[st] = np.flatnonzero((sd > 1e-6) & ((on > 0.02) if st != 'z' else True))
    return L


def scan(net_, R0, S, sd, i0, ks, tag, disp_fn=None, keep_per_c=True, aff=None):
    """runs the 2 x 6 patches; returns per-jet accumulators |d| (J, U) per stage for prod / sum, per-c arrays, calls, clips"""
    L = live_units(R0); call0 = R0['logits'].argmax(1)
    acc = {kind: {st: np.zeros((J, len(L[st]))) for st in STG} for kind in ('prod', 'sum')}
    for kind in acc: acc[kind]['m'] = np.zeros((J, 1))
    perc = {kind: {} for kind in acc}; calls = {kind: {} for kind in acc}; clip = np.zeros(J, bool); Spatch = {kind: {} for kind in acc}
    for kind in ('prod', 'sum'):
        for c in EPS:
            cols, cl = patch_values(S, sd, i0, ks, c, kind, None if disp_fn is None else disp_fn(c, kind), aff=aff)
            clip |= cl
            Rp, Sp = run_patch(net_, S, cols)
            Spatch[kind][c] = {k: v.astype(np.float32) for k, v in cols.items()}
            pc = {}
            for st in STG:
                d = Rp[st][:, L[st]] - R0[st][:, L[st]]; acc[kind][st] += np.abs(d)
                if keep_per_c and st == 'z': pc[st] = d.astype(np.float32)      # signed
            dm = (margin(Rp['logits']) - margin(R0['logits']))[:, None]; acc[kind]['m'] += np.abs(dm)
            if keep_per_c: pc['m'] = dm.astype(np.float32)
            perc[kind][c] = pc
            calls[kind][c] = Rp['logits'].argmax(1) != call0
        log(f'  {tag} {kind}: done')
    return dict(L=L, acc=acc, perc=perc, calls=calls, clip=clip, Spatch=Spatch)


def unit_stats(Ap, As, m, boot=True):
    """per unit (columns) R, F, CI, p over jets m"""
    sp, ss = Ap[m].sum(0), As[m].sum(0)
    R = sp / np.maximum(ss, 1e-30); F = sp / np.maximum(sp + ss, 1e-30)
    out = dict(R=R, F=F)
    if boot:
        w = WB[:, m]; bp, bs = w @ Ap[m], w @ As[m]
        Rb = bp / np.maximum(bs, 1e-30)
        out['R_lo'], out['R_hi'] = np.percentile(Rb, 2.5, 0), np.percentile(Rb, 97.5, 0)
        out['p'] = np.array([pval_z(Ap[:, i] - As[:, i], m) for i in range(Ap.shape[1])])
    return out


def summarize(sc, R0, m, tag, boot=True):
    out = {}; allp = []; keys = []
    sdu = {st: R0[st].std(0) for st in STG}
    for st in STG + ('m',):
        Ap, As = sc['acc']['prod'][st], sc['acc']['sum'][st]
        u = unit_stats(Ap, As, m, boot)
        units = sc['L'][st] if st != 'm' else np.array([0])
        sd_ = sdu[st][units] if st != 'm' else np.array([margin(R0['logits']).std()])
        resp = (As[m].mean(0) + Ap[m].mean(0)) / (2 * len(EPS)) / np.maximum(sd_, 1e-12)   # mean |d| per patch, in unit sd
        ok = resp > 0.01
        row = dict(units=[int(v) for v in units], R=u['R'].tolist(), F=u['F'].tolist(), resp_sd=resp.tolist(), responsive=ok.tolist())
        if boot:
            row.update(R_lo=u['R_lo'].tolist(), R_hi=u['R_hi'].tolist(), p=u['p'].tolist())
            for i in np.flatnonzero(ok): allp.append(u['p'][i]); keys.append((st, i))
        lr = np.log10(np.maximum(u['R'][ok], 1e-6))
        row['n_responsive'] = int(ok.sum())
        row['log10R_median'] = float(np.median(lr)) if ok.any() else None
        row['log10R_q25_q75'] = [float(np.percentile(lr, 25)), float(np.percentile(lr, 75))] if ok.any() else None
        row['frac_R_below_0p3'] = float(np.mean(u['R'][ok] < 0.3)) if ok.any() else None
        # pooled over units: |d| summed over responsive units in unit-sd
        if ok.any():
            row['R_pooled_sd'] = float((Ap[m][:, ok] / sd_[ok]).sum() / max((As[m][:, ok] / sd_[ok]).sum(), 1e-30))
        out[st] = row
    if boot and allp:
        adj = holm(allp)
        for st in STG + ('m',):
            out[st]['p_holm_all'] = [None] * len(out[st]['units'])
        for (st, i), a in zip(keys, adj): out[st]['p_holm_all'][i] = float(a)
        for st in STG + ('m',):
            idx = [i for (s_, i) in keys if s_ == st]
            if not idx: continue
            ph = holm([out[st]['p'][i] for i in idx])
            out[st]['p_holm_stage'] = [None] * len(out[st]['units'])
            for i, a in zip(idx, ph): out[st]['p_holm_stage'][i] = float(a)
            R_ = np.array(out[st]['R']); pa = np.array([out[st]['p_holm_all'][i] for i in idx])
            out[st]['n_R_below_1_holm05'] = int(((R_[idx] < 1) & (pa < 0.05)).sum())
            out[st]['n_R_above_1_holm05'] = int(((R_[idx] > 1) & (pa < 0.05)).sum())
    msg = ' | '.join(f'{st}: n {out[st]["n_responsive"]} log10R med {out[st]["log10R_median"]} IQR {out[st]["log10R_q25_q75"]} '
                     f'R<0.3 {out[st]["frac_R_below_0p3"]} Rpool {out[st].get("R_pooled_sd")} '
                     f'sig<1 {out[st].get("n_R_below_1_holm05")} sig>1 {out[st].get("n_R_above_1_holm05")}' for st in STG + ('m',))
    log(f'{tag}: {msg}')
    return out


def named(sc, m):
    """z0, z7, z14, z10 and W - Z margin: R with CI, p, per c"""
    out = {}
    Lz = list(sc['L']['z'])
    tg = [(f'z{u}', 'z', Lz.index(u)) for u in NAMED_Z if u in Lz] + [('W-Z margin', 'm', 0)]
    for nm, st, i in tg:
        Ap, As = sc['acc']['prod'][st][:, i], sc['acc']['sum'][st][:, i]
        sp, ss = Ap[m].sum(), As[m].sum(); w = WB[:, m]
        Rb = (w @ Ap[m]) / np.maximum(w @ As[m], 1e-30)
        row = dict(R=float(sp / max(ss, 1e-30)), R_ci=[float(np.percentile(Rb, 2.5)), float(np.percentile(Rb, 97.5))],
                   p=pval_z(Ap - As, m), mean_abs_d_prod=float(Ap[m].mean() / len(EPS)), mean_abs_d_sum=float(As[m].mean() / len(EPS)))
        if sc['perc']['prod'][EPS[0]]:
            row['R_per_c'] = {str(c): float(np.abs(sc['perc']['prod'][c][st][m, i]).sum() / max(np.abs(sc['perc']['sum'][c][st][m, i]).sum(), 1e-30)) for c in EPS}
        out[nm] = row
    return out


def calls_tab(sc, m):
    out = {}
    for c in EPS:
        a, b = sc['calls']['prod'][c][m], sc['calls']['sum'][c][m]
        out[str(c)] = dict(prod_changed=float(a.mean()), sum_changed=float(b.mean()), mcnemar=mcnemar(a, b))
    a = np.stack([sc['calls']['prod'][c][m] for c in EPS]); b = np.stack([sc['calls']['sum'][c][m] for c in EPS])
    out['mean_over_c'] = dict(prod_changed=float(a.mean()), sum_changed=float(b.mean()))
    return out


def rule_pops(Spatch, S, sd, i0, ks, rng_, aff=None):
    """rule predictions on the same patched S: per rule, per kind, (J, U) accumulators of |d|"""
    def zs(v, k): return (v - S[:, k].mean()) / sd[k]
    def full(kind, c):
        Sp = S.copy()
        for k, v in Spatch[kind][c].items(): Sp[:, k] = v
        return Sp
    eps = 1e-3 * np.array([sd[i0]] + [sd[k] for k in ks])
    def vsum(Sp): return zs(Sp[:, i0], i0) + np.mean([zs(Sp[:, k], k) for k in ks], 0)
    A_ = {u: (aff or {}).get(u, (1.0, 0.0)) for u in [i0] + list(ks)}
    def mo(Sp, u, e): return np.maximum(A_[u][0] * Sp[:, u] + A_[u][1], 0) + e * A_[u][0]
    def vprod(Sp): return np.log(mo(Sp, i0, eps[0])) + np.mean([np.log(mo(Sp, k, e)) for k, e in zip(ks, eps[1:])], 0)
    v0s, v0p = vsum(S), vprod(S)
    n = 64; sg = rng_.choice([-1.0, 1.0], n); qs_ = rng_.uniform(0.05, 0.95, n)
    th = rng_.uniform(0, 2 * np.pi, n)
    def win(v, v0): zz = (v - v0.mean()) / v0.std(); q = np.quantile((v0 - v0.mean()) / v0.std(), qs_); return np.maximum(sg * (zz[:, None] - q), 0)
    def sep(Sp, S_ref):
        a = zs(Sp[:, i0], i0); b = np.mean([zs(Sp[:, k], k) for k in ks], 0)
        a0 = zs(S_ref[:, i0], i0); b0 = np.mean([zs(S_ref[:, k], k) for k in ks], 0)
        q = np.array([np.quantile(a0 * np.cos(t) + b0 * np.sin(t), qq) for t, qq in zip(th, qs_)])
        return np.maximum(a[:, None] * np.cos(th) + b[:, None] * np.sin(th) - q, 0)
    live = sd > 1e-6; mu = S.mean(0)
    W1 = rng_.normal(0, 1, (int(live.sum()), 64)) / np.sqrt(live.sum())
    pre0 = ((S[:, live] - mu[live]) / sd[live]) @ W1; b1 = -np.array([np.quantile(pre0[:, i], q) for i, q in enumerate(rng_.uniform(0.2, 0.8, 64))])
    def r1(Sp): return np.maximum(((Sp[:, live] - mu[live]) / sd[live]) @ W1 + b1, 0)
    H0 = r1(S); hmu, hsd = H0.mean(0), H0.std(0); hl = hsd > 1e-9
    W2 = rng_.normal(0, 1, (int(hl.sum()), 32)) / np.sqrt(hl.sum())
    pre2 = ((H0[:, hl] - hmu[hl]) / hsd[hl]) @ W2; b2 = -np.array([np.quantile(pre2[:, i], q) for i, q in enumerate(rng_.uniform(0.2, 0.8, 32))])
    def r2(H): return np.maximum(((H[:, hl] - hmu[hl]) / hsd[hl]) @ W2 + b2, 0)
    base = dict(V_sum=v0s[:, None], V_prod=v0p[:, None], win_sum=win(v0s, v0s), win_prod=win(v0p, v0p), sep_relu=sep(S, S), rand1_S=H0, rand2_S=r2(H0))
    acc = {nm: {kind: np.zeros_like(b) for kind in ('prod', 'sum')} for nm, b in base.items()}
    for kind in ('prod', 'sum'):
        for c in EPS:
            Sp = full(kind, c); vs, vp = vsum(Sp), vprod(Sp); h1 = r1(Sp)
            new = dict(V_sum=vs[:, None], V_prod=vp[:, None], win_sum=win(vs, v0s), win_prod=win(vp, v0p), sep_relu=sep(Sp, S), rand1_S=h1, rand2_S=r2(h1))
            for nm in base: acc[nm][kind] += np.abs(new[nm] - base[nm])
    return acc, base


def rule_summary(acc, base, m):
    out = {}
    for nm, a in acc.items():
        Ap, As = a['prod'][m], a['sum'][m]; sd_ = base[nm][m].std(0)
        ok = ((Ap.mean(0) + As.mean(0)) / (2 * len(EPS))) > 0.01 * np.maximum(sd_, 1e-12)
        if not ok.any(): out[nm] = dict(n=0); continue
        R = Ap[:, ok].sum(0) / np.maximum(As[:, ok].sum(0), 1e-30); lr = np.log10(np.clip(R, 1e-6, 1e6))
        out[nm] = dict(n=int(ok.sum()), log10R_median=float(np.median(lr)), log10R_q25_q75=[float(np.percentile(lr, 25)), float(np.percentile(lr, 75))],
                       frac_R_below_0p3=float(np.mean(R < 0.3)),
                       R_pooled_sd=float((Ap[:, ok] / sd_[ok]).sum() / max((As[:, ok] / sd_[ok]).sum(), 1e-30)))
    return out


def angle_analysis(sc, S, sd, i0, ks, m, tag):
    """Per jet, the iso-product step's direction in (z0, z_k) differs (tangent angle theta = atan(|dz_k| / |dz0|)).
    Product-of-carriers rule: R ~ 0 in every theta bin.  Fixed-weight additive rule dU = w0 dz0 + sum_k w_k dz_k (one weight
    vector for all jets, least squares on even jets over both edits and all steps): its R per bin on the odd jets.
    Also the held-out R^2 of that additive fit for each edit."""
    us = [i0] + list(ks)
    dz = {kind: {c: np.stack([(sc['Spatch'][kind][c][u] - S[:, u]) / sd[u] for u in us], 1) for c in EPS} for kind in ('prod', 'sum')}
    e1 = [c for c in EPS if c > 0][0]
    th = np.degrees(np.arctan2(np.linalg.norm(dz['prod'][e1][:, 1:], axis=1), np.abs(dz['prod'][e1][:, 0])))
    edges = np.percentile(th[m], [25, 50, 75]); B = np.digitize(th, edges)
    fit = m & (np.arange(J) % 2 == 0); te = m & (np.arange(J) % 2 == 1)
    Lz = list(sc['L']['z']); out = dict(theta_deg_quartile_edges=[float(v) for v in edges],
                                         theta_deg_median_by_bin=[float(np.median(th[te & (B == b)])) for b in range(4)])
    tg = [(f'z{u}', 'z', Lz.index(u)) for u in NAMED_Z if u in Lz] + [('W-Z margin', 'm', 0)]
    for nm, st, i in tg:
        X = {k: np.concatenate([dz[k][c] for c in EPS]) for k in dz}; Y = {k: np.concatenate([sc['perc'][k][c][st][:, i] for c in EPS]).astype(np.float64) for k in dz}
        rep_ = lambda mm: np.tile(mm, len(EPS))
        Xf = np.concatenate([X['prod'][rep_(fit)], X['sum'][rep_(fit)]]); Yf = np.concatenate([Y['prod'][rep_(fit)], Y['sum'][rep_(fit)]])
        w = np.linalg.lstsq(Xf, Yf, rcond=None)[0]
        row = dict(w_additive=[float(v) for v in w])
        for k in dz:
            yt, yp = Y[k][rep_(te)], X[k][rep_(te)] @ w
            row[f'R2_additive_{k}'] = float(1 - ((yt - yp) ** 2).sum() / max(((yt - yt.mean()) ** 2).sum(), 1e-30))
        obs, pred = [], []
        for b in range(4):
            mb = rep_(te & (B == b))
            obs.append(float(np.abs(Y['prod'][mb]).sum() / max(np.abs(Y['sum'][mb]).sum(), 1e-30)))
            pred.append(float(np.abs(X['prod'][mb] @ w).sum() / max(np.abs(X['sum'][mb] @ w).sum(), 1e-30)))
        # bootstrap CI of observed R per bin (test jets)
        ci = []
        for b in range(4):
            jb = np.flatnonzero(te & (B == b)); ap = np.abs(np.stack([sc['perc']['prod'][c][st][jb, i] for c in EPS])).sum(0)
            as_ = np.abs(np.stack([sc['perc']['sum'][c][st][jb, i] for c in EPS])).sum(0); wb = WB[:, jb]
            rb = (wb @ ap) / np.maximum(wb @ as_, 1e-30); ci.append([float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))])
        row.update(R_obs_by_theta_bin=obs, R_obs_ci=ci, R_additive_pred_by_theta_bin=pred)
        out[nm] = row
        log(f'{tag} angle {nm}: w {np.round(w, 3).tolist()}, R2 add prod {row["R2_additive_prod"]:.3f} sum {row["R2_additive_sum"]:.3f}; '
            f'R obs by theta bin {np.round(obs, 3).tolist()} vs additive {np.round(pred, 3).tolist()}')
    log(f'{tag} theta edges {np.round(edges, 1).tolist()} deg')
    return out


# ====================================================================== 0. network, checks
net = Jedi(64)
R0 = net.run(x, keep=KEEP, batch=2500); call0 = R0['logits'].argmax(1)
res['acc'] = float((call0 == y).mean()); res['acc_binom_sd'] = float(math.sqrt(res['acc'] * (1 - res['acc']) / J))
S = R0['S'].astype(np.float64); sd = S.std(0); mu = S.mean(0)
Rid, _ = run_patch(net, S, {0: S[:, 0], 50: S[:, 50], 29: S[:, 29]})
res['identity_max_abs_diff'] = {st: float(np.abs(Rid[st] - R0[st]).max()) for st in STG + ('logits',)}
res['S_carriers'] = {str(k): dict(mean=float(mu[k]), sd=float(sd[k]), frac_zero=float((S[:, k] <= 0).mean())) for k in (0, 50, 29)}
log(f'acc {res["acc"]:.4f} (sd {res["acc_binom_sd"]:.4f}); identity {res["identity_max_abs_diff"]}; carriers {res["S_carriers"]}')
res['tail'] = enable_tail(net, R0, S, 'trained')
L_live = live_units(R0)
res['live'] = {st: [int(v) for v in L_live[st]] for st in STG}
log('live units: ' + ', '.join(f'{st} {len(v)}' for st, v in L_live.items()))

# h4 is affine in S on its on-region: check on one patch (iso-sum, c = 1.25, arm A)
cols, _ = patch_values(S, sd, 0, [50], 0.3, 'sum'); Rp, Sp = run_patch(net, S, cols)
K4 = net.KB['d4'][0].reshape(64, -1)
pred = (Sp - S) @ K4; on2 = (R0['h4'] > 0) & (Rp['h4'] > 0); dh = Rp['h4'] - R0['h4']
sel = on2 & (np.abs(pred) > 1e-6)
res['h4_affine_check'] = dict(n=int(sel.sum()), slope=float((pred[sel] @ dh[sel]) / (pred[sel] @ pred[sel])),
                              corr=float(np.corrcoef(pred[sel], dh[sel])[0, 1]))
log(f'h4 affine on-region check (iso-sum e 0.3): {res["h4_affine_check"]}')
OUT_JSON.write_text(json.dumps(res))

# ====================================================================== 1. main arms
xr = x.astype(np.float64); M0 = np.where(real, xr[..., 0], 0).sum(1); M2 = np.where(real, xr[..., 0] * (xr[..., 1] ** 2 + xr[..., 2] ** 2), 0).sum(1)
AFF = {}
for u, Mv in ((0, M0), (50, M2)):
    ok = S[:, u] > 0; a_, b_ = np.polyfit(S[ok, u], Mv[ok], 1); r_ = np.corrcoef(S[ok, u], Mv[ok])[0, 1]
    AFF[u] = (float(a_), float(b_)); res.setdefault('affine_fit', {})[f'S{u}'] = dict(a=float(a_), b=float(b_), r=float(r_), n=int(ok.sum()))
log(f'affine fits moment = a S + b: {res["affine_fit"]}')
res['arms'] = {}
for arm, ks in ARMS.items():
    sc = scan(net, R0, S, sd, 0, ks, f'arm {arm}', aff=AFF if arm == 'C' else None)
    noclip = ~sc['clip']
    a = dict(clip_frac=float(sc['clip'].mean()), N_noclip=int(noclip.sum()))
    a['units_noclip'] = summarize(sc, R0, noclip, f'arm {arm} no-clip all classes')
    a['units_all'] = summarize(sc, R0, SUBS['all'], f'arm {arm} all jets', boot=False)
    a['angle'] = angle_analysis(sc, S, sd, 0, ks, noclip, f'arm {arm}')
    a['named'] = {}; a['calls'] = {}; a['units_by_class'] = {}
    for sn, sm in SUBS.items():
        m = sm & noclip
        a['named'][sn] = named(sc, m); a['calls'][sn] = calls_tab(sc, m)
        if sn != 'all': a['units_by_class'][sn] = summarize(sc, R0, m, f'arm {arm} no-clip {sn}', boot=False)
        log(f'arm {arm} {sn} (n {int(m.sum())}): named ' + '; '.join(f'{k} R {v["R"]:.3f} [{v["R_ci"][0]:.3f},{v["R_ci"][1]:.3f}] p {v["p"]:.1e}'
                                                                 for k, v in a['named'][sn].items()))
        log(f'   calls mean over c: {a["calls"][sn]["mean_over_c"]}; per c ' + ' '.join(f'{c}:{v["prod_changed"]:.3f}/{v["sum_changed"]:.3f}'
                                                                                 for c, v in a['calls'][sn].items() if c != 'mean_over_c'))
    # rule predictions (b: random ReLU layers on the trained S included), 3 seeds
    a['rules'] = {}
    for s_ in range(3):
        acc_r, base_r = rule_pops(sc['Spatch'], S, sd, 0, ks, np.random.default_rng(500 + s_), aff=AFF if arm == 'C' else None)
        a['rules'][str(s_)] = rule_summary(acc_r, base_r, noclip)
        if s_ == 0:
            Lt = {}
            for nm in ('rand1_S', 'rand2_S'):
                Ap, As = acc_r[nm]['prod'], acc_r[nm]['sum']
                u = unit_stats(Ap, As, noclip)
                ok = ((Ap[noclip].mean(0) + As[noclip].mean(0)) / (2 * len(EPS))) > 0.01 * np.maximum(base_r[nm][noclip].std(0), 1e-12)
                ok &= base_r[nm].std(0) > 1e-9
                idx = np.flatnonzero(ok); ph = holm(u['p'][idx]) if len(idx) else []
                Lt[nm] = dict(n=int(len(idx)), n_R_below_1_holm05=int(((u['R'][idx] < 1) & (np.asarray(ph) < 0.05)).sum()) if len(idx) else 0,
                              n_R_above_1_holm05=int(((u['R'][idx] > 1) & (np.asarray(ph) < 0.05)).sum()) if len(idx) else 0,
                              log10R=np.log10(np.clip(u['R'][idx], 1e-6, 1e6)).tolist())
            a['rules_tests_seed0'] = Lt
        del acc_r, base_r
        log(f'arm {arm} rules seed {s_}: ' + ' | '.join(f'{k}: n {v.get("n")} log10R med {v.get("log10R_median")} R<0.3 {v.get("frac_R_below_0p3")} '
                                                     f'Rpool {v.get("R_pooled_sd")}' for k, v in a['rules'][str(s_)].items()))
    res['arms'][arm] = a
    if arm == 'A': scA = sc
    else: del sc
    OUT_JSON.write_text(json.dumps(res))

# ====================================================================== 2. control (a): random pairs of live S units, same standardised displacement
dz0, dzk = {}, {}
for kind in ('prod', 'sum'):
    for c in EPS:
        cols, _ = patch_values(S, sd, 0, [50], c, kind)
        dz0[(kind, c)] = (cols[0] - S[:, 0]) / sd[0]; dzk[(kind, c)] = (cols[50] - S[:, 50]) / sd[50]
liveS = [int(k) for k in np.flatnonzero((sd > 1e-6) & ((S > 0).mean(0) > 0.02)) if k not in (0, 50, 29)]
res['liveS_pool'] = liveS
rngp = np.random.default_rng(SEED + 7)
ctrl = []; noclipA = ~scA['clip']
for d_ in range(NDRAW):
    A_, B_ = (int(v) for v in rngp.choice(liveS, 2, replace=False))
    sc = scan(net, R0, S, sd, A_, [B_], f'pair {d_} (S{A_}, S{B_})', disp_fn=lambda c, kind: (dz0[(kind, c)], dzk[(kind, c)]), keep_per_c=False)
    m = noclipA & ~sc['clip']
    s_ = summarize(sc, R0, m, f'pair {d_}', boot=False)
    nz = {}
    Lz = list(sc['L']['z'])
    for u in NAMED_Z:
        if u in Lz: i = Lz.index(u); nz[f'z{u}'] = float(sc['acc']['prod']['z'][m, i].sum() / max(sc['acc']['sum']['z'][m, i].sum(), 1e-30))
    nz['W-Z margin'] = float(sc['acc']['prod']['m'][m, 0].sum() / max(sc['acc']['sum']['m'][m, 0].sum(), 1e-30))
    cp = np.stack([sc['calls']['prod'][c][m] for c in EPS]).mean(); cs_ = np.stack([sc['calls']['sum'][c][m] for c in EPS]).mean()
    ctrl.append(dict(units=[A_, B_], N=int(m.sum()), stages={st: {k: s_[st][k] for k in ('n_responsive', 'log10R_median', 'frac_R_below_0p3', 'R_pooled_sd')} for st in s_},
                     log10R={st: np.log10(np.clip(np.array(s_[st]['R'])[np.array(s_[st]['responsive'])], 1e-6, 1e6)).tolist() for st in s_},
                     named=nz, calls_prod=float(cp), calls_sum=float(cs_), clip_frac=float(sc['clip'].mean())))
    log(f'pair {d_}: named {nz}; calls {cp:.3f}/{cs_:.3f}')
    del sc
res['control_random_pairs'] = ctrl
# permutation-style comparison: network per-unit log10 R vs pooled random-pair per-unit log10 R per stage
cmp = {}
netU = res['arms']['A']['units_noclip']
for st in STG + ('m',):
    nv = np.log10(np.clip(np.array(netU[st]['R'])[np.array(netU[st]['responsive'])], 1e-6, 1e6))
    cv = np.array([v for d in ctrl for v in d['log10R'][st]]); dm = [d['stages'][st]['log10R_median'] for d in ctrl if d['stages'][st]['log10R_median'] is not None]
    if len(nv) == 0 or len(dm) == 0: continue
    med = float(np.median(nv)); rank = float(np.mean(np.array(dm) <= med))
    cmp[st] = dict(net_median=med, ctrl_draw_medians_q05_q50_q95=[float(np.percentile(dm, q)) for q in (5, 50, 95)],
                   frac_draws_with_median_le_net=rank, p_two_sided=float(min(1.0, 2 * min((np.sum(np.array(dm) <= med) + 1) / (len(dm) + 1),
                                                                                       (np.sum(np.array(dm) >= med) + 1) / (len(dm) + 1)))))
for nm in ['z0', 'z7', 'z14', 'z10', 'W-Z margin']:
    if nm not in res['arms']['A']['named']['all']: continue
    v = res['arms']['A']['named']['all'][nm]['R']; dv = np.array([d['named'].get(nm, np.nan) for d in ctrl]); dv = dv[np.isfinite(dv)]
    cmp[nm] = dict(net_R=v, ctrl_R_q05_q50_q95=[float(np.percentile(dv, q)) for q in (5, 50, 95)],
                   p_two_sided=float(min(1.0, 2 * min((np.sum(dv <= v) + 1) / (len(dv) + 1), (np.sum(dv >= v) + 1) / (len(dv) + 1)))))
cp = np.array([d['calls_prod'] for d in ctrl]); cs_ = np.array([d['calls_sum'] for d in ctrl])
cmp['calls'] = dict(prod_q05_q50_q95=[float(np.percentile(cp, q)) for q in (5, 50, 95)], sum_q05_q50_q95=[float(np.percentile(cs_, q)) for q in (5, 50, 95)])
res['control_random_pairs_vs_net'] = cmp
log(f'random pairs vs network: {cmp}')
del scA
OUT_JSON.write_text(json.dumps(res))

# ====================================================================== 3. control (c): untrained network
unet = Jedi(64, untrained=True)
U0 = unet.run(x, keep=KEEP, batch=2500); Su = U0['S'].astype(np.float64); sdu = Su.std(0)
pair = (0, 50)
if not (sdu[0] > 1e-6 and sdu[50] > 1e-6):
    o = np.argsort(-sdu); pair = (int(o[0]), int(o[1]))
res['untrained'] = dict(acc=float((U0['logits'].argmax(1) == y).mean()), pair=list(pair), sd=[float(sdu[pair[0]]), float(sdu[pair[1]])])
res['untrained']['tail'] = enable_tail(unet, U0, Su, 'untrained')
sc = scan(unet, U0, Su, sdu, pair[0], [pair[1]], 'untrained', keep_per_c=False)
m = ~sc['clip']
res['untrained'].update(clip_frac=float(sc['clip'].mean()), units=summarize(sc, U0, m, 'untrained', boot=True), calls=calls_tab(sc, m))
del sc
log(f'untrained: acc {res["untrained"]["acc"]:.3f}, pair {pair}, calls {res["untrained"]["calls"]["mean_over_c"]}')
OUT_JSON.write_text(json.dumps(res))

# ====================================================================== 4. calibration: real input edits
def edit(xx, c, k):
    e = xx.copy()
    e[..., 0] = np.where(real, e[..., 0] * np.float32(c), e[..., 0])
    e[..., 1:] = np.where(real[..., None], e[..., 1:] * np.float32(k), e[..., 1:])
    return e


res['calibration'] = {}
others = [k for k in liveS]
for nm, kf in (('isoprod_real: pT x c, angles x 1/c', lambda c: 1 / c), ('M0only_real: pT x c, angles x c^-1/2', lambda c: c ** -0.5)):
    rows = {}
    for c in CS:
        xe = edit(x, c, kf(c)); ir = in_range(net, xe)
        Re = net.run(xe, keep=KEEP, batch=2500); Se = Re['S']
        ok = ir & (S[:, 0] > 0) & (S[:, 50] > 0) & (S[:, 29] > 0)
        row = dict(in_range=int(ir.sum()),
                   S0_ratio_median=float(np.median(Se[ok, 0] / S[ok, 0])), S50_ratio_median=float(np.median(Se[ok, 50] / S[ok, 50])),
                   S29_ratio_median=float(np.median(Se[ok, 29] / S[ok, 29])),
                   S0xS50_ratio_median=float(np.median(Se[ok, 0] * Se[ok, 50] / (S[ok, 0] * S[ok, 50]))),
                   dS0_sd=float(np.mean(np.abs(Se[ir, 0] - S[ir, 0])) / sd[0]), dS50_sd=float(np.mean(np.abs(Se[ir, 50] - S[ir, 50])) / sd[50]),
                   dS29_sd=float(np.mean(np.abs(Se[ir, 29] - S[ir, 29])) / sd[29]),
                   dSother_sd_mean=float(np.mean([np.mean(np.abs(Se[ir, k] - S[ir, k])) / sd[k] for k in others])),
                   dSother_sd_max=float(np.max([np.mean(np.abs(Se[ir, k] - S[ir, k])) / sd[k] for k in others])),
                   calls_changed=float((Re['logits'][ir].argmax(1) != call0[ir]).mean()),
                   dmargin_abs=float(np.mean(np.abs(margin(Re['logits'][ir]) - margin(R0['logits'][ir])))),
                   dz_abs={f'z{u}': float(np.mean(np.abs(Re['z'][ir, u] - R0['z'][ir, u]))) for u in NAMED_Z})
        W = ir & (y == 2) & (call0 == 2); row['W_to_Z'] = float((Re['logits'][W].argmax(1) == 3).mean())
        rows[str(c)] = row
        log(f'calib {nm} c {c}: {row}')
    res['calibration'][nm] = rows
# S-patch counterparts at the same c (arm A, all jets): |d margin|, |d z|
res['wall_s'] = float(time.time() - t0)
OUT_JSON.write_text(json.dumps(res))
log(f'done -> {OUT_JSON}')
