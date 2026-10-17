"""E21 verification (skeptic): fresh sample, same S-level iso-product / iso-sum patches, plus the missing controls.

Sample: sample(per_class=2000, seed=91), 10,000 jets, exact Jedi(64) (3-feature n64).  Patches as in e21_isoproduct_jedi.py
(same patch formulas, copied): arms A (S0, S50), B (S0, S50, S29), C (S0, S50 with affine moments M0 = a S0 + b, M2 = a S50 + b).
R = sum_j |dU|_iso-prod / sum_j |dU|_iso-sum over the six steps (+-0.15, 0.3, 0.5 sd), U in {z0, z7, z14, z10, W - Z margin}.

Added here:
 1. Same-jet comparison for control (a): each random S-unit pair (arm A displacements) is compared with the network's arm A R on
    the SAME jets (E21 compared the network on 6,355 jets with each draw on its own 341-5,560 jets).
 2. Arm B controls (missing in E21): arm B displacements applied to (S0, S50, random k) and to random triples (20 draws each),
    network arm B R on the same jets.
 3. Untrained control with a pair of untrained S units that are non-zero on most jets.
 4. Gradient-ratio test (read-out free, geometry free): per jet, central differences dU/dS0 and dU/dS50 (step 0.2 sd, all other S
    fixed).  For U = f(S0 * S50) the ratio is exact: dU/dS50 / dU/dS0 = S0 / S50, so the across-jet OLS slope of
    log(g50 / g0) on log(S0 / S50) is 1 (for f(M0 M2) with affine moments: slope 1 on log(m0 / m50)); a fixed-weight sum gives 0.
    Controls: synthetic product / sum units through the same finite differences (slope 1 / 0 check), random ReLU layers on S
    (5 seeds, random linear read-out), 20 random S-unit pairs (A, B) of the trained network (log(gB / gA) on log(SA / SB)), and
    the untrained network's pair.
Writes $AI_RESULTS/jedi/e21_verify.json.   Run: python -m interp.e21_verify
"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT_DIR = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
OUT_JSON = OUT_DIR / 'e21_verify.json'
PER, SEED = 2000, 91
EPS = (-0.5, -0.3, -0.15, 0.15, 0.3, 0.5)
KINDS = ('prod', 'sum')
STG = ('h4', 'h5', 'z'); KEEP = ('S',) + STG + ('logits',)
TG = ['z0', 'z7', 'z14', 'z10', 'm']
NBOOT = 300; NDRAW = 20; HGR = 0.2
t0 = time.time()


def log(*a):
    print(f'[{time.time() - t0:6.0f}s]', *a, flush=True)


from interp.jedi_core import Jedi, sample  # noqa: E402

OUT_DIR.mkdir(parents=True, exist_ok=True)
x, y, Q = sample(per_class=PER, seed=SEED); del Q
x = np.ascontiguousarray(x).astype(np.float32); J = len(x); real = x[..., 0] > 0
res = dict(N=J, per_class=PER, seed=SEED, eps=EPS, nboot=NBOOT, ndraw=NDRAW, grad_step_sd=HGR)
WB = np.random.default_rng(SEED + 1).poisson(1.0, size=(NBOOT, J)).astype(np.float64)
log(f'sample N {J}')


def save(): OUT_JSON.write_text(json.dumps(res))


def mcnemar(a, b):
    n10 = int((a & ~b).sum()); n01 = int((~a & b).sum()); n = n10 + n01
    if n == 0: return dict(n10=0, n01=0, p=1.0)
    k = min(n10, n01)
    lp = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(k + 1)]
    mx = max(lp); return dict(n10=n10, n01=n01, p=float(min(1.0, 2 * math.exp(mx) * sum(math.exp(v - mx) for v in lp))))


def pval_z(d):
    s = d.std(ddof=1) / math.sqrt(len(d))
    return 1.0 if s <= 0 else float(math.erfc(abs(d.mean() / s) / math.sqrt(2)))


def patch_values(S, sd, i0, ks, e, kind, aff=None):
    """identical formulas to e21_isoproduct_jedi.patch_values (formula branch)"""
    out = {}; flag = np.zeros(len(S), bool); us = [i0] + list(ks)
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
        return net_.run_jetwise(x, {'S': (m, Sp)}, keep=KEEP[1:], batch=2500)
    f, shp = TAILS[id(net_)]
    r = f(Sp.astype(np.float32).reshape((len(Sp),) + shp))
    return {k: np.asarray(v, np.float64) for k, v in r.items()}


def enable_tail(net_, R0_, S_, tag):
    for shp in ((1, 64), (64,)):
        try:
            TAILS[id(net_)] = (make_tail(net_), shp)
            o_ = np.argsort(-S_.std(0)); cols = patch_values(S_, S_.std(0), int(o_[0]), [int(o_[1])], 0.3, 'sum')[0]
            Re = run_patch(net_, S_, cols, exact=True); Rt = run_patch(net_, S_, cols); Rb = run_patch(net_, S_, {})
            d = max(float(np.abs(Re[k] - Rt[k]).max()) for k in Re); d0 = max(float(np.abs(Rb[k] - R0_[k]).max()) for k in Re)
            log(f'{tag}: tail {shp}: max |diff| vs run_jetwise {d:.2e}, vs base {d0:.2e}')
            if d < 1e-6 and d0 < 1e-6: return dict(shape=list(shp), max_diff_patch=d, max_diff_base=d0)
        except Exception as e:  # noqa: BLE001
            log(f'{tag}: tail {shp} failed: {e!r}')
        TAILS.pop(id(net_), None)
    return dict(shape=None)


def margin(lo): return lo[:, 2] - lo[:, 3]


def tvals(R): return dict(z0=R['z'][:, 0], z7=R['z'][:, 7], z14=R['z'][:, 14], z10=R['z'][:, 10], m=margin(R['logits']))


def live_units(R0):
    L = {}
    for st in STG:
        b = R0[st]; sd_ = b.std(0); on = (b > 0).mean(0)
        L[st] = np.flatnonzero((sd_ > 1e-6) & ((on > 0.02) if st != 'z' else True))
    return L


def arm_disp(S, sd, i0, ks, aff=None):
    D = {}; flag = np.zeros(J, bool)
    for kind in KINDS:
        for c in EPS:
            cols, fl = patch_values(S, sd, i0, ks, c, kind, aff); flag |= fl
            D[(kind, c)] = [(cols[u] - S[:, u]) / sd[u] for u in [i0] + list(ks)]
    return D, flag


def scan(net_, R0, S, sd, units, D, flag0, stages=True, signed=False):
    L = live_units(R0); call0 = R0['logits'].argmax(1); T0 = tvals(R0)
    acc = {k: {st: np.zeros((J, len(L[st]))) for st in STG} for k in KINDS} if stages else None
    tac = {k: {t: np.zeros(J) for t in TG} for k in KINDS}; sig = {k: {} for k in KINDS}
    calls = {k: {} for k in KINDS}; clip = flag0.copy()
    for kind in KINDS:
        for c in EPS:
            cols = {}
            for u, dz in zip(units, D[(kind, c)]):
                v = S[:, u] + dz * sd[u]; clip |= v < -1e-9; cols[u] = np.maximum(v, 0)
            Rp = run_patch(net_, S, cols)
            if stages:
                for st in STG: acc[kind][st] += np.abs(Rp[st][:, L[st]] - R0[st][:, L[st]])
            T = tvals(Rp)
            for t in TG:
                d = T[t] - T0[t]; tac[kind][t] += np.abs(d)
                if signed: sig[kind][(c, t)] = d
            calls[kind][c] = Rp['logits'].argmax(1) != call0
    return dict(L=L, acc=acc, tac=tac, sig=sig, calls=calls, clip=clip)


def Rci(ap, as_, m):
    R = float(ap[m].sum() / max(as_[m].sum(), 1e-30)); w = WB[:, m]
    Rb = (w @ ap[m]) / np.maximum(w @ as_[m], 1e-30)
    return dict(R=R, ci=[float(np.percentile(Rb, 2.5)), float(np.percentile(Rb, 97.5))], p_R1=pval_z(ap[m] - as_[m]), n=int(m.sum()))


def named(sc, m): return {t: Rci(sc['tac']['prod'][t], sc['tac']['sum'][t], m) for t in TG}


def calls_tab(sc, m):
    a = np.stack([sc['calls']['prod'][c][m] for c in EPS]); b = np.stack([sc['calls']['sum'][c][m] for c in EPS])
    return dict(prod=float(a.mean()), sum=float(b.mean()),
                mcnemar={str(c): mcnemar(sc['calls']['prod'][c][m], sc['calls']['sum'][c][m]) for c in (-0.5, 0.5)})


def stage_medians(sc, R0, m):
    out = {}
    for st in STG:
        Ap, As = sc['acc']['prod'][st][m], sc['acc']['sum'][st][m]; sd_ = R0[st][:, sc['L'][st]].std(0)
        ok = ((Ap.mean(0) + As.mean(0)) / (2 * len(EPS))) > 0.01 * np.maximum(sd_, 1e-12)
        if not ok.any(): out[st] = None; continue
        lr = np.log10(np.clip(Ap[:, ok].sum(0) / np.maximum(As[:, ok].sum(0), 1e-30), 1e-6, 1e6))
        out[st] = dict(n=int(ok.sum()), log10R_median=float(np.median(lr)), q25_q75=[float(np.percentile(lr, 25)), float(np.percentile(lr, 75))])
    return out


def rank_p(v, dv):
    dv = np.asarray(dv, float); dv = dv[np.isfinite(dv)]
    return float(min(1.0, 2 * min((np.sum(dv <= v) + 1) / (len(dv) + 1), (np.sum(dv >= v) + 1) / (len(dv) + 1))))


# ---------------------------------------------------------------- gradient-ratio test
def wslope(X, Y, w):
    sw = w.sum(); mx = (w * X).sum() / sw; my = (w * Y).sum() / sw
    return float((w * (X - mx) * (Y - my)).sum() / max((w * (X - mx) ** 2).sum(), 1e-30))


def ratio_slope(gi, gk, Xi, Xk, ok):
    """OLS slope of log(gk / gi) on log(Xi / Xk) over jets with ok, both gradients non-zero and of the same sign"""
    nz = ok & (np.abs(gi) > 1e-9) & (np.abs(gk) > 1e-9)
    sel = nz & (gi * gk > 0) & (Xi > 0) & (Xk > 0)
    out = dict(n_ok=int(ok.sum()), n_both_nonzero=int(nz.sum()), frac_same_sign=float((gi[nz] * gk[nz] > 0).mean()) if nz.any() else None,
               n=int(sel.sum()))
    if sel.sum() < 30: return out
    X = np.log(Xi[sel] / Xk[sel]); Y = np.log(gk[sel] / gi[sel])
    b = wslope(X, Y, np.ones(len(X))); wb = WB[:, sel]
    bs = np.array([wslope(X, Y, wb[i]) for i in range(NBOOT)])
    rx = np.argsort(np.argsort(X)); ry = np.argsort(np.argsort(Y))
    out.update(slope=b, ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
               spearman=float(np.corrcoef(rx, ry)[0, 1]), sd_X=float(X.std()), sd_Y=float(Y.std()))
    return out


def grads(net_, S, sd, u, h=HGR):
    hu = h * sd[u]
    Tp = tvals(run_patch(net_, S, {u: S[:, u] + hu})); Tm = tvals(run_patch(net_, S, {u: np.maximum(S[:, u] - hu, 0)}))
    return {t: (Tp[t] - Tm[t]) / (2 * hu) for t in TG}, S[:, u] > hu


# ====================================================================== 0. network, checks
net = Jedi(64)
R0 = net.run(x, keep=KEEP, batch=2500); call0 = R0['logits'].argmax(1)
res['acc'] = float((call0 == y).mean())
S = R0['S'].astype(np.float64); sd = S.std(0); mu = S.mean(0)
res['S_carriers'] = {str(k): dict(mean=float(mu[k]), sd=float(sd[k]), frac_zero=float((S[:, k] <= 0).mean())) for k in (0, 50, 29)}
res['tail'] = enable_tail(net, R0, S, 'trained')
Rid = run_patch(net, S, {0: S[:, 0], 50: S[:, 50], 29: S[:, 29]}, exact=True)
res['identity_max_abs_diff'] = {st: float(np.abs(Rid[st] - R0[st]).max()) for st in STG + ('logits',)}
try:
    q = net.L['d4'].iq.quantizer
    bits = {a: np.asarray(getattr(q, a)).reshape(-1) for a in ('k', 'i', 'f')}
    res['d4_input_quantizer'] = {f'S{u}': {a: float(v[u]) if len(v) == 64 else float(v[0]) for a, v in bits.items()} for u in (0, 50, 29)}
except Exception as e:  # noqa: BLE001
    res['d4_input_quantizer'] = repr(e)
log(f'acc {res["acc"]:.4f}; identity {res["identity_max_abs_diff"]}; carriers {res["S_carriers"]}; d4 iq {res["d4_input_quantizer"]}')
save()

xr = x.astype(np.float64); M0 = np.where(real, xr[..., 0], 0).sum(1)
M2 = np.where(real, xr[..., 0] * (xr[..., 1] ** 2 + xr[..., 2] ** 2), 0).sum(1)
AFF = {}
for u, Mv in ((0, M0), (50, M2)):
    ok = S[:, u] > 0; a_, b_ = np.polyfit(S[ok, u], Mv[ok], 1)
    AFF[u] = (float(a_), float(b_)); res.setdefault('affine_fit', {})[f'S{u}'] = dict(a=float(a_), b=float(b_), r=float(np.corrcoef(S[ok, u], Mv[ok])[0, 1]))
log(f'affine {res["affine_fit"]}')

# ====================================================================== 1. arms A, B, C on the fresh sample
ARMS = {'A': ([50], None), 'B': ([50, 29], None), 'C': ([50], AFF)}
res['arms'] = {}; SC = {}; DISP = {}
for arm, (ks, aff) in ARMS.items():
    D, fl = arm_disp(S, sd, 0, ks, aff); DISP[arm] = (D, fl)
    sc = scan(net, R0, S, sd, [0] + ks, D, fl, signed=(arm == 'A'))
    m = ~sc['clip']
    a = dict(clip_frac=float(sc['clip'].mean()), N_noclip=int(m.sum()), named=named(sc, m), calls=calls_tab(sc, m),
             stage_log10R=stage_medians(sc, R0, m),
             by_class={cl: dict(named={t: Rci(sc['tac']['prod'][t], sc['tac']['sum'][t], m & (y == ci))['R'] for t in ('z0', 'z7', 'm')},
                                calls=calls_tab(sc, m & (y == ci))) for cl, ci in (('W', 2), ('Z', 3), ('t', 4))})
    if arm == 'A':
        dz = D[('prod', 0.15)]
        th = np.degrees(np.arctan2(np.abs(dz[1]), np.abs(dz[0]))); edges = np.percentile(th[m], [25, 50, 75]); B = np.digitize(th, edges)
        a['angle'] = dict(edges=edges.tolist(), median=[float(np.median(th[m & (B == b)])) for b in range(4)],
                          R={t: [Rci(sc['tac']['prod'][t], sc['tac']['sum'][t], m & (B == b)) for b in range(4)] for t in ('z7', 'm', 'z0')})
    res['arms'][arm] = a; SC[arm] = sc
    log(f'arm {arm}: N {m.sum()} named ' + '; '.join(f'{t} {v["R"]:.3f} [{v["ci"][0]:.3f},{v["ci"][1]:.3f}]' for t, v in a['named'].items())
        + f' | calls {a["calls"]["prod"]:.3f}/{a["calls"]["sum"]:.3f} | stages {a["stage_log10R"]}')
    if arm == 'C': del SC['C']
    save()
res['arms']['A']['angle_log'] = {t: [round(v['R'], 3) for v in res['arms']['A']['angle']['R'][t]] for t in ('z7', 'm', 'z0')}
log(f'angle A: {res["arms"]["A"]["angle"]["median"]} {res["arms"]["A"]["angle_log"]}')

liveS = [int(k) for k in np.flatnonzero((sd > 1e-6) & ((S > 0).mean(0) > 0.02)) if k not in (0, 50, 29)]
res['liveS_pool'] = liveS
rng = np.random.default_rng(SEED + 7)


def control(arm, units_fn, tag):
    D, fl = DISP[arm]; scN = SC[arm]; rows = []
    for d_ in range(NDRAW):
        us = units_fn()
        sc = scan(net, R0, S, sd, us, D, np.zeros(J, bool), stages=False)
        m = ~sc['clip'] & ~scN['clip']
        row = dict(units=us, N=int(m.sum()))
        if m.sum() >= 100:
            row['ctrl'] = {t: float(sc['tac']['prod'][t][m].sum() / max(sc['tac']['sum'][t][m].sum(), 1e-30)) for t in TG}
            row['net_same_jets'] = {t: float(scN['tac']['prod'][t][m].sum() / max(scN['tac']['sum'][t][m].sum(), 1e-30)) for t in TG}
            row['calls_ctrl'] = calls_tab(sc, m); row['calls_net_same_jets'] = calls_tab(scN, m)
            for k in ('calls_ctrl', 'calls_net_same_jets'): row[k].pop('mcnemar')
        rows.append(row)
        log(f'{tag} {d_} {us} N {m.sum()}: ' + (' '.join(f'{t} ctrl {row["ctrl"][t]:.2f} net {row["net_same_jets"][t]:.2f}' for t in ('z7', 'm'))
                                                  + f' calls ctrl {row["calls_ctrl"]["prod"]:.3f}/{row["calls_ctrl"]["sum"]:.3f}' if 'ctrl' in row else ''))
    v = [r for r in rows if 'ctrl' in r]; summ = {}
    for t in TG:
        c_ = np.array([r['ctrl'][t] for r in v]); n_ = np.array([r['net_same_jets'][t] for r in v])
        netR = res['arms'][arm]['named'][t]['R']
        summ[t] = dict(net_R_full=netR, ctrl_q05_q50_q95=[float(np.percentile(c_, q)) for q in (5, 50, 95)], ctrl_min_max=[float(c_.min()), float(c_.max())],
                       rank_p_full=rank_p(netR, c_), frac_draws_net_below_ctrl_same_jets=float(np.mean(n_ < c_)),
                       log_ratio_net_over_ctrl_same_jets_median=float(np.median(np.log(n_ / c_))))
    cp = np.array([r['calls_ctrl']['prod'] for r in v]); cs = np.array([r['calls_ctrl']['sum'] for r in v])
    summ['calls'] = dict(prod_q05_q50_q95=[float(np.percentile(cp, q)) for q in (5, 50, 95)], sum_q05_q50_q95=[float(np.percentile(cs, q)) for q in (5, 50, 95)],
                         sum_minus_prod_q05_q50_q95=[float(np.percentile(cs - cp, q)) for q in (5, 50, 95)],
                         net=dict(prod=res['arms'][arm]['calls']['prod'], sum=res['arms'][arm]['calls']['sum']),
                         rank_p_sum_minus_prod=rank_p(res['arms'][arm]['calls']['sum'] - res['arms'][arm]['calls']['prod'], cs - cp))
    log(f'{tag} summary: {summ}')
    return dict(draws=rows, summary=summ)


# ====================================================================== 2. controls with arm A and arm B displacements
res['ctrl_A_pairs'] = control('A', lambda: [int(v) for v in rng.choice(liveS, 2, replace=False)], 'A pairs'); save()
res['ctrl_B_S0S50_plus_random'] = control('B', lambda: [0, 50, int(rng.choice(liveS))], 'B S0,S50,k'); save()
res['ctrl_B_triples'] = control('B', lambda: [int(v) for v in rng.choice(liveS, 3, replace=False)], 'B triples'); save()
del SC

# ====================================================================== 3. gradient-ratio test
G = {u: grads(net, S, sd, u) for u in (0, 50, 29)}
okA = G[0][1] & G[50][1]
gr = {}
for t in TG:
    gr[t] = dict(S0_S50=ratio_slope(G[0][0][t], G[50][0][t], S[:, 0], S[:, 50], okA),
                 S0_S50_affine=ratio_slope(G[0][0][t] / AFF[0][0], G[50][0][t] / AFF[50][0], AFF[0][0] * S[:, 0] + AFF[0][1],
                                           AFF[50][0] * S[:, 50] + AFF[50][1], okA),
                 S0_S29=ratio_slope(G[0][0][t], G[29][0][t], S[:, 0], S[:, 29], G[0][1] & G[29][1]))
res['grad_ratio_network'] = gr
log('grad ratio network: ' + ' | '.join(f'{t}: ' + ', '.join(f'{k} {v.get("slope", float("nan")):.3f} {v.get("ci")} n {v["n"]} same {v["frac_same_sign"]}' for k, v in gr[t].items()) for t in TG))
# synthetic units through the same finite differences
syn = {}
for nm, fn in (('log_product', lambda S_: np.log(np.maximum(S_[:, 0], 1e-9) * np.maximum(S_[:, 50], 1e-9))),
               ('product', lambda S_: S_[:, 0] * S_[:, 50]),
               ('sum_std', lambda S_: (S_[:, 0] - mu[0]) / sd[0] + (S_[:, 50] - mu[50]) / sd[50]),
               ('relu_window_on_sum', lambda S_: np.maximum((S_[:, 0] - mu[0]) / sd[0] + (S_[:, 50] - mu[50]) / sd[50] - 0.3, 0))):
    g = {}
    for u in (0, 50):
        hu = HGR * sd[u]; Sp, Sm = S.copy(), S.copy(); Sp[:, u] += hu; Sm[:, u] = np.maximum(Sm[:, u] - hu, 0)
        g[u] = (fn(Sp) - fn(Sm)) / (2 * hu)
    syn[nm] = ratio_slope(g[0], g[50], S[:, 0], S[:, 50], okA)
res['grad_ratio_synthetic'] = syn
log(f'grad ratio synthetic: {syn}')
# random ReLU layers on S with a random linear read-out (5 seeds)
live = sd > 1e-6
rr = []
for s_ in range(5):
    rg = np.random.default_rng(700 + s_)
    W1 = rg.normal(0, 1, (int(live.sum()), 64)) / np.sqrt(live.sum())
    pre0 = ((S[:, live] - mu[live]) / sd[live]) @ W1; b1 = -np.array([np.quantile(pre0[:, i], q) for i, q in enumerate(rg.uniform(0.2, 0.8, 64))])
    H0 = np.maximum(pre0 + b1, 0); hmu, hsd = H0.mean(0), H0.std(0); hl = hsd > 1e-9
    W2 = rg.normal(0, 1, (int(hl.sum()), 32)) / np.sqrt(hl.sum())
    pre2 = ((H0[:, hl] - hmu[hl]) / hsd[hl]) @ W2; b2 = -np.array([np.quantile(pre2[:, i], q) for i, q in enumerate(rg.uniform(0.2, 0.8, 32))])
    w3 = rg.normal(0, 1, 32)

    def f_(S_):
        h = np.maximum(((S_[:, live] - mu[live]) / sd[live]) @ W1 + b1, 0)
        return np.maximum(((h[:, hl] - hmu[hl]) / hsd[hl]) @ W2 + b2, 0) @ w3
    g = {}
    for u in (0, 50):
        hu = HGR * sd[u]; Sp, Sm = S.copy(), S.copy(); Sp[:, u] += hu; Sm[:, u] = np.maximum(Sm[:, u] - hu, 0)
        g[u] = (f_(Sp) - f_(Sm)) / (2 * hu)
    rr.append(ratio_slope(g[0], g[50], S[:, 0], S[:, 50], okA))
res['grad_ratio_random_relu'] = rr
log(f'grad ratio random ReLU: {[ (r.get("slope"), r.get("n")) for r in rr]}')
# random S-unit pairs of the trained network
rp = []; GC = {}
for d_ in range(NDRAW):
    A_, B_ = (int(v) for v in rng.choice(liveS, 2, replace=False))
    for u in (A_, B_):
        if u not in GC: GC[u] = grads(net, S, sd, u)
    rp.append(dict(units=[A_, B_], **{t: ratio_slope(GC[A_][0][t], GC[B_][0][t], S[:, A_], S[:, B_], GC[A_][1] & GC[B_][1]) for t in ('m', 'z7', 'z0')}))
    log(f'grad pair {A_},{B_}: ' + ' '.join(f'{t} {rp[-1][t].get("slope", float("nan")):.3f} n {rp[-1][t]["n"]}' for t in ('m', 'z7', 'z0')))
res['grad_ratio_random_pairs'] = rp
sm = {}
for t in ('m', 'z7', 'z0'):
    v = np.array([r[t].get('slope', np.nan) for r in rp]); v = v[np.isfinite(v)]
    sm[t] = dict(net_S0_S50=gr[t]['S0_S50'].get('slope'), pairs_q05_q50_q95=[float(np.percentile(v, q)) for q in (5, 50, 95)] if len(v) else None,
                 pairs_min_max=[float(v.min()), float(v.max())] if len(v) else None, n_draws=int(len(v)),
                 rank_p=rank_p(gr[t]['S0_S50'].get('slope', np.nan), v) if len(v) else None)
res['grad_ratio_summary'] = sm
log(f'grad ratio summary: {sm}')
save()

# ====================================================================== 4. untrained control with a usable pair
unet = Jedi(64, untrained=True)
U0 = unet.run(x, keep=KEEP, batch=2500); Su = U0['S'].astype(np.float64); sdu = Su.std(0); fnz = (Su > 0).mean(0)
cand = np.flatnonzero(sdu > 1e-6)
order = cand[np.lexsort((-sdu[cand], -fnz[cand]))]
res['untrained'] = dict(acc=float((U0['logits'].argmax(1) == y).mean()), n_live_S=int(len(cand)),
                        frac_nonzero_top=[[int(k), float(fnz[k]), float(sdu[k])] for k in order[:6]])
if len(order) >= 2:
    p0, p1 = int(order[0]), int(order[1])
    res['untrained']['pair'] = [p0, p1]
    res['untrained']['tail'] = enable_tail(unet, U0, Su, 'untrained')
    Du, flu = arm_disp(Su, sdu, p0, [p1])
    scu = scan(unet, U0, Su, sdu, [p0, p1], Du, flu)
    mu_ = ~scu['clip']
    res['untrained'].update(N_noclip=int(mu_.sum()), named=named(scu, mu_), stage_log10R=stage_medians(scu, U0, mu_), calls=calls_tab(scu, mu_))
    Gu = {u: grads(unet, Su, sdu, u) for u in (p0, p1)}
    res['untrained']['grad_ratio'] = {t: ratio_slope(Gu[p0][0][t], Gu[p1][0][t], Su[:, p0], Su[:, p1], Gu[p0][1] & Gu[p1][1]) for t in TG}
    log(f'untrained: acc {res["untrained"]["acc"]:.3f} pair {p0},{p1} N {mu_.sum()} named '
        + ' '.join(f'{t} {v["R"]:.3f}' for t, v in res['untrained']['named'].items()) + f' stages {res["untrained"]["stage_log10R"]}'
        + ' grad ' + ' '.join(f'{t} {v.get("slope")}' for t, v in res['untrained']['grad_ratio'].items()))
res['wall_s'] = float(time.time() - t0)
save()
log(f'done -> {OUT_JSON}')
