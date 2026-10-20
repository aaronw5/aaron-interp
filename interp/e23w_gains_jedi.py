"""E23w (claims A4, A9): functional form of JEDI-linear's mass read-out on natural off-centre jets.  Per stage (S, h4, h5,
z, W - Z logit margin), how strongly does each unit respond to the part of a rigid translation that is linear in the
centroid offset (odd term, c.M1) and to the part quadratic in the shift (even term, |c|^2 M0), relative to a matched real
change of 1/2 log(M0 M2)?  And (E21w) does any h4 / h5 / z unit depend on M0 and M2 through their product?

Model: official JEDI-linear 3-feature n64 (interp.jedi_core.Jedi(64)).  Reference networks (gain pattern at the outputs only):
  Deep Sets d64 seed 0 (float; pooled latent L + margin) and X12 HGQ-150 retrain seed 0 (S, z, margin).
Sample: sample(PER=5000, SEED=2311) (25,000 jets); --exclude E drops jets also drawn by sample(PER, E) (replicate:
  --seed 2312 --exclude 2311).  Unedited accuracy on the sample is checked against 0.806-0.813.
Moments about the stored axis (real particles): M0 = sum pT, M1 = sum pT (deta, dphi), M2 = sum pT r^2, d = |M1| / M0.
Sets: tail = top 5 % of d within each class (250 per class at PER 5000); bulk = 2,000 random jets with d below the median;
  product set = 2,000 random jets of the sample.

ARM 1, gains.  Edit: rigid translation c of every real particle (pT, slot order, padding unchanged), |c| in RINGS =
  {0.01, 0.02, 0.04, 0.06} rad at 8 azimuths (k * 45 deg): 32 cells + the unedited jet.  For each cell, M2' = M2 + 2 c.M1 +
  |c|^2 M0, prediction pred = 1/2 log(M2'/M2) = 1/2 dlog(M0 M2).  Matched real change ("cal"): the same jet dilated about the
  stored axis by exp(pred) (angles x exp(pred); M0 fixed, M2 x exp(2 pred), m x ~exp(pred)).
  Per jet and unit U (live S (19, E1), h4 / h5 units with sd > 0, all 16 z, margin = logit W - logit Z), with c and -c both on
  the grid: odd O = 1/2 (D(c) - D(-c)), even E = 1/2 (D(c) + D(-c)), for the translation (T) and for its matched real change (C).
    a1 (odd, centroid-linear gain) = coefficient of O_C in the regression of O_T on [O_C, c_eta, c_phi] (the two c terms take a
       fixed-direction odd response that does not follow M1, e.g. half-plane cuts) = sum O_T Ot_C / sum Ot_C^2, Ot_C = O_C
       residualised on c per unit;
    a2 (even gain) = sum E_T E_C / sum E_C^2.
  Pooled over the 16 (c, -c) pairs and the jets (primary), and per ring.  Stage values pool the units of the stage, each
  standardised by its sd on the sample.  Jet bootstrap (Poisson weights, NB = 1000) for every gain and for a1 / a2.
  Predictions: invariant under recentring (m, or any recentred spread): a1 = a2 = 0 (ratio undefined); single function of
  M0 M2 about the stored axis: a1 = a2 = 1; E23v read-out on z: a1 1.34-1.82, a2 0.62-0.71.
  Raw form (as planned): per unit, D_T regressed on x1 = c.M1/M0 and x2 = |c|^2 (all 32 cells), and the same regression on
  D_C; ratio (b1/b2)_T / (b1/b2)_C (M2 prediction 1).
  Location: first stage (S, h4, h5, z, margin) whose a1/a2 interval excludes 1; H0 a1/a2 = 1, normal p from the bootstrap sd,
  Holm over the 5 stages.
  Controls: (i) reflected tail jets (deta, dphi) -> (-deta, -dphi): M1 flips; a1 measured with the reflected jets' own M1
  must stay positive, and the reflected odd response projected on the ORIGINAL jets' Ot_C must flip sign (a1_cross ~ -a1);
  (ii) bulk jets (|M1| small): M1-linear odd amplitude must vanish; (iii) rotation about the stored axis with mean particle
  displacement |c| (M2 unchanged, pred 0): RMS response vs the translation's; (iv) untrained network (3 redraw seeds with >= 2
  live S units): stage gains.
ARM 2, product (E21w).  Input edits on the product set, step DL = 0.05 in log M0 and log M2:
  M2 edit: angles x exp(DL/2) (M0 fixed, M2 x exp(DL));
  M0 edit: pT of the 14 hardest real particles x f so that M0 x exp(DL), then angles x sqrt(M2/M2') so M2 is unchanged.
  g2(s0) = [U(s0 M0, +M2) - U(s0 M0, -M2)] / (2 DL) at s0 = +-; primary: jet-pooled mixed slope = [log|sum g2(+)| -
  log|sum g2(-)|] / (2 DL) (jet-bootstrap CI); secondary: per-jet slope, median over jets with g2(+) g2(-) > 0 (quantized
  finite differences tie exactly, g2(+) == g2(-), on many jets, which pins per-jet medians at 0).  A unit linear in M0 M2 gives 1; a unit
  additive in M0, M2 or in log M0, log M2 gives 0.  Also the per-jet ratio rho = (dU/dlogM2) / (dU/dlogM0) at the base point
  (any function of M0 M2: 1).
  Null: 200 random ordered pairs (A, B) of live S units of the trained network, patched at S (all other S kept) with per-jet steps
  of the same standardised size as the input edits move S (||dS/sd|| of the M0 edit for A, of the M2 edit for B); the same
  statistic per unit.  The pair (S0, S50) is also run.  Per unit: z = (net - null mean) / null sd, normal p and rank p;
  BH and Holm over units.  Untrained redraws (3 seeds with >= 2 live S units): the input-edit slope per unit.
  H0: every unit's slope inside the random-pair band; H1: some unit outside, with slope near 1.
  ./job.sh e23w-run 'python -m interp.e23w_gains_jedi'                                  -> $AI_RESULTS/jedi/e23w_gains.json
  ./job.sh e23w-verify 'python -m interp.e23w_gains_jedi --seed 2312 --exclude 2311 --tag verify' -> e23w_gains_verify.json
"""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
RINGS = (0.01, 0.02, 0.04, 0.06); NAZ = 8
LIVE_S = [0, 1, 13, 17, 19, 20, 29, 30, 31, 40, 43, 50, 51, 52, 56, 58, 60, 62, 63]     # E1: >= 1 % calls changed
NB = 1000; DL = 0.05; NPAIR = 200; HARD = 14; W_, Z_ = 2, 3
T0 = time.time()


def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, np.ndarray): return clean(o.tolist())
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    return o


def pnorm(z):
    return float(math.erfc(abs(z) / math.sqrt(2))) if np.isfinite(z) else None


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); adj = np.empty(m); run = 0.0
    for i, j in enumerate(o):
        run = max(run, min(1.0, (m - i) * p[j])); adj[j] = run
    return adj


def bh(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); q = np.empty(m); prev = 1.0
    for i in range(m - 1, -1, -1):
        j = o[i]; prev = min(prev, p[j] * m / (i + 1)); q[j] = prev
    return q


def ci(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [None, None]


# ------------------------------------------------------------------------------------------------ moments and edits
def moments(x):
    x = np.asarray(x, np.float64); pt, e, p = x[..., 0], x[..., 1], x[..., 2]; rl = pt > 0
    pt = np.where(rl, pt, 0); e = np.where(rl, e, 0); p = np.where(rl, p, 0)
    M0 = pt.sum(1); M1e = (pt * e).sum(1); M1p = (pt * p).sum(1); M2 = (pt * (e * e + p * p)).sum(1)
    r = np.sqrt(e * e + p * p); rbar = (pt * r).sum(1) / np.maximum(M0, 1e-12)
    return dict(M0=M0, M1e=M1e, M1p=M1p, M2=np.maximum(M2, 1e-12), d=np.hypot(M1e, M1p) / M0, rbar=rbar)


def shift(x, ce, cp):
    rl = x[..., 0] > 0; e = x.copy(); ce = np.broadcast_to(np.asarray(ce, np.float64), (len(x),)); cp = np.broadcast_to(np.asarray(cp, np.float64), (len(x),))
    e[..., 1] = np.where(rl, x[..., 1].astype(np.float64) + ce[:, None], x[..., 1]).astype(np.float32)
    e[..., 2] = np.where(rl, x[..., 2].astype(np.float64) + cp[:, None], x[..., 2]).astype(np.float32)
    return e


def scale_ang(x, k):
    rl = x[..., 0] > 0; e = x.copy(); k = np.broadcast_to(np.asarray(k, np.float64), (len(x),))[:, None]
    for c in (1, 2): e[..., c] = np.where(rl, x[..., c].astype(np.float64) * k, x[..., c]).astype(np.float32)
    return e


def rotate(x, th):
    rl = x[..., 0] > 0; e = x.copy(); de, dp = x[..., 1].astype(np.float64), x[..., 2].astype(np.float64)
    c, s = np.cos(th)[:, None], np.sin(th)[:, None]
    e[..., 1] = np.where(rl, c * de - s * dp, x[..., 1]).astype(np.float32)
    e[..., 2] = np.where(rl, s * de + c * dp, x[..., 2]).astype(np.float32)
    return e


def m0_edit(x, dl):
    """M0 x exp(dl) by rescaling the pT of the HARD hardest real particles, then a dilation that restores M2 exactly"""
    x64 = x.astype(np.float64); pt = np.where(x64[..., 0] > 0, x64[..., 0], 0); M = moments(x)
    rank = np.argsort(np.argsort(-pt, 1), 1); hard = (rank < HARD) & (pt > 0)
    M0h = np.where(hard, pt, 0).sum(1); f = 1 + (np.exp(dl) - 1) * M["M0"] / M0h
    e = x.copy(); e[..., 0] = np.where(hard, pt * f[:, None], x[..., 0]).astype(np.float32)
    M2n = moments(e)['M2']; return scale_ang(e, np.sqrt(M['M2'] / M2n))


def m2_edit(x, dl):
    return scale_ang(x, np.full(len(x), np.exp(dl / 2)))


# ------------------------------------------------------------------------------------------------ networks
class Feat:
    """unit vector of a network: stages -> live columns; F(x) -> (J, K) float64 plus stage slices"""
    def __init__(self, name, run, stages, live):
        self.name, self.run, self.stages, self.live = name, run, stages, live
        self.sl = {}; k = 0
        for st in stages:
            n = 1 if st == 'margin' else len(live[st]); self.sl[st] = slice(k, k + n); k += n
        self.K = k

    def __call__(self, x):
        R = self.run(x); out = []
        for st in self.stages:
            out.append((R['logits'][:, W_] - R['logits'][:, Z_])[:, None] if st == 'margin' else R[st][:, self.live[st]])
        return np.concatenate(out, 1)

    def units(self):
        u = []
        for st in self.stages: u += ['margin'] if st == 'margin' else [f'{st}{c}' for c in self.live[st]]
        return u


def jedi_feat(net, x_live, name, fixed_S=None):
    run = lambda xx: net.run(xx, keep=('S', 'h4', 'h5', 'z', 'logits'), batch=4000)
    R = run(x_live); live = {st: [int(c) for c in np.flatnonzero(R[st].std(0) > 1e-9)] for st in ('S', 'h4', 'h5', 'z')}
    if fixed_S is not None: live['S'] = list(fixed_S)
    live['z'] = list(range(16)) if fixed_S is not None else live['z']
    return Feat(name, run, ('S', 'h4', 'h5', 'z', 'margin'), live), R


# ------------------------------------------------------------------------------------------------ arm 1: gains
def cells():
    out = []
    for r in RINGS:
        for k in range(NAZ):
            a = 2 * np.pi * k / NAZ; out.append((r, k, r * np.cos(a), r * np.sin(a)))
    return out


def gain_arm(F, xs, M, sdU, keep_arrays=False, rot=True, seed=0):
    """-> per-jet sufficient statistics and summaries; sdU (K,) standardisation"""
    J = len(xs); C = cells(); U0 = F(xs); K = U0.shape[1]
    DT = np.zeros((len(C), J, K), np.float32); DC = np.zeros_like(DT); PRED = np.zeros((len(C), J))
    for i, (r, k, ce, cp) in enumerate(C):
        cM1 = ce * M['M1e'] + cp * M['M1p']
        pred = 0.5 * np.log1p((2 * cM1 + r * r * M['M0']) / M['M2']); PRED[i] = pred
        DT[i] = F(shift(xs, ce, cp)) - U0; DC[i] = F(scale_ang(xs, np.exp(pred))) - U0
    pairs = [(i, i + NAZ // 2) for i, (r, k, _, _) in enumerate(C) if k < NAZ // 2]
    ring = np.array([RINGS.index(C[i][0]) for i, _ in pairs]); cvec = np.array([[C[i][2], C[i][3]] for i, _ in pairs])
    OT = np.stack([0.5 * (DT[i] - DT[j]) for i, j in pairs]) / sdU; ET = np.stack([0.5 * (DT[i] + DT[j]) for i, j in pairs]) / sdU
    OC = np.stack([0.5 * (DC[i] - DC[j]) for i, j in pairs]) / sdU; EC = np.stack([0.5 * (DC[i] + DC[j]) for i, j in pairs]) / sdU
    # residualise O_C on the fixed direction c (per unit, pooled over jets): Frisch-Waugh for the coefficient of O_C
    A = np.einsum('pa,pb->ab', cvec, cvec) * J
    bC = np.linalg.solve(A, np.einsum('pa,pju->au', cvec, OC)); OCt = OC - np.einsum('pa,au->pu', cvec, bC)[:, None, :]
    bT = np.linalg.solve(A, np.einsum('pa,pju->au', cvec, OT))       # fixed-direction odd response of the translation
    st = {}
    for nm, sel in [('all', np.ones(len(pairs), bool))] + [(f'r{r}', ring == q) for q, r in enumerate(RINGS)]:
        st[nm] = dict(n1=(OT[sel] * OCt[sel]).sum(0), d1=(OCt[sel] ** 2).sum(0), n2=(ET[sel] * EC[sel]).sum(0), d2=(EC[sel] ** 2).sum(0))
    # raw regression on x1 = c.M1/M0, x2 = |c|^2 (all 32 cells)
    x1 = np.stack([(ce * M['M1e'] + cp * M['M1p']) / M['M0'] for (_, _, ce, cp) in C]); x2 = np.stack([np.full(J, r * r) for (r, _, _, _) in C])
    X = np.stack([x1, x2], -1)                                            # (32, J, 2)
    raw = dict(XtX=np.einsum('cja,cjb->jab', X, X), XtYT=np.einsum('cja,cju->jua', X, DT / sdU), XtYC=np.einsum('cja,cju->jua', X, DC / sdU))
    # amplitudes: RMS of the M1-linear odd term and of the fixed-direction odd term (per unit, standardised)
    amp = dict(m1_linear_rms=None, fixed_dir_rms=np.sqrt(((np.einsum('pa,au->pu', cvec, bT)) ** 2).mean(0)),
               odd_rms=np.sqrt((OT ** 2).mean((0, 1))), even_rms=np.sqrt((ET ** 2).mean((0, 1))),
               pred_rms_odd_cal=np.sqrt((OC ** 2).mean((0, 1))), pred_rms_even_cal=np.sqrt((EC ** 2).mean((0, 1))))
    a1 = st['all']['n1'].sum(0) / np.maximum(st['all']['d1'].sum(0), 1e-30)
    amp['m1_linear_rms'] = np.sqrt(((a1[None, None] * OCt) ** 2).mean((0, 1)))
    out = dict(st=st, raw=raw, amp=amp, pred_q=[float(np.percentile(PRED, q)) for q in (1, 50, 99)], J=J, K=K)
    if rot:                                                               # matched-displacement rotation per ring
        rng = np.random.default_rng(seed + 77); rr = {}
        for r in RINGS:
            th = 2 * np.arcsin(np.minimum(1.0, r / np.maximum(2 * M['rbar'], 1e-12))) * rng.choice([-1, 1], J)
            DR = (F(rotate(xs, th)) - U0) / sdU; DTr = DT[[i for i, c in enumerate(C) if c[0] == r]] / sdU
            rr[f'r{r}'] = dict(rot_rms=np.sqrt((DR ** 2).mean(0)), trans_rms=np.sqrt((DTr ** 2).mean((0, 1))), rot_mean=DR.mean(0))
        out['rot'] = rr
    if keep_arrays: out['arr'] = dict(OT=OT, OCt=OCt)
    return out


def summarize(G, F, WB, units_idx=None):
    """stage / unit gains with bootstrap from per-jet stats G['st'] (each (J, K))"""
    res = {}

    def gains(sl_units, nm):
        s = G['st'][nm]; n1, d1, n2, d2 = (s[k][:, sl_units].sum(1) for k in ('n1', 'd1', 'n2', 'd2'))
        a1, a2 = n1.sum() / d1.sum(), n2.sum() / d2.sum()
        b1 = (WB @ n1) / (WB @ d1); b2 = (WB @ n2) / (WB @ d2); rb = b1 / b2; r = a1 / a2
        sd = np.nanstd(rb); z = (r - 1) / sd if sd > 0 else np.nan
        pb = float(min(1.0, 2 * min((rb <= 1).mean(), (rb >= 1).mean()))) if np.all(np.isfinite(rb)) else None
        return dict(a1=float(a1), a1_ci=ci(b1), a2=float(a2), a2_ci=ci(b2), ratio=float(r), ratio_ci=ci(rb), p_ratio1_normal=pnorm(z),
                    p_ratio1_boot=pb, p_a1_eq1=pnorm((a1 - 1) / np.std(b1)), p_a2_eq1=pnorm((a2 - 1) / np.std(b2)),
                    p_a1_eq0=pnorm(a1 / np.std(b1)), p_a2_eq0=pnorm(a2 / np.std(b2)))
    for st_, sl in F.sl.items():
        idx = np.arange(sl.start, sl.stop)
        res[st_] = {nm: gains(idx, nm) for nm in G['st']}
        # raw regression, stage pooled
        XtX = G['raw']['XtX']; YT = G['raw']['XtYT'][:, idx].sum(1); YC = G['raw']['XtYC'][:, idx].sum(1)
        def rawfit(w):
            A = np.einsum('j,jab->ab', w, XtX); bt = np.linalg.solve(A, np.einsum('j,ja->a', w, YT)); bc = np.linalg.solve(A, np.einsum('j,ja->a', w, YC))
            return bt, bc
        bt, bc = rawfit(np.ones(G['J'])); rr = []
        for i in range(min(200, len(WB))):
            t_, c_ = rawfit(WB[i]); rr.append((t_[0] / t_[1]) / (c_[0] / c_[1]))
        res[st_]['raw'] = dict(b1_T=float(bt[0]), b2_T=float(bt[1]), b1_C=float(bc[0]), b2_C=float(bc[1]),
                               ratio_T=float(bt[0] / bt[1]), ratio_C=float(bc[0] / bc[1]), ratio_T_over_C=float((bt[0] / bt[1]) / (bc[0] / bc[1])),
                               ratio_T_over_C_ci=ci(rr))
        a = G['amp']; res[st_]['amp'] = {k: float(np.sqrt((np.asarray(v)[idx] ** 2).mean())) for k, v in a.items()}
        if 'rot' in G:
            res[st_]['rotation'] = {k: dict(rot_rms=float(np.sqrt((v['rot_rms'][idx] ** 2).mean())), trans_rms=float(np.sqrt((v['trans_rms'][idx] ** 2).mean())),
                                            rot_mean=float(v['rot_mean'][idx].mean())) for k, v in G['rot'].items()}
    if units_idx is not None:
        un = F.units(); s = G['st']['all']; pu = {}
        a1 = s['n1'].sum(0) / s['d1'].sum(0); a2 = s['n2'].sum(0) / s['d2'].sum(0)
        b1 = (WB @ s['n1']) / (WB @ s['d1']); b2 = (WB @ s['n2']) / (WB @ s['d2']); rb = b1 / b2
        for k in range(F.K):
            sd = np.nanstd(rb[:, k]); r = a1[k] / a2[k]
            pu[un[k]] = dict(a1=float(a1[k]), a1_ci=ci(b1[:, k]), a2=float(a2[k]), a2_ci=ci(b2[:, k]), ratio=float(r), ratio_ci=ci(rb[:, k]),
                             p_ratio1=pnorm((r - 1) / sd) if sd > 0 else None, cal_even_rms=float(G['amp']['pred_rms_even_cal'][k]),
                             cal_odd_rms=float(G['amp']['pred_rms_odd_cal'][k]))
        ps = [v['p_ratio1'] if v['p_ratio1'] is not None else 1.0 for v in pu.values()]
        for (k, v), h in zip(pu.items(), holm(ps)): v['p_ratio1_holm'] = float(h)
        res['units'] = pu
    return res


# ------------------------------------------------------------------------------------------------ arm 2: product
def make_tail(net):
    import jax, jax.numpy as jnp
    L = net.L; KB = {k: (jnp.asarray(net.KB[k][0], jnp.float32), jnp.asarray(net.KB[k][1], jnp.float32)) for k in ('d4', 'd5', 'd6', 'd7')}

    def dense(k, x, act=True):
        z = jnp.einsum(L[k].equation, L[k].iq(x, training=False), KB[k][0]) + KB[k][1]
        return L[k].activation(z) if act and L[k].activation is not None else z

    def f(S):
        J = S.shape[0]; h4 = dense('d4', S); h5 = dense('d5', h4); z = dense('d6', h5, act=False); lo = dense('d7', jax.nn.relu(z))
        return dict(h4=h4.reshape(J, -1), h5=h5.reshape(J, -1), z=z.reshape(J, -1), logits=lo.reshape(J, -1))
    fj = jax.jit(f)

    def run(S, shape):
        out = {k: [] for k in ('h4', 'h5', 'z', 'logits')}
        for i in range(0, len(S), 4000):
            r = fj(jnp.asarray(S[i:i + 4000].reshape((-1,) + shape), jnp.float32))
            for k in out: out[k].append(np.asarray(r[k], np.float64))
        return {k: np.concatenate(v) for k, v in out.items()}
    return run


def mixed_slope(Upp, Upm, Ump, Umm):
    """per jet, per unit: g2(s0) = (U(s0,+) - U(s0,-)) / (2 DL); slope = (log|g2(+)| - log|g2(-)|) / (2 DL); NaN if invalid"""
    gp = (Upp - Upm) / (2 * DL); gm = (Ump - Umm) / (2 * DL)
    ok = (gp * gm > 0) & (np.abs(gp) > 1e-7) & (np.abs(gm) > 1e-7)
    with np.errstate(divide='ignore', invalid='ignore'):
        s = np.where(ok, (np.log(np.abs(gp)) - np.log(np.abs(gm))) / (2 * DL), np.nan)
    return s


def pooled_slope(V, w=None):
    """jet-pooled mixed slope per unit: g2(s0) summed (weighted) over jets, slope = log(|G2(+)| / |G2(-)|) / (2 DL) if same sign.
    Pooling avoids the exact ties of quantized per-jet finite differences (per-jet g2(+) == g2(-) gives slope 0 exactly)."""
    gp = V[(1, 1)] - V[(1, -1)]; gm = V[(-1, 1)] - V[(-1, -1)]
    if w is None: A, B = gp.sum(0), gm.sum(0)
    else: A, B = w @ gp, w @ gm
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where((A * B > 0) & (np.abs(A) > 0), (np.log(np.abs(A)) - np.log(np.abs(B))) / (2 * DL), np.nan)


def unit_median(s, WBi):
    """median over valid jets per unit, bootstrap CI via resampled jet indices WBi (nb, J)"""
    med = np.nanmedian(s, 0); bs = []
    for I in WBi:
        bs.append(np.nanmedian(s[I], 0))
    bs = np.array(bs); return med, np.nanpercentile(bs, 2.5, 0), np.nanpercentile(bs, 97.5, 0), np.isfinite(s).mean(0)


def product_edits(xp):
    E = {}
    for s0 in (1, -1):
        xa = m0_edit(xp, s0 * DL); E[(s0, 0)] = xa
        for s2 in (1, -1): E[(s0, s2)] = m2_edit(xa, s2 * DL)
    for s2 in (1, -1): E[(0, s2)] = m2_edit(xp, s2 * DL)
    return E


def product_net(F, E, x0):
    U = {k: F(v) for k, v in E.items()}; U[(0, 0)] = F(x0)
    s = mixed_slope(U[(1, 1)], U[(1, -1)], U[(-1, 1)], U[(-1, -1)])
    s_alt = mixed_slope(U[(1, 1)], U[(-1, 1)], U[(1, -1)], U[(-1, -1)])        # d log|dU/dlogM0| / d log M2
    g2 = (U[(0, 1)] - U[(0, -1)]) / (2 * DL); g0 = (U[(1, 0)] - U[(-1, 0)]) / (2 * DL)
    with np.errstate(divide='ignore', invalid='ignore'):
        rho = np.where((np.abs(g0) > 1e-7) & (np.abs(g2) > 1e-7), g2 / g0, np.nan)
    return s, s_alt, rho, U


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--per', type=int, default=5000); ap.add_argument('--seed', type=int, default=2311)
    ap.add_argument('--exclude', type=int, default=None); ap.add_argument('--tag', default='')
    ap.add_argument('--nbulk', type=int, default=2000); ap.add_argument('--nprod', type=int, default=2000); ap.add_argument('--npair', type=int, default=NPAIR)
    a = ap.parse_args()
    from interp.jedi_core import Jedi, sample
    from interp.e23_tail_jedi import sample_idx
    OUT.mkdir(parents=True, exist_ok=True)
    JS = OUT / f'e23w_gains{"_" + a.tag if a.tag else ""}{"_" + str(a.per) if a.per != 5000 else ""}.json'
    R = dict(plan='module docstring', per_class=a.per, seed=a.seed, exclude=a.exclude, rings=RINGS, n_az=NAZ, nb=NB, DL=DL, hard=HARD)
    save = lambda: JS.write_text(json.dumps(clean(R)))
    x, y, _ = sample(a.per, a.seed); x = np.ascontiguousarray(x, np.float32)
    if a.exclude is not None:
        keep = ~np.isin(sample_idx(a.per, a.seed), sample_idx(a.per, a.exclude)); x, y = x[keep], y[keep]; R['n_overlap_removed'] = int((~keep).sum())
    J = len(x); R['N'] = J; rng = np.random.default_rng(a.seed + 5)
    net = Jedi(64)
    F, R0 = jedi_feat(net, x, 'jedi', fixed_S=LIVE_S)
    acc = float((R0['logits'].argmax(1) == y).mean()); R['acc'] = acc; R['acc_in_range'] = bool(0.806 <= acc <= 0.813)
    R['live'] = F.live; R['K'] = F.K
    log(f'sample N {J} (seed {a.seed}, exclude {a.exclude}); accuracy {acc:.4f}; units K {F.K}: ' + ', '.join(f'{s} {len(v)}' for s, v in F.live.items()))
    U0all = F(x); sdU = U0all.std(0); sdU[sdU < 1e-9] = 1.0
    M = moments(x); d = M['d']
    tail = np.sort(np.concatenate([np.flatnonzero((y == c) & (d >= np.quantile(d[y == c], 0.95))) for c in range(5)]))
    bulk = np.sort(rng.choice(np.flatnonzero(d < np.median(d)), min(a.nbulk, int((d < np.median(d)).sum())), replace=False))
    prod = np.sort(rng.choice(J, min(a.nprod, J), replace=False))
    sub = lambda I: {k: v[I] for k, v in M.items()}
    R['sets'] = {nm: dict(n=int(len(I)), n_class=[int((y[I] == c).sum()) for c in range(5)], d_mrad_p50=float(1e3 * np.median(d[I])),
                          d_mrad_p10_p90=[float(1e3 * np.percentile(d[I], 10)), float(1e3 * np.percentile(d[I], 90))],
                          M2_over_M0_p50=float(np.median(M['M2'][I] / M['M0'][I])), acc=float((R0['logits'][I].argmax(1) == y[I]).mean()))
                 for nm, I in (('tail', tail), ('bulk', bulk), ('prod', prod))}
    log('sets', json.dumps(clean(R['sets'])))
    # edit checks on tail
    xt = x[tail]; Mt = sub(tail); c = cells()[3]
    e = shift(xt, c[2], c[3]); Me = moments(e); pred = 0.5 * np.log1p((2 * (c[2] * Mt['M1e'] + c[3] * Mt['M1p']) + c[0] ** 2 * Mt['M0']) / Mt['M2'])
    Mc = moments(scale_ang(xt, np.exp(pred)))
    R['edit_checks'] = dict(shift_pred_vs_actual_max=float(np.abs(0.5 * np.log(Me['M2'] / Mt['M2']) - pred).max()),
                            shift_dM0_max=float(np.abs(Me['M0'] / Mt['M0'] - 1).max()),
                            cal_pred_vs_actual_max=float(np.abs(0.5 * np.log(Mc['M2'] / Mt['M2']) - pred).max()),
                            padding_unchanged=bool(np.array_equal(e[xt[..., 0] <= 0], xt[xt[..., 0] <= 0])))
    log('edit checks', R['edit_checks']); save()
    WBt = np.random.default_rng(a.seed + 11).poisson(1.0, (NB, len(tail))).astype(np.float64)
    WBb = np.random.default_rng(a.seed + 12).poisson(1.0, (NB, len(bulk))).astype(np.float64)

    # ---------------- arm 1, official
    G = gain_arm(F, xt, Mt, sdU, keep_arrays=True, seed=a.seed); R['gains'] = dict(tail=summarize(G, F, WBt, units_idx=True), pred_q_tail=G['pred_q'])
    log('tail gains (all rings): ' + ' | '.join(f'{s}: a1 {v["all"]["a1"]:.3f} {np.round(v["all"]["a1_ci"], 3)} a2 {v["all"]["a2"]:.3f} {np.round(v["all"]["a2_ci"], 3)} '
                                               f'ratio {v["all"]["ratio"]:.3f} {np.round(v["all"]["ratio_ci"], 3)}' for s, v in R['gains']['tail'].items() if s != 'units'))
    stg = list(F.sl); ps = [R['gains']['tail'][s]['all']['p_ratio1_normal'] or 1.0 for s in stg]
    hp = holm(ps); R['location'] = dict(stages=stg, p_ratio1=ps, p_holm=[float(v) for v in hp],
                                        first_stage_excluding_1=next((s for s, h in zip(stg, hp) if h < 0.05), None))
    log('location', R['location']); save()
    # reflected tail jets
    xr = scale_ang(xt, np.full(len(xt), -1.0)); Mr = moments(xr)
    Gr = gain_arm(F, xr, Mr, sdU, keep_arrays=True, rot=False, seed=a.seed)
    R['gains']['reflected'] = summarize(Gr, F, WBt)
    OT_r, OCt_o = Gr['arr']['OT'], G['arr']['OCt']; cross = {}
    for s, sl in F.sl.items():
        n = (OT_r[..., sl] * OCt_o[..., sl]).sum((0, 2)); dd = (OCt_o[..., sl] ** 2).sum((0, 2)); bb = (WBt @ n) / (WBt @ dd)
        cross[s] = dict(a1_cross=float(n.sum() / dd.sum()), ci=ci(bb), a1_orig=R['gains']['tail'][s]['all']['a1'])
    R['gains']['reflected_cross'] = cross
    log('reflected: ' + ' | '.join(f'{s}: a1 {v["all"]["a1"]:.3f} a2 {v["all"]["a2"]:.3f}; cross {cross[s]["a1_cross"]:.3f} {np.round(cross[s]["ci"], 3)}'
                                    for s, v in R['gains']['reflected'].items())); save()
    del Gr, OT_r
    # bulk
    Gb = gain_arm(F, x[bulk], sub(bulk), sdU, rot=False, seed=a.seed); R['gains']['bulk'] = summarize(Gb, F, WBb)
    log('bulk: ' + ' | '.join(f'{s}: a1 {v["all"]["a1"]:.3f} {np.round(v["all"]["a1_ci"], 2)} a2 {v["all"]["a2"]:.3f} m1-lin rms {v["amp"]["m1_linear_rms"]:.4f} '
                               f'(tail {R["gains"]["tail"][s]["amp"]["m1_linear_rms"]:.4f})' for s, v in R['gains']['bulk'].items())); save()
    del Gb, G

    # ---------------- arm 2, product (official)
    xp = x[prod]; Ep = product_edits(xp); Mp = moments(xp)
    R['product'] = dict(edit_checks={str(k): dict(dlogM0=float(np.median(np.log(moments(v)['M0'] / Mp['M0']))),
                                                  dlogM0_maxdev=float(np.abs(np.log(moments(v)['M0'] / Mp['M0']) - k[0] * DL).max()),
                                                  dlogM2_maxdev=float(np.abs(np.log(moments(v)['M2'] / Mp['M2']) - k[1] * DL).max())) for k, v in Ep.items()})
    log('product edit checks', R['product']['edit_checks'])
    s, s_alt, rho, Up = product_net(F, Ep, xp)
    WBi = np.random.default_rng(a.seed + 13).integers(0, len(xp), (300, len(xp)))
    un = F.units(); med, lo, hi, fv = unit_median(s, WBi); med2 = np.nanmedian(s_alt, 0); rmed = np.nanmedian(rho, 0)
    # S displacement of the edits (standardised) -> null step sizes
    runS = lambda xx: net.run(xx, keep=('S',), batch=4000)['S']
    S0p = runS(xp); sdS = R0['S'].std(0); sdS[sdS < 1e-9] = 1.0
    dS0 = (runS(Ep[(1, 0)]) - runS(Ep[(-1, 0)])) / 2; dS2 = (runS(Ep[(0, 1)]) - runS(Ep[(0, -1)])) / 2
    n0 = np.linalg.norm(dS0[:, LIVE_S] / sdS[LIVE_S], axis=1); n2 = np.linalg.norm(dS2[:, LIVE_S] / sdS[LIVE_S], axis=1)
    R['product']['S_step_sd'] = dict(M0_edit_p50=float(np.median(n0)), M2_edit_p50=float(np.median(n2)),
                                     top_units_M0=[[int(LIVE_S[i]), float(np.median(np.abs(dS0[:, LIVE_S[i]]) / sdS[LIVE_S[i]]))] for i in np.argsort(-np.median(np.abs(dS0[:, LIVE_S]) / sdS[LIVE_S], 0))[:5]],
                                     top_units_M2=[[int(LIVE_S[i]), float(np.median(np.abs(dS2[:, LIVE_S[i]]) / sdS[LIVE_S[i]]))] for i in np.argsort(-np.median(np.abs(dS2[:, LIVE_S]) / sdS[LIVE_S], 0))[:5]])
    log('S steps', R['product']['S_step_sd'])
    # tail function check
    tail_fn = make_tail(net); shape = None
    for shp in ((64,), (1, 64)):
        try:
            T_ = tail_fn(S0p, shp)
            Rb = net.run(xp, keep=('h4', 'h5', 'z', 'logits'), batch=4000)
            dev = max(float(np.abs(T_[k] - Rb[k]).max()) for k in ('h4', 'h5', 'z', 'logits')); R['product']['tail_identity_max'] = dev
            if dev < 1e-4: shape = shp; break
        except Exception as ex:  # noqa: BLE001
            log('tail shape', shp, 'failed', repr(ex)[:200])
    log('tail fn shape', shape, 'identity', R['product'].get('tail_identity_max'))

    def featS(S):
        T_ = tail_fn(S, shape); T_['S'] = S; out = []
        for st_ in F.stages: out.append((T_['logits'][:, W_] - T_['logits'][:, Z_])[:, None] if st_ == 'margin' else T_[st_][:, F.live[st_]])
        return np.concatenate(out, 1)

    def pair_slope(A_, B_):
        hA = n0 * sdS[A_]; hB = n2 * sdS[B_]; V = {}
        ok = (S0p[:, A_] - hA >= 0) & (S0p[:, B_] - hB >= 0)
        for sa in (1, -1):
            for sb in (1, -1):
                S_ = S0p.copy(); S_[:, A_] += sa * hA; S_[:, B_] += sb * hB; V[(sa, sb)] = featS(S_)
        sl_ = mixed_slope(V[(1, 1)], V[(1, -1)], V[(-1, 1)], V[(-1, -1)]); sl_[~ok] = np.nan
        Vok = {k: v[ok] for k, v in V.items()}
        return np.nanmedian(sl_, 0), int(ok.sum()), pooled_slope(Vok)
    pu = {}
    if shape is not None:
        with np.errstate(all='ignore'):
            net05, n05, net05p = pair_slope(0, 50)
            rngp = np.random.default_rng(a.seed + 17); nullm = []; null = []; npj = []
            for i in range(a.npair):
                A_, B_ = (int(v) for v in rngp.choice(LIVE_S, 2, replace=False)); v_, n_, vp_ = pair_slope(A_, B_); nullm.append(v_); null.append(vp_); npj.append(n_)
                if i % 50 == 0: log(f'pair {i} ({A_},{B_}) jets {n_} margin median {v_[-1]:.3f} pooled {vp_[-1]:.3f}')
        null = np.array(null); nullm = np.array(nullm)
        # primary statistic: jet-pooled slope (network on all product jets, jet bootstrap); per-jet median kept as secondary
        medm = med; med = pooled_slope(Up); WBp = np.random.default_rng(a.seed + 14).poisson(1.0, (300, len(xp))).astype(np.float64)
        bsp = np.array([pooled_slope(Up, w) for w in WBp]); lo, hi = np.nanpercentile(bsp, 2.5, 0), np.nanpercentile(bsp, 97.5, 0)
        zm = (medm - np.nanmean(nullm, 0)) / np.nanstd(nullm, 0)
        R['product']['pair_S0_S50'] = dict(n_jets=n05, slope={u: float(v) for u, v in zip(un, net05)}, pooled={u: float(v) for u, v in zip(un, net05p)})
        R['product']['null_jets_per_pair_p50'] = float(np.median(npj))
        zsc = (med - np.nanmean(null, 0)) / np.nanstd(null, 0)
        pz = np.array([pnorm(v) if np.isfinite(v) else 1.0 for v in zsc])
        rk = np.array([min(1.0, 2 * min((1 + np.sum(null[:, k] <= med[k])) / (1 + np.isfinite(null[:, k]).sum()),
                                        (1 + np.sum(null[:, k] >= med[k])) / (1 + np.isfinite(null[:, k]).sum()))) for k in range(F.K)])
        qbh = bh(pz); ph = holm(pz); qbr = bh(rk)
        for k, u in enumerate(un):
            pu[u] = dict(slope=float(med[k]), ci=[float(lo[k]), float(hi[k])], frac_valid=float(fv[k]), slope_alt=float(med2[k]), rho=float(rmed[k]),
                         median_slope=float(medm[k]), median_null_q025_q50_q975=[float(np.nanpercentile(nullm[:, k], q)) for q in (2.5, 50, 97.5)],
                         median_z=float(zm[k]), median_p_norm=pnorm(zm[k]), S0S50_pooled=float(net05p[k]),
                         null_q025_q50_q975=[float(np.nanpercentile(null[:, k], q)) for q in (2.5, 50, 97.5)], z=float(zsc[k]),
                         p_norm=float(pz[k]), q_bh=float(qbh[k]), p_holm=float(ph[k]), p_rank=float(rk[k]), q_bh_rank=float(qbr[k]),
                         S0S50_patch=float(net05[k]))
        sig = [u for u in un if pu[u]['q_bh'] < 0.05]
        R['product']['units_bh_sig'] = sig; R['product']['units_bh_sig_slope_near1'] = [u for u in sig if abs(pu[u]['slope'] - 1) < 0.3]
        R['product']['stage_summary'] = {st_: dict(n=sl.stop - sl.start, median_slope=float(np.nanmedian(med[sl])),
                                                   n_bh_sig=int(sum(pu[u]['q_bh'] < 0.05 for u in un[sl])),
                                                   n_above_null=int(sum(pu[u]['q_bh'] < 0.05 and pu[u]['z'] > 0 for u in un[sl])),
                                                   null_median_of_medians=float(np.nanmedian(np.nanmedian(null[:, sl], 0))))
                                         for st_, sl in F.sl.items()}
    R['product']['units'] = pu
    log('product: ' + ' | '.join(f'{st_}: {v}' for st_, v in R['product'].get('stage_summary', {}).items()))
    log('product margin / z units: ' + ' '.join(f'{u} {pu[u]["slope"]:.2f} [{pu[u]["ci"][0]:.2f},{pu[u]["ci"][1]:.2f}] null {np.round(pu[u]["null_q025_q50_q975"], 2)} q {pu[u]["q_bh"]:.3g}'
                                                for u in un if u.startswith('z') or u == 'margin') if pu else 'no null')
    save()

    # ---------------- untrained redraws (>= 2 live S units on the product set)
    R['untrained'] = []
    for sd_ in range(0, 80):
        if len(R['untrained']) >= 3: break
        un_ = Jedi(64, untrained=True, seed=sd_); Su = un_.run(xp, keep=('S',), batch=4000)['S']; nl = int((Su.std(0) > 1e-6).sum())
        if nl < 2: continue
        Fu, Ru = jedi_feat(un_, xp, f'untrained{sd_}')
        if any(len(Fu.live[s_]) == 0 for s_ in ('S', 'h4', 'h5', 'z')):
            log(f'untrained seed {sd_}: a stage has no live unit {[(s_, len(v)) for s_, v in Fu.live.items()]}; skipped'); continue
        sdu = Fu(xp).std(0); sdu[sdu < 1e-9] = 1.0
        Gu = gain_arm(Fu, xt, Mt, sdu, rot=False, seed=sd_); gu = summarize(Gu, Fu, WBt)
        su, _, rhou, Uu = product_net(Fu, product_edits(xp), xp); mu_ = np.nanmedian(su, 0); pu_ = pooled_slope(Uu)
        R['untrained'].append(dict(seed=sd_, n_live=dict({k: len(v) for k, v in Fu.live.items()}), acc=float((Ru['logits'].argmax(1) == y[prod]).mean()),
                                   gains={k: v['all'] for k, v in gu.items()},
                                   product={st_: dict(median_slope=float(np.nanmedian(mu_[sl])), frac_valid=float(np.isfinite(su[:, sl]).mean()),
                                                      pooled_slope_median=float(np.nanmedian(pu_[sl])), pooled_slope_units=pu_[sl]) for st_, sl in Fu.sl.items()}))
        log(f'untrained seed {sd_}: live {R["untrained"][-1]["n_live"]}; gains ' + ' | '.join(f'{k} a1 {v["a1"]:.3f} a2 {v["a2"]:.3f} r {v["ratio"]:.2f}' for k, v in R['untrained'][-1]['gains'].items())
            + '; product ' + str(R['untrained'][-1]['product']))
        save()

    # ---------------- reference networks (gain pattern)
    R['reference'] = {}
    try:
        import interp.e24_azimuth_prong as E24
        for nm in ('dsets_s0', 'x12:hgq-r150_s0'):
            run, accref, kind, obj = E24.get_model(nm)
            if kind == 'dsets':
                rr = lambda xx: (lambda o: dict(S=o['L'], logits=o['logits']))(obj.run(xx)); stages = ('S', 'margin')
            else:
                rr = lambda xx: obj.run(xx, keep=('S', 'z', 'logits')); stages = ('S', 'z', 'margin')
            Rf = rr(x); live = {s_: [int(c_) for c_ in np.flatnonzero(Rf[s_].std(0) > 1e-9)] for s_ in stages if s_ != 'margin'}
            Fr = Feat(nm, rr, stages, live); sdr = Fr(x).std(0); sdr[sdr < 1e-9] = 1.0
            Gf = gain_arm(Fr, xt, Mt, sdr, rot=False, seed=a.seed); gf = summarize(Gf, Fr, WBt)
            R['reference'][nm] = dict(acc_sample=float((Rf['logits'].argmax(1) == y).mean()), acc_ref=accref, n_live={k: len(v) for k, v in live.items()},
                                      gains={k: v['all'] for k, v in gf.items()})
            log(f'reference {nm}: acc {R["reference"][nm]["acc_sample"]:.4f}; ' + ' | '.join(f'{k} a1 {v["a1"]:.3f} {np.round(v["a1_ci"], 2)} a2 {v["a2"]:.3f} {np.round(v["a2_ci"], 2)} r {v["ratio"]:.2f} {np.round(v["ratio_ci"], 2)}'
                                                                                         for k, v in R['reference'][nm]['gains'].items()))
            save()
    except Exception as ex:  # noqa: BLE001
        R['reference_error'] = repr(ex); log('reference error', repr(ex))
    R['wall_s'] = time.time() - T0; save(); log('done ->', JS)


if __name__ == '__main__':
    main()
