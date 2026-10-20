"""E23x (claims A4n, A4d, A3c): which centre carries JEDI-linear's direction-dependent (odd) response to a rigid
translation, and does that response pass through the context-only objects S31 (count dipole) and S58 (count quadrupole)?

Model: official JEDI-linear 3-feature n64 (interp.jedi_core.Jedi(64)), exact forward.  Samples: test = sample(1500, 2311)
(arms 1-4, 6), fit = sample(1500, 2310) minus rows shared with the test sample (fit side of arm 5).  Unedited accuracy is
checked against 0.806-0.813.
Units (standardised by their sd on the unedited test jets): 19 live S (E1 list), live h4, live h5, all 16 z (pre-ReLU),
W - Z logit margin.  Freezes of S units act at S and are applied by an exact jitted copy of the d4..d7 head (checked: max
|head(S) - full forward| on logits = 0); the g freeze uses run_jetwise (g held at the unedited jet's g).

Edit (arms 1-4): rigid translation c of every real particle, |c| in RINGS = {0.03, 0.06} rad at 8 azimuths.  For each mirror
pair (c, -c): odd O = (f(c) - f(-c)) / 2, even E = (f(c) + f(-c)) / 2 - f(0).  Odd power of a stage = sum over pairs, jets,
units of O^2.  Share removed by a freeze = 1 - P(O_frozen) / P(O); path share = P(O - O_frozen) / P(O).
  arm 1: S31 and S58 frozen at the unedited values;  arm 2: all 64 g units frozen (context round held);
  arm 3: S31 only / S58 only;
  arm 4 (control): every pair of the 17 other live S units frozen (136 pairs; the plan's 200 draws exceed the number of
     distinct pairs, so all pairs are enumerated); matched subset = the 20 pairs nearest to S31 + S58 in summed raw sd and
     summed mean |odd response| (distance |log sd ratio| + |log response ratio|; no pair is within x0.5-2 of both).  Singles
     (17) for arm 3; matched = the 5 nearest to S31 or to S58 the same way.
Calls: % of (jet, cell) calls changed vs the unedited call, W -> Z fraction (true W jets called Z) at |c| = 0.06, frozen vs
  unfrozen McNemar on (jet, cell) pairs plus a jet-bootstrap CI of the difference.
Even check (iii): 1 - P(E_frozen) / P(E) per freeze.
Arm 5, centre-moving edits (subsets of N5 random jets of the fit and of the test sample; same 16 cells):
  5a 'soft': move only real particles with pT < 11 GeV and pT rank >= 14 by c; dilate all other real particles about the
     stored axis by k so that M2 = sum pT r^2 is unchanged exactly (k^2 in [0.25, 4] at all 16 cells, else the jet is dropped);
  5b 'hard': move only the 14 hardest by c; dilate the rest (ranks >= 14) to hold M2 the same way;
  rigid: the arm-1 translation.  Measured per cell: change of the count centroid sum_i dpos_i / N, of the pT centroid
  sum_i pT_i dpos_i / M0 (mrad), relative M2 change.  Odd parts of both centre changes are the regressors of the odd response
  (per unit OLS, no intercept, fitted on the fit jets over the three edits): count model, pT model, both; held-out R^2 on the
  test jets (stage pooled, 2,000-jet bootstrap); per-10-mrad gains of each centre in the joint model (fit-jet bootstrap).
  5a vs 5b: RMS odd margin / stage response at |c| = 0.06 on jets feasible for both; S31+S58 freeze shares under 5a and 5b.
Arm 6: rigid 180 and 90 deg rotations about the stored axis; RMS response per stage; share removed by S58 / S31 / both / g
  frozen; single-unit null (17 other live S units).
Arm 7: untrained redraws (seeds 0-2, N7 test jets): S31/S58 live?, share removed by the S31+S58 freeze vs random pairs.
H0: S31+S58 freeze removes no more odd power than the random-pair band; H1: removes > 50 % at h4, h5, z and the margin.
Holm over arm x stage tests (rank p against the all-pair band; normal p from the band mean / sd and the bootstrap sd).
  ./job.sh e23x-run 'python -m interp.e23x_centre_jedi'            -> $AI_RESULTS/jedi/e23x_centre.json
"""
import argparse, itertools, json, math, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp.e23w_gains_jedi import LIVE_S, holm, ci, clean, shift  # noqa: E402

OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
RINGS = (0.03, 0.06); NAZ = 8; SOFT = 11.0; HARD = 14; W_, Z_ = 2, 3; NB = 2000
STAGES = ('S', 'h4', 'h5', 'z', 'margin'); DOWN = ('h4', 'h5', 'z', 'margin')
T0 = time.time()


def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def pnorm(z): return float(math.erfc(abs(z) / math.sqrt(2))) if np.isfinite(z) else None


def mcnemar(a, b):
    a = np.asarray(a, bool).ravel(); b = np.asarray(b, bool).ravel(); n10 = int((a & ~b).sum()); n01 = int((~a & b).sum()); n = n10 + n01
    if n == 0: return dict(n10=0, n01=0, p=1.0)
    try:
        from scipy.stats import binomtest
        p = float(binomtest(min(n10, n01), n, 0.5).pvalue)
    except Exception:  # noqa: BLE001
        p = pnorm((n10 - n01) / math.sqrt(n))
    return dict(n10=n10, n01=n01, p=p)


# ------------------------------------------------------------------------------------------------ exact head (S -> logits)
class Head:
    def __init__(self, net, B=8192):
        import jax, jax.numpy as jnp
        L = net.L; KB = {}
        for k in ('d4', 'd5', 'd6', 'd7'):
            K, b = L[k].get_fused_qkernel_and_qbias(False, L[k].moving_mean, L[k].moving_variance); KB[k] = (jnp.asarray(K), jnp.asarray(b))
        kd = bool(getattr(L['qs1'], 'keepdims', False)); self.kd = kd

        def dense(k, x, act=True):
            z = jnp.einsum(L[k].equation, L[k].iq(x, training=False), KB[k][0]) + KB[k][1]
            return L[k].activation(z) if act and L[k].activation is not None else z

        def f(S):
            J = S.shape[0]; S_ = S.reshape(J, 1, -1) if kd else S
            h4 = dense('d4', S_); h5 = dense('d5', h4); z = dense('d6', h5, act=False); lo = dense('d7', jax.nn.relu(z))
            return h4.reshape(J, -1), h5.reshape(J, -1), z.reshape(J, -1), lo.reshape(J, -1)
        self._f = jax.jit(f); self.B = B

    def __call__(self, S):
        S = np.asarray(S, np.float32); N = len(S); out = [[], [], [], []]
        for i in range(0, N, self.B):
            s = S[i:i + self.B]; n = len(s)
            if n < self.B: s = np.concatenate([s, np.zeros((self.B - n, S.shape[1]), np.float32)])
            r = self._f(s)
            for o, v in zip(out, r): o.append(np.asarray(v[:n], np.float32))
        return [np.concatenate(o) for o in out]


class Units:
    """S (N, 64) -> unit matrix (N, K) [live S, live h4, live h5, z, margin] and calls"""
    def __init__(self, head, liveS, l4, l5):
        self.head, self.liveS, self.l4, self.l5 = head, list(liveS), list(l4), list(l5)
        self.sl = {}; k = 0
        for st, n in (('S', len(liveS)), ('h4', len(l4)), ('h5', len(l5)), ('z', 16), ('margin', 1)): self.sl[st] = slice(k, k + n); k += n
        self.K = k

    def __call__(self, S):
        sh = S.shape; S2 = S.reshape(-1, 64); h4, h5, z, lo = self.head(S2)
        U = np.concatenate([S2[:, self.liveS], h4[:, self.l4], h5[:, self.l5], z, (lo[:, W_] - lo[:, Z_])[:, None]], 1).astype(np.float32)
        return U.reshape(sh[:-1] + (self.K,)), lo.argmax(1).reshape(sh[:-1])


def freeze(Sc, S0, units):
    if not units: return Sc
    Sf = Sc.copy(); Sf[..., units] = S0[..., units]          # S0 (J, 64) broadcasts over the cell axis of Sc (C, J, 64)
    return Sf


# ------------------------------------------------------------------------------------------------ edits
def cells():
    return [(r, k, r * np.cos(2 * np.pi * k / NAZ), r * np.sin(2 * np.pi * k / NAZ)) for r in RINGS for k in range(NAZ)]


C = cells(); PAIRS = [(i, i + NAZ // 2) for i, (r, k, _, _) in enumerate(C) if k < NAZ // 2]
PRING = np.array([RINGS.index(C[i][0]) for i, _ in PAIRS])


def rotate_fixed(x, kind):
    rl = x[..., 0] > 0; xe = x.copy(); e, f = x[..., 1], x[..., 2]
    if kind == 'rot180': ne, nf = -e, -f
    else: ne, nf = -f, e
    xe[..., 1] = np.where(rl, ne, e); xe[..., 2] = np.where(rl, nf, f)
    return xe


def centre_edit(x, kind, ce, cp):
    """kind 'soft' / 'hard': move a subset by c, dilate the rest of the real particles to hold M2 -> (x', feasible)"""
    x64 = x.astype(np.float64); rl = x64[..., 0] > 0; pt = np.where(rl, x64[..., 0], 0); e = np.where(rl, x64[..., 1], 0); p = np.where(rl, x64[..., 2], 0)
    rank = np.argsort(np.argsort(-pt, 1), 1)
    mv = rl & (pt < SOFT) & (rank >= HARD) if kind == 'soft' else rl & (rank < HARD)
    cm = rl & ~mv
    e1 = e + ce * mv; p1 = p + cp * mv
    M2 = (pt * (e * e + p * p)).sum(1); M2m = np.where(mv, pt * (e1 * e1 + p1 * p1), 0).sum(1); M2c = np.where(cm, pt * (e * e + p * p), 0).sum(1)
    k2 = (M2 - M2m) / np.maximum(M2c, 1e-30); ok = (M2c > 0) & (k2 >= 0.25) & (k2 <= 4) & (mv.sum(1) > 0)
    k = np.sqrt(np.clip(k2, 0.25, 4))[:, None]
    e1 = np.where(cm, e * k, e1); p1 = np.where(cm, p * k, p1)
    xe = x.copy(); xe[..., 1] = np.where(rl, e1, x[..., 1]).astype(np.float32); xe[..., 2] = np.where(rl, p1, x[..., 2]).astype(np.float32)
    return xe, ok


def measure(x, xe):
    """count-centroid and pT-centroid change (J, 2) in rad, relative M2 change (J,)"""
    a = x.astype(np.float64); b = xe.astype(np.float64); rl = a[..., 0] > 0; pt = np.where(rl, a[..., 0], 0)
    d = np.where(rl[..., None], b[..., 1:3] - a[..., 1:3], 0); n = rl.sum(1)[:, None]; M0 = pt.sum(1)[:, None]
    r2a = np.where(rl, a[..., 1] ** 2 + a[..., 2] ** 2, 0); r2b = np.where(rl, b[..., 1] ** 2 + b[..., 2] ** 2, 0)
    M2a = (pt * r2a).sum(1); M2b = (pt * r2b).sum(1); Na = r2a.sum(1); Nb = r2b.sum(1)
    # q_count = 1/2 dlog sum_i r_i^2 (unweighted second moment), q_pt = 1/2 dlog M2 (pT-weighted); their odd parts are the
    # E23wv centre predictors c . sum_i pos_i / sum_i r_i^2 (count) and c . M1 / M2 (pT) to first order
    return d.sum(1) / n, (pt[..., None] * d).sum(1) / M0, M2b / M2a - 1, 0.5 * np.log(Nb / Na), 0.5 * np.log(M2b / M2a)


# ------------------------------------------------------------------------------------------------ odd / even and shares
def odd_even(Uc, U0, sd):
    """Uc (C, J, K), U0 (J, K) -> O, E (P, J, K) standardised"""
    O = np.stack([0.5 * (Uc[i] - Uc[j]) for i, j in PAIRS]) / sd
    E = np.stack([0.5 * (Uc[i] + Uc[j]) - U0 for i, j in PAIRS]) / sd
    return O.astype(np.float32), E.astype(np.float32)


def perjet(A, sl, sel=None):
    """A (P, J, K) -> {stage: (J,) sum over selected pairs and stage units of A^2}"""
    if sel is not None: A = A[sel]
    A2 = (A * A).sum(0, dtype=np.float64)
    return {st: A2[:, s].sum(1) for st, s in sl.items()}


def share(num_j, den_j, WB):
    """1 - sum num / sum den with jet bootstrap"""
    pt = 1 - num_j.sum() / den_j.sum(); b = 1 - (WB @ num_j) / (WB @ den_j)
    return dict(share=float(pt), ci=ci(b), sd=float(np.std(b)), p_le_half=float((b <= 0.5).mean())), b


def freeze_summary(Of, Ef, O, E, sl, WB, sels):
    out = {}
    for nm, sel in sels.items():
        a, b, c_ = perjet(Of, sl, sel), perjet(O, sl, sel), perjet(O - Of, sl, sel); ea, eb = perjet(Ef, sl, sel), perjet(E, sl, sel)
        out[nm] = {}
        for st in sl:
            s_, _ = share(a[st], b[st], WB); pth = float(c_[st].sum() / b[st].sum()); ev, _ = share(ea[st], eb[st], WB)
            out[nm][st] = dict(odd_removed=s_, odd_path_share=pth, even_removed=ev, odd_rms=float(np.sqrt(b[st].sum() / (b[st].size * (sl[st].stop - sl[st].start) * (len(sel) if sel is not None else O.shape[0])))))
    return out


def share_point(Of, O, sl, sel=None):
    a, b = perjet(Of, sl, sel), perjet(O, sl, sel)
    return {st: float(1 - a[st].sum() / b[st].sum()) for st in sl}


def calls_summary(call_c, call_f, call0, y, sel_cells):
    """call_c, call_f (C, J) unfrozen / frozen calls; sel_cells: cell indices (|c| = 0.06)"""
    cc = call_c[sel_cells] != call0[None]; cf = call_f[sel_cells] != call0[None]; W = y == W_
    wz = lambda c_: float((c_[sel_cells][:, W] == Z_).mean())
    d_j = (cc.mean(0) - cf.mean(0)); rng = np.random.default_rng(7); bb = np.array([d_j[rng.integers(0, len(d_j), len(d_j))].mean() for _ in range(NB)])
    return dict(changed_unfrozen=float(cc.mean()), changed_frozen=float(cf.mean()), diff=float(d_j.mean()), diff_ci=ci(bb),
                W_to_Z_unfrozen=wz(call_c), W_to_Z_frozen=wz(call_f), W_to_Z_unedited=float((call0[W] == Z_).mean()), mcnemar=mcnemar(cc, cf))


# ------------------------------------------------------------------------------------------------ regression (arm 5)
def ols_stats(X, Y):
    """X (P, J, a), Y (P, J, K) -> per-jet XtX (J, a, a), XtY (J, K, a)"""
    X = X.astype(np.float64); Y = Y.astype(np.float64)
    return np.einsum('pja,pjb->jab', X, X), np.einsum('pja,pju->jua', X, Y)


def solve(XtX, XtY, w=None):
    if w is None:
        A = XtX.sum(0); B = XtY.sum(0); return np.linalg.solve(A + 1e-12 * np.eye(len(A)), B.T).T          # (K, a)
    A = np.einsum('nj,jab->nab', w, XtX); J, K, a = XtY.shape; B = (w @ XtY.reshape(J, K * a)).reshape(len(w), K, a)
    return np.transpose(np.linalg.solve(A + 1e-12 * np.eye(a), np.transpose(B, (0, 2, 1))), (0, 2, 1))     # (n, K, a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per', type=int, default=1500); ap.add_argument('--seed_fit', type=int, default=2310); ap.add_argument('--seed_test', type=int, default=2311)
    ap.add_argument('--n5', type=int, default=3000); ap.add_argument('--n7', type=int, default=1500); ap.add_argument('--ntest', type=int, default=0)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    from interp.jedi_core import Jedi, sample
    from interp.e23_tail_jedi import sample_idx
    OUT.mkdir(parents=True, exist_ok=True); JS = OUT / f'e23x_centre{"_" + a.tag if a.tag else ""}.json'
    R = dict(node='E23x', plan='module docstring', rings=RINGS, naz=NAZ, nb=NB, soft_gev=SOFT, hard=HARD); save = lambda: JS.write_text(json.dumps(clean(R)))

    # ---------------- samples
    xt, yt, _ = sample(a.per, a.seed_test); xt = np.ascontiguousarray(xt, np.float32)
    xf, yf, _ = sample(a.per, a.seed_fit); xf = np.ascontiguousarray(xf, np.float32)
    keep = ~np.isin(sample_idx(a.per, a.seed_fit), sample_idx(a.per, a.seed_test)); xf, yf = xf[keep], yf[keep]
    R['fit_overlap_removed'] = int((~keep).sum())
    if a.ntest:
        r_ = np.random.default_rng(1); s_ = np.sort(r_.choice(len(xt), a.ntest, replace=False)); xt, yt = xt[s_], yt[s_]
    J = len(xt); R['N_test'] = J; R['N_fit'] = len(xf)
    net = Jedi(64); head = Head(net)
    B0 = net.run(xt, keep=('g', 'S', 'logits'), batch=4000); S0, g0, call0 = B0['S'].astype(np.float32), B0['g'], B0['logits'].argmax(1)
    R['acc_test'] = float((call0 == yt).mean()); R['acc_fit'] = float((net.run(xf, keep=('logits',))['logits'].argmax(1) == yf).mean())
    h4, h5, z, lo = head(S0); R['head_exact_max_abs_logit'] = float(np.abs(lo - B0['logits']).max()); R['head_call_agree'] = float((lo.argmax(1) == call0).mean())
    l4 = [int(c) for c in np.flatnonzero(h4.std(0) > 1e-9)]; l5 = [int(c) for c in np.flatnonzero(h5.std(0) > 1e-9)]
    UN = Units(head, LIVE_S, l4, l5); U0, _ = UN(S0); sd = U0.std(0); sd[sd < 1e-9] = 1.0; sl = UN.sl
    R['K'] = UN.K; R['n_live'] = dict(S=len(LIVE_S), h4=len(l4), h5=len(l5), z=16)
    log(f'test N {J} acc {R["acc_test"]:.4f}; fit N {len(xf)} (removed {R["fit_overlap_removed"]}) acc {R["acc_fit"]:.4f}; head max |dlogit| {R["head_exact_max_abs_logit"]:.2e}; live {R["n_live"]}')
    save()
    WB = np.random.default_rng(11).poisson(1.0, (NB, J)).astype(np.float32)
    sels = {'pooled': None, 'r0.03': np.flatnonzero(PRING == 0), 'r0.06': np.flatnonzero(PRING == 1)}
    c06 = [i for i, (r, _, _, _) in enumerate(C) if r == 0.06]

    # ---------------- arm 1-4: rigid shift
    Sc = np.zeros((len(C), J, 64), np.float32); Sg = np.zeros_like(Sc)
    for i, (r, k, ce, cp) in enumerate(C):
        xs = shift(xt, ce, cp)
        Sc[i] = net.run(xs, keep=('S',), batch=4000)['S']
        Sg[i] = net.run_jetwise(xs, {'g': (np.ones(64), g0)}, keep=('S',), batch=4000)['S']
    log('rigid shift S done (free and g-held)')
    R['g_freeze_S31_S58_max_abs_change'] = float(np.abs(Sg[..., [31, 58]] - S0[None][..., [31, 58]]).max())
    R['shift_S31_S58_mean_abs_change_over_sd'] = {f'S{u}': float(np.abs(Sc[..., u] - S0[None, :, u]).mean() / S0[:, u].std()) for u in (31, 58)}
    Uc, call_c = UN(Sc); O, E = odd_even(Uc, U0, sd)
    # sanity: head(S) of the shifted jets vs the full forward
    lo_full = net.run(shift(xt[:500], C[-1][2], C[-1][3]), keep=('logits',))['logits'].argmax(1); R['head_vs_full_shift_call_agree'] = float((lo_full == call_c[-1, :500]).mean())
    R['odd_rms_unfrozen'] = {st: float(np.sqrt((O[..., s] ** 2).mean())) for st, s in sl.items()}
    R['even_rms_unfrozen'] = {st: float(np.sqrt((E[..., s] ** 2).mean())) for st, s in sl.items()}
    log('odd rms', R['odd_rms_unfrozen'], 'even rms', R['even_rms_unfrozen'])
    arms = {'S31_S58': [31, 58], 'S31': [31], 'S58': [58]}
    R['freeze'] = {}; R['calls'] = {}; BOOT = {}
    for nm, units in list(arms.items()) + [('g_all', None)]:
        Sf = Sg if units is None else freeze(Sc, S0, units)
        Uf, call_f = UN(Sf); Of, Ef = odd_even(Uf, U0, sd)
        R['freeze'][nm] = freeze_summary(Of, Ef, O, E, sl, WB, sels)
        a_, b_ = perjet(Of, sl), perjet(O, sl); BOOT[nm] = {st: share(a_[st], b_[st], WB)[1] for st in sl}
        R['calls'][nm] = calls_summary(call_c, call_f, call0, yt, c06)
        log(f'freeze {nm}: ' + ' | '.join(f'{st} odd {v["odd_removed"]["share"]:.3f} {np.round(v["odd_removed"]["ci"], 3)} path {v["odd_path_share"]:.3f} even {v["even_removed"]["share"]:+.3f}'
                                          for st, v in R['freeze'][nm]['pooled'].items()) + f' | calls {R["calls"][nm]}')
        save()
    del Sg

    # arm 4: random S-unit pairs / singles
    others = [u for u in LIVE_S if u not in (31, 58)]
    sdraw = S0.std(0); oddabs = np.abs(0.5 * (Sc[[i for i, _ in PAIRS]] - Sc[[j for _, j in PAIRS]])).mean((0, 1))
    R['unit_match'] = {f'S{u}': dict(sd=float(sdraw[u]), mean_abs_odd=float(oddabs[u])) for u in LIVE_S}
    tgt = (sdraw[31] + sdraw[58], oddabs[31] + oddabs[58])
    dist = lambda v_sd, v_r, t_sd, t_r: abs(math.log(v_sd / t_sd)) + abs(math.log(max(v_r, 1e-12) / t_r))
    null_pairs = []; null_single = []
    for u, v in itertools.combinations(others, 2):
        Of, _ = odd_even(UN(freeze(Sc, S0, [u, v]))[0], U0, sd)
        null_pairs.append(dict(units=[u, v], dist=dist(sdraw[u] + sdraw[v], oddabs[u] + oddabs[v], *tgt), sd_ratio=float((sdraw[u] + sdraw[v]) / tgt[0]),
                               resp_ratio=float((oddabs[u] + oddabs[v]) / tgt[1]), share=share_point(Of, O, sl)))
    for u in others:
        Of, _ = odd_even(UN(freeze(Sc, S0, [u]))[0], U0, sd)
        null_single.append(dict(unit=u, d31=dist(sdraw[u], oddabs[u], sdraw[31], oddabs[31]), d58=dist(sdraw[u], oddabs[u], sdraw[58], oddabs[58]),
                                resp_ratio31=float(oddabs[u] / oddabs[31]), resp_ratio58=float(oddabs[u] / oddabs[58]), share=share_point(Of, O, sl)))
    for r_ in null_pairs: r_['matched'] = False
    for i_ in np.argsort([r_['dist'] for r_ in null_pairs])[:20]: null_pairs[i_]['matched'] = True
    for k_ in ('31', '58'):
        for r_ in null_single: r_['matched' + k_] = False
        for i_ in np.argsort([r_['d' + k_] for r_ in null_single])[:5]: null_single[i_]['matched' + k_] = True
    R['null_matched_resp_ratio_range'] = dict(pairs=[float(min(r_['resp_ratio'] for r_ in null_pairs if r_['matched'])), float(max(r_['resp_ratio'] for r_ in null_pairs if r_['matched']))])
    R['null_pairs'] = null_pairs; R['null_single'] = null_single
    log(f'null: {len(null_pairs)} pairs ({sum(p["matched"] for p in null_pairs)} matched), {len(null_single)} singles '
        f'(matched to S31 {sum(s["matched31"] for s in null_single)}, S58 {sum(s["matched58"] for s in null_single)})')

    def band(rows, flag=None):
        rows = [r_ for r_ in rows if flag is None or r_[flag]]
        if not rows: return None
        out = {}
        for st in STAGES:
            v = np.array([r_['share'][st] for r_ in rows]); out[st] = dict(n=len(v), mean=float(v.mean()), sd=float(v.std()), lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)), max=float(v.max()))
        return out
    R['null_band'] = dict(pairs_all=band(null_pairs), pairs_matched=band(null_pairs, 'matched'), single_all=band(null_single),
                          single_matched31=band(null_single, 'matched31'), single_matched58=band(null_single, 'matched58'))
    tests = []
    for nm, rows in (('S31_S58', null_pairs), ('S31', null_single), ('S58', null_single)):
        R['freeze'][nm]['vs_null'] = {}
        for st in DOWN:
            obs = R['freeze'][nm]['pooled'][st]['odd_removed']['share']; v = np.array([r_['share'][st] for r_ in rows])
            p_rank = float((1 + (v >= obs).sum()) / (len(v) + 1)); zz = (obs - v.mean()) / math.sqrt(v.var() + np.var(BOOT[nm][st]) + 1e-30)
            p_z = float(0.5 * math.erfc(zz / math.sqrt(2)))
            R['freeze'][nm]['vs_null'][st] = dict(p_rank=p_rank, z=float(zz), p_normal_one_sided=p_z, null_max=float(v.max()))
            tests.append((f'arm_{nm}_{st}_vs_null', max(p_rank, p_z)))   # conservative: the larger of the two p
    for st in DOWN:
        tests.append((f'arm_S31_S58_{st}_gt_half', R['freeze']['S31_S58']['pooled'][st]['odd_removed']['p_le_half']))
    log('vs null', {nm: R['freeze'][nm]['vs_null'] for nm in arms})
    save()

    # ---------------- arm 6: 180 / 90 deg rotations
    R['rotation'] = {}
    for kind in ('rot180', 'rot90'):
        xr = rotate_fixed(xt, kind); Sr = net.run(xr, keep=('S',), batch=4000)['S'].astype(np.float32)
        Srg = net.run_jetwise(xr, {'g': (np.ones(64), g0)}, keep=('S',), batch=4000)['S'].astype(np.float32)
        Ur, cr = UN(Sr); D = ((Ur - U0) / sd)[None]
        row = dict(rms={st: float(np.sqrt((D[..., s] ** 2).mean())) for st, s in sl.items()},
                   dS_over_sd={f'S{u}': float(np.abs(Sr[:, u] - S0[:, u]).mean() / S0[:, u].std()) for u in (31, 58)},
                   calls_changed=float((cr != call0).mean()))
        for nm, Sf in (('S58', freeze(Sr, S0, [58])), ('S31', freeze(Sr, S0, [31])), ('S31_S58', freeze(Sr, S0, [31, 58])), ('g_all', Srg)):
            Uf, cf = UN(Sf); Df = ((Uf - U0) / sd)[None]
            a_, b_ = perjet(Df, sl), perjet(D, sl)
            row[nm] = {st: share(a_[st], b_[st], WB)[0] for st in sl}
            row[nm]['calls'] = dict(changed=float((cf != call0).mean()), mcnemar=mcnemar(cr != call0, cf != call0))
        nul = []
        for u in others:
            Uf, _ = UN(freeze(Sr, S0, [u])); Df = ((Uf - U0) / sd)[None]; a_, b_ = perjet(Df, sl), perjet(D, sl)
            nul.append(dict(unit=u, share={st: float(1 - a_[st].sum() / b_[st].sum()) for st in sl}))
        row['null_single'] = nul; row['vs_null_S58'] = {}
        for st in DOWN:
            v = np.array([n_['share'][st] for n_ in nul]); obs = row['S58'][st]['share']
            row['vs_null_S58'][st] = dict(p_rank=float((1 + (v >= obs).sum()) / (len(v) + 1)), null_max=float(v.max()), null_mean=float(v.mean()))
            tests.append((f'{kind}_S58_{st}_vs_null', row['vs_null_S58'][st]['p_rank']))
        R['rotation'][kind] = row
        log(kind, json.dumps(clean({k: v for k, v in row.items() if k != 'null_single'}))[:1500]); save()
    ps = [t[1] for t in tests]; R['holm'] = dict(zip([t[0] for t in tests], [float(v) for v in holm(ps)])); R['holm_raw'] = dict(tests)
    save()

    # ---------------- arm 5: centre-moving edits
    rng = np.random.default_rng(23)
    i5f = np.sort(rng.choice(len(xf), min(a.n5, len(xf)), replace=False)); i5t = np.sort(rng.choice(J, min(a.n5, J), replace=False))
    data = {}
    for side, xx, S0s, U0s in (('fit', xf[i5f], None, None), ('test', xt[i5t], S0[i5t], U0[i5t])):
        if S0s is None:
            S0s = net.run(xx, keep=('S',), batch=4000)['S'].astype(np.float32); U0s, _ = UN(S0s)
        for ed in ('rigid', 'soft', 'hard'):
            if ed == 'rigid' and side == 'test':
                Sce = Sc[:, i5t]; okj = np.ones(len(xx), bool); dn = np.zeros((len(C), len(xx), 2)); dp = np.zeros_like(dn); dm = np.zeros((len(C), len(xx))); qn = np.zeros_like(dm); qp = np.zeros_like(dm)
                for i, (r, k, ce, cp) in enumerate(C):
                    dn[i], dp[i], dm[i], qn[i], qp[i] = measure(xx, shift(xx, ce, cp))
            else:
                Sce = np.zeros((len(C), len(xx), 64), np.float32); okj = np.ones(len(xx), bool); dn = np.zeros((len(C), len(xx), 2)); dp = np.zeros_like(dn); dm = np.zeros((len(C), len(xx))); qn = np.zeros_like(dm); qp = np.zeros_like(dm)
                for i, (r, k, ce, cp) in enumerate(C):
                    if ed == 'rigid': xe = shift(xx, ce, cp)
                    else: xe, ok = centre_edit(xx, ed, ce, cp); okj &= ok
                    dn[i], dp[i], dm[i], qn[i], qp[i] = measure(xx, xe); Sce[i] = net.run(xe, keep=('S',), batch=4000)['S']
            Ue, _ = UN(Sce); Oe, _ = odd_even(Ue, U0s, sd)
            Of, _ = odd_even(UN(freeze(Sce, S0s, [31, 58]))[0], U0s, sd)
            odd = lambda v: np.stack([0.5 * (v[i] - v[j]) for i, j in PAIRS])
            # regressors: [q_count (%), dcount_eta, dcount_phi (mrad), q_pt (%), dpt_eta, dpt_phi (mrad)], odd parts
            Xo = np.concatenate([odd(qn)[..., None] * 100, odd(dn) * 1e3, odd(qp)[..., None] * 100, odd(dp) * 1e3], -1)
            Xo[:, ~okj] = 0; Oe[:, ~okj] = 0; Of[:, ~okj] = 0
            c06p = np.flatnonzero(PRING == 1)
            meas = dict(n_feasible=int(okj.sum()), frac_feasible=float(okj.mean()),
                        dcount_over_c_median=float(np.median(np.hypot(Xo[c06p][:, okj][..., 1], Xo[c06p][:, okj][..., 2]) / 60.0)),
                        dpt_over_c_median=float(np.median(np.hypot(Xo[c06p][:, okj][..., 4], Xo[c06p][:, okj][..., 5]) / 60.0)),
                        q_count_odd_rms_pct=float(np.sqrt((Xo[c06p][:, okj][..., 0] ** 2).mean())), q_pt_odd_rms_pct=float(np.sqrt((Xo[c06p][:, okj][..., 3] ** 2).mean())),
                        dM2_rel_max_abs=float(np.abs(dm[:, okj]).max()) if okj.any() else None)
            data[(side, ed)] = dict(X=Xo, O=Oe, Of=Of, ok=okj, meas=meas)
            log(f'arm5 {side} {ed}: {meas}')
    R['arm5_measured'] = {f'{s}_{e}': d['meas'] for (s, e), d in data.items()}
    # 5a vs 5b on test jets feasible for both, |c| = 0.06
    okb = data[('test', 'soft')]['ok'] & data[('test', 'hard')]['ok']; c06p = np.flatnonzero(PRING == 1); R['arm5_both_feasible_test'] = int(okb.sum())
    Wb = np.random.default_rng(31).poisson(1.0, (NB, int(okb.sum()))).astype(np.float32)
    cmp = {}
    for st, s in sl.items():
        pa = (data[('test', 'soft')]['O'][c06p][:, okb][..., s].astype(np.float64) ** 2).sum((0, 2)); pb = (data[('test', 'hard')]['O'][c06p][:, okb][..., s].astype(np.float64) ** 2).sum((0, 2))
        prg = (data[('test', 'rigid')]['O'][c06p][:, okb][..., s].astype(np.float64) ** 2).sum((0, 2))
        rb = np.sqrt((Wb @ pa) / (Wb @ pb))
        cmp[st] = dict(rms_soft=float(np.sqrt(pa.mean() / (len(c06p) * (s.stop - s.start)))), rms_hard=float(np.sqrt(pb.mean() / (len(c06p) * (s.stop - s.start)))),
                       rms_rigid=float(np.sqrt(prg.mean() / (len(c06p) * (s.stop - s.start)))), ratio_soft_over_hard=float(np.sqrt(pa.sum() / pb.sum())), ratio_ci=ci(rb),
                       p_soft_le_hard=float((rb <= 1).mean()))
    R['arm5_soft_vs_hard'] = cmp; log('arm5 soft vs hard (|c| 0.06)', json.dumps(clean(cmp)))
    # S31+S58 freeze under each edit (test)
    R['arm5_freeze_S31_S58'] = {}
    for ed in ('rigid', 'soft', 'hard'):
        d = data[('test', ed)]; okj = d['ok']; Wj = np.random.default_rng(37).poisson(1.0, (NB, int(okj.sum()))).astype(np.float32)
        a_, b_ = perjet(d['Of'][:, okj], sl), perjet(d['O'][:, okj], sl)
        R['arm5_freeze_S31_S58'][ed] = {st: share(a_[st], b_[st], Wj)[0] for st in sl}
    log('arm5 S31+S58 freeze', json.dumps(clean(R['arm5_freeze_S31_S58']))[:1500]); save()
    # regression: odd response on odd centre changes (fit on fit jets, test on test jets)
    models = {'count_moment': [0], 'pt_moment': [3], 'count': [0, 1, 2], 'pt': [3, 4, 5], 'both': [0, 1, 2, 3, 4, 5], 'centroid_shift_only': [1, 2, 4, 5]}
    Xf = np.concatenate([data[('fit', e)]['X'] for e in ('rigid', 'soft', 'hard')]); Yf = np.concatenate([data[('fit', e)]['O'] for e in ('rigid', 'soft', 'hard')])
    XtXf, XtYf = ols_stats(Xf, Yf)
    reg = {'condition_number_fit': float(np.linalg.cond(XtXf.sum(0)))}
    corr = np.corrcoef(Xf.reshape(-1, 6)[np.abs(Xf.reshape(-1, 6)).sum(1) > 0].T); reg['regressor_corr_fit'] = corr
    Wf = np.random.default_rng(41).poisson(1.0, (NB, XtXf.shape[0])).astype(np.float64)
    for ev in ('all', 'rigid', 'soft', 'hard'):
        es = ('rigid', 'soft', 'hard') if ev == 'all' else (ev,)
        Xt_ = np.concatenate([data[('test', e)]['X'] for e in es]); Yt_ = np.concatenate([data[('test', e)]['O'] for e in es]).astype(np.float64)
        Wt = np.random.default_rng(43).poisson(1.0, (NB, Xt_.shape[1])).astype(np.float64)
        reg[ev] = {}
        for mn, cols in models.items():
            beta = solve(XtXf[:, cols][:, :, cols], XtYf[..., cols])                                              # (K, a)
            res = Yt_ - np.einsum('pja,ua->pju', Xt_[..., cols], beta)
            reg[ev][mn] = {}
            for st, s in sl.items():
                rj = (res[..., s] ** 2).sum((0, 2)); tj = (Yt_[..., s] ** 2).sum((0, 2))
                reg[ev][mn][st] = dict(r2=float(1 - rj.sum() / tj.sum()), ci=ci(1 - (Wt @ rj) / (Wt @ tj)))
        if ev in ('all',):
            # paired difference count - pt held-out R^2
            cc_, cp_ = models['count'], models['pt']
            bc = solve(XtXf[:, cc_][:, :, cc_], XtYf[..., cc_]); bp = solve(XtXf[:, cp_][:, :, cp_], XtYf[..., cp_])
            rc = Yt_ - np.einsum('pja,ua->pju', Xt_[..., cc_], bc); rp = Yt_ - np.einsum('pja,ua->pju', Xt_[..., cp_], bp)
            reg['count_minus_pt'] = {}
            for st, s in sl.items():
                tj = (Yt_[..., s] ** 2).sum((0, 2)); cj = (rc[..., s] ** 2).sum((0, 2)); pj = (rp[..., s] ** 2).sum((0, 2))
                bb = ((Wt @ pj) - (Wt @ cj)) / (Wt @ tj); obs = float((pj.sum() - cj.sum()) / tj.sum())
                reg['count_minus_pt'][st] = dict(diff=obs, ci=ci(bb), p_two_sided=float(min(1, 2 * min((bb <= 0).mean(), (bb >= 0).mean()))))
    # joint-model gains (stage RMS coefficient): per 1 % odd q (moments) and per 10 mrad (centroid shifts), fit-jet bootstrap
    bj = solve(XtXf, XtYf); bjb = solve(XtXf, XtYf, Wf)
    gains = {}
    for st, s in sl.items():
        gm = lambda b, c: np.sqrt((b[..., s, c] ** 2).mean(-1)); gv = lambda b, c: np.sqrt((b[..., s, c] ** 2 + b[..., s, c + 1] ** 2).mean(-1)) * 10
        gains[st] = dict(q_count_per1pct=float(gm(bj, 0)), q_count_ci=ci(gm(bjb, 0)), q_pt_per1pct=float(gm(bj, 3)), q_pt_ci=ci(gm(bjb, 3)),
                         q_ratio=float(gm(bj, 0) / gm(bj, 3)), q_ratio_ci=ci(gm(bjb, 0) / gm(bjb, 3)),
                         dcount_per10mrad=float(gv(bj, 1)), dcount_ci=ci(gv(bjb, 1)), dpt_per10mrad=float(gv(bj, 4)), dpt_ci=ci(gv(bjb, 4)))
    m_ = sl['margin'].start
    gains['margin_coef'] = dict(coef=bj[m_], ci=[ci(bjb[:, m_, i]) for i in range(6)], names=['q_count_pct', 'dcount_eta_mrad', 'dcount_phi_mrad', 'q_pt_pct', 'dpt_eta_mrad', 'dpt_phi_mrad'])
    reg['gains'] = gains
    # count-moment slope per edit (O on [q_count, dcount] per edit, test jets), stage gain relative to the rigid edit
    Bq = {}
    for e in ('rigid', 'soft', 'hard'):
        Xe_, Ye_ = data[('test', e)]['X'][..., [0, 1, 2]], data[('test', e)]['O']; XtX_, XtY_ = ols_stats(Xe_, Ye_)
        Wq = np.random.default_rng(59).poisson(1.0, (NB, XtX_.shape[0])).astype(np.float64)       # same jets, same weights across edits
        Bq[e] = (solve(XtX_, XtY_), solve(XtX_, XtY_, Wq))
    reg['count_slope_vs_rigid'] = {}
    for e in ('soft', 'hard'):
        reg['count_slope_vs_rigid'][e] = {}
        for st, s in sl.items():
            g_ = lambda be, br: (be[..., s, 0] * br[..., s, 0]).sum(-1) / (br[..., s, 0] ** 2).sum(-1)
            reg['count_slope_vs_rigid'][e][st] = dict(gain=float(g_(Bq[e][0], Bq['rigid'][0])), ci=ci(g_(Bq[e][1], Bq['rigid'][1])))
    R['arm5_regression'] = reg
    log('arm5 regression', json.dumps(clean({k: reg[k] for k in ('all', 'rigid', 'soft', 'hard', 'count_minus_pt', 'gains', 'count_slope_vs_rigid', 'condition_number_fit')}))[:8000]); save()
    del data

    # ---------------- arm 7: untrained redraws
    R['untrained'] = []
    xu = xt[np.sort(np.random.default_rng(47).choice(J, min(a.n7, J), replace=False))]
    for s_ in (0, 1, 2):
        un = Jedi(64, untrained=True, seed=s_); hu = Head(un)
        Bu = un.run(xu, keep=('S',), batch=4000); S0u = Bu['S'].astype(np.float32); h4u, h5u, _, _ = hu(S0u)
        liveu = [int(c) for c in np.flatnonzero(S0u.std(0) > 1e-9)]
        row = dict(seed=s_, live_S=liveu, S31_live=31 in liveu, S58_live=58 in liveu, S31_sd=float(S0u[:, 31].std()), S58_sd=float(S0u[:, 58].std()))
        if len(liveu) >= 2:
            Uu = Units(hu, liveu, [int(c) for c in np.flatnonzero(h4u.std(0) > 1e-9)], [int(c) for c in np.flatnonzero(h5u.std(0) > 1e-9)])
            U0u, _ = Uu(S0u); sdu = U0u.std(0); sdu[sdu < 1e-9] = 1.0
            Scu = np.stack([un.run(shift(xu, ce, cp), keep=('S',), batch=4000)['S'] for (_, _, ce, cp) in C]).astype(np.float32)
            Ou, _ = odd_even(Uu(Scu)[0], U0u, sdu)
            row['S31_S58_shift_mean_abs'] = {f'S{u}': float(np.abs(Scu[..., u] - S0u[None, :, u]).mean()) for u in (31, 58)}
            row['freeze_S31_S58'] = share_point(odd_even(Uu(freeze(Scu, S0u, [31, 58]))[0], U0u, sdu)[0], Ou, Uu.sl)
            oth = [u for u in liveu if u not in (31, 58)]; prs = list(itertools.combinations(oth, 2)); rr = np.random.default_rng(53)
            if len(prs) > 60: prs = [prs[i] for i in rr.choice(len(prs), 60, replace=False)]
            nv = [share_point(odd_even(Uu(freeze(Scu, S0u, list(p_)))[0], U0u, sdu)[0], Ou, Uu.sl) for p_ in prs]
            row['null_pairs_band'] = {st: dict(n=len(nv), lo=float(np.percentile([v[st] for v in nv], 2.5)), hi=float(np.percentile([v[st] for v in nv], 97.5))) for st in STAGES} if nv else None
        R['untrained'].append(row); log('untrained', json.dumps(clean(row))[:800]); save()
    R['wall_s'] = time.time() - T0; save(); log('done ->', JS)


if __name__ == '__main__':
    main()
