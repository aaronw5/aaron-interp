"""E39x (claims A1s, A1x; loop A round 19, stage 3, exploratory method node; E39/E39v follow-up): how many dimensions does the
content shared between retrained JEDI-linear networks need, does it sit in the four named S roles, and can edit inheritance be
stated at network level?

Networks (no training of networks): the 10 trained JEDI-linear networks of E33x/E39 (official 3f n64 / n32 / n128, perm-inv n64,
16f n64, X12 HGQ-r150 s0/s1/s2, X12 float s0/s1) through interp.e33_program_recurrence_jedi.Model, plus one untrained redraw
(jetdistill.network.redraw_untrained, seed 0) of each as donor control.  hgq150_s1 (57 %) is reported, left out of pooled numbers.

PLAN (fixed before the first run)
Samples: fit = sample(2000, 4941) (10,000 jets; maps and task-loss training); test = sample(2000, 4942) minus rows in fit.
Stitch (as E39): A's layers up to L in {S, h4}; a map into B's L; every unit clipped to B's [min, max] on the fit jets; B's exact
  fixed-point tail.  Same-call % = share of test jets where the stitched argmax equals B's own unedited argmax.
Arm 1, reduced-rank regression (RRR): ridge map W (E39 ridge, alpha on a 20 % hold-out); rank-r map W_r = W V_r V_r^T with V_r the
  top-r right singular vectors of the fitted values (L_A - mean) W on the fit jets (the rank-r affine map closest to the ridge
  fit in squared error of the prediction), r in 1, 2, 4, 8, 16, 64 (64 = full).  Controls at each r: random map with W_r's
  singular values (Haar orthogonal factors), untrained donor (redraw of A, its own RRR map), and a reference: logit-only donor
  (ridge map from A's 5 logits into B's L, E39v).
Arm 2, task-loss map: rank-r map (U V, bias) initialised at the RRR map, trained 300 Adam steps (lr 3e-3, batch 2048, fit jets)
  on the cross-entropy of B's frozen tail against the true labels; the gradient goes through a float copy of B's tail (fused
  kernels, no input quantizers; its call agreement with the exact tail is logged); evaluated through B's exact tail.  r in 1, 2,
  4, 8, 64.  Compared with RRR at the same r (same-call, accuracy, test cross-entropy).
Arm 3, subset stitch at S: for pairs where k >= 1 of the four E33x roles (r2 = Sum pT r^2 unit, soft = soft-particle count,
  hard = hard-window pT sum, core = Sum r core) are named with held-out formula R^2 >= 0.9 in both nets: ridge map from A's k role
  units to B's k role units; B's other 63-k... units held at B's fit mean; clip; B's tail.  Control: 200 random subsets of k live
  units in A and k live units in B (live: fit s.d. > 1e-6), same procedure.  Per-pair p = (1 + #random >= named) / 201, Holm over
  pairs.  Reference: B's own role units with the rest at B's mean (no stitch) vs B's own random live subsets.
Arm 4, edit inheritance at network level (8 networks with 3-feature 64-slot input; n32 sees the first 32 slots): edits from
  interp.e39_stitch_retrains_jedi.build_edits: T = Delta eta + 0.06 rad on every real particle, stored axis, true W jets, response
  1[call Z] after - before; D = E35 prong sharing dc +0.3 at fixed four-momentum, response Delta(l_W - l_Z) on W jets (DW) and Z
  jets (DZ); N = one 1 GeV particle at r = 0.3, q and g jets, response Delta(l_g - l_q).  Stitch variants: full-rank ridge, RRR rank
  4, untrained donor (full), logit-only donor.  I = (r_stitch - r_B) / (r_A - r_B) pooled over eligible ordered pairs (|r_A - r_B|
  > 2 jet-bootstrap SE) as sum (r_s - r_B)(r_A - r_B) / sum (r_A - r_B)^2.  Network-level bootstrap: resample the 7 non-degenerate
  networks with replacement (2,000 draws), pairs of distinct networks weighted by the product of their counts.  Constant-response
  null 0.5 (computed); H0 I = 0.5, bootstrap two-sided p.
Network bootstrap is also used for the pooled same-call means (9 non-degenerate networks, 72 ordered pairs).
Tests: H0 same-call flat above the logit-only donor only at full rank; H1 rank <= 4 reaches >= 85 % and named > random subsets.
  Competing (content = class scores): rank-4 > logit-only with non-overlapping network-level CI excludes it; per pair exact
  McNemar on jets where exactly one of (rank r, logit-only) gives B's call, Holm over 72 pairs.
Output: $AI_RESULTS/jedi/e39x.json (e39x_smoke.json with --smoke).
"""
import argparse, gc, json, os, sys, time
from pathlib import Path
import numpy as np

os.environ.setdefault('KERAS_BACKEND', 'jax')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
T0 = time.time()

from interp.e39_stitch_retrains_jedi import (NETS, EDIT_NETS, DEGEN, LAYERS, ridge, apply, Tail, Boot, build_edits, resp,  # noqa: E402
                                             to_net, r2, holm, binom_p, clean, redrawn, haar)

SF, ST_ = 4941, 4942
RANKS = (1, 2, 4, 8, 16, 64)
TL_RANKS = (1, 2, 4, 8, 64)
RN = ('T', 'DW', 'DZ', 'N')
EDITS4 = ('T', 'D', 'N')
ROLES = ('r2', 'soft', 'hard', 'core')
VARIANTS = ('full', 'rank4', 'untrained', 'logit')


def log(*a): print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


# ------------------------------------------------------------------------------------------------ maps
def rrr(M, Xfit):
    """ridge map M -> {r: rank-r map} (reduced-rank regression on the fitted values)"""
    Yh = (np.asarray(Xfit, np.float64) - M['mx']) @ M['W']
    _, s, Vt = np.linalg.svd(Yh, full_matrices=False); out = {}
    for r in RANKS:
        if r >= M['W'].shape[1]: out[r] = dict(W=M['W'], mx=M['mx'], my=M['my'], Vt=Vt)
        else: out[r] = dict(W=M['W'] @ Vt[:r].T @ Vt[:r], mx=M['mx'], my=M['my'], Vt=Vt[:r])
    return out, s


def random_like(Mr, rng):
    s = np.linalg.svd(Mr['W'], compute_uv=False); d = len(s)
    return dict(W=haar(d, rng) @ np.diag(s) @ haar(d, rng).T, mx=Mr['mx'], my=Mr['my'])


# ------------------------------------------------------------------------------------------------ network-level bootstrap
class NetBoot:
    def __init__(self, nets, nb, seed):
        rng = np.random.default_rng(seed); k = len(nets); self.nets = list(nets)
        self.C = np.stack([np.bincount(rng.integers(0, k, k), minlength=k) for _ in range(nb)]).astype(np.float64)

    def ratio(self, NUM, DEN):
        """pooled sum_{A != B} c_A c_B NUM / sum c_A c_B DEN (diag ignored) -> bootstrap draws"""
        NUM = np.array(NUM, np.float64); DEN = np.array(DEN, np.float64); np.fill_diagonal(NUM, 0); np.fill_diagonal(DEN, 0)
        n = np.einsum('bi,ij,bj->b', self.C, NUM, self.C); d = np.einsum('bi,ij,bj->b', self.C, DEN, self.C)
        with np.errstate(invalid='ignore', divide='ignore'): v = n / d
        return v[np.isfinite(v)]

    def mean(self, V):
        return self.ratio(V, np.ones_like(V))

    def ratio_diff(self, N1, N2, DEN):
        N1, N2, DEN = (np.array(m, np.float64) for m in (N1, N2, DEN))
        for m in (N1, N2, DEN): np.fill_diagonal(m, 0)
        q = lambda M: np.einsum('bi,ij,bj->b', self.C, M, self.C)  # noqa: E731
        with np.errstate(invalid='ignore', divide='ignore'): v = (q(N1) - q(N2)) / q(DEN)
        return v[np.isfinite(v)]


def summ(est, draws, null=None):
    o = dict(est=None if est is None or not np.isfinite(est) else float(est), n_draws=int(len(draws)))
    if len(draws):
        o['ci'] = [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]
        if null is not None:
            o['null'] = null; o['p_vs_null'] = float(min(1.0, 2 * min((draws <= null).mean(), (draws >= null).mean()) + 1 / len(draws)))
    return o


def mat(nets, f):
    return np.array([[f(A, B) if A != B else 0.0 for B in nets] for A in nets], np.float64)


# ------------------------------------------------------------------------------------------------ float tail (for gradients)
def float_tail(model, layer):
    import jax, jax.numpy as jnp
    L = model.L; ks = ('d4', 'd5', 'd6', 'd7') if layer == 'S' else ('d5', 'd6', 'd7')
    acts = {}
    for k in ks:
        a = L[k].activation; nm = getattr(a, '__name__', str(a)) if a is not None else 'linear'
        acts[k] = jax.nn.relu if 'relu' in nm else (lambda t: t) if nm == 'linear' else a
    kern = tuple((jnp.asarray(np.asarray(model.KBnp[k][0], np.float32).reshape(-1, np.asarray(model.KBnp[k][0]).shape[-1])),
                  jnp.asarray(np.asarray(model.KBnp[k][1], np.float32).reshape(-1))) for k in ks)
    names = tuple(acts[k] for k in ks)
    return kern, names


def make_trainer(full, layer, acts, steps, lr):
    import jax, jax.numpy as jnp

    def tail(Y, kern):
        if layer == 'S':
            (K4, b4), (K5, b5), (K6, b6), (K7, b7) = kern; a4, a5, _, a7 = acts
            h = a4(Y @ K4 + b4)
        else:
            (K5, b5), (K6, b6), (K7, b7) = kern; _, a5, _, a7 = (None,) + tuple(acts)
            h = Y
        h5 = a5(h @ K5 + b5); z = h5 @ K6 + b6
        return a7(jax.nn.relu(z) @ K7 + b7)

    def mapf(p, X, mx):
        Xc = X - mx
        return (Xc @ p['W'] if full else (Xc @ p['U']) @ p['V']) + p['c']

    def loss(p, X, y, kern, lo, hi, mx):
        lg = tail(jnp.clip(mapf(p, X, mx), lo, hi), kern)
        lp = lg - jax.scipy.special.logsumexp(lg, axis=1, keepdims=True)
        return -jnp.mean(jnp.take_along_axis(lp, y[:, None], 1))

    g = jax.value_and_grad(loss)

    @jax.jit
    def train(p, Xf, yf, kern, lo, hi, mx, idx):
        m0 = jax.tree_util.tree_map(jnp.zeros_like, p); v0 = jax.tree_util.tree_map(jnp.zeros_like, p)

        def step(c, it):
            p, m, v = c; t, ii = it
            l, gr = g(p, Xf[ii], yf[ii], kern, lo, hi, mx)
            m = jax.tree_util.tree_map(lambda a, b: .9 * a + .1 * b, m, gr)
            v = jax.tree_util.tree_map(lambda a, b: .999 * a + .001 * b * b, v, gr)
            p = jax.tree_util.tree_map(lambda q, a, b: q - lr * (a / (1 - .9 ** t)) / (jnp.sqrt(b / (1 - .999 ** t)) + 1e-8), p, m, v)
            return (p, m, v), l
        (p, _, _), ls = jax.lax.scan(step, (p, m0, v0), (jnp.arange(1, steps + 1, dtype=jnp.float32), idx))
        return p, ls

    @jax.jit
    def full_loss(p, X, y, kern, lo, hi, mx): return loss(p, X, y, kern, lo, hi, mx)

    @jax.jit
    def calls(Y, kern): return jnp.argmax(tail(Y, kern), 1)
    return train, full_loss, calls


def ce(lo, y):
    m = lo.max(1, keepdims=True); lp = lo - m - np.log(np.exp(lo - m).sum(1, keepdims=True))
    return float(-lp[np.arange(len(y)), y].mean())


# ------------------------------------------------------------------------------------------------ main
def main(a):
    import interp.e33_program_recurrence_jedi as E33
    from jetdistill.config import RESULTS
    smoke = a.smoke; per = 200 if smoke else 2000; NB = 200 if smoke else 1000; NBN = 2000; NSUB = 20 if smoke else 200
    STEPS = 30 if smoke else 300
    OUT.mkdir(parents=True, exist_ok=True); JS = OUT / ('e39x_smoke.json' if smoke else 'e39x.json')
    nets = NETS if not smoke else ['official_n64', 'official_n32', 'f16_n64', 'hgq150_s0']
    enets = [n for n in EDIT_NETS if n in nets]; nets_nd = [n for n in nets if n not in DEGEN]; enets_nd = [n for n in enets if n not in DEGEN]
    R = dict(node='E39x', plan='module docstring (fixed before the first run)', smoke=smoke, nets=nets, edit_nets=enets, ranks=RANKS,
             task_ranks=TL_RANKS, seeds=dict(fit=SF, test=ST_, untrained=0))
    save = lambda: JS.write_text(json.dumps(clean(R)))  # noqa: E731

    yall = np.asarray(np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r')['y'])
    idf, idt = E33.rows(yall, per, SF), E33.rows(yall, per, ST_); keep = ~np.isin(idt, idf)
    R['sample'] = dict(fit=f'sample({per}, {SF})', test=f'sample({per}, {ST_}) minus rows in fit', N_fit=len(idf), N_test=int(keep.sum()),
                       overlap_removed=int((~keep).sum()))
    cache = {}

    def data(n, f16):
        if (n, f16) not in cache:
            xf, _, _, yf, _ = E33.load(per, SF, n, f16); xt, _, x3t, yt, _ = E33.load(per, ST_, n, f16)
            cache[(n, f16)] = (xf, yf, xt[keep], yt[keep], x3t[keep])
        return cache[(n, f16)]
    _, yf, _, yt, x3t = data(64, False); J = len(yt); log(f'samples {R["sample"]}')
    EDT = build_edits(x3t, yt, smoke); EDT = {e: EDT[e] for e in EDITS4}; R['edits'] = {k: v[3] for k, v in EDT.items()}
    BT = Boot(J, NB, 4943); yD = yt[EDT['D'][0]]
    NR = dict(T=len(EDT['T'][0]), DW=int((yD == 2).sum()), DZ=int((yD == 3).sum()), N=len(EDT['N'][0]))
    BE = {e: Boot(NR[e], NB, 4944 + i) for i, e in enumerate(RN)}
    SRC = ('S', 'h4', 'logits')

    def stages(M, nm, with_edits):
        xf, _, xt, _, _ = data(M.n, M.f16)
        Rf = M.run(xf, keep=SRC); Rt = M.run(xt, keep=SRC)
        st = {('fit', s): Rf[s].astype(np.float32) for s in SRC}; st.update({('test', s): Rt[s].astype(np.float32) for s in SRC})
        rsp = {}
        if with_edits:
            for e in EDITS4:
                r, xid, xe, _ = EDT[e]
                Ri = M.run(to_net(nm, xid), keep=SRC); Re = M.run(to_net(nm, xe), keep=SRC)
                for s in SRC: st[(e + '_id', s)] = Ri[s].astype(np.float32); st[(e + '_ed', s)] = Re[s].astype(np.float32)
                rsp.update(resp(e, Ri['logits'], Re['logits'], yt[r]))
        return st, rsp

    ST, UST, TAIL, CALL, LIM, RESP, INFO, FT, MEAN, LIVE = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    for nm in nets:
        M = E33.Model(nm); ST[nm], rsp = stages(M, nm, nm in enets)
        LIM[nm] = {L: (ST[nm][('fit', L)].min(0).astype(np.float64), ST[nm][('fit', L)].max(0).astype(np.float64)) for L in LAYERS}
        MEAN[nm] = {L: ST[nm][('fit', L)].astype(np.float64).mean(0) for L in LAYERS}
        LIVE[nm] = np.flatnonzero(ST[nm][('fit', 'S')].std(0) > 1e-6)
        TAIL[nm] = Tail(M); lo = TAIL[nm]('S', ST[nm][('test', 'S')]); CALL[nm] = ST[nm][('test', 'logits')].argmax(1)
        chk = {L: float(np.abs(TAIL[nm](L, ST[nm][('test', L)]) - ST[nm][('test', 'logits')]).max()) for L in LAYERS}
        FT[nm] = {L: float_tail(M, L) for L in LAYERS}
        if nm in enets: RESP[nm] = rsp
        INFO[nm] = dict(acc=float((CALL[nm] == yt).mean()), acc_ci=BT.ci(CALL[nm] == yt), tail_check=chk, n_live_S=int(len(LIVE[nm])),
                        test_ce=ce(lo, yt), resp={e: [float(v.mean()), BE[e].ci(v)] for e, v in rsp.items()})
        log(f'{nm}: acc {INFO[nm]["acc"]:.4f} {INFO[nm]["acc_ci"]}; tail {chk}; live S {len(LIVE[nm])}; resp '
            + json.dumps(clean({k: v[0] for k, v in INFO[nm]['resp'].items()})))
        del M; gc.collect()
        with redrawn(0):
            Mu = E33.Model(nm)
        UST[nm], _ = stages(Mu, nm, nm in enets); del Mu; gc.collect()
        INFO[nm]['untrained_acc'] = float((UST[nm][('test', 'logits')].argmax(1) == yt).mean())
    R['nets_info'] = INFO; save()
    if not smoke and not (0.806 <= INFO['official_n64']['acc'] <= 0.813): log('WARNING: official_n64 accuracy outside 80.6-81.3 %')
    native = {(A, B): float((CALL[A] == CALL[B]).mean()) for A in nets for B in nets}

    # float-tail check (gradient surrogate vs exact)
    import jax.numpy as jnp
    TR = {}
    for L in LAYERS:
        for full, r in [(r_ == 64, r_) for r_ in TL_RANKS]:
            TR[(L, r)] = make_trainer(full, L, FT[nets[0]][L][1], STEPS, 3e-3)
    R['float_tail_check'] = {}
    for nm in nets:
        for L in LAYERS:
            c = np.asarray(TR[(L, 64)][2](jnp.asarray(ST[nm][('test', L)]), FT[nm][L][0]))
            R['float_tail_check'][f'{nm}|{L}'] = float((c == CALL[nm]).mean())
    log('float tail call agreement with exact tail: ' + json.dumps(clean(R['float_tail_check'])))

    def run_map(B, L, Mp, X):
        lo_, hi_ = LIM[B][L]; Y = apply(Mp, X); return TAIL[B](L, np.clip(Y, lo_, hi_)), float(((Y < lo_) | (Y > hi_)).mean())

    def edit_resp(src, B, L, Mp, key):
        o = {}
        for e in EDITS4:
            li, _ = run_map(B, L, Mp, src[(e + '_id', key)]); le, _ = run_map(B, L, Mp, src[(e + '_ed', key)])
            o.update(resp(e, li, le, yt[EDT[e][0]]))
        return o

    # ================================================================ arms 1, 2, 4
    PR = {L: {} for L in LAYERS}; JETSUM = {L: {} for L in LAYERS}; SRESP = {L: {v: {} for v in VARIANTS} for L in LAYERS}
    for L in LAYERS:
        for A in nets:
            tA = time.time()
            for B in nets:
                if A == B: continue
                rng = np.random.default_rng([4945, nets.index(A), nets.index(B), LAYERS.index(L)])
                ed = A in enets and B in enets
                okB = CALL[B]; d = dict(native=native[(A, B)])
                # logit-only donor
                Ml = ridge(ST[A][('fit', 'logits')], ST[B][('fit', L)], rng); lo, _ = run_map(B, L, Ml, ST[A][('test', 'logits')])
                sl = lo.argmax(1) == okB; d['logit'] = dict(same_B=float(sl.mean()), acc=float((lo.argmax(1) == yt).mean()), ce=ce(lo, yt))
                if ed: SRESP[L]['logit'][(A, B)] = edit_resp({k: v for k, v in ST[A].items()}, B, L, Ml, 'logits')
                # trained ridge + RRR
                M = ridge(ST[A][('fit', L)], ST[B][('fit', L)], rng); MR, sv = rrr(M, ST[A][('fit', L)])
                d['alpha'] = M['alpha']; d['fitted_sv'] = sv[:16]; tot = (sv ** 2).sum(); d['fitted_var_frac'] = {r: float((sv[:r] ** 2).sum() / tot) for r in RANKS}
                Mu = ridge(UST[A][('fit', L)], ST[B][('fit', L)], rng); MRu, _ = rrr(Mu, UST[A][('fit', L)])
                d['rank'] = {}
                for r in RANKS:
                    lo, clp = run_map(B, L, MR[r], ST[A][('test', L)]); c = lo.argmax(1); s = c == okB
                    lr_, _ = run_map(B, L, random_like(MR[r], rng), ST[A][('test', L)]); su = None
                    lu, _ = run_map(B, L, MRu[r], UST[A][('test', L)]); su = lu.argmax(1) == okB
                    n10, n01 = int((s & ~sl).sum()), int((~s & sl).sum())
                    d['rank'][r] = dict(same_B=float(s.mean()), acc=float((c == yt).mean()), ce=ce(lo, yt), clipped=clp,
                                        map_r2_test=r2(apply(MR[r], ST[A][('test', L)]), ST[B][('test', L)].astype(np.float64)),
                                        random_same_B=float((lr_.argmax(1) == okB).mean()), untrained_same_B=float(su.mean()),
                                        vs_logit=dict(rank_only=n10, logit_only=n01, p=binom_p(n10, n10 + n01)))
                    JETSUM[L].setdefault(('rank', r), np.zeros(J)); JETSUM[L][('rank', r)] += s
                    if ed and r == 4: SRESP[L]['rank4'][(A, B)] = edit_resp(ST[A], B, L, MR[r], L)
                    if ed and r == 64: SRESP[L]['full'][(A, B)] = edit_resp(ST[A], B, L, MR[r], L)
                if ed: SRESP[L]['untrained'][(A, B)] = edit_resp(UST[A], B, L, MRu[64], L)
                JETSUM[L].setdefault('logit', np.zeros(J)); JETSUM[L]['logit'] += sl
                # task-loss maps
                d['task'] = {}
                Xf = jnp.asarray(ST[A][('fit', L)]); yfj = jnp.asarray(yf.astype(np.int32)); lo_, hi_ = LIM[B][L]
                idx = jnp.asarray(rng.integers(0, len(yf), (STEPS, 2048)).astype(np.int32))
                for r in TL_RANKS:
                    Mr = MR[r]
                    if r == 64: p0 = dict(W=jnp.asarray(Mr['W'], jnp.float32), c=jnp.asarray(Mr['my'], jnp.float32))
                    else: p0 = dict(U=jnp.asarray(M['W'] @ Mr['Vt'].T, jnp.float32), V=jnp.asarray(Mr['Vt'], jnp.float32), c=jnp.asarray(Mr['my'], jnp.float32))
                    args = (FT[B][L][0], jnp.asarray(lo_, jnp.float32), jnp.asarray(hi_, jnp.float32), jnp.asarray(Mr['mx'], jnp.float32))
                    l0 = float(TR[(L, r)][1](p0, Xf, yfj, *args)); p, ls = TR[(L, r)][0](p0, Xf, yfj, *args, idx)
                    l1 = float(TR[(L, r)][1](p, Xf, yfj, *args))
                    Wt = np.asarray(p['W'], np.float64) if r == 64 else np.asarray(p['U'], np.float64) @ np.asarray(p['V'], np.float64)
                    Mt = dict(W=Wt, mx=Mr['mx'], my=np.asarray(p['c'], np.float64))
                    lo, clp = run_map(B, L, Mt, ST[A][('test', L)]); c = lo.argmax(1); s = c == okB
                    d['task'][r] = dict(same_B=float(s.mean()), acc=float((c == yt).mean()), ce=ce(lo, yt), clipped=clp, fit_ce_float_init=l0,
                                        fit_ce_float_end=l1, minibatch_ce_last=float(np.asarray(ls)[-10:].mean()))
                    JETSUM[L].setdefault(('task', r), np.zeros(J)); JETSUM[L][('task', r)] += s
                PR[L][(A, B)] = d
            log(f'[{L}] donor {A} ({time.time() - tA:.0f} s): ' + '; '.join(
                f'{B}: r1 {PR[L][(A, B)]["rank"][1]["same_B"]:.3f} r4 {PR[L][(A, B)]["rank"][4]["same_B"]:.3f} full {PR[L][(A, B)]["rank"][64]["same_B"]:.3f} '
                f'logit {PR[L][(A, B)]["logit"]["same_B"]:.3f} task4 {PR[L][(A, B)]["task"][4]["same_B"]:.3f}' for B in nets if B != A))
        R.setdefault('pairs', {})[L] = {f'{A}|{B}': v for (A, B), v in PR[L].items()}; save()

    # ---- pooled same-call vs rank (network bootstrap over non-degenerate nets) + jet bootstrap of the pair-mean indicator
    NBt = NetBoot(nets_nd, NBN, 4946); R['rank_summary'] = {}
    npair = len(nets) * (len(nets) - 1)
    for L in LAYERS:
        P = PR[L]; o = {}
        get = {('rank', r): (lambda A, B, r=r: P[(A, B)]['rank'][r]['same_B']) for r in RANKS}
        get.update({('random', r): (lambda A, B, r=r: P[(A, B)]['rank'][r]['random_same_B']) for r in RANKS})
        get.update({('untrained', r): (lambda A, B, r=r: P[(A, B)]['rank'][r]['untrained_same_B']) for r in RANKS})
        get.update({('task', r): (lambda A, B, r=r: P[(A, B)]['task'][r]['same_B']) for r in TL_RANKS})
        get.update({('task_acc', r): (lambda A, B, r=r: P[(A, B)]['task'][r]['acc']) for r in TL_RANKS})
        get.update({('rank_acc', r): (lambda A, B, r=r: P[(A, B)]['rank'][r]['acc']) for r in RANKS})
        get.update({('task_ce', r): (lambda A, B, r=r: P[(A, B)]['task'][r]['ce']) for r in TL_RANKS})
        get.update({('rank_ce', r): (lambda A, B, r=r: P[(A, B)]['rank'][r]['ce']) for r in RANKS})
        get['logit'] = lambda A, B: P[(A, B)]['logit']['same_B']; get['native'] = lambda A, B: P[(A, B)]['native']
        V = {k: mat(nets_nd, f) for k, f in get.items()}; nd = len(nets_nd) * (len(nets_nd) - 1)
        for k, Vm in V.items():
            key = k if isinstance(k, str) else f'{k[0]}_{k[1]}'
            o[key] = summ(Vm.sum() / nd, NBt.mean(Vm))
            if k == 'logit' or (isinstance(k, tuple) and k[0] in ('rank', 'task')):
                if k in JETSUM[L]: o[key]['jet_ci_all_pairs'] = BT.ci(JETSUM[L][k] / npair)
        for r in RANKS:
            Dm = V[('rank', r)] - V['logit']; o[f'rank{r}_minus_logit'] = summ(Dm.sum() / nd, NBt.mean(Dm), null=0.0)
            Dm = V[('rank', r)] - V[('untrained', r)]; o[f'rank{r}_minus_untrained'] = summ(Dm.sum() / nd, NBt.mean(Dm), null=0.0)
            ks = [(A, B) for A in nets_nd for B in nets_nd if A != B]
            hp = holm([P[k]['rank'][r]['vs_logit']['p'] for k in ks])
            o[f'rank{r}_vs_logit_mcnemar'] = dict(n_pairs=len(ks), holm_lt_0p05_rank_higher=int(sum(h < .05 and P[k]['rank'][r]['vs_logit']['rank_only'] > P[k]['rank'][r]['vs_logit']['logit_only'] for h, k in zip(hp, ks))),
                                                  holm_lt_0p05_logit_higher=int(sum(h < .05 and P[k]['rank'][r]['vs_logit']['rank_only'] < P[k]['rank'][r]['vs_logit']['logit_only'] for h, k in zip(hp, ks))))
        for r in TL_RANKS:
            for m in ('', '_acc', '_ce'):
                Dm = V[(f'task{m}', r)] - V[(f'rank{m}', r)]; o[f'task{r}_minus_rank{r}{m.replace("_", "_") or "_same"}'] = summ(Dm.sum() / nd, NBt.mean(Dm), null=0.0)
        R['rank_summary'][L] = o
        log(f'[{L}] same-call vs rank (net bootstrap): ' + ', '.join(f'r{r} {o[f"rank_{r}"]["est"]:.3f} {o[f"rank_{r}"].get("ci")}' for r in RANKS)
            + f'; logit {o["logit"]["est"]:.3f} {o["logit"].get("ci")}; native {o["native"]["est"]:.3f}; task ' + ', '.join(f'r{r} {o[f"task_{r}"]["est"]:.3f}' for r in TL_RANKS)
            + f'; r4-logit {o["rank4_minus_logit"]}; random r4 {o["random_4"]["est"]:.3f}; untrained r4 {o["untrained_4"]["est"]:.3f}')
    save()

    # ================================================================ arm 3: subset stitch at S
    try:
        ex = json.loads((OUT / 'e33x.json').read_text())['roles_units']
        named = {m: {r_: int(v[0][1:]) for r_, v in ex[m].items() if v[2] is not None and v[2] >= 0.9} for m in nets if m in ex}
        R['named_units'] = {m: {k: f'S{u}' for k, u in v.items()} for m, v in named.items()}

        def subset_eval(A, B, ua, ub, Xa_fit, Xa_test, rng):
            M = ridge(Xa_fit[:, ua], ST[B][('fit', 'S')][:, ub], rng)
            Y = np.tile(MEAN[B]['S'], (J, 1)); Y[:, ub] = apply(M, Xa_test[:, ua])
            lo_, hi_ = LIM[B]['S']; c = TAIL[B]('S', np.clip(Y, lo_, hi_)).argmax(1)
            return float((c == CALL[B]).mean())

        def own_eval(B, ub):
            Y = np.tile(MEAN[B]['S'], (J, 1)); Y[:, ub] = ST[B][('test', 'S')][:, ub]; return float((TAIL[B]('S', Y).argmax(1) == CALL[B]).mean())
        own = {}
        for B in nets_nd:
            if B not in named or not named[B]: continue
            rng = np.random.default_rng([4947, nets.index(B)]); ub = [named[B][k] for k in ROLES if k in named[B]]; k = len(ub)
            rv = [own_eval(B, list(rng.choice(LIVE[B], k, replace=False))) for _ in range(NSUB)]
            own[B] = dict(k=k, named=own_eval(B, ub), random_mean=float(np.mean(rv)), random_q95=float(np.quantile(rv, .95)),
                          p=float((1 + sum(v >= own_eval(B, ub) for v in rv)) / (1 + NSUB)))
        R['subset_own'] = own; log('subset own (B units only, rest at mean): ' + json.dumps(clean(own)))
        rows = []
        for A in nets_nd:
            for B in nets_nd:
                if A == B or A not in named or B not in named: continue
                shared = [k for k in ROLES if k in named[A] and k in named[B]]
                if not shared: continue
                rng = np.random.default_rng([4948, nets.index(A), nets.index(B)]); k = len(shared)
                ua = [named[A][s] for s in shared]; ub = [named[B][s] for s in shared]
                Xaf, Xat = ST[A][('fit', 'S')], ST[A][('test', 'S')]
                v = subset_eval(A, B, ua, ub, Xaf, Xat, rng)
                rv = np.array([subset_eval(A, B, list(rng.choice(LIVE[A], k, replace=False)), list(rng.choice(LIVE[B], k, replace=False)), Xaf, Xat, rng)
                               for _ in range(NSUB)])
                # mean-only reference: all of B's S at its mean
                rows.append(dict(pair=f'{A}|{B}', roles=shared, k=k, named=v, random_mean=float(rv.mean()), random_q05=float(np.quantile(rv, .05)),
                                 random_q95=float(np.quantile(rv, .95)), p=float((1 + (rv >= v).sum()) / (1 + NSUB)),
                                 full_rank_same_B=PR['S'][(A, B)]['rank'][64]['same_B'], rank_k_same_B=PR['S'][(A, B)]['rank'][min(RANKS, key=lambda r: abs(r - k))]['same_B']))
            log(f'subset donor {A}: ' + '; '.join(f'{r_["pair"].split("|")[1]} k{r_["k"]} {r_["named"]:.3f} vs {r_["random_mean"]:.3f} p {r_["p"]:.3f}' for r_ in rows if r_['pair'].startswith(A + '|')))
        hp = holm([r_['p'] for r_ in rows])
        for r_, h in zip(rows, hp): r_['p_holm'] = float(h)
        allmean = {B: float((TAIL[B]('S', np.tile(MEAN[B]['S'], (J, 1))).argmax(1) == CALL[B]).mean()) for B in nets_nd}
        Dn = mat(nets_nd, lambda A, B: next((r_['named'] - r_['random_mean'] for r_ in rows if r_['pair'] == f'{A}|{B}'), np.nan))
        msk = np.isfinite(Dn).astype(float); np.fill_diagonal(msk, 0); Dn0 = np.nan_to_num(Dn)
        R['subset'] = dict(rows=rows, n_pairs=len(rows), n_holm_lt_0p05=int((np.array(hp) < .05).sum()), all_at_mean_same_B=allmean,
                           mean_named=float(np.mean([r_['named'] for r_ in rows])) if rows else None,
                           mean_random=float(np.mean([r_['random_mean'] for r_ in rows])) if rows else None,
                           named_minus_random_netboot=summ(Dn0.sum() / max(msk.sum(), 1), NBt.ratio(Dn0, msk), null=0.0),
                           by_k={int(k): dict(n=sum(r_['k'] == k for r_ in rows), named=float(np.mean([r_['named'] for r_ in rows if r_['k'] == k])),
                                              random=float(np.mean([r_['random_mean'] for r_ in rows if r_['k'] == k]))) for k in sorted({r_['k'] for r_ in rows})})
        log(f'subset: {len(rows)} pairs; named {R["subset"]["mean_named"]} vs random {R["subset"]["mean_random"]}; Holm<0.05 {R["subset"]["n_holm_lt_0p05"]}; '
            f'net boot diff {R["subset"]["named_minus_random_netboot"]}; all-at-mean {allmean}')
    except Exception as err:  # noqa: BLE001
        import traceback; traceback.print_exc(); R['subset'] = dict(error=repr(err)); log(f'subset failed: {err!r}')
    save()

    # ================================================================ arm 4: inheritance at network level
    NBe = NetBoot(enets_nd, NBN, 4949); R['inheritance'] = {}
    for L in LAYERS:
        R['inheritance'][L] = {}
        for e in RN:
            Bx = BE[e]; elig = {}
            for A in enets:
                for B in enets:
                    if A == B: continue
                    se = float(Bx.means(RESP[A][e] - RESP[B][e]).std()); elig[(A, B)] = abs(float((RESP[A][e] - RESP[B][e]).mean())) > 2 * se
            rA = {n: float(RESP[n][e].mean()) for n in enets}
            E = mat(enets_nd, lambda A, B: float(elig[(A, B)]))
            DEN = mat(enets_nd, lambda A, B: (rA[A] - rA[B]) ** 2) * E
            o = dict(n_eligible=int(E.sum()), n_pairs=len(enets_nd) * (len(enets_nd) - 1), r_net={n: rA[n] for n in enets})
            for v in VARIANTS:
                rs = {k: float(val[e].mean()) for k, val in SRESP[L][v].items()}
                NUM = mat(enets_nd, lambda A, B: (rs[(A, B)] - rA[B]) * (rA[A] - rA[B])) * E
                est = NUM.sum() / DEN.sum() if DEN.sum() > 0 else np.nan
                o[v] = summ(est, NBe.ratio(NUM, DEN), null=0.5)
                o[v]['rows'] = {f'{A}|{B}': dict(r_A=rA[A], r_B=rA[B], r_stitch=rs[(A, B)], eligible=bool(elig[(A, B)])) for (A, B) in rs}
                cm = np.mean([rs[(A, B)] for A in enets_nd for B in enets_nd if A != B and elig[(A, B)]]) if E.sum() else np.nan
                NUMc = mat(enets_nd, lambda A, B: (cm - rA[B]) * (rA[A] - rA[B])) * E
                o[v]['const_null'] = float(NUMc.sum() / DEN.sum()) if DEN.sum() > 0 else None
            for v in ('untrained', 'logit', 'rank4'):
                rs1 = {k: float(val[e].mean()) for k, val in SRESP[L]['full'].items()}; rs2 = {k: float(val[e].mean()) for k, val in SRESP[L][v].items()}
                N1 = mat(enets_nd, lambda A, B: (rs1[(A, B)] - rA[B]) * (rA[A] - rA[B])) * E
                N2 = mat(enets_nd, lambda A, B: (rs2[(A, B)] - rA[B]) * (rA[A] - rA[B])) * E
                o[f'full_minus_{v}'] = summ((N1.sum() - N2.sum()) / DEN.sum() if DEN.sum() > 0 else np.nan, NBe.ratio_diff(N1, N2, DEN), null=0.0)
            R['inheritance'][L][e] = o
            log(f'[{L}] {e}: eligible {o["n_eligible"]}/{o["n_pairs"]}; ' + '; '.join(f'{v} I {o[v]["est"]} {o[v].get("ci")} p(0.5) {o[v].get("p_vs_null")} const {o[v]["const_null"]}' for v in VARIANTS)
                + f'; full-untrained {o["full_minus_untrained"]}; full-logit {o["full_minus_logit"]}')
    save(); log(f'wrote {JS}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); main(ap.parse_args())
