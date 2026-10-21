"""E25x (claims A4c, A4, A9; front-to-back construction of the mass inputs): where M0 = Σ pT and M2 = Σ pT ΔR² (stored axis)
first become causal low-dimensional variables between the input and S, in the official 3-feature and 16-feature JEDI-linear n64.
   python -m interp.e25x_p_to_s --model {3f,16f} --part trained            -> $AI_RESULTS/jedi/e25x_<model>_trained[_smoke].json
   python -m interp.e25x_p_to_s --model {3f,16f} --part base --shard i --nshard n  -> e25x_<model>_base<i>[_smoke].json
   python -m interp.e25x_p_to_s --model {3f,16f} --part combine             -> e25x_<model>[_smoke].json

Sample: fit = interp.jedi_core.sample(2000, 2551) (10,000 jets), test = sample(2000, 2552) minus rows also in the fit sample.
All 16 jetConstituentList columns (val_c16_f16.npy, first 64 slots) at those rows, checked against sample() x; the 3f model reads
the pt / etarel / phirel columns of the same (edited) arrays. Pairs (base, source): 3,000 from the fit sample (rng 2553; last 20 %
for model selection), 3,000 from the test sample (rng 2554). Edits = interp.e25w_jedi16_mass.pair_edits on the base jet:
  M0: pT x c, angles x c^-1/2 (c = M0_s / M0_b; M2 and every pT_i r_i^2 unchanged; every pT_i edited)
  M2: angles x sqrt(M2_s / M2_b) (pT_i unchanged; every pT_i r_i^2 edited)
  D (input, E25w): pT x sqrt(M0_s M2_s / (M0_b M2_b)), angles fixed (a pT rescale).
Counterfactual label = the network's exact call on the edited base jet. Effective IIA = agreement on test pairs whose label differs
from the base call; 1,000-resample bootstrap over pairs.
Interventions (h' = clip(h_b + (h_x - h_b) Q Q^T), Q orthonormal k columns, k = 1, 2; clip = per-unit range of the stage over the
fit sample widened to the base value; at S the E25w quantizer range):
  design "src" (jet level, g and S; E25w/X10): h_x = the source jet's stage vector.
  design "don" (all stages): h_x = the same stage of the edited base jet (donor). At the per-particle stages a (l1 output; its pool
    is P) and u (l3 output; its pool is S) the same Q acts on every particle i: a_i' = a_i + (a_i^don - a_i) Q Q^T, and the rest of
    the exact network (pool to P, g, l2, add, l3, pool to S, h4, h5, z, d7) runs on a'. Particle i of base and donor is the same
    particle with pT_i (M0 edit) or pT_i r_i^2 (M2 edit) changed, so this asks whether a k-dim per-particle direction carries the
    edited per-particle quantity such that the pooled sum reproduces the donor's call. g: g' mixed, l2 path from the base jet.
  Q trained with Adam 0.01, straight-through estimator through every quantizer, minibatch 512 (per-particle and g) or full batch
  (S, interp.e25w_jedi16_mass.das_fit), balanced class-change weights, selection on balanced held-out agreement every 50 steps.
Controls: 1,000 random orthonormal k-subspaces per trained cell (p_rot = (1 + #rand >= DAS) / 1,001; Holm); full swap at the stage;
  per-particle linear read-out (ridge) of pT_i and pT_i r_i^2 from a_i and u_i (R^2 on test particles; calibration = read-out change
  / true change under the M0 and M2 donor edits).
Baselines (part base, each with a logistic-regression refit of d7 on the fit sample; qualifies if test accuracy > 0.30 (chance
  0.20) and >= 3 distinct calls; the E25w bar of 0.40 was met by no 3f baseline in the first run of this script, so it was lowered
  before any baseline DAS was computed): untr_s (jedi_core untrained=True, every kernel / bias / BN scale redrawn, s 0-9), init_s
  (l1 trained; l2 dd l3 d4 d5 d6 Glorot re-init, s 0-14), gauss_s (same layers N(mean, sd), s 0-14), perm_s (X8 variant a, s 0-4). Same pairs, edits, DAS
  (k 1, 2; 300 steps) with each baseline's own labels.  H0 per cell (stage x variable x k, design don and src): trained IIA <= the
  95th percentile of the qualifying baselines; p = (1 + #{bootstrap replicates of trained IIA <= q95}) / (1 + 1000); also the rank
  p = (1 + #{b: IIA_b >= IIA_tr}) / (1 + n_b). Holm over the cells.
Arm 3 (D without a pT rescale): the k = 1 src DAS directions q0 (M0) and q2 (M2) at S; calibration [q0.S, q2.S] = alpha + B [log M0,
  log M2] (OLS, fit sample). do_S(D): S' = S_b + Qm (Qm^T Qm)^-1 B (delta, delta), delta = D_s - D_b (M0 x lam, M2 x lam in the S
  coordinates, lam = e^delta); fixed-D control: B (delta, -delta) (M0 x lam, M2 / lam). Labels = exact call from S'. DAS (src design,
  k 1, 2) for these labels at S and at h4; paired bootstrap IIA(h4) - IIA(S) (same pairs, same labels); the same for the input
  do(D) labels (pT rescale, E25w).
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True, choices=['3f', '16f'])
ap.add_argument('--part', default='trained', choices=['trained', 'base', 'combine'])
ap.add_argument('--var', default='M0', choices=['M0', 'M2', 'arm3']); ap.add_argument('--shard', type=int, default=0); ap.add_argument('--nshard', type=int, default=1); ap.add_argument('--smoke', action='store_true')
args = ap.parse_args()
SM = args.smoke
import interp.e25w_jedi16_mass as E  # noqa: E402  (its argparse runs with a fixed smoke list on import; only functions are used)
from interp.e19_common import wilson, holm  # noqa: E402

T0 = time.time()
def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)
PER = 120 if SM else 2000
NTR, NTE = (500, 400) if SM else (3000, 3000)
NRAND = 10 if SM else 1000
STEPS = 60 if SM else 600
BSTEPS = 40 if SM else 250
NB = 200 if SM else 1000
BS = 512
QACC = 0.30   # qualification: test accuracy > 0.30 (chance 0.20) and >= 3 distinct calls (first 3f run: no baseline reached 0.40)
EB = 500
VARS = ('M0', 'M2')
KS = (1, 2)
FAM = {'3f': '3-feature', '16f': '16-feature'}[args.model]
SEL = (lambda A: A[..., [5, 8, 11]]) if args.model == '3f' else (lambda A: A)
TAG = f'{args.model}{"_smoke" if SM else ""}'


def clean(o): return E.clean(o)


# ------------------------------------------------------------------------------------------------ sample
def load():
    from jetdistill.config import RESULTS, DATA
    from interp.jedi_core import sample
    y = np.asarray(np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r')['y'])
    V = np.load(Path(DATA) / 'val_c16_f16.npy', mmap_mode='r'); out = {}; dmax = 0.0
    for nm, seed in (('fit', 2551), ('test', 2552)):
        idx = E.rows(y, PER, seed); x3, ys, _ = sample(PER, seed); assert np.array_equal(ys, y[idx])
        X = np.asarray(V[idx, :64]).astype(np.float32); d = float(np.abs(X[..., [5, 8, 11]] - x3).max()); dmax = max(dmax, d)
        if d > 1e-3: raise SystemExit(f'ABORT: 16-feature columns differ from sample() x ({nm}: {d})')
        out[nm] = (X, ys, idx)
    keep = ~np.isin(out['test'][2], out['fit'][2])
    out['test'] = tuple(a[keep] for a in out['test'])
    log(f'fit N {len(out["fit"][0])}; test N {len(out["test"][0])} ({int((~keep).sum())} overlapping rows removed); max col diff {dmax:.2g}')
    return out, dmax, int((~keep).sum())


def make_pairs(n, npair, seed, I):
    return E.make_pairs(np.arange(n), npair, seed, I)


# ------------------------------------------------------------------------------------------------ per-particle forward
class PNet:
    """exact JEDI-linear forward with mixing at a (per particle), g, u (per particle); fused kernels as an argument"""
    _cache = {}

    def __init__(self, net):
        self.net = net; self.j = net.j; self.L = net.L
        key = id(self.j)
        if key not in PNet._cache: PNet._cache[key] = self._build()
        self.reps_f, self.ev, self.vg, self.gshape = PNet._cache[key]

    def _build(self):
        import jax, jax.numpy as jnp
        L = self.L; sg = jax.lax.stop_gradient

        def Q_(q, x, ste): return x + sg(q - x) if ste else q

        def dense(k, x, KB, ste, act=True):
            q = Q_(L[k].iq(x, training=False), x, ste); z = jnp.einsum(L[k].equation, q, KB[k][0]) + KB[k][1]
            return L[k].activation(z) if act and L[k].activation is not None else z

        def pool(k, a, ste):
            return jnp.sum(Q_(L[k].iq(a, training=False), a, ste), axis=L[k].axes, keepdims=L[k].keepdims) * L[k].scale

        def from_u(u, KB, ste):
            J = u.shape[0]; S = pool('qs1', u, ste); h4 = dense('d4', S, KB, ste); h5 = dense('d5', h4, KB, ste)
            z = dense('d6', h5, KB, ste, act=False); return dense('d7', jax.nn.relu(z), KB, ste).reshape(J, -1)

        def from_sg(s, g, KB, ste):
            qs_, qd_ = L['add'].iq([s, g], training=False)
            if ste: qs_, qd_ = Q_(qs_, s, True), Q_(qd_, g, True)
            return from_u(dense('l3', qs_ + qd_, KB, ste), KB, ste)

        def from_a(a, KB, ste):
            P = pool('qs', a, ste); g = dense('dd', P, KB, ste); s = dense('l2', a, KB, ste); return from_sg(s, g, KB, ste)

        def reps(xn, KB):
            J = xn.shape[0]; a = dense('l1', xn, KB, False); P = pool('qs', a, False); g = dense('dd', P, KB, False)
            s = dense('l2', a, KB, False); qs_, qd_ = L['add'].iq([s, g], training=False); u = dense('l3', qs_ + qd_, KB, False)
            return dict(a=a, s=s, g=g.reshape(J, -1), u=u, logits=from_u(u, KB, False))
        F = 16 if self.j.family.startswith('16') else 3
        KBs = {k: (jax.ShapeDtypeStruct(np.shape(v[0]), jnp.float32), jax.ShapeDtypeStruct(np.shape(v[1]), jnp.float32)) for k, v in self.net.KBn.items()}
        gsh = jax.eval_shape(lambda xn, KB: dense('dd', pool('qs', dense('l1', xn, KB, False), False), KB, False),
                             jax.ShapeDtypeStruct((2, self.j.n, F), jnp.float32), KBs).shape[1:]

        def logits_stage(stage, ste, ctx, hb, hx, Q, lo, hi, KB):
            h = jnp.clip(hb + ((hx - hb) @ Q) @ Q.T, jnp.minimum(lo, hb), jnp.maximum(hi, hb))
            if stage == 'a': return from_a(h, KB, ste)
            if stage == 'u': return from_u(h, KB, ste)
            return from_sg(ctx, h.reshape((h.shape[0],) + tuple(gsh)), KB, ste)
        ev = {st: jax.jit(lambda ctx, hb, hx, Q, lo, hi, KB, st=st: logits_stage(st, False, ctx, hb, hx, Q, lo, hi, KB)) for st in ('a', 'g', 'u')}

        def loss(stage, Wm, ctx, hb, hx, Y, Wt, lo, hi, KB):
            Q, _ = jnp.linalg.qr(Wm); lp = jax.nn.log_softmax(logits_stage(stage, True, ctx, hb, hx, Q, lo, hi, KB))
            return -(Wt * lp[jnp.arange(Y.shape[0]), Y]).sum()
        vg = {st: jax.jit(jax.value_and_grad(lambda *a_, st=st: loss(st, *a_))) for st in ('a', 'g', 'u')}
        return jax.jit(reps), ev, vg, gsh

    def reps(self, x, keep):
        out = {k: [] for k in keep}
        for i in range(0, len(x), EB):
            xb = np.asarray(x[i:i + EB]); nb = len(xb); xn = self.j.inputs(xb, self.j.n)
            if nb < EB: xn = np.concatenate([xn, np.zeros((EB - nb,) + xn.shape[1:], np.float32)])
            r = self.reps_f(xn, self.net.KB)
            for k in keep: out[k].append(np.asarray(r[k][:nb], np.float32))
        return {k: np.concatenate(v) for k, v in out.items()}

    def ranges(self, x):
        lo, hi = {}, {}
        for i in range(0, len(x), EB):
            r = self.reps(x[i:i + EB], ('a', 'g', 'u'))
            for k, v in r.items():
                v = v.reshape(-1, v.shape[-1]); lo[k] = np.minimum(lo.get(k, np.inf), v.min(0)); hi[k] = np.maximum(hi.get(k, -np.inf), v.max(0))
        return {k: (lo[k].astype(np.float32), hi[k].astype(np.float32)) for k in lo}

    def calls(self, st, ctx, hb, hx, Q, rg):
        lo, hi = (np.asarray(v, np.float32) for v in rg); Q = np.asarray(Q, np.float32); out = []
        for i in range(0, len(hb), EB):
            sl = slice(i, i + EB); nb = len(hb[sl]); arrs = [ctx[sl], hb[sl], hx[sl]]
            if nb < EB: arrs = [np.concatenate([v, np.repeat(v[-1:], EB - nb, 0)]) for v in arrs]
            out.append(np.asarray(self.ev[st](*arrs, Q, lo, hi, self.net.KB))[:nb].argmax(1))
        return np.concatenate(out)


def dummy(n): return np.zeros((n, 1), np.float32)


def fit_pp(pn, st, ctx, hb, hx, lab, basec, rg, k, seed, steps, bsize=BS):
    """DAS at a / g / u: minibatch Adam with STE; selection on balanced held-out agreement (last 20 %)"""
    import jax.numpy as jnp
    n = len(lab); nv = n // 5; tr = np.arange(n - nv); va = np.arange(n - nv, n)
    chg = lab != basec; w = np.where(chg, 0.5 / max(chg[tr].sum(), 1), 0.5 / max((~chg[tr]).sum(), 1)).astype(np.float32)
    lo, hi = (jnp.asarray(np.asarray(v, np.float32)) for v in rg); vg = pn.vg[st]
    rng = np.random.default_rng(seed); d = hb.shape[-1]; Wm = jnp.asarray(rng.normal(size=(d, k)).astype(np.float32))
    m = jnp.zeros_like(Wm); v = jnp.zeros_like(Wm); b1, b2 = 0.9, 0.999; best = (-1, None, 0); chv = chg[va]; Lf = None; L_ = None
    bs = min(bsize, len(tr))
    for t in range(1, steps + 1):
        bi = rng.choice(tr, bs, replace=False); wt = w[bi] * (len(tr) / bs)
        L_, g = vg(Wm, ctx[bi], hb[bi], hx[bi], jnp.asarray(lab[bi].astype(np.int32)), jnp.asarray(wt), lo, hi, pn.net.KB)
        Lf = Lf if Lf is not None else float(L_)
        m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
        Wm = Wm - 0.01 * (m / (1 - b1 ** t)) / (jnp.sqrt(v / (1 - b2 ** t)) + 1e-8)
        if t % 50 == 0 or t == steps:
            Q = np.linalg.qr(np.asarray(Wm, np.float64))[0]
            c = pn.calls(st, ctx[va], hb[va], hx[va], Q, rg); ok = c == lab[va]
            sc = 0.5 * ok[chv].mean() + 0.5 * ok[~chv].mean() if chv.any() and (~chv).any() else ok.mean()
            if sc > best[0]: best = (float(sc), Q, t)
    return best[1], dict(val_balanced=best[0], best_step=best[2], loss_first=Lf, loss_last=float(L_))


def boot_reps(c, lab, basec, nb=NB, seed=7):
    rng = np.random.default_rng(seed); ok = (c == lab).astype(float); EH = (lab != basec).astype(float); n = len(c); v = []
    for _ in range(nb):
        i = rng.integers(0, n, n); d = EH[i].sum()
        if d >= 10: v.append((ok[i] * EH[i]).sum() / d)
    return np.asarray(v)


# ------------------------------------------------------------------------------------------------ setup shared by parts
def setup():
    D, dmax, nover = load(); Xf, yf, _ = D['fit']; Xt, yt, _ = D['test']
    If, It = E.jinfo(Xf), E.jinfo(Xt)
    pairs = dict(train=make_pairs(len(Xf), NTR, 2553, If), test=make_pairs(len(Xt), NTE, 2554, It))
    PE, meta = {}, {}
    for arm, X, I in (('train', Xf, If), ('test', Xt, It)):
        b, s = pairs[arm]; pe, meta[arm] = E.pair_edits(X, I, b, s)
        PE[arm] = {V: SEL(v) for V, v in pe.items()}
    S = dict(fit=SEL(Xf), test=SEL(Xt), yf=yf, yt=yt, If=If, It=It, pairs=pairs, PE=PE)
    R = dict(sample=dict(fit=dict(per_class=PER, seed=2551, N=len(Xf)), test=dict(per_class=PER, seed=2552, N=len(Xt), removed_overlap=nover),
                         max_col_diff=dmax, pairs=dict(train=NTR, test=NTE, rng=(2553, 2554))), pair_meta=meta, procedure=__doc__)
    return S, R


def stage_pair(pn, net, S, arm, V, st):
    """per pair: (ctx, base, donor) arrays at stage st for the edited-base donor design"""
    X = S['fit'] if arm == 'train' else S['test']; b, _ = S['pairs'][arm]; xd = S['PE'][arm][V]
    if st == 'S': return None, net.run(X[b], ('S',))['S'], net.run(xd, ('S',))['S']
    keep = (st, 's') if st == 'g' else (st,)
    rb = pn.reps(X[b], keep); rd = pn.reps(xd, (st,))
    return (rb['s'] if st == 'g' else dummy(len(b))), rb[st], rd[st]


def labels_for(net, S, rf, rt):
    cf, ct = rf['logits'].argmax(1), rt['logits'].argmax(1); lab = {}
    for arm in ('train', 'test'):
        b, s = S['pairs'][arm]; c_all = cf if arm == 'train' else ct
        lab[arm] = dict(base=c_all[b], source=c_all[s], **{V: net.calls(S['PE'][arm][V]) for V in ('M0', 'M2', 'D')})
    return lab


def run_net_stages(net, pn, S, R_net, steps, nrand, trained, vars_=VARS, bsize=BS):
    """DAS cells for one network: stages a, u, g (PNet) and S (E25w das_fit), design don (+ src at g, S)"""
    import resource
    Xf, Xt = S['fit'], S['test']; out = {}
    rg = pn.ranges(Xf)
    rf = net.run(Xf, ('S', 'logits')); rt = net.run(Xt, ('S', 'logits'))
    gf = pn.reps(Xf, ('g',))['g']; gt = pn.reps(Xt, ('g',))['g']
    lab = labels_for(net, S, rf, rt)
    R_net['cf_change_frac_test'] = {V: float((lab['test'][V] != lab['test']['base']).mean()) for V in ('M0', 'M2', 'D')}
    R_net['n_eff_test'] = {V: int((lab['test'][V] != lab['test']['base']).sum()) for V in ('M0', 'M2', 'D')}
    log(f'  {net.name}: counterfactual call changes (test) {R_net["cf_change_frac_test"]}')
    btr, str_ = S['pairs']['train']; bte, ste_ = S['pairs']['test']
    for V in vars_:
        Ltr, Lte, base_tr, base_te = lab['train'][V], lab['test'][V], lab['train']['base'], lab['test']['base']
        if trained:  # exactness checks
            rb = pn.reps(Xt[bte[:500]], ('logits',))['logits']; rd = pn.reps(S['PE']['test'][V][:500], ('logits',))['logits']
            chk = dict(pnet_base_vs_net=float(np.abs(rb - rt['logits'][bte[:500]]).max()), pnet_donor_call_eq_label=float((rd.argmax(1) == Lte[:500]).mean()))
            R_net.setdefault('checks', {})[V] = chk; log(f'  checks {V}: {chk}')
        for st in ('a', 'u', 'g', 'S'):
            ctx_tr, hb_tr, hd_tr = stage_pair(pn, net, S, 'train', V, st); ctx_te, hb_te, hd_te = stage_pair(pn, net, S, 'test', V, st)
            cells = [('don', hd_tr, hd_te)]
            if st == 'g': cells.append(('src', gf[str_], gt[ste_]))
            if st == 'S': cells.append(('src', rf['S'][str_], rt['S'][ste_]))
            for des, hx_tr, hx_te in cells:
                t1 = time.time()
                if st == 'S':
                    def calls(Q, hb_te=hb_te, hx_te=hx_te): return E.eval_Q(net, 'S', hb_te, hx_te, Q)  # noqa: E731
                else:
                    def calls(Q, st=st, ctx_te=ctx_te, hb_te=hb_te, hx_te=hx_te): return pn.calls(st, ctx_te, hb_te, hx_te, Q, rg[st])  # noqa: E731
                d = hb_te.shape[-1]; full = calls(np.eye(d))
                res = out.setdefault(st, {}).setdefault(des, {}).setdefault(V, {})
                for k in KS:
                    if st == 'S': Q, info = E.das_fit(net, 'S', hb_tr, hx_tr, Ltr, base_tr, k, seed=10 + k, steps=steps)
                    else: Q, info = fit_pp(pn, st, ctx_tr, hb_tr, hx_tr, Ltr, base_tr, rg[st], k, seed=10 + k, steps=steps, bsize=bsize)
                    c = calls(Q); e = E.eff(c, Lte, base_te); br = boot_reps(c, Lte, base_te)
                    r = dict(train=info, eff=e, eff_ci=[float(np.quantile(br, .025)), float(np.quantile(br, .975))] if len(br) else None,
                             n_eff=int((Lte != base_te).sum()), iia_all=float((c == Lte).mean()), full_swap=E.eff(full, Lte, base_te),
                             boot=br.round(4).tolist())
                    if nrand:
                        rr = np.random.default_rng(100 + k); nv = np.array([E.eff(calls(E.rand_Q(d, k, rr)), Lte, base_te) or 0.0 for _ in range(nrand)])
                        r['random'] = dict(mean=float(nv.mean()), q=np.quantile(nv, [.025, .5, .975]).tolist(), max=float(nv.max()), n=int(len(nv)))
                        r['p_rot'] = float((1 + (nv >= e).sum()) / (1 + len(nv))) if e is not None else None
                    if trained: r['Q'] = np.asarray(Q).T.tolist()
                    res[str(k)] = r
                    log(f'  {net.name} {st} {des} {V} k{k}: eff {e} {r["eff_ci"]} (n {r["n_eff"]}); full {r["full_swap"]}; '
                        f'rand q975 {r.get("random", {}).get("q", [None] * 3)[2]}; p_rot {r.get("p_rot")}; best step {info["best_step"]}; {time.time() - t1:.0f}s; '
                        f'rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MiB')
            del ctx_tr, hb_tr, hd_tr, ctx_te, hb_te, hd_te
    return out


# ------------------------------------------------------------------------------------------------ per-particle read-out
def probe(pn, S):
    """ridge read-out of pT_i and pT_i r_i^2 from a_i and u_i on real particles (fit jets) -> test R^2 and edit calibration"""
    rng = np.random.default_rng(2555); Xf, Xt = S['fit'], S['test']
    fi = rng.choice(len(Xf), min(2000, len(Xf)), replace=False); rf = pn.reps(Xf[fi], ('a', 'u'))
    def tgt(X):  # noqa: E306
        pt = X[..., 0 if X.shape[-1] == 3 else 5].astype(np.float64)
        er, pr = (X[..., 1], X[..., 2]) if X.shape[-1] == 3 else (X[..., 8], X[..., 11])
        return pt, pt * (er.astype(np.float64) ** 2 + pr.astype(np.float64) ** 2), pt > 0
    ptf, m2f, rlf = tgt(Xf[fi]); out = {}
    p = rng.permutation(int(rlf.sum())); n1 = 3 * len(p) // 4
    for st in ('a', 'u'):
        A = rf[st][rlf].astype(np.float64); out[st] = {}
        for V, t, tpart in (('pT', ptf[rlf], 0), ('pTr2', m2f[rlf], 1)):
            m = E.RidgeR(A, t, np.sort(p[:n1]), np.sort(p[n1:]))
            b, _ = S['pairs']['test']; rb = pn.reps(Xt[b], ('a', 'u')); rd = pn.reps(S['PE']['test']['M0' if V == 'pT' else 'M2'], ('a', 'u'))
            tb = tgt(Xt[b]); td = tgt(S['PE']['test']['M0' if V == 'pT' else 'M2'])
            rl = tb[2]; Ab = rb[st][rl].astype(np.float64); Ad = rd[st][rl].astype(np.float64)
            yb, yd = tb[tpart][rl], td[tpart][rl]; pb, pd = m(Ab), m(Ad)
            r2 = 1 - ((pb - yb) ** 2).mean() / yb.var()
            dt, dp = yd - yb, pd - pb; slope = float((dp * dt).sum() / max((dt * dt).sum(), 1e-30))
            # jet-level: pooled read-out sum vs true sum change
            jb = np.zeros(len(b)); jd = np.zeros(len(b)); jt = np.zeros(len(b)); ji = np.nonzero(rl)[0]
            np.add.at(jb, ji, pb); np.add.at(jd, ji, pd); np.add.at(jt, ji, dt)
            out[st][V] = dict(R2_test_particles=float(r2), alpha=float(m.alpha), calib_slope_edit=slope,
                              jet_sum_calib=float((jd - jb).mean() / jt.mean()) if abs(jt.mean()) > 1e-12 else None,
                              jet_sum_corr=float(np.corrcoef(jd - jb, jt)[0, 1]) if jt.std() > 0 else None, n_particles_test=int(rl.sum()))
        log(f'  probe {st}: ' + '; '.join(f'{V} R2 {v["R2_test_particles"]:.3f} edit slope {v["calib_slope_edit"]:.3f} jet-sum calib {v["jet_sum_calib"]}' for V, v in out[st].items()))
    return out


# ------------------------------------------------------------------------------------------------ Arm 3
def arm3(net, S, steps, nrand):
    rf = net.run(S['fit'], ('S', 'logits')); rt = net.run(S['test'], ('S', 'logits')); lab = labels_for(net, S, rf, rt)
    btr, str_ = S['pairs']['train']; Qkeep = {}
    for V in VARS:  # the k = 1 src DAS direction at S (as in part trained, same seed)
        Qkeep[V], _ = E.das_fit(net, 'S', rf['S'][btr], rf['S'][str_], lab['train'][V], lab['train']['base'], 1, seed=11, steps=steps)
    Q2 = np.stack([Qkeep['M0'][:, 0], Qkeep['M2'][:, 0]], 1).astype(np.float64)
    If, It = S['If'], S['It']; out = dict(cos_q0_q2=float(Q2[:, 0] @ Q2[:, 1]))
    def X_(I): ok = (I['M0'] > 0) & (I['M2'] > 0); return np.stack([np.ones(len(ok)), np.log(np.maximum(I['M0'], 1e-9)), np.log(np.maximum(I['M2'], 1e-12))], 1), ok  # noqa: E301,E306
    Xm, ok = X_(If); Cf = rf['S'].astype(np.float64) @ Q2
    coef, *_ = np.linalg.lstsq(Xm[ok], Cf[ok], rcond=None); B = coef[1:].T          # rows: c0, c2; cols: logM0, logM2
    pred = Xm[ok] @ coef; r2 = 1 - ((Cf[ok] - pred) ** 2).sum(0) / ((Cf[ok] - Cf[ok].mean(0)) ** 2).sum(0)
    Xt_, okt = X_(It); Ct = rt['S'].astype(np.float64) @ Q2; pt_ = Xt_[okt] @ coef
    r2t = 1 - ((Ct[okt] - pt_) ** 2).sum(0) / ((Ct[okt] - Ct[okt].mean(0)) ** 2).sum(0)
    out['calib'] = dict(B=B, R2_fit=r2, R2_test=r2t)
    log(f'  arm3 calibration: B {np.round(B, 4).tolist()}; R2 fit {np.round(r2, 3).tolist()} test {np.round(r2t, 3).tolist()}; cos(q0,q2) {out["cos_q0_q2"]:.3f}')
    P = Q2 @ np.linalg.inv(Q2.T @ Q2)
    Dlab = {}
    for arm, I, reps in (('train', If, rf), ('test', It, rt)):
        b, s = S['pairs'][arm]; DD = 0.5 * (np.log(np.maximum(I['M0'], 1e-9)) + np.log(np.maximum(I['M2'], 1e-12))); dl = DD[s] - DD[b]
        Dlab[arm] = {}
        for nm, sgn in (('Dcoord', 1.0), ('Dfixed', -1.0)):
            dc = np.stack([dl, sgn * dl], 1) @ B.T; Sp = reps['S'][b] + dc @ P.T
            Dlab[arm][nm] = net.exact('S', E.clip(net.rg['S'], reps['S'][b], Sp.astype(np.float32))).argmax(1)
        Dlab[arm]['Dinput'] = lab[arm]['D']
    base_te = lab['test']['base']
    out['call_change_frac_test'] = {nm: float((v != base_te).mean()) for nm, v in Dlab['test'].items()}
    log(f'  arm3 call changes (test): {out["call_change_frac_test"]}')
    btr, str_ = S['pairs']['train']; bte, ste_ = S['pairs']['test']
    rfh = net.run(S['fit'], ('h4',))['h4']; rth = net.run(S['test'], ('h4',))['h4']
    H = dict(S=(rf['S'], rt['S']), h4=(rfh, rth))
    for nm in ('Dcoord', 'Dinput'):
        Ltr, Lte = Dlab['train'][nm], Dlab['test'][nm]; res = out.setdefault(nm, {})
        for k in KS:
            cs = {}
            for st in ('S', 'h4'):
                hf, ht = H[st]
                Q, info = E.das_fit(net, st, hf[btr], hf[str_], Ltr, lab['train']['base'], k, seed=20 + k, steps=steps)
                c = E.eval_Q(net, st, ht[bte], ht[ste_], Q); cs[st] = c; br = boot_reps(c, Lte, base_te)
                r = dict(train=info, eff=E.eff(c, Lte, base_te), eff_ci=[float(np.quantile(br, .025)), float(np.quantile(br, .975))] if len(br) else None,
                         n_eff=int((Lte != base_te).sum()), full_swap=E.eff(lab['test']['source'], Lte, base_te))
                if nrand:
                    rr = np.random.default_rng(300 + k); d = ht.shape[1]
                    nv = np.array([E.eff(E.eval_Q(net, st, ht[bte], ht[ste_], E.rand_Q(d, k, rr)), Lte, base_te) or 0.0 for _ in range(nrand)])
                    r['random'] = dict(mean=float(nv.mean()), q=np.quantile(nv, [.025, .5, .975]).tolist(), n=int(len(nv)))
                    r['p_rot'] = float((1 + (nv >= r['eff']).sum()) / (1 + len(nv))) if r['eff'] is not None else None
                res.setdefault(st, {})[str(k)] = r
            dd = E.boot_diff(cs['h4'], cs['S'], Lte, base_te, NB)
            # one-sided p for h4 > S from the same paired bootstrap
            rng = np.random.default_rng(9); EH = (Lte != base_te).astype(float); a_ = (cs['h4'] == Lte) * EH; b_ = (cs['S'] == Lte) * EH; v = []
            for _ in range(NB):
                i = rng.integers(0, len(Lte), len(Lte)); dn = EH[i].sum()
                if dn >= 10: v.append((a_[i].sum() - b_[i].sum()) / dn)
            v = np.asarray(v); dd['p_one_sided_h4_gt_S'] = float((1 + (v <= 0).sum()) / (1 + len(v)))
            res.setdefault('h4_minus_S', {})[str(k)] = dd
            log(f'  arm3 {nm} k{k}: S {res["S"][str(k)]["eff"]} {res["S"][str(k)]["eff_ci"]}; h4 {res["h4"][str(k)]["eff"]} {res["h4"][str(k)]["eff_ci"]}; '
                f'h4-S {dd["diff"]:+.3f} {dd["ci"]} p1 {dd["p_one_sided_h4_gt_S"]:.4f}; rand S q975 {res["S"][str(k)].get("random", {}).get("q", [None]*3)[2]} '
                f'h4 q975 {res["h4"][str(k)].get("random", {}).get("q", [None]*3)[2]}')
    return out


# ------------------------------------------------------------------------------------------------ parts
def part_trained():
    import resource
    from interp.jedi_core import Jedi
    S, R = setup(); JS = OUT / f'e25x_{TAG}_trained_{args.var}.json'
    def save(): JS.write_text(json.dumps(clean(R), indent=1))  # noqa: E306
    j = Jedi(64, FAM); net = E.Net(j, name=args.model); pn = PNet(net)
    M = R['trained'] = {}
    for nm, X, y in (('fit', S['fit'], S['yf']), ('test', S['test'], S['yt'])):
        c = net.calls(X); M[f'acc_{nm}'] = float((c == y).mean()); M[f'acc_{nm}_ci'] = wilson(int((c == y).sum()), len(y))
    ref = j.run(S['test'][:1000])['logits']; own = net.run(S['test'][:1000])['logits']; pr = pn.reps(S['test'][:1000], ('logits',))['logits']
    M['own_vs_jedi_run'] = float(np.abs(own - ref).max()); M['pnet_vs_jedi_run'] = float(np.abs(pr - ref).max())
    log(f'accuracy fit {M["acc_fit"]:.4f} {M["acc_fit_ci"]}; test {M["acc_test"]:.4f} {M["acc_test_ci"]}; |dlogit| Net {M["own_vs_jedi_run"]:.2e} PNet {M["pnet_vs_jedi_run"]:.2e}')
    save()
    if args.var == 'arm3':
        M['probe'] = probe(pn, S); save()
        M['arm3'] = arm3(net, S, STEPS, 200 if not SM else 10); save()
    else:
        M['das'] = run_net_stages(net, pn, S, M, STEPS, NRAND, True, vars_=(args.var,)); save()
    R['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024; R['wall_s'] = time.time() - T0
    save(); log('done')


def baseline_list():
    return [('untr', s) for s in range(10)] + [('init', s) for s in range(15)] + [('gauss', s) for s in range(15)] + [('perm', s) for s in range(5)]


def part_base():
    import gc, resource
    from interp.jedi_core import Jedi
    S, R = setup(); JS = OUT / f'e25x_{TAG}_base{args.shard}.json'
    def save(): JS.write_text(json.dumps(clean(R), indent=1))  # noqa: E306
    j = Jedi(64, FAM); tr = E.Net(j, name='trained'); R['nets'] = {}
    bl = baseline_list()[args.shard::args.nshard]
    if SM: bl = bl[:2]
    for kind, s in bl:
        name = f'{kind}{s}'; t1 = time.time()
        if kind == 'untr':
            ju = Jedi(64, FAM, untrained=True, seed=s); KB0 = {k: (np.asarray(v[0], np.float32), np.asarray(v[1], np.float32)) for k, v in ju.KB.items()}; del ju; gc.collect()
        else: KB0 = E.variant_KB(tr.KBn, kind, s)
        pre = E.Net(j, KB0, name=name + '_pre'); z = pre.run(S['fit'], ('z',))['z']
        KB1, nlive = E.refit_d7(pre, z, S['yf'], np.arange(len(S['yf']))); net = E.Net(j, KB1, name=name); del pre, z
        c = net.calls(S['test']); acc = float((c == S['yt']).mean()); nd = int(len(np.unique(c)))
        M = dict(acc_test=acc, acc_test_ci=wilson(int((c == S['yt']).sum()), len(c)), n_distinct_calls=nd, d7_live_inputs=nlive,
                 call_dist_test=np.bincount(c, minlength=5), qualifies=bool(acc > QACC and nd >= 3))
        log(f'{name}: test acc {acc:.4f}; distinct calls {nd}; qualifies {M["qualifies"]}')
        if M['qualifies']:
            pn = PNet(net); M['das'] = run_net_stages(net, pn, S, M, BSTEPS, 0, False, bsize=384)
            for st in M['das'].values():
                for des in st.values():
                    for V in des.values():
                        for r in V.values(): r.pop('boot', None)
        M['wall_s'] = time.time() - t1; R['nets'][name] = M; save()
        log(f'{name}: wall {M["wall_s"]:.0f}s; rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MiB')
    R['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024; R['wall_s'] = time.time() - T0
    save(); log('done')


def part_combine():
    T = {}; D = {}
    for v in ('M0', 'M2', 'arm3'):
        f = OUT / f'e25x_{TAG}_trained_{v}.json'
        if not f.exists(): continue
        t = json.loads(f.read_text())['trained']; T[v] = {k: t[k] for k in ('acc_fit', 'acc_fit_ci', 'acc_test', 'acc_test_ci', 'own_vs_jedi_run', 'pnet_vs_jedi_run', 'checks', 'cf_change_frac_test', 'n_eff_test') if k in t}
        for st, dd in t.get('das', {}).items():
            for des, vv in dd.items():
                for V, kk in vv.items(): D.setdefault(st, {}).setdefault(des, {})[V] = kk
        if v == 'arm3': T['arm3'] = t.get('arm3'); T['probe'] = t.get('probe')
    nets = {}
    for f in sorted(OUT.glob(f'e25x_{TAG}_base*.json')):
        nets.update(json.loads(f.read_text())['nets'])
    q = {n: m for n, m in nets.items() if m['qualifies'] and 'das' in m}
    R = dict(trained=T, n_baselines=len(nets), qualifying=sorted(q), n_qualifying=len(q),
             baseline_acc={n: m['acc_test'] for n, m in nets.items()}, cells={})
    pv = []
    for st, dd in D.items():
        for des, vv in dd.items():
            for V, kk in vv.items():
                for k, r in kk.items():
                    bv = np.array([m['das'][st][des][V][k]['eff'] for m in q.values() if m['das'].get(st, {}).get(des, {}).get(V, {}).get(k, {}).get('eff') is not None])
                    e = r['eff']; br = np.asarray(r['boot'])
                    c = dict(eff=e, eff_ci=r['eff_ci'], n_eff=r['n_eff'], full_swap=r['full_swap'], random_q975=r.get('random', {}).get('q', [None] * 3)[2],
                             p_rot=r.get('p_rot'), n_base=int(len(bv)))
                    if len(bv) and e is not None:
                        q95 = float(np.quantile(bv, .95)); c.update(base_q95=q95, base_max=float(bv.max()), base_median=float(np.median(bv)),
                                                                    p_q95=float((1 + (br <= q95).sum()) / (1 + len(br))),
                                                                    p_rank=float((1 + (bv >= e).sum()) / (1 + len(bv))),
                                                                    base_by_family={f: [float(m['das'][st][des][V][k]['eff']) for n, m in q.items() if n.startswith(f) and m['das'][st][des][V][k]['eff'] is not None] for f in ('untr', 'init', 'gauss', 'perm')})
                    c['Q'] = r.get('Q'); key = f'{st}|{des}|{V}|k{k}'; R['cells'][key] = c; pv.append((key, c.get('p_q95'), c.get('p_rot')))
    for i, nm in ((1, 'holm_p_q95'), (2, 'holm_p_rot')):
        h = holm([p[i] for p in pv])
        for (key, *_), hp in zip(pv, h): R['cells'][key][nm] = hp
    for key, c in R['cells'].items():
        log(f'{key}: eff {c["eff"]} {c["eff_ci"]}; base q95 {c.get("base_q95")} (n {c["n_base"]}); Holm p_q95 {c.get("holm_p_q95")}; '
            f'rand q975 {c["random_q975"]}; Holm p_rot {c.get("holm_p_rot")}; full {c["full_swap"]}')
    (OUT / f'e25x_{TAG}.json').write_text(json.dumps(clean(R), indent=1)); log('done')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    dict(trained=part_trained, base=part_base, combine=part_combine)[args.part]()
