"""E39 (claims A3/A4 retrain stability, E33x follow-up; loop A round 18, exploratory): model stitching across the 10 trained
JEDI-linear networks of E33x.  Do retrains carry the same second-round object layer S (and the first dense layer h4) up to an
affine change of basis, and whose edit responses does a stitched network show: the donor's or the recipient's?

Networks (no training): the 10 trained networks of E33x through interp.e33_program_recurrence_jedi.Model (official 3f n64,
n32, n128, perm-inv n64, 16f n64, X12 HGQ-r150 s0/s1/s2, X12 float s0/s1; hgq150_s1 at 57 % is kept as a degenerate case and
left out of pooled summaries), plus untrained redraws (jetdistill.network.redraw_untrained, seeds 0-2) of each as donors.

PLAN (fixed before the first run)
Samples: fit = jedi_core.sample(2000, 3939) (10,000 jets) for the maps; test = sample(2000, 3940) minus rows shared with fit.
  n32: first 32 slots; n128: same rows of val_ptetaphi_f16.npy (128 slots); 16f: same rows of val_c16_f16.npy (E33 loader).
Stitch: for donor A, recipient B, layer L in {S, h4}: ridge map L_B ~ (L_A - mean_A) W + mean_B fitted on the fit jets in float64
  (lambda = alpha * tr(Xc'Xc) / 64, alpha in 1e-7 .. 1e-1, chosen on a 20 % hold-out of the fit jets by standardised MSE, then
  refit on all fit jets).  Stitched network = A's layers up to L, the map, a clip of every unit to B's own [min, max] on the fit
  jets (the next HGQ input quantizer wraps out-of-range values; the clipped share is recorded), then B's exact fixed-point tail
  (B's own input quantizer of d4 for L = S, of d5 for L = h4).  90 ordered pairs x 2 layers.
Metrics on test jets: stitched accuracy; same-call % with B and with A; native A-B agreement (both unedited networks); 5-class
  confusion (truth x stitched call); on the jets where A and B call differently, the share where the stitched call equals B's
  and the share where it equals A's (run 2); map test R^2 (1 - SSE / SST over B's units).
Controls: (i) random affine map with the fitted map's singular values (Haar-random left / right orthogonal factors, rng per pair);
  (ii) untrained donor: redraw of A (seeds 0-2) with its own ridge map into B; (iii) map fitted on shuffled rows (jet i of A
  paired with jet perm(i) of B; marginals kept, jet identity removed); (iv) self-stitch A = B through the ridge map.
Statistics: 1,000-draw jet bootstrap (95 %) for accuracy, same-call and responses; McNemar (exact binomial on discordant
  correct / wrong) for stitched vs B correctness; per pair one-sided paired sign-flip permutation (5,000) of the per-jet
  same-call indicator, trained donor vs the mean over the 3 untrained donors (H0: trained-donor same-call <= untrained-donor
  same-call), Holm over the 90 pairs per layer; across pairs a sign-flip permutation (10,000) of the 90 paired differences.
Edits (8 networks with the 3-feature 64-slot input: official n64, n32 (first 32 slots of the edited jet), perminv n64,
  hgq150 s0-s2, float s0-s1; n128 and 16f excluded: their inputs carry slots / columns the edits below do not define):
  T  rigid translation: every real particle's Delta eta + 0.06 rad, inputs kept relative to the stored E-scheme axis (padding
     untouched; E6/E6v); true W jets; per-jet response 1[call Z after] - 1[call Z before]  (W -> Z flow).
  D  prong sharing (E35 = H45 edit_one), dc = +0.3: two exclusive-kT prongs, jet boosted to rest, all particles rotated so that
     |cos theta*| rises by 0.3, boosted back (jet four-momentum, prong masses, N fixed), inputs rebuilt about the E-scheme axis;
     W+Z jets with accepted prong geometry; response = change of l_W - l_Z vs the rebuilt unedited jet, on W jets (DW) and on
     Z jets (DZ) separately (run 2: run 1 pooled W and Z jets, whose responses have opposite sign (E37v), into one mean).
  N  one soft particle added: pT 1 GeV at r = 0.3 from the axis, azimuth uniform (rng 3941), inserted at its pT rank; q+g jets
     with <= 63 particles; response = change of l_g - l_q.
  K  angles (Delta eta, Delta phi) of the pT-rank 13 and 14 particles swapped (= swapping their pT, E20 A13); jets with >= 15
     particles; response = change of log p(true class) (K) and its absolute value (Kabs; run 2: K is near 0 in every network).
  r_X = mean per-jet response of network X.  For each ordered pair (A, B) of edit networks and layer L: eligible iff
  |r_A - r_B| > 2 bootstrap SE of the paired difference.  Inheritance index I = (r_stitch - r_B) / (r_A - r_B) (1 = responds
  like donor A, 0 = like recipient B).  Pooled index over eligible pairs (hgq150_s1 excluded) = sum (r_s - r_B)(r_A - r_B) /
  sum (r_A - r_B)^2 with a jet bootstrap CI; same for the untrained-donor and random-map stitches (controls); also restricted to
  accuracy-matched stitches (stitched accuracy >= B's - 2 points).
Secondary: same-call at S vs the number of the four E33x named roles (r2, soft, hard, core; R^2 >= 0.9) present in both networks
  (Spearman over the trained pairs).
Output: $AI_RESULTS/jedi/e39.json (and e39_smoke.json with --smoke).
"""
import argparse, contextlib, gc, json, math, os, sys, time
from pathlib import Path
import numpy as np

os.environ.setdefault('KERAS_BACKEND', 'jax')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
T0 = time.time()

NETS = ['official_n64', 'official_n32', 'official_n128', 'perminv_n64', 'f16_n64', 'hgq150_s0', 'hgq150_s1', 'hgq150_s2',
        'float_s0', 'float_s1']
EDIT_NETS = ['official_n64', 'official_n32', 'perminv_n64', 'hgq150_s0', 'hgq150_s1', 'hgq150_s2', 'float_s0', 'float_s1']
DEGEN = {'hgq150_s1'}
GROUP = dict(official_n64='Q3', official_n32='Q3', official_n128='Q3', hgq150_s0='Q3', hgq150_s2='Q3', hgq150_s1='degenerate',
             float_s0='F3', float_s1='F3', perminv_n64='perminv', f16_n64='16f')
LAYERS = ('S', 'h4')
EDITS = ('T', 'D', 'N', 'K')                     # edited inputs
RNAMES = ('T', 'DW', 'DZ', 'N', 'K', 'Kabs')    # responses (D split by class, K signed and absolute; run 2)
ALPHAS = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
CH = 2000   # tail batch


def log(*a): print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, np.ndarray): return clean(o.tolist())
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else round(float(o), 6)
    return o


def binom_p(k, n):
    if n == 0: return 1.0
    k = min(k, n - k)
    lp = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(k + 1)]
    m = max(lp); return float(min(1.0, 2 * math.exp(m) * sum(math.exp(v - m) for v in lp)))


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); out = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * ps[i])); out[i] = run
    return out


class Boot:
    def __init__(self, J, nb, seed):
        rng = np.random.default_rng(seed); self.J = J; self.W = np.zeros((nb, J), np.float32)
        for b in range(nb): self.W[b] = np.bincount(rng.integers(0, J, J), minlength=J)

    def means(self, v): return (self.W @ np.asarray(v, np.float32)) / self.J

    def ci(self, v):
        m = self.means(v); return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


@contextlib.contextmanager
def redrawn(seed):
    """keras.models.load_model returns a model with weights redrawn (untrained control) while active (as e33x)"""
    import keras
    from jetdistill.network import redraw_untrained
    orig = keras.models.load_model

    def f(*a, **k):
        m = orig(*a, **k); redraw_untrained(m, seed); return m
    keras.models.load_model = f
    try: yield
    finally: keras.models.load_model = orig


class Tail:
    """B's exact fixed-point tail from S (d4 d5 d6 d7) or from h4 (d5 d6 d7)"""

    def __init__(self, model):
        import jax, jax.numpy as jnp
        L = model.L; self.kd = bool(getattr(L['qs1'], 'keepdims', False))
        KB = {k: (jnp.asarray(np.asarray(model.KBnp[k][0], np.float32)), jnp.asarray(np.asarray(model.KBnp[k][1], np.float32)))
              for k in ('d4', 'd5', 'd6', 'd7')}

        def dense(k, t, act=True):
            z = jnp.einsum(L[k].equation, L[k].iq(t, training=False), KB[k][0]) + KB[k][1]
            return L[k].activation(z) if act and L[k].activation is not None else z

        def fromS(S):
            h4 = dense('d4', S); h5 = dense('d5', h4); z = dense('d6', h5, act=False)
            return dense('d7', jax.nn.relu(z)).reshape(S.shape[0], -1)

        def fromh4(h4):
            h5 = dense('d5', h4); z = dense('d6', h5, act=False)
            return dense('d7', jax.nn.relu(z)).reshape(h4.shape[0], -1)
        self.f = dict(S=jax.jit(fromS), h4=jax.jit(fromh4))

    def __call__(self, layer, X):
        out = []
        for i in range(0, len(X), CH):
            xb = np.asarray(X[i:i + CH], np.float32); nb = len(xb)
            if nb < CH: xb = np.concatenate([xb, np.repeat(xb[-1:], CH - nb, 0)])
            if self.kd: xb = xb[:, None, :]
            out.append(np.asarray(self.f[layer](xb), np.float64)[:nb])
        return np.concatenate(out)


# ------------------------------------------------------------------------------------------------ maps
def ridge(X, Y, rng):
    """affine map Y ~ (X - mx) W + my; alpha chosen on a 20 % hold-out (standardised MSE)"""
    X = np.asarray(X, np.float64); Y = np.asarray(Y, np.float64); n = len(X)
    p = rng.permutation(n); ho = p[: n // 5]; tr = p[n // 5:]
    sy = Y.std(0); sy = np.where(sy > 1e-12, sy, 1.0)

    def fit(Xa, Ya, alpha):
        mx, my = Xa.mean(0), Ya.mean(0); Xc = Xa - mx; G = Xc.T @ Xc; lam = alpha * max(np.trace(G), 1e-12) / G.shape[0]
        W = np.linalg.solve(G + lam * np.eye(G.shape[0]), Xc.T @ (Ya - my)); return W, mx, my
    errs = []
    for a in ALPHAS:
        W, mx, my = fit(X[tr], Y[tr], a); E = ((X[ho] - mx) @ W + my - Y[ho]) / sy; errs.append(float((E ** 2).mean()))
    a = ALPHAS[int(np.argmin(errs))]; W, mx, my = fit(X, Y, a)
    return dict(W=W, mx=mx, my=my, alpha=a, ho_mse=min(errs))


def apply(M, X): return (np.asarray(X, np.float64) - M['mx']) @ M['W'] + M['my']


def haar(d, rng):
    Q, R = np.linalg.qr(rng.standard_normal((d, d))); return Q * np.sign(np.diag(R))


def random_spectrum(M, rng):
    s = np.linalg.svd(M['W'], compute_uv=False); d = len(s)
    return dict(W=haar(d, rng) @ np.diag(s) @ haar(d, rng).T, mx=M['mx'], my=M['my'], alpha=None)


def r2(Yh, Y):
    return float(1 - ((Yh - Y) ** 2).sum() / max(((Y - Y.mean(0)) ** 2).sum(), 1e-12))


# ------------------------------------------------------------------------------------------------ edits
def build_edits(x, y, smoke):
    """x (J, 64, 3) test jets -> dict edit -> (rows, x_id, x_edit, info)"""
    from interp.h45_prong_sharing import edit_one, prong_offsets_ok, model_x, axis, p4
    from interp.e35_prong_sharing_route_jedi import kt2_labels
    real = x[..., 0] > 0; nn = real.sum(1); E = {}
    # T
    r = np.flatnonzero(y == 2); xe = x[r].copy(); xe[..., 1] += np.float32(0.06) * real[r]
    E['T'] = (r, x[r], xe, dict(n=len(r)))
    # N
    r = np.flatnonzero(((y == 0) | (y == 1)) & (nn <= 63)); rng = np.random.default_rng(3941); th = rng.uniform(0, 2 * np.pi, len(r))
    xe = x[r].copy(); k = nn[r]
    xe[np.arange(len(r)), k] = np.stack([np.ones(len(r)), 0.3 * np.cos(th), 0.3 * np.sin(th)], -1).astype(np.float32)
    o = np.argsort(-xe[..., 0], 1, kind='stable'); xe = np.take_along_axis(xe, o[..., None], 1)
    E['N'] = (r, x[r], xe, dict(n=len(r), sorted_ok=bool((np.diff(xe[..., 0], axis=1) <= 0).all())))
    # K
    r = np.flatnonzero(nn >= 15); xe = x[r].copy(); xe[:, [13, 14], 1:] = xe[:, [14, 13], 1:]
    E['K'] = (r, x[r], xe, dict(n=len(r)))
    # D
    cand = np.flatnonzero(((y == 2) | (y == 3)) & (nn >= 3))
    if smoke: cand = cand[:300]
    P4 = p4(x[..., 0].astype(np.float64), x[..., 1].astype(np.float64), x[..., 2].astype(np.float64), 0.0) * real[..., None]
    PE = P4.copy(); ok = np.zeros(len(x), bool); reasons = {}; dP = []; t = time.time()
    for j in cand:
        n_ = nn[j]; P = P4[j, :n_]; lab = kt2_labels(P)
        res, inf = edit_one(P, lab, 0.3, np.random.default_rng([3942, int(j)]))
        reasons[inf['reason']] = reasons.get(inf['reason'], 0) + 1
        if res is None: continue
        if not prong_offsets_ok(res['D'], lab): reasons['offset'] = reasons.get('offset', 0) + 1; continue
        PE[j, :n_] = res['D']; ok[j] = True; dP.append(float(np.abs(res['D'].sum(0) - P.sum(0)).max() / max(P.sum(0)[3], 1e-9)))
    r = np.flatnonzero(ok)
    xid = model_x(P4[r], nn[r], axis(P4[r], nn[r])); xe = model_x(PE[r], nn[r], axis(PE[r], nn[r]))
    rebuild = float(np.abs(xid - x[r]).max())
    E['D'] = (r, xid, xe, dict(n=len(r), candidates=len(cand), reasons=reasons, max_rel_dP=max(dP) if dP else None,
                               rebuild_vs_stored_max=rebuild, seconds=time.time() - t))
    log('edits: ' + ', '.join(f'{k} n {v[3]["n"]}' for k, v in E.items()) + f'; D reasons {reasons}, max rel dP4 {E["D"][3]["max_rel_dP"]}, '
        f'rebuild vs stored max |dx| {rebuild:.3g}')
    return E


def resp(e, lo_id, lo_ed, y):
    """edit e -> {response name: per-jet response}"""
    if e == 'T': return dict(T=(lo_ed.argmax(1) == 3).astype(np.float64) - (lo_id.argmax(1) == 3))
    if e == 'D':
        d = (lo_ed[:, 2] - lo_ed[:, 3]) - (lo_id[:, 2] - lo_id[:, 3]); return dict(DW=d[y == 2], DZ=d[y == 3])
    if e == 'N': return dict(N=(lo_ed[:, 0] - lo_ed[:, 1]) - (lo_id[:, 0] - lo_id[:, 1]))
    lp = lambda L: L - L.max(1, keepdims=True) - np.log(np.exp(L - L.max(1, keepdims=True)).sum(1, keepdims=True))  # noqa: E731
    a, b = lp(lo_ed), lp(lo_id); i = np.arange(len(y)); k = a[i, y] - b[i, y]; return dict(K=k, Kabs=np.abs(k))


def to_net(name, x):
    return x[:, :32] if name == 'official_n32' else x


# ------------------------------------------------------------------------------------------------ main
def main(a):
    import interp.e33_program_recurrence_jedi as E33
    from jetdistill.config import RESULTS
    smoke = a.smoke; per = 200 if smoke else 2000; NB = 200 if smoke else 1000; NPERM = 200 if smoke else 5000
    OUT.mkdir(parents=True, exist_ok=True); JS = OUT / ('e39_smoke.json' if smoke else 'e39.json')
    R = dict(node='E39', plan='module docstring (fixed before the first run)', smoke=smoke, nets=NETS, edit_nets=EDIT_NETS, group=GROUP)
    save = lambda: JS.write_text(json.dumps(clean(R)))  # noqa: E731
    nets = NETS if not smoke else ['official_n64', 'official_n32', 'f16_n64', 'hgq150_s0']
    enets = [n for n in EDIT_NETS if n in nets]
    useeds = (0, 1, 2) if not smoke else (0,)

    # ---- samples
    yall = np.asarray(np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r')['y'])
    idf, idt = E33.rows(yall, per, 3939), E33.rows(yall, per, 3940); keep = ~np.isin(idt, idf)
    R['sample'] = dict(fit=f'sample({per}, 3939)', test=f'sample({per}, 3940) minus rows in fit', N_fit=len(idf), N_test=int(keep.sum()),
                       overlap_removed=int((~keep).sum()))
    cache = {}

    def data(n, f16):
        if (n, f16) not in cache:
            xf, _, x3f, yf, _ = E33.load(per, 3939, n, f16); xt, _, x3t, yt, _ = E33.load(per, 3940, n, f16)
            cache[(n, f16)] = (xf, yf, xt[keep], yt[keep], x3t[keep])
        return cache[(n, f16)]
    _, yf, _, yt, x3t = data(64, False)
    J = len(yt); log(f'samples: fit {len(yf)}, test {J} (overlap removed {int((~keep).sum())})')
    EDT = build_edits(x3t, yt, smoke); R['edits'] = {k: v[3] for k, v in EDT.items()}
    BT = Boot(J, NB, 3943); yD = yt[EDT['D'][0]]; NR = dict(T=len(EDT['T'][0]), DW=int((yD == 2).sum()), DZ=int((yD == 3).sum()), N=len(EDT['N'][0]), K=len(EDT['K'][0]), Kabs=len(EDT['K'][0]))
    BE = {e: Boot(NR[e], NB, 3944 + i) for i, e in enumerate(RNAMES)}

    # ---- trained networks: stages on fit / test / edits; tails
    ST, TAIL, CALL, LIM, RESP, INFO = {}, {}, {}, {}, {}, {}
    for nm in nets:
        M = E33.Model(nm); xf, _, xt, _, _ = data(M.n, M.f16)
        Rf = M.run(xf, keep=('S', 'h4', 'logits')); Rt = M.run(xt, keep=('S', 'h4', 'logits'))
        ST[nm] = {('fit', L): Rf[L].astype(np.float32) for L in LAYERS}; ST[nm].update({('test', L): Rt[L].astype(np.float32) for L in LAYERS})
        LIM[nm] = {L: (Rf[L].min(0), Rf[L].max(0)) for L in LAYERS}
        TAIL[nm] = Tail(M); CALL[nm] = Rt['logits'].argmax(1)
        chk = {L: float(np.abs(TAIL[nm](L, Rt[L]) - Rt['logits']).max()) for L in LAYERS}
        acc = float((CALL[nm] == yt).mean())
        INFO[nm] = dict(acc=acc, acc_ci=BT.ci(CALL[nm] == yt), tail_check_maxabs=chk, path=M.path, group=GROUP[nm])
        if nm in enets:
            RESP[nm] = {}
            for e in EDITS:
                r, xid, xe, _ = EDT[e]
                Ri = M.run(to_net(nm, xid), keep=('S', 'h4', 'logits')); Re = M.run(to_net(nm, xe), keep=('S', 'h4', 'logits'))
                for L in LAYERS: ST[nm][(e + '_id', L)] = Ri[L].astype(np.float32); ST[nm][(e + '_ed', L)] = Re[L].astype(np.float32)
                RESP[nm].update(resp(e, Ri['logits'], Re['logits'], yt[r]))
            INFO[nm]['resp'] = {e: [float(RESP[nm][e].mean()), BE[e].ci(RESP[nm][e])] for e in RNAMES}
        log(f'{nm}: acc {acc:.4f} {INFO[nm]["acc_ci"]}; tail check {chk}' + (f'; resp ' + json.dumps(clean(INFO[nm].get("resp"))) if nm in enets else ''))
        del M; gc.collect()
    R['nets_info'] = INFO; save()
    if not smoke and not (0.806 <= INFO['official_n64']['acc'] <= 0.813): log('WARNING: official_n64 accuracy outside 80.6-81.3 %')

    native = {(A, B): float((CALL[A] == CALL[B]).mean()) for A in nets for B in nets}
    R['native_agreement'] = {f'{A}|{B}': v for (A, B), v in native.items()}

    def stitch(A_st, B, L, M, edits_from=None):
        """A_st: dict of donor stage arrays; returns test logits and (optionally) edit responses through B's tail"""
        lo_, hi_ = LIM[B][L]; Yh = apply(M, A_st[('test', L)]); clipped = float(((Yh < lo_) | (Yh > hi_)).mean())
        lo = TAIL[B](L, np.clip(Yh, lo_, hi_)); out = dict(lo=lo, clipped=clipped)
        if edits_from:
            out['resp'] = {}
            for e in EDITS:
                r = EDT[e][0]
                li = TAIL[B](L, np.clip(apply(M, A_st[(e + '_id', L)]), lo_, hi_)); le = TAIL[B](L, np.clip(apply(M, A_st[(e + '_ed', L)]), lo_, hi_))
                out['resp'].update(resp(e, li, le, yt[r]))
        return out

    def metrics(lo, A, B):
        c = lo.argmax(1); cf = np.zeros((5, 5), int); np.add.at(cf, (yt, c), 1)
        sB = c == CALL[B]; sA = (c == CALL[A]) if A in CALL else None
        okB, okS = CALL[B] == yt, c == yt
        n10, n01 = int((okB & ~okS).sum()), int((~okB & okS).sum())
        dis = (CALL[A] != CALL[B]) if A in CALL else np.zeros(len(c), bool)
        disagree = dict(n=int(dis.sum()), follows_B=float((c[dis] == CALL[B][dis]).mean()) if dis.any() else None,
                        follows_A=float((c[dis] == CALL[A][dis]).mean()) if dis.any() else None)
        return dict(disagree=disagree, acc=float(okS.mean()), acc_ci=BT.ci(okS), same_B=float(sB.mean()), same_B_ci=BT.ci(sB),
                    same_A=None if sA is None else float(sA.mean()), same_A_ci=None if sA is None else BT.ci(sA),
                    confusion=cf, mcnemar=dict(B_only_correct=n10, stitch_only_correct=n01, p=binom_p(n10, n10 + n01))), sB

    # ---- trained pairs + controls (i), (iii), (iv)
    PAIRS = {L: {} for L in LAYERS}; SAME = {L: {} for L in LAYERS}; SRESP = {L: {} for L in LAYERS}; RRESP = {L: {} for L in LAYERS}
    for L in LAYERS:
        for A in nets:
            for B in nets:
                rng = np.random.default_rng([3945, nets.index(A), nets.index(B), LAYERS.index(L)])
                M = ridge(ST[A][('fit', L)], ST[B][('fit', L)], rng)
                ed = A in enets and B in enets and A != B
                o = stitch(ST[A], B, L, M, edits_from=ed)
                m, sB = metrics(o['lo'], A, B); SAME[L][(A, B)] = sB
                Yt = apply(M, ST[A][('test', L)])
                m.update(alpha=M['alpha'], ho_mse=M['ho_mse'], map_r2_test=r2(Yt, ST[B][('test', L)].astype(np.float64)), clipped=o['clipped'],
                         acc_unclipped=float((TAIL[B](L, Yt).argmax(1) == yt).mean()), native_agreement=native[(A, B)])
                if ed:
                    SRESP[L][(A, B)] = o['resp']
                if A != B:
                    Mr = random_spectrum(M, rng); orr = stitch(ST[A], B, L, Mr, edits_from=ed)
                    mr, _ = metrics(orr['lo'], A, B); m['random_map'] = dict(acc=mr['acc'], same_B=mr['same_B'], same_B_ci=mr['same_B_ci'], clipped=orr['clipped'])
                    if ed: RRESP[L][(A, B)] = orr['resp']
                    perm = rng.permutation(len(yf)); Ms = ridge(ST[A][('fit', L)][perm], ST[B][('fit', L)], rng)
                    os_ = stitch(ST[A], B, L, Ms); ms, _ = metrics(os_['lo'], A, B)
                    m['shuffled_rows'] = dict(acc=ms['acc'], same_B=ms['same_B'], same_B_ci=ms['same_B_ci'], alpha=Ms['alpha'])
                PAIRS[L][(A, B)] = m
            log(f'[{L}] donor {A}: same_B ' + ', '.join(f'{B} {PAIRS[L][(A, B)]["same_B"]:.3f}' for B in nets)
                + ' | random ' + ', '.join(f'{PAIRS[L][(A, B)]["random_map"]["same_B"]:.2f}' for B in nets if B != A)
                + ' | shuffled ' + ', '.join(f'{PAIRS[L][(A, B)]["shuffled_rows"]["same_B"]:.2f}' for B in nets if B != A))
    R['pairs'] = {L: {f'{A}|{B}': v for (A, B), v in PAIRS[L].items()} for L in LAYERS}; save()

    # ---- control (ii): untrained donors
    UN = {L: {} for L in LAYERS}; USAME = {L: {} for L in LAYERS}; URESP = {L: {} for L in LAYERS}; UACC = {}
    for A in nets:
        for sd in useeds:
            with redrawn(sd):
                M_ = E33.Model(A)
            xf, _, xt, _, _ = data(M_.n, M_.f16)
            Rf = M_.run(xf, keep=('S', 'h4')); Rt = M_.run(xt, keep=('S', 'h4', 'logits'))
            Ast = {('fit', L): Rf[L].astype(np.float32) for L in LAYERS}; Ast.update({('test', L): Rt[L].astype(np.float32) for L in LAYERS})
            UACC[f'{A}|u{sd}'] = float((Rt['logits'].argmax(1) == yt).mean())
            if A in enets:
                for e in EDITS:
                    r, xid, xe, _ = EDT[e]
                    Ri = M_.run(to_net(A, xid), keep=('S', 'h4')); Re = M_.run(to_net(A, xe), keep=('S', 'h4'))
                    for L in LAYERS: Ast[(e + '_id', L)] = Ri[L].astype(np.float32); Ast[(e + '_ed', L)] = Re[L].astype(np.float32)
            del M_; gc.collect()
            for L in LAYERS:
                for B in nets:
                    if B == A: continue
                    rng = np.random.default_rng([3946, nets.index(A), nets.index(B), LAYERS.index(L), sd])
                    M = ridge(Ast[('fit', L)], ST[B][('fit', L)], rng); ed = A in enets and B in enets
                    o = stitch(Ast, B, L, M, edits_from=ed); c = o['lo'].argmax(1); sB = (c == CALL[B]).astype(np.float32)
                    USAME[L][(A, B)] = USAME[L].get((A, B), 0) + sB / len(useeds)
                    UN[L].setdefault((A, B), []).append(dict(seed=sd, acc=float((c == yt).mean()), same_B=float(sB.mean()), clipped=o['clipped'],
                                                             map_r2_test=r2(apply(M, Ast[('test', L)]), ST[B][('test', L)].astype(np.float64))))
                    if ed:
                        d = URESP[L].setdefault((A, B), {e: 0.0 for e in RNAMES})
                        for e in RNAMES: d[e] = d[e] + o['resp'][e] / len(useeds)
        log(f'untrained donor {A}: own acc ' + ', '.join(f'{UACC[f"{A}|u{s}"]:.3f}' for s in useeds) + '; same_B at S '
            + ', '.join(f'{B} {np.mean([u["same_B"] for u in UN["S"][(A, B)]]):.3f}' for B in nets if B != A))
    R['untrained_acc'] = UACC
    R['untrained_pairs'] = {L: {f'{A}|{B}': v for (A, B), v in UN[L].items()} for L in LAYERS}

    # ---- tests: trained vs untrained same-call (per pair sign-flip, Holm), across pairs
    rngp = np.random.default_rng(3947); R['tests'] = {}
    for L in LAYERS:
        keys = [(A, B) for A in nets for B in nets if A != B]; ps, rows = [], []
        for k in keys:
            d = SAME[L][k].astype(np.float32) - USAME[L][k]; obs = float(d.mean()); cnt = 0
            for c0 in range(0, NPERM, 200):
                sg = rngp.choice(np.array([-1.0, 1.0], np.float32), (min(200, NPERM - c0), len(d)))
                cnt += int(((sg @ d) / len(d) >= obs).sum())
            p = (1 + cnt) / (1 + NPERM); se = float(d.std() / np.sqrt(len(d)))
            ps.append(p); rows.append(dict(pair=f'{k[0]}|{k[1]}', diff=obs, diff_ci=BT.ci(d), z=obs / se if se > 0 else None, p_perm=p,
                                           untrained_same_B=float(USAME[L][k].mean())))
        hp = holm(ps); hm = holm([PAIRS[L][k]['mcnemar']['p'] for k in keys])
        for i, rw in enumerate(rows): rw['p_holm'] = float(hp[i]); rw['mcnemar_p_holm'] = float(hm[i])
        tr = np.array([PAIRS[L][k]['same_B'] for k in keys]); un = np.array([float(USAME[L][k].mean()) for k in keys]); dd = tr - un
        sg = rngp.choice([-1.0, 1.0], (10000, len(dd))); pn = float((1 + ((sg @ dd) / len(dd) >= dd.mean()).sum()) / 10001)
        nd = [i for i, k in enumerate(keys) if k[0] not in DEGEN and k[1] not in DEGEN]
        grp = {}
        for i, k in enumerate(keys):
            g = f'{GROUP[k[0]]}>{GROUP[k[1]]}'; grp.setdefault(g, []).append((tr[i], un[i], native[k], PAIRS[L][k]['acc'] - INFO[k[1]]['acc']))
        R['tests'][L] = dict(per_pair=rows, n_pairs=len(keys), n_holm_lt_0p05=int((hp < 0.05).sum()),
                             mcnemar_holm_lt_0p05=int((hm < 0.05).sum()),
                             trained_same_B_mean=float(tr.mean()), untrained_same_B_mean=float(un.mean()), across_pairs_p=pn,
                             trained_same_B_mean_nondegenerate=float(tr[nd].mean()), untrained_same_B_mean_nondegenerate=float(un[nd].mean()),
                             groups={g: dict(n=len(v), trained_same_B=float(np.mean([a[0] for a in v])), untrained_same_B=float(np.mean([a[1] for a in v])),
                                             native_agreement=float(np.mean([a[2] for a in v])), acc_minus_B=float(np.mean([a[3] for a in v])))
                                     for g, v in grp.items()})
        log(f'[{L}] trained same_B mean {tr.mean():.4f} vs untrained {un.mean():.4f}; across-pairs p {pn:.2g}; per-pair Holm < 0.05: '
            f'{int((hp < 0.05).sum())}/{len(keys)}; McNemar Holm < 0.05: {int((hm < 0.05).sum())}; groups ' + json.dumps(clean(R['tests'][L]['groups'])))
    save()

    # ---- inheritance index
    R['inheritance'] = {}
    for L in LAYERS:
        R['inheritance'][L] = {}
        for e in RNAMES:
            Bx = BE[e]; rows = []; elig = []
            for A in enets:
                for B in enets:
                    if A == B: continue
                    rA, rB, rS = RESP[A][e], RESP[B][e], SRESP[L][(A, B)][e]
                    dm = Bx.means(rA - rB); se = float(dm.std()); diff = float((rA - rB).mean())
                    ok = abs(diff) > 2 * se
                    I = float((rS.mean() - rB.mean()) / diff) if diff != 0 else None
                    Iu = float((URESP[L][(A, B)][e].mean() - rB.mean()) / diff) if diff != 0 else None
                    Ir = float((RRESP[L][(A, B)][e].mean() - rB.mean()) / diff) if diff != 0 else None
                    accm = PAIRS[L][(A, B)]['acc'] >= INFO[B]['acc'] - 0.02
                    rows.append(dict(pair=f'{A}|{B}', r_A=float(rA.mean()), r_B=float(rB.mean()), r_stitch=float(rS.mean()),
                                     r_stitch_ci=Bx.ci(rS), r_untrained_stitch=float(URESP[L][(A, B)][e].mean()), r_random_stitch=float(RRESP[L][(A, B)][e].mean()),
                                     diff_se=se, eligible=ok, I=I, I_untrained=Iu, I_random=Ir, acc_matched=bool(accm)))
                    if ok and A not in DEGEN and B not in DEGEN: elig.append((A, B, accm))

            def pooled(sel, which):
                if not sel: return None, None
                bA = np.stack([Bx.means(RESP[A][e]) for A, B, _ in sel]); bB = np.stack([Bx.means(RESP[B][e]) for A, B, _ in sel])
                src = {'stitch': SRESP[L], 'untrained': URESP[L], 'random': RRESP[L]}[which]
                bS = np.stack([Bx.means(src[(A, B)][e]) for A, B, _ in sel])
                pA = np.array([RESP[A][e].mean() for A, B, _ in sel]); pB = np.array([RESP[B][e].mean() for A, B, _ in sel])
                pS = np.array([src[(A, B)][e].mean() for A, B, _ in sel])
                est = float(((pS - pB) * (pA - pB)).sum() / ((pA - pB) ** 2).sum())
                bs = ((bS - bB) * (bA - bB)).sum(0) / ((bA - bB) ** 2).sum(0)
                return est, [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
            out = dict(rows=rows, n_pairs=len(rows), n_eligible=len(elig))
            for which in ('stitch', 'untrained', 'random'):
                v, ci = pooled(elig, which); out[f'pooled_{which}'] = v; out[f'pooled_{which}_ci'] = ci
            v, ci = pooled([t for t in elig if t[2]], 'stitch'); out['pooled_stitch_accmatched'] = v; out['pooled_stitch_accmatched_ci'] = ci
            out['n_eligible_accmatched'] = sum(t[2] for t in elig)
            Is = [r_['I'] for r_ in rows if r_['eligible'] and r_['I'] is not None and r_['pair'].split('|')[0] not in DEGEN and r_['pair'].split('|')[1] not in DEGEN]
            out['median_I'] = float(np.median(Is)) if Is else None
            # non-eligible pairs: is the common response kept?  r_stitch / r_B
            ne = [r_ for r_ in rows if not r_['eligible'] and abs(r_['r_B']) > 0 and r_['pair'].split('|')[0] not in DEGEN and r_['pair'].split('|')[1] not in DEGEN]
            out['common_pairs_ratio_stitch_over_B_median'] = float(np.median([r_['r_stitch'] / r_['r_B'] for r_ in ne])) if ne else None
            out['common_pairs_ratio_untrained_over_B_median'] = float(np.median([r_['r_untrained_stitch'] / r_['r_B'] for r_ in ne])) if ne else None
            out['n_common'] = len(ne)
            R['inheritance'][L][e] = out
            log(f'[{L}] edit {e}: eligible {len(elig)}/{len(rows)}; pooled I stitch {out["pooled_stitch"]} {out["pooled_stitch_ci"]}, acc-matched '
                f'{out["pooled_stitch_accmatched"]} (n {out["n_eligible_accmatched"]}); untrained {out["pooled_untrained"]} {out["pooled_untrained_ci"]}; random '
                f'{out["pooled_random"]}; median I {out["median_I"]}; common pairs r_s/r_B {out["common_pairs_ratio_stitch_over_B_median"]} (untrained '
                f'{out["common_pairs_ratio_untrained_over_B_median"]}, n {len(ne)})')
    save()

    # ---- secondary: E33x named-role recurrence per pair
    try:
        ex = json.loads((OUT / 'e33x.json').read_text())['roles_units']
        named = {m: {r_ for r_, v in ex[m].items() if v[2] is not None and v[2] >= 0.9} for m in nets if m in ex}
        from scipy.stats import spearmanr
        sec = {}
        for L in LAYERS:
            ks = [(A, B) for A in nets for B in nets if A != B and A in named and B in named and A not in DEGEN and B not in DEGEN]
            cnt = [len(named[A] & named[B]) for A, B in ks]; sm = [PAIRS[L][k]['same_B'] for k in ks]
            rho = spearmanr(cnt, sm); sec[L] = dict(n=len(ks), spearman=float(rho.statistic), p=float(rho.pvalue),
                                                    by_count={int(c): float(np.mean([s for c_, s in zip(cnt, sm) if c_ == c])) for c in sorted(set(cnt))})
        R['secondary_recurrence'] = dict(named={m: sorted(v) for m, v in named.items()}, **sec); log('secondary: ' + json.dumps(clean(sec)))
    except Exception as err:  # noqa: BLE001
        R['secondary_recurrence'] = dict(error=repr(err)); log(f'secondary failed: {err!r}')
    save(); log(f'wrote {JS}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); main(ap.parse_args())
