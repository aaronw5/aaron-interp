"""R2 (rigour pass 2): hypothesis tests and 95 % intervals for the JEDI-linear claims A1 (units that matter; float + masks),
A2 (context), A3 (rotation), A9 (layers), B2/B3 (the if-statement program) and the small-model accuracies.

Fresh jets: interp.jedi_core.sample(per_class, seed=61), a class-balanced draw from the 260,000-jet test file. For B2/B3 the
jets that the E11 fit/dev sample (seed 0, 6,000/class), the E11 test sample (seed 11) and the E11v fresh sample (seed 31) also
drew are removed. Nothing is refit except the A9 ridge probes (fitted on a random half of the seed-61 jets and scored on the
other half, as E10). Overlaps of each sample with the earlier samples are recorded.
Intervals are 95 %. Bootstrap = resampling of jets within each true class (class counts fixed), 2,000 replicates unless stated;
Wilson intervals for proportions; McNemar = exact two-sided binomial test on the discordant pairs of two paired binary
outcomes; Holm correction within each stated family. The network is deterministic: an edit that changes k >= 1 calls rejects
"this edit changes no call" with p = 0, so the tests with content are against stated thresholds or against control edits.

Parts (python -m interp.r2_stats --part <x> [--smoke]; writes $AI_RESULTS/stats/r2_<x>.json; the log is stdout):
  a1units  every unit of every stage whose replacement by its mean changes some slot value: mean-ablation cost (fraction of
           calls changed), bootstrap CI, one-sided binomial tests of H0: cost <= 1 % (and <= 0.1 %) vs H1: cost > threshold
           and of H0: cost >= threshold vs H1: cost < threshold, Holm over all tested units; resample ablation (the unit's
           value taken from another jet, one fixed derangement of the jets) as the on-distribution alternative edit, McNemar
           of mean vs resample ablation per unit. Units whose mean replacement is the identity on every slot: cost 0 by
           construction (two per stage are run to confirm).
  a1float  fixed point vs float + zero-bit masks (E3m) vs fully float: same-call % (Wilson), accuracy difference (paired
           bootstrap CI, McNemar).
  a2       g -> mean / 0 / another jet's g; single g units -> mean; S_c -> its mean-context value singly, for S0, S30, S31,
           S58 together, for every 4-subset of the other live S units (permutation null) and of all live S units.
  a3       rigid rotations by 45/90/180 deg, the two reflections, per-particle azimuth scramble: calls changed (Wilson); null =
           the same jets rerun unedited; matched-spread controls (every live S unit shifted by Gaussian noise with the edit's
           per-unit rms, or by its own edit shift taken from another jet); Cochran's Q across the five r-preserving edits;
           per-object rms shift / SD with bootstrap CI (10,000 replicates), tests vs 0.1 SD, Holm over the live objects.
  a9       E10's ridge read-outs refit on the seed-61 jets; Spearman rho between depth and first stage (R^2 >= 0.9 / 0.7);
           bootstrap over observables (10,000), permutation of the depth labels (10,000), bootstrap over held-out jets
           (probes fixed; 1,000), both levels together (2,000).
  b2       E11's stored programs on the fresh jets: same call (Wilson); accuracy program - network (paired bootstrap,
           McNemar); rejection at 50 % signal efficiency program / network (2,000 plain bootstrap resamples, as E11v);
           nested program vs direct formula same call (McNemar).
  b3       E11v's edits (all particles shifted by 0.06 rad in eta, padding fixed; pT x 0.8) on 2,000 fresh jets per class,
           physics inputs recomputed from the edited particles: same call with the network per program, nested program vs
           direct formula (McNemar), class flows (Wilson; McNemar vs the network).
  small    test-set (260,000 jets) accuracy of every N = 64 small tagger (test_probs.npy) and of JEDI-linear (jedi_core
           forward), bootstrap CI (2,000 plain resamples, the same jets for every model), paired differences and McNemar
           vs JEDI-linear and between seeds.
  table    every r2_*.json -> $AI_RESULTS/site/T14_claim_stats.png
--smoke: small samples and few replicates, writes r2_<x>_smoke.json (code test, not a result).
"""
import argparse, itertools, json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RES = Path(os.environ.get('AI_RESULTS', 'results')); OUT = RES / 'stats'; SITE = RES / 'site'
SMALL = Path(os.environ.get('AI_SMALL_MODELS', '/j-jepa-vol/aaron-interp/models/small'))
SEED = 61; CL = ['g', 'q', 'W', 'Z', 't']; Z95 = 1.959963984540054
T0 = time.time(); SMOKE = False; NBOOT = 2000
PRIOR = {'E1 (2000/class, seed 0)': (2000, 0), 'E4/E10 (4000/class, seed 0)': (4000, 0), 'E11 fit+dev (6000/class, seed 0)': (6000, 0),
         'E5 causal (2000/class, seed 1)': (2000, 1), 'E6 (3000/class, seed 6)': (3000, 6), 'E5v (2000/class, seed 7)': (2000, 7),
         'E11 test (2000/class, seed 11)': (2000, 11), 'E6v (2000/class, seed 17)': (2000, 17), 'E11v fresh (4000/class, seed 31)': (4000, 31)}


def log(*a):
    print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


def jsonable(v):
    if isinstance(v, dict): return {str(k): jsonable(w) for k, w in v.items()}
    if isinstance(v, (list, tuple)): return [jsonable(w) for w in v]
    if isinstance(v, np.ndarray): return jsonable(v.tolist())
    if isinstance(v, (bool, np.bool_)): return bool(v)
    if isinstance(v, (int, np.integer)): return int(v)
    if isinstance(v, (float, np.floating)): return float(v) if np.isfinite(v) else None
    return v


def save(name, res):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f'r2_{name}{"_smoke" if SMOKE else ""}.json'
    p.write_text(json.dumps(jsonable(res), indent=1, ensure_ascii=False)); log('wrote', p)


# ============================================================================================================ statistics
def wilson(k, n):
    if n == 0: return [None, None]
    p = k / n; z2 = Z95 ** 2; d = 1 + z2 / n; c = (p + z2 / (2 * n)) / d
    h = Z95 * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / d
    return [float(max(0.0, c - h)), float(min(1.0, c + h))]


def prop(k, n):
    k, n = int(k), int(n)
    return dict(k=k, n=n, est=k / n if n else None, ci=wilson(k, n))


def mcnemar(b, c):
    """exact two-sided McNemar: binomial test of b vs c discordant pairs at 1/2"""
    from scipy.stats import binom
    n = int(b + c)
    return 1.0 if n == 0 else float(min(1.0, 2 * binom.cdf(min(b, c), n, 0.5)))


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def binom_gt(k, n, p0):
    """P(X >= k | n, p0): one-sided p for H1 rate > p0"""
    from scipy.stats import binom
    return float(binom.sf(int(k) - 1, int(n), p0))


def binom_lt(k, n, p0):
    """P(X <= k | n, p0): one-sided p for H1 rate < p0"""
    from scipy.stats import binom
    return float(binom.cdf(int(k), int(n), p0))


def ci(v):
    v = np.asarray(v, float)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


class Boot:
    """bootstrap over jets: replicate weights (multiplicities), regenerated identically from the seed for every call, so all
    statistics of one part use the same replicates (paired). strat=True resamples inside each true class."""

    def __init__(self, y, B=None, seed=SEED, strat=True):
        self.y = np.asarray(y); self.B = B or NBOOT; self.seed = seed
        self.groups = [np.flatnonzero(self.y == c) for c in np.unique(self.y)] if strat else [np.arange(len(self.y))]

    def chunks(self, size=200):
        rng = np.random.default_rng(self.seed); J = len(self.y); done = 0
        while done < self.B:
            b = min(size, self.B - done); W = np.zeros((b, J), np.float32)
            for g in self.groups:
                idx = rng.integers(0, len(g), (b, len(g)))
                for i in range(b): W[i, g] = np.bincount(idx[i], minlength=len(g))
            done += b
            yield W

    def means(self, M):
        """M (k, J) or (J,) per-jet values -> (k, B) replicate means"""
        M = np.asarray(M, np.float32); M = M[None] if M.ndim == 1 else M; J = M.shape[1]
        return np.concatenate([(M @ W.T) / J for W in self.chunks()], 1).astype(np.float64)


def paired_acc(ca, cb, y, boot):
    """accuracy(a) - accuracy(b) on the same jets: estimate, bootstrap CI, exact McNemar"""
    ra, rb = ca == y, cb == y; b = int((ra & ~rb).sum()); c = int((~ra & rb).sum()); J = len(y)
    d = boot.means(ra.astype(np.float32) - rb.astype(np.float32))[0]
    return dict(acc_a=prop(ra.sum(), J), acc_b=prop(rb.sum(), J), diff=(b - c) / J, ci=ci(d), a_right_b_wrong=b, a_wrong_b_right=c,
                p_mcnemar=mcnemar(b, c))


def paired_binary(ea, eb, boot):
    """two paired binary outcomes (e.g. 'agrees with the network' for two programs): rates, difference a - b, CI, McNemar"""
    ea, eb = np.asarray(ea, bool), np.asarray(eb, bool); J = len(ea)
    b = int((ea & ~eb).sum()); c = int((~ea & eb).sum())
    d = boot.means(ea.astype(np.float32) - eb.astype(np.float32))[0]
    return dict(a=prop(ea.sum(), J), b=prop(eb.sum(), J), diff=(b - c) / J, ci=ci(d), a_only=b, b_only=c, p_mcnemar=mcnemar(b, c))


def cochran_q(X):
    """Cochran's Q for k paired binary outcomes, X (J, k) bool: H0 all k rates equal"""
    from scipy.stats import chi2
    X = np.asarray(X, float); k = X.shape[1]; C = X.sum(0); R = X.sum(1); N = X.sum()
    den = k * N - (R ** 2).sum()
    if den <= 0: return dict(Q=0.0, df=k - 1, p=1.0)
    Q = (k - 1) * (k * (C ** 2).sum() - N ** 2) / den
    return dict(Q=float(Q), df=k - 1, p=float(chi2.sf(Q, k - 1)))


def derangement(J, rng):
    o = rng.permutation(J); p = np.empty(J, int); p[o] = o[np.roll(np.arange(J), 1)]
    return p


# ============================================================================================================ data / network
def indices(per_class, seed):
    """the test-file rows interp.jedi_core.sample(per_class, seed) draws (same rng calls)"""
    from jetdistill.config import RESULTS
    y = np.asarray(np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r')['y']); rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), per_class, replace=False) for c in range(5)]))


def fresh(per):
    from interp.jedi_core import sample
    x, y, Q = sample(per, SEED); idx = indices(per, SEED)
    ov = {k: int(np.isin(idx, indices(*v)).sum()) for k, v in PRIOR.items()}
    info = dict(sample=f'jedi_core.sample({per}, seed={SEED})', n=int(len(y)), per_class=[int((y == k).sum()) for k in range(5)],
                overlap_with_earlier_samples=ov)
    log(f'sample {per}/class seed {SEED}: {len(y)} jets; overlaps {ov}')
    return x, y, Q, idx, info


def make_tail(net):
    """the network from S on (d4, d5, d6, ReLU, d7, with their quantizers), jitted: S (J, 64) -> logits; checked against the
    full forward pass by the callers"""
    import jax, jax.numpy as jnp
    L = net.L; KB = {}
    for k in ('d4', 'd5', 'd6', 'd7'):
        K, b = L[k].get_fused_qkernel_and_qbias(False, L[k].moving_mean, L[k].moving_variance); KB[k] = (jnp.asarray(K), jnp.asarray(b))
    keep3 = bool(getattr(L['qs1'], 'keepdims', False))

    def dense(k, t, act=True):
        z = jnp.einsum(L[k].equation, L[k].iq(t, training=False), KB[k][0]) + KB[k][1]
        return L[k].activation(z) if act and L[k].activation is not None else z

    def f(S):
        J = S.shape[0]
        if keep3: S = S[:, None, :]
        z = dense('d6', dense('d5', dense('d4', S)), act=False)
        return dense('d7', jax.nn.relu(z)).reshape(J, -1)
    fj = jax.jit(f)

    def run(S):
        return np.asarray(fj(jnp.asarray(np.asarray(S, np.float32))), np.float64)
    return run


def check_tail(tail, S, logits):
    d = float(np.abs(tail(S) - logits).max()); same = float((tail(S).argmax(1) == logits.argmax(1)).mean())
    log(f'tail check: max |logit diff| {d:.2e}, same call {same:.6f}')
    assert same == 1.0 and d < 1e-4, d
    return dict(max_abs_logit_diff=d, same_call=same)


# ============================================================================================================ A1 units
def part_a1units():
    from interp.jedi_core import Jedi, STAGES, WIDTH
    per = 200 if SMOKE else 4000
    x, y, Q, idx, info = fresh(per); del Q
    J = len(y); real = x[..., 0] > 0; net = Jedi(64); boot = Boot(y)
    call = net.run(x)['logits'].argmax(1)
    rng = np.random.default_rng(SEED); partner = derangement(J, rng)
    res = dict(info=info, acc=prop((call == y).sum(), J), partner='one fixed derangement of the jets (rng 61)', stages={}, units=[])
    E1 = None
    p = RES / 'jedi' / 'e1_e3.json'
    if p.exists(): E1 = json.loads(p.read_text())['E1']
    rows, Ch = [], []
    PP = ('a', 's', 'u')
    for st in STAGES:
        W_ = WIDTH[st]; pp = st in PP
        s1 = np.zeros(W_); s2 = np.zeros(W_); n_r = 0; mx_r = np.full(W_, -np.inf); mn_a = np.full(W_, np.inf); mx_a = np.full(W_, -np.inf)
        for i in range(0, J, 5000):
            v = net.run(x[i:i + 5000], keep=(st,))[st]
            fl = v[real[i:i + 5000]] if pp else v
            s1 += fl.sum(0); s2 += (fl ** 2).sum(0); n_r += len(fl); mx_r = np.maximum(mx_r, fl.max(0))
            va = v.reshape(-1, W_); mn_a = np.minimum(mn_a, va.min(0)); mx_a = np.maximum(mx_a, va.max(0))
        mean = s1 / n_r; sd = np.sqrt(np.maximum(s2 / n_r - mean ** 2, 0)); mean32 = mean.astype(np.float32)
        ident = (mn_a == mx_a) & (mn_a.astype(np.float32) == mean32)
        dead = (mx_r <= 0) if st != 'z' else np.zeros(W_, bool)
        tested = np.flatnonzero(~ident)
        # resample-ablation values of the tested units
        cols = []
        for i in range(0, J, 5000):
            v = net.run(x[i:i + 5000], keep=(st,))[st]
            cols.append(v[..., tested].astype(np.float32))
        cols = np.concatenate(cols)
        st_rows = []
        for j, k in enumerate(tested):
            m = np.zeros(W_); m[k] = 1
            cm = net.run(x, override={st: (m, mean32)})['logits'].argmax(1)

            def getv(a, b, j=j, k=k):
                V = np.zeros((b - a,) + cols.shape[1:-1] + (W_,), np.float32); V[..., k] = cols[partner[a:b], ..., j]
                return V
            cr = calls_jetwise(net, x, st, m, getv)
            chm, chr_ = cm != call, cr != call; Ch += [chm, chr_]
            r = dict(stage=st, unit=int(k), name=f'{st}{k}', dead_on_real=bool(dead[k]), sd_real=float(sd[k]),
                     mean_cost=prop(chm.sum(), J), resample_cost=prop(chr_.sum(), J),
                     mean_vs_resample=dict(mean_only=int((chm & ~chr_).sum()), resample_only=int((~chm & chr_).sum()),
                                           p_mcnemar=mcnemar(int((chm & ~chr_).sum()), int((~chm & chr_).sum()))),
                     e1_seed0_cost=(E1[st]['changed'][k] if E1 and st in E1 else None))
            st_rows.append(r)
        # confirm the null by construction on up to two identity units
        conf = []
        for k in np.flatnonzero(ident)[:2]:
            m = np.zeros(W_); m[k] = 1
            conf.append(dict(unit=int(k), calls_changed=int((net.run(x, override={st: (m, mean32)})['logits'].argmax(1) != call).sum())))
        res['stages'][st] = dict(width=W_, identity_edit=int(ident.sum()), tested=int(len(tested)), dead_on_real=int(dead.sum()),
                                 identity_units_run=conf)
        rows += st_rows
        log(f'{st}: width {W_}, identity edits {int(ident.sum())} (run: {conf}), tested {len(tested)}; '
            + ', '.join(f"{r['name']} {r['mean_cost']['est']*100:.2f}%/{r['resample_cost']['est']*100:.2f}%" for r in sorted(st_rows, key=lambda r: -r['mean_cost']['k'])[:6]))
    Bm = boot.means(np.stack(Ch).astype(np.float32))
    for i, r in enumerate(rows):
        r['mean_cost']['ci_boot'] = ci(Bm[2 * i]); r['resample_cost']['ci_boot'] = ci(Bm[2 * i + 1])
        k = r['mean_cost']['k']
        r['p_vs_0'] = 0.0 if k > 0 else 1.0
        for tau, lab in ((0.01, '1pct'), (0.001, '0p1pct')):
            r[f'p_gt_{lab}'] = binom_gt(k, J, tau); r[f'p_lt_{lab}'] = binom_lt(k, J, tau)
        e1 = r['e1_seed0_cost']
        r['e1_seed0_in_ci'] = None if e1 is None else bool(r['mean_cost']['ci_boot'][0] - 1e-12 <= e1 <= r['mean_cost']['ci_boot'][1] + 1e-12)
    fam = dict(n_tested=len(rows), family='all tested units of all stages', alpha=0.05)
    for key in ('p_gt_1pct', 'p_lt_1pct', 'p_gt_0p1pct', 'p_lt_0p1pct'):
        adj = holm([r[key] for r in rows])
        for r, a in zip(rows, adj): r[key + '_holm'] = float(a)
    adj = holm([r['mean_vs_resample']['p_mcnemar'] for r in rows])
    for r, a in zip(rows, adj): r['mean_vs_resample']['p_holm'] = float(a)
    for st in STAGES:
        R = [r for r in rows if r['stage'] == st]; d = res['stages'][st]
        for tau, lab in ((0.01, '1pct'), (0.001, '0p1pct')):
            for al, al_l in ((0.05, ''), (0.01, '_a01')):
                d[f'above_{lab}{al_l}'] = sum(r[f'p_gt_{lab}_holm'] < al for r in R)
                d[f'below_{lab}{al_l}'] = sum(r[f'p_lt_{lab}_holm'] < al for r in R)
            d[f'point_ge_{lab}'] = sum(r['mean_cost']['est'] >= tau for r in R)
            d[f'unresolved_{lab}'] = sum((r[f'p_gt_{lab}_holm'] >= 0.05) and (r[f'p_lt_{lab}_holm'] >= 0.05) for r in R)
        d['k_gt_0'] = sum(r['mean_cost']['k'] > 0 for r in R)
        d['resample_point_ge_1pct'] = sum(r['resample_cost']['est'] >= 0.01 for r in R)
        d['mean_vs_resample_differ_holm05'] = sum(r['mean_vs_resample']['p_holm'] < 0.05 for r in R)
        d['resample_larger_holm05'] = sum((r['mean_vs_resample']['p_holm'] < 0.05) and (r['resample_cost']['k'] > r['mean_cost']['k']) for r in R)
        d['e1_seed0_outside_ci'] = sum(r['e1_seed0_in_ci'] is False for r in R)
        d['unresolved_1pct_units'] = [f"{r['name']} {r['mean_cost']['est']*100:.2f}% [{r['mean_cost']['ci_boot'][0]*100:.2f}, {r['mean_cost']['ci_boot'][1]*100:.2f}]"
                                      for r in R if (r['p_gt_1pct_holm'] >= 0.05) and (r['p_lt_1pct_holm'] >= 0.05)]
        log(f"{st}: tested {d['tested']}, k>0 {d['k_gt_0']}, point >=1% {d['point_ge_1pct']}, Holm above 1% {d['above_1pct']}, below {d['below_1pct']}, "
            f"unresolved {d['unresolved_1pct']} {d['unresolved_1pct_units']}; >=0.1%: point {d['point_ge_0p1pct']}, above {d['above_0p1pct']}; "
            f"mean vs resample differ {d['mean_vs_resample_differ_holm05']} (resample larger {d['resample_larger_holm05']}); E1 seed-0 outside CI {d['e1_seed0_outside_ci']}")
    sp = None
    try:
        from scipy.stats import spearmanr
        sp = float(spearmanr([r['mean_cost']['est'] for r in rows], [r['resample_cost']['est'] for r in rows])[0])
    except Exception:  # noqa: BLE001
        pass
    res['family'] = fam; res['spearman_mean_vs_resample_cost'] = sp; res['units'] = rows
    res['null'] = ('H0_u: the call does not depend on unit u. Under H0 every replacement of u changes 0 calls; units whose mean replacement '
                   'is the identity on every slot give 0 by construction (two per stage run). The network is deterministic, so k >= 1 '
                   'changed calls rejects H0_u with p = 0. Tests with content: H0: cost <= threshold (1 %, 0.1 %; the thresholds used to '
                   'call a unit live) by an exact one-sided binomial test, Holm over all tested units; and the resample ablation (value '
                   'from another jet) as the alternative edit, McNemar per unit.')
    save('a1units', res)


def calls_jetwise(net, x, st, mask, getv, batch=5000):
    """calls with the stage's masked units replaced per jet by getv(a, b) (full-width values for jets a..b)"""
    m = np.asarray(mask, np.float32); out = []
    for i in range(0, len(x), batch):
        b = min(i + batch, len(x)); v = getv(i, b)
        if st in ('P', 'g'): v = v.reshape(b - i, 1, -1)
        r = net._f(net.inputs(x[i:b], net.n), {st: (m, v)})
        out.append(np.asarray(r['logits']).reshape(b - i, -1).argmax(1))
    return np.concatenate(out)


# ============================================================================================================ A1 float
def quant_variants(net):
    """E3m (interp/e3m_masks.py) forward passes: 'fixed' (the network), 'mask' (float, every layer's zero-bit input entries
    kept as fixed zeros, the add layer unmasked as in E3m), 'float' (no quantizer)"""
    import jax, jax.numpy as jnp
    L = net.L
    KB = {k: (jnp.asarray(v[0], jnp.float32), jnp.asarray(v[1], jnp.float32)) for k, v in net.KB.items()}

    def bits0(name):
        q = L[name].iq; qs = q.quantizers if hasattr(q, 'quantizers') else [q]; qq = qs[0]
        qq = qq.quantizer if hasattr(qq, 'quantizer') else qq
        return np.asarray(qq.b) if hasattr(qq, 'b') else None
    MASK = {}
    for name in ('l1', 'qs', 'l2', 'dd', 'l3', 'qs1', 'd4', 'd5', 'd6', 'd7'):
        b = bits0(name)
        if b is not None: MASK[name] = jnp.asarray(b > 0, jnp.float32)

    def mk(mode):
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
            u = dense('l3', t); h = dense('d5', dense('d4', pool('qs1', u)))
            return dense('d7', jax.nn.relu(dense('d6', h, act=False))).reshape(xn.shape[0], -1)
        return jax.jit(f)
    return {m: mk(m) for m in ('fixed', 'mask', 'float')}


def part_a1float():
    from interp.jedi_core import Jedi
    from interp.e4_e5_jedi import make_masked
    per = 200 if SMOKE else 4000
    x, y, Q, idx, info = fresh(per); del Q
    J = len(y); net = Jedi(64); boot = Boot(y)
    cnet = net.run(x)['logits'].argmax(1)
    F = quant_variants(net); C = {}
    for m, f in F.items():
        C[m] = np.concatenate([np.asarray(f(net.inputs(x[i:i + 4000], 64))).argmax(1) for i in range(0, J, 4000)])
    C['mask_e5'] = make_masked(net, [0])(x)['logits'].argmax(1)
    res = dict(info=info, fixed_equals_network=float((C['fixed'] == cnet).mean()), acc_fixed=prop((cnet == y).sum(), J), variants={})
    for m in ('mask', 'float', 'mask_e5'):
        c = C[m]; same = c == cnet
        res['variants'][m] = dict(same_call=prop(same.sum(), J), calls_changed=prop((~same).sum(), J), acc=prop((c == y).sum(), J),
                                  acc_minus_fixed=paired_acc(c, cnet, y, boot),
                                  per_class_acc_diff={CL[k]: float((c[y == k] == k).mean() - (cnet[y == k] == k).mean()) for k in range(5)})
        v = res['variants'][m]
        log(f"{m}: same call {v['same_call']['est']*100:.2f}% {np.round(np.array(v['same_call']['ci'])*100, 2)}; acc {v['acc']['est']*100:.2f}% vs fixed "
            f"{res['acc_fixed']['est']*100:.2f}%; diff {v['acc_minus_fixed']['diff']*100:+.2f} {np.round(np.array(v['acc_minus_fixed']['ci'])*100, 2)}, "
            f"McNemar p {v['acc_minus_fixed']['p_mcnemar']:.2e}")
    res['mask_vs_mask_e5_same_call'] = float((C['mask'] == C['mask_e5']).mean())
    res['note'] = ('mask = interp/e3m_masks.py mode "mask" (the E3m numbers 80.68 % / 4.49 %); mask_e5 = interp/e4_e5_jedi.make_masked '
                   '(also masks the add-layer inputs). H0: accuracy(variant) = accuracy(fixed point); McNemar on the jets where exactly one is right.')
    save('a1float', res)


# ============================================================================================================ A2 context
TOP4 = [0, 30, 31, 58]


def part_a2():
    from interp.jedi_core import Jedi, sample
    from interp.e4_e5_jedi import run_ov
    per = 200 if SMOKE else 4000
    x, y, Q, idx, info = fresh(per); del Q
    J = len(y); net = Jedi(64); boot = Boot(y); tail = make_tail(net); rng = np.random.default_rng(SEED)
    B0 = net.run(x, keep=('g', 'S', 'logits')); g, S = B0['g'], B0['S']; call = B0['logits'].argmax(1)
    res = dict(info=info, acc=prop((call == y).sum(), J), tail_check=check_tail(tail, S, B0['logits']))
    live_g = np.flatnonzero(g.std(0) > 1e-9); live_S = np.flatnonzero(S.std(0) > 1e-9)
    res['live_g'] = live_g.tolist(); res['live_S'] = live_S.tolist()
    gm = g.mean(0)
    x0, _, _ = sample(200 if SMOKE else 4000, 0); gm0 = net.run(x0, keep=('g',))['g'].mean(0); del x0
    one = np.ones(64)
    fresh_only = ~np.isin(idx, indices(per, 0))              # jets not in the E4 sample on which S0, S30, S31, S58 were chosen
    res['n_not_in_E4_sample'] = int(fresh_only.sum())

    def est(c2):
        ch = c2 != call
        return dict(calls_changed=prop(ch.sum(), J), acc=prop((c2 == y).sum(), J), acc_minus_unedited=paired_acc(c2, call, y, boot),
                    calls_changed_per_class={CL[k]: prop(ch[y == k].sum(), (y == k).sum()) for k in range(5)},
                    calls_changed_not_in_E4_sample=prop(ch[fresh_only].sum(), fresh_only.sum()))
    E = {}
    Rg = net.run(x, override={'g': (one, gm)}, keep=('S', 'logits')); cg = Rg['logits'].argmax(1); Sm = Rg['S']
    E['g_to_mean'] = est(cg)
    E['g_to_mean_E4_sample_mean'] = est(net.run(x, override={'g': (one, gm0)})['logits'].argmax(1))
    E['g_to_zero'] = est(net.run(x, override={'g': (one, np.zeros(64))})['logits'].argmax(1))
    pt = derangement(J, rng)
    E['g_to_another_jets_g'] = est(net.run_jetwise(x, {'g': (one, g[pt])})['logits'].argmax(1))
    res['edits'] = E
    for k_, v in E.items():
        log(f"{k_}: calls changed {v['calls_changed']['est']*100:.2f}% {np.round(np.array(v['calls_changed']['ci'])*100, 2)}, acc {v['acc']['est']*100:.2f}%")
    single_g = {}
    for c in live_g:
        m = np.zeros(64); m[c] = 1
        ch = net.run(x, override={'g': (m, gm)})['logits'].argmax(1) != call
        single_g[int(c)] = prop(ch.sum(), J)
    res['single_g_to_mean'] = dict(sorted(single_g.items(), key=lambda kv: -kv[1]['k']))
    log('single g -> mean: ' + ', '.join(f"g{c} {v['est']*100:.2f}% {np.round(np.array(v['ci'])*100, 2)}" for c, v in list(res['single_g_to_mean'].items())[:6]))
    S32, Sm32 = S.astype(np.float32), Sm.astype(np.float32); gch = cg != call

    def patch(units):
        S2 = S32.copy(); u = list(units); S2[:, u] = Sm32[:, u]
        return tail(S2).argmax(1)

    def summ(c2):
        ch = c2 != call
        return dict(calls_changed=prop(ch.sum(), J), same_as_g_to_mean_all=prop((c2 == cg).sum(), J),
                    same_as_g_to_mean_on_its_changed=prop((c2 == cg)[gch].sum(), gch.sum()),
                    changed_where_g_to_mean_unchanged=prop((ch & ~gch).sum(), J),
                    calls_changed_not_in_E4_sample=prop(ch[fresh_only].sum(), fresh_only.sum()))
    res['single_S_to_meanctx'] = {int(c): prop((patch([c]) != call).sum(), J) for c in live_S}
    c4 = patch(TOP4)
    m4 = np.zeros(64); m4[TOP4] = 1
    c4_full = run_ov(net, x, {'S': (m4, Sm)}, keep=('logits',))['logits'].argmax(1)
    res['top4_tail_vs_full_forward_same_call'] = float((c4 == c4_full).mean())
    res['top4'] = dict(units=TOP4, **summ(c4))
    res['all_live_S'] = summ(patch(live_S))
    t4 = res['top4']
    log(f"S0,S30,S31,S58 -> mean-context: {t4['calls_changed']['est']*100:.2f}% {np.round(np.array(t4['calls_changed']['ci'])*100, 2)}; same as g->mean on its changed jets "
        f"{t4['same_as_g_to_mean_on_its_changed']['est']*100:.1f}% {np.round(np.array(t4['same_as_g_to_mean_on_its_changed']['ci'])*100, 1)}; tail vs full {res['top4_tail_vs_full_forward_same_call']}")
    obs = int((c4 != call).sum()); obs_agree = float((c4 == cg)[gch].mean()); obs_f = int((c4 != call)[fresh_only].sum())
    others = [int(c) for c in live_S if c not in TOP4]
    nulls = {}
    best = None
    for name, pool in (('four_of_the_other_live', others), ('four_of_all_live', [int(c) for c in live_S])):
        cost, agree, cost_f, subs = [], [], [], []
        for sub in itertools.combinations(pool, 4):
            c2 = patch(sub); ch = c2 != call
            cost.append(int(ch.sum())); agree.append(float((c2 == cg)[gch].mean())); cost_f.append(int(ch[fresh_only].sum())); subs.append(sub)
        cost, agree, cost_f = np.array(cost), np.array(agree), np.array(cost_f)
        incl = name == 'four_of_all_live'
        p_cost = float(((cost >= obs).sum() + (0 if incl else 1)) / (len(cost) + (0 if incl else 1)))
        p_agree = float(((agree >= obs_agree).sum() + (0 if incl else 1)) / (len(agree) + (0 if incl else 1)))
        p_f = float(((cost_f >= obs_f).sum() + (0 if incl else 1)) / (len(cost_f) + (0 if incl else 1)))
        top = np.argsort(-cost)[:5]
        nulls[name] = dict(pool=pool, n_subsets=int(len(cost)), observed_cost=obs / J, null_cost_mean=float(cost.mean() / J),
                           null_cost_q95=float(np.percentile(cost, 95) / J), null_cost_q99=float(np.percentile(cost, 99) / J), null_cost_max=float(cost.max() / J),
                           p_cost=p_cost, observed_agree_on_g_changed=obs_agree, null_agree_mean=float(agree.mean()), null_agree_max=float(agree.max()), p_agree=p_agree,
                           not_in_E4_sample=dict(n=int(fresh_only.sum()), observed_cost=obs_f / max(fresh_only.sum(), 1), null_cost_max=float(cost_f.max() / max(fresh_only.sum(), 1)), p_cost=p_f),
                           top5_subsets=[dict(units=list(subs[i]), cost=float(cost[i] / J)) for i in top],
                           p_definition=('exhaustive enumeration; p = (1 + #{null >= observed}) / (1 + #subsets)' if not incl else
                                         'exhaustive enumeration including the observed set; p = #{subsets with cost >= observed} / #subsets'))
        if name == 'four_of_the_other_live': best = subs[int(np.argmax(cost))]
        log(f"null {name}: {len(cost)} subsets, cost mean {cost.mean()/J*100:.2f}%, q99 {np.percentile(cost, 99)/J*100:.2f}%, max {cost.max()/J*100:.2f}% "
            f"(observed {obs/J*100:.2f}%); p {p_cost:.2e}; agree max {agree.max():.3f} vs {obs_agree:.3f}, p {p_agree:.2e}")
    cb = patch(best)
    nulls['top4_minus_best_null_subset'] = dict(best=list(best), **paired_binary(c4 != call, cb != call, boot))
    res['null'] = nulls
    res['hypotheses'] = ('H0: S0, S30, S31, S58 carry the context effect no more than any other four live S objects (exchangeable); '
                         'H1: they carry more. The four were chosen on the E4 sample (seed 0) by single-object cost; on seed 61 they are fixed in advance.')
    save('a2', res)


# ============================================================================================================ A3 rotation
def rotate(x, deg):
    th = np.deg2rad(deg); c, s = np.cos(th), np.sin(th); xe = x.copy()
    xe[..., 1] = c * x[..., 1] - s * x[..., 2]; xe[..., 2] = s * x[..., 1] + c * x[..., 2]
    return xe.astype(np.float32)


def reflect(x, axis):
    xe = x.copy(); xe[..., axis] = -xe[..., axis]
    return xe


def scramble(x, rng):
    xe = x.copy(); real = x[..., 0] > 0
    r = np.hypot(x[..., 1], x[..., 2]); th = rng.uniform(-np.pi, np.pi, r.shape)
    xe[..., 1] = np.where(real, r * np.cos(th), 0); xe[..., 2] = np.where(real, r * np.sin(th), 0)
    return xe.astype(np.float32)


def part_a3():
    from interp.jedi_core import Jedi
    from interp.e4_e5_jedi import make_masked
    per = 200 if SMOKE else 4000; BB = 300 if SMOKE else 10000
    x, y, Q, idx, info = fresh(per); del Q
    J = len(y); net = Jedi(64); tail = make_tail(net); rng = np.random.default_rng(SEED)
    B0 = net.run(x, keep=('S', 'logits')); S = B0['S']; call = B0['logits'].argmax(1)
    res = dict(info=info, acc=prop((call == y).sum(), J), tail_check=check_tail(tail, S, B0['logits']))
    rerun = net.run(x)['logits']
    res['null_rerun_unedited'] = dict(calls_changed=prop((rerun.argmax(1) != call).sum(), J), max_abs_logit_diff=float(np.abs(rerun - B0['logits']).max()))
    live = np.flatnonzero(S.std(0) > 1e-9); res['live_S'] = live.tolist(); nl = len(live)
    MF = make_masked(net, live); Mb = MF(x); Smf = Mb['S'].astype(np.float64); cmf = Mb['logits'].argmax(1)
    res['masked_float_calls_differ_from_fixed'] = prop((cmf != call).sum(), J)
    EDITS = [('rot45', lambda: rotate(x, 45)), ('rot90', lambda: rotate(x, 90)), ('rot180', lambda: rotate(x, 180)),
             ('phi_flip', lambda: reflect(x, 2)), ('eta_flip', lambda: reflect(x, 1)), ('scramble', lambda: scramble(x, rng))]
    RPRES = ['rot45', 'rot90', 'rot180', 'phi_flip', 'eta_flip']
    ch_all, D2, D2m = {}, {}, {}
    res['edits'] = {}
    S32 = S.astype(np.float32)
    for name, f in EDITS:
        xe = f(); E = net.run(xe, keep=('S', 'logits')); ce = E['logits'].argmax(1); ch = ce != call; ch_all[name] = ch
        cme = MF(xe); chm = cme['logits'].argmax(1) != cmf
        dS = (E['S'] - S)[:, live]; dSm = (cme['S'].astype(np.float64) - Smf)[:, live]
        D2[name] = dS ** 2; D2m[name] = dSm ** 2
        rms = np.sqrt(D2[name].mean(0))
        rec = dict(calls_changed=prop(ch.sum(), J), calls_changed_masked_float=prop(chm.sum(), J), acc=prop((ce == y).sum(), J),
                   calls_changed_per_class={CL[k]: prop(ch[y == k].sum(), (y == k).sum()) for k in range(5)},
                   p_vs_rerun_null_mcnemar=mcnemar(int(ch.sum()), 0), p_gt_3pct=binom_gt(ch.sum(), J, 0.03),
                   float_r_max_abs_change=float(np.abs(np.hypot(xe[..., 1], xe[..., 2]) - np.hypot(x[..., 1], x[..., 2])).max()))
        # matched per-unit spread controls (live S units only; S clipped at 0, as a sum of ReLU outputs)
        gau, perm_ = [], []
        for d in range(3 if SMOKE else 10):
            r_ = np.random.default_rng(1000 + d)
            S2 = S32.copy(); S2[:, live] = np.maximum(S2[:, live] + r_.normal(0, 1, (J, nl)).astype(np.float32) * rms.astype(np.float32), 0)
            cgau = tail(S2).argmax(1) != call; gau.append(cgau)
            S3 = S32.copy()
            for jj, c in enumerate(live): S3[:, c] = np.maximum(S3[:, c] + dS[derangement(J, r_), jj].astype(np.float32), 0)
            perm_.append(tail(S3).argmax(1) != call)
        rec['control_gaussian_matched_rms'] = dict(draws=[int(v.sum()) for v in gau], rate_mean=float(np.mean([v.mean() for v in gau])),
                                                   rate_min=float(min(v.mean() for v in gau)), rate_max=float(max(v.mean() for v in gau)),
                                                   draw0=prop(gau[0].sum(), J), edit_vs_draw0=paired_binary(ch, gau[0], Boot(y, B=200 if SMOKE else 2000)))
        rec['control_shift_permuted_across_jets'] = dict(draws=[int(v.sum()) for v in perm_], rate_mean=float(np.mean([v.mean() for v in perm_])),
                                                         rate_min=float(min(v.mean() for v in perm_)), rate_max=float(max(v.mean() for v in perm_)),
                                                         draw0=prop(perm_[0].sum(), J), edit_vs_draw0=paired_binary(ch, perm_[0], Boot(y, B=200 if SMOKE else 2000)))
        # carriers: one edited S_c patched into the unedited network
        if name in ('rot45', 'rot180', 'scramble'):
            car = {}
            for jj, c in enumerate(live):
                S2 = S32.copy(); S2[:, c] = E['S'][:, c]; car[int(c)] = prop((tail(S2).argmax(1) != call).sum(), J)
            S2 = S32.copy(); S2[:, live] = E['S'][:, live]
            rec['carriers_single_S_patch'] = dict(sorted(car.items(), key=lambda kv: -kv[1]['k']))
            rec['all_live_S_patch'] = prop((tail(S2).argmax(1) != call).sum(), J)
        res['edits'][name] = rec
        g_ = rec['control_gaussian_matched_rms']; p_ = rec['control_shift_permuted_across_jets']
        log(f"{name}: calls changed {rec['calls_changed']['est']*100:.2f}% {np.round(np.array(rec['calls_changed']['ci'])*100, 2)} (masked float "
            f"{rec['calls_changed_masked_float']['est']*100:.2f}%); p(>3%) {rec['p_gt_3pct']:.2e}; matched Gaussian {g_['rate_min']*100:.2f}-{g_['rate_max']*100:.2f}%, "
            f"permuted shift {p_['rate_min']*100:.2f}-{p_['rate_max']*100:.2f}%")
    res['cochran_q_r_preserving'] = cochran_q(np.stack([ch_all[e] for e in RPRES], 1))
    res['pairwise_r_preserving_mcnemar'] = {f'{a} vs {b}': mcnemar(int((ch_all[a] & ~ch_all[b]).sum()), int((~ch_all[a] & ch_all[b]).sum()))
                                            for a, b in itertools.combinations(RPRES, 2)}
    # per-object rms shift / SD: bootstrap (stratified, BB replicates)
    names = [e for e, _ in EDITS]
    Sc = S[:, live] - S[:, live].mean(0); Smc = Smf[:, live] - Smf[:, live].mean(0)
    rows = [D2[e].T for e in names] + [D2m[e].T for e in names] + [Sc.T ** 2, Smc.T ** 2, Sc.T, Smc.T]
    M = np.concatenate(rows, 0).astype(np.float32)
    bt = Boot(y, B=BB); Bm = bt.means(M); ne = len(names)
    dB = Bm[:ne * nl].reshape(ne, nl, BB); dBm = Bm[ne * nl:2 * ne * nl].reshape(ne, nl, BB)
    o = 2 * ne * nl; s2, s2m, s1, s1m = (Bm[o + i * nl:o + (i + 1) * nl] for i in range(4))
    var, varm = np.maximum(s2 - s1 ** 2, 1e-30), np.maximum(s2m - s1m ** 2, 1e-30)
    stat = np.sqrt(dB / var[None]); statm = np.sqrt(dBm / varm[None])            # (edit, unit, B)
    pt_ = np.sqrt(np.stack([D2[e].mean(0) for e in names]) / S[:, live].var(0)[None])
    ptm = np.sqrt(np.stack([D2m[e].mean(0) for e in names]) / np.maximum(Smf[:, live].var(0), 1e-30)[None])
    rp = [names.index(e) for e in RPRES]
    tau = 0.1; obj = []
    for jj, c in enumerate(live):
        d = dict(S=int(c))
        for i, e in enumerate(names):
            d[e] = dict(rms_over_sd=float(pt_[i, jj]), ci=ci(stat[i, jj]), masked_float=float(ptm[i, jj]), masked_float_ci=ci(statm[i, jj]),
                        p_gt=float((1 + (stat[i, jj] <= tau).sum()) / (BB + 1)), p_lt=float((1 + (stat[i, jj] >= tau).sum()) / (BB + 1)))
        mx = stat[rp, jj].max(0); mxp = pt_[rp, jj].max()
        d['max_r_preserving'] = dict(rms_over_sd=float(mxp), ci=ci(mx), argmax=RPRES[int(np.argmax(pt_[rp, jj]))],
                                     p_gt=float((1 + (mx <= tau).sum()) / (BB + 1)), p_lt=float((1 + (mx >= tau).sum()) / (BB + 1)))
        mxm = statm[rp, jj].max(0)
        d['max_r_preserving_masked_float'] = dict(rms_over_sd=float(ptm[rp, jj].max()), ci=ci(mxm),
                                                  p_gt=float((1 + (mxm <= tau).sum()) / (BB + 1)), p_lt=float((1 + (mxm >= tau).sum()) / (BB + 1)))
        obj.append(d)
    summ = {}
    for key in ['rot180', 'rot45', 'max_r_preserving', 'max_r_preserving_masked_float']:
        for side in ('p_gt', 'p_lt'):
            adj = holm([d[key][side] for d in obj])
            for d, a in zip(obj, adj): d[key][side + '_holm'] = float(a)
        summ[key] = dict(n_objects=nl, threshold_sd=tau,
                         above_holm05=[f"S{d['S']}" for d in obj if d[key]['p_gt_holm'] < 0.05],
                         above_holm01=[f"S{d['S']}" for d in obj if d[key]['p_gt_holm'] < 0.01],
                         below_holm05=[f"S{d['S']}" for d in obj if d[key]['p_lt_holm'] < 0.05],
                         below_holm01=[f"S{d['S']}" for d in obj if d[key]['p_lt_holm'] < 0.01],
                         unresolved05=[f"S{d['S']}" for d in obj if d[key]['p_gt_holm'] >= 0.05 and d[key]['p_lt_holm'] >= 0.05])
        log(f"{key}: above 0.1 SD (Holm 0.05) {len(summ[key]['above_holm05'])} {summ[key]['above_holm05']}; below {len(summ[key]['below_holm05'])} "
            f"{summ[key]['below_holm05']}; unresolved {summ[key]['unresolved05']}")
    res['objects'] = obj; res['objects_summary'] = summ; res['n_boot_objects'] = BB
    res['hypotheses'] = ('calls: H0 the edit changes no call (rerun null: 0 exactly) and H0 rate <= 3 % (E5v threshold); objects: per live S unit '
                         'H0 rms(dS)/SD(S) <= 0.1 vs H1 > 0.1, and H0 >= 0.1 vs H1 < 0.1; p = (1 + #replicates on the H0 side) / (B + 1) '
                         '(percentile-bootstrap test), Holm over the live S units; "max_r_preserving" = the largest of the five r-preserving edits.')
    save('a3', res)


# ============================================================================================================ A9 layers
def spearman(a, b):
    from scipy.stats import spearmanr
    if np.std(a) == 0 or np.std(b) == 0: return np.nan
    return float(spearmanr(a, b)[0])


def part_a9():
    from interp.jedi_core import Jedi
    from interp import e10_layers_jedi as E10
    per = 300 if SMOKE else 4000; NB = 300 if SMOKE else 10000; NJ = 100 if SMOKE else 1000; N2 = 200 if SMOKE else 2000
    x, y, Q, idx, info = fresh(per)
    O = E10.derive(Q, x); del Q
    J = len(y); tr, te = E10.split(J); net = Jedi(64)
    A = net.run(x, keep=E10.STG + ('logits',)); LV = E10.live(A)
    res = dict(info=info, split='E10: rng(0) permutation, first half fits the ridge probes, second half scores them', acc=prop((A['logits'].argmax(1) == y).sum(), J),
               live={s: int(len(v)) for s, v in LV.items()}, obs={})
    nO, nS = len(E10.OBS), len(E10.STG)
    R2 = np.zeros((nO, nS)); PRED = np.zeros((nO, nS, len(te)), np.float32); TC = np.zeros((nO, len(te)), np.float32)
    for oi, (oid, depth, slot, why) in enumerate(E10.OBS):
        t = O[oid]; lo, hi = np.percentile(t[tr], [0.5, 99.5]); tc = np.clip(t, lo, hi); TC[oi] = tc[te]
        for si, s in enumerate(E10.STG):
            u = LV[s]
            if len(u) == 0: PRED[oi, si] = tc[te].mean(); continue
            X = A[s][:, u]; pl = E10.Probe(X[tr], t[tr])(X[te]); PRED[oi, si] = pl; R2[oi, si] = E10.r2(tc[te], pl)
    depth = np.array([o[1] for o in E10.OBS]); slot = np.array([o[2] for o in E10.OBS])

    def first(R, thr):
        ok = R >= thr; f = np.where(ok.any(-1), ok.argmax(-1), E10.NEVER)
        return f
    F = {thr: first(R2, thr) for thr in (0.9, 0.7)}
    for oi, (oid, d_, s_, _) in enumerate(E10.OBS):
        res['obs'][oid] = dict(depth=int(d_), slot=int(s_), R2={s: float(R2[oi, si]) for si, s in enumerate(E10.STG)},
                               first_09=int(F[0.9][oi]), first_07=int(F[0.7][oi]))
    # jet bootstrap of the held-out R² (probes fixed): R²* = 1 - Σw(t - p)² / (Σw t² - (Σw t)² / n)
    RES2 = ((TC[:, None, :] - PRED) ** 2).reshape(nO * nS, -1); n_te = len(te)
    rngj = np.random.default_rng(SEED + 1); FJ = {0.9: [], 0.7: []}
    done = 0
    while done < NJ:
        b = min(100, NJ - done); W = np.stack([np.bincount(rngj.integers(0, n_te, n_te), minlength=n_te) for _ in range(b)]).astype(np.float32)
        rss = (RES2 @ W.T).reshape(nO, nS, b); st1 = TC @ W.T; st2 = (TC ** 2) @ W.T; tss = st2 - st1 ** 2 / n_te
        Rb = 1 - rss / np.maximum(tss, 1e-30)[:, None, :]
        for thr in (0.9, 0.7): FJ[thr].append(first(np.moveaxis(Rb, 2, 0), thr))
        done += b
    FJ = {k: np.concatenate(v) for k, v in FJ.items()}               # (NJ, nO)
    E10s = None
    p = RES / 'jedi' / 'e10_layers.json'
    if p.exists(): E10s = json.loads(p.read_text())
    rng = np.random.default_rng(SEED); out = {}
    for name, sel in (('all', np.ones(nO, bool)), ('no_slot', slot == 0)):
        ids = np.flatnonzero(sel); d = depth[ids]; out[name] = dict(n=int(len(ids)))
        for thr, lab in ((0.9, '09'), (0.7, '07')):
            f = F[thr][ids]; rho = spearman(d, f)
            bo = np.array([spearman(d[i], f[i]) for i in (rng.integers(0, len(ids), len(ids)) for _ in range(NB))])
            pe = np.array([spearman(rng.permutation(d), f) for _ in range(NB)])
            bj = np.array([spearman(d, FJ[thr][b, ids]) for b in range(len(FJ[thr]))])
            b2 = []
            for b in range(N2):
                i = rng.integers(0, len(ids), len(ids)); fj = FJ[thr][rng.integers(0, len(FJ[thr])), ids]
                b2.append(spearman(d[i], fj[i]))
            b2 = np.array(b2)
            fin = lambda v: v[np.isfinite(v)]
            e = dict(rho=rho, ci_boot_observables=ci(fin(bo)), n_boot_observables_finite=int(np.isfinite(bo).sum()),
                     p_perm_one_sided=float((1 + (pe >= rho).sum()) / (NB + 1)), p_perm_two_sided=float((1 + (np.abs(pe) >= abs(rho)).sum()) / (NB + 1)),
                     ci_boot_jets_probes_fixed=ci(fin(bj)), ci_boot_jets_and_observables=ci(fin(b2)),
                     median_first_by_depth={int(q): float(np.median(f[d == q])) for q in range(4)},
                     never_by_depth={int(q): int((f[d == q] == E10.NEVER).sum()) for q in range(4)},
                     n_by_depth={int(q): int((d == q).sum()) for q in range(4)},
                     first_stage_stability=float((FJ[thr][:, ids] == f[None]).mean()))
            if E10s is not None:
                f0 = np.array([E10s['obs'][E10.OBS[i][0]][f'first_lin_{thr}'] for i in ids])
                e['seed0_rho'] = E10s['summary'][name][f'first_lin_{thr}']['spearman']
                e['first_stage_differs_from_seed0'] = [E10.OBS[i][0] for i, a, b in zip(ids, f, f0) if a != b]
            out[name][lab] = e
            log(f"{name} R2>={thr}: rho {rho:.3f}, CI observables {np.round(e['ci_boot_observables'], 3)}, jets {np.round(e['ci_boot_jets_probes_fixed'], 3)}, "
                f"both {np.round(e['ci_boot_jets_and_observables'], 3)}; p_perm one-sided {e['p_perm_one_sided']:.2e}, two-sided {e['p_perm_two_sided']:.2e}; "
                f"seed-0 rho {e.get('seed0_rho')}; first stage differs from seed 0 for {len(e.get('first_stage_differs_from_seed0', []))}")
    res['spearman'] = out; res['n_perm'] = NB; res['n_boot_jets'] = NJ
    res['hypotheses'] = ('H0: depth labels exchangeable across observables (no association with first stage); H1: rho > 0 (one-sided) / rho != 0. '
                         'never = 6 (stage index after z), as E10; ties average-ranked.')
    save('a9', res)


# ============================================================================================================ B2 / B3 program
def fresh_b(per):
    i0, i11, i31 = indices(6000, 0), indices(2000, 11), indices(4000, 31); i61 = indices(per, SEED)
    keep = ~np.isin(i61, np.concatenate([i0, i11, i31])); return i61[keep], dict(drawn=int(len(i61)), removed=int((~keep).sum()))


def rej_boot(V, lo_p, lo_n, y, B, seed=SEED):
    Pp, Pn = V.softmax(lo_p), V.softmax(lo_n); Tp, Tn = V.tasks(Pp, y), V.tasks(Pn, y); out = {}; J = len(y)
    for name in Tp:
        rp, _ = V.rej50(*Tp[name]); rn, _ = V.rej50(*Tn[name]); lr = []
        rng = np.random.default_rng(seed)
        for _ in range(B):
            ix = rng.integers(0, J, J); sp, sg, bg = (a[ix] for a in Tp[name]); sn = Tn[name][0][ix]
            a_, _ = V.rej50(sp, sg, bg); b_, _ = V.rej50(sn, sg, bg)
            lr.append(np.log(a_) - np.log(b_) if np.isfinite(a_) or np.isfinite(b_) else np.nan)
        lr = np.array(lr); lf = lr[~np.isnan(lr)]
        q = lambda v: (float(np.exp(v)) if np.isfinite(v) else None)
        lo_, hi_ = (np.percentile(lf, [2.5, 97.5]) if len(lf) else (np.nan, np.nan))
        p2 = float(min(1.0, 2 * min((lf <= 0).mean(), (lf >= 0).mean()))) if len(lf) else None
        out[name] = dict(program=rp if np.isfinite(rp) else None, network=rn if np.isfinite(rn) else None,
                         ratio=float(rp / rn) if np.isfinite(rp) and np.isfinite(rn) else None, ci=[q(lo_), q(hi_)], p_ratio_eq_1=p2,
                         n_boot=B, n_boot_used=int(len(lf)), n_boot_infinite=int((~np.isfinite(lf)).sum()),
                         n_sig=int(Tp[name][1].sum()), n_bkg=int(Tp[name][2].sum()), bkg_passing_network=int(round(Tn[name][2].sum() / rn)) if np.isfinite(rn) and rn > 0 else None)
    return out


def part_b2():
    from interp.jedi_core import Jedi
    from interp import e11_verify as V
    E = V.E; per = 300 if SMOKE else 5000; B = 200 if SMOKE else 2000
    ifr, ov = fresh_b(per)
    D, _ = V.load(dict(fresh=ifr), need_q=['fresh']); x, y, Q = D['fresh']['x'], D['fresh']['y'], D['fresh']['Q']; del D
    J = len(y); net = Jedi(64); last = V.Last(net); boot = Boot(y, B=B)
    cn = net.run(x)['logits']; lon = cn; cn = cn.argmax(1)
    info = dict(sample=f'sample({per}, seed {SEED}) minus the rows of sample(6000, 0), sample(2000, 11), sample(4000, 31)', **ov, n=J,
                per_class=[int((y == k).sum()) for k in range(5)])
    log(f'b2: {info}')
    res = dict(info=info, network_acc=prop((cn == y).sum(), J), programs={})
    calls = {}
    for prefix in ('e11', 'e11hp'):
        hp = prefix == 'e11hp'; fa, fb = V.formulas(prefix); phys = V.physics(x, Q, hp); Vv = V.chain(fb, phys)
        lo_p = last.jax_logits(Vv['z']); lo_a = last.jax_logits(E.evaluate(fa['z']['units'], phys, False)); del phys, Vv
        for nm, lo in ((f'{prefix}_nested', lo_p), (f'{prefix}_z_on_physics', lo_a)):
            c = lo.argmax(1); calls[nm] = c
            r = dict(same_call=prop((c == cn).sum(), J), acc=prop((c == y).sum(), J), acc_minus_network=paired_acc(c, cn, y, boot),
                     per_class_acc_minus_network={CL[k]: paired_acc(c[y == k], cn[y == k], y[y == k], Boot(y[y == k], B=B)) for k in range(5)},
                     rejection50=rej_boot(V, lo, lon, y, B))
            res['programs'][nm] = r
            w = r['rejection50']['W vs rest']
            log(f"{nm}: same call {r['same_call']['est']*100:.2f}% {np.round(np.array(r['same_call']['ci'])*100, 2)}; acc diff {r['acc_minus_network']['diff']*100:+.2f} "
                f"{np.round(np.array(r['acc_minus_network']['ci'])*100, 2)} p {r['acc_minus_network']['p_mcnemar']:.3g}; W rej ratio {w['ratio']} {w['ci']} p {w['p_ratio_eq_1']} "
                f"(bkg passing {w['bkg_passing_network']}, infinite replicates {w['n_boot_infinite']})")
    res['nested_vs_direct_same_call'] = {f'{a} vs {b}': paired_binary(calls[a] == cn, calls[b] == cn, boot) for a, b in
                                         (('e11_nested', 'e11_z_on_physics'), ('e11hp_nested', 'e11_z_on_physics'), ('e11hp_nested', 'e11_nested'))}
    for k, v in res['nested_vs_direct_same_call'].items(): log(f"unedited same call {k}: {v['diff']*100:+.2f} {np.round(np.array(v['ci'])*100, 2)} p {v['p_mcnemar']:.3g}")
    res['hypotheses'] = ('same call: estimate + Wilson; H0 accuracy(program) = accuracy(network), McNemar; rejection ratio: plain bootstrap (as E11v), '
                         'H0 ratio = 1, p = 2 min(P*(log ratio <= 0), P*(log ratio >= 0)).')
    save('b2', res)


def part_b3():
    from interp.jedi_core import Jedi
    from interp import e11_verify as V
    from jetdistill.observables import compute
    E = V.E; per_class = 100 if SMOKE else 2000; B = 200 if SMOKE else 2000
    ifr, ov = fresh_b(400 if SMOKE else 5000)
    D, _ = V.load(dict(fresh=ifr), need_q=['fresh']); yf = D['fresh']['y']; rng = np.random.default_rng(SEED)
    pick = np.sort(np.concatenate([rng.choice(np.flatnonzero(yf == c), per_class, replace=False) for c in range(5)]))
    x0 = D['fresh']['x'][pick]; y = yf[pick]; del D
    xs = x0.copy(); rl = xs[..., 0] > 0; xs[..., 1] += np.float32(0.06) * rl
    xp = x0.copy(); xp[..., 0] *= np.float32(0.8)
    X = dict(unedited=x0, shift_eta_0p06=xs, pt_x0p8=xp); Qr = {}
    for k, v in X.items():
        t = time.time(); q = compute(v.astype(np.float64), 64); Qr[k] = {kk: np.asarray(vv, np.float32).astype(np.float64) for kk, vv in q.items()}
        log(f'observables recomputed for {k}: {len(q)} in {time.time() - t:.0f} s')
    J = len(y); net = Jedi(64); last = V.Last(net); boot = Boot(y, B=B)
    C = {k: net.run(v)['logits'].argmax(1) for k, v in X.items()}
    info = dict(sample=f'{per_class}/class (rng {SEED}) from sample(5000, seed {SEED}) minus seeds 0/11/31 rows', **ov, n=J)
    res = dict(info=info, network_acc={k: prop((C[k] == y).sum(), J) for k in X},
               mass_rel_change={k: float(np.mean(Qr[k]['mass'] / np.maximum(Qr['unedited']['mass'], 1e-9) - 1)) for k in X if k != 'unedited'})
    P = {}
    for prefix in ('e11', 'e11hp'):
        hp = prefix == 'e11hp'; fa, fb = V.formulas(prefix)
        for k in X:
            ph = V.physics(X[k], Qr[k], hp)
            P[(f'{prefix}_nested', k)] = last.jax_logits(V.chain(fb, ph)['z']).argmax(1)
            if not hp: P[('e11_z_on_physics', k)] = last.jax_logits(E.evaluate(fa['z']['units'], ph, False)).argmax(1)
            del ph
    progs = ['e11_nested', 'e11hp_nested', 'e11_z_on_physics']
    res['same_call_with_network'] = {k: {p: prop((P[(p, k)] == C[k]).sum(), J) for p in progs} for k in X}
    res['nested_vs_direct'] = {k: {f'{a} vs {b}': paired_binary(P[(a, k)] == C[k], P[(b, k)] == C[k], boot) for a, b in
                                   (('e11_nested', 'e11_z_on_physics'), ('e11hp_nested', 'e11_z_on_physics'), ('e11hp_nested', 'e11_nested'))} for k in X}
    FLOWS = [('W', 'Z'), ('Z', 'q'), ('q', 'W'), ('Z', 't'), ('W', 'q'), ('W', 't')]
    fl = {}
    for k in X:
        fl[k] = {}
        for a, b in FLOWS:
            m = y == CL.index(a); bi = CL.index(b); e = {'network': prop((C[k][m] == bi).sum(), m.sum())}
            for p in progs:
                e[p] = prop((P[(p, k)][m] == bi).sum(), m.sum())
                pa, pn = P[(p, k)][m] == bi, C[k][m] == bi
                e[f'{p}_minus_network'] = dict(diff=float(pa.mean() - pn.mean()), p_mcnemar=mcnemar(int((pa & ~pn).sum()), int((~pa & pn).sum())))
            fl[k][f'{a}->{b}'] = e
    res['flows'] = fl
    for k in X:
        s = res['same_call_with_network'][k]; nv = res['nested_vs_direct'][k]['e11_nested vs e11_z_on_physics']; w = fl[k]['W->Z']
        log(f"{k}: same call nested {s['e11_nested']['est']*100:.1f}% {np.round(np.array(s['e11_nested']['ci'])*100, 1)}, hp {s['e11hp_nested']['est']*100:.1f}%, "
            f"direct {s['e11_z_on_physics']['est']*100:.1f}% {np.round(np.array(s['e11_z_on_physics']['ci'])*100, 1)}; nested - direct {nv['diff']*100:+.1f} "
            f"{np.round(np.array(nv['ci'])*100, 1)} p {nv['p_mcnemar']:.3g}; W->Z network {w['network']['est']*100:.1f}% nested {w['e11_nested']['est']*100:.1f}% "
            f"(p vs network {w['e11_nested_minus_network']['p_mcnemar']:.3g}) direct {w['e11_z_on_physics']['est']*100:.1f}%")
    res['hypotheses'] = ('per edit: H0 same-call rate(nested) = same-call rate(direct formula), McNemar on the jets where exactly one agrees with the '
                         'network; flows: H0 program flow rate = network flow rate on the same true-class jets, McNemar.')
    save('b3', res)


# ============================================================================================================ small models
def part_small():
    from interp.jedi_core import Jedi
    from jetdistill.config import RESULTS, DATA
    import h5py
    B = 200 if SMOKE else 2000
    d = np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r'); y = np.asarray(d['y']).astype(int)
    with h5py.File(Path(DATA) / 'val_meta.h5', 'r') as f: yl = f['label'][:].astype(int)
    sel = np.arange(len(y)) if not SMOKE else np.sort(np.random.default_rng(0).choice(len(y), 20000, replace=False))
    res = dict(n_test=int(len(sel)), labels_equal_val_meta=bool(np.array_equal(y, yl)))
    x = np.asarray(d['x'][sel], np.float32); y = y[sel]
    net = Jedi(64); cj = net.run(x, batch=5000)['logits'].argmax(1); del x
    res['jedi'] = dict(acc=prop((cj == y).sum(), len(y)), official=0.80921, source='interp.jedi_core.Jedi(64) forward on full_test.npz')
    log(f"JEDI-linear on {len(y)} test jets: {res['jedi']['acc']['est']*100:.3f}%; labels equal val_meta {res['labels_equal_val_meta']}")
    runs = {}
    for tp in sorted(SMALL.glob('*-n64*/seed*/test_probs.npy')):
        name = f'{tp.parent.parent.name}/{tp.parent.name}'; P = np.load(tp)
        if len(P) != res['n_test'] and not SMOKE: log(f'skip {name}: {len(P)} rows'); continue
        T = json.loads((tp.parent / 'test.json').read_text()) if (tp.parent / 'test.json').exists() else {}
        runs[name] = dict(calls=P[sel].argmax(1), test_json_acc=T.get('accuracy'), best_epoch=T.get('best_epoch'), n_params=T.get('n_params'))
    names = list(runs); Cm = np.stack([runs[n]['calls'] == y for n in names] + [cj == y]).astype(np.float32)
    bt = Boot(y, B=B, strat=False); Bm = bt.means(Cm); bj = Bm[-1]
    res['runs'] = {}
    for i, n in enumerate(names):
        c = runs[n]['calls']; r = paired_acc(c, cj, y, bt)
        res['runs'][n] = dict(acc=prop((c == y).sum(), len(y)), acc_ci_boot=ci(Bm[i]), test_json_acc=runs[n]['test_json_acc'], best_epoch=runs[n]['best_epoch'],
                              n_params=runs[n]['n_params'], minus_jedi=r)
        log(f"{n}: {res['runs'][n]['acc']['est']*100:.3f}% {np.round(np.array(res['runs'][n]['acc_ci_boot'])*100, 3)} (test.json {runs[n]['test_json_acc']}); "
            f"- JEDI {r['diff']*100:+.3f} {np.round(np.array(r['ci'])*100, 3)} p {r['p_mcnemar']:.3g}")
    res['jedi']['acc_ci_boot'] = ci(bj)
    # seed pairs and two-seed means (X1 table: seed 1 of tf-2b-* replaced by the patience-30 rerun)
    arch = sorted({n.split('/')[0] for n in names if not n.split('/')[0].endswith(('-p30', '-axhard', '-e100'))})
    res['architectures'] = {}
    for a in arch:
        s0, s1 = f'{a}/seed0', (f'{a}-p30/seed1' if f'{a}-p30/seed1' in runs else f'{a}/seed1')
        if s0 not in runs or s1 not in runs: continue
        c0, c1 = runs[s0]['calls'], runs[s1]['calls']; a0, a1 = (c0 == y).mean(), (c1 == y).mean()
        mean_vec = 0.5 * ((c0 == y).astype(np.float32) + (c1 == y).astype(np.float32)) - (cj == y).astype(np.float32)
        dm = bt.means(mean_vec)[0]
        from scipy.stats import t as tdist
        half = float(tdist.ppf(0.975, 1) * abs(a0 - a1) / 2)
        res['architectures'][a] = dict(seed0=s0, seed1=s1, acc_seed0=float(a0), acc_seed1=float(a1), mean=float((a0 + a1) / 2), spread_half_diff=float(abs(a0 - a1) / 2),
                                       seed1_minus_seed0=paired_acc(c1, c0, y, bt),
                                       mean_minus_jedi=dict(diff=float((a0 + a1) / 2 - (cj == y).mean()), ci_test_jets=ci(dm)),
                                       seed_t_interval_1df=[float((a0 + a1) / 2 - half), float((a0 + a1) / 2 + half)],
                                       note='seed_t_interval_1df: mean ± t(0.975, 1 df) · |a0 − a1| / 2 (two seeds; test-set noise not included)')
        r = res['architectures'][a]
        log(f"{a}: seeds {a0*100:.3f} / {a1*100:.3f}% (seed1 - seed0 {r['seed1_minus_seed0']['diff']*100:+.3f}, p {r['seed1_minus_seed0']['p_mcnemar']:.3g}); "
            f"mean - JEDI {r['mean_minus_jedi']['diff']*100:+.3f} {np.round(np.array(r['mean_minus_jedi']['ci_test_jets'])*100, 3)}; t(1 df) interval {np.round(np.array(r['seed_t_interval_1df'])*100, 2)}")
    res['hypotheses'] = 'H0: accuracy(model) = accuracy(JEDI-linear) on the same test jets, McNemar; CI from 2,000 plain bootstrap resamples of the 260,000 jets.'
    save('small', res)


# ============================================================================================================ table
def part_table():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    L = {}
    for k in ('a1units', 'a1float', 'a2', 'a3', 'a9', 'b2', 'b3', 'small'):
        p = OUT / f'r2_{k}{"_smoke" if SMOKE else ""}.json'
        if p.exists(): L[k] = json.loads(p.read_text())
    pc = lambda v, d=2: '—' if v is None else f'{v * 100:.{d}f} %'
    iv = lambda c, d=2: '—' if c is None or c[0] is None else f'[{c[0] * 100:.{d}f}, {c[1] * 100:.{d}f}] %'
    pv = lambda p: '—' if p is None else ('< 1e-300' if p == 0 else (f'{p:.1e}' if p < 1e-3 else f'{p:.3f}'))
    vd = lambda p: '—' if p is None else ('H0 rejected' if p < 0.01 else 'not rejected')
    rows = []
    if 'a1units' in L:
        A = L['a1units']; st = A['stages']; o = ['a', 'P', 'g', 's', 'u', 'S', 'h4', 'h5', 'z']
        rows.append(['A1', 'units with mean-ablation cost > 1 % of calls, per stage a/P/g/s/u/S/h4/h5/z (point count)',
                     '/'.join(str(st[s]['point_ge_1pct']) for s in o), '—',
                     'H0 cost ≤ 1 %, binomial, Holm over ' + str(A['family']['n_tested']) + ' tested units',
                     'counts at Holm 0.05', '/'.join(str(st[s]['above_1pct']) for s in o) + ' above; ' + '/'.join(str(st[s]['unresolved_1pct']) for s in o) + ' unresolved'])
        rows.append(['A1', 'units with cost > 0.1 % (point count; E11 live set)', '/'.join(str(st[s]['point_ge_0p1pct']) for s in o), '—',
                     'H0 cost ≤ 0.1 %, binomial, Holm', 'counts at Holm 0.05', '/'.join(str(st[s]['above_0p1pct']) for s in o) + ' above'])
        U = {r['name']: r for r in A['units']}
        for nm in ('a32', 'S29', 'h4_49', 'z1'):
            nm2 = nm.replace('_', '')
            if nm2 in U:
                r = U[nm2]
                rows.append(['A1', f'cost of {nm2} (mean ablation) vs resample ablation', pc(r['mean_cost']['est']), iv(r['mean_cost']['ci_boot']),
                             f"resample {pc(r['resample_cost']['est'])}", pv(r['mean_vs_resample']['p_mcnemar']), vd(r['mean_vs_resample']['p_mcnemar']) + ' (mean = resample)'])
    if 'a1float' in L:
        F = L['a1float']['variants']['mask']
        rows.append(['A1', 'float + zero-bit masks: same call as fixed point', pc(F['same_call']['est']), iv(F['same_call']['ci']), '—', '—', '—'])
        rows.append(['A1', 'float + masks accuracy − fixed point', pc(F['acc_minus_fixed']['diff']), iv(F['acc_minus_fixed']['ci']), 'H0 equal accuracy (McNemar)',
                     pv(F['acc_minus_fixed']['p_mcnemar']), vd(F['acc_minus_fixed']['p_mcnemar'])])
    if 'a2' in L:
        A = L['a2']; g = A['edits']['g_to_mean']; t4 = A['top4']; nu = A['null']['four_of_the_other_live']; na = A['null']['four_of_all_live']
        rows.append(['A2', 'g → mean: calls changed', pc(g['calls_changed']['est']), iv(g['calls_changed']['ci']), 'rerun unedited: 0', '0 (exact)', 'H0 rejected'])
        rows.append(['A2', 'S0, S30, S31, S58 → mean-context value: calls changed', pc(t4['calls_changed']['est']), iv(t4['calls_changed']['ci']),
                     f"4 of the other {len(nu['pool'])} live S ({nu['n_subsets']} subsets): max {pc(nu['null_cost_max'])}", pv(nu['p_cost']), vd(nu['p_cost'])])
        rows.append(['A2', 'same 4: rank among all 4-subsets of the live S objects', pc(t4['calls_changed']['est']), '—',
                     f"{na['n_subsets']} subsets: max {pc(na['null_cost_max'])}", pv(na['p_cost']), vd(na['p_cost'])])
        rows.append(['A2', 'same 4: same call as g → mean on the jets g → mean changes', pc(t4['same_as_g_to_mean_on_its_changed']['est'], 1),
                     iv(t4['same_as_g_to_mean_on_its_changed']['ci'], 1), f"other 4-subsets: max {pc(nu['null_agree_max'], 1)}", pv(nu['p_agree']), vd(nu['p_agree'])])
    if 'a3' in L:
        A = L['a3']
        for e in ('rot180', 'rot45'):
            r = A['edits'][e]; gc = r['control_gaussian_matched_rms']
            rows.append(['A3', f"{e.replace('rot', 'rotation ')}°: calls changed", pc(r['calls_changed']['est']), iv(r['calls_changed']['ci']),
                         f"rerun 0; matched-rms noise {gc['rate_min'] * 100:.2f}–{gc['rate_max'] * 100:.2f} %", pv(gc['edit_vs_draw0']['p_mcnemar']) + ' (vs noise)',
                         vd(gc['edit_vs_draw0']['p_mcnemar']) + ' (edit = noise)'])
        r = A['edits']['rot180']
        rows.append(['A3', 'rotation 180°: calls changed > 3 % (E5v threshold)', pc(r['calls_changed']['est']), iv(r['calls_changed']['ci']), 'H0 rate ≤ 3 %, binomial',
                     pv(r['p_gt_3pct']), vd(r['p_gt_3pct'])])
        q = A['cochran_q_r_preserving']
        rows.append(['A3', 'five r-preserving edits change the same fraction of calls', '—', '—', "Cochran's Q", pv(q['p']), vd(q['p'])])
        s = A['objects_summary']['max_r_preserving']; n = s['n_objects']
        rows.append(['A3', 'objects with rms shift / SD > 0.1 (largest of the five edits)', f"{len(s['above_holm01'])} of {n} (Holm 0.01)", '—',
                     'H0 ≤ 0.1 SD per object, bootstrap 10,000, Holm', '—', ', '.join(s['above_holm01'])[:60]])
        rows.append(['A3', 'objects with rms shift / SD < 0.1 (largest of the five edits)', f"{len(s['below_holm01'])} of {n} (Holm 0.01)", '—',
                     'H0 ≥ 0.1 SD per object', '—', 'unresolved: ' + (', '.join(s['unresolved05']) or 'none')[:60]])
    if 'a9' in L:
        for thr, lab in (('09', 'R² ≥ 0.9'), ('07', 'R² ≥ 0.7')):
            e = L['a9']['spearman']['all'][thr]
            rows.append(['A9', f'Spearman ρ depth vs first stage ({lab}; 45 observables)', f"{e['rho']:.2f}",
                         f"obs. [{e['ci_boot_observables'][0]:.2f}, {e['ci_boot_observables'][1]:.2f}]; jets+obs. [{e['ci_boot_jets_and_observables'][0]:.2f}, {e['ci_boot_jets_and_observables'][1]:.2f}]",
                         'depth labels permuted (10,000)', pv(e['p_perm_one_sided']), vd(e['p_perm_one_sided'])])
    if 'b2' in L:
        P = L['b2']['programs']['e11_nested']
        rows.append(['B2', 'nested program: same call as the network (fresh jets)', pc(P['same_call']['est']), iv(P['same_call']['ci']), '—', '—', '—'])
        rows.append(['B2', 'accuracy program − network', pc(P['acc_minus_network']['diff']), iv(P['acc_minus_network']['ci']), 'H0 equal (McNemar)',
                     pv(P['acc_minus_network']['p_mcnemar']), vd(P['acc_minus_network']['p_mcnemar'])])
        w = P['rejection50']['W vs rest']
        rows.append(['B2', 'W-vs-rest rejection at 50 % eff., program / network', '—' if w['ratio'] is None else f"{w['ratio']:.2f}×",
                     '—' if w['ci'][0] is None else f"[{w['ci'][0]:.2f}, {w['ci'][1] if w['ci'][1] is not None else float('inf'):.2f}]×",
                     f"H0 ratio = 1 (bootstrap 2,000); {w['bkg_passing_network']} bkg jets pass", pv(w['p_ratio_eq_1']), vd(w['p_ratio_eq_1'])])
        nv = L['b2']['nested_vs_direct_same_call']['e11_nested vs e11_z_on_physics']
        rows.append(['B3', 'unedited jets: same call, nested − direct formula', pc(nv['diff']), iv(nv['ci']), 'H0 equal (McNemar)', pv(nv['p_mcnemar']), vd(nv['p_mcnemar'])])
    if 'b3' in L:
        nv = L['b3']['nested_vs_direct']['shift_eta_0p06']['e11_nested vs e11_z_on_physics']
        rows.append(['B3', 'η shift 0.06: same call, nested − direct formula', pc(nv['diff'], 1), iv(nv['ci'], 1),
                     f"nested {pc(nv['a']['est'], 1)}, direct {pc(nv['b']['est'], 1)}; McNemar", pv(nv['p_mcnemar']), vd(nv['p_mcnemar'])])
        w = L['b3']['flows']['shift_eta_0p06']['W->Z']
        rows.append(['B3', 'η shift 0.06: W → Z, nested program − network', pc(w['e11_nested_minus_network']['diff'], 1), '—',
                     f"network {pc(w['network']['est'], 1)}, nested {pc(w['e11_nested']['est'], 1)}, direct {pc(w['e11_z_on_physics']['est'], 1)}",
                     pv(w['e11_nested_minus_network']['p_mcnemar']), vd(w['e11_nested_minus_network']['p_mcnemar'])])
    if 'small' in L:
        S = L['small']
        for a in ('linf-2b-d32-k8', 'dsets-d64'):
            if a not in S['architectures']: continue
            r = S['architectures'][a]
            for sd in ('seed0', 'seed1'):
                n = r[sd]; m = S['runs'][n]['minus_jedi']
                rows.append(['small', f'{a} {sd} − JEDI-linear accuracy (260,000 jets)', pc(m['diff'], 3), iv(m['ci'], 3), 'H0 equal (McNemar)', pv(m['p_mcnemar']), vd(m['p_mcnemar'])])
            m = r['seed1_minus_seed0']
            rows.append(['small', f'{a} seed 1 − seed 0', pc(m['diff'], 3), iv(m['ci'], 3), 'H0 equal (McNemar)', pv(m['p_mcnemar']), vd(m['p_mcnemar'])])
    cols = ['claim', 'quantity', 'estimate', '95 % interval', 'null / control', 'p-value', 'verdict at 0.01']
    widths = [0.55, 4.6, 1.5, 2.6, 3.6, 0.95, 2.2]
    fig, ax = plt.subplots(figsize=(sum(widths), 0.9 + 0.36 * len(rows)), dpi=160); ax.axis('off')
    t = ax.table(cellText=rows, colLabels=cols, colWidths=[w / sum(widths) for w in widths], loc='upper center', cellLoc='left', colLoc='left')
    t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.55)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor('#d8d4ca'); cell.PAD = 0.02
        if r == 0: cell.set_facecolor('#eeebe3'); cell.set_text_props(weight='bold')
        elif r % 2 == 0: cell.set_facecolor('#f8f7f3')
    ax.set_title('Claim statistics (R2): JEDI-linear claims A1, A2, A3, A9, B2, B3 and the small taggers; fresh jets (sample seed 61)', loc='left', fontsize=10, weight='bold', pad=6)
    SITE.mkdir(parents=True, exist_ok=True); p = SITE / f'T14_claim_stats{"_smoke" if SMOKE else ""}.png'
    fig.tight_layout(); fig.savefig(p, bbox_inches='tight'); log('wrote', p)
    (OUT / f'r2_table_rows{"_smoke" if SMOKE else ""}.json').write_text(json.dumps(dict(cols=cols, rows=rows), indent=1, ensure_ascii=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--part', required=True, choices=['a1units', 'a1float', 'a2', 'a3', 'a9', 'b2', 'b3', 'small', 'table'])
    ap.add_argument('--smoke', action='store_true')
    A = ap.parse_args(); SMOKE = A.smoke; NBOOT = 200 if SMOKE else 2000
    log('args', vars(A))
    globals()[f'part_{A.part}']()
    log('done')
