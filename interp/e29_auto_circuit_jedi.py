"""E29 (claims A1, A2, A3, A5, A6; exploratory method; Paper B baseline): automated circuit discovery on JEDI-linear.

Question: do two standard automated circuit-discovery methods (edge attribution patching, EAP; greedy edge pruning in the
style of ACDC) recover the units that the hand-designed tests named (13 live l1 cuts, 17 hinge S objects, context set
S0/S30/S31/S58), and where does the hand-named program sit on the algorithm's faithfulness-vs-size curve?

PLAN (fixed before the first run)
Sample: fit / discovery = interp.jedi_core.sample(1000, 2901) (5,000 jets; ACDC uses its [::5] rows, 1,000 jets);
  test = sample(1600, 2902) (8,000 jets) minus any jet byte-identical to a fit jet; random-circuit nulls on test[::8].
Network: official JEDI-linear 3-feature n64 (jedi_core.Jedi); controls: Jedi(64, untrained=True, seed 0..2), Jedi(32),
  Jedi(64, '3-feature-perminv').
Graph: nodes = live units (activation s.d. > 1e-9 on the fit jets) of a (l1 cuts, per particle), P, g, s, u (per particle;
  one node per channel, all particles together), S, h4, h5, z, and the 5 logits. Edges = nonzero fused-kernel entries
  between live units of consecutive stages: a->P (identity, pooling), a->s (l2), P->g (dd), s->u and g->u (l3 acting on
  the quantized s + g sum), u->S (identity, pooling), S->h4, h4->h5, h5->z, z->logit.
Edge-masked forward (own JAX re-implementation with the network's quantizers and fused kernels): the receiver of edge
  i->j reads q(clean_i) if the edge is on and q(corrupt_i) if off (q = the receiver layer's input quantizer). For l3 the
  quantizer acts on the sum s_k + g_k, so the four on/off combinations of (s_k->u_m, g_k->u_m) are quantized separately
  (exact). Checks: all edges on = jedi_core.Jedi.run (max |dlogit|); node-level ablation via masks = Jedi.run_jetwise
  with per-slot mean overrides.
Corruptions: (mean) every unit at its fit-sample mean (per particle slot for a, s, u); (resample) every unit at its value
  on a donor jet: W<->Z, q<->g, t -> W or Z, donor = one of the 5 nearest in jet pT (sum of particle pT) of the partner
  class(es) in the same sample (rng 2903).
Method 1 EAP (integrated-gradient form, 5 steps alpha = 0.1..0.9, masks 1 - alpha on all graph edges): attribution A_e =
  mean over alpha of d m / d M_e, m = mean log p(clean call). Completeness: sum_e A_e vs m(clean) - m(all corrupted).
  Plain EAP (gradient at alpha = 0) reported as Spearman with A. Check of A on 40 edges (20 top, 20 random) against the
  exact metric change from removing that edge alone.
  Circuits: top-k edges by |A|, k in KS; edge circuit pruned to edges on a path to the logits; node circuit = endpoints.
  Node ranking: node score = sum of A over its outgoing edges; top-n nodes, n in NS and at the hand circuits' sizes.
Method 2 ACDC-style: receivers in reverse topological order (logits, z, h5, h4, S, u, g, s, P); for each, its incoming
  edges in ascending |A|; remove an edge if KL(clean || circuit) rises by < tau; tau ladder TAUS ascending, each tau
  continuing from the previous circuit (nested circuits); KL on the 1,000 ACDC jets.
Literal circuit (node level): only its units live; every other unit at the corrupted value (mean / donor), evaluated on
  the test jets; also the edge-level circuit (only its edges on).
Metrics: accuracy, F = (acc_circ - acc_0) / (acc_full - acc_0) (acc_0 = every unit corrupted), F_call (same with
  same-call), KL, 5 x 5 confusion, W->Z flow (share of true W called Z, circuit minus full network), q->g flow.
  Jaccard of the circuit's a and S sets with NAMED_L1 (13), NAMED_S17, NAMED_S19 and CONTEXT.
Hand circuits (node level): handA = 13 l1 cuts + live P, g, s (E1) + 17 named S (u same channels) + h4 / h5 with E11
  held-out R^2 >= 0.9 + z with R^2 >= 0.9 or a named mass window (z0, z7, z10, z14); handB = handA with S31, S58 (19 S);
  handC = every E1-live unit (the whole hand-described network).
Controls: (a) NRAND random node circuits per found node circuit, same number of units per stage, drawn from live units
  (and NRAND_ALL drawn from all 64 / width units, mean corruption); Jaccard nulls from 500 size-matched live draws;
  (b) hand circuits scored with the same F; (c) the full pipeline on the control networks.
Stats: 1,000-replicate bootstrap over test jets (95 % percentile); permutation p = (1 + #null >= obs) / (1 + n_null);
  McNemar (exact) on paired correctness, found circuit vs hand circuit at matched node count; Holm across the 8 sizes x 2
  corruptions per test family.
Usage: python -m interp.e29_auto_circuit_jedi [--smoke] [--plot]  ->  $AI_RESULTS/jedi/e29_auto_circuit[_smoke].json, .png
"""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np

os.environ.setdefault('KERAS_BACKEND', 'jax')
OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'jedi'
CL = ['g', 'q', 'W', 'Z', 't']
T0 = time.time()

NAMED_L1 = [3, 6, 15, 21, 29, 30, 32, 36, 40, 43, 46, 56, 57]
NAMED_S19 = [0, 1, 13, 17, 19, 20, 29, 30, 31, 40, 43, 50, 51, 52, 56, 58, 60, 62, 63]
NAMED_S17 = [c for c in NAMED_S19 if c not in (31, 58)]
CONTEXT = [0, 30, 31, 58]
E1_LIVE = dict(P=[3, 6, 15, 29, 30, 32, 36, 40, 43, 46, 56, 57], g=[4, 5, 29, 32, 36, 39, 44, 47, 49, 55, 63],
               s=[6, 14, 18, 32, 43, 52, 53, 57, 60],
               h4=[1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 21, 22, 24, 25, 26, 27, 28, 29, 30, 32, 34, 35, 36,
                   37, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
               h5=list(range(32)), z=list(range(16)))
E11_R2 = dict(h4=[1, 2, 3, 4, 6, 9, 10, 14, 16, 17, 18, 22, 24, 25, 27, 28, 30, 34, 35, 39, 41, 42, 44, 45, 46, 47, 48, 49, 50,
                  51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 63],
              h5=[0, 1, 2, 3, 5, 8, 10, 11, 13, 14, 16, 17, 18, 19, 20, 22, 24, 25, 26, 29, 31], z=[0, 1, 2, 8, 9, 12])
Z_WINDOWS = [0, 7, 10, 14]
STG = ['a', 'P', 'g', 's', 'u', 'S', 'h4', 'h5', 'z', 'logits']
# mask layers: name, source stage, receiver stage, kernel layer (None = identity pooling)
MLAY = [('aP', 'a', 'P', None), ('as', 'a', 's', 'l2'), ('Pg', 'P', 'g', 'dd'), ('su', 's', 'u', 'l3'), ('gu', 'g', 'u', 'l3'),
        ('uS', 'u', 'S', None), ('Sh4', 'S', 'h4', 'd4'), ('h4h5', 'h4', 'h5', 'd5'), ('h5z', 'h5', 'z', 'd6'), ('zl', 'z', 'logits', 'd7')]
BODY_B = ('aP', 'as', 'Pg'); BODY_C = ('su', 'gu', 'uS'); HEAD = ('Sh4', 'h4h5', 'h5z', 'zl')


def log(*a):
    print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


def jac(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else float('nan')


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj.tolist()


def mcnemar(a, b):
    """exact two-sided McNemar on paired booleans (a correct, b correct)"""
    n10 = int((a & ~b).sum()); n01 = int((~a & b).sum()); n = n10 + n01
    if n == 0: return dict(a_only=0, b_only=0, p=1.0)
    k = min(n10, n01); p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2
    return dict(a_only=n10, b_only=n01, p=float(min(1.0, p)))


# ======================================================================================================== masked network
class MNet:
    def __init__(self, n=64, family='3-feature', untrained=False, seed=0):
        import jax, jax.numpy as jnp
        from interp.jedi_core import Jedi
        self.jax, self.jnp = jax, jnp
        self.J = Jedi(n, family, untrained, seed); L = self.J.L; self.n = n
        self.KB = {k: (jnp.asarray(self.J.KB[k][0], jnp.float32), jnp.asarray(self.J.KB[k][1], jnp.float32)) for k in self.J.KB}
        self.W = dict(a=64, P=64, g=64, s=64, u=64, S=64, h4=64, h5=32, z=16, logits=5)
        self.orient, self.K2 = {}, {}
        for k in ('l2', 'dd', 'l3', 'd4', 'd5', 'd6', 'd7'):
            eq = L[k].equation.replace(' ', ''); lhs, out = eq.split('->'); xin, ker = lhs.split(',')
            K = np.asarray(self.J.KB[k][0]); ia, io = ker.index(xin[-1]), ker.index(out[-1])
            assert all(K.shape[d] == 1 for d in range(K.ndim) if d not in (ia, io)), (k, eq, K.shape)
            K2 = np.squeeze(K); K2 = K2 if ia < io else K2.T
            self.orient[k] = (ia < io, K.shape, eq); self.K2[k] = K2
        log(f'net n{n} {family} untrained={untrained} seed={seed}: equations', {k: v[2] for k, v in self.orient.items()},
            'kernels (in,out)', {k: v.shape for k, v in self.K2.items()})

        def tok(k, M):                                    # (in, out) mask -> kernel layout
            same, shape, _ = self.orient[k]
            return (M if same else M.T).reshape(shape)

        def mdense(k, xc, xr, M, act=True):
            q1 = L[k].iq(xc, training=False); q0 = L[k].iq(xr, training=False); K, b = self.KB[k]; Mk = tok(k, M)
            z = jnp.einsum(L[k].equation, q1, K * Mk) + jnp.einsum(L[k].equation, q0, K * (1 - Mk)) + b
            return L[k].activation(z) if act and L[k].activation is not None else z

        def mpool(k, xc, xr, m):
            v = m * L[k].iq(xc, training=False) + (1 - m) * L[k].iq(xr, training=False)
            return jnp.sum(v, axis=L[k].axes, keepdims=L[k].keepdims) * L[k].scale

        def stA(xin):
            K, b = self.KB['l1']; z = jnp.einsum(L['l1'].equation, L['l1'].iq(xin, training=False), K) + b
            return L['l1'].activation(z)

        def stB(a, C, M):
            P = mpool('qs', a, C['a'], M['aP'])
            g = mdense('dd', P, C['P'], M['Pg'])
            s = mdense('l2', a, C['a'], M['as'])
            return P, g, s

        def stC(s, g, C, M):
            q1s, q1g = L['add'].iq([s, g], training=False); q0s, q0g = L['add'].iq([C['s'], C['g']], training=False)
            K, b = self.KB['l3']; Ms, Mg = tok('l3', M['su']), tok('l3', M['gu']); e = L['l3'].equation
            iq = lambda t: L['l3'].iq(t, training=False)
            z = (jnp.einsum(e, iq(q1s + q1g), K * Ms * Mg) + jnp.einsum(e, iq(q1s + q0g), K * Ms * (1 - Mg))
                 + jnp.einsum(e, iq(q0s + q1g), K * (1 - Ms) * Mg) + jnp.einsum(e, iq(q0s + q0g), K * (1 - Ms) * (1 - Mg)) + b)
            u = L['l3'].activation(z)
            S = mpool('qs1', u, C['u'], M['uS'])
            return u, S

        def stD(S, C, M):
            h4 = mdense('d4', S, C['S'], M['Sh4']); h5 = mdense('d5', h4, C['h4'], M['h4h5'])
            z = mdense('d6', h5, C['h5'], M['h5z'], act=False)
            lo = mdense('d7', jax.nn.relu(z), jax.nn.relu(C['z']), M['zl'])
            return h4, h5, z, lo.reshape(S.shape[0], -1)

        def full(xin, C, M):
            a = stA(xin); P, g, s = stB(a, C, M); u, S = stC(s, g, C, M); h4, h5, z, lo = stD(S, C, M)
            return dict(a=a, P=P, g=g, s=s, u=u, S=S, h4=h4, h5=h5, z=z, logits=lo)

        self._stA, self._stB, self._stC, self._stD = (jax.jit(f) for f in (stA, stB, stC, stD))
        self._full = jax.jit(full)
        self._logits = jax.jit(lambda xin, C, M: full(xin, C, M)['logits'])

        def metric(M, xin, C, call):
            lp = jax.nn.log_softmax(full(xin, C, M)['logits'], axis=-1)
            return jnp.mean(jnp.take_along_axis(lp, call[:, None], axis=1))
        self._grad = jax.jit(jax.grad(metric))
        self._metric = jax.jit(metric)

    # ---- helpers
    def xin(self, x):
        return np.asarray(self.J.inputs(x, self.n), np.float32)

    def ones(self):
        M = {}
        for nm, src, dst, k in MLAY:
            M[nm] = np.ones((self.W[src],), np.float32) if k is None else np.ones(self.K2[k].shape, np.float32)
        return M

    def dummyC(self, xb):
        """zero corrupted activations of the internal (broadcastable) shapes; traced once"""
        if not hasattr(self, '_Z0'):
            n = self.n; z = lambda *sh: np.zeros(sh, np.float32)
            Z = dict(a=z(1, n, 64), P=z(1, 1, 64), g=z(1, 1, 64), s=z(1, n, 64), u=z(1, n, 64), S=z(1, 64), h4=z(1, 64), h5=z(1, 32), z=z(1, 16))
            sh = self.jax.eval_shape(self._full, xb, Z, self.ones())
            self._Z0 = {k: np.zeros((1,) + tuple(sh[k].shape[1:]), np.float32) for k in Z}
            log('internal shapes', {k: tuple(sh[k].shape) for k in sh})
        return self._Z0


# ======================================================================================================== graph
class Graph:
    def __init__(self, net, live):
        self.net, self.live = net, live; self.edges = []; self.G = {}
        for nm, src, dst, k in MLAY:
            if k is None:
                g = np.zeros((net.W[src],), np.float32)
                for i in live[src]:
                    g[i] = 1; self.edges.append((nm, i, i))
            else:
                K2 = net.K2[k]; g = np.zeros(K2.shape, np.float32); dl = live[dst] if dst != 'logits' else list(range(5))
                for i in live[src]:
                    for j in dl:
                        if K2[i, j] != 0: g[i, j] = 1; self.edges.append((nm, i, j))
            self.G[nm] = g
        self.src = {nm: s for nm, s, d, k in MLAY}; self.dst = {nm: d for nm, s, d, k in MLAY}
        self.eix = {e: t for t, e in enumerate(self.edges)}
        self.nodes = [(st, i) for st in STG[:-1] for i in live[st]] + [('logits', j) for j in range(5)]
        log(f'graph: {len(self.nodes)} nodes, {len(self.edges)} edges; per layer', {nm: int(self.G[nm].sum()) for nm in self.G},
            'live per stage', {st: len(live[st]) for st in STG[:-1]})

    def src_node(self, e):
        return (self.src[e[0]], e[1])

    def dst_node(self, e):
        return (self.dst[e[0]], e[2])

    def masks_from_edges(self, on):
        """on: iterable of edge indices kept; every other graph edge off"""
        M = {nm: 1 - self.G[nm] for nm in self.G}
        for t in on:
            nm, i, j = self.edges[t]
            if M[nm].ndim == 1: M[nm][i] = 1
            else: M[nm][i, j] = 1
        return M

    def masks_from_nodes(self, nodes):
        """literal node circuit: a graph edge is on iff its source unit is in the set"""
        ns = set(nodes); M = {nm: 1 - self.G[nm] for nm in self.G}
        for nm, i, j in self.edges:
            if (self.src[nm], i) in ns:
                if M[nm].ndim == 1: M[nm][i] = 1
                else: M[nm][i, j] = 1
        return M

    def prune(self, on):
        """keep edges on a path to the logits (receiver reaches the logits through kept edges)"""
        on = set(on); reach = {('logits', j) for j in range(5)}
        for st in reversed(STG[:-1]):           # nodes at stage st reach logits if an outgoing kept edge reaches
            for t in on:
                e = self.edges[t]
                if self.src[e[0]] == st and self.dst_node(e) in reach: reach.add(self.src_node(e))
        return sorted(t for t in on if self.dst_node(self.edges[t]) in reach)

    def nodes_of(self, on):
        ns = set()
        for t in on:
            e = self.edges[t]; ns.add(self.src_node(e)); ns.add(self.dst_node(e))
        return sorted(n for n in ns if n[0] != 'logits')


def stage_sets(nodes):
    d = {st: sorted(i for s, i in nodes if s == st) for st in STG[:-1]}
    return d


# ======================================================================================================== data
def load(per_fit, seed_fit, per_test, seed_test):
    from interp.jedi_core import sample
    xf, yf, Qf = sample(per_fit, seed_fit); xt, yt, Qt = sample(per_test, seed_test)
    hf = {hash(r.tobytes()) for r in xf}; keep = np.array([hash(r.tobytes()) not in hf for r in xt])
    log(f'fit sample({per_fit}, {seed_fit}) N {len(xf)}; test sample({per_test}, {seed_test}) N {len(xt)}, '
        f'{int((~keep).sum())} jets shared with fit removed')
    ptt = np.asarray(Qt['sum_pt'] if 'sum_pt' in Qt else xt[..., 0].sum(1))[keep]; xt, yt = xt[keep], yt[keep]
    ptf = Qf['sum_pt'] if 'sum_pt' in Qf else xf[..., 0].sum(1)
    return xf, yf, np.asarray(ptf, np.float64), xt, yt, np.asarray(ptt, np.float64)


def donors(pt, y, rng):
    """W<->Z, q<->g, t -> W or Z; one of the 5 nearest in jet pT among the partner class(es)"""
    part = {0: [1], 1: [0], 2: [3], 3: [2], 4: [2, 3]}; don = np.zeros(len(y), int)
    for c, pc in part.items():
        ia = np.flatnonzero(y == c); ib = np.flatnonzero(np.isin(y, pc)); o = ib[np.argsort(pt[ib])]; ps = pt[o]
        for i in ia:
            j = np.searchsorted(ps, pt[i]); lo, hi = max(0, j - 5), min(len(o), j + 5)
            cand = o[lo:hi]; cand = cand[np.argsort(np.abs(pt[cand] - pt[i]))[:5]]; don[i] = rng.choice(cand)
    return don


# ======================================================================================================== evaluation
class Evaluator:
    """logits of the masked network on a jet set under one corruption (mean: fixed C; resample: activations of the aligned
    donor jets xdon, xdon[i] = donor of jet i)"""
    def __init__(self, net, xin, corr, Cmean, xdon=None, batch=1000, cache=False):
        self.net, self.xin, self.corr, self.Cmean, self.xdon, self.batch = net, xin, corr, Cmean, xdon, batch
        self.cacheC = None
        if corr == 'resample' and cache:
            self.cacheC = [self.donorC(i) for i in range(0, len(xin), batch)]

    def donorC(self, i, size=None):
        r = self.net._full(self.xdon[i:i + (size or self.batch)], self.Cmean, self.net.ones())
        return {k: np.asarray(r[k]) for k in ('a', 'P', 'g', 's', 'u', 'S', 'h4', 'h5', 'z')}

    def C(self, b, i):
        if self.corr == 'mean': return self.Cmean
        return self.cacheC[b] if self.cacheC is not None else self.donorC(i)

    def logits(self, M):
        out = []
        for b, i in enumerate(range(0, len(self.xin), self.batch)):
            out.append(np.asarray(self.net._logits(self.xin[i:i + self.batch], self.C(b, i), M), np.float64))
        return np.concatenate(out)


def softmax(lo):
    e = np.exp(lo - lo.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)


def kl(p, lo):
    lq = lo - lo.max(1, keepdims=True); lq = lq - np.log(np.exp(lq).sum(1, keepdims=True))
    return float(np.mean(np.sum(p * (np.log(np.clip(p, 1e-30, None)) - lq), 1)))


class Scorer:
    def __init__(self, y, lo_full, lo_zero, B, rng):
        self.y, self.call0 = y, lo_full.argmax(1); self.p0 = softmax(lo_full)
        self.cf = self.call0 == y; self.cz = lo_zero.argmax(1) == y; self.sz = lo_zero.argmax(1) == self.call0
        self.bi = rng.integers(0, len(y), (B, len(y)))
        self.acc_full, self.acc_zero = float(self.cf.mean()), float(self.cz.mean())
        self.kl_zero = kl(self.p0, lo_zero)

    def score(self, lo, keep_vec=False):
        c = lo.argmax(1); cc = c == self.y; sc = c == self.call0
        F = (cc.mean() - self.acc_zero) / (self.acc_full - self.acc_zero) if self.acc_full != self.acc_zero else float('nan')
        Fc = (sc.mean() - self.sz.mean()) / (1 - self.sz.mean()) if self.sz.mean() < 1 else float('nan')
        bF = (cc[self.bi].mean(1) - self.cz[self.bi].mean(1)) / np.where(
            (self.cf[self.bi].mean(1) - self.cz[self.bi].mean(1)) == 0, np.nan, self.cf[self.bi].mean(1) - self.cz[self.bi].mean(1))
        conf = np.zeros((5, 5), int); np.add.at(conf, (self.y, c), 1)
        conf0 = np.zeros((5, 5), int); np.add.at(conf0, (self.y, self.call0), 1)
        nW, nq = max(1, conf[2].sum()), max(1, conf[1].sum())
        r = dict(acc=float(cc.mean()), acc_ci=np.nanpercentile(cc[self.bi].mean(1), [2.5, 97.5]).tolist(),
                 F=float(F), F_ci=np.nanpercentile(bF, [2.5, 97.5]).tolist(), F_call=float(Fc), same_call=float(sc.mean()),
                 kl=kl(self.p0, lo), F_kl=float(1 - kl(self.p0, lo) / self.kl_zero) if self.kl_zero > 1e-9 else float('nan'),
                 confusion=conf.tolist(),
                 WtoZ_flow=float((conf[2, 3] - conf0[2, 3]) / nW), qtog_flow=float((conf[1, 0] - conf0[1, 0]) / nq))
        if keep_vec: r['_cc'] = cc
        return r


def overlap(nodes):
    d = stage_sets(nodes)
    return dict(n_a=len(d['a']), n_S=len(d['S']), J_l1=jac(d['a'], NAMED_L1), J_S17=jac(d['S'], NAMED_S17),
                J_S19=jac(d['S'], NAMED_S19), ctx_found=sorted(set(d['S']) & set(CONTEXT)), J_ctx=jac(d['S'], CONTEXT))


def overlap_null(nodes, live, rng, n=500):
    """size-matched random draws from live units (and from all 64) -> Jaccard null and permutation p"""
    d = stage_sets(nodes); res = {}
    for key, st, ref in (('J_l1', 'a', NAMED_L1), ('J_S17', 'S', NAMED_S17), ('J_S19', 'S', NAMED_S19), ('J_ctx', 'S', CONTEXT)):
        k = len(d[st]); obs = jac(d[st], ref)
        for pool_nm, pool in (('live', live[st]), ('all64', list(range(64)))):
            if k == 0 or k > len(pool): res[f'{key}_{pool_nm}'] = dict(obs=obs, k=k, p=float('nan')); continue
            nl = np.array([jac(rng.choice(pool, k, replace=False), ref) for _ in range(n)])
            res[f'{key}_{pool_nm}'] = dict(obs=obs, k=k, null_mean=float(nl.mean()), null_95=float(np.percentile(nl, 95)),
                                           p=float((1 + (nl >= obs - 1e-12).sum()) / (1 + n)))
    return res


def random_nodes(nodes, live, rng, unrestricted=False):
    d = stage_sets(nodes); out = []
    for st, us in d.items():
        pool = list(range(64 if st not in ('h5', 'z') else (32 if st == 'h5' else 16))) if unrestricted else live[st]
        if len(us): out += [(st, int(i)) for i in rng.choice(pool, min(len(us), len(pool)), replace=False)]
    return out


def hand_circuits(live):
    A = [('a', i) for i in NAMED_L1] + [(st, i) for st in ('P', 'g', 's') for i in E1_LIVE[st]]
    A += [('u', i) for i in NAMED_S17] + [('S', i) for i in NAMED_S17]
    A += [('h4', i) for i in E11_R2['h4']] + [('h5', i) for i in E11_R2['h5']] + [('z', i) for i in sorted(set(E11_R2['z']) | set(Z_WINDOWS))]
    B = A + [('u', 31), ('u', 58), ('S', 31), ('S', 58)]
    Cc = [('a', i) for i in NAMED_L1] + [(st, i) for st in ('P', 'g', 's', 'h4', 'h5', 'z') for i in E1_LIVE[st]]
    Cc += [(st, i) for st in ('u', 'S') for i in NAMED_S19]
    return dict(handA=sorted(set(A)), handB=sorted(set(B)), handC=sorted(set(Cc)))


# ======================================================================================================== one network
def eap(net, gr, xin, Cfn, call, steps, batch):
    """integrated-gradient edge attribution; Cfn(i) -> corrupted acts of batch i"""
    jnp = net.jnp; acc = {nm: np.zeros_like(gr.G[nm], np.float64) for nm in gr.G}; plain = {nm: np.zeros_like(gr.G[nm], np.float64) for nm in gr.G}
    alphas = [(k + 0.5) / steps for k in range(steps)]; nb = 0
    for i in range(0, len(xin), batch):
        C = Cfn(i); xb = xin[i:i + batch]; cb = jnp.asarray(call[i:i + batch], jnp.int32); w = len(xb) / len(xin)
        for al in alphas + [0.0]:
            M = {nm: (1 - al * gr.G[nm]).astype(np.float32) for nm in gr.G}
            g = net._grad(M, xb, C, cb)
            for nm in gr.G:
                v = np.asarray(g[nm], np.float64) * gr.G[nm] * w
                if al == 0.0: plain[nm] += v
                else: acc[nm] += v / steps
        nb += 1
    A = np.array([acc[nm][i] if acc[nm].ndim == 1 else acc[nm][i, j] for nm, i, j in gr.edges])
    A0 = np.array([plain[nm][i] if plain[nm].ndim == 1 else plain[nm][i, j] for nm, i, j in gr.edges])
    return A, A0


def metric_np(lo, call):
    lq = lo - lo.max(1, keepdims=True); lq = lq - np.log(np.exp(lq).sum(1, keepdims=True))
    return float(np.mean(lq[np.arange(len(call)), call]))


def acdc(net, gr, xin, C, p0, A, taus):
    """greedy edge pruning in reverse topological order with cached upstream stages; nested over ascending taus"""
    jnp = net.jnp; on = np.ones(len(gr.edges), bool)
    recv_order = ['logits', 'z', 'h5', 'h4', 'S', 'u', 'g', 's', 'P']
    inc = {}
    for t, e in enumerate(gr.edges): inc.setdefault(gr.dst_node(e), []).append(t)
    out_of = {}
    for t, e in enumerate(gr.edges): out_of.setdefault(gr.src_node(e), []).append(t)
    p0j = p0

    def masks():
        return gr.masks_from_edges(np.flatnonzero(on))

    def klv(lo):
        return kl(p0j, np.asarray(lo, np.float64))
    a = net._stA(xin); res = []; nev = 0
    for tau in taus:
        t1 = time.time()
        for rst in recv_order:
            M = masks()
            if rst in ('logits', 'z', 'h5', 'h4'):
                P, g, s = net._stB(a, C, M); u, S = net._stC(s, g, C, M); f = lambda M_: net._stD(S, C, M_)[3]
            elif rst in ('S', 'u'):
                P, g, s = net._stB(a, C, M); f = lambda M_: net._stD(net._stC(s, g, C, M_)[1], C, M_)[3]
            else:
                def f(M_):
                    P_, g_, s_ = net._stB(a, C, M_)
                    return net._stD(net._stC(s_, g_, C, M_)[1], C, M_)[3]
            cur = klv(f(M)); nev += 1
            recv = sorted({n for n in inc if n[0] == rst})
            for rn in recv:
                live_in = [t for t in inc[rn] if on[t]]
                if not live_in: continue
                has_out = rn[0] == 'logits' or any(on[t] for t in out_of.get(rn, []))
                if not has_out:
                    for t in live_in: on[t] = False
                    continue
                for t in sorted(live_in, key=lambda t: abs(A[t])):
                    on[t] = False; M2 = masks(); k2 = klv(f(M2)); nev += 1
                    if k2 - cur < tau: cur = k2
                    else: on[t] = True
        kept = gr.prune(np.flatnonzero(on))
        res.append(dict(tau=tau, edges=kept, kl_disc=cur, n_raw=int(on.sum())))
        log(f'  ACDC tau {tau:g}: {int(on.sum())} edges on, {len(kept)} after pruning, KL {cur:.4f}, {nev} evaluations, {time.time() - t1:.0f} s')
    return res


def run_network(tag, net, data, args, rng, full=True):
    xf, yf, ptf, xt, yt, ptt = data
    xinf, xint = net.xin(xf), net.xin(xt)
    net.dummyC(xinf[:1])
    # ---------------- exactness checks
    Mone = net.ones(); Z = net._Z0
    lo_mine = np.concatenate([np.asarray(net._logits(xint[i:i + 1000], Z, Mone), np.float64) for i in range(0, len(xint), 1000)])
    lo_ref = net.J.run(xt)['logits']
    acc_full = float((lo_ref.argmax(1) == yt).mean())
    chk = dict(max_abs_dlogit_all_on=float(np.abs(lo_mine - lo_ref).max()), calls_differ=int((lo_mine.argmax(1) != lo_ref.argmax(1)).sum()),
               acc_test=acc_full, acc_test_ci=None)
    log(f'[{tag}] unedited test accuracy {acc_full:.4f} (N {len(yt)}); own forward vs Jedi.run max |dlogit| {chk["max_abs_dlogit_all_on"]:.2e}, '
        f'calls differ {chk["calls_differ"]}')
    # ---------------- live units and means (fit jets)
    ssum, smax, smin = {}, {}, {}                    # streaming per-unit sum / max / min over fit jets (and slots)
    for i in range(0, len(xinf), 1000):
        r = net._full(xinf[i:i + 1000], Z, Mone)
        for k in ('a', 'P', 'g', 's', 'u', 'S', 'h4', 'h5', 'z'):
            v = np.asarray(r[k]); w = v.shape[-1]; f = v.reshape(-1, w)
            ssum[k] = ssum.get(k, 0) + v.astype(np.float64).sum(0, keepdims=True)
            smax[k] = np.maximum(smax.get(k, -np.inf), f.max(0)); smin[k] = np.minimum(smin.get(k, np.inf), f.min(0))
    Cmean = {k: (ssum[k] / len(xinf)).astype(np.float32) for k in ssum}
    live = {k: [int(c) for c in np.flatnonzero(smax[k] - smin[k] > 1e-9)] for k in ssum}
    gr = Graph(net, live)
    # node-level ablation check against jedi_core.run_jetwise (mean override, per slot), on 1,000 test jets
    rr = np.random.default_rng(29); hc = hand_circuits(live)
    chk_sets = dict(handA=hc['handA'], random=random_nodes(hc['handA'], live, rr))
    for nmc, nodes in chk_sets.items():
        Mn = gr.masks_from_nodes(nodes); xs = xt[:1000:1] if len(xt) <= 1000 else xt[::8][:1000]; xs_in = net.xin(xs)
        lo_m = np.asarray(net._logits(xs_in, Cmean, Mn), np.float64); J = len(xs); ns = set(nodes); O = {}
        for st in ('a', 'P', 'g', 's', 'u', 'S', 'h4', 'h5', 'z'):
            w = Cmean[st].shape[-1]; m = np.array([0.0 if (st, c) in ns else 1.0 for c in range(w)], np.float32)
            v = np.broadcast_to(Cmean[st], (J,) + Cmean[st].shape[1:]).reshape(J, *Cmean[st].shape[1:])
            if st in ('P', 'g', 'S', 'h4', 'h5', 'z'): v = v.reshape(J, -1)
            O[st] = (m, v)
        lo_j = net.J.run_jetwise(xs, O)['logits']
        chk[f'node_ablation_{nmc}'] = dict(max_abs_dlogit=float(np.abs(lo_m - lo_j).max()), calls_differ=int((lo_m.argmax(1) != lo_j.argmax(1)).sum()), N=J)
        log(f'[{tag}] node-ablation check ({nmc}, {len(nodes)} units live): masked forward vs Jedi.run_jetwise max |dlogit| '
            f'{chk[f"node_ablation_{nmc}"]["max_abs_dlogit"]:.2e}, calls differ {chk[f"node_ablation_{nmc}"]["calls_differ"]} / {J}')
    out = dict(tag=tag, checks=chk, live={k: v for k, v in live.items()}, n_nodes=len(gr.nodes), n_edges=len(gr.edges),
               edges_per_layer={nm: int(gr.G[nm].sum()) for nm in gr.G}, corruptions={})
    corrs = ['mean', 'resample']
    don_f = donors(ptf, yf, np.random.default_rng(2903)); don_t = donors(ptt, yt, np.random.default_rng(2904))
    sub_r = np.arange(0, len(xt), 8)
    call_f = net.J.run(xf)['logits'].argmax(1); call_t = lo_ref.argmax(1)
    for corr in corrs:
        t1 = time.time(); R = dict()
        evf = Evaluator(net, xinf, corr, Cmean, xinf[don_f], batch=args.batch)
        # ---------------- EAP-IG
        A, A0 = eap(net, gr, xinf, lambda i: Cmean if corr == 'mean' else evf.donorC(i, args.gbatch), call_f, args.steps, args.gbatch)
        Mz = {nm: (1 - gr.G[nm]).astype(np.float32) for nm in gr.G}
        lo_f_full = evf.logits(net.ones()); lo_f_zero = evf.logits(Mz)
        comp = dict(sum_A=float(A.sum()), m_clean_minus_m_corrupt=metric_np(lo_f_full, call_f) - metric_np(lo_f_zero, call_f))
        from scipy.stats import spearmanr
        comp['spearman_plain_vs_ig'] = float(spearmanr(np.abs(A), np.abs(A0))[0])
        # single-edge check: 20 top + 20 random edges, exact metric change on 1,000 fit jets
        if full:
            sel = list(np.argsort(-np.abs(A))[:20]) + list(rng.choice(len(A), 20, replace=False))
            ev1 = Evaluator(net, xinf[::5], corr, Cmean, xinf[don_f[::5]], batch=1000, cache=True)
            m_full = metric_np(ev1.logits(net.ones()), call_f[::5]); dm = []
            for t in sel:
                keep = np.ones(len(gr.edges), bool); keep[t] = False
                dm.append(m_full - metric_np(ev1.logits(gr.masks_from_edges(np.flatnonzero(keep))), call_f[::5]))
            comp['single_edge_check'] = dict(spearman=float(spearmanr(A[sel], dm)[0]), pearson=float(np.corrcoef(A[sel], dm)[0, 1]),
                                             A=[float(A[t]) for t in sel], exact=[float(v) for v in dm])
        log(f'[{tag}/{corr}] EAP-IG: sum A {comp["sum_A"]:.4f} vs m(clean) - m(corrupt) {comp["m_clean_minus_m_corrupt"]:.4f}; '
            f'Spearman |plain| vs |IG| {comp["spearman_plain_vs_ig"]:.3f}; single-edge check {comp.get("single_edge_check", {}).get("spearman")}')
        R['eap_completeness'] = comp
        order = np.argsort(-np.abs(A))
        R['top_edges'] = [dict(edge=list(map(lambda v: v if isinstance(v, str) else int(v), gr.edges[t])), A=float(A[t])) for t in order[:60]]
        # node scores
        nsc = {}
        for t, e in enumerate(gr.edges): nsc[gr.src_node(e)] = nsc.get(gr.src_node(e), 0.0) + A[t]
        node_rank = sorted(nsc, key=lambda n: -abs(nsc[n]))
        R['top_nodes'] = [dict(node=[n[0], int(n[1])], score=float(nsc[n])) for n in node_rank[:60]]
        # ---------------- test evaluation set-up
        evt = Evaluator(net, xint, corr, Cmean, xint[don_t], batch=args.batch)
        lo_t_full = evt.logits(net.ones()); lo_t_zero = evt.logits(Mz)
        sc = Scorer(yt, lo_t_full, lo_t_zero, args.boot, np.random.default_rng(2905))
        R['acc_full'] = sc.acc_full; R['acc_all_corrupted'] = sc.acc_zero
        R['acc_full_ci'] = np.percentile(sc.cf[sc.bi].mean(1), [2.5, 97.5]).tolist()
        log(f'[{tag}/{corr}] test: full {sc.acc_full:.4f} {R["acc_full_ci"]}, all corrupted {sc.acc_zero:.4f}')
        # ---------------- circuits
        circ = []
        for k in args.ks:
            if k > len(gr.edges): continue
            kept = gr.prune(order[:k]); nodes = gr.nodes_of(kept)
            se = sc.score(evt.logits(gr.masks_from_edges(kept))); sn = sc.score(evt.logits(gr.masks_from_nodes(nodes)), keep_vec=True)
            circ.append(dict(method='EAP_edges', k=k, n_edges=len(kept), n_nodes=len(nodes), edge_level=se, node_level=sn,
                             nodes=[[s, int(i)] for s, i in nodes], overlap=overlap(nodes), edges=[int(t) for t in kept]))
            log(f'[{tag}/{corr}] EAP top-{k} edges: {len(kept)} kept, {len(nodes)} units; edge-level F {se["F"]:.3f}, '
                f'node-level F {sn["F"]:.3f} {[round(v, 3) for v in sn["F_ci"]]} acc {sn["acc"]:.4f}; overlap {circ[-1]["overlap"]}')
        nsizes = sorted(set([n for n in args.ns if n <= len(node_rank)] + [len(v) for v in hc.values()]))
        for n in nsizes:
            nodes = sorted(node_rank[:n]); sn = sc.score(evt.logits(gr.masks_from_nodes(nodes)), keep_vec=True)
            circ.append(dict(method='EAP_nodes', k=n, n_nodes=n, n_edges=None, node_level=sn, nodes=[[s, int(i)] for s, i in nodes], overlap=overlap(nodes)))
            log(f'[{tag}/{corr}] EAP top-{n} units: F {sn["F"]:.3f} acc {sn["acc"]:.4f}; overlap {circ[-1]["overlap"]}')
        if corr == 'mean' or full:
            ia = np.arange(0, len(xinf), 5)
            Ca = Cmean if corr == 'mean' else Evaluator(net, xinf[ia], corr, Cmean, xinf[don_f[ia]]).donorC(0, len(ia))
            p0 = softmax(np.asarray(net._logits(xinf[ia], Ca, net.ones()), np.float64))
            for res in acdc(net, gr, xinf[ia], Ca, p0, A, args.taus):
                kept = res['edges']; nodes = gr.nodes_of(kept)
                se = sc.score(evt.logits(gr.masks_from_edges(kept))); sn = sc.score(evt.logits(gr.masks_from_nodes(nodes)), keep_vec=True)
                circ.append(dict(method='ACDC', k=res['tau'], n_edges=len(kept), n_nodes=len(nodes), edge_level=se, node_level=sn, kl_disc=res['kl_disc'],
                                 nodes=[[s, int(i)] for s, i in nodes], overlap=overlap(nodes), edges=[int(t) for t in kept]))
                log(f'[{tag}/{corr}] ACDC tau {res["tau"]:g}: {len(kept)} edges, {len(nodes)} units; edge-level F {se["F"]:.3f}, node-level F {sn["F"]:.3f} '
                    f'acc {sn["acc"]:.4f}; overlap {circ[-1]["overlap"]}')
        # hand circuits
        hand = {}
        for hn, nodes in hc.items():
            sn = sc.score(evt.logits(gr.masks_from_nodes(nodes)), keep_vec=True); hand[hn] = dict(n_nodes=len(nodes), node_level=sn, overlap=overlap(nodes))
            log(f'[{tag}/{corr}] hand circuit {hn} ({len(nodes)} units): F {sn["F"]:.3f} {[round(v, 3) for v in sn["F_ci"]]} acc {sn["acc"]:.4f} '
                f'same-call {sn["same_call"]:.4f}')
        # McNemar found vs hand at matched node count (EAP node ranking at exactly that size)
        for hn in hand:
            m = [c for c in circ if c['method'] == 'EAP_nodes' and c['n_nodes'] == hand[hn]['n_nodes']]
            if m:
                hand[hn]['eap_matched'] = dict(F=m[0]['node_level']['F'], acc=m[0]['node_level']['acc'],
                                               mcnemar=mcnemar(m[0]['node_level']['_cc'], hand[hn]['node_level']['_cc']),
                                               J_nodes=jac(map(tuple, m[0]['nodes']), hc[hn]))
                log(f'[{tag}/{corr}] {hn} vs EAP top-{hand[hn]["n_nodes"]} units: F {m[0]["node_level"]["F"]:.3f} vs {hand[hn]["node_level"]["F"]:.3f}; '
                    f'McNemar {hand[hn]["eap_matched"]["mcnemar"]}; unit Jaccard {hand[hn]["eap_matched"]["J_nodes"]:.3f}')
        # ---------------- nulls
        rnd = np.random.default_rng(2906 + (corr == 'resample'))
        evr = Evaluator(net, xint[sub_r], corr, Cmean, xint[don_t[sub_r]], batch=len(sub_r), cache=True)
        scr = Scorer(yt[sub_r], evr.logits(net.ones()), evr.logits(Mz), 200, np.random.default_rng(2907))
        for c in circ:
            c['overlap_null'] = overlap_null([tuple(v) for v in c['nodes']], live, rnd)
            if full and c['method'] in ('EAP_edges', 'ACDC') and args.nrand > 0 and c['n_nodes'] > 0:
                nodes = [tuple(v) for v in c['nodes']]; obs = scr.score(evr.logits(gr.masks_from_nodes(nodes)))['F']
                fl = np.array([scr.score(evr.logits(gr.masks_from_nodes(random_nodes(nodes, live, rnd))))['F'] for _ in range(args.nrand)])
                c['F_null_live'] = dict(obs_sub=obs, mean=float(np.nanmean(fl)), q95=float(np.nanpercentile(fl, 95)), max=float(np.nanmax(fl)),
                                        p=float((1 + (fl >= obs).sum()) / (1 + len(fl))), n=len(fl))
                if corr == 'mean' and args.nrand_all > 0:
                    fa = np.array([scr.score(evr.logits(gr.masks_from_nodes(random_nodes(nodes, live, rnd, True))))['F'] for _ in range(args.nrand_all)])
                    c['F_null_all64'] = dict(mean=float(np.nanmean(fa)), q95=float(np.nanpercentile(fa, 95)), p=float((1 + (fa >= obs).sum()) / (1 + len(fa))), n=len(fa))
                log(f'[{tag}/{corr}] null {c["method"]} {c["k"]}: F(sub) {obs:.3f} vs random live mean {c["F_null_live"]["mean"]:.3f} '
                    f'q95 {c["F_null_live"]["q95"]:.3f} p {c["F_null_live"]["p"]:.4f}' + (f'; all-64 mean {c["F_null_all64"]["mean"]:.3f}' if 'F_null_all64' in c else ''))
        for hn in hand:
            hand[hn]['overlap_null'] = overlap_null(hc[hn], live, rnd)
            if full and args.nrand > 0:
                obs = scr.score(evr.logits(gr.masks_from_nodes(hc[hn])))['F']
                fl = np.array([scr.score(evr.logits(gr.masks_from_nodes(random_nodes(hc[hn], live, rnd))))['F'] for _ in range(args.nrand)])
                hand[hn]['F_null_live'] = dict(obs_sub=obs, mean=float(np.nanmean(fl)), q95=float(np.nanpercentile(fl, 95)), p=float((1 + (fl >= obs).sum()) / (1 + len(fl))), n=len(fl))
        for c in circ: c['node_level'].pop('_cc', None)
        for hn in hand: hand[hn]['node_level'].pop('_cc', None)
        R['circuits'] = circ; R['hand'] = hand; R['wall_s'] = time.time() - t1
        out['corruptions'][corr] = R
        log(f'[{tag}/{corr}] done in {time.time() - t1:.0f} s')
    out['edges'] = [[nm, int(i), int(j)] for nm, i, j in gr.edges]
    return out


def holm_summary(res):
    """Holm across the 8 EAP edge sizes x 2 corruptions per test family (trained n64)"""
    fam = {'J_S17_live': [], 'J_l1_live': [], 'F_live': []}; keys = []
    for corr, R in res['corruptions'].items():
        for c in R['circuits']:
            if c['method'] != 'EAP_edges': continue
            keys.append((corr, c['k']))
            fam['J_S17_live'].append(c['overlap_null']['J_S17_live']['p']); fam['J_l1_live'].append(c['overlap_null']['J_l1_live']['p'])
            fam['F_live'].append(c.get('F_null_live', {}).get('p', float('nan')))
    out = {}
    for f, ps in fam.items():
        ps2 = [1.0 if (p is None or not np.isfinite(p)) else p for p in ps]
        out[f] = [dict(corr=k[0], k=k[1], p=p, p_holm=q) for k, p, q in zip(keys, ps, holm(ps2))]
    return out


# ======================================================================================================== figure
def plot(path_json, path_png):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    R = json.loads(Path(path_json).read_text()); T = R['networks']['n64']; Rm = T['corruptions']['mean']
    fig = plt.figure(figsize=(15, 9.5)); gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])
    ax = fig.add_subplot(gs[0, 0])
    for corr, col in (('mean', '#1f5fa8'), ('resample', '#c0602a')):
        C = T['corruptions'][corr]['circuits']
        for meth, mk, ls in (('EAP_edges', 'o', '-'), ('ACDC', 's', '--')):
            pts = sorted((c['n_nodes'], c['node_level']['F']) for c in C if c['method'] == meth and c['n_nodes'] > 0)
            if pts: ax.plot(*zip(*pts), marker=mk, ls=ls, color=col, label=f'{meth.replace("_edges", " top-k edges")} ({corr} corruption)')
        for hn, mk in (('handA', '*'), ('handB', 'P'), ('handC', 'X')):
            h = T['corruptions'][corr]['hand'][hn]; ax.plot(h['n_nodes'], h['node_level']['F'], mk, ms=13, color=col, mec='k')
        nl = sorted((c['n_nodes'], c['F_null_live']['q95'], c['F_null_live']['mean']) for c in C if c['method'] == 'EAP_edges' and 'F_null_live' in c)
        if nl and corr == 'mean':
            ax.fill_between([v[0] for v in nl], [-0.05] * len(nl), [v[1] for v in nl], color='0.8', alpha=0.6, label='random live units, same count per layer (95 % band)')
    ax.plot([], [], '*', color='0.3', ms=12, mec='k', label='hand program (13 cuts, 17 S, named h4/h5/z)')
    ax.plot([], [], 'P', color='0.3', ms=10, mec='k', label='hand program + S31, S58'); ax.plot([], [], 'X', color='0.3', ms=10, mec='k', label='all hand-described units')
    ax.set_xscale('log'); ax.set_xlabel('units kept live in the circuit (all others set to the corrupted value)')
    ax.set_ylabel('faithfulness F  (0 = all units corrupted, 1 = full network)'); ax.axhline(0.9, color='0.5', lw=0.8, ls=':')
    ax.set_title('Accuracy recovered vs circuit size (8,000 test jets)'); ax.legend(fontsize=7.5, loc='upper left'); ax.set_ylim(-0.05, 1.08)
    ax2 = fig.add_subplot(gs[0, 1])
    C = Rm['circuits']; E = sorted([c for c in C if c['method'] == 'EAP_edges'], key=lambda c: c['n_nodes'])
    ax2.plot([c['n_nodes'] for c in E], [c['overlap']['J_S17'] for c in E], 'o-', color='#1f5fa8', label='S units vs 17 named S objects')
    ax2.plot([c['n_nodes'] for c in E], [c['overlap_null']['J_S17_live'].get('null_95', np.nan) for c in E], ':', color='#1f5fa8', label='  random live S, 95 %')
    ax2.plot([c['n_nodes'] for c in E], [c['overlap']['J_l1'] for c in E], 's-', color='#2a9d4a', label='l1 cuts vs 13 named cuts')
    ax2.plot([c['n_nodes'] for c in E], [c['overlap_null']['J_l1_live'].get('null_95', np.nan) for c in E], ':', color='#2a9d4a', label='  random live cuts, 95 %')
    for key, col, lab in (('untrained0', '#999999', 'untrained redraw 0, S vs named 17'),):
        if key in R['networks']:
            E2 = sorted([c for c in R['networks'][key]['corruptions']['mean']['circuits'] if c['method'] == 'EAP_edges'], key=lambda c: c['n_nodes'])
            ax2.plot([c['n_nodes'] for c in E2], [c['overlap']['J_S17'] for c in E2], 'o--', color=col, label=lab)
    ax2.set_xscale('log'); ax2.set_ylim(-0.03, 1.05); ax2.set_xlabel('units kept live (EAP top-k edge circuits, mean corruption)')
    ax2.set_ylabel('Jaccard overlap with the hand-named set'); ax2.legend(fontsize=7.5); ax2.set_title('Do the found units coincide with the hand-named ones?')
    # layer diagram
    ax3 = fig.add_subplot(gs[1, :]); live = T['live']; stages = ['a', 'P', 'g', 's', 'u', 'S', 'h4', 'h5', 'z']
    fit = [c for c in E if c['node_level']['F'] >= 0.9]; c0 = fit[0] if fit else E[-1]
    found = {tuple(v) for v in c0['nodes']}
    hA = set()
    hA |= {('a', i) for i in NAMED_L1} | {(st, i) for st in ('P', 'g', 's') for i in E1_LIVE[st]} | {(st, i) for st in ('u', 'S') for i in NAMED_S17}
    hA |= {('h4', i) for i in E11_R2['h4']} | {('h5', i) for i in E11_R2['h5']} | {('z', i) for i in set(E11_R2['z']) | set(Z_WINDOWS)}
    pos = {}
    for xi, st in enumerate(stages):
        us = live[st]
        for r, u in enumerate(us): pos[(st, u)] = (xi, (r + 0.5) / max(1, len(us)))
    for j in range(5): pos[('logits', j)] = (len(stages), (j + 0.5) / 5)
    edges = T['edges']; SRC = {nm: s for nm, s, d, k in MLAY}; DST = {nm: d for nm, s, d, k in MLAY}
    for t in c0['edges']:
        nm, i, j = edges[t]; a, b = pos.get((SRC[nm], i)), pos.get((DST[nm], j))
        if a and b: ax3.plot([a[0], b[0]], [a[1], b[1]], color='0.55', lw=0.35, alpha=0.5, zorder=1)
    for n, (xx, yy) in pos.items():
        if n[0] == 'logits': ax3.scatter(xx, yy, s=60, c='k', zorder=3); ax3.text(xx + 0.08, yy, CL[n[1]], va='center', fontsize=9); continue
        f, h = n in found, n in hA
        col = '#7b3fa0' if f and h else '#1f5fa8' if f else '#e0a030' if h else '#dddddd'
        ax3.scatter(xx, yy, s=22 if n[0] in ('h4', 'h5') else 34, c=col, edgecolors='k', linewidths=0.3, zorder=3)
        if n[0] in ('a', 'S') and (f or h): ax3.text(xx - 0.12, yy, str(n[1]), ha='right', va='center', fontsize=6)
    lab = {'a': 'l1 cuts a\n(per particle)', 'P': 'P\n(pooled cuts)', 'g': 'g\n(jet context)', 's': 's\n(per particle)', 'u': 'u\n(per particle)',
           'S': 'S\n(second-round\nobjects)', 'h4': 'h4', 'h5': 'h5', 'z': 'z'}
    ax3.set_xticks(range(len(stages) + 1)); ax3.set_xticklabels([lab[s] for s in stages] + ['logits'], fontsize=8.5); ax3.set_yticks([])
    for col, l in (('#7b3fa0', 'found by EAP and hand-named'), ('#1f5fa8', 'found by EAP only'), ('#e0a030', 'hand-named only'), ('#dddddd', 'live, in neither')):
        ax3.scatter([], [], c=col, edgecolors='k', linewidths=0.3, s=34, label=l)
    ax3.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)
    ax3.set_title(f'Live units of JEDI-linear (n64): EAP circuit with {c0["n_edges"]} edges / {c0["n_nodes"]} units (smallest top-k with F >= 0.9, '
                  f'F {c0["node_level"]["F"]:.2f}) vs the hand-named program; grey lines = circuit edges', fontsize=10)
    for sp in ('top', 'right', 'left'): ax3.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(path_png, dpi=130); log('figure', path_png)


def plot_final(path_json, path_fu, path_png):
    """F vs units kept (found circuits, hand-program variants, random live circuits); units kept per layer; layer diagram of
    the 119-unit ACDC circuit (mean corruption, tau 3e-3) against the hand program with all z units"""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    R = json.loads(Path(path_json).read_text()); T = R['networks']['n64']; FU = json.loads(Path(path_fu).read_text())
    BL, OR, GR = '#1f5fa8', '#c0602a', '#2a9d4a'
    fig = plt.figure(figsize=(15, 10.5)); gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], width_ratios=[1.35, 1])
    ax = fig.add_subplot(gs[0, 0])
    band = []
    for corr, col in (('mean', BL), ('resample', OR)):
        C = T['corruptions'][corr]['circuits']
        for meth, mk, ls, lab in (('ACDC', 's', '-', 'ACDC-style pruning'), ('EAP_edges', 'o', ':', 'EAP, top-k edges')):
            pts = sorted((c['n_nodes'], c['node_level']['F']) for c in C if c['method'] == meth and c['n_nodes'] > 0)
            ax.plot(*zip(*pts), marker=mk, ls=ls, color=col, ms=5, label=f'{lab} ({corr} corruption)')
        if corr == 'mean':
            band = sorted((c['n_nodes'], c['F_null_live']['q95']) for c in C if c['method'] in ('ACDC', 'EAP_edges') and 'F_null_live' in c and c['n_nodes'] > 0)
    for tag, ls in (('n32', (0, (4, 2))), ('perminv', (0, (1, 1)))):
        Cc = R['networks'].get(tag, {}).get('corruptions', {}).get('mean', {}).get('circuits', [])
        pts = sorted((c['n_nodes'], c['node_level']['F']) for c in Cc if c['method'] == 'ACDC' and c['n_nodes'] > 0)
        if pts: ax.plot(*zip(*pts), ls=ls, color='0.45', lw=1.2, label=f'ACDC on {"JEDI-linear n32" if tag == "n32" else "perm.-invariant n64"} (mean)')
    xs, ys = zip(*band); ax.fill_between(xs, [-0.05] * len(xs), ys, color='0.85', zorder=0, label='random live units, same count per layer (95th pct.)')
    for vn, mk, lab in (('handA', '*', 'hand program: 13 cuts, 17 S, E11-named h4/h5/z (9 of 16 z)'), ('handA_allz', 'D', 'hand program with all 16 z units'),
                        ('handA_allhead', 'X', 'hand program with every live h4/h5/z unit')):
        for corr, col, fc in (('mean', BL, BL), ('resample', OR, 'none')):
            v = FU['corruptions'][corr]['variants'][vn]
            ax.errorbar(v['n_nodes'], v['F'], yerr=[[v['F'] - v['F_ci'][0]], [v['F_ci'][1] - v['F']]], fmt=mk, ms=11 if mk == '*' else 8, mfc=fc, mec=col, color=col, zorder=5)
        ax.plot([], [], mk, color='k', mfc='0.4', ms=9, label=lab)
    ax.axhline(0.9, color='0.5', lw=0.8, ls='--'); ax.set_xlim(0, 192); ax.set_ylim(-0.05, 1.06)
    ax.set_xlabel('units kept live (every other unit set to its mean, or to a donor jet\'s value)')
    ax.set_ylabel('faithfulness F  (0 = all units corrupted, 1 = full network)')
    ax.set_title('Accuracy recovered vs circuit size (JEDI-linear n64, 7,849 test jets)', fontsize=11)
    ax.legend(fontsize=7, loc='upper left', ncol=1, framealpha=0.9)
    # units per layer
    ax2 = fig.add_subplot(gs[0, 1]); stages = ['a', 'P', 'g', 's', 'S', 'h4', 'h5', 'z']; live = T['live']
    acdc = {corr: [c for c in T['corruptions'][corr]['circuits'] if c['method'] == 'ACDC' and c['k'] == 0.003][0] for corr in ('mean', 'resample')}
    hz = FU['corruptions']['mean']['variants']['handA_allz']['per_stage']
    rows = [('live units', [len(live[s]) for s in stages], '0.8'), ('hand program (all z)', [hz[s] for s in stages], '#e0a030'),
            (f'ACDC, mean ({acdc["mean"]["n_nodes"]} units)', [sum(1 for st, i in acdc['mean']['nodes'] if st == s) for s in stages], BL),
            (f'ACDC, resample ({acdc["resample"]["n_nodes"]} units)', [sum(1 for st, i in acdc['resample']['nodes'] if st == s) for s in stages], OR)]
    w = 0.2
    for r, (lab, v, col) in enumerate(rows):
        ax2.bar(np.arange(len(stages)) + (r - 1.5) * w, v, w, color=col, label=lab, edgecolor='k', lw=0.3)
    ax2.set_xticks(range(len(stages))); ax2.set_xticklabels(['l1 cuts', 'P', 'g\n(context)', 's', 'S', 'h4', 'h5', 'z'])
    ax2.set_ylabel('number of units'); ax2.legend(fontsize=8); ax2.set_title('Units kept per layer: ACDC (F >= 0.94) vs hand program', fontsize=11)
    # layer diagram
    ax3 = fig.add_subplot(gs[1, :]); st9 = ['a', 'P', 'g', 's', 'u', 'S', 'h4', 'h5', 'z']; c0 = acdc['mean']
    found = {tuple(v) for v in c0['nodes']}; nm = R['named']
    hA = {('a', i) for i in nm['l1']} | {(st, i) for st in ('P', 'g', 's') for i in nm['E1_live'][st]} | {(st, i) for st in ('u', 'S') for i in nm['S17']}
    hA |= {('h4', i) for i in nm['E11_r2_ge_0p9']['h4']} | {('h5', i) for i in nm['E11_r2_ge_0p9']['h5']} | {('z', i) for i in live['z']}
    pos = {}
    for xi, st in enumerate(st9):
        for r, u in enumerate(live[st]): pos[(st, u)] = (xi, (r + 0.5) / len(live[st]))
    for j in range(5): pos[('logits', j)] = (len(st9), (j + 0.5) / 5)
    edges = T['edges']; SRC = {n_: s_ for n_, s_, d_, k_ in MLAY}; DST = {n_: d_ for n_, s_, d_, k_ in MLAY}
    for t in c0['edges']:
        n_, i, j = edges[t]; a, b = pos.get((SRC[n_], i)), pos.get((DST[n_], j))
        if a and b: ax3.plot([a[0], b[0]], [a[1], b[1]], color='0.55', lw=0.4, alpha=0.55, zorder=1)
    for n, (xx, yy) in pos.items():
        if n[0] == 'logits': ax3.scatter(xx, yy, s=60, c='k', zorder=3); ax3.text(xx + 0.08, yy, CL[n[1]], va='center', fontsize=10); continue
        f, h = n in found, n in hA
        col = '#7b3fa0' if f and h else BL if f else '#e0a030' if h else '#dddddd'
        ax3.scatter(xx, yy, s=20 if n[0] in ('h4', 'h5') else 36, c=col, edgecolors='k', linewidths=0.3, zorder=3)
        if n[0] in ('a', 'g', 'S'): ax3.text(xx - 0.1, yy, str(n[1]), ha='right', va='center', fontsize=6.5)
    lab = {'a': 'l1 cuts\n(per particle)', 'P': 'P\n(pooled cuts)', 'g': 'g\n(jet context)', 's': 's\n(per particle)', 'u': 'u\n(per particle)',
           'S': 'S\n(second-round\nobjects)', 'h4': 'h4', 'h5': 'h5', 'z': 'z'}
    ax3.set_xticks(range(len(st9) + 1)); ax3.set_xticklabels([lab[s_] for s_ in st9] + ['logits'], fontsize=9); ax3.set_yticks([])
    for col, l in (('#7b3fa0', 'in ACDC circuit and in hand program'), (BL, 'ACDC circuit only'), ('#e0a030', 'hand program only'), ('#dddddd', 'live, in neither')):
        ax3.scatter([], [], c=col, edgecolors='k', linewidths=0.3, s=36, label=l)
    ax3.legend(fontsize=8.5, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=4, frameon=False)
    ax3.set_title(f'ACDC circuit (mean corruption, tau = 0.003): {c0["n_edges"]} edges, {c0["n_nodes"]} units, F = {c0["node_level"]["F"]:.2f}; '
                  f'hand program with all z: 156 units, F = 0.98. Grey lines = circuit edges.', fontsize=10.5)
    for sp in ('top', 'right', 'left'): ax3.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(path_png, dpi=125); log('figure', path_png)


# ======================================================================================================== follow-up
def followup(args):
    """hand-program variants that separate the l1 / S naming from the h4 / h5 / z choice, and paired McNemar tests of the
    found circuits (read from e29_auto_circuit.json) against the hand program; same samples, means, donors and bootstrap"""
    R = json.loads((OUT / 'e29_auto_circuit.json').read_text()); T = R['networks']['n64']
    data = load(1000, 2901, 1600, 2902); xf, yf, ptf, xt, yt, ptt = data
    net = MNet(64); xinf, xint = net.xin(xf), net.xin(xt); net.dummyC(xinf[:1]); Z = net._Z0; Mone = net.ones()
    ssum, smax, smin = {}, {}, {}
    for i in range(0, len(xinf), 1000):
        r = net._full(xinf[i:i + 1000], Z, Mone)
        for k in ('a', 'P', 'g', 's', 'u', 'S', 'h4', 'h5', 'z'):
            v = np.asarray(r[k]); f = v.reshape(-1, v.shape[-1])
            ssum[k] = ssum.get(k, 0) + v.astype(np.float64).sum(0, keepdims=True)
            smax[k] = np.maximum(smax.get(k, -np.inf), f.max(0)); smin[k] = np.minimum(smin.get(k, np.inf), f.min(0))
    Cmean = {k: (ssum[k] / len(xinf)).astype(np.float32) for k in ssum}
    live = {k: [int(c) for c in np.flatnonzero(smax[k] - smin[k] > 1e-9)] for k in ssum}
    assert live == {k: v for k, v in T['live'].items()}, 'live sets differ from the main run'
    gr = Graph(net, live); hc = hand_circuits(live); A = hc['handA']
    allz = [('z', i) for i in live['z']]; allh = [(st, i) for st in ('h4', 'h5', 'z') for i in live[st]]
    up = [n for n in A if n[0] not in ('h4', 'h5', 'z')]
    V = dict(handA=A, handA_allz=sorted(set(A) | set(allz)), handA_allhead=sorted(set(up) | set(allh)),
             handB_allhead=sorted(set(up) | set(allh) | {('u', 31), ('u', 58), ('S', 31), ('S', 58)}),
             handA_allhead_noctxS=sorted((set(up) | set(allh)) - {('u', 0), ('S', 0), ('u', 30), ('S', 30)}),
             handA_allhead_g49only=sorted((set(up) | set(allh)) - {('g', i) for i in live['g'] if i != 49}))
    don_t = donors(ptt, yt, np.random.default_rng(2904)); sub_r = np.arange(0, len(xt), 8); out = dict(variants={k: len(v) for k, v in V.items()}, corruptions={})
    for corr in ('mean', 'resample'):
        t1 = time.time(); evt = Evaluator(net, xint, corr, Cmean, xint[don_t], batch=args.batch)
        Mz = {nm: (1 - gr.G[nm]).astype(np.float32) for nm in gr.G}
        sc = Scorer(yt, evt.logits(net.ones()), evt.logits(Mz), args.boot, np.random.default_rng(2905))
        evr = Evaluator(net, xint[sub_r], corr, Cmean, xint[don_t[sub_r]], batch=len(sub_r), cache=True)
        scr = Scorer(yt[sub_r], evr.logits(net.ones()), evr.logits(Mz), 200, np.random.default_rng(2907)); rnd = np.random.default_rng(2909)
        res = dict(acc_full=sc.acc_full, acc_zero=sc.acc_zero, variants={}, found_vs_hand=[])
        cc = {}
        for vn, nodes in V.items():
            r = sc.score(evt.logits(gr.masks_from_nodes(nodes)), keep_vec=True); cc[vn] = r.pop('_cc')
            obs = scr.score(evr.logits(gr.masks_from_nodes(nodes)))['F']
            fl = np.array([scr.score(evr.logits(gr.masks_from_nodes(random_nodes(nodes, live, rnd))))['F'] for _ in range(args.nrand)])
            r['F_null_live'] = dict(obs_sub=obs, mean=float(np.nanmean(fl)), q95=float(np.nanpercentile(fl, 95)), p=float((1 + (fl >= obs).sum()) / (1 + len(fl))), n=len(fl))
            r['n_nodes'] = len(nodes); r['per_stage'] = {st: len(v) for st, v in stage_sets(nodes).items()}; res['variants'][vn] = r
            log(f'[followup/{corr}] {vn} ({len(nodes)} units): F {r["F"]:.3f} {[round(v, 3) for v in r["F_ci"]]} acc {r["acc"]:.4f} same-call {r["same_call"]:.4f} '
                f'W->Z {r["WtoZ_flow"]:+.3f} q->g {r["qtog_flow"]:+.3f}; random-live null mean {r["F_null_live"]["mean"]:.3f} q95 {r["F_null_live"]["q95"]:.3f} p {r["F_null_live"]["p"]:.4f}')
        for c in T['corruptions'][corr]['circuits']:
            if c['method'] not in ('ACDC', 'EAP_edges') or not (60 <= c['n_nodes'] <= 160): continue
            nodes = [tuple(v) for v in c['nodes']]; r = sc.score(evt.logits(gr.masks_from_nodes(nodes)), keep_vec=True); v = r.pop('_cc')
            for vn in ('handA', 'handA_allz', 'handA_allhead'):
                res['found_vs_hand'].append(dict(method=c['method'], k=c['k'], n_nodes=c['n_nodes'], F=r['F'], hand=vn, hand_n=len(V[vn]),
                                                 hand_F=res['variants'][vn]['F'], mcnemar=mcnemar(v, cc[vn]), J_units=jac(nodes, V[vn])))
                log(f'[followup/{corr}] {c["method"]} {c["k"]} ({c["n_nodes"]} units, F {r["F"]:.3f}) vs {vn} ({len(V[vn])} units, F {res["variants"][vn]["F"]:.3f}): '
                    f'McNemar {res["found_vs_hand"][-1]["mcnemar"]}, unit Jaccard {res["found_vs_hand"][-1]["J_units"]:.3f}')
        out['corruptions'][corr] = res; log(f'[followup/{corr}] {time.time() - t1:.0f} s')
    (OUT / 'e29_followup.json').write_text(json.dumps(out, default=float)); log('wrote', OUT / 'e29_followup.json')


# ======================================================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true'); ap.add_argument('--plot', action='store_true'); ap.add_argument('--followup', action='store_true')
    ap.add_argument('--steps', type=int, default=5); ap.add_argument('--batch', type=int, default=1000)
    ap.add_argument('--gbatch', type=int, default=500)
    ap.add_argument('--boot', type=int, default=1000); ap.add_argument('--nrand', type=int, default=200)
    ap.add_argument('--nrand-all', type=int, default=100)
    ap.add_argument('--controls', default='untrained0,untrained1,untrained2,n32,perminv')
    a = ap.parse_args()
    a.ks = [10, 20, 40, 80, 150, 250, 400, 800]; a.ns = [5, 10, 20, 30, 45, 60, 80, 100, 130, 160]
    a.taus = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
    sfx = '_smoke' if a.smoke else ''; OUT.mkdir(parents=True, exist_ok=True)
    pj, pp = OUT / f'e29_auto_circuit{sfx}.json', OUT / f'e29_auto_circuit{sfx}.png'
    if a.plot:
        fu = OUT / 'e29_followup.json'
        plot_final(pj, fu, pp) if fu.exists() and not a.smoke else plot(pj, pp); return
    if a.followup: followup(a); return
    if a.smoke:
        data = load(40, 2901, 40, 2902); a.nrand, a.nrand_all, a.boot = 5, 3, 100; a.ks = [10, 80, 400]; a.ns = [10, 60]
        a.taus = [1e-4, 1e-2]; a.controls = 'untrained0'
    else:
        data = load(1000, 2901, 1600, 2902)
    rng = np.random.default_rng(2908)
    res = dict(node='E29', plan=__doc__, sample=dict(fit=[1000, 2901], test=[1600, 2902], smoke=a.smoke), ks=a.ks, ns=a.ns, taus=a.taus,
               steps=a.steps, nrand=a.nrand, nrand_all=a.nrand_all, boot=a.boot, named=dict(l1=NAMED_L1, S17=NAMED_S17, S19=NAMED_S19, context=CONTEXT,
               E1_live=E1_LIVE, E11_r2_ge_0p9=E11_R2, z_windows=Z_WINDOWS), networks={})
    net = MNet(64)
    res['networks']['n64'] = run_network('n64', net, data, a, rng, full=True)
    res['holm'] = holm_summary(res['networks']['n64'])
    pj.write_text(json.dumps(res, default=float)); log('wrote', pj)
    try: plot(pj, pp)
    except Exception as ex: log('plot failed', repr(ex))
    for ctl in [c for c in a.controls.split(',') if c]:
        kw = dict(untrained0=dict(untrained=True, seed=0), untrained1=dict(untrained=True, seed=1), untrained2=dict(untrained=True, seed=2),
                  n32=dict(n=32), perminv=dict(family='3-feature-perminv'))[ctl]
        try:
            del net; import gc; gc.collect()
            net = MNet(**kw); res['networks'][ctl] = run_network(ctl, net, data, a, rng, full=False)
        except Exception as ex:
            import traceback; traceback.print_exc(); res['networks'][ctl] = dict(error=repr(ex)); net = None
        pj.write_text(json.dumps(res, default=float)); log('wrote', pj, 'after', ctl)
    try: plot(pj, pp)
    except Exception as ex: log('plot failed', repr(ex))
    log('done')


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
