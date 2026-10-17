"""E16 (JEDI-linear 3-feature n64; claim A11): neuron atlas, following Cammarata, "Neural Networks Are Already Clean"
(Goodfire, https://already-clean.goodfire.com): each live unit is described by the pattern it responds to (jet images and
observables, not the class labels), its weak activations, its negative pole (z), its upstream inputs, its downstream
outputs, and a steering test.

Sample: interp.jedi_core.sample(per_class=4000, seed=21) = 20,000 test jets.  Exact fixed-point network (Jedi(64)).
Live unit: mean ablation (the unit set to its mean over the sample: over jets for P, g, S, h4, h5, z; over real particles
for the first-layer cuts a) changes >= 0.1 % of the calls.  Every number below is measured on this sample.
Per live unit of P, g, S, h4, h5, z:
 1 activation: per-class histogram; on = value > 0 (z: pre-ReLU value > 0; the negative side is kept).
 2 activation bins by rank among the jets where the unit is on (all jets if it is on in < 200): top 1 %, 90-99 %, 50-90 %
   (>= 20 jets per bin); z also: the most negative 1 % of all jets.  Per bin: mean pT-weighted (Δη, Δφ) image and mean
   particle-count image (32 x 32 bins over ±0.4 rad, in the network's frame), the mean pT image after alignment (pT-weighted
   principal axis -> Δη', hardest particle at Δη' > 0, the other particles' pT-weighted Δφ' > 0); class mix; medians of
   10 observables (OBS10).
 3 description: each candidate observable (the test file's Q library, E11's hand-built pT-rank-window sums incl. the
   half-plane sums) as a 1-D response curve (20 quantile bins of the fit half; linear interpolation between the bins'
   (mean q, mean activation), linear beyond the end bins; R² on the other half of the jets); pairs among the 12 best distinct singles (|Spearman| <
   0.97) as the better of an additive model f1(q1) + f2(q2) (3 backfitting rounds) and an 8 x 8 table, same split.  Pattern line = the range of the best observable where its response curve is above half
   its span (plus the direction of the second observable when the pair adds >= 0.03 R²).
 4 main job vs offshoots: k-means on the 10 observables (clipped at their 1 / 99 % points, standardised over the on-jets)
   of the on-jets, fitted on 3,000 on-jets drawn with probability proportional to the activation (so the clusters follow
   the activation mass); k = 2..4 by silhouette on the same draw, k = 1 if the best silhouette < SIL_MIN.  Activation mass
   per cluster = Σ activation (positive part).  Each cluster with >= 10 % of the mass gets its own best observable
   (1-D binned R² inside the cluster) and |Spearman ρ| of that observable with the unit's best one over all jets.
   Category (in this order): unexplained if the best <= 2-observable R² < R2_MIN; one main job if a cluster holds >=
   MAIN_SHARE of the activation mass; related jobs if every cluster with >= 10 % of the mass has |ρ| >= RHO_SHARED;
   else unrelated jobs.
 5 upstream: |weight x spread| of each input as the layer receives it (after its input quantizer and zero-bit masks);
   S: the per-particle u table in (pT, r) and the pT-rank window; P: the l1 cut in GeV / rad.  downstream: next-stage
   units by |weight x spread|; the change of each logit when the unit is set to its 90th vs its 10th percentile on 2,000
   jets (percentiles over the on-jets when the two coincide over all jets).
 6 steering: the unit set to its 99th percentile on up to 4,000 jets where it is off (where it is on in > 99 % of jets:
   the jets below its 10th percentile); z also: set to its 1st percentile on jets where it is on.  Class flow (original
   call -> new call).  Control: a random other live unit of the same stage, same procedure.
The first-layer cuts a (13 live) get a reference card: the cut, passing-particle images, cost, downstream, steering.
Outputs ($AI_RESULTS/jedi/e16/): atlas_<stage>.json, images_<stage>.npz, card_<stage><unit>.png, atlas_summary.json/png.
Modes:
  run --stages z,h5 [--per-class 4000 --seed 21]    analysis + cards for those stages
  cards --stages z,h5                                 cards again from the JSON / npz (no network)
  summary                                            atlas_summary.json / .png from every atlas_<stage>.json
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'jedi' / os.environ.get('E16_DIR', 'e16')
CL = ['g', 'q', 'W', 'Z', 't']
ALL_STAGES = ('a', 'P', 'g', 'S', 'h4', 'h5', 'z')
JET_STAGES = ('P', 'g', 'S', 'h4', 'h5', 'z')
WIDTH = dict(a=64, P=64, g=64, S=64, h4=64, h5=32, z=16)
PREV = dict(P='a', g='P', S='u', h4='S', h5='h4', z='h5')
IN_LAYER = dict(g='dd', h4='d4', h5='d5', z='d6', logits='d7')       # the layer that reads the stage before
NEXT = dict(a='P', P='g', g='S', S='h4', h4='h5', h5='z', z='logits')
OBS10 = ['mass', 'sum_pt', 'n_particles', 'girth', 'tau21', 'tau32', 'D2', 'z_0', 'dr01', 'sd_mass']
LIVE_THR = 0.001          # live: mean ablation changes >= 0.1 % of calls
SIL_MIN = 0.25            # k = 1 below this silhouette
MAIN_SHARE = 0.80         # one main job: one cluster holds >= 80 % of the activation mass
RHO_SHARED = 0.70         # clusters share the pattern: their own best observable has |Spearman rho| >= 0.7 with the unit's
R2_MIN = 0.50             # unexplained: best <= 2-observable held-out R² below 0.5
MIN_BIN = 20
NB1, NB2 = 20, 8
EXT = 0.4; NPIX = 32
PE = [0, 1, 2, 3, 5, 10, 20, 50, 100, 1e5]       # GeV, the E5 (pT, r) table bins
RE = [0, .025, .05, .1, .15, .2, .3, 1.0]        # rad
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


def uname(st, k):
    return f'{st}{k}'


# ================================================================== observables and descriptions
def obs10(Q, x):
    pt = x[..., 0].astype(np.float64)
    O = {}
    for k in OBS10:
        if k in Q:
            O[k] = np.asarray(Q[k], np.float64)
    if 'z_0' not in O: O['z_0'] = pt[:, 0] / pt.sum(1)
    if 'dr01' not in O: O['dr01'] = np.hypot(x[:, 0, 1] - x[:, 1, 1], x[:, 0, 2] - x[:, 1, 2]).astype(np.float64)
    for k, v in O.items():
        f = np.isfinite(v)
        if not f.all(): O[k] = np.where(f, v, np.median(v[f]))
    missing = [k for k in OBS10 if k not in O]
    assert not missing, missing
    return O


def build_library(Q, x):
    """candidate observables: the Q library (finite, non-constant) + E11's hand-built rank-window sums incl. half-planes"""
    from interp.e11_mars_all import handbuilt
    names, V = [], []
    for k in sorted(Q):
        v = np.asarray(Q[k], np.float64)
        if v.ndim != 1: continue
        f = np.isfinite(v)
        if f.mean() < 0.99: continue
        if not f.all(): v = np.where(f, v, np.median(v[f]))
        if v.std() < 1e-12: continue
        names.append(k); V.append(v)
    for k, v in handbuilt(x, halfplanes=True).items():
        if v.std() < 1e-12 or k in names: continue
        names.append(k); V.append(v)
    return names, np.asarray(V, np.float32)


class Describer:
    """1-D binned responses (NB1 quantile bins) and 2-D (NB2 x NB2) of a target on every candidate; bins from the train
    half, R² on the test half"""

    def __init__(self, names, V, tr, te):
        self.names, self.V, self.tr, self.te = names, V, tr, te
        n, J = V.shape
        self.B1 = np.zeros((n, J), np.int16); self.n1 = np.zeros(n, int)
        self.B2 = np.zeros((n, J), np.int16); self.n2 = np.zeros(n, int)
        self.qmed = []
        for c in range(n):
            v = V[c]
            for nb, B, nn in ((NB1, self.B1, self.n1), (NB2, self.B2, self.n2)):
                e = np.unique(np.quantile(v[tr], np.linspace(0, 1, nb + 1)[1:-1]))
                B[c] = np.searchsorted(e, v, side='right'); nn[c] = len(e) + 1
            b = self.B1[c]; cnt = np.bincount(b, minlength=self.n1[c])
            srt = np.argsort(b, kind='stable'); vs = v[srt]; off = np.concatenate([[0], np.cumsum(cnt)])
            self.qmed.append(np.array([np.median(vs[off[i]:off[i + 1]]) if cnt[i] else np.nan for i in range(self.n1[c])]))
        self._rank = {}

    def rank(self, c):
        if c not in self._rank:
            from scipy.stats import rankdata
            self._rank[c] = rankdata(self.V[c]).astype(np.float32)
        return self._rank[c]

    def rho(self, c1, c2):
        return float(np.corrcoef(self.rank(c1), self.rank(c2))[0, 1])

    @staticmethod
    def _r2(b, nb, t, tr, te):
        """step function: bin means of the train jets, scored on the test jets"""
        cnt = np.bincount(b[tr], minlength=nb); s = np.bincount(b[tr], t[tr], minlength=nb)
        m = np.where(cnt > 0, s / np.maximum(cnt, 1), t[tr].mean())
        yt = t[te]; sst = ((yt - yt.mean()) ** 2).sum()
        return 1 - ((yt - m[b[te]]) ** 2).sum() / max(sst, 1e-300)

    def fit_curve(self, c, t, tr):
        """response curve of target t on candidate c from the jets tr: knots (bin mean of q, bin mean of t), NB1 bins"""
        b = self.B1[c][tr]; q = self.V[c][tr].astype(np.float64); nb = self.n1[c]
        cnt = np.bincount(b, minlength=nb); ok = cnt > 0
        qk = (np.bincount(b, q, nb)[ok] / cnt[ok]); mk = (np.bincount(b, t[tr], nb)[ok] / cnt[ok])
        return qk, mk

    def pred_curve(self, c, knots, idx):
        """linear interpolation between the knots, linear extrapolation beyond the end knots (slope of the end segment)"""
        qk, mk = knots; q = self.V[c][idx].astype(np.float64)
        p = np.interp(q, qk, mk)
        if len(qk) >= 2:
            s0 = (mk[1] - mk[0]) / max(qk[1] - qk[0], 1e-12); s1 = (mk[-1] - mk[-2]) / max(qk[-1] - qk[-2], 1e-12)
            lo = q < qk[0]; hi = q > qk[-1]
            p[lo] = mk[0] + s0 * (q[lo] - qk[0]); p[hi] = mk[-1] + s1 * (q[hi] - qk[-1])
        return p

    def r2_c(self, c, t, tr=None, te=None):
        """held-out R² of the 1-D response curve: linear interpolation between the train bins' (mean q, mean t)"""
        tr = self.tr if tr is None else tr; te = self.te if te is None else te
        yt = t[te]; sst = ((yt - yt.mean()) ** 2).sum()
        return 1 - ((yt - self.pred_curve(c, self.fit_curve(c, t, tr), te)) ** 2).sum() / max(sst, 1e-300)

    def r2_1d(self, t, tr=None, te=None):
        return np.array([self.r2_c(c, t, tr, te) for c in range(len(self.names))])

    def r2_2d(self, t, c1, c2, tr=None, te=None):
        """best of (a) the additive model f1(q1) + f2(q2) (response curves, 3 backfitting rounds) and (b) an 8 x 8 table"""
        tr = self.tr if tr is None else tr; te = self.te if te is None else te
        b = self.B2[c1].astype(np.int64) * self.n2[c2] + self.B2[c2]
        r_tab = self._r2(b, self.n2[c1] * self.n2[c2], t, tr, te)
        f2 = np.zeros(len(t)); mu = t[tr].mean()
        for _ in range(3):
            k1 = self.fit_curve(c1, t - f2, tr); f1 = np.zeros(len(t)); f1[tr] = self.pred_curve(c1, k1, tr)
            k2 = self.fit_curve(c2, t - f1, tr); f2 = np.zeros(len(t)); f2[tr] = self.pred_curve(c2, k2, tr)
        p = self.pred_curve(c1, k1, te) + self.pred_curve(c2, k2, te)
        yt = t[te]; sst = ((yt - yt.mean()) ** 2).sum()
        r_add = 1 - ((yt - p) ** 2).sum() / max(sst, 1e-300)
        return max(r_tab, r_add)

    def curve(self, t, c):
        b = self.B1[c]; nb = self.n1[c]
        cnt = np.bincount(b, minlength=nb); s = np.bincount(b, t, minlength=nb); s2 = np.bincount(b, t * t, minlength=nb)
        m = s / np.maximum(cnt, 1); sd = np.sqrt(np.maximum(s2 / np.maximum(cnt, 1) - m ** 2, 0))
        ok = cnt > 0
        return dict(q=self.qmed[c][ok].tolist(), f=m[ok].tolist(), sd=sd[ok].tolist(), n=cnt[ok].tolist())

    def describe(self, t):
        r1 = self.r2_1d(t)
        order = np.argsort(-r1)
        top = []
        for c in order:
            if len(top) >= 12: break
            if all(abs(self.rho(c, d)) < 0.97 for d in top): top.append(int(c))
        best_pair, best_r2 = None, -np.inf
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                r = self.r2_2d(t, top[i], top[j])
                if r > best_r2: best_pair, best_r2 = (top[i], top[j]), r
        c0 = int(order[0])
        res = dict(best1=dict(name=self.names[c0], label=label(self.names[c0]), R2=float(r1[c0]), curve=self.curve(t, c0)),
                   top1d=[dict(name=self.names[c], label=label(self.names[c]), R2=float(r1[c])) for c in top[:6]])
        if best_pair is not None:
            a, b = best_pair
            # direction of the second observable: sign of its correlation with the residual of the first's response
            first, second = (a, b) if r1[a] >= r1[b] else (b, a)
            cur = self.curve(t, first)
            bb = self.B1[first]; cnt = np.bincount(bb, minlength=self.n1[first]); s = np.bincount(bb, t, minlength=self.n1[first])
            resid = t - (s / np.maximum(cnt, 1))[bb]
            sgn = float(np.sign(np.corrcoef(resid, self.V[second])[0, 1]))
            res['best2'] = dict(names=[self.names[first], self.names[second]], labels=[label(self.names[first]), label(self.names[second])],
                                R2=float(best_r2), second_sign=sgn)
        res['R2_1'] = float(r1[c0]); res['R2_2'] = float(max(r1[c0], best_r2))
        res['_c0'] = c0
        return res


# ------------------------------------------------------------------ labels and units of the observables
WIN = {'all': '', 'top14': '14 hardest', 'r1-63': 'ranks 1–63', 'r13-63': 'ranks 13–63', 'top39': '39 hardest', 'top26': '26 hardest',
       'top3': '3 hardest', 'top11': '11 hardest', 'r8-63': 'ranks 8–63'}


def label(name):
    import re
    if name.startswith('H:'):
        k = name[2:]
        m = re.fullmatch(r'Σ\(n·r\)₊\[(\d+)°\]\[(.+)\]', k)
        if m: return f'Σ(n̂·r⃗)₊, n̂ at {m.group(1)}°' + (f' ({WIN[m.group(2)]})' if WIN.get(m.group(2)) else '')
        m = re.fullmatch(r'N\(n·r>0\)\[(\d+)°\]\[(.+)\]', k)
        if m: return f'N(n̂·r⃗ > 0), n̂ at {m.group(1)}°' + (f' ({WIN[m.group(2)]})' if WIN.get(m.group(2)) else '')
        m = re.fullmatch(r'(.+)\[(.+)\]', k)
        if m:
            w = WIN.get(m.group(2), m.group(2))
            return m.group(1) + (f' ({w})' if w else '')
        return k
    try:
        from jetdistill.observables.symbols import symbol
        s = symbol(name)
    except Exception:
        s = name
    return {'mass': 'jet mass', 'n': 'multiplicity', 'ΣzΔR': 'girth', 'ΣzΔR²': 'girth²'}.get(s, s)


def unit_of(name):
    import re
    k = name[2:] if name.startswith('H:') else name
    if name.startswith('H:'):
        if k.startswith('N'): return ''
        if k.startswith('ΣpT·r²'): return ' GeV·rad²'
        if k.startswith('ΣpT·r'): return ' GeV·rad'
        if k.startswith('ΣpT'): return ' GeV'
        if k.startswith('Σr²'): return ' rad²'
        if k.startswith(('Σr', 'Σ(r', 'Σ(n')): return ' rad'
        if k.startswith('√(M0'): return ' GeV'
        if k.startswith('|M1|'): return ' GeV·rad'
        return ''
    if k == 'orientation_deg': return '°'
    if '_over_' in k or k.startswith(('mratio', 'sj3_pairm')) or k.endswith('_over_m'): return ''
    if 'mass' in k or k in ('sum_pt', 'm01', 'm012') or re.fullmatch(r'pt_\d+|sum_pt_top\d+|soft\d+_pt|pt1_dr01', k): return ' GeV'
    if k in ('girth', 'max_dr', 'sd_rg', 'sj2_dr', 'centroid_offset') or re.fullmatch(r'dr.*|eta_\d+|phi_\d+|abseta_\d+|absphi_\d+|mean_eta|mean_phi|soft\d+_dr\d*|sj3_dr.*', k):
        return ' rad'
    return ''


def fmt(v):
    v = float(v)
    if v == 0: return '0'
    a = abs(v)
    if a >= 100: return f'{v:.0f}'
    if a >= 10: return f'{v:.1f}'
    if a >= 1: return f'{v:.2f}'
    if a >= 0.01: return f'{v:.3f}'
    return f'{v:.2g}'


def pattern_line(desc, negate=False, zero=False):
    """the range of the best observable where the response is above half its span (zero: where it is > 0, if it crosses
    0 — the on-region of a pre-ReLU neuron; negate: where it is most negative / < 0); the peak when it is interior;
    + the second observable's direction"""
    b = desc['best1']; q = np.array(b['curve']['q']); f = np.array(b['curve']['f']); lab = b['label']; un = unit_of(b['name'])
    if negate: f = -f
    if len(f) < 2 or f.max() - f.min() < 1e-9: return f'flat in {lab}'
    above = f > 0 if (zero and f.max() > 0 > f.min()) else f >= f.min() + 0.5 * (f.max() - f.min())
    i = np.flatnonzero(above); n = len(f)
    edge = lambda k: 0.5 * (q[k] + q[k + 1])
    pk = fmt(q[int(np.argmax(f))])
    if i[-1] - i[0] + 1 == len(i):
        ia = int(np.argmax(f)); interior = 1 < ia < n - 2
        if i[0] == 0 and i[-1] == n - 1: s = f'flat in {lab}'
        elif i[-1] == n - 1: s = f'{lab} > {fmt(edge(i[0] - 1))}{un}' + (f' (peak {pk})' if interior else '')
        elif i[0] == 0: s = f'{lab} < {fmt(edge(i[-1]))}{un}' + (f' (peak {pk})' if interior else '')
        else: s = f'{lab} {fmt(edge(i[0] - 1))}–{fmt(edge(i[-1]))}{un} (peak {pk})'
    else:
        lo_seg = above[0]; hi_seg = above[-1]
        if lo_seg and hi_seg and not above[n // 2]:
            j = np.flatnonzero(~above)
            s = f'{lab} < {fmt(edge(j[0] - 1))} or > {fmt(edge(j[-1]))}{un}'
        else:
            s = f'{lab}, non-monotone, highest at {pk}{un}'
    b2 = desc.get('best2')
    if b2 and not negate and desc['R2_2'] - desc['R2_1'] >= 0.03:
        s += f'; {"higher" if b2["second_sign"] > 0 else "lower"} {b2["labels"][1]}'
    return s


# ================================================================== network helpers
class Net:
    def __init__(self, per_class, seed):
        from interp.jedi_core import Jedi, sample
        from interp.e4_e5_jedi import bits
        import jax, jax.numpy as jnp
        self.jax, self.jnp = jax, jnp
        self.x, self.y, self.Q = sample(per_class, seed)
        self.J = len(self.x)
        self.net = Jedi(64)
        n = self.net
        self.L = n.L
        self.M3 = bits(n.L['l3']).reshape(64, 64) > 0           # l3 input kept (slot, channel)
        self.KEEP = bits(n.L['qs1']).reshape(64, 64) > 0        # u_c of slot i enters S_c
        self.KEEPP = bits(n.L['qs']).reshape(64, 64) > 0        # a_c of slot i enters P_c
        self.R = n.run(self.x, keep=JET_STAGES + ('logits',), batch=2500)
        self.call = self.R['logits'].argmax(1)
        self.real = self.x[..., 0] > 0
        self.kd_S = bool(n.L['qs1'].keepdims)
        self._tail = self._make_tail()
        lo = self.tail(self.R['S'])
        log(f'tail check: max |Δlogit| {np.abs(lo - self.R["logits"]).max():.3g}, calls equal {(lo.argmax(1) == self.call).mean():.5f}')

    # ---- tail S -> logits with per-jet overrides of S, h4, h5, z
    def _make_tail(self):
        jax, jnp, L = self.jax, self.jnp, self.L
        KB = {k: (jnp.asarray(self.net.KB[k][0], jnp.float32), jnp.asarray(self.net.KB[k][1], jnp.float32)) for k in ('d4', 'd5', 'd6', 'd7')}

        def dense(k, t, act=True):
            z = jnp.einsum(L[k].equation, L[k].iq(t, training=False), KB[k][0]) + KB[k][1]
            return L[k].activation(z) if act and L[k].activation is not None else z

        def f(S3, mS, vS, m4, v4, m5, v5, mz, vz):
            S3 = S3 * (1 - mS) + mS * vS
            h4 = dense('d4', S3); h4 = h4 * (1 - m4) + m4 * v4
            h5 = dense('d5', h4); h5 = h5 * (1 - m5) + m5 * v5
            z = dense('d6', h5, act=False); z = z * (1 - mz) + mz * vz
            return dense('d7', jax.nn.relu(z)).reshape(S3.shape[0], -1)
        return jax.jit(f)

    def _shape(self, B, W):
        return (B, 1, W) if self.kd_S else (B, W)

    def tail(self, S, ov=None, batch=None):
        """logits from S with overrides {stage: (mask (W,), values (W,) or (len(S), W))}; inputs are padded to a fixed batch
        (5000 for the full sample, 2000 for subsets) so that the jitted function is compiled once per shape"""
        jnp = self.jnp; ov = dict(ov or {}); out = []; n0 = len(S)
        batch = batch or (5000 if n0 % 5000 == 0 else 2000)
        pad = (-n0) % batch
        if pad:
            S = np.concatenate([S, np.repeat(S[:1], pad, 0)])
            ov = {k: (m, (np.concatenate([np.asarray(v), np.repeat(np.asarray(v)[:1], pad, 0)]) if np.ndim(v) == 2 else v)) for k, (m, v) in ov.items()}
        for i in range(0, len(S), batch):
            Sb = S[i:i + batch]; B = len(Sb)
            args = [jnp.asarray(Sb.reshape(self._shape(B, 64)), jnp.float32)]
            for st in ('S', 'h4', 'h5', 'z'):
                W = WIDTH[st]
                if st in ov:
                    m, v = ov[st]; v = np.asarray(v, np.float32)
                    v = v.reshape(self._shape(1, W)) if v.ndim == 1 else v[i:i + batch].reshape(self._shape(B, W))
                    args += [jnp.asarray(m, jnp.float32), jnp.asarray(v)]
                else:
                    args += [jnp.zeros(W, jnp.float32), jnp.zeros(self._shape(1, W), jnp.float32)]
            out.append(np.asarray(self._tail(*args), np.float64))
        return np.concatenate(out)[:n0]

    # ---- logits with one jet-level unit set per jet (values (n,) for the jets idx)
    def set_unit(self, st, k, idx, values):
        W = WIDTH[st]; m = np.zeros(W, np.float32); m[k] = 1
        v = np.zeros((len(idx), W), np.float32); v[:, k] = values
        if st in ('S', 'h4', 'h5', 'z'):
            return self.tail(self.R['S'][idx], {st: (m, v)})
        if st in ('P', 'g'):
            n0 = len(idx); pad = (-n0) % 2000
            xi = self.x[idx]
            if pad: xi = np.concatenate([xi, np.repeat(xi[:1], pad, 0)]); v = np.concatenate([v, np.repeat(v[:1], pad, 0)])
            return self.net.run_jetwise(xi, {st: (m, v)}, batch=2000)['logits'][:n0]
        raise ValueError(st)

    def ablate(self, st, k, mean):
        W = WIDTH[st]; m = np.zeros(W, np.float32); m[k] = 1; v = np.zeros(W, np.float32); v[k] = mean
        if st in ('S', 'h4', 'h5', 'z'):
            return self.tail(self.R['S'], {st: (m, v)}).argmax(1)
        return self.net.run(self.x, override={st: (m, v)}, batch=2500)['logits'].argmax(1)

    def qin(self, layer, X):
        """the layer's quantized input for jet-level X (J, W) -> (J, W); dd reads P, which keeps the pooled axis when qs
        has keepdims (qs1 may not: S, h4, h5, z then have shape (J, W))"""
        jnp = self.jnp; out = []
        kd = bool(self.L['qs'].keepdims) if layer == 'dd' else self.kd_S
        for i in range(0, len(X), 5000):
            Xb = X[i:i + 5000]; shp = (len(Xb), 1, X.shape[1]) if kd else (len(Xb), X.shape[1])
            out.append(np.asarray(self.L[layer].iq(jnp.asarray(Xb.reshape(shp), jnp.float32), training=False)).reshape(len(Xb), -1))
        return np.concatenate(out).astype(np.float64)

    def cuts(self):
        """l1 statements in physical units: alpha·(pT, Δη, Δφ) + beta > 0"""
        K1, b1 = self.net.KB['l1']
        alpha = K1 / self.net.scale[:, None]; beta = b1 - (K1 * (self.net.shift / self.net.scale)[:, None]).sum(0)
        return alpha, beta

    # ---- per-particle pass: statistics of the l3 inputs (S upstream), the u table, the a statistics
    def particle_pass(self, live_S, want_a=False, batch=2000):
        jax, jnp, L = self.jax, self.jnp, self.L
        KB = {k: (jnp.asarray(self.net.KB[k][0], jnp.float32), jnp.asarray(self.net.KB[k][1], jnp.float32)) for k in ('l1', 'l2', 'dd', 'l3')}
        m3 = jnp.asarray(self.M3, jnp.float32)

        def dense(k, t):
            z = jnp.einsum(L[k].equation, L[k].iq(t, training=False), KB[k][0]) + KB[k][1]
            return L[k].activation(z) if L[k].activation is not None else z

        def f(xin):
            a = dense('l1', xin)
            qa2 = L['l2'].iq(a, training=False)
            s = jax.nn.relu(jnp.einsum(L['l2'].equation, qa2, KB['l2'][0]) + KB['l2'][1])
            P = jnp.sum(L['qs'].iq(a, training=False), axis=L['qs'].axes, keepdims=L['qs'].keepdims) * L['qs'].scale
            g = dense('dd', P)
            qs_, qd_ = L['add'].iq([s, g], training=False)
            u = dense('l3', qs_ + qd_)
            vs = m3 * qs_; vg = m3 * jnp.broadcast_to(qd_, qs_.shape)
            return a, qa2, vs, vg, u
        fj = jax.jit(f)
        nS = len(live_S); ls = np.asarray(live_S, int)
        acc = dict(cnt=np.zeros(nS), s1=np.zeros((nS, 64)), s2=np.zeros((nS, 64)), g1=np.zeros((nS, 64)), g2=np.zeros((nS, 64)),
                   tab_sum=np.zeros((nS, len(PE) - 1, len(RE) - 1)), tab_on=np.zeros((nS, len(PE) - 1, len(RE) - 1)),
                   tab_n=np.zeros((nS, len(PE) - 1, len(RE) - 1)),
                   a_sum=np.zeros(64), a_n=0, a_min=np.full(64, np.inf), a_max=np.full(64, -np.inf), a_on=np.zeros(64),
                   qa_1=np.zeros(64), qa_2=np.zeros(64), a_sample=[], img_pt=np.zeros((64, NPIX, NPIX)), img_n=np.zeros((64, NPIX, NPIX)),
                   pt_hist_on=np.zeros((64, 40)), pt_hist_all=np.zeros(40), a_pad_min=np.full(64, np.inf), a_pad_max=np.full(64, -np.inf))
        pt_edges = np.logspace(-1, 3.2, 41)
        checked = False
        for i in range(0, self.J, batch):
            xb = self.x[i:i + batch]; B = len(xb)
            a, qa2, vs, vg, u = (np.asarray(t, np.float32).reshape(B, 64, 64) for t in fj(self.net.inputs(xb, 64)))
            if not checked:
                ur = np.asarray(self.net._f(self.net.inputs(xb[:50], 64), {})['u'])
                log(f'particle pass check: max |u - forward u| {np.abs(ur - u[:50]).max():.3g}'); checked = True
            real = xb[..., 0] > 0
            pt = xb[..., 0].astype(np.float64); r = np.hypot(xb[..., 1], xb[..., 2]).astype(np.float64)
            if nS:
                Wm = (real[:, :, None] & self.KEEP[None, :, ls]).astype(np.float32)     # (B, 64 slots, nS)
                acc['cnt'] += Wm.sum((0, 1))
                acc['s1'] += np.einsum('bic,bik->ck', Wm, vs); acc['s2'] += np.einsum('bic,bik->ck', Wm, vs * vs)
                acc['g1'] += np.einsum('bic,bik->ck', Wm, vg); acc['g2'] += np.einsum('bic,bik->ck', Wm, vg * vg)
                ip = np.clip(np.searchsorted(PE, pt, side='right') - 1, 0, len(PE) - 2)
                ir = np.clip(np.searchsorted(RE, r, side='right') - 1, 0, len(RE) - 2)
                cell = ip * (len(RE) - 1) + ir
                nc = (len(PE) - 1) * (len(RE) - 1)
                for j, c in enumerate(ls):
                    w = Wm[..., j] > 0
                    acc['tab_sum'][j] += np.bincount(cell[w], u[..., c][w], nc).reshape(len(PE) - 1, len(RE) - 1)
                    acc['tab_on'][j] += np.bincount(cell[w], (u[..., c][w] > 0).astype(float), nc).reshape(len(PE) - 1, len(RE) - 1)
                    acc['tab_n'][j] += np.bincount(cell[w], None, nc).reshape(len(PE) - 1, len(RE) - 1)
            if want_a:
                ar = a[real]                                     # (n_real, 64)
                acc['a_sum'] += ar.sum(0); acc['a_n'] += len(ar); acc['a_on'] += (ar > 0).sum(0)
                acc['a_min'] = np.minimum(acc['a_min'], ar.min(0)); acc['a_max'] = np.maximum(acc['a_max'], ar.max(0))
                if (~real).any():
                    ap = a[~real]; acc['a_pad_min'] = np.minimum(acc['a_pad_min'], ap.min(0)); acc['a_pad_max'] = np.maximum(acc['a_pad_max'], ap.max(0))
                qr = qa2[real]; acc['qa_1'] += qr.sum(0); acc['qa_2'] += (qr.astype(np.float64) ** 2).sum(0)
                if i < 4000: acc['a_sample'].append(ar)
                e = xb[..., 1][real]; f_ = xb[..., 2][real]; p = xb[..., 0][real]
                ib = np.floor((e + EXT) / (2 * EXT) * NPIX).astype(int); jb = np.floor((f_ + EXT) / (2 * EXT) * NPIX).astype(int)
                ok = (ib >= 0) & (ib < NPIX) & (jb >= 0) & (jb < NPIX)
                pix = ib * NPIX + jb
                ipt = np.clip(np.searchsorted(pt_edges, p) - 1, 0, 39)
                acc['pt_hist_all'] += np.bincount(ipt, minlength=40)
                for c in range(64):
                    on = ar[:, c] > 0
                    if not on.any(): continue
                    w = on & ok
                    acc['img_n'][c] += np.bincount(pix[w], None, NPIX * NPIX).reshape(NPIX, NPIX)
                    acc['img_pt'][c] += np.bincount(pix[w], p[w], NPIX * NPIX).reshape(NPIX, NPIX)
                    acc['pt_hist_on'][c] += np.bincount(ipt[on], minlength=40)
        if want_a:
            acc['a_sample'] = np.concatenate(acc['a_sample'])
        acc['pt_edges'] = pt_edges
        return acc


# ================================================================== per-unit pieces
def bins_of(act, on, rng, z=False):
    """rank bins among the on-jets (all jets if fewer than 200 are on): top 1 %, 90-99 %, 50-90 %; z: also the lowest 1 %"""
    base = np.flatnonzero(on) if on.sum() >= 200 else np.arange(len(act))
    pr = rng.permutation(len(base)); base = base[pr]
    srt = base[np.argsort(-act[base], kind='stable')]
    n = len(srt)
    n1 = max(int(np.ceil(0.01 * n)), min(MIN_BIN, n)); n10 = max(int(np.ceil(0.10 * n)), n1 + min(MIN_BIN, n - n1))
    n50 = max(int(np.ceil(0.50 * n)), n10 + min(MIN_BIN, n - n10))
    B = {'top 1 %': srt[:n1], '90–99 %': srt[n1:n10], '50–90 %': srt[n10:n50]}
    if z:
        allr = rng.permutation(len(act)); lo = allr[np.argsort(act[allr], kind='stable')]
        B['lowest 1 %'] = lo[:max(int(np.ceil(0.01 * len(act))), MIN_BIN)]
    return {k: v for k, v in B.items() if len(v)}, ('on-jets' if on.sum() >= 200 else 'all jets')


def align(x):
    pt = x[..., 0].astype(np.float64); e = x[..., 1].astype(np.float64); f = x[..., 2].astype(np.float64); J = len(x)
    Cee, Cff, Cef = (pt * e * e).sum(1), (pt * f * f).sum(1), (pt * e * f).sum(1)
    th = 0.5 * np.arctan2(2 * Cef, Cee - Cff)
    c, s = np.cos(th)[:, None], np.sin(th)[:, None]
    xa = c * e + s * f; ya = -s * e + c * f
    h = pt.argmax(1); sx = np.sign(xa[np.arange(J), h]); sx[sx == 0] = 1; xa *= sx[:, None]
    w = pt.copy(); w[np.arange(J), h] = 0
    sy = np.sign((w * ya).sum(1)); sy[sy == 0] = 1; ya *= sy[:, None]
    return xa.astype(np.float32), ya.astype(np.float32)


def images(idx, PX):
    pt = PX['pt'][idx]; m = pt > 0; w = pt[m]
    rg = [[-EXT, EXT], [-EXT, EXT]]
    Hpt = np.histogram2d(PX['eta'][idx][m], PX['phi'][idx][m], bins=NPIX, range=rg, weights=w)[0]
    Hn = np.histogram2d(PX['eta'][idx][m], PX['phi'][idx][m], bins=NPIX, range=rg)[0]
    Ha = np.histogram2d(PX['xa'][idx][m], PX['ya'][idx][m], bins=NPIX, range=rg, weights=w)[0]
    return (np.stack([Hpt, Hn, Ha]) / max(len(idx), 1)).astype(np.float32)


def class_mix(y, idx):
    return (np.bincount(y[idx], minlength=5) / max(len(idx), 1)).tolist()


def med10(O, idx):
    return {k: float(np.median(O[k][idx])) for k in OBS10}


def shift_desc(O, sd, ref, idx, k=2):
    """the k observables of OBS10 whose median moves most (in units of the on-jet SD) from ref jets to idx jets"""
    d = {o: (np.median(O[o][idx]) - np.median(O[o][ref])) / max(sd[o], 1e-12) for o in OBS10}
    top = sorted(d, key=lambda o: -abs(d[o]))[:k]
    return [dict(obs=o, dz=float(d[o]), med=float(np.median(O[o][idx])), med_ref=float(np.median(O[o][ref]))) for o in top]


def clusters(F, act, on, y, O, D, desc, rng, tr_mask):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    idx = np.flatnonzero(on)
    if len(idx) < 60: return dict(k=0, note='fewer than 60 on-jets', clusters=[], main_share=1.0, main_share_forced=1.0)
    X = F[idx]; X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    w = np.maximum(act[idx], 0)
    if w.sum() <= 0: w = np.ones(len(idx))
    # activation-weighted: the fit and the silhouette use 3,000 on-jets drawn with probability proportional to the activation
    sub = rng.choice(len(idx), min(3000, len(idx)), replace=False, p=w / w.sum()) if (w > 0).sum() >= min(3000, len(idx)) \
        else rng.choice(len(idx), min(3000, len(idx)), replace=False)
    sil, models = {}, {}
    for k in (2, 3, 4):
        km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(X[sub])
        if len(set(km.labels_)) < 2: continue
        sil[k] = float(silhouette_score(X[sub], km.labels_)); models[k] = km
    out = dict(silhouette=sil)
    kb = max(sil, key=sil.get) if sil else 1

    def summarise(lab, k):
        cl = []
        c0 = desc['_c0']
        for j in range(k):
            sel = lab == j; ids = idx[sel]
            e = dict(n=int(sel.sum()), share_jets=float(sel.mean()), share_mass=float(w[sel].sum() / w.sum()),
                     mean_act=float(act[ids].mean()), class_mix=class_mix(y, ids), med=med10(O, ids))
            cl.append(e)
        order = np.argsort([-c['share_mass'] for c in cl])
        cl = [cl[o] for o in order]; lab2 = np.argsort(order)[lab]
        for j, e in enumerate(cl):
            if e['share_mass'] < 0.10 or e['n'] < 60: continue
            ids = idx[lab2 == j]; trs = ids[tr_mask[ids]]; tes = ids[~tr_mask[ids]]
            if len(trs) < 30 or len(tes) < 30 or act[ids].std() < 1e-9:
                e['own_best'] = None; continue
            r1 = D.r2_1d(act, trs, tes); cb = int(np.argmax(r1))
            e['own_best'] = dict(name=D.names[cb], label=label(D.names[cb]), R2=float(r1[cb]))
            e['rho_with_unit_best'] = abs(D.rho(cb, c0))
            e['unit_best_R2_in_cluster'] = float(D.r2_c(c0, act, trs, tes))
        return cl, lab2

    kf = kb
    labf = models[kf].predict(X) if sil else np.zeros(len(idx), int)
    clf, _ = summarise(labf, kf if sil else 1)
    k = kb if (sil and sil[kb] >= SIL_MIN) else 1
    if k == 1:
        cl, _ = summarise(np.zeros(len(idx), int), 1)
    else:
        cl = clf
    out.update(k=k, k_forced=kf, clusters=cl, clusters_forced=clf, main_share=cl[0]['share_mass'], main_share_forced=clf[0]['share_mass'])
    # offshoots: clusters other than the main one, described by their observable shifts from the main cluster
    sdo = {o: float(np.std(O[o][idx])) for o in OBS10}
    for e in out['clusters'][1:]:
        e['vs_main'] = sorted(({'obs': o, 'dz': (e['med'][o] - cl[0]['med'][o]) / max(sdo[o], 1e-12)} for o in OBS10), key=lambda d: -abs(d['dz']))[:2]
    for e in out['clusters_forced'][1:]:
        e['vs_main'] = sorted(({'obs': o, 'dz': (e['med'][o] - clf[0]['med'][o]) / max(sdo[o], 1e-12)} for o in OBS10), key=lambda d: -abs(d['dz']))[:2]
    return out


def category(desc, cl, forced=False):
    if desc['R2_2'] < R2_MIN: return 'unexplained'
    clusters_ = cl.get('clusters_forced' if forced else 'clusters', [])
    share = cl.get('main_share_forced' if forced else 'main_share', 1.0)
    if share >= MAIN_SHARE or not clusters_: return 'one main job'
    big = [c for c in clusters_ if c['share_mass'] >= 0.10 and c.get('own_best')]
    if big and all(c.get('rho_with_unit_best', 0) >= RHO_SHARED for c in big): return 'related jobs'
    return 'unrelated jobs'


def flow(before, after):
    M = np.zeros((5, 5), int)
    np.add.at(M, (before, after), 1)
    return M


def steer(N, st, k, act, value, idx, top_major):
    lo = N.set_unit(st, k, idx, np.full(len(idx), value, np.float32))
    new = lo.argmax(1); old = N.call[idx]
    ch = new != old
    dest = np.bincount(new[ch], minlength=5)
    return dict(n=int(len(idx)), value=float(value), changed=float(ch.mean()), flow=flow(old, new).tolist(),
                dest=(dest / max(ch.sum(), 1)).tolist(), to_top_major=float((new[ch] == top_major).mean()) if ch.any() else 0.0,
                new_call_mix=(np.bincount(new, minlength=5) / len(new)).tolist(), old_call_mix=(np.bincount(old, minlength=5) / len(old)).tolist())


def steer_targets(act, on, rng, nmax=4000):
    off = np.flatnonzero(~on)
    if len(off) >= 0.01 * len(act):
        rule = 'off jets'
    else:
        off = np.flatnonzero(act <= np.percentile(act, 10)); rule = 'jets below the 10th percentile (on in > 99 %)'
    if len(off) > nmax: off = np.sort(rng.choice(off, nmax, replace=False))
    return off, rule


def p99_value(act, on):
    v = float(np.percentile(act, 99))
    if v <= 0 and on.any(): v = float(np.percentile(act[on], 99))
    return v


# ================================================================== the run
def run(stages, per_class, seed):
    OUT.mkdir(parents=True, exist_ok=True)
    N = Net(per_class, seed)
    x, y, Q, R, J = N.x, N.y, N.Q, N.R, N.J
    acc0 = float((N.call == y).mean())
    log(f'{J} jets ({per_class}/class, seed {seed}); accuracy {acc0:.4f}; stages {stages}')
    O = obs10(Q, x)
    F = np.stack([np.clip(O[k], *np.percentile(O[k], [1, 99])) for k in OBS10], 1)
    names, V = build_library(Q, x)
    prm = np.random.default_rng(seed).permutation(J); tr = np.sort(prm[:J // 2]); te = np.sort(prm[J // 2:])
    tr_mask = np.zeros(J, bool); tr_mask[tr] = True
    D = Describer(names, V, tr, te)
    log(f'{len(names)} candidate observables')
    xa, ya = align(x)
    PX = dict(pt=x[..., 0], eta=x[..., 1], phi=x[..., 2], xa=xa, ya=ya)
    alpha, beta = N.cuts()
    e5 = {}
    p5 = OUT.parent / 'e4_e5.json'
    if p5.exists():
        for d in json.loads(p5.read_text())['E5']['units']:
            if 'hinge' in d: e5[int(d['S'])] = dict(hinge=d['hinge'], R2=d.get('hinge_R2'), desc=d.get('hinge_desc'))
    rng0 = np.random.default_rng(seed + 1)
    dl_idx = np.sort(rng0.choice(J, min(2000, J), replace=False))

    for st in stages:
        t_st = time.time()
        res = dict(stage=st, N=J, per_class=per_class, seed=seed, acc=acc0, thresholds=dict(live=LIVE_THR, sil_min=SIL_MIN, main_share=MAIN_SHARE,
                   rho_shared=RHO_SHARED, r2_min=R2_MIN), units=[])
        if st == 'a':
            res.update(run_a(N, alpha, beta, rng0))
            imgs = res.pop('_imgs')
            (OUT / f'atlas_{st}.json').write_text(json.dumps(res, indent=1, default=float))
            np.savez_compressed(OUT / f'images_{st}.npz', **imgs)
            log(f'stage a done ({time.time() - t_st:.0f} s)')
            try:
                from interp.e16_plot import cards
                cards(st)
            except Exception as e:
                import traceback; traceback.print_exc()
                log(f'cards for {st} failed: {e}')
            continue
        A = R[st]; W = WIDTH[st]
        # ---- liveness
        mean = A.mean(0); sd = A.std(0)
        cost = np.zeros(W); accab = np.full(W, acc0)
        for k in range(W):
            if sd[k] < 1e-9: continue
            c2 = N.ablate(st, k, mean[k]); cost[k] = (c2 != N.call).mean(); accab[k] = (c2 == y).mean()
        live = [int(k) for k in np.argsort(-cost) if cost[k] >= LIVE_THR]
        res['cost'] = cost.tolist(); res['acc_ablated'] = accab.tolist(); res['live'] = live
        log(f'{st}: {len(live)} live of {W} (non-constant {(sd >= 1e-9).sum()}); costs ' + ', '.join(f'{k}:{cost[k]*100:.1f}%' for k in live[:10]))
        # ---- upstream inputs as received
        up = {}
        if st in IN_LAYER:
            prev = PREV[st]; qi = N.qin(IN_LAYER[st], R[prev]); K = N.net.KB[IN_LAYER[st]][0]
            up['spread'] = qi.std(0); up['K'] = K; up['prev'] = prev
        if st == 'S' or st == 'g':
            liveS = [k for k in range(64) if R['S'][:, k].std() > 1e-9]
            if getattr(N, '_pp', None) is None: N._pp = N.particle_pass(liveS)
            up['pp'] = N._pp; up['liveS'] = liveS
        nxt = NEXT[st]
        if nxt in IN_LAYER:
            qn = N.qin(IN_LAYER[nxt], R[st] if nxt != 'logits' else np.maximum(R[st], 0)); Kn = N.net.KB[IN_LAYER[nxt]][0]
        else:
            qn = Kn = None
        live_next = None
        if nxt != 'logits' and nxt in R:
            live_next = [k for k in range(WIDTH[nxt]) if R[nxt][:, k].std() > 1e-9]
        imgs = {}
        for k in live:
            t_u = time.time()
            act = A[:, k]; on = act > 0
            rng = np.random.default_rng(1000 * ALL_STAGES.index(st) + k)
            u = dict(stage=st, unit=k, id=uname(st, k), cost=float(cost[k]), acc_ablated=float(accab[k]), on_frac=float(on.mean()),
                     neg_frac=float((act < 0).mean()) if st == 'z' else 0.0, mean=float(mean[k]), sd=float(sd[k]),
                     pct={p: float(np.percentile(act, p)) for p in (1, 10, 50, 90, 99)})
            ed = np.linspace(act.min(), act.max(), 41) if act.max() > act.min() else np.array([act.min() - 1, act.max() + 1])
            u['hist'] = dict(edges=ed.tolist(), counts=[np.histogram(act[y == c], ed)[0].tolist() for c in range(5)])
            B, base = bins_of(act, on, rng, z=(st == 'z'))
            u['bin_base'] = base
            onidx = np.flatnonzero(on) if on.sum() >= 200 else np.arange(J)
            sd_on = {o: float(np.std(O[o][onidx])) for o in OBS10}
            u['bins'] = {}
            im = []
            for bn, ids in B.items():
                u['bins'][bn] = dict(n=int(len(ids)), act_range=[float(act[ids].min()), float(act[ids].max())], class_mix=class_mix(y, ids),
                                     med=med10(O, ids))
                im.append(images(ids, PX))
            u['bin_names'] = list(B)
            u['med_on'] = med10(O, onidx); u['med_all'] = med10(O, np.arange(J)); u['class_mix_on'] = class_mix(y, onidx)
            imgs[uname(st, k)] = np.stack(im)
            # weak range and negative pole
            if '50–90 %' in B:
                u['weak_vs_top'] = shift_desc(O, sd_on, B['top 1 %'], B['50–90 %'])
                u['weak_class_overlap'] = float(np.minimum(u['bins']['top 1 %']['class_mix'], u['bins']['50–90 %']['class_mix']).sum())
            if st == 'z':
                sd_all = {o: float(np.std(O[o])) for o in OBS10}
                u['neg_vs_all'] = shift_desc(O, sd_all, np.arange(J), B['lowest 1 %'])
            # ---- description
            desc = D.describe(act)
            u['desc'] = {kk: vv for kk, vv in desc.items() if not kk.startswith('_')}
            u['pattern'] = pattern_line(desc, zero=(st == 'z'))
            if st == 'z':      # the negative side alone: best observable for min(z, 0)
                tneg = np.minimum(act, 0)
                if tneg.std() > 1e-9:
                    dn = D.describe(tneg)
                    u['neg_desc'] = dict(best1={kk: vv for kk, vv in dn['best1'].items() if kk != 'curve'}, R2_1=dn['R2_1'], R2_2=dn['R2_2'],
                                         pattern=pattern_line(dn, negate=True, zero=True), curve=dn['best1']['curve'])
            # ---- clusters
            cl = clusters(F, act, on, y, O, D, desc, rng, tr_mask)
            u['clusters'] = cl
            u['category'] = category(desc, cl); u['category_forced_k2'] = category(desc, cl, forced=True)
            # ---- upstream
            u['upstream'] = upstream(N, st, k, up, alpha, beta, e5)
            # ---- downstream
            dn_ = dict()
            if Kn is not None:
                spread = qn[:, k].std()
                if nxt == 'logits':
                    dn_['weights'] = [dict(cls=CL[c], w=float(Kn[k, c]), w_x_spread=float(Kn[k, c] * spread)) for c in range(5)]
                else:
                    cand = live_next if live_next is not None else range(Kn.shape[1])
                    sc = sorted(((abs(Kn[k, m]) * spread, m) for m in cand), reverse=True)[:5]
                    dn_['units'] = [dict(unit=uname(nxt, int(m)), w=float(Kn[k, m]), score=float(s)) for s, m in sc if s > 0]
            if st == 'g':
                pp = up['pp']; ls = up['liveS']; K3 = N.net.KB['l3'][0]
                g_sd = np.sqrt(np.maximum(pp['g2'] / np.maximum(pp['cnt'][:, None], 1) - (pp['g1'] / np.maximum(pp['cnt'][:, None], 1)) ** 2, 0))
                sc = sorted(((abs(K3[k, c]) * g_sd[j, k], c) for j, c in enumerate(ls)), reverse=True)[:5]
                dn_['units'] = [dict(unit=uname('S', int(c)), w=float(K3[k, c]), score=float(s)) for s, c in sc if s > 0]
            p10, p90 = u['pct'][10], u['pct'][90]; rule = 'all jets'
            if p90 - p10 < 1e-9 and on.sum() >= 20:
                p10, p90 = float(np.percentile(act[on], 10)), float(np.percentile(act[on], 90)); rule = 'on-jets'
            l90 = N.set_unit(st, k, dl_idx, np.full(len(dl_idx), p90, np.float32))
            l10 = N.set_unit(st, k, dl_idx, np.full(len(dl_idx), p10, np.float32))
            dlo = l90 - l10
            dn_.update(p10=p10, p90=p90, pct_rule=rule, dlogit_mean=dlo.mean(0).tolist(), dlogit_sd=dlo.std(0).tolist(),
                       calls_differ=float((l90.argmax(1) != l10.argmax(1)).mean()),
                       call_mix_p90=(np.bincount(l90.argmax(1), minlength=5) / len(dl_idx)).tolist(),
                       call_mix_p10=(np.bincount(l10.argmax(1), minlength=5) / len(dl_idx)).tolist())
            u['downstream'] = dn_
            # ---- steering
            top_major = int(np.argmax(u['bins']['top 1 %']['class_mix']))
            tg, rule = steer_targets(act, on, rng)
            v99 = p99_value(act, on)
            u['steer'] = dict(steer(N, st, k, act, v99, tg, top_major), rule=rule, top_major=CL[top_major])
            ctrl = [c for c in live if c != k]
            if ctrl:
                kc = int(rng.choice(ctrl)); ac = A[:, kc]; onc = ac > 0
                tgc, rulec = steer_targets(ac, onc, rng)
                u['control'] = dict(steer(N, st, kc, ac, p99_value(ac, onc), tgc, top_major), unit=uname(st, kc), rule=rulec)
            if st == 'z':
                tgn = np.flatnonzero(on)
                if len(tgn) > 4000: tgn = np.sort(rng.choice(tgn, 4000, replace=False))
                if len(tgn):
                    u['steer_neg'] = dict(steer(N, st, k, act, u['pct'][1], tgn, top_major), rule='on jets')
                    if ctrl:
                        tgcn = np.flatnonzero(onc)
                        if len(tgcn) > 4000: tgcn = np.sort(rng.choice(tgcn, 4000, replace=False))
                        if len(tgcn): u['control_neg'] = dict(steer(N, st, kc, ac, float(np.percentile(ac, 1)), tgcn, top_major), unit=uname(st, kc))
            res['units'].append(u)
            log(f'  {uname(st, k):5s} cost {cost[k]*100:5.2f}% on {on.mean():.2f} | {u["pattern"][:90]} | R² {desc["R2_1"]:.2f}/{desc["R2_2"]:.2f} | '
                f'k {cl.get("k")} main {cl.get("main_share", 0):.2f} {u["category"]} | steer {u["steer"]["changed"]*100:.1f}% '
                f'(ctrl {u.get("control", {}).get("changed", 0)*100:.1f}%) [{time.time() - t_u:.1f} s]')
        (OUT / f'atlas_{st}.json').write_text(json.dumps(res, indent=1, default=float))
        np.savez_compressed(OUT / f'images_{st}.npz', **imgs)
        log(f'stage {st} done: {len(live)} units ({time.time() - t_st:.0f} s)')
        try:
            from interp.e16_plot import cards
            cards(st)
        except Exception as e:  # the analysis is saved; cards can be redone with the cards mode
            import traceback; traceback.print_exc()
            log(f'cards for {st} failed: {e}')


def upstream(N, st, k, up, alpha, beta, e5):
    if st == 'P':
        c = k; a = alpha[:, c]; b = float(beta[c])
        return dict(kind='cut', cut=cut_text(a, b), alpha=a.tolist(), beta=b, slots=int(N.KEEPP[:, c].sum()))
    if st == 'S':
        pp = up['pp']; j = up['liveS'].index(k); K3 = N.net.KB['l3'][0]
        cnt = max(pp['cnt'][j], 1)
        sd_s = np.sqrt(np.maximum(pp['s2'][j] / cnt - (pp['s1'][j] / cnt) ** 2, 0))
        sd_g = np.sqrt(np.maximum(pp['g2'][j] / cnt - (pp['g1'][j] / cnt) ** 2, 0))
        sc = [(abs(K3[m, k]) * sd_s[m], f's{m}', float(K3[m, k])) for m in range(64)] + [(abs(K3[m, k]) * sd_g[m], f'g{m}', float(K3[m, k])) for m in range(64)]
        sc = sorted(sc, reverse=True)[:5]
        tab = pp['tab_sum'][j] / np.maximum(pp['tab_n'][j], 1); ton = pp['tab_on'][j] / np.maximum(pp['tab_n'][j], 1)
        slots = np.flatnonzero(N.KEEP[:, k])
        return dict(kind='hinge', units=[dict(unit=n, w=w, score=float(s)) for s, n, w in sc if s > 0],
                    table=tab.tolist(), table_on=ton.tolist(), table_n=pp['tab_n'][j].tolist(), pT_edges=PE, r_edges=RE,
                    window=ranges(slots), n_slots=int(len(slots)), e5=e5.get(k),
                    ctx_share=float((sd_g ** 2 * K3[:, k] ** 2).sum() / max((sd_g ** 2 * K3[:, k] ** 2).sum() + (sd_s ** 2 * K3[:, k] ** 2).sum(), 1e-12)))
    K = up['K']; spread = up['spread']; prev = up['prev']
    sc = sorted(((abs(K[m, k]) * spread[m], m) for m in range(K.shape[0])), reverse=True)[:5]
    return dict(kind='dense', units=[dict(unit=uname(prev, int(m)), w=float(K[m, k]), score=float(s)) for s, m in sc if s > 0])


def ranges(slots):
    if len(slots) == 0: return 'none'
    out, s0, p = [], slots[0], slots[0]
    for s in list(slots[1:]) + [None]:
        if s is not None and s == p + 1: p = s; continue
        out.append(f'{s0}' if s0 == p else f'{s0}–{p}')
        if s is not None: s0 = p = s
    return 'ranks ' + ', '.join(out)


def cut_text(a, b):
    """alpha·(pT, Δη, Δφ) + beta > 0 in words"""
    ap, ae, af = (float(v) for v in a)
    if abs(ae) < 1e-9 and abs(af) < 1e-9:
        if abs(ap) < 1e-12: return f'constant {b:.2f}'
        t = -b / ap
        return f'pT ramp {ap:+.3f}·pT {b:+.2f}: on for pT ' + ('> ' if ap > 0 else '< ') + f'{t:.1f} GeV' if (t > 0) else f'pT ramp {ap:+.3f}·pT {b:+.2f} (always on)'
    nrm = np.hypot(ae, af); th = np.degrees(np.arctan2(af, ae)); d = -b / nrm
    s = f'half-plane n̂·(Δη, Δφ) > {d:+.4f} rad, n̂ at {th:.0f}°'
    if abs(ap) > 1e-9: s += f' (+{ap:.3f}·pT)'
    return s


def run_a(N, alpha, beta, rng0):
    """first-layer cuts: reference rows"""
    pp = N.particle_pass([], want_a=True)
    mean = pp['a_sum'] / max(pp['a_n'], 1)
    const = pp['a_max'] - pp['a_min'] < 1e-9          # constant over real particles (E1: these change no call)
    cost = np.zeros(64); accab = np.zeros(64)
    acc0 = float((N.call == N.y).mean())
    for k in range(64):
        accab[k] = acc0
        if const[k]: continue
        m = np.zeros(64, np.float32); m[k] = 1; v = np.zeros(64, np.float32); v[k] = mean[k]
        c2 = N.net.run(N.x, override={'a': (m, v)}, batch=2500)['logits'].argmax(1)
        cost[k] = (c2 != N.call).mean(); accab[k] = (c2 == N.y).mean()
    live = [int(k) for k in np.argsort(-cost) if cost[k] >= LIVE_THR]
    log(f'a: {len(live)} live of 64 (non-constant {(~const).sum()}); costs ' + ', '.join(f'{k}:{cost[k]*100:.1f}%' for k in live))
    qa_sd = np.sqrt(np.maximum(pp['qa_2'] / max(pp['a_n'], 1) - (pp['qa_1'] / max(pp['a_n'], 1)) ** 2, 0))
    K2 = N.net.KB['l2'][0]
    liveP = [k for k in range(64) if N.R['P'][:, k].std() > 1e-9]
    units, imgs = [], {}
    J = N.J
    for k in live:
        rng = np.random.default_rng(k)
        u = dict(stage='a', unit=k, id=uname('a', k), cost=float(cost[k]), acc_ablated=float(accab[k]),
                 on_frac=float(pp['a_on'][k] / max(pp['a_n'], 1)), mean=float(mean[k]),
                 cut=cut_text(alpha[:, k], float(beta[k])), alpha=alpha[:, k].tolist(), beta=float(beta[k]),
                 slots_pooled=int(N.KEEPP[:, k].sum()), window=ranges(np.flatnonzero(N.KEEPP[:, k])),
                 pt_hist_on=pp['pt_hist_on'][k].tolist(), pt_hist_all=pp['pt_hist_all'].tolist(), pt_edges=pp['pt_edges'].tolist())
        u['pattern'] = u['cut']
        sc = sorted(((abs(K2[k, m]) * qa_sd[k], m) for m in range(64)), reverse=True)[:5]
        u['downstream'] = dict(units=[dict(unit=f's{int(m)}', w=float(K2[k, m]), score=float(s)) for s, m in sc if s > 0],
                               pooled=f'P{k}' + (' (live)' if k in liveP else ' (constant)'))
        imgs[uname('a', k)] = np.stack([pp['img_pt'][k] / J, pp['img_n'][k] / J])
        # steering: a_k set to its 99th percentile on the real particles where it is off, 2,000 jets with such particles
        smp = pp['a_sample'][:, k]; v99 = float(np.percentile(smp, 99))
        if v99 <= 0 and (smp > 0).any(): v99 = float(np.percentile(smp[smp > 0], 99))
        u['p99'] = v99
        idx = np.sort(rng.choice(J, min(2000, J), replace=False))
        a = N.net.run(N.x[idx], keep=('a',), batch=1000)['a'].astype(np.float32)
        real = N.x[idx][..., 0] > 0
        vals = a.copy(); off = real & (a[..., k] <= 0); vals[..., k] = np.where(off, v99, a[..., k])
        m = np.zeros(64, np.float32); m[k] = 1
        lo = N.net.run_jetwise(N.x[idx], {'a': (m, vals)}, batch=1000)['logits']
        new = lo.argmax(1); old = N.call[idx]; ch = new != old
        u['steer'] = dict(n=int(len(idx)), value=v99, particles_set=float(off.sum() / max(real.sum(), 1)), changed=float(ch.mean()),
                          flow=flow(old, new).tolist(), dest=(np.bincount(new[ch], minlength=5) / max(ch.sum(), 1)).tolist(),
                          rule='real particles where the cut is off', top_major=None, to_top_major=0.0)
        ctrl = [c for c in live if c != k]
        if ctrl:
            kc = int(rng.choice(ctrl)); smc = pp['a_sample'][:, kc]; vc = float(np.percentile(smc, 99))
            vals = a.copy(); offc = real & (a[..., kc] <= 0); vals[..., kc] = np.where(offc, vc, a[..., kc])
            mc = np.zeros(64, np.float32); mc[kc] = 1
            nc = N.net.run_jetwise(N.x[idx], {'a': (mc, vals)}, batch=1000)['logits'].argmax(1)
            u['control'] = dict(unit=uname('a', kc), changed=float((nc != old).mean()), value=vc, n=int(len(idx)))
        u['category'] = 'reference (cut)'
        units.append(u)
        log(f'  a{k:<3d} cost {cost[k]*100:5.2f}% | {u["cut"]} | steer {u["steer"]["changed"]*100:.1f}% (ctrl {u.get("control", {}).get("changed", 0)*100:.1f}%)')
    return dict(cost=cost.tolist(), acc_ablated=accab.tolist(), live=live, units=units, _imgs=imgs)


def refresh(st):
    """recompute the pattern lines of a stage's JSON from the stored response curves (no network); rewrite the JSON"""
    p = OUT / f'atlas_{st}.json'
    d = json.loads(p.read_text())
    if st == 'a': return d
    for u in d['units']:
        u['pattern'] = pattern_line(u['desc'], zero=(st == 'z'))
        nd = u.get('neg_desc')
        if nd and 'curve' in nd:
            nd['pattern'] = pattern_line(dict(best1=dict(nd['best1'], curve=nd['curve']), R2_1=nd.get('R2_1', 0), R2_2=nd['R2_2']), negate=True, zero=True)
    p.write_text(json.dumps(d, indent=1))
    return d


# ================================================================== summary
def summary():
    from interp.e16_plot import summary_png
    rows, tally, tally_forced = [], {}, {}
    per_stage = {}
    for st in ALL_STAGES:
        p = OUT / f'atlas_{st}.json'
        if not p.exists(): continue
        d = refresh(st); per_stage[st] = d
    for st, d in per_stage.items():
        T = {'one main job': 0, 'related jobs': 0, 'unrelated jobs': 0, 'unexplained': 0}; Tf = dict(T)
        for u in d['units']:
            row = dict(stage=st, unit=u['unit'], id=u['id'], cost=u['cost'], pattern=u['pattern'], category=u['category'])
            if st == 'a':
                row.update(R2=None, main_share=None, offshoot='', weak='', negative='',
                           steering=f"{u['steer']['changed']*100:.1f} % calls (control {u.get('control', {}).get('unit', '')}: {u.get('control', {}).get('changed', 0)*100:.1f} %)")
                rows.append(row); continue
            T[u['category']] += 1; Tf[u['category_forced_k2']] += 1
            cl = u['clusters']
            off = ''
            for c in cl.get('clusters', [])[1:]:
                if c['share_mass'] < 0.05: continue
                sh = 'same observable' if c.get('rho_with_unit_best', 0) >= RHO_SHARED else (f"tracks {c['own_best']['label']}" if c.get('own_best') else 'small')
                vs = ', '.join(f"{'higher' if v['dz'] > 0 else 'lower'} {OBS_SHORT[v['obs']]}" for v in c.get('vs_main', []))
                off += f"{c['share_mass']*100:.0f} %: {vs} ({sh}); "
            weak = ', '.join(f"{OBS_SHORT[v['obs']]} {fmt(v['med_ref'])}→{fmt(v['med'])}" for v in u.get('weak_vs_top', []))
            neg = ''
            if st == 'z':
                cm = u['bins']['lowest 1 %']['class_mix']; top = int(np.argmax(cm))
                neg = f"{CL[top]} {cm[top]*100:.0f} %; " + ', '.join(f"{'high' if v['dz'] > 0 else 'low'} {OBS_SHORT[v['obs']]} ({fmt(v['med'])})" for v in u['neg_vs_all'])
                if u.get('neg_desc'): neg += f"; {u['neg_desc']['pattern']} (R² {u['neg_desc']['R2_2']:.2f})"
            s = u['steer']; dest = int(np.argmax(s['dest']))
            stx = f"{s['changed']*100:.1f} % of {s['n']} calls, {s['dest'][dest]*100:.0f} % to {CL[dest]} (control {u.get('control', {}).get('unit', '')}: {u.get('control', {}).get('changed', 0)*100:.1f} %)"
            if 'steer_neg' in u:
                sn = u['steer_neg']; dn = int(np.argmax(sn['dest']))
                stx += f"; p1 on on-jets: {sn['changed']*100:.1f} %, {sn['dest'][dn]*100:.0f} % to {CL[dn]}"
            row.update(R2=[u['desc']['R2_1'], u['desc']['R2_2']], main_share=cl.get('main_share'), k=cl.get('k'), offshoot=off.strip('; '),
                       weak=weak, negative=neg, steering=stx, category_forced_k2=u['category_forced_k2'],
                       on_frac=u['on_frac'], top_class_mix=u['bins']['top 1 %']['class_mix'])
            rows.append(row)
        if st != 'a':
            tally[st] = T; tally_forced[st] = Tf
    tot = {k: sum(t[k] for t in tally.values()) for k in ('one main job', 'related jobs', 'unrelated jobs', 'unexplained')}
    totf = {k: sum(t[k] for t in tally_forced.values()) for k in tot}
    res = dict(thresholds=dict(live=f'mean ablation changes >= {LIVE_THR*100:.1f} % of calls', unexplained=f'best <= 2-observable held-out R² < {R2_MIN}',
                               one_main_job=f'one k-means cluster holds >= {MAIN_SHARE*100:.0f} % of the activation mass (k = 1 when the best silhouette over k = 2..4 < {SIL_MIN})',
                               related=f'every cluster with >= 10 % of the mass has its own best observable at |Spearman rho| >= {RHO_SHARED} with the unit\'s best observable',
                               forced_k2='sensitivity: k = argmax silhouette over 2..4 with no floor'),
               tally=tally, total=tot, tally_forced_k2=tally_forced, total_forced_k2=totf, rows=rows,
               live={st: d['live'] for st, d in per_stage.items()})
    (OUT / 'atlas_summary.json').write_text(json.dumps(res, indent=1))
    summary_png(res, OUT / 'atlas_summary.png')
    print(json.dumps(dict(tally=tally, total=tot, total_forced_k2=totf), indent=1))


OBS_SHORT = dict(mass='mass', sum_pt='ΣpT', n_particles='n', girth='girth', tau21='τ21', tau32='τ32', D2='D2', z_0='pT0/ΣpT', dr01='ΔR01',
                 sd_mass='m_SD')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['run', 'cards', 'summary'])
    ap.add_argument('--stages', default='z')
    ap.add_argument('--per-class', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=21)
    a = ap.parse_args()
    stages = list(ALL_STAGES) if a.stages == 'all' else a.stages.split(',')
    if a.mode == 'run':
        run(stages, a.per_class, a.seed)
    elif a.mode == 'cards':
        from interp.e16_plot import cards
        for st in stages:
            if (OUT / f'atlas_{st}.json').exists(): refresh(st)
        for st in stages:
            if (OUT / f'atlas_{st}.json').exists(): cards(st)
    else:
        summary()
    log('done')
