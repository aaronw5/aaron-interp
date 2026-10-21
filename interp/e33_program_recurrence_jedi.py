"""E33 (claims A1, A3, A10, A16; review weakness "multi-seed / retrained stability"; Paper B transfer of the naming method):
do the named program elements of JEDI-linear 3-feature n64 recur in other official checkpoints and in retrained copies?

Models (no training here):
  official_n64 (reference), official_n32, official_n128, perminv_n64 (3-feature-perminv), f16_n64 (16-feature) -- official
  checkpoints through interp.jedi_core.Jedi;  hgq150_s0..s2 (X12 HGQ arm, 150 % beta run, models/small/jedi-n64-x12-hgq-r150),
  float_s0, float_s1 (X12 float arm) -- final checkpoints via interp.x12_retrained_jedi.model_path;  untrained_n64 =
  Jedi(64, untrained=True, seed=0) (control row: weights redrawn, trained number formats kept).
One job per model:  python -m interp.e33_program_recurrence_jedi --model official_n32   -> $AI_RESULTS/jedi/e33_<model>.json
Combine:            python -m interp.e33_program_recurrence_jedi --combine              -> e33_summary.json, e33_recurrence.png

PLAN (fixed before the first run)
Samples: fit = jedi_core.sample(2000, 3301) (10,000 jets), test = sample(2000, 3302) (10,000 jets). n32: first 32 slots;
  n128: the same rows of val_ptetaphi_f16.npy with 128 slots (first 64 checked equal to sample()); f16: the same rows of
  val_c16_f16.npy (columns pt, etarel, phirel checked equal to sample()).  All formulas use (pT, r = hypot(deta, dphi), rank).
Step 0  accuracy on test (Wilson 95 %); own forward (per-stage override, below) checked against jedi_core.Jedi.run (official,
  untrained) or x12_retrained_jedi.Net.run (retrained) on 1,000 jets (max |dlogit|, calls differing).
Step 1  live units: activation s.d. > 1e-9 over the fit jets (per-particle stages a, s, u: over all slots); stages a P g s u S
  h4 h5 z.  Node-level ACDC-style pruning (E29 corruptions): mean = unit at its fit-sample mean (per slot for a, s, u);
  resample = unit at its value on a donor jet of the partner class (W<->Z, q<->g, t -> W or Z) among the 5 nearest in
  jet sum pT (rng 3303 fit, 3304 test).  Circuit = set of live units; every other unit (dead ones too) corrupted.
  Discovery jets = fit[::10] (1,000).  Receivers visited z, h5, h4, S, u, g, s, P, a; inside a stage units in ascending
  single-unit-ablation KL; a unit is removed if KL(full || circuit) rises by < tau over the current circuit; tau ladder
  1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2 (nested).  Each circuit scored on the 10,000 test jets:
  F = (acc_circ - 0.2) / (acc_full - 0.2) (share of above-chance accuracy).  circuit95 = the smallest ladder circuit with
  F >= 0.95.  Null: 500 random circuits with the same number of units per stage drawn from the live units, F on test[::10];
  p = (1 + #{F_null >= F_circ}) / 501.
Step 2  naming of every live S unit with the rank-window hinge library: jet features = sums over rank windows (prefix [0, k),
  suffix [k, n), all) of per-particle basis functions (count, pT, ln pT, pT^2, r, r^2, pT r, pT r^2, ln pT r, (r - t)+,
  (t - r)+, 1[pT < t], (pT - t)+, (t - pT)+, 1[r > t], ln pT (r - t)+, 1[pT < 2, r > 0.1]); greedy forward selection
  (orthogonal matching pursuit, <= 8 terms, stop when R^2 gains < 0.001) on the fit jets, R^2 on the test jets; named iff
  test R^2 >= 0.9.  Live l1 cuts typed as in X12 (spread-weighted share of the pT weight >= 0.9: ramp; of the angle weights
  >= 0.9: angular; 16-feature: largest E25 feature group >= 0.9).  Hand-named set = pure live cuts + named S units; share
  of it inside circuit95 vs the 500 random circuits (p as above).
Step 3  four reference roles, each a fixed formula family with fitted parameters, fitted per live S unit on the fit jets
  (unit and parameter chosen by fit R^2), R^2 on the test jets:
    hard  = a + b * sum of pT over the k hardest particles (k = 1..n)                      (S0-like, k = 14 in the reference)
    r2    = a + b sum pT r^2 + c sum r^2 + d N(pT < 2 GeV, r > 0.1)                       (S29-like)
    soft  = a + b sum over slots of rank >= k of (t - pT)+ (empty slot: pT = 0, adds t), t in {1, 1.5, 2, 2.5, 3, 5} GeV,
            k in {0, 4, 9, 13, 20}                                (S19-like soft / empty-slot count: ReLU(2.86 - 1.39 pT) on 13-63)
    core  = a + b sum over all slots of (t - r)+ (empty slot: r = 0, adds t), t in {0.05, 0.1, 0.165, 0.25, 0.4}
                                                                  (S30-like core sum: ReLU(3.47 - 21 r), 3.38 per empty slot)
  Changed after the 4-model smoke run, before the main run: soft was a step count N(pT > t, rank >= k) and core was
  a + b sum_real (t - r)+ + c N; a single particle crossing the step-count edge moved S19 by 0.07 of the count prediction
  (the unit is a pT ramp), and the extra N term let the soft-count unit win the core role.  Roles are assigned to distinct
  units: Hungarian assignment of the 4 roles to live S units maximising the summed fit R^2 (each unit at most one role).
  Matching across models: Hungarian assignment on |Pearson correlation| between the model's live S units and the reference's
  live S units on the shared test jets (reference = official_n64 run on the 3-feature 64-slot copy of the same jets); mean
  matched |corr| vs 250 random rotations of the model's standardised live-S space (p = (1 + #null >= obs) / 251).
Step 4  causal check of each role's best unit on the test jets (S recomputed by the full network, g included):
    hard  pT x 1.1 of one particle of rank j < k (capped at the pT of rank j - 1, so ranks are kept); control: the same
          nudge on a particle of rank >= k (formula predicts 0).
    r2    one particle's (deta, dphi) x 1.1 (radial +10 % at fixed pT, azimuth and rank); control: the same particle rotated
          about the jet axis by 0.1 rad (same displacement, formula predicts 0).
    soft  the hardest particle with rank >= max(k, 1) and 0.5 t <= pT <= t raised to min(1.2 t, pT of rank j - 1) (crosses
          the fitted pT edge, rank kept; jets where it cannot cross are dropped); control: the same pT step on a particle of
          rank >= max(k, 1) with pT > 1.5 t (capped likewise; predicts 0).  (Ramp formula: predicted change -b (t - pT_old).)
    core  radial +10 % of one particle with r < t / 1.1 (stays inside the core); control: radial +10 % of a particle with
          r > t (predicts 0) and the azimuthal 0.1 rad rotation.
  predicted dS = fitted formula(edited) - formula(unedited); slope = sum(obs * pred) / sum(pred^2) with a 1,000-draw jet
  bootstrap CI; control response = mean |obs_control| / mean |pred_in| (same jets), CI; paired sign-flip p (10,000) for
  E|obs_in| = E|obs_control|.  Object "found" iff test R^2 >= 0.9 and the slope CI covers 1.  Secondary (added after the
  smoke run, reported separately): found_equiv = R^2 >= 0.9 and slope CI inside [0.8, 1.25]; found_dir = R^2 >= 0.9, slope CI
  above 0 and |obs_in| > |obs_control| for every control (sign-flip p < 0.05).
  Null: 250 random rotations of the standardised live-S space; the same role fits on every rotated direction; null rate =
  share of rotations in which some direction reaches test R^2 >= 0.9 (R^2 only: an upper bound on the joint R^2 + slope rate).
  H0: a role recurs at the rotation rate; H1: above it.  Per model p = (1 + #rotations found) / 251, Holm over roles x models
  (combine); across the 9 trained models the count with the role found vs the Poisson-binomial distribution of the null
  rates (one-sided).
Step 5  z mass windows: every real particle's (deta, dphi) x 1.05 and x 0.95 (jet mass +-5 %); per jet and z unit response
  R = (z(1.05) - z(0.95)) / (ln 1.05 - ln 0.95) (= dz / d ln m).  Test jets in 10 GeV bins of jet mass (from the constituents,
  massless), 20-250 GeV, bins with >= 40 jets; window centre = mass where the bin-mean R crosses from positive to negative
  (linear interpolation) with |mean / se| >= 3 in a bin within 30 GeV on each side; 200-draw jet bootstrap CI of the centre.
  Reference windows (E6): z0 75, z7 / z14 94-98, z10 135-148 GeV.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

os.environ.setdefault('KERAS_BACKEND', 'jax')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
T0 = time.time()

MODELS = dict(official_n64=('official', '3-feature', 64), official_n32=('official', '3-feature', 32),
              official_n128=('official', '3-feature', 128), perminv_n64=('official', '3-feature-perminv', 64),
              f16_n64=('official', '16-feature', 64),
              hgq150_s0=('x12', 'hgq-r150_s0', 64), hgq150_s1=('x12', 'hgq-r150_s1', 64), hgq150_s2=('x12', 'hgq-r150_s2', 64),
              float_s0=('x12', 'float_s0', 64), float_s1=('x12', 'float_s1', 64),
              untrained_n64=('untrained', '3-feature', 64))
TRAINED = [m for m in MODELS if m != 'untrained_n64']
STAGES = ['a', 'P', 'g', 's', 'u', 'S', 'h4', 'h5', 'z']
PER = ('a', 's', 'u')
WIDTH = dict(a=64, P=64, g=64, s=64, u=64, S=64, h4=64, h5=32, z=16)
PRUNE_ORDER = ['z', 'h5', 'h4', 'S', 'u', 'g', 's', 'P', 'a']
TAUS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
ROLES = ['hard', 'r2', 'soft', 'core']
T_SOFT = [1.0, 1.5, 2.0, 2.5, 3.0, 5.0]; K_SOFT = [0, 4, 9, 13, 20]
T_CORE = [0.05, 0.1, 0.165, 0.25, 0.4]
REF_WINDOWS = dict(z0=75.0, z7=95.0, z14=97.0, z10=141.0)
BANDS = [(65, 85), (85, 110), (120, 160)]
KEYS = ('l1', 'qs', 'l2', 'dd', 'add', 'l3', 'qs1', 'd4', 'd5', 'd6', 'd7')
GROUPS16 = dict(pT_like=[5, 6, 3, 4, 2], momentum_xy=[0, 1], angle_rel=[8, 11], rot_frame=[9, 12, 15], radial=[13], absolute=[7, 10, 14])
B = 1000


def log(*a): print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, np.ndarray): return clean(o.tolist())
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    return o


def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def holm(ps):
    idx = [i for i, p in enumerate(ps) if p is not None]; order = sorted(idx, key=lambda i: ps[i]); m = len(order)
    out = [None] * len(ps); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i])); out[i] = run
    return out


# ================================================================================================ data
def rows(y, per, seed):
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), per, replace=False) for c in range(5)]))


def load(per, seed, n, f16):
    """-> xin (network input, raw physical), xk (pT, deta, dphi over the model's n slots), x3 (64-slot 3-feature copy), y"""
    from interp.jedi_core import sample
    from jetdistill.config import RESULTS, DATA
    x3, y, _ = sample(per, seed)
    yall = np.asarray(np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r')['y'])
    idx = rows(yall, per, seed); assert np.array_equal(yall[idx], y)
    chk = 0.0
    if n == 128:
        xk = np.asarray(np.load(Path(DATA) / 'val_ptetaphi_f16.npy', mmap_mode='r')[idx, :128]).astype(np.float32)
        chk = float(np.abs(xk[:, :64] - x3).max()); xin = xk
    elif f16:
        xin = np.asarray(np.load(Path(DATA) / 'val_c16_f16.npy', mmap_mode='r')[idx, :64]).astype(np.float32)
        chk = float(np.abs(xin[..., [5, 8, 11]] - x3).max()); xk = x3
    else:
        xk = x3[:, :n].copy(); xin = xk
    log(f'load sample({per}, {seed}): N {len(y)}, n {n}, f16 {f16}, max |copy - sample x| {chk:.3g}')
    if chk > 1e-3: raise SystemExit('ABORT: data copy differs from sample()')
    return xin, xk, x3, y, chk


def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi


def to16(X, x0, x1, dil=None):
    """16-feature copy of an edit made on the (pT, etarel, phirel) columns: changed particles get px, py, pz, e, erel, pt, ptrel,
    eta, etarel, phi, phirel, deltaR, costheta recomputed (particle mass, stored jet axis, jet pT and E kept); rotated-frame
    columns (etarot, phirot, costhetarel) unchanged for one-particle edits, multiplied by dil for an angular dilation"""
    Xd = X.astype(np.float64); x0 = x0.astype(np.float64); x1 = x1.astype(np.float64)
    real = x0[..., 0] > 0; ch = (x1 != x0).any(-1) & real
    pt0 = x0[..., 0]; pt1, er1, pr1 = x1[..., 0], x1[..., 1], x1[..., 2]
    eta = Xd[..., 7] + (er1 - x0[..., 1]); phi = wrap(Xd[..., 10] + (pr1 - x0[..., 2]))
    m2 = np.clip(Xd[..., 3] ** 2 - (Xd[..., 5] * np.cosh(Xd[..., 7])) ** 2, 0, None)
    e = np.sqrt(m2 + (pt1 * np.cosh(eta)) ** 2)
    rpt = Xd[..., 6] / np.where(pt0 > 0, pt0, 1.0); re_ = Xd[..., 4] / np.where(Xd[..., 3] > 0, Xd[..., 3], 1.0)
    new = {0: pt1 * np.cos(phi), 1: pt1 * np.sin(phi), 2: pt1 * np.sinh(eta), 3: e, 4: e * re_, 5: pt1, 6: pt1 * rpt, 7: eta,
           8: er1, 10: phi, 11: pr1, 13: np.hypot(er1, pr1), 14: np.tanh(eta)}
    E = Xd.copy()
    for c, v in new.items(): E[..., c] = np.where(ch, v, Xd[..., c])
    if dil is not None:
        for c in (9, 12, 15): E[..., c] = np.where(real, Xd[..., c] * dil, Xd[..., c])
    return E.astype(np.float32)


def jet_mass(xk):
    x = xk.astype(np.float64); pt, eta, phi = x[..., 0], x[..., 1], x[..., 2]
    p4 = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), pt * np.cosh(eta)], -1).sum(1)
    return np.sqrt(np.maximum(p4[:, 3] ** 2 - (p4[:, :3] ** 2).sum(1), 0.0))


def donors(y, ptsum, rng):
    """resample corruption: per jet a donor of the partner class among the 5 nearest in jet sum pT"""
    part = np.array([1, 0, 3, 2, -1])[y]; t = part < 0; part[t] = rng.choice([2, 3], int(t.sum()))
    out = np.empty(len(y), int)
    for c in range(5):
        cand = np.flatnonzero(y == c); pc = ptsum[cand]
        for i in np.flatnonzero(part == c):
            nn = np.argpartition(np.abs(pc - ptsum[i]), 5)[:5]; out[i] = cand[rng.choice(nn)]
    return out


# ================================================================================================ model
class Model:
    """JEDI-linear-architecture network with a per-stage override forward (all nine stages; values per jet)"""

    def __init__(self, name):
        import keras, hgq  # noqa: F401
        import jax, jax.numpy as jnp
        kind, fam, n = MODELS[name]; self.name, self.kind, self.n = name, kind, n
        self.f16 = fam.startswith('16-feature') if kind != 'x12' else False; self.F = 16 if self.f16 else 3
        if kind in ('official', 'untrained'):
            from interp.jedi_core import Jedi
            j = Jedi(n, fam, untrained=(kind == 'untrained'), seed=0); self.ref = j
            self.L = j.L; self._inp = lambda x: j.inputs(x, n); self.shift, self.scale = j.shift, j.scale; self.path = str(j.path)
        else:
            from interp.x12_retrained_jedi import model_path
            from jetdistill.config import DATA
            p, _, tj = model_path(fam, 'final'); self.path = str(p)
            model = keras.models.load_model(p, compile=False); self.L = {k: model.get_layer(k) for k in KEYS}
            st = json.loads((DATA / 'norm_stats.json').read_text())['64']
            sh, sc = np.float32(st['shift']), np.float32(st['scale']); self.shift, self.scale = np.asarray(st['shift']), np.asarray(st['scale'])
            self._inp = lambda x: (np.asarray(x[:, :64]).astype(np.float32) - sh) / sc
            self.test_json = json.loads(Path(tj).read_text()) if Path(tj).exists() else None
            self.ref = None
        L = self.L; KB = {}; self.KBnp = {}
        for k in ('l1', 'l2', 'dd', 'l3', 'd4', 'd5', 'd6', 'd7'):
            K, b = L[k].get_fused_qkernel_and_qbias(False, L[k].moving_mean, L[k].moving_variance)
            KB[k] = (jnp.asarray(K), jnp.asarray(b)); self.KBnp[k] = (np.asarray(K, np.float64), np.asarray(b, np.float64))

        def dense(k, x, act=True):
            z = jnp.einsum(L[k].equation, L[k].iq(x, training=False), KB[k][0]) + KB[k][1]
            return L[k].activation(z) if act and L[k].activation is not None else z

        def pool(k, a): return jnp.sum(L[k].iq(a, training=False), axis=L[k].axes, keepdims=L[k].keepdims) * L[k].scale

        def fwd(x, M, V):
            def ov(nm, t):
                if M is None: return t
                return t * (1 - M[nm]) + M[nm] * V[nm].reshape(t.shape)
            a = ov('a', dense('l1', x)); P = ov('P', pool('qs', a)); g = ov('g', dense('dd', P)); s = ov('s', dense('l2', a))
            qs_, qd_ = L['add'].iq([s, g], training=False)
            u = ov('u', dense('l3', qs_ + qd_)); S = ov('S', pool('qs1', u))
            h4 = ov('h4', dense('d4', S)); h5 = ov('h5', dense('d5', h4)); z = ov('z', dense('d6', h5, act=False))
            lo = dense('d7', jax.nn.relu(z)); J = x.shape[0]
            return dict(a=a, P=P.reshape(J, -1), g=g.reshape(J, -1), s=s, u=u, S=S.reshape(J, -1), h4=h4.reshape(J, -1),
                        h5=h5.reshape(J, -1), z=z.reshape(J, -1), logits=lo.reshape(J, -1))
        self._f0 = jax.jit(lambda x: fwd(x, None, None)); self._f = jax.jit(fwd); self.jnp = jnp

    def _pad(self, a, nb):
        if nb == B: return a
        return np.concatenate([a, np.repeat(a[-1:], B - nb, 0)])

    def clean_stages(self, xb):
        """raw jets (<= B) -> f0 outputs (jax arrays, padded to B)"""
        return self._f0(self._inp(self._pad(xb, len(xb))))

    def run(self, x, keep=('logits',), M=None, V=None):
        """M: {stage: (units,) 1 = override}; V: dict of per-stage arrays with leading dim B (constant), or callable (i0, i1) -> dict"""
        jnp = self.jnp; out = {k: [] for k in keep}
        Mj = None if M is None else {st: jnp.asarray(np.asarray(M.get(st, np.zeros(WIDTH[st])), np.float32)) for st in STAGES}
        for i in range(0, len(x), B):
            xb = x[i:i + B]; nb = len(xb); xn = self._inp(self._pad(xb, nb))
            if Mj is None: r = self._f0(xn)
            else:
                Vb = V(i, i + nb) if callable(V) else V
                r = self._f(xn, Mj, Vb)
            for k in keep: out[k].append(np.asarray(r[k][:nb], np.float64))
        return {k: np.concatenate(v) for k, v in out.items()}


def mean_V(means):
    """constant corruption values broadcast to the batch"""
    import jax.numpy as jnp
    return {st: jnp.asarray(np.broadcast_to(means[st], (B,) + means[st].shape).astype(np.float32)) for st in STAGES}


def donor_V(model, x, dn):
    return lambda i0, i1: model.clean_stages(x[dn[i0:i1]])


def log_softmax(lo):
    m = lo.max(1, keepdims=True); return lo - m - np.log(np.exp(lo - m).sum(1, keepdims=True))


# ================================================================================================ step 1: live units, ACDC
def stage_stats(model, x):
    """per-stage mean (per slot for a, s, u) and s.d. over x (streamed)"""
    acc = {st: [0.0, 0.0, 0] for st in STAGES}
    for i in range(0, len(x), B):
        xb = x[i:i + B]; nb = len(xb); r = model.clean_stages(xb)
        for st in STAGES:
            v = np.asarray(r[st][:nb], np.float64); acc[st][0] = acc[st][0] + v.sum(0); acc[st][1] = acc[st][1] + (v ** 2).sum(0); acc[st][2] += nb
    means, live = {}, {}
    for st in STAGES:
        s1, s2, nn = acc[st]; m = s1 / nn; means[st] = m
        if st in PER:
            mu = s1.sum(0) / (nn * s1.shape[0]); var = s2.sum(0) / (nn * s1.shape[0]) - mu ** 2
        else:
            var = s2 / nn - m ** 2
        live[st] = [int(c) for c in np.flatnonzero(np.sqrt(np.maximum(var, 0)) > 1e-9)]
    return means, live


def mask_of(circ):
    """circuit {stage: kept units} -> override mask (1 = corrupted) for every unit not kept (dead units included)"""
    M = {}
    for st in STAGES:
        m = np.ones(WIDTH[st], np.float32); m[circ.get(st, [])] = 0; M[st] = m
    return M


def acdc(model, xd, live, V, lp_full, taus=TAUS):
    pf = np.exp(lp_full)

    def kl(circ):
        lo = model.run(xd, M=mask_of(circ), V=V)['logits']; return float((pf * (lp_full - log_softmax(lo))).sum(1).mean())
    full = {st: list(live[st]) for st in STAGES}
    single = {}
    for st in STAGES:
        for c in live[st]:
            cc = {k: list(v) for k, v in full.items()}; cc[st] = [u for u in cc[st] if u != c]; single[(st, c)] = kl(cc)
    log(f'   single-unit KL done ({len(single)} units); max {max(single.values()):.4f}')
    circ = {k: list(v) for k, v in full.items()}; cur = kl(circ); ladder = []
    for tau in taus:
        for st in PRUNE_ORDER:
            for c in sorted(circ[st], key=lambda c: single[(st, c)]):
                cc = {k: list(v) for k, v in circ.items()}; cc[st] = [u for u in cc[st] if u != c]; k_ = kl(cc)
                if k_ - cur < tau: circ, cur = cc, k_
        ladder.append(dict(tau=tau, kl=cur, circuit={k: list(v) for k, v in circ.items()}, size=sum(len(v) for v in circ.values()),
                           per_stage={k: len(v) for k, v in circ.items()}))
        log(f'   tau {tau:g}: KL {cur:.4f}, size {ladder[-1]["size"]}, per stage {ladder[-1]["per_stage"]}')
    return ladder, {f'{k[0]}{k[1]}': v for k, v in single.items()}


def acc_F(lo, y, acc_full):
    a = float((lo.argmax(1) == y).mean()); return a, (a - 0.2) / (acc_full - 0.2)


# ================================================================================================ step 2/3: features
def particles(xk):
    x = xk.astype(np.float64); pt = x[..., 0]; real = pt > 0; r = np.where(real, np.hypot(x[..., 1], x[..., 2]), 0.0)
    lpt = np.where(real, np.log(np.where(real, pt, 1.0)), 0.0); return pt, r, real, lpt


def library(xk):
    pt, r, real, lpt = particles(xk); n = xk.shape[1]; rf = real.astype(np.float64)
    phi = dict(cnt=rf, pT=pt, lnpT=lpt, pT2=pt ** 2, r=r, r2=r ** 2, pTr=pt * r, pTr2=pt * r ** 2, lnpTr=lpt * r,
               soft_wide=rf * ((pt < 2) & (r > 0.1)))
    for t in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3): phi[f'(r-{t})+'] = rf * np.maximum(r - t, 0)
    for t in (0.04, 0.08, 0.12, 0.165, 0.25): phi[f'({t}-r)+'] = rf * np.maximum(t - r, 0)
    for t in (1, 2, 3, 5): phi[f'1[pT<{t}]'] = rf * (pt < t)
    for t in (2, 5, 10, 30): phi[f'(pT-{t})+'] = rf * np.maximum(pt - t, 0)
    for t in (2, 5): phi[f'({t}-pT)+'] = rf * np.maximum(t - pt, 0)
    for t in (0.1, 0.2): phi[f'1[r>{t}]'] = rf * (r > t)
    for t in (0.1, 0.2): phi[f'lnpT(r-{t})+'] = lpt * np.maximum(r - t, 0)
    pre = [k for k in (1, 2, 3, 4, 6, 9, 14, 20, 26, 39, 64, 96) if k < n]; suf = [k for k in (1, 2, 4, 9, 14, 20, 26, 39, 64, 96) if k < n]
    cols, names = [], []
    for nm, v in phi.items():
        cs = np.cumsum(v, 1); tot = cs[:, -1]
        cols.append(tot); names.append(f'{nm}[all]')
        for k in pre: cols.append(cs[:, k - 1]); names.append(f'{nm}[0:{k}]')
        for k in suf: cols.append(tot - cs[:, k - 1]); names.append(f'{nm}[{k}:]')
    return np.stack(cols, 1), names


def omp(Xf, tf, Xt, tt, names, maxk=8, tol=1e-3):
    mu, sd = Xf.mean(0), Xf.std(0); ok = np.flatnonzero(sd > 1e-9)
    R = (Xf[:, ok] - mu[ok]) / sd[ok]; res = tf - tf.mean(); sst = float((res ** 2).sum())
    if sst <= 1e-12: return dict(r2_fit=None, r2_test=None, terms=[])
    norms = (R ** 2).sum(0); chosen = []; r2 = 0.0
    for _ in range(maxk):
        sc = (R.T @ res) ** 2 / np.maximum(norms, 1e-12); sc[norms < 1e-6 * len(res)] = -1; sc[chosen] = -1
        j = int(np.argmax(sc)); gain = sc[j] / sst
        if gain < tol: break
        q = R[:, j] / np.sqrt(norms[j]); res = res - q * (q @ res); R -= np.outer(q, q @ R); norms = (R ** 2).sum(0)
        chosen.append(j); r2 += gain
    cols = ok[chosen]; A = np.column_stack([np.ones(len(tf)), Xf[:, cols]]); w, *_ = np.linalg.lstsq(A, tf, rcond=None)
    pt_ = np.column_stack([np.ones(len(tt)), Xt[:, cols]]) @ w; vt = ((tt - tt.mean()) ** 2).sum()
    r2t = float(1 - ((tt - pt_) ** 2).sum() / vt) if vt > 1e-12 else None
    return dict(r2_fit=float(1 - ((tf - A @ w) ** 2).sum() / sst), r2_test=r2t, terms=[names[c] for c in cols], coef=w.tolist())


def role_feats(xk):
    """per role: list of (param label, feature matrix (J, p))"""
    pt, r, real, lpt = particles(xk); n = xk.shape[1]; rf = real.astype(np.float64); rank = np.arange(n)[None]
    cs = np.cumsum(pt, 1)
    out = dict(hard=[(dict(k=k), cs[:, k - 1:k]) for k in range(1, n + 1)])
    out['r2'] = [(dict(), np.stack([(pt * r ** 2).sum(1), (rf * r ** 2).sum(1), (rf * ((pt < 2) & (r > 0.1))).sum(1)], 1))]
    out['soft'] = [(dict(t=t, k=k), (np.maximum(t - pt, 0) * (rank >= k)).sum(1)[:, None]) for t in T_SOFT for k in K_SOFT if k < n]
    out['core'] = [(dict(t=t), np.maximum(t - r, 0).sum(1)[:, None]) for t in T_CORE]
    return out


def role_value(role, prm, xk):
    pt, r, real, lpt = particles(xk); rf = real.astype(np.float64); n = xk.shape[1]
    if role == 'hard': return pt[:, :prm['k']].sum(1)[:, None]
    if role == 'r2': return np.stack([(pt * r ** 2).sum(1), (rf * r ** 2).sum(1), (rf * ((pt < 2) & (r > 0.1))).sum(1)], 1)
    if role == 'soft': return (np.maximum(prm['t'] - pt, 0) * (np.arange(n)[None] >= prm['k'])).sum(1)[:, None]
    return np.maximum(prm['t'] - r, 0).sum(1)[:, None]


def fit_set(Zf, Zt, Gf, Gt):
    """linear fit of every column of Z on [1, G]: fit R^2, test R^2 (per column), weights"""
    Af = np.column_stack([np.ones(len(Gf)), Gf]); At = np.column_stack([np.ones(len(Gt)), Gt])
    W, *_ = np.linalg.lstsq(Af, Zf, rcond=None)
    vf = ((Zf - Zf.mean(0)) ** 2).sum(0); vt = ((Zt - Zt.mean(0)) ** 2).sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        r2f = 1 - ((Zf - Af @ W) ** 2).sum(0) / vf; r2t = 1 - ((Zt - At @ W) ** 2).sum(0) / vt
    r2f[vf <= 1e-12] = -np.inf; r2t[vt <= 1e-12] = -np.inf
    return r2f, r2t, W


def single_fit(Zf, Zt, Ff, Ft):
    """one feature at a time (columns of F) x every unit (columns of Z): fit R^2 (K, U), test R^2 (K, U), a, b"""
    J = len(Zf); mf = Ff.mean(0); mz = Zf.mean(0); Fc = Ff - mf; Zc = Zf - mz
    cov = Fc.T @ Zc / J; vF = (Fc ** 2).mean(0); vZ = (Zc ** 2).mean(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        b = cov / vF[:, None]; r2f = cov ** 2 / (vF[:, None] * vZ[None])
    b[~np.isfinite(b)] = 0; r2f[~np.isfinite(r2f)] = -np.inf
    a = mz[None] - b * mf[:, None]
    Jt = len(Zt); sz = Zt.sum(0); sz2 = (Zt ** 2).sum(0); sf = Ft.sum(0); sf2 = (Ft ** 2).sum(0); szf = Ft.T @ Zt
    sse = sz2[None] - 2 * a * sz[None] - 2 * b * szf + Jt * a ** 2 + 2 * a * b * sf[:, None] + b ** 2 * sf2[:, None]
    vt = sz2 - sz ** 2 / Jt
    with np.errstate(invalid='ignore', divide='ignore'):
        r2t = 1 - sse / vt[None]
    r2t[:, vt <= 1e-12] = -np.inf
    return r2f, r2t, a, b


def roles_fit(Zf, Zt, RF, RT):
    """per role and unit: best parameter by fit R^2; roles assigned to distinct units (Hungarian on fit R^2); test R^2 reported"""
    from scipy.optimize import linear_sum_assignment
    U = Zf.shape[1]; info = {}; ar = np.arange(U)
    for role in ROLES:
        if role != 'r2':
            Ff = np.concatenate([g for _, g in RF[role]], 1); Ft = np.concatenate([g for _, g in RT[role]], 1)
            r2f, r2t, a, b = single_fit(Zf, Zt, Ff, Ft); kb = np.argmax(r2f, 0)
            info[role] = dict(f=r2f[kb, ar], t=r2t[kb, ar], prm=[RF[role][k][0] for k in kb], coef=[[a[k, u], b[k, u]] for u, k in enumerate(kb)])
        else:
            r2f, r2t, W = fit_set(Zf, Zt, RF['r2'][0][1], RT['r2'][0][1])
            info[role] = dict(f=r2f, t=r2t, prm=[{}] * U, coef=W.T.tolist())
    Fm = np.array([info[r]['f'] for r in ROLES]); Fm[~np.isfinite(Fm)] = -1
    ri, ci = linear_sum_assignment(-Fm); asg = dict(zip(ri, ci)); res = {}
    for q, role in enumerate(ROLES):
        if q not in asg: res[role] = dict(unit_idx=None, param=None, r2_fit=None, r2_test=None, coef=None, per_unit_test=info[role]['t']); continue
        u = int(asg[q]); I = info[role]
        res[role] = dict(unit_idx=u, param=I['prm'][u], r2_fit=float(I['f'][u]), r2_test=float(I['t'][u]), coef=[float(v) for v in I['coef'][u]],
                         per_unit_test=I['t'], best_unit_unassigned=int(np.argmax(I['f'])))
    return res


def rand_rot(U, rng):
    q, r = np.linalg.qr(rng.standard_normal((U, U))); return q * np.sign(np.diag(r))[None]


# ================================================================================================ step 4: edits
def boot_slope(o, p, rng, nb=1000):
    o = np.asarray(o, np.float64); p = np.asarray(p, np.float64); n = len(o)
    if n < 10 or (p ** 2).sum() <= 0: return None, [None, None]
    s = float((o * p).sum() / (p ** 2).sum()); bs = []
    for _ in range(nb):
        i = rng.integers(0, n, n); d = (p[i] ** 2).sum(); bs.append((o[i] * p[i]).sum() / d if d > 0 else np.nan)
    return s, [float(np.nanquantile(bs, .025)), float(np.nanquantile(bs, .975))]


def boot_ratio(a, b, rng, nb=1000):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64); n = len(a)
    if n < 10 or b.mean() == 0: return None, [None, None]
    bs = []
    for _ in range(nb):
        i = rng.integers(0, n, n); d = b[i].mean(); bs.append(a[i].mean() / d if d > 0 else np.nan)
    return float(a.mean() / b.mean()), [float(np.nanquantile(bs, .025)), float(np.nanquantile(bs, .975))]


def signflip_p(d, rng, P_=10000):
    d = np.asarray(d, np.float64); n = len(d)
    if n == 0 or not (d != 0).any(): return 1.0
    obs = abs(d.mean()); cnt = 0
    for i in range(0, P_, 500):
        s = rng.choice([-1.0, 1.0], size=(min(500, P_ - i), n)); cnt += int((np.abs((s * d).mean(1)) >= obs - 1e-15).sum())
    return (1 + cnt) / (1 + P_)


def pick(elig, rng):
    """per jet a random True column of elig (J, n); -1 if none"""
    u = rng.random(elig.shape) * elig; j = np.argmax(u, 1); j[~elig.any(1)] = -1; return j


def make_edits(role, prm, xk, rng):
    """-> dict name -> (jet index, edited xk rows) for the in-window edit and controls"""
    x = xk.astype(np.float64); pt, r, real, _ = particles(xk); J, n = pt.shape; rank = np.arange(n)[None]
    prev = np.concatenate([np.full((J, 1), np.inf), pt[:, :-1]], 1)
    out = {}

    def ptnudge(j, newpt, name):
        ok = j >= 0; I = np.flatnonzero(ok); e = x[I].copy(); e[np.arange(len(I)), j[I], 0] = newpt[I]; out[name] = (I, e.astype(np.float32))

    def radial(j, fac, name):
        I = np.flatnonzero(j >= 0); e = x[I].copy(); e[np.arange(len(I)), j[I], 1:3] *= fac; out[name] = (I, e.astype(np.float32))

    def azim(j, dth, name):
        I = np.flatnonzero(j >= 0); e = x[I].copy(); a = np.arange(len(I)); de, dp = e[a, j[I], 1].copy(), e[a, j[I], 2].copy()
        c, s = np.cos(dth), np.sin(dth); e[a, j[I], 1] = c * de - s * dp; e[a, j[I], 2] = s * de + c * dp; out[name] = (I, e.astype(np.float32))
    if role == 'hard':
        k = prm['k']; j = pick(real & (rank < k), rng); jj = np.maximum(j, 0)
        newpt = np.minimum(1.1 * pt[np.arange(J), jj], prev[np.arange(J), jj]); ptnudge(j, newpt, 'in')
        jc = pick(real & (rank >= max(k, 1)), rng); jjc = np.maximum(jc, 0)
        ptnudge(jc, np.minimum(1.1 * pt[np.arange(J), jjc], prev[np.arange(J), jjc]), 'ctrl_outside')
    elif role == 'r2':
        j = pick(real & (r > 0.005), rng); radial(j, 1.1, 'in'); azim(j, 0.1, 'ctrl_azimuthal')
    elif role == 'soft':
        t, k = prm['t'], max(prm['k'], 1)
        cand = real & (rank >= k) & (pt <= t) & (pt >= 0.5 * t)
        j = np.where(cand.any(1), np.argmax(cand, 1), -1); jj = np.maximum(j, 0); A = np.arange(J)
        newpt = np.minimum(1.2 * t, prev[A, jj]); j[(newpt <= t * 1.001)] = -1; ptnudge(j, newpt, 'in')
        step = np.where(j >= 0, newpt - pt[A, jj], 0.5 * t)
        jc = pick(real & (rank >= k) & (pt > 1.5 * t), rng); jjc = np.maximum(jc, 0)
        ptnudge(jc, np.minimum(pt[A, jjc] + step, prev[A, jjc]), 'ctrl_far_from_edge')
    else:
        t = prm['t']; j = pick(real & (r < t / 1.1) & (r > 0.005), rng); radial(j, 1.1, 'in')
        jc = pick(real & (r > t), rng); radial(jc, 1.1, 'ctrl_outside'); azim(j, 0.1, 'ctrl_azimuthal')
    return out


# ================================================================================================ step 5: z windows
def z_windows(model, xin, xk, rng, nb=200):
    def dil(s):
        e = xk.astype(np.float64).copy(); rl = e[..., 0] > 0; e[..., 1:3] *= np.where(rl, s, 1.0)[..., None]; e = e.astype(np.float32)
        return to16(xin, xk, e, dil=s) if model.f16 else e
    zp = model.run(dil(1.05), keep=('z',))['z']; zm = model.run(dil(0.95), keep=('z',))['z']
    R = (zp - zm) / (np.log(1.05) - np.log(0.95)); m = jet_mass(xk)
    edges = np.arange(20, 251, 10); bi = np.digitize(m, edges) - 1; nbins = len(edges) - 1; ctr = 0.5 * (edges[1:] + edges[:-1])

    def profile(idx):
        mu = np.full((nbins, R.shape[1]), np.nan); se = np.full((nbins, R.shape[1]), np.nan); cnt = np.zeros(nbins, int)
        b_ = bi[idx]; Rr = R[idx]
        for b in range(nbins):
            s = b_ == b; cnt[b] = int(s.sum())
            if cnt[b] >= 40: mu[b] = Rr[s].mean(0); se[b] = Rr[s].std(0) / np.sqrt(cnt[b])
        return mu, se, cnt

    def crossings(mu, se):
        out = []
        for c in range(mu.shape[1]):
            cs = []
            for b in range(nbins - 1):
                if not (np.isfinite(mu[b, c]) and np.isfinite(mu[b + 1, c])): continue
                if mu[b, c] > 0 >= mu[b + 1, c]:
                    x0 = ctr[b] + 10 * mu[b, c] / (mu[b, c] - mu[b + 1, c])
                    lo = [bb for bb in range(nbins) if np.isfinite(mu[bb, c]) and x0 - 30 <= ctr[bb] < x0 and mu[bb, c] / se[bb, c] >= 3]
                    hi = [bb for bb in range(nbins) if np.isfinite(mu[bb, c]) and x0 < ctr[bb] <= x0 + 30 and mu[bb, c] / se[bb, c] <= -3]
                    if lo and hi: cs.append(float(x0))
            out.append(cs)
        return out
    mu, se, cnt = profile(np.arange(len(m))); obs = crossings(mu, se)
    boots = [[[] for _ in obs[c]] for c in range(len(obs))]
    for _ in range(nb):
        i = rng.integers(0, len(m), len(m)); mb, sb, _ = profile(i)
        for c in range(len(obs)):
            for q, x0 in enumerate(obs[c]):
                # crossing of the bootstrap profile nearest the observed one (sign change only)
                xs = []
                for b in range(nbins - 1):
                    if np.isfinite(mb[b, c]) and np.isfinite(mb[b + 1, c]) and mb[b, c] > 0 >= mb[b + 1, c]:
                        xs.append(ctr[b] + 10 * mb[b, c] / (mb[b, c] - mb[b + 1, c]))
                if xs: boots[c][q].append(min(xs, key=lambda v: abs(v - x0)))
    units = {}
    for c in range(len(obs)):
        if not obs[c]: continue
        units[f'z{c}'] = [dict(centre=x0, ci=[float(np.quantile(boots[c][q], .025)), float(np.quantile(boots[c][q], .975))] if len(boots[c][q]) > 20 else [None, None])
                          for q, x0 in enumerate(obs[c])]
    return dict(units=units, bin_centres=ctr, bin_counts=cnt, profile_mean=mu, profile_se=se,
                band_hits={f'{lo}-{hi}': sorted({u for u, w in units.items() for e in w if lo <= e['centre'] < hi}) for lo, hi in BANDS})


# ================================================================================================ one model
def run_model(name, smoke=False):
    from scipy.optimize import linear_sum_assignment
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f'e33_{name}' + ('_smoke' if smoke else ''); JSON = OUT / f'{tag}.json'
    R = dict(node='E33', model=name, spec=MODELS[name], plan='module docstring (fixed before the first run)', smoke=smoke)
    def save(): JSON.write_text(json.dumps(clean(R), indent=1))
    per = 200 if smoke else 2000; NRAND = 20 if smoke else 500; NROT = 10 if smoke else 250; NBOOT = 100 if smoke else 1000
    rng = np.random.default_rng(3300)
    kind, fam, n = MODELS[name]
    model = Model(name); R['path'] = model.path
    f16 = model.f16
    xf, xkf, x3f, yf, c1 = load(per, 3301, n, f16); xt, xkt, x3t, yt, c2 = load(per, 3302, n, f16)
    R['sample'] = dict(fit=f'sample({per}, 3301)', test=f'sample({per}, 3302)', N_fit=len(yf), N_test=len(yt), copy_check=[c1, c2])

    # ---- step 0
    St = model.run(xt, keep=('S', 'z', 'logits')); lo_t = St['logits']; acc_full = float((lo_t.argmax(1) == yt).mean())
    xs = xt[:B]
    if model.ref is not None:
        rr = model.ref.run(xs, keep=('logits',))['logits']
    else:
        from interp.x12_retrained_jedi import Net
        net_ = Net('x12', model.path); rr = net_.run(xs, keep=('logits',))['logits']; del net_
    R['step0'] = dict(acc=acc_full, acc_ci=wilson(int(acc_full * len(yt)), len(yt)), forward_check_maxabs=float(np.abs(rr - lo_t[:B]).max()),
                      forward_check_calls_differ=int((rr.argmax(1) != lo_t[:B].argmax(1)).sum()),
                      stored_test_acc=(model.test_json or {}).get('accuracy') if model.kind == 'x12' else None)
    log(f'step 0: {name} acc {acc_full:.4f} {R["step0"]["acc_ci"]}; forward check max|dlogit| {R["step0"]["forward_check_maxabs"]:.3g}, '
        f'{R["step0"]["forward_check_calls_differ"]} calls differ'); save()

    # ---- step 1
    means, live = stage_stats(model, xf)
    R['live'] = live; R['n_live'] = {st: len(v) for st, v in live.items()}; R['n_live_total'] = sum(R['n_live'].values())
    log(f'step 1: live units {R["n_live"]} total {R["n_live_total"]}')
    xd = xf[::10]; yd = yf[::10]; lp_d = log_softmax(model.run(xd)['logits'])
    ptsum = lambda xk: xk[..., 0].astype(np.float64).sum(1)  # noqa: E731
    dn_d = donors(yd, ptsum(xkf[::10]), np.random.default_rng(3303)); dn_t = donors(yt, ptsum(xkt), np.random.default_rng(3304))
    Vmean = mean_V(means)
    sub = np.arange(0, len(yt), 10); xs_ = xt[sub]; ys_ = yt[sub]
    acc_full_sub = float((lo_t[sub].argmax(1) == ys_).mean())
    R['acdc'] = {}
    skip_acdc = acc_full - 0.2 < 0.02     # constant-call network (untrained redraw): F undefined
    if skip_acdc: log('step 1: accuracy at chance; circuit search skipped'); R['acdc_skipped'] = True
    for corr in (() if skip_acdc else ('mean', 'resample')):
        Vd = Vmean if corr == 'mean' else model.clean_stages(xd[dn_d])
        ladder, single = acdc(model, xd, live, Vd, lp_d, [1e-3, 3e-2] if smoke else TAUS)
        Vt = Vmean if corr == 'mean' else donor_V(model, xt, dn_t)
        allc = {st: [] for st in STAGES}
        acc0 = float((model.run(xt, M=mask_of(allc), V=Vt)['logits'].argmax(1) == yt).mean())
        for L_ in ladder:
            lo = model.run(xt, M=mask_of(L_['circuit']), V=Vt)['logits']; L_['acc'], L_['F'] = acc_F(lo, yt, acc_full)
            log(f'   [{corr}] tau {L_["tau"]:g}: size {L_["size"]} acc {L_["acc"]:.4f} F {L_["F"]:.3f}')
        ok = [L_ for L_ in ladder if L_['F'] >= 0.95]
        c95 = min(ok, key=lambda L_: L_['size']) if ok else ladder[0]
        # random null on test[::10]
        Vs = Vmean if corr == 'mean' else model.clean_stages(xt[dn_t[sub]])
        lo95 = model.run(xs_, M=mask_of(c95['circuit']), V=Vs)['logits']; _, F95s = acc_F(lo95, ys_, acc_full_sub)
        Fnull = []; draws = []
        for _ in range(NRAND):
            rc = {st: sorted(rng.choice(live[st], len(c95['circuit'][st]), replace=False).tolist()) for st in STAGES}
            draws.append(rc); Fnull.append(acc_F(model.run(xs_, M=mask_of(rc), V=Vs)['logits'], ys_, acc_full_sub)[1])
        Fnull = np.array(Fnull)
        R['acdc'][corr] = dict(ladder=ladder, single_kl=single, acc0=acc0, circuit95=c95, reached95=bool(ok),
                               size95=c95['size'], share95=c95['size'] / R['n_live_total'], F95_sub=F95s,
                               null=dict(n=NRAND, F_mean=float(Fnull.mean()), F_q=[float(np.quantile(Fnull, q)) for q in (.025, .5, .975)],
                                         p=float((1 + (Fnull >= F95s).sum()) / (1 + NRAND))))
        R['acdc'][corr]['_draws'] = draws
        log(f'step 1 [{corr}]: acc0 {acc0:.4f}; circuit95 size {c95["size"]} / {R["n_live_total"]} (tau {c95["tau"]:g}, F {c95["F"]:.3f}, '
            f'reached {bool(ok)}); F on test[::10] {F95s:.3f} vs random {Fnull.mean():.3f} [{np.quantile(Fnull, .025):.3f}, '
            f'{np.quantile(Fnull, .975):.3f}], p {R["acdc"][corr]["null"]["p"]:.4f}')
        save()

    # ---- step 2: naming
    Sf = model.run(xf, keep=('S',))['S']; Stt = St['S']; liveS = live['S']
    LF, names = library(xkf); LT, _ = library(xkt)
    R['naming'] = {}
    for c in liveS:
        R['naming'][f'S{c}'] = omp(LF, Sf[:, c], LT, Stt[:, c], names)
    del LF, LT
    named_S = [c for c in liveS if (R['naming'][f'S{c}']['r2_test'] or -1) >= 0.9]
    # l1 cut types
    K1 = model.KBnp['l1'][0].reshape(model.F, 64); real = xkf[..., 0] > 0
    sc = np.broadcast_to(np.asarray(model.scale, np.float64).reshape(-1)[-model.F:], (model.F,))
    alpha = K1 / sc[:, None]; Xr = xf[real].astype(np.float64); sdx = Xr.std(0)
    w2 = (np.abs(alpha) * sdx[:, None]) ** 2; tot = w2.sum(0)
    types = {}
    for c in live['a']:
        if tot[c] == 0: types[c] = 'constant'; continue
        if model.F == 3:
            spt = w2[0, c] / tot[c]; sang = (w2[1, c] + w2[2, c]) / tot[c]
            types[c] = 'ramp' if spt >= 0.9 else ('angular' if sang >= 0.9 else 'mixed')
        else:
            gs = {g: w2[ix, c].sum() / tot[c] for g, ix in GROUPS16.items()}; gb = max(gs, key=gs.get)
            types[c] = gb if gs[gb] >= 0.9 else 'mixed'
    pure_a = [c for c, t in types.items() if t not in ('mixed', 'constant')]
    R['l1_types'] = {f'a{c}': t for c, t in types.items()}; R['named'] = dict(pure_l1=pure_a, named_S=named_S)
    log(f'step 2: {len(named_S)} / {len(liveS)} live S units named (test R2 >= 0.9); l1 types {dict(zip(*np.unique(list(types.values()), return_counts=True)))}; '
        + ', '.join(f'S{c}:{R["naming"][f"S{c}"]["r2_test"]:.3f}' for c in liveS if R['naming'][f'S{c}']['r2_test'] is not None))
    for corr in (() if skip_acdc else ('mean', 'resample')):
        A = R['acdc'][corr]; c95 = A['circuit95']['circuit']
        hn = [('a', c) for c in pure_a] + [('S', c) for c in named_S]
        inside = sum(c in c95[st] for st, c in hn)
        nulls = np.array([sum(c in d[st] for st, c in hn) for d in A['_draws']])
        A['named_inside'] = dict(n_named=len(hn), inside=inside, share=inside / max(len(hn), 1), null_mean=float(nulls.mean()) / max(len(hn), 1),
                                 p=float((1 + (nulls >= inside).sum()) / (1 + len(nulls))))
        if name == 'official_n64':
            from interp.e29_auto_circuit_jedi import NAMED_L1, NAMED_S17
            A['e29_hand_inside'] = dict(l1=sum(c in c95['a'] for c in NAMED_L1), S17=sum(c in c95['S'] for c in NAMED_S17))
        log(f'   [{corr}] hand-named units inside circuit95: {inside} / {len(hn)} (random circuits {nulls.mean():.1f}, p {A["named_inside"]["p"]:.4f})')
    for corr in R['acdc']: R['acdc'][corr].pop('_draws')
    save()

    # ---- step 3: roles
    RF = role_feats(xkf); RT = role_feats(xkt)
    Zf = Sf[:, liveS]; Zt = Stt[:, liveS]
    rf = roles_fit(Zf, Zt, RF, RT)
    R['roles'] = {}
    for role in ROLES:
        e = rf[role]; e['unit'] = f'S{liveS[e["unit_idx"]]}' if e['unit_idx'] is not None else None
        e['best_unit_unassigned'] = f'S{liveS[e["best_unit_unassigned"]]}' if e.get('best_unit_unassigned') is not None else None
        e['per_unit_test'] = {f'S{c}': float(v) for c, v in zip(liveS, e['per_unit_test'])}
        R['roles'][role] = e
        log(f'step 3: role {role}: assigned {e["unit"]} {e["param"]} R2 fit {e["r2_fit"]} test {e["r2_test"]} (best unit before assignment {e["best_unit_unassigned"]})')
    # rotation null
    mu, sd = Zf.mean(0), Zf.std(0); sd[sd == 0] = 1; Zfs = (Zf - mu) / sd; Zts = (Zt - mu) / sd
    rot = {role: [] for role in ROLES}; rng_r = np.random.default_rng(3305); Qs = []
    for i in range(NROT):
        Q = rand_rot(len(liveS), rng_r); Qs.append(Q); rr_ = roles_fit(Zfs @ Q, Zts @ Q, RF, RT)
        for role in ROLES: rot[role].append(rr_[role]['r2_test'] if rr_[role]['r2_test'] is not None else -1.0)
    for role in ROLES:
        v = np.array(rot[role]); R['roles'][role]['rot_null'] = dict(n=NROT, r2_test_q=[float(np.quantile(v, q)) for q in (.5, .95)],
                                                                     rate_r2_09=float((v >= 0.9).mean()))
        log(f'   rotation null {role}: median test R2 {np.median(v):.3f}, p95 {np.quantile(v, .95):.3f}, rate R2>=0.9 {(v >= 0.9).mean():.3f}')
    # Hungarian matching to the reference
    if name != 'official_n64':
        ref = Model('official_n64'); Sref = ref.run(x3t, keep=('S',))['S']
        lr = [c for c in range(64) if Sref[:, c].std() > 1e-9]
        def corrmat(A_, B_):
            A_ = (A_ - A_.mean(0)) / np.maximum(A_.std(0), 1e-12); B_ = (B_ - B_.mean(0)) / np.maximum(B_.std(0), 1e-12); return A_.T @ B_ / len(A_)
        C = np.abs(corrmat(Sref[:, lr], Zt)); ri, ci = linear_sum_assignment(-C)
        obs = float(C[ri, ci].mean()); match = {f'S{lr[i]}': dict(unit=f'S{liveS[j]}', abs_corr=float(C[i, j])) for i, j in zip(ri, ci)}
        nullm = []
        for Q in Qs:
            Cq = np.abs(corrmat(Sref[:, lr], Zts @ Q)); a_, b_ = linear_sum_assignment(-Cq); nullm.append(Cq[a_, b_].mean())
        nullm = np.array(nullm)
        R['match'] = dict(n_ref_live=len(lr), mean_abs_corr=obs, per_ref_unit=match, null_mean=float(nullm.mean()),
                          null_q975=float(np.quantile(nullm, .975)), p=float((1 + (nullm >= obs).sum()) / (1 + len(nullm))))
        log(f'   Hungarian match to official_n64: mean |corr| {obs:.3f} vs rotations {nullm.mean():.3f} (p {R["match"]["p"]:.4f}); '
            + ', '.join(f'{k}->{v["unit"]}:{v["abs_corr"]:.2f}' for k, v in match.items() if k in ('S29', 'S1', 'S0', 'S19', 'S30', 'S50')))
        del ref
    save()

    # ---- step 4: causal edits
    rng_e = np.random.default_rng(3306)
    for role in ROLES:
        e = R['roles'][role]
        if e['unit_idx'] is None: e.update(causal={}, found=False, found_equiv=False, found_dir=False); continue
        c = liveS[e['unit_idx']]; prm = e['param']; coef = np.array(e['coef'][1:])
        f0 = role_value(role, prm, xkt) @ coef; ed = make_edits(role, prm, xkt, rng_e); res = {}
        for nm, (I, xe) in ed.items():
            if len(I) < 10: res[nm] = dict(n=len(I)); continue
            xin_e = to16(xt[I], xkt[I], xe) if f16 else xe
            Se = model.run(xin_e, keep=('S',))['S'][:, c]; obs = Se - Stt[I, c]; pred = role_value(role, prm, xe) @ coef - f0[I]
            res[nm] = dict(I=I, obs=obs, pred=pred)
        out = {}
        if 'in' in res and 'obs' in res['in']:
            o, p = res['in']['obs'], res['in']['pred']; s, ci = boot_slope(o, p, rng_e, NBOOT)
            out['in'] = dict(n=len(o), slope=s, slope_ci=ci, mean_pred=float(p.mean()), mean_obs=float(o.mean()),
                             covers1=bool(ci[0] is not None and ci[0] <= 1 <= ci[1]), within_08_125=bool(ci[0] is not None and ci[0] >= 0.8 and ci[1] <= 1.25))
            for nm in res:
                if nm == 'in' or 'obs' not in res[nm]: out[nm] = res[nm] if nm != 'in' else out[nm]; continue
                Ic, ii, jj = np.intersect1d(res['in']['I'], res[nm]['I'], return_indices=True)
                oc = np.abs(res[nm]['obs'][jj]); pin = np.abs(p[ii]); oin = np.abs(o[ii])
                rc, rci = boot_ratio(oc, pin, rng_e, NBOOT); ri_, rici = boot_ratio(oin, pin, rng_e, NBOOT)
                out[nm] = dict(n=len(Ic), ratio_ctrl=rc, ratio_ctrl_ci=rci, ratio_in=ri_, ratio_in_ci=rici,
                               ctrl_pred_mean_abs=float(np.abs(res[nm]['pred']).mean()),
                               p_signflip=signflip_p(oin - oc, rng_e, 2000 if smoke else 10000))
        e['causal'] = out; e['found'] = bool((e['r2_test'] or -1) >= 0.9 and out.get('in', {}).get('covers1', False))
        e['found_equiv'] = bool((e['r2_test'] or -1) >= 0.9 and out.get('in', {}).get('within_08_125', False))
        ctr = [v for k, v in out.items() if k != 'in' and 'p_signflip' in v]
        e['found_dir'] = bool((e['r2_test'] or -1) >= 0.9 and out.get('in', {}).get('slope_ci', [None])[0] is not None and out['in']['slope_ci'][0] > 0
                              and ctr and all(v['p_signflip'] < 0.05 and (v['ratio_in'] or 0) > (v['ratio_ctrl'] or 0) for v in ctr))
        log(f'step 4: role {role} S{c}: in-window slope {out.get("in", {}).get("slope")} {out.get("in", {}).get("slope_ci")} (N {out.get("in", {}).get("n")}); '
            + '; '.join(f'{k}: |obs|/|pred_in| {v.get("ratio_ctrl")} {v.get("ratio_ctrl_ci")} vs in {v.get("ratio_in")} p {v.get("p_signflip")}'
                        for k, v in out.items() if k != 'in') + f'; found {e["found"]} (equiv {e["found_equiv"]})')
        save()

    # ---- step 5: z windows
    R['zwin'] = z_windows(model, xt, xkt, np.random.default_rng(3307), nb=30 if smoke else 200)
    log('step 5: z windows ' + ', '.join(f'{u}: ' + '/'.join(f'{w["centre"]:.0f} [{w["ci"][0] or float("nan"):.0f}, {w["ci"][1] or float("nan"):.0f}]' for w in ws)
                                      for u, ws in R['zwin']['units'].items()) + f'; bands {R["zwin"]["band_hits"]}')
    R['wall_s'] = time.time() - T0; save(); log('done', JSON)


# ================================================================================================ combine
def pbinom_tail(ps, k):
    """P(X >= k), X = sum of independent Bernoulli(ps)"""
    d = np.zeros(len(ps) + 1); d[0] = 1
    for p in ps: d[1:] = d[1:] * (1 - p) + d[:-1] * p; d[0] *= (1 - p)
    return float(d[k:].sum())


def combine():
    rows_ = {}
    for m in MODELS:
        f = OUT / f'e33_{m}.json'
        if f.exists(): rows_[m] = json.loads(f.read_text())
    S = dict(node='E33', models=list(rows_), table={})
    for m, r in rows_.items():
        if 'zwin' not in r: S['table'][m] = dict(incomplete=True); continue
        ac = r.get('acdc', {})
        t = dict(acc=r['step0']['acc'], live=r['n_live_total'], n_live=r['n_live'],
                 share95_mean=ac.get('mean', {}).get('share95'), share95_resample=ac.get('resample', {}).get('share95'),
                 size95={k: ac[k]['size95'] for k in ac}, F95_sub={k: ac[k]['F95_sub'] for k in ac},
                 null_F={k: ac[k]['null']['F_mean'] for k in ac}, acc0={k: ac[k]['acc0'] for k in ac},
                 reached95={k: ac[k]['reached95'] for k in ac}, circ_null_p={k: ac[k]['null']['p'] for k in ac},
                 named_inside={k: ac[k]['named_inside'] for k in ac},
                 n_named_S=len(r['named']['named_S']), n_live_S=len(r['live']['S']), n_pure_l1=len(r['named']['pure_l1']),
                 n_live_l1=len(r['live']['a']), roles={}, match=r.get('match', {}).get('mean_abs_corr'),
                 match_null=r.get('match', {}).get('null_mean'), match_p=r.get('match', {}).get('p'),
                 zwin={u: [w['centre'] for w in ws] for u, ws in r['zwin']['units'].items()}, bands=r['zwin']['band_hits'])
        for role in ROLES:
            e = r['roles'][role]; ci = e.get('causal', {}).get('in', {})
            t['roles'][role] = dict(unit=e['unit'], param=e['param'], r2=e['r2_test'], slope=ci.get('slope'), slope_ci=ci.get('slope_ci'),
                                    found=e['found'], found_equiv=e['found_equiv'], found_dir=e['found_dir'], null_rate=e['rot_null']['rate_r2_09'],
                                    p_rot=(1 + e['rot_null']['rate_r2_09'] * e['rot_null']['n']) / (1 + e['rot_null']['n']) if (e['r2_test'] or -1) >= 0.9 else None)
        S['table'][m] = t
    done = [m for m in TRAINED if m in S['table'] and not S['table'][m].get('incomplete')]
    # Holm over roles x models
    keys = [(m, role) for m in done for role in ROLES]; ps = [S['table'][m]['roles'][role]['p_rot'] for m, role in keys]
    hp = holm(ps)
    for (m, role), h in zip(keys, hp): S['table'][m]['roles'][role]['p_rot_holm'] = h
    S['recurrence'] = {}
    for role in ROLES:
        found = [m for m in done if S['table'][m]['roles'][role]['found']]
        feq = [m for m in done if S['table'][m]['roles'][role]['found_equiv']]
        fdir = [m for m in done if S['table'][m]['roles'][role]['found_dir']]
        r2ok = [m for m in done if (S['table'][m]['roles'][role]['r2'] or -1) >= 0.9]
        # null rate per model floored at the rotation resolution: (1 + #rotations with R^2 >= 0.9) / (1 + 250)
        nr = [(1 + S['table'][m]['roles'][role]['null_rate'] * 250) / 251 for m in done]
        S['recurrence'][role] = dict(n_models=len(done), found=found, n_found=len(found), found_equiv=feq, found_dir=fdir, r2_ok=r2ok,
                                     expected_null=float(sum(nr)), p_found=pbinom_tail(nr, len(found)), p_dir=pbinom_tail(nr, len(fdir)),
                                     p_r2=pbinom_tail(nr, len(r2ok)))
    hp = holm([S['recurrence'][r]['p_found'] for r in ROLES])
    for r, h in zip(ROLES, hp): S['recurrence'][r]['p_found_holm'] = h
    S['bands'] = {f'{lo}-{hi}': [m for m in done if S['table'][m]['bands'][f'{lo}-{hi}']] for lo, hi in BANDS}
    (OUT / 'e33_summary.json').write_text(json.dumps(clean(S), indent=1)); log('summary', json.dumps(clean(S['recurrence'])), S['bands'])
    figure(S, rows_)


def figure(S, rows_):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ms = [m for m in MODELS if m in S['table'] and not S['table'][m].get('incomplete')]
    lab = {'official_n64': 'official n64 (ref)', 'official_n32': 'official n32', 'official_n128': 'official n128', 'perminv_n64': 'perm-inv n64',
           'f16_n64': '16-feature n64', 'hgq150_s0': 'retrained HGQ s0', 'hgq150_s1': 'retrained HGQ s1', 'hgq150_s2': 'retrained HGQ s2',
           'float_s0': 'retrained float s0', 'float_s1': 'retrained float s1', 'untrained_n64': 'untrained redraw'}
    rl = {'hard': 'sum pT of k hardest\n(S0-like)', 'r2': 'sum pT r² + sum r²\n(S29-like)', 'soft': 'soft/empty-slot ramp\nsum (t - pT)+ (S19-like)', 'core': 'core sum (t - r)+\n(S30-like)'}
    fig, ax = plt.subplots(1, 3, figsize=(16, 0.5 * len(ms) + 2.2), gridspec_kw=dict(width_ratios=[2.2, 1.2, 2]))
    M = np.array([[S['table'][m]['roles'][r]['r2'] for r in ROLES] for m in ms], float)
    im = ax[0].imshow(np.clip(M, 0, 1), vmin=0, vmax=1, cmap='viridis', aspect='auto')
    for i, m in enumerate(ms):
        for j, r in enumerate(ROLES):
            e = S['table'][m]['roles'][r]; s = e['slope']
            txt = (f"R² {e['r2']:.2f}" if e['r2'] is not None else 'none') + (f"\nslope {s:.2f}" if s is not None else '')
            ax[0].text(j, i, txt + (' ✓' if e['found'] else (' *' if e['found_dir'] else '')), ha='center', va='center', fontsize=7.5, color='w' if (M[i, j] or 0) < 0.6 else 'k')
    ax[0].set_xticks(range(4)); ax[0].set_xticklabels([rl[r] for r in ROLES], fontsize=8); ax[0].set_yticks(range(len(ms)))
    ax[0].set_yticklabels([lab[m] for m in ms], fontsize=9); ax[0].set_title('best second-round unit per role: held-out R², edit slope\n(✓ R² ≥ 0.9 and slope CI covers 1; * R² ≥ 0.9, edit response > control)', fontsize=9)
    plt.colorbar(im, ax=ax[0], fraction=0.03)
    y = np.arange(len(ms))
    nz = lambda v: np.nan if v is None else v  # noqa: E731
    ax[1].barh(y - 0.2, [nz(S['table'][m]['share95_mean']) for m in ms], 0.4, label='mean corruption')
    ax[1].barh(y + 0.2, [nz(S['table'][m]['share95_resample']) for m in ms], 0.4, label='resample corruption')
    ax[1].axvspan(0.5, 0.65, color='0.85', zorder=0); ax[1].set_yticks(y); ax[1].set_yticklabels([]); ax[1].set_ylim(len(ms) - 0.5, -0.5); ax[1].set_xlim(0, 1)
    ax[1].set_xlabel('circuit size / live units\n(smallest pruned circuit keeping 95 % of above-chance accuracy)', fontsize=8); ax[1].legend(fontsize=7, loc='lower right')
    for q, (lo, hi) in enumerate(BANDS): ax[2].axvspan(lo, hi, color=['0.93', '0.82', '0.93'][q], zorder=0)
    for i, m in enumerate(ms):
        for u, ws in rows_[m]['zwin']['units'].items():
            for w in ws:
                ci = w['ci']; ax[2].plot(w['centre'], i, 'o', color='C3', ms=4)
                if ci[0] is not None: ax[2].plot(ci, [i, i], '-', color='C3', lw=1)
    ax[2].set_yticks(y); ax[2].set_yticklabels([]); ax[2].set_ylim(len(ms) - 0.5, -0.5); ax[2].set_xlim(20, 250)
    ax[2].set_xlabel('jet mass where a z unit\'s response to a ±5 % mass change\nturns from + to − (GeV); grey = reference windows', fontsize=8)
    fig.suptitle('E33: recurrence of the JEDI-linear program elements across checkpoints (test N = 10,000 jets each)', fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / 'e33_recurrence.png', dpi=130); log('figure', OUT / 'e33_recurrence.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--model', choices=list(MODELS)); ap.add_argument('--combine', action='store_true')
    ap.add_argument('--smoke', action='store_true'); ap.add_argument('--plot', action='store_true')
    a = ap.parse_args()
    if a.combine: combine()
    elif a.plot:
        S = json.loads((OUT / 'e33_summary.json').read_text()); figure(S, {m: json.loads((OUT / f'e33_{m}.json').read_text()) for m in S['models']})
    else: run_model(a.model, a.smoke)
