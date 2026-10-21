"""E28w (claim A15; stage 4): which program element of JEDI-linear (3-feature n64, official, fixed point) carries the call changes
caused by an exact collinear split?

Edit (as E28 / E28v): the particle at pT rank k in {0, 5, 13, 21} (21 = the first rank > 20) is replaced by two particles at the same
  (deta, dphi) with pT z pT_k and (1 - z) pT_k, z in {0.5, 0.7, 0.9}; the jet is re-sorted by pT. M0 = sum pT, M1, M2 and the
  massless four-momentum are unchanged; N rises by one and the softer particles move down one slot. Jets with N = 64 keep the 64
  hardest of the 65 particles (the softest real particle leaves the input: "overflow"). Applicable when N >= k + 2.
Samples: main = sample(per_class=1500, seed=2801) (7,500 jets, all N); strata from sample(per_class=3000, seed=2802):
  gbal = jets with N <= 60, the same number per class (min over classes, at most 1000), random choice;
  hiN  = jets with N > 60 (at most 4000, random), where the overflow occurs at N = 64.
Hypotheses (program elements) and their tests; "share removed" = 1 - #calls changed (edit + intervention) / #calls changed (edit),
  on the same jets, i.e. the fraction of the split's call changes that disappear when that element is kept at its unsplit state:
  H_soft  a32 (l1 pT ramp, on for pT < 11 GeV): the two halves' a32 are set to the parent's a32 (unedited jet, parent's slot);
          strata: 'cross' = parent a32 = 0 and a half's a32 > 0 (a half switches the ramp on), 'nocross' = the rest.
          null: the same clamp on each other live l1 unit (the other pT ramps a15 a21 a56 and the angular cuts); also all l1 units.
  H_rank  rank-restore path patch (exact, additive): both pooled sums are sums of per-slot quantized terms, P = s0 sum_i q_i(a_i)
          and S = s1 sum_i q_i(u_i). The patched jet uses, for every particle other than the two halves, its term computed in its
          ORIGINAL (unsplit) slot, and for the halves their terms in their split slots; the parent's term and one empty-slot term
          (slot N; for N = 64 the softest particle, which the split pushes out) are removed, so the count of particles and of empty
          slots equals the split jet's. 'rank_PS': both P (hence g) and S restored; 'rank_S': only the S sum restored (g = the split
          jet's g). Identity check: on jets where no other particle changes slot, the patched S equals the split S.
          Stratum 'noshift': jets where the re-sorted split moves no other particle (half 1 stays at k, half 2 lands in slot N).
          Input-level variant 'noresort' (half 1 in slot k, half 2 in slot N without re-sorting; N <= 63) is reported but puts a
          hard particle into a soft slot, which ordinary jets never have.
  H_slot  for N = 64 jets: drop_softest (remove the softest particle of the unsplit jet, the particle the split pushes out) and
          split_nooverflow (split of the jet without its softest particle, compared with that jet): share removed =
          1 - #changed(split_nooverflow) / #changed(split).
  H_S     path patches: each live S unit, named S sets (S19, S0, S30, S29, S0+S19, S19+S30, S19+S29+S30, the 17 rank-window S),
          each live h4 unit, all of g, held at the unsplit jet's value. Null: every (enumerated) live-S set of the same size
          (19 / 171 / 969 / 171 sets for sizes 1 / 2 / 3 / 17), every live h4 unit for h4 singles; rank p = (1 + #null >= named) /
          (1 + #null); McNemar edit vs edit + patch, Holm over the units of a scan.
Controls: rot (the split particle rotated about the jet axis by a random azimuth: N, ranks, M0, M1, M2 fixed); matched data pair (merge
  the closest pair of real particles at dR < 0.01 into one particle, pT summed, pT-weighted angles); untrained JEDI-linear
  (Jedi(untrained=True, seed 0, 1); calls from logits minus their unedited mean).
Statistics: 1000 multinomial bootstrap resamples over jets (shares and rates); exact McNemar; binomial sign test for q<->g flows.
Run: ./job.sh e28w-run 1 2 'python -m interp.e28w_split_route_jedi'     (--smoke N: N jets per class, fewer null sets)
Writes $AI_RESULTS/jedi/e28w_split_route[_smoke].json and e28w_split_route_perjet.npz.
"""
import argparse
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); OUT = RES / 'jedi'
T0_ = time.time()
CL = ['g', 'q', 'W', 'Z', 't']
LIVE_S = [29, 1, 0, 50, 19, 30, 43, 60, 13, 17, 56, 40, 31, 51, 63, 52, 62, 58, 20]   # E5 / E20
WINDOWLESS = [29, 30]
WINDOW = [u for u in LIVE_S if u not in WINDOWLESS]
NAMED_S = {'S19': [19], 'S0': [0], 'S30': [30], 'S29': [29], 'S0+S19': [0, 19], 'S19+S30': [19, 30],
           'S19+S29+S30': [19, 29, 30], 'window17': WINDOW}
PT_RAMPS = [15, 21, 32, 56]
RANKS = (0, 5, 13, 21); ZS = (0.5, 0.7, 0.9)
B_FWD = 1500


def log(*a): print(f'[{time.time() - T0_:7.0f}s]', *a, flush=True)


# ------------------------------------------------------------------ statistics
def binom_p(k, n):
    if n == 0: return 1.0
    k = min(k, n - k)
    lp = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(k + 1)]
    m = max(lp); return float(min(1.0, 2 * math.exp(m) * sum(math.exp(v - m) for v in lp)))


def mcnemar(a, b):
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    return dict(n_first_only=n10, n_second_only=n01, p=binom_p(n10, n10 + n01))


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); out = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * ps[i])); out[i] = run
    return out.tolist()


def qci(v): return [float(np.quantile(v, .025)), float(np.quantile(v, .975))]


class Boot:
    def __init__(self, J, B, seed):
        rng = np.random.default_rng(seed); self.W = np.zeros((B, J), np.float32)
        for b in range(B): self.W[b] = np.bincount(rng.integers(0, J, J), minlength=J)

    def mean(self, v, sel):
        W = self.W[:, sel]; return (W @ v[sel].astype(np.float32)) / np.maximum(W.sum(1), 1)

    def removed(self, chE, chH, sel):
        W = self.W[:, sel]; return 1 - (W @ chH[sel].astype(np.float32)) / np.maximum(W @ chE[sel].astype(np.float32), 1)

    def ratio(self, chA, chE, sel):
        W = self.W[:, sel]; return (W @ chA[sel].astype(np.float32)) / np.maximum(W @ chE[sel].astype(np.float32), 1)


# ------------------------------------------------------------------ edits on raw (J, 64, 3) pT-sorted arrays, padding = 0
def nreal(X): return (X[:, :, 0] > 0).sum(1)


def resort(Y):
    o = np.argsort(-Y[:, :, 0], axis=1, kind='stable'); return np.take_along_axis(Y, o[:, :, None], 1), o


def split_ext(X, slot, z):
    """re-sorted split with overflow; returns Y (J, 64, 3), slots of half 1 / half 2 in Y (-1 = n/a or pushed out)"""
    J = len(X); N = nreal(X); rows = np.flatnonzero(slot >= 0); k = slot[rows]
    Y = np.concatenate([X, np.zeros((J, 1, 3), np.float32)], 1)
    p = X[rows, k, 0].astype(np.float64)
    Y[rows, k, 0] = (z * p).astype(np.float32)
    Y[rows, N[rows], 0] = (p - Y[rows, k, 0].astype(np.float64)).astype(np.float32)
    Y[rows, N[rows], 1] = X[rows, k, 1]; Y[rows, N[rows], 2] = X[rows, k, 2]
    Ys, o = resort(Y)
    inv = np.argsort(o, axis=1)
    d1 = np.full(J, -1); d2 = np.full(J, -1)
    d1[rows] = inv[rows, k]; d2[rows] = inv[rows, N[rows]]
    d1[d1 >= 64] = -1; d2[d2 >= 64] = -1
    return Ys[:, :64].copy(), d1, d2


def noresort(X, slot, z):
    Y = X.copy(); N = nreal(X); rows = np.flatnonzero((slot >= 0) & (N <= 63)); k = slot[rows]
    p = X[rows, k, 0].astype(np.float64)
    Y[rows, k, 0] = (z * p).astype(np.float32)
    Y[rows, N[rows], 0] = (p - Y[rows, k, 0].astype(np.float64)).astype(np.float32)
    Y[rows, N[rows], 1] = X[rows, k, 1]; Y[rows, N[rows], 2] = X[rows, k, 2]
    ok = np.zeros(len(X), bool); ok[rows] = True
    return Y, ok


def shiftonly(Ys, X, slot, d1, d2):
    """split slots, unsplit content: half 1 back to the parent's pT, half 2 -> padding (0, 0, 0); N <= 63 only"""
    Y = Ys.copy(); N = nreal(X); rows = np.flatnonzero((slot >= 0) & (N <= 63) & (d1 >= 0) & (d2 >= 0))
    Y[rows, d1[rows], 0] = X[rows, slot[rows], 0]; Y[rows, d2[rows], :] = 0
    ok = np.zeros(len(X), bool); ok[rows] = True
    return Y, ok


def rotate(X, slot, rng):
    Y = X.copy(); rows = np.flatnonzero(slot >= 0); k = slot[rows]
    de, dp = X[rows, k, 1].astype(np.float64), X[rows, k, 2].astype(np.float64)
    r = np.hypot(de, dp); th = np.arctan2(de, dp) + rng.uniform(0, 2 * np.pi, len(rows))
    Y[rows, k, 2] = (r * np.cos(th)).astype(np.float32); Y[rows, k, 1] = (r * np.sin(th)).astype(np.float32)
    return Y


def drop_softest(X, rows=None):
    Y = X.copy(); N = nreal(X); rows = np.arange(len(X)) if rows is None else rows
    rows = rows[N[rows] > 0]; Y[rows, N[rows] - 1] = 0; return Y


def merge_pair(X, dr_max=0.01):
    """merge the closest pair of real particles (dR < dr_max) into one: pT summed, pT-weighted (deta, dphi); re-sorted"""
    Y = X.copy(); N = nreal(X); ok = np.zeros(len(X), bool); dmin = np.full(len(X), np.inf)
    for j in range(len(X)):
        n = N[j]
        if n < 2: continue
        e, f = X[j, :n, 1].astype(np.float64), X[j, :n, 2].astype(np.float64)
        D = np.hypot(e[:, None] - e[None], f[:, None] - f[None]); np.fill_diagonal(D, np.inf)
        a, b = np.unravel_index(D.argmin(), D.shape); dmin[j] = D[a, b]
        if D[a, b] < dr_max:
            a, b = min(a, b), max(a, b); pa, pb = float(X[j, a, 0]), float(X[j, b, 0])
            Y[j, a, 0] = pa + pb
            Y[j, a, 1] = (pa * e[a] + pb * e[b]) / (pa + pb); Y[j, a, 2] = (pa * f[a] + pb * f[b]) / (pa + pb)
            Y[j, b] = 0; ok[j] = True
    Y, _ = resort(Y)
    return Y, ok, dmin


def moments(X):
    pt = X[:, :, 0].astype(np.float64); r = np.hypot(X[:, :, 1], X[:, :, 2]).astype(np.float64)
    px, py = (pt * np.cos(X[:, :, 2])).sum(1), (pt * np.sin(X[:, :, 2])).sum(1)
    pz, E = (pt * np.sinh(X[:, :, 1])).sum(1), (pt * np.cosh(X[:, :, 1])).sum(1)
    return dict(M0=pt.sum(1), M1=(pt * r).sum(1), M2=(pt * r * r).sum(1),
                m=np.sqrt(np.maximum(E ** 2 - px ** 2 - py ** 2 - pz ** 2, 0)))


# ------------------------------------------------------------------ network helpers
class Net:
    def __init__(self, untrained=False, seed=0):
        import jax, jax.numpy as jnp
        from interp.jedi_core import Jedi
        self.jax, self.jnp = jax, jnp
        self.n = Jedi(untrained=untrained, seed=seed); L = self.n.L; self.L = L
        self.kd_S = bool(getattr(L['qs1'], 'keepdims', False))
        KB = {k: (jnp.asarray(self.n.KB[k][0], jnp.float32), jnp.asarray(self.n.KB[k][1], jnp.float32)) for k in ('d4', 'd5', 'd6', 'd7')}

        def dense(k, t, act=True):
            z = jnp.einsum(L[k].equation, L[k].iq(t, training=False), KB[k][0]) + KB[k][1]
            return L[k].activation(z) if act and L[k].activation is not None else z

        def tail(S, mS, vS, m4, v4):
            S = S * (1 - mS) + mS * vS
            h4 = dense('d4', S); h4 = h4 * (1 - m4) + m4 * v4
            h5 = dense('d5', h4); z = dense('d6', h5, act=False)
            return dense('d7', jax.nn.relu(z)).reshape(S.shape[0], -1)
        self._tail = jax.jit(tail)
        KA = {k: (jnp.asarray(self.n.KB[k][0], jnp.float32), jnp.asarray(self.n.KB[k][1], jnp.float32)) for k in ('l1', 'l2', 'dd', 'l3')}

        def dense1(k, t):
            z = jnp.einsum(L[k].equation, L[k].iq(t, training=False), KA[k][0]) + KA[k][1]
            return L[k].activation(z) if L[k].activation is not None else z

        def contrib(xin, gv, use):
            a = dense1('l1', xin); Ca = L['qs'].iq(a, training=False)
            P = jnp.sum(Ca, axis=L['qs'].axes, keepdims=L['qs'].keepdims) * L['qs'].scale
            g = dense1('dd', P); g = g * (1 - use) + use * gv
            s = dense1('l2', a); qs_, qd_ = L['add'].iq([s, g], training=False)
            u = dense1('l3', qs_ + qd_); Cu = L['qs1'].iq(u, training=False)
            return Ca, Cu, g
        self._contrib = jax.jit(contrib)
        self._dd = jax.jit(lambda P: dense1('dd', P))
        self.kd_P = bool(getattr(L['qs'], 'keepdims', False)); self.sc = (float(np.asarray(L['qs'].scale)), float(np.asarray(L['qs1'].scale)))

    def rank_patch(self, X0, Ys, slot, d1, d2, mode):
        """S of the rank-restore path patch (see docstring); mode 'PS' or 'S'. Returns S (J, 64) and, as a check, the split's S
        recomputed from its own per-slot terms."""
        jnp = self.jnp; J = len(X0); N = nreal(X0); out, chk = [], []
        e = np.where(N < 64, N, np.where(d2 >= 0, 63, -1))
        for i in range(0, J, B_FWD):
            n = min(B_FWD, J - i); sl = slice(i, i + n)
            def pad(a):
                a = np.asarray(a[sl], np.float32)
                return np.concatenate([a, np.zeros((B_FWD - n,) + a.shape[1:], np.float32)]) if n < B_FWD else a
            x0 = jnp.asarray(self.n.inputs(pad(X0), 64)); xs = jnp.asarray(self.n.inputs(pad(Ys), 64))
            gz = jnp.zeros((B_FWD, 1, 64) if self.kd_P else (B_FWD, 64), jnp.float32)
            Ca0, Cu0, g0 = (np.asarray(t) for t in self._contrib(x0, gz, 0.0))
            Cas, Cus, gs = (np.asarray(t) for t in self._contrib(xs, gz, 0.0))
            W = np.ones((B_FWD, 64), np.float32); rr = np.arange(n); k = slot[sl]; ap = k >= 0
            W[rr[ap], k[ap]] = 0; ee = e[sl]; ae = ap & (ee >= 0); W[rr[ae], ee[ae]] = 0
            W[n:] = 1
            H = np.zeros((B_FWD, 64), np.float32)
            a1 = ap & (d1[sl] >= 0); a2 = ap & (d2[sl] >= 0)
            def halves(C):
                h = np.zeros((B_FWD, C.shape[2]), np.float32)
                h[rr[a1]] += C[rr[a1], d1[sl][a1]]; h[rr[a2]] += C[rr[a2], d2[sl][a2]]; return h
            notap = np.zeros(B_FWD, bool); notap[:n] = ~ap; notap[n:] = True
            if mode == 'PS':
                Pp = (W[:, :, None] * Ca0).sum(1) + halves(Cas)
                Pp[notap] = Cas[notap].sum(1)
                gr = np.asarray(self._dd(jnp.asarray((Pp * self.sc[0]).reshape((B_FWD, 1, 64) if self.kd_P else (B_FWD, 64)))))
            else:
                gr = gs
            _, Cu0r, _ = (np.asarray(t) for t in self._contrib(x0, jnp.asarray(gr), 1.0))
            _, Cusr, _ = (np.asarray(t) for t in self._contrib(xs, jnp.asarray(gr), 1.0))
            Sp = (W[:, :, None] * Cu0r).sum(1) + halves(Cusr)
            Sp[notap] = Cusr[notap].sum(1)
            out.append(Sp[:n] * self.sc[1]); chk.append(Cus.sum(1)[:n] * self.sc[1])
        return np.concatenate(out).astype(np.float64), np.concatenate(chk).astype(np.float64)

    def shp(self, B, W=64): return (B, 1, W) if self.kd_S else (B, W)

    def fwd(self, X, ov_fn=None, keep=('logits', 'S', 'h4', 'g'), a_units=None):
        out = {k: [] for k in keep}
        for i in range(0, len(X), B_FWD):
            xb = X[i:i + B_FWD]; n = len(xb)
            if n < B_FWD: xb = np.concatenate([xb, np.zeros((B_FWD - n, 64, 3), np.float32)])
            O = ov_fn(i, n) if ov_fn else {}
            r = self.n._f(self.n.inputs(xb, 64), O)
            for k in keep:
                v = np.asarray(r[k])[:n]
                out[k].append(v[:, :, a_units].astype(np.float32) if k == 'a' else v.reshape(n, -1).astype(np.float64))
        return {k: np.concatenate(v) for k, v in out.items()}

    def tail(self, S, mS=None, vS=None, m4=None, v4=None):
        jnp = self.jnp; J = len(S); out = []
        mS = np.zeros(64, np.float32) if mS is None else np.asarray(mS, np.float32)
        m4 = np.zeros(64, np.float32) if m4 is None else np.asarray(m4, np.float32)
        vS = np.zeros_like(S) if vS is None else vS; v4 = np.zeros_like(S) if v4 is None else v4
        for i in range(0, J, B_FWD):
            n = min(B_FWD, J - i)
            def pad(a):
                a = np.asarray(a[i:i + n], np.float32)
                return np.concatenate([a, np.zeros((B_FWD - n,) + a.shape[1:], np.float32)]) if n < B_FWD else a
            r = self._tail(jnp.asarray(pad(S).reshape(self.shp(B_FWD))), jnp.asarray(mS), jnp.asarray(pad(vS).reshape(self.shp(B_FWD))),
                           jnp.asarray(m4), jnp.asarray(pad(v4).reshape(self.shp(B_FWD))))
            out.append(np.asarray(r)[:n])
        return np.concatenate(out).astype(np.float64)


def flows(c0, ce, sel):
    F = np.zeros((5, 5), int); np.add.at(F, (c0[sel], ce[sel]), 1)
    return dict(flow=F.tolist(), q_to_g=int(F[1, 0]), g_to_q=int(F[0, 1]), p_qg=binom_p(int(F[1, 0]), int(F[1, 0] + F[0, 1])),
                W_to_Z=int(F[2, 3]), Z_to_W=int(F[3, 2]))


# ------------------------------------------------------------------ one sample
def analyse(net, X0, ys, tag, rng, B, full, live, smoke):
    """full = run every scan for every z (main sample); otherwise scans at z 0.5 only"""
    J = len(X0); N0 = nreal(X0); bt = Boot(J, B, int(rng.integers(1 << 30)))
    LIVE_L1, LIVE_H4 = live['l1'], live['h4']; il1 = {u: i for i, u in enumerate(LIVE_L1)}
    R0 = net.fwd(X0, keep=('logits', 'S', 'h4', 'g', 'a'), a_units=LIVE_L1); c0 = R0['logits'].argmax(1); corr0 = c0 == ys
    lt = net.tail(R0['S']); tail_ok = float((lt.argmax(1) == c0).mean()); tail_dev = float(np.abs(lt - R0['logits']).max())
    res = dict(n=J, acc=float(corr0.mean()), class_counts=np.bincount(ys, minlength=5).tolist(),
               N_counts=dict(le60=int((N0 <= 60).sum()), n61_63=int(((N0 > 60) & (N0 < 64)).sum()), n64=int((N0 == 64).sum())),
               tail_check=dict(calls_equal=tail_ok, max_abs_logit_diff=tail_dev))
    log(f'[{tag}] {J} jets, acc {res["acc"]:.4f}, classes {res["class_counts"]}, N {res["N_counts"]}; tail check {tail_ok:.5f} {tail_dev:.2e}')
    Mo0 = moments(X0); perjet = dict(y=ys, c0=c0, N=N0); edits = {}

    def changed(ce, ok): return (ce != c0) & ok

    def rate(ch, sel): return dict(changed=float(ch[sel].mean()) if len(sel) else None, ci=qci(bt.mean(ch, sel)) if len(sel) else None)

    def share(chE, chH, sel):
        nE, nH = int(chE[sel].sum()), int(chH[sel].sum())
        return dict(removed=float(1 - nH / nE) if nE else None, removed_ci=qci(bt.removed(chE, chH, sel)) if nE else None,
                    n_changed_edit=nE, n_changed_patched=nH, mcnemar=mcnemar(chE[sel], chH[sel]))

    # ---- enumerated null sets of live S
    comb = {k: [list(c) for c in itertools.combinations(LIVE_S, k)] for k in (1, 2, 3)}
    comb[17] = [list(c) for c in itertools.combinations(LIVE_S, 17)]
    if smoke: comb = {k: v[:25] for k, v in comb.items()}

    for k in RANKS:
        slot = np.where(N0 >= k + 2, k, -1); ok = slot >= 0; sel = np.flatnonzero(ok)
        Yr = rotate(X0, slot, rng); cr = net.fwd(Yr, keep=('logits',))['logits'].argmax(1); chR = changed(cr, ok)
        edits[f'rot_r{k}'] = dict(rate(chR, sel), n=len(sel), **flows(c0, cr, sel)); perjet[f'ch_rot_r{k}'] = chR
        log(f'[{tag}] rot_r{k}: n {len(sel)} changed {chR[sel].mean():.4f}')
        for z in ZS:
            name = f'split_r{k}_z{z}'; scan = full or z == 0.5
            Ys, d1, d2 = split_ext(X0, slot, z)
            Re = net.fwd(Ys, keep=('logits', 'S', 'h4', 'g', 'a'), a_units=LIVE_L1); ce = Re['logits'].argmax(1); chE = changed(ce, ok)
            perjet[f'ch_{name}'] = chE
            Mo1 = moments(Ys); nov = sel[N0[sel] < 64]
            r = dict(rank=k, z=z, n=len(sel), **rate(chE, sel), **flows(c0, ce, sel),
                     acc_before=float(corr0[sel].mean()), acc_after=float((ce == ys)[sel].mean()),
                     mcnemar_correct=mcnemar(corr0[sel], (ce == ys)[sel]),
                     moment_max_rel_change_no_overflow={m: float(np.max(np.abs(Mo1[m][nov] - Mo0[m][nov]) / np.maximum(Mo0[m][nov], 1e-9)))
                                                        if len(nov) else None for m in Mo0},
                     by_N={nm: dict(n=int(len(s_)), changed=float(chE[s_].mean()) if len(s_) else None)
                           for nm, s_ in (('le60', sel[N0[sel] <= 60]), ('n61_63', sel[(N0[sel] > 60) & (N0[sel] < 64)]), ('n64', sel[N0[sel] == 64]))},
                     vs_rot=dict(diff=float((chE[sel].astype(float) - chR[sel]).mean()),
                                 diff_ci=qci(bt.mean(chE.astype(np.float32) - chR.astype(np.float32), sel)), mcnemar=mcnemar(chE[sel], chR[sel])),
                     parent_pt_median=float(np.median(X0[sel, k, 0])) if len(sel) else None)
            # -- H_soft: l1 clamps (half a_u := parent's a_u)
            a0p = R0['a'][np.arange(J), np.maximum(slot, 0)]          # (J, n_live_l1) parent's a at its unsplit slot
            ae = Re['a']
            ah1 = np.where((d1 >= 0)[:, None], ae[np.arange(J), np.maximum(d1, 0)], 0)
            ah2 = np.where((d2 >= 0)[:, None], ae[np.arange(J), np.maximum(d2, 0)], 0)
            i32 = il1[32]
            cross = ok & (a0p[:, i32] == 0) & ((ah1[:, i32] > 0) | (ah2[:, i32] > 0))
            r['a32'] = dict(parent_on=float((a0p[sel, i32] > 0).mean()) if len(sel) else None, cross_frac=float(cross[sel].mean()) if len(sel) else None)

            def clamp_fn(units):
                m = np.zeros(64, np.float32); m[units] = 1
                def f(i, n):
                    V = np.zeros((B_FWD, 64, 64), np.float32)
                    for u in units:
                        iu = il1[u]; V[:n, :, u] = ae[i:i + n, :, iu]
                        rr = np.arange(n)
                        a1 = d1[i:i + n] >= 0; V[rr[a1], d1[i:i + n][a1], u] = a0p[i:i + n][a1, iu]
                        a2 = d2[i:i + n] >= 0; V[rr[a2], d2[i:i + n][a2], u] = a0p[i:i + n][a2, iu]
                    return {'a': (m, V)}
                return f
            clamps = {'a32': [32]}
            if scan:
                for u in LIVE_L1:
                    if u != 32: clamps[f'a{u}'] = [u]
            H = {}
            for cn, units in clamps.items():
                ch_ = changed(net.fwd(Ys, clamp_fn(units), keep=('logits',))['logits'].argmax(1), ok)
                H[cn] = share(chE, ch_, sel)
                if cn == 'a32':
                    for st, msk in (('cross', cross), ('nocross', ok & ~cross)):
                        s_ = np.flatnonzero(msk); H[f'a32_{st}'] = dict(share(chE, ch_, s_), n=len(s_), changed_edit=float(chE[s_].mean()) if len(s_) else None)
                    perjet[f'ch_{name}_a32'] = ch_
            if scan:
                nul = [H[c]['removed'] for c in clamps if c != 'a32' and H[c]['removed'] is not None]
                H['null_other_l1'] = dict(n=len(nul), q95=float(np.quantile(nul, .95)) if nul else None, max=float(max(nul)) if nul else None,
                                          rank_p_a32=float((1 + sum(v >= (H['a32']['removed'] or 0) for v in nul)) / (1 + len(nul))),
                                          other_pt_ramps={f'a{u}': H[f'a{u}']['removed'] for u in PT_RAMPS if u != 32 and f'a{u}' in H})
            r['H_soft'] = H
            # -- H_rank
            Yn, okn = noresort(X0, slot, z); sn = np.flatnonzero(okn)
            cn_ = net.fwd(Yn, keep=('logits',))['logits'].argmax(1); chN = changed(cn_, okn)
            perjet[f'ch_{name}_noresort'] = chN
            noshift = ok & (d1 == slot) & (d2 == N0) & (N0 < 64); shift = ok & ~noshift
            HR = dict(noresort_input=dict(n=len(sn), split_changed=float(chE[sn].mean()) if len(sn) else None, **rate(chN, sn),
                                          **flows(c0, cn_, sn)),
                      strata={nm_: dict(n=int(m_.sum()), split=rate(chE, np.flatnonzero(m_)), rot=rate(chR, np.flatnonzero(m_)),
                                        split_vs_rot=mcnemar(chE[m_], chR[m_])) for nm_, m_ in (('noshift', noshift), ('shift', shift))})
            for mode in ('PS', 'S'):
                Sp, Schk = net.rank_patch(X0, Ys, slot, d1, d2, mode)
                cp = net.tail(Sp).argmax(1); chP = changed(cp, ok); perjet[f'ch_{name}_rank{mode}'] = chP
                HR[f'rank_{mode}'] = dict(share(chE, chP, sel), changed=float(chP[sel].mean()), **flows(c0, cp, sel),
                                          check_split_S_from_terms=float(np.abs(Schk - Re['S'])[sel].max()),
                                          check_noshift_identity=float(np.abs(Sp - Re['S'])[noshift].max()) if noshift.any() else None,
                                          by_shift={nm_: share(chE, chP, np.flatnonzero(m_)) for nm_, m_ in (('noshift', noshift), ('shift', shift))},
                                          vs_rot=mcnemar(chP[sel], chR[sel]))
            r['H_rank'] = HR
            # -- H_S: g hold, S / h4 path patches
            if scan:
                chG = changed(net.fwd(Ys, lambda i, n: {'g': (np.ones(64, np.float32), np.concatenate(
                    [R0['g'][i:i + n], np.zeros((B_FWD - n, 64))]).astype(np.float32).reshape(B_FWD, 1, 64))}, keep=('logits',))['logits'].argmax(1), ok)
                HS = dict(g_all=share(chE, chG, sel))
                tail_e = net.tail(Re['S']).argmax(1); HS['tail_reproduces_edit'] = float((tail_e == ce)[sel].mean())

                def sh_S(units):
                    m = np.zeros(64, np.float32); m[units] = 1
                    ch_ = changed(net.tail(Re['S'], m, R0['S']).argmax(1), ok); n_ = int(ch_[sel].sum()); nE = int(chE[sel].sum())
                    return (1 - n_ / nE) if nE else np.nan, ch_
                nulls = {}
                for kk, sets in comb.items():
                    nulls[kk] = np.array([sh_S(s)[0] for s in sets])
                singles = {}
                for u in LIVE_S:
                    v, ch_ = sh_S([u]); singles[f'S{u}'] = dict(share(chE, ch_, sel))
                ps = holm([singles[k_]['mcnemar']['p'] for k_ in singles])
                for k_, p in zip(singles, ps): singles[k_]['p_holm'] = p
                named = {}
                for nm, units in NAMED_S.items():
                    v, ch_ = sh_S(units); nl = nulls[len(units)]; nl = nl[np.isfinite(nl)]
                    named[nm] = dict(share(chE, ch_, sel), size=len(units), null_n=len(nl), null_mean=float(nl.mean()),
                                     null_q95=float(np.quantile(nl, .95)), rank_p=float((1 + (nl >= v).sum()) / (1 + len(nl))))
                ps = holm([named[k_]['rank_p'] for k_ in named])
                for k_, p in zip(named, ps): named[k_]['rank_p_holm'] = p
                h4s = {}
                for u in LIVE_H4:
                    m = np.zeros(64, np.float32); m[u] = 1
                    ch_ = changed(net.tail(Re['S'], None, None, m, R0['h4']).argmax(1), ok); h4s[f'h4_{u}'] = share(chE, ch_, sel)
                ps = holm([h4s[k_]['mcnemar']['p'] for k_ in h4s])
                for k_, p in zip(h4s, ps): h4s[k_]['p_holm'] = p
                hv = np.array([h4s[k_]['removed'] for k_ in h4s if h4s[k_]['removed'] is not None])
                HS.update(S_singles=singles, S_named=named, h4_singles=h4s,
                          h4_null=dict(q95=float(np.quantile(hv, .95)) if len(hv) else None, n=len(hv)),
                          S_null_summary={str(kk): dict(n=int(np.isfinite(v_).sum()), mean=float(np.nanmean(v_)), q95=float(np.nanquantile(v_, .95)),
                                                        max=float(np.nanmax(v_))) for kk, v_ in nulls.items()})
                r['H_S'] = HS
                topS = sorted(singles, key=lambda q: -(singles[q]['removed'] or -9))[:4]
                topH = sorted(h4s, key=lambda q: -(h4s[q]['removed'] or -9))[:4]
                log(f'   H_S g {HS["g_all"]["removed"]}; named ' + ', '.join(f'{q} {named[q]["removed"]:.3f} (q95 {named[q]["null_q95"]:.3f}, p {named[q]["rank_p"]:.3f})' for q in named)
                    + '; top S ' + ', '.join(f'{q} {singles[q]["removed"]:.3f}' for q in topS) + '; top h4 ' + ', '.join(f'{q} {h4s[q]["removed"]:.3f}' for q in topH))
            edits[name] = r
            hs = r['H_soft']; hr = r['H_rank']
            log(f'[{tag}] {name}: n {len(sel)} changed {r["changed"]:.4f} {r["ci"]} rot {chR[sel].mean():.4f} q->g {r["q_to_g"]} g->q {r["g_to_q"]}; '
                f'a32 removed {hs["a32"]["removed"]} (cross n {hs["a32_cross"]["n"]} {hs["a32_cross"]["removed"]}, nocross {hs["a32_nocross"]["removed"]}); '
                f'noresort(input) changed {hr["noresort_input"]["changed"]}; rank_PS removed {hr["rank_PS"]["removed"]} chk {hr["rank_PS"]["check_split_S_from_terms"]:.1e} '
                f'id {hr["rank_PS"]["check_noshift_identity"]}; rank_S removed {hr["rank_S"]["removed"]}; noshift n {hr["strata"]["noshift"]["n"]} '
                f'split {hr["strata"]["noshift"]["split"]["changed"]} rot {hr["strata"]["noshift"]["rot"]["changed"]}; null l1 q95 {hs.get("null_other_l1", {}).get("q95")}')
        # -- H_slot (N = 64 jets)
        s64 = np.flatnonzero(ok & (N0 == 64))
        if len(s64):
            Xd = drop_softest(X0); cd = net.fwd(Xd, keep=('logits',))['logits'].argmax(1); chD = changed(cd, ok & (N0 == 64))
            Ys5, _, _ = split_ext(X0, slot, 0.5); chE5 = perjet[f'ch_split_r{k}_z0.5']
            Ysd, _, _ = split_ext(Xd, np.where(N0 == 64, slot, -1), 0.5)
            cdd = net.fwd(Ysd, keep=('logits',))['logits'].argmax(1); chNO = (cdd != cd) & ok & (N0 == 64)
            edits[f'H_slot_r{k}'] = dict(n=len(s64), split_changed=float(chE5[s64].mean()), drop_softest=rate(chD, s64),
                                         drop_and_split_both=int((chD & chE5)[s64].sum()),
                                         share_of_split_changes_also_from_drop=float((chD & chE5)[s64].sum() / max(chE5[s64].sum(), 1)),
                                         split_nooverflow=dict(rate(chNO, s64), **share(chE5, chNO, s64)),
                                         flows_drop=flows(c0, cd, s64))
            log(f'[{tag}] H_slot r{k}: n64 {len(s64)} split {chE5[s64].mean():.4f} drop {chD[s64].mean():.4f} split_nooverflow {chNO[s64].mean():.4f}')
    res['edits'] = edits
    return res, perjet


# ------------------------------------------------------------------ main
def main(a):
    from interp.jedi_core import sample
    sm = a.smoke; B = 200 if sm else 1000; tag = 'e28w_split_route' + ('_smoke' if sm else '')
    OUT.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(2801)
    net = Net(); log('network loaded', net.n.path, 'qs1 keepdims', net.kd_S)
    x, y, _ = sample(per_class=sm or 1500, seed=2801); x = np.asarray(x, np.float32)
    srt = bool(np.all(np.diff(x[:, :, 0], axis=1) <= 0)); log(f'main sample {len(x)} jets, pT-sorted {srt}')
    # ---- l1 cut table and live units
    K1, b1 = net.n.KB['l1']; alpha = K1 / net.n.scale[:, None]; beta = b1 - (K1 * (net.n.shift / net.n.scale)[:, None]).sum(0)
    R = net.fwd(x[:3000], keep=('a', 'g', 'h4', 'S', 'logits'), a_units=list(range(64)))
    real = x[:3000, :, 0] > 0
    l1_live = [int(u) for u in range(64) if R['a'][real][:, u].std() > 0]
    live = dict(l1=l1_live, g=[int(u) for u in range(64) if R['g'][:, u].std() > 0],
                h4=[int(u) for u in range(64) if R['h4'][:, u].std() > 0], S=[int(u) for u in range(64) if R['S'][:, u].std() > 0])
    if 32 not in live['l1']: live['l1'].append(32)
    cut = {f'a{u}': dict(alpha_pt=float(alpha[0, u]), alpha_deta=float(alpha[1, u]), alpha_dphi=float(alpha[2, u]), beta=float(beta[u]),
                         pt_threshold_at_axis=float(-beta[u] / alpha[0, u]) if alpha[0, u] != 0 else None) for u in live['l1']}
    log('live units', {k: len(v) for k, v in live.items()}, 'l1', live['l1'], 'S live == LIVE_S', sorted(live['S']) == sorted(LIVE_S))
    log('a32 cut', cut['a32'])
    del R
    res = dict(seeds=dict(main=2801, strata=2802), n_boot=B, live=live, l1_cuts=cut, ranks=list(RANKS), zs=list(ZS), named_S=NAMED_S,
               pT_sorted=srt)
    res['main'], pj = analyse(net, x, y, 'main', rng, B, True, live, sm)
    np.savez_compressed(OUT / f'{tag}_perjet.npz', **{f'main_{k}': v for k, v in pj.items()})
    (OUT / f'{tag}.json').write_text(json.dumps(res, indent=1)); log('main written')

    # ---- example jet for the figure: gluon, N <= 40, split r13 z0.5 changes the call, the rank-restore patch removes the change
    js = np.flatnonzero(pj['ch_split_r13_z0.5'] & ~pj['ch_split_r13_z0.5_rankPS'] & (y == 0) & (pj['N'] <= 40))
    if not len(js): js = np.flatnonzero(pj['ch_split_r13_z0.5'] & (pj['N'] <= 40))
    if len(js):
        j = int(js[0]); xj = x[j:j + 1]; sl = np.array([13]); Ys, d1, d2 = split_ext(xj, sl, 0.5); Yn, _ = noresort(xj, sl, 0.5)
        L = {nm: net.fwd(v, keep=('logits',))['logits'][0] for nm, v in (('unsplit', xj), ('split', Ys), ('noresort', Yn))}
        L['rank_PS'] = net.tail(net.rank_patch(xj, Ys, sl, d1, d2, 'PS')[0])[0]
        sm_ = lambda l: (np.exp(l - l.max()) / np.exp(l - l.max()).sum()).tolist()
        res['example'] = dict(index=j, label=CL[y[j]], N=int(nreal(xj)[0]), d1=int(d1[0]), d2=int(d2[0]),
                              pt_before=xj[0, :, 0].tolist(), pt_split=Ys[0, :, 0].tolist(), pt_noresort=Yn[0, :, 0].tolist(),
                              calls={k: CL[int(v.argmax())] for k, v in L.items()}, probs={k: sm_(v) for k, v in L.items()})
        log('example', {k: res['example'][k] for k in ('index', 'label', 'N', 'd1', 'd2', 'calls')})

    # ---- matched data pair (merge a real near-collinear pair)
    Ym, okm, dmin = merge_pair(x); sel = np.flatnonzero(okm); bt = Boot(len(x), B, 7)
    c0 = pj['c0']; cm = net.fwd(Ym, keep=('logits',))['logits'].argmax(1); chm = (cm != c0) & okm
    Mo0, Mo1 = moments(x), moments(Ym)
    res['data_pair_merge'] = dict(n=len(sel), frac_jets_with_pair=float(okm.mean()), changed=float(chm[sel].mean()) if len(sel) else None,
                                  ci=qci(bt.mean(chm, sel)) if len(sel) else None, **(flows(c0, cm, sel) if len(sel) else {}),
                                  dmin_quantiles=dict(zip(('q05', 'q25', 'q50', 'q75', 'q95'), np.quantile(dmin[np.isfinite(dmin)], [.05, .25, .5, .75, .95]).tolist())),
                                  max_rel_dmass=float(np.max(np.abs(Mo1['m'][sel] - Mo0['m'][sel]) / np.maximum(Mo0['m'][sel], 1e-9))) if len(sel) else None,
                                  split_r13_z05_same_jets=float(pj['ch_split_r13_z0.5'][sel].mean()) if len(sel) else None,
                                  rot_r13_same_jets=float(pj['ch_rot_r13'][sel].mean()) if len(sel) else None)
    log('data pair merge', {k: v for k, v in res['data_pair_merge'].items() if k != 'flow'})

    # ---- untrained control (centred logits)
    res['untrained'] = {}
    N0 = nreal(x)
    for s in (0, 1):
        un = Net(untrained=True, seed=s); L0 = un.fwd(x, keep=('logits',))['logits']; off = L0.mean(0); cu = (L0 - off).argmax(1)
        d = dict(call_dist_centred=np.bincount(cu, minlength=5).tolist(), acc_centred=float((cu == y).mean()))
        for k in RANKS:
            slot = np.where(N0 >= k + 2, k, -1); ok = slot >= 0; sel = np.flatnonzero(ok)
            for nm, Ye in ((f'split_r{k}_z0.5', split_ext(x, slot, 0.5)[0]), (f'rot_r{k}', rotate(x, slot, rng))):
                okk = ok
                ss = np.flatnonzero(okk); ch = ((un.fwd(Ye, keep=('logits',))['logits'] - off).argmax(1) != cu) & okk
                d[nm] = dict(changed=float(ch[ss].mean()), ci=qci(bt.mean(ch, ss)))
        res['untrained'][str(s)] = d; log(f'untrained {s}', d); del un
    (OUT / f'{tag}.json').write_text(json.dumps(res, indent=1))

    # ---- strata from sample(3000, 2802)
    x2, y2, _ = sample(per_class=(sm * 2) if sm else 3000, seed=2802); x2 = np.asarray(x2, np.float32); N2 = nreal(x2)
    lo = [np.flatnonzero((y2 == c) & (N2 <= 60)) for c in range(5)]; m = min(min(len(v) for v in lo), 1000 if not sm else sm)
    gb = np.sort(np.concatenate([rng.choice(v, m, replace=False) for v in lo]))
    hi = np.flatnonzero(N2 > 60); hi = np.sort(rng.choice(hi, min(len(hi), 4000 if not sm else 2 * sm), replace=False))
    for nm, idx in (('gbal', gb), ('hiN', hi)):
        res[nm], pjs = analyse(net, x2[idx], y2[idx], nm, rng, B, False, live, sm)
        res[nm]['sample_indices_n'] = int(len(idx))
        np.savez_compressed(OUT / f'{tag}_perjet_{nm}.npz', **pjs)
        (OUT / f'{tag}.json').write_text(json.dumps(res, indent=1))
    res['wall_s'] = time.time() - T0_
    (OUT / f'{tag}.json').write_text(json.dumps(res, indent=1))
    log(f'wrote {OUT / tag}.json; E28w DONE')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', type=int, default=0); main(ap.parse_args())
