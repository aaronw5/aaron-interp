"""E40 (claims A1, A3; new exploratory claim A19qg; stage 3): does JEDI-linear's quark-vs-gluon call count particles (an
IRC-unsafe quantity), and which objects carry that count?

Network: official JEDI-linear 3-feature n64 (exact fixed-point forward, interp.jedi_core via e28w's Net).
Margin = logit_g - logit_q = log p(g) - log p(q) (positive = toward gluon). Calls = arg-max of the 5 logits.
Samples: fit = jedi_core.sample(2000, seed 4001) (natural slope d margin / dN on unedited q/g jets);
         test = jedi_core.sample(2000, seed 4002) (all edits; q and g primary, W/Z/t side rows).
Particles are massless four-vectors with eta = deta, phi = dphi (the frame of the input; as e28w.moments).
Edits (test jets):
  C1 split_{r0,r13,soft}_f{0.5,0.2}: exact collinear split of the particle at pT rank 0, rank 13, or a random rank >= 30 (soft) into
     two particles at the same (deta, dphi) with pT f pT and (1 - f) pT; re-sorted by pT. Jet four-momentum, M0, M1, M2 exact;
     N + 1. Only jets with N <= 63 (no particle leaves the 64 input slots) and N >= rank + 2 (soft: N >= 31).
  C2 emit_k{1,3,5}: k particles of 0.5 GeV at r = 0.3, uniform random azimuth, are added; all other particles are scaled by one
     common factor and given one common Lorentz boost so that the summed jet four-momentum is exactly the unedited one
     (massless particles stay massless). emit_k3_pt0.05: same with 0.05 GeV (infrared limit). N + k; jets with N + k <= 64.
  C3 merge2: the two softest particles are replaced by one particle with their summed pT at their pT-weighted (deta, dphi); N - 1.
Controls:
  (a) nc_*: matched non-collinear split, same f, the two halves separated by dR = 0.05 along a random direction with their
      pT-weighted position at the parent's position (pair three-momentum fixed to first order; the pair gains a small mass).
  (b) rankpres_*: rank-preserving split computed exactly from per-slot pooled terms (e28w Net.rank_patch, mode PS): every
      particle other than the two halves contributes its term from its unsplit slot, so no particle crosses a rank-window edge
      (in particular the 13|14 edge); only the halves and the filled empty slot change.
  rot_*: the same particle rotated about the jet axis by a random azimuth (N, ranks fixed). comp_k5: the compensation of
      emit_k5 (scale + boost) without adding the particles. move2: the two softest particles moved to their pT-weighted
      position without merging (N fixed; same angular change as merge2).
  (c) untrained JEDI-linear, 5 seeds (Jedi untrained=True, seeds 0-4): margin change in units of that network's margin SD.
  (d) null_resort: particles randomly permuted then re-sorted by pT (an exact no-op).
Mechanism: S-level holds (S object set at its unedited value; the rest of the edited jet unchanged) for S19 (soft multiplicity),
  S0 (pT of the 14 hardest), S29 (sum pT r^2 + sum r^2 + soft count), S19+S0; null = every live-S set of the same size
  (19 singles / 171 pairs, enumerated). Also all g (context) held. share removed = 1 - sum dmargin(hold) / sum dmargin(edit).
Read-out calibration: R_N = mean edit dmargin / (natural slope x dN), slope = OLS of margin on N over fit q/g jets (pooled; and
  within class).
Statistics: 1000 bootstrap resamples over jets; sign-flip permutation for paired differences; exact McNemar; Holm.
Run: ./job.sh e40-run 1 2 'python -m interp.e40_qg_count_jedi'   (--smoke N: N jets per class)
Writes $AI_RESULTS/jedi/e40_qg_count[_smoke].json and e40_qg_count_perjet[_smoke].npz.
"""
import argparse
import gc
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp.e28w_split_route_jedi import Net, nreal, resort, rotate, mcnemar, holm, qci, LIVE_S, binom_p, B_FWD  # noqa: E402

RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); OUT = RES / 'jedi'
T0_ = time.time()
CL = ['g', 'q', 'W', 'Z', 't']
NAMED = {'S19': [19], 'S0': [0], 'S29': [29], 'S19+S0': [19, 0]}


def log(*a): print(f'[{time.time() - T0_:7.0f}s]', *a, flush=True)


# ------------------------------------------------------------------ four-vectors (massless, eta = deta, phi = dphi)
def four(pt, e, h):
    return np.stack([pt * np.cosh(e), pt * np.cos(h), pt * np.sin(h), pt * np.sinh(e)], -1)


def mass(P): return np.sqrt(np.maximum(P[..., 0] ** 2 - (P[..., 1:] ** 2).sum(-1), 0))


def jet4(X):
    x = X.astype(np.float64); return four(x[:, :, 0], x[:, :, 1], x[:, :, 2]).sum(1)


def boost_to(A, T):
    """Lorentz matrix L with L A = T for two four-vectors of equal mass (A to rest, then rest to T)"""
    def L(b):  # boost into the frame moving with velocity b
        b2 = b @ b; g = 1 / np.sqrt(1 - b2); M = np.eye(4)
        M[0, 0] = g; M[0, 1:] = -g * b; M[1:, 0] = -g * b
        if b2 > 0: M[1:, 1:] += (g - 1) * np.outer(b, b) / b2
        return M
    return L(-T[1:] / T[0]) @ L(A[1:] / A[0])


# ------------------------------------------------------------------ edits on raw (J, 64, 3) pT-sorted arrays, padding = 0
def split_gen(X, slot, f, sep, rng):
    """split the particle at `slot` into f pT and (1-f) pT; the halves sit at pT-weighted centroid = parent, separation sep"""
    J = len(X); N = nreal(X); rows = np.flatnonzero(slot >= 0); k = slot[rows]
    Y = np.concatenate([X, np.zeros((J, 1, 3), np.float32)], 1)
    p = X[rows, k, 0].astype(np.float64); e = X[rows, k, 1].astype(np.float64); h = X[rows, k, 2].astype(np.float64)
    th = rng.uniform(0, 2 * np.pi, len(rows)); ue, uh = np.sin(th) * sep, np.cos(th) * sep
    p1 = (f * p).astype(np.float32); p2 = (p - p1.astype(np.float64)).astype(np.float32)
    Y[rows, k, 0] = p1; Y[rows, k, 1] = e - (1 - f) * ue; Y[rows, k, 2] = h - (1 - f) * uh
    n = N[rows]; Y[rows, n, 0] = p2; Y[rows, n, 1] = e + f * ue; Y[rows, n, 2] = h + f * uh
    Ys, o = resort(Y); inv = np.argsort(o, axis=1)
    d1 = np.full(J, -1); d2 = np.full(J, -1); d1[rows] = inv[rows, k]; d2[rows] = inv[rows, n]
    return Ys[:, :64].copy(), d1, d2


def emit(X, k, ptk, r, rng, add=True, ok_in=None):
    """C2: add k particles (ptk GeV, radius r, random azimuth); the others get one common scale + one common boost so that the
    summed four-momentum is exact. add=False: the compensation only (control)."""
    J = len(X); N = nreal(X); Y = X.copy(); ok = (N + k <= 64) if ok_in is None else ok_in.copy()
    devP = 0.0; maxshift = []
    for j in np.flatnonzero(ok):
        n = N[j]; x = X[j, :n].astype(np.float64)
        P4 = four(x[:, 0], x[:, 1], x[:, 2]); P = P4.sum(0)
        th = rng.uniform(0, 2 * np.pi, k); ea, ha = r * np.sin(th), r * np.cos(th)
        Q4 = four(np.full(k, ptk), ea, ha); T = P - Q4.sum(0)
        mP, mT = mass(P), mass(T)
        if not (mT > 0 and mP > 0): ok[j] = False; continue
        c = mT / mP; Lm = boost_to(c * P, T); Pn = (c * P4) @ Lm.T
        pt = np.hypot(Pn[:, 1], Pn[:, 2]); en = np.arcsinh(Pn[:, 3] / pt); hn = np.arctan2(Pn[:, 2], Pn[:, 1])
        Y[j, :n, 0], Y[j, :n, 1], Y[j, :n, 2] = pt, en, hn
        if add:
            Y[j, n:n + k, 0], Y[j, n:n + k, 1], Y[j, n:n + k, 2] = ptk, ea, ha
            tot = Pn.sum(0) + Q4.sum(0)
        else:
            tot = Pn.sum(0) + Q4.sum(0)   # the four-momentum the jet would have with the particles; the particles are absent
        devP = max(devP, float(np.abs(tot - P).max() / P[0]))
        maxshift.append(float(np.abs(pt / x[:, 0] - 1).max()))
    Ys, _ = resort(Y)
    return Ys, ok, dict(max_rel_dev_jet4=devP, median_max_rel_pt_change=float(np.median(maxshift)) if maxshift else None)


def merge_soft(X, move_only=False):
    """C3: the two softest particles -> one at their pT-weighted (deta, dphi) with summed pT (move_only: both moved there)"""
    Y = X.copy(); N = nreal(X); rows = np.flatnonzero(N >= 3); a, b = N[rows] - 2, N[rows] - 1
    pa, pb = X[rows, a, 0].astype(np.float64), X[rows, b, 0].astype(np.float64)
    e = (pa * X[rows, a, 1] + pb * X[rows, b, 1]) / (pa + pb); h = (pa * X[rows, a, 2] + pb * X[rows, b, 2]) / (pa + pb)
    if move_only:
        Y[rows, a, 1] = e; Y[rows, a, 2] = h; Y[rows, b, 1] = e; Y[rows, b, 2] = h
    else:
        Y[rows, a, 0] = pa + pb; Y[rows, a, 1] = e; Y[rows, a, 2] = h; Y[rows, b] = 0
    ok = np.zeros(len(X), bool); ok[rows] = True
    Ys, _ = resort(Y); return Ys, ok


def null_resort(X, rng):
    Y = X.copy(); N = nreal(X)
    for j in range(len(X)):
        Y[j, :N[j]] = X[j, rng.permutation(N[j])]
    return resort(Y)[0]


# ------------------------------------------------------------------ statistics
class Boot:
    def __init__(self, J, B, seed):
        rng = np.random.default_rng(seed); self.W = np.zeros((B, J), np.float32)
        for b in range(B): self.W[b] = np.bincount(rng.integers(0, J, J), minlength=J)

    def mean(self, v, sel):
        W = self.W[:, sel]; return (W @ v[sel].astype(np.float32)) / np.maximum(W.sum(1), 1)

    def share(self, dE, dH, sel):
        W = self.W[:, sel]; return 1 - (W @ dH[sel].astype(np.float32)) / (W @ dE[sel].astype(np.float32))


def signflip_p(d, n_perm=5000, seed=0):
    d = np.asarray(d, np.float64); d = d[np.isfinite(d)]
    if len(d) == 0: return None
    rng = np.random.default_rng(seed); obs = abs(d.mean()); cnt = 0
    for i in range(0, n_perm, 500):
        s = rng.choice([-1.0, 1.0], size=(min(500, n_perm - i), len(d)))
        cnt += int((np.abs(s @ d / len(d)) >= obs - 1e-15).sum())
    return (1 + cnt) / (1 + n_perm)


def ols_slope(m, N, cls=None):
    if cls is None:
        A = np.stack([N, np.ones_like(N)], 1)
    else:
        A = np.stack([N] + [(cls == c).astype(float) for c in np.unique(cls)], 1)
    return float(np.linalg.lstsq(A, m, rcond=None)[0][0])


# ------------------------------------------------------------------ main
def main(a):
    OUT.mkdir(parents=True, exist_ok=True)
    from interp.jedi_core import sample
    sm = a.smoke; per = sm if sm else 2000; B = 200 if sm else 1000; tag = 'e40_qg_count' + ('_smoke' if sm else '')
    rng = np.random.default_rng(4040)
    net = Net()
    # ---- fit sample: accuracy + natural slope
    xf, yf, _ = sample(per, 4001)
    Rf = net.fwd(xf, keep=('logits',)); lf = Rf['logits']; accf = float((lf.argmax(1) == yf).mean())
    qgf = np.flatnonzero(yf <= 1); mf = lf[qgf, 0] - lf[qgf, 1]; Nf = nreal(xf)[qgf].astype(float); cf = yf[qgf]
    sl_pool = ols_slope(mf, Nf); sl_within = ols_slope(mf, Nf, cf)
    rb = np.random.default_rng(7); bs_pool, bs_within = [], []
    for _ in range(B):
        i = rb.integers(0, len(mf), len(mf)); bs_pool.append(ols_slope(mf[i], Nf[i])); bs_within.append(ols_slope(mf[i], Nf[i], cf[i]))
    bs_pool, bs_within = np.array(bs_pool), np.array(bs_within)
    edges = [0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 64, 65]
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (Nf >= lo) & (Nf < hi)
        bins.append(dict(lo=lo, hi=hi - 1, n=int(s.sum()), mean_margin=float(mf[s].mean()) if s.any() else None,
                         n_g=int((s & (cf == 0)).sum()), mean_margin_g=float(mf[s & (cf == 0)].mean()) if (s & (cf == 0)).any() else None,
                         n_q=int((s & (cf == 1)).sum()), mean_margin_q=float(mf[s & (cf == 1)].mean()) if (s & (cf == 1)).any() else None))
    nat = dict(slope_pooled=sl_pool, slope_pooled_ci=qci(bs_pool), slope_within_class=sl_within, slope_within_ci=qci(bs_within),
               n_qg=len(mf), margin_sd=float(mf.std()), mean_N_g=float(Nf[cf == 0].mean()), mean_N_q=float(Nf[cf == 1].mean()), bins=bins)
    log(f'fit sample seed 4001: {len(xf)} jets acc {accf:.4f}; natural slope pooled {sl_pool:.4f} {qci(bs_pool)} within {sl_within:.4f} {qci(bs_within)}')
    del xf, Rf; gc.collect()

    # ---- test sample
    X0, y, _ = sample(per, 4002); J = len(X0); N0 = nreal(X0)
    R0 = net.fwd(X0, keep=('logits', 'S', 'g')); L0 = R0['logits']; c0 = L0.argmax(1); acc = float((c0 == y).mean())
    m0 = L0[:, 0] - L0[:, 1]; S0v = R0['S']; G0 = R0['g']
    lt = net.tail(S0v); tail_ok = float((lt.argmax(1) == c0).mean()); tail_dev = float(np.abs(lt - L0).max())
    log(f'test sample seed 4002: {J} jets acc {acc:.4f} (band 0.806-0.813); tail check {tail_ok:.5f} {tail_dev:.2e}')
    qg = y <= 1; bt = Boot(J, B, 11)
    res = dict(seeds=dict(fit=4001, test=4002), per_class=per, n_boot=B, acc_fit=accf, acc_test=acc,
               tail_check=dict(calls_equal=tail_ok, max_abs_logit_diff=tail_dev), natural=nat,
               test_N_mean=dict((CL[c], float(N0[y == c].mean())) for c in range(5)),
               test_N64_frac=dict((CL[c], float((N0[y == c] == 64).mean())) for c in range(5)), live_S=LIVE_S, named=NAMED)
    perjet = dict(y=y, N=N0, m0=m0, c0=c0)
    edits, Xed, Sed, OK = {}, {}, {}, {}
    mom0 = jet4(X0); mjet0 = mass(mom0)

    def record(name, Xe, ok, dN, extra=None, S=None, logits=None):
        if logits is None:
            R = net.fwd(Xe, keep=('logits', 'S')); Le, Se = R['logits'], R['S']
        else:
            Le, Se = logits, S
        ce = Le.argmax(1); d = (Le[:, 0] - Le[:, 1]) - m0; d[~ok] = np.nan
        out = dict(dN=dN, n=int(ok.sum()), n_qg=int((ok & qg).sum()))
        for nm, cs in (('qg', qg), ('g', y == 0), ('q', y == 1), ('W', y == 2), ('Z', y == 3), ('t', y == 4)):
            sel = np.flatnonzero(ok & cs)
            if not len(sel): continue
            ch = (ce != c0)
            out[nm] = dict(n=len(sel), dmargin=float(d[sel].mean()), dmargin_ci=qci(bt.mean(np.nan_to_num(d), sel)),
                           calls_changed=float(ch[sel].mean()), calls_changed_ci=qci(bt.mean(ch.astype(float), sel)))
        sel = np.flatnonzero(ok & qg)
        F = np.zeros((5, 5), int); np.add.at(F, (c0[sel], ce[sel]), 1)
        out['qg'].update(q_to_g=int(F[1, 0]), g_to_q=int(F[0, 1]), p_sign=binom_p(int(F[1, 0]), int(F[1, 0] + F[0, 1])),
                         q_to_g_frac=float(F[1, 0] / len(sel)), flow=F.tolist(),
                         dmargin_perm_p=signflip_p(d[sel], 2000 if sm else 5000, 1))
        if not np.isfinite(out['qg']['dmargin']): out['qg']['dmargin'] = None
        if dN:
            bm = bt.mean(np.nan_to_num(d), sel)
            out['R_N_pooled'] = float(out['qg']['dmargin'] / (sl_pool * dN))
            out['R_N_pooled_ci'] = qci(bm / (bs_pool[:B] * dN))
            out['R_N_within'] = float(out['qg']['dmargin'] / (sl_within * dN))
            out['R_N_within_ci'] = qci(bm / (bs_within[:B] * dN))
        if Xe is not None:
            mom = jet4(Xe); out['max_rel_jet4_change'] = float((np.abs(mom - mom0).max(1) / mom0[:, 0])[ok].max())
            out['jet_mass_change_GeV'] = dict(mean=float((mass(mom) - mjet0)[ok].mean()), max_abs=float(np.abs(mass(mom) - mjet0)[ok].max()))
            out['N_change'] = dict(mean=float((nreal(Xe) - N0)[ok].mean()))
        if extra: out.update(extra)
        edits[name] = out; perjet['d_' + name] = d.astype(np.float32); perjet['c_' + name] = ce.astype(np.int8)
        OK[name] = ok; Sed[name] = Se
        if Xe is not None: Xed[name] = Xe
        q = out['qg']
        log(f'{name}: n_qg {q["n"]} dmargin {q["dmargin"]:+.4f} {q["dmargin_ci"]} calls {q["calls_changed"]:.4f} '
            f'q->g {q["q_to_g"]} g->q {q["g_to_q"]}' + (f' R_N {out["R_N_pooled"]:.2f}' if dN else ''))
        return d, ce

    # (d) null edit
    record('null_resort', null_resort(X0, rng), np.ones(J, bool), 0)
    # C1 + (a) + rot + (b)
    soft_rank = np.array([rng.integers(30, n) if n >= 31 else -1 for n in N0])
    slots = {'r0': np.where((N0 >= 2) & (N0 <= 63), 0, -1), 'r13': np.where((N0 >= 15) & (N0 <= 63), 13, -1),
             'soft': np.where((N0 >= 31) & (N0 <= 63), soft_rank, -1)}
    for loc, slot in slots.items():
        ok = slot >= 0
        record(f'rot_{loc}', rotate(X0, slot, rng), ok, 0)
        for f in (0.5, 0.2):
            Ys, d1, d2 = split_gen(X0, slot, f, 0.0, rng)
            okk = ok & (d1 >= 0) & (d2 >= 0)
            cross = ((d1 >= 14) | (d2 >= 14)) & (slot <= 13) & okk   # a half lands beyond the 13|14 window edge
            record(f'split_{loc}_f{f}', Ys, okk, 1, extra=dict(frac_half_beyond_13=float(cross[okk].mean())))
            Sp, chk = net.rank_patch(X0, Ys, slot, d1, d2, 'PS')
            lp = net.tail(Sp)
            record(f'rankpres_{loc}_f{f}', None, okk, 1, S=Sp, logits=lp,
                   extra=dict(check_max_abs_dS_split=float(np.abs(chk - Sed[f'split_{loc}_f{f}'])[okk].max())))
            Yn, _, _ = split_gen(X0, slot, f, 0.05, rng)
            record(f'nc_{loc}_f{f}', Yn, okk, 1)
    # C2
    for k in (1, 3, 5):
        Ye, oke, info = emit(X0, k, 0.5, 0.3, np.random.default_rng(100 + k))
        record(f'emit_k{k}', Ye, oke, k, extra=info)
    Ye, oke, info = emit(X0, 3, 0.05, 0.3, np.random.default_rng(103))
    record('emit_k3_pt0.05', Ye, oke, 3, extra=info)
    Ye, oke, info = emit(X0, 5, 0.5, 0.3, np.random.default_rng(105), add=False, ok_in=OK['emit_k5'])
    record('comp_k5', Ye, oke, 0, extra=info)
    # C3
    Ym, okm = merge_soft(X0); record('merge2', Ym, okm, -1)
    Ymv, okv = merge_soft(X0, move_only=True); record('move2', Ymv, okv, 0)

    # ---- paired comparisons (H0: collinear = control)
    pairs = [(f'split_{l}_f{f}', f'nc_{l}_f{f}') for l in slots for f in (0.5, 0.2)] + \
            [(f'split_{l}_f{f}', f'rot_{l}') for l in slots for f in (0.5, 0.2)] + \
            [(f'split_{l}_f{f}', f'rankpres_{l}_f{f}') for l in slots for f in (0.5, 0.2)] + \
            [('emit_k5', 'comp_k5'), ('emit_k3', 'emit_k3_pt0.05'), ('merge2', 'move2')]
    comp = {}
    for e1, e2 in pairs:
        ok = OK[e1] & OK[e2] & qg; sel = np.flatnonzero(ok)
        d1_, d2_ = perjet['d_' + e1], perjet['d_' + e2]; dd = (d1_ - d2_).astype(np.float64)
        ch1 = perjet['c_' + e1] != c0; ch2 = perjet['c_' + e2] != c0
        f1 = (c0 == 1) & (perjet['c_' + e1] == 0); f2 = (c0 == 1) & (perjet['c_' + e2] == 0)
        comp[f'{e1}|{e2}'] = dict(n=len(sel), diff=float(dd[sel].mean()), diff_ci=qci(bt.mean(np.nan_to_num(dd), sel)),
                                  perm_p=signflip_p(dd[sel], 2000 if sm else 5000, 2),
                                  ratio=float(d1_[sel].mean() / d2_[sel].mean()) if d2_[sel].mean() != 0 else None,
                                  mcnemar_changed=mcnemar(ch1[sel], ch2[sel]), mcnemar_q_to_g=mcnemar(f1[sel], f2[sel]))
        log(f'{e1} vs {e2}: diff {comp[f"{e1}|{e2}"]["diff"]:+.4f} {comp[f"{e1}|{e2}"]["diff_ci"]} p {comp[f"{e1}|{e2}"]["perm_p"]:.4g}')
    res['edits'] = edits; res['compare'] = comp
    (OUT / f'{tag}.json').write_text(json.dumps(res, indent=1)); log('edits written')

    # ---- mechanism: S holds, g hold
    HOLD_EDITS = [f'split_{l}_f{f}' for l in slots for f in (0.5, 0.2)] + [f'nc_{l}_f0.5' for l in slots] + \
                 ['emit_k1', 'emit_k3', 'emit_k5', 'merge2']
    comb = {1: [[u] for u in LIVE_S], 2: [list(c) for c in itertools.combinations(LIVE_S, 2)]}
    if sm: comb[2] = comb[2][:30]
    holds = {}
    for en in HOLD_EDITS:
        ok = OK[en] & qg; sel = np.flatnonzero(ok); dE = np.nan_to_num(perjet['d_' + en]).astype(np.float64)
        SE = Sed[en]; tot = dE[sel].sum()

        def held(units):
            m = np.zeros(64, np.float32); m[units] = 1
            lh = net.tail(SE[sel], m, S0v[sel]); dH = np.zeros(J); dH[sel] = (lh[:, 0] - lh[:, 1]) - m0[sel]
            return dH
        H = dict(total_dmargin=float(tot), n=len(sel))
        null = {}
        for sz, sets in comb.items():
            null[sz] = {tuple(s): float(1 - held(s)[sel].sum() / tot) for s in sets}
        H['singles'] = {f'S{s[0]}': v for s, v in null[1].items()}
        H['pair_null_q95'] = float(np.quantile(list(null[2].values()), .95)); H['pair_null_mean'] = float(np.mean(list(null[2].values())))
        H['single_null_q95'] = float(np.quantile(list(null[1].values()), .95))
        for nm, units in NAMED.items():
            dH = held(units); sh = 1 - dH[sel].sum() / tot; bs = bt.share(dE, dH, sel)
            nl = [v for s, v in null[len(units)].items() if sorted(s) != sorted(units)]
            H[nm] = dict(share=float(sh), ci=qci(bs), p_share_le0=float(min(1, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))),
                         p_share_gt_half=float((bs <= 0.5).mean()),
                         rank_p=float((1 + sum(v >= sh for v in nl)) / (1 + len(nl))), n_null=len(nl))
        # g held (context units), full forward with g frozen at the unedited jet's g
        Xe = Xed[en]
        Gs = G0[sel]; gm = np.ones(64, np.float32)

        def ovg(i, n):
            v = np.zeros((B_FWD, 64), np.float32); v[:n] = Gs[i:i + n]
            return {'g': (gm, v.reshape(B_FWD, 1, 64) if net.kd_P else v)}
        r = net.fwd(Xe[sel], ov_fn=ovg, keep=('logits',))['logits']
        dH = np.zeros(J); dH[sel] = (r[:, 0] - r[:, 1]) - m0[sel]; bs = bt.share(dE, dH, sel)
        H['g_all'] = dict(share=float(1 - dH[sel].sum() / tot), ci=qci(bs))
        holds[en] = H
        log(f'holds {en}: ' + ' '.join(f'{nm} {H[nm]["share"]:+.3f} (rank p {H[nm]["rank_p"]:.3f})' for nm in NAMED) +
            f' g {H["g_all"]["share"]:+.3f}; top singles ' +
            ', '.join(f'{k} {v:+.2f}' for k, v in sorted(H['singles'].items(), key=lambda t: -t[1])[:4]))
    # Holm over named holds x primary edits (bootstrap two-sided p of share = 0, and rank p)
    fam = [(en, nm) for en in ['split_r0_f0.5', 'split_r13_f0.5', 'split_soft_f0.5', 'emit_k3', 'merge2'] for nm in NAMED]
    hb = holm([holds[en][nm]['p_share_le0'] for en, nm in fam]); hr = holm([holds[en][nm]['rank_p'] for en, nm in fam])
    for (en, nm), p1, p2 in zip(fam, hb, hr):
        holds[en][nm]['holm_boot'] = p1; holds[en][nm]['holm_rank'] = p2
    res['holds'] = holds; res['holm_family'] = [f'{e}:{n}' for e, n in fam]
    (OUT / f'{tag}.json').write_text(json.dumps(res, indent=1)); log('holds written')

    # ---- example jet for the figure: first gluon jet with 35 <= N <= 50 in the test sample that has a soft split
    ex = int(np.flatnonzero((y == 0) & (N0 >= 35) & (N0 <= 50) & OK['split_soft_f0.5'])[0])
    res['example'] = dict(index=ex, N=int(N0[ex]), slot=int(slots['soft'][ex]), x0=X0[ex, :N0[ex]].tolist(),
                          x_split=Xed['split_soft_f0.5'][ex, :N0[ex] + 1].tolist(), x_nc=Xed['nc_soft_f0.5'][ex, :N0[ex] + 1].tolist(),
                          m0=float(m0[ex]), m_split=float(m0[ex] + perjet['d_split_soft_f0.5'][ex]),
                          m_nc=float(m0[ex] + perjet['d_nc_soft_f0.5'][ex]), c0=CL[c0[ex]], c_split=CL[perjet['c_split_soft_f0.5'][ex]],
                          jet_mass=float(mjet0[ex]))
    (OUT / f'{tag}.json').write_text(json.dumps(res, indent=1))
    np.savez_compressed(OUT / f'{tag}_perjet.npz', **perjet)
    del Sed; gc.collect()

    # ---- (c) untrained networks, 5 seeds: margin change in units of that network's margin SD (q/g jets)
    UNT_EDITS = [k for k in edits if k != 'null_resort' and not k.startswith('rankpres')]
    unt = {k: [] for k in UNT_EDITS}; unt_raw = {k: [] for k in UNT_EDITS}; sds = []
    trained_sd = float(m0[qg].std())
    res['trained_margin_sd'] = trained_sd
    for s in range(5):
        un = Net(untrained=True, seed=s); lu = un.fwd(X0, keep=('logits',))['logits']; mu = lu[:, 0] - lu[:, 1]; sd = float(mu[qg].std())
        sds.append(sd)
        for k in UNT_EDITS:
            ok = OK[k] & qg; sel = np.flatnonzero(ok)
            le = un.fwd(Xed[k][sel], keep=('logits',))['logits']; draw = (le[:, 0] - le[:, 1]) - mu[sel]
            unt[k].append(float(draw.mean() / sd) if sd > 0 else 0.0)
            unt_raw[k].append(dict(mean_raw=float(draw.mean()), frac_jets_changed=float((np.abs(draw) > 0).mean()),
                                   calls_changed=float((le.argmax(1) != lu[sel].argmax(1)).mean())))
        log(f'untrained seed {s}: margin sd {sd:.4g}; ' + ', '.join(f'{k} {unt[k][-1]:+.4f}' for k in UNT_EDITS[:8]))
        del un; gc.collect()
    res['untrained'] = dict(margin_sd=sds, note='seed with margin sd 0 (constant margin on q/g jets) entered as 0',
                            per_edit={k: dict(per_seed=v, raw=unt_raw[k], mean=float(np.mean(v)), sd_over_seeds=float(np.std(v, ddof=1)),
                                                            max_abs=float(np.max(np.abs(v))),
                                                            trained_sd_units=float(edits[k]['qg']['dmargin'] / trained_sd))
                                                    for k, v in unt.items()})
    (OUT / f'{tag}.json').write_text(json.dumps(res, indent=1)); log('done', OUT / f'{tag}.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', type=int, default=0); main(ap.parse_args())
