"""E28 (claims A3 / A1; exploratory): collinear and soft splitting inside JEDI-linear (3-feature n64, official, fixed point).
Which named second-round objects move, and which calls change, when one particle is split collinearly or soft particles are added?

Sample: interp.jedi_core.sample(per_class=2000, seed=1202) = 10,000 test jets; accuracy check on all of them; edits on the jets with
  at most 60 real particles (pT > 0), so that empty slots exist for the extra particle(s) and no real particle leaves the 64 slots.
Inputs are the raw (pT [GeV], deta_rel, dphi_rel) of each particle; r = sqrt(deta^2 + dphi^2).
Edits
  split_r{k}_z{z}   k in {0, 5, 13, soft} (soft = a random rank in 21..N-1, N >= 22), z in {0.5, 0.8}: the particle of pT rank k is
                    replaced by two particles at the same (deta, dphi) with pT z pT_k and (1 - z) pT_k; the jet is re-sorted by pT.
                    M0 = sum pT, M1 = sum pT r, M2 = sum pT r^2 (and every kappa = 1 angularity) are unchanged; only N and ranks move.
  splitfix_r{k}     split (z 0.5) followed by removal of the softest real particle (count fixed; ranks still shift).
  soft_add{n}       n in 1..5 particles of pT 0.5 GeV, each at r drawn from the jet's own particles (uniform over real particles),
                    random azimuth about the axis (requires N + n <= 64). Reported: dM0 = 0.5 n GeV, relative dM2.
Controls
  (a) swap_r{k}     matched-rank control: the angles of the particles at ranks k and k+1 are exchanged (the particle that sat at rank k
                    is moved one rank down, its pT exchanged with its neighbour); N fixed, no split.
  (b) pad0_at / pad0_origin: an extra entry with pT 0 in the first empty slot, at the angles of the rank-0 particle / at (0, 0).
  (c) untrained network: jedi_core.Jedi(untrained=True, seed) for seeds 0, 1, same edits.
  (d) scale: drop_softest (the softest real particle removed, N - 1) and matched data pairs: for every jet, its nearest neighbour of
      the same true class in standardised (log M0, log M2/M0, M1/M0, log m) with N' = N + 1, and the nearest with N' = N; call
      disagreement rates of the two pair types.
Holds (mediation; run_jetwise override of S at the unedited jet's values): S19, S0, S0+S19, the 17 rank-window objects (live S with a
  slot window other than 0-63, E20), the two windowless objects S29+S30, all 19 live S (sanity: removes 100 %), g (all 64 context
  units), and each live S unit alone (Holm over the 19 units per edit).  Share removed = 1 - #changed(edit + hold) / #changed(edit).
Statistics: 1000 multinomial bootstrap resamples over jets for every rate and share; exact McNemar on correct/incorrect per edit; split vs
  matched-rank control: exact McNemar and sign-flip permutation (10,000) on the paired changed indicators; Holm over the grid of tests.
Run: ./job.sh e28-run 1 2 'python -m interp.e28_irc_split_jedi'   (--smoke N: N jets per class)
Writes $AI_RESULTS/jedi/e28_irc_split[_smoke].json and e28_irc_split_perjet.npz.
"""
import argparse
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
WINDOWLESS = [29, 30]                                                                   # slot window 0-63 (E20)
WINDOW = [u for u in LIVE_S if u not in WINDOWLESS]
NAMED = [0, 1, 19, 29, 30]
RANKS = (0, 5, 13, 'soft'); ZS = (0.5, 0.8)


def log(*a):
    print(f'[{time.time() - T0_:7.0f}s]', *a, flush=True)


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
    """multinomial bootstrap weights over the J jets of the edit subset (the same resamples for every quantity)"""

    def __init__(self, J, B, seed):
        rng = np.random.default_rng(seed); self.W = np.zeros((B, J), np.float32)
        for b in range(B): self.W[b] = np.bincount(rng.integers(0, J, J), minlength=J)

    def mean(self, v, sel):
        W = self.W[:, sel]; return (W @ v[sel].astype(np.float32)) / np.maximum(W.sum(1), 1)

    def ratio_removed(self, chE, chH, sel):
        W = self.W[:, sel]; return 1 - (W @ chH[sel].astype(np.float32)) / np.maximum(W @ chE[sel].astype(np.float32), 1)


def perm_signflip(a, b, n=10000, seed=0):
    d = a.astype(int) - b.astype(int); d = d[d != 0]
    if len(d) == 0: return 1.0
    obs = abs(d.sum()); rng = np.random.default_rng(seed)
    null = np.abs((2 * rng.binomial(len(d), 0.5, n) - len(d)))
    return float((1 + (null >= obs).sum()) / (1 + n))


# ------------------------------------------------------------------ edits on raw (J, 64, 3) arrays, pT-sorted, padding = 0
def nreal(X): return (X[:, :, 0] > 0).sum(1)


def resort(Y):
    o = np.argsort(-Y[:, :, 0], axis=1, kind='stable'); return np.take_along_axis(Y, o[:, :, None], 1)


def moments(X):
    pt = X[:, :, 0].astype(np.float64); r = np.hypot(X[:, :, 1], X[:, :, 2]).astype(np.float64)
    M0 = pt.sum(1); M1 = (pt * r).sum(1); M2 = (pt * r * r).sum(1)
    px, py = (pt * np.cos(X[:, :, 2])).sum(1), (pt * np.sin(X[:, :, 2])).sum(1)
    pz, E = (pt * np.sinh(X[:, :, 1])).sum(1), (pt * np.cosh(X[:, :, 1])).sum(1)
    m = np.sqrt(np.maximum(E ** 2 - px ** 2 - py ** 2 - pz ** 2, 0))
    return dict(M0=M0, M1=M1, M2=M2, m=m)


def pick_rank(N, k, rng):
    """slot index per jet (-1 = not applicable)"""
    if k == 'soft':
        ok = N >= 22; s = np.full(len(N), -1)
        s[ok] = 21 + (rng.random(ok.sum()) * (N[ok] - 21)).astype(int); return s
    return np.where(N > k + 1, k, -1)      # at least one particle below (needed by the matched-rank swap)


def split(X, slot, z):
    Y = X.copy(); N = nreal(X); rows = np.flatnonzero(slot >= 0); k = slot[rows]
    p = X[rows, k, 0].astype(np.float64)
    Y[rows, k, 0] = (z * p).astype(np.float32)
    Y[rows, N[rows], 0] = (p - Y[rows, k, 0].astype(np.float64)).astype(np.float32)
    Y[rows, N[rows], 1] = X[rows, k, 1]; Y[rows, N[rows], 2] = X[rows, k, 2]
    return resort(Y)


def drop_softest(X, rows=None):
    Y = X.copy(); N = nreal(X); rows = np.arange(len(X)) if rows is None else rows
    Y[rows, N[rows] - 1] = 0; return Y


def swap(X, slot):
    Y = X.copy(); rows = np.flatnonzero(slot >= 0); k = slot[rows]
    for c in (1, 2):
        a, b = X[rows, k, c].copy(), X[rows, k + 1, c].copy(); Y[rows, k, c] = b; Y[rows, k + 1, c] = a
    return Y


def soft_add(X, n, rng, pt=0.5):
    Y = X.copy(); N = nreal(X); ok = N + n <= 64; rows = np.flatnonzero(ok)
    r_all = np.hypot(X[:, :, 1], X[:, :, 2])
    for j in range(n):
        src = (rng.random(len(rows)) * N[rows]).astype(int); r = r_all[rows, src]; th = rng.uniform(0, 2 * np.pi, len(rows))
        Y[rows, N[rows] + j, 0] = pt; Y[rows, N[rows] + j, 1] = r * np.cos(th); Y[rows, N[rows] + j, 2] = r * np.sin(th)
    return resort(Y), ok


def pad0(X, at_origin):
    Y = X.copy(); N = nreal(X); rows = np.arange(len(X))
    Y[rows, N, 0] = 0.0
    if not at_origin: Y[rows, N, 1] = X[rows, 0, 1]; Y[rows, N, 2] = X[rows, 0, 2]
    return Y


def top14(X): return X[:, :14, 0].astype(np.float64).sum(1)
def soft_count(X): return ((X[:, 13:, 0] > 0) & (X[:, 13:, 0] < 2)).sum(1)


# ------------------------------------------------------------------ data-pair scale control (d)
def matched_pairs(x, y, calls, rng, Bn=1000):
    Mo = moments(x); N = nreal(x)
    F = np.stack([np.log(Mo['M0']), np.log(Mo['M2'] / Mo['M0'] + 1e-9), Mo['M1'] / Mo['M0'], np.log(Mo['m'] + 1e-3)], 1)
    F = (F - F.mean(0)) / F.std(0)
    res = {}
    for name, dN in (('dN1', 1), ('dN0', 0)):
        dif, dist = [], []
        for c in range(5):
            I = np.flatnonzero(y == c); D = ((F[I, None, :] - F[None, I, :]) ** 2).sum(-1)
            ok = (N[I][None, :] == N[I][:, None] + dN)
            if dN == 0: np.fill_diagonal(ok, False)
            D = np.where(ok, D, np.inf); j = D.argmin(1); has = np.isfinite(D.min(1))
            dif.append(calls[I[has]] != calls[I[j[has]]]); dist.append(np.sqrt(D[has, j[has]]))
        dif = np.concatenate(dif); dist = np.concatenate(dist)
        bs = [dif[rng.integers(0, len(dif), len(dif))].mean() for _ in range(Bn)]
        res[name] = dict(n=int(len(dif)), call_differs=float(dif.mean()), ci=qci(bs), median_match_dist=float(np.median(dist)),
                         _dif=dif)
    a, b = res['dN1'].pop('_dif'), res['dN0'].pop('_dif')
    bs = [a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean() for _ in range(Bn)]
    res['diff_dN1_minus_dN0'] = dict(value=float(a.mean() - b.mean()), ci=qci(bs))
    return res


# ------------------------------------------------------------------ main
def main(a):
    from interp.jedi_core import Jedi, sample
    sm = a.smoke; pc = sm or 2000; B = 200 if sm else 1000
    tag = 'e28_irc_split' + ('_smoke' if sm else '')
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1202)
    x, y, Q = sample(per_class=pc, seed=1202); x = np.asarray(x, np.float32)
    N_all = nreal(x)
    srt = bool(np.all(np.diff(x[:, :, 0], axis=1) <= 0)); padang = float(np.abs(x[:, :, 1:][x[:, :, 0] == 0]).max()) if (x[:, :, 0] == 0).any() else 0.0
    log(f'sample {len(x)} jets; pT-sorted {srt}; max |angle| in padding slots {padang}; N<=60: {(N_all <= 60).mean():.3f}')
    net = Jedi(); log('network loaded', net.path)
    keep = ('logits', 'S', 'g')
    R_all = net.run(x, keep=keep, batch=2000); c_all = R_all['logits'].argmax(1); acc_all = float((c_all == y).mean())
    log(f'unedited accuracy on all {len(x)} jets: {acc_all:.4f}')
    res = dict(N=int(len(x)), per_class=pc, seed=1202, n_boot=B, sorted=srt, padding_max_abs_angle=padang, acc_all=acc_all,
               live_S=LIVE_S, window_objects=WINDOW, windowless=WINDOWLESS, named=NAMED)
    res['scale_pairs'] = matched_pairs(x, y, c_all, rng)
    log('matched data pairs', res['scale_pairs'])

    sub = np.flatnonzero(N_all <= 60); X0 = x[sub]; ys = y[sub]; N0 = N_all[sub]; J = len(X0)
    R0 = {k: v[sub] for k, v in R_all.items()}; c0 = c_all[sub]; corr0 = c0 == ys
    sdS = R0['S'].std(0); res['n_sub'] = J; res['acc_sub'] = float(corr0.mean())
    res['class_counts_sub'] = np.bincount(ys, minlength=5).tolist()
    log(f'edit subset: {J} jets (N <= 60), accuracy {corr0.mean():.4f}')
    bt = Boot(J, B, 1203)
    Mo0 = moments(X0); T14_0 = top14(X0); SC0 = soft_count(X0)

    # ---------------- build the edits
    E = {}   # name -> (X_edit, applicable mask, meta)
    slots = {k: pick_rank(N0, k, rng) for k in RANKS}
    for k in RANKS:
        s = slots[k]; ok = s >= 0
        for z in ZS: E[f'split_r{k}_z{z}'] = (split(X0, s, z), ok, dict(kind='split', rank=k, z=z))
        Xs = split(X0, s, 0.5); E[f'splitfix_r{k}'] = (drop_softest(Xs, np.flatnonzero(ok)), ok, dict(kind='splitfix', rank=k))
        E[f'swap_r{k}'] = (swap(X0, s), ok, dict(kind='swap', rank=k))
    for n in range(1, 6):
        Xn, ok = soft_add(X0, n, rng); E[f'soft_add{n}'] = (Xn, ok, dict(kind='soft', n=n))
    E['pad0_at'] = (pad0(X0, False), np.ones(J, bool), dict(kind='null'))
    E['pad0_origin'] = (pad0(X0, True), np.ones(J, bool), dict(kind='null'))
    E['drop_softest'] = (drop_softest(X0), np.ones(J, bool), dict(kind='scale'))
    log('edits built:', list(E))

    perjet = dict(sub=sub, y=ys, c0=c0, N=N0)
    out = {}; chg = {}
    for name, (Xe, ok, meta) in E.items():
        Re = net.run(Xe, keep=keep, batch=2000); ce = Re['logits'].argmax(1); ch = (ce != c0) & ok; chg[name] = ch
        sel = np.flatnonzero(ok); Mo1 = moments(Xe)
        r = dict(meta, n=int(ok.sum()), changed=float(ch[sel].mean()), changed_ci=qci(bt.mean(ch, sel)),
                 acc_before=float(corr0[sel].mean()), acc_after=float((ce == ys)[sel].mean()),
                 mcnemar_correct=mcnemar(corr0[sel], (ce == ys)[sel]))
        F = np.zeros((5, 5), int); np.add.at(F, (c0[sel], ce[sel]), 1); r['flow'] = F.tolist()
        r['flow_qg'] = int(F[0, 1] + F[1, 0]); r['flow_WZ'] = int(F[2, 3] + F[3, 2])
        r['moment_change'] = {k: dict(max_abs_rel=float(np.max(np.abs(Mo1[k][sel] - Mo0[k][sel]) / np.maximum(Mo0[k][sel], 1e-9))),
                                      mean_rel=float(np.mean((Mo1[k][sel] - Mo0[k][sel]) / np.maximum(Mo0[k][sel], 1e-9))))
                              for k in ('M0', 'M1', 'M2', 'm')}
        r['d_top14_pT_GeV'] = float(np.mean(top14(Xe)[sel] - T14_0[sel])); r['d_softcount_rank13plus'] = float(np.mean(soft_count(Xe)[sel] - SC0[sel]))
        dS = (Re['S'] - R0['S'])[sel] / np.where(sdS > 0, sdS, 1)
        r['dS_sd'] = {str(u): dict(mean=float(dS[:, u].mean()), mean_abs=float(np.abs(dS[:, u]).mean()),
                                   ci=qci((bt.W[:, sel] @ dS[:, u].astype(np.float32)) / np.maximum(bt.W[:, sel].sum(1), 1)),
                                   frac_nonzero=float((np.abs(dS[:, u]) > 0).mean())) for u in LIVE_S}
        dg = np.abs(Re['g'] - R0['g'])[sel]; r['g_max_abs_change_mean'] = float(dg.max(1).mean())
        out[name] = r; perjet[f'ch_{name}'] = ch
        log(f'{name}: n {r["n"]} changed {r["changed"]:.4f} {r["changed_ci"]} acc {r["acc_before"]:.4f}->{r["acc_after"]:.4f} '
            f'qg {r["flow_qg"]} WZ {r["flow_WZ"]} dS0 {r["dS_sd"]["0"]["mean"]:+.3f} dS19 {r["dS_sd"]["19"]["mean"]:+.3f} '
            f'dS29 {r["dS_sd"]["29"]["mean"]:+.3f} dM2 {r["moment_change"]["M2"]["max_abs_rel"]:.2e}')
        if meta['kind'] in ('split', 'soft'):
            holds = {'S19': [19], 'S0': [0], 'S0+S19': [0, 19], 'window17': WINDOW, 'windowless_S29_S30': WINDOWLESS,
                     'all_live_S': LIVE_S, 'g_all': None}
            for u in LIVE_S: holds[f'S{u}_alone'] = [u]
            H = {}
            for hn, units in holds.items():
                if units is None: ov = {'g': (np.ones(64), R0['g'])}
                else:
                    m = np.zeros(64); m[units] = 1; ov = {'S': (m, R0['S'])}
                Lh = net.run_jetwise(Xe, ov, batch=2000)['logits']; chH = (Lh.argmax(1) != c0) & ok
                rb = bt.ratio_removed(ch, chH, sel); nE = int(ch[sel].sum()); nH = int(chH[sel].sum())
                FH = np.zeros((5, 5), int); np.add.at(FH, (c0[sel], Lh.argmax(1)[sel]), 1)
                H[hn] = dict(changed=float(chH[sel].mean()), removed=float(1 - nH / nE) if nE else None, removed_ci=qci(rb),
                             mcnemar_edit_vs_hold=mcnemar(ch[sel], chH[sel]), flow_qg=int(FH[0, 1] + FH[1, 0]), flow_WZ=int(FH[2, 3] + FH[3, 2]))
            us = [f'S{u}_alone' for u in LIVE_S]; ph = holm([H[k]['mcnemar_edit_vs_hold']['p'] for k in us])
            for k, p in zip(us, ph): H[k]['p_holm_19'] = p
            r['holds'] = H
            log(f'   holds: ' + ', '.join(f'{k} {v["removed"] if v["removed"] is None else round(v["removed"], 3)}' for k, v in H.items()
                                         if not k.endswith('_alone')))
            top = sorted(us, key=lambda k: -(H[k]['removed'] or 0))[:5]
            log('   top single units: ' + ', '.join(f'{k} {H[k]["removed"]:.3f}' for k in top))

    # ---------------- split vs matched-rank control
    comp = {}
    for k in RANKS:
        for z in ZS:
            a_, b_ = chg[f'split_r{k}_z{z}'], chg[f'swap_r{k}']; sel = np.flatnonzero(E[f'swap_r{k}'][1])
            d = a_.astype(np.float32) - b_.astype(np.float32)
            comp[f'split_r{k}_z{z}_vs_swap'] = dict(n=len(sel), split=float(a_[sel].mean()), swap=float(b_[sel].mean()),
                                                     diff=float(d[sel].mean()), diff_ci=qci(bt.mean(d, sel)),
                                                     mcnemar=mcnemar(a_[sel], b_[sel]), p_perm=perm_signflip(a_[sel], b_[sel]))
        a_, b_ = chg[f'splitfix_r{k}'], chg[f'split_r{k}_z0.5']; sel = np.flatnonzero(E[f'swap_r{k}'][1])
        d = a_.astype(np.float32) - b_.astype(np.float32)
        comp[f'splitfix_r{k}_vs_split_r{k}_z0.5'] = dict(n=len(sel), diff=float(d[sel].mean()), diff_ci=qci(bt.mean(d, sel)),
                                                       mcnemar=mcnemar(a_[sel], b_[sel]))
    keys = [k for k in comp if k.endswith('_vs_swap')]
    for k, p in zip(keys, holm([comp[k]['p_perm'] for k in keys])): comp[k]['p_perm_holm8'] = p
    for k, p in zip(keys, holm([comp[k]['mcnemar']['p'] for k in keys])): comp[k]['p_mcnemar_holm8'] = p
    ek = [k for k in out if out[k]['kind'] in ('split', 'soft', 'swap', 'splitfix')]
    for k, p in zip(ek, holm([out[k]['mcnemar_correct']['p'] for k in ek])): out[k]['mcnemar_correct']['p_holm'] = p
    # null: any edit vs nothing: exact McNemar of changed vs never-changed is the binomial test of #changed > 0 (H0: 0 changes)
    res['edits'] = out; res['split_vs_control'] = comp
    for k, v in comp.items(): log(k, {kk: vv for kk, vv in v.items() if kk != 'mcnemar'})

    # ---------------- example gluon jet for the figure
    ex = None
    for cand in ('split_r13_z0.5', 'split_r0_z0.5', 'split_r5_z0.5'):
        js = np.flatnonzero(chg[cand] & (ys == 0) & (c0 == 0) & (N0 <= 40))
        if not len(js): js = np.flatnonzero(chg[cand] & (ys == 0))
        if len(js):
            j = int(js[0]); Xe = E[cand][0][j:j + 1]; Re = net.run(Xe, keep=keep); pr = lambda L: (np.exp(L - L.max()) / np.exp(L - L.max()).sum()).tolist()
            ex = dict(edit=cand, row=j, sample_index=int(sub[j]), label='g', slot=int(slots[13 if '13' in cand else (0 if 'r0' in cand else 5)][j]),
                      before=X0[j, :N0[j]].tolist(), after=Xe[0, :N0[j] + 1].tolist(), call_before=CL[c0[j]], call_after=CL[int(Re['logits'][0].argmax())],
                      p_before=pr(R0['logits'][j]), p_after=pr(Re['logits'][0]),
                      S_before={str(u): float(R0['S'][j, u]) for u in NAMED}, S_after={str(u): float(Re['S'][0, u]) for u in NAMED},
                      sd_S={str(u): float(sdS[u]) for u in NAMED})
            break
    res['example'] = ex; log('example', None if ex is None else {k: ex[k] for k in ('edit', 'sample_index', 'call_before', 'call_after', 'S_before', 'S_after')})

    # ---------------- untrained network control (c)
    res['untrained'] = {}
    for s in (0, 1):
        un = Jedi(untrained=True, seed=s); L0u = un.run(X0, batch=2000)['logits']; off = L0u.mean(0); cu = L0u.argmax(1)
        cuc = (L0u - off).argmax(1)   # centred: logits minus their unedited mean (as X10 / E26), so that calls are not constant
        d = dict(acc=float((cu == ys).mean()), call_dist=np.bincount(cu, minlength=5).tolist(),
                 acc_centred=float((cuc == ys).mean()), call_dist_centred=np.bincount(cuc, minlength=5).tolist())
        for name in ('split_r0_z0.5', 'split_r5_z0.5', 'split_r13_z0.5', 'split_rsoft_z0.5', 'swap_r0', 'swap_r5', 'swap_r13', 'swap_rsoft',
                     'soft_add1', 'soft_add5', 'pad0_at', 'drop_softest'):
            Xe, ok, _ = E[name]; sel = np.flatnonzero(ok); Lu = un.run(Xe, batch=2000)['logits']
            ch = Lu.argmax(1) != cu; chc = (Lu - off).argmax(1) != cuc
            d[name] = dict(changed=float(ch[sel].mean()), changed_ci=qci(bt.mean(ch, sel)),
                           changed_centred=float(chc[sel].mean()), changed_centred_ci=qci(bt.mean(chc, sel)))
        res['untrained'][str(s)] = d; log(f'untrained {s}:', d)
        del un

    res['wall_s'] = time.time() - T0_
    (OUT / f'{tag}.json').write_text(json.dumps(res, indent=1))
    np.savez_compressed(OUT / f'{tag}_perjet.npz', **perjet)
    log(f'wrote {OUT / tag}.json; E28 DONE')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', type=int, default=0); main(ap.parse_args())
