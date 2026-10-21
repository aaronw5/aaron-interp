"""E26 (claims A1, A3, A4, A9; exploratory, links to loop D's D9 / H12): which stages, units and mass-direction weights of
JEDI-linear (3-feature n64) carry the call change under an exact Lorentz boost along the jet axis at fixed jet mass.

Sample: interp.jedi_core.sample(per_class=2000, seed=2601) = 10,000 jets (seed 2602 kept for the verifier).
Edits (imported from interp.h12_boost_panel, not re-implemented): B (exact boost, |P_jet| x lam, m fixed exactly, angles ~ 1/lam),
  P (pT x lam at fixed angles), A (angles x 1/lam at fixed pT), controls R (rotation about the stored axis, pT-weighted displacement
  matched to B per jet), T (random-direction translation of the same length); lam in {0.8, 0.9, 1.1, 1.25}.
  Added here: M0-only (pT x c, angles x c^-1/2: M0 x c, M2 fixed), M2-only (angles x sqrt(c): M2 x c, M0 fixed), Bc (B with every
  particle whose pT crosses the a32 ramp edge clamped 0.01 % on its unboosted side of the edge), D(k) (angle dilation x k, the
  matched real-mass calibration edit).
Step 1 (holds). The network sees the boosted jet; one stage or unit is held at its unboosted value:
  l1 unit k (per particle: the value the cut takes on the same particle before the boost, placed in the particle's boosted pT-rank
  slot; full forward via run_jetwise), g (all units, and each live unit), each live S, h4, h5 unit (exact fixed-point tail from that
  stage, interp.x10_das_jedi.JNet.exact). Named sets: the 4 pT-ramp l1 units, the 9 angular l1 units, all live l1 units (sanity:
  must remove ~100 %), S0 + S19, ramps + S0 + S19, ramps + S0 + S19 + g; X10 DAS M0 + M2 subspaces at S and h4 (Alternative 2).
  Null: random live-S-unit subsets of size 1-5 (500 draws each, every lam); random orthonormal S subspaces of dim 1-5 and h4 dims
  2 / 4 (500 each, lam 0.9 / 1.1); random 4-unit l1 subsets (+ 2 random S units: "mixed" null for the 6-object thesis set; lam 0.9).
  Metric: fraction of B's call changes removed = 1 - #changed(hold) / #changed(B) (paired McNemar, B vs hold), fraction of the mean
  W-jet shift of l_W - l_Z and of log p(W) / (p(g) + p(q)) removed; 95 % Poisson bootstrap over jets (1000, the same weights for every
  hold); p_rand = (1 + #null >= obs) / (1 + #null); Holm over all single units held (per lam and over all lam).
Step 2 (mass direction). Read-outs: standardised ridge of log m on h4 (E15) and on z (X10), fit on half A (unedited), evaluated on
  half B. Slopes s_E = d mean(dlog m_hat) / d log lam (through the origin, 4 lam) for B, P, A, M0-only, M2-only. Model
  log m_hat = a log M0 + b log M2: s_M0 = a, s_M2 = b, s_P = a + b, s_A = -2 b, s_B = a - b (first order). Scale-free
  kappa = (a - b) / (a + b) from (i) s_B / s_P, (ii) (s_M0 - s_M2) / (s_M0 + s_M2), (iii) (s_P + s_A) / s_P; equal weights -> 0.
  Matched calibration: for lam 0.9 / 1.1 the angle dilation D(k*) with log k* = dlog m_hat(B) (a real mass change of the size the
  read-out moved); calibrated B response = dlog m_hat(B) * log k* / dlog m_hat(D(k*)). X10 DAS subspaces (refit here with X10's
  exact data, pairs and seeds: F = sample(4000, 0), 8000 pairs, seed 10 + k) for M0, M2 at S (k 1, 2): patch only the subspace
  from B onto the unboosted state, z read-out.
Step 3: class flow of B, in-window split (boosted jet pT inside the 0.5-99.5 % range of unedited jet pT).
Controls: R, T, random holds, untrained n64 redraws seeds 0-2 (logits centred on their unedited mean, as X10).
Run: ./job.sh e26-run 1 2 'python -m interp.e26_boost_route --seed 2601'   (--smoke N: N jets per class)
Writes $AI_RESULTS/jedi/e26_boost_route[_s<seed>][_smoke].json and _perjet.npz.
"""
import argparse
import json
import re
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); OUT = RES / 'jedi'
T0_ = time.time()
LAMS = (0.8, 0.9, 1.1, 1.25); FULL_LAMS = (0.9, 1.1); W_, Z_ = 2, 3
CL = ['g', 'q', 'W', 'Z', 't']


def log(*a):
    print(f'[{time.time() - T0_:7.0f}s]', *a, flush=True)


from interp.h12_boost_panel import (boost, pscale, ascale, rotate, translate, displacement, jet_obs, sortT, hls_prep, copyT,  # noqa: E402
                                    logodds, mcnemar, holm, rel, set_rel)
from interp import x10_das_jedi as X  # noqa: E402


def reorder(T, o):
    tk = lambda a: np.take_along_axis(a, o, 1)
    U = {k: tk(T[k]) for k in ('pt', 'eta', 'phi', 'm')}; U['ext'] = {}; U['ax_eta'] = T['ax_eta']; U['ax_phi'] = T['ax_phi']; return U


def pair_inputs(T0, U):
    """boosted input (pT-sorted) and the unboosted particles placed in the same (boosted-order) slots"""
    o = np.argsort(-U['pt'], 1, kind='stable')
    return hls_prep(reorder(U, o)), hls_prep(reorder(T0, o)), o


def edit_m0(T, c):
    U = copyT(T); U['pt'] = T['pt'] * c; de, dp = rel(T); set_rel(U, de * c ** -0.5, dp * c ** -0.5); return U


def edit_m2(T, c):
    U = copyT(T); de, dp = rel(T); k = np.sqrt(c); set_rel(U, de * k, dp * k); return U


def clamp_pt(T0, U, thr):
    """B with each particle whose pT crosses thr under the boost clamped just on its unboosted side"""
    V = copyT(U); real = T0['pt'] > 0; b0 = T0['pt'] < thr; b1 = U['pt'] < thr; cross = real & (b0 != b1)
    V['pt'] = np.where(cross, np.where(b0, thr * (1 - 1e-4), thr * (1 + 1e-4)), U['pt'])
    return V, cross


class Net:
    def __init__(self, untrained=False, seed=0):
        self.jn = X.JNet(untrained=untrained, seed=seed); self.j = self.jn.net

    def fwd(self, x, ov=None, keep=('logits', 'g', 'S', 'h4', 'h5', 'z')):
        r = self.j.run_jetwise(x, ov, keep=keep, batch=2000) if ov else self.j.run(x, keep=keep, batch=2000)
        if 'logits' in r: r['logits'] = r['logits'] - self.jn.off
        return r

    def a_of(self, x, bs=1000):
        out = np.empty((len(x), x.shape[1], 64), np.float32)
        for i in range(0, len(x), bs): out[i:i + bs] = self.j.run(x[i:i + bs], keep=('a',), batch=bs)['a'].astype(np.float32)
        return out

    def tail(self, st, h):
        return self.jn.exact(st, h)   # (logits - off, z)


def wz(L): return L[:, W_] - L[:, Z_]
def wq(L): return logodds(L, W_, [0, 1])


def qci(v): return np.quantile(v, [.025, .975]).tolist()


class Ctx:
    """per-lam reference: unboosted logits L0 / calls c0, boosted LB / cB, bootstrap weights"""

    def __init__(self, L0, LB, y, Wb, inw):
        self.L0, self.LB, self.c0, self.cB = L0, LB, L0.argmax(1), LB.argmax(1); self.chB = self.cB != self.c0; self.Wb = Wb
        self.isW = y == W_; self.bB = Wb @ self.chB.astype(np.float32); self.inw = inw
        self.dWZ = (wz(LB) - wz(L0))[self.isW]; self.dWQ = (wq(LB) - wq(L0))[self.isW]; self.WbW = Wb[:, self.isW]
        self.nW = self.WbW.sum(1)

    def rem(self, LH):
        chH = LH.argmax(1) != self.c0; return 1 - chH.sum() / max(self.chB.sum(), 1)

    def stats(self, LH, full=True):
        cH = LH.argmax(1); chH = cH != self.c0; nB = int(self.chB.sum()); nH = int(chH.sum())
        r = dict(changed=float(chH.mean()), removed=1 - nH / nB if nB else None)
        if not full: return r
        rb = 1 - (self.Wb @ chH.astype(np.float32)) / np.maximum(self.bB, 1)
        r['removed_ci'] = qci(rb); r['mcnemar'] = mcnemar(self.chB, chH)
        dh = (wz(LH) - wz(self.L0))[self.isW]; dq = (wq(LH) - wq(self.L0))[self.isW]
        mB, mH = self.WbW @ self.dWZ / self.nW, self.WbW @ dh / self.nW; qB, qH = self.WbW @ self.dWQ / self.nW, self.WbW @ dq / self.nW
        r['dWZ'] = [float(self.dWZ.mean()), float(dh.mean())]; r['WZ_removed'] = float(1 - dh.mean() / self.dWZ.mean()); r['WZ_removed_ci'] = qci(1 - mH / mB)
        r['dWQ'] = [float(self.dWQ.mean()), float(dq.mean())]; r['WQ_removed'] = float(1 - dq.mean() / self.dWQ.mean()); r['WQ_removed_ci'] = qci(1 - qH / qB)
        iw = self.inw
        r['in_window'] = dict(n=int(iw.sum()), removed=float(1 - chH[iw].sum() / max(self.chB[iw].sum(), 1)))
        F = np.zeros((5, 5), int); np.add.at(F, (self.c0, cH), 1); r['flow'] = F.tolist()
        return r


def prand(null, obs):
    null = np.asarray(null, float); return float((1 + (null >= obs).sum()) / (1 + len(null)))


def main(a):
    from interp.jedi_core import sample
    from interp.h1_irc_panel import hls_table
    sm = a.smoke; pc = sm or 2000; seed = a.seed
    tag = f'e26_boost_route{"" if seed == 2601 else f"_s{seed}"}{"_smoke" if sm else ""}'
    OUT.mkdir(parents=True, exist_ok=True); JS = OUT / f'{tag}.json'
    NRS = 50 if sm else 500; NRQ = 50 if sm else 500; NRL = 4 if sm else 20; NB = 200 if sm else 1000
    R = dict(sample=f'interp.jedi_core.sample(per_class={pc}, seed={seed})', lams=list(LAMS), full_lams=list(FULL_LAMS))

    def save(): JS.write_text(json.dumps(X.clean(R), indent=1))
    x, y, _ = sample(pc, seed); J = len(y); T0 = hls_table(x); x0 = hls_prep(sortT(T0))
    R['N'] = J; R['labels'] = np.bincount(y, minlength=5).tolist()
    net = Net(); R0 = net.fwd(x0); Lraw = net.j.run(x, keep=('logits',))['logits']
    R['input_roundtrip_max_dlogit'] = float(np.abs(Lraw - R0['logits']).max())
    c0 = R0['logits'].argmax(1); R['acc'] = float((c0 == y).mean())
    R['tail_check'] = net.jn.check(x0[:1000], {k: v[:1000] for k, v in R0.items()})
    log(f'N {J}; accuracy {R["acc"]:.4f} (expected 0.806-0.813); round trip max dlogit {R["input_roundtrip_max_dlogit"]:.2e}; tail {R["tail_check"]}')
    O0 = jet_obs(T0); win = np.percentile(O0['jpt'], [0.5, 99.5]); R['window_GeV'] = win.tolist()
    rngB = np.random.default_rng(seed + 11); Wb = rngB.poisson(1.0, (NB, J)).astype(np.float32)

    # ---------------------------------------------------------------- l1 units: pT ramps and angular half-planes
    K1 = np.asarray(net.j.KB['l1'][0], np.float64); K1 = K1.reshape(-1, K1.shape[-1]); b1 = np.asarray(net.j.KB['l1'][1], np.float64).ravel()
    sh, sc = np.broadcast_to(np.ravel(net.j.shift), (3,)), np.broadcast_to(np.ravel(net.j.scale), (3,))
    nz = np.abs(K1) > 0; live1 = np.flatnonzero(nz.any(0)); ramps = [int(u) for u in live1 if nz[0, u] and not nz[1:, u].any()]
    angs = [int(u) for u in live1 if not nz[0, u] and nz[1:, u].any()]
    thr = {u: float(sh[0] - b1[u] * sc[0] / K1[0, u]) for u in ramps}
    R['l1'] = dict(live=live1.tolist(), ramps=ramps, angular=angs, other=[int(u) for u in live1 if u not in ramps + angs],
                   ramp_edge_GeV={str(u): thr[u] for u in ramps}, ramp_on_below={str(u): bool(K1[0, u] < 0) for u in ramps})
    a_s = net.a_of(x0[:500]); pts = x0[:500, :, 0]; real = pts > 0
    for u in ramps:
        on = a_s[..., u] > 0; mx = lambda v: float(v.max()) if v.size else None; mn = lambda v: float(v.min()) if v.size else None
        R['l1'].setdefault('ramp_edge_empirical', {})[str(u)] = dict(on_frac=float(on[real].mean()), max_pt_on=mx(pts[real & on]), min_pt_on=mn(pts[real & on]),
                                                                    max_pt_off=mx(pts[real & ~on]), min_pt_off=mn(pts[real & ~on]))
    del a_s
    log(f'l1: {R["l1"]}')
    edge_unit = 32 if 32 in ramps else min(ramps, key=lambda u: abs(thr[u] - 11))
    R['clamp_unit'] = edge_unit; thr_c = thr[edge_unit]

    # ---------------------------------------------------------------- X10 DAS subspaces (refit with X10's data, pairs, seeds)
    t = time.time(); pcF = 300 if sm else 4000; xF, yF, _ = sample(pcF, 0); mF = X.moments(xF); NF = len(yF)
    npair = 800 if sm else 8000; steps = 100 if sm else 800
    r2 = np.random.default_rng(2); fb = r2.integers(0, NF, npair); fs = r2.integers(0, NF, npair)
    while (fb == fs).any(): fs[fb == fs] = r2.integers(0, NF, (fb == fs).sum())
    RF = net.jn.reps(xF, keep=('S', 'h4', 'logits')); baseF = RF['logits'].argmax(1)[fb]
    labF = {V: net.jn.calls(X.do_edit(xF[fb], V, (mF[0][fb], mF[1][fb]), (mF[0][fs], mF[1][fs]))) for V in ('M0', 'M2')}
    Q = {}; R['das'] = {}
    for st in ('S', 'h4'):
        for k in (1, 2):
            for V in ('M0', 'M2'):
                q, info = X.das_fit(net.jn.tail(st), RF[st][fb], RF[st][fs], labF[V], baseF, k, seed=10 + k, steps=steps, rg=net.jn.rng_[st])
                Q[(st, V, k)] = q; R['das'][f'{st}_{V}_k{k}'] = info
            Q[(st, 'M0M2', k)] = np.linalg.qr(np.concatenate([Q[(st, 'M0', k)], Q[(st, 'M2', k)]], 1))[0]
    del xF, RF
    log(f'DAS refit ({time.time() - t:.0f} s): {R["das"]}'); save()

    # ---------------------------------------------------------------- read-outs (step 2)
    m0 = X.mass(x0); logm = np.log(np.maximum(m0, 1e-3)); pr = np.random.default_rng(seed + 3).permutation(J); hA, hB = pr[:J // 2], pr[J // 2:]
    nf = 2 * len(hA) // 3; RO = {st: X.RidgeR(R0[st], logm, hA[:nf], hA[nf:]) for st in ('h4', 'z')}
    R['readout'] = {st: dict(alpha=RO[st].alpha, r2_test=float(1 - ((RO[st](R0[st][hB]) - logm[hB]) ** 2).mean() / logm[hB].var())) for st in RO}
    mh0 = {st: RO[st](R0[st]) for st in RO}
    WbB = Wb[:, hB]; nbB = WbB.sum(1)
    DL = {}   # (edit, lam) -> {st: per-jet dlog m_hat on hB}

    def readouts(Rx, name, lam):
        DL[(name, lam)] = {st: (RO[st](Rx[st]) - mh0[st])[hB] for st in RO}
    log(f'read-outs: {R["readout"]}')

    # ---------------------------------------------------------------- live units
    UB9, _ = boost(T0, 0.9); xB9 = pair_inputs(T0, UB9)[0]; RB9 = net.fwd(xB9)
    live = {st: np.flatnonzero(np.maximum(R0[st].std(0), RB9[st].std(0)) > 1e-9).tolist() for st in ('g', 'S', 'h4', 'h5')}
    R['live'] = {k: dict(n=len(v), units=v) for k, v in live.items()}; del RB9
    log('live units: ' + ', '.join(f'{k} {len(v)}' for k, v in live.items()))
    named_S = [u for u in (0, 19) if u in live['S']]; R['thesis_S'] = named_S

    rng = np.random.default_rng(seed + 1000); rngN = np.random.default_rng(seed + 2000)
    R['lam'] = {}; NPZ = dict(y=y.astype(np.int8), c0=c0.astype(np.int8))
    for lam in LAMS:
        t = time.time(); E = dict(); R['lam'][str(lam)] = E
        U, info = boost(T0, lam); xB, x0B, o = pair_inputs(T0, U); O1 = jet_obs(U); inw = (O1['jpt'] >= win[0]) & (O1['jpt'] <= win[1])
        E['boost_check'] = dict(dm_rel_absmax=float(np.max(np.abs(O1['m'] / np.maximum(O0['m'], 1e-9) - 1))), djpt_rel_mean=float(np.mean(O1['jpt'] / O0['jpt'] - 1)),
                                dmA_rel_mean=float(np.mean(O1['mA'] / O0['mA'] - 1)), dsumpt_rel_mean=float(np.mean(O1['sumpt'] / O0['sumpt'] - 1)),
                                in_window_frac=float(inw.mean()), rank_order_changed_frac=float((o != np.arange(o.shape[1])).any(1).mean()))
        RB = net.fwd(xB); ctx = Ctx(R0['logits'], RB['logits'], y, Wb, inw); readouts(RB, 'B', lam)
        chB = ctx.chB; NPZ[f'cB_{lam}'] = ctx.cB.astype(np.int8)
        F = np.zeros((5, 5), int); np.add.at(F, (c0, ctx.cB), 1)
        E['B'] = dict(changed=float(chB.mean()), ci=qci((Wb @ chB.astype(np.float32)) / Wb.sum(1)), flow=F.tolist(),
                      in_window=dict(n=int(inw.sum()), changed=float(chB[inw].mean()), out_n=int((~inw).sum()), out_changed=float(chB[~inw].mean()) if (~inw).any() else None),
                      dWZ_W=float(ctx.dWZ.mean()), dWQ_W=float(ctx.dWQ.mean()), acc_after=float((ctx.cB == y).mean()))
        # controls and decomposition edits
        D = displacement(T0, U); E['controls'] = {}
        for nm, Ux in (('R', rotate(T0, D, rng)[0]), ('T', translate(T0, D, rng)[0])):
            ch = net.fwd(hls_prep(sortT(Ux)), keep=('logits',))['logits'].argmax(1) != c0
            E['controls'][nm] = dict(changed=float(ch.mean()), ci=qci((Wb @ ch.astype(np.float32)) / Wb.sum(1)), B_vs=mcnemar(chB, ch))
        for nm, Ux in (('P', pscale(T0, lam)[0]), ('A', ascale(T0, lam)[0]), ('M0', edit_m0(T0, lam)), ('M2', edit_m2(T0, lam))):
            Rx = net.fwd(hls_prep(sortT(Ux)), keep=('logits', 'h4', 'z')); readouts(Rx, nm, lam); ch = Rx['logits'].argmax(1) != c0
            E['controls'][nm] = dict(changed=float(ch.mean()), B_vs=mcnemar(chB, ch))
        # Alternative 3: B with the edge-crossing particles clamped
        Uc, cross = clamp_pt(T0, U, thr_c); Lc = net.fwd(hls_prep(sortT(Uc)), keep=('logits',))['logits']
        E['clamp'] = dict(unit=edge_unit, edge_GeV=thr_c, jets_with_crossing=float(cross.any(1).mean()), crossing_per_jet=float(cross.sum(1).mean()),
                          dm_rel_absmax=float(np.max(np.abs(jet_obs(Uc)['m'] / np.maximum(O0['m'], 1e-9) - 1))), **ctx.stats(Lc))
        log(f'lam {lam}: B changed {E["B"]["changed"]:.4f} (in window {E["B"]["in_window"]["changed"]:.4f}); R {E["controls"]["R"]["changed"]:.4f} '
            f'T {E["controls"]["T"]["changed"]:.4f} P {E["controls"]["P"]["changed"]:.4f} A {E["controls"]["A"]["changed"]:.4f}; '
            f'clamp removed {E["clamp"]["removed"]:.3f} (crossing jets {E["clamp"]["jets_with_crossing"]:.3f}); checks {E["boost_check"]}')
        # matched calibration (angle dilation of the size the read-out moved)
        if lam in FULL_LAMS:
            E['calibration'] = {}
            for st in ('h4', 'z'):
                dB = float(DL[('B', lam)][st].mean()); k_ = float(np.exp(dB if abs(dB) >= 0.005 else np.copysign(0.005, dB))); Rd = net.fwd(hls_prep(sortT(ascale(T0, 1 / k_)[0])), keep=(st,))
                dd = (RO[st](Rd[st]) - mh0[st])[hB]
                bt = [(WbB[i] @ DL[('B', lam)][st] / nbB[i]) * np.log(k_) / (WbB[i] @ dd / nbB[i]) for i in range(NB)]
                E['calibration'][st] = dict(dlogmhat_B=dB, k_star=k_, dlogmhat_dilation=float(dd.mean()), dlogm_true_dilation=float(np.log(k_)),
                                            gain=float(dd.mean() / np.log(k_)), calibrated_B=float(dB * np.log(k_) / dd.mean()), calibrated_B_ci=qci(bt))
        # ---- tail holds: every live S, h4, h5 unit
        E['holds'] = {}
        for st in ('S', 'h4', 'h5'):
            for u in live[st]:
                h = RB[st].copy(); h[:, u] = R0[st][:, u]; E['holds'][f'{st}{u}'] = ctx.stats(net.tail(st, h)[0])
        h = RB['S'].copy(); h[:, named_S] = R0['S'][:, named_S]; LS0 = net.tail('S', h)[0]; E['holds']['S0+S19'] = ctx.stats(LS0)
        for st in ('S', 'h4', 'h5'):
            E['holds'][f'{st}_all'] = ctx.stats(net.tail(st, R0[st])[0])
        # DAS subspaces (Alternative 2) and random subspaces
        E['das_holds'] = {}
        for st in ('S', 'h4'):
            lo_, hi_ = net.jn.rng_[st]; lo2, hi2 = np.minimum(np.minimum(lo_, RB[st]), R0[st]), np.maximum(np.maximum(hi_, RB[st]), R0[st])
            sub = lambda Qm: np.clip(RB[st] + ((R0[st] - RB[st]) @ Qm) @ Qm.T, lo2, hi2)
            for k in (1, 2):
                for V in ('M0', 'M2', 'M0M2'):
                    E['das_holds'][f'{st}_{V}_k{k}'] = ctx.stats(net.tail(st, sub(Q[(st, V, k)]))[0])
            if lam in FULL_LAMS:
                dims = (1, 2, 3, 4, 5) if st == 'S' else (2, 4); rq = np.random.default_rng(seed + 300 + (st == 'h4'))
                E.setdefault('null_subspace', {})[st] = {}
                for d in dims:
                    v = [ctx.stats(net.tail(st, sub(X.rand_Q(64, d, rq)))[0], full=False)['removed'] for _ in range(NRQ)]
                    E['null_subspace'][st][str(d)] = dict(q=np.quantile(v, [.025, .5, .975, 1]).tolist(), mean=float(np.mean(v)), vals=v)
        # random live-S subsets
        E['null_S'] = {}
        for d in (1, 2, 3, 4, 5):
            v = []
            for _ in range(NRS):
                us = rngN.choice(live['S'], d, replace=False); h = RB['S'].copy(); h[:, us] = R0['S'][:, us]; v.append(ctx.rem(net.tail('S', h)[0]))
            E['null_S'][str(d)] = dict(q=np.quantile(v, [.025, .5, .975, 1]).tolist(), mean=float(np.mean(v)), vals=v)
        log(f'lam {lam}: tail holds done; S0+S19 removed {E["holds"]["S0+S19"]["removed"]:.3f}; S_all {E["holds"]["S_all"]["removed"]:.3f}; '
            f'DAS S M0M2 k2 {E["das_holds"]["S_M0M2_k2"]["removed"]:.3f}; null S size 2 {E["null_S"]["2"]["q"]} ({time.time() - t:.0f} s)')
        # ---- full-forward holds (l1 per particle, g)
        if lam in FULL_LAMS:
            a0B = net.a_of(x0B)
            def ahold(units, keep=('logits',)):
                m = np.zeros(64); m[list(units)] = 1; return net.fwd(xB, {'a': (m, a0B)}, keep=keep)
            for u in live1:
                E['holds'][f'a{u}'] = ctx.stats(ahold([u])['logits'])
            Ra = ahold(ramps, keep=('logits', 'S')); E['holds']['a_ramps'] = ctx.stats(Ra['logits'])
            h = Ra['S'].copy(); h[:, named_S] = R0['S'][:, named_S]; E['holds']['ramps+S0+S19'] = ctx.stats(net.tail('S', h)[0])
            E['holds']['a_angular'] = ctx.stats(ahold(angs)['logits']); E['holds']['a_all_live'] = ctx.stats(ahold(live1)['logits'])
            E['holds']['g_all'] = ctx.stats(net.fwd(xB, {'g': (np.ones(64), R0['g'])}, keep=('logits',))['logits'])
            for u in live['g']:
                m = np.zeros(64); m[u] = 1; E['holds'][f'g{u}'] = ctx.stats(net.fwd(xB, {'g': (m, R0['g'])}, keep=('logits',))['logits'])
            m = np.zeros(64); m[ramps] = 1
            Rag = net.fwd(xB, {'a': (m, a0B), 'g': (np.ones(64), R0['g'])}, keep=('logits', 'S')); h = Rag['S'].copy(); h[:, named_S] = R0['S'][:, named_S]
            E['holds']['ramps+S0+S19+g'] = ctx.stats(net.tail('S', h)[0]); E['holds']['ramps+g'] = ctx.stats(Rag['logits'])
            if lam == 0.9:
                vl, vm = [], []
                for _ in range(NRL):
                    us = rngN.choice(live1, len(ramps), replace=False); Rr = ahold(us, keep=('logits', 'S')); vl.append(ctx.rem(Rr['logits']))
                    s2 = rngN.choice(live['S'], len(named_S), replace=False); h = Rr['S'].copy(); h[:, s2] = R0['S'][:, s2]; vm.append(ctx.rem(net.tail('S', h)[0]))
                E['null_l1'] = dict(size=len(ramps), vals=vl, q=np.quantile(vl, [.025, .5, .975, 1]).tolist())
                E['null_mixed'] = dict(size=len(ramps) + len(named_S), vals=vm, q=np.quantile(vm, [.025, .5, .975, 1]).tolist())
                NPZ['cB_hold_ramps+S0+S19_0.9'] = net.tail('S', np.where(np.isin(np.arange(64), named_S), R0['S'], Ra['S']))[0].argmax(1).astype(np.int8)
            del a0B
            top = sorted(((k, v['removed']) for k, v in E['holds'].items()), key=lambda kv: -kv[1])[:12]
            log(f'lam {lam}: full holds done; a_all_live {E["holds"]["a_all_live"]["removed"]:.3f}; ramps {E["holds"]["a_ramps"]["removed"]:.3f}; '
                f'ramps+S0+S19 {E["holds"]["ramps+S0+S19"]["removed"]:.3f}; g_all {E["holds"]["g_all"]["removed"]:.3f}; top {top} ({time.time() - t:.0f} s)')
        # DAS patches for the mass read-out (z): B's M0 / M2 subspace component only, onto the unboosted state at S
        E['das_patch_z'] = {}
        lo_, hi_ = net.jn.rng_['S']; lo2, hi2 = np.minimum(np.minimum(lo_, RB['S']), R0['S']), np.maximum(np.maximum(hi_, RB['S']), R0['S'])
        for k in (1, 2):
            for V in ('M0', 'M2', 'M0M2'):
                q = Q[('S', V, k)]; z = net.tail('S', np.clip(R0['S'] + ((RB['S'] - R0['S']) @ q) @ q.T, lo2, hi2))[1]
                dz = (RO['z'](z) - mh0['z'])[hB]; E['das_patch_z'][f'{V}_k{k}'] = dict(dlogmhat=float(dz.mean()), ci=qci(WbB @ dz / nbB),
                                                                                       frac_of_B=float(dz.mean() / DL[('B', lam)]['z'].mean()))
        save()

    # ---------------------------------------------------------------- statistics over holds
    S2 = R['stats'] = {}
    for lam in LAMS:
        E = R['lam'][str(lam)]; single = [k for k in E['holds'] if re.fullmatch(r'(a|g|S|h4|h5)\d+', k)]
        ps = [E['holds'][k]['mcnemar']['p'] for k in single]; hp = holm(ps)
        nullS1 = E['null_S']['1']['vals']; nullQ1 = E.get('null_subspace', {}).get('S', {}).get('1', {}).get('vals')
        rows = {}
        for k, h_ in zip(single, hp):
            v = E['holds'][k]; v['holm_mcnemar'] = h_
            v['p_rand_Sunit'] = prand(nullS1, v['removed'])
            if nullQ1 is not None: v['p_rand_Ssub1'] = prand(nullQ1, v['removed'])
            rows[k] = v['removed']
        if nullQ1 is not None:
            hq = holm([E['holds'][k]['p_rand_Ssub1'] for k in single])
            for k, h_ in zip(single, hq): E['holds'][k]['holm_rand_Ssub1'] = h_
        top = sorted(rows.items(), key=lambda kv: -kv[1])[:10]
        S2[str(lam)] = dict(n_single=len(single), n_holm_mcnemar_005=int(sum(h_ < 0.05 for h_ in hp)), top10=top,
                            n_holm_rand_005=int(sum(E['holds'][k].get('holm_rand_Ssub1', 1) < 0.05 for k in single)))
        # named sets vs size-matched nulls
        cmp = {}
        for k, d in (('S0+S19', 2), ('a_ramps', None), ('ramps+S0+S19', None)):
            if k not in E['holds']: continue
            obs = E['holds'][k]['removed']
            if k == 'S0+S19':
                cmp[k] = dict(removed=obs, p_rand_Ssubset=prand(E['null_S']['2']['vals'], obs),
                              p_rand_Ssub=prand(E['null_subspace']['S']['2']['vals'], obs) if 'null_subspace' in E else None)
            elif k == 'a_ramps' and 'null_l1' in E: cmp[k] = dict(removed=obs, p_rand_l1=prand(E['null_l1']['vals'], obs))
            elif k == 'ramps+S0+S19' and 'null_mixed' in E: cmp[k] = dict(removed=obs, p_rand_mixed=prand(E['null_mixed']['vals'], obs))
        for st, ks in (('S', ('S_M0M2_k1', 'S_M0M2_k2')), ('h4', ('h4_M0M2_k1', 'h4_M0M2_k2'))):
            for kk in ks:
                d = '2' if kk.endswith('k1') else '4'; obs = E['das_holds'][kk]['removed']
                nv = E.get('null_subspace', {}).get(st, {}).get(d)
                cmp[kk] = dict(removed=obs, p_rand_sub=prand(nv['vals'], obs) if nv else None)
        S2[str(lam)]['named_vs_null'] = cmp
        log(f'stats lam {lam}: {S2[str(lam)]}')
    allk = [(lam, k) for lam in LAMS for k, v in R['lam'][str(lam)]['holds'].items() if 'holm_mcnemar' in v]
    hp = holm([R['lam'][str(l)]['holds'][k]['mcnemar']['p'] for l, k in allk]); R['holm_all_lams'] = dict(n=len(allk), n_005=int(sum(h_ < .05 for h_ in hp)))

    # ---------------------------------------------------------------- step 2: dose slopes and kappa
    lg = np.log(np.asarray(LAMS)); S3 = R['mass_direction'] = {}
    for st in ('h4', 'z'):
        per = {E_: np.stack([DL[(E_, l)][st] for l in LAMS]) for E_ in ('B', 'P', 'A', 'M0', 'M2')}   # (4, nB)
        mean_l = {E_: (per[E_] @ WbB.T) / nbB for E_ in per}                                          # (4, NB)
        slope = lambda M: (lg @ M) / (lg @ lg)
        sb = {E_: slope(mean_l[E_]) for E_ in per}; s0 = {E_: float(slope(per[E_].mean(1))) for E_ in per}
        kap = dict(B_over_P=(s0['B'] / s0['P'], sb['B'] / sb['P']), M0_M2=((s0['M0'] - s0['M2']) / (s0['M0'] + s0['M2']), (sb['M0'] - sb['M2']) / (sb['M0'] + sb['M2'])),
                   P_A=((s0['P'] + s0['A']) / s0['P'], (sb['P'] + sb['A']) / sb['P']))
        S3[st] = dict(per_lam={E_: per[E_].mean(1).tolist() for E_ in per}, slope={E_: s0[E_] for E_ in per}, slope_ci={E_: qci(sb[E_]) for E_ in per},
                      a_b_from_moment_edits=dict(a=s0['M0'], b=s0['M2']), consistency_B_minus_PplusA=dict(v=s0['B'] - s0['P'] - s0['A'], ci=qci(sb['B'] - sb['P'] - sb['A'])),
                      kappa={k: dict(v=float(v[0]), ci=qci(v[1])) for k, v in kap.items()},
                      p_kappa_B_over_P_zero=float(min(1, 2 * min((kap['B_over_P'][1] <= 0).mean(), (kap['B_over_P'][1] >= 0).mean()) + 1 / NB)))
        log(f'mass direction {st}: slopes {S3[st]["slope"]}; kappa {S3[st]["kappa"]}; consistency {S3[st]["consistency_B_minus_PplusA"]}')
    save()

    # ---------------------------------------------------------------- untrained redraws (seeds 0-2)
    R['untrained'] = {}
    U9, _ = boost(T0, 0.9); xB9, x0B9, _ = pair_inputs(T0, U9)
    shp = net.jn.shp; del net
    for s in (0, 1, 2):
        un = Net(untrained=True, seed=s); un.jn.shp = shp; r0 = un.fwd(x0, keep=('logits', 'S')); un.jn.off = r0['logits'].mean(0); L0u = r0['logits'] - un.jn.off
        rb = un.fwd(xB9, keep=('logits', 'S')); cu = Ctx(L0u, rb['logits'], y, Wb, np.ones(J, bool))
        a0 = un.a_of(x0B9); m = np.zeros(64); m[ramps] = 1; ra = un.fwd(xB9, {'a': (m, a0)}, keep=('logits', 'S'))
        h = ra['S'].copy(); h[:, named_S] = r0['S'][:, named_S]; th_ = cu.stats(un.tail('S', h)[0], full=False)
        all_ = cu.stats(un.fwd(xB9, {'a': (np.ones(64), a0)}, keep=('logits',))['logits'], full=False)
        R['untrained'][str(s)] = dict(acc=float((L0u.argmax(1) == y).mean()), call_dist=np.bincount(L0u.argmax(1), minlength=5).tolist(),
                                      B09_changed=float(cu.chB.mean()), ramps_S0_S19_removed=th_['removed'], a_all_removed=all_['removed'],
                                      live_S=int((r0['S'].std(0) > 1e-9).sum()))
        log(f'untrained {s}: {R["untrained"][str(s)]}'); del un, a0
    save(); np.savez_compressed(OUT / f'{tag}_perjet.npz', **NPZ); log(f'wrote {JS}; E26 DONE')


def null1(a):
    """supplement (after the main run): larger random-subspace nulls at S (dim 1: --null1 draws; dims 2, 4: draws / 4) and the full
    enumeration of live-S-unit subsets of size 1-3, at lam 0.9 / 1.1, so that Holm over the 130 single-unit holds has p resolution
    below 0.05 / 130; reads the main json for the observed removals. Writes e26_boost_route[_s<seed>]_null1.json."""
    from itertools import combinations
    from interp.jedi_core import sample
    from interp.h1_irc_panel import hls_table
    seed = a.seed; base = f'e26_boost_route{"" if seed == 2601 else f"_s{seed}"}'; M = json.loads((OUT / f'{base}.json').read_text())
    x, y, _ = sample(2000, seed); J = len(y); T0 = hls_table(x); x0 = hls_prep(sortT(T0)); net = Net(); R0 = net.fwd(x0, keep=('logits', 'S'))
    net.jn.check(x0[:1000], {**net.fwd(x0[:1000]), 'logits': R0['logits'][:1000]})
    Wb = np.ones((1, J), np.float32); live = M['live']['S']['units']; R_ = dict(N=J, draws={'1': a.null1, '2': a.null1 // 4, '4': a.null1 // 4}, lam={})
    for lam in FULL_LAMS:
        t = time.time(); U, _ = boost(T0, lam); xB = pair_inputs(T0, U)[0]; RB = net.fwd(xB, keep=('logits', 'S'))
        ctx = Ctx(R0['logits'], RB['logits'], y, Wb, np.ones(J, bool)); E = M['lam'][str(lam)]
        assert abs(ctx.chB.mean() - E['B']['changed']) < 1e-9, 'B call change differs from the main run'
        lo_, hi_ = net.jn.rng_['S']; lo2, hi2 = np.minimum(np.minimum(lo_, RB['S']), R0['S']), np.maximum(np.maximum(hi_, RB['S']), R0['S'])
        rq = np.random.default_rng(seed + 900); nul = {}
        for d, n in R_['draws'].items():
            v = []
            for _ in range(n):
                Qm = X.rand_Q(64, int(d), rq); v.append(ctx.rem(net.tail('S', np.clip(RB['S'] + ((R0['S'] - RB['S']) @ Qm) @ Qm.T, lo2, hi2))[0]))
            nul[d] = np.asarray(v)
        enum = {}
        for d in (1, 2, 3):
            v = {}
            for us in combinations(live, d):
                h = RB['S'].copy(); h[:, list(us)] = R0['S'][:, list(us)]; v['+'.join(f'S{u}' for u in us)] = ctx.rem(net.tail('S', h)[0])
            enum[str(d)] = v
        single = [k for k in E['holds'] if re.fullmatch(r'(a|g|S|h4|h5)\d+', k)]
        p1 = [prand(nul['1'], E['holds'][k]['removed']) for k in single]; h1 = holm(p1)
        rows = {k: dict(removed=E['holds'][k]['removed'], p_rand_sub1=p, holm=h) for k, p, h in zip(single, p1, h1)}
        e2 = np.asarray(list(enum['2'].values())); obs2 = E['holds']['S0+S19']['removed']
        named = {'S0+S19': dict(removed=obs2, p_rand_sub2=prand(nul['2'], obs2), frac_S_pairs_ge=float((e2 >= obs2 - 1e-12).mean()), n_pairs=len(e2))}
        for kk, d in (('S_M0M2_k1', '2'), ('S_M0M2_k2', '4'), ('S_M0_k1', '1'), ('S_M2_k1', '1')):
            o_ = E['das_holds'][kk]['removed']; named[kk] = dict(removed=o_, p_rand=prand(nul[d], o_), null_max=float(nul[d].max()))
        best = {d: sorted(v.items(), key=lambda kv: -kv[1])[:8] for d, v in enum.items()}
        R_['lam'][str(lam)] = dict(null_q={d: np.quantile(v, [.025, .5, .975, .999, 1]).tolist() for d, v in nul.items()}, singles=rows,
                                   n_single=len(single), n_holm_005=int(sum(h < .05 for h in h1)), holm_005=[k for k, h in zip(single, h1) if h < .05],
                                   named=named, enum_best=best, enum_q={d: np.quantile(list(v.values()), [.5, .975, 1]).tolist() for d, v in enum.items()})
        log(f'null1 lam {lam}: null q {R_["lam"][str(lam)]["null_q"]}; Holm < 0.05: {R_["lam"][str(lam)]["holm_005"]}; named {named}; '
            f'best pairs {best["2"][:4]} ({time.time() - t:.0f} s)')
        (OUT / f'{base}_null1.json').write_text(json.dumps(X.clean(R_), indent=1))
    log('null1 DONE')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, default=2601); ap.add_argument('--smoke', type=int, default=0)
    ap.add_argument('--null1', type=int, default=0, help='supplement: N random 1-dim S subspaces (after the main run)')
    a_ = ap.parse_args()
    null1(a_) if a_.null1 else main(a_)
