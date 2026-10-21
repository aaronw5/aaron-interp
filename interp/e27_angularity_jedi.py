"""E27 (claims A4 / A3; exploratory): which radial observable is JEDI-linear's mass variable. Matched radial edits along the
angularity family lambda(kappa, beta) = sum_i pT_i^kappa r_i^beta, r_i = sqrt(detarel_i^2 + dphirel_i^2) about the stored axis.

Network: official JEDI-linear 3-feature n64 (interp.jedi_core.Jedi), and the untrained redraw (Jedi(untrained=True, seed=0)).
Sample: interp.jedi_core.sample(per_class=2000, seed=1201) = 10,000 jets, all 5 classes.
Family: kappa in {0 (count), 1 (pT-weighted), 2}, beta in {0.5, 1, 1.5, 2, 3}: 15 members. (kappa 1, beta 2) = M2 = sum pT r^2.
Edit (radial only): every real particle keeps its pT (so its pT rank and input slot) and its azimuth about the axis; its distance
  to the axis is multiplied by exp(delta_i): (detarel, dphirel) -> exp(delta_i) (detarel, dphirel). M0 = sum pT and the particle
  count are untouched. delta is solved per jet and per particle (a function of r alone cannot separate kappa 0 / 1 / 2 members,
  which differ only in pT weighting) as the minimum-norm Gauss-Newton solution (pseudo-inverse) of a set of constraints on
  log lambda_k and on the vector M1 = sum pT (deta, dphi) (2 rows scaled by |M1|; M1 held fixes |M1| and its direction).

Debug attempt 1 (2026-10-20, before any cluster run): the planned design (one member +5 %, the other 14 and M1 held within
  0.5 %) is infeasible: on 100 real W jets the minimum-norm solution needs max |delta| of 5-1700 (radius factors up to e^1700)
  for 14 of 15 members, and a Tikhonov-relaxed solve meets the 0.5 % tolerance with |delta| <= log 2 for 0 % of jets except the
  beta = 3 members. The power sums over a jet's 25-50 particles are nearly collinear in beta. Arm 0 records this on the full sample
  (diagnostic only). The measurement uses pairwise decoupling edits instead (arm 1):
Arm 1, per member a != (1, 2), on each jet:
  A_a  : lambda_a x 1.05 with M2 and M1 held (is the network's response carried by member a once M2 is fixed?)
  B_a  : M2 x 1.05 with lambda_a and M1 held (is the response to M2 still there once member a is fixed?)
  nA_a / nB_a (null edits): a random radial direction in the null space of the same 4 constraint rows (lambda_a, M2, M1 held),
         scaled so the angular displacement norm D = sqrt(sum_i (Delta r_i)^2) equals that of A_a / B_a on the same jet.
  M2h  : M2 x 1.05 with only M1 held (minimum-norm M2 edit), and its null nM2h.
  ref  : uniform dilation r -> sqrt(1.05) r (M2 x 1.05, M0 fixed; every beta > 0 member changes) = the pure M2 reference (c).
  A jet is kept for member a if A_a, B_a, nA_a, nB_a all meet: constrained members within 0.5 % of their targets, |M1| change
  <= 0.5 %, max |delta_i| <= log 2, and for the nulls D within 5 % of the matched edit's.
Dose: 5 % for every member; the kappa-1 members (the pT-weighted neighbours of M2, where A_a at 5 % meets the cap on only
  15-23 % of jets) are also run at 2 % (tolerance 0.2 %; ref and M2h at 2 %).
Metrics (trained and untrained, on member a's kept jets): calls changed (% of all jets, 5 classes); W->Z flow = share of jets
  called W before the edit that are called Z after; h4 / h5 log-m read-out (standardised ridge of log m, fit on half A of the
  jets: 2/3 fit, 1/3 alpha selection; evaluated on half B); R = mean Delta log m_hat (edit) / mean Delta log m_hat (ref) on the
  same jets. A 5 % change of the variable that is the network's mass partner of M0 predicts Delta log m = 0.5 log 1.05, the same
  as ref. Thesis (M2-like): R(A_a) ~ 0 and R(B_a) ~ 1 for every a. Member a carries it: R(A_a) ~ 1, R(B_a) ~ 0. Spread: both
  intermediate.
Statistics: 95 % bootstrap CIs over jets (2,000 Poisson resamples, ratio recomputed per resample); H0 per member and edit: response
  = matched-null response; read-out: paired sign-flip permutation (20,000) on per-jet (edit - null); calls and W->Z: McNemar
  (exact binomial) edit vs null; Holm over the 14 members per edit type and metric.
Outputs: $AI_RESULTS/jedi/e27_angularity.json, e27_angularity_perjet.npz.
  python -m interp.e27_angularity_jedi [--smoke 100] [--seed 1201]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); OUT = RES / 'jedi'
T0_ = time.time(); W_, Z_ = 2, 3
KAPS = (0, 1, 2); BETS = (0.5, 1.0, 1.5, 2.0, 3.0)
MEM = [(k, b) for k, b in ((k, b) for k in KAPS for b in BETS)]; NM = len(MEM); I12 = MEM.index((1, 2.0)); M1R = [NM, NM + 1]
EPS = np.log(1.05); TOL = 0.005; DMAX = np.log(2.0)
OTHERS = [i for i in range(NM) if i != I12]
DOSES = (('', np.log(1.05), OTHERS), ('2', np.log(1.02), [i for i in OTHERS if MEM[i][0] == 1]))


def log(*a):
    print(f'[{time.time() - T0_:7.0f}s]', *a, flush=True)


def mname(i): return f'k{MEM[i][0]}_b{MEM[i][1]:g}'


# ------------------------------------------------------------------------------------------------ family and solver
def fam(pt, r, real):
    """lambda (J, 15) and per-particle terms pT^kappa r^beta (J, 15, n)"""
    w = {0: real.astype(np.float64), 1: pt, 2: pt ** 2}
    T = np.stack([w[k] * r ** b for k, b in MEM], 1)
    return T.sum(2), T


def constraints(pt, de, dp, real, delta, ref):
    lam0, M10, nM1 = ref
    s = np.exp(np.clip(delta, -3, 3)); r = np.hypot(de, dp) * s
    lam, T = fam(pt, r, real)
    M1 = np.stack([(pt * de * s).sum(1), (pt * dp * s).sum(1)], 1)
    c = np.concatenate([np.log(lam / lam0), (M1 - M10) / nM1[:, None]], 1)
    bet = np.array([b for _, b in MEM])
    Jl = bet[None, :, None] * T / lam[:, :, None]                     # d log lambda_k / d delta_i
    Jm = np.stack([pt * de * s, pt * dp * s], 1) / nM1[:, None, None]  # d (M1 / |M1_0|) / d delta_i
    return c, np.concatenate([Jl, Jm], 1) * real[:, None, :]


def gn(pt, de, dp, real, rows, b, delta, ref, iters=12):
    """Gauss-Newton with minimum-norm (pseudo-inverse) steps: constraint rows -> b (J, len(rows)); returns delta, all 17 c"""
    delta = np.clip(np.nan_to_num(delta), -3, 3) * real
    for _ in range(iters):
        c, Jac = constraints(pt, de, dp, real, delta, ref)
        Jr = np.nan_to_num(Jac[:, rows]); bad = ~np.isfinite(Jr).all((1, 2)) | ~np.isfinite(c[:, rows]).all(1)
        Jr[bad] = 0
        step = np.einsum('jik,jk->ji', np.linalg.pinv(Jr, rcond=1e-12), np.nan_to_num(b - c[:, rows]))
        delta = np.clip(delta + np.nan_to_num(step), -3, 3) * real
    c, _ = constraints(pt, de, dp, real, delta, ref)
    return delta, c


def reference(pt, de, dp, real):
    r = np.hypot(de, dp); lam0, _ = fam(pt, r, real)
    M10 = np.stack([(pt * de).sum(1), (pt * dp).sum(1)], 1); nM1 = np.maximum(np.hypot(M10[:, 0], M10[:, 1]), 1e-9)
    return lam0, M10, nM1


def disp(de, dp, delta):
    return np.sqrt(((np.hypot(de, dp) * (np.exp(delta) - 1)) ** 2).sum(1))


def ok_mask(c, delta, rows, b, tol=TOL):
    """constrained lambda rows within TOL (relative) of target, |M1 - M1_0| / |M1_0| <= TOL, max |delta| <= log 2"""
    lr = [k for k in rows if k < NM]; bl = b[:, [rows.index(k) for k in lr]]
    rl = np.exp(c[:, lr] - bl) - 1; m1 = np.hypot(c[:, NM], c[:, NM + 1])
    return np.isfinite(c).all(1) & (np.abs(rl) <= tol).all(1) & (m1 <= tol) & (np.abs(delta).max(1) <= DMAX)


def apply(x, delta):
    e = np.array(x, np.float64); s = np.exp(delta)
    e[..., 1] *= s; e[..., 2] *= s
    return e.astype(np.float32)


def null_edit(pt, de, dp, real, rows, Dm, ref, rng, tol=TOL):
    """random radial direction in the null space of the rows at delta = 0, displacement norm matched to Dm, re-held by GN"""
    _, Jac0 = constraints(pt, de, dp, real, np.zeros_like(pt), ref); Jr = Jac0[:, rows]
    g = rng.standard_normal(pt.shape) * real * (np.hypot(de, dp) > 0)
    gN = g - np.einsum('jik,jk->ji', np.linalg.pinv(Jr, rcond=1e-12), np.einsum('jki,ji->jk', Jr, g))
    s = np.minimum(Dm / np.maximum(disp(de, dp, gN * 1e-3) / 1e-3, 1e-12), 1e3); b0 = np.zeros((len(pt), len(rows)))
    for _ in range(4):
        dn, cn = gn(pt, de, dp, real, rows, b0, gN * s[:, None], ref, iters=6)
        Dn = disp(de, dp, dn); s = np.minimum(s * np.where(Dn > 0, Dm / np.maximum(Dn, 1e-12), 1.0), 1e3)
    dn, cn = gn(pt, de, dp, real, rows, b0, gN * s[:, None], ref, iters=8)
    Dn = disp(de, dp, dn)
    ok = ok_mask(cn, dn, rows, b0, tol) & (np.abs(Dn / np.maximum(Dm, 1e-12) - 1) <= 0.05)
    return dn, cn, ok


# ------------------------------------------------------------------------------------------------ statistics
def boot_mean(Wb, v):
    return (Wb @ v) / np.maximum(Wb.sum(1), 1e-9)


def qci(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return np.quantile(v, [.025, .975]).tolist() if len(v) else [None, None]


def signflip_p(d, nperm=20000, seed=0):
    d = np.asarray(d, np.float64); n = len(d)
    if n == 0: return 1.0
    obs = abs(d.mean()); rng = np.random.default_rng(seed); cnt = 0; done = 0
    while done < nperm:
        k = min(1000, nperm - done); S = rng.choice((-1.0, 1.0), (k, n)); cnt += int((np.abs(S @ d / n) >= obs - 1e-15).sum()); done += k
    return (1 + cnt) / (1 + nperm)


class NetRun:
    def __init__(self, untrained):
        from interp.jedi_core import Jedi
        self.j = Jedi(64, untrained=untrained, seed=0); self.off = np.zeros(5); self.untrained = untrained

    def fwd(self, x):
        r = self.j.run(x, keep=('logits', 'h4', 'h5'), batch=2000); r['logits'] = r['logits'] - self.off; return r


def main(a):
    from interp.jedi_core import sample
    from interp.x10_das_jedi import clean, holm, mass, RidgeR
    from interp.h12_boost_panel import mcnemar
    sm = a.smoke; pc = sm or 2000; seed = a.seed; NB = 200 if sm else 2000; NP = 2000 if sm else 20000
    tag = f'e27_angularity{"" if seed == 1201 else f"_s{seed}"}{"_smoke" if sm else ""}'
    OUT.mkdir(parents=True, exist_ok=True); JS = OUT / f'{tag}.json'
    R = dict(sample=f'interp.jedi_core.sample(per_class={pc}, seed={seed})', members=[mname(i) for i in range(NM)],
             eps='+5 %', tol=TOL, delta_max=float(DMAX))

    def save(): JS.write_text(json.dumps(clean(R), indent=1))
    x, y, _ = sample(pc, seed); J = len(y); R['N'] = J; R['labels'] = np.bincount(y, minlength=5).tolist()
    x64 = np.asarray(x, np.float64); pt, de, dp = x64[..., 0], x64[..., 1], x64[..., 2]; real = pt > 0
    R['n_real'] = dict(mean=float(real.sum(1).mean()), min=int(real.sum(1).min()), q10=float(np.quantile(real.sum(1), .1)))
    ref = reference(pt, de, dp, real)

    # ---------------------------------------------------------------- networks, base pass, read-outs
    nets = {'trained': NetRun(False), 'untrained': NetRun(True)}
    base = {}
    for nm, net in nets.items():
        r = net.fwd(x)
        if net.untrained: net.off = r['logits'].mean(0); r['logits'] = r['logits'] - net.off
        base[nm] = r
    c0 = {nm: base[nm]['logits'].argmax(1) for nm in nets}
    R['acc'] = float((c0['trained'] == y).mean()); R['acc_untrained_centred'] = float((c0['untrained'] == y).mean())
    R['untrained_call_shares'] = np.bincount(c0['untrained'], minlength=5).tolist()
    log(f'N {J}; trained accuracy {R["acc"]:.4f} (expected 0.806-0.813); untrained centred {R["acc_untrained_centred"]:.4f}, calls {R["untrained_call_shares"]}')
    logm = np.log(np.maximum(mass(x), 1e-3)); pr = np.random.default_rng(seed + 3).permutation(J); hA, hB = np.sort(pr[:J // 2]), np.sort(pr[J // 2:])
    nf = 2 * len(hA) // 3; RO = {}; R['readout'] = {}
    for nm in nets:
        for st in ('h4', 'h5'):
            ro = RidgeR(base[nm][st], logm, hA[:nf], hA[nf:]); RO[(nm, st)] = ro
            R['readout'][f'{nm}_{st}'] = dict(alpha=ro.alpha, r2_test=float(1 - ((ro(base[nm][st][hB]) - logm[hB]) ** 2).mean() / logm[hB].var()))
    mh0 = {k: ro(base[k[0]][k[1]]) for k, ro in RO.items()}
    isB = np.zeros(J, bool); isB[hB] = True
    log(f'read-outs (held-out R2 of log m): {R["readout"]}'); save()

    # ---------------------------------------------------------------- arm 0: full-hold design (diagnostic only)
    t = time.time(); R['arm0_full_hold'] = {}; allrows = list(range(NM)) + M1R
    for i in range(NM):
        b = np.zeros((J, NM + 2)); b[:, i] = EPS
        dl, c = gn(pt, de, dp, real, allrows, b, np.zeros_like(pt), ref, iters=10)
        ok = ok_mask(c, dl, allrows, b)
        R['arm0_full_hold'][mname(i)] = dict(kept=int(ok.sum()), kept_frac=float(ok.mean()), median_max_abs_delta=float(np.median(np.abs(dl).max(1))))
    log(f'arm 0 (one member +5 %, 14 others + M1 held): kept per member {[v["kept"] for v in R["arm0_full_hold"].values()]} ({time.time() - t:.0f} s)'); save()

    # ---------------------------------------------------------------- arm 1: pairwise decoupling edits (dose 5 %; kappa-1 members also 2 %)
    t = time.time(); D = {}; KEEP = {}; CH = {}; DISP = {}; KS = {}
    for sfx, eps, mems in DOSES:
        tol = TOL * eps / EPS   # 0.5 % at the 5 % dose, 0.2 % at 2 %
        for i in mems:
            rows = [i, I12] + M1R
            for kind, tgt in (('A', i), ('B', I12)):
                b = np.zeros((J, 4)); b[:, rows.index(tgt)] = eps; k_ = (kind + sfx, i); kn = ('n' + kind + sfx, i)
                dl, c = gn(pt, de, dp, real, rows, b, np.zeros_like(pt), ref)
                D[k_] = dl; KEEP[k_] = ok_mask(c, dl, rows, b, tol); CH[k_] = np.exp(c[:, :NM]) - 1; DISP[k_] = disp(de, dp, dl)
                dn, cn, okn = null_edit(pt, de, dp, real, rows, DISP[k_], ref, np.random.default_rng(seed * 1000 + 2 * i + (kind == 'B') + 100 * len(sfx)), tol)
                D[kn] = dn; KEEP[kn] = okn; CH[kn] = np.exp(cn[:, :NM]) - 1; DISP[kn] = disp(de, dp, dn)
            KS[(sfx, i)] = KEEP[('A' + sfx, i)] & KEEP[('B' + sfx, i)] & KEEP[('nA' + sfx, i)] & KEEP[('nB' + sfx, i)]
            log(f'dose {eps:.4f} member {mname(i)}: kept A {KEEP[("A" + sfx, i)].sum()}, B {KEEP[("B" + sfx, i)].sum()}, nA {KEEP[("nA" + sfx, i)].sum()}, '
                f'nB {KEEP[("nB" + sfx, i)].sum()}; all four {KS[(sfx, i)].sum()}')
        rows = [I12] + M1R; b = np.zeros((J, 3)); b[:, 0] = eps; k_ = ('M2h' + sfx, 0); kn = ('nM2h' + sfx, 0)
        dl, c = gn(pt, de, dp, real, rows, b, np.zeros_like(pt), ref); D[k_] = dl; KEEP[k_] = ok_mask(c, dl, rows, b, tol); CH[k_] = np.exp(c[:, :NM]) - 1
        DISP[k_] = disp(de, dp, dl)
        dn, cn, okn = null_edit(pt, de, dp, real, rows, DISP[k_], ref, np.random.default_rng(seed * 1000 + 999 - len(sfx)), tol)
        D[kn] = dn; KEEP[kn] = okn; CH[kn] = np.exp(cn[:, :NM]) - 1; DISP[kn] = disp(de, dp, dn)
        kr = ('ref' + sfx, 0); D[kr] = np.full_like(pt, 0.5 * eps) * real; KEEP[kr] = np.ones(J, bool)
        c_ref, _ = constraints(pt, de, dp, real, D[kr], ref); CH[kr] = np.exp(c_ref[:, :NM]) - 1; DISP[kr] = disp(de, dp, D[kr])
        KS[(sfx, 'M2h')] = KEEP[k_] & KEEP[kn]; KS[(sfx, 'ref')] = np.ones(J, bool)
    kname = lambda k: f'{k[0] or "5"}_{mname(k[1]) if isinstance(k[1], int) else k[1]}'
    R['solver'] = dict(seconds=time.time() - t, kept={kname(k): int(v.sum()) for k, v in KS.items()},
                       kept_by_class={kname(k): np.bincount(y[v], minlength=5).tolist() for k, v in KS.items()})

    def kset(key):
        kind, i = key; sfx = '2' if kind.endswith('2') else ''; base_ = kind.rstrip('2').lstrip('n')
        return KS[(sfx, i)] if base_ in ('A', 'B') else KS[(sfx, 'M2h' if base_ == 'M2h' else 'ref')]

    def ename(key):
        return f'{key[0]}_{mname(key[1])}' if key[0].rstrip('2').lstrip('n') in ('A', 'B') else key[0]
    # what each edit does to all 15 members (median relative change on the kept jets), displacement sizes
    R['edit_member_changes'] = {}
    for key, ch in CH.items():
        ks = kset(key)
        if not ks.any(): continue
        R['edit_member_changes'][ename(key)] = dict(median_change={mname(j): float(np.median(ch[ks, j])) for j in range(NM)},
                                                    D_rad_median=float(np.median(DISP[key][ks])), max_abs_delta_median=float(np.median(np.abs(D[key][ks]).max(1))))
    log(f'solver ({R["solver"]["seconds"]:.0f} s): kept {R["solver"]["kept"]}'); save()

    # ---------------------------------------------------------------- forward passes
    PJ = {}
    for key in D:
        xe = apply(x, D[key])
        for nm, net in nets.items():
            r = net.fwd(xe); ce = r['logits'].argmax(1)
            PJ[(nm, key)] = dict(ch=ce != c0[nm], wz=(c0[nm] == W_) & (ce == Z_), call=ce.astype(np.int8),
                                 **{st: RO[(nm, st)](r[st]) - mh0[(nm, st)] for st in ('h4', 'h5')})
        p = PJ[('trained', key)]; ks = kset(key)
        log(f'forward {ename(key)}: trained calls changed {p["ch"][ks].mean():.4f}, W->Z {p["wz"][ks].sum()} of {(c0["trained"][ks] == W_).sum()}, '
            f'dlogm h4 {p["h4"][ks & isB].mean():+.4f} h5 {p["h5"][ks & isB].mean():+.4f}')

    # ---------------------------------------------------------------- metrics
    Wb = np.random.default_rng(seed + 11).poisson(1.0, (NB, J)).astype(np.float32)

    def metrics(nm, key, ks, refkey):
        P = PJ[(nm, key)]; Pr = PJ[(nm, refkey)]; isW = ks & (c0[nm] == W_); kb = ks & isB
        o = dict(n=int(ks.sum()), n_W_called=int(isW.sum()), n_readout=int(kb.sum()),
                 calls_changed=float(P['ch'][ks].mean()), calls_changed_ci=qci(boot_mean(Wb[:, ks], P['ch'][ks].astype(float))),
                 wz=float(P['wz'][isW].mean()) if isW.any() else None, wz_ci=qci(boot_mean(Wb[:, isW], P['wz'][isW].astype(float))) if isW.any() else None,
                 wz_over_ref=float(P['wz'][isW].mean() / Pr['wz'][isW].mean()) if isW.any() and Pr['wz'][isW].mean() > 0 else None)
        for st in ('h4', 'h5'):
            v, vr = P[st][kb], Pr[st][kb]; bm, br = boot_mean(Wb[:, kb], v), boot_mean(Wb[:, kb], vr)
            o[f'dlogm_{st}'] = float(v.mean()); o[f'dlogm_{st}_ci'] = qci(bm)
            o[f'R_{st}'] = float(v.mean() / vr.mean()); o[f'R_{st}_ci'] = qci(bm / br)
        return o

    def test(nm, key, nkey, ks, refkey, pseed):
        P, Q = PJ[(nm, key)], PJ[(nm, nkey)]; isW = ks & (c0[nm] == W_); kb = ks & isB; Pr = PJ[(nm, refkey)]
        o = dict(mcnemar_calls=mcnemar(P['ch'][ks], Q['ch'][ks]), mcnemar_wz=mcnemar(P['wz'][isW], Q['wz'][isW]))
        for st in ('h4', 'h5'):
            d = P[st][kb] - Q[st][kb]
            o[f'perm_p_{st}'] = signflip_p(d, NP, seed=pseed + (st == 'h5'))
            o[f'diff_R_{st}'] = float(d.mean() / Pr[st][kb].mean()); o[f'diff_R_{st}_ci'] = qci(boot_mean(Wb[:, kb], d) / boot_mean(Wb[:, kb], Pr[st][kb]))
        return o

    R['results'] = {}
    for nm in nets:
        R['results'][nm] = {}
        for sfx, eps, mems in DOSES:
            rk = ('ref' + sfx, 0); km = KS[(sfx, 'M2h')]
            out = dict(eps=eps, ref=metrics(nm, rk, KS[(sfx, 'ref')], rk), M2h=metrics(nm, ('M2h' + sfx, 0), km, rk),
                       nM2h=metrics(nm, ('nM2h' + sfx, 0), km, rk), M2h_vs_null=test(nm, ('M2h' + sfx, 0), ('nM2h' + sfx, 0), km, rk, 7000 + len(sfx)),
                       members={})
            for i in mems:
                ks = KS[(sfx, i)]; e = dict(kappa=MEM[i][0], beta=MEM[i][1])
                for j, kind in enumerate(('A', 'B')):
                    e[kind] = metrics(nm, (kind + sfx, i), ks, rk); e['n' + kind] = metrics(nm, ('n' + kind + sfx, i), ks, rk)
                    e[f'{kind}_vs_null'] = test(nm, (kind + sfx, i), ('n' + kind + sfx, i), ks, rk, 100 * i + 10 * j + 1000 * len(sfx))
                e['ref_same_jets'] = metrics(nm, rk, ks, rk); e['M2h_same_jets'] = metrics(nm, ('M2h' + sfx, 0), ks & km, rk)
                out['members'][mname(i)] = e
            for kind in ('A', 'B'):
                for f, g in (('mcnemar_calls', lambda t_: t_['mcnemar_calls']['p']), ('mcnemar_wz', lambda t_: t_['mcnemar_wz']['p']),
                             ('perm_p_h4', lambda t_: t_['perm_p_h4']), ('perm_p_h5', lambda t_: t_['perm_p_h5'])):
                    adj = holm([g(out['members'][mname(i)][f'{kind}_vs_null']) for i in mems])
                    for i, p in zip(mems, adj): out['members'][mname(i)][f'{kind}_vs_null'][f'holm_{f}'] = float(p)
            R['results'][nm][sfx or '5'] = out
            log(f'{nm} dose {eps:.4f}: ref calls {out["ref"]["calls_changed"]:.3f}, W->Z {out["ref"]["wz"]}, dlogm h4 {out["ref"]["dlogm_h4"]:+.4f}; '
                f'M2h R_h4 {out["M2h"]["R_h4"]:+.2f} W->Z {out["M2h"]["wz"]} (null R {out["nM2h"]["R_h4"]:+.2f} W->Z {out["nM2h"]["wz"]})')
            for i in mems:
                e = out['members'][mname(i)]
                log(f'  {nm} {mname(i)} (n {e["A"]["n"]}): A R_h4 {e["A"]["R_h4"]:+.2f} (null {e["nA"]["R_h4"]:+.2f}) W->Z {e["A"]["wz"]} (null {e["nA"]["wz"]}) | '
                    f'B R_h4 {e["B"]["R_h4"]:+.2f} (null {e["nB"]["R_h4"]:+.2f}) W->Z {e["B"]["wz"]} (null {e["nB"]["wz"]})')
            save()

    # ---------------------------------------------------------------- per-jet arrays and example jet (M2h edit on a W jet)
    pM = PJ[('trained', ('M2h', 0))]; kM = KS[('', 'M2h')]
    cand = np.flatnonzero(kM & (y == W_) & (c0['trained'] == W_) & (pM['call'] == Z_))
    if len(cand) == 0: cand = np.flatnonzero(kM & (y == W_) & (c0['trained'] == W_))
    if len(cand) == 0: cand = np.flatnonzero(kM)
    ex = int(cand[np.argsort(real[cand].sum(1))[len(cand) // 2]])
    R['example'] = dict(index=ex, label=int(y[ex]), call_before=int(c0['trained'][ex]), call_after=int(pM['call'][ex]),
                        mass_GeV=float(np.exp(logm[ex])), n_real=int(real[ex].sum()), lambda_change={mname(j): float(CH[('M2h', 0)][ex, j]) for j in range(NM)},
                        dlogm_h4=float(pM['h4'][ex]), D_rad=float(DISP[('M2h', 0)][ex]))
    arrs = dict(y=y, c0=c0['trained'], isB=isB, x_ex=x[ex], x_ex_M2h=apply(x[ex:ex + 1], D[('M2h', 0)][ex:ex + 1])[0])
    for k_, v in KS.items(): arrs[f'keep_{kname(k_)}'] = v
    for (nm, key), P in PJ.items():
        tg = f'{nm}_{ename(key)}'
        for f in ('h4', 'h5', 'call'): arrs[f'{tg}_{f}'] = P[f].astype(np.float32 if f != 'call' else np.int8)
    np.savez_compressed(OUT / f'{tag}_perjet.npz', **arrs)
    R['seconds'] = time.time() - T0_; save(); log(f'done; wrote {JS}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', type=int, default=0); ap.add_argument('--seed', type=int, default=1201)
    main(ap.parse_args())
