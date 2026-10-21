"""E26 verifier (fresh sample seed 2602): re-measures the key E26 holds and adds the controls E26 lacks.

1. Replication on interp.jedi_core.sample(2000, 2602): B call change and the removed share for S0, S29, S0+S29, S0+S19, g49, g (all),
   a15, 4 pT ramps, a32, all live l1 cuts, Bc (11 GeV crossings clamped), the whole S stage, and X10 DAS M0 / M2 / M0+M2 directions
   at S (refit with X10's data, pairs, seeds as in E26), at lam 0.9 and 1.1.
2. Null adequacy. E26's random-subspace null draws orthonormal directions in all 64 S coordinates although only the live S units
   vary; this verifier compares (a) random directions in all 64 coordinates (E26's null), (b) random directions restricted to the
   live S coordinates, (c) the top-k principal directions of the unedited S distribution (half A, centred; a data-variance subspace
   with no boost or mass information), (d) the top-k right singular vectors of the boost difference S(B) - S(0) (half A; an
   oracle bound for a k-dim linear hold). All evaluated on half B (5,000 jets). Also: share of a random 64-dim projection that
   lands on dead S coordinates, and the d4 weight norm from dead S coordinates.
3. Mass read-out at h4 (E15 ridge, fit on half A unedited, read on half B): per-jet distribution of dlog m_hat under B (not only
   the mean), its association with the call change, and a causal check: hold / patch the 1-dim read-out direction in h4 and count
   the call changes removed / produced, vs random 1-dim live-h4 directions.
Run: ./job.sh e26-verify 'python -m interp.e26_verify --seed 2602'
Writes $AI_RESULTS/jedi/e26_verify.json
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
T0_ = time.time()


def log(*a):
    print(f'[{time.time() - T0_:7.0f}s]', *a, flush=True)


from interp.e26_boost_route import Net, Ctx, pair_inputs, clamp_pt, qci, prand  # noqa: E402
from interp.h12_boost_panel import boost, pscale, hls_prep, sortT  # noqa: E402
from interp import x10_das_jedi as X  # noqa: E402


def main(a):
    from interp.jedi_core import sample
    from interp.h1_irc_panel import hls_table
    sm = a.smoke; pc = sm or 2000; seed = a.seed; NB = 200 if sm else 1000
    ND1 = 100 if sm else 2000; NDk = 50 if sm else 500; NH = 50 if sm else 300
    OUT.mkdir(parents=True, exist_ok=True); JS = OUT / f'e26_verify{"_smoke" if sm else ""}.json'
    R = dict(sample=f'interp.jedi_core.sample(per_class={pc}, seed={seed})')

    def save(): JS.write_text(json.dumps(X.clean(R), indent=1))
    x, y, _ = sample(pc, seed); J = len(y); T0 = hls_table(x); x0 = hls_prep(sortT(T0))
    net = Net(); R0 = net.fwd(x0); c0 = R0['logits'].argmax(1)
    R['N'] = J; R['acc'] = float((c0 == y).mean())
    R['tail_check'] = net.jn.check(x0[:1000], {k: v[:1000] for k, v in R0.items()})
    log(f'N {J} acc {R["acc"]:.4f} tail {R["tail_check"]}')
    Wb = np.random.default_rng(seed + 11).poisson(1.0, (NB, J)).astype(np.float32)
    pr = np.random.default_rng(seed + 3).permutation(J); hA, hB = np.sort(pr[:J // 2]), np.sort(pr[J // 2:])

    # l1 live units
    K1 = np.asarray(net.j.KB['l1'][0], np.float64); K1 = K1.reshape(-1, K1.shape[-1]); b1 = np.asarray(net.j.KB['l1'][1], np.float64).ravel()
    sh, sc = np.broadcast_to(np.ravel(net.j.shift), (3,)), np.broadcast_to(np.ravel(net.j.scale), (3,))
    nz = np.abs(K1) > 0; live1 = np.flatnonzero(nz.any(0)); ramps = [int(u) for u in live1 if nz[0, u] and not nz[1:, u].any()]
    thr = {u: float(sh[0] - b1[u] * sc[0] / K1[0, u]) for u in ramps}; thr_c = thr[32]
    R['l1'] = dict(live=live1.tolist(), ramps=ramps, edge32=thr_c)

    # DAS refit at S (as E26: X10 data, pairs, seeds)
    t = time.time(); pcF = 300 if sm else 4000; xF, yF, _ = sample(pcF, 0); mF = X.moments(xF); NF = len(yF)
    npair = 800 if sm else 8000; steps = 100 if sm else 800
    r2 = np.random.default_rng(2); fb = r2.integers(0, NF, npair); fs = r2.integers(0, NF, npair)
    while (fb == fs).any(): fs[fb == fs] = r2.integers(0, NF, (fb == fs).sum())
    RF = net.jn.reps(xF, keep=('S', 'logits')); baseF = RF['logits'].argmax(1)[fb]
    labF = {V: net.jn.calls(X.do_edit(xF[fb], V, (mF[0][fb], mF[1][fb]), (mF[0][fs], mF[1][fs]))) for V in ('M0', 'M2')}
    Q = {}; R['das'] = {}
    for k in (1, 2):
        for V in ('M0', 'M2'):
            q, info = X.das_fit(net.jn.tail('S'), RF['S'][fb], RF['S'][fs], labF[V], baseF, k, seed=10 + k, steps=steps, rg=net.jn.rng_['S'])
            Q[(V, k)] = q; R['das'][f'S_{V}_k{k}'] = info
        Q[('M0M2', k)] = np.linalg.qr(np.concatenate([Q[('M0', k)], Q[('M2', k)]], 1))[0]
    del xF, RF
    log(f'DAS ({time.time() - t:.0f} s): {R["das"]}'); save()

    # h4 read-out (E15 ridge), fit on half A
    m0 = X.mass(x0); logm = np.log(np.maximum(m0, 1e-3)); nf = 2 * len(hA) // 3
    RO = X.RidgeR(R0['h4'], logm, hA[:nf], hA[nf:]); mh0 = RO(R0['h4'])
    R['readout_h4'] = dict(alpha=RO.alpha, r2_test=float(1 - ((mh0[hB] - logm[hB]) ** 2).mean() / logm[hB].var()),
                           resid_sd=float((mh0[hB] - logm[hB]).std()))
    wr = np.where(np.isfinite(RO.sd), RO.w / RO.sd, 0.0); qro = (wr / np.linalg.norm(wr))[:, None]

    d4K = np.asarray(net.j.KB['d4'][0], np.float64); d4K = d4K.reshape(-1, d4K.shape[-1])
    R['lam'] = {}
    for lam in (0.9, 1.1):
        t = time.time(); E = {}; R['lam'][str(lam)] = E
        U, _ = boost(T0, lam); xB, x0B, o = pair_inputs(T0, U); RB = net.fwd(xB)
        ctx = Ctx(R0['logits'], RB['logits'], y, Wb, np.ones(J, bool)); chB = ctx.chB
        E['B'] = dict(changed=float(chB.mean()), ci=qci((Wb @ chB.astype(np.float32)) / Wb.sum(1)))
        liveS = np.flatnonzero(np.maximum(R0['S'].std(0), RB['S'].std(0)) > 1e-9); deadS = np.setdiff1d(np.arange(64), liveS)
        liveH = np.flatnonzero(np.maximum(R0['h4'].std(0), RB['h4'].std(0)) > 1e-9)
        E['liveS'] = liveS.tolist(); E['n_live_h4'] = int(len(liveH))
        H = E['holds'] = {}

        def Shold(units):
            h = RB['S'].copy(); h[:, units] = R0['S'][:, units]; return net.tail('S', h)[0]
        for nm, us in (('S0', [0]), ('S29', [29]), ('S19', [19]), ('S0+S29', [0, 29]), ('S0+S19', [0, 19])):
            if all(u in liveS for u in us): H[nm] = ctx.stats(Shold(us))
        H['S_all'] = ctx.stats(net.tail('S', R0['S'])[0])
        lo_, hi_ = net.jn.rng_['S']; lo2, hi2 = np.minimum(np.minimum(lo_, RB['S']), R0['S']), np.maximum(np.maximum(hi_, RB['S']), R0['S'])
        sub = lambda Qm, rows=slice(None): net.tail('S', np.clip(RB['S'][rows] + ((R0['S'][rows] - RB['S'][rows]) @ Qm) @ Qm.T, lo2[rows], hi2[rows]))[0]
        for (V, k), q in Q.items(): H[f'DAS_S_{V}_k{k}'] = ctx.stats(sub(q))
        a0B = net.a_of(x0B)
        def ahold(units):
            m = np.zeros(64); m[list(units)] = 1; return net.fwd(xB, {'a': (m, a0B)}, keep=('logits',))['logits']
        H['a15'] = ctx.stats(ahold([15])); H['a32'] = ctx.stats(ahold([32])); H['a_ramps'] = ctx.stats(ahold(ramps)); H['a_all_live'] = ctx.stats(ahold(live1))
        del a0B
        H['g_all'] = ctx.stats(net.fwd(xB, {'g': (np.ones(64), R0['g'])}, keep=('logits',))['logits'])
        m = np.zeros(64); m[49] = 1; H['g49'] = ctx.stats(net.fwd(xB, {'g': (m, R0['g'])}, keep=('logits',))['logits'])
        Uc, cross = clamp_pt(T0, U, thr_c); H['clamp'] = ctx.stats(net.fwd(hls_prep(sortT(Uc)), keep=('logits',))['logits'])
        log(f'lam {lam}: B {E["B"]}; ' + '; '.join(f'{k} {v["removed"]:.3f} {v["removed_ci"]}' for k, v in H.items()) + f' ({time.time() - t:.0f} s)')
        save()

        # ---- null adequacy on half B
        cB = Ctx(R0['logits'][hB], RB['logits'][hB], y[hB], Wb[:, hB], np.ones(len(hB), bool)); N_ = E['null_halfB'] = {}
        N_['B_changed'] = float(cB.chB.mean()); N_['n'] = int(len(hB))
        subB = lambda Qm: cB.rem(sub(Qm, hB))
        N_['DAS'] = {f'{V}_k{k}': subB(q) for (V, k), q in Q.items()}
        N_['S_units'] = {nm: cB.rem(Shold(us)[hB]) for nm, us in (('S0', [0]), ('S0+S29', [0, 29]))}
        rq = np.random.default_rng(seed + 500)
        def rq_live(d):
            Qm = np.zeros((64, d)); Qm[liveS] = X.rand_Q(len(liveS), d, rq); return Qm
        dsel = (1, 2, 4)
        for nm, gen in (('rand64', lambda d: X.rand_Q(64, d, rq)), ('rand_live', rq_live)):
            N_[nm] = {}
            for d in dsel:
                v = np.asarray([subB(gen(d)) for _ in range(ND1 if d == 1 else NDk)])
                N_[nm][str(d)] = dict(n=len(v), q=np.quantile(v, [.025, .5, .975, .999, 1]).tolist(), mean=float(v.mean()), vals=v.tolist())
        # dead-coordinate leakage of a random 64-dim projection
        D = R0['S'][hB] - RB['S'][hB]; fr = []
        for _ in range(200):
            Qm = X.rand_Q(64, 2, rq); pj = (D @ Qm) @ Qm.T; fr.append(float((pj[:, deadS] ** 2).sum() / max((pj ** 2).sum(), 1e-30)))
        N_['rand64_dim2_energy_on_dead'] = float(np.mean(fr))
        N_['d4_rownorm'] = dict(live_mean=float(np.linalg.norm(d4K[liveS], axis=1).mean()), dead_mean=float(np.linalg.norm(d4K[deadS], axis=1).mean()) if len(deadS) else None,
                                dead_max=float(np.linalg.norm(d4K[deadS], axis=1).max()) if len(deadS) else None)
        # PCA controls (fit on half A)
        Sa = R0['S'][hA]; _, _, Vt = np.linalg.svd(Sa - Sa.mean(0), full_matrices=False)
        Da = RB['S'][hA] - R0['S'][hA]; _, _, Vd = np.linalg.svd(Da, full_matrices=False)
        N_['pca_S'] = {str(d): subB(Vt[:d].T) for d in dsel}; N_['svd_boost_diff'] = {str(d): subB(Vd[:d].T) for d in dsel}
        # overlap of DAS M0M2 k1 with the PCA / boost-diff subspaces (sum of squared cosines / dim)
        ov = lambda A_, B_: float(np.linalg.norm(A_.T @ B_) ** 2 / A_.shape[1])
        N_['overlap_DAS_M0M2_k1'] = dict(pca_S_2=ov(Q[('M0M2', 1)], Vt[:2].T), svd_boost_diff_2=ov(Q[('M0M2', 1)], Vd[:2].T),
                                         S0_S29_axes=ov(Q[('M0M2', 1)], np.eye(64)[:, [0, 29]]), live_frac=ov(Q[('M0M2', 1)], np.eye(64)[:, liveS]))
        DIM = {'M0_k1': 1, 'M2_k1': 1, 'M0M2_k1': 2, 'M0_k2': 2, 'M2_k2': 2, 'M0M2_k2': 4, 'S0': 1, 'S0+S29': 2}
        for k_ in ('DAS', 'S_units'):
            N_[f'p_{k_}_vs_rand_live'] = {nm: prand(N_['rand_live'][str(DIM[nm])]['vals'], v) for nm, v in N_[k_].items()}
            N_[f'p_{k_}_vs_rand64'] = {nm: prand(N_['rand64'][str(DIM[nm])]['vals'], v) for nm, v in N_[k_].items()}
        log(f'lam {lam} nulls: DAS {N_["DAS"]}; units {N_["S_units"]}; ' + '; '.join(f'{nm} d{d} q {N_[nm][str(d)]["q"]}' for nm in ('rand64', 'rand_live') for d in dsel)
            + f'; pca_S {N_["pca_S"]}; svd_boost_diff {N_["svd_boost_diff"]}; dead energy {N_["rand64_dim2_energy_on_dead"]:.3f}; d4 {N_["d4_rownorm"]}; overlap {N_["overlap_DAS_M0M2_k1"]}; p {N_["p_DAS_vs_rand_live"]} {N_["p_S_units_vs_rand_live"]}')
        save()

        # ---- h4 mass read-out: per-jet response and causal hold of the read-out direction
        M_ = E['readout_h4'] = {}
        dB = (RO(RB['h4']) - mh0)[hB]; RP = net.fwd(hls_prep(sortT(pscale(T0, lam)[0])), keep=('h4',)); dP = (RO(RP['h4']) - mh0)[hB]
        ch = cB.chB
        M_['B'] = dict(mean=float(dB.mean()), sd=float(dB.std()), q=np.quantile(dB, [.025, .25, .5, .75, .975]).tolist(), absmean=float(np.abs(dB).mean()),
                       mean_changed=float(dB[ch].mean()), mean_unchanged=float(dB[~ch].mean()), absmean_changed=float(np.abs(dB[ch]).mean()), absmean_unchanged=float(np.abs(dB[~ch]).mean()),
                       by_true_class={X_: float(dB[y[hB] == c].mean()) for c, X_ in enumerate(['g', 'q', 'W', 'Z', 't'])})
        M_['P'] = dict(mean=float(dP.mean()), sd=float(dP.std()), true=float(np.log(lam)))
        lo4, hi4 = net.jn.rng_['h4']; lo4b, hi4b = np.minimum(np.minimum(lo4, RB['h4']), R0['h4']), np.maximum(np.maximum(hi4, RB['h4']), R0['h4'])
        h4hold = lambda Qm: net.tail('h4', np.clip(RB['h4'] + ((R0['h4'] - RB['h4']) @ Qm) @ Qm.T, lo4b, hi4b))[0][hB]
        h4patch = lambda Qm: net.tail('h4', np.clip(R0['h4'] + ((RB['h4'] - R0['h4']) @ Qm) @ Qm.T, lo4b, hi4b))[0][hB]
        M_['hold_readout_dir_removed'] = cB.rem(h4hold(qro))
        M_['patch_readout_dir_changed'] = float((h4patch(qro).argmax(1) != c0[hB]).mean())
        rh = np.random.default_rng(seed + 700); vr, vp = [], []
        for _ in range(NH):
            Qm = np.zeros((64, 1)); Qm[liveH] = X.rand_Q(len(liveH), 1, rh); vr.append(cB.rem(h4hold(Qm))); vp.append(float((h4patch(Qm).argmax(1) != c0[hB]).mean()))
        M_['null_h4_live_1dim'] = dict(removed_q=np.quantile(vr, [.025, .5, .975, 1]).tolist(), patch_changed_q=np.quantile(vp, [.025, .5, .975, 1]).tolist(),
                                       p_hold_ge=prand(vr, M_['hold_readout_dir_removed']), p_patch_ge=prand(vp, M_['patch_readout_dir_changed']))
        log(f'lam {lam} read-out h4: {M_}')
        save()
    # Holm for the named single units vs the live-restricted 1-dim null (lam 0.9 / 1.1), m = 130 as in E26
    R['holm_note'] = 'p_rand vs live-restricted random 1-dim S directions on half B; Holm with m = 130 (E26 family size)'
    for lam in ('0.9', '1.1'):
        N_ = R['lam'][lam]['null_halfB']; v1 = N_['rand_live']['1']['vals']; p = prand(v1, N_['S_units']['S0'])
        R['lam'][lam]['S0_p_live1'] = dict(p=p, holm_bound=min(1, 130 * p))
    save(); log(f'wrote {JS}; E26 verify DONE')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, default=2602); ap.add_argument('--smoke', type=int, default=0)
    main(ap.parse_args())
