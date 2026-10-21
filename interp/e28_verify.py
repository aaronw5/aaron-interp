"""E28 verification (fresh sample): collinear splits / soft additions in JEDI-linear (3-feature n64, official, fixed point).

Re-runs the key edits of interp.e28_irc_split_jedi on a new sample (jedi_core.sample(per_class=2000, seed=2029), jets with N <= 60) and
adds two controls that E28 did not have:
  rot_r{k}   moment-preserving matched-rank control: the particle of pT rank k is rotated about the jet axis by a uniform random
             azimuth (pT and r = sqrt(deta^2 + dphi^2) kept). N, every pT rank, M0 = sum pT, M1 = sum pT r, M2 = sum pT r^2 are
             unchanged exactly; only the jet mass (pairwise angles) moves. This is the same-rank, same-count edit that the swap is not.
  random holds  calibration of the hold shares: the share of split-changed calls removed by holding a random 2 of the 19 live S units
             (compare with S0+S19 and S29+S30), and by holding a random 17 of the 19 (= letting 2 random units respond; compare with
             the 17 rank-window units = letting only S29, S30 respond). Plus holds 'all live S except S29' / 'except S30'
             (sufficiency of one windowless object) and S29 / S30 alone.
Edits (same code as E28): split_r{k}_z{0.5,0.8}, swap_r{k}, splitfix_r{k}, soft_add{1,5}, drop_softest, pad0_at, pad0_origin.
Statistics: 1000 multinomial bootstrap resamples over jets; exact McNemar and sign-flip permutation for split vs swap and split vs rot,
Holm over the 8 (rank, z) cells per control; random-set p = (1 + #random sets with share >= named) / (1 + #sets).
Run: ./job.sh e28-verify 'python -m interp.e28_verify'  -> $AI_RESULTS/jedi/e28_verify.json
"""
import json
import time

import numpy as np

from interp.e28_irc_split_jedi import (LIVE_S, WINDOW, RANKS, ZS, Boot, qci, mcnemar, holm, perm_signflip, binom_p, nreal, moments,
                                       pick_rank, split, swap, drop_softest, soft_add, pad0, OUT)

SEED = 2029
T0 = time.time()


def log(*a): print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


def rotate(X, slot, rng):
    Y = X.copy(); rows = np.flatnonzero(slot >= 0); k = slot[rows]
    de, dp = X[rows, k, 1].astype(np.float64), X[rows, k, 2].astype(np.float64)
    r = np.hypot(de, dp); th = np.arctan2(de, dp) + rng.uniform(0, 2 * np.pi, len(rows))
    Y[rows, k, 2] = (r * np.cos(th)).astype(np.float32); Y[rows, k, 1] = (r * np.sin(th)).astype(np.float32)
    return Y


def main():
    from interp.jedi_core import Jedi, sample
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    x, y, _ = sample(per_class=2000, seed=SEED); x = np.asarray(x, np.float32); N_all = nreal(x)
    net = Jedi(); keep = ('logits', 'S', 'g')
    R_all = net.run(x, keep=keep, batch=2000); c_all = R_all['logits'].argmax(1)
    res = dict(seed=SEED, N=len(x), acc_all=float((c_all == y).mean()))
    sub = np.flatnonzero(N_all <= 60); X0 = x[sub]; ys = y[sub]; N0 = N_all[sub]; J = len(X0)
    R0 = {k: v[sub] for k, v in R_all.items()}; c0 = c_all[sub]; corr0 = c0 == ys; sdS = R0['S'].std(0)
    res.update(n_sub=J, acc_sub=float(corr0.mean()), class_counts_sub=np.bincount(ys, minlength=5).tolist())
    log(f"sample {len(x)}, acc {res['acc_all']:.4f}; subset {J} acc {res['acc_sub']:.4f} classes {res['class_counts_sub']}")
    bt = Boot(J, 1000, SEED + 1); Mo0 = moments(X0)
    E = {}; slots = {k: pick_rank(N0, k, rng) for k in RANKS}
    for k in RANKS:
        s = slots[k]; ok = s >= 0
        for z in ZS: E[f'split_r{k}_z{z}'] = (split(X0, s, z), ok)
        E[f'splitfix_r{k}'] = (drop_softest(split(X0, s, 0.5), np.flatnonzero(ok)), ok)
        E[f'swap_r{k}'] = (swap(X0, s), ok)
        E[f'rot_r{k}'] = (rotate(X0, s, rng), ok)
    for n in (1, 5):
        Xn, ok = soft_add(X0, n, rng); E[f'soft_add{n}'] = (Xn, ok)
    E['pad0_at'] = (pad0(X0, False), np.ones(J, bool)); E['pad0_origin'] = (pad0(X0, True), np.ones(J, bool))
    E['drop_softest'] = (drop_softest(X0), np.ones(J, bool))
    # random hold sets (fixed across edits)
    hr = np.random.default_rng(SEED + 7); others = list(LIVE_S)
    rand2 = [sorted(hr.choice(others, 2, replace=False).tolist()) for _ in range(20)]
    rand17 = [[u for u in LIVE_S if u not in p] for p in [sorted(hr.choice(others, 2, replace=False).tolist()) for _ in range(20)]]
    res['rand2_sets'] = rand2; res['rand17_free_pairs'] = [[u for u in LIVE_S if u not in s] for s in rand17]
    out = {}; chg = {}
    for name, (Xe, ok) in E.items():
        Re = net.run(Xe, keep=keep, batch=2000); ce = Re['logits'].argmax(1); ch = (ce != c0) & ok; chg[name] = ch
        sel = np.flatnonzero(ok); Mo1 = moments(Xe)
        F = np.zeros((5, 5), int); np.add.at(F, (c0[sel], ce[sel]), 1)
        dS = (Re['S'] - R0['S'])[sel] / np.where(sdS > 0, sdS, 1)
        r = dict(n=int(ok.sum()), changed=float(ch[sel].mean()), changed_ci=qci(bt.mean(ch, sel)),
                 acc_before=float(corr0[sel].mean()), acc_after=float((ce == ys)[sel].mean()),
                 mcnemar_correct=mcnemar(corr0[sel], (ce == ys)[sel]), flow=F.tolist(),
                 q_to_g=int(F[1, 0]), g_to_q=int(F[0, 1]), p_qg=binom_p(int(F[1, 0]), int(F[1, 0] + F[0, 1])),
                 W_to_Z=int(F[2, 3]), Z_to_W=int(F[3, 2]), p_WZ=binom_p(int(F[2, 3]), int(F[2, 3] + F[3, 2])),
                 moment_max_rel={k: float(np.max(np.abs(Mo1[k][sel] - Mo0[k][sel]) / np.maximum(Mo0[k][sel], 1e-9))) for k in ('M0', 'M1', 'M2', 'm')},
                 mass_mean_abs_rel=float(np.mean(np.abs(Mo1['m'][sel] - Mo0['m'][sel]) / np.maximum(Mo0['m'][sel], 1e-9))),
                 dS_sd={str(u): dict(mean=float(dS[:, u].mean()), mean_abs=float(np.abs(dS[:, u]).mean())) for u in (0, 1, 19, 29, 30, 56)},
                 S19_frac_at_minus_0p100=float((np.abs(dS[:, 19] + 0.100) < 0.01).mean()))
        out[name] = r
        log(f"{name}: n {r['n']} changed {r['changed']:.4f} {np.round(r['changed_ci'], 4)} acc {r['acc_before']:.4f}->{r['acc_after']:.4f} "
            f"q>g {r['q_to_g']} g>q {r['g_to_q']} W>Z {r['W_to_Z']} Z>W {r['Z_to_W']} dS19 {r['dS_sd']['19']['mean']:+.3f} "
            f"dS0 |{r['dS_sd']['0']['mean_abs']:.3f}| dM2 {r['moment_max_rel']['M2']:.1e} dm {r['moment_max_rel']['m']:.1e}")
        if name in ('split_r0_z0.5', 'split_r5_z0.5', 'split_r13_z0.5', 'split_rsoft_z0.5', 'soft_add1', 'soft_add5'):
            holds = {'S19': [19], 'S0': [0], 'S0+S19': [0, 19], 'window17': WINDOW, 'S29+S30': [29, 30], 'S29': [29], 'S30': [30],
                     'S56': [56], 'all_live': LIVE_S, 'all_but_S29': [u for u in LIVE_S if u != 29],
                     'all_but_S30': [u for u in LIVE_S if u != 30], 'g_all': None}
            for i, s in enumerate(rand2): holds[f'rand2_{i}'] = s
            for i, s in enumerate(rand17): holds[f'rand17_{i}'] = s
            H = {}; nE = int(ch[sel].sum())
            for hn, units in holds.items():
                if units is None: ov = {'g': (np.ones(64), R0['g'])}
                else:
                    m = np.zeros(64); m[units] = 1; ov = {'S': (m, R0['S'])}
                chH = (net.run_jetwise(Xe, ov, batch=2000)['logits'].argmax(1) != c0) & ok; nH = int(chH[sel].sum())
                H[hn] = dict(removed=float(1 - nH / nE), removed_ci=qci(bt.ratio_removed(ch, chH, sel)), p=mcnemar(ch[sel], chH[sel])['p'])
            r2 = np.array([H[f'rand2_{i}']['removed'] for i in range(20)]); r17 = np.array([H[f'rand17_{i}']['removed'] for i in range(20)])
            cal = dict(rand2_mean=float(r2.mean()), rand2_range=[float(r2.min()), float(r2.max())],
                       rand17_mean=float(r17.mean()), rand17_range=[float(r17.min()), float(r17.max())])
            for k_, arr in (('S0+S19', r2), ('S29+S30', r2)):
                cal[f'p_{k_}_ge_rand2'] = float((1 + (arr >= H[k_]['removed']).sum()) / 21)
            cal['p_window17_le_rand17'] = float((1 + (r17 <= H['window17']['removed']).sum()) / 21)
            r['holds'] = H; r['hold_calibration'] = cal
            log('   holds ' + ', '.join(f"{k} {v['removed']:.3f}" for k, v in H.items() if not k.startswith('rand')) + f' | {cal}')
    comp = {}
    for ctl in ('swap', 'rot'):
        keys = []
        for k in RANKS:
            for z in ZS:
                a_, b_ = chg[f'split_r{k}_z{z}'], chg[f'{ctl}_r{k}']; sel = np.flatnonzero(E[f'{ctl}_r{k}'][1])
                d = a_.astype(np.float32) - b_.astype(np.float32); key = f'split_r{k}_z{z}_vs_{ctl}'; keys.append(key)
                comp[key] = dict(n=len(sel), split=float(a_[sel].mean()), control=float(b_[sel].mean()), diff=float(d[sel].mean()),
                                 diff_ci=qci(bt.mean(d, sel)), p_mcnemar=mcnemar(a_[sel], b_[sel])['p'], p_perm=perm_signflip(a_[sel], b_[sel]))
        for key, p in zip(keys, holm([comp[k]['p_perm'] for k in keys])): comp[key]['p_perm_holm8'] = p
    for k in RANKS:
        a_, b_ = chg[f'splitfix_r{k}'], chg[f'split_r{k}_z0.5']; sel = np.flatnonzero(E[f'split_r{k}_z0.5'][1])
        d = a_.astype(np.float32) - b_.astype(np.float32)
        comp[f'splitfix_r{k}_vs_split'] = dict(diff=float(d[sel].mean()), diff_ci=qci(bt.mean(d, sel)), p_mcnemar=mcnemar(a_[sel], b_[sel])['p'])
    for k, v in comp.items(): log(k, {kk: (np.round(vv, 4).tolist() if isinstance(vv, (list, float)) else vv) for kk, vv in v.items()})
    res['edits'] = out; res['comparisons'] = comp; res['wall_s'] = time.time() - T0
    (OUT / 'e28_verify.json').write_text(json.dumps(res, indent=1)); log('wrote', OUT / 'e28_verify.json', 'E28 VERIFY DONE')


if __name__ == '__main__':
    main()
