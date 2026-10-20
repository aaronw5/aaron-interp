"""X10 second verification: angle-only Delta edit (is the 'D' subspace a Delta variable or a pT-scale variable?) and the
full-swap (source-call) baseline for the untrained-network control, on samples disjoint from X10 and the first verify.

  ./job.sh x10-verify3  'python -m interp.x10_verify3'               (trained; fresh sample; writes x10_verify3.json)
  ./job.sh x10-verify3u 'python -m interp.x10_verify3 --untrained'   (X10's own T and test pairs; trained + untrained 0-2;
                                                                       writes x10_verify3_untrained.json)
  --smoke: small sizes.

Same code path as X10 (interp.x10_das_jedi: JNet exact fixed-point tail, das_fit with straight-through gradient and range
clip Pi, scores, boot_eff, boot_diff, rand_Q; same hyper-parameters).

Trained run.
 Samples. F3 = sample(2500, 41), T3 = sample(2500, 43); every jet of X10's F (seed 0) / T (seed 17) and of the first verify's
   F' (seed 23) / T' (seed 29) is removed from both, and F3's jets from T3. Pairs: 6,000 train from F3 (seed 52, last 20 % for
   model selection), 4,000 test from T3 (seed 51).
 Counterfactuals: X10's M0, M2, D (pT x c, c = sqrt(M0_s M2_s / (M0_b M2_b)); M0 and M2 both x c), WZ, and
   Da = angle-only Delta edit: angles x c with the same c (M0 unchanged, M2 x c^2, so Delta = 1/2 log(M0 M2) set to the source's
   value exactly as in D, but M0 and the pT ranks / absolute pT thresholds untouched).
 DAS for V in (M0, M2, D, WZ, Da) at S, h4, h5, k = 1, 2, 4; 200 random subspaces per stage and k; transfer of every subspace
   to every counterfactual; paired bootstrap on the Da call-changing pairs: D-DAS minus Da-DAS, D-DAS minus full swap,
   M2-DAS minus D-DAS; overlap of subspaces: mean squared cosine of principal angles ||Q1^T Q2||_F^2 / k (random: k / d).
 Representation check: gap closed in the D-DAS subspace by the D edit and by the Da edit (1 = the edit moves the subspace
   coordinates to the source's value), and in the Da-DAS subspace by both edits.
 Predictions. Delta stored as one variable in span(Q_D): D-DAS scores on Da counterfactuals near its D score; gap closed by
   the Da edit near that of the D edit; Q_D and Q_Da overlap (cos^2 near 1). Q_D = a pT-scale (M0-and-M2-together) direction,
   not Delta: D-DAS on Da near the full-swap / random level; Da edit does not close the gap in Q_D; low overlap.
Untrained run. X10's T (sample(4000, 17) minus F) and test pairs (seed 1); per network (trained, untrained 0-2; untrained
 calls from logits centred on the mean over F = sample(4000, 0), as X10): counterfactual calls for M0, M2, D, WZ, the
 source-call (full-swap) baseline on E_H, the exact full swap (Q = I, clip) and X10's stored DAS effective IIA per cell
 minus the full-swap baseline.
"""
import argparse, json, time
import numpy as np
from interp.x10_das_jedi import (OUT, KS, STAGES, clean, moments, do_edit, scale, wz_call, indices, JNet, das_fit, proj,
                                 scores, boot_eff, boot_diff, rand_Q)

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
V5 = ('M0', 'M2', 'D', 'WZ', 'Da'); E4 = ('M0', 'M2', 'D', 'Da')


def edit(x, V, mb, ms):
    if V != 'Da': return do_edit(x, V, mb, ms)
    (M0b, M2b), (M0s, M2s) = mb, ms; c = np.sqrt(M0s * M2s / (M0b * M2b)); return scale(x, np.ones(len(x)), c)


def pairs(n, N, seed):
    r = np.random.default_rng(seed); b = r.integers(0, N, n); s = r.integers(0, N, n)
    while (b == s).any(): s[b == s] = r.integers(0, N, (b == s).sum())
    return b, s


def boot_two(c, l1, l2, base, B=1000, seed=11):
    """eff(c vs l1) - eff(c vs l2), each on its own E_H, pairs resampled jointly"""
    rng = np.random.default_rng(seed); o1 = (c == l1) & (l1 != base); e1 = l1 != base; o2 = (c == l2) & (l2 != base); e2 = l2 != base
    n = len(c); v = []
    for _ in range(B):
        i = rng.integers(0, n, n); v.append(o1[i].sum() / max(e1[i].sum(), 1) - o2[i].sum() / max(e2[i].sum(), 1))
    v = np.asarray(v); d = o1.sum() / max(e1.sum(), 1) - o2.sum() / max(e2.sum(), 1)
    return dict(diff=float(d), ci=np.quantile(v, [.025, .975]).tolist(), p=float(min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()) + 1 / (B + 1))))


def trained(a):
    sm = a.smoke; JS = OUT / ('x10_verify3_smoke.json' if sm else 'x10_verify3.json'); OUT.mkdir(parents=True, exist_ok=True)
    R = dict(smoke=sm); save = lambda: JS.write_text(json.dumps(clean(R), indent=1))
    from interp.jedi_core import sample
    pc = 300 if sm else 2500
    old = np.union1d(np.union1d(indices(4000, 0), indices(4000, 17)), np.union1d(indices(3000, 23), indices(3000, 29)))
    xF, yF, _ = sample(pc, 41); xT, yT, _ = sample(pc, 43); iF, iT = indices(pc, 41), indices(pc, 43)
    kF = ~np.isin(iF, old); kT = ~np.isin(iT, np.union1d(old, iF)); xF, yF, xT, yT = xF[kF], yF[kF], xT[kT], yT[kT]
    NF, NT = len(yF), len(yT); R['N'] = dict(F=NF, T=NT); log('N', R['N'])
    mF, mT = moments(xF), moments(xT)
    npair, ntest = (800, 600) if sm else (6000, 4000)
    fb, fs = pairs(npair, NF, 52); tb, ts = pairs(ntest, NT, 51)
    R['test_pairs_sharing_base'] = float(1 - len(np.unique(tb)) / ntest)
    cD = np.sqrt(mT[0][ts] * mT[1][ts] / (mT[0][tb] * mT[1][tb])); R['Da_angle_scale_q'] = np.quantile(cD, [.025, .5, .975]).tolist()
    R['Da_frac_angle_scale_gt3_or_lt1/3'] = float(((cD > 3) | (cD < 1 / 3)).mean())
    net = JNet(); RF = net.reps(xF); RT = net.reps(xT); cF = RF['logits'].argmax(1); cT = RT['logits'].argmax(1)
    R['acc_T'] = float((cT == yT).mean()); R['acc_F'] = float((cF == yF).mean()); log('acc', R['acc_T'], R['acc_F'])
    R['tail_check'] = net.check(xT[:1000], {k: v[:1000] for k, v in RT.items()}); save()
    lab = dict(train=dict(base=cF[fb], src=cF[fs]), test=dict(base=cT[tb], src=cT[ts])); RE = {}
    for V in E4:
        lab['train'][V] = net.calls(edit(xF[fb], V, (mF[0][fb], mF[1][fb]), (mF[0][fs], mF[1][fs])))
        RE[V] = net.reps(edit(xT[tb], V, (mT[0][tb], mT[1][tb]), (mT[0][ts], mT[1][ts])), keep=('S', 'h4', 'h5', 'logits'))
        lab['test'][V] = RE[V]['logits'].argmax(1)
    lab['train']['WZ'] = wz_call(RF['logits'][fb], RF['logits'][fs]); lab['test']['WZ'] = wz_call(RT['logits'][tb], RT['logits'][ts])
    bT, sT = lab['test']['base'], lab['test']['src']
    R['cf_change_frac'] = {V: float((lab['test'][V] != bT).mean()) for V in V5}
    R['source_call_baseline'] = {V: dict(eff=float((lab['test'][V] == sT)[lab['test'][V] != bT].mean()), n=int((lab['test'][V] != bT).sum())) for V in V5}
    R['D_vs_Da_same_call'] = dict(all=float((lab['test']['D'] == lab['test']['Da']).mean()),
                                  on_Da_EH=float((lab['test']['D'] == lab['test']['Da'])[lab['test']['Da'] != bT].mean()))
    log('cf change', R['cf_change_frac']); log('source-call baseline', R['source_call_baseline']); log('D vs Da', R['D_vs_Da_same_call']); save()
    nrand = 30 if sm else 200; steps = 100 if sm else 800; B = 100 if sm else 1000
    R['stages'] = {}
    for st in STAGES:
        ft = net.tail(st); d = RT[st].shape[1]; rg = net.rng_[st]; hb, hs = RT[st][tb], RT[st][ts]; res = dict(d=d, k={})
        call = lambda Q: net.exact(st, proj(rg, hb, hb + ((hs - hb) @ Q) @ Q.T))[0].argmax(1)

        def gap(Q, V):
            he = RE[V][st]; return float(1 - (((he - hs) @ Q) ** 2).sum() / max((((hb - hs) @ Q) ** 2).sum(), 1e-12))
        for k in KS:
            rk = dict(das={}, random={}); rr = np.random.default_rng(2000 + k); null = {V: [] for V in V5}; ngap = {V: [] for V in E4}
            for _ in range(nrand):
                Q = rand_Q(d, k, rr); c = call(Q)
                for V in V5:
                    e = scores(c, lab['test'][V], bT)['eff']; null[V].append(np.nan if e is None else e)
                for V in E4: ngap[V].append(gap(Q, V))
            Qs, cs = {}, {}
            for V in V5:
                Q, info = das_fit(ft, RF[st][fb], RF[st][fs], lab['train'][V], lab['train']['base'], k, seed=10 + k, steps=steps, rg=rg)
                c = call(Q); Qs[V], cs[V] = Q, c; LV = lab['test'][V]; s_ = scores(c, LV, bT); s_['eff_ci'] = boot_eff(c, LV, bT, B=B)
                nv = np.asarray(null[V]); fin = np.isfinite(nv)
                s_['p_rot'] = float((1 + (nv[fin] >= s_['eff']).sum()) / (1 + fin.sum())) if s_['eff'] is not None and fin.any() else None
                s_['transfer'] = {V2: scores(c, lab['test'][V2], bT)['eff'] for V2 in V5}
                if V in E4: s_['gap_closed'] = {V2: gap(Q, V2) for V2 in E4}
                s_['train'] = info; rk['das'][V] = s_
                rk['random'][V] = dict(mean=float(np.nanmean(nv)), q=np.nanquantile(nv, [.025, .5, .975]).tolist() if fin.any() else None)
                log(f'{st} k={k} {V}: eff {s_["eff"]} {s_["eff_ci"]} n {s_["n_eff"]} p_rot {s_["p_rot"]} random {rk["random"][V]["mean"]:.3f}; '
                    f'transfer {s_["transfer"]}; gap {s_.get("gap_closed")}')
            rk['random_gap_closed_q'] = {V: np.quantile(ngap[V], [.025, .5, .975]).tolist() for V in E4}
            LA = lab['test']['Da']
            rk['on_Da'] = dict(D_minus_Da_das=boot_diff(cs['D'], cs['Da'], LA, bT, B=B), D_minus_full_swap=boot_diff(cs['D'], sT, LA, bT, B=B),
                               M2_minus_D=boot_diff(cs['M2'], cs['D'], LA, bT, B=B), Da_minus_full_swap=boot_diff(cs['Da'], sT, LA, bT, B=B),
                               D_das_on_D_minus_on_Da=boot_two(cs['D'], lab['test']['D'], LA, bT, B=B))
            rk['overlap_cos2'] = {f'{V1}-{V2}': float((np.linalg.norm(Qs[V1].T @ Qs[V2]) ** 2) / k) for i, V1 in enumerate(V5) for V2 in V5[i + 1:]}
            rk['overlap_random_expect'] = k / d
            log(f'{st} k={k} on Da: {rk["on_Da"]}'); log(f'{st} k={k} overlap cos2 {rk["overlap_cos2"]} (random {k / d:.3f}); random gap {rk["random_gap_closed_q"]}')
            res['k'][k] = rk; R['stages'][st] = res; save()
    log(f'wrote {JS}; DONE')


def untrained(a):
    sm = a.smoke; JS = OUT / ('x10_verify3_untrained_smoke.json' if sm else 'x10_verify3_untrained.json'); R = dict(smoke=sm)
    save = lambda: JS.write_text(json.dumps(clean(R), indent=1))
    from interp.jedi_core import sample
    pc = 300 if sm else 4000
    xF, _, _ = sample(pc, 0); xT, yT, _ = sample(pc, 17); keep = ~np.isin(indices(pc, 17), indices(pc, 0)); xT, yT = xT[keep], yT[keep]
    mT = moments(xT); ntest = 600 if sm else 5000; tb, ts = pairs(ntest, len(yT), 1); R['N'] = dict(T=len(yT), pairs=ntest)
    for nm in ('trained', 'untrained0', 'untrained1', 'untrained2'):
        net = JNet() if nm == 'trained' else JNet(untrained=True, seed=int(nm[-1]))
        RT = net.reps(xT, keep=('S', 'h4', 'h5', 'logits'))
        if nm != 'trained':
            net.off = net.reps(xF, keep=('logits',))['logits'].mean(0); RT['logits'] = RT['logits'] - net.off
        out0 = net.check(xT[:500], {k: v[:500] for k, v in RT.items()})
        cT = RT['logits'].argmax(1); bT, sT = cT[tb], cT[ts]; lab = {}
        for V in ('M0', 'M2', 'D'): lab[V] = net.calls(do_edit(xT[tb], V, (mT[0][tb], mT[1][tb]), (mT[0][ts], mT[1][ts])))
        lab['WZ'] = wz_call(RT['logits'][tb], RT['logits'][ts])
        out = dict(acc=float((cT == yT).mean()), calls=np.bincount(cT, minlength=5).tolist(), base_differs_from_source=float((bT != sT).mean()), tail_check=out0)
        stored = None
        p = OUT / f'x10_{nm}.json'
        if p.exists(): stored = json.loads(p.read_text())
        for V in ('M0', 'M2', 'D', 'WZ'):
            EH = lab[V] != bT; o = dict(n_eff=int(EH.sum()), source_call_baseline=float((lab[V] == sT)[EH].mean()) if EH.sum() >= 20 else None)
            if EH.sum() >= 20:
                o['source_call_baseline_ci'] = boot_eff(sT, lab[V], bT, B=200 if sm else 1000)
            for st in STAGES:
                cfs = net.exact(st, proj(net.rng_[st], RT[st][tb], RT[st][ts]))[0].argmax(1)
                o[f'full_swap_eff_{st}'] = scores(cfs, lab[V], bT)['eff']; o[f'full_swap_equals_source_{st}'] = float((cfs == sT).mean())
                if stored is not None and not sm:
                    for k in KS:
                        e = stored['stages'][st]['k'][str(k)]['das'][V]['rand']
                        o[f'stored_das_{st}_k{k}'] = e['eff']; o[f'stored_n_{st}_k{k}'] = e['n_eff']
                        o[f'das_minus_baseline_{st}_k{k}'] = None if (e['eff'] is None or o['source_call_baseline'] is None) else e['eff'] - o['source_call_baseline']
            out[V] = o
        R[nm] = out; log(nm, json.dumps(clean(out))); save()
    log(f'wrote {JS}; DONE')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); ap.add_argument('--untrained', action='store_true')
    a = ap.parse_args(); untrained(a) if a.untrained else trained(a)
