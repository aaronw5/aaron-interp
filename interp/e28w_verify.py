"""E28w verification (fresh sample) + missing controls.
Replicates on sample(per_class=1200, seed=3907) (6,000 jets): split (z 0.5) vs rot at ranks 0/5/13/21; rank_PS share; a32 clamp share
  (worker's: both halves := parent's a32) and a sum-preserving variant (half 1 := parent's a32, half 2 := 0, so the pooled a32 sum and
  P equal the unsplit jet's for the a32 column); S19+S30 held vs all 171 live-S pairs; all live S held (sanity, must remove 1.0);
  window17 / free-pair sufficiency: hold all live S except a pair, fraction of split changes reproduced (same jet, same new call),
  rank of (S29, S30) among the 171 pairs; H_slot at rank 13 (N = 64); merge of closest real pair.
New controls (jets with N <= 63, nothing else moves slot):
  split_soft  exact collinear split of the softest particle (rank N-1, z 0.5): N+1, no rank shift, M0/M1/M2 exact.
  rot_soft    rotation of the softest particle about the jet axis.
  add_soft    one extra particle, pT = 0.999 x the softest pT, at the (deta, dphi) of a random real particle of the jet; lands in slot N.
  bump_hard   M0-matched no-count control: the hardest particle's pT raised by the softest pT (no rank changes, N fixed).
Run: ./job.sh e28w-verify 'python -m interp.e28w_verify'   -> $AI_RESULTS/jedi/e28w_verify.json
"""
import itertools, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from interp.e28w_split_route_jedi import (Net, split_ext, rotate, nreal, merge_pair, drop_softest, Boot, mcnemar, holm, qci,
                                          flows, binom_p, LIVE_S, NAMED_S, B_FWD, RANKS)
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); OUT = RES / 'jedi'; T0 = time.time()
def log(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


def main():
    from interp.jedi_core import sample
    OUT.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(3907)
    net = Net()
    x, y, _ = sample(per_class=1200, seed=3907); x = np.asarray(x, np.float32); J = len(x); N0 = nreal(x)
    bt = Boot(J, 1000, 11)
    R0 = net.fwd(x, keep=('logits', 'S', 'a'), a_units=[32]); c0 = R0['logits'].argmax(1)
    # independent call check through jedi_core.run
    c0b = net.n.run(x)['logits'].argmax(1)
    res = dict(seed=3907, n=J, acc=float((c0 == y).mean()), calls_equal_jedi_core=float((c0 == c0b).mean()),
               N_counts=dict(le60=int((N0 <= 60).sum()), n61_63=int(((N0 > 60) & (N0 < 64)).sum()), n64=int((N0 == 64).sum())))
    log(res)
    pairs = list(itertools.combinations(LIVE_S, 2)); i29 = pairs.index((29, 30))
    allS = np.ones(64, np.float32)

    def changed(ce, ok): return (ce != c0) & ok
    def rate(ch, sel): return dict(changed=float(ch[sel].mean()), ci=qci(bt.mean(ch, sel)), n=int(len(sel)))
    def share(chE, chH, sel):
        nE, nH = int(chE[sel].sum()), int(chH[sel].sum())
        return dict(removed=1 - nH / nE if nE else None, ci=qci(bt.removed(chE, chH, sel)) if nE else None, nE=nE, nH=nH,
                    mcnemar=mcnemar(chE[sel], chH[sel]))

    def S_scan(Re, ce, chE, ok, sel):
        """hold-sets and free-pair sufficiency for one edit"""
        out = {}
        mA = np.zeros(64, np.float32); mA[LIVE_S] = 1
        out['all_live_S_held'] = share(chE, changed(net.tail(Re['S'], mA, R0['S']).argmax(1), ok), sel)
        out['all_64_S_held'] = share(chE, changed(net.tail(Re['S'], allS, R0['S']).argmax(1), ok), sel)
        rem, repro, spur = [], [], []
        nE = max(int(chE[sel].sum()), 1)
        for (u, v) in pairs:
            m = np.zeros(64, np.float32); m[[u, v]] = 1
            ch = changed(net.tail(Re['S'], m, R0['S']).argmax(1), ok); rem.append(1 - ch[sel].sum() / nE)
            mf = mA.copy(); mf[[u, v]] = 0
            cf = net.tail(Re['S'], mf, R0['S']).argmax(1)
            repro.append(float(((cf == ce) & chE)[sel].sum() / nE)); spur.append(float((changed(cf, ok) & ~chE)[sel].mean()))
        rem, repro = np.array(rem), np.array(repro)
        j1930 = pairs.index((19, 30))
        out['S19+S30_held'] = dict(removed=float(rem[j1930]), null_q95=float(np.quantile(rem, .95)),
                                   rank_p=float((1 + (rem >= rem[j1930]).sum() - 1) / len(rem)))
        top = np.argsort(-repro)[:5]
        out['free_pair_sufficiency'] = dict(
            S29_S30_reproduced=float(repro[i29]), S29_S30_spurious_rate=spur[i29], null_mean=float(repro.mean()),
            null_q95=float(np.quantile(repro, .95)), rank_p_high=float(((repro >= repro[i29]).sum()) / len(repro)),
            top5=[(f'S{pairs[t][0]}+S{pairs[t][1]}', float(repro[t])) for t in top], n_pairs=len(pairs))
        return out

    E = {}
    for k in RANKS:
        slot = np.where(N0 >= k + 2, k, -1); ok = slot >= 0; sel = np.flatnonzero(ok)
        Yr = rotate(x, slot, rng); cr = net.fwd(Yr, keep=('logits',))['logits'].argmax(1); chR = changed(cr, ok)
        for z in ((0.5, 0.9) if k == 21 else (0.5,)):
            Ys, d1, d2 = split_ext(x, slot, z)
            Re = net.fwd(Ys, keep=('logits', 'S', 'a'), a_units=[32]); ce = Re['logits'].argmax(1); chE = changed(ce, ok)
            ceb = net.n.run(Ys)['logits'].argmax(1)
            r = dict(split=rate(chE, sel), rot=rate(chR, sel), jedi_core_calls_equal=float((ce == ceb).mean()),
                     diff_ci=qci(bt.mean(chE.astype(np.float32) - chR.astype(np.float32), sel)), mcnemar_vs_rot=mcnemar(chE[sel], chR[sel]),
                     **{f'flow_{a}': b for a, b in flows(c0, ce, sel).items() if a != 'flow'},
                     by_N={nm: float(chE[s_].mean()) if len(s_) else None for nm, s_ in
                           (('le60', sel[N0[sel] <= 60]), ('n61_63', sel[(N0[sel] > 60) & (N0[sel] < 64)]), ('n64', sel[N0[sel] == 64]))})
            # rank patch
            Sp, chk = net.rank_patch(x, Ys, slot, d1, d2, 'PS'); chP = changed(net.tail(Sp).argmax(1), ok)
            r['rank_PS'] = dict(share(chE, chP, sel), check=float(np.abs(chk - Re['S'])[sel].max()))
            # a32 clamps
            a0p = R0['a'][np.arange(J), np.maximum(slot, 0), 0]; ae = Re['a'][:, :, 0]
            def clamp(v2_parent):
                m = np.zeros(64, np.float32); m[32] = 1
                def f(i, n):
                    V = np.zeros((B_FWD, 64, 64), np.float32); V[:n, :, 32] = ae[i:i + n]; rr = np.arange(n)
                    a1 = d1[i:i + n] >= 0; V[rr[a1], d1[i:i + n][a1], 32] = a0p[i:i + n][a1]
                    a2 = d2[i:i + n] >= 0; V[rr[a2], d2[i:i + n][a2], 32] = a0p[i:i + n][a2] if v2_parent else 0.0
                    return {'a': (m, V)}
                return f
            for nm, v2 in (('a32_both_parent', True), ('a32_sum_preserving', False)):
                ch = changed(net.fwd(Ys, clamp(v2), keep=('logits',))['logits'].argmax(1), ok); r[nm] = share(chE, ch, sel)
            r['S'] = S_scan(Re, ce, chE, ok, sel)
            E[f'split_r{k}_z{z}'] = r; E[f'split_r{k}_z{z}']['_chE'] = chE
            log(k, z, {a: (b if not isinstance(b, dict) else {c: d for c, d in b.items() if c not in ('mcnemar',)}) for a, b in r.items() if a not in ('S', '_chE')})
            log('   S', r['S'])
    # ---- new controls on N <= 63 (no overflow, nothing else moves)
    lo = (N0 <= 63) & (N0 >= 2); sl = np.flatnonzero(lo); soft = np.where(lo, N0 - 1, -1)
    pmin = x[np.arange(J), np.maximum(N0 - 1, 0), 0]
    Ysf, d1s, d2s = split_ext(x, soft, 0.5)
    noshift_ok = bool(np.all((d1s[sl] == N0[sl] - 1) & (d2s[sl] == N0[sl])))
    Xa = x.copy(); jr = rng.integers(0, np.maximum(N0, 1)); Xa[sl, N0[sl], 0] = 0.999 * pmin[sl]
    Xa[sl, N0[sl], 1] = x[sl, jr[sl], 1]; Xa[sl, N0[sl], 2] = x[sl, jr[sl], 2]
    sorted_ok = bool(np.all(np.diff(Xa[sl, :, 0], axis=1) <= 0))
    Xb = x.copy(); Xb[sl, 0, 0] += pmin[sl]
    Xrs = rotate(x, soft, rng)
    C = {}
    calls = {}
    for nm, Y in (('split_soft', Ysf), ('add_soft', Xa), ('bump_hard', Xb), ('rot_soft', Xrs)):
        Rr = net.fwd(Y, keep=('logits', 'S')); cc = Rr['logits'].argmax(1); ch = changed(cc, lo); calls[nm] = (cc, ch)
        C[nm] = dict(rate(ch, sl), **{f'flow_{a}': b for a, b in flows(c0, cc, sl).items() if a != 'flow'})
        if nm in ('split_soft', 'add_soft'): C[nm]['S'] = S_scan(Rr, cc, ch, lo, sl)
        log(nm, {a: b for a, b in C[nm].items() if a != 'S'}); log('   ', C[nm].get('S'))
    C['checks'] = dict(split_soft_noshift=noshift_ok, add_soft_sorted=sorted_ok, median_pmin=float(np.median(pmin[sl])),
                       median_rel_M0_change=float(np.median(pmin[sl] / x[sl, :, 0].sum(1))))
    C['add_vs_bump'] = dict(diff_ci=qci(bt.mean(calls['add_soft'][1].astype(np.float32) - calls['bump_hard'][1].astype(np.float32), sl)),
                            mcnemar=mcnemar(calls['add_soft'][1][sl], calls['bump_hard'][1][sl]))
    C['split_soft_vs_rot'] = dict(diff_ci=qci(bt.mean(calls['split_soft'][1].astype(np.float32) - calls['rot_soft'][1].astype(np.float32), sl)),
                                  mcnemar=mcnemar(calls['split_soft'][1][sl], calls['rot_soft'][1][sl]))
    # overlap: of split_rk changes (N<=63 jets), fraction also changed by add_soft to the same class
    for k in RANKS:
        chE = E[f'split_r{k}_z0.5']['_chE'] & lo; ss = np.flatnonzero(chE)
        if len(ss) == 0: continue
        # need the split call: recompute cheaply
        C[f'overlap_r{k}'] = dict(n_split_changes=int(len(ss)), also_add_soft=float(calls['add_soft'][1][ss].mean()),
                                  also_split_soft=float(calls['split_soft'][1][ss].mean()), also_bump=float(calls['bump_hard'][1][ss].mean()),
                                  also_rot_soft=float(calls['rot_soft'][1][ss].mean()))
    res['controls_N_le63'] = C
    # ---- H_slot at rank 13, N = 64
    s64 = np.flatnonzero(N0 == 64); k = 13; slot = np.where(N0 >= k + 2, k, -1)
    Xd = drop_softest(x); cd = net.fwd(Xd, keep=('logits',))['logits'].argmax(1); chD = (cd != c0)
    Ysd, _, _ = split_ext(Xd, np.where(N0 == 64, slot, -1), 0.5); cdd = net.fwd(Ysd, keep=('logits',))['logits'].argmax(1)
    chNO = (cdd != cd); chE13 = E['split_r13_z0.5']['_chE']
    res['H_slot_r13'] = dict(n=len(s64), split=rate(chE13, s64), drop_softest=rate(chD, s64), split_nooverflow=rate(chNO, s64),
                             drop_flows={a: b for a, b in flows(c0, cd, s64).items()})
    log('H_slot', res['H_slot_r13'])
    # ---- merge pair
    Ym, okm, dmin = merge_pair(x); sm = np.flatnonzero(okm); cm = net.fwd(Ym, keep=('logits',))['logits'].argmax(1); chm = changed(cm, okm)
    res['merge_pair'] = dict(frac=float(okm.mean()), **rate(chm, sm), **{f'flow_{a}': b for a, b in flows(c0, cm, sm).items() if a != 'flow'},
                             dmin_q50=float(np.median(dmin[np.isfinite(dmin)])), frac_dmin_zero=float((dmin == 0).mean()))
    log('merge', res['merge_pair'])
    for v in E.values(): v.pop('_chE')
    res['edits'] = E; res['wall_s'] = time.time() - T0
    (OUT / 'e28w_verify.json').write_text(json.dumps(res, indent=1, default=float)); log('DONE')


if __name__ == '__main__':
    main()
