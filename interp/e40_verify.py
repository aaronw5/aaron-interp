"""E40 verify: rerun the key E40 edits on fresh samples and add an emission-radius control.

Network: official JEDI-linear 3-feature n64 (exact forward via e28w Net). Margin = logit_g - logit_q.
Samples (fresh seeds): fit = sample(1500, 4501) q/g jets (natural slope of margin vs N); test = sample(1500, 4502), 5-class
  accuracy, edits on its 3,000 q/g jets.
Edits (re-used from interp.e40_qg_count_jedi, unchanged): null_resort; collinear split f 0.5 of rank 0 / rank 13 / random rank >= 30
  with the non-collinear (dR 0.05) control, rotation (soft), rank-preserving per-slot reconstruction (r13, soft); emissions of k
  massless particles at fixed jet four-momentum; merge of the two softest particles and move2.
New control (E40 issue 3): emission radius scan. emit k = 3 particles of 0.5 GeV at r = 0.02, 0.1, 0.15, 0.3, 0.5 (and 0.05 GeV at
  r = 0.02 and 0.3), each with its compensation-only control at r = 0.02 and 0.3. Prediction if the call counts particles regardless
  of position: the response is roughly r-independent. Prediction if the count enters only through wide-angle / empty-slot objects
  (S1 = sum (r - 0.2)_+, pT-independent; S30 = sum ReLU(3.47 - 21 r), 3.38 per empty slot): response small for r < 0.165 and rising
  beyond. Also the soft-split response stratified by the parent's r (< 0.165, 0.165-0.2, > 0.2).
Holds: every live S single (19), S19+S0, at unedited values; share removed = 1 - sum dmargin(hold) / sum dmargin(edit).
Statistics: 1000 bootstrap resamples over jets; sign-flip permutation (5000) for paired differences; exact McNemar on calls.
Run: ./job.sh e40-verify 'python -m interp.e40_verify'   writes $AI_RESULTS/jedi/e40_verify.json
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp.e28w_split_route_jedi import Net, nreal, rotate, mcnemar, qci, LIVE_S, binom_p  # noqa: E402
from interp.e40_qg_count_jedi import (split_gen, emit, merge_soft, null_resort, jet4, mass, Boot, signflip_p,  # noqa: E402
                                      ols_slope)

RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); OUT = RES / 'jedi'
T0_ = time.time()


def log(*a): print(f'[{time.time() - T0_:7.0f}s]', *a, flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from interp.jedi_core import sample
    B = 1000; rng = np.random.default_rng(4540); net = Net()
    xf, yf, _ = sample(1500, 4501)
    lf = net.fwd(xf, keep=('logits',))['logits']; accf = float((lf.argmax(1) == yf).mean())
    q = yf <= 1; mf = lf[q, 0] - lf[q, 1]; Nf = nreal(xf)[q].astype(float); cf = yf[q]
    sl = ols_slope(mf, Nf); slw = ols_slope(mf, Nf, cf); rb = np.random.default_rng(7); bsl = []
    for _ in range(B):
        i = rb.integers(0, len(mf), len(mf)); bsl.append(ols_slope(mf[i], Nf[i]))
    bsl = np.array(bsl)
    log(f'fit 4501 acc {accf:.4f}; slope pooled {sl:.4f} {qci(bsl)} within {slw:.4f}')
    del xf, lf; gc.collect()

    xa, ya, _ = sample(1500, 4502)
    la = net.fwd(xa, keep=('logits',))['logits']; acc = float((la.argmax(1) == ya).mean())
    sel = ya <= 1; X0 = xa[sel].copy(); y = ya[sel]; del xa, la; gc.collect()
    J = len(X0); N0 = nreal(X0)
    R0 = net.fwd(X0, keep=('logits', 'S')); L0 = R0['logits']; S0v = R0['S']; c0 = L0.argmax(1); m0 = L0[:, 0] - L0[:, 1]
    tail_dev = float(np.abs(net.tail(S0v) - L0).max())
    log(f'test 4502 acc {acc:.4f}; q/g jets {J}; tail max |dlogit| {tail_dev:.2e}')
    bt = Boot(J, B, 11); mom0 = jet4(X0)
    res = dict(seeds=dict(fit=4501, test=4502, edit_rng=4540), per_class=1500, acc_fit=accf, acc_test=acc, tail_dev=tail_dev,
               slope_pooled=sl, slope_pooled_ci=qci(bsl), slope_within=slw, n_qg=J, margin_sd=float(m0.std()))
    E, D, C, OK, SE = {}, {}, {}, {}, {}

    def rec(name, Xe, ok, dN, S=None, logits=None, keepS=False):
        if logits is None:
            R = net.fwd(Xe, keep=('logits', 'S')); Le, Se = R['logits'], R['S']
        else:
            Le, Se = logits, S
        ce = Le.argmax(1); d = (Le[:, 0] - Le[:, 1]) - m0; d[~ok] = np.nan; s = np.flatnonzero(ok)
        bm = bt.mean(np.nan_to_num(d), s)
        o = dict(n=len(s), dN=dN, dmargin=float(d[s].mean()), ci=qci(bm), perm_p=signflip_p(d[s], 5000, 1),
                 calls_changed=float((ce != c0)[s].mean()), q_to_g=int(((c0 == 1) & (ce == 0))[s].sum()),
                 g_to_q=int(((c0 == 0) & (ce == 1))[s].sum()),
                 dmargin_g=float(d[s][y[s] == 0].mean()) if (y[s] == 0).any() else None,
                 dmargin_q=float(d[s][y[s] == 1].mean()) if (y[s] == 1).any() else None)
        if dN:
            o['R_N'] = float(o['dmargin'] / (sl * dN)); o['R_N_ci'] = qci(bm / (bsl * dN))
        if Xe is not None:
            o['max_rel_jet4'] = float((np.abs(jet4(Xe) - mom0).max(1) / mom0[:, 0])[ok].max())
            o['dN_obs'] = float((nreal(Xe) - N0)[ok].mean())
        E[name] = o; D[name] = d; C[name] = ce; OK[name] = ok
        if keepS: SE[name] = Se
        log(f'{name}: n {o["n"]} d {o["dmargin"]:+.4f} {o["ci"]} calls {o["calls_changed"]:.4f} q->g {o["q_to_g"]} g->q {o["g_to_q"]}'
            + (f' R_N {o["R_N"]:.2f}' if dN else ''))

    rec('null_resort', null_resort(X0, rng), np.ones(J, bool), 0)
    soft_rank = np.array([rng.integers(30, n) if n >= 31 else -1 for n in N0])
    slots = {'r0': np.where((N0 >= 2) & (N0 <= 63), 0, -1), 'r13': np.where((N0 >= 15) & (N0 <= 63), 13, -1),
             'soft': np.where((N0 >= 31) & (N0 <= 63), soft_rank, -1)}
    for loc, slot in slots.items():
        ok = slot >= 0
        Ys, d1, d2 = split_gen(X0, slot, 0.5, 0.0, rng); okk = ok & (d1 >= 0) & (d2 >= 0)
        rec(f'split_{loc}', Ys, okk, 1, keepS=loc != 'r0')
        Yn, _, _ = split_gen(X0, slot, 0.5, 0.05, rng); rec(f'nc_{loc}', Yn, okk, 1)
        if loc != 'r0':
            Sp, chk = net.rank_patch(X0, Ys, slot, d1, d2, 'PS')
            rec(f'rankpres_{loc}', None, okk, 1, S=Sp, logits=net.tail(Sp))
            E[f'rankpres_{loc}']['check_max_abs_dS'] = float(np.abs(chk - SE[f'split_{loc}'])[okk].max())
        if loc == 'soft':
            rec('rot_soft', rotate(X0, slot, rng), ok, 0)
            rr = np.full(J, np.nan); rows = np.flatnonzero(ok); rr[rows] = np.hypot(X0[rows, slot[rows], 1], X0[rows, slot[rows], 2])
            res['soft_parent_r'] = rr
    # emissions: radius scan
    for k, pt, r in [(1, 0.5, 0.3), (3, 0.5, 0.3), (3, 0.05, 0.3), (3, 0.5, 0.02), (3, 0.05, 0.02), (3, 0.5, 0.1), (3, 0.5, 0.15),
                     (3, 0.5, 0.5)]:
        nm = f'emit_k{k}_pt{pt}_r{r}'; seed = 100 + k + int(1000 * r)
        Ye, oke, info = emit(X0, k, pt, r, np.random.default_rng(seed))
        rec(nm, Ye, oke, k, keepS=(k == 3)); E[nm].update(info)
        if pt == 0.5 and k == 3 and r in (0.02, 0.3):
            Yc, okc, infc = emit(X0, k, pt, r, np.random.default_rng(seed), add=False, ok_in=oke)
            rec(f'comp_k3_r{r}', Yc, okc, 0); E[f'comp_k3_r{r}'].update(infc)
    Ym, okm = merge_soft(X0); rec('merge2', Ym, okm, -1, keepS=True)
    Yv, okv = merge_soft(X0, move_only=True); rec('move2', Yv, okv, 0)
    del Ys, Yn, Ye, Ym, Yv; gc.collect()

    # paired comparisons
    pairs = [('split_soft', 'nc_soft'), ('split_r13', 'nc_r13'), ('split_r0', 'nc_r0'), ('split_soft', 'rankpres_soft'),
             ('split_r13', 'rankpres_r13'), ('split_soft', 'rot_soft'), ('merge2', 'move2'),
             ('emit_k3_pt0.5_r0.3', 'comp_k3_r0.3'), ('emit_k3_pt0.5_r0.02', 'comp_k3_r0.02'),
             ('emit_k3_pt0.5_r0.3', 'emit_k3_pt0.05_r0.3'), ('emit_k3_pt0.5_r0.3', 'emit_k3_pt0.5_r0.02'),
             ('emit_k3_pt0.05_r0.3', 'emit_k3_pt0.05_r0.02'), ('emit_k3_pt0.5_r0.02', 'emit_k3_pt0.05_r0.02')]
    comp = {}
    for a_, b_ in pairs:
        s = np.flatnonzero(OK[a_] & OK[b_]); dd = (D[a_] - D[b_]).astype(np.float64)
        comp[f'{a_}|{b_}'] = dict(n=len(s), diff=float(dd[s].mean()), ci=qci(bt.mean(np.nan_to_num(dd), s)),
                                  perm_p=signflip_p(dd[s], 5000, 2), ratio=float(D[a_][s].mean() / D[b_][s].mean()),
                                  mcnemar_calls=mcnemar((C[a_] != c0)[s], (C[b_] != c0)[s]))
        log(f'{a_} vs {b_}: diff {comp[f"{a_}|{b_}"]["diff"]:+.4f} {comp[f"{a_}|{b_}"]["ci"]} p {comp[f"{a_}|{b_}"]["perm_p"]:.3g}')
    # soft split by parent radius
    rr = res.pop('soft_parent_r'); strat = {}
    for lo, hi in [(0, 0.165), (0.165, 0.2), (0.2, 9)]:
        for nm in ('split_soft', 'nc_soft', 'rankpres_soft'):
            s = np.flatnonzero(OK[nm] & (rr >= lo) & (rr < hi))
            if len(s): strat[f'{nm}_r{lo}-{hi}'] = dict(n=len(s), dmargin=float(D[nm][s].mean()), ci=qci(bt.mean(np.nan_to_num(D[nm]), s)))
    res['soft_split_by_parent_r'] = strat; log('soft split by parent r:', {k: (v['n'], round(v['dmargin'], 4)) for k, v in strat.items()})

    # holds
    holds = {}
    for en in SE:
        s = np.flatnonzero(OK[en]); dE = np.nan_to_num(D[en]).astype(np.float64); tot = dE[s].sum(); H = dict(n=len(s), total=float(tot))

        def held(units):
            m = np.zeros(64, np.float32); m[units] = 1
            lh = net.tail(SE[en][s], m, S0v[s]); dH = np.zeros(J); dH[s] = (lh[:, 0] - lh[:, 1]) - m0[s]; return dH
        sing = {}
        for u in LIVE_S:
            dH = held([u]); sing[f'S{u}'] = float(1 - dH[s].sum() / tot)
            if u in (1, 19, 30, 0, 29): H[f'S{u}_ci'] = qci(bt.share(dE, dH, s))
        H['singles'] = sing; dH = held([19, 0]); H['S19+S0'] = float(1 - dH[s].sum() / tot)
        dH = held([1, 30]); H['S1+S30'] = float(1 - dH[s].sum() / tot); H['S1+S30_ci'] = qci(bt.share(dE, dH, s))
        H['single_q95'] = float(np.quantile(list(sing.values()), .95))
        holds[en] = H
        log(f'holds {en}: ' + ', '.join(f'{k} {v:+.2f}' for k, v in sorted(sing.items(), key=lambda t: -t[1])[:5]) +
            f'; S19+S0 {H["S19+S0"]:+.2f}; S1+S30 {H["S1+S30"]:+.2f}')
    res['edits'] = E; res['compare'] = comp; res['holds'] = holds
    (OUT / 'e40_verify.json').write_text(json.dumps(res, indent=1)); log('done', OUT / 'e40_verify.json')


if __name__ == '__main__':
    main()
