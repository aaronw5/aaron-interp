"""E35v (skeptic of E35): rerun of E35's key measurements on a fresh sample plus three checks E35 did not run.

Sample: jedi_core.sample(2000, 3601) (seed not used by E35: 3501 / 3502), W and Z jets only (4,000 jets).
Edits and controls: E35's make_edits (D(dc), cn, cphi, R; dc in {-0.3, -0.15, +0.15, +0.3}; inputs rebuilt about each jet's own axis).
Checks:
  0  data: real particles occupy the first N slots (padding at the end), so P4[j, :N] is the jet; invariants of D and cn re-measured
     independently (jet four-momentum, the two kT-prong masses under D, both prong four-momenta under cn).
  A  responses: mean Delta margin (margin = logit_W - logit_Z) per (dc, class, edit), 95 % jet bootstrap; D - cn sign-flip p; McNemar.
  B  holds (E35's exact fixed-point tail, same projection and clipping rule): per-cell share = 1 - mean(held)/mean(free) and E35's
     pooled share; per-cell nulls for the dc > 0 cells: all 19 single live-S holds, all 171 pairs, all 969 triples, 500 random 1-dim
     and 3-dim live-S directions; p = (1 + #null >= named)/(1 + #null).
  C  positive control of the hold method (missing in E35): the same holds applied to a real mass change (all angles x 0.90 and x 1.10,
     the 'dilation'); if a hold removes most of a real dilation's response, its failure to remove D's response is informative.
  D  learned low-dimensional holds (missing in E35): the mean standardised S displacement under D (k = 1) and the top-3 SVD directions
     (k = 3), fitted on a random half of the cell's jets and held on the other half; same for h4 (k = 1); cosine between D's mean h4
     displacement and the dilation direction; all-live-S hold (sanity, must remove 1).
  E  calibration: ridge probe of log m on standardised h4 (fit half / test half), probe read-out under D, matched dilation, R.
  F  mask program (E3m): share of network flips reproduced at dc +0.3.
Run: ./job.sh e35-verify 'python -m interp.e35v_verify'   (--smoke K: K jets per class)
Writes $AI_RESULTS/jedi/e35_verify[_smoke].json
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
from interp.e35_prong_sharing_route_jedi import (make_edits, edit_jet, kt2_labels, Tail, proj, unitQ, randQ, mask_program, BootW,  # noqa: E402
                                                 signflip_p, margin, jsonable, DCS, VAR)
from interp.e28w_split_route_jedi import mcnemar, holm, qci, LIVE_S  # noqa: E402
from interp.h45_prong_sharing import p4, ptetaphim  # noqa: E402

RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); OUT = RES / 'jedi'
T0 = time.time()
CL = {2: 'W', 3: 'Z'}


def log(*a): print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


def main(a):
    from interp.jedi_core import Jedi, sample
    sm = a.smoke; per = sm or 2000; seed = 3601; tag = 'e35_verify' + ('_smoke' if sm else ''); OUT.mkdir(parents=True, exist_ok=True)
    net = Jedi(64, '3-feature'); tail = Tail(net); log('network loaded')
    x, y, _ = sample(per_class=per, seed=seed); keepj = np.flatnonzero((y == 2) | (y == 3)); x = np.asarray(x[keepj], np.float32); y = y[keepj]; J = len(x)
    res = dict(node='E35v', seed=seed, N=J, per_class=per)
    # ---- 0 data checks
    real = x[..., 0] > 0; nn = real.sum(1); contig = bool((real == (np.arange(64)[None] < nn[:, None])).all())
    res['padding_contiguous'] = contig; res['n_ge3'] = int((nn >= 3).sum()); log(f'padding contiguous {contig}; N>=3 {res["n_ge3"]}/{J}')
    X, acc, accphi, info, LAB, nn2, minv = make_edits(x, seed, sm)
    rng = np.random.default_rng(3602); inv = dict(dm_prong_D=[], dp4_prong_cn=[], dp4_jet_D=[])
    el = np.flatnonzero(nn >= 3)
    for j in rng.choice(el, min(300, len(el)), replace=False):
        n = nn[j]; P = p4(x[j, :n, 0].astype(np.float64), x[j, :n, 1].astype(np.float64), x[j, :n, 2].astype(np.float64), 0.0); lab = LAB[j, :n]
        r, _ = edit_jet(P, lab, 0.3, np.random.default_rng([seed, int(j), 3]))
        if r is None: continue
        for k in (0, 1):
            inv['dm_prong_D'].append(abs(ptetaphim(r['D'][lab == k].sum(0))[3] - ptetaphim(P[lab == k].sum(0))[3]))
            inv['dp4_prong_cn'].append(float(np.abs(r['cn'][lab == k].sum(0) - P[lab == k].sum(0)).max()))
        inv['dp4_jet_D'].append(float(np.abs(r['D'].sum(0) - P.sum(0)).max() / P.sum(0)[3]))
    res['invariants'] = {k: float(np.max(v)) for k, v in inv.items()}; res['invariants']['dm_inv_GeV_absmax_D03'] = float(np.abs(minv[(0.3, 'D')] - minv['id']).max())
    log('invariants', res['invariants'])
    # ---- A responses
    keep = ('logits', 'S', 'h4', 'h5', 'g')
    R0 = net.run(X['id'], keep=keep); L0 = R0['logits']; c0 = L0.argmax(1); m0 = margin(L0)
    res['acc_rebuilt_WZ'] = float((c0 == y).mean())
    RE = {k: net.run(v, keep=keep if k[1] == 'D' else ('logits',)) for k, v in X.items() if k != 'id'}
    bt = BootW(J, 300 if sm else 1000, seed + 1); srng = np.random.default_rng(seed + 9); A = {}
    for dc in DCS:
        e = {}
        dD = margin(RE[(dc, 'D')]['logits']) - m0; chD = RE[(dc, 'D')]['logits'].argmax(1) != c0
        for kind in VAR:
            sel = acc[dc] & (accphi[dc] if kind == 'cphi' else True); Le = RE[(dc, kind)]['logits']; d = margin(Le) - m0; ce = Le.argmax(1)
            for c in (2, 3):
                s = np.flatnonzero((y == c) & sel)
                e[f'{kind}|{CL[c]}'] = dict(n=len(s), dmargin=float(d[s].mean()), ci=qci(bt.mean(d, s)), W_to_Z=float(((c0[s] == 2) & (ce[s] == 3)).mean()),
                                            Z_to_W=float(((c0[s] == 3) & (ce[s] == 2)).mean()))
                if kind != 'D':
                    dK = d; chK = ce != c0; df = dD - dK
                    e[f'D-{kind}|{CL[c]}'] = dict(mean=float(df[s].mean()), ci=qci(bt.mean(df, s)), p_signflip=signflip_p(df[s], srng, 2000), mcnemar=mcnemar(chD[s], chK[s]))
        A[str(dc)] = e
        log(f'dc {dc:+.2f}: ' + '; '.join(f'{k} {v["dmargin"]:+.3f} {np.round(v["ci"], 3).tolist()}' for k, v in e.items() if 'dmargin' in v))
    res['responses'] = A
    # ---- B holds
    sdS = R0['S'].std(0); sd4 = R0['h4'].std(0); sd5 = R0['h5'].std(0)
    live4 = [int(u) for u in np.flatnonzero(sd4 > 0)]; live5 = [int(u) for u in np.flatnonzero(sd5 > 0)]
    xd = X['id'].copy(); xd[..., 1:] *= np.float32(1.05); Rd = net.run(xd, keep=('h4', 'h5'))
    s4 = np.where(sd4 > 0, sd4, 1); s5 = np.where(sd5 > 0, sd5, 1)
    d4 = ((Rd['h4'] - R0['h4']) / s4).mean(0); d4[sd4 == 0] = 0; d4 /= np.linalg.norm(d4)
    d5 = ((Rd['h5'] - R0['h5']) / s5).mean(0); d5[sd5 == 0] = 0; d5 /= np.linalg.norm(d5)
    M4dir = proj(d4[:, None], sd4); M5dir = proj(d5[:, None], sd5)
    # dilation edits (positive control C)
    DIL = {}
    for f in (0.90, 1.10):
        xx = X['id'].copy(); xx[..., 1:] *= np.float32(f); DIL[f] = net.run(xx, keep=('logits', 'S', 'h4', 'h5'))
    allS = np.concatenate([R0['S']] + [RE[(dc, 'D')]['S'] for dc in DCS] + [DIL[f]['S'] for f in DIL])
    allh4 = np.concatenate([R0['h4']] + [RE[(dc, 'D')]['h4'] for dc in DCS] + [DIL[f]['h4'] for f in DIL])
    allh5 = np.concatenate([R0['h5']] + [RE[(dc, 'D')]['h5'] for dc in DCS] + [DIL[f]['h5'] for f in DIL])
    lo = np.zeros((3, 64), np.float32); hi = np.zeros((3, 64), np.float32)
    lo[0], hi[0] = allS.min(0), allS.max(0); lo[1], hi[1] = allh4.min(0), allh4.max(0); lo[2, :32], hi[2, :32] = allh5.min(0), allh5.max(0)

    def held_margin(SE, kw, idx=None):
        idx = np.arange(J) if idx is None else idx
        lo_, hi_ = np.full((3, 64), -1e9, np.float32), np.full((3, 64), 1e9, np.float32)
        for i, key in enumerate(('MS', 'M4', 'M5')):
            if kw.get(key) is not None and not (key == 'MS' and np.allclose(np.diag(np.diag(kw[key])), kw[key])): lo_[i], hi_[i] = lo[i], hi[i]
        return margin(tail(SE[idx], R0['S'][idx], kw.get('MS'), R0['h4'][idx], kw.get('M4'), R0['h5'][idx], kw.get('M5'), lo_, hi_)) - m0[idx]

    HOLDS = {f'S{u}': dict(MS=proj(unitQ([u]), sdS)) for u in (29, 1, 0, 19, 30, 31, 58, 60, 56)}
    HOLDS['named7'] = dict(MS=proj(unitQ([29, 1, 0, 19, 30, 31, 58]), sdS)); HOLDS['radial'] = dict(MS=proj(unitQ([1, 30, 19]), sdS))
    HOLDS['S29+S1'] = dict(MS=proj(unitQ([29, 1]), sdS)); HOLDS['S60+S51'] = dict(MS=proj(unitQ([60, 51]), sdS))
    HOLDS['h4dir'] = dict(M4=M4dir); HOLDS['h5dir'] = dict(M5=M5dir); HOLDS['mass'] = dict(MS=proj(unitQ([29]), sdS), M4=M4dir)
    HOLDS['allS'] = dict(MS=proj(unitQ(LIVE_S), sdS))
    CELLS = [(dc, c) for dc in DCS for c in (2, 3)]; CS = {k: np.flatnonzero(acc[k[0]] & (y == k[1])) for k in CELLS}
    FREE = {dc: margin(RE[(dc, 'D')]['logits']) - m0 for dc in DCS}
    res['tail_free_check'] = float(max(np.abs(held_margin(RE[(dc, 'D')]['S'], {}) - FREE[dc]).max() for dc in DCS))
    sgn = {k: float(np.sign(FREE[k[0]][CS[k]].mean())) for k in CELLS}
    hold_res = {}
    for nm, kw in HOLDS.items():
        H = {dc: held_margin(RE[(dc, 'D')]['S'], kw) for dc in DCS}
        pcell = {}
        for (dc, c) in CELLS:
            s = CS[(dc, c)]; pcell[f'{dc}|{CL[c]}'] = dict(share=float(1 - H[dc][s].mean() / FREE[dc][s].mean()),
                                                       ci=qci(1 - bt.mean(H[dc], s) / bt.mean(FREE[dc], s)) if dc > 0 else None)
        pooled = 1 - sum(sgn[k] * H[k[0]][CS[k]].mean() for k in CELLS) / sum(sgn[k] * FREE[k[0]][CS[k]].mean() for k in CELLS)
        pooled_pos = 1 - sum(sgn[k] * H[k[0]][CS[k]].mean() for k in CELLS if k[0] > 0) / sum(sgn[k] * FREE[k[0]][CS[k]].mean() for k in CELLS if k[0] > 0)
        hold_res[nm] = dict(pooled=float(pooled), pooled_dcpos=float(pooled_pos), per_cell=pcell)
        log(f'hold {nm}: pooled {pooled:+.3f} dc>0 pooled {pooled_pos:+.3f}; ' + ', '.join(f'{k} {v["share"]:+.2f}' for k, v in pcell.items()))
    res['holds'] = hold_res
    # per-cell nulls (dc > 0)
    POS = [(dc, c) for dc in DCS if dc > 0 for c in (2, 3)]; nr = np.random.default_rng(seed + 3); nnull = 100 if sm else 500
    def cell_shares(kw):
        out = {}
        for dc in (0.15, 0.3):
            idx = np.concatenate([CS[(dc, 2)], CS[(dc, 3)]]); h = held_margin(RE[(dc, 'D')]['S'], kw, idx); n2 = len(CS[(dc, 2)])
            for c, hh in ((2, h[:n2]), (3, h[n2:])):
                out[f'{dc}|{CL[c]}'] = float(1 - hh.mean() / FREE[dc][CS[(dc, c)]].mean())
        return out
    t = time.time(); NUL = {}
    NUL['singles'] = {f'S{u}': cell_shares(dict(MS=proj(unitQ([u]), sdS))) for u in LIVE_S}
    NUL['pairs'] = {f'S{p[0]}+S{p[1]}': cell_shares(dict(MS=proj(unitQ(list(p)), sdS))) for p in itertools.combinations(LIVE_S, 2)}
    tri = list(itertools.combinations(LIVE_S, 3))
    if sm: tri = [tri[i] for i in nr.choice(len(tri), 100, replace=False)]
    NUL['triples'] = [cell_shares(dict(MS=proj(unitQ(list(p)), sdS))) for p in tri]
    NUL['dirs1'] = [cell_shares(dict(MS=proj(randQ(nr, LIVE_S, 1), sdS))) for _ in range(nnull)]
    NUL['dirs3'] = [cell_shares(dict(MS=proj(randQ(nr, LIVE_S, 3), sdS))) for _ in range(nnull)]
    NUL['h4dirs'] = [cell_shares(dict(M4=proj(randQ(nr, live4, 1), sd4))) for _ in range(nnull)]
    log(f'per-cell nulls in {time.time() - t:.0f}s')
    cellnull = {}
    for (dc, c) in POS:
        k = f'{dc}|{CL[c]}'; sg = [v[k] for v in NUL['singles'].values()]; pr = [v[k] for v in NUL['pairs'].values()]; tr = [v[k] for v in NUL['triples']]
        top_s = sorted(NUL['singles'], key=lambda u: -NUL['singles'][u][k])[:3]; top_p = sorted(NUL['pairs'], key=lambda u: -NUL['pairs'][u][k])[:3]
        top_t = sorted(range(len(tri)), key=lambda i: -tr[i])[:3]
        pv = lambda val, nv: float((1 + (np.asarray(nv) >= val).sum()) / (1 + len(nv)))
        cellnull[k] = dict(top_singles=[(u, NUL['singles'][u][k]) for u in top_s], top_pairs=[(u, NUL['pairs'][u][k]) for u in top_p],
                           top_triples=[('+'.join(f'S{q}' for q in tri[i]), tr[i]) for i in top_t],
                           n_pairs_ge_0p5=int((np.asarray(pr) >= 0.5).sum()), n_triples_ge_0p5=int((np.asarray(tr) >= 0.5).sum()), n_triples=len(tr),
                           q95=dict(singles=float(np.quantile(sg, .95)), pairs=float(np.quantile(pr, .95)), triples=float(np.quantile(tr, .95)),
                                    dirs1=float(np.quantile([v[k] for v in NUL['dirs1']], .95)), dirs3=float(np.quantile([v[k] for v in NUL['dirs3']], .95)),
                                    h4dirs=float(np.quantile([v[k] for v in NUL['h4dirs']], .95))),
                           p=dict(S1_vs_singles=pv(hold_res['S1']['per_cell'][k]['share'], sg), radial_vs_triples=pv(hold_res['radial']['per_cell'][k]['share'], tr),
                                  radial_vs_dirs3=pv(hold_res['radial']['per_cell'][k]['share'], [v[k] for v in NUL['dirs3']]),
                                  S29_vs_dirs1=pv(hold_res['S29']['per_cell'][k]['share'], [v[k] for v in NUL['dirs1']]),
                                  h4dir_vs_h4dirs=pv(hold_res['h4dir']['per_cell'][k]['share'], [v[k] for v in NUL['h4dirs']])))
        log(f'cell {k}: top singles {cellnull[k]["top_singles"]}; top pairs {cellnull[k]["top_pairs"]}; top triples {cellnull[k]["top_triples"]}; pairs>=0.5 {cellnull[k]["n_pairs_ge_0p5"]}; p {cellnull[k]["p"]}')
    fam = [(k, nm) for k in cellnull for nm in cellnull[k]['p']]; ph = holm([cellnull[k]['p'][nm] for k, nm in fam])
    for (k, nm), p in zip(fam, ph): cellnull[k].setdefault('p_holm', {})[nm] = float(p)
    res['cell_nulls'] = cellnull
    # ---- C positive control: holds on a real dilation
    pc = {}
    for f in DIL:
        dfree = margin(DIL[f]['logits']) - m0; r = {'free': {CL[c]: float(dfree[y == c].mean()) for c in (2, 3)}, 'changed': {CL[c]: float((DIL[f]['logits'].argmax(1) != c0)[y == c].mean()) for c in (2, 3)}}
        for nm in ('S29', 'S1', 'radial', 'named7', 'h4dir', 'h5dir', 'mass', 'S29+S1', 'allS'):
            h = held_margin(DIL[f]['S'], HOLDS[nm])
            r[nm] = {CL[c]: dict(share=float(1 - h[y == c].mean() / dfree[y == c].mean()), ci=qci(1 - bt.mean(h, np.flatnonzero(y == c)) / bt.mean(dfree, np.flatnonzero(y == c)))) for c in (2, 3)}
        pc[str(f)] = r; log(f'dilation x{f}: free {r["free"]}; ' + '; '.join(f'{nm} W {r[nm]["W"]["share"]:+.2f} Z {r[nm]["Z"]["share"]:+.2f}' for nm in r if nm not in ('free', 'changed')))
    res['positive_control_dilation'] = pc
    # ---- D learned low-dim holds (fit half / test half per cell)
    lr = {}; rp = np.random.default_rng(seed + 5)
    for (dc, c) in POS:
        s = rp.permutation(CS[(dc, c)]); fit, tst = s[:len(s) // 2], s[len(s) // 2:]; k = f'{dc}|{CL[c]}'
        SE = RE[(dc, 'D')]['S']; dS = (SE[fit] - R0['S'][fit]) / np.where(sdS > 0, sdS, 1); dS[:, sdS == 0] = 0
        u = dS.mean(0); Q1 = (u / np.linalg.norm(u))[:, None]; Q3 = np.linalg.svd(dS, full_matrices=False)[2][:3].T
        dh4 = (RE[(dc, 'D')]['h4'][fit] - R0['h4'][fit]) / s4; dh4[:, sd4 == 0] = 0; v4 = dh4.mean(0); v4 /= np.linalg.norm(v4)
        fr = FREE[dc][tst]; r = dict(n_fit=len(fit), n_test=len(tst), free_test=float(fr.mean()), cos_Dh4_dilation_h4=float(v4 @ d4),
                                     top_units_meanS=[(f'S{int(q)}', float(Q1[q, 0])) for q in np.argsort(-np.abs(Q1[:, 0]))[:6]])
        bt2 = BootW(len(tst), 300 if sm else 1000, seed + 6)
        for nm, kw in (('S_meandir_k1', dict(MS=proj(Q1, sdS))), ('S_svd_k3', dict(MS=proj(Q3, sdS))), ('h4_meandir_k1', dict(M4=proj(v4[:, None], sd4))),
                       ('h4dir_dilation', dict(M4=M4dir)), ('allS', HOLDS['allS'])):
            h = held_margin(SE, kw, tst); r[nm] = dict(share=float(1 - h.mean() / fr.mean()), ci=qci(1 - bt2.mean(h, np.arange(len(tst))) / bt2.mean(fr, np.arange(len(tst)))))
        n1 = [1 - held_margin(SE, dict(MS=proj(randQ(rp, LIVE_S, 1), sdS)), tst).mean() / fr.mean() for _ in range(200 if not sm else 30)]
        n3 = [1 - held_margin(SE, dict(MS=proj(randQ(rp, LIVE_S, 3), sdS)), tst).mean() / fr.mean() for _ in range(200 if not sm else 30)]
        r['null_dirs1_q95'] = float(np.quantile(n1, .95)); r['null_dirs3_q95'] = float(np.quantile(n3, .95))
        r['p_k1'] = float((1 + (np.asarray(n1) >= r['S_meandir_k1']['share']).sum()) / (1 + len(n1))); r['p_k3'] = float((1 + (np.asarray(n3) >= r['S_svd_k3']['share']).sum()) / (1 + len(n3)))
        lr[k] = r; log(f'learned {k}: ' + json.dumps(jsonable({q: v for q, v in r.items()})))
    res['learned_holds'] = lr
    # ---- E calibration (fit half / test half of these W+Z jets)
    rp2 = np.random.default_rng(seed + 4).permutation(J); fit, tst = rp2[:J // 2], rp2[J // 2:]
    lm = np.log(np.maximum(minv['id'], 1e-3)); mu, sd_ = R0['h4'][fit].mean(0), np.where(R0['h4'][fit].std(0) > 0, R0['h4'][fit].std(0), 1)
    Xf = np.c_[np.ones(len(fit)), (R0['h4'][fit] - mu) / sd_]; lam = 1e-3 * len(fit) * np.diag(np.r_[0.0, np.ones(64)])
    w = np.linalg.solve(Xf.T @ Xf + lam + 1e-9 * np.eye(65), Xf.T @ lm[fit]); pr = lambda h: np.c_[np.ones(len(h)), (h - mu) / sd_] @ w
    r2 = 1 - ((pr(R0['h4'][tst]) - lm[tst]) ** 2).mean() / lm[tst].var()
    gain = float(np.mean([(pr(DIL[f]['h4'][tst]) - pr(R0['h4'][tst])).mean() / math.log(f) for f in DIL]))
    cal = dict(probe_r2_test=float(r2), gain=gain, per={})
    for dc in (0.3, -0.3):
        for c in (2, 3):
            s = tst[acc[dc][tst] & (y[tst] == c)]; dlm = float((pr(RE[(dc, 'D')]['h4'][s]) - pr(R0['h4'][s])).mean()); k_ = math.exp(dlm / gain)
            xx = X['id'][s].copy(); xx[..., 1:] *= np.float32(k_); dd = margin(net.run(xx)['logits']) - m0[s]; dD = FREE[dc][s]
            cal['per'][f'{dc}|{CL[c]}'] = dict(n=len(s), dlogm_probe=dlm, k=k_, dmargin_D=float(dD.mean()), dmargin_dil=float(dd.mean()), R=float(dD.mean() / dd.mean()))
    res['calibration'] = cal; log('calibration', json.dumps(jsonable(cal)))
    # ---- F mask program
    prog = mask_program(net); P0 = prog(X['id']); P1 = prog(X[(0.3, 'D')]); mp = {}
    for c in (2, 3):
        s = np.flatnonzero((y == c) & acc[0.3]); chN = (RE[(0.3, 'D')]['logits'].argmax(1) != c0)[s]; chP = (P1.argmax(1) != P0.argmax(1))[s]
        mp[CL[c]] = dict(net_changed=int(chN.sum()), prog_changed=int(chP.sum()), net_flips_reproduced=float(chP[chN].mean()), mcnemar=mcnemar(chN, chP))
    res['mask_program_dc03'] = mp; log('mask program', mp)
    res['wall_s'] = time.time() - T0
    (OUT / f'{tag}.json').write_text(json.dumps(jsonable(res), indent=1)); log(f'wrote {OUT / tag}.json; E35v DONE')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', type=int, default=0); main(ap.parse_args())
