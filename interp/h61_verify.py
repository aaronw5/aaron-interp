"""H61 verification (job d-h61-verify*): fresh-event rerun of the Delphes, MPI-off V0 / Hz+ / split / merge comparison of H61,
plus three controls H61 did not run.

Fresh sample: the H61 generator and Delphes chain (interp.h61_top_delphes.gen_worker, unchanged) with worker seeds 6201-6210
(H61 used 6101-6120), written to data/top/gen_h61v/{V0,Hzp,split,merge,...}.npz.  Pairs are matched by the same key
(worker, class, event, slot).  split / merge are rebuilt by H61.build_edits on the fresh pairs (same k = max(N_Hz+ - N_V0, 0)).

Controls added here (all on the V0 jet, same k as the split, same pairs):
  jitter     the k softest objects are each moved by 0.005 in (eta, phi) in a random direction, pT kept: the same particles and
             the same angular displacement as one half of a split, but N unchanged (tests sensitivity to small moves of soft objects)
  split_rand the split applied to k objects drawn at random from the whole jet (not the softest): N rises by k, IRC-safe content fixed
  split_d02  the split with the two halves 0.02 apart instead of 0.01 (dependence on the arbitrary separation)
Steps:  gen --setting V0|Hzp (env-gen, 1 CPU)   score (GPU, env-part; scores, observables and statistics)
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from interp import h61_top_delphes as H61
from interp import h60_top_hadronisation as H60

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
V = Path('/j-jepa-vol/aaron-interp')
RES = Path(os.environ.get('AI_RESULTS', V / 'results')) / 'physics'
GENV = V / 'data' / 'top' / 'gen_h61v'
H61.GEN = GENV
H61.WSEEDS = lambda s: list(range(6201, 6211))


def jitter_soft(x, k, rng, delta=0.01):
    n = len(x); k = int(min(k, n - 1))
    if k <= 0: return x.copy()
    ch = np.argsort(x[:, 0], kind='stable')[:k]; th = rng.uniform(0, 2 * np.pi, k); y = x.copy()
    y[ch, 1] += 0.5 * delta * np.cos(th); y[ch, 2] += 0.5 * delta * np.sin(th)
    return y[np.argsort(-y[:, 0], kind='stable')]


def split_random(x, k, rng, delta=0.01):
    n = len(x); k = int(min(k, n - 1))
    if k <= 0: return x.copy()
    ch = rng.choice(n, k, replace=False); th = rng.uniform(0, 2 * np.pi, k)
    a = x[ch].copy(); a[:, 0] /= 2; b = a.copy()
    a[:, 1] += 0.5 * delta * np.cos(th); a[:, 2] += 0.5 * delta * np.sin(th)
    b[:, 1] -= 0.5 * delta * np.cos(th); b[:, 2] -= 0.5 * delta * np.sin(th)
    y = np.r_[np.delete(x, ch, 0), a, b]; return y[np.argsort(-y[:, 0], kind='stable')]


def build_controls():
    G0, G1 = H61.load_gen('V0'), H61.load_gen('Hzp')
    pos1 = {int(k): j for j, k in enumerate(G1['key'])}
    pr = np.array([(j, pos1[int(k)]) for j, k in enumerate(G0['key']) if int(k) in pos1])
    k = np.maximum(G1['n'][pr[:, 1]] - G0['n'][pr[:, 0]], 0)
    for name, f, seed in (('jitter', lambda x, kk, r: jitter_soft(x, kk, r), 6171), ('split_rand', lambda x, kk, r: split_random(x, kk, r), 6172),
                          ('split_d02', lambda x, kk, r: H61.split_soft(x, kk, r, delta=0.02), 6173)):
        rng = np.random.default_rng(seed); L = [f(H61._rows(G0, i0), kk, rng) for (i0, _), kk in zip(pr, k)]
        np.savez_compressed(GENV / f'{name}.npz', **H61._pack(L, G0, pr[:, 0])); log('wrote', name, len(L))


def step_gen(a):
    H61.step_gen(a)


SAMP = ['V0', 'Hzp', 'split', 'merge', 'jitter', 'split_rand', 'split_d02']
NETS = ['ParT_top', 'PN_top', 'LNet_top', 'PFN_top', 'EFN_top', 'ParT_kin', 'PN_kin', 'perm0']
TESTNETS = ['ParT_top', 'PN_top', 'LNet_top', 'PFN_top', 'ParT_kin', 'PN_kin']
OBS = ['N', 'm', 'tau32', 'mW', 'pTD']


def step_score(a):
    import torch
    from sklearn.metrics import roc_auc_score
    from interp import h4_top_wmass as H
    from interp import h19_wpair_cross as X
    RES.mkdir(parents=True, exist_ok=True)
    H61.build_edits(); build_controls()
    M = X.Models(); M.massless = True
    perm = H.WeaverNet('ParT', X.RES / 'h4_ckpt_part_kin.pt', M.dev); g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for prm in perm.model.parameters():
            fl = prm.data.view(-1); fl.copy_(fl[torch.randperm(fl.numel(), generator=g).to(fl.device)])
    S, O, Y, K = {}, {}, {}, {}
    for s in SAMP:
        G = H61.load_gen(s); P, n = H60.massless_P4(G); order = np.argsort((G['key'] * 2654435761) % 1_000_003, kind='stable')
        ob = H._obs(P, n); ob.update(H61.pt_feats(P, n)); O[s] = {k: np.asarray(ob[k], np.float64) for k in OBS if k != 'N'}; O[s]['N'] = n.astype(float)
        sc = M.score('top', P, n, X.top_base(P, n), order); sc['perm0'] = perm.score(P, n)
        S[s] = {m: np.asarray(sc[m], np.float64) for m in NETS}; Y[s] = G['y'].astype(int); K[s] = G['key'].astype(np.int64)
        log(f'{s}: {len(n)} jets <N> top {n[Y[s] == 1].mean():.1f} QCD {n[Y[s] == 0].mean():.1f} | ' +
            '; '.join(f'{m} acc {np.mean((S[s][m] > 0) == (Y[s] == 1)):.4f} auc {roc_auc_score(Y[s], S[s][m]):.4f}' for m in NETS))
    np.savez_compressed(RES / 'h61_verify_scores.npz', **{f'{s}|{m}': v for s in SAMP for m, v in S[s].items()},
                        **{f'y|{s}': Y[s] for s in SAMP}, **{f'key|{s}': K[s] for s in SAMP}, **{f'obs|{s}|{k}': v for s in SAMP for k, v in O[s].items()})
    stats(S, O, Y, K, a.boot)


def stats(S, O, Y, K, B):
    rng = np.random.default_rng(6164)
    R = dict(node='H61-verify', workers='6201-6210 (fresh events; H61 used 6101-6120)', accuracy={}, paired={}, share={}, tests=[], wp={})
    for s in S:
        from sklearn.metrics import roc_auc_score
        R['accuracy'][s] = {m: dict(acc=float(np.mean((S[s][m] > 0) == (Y[s] == 1))), auc=float(roc_auc_score(Y[s], S[s][m]))) for m in NETS}
    pos0 = {int(k): j for j, k in enumerate(K['V0'])}; mi = {m: i for i, m in enumerate(NETS)}
    for v in SAMP[1:]:
        pr = np.array([(pos0[int(k)], j) for j, k in enumerate(K[v]) if int(k) in pos0]); R['paired'][v] = {}
        for cls, cn in ((1, 'top'), (0, 'qcd')):
            sel = Y['V0'][pr[:, 0]] == cls; a0, a1 = pr[sel, 0], pr[sel, 1]; J = len(a0)
            D = np.stack([S[v][m][a1] - S['V0'][m][a0] for m in NETS], 1); Do = np.stack([O[v][k][a1] - O['V0'][k][a0] for k in OBS], 1)
            idx = rng.integers(0, J, (B, J)); bD = np.stack([D[ix].mean(0) for ix in idx]); bO = np.stack([Do[ix].mean(0) for ix in idx]); d = D.mean(0)
            r = dict(n_pairs=int(J), obs={k: dict(d=float(Do[:, i].mean()), ci=np.percentile(bO[:, i], [2.5, 97.5]).tolist(),
                                                  rel=float(Do[:, i].mean() / O['V0'][k][a0].mean())) for i, k in enumerate(OBS)}, models={})
            e = mi['EFN_top']
            for m in NETS:
                i = mi[m]; rr = dict(d=float(d[i]), ci=np.percentile(bD[:, i], [2.5, 97.5]).tolist(), p0=H60._zp(d[i], bD[:, i].std()))
                if m != 'EFN_top':
                    dd = d[i] - d[e]; bdd = bD[:, i] - bD[:, e]
                    rr['minus_EFN'] = dict(d=float(dd), ci=np.percentile(bdd, [2.5, 97.5]).tolist(), p=H60._zp(dd, bdd.std()))
                    if m in TESTNETS: R['tests'].append(dict(variant=v, cls=cn, model=m, d=float(dd), p=rr['minus_EFN']['p']))
                r['models'][m] = rr
            R['paired'][v][cn] = r
            th = {m: np.quantile(S['V0'][m][pr[Y['V0'][pr[:, 0]] == 1, 0]], 0.5) for m in NETS}
            R['wp'].setdefault(v, {})[cn] = {m: dict(zip(('gained', 'lost', 'p_mcnemar'), H60._mcnemar(S['V0'][m][a0] > th[m], S[v][m][a1] > th[m])),
                                                     eff_v0=float(np.mean(S['V0'][m][a0] > th[m])), eff_var=float(np.mean(S[v][m][a1] > th[m]))) for m in NETS}
            log(v, cn, f"pairs {J} dN {r['obs']['N']['d']:+.2f} dm {r['obs']['m']['d']:+.2f} dtau32 {r['obs']['tau32']['d']:+.4f} dmW {r['obs']['mW']['d']:+.2f} | "
                + ' '.join(f"{m} {r['models'][m]['d']:+.3f} [{r['models'][m]['ci'][0]:+.3f},{r['models'][m]['ci'][1]:+.3f}]" for m in NETS))
    adj = H60._holm([t['p'] for t in R['tests']])
    for t, h in zip(R['tests'], adj): t['holm'] = float(h)
    R['holm'] = dict(n_tests=len(adj), n_pass_005=int((adj < 0.05).sum()))
    # shares on the same V0 keys: split / Hz+, merge removed, controls / Hz+, and per added object (divided by dN ratio)
    ph = {int(k): j for j, k in enumerate(K['Hzp'])}; pe = {v: {int(k): j for j, k in enumerate(K[v])} for v in SAMP[2:]}
    for cls, cn in ((1, 'top'), (0, 'qcd')):
        rows = [(i0, ph[int(k)], *[pe[v][int(k)] for v in SAMP[2:]]) for i0, k in enumerate(K['V0'])
                if Y['V0'][i0] == cls and int(k) in ph and all(int(k) in pe[v] for v in SAMP[2:])]
        rows = np.array(rows); J = len(rows); bi = rng.integers(0, J, (B, J)); R['share'][cn] = {}
        dNh = O['Hzp']['N'][rows[:, 1]] - O['V0']['N'][rows[:, 0]]
        for m in NETS:
            dh = S['Hzp'][m][rows[:, 1]] - S['V0'][m][rows[:, 0]]; bh = np.array([dh[b].mean() for b in bi]); rr = dict(n=int(J), hzp=float(dh.mean()))
            for c, v in enumerate(SAMP[2:]):
                dv = S[v][m][rows[:, 2 + c]] - S['V0'][m][rows[:, 0]]; bv = np.array([dv[b].mean() for b in bi])
                dNv = O[v]['N'][rows[:, 2 + c]] - O['V0']['N'][rows[:, 0]]
                if v == 'merge':
                    rr['merge_removed'] = float(1 - dv.mean() / dh.mean()); rr['merge_removed_ci'] = np.percentile(1 - bv / bh, [2.5, 97.5]).tolist()
                else:
                    rr[v] = float(dv.mean() / dh.mean()); rr[f'{v}_ci'] = np.percentile(bv / bh, [2.5, 97.5]).tolist()
                    if abs(dNv.mean()) > 0.5: rr[f'{v}_per_object'] = float((dv.mean() / dNv.mean()) / (dh.mean() / dNh.mean()))
            R['share'][cn][m] = rr
        log('share', cn, {m: {k: (round(x, 2) if isinstance(x, float) else x) for k, x in v.items() if not k.endswith('_ci')} for m, v in R['share'][cn].items()})
    (RES / 'h61_verify.json').write_text(json.dumps(R, indent=1, default=float)); log('wrote h61_verify.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', required=True, choices=['gen', 'score'])
    ap.add_argument('--setting', default='V0'); ap.add_argument('--jets', type=int, default=5000); ap.add_argument('--boot', type=int, default=2000)
    a = ap.parse_args()
    dict(gen=step_gen, score=step_score)[a.step](a)
