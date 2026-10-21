"""H60 verify (loop D skeptic): fresh-seed rerun of the H60 hadronisation-only shifts plus three controls missing from H60.

Fresh sample: the H60 generator code unchanged (interp.h60_top_hadronisation.gen_worker), seed base 7272 instead of 6060, settings
V0 (Monash), Hzp, Hzm; 4,000 top + 4,000 QCD jets each -> /j-jepa-vol/aaron-interp/data/top/gen_h60_verify/<S>.npz.

Representations / edits scored (all event-paired by key = (worker, event, slot), same random batch order for LorentzNet):
  raw     particle-level constituents, massless (as H60)
  det     crude detector proxy, applied identically to every setting: charged particles (|pid| in 11, 13, 211, 321, 2212, 3112,
          3222, 3312, 3334) with pT > 0.5 GeV kept one by one; every other particle (photons, neutral hadrons, charged below
          0.5 GeV) summed into 0.1 x 0.1 (eta, phi) cells, cell kept as one massless particle if its pT > 1 GeV.  Purpose: bring
          N to the Delphes range of the training data (Top Landscape mean N 54.9 top / 43.5 QCD) and test whether the H60 shift
          survives in-range N.  Not Delphes.
  split   V0 jet with k_j = max(N_Hzp,j - N_V0,j, 0) distinct random particles of its softer half each replaced by two pT/2 halves
          at +-delta/2 (delta = 0.01) about the parent in a random direction (H18v exact-split edit): count rises by the paired
          Hz+ amount at fixed IRC-safe content.  Prediction if the networks read only N at fixed energy flow: split shift = Hz+
          shift.
  merge   Hz+ jet with k_j = max(N_Hzp,j - N_V0,j, 0) nearest-in-DeltaR pairs among its softer half merged (three-momenta added,
          massless): count back to the V0 count by near-collinear recombination.  Prediction if N drives the shift: merge - V0 ~ 0.
Models: Top Landscape ParT, ParticleNet, LorentzNet (batch 32), PFN, EFN (IRC safe); JetClass ParT_kin, PN_kin; one weight-permuted
ParT (null).  LR(N) refit on Top Landscape val N (25k + 25k, rng 4194) as the count-only reference.
Stats: paired mean shift with 95 % bootstrap CI over pairs (B 2000), sign-flip permutation p (2000) for mean = 0 and for
net - EFN = 0; Holm across all tests; share = Delta s(split) / Delta s(Hz+) with paired bootstrap CI.  C2ST (HGB, 2-fold AUC) of
V0 raw / det vs Top Landscape test jets on (N, m, m_SD, tau21, tau32, mW).
Run: ./job.sh d-h60-verify-g<s> '/j-jepa-vol/aaron-interp/env-gen/bin/python -m interp.h60_verify --step gen --setting <S>'
     ./interp/h54_gpu_job.sh d-h60-verify 'python -m interp.h60_verify --step score'
Output: $AI_RESULTS/physics/h60_verify.json
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from interp import h60_top_hadronisation as H60

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
V = Path('/j-jepa-vol/aaron-interp')
RES = Path(os.environ.get('AI_RESULTS', V / 'results')) / 'physics'
GENV = V / 'data' / 'top' / 'gen_h60_verify'
SEEDV = 7272
SETS = ['V0', 'Hzp', 'Hzm']
CHARGED = {11, 13, 211, 321, 2212, 3112, 3222, 3312, 3334}
LMAX = 200


def step_gen(a):
    H60.SEED = SEEDV; H60.GEN = GENV
    H60.step_gen(argparse.Namespace(setting=a.setting, jets=a.jets))


# ------------------------------------------------------------------------------------------------ representations
def load(s):
    Z = np.load(GENV / f'{s}.npz'); return {k: Z[k] for k in Z.files}


def lists_raw(G):
    out = []
    for j in range(len(G['n'])):
        x = G['X'][j, :G['n'][j]].astype(np.float64); out.append(x[:, [0, 1, 2]].copy())
    return out, [G['X'][j, :G['n'][j], 4].astype(np.int64) for j in range(len(G['n']))]


def det_proxy(x, pid):
    pt, eta, phi = x[:, 0], x[:, 1], x[:, 2]
    ch = np.isin(np.abs(pid), list(CHARGED)) & (pt > 0.5)
    keep = x[ch]
    r = ~ch
    if r.any():
        px = pt[r] * np.cos(phi[r]); py = pt[r] * np.sin(phi[r]); pz = pt[r] * np.sinh(eta[r])
        ie = np.floor(eta[r] / 0.1).astype(np.int64); ip = np.floor(phi[r] / 0.1).astype(np.int64)
        cid = ie * 1000 + ip; u, inv = np.unique(cid, return_inverse=True)
        S = np.zeros((len(u), 3)); np.add.at(S, inv, np.c_[px, py, pz])
        cpt = np.hypot(S[:, 0], S[:, 1]); ok = cpt > 1.0
        cells = np.c_[cpt[ok], np.arcsinh(S[ok, 2] / cpt[ok]), np.arctan2(S[ok, 1], S[ok, 0])]
        keep = np.r_[keep, cells]
    return keep[np.argsort(-keep[:, 0], kind='stable')]


def split_edit(x, k, rng, delta=0.01):
    n = len(x); soft = np.arange(n // 2, n); k = int(min(k, len(soft)))
    if k <= 0: return x.copy()
    ch = rng.choice(soft, k, replace=False); th = rng.uniform(0, 2 * np.pi, k)
    a = x[ch].copy(); a[:, 0] /= 2; b = a.copy()
    a[:, 1] += 0.5 * delta * np.cos(th); a[:, 2] += 0.5 * delta * np.sin(th)
    b[:, 1] -= 0.5 * delta * np.cos(th); b[:, 2] -= 0.5 * delta * np.sin(th)
    y = np.r_[np.delete(x, ch, 0), a, b]
    return y[np.argsort(-y[:, 0], kind='stable')]


def merge_edit(x, k):
    n = len(x); k = int(k)
    if k <= 0: return x.copy()
    hard = x[: n // 2]; s = x[n // 2:].copy()
    p = np.c_[s[:, 0] * np.cos(s[:, 2]), s[:, 0] * np.sin(s[:, 2]), s[:, 0] * np.sinh(s[:, 1])]
    k = min(k, len(s) - 1)
    for _ in range(k):
        pt = np.hypot(p[:, 0], p[:, 1]); eta = np.arcsinh(p[:, 2] / pt); phi = np.arctan2(p[:, 1], p[:, 0])
        de = eta[:, None] - eta[None]; dp = (phi[:, None] - phi[None] + np.pi) % (2 * np.pi) - np.pi
        d = np.hypot(de, dp); np.fill_diagonal(d, np.inf); i, j = np.unravel_index(np.argmin(d), d.shape)
        p[i] = p[i] + p[j]; p = np.delete(p, j, 0)
    pt = np.hypot(p[:, 0], p[:, 1]); s2 = np.c_[pt, np.arcsinh(p[:, 2] / pt), np.arctan2(p[:, 1], p[:, 0])]
    y = np.r_[hard, s2]; return y[np.argsort(-y[:, 0], kind='stable')]


def to_P4(L):
    L = [x[:LMAX] for x in L]; n = np.array([len(x) for x in L]); P = np.zeros((len(L), max(n.max(), 1), 4), np.float32)
    for j, x in enumerate(L):
        pt, eta, phi = x[:, 0], x[:, 1], x[:, 2]
        P[j, :len(x)] = np.c_[pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), pt * np.cosh(eta)]
    return P, n.astype(np.int64)


# ------------------------------------------------------------------------------------------------ scoring + stats
def step_score(a):
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_predict
    from interp import data_top as dt
    from interp import h4_top_wmass as H
    from interp import h19_wpair_cross as X
    RES.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(SEEDV)
    M = X.Models(); M.massless = True
    perm = H.WeaverNet('ParT', X.RES / 'h4_ckpt_part_kin.pt', M.dev); g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for prm in perm.model.parameters():
            fl = prm.data.view(-1); fl.copy_(fl[torch.randperm(fl.numel(), generator=g).to(fl.device)])
    G = {s: load(s) for s in SETS}; raw = {s: lists_raw(G[s]) for s in SETS}
    key = {s: G[s]['key'].astype(np.int64) for s in SETS}; y = {s: G[s]['y'].astype(int) for s in SETS}
    # parton-level identity of pairs: stored truth (eta, phi) equal
    pos0 = {int(k): j for j, k in enumerate(key['V0'])}
    pairs = {}
    for s in ('Hzp', 'Hzm'):
        pr = np.array([(pos0[int(k)], j) for j, k in enumerate(key[s]) if int(k) in pos0]); pairs[s] = pr
    same_truth = {s: float(np.mean(np.all(np.isclose(G['V0']['truth'][pairs[s][:, 0]], G[s]['truth'][pairs[s][:, 1]], atol=1e-9), axis=(1, 2))))
                  for s in pairs}
    log('pairs', {s: len(v) for s, v in pairs.items()}, 'truth identical', same_truth)
    # representations
    reps = {}
    for s in SETS:
        L, pid = raw[s]; reps[f'{s}|raw'] = (L, key[s], y[s])
        reps[f'{s}|det'] = ([det_proxy(x, p) for x, p in zip(L, pid)], key[s], y[s])
    unk = {int(v) for s in SETS for p in raw[s][1] for v in np.unique(np.abs(p))} - CHARGED - {22, 130, 2112, 310, 3122, 3322, 12, 14, 16}
    log('det proxy built; pids not in charged/neutral lists:', sorted(unk))
    # edits paired on (V0, Hzp)
    pr = pairs['Hzp']; nV0 = G['V0']['n']; nHp = G['Hzp']['n']
    kk = np.maximum(nHp[pr[:, 1]] - nV0[pr[:, 0]], 0)
    LV, LH = raw['V0'][0], raw['Hzp'][0]; er = np.random.default_rng([SEEDV, 1])
    reps['V0|split'] = ([split_edit(LV[i0], k, er) for (i0, _), k in zip(pr, kk)], key['V0'][pr[:, 0]], y['V0'][pr[:, 0]])
    reps['Hzp|merge'] = ([merge_edit(LH[i1], k) for (_, i1), k in zip(pr, kk)], key['Hzp'][pr[:, 1]], y['Hzp'][pr[:, 1]])
    log('edits built; mean k', float(kk.mean()))
    # LR(N) on Top Landscape val N
    iv, yv = H60.tl_val_rows(); Nv = np.minimum(np.load(dt.DST / 'val' / 'jet.npy')[iv, 5], 200).astype(float)
    lrN = LogisticRegression().fit(Nv[:, None], yv[iv].astype(int))
    # Top Landscape test sample (rng 6061, as H60) for C2ST / N comparison
    it, yt = H60.tl_test_rows(); full = np.load(dt.DST / 'test' / 'full.npy'); Pt = np.asarray(full[it][..., :4], np.float32); del full
    nt = np.minimum(np.load(dt.DST / 'test' / 'jet.npy')[it, 5].astype(np.int64), Pt.shape[1]); Ot = H._obs(Pt, nt); ytt = yt[it].astype(int)
    S, O, info = {}, {}, {}
    for r, (L, k_, yy) in reps.items():
        P, n = to_P4(L); h = (k_ * 2654435761) % 1_000_003; order = np.argsort(h, kind='stable')
        t = time.time(); sc = M.score('top', P, n, X.top_base(P, n), order); sc['perm0'] = perm.score(P, n)
        sc['lrN'] = lrN.decision_function(n[:, None].astype(float)); S[r] = {m: np.asarray(v, np.float64) for m, v in sc.items()}
        O[r] = {kq: np.asarray(v, np.float64) for kq, v in H._obs(P, n).items() if kq in ('m', 'mSD', 'tau32', 'tau21', 'mW')}; O[r]['N'] = n.astype(float)
        info[r] = dict(jets=len(n), N_top=float(n[yy == 1].mean()), N_qcd=float(n[yy == 0].mean()), N_gt128=float(np.mean(n > 128)),
                       models={m: dict(acc=float(np.mean((v > 0) == (yy == 1))), auc=float(roc_auc_score(yy, v))) for m, v in S[r].items()})
        log(r, f'{time.time() - t:.0f}s', {kq: info[r][kq] for kq in ('N_top', 'N_qcd', 'N_gt128')},
            ' '.join(f"{m} {v['acc']:.3f}/{v['auc']:.4f}" for m, v in info[r]['models'].items()))
    models = list(S['V0|raw'])
    # C2ST vs Top Landscape test
    c2 = {}
    for rep in ('raw', 'det'):
        for cls, cn in ((1, 'top'), (0, 'qcd')):
            for feats, tag in ((['N', 'm', 'mSD', 'tau21', 'tau32', 'mW'], 'all'), (['m', 'mSD', 'tau21', 'tau32', 'mW'], 'no_N')):
                ot = dict(Ot, N=nt.astype(float))
                F0 = np.stack([ot[f][ytt == cls] for f in feats], 1); yy = y['V0']
                F1 = np.stack([O[f'V0|{rep}'][f][yy == cls] for f in feats], 1)
                Xc = np.r_[F0, F1]; yc = np.r_[np.zeros(len(F0)), np.ones(len(F1))]
                pc = cross_val_predict(HistGradientBoostingClassifier(max_iter=200, random_state=0), Xc, yc, cv=2, method='predict_proba')[:, 1]
                c2[f'{rep}_{cn}_{tag}'] = float(roc_auc_score(yc, pc))
    c2['TL_N'] = dict(top=float(nt[ytt == 1].mean()), qcd=float(nt[ytt == 0].mean()))
    log('C2ST', c2)
    # paired comparisons: (label, rep_a, rep_b, index pairs into a and b)
    idxV0 = {int(k): j for j, k in enumerate(key['V0'])}
    comps = []
    for s in ('Hzp', 'Hzm'):
        for rep in ('raw', 'det'):
            comps.append((f'{s}-V0|{rep}', f'V0|{rep}', f'{s}|{rep}', pairs[s][:, 0], pairs[s][:, 1]))
    ar = np.arange(len(pr))
    comps.append(('split-V0|raw', 'V0|raw', 'V0|split', pr[:, 0], ar))
    comps.append(('merge-V0|raw', 'V0|raw', 'Hzp|merge', pr[:, 0], ar))
    comps.append(('merge-Hzp|raw', 'Hzp|raw', 'Hzp|merge', pr[:, 1], ar))
    B = a.boot; R = dict(node='H60 verify', seed=SEEDV, pairs={s: int(len(v)) for s, v in pairs.items()}, truth_identical=same_truth,
                         split_merge_mean_k=float(kk.mean()), c2st=c2, info=info, comps={}, lrN=dict(coef=float(lrN.coef_[0, 0])))
    tests = []
    for lab, ra, rb, ia, ib in comps:
        R['comps'][lab] = {}
        for cls, cn in ((1, 'top'), (0, 'qcd')):
            sel = y['V0'][ia] == cls if ra.startswith('V0') else y['Hzp'][ia] == cls
            a0, a1 = ia[sel], ib[sel]; J = len(a0)
            D = np.stack([S[rb][m][a1] - S[ra][m][a0] for m in models], 1)
            Do = np.stack([O[rb][k][a1] - O[ra][k][a0] for k in ('N', 'm', 'tau32', 'mW')], 1)
            bi = rng.integers(0, J, (B, J)); bD = np.stack([D[b].mean(0) for b in bi]); bO = np.stack([Do[b].mean(0) for b in bi])
            sg = rng.choice([-1.0, 1.0], (2000, J)); d = D.mean(0); mi = {m: i for i, m in enumerate(models)}
            r = dict(n_pairs=J, obs={k: dict(d=float(Do[:, i].mean()), ci=np.percentile(bO[:, i], [2.5, 97.5]).tolist(),
                                             rel=float(Do[:, i].mean() / np.mean(O[ra][k][a0]))) for i, k in enumerate(('N', 'm', 'tau32', 'mW'))}, models={})
            for m in models:
                i = mi[m]; null = np.abs(sg @ D[:, i] / J); p = float((np.sum(null >= abs(d[i])) + 1) / 2001)
                dd = D[:, i] - D[:, mi['EFN_top']]; nulle = np.abs(sg @ dd / J); pe = float((np.sum(nulle >= abs(dd.mean())) + 1) / 2001)
                r['models'][m] = dict(d=float(d[i]), ci=np.percentile(bD[:, i], [2.5, 97.5]).tolist(), p=p,
                                      minus_EFN=dict(d=float(dd.mean()), ci=np.percentile(bD[:, i] - bD[:, mi['EFN_top']], [2.5, 97.5]).tolist(), p=pe))
                if m not in ('lrN',):
                    tests.append((lab, cn, m, 'mean0', p))
                    if m != 'EFN_top': tests.append((lab, cn, m, 'minusEFN', pe))
            R['comps'][lab][cn] = r
            log(lab, cn, J, f"dN {r['obs']['N']['d']:+.2f} ({100 * r['obs']['N']['rel']:+.1f}%) dm {r['obs']['m']['d']:+.2f} dtau32 {r['obs']['tau32']['d']:+.4f} |",
                ' '.join(f"{m} {r['models'][m]['d']:+.3f}" for m in models))
    # split share of the Hz+ shift (same V0 jets: split pairs are the Hz+ pairs)
    R['share_split'] = {}; R['share_merge_removed'] = {}
    for cls, cn in ((1, 'top'), (0, 'qcd')):
        sel = y['V0'][pr[:, 0]] == cls; i0, i1 = pr[sel, 0], pr[sel, 1]; ii = ar[sel]; J = len(i0)
        bi = rng.integers(0, J, (B, J)); R['share_split'][cn] = {}; R['share_merge_removed'][cn] = {}
        for m in models:
            dh = S['Hzp|raw'][m][i1] - S['V0|raw'][m][i0]; ds = S['V0|split'][m][ii] - S['V0|raw'][m][i0]
            dm = S['Hzp|merge'][m][ii] - S['V0|raw'][m][i0]
            bs = np.array([ds[b].mean() / dh[b].mean() for b in bi]); bm = np.array([1 - dm[b].mean() / dh[b].mean() for b in bi])
            R['share_split'][cn][m] = dict(share=float(ds.mean() / dh.mean()), ci=np.percentile(bs, [2.5, 97.5]).tolist())
            R['share_merge_removed'][cn][m] = dict(share=float(1 - dm.mean() / dh.mean()), ci=np.percentile(bm, [2.5, 97.5]).tolist())
        log('share split', cn, {m: round(v['share'], 2) for m, v in R['share_split'][cn].items()})
        log('share removed by merge', cn, {m: round(v['share'], 2) for m, v in R['share_merge_removed'][cn].items()})
    # Holm
    p = np.array([t[4] for t in tests]); o = np.argsort(p); adj = np.empty(len(p)); run = 0
    for rr, i in enumerate(o): run = max(run, min(1.0, (len(p) - rr) * p[i])); adj[i] = run
    R['holm'] = dict(n_tests=len(p), n_pass=int((adj < 0.05).sum()),
                     tests=[dict(comp=t[0], cls=t[1], model=t[2], test=t[3], p=float(t[4]), holm=float(h)) for t, h in zip(tests, adj)])
    (RES / 'h60_verify.json').write_text(json.dumps(R, indent=1, default=float)); log('wrote', RES / 'h60_verify.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', required=True, choices=['gen', 'score'])
    ap.add_argument('--setting', default='V0'); ap.add_argument('--jets', type=int, default=4000)
    ap.add_argument('--boot', type=int, default=2000)
    a = ap.parse_args()
    dict(gen=step_gen, score=step_score)[a.step](a)
