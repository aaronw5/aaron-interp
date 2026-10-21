"""H59 (loop D, claims D16, D4m): jet-charge null on the retrained QG particle-ID (kinpid) networks, and which jet-charge
weighting Q_kappa the networks follow.

Q_kappa = sum_i q_i (pT_i / pT_J)^kappa  (q_i = particle charge from its PDG id; pT_J = |vector sum of the particle pT|;
kappa = 0 is the net number of + minus - charged particles, large kappa weights the hardest particles).
Score s = log p(quark) - log p(gluon) (log-odds; positive = more quark-like).

Samples (fresh): Pythia = interp.data_qg.sample(per_class=5000, seed=SEED) (KMT Pythia 8 test split, random rows per class);
  Herwig = h14.load_sample('hw', 5000, SEED + 1) (Herwig 7.1 files 18-19). SEED = 5901 (run), 5911 (verify).
Models (all Pythia-trained, weaver checkpoints in /j-jepa-vol/aaron-interp/models/qg/<name>)
  part_kinpid, pnet_kinpid           seed 0 (H2)
  part_kinpid_s1, pnet_kinpid_s1     second training seed (H56 relaunch of H42 trainings)
  part_kinpid_shuf, pnet_kinpid_shuf labels = hash(jet_pt, jet_eta, jet_phi) mod 2 (class-independent)
  part_perm_s1..6, pnet_perm_s1..6   seed-0 nets with every weight tensor's entries permuted (H40v construction, seeds 1..6)
  part_kinpid_ln30, pnet_kinpid_ln30 (if final) 30 % of the training labels flipped at random (interp/h59_kinpid_ln_train.sh)
References: LR_k<kappa> = logistic regression on (pT_J, N, Q_kappa), kappa in KAPPAS, fitted on Pythia test-split jets outside the
  sample (positive controls with a known kappa). Under any edit below pT_J and N are fixed, so LR_k's Delta s = c_kappa Delta Q_kappa.

Part 1 (null) edits, PDG ids only (every (pT, y, phi) and N fixed; asserted): interp.h40_qg_pid_charge.make_edits
  E1  every charged particle pid -> -pid (charge conjugation), read as Delta s x sign(Q_0.3)
  E2k1 - C2  hardest charged particle flipped, minus the H40 control C2 (one + and one - flipped with |Delta Q_0.3,0.5| < 0.01),
       x sign(charge of the hardest charged particle)
  E3g_4 / E3n_4 - C3_4  2 pi+ and 2 pi- below the jet's median particle pT -> photon / K0L, minus the same pions pi+ <-> pi-
Part 2 edits (charge-sign flips, pid -> -pid; four-momenta, species and charged count fixed):
  Fh_k  the k hardest charged particles (k = 1, 2, 3)
  Fs    every charged particle with z = pT/pT_J below the median z of that jet's charged particles (the soft half)
  FrS   the same number of charged particles as Fs, chosen at random among all charged particles (size-matched control)
  Fr_k  k charged particles chosen at random (size-matched to Fh_k)
  For each (jet, edit) Delta Q_kappa for kappa in KAPPAS; fit Delta s = a_kappa Delta Q_kappa (through the origin) on quark jets
  of both generators, 2-fold cross-fit by jet, held-out R^2, bootstrap over jets.
Steps
  --step run [--seed 5901] [--tag ''] (GPU, part_job.sh) -> $AI_RESULTS/physics/h59_charge_kappa<tag>.json, h59_perjet_<gen><tag>.npz
  --step fig [--tag '']  -> results/physics/h59_kappa<tag>.png
  --smoke N
"""
import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import h40_qg_pid_charge as H40  # noqa: E402

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
V = Path('/j-jepa-vol/aaron-interp')
RES = Path(os.environ.get('AI_RESULTS', V / 'results')) / 'physics'
KAPPAS = (0.0, 0.3, 0.5, 1.0, 2.0)
KN = {k: f'k{k:g}' for k in KAPPAS}
ARCH = dict(part='part', pnet='pn')
TRAINED = ['part_kinpid', 'pnet_kinpid', 'part_kinpid_s1', 'pnet_kinpid_s1', 'part_kinpid_shuf', 'pnet_kinpid_shuf',
           'part_kinpid_ln30', 'pnet_kinpid_ln30']
NPERM = 6
P1_EDITS = ['E1', 'E2k1', 'C2', 'E3g_4', 'E3n_4', 'C3_4']
P2_EDITS = ['Fh1', 'Fh2', 'Fh3', 'Fs', 'FrS', 'Fr1', 'Fr2', 'Fr3']
KNOWN = dict(part_kinpid='weaver test acc 0.8400; H42 Pythia AUC 0.9084 / acc 0.834, Herwig AUC 0.8077',
             pnet_kinpid='weaver test acc 0.8338; H42 Pythia AUC 0.9038 / acc 0.829, Herwig AUC 0.8083',
             part_kinpid_s1='weaver test acc 0.8396', pnet_kinpid_s1='weaver test acc 0.8358',
             part_kinpid_shuf='weaver test acc 0.5003 (constant, H56)', pnet_kinpid_shuf='weaver test acc 0.5003 (constant, H56)')


def charge(pid):
    return H40.charge(pid)


def zfrac(a):
    ptj = np.hypot((a[:, 0] * np.cos(a[:, 2])).sum(), (a[:, 0] * np.sin(a[:, 2])).sum())
    return a[:, 0] / ptj


def qk_list(L, kappa):
    return np.array([(charge(a[:, 3]) * zfrac(a) ** kappa).sum() for a in L])


# ============================================================================================================== edits
def part2_edits(L0, gen, seed):
    """-> dict name -> (list of edited arrays, applied mask, number flipped)"""
    rng = np.random.default_rng([seed, zlib.crc32(gen.encode()), 59])
    J = len(L0); E = {n: ([a.copy() for a in L0], np.zeros(J, bool), np.zeros(J)) for n in P2_EDITS}
    for j, a in enumerate(L0):
        ch = np.flatnonzero(charge(a[:, 3]) != 0)          # pT-sorted (lists are pT-sorted)
        z = zfrac(a)
        for k in (1, 2, 3):
            if len(ch) >= k:
                for nm, sel in ((f'Fh{k}', ch[:k]), (f'Fr{k}', rng.choice(ch, k, replace=False))):
                    E[nm][0][j][sel, 3] *= -1; E[nm][1][j] = True; E[nm][2][j] = k
        if len(ch) >= 2:
            soft = ch[z[ch] < np.median(z[ch])]
            if len(soft):
                E['Fs'][0][j][soft, 3] *= -1; E['Fs'][1][j] = True; E['Fs'][2][j] = len(soft)
                r = rng.choice(ch, len(soft), replace=False)
                E['FrS'][0][j][r, 3] *= -1; E['FrS'][1][j] = True; E['FrS'][2][j] = len(soft)
    return E


# ============================================================================================================= models
def load_models(dev, smoke):
    from interp import h5_qg_counting as H
    from interp import h40v_verify as HV
    M = H.Models(dev); used, skipped = [], {}
    for nm in TRAINED:
        d = V / 'models' / 'qg' / nm
        if not (d / 'net_best_epoch_state.pt').exists(): skipped[nm] = 'no checkpoint'; continue
        n_ep = len(list(d.glob('net_epoch-*_state.pt')))
        if not ((d / 'predict_output').exists() or n_ep >= 20): skipped[nm] = f'not final ({n_ep} epochs written)'; continue
        M.add_weaver(nm, ARCH[nm.split('_')[0]], True); used.append(nm)
    perms = []
    for s in range(1, (2 if smoke else NPERM) + 1):
        for src, tag in (('part_kinpid', 'part'), ('pnet_kinpid', 'pnet')):
            nm = f'{tag}_perm_s{s}'; HV.add_perm(M, nm, src, s); perms.append(nm)
    log('models used', used, 'skipped', skipped, 'permuted', perms)
    return M, used, skipped, perms


def fit_lrs(seed, exclude_idx):
    """LR on (pT_J, N, Q_kappa) per kappa, Pythia test split outside the sample (standardised inputs)"""
    from sklearn.linear_model import LogisticRegression
    from interp import data_qg
    d = data_qg.DST / 'test'; y = np.load(d / 'label.npy').astype(int)
    F = np.load(d / 'full.npy', mmap_mode='r'); P = np.load(d / 'pid.npy', mmap_mode='r')
    Q = {k: [] for k in KAPPAS}; PT, N = [], []
    for s in range(0, len(y), 20000):
        f = np.asarray(F[s:s + 20000], np.float64); p = np.asarray(P[s:s + 20000]).astype(np.int64)
        pt = np.hypot(f[..., 0], f[..., 1]); real = pt > 0; ptj = np.hypot(f[..., 0].sum(1), f[..., 1].sum(1))
        z = np.where(real, pt / ptj[:, None], 1.0); q = charge(p) * real
        for k in KAPPAS: Q[k].append((q * z ** k).sum(1))
        PT.append(ptj); N.append(real.sum(1))
    PT = np.concatenate(PT); N = np.concatenate(N).astype(float); Q = {k: np.concatenate(v) for k, v in Q.items()}
    free = np.ones(len(y), bool); free[exclude_idx] = False
    LR = {}
    for k in KAPPAS:
        X = np.stack([PT, N, Q[k]], 1)[free]; mu, sd = X.mean(0), X.std(0)
        m = LogisticRegression(max_iter=3000, C=1e3).fit((X - mu) / sd, y[free])
        w = m.coef_[0] / sd; b = m.intercept_[0] - (m.coef_[0] * mu / sd).sum()
        LR[f'LR_{KN[k]}'] = dict(kappa=k, w_pt=float(w[0]), w_N=float(w[1]), w_Q=float(w[2]), b=float(b), n=int(free.sum()))
    log('LR references', {k: round(v['w_Q'], 4) for k, v in LR.items()})
    return LR


def lr_score(lr, ptj, N, Qk):
    return lr['w_pt'] * ptj + lr['w_N'] * N + lr['w_Q'] * Qk + lr['b']


# =========================================================================================================== scoring
def run_gen(gen, M, nets, LR, per, seed):
    from interp import h5_qg_counting as H
    from interp import h14_qg_generator as H14
    sd = seed if gen == 'py' else seed + 1
    L0, y, info = H14.load_sample(gen, per, sd); y = np.asarray(y).astype(int); J = len(y)
    log(f'{gen}: {J} jets ({info["source"]}, seed {sd}), quark fraction {y.mean():.3f}')
    E1 = H40.make_edits(L0, f'{gen}_h59_{seed}', full=False); E = {k: E1[k] for k in P1_EDITS}
    E.update(part2_edits(L0, gen, seed))
    log(f'{gen}: edits applied ' + ', '.join(f'{k} {int(v[1].sum())}' for k, v in E.items()))
    out = dict(y=y, idx=np.asarray(info['idx']))
    ptj = np.array([np.hypot((a[:, 0] * np.cos(a[:, 2])).sum(), (a[:, 0] * np.sin(a[:, 2])).sum()) for a in L0])
    Nn = np.array([len(a) for a in L0], float); out['ptj'] = ptj; out['N'] = Nn
    out['nch'] = np.array([(charge(a[:, 3]) != 0).sum() for a in L0], float)
    for k in KAPPAS: out[f'Q|base|{KN[k]}'] = qk_list(L0, k)
    out['qlead'] = np.array([(lambda q: q[q != 0][0] if (q != 0).any() else 0.0)(charge(a[:, 3])) for a in L0])
    J0 = H.from_list(L0, Lpad=180); t = time.time()
    for nm in nets: out[f'base|{nm}'] = -M.logit(nm, J0)
    log(f'{gen} base scored ({time.time() - t:.0f}s)')
    for ln, lr in LR.items(): out[f'base|{ln}'] = lr_score(lr, ptj, Nn, out[f'Q|base|{KN[lr["kappa"]]}'])
    unedited = {nm: H40.roc(out[f'base|{nm}'], y) for nm in nets + list(LR)}
    for nm in nets + list(LR):
        unedited[nm]['score_sd'] = float(np.std(out[f'base|{nm}']))
        log(f'{gen} unedited {nm}: AUC {unedited[nm]["AUC"]:.4f} acc {unedited[nm]["acc"]:.4f} score sd {unedited[nm]["score_sd"]:.4f}')
    for en, (Le, ok, nr) in E.items():
        Je = H.from_list(Le, Lpad=180)
        assert np.allclose(Je['pt'], J0['pt']) and np.allclose(Je['eta'], J0['eta']) and np.allclose(Je['phi'], J0['phi'])
        assert (Je['n'] == J0['n']).all()
        out[f'ok|{en}'] = ok; out[f'nflip|{en}'] = nr
        for k in KAPPAS: out[f'Q|{en}|{KN[k]}'] = qk_list(Le, k)
        for nm in nets: out[f'{en}|{nm}'] = -M.logit(nm, Je)
        for ln, lr in LR.items(): out[f'{en}|{ln}'] = lr_score(lr, ptj, Nn, out[f'Q|{en}|{KN[lr["kappa"]]}'])
        if en in P2_EDITS:   # charged count fixed
            nch = np.array([(charge(a[:, 3]) != 0).sum() for a in Le]); assert (nch == out['nch']).all()
        log(f'{gen} {en}: applied {int(ok.sum())}, mean flipped {nr[ok].mean() if ok.any() else 0:.2f}; mean dQ0.3 '
            f'{np.mean((out[f"Q|{en}|k0.3"] - out["Q|base|k0.3"])[ok]):+.4f}; '
            + '; '.join(f'{nm} {np.mean((out[f"{en}|{nm}"] - out[f"base|{nm}"])[ok]):+.4f}' for nm in nets[:6]))
    mx = np.abs(out['Q|E1|k0.3'] + out['Q|base|k0.3']).max(); log(f'{gen} check E1: max |Q_E1 + Q| = {mx:.2e}')
    return out, unedited, dict(J=J, per_class=per, seed=sd, source=info['source'])


# ======================================================================================================== statistics
def boot_ci(d, B, rng):
    n = len(d); s = np.concatenate([d[rng.integers(0, n, (min(200, B - b), n))].mean(1) for b in range(0, B, 200)])
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))], s


def p_normal(d):
    from scipy.stats import norm
    se = d.std(ddof=1) / np.sqrt(len(d))
    return float(2 * norm.sf(abs(d.mean() / se))) if se > 0 else 1.0


def part1_stats(out, names, B, seed):
    rng = np.random.default_rng(seed + 11); S = {}
    sq = np.sign(out['Q|base|k0.3']); sl = np.sign(out['qlead'])
    defs = [('E1_signQ', 'E1', None, sq), ('E2k1-C2_signlead', 'E2k1', 'C2', sl), ('E3g_4-C3', 'E3g_4', 'C3_4', None),
            ('E3n_4-C3', 'E3n_4', 'C3_4', None)]
    for cn, en, ctl, sg in defs:
        ok = out[f'ok|{en}'] & (out[f'ok|{ctl}'] if ctl else True) & ((sg != 0) if sg is not None else True)
        row = dict(n_jets=int(ok.sum())); D = {}
        for nm in names:
            base = out[f'base|{nm}']; de = out[f'{en}|{nm}'] - base; dc = (out[f'{ctl}|{nm}'] - base) if ctl else 0 * de
            d = (de - dc)[ok] * (sg[ok] if sg is not None else 1); D[nm] = d
            ci, _ = boot_ci(d, B, rng); const = bool(base.std() < 1e-5)   # constant network: float-level ties only
            row[nm] = dict(mean=float(d.mean()), ci=ci, p=1.0 if const else p_normal(d), constant_network=const,
                           mean_over_score_sd=float(d.mean() / max(base.std(), 1e-12)))
        # seed difference (paired over jets)
        for arch in ('part', 'pnet'):
            a, b = f'{arch}_kinpid', f'{arch}_kinpid_s1'
            if a in D and b in D:
                dd = D[b] - D[a]; ci, _ = boot_ci(dd, B, rng)
                row[f'{arch}_s1_minus_s0'] = dict(mean=float(dd.mean()), ci=ci, p=p_normal(dd), ratio=float(D[b].mean() / D[a].mean()))
        S[cn] = row
    return S


def r2_fit(ds, dq, jet, rng, B, folds):
    """Delta s = a Delta Q through the origin; 2-fold cross-fit by jet -> held-out R^2, a; bootstrap over jets (test residuals)"""
    a_f = []; res = np.empty_like(ds)
    for f in (0, 1):
        tr = folds[jet] == f
        a = (dq[tr] * ds[tr]).sum() / max((dq[tr] ** 2).sum(), 1e-12); a_f.append(a)
        te = ~tr; res[te] = ds[te] - a * dq[te]
    a_all = (dq * ds).sum() / max((dq ** 2).sum(), 1e-12)
    sse_j = np.bincount(jet, res ** 2); sst_j = np.bincount(jet, (ds - ds.mean()) ** 2)
    r2 = 1 - sse_j.sum() / max(sst_j.sum(), 1e-12)
    uj = np.unique(jet); bs = []
    for _ in range(B):
        k = rng.choice(uj, len(uj)); bs.append(1 - sse_j[k].sum() / max(sst_j[k].sum(), 1e-12))
    return dict(a=float(a_all), a_folds=[float(x) for x in a_f], R2=float(r2)), np.array(bs), sse_j, sst_j


def part2_stats(outs, names, B, seed, cls=1):
    """pooled over generators, quark jets (cls=1) -> per model: per-kappa a, held-out R^2, slope ratio Fs/Fh, increments"""
    rng = np.random.default_rng(seed + 22); R = {}
    rows = []   # (gen, jet index, edit)
    for gi, (gen, out) in enumerate(outs.items()):
        y = out['y']
        for en in P2_EDITS:
            ok = out[f'ok|{en}'] & (y == cls)
            for j in np.flatnonzero(ok): rows.append((gi, j, en))
    gi_a = np.array([r[0] for r in rows]); j_a = np.array([r[1] for r in rows]); e_a = np.array([r[2] for r in rows])
    gl = list(outs)
    jet_key = gi_a * 10 ** 6 + j_a; ujk, jet = np.unique(jet_key, return_inverse=True)
    folds = np.random.default_rng(seed + 33).integers(0, 2, len(ujk))
    dQ = {}
    for k in KAPPAS:
        v = np.empty(len(rows))
        for gi, gen in enumerate(gl):
            for en in P2_EDITS:
                m = (gi_a == gi) & (e_a == en)
                v[m] = (outs[gen][f'Q|{en}|{KN[k]}'] - outs[gen][f'Q|base|{KN[k]}'])[j_a[m]]
        dQ[k] = v
    R['_rows'] = dict(n_rows=len(rows), n_jets=int(len(ujk)), per_edit={en: int((e_a == en).sum()) for en in P2_EDITS},
                      mean_abs_dQ={en: {KN[k]: float(np.abs(dQ[k][e_a == en]).mean()) for k in KAPPAS} for en in P2_EDITS})
    for nm in names:
        ds = np.empty(len(rows))
        for gi, gen in enumerate(gl):
            for en in P2_EDITS:
                m = (gi_a == gi) & (e_a == en)
                ds[m] = (outs[gen][f'{en}|{nm}'] - outs[gen][f'base|{nm}'])[j_a[m]]
        r = dict(var_ds=float(ds.var()), mean_abs_ds={en: float(np.abs(ds[e_a == en]).mean()) for en in P2_EDITS}); fits = {}; BS = {}
        bsd = max(float(np.std(outs[g][f'base|{nm}'])) for g in gl)
        if ds.var() < 1e-10 or bsd < 1e-5:   # constant network (float-level noise only)
            r['degenerate'] = True; r['base_score_sd'] = bsd; R[nm] = r; continue
        st = rng.bit_generator.state
        for k in KAPPAS:
            rng.bit_generator.state = st      # same bootstrap jet draws for every kappa (paired)
            f, bs, _, _ = r2_fit(ds, dQ[k], jet, rng, B, folds)
            f['R2_ci'] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]; fits[KN[k]] = f; BS[k] = bs
        r['fits'] = fits
        best = max(KAPPAS, key=lambda k: fits[KN[k]]['R2']); sec = max([k for k in KAPPAS if k != best], key=lambda k: fits[KN[k]]['R2'])
        dif = BS[best] - BS[sec]
        r['best_kappa'] = best; r['second_kappa'] = sec
        r['best_minus_second'] = dict(mean=float(fits[KN[best]]['R2'] - fits[KN[sec]]['R2']),
                                      ci=[float(np.percentile(dif, 2.5)), float(np.percentile(dif, 97.5))],
                                      p=float(min(1, 2 * (1 + (dif <= 0).sum()) / (len(dif) + 1))))
        r['best_kappa_boot_freq'] = {KN[k]: float(np.mean(np.argmax(np.stack([BS[q] for q in KAPPAS]), 0) == i)) for i, k in enumerate(KAPPAS)}
        # slope ratio Fs / Fh (Fh1-3 pooled) per kappa: equals 1 at the kappa the network reads
        hm = np.isin(e_a, ['Fh1', 'Fh2', 'Fh3']); sm = e_a == 'Fs'
        uj = np.unique(jet)
        def ratios(jsel):
            o = []
            for k in KAPPAS:
                x = dQ[k]; sh = (x[hm & jsel] * ds[hm & jsel]).sum() / max((x[hm & jsel] ** 2).sum(), 1e-12)
                ss = (x[sm & jsel] * ds[sm & jsel]).sum() / max((x[sm & jsel] ** 2).sum(), 1e-12); o.append((sh, ss))
            return o
        o = ratios(np.ones(len(ds), bool))
        r['slope_Fh'] = {KN[k]: float(o[i][0]) for i, k in enumerate(KAPPAS)}; r['slope_Fs'] = {KN[k]: float(o[i][1]) for i, k in enumerate(KAPPAS)}
        r['ratio_Fs_Fh'] = {KN[k]: float(o[i][1] / o[i][0]) if o[i][0] != 0 else None for i, k in enumerate(KAPPAS)}
        def cross(rt):
            rt = np.array(rt); ks = np.array(KAPPAS)
            for i in range(len(ks) - 1):
                if abs(rt[i] - 1) < 1e-6: return float(ks[i])
                if (rt[i] - 1) * (rt[i + 1] - 1) <= 0 and rt[i] != rt[i + 1]:
                    return float(ks[i] + (1 - rt[i]) * (ks[i + 1] - ks[i]) / (rt[i + 1] - rt[i]))
            return float('nan')
        r['kappa_ratio1'] = cross([o[i][1] / o[i][0] if o[i][0] else np.nan for i in range(len(KAPPAS))])
        # bootstrap of the crossing (jet resampling via weights)
        cb = []; cnt = np.bincount(jet)
        for _ in range(min(B, 300)):
            w = np.bincount(rng.choice(uj, len(uj)), minlength=len(cnt))[jet].astype(float); rr = []
            for k in KAPPAS:
                x = dQ[k]; sh = (w * x * ds)[hm].sum() / max((w * x * x)[hm].sum(), 1e-12); ss = (w * x * ds)[sm].sum() / max((w * x * x)[sm].sum(), 1e-12)
                rr.append(ss / sh if sh else np.nan)
            cb.append(cross(rr))
        cb = np.array(cb); r['kappa_ratio1_boot'] = dict(ci=[float(np.nanpercentile(cb, 2.5)), float(np.nanpercentile(cb, 97.5))] if np.isfinite(cb).any() else None,
                                                          frac_no_crossing=float(np.mean(~np.isfinite(cb))))
        R[nm] = r
    return R


def increments(outs, names, B, seed, cls=1):
    """leading-only test: (Delta s(Fh2) - Delta s(Fh1)) x (-q_2) and (Fh3 - Fh2) x (-q_3) [sign = direction of Delta Q], quark jets;
    and the per-edit response per unit Delta Q_0 / per flipped particle; random-flip controls"""
    rng = np.random.default_rng(seed + 44); R = {}
    for nm in names:
        r = {}
        for a_, b_, i in (('Fh1', 'Fh2', 1), ('Fh2', 'Fh3', 2)):
            ds, cnt = [], 0
            for gen, out in outs.items():
                ok = out[f'ok|{b_}'] & (out['y'] == cls)
                # charge of the (i+1)-th charged particle = -(Delta Q_0 increment) / 2
                dq0 = (out[f'Q|{b_}|k0'] - out[f'Q|{a_}|k0'])[ok]; sgn = np.sign(dq0)
                ds.append((out[f'{b_}|{nm}'] - out[f'{a_}|{nm}'])[ok] * sgn)
            d = np.concatenate(ds); ci, _ = boot_ci(d, B, rng)
            r[f'{b_}-{a_}_signed'] = dict(mean=float(d.mean()), ci=ci, p=p_normal(d), n=int(len(d)))
        d1 = []
        for gen, out in outs.items():
            ok = out['ok|Fh1'] & (out['y'] == cls); d1.append((out[f'Fh1|{nm}'] - out[f'base|{nm}'])[ok] * np.sign(out['Q|Fh1|k0'] - out['Q|base|k0'])[ok])
        d1 = np.concatenate(d1); ci, _ = boot_ci(d1, B, rng); r['Fh1_signed'] = dict(mean=float(d1.mean()), ci=ci, p=p_normal(d1))
        # per-edit response signed by the direction of the net-charge change (kappa = 0), per unit |Delta Q_0| / 2 (= per unit net flipped charge)
        for en in P2_EDITS:
            ds, dq = [], []
            for gen, out in outs.items():
                ok = out[f'ok|{en}'] & (out['y'] == cls); x = (out[f'Q|{en}|k0'] - out['Q|base|k0'])[ok]; m = x != 0
                ds.append((out[f'{en}|{nm}'] - out[f'base|{nm}'])[ok][m]); dq.append(x[m])
            ds, dq = np.concatenate(ds), np.concatenate(dq)
            d = ds * np.sign(dq); ci, _ = boot_ci(d, B, rng)
            r[f'{en}_signed_by_dQ0'] = dict(mean=float(d.mean()), ci=ci, n=int(len(d)), mean_abs_dQ0=float(np.abs(dq).mean()),
                                            per_unit_net_charge=float(d.mean() / (np.abs(dq).mean() / 2)))
        R[nm] = r
    return R


def holm_part1(S, models, gens):
    pl = []
    for gen in gens:
        for cn, row in S[gen].items():
            for nm in models:
                if nm in row and not row[nm].get('constant_network'): pl.append((gen, cn, nm, row[nm]['p']))
    adj = H40.holm([p for *_, p in pl])
    for (gen, cn, nm, _), a in zip(pl, adj): S[gen][cn][nm]['p_holm'] = float(a)
    return dict(n_tests=len(pl), n_reject_05=int((adj < 0.05).sum()))


def perm_band(S, gens, perms, used):
    """per (gen, contrast, arch): permuted-network values; shuffled / ln30 value and rank p = (1 + #perm |e| >= |v|) / (K + 1)"""
    B = {}
    for gen in gens:
        for cn, row in S[gen].items():
            for arch in ('part', 'pnet'):
                pv = np.array([row[p]['mean'] for p in perms if p.startswith(arch)])
                d = dict(perm_values=pv.tolist(), perm_min=float(pv.min()), perm_max=float(pv.max()), perm_max_abs=float(np.abs(pv).max()))
                for var in ('kinpid', 'kinpid_s1', 'kinpid_shuf', 'kinpid_ln30'):
                    nm = f'{arch}_{var}'
                    if nm in row:
                        v = row[nm]['mean']; d[var] = dict(value=v, inside_band=bool(pv.min() <= v <= pv.max()),
                                                           rank_p=float((1 + (np.abs(pv) >= abs(v)).sum()) / (len(pv) + 1)))
                B[f'{gen}|{cn}|{arch}'] = d
    return B


# ============================================================================================================== steps
def step_run(a):
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; RES.mkdir(parents=True, exist_ok=True)
    sm = '_smoke' if a.smoke else ''; per = a.smoke or 5000; B = 200 if a.smoke else 1000; tag = a.tag + sm
    log(f'device {dev}; per class {per}; B {B}; seed {a.seed}; tag {tag!r}')
    M, used, skipped, perms = load_models(dev, a.smoke)
    nets = used + perms
    from interp import data_qg
    LR = fit_lrs(a.seed, data_qg.sample(per, a.seed)['idx'])
    res = dict(seed=a.seed, models=M.info, used=used, skipped=skipped, perms=perms, LR=LR, kappas=list(KAPPAS), known=KNOWN)
    outs = {}
    for gen in ('py', 'hw'):
        out, une, samp = run_gen(gen, M, nets, LR, per, a.seed); outs[gen] = out
        res.setdefault('unedited', {})[gen] = une; res.setdefault('sample', {})[gen] = samp
        np.savez_compressed(RES / f'h59_perjet_{gen}{tag}.npz', **{k: (v.astype(np.float32) if v.dtype == np.float64 else v) for k, v in out.items()})
    refs = list(LR)
    S = {gen: part1_stats(outs[gen], nets + refs, B, a.seed) for gen in outs}
    res['part1'] = S
    res['holm_part1'] = holm_part1(S, used, list(outs)); log('Holm part 1', res['holm_part1'])
    res['perm_band'] = perm_band(S, list(outs), perms, used)
    for gen in outs:
        for cn in S[gen]:
            log(f'P1 {gen} {cn}: ' + '; '.join(f'{nm} {S[gen][cn][nm]["mean"]:+.4f} {np.round(S[gen][cn][nm]["ci"], 4).tolist()}'
                                                for nm in used + ['LR_k0.3']) + '; perm max|.| part '
                f'{res["perm_band"][f"{gen}|{cn}|part"]["perm_max_abs"]:.4f} pnet {res["perm_band"][f"{gen}|{cn}|pnet"]["perm_max_abs"]:.4f}')
            for arch in ('part', 'pnet'):
                if f'{arch}_s1_minus_s0' in S[gen][cn]:
                    x = S[gen][cn][f'{arch}_s1_minus_s0']; log(f'   {arch} s1 - s0 {x["mean"]:+.4f} {np.round(x["ci"], 4).tolist()} ratio {x["ratio"]:.3f}')
    (RES / f'h59_charge_kappa{tag}.json').write_text(json.dumps(res, indent=1, default=float))
    res['part2'] = {'quark': part2_stats(outs, nets + refs, B, a.seed, 1), 'gluon': part2_stats(outs, used + refs, min(B, 300), a.seed, 0)}
    for gen in outs:
        res['part2'][f'quark_{gen}'] = part2_stats({gen: outs[gen]}, used + refs, min(B, 300), a.seed, 1)
    res['increments'] = increments(outs, nets + refs, B, a.seed, 1)
    tested = [n for n in used if not res['part2']['quark'][n].get('degenerate')]
    adj = H40.holm([res['part2']['quark'][n]['best_minus_second']['p'] for n in tested])
    for n, p in zip(tested, adj): res['part2']['quark'][n]['best_minus_second']['p_holm'] = float(p)
    for nm in nets + refs:
        r = res['part2']['quark'][nm]
        if r.get('degenerate'): log(f'P2 {nm}: degenerate (var ds {r["var_ds"]:.2e})'); continue
        log(f'P2 {nm}: R2 ' + ', '.join(f'{k} {v["R2"]:.3f}' for k, v in r['fits'].items()) + f'; best {r["best_kappa"]} '
            f'(- second {r["best_minus_second"]["mean"]:+.4f} {np.round(r["best_minus_second"]["ci"], 4).tolist()}); ratio Fs/Fh '
            + ', '.join(f'{k} {v:.2f}' for k, v in r['ratio_Fs_Fh'].items() if v is not None) + f'; kappa(ratio=1) {r["kappa_ratio1"]:.2f} {r["kappa_ratio1_boot"]}')
        ic = res['increments'][nm]
        log(f'   increments Fh1 {ic["Fh1_signed"]["mean"]:+.4f}, Fh2-Fh1 {ic["Fh2-Fh1_signed"]["mean"]:+.4f} {np.round(ic["Fh2-Fh1_signed"]["ci"], 4).tolist()}, '
            f'Fh3-Fh2 {ic["Fh3-Fh2_signed"]["mean"]:+.4f}; per unit net charge Fh1 {ic["Fh1_signed_by_dQ0"]["per_unit_net_charge"]:+.4f} '
            f'Fs {ic["Fs_signed_by_dQ0"]["per_unit_net_charge"]:+.4f} FrS {ic["FrS_signed_by_dQ0"]["per_unit_net_charge"]:+.4f}')
    (RES / f'h59_charge_kappa{tag}.json').write_text(json.dumps(res, indent=1, default=float))
    log('wrote', RES / f'h59_charge_kappa{tag}.json')
    try:
        a.tag = tag; step_fig(a, already_tagged=True)
    except Exception as e:  # noqa: BLE001
        log('figure failed', repr(e))


def step_fig(a, already_tagged=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tag = a.tag if already_tagged else a.tag + ('_smoke' if a.smoke else '')
    R = json.loads((RES / f'h59_charge_kappa{tag}.json').read_text()); P = R['part2']['quark']
    ks = R['kappas']
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    sty = dict(part_kinpid=('#1f77b4', '-', 'ParT kinpid seed 0'), part_kinpid_s1=('#1f77b4', '--', 'ParT kinpid seed 1'),
               pnet_kinpid=('#d62728', '-', 'ParticleNet kinpid seed 0'), pnet_kinpid_s1=('#d62728', '--', 'ParticleNet kinpid seed 1'))
    ax = axs[0]
    for nm, (c, ls, lab) in sty.items():
        if nm in P and 'fits' in P[nm]:
            r2 = [P[nm]['fits'][f'k{k:g}']['R2'] for k in ks]; lo = [P[nm]['fits'][f'k{k:g}']['R2_ci'][0] for k in ks]; hi = [P[nm]['fits'][f'k{k:g}']['R2_ci'][1] for k in ks]
            ax.errorbar(ks, r2, yerr=[np.array(r2) - lo, np.array(hi) - r2], color=c, ls=ls, marker='o', ms=4, capsize=2, label=lab)
    pm = [p for p in R['perms'] if 'fits' in P.get(p, {})]
    if pm:
        v = np.array([[P[p]['fits'][f'k{k:g}']['R2'] for k in ks] for p in pm])
        ax.fill_between(ks, v.min(0), v.max(0), color='0.8', label=f'{len(pm)} weight-permuted nets (range)')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel(r'$\kappa$ in $\Delta Q_\kappa$, $Q_\kappa=\sum_i q_i\,(p_{T,i}/p_{T,J})^\kappa$')
    ax.set_ylabel(r'held-out $R^2$ of $\Delta s = a_\kappa\,\Delta Q_\kappa$')
    ax.set_title('Charge-flip edits, quark jets (Pythia + Herwig)', fontsize=10); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax = axs[1]
    for nm, (c, ls, lab) in list(sty.items()) + [('LR_k0', ('0.3', ':', r'LR($p_T,N,Q_0$)')), ('LR_k1', ('0.5', '-.', r'LR($p_T,N,Q_1$)'))]:
        if nm in P and 'ratio_Fs_Fh' in P[nm]:
            ax.plot(ks, [P[nm]['ratio_Fs_Fh'][f'k{k:g}'] for k in ks], color=c, ls=ls, marker='o', ms=4, label=lab)
    ax.axhline(1, color='k', lw=0.8)
    ax.set_xlabel(r'$\kappa$'); ax.set_ylabel(r'slope$_{\rm soft\ flips}$ / slope$_{\rm hard\ flips}$ of $\Delta s$ vs $\Delta Q_\kappa$')
    ax.set_yscale('log'); ax.set_title('ratio = 1 at the $\\kappa$ the score follows', fontsize=10); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle('Which jet charge $Q_\\kappa$ the QG kinpid networks follow (charge-sign flips at fixed four-momenta)', fontsize=10.5)
    fig.tight_layout(); p = RES / f'h59_kappa{tag}.png'; fig.savefig(p, dpi=140); log('wrote', p)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', choices=['run', 'fig'], required=True)
    ap.add_argument('--seed', type=int, default=5901)
    ap.add_argument('--smoke', type=int, default=0)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    {'run': step_run, 'fig': step_fig}[a.step](a)
