"""H59 verify (loop D, skeptic): independent check of "the QG kinpid networks follow Q_kappa with kappa ~ 0.33-0.37".

Q_kappa = sum_i q_i z_i^kappa, z_i = pT_i / pT_J. Score s = log p(quark) - log p(gluon) (= -Models.logit).
Fresh samples: Pythia data_qg.sample(5000, SEED) and Herwig files 18-19 (5000 per class, SEED + 1), SEED = 5931.
Models: part_kinpid, pnet_kinpid (seed 0), *_s1 (seed 1), 3 weight-permuted copies of each seed-0 net (H40v add_perm, seeds 1-3);
LR_k0.3 / LR_k0.5 references from h59.fit_lrs (positive controls with a known kappa).
Edits (PDG id sign flips only; four-momenta, N and n_ch fixed and asserted; particle lists sorted by pT first):
  S_r   flip ONLY the r-th hardest charged particle, r in RANKS (single flip)
  Sx    flip one random charged particle with z below the jet's median charged z (single soft flip)
  Fh2, Fh3, Fs, FrS  as in h59 (additivity and slope-ratio rerun)
  E1    all charged particles -> antiparticle (Part 1 seed check), read x sign(Q_0.3)
Analyses (quark jets, both generators pooled; gluon jets reported separately):
  A  single-flip power law. For a flip of a particle with charge q at z, a linear Q_kappa model predicts
     u = Delta s x (-q) / 2 = a z^kappa. Fit (a, kappa) by least squares over rows on a kappa grid (0..1.5, step 0.01),
     jet bootstrap (B = 500). Fits: all single flips; leading charged particle excluded (r >= 2 and Sx); antisymmetric part
     only (mean over q = + and q = - within z bins). Leading-particle excess = mean u(S_1) / (a z_1^kappa) from the fit
     without the leader.
  B  additivity: Delta s(Fh_k) vs sum_r<=k Delta s(S_r): slope through origin, R^2, mean residual with CI.
  C  slope-ratio kappa* (as h59) on a fine grid (0..1 step 0.025), with Delta Q over all particles and over the first 128
     (the networks' input length).
  D  E1 x sign(Q_0.3), s0 vs s1, permuted values.
  --step run (GPU) -> $AI_RESULTS/physics/h59_verify_kappa.json ; --smoke N
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
from interp import h40_qg_pid_charge as H40  # noqa: E402

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
V = Path('/j-jepa-vol/aaron-interp')
RES = Path(os.environ.get('AI_RESULTS', V / 'results')) / 'physics'
RANKS = (1, 2, 3, 4, 6, 8)
SINGLE = [f'S{r}' for r in RANKS] + ['Sx']
MULTI = ['Fh2', 'Fh3', 'Fs', 'FrS', 'E1']
EDITS = SINGLE + MULTI
KG = np.round(np.arange(0, 1.5001, 0.01), 3)
KR = np.round(np.arange(0, 1.0001, 0.025), 3)
L_IN = 128
TRAINED = ['part_kinpid', 'pnet_kinpid', 'part_kinpid_s1', 'pnet_kinpid_s1']


def zfrac(a):
    ptj = np.hypot((a[:, 0] * np.cos(a[:, 2])).sum(), (a[:, 0] * np.sin(a[:, 2])).sum())
    return a[:, 0] / ptj


def make(L0, seed):
    """-> edited lists per edit, flipped-index lists per edit (indices into the pT-sorted list)"""
    rng = np.random.default_rng([seed, 5959])
    E = {e: [a.copy() for a in L0] for e in EDITS}; IDX = {e: [np.zeros(0, int) for _ in L0] for e in EDITS}
    for j, a in enumerate(L0):
        q = H40.charge(a[:, 3]); ch = np.flatnonzero(q != 0); z = zfrac(a)
        sel = {}
        for r in RANKS:
            if len(ch) >= r: sel[f'S{r}'] = ch[r - 1:r]
        if len(ch) >= 2:
            soft = ch[z[ch] < np.median(z[ch])]
            if len(soft):
                sel['Sx'] = rng.choice(soft, 1); sel['Fs'] = soft; sel['FrS'] = rng.choice(ch, len(soft), replace=False)
        for k in (2, 3):
            if len(ch) >= k: sel[f'Fh{k}'] = ch[:k]
        if len(ch): sel['E1'] = ch
        for e, s in sel.items():
            E[e][j][s, 3] *= -1; IDX[e][j] = np.asarray(s, int)
    return E, IDX


def load_models():
    import torch
    from interp import h5_qg_counting as H
    from interp import h40v_verify as HV
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    M = H.Models(dev)
    for nm in TRAINED: M.add_weaver(nm, 'part' if nm.startswith('part') else 'pn', True)
    perms = []
    for s in (1, 2, 3):
        for src, tag in (('part_kinpid', 'part'), ('pnet_kinpid', 'pnet')):
            nm = f'{tag}_perm_s{s}'; HV.add_perm(M, nm, src, s); perms.append(nm)
    return M, perms, dev


def run_gen(gen, M, nets, LR, per, seed):
    from interp import h5_qg_counting as H
    from interp import h14_qg_generator as H14
    sd = seed if gen == 'py' else seed + 1
    L0, y, info = H14.load_sample(gen, per, sd); y = np.asarray(y).astype(int)
    unsorted = sum(bool(np.any(np.diff(a[:, 0]) > 0)) for a in L0)
    zero = sum(bool(np.any(a[:, 0] <= 0)) for a in L0)
    L0 = [a[np.argsort(-a[:, 0], kind='stable')] for a in L0]
    log(f'{gen}: {len(y)} jets seed {sd}; lists not pT-sorted before sorting: {unsorted}; lists with pT <= 0 entries: {zero}')
    E, IDX = make(L0, sd)
    out = dict(y=y, n=np.array([len(a) for a in L0]), checks=dict(unsorted=unsorted, zero_pt=zero))
    # per-edit per-jet: flipped z, charge (before), rank in the full list
    out['zq'] = {e: [(zfrac(a)[IDX[e][j]], H40.charge(a[IDX[e][j], 3]), IDX[e][j]) for j, a in enumerate(L0)] for e in EDITS}
    J0 = H.from_list(L0, Lpad=180); S = {}
    for nm in nets: S[('base', nm)] = -M.logit(nm, J0)
    une = {nm: H40.roc(S[('base', nm)], y) for nm in nets}
    for nm in nets: log(f'{gen} unedited {nm}: AUC {une[nm]["AUC"]:.4f} acc {une[nm]["acc"]:.4f}')
    nch0 = np.array([(H40.charge(a[:, 3]) != 0).sum() for a in L0])
    for e in EDITS:
        Je = H.from_list(E[e], Lpad=180)
        assert np.allclose(Je['pt'], J0['pt']) and np.allclose(Je['eta'], J0['eta']) and np.allclose(Je['phi'], J0['phi'])
        assert (Je['n'] == J0['n']).all()
        assert (np.array([(H40.charge(a[:, 3]) != 0).sum() for a in E[e]]) == nch0).all()
        for nm in nets: S[(e, nm)] = -M.logit(nm, Je)
        log(f'{gen} {e} scored; applied {sum(len(i) > 0 for i in IDX[e])}')
    # LR scores: Delta s = w_Q Delta Q_kappa exactly (pT_J, N fixed)
    out['S'] = S; out['une'] = une
    return out


def dQ(zq_j, kappa, lim=None):
    z, q, r = zq_j
    m = np.ones(len(z), bool) if lim is None else (r < lim)
    return float((-2 * q[m] * z[m] ** kappa).sum())


def boot_idx(nj, B, rng):
    return [rng.integers(0, nj, nj) for _ in range(B)]


def single_rows(outs, nm, cls, LR=None):
    """rows of single flips: jet key, edit, z, q, u = Delta s x (-q) / 2"""
    R = dict(jet=[], e=[], z=[], q=[], u=[], rank=[])
    for gi, (gen, o) in enumerate(outs.items()):
        for e in SINGLE:
            for j in np.flatnonzero(o['y'] == cls):
                z, q, r = o['zq'][e][j]
                if len(z) != 1: continue
                if LR is not None: ds = LR['w_Q'] * (-2 * q[0] * z[0] ** LR['kappa'])
                else: ds = o['S'][(e, nm)][j] - o['S'][('base', nm)][j]
                R['jet'].append(gi * 10 ** 6 + j); R['e'].append(e); R['z'].append(z[0]); R['q'].append(q[0]); R['u'].append(ds * (-q[0]) / 2); R['rank'].append(r[0])
    R = {k: np.array(v) for k, v in R.items()}
    _, R['jid'] = np.unique(R['jet'], return_inverse=True)
    return R


def powfit(R, mask, B, rng, antisym=False):
    """least squares u = a z^kappa over kappa grid; per-jet sufficient statistics for the jet bootstrap"""
    z, u, jid = R['z'][mask], R['u'][mask], R['jid'][mask]
    if antisym:   # weight rows so that q = + and q = - carry equal total weight within each of 15 log-z bins
        b = np.clip(np.digitize(np.log10(z), np.linspace(-3.5, 0, 15)), 0, 15); q = R['q'][mask]; w = np.zeros(len(z))
        for bb in np.unique(b):
            for s in (1, -1):
                m = (b == bb) & (q == s)
                if m.any() and ((b == bb) & (q == -s)).any(): w[m] = 0.5 / m.sum() * ((b == bb).sum())
    else:
        w = np.ones(len(z))
    nj = jid.max() + 1 if len(jid) else 0
    lz = np.log(z)
    S1 = np.stack([np.bincount(jid, w * u * np.exp(k * lz), nj) for k in KG]); S2 = np.stack([np.bincount(jid, w * np.exp(2 * k * lz), nj) for k in KG])
    U = np.bincount(jid, w * u * u, nj)

    def best(cnt):
        s1 = S1 @ cnt; s2 = S2 @ cnt; sse = U @ cnt - s1 ** 2 / s2; i = int(np.argmin(sse)); return KG[i], s1[i] / s2[i], 1 - sse[i] / (U @ cnt)
    k0, a0, r0 = best(np.ones(nj))
    bs = np.array([best(np.bincount(rng.integers(0, nj, nj), minlength=nj).astype(float))[:2] for _ in range(B)])
    sse_curve = (U.sum() - (S1.sum(1)) ** 2 / S2.sum(1)) / U.sum()
    return dict(kappa=float(k0), kappa_ci=np.percentile(bs[:, 0], [2.5, 97.5]).tolist(), a=float(a0), a_ci=np.percentile(bs[:, 1], [2.5, 97.5]).tolist(),
                frac_var_explained_uncentred=float(r0), n_rows=int(mask.sum()), n_jets=int(nj),
                sse_rel_at={f'{k:g}': float(sse_curve[np.argmin(np.abs(KG - k))]) for k in (0, 0.2, 0.3, 0.35, 0.4, 0.5, 0.7, 1.0)})


def analysis_A(outs, nm, cls, B, rng, LR=None):
    R = single_rows(outs, nm, cls, LR); res = {}
    allm = np.ones(len(R['u']), bool); nolead = R['e'] != 'S1'
    res['all'] = powfit(R, allm, B, rng); res['no_leader'] = powfit(R, nolead, B, rng); res['antisym_all'] = powfit(R, allm, B, rng, True)
    res['antisym_no_leader'] = powfit(R, nolead, B, rng, True)
    k, a = res['no_leader']['kappa'], res['no_leader']['a']
    per = {}
    for e in SINGLE:
        m = R['e'] == e
        if not m.any(): continue
        pred = a * R['z'][m] ** k; d = R['u'][m]; jj = R['jid'][m]
        rat = d.mean() / pred.mean()
        bs = []
        for _ in range(300):
            ii = rng.integers(0, m.sum(), m.sum()); bs.append(d[ii].mean() / pred[ii].mean())
        # symmetric (sign-independent) part: mean Delta s for + vs - flips
        dsp = (2 * d / -R['q'][m])
        per[e] = dict(n=int(m.sum()), mean_z=float(R['z'][m].mean()), mean_u=float(d.mean()), u_ci=np.percentile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(300)], [2.5, 97.5]).tolist(),
                      ratio_to_noleader_fit=float(rat), ratio_ci=np.percentile(bs, [2.5, 97.5]).tolist(),
                      mean_ds_plus_flip=float(dsp[R['q'][m] > 0].mean()) if (R['q'][m] > 0).any() else None,
                      mean_ds_minus_flip=float(dsp[R['q'][m] < 0].mean()) if (R['q'][m] < 0).any() else None)
    res['per_edit'] = per
    # z-binned means (for the figure)
    bins = np.logspace(-3.5, 0, 15); b = np.digitize(R['z'], bins)
    res['binned'] = [dict(z=float(R['z'][b == i].mean()), u=float(R['u'][b == i].mean()), n=int((b == i).sum()),
                          u_se=float(R['u'][b == i].std() / np.sqrt((b == i).sum()))) for i in np.unique(b) if (b == i).sum() >= 30]
    return res


def analysis_B(outs, nm, cls, B, rng):
    res = {}
    for k in (2, 3):
        x, yv = [], []
        for gen, o in outs.items():
            ok = (o['y'] == cls) & np.array([len(o['zq'][f'Fh{k}'][j][0]) == k for j in range(len(o['y']))])
            sm = sum(o['S'][(f'S{r}', nm)] - o['S'][('base', nm)] for r in range(1, k + 1))
            x.append(sm[ok]); yv.append((o['S'][(f'Fh{k}', nm)] - o['S'][('base', nm)])[ok])
        x, yv = np.concatenate(x), np.concatenate(yv)
        sl = (x * yv).sum() / (x * x).sum(); r2 = 1 - ((yv - sl * x) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
        res_d = yv - x; bs = [res_d[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
        bsl = []
        for _ in range(B):
            ii = rng.integers(0, len(x), len(x)); bsl.append((x[ii] * yv[ii]).sum() / (x[ii] ** 2).sum())
        res[f'Fh{k}'] = dict(n=int(len(x)), slope=float(sl), slope_ci=np.percentile(bsl, [2.5, 97.5]).tolist(), R2=float(r2),
                             mean_abs_sum_singles=float(np.abs(x).mean()), mean_abs_multi=float(np.abs(yv).mean()),
                             mean_residual=float(res_d.mean()), residual_ci=np.percentile(bs, [2.5, 97.5]).tolist())
    return res


def analysis_C(outs, nm, cls, B, rng, LR=None):
    res = {}
    for lim_name, lim in (('all', None), ('first128', L_IN)):
        rows = []
        for gi, (gen, o) in enumerate(outs.items()):
            for e in ('S1', 'Fh2', 'Fh3', 'Fs'):
                for j in np.flatnonzero(o['y'] == cls):
                    zq = o['zq'][e][j]
                    if len(zq[0]) == 0: continue
                    if LR is not None: ds = LR['w_Q'] * dQ(zq, LR['kappa'])
                    else: ds = o['S'][(e, nm)][j] - o['S'][('base', nm)][j]
                    z_, q_, r_ = zq; m_ = np.ones(len(z_), bool) if lim is None else (r_ < lim)
                    rows.append((gi * 10 ** 6 + j, e, ds, (-2 * q_[m_, None] * z_[m_, None] ** KR[None]).sum(0)))
        jk = np.array([r[0] for r in rows]); e = np.array([r[1] for r in rows]); ds = np.array([r[2] for r in rows]); X = np.array([r[3] for r in rows])
        _, jid = np.unique(jk, return_inverse=True); nj = jid.max() + 1
        hm = e != 'Fs'; sm = e == 'Fs'

        def kstar(w):
            sh = ((w * ds)[hm, None] * X[hm]).sum(0) / ((w[hm, None] * X[hm] ** 2).sum(0)); ss = ((w * ds)[sm, None] * X[sm]).sum(0) / ((w[sm, None] * X[sm] ** 2).sum(0))
            rt = ss / sh
            for i in range(len(KR) - 1):
                if (rt[i] - 1) * (rt[i + 1] - 1) <= 0 and rt[i] != rt[i + 1]:
                    return float(KR[i] + (1 - rt[i]) * (KR[i + 1] - KR[i]) / (rt[i + 1] - rt[i])), rt
            return float('nan'), rt
        k0, rt = kstar(np.ones(len(ds)))
        bs = [kstar(np.bincount(rng.integers(0, nj, nj), minlength=nj)[jid].astype(float))[0] for _ in range(B)]
        res[lim_name] = dict(kappa_star=k0, ci=np.nanpercentile(bs, [2.5, 97.5]).tolist(), ratio={f'{k:g}': float(v) for k, v in zip(KR, rt)})
    return res


def analysis_D(outs, nets, B, rng):
    res = {}
    for gen, o in outs.items():
        sq = np.sign(np.array([dQ(o['zq']['E1'][j], 0.3) for j in range(len(o['y']))]) / -2)   # Q_0.3 = -dQ/2 for full conjugation
        ok = sq != 0; r = {}
        for nm in nets:
            d = ((o['S'][('E1', nm)] - o['S'][('base', nm)]) * sq)[ok]
            r[nm] = dict(mean=float(d.mean()), ci=np.percentile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)], [2.5, 97.5]).tolist())
        for arch in ('part', 'pnet'):
            r[f'{arch}_s1_over_s0'] = r[f'{arch}_kinpid_s1']['mean'] / r[f'{arch}_kinpid']['mean']
        res[gen] = r
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, default=5931); ap.add_argument('--smoke', type=int, default=0)
    a = ap.parse_args()
    from interp import h59_charge_kappa as H59
    from interp import data_qg
    per = a.smoke or 5000; B = 100 if a.smoke else 500; tag = '_smoke' if a.smoke else ''
    RES.mkdir(parents=True, exist_ok=True)
    M, perms, dev = load_models(); nets = TRAINED + perms
    log('device', dev, 'nets', nets)
    LR = H59.fit_lrs(a.seed, data_qg.sample(per, a.seed)['idx'])
    outs = {gen: run_gen(gen, M, nets, LR, per, a.seed) for gen in ('py', 'hw')}
    rng = np.random.default_rng(a.seed + 7)
    res = dict(seed=a.seed, per_class=per, n_boot=B, nets=nets, checks={g: o['checks'] for g, o in outs.items()},
               unedited={g: o['une'] for g, o in outs.items()}, A={}, Badd={}, C={}, A_gluon={}, C_gluon={})
    for nm in nets:
        res['A'][nm] = analysis_A(outs, nm, 1, B, rng)
        log(f'A {nm}: kappa all {res["A"][nm]["all"]["kappa"]} {res["A"][nm]["all"]["kappa_ci"]}, no leader {res["A"][nm]["no_leader"]["kappa"]} '
            f'{res["A"][nm]["no_leader"]["kappa_ci"]}, antisym {res["A"][nm]["antisym_all"]["kappa"]}, a {res["A"][nm]["all"]["a"]:.3f}; '
            f'S1 ratio to no-leader fit {res["A"][nm]["per_edit"].get("S1", {}).get("ratio_to_noleader_fit")}')
        if nm in TRAINED:
            res['Badd'][nm] = analysis_B(outs, nm, 1, B, rng); log(f'B {nm}: {res["Badd"][nm]}')
            res['C'][nm] = analysis_C(outs, nm, 1, min(B, 300), rng); log(f'C {nm}: k* all {res["C"][nm]["all"]["kappa_star"]:.3f} {res["C"][nm]["all"]["ci"]}, first128 {res["C"][nm]["first128"]["kappa_star"]:.3f}')
            res['A_gluon'][nm] = analysis_A(outs, nm, 0, min(B, 200), rng)
            res['C_gluon'][nm] = analysis_C(outs, nm, 0, 100, rng)
            log(f'gluon {nm}: kappa single all {res["A_gluon"][nm]["all"]["kappa"]}, no leader {res["A_gluon"][nm]["no_leader"]["kappa"]}, k* {res["C_gluon"][nm]["all"]["kappa_star"]:.3f}')
    for ln in ('LR_k0.3', 'LR_k0.5', 'LR_k1'):
        res['A'][ln] = analysis_A(outs, None, 1, 100, rng, LR=LR[ln]); res['C'][ln] = analysis_C(outs, None, 1, 50, rng, LR=LR[ln])
        log(f'A {ln}: kappa all {res["A"][ln]["all"]["kappa"]}, no leader {res["A"][ln]["no_leader"]["kappa"]}; S1 ratio {res["A"][ln]["per_edit"]["S1"]["ratio_to_noleader_fit"]:.3f}; k* {res["C"][ln]["all"]["kappa_star"]:.3f}')
    res['D'] = analysis_D(outs, nets, B, rng); log('D', json.dumps(res['D'])[:1500])
    res['LR'] = {k: LR[k] for k in LR}
    p = RES / f'h59_verify_kappa{tag}.json'; p.write_text(json.dumps(res, indent=1, default=float)); log('wrote', p)


if __name__ == '__main__':
    main()
