"""X13 verification (fresh sample, trained networks only; no training).

Re-measures, on sample(per_class=1000, seed=2727) with a fresh fit / test split (rng 2727) and a ridge read-out whose
alpha-selection quarter is a random quarter of the fit half (X13 used the last quarter of the index-sorted fit half):
  R      = mean_test d(T) / mean_test d(dil)   (T = 0.06 rad translation, mean over 8 azimuths)
  R_rad  = mean_test d(R_k) / mean_test d(dil) (radial-only edit, same r_i as T at fixed azimuth)
  rot    = rigid rotation about the stored axis (null)
for the 56 trained networks of X13 (edits and network wrappers imported from interp.x13_axis_size; A' edits dropped).
Added control (input-only "oracle" read-outs, no network): R and R_rad of
  log m                          (true invariant mass; translation-invariant)
  1/2 log(M0 M2)                 (second moment about the stored axis; the "radial" quantity)
  1/2 log(M0 M2 - |M1|^2)        (recentred second moment; translation-invariant)
  w log m + (1-w) 1/2 log(M0 M2) (linear mixtures, w = 0.25, 0.5, 0.75)
These give the R_rad expected for a read-out that tracks true mass only (R = 0) and for one that tracks the axis moment only
(R = 1).  Mixture prediction: R_rad = rho + (1 - rho) R, rho = R_rad of the log m oracle.
Groups:  --group jax    official JEDI-linear n32/64/128, perminv n64, 8 X12 retrains, 9 PFN/EFN, oracles
         --group torch  X0/X1 small torch models (run with /opt/conda/bin/python)
         --analyze      -> $AI_RESULTS/jedi/x13_verify_summary.json
Outputs: $AI_RESULTS/jedi/x13_verify_{jax,torch}.json, x13_verify_summary.json.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import interp.x13_axis_size as X

SEED = 2727; X.SEED = SEED
OUT = X.OUT; B = 1000; T0 = time.time()
EDITS = ['0'] + [f'T{k}' for k in range(8)] + [f'R{k}' for k in range(8)] + ['dil', 'rot']


def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def ratio_stats(dT, dR, dD, dRot, I):
    m = lambda v: float(v.mean())
    bt, br, bd, bo = dT[I].mean(1), dR[I].mean(1), dD[I].mean(1), dRot[I].mean(1)
    return dict(R=m(dT) / m(dD), R_rad=m(dR) / m(dD), Q_rad=m(dR) / m(dT) if abs(m(dT)) > 1e-12 else None,
                rot_over_dil=m(dRot) / m(dD), mean=dict(T=m(dT), R=m(dR), D=m(dD), rot=m(dRot)),
                ci=dict(R=X.ci(bt / bd), R_rad=X.ci(br / bd), rot_over_dil=X.ci(bo / bd), T_minus_rot=X.ci(bt - bo)),
                p_T_vs_rot=X.bp(bt - bo), rep_R=(bt / bd).tolist(), rep_Rrad=(br / bd).tolist())


class State:
    def __init__(self):
        self.x128, self.y, self.fit, self.te = X.load_sample()
        self.fit_shuf = np.random.default_rng(SEED + 1).permutation(self.fit)    # random alpha-selection quarter
        self.I = np.random.default_rng(272730).integers(0, len(self.te), (B, len(self.te)))
        self.cache = {}
        log(f'sample(per_class=1000, seed={SEED}): N {len(self.y)}, fit {len(self.fit)}, test {len(self.te)}; '
            f'classes in last quarter of sorted fit (X13 alpha set): {np.bincount(self.y[self.fit[3 * len(self.fit) // 4:]], minlength=5).tolist()}')

    def edits(self, n):
        if n not in self.cache:
            E = X.make_edits(self.x128[:, :n]); E = {k: E[k] for k in EDITS}
            self.cache[n] = (E, np.log(np.maximum(X.moments(self.x128[:, :n])['m'], 1e-3)))
        return self.cache[n]


def measure(st, name, fwd, n, meta):
    E, logm = st.edits(n); t1 = time.time(); fit, te = st.fit_shuf, st.te
    R0 = fwd(E['0']); acc = float((R0['logits'].argmax(1) == st.y).mean())
    rd = X.Ridge(R0['last'][fit], logm[fit])
    p0 = rd(R0['last'][te]); r2 = float(1 - ((p0 - logm[te]) ** 2).sum() / ((logm[te] - logm[te].mean()) ** 2).sum())
    d = {e: rd(fwd(E[e])['last'][te]) - p0 for e in EDITS[1:]}
    dT = np.mean([d[f'T{k}'] for k in range(8)], 0); dR = np.mean([d[f'R{k}'] for k in range(8)], 0)
    row = dict(name=name, n=n, **meta, acc=acc, readout_r2=r2, **ratio_stats(dT, dR, d['dil'], d['rot'], st.I))
    log(f'{name:30s} n{n:<3d} acc {acc:.4f} R² {r2:.3f} | R {row["R"]:+.3f} {row["ci"]["R"]} R_rad {row["R_rad"]:.3f} '
        f'rot/dil {row["rot_over_dil"]:+.3f} [{time.time() - t1:.0f}s]')
    return row


def oracles(st):
    rows = []
    for n in (64,):
        E, _ = st.edits(n)
        def feats(x):
            M = X.moments(x); m1 = (M['M1'] ** 2).sum(1)
            return dict(logm=np.log(np.maximum(M['m'], 1e-3)), axisM2=0.5 * np.log(M['M0'] * M['M2']),
                        recM2=0.5 * np.log(np.maximum(M['M0'] * M['M2'] - m1, 1e-12)))
        F = {e: feats(E[e][st.te]) for e in EDITS}
        for w in (0.25, 0.5, 0.75):
            for e in EDITS: F[e][f'mix{w}'] = w * F[e]['logm'] + (1 - w) * F[e]['axisM2']
        for k in F['0']:
            d = {e: F[e][k] - F['0'][k] for e in EDITS[1:]}
            dT = np.mean([d[f'T{j}'] for j in range(8)], 0); dR = np.mean([d[f'R{j}'] for j in range(8)], 0)
            r = dict(name=f'oracle:{k}', n=n, **ratio_stats(dT, dR, d['dil'], d['rot'], st.I)); r.pop('rep_R'); r.pop('rep_Rrad')
            rows.append(r); log(f'oracle {k:10s}: R {r["R"]:+.3f} R_rad {r["R_rad"]:.3f} {r["ci"]["R_rad"]}')
    return rows


def run(group):
    st = State(); rows = []; orc = []
    fo = OUT / f'x13_verify_{group}.json'
    try:
        if group == 'jax':
            orc = oracles(st)
            from interp.jedi_core import Jedi
            for nm, n, fam in sorted(X.OFFICIAL, key=lambda t: t[1]):
                j = Jedi(n, fam); rows.append(measure(st, nm, X.jedi_fwd(j), n, dict(family=nm, cls='jedi-linear', seed=0))); del j; X.free()
            for m in X.X12:
                net, _ = X.x12_net(m); arm = m.split('_s')[0]
                rows.append(measure(st, f'x12-{m}', X.jedi_fwd(net), 64, dict(family=f'x12-{arm}', seed=int(m.split('_s')[1]),
                                                                            cls='jedi-linear' if arm != 'float' else 'jedi-linear-float')))
                del net; X.free()
            from interp.x9_efn_pfn import Flow
            for nm in X.FLOWS:
                F = Flow(nm)
                f = lambda x, F=F: (lambda r: dict(logits=r['logits'], last=r['h']))(F.run(x, keep=('L', 'h', 'logits')))
                rows.append(measure(st, nm, f, 64, dict(family=nm.rsplit('_s', 1)[0], cls=nm.split('_')[0], seed=int(nm.rsplit('_s', 1)[1]))))
        elif group == 'torch':
            import importlib.util, torch
            torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
            spec = importlib.util.spec_from_file_location('small_models', ROOT / 'train' / 'models.py')
            M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
            for run_ in X.S64 + X.S32:
                for p in sorted((X.SMALL / run_).glob('seed*')):
                    if not (p / 'best.pt').exists(): continue
                    arch = json.loads((p / 'arch.json').read_text())
                    net = M.build_model(arch['spec'], arch['n_particles'], cls_token=False, final_ln=arch['final_ln'])
                    net.load_state_dict(torch.load(p / 'best.pt', map_location='cpu')); net.eval()
                    rows.append(measure(st, f'{run_}_s{p.name[4:]}', X.torch_runner(net, arch), arch['n_particles'],
                                        dict(family=run_, cls=arch['spec']['arch'], seed=int(p.name[4:]))))
        else: raise ValueError(group)
    finally:
        fo.write_text(json.dumps(X.clean(dict(group=group, sample=dict(per_class=1000, seed=SEED, N=len(st.y), N_test=len(st.te)),
                                              rows=rows, oracles=orc))))
        log('wrote', fo, len(rows))


def analyze():
    from scipy import stats
    rows, orc = [], []
    for g in ('jax', 'torch'):
        G = json.loads((OUT / f'x13_verify_{g}.json').read_text()); rows += G['rows']; orc += G['oracles']
    O = json.loads((OUT / 'x13_summary.json').read_text())['nets']
    names = [r['name'] for r in rows]; log(len(rows), 'networks')
    R = np.array([r['R'] for r in rows]); Rr = np.array([r['R_rad'] for r in rows])
    RA = np.stack([r['rep_R'] for r in rows], 1); RB = np.stack([r['rep_Rrad'] for r in rows], 1)
    rng = np.random.default_rng(272740); S = dict(sample=dict(per_class=1000, seed=SEED), n_nets=len(rows))
    # 1. reproduction of per-network R, R_rad against X13 (seed 1313)
    Ro = np.array([O[n]['last']['R'] for n in names]); Rro = np.array([O[n]['last']['R_rad'] for n in names])
    S['repro'] = dict(spearman_R=X.spearman(R, Ro), pearson_R=float(np.corrcoef(R, Ro)[0, 1]), max_abs_dR=float(np.abs(R - Ro).max()),
                      median_abs_dR=float(np.median(np.abs(R - Ro))), spearman_Rrad=X.spearman(Rr, Rro),
                      worst=sorted([(n, float(a), float(b)) for n, a, b in zip(names, R, Ro)], key=lambda t: -abs(t[1] - t[2]))[:6])
    # 2. R vs R_rad
    bs = []; ols = []
    A = np.c_[np.ones_like(R), R]
    for b in range(B):
        j = rng.integers(0, len(R), len(R)); bs.append(X.spearman(RA[b, j], RB[b, j]))
        ols.append(np.linalg.lstsq(np.c_[np.ones(len(j)), RA[b, j]], RB[b, j], rcond=None)[0])
    perm = [X.spearman(R, rng.permutation(Rr)) for _ in range(10000)]; rho = X.spearman(R, Rr)
    coef = np.linalg.lstsq(A, Rr, rcond=None)[0]; ols = np.array(ols)
    om = {o['name']: o for o in orc}; rho_m = om['oracle:logm']['R_rad']
    pred = rho_m + (1 - rho_m) * R; res = Rr - pred
    resb = np.array([(RB[b] - (rho_m + (1 - rho_m) * RA[b])).mean() for b in range(B)])
    S['R_vs_Rrad'] = dict(spearman=rho, ci=X.ci(bs), p_perm=float((1 + np.sum(np.abs(perm) >= abs(rho) - 1e-12)) / 10001),
                          ols_intercept=float(coef[0]), ols_slope=float(coef[1]), ols_ci_intercept=X.ci(ols[:, 0]), ols_ci_slope=X.ci(ols[:, 1]),
                          mixture_pred_intercept=rho_m, mixture_pred_slope=1 - rho_m,
                          mixture_residual_mean=float(res.mean()), mixture_residual_ci_jets=X.ci(resb),
                          mixture_residual_sd=float(res.std(ddof=1)), Rrad_sd_log=float(np.log(Rr).std(ddof=1)),
                          Rrad_median=float(np.median(Rr)), Rrad_iqr=[float(np.quantile(Rr, .25)), float(np.quantile(Rr, .75))],
                          R_median=float(np.median(R)), R_iqr=[float(np.quantile(R, .25)), float(np.quantile(R, .75))])
    S['oracles'] = orc
    S['negative_dT'] = [n for n, r in zip(names, rows) if r['mean']['T'] < 0]
    S['T_vs_rot_sign'] = dict(T_gt_rot=int(sum(r['ci']['T_minus_rot'][0] > 0 for r in rows)),
                              T_lt_rot=int(sum(r['ci']['T_minus_rot'][1] < 0 for r in rows)))
    # 3. architecture vs seed
    fam = np.array([r['family'] for r in rows]); cls = np.array([r['cls'] for r in rows])
    def eta2(v, lab):
        sst = ((v - v.mean()) ** 2).sum(); return float(sum((lab == l).sum() * (v[lab == l].mean() - v.mean()) ** 2 for l in np.unique(lab)) / sst)
    VD = {}
    for key, tf_ in (('logR', np.log), ('R_linear', lambda q: q)):
        ok = np.isfinite(R) & ((R > 0) | (key == 'R_linear'))
        multi = np.array([ok[i] and ((fam == fam[i]) & ok).sum() >= 2 for i in range(len(R))])
        v, lf = tf_(R[multi]), fam[multi]; e = eta2(v, lf); pe = np.array([eta2(v, rng.permutation(lf)) for _ in range(10000)])
        vc, lc = tf_(R[ok]), cls[ok]; ec = eta2(vc, lc); pc = np.array([eta2(vc, rng.permutation(lc)) for _ in range(10000)])
        VD[key] = dict(n=int(multi.sum()), n_fam=int(len(np.unique(lf))), between_share=e, perm_q95=float(np.quantile(pe, .95)),
                       perm_p=float((1 + (pe >= e - 1e-12).sum()) / 10001), class_share=ec, class_q95=float(np.quantile(pc, .95)),
                       class_p=float((1 + (pc >= ec - 1e-12).sum()) / 10001),
                       within_sd=float(np.sqrt(np.mean([v[lf == f].var(ddof=1) for f in np.unique(lf)]))), total_sd=float(v.std(ddof=1)))
    # without the attention models (Linformer / transformer): is the between-family share above permutation there?
    keep = ~np.isin(cls, ['tf', 'linf']) & (R > 0)
    multi = np.array([keep[i] and ((fam == fam[i]) & keep).sum() >= 2 for i in range(len(R))])
    v, lf = np.log(R[multi]), fam[multi]; e = eta2(v, lf); pe = np.array([eta2(v, rng.permutation(lf)) for _ in range(10000)])
    VD['logR_no_attention'] = dict(n=int(multi.sum()), n_fam=int(len(np.unique(lf))), between_share=e, perm_q95=float(np.quantile(pe, .95)),
                                   perm_p=float((1 + (pe >= e - 1e-12).sum()) / 10001))
    S['variance'] = VD
    # 4. key seed pairs
    PD = {}
    for a_, b_ in (('tf-1b-d32-n64_s0', 'tf-1b-d32-n64_s1'), ('x12-float_s0', 'x12-float_s1'), ('tf-1b-d32-n32_s0', 'tf-1b-d32-n32_s1'),
                   ('linf-1b-d32-k4-n32_s0', 'linf-1b-d32-k4-n32_s1'), ('pfn_64-64-32_s0', 'pfn_64-64-32_s1'), ('x12-hgq_s0', 'x12-hgq_s2')):
        if a_ in names and b_ in names:
            i, j = names.index(a_), names.index(b_); db = RA[:, i] - RA[:, j]
            PD[f'{a_} - {b_}'] = dict(R_a=float(R[i]), R_b=float(R[j]), diff=float(R[i] - R[j]), ci=X.ci(db), p=X.bp(db),
                                      x13=[O[a_]['last']['R'], O[b_]['last']['R']])
    S['seed_pairs'] = PD
    S['nets'] = {r['name']: {k: v for k, v in r.items() if not k.startswith('rep_')} for r in rows}
    for k in ('repro', 'R_vs_Rrad', 'negative_dT', 'T_vs_rot_sign', 'variance', 'seed_pairs'): log(k, json.dumps(X.clean(S[k])))
    (OUT / 'x13_verify_summary.json').write_text(json.dumps(X.clean(S), indent=1)); log('wrote summary')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--group', default=''); ap.add_argument('--analyze', action='store_true')
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    analyze() if a.analyze else run(a.group)
