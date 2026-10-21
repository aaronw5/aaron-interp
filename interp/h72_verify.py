"""H72 verify (loop D, skeptic): fresh rows; is the Rm+m loss of the top taggers a particle-count effect?

H72 found that removing the particles beyond r_cut and then restoring the jet's m and pT (Rm+m) still costs the JetClass /
Top ParT, PN and LorentzNet most of their top calls, and read this as "consistent with the D24 count channel". D24 v2 says the
same networks LOWER their top score when the object count RISES at fixed IRC-safe content, i.e. they should RAISE it when N
falls at fixed m. This script tests the count reading directly on fresh rows.

Arms (r_cut index i = 1, 2: 0.4 / 0.3 for R = 0.8 jets, 0.2 / 0.15 for QG), built with interp.h72_angular_reach.conditions:
  rm<i>, rmm<i>, crm<i>   as H72 (remove outer particles; + m, pT restore; pT-matched inside removal + restore)
  rmmsp<i>  rmm<i>, then exact-content count restore: k = N0 - N(rmm) collinear splits (a random real particle split into two
            pT/2 halves 0.01 apart, H1 split z = 0.5, dR = 0.01; jet four-momentum and IRC-safe observables fixed up to the
            0.01 opening) so that N returns to the unedited N0. rmmsp - rmm = the effect of the count alone at the rmm content.
  spl<i>    k = (number of outer particles) such splits on the UNEDITED jet: count up at fixed IRC-safe content (D24 direction).
Observables on 500 random signal jets (JetClass t, Top t): m, m_SD, tau21, tau32, N (h1_irc_panel.observables) under base,
  rm1, rmm1, rmmsp1; QG: m, girth, N, pT^D.
Statistics: signal jets; ds = mean change of the log-odds score (signal vs QCD / quark vs gluon) with bootstrap 95 % CI (1000);
  d_effS at the threshold keeping 50 % of unedited signal; pairs: paired bootstrap CI, sign-flip permutation p (10^4), exact
  McNemar on the calls; Holm over all pair tests.
Run: ./part_job.sh d-h72-verify 'python -m interp.h72_verify'
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import h72_angular_reach as H72  # noqa: E402
from interp import h65_pileup_dose as H65  # noqa: E402
from interp.h1_irc_panel import finalize, split, rank_slot, holm, binom_two_sided  # noqa: E402
from interp.h18_softcount_cross import nreal, auc  # noqa: E402

RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); OUT = RES / 'physics'
H72.SEEDS.update(hls=7261, jc=7262, top=7263, qg=7264)      # same dict object as H65.SEEDS: fresh rows everywhere
PER = 2000; IDX = (1, 2); NOBS = 500; B = 1000
T0_ = time.time()


def log(*a):
    print(f'[{time.time() - T0_:7.0f}s]', *a, flush=True)


def split_to(T, need, cap, rng):
    """need (J,) collinear splits of random real particles per jet (z 0.5, dR 0.01)"""
    U = finalize(T, cap)[0]; need = need.astype(int).copy()
    while (need > 0).any():
        s = rank_slot(U, 'rnd', rng); s = np.where(need > 0, s, -1)
        U, ok = split(U, s, 0.5, 0.01, rng); U = finalize(U, cap)[0]; need -= ok.astype(int)
    return U


def ci(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v.size else [None, None]


def boot_mean(d, rng):
    n = len(d)
    if n == 0: return [None, None]
    return ci(d[rng.integers(0, n, (B, n))].mean(1))


def signflip(d, seed=1, P=10000):
    d = np.asarray(d, float)
    if len(d) == 0 or np.all(d == 0): return 1.0
    rng = np.random.default_rng(seed); obs = abs(d.mean()); c = 0
    for a in range(0, P, 500):
        sg = rng.integers(0, 2, (min(500, P - a), len(d))) * 2.0 - 1.0; c += int((np.abs((sg * d).mean(1)) >= obs - 1e-12).sum())
    return (c + 1) / (P + 1)


def obs_qg(Uf):
    pt = Uf['pt']; S = pt.sum(1)
    P = H72.jet4(Uf); m = H72._m4(P)
    return dict(m=m, girth=H72.girth(Uf), N=nreal(Uf).astype(float), ptD=np.sqrt((pt ** 2).sum(1)) / np.maximum(S, 1e-12))


def run(ds):
    H65.PER = dict(H65.PER); H65.PER[ds] = PER
    T0, y, info, J0 = H65.load(ds, 0); T0 = finalize(T0, H72.CAP[ds])[0]; ae, ap = T0['ax_eta'], T0['ax_phi']
    cls = list(H72.SIG[ds])[-1] if ds == 'jc' else list(H72.SIG[ds])[0]          # jc: 't'
    k = H72.SIG[ds][cls]; rows = np.isin(y, [k] + H72.QCD[ds]); sig = y == k
    scorer, _ = H65.make_scorer(ds, J0); N0 = nreal(T0)
    conds = {}; extra = {}
    keep = {'base'} | {f'{e}{i}' for i in IDX for e in ('rm', 'rmm', 'crm')}
    for name, U, ok, inf in H72.conditions(T0, ds):
        if name not in keep: continue
        conds[name] = (finalize(U, H72.CAP[ds])[0], ok)
        if name.startswith('rmm'): extra[name] = dict(b=inf['b'], nout=inf['nout'])
    rng = np.random.default_rng([H72.SEEDS[ds], 77])
    for i in IDX:
        V, ok = conds[f'rmm{i}']; need = np.maximum(N0 - nreal(V), 0) * ok
        conds[f'rmmsp{i}'] = (split_to(V, need, H72.CAP[ds], rng), ok)
        conds[f'spl{i}'] = (split_to(T0, extra[f'rmm{i}']['nout'], H72.CAP[ds], rng), np.ones(len(y), bool))
    S = {}; R = dict(ds=ds, cls=cls, n=int(len(y)), n_sig=int(sig.sum()), seed=H72.SEEDS[ds], r_cut=[H72.rcuts(ds)[i] for i in IDX],
                     sample=info.get('sample'), unedited={}, edits={}, models={})
    for c, (Uf, ok) in conds.items():
        L = scorer(Uf, ae, ap)
        for mn, l in L.items():
            sc = H65.class_scores(np.asarray(l, float), ds)[cls]; S.setdefault(mn, {})[c] = sc
        R['edits'][c] = dict(valid_sig=float(ok[sig].mean()), N_sig=float(nreal(Uf)[sig].mean()), dN_sig=float((nreal(Uf) - N0)[sig].mean()))
        if c.startswith('rmm') and c in extra:
            b = extra[c]['b'][sig & ok]; R['edits'][c].update(b_median=float(np.median(b)), b_q90=float(np.quantile(b, 0.9)))
        P = H72.jet4(Uf); P0 = H72.jet4(T0)
        R['edits'][c].update(dm_sig=float((H72._m4(P) - H72._m4(P0))[sig & ok].mean()),
                             dpt_sig=float((np.hypot(P[:, 0], P[:, 1]) - np.hypot(P0[:, 0], P0[:, 1]))[sig & ok].mean()))
        log(ds, c, R['edits'][c], ' '.join(f'{mn} {np.nanmean((S[mn][c] - S[mn]["base"])[sig & ok]):+.3f}' for mn in S))
    for mn in S:
        b0 = S[mn]['base']; R['unedited'][mn] = dict(AUC=auc(b0[rows], sig[rows].astype(int)))
    # observables
    rs = np.random.default_rng([H72.SEEDS[ds], 78]); oj = np.sort(rs.choice(np.flatnonzero(sig), min(NOBS, sig.sum()), replace=False))
    OB = {}
    for c in ('base', 'rm1', 'rmm1', 'rmmsp1'):
        Uf = conds[c][0]; sub = {kk: (v[oj] if isinstance(v, np.ndarray) else {k2: v2[oj] for k2, v2 in v.items()}) for kk, v in Uf.items()}
        if ds == 'qg': OB[c] = obs_qg(sub)
        else:
            from interp.h1_irc_panel import observables
            OB[c] = observables(sub); OB[c]['girth'] = H72.girth(sub)
        log(ds, 'observables', c, 'done')
    okj = conds['rmm1'][1][oj]
    R['observables_sig'] = {c: {o: dict(mean=float(np.mean(v[okj])), d_mean=float(np.mean((v - OB['base'][o])[okj])))
                                for o, v in OB[c].items()} for c in OB}
    R['observables_sig']['n'] = int(okj.sum())
    # statistics
    pairs = [('rm', 'base'), ('rmm', 'base'), ('rmmsp', 'base'), ('spl', 'base'), ('rmmsp', 'rmm'), ('rmm', 'crm')]
    tests = []
    for mn in S:
        b0 = S[mn]['base']; thr = float(np.median(b0[sig])); M = dict(threshold=thr, cond={}, pairs={})
        for c in conds:
            if c == 'base': continue
            ok = conds[c][1] & sig & np.isfinite(S[mn][c]); d = (S[mn][c] - b0)[ok]
            M['cond'][c] = dict(n=int(ok.sum()), ds=float(d.mean()), ds_ci=boot_mean(d, rng),
                                d_effS=float((S[mn][c][ok] > thr).mean() - (b0[ok] > thr).mean()))
        for i in IDX:
            for a, c in pairs:
                an, cn = f'{a}{i}', (c if c == 'base' else f'{c}{i}')
                ok = conds[an][1] & conds[cn][1] & sig & np.isfinite(S[mn][an]) & np.isfinite(S[mn][cn])
                d = (S[mn][an] - S[mn][cn])[ok]; ca = S[mn][an][ok] > thr; cc = S[mn][cn][ok] > thr
                n01 = int((ca & ~cc).sum()); n10 = int((~ca & cc).sum())
                e = dict(n=int(ok.sum()), dds=float(d.mean()), dds_ci=boot_mean(d, rng), p_perm=signflip(d, seed=i),
                         d_effS=float(ca.mean() - cc.mean()), mcnemar=dict(arm_only=n01, ctrl_only=n10,
                                                                           p=float(binom_two_sided(n01, n01 + n10)) if n01 + n10 else 1.0))
                M['pairs'][f'{an}-{cn}'] = e
                if mn not in H72.NULLS: tests.append((ds, mn, f'{an}-{cn}', e))
        # fraction of the rm score loss left after the m, pT restore and after the count restore
        for i in IDX:
            r = M['cond'][f'rm{i}']['ds']
            M[f'frac_rmm_of_rm_{i}'] = M['cond'][f'rmm{i}']['ds'] / r if r else None
            M[f'frac_rmmsp_of_rm_{i}'] = M['cond'][f'rmmsp{i}']['ds'] / r if r else None
        R['models'][mn] = M
        log(ds, mn, 'AUC', R['unedited'][mn], ' '.join(f'{c} {v["ds"]:+.3f}' for c, v in M['cond'].items()))
    return R, tests


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True); ALL = {}; TESTS = []
    for ds in (sys.argv[1:] or ['jc', 'top', 'qg']):
        R, t = run(ds); ALL[ds] = R; TESTS += t
        (OUT / f'h72_verify_{ds}.json').write_text(json.dumps(R, indent=1, default=float))
    if TESTS:
        hp = holm([e['p_perm'] for *_, e in TESTS]); hm = holm([e['mcnemar']['p'] for *_, e in TESTS])
        for (ds, mn, pr, e), a, b in zip(TESTS, hp, hm):
            ALL[ds]['models'][mn]['pairs'][pr].update(p_perm_holm=float(a), p_mcn_holm=float(b))
        ALL['holm'] = dict(n_tests=len(TESTS), perm_sig=int(sum(h < 0.05 for h in hp)), mcn_sig=int(sum(h < 0.05 for h in hm)))
    (OUT / 'h72_verify.json').write_text(json.dumps(ALL, indent=1, default=float)); log('wrote h72_verify.json')
