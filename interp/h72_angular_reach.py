"""H72 (loop D, round D15; new D claim): angular reach -- from how far out of the jet axis does each tagger take information?

Removing, or pulling inward, the real particles beyond a radius r_cut from the jet axis tests whether a network follows plain
quantities that weight the periphery (jet mass m, girth), a groomed / core quantity (soft-drop mass m_SD), or the particle count N.

Geometry. Delta R_i = sqrt(d_eta_i^2 + d_phi_i^2) of particle i from the E-scheme axis (direction of the summed constituent
3-momenta of the particles the model sees, after the slot cut). r_cut = f * R, f in {0.75, 0.5, 0.375, 0.25}: r_cut = 0.6 / 0.4 /
0.3 / 0.2 for R = 0.8 jets (hls4ml, JetClass, Top Landscape), 0.3 / 0.2 / 0.15 / 0.1 for the R = 0.4 quark-gluon jets.
'Outer' particles = real particles with Delta R > r_cut.
Arms (edits of the real jet)
  rm<i>   remove every outer particle (N falls, pT falls, mass and girth fall)
  pi<i>   pull every outer particle radially in to Delta R = r_cut along its own direction from the axis, pT (and particle
          mass) fixed: N and summed pT unchanged, only angles change
  rmm<i>  rm, then restore the original jet m and pT: all remaining momenta scaled by one factor a (restores pT) and all angular
          offsets from the remaining jet's E-scheme axis scaled by one factor b (restores m); iterated 8 times; valid where
          |m/m0 - 1| < 2 %, |pT/pT0 - 1| < 1 % and b <= 5 (jets whose core alone has too little mass are excluded)
Controls
  cr<i>   remove a pT-matched set of random INSIDE particles (Delta R <= r_cut): for each outer particle (random order) the unused
          inside particle nearest in log pT is removed; same count N removed, summed pT removed within 20 % (valid mask), at
          least one inside particle left. Matches rm in N and pT lost; differs in where the removed particles were.
  rot<i>  every outer particle rotated about the axis to an independent random azimuth at its own Delta R (same annulus, same
          radii, same pT, angles scrambled). Matches pi in 'which particles are touched'; keeps their radii.
  crm<i>  cr followed by the same m, pT restore as rmm (control for rmm).
Pairs: rm - cr, pi - rot, rmm - crm. Model inputs are always taken about the stored (unedited) axis, as in H65.

Samples (random rows; seeds 7201-7204) and models: as H65 (interp.h65_pileup_dose.load / make_scorer), 2,000 jets per class:
  hls  jets of jedi_core.sample(2000, 7201), 5 classes; JEDI-linear n64 3-feature; JEDI with redrawn weights
  jc   JetClass first test file per class, QCD / Zqq / Wqq / Tbqq, seed 7202; ParT_full, ParT_kin, PN_full, PN_kin,
       ParT_full permuted, ParT_full untrained
  top  Top Landscape test, seed 7203; ParT, PN, LNet, PFN, EFN, PFN permuted
  qg   Komiske-Metodiev-Thaler Pythia test, seed 7204; part_kin, pnet_kin, pfn_k, efn_k, lorentznet, part_kin permuted
References on the same edited jets: win_m (plain-mass window), win_msd (m_SD, C/A, beta 0, z_cut 0.1, window), LR_girth
  (girth g = sum pT_i Delta R_i / sum pT_i about the edited jet's E-scheme axis, histogram likelihood ratio), LR_N, MLP_mptN
  (MLP on m, pT, N), GBDT_EFP (gradient-boosted trees on the 54 log energy-flow polynomials, d <= 5, beta 1). LR / MLP / GBDT
  are 2-fold cross-fit on the unedited sample (each jet scored by the half not trained on it).
Metrics (--part summary), signal jets of each task (signal class vs QCD); threshold at eps_S = 0.5 on unedited signal:
  d_effS  change of signal efficiency (fraction of signal jets still called signal); ds = mean change of the score (log-odds)
  arm - control: per-jet paired difference, bootstrap 95 % CI (B = 1000), sign-flip permutation p (10^4), exact McNemar on
  the calls (arm vs control); Holm across models x r_cut x arms (trained nets only).
  r_50 = the r_cut at which |d_effS| of an arm first reaches half its value at f = 0.25, walking in from (R, 0) with linear
  interpolation; bootstrap CI.
Run
  ./job.sh d-h72-hls 1 4 'python -m interp.h72_angular_reach --part run --ds hls --what both'
  ./part_job.sh d-h72-gpu 'python -m interp.h72_angular_reach --part run --ds jc --what nets && ... --ds top ... && ... --ds qg ...'
  AI_PY=conda ./job.sh d-h72-obs 1 4 '... --part run --ds top --what obs && ... qg && JETDISTILL_TAGGER=part ... jc'
  ./job.sh d-h72-efp 4 8 '/j-jepa-vol/aaron-interp/venv-ef/bin/python -m interp.h72_angular_reach --part efp --ds <ds>'
  ./job.sh d-h72-sum 1 4 'python -m interp.h72_angular_reach --part summary'
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp.h1_irc_panel import finalize, jet4, wrap, holm, binom_two_sided, copyT  # noqa: E402
from interp.h18_softcount_cross import nreal, auc, eaxis  # noqa: E402
from interp.h49_groomed_mass import softdrop, _m4  # noqa: E402
from interp import h65_pileup_dose as H65  # noqa: E402

RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results'))
OUT = RES / 'physics'
SEEDS = dict(hls=7201, jc=7202, top=7203, qg=7204)
H65.SEEDS = SEEDS                         # H65.load / make_scorer read the module-level seeds at call time
FR = (0.75, 0.5, 0.375, 0.25)
RJ, CAP, SIG, QCD, KNOWN = H65.RJ, H65.CAP, H65.SIG, H65.QCD, H65.KNOWN
NULLS = H65.NULLS
ARMS = ('rm', 'pi', 'rmm'); PAIR = dict(rm='cr', pi='rot', rmm='crm')
KINDS = ('rm', 'pi', 'rmm', 'cr', 'rot', 'crm')
CONDS = ['base'] + [f'{e}{i}' for i in range(len(FR)) for e in KINDS]
BMAX = 5.0
T0_ = time.time()


def log(*a):
    print(f'[{time.time() - T0_:7.0f}s]', *a, flush=True)


def sfx(smoke):
    return '_smoke' if smoke else ''


def rcuts(ds):
    return [round(f * RJ[ds], 4) for f in FR]


# ===================================================================================================================== edits
def geom(T):
    ee, ep = eaxis(T); real = T['pt'] > 0
    de = np.where(real, T['eta'] - ee[:, None], 0.0); dp = np.where(real, wrap(T['phi'] - ep[:, None]), 0.0)
    return ee, ep, de, dp, np.hypot(de, dp), real


def girth(T):
    ee, ep, de, dp, dr, real = geom(T)
    return (T['pt'] * dr).sum(1) / np.maximum(T['pt'].sum(1), 1e-12)


def remove(T, mask):
    U = copyT(T); U['pt'] = np.where(mask, 0.0, U['pt']); return U


def move(T, out, ee, ep, de, dp, fac=None, theta=None, dr=None):
    U = copyT(T)
    if fac is not None:
        ne, npp = de * fac, dp * fac
    else:
        ne, npp = dr * np.cos(theta), dr * np.sin(theta)
    U['eta'] = np.where(out, ee[:, None] + ne, U['eta']); U['phi'] = np.where(out, wrap(ep[:, None] + npp), U['phi'])
    return U


def match_inside(T, out, rng):
    """pT-matched random inside removal; returns (mask of inside particles to remove, valid (J,), pT-loss ratio (J,))"""
    pt = T['pt']; real = pt > 0; J = len(pt); rem = np.zeros_like(real); ok = np.ones(J, bool); ratio = np.ones(J)
    for j in np.flatnonzero(out.any(1)):
        po = pt[j, out[j]]; cand = np.flatnonzero(real[j] & ~out[j]); used = np.zeros(len(cand), bool)
        if len(cand) <= len(po): ok[j] = False
        lc = np.log(pt[j, cand]) + 1e-9 * rng.random(len(cand))
        for i in rng.permutation(len(po)):
            if used.all(): break
            d = np.abs(lc - math.log(po[i])); d[used] = np.inf; used[int(np.argmin(d))] = True
        rem[j, cand[used]] = True; ratio[j] = pt[j, cand[used]].sum() / po.sum()
        ok[j] &= (abs(ratio[j] - 1) < 0.2) & (used.sum() == len(po)) & (~used).any()
    return rem, ok, ratio


def restore(U, P0, iters=8):
    """scale momenta (restores pT) and angular offsets about the jet's own E-scheme axis (restores m); returns (table, valid, b)"""
    V = copyT(U); real = V['pt'] > 0; m0 = _m4(P0); pt0 = np.hypot(P0[:, 0], P0[:, 1]); Bc = np.ones(len(m0))
    for _ in range(iters):
        P = jet4(V); a = pt0 / np.maximum(np.hypot(P[:, 0], P[:, 1]), 1e-12)
        V['pt'] = V['pt'] * a[:, None]; V['m'] = V['m'] * a[:, None]
        m = _m4(jet4(V)); b = np.clip(m0 / np.maximum(m, 1e-9), 1 / BMAX, BMAX / Bc); Bc = Bc * b
        ee, ep, de, dp, dr, _ = geom(V)
        V['eta'] = np.where(real, ee[:, None] + de * b[:, None], 0.0); V['phi'] = np.where(real, wrap(ep[:, None] + dp * b[:, None]), 0.0)
    P = jet4(V); m = _m4(P); pt = np.hypot(P[:, 0], P[:, 1])
    ok = (np.abs(m / np.maximum(m0, 1e-9) - 1) < 0.02) & (np.abs(pt / pt0 - 1) < 0.01) & (Bc <= BMAX * (1 + 1e-9)) & real.any(1)
    return V, ok, Bc


def conditions(T0, ds):
    """yields (name, edited table (unfinalized), valid mask (J,), info dict of per-jet arrays)"""
    J = len(T0['pt']); ee, ep, de, dp, dr, real = geom(T0); P0 = jet4(T0); seed = SEEDS[ds]
    yield 'base', T0, np.ones(J, bool), {}
    for i, r in enumerate(rcuts(ds)):
        out = real & (dr > r); nout = out.sum(1); ptout = (T0['pt'] * out).sum(1) / T0['pt'].sum(1)
        inf = dict(nout=nout, fpt_out=ptout)
        Urm = remove(T0, out); yield f'rm{i}', Urm, np.ones(J, bool), inf
        fac = np.where(out, r / np.maximum(dr, 1e-12), 1.0)
        yield f'pi{i}', move(T0, out, ee, ep, de, dp, fac=fac), np.ones(J, bool), inf
        V, ok, Bc = restore(Urm, P0); yield f'rmm{i}', V, ok, dict(inf, b=Bc)
        rng = np.random.default_rng([seed, 11, i]); rem, okc, ratio = match_inside(T0, out, rng)
        Ucr = remove(T0, rem); yield f'cr{i}', Ucr, okc, dict(inf, ratio=ratio)
        th = 2 * np.pi * np.random.default_rng([seed, 12, i]).random(dr.shape)
        yield f'rot{i}', move(T0, out, ee, ep, de, dp, theta=th, dr=dr), np.ones(J, bool), inf
        V2, ok2, B2 = restore(Ucr, P0); yield f'crm{i}', V2, okc & ok2, dict(inf, b=B2)


# ===================================================================================================================== run
def run(ds, what, smoke):
    OUT.mkdir(parents=True, exist_ok=True); s = sfx(smoke)
    H65.PER = dict(H65.PER); H65.PER[ds] = 2000
    T0, y, info, J0 = H65.load(ds, smoke); T0 = finalize(T0, CAP[ds])[0]; W = T0['pt'].shape[1]
    ae, ap = T0['ax_eta'], T0['ax_phi']
    ee, _, _, _, dr0, real0 = geom(T0)
    log(f'{ds}: {len(y)} jets, slot limit {CAP[ds]}, <N> {nreal(T0).mean():.1f}; r_cut {rcuts(ds)}; median particle Delta R '
        f'{np.median(dr0[real0]):.3f}; axis offset stored vs E-scheme median {np.median(np.hypot(ee - ae, 0)):.4f}')
    do_obs = what in ('obs', 'both'); do_nets = what in ('nets', 'both')
    scorer, minfo = H65.make_scorer(ds, J0) if do_nets else (None, {})
    OBS = dict(y=y); NETS = dict(y=y); PARTS = {}; un = {}
    for name, U, ok, inf in conditions(T0, ds):
        t = time.time(); Uf = finalize(U, CAP[ds])[0]; n = nreal(Uf)
        OBS[f'n|{name}'] = n.astype(np.int16); NETS[f'n|{name}'] = n.astype(np.int16); OBS[f'ok|{name}'] = ok
        for k, v in inf.items(): OBS[f'{k}|{name}'] = np.asarray(v, np.float32)
        msg = f'{ds} {name}: valid {ok.mean():.3f}, <N> {n.mean():.1f}'
        if inf: msg += f', outer particles {inf["nout"].mean():.1f} ({100 * inf["fpt_out"].mean():.1f} % of pT)'
        if do_obs:
            P = jet4(Uf); OBS[f'm|{name}'] = _m4(P).astype(np.float32); OBS[f'pt|{name}'] = np.hypot(P[:, 0], P[:, 1]).astype(np.float32)
            OBS[f'msd|{name}'] = softdrop(Uf)['msd'].astype(np.float32); OBS[f'girth|{name}'] = girth(Uf).astype(np.float32)
            rl = Uf['pt'] > 0; off = np.r_[0, np.cumsum(rl.sum(1))]
            PARTS[f'off|{name}'] = off.astype(np.int64)
            PARTS[f'p|{name}'] = np.stack([Uf['pt'][rl], (Uf['eta'] - ae[:, None])[rl], wrap(Uf['phi'] - ap[:, None])[rl]], 1).astype(np.float32)
            msg += (f' | dm {np.mean(OBS[f"m|{name}"] - OBS["m|base"]):+.2f} dmsd {np.mean(OBS[f"msd|{name}"] - OBS["msd|base"]):+.2f} GeV'
                    f' dgirth {np.mean(OBS[f"girth|{name}"] - OBS["girth|base"]):+.4f} dpt {np.mean(OBS[f"pt|{name}"] - OBS["pt|base"]):+.1f}')
        if do_nets:
            S = scorer(Uf, ae, ap)
            for mn, L in S.items():
                L = np.asarray(L, np.float64)
                for cls, sc in H65.class_scores(L, ds).items(): NETS[f's|{mn}|{cls}|{name}'] = sc.astype(np.float32)
                if name == 'base':
                    un[mn] = {}
                    if L.ndim == 2:
                        a = L.argmax(1)
                        if ds == 'hls': un[mn]['acc_5class'] = float((a == y).mean())
                        un[mn]['per_class_acc'] = {str(k): float((a[y == k] == k).mean()) for k in sorted(set(y.tolist()))}
                    for cls, sc in H65.class_scores(L, ds).items():
                        rows = np.isin(y, [SIG[ds][cls]] + QCD[ds]); ys = (y[rows] == SIG[ds][cls]).astype(int)
                        un[mn][f'AUC_{cls}_vs_QCD'] = auc(sc[rows], ys); un[mn][f'acc_{cls}_s>0'] = float(((sc[rows] > 0) == ys).mean())
                    log(f'{ds} unedited {mn}: {un[mn]}  known: {KNOWN.get(mn, "-")}')
            c0 = list(SIG[ds])[0]; sg = y == SIG[ds][c0]
            msg += f' | ds({c0} jets) ' + ' '.join(f'{mn} {np.mean((NETS[f"s|{mn}|{c0}|{name}"] - NETS[f"s|{mn}|{c0}|base"])[sg]):+.3f}' for mn in S)
        log(msg + f' ({time.time() - t:.0f} s)')
    if do_obs:
        # example jets for the figure: one QCD and one signal jet near the class-median multiplicity
        nb = nreal(T0); ksig = SIG[ds]['t'] if 't' in SIG[ds] else SIG[ds][list(SIG[ds])[0]]
        for tag, cl in (('bkg', QCD[ds][0]), ('sig', ksig)):
            rr = np.flatnonzero(y == cl); j = int(rr[np.argmin(np.abs(nb[rr] - np.median(nb[rr])))])
            e_, p_, de, dp, dr, real = geom({k: (v[j:j + 1] if not isinstance(v, dict) else {}) for k, v in T0.items()})
            b = real[0]
            OBS.update({f'ex{tag}|j': np.array(j), f'ex{tag}|pt': T0['pt'][j, b], f'ex{tag}|de': de[0, b], f'ex{tag}|dp': dp[0, b]})
        np.savez_compressed(OUT / f'h72_{ds}_obs{s}.npz', **OBS)
        np.savez_compressed(OUT / f'h72_{ds}_parts{s}.npz', **PARTS)
        (OUT / f'h72_{ds}_obs{s}.json').write_text(json.dumps(dict(info, n=int(len(y)), r_cut=rcuts(ds)), indent=1, default=float))
        log(f'wrote h72_{ds}_obs{s}.npz, h72_{ds}_parts{s}.npz')
    if do_nets:
        np.savez_compressed(OUT / f'h72_{ds}_nets{s}.npz', **NETS)
        (OUT / f'h72_{ds}_nets{s}.json').write_text(json.dumps(dict(info, n=int(len(y)), unedited=un, models=minfo,
                                                                     known={k: KNOWN.get(k, '-') for k in un}), indent=1, default=str))
        log(f'wrote h72_{ds}_nets{s}.npz')


# ===================================================================================================================== EFP reference
def efp(ds, smoke, ncpu=4):
    from multiprocessing import Pool
    from sklearn.ensemble import HistGradientBoostingClassifier
    from interp import h14_qg_generator as G
    s = sfx(smoke); Pz = np.load(OUT / f'h72_{ds}_parts{s}.npz'); y = np.load(OUT / f'h72_{ds}_obs{s}.npz')['y']
    pool = Pool(ncpu); F = {}
    for c in CONDS:
        t = time.time(); off, p = Pz[f'off|{c}'], Pz[f'p|{c}'].astype(np.float64)
        L = [p[off[i]:off[i + 1]] if off[i + 1] > off[i] else np.array([[1e-3, 0.0, 0.0]]) for i in range(len(off) - 1)]
        F[c] = np.log(np.maximum(G.efps(L, pool), 1e-12)); log(f'{ds} {c}: EFPs {F[c].shape} ({time.time() - t:.0f} s)')
    pool.close()
    OUTZ = dict(y=y); info = {}
    fold = np.random.default_rng([SEEDS[ds], 21]).permutation(len(y)) % 2
    for cls, k in SIG[ds].items():
        rows = np.isin(y, [k] + QCD[ds]); g = (y == k).astype(int)
        for c in CONDS: OUTZ[f's|GBDT_EFP|{cls}|{c}'] = np.zeros(len(y), np.float32)
        for f in (0, 1):
            tr = rows & (fold != f); te = fold == f
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, early_stopping=True, validation_fraction=0.15,
                                               random_state=0).fit(F['base'][tr], g[tr])
            for c in CONDS: OUTZ[f's|GBDT_EFP|{cls}|{c}'][te] = m.decision_function(F[c][te]).astype(np.float32)
        info[cls] = dict(AUC=auc(OUTZ[f's|GBDT_EFP|{cls}|base'][rows], g[rows]))
        log(f'{ds} GBDT_EFP {cls}: cross-fit AUC {info[cls]["AUC"]:.4f}; ds(rm3, signal) '
            f'{np.mean((OUTZ[f"s|GBDT_EFP|{cls}|rm3"] - OUTZ[f"s|GBDT_EFP|{cls}|base"])[y == k]):+.3f}')
    np.savez_compressed(OUT / f'h72_{ds}_efp{s}.npz', **OUTZ)
    (OUT / f'h72_{ds}_efp{s}.json').write_text(json.dumps(info, indent=1))
    log(f'wrote h72_{ds}_efp{s}.npz')


# ===================================================================================================================== statistics
def ci(v):
    v = np.asarray(v, np.float64); v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v.size else [None, None]


def signflip_p(d, P=10000, seed=1):
    d = np.asarray(d, np.float64); n = len(d)
    if n == 0 or np.all(d == 0): return 1.0
    rng = np.random.default_rng(seed); obs = abs(d.mean()); cnt = 0
    for a in range(0, P, 500):
        sg = rng.integers(0, 2, (min(500, P - a), n)) * 2.0 - 1.0; cnt += int((np.abs((sg * d).mean(1)) >= obs - 1e-12).sum())
    return (cnt + 1) / (P + 1)


def mcnemar(a, b):
    n01 = int((a & ~b).sum()); n10 = int((~a & b).sum())
    return dict(arm_only=n01, ctrl_only=n10, p=float(binom_two_sided(n01, n01 + n10)) if n01 + n10 else 1.0)


def reach(vals):
    """vals: |d| at f = FR (0.75 .. 0.25), anchor 0 at f = 1; returns f at which |d| first reaches half of |d(0.25)| walking in"""
    f = np.r_[1.0, FR]; g = np.r_[0.0, np.abs(vals)]; tgt = 0.5 * g[-1]
    if not np.isfinite(tgt) or tgt <= 0: return float('nan')
    for a in range(len(f) - 1):
        if g[a] < tgt <= g[a + 1]:
            return float(f[a] + (tgt - g[a]) / (g[a + 1] - g[a]) * (f[a + 1] - f[a]))
        if g[a] >= tgt: return float(f[a])
    return float(f[-1])


def mlp_ref(O, rows, sig, seed):
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    X = {c: np.stack([O[f'm|{c}'], np.log(np.maximum(O[f'pt|{c}'], 1e-3)), O[f'n|{c}']], 1)[rows].astype(np.float64) for c in CONDS}
    fold = np.random.default_rng([seed, 22]).permutation(rows.sum()) % 2; outs = {c: np.zeros(rows.sum()) for c in CONDS}; aucs = []
    for f in (0, 1):
        tr = fold != f; te = fold == f; sc = StandardScaler().fit(X['base'][tr])
        m = MLPClassifier((32, 32), max_iter=600, early_stopping=True, random_state=seed + f).fit(sc.transform(X['base'][tr]), sig[tr])
        for c in CONDS:
            p = np.clip(m.predict_proba(sc.transform(X[c][te]))[:, 1], 1e-7, 1 - 1e-7); outs[c][te] = np.log(p) - np.log1p(-p)
    return outs


def summary(smoke, B=1000):
    s = sfx(smoke); RESU = dict(datasets={}, conds=CONDS, FR=list(FR)); fam = dict(perm=[], mcn=[])
    for ds in ('hls', 'jc', 'top', 'qg'):
        fo, fn, fe = OUT / f'h72_{ds}_obs{s}.npz', OUT / f'h72_{ds}_nets{s}.npz', OUT / f'h72_{ds}_efp{s}.npz'
        if not (fo.exists() and fn.exists()): log(f'{ds}: missing {fo.name} or {fn.name}'); continue
        O = dict(np.load(fo)); S = dict(np.load(fn)); y = O['y']; E = dict(np.load(fe)) if fe.exists() else {}
        assert (S['y'] == y).all()
        mism = int(sum((O[f'n|{c}'] != S[f'n|{c}']).sum() for c in CONDS)); log(f'{ds}: N mismatch obs vs nets {mism}')
        jinfo = json.loads((OUT / f'h72_{ds}_nets{s}.json').read_text())
        models = sorted({k.split('|')[1] for k in S if k.startswith('s|')})
        rc = rcuts(ds)
        D = dict(n=int(len(y)), r_cut=rc, unedited=jinfo['unedited'], known=jinfo.get('known'), n_mismatch=mism, classes={},
                 efp=json.loads((OUT / f'h72_{ds}_efp{s}.json').read_text()) if fe.exists() else None, edits={})
        for c in CONDS[1:]:
            D['edits'][c] = dict(valid=float(O[f'ok|{c}'].mean()), dN=float((O[f'n|{c}'] - O['n|base']).mean()),
                                 nout=float(O[f'nout|{c}'].mean()), fpt_out=float(O[f'fpt_out|{c}'].mean()))
        for cls, k in SIG[ds].items():
            rows = np.isin(y, [k] + QCD[ds]); sig = (y == k)[rows]; nr = int(rows.sum()); ns = int(sig.sum())
            sidx = np.flatnonzero(sig); bidx = np.flatnonzero(~sig)
            bs = np.random.default_rng([SEEDS[ds], 99, k]).integers(0, ns, (B, ns))
            okc = {c: O[f'ok|{c}'][rows] for c in CONDS}
            # jets left with no particle (all real particles beyond r_cut) or with a non-finite score from any network
            # are excluded from that condition for every tagger (counted in 'excluded')
            excl = {}
            for c in CONDS:
                fin = O[f'n|{c}'][rows] > 0
                for mn in models: fin &= np.isfinite(S[f's|{mn}|{cls}|{c}'][rows])
                excl[c] = dict(empty=int((O[f'n|{c}'][rows] == 0).sum()), excluded=int((~fin).sum()), excluded_sig=int((~fin & sig).sum()))
                okc[c] = okc[c] & fin
            T = {}
            for mn in models: T[mn] = ('net', {c: S[f's|{mn}|{cls}|{c}'][rows].astype(np.float64) for c in CONDS})
            if E: T['GBDT_EFP'] = ('ref', {c: E[f's|GBDT_EFP|{cls}|{c}'][rows].astype(np.float64) for c in CONDS})
            jit = np.random.default_rng([SEEDS[ds], 98]).random(nr) * 1e-6
            fN = H65.lr_n(O['n|base'][rows].astype(int), sig, SEEDS[ds]); T['LR_N'] = ('ref', {c: fN(O[f'n|{c}'][rows].astype(int)) + jit for c in CONDS})
            fG = H65.lr_n(O['girth|base'][rows].astype(np.float64), sig, SEEDS[ds] + 1)
            T['LR_girth'] = ('ref', {c: fG(O[f'girth|{c}'][rows].astype(np.float64)) + jit for c in CONDS})
            T['MLP_mptN'] = ('ref', mlp_ref(O, rows, sig.astype(int), SEEDS[ds]))
            for ob in ('m', 'msd'):
                X = {c: O[f'{ob}|{c}'][rows].astype(np.float64) for c in CONDS}
                if ds == 'qg':
                    t = float(np.median(X['base'][sig])); T[f'win_{ob}'] = ('win', {c: (X[c] < t).astype(float) for c in CONDS}, [None, t])
                else:
                    lo, hi = H65.shortest50(X['base'][sig]); T[f'win_{ob}'] = ('win', {c: ((X[c] >= lo) & (X[c] <= hi)).astype(float) for c in CONDS}, [lo, hi])
            C = dict(n_rows=nr, n_sig=ns, taggers={}, observables={}, excluded=excl)
            log(f'{ds} {cls}: excluded (empty or non-finite score) ' + ', '.join(f'{c} {e["excluded"]}' for c, e in excl.items() if e['excluded']))
            for c in CONDS[1:]:
                v = okc[c]
                C['observables'][c] = {f'd{ob}_sig': float((O[f'{ob}|{c}'][rows] - O[f'{ob}|base'][rows])[sig & v].mean()) for ob in ('m', 'msd', 'girth', 'pt', 'n')}
                C['observables'][c].update({f'd{ob}_bkg': float((O[f'{ob}|{c}'][rows] - O[f'{ob}|base'][rows])[~sig & v].mean()) for ob in ('m', 'msd', 'girth', 'n')})
            effB = {}
            for tn, tv in T.items():
                kind, sc = tv[0], tv[1]
                if kind == 'win':
                    calls = {c: sc[c] > 0.5 for c in CONDS}; wp = tv[2]; cont = False
                else:
                    thr = float(np.median(sc['base'][sig])); calls = {c: sc[c] > thr for c in CONDS}; wp = thr; cont = True
                R = dict(kind=kind, wp=wp, effS0=float(calls['base'][sig].mean()), effB0=float(calls['base'][~sig].mean()),
                         AUC0=(auc(sc['base'], sig.astype(int)) if cont else None), cond={}, pairs={}, r50={})
                bE = {}
                for c in CONDS[1:]:
                    v = okc[c][sidx].astype(float); ch = calls[c][sidx].astype(float) - calls['base'][sidx].astype(float)
                    w = v[bs]; bE[c] = (ch[bs] * w).sum(1) / np.maximum(w.sum(1), 1)
                    e = dict(d_effS=float((ch * v).sum() / max(v.sum(), 1)), d_effS_ci=ci(bE[c]),
                             d_effB=float((calls[c][bidx].astype(float) - calls['base'][bidx]).dot(okc[c][bidx]) / max(okc[c][bidx].sum(), 1)),
                             n_valid=int(v.sum()))
                    if cont:
                        d = np.nan_to_num((sc[c] - sc['base'])[sidx]); db = (d[bs] * w).sum(1) / np.maximum(w.sum(1), 1)
                        e.update(ds=float((d * v).sum() / max(v.sum(), 1)), ds_ci=ci(db),
                                 ds_bkg=float(np.nan_to_num((sc[c] - sc['base'])[bidx])[okc[c][bidx]].mean()))
                    R['cond'][c] = e
                for arm in ARMS:
                    ctl = PAIR[arm]
                    for i in range(len(FR)):
                        a_, c_ = f'{arm}{i}', f'{ctl}{i}'; v = (okc[a_] & okc[c_])[sidx]; vf = v.astype(float); wv = vf[bs]
                        e = dict(n=int(v.sum()))
                        dE = (calls[a_][sidx].astype(float) - calls[c_][sidx].astype(float))
                        dEb = (dE[bs] * wv).sum(1) / np.maximum(wv.sum(1), 1)
                        e.update(d_effS=float(dE[v].mean()) if v.any() else None, d_effS_ci=ci(dEb), mcnemar=mcnemar(calls[a_][sidx][v], calls[c_][sidx][v]))
                        if cont:
                            d = np.nan_to_num((sc[a_] - sc[c_])[sidx]); db = (d[bs] * wv).sum(1) / np.maximum(wv.sum(1), 1)
                            e.update(dds=float(d[v].mean()) if v.any() else None, dds_ci=ci(db), p_perm=signflip_p(d[v], seed=SEEDS[ds] + i))
                        if kind == 'net' and tn not in NULLS:
                            if cont: fam['perm'].append((f'{ds}|{cls}|{tn}|{a_}-{c_}', e['p_perm']))
                            fam['mcn'].append((f'{ds}|{cls}|{tn}|{a_}-{c_}', e['mcnemar']['p']))
                        R['pairs'][f'{a_}-{c_}'] = e
                    # reach on the raw arm and on arm - control
                    raw = [R['cond'][f'{arm}{i}']['d_effS'] for i in range(len(FR))]
                    rawb = np.stack([bE[f'{arm}{i}'] for i in range(len(FR))], 1)
                    rb = np.array([reach(r_) for r_ in rawb])
                    f50 = reach(np.array(raw))
                    R['r50'][arm] = dict(f=f50, r=f50 * RJ[ds] if np.isfinite(f50) else None, f_ci=ci(rb),
                                         r_ci=[x * RJ[ds] if x is not None else None for x in ci(rb)], d_effS_curve=raw,
                                         frac_at=[abs(x) / abs(raw[-1]) if abs(raw[-1]) > 0 else None for x in raw])
                    if cont:
                        rawd = [R['cond'][f'{arm}{i}']['ds'] for i in range(len(FR))]
                        f50d = reach(np.array(rawd)); R['r50'][arm].update(f_ds=f50d, ds_curve=rawd)
                effB[tn] = np.stack([bE[f'rm{i}'] for i in range(len(FR))], 1)
                C['taggers'][tn] = R
            refs = [t for t in T if T[t][0] != 'net']
            for tn in models:
                pt_ = {rf: float(np.sqrt(sum((C['taggers'][tn]['cond'][f'rm{i}']['d_effS'] - C['taggers'][rf]['cond'][f'rm{i}']['d_effS']) ** 2
                                             for i in range(len(FR))))) for rf in refs}
                Dm = np.stack([np.sqrt(((effB[tn] - effB[rf]) ** 2).sum(1)) for rf in refs], 1); am = Dm.argmin(1)
                C['taggers'][tn]['nearest_rm'] = dict(dist=pt_, nearest=min(pt_, key=pt_.get), boot_freq={rf: float((am == j).mean()) for j, rf in enumerate(refs)})
            D['classes'][cls] = C
            log(f'{ds} {cls}: ' + '; '.join(f'{tn} rm d_effS ' + '/'.join(f'{C["taggers"][tn]["cond"][f"rm{i}"]["d_effS"]:+.2f}' for i in range(len(FR)))
                                           + f' r50 {C["taggers"][tn]["r50"]["rm"]["f"]:.2f}' for tn in C['taggers']))
        RESU['datasets'][ds] = D
    for f, L in fam.items():
        if not L: continue
        ps = holm([p if p is not None else 1.0 for _, p in L])
        RESU[f'holm_{f}'] = {k: dict(p=p, p_holm=float(h)) for (k, p), h in zip(L, ps)}
        log(f'Holm family {f}: {len(L)} tests, {sum(h < 0.05 for h in ps)} with p_holm < 0.05')
    (OUT / f'h72_summary{s}.json').write_text(json.dumps(RESU, indent=1, default=float))
    log(f'wrote h72_summary{s}.json')
    try:
        figure(RESU, s)
    except Exception as e:  # noqa: BLE001
        import traceback; traceback.print_exc(); log('figure failed', repr(e))


def figure(RESU, s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ds = 'jc' if 'jc' in RESU['datasets'] else list(RESU['datasets'])[0]
    O = np.load(OUT / f'h72_{ds}_obs{s}.npz'); rc = rcuts(ds); R = RJ[ds]
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2)); th = np.linspace(0, 2 * np.pi, 200)
    cols = ['#9b2c2c', '#dd6b20', '#d69e2e', '#2f855a']
    for a, tag, lab in ((ax[0], 'bkg', 'QCD jet'), (ax[1], 'sig', 'top jet' if ds != 'qg' else 'quark jet')):
        pt, de, dp = O[f'ex{tag}|pt'], O[f'ex{tag}|de'], O[f'ex{tag}|dp']; dr = np.hypot(de, dp)
        a.plot(R * np.cos(th), R * np.sin(th), color='0.6', lw=1, ls='--')
        for r, c in zip(rc, cols): a.plot(r * np.cos(th), r * np.sin(th), color=c, lw=1)
        inner = dr <= rc[-1]
        a.scatter(de[inner], dp[inner], s=5 + 300 * pt[inner] / pt.max(), color='#2b6cb0', alpha=0.8, label=f'kept at every r_cut ({inner.sum()})')
        edges = list(rc[::-1]) + [10.0]
        for lo, hi, c in zip(edges[:-1], edges[1:], cols[::-1]):
            sel = (dr > lo) & (dr <= hi)
            a.scatter(de[sel], dp[sel], s=5 + 300 * pt[sel] / pt.max(), color=c, marker='X', alpha=0.85,
                      label=f'removed at r_cut <= {lo:g} ({sel.sum()}, {100 * pt[sel].sum() / pt.sum():.0f} % of pT)')
        a.set_xlim(-1.05 * R, 1.05 * R); a.set_ylim(-1.05 * R, 1.05 * R); a.set_aspect('equal')
        a.set_xlabel('Δη from E-scheme axis'); a.set_ylabel('Δφ from E-scheme axis')
        a.set_title(f'JetClass {lab}, {len(pt)} particles; circles r_cut = ' + ' / '.join(f'{r:g}' for r in rc), fontsize=9)
        a.legend(loc='lower left', fontsize=6.5, frameon=False)
    b = ax[2]; cls = 't' if 't' in RESU['datasets'][ds]['classes'] else list(RESU['datasets'][ds]['classes'])[0]
    C = RESU['datasets'][ds]['classes'][cls]['taggers']
    sty = [('ParT_full', '#2b6cb0', '-'), ('PN_full', '#4299e1', '-'), ('ParT_kin', '#2c5282', ':'), ('win_m', '#c53030', '--'),
           ('win_msd', '#2f855a', '--'), ('LR_girth', '#805ad5', '--'), ('LR_N', '#d69e2e', '--'), ('GBDT_EFP', '0.4', '--')]
    lab = dict(win_m='plain-mass window', win_msd='soft-drop-mass window', LR_girth='girth (LR)', LR_N='particle count N (LR)',
               GBDT_EFP='GBDT on EFPs')
    for tn, col, ls in sty:
        if tn not in C: continue
        yv = [0] + [100 * C[tn]['cond'][f'rm{i}']['d_effS'] for i in range(len(FR))]
        b.plot([R] + rc, yv, color=col, ls=ls, marker='o', ms=3, label=lab.get(tn, tn))
    if 'ParT_full' in C:
        yv = [0] + [100 * C['ParT_full']['cond'][f'cr{i}']['d_effS'] for i in range(len(FR))]
        b.plot([R] + rc, yv, color='#2b6cb0', ls='-.', lw=1, marker='s', ms=3, label='ParT_full, control: random inside removal')
    b.invert_xaxis(); b.set_xlabel('r_cut: particles beyond this ΔR removed'); b.set_ylabel('change in top-jet efficiency (percentage points)')
    b.set_title(f'JetClass top vs QCD, threshold fixed at 50 % top efficiency', fontsize=9); b.grid(alpha=0.3)
    b.legend(fontsize=7, frameon=False, loc='lower left')
    fig.tight_layout(); fig.savefig(OUT / f'h72_reach{s}.png', dpi=130); log(f'wrote h72_reach{s}.png')


if __name__ == '__main__':
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--part', required=True, choices=['run', 'efp', 'summary'])
    ap_.add_argument('--ds', default=None, choices=['hls', 'jc', 'top', 'qg'])
    ap_.add_argument('--what', default='both', choices=['obs', 'nets', 'both'])
    ap_.add_argument('--smoke', type=int, default=0)
    a = ap_.parse_args()
    if a.part == 'summary': summary(a.smoke)
    elif a.part == 'efp': efp(a.ds, a.smoke)
    else: run(a.ds, a.what, a.smoke)
