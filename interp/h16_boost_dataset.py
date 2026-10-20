"""H16 (loop D, claim D9, open question a): boost battery on the small binary top taggers of interp/h16_train.py (identical
jedis-d64 / tf-2b-d32 / PFN trained on hls4ml t vs q/g over the full pT spectrum (hlsA), on hls4ml jets inside +-8 % of the
median pT_J (hlsB), and on Top Landscape t vs QCD (top); inputs pT_i / sum pT ('frac') or absolute pT ('abs'); 2 seeds).
Question: is the hls4ml-trained nets' response to a mass-preserving boost along the jet axis (H12: 21-38 % of 5-class calls at
pT x 0.9) caused by the training dataset (H1) or by the small architectures / their inputs (H0)?
Edits (interp.h12_boost_panel; lam = factor on the jet pT; inputs re-measured about the edited stored axis):
  B   exact Lorentz boost along the jet momentum, |P| -> lam |P|; invariant m and m_A = sqrt(M0 M2) fixed, angles ~ 1/lam
  P   pT-only: four-vectors x lam (m -> lam m); a no-op for 'frac' inputs
  A   angle-only: (deta, dphi) x 1/lam at fixed pT (m, m_A -> ~ m / lam)
  R   rotation about the axis, pT-weighted mean displacement matched per jet to B's (control)
  M   m held, m_A moved: translation of all particles by t u (u random direction, axis kept) with t solved per jet so that
      sum pT |r + t u|^2 = k^2 M2, i.e. m_A -> k m_A, k in {1/0.95, 1/0.9} (the m_A change of A at lam 0.95 / 0.9); invariant m and
      jet pT exactly unchanged (all pair differences kept). Control RM: rotation with pT-weighted mean displacement t.
Samples (seed 4161, random rows, never first-k): hls_all = hls4ml val archive, 5,000 t + 5,000 q/g (random rows of q union g),
  all 150 stored particles, massless at (eta, phi) = (deta, dphi), axis (0, 0); hls_win = the same restricted to the hlsB pT_J
  window (5,000 + 5,000); top = Top Landscape test split, 5,000 top + 5,000 QCD, all constituents (four-vectors), axis = E-scheme
  sum of the constituents, deta sign = sign of the unedited axis eta (the conversion's convention). Networks see the 64 hardest.
Models: hlsA and hlsB nets on hls_all and hls_win, top nets on top; each config's native sample: hlsA -> hls_all, hlsB -> hls_win,
  top -> top. In-window subset = jets whose edited pT_J lies in the model's training window (hlsA: 0.5-99.5 % of the training
  pool; hlsB: +-8 % of the median; top: 550-650 GeV).
Metrics: % of binary calls (argmax) changed with a 95 % bootstrap over jets (and on the in-window subset); mean Delta log-odds
  (logit_t - logit_bkg) with 95 % bootstrap; normalised response rms(Delta s) / std(s_0); signal efficiency and 1/eps_B at the
  unedited eps_S = 0.5 threshold; exact McNemar B vs R, B vs P, B vs A, M vs RM per model x lam (Holm across all models, samples
  and lam per family in the summary); seed spread; dataset contrasts per (var, arch) of the seed-mean B % changed (hlsA - top,
  hlsB - top, hlsA - hlsB) with normal-approximation p-values from the binomial standard errors, Holm across contrasts.
Run (CPU, torch):
  AI_PY=conda ./job.sh d-h16-run 1 2 'python -m interp.h16_boost_dataset --part all'
  -> $AI_RESULTS/physics/h16_<sample>.json, h16_summary.json, h16_boost_dataset.png; --smoke N: N jets per class
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
from interp.h12_boost_panel import (boost, pscale, ascale, rotate, jet_obs, copyT, rel, set_rel, sortT, boot_frac, boot_mean,  # noqa: E402
                                    mcnemar, holm, table_from_full)
from interp import h16_train as HT  # noqa: E402

RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results'))
OUT = RES / 'physics'
V = Path('/j-jepa-vol/aaron-interp')
SEED = 4161
LAMS = (0.9, 0.95, 1.05, 1.1)
MKS = (round(1 / 0.95, 6), round(1 / 0.9, 6))
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


# ================================================================================================================ edits
def mA_shift(T, k, rng):
    """translation (axis kept) that sends m_A -> k m_A at fixed invariant m and jet pT; returns (table, t)"""
    de, dp = rel(T); w = T['pt']; M0 = w.sum(1); M2 = (w * (de ** 2 + dp ** 2)).sum(1)
    S = np.stack([(w * de).sum(1), (w * dp).sum(1)], 1); a = rng.uniform(0, 2 * np.pi, len(M0)); u = np.stack([np.cos(a), np.sin(a)], 1)
    us = (u * S).sum(1); t = (-us + np.sqrt(us ** 2 + M0 * (k ** 2 - 1) * M2)) / np.maximum(M0, 1e-12)
    U = copyT(T); real = T['pt'] > 0
    set_rel(U, np.where(real, de + (t * u[:, 0])[:, None], 0), np.where(real, dp + (t * u[:, 1])[:, None], 0))
    return U, t


def edits(T, rng):
    for lam in LAMS:
        B, _ = boost(T, lam); d0, p0 = rel(T); d1, p1 = rel(B); w = T['pt']
        D = (w * np.hypot(d1 - d0, p1 - p0)).sum(1) / np.maximum(w.sum(1), 1e-12)
        yield f'B{lam}', 'B', lam, B
        yield f'P{lam}', 'P', lam, pscale(T, lam)[0]
        yield f'A{lam}', 'A', lam, ascale(T, lam)[0]
        yield f'R{lam}', 'R', lam, rotate(T, D, rng)[0]
    for k in MKS:
        U, t = mA_shift(T, k, rng)
        yield f'M{k:.4f}', 'M', k, U
        yield f'RM{k:.4f}', 'RM', k, rotate(T, t, rng)[0]


def net_input(T, sgn):
    S = sortT(T); de, dp = rel(S); real = S['pt'] > 0
    return np.stack([np.where(real, S['pt'], 0), de * sgn[:, None], dp], -1)[:, :HT.NP].astype(np.float32)


# ================================================================================================================ samples
def hls_sample(per, seed, window=None):
    import h5py
    X = np.load(HT.DATA / 'val_ptetaphi_f16.npy', mmap_mode='r')
    with h5py.File(HT.DATA / 'val_meta.h5', 'r') as f: y = f['label'][:].astype(np.int64)
    ok = np.ones(len(y), bool)
    if window is not None:
        jp = np.concatenate([HT.jpt_of(np.asarray(X[i:i + 50000], np.float32)) for i in range(0, len(X), 50000)])
        ok = (jp >= window[0]) & (jp <= window[1])
    rng = np.random.default_rng(seed)
    idx = np.sort(np.r_[rng.choice(np.flatnonzero(ok & (y == 4)), per, replace=False), rng.choice(np.flatnonzero(ok & np.isin(y, [0, 1])), per, replace=False)])
    x = np.asarray(X[idx], np.float64); real = x[..., 0] > 0; J = len(idx)
    T = dict(pt=np.where(real, x[..., 0], 0), eta=np.where(real, x[..., 1], 0), phi=np.where(real, x[..., 2], 0), m=np.zeros(x.shape[:2]),
             ax_eta=np.zeros(J), ax_phi=np.zeros(J), ext={})
    raw64 = np.asarray(X[idx, :HT.NP], np.float32)
    return T, (y[idx] == 4).astype(int), np.ones(J), idx, raw64


def top_sample(per, seed):
    from interp import data_top as dt
    d = dt.DST / 'test'; yl = np.load(d / 'label.npy'); rng = np.random.default_rng(seed)
    idx = np.sort(np.r_[rng.choice(np.flatnonzero(yl == 1), per, replace=False), rng.choice(np.flatnonzero(yl == 0), per, replace=False)])
    F = np.load(d / 'full.npy', mmap_mode='r')[idx]; jet = np.load(d / 'jet.npy', mmap_mode='r')[idx]
    nn = np.minimum(jet[:, 5].astype(np.int64), F.shape[1]); T = table_from_full(np.asarray(F), nn)
    sgn = np.where(np.asarray(jet[:, 1]) < 0, -1.0, 1.0)
    raw64 = np.asarray(np.load(d / 'kin3.npy', mmap_mode='r')[idx, :HT.NP], np.float32)
    return T, yl[idx].astype(int), sgn, idx, raw64


# ================================================================================================================ battery
def runs_for(dss):
    out = {}
    for ds in dss:
        for var in ('frac', 'abs'):
            for arch in ('jedis', 'tf', 'pfn'):
                for s in (0, 1):
                    d = HT.MOD / f'{ds}-{var}-{arch}' / f'seed{s}'
                    if (d / 'test.json').exists(): out[f'{ds}-{var}-{arch}-s{s}'] = d
    return out


def battery(name, T0, y, sgn, raw64, runs, windows, smoke):
    rng = np.random.default_rng(SEED + 1); erng = np.random.default_rng(SEED + 2); J = len(y)
    X0 = net_input(T0, sgn); O0 = jet_obs(T0)
    rt = float(np.abs(X0 - raw64).max()); rt_rel = float(np.abs(X0[..., 0] - raw64[..., 0]).max() / max(raw64[..., 0].max(), 1e-9))
    log(f'{name}: {J} jets; round trip (table -> inputs) vs stored 64-slot inputs: max abs diff {rt:.2e} (pT relative {rt_rel:.2e})')
    # slot-order-free check (pT ties in the float16 store can swap slots): pT-weighted centroid and spread of (deta, dphi)
    cen = lambda x: np.stack([(x[..., 0] * x[..., 1]).sum(1), (x[..., 0] * x[..., 2]).sum(1), (x[..., 0] * (x[..., 1] ** 2 + x[..., 2] ** 2)).sum(1)], 1) / np.maximum(x[..., 0].sum(1, keepdims=True), 1e-9)
    dc = np.abs(cen(X0.astype(np.float64)) - cen(raw64.astype(np.float64)))
    rt_inv = dict(centroid_deta=np.percentile(dc[:, 0], [50, 99, 100]).tolist(), centroid_dphi=np.percentile(dc[:, 1], [50, 99, 100]).tolist(),
                  r2=np.percentile(dc[:, 2], [50, 99, 100]).tolist(), frac_jets_slot_maxdiff_gt_0p01=float((np.abs(X0 - raw64).max((1, 2)) > 1e-2).mean()))
    log(f'{name}: order-free round trip (median / 99 % / max): centroid deta {rt_inv["centroid_deta"]}, dphi {rt_inv["centroid_dphi"]}, '
        f'<r^2> {rt_inv["r2"]}; jets with a slot diff > 0.01: {rt_inv["frac_jets_slot_maxdiff_gt_0p01"]:.4f}')
    res = dict(sample=name, n=J, seed=SEED, roundtrip_maxdiff=rt, roundtrip_orderfree=rt_inv, models={}, edits={}, tests={},
               refs=dict(jpt=np.percentile(O0['jpt'], [0.5, 50, 99.5]).tolist(), m=np.percentile(O0['m'], [25, 50, 75]).tolist()))
    nets, S0, thr = {}, {}, {}
    from sklearn.metrics import roc_auc_score
    for rn, d in runs.items():
        net = HT.Net(d); L = net(X0); s = L[:, 1] - L[:, 0]; nets[rn] = net; S0[rn] = s; c = (s > 0).astype(int)
        thr[rn] = float(np.quantile(s[y == 1], 0.5)); known = json.loads((d / 'test.json').read_text())
        fb = float((s[y == 0] > thr[rn]).mean())
        res['models'][rn] = dict(acc=float((c == y).mean()), acc_ci=boot_frac(c == y, rng), auc=float(roc_auc_score(y, s)),
                                 known_test_acc=known['accuracy'], known_test_auc=known['auc'], thr_eps0p5=thr[rn],
                                 bkg_eff_eps0p5=fb, rej_eps0p5=1 / fb if fb > 0 else None, window=windows[rn.split('-')[0]],
                                 n_params=net.A['n_params'], best_epoch=net.A.get('best_epoch'))
        log(f'{name} {rn}: accuracy {res["models"][rn]["acc"]:.4f} (test.json {known["accuracy"]:.4f}) AUC {res["models"][rn]["auc"]:.4f} '
            f'1/eps_B(0.5) {res["models"][rn]["rej_eps0p5"]}')
    CH = {}
    for en, kind, lam, U in edits(T0, erng):
        t = time.time(); O1 = jet_obs(U); X1 = net_input(U, sgn)
        e = dict(kind=kind, lam=lam, rel_change={k: dict(mean=float(np.mean(O1[k] / np.maximum(O0[k], 1e-12) - 1)),
                                                         p99_abs=float(np.percentile(np.abs(O1[k] / np.maximum(O0[k], 1e-12) - 1), 99)))
                                                 for k in ('m', 'jpt', 'mA', 'mdr')}, models={})
        for rn, net in nets.items():
            L1 = net(X1); s1 = L1[:, 1] - L1[:, 0]
            s0 = S0[rn]; ch = (s0 > 0) != (s1 > 0); w = windows[rn.split('-')[0]]
            iw = (O1['jpt'] >= w[0]) & (O1['jpt'] <= w[1]) & (O0['jpt'] >= w[0]) & (O0['jpt'] <= w[1])
            ds = s1 - s0; mu, ci = boot_mean(ds, rng); fb = float((s1[y == 0] > thr[rn]).mean())
            e['models'][rn] = dict(changed=float(ch.mean()), ci=boot_frac(ch, rng),
                                   in_window=dict(n=int(iw.sum()), changed=float(ch[iw].mean()) if iw.any() else None, ci=boot_frac(ch[iw], rng)),
                                   acc_after=float(((s1 > 0).astype(int) == y).mean()), dlogodds=mu, dlogodds_ci=ci,
                                   dlogodds_sig=float(ds[y == 1].mean()), dlogodds_bkg=float(ds[y == 0].mean()),
                                   norm_resp=float(np.sqrt((ds ** 2).mean()) / max(s0.std(), 1e-12)),
                                   sig_eff=float((s1[y == 1] > thr[rn]).mean()), bkg_eff=fb, rej=1 / fb if fb > 0 else None)
            CH[(rn, en)] = ch
        res['edits'][en] = e
        log(f'{name} {en}: dm {e["rel_change"]["m"]["mean"]:+.2e} djpt {e["rel_change"]["jpt"]["mean"]:+.4f} dmA {e["rel_change"]["mA"]["mean"]:+.4f} | '
            + ' '.join(f'{rn} {e["models"][rn]["changed"]:.4f}' for rn in nets) + f' ({time.time() - t:.0f} s)')
    for rn in nets:
        for lam in LAMS:
            for a, b in (('B', 'R'), ('B', 'P'), ('B', 'A')):
                t_ = mcnemar(CH[(rn, f'{a}{lam}')], CH[(rn, f'{b}{lam}')]); t_['diff'] = float(CH[(rn, f'{a}{lam}')].mean() - CH[(rn, f'{b}{lam}')].mean())
                res['tests'][f'{rn}|{a}{lam}_vs_{b}{lam}'] = t_
        for k in MKS:
            t_ = mcnemar(CH[(rn, f'M{k:.4f}')], CH[(rn, f'RM{k:.4f}')]); t_['diff'] = float(CH[(rn, f'M{k:.4f}')].mean() - CH[(rn, f'RM{k:.4f}')].mean())
            res['tests'][f'{rn}|M{k:.4f}_vs_RM{k:.4f}'] = t_
    OUT.mkdir(parents=True, exist_ok=True); p = OUT / f'h16_{name}{"_smoke" if smoke else ""}.json'
    p.write_text(json.dumps(res, indent=1, default=float)); log(f'wrote {p}')
    return res


def part_samples(smoke):
    per = smoke or 5000
    if smoke: HT.MOD = HT.MOD / '_smoke'; HT.H16D = HT.H16D / '_smoke'
    W = json.loads((HT.H16D / 'windows.json').read_text())
    windows = dict(hlsA=W['hlsA_window'], hlsB=W['hlsB_window'], top=[550.0, 650.0])
    hls_runs = runs_for(('hlsA', 'hlsB')); top_runs = runs_for(('top',))
    log(f'runs: hls {len(hls_runs)}, top {len(top_runs)}; windows {windows}')
    T, y, sgn, idx, raw = hls_sample(per, SEED); battery('hls_all', T, y, sgn, raw, hls_runs, windows, smoke)
    T, y, sgn, idx, raw = hls_sample(per, SEED, windows['hlsB']); battery('hls_win', T, y, sgn, raw, hls_runs, windows, smoke)
    T, y, sgn, idx, raw = top_sample(per, SEED); battery('top', T, y, sgn, raw, top_runs, windows, smoke)


# ================================================================================================================ summary
NATIVE = dict(hlsA='hls_all', hlsB='hls_win', top='top')


def part_summary(smoke):
    from scipy.stats import norm
    sfx = '_smoke' if smoke else ''
    R = {s: json.loads((OUT / f'h16_{s}{sfx}.json').read_text()) for s in ('hls_all', 'hls_win', 'top')}
    S = dict(native=NATIVE, configs={}, holm={}, contrasts={})
    # Holm per family across samples x models x lam
    import re
    fam_of = lambda k: re.sub(r'[0-9.]', '', k.split('|')[1])
    for fam in ('B_vs_R', 'B_vs_P', 'B_vs_A', 'M_vs_RM'):
        keys = [(s, k, t['p']) for s, r in R.items() for k, t in r['tests'].items() if fam_of(k) == fam]
        hp = holm([x[2] for x in keys])
        for (s, k, p), h in zip(keys, hp): S['holm'][f'{s}|{k}'] = dict(p=p, holm=h, family=fam, n_family=len(keys))
    edits_all = list(R['top']['edits'])
    cfgs = sorted({rn.rsplit('-s', 1)[0] for r in R.values() for rn in r['models']})
    for c in cfgs:
        ds = c.split('-')[0]; out = {}
        for samp in ('hls_all', 'hls_win', 'top'):
            r = R[samp]; seeds = [rn for rn in r['models'] if rn.rsplit('-s', 1)[0] == c]
            if not seeds: continue
            row = dict(acc={rn: r['models'][rn]['acc'] for rn in seeds}, known_acc={rn: r['models'][rn]['known_test_acc'] for rn in seeds},
                       auc={rn: r['models'][rn]['auc'] for rn in seeds}, rej={rn: r['models'][rn]['rej_eps0p5'] for rn in seeds}, edits={})
            for en in edits_all:
                v = [r['edits'][en]['models'][rn] for rn in seeds]; ch = [x['changed'] for x in v]
                row['edits'][en] = dict(changed=ch, mean=float(np.mean(ch)), spread=float(np.max(ch) - np.min(ch)), ci=[x['ci'] for x in v],
                                        in_window=[x['in_window']['changed'] for x in v], in_window_n=[x['in_window']['n'] for x in v],
                                        norm_resp=[x['norm_resp'] for x in v], dlogodds=[x['dlogodds'] for x in v],
                                        sig_eff=[x['sig_eff'] for x in v], rej=[x['rej'] for x in v])
            for lam in LAMS:
                b = row['edits'][f'B{lam}']['mean']; p = row['edits'][f'P{lam}']['mean']; a = row['edits'][f'A{lam}']['mean']; rr = row['edits'][f'R{lam}']['mean']
                row[f'ratios_{lam}'] = dict(B_over_P=b / p if p > 0 else None, B_over_A=b / a if a > 0 else None, B_over_R=b / rr if rr > 0 else None)
            out[samp] = row
        S['configs'][c] = dict(native=NATIVE[ds], samples=out)
    # dataset contrasts on native samples, seed-mean B % changed, normal approximation with binomial SEs (seeds averaged)
    tests = []
    for var in ('frac', 'abs'):
        for arch in ('jedis', 'tf', 'pfn'):
            for lam in (0.9, 1.1, 0.95, 1.05):
                for a, b in (('hlsA', 'top'), ('hlsB', 'top'), ('hlsA', 'hlsB')):
                    ca, cb = f'{a}-{var}-{arch}', f'{b}-{var}-{arch}'
                    if ca not in S['configs'] or cb not in S['configs']: continue
                    ea = S['configs'][ca]['samples'][NATIVE[a]]['edits'][f'B{lam}']; eb = S['configs'][cb]['samples'][NATIVE[b]]['edits'][f'B{lam}']
                    na, nb = R[NATIVE[a]]['n'], R[NATIVE[b]]['n']
                    se2 = lambda ch, n: sum(x * (1 - x) / n for x in ch) / len(ch) ** 2
                    d = ea['mean'] - eb['mean']; se = math.sqrt(se2(ea['changed'], na) + se2(eb['changed'], nb))
                    pv = float(2 * norm.sf(abs(d) / max(se, 1e-12)))
                    k = f'{var}-{arch}|B{lam}|{a}-{b}'
                    S['contrasts'][k] = dict(diff=d, ci=[d - 1.96 * se, d + 1.96 * se], p=pv, a=ea['mean'], b=eb['mean'],
                                             seed_spread=[ea['spread'], eb['spread']]); tests.append((k, pv))
    hp = holm([t[1] for t in tests])
    for (k, _), h in zip(tests, hp): S['contrasts'][k]['holm'] = h
    p = OUT / f'h16_summary{sfx}.json'; p.write_text(json.dumps(S, indent=1, default=float)); log(f'wrote {p}')
    for c, v in S['configs'].items():
        n = v['native']; e = v['samples'][n]['edits']
        log(f'{c} on {n}: acc {list(v["samples"][n]["acc"].values())} B0.9 {e["B0.9"]["changed"]} P0.9 {e["P0.9"]["changed"]} A0.9 {e["A0.9"]["changed"]} '
            f'R0.9 {e["R0.9"]["changed"]} M(1/0.9) {e[f"M{MKS[1]:.4f}"]["changed"]} RM {e[f"RM{MKS[1]:.4f}"]["changed"]}')
    try: figure(S, sfx)
    except Exception:
        import traceback; traceback.print_exc()


def figure(S, sfx):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    cols = dict(hlsA='#d1495b', hlsB='#edae49', top='#00798c')
    lab = dict(hlsA='hls4ml, full pT spectrum', hlsB='hls4ml, pT within ±8 % of median', top='Top Landscape (550–650 GeV)')
    for ax, var in zip(axes, ('frac', 'abs')):
        x = 0
        for arch, an in (('jedis', 'JEDI-style d64'), ('tf', 'transformer 2b-d32'), ('pfn', 'PFN')):
            for i, ds in enumerate(('hlsA', 'hlsB', 'top')):
                c = f'{ds}-{var}-{arch}'
                if c not in S['configs']: continue
                e = S['configs'][c]['samples'][NATIVE[ds]]['edits']
                ax.bar(x + i * 0.27, 100 * e['B0.9']['mean'], 0.25, color=cols[ds], label=lab[ds] if arch == 'jedis' else None)
                for v in e['B0.9']['changed']: ax.plot(x + i * 0.27, 100 * v, 'k.', ms=4)
                ax.bar(x + i * 0.27, 100 * e['R0.9']['mean'], 0.25, color='0.3', alpha=0.8, label='rotation control' if (arch == 'jedis' and i == 0) else None)
            ax.text(x + 0.27, -0.06, an, transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=9)
            x += 1
        ax.set_xticks([]); ax.set_title('inputs $p_{T,i}/\\sum p_T$, $\\Delta\\eta$, $\\Delta\\phi$' if var == 'frac' else 'inputs $p_{T,i}$ [GeV], $\\Delta\\eta$, $\\Delta\\phi$', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    axes[0].set_ylabel('% of top-vs-QCD calls changed\nby a mass-preserving boost, jet $p_T\\times0.9$')
    axes[0].legend(fontsize=8, loc='upper left')
    fig.suptitle('Same small networks, different training data: boost response (dots = 2 seeds; grey = matched rotation)', fontsize=11)
    fig.tight_layout(); p = OUT / f'h16_boost_dataset{sfx}.png'; fig.savefig(p, dpi=130); log(f'wrote {p}')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--part', default='all', choices=['samples', 'summary', 'all'])
    ap.add_argument('--smoke', type=int, default=0); a = ap.parse_args()
    if a.part in ('samples', 'all'): part_samples(a.smoke)
    if a.part in ('summary', 'all'): part_summary(a.smoke)


if __name__ == '__main__':
    main()
