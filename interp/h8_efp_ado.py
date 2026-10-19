"""H8 (loop D; novelty test for D1/D4/D5): does an IRC-safe, axis-free surrogate built from energy-flow polynomials (EFPs), jet
mass, soft-drop mass, N-subjettiness ratios and C2/D2, chosen by average decision ordering (ADO) to match each network
(Faucett et al. 2021 guided search), reproduce the networks' calls on natural jets, and their call changes under the WTA-axis,
translation, boost and collinear-split edits?

Models and samples
  hls4ml (5 classes g q W Z t): JEDI-linear official n64 3-feature (interp.jedi_core.Jedi) and the same network with every
    weight redrawn (Jedi(untrained=True, seed=0), null). Jets: interp.jedi_core.sample(per_class=2000, seed=801) indices
    (h1_irc_panel.hls_sample), N = 10,000.
  JetClass (10 classes): ParT_full, ParT_kin and ParT_full with weights redrawn (ParTNetwork('full', untrained=True, seed=0),
    null). Jets: 1,000 random rows per class (rng 802) of each class's first test file (h1_irc_panel.jc_sample), rebuilt with
    eta = jet_eta + sign(jet_eta) deta (part_core.build). Accuracy on all 10,000; edits and surrogates on QCD, Wqq, Zqq, Tbqq.
Binary tasks: W vs QCD, top vs QCD, Z vs W. Network score = log sum_sig p - log sum_bkg p (hls4ml QCD = g + q; JetClass QCD =
  ZJetsToNuNu); binary call = score > 0.
Edits (applied to the same jets)
  wta        (D4) particles' (deta, dphi) recomputed about the winner-take-all axis (C/A, WTA recombination); 4-vectors unchanged.
  translate  (A4 / P1) dphi + 0.06 rad for every real particle (jet axis displaced by 0.06 rad); invariant mass unchanged.
  boost      (H3) pT x 0.9, (deta, dphi) / 0.9 (longitudinal boost along the jet axis; m, tau_N ratios ~ unchanged).
  split      (H1 split_r0_z0.05_dR0, H1's main panel) hardest particle -> two collinear daughters (5 % / 95 %) (exact; padding cut to 64 / 128 slots).
Observable pool (computed on the particles the network sees, 64 / 128 slots)
  EFPs, all multigraphs with d <= 5 edges (energyflow EFPSet('d<=5'), hadronic measure, z_i = pT_i / sum pT, theta = dR^beta),
  beta in {0.5, 1, 2}, constant d = 0 graph dropped; m (invariant mass of the real particles), m_SD (C/A, z_cut 0.1, beta 0),
  tau21, tau32 (exclusive-kT axes), C2, D2 (beta 1, from the EFPs: C2 = (2/3) T / E^2, D2 = (4/3) T / E^3 with E the one-edge
  and T the triangle EFP). Pool "irc": all of these (IRC safe, translation invariant). Pool "ext": irc + N, pT^D (IRC unsafe).
Guided search (per model x task x pool): jets split in halves (rng 803); fit half split 2:1 into A (search) and V (stopping)
  (rng 804). Step 1: the observable with the largest ADO (orientation-free, max(ADO, 1 - ADO)) with the network score on
  150,000 random (signal, background) pairs of A. Step k: surrogate = HistGradientBoosting regression of the network score on the
  chosen observables (2-fold out-of-fold predictions on A); among the A pairs where surrogate and network order the pair
  differently, add the observable with the largest ADO with the network; stop when the ADO on V pairs gains < 0.002 (the last
  observable is then dropped) or at 8 observables. Final surrogates (HGB and an MLP, 32-32, on log EFPs / standardised inputs)
  are fitted on the whole fit half and evaluated on the test half.
Metrics (test half): ADO over all (sig, bkg) pairs with a 95 % bootstrap interval (jets resampled per class, 50,000 pairs per
  replicate, B = 400); same-call %; per edit: % binary calls changed for network and surrogate, difference with a paired
  bootstrap interval, exact McNemar (network-only vs surrogate-only changes), mean signed and absolute change of the score, per-jet
  correlation of the two changes; BH over the trained models x tasks x edits (irc pool, HGB). Null: the weight-redrawn networks
  through the same search.
Run
  ./job.sh d-h8-jedi 1 2 'python -m interp.h8_efp_ado --part jedi'          (jax env)       -> physics/h8_jedi_net.{npz,json}
  ./part_job.sh d-h8-edit 'python -m interp.h8_efp_ado --part jetclass'     (GPU)           -> physics/h8_jc_net.{npz,json}
  ./job.sh d-h8-efp 4 8 '/j-jepa-vol/aaron-interp/venv-ef/bin/python -m interp.h8_efp_ado --part efp'
                                                                            -> physics/h8_obs_{jedi,jc}.npz, physics/h8_efp.json
  python -m interp.h8_efp_ado --part plot                                   -> physics/h8_fig.png
  --smoke N: N jets per class, files *_smoke.*
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
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results'))
OUT = RES / 'physics'
T0 = time.time()
CONDS = ('natural', 'wta', 'translate', 'boost', 'split')
EDITS = CONDS[1:]
SPLIT = dict(name='split_r0_z0.05_dR0', kind='split', r=0, z=0.05, dR=0.0)
SHIFT, BOOST = 0.06, 0.9
BETAS = (0.5, 1.0, 2.0)


def log(*a):
    print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


def sfx(smoke):
    return '_smoke' if smoke else ''


# ================================================================================================================ networks + edits
def part_jedi(smoke):
    from interp.jedi_core import Jedi
    from interp.h1_irc_panel import hls_sample, hls_x, apply
    from interp.e17_axis_jedi import wta_axis_one, reaxis, wta_selftest
    per = smoke or 2000
    T, y = hls_sample(per, seed=801); x0 = hls_x(T); real = x0[..., 0] > 0
    log(f'hls4ml: {len(y)} jets, mean multiplicity {real.sum(1).mean():.1f}, at 64 slots {(real.sum(1) >= 64).mean():.3f}')
    st = wta_selftest(); log(f'WTA self-test {st}')
    t = time.time(); pt, de, dp = (x0[..., i].astype(np.float64) for i in range(3))
    wta = np.array([wta_axis_one(pt[j, real[j]], de[j, real[j]], dp[j, real[j]]) for j in range(len(y))])
    log(f'WTA axes ({time.time() - t:.0f} s): median displacement {np.median(np.hypot(wta[:, 0], wta[:, 1])):.4f}')
    X = dict(natural=x0, wta=reaxis(x0, wta[:, 0], wta[:, 1]))
    xt = x0.copy(); xt[..., 2] = np.where(real, xt[..., 2] + SHIFT, xt[..., 2]); X['translate'] = xt
    xb = x0.copy(); xb[..., 0] *= BOOST; xb[..., 1] /= BOOST; xb[..., 2] /= BOOST; X['boost'] = xb
    U, applied, info = apply(T, SPLIT, 64, 0.8, 801); X['split'] = hls_x(U)
    over = info['overflow']; log(f'split: applied {applied.mean():.4f}, overflow {over.mean():.4f}, dp3 max {np.nanmax(info["dp3_rel"]):.1e}')
    nets = dict(jedi=Jedi(64, '3-feature'), jedi_redrawn=Jedi(64, '3-feature', untrained=True, seed=0))
    NPZ = dict(y=y.astype(np.int8), split_overflow=over, wta_axis=wta.astype(np.float32), classes=np.array(['g', 'q', 'W', 'Z', 't']))
    meta = dict(sample=f'interp.jedi_core.sample(per_class={per}, seed=801) (h1_irc_panel.hls_sample)', N=int(len(y)), wta_selftest=st,
                split_overflow=float(over.mean()), models={})
    for c in CONDS:
        NPZ[f'x_{c}'] = X[c].astype(np.float32)
    for mn, net in nets.items():
        acc = None
        for c in CONDS:
            L = net.run(X[c])['logits']; NPZ[f'L_{mn}_{c}'] = L.astype(np.float32)
            if c == 'natural':
                L0 = L; acc = float((L.argmax(1) == y).mean()); meta['models'][mn] = dict(acc=acc, changed5={})
            else:
                meta['models'][mn]['changed5'][c] = float((L.argmax(1) != L0.argmax(1)).mean())
        log(f'{mn}: accuracy {acc:.4f}; 5-class calls changed ' + ' '.join(f'{k} {v:.4f}' for k, v in meta['models'][mn]['changed5'].items()))
    meta['known_acc'] = dict(jedi='0.806 (E17, seed 41, 20,000 jets); 0.811 (H3, seed 501)', jedi_wta_changed='0.303 (D4)')
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f'h8_jedi_net{sfx(smoke)}.npz', **NPZ)
    (OUT / f'h8_jedi_net{sfx(smoke)}.json').write_text(json.dumps(meta, indent=1)); log('wrote h8_jedi_net')


def part_jetclass(smoke):
    from interp.h1_irc_panel import jc_sample, jc_table, jc_jets, apply
    from interp.part_core import ParTNetwork, CL, build, kin, jmass, sub, inputs
    from interp.p1v_verify_part import fwd, wta_axes, reaxis, wta_selftest
    per = smoke or 1000
    J0 = jc_sample(per, seed=802); y = J0['label'].astype(int)
    Jb = build(J0, *kin(J0))
    keep = [CL.index(c) for c in ('QCD', 'Wqq', 'Zqq', 'Tbqq')]; rows = np.flatnonzero(np.isin(y, keep))
    J4, Jb4 = sub(J0, rows), sub(Jb, rows); y4 = y[rows]
    st = wta_selftest(); log(f'WTA self-test {st}')
    t = time.time(); wd, wp = wta_axes(Jb4)
    sgn = np.where(Jb4['jet_eta'] < 0, -1.0, 1.0); je, jp = Jb4['jet_eta'].astype(np.float64), Jb4['jet_phi'].astype(np.float64)
    Jw, nflip = reaxis(Jb4, je + sgn * wd, jp + wp)
    log(f'WTA axes ({time.time() - t:.0f} s): median displacement {np.median(np.hypot(wd, wp)):.4f}; eta-sign flips {nflip}')
    pt, de, dp, ms = kin(J4); real = pt > 0
    JJ = dict(natural=Jb4, wta=Jw, translate=build(J4, pt, de, dp + SHIFT * real, ms), boost=build(J4, pt * BOOST, de / BOOST, dp / BOOST, ms))
    T = jc_table(J4); U, applied, info = apply(T, SPLIT, 128, 0.8, 802); JJ['split'] = jc_jets(J4, U)
    over = info['overflow']; log(f'split: applied {applied.mean():.4f}, overflow {over.mean():.4f}')
    nets = {'ParT_full': (ParTNetwork('full'), 'full'), 'ParT_kin': (ParTNetwork('kin'), 'kin'),
            'ParT_full_redrawn': (ParTNetwork('full', untrained=True, seed=0), 'full')}
    NPZ = dict(y=y4.astype(np.int8), split_overflow=over, classes=np.array(list(CL)))
    meta = dict(sample=f'random rows (rng 802) of each class first test file, {per} per class (h1_irc_panel.jc_sample)', N_all=int(len(y)),
                N_4class=int(len(rows)), wta_selftest=st, split_overflow=float(over.mean()), wta_eta_sign_flips=nflip, models={})
    for c in CONDS:
        p_, d_, f_, _ = kin(JJ[c])
        NPZ[f'pt_{c}'] = p_.astype(np.float32); NPZ[f'de_{c}'] = d_.astype(np.float32); NPZ[f'dp_{c}'] = f_.astype(np.float32)
        NPZ[f'm_{c}'] = jmass(JJ[c]).astype(np.float32)
    for mn, (net, n) in nets.items():
        La = fwd(net, *inputs(Jb, n))[1]; ca = La.argmax(1)
        e = dict(acc_all10=float((ca == y).mean()), acc_4class_rows=float((ca[rows] == y[rows]).mean()),
                 per_class={CL[k]: float((ca[y == k] == k).mean()) for k in range(len(CL))}, changed10={})
        for c in CONDS:
            L = La[rows] if c == 'natural' else fwd(net, *inputs(JJ[c], n))[1]; NPZ[f'L_{mn}_{c}'] = L.astype(np.float32)
            if c != 'natural': e['changed10'][c] = float((L.argmax(1) != La[rows].argmax(1)).mean())
        meta['models'][mn] = e
        log(f'{mn}: accuracy (10 classes, {len(y)} jets) {e["acc_all10"]:.4f}; on the 4-class rows {e["acc_4class_rows"]:.4f}; calls changed '
            + ' '.join(f'{k} {v:.4f}' for k, v in e['changed10'].items()))
    meta['known_acc'] = dict(ParT_full='0.859 (D4) / 0.8638 (H3)', ParT_kin='0.745 (D4) / 0.7455 (H3)', wta_changed='ParT_full 0.117, ParT_kin 0.341 (D4)')
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f'h8_jc_net{sfx(smoke)}.npz', **NPZ)
    (OUT / f'h8_jc_net{sfx(smoke)}.json').write_text(json.dumps(meta, indent=1)); log('wrote h8_jc_net')


# ================================================================================================================ observables
_EFP = {}


def _efpsets():
    import energyflow as ef
    if not _EFP:
        for b in BETAS:
            _EFP[b] = ef.EFPSet('d<=5', measure='hadr', beta=b, normed=True, coords='ptyphim', check_input=False)
    return _EFP


def efp_info():
    S = _efpsets()[1.0]; G = S.graphs(); spec = np.asarray(S.specs)
    # specs columns (energyflow): n vertices, e distinct edges, d edges with multiplicity, v max valency, k id, c VE complexity,
    # p prime factors, h valency-1 vertices
    return [dict(graph=[list(map(int, e)) for e in g], n=int(spec[i][0]), e=int(spec[i][1]), d=int(spec[i][2]), k=int(spec[i][4]),
                 c=int(spec[i][5]), p=int(spec[i][6])) for i, g in enumerate(G)]


def obs_chunk(args):
    """pt, de, dp (n, W) float64, m (n,) or None -> dict of observables (n, ...)"""
    pt, de, dp, m = args
    from interp.e19_common import kt3
    from jetdistill.observables.compute import _softdrop
    pt = np.asarray(pt, np.float64); de = np.asarray(de, np.float64); dp = np.asarray(dp, np.float64)
    real = pt > 0; n = len(pt); O = {}
    ev = [np.stack([pt[j, real[j]], de[j, real[j]], dp[j, real[j]]], 1) for j in range(n)]
    for b, S in _efpsets().items():
        O[f'efp_b{b:g}'] = np.array([S.compute(e) for e in ev])
    S = pt.sum(1); z = pt / np.maximum(S, 1e-300)[:, None]
    O['N'] = real.sum(1).astype(float); O['ptD'] = np.sqrt((pt ** 2).sum(1)) / np.maximum(S, 1e-300)
    if m is None:
        p4 = np.stack([(pt * np.cos(dp)).sum(1), (pt * np.sin(dp)).sum(1), (pt * np.sinh(de)).sum(1), (pt * np.cosh(de)).sum(1)], 1)
        m = np.sqrt(np.maximum(p4[:, 3] ** 2 - (p4[:, :3] ** 2).sum(1), 0))
    O['m'] = np.asarray(m, np.float64)
    tau = {k: np.zeros(n) for k in (1, 2, 3)}
    for j in range(n):
        r = real[j]; e, f, w = de[j, r], dp[j, r], z[j, r]
        if r.sum() == 0: continue
        for k in (1, 2, 3):
            ax = kt3(pt[j, r], e, f, k)[1]; ax = ax[ax[:, 0] > 0]
            dd = np.hypot(e[:, None] - ax[None, :, 1], (f[:, None] - ax[None, :, 2] + np.pi) % (2 * np.pi) - np.pi).min(1)
            tau[k][j] = (w * dd).sum()
    O['tau21'] = tau[2] / np.maximum(tau[1], 1e-12); O['tau32'] = tau[3] / np.maximum(tau[2], 1e-12)
    msd = np.zeros(n); order = np.argsort(real.sum(1), kind='stable')
    for a in range(0, n, 50):
        idx = order[a:a + 50]; w = max(int(real[idx].sum(1).max()), 2)
        msd[idx] = _softdrop(pt[idx, :w], de[idx, :w], dp[idx, :w])[0]
    O['mSD'] = msd
    return O


def observables(pt, de, dp, m, pool, chunk=100):
    n = len(pt); jobs = [(pt[a:a + chunk], de[a:a + chunk], dp[a:a + chunk], None if m is None else m[a:a + chunk]) for a in range(0, n, chunk)]
    R = list(pool.imap(obs_chunk, jobs)) if pool is not None else [obs_chunk(j) for j in jobs]
    return {k: np.concatenate([r[k] for r in R]) for k in R[0]}


def feature_table(O, info):
    """-> names, F (n, K) for the irc pool, plus N, ptD (ext)"""
    edge = next(i for i, g in enumerate(info) if sorted(map(tuple, g['graph'])) == [(0, 1)])
    tri = next(i for i, g in enumerate(info) if sorted(tuple(sorted(e)) for e in g['graph']) == [(0, 1), (0, 2), (1, 2)])
    keep = [i for i, g in enumerate(info) if g['d'] > 0]
    names, cols = [], []
    for b in BETAS:
        for i in keep: names.append(f'efp_b{b:g}_{i}'); cols.append(O[f'efp_b{b:g}'][:, i])
    E, T = O['efp_b1'][:, edge], O['efp_b1'][:, tri]
    extra = dict(m=O['m'], mSD=O['mSD'], tau21=O['tau21'], tau32=O['tau32'], C2=(2 / 3) * T / np.maximum(E, 1e-300) ** 2,
                 D2=(4 / 3) * T / np.maximum(E, 1e-300) ** 3, N=O['N'], ptD=O['ptD'])
    for k, v in extra.items(): names.append(k); cols.append(v)
    return names, np.stack(cols, 1), dict(edge=edge, tri=tri)


# ================================================================================================================ ADO search
def logsumexp(L, idx):
    A = L[:, idx]; M = A.max(1, keepdims=True); return (M + np.log(np.exp(A - M).sum(1, keepdims=True)))[:, 0]


def score(L, sig, bkg):
    L = np.asarray(L, np.float64); return logsumexp(L, sig) - logsumexp(L, bkg)


def pairs(sig_rows, bkg_rows, n, rng):
    return rng.choice(sig_rows, n), rng.choice(bkg_rows, n)


def ado_on(f, g, si, bi):
    return float((1 + np.sign(f[si] - f[bi]) * np.sign(g[si] - g[bi])).mean() / 2)


def ado_all(f, g, sig_rows, bkg_rows, chunk=500):
    tot, cnt = 0.0, 0
    for a in range(0, len(sig_rows), chunk):
        s = sig_rows[a:a + chunk]
        p = np.sign(f[s][:, None] - f[bkg_rows][None]) * np.sign(g[s][:, None] - g[bkg_rows][None])
        tot += (1 + p).sum() / 2; cnt += p.size
    return float(tot / cnt)


def transform(F, names):
    G = F.astype(np.float64).copy()
    for j, nm in enumerate(names):
        if nm.startswith('efp') or nm in ('m', 'mSD', 'C2', 'D2'): G[:, j] = np.log(np.maximum(G[:, j], 1e-12))
    return G


def hgb():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                                         early_stopping=True, validation_fraction=0.15, n_iter_no_change=20, random_state=0)


def mlp():
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 32), alpha=1e-3, learning_rate_init=3e-3, max_iter=400, early_stopping=True,
                                                         validation_fraction=0.15, n_iter_no_change=20, random_state=0))


def oof(X, t, rows, k=2, seed=0):
    p = np.zeros(len(rows)); f = np.random.default_rng(seed).integers(0, k, len(rows))
    for i in range(k):
        m = hgb().fit(X[rows[f != i]], t[rows[f != i]]); p[f == i] = m.predict(X[rows[f == i]])
    out = np.full(len(X), np.nan); out[rows] = p; return out


def guided_search(F, names, s, A_sig, A_bkg, V_sig, V_bkg, rng, max_obs=8, tol=0.002, npairs=150000):
    si, bi = pairs(A_sig, A_bkg, npairs, rng); vsi, vbi = pairs(V_sig, V_bkg, 100000, rng)
    A = np.concatenate([A_sig, A_bkg]); dn = np.sign(s[si] - s[bi])
    D = np.sign(F[si] - F[bi]).astype(np.int8)                     # (P, K)
    chosen, steps, prev = [], [], None
    sel = np.ones(len(si), bool)
    while len(chosen) < max_obs:
        if sel.sum() == 0: break
        a = (1 + (D[sel] * dn[sel, None]).mean(0)) / 2; a = np.maximum(a, 1 - a); a[chosen] = -1
        j = int(np.argmax(a)); trial = chosen + [j]
        m = hgb().fit(F[A][:, trial], s[A]); pv = m.predict(F[:, trial])        # fitted on all of A, evaluated on V
        adov = ado_on(s, pv, vsi, vbi)
        st = dict(added=names[j], ado_on_disagreeing=float(a[j]), n_disagreeing=int(sel.sum()), ado_V=adov)
        if prev is not None and adov - prev < tol:
            st['kept'] = False; steps.append(st); break
        st['kept'] = True; steps.append(st); chosen, prev = trial, adov
        po = oof(F[:, chosen], s, A, seed=len(chosen))
        sel = (np.sign(s[si] - s[bi]) * np.sign(po[si] - po[bi])) <= 0
        log(f'    step {len(chosen)}: + {names[j]} (ADO on {st["n_disagreeing"]} disagreeing pairs {a[j]:.4f}); ADO(V) {adov:.4f}')
    return chosen, steps


# ================================================================================================================ statistics
def binom_two_sided(k, n):
    if n == 0: return 1.0
    k = min(k, n - k)
    lp = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(k + 1)]
    mx = max(lp); return float(min(1.0, 2 * math.exp(mx) * sum(math.exp(v - mx) for v in lp)))


def bh(ps):
    ps = np.asarray(ps, float); m = len(ps); o = np.argsort(ps); q = np.empty(m); run = 1.0
    for r in range(m - 1, -1, -1):
        i = o[r]; run = min(run, ps[i] * m / (r + 1)); q[i] = run
    return q.tolist()


def ci_mean(x, rng, B=1000):
    x = np.asarray(x, np.float64); n = len(x)
    if n == 0: return [None, None]
    idx = rng.integers(0, n, (B, n)); m = x[idx].mean(1); return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def ado_boot(f, g, sig, bkg, rng, B=400, n=50000):
    out = np.empty(B)
    for b in range(B):
        s_, k_ = rng.choice(sig, len(sig)), rng.choice(bkg, len(bkg)); si, bi = rng.choice(s_, n), rng.choice(k_, n)
        out[b] = ado_on(f, g, si, bi)
    return [float(np.quantile(out, .025)), float(np.quantile(out, .975))]


def edit_stats(sn0, sn1, ss0, ss1, rows, rng, B=1000):
    """network score before/after (sn0, sn1), surrogate (ss0, ss1), on rows"""
    cn = (sn0[rows] > 0) != (sn1[rows] > 0); cs = (ss0[rows] > 0) != (ss1[rows] > 0); n = len(rows)
    idx = rng.integers(0, n, (B, n)); d = cn[idx].mean(1) - cs[idx].mean(1)
    dn, ds = sn1[rows] - sn0[rows], ss1[rows] - ss0[rows]
    n10, n01 = int((cn & ~cs).sum()), int((~cn & cs).sum())
    r = float(np.corrcoef(dn, ds)[0, 1]) if ds.std() > 1e-12 and dn.std() > 1e-12 else None
    return dict(n=n, net_changed=float(cn.mean()), net_ci=ci_mean(cn, rng), sur_changed=float(cs.mean()), sur_ci=ci_mean(cs, rng),
                diff=float(cn.mean() - cs.mean()), diff_ci=[float(np.quantile(d, .025)), float(np.quantile(d, .975))],
                n_net_only=n10, n_sur_only=n01, p_mcnemar=binom_two_sided(n10, n10 + n01),
                net_dscore=float(dn.mean()), net_dscore_ci=ci_mean(dn, rng), sur_dscore=float(ds.mean()), sur_dscore_ci=ci_mean(ds, rng),
                net_absd=float(np.abs(dn).mean()), sur_absd=float(np.abs(ds).mean()), corr_delta=r)


# ================================================================================================================ main analysis
TASKS = dict(jedi={'W_vs_QCD': (['W'], ['g', 'q']), 'top_vs_QCD': (['t'], ['g', 'q']), 'Z_vs_W': (['Z'], ['W'])},
             jc={'W_vs_QCD': (['Wqq'], ['QCD']), 'top_vs_QCD': (['Tbqq'], ['QCD']), 'Z_vs_W': (['Zqq'], ['Wqq'])})
MODELS = dict(jedi=('jedi', 'jedi_redrawn'), jc=('ParT_full', 'ParT_kin', 'ParT_full_redrawn'))


def load_obs(ds, smoke, pool):
    p = OUT / f'h8_obs_{ds}{sfx(smoke)}.npz'
    net = dict(np.load(OUT / f'h8_{"jedi" if ds == "jedi" else "jc"}_net.npz', allow_pickle=False))
    if smoke:                                    # smoke: a random subset of smoke jets per class of the full sample
        y = net['y']; N = len(y); r = np.random.default_rng(0)
        idx = np.sort(np.concatenate([r.choice(np.flatnonzero(y == c), min(smoke, (y == c).sum()), replace=False) for c in np.unique(y)]))
        net = {k: (v[idx] if v.ndim and len(v) == N else v) for k, v in net.items()}
    if p.exists():
        d = np.load(p); return net, {c: {k[len(c) + 1:]: d[k] for k in d.files if k.startswith(c + '|')} for c in CONDS}
    OBS = {}
    for c in CONDS:
        t = time.time()
        if ds == 'jedi':
            x = net[f'x_{c}']; OBS[c] = observables(x[..., 0], x[..., 1], x[..., 2], None, pool)
        else:
            OBS[c] = observables(net[f'pt_{c}'], net[f'de_{c}'], net[f'dp_{c}'], net[f'm_{c}'], pool)
        log(f'{ds} {c}: observables for {len(OBS[c]["m"])} jets ({time.time() - t:.0f} s)')
    np.savez_compressed(p, **{f'{c}|{k}': v for c, O in OBS.items() for k, v in O.items()})
    return net, OBS


def part_efp(smoke):
    from multiprocessing import Pool
    nproc = int(os.environ.get('OMP_NUM_THREADS', '1'))
    os.environ['OMP_NUM_THREADS'] = '1'
    info = efp_info(); log(f'EFPs d<=5: {len(info)} graphs (incl. d = 0); betas {BETAS}')
    res = dict(efps=info, settings=dict(betas=BETAS, shift=SHIFT, boost=BOOST, split=SPLIT, tol=0.002, max_obs=8, seeds=dict(half=803, AV=804, pairs=805)),
               datasets={})
    tests = []
    with Pool(nproc) if nproc > 1 else _NoPool() as pool:
        for ds in ('jedi', 'jc'):
            net, OBS = load_obs(ds, smoke, pool if nproc > 1 else None)
            y = net['y'].astype(int); cls = [str(c) for c in net['classes']]; N = len(y)
            names, F0, ix = feature_table(OBS['natural'], info)
            FE = {c: feature_table(OBS[c], info)[1] for c in CONDS}
            inv = {c: float(np.nanmax(np.abs(FE[c][:, :-2] - F0[:, :-2]) / np.maximum(np.abs(F0[:, :-2]), 1e-12))) for c in EDITS}
            log(f'{ds}: max relative change of the irc observables per edit ' + ' '.join(f'{k} {v:.2e}' for k, v in inv.items()))
            rng = np.random.default_rng(803); half = rng.permutation(N); fit, test = np.sort(half[:N // 2]), np.sort(half[N // 2:])
            rngAV = np.random.default_rng(804); fp = rngAV.permutation(fit); A_, V_ = np.sort(fp[:2 * len(fp) // 3]), np.sort(fp[2 * len(fp) // 3:])
            over = net['split_overflow'].astype(bool)
            D = dict(N=N, classes=cls, n_fit=len(fit), n_test=len(test), irc_obs_max_rel_change=inv, split_overflow=float(over.mean()),
                     names=names, edge_index=ix['edge'], triangle_index=ix['tri'], models={})
            for mn in MODELS[ds]:
                D['models'][mn] = {}
                for tk, (sg, bk) in TASKS[ds].items():
                    si_, bi_ = [cls.index(c) for c in sg], [cls.index(c) for c in bk]
                    S = {c: score(net[f'L_{mn}_{c}'], si_, bi_) for c in CONDS}; s = S['natural']
                    issig, isbkg = np.isin(y, si_), np.isin(y, bi_); task = issig | isbkg
                    E = dict(n_task=int(task.sum()), net_pos_frac=float((s[task] > 0).mean()), pools={})
                    for pn, cols in (('irc', [j for j, nm in enumerate(names) if nm not in ('N', 'ptD')]), ('ext', list(range(len(names))))):
                        Fp = F0[:, cols]; nmp = [names[j] for j in cols]
                        log(f'{ds} {mn} {tk} pool {pn}: search on {int((task[A_]).sum())} jets')
                        chosen, steps = guided_search(Fp, nmp, s, A_[issig[A_]], A_[isbkg[A_]], V_[issig[V_]], V_[isbkg[V_]], np.random.default_rng(805))
                        fr = fit[task[fit]]; te = test[task[test]]; tsig, tbkg = test[issig[test]], test[isbkg[test]]
                        P = dict(chosen=[nmp[j] for j in chosen], steps=steps, surrogates={})
                        single = max(ado_all(s, Fp[:, chosen[0]], tsig, tbkg), 1 - ado_all(s, Fp[:, chosen[0]], tsig, tbkg))
                        P['ado_first_obs_test'] = single
                        for sn, mk, tf in (('hgb', hgb, lambda X: X), ('mlp', mlp, lambda X, n_=[nmp[j] for j in chosen]: transform(X, n_))):
                            mdl = mk().fit(tf(Fp[fr][:, chosen]), s[fr])
                            SS = {c: mdl.predict(tf(FE[c][:, cols][:, chosen])) for c in CONDS}; ss = SS['natural']
                            rb = np.random.default_rng(806)
                            Sx = dict(ado_test=ado_all(s, ss, tsig, tbkg), ado_ci=ado_boot(s, ss, tsig, tbkg, rb),
                                      same_call=float(((s[te] > 0) == (ss[te] > 0)).mean()), same_call_ci=ci_mean((s[te] > 0) == (ss[te] > 0), rb),
                                      r2_score=float(1 - ((s[te] - ss[te]) ** 2).sum() / ((s[te] - s[te].mean()) ** 2).sum()), edits={})
                            Sx['auc_net'] = float(np.mean([(s[tsig][:, None] > s[tbkg][None]).mean()]))
                            Sx['auc_sur'] = float(np.mean([(ss[tsig][:, None] > ss[tbkg][None]).mean()]))
                            for c in EDITS:
                                Sx['edits'][c] = edit_stats(s, S[c], ss, SS[c], te, rb)
                                if c == 'split': Sx['edits']['split_no_overflow'] = edit_stats(s, S[c], ss, SS[c], te[~over[te]], rb)
                            P['surrogates'][sn] = Sx
                            log(f'  {sn}: chosen {P["chosen"]}; ADO test {Sx["ado_test"]:.4f} {Sx["ado_ci"]}; same-call {Sx["same_call"]:.4f}; '
                                + ' '.join(f'{c} net {Sx["edits"][c]["net_changed"]:.3f} sur {Sx["edits"][c]["sur_changed"]:.3f}' for c in EDITS))
                            if sn == 'hgb' and pn == 'irc' and 'redrawn' not in mn:
                                for c in EDITS: tests.append((ds, mn, tk, c, Sx['edits'][c]['p_mcnemar']))
                        E['pools'][pn] = P
                    D['models'][mn][tk] = E
            res['datasets'][ds] = D
    q = bh([t[4] for t in tests])
    res['bh'] = dict(n_tests=len(tests), family='trained models x tasks x edits, irc pool, HGB surrogate')
    for (ds, mn, tk, c, p), qq in zip(tests, q):
        res['datasets'][ds]['models'][mn][tk]['pools']['irc']['surrogates']['hgb']['edits'][c]['bh_q'] = qq
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f'h8_efp{sfx(smoke)}.json'; p.write_text(json.dumps(res, indent=1, default=float)); log(f'wrote {p}')


class _NoPool:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def part_diag(smoke):
    """invariance check of the cached observables: per observable family and edit, the fraction of jets whose value changes by
    more than 1e-4 / 1e-2 (relative), the max relative change, and the jets' multiplicity (split: overflow vs not)"""
    info = efp_info(); out = {}
    for ds in ('jedi', 'jc'):
        p = OUT / f'h8_obs_{ds}{sfx(smoke)}.npz'
        if not p.exists(): continue
        d = np.load(p); OBS = {c: {k[len(c) + 1:]: d[k] for k in d.files if k.startswith(c + '|')} for c in CONDS}
        net = np.load(OUT / f'h8_{"jedi" if ds == "jedi" else "jc"}_net.npz'); over = net['split_overflow'].astype(bool)
        names, F0, _ = feature_table(OBS['natural'], info); fam = np.array([nm.split('_')[0] if nm.startswith('efp') else nm for nm in names])
        Nn = OBS['natural']['N']; out[ds] = {}
        for c in EDITS:
            F1 = feature_table(OBS[c], info)[1]; rel = np.abs(F1 - F0) / np.maximum(np.abs(F0), 1e-12); e = {}
            for f in dict.fromkeys(fam):
                r = rel[:, fam == f].max(1)
                e[f] = {'frac_gt_1e-4': float((r > 1e-4).mean()), 'frac_gt_1e-2': float((r > 1e-2).mean()), 'max': float(r.max()),
                        'N_median_changed': float(np.median(Nn[r > 1e-2])) if (r > 1e-2).any() else None}
                if c == 'split': e[f]['frac_gt_1e-4_no_overflow'] = float((r[~over] > 1e-4).mean())
            out[ds][c] = e
            log(f'{ds} {c}: ' + ' '.join(f'{f} {v["frac_gt_1e-4"]:.4f}/{v["frac_gt_1e-2"]:.4f}' for f, v in e.items()))
    (OUT / f'h8_diag{sfx(smoke)}.json').write_text(json.dumps(out, indent=1)); log('wrote h8_diag')


# ================================================================================================================ figure
def part_plot(smoke):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    R = json.loads((OUT / f'h8_efp{sfx(smoke)}.json').read_text())
    rows = [('jedi', 'jedi', 'JEDI-linear'), ('jc', 'ParT_full', 'ParT full'), ('jc', 'ParT_kin', 'ParT kin')]
    tasks = [('W_vs_QCD', 'W vs QCD'), ('top_vs_QCD', 'top vs QCD'), ('Z_vs_W', 'Z vs W')]
    ed = [('wta', 'WTA axis'), ('translate', 'translate 0.06 rad'), ('boost', 'boost pT x 0.9'), ('split', 'collinear split')]
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.6), sharey=False)
    labels = [f'{r[2]}\n{t[1]}' for r in rows for t in tasks]; xs = np.arange(len(labels)); w = 0.38
    for a, (c, cl) in zip(ax, ed):
        vn, vs, en, es = [], [], [], []
        for ds, mn, _ in rows:
            for tk, _ in tasks:
                e = R['datasets'][ds]['models'][mn][tk]['pools']['irc']['surrogates']['hgb']['edits'][c]
                vn.append(100 * e['net_changed']); vs.append(100 * e['sur_changed'])
                en.append([100 * (e['net_changed'] - e['net_ci'][0]), 100 * (e['net_ci'][1] - e['net_changed'])])
                es.append([100 * (e['sur_changed'] - e['sur_ci'][0]), 100 * (e['sur_ci'][1] - e['sur_changed'])])
        a.bar(xs - w / 2, vn, w, color='#3b6fb6', label='network', yerr=np.array(en).T, ecolor='k', error_kw=dict(lw=0.6))
        a.bar(xs + w / 2, vs, w, color='#e8a33d', label='EFP + shape surrogate', yerr=np.array(es).T, ecolor='k', error_kw=dict(lw=0.6))
        a.set_xticks(xs); a.set_xticklabels(labels, rotation=60, ha='right', fontsize=7); a.set_title(cl, fontsize=10)
        a.set_ylabel('binary calls changed (% of test jets)')
    ax[0].legend(fontsize=8, frameon=False)
    ado = [R['datasets'][ds]['models'][mn][tk]['pools']['irc']['surrogates']['hgb']['ado_test'] for ds, mn, _ in rows for tk, _ in tasks]
    fig.suptitle(f'Binary calls changed by each edit: network vs its ADO-chosen IRC-safe surrogate (EFPs, m, mSD, tau21, tau32, C2, D2; '
                 f'HGB; test half; ADO on natural jets {min(ado):.2f}-{max(ado):.2f}; bars 95 % bootstrap CI)', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(OUT / f'h8_fig{sfx(smoke)}.png', dpi=130); log(f'wrote h8_fig')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--part', required=True, choices=['jedi', 'jetclass', 'efp', 'diag', 'plot'])
    ap.add_argument('--smoke', type=int, default=0); a = ap.parse_args()
    log(f'H8 part {a.part} smoke {a.smoke}')
    dict(jedi=part_jedi, jetclass=part_jetclass, efp=part_efp, diag=part_diag, plot=part_plot)[a.part](a.smoke)
    log('done')
