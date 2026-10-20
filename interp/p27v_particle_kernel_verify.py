"""P27 verify: independent re-run of the per-particle kernel on fresh rows, plus two controls the P27 run did not have.

Sample: JetClass test_20M WToQQ_107 and ZToQQ_107 (P27 used _106), PER random rows each (numpy default_rng(2777), never first-k).
Models: ParT_full, ParT_kin, PN_full, PN_kin (released weights). Build: part_core.build (eta = jet_eta + sign(jet_eta) deta),
pT kept, so the pT ordering does not change. Eligible particles: pT share z > 0.5 %.
Edits per eligible particle: rad / radin (radial move out / in by 0.01 rad), az (rotation about the axis at fixed r, chord 0.01,
random sign), rad3 / radin3 (the same radial moves with 0.003 rad).
Read-out d = change of logit_Z - logit_W versus the rebuilt unedited jet.
Route split (new control 1): every edit is run three times,
  full   edited per-particle features and edited pair inputs (ParT: four-vectors -> pair features; PN: kNN coordinates);
  part   edited per-particle features, unedited pair inputs / kNN coordinates;
  pair   unedited per-particle features, edited pair inputs / kNN coordinates.
Step-size test (new control 2): for a smooth response d(delta) = a delta + b delta^2 the even part (d_out + d_in)/2 scales as
delta^2 and the odd part as delta, so going from 0.01 to 0.003 multiplies the even/odd variance ratio by 0.09.
pT-vs-M2 test (new control 3): d regressed on 10 z-quantile dummies plus x = dM2/M2; M2 reading predicts the within-z slope
beta_w equal to the M2-only slope beta_M2 (ratio 1), a pT-only reading predicts beta_w = 0.
  ./sync.sh && ./part_job.sh c-p27-verify 'python -m interp.p27v_particle_kernel_verify --smoke && python -m interp.p27v_particle_kernel_verify'
  -> $AI_RESULTS/part/p27_verify[_smoke].json, p27_verify[_smoke].npz, p27_verify[_smoke].png
"""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); ap.add_argument('--analyse', action='store_true')
ARGS = ap.parse_args(); SMOKE = ARGS.smoke
SEED = 2777; PER = 30 if SMOKE else 1000; NB = 100 if SMOKE else 1000
DR, DR3, ZMIN, BS, JCH = 0.01, 0.003, 0.005, 256, 32
TEST = Path('/j-jepa-vol/JetClass/Pythia/test_20M')
FILES = dict(Wqq='WToQQ_107.root', Zqq='ZToQQ_107.root')
OUTD = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'; TAG = '_smoke' if SMOKE else ''
FN, NPZ, PNG = OUTD / f'p27_verify{TAG}.json', OUTD / f'p27_verify{TAG}.npz', OUTD / f'p27_verify{TAG}.png'
ETYPES = ['base', 'rad', 'radin', 'az', 'rad3', 'radin3']
COMBOS = ['full', 'part', 'pair']
MODELS = ['ParT_full', 'ParT_kin', 'PN_full', 'PN_kin']
T0 = time.time()


def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def boot_W(n, nb, rng):
    W = np.zeros((nb, n)); [W.__setitem__(b, np.bincount(rng.integers(0, n, n), minlength=n)) for b in range(nb)]; return W


def ci(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]; return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def st(v, bs, v0=None):
    from scipy.stats import norm
    o = dict(v=float(v), ci=ci(bs))
    if v0 is not None:
        sd = float(np.nanstd(bs)); o['H0'] = v0
        o['p_norm'] = float(2 * norm.sf(abs(v - v0) / sd)) if sd > 0 else None
        b = np.asarray(bs, float); o['p_boot'] = float(max(1 / len(b), min(1, 2 * min((b <= v0).mean(), (b >= v0).mean()))))
    return o


def load_rows(path, n, rng, chunk=25000):
    import uproot
    from interp.part_core import read_root, sub, cat
    N = uproot.open(path)['tree'].num_entries
    if SMOKE: N = min(N, 25000)
    rows = np.sort(rng.choice(N, n, replace=False)); parts = []
    for s in range(0, N, chunk):
        r = rows[(rows >= s) & (rows < s + chunk)]
        if len(r): parts.append(sub(read_root(path, s, min(s + chunk, N)), r - s))
    return cat(parts), rows


def radial(de, dp, d):
    r = np.hypot(de, dp); ok = r > 1e-6
    return de + d * np.where(ok, de / np.where(ok, r, 1), 1.0), dp + d * np.where(ok, dp / np.where(ok, r, 1), 0.0)


def azim(de, dp, s):
    r = np.hypot(de, dp); th = s * 2 * np.arcsin(np.minimum(1.0, DR / np.maximum(2 * r, 1e-12)))
    return np.cos(th) * de - np.sin(th) * dp, np.sin(th) * de + np.cos(th) * dp


def plan(pt, de, dp, rng):
    z = pt / pt.sum(1, keepdims=True); meta, DE, DP = [], [], []
    for j in range(len(pt)):
        el = np.where(z[j] > ZMIN)[0]; sg = rng.choice([-1.0, 1.0], len(el))
        def add(t, i=-1, s=0.0, fn=None):
            e, p = de[j].copy(), dp[j].copy()
            if fn is not None: e[i], p[i] = fn(e[i], p[i])
            meta.append((j, ETYPES.index(t), i, s)); DE.append(e); DP.append(p)
        add('base')
        for k, i in enumerate(el):
            add('rad', i, fn=lambda a, b: radial(a, b, DR)); add('radin', i, fn=lambda a, b: radial(a, b, -DR))
            add('az', i, sg[k], fn=lambda a, b, s=sg[k]: azim(a, b, s))
            add('rad3', i, fn=lambda a, b: radial(a, b, DR3)); add('radin3', i, fn=lambda a, b: radial(a, b, -DR3))
    return np.array(meta), np.stack(DE), np.stack(DP)


def load_models():
    import torch
    from interp.part_core import ParTNetwork
    from interp import pn_net
    from jetdistill.part.network import FEATURES
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; M = {}
    for f in ('full', 'kin'):
        M[f'ParT_{f}'] = dict(kind='part', feat=f, model=ParTNetwork(f).model.to(dev).eval())
    PND = Path(os.environ.get('JETDISTILL_MODELS', '/j-jepa-vol/aaron-interp/models/part')).parent / 'particlenet'
    for f in ('full', 'kin'):
        M[f'PN_{f}'] = dict(kind='pn', feat=f, model=pn_net.load_particlenet(PND / f'ParticleNet_{f}.pt', len(FEATURES[f]), dev).eval())
    for mm in M.values():
        mm['dev'] = dev
        for p in mm['model'].parameters(): p.requires_grad_(False)
    return M, torch


def run(mm, Xf, Xp, torch):
    """Xf: inputs giving the per-particle features; Xp: inputs giving the pair four-vectors (ParT) or kNN coordinates (PN)"""
    xf, _, m = Xf[mm['feat']]; xp, vp, _ = Xp[mm['feat']]; dev = mm['dev']
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
    with torch.no_grad():
        if mm['kind'] == 'part': L = mm['model'](T(xf), T(vp), T(m))
        else:
            from jetdistill.part.network import FEATURES
            P = xf.shape[-1]; pad = lambda a: np.concatenate([a, np.zeros(a.shape[:2] + (128 - P,), np.float32)], -1) if P < 128 else a
            names = [f[0] for f in FEATURES[mm['feat']]]; pts = pad(xp)[:, [names.index('deta'), names.index('dphi')]]
            L = mm['model'](T(pts), T(pad(xf)), T(pad(m)))
    return L.float().cpu().numpy()


def collect():
    import torch
    from interp.part_core import kin, build, sub, cat, inputs, jmass
    from jetdistill.config import CLASSES as CL
    OUTD.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(SEED); Js, rowinfo = [], {}
    for c, fn in FILES.items():
        J, rows = load_rows(TEST / fn, PER, rng); Js.append(J); rowinfo[c] = dict(file=fn, n=len(rows), first_rows=rows[:5].tolist())
        log(f'loaded {fn}: {len(rows)} random rows')
    J = cat(Js); pt, de, dp, ms = kin(J); z = pt / np.maximum(pt.sum(1, keepdims=True), 1e-9)
    keep = np.where((z > ZMIN).sum(1) >= 2)[0]; J = sub(J, keep); pt, de, dp, ms = kin(J); N = len(keep); y = J['label'].astype(int)
    iW, iZ = CL.index('Wqq'), CL.index('Zqq'); M, torch = load_models(); log(f'N = {N}; models {list(M)}')
    J0 = build(J, pt, de, dp, ms)  # rebuilt unedited jets
    XI = lambda Jb: {f: inputs(Jb, f) for f in ('full', 'kin')}
    acc = {}
    for k, mm in M.items():
        L10 = np.concatenate([run(mm, X, X, torch) for X in (XI(sub(J0, np.arange(b, min(b + BS, N)))) for b in range(0, N, BS))])
        acc[k] = dict(all=float((L10.argmax(1) == y).mean()), **{c: float((L10.argmax(1)[y == CL.index(c)] == CL.index(c)).mean()) for c in FILES})
    log(f'accuracy (10-class argmax) on this W+Z sample: {acc}')
    erng = np.random.default_rng(SEED + 1); META, LOG = [], {(k, c): [] for k in M for c in COMBOS}
    for c0 in range(0, N, JCH):
        jj = np.arange(c0, min(c0 + JCH, N)); meta, DE, DP = plan(pt[jj], de[jj], dp[jj], erng); jl = meta[:, 0].astype(int)
        Jr = sub(J, jj[jl]); J0r = sub(J0, jj[jl])
        for b0 in range(0, len(jl), BS):
            sl = np.arange(len(jl))[b0:b0 + BS]; Jb = sub(Jr, sl); p_, _, _, m_ = kin(Jb)
            Xe = XI(build(Jb, p_, DE[sl], DP[sl], m_)); Xb = XI(sub(J0r, sl))
            for k, mm in M.items():
                LOG[(k, 'full')].append(run(mm, Xe, Xe, torch)[:, [iW, iZ]])
                LOG[(k, 'part')].append(run(mm, Xe, Xb, torch)[:, [iW, iZ]])
                LOG[(k, 'pair')].append(run(mm, Xb, Xe, torch)[:, [iW, iZ]])
        meta[:, 0] = jj[jl]; META.append(meta)
        if (c0 // JCH) % 10 == 0: log(f'  jets {c0 + len(jj)} / {N}')
    META = np.concatenate(META)
    np.savez_compressed(NPZ, meta=META, y=y, pt=pt, de=de, dp=dp, mjet=jmass(J),
                        **{f'L_{k}_{c}': np.concatenate(v).astype(np.float32) for (k, c), v in LOG.items()})
    (OUTD / f'p27_verify{TAG}_info.json').write_text(json.dumps(dict(sample=dict(files=rowinfo, seed=SEED, N=N), acc_sample=acc,
                                                                    forwards_per_model_per_combo=int(len(META))), indent=1))
    log(f'wrote {NPZ}: {len(META)} edits')


def lsq(X, y):
    G = X.T @ X; return np.linalg.lstsq(G + 1e-10 * np.trace(G) / len(G) * np.eye(len(G)), X.T @ y, rcond=None)[0]


def analyse():
    from jetdistill.config import CLASSES as CL
    Z = np.load(NPZ); meta = Z['meta']; y = Z['y']; pt, de, dp = Z['pt'], Z['de'], Z['dp']; N = len(y)
    jet, et, pi, sg = meta[:, 0].astype(int), meta[:, 1].astype(int), meta[:, 2].astype(int), meta[:, 3]
    info = json.loads((OUTD / f'p27_verify{TAG}_info.json').read_text())
    Wb = boot_W(N, NB, np.random.default_rng(SEED + 5)); folds = np.random.default_rng(SEED + 6).permutation(N) % 5
    ib = np.where(et == 0)[0]; brow = np.full(N, -1); brow[jet[ib]] = ib
    idx = {t: np.where(et == ETYPES.index(t))[0] for t in ETYPES[1:]}
    for t in ETYPES[1:]: assert np.array_equal(jet[idx[t]], jet[idx['rad']]) and np.array_equal(pi[idx[t]], pi[idx['rad']])
    jr, ir = jet[idx['rad']], pi[idx['rad']]; M0 = pt.sum(1); M2 = (pt * (de ** 2 + dp ** 2)).sum(1)
    p_ = pt[jr, ir]; r = np.hypot(de[jr, ir], dp[jr, ir]); z = p_ / M0[jr]; x = p_ * ((r + DR) ** 2 - r ** 2) / M2[jr]
    xodd = (x - p_ * ((r - DR) ** 2 - r ** 2) / M2[jr]) / 2
    cj = np.bincount(jr, minlength=N).astype(float); S = lambda v: np.bincount(jr, v, minlength=N)
    res = dict(info, settings=dict(dr=DR, dr3=DR3, zmin=ZMIN, NB=NB, readout='d = change of logit_Z - logit_W'),
               n_particles=int(len(jr)), per_jet=float(len(jr) / N),
               rank_is_pt_order=bool(np.all(np.diff(pt, axis=1) <= 1e-3)), models={})
    zq = np.quantile(z, np.linspace(0, 1, 11)); zb = np.clip(np.searchsorted(zq, z, side='right') - 1, 0, 9)
    for mn in MODELS:
        out = {}; D = {}
        for c in COMBOS:
            sc = Z[f'L_{mn}_{c}'].astype(np.float64); sc = sc[:, 1] - sc[:, 0]; base = Z[f'L_{mn}_full'].astype(np.float64); base = (base[:, 1] - base[:, 0])[brow]
            D[c] = {t: sc[idx[t]] - base[jr] for t in ETYPES[1:]}
        # numerical: the 'part'/'pair' combos of the base row must equal 'full'
        for c in ('part', 'pair'):
            sc = Z[f'L_{mn}_{c}'][ib].astype(np.float64); sf = Z[f'L_{mn}_full'][ib].astype(np.float64)
            out[f'base_mismatch_{c}'] = float(np.abs((sc[:, 1] - sc[:, 0]) - (sf[:, 1] - sf[:, 0])).max())
        for c in COMBOS:
            d = D[c]; o = {}
            for tag, a, b, xo in (('dr0.01', d['rad'], d['radin'], xodd), ('dr0.003', d['rad3'], d['radin3'], None)):
                odd, ev = (a - b) / 2, (a + b) / 2; so, se = S(odd ** 2), S(ev ** 2)
                o[f'even_share_{tag}'] = st(se.sum() / (so.sum() + se.sum()), (Wb @ se) / (Wb @ (so + se)))
                o[f'odd_mean_{tag}'] = st(odd.mean(), (Wb @ S(odd)) / (Wb @ cj)); o[f'even_mean_{tag}'] = st(ev.mean(), (Wb @ S(ev)) / (Wb @ cj))
                o[f'_so_{tag}'], o[f'_se_{tag}'], o[f'_odd_{tag}'] = so, se, odd
            # step-size scaling: smooth response -> even/odd variance ratio x 0.09, odd amplitude x 0.3
            q1 = (Wb @ o['_se_dr0.01']) / (Wb @ o['_so_dr0.01']); q3 = (Wb @ o['_se_dr0.003']) / (Wb @ o['_so_dr0.003'])
            v1 = o['_se_dr0.01'].sum() / o['_so_dr0.01'].sum(); v3 = o['_se_dr0.003'].sum() / o['_so_dr0.003'].sum()
            o['even_odd_var_ratio_0.003_over_0.01'] = st(v3 / v1, q3 / q1, (DR3 / DR) ** 2)
            sn1, sn3 = S(o['_odd_dr0.01'] * o['_odd_dr0.003']), S(o['_odd_dr0.01'] ** 2)
            o['odd_slope_0.003_on_0.01'] = st(sn1.sum() / sn3.sum(), (Wb @ sn1) / (Wb @ sn3), DR3 / DR)
            a1, a3 = S(np.abs(d['rad'])), S(np.abs(d['rad3']))
            o['meanabs_ratio_rad3_over_rad'] = st(a3.sum() / a1.sum(), (Wb @ a3) / (Wb @ a1), DR3 / DR)
            m1, m3 = S(d['rad']), S(d['rad3']); o['mean_ratio_rad3_over_rad'] = st(m3.sum() / m1.sum(), (Wb @ m3) / (Wb @ m1), DR3 / DR)
            o['mean_rad'] = st(d['rad'].mean(), (Wb @ S(d['rad'])) / (Wb @ cj)); o['mean_abs_rad'] = float(np.abs(d['rad']).mean())
            aa, ar = S(np.abs(d['az'])), S(np.abs(d['rad'])); o['az_over_rad_meanabs'] = st(aa.sum() / ar.sum(), (Wb @ aa) / (Wb @ ar))
            o['corr_absrad_absaz'] = float(np.corrcoef(np.abs(d['rad']), np.abs(d['az']))[0, 1])
            ss = S(sg[idx['az']] * d['az']); o['az_signed_mean'] = st(ss.sum() / cj.sum(), (Wb @ ss) / (Wb @ cj), 0.0)
            out[c] = o
        # route additivity and shares (outward move, 0.01)
        f, p, q = D['full']['rad'], D['part']['rad'], D['pair']['rad']
        e1, e2 = S((f - p - q) ** 2), S((f - f.mean()) ** 2); out['route_additivity_R2'] = st(1 - e1.sum() / e2.sum(), 1 - (Wb @ e1) / (Wb @ e2))
        for c, v in (('part', p), ('pair', q)):
            out[f'route_mean_share_{c}'] = st(v.sum() / f.sum(), (Wb @ S(v)) / (Wb @ S(f)))
            sv, sf = S(v ** 2), S(f ** 2); out[f'route_meansq_share_{c}'] = st(sv.sum() / sf.sum(), (Wb @ sv) / (Wb @ sf))
            for t in ('dr0.01',):
                out[f'route_even_meansq_share_{c}'] = st(out[c]['_se_dr0.01'].sum() / out['full']['_se_dr0.01'].sum(),
                                                         (Wb @ out[c]['_se_dr0.01']) / (Wb @ out['full']['_se_dr0.01']))
        # forms on the full outward response and on its odd part: held-out R^2 (5-fold over jets)
        D_ = {'M2': x[:, None], 'pt_only': np.stack([z, z ** 2, np.sqrt(z)], 1), 'pt_plus_M2': np.stack([z, z ** 2, np.sqrt(z), x], 1),
              'r_only': np.stack([r, r ** 2, np.sqrt(r)], 1)}
        forms = {}
        for tname, yv, xv in (('rad', f, x), ('odd', out['full']['_odd_dr0.01'], xodd), ('rad_part', p, x), ('odd_part', out['part']['_odd_dr0.01'], xodd),
                              ('rad_pair', q, x), ('odd_pair', out['pair']['_odd_dr0.01'], xodd)):
            FF = {}
            for fn_, X in D_.items():
                X = X.copy()
                if fn_ in ('M2', 'pt_plus_M2'): X[:, -1] = xv
                ph = np.zeros(len(yv))
                for k in range(5):
                    tr = folds[jr] != k; ph[~tr] = X[~tr] @ lsq(X[tr], yv[tr])
                sse, sst = S((yv - ph) ** 2), S((yv - yv.mean()) ** 2)
                qb = np.quantile(ph, np.linspace(0, 1, 21)); bb = np.clip(np.searchsorted(qb, ph, side='right') - 1, 0, 19)
                nb_ = np.bincount(bb, minlength=20).astype(float); yb = np.bincount(bb, yv, minlength=20) / np.maximum(nb_, 1); pb = np.bincount(bb, ph, minlength=20) / np.maximum(nb_, 1)
                FF[fn_] = dict(R2_heldout=st(1 - sse.sum() / sst.sum(), 1 - (Wb @ sse) / (Wb @ sst)),
                               R2_binned=float(1 - (nb_ * (yb - pb) ** 2).sum() / (nb_ * (yb - yv.mean()) ** 2).sum()), _sse=sse, _sst=sst)
            for fn_ in FF:
                if fn_ != 'M2':
                    FF[fn_]['dR2_vs_M2'] = st(FF[fn_]['R2_heldout']['v'] - FF['M2']['R2_heldout']['v'],
                                              (Wb @ FF['M2']['_sse'] - Wb @ FF[fn_]['_sse']) / (Wb @ FF['M2']['_sst']), 0.0)
            forms[tname] = FF
            # within-z-decile slope on x versus the M2-only slope (pT-only reading: 0; M2 reading: 1)
            Xw = np.concatenate([(zb[:, None] == np.arange(10)[None]).astype(float), xv[:, None]], 1); Xm = xv[:, None]
            k_ = Xw.shape[1]
            Gj = np.zeros((N, k_, k_)); hj = np.zeros((N, k_))
            np.add.at(Gj, jr, Xw[:, :, None] * Xw[:, None, :]); np.add.at(hj, jr, Xw * yv[:, None])
            gm, hm = S(xv * xv), S(xv * yv)
            bw = np.linalg.solve(Gj.sum(0), hj.sum(0))[-1]; bm = hm.sum() / gm.sum()
            Gb = np.einsum('bn,nij->bij', Wb, Gj); hb = Wb @ hj; bwb = np.linalg.solve(Gb, hb[..., None])[:, -1, 0]; bmb = (Wb @ hm) / (Wb @ gm)
            forms[tname]['within_z_slope_over_M2_slope'] = st(bw / bm, bwb / bmb, 0.0)
            forms[tname]['within_z_slope_over_M2_slope_vs1'] = st(bw / bm, bwb / bmb, 1.0)
        out['forms'] = forms
        allr = {}
        for k, v in out.items():
            if isinstance(v, dict): allr[k] = {kk: ({k3: v3 for k3, v3 in vv.items() if not k3.startswith('_')} if isinstance(vv, dict) else vv)
                                               for kk, vv in v.items() if not kk.startswith('_')}
            else: allr[k] = v
        res['models'][mn] = allr
        fo = out['full']
        log(f"{mn}: mean d {fo['mean_rad']['v']:.4f}, even share 0.01 {fo['even_share_dr0.01']['v']:.3f} / 0.003 {fo['even_share_dr0.003']['v']:.3f}, "
            f"az/rad {fo['az_over_rad_meanabs']['v']:.3f}, R2 M2 {forms['rad']['M2']['R2_heldout']['v']:.3f} odd {forms['odd']['M2']['R2_heldout']['v']:.3f}; "
            f"pt_only {forms['rad']['pt_only']['R2_heldout']['v']:.3f}; part-route R2 M2 {forms['rad_part']['M2']['R2_heldout']['v']:.3f} odd {forms['odd_part']['M2']['R2_heldout']['v']:.3f}; within-z slope ratio {forms['odd']['within_z_slope_over_M2_slope']['v']:.2f}; "
            f"route add {out['route_additivity_R2']['v']:.3f}, mean share part {out['route_mean_share_part']['v']:.2f} pair {out['route_mean_share_pair']['v']:.2f}; "
            f"even share part {out['part']['even_share_dr0.01']['v']:.2f} pair {out['pair']['even_share_dr0.01']['v']:.2f}")
    def clean(o):
        if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items() if not str(k).startswith('_')}
        if isinstance(o, (list, tuple)): return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)): return o.item()
        if isinstance(o, np.ndarray): return clean(o.tolist())
        if isinstance(o, float) and not math.isfinite(o): return None
        return o
    FN.write_text(json.dumps(clean(res), indent=1)); log(f'wrote {FN}'); return res


def plot(res):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    R = res['models']; fig, ax = plt.subplots(1, 3, figsize=(16, 4.6)); xs = np.arange(len(MODELS))
    a = ax[0]
    for k, (c, lab) in enumerate((('dr0.01', 'step 0.01 rad'), ('dr0.003', 'step 0.003 rad'))):
        v = np.array([R[m]['full'][f'even_share_{c}']['v'] for m in MODELS]); lo = np.array([R[m]['full'][f'even_share_{c}']['ci'][0] for m in MODELS]); hi = np.array([R[m]['full'][f'even_share_{c}']['ci'][1] for m in MODELS])
        a.bar(xs + (k - 0.5) * 0.35, v, 0.35, yerr=[v - lo, hi - v], capsize=2, label=lab)
    q = np.array([R[m]['full']['even_share_dr0.01']['v'] for m in MODELS]); r_ = q / (1 - q) * 0.09
    a.plot(xs + 0.175, r_ / (1 + r_), 'k_', ms=18, mew=2, label='smooth-response prediction for 0.003')
    a.set_xticks(xs); a.set_xticklabels(MODELS); a.set_ylabel('even share of per-particle variance'); a.legend(fontsize=7); a.set_title('(a) even part (out+in)/2 vs step size', fontsize=9)
    a = ax[1]
    for k, c in enumerate(('part', 'pair')):
        v = np.array([R[m][f'route_mean_share_{c}']['v'] for m in MODELS]); lo = np.array([R[m][f'route_mean_share_{c}']['ci'][0] for m in MODELS]); hi = np.array([R[m][f'route_mean_share_{c}']['ci'][1] for m in MODELS])
        a.bar(xs + (k - 0.5) * 0.35, v, 0.35, yerr=[v - lo, hi - v], capsize=2, label=f'{c}: mean response share')
        w = np.array([R[m][f'route_even_meansq_share_{c}']['v'] for m in MODELS]); a.plot(xs + (k - 0.5) * 0.35, w, 'kx' if k == 0 else 'k+', ms=8, label=f'{c}: even-part mean-square share')
    a.axhline(1, color='k', lw=0.5); a.set_xticks(xs); a.set_xticklabels(MODELS); a.legend(fontsize=7)
    a.set_title('(b) routes: per-particle features only (part) vs pair inputs / kNN coords only (pair)', fontsize=9)
    a = ax[2]
    for k, fn_ in enumerate(('M2', 'pt_only', 'pt_plus_M2', 'r_only')):
        v = np.array([R[m]['forms']['odd'][fn_]['R2_heldout']['v'] for m in MODELS]); a.bar(xs + (k - 1.5) * 0.2, v, 0.2, label=fn_)
    a.set_xticks(xs); a.set_xticklabels(MODELS); a.axhline(0, color='k', lw=0.5); a.set_ylabel('held-out R$^2$, odd part'); a.legend(fontsize=7)
    a.set_title('(c) per-particle forms, fresh rows (_107)', fontsize=9)
    fig.tight_layout(); fig.savefig(PNG, dpi=120); log(f'wrote {PNG}')


if __name__ == '__main__':
    if not ARGS.analyse: collect()
    plot(analyse())
