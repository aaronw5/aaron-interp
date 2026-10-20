"""E25: the official 16-feature JEDI-linear (n64): use of its l1 cuts, origin of its radial second-round objects, and whether its
mass response is relative to the stored jet axis.   python -m interp.e25_jedi16 --sample {main,fresh}
-> $AI_RESULTS/jedi/e25_<sample>.json, a log line per step.

Sample: main = the rows of interp.jedi_core.sample(3000, 2501) (N 15,000); fresh = sample(1500, 2502) minus every main row.
All 16 jetConstituentList columns are read from val_c16_f16.npy at the same rows (opt-in loader load16); its pt/etarel/phirel
columns must equal sample()'s x (abort otherwise).  Column relations used to recompute features after an edit (measured in
e25_diag.json): pT = hypot(px, py), phi = atan2(py, px), eta = asinh(pz/pT), costheta = tanh(eta), deltaR = hypot(etarel, phirel),
eta - etarel and phi - phirel constant per jet (stored axis), ptrel = pT/pT_J, erel = E/E_J; etarot, phirot = offsets from the
leading particle in a frame aligned with the second particle (unchanged by a common translation or rotation, scaled by a
dilation); costhetarel equals etarot within 0.014 for 99 % of particles.
Steps
 S0  accuracy of 16-feature n64, 16-feature-perminv n64, 3-feature n64 on the same jets; l1 input-quantizer kept bits per feature.
 S1  E1: resample ablation of each a/P/g/S unit of 16-feature n64 (unit value replaced by a random other jet's value; a at real slots
     only): % calls changed, binomial 95 % CI, binomial test vs 1 %, Holm per stage.
 S2  E2: per live l1 unit the 16-entry fused kernel in physical units (alpha = K / scale, per GeV / per rad / per unit), spread-weighted
     |alpha_f| sd_f, group shares (pT-like: pt ptrel e erel pz; momentum-xy: px py; angle-relative: etarel phirel; rotated frame:
     etarot phirot costhetarel; radial: deltaR; absolute: eta phi costheta).  Purity = fraction of units whose nonzero entries lie in
     one group (exact) / whose largest group carries >= 95 % of the squared spread-weighted weight (soft).  Null: each feature's
     column permuted over the 64 units, 20,000 draws.  Also for 16-feature-perminv.
 S3  radial objects: 200 jets with >= 12 real particles; the particle at slot k (3, 10) moved to (r, theta) about the stored axis
     (r 0.02-0.3, theta every 15 deg), pT kept, all 16 features recomputed (etarot/phirot by the jet's fitted lead-frame map), g frozen
     at the jet's own value.  Within-jet (fixed pT) radial share of u_c at slot k: 1 - sum (f - ring mean)^2 / sum (f - jet mean)^2.
     Top-10 position-dependent S units by E1.  Null: Dirichlet(1) combinations of the live position-dependent l1 maps, 2,000 draws.
     Same for 3-feature n64.  Input-group zeroing (normalised input set to 0 = training mean at real slots): calls changed and
     per-unit R^2 of S_c(edited) vs S_c for the top-10 units.
 S4  edits (W -> Z on true W jets, Wilson + bootstrap CI; McNemar vs R; flow ratio vs dilM2, paired bootstrap; Holm over arms x models):
     T rigid +0.06 rad in eta about the stored axis (all columns recomputed); T_dR T with deltaR, costhetarel held; T_rel only
     etarel/phirel edited; T_dRonly only deltaR edited; Tabs eta + 0.06 with every relative column held (px py pz e eta costheta
     edited); Tphi phi + 1.0 rad (px py phi edited); R matched rotation about the axis; dilM2 angles about the axis x sqrt(M2(T)/M2)
     (M2 = sum pT r^2).  Models: 16-feature n64, 16-feature-perminv n64, 3-feature n64 (columns 5, 8, 11 of the edited jets),
     16-feature untrained redraws seeds 0-2.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'; OUT.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser(); ap.add_argument('--sample', default='main', choices=['main', 'fresh'])
ap.add_argument('--resume', action='store_true', help='reuse S1/S2 from the existing JSON')
ap.add_argument('--range', action='store_true', help='only the l1 input-range check (unedited + arms) -> e25_<sample>_range.json')
ap.add_argument('--skip', default='', help='comma list of steps to skip (S1,S3)')
args = ap.parse_args()
T0 = time.time()
def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)
RNG = np.random.default_rng(2503 if args.sample == 'main' else 2504)
FEAT16 = ['px', 'py', 'pz', 'e', 'erel', 'pt', 'ptrel', 'eta', 'etarel', 'etarot', 'phi', 'phirel', 'phirot', 'deltaR', 'costheta',
          'costhetarel']
C = {k: i for i, k in enumerate(FEAT16)}
GROUPS = dict(pT_like=['pt', 'ptrel', 'e', 'erel', 'pz'], momentum_xy=['px', 'py'], angle_rel=['etarel', 'phirel'],
              rot_frame=['etarot', 'phirot', 'costhetarel'], radial=['deltaR'], absolute=['eta', 'phi', 'costheta'])
GI = {g: [C[f] for f in fs] for g, fs in GROUPS.items()}
B = 2500
JSON = OUT / f'e25_{args.sample}.json'; R = dict(sample=args.sample, procedure=__doc__)


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray): return clean(o.tolist())
    if isinstance(o, np.bool_): return bool(o)
    return o


def save(): JSON.write_text(json.dumps(clean(R), indent=1))


from interp.e19_common import wilson, mcnemar, holm, boot_mean, boot_ratio  # noqa: E402


# ------------------------------------------------------------------------------------------------ sample
def rows(y, per, seed):
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), per, replace=False) for c in range(5)]))


def load16(per, seed, exclude=None):
    """opt-in loader: the rows of jedi_core.sample(per, seed) with all 16 columns (val_c16_f16.npy, first 64 slots)"""
    import h5py
    from jetdistill.config import RESULTS, DATA
    from interp.jedi_core import sample
    y = np.asarray(np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz')['y'])
    idx = rows(y, per, seed); x3, ys, _ = sample(per_class=per, seed=seed)
    assert np.array_equal(ys, y[idx])
    with h5py.File(Path(DATA) / 'val_meta.h5', 'r') as f: yv = f['label'][:].astype(int)
    assert np.array_equal(yv[idx], ys), 'val_meta labels != sample labels'
    X = np.asarray(np.load(Path(DATA) / 'val_c16_f16.npy', mmap_mode='r')[idx, :64]).astype(np.float32)
    d = np.abs(X[..., [5, 8, 11]] - x3).max(); log(f'load16: per {per} seed {seed}: max |16f cols (pt, etarel, phirel) - sample x| = {d:.3g}')
    if d > 1e-3: raise SystemExit(f'ABORT: 16-feature columns differ from sample() x by {d}')
    keep = np.ones(len(idx), bool) if exclude is None else ~np.isin(idx, exclude)
    return X[keep], ys[keep], idx[keep], float(d)


# ------------------------------------------------------------------------------------------------ kinematics / edits
def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi


def jetinfo(X):
    X = X.astype(np.float64); pt = X[..., C['pt']]; rl = pt > 0; nr = np.maximum(rl.sum(1), 1)
    def jmed(v):  # noqa: E306
        v = np.where(rl, v, np.nan); return np.nanmedian(v, axis=1)
    etaJ = jmed(X[..., C['eta']] - X[..., C['etarel']])
    z = np.where(rl, np.exp(1j * (X[..., C['phi']] - X[..., C['phirel']])), 0).sum(1); phiJ = np.angle(z)
    ptJ = jmed(pt / np.where(rl, X[..., C['ptrel']], 1)); EJ = jmed(X[..., C['e']] / np.where(rl, X[..., C['erel']], 1))
    m2 = np.clip(X[..., C['e']] ** 2 - (pt * np.cosh(X[..., C['eta']])) ** 2, 0, None)
    return dict(rl=rl, etaJ=etaJ, phiJ=phiJ, ptJ=ptJ, EJ=EJ, m2=m2)


def cols(pt, eta, phi, er, pr, m2, ptJ, EJ):
    """the 13 columns computed from (pT, eta, phi, etarel, phirel); rotated-frame columns handled separately"""
    out = dict(px=pt * np.cos(phi), py=pt * np.sin(phi), pz=pt * np.sinh(eta), pt=pt, ptrel=pt / ptJ[:, None], eta=eta, etarel=er,
               phi=phi, phirel=pr, deltaR=np.hypot(er, pr), costheta=np.tanh(eta))
    out['e'] = np.sqrt(m2 + (pt * np.cosh(eta)) ** 2); out['erel'] = out['e'] / EJ[:, None]
    return out


def edit(X, info, der=0.0, dpr=0.0, rot=None, scale=None, deta_abs=0.0, dphi_abs=0.0, edit_cols=None, rot_factor=None):
    """edited copy of X (raw, float32).  Relative-coordinate edit: (etarel, phirel) -> rotation by rot (J,), scale (J,), then + (der, dpr);
    absolute shifts deta_abs, dphi_abs move eta/phi without touching relative columns.  Every recomputed column is changed by
    F(edited) - F(unedited) (stored values kept exactly when the edit is zero).  edit_cols: columns allowed to change (None = all 13
    computed).  rot_factor (J,): etarot, phirot, costhetarel multiplied by it (dilation); None = unchanged."""
    Xd = X.astype(np.float64); rl = info['rl']; pt = Xd[..., C['pt']]; eta = Xd[..., C['eta']]; phi = Xd[..., C['phi']]
    er, pr = Xd[..., C['etarel']], Xd[..., C['phirel']]
    er2, pr2 = er.copy(), pr.copy()
    if rot is not None:
        c, s = np.cos(rot)[:, None], np.sin(rot)[:, None]; er2, pr2 = c * er - s * pr, s * er + c * pr
    if scale is not None: er2, pr2 = er2 * scale[:, None], pr2 * scale[:, None]
    er2 = er2 + der; pr2 = pr2 + dpr
    eta2 = eta + (er2 - er) + deta_abs; phi2 = wrap(phi + (pr2 - pr) + dphi_abs)
    F0 = cols(pt, eta, phi, er, pr, info['m2'], info['ptJ'], info['EJ']); F1 = cols(pt, eta2, phi2, er2, pr2, info['m2'], info['ptJ'], info['EJ'])
    E = Xd.copy()
    for k in F0:
        if edit_cols is not None and k not in edit_cols: continue
        d = F1[k] - F0[k]
        if k == 'phi': d = wrap(d)
        E[..., C[k]] = np.where(rl, Xd[..., C[k]] + d, Xd[..., C[k]])
        if k == 'phi': E[..., C[k]] = np.where(rl, wrap(E[..., C[k]]), E[..., C[k]])
    if rot_factor is not None:
        for k in ('etarot', 'phirot', 'costhetarel'):
            if edit_cols is None or k in edit_cols: E[..., C[k]] = np.where(rl, Xd[..., C[k]] * rot_factor[:, None], Xd[..., C[k]])
    return E.astype(np.float32)


def moments(X):
    X = X.astype(np.float64); pt = X[..., C['pt']]; r2 = X[..., C['etarel']] ** 2 + X[..., C['phirel']] ** 2
    return pt.sum(1), (pt * r2).sum(1)


def edits(X, info):
    pt = X[..., C['pt']].astype(np.float64); rl = info['rl']; r = np.hypot(X[..., C['etarel']], X[..., C['phirel']]).astype(np.float64)
    rbar = (pt * r).sum(1) / np.maximum(pt.sum(1), 1e-12)
    phi = 2 * np.arcsin(np.minimum(1.0, 0.06 / np.maximum(2 * rbar, 1e-12))) * np.random.default_rng(67).choice([-1, 1], len(X))
    M0, M2 = moments(X); er = X[..., C['etarel']].astype(np.float64) + 0.06 * rl
    M2T = (pt * (er ** 2 + X[..., C['phirel']].astype(np.float64) ** 2)).sum(1); sT = np.sqrt(M2T / np.maximum(M2, 1e-30))
    rel = ['etarel', 'phirel']; absc = ['px', 'py', 'pz', 'e', 'eta', 'phi', 'costheta']
    allc = ['px', 'py', 'pz', 'e', 'erel', 'pt', 'ptrel', 'eta', 'etarel', 'phi', 'phirel', 'deltaR', 'costheta']
    E = dict(T=edit(X, info, der=0.06),
             T_dR=edit(X, info, der=0.06, edit_cols=[c for c in allc if c != 'deltaR']),
             T_rel=edit(X, info, der=0.06, edit_cols=rel),
             T_dRonly=edit(X, info, der=0.06, edit_cols=['deltaR']),
             Tabs=edit(X, info, deta_abs=0.06, edit_cols=absc),
             Tphi=edit(X, info, dphi_abs=1.0, edit_cols=['px', 'py', 'phi']),
             R=edit(X, info, rot=phi),
             dilM2=edit(X, info, scale=sT, rot_factor=sT))
    meta = dict(rotation_median_deg=float(np.degrees(np.median(np.abs(phi)))), sT_median=float(np.median(sT)),
                dlogM2_T_mean=float(np.log(M2T / np.maximum(M2, 1e-30)).mean()), half_dlog_M0M2_T_mean=float(0.5 * np.log(M2T / np.maximum(M2, 1e-30)).mean()))
    return E, meta


# ------------------------------------------------------------------------------------------------ model helpers
def quantizers(L, k):
    iq = L[k].iq; return list(iq.quantizers) if hasattr(iq, 'quantizers') else [iq]


def l1_input_info(net, F):
    q = quantizers(net.L, 'l1')[0].quantizer; out = {}
    try:
        def slots(a):
            a = np.asarray(a, np.float64); a2 = a.reshape(a.shape[-2:]) if a.ndim >= 2 else a.reshape(1, -1)
            return np.broadcast_to(a2, (64, F))
        k_, i_, f_ = (slots(getattr(q, a_)) for a_ in ('k', 'i', 'f'))
        bits = np.maximum(k_ + i_ + f_, 0)
        out = dict(kept_slot_frac=(bits > 0).mean(0).tolist(), kept=bits > 0, lo=np.where(k_ > 0, -2.0 ** i_, 0.0), hi=2.0 ** i_ - 2.0 ** -f_)
    except Exception as e:  # noqa: BLE001
        log('l1 quantizer table unavailable:', repr(e))
    return out


def range_frac(net, x, info):
    if 'lo' not in info: return None
    rl = x[..., C['pt'] if x.shape[-1] == 16 else 0] > 0; xn = (x.astype(np.float64) - net.shift) / net.scale
    bad = ((xn < info['lo'][None]) | (xn > info['hi'][None])) & rl[..., None] & info['kept'][None]   # only inputs the l1 quantizer keeps (bits > 0)
    return dict(any_slot_jets=float(bad.any((1, 2)).mean()), per_feature=bad.sum((0, 1)) / max(rl.sum(), 1))


def calls(net, x): return net.run(x, batch=B)['logits'].argmax(1)


def run_all(net, x, keep):
    out = {k: [] for k in keep}
    for i in range(0, len(x), B):
        r = net.run(x[i:i + B], keep=keep, batch=B)
        for k in keep: out[k].append(r[k].astype(np.float32))
    return {k: np.concatenate(v) for k, v in out.items()}


# ------------------------------------------------------------------------------------------------ S1 resample ablation
def e1(net, X, y, st):
    from scipy import stats
    c0 = st['logits'].argmax(1); J = len(X); rl = X[..., C['pt']] > 0; perm = RNG.permutation(J)
    res = {}
    for s in ('a', 'P', 'g', 'S'):
        V = st[s]; t1 = time.time()
        val = V[perm]
        if s == 'a': np.copyto(val, V, where=~rl[..., None])
        ch = np.zeros(64); var = V.reshape(-1, 64).var(0)
        for c in range(64):
            if var[c] == 0: continue
            m = np.zeros(64, np.float32); m[c] = 1
            cc = np.concatenate([net.run_jetwise(X[i:i + B], {s: (m, val[i:i + B])}, batch=B)['logits'].argmax(1) for i in range(0, J, B)])
            ch[c] = (cc != c0).mean()
        k = np.round(ch * J).astype(int)
        p = [float(stats.binomtest(int(k[c]), J, 0.01, alternative='greater').pvalue) if var[c] > 0 else 1.0 for c in range(64)]
        ph = holm(p); live = [c for c in range(64) if ch[c] >= 0.01 and ph[c] < 0.05]
        res[s] = dict(changed=ch, changed_ci=[wilson(int(k[c]), J) for c in range(64)], holm_p=ph, live=live, n_live=len(live),
                      n_constant=int((var == 0).sum()), on_frac=(V.reshape(-1, 64)[rl.reshape(-1)] > 0).mean(0) if s == 'a' else (V > 0).mean(0))
        log(f'S1 {s}: live (>=1 % calls, Holm < .05) {len(live)}/64; constant {int((var == 0).sum())}; top ' +
            ', '.join(f'{c}:{ch[c] * 100:.1f}%' for c in np.argsort(-ch)[:8]) + f' ({time.time() - t1:.0f}s)')
    return res


# ------------------------------------------------------------------------------------------------ S2 kernel
def e2(net, X, live):
    F = X.shape[-1]; K, b = net.KB['l1']; K = K.reshape(F, 64); b = b.reshape(-1)
    iq = l1_input_info(net, F)
    keptf = np.array(iq['kept_slot_frac']) > 0 if 'kept_slot_frac' in iq else np.ones(F, bool)
    Keff = K * keptf[:, None]
    alpha = Keff / net.scale[:, None]; beta = b - (Keff * (net.shift / net.scale)[:, None]).sum(0)
    rl = X[..., C['pt']] > 0; sd = X[rl].astype(np.float64).std(0); w = np.abs(alpha) * sd[:, None]       # spread-weighted, (F, 64)
    nz = Keff != 0; gl = list(GROUPS)
    def gshare(wc): t = (wc ** 2).sum(); return {g: float((wc[GI[g]] ** 2).sum() / t) if t > 0 else None for g in gl}
    def stat(nzm, wm, units):
        u = [c for c in units if nzm[:, c].any()]
        if not u: return np.nan, np.nan
        ex = np.mean([sum(nzm[GI[g], c].any() for g in gl) == 1 for c in u])
        so = np.mean([max((wm[GI[g], c] ** 2).sum() for g in gl) / max((wm[:, c] ** 2).sum(), 1e-300) >= 0.95 for c in u])
        return ex, so
    units = []
    for c in live:
        sh = gshare(w[:, c]); dom = max((v or 0, g) for g, v in sh.items())[1] if any(sh.values()) else None
        units.append(dict(c=int(c), alpha={f: float(alpha[C[f], c]) for f in FEAT16}, beta=float(beta[c]),
                          nonzero=[f for f in FEAT16 if nz[C[f], c]], group_share=sh, dominant=dom,
                          dominant_share=sh[dom] if dom else None, deltaR_share=sh['radial']))
        log(f'S2 a{c}: dominant {dom} {sh[dom] if dom else 0:.3f}; nonzero {units[-1]["nonzero"]}; ' +
            ', '.join(f'{f}={alpha[C[f], c]:+.3g}' for f in FEAT16 if nz[C[f], c]) + f'; beta {beta[c]:+.3g}')
    obs = {}; nulls = {}
    for name, U in (('live', list(live)), ('all', list(range(64)))):
        oe, os_ = stat(nz, w, U); ne, ns = [], []
        for _ in range(20000):
            P = np.array([RNG.permutation(64) for _ in range(F)])
            nzp = np.take_along_axis(nz, P, 1); wp = np.take_along_axis(w, P, 1); e_, s_ = stat(nzp, wp, U); ne.append(e_); ns.append(s_)
        ne, ns = np.array(ne), np.array(ns)
        obs[name] = dict(exact=oe, soft95=os_, n_units=int(sum(nz[:, c].any() for c in U)))
        nulls[name] = dict(exact_q=[float(np.nanquantile(ne, q)) for q in (.025, .5, .975)], exact_p=float((1 + (ne >= oe - 1e-12).sum()) / (1 + len(ne))),
                           soft_q=[float(np.nanquantile(ns, q)) for q in (.025, .5, .975)], soft_p=float((1 + (ns >= os_ - 1e-12).sum()) / (1 + len(ns))))
        log(f'S2 purity ({name}): obs exact {oe:.3f} soft {os_:.3f}; null exact {nulls[name]["exact_q"]} p {nulls[name]["exact_p"]:.4f}; '
            f'soft {nulls[name]["soft_q"]} p {nulls[name]["soft_p"]:.4f}')
    dom_counts = {g: sum(1 for u in units if u['dominant'] == g) for g in gl}
    return dict(units=units, nonzero_per_feature={f: int(nz[C[f]].sum()) for f in FEAT16}, units_any_nonzero=int(nz.any(0).sum()),
                l1_input_kept_slot_frac={f: (iq['kept_slot_frac'][C[f]] if 'kept_slot_frac' in iq else None) for f in FEAT16},
                feature_sd={f: float(sd[C[f]]) for f in FEAT16}, purity=obs, null=nulls, dominant_counts=dom_counts)


# ------------------------------------------------------------------------------------------------ S3 radial maps
RGRID = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.3]); TGRID = np.radians(np.arange(0, 360, 15))


def leadmap(Xj, rlj):
    """per-jet least-squares map (lead-relative (deta, dphi)) -> (etarot, phirot)"""
    d = np.stack([Xj[:, C['etarel']] - Xj[0, C['etarel']], wrap(Xj[:, C['phirel']] - Xj[0, C['phirel']])], 1)[rlj]
    Y = np.stack([Xj[:, C['etarot']], Xj[:, C['phirot']]], 1)[rlj]
    W, *_ = np.linalg.lstsq(d, Y, rcond=None); return W, float(np.abs(d @ W - Y).max())


def move_variants(Xj, info_j, k, W):
    """jet j copied len(RGRID)*len(TGRID) times with particle k moved to (r cos t, r sin t) about the stored axis"""
    RR, TT = np.meshgrid(RGRID, TGRID, indexing='ij'); n = RR.size
    E = np.repeat(Xj[None].astype(np.float64), n, 0)
    er0, pr0 = E[:, k, C['etarel']].copy(), E[:, k, C['phirel']].copy()
    er1, pr1 = (RR * np.cos(TT)).ravel(), (RR * np.sin(TT)).ravel()
    pt = E[:, k, C['pt']]; eta1 = E[:, k, C['eta']] + (er1 - er0); phi1 = wrap(E[:, k, C['phi']] + (pr1 - pr0))
    ptJ, EJ = np.full(n, info_j['ptJ']), np.full(n, info_j['EJ']); m2 = np.full(n, info_j['m2'][k])
    F0 = cols(pt[:, None], E[:, k, C['eta']][:, None], E[:, k, C['phi']][:, None], er0[:, None], pr0[:, None], m2[:, None], ptJ, EJ)
    F1 = cols(pt[:, None], eta1[:, None], phi1[:, None], er1[:, None], pr1[:, None], m2[:, None], ptJ, EJ)
    for f in F0:
        d = F1[f][:, 0] - F0[f][:, 0]
        E[:, k, C[f]] = wrap(E[:, k, C[f]] + d) if f == 'phi' else E[:, k, C[f]] + d
    drot = np.stack([er1 - er0, pr1 - pr0], 1) @ W
    E[:, k, C['etarot']] += drot[:, 0]; E[:, k, C['phirot']] += drot[:, 1]; E[:, k, C['costhetarel']] += drot[:, 0]
    return E.astype(np.float32)


def share(f):
    """f (jets, r, theta) -> pooled within-jet radial share"""
    az = ((f - f.mean(2, keepdims=True)) ** 2).sum(); tot = ((f - f.mean((1, 2), keepdims=True)) ** 2).sum()
    return float(1 - az / tot) if tot > 1e-18 else np.nan, float(tot)


def e3(nets, X, st, e1res, live_a):
    rl = X[..., C['pt']] > 0; info = jetinfo(X)
    cand = np.flatnonzero(rl.sum(1) >= 12); jets = np.sort(RNG.choice(cand, 200, replace=False))
    out = dict(jets=len(jets), slots={}, fit_resid=[])
    for k in (3, 10):
        maps = {name: dict(u=[], a=[]) for name in nets}
        for j in jets:
            W, res = leadmap(X[j].astype(np.float64), rl[j]); out['fit_resid'].append(res)
            ij = {kk: (v[j] if np.ndim(v) else v) for kk, v in info.items() if kk != 'rl'}
            V = move_variants(X[j], ij, k, W)
            for name, net in nets.items():
                xin = V if name.startswith('16') else V[..., [5, 8, 11]]
                g = st[name]['g'][j]
                r = net.run_jetwise(xin, {'g': (np.ones(64, np.float32), np.repeat(g[None], len(V), 0))}, keep=('u', 'a'), batch=len(V))
                # copies: a reshaped slice is a view that would keep the whole (144, 64, 64) float64 output alive (OOM in e25-run/-run2)
                for st_ in ('u', 'a'): maps[name][st_].append(np.array(r[st_][:, k, :], np.float32).reshape(len(RGRID), len(TGRID), 64))
                del r
        res_k = {}
        for name in nets:
            U = np.stack(maps[name]['u']); A = np.stack(maps[name]['a'])           # (jets, r, t, 64)
            ch = np.asarray(e1res[name]['S']['changed']) if name in e1res else np.asarray(st[name]['Svar'])
            pos = []
            for c in range(64):
                sh, tot = share(U[..., c]); across = float(st[name]['u_k_var'][k][c])
                pos.append(dict(c=c, share=sh, map_var=tot / U[..., c].size, across_jet_var=across))
            dep = [p for p in pos if p['across_jet_var'] > 0 and p['map_var'] > 0.01 * p['across_jet_var']]
            top = sorted(dep, key=lambda p: -ch[p['c']])[:10]
            la = live_a[name]
            Aa = A[..., la]; avar = np.array([share(Aa[..., i])[1] for i in range(len(la))]); keep = avar > 1e-12
            Aa = Aa[..., keep]; ds = []
            for _ in range(2000):
                wgt = RNG.dirichlet(np.ones(Aa.shape[-1])); ds.append(share(Aa @ wgt)[0])
            ds = np.array(ds); own = [share(Aa[..., i])[0] for i in range(Aa.shape[-1])]
            eq = share(Aa.sum(-1))[0]
            for p in top: p['E1_changed'] = float(ch[p['c']]); p['above_null_975'] = bool(p['share'] > np.nanquantile(ds, .975))
            res_k[name] = dict(top10=top, n_position_dependent=len(dep), null=dict(n_units=int(keep.sum()), own_shares=own, equal_weights=eq,
                               dirichlet_q=[float(np.nanquantile(ds, q)) for q in (.025, .5, .975)], frac_ge_097=float(np.nanmean(ds >= 0.97))))
            log(f'S3 slot {k} {name}: {len(dep)} position-dependent u units; top10 ' +
                ', '.join(f"S{p['c']}({p['E1_changed'] * 100:.1f}%):{p['share']:.3f}" for p in top) +
                f"; Dirichlet null {res_k[name]['null']['dirichlet_q']} (units {int(keep.sum())}), equal-weight {eq:.3f}")
        out['slots'][str(k)] = res_k
    out['fit_resid'] = [float(np.quantile(out['fit_resid'], q)) for q in (.5, .99, 1)]
    return out


def zero_groups(net, X, S0, c0, top):
    rl = X[..., C['pt']] > 0; out = {}
    groups = dict(GROUPS); groups['etarel_only'] = ['etarel']; groups['phirel_only'] = ['phirel']
    for g, fs in groups.items():
        E = X.copy()
        for f in fs: E[..., C[f]] = np.where(rl, np.float32(net.shift[C[f]]), X[..., C[f]])
        r = run_all(net, E, ('S', 'logits')); cc = r['logits'].argmax(1); k = int((cc != c0).sum())
        r2 = {}
        for c in top:
            v = S0[:, c].var(); r2[c] = float(1 - ((r['S'][:, c] - S0[:, c]) ** 2).mean() / v) if v > 0 else None
        out[g] = dict(calls_changed=k / len(X), ci=wilson(k, len(X)), S_R2=r2)
        log(f'S3 zero {g} {fs}: calls changed {k / len(X) * 100:.2f}%; S R2 ' + ', '.join(f'S{c}:{v:.3f}' for c, v in r2.items() if v is not None))
    return out


# ------------------------------------------------------------------------------------------------ S4 edits
def s4(models, X, y, E):
    W = y == 2; nW = int(W.sum()); out = {}; tests = []
    for name, net in models.items():
        sel = (lambda x: x) if name.startswith('16') else (lambda x: x[..., [5, 8, 11]])
        c0 = calls(net, sel(X)); z0 = c0[W] == 3; Z = {}; d = dict(N=len(X), N_W=nW, acc=float((c0 == y).mean()), W_to_Z_unedited=float(z0.mean()), arms={})
        for arm, xe in E.items():
            ce = calls(net, sel(xe)); Z[arm] = ce[W] == 3
            d['arms'][arm] = dict(W_to_Z=float(Z[arm].mean()), wilson=wilson(int(Z[arm].sum()), nW),
                                  boot=boot_mean(Z[arm].astype(float), RNG), calls_changed=float((ce != c0).mean()),
                                  W_calls_changed=float((ce[W] != c0[W]).mean()), acc=float((ce == y).mean()))
        fD = Z['dilM2'].astype(float) - z0
        for arm in E:
            if arm != 'R':
                d['arms'][arm]['mcnemar_vs_R'] = mcnemar(Z['R'], Z[arm]); tests.append((name, arm, d['arms'][arm]['mcnemar_vs_R']['p']))
            fa = Z[arm].astype(float) - z0
            d['arms'][arm]['flow'] = float(fa.mean()); d['arms'][arm]['flow_ci'] = boot_mean(fa, RNG)
            if fD.mean() > 0:
                d['arms'][arm]['ratio_flow_dilM2'] = float(fa.mean() / fD.mean()); d['arms'][arm]['ratio_flow_dilM2_ci'] = boot_ratio(fa, fD, RNG)
        for a, b in (('T_rel', 'T'), ('T_dR', 'T'), ('Tabs', 'T'), ('T_dRonly', 'T')):
            d[f'mcnemar_{a}_vs_{b}'] = mcnemar(Z[a], Z[b])
        out[name] = d
        log(f'S4 {name}: acc {d["acc"]:.4f}; W->Z unedited {d["W_to_Z_unedited"]:.3f}; ' +
            '; '.join(f'{a} {v["W_to_Z"]:.3f} (calls {v["calls_changed"] * 100:.1f}%, /dil {v.get("ratio_flow_dilM2", float("nan")):.2f})' for a, v in d['arms'].items()))
    ph = holm([t[2] for t in tests])
    for (name, arm, p), h in zip(tests, ph): out[name]['arms'][arm]['holm_p_vs_R'] = h
    return out


# ------------------------------------------------------------------------------------------------ main
def range_only(X):
    from interp.jedi_core import Jedi
    info = jetinfo(X); E, _ = edits(X, info); out = {}
    for name in ('16-feature', '16-feature-perminv'):
        net = Jedi(64, name); iq = l1_input_info(net, 16)
        out[name] = dict(unedited=range_frac(net, X, iq), **{arm: range_frac(net, xe, iq) for arm, xe in E.items()},
                         kept_slots={f: np.flatnonzero(iq['kept'][:, C[f]]).tolist() for f in FEAT16})
        log(name, {k: (v['any_slot_jets'] if isinstance(v, dict) and 'any_slot_jets' in v else None) for k, v in out[name].items()})
    (OUT / f'e25_{args.sample}_range.json').write_text(json.dumps(clean(out), indent=1)); log('done')


def main():
    from interp.jedi_core import Jedi
    skip = set(args.skip.split(',')) - {''}
    if args.sample == 'main': X, y, idx, d = load16(3000, 2501)
    else:
        from jetdistill.config import RESULTS
        yy = np.asarray(np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz')['y'])
        X, y, idx, d = load16(1500, 2502, exclude=rows(yy, 3000, 2501))
    if args.range: return range_only(X)
    global OLD
    OLD = json.loads(JSON.read_text()) if (args.resume and JSON.exists()) else {}
    R.update(N=len(X), class_counts=np.bincount(y, minlength=5), max_col_diff_vs_sample=d, seed=dict(main=(3000, 2501), fresh=(1500, 2502))[args.sample])
    log(f'sample {args.sample}: N {len(X)}'); info = jetinfo(X)
    R['axis_check'] = dict(etaJ_spread_med=float(np.nanmedian(np.nanmax(np.abs(np.where(info['rl'], X[..., C['eta']] - X[..., C['etarel']], np.nan) - info['etaJ'][:, None]), 1))))
    nets = {'16f': Jedi(64, '16-feature'), '16f_perminv': Jedi(64, '16-feature-perminv'), '3f': Jedi(64, '3-feature')}
    st = {}; R['S0'] = {}
    for name, net in nets.items():
        xin = X if name.startswith('16') else X[..., [5, 8, 11]]
        st[name] = run_all(net, xin, ('a', 'P', 'g', 'S', 'logits') if name == '16f' else ('g', 'S', 'logits'))
        c0 = st[name]['logits'].argmax(1); iq = l1_input_info(net, xin.shape[-1])
        R['S0'][name] = dict(acc=float((c0 == y).mean()), acc_ci=wilson(int((c0 == y).sum()), len(y)), range_unedited=range_frac(net, xin, iq),
                             l1_kept_slot_frac=iq.get('kept_slot_frac'))
        log(f'S0 {name}: accuracy {R["S0"][name]["acc"]:.4f}; l1 input kept-slot fraction {iq.get("kept_slot_frac")}; out-of-range {R["S0"][name]["range_unedited"]}')
    save()

    e1res = {}
    old = OLD
    if 'S1' in old and 'S2' in old and '16f_perminv' in old['S2']:
        R['S1'], R['S2'] = old['S1'], old['S2']; e1res['16f'] = R['S1']; live16 = R['S1']['a']['live']; log('resume: S1, S2 from', JSON.name)
    else:
        if 'S1' not in skip:
            e1res['16f'] = e1(nets['16f'], X, y, st['16f']); R['S1'] = e1res['16f']; save()
        live16 = e1res['16f']['a']['live'] if '16f' in e1res else list(range(64))
        R['S2'] = {'16f': e2(nets['16f'], X, live16)}; save()
        R['S2']['16f_perminv'] = e2(nets['16f_perminv'], X, [c for c in range(64) if (nets['16f_perminv'].KB['l1'][0].reshape(16, 64)[:, c] != 0).any()])
        save()
    for name in list(st): st[name].pop('a', None)
    import gc; gc.collect()

    # S3
    if 'S3' not in skip:
        for name in ('16f', '3f'):
            st[name]['Svar'] = st[name]['S'].var(0)
            st[name]['u_k_var'] = {}
        for name in ('16f', '3f'):
            xin = X if name == '16f' else X[..., [5, 8, 11]]
            vs = {3: [], 10: []}
            for i in range(0, len(X), B):
                u = nets[name].run(xin[i:i + B], keep=('u',), batch=B)['u']
                for k in vs: vs[k].append(u[:, k, :].astype(np.float32))
                del u
            for k in vs: st[name]['u_k_var'][k] = np.concatenate(vs[k])[(X[:, k, C['pt']] > 0)].var(0)
        e1S = {'16f': e1res.get('16f', {'S': {'changed': st['16f']['Svar']}})}
        e1S['3f'] = {'S': {'changed': st['3f']['Svar'] / max(st['3f']['Svar'].max(), 1e-30)}}   # 3f: rank by S variance (no E1 run here)
        live_a = {'16f': [c for c in live16 if c < 64], '3f': [c for c in range(64) if (nets['3f'].KB['l1'][0].reshape(3, 64)[:, c] != 0).any()]}
        R['S3'] = dict(maps=e3({'16f': nets['16f'], '3f': nets['3f']}, X, st, e1S, live_a)); save()
        top = [p['c'] for p in R['S3']['maps']['slots']['3']['16f']['top10']]
        R['S3']['zero_groups'] = zero_groups(nets['16f'], X, st['16f']['S'], st['16f']['logits'].argmax(1), top); save()
    for name in list(st): st[name].pop('a', None)

    # S4
    E, meta = edits(X, info); R['S4_meta'] = meta; log('S4 edit meta', meta)
    for arm, xe in E.items():
        dd = np.abs(xe.astype(np.float64) - X.astype(np.float64)).max((0, 1)); R.setdefault('S4_cols_changed', {})[arm] = {f: float(dd[C[f]]) for f in FEAT16}
    iq = l1_input_info(nets['16f'], 16); R['S4_range'] = {arm: range_frac(nets['16f'], xe, iq) for arm, xe in E.items()}
    models = dict(nets)
    for s in (0, 1, 2): models[f'16f_untrained_s{s}'] = Jedi(64, '16-feature', untrained=True, seed=s)
    R['S4'] = s4(models, X, y, E)
    R['wall_s'] = time.time() - T0
    import resource; R['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024; save(); log('done')


main()
