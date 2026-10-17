"""P1v — verification of P1 (inbox/P1_translation.md): ParT under a rigid φ translation of every particle at a fixed jet axis.

Checks (task P1v):
 (1) edit hygiene at φ 0.06: every network input before / after the edit — the per-particle features as fed (network.inputs:
     scaled, clipped, wrap-padded) and raw, the 4-vectors, and the 4 pair features computed with weaver's own pairwise_lv_fts
     (the function PairEmbed calls), on real-real pairs; float32 noise reference: the same pair features in float64.
 (2) which input carries the response: φ δ ∈ {0.03, 0.06, 0.09, −0.06} with (a) the full edit, (b) deltaR fed unedited,
     (c) dphi fed unedited, (d) both fed unedited (4-vectors still edited: only the pair route remains), (f) per-particle
     features edited but 4-vectors unedited (pair route removed). Metrics as P1 (ridge read-out fitted on the unedited jets,
     same seed / split / alphas).
 (3) fresh sample: rows 6000-7000 of each class's first test file, φ 0.06; read-out refitted on the fresh sample and P1's
     read-out (fitted on rows 3000-4000) applied unchanged.
 (4) axis convention: deta, dphi, deltaR recomputed about (i) the stored axis through the same code (identity control),
     (ii) the E-scheme axis recomputed from the stored particles (control), (iii) the hardest particle, (iv) the WTA axis
     (C/A clustering of all particles in (η, φ), winner-take-all recombination: pT summed, direction of the harder branch);
     4-vectors, jet_pt, jet_energy unchanged, so only deta, dphi, deltaR change. deta keeps JetClass's sign convention
     (deta = sign(axis η)·(η − axis η)).
 (5) numeric checks of the P1 pipeline: padding unchanged, network.run == the forward on precomputed inputs, P1's numbers.
  ./part_job.sh p1v-run 'python -m interp.p1v_verify_part'          -> results/part/p1v_verify.json
  ./part_job.sh p1v-quick 'python -m interp.p1v_verify_part 50'     (50 jets per class, smoke test)
"""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp.part_core import ParTNetwork, inputs, sample, kin, build, jmass, moments, OUT, CL, N98
from jetdistill.part.network import FEATURES, raw_features

log = lambda *a: print(*a, flush=True)
iW, iZ = CL.index('Wqq'), CL.index('Zqq')
SEED = 1
ALPHAS = 10.0 ** np.arange(-4, 4.1, 0.5)
DELTAS = (0.03, 0.06, 0.09, -0.06)


class Ridge:            # verbatim from interp/p1_translation_part.py
    def __init__(self, X, t, idx_fit, idx_val, idx_all):
        self.mu = X[idx_all].mean(0); self.sd = X[idx_all].std(0); self.sd[self.sd < 1e-9] = np.inf
        Z = (X - self.mu) / self.sd; self.tm = t[idx_all].mean(); self.ts = t[idx_all].std()
        tt = (t - self.tm) / self.ts; best = None
        for a in ALPHAS:
            w = self._fit(Z[idx_fit], tt[idx_fit], a); e = ((Z[idx_val] @ w - tt[idx_val]) ** 2).mean()
            if best is None or e < best[0]: best = (e, a)
        self.alpha = best[1]; self.w = self._fit(Z[idx_all], tt[idx_all], self.alpha)

    @staticmethod
    def _fit(Z, t, a):
        Zc = Z - Z.mean(0); return np.linalg.solve(Zc.T @ Zc + a * len(Z) * np.eye(Z.shape[1]), Zc.T @ (t - t.mean()))

    def __call__(self, X):
        return ((X - self.mu) / self.sd) @ self.w * self.ts + self.tm


r2 = lambda p, t: float(1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum())


def split(N):           # as P1: fixed 50/50 split (seed 1), alpha chosen on the last quarter of the train half
    rng = np.random.default_rng(SEED); perm = rng.permutation(N); tr, te = np.sort(perm[:N // 2]), np.sort(perm[N // 2:])
    return tr[: len(tr) * 3 // 4], tr[len(tr) * 3 // 4:], tr, te


def prep(per, start):
    J0 = sample(per, start); pt, de, dp, ms = kin(J0); Jb = build(J0, pt, de, dp, ms); m0 = jmass(Jb)
    return dict(J0=J0, y=J0['label'].astype(int), pt=pt, de=de, dp=dp, ms=ms, real=pt > 0, Jb=Jb, m0=m0, logm0=np.log(np.maximum(m0, 1.0)),
                M=moments(*kin(Jb)[:3]), start=start)


def phi_edit(S, d):
    return build(S['J0'], S['pt'], S['de'], S['dp'] + d * S['real'], S['ms'])


def pred_of(S, Je):
    return 0.5 * np.log(moments(*kin(Je)[:3])['M0M2'] / S['M']['M0M2'])


def fwd(net, x, v, m, batch=256):
    """network.run on precomputed inputs (same multiplicity order and batches; each batch cut to its largest multiplicity —
    the wrap padding of slot k is particle k mod n, independent of the cut)"""
    torch = net.torch; N = len(x); order = np.argsort(m[:, 0].sum(1), kind='stable')
    Z = np.zeros((N, 128), np.float32); L = np.zeros((N, 10), np.float32)
    dev = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(net.device)
    with torch.no_grad():
        for i in range(0, N, batch):
            idx = order[i:i + batch]; P = int(m[idx, 0].sum(1).max())
            out = net.model(dev(x[idx, :, :P]), dev(v[idx, :, :P]), dev(m[idx, :, :P]))
            L[idx] = out.cpu().numpy(); Z[idx] = net._cls['h'].cpu().numpy()
    return Z.astype(np.float64), L.astype(np.float64)


def metrics(z, L, B, pred=None):
    y, te = B['y'], B['te']; c = L.argmax(1); z0, L0, c0, sd0 = B['z0'], B['L0'], B['c0'], B['sd0']
    d97 = (z[:, N98] - z0[:, N98]) / sd0[N98]; dl = B['ridge'](z[te]) - B['ridge'](z0[te])
    e = dict(acc=float((c == y).mean()), changed=float((c != c0).mean()), W_to_Z=float((c[y == iW] == iZ).mean()), Z_to_W=float((c[y == iZ] == iW).mean()),
             W_to_W=float((c[y == iW] == iW).mean()), dlogit_WZ_on_W=float(((L[:, iW] - L[:, iZ]) - (L0[:, iW] - L0[:, iZ]))[y == iW].mean()),
             n98_all=float(d97.mean()), n98_W=float(d97[y == iW].mean()), n98_Z=float(d97[y == iZ].mean()),
             dlogm_hat=float(dl.mean()), dlogm_hat_W=float(dl[y[te] == iW].mean()), dlogm_hat_Z=float(dl[y[te] == iZ].mean()),
             max_abs_dlogit=float(np.abs(L - L0).max()), max_abs_dz=float(np.abs(z - z0).max()))
    if pred is not None:
        e['pred'] = float(pred[te].mean()); e['pred_W'] = float(pred[te][y[te] == iW].mean())
        e['ratio'] = e['dlogm_hat'] / e['pred'] if abs(e['pred']) > 1e-9 else None
    return e


def fmt(name, e):
    s = (f'   {name:28s} | acc {100 * e["acc"]:5.1f} changed {100 * e["changed"]:5.2f} % | W→Z {e["W_to_Z"]:.3f} Z→W {e["Z_to_W"]:.3f} | '
         f'Δ(lW−lZ)|W {e["dlogit_WZ_on_W"]:+.2f} | idx97/sd all {e["n98_all"]:+.3f} W {e["n98_W"]:+.3f} Z {e["n98_Z"]:+.3f} | '
         f'Δlog m̂ {e["dlogm_hat"]:+.4f} (W {e["dlogm_hat_W"]:+.4f}) | max|ΔL| {e["max_abs_dlogit"]:.2e}')
    if 'pred' in e: s += f' | pred {e["pred"]:+.4f} ratio {e["ratio"] if e["ratio"] is None else round(e["ratio"], 3)}'
    return s


# ------------------------------------------------------------------------------------------------ check 1: pair features
def pair_check(vb, ve, m, batch=48):
    """weaver's pairwise_lv_fts (as PairEmbed calls it: eps 1e-8, 4 outputs) on the baseline and edited 4-vectors; |Δ| on
    real-real pairs (diagonal included, as PairEmbed's tril with offset 0), and the float32-vs-float64 spread of the baseline"""
    import torch
    from functools import partial
    from weaver.nn.model.ParticleTransformer import pairwise_lv_fts
    f = partial(pairwise_lv_fts, num_outputs=4, eps=1e-8, for_onnx=False)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    TH = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
    acc = dict(n=0, max=np.zeros(4), max_offdiag=np.zeros(4), gt=np.zeros((len(TH), 4)), f64_max=np.zeros(4), f64_gt=np.zeros((len(TH), 4)),
               sum_sq=np.zeros(4), sum_=np.zeros(4), max_dR_gt_1e2=np.zeros(4))
    order = np.argsort(m[:, 0].sum(1), kind='stable')
    with torch.no_grad():
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]; P = int(m[idx, 0].sum(1).max())
            tb = torch.from_numpy(np.ascontiguousarray(vb[idx, :, :P])).to(dev); tE = torch.from_numpy(np.ascontiguousarray(ve[idx, :, :P])).to(dev)
            mm = torch.from_numpy(np.ascontiguousarray(m[idx, 0, :P])).to(dev) > 0
            pm = (mm[:, :, None] & mm[:, None, :])                                   # (B, P, P) real-real pairs
            off = pm & ~torch.eye(P, dtype=torch.bool, device=dev)[None]
            Fb = f(tb.unsqueeze(-1), tb.unsqueeze(-2)); Fe = f(tE.unsqueeze(-1), tE.unsqueeze(-2))
            tb64 = tb.double(); Fb64 = f(tb64.unsqueeze(-1), tb64.unsqueeze(-2))
            D = (Fe - Fb).abs(); D64 = (Fb.double() - Fb64).abs(); far = off & (Fb[:, 2] > np.log(1e-2))   # ΔR > 0.01
            acc['n'] += int(pm.sum())
            for k in range(4):
                dk = D[:, k][pm]; acc['max'][k] = max(acc['max'][k], float(dk.max())); acc['max_offdiag'][k] = max(acc['max_offdiag'][k], float(D[:, k][off].max()))
                acc['max_dR_gt_1e2'][k] = max(acc['max_dR_gt_1e2'][k], float(D[:, k][far].max()) if bool(far.any()) else 0.0)
                d64 = D64[:, k][pm]; acc['f64_max'][k] = max(acc['f64_max'][k], float(d64.max()))
                for t, th in enumerate(TH):
                    acc['gt'][t, k] += int((dk > th).sum()); acc['f64_gt'][t, k] += int((d64 > th).sum())
                fv = Fb[:, k][pm].double(); acc['sum_'][k] += float(fv.sum()); acc['sum_sq'][k] += float((fv ** 2).sum())
    n = acc['n']; sd = np.sqrt(acc['sum_sq'] / n - (acc['sum_'] / n) ** 2)
    names = ['lnkT', 'lnz', 'lnDeltaR', 'lnm2']
    return dict(pairs=n, thresholds=list(TH), features=names, spread_sd=dict(zip(names, sd.tolist())),
                max_abs_change=dict(zip(names, acc['max'].tolist())), max_abs_change_offdiag=dict(zip(names, acc['max_offdiag'].tolist())),
                max_abs_change_dR_gt_0p01=dict(zip(names, acc['max_dR_gt_1e2'].tolist())),
                frac_change_gt={f'{th:g}': dict(zip(names, (acc['gt'][t] / n).tolist())) for t, th in enumerate(TH)},
                f32_vs_f64_max=dict(zip(names, acc['f64_max'].tolist())),
                f32_vs_f64_frac_gt={f'{th:g}': dict(zip(names, (acc['f64_gt'][t] / n).tolist())) for t, th in enumerate(TH)})


def feature_check(Jb, Je, n):
    names = [f[0] for f in FEATURES[n]]; xb, vb, mb = inputs(Jb, n); xe, ve, me = inputs(Je, n); real = mb[:, 0] > 0
    out = dict(mask_equal=bool(np.array_equal(mb, me)), P=[int(xb.shape[-1]), int(xe.shape[-1])], fed={}, raw={}, vec={})
    for i, k in enumerate(names):
        out['fed'][k] = dict(max_real=float(np.abs(xe[:, i] - xb[:, i])[real].max()), max_all_slots=float(np.abs(xe[:, i] - xb[:, i]).max()),
                             mean_real=float((xe[:, i] - xb[:, i])[real].mean()))
        rb, re_ = raw_features(Jb, k), raw_features(Je, k); r128 = Jb['mask'] > 0
        out['raw'][k] = dict(max_real=float(np.abs(re_ - rb)[r128].max()), mean_real=float((re_ - rb)[r128].mean()))
    for j, k in enumerate(('px', 'py', 'pz', 'energy')):
        out['vec'][k] = float(np.abs(ve[:, j] - vb[:, j])[real].max())
    out['jet_pt_max_rel'] = float(np.abs(Je['jet_pt'] / Jb['jet_pt'] - 1).max()); out['jet_energy_max_rel'] = float(np.abs(Je['jet_energy'] / Jb['jet_energy'] - 1).max())
    return out, (xb, vb, mb, xe, ve, me, names)


# ------------------------------------------------------------------------------------------------ check 4: axes
def wta_axis_one(pt, y, p):
    """C/A clustering (smallest ΔR² in (y, p) merged first, no beam distance) with winner-take-all recombination: the merged
    branch has pT_i + pT_j and the (y, p) of the harder branch; returns the (y, p) of the final branch"""
    n = len(pt)
    if n == 1: return float(y[0]), float(p[0])
    pt = pt.astype(np.float64).copy(); y = y.astype(np.float64).copy(); p = p.astype(np.float64).copy(); alive = np.ones(n, bool)
    D = (y[:, None] - y[None, :]) ** 2 + (p[:, None] - p[None, :]) ** 2; np.fill_diagonal(D, np.inf); i = 0
    for _ in range(n - 1):
        i, j = divmod(int(np.argmin(D)), n)
        if pt[j] > pt[i]: y[i], p[i] = y[j], p[j]
        pt[i] += pt[j]; alive[j] = False; D[j, :] = np.inf; D[:, j] = np.inf
        d = (y - y[i]) ** 2 + (p - p[i]) ** 2; d[~alive] = np.inf; d[i] = np.inf; D[i, :] = d; D[:, i] = d
    return float(y[i]), float(p[i])


def wta_selftest():
    # prong A: pT 40 + 5 at (0, 0), (0.01, 0); prong B: 30 + 25 at (0.3, 0), (0.31, 0): B is harder (55 > 45) -> WTA axis (0.3, 0),
    # hardest particle (0, 0). Second case: A 50 + 10 (60 > 55) -> (0, 0).
    a = wta_axis_one(np.array([40., 30., 25., 5.]), np.array([0., 0.3, 0.31, 0.01]), np.zeros(4))
    b = wta_axis_one(np.array([50., 30., 25., 10.]), np.array([0., 0.3, 0.31, 0.01]), np.zeros(4))
    c = wta_axis_one(np.array([7.]), np.array([0.2]), np.array([-0.1]))
    ok = a == (0.3, 0.0) and b == (0.0, 0.0) and c == (0.2, -0.1)
    return dict(ok=bool(ok), a=a, b=b, c=c)


def wta_axes(J):
    """per jet, the WTA axis in the stored (deta, dphi) coordinates (the stored deta carries sign(jet_eta): distances are the
    same as in (η, φ))"""
    m = J['mask'] > 0; pt = np.hypot(J['px'], J['py']); out = np.zeros((len(pt), 2))
    for r in range(len(pt)):
        k = m[r]; out[r] = wta_axis_one(pt[r, k], J['deta'][r, k], J['dphi'][r, k])
    return out[:, 0], out[:, 1]


def reaxis(J, eta_a, phi_a):
    """the same particles (4-vectors, jet_pt, jet_energy unchanged) with deta, dphi about the axis (eta_a, phi_a) [absolute];
    deta = sign(eta_a)·(η − eta_a) (JetClass convention), dphi wrapped to [−π, π)"""
    m = J['mask'] > 0; sgn = np.where(J['jet_eta'] < 0, -1.0, 1.0)
    eta_i = J['jet_eta'][:, None].astype(np.float64) + sgn[:, None] * J['deta']; phi_i = J['jet_phi'][:, None].astype(np.float64) + J['dphi']
    sa = np.where(eta_a < 0, -1.0, 1.0)
    de = sa[:, None] * (eta_i - eta_a[:, None]); dp = (phi_i - phi_a[:, None] + np.pi) % (2 * np.pi) - np.pi
    E = dict(J); E['deta'] = np.where(m, de, 0).astype(np.float32); E['dphi'] = np.where(m, dp, 0).astype(np.float32)
    E['jet_eta'] = np.asarray(eta_a, np.float32); E['jet_phi'] = np.asarray(phi_a, np.float32)
    return E, int((sa != sgn).sum())


def axis_set(S):
    J = S['Jb']; m = J['mask'] > 0; sgn = np.where(J['jet_eta'] < 0, -1.0, 1.0); je, jp = J['jet_eta'].astype(np.float64), J['jet_phi'].astype(np.float64)
    from_stored = lambda de, dp: (je + sgn * de, jp + dp)
    Px, Py, Pz = ((J[k].astype(np.float64) * m).sum(1) for k in ('px', 'py', 'pz'))
    t = time.time(); wd, wp = wta_axes(J); twta = time.time() - t
    A = dict(stored=(je, jp), escheme=(np.arcsinh(Pz / np.hypot(Px, Py)), np.arctan2(Py, Px)),
             hardest=from_stored(J['deta'][:, 0].astype(np.float64), J['dphi'][:, 0].astype(np.float64)), wta=from_stored(wd, wp))
    return A, dict(wta_seconds=twta, wta_is_hardest=float(np.mean((np.abs(wd - J['deta'][:, 0]) < 1e-7) & (np.abs(wp - J['dphi'][:, 0]) < 1e-7))))


# ------------------------------------------------------------------------------------------------ main
def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    PER = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    res = dict(per_class=PER, deltas=list(DELTAS), seed=SEED, wta_selftest=wta_selftest())
    log(f'WTA self-test: {res["wta_selftest"]}')
    S = prep(PER, 3000); N = len(S['y']); trf, trv, tr, te = split(N)
    log(f'sample A: {N} jets, rows 3000-{3000 + PER}; mass median {np.median(S["m0"]):.1f} GeV ({time.time() - t0:.0f} s)')
    S2 = prep(PER, 6000); N2 = len(S2['y']); sp2 = split(N2)
    log(f'sample B (fresh): {N2} jets, rows 6000-{6000 + PER}; mass median {np.median(S2["m0"]):.1f} GeV; '
        f'label counts equal: {bool(np.array_equal(np.bincount(S["y"], minlength=10), np.bincount(S2["y"], minlength=10)))} ({time.time() - t0:.0f} s)')
    EDITS = {d: phi_edit(S, d) for d in DELTAS}; PRED = {d: pred_of(S, Je) for d, Je in EDITS.items()}
    # padding and invariant-mass checks of the edits (P1's own checks, recomputed)
    res['edits'] = {}
    for d, Je in EDITS.items():
        res['edits'][str(d)] = dict(mask_equal=bool(np.array_equal(Je['mask'], S['Jb']['mask'])),
                                    padded_zero=bool(np.all(Je['deta'][Je['mask'] == 0] == 0) and np.all(Je['dphi'][Je['mask'] == 0] == 0)),
                                    other_branches_equal=bool(all(np.array_equal(Je[k], S['Jb'][k]) for k in ('charge', 'isChargedHadron', 'isNeutralHadron', 'isPhoton', 'isElectron', 'isMuon', 'd0val', 'd0err', 'dzval', 'dzerr', 'deta', 'pz', 'energy'))),
                                    dlogm_absmax=float(np.abs(np.log(np.maximum(jmass(Je), 1.0)) - S['logm0']).max()),
                                    pred_mean=float(PRED[d].mean()), pred_median=float(np.median(PRED[d])))
        log(f'edit φ {d:+.2f}: {res["edits"][str(d)]}')
    A, ainfo = axis_set(S); res['axes_info'] = ainfo
    log(f'axes: WTA clustering {ainfo["wta_seconds"]:.0f} s; WTA axis = hardest particle in {100 * ainfo["wta_is_hardest"]:.1f} % of jets')
    AX = {}
    for a, (ea, pa) in A.items():
        E4, flips = reaxis(S['Jb'], ea, pa); pt4 = kin(E4)[0]; M4 = moments(pt4, E4['deta'], E4['dphi'])
        disp = np.hypot(ea - S['Jb']['jet_eta'], (pa - S['Jb']['jet_phi'] + np.pi) % (2 * np.pi) - np.pi)     # axis displacement in (η, φ)
        r = M4['M1'] ** 2 / M4['M0M2']; pred = 0.5 * np.log(M4['M0M2'] / S['M']['M0M2'])
        info = dict(sign_flips=flips, disp_median=float(np.median(disp)), disp_p90=float(np.quantile(disp, 0.9)),
                    disp_median_W=float(np.median(disp[S['y'] == iW])), disp_median_QCD=float(np.median(disp[S['y'] == 0])),
                    M1sq_over_M0M2_median=float(np.median(r)), M1sq_over_M0M2_mean=float(r.mean()),
                    pred_mean=float(pred.mean()), pred_median=float(np.median(pred)), pred_W_mean=float(pred[S['y'] == iW].mean()),
                    vec_identical=bool(all(np.array_equal(E4[k], S['Jb'][k]) for k in ('px', 'py', 'pz', 'energy', 'mask', 'jet_pt', 'jet_energy'))),
                    max_abs_ddeta=float(np.abs(E4['deta'] - S['Jb']['deta']).max()), max_abs_ddphi=float(np.abs(E4['dphi'] - S['Jb']['dphi']).max()))
        AX[a] = (E4, pred); res.setdefault('axes', {})[a] = dict(info=info)
        log(f'axis {a}: {info}')
    res['models'] = {}
    P1 = {}                                       # P1's read-out per model (for the fresh sample)
    for n in ('full', 'kin'):
        net = ParTNetwork(n); t1 = time.time(); row = {}
        # ---- check 1 + baseline identity
        chk, (xb, vb, mb, xe6, ve6, me6, names) = feature_check(S['Jb'], EDITS[0.06], n); row['check1_features'] = chk
        log(f'ParT_{n}: check 1 (φ 0.06) mask equal {chk["mask_equal"]}, P {chk["P"]}; max |Δ| per fed feature (real slots / all slots): '
            + ', '.join(f'{k} {v["max_real"]:.2e}/{v["max_all_slots"]:.2e}' for k, v in chk['fed'].items()))
        log('   raw features, max |Δ| (mean Δ) on real slots: ' + ', '.join(f'{k} {v["max_real"]:.2e} ({v["mean_real"]:+.2e})' for k, v in chk['raw'].items()))
        log(f'   4-vectors max |Δ| [GeV]: {chk["vec"]}; jet_pt max rel {chk["jet_pt_max_rel"]:.1e}, jet_energy {chk["jet_energy_max_rel"]:.1e}')
        if n == 'full':
            pc = pair_check(vb, ve6, mb); res['check1_pairs'] = pc
            log(f'   pair features on {pc["pairs"]} real-real pairs: max |Δ| {pc["max_abs_change"]}; off-diagonal {pc["max_abs_change_offdiag"]}; ΔR > 0.01 {pc["max_abs_change_dR_gt_0p01"]}')
            log(f'   fraction |Δ| > 1e-3: {pc["frac_change_gt"]["0.001"]}; > 1e-2: {pc["frac_change_gt"]["0.01"]}; spread sd {pc["spread_sd"]}')
            log(f'   float32 vs float64 (baseline): max {pc["f32_vs_f64_max"]}; fraction > 1e-3 {pc["f32_vs_f64_frac_gt"]["0.001"]}')
        z0, L0 = fwd(net, xb, vb, mb); Rr = net.run(S['Jb'])
        row['run_vs_fwd'] = dict(max_dlogit=float(np.abs(Rr['logits'] - L0).max()), max_dz=float(np.abs(Rr['z'] - z0).max()))
        sd0 = z0.std(0); c0 = L0.argmax(1); y = S['y']
        ridge = Ridge(z0, S['logm0'], trf, trv, tr); P1[n] = ridge
        B = dict(y=y, te=te, z0=z0, L0=L0, c0=c0, sd0=sd0, ridge=ridge)
        row['baseline'] = dict(acc=float((c0 == y).mean()), W_to_Z=float((c0[y == iW] == iZ).mean()), Z_to_W=float((c0[y == iZ] == iW).mean()),
                               ridge_r2=r2(ridge(z0[te]), S['logm0'][te]), alpha=float(ridge.alpha), n98_sd=float(sd0[N98]))
        log(f'   network.run vs forward on precomputed inputs: {row["run_vs_fwd"]}; baseline {row["baseline"]}')
        # ---- check 2
        iR, iP = names.index('deltaR'), names.index('dphi'); row['check2'] = {}
        for d, Je in EDITS.items():
            if d == 0.06: xe, ve, me = xe6, ve6, me6
            else: xe, ve, me = inputs(Je, n)
            assert np.array_equal(me, mb) and xe.shape == xb.shape
            V = {}
            V['a_full_edit'] = (xe, ve)
            x = xe.copy(); x[:, iR] = xb[:, iR]; V['b_deltaR_unedited'] = (x, ve)
            x = xe.copy(); x[:, iP] = xb[:, iP]; V['c_dphi_unedited'] = (x, ve)
            x = xe.copy(); x[:, [iR, iP]] = xb[:, [iR, iP]]; V['d_both_unedited'] = (x, ve)
            dmax = float(np.abs(x - xb).max())
            V['f_vectors_unedited'] = (xe, vb)
            row['check2'][str(d)] = {'d_x_minus_xb_max': dmax}
            log(f'   check 2, φ {d:+.2f} (pred ½Δlog M0M2 on test half {PRED[d][te].mean():+.4f}); variant (d): max |x_d − x_baseline| over all features {dmax:.2e}')
            for k, (x, v) in V.items():
                z, L = fwd(net, x, v, mb); e = metrics(z, L, B, PRED[d]); row['check2'][str(d)][k] = e; log(fmt(k, e))
            del V, x
        # ---- check 4
        row['check4'] = {}
        for a, (E4, pred) in AX.items():
            x4, v4, m4 = inputs(E4, n); assert np.array_equal(m4, mb)
            other = [i for i, k in enumerate(names) if k not in ('deta', 'dphi', 'deltaR')]
            chg = {k: float(np.abs(x4[:, i] - xb[:, i]).max()) for i, k in enumerate(names)}
            z, L = fwd(net, x4, v4, m4); e = metrics(z, L, B, pred); e['fed_feature_max_change'] = chg
            e['other_features_max_change'] = float(max(chg[names[i]] for i in other)); e['vec_max_change'] = float(np.abs(v4 - vb).max())
            e['own_class'] = [float((L.argmax(1)[y == k] == k).mean()) for k in range(10)]
            row['check4'][a] = e; log(fmt(f'axis {a}', e) + f' | other features max Δ {e["other_features_max_change"]:.1e}, 4-vectors {e["vec_max_change"]:.1e}')
        # ---- check 3 (fresh sample)
        x2b, v2b, m2b = inputs(S2['Jb'], n); z20, L20 = fwd(net, x2b, v2b, m2b)
        J2e = phi_edit(S2, 0.06); p2 = pred_of(S2, J2e); x2e, v2e, m2e = inputs(J2e, n); z2, L2 = fwd(net, x2e, v2e, m2e)
        trf2, trv2, tr2, te2 = sp2; rid2 = Ridge(z20, S2['logm0'], trf2, trv2, tr2)
        B2 = dict(y=S2['y'], te=te2, z0=z20, L0=L20, c0=L20.argmax(1), sd0=z20.std(0), ridge=rid2)
        e = metrics(z2, L2, B2, p2); c20 = L20.argmax(1); y2 = S2['y']
        dlP1 = ridge(z2) - ridge(z20)
        e.update(baseline_acc=float((c20 == y2).mean()), baseline_W_to_Z=float((c20[y2 == iW] == iZ).mean()), baseline_Z_to_W=float((c20[y2 == iZ] == iW).mean()),
                 ridge_r2_refit=r2(rid2(z20[te2]), S2['logm0'][te2]), ridge_r2_P1readout_all=r2(ridge(z20), S2['logm0']),
                 dlogm_hat_P1readout_all=float(dlP1.mean()), pred_all=float(p2.mean()), ratio_P1readout_all=float(dlP1.mean() / p2.mean()))
        row['check3'] = e
        log(fmt('check 3: fresh rows, φ +0.06', e) + f' | baseline acc {e["baseline_acc"]:.3f} W→Z {e["baseline_W_to_Z"]:.3f} | R² refit {e["ridge_r2_refit"]:.3f}, '
            f'P1 read-out on fresh rows R² {e["ridge_r2_P1readout_all"]:.3f} Δlog m̂ {e["dlogm_hat_P1readout_all"]:+.4f} vs pred {e["pred_all"]:+.4f}')
        res['models'][n] = row; log(f'ParT_{n} done ({time.time() - t1:.0f} s)')
        del net
    json.dump(res, open(OUT / ('p1v_verify.json' if PER == 1000 else f'p1v_verify_{PER}.json'), 'w'), indent=1)
    log(f'P1v DONE ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
