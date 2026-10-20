"""H36 (loop D; claim D4 / D4m across datasets): axis convention on Top Landscape and quark-gluon taggers, predicted before
measuring from each model's response to rigid translations of its inputs.

Question. D4: recomputing the inputs about the winner-take-all (WTA) axis instead of the stored E-scheme axis changes 12-34 % of
the calls of hls4ml / JetClass taggers. Does each Top / QG model's call change under the WTA and hardest-particle (HP) axes follow
from its measured response to a rigid translation of the same size (H1), or is it larger (H0: the networks read the axis position
relative to the prongs more finely than a direction-averaged translation captures)?

Identity used. Every Top / QG input that depends on the axis is (deta_i, dphi_i) or a function of them (dR_i = hypot); pT, E, the
jet pT / energy and the pair features (four-vector based) do not depend on the axis. Recomputing about a new axis a is therefore an
exact rigid translation of (deta, dphi) by -(sign(eta_J) (eta_a - eta_J), phi_a - phi_J) (up to the phi wrap). TN (translation by
the same per-jet offset, applied to the stored deta/dphi arrays) is computed through a separate code path as an implementation
check (WTA - TN = 0 expected for every model). The test of H1 vs H0 is therefore: observed vs the prediction from the
direction-averaged translation response, and axis vs R (random direction, same per-jet size): the WTA / HP offsets point at the
hardest particle / prong, R does not.

Samples (random rows, never first-k)
  top  Top Landscape test split: interp.h4_top_wmass._sample(6000, 3601) (= data_top.sample rows); per class a random 1,000
       -> calibration, 5,000 -> main (rng 36011).
  py   QG Pythia test split: h14_qg_generator.load_sample('py', 6000, 3602); same split (rng 36021).
  hw   QG Herwig 7.1 test files 18-19: load_sample('hw', 6000, 3602); same split (rng 36022).
Axes per jet: E (stored = constituent four-momentum sum), WTA (C/A clustering of all constituents to one jet with WTA pT
  recombination: the merged direction is that of the harder branch, so the axis is one particle's direction), HP (the highest-pT
  constituent). Offset delta = (eta_a - eta_J, phi_a - phi_J), |delta| in rad.
Models (score l = log-odds; Top: top-positive; QG: gluon-positive, as H24)
  top: ParT_top, PN_top (H4 snapshots of the H2 weaver trainings), LNet_top (released LorentzNet, four-vectors only: axis cannot
       enter), PFN_top, EFN_top (H2 energyflow, numpy), ParT_top_perm (ParT_top with every weight tensor permuted, seed 0).
  py/hw (all models on both samples): part_kin, pnet_kin (Pythia-trained weaver), pfn_k, efn_k (H2 energyflow, ported),
       part_kin_perm (permuted null), hw_part_kin, hw_pnet_kin (Herwig-trained), lorentznet (released QG LorentzNet, 4-vectors +
       PID); classical LR_girth (histogram LR of girth about the axis in use) and LR_N (multiplicity; axis-free) trained on 20k
       jets per class of the sample's own generator train split (h14_qg_generator.efp_train_lists, seed 36031).
  Top classical reference LR(m_SD, tau32): m_SD (C/A soft drop) and tau32 (exclusive-kT axes) use no jet axis, so its calls do not
  change under any axis convention (0 % by construction; not recomputed).
Step 1 (calibration, the 2,000 calibration jets): rigid translations (deta, dphi) += r (cos t_k, sin t_k), r in RCAL (2-400 mrad;
  the plan's 2-20 mrad grid extended to 400 mrad because WTA offsets of top jets reach 0.1-0.3 rad), t_k = k pi / 4, k = 0..7;
  per model and class the distribution of dl = l(T) - l(0) at each r (8 directions x jets).
Step 2 (pre-registered prediction, written to physics/h36_prereg.json before step 3 is scored): for each main jet j with
  unedited score l0_j, class y_j and offset size r_j, p_j = P(sign(l0_j + dl) != sign(l0_j)) under the calibration distribution of
  dl for class y_j at r_j (empirical, linear interpolation in log r between grid sizes; r < 2 mrad: p scaled by r / 2 mrad; r >
  400 mrad: value at 400 mrad, fraction recorded). Predicted % calls changed = mean_j p_j; predicted mean |dl| likewise.
  Secondary prediction (pred2, also written before step 3; added after the smoke run): the same with the calibration dl
  distribution taken per class and per quartile of |l0| among the calibration jets (margin-conditioned).
  E2: the unedited main jets scored a second time (GPU non-determinism floor; LorentzNet logits differ by ~1e-4 between runs).
Step 3 (measure, main jets): E (unedited), WTA, HP (inputs recomputed about the axis), TN_WTA, TN_HP (stored arrays translated by
  the per-jet offset), R_WTA, R_HP (translation of the same per-jet size in a uniformly random direction, rng 36041).
Metrics per model x sample x axis: % calls changed (call = sign l; 95 % bootstrap over jets, B = 2000); R the same; McNemar on the
  paired changed indicators (axis vs R); accuracy and AUC before -> after; observed / predicted % changed with a 95 % interval
  from a joint bootstrap of main jets and calibration jets (BCAL = 500); two-sided bootstrap p for H1 (observed = predicted);
  mean |dl| observed vs predicted; max |l_axis - l_TN| (implementation check). Holm across all model x sample x axis cells for
  each family (McNemar vs R; H1 test).
Run: ./sync.sh && ./part_job.sh d-h36-run 'python -m interp.h36_axis_top_qg'   (H36_SMOKE=1: 150 main + 60 calibration per class)
Outputs: $AI_RESULTS/physics/h36_prereg.json, h36_axis_top_qg.json, h36_perjet.npz, h36_axis.png
"""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import h4_top_wmass as H4        # noqa: E402
from interp import h5_qg_counting as H5      # noqa: E402
from interp import h14_qg_generator as H14   # noqa: E402

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
SMOKE = bool(os.environ.get('H36_SMOKE'))
RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'physics'
TAG = 'h36_smoke' if SMOKE else 'h36'
PER_MAIN = 150 if SMOKE else 5000
PER_CAL = 60 if SMOKE else 1000
NTRAIN_LR = 2000 if SMOKE else 20000
B = 300 if SMOKE else 2000
BCAL = 60 if SMOKE else 500
RCAL = np.array([2, 5, 10, 20, 50, 100, 200, 400], float) * 1e-3
NDIR = 8
NCPU = int(os.environ.get('OMP_NUM_THREADS', '4'))
AXES = ('WTA', 'HP')
SEEDS = dict(top=3601, py=3602, hw=3602)
SPLIT = dict(top=36011, py=36021, hw=36022)
QG_NETS = ['part_kin', 'pnet_kin', 'pfn_k', 'efn_k', 'part_kin_perm', 'hw_part_kin', 'hw_pnet_kin', 'lorentznet']
TOP_NETS = ['ParT_top', 'PN_top', 'LNet_top', 'PFN_top', 'EFN_top', 'ParT_top_perm']
TRAINED = dict(part_kin='py', pnet_kin='py', pfn_k='py', efn_k='py', part_kin_perm='py (permuted)', hw_part_kin='hw',
               hw_pnet_kin='hw', lorentznet='py (released)', LR_girth='own', LR_N='own')


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ============================================================================================================== axes
def wta_index(arg):
    """C/A clustering to one jet with WTA pT recombination; returns the index of the particle whose direction is the axis"""
    eta, phi, pt = arg; n = len(pt)
    if n == 1: return 0
    D = (eta[:, None] - eta[None]) ** 2 + wrap(phi[:, None] - phi[None]) ** 2; np.fill_diagonal(D, np.inf)
    w = pt.astype(np.float64).copy(); i = 0
    for _ in range(n - 1):
        k = int(np.argmin(D)); i, j = divmod(k, n)
        if w[j] > w[i]: i, j = j, i
        w[i] += w[j]; D[j, :] = np.inf; D[:, j] = np.inf      # i keeps its direction, so its distances are unchanged
    return i


def axes_of(eta_l, phi_l, pt_l, jeta, jphi):
    """per-jet particle arrays -> dict axis -> (eta_a, phi_a), plus offsets"""
    import multiprocessing as mp
    with mp.get_context('fork').Pool(NCPU) as pool:
        iw = np.array(pool.map(wta_index, list(zip(eta_l, phi_l, pt_l)), chunksize=32))
    ih = np.array([int(np.argmax(p)) for p in pt_l])
    A = {'WTA': (np.array([e[i] for e, i in zip(eta_l, iw)]), np.array([p[i] for p, i in zip(phi_l, iw)])),
         'HP': (np.array([e[i] for e, i in zip(eta_l, ih)]), np.array([p[i] for p, i in zip(phi_l, ih)]))}
    off = {a: (A[a][0] - jeta, wrap(A[a][1] - jphi)) for a in A}
    return A, off, dict(wta_is_hardest=float(np.mean(iw == ih)))


# ============================================================================================================== Top
def top_J(P4, nn, cond):
    """Top Landscape inputs (h4_top_wmass.jet_inputs, E-scheme axis) with the axis replaced (cond ('ax', eta_a, phi_a)) or the
    stored (deta, dphi) translated (cond ('sh', tx, ty)); every other input unchanged"""
    J = H4.jet_inputs(P4, nn); m = J['mask']
    if cond is None: return J
    if cond[0] == 'ax':
        P = P4.astype(np.float64); ptc = np.where(m, J['pt'], 1.0)
        eta = np.arcsinh(np.where(m, P[..., 2], 0) / ptc); phi = np.arctan2(P[..., 1], P[..., 0])
        S = (P * m[..., None]).sum(1); jeta = np.arcsinh(S[:, 2] / np.hypot(S[:, 0], S[:, 1])); sg = np.where(jeta < 0, -1.0, 1.0)
        J['deta'] = np.where(m, (eta - cond[1][:, None]) * sg[:, None], 0); J['dphi'] = np.where(m, wrap(phi - cond[2][:, None]), 0)
    else:
        J['deta'] = np.where(m, J['deta'] + cond[1][:, None], 0); J['dphi'] = np.where(m, J['dphi'] + cond[2][:, None], 0)
    return J


class TopModels:
    def __init__(self):
        import torch
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'; self.torch = torch; self.dev = dev
        self.w = {'ParT_top': H4.WeaverNet('ParT', RES / 'h4_ckpt_part_kin.pt', dev), 'PN_top': H4.WeaverNet('PN', RES / 'h4_ckpt_pnet_kin.pt', dev)}
        perm = H4.WeaverNet('ParT', RES / 'h4_ckpt_part_kin.pt', 'cpu'); g = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for p in perm.model.parameters():
                f = p.data.view(-1); f.copy_(f[torch.randperm(f.numel(), generator=g)])
        perm.model.to(dev); perm.dev = dev; self.w['ParT_top_perm'] = perm
        self.lnet = H4.LNet(dev, mass64=True); self.flows = {'PFN_top': H4.Flow('pfn'), 'EFN_top': H4.Flow('efn')}
        log(f'Top models loaded on {dev}')

    def score(self, P4, nn, cond, order, chunk=2000):
        torch = self.torch; out = {k: np.zeros(len(nn)) for k in TOP_NETS}
        for s in range(0, len(nn), chunk):
            sl = slice(s, s + chunk); c = None if cond is None else (cond[0], cond[1][sl], cond[2][sl])
            J = top_J(P4[sl], nn[sl], c)
            for b in range(0, len(nn[sl]), 512):
                Jb = {k: (v[b:b + 512] if np.ndim(v) else v) for k, v in J.items()}; pts, x, v, m = H4.weaver_inputs(Jb)
                d = lambda a: torch.from_numpy(a).to(self.dev)
                with torch.no_grad():
                    for name, wn in self.w.items():
                        o = (wn.model(d(x), v=d(v), mask=d(m)) if wn.kind == 'ParT' else wn.model(d(pts), d(x), d(m))).float().cpu().numpy()
                        out[name][s + b:s + b + len(o)] = o[:, 0] - o[:, 1]      # class 0 = top
            k3 = np.stack([J['pt'][:, :128], J['deta'][:, :128], J['dphi'][:, :128]], -1).astype(np.float16).astype(np.float32)
            for name, f in self.flows.items(): out[name][sl] = f.score(k3)
        out['LNet_top'] = self.lnet.score(P4, nn, order)        # four-vectors only: no axis-dependent input exists
        return out


def load_top():
    S = H4._sample(PER_MAIN + PER_CAL, SEEDS['top']); H4._FULL.clear()
    nn = np.minimum(S['jet'][:, 5].astype(np.int64), S['full'].shape[1])
    P4 = H4._identity(S['full'][..., :4].astype(np.float32), nn).astype(np.float32); y = S['label'].astype(int)
    P = P4.astype(np.float64); pt = np.hypot(P[..., 0], P[..., 1]); real = np.arange(P.shape[1])[None, :] < nn[:, None]
    eta = np.arcsinh(np.where(real, P[..., 2], 0) / np.where(real, pt, 1)); phi = np.arctan2(P[..., 1], P[..., 0])
    Ssum = (P * real[..., None]).sum(1); jeta = np.arcsinh(Ssum[:, 2] / np.hypot(Ssum[:, 0], Ssum[:, 1])); jphi = np.arctan2(Ssum[:, 1], Ssum[:, 0])
    A, off, ai = axes_of([eta[j, :nn[j]] for j in range(len(nn))], [phi[j, :nn[j]] for j in range(len(nn))],
                         [pt[j, :nn[j]] for j in range(len(nn))], jeta, jphi)
    return dict(P4=P4, nn=nn, y=y, idx=S['idx'], jeta=jeta, jphi=jphi, A=A, off=off, axinfo=ai)


# ============================================================================================================== QG
_KIN0 = H5.kinematics


def _kin(Jt):
    """h5_qg_counting.kinematics with an optional axis replacement / translation (Jt['_cond']); used by weaver_inputs, ef_inputs"""
    K = _KIN0(Jt); c = Jt.get('_cond')
    if c is None: return K
    real = K['real']; sg = np.sign(K['jeta']); sg[sg == 0] = 1
    if c[0] == 'ax':
        K['deta'] = (Jt['eta'] - c[1][:, None]) * sg[:, None] * real; K['dphi'] = wrap(Jt['phi'] - c[2][:, None]) * real
    else:
        K['deta'] = (K['deta'] + c[1][:, None]) * real; K['dphi'] = (K['dphi'] + c[2][:, None]) * real
    return K


H5.kinematics = _kin


def girth_of(Jt):
    K = H5.kinematics(Jt); pt = Jt['pt'] * K['real']; z = pt / pt.sum(1, keepdims=True)
    return (z * np.hypot(K['deta'], K['dphi'])).sum(1)


def _lr_hist(x, y, bins):
    hq = np.histogram(x[y == 1], bins)[0] + 0.5; hg = np.histogram(x[y == 0], bins)[0] + 0.5
    return np.log(hg / hg.sum()) - np.log(hq / hq.sum())      # gluon-positive


class QGRefs:
    """LR(girth) (40 quantile bins) and LR(N) (integer bins) from the generator's train split"""
    def __init__(self, gen):
        Lt, yt, Nt = H14.efp_train_lists(gen, n_per_class=NTRAIN_LR, seed=36031)
        Jt = H5.from_list([np.c_[x[:, :3], np.zeros(len(x))] for x in Lt]); g = girth_of(Jt)
        bg = np.quantile(g, np.linspace(0, 1, 41)); bg[0], bg[-1] = 0, 10; self.bg = bg; self.tg = _lr_hist(g, yt, bg)
        bn = np.arange(-0.5, 301.5, 1.0); self.bn = bn; self.tn = _lr_hist(Nt, yt, bn)
        self.info = dict(n_train=int(len(yt)), seed=36031, gen=gen)
        log(f'refs {gen}: {len(yt)} train jets')

    def score(self, Jt):
        g = girth_of(Jt); N = (Jt['pt'] > 0).sum(1)
        return dict(LR_girth=self.tg[np.clip(np.searchsorted(self.bg, g, side='right') - 1, 0, 39)],
                    LR_N=self.tn[np.clip(N, 0, 300)])


def load_qg_models():
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; M = H5.Models(dev)
    for name, arch in (('part_kin', 'part'), ('pnet_kin', 'pn')): M.add_weaver(name, arch, False)
    M.add_permuted('part_kin_perm', 'part_kin')
    for kind in ('efn', 'pfn'): H14.add_keras_ef(M, kind)
    for name, arch in (('part_kin', 'part'), ('pnet_kin', 'pn')):
        H14.add_weaver_dir(M, 'hw_' + name, arch, False, H5.V / 'models' / 'qg_herwig' / name)
    M.add_lnet(); log('QG models', list(M.m), 'on', dev)
    return M


def qg_score(M, R, Jt, cond):
    Jt = dict(Jt); Jt['_cond'] = cond
    out = {nm: M.logit(nm, Jt) for nm in QG_NETS}; out.update(R.score(Jt))
    return out


def load_qg(gen):
    L, y, info = H14.load_sample(gen, PER_MAIN + PER_CAL, SEEDS[gen])
    Jt = H5.from_list(L, Lpad=180); assert Jt['n_full'].max() <= 180
    K = _KIN0(Jt)
    A, off, ai = axes_of([a[:, 1] for a in L], [a[:, 2] for a in L], [a[:, 0] for a in L], K['jeta'], K['jphi'])
    return dict(Jt=Jt, y=y.astype(int), idx=info['idx'], jeta=K['jeta'], jphi=K['jphi'], A=A, off=off, axinfo=ai)


def sub_jt(Jt, rows):
    return {k: (v[rows] if np.ndim(v) else v) for k, v in Jt.items()}


# ============================================================================================================== prediction
def pgrid(DL, ycal, l0, y, rows_cal=None, l0cal=None, nbin=1):
    """per main jet x grid size: P(call flips) and E|dl| under the calibration dl distribution of the jet's class (nbin = 1:
    primary prediction) or of its class and |l0| quantile bin among the calibration jets of that class (nbin = 4: secondary,
    margin-conditioned prediction). DL: (ncal, S, D) calibration dl; ycal labels; rows_cal: bootstrap rows (None = all)"""
    S = DL.shape[1]; Pg = np.zeros((len(l0), S)); Ag = np.zeros((len(l0), S))
    for k in (0, 1):
        rc = np.flatnonzero(ycal == k) if rows_cal is None else rows_cal[ycal[rows_cal] == k]
        if nbin > 1:
            ed = np.quantile(np.abs(l0cal[rc]), np.linspace(0, 1, nbin + 1)[1:-1])
            bc = np.searchsorted(ed, np.abs(l0cal[rc])); bm = np.searchsorted(ed, np.abs(l0))
        else:
            bc = np.zeros(len(rc), int); bm = np.zeros(len(l0), int)
        for q in range(nbin):
            rq = rc[bc == q]; rm = (y == k) & (bm == q); x = -l0[rm]
            if len(rq) == 0 or not rm.any(): continue
            for si in range(S):
                v = np.sort(DL[rq, si, :].ravel()); n = len(v)
                lt = np.searchsorted(v, x, 'left') / n; gt = 1 - np.searchsorted(v, x, 'right') / n
                Pg[rm, si] = np.where(l0[rm] > 0, lt, gt); Ag[rm, si] = np.abs(v).mean()
    return Pg, Ag


def interp_r(G, r):
    lr = np.log(RCAL); x = np.log(np.clip(r, RCAL[0], RCAL[-1])); i = np.clip(np.searchsorted(lr, x) - 1, 0, len(RCAL) - 2)
    t = (x - lr[i]) / (lr[i + 1] - lr[i]); a = np.arange(len(r))
    out = G[a, i] * (1 - t) + G[a, i + 1] * t
    small = r < RCAL[0]; out[small] = G[small, 0] * r[small] / RCAL[0]
    return out


# ============================================================================================================== stats
def auc(s, y):
    return float(H4._auc(s, y))


def run_dataset(ds, S, scorer, names, sign_top):
    """calibration, pre-registered prediction, measurement for one sample; returns (result dict, per-jet arrays, prereg dict)"""
    y_all = S['y']; rng = np.random.default_rng(SPLIT[ds])
    cal = np.sort(np.concatenate([rng.permutation(np.flatnonzero(y_all == c))[:PER_CAL] for c in (0, 1)]))
    main = np.setdiff1d(np.arange(len(y_all)), cal)
    yc, ym = y_all[cal], y_all[main]
    # step 1: calibration translations
    log(f'[{ds}] step 1: calibration, {len(cal)} jets x {len(RCAL)} sizes x {NDIR} directions')
    sc_cal0 = scorer(cal, None)
    DL = {nm: np.zeros((len(cal), len(RCAL), NDIR)) for nm in names}
    for si, r in enumerate(RCAL):
        for k in range(NDIR):
            th = k * np.pi / 4; tx = np.full(len(cal), r * np.cos(th)); ty = np.full(len(cal), r * np.sin(th))
            s = scorer(cal, ('sh', tx, ty))
            for nm in names: DL[nm][:, si, k] = s[nm] - sc_cal0[nm]
        log(f'[{ds}]   r = {r * 1e3:.0f} mrad: % calls changed ' + ', '.join(
            f'{nm} {100 * np.mean(np.sign(sc_cal0[nm][:, None] + DL[nm][:, si, :]) != np.sign(sc_cal0[nm][:, None])):.2f}' for nm in names))
    calib = {nm: dict(pct_changed=[float(100 * np.mean(np.sign(sc_cal0[nm][:, None] + DL[nm][:, si, :]) != np.sign(sc_cal0[nm][:, None])))
                                   for si in range(len(RCAL))],
                      mean_abs_dl=[float(np.abs(DL[nm][:, si, :]).mean()) for si in range(len(RCAL))],
                      mean_dl_by_class={str(c): [float(DL[nm][yc == c, si, :].mean()) for si in range(len(RCAL))] for c in (0, 1)},
                      slope_abs_dl_per_mrad_at_20=float(np.abs(DL[nm][:, 3, :]).mean() / 20.0))
             for nm in names}
    # step 2: unedited main scores and pre-registered prediction
    sc = {'E': scorer(main, None)}
    off = {a: (S['off'][a][0][main], S['off'][a][1][main]) for a in AXES}
    rr = {a: np.hypot(*off[a]) for a in AXES}
    pre = dict(dataset=ds, written_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), n_main=int(len(main)), n_cal=int(len(cal)),
               offsets={a: dict(median_mrad=float(1e3 * np.median(rr[a])), p90_mrad=float(1e3 * np.quantile(rr[a], 0.9)),
                                frac_above_grid=float(np.mean(rr[a] > RCAL[-1])), frac_below_grid=float(np.mean(rr[a] < RCAL[0])),
                                by_class={str(c): float(1e3 * np.median(rr[a][ym == c])) for c in (0, 1)}) for a in AXES},
               models={})
    P, P2 = {}, {}
    for nm in names:
        Pg, Ag = pgrid(DL[nm], yc, sc['E'][nm], ym); P[nm] = (Pg, Ag)
        P2[nm] = pgrid(DL[nm], yc, sc['E'][nm], ym, l0cal=sc_cal0[nm], nbin=4)
        pre['models'][nm] = {a: dict(pred_pct_changed=float(100 * interp_r(Pg, rr[a]).mean()),
                                     pred_mean_abs_dl=float(interp_r(Ag, rr[a]).mean()),
                                     pred2_pct_changed_margin_conditioned=float(100 * interp_r(P2[nm][0], rr[a]).mean())) for a in AXES}
    log(f'[{ds}] step 2 prediction: ' + '; '.join(f'{nm} WTA {pre["models"][nm]["WTA"]["pred_pct_changed"]:.2f} % HP '
                                                    f'{pre["models"][nm]["HP"]["pred_pct_changed"]:.2f} %' for nm in names))
    # step 3: measurement
    sc['E2'] = scorer(main, None)      # repeat of the unedited scoring: GPU non-determinism floor
    sg = np.where(S['jeta'][main] < 0, -1.0, 1.0); rngR = np.random.default_rng(36041 + list(SEEDS).index(ds))
    for a in AXES:
        sc[a] = scorer(main, ('ax', S['A'][a][0][main], S['A'][a][1][main]))
        sc[f'TN_{a}'] = scorer(main, ('sh', -sg * off[a][0], -off[a][1]))
        th = rngR.uniform(0, 2 * np.pi, len(main))
        sc[f'R_{a}'] = scorer(main, ('sh', rr[a] * np.cos(th), rr[a] * np.sin(th)))
        log(f'[{ds}] step 3 measured {a}, TN_{a}, R_{a}')
    # statistics
    out = dict(n_main=int(len(main)), n_cal=int(len(cal)), axinfo=S['axinfo'], prediction=pre, calibration=calib, models={})
    rngb = np.random.default_rng(36051); J = len(main); bi = rngb.integers(0, J, (B, J))
    bcal = [np.concatenate([rngb.choice(np.flatnonzero(yc == c), (yc == c).sum()) for c in (0, 1)]) for _ in range(BCAL)]
    bmain = rngb.integers(0, J, (BCAL, J))
    for nm in names:
        l0 = sc['E'][nm]; c0 = l0 > 0; ytgt = ym == 1 if sign_top else ym == 0      # positive score = class 1 (top) / gluon (QG y=0)
        sc_or = lambda l: l if sign_top else -l
        yq = (ym == 1).astype(int)
        Mm = dict(trained_on=TRAINED.get(nm, ds), unedited=dict(acc=float(np.mean(c0 == ytgt)), auc=auc(sc_or(l0), yq)), axes={})
        Pg, Ag = P[nm]
        # calibration bootstrap: prediction grids recomputed per resample of calibration jets (Pg depends on main l0 only via rows)
        Pgb = [pgrid(DL[nm], yc, l0, ym, rows_cal=bcal[b]) for b in range(BCAL)]
        Pgb2 = [pgrid(DL[nm], yc, l0, ym, rows_cal=bcal[b], l0cal=sc_cal0[nm], nbin=4) for b in range(BCAL)]
        Mm['repeat_max_abs_dl'] = float(np.max(np.abs(sc['E2'][nm] - l0))); Mm['repeat_pct_changed'] = float(100 * np.mean((sc['E2'][nm] > 0) != c0))
        for a in AXES:
            l1 = sc[a][nm]; ch = (l1 > 0) != c0; chR = (sc[f'R_{a}'][nm] > 0) != c0; chT = (sc[f'TN_{a}'][nm] > 0) != c0
            p = interp_r(Pg, rr[a]); pa = interp_r(Ag, rr[a]); dl = np.abs(l1 - l0)
            m_ = lambda x: np.percentile(x[bi].mean(1) * 100, [2.5, 97.5]).tolist()
            obs_b = np.array([ch[bmain[b]].mean() for b in range(BCAL)])
            pred_b = np.array([interp_r(Pgb[b][0], rr[a])[bmain[b]].mean() for b in range(BCAL)])
            diff_b = obs_b - pred_b
            p2 = interp_r(P2[nm][0], rr[a]); pred2_b = np.array([interp_r(Pgb2[b][0], rr[a])[bmain[b]].mean() for b in range(BCAL)])
            d2 = obs_b - pred2_b
            ratio = float(ch.mean() / p.mean()) if p.mean() > 0 else None
            ratio_ci = np.percentile(obs_b / np.where(pred_b > 0, pred_b, np.nan), [2.5, 97.5]).tolist() if p.mean() > 0 else None
            p_h1 = float(min(1.0, 2 * min(np.mean(diff_b <= 0), np.mean(diff_b >= 0)))) if p.mean() > 0 or ch.any() else 1.0
            mc = H4._mcnemar(chR, ch)
            Mm['axes'][a] = dict(
                pct_changed=float(100 * ch.mean()), ci95=m_(ch), pct_changed_R=float(100 * chR.mean()), ci95_R=m_(chR),
                pct_changed_TN=float(100 * chT.mean()), max_abs_l_axis_minus_TN=float(np.max(np.abs(l1 - sc[f'TN_{a}'][nm]))),
                pct_changed_by_class={str(c): float(100 * ch[ym == c].mean()) for c in (0, 1)},
                pct_changed_R_by_class={str(c): float(100 * chR[ym == c].mean()) for c in (0, 1)},
                mcnemar_axis_vs_R=mc,
                pred_pct_changed=float(100 * p.mean()), pred_ci95_calibration=np.percentile(100 * pred_b, [2.5, 97.5]).tolist(),
                obs_over_pred=ratio, obs_over_pred_ci95=ratio_ci, p_H1_obs_eq_pred=p_h1,
                pred2_pct_changed=float(100 * p2.mean()), pred2_ci95_calibration=np.percentile(100 * pred2_b, [2.5, 97.5]).tolist(),
                obs_over_pred2=float(ch.mean() / p2.mean()) if p2.mean() > 0 else None,
                obs_over_pred2_ci95=np.percentile(obs_b / np.where(pred2_b > 0, pred2_b, np.nan), [2.5, 97.5]).tolist() if p2.mean() > 0 else None,
                p_H1b_obs_eq_pred2=float(min(1.0, 2 * min(np.mean(d2 <= 0), np.mean(d2 >= 0)))) if p2.mean() > 0 or ch.any() else 1.0,
                obs_minus_pred_pp=float(100 * (ch.mean() - p.mean())), obs_minus_pred_ci95=np.percentile(100 * diff_b, [2.5, 97.5]).tolist(),
                mean_abs_dl=float(dl.mean()), mean_abs_dl_ci95=np.percentile(dl[bi].mean(1), [2.5, 97.5]).tolist(),
                mean_abs_dl_R=float(np.abs(sc[f'R_{a}'][nm] - l0).mean()), pred_mean_abs_dl=float(pa.mean()),
                mean_dl=float((l1 - l0).mean()), mean_dl_R=float((sc[f'R_{a}'][nm] - l0).mean()),
                acc_after=float(np.mean((l1 > 0) == ytgt)), auc_after=auc(sc_or(l1), yq),
                acc_R=float(np.mean((sc[f'R_{a}'][nm] > 0) == ytgt)),
                by_offset={f'{lo_*1e3:.0f}-{hi_*1e3:.0f}mrad': dict(n=int(sel.sum()), obs=float(100 * ch[sel].mean()), pred=float(100 * p[sel].mean()),
                                                                   R=float(100 * chR[sel].mean()))
                           for lo_, hi_ in ((0, .01), (.01, .03), (.03, .1), (.1, .3), (.3, 10)) for sel in [(rr[a] >= lo_) & (rr[a] < hi_)] if sel.sum() >= 20})
        out['models'][nm] = Mm
        log(f'[{ds}] {nm:14s} acc {Mm["unedited"]["acc"]:.4f} AUC {Mm["unedited"]["auc"]:.4f} | ' + ' | '.join(
            f'{a}: obs {v["pct_changed"]:.2f} {np.round(v["ci95"], 2)} pred {v["pred_pct_changed"]:.2f} pred2 {v["pred2_pct_changed"]:.2f} R {v["pct_changed_R"]:.2f} '
            f'TN {v["pct_changed_TN"]:.2f} (max|l-lTN| {v["max_abs_l_axis_minus_TN"]:.1e}) o/p {v["obs_over_pred"]} McN p {v["mcnemar_axis_vs_R"]["p"]:.2g} '
            f'acc {v["acc_after"]:.4f} AUC {v["auc_after"]:.4f}' for a, v in Mm['axes'].items()))
    pj = {f'{ds}|y': ym, f'{ds}|idx': S['idx'][main]}
    for a in AXES: pj[f'{ds}|r_{a}'] = rr[a].astype(np.float32)
    for k, d in sc.items():
        for nm, v in d.items(): pj[f'{ds}|{k}|{nm}'] = np.asarray(v, np.float32)
    for nm in names: pj[f'{ds}|cal_dl|{nm}'] = DL[nm].astype(np.float32)
    pj[f'{ds}|cal_y'] = yc
    return out, pj, pre


def main():
    RES.mkdir(parents=True, exist_ok=True)
    log(f'H36 start; smoke={SMOKE}; main {PER_MAIN}/class, calibration {PER_CAL}/class, r grid {RCAL.tolist()}')
    R = dict(node='H36', smoke=SMOKE, rcal=RCAL.tolist(), ndir=NDIR, datasets={}); PJ = {}; PRE = {}
    prereg_path = RES / f'{TAG}_prereg.json'
    # Top Landscape
    S = load_top(); log(f'top: {len(S["y"])} jets ({S["y"].sum()} top); axes {S["axinfo"]}')
    TM = TopModels(); ordr = {}

    def top_scorer(rows, cond):
        key = len(rows)
        if key not in ordr: ordr[key] = np.random.default_rng(605).permutation(key)
        return TM.score(S['P4'][rows], S['nn'][rows], cond, ordr[key])
    R['datasets']['top'], pj, PRE['top'] = run_dataset('top', S, top_scorer, TOP_NETS, sign_top=True); PJ.update(pj)
    prereg_path.write_text(json.dumps(PRE, indent=1))
    R['datasets']['top']['classical'] = 'LR(m_SD, tau32): no jet-axis input (C/A soft drop, exclusive-kT tau axes): 0 calls change by construction'
    del TM, S
    import torch
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    # quark-gluon
    QS = {}
    for gen in ('py', 'hw'):   # samples and axes (multiprocessing) before the QG models are put on the GPU
        QS[gen] = load_qg(gen); log(f'{gen}: {len(QS[gen]["y"])} jets (quark frac {QS[gen]["y"].mean():.3f}); axes {QS[gen]["axinfo"]}')
    M = load_qg_models(); R['qg_models'] = M.info
    for gen in ('py', 'hw'):
        S = QS[gen]; REF = QGRefs(gen)

        def qg_scorer(rows, cond, S=S, REF=REF):
            return qg_score(M, REF, sub_jt(S['Jt'], rows), cond)
        R['datasets'][gen], pj, PRE[gen] = run_dataset(gen, S, qg_scorer, QG_NETS + ['LR_girth', 'LR_N'], sign_top=False); PJ.update(pj)
        R['datasets'][gen]['refs'] = REF.info
        prereg_path.write_text(json.dumps(PRE, indent=1))
    # Holm across model x sample x axis
    cells = [(ds, nm, a) for ds in R['datasets'] for nm in R['datasets'][ds]['models'] for a in AXES]
    pk = dict(mcnemar=lambda v: v['mcnemar_axis_vs_R']['p'], h1=lambda v: v['p_H1_obs_eq_pred'], h1b=lambda v: v['p_H1b_obs_eq_pred2'])
    for key, hk in (('mcnemar', 'p_holm_mcnemar_vs_R'), ('h1', 'p_holm_H1'), ('h1b', 'p_holm_H1b')):
        ps = [pk[key](R['datasets'][ds]['models'][nm]['axes'][a]) for ds, nm, a in cells]
        for (ds, nm, a), pa in zip(cells, H4._holm(ps)): R['datasets'][ds]['models'][nm]['axes'][a][hk] = float(pa)
    R['holm_family'] = f'{len(cells)} cells (models x samples x axes) per family: McNemar axis vs R; bootstrap H1 obs = pred'
    (RES / f'{TAG}_axis_top_qg.json').write_text(json.dumps(R, indent=1, default=float)); log(f'wrote {TAG}_axis_top_qg.json')
    np.savez_compressed(RES / f'{TAG}_perjet.npz', **PJ)
    try: fig(R)
    except Exception as ex: log('figure failed:', repr(ex))


LAB = {'ParT_top': 'ParT', 'PN_top': 'ParticleNet', 'LNet_top': 'LorentzNet', 'PFN_top': 'PFN', 'EFN_top': 'EFN', 'ParT_top_perm': 'ParT, permuted',
       'part_kin': 'ParT (Py)', 'pnet_kin': 'PN (Py)', 'pfn_k': 'PFN (Py)', 'efn_k': 'EFN (Py)', 'part_kin_perm': 'ParT permuted',
       'hw_part_kin': 'ParT (Hw)', 'hw_pnet_kin': 'PN (Hw)', 'lorentznet': 'LorentzNet (Py)', 'LR_girth': 'LR(girth)', 'LR_N': 'LR(N)'}


def fig(R):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dsn = {'top': 'Top Landscape (t vs QCD)', 'py': 'Quark-gluon, Pythia jets', 'hw': 'Quark-gluon, Herwig jets'}
    f, axs = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, ds in zip(axs, ('top', 'py', 'hw')):
        D = R['datasets'].get(ds)
        if D is None: continue
        mx = 0.5
        for a, mk in (('WTA', 'o'), ('HP', 's')):
            for i, (nm, Mm) in enumerate(D['models'].items()):
                v = Mm['axes'][a]; col = plt.cm.tab10(i % 10)
                ax.errorbar(v['pred_pct_changed'], v['pct_changed'], yerr=[[v['pct_changed'] - v['ci95'][0]], [v['ci95'][1] - v['pct_changed']]],
                            xerr=[[max(0, v['pred_pct_changed'] - v['pred_ci95_calibration'][0])], [max(0, v['pred_ci95_calibration'][1] - v['pred_pct_changed'])]],
                            fmt=mk, color=col, ms=6, capsize=2, label=LAB.get(nm, nm) if a == 'WTA' else None)
                ax.plot(v['pred_pct_changed'], v['pct_changed_R'], marker=mk, mfc='none', color=col, ms=6, ls='none')
                mx = max(mx, v['pct_changed'], v['pred_pct_changed'], v['pct_changed_R'])
        ax.plot([0, mx * 1.1], [0, mx * 1.1], color='0.6', lw=0.8)
        ax.set_xlim(-0.02 * mx, mx * 1.1); ax.set_ylim(-0.02 * mx, mx * 1.1)
        o = D['prediction']['offsets']
        ax.set_title(f'{dsn[ds]}\nmedian axis offset: WTA {o["WTA"]["median_mrad"]:.0f} mrad, hardest {o["HP"]["median_mrad"]:.0f} mrad', fontsize=9, loc='left')
        ax.set_xlabel('predicted % calls changed (from translation response)'); ax.legend(fontsize=7, loc='upper left')
    axs[0].set_ylabel('measured % calls changed\nfilled: new axis (o WTA, s hardest); open: random-direction shift')
    f.suptitle('Calls changed when inputs are recomputed about the WTA / hardest-particle axis, against the prediction from each model\'s translation response', fontsize=10)
    f.tight_layout(); f.savefig(RES / f'{TAG}_axis.png', dpi=130); log(f'wrote {TAG}_axis.png')


if __name__ == '__main__':
    if '--fig' in sys.argv: fig(json.loads((RES / f'{TAG}_axis_top_qg.json').read_text()))
    else: main()
