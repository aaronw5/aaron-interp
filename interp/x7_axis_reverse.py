"""X7 (claim D4 / A12, review weakness 2): is the call change under a change of axis convention set by each model's
axis-relative mass formula, or a generic covariate-shift response?

Models and inputs
  small  (conda env, torch): {dsets-d64, linf-2b-d32-k8}-n64 (trained on stored-axis inputs) and -axhard (trained on inputs
         about the hardest particle), seeds 0, 1 (/j-jepa-vol/aaron-interp/models/small/<run>/seed<s>/best.pt). Rows: the val
         archive, X4's sample_rows (rng 71, 4,000 per class = 20,000 jets); inputs exactly as X4 / training (pt_sort, then
         axis 'hardest' via train_small.recentre_hardest (torch float32; a numpy re-implementation is checked against it),
         (x - shift) / scale from arch.json, padded rows 0).
  jedi   (jax env): JEDI-linear 3-feature n64 on interp.jedi_core.sample(4000, seed=107) (20,000 jets), and two invariant
         controls on the same jets:
           A  JEDI-linear applied after recentring every jet on the pT centroid of its 64 particles (the composed function is
              translation invariant: any axis change gives the same input up to float rounding);
           B  L1 logistic regression (saga, multinomial, standardised features, C from {0.01, 0.1, 1} on a held-out third) on
              the 32 B4 EFP features (pairwise dR, beta 1, 2, multigraphs with <= 3 edges, plus logs), fitted to truth labels
              on sample(6000, seed 0) minus every row of the evaluation sample.
Axis conventions (per jet, in the stored coordinates): stored (0, 0); hardest = the highest-pT particle; wta = C/A clustering
  with winner-take-all recombination (interp.e17_axis_jedi.wta_axis_one). New offsets are stored offsets minus the axis,
  dphi wrapped to [-pi, pi); pT and padding unchanged.
Edits
  forward  models trained on stored-axis inputs, evaluated on inputs about the hardest particle and the WTA axis
           (for JEDI-linear: repeats E17 on these rows).
  reverse  axhard models evaluated on stored-axis inputs and on WTA inputs.
  control  matched rotation: the model's natural inputs rotated about the model's training axis (the origin of its input
           coordinates) by a per-jet angle theta = 2 arcsin(dbar / (2 rbar)) (random sign, rng 73; clipped at pi), where dbar is
           the mean over real particles of the per-particle input displacement of the axis edit and rbar the mean particle
           distance from the axis. Mean per-particle displacement equals that of the axis edit; M0 and M2 about the axis are
           unchanged.
Per (model, edit): accuracy before -> after (bootstrap CI of the difference, McNemar on correctness), % calls changed, W -> Z,
  Z -> W, W -> QCD (g or q), Z -> QCD with McNemar, the W <-> Z direction of discordant calls (binomial test), the 5 x 5 class
  flow; prediction per jet pred = 1/2 dlog(M0 M2) about the model's training axis (M0 fixed, so 1/2 dlog M2); per-jet
  logit-margin change dm = d(L_Z - L_W) on true W and Z jets, Spearman / Pearson correlation with pred (bootstrap CI over jets,
  asymptotic and permutation p); read-out: ridge (E6v / X4 Ridge) from the last hidden layer (Deep Sets h3, Linformer pool,
  JEDI z) to 1/2 log(M0 M2) about the training axis, fitted on the fit half of the natural jets (rng(71) / rng(107)
  permutation), changes evaluated on the other half; calibration edit: the natural jet's offsets about the training axis
  scaled by exp(pred) per jet (a real change of 1/2 log(M0 M2) of exactly pred); response ratio = mean read-out change / mean
  calibration change.  Paired bootstrap (over test-half jets) of forward minus reverse response ratio per architecture and seed.
Bootstrap: 1,000 resamples over jets (2,000 for ratios).
  AI_PY=conda ./job.sh x7-small 'python -m interp.x7_axis_reverse --mode small'     -> $AI_RESULTS/jedi/x7_small.json/.npz
  ./job.sh x7-jedi 'python -m interp.x7_axis_reverse --mode jedi'                    -> $AI_RESULTS/jedi/x7_jedi.json/.npz
  ./job.sh x7-comb 'python -m interp.x7_axis_reverse --mode combine'                 -> x7_summary.json, x7_axis_reverse.png
  (--smoke: 1/20 of the rows, outputs *_smoke)
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
from interp.x2_small_models import Ridge, mcnemar, wilson, ci95, load_net, DSRunner, TFRunner, run  # noqa: E402
from interp.x4_axis_models import (archive, rows, sample_rows, pt_sort, to_inputs, scale_about_axis, axis_moment,  # noqa: E402
                                   run_name, SMALL)
from interp.e17_axis_jedi import wta_axis_one, wta_selftest, reaxis  # noqa: E402

RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results'))
OUT = RES / 'jedi'
CL = ['g', 'q', 'W', 'Z', 't']; W_, Z_ = 2, 3; QCD = (0, 1)
MODELS = ('dsets-d64', 'linf-2b-d32-k8'); SEEDS = (0, 1)
SHORT = {'dsets-d64': 'Deep Sets', 'linf-2b-d32-k8': 'Linformer'}
B_BOOT = 1000; B_RATIO = 2000
SEED_JEDI = 107


def log(*a):
    print(time.strftime('[%H:%M:%S]'), *a, flush=True)


# ================================================================================================ geometry
def wta_axes(x):
    pt = x[..., 0].astype(np.float64); de = x[..., 1].astype(np.float64); dp = x[..., 2].astype(np.float64); real = pt > 0
    a = np.array([wta_axis_one(pt[j, real[j]], de[j, real[j]], dp[j, real[j]]) for j in range(len(x))])
    return a[:, 0], a[:, 1]


def centroid(x):
    pt = x[..., 0].astype(np.float64); M0 = np.maximum(pt.sum(1), 1e-12)
    return (pt * x[..., 1]).sum(1) / M0, (pt * x[..., 2]).sum(1) / M0


def recentre_hardest_np(x):
    """numpy float32 re-implementation of train_small.recentre_hardest (offsets about slot 0, dphi wrapped, padding 0)"""
    x = np.asarray(x, np.float32); d = x[..., 1:3] - x[:, :1, 1:3]
    dp = np.remainder(d[..., 1] + np.float32(np.pi), np.float32(2 * np.pi)) - np.float32(np.pi)
    d = np.stack([d[..., 0], dp], -1) * (x[..., :1] > 0).astype(np.float32)
    return np.concatenate([x[..., :1], d], -1).astype(np.float32)


def displacement(xn, xe):
    """per jet: mean over real particles of |input offset change| and mean distance of the real particles from the axis"""
    real = xn[..., 0] > 0; n = np.maximum(real.sum(1), 1)
    de = xe[..., 1].astype(np.float64) - xn[..., 1]
    dp = np.remainder(xe[..., 2].astype(np.float64) - xn[..., 2] + np.pi, 2 * np.pi) - np.pi
    d = np.where(real, np.hypot(de, dp), 0).sum(1) / n
    r = np.where(real, np.hypot(xn[..., 1].astype(np.float64), xn[..., 2].astype(np.float64)), 0).sum(1) / n
    return d, r


def matched_rotation(xn, xe, seed=73):
    """rotation of the real particles of xn about the origin by a per-jet angle whose mean per-particle displacement equals
    that of the edit xn -> xe; returns the rotated inputs and a check dict"""
    d, r = displacement(xn, xe)
    q = np.where(r > 0, d / np.maximum(2 * r, 1e-12), 0.0); clip = q > 1
    th = 2 * np.arcsin(np.minimum(q, 1.0)) * np.random.default_rng(seed).choice([-1.0, 1.0], len(xn))
    xr = rotate_jets(xn, th)
    d2, _ = displacement(xn, xr)
    am0, am1 = axis_moment(xn), axis_moment(xr)
    chk = dict(theta_median=float(np.median(np.abs(th))), theta_p90=float(np.quantile(np.abs(th), 0.9)),
               clip_frac=float(clip.mean()), disp_edit_mean=float(d.mean()), disp_rot_mean=float(d2.mean()),
               disp_ratio_median=float(np.median(d2[d > 0] / d[d > 0])) if (d > 0).any() else None,
               dM2_rot_maxabs=float(np.abs(am1 - am0).max()))
    return xr, chk


def rotate_jets(x, th):
    th = np.asarray(th, np.float64)[:, None]; real = x[..., 0] > 0; e = x.copy()
    de, dp = x[..., 1].astype(np.float64), x[..., 2].astype(np.float64)
    e[..., 1] = np.where(real, np.cos(th) * de - np.sin(th) * dp, x[..., 1]).astype(np.float32)
    e[..., 2] = np.where(real, np.sin(th) * de + np.cos(th) * dp, x[..., 2]).astype(np.float32)
    return e


def scale_jets(x, k):
    """real particles' offsets about the origin multiplied by k (per jet)"""
    k = np.asarray(k, np.float64)[:, None]; real = x[..., 0] > 0; e = x.copy()
    for f in (1, 2):
        e[..., f] = np.where(real, k * x[..., f], x[..., f]).astype(np.float32)
    return e


# ================================================================================================ statistics
def boot(n, seed, B=B_BOOT):
    return np.random.default_rng(seed).integers(0, n, size=(B, n)).astype(np.int32)


def bmeans(a, bi):
    a = np.asarray(a, np.float64)
    return np.concatenate([a[bi[i:i + 100]].mean(1) for i in range(0, len(bi), 100)])


def bstat(a, bi):
    b = bmeans(a, bi); m = float(np.mean(a))
    return dict(mean=m, ci=ci95(b), p_boot=float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()))))


def spearman_boot(xv, yv, seed, B=B_BOOT, perm=1000):
    from scipy.stats import spearmanr, pearsonr, rankdata
    if np.std(xv) < 1e-9 or np.std(yv) < 1e-12:
        return None
    rs, ps = spearmanr(xv, yv); rp, pp = pearsonr(xv, yv)
    rng = np.random.default_rng(seed); n = len(xv); bs = []
    for _ in range(B):
        ii = rng.integers(0, n, n); a, b = rankdata(xv[ii]), rankdata(yv[ii])
        bs.append(np.corrcoef(a, b)[0, 1])
    rx, ry = rankdata(xv), rankdata(yv); pv = []
    for _ in range(perm):
        pv.append(np.corrcoef(rx, rng.permutation(ry))[0, 1])
    pv = np.abs(np.array(pv))
    return dict(n=int(n), spearman=float(rs), ci=ci95(np.array(bs)), p=float(ps), p_perm=float((1 + (pv >= abs(rs)).sum()) / (1 + perm)),
                pearson=float(rp), p_pearson=float(pp))


def call_metrics(L0, L1, y, pred, seed):
    """call changes between logits L0 (natural) and L1 (edited); pred = per-jet 1/2 dlog(M0 M2) about the training axis"""
    from scipy.stats import binomtest
    c0, c1 = L0.argmax(1), L1.argmax(1); J = len(y); bi = boot(J, seed)
    ok0, ok1 = c0 == y, c1 == y; ch = c1 != c0
    out = dict(n=int(J), acc_before=float(ok0.mean()), acc_after=float(ok1.mean()),
               acc_diff=bstat(ok1.astype(float) - ok0, bi), changed=bstat(ch.astype(float), bi),
               mcnemar_correct=mcnemar(ok0, ok1),
               changed_class=[float(ch[y == k].mean()) for k in range(5)])
    tr = {}
    for nm, tru, to in (('W_to_Z', W_, (Z_,)), ('Z_to_W', Z_, (W_,)), ('W_to_QCD', W_, QCD), ('Z_to_QCD', Z_, QCD)):
        m = y == tru; a0, a1 = np.isin(c0[m], to), np.isin(c1[m], to); n = int(m.sum())
        tr[nm] = dict(n=n, before=float(a0.mean()), after=float(a1.mean()), before_ci=wilson(int(a0.sum()), n),
                      after_ci=wilson(int(a1.sum()), n), mcnemar=mcnemar(a0, a1))
    out['transitions'] = tr
    b = int(((c0 == W_) & (c1 == Z_)).sum()); c = int(((c0 == Z_) & (c1 == W_)).sum())
    out['direction_WZ'] = dict(W_call_to_Z_call=b, Z_call_to_W_call=c, frac_toward_Z=b / (b + c) if b + c else None,
                               p_binom=float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0)
    fl = np.zeros((5, 5), int); np.add.at(fl, (c0, c1), 1); out['flow'] = fl.tolist()
    dm = (L1[:, Z_] - L1[:, W_]) - (L0[:, Z_] - L0[:, W_]); wz = (y == W_) | (y == Z_)
    biw = boot(int(wz.sum()), seed + 1)
    out['margin'] = dict(dm_WZ=bstat(dm[wz], biw), dm_W=float(dm[y == W_].mean()), dm_Z=float(dm[y == Z_].mean()),
                         logit_maxabs_change=float(np.abs(L1 - L0).max()))
    if pred is not None:
        out['pred'] = dict(all=bstat(pred, bi), W=float(pred[y == W_].mean()), Z=float(pred[y == Z_].mean()),
                           quantiles=np.percentile(pred, [1, 10, 50, 90, 99]).tolist(), maxabs=float(np.abs(pred).max()))
        out['corr_dm_pred_WZ'] = spearman_boot(pred[wz], dm[wz], seed + 2)
        out['corr_dm_pred_W'] = spearman_boot(pred[y == W_], dm[y == W_], seed + 3)
        out['corr_dm_pred_Z'] = spearman_boot(pred[y == Z_], dm[y == Z_], seed + 4)
    return out, dict(c0=c0.astype(np.int8), c1=c1.astype(np.int8), dm=dm.astype(np.float32))


def readout_metrics(rd, H0, H1, Hc, te, pred, seed):
    """read-out change obs (edit) and cal (calibration edit: real change of 1/2 log(M0 M2) of size pred) on the test half"""
    r0 = rd(H0[te]); obs = rd(H1[te]) - r0; pr = pred[te]
    cal = rd(Hc[te]) - r0 if Hc is not None else None
    bi = boot(len(te), seed, B_RATIO); bo = bmeans(obs, bi); bp = bmeans(pr, bi)
    out = dict(n=int(len(te)), obs=bstat(obs, bi), pred=bstat(pr, bi),
               ratio_obs_pred=dict(ratio=float(obs.mean() / pr.mean()), ci=ci95(bo / bp)) if abs(pr.mean()) > 1e-6 else None,
               corr_obs_pred=spearman_boot(pr, obs, seed + 5))
    if cal is not None:
        bc = bmeans(cal, bi)
        out.update(cal=bstat(cal, bi), ratio_cal_pred=dict(ratio=float(cal.mean() / pr.mean()), ci=ci95(bc / bp)) if abs(pr.mean()) > 1e-6 else None,
                   ratio_obs_cal=dict(ratio=float(obs.mean() / cal.mean()), ci=ci95(bo / bc),
                                      p_ratio0=float(min(1.0, 2 * min((bo <= 0).mean(), (bo >= 0).mean()))),
                                      p_ratio1=float(min(1.0, 2 * min((bo - bc <= 0).mean(), (bo - bc >= 0).mean()))))
                   if abs(cal.mean()) > 1e-6 else None,
                   perjet_slope_obs_on_cal=float(np.polyfit(cal, obs, 1)[0]) if cal.std() > 1e-9 else None)
    return out, dict(obs=obs.astype(np.float32), cal=None if cal is None else cal.astype(np.float32))


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run_ = 0.0
    for r, i in enumerate(o):
        run_ = max(run_, min(1.0, (m - r) * p[i])); adj[i] = run_
    return adj


def ratio_diff(oa, ca, ob, cb, seed):
    """paired bootstrap over the same jets of mean(oa)/mean(ca) - mean(ob)/mean(cb)"""
    bi = boot(len(oa), seed, B_RATIO)
    ra, rb = bmeans(oa, bi) / bmeans(ca, bi), bmeans(ob, bi) / bmeans(cb, bi); d = ra - rb
    return dict(fwd=float(oa.mean() / ca.mean()), rev=float(ob.mean() / cb.mean()),
                diff=float(oa.mean() / ca.mean() - ob.mean() / cb.mean()), ci=ci95(d),
                p_boot=float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))))


def summary_line(tag, m):
    t = m['transitions']; d = m['direction_WZ']; c = m.get('corr_dm_pred_WZ')
    return (f'{tag}: acc {m["acc_before"]:.4f} -> {m["acc_after"]:.4f}; changed {100 * m["changed"]["mean"]:.2f} % '
            f'{[round(100 * v, 2) for v in m["changed"]["ci"]]}; W->Z {100 * t["W_to_Z"]["before"]:.1f} -> {100 * t["W_to_Z"]["after"]:.1f} % '
            f'(p {t["W_to_Z"]["mcnemar"]["p"]:.2e}); Z->W {100 * t["Z_to_W"]["before"]:.1f} -> {100 * t["Z_to_W"]["after"]:.1f} % '
            f'(p {t["Z_to_W"]["mcnemar"]["p"]:.2e}); W->QCD {100 * t["W_to_QCD"]["before"]:.1f} -> {100 * t["W_to_QCD"]["after"]:.1f} %; '
            f'calls W->Z {d["W_call_to_Z_call"]} vs Z->W {d["Z_call_to_W_call"]}; '
            + (f'pred {m["pred"]["all"]["mean"]:+.4f}; ' if 'pred' in m else '')
            + (f'rho(dm, pred) WZ {c["spearman"]:+.3f} {[round(v, 3) for v in c["ci"]]}, W {m["corr_dm_pred_W"]["spearman"]:+.3f} '
               f'{[round(v, 3) for v in m["corr_dm_pred_W"]["ci"]]}, Z {m["corr_dm_pred_Z"]["spearman"]:+.3f} {[round(v, 3) for v in m["corr_dm_pred_Z"]["ci"]]}'
               if c and m.get('corr_dm_pred_W') and m.get('corr_dm_pred_Z') else 'rho n/a'))


# ================================================================================================ small models
def small_mode(smoke):
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    xa, y = archive(); samp = sample_rows(y)
    if smoke: samp = samp[::20]
    X = rows(xa, samp); yn = y[samp]; J = len(X)
    log(f'X7 small: {J} jets (val archive, X4 rows rng 71{", every 20th (smoke)" if smoke else ""}) [{time.time() - t0:.0f}s]')
    res = dict(n=J, smoke=smoke, wta_selftest=wta_selftest())
    Xs = pt_sort(X); Xh = to_inputs(X, 'hardest'); Xh_np = recentre_hardest_np(Xs)
    res['check_recentre'] = dict(numpy_vs_torch_maxabs=float(np.abs(Xh_np - Xh).max()), sorted_identity=bool(np.array_equal(Xs, X)))
    log('recentre check', res['check_recentre'], 'wta selftest', res['wta_selftest'])
    tw = time.time(); ae, ap = wta_axes(Xs); Xw = reaxis(Xs, ae, ap)
    r_ = np.arange(J)
    res['axes'] = dict(wta_seconds=time.time() - tw, wta_is_hardest=float(np.mean((np.abs(ae - Xs[:, 0, 1]) < 1e-7) & (np.abs(ap - Xs[:, 0, 2]) < 1e-7))),
                       disp_hardest_median=float(np.median(np.hypot(Xs[:, 0, 1], Xs[:, 0, 2]))),
                       disp_wta_median=float(np.median(np.hypot(ae, ap))))
    IN = dict(stored=Xs, hardest=Xh, wta=Xw); AM = {k: axis_moment(v) for k, v in IN.items()}
    res['axes']['M2_rel'] = {f'{a}_vs_{b}': np.percentile(AM[a] - AM[b], [10, 50, 90]).tolist() for a in IN for b in IN if a != b}
    log('axes', json.dumps(res['axes']))
    perm = np.random.default_rng(71).permutation(J); tr, te = np.sort(perm[:J // 2]), np.sort(perm[J // 2:])
    trf, trv = tr[: len(tr) * 3 // 4], tr[len(tr) * 3 // 4:]
    res['models'] = {}; NPZ = {}; RO = {}
    for model in MODELS:
        for seed in SEEDS:
            for tax in ('stored', 'hardest'):
                name = run_name(model, tax); rd_ = SMALL / name / f'seed{seed}'
                net, arch = load_net(rd_); assert arch['inputs'].get('axis', 'stored') == tax
                runner = DSRunner(net) if arch['spec']['arch'] == 'dsets' else TFRunner(net); last = runner.last
                fwd = lambda xin: run(runner, xin, arch, want=('logits', last))
                key = f'{model}|{tax}|s{seed}'; R0 = fwd(IN[tax]); L0 = R0['logits']; c0 = L0.argmax(1)
                Pref = np.asarray(np.load(rd_ / 'test_probs.npy', mmap_mode='r')[samp])
                e = np.exp(L0 - L0.max(1, keepdims=True)); p = e / e.sum(1, keepdims=True)
                M = dict(model=model, train_axis=tax, seed=seed, last=last,
                         check=dict(acc=float((c0 == yn).mean()), probs_maxabs_diff=float(np.abs(Pref - p).max()),
                                    argmax_agree=float((Pref.argmax(1) == c0).mean())))
                rd = Ridge(R0[last], AM[tax], trf, trv, tr)
                M['readout_r2_te'] = float(1 - ((rd(R0[last][te]) - AM[tax][te]) ** 2).sum() / ((AM[tax][te] - AM[tax][te].mean()) ** 2).sum())
                log(f'{key}: check {M["check"]}; read-out R2 {M["readout_r2_te"]:.4f} [{time.time() - t0:.0f}s]')
                M['edits'] = {}
                for ed in [a for a in ('stored', 'hardest', 'wta') if a != tax]:
                    xe = IN[ed]; R1 = fwd(xe); pred = AM[ed] - AM[tax]
                    cal = to_inputs(scale_about_axis(X, np.exp(pred), tax), tax); Rc = fwd(cal)
                    cal_err = float(np.abs(axis_moment(cal) - AM[tax] - pred).max())
                    xr, rchk = matched_rotation(IN[tax], xe); Rr = fwd(xr)
                    E, pj = call_metrics(L0, R1['logits'], yn, pred, 100 + seed)
                    Er, pjr = call_metrics(L0, Rr['logits'], yn, np.zeros(J), 200 + seed)
                    ro, roj = readout_metrics(rd, R0[last], R1[last], Rc[last], te, pred, 300 + seed)
                    ror, rorj = readout_metrics(rd, R0[last], Rr[last], None, te, np.zeros(J), 400 + seed)
                    chg_e, chg_r = pj['c1'] != pj['c0'], pjr['c1'] != pjr['c0']
                    M['edits'][ed] = dict(edit=E, rotation=Er, rotation_check=rchk, readout=ro, readout_rotation=ror,
                                          cal_check_maxabs=cal_err, cal_changed=float((Rc['logits'].argmax(1) != c0).mean()),
                                          cal_metrics=call_metrics(L0, Rc['logits'], yn, None, 250 + seed)[0],
                                          mcnemar_changed_edit_vs_rotation=mcnemar(chg_r, chg_e))
                    NPZ[f'{key}|{ed}|pred'] = pred.astype(np.float32); NPZ[f'{key}|{ed}|dm'] = pj['dm']; NPZ[f'{key}|{ed}|dm_rot'] = pjr['dm']
                    NPZ[f'{key}|{ed}|c1'] = pj['c1']; NPZ[f'{key}|{ed}|c_rot'] = pjr['c1']
                    NPZ[f'{key}|{ed}|obs'] = roj['obs']; NPZ[f'{key}|{ed}|cal'] = roj['cal']
                    RO[key, ed] = roj
                    log(summary_line(f'  {key} on {ed}', E))
                    log(summary_line(f'  {key} rotation matched to {ed}', Er) + f'; rot check {rchk}')
                    rr = ro.get('ratio_obs_cal')
                    log(f'  {key} on {ed}: read-out obs {ro["obs"]["mean"]:+.4f} cal {ro["cal"]["mean"]:+.4f} pred {ro["pred"]["mean"]:+.4f} '
                        f'ratio obs/cal {rr} ; rotation read-out {ror["obs"]["mean"]:+.4f}; cal err {cal_err:.2e} [{time.time() - t0:.0f}s]')
                NPZ[f'{key}|c0'] = c0.astype(np.int8)
                res['models'][key] = M
    # forward (stored-trained on hardest / wta) minus reverse (axhard on stored / wta) response ratio, paired over test jets
    FR = {}
    for model in MODELS:
        for seed in SEEDS:
            for ef, er in (('hardest', 'stored'), ('wta', 'wta')):
                a, b = RO[f'{model}|stored|s{seed}', ef], RO[f'{model}|hardest|s{seed}', er]
                FR[f'{model}|s{seed}|fwd_{ef}_minus_rev_{er}'] = ratio_diff(a['obs'], a['cal'], b['obs'], b['cal'], 500 + seed)
                log(f'fwd-rev {model} s{seed} {ef}/{er}: {FR[f"{model}|s{seed}|fwd_{ef}_minus_rev_{er}"]}')
    res['fwd_minus_rev'] = FR
    x4p = RES / 'small' / 'x4_summary.json'
    if x4p.exists():
        x4 = json.loads(x4p.read_text())['slopes']
        res['x4_frac_of_cal'] = {k: v['frac_of_cal_b'] for k, v in x4.items()}
    res['elapsed_s'] = time.time() - t0; tag = 'x7_small' + ('_smoke' if smoke else '')
    (OUT / f'{tag}.json').write_text(json.dumps(res, indent=1)); np.savez_compressed(OUT / f'{tag}.npz', y=yn, **NPZ)
    log(f'wrote {OUT / tag}.json/.npz [{time.time() - t0:.0f}s]')


# ================================================================================================ JEDI and invariant controls
def efps(x, chunk=500):
    """B4's energy-flow polynomials (copied from interp.b4_baselines_jedi.efps): z_i = pT_i / sum pT, theta_ij = dR_ij^beta,
    all connected multigraphs with 1-3 edges, beta in {1, 2}, plus logs"""
    x = np.asarray(x, np.float64); out = []
    for i0 in range(0, len(x), chunk):
        xb = x[i0:i0 + chunk]; pt = xb[..., 0]; z = pt / np.maximum(pt.sum(1, keepdims=True), 1e-12)
        d = np.sqrt((xb[:, :, None, 1] - xb[:, None, :, 1]) ** 2 + (xb[:, :, None, 2] - xb[:, None, :, 2]) ** 2)
        F = []
        for beta in (1, 2):
            T = d ** beta; zt = np.einsum('ij,ijk->ik', z, T); zt2 = np.einsum('ij,ijk->ik', z, T ** 2); zt3 = np.einsum('ij,ijk->ik', z, T ** 3)
            B = T * z[:, None, :]; B2 = B @ B
            F += [(z * zt).sum(1), (z * zt2).sum(1), (z * zt ** 2).sum(1), (z * zt3).sum(1), (z * zt2 * zt).sum(1),
                  np.einsum('ijk,ikj->i', B2, B), np.einsum('ij,ijk,ik->i', z * zt, T, z * zt), (z * zt ** 3).sum(1)]
        F = np.stack(F, 1); out.append(np.concatenate([F, np.log(F + 1e-8)], 1))
    return np.concatenate(out).astype(np.float32)


def sample_idx(per_class, seed):
    """the test-file rows interp.jedi_core.sample(per_class, seed) returns (same rng calls)"""
    from jetdistill.config import RESULTS
    yy = np.asarray(np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r')['y']); rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(yy == c), per_class, replace=False) for c in range(5)]))


def jedi_mode(smoke):
    from interp.jedi_core import Jedi, sample
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    PER = 200 if smoke else 4000
    x, y, _ = sample(per_class=PER, seed=SEED_JEDI); x = np.ascontiguousarray(x, np.float32); J = len(x)
    net = Jedi(64)
    log(f'X7 jedi: {J} jets (sample({PER}, seed {SEED_JEDI})); model {net.path} [{time.time() - t0:.0f}s]')
    res = dict(n=J, per_class=PER, seed=SEED_JEDI, smoke=smoke, wta_selftest=wta_selftest())
    tw = time.time(); ae, ap = wta_axes(x); k = x[..., 0].argmax(1); r_ = np.arange(J)
    IN = dict(stored=x, hardest=reaxis(x, x[r_, k, 1].astype(np.float64), x[r_, k, 2].astype(np.float64)), wta=reaxis(x, ae, ap))
    AM = {kk: axis_moment(v) for kk, v in IN.items()}
    res['axes'] = dict(wta_seconds=time.time() - tw, hardest_is_slot0=float(np.mean(k == 0)),
                       wta_is_hardest=float(np.mean((np.abs(ae - x[r_, k, 1]) < 1e-7) & (np.abs(ap - x[r_, k, 2]) < 1e-7))),
                       disp_hardest_median=float(np.median(np.hypot(x[r_, k, 1], x[r_, k, 2]))),
                       disp_wta_median=float(np.median(np.hypot(ae, ap))))
    log('axes', json.dumps(res['axes']))
    perm = np.random.default_rng(SEED_JEDI).permutation(J); tr, te = np.sort(perm[:J // 2]), np.sort(perm[J // 2:])
    trf, trv = tr[: len(tr) * 3 // 4], tr[len(tr) * 3 // 4:]
    NPZ = {}; res['models'] = {}

    # ---------------------------------------------------------------- JEDI-linear
    jr = lambda xx: net.run(xx, keep=('z', 'logits'))
    R0 = jr(x); L0 = R0['logits']; c0 = L0.argmax(1)
    rd = Ridge(R0['z'], AM['stored'], trf, trv, tr)
    M = dict(check=dict(acc=float((c0 == y).mean())),
             readout_r2_te=float(1 - ((rd(R0['z'][te]) - AM['stored'][te]) ** 2).sum() / ((AM['stored'][te] - AM['stored'][te].mean()) ** 2).sum()),
             edits={})
    log(f'JEDI-linear: acc {M["check"]["acc"]:.4f}; read-out R2 {M["readout_r2_te"]:.4f} [{time.time() - t0:.0f}s]')
    ROT = {}
    for ed in ('hardest', 'wta'):
        xe = IN[ed]; R1 = jr(xe); pred = AM[ed] - AM['stored']; cal = scale_jets(x, np.exp(pred)); Rc = jr(cal)
        xr, rchk = matched_rotation(x, xe); ROT[ed] = xr; Rr = jr(xr)
        E, pj = call_metrics(L0, R1['logits'], y, pred, 100); Er, pjr = call_metrics(L0, Rr['logits'], y, np.zeros(J), 200)
        ro, roj = readout_metrics(rd, R0['z'], R1['z'], Rc['z'], te, pred, 300)
        ror, _ = readout_metrics(rd, R0['z'], Rr['z'], None, te, np.zeros(J), 400)
        M['edits'][ed] = dict(edit=E, rotation=Er, rotation_check=rchk, readout=ro, readout_rotation=ror,
                              cal_check_maxabs=float(np.abs(axis_moment(cal) - AM['stored'] - pred).max()),
                              cal_changed=float((Rc['logits'].argmax(1) != c0).mean()), cal_metrics=call_metrics(L0, Rc['logits'], y, None, 250)[0],
                              mcnemar_changed_edit_vs_rotation=mcnemar(pjr['c1'] != pjr['c0'], pj['c1'] != pj['c0']))
        for nm, v in (('pred', pred.astype(np.float32)), ('dm', pj['dm']), ('dm_rot', pjr['dm']), ('c1', pj['c1']), ('c_rot', pjr['c1']),
                      ('obs', roj['obs']), ('cal', roj['cal'])):
            NPZ[f'jedi|{ed}|{nm}'] = v
        log(summary_line(f'  JEDI-linear on {ed}', E)); log(summary_line(f'  JEDI-linear rotation matched to {ed}', Er) + f'; {rchk}')
        log(f'  read-out obs {ro["obs"]["mean"]:+.4f} cal {ro["cal"]["mean"]:+.4f} pred {ro["pred"]["mean"]:+.4f} ratio obs/cal {ro["ratio_obs_cal"]}; '
            f'rotation read-out {ror["obs"]["mean"]:+.4f} [{time.time() - t0:.0f}s]')
    NPZ['jedi|c0'] = c0.astype(np.int8); res['models']['jedi'] = M

    # ---------------------------------------------------------------- A: JEDI-linear after centroid recentring
    def recA(xx):
        ce, cp = centroid(xx); return reaxis(xx, ce, cp)
    A0 = recA(x); RA0 = jr(A0); LA0 = RA0['logits']; cA = LA0.argmax(1)
    MA = dict(check=dict(acc=float((cA == y).mean()), changed_vs_jedi_natural=float((cA != c0).mean()),
                         centroid_disp_median=float(np.median(np.hypot(*centroid(x))))), edits={})
    log(f'A (centroid-recentred JEDI-linear): {MA["check"]} [{time.time() - t0:.0f}s]')
    for ed in ('hardest', 'wta'):
        Ae = recA(IN[ed]); RA1 = jr(Ae); pA = axis_moment(Ae) - axis_moment(A0)
        Ar = recA(ROT[ed]); RAr = jr(Ar)
        E, pj = call_metrics(LA0, RA1['logits'], y, None, 110); Er, pjr = call_metrics(LA0, RAr['logits'], y, None, 210)
        MA['edits'][ed] = dict(edit=E, rotation=Er, input_maxabs_change=float(np.abs(Ae - A0).max()),
                               pred_maxabs=float(np.abs(pA).max()), z_maxabs_change=float(np.abs(RA1['z'] - RA0['z']).max()),
                               dM2_rot_maxabs=float(np.abs(axis_moment(Ar) - axis_moment(A0)).max()))
        NPZ[f'A|{ed}|c1'] = pj['c1']; NPZ[f'A|{ed}|c_rot'] = pjr['c1']
        log(summary_line(f'  A on {ed}', E) + f'; input max change {MA["edits"][ed]["input_maxabs_change"]:.2e}, logits max change {E["margin"]["logit_maxabs_change"]:.2e}')
        log(summary_line(f'  A rotation matched to {ed}', Er))
    NPZ['A|c0'] = cA.astype(np.int8); res['models']['A_centroid_jedi'] = MA

    # ---------------------------------------------------------------- B: EFP L1 logistic
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    PF = 300 if smoke else 6000
    xf, yf, _ = sample(per_class=PF, seed=0); i_f, i_e = sample_idx(PF, 0), sample_idx(PER, SEED_JEDI)
    keep = ~np.isin(i_f, i_e); xf, yf = np.asarray(xf, np.float32)[keep], yf[keep]
    Gf = efps(xf); sc = StandardScaler().fit(Gf); Gf = sc.transform(Gf)
    pf = np.random.default_rng(5).permutation(len(yf)); fi, vi = pf[: 2 * len(pf) // 3], pf[2 * len(pf) // 3:]
    best = None; path = []
    for C in (0.01, 0.1, 1.0):
        m = LogisticRegression(penalty='l1', solver='saga', C=C, max_iter=500, tol=1e-4, random_state=0).fit(Gf[fi], yf[fi])
        a = float((m.predict(Gf[vi]) == yf[vi]).mean())
        path.append(dict(C=C, val_acc=a, nonzero=int((np.abs(m.coef_) > 0).sum()), n_iter=int(np.max(m.n_iter_))))
        log(f'  EFP logistic C {C}: {path[-1]} [{time.time() - t0:.0f}s]')
        if best is None or a > best[0]: best = (a, C, m)
    mB = best[2]; fB = lambda xx: mB.decision_function(sc.transform(efps(xx)))
    LB0 = fB(x); cB = LB0.argmax(1)
    MB = dict(fit=dict(n_fit=int(len(fi)), n_val=int(len(vi)), removed_overlap=int((~keep).sum()), path=path, C=best[1]),
              check=dict(acc=float((cB == y).mean())), edits={})
    log(f'B (EFP L1 logistic): {MB["fit"]}; acc on the evaluation sample {MB["check"]["acc"]:.4f} [{time.time() - t0:.0f}s]')
    for ed in ('hardest', 'wta'):
        LB1 = fB(IN[ed]); LBr = fB(ROT[ed])
        E, pj = call_metrics(LB0, LB1, y, None, 120); Er, pjr = call_metrics(LB0, LBr, y, None, 220)
        MB['edits'][ed] = dict(edit=E, rotation=Er, efp_maxabs_rel_change=float(np.abs(efps(IN[ed])[:, :16] / np.maximum(efps(x)[:, :16], 1e-12) - 1).max()))
        NPZ[f'B|{ed}|c1'] = pj['c1']; NPZ[f'B|{ed}|c_rot'] = pjr['c1']
        log(summary_line(f'  B on {ed}', E) + f'; EFP max rel change {MB["edits"][ed]["efp_maxabs_rel_change"]:.2e}')
        log(summary_line(f'  B rotation matched to {ed}', Er))
    NPZ['B|c0'] = cB.astype(np.int8); res['models']['B_efp_logistic'] = MB
    res['elapsed_s'] = time.time() - t0; tag = 'x7_jedi' + ('_smoke' if smoke else '')
    (OUT / f'{tag}.json').write_text(json.dumps(res, indent=1)); np.savez_compressed(OUT / f'{tag}.npz', y=y, **NPZ)
    log(f'wrote {OUT / tag}.json/.npz [{time.time() - t0:.0f}s]')


# ================================================================================================ combine
def rows_all(S, Jd):
    """(label, model kind, direction, edit, metrics block) for every (model, edit)"""
    out = []
    for ed in ('hardest', 'wta'):
        out.append((f'JEDI-linear | stored-trained -> {ed}', 'jedi', 'forward', ed, Jd['models']['jedi']['edits'][ed]))
        out.append((f'A (centroid-recentred JEDI) -> {ed}', 'A', 'invariant', ed, Jd['models']['A_centroid_jedi']['edits'][ed]))
        out.append((f'B (EFP logistic) -> {ed}', 'B', 'invariant', ed, Jd['models']['B_efp_logistic']['edits'][ed]))
    for key, M in S['models'].items():
        model, tax, s = key.split('|')
        for ed, E in M['edits'].items():
            out.append((f'{SHORT[model]} {s} | {"stored" if tax == "stored" else "axhard"}-trained -> {ed}', model,
                        'forward' if tax == 'stored' else 'reverse', ed, E))
    return out


def combine(smoke):
    sfx = '_smoke' if smoke else ''
    S = json.loads((OUT / f'x7_small{sfx}.json').read_text()); Jd = json.loads((OUT / f'x7_jedi{sfx}.json').read_text())
    R = rows_all(S, Jd); tests = []
    for lab, kind, dr, ed, E in R:
        for which in ('edit', 'rotation'):
            m = E[which]
            tests += [(lab, which, 'correct', m['mcnemar_correct']['p']), (lab, which, 'W_to_Z', m['transitions']['W_to_Z']['mcnemar']['p']),
                      (lab, which, 'Z_to_W', m['transitions']['Z_to_W']['mcnemar']['p']), (lab, which, 'direction_WZ', m['direction_WZ']['p_binom'])]
        for ck in ('corr_dm_pred_WZ', 'corr_dm_pred_W', 'corr_dm_pred_Z'):
            c = E['edit'].get(ck)
            if c: tests.append((lab, 'edit', ck, c['p']))
        c = E.get('readout', {}).get('corr_obs_pred')
        if c: tests.append((lab, 'edit', 'readout_corr_obs_pred', c['p']))
        if 'mcnemar_changed_edit_vs_rotation' in E: tests.append((lab, 'edit_vs_rotation', 'changed', E['mcnemar_changed_edit_vs_rotation']['p']))
    adj = holm([t[3] for t in tests])
    H = [dict(row=t[0], edit=t[1], test=t[2], p=t[3], p_holm=float(a)) for t, a in zip(tests, adj)]
    table = []
    for lab, kind, dr, ed, E in R:
        e, r = E['edit'], E['rotation']; ro = E.get('readout', {}); c = e.get('corr_dm_pred_WZ')
        table.append(dict(row=lab, kind=kind, direction=dr, edit=ed, acc_before=e['acc_before'], acc_after=e['acc_after'],
                          changed=e['changed'], changed_rot=r['changed'], acc_after_rot=r['acc_after'],
                          changed_cal=E['cal_metrics']['changed'] if 'cal_metrics' in E else ({'mean': 0.0, 'ci': [0.0, 0.0]} if kind in ('A', 'B') else None),
                          direction_cal=E['cal_metrics']['direction_WZ'] if 'cal_metrics' in E else None,
                          W_to_Z=[e['transitions']['W_to_Z']['before'], e['transitions']['W_to_Z']['after']],
                          Z_to_W=[e['transitions']['Z_to_W']['before'], e['transitions']['Z_to_W']['after']],
                          W_to_QCD=[e['transitions']['W_to_QCD']['before'], e['transitions']['W_to_QCD']['after']],
                          Z_to_QCD=[e['transitions']['Z_to_QCD']['before'], e['transitions']['Z_to_QCD']['after']],
                          direction_WZ=e['direction_WZ'], pred=e.get('pred', {}).get('all'), rho=c,
                          rho_W=e.get('corr_dm_pred_W'), rho_Z=e.get('corr_dm_pred_Z'), rho_readout=ro.get('corr_obs_pred'),
                          ratio_obs_cal=ro.get('ratio_obs_cal'), readout_obs=ro.get('obs'), readout_rot=E.get('readout_rotation', {}).get('obs')))
    out = dict(holm=dict(n_tests=len(H), tests=H), table=table, fwd_minus_rev=S['fwd_minus_rev'], x4_frac_of_cal=S.get('x4_frac_of_cal'),
               checks=dict(jedi_acc=Jd['models']['jedi']['check']['acc'], A=Jd['models']['A_centroid_jedi']['check'], B=Jd['models']['B_efp_logistic']['check'],
                           small={k: v['check'] for k, v in S['models'].items()}, recentre=S['check_recentre']))
    (OUT / f'x7_summary{sfx}.json').write_text(json.dumps(out, indent=1)); log(f'wrote x7_summary{sfx}.json ({len(H)} tests, Holm)')
    for t in table:
        log(f'{t["row"]}: changed {100 * t["changed"]["mean"]:.2f} % rot {100 * t["changed_rot"]["mean"]:.2f} %; ratio {t["ratio_obs_cal"]}; rho {t["rho"] and round(t["rho"]["spearman"], 3)}')
    plot(sfx)


def plot(sfx=''):
    """figure from x7_summary.json: calls changed (axis change / matched real change of 1/2 log(M0 M2) / matched rotation) and
    net W <-> Z call flow"""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    table = json.loads((OUT / f'x7_summary{sfx}.json').read_text())['table']
    sel = [t for t in table if t['edit'] == ('hardest' if t['direction'] != 'reverse' else 'stored')]
    sel = sorted(sel, key=lambda t: ({'jedi': 0, 'A': 1, 'B': 2}.get(t['kind'], 3), t['direction'] == 'reverse', t['row']))
    def lab(t):
        if t['kind'] == 'A': return 'invariant control A:\nJEDI-linear after centroid recentring'
        if t['kind'] == 'B': return 'invariant control B:\nEFP L1 logistic'
        nm = t['row'].split(' | ')[0]
        return (f'{nm}, trained on stored axis,\nevaluated about hardest particle' if t['direction'] == 'forward'
                else f'{nm}, trained about hardest particle,\nevaluated on stored axis')
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 0.62 * len(sel) + 1.6), dpi=150, sharey=True, gridspec_kw=dict(width_ratios=[1.5, 1]))
    yy = np.arange(len(sel))[::-1]; h = 0.26
    spec = (('changed', '#c8553d', h, 'axis change (the edit)'),
            ('changed_cal', '#e9a03b', 0.0, 'real change of ½log(M0·M2) of the predicted size (formula prediction)'),
            ('changed_rot', '#2a6f97', -h, 'rotation with the same mean particle displacement (M0, M2 unchanged)'))
    for i, t in zip(yy, sel):
        for key, col, off, nm in spec:
            v = t.get(key)
            if v is None: continue
            axs[0].barh(i + off, 100 * v['mean'], height=h * 0.95, color=col, label=nm if i == yy[0] else None)
            axs[0].errorbar(100 * v['mean'], i + off, xerr=[[100 * (v['mean'] - v['ci'][0])], [100 * (v['ci'][1] - v['mean'])]], fmt='none', ecolor='k', lw=0.7)
        d = t['direction_WZ']; net = 100 * (d['W_call_to_Z_call'] - d['Z_call_to_W_call']) / 20000
        axs[1].barh(i, net, height=0.5, color='#6a4c93' if net >= 0 else '#3a7d44')
        pr = t.get('pred')
        axs[1].text(max(net, 0) + 0.4, i, f'{d["W_call_to_Z_call"]} / {d["Z_call_to_W_call"]}' + (f'   (pred {pr["mean"]:+.3f})' if pr else ''),
                    va='center', ha='left', fontsize=7.5)
    axs[0].set_yticks(yy); axs[0].set_yticklabels([lab(t) for t in sel], fontsize=7.8)
    axs[0].set_xlabel('calls changed (% of 20,000 jets, 95 % bootstrap CI)')
    axs[0].legend(fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.09 - 0.25 / len(sel) * 4), ncol=1, frameon=False)
    axs[0].grid(alpha=0.25, axis='x')
    axs[1].axvline(0, color='k', lw=0.8); axs[1].grid(alpha=0.25, axis='x'); axs[1].set_xlim(-16, 16)
    axs[1].set_xlabel('net call flow under the axis change, % of jets:\n(W call -> Z call) - (Z call -> W call); text: counts, mean predicted ½Δlog M2')
    fig.suptitle('X7: calls changed by a change of axis convention, in both directions, with a matched real mass change and a matched rotation', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(OUT / f'x7_axis_reverse{sfx}.png', bbox_inches='tight'); log(f'wrote x7_axis_reverse{sfx}.png')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=('small', 'jedi', 'combine', 'plot'), required=True); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    dict(small=small_mode, jedi=jedi_mode, combine=combine, plot=lambda sm: plot('_smoke' if sm else ''))[a.mode](a.smoke)


if __name__ == '__main__':
    main()
