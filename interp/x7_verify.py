"""X7 verification: rerun of the axis-convention response on fresh jets, with two extra controls.

jedi  (jax env): JEDI-linear 3-feature n64 on interp.jedi_core.sample(PER, seed 211) (fresh seed; X7 used 107).
small (conda env): the 8 small models (Deep Sets d64 / Linformer 2b-d32-k8, stored / axhard, seeds 0, 1) on fresh val-archive
      rows (rng 911, PER per class, X4/X7's rng-71 rows excluded).
Edits per model (training axis tax, other axis ed in {stored, hardest}):
  edit      inputs measured about the other axis (X7's edit): a rigid translation of the real particles by -t
  cal       offsets about the training axis scaled by exp(pred), pred = 1/2 dlog(M0 M2) of the edit (X7's calibration)
  rot       X7's matched rotation (same mean particle displacement, M2 fixed)
  m2t       NEW: the edit followed by scaling the offsets about the new origin by exp(-pred): the same translation, but
            1/2 log(M0 M2) about the model's axis restored to its natural value (pred = 0 to float precision). Particles move
            by about the same amount as under the edit, and the shape relative to the axis changes as in the edit.
  rnd       NEW (JEDI only): a rigid translation of the same length |t| in a uniformly random direction (rng 212); its own
            pred_rnd and its own calibration cal_rnd.
Metrics: % calls changed (bootstrap CI over jets), W->Z / Z->W switched calls (binomial), transitions, McNemar of changed
calls edit vs each control, per-class Spearman of dm = d(L_Z - L_W) with pred, and the partial Spearman given the translation
length |t| and given |t| / rbar; ridge read-out ratio (edit / cal and m2t / cal) as in X7. Holm over all p-values.
  ./job.sh x7-verify 'python -m interp.x7_verify --mode jedi'            -> $AI_RESULTS/jedi/x7_verify_jedi.json
  AI_PY=conda ./job.sh x7-verify2 'python -m interp.x7_verify --mode small' -> $AI_RESULTS/jedi/x7_verify_small.json
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
from interp.x7_axis_reverse import (matched_rotation, scale_jets, boot, bstat, spearman_boot, call_metrics,  # noqa: E402
                                    readout_metrics, holm, W_, Z_)
from interp.x2_small_models import Ridge, mcnemar  # noqa: E402
from interp.x4_axis_models import axis_moment  # noqa: E402
from interp.e17_axis_jedi import reaxis  # noqa: E402

RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); OUT = RES / 'jedi'


def log(*a):
    print(time.strftime('[%H:%M:%S]'), *a, flush=True)


def partial_spearman(xv, yv, zv):
    from scipy.stats import rankdata
    rx, ry, rz = rankdata(xv), rankdata(yv), rankdata(zv)
    A = np.stack([rz, np.ones_like(rz)], 1)
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]; ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    r = float(np.corrcoef(ex, ey)[0, 1]); n = len(xv)
    from scipy.stats import t as tdist
    tt = r * np.sqrt((n - 3) / max(1e-12, 1 - r * r)); p = float(2 * tdist.sf(abs(tt), n - 3))
    rng = np.random.default_rng(9); bs = []
    for _ in range(300):
        ii = rng.integers(0, n, n); bs.append(np.corrcoef(ex[ii], ey[ii])[0, 1])
    return dict(r=r, ci=np.percentile(bs, [2.5, 97.5]).tolist(), p=p)


def rbar(x):
    real = x[..., 0] > 0; n = np.maximum(real.sum(1), 1)
    return np.where(real, np.hypot(x[..., 1].astype(np.float64), x[..., 2].astype(np.float64)), 0).sum(1) / n


def block(L0, Lx, y, pred, seed, tlen=None, rb=None):
    E, pj = call_metrics(L0, Lx, y, pred, seed)
    keep = dict(acc_before=E['acc_before'], acc_after=E['acc_after'], changed=E['changed'], direction_WZ=E['direction_WZ'],
                transitions={k: dict(before=v['before'], after=v['after'], p=v['mcnemar']['p']) for k, v in E['transitions'].items()},
                pred=E.get('pred', {}).get('all'))
    for k in ('corr_dm_pred_W', 'corr_dm_pred_Z'):
        if E.get(k): keep[k] = dict(spearman=E[k]['spearman'], ci=E[k]['ci'], p=E[k]['p'])
    if pred is not None and tlen is not None:
        for cname, cl in (('W', W_), ('Z', Z_)):
            m = y == cl
            keep[f'partial_{cname}_given_t'] = partial_spearman(pred[m], pj['dm'][m], tlen[m])
            keep[f'partial_{cname}_given_t_over_rbar'] = partial_spearman(pred[m], pj['dm'][m], tlen[m] / np.maximum(rb[m], 1e-9))
    return keep, pj


def summarise(tag, B):
    d = B['direction_WZ']
    s = (f'{tag}: changed {100 * B["changed"]["mean"]:.2f} % {[round(100 * v, 2) for v in B["changed"]["ci"]]}; acc '
         f'{B["acc_before"]:.4f} -> {B["acc_after"]:.4f}; W->Z/Z->W calls {d["W_call_to_Z_call"]}/{d["Z_call_to_W_call"]} (p {d["p_binom"]:.1e})')
    if B.get('pred'): s += f'; pred {B["pred"]["mean"]:+.4f}'
    if 'corr_dm_pred_W' in B: s += f'; rho W {B["corr_dm_pred_W"]["spearman"]:+.3f} Z {B["corr_dm_pred_Z"]["spearman"]:+.3f}'
    if 'partial_W_given_t' in B:
        s += (f'; partial|t W {B["partial_W_given_t"]["r"]:+.3f} Z {B["partial_Z_given_t"]["r"]:+.3f}; partial|t/rbar W '
              f'{B["partial_W_given_t_over_rbar"]["r"]:+.3f} Z {B["partial_Z_given_t_over_rbar"]["r"]:+.3f}')
    return s


def evaluate(fwd, last, xn, xe, tax_moment_fn, y, seed, extra=None):
    """fwd(x) -> dict(logits, last); xn natural inputs (origin = training axis), xe = edited (origin = other axis)"""
    J = len(y); AMn, AMe = axis_moment(xn), axis_moment(xe); pred = AMe - AMn
    tlen = np.hypot(*(xn[:, 0, 1:3].astype(np.float64) - xe[:, 0, 1:3]).T)  # slot-0 displacement = |t| (translation)
    rb = rbar(xn)
    cal = tax_moment_fn(np.exp(pred)); m2t = scale_jets(xe, np.exp(-pred)); rot, rchk = matched_rotation(xn, xe)
    chk = dict(cal_err=float(np.abs(axis_moment(cal) - AMn - pred).max()), m2t_err=float(np.abs(axis_moment(m2t) - AMn).max()),
               rot=rchk, t_median=float(np.median(tlen)), rbar_median=float(np.median(rb)))
    R0 = fwd(xn); L0 = R0['logits']
    perm = np.random.default_rng(seed).permutation(J); tr, te = np.sort(perm[:J // 2]), np.sort(perm[J // 2:])
    rd = Ridge(R0[last], AMn, tr[: len(tr) * 3 // 4], tr[len(tr) * 3 // 4:], tr)
    out = dict(check=dict(acc=float((L0.argmax(1) == y).mean()), **chk)); CH = {}; RO = {}
    Rc = fwd(cal)
    for nm, xx, pr in (('edit', xe, pred), ('cal', cal, pred), ('rot', rot, None), ('m2t', m2t, None)):
        R1 = Rc if nm == 'cal' else fwd(xx)
        out[nm], pj = block(L0, R1['logits'], y, pr, seed + len(nm), tlen if nm == 'edit' else None, rb)
        CH[nm] = pj['c1'] != pj['c0']
        if nm in ('edit', 'm2t', 'rot'):
            ro, roj = readout_metrics(rd, R0[last], R1[last], Rc[last], te, pred, seed + 7)
            out[nm]['readout'] = dict(obs=ro['obs'], cal=ro['cal'], ratio_obs_cal=ro['ratio_obs_cal'])
        log(summarise(f'  {nm}', out[nm]) + (f'; read-out ratio {out[nm]["readout"]["ratio_obs_cal"]}' if 'readout' in out[nm] else ''))
    for nm in ('cal', 'rot', 'm2t'):
        out[f'mcnemar_changed_edit_vs_{nm}'] = mcnemar(CH[nm], CH['edit'])
    if extra:
        for nm, xx in extra.items():
            pr = axis_moment(xx) - AMn; calx = tax_moment_fn(np.exp(pr))
            for nn, x2, p2 in ((nm, xx, pr), (nm + '_cal', calx, pr), (nm + '_m2', scale_jets(xx, np.exp(-pr)), None)):
                out[nn], pj = block(L0, fwd(x2)['logits'], y, p2, seed + 11)
                CH[nn] = pj['c1'] != pj['c0']; log(summarise(f'  {nn}', out[nn]))
            out[f'mcnemar_changed_{nm}_vs_edit'] = mcnemar(CH['edit'], CH[nm])
    return out


def pvals(o, prefix):
    P = []
    for k, v in o.items():
        if isinstance(v, dict):
            if 'direction_WZ' in v: P.append((f'{prefix}{k}|direction', v['direction_WZ']['p_binom']))
            for c in ('corr_dm_pred_W', 'corr_dm_pred_Z', 'partial_W_given_t', 'partial_Z_given_t', 'partial_W_given_t_over_rbar', 'partial_Z_given_t_over_rbar'):
                if c in v: P.append((f'{prefix}{k}|{c}', v[c]['p']))
            if k.startswith('mcnemar'): P.append((f'{prefix}{k}', v['p']))
    return P


def jedi_mode(per):
    from interp.jedi_core import Jedi, sample
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    x, y, _ = sample(per_class=per, seed=211); x = np.ascontiguousarray(x, np.float32); J = len(x); net = Jedi(64)
    log(f'x7-verify jedi: {J} jets sample({per}, 211)')
    r_ = np.arange(J); k = x[..., 0].argmax(1); te_, tp_ = x[r_, k, 1].astype(np.float64), x[r_, k, 2].astype(np.float64)
    xe = reaxis(x, te_, tp_)
    ang = np.random.default_rng(212).uniform(0, 2 * np.pi, J); L = np.hypot(te_, tp_)
    xrnd = reaxis(x, L * np.cos(ang), L * np.sin(ang))
    fwd = lambda xx: net.run(xx, keep=('z', 'logits'))
    res = dict(n=J, per_class=per, seed=211, hardest_is_slot0=float((k == 0).mean()))
    res['jedi'] = evaluate(fwd, 'z', x, xe, lambda kk: scale_jets(x, kk), y, 600, extra=dict(rnd=xrnd))
    P = pvals(res['jedi'], 'jedi|'); adj = holm([p for _, p in P])
    res['holm'] = [dict(test=t, p=p, p_holm=float(a)) for (t, p), a in zip(P, adj)]
    res['elapsed_s'] = time.time() - t0
    (OUT / 'x7_verify_jedi.json').write_text(json.dumps(res, indent=1)); log('wrote x7_verify_jedi.json', f'[{time.time() - t0:.0f}s]')


def small_mode(per):
    import torch
    torch.set_num_threads(1)
    from interp.x2_small_models import load_net, DSRunner, TFRunner, run
    from interp.x4_axis_models import archive, rows, pt_sort, to_inputs, scale_about_axis, run_name, SMALL, sample_rows
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    xa, yall = archive(); old = sample_rows(yall); rng = np.random.default_rng(911)
    samp = np.sort(np.concatenate([rng.choice(np.setdiff1d(np.flatnonzero(yall == c), old), per, replace=False) for c in range(5)]))
    X = rows(xa, samp); y = yall[samp]; J = len(X)
    log(f'x7-verify small: {J} fresh val-archive rows (rng 911, {per}/class, X4 rows excluded: overlap {np.isin(samp, old).sum()})')
    Xs = pt_sort(X); IN = dict(stored=Xs, hardest=to_inputs(X, 'hardest'))
    res = dict(n=J, per_class=per, rng=911, models={}); P = []
    for model in ('dsets-d64', 'linf-2b-d32-k8'):
        for seed in (0, 1):
            for tax in ('stored', 'hardest'):
                rd_ = SMALL / run_name(model, tax) / f'seed{seed}'; net, arch = load_net(rd_)
                assert arch['inputs'].get('axis', 'stored') == tax
                runner = DSRunner(net) if arch['spec']['arch'] == 'dsets' else TFRunner(net); last = runner.last
                fwd = lambda xin, runner=runner, arch=arch, last=last: run(runner, xin, arch, want=('logits', last))
                ed = 'hardest' if tax == 'stored' else 'stored'; key = f'{model}|{tax}|s{seed}'
                log(f'{key} on {ed} [{time.time() - t0:.0f}s]')
                calfn = lambda kk, tax=tax: to_inputs(scale_about_axis(X, kk, tax), tax)
                o = evaluate(fwd, last, IN[tax], IN[ed], calfn, y, 700 + seed)
                res['models'][key] = o; P += pvals(o, key + '|')
    adj = holm([p for _, p in P]); res['holm'] = [dict(test=t, p=p, p_holm=float(a)) for (t, p), a in zip(P, adj)]
    res['elapsed_s'] = time.time() - t0
    (OUT / 'x7_verify_small.json').write_text(json.dumps(res, indent=1)); log('wrote x7_verify_small.json', f'[{time.time() - t0:.0f}s]')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--mode', choices=('jedi', 'small'), required=True)
    ap.add_argument('--per', type=int, default=None); a = ap.parse_args()
    (jedi_mode if a.mode == 'jedi' else small_mode)(a.per or (3000 if a.mode == 'jedi' else 2000))
