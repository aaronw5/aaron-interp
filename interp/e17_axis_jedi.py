"""E17 — JEDI-linear (3-feature, n64) under a change of jet-axis convention; comparison with ParT (P1v check 4).

Procedure: 4,000 test jets per class (seed 41) from interp.jedi_core.sample. For every jet the (Δη, Δφ) of each real particle
(pT > 0) are recomputed about a new axis given in the stored relative coordinates; padding slots keep their stored values;
pT and slot order unchanged. Axes:
  stored    the stored axis through the same code (identity control)
  centroid  pT-weighted centroid of the 64 stored particles, (Σ pT Δη, Σ pT Δφ) / Σ pT (control: close to the stored axis)
  wta       winner-take-all axis: C/A clustering of all real particles in (Δη, Δφ) (smallest ΔR² merged first, no beam
            distance), WTA recombination (pT summed, direction of the harder branch); wta_axis_one copied from
            interp/p1v_verify_part.py
  hardest   the highest-pT particle
Per axis: axis displacement, |M1|²/(M0·M2) about the new axis, accuracy, calls changed (overall, per true class), W→Z, Z→W,
class flow table (call on the stored axis → call on the new axis), the ridge read-out of log m on z (fitted on the unedited
jets of the train half; alpha chosen on the last quarter of that half; applied to the test half) and the prediction
½Δlog(M0·M2) about the new axis (M0 unchanged, so ½Δlog M2); input-range flags (l1 input quantizer WRAP range; training 99 %
|Δη|, |Δφ| from E6v: 0.342, 0.336 rad). m (massless four-vectors of the 64 particles) is invariant under the re-axis.
  ./job.sh e17 'python -m interp.e17_axis_jedi'            -> results/jedi/e17_axis.json (+ the figures, if p1v json exists)
  ./job.sh e17-smoke 'python -m interp.e17_axis_jedi 100'  (100 jets per class -> e17_axis_100.json)
  python -m interp.e17_axis_jedi --plot                    figures only: results/jedi/e17_axis.png (table, JEDI vs ParT from
                                                           results/part/p1v_verify.json), results/jedi/e17_axis_bins.png
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results'))
OUT = RES / 'jedi'
CL = ['g', 'q', 'W', 'Z', 't']; W, Z = 2, 3
SEED = 41
AXES = ('stored', 'centroid', 'wta', 'hardest')
ALPHAS = 10.0 ** np.arange(-4, 4.1, 0.5)
TR99 = dict(eta=0.341552734375, phi=0.335693359375)       # E6v (results/jedi/e6_verify.json input_range.train_abs_p99)
DBINS = (0.0, 0.005, 0.01, 0.02, 0.04, 0.07, 0.1, 0.15, 0.25, 10.0)
log = lambda *a: print(*a, flush=True)


# ------------------------------------------------------------------------------------------------ axes
def wta_axis_one(pt, y, p):     # verbatim from interp/p1v_verify_part.py
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


def wta_selftest():             # as interp/p1v_verify_part.py
    a = wta_axis_one(np.array([40., 30., 25., 5.]), np.array([0., 0.3, 0.31, 0.01]), np.zeros(4))
    b = wta_axis_one(np.array([50., 30., 25., 10.]), np.array([0., 0.3, 0.31, 0.01]), np.zeros(4))
    c = wta_axis_one(np.array([7.]), np.array([0.2]), np.array([-0.1]))
    return dict(ok=bool(a == (0.3, 0.0) and b == (0.0, 0.0) and c == (0.2, -0.1)), a=a, b=b, c=c)


def axis_set(x):
    """per jet, each axis in the stored (Δη, Δφ) coordinates"""
    pt = x[..., 0].astype(np.float64); de = x[..., 1].astype(np.float64); dp = x[..., 2].astype(np.float64); real = pt > 0
    M0 = pt.sum(1); k = pt.argmax(1); r = np.arange(len(x))
    t = time.time(); wta = np.array([wta_axis_one(pt[j, real[j]], de[j, real[j]], dp[j, real[j]]) for j in range(len(x))])
    A = dict(stored=(np.zeros(len(x)), np.zeros(len(x))), centroid=((pt * de).sum(1) / M0, (pt * dp).sum(1) / M0),
             wta=(wta[:, 0], wta[:, 1]), hardest=(de[r, k], dp[r, k]))
    info = dict(wta_seconds=time.time() - t,
                wta_is_hardest=float(np.mean((np.abs(wta[:, 0] - de[r, k]) < 1e-9) & (np.abs(wta[:, 1] - dp[r, k]) < 1e-9))),
                hardest_is_slot0=float(np.mean(k == 0)),
                pt_sorted_desc=float(np.mean(np.all(np.diff(pt, axis=1) <= 0, axis=1))),
                padding_is_trailing=float(np.mean([np.all(real[j, :real[j].sum()]) for j in range(len(x))])),
                multiplicity_mean=float(real.sum(1).mean()), padding_slot_fraction=float(1 - real.mean()))
    return A, info


def reaxis(x, ae, ap):
    """real particles' (Δη, Δφ) about the axis (ae, ap) [stored relative coordinates]; Δφ wrapped to [−π, π); padding and pT
    untouched"""
    real = x[..., 0] > 0; e = x.copy()
    de = x[..., 1].astype(np.float64) - ae[:, None]; dp = (x[..., 2].astype(np.float64) - ap[:, None] + np.pi) % (2 * np.pi) - np.pi
    e[..., 1] = np.where(real, de, x[..., 1]).astype(np.float32); e[..., 2] = np.where(real, dp, x[..., 2]).astype(np.float32)
    return e


def moments(xx):                # as interp/e6_verify.py
    xx = np.asarray(xx, np.float64); pt, eta, phi = xx[..., 0], xx[..., 1], xx[..., 2]
    r2 = eta ** 2 + phi ** 2
    M0 = pt.sum(1); M1e = (pt * eta).sum(1); M1p = (pt * phi).sum(1); M2 = (pt * r2).sum(1)
    p4 = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), pt * np.cosh(eta)], -1).sum(1)
    m = np.sqrt(np.maximum(p4[:, 3] ** 2 - (p4[:, :3] ** 2).sum(1), 0.0))
    return dict(M0=M0, M1e=M1e, M1p=M1p, M2=M2, m=m, logm=np.log(np.maximum(m, 1e-3)))


class Ridge:                    # as interp/e6_verify.py
    def __init__(self, X, t, idx_fit, idx_val, idx_all):
        self.mu = X[idx_all].mean(0); self.sd = X[idx_all].std(0); self.sd[self.sd < 1e-9] = np.inf
        Zs = (X - self.mu) / self.sd; self.tm = t[idx_all].mean(); self.ts = t[idx_all].std(); tt = (t - self.tm) / self.ts; best = None
        for a in ALPHAS:
            w = self._fit(Zs[idx_fit], tt[idx_fit], a); e = ((Zs[idx_val] @ w - tt[idx_val]) ** 2).mean()
            if best is None or e < best[0]: best = (e, a)
        self.alpha = best[1]; self.w = self._fit(Zs[idx_all], tt[idx_all], self.alpha)

    @staticmethod
    def _fit(Zs, t, a):
        Zc = Zs - Zs.mean(0); return np.linalg.solve(Zc.T @ Zc + a * len(Zs) * np.eye(Zs.shape[1]), Zc.T @ (t - t.mean()))

    def __call__(self, X):
        return ((X - self.mu) / self.sd) @ self.w * self.ts + self.tm


def r2(p, t):
    return float(1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum())


# ------------------------------------------------------------------------------------------------ main
def main(PER):
    from interp.jedi_core import Jedi, sample
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    res = dict(per_class=PER, seed=SEED, classes=CL, wta_selftest=wta_selftest())
    log(f'WTA self-test: {res["wta_selftest"]}')
    x, y, _ = sample(per_class=PER, seed=SEED); x = np.ascontiguousarray(x, np.float32); J = len(x); res['N'] = J
    net = Jedi(64)
    log(f'E17: {J} jets ({PER}/class, seed {SEED}); model {net.path} ({time.time() - t0:.0f} s)')
    # input ranges (as E6v): l1 input quantizer, fixed point per slot, WRAP overflow
    q = net.L['l1'].iq.quantizer
    kq, iq, fq = (np.asarray(getattr(q, a), np.float64).reshape(64, 3) for a in ('k', 'i', 'f'))
    QLO = np.where(kq > 0, -2.0 ** iq, 0.0); QHI = 2.0 ** iq - 2.0 ** (-fq)

    def flags(xx):
        xn = (np.asarray(xx, np.float64) - net.shift) / net.scale; rl = xx[..., 0] > 0
        wrap = ((xn < QLO) | (xn > QHI)).any((1, 2))
        out = (((np.abs(xx[..., 1]) > TR99['eta']) | (np.abs(xx[..., 2]) > TR99['phi'])) & rl)
        return wrap, out.any(1), out.sum() / rl.sum()

    M = moments(x)
    R0 = net.run(x, keep=('z', 'logits')); z0, L0 = R0['z'], R0['logits']; c0 = L0.argmax(1)
    rng = np.random.default_rng(SEED); perm = rng.permutation(J)
    tr, te = np.sort(perm[:J // 2]), np.sort(perm[J // 2:]); trf, trv = tr[: len(tr) * 3 // 4], tr[len(tr) * 3 // 4:]
    ridge = Ridge(z0, M['logm'], trf, trv, tr); lm0 = ridge(z0[te])
    w0, o0, of0 = flags(x)
    res['baseline'] = dict(acc=float((c0 == y).mean()), W_to_Z=float((c0[y == W] == Z).mean()), Z_to_W=float((c0[y == Z] == W).mean()),
                           own_class=[float((c0[y == k] == k).mean()) for k in range(5)],
                           confusion=np.array([[int(((y == a) & (c0 == b)).sum()) for b in range(5)] for a in range(5)]).tolist(),
                           ridge_r2_te=r2(lm0, M['logm'][te]), ridge_alpha=float(ridge.alpha),
                           M1sq_over_M0M2_median=float(np.median((M['M1e'] ** 2 + M['M1p'] ** 2) / (M['M0'] * M['M2']))),
                           frac_jets_wrap=float(w0.mean()), frac_jets_beyond_train99=float(o0.mean()), frac_particles_beyond_train99=float(of0))
    log(f'baseline: {res["baseline"]} ({time.time() - t0:.0f} s)')
    A, ainfo = axis_set(x); res['axes_info'] = ainfo
    log(f'axes: {ainfo}')
    res['axes'] = {}
    for a in AXES:
        ae, ap = A[a]; xa = reaxis(x, ae, ap); Ma = moments(xa)
        disp = np.hypot(ae, ap); ratio = (Ma['M1e'] ** 2 + Ma['M1p'] ** 2) / (Ma['M0'] * Ma['M2'])
        pred = 0.5 * np.log((Ma['M0'] * Ma['M2']) / (M['M0'] * M['M2']))
        Ra = net.run(xa, keep=('z', 'logits')); z, L = Ra['z'], Ra['logits']; c = L.argmax(1)
        dl = ridge(z[te]) - lm0; pt_ = pred[te]
        slope = float(np.polyfit(pt_, dl, 1)[0]) if pt_.std() > 1e-12 else None
        corr = float(np.corrcoef(pt_, dl)[0, 1]) if pt_.std() > 1e-12 and dl.std() > 1e-12 else None
        flow = np.zeros((5, 5), int); np.add.at(flow, (c0, c), 1)
        wa, oa, ofa = flags(xa)
        bins = []
        for lo, hi in zip(DBINS[:-1], DBINS[1:]):
            s = (disp >= lo) & (disp < hi); st = s[te]
            wz = s & ((y == W) | (y == Z)); oth = s & ~((y == W) | (y == Z))
            bins.append(dict(lo=lo, hi=hi, n=int(s.sum()), changed=float((c[s] != c0[s]).mean()) if s.any() else None,
                             n_class=[int((s & (y == k)).sum()) for k in range(5)],
                             n_WZ=int(wz.sum()), changed_WZ=float((c[wz] != c0[wz]).mean()) if wz.any() else None,
                             n_gqt=int(oth.sum()), changed_gqt=float((c[oth] != c0[oth]).mean()) if oth.any() else None,
                             n_te=int(st.sum()), dlogm_hat=float(dl[st].mean()) if st.any() else None,
                             pred=float(pt_[st].mean()) if st.any() else None))
        e = dict(disp_median=float(np.median(disp)), disp_p90=float(np.quantile(disp, 0.9)), disp_mean=float(disp.mean()),
                 disp_median_class=[float(np.median(disp[y == k])) for k in range(5)],
                 M1sq_over_M0M2_median=float(np.median(ratio)), M1sq_over_M0M2_mean=float(ratio.mean()),
                 max_abs_dlogm_true=float(np.abs(Ma['logm'] - M['logm'])[M['m'] > 1].max()),
                 padding_unchanged=bool(np.array_equal(xa[x[..., 0] <= 0], x[x[..., 0] <= 0])),
                 pt_unchanged=bool(np.array_equal(xa[..., 0], x[..., 0])),
                 max_abs_ddeta=float(np.abs(xa[..., 1] - x[..., 1]).max()), max_abs_ddphi=float(np.abs(xa[..., 2] - x[..., 2]).max()),
                 acc=float((c == y).mean()), changed=float((c != c0).mean()),
                 changed_class=[float((c[y == k] != c0[y == k]).mean()) for k in range(5)],
                 own_class=[float((c[y == k] == k).mean()) for k in range(5)],
                 W_to_Z=float((c[y == W] == Z).mean()), Z_to_W=float((c[y == Z] == W).mean()),
                 W_to_W=float((c[y == W] == W).mean()), Z_to_Z=float((c[y == Z] == Z).mean()),
                 dlogit_WZ_on_W=float(((L[:, W] - L[:, Z]) - (L0[:, W] - L0[:, Z]))[y == W].mean()),
                 max_abs_dlogit=float(np.abs(L - L0).max()), max_abs_dz=float(np.abs(z - z0).max()),
                 flow=flow.tolist(), confusion=np.array([[int(((y == i) & (c == j)).sum()) for j in range(5)] for i in range(5)]).tolist(),
                 dlogm_hat=float(dl.mean()), dlogm_hat_W=float(dl[y[te] == W].mean()), dlogm_hat_Z=float(dl[y[te] == Z].mean()),
                 pred=float(pt_.mean()), pred_median=float(np.median(pt_)), pred_W=float(pt_[y[te] == W].mean()), pred_Z=float(pt_[y[te] == Z].mean()),
                 ratio=float(dl.mean() / pt_.mean()) if abs(pt_.mean()) > 1e-9 else None, perjet_slope=slope, perjet_corr=corr,
                 frac_jets_wrap=float(wa.mean()), frac_jets_beyond_train99=float(oa.mean()), frac_particles_beyond_train99=float(ofa),
                 bins=bins)
        res['axes'][a] = e
        log(f'axis {a}: ' + ', '.join(f'{k} {v:.4g}' if isinstance(v, float) else f'{k} {v}' for k, v in e.items() if k not in ('bins', 'flow', 'confusion')))
        log(f'   flow (rows: call on stored axis {CL}; cols: call on {a} axis): {flow.tolist()}')
    name = 'e17_axis.json' if PER == 4000 else f'e17_axis_{PER}.json'
    json.dump(res, open(OUT / name, 'w'), indent=1)
    log(f'wrote {OUT / name}; E17 DONE ({time.time() - t0:.0f} s)')
    return name


# ------------------------------------------------------------------------------------------------ figures
SURF, INK, INK2, GRID = '#fcfcfb', '#1f1f1e', '#5f5e59', '#e4e3df'
PAL = dict(centroid='#2a78d6', wta='#eb6834', hardest='#1baf7a')
LABEL = dict(stored='stored axis (identity)', centroid='pT centroid of 64 stored', escheme='E-scheme of 128 stored',
             wta='WTA (C/A, WTA recomb.)', hardest='hardest particle')


def plot(name='e17_axis.json'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    R = json.loads((OUT / name).read_text()); sfx = name[len('e17_axis'):-len('.json')]
    P = json.loads((RES / 'part' / 'p1v_verify.json').read_text())
    pct = lambda v: f'{100 * v:.1f}'
    sg = lambda v: '0' if abs(v) < 5e-5 else (f'{v:+.3f}' if abs(v) >= 0.01 else f'{v:+.4f}').replace('-', '−')
    ds = lambda v: f'{v:.3f}' if v >= 0.01 else f'{v:.4f}'
    rt = lambda v: f'{v:.3f}' if v >= 0.01 else f'{v:.4f}'
    rows = []          # (group, axis label, disp, M1 ratio, pred, dlogm, acc, changed, W→Z, Z→W)
    b = R['baseline']
    for a in AXES:
        e = R['axes'][a]
        rows.append(('JEDI-linear 3f n64', LABEL[a], ds(e['disp_median']) if a != 'stored' else '0',
                     rt(e['M1sq_over_M0M2_median']), sg(e['pred']), sg(e['dlogm_hat']), pct(e['acc']), pct(e['changed']),
                     f'{b["W_to_Z"]:.3f} → {e["W_to_Z"]:.3f}', f'{b["Z_to_W"]:.3f} → {e["Z_to_W"]:.3f}'))
    for n, nm in (('full', 'ParT_full'), ('kin', 'ParT_kin')):
        B = P['models'][n]['baseline']
        for a in ('stored', 'escheme', 'wta', 'hardest'):
            e = P['models'][n]['check4'][a]; info = P['axes'][a]['info']
            rows.append((nm, LABEL[a], ds(info['disp_median']) if a != 'stored' else '0',
                         rt(info['M1sq_over_M0M2_median']), sg(e['pred']), sg(e['dlogm_hat']), pct(e['acc']), pct(e['changed']),
                         f'{B["W_to_Z"]:.3f} → {e["W_to_Z"]:.3f}', f'{B["Z_to_W"]:.3f} → {e["Z_to_W"]:.3f}'))
    head = ['model', 'axis', 'axis shift\nmedian [rad]', '|M1|²/(M0·M2)\nmedian', '½Δlog(M0·M2)\nnew vs stored axis',
            'Δlog m̂\n(ridge read-out)', 'accuracy\n[%]', 'calls\nchanged [%]', 'W → Z\n(stored → new)', 'Z → W\n(stored → new)']
    xs = [0.005, 0.115, 0.29, 0.38, 0.475, 0.575, 0.665, 0.74, 0.825, 0.93]
    al = ['left', 'left'] + ['center'] * 8
    fig = plt.figure(figsize=(15, 6.4), facecolor=SURF); ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off'); ax.set_xlim(0, 1)
    nr = len(rows); top = 0.81; dy = 0.06; ax.set_ylim(0, 1)
    ax.text(0.005, 0.965, 'Jet-axis convention: same particles, (Δη, Δφ) recomputed about another axis', fontsize=14, color=INK, va='top')
    ax.text(0.005, 0.925, f'JEDI-linear: {R["N"]:,} hls4ml test jets (5 classes, seed {R["seed"]}).  ParT: 10,000 JetClass test jets '
            f'(10 classes), P1v check 4.\nΔlog m̂: ridge read-out of log m (JEDI: on z, 16 units; ParT: on the class token, 128), '
            f'fitted on unedited jets of the train half; Δlog m̂ and ½Δlog(M0·M2) are means over the test half.', fontsize=10, color=INK2, va='top')
    for x, h, a in zip(xs, head, al):
        ax.text(x, top, h, fontsize=10, color=INK2, ha=a, va='center', fontweight='bold')
    ax.plot([0.005, 0.995], [top - 0.04, top - 0.04], color=INK2, lw=0.8)
    for i, r in enumerate(rows):
        yy = top - 0.04 - dy * (i + 0.6)
        if i % 4 == 0 and i:
            ax.plot([0.005, 0.995], [yy + dy * 0.5, yy + dy * 0.5], color=GRID, lw=1.2)
        for j, (x, v, a) in enumerate(zip(xs, r, al)):
            if j == 0 and i % 4: continue
            bold = j in (7,) and r[1].startswith('WTA')
            ax.text(x, yy, v, fontsize=10.5, color=INK if j != 1 else INK2, ha=a, va='center', fontweight='bold' if (j == 0 or bold) else 'normal')
    fig.savefig(OUT / f'e17_axis{sfx}.png', dpi=130, facecolor=SURF); plt.close(fig); log('wrote', OUT / f'e17_axis{sfx}.png')

    # binned figure: calls changed and Δlog m̂ vs axis displacement (JEDI)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURF)
    for a_ in axs:
        a_.set_facecolor(SURF); a_.grid(True, color=GRID, lw=0.8); a_.set_axisbelow(True)
        for s in ('top', 'right'): a_.spines[s].set_visible(False)
        for s in ('left', 'bottom'): a_.spines[s].set_color(GRID)
        a_.tick_params(colors=INK2, labelsize=9.5); a_.set_xscale('log')
        a_.set_xlabel('axis displacement from the stored axis [rad]', color=INK2, fontsize=10.5)
    mk = dict(centroid='o', wta='s', hardest='^')
    xb = lambda q: np.sqrt(max(q['lo'], 1e-3) * min(q['hi'], 0.4))
    for a in ('centroid', 'wta', 'hardest'):
        B = [q for q in R['axes'][a]['bins'] if q['n'] >= 50]
        Bw = [q for q in B if q.get('n_WZ', 0) >= 50]; Bo = [q for q in B if q.get('n_gqt', 0) >= 50]
        axs[0].plot([xb(q) for q in Bw], [100 * q['changed_WZ'] for q in Bw], color=PAL[a], lw=2, marker=mk[a], ms=7, mfc=SURF, mew=1.8,
                    label=f'{LABEL[a]}: W and Z jets')
        axs[0].plot([xb(q) for q in Bo], [100 * q['changed_gqt'] for q in Bo], color=PAL[a], lw=1.5, ls='--', marker=mk[a], ms=5,
                    mfc=SURF, mew=1.2, label=f'{LABEL[a]}: g, q, t jets')
        Bt = [q for q in B if q['n_te'] >= 25]
        xt = [xb(q) for q in Bt]
        axs[1].plot(xt, [q['dlogm_hat'] for q in Bt], color=PAL[a], lw=2, marker=mk[a], ms=7, mfc=SURF, mew=1.8, label=f'Δlog m̂, {LABEL[a]}')
        axs[1].plot(xt, [q['pred'] for q in Bt], color=PAL[a], lw=1.5, ls='--')
    axs[0].set_ylabel('calls changed [%]', color=INK2, fontsize=10.5)
    axs[0].set_title('JEDI calls changed, by axis displacement', color=INK, fontsize=11.5, loc='left')
    axs[0].legend(frameon=False, fontsize=8.5, loc='upper left', labelcolor=INK)
    axs[1].set_ylabel('mean Δlog m̂ (solid) and ½Δlog(M0·M2) (dashed)', color=INK2, fontsize=10.5)
    axs[1].set_title('JEDI read-out of log m vs the M0·M2 prediction', color=INK, fontsize=11.5, loc='left')
    axs[1].plot([], [], color=INK2, lw=1.5, ls='--', label='dashed: ½Δlog(M0·M2), same colour')
    axs[1].legend(frameon=False, fontsize=9.5, loc='upper left', labelcolor=INK)
    fig.text(0.01, 0.005, f'{R["N"]:,} jets; points: bins with ≥ 50 jets of that group (right: all classes, ≥ 25 test-half jets); bin edges {list(DBINS[1:-1])} rad',
             color=INK2, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT / f'e17_axis_bins{sfx}.png', dpi=140, facecolor=SURF); plt.close(fig)
    log('wrote', OUT / f'e17_axis_bins{sfx}.png')


if __name__ == '__main__':
    if '--plot' in sys.argv:
        args = [a for a in sys.argv[1:] if a != '--plot']; plot(args[0] if args else 'e17_axis.json'); sys.exit(0)
    PER = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    nm = main(PER)
    if (RES / 'part' / 'p1v_verify.json').exists(): plot(nm)
    else: log(f'no {RES / "part" / "p1v_verify.json"}: figures skipped')
