"""E6 verification: re-test of the translation claim (A4: the network's mass variable follows √(M0·M2), no −|M1|² term).

Checks (fresh sample, 2,000 test jets per class, seed 17; exact fixed-point network; CPU):
 1 exactness: m recomputed from the edited particles (massless four-vectors; pairwise form checked on a subset); padded
   slots (pT = 0) are not translated (the worker's edit_shift masked them; the unmasked variant is run once for the record).
 2 translation δ = 0.03, 0.06 along η and φ: W→Z fraction, Δ of the ridge read-out of log m per stage (fitted on the
   unedited train half, applied to the test half), prediction ½Δlog M2 from the edited particles; per-jet slope and R².
 3 dose–response δ ∈ {0, 0.01, 0.02, 0.03, 0.045, 0.06, 0.09} along +η vs the √(M0M2), G1 and √Q2 predictions.
 4 directions ±η, ±φ, ±diagonal at 0.03 / 0.06.
 5 input range: the l1 input quantizer is fixed point with WRAP overflow and per-slot ranges; jets with any particle outside
   the quantizer range or outside the training 99 % |Δη|, |Δφ| after the shift are flagged and the response compared in /
   out of range.
 6 the nine angle-only l1 units (zero pT weight): shift of their pooled P values at δ = 0.03 / 0.06; P frozen at the
   unedited values (run_jetwise override) and the read-out / calls remeasured.
Run:  PYTHONPATH=. python interp/e6_verify.py           (writes results/jedi/e6_verify.json, logs/e6_verify.log)
      python interp/e6_verify.py --plot                  (reads the json; writes results/jedi/e6_verify_translation.png;
                                                          needs matplotlib — run with an interpreter that has it)
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# outputs: AI_RESULTS/jedi/ (repo results/ if unset, as the other interp scripts); log AI_LOGS/ (repo logs/ if unset);
# data / models through jetdistill.config (JETDISTILL_DATA, JETDISTILL_MODELS, JETDISTILL_RESULTS)
OUT_DIR = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
OUT_JSON = OUT_DIR / 'e6_verify.json'
OUT_PNG = OUT_DIR / 'e6_verify_translation.png'
OUT_LOG = Path(os.environ.get('AI_LOGS', ROOT / 'logs')) / 'e6_verify.log'
SEED = 17; PER = 2000
CL = ['g', 'q', 'W', 'Z', 't']; W, Z, T = 2, 3, 4
STG = ('S', 'h4', 'h5', 'z'); KEEP = ('P',) + STG + ('logits',)
DOSE = (0.01, 0.02, 0.03, 0.045, 0.06, 0.09)
PAL = {'S': '#2a78d6', 'h4': '#eb6834', 'h5': '#1baf7a', 'z': '#eda100'}


# ---------------------------------------------------------------- plot (separate mode: needs matplotlib)
def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    R = json.loads(OUT_JSON.read_text()); D = R['dose']
    d = np.array([0.0] + list(DOSE)); surf, ink, ink2, grid = '#fcfcfb', '#1f1f1e', '#5f5e59', '#e4e3df'
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=surf)
    for a in ax:
        a.set_facecolor(surf); a.grid(True, color=grid, lw=0.8); a.set_axisbelow(True)
        for s in ('top', 'right'): a.spines[s].set_visible(False)
        for s in ('left', 'bottom'): a.spines[s].set_color(grid)
        a.tick_params(colors=ink2, labelsize=9); a.set_xlabel('translation δ along η  [rad]', color=ink2, fontsize=10)
        a.set_xlim(-0.002, 0.1)
    mk = {'S': 'o', 'h4': 's', 'h5': '^', 'z': 'D'}
    pred = np.array([0.0] + [D[f'{x}']['pred_sqrtM0M2'] for x in DOSE])
    g1 = np.array([0.0] + [D[f'{x}']['pred_G1'] for x in DOSE])
    ax[0].plot(d, pred, color=ink, lw=2, ls='--', label='prediction √(M0·M2)')
    ax[0].plot(d, g1, color=ink2, lw=1.5, ls=':', label='prediction G1 = Σ pT|r|')
    ax[0].axhline(0, color=ink2, lw=1, ls='-.')
    for st in STG:
        v = np.array([0.0] + [D[f'{x}']['dlogmhat'][st] for x in DOSE])
        ax[0].plot(d, v, color=PAL[st], lw=2, marker=mk[st], ms=6, mfc=surf, mew=1.8, label=f'read-out at {st}')
        ax[0].annotate(st, (d[-1], v[-1]), xytext=(5, 0), textcoords='offset points', color=ink, fontsize=9, va='center')
    ax[0].annotate('√(M0·M2)', (d[-1], pred[-1]), xytext=(5, 0), textcoords='offset points', color=ink, fontsize=9, va='center')
    ax[0].annotate('G1', (d[-1], g1[-1]), xytext=(5, 0), textcoords='offset points', color=ink2, fontsize=9, va='center')
    ax[0].annotate('recentred m (0)', (0.075, 0.0), xytext=(0, 5), textcoords='offset points', color=ink2, fontsize=8.5)
    ax[0].set_ylabel('mean Δ log m̂  (test half)', color=ink2, fontsize=10)
    ax[0].set_title('Read-out of log m under a rigid translation (m unchanged)', color=ink, fontsize=11, loc='left')
    ax[0].legend(frameon=False, fontsize=8.5, loc='upper left', labelcolor=ink)
    ax[0].set_xlim(-0.002, 0.112)
    wz = np.array([R['calls_unedited']['W_to_Z']] + [D[f'{x}']['W_to_Z'] for x in DOSE])
    zt = np.array([R['calls_unedited']['Z_to_t']] + [D[f'{x}']['Z_to_t'] for x in DOSE])
    ax[1].plot(d, wz, color=PAL['S'], lw=2, marker='o', ms=6, mfc=surf, mew=1.8, label='W jets called Z')
    ax[1].plot(d, zt, color=PAL['h4'], lw=2, marker='s', ms=6, mfc=surf, mew=1.8, label='Z jets called t')
    if 'pfreeze' in R:
        pf = R['pfreeze']; dd = sorted(float(k.split('_')[0]) for k in pf if k.endswith('_eta'))
        ax[1].plot([0.0] + dd, [R['calls_unedited']['W_to_Z']] + [pf[f'{k}_eta']['W_to_Z'] for k in dd], color=PAL['S'], lw=1.5,
                   ls='--', marker='o', ms=5, mfc=surf, label='W → Z, nine angle P units frozen')
    ax[1].annotate('W → Z', (d[-1], wz[-1]), xytext=(5, 0), textcoords='offset points', color=ink, fontsize=9, va='center')
    ax[1].annotate('Z → t', (d[-1], zt[-1]), xytext=(5, 0), textcoords='offset points', color=ink, fontsize=9, va='center')
    ax[1].set_ylabel('fraction of jets', color=ink2, fontsize=10); ax[1].set_ylim(-0.02, 1.02)
    ax[1].set_title('Calls under the same translation', color=ink, fontsize=11, loc='left')
    ax[1].legend(frameon=False, fontsize=8.5, loc='upper left', labelcolor=ink)
    fig.suptitle(f'JEDI-linear 64 p, {R["N"]} test jets (seed {R["seed"]}); particles shifted rigidly along η, padding fixed',
                 color=ink2, fontsize=9, x=0.01, ha='left', y=0.995)
    fig.tight_layout(); fig.savefig(OUT_PNG, dpi=150, facecolor=surf); print('wrote', OUT_PNG)


if '--plot' in sys.argv:
    plot(); sys.exit(0)

# ---------------------------------------------------------------- main
from interp.jedi_core import Jedi, sample  # noqa: E402
from jetdistill.config import DATA  # noqa: E402

OUT_LOG.parent.mkdir(parents=True, exist_ok=True); OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
_logf = open(OUT_LOG, 'w')


def log(*a):
    s = ' '.join(str(v) for v in a); print(s, flush=True); _logf.write(s + '\n'); _logf.flush()


t0 = time.time()
x, y, Q = sample(per_class=PER, seed=SEED); x = x.astype(np.float32); J = len(x)
net = Jedi(64)
res = dict(N=J, per_class=PER, seed=SEED)
log(f'E6 verify: {J} jets ({PER}/class, seed {SEED})')


def moments(xx):
    xx = np.asarray(xx, np.float64); pt, eta, phi = xx[..., 0], xx[..., 1], xx[..., 2]
    r2 = eta ** 2 + phi ** 2; r = np.sqrt(r2)
    M0 = pt.sum(1); M1e = (pt * eta).sum(1); M1p = (pt * phi).sum(1); M2 = (pt * r2).sum(1)
    p4 = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), pt * np.cosh(eta)], -1).sum(1)
    m = np.sqrt(np.maximum(p4[:, 3] ** 2 - (p4[:, :3] ** 2).sum(1), 0.0))
    return dict(M0=M0, M1e=M1e, M1p=M1p, M2=M2, m=m, logm=np.log(np.maximum(m, 1e-3)), G1=(pt * r).sum(1),
                Q2=(pt ** 2 * r2).sum(1), m2a=M0 * M2 - M1e ** 2 - M1p ** 2)


def m2_pairwise(xx):
    """m² = Σ_{i<j} 2 pT_i pT_j (cosh Δη_ij − cos Δφ_ij) (massless); O(n²) per jet, use on a subset"""
    xx = np.asarray(xx, np.float64); pt, eta, phi = xx[..., 0], xx[..., 1], xx[..., 2]
    de = eta[:, :, None] - eta[:, None, :]; dp = phi[:, :, None] - phi[:, None, :]
    pp = pt[:, :, None] * pt[:, None, :] * (np.cosh(de) - np.cos(dp))
    return pp.sum((1, 2))        # full double sum = 2 Σ_{i<j}


def edit_shift(xx, d_eta, d_phi, mask_padding=True):
    e = xx.copy(); rl = (e[..., 0] > 0) if mask_padding else np.ones(e.shape[:2], bool)
    e[..., 1] += np.float32(d_eta) * rl; e[..., 2] += np.float32(d_phi) * rl; return e


M = moments(x)
R0 = net.run(x, keep=KEEP); call0 = R0['logits'].argmax(1)
res['acc'] = float((call0 == y).mean()); log(f'accuracy {res["acc"]:.4f}   ({time.time()-t0:.0f}s)')


def calls(c):
    return dict(W_to_Z=float((c[y == W] == Z).mean()), W_to_W=float((c[y == W] == W).mean()), W_to_t=float((c[y == W] == T).mean()),
                Z_to_Z=float((c[y == Z] == Z).mean()), Z_to_t=float((c[y == Z] == T).mean()), Z_to_W=float((c[y == Z] == W).mean()),
                t_to_t=float((c[y == T] == T).mean()), acc=float((c == y).mean()), W_changed=float((c[y == W] != call0[y == W]).mean()))


res['calls_unedited'] = calls(call0)

# ---------------------------------------------------------------- 1. exactness and padding
sub = np.arange(0, J, 25)
m2p = m2_pairwise(x[sub]); res['exact'] = dict(pairwise_vs_4vector_max_rel=float(np.max(np.abs(m2p - M['m'][sub] ** 2) / np.maximum(M['m'][sub] ** 2, 1.0))))
xe = edit_shift(x, 0.06, 0.0); Me = moments(xe); pad = x[..., 0] <= 0
res['exact'].update(dlogm_eta006_max=float(np.max(np.abs(Me['logm'] - M['logm'])[M['m'] > 1])),
                    dlogm_eta006_mean=float(np.mean(Me['logm'] - M['logm'])),
                    padding_slots_fraction=float(pad.mean()),
                    padding_moved_max=float(np.max(np.abs(xe[pad][:, 1:] - x[pad][:, 1:]))),
                    padding_input_values=dict(raw=[0.0, 0.0, 0.0], normalised=((np.zeros(3) - net.shift) / net.scale).tolist()),
                    worker_edit_masked_padding=True,
                    note='worker edit_shift: rl = e[...,0] > 0; e[...,1] += d_eta*rl (padding fixed at 0); same here')
log('1 exactness: pairwise m² vs four-vector m² max rel diff {pairwise_vs_4vector_max_rel:.2e}; translation η+0.06: max |Δlog m| '
    '{dlogm_eta006_max:.2e}, mean {dlogm_eta006_mean:+.2e}; padded slots {padding_slots_fraction:.3f} of slots, max moved '
    '{padding_moved_max:.1e}; padding raw (0,0,0) -> normalised {padding_input_values[normalised]}'.format(**res['exact']))
xu = edit_shift(x, 0.06, 0.0, mask_padding=False)
Ru = net.run(xu, keep=('logits',))
res['exact']['padding_translated_variant'] = dict(calls=calls(Ru['logits'].argmax(1)))
log(f'   (for the record) padding ALSO translated by 0.06 along η: W->Z {res["exact"]["padding_translated_variant"]["calls"]["W_to_Z"]:.3f}, '
    f'acc {res["exact"]["padding_translated_variant"]["calls"]["acc"]:.3f}')

# ---------------------------------------------------------------- readout of log m (ridge, unedited train half)
rng = np.random.default_rng(SEED); perm = rng.permutation(J)
tr, te = np.sort(perm[:J // 2]), np.sort(perm[J // 2:]); trf, trv = tr[: len(tr) * 3 // 4], tr[len(tr) * 3 // 4:]
ALPHAS = 10.0 ** np.arange(-4, 4.1, 0.5)


class Ridge:
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


readers = {st: Ridge(R0[st], M['logm'], trf, trv, tr) for st in STG}
base = {st: readers[st](R0[st][te]) for st in STG}
res['readout_r2_logm'] = {st: r2(base[st], M['logm'][te]) for st in STG}
log('readout log m held-out R²: ' + ', '.join(f'{st} {v:.3f}' for st, v in res['readout_r2_logm'].items()))

# ---------------------------------------------------------------- input ranges (check 5 preparation)
q = net.L['l1'].iq.quantizer
kq = np.asarray(q.k, np.float64).reshape(64, 3); iq = np.asarray(q.i, np.float64).reshape(64, 3); fq = np.asarray(q.f, np.float64).reshape(64, 3)
QLO = np.where(kq > 0, -2.0 ** iq, 0.0); QHI = 2.0 ** iq - 2.0 ** (-fq)          # normalised units, per (slot, feature)
Xtr = np.load(DATA / 'train_ptetaphi_f16.npy', mmap_mode='r')
tri = np.sort(np.random.default_rng(1).choice(Xtr.shape[0], 20000, replace=False)); A = np.asarray(Xtr[tri][:, :64]).astype(np.float32)
real = A[..., 0] > 0
TR99 = dict(eta=float(np.percentile(np.abs(A[..., 1][real]), 99)), phi=float(np.percentile(np.abs(A[..., 2][real]), 99)))
TR01 = dict(eta=float(np.percentile(np.abs(A[..., 1][real]), 1)), phi=float(np.percentile(np.abs(A[..., 2][real]), 1)))
del A
res['input_range'] = dict(quantizer=dict(overflow_mode=str(np.asarray(q.overflow_mode).ravel()[0]) if hasattr(q, 'overflow_mode') else 'WRAP',
                                         eta_range_rad_slot0=[float(QLO[0, 1] * net.scale[1] + net.shift[1]), float(QHI[0, 1] * net.scale[1] + net.shift[1])],
                                         eta_range_rad_slots1to8=[float(QLO[1, 1] * net.scale[1] + net.shift[1]), float(QHI[1, 1] * net.scale[1] + net.shift[1])],
                                         phi_range_rad_slot0=[float(QLO[0, 2] * net.scale[2] + net.shift[2]), float(QHI[0, 2] * net.scale[2] + net.shift[2])],
                                         phi_range_rad_min_hi=float((QHI[:, 2] * net.scale[2] + net.shift[2]).min()),
                                         i_bits_eta=iq[:, 1].tolist(), i_bits_phi=iq[:, 2].tolist(), f_bits_eta=fq[:, 1].tolist(), f_bits_phi=fq[:, 2].tolist()),
                          train_abs_p99=TR99, train_abs_p01=TR01, inputs_clip=False,
                          note='jetdistill.data.inputs: (x - shift)/scale, no clipping; l1 input quantizer: fixed point per slot, WRAP overflow')
log(f'5 input range: quantizer WRAP; η range slot0 {res["input_range"]["quantizer"]["eta_range_rad_slot0"]} rad, slots1-8 '
    f'{res["input_range"]["quantizer"]["eta_range_rad_slots1to8"]}; φ slot0 {res["input_range"]["quantizer"]["phi_range_rad_slot0"]}; '
    f'training |Δη| 1/99% {TR01["eta"]:.4f}/{TR99["eta"]:.3f}, |Δφ| {TR01["phi"]:.4f}/{TR99["phi"]:.3f}')


def flags(xx):
    """per jet: any slot outside the quantizer range (wrap), any real particle outside the training 99 % |angle|"""
    xn = (np.asarray(xx, np.float64) - net.shift) / net.scale
    wrap = ((xn < QLO) | (xn > QHI)).any((1, 2))
    rl = xx[..., 0] > 0
    out = ((np.abs(xx[..., 1]) > TR99['eta']) & rl).any(1) | ((np.abs(xx[..., 2]) > TR99['phi']) & rl).any(1)
    return wrap, out


wrap0, out0 = flags(x)
res['input_range']['unedited'] = dict(frac_wrap=float(wrap0.mean()), frac_outside_train99=float(out0.mean()))
log(f'   unedited jets: any-slot-wraps {wrap0.mean():.4f}, any particle beyond train 99% {out0.mean():.3f}')

# ---------------------------------------------------------------- P units (check 6 preparation)
K1, b1 = net.KB['l1']
ANG = [int(u) for u in np.flatnonzero((np.abs(K1[0]) < 1e-9) & (np.abs(K1[1:]).sum(0) > 1e-9))]
PTU = [int(u) for u in np.flatnonzero(np.abs(K1[0]) > 1e-9)]
CON = [int(u) for u in np.flatnonzero((np.abs(K1).sum(0) < 1e-9) & (np.abs(b1) > 1e-9))]
res['l1_units'] = dict(angle_only=ANG, pt_only=PTU, constant=CON, n_live=len(ANG) + len(PTU) + len(CON),
                       angle_weights={u: dict(w_eta=float(K1[1, u]), w_phi=float(K1[2, u]), b=float(b1[u])) for u in ANG})
log(f'6 l1 live units: angle-only {ANG} ({len(ANG)}), pT-only {PTU}, constant {CON}; no unit mixes pT with angle')
maskP = np.zeros(64); maskP[ANG] = 1
sdP = R0['P'].std(0)


# ---------------------------------------------------------------- measurement of one edit
def measure(xe, RR, Me, tag, wrapf=None, outf=None):
    c = RR['logits'].argmax(1); row = dict(calls=calls(c)); row.update(calls(c))
    dM2 = 0.5 * np.log(Me['M2'] / M['M2']); dG1 = np.log(Me['G1'] / M['G1']); dQ2 = 0.5 * np.log(Me['Q2'] / M['Q2'])
    row.update(pred_sqrtM0M2=float(dM2[te].mean()), pred_sqrtM0M2_median=float(np.median(dM2[te])), pred_G1=float(dG1[te].mean()),
               pred_sqrtQ2=float(dQ2[te].mean()), pred_recentred=float(np.mean(Me['logm'][te] - M['logm'][te])),
               pred_sqrtM0M2_W=float(dM2[(y == W)].mean()), pred_sqrtM0M2_WZ_te=float(dM2[te][(y[te] == W) | (y[te] == Z)].mean()))
    row['dlogmhat'] = {}; row['dlogmhat_se'] = {}; row['dlogmhat_median'] = {}; row['perjet'] = {}; row['dlogmhat_WZ'] = {}
    for st in STG:
        obs = readers[st](RR[st][te]) - base[st]; p = dM2[te]
        row['dlogmhat'][st] = float(obs.mean()); row['dlogmhat_se'][st] = float(obs.std() / np.sqrt(len(obs)))
        row['dlogmhat_median'][st] = float(np.median(obs))
        wz = (y[te] == W) | (y[te] == Z); row['dlogmhat_WZ'][st] = float(obs[wz].mean())
        A_ = np.c_[p, np.ones(len(p))]; w_ = np.linalg.lstsq(A_, obs, rcond=None)[0]
        row['perjet'][st] = dict(slope_origin=float((obs @ p) / (p @ p)), slope=float(w_[0]), intercept=float(w_[1]),
                                 r2=float(np.corrcoef(obs, p)[0, 1] ** 2), corr_G1=float(np.corrcoef(obs, dG1[te])[0, 1]),
                                 corr_sqrtQ2=float(np.corrcoef(obs, dQ2[te])[0, 1]), corr_sqrtM0M2=float(np.corrcoef(obs, p)[0, 1]),
                                 ratio_mean=float(obs.mean() / p.mean()))
    if wrapf is not None:
        inr = ~(wrapf | outf | out0); row['range'] = dict(frac_wrap=float(wrapf.mean()), frac_outside_train99=float(outf.mean()),
                                                           frac_in_range=float(inr.mean()))
        for nm, sel in (('in_range', inr), ('out_of_range', ~inr)):
            if sel[te].sum() < 50 or sel[y == W].sum() < 50:
                row['range'][nm] = dict(n_te=int(sel[te].sum())); continue
            d_ = {}
            for st in STG:
                obs = (readers[st](RR[st][te]) - base[st])[sel[te]]; p = dM2[te][sel[te]]
                d_[st] = dict(obs=float(obs.mean()), pred=float(p.mean()), ratio=float(obs.mean() / p.mean()))
            row['range'][nm] = dict(n_te=int(sel[te].sum()), n_W=int(sel[y == W].sum()), dlogmhat=d_,
                                    W_to_Z=float((c[(y == W) & sel] == Z).mean()), W_to_Z_unedited=float((call0[(y == W) & sel] == Z).mean()))
    log(f'   {tag:12s} W->Z {row["W_to_Z"]:.3f} (W changed {row["W_changed"]:.3f}) Z->t {row["Z_to_t"]:.3f} acc {row["acc"]:.3f} | '
        f'pred ½ΔlogM2 {row["pred_sqrtM0M2"]:+.3f} (med {row["pred_sqrtM0M2_median"]:+.3f}) G1 {row["pred_G1"]:+.3f} √Q2 {row["pred_sqrtQ2"]:+.3f} '
        f'm {row["pred_recentred"]:+.1e} | Δlog m̂ ' + ' '.join(f'{st} {row["dlogmhat"][st]:+.3f}' for st in STG)
        + ' | per-jet h5: slope %.2f r² %.2f (corr G1 %.2f, √Q2 %.2f)' % (row['perjet']['h5']['slope'], row['perjet']['h5']['r2'],
                                                                        row['perjet']['h5']['corr_G1'], row['perjet']['h5']['corr_sqrtQ2'])
        + (f' | wrap {row["range"]["frac_wrap"]:.4f} out99 {row["range"]["frac_outside_train99"]:.3f}' if wrapf is not None else ''))
    return row


EDITS = {}
for dd in DOSE: EDITS[f'{dd}_eta'] = (dd, 0.0)
for dd in (0.03, 0.06):
    EDITS[f'{dd}_phi'] = (0.0, dd); EDITS[f'{dd}_-eta'] = (-dd, 0.0); EDITS[f'{dd}_-phi'] = (0.0, -dd)
    EDITS[f'{dd}_diag'] = (dd / np.sqrt(2), dd / np.sqrt(2)); EDITS[f'{dd}_-diag'] = (-dd / np.sqrt(2), -dd / np.sqrt(2))
res['edits'] = {}; XE = {}; RE = {}
log('2/3/4 translations (padding fixed; free network):')
for name, (de, dp) in EDITS.items():
    xe = edit_shift(x, de, dp); XE[name] = xe; RE[name] = net.run(xe, keep=KEEP); wf, of = flags(xe)
    res['edits'][name] = measure(xe, RE[name], moments(xe), name, wf, of)
    res['edits'][name]['delta'] = dict(eta=float(de), phi=float(dp))
log(f'   ({time.time()-t0:.0f}s)')
res['dose'] = {f'{dd}': res['edits'][f'{dd}_eta'] for dd in DOSE}

# dose-response exponents (log-log slope of the mean response vs δ)
ld = np.log(np.array(DOSE))
def expo(vals, upto=0.06):
    sel = np.array(DOSE) <= upto; v = np.array(vals)[sel]; ok = v > 0
    return float(np.polyfit(ld[sel][ok], np.log(v[ok]), 1)[0]) if ok.sum() >= 3 else None
res['dose_exponent'] = dict(pred_sqrtM0M2=expo([res['dose'][f'{dd}']['pred_sqrtM0M2'] for dd in DOSE]),
                            pred_G1=expo([res['dose'][f'{dd}']['pred_G1'] for dd in DOSE]),
                            pred_sqrtQ2=expo([res['dose'][f'{dd}']['pred_sqrtQ2'] for dd in DOSE]),
                            **{st: expo([res['dose'][f'{dd}']['dlogmhat'][st] for dd in DOSE]) for st in STG})
res['dose_exponent_to009'] = dict(pred_sqrtM0M2=expo([res['dose'][f'{dd}']['pred_sqrtM0M2'] for dd in DOSE], 0.09),
                                  **{st: expo([res['dose'][f'{dd}']['dlogmhat'][st] for dd in DOSE], 0.09) for st in STG})
log('3 dose-response log-log exponent (δ ≤ 0.06): ' + ', '.join(f'{k} {v:.2f}' for k, v in res['dose_exponent'].items() if v is not None)
    + ' | to 0.09: ' + ', '.join(f'{k} {v:.2f}' for k, v in res['dose_exponent_to009'].items() if v is not None))
log('   obs/pred ratio per δ (h5): ' + ', '.join(f'{dd}: {res["dose"][f"{dd}"]["dlogmhat"]["h5"] / res["dose"][f"{dd}"]["pred_sqrtM0M2"]:.2f}' for dd in DOSE))

# ---------------------------------------------------------------- 4. directions summary
res['directions'] = {}
for dd in (0.03, 0.06):
    names = [f'{dd}_eta', f'{dd}_-eta', f'{dd}_phi', f'{dd}_-phi', f'{dd}_diag', f'{dd}_-diag']
    v = {n: res['edits'][n]['dlogmhat']['h5'] for n in names}; wz = {n: res['edits'][n]['W_to_Z'] for n in names}
    p = {n: res['edits'][n]['pred_sqrtM0M2'] for n in names}
    res['directions'][f'{dd}'] = dict(dlogmhat_h5=v, W_to_Z=wz, pred=p, spread_obs_h5=float(max(v.values()) - min(v.values())),
                                      spread_pred=float(max(p.values()) - min(p.values())),
                                      dlogmhat_all={n: res['edits'][n]['dlogmhat'] for n in names})
    log(f'4 directions δ={dd}: Δlog m̂(h5) ' + ' '.join(f'{n.split("_")[1]} {v[n]:+.3f}' for n in names) + f' (spread {res["directions"][f"{dd}"]["spread_obs_h5"]:.3f}; pred spread {res["directions"][f"{dd}"]["spread_pred"]:.3f}) | W->Z '
        + ' '.join(f'{n.split("_")[1]} {wz[n]:.3f}' for n in names))
    # even / odd parts in δ: a function of r² about the axis is even up to the 2δ·M1 term (pred odd ≈ 0)
    oe = {}
    for ax_ in ('eta', 'phi', 'diag'):
        a_, b_ = res['edits'][f'{dd}_{ax_}'], res['edits'][f'{dd}_-{ax_}']
        oe[ax_] = {st: dict(even=0.5 * (a_['dlogmhat'][st] + b_['dlogmhat'][st]), odd=0.5 * (a_['dlogmhat'][st] - b_['dlogmhat'][st]),
                            se=float(np.hypot(a_['dlogmhat_se'][st], b_['dlogmhat_se'][st]) / 2)) for st in STG}
        oe[ax_]['pred'] = dict(even=0.5 * (a_['pred_sqrtM0M2'] + b_['pred_sqrtM0M2']), odd=0.5 * (a_['pred_sqrtM0M2'] - b_['pred_sqrtM0M2']))
    res['directions'][f'{dd}']['even_odd'] = oe
    log(f'   even/odd parts in δ (h5): ' + ' '.join(f'{ax_}: even {oe[ax_]["h5"]["even"]:+.3f} odd {oe[ax_]["h5"]["odd"]:+.3f} (se {oe[ax_]["h5"]["se"]:.3f}; pred odd {oe[ax_]["pred"]["odd"]:+.4f})' for ax_ in ('eta', 'phi', 'diag')))

# ---------------------------------------------------------------- 5. range summary
for n in ('0.03_eta', '0.06_eta', '0.09_eta', '0.06_phi'):
    rg = res['edits'][n]['range']
    s = f'5 range {n}: wrap {rg["frac_wrap"]:.4f}, beyond train 99% {rg["frac_outside_train99"]:.3f}, in-range jets {rg["frac_in_range"]:.3f}'
    for nm in ('in_range', 'out_of_range'):
        r_ = rg.get(nm, {})
        if 'dlogmhat' in r_:
            s += f' | {nm} (n_te {r_["n_te"]}, n_W {r_["n_W"]}): h5 obs {r_["dlogmhat"]["h5"]["obs"]:+.3f} pred {r_["dlogmhat"]["h5"]["pred"]:+.3f} ratio {r_["dlogmhat"]["h5"]["ratio"]:.2f}; W->Z {r_["W_to_Z"]:.3f} (unedited {r_["W_to_Z_unedited"]:.3f})'
        else: s += f' | {nm}: n_te {r_.get("n_te")}'
    log(s)

# ---------------------------------------------------------------- 6. P units: shift and freeze
res['P_units'] = {}
for n in ('0.03_eta', '0.06_eta', '0.06_phi', '0.06_-eta'):
    dP = RE[n]['P'] - R0['P']
    res['P_units'][n] = {u: dict(mean_shift_sd=float(dP[:, u].mean() / sdP[u]), abs_shift_sd=float(np.abs(dP[:, u]).mean() / sdP[u])) for u in ANG}
    res['P_units'][n]['other_units_max_abs_shift_sd'] = float(max(np.abs(dP[:, u]).mean() / sdP[u] for u in range(64) if u not in ANG and sdP[u] > 1e-9))
lin = {u: res['P_units']['0.06_eta'][u]['mean_shift_sd'] / res['P_units']['0.03_eta'][u]['mean_shift_sd']
       if abs(res['P_units']['0.03_eta'][u]['mean_shift_sd']) > 1e-6 else None for u in ANG}
res['P_units']['linearity_ratio_006_over_003_eta'] = lin
log('6 P angle units mean shift at η+0.06 (sd units): ' + ' '.join(f'P{u} {res["P_units"]["0.06_eta"][u]["mean_shift_sd"]:+.2f}' for u in ANG)
    + f'; other P units max |shift| {res["P_units"]["0.06_eta"]["other_units_max_abs_shift_sd"]:.1e}; ratio 0.06/0.03: '
    + ' '.join(f'P{u} {lin[u]:.2f}' for u in ANG if lin[u] is not None))
res['pfreeze'] = {}
log('   nine angle P units frozen at unedited values (run_jetwise override on P):')
for n in ('0.03_eta', '0.06_eta', '0.09_eta', '0.03_phi', '0.06_phi'):
    RF = net.run_jetwise(XE[n], {'P': (maskP, R0['P'])}, keep=KEEP)
    res['pfreeze'][n] = measure(XE[n], RF, moments(XE[n]), f'Pfrz {n}')
RF = net.run_jetwise(XE['0.06_eta'], {'P': (np.ones(64), R0['P'])}, keep=KEEP)
res['pfreeze']['0.06_eta_allP'] = measure(XE['0.06_eta'], RF, moments(XE['0.06_eta']), 'allPfrz 0.06η')
chk = net.run_jetwise(x, {'P': (maskP, R0['P'])}, keep=('logits',))
res['pfreeze']['identity_check_max_logit_diff'] = float(np.abs(chk['logits'] - R0['logits']).max())

# ---------------------------------------------------------------- headline comparison with the worker (seed 6, 3000/class)
res['worker_headline'] = dict(dlogmhat_006=dict(S=0.288, h4=0.281, h5=0.239, z=0.253), pred=0.306, W_to_Z=0.765)
res['elapsed_s'] = float(time.time() - t0)
OUT_JSON.write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
log(f'wrote {OUT_JSON} ({time.time()-t0:.0f}s)')
