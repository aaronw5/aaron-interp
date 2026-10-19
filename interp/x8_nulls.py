"""X8 (claims A1, A3, A4, D4, A10; round-4 review weakness 12): functional nulls for the headline causal tests.

Question: do the measured JEDI-linear responses (0.06 rad translation W -> Z, WTA re-axis call changes, rank-window
across-edge angle swap, radial response of the S unit most correlated with sum pT r^2) exceed what the same tests return on
networks of the same architecture and inputs with no trained function; and do the unit costs / context-round result
(g -> mean 14.9 %) hold under resample ablation as well as mean ablation?

FROZEN ANALYSIS PLAN (written before the first run; not changed after seeing results)
Sample: interp.jedi_core.sample(per_class=2000, seed=307) = 10,000 test jets (fresh: seed 307 not used by an earlier node).
  Read-out split: rng(307).permutation(10,000): fit half = first 5,000 (alpha chosen after a fit on its first 3/4, validated on
  its last 1/4, refit on the whole fit half; E6v Ridge), test half = the other 5,000 (all read-out changes are measured there).
Models
  jedi    JEDI-linear 3-feature n64 (exact fixed-point network, interp.jedi_core.Jedi(64); own jit forward with the fused
          quantized kernels passed as arguments, checked equal to Jedi.run) and 15 weight-permuted copies (no training):
            variant a  (masks in place)  rng(s), s = 0..4: in every fused quantized kernel the non-zero entries are permuted among
                       the non-zero positions, likewise the non-zero bias entries of every layer; every input quantizer (bit
                       widths, zero-bit masks incl. the qs / qs1 rank windows) unchanged.
            variant c  (added after the 500-jet smoke run, before the full run, because variant b gave a constant-output
                       network there): kernels / biases as variant a (rng(s)); the slot-dependent input-quantizer tables of qs,
                       l2, add, l3, qs1 (slot x channel) permuted along the slot axis, independently per channel (rng(200 + s));
                       kept-slot count per channel unchanged, l1 and the jet-level layers unchanged.
            variant b  (masks permuted)  rng(s): all kernel entries permuted within the matrix, all bias entries within the layer;
                       and every input quantizer's bit tables (k, i, f: one permutation per quantizer, rng(100 + s)) permuted
                       within the table.
  dsets   dsets-d64 (train/models.py) n64: trained seeds 0, 1 (best.pt); shuffled-label seeds 0, 1 (train_small --shuffle_labels,
          30 epochs, final-epoch checkpoint, no selection on true labels); 5 weight-permuted copies of trained seed 0 (every
          nn.Linear: weight entries permuted within the matrix, bias within the layer, rng(s), s = 0..4).
  jedis   jedis-d64 (float JEDI-style with a context round g = masked mean of the first per-particle MLP): same set as dsets.
  Inputs of dsets / jedis exactly as training (x - shift) / scale from arch.json, padding 0, mask pT > 0.
Layers read: S = pooled second-round objects (JEDI S; dsets sum phi2; jedis sum u); last = last hidden layer (JEDI z, pre-ReLU;
  dsets h3; jedis h2).
Edits (all on raw particles, padding rows untouched, pT order kept)
  T   translation of every real particle by 0.06 rad along +deta (E6v)        matched null edit Tr: 0.06 rad, random direction per
                                                                               jet (rng 3071)
  W   re-axis about the WTA axis (C/A, WTA recombination; E17 reaxis)          Wr: translation by the per-jet WTA displacement |a|
                                                                               in a random direction (rng 3072)
  K   radial edit (deta, dphi) -> 1.1 (deta, dphi) at fixed pT (m x ~1.1; also the read-out calibration edit, E23)
                                                                               Kr: rotation about the axis by 2 arcsin(0.05) rad
                                                                               (equal per-particle displacement), random sign (rng 3073)
  A13 / I5, A13 / I11, A0 / I3: swap (deta, dphi) of ranks k, k+1 (across-edge 13|14 [E20, X6], 0|1 [X6]) vs the inside pair
      (5|6 [E20], 11|12 and 3|4 [X6]); jets with a particle at rank k+1 (both edits defined: common jets).
Metrics per model (10,000 jets unless stated)
  1 translation: W -> Z flow = mean over true W jets of [call_T = Z] - [call_0 = Z] (headline H1); % calls changed; calibrated
    read-out ratio = (mean dR_T / mean pred_T) / (mean dR_K / mean dlogm_K) on the test half, R = ridge read-out of log m at the
    last hidden layer fitted on unedited fit-half jets, pred = 1/2 dlog(M0 M2) (sqrt(M0 M2) prediction), dlogm_K exact from the
    massless four-vectors of the edited particles.
  2 WTA: % calls changed (headline H2), accuracy before -> after, calibrated read-out ratio (pred = 1/2 dlog M2 about WTA).
  3 windows: units for edge k = the model's live S units (mean-ablation cost >= 1 % of calls on x[::5], 2,000 jets) whose own qs1
    zero-bit mask changes between slots k and k+1; if none (no masks, or none at k), the 10 live S units of largest cost (if no
    live S unit, the 10 S units of largest s.d.). Per common jet m = mean over units |dS_c| / sd(S_c); across / inside ratio of
    means (headline H3 = A13 / I5).
  4 radial: unit u* = the S unit with the largest |Spearman rho| with sum pT r^2 on unedited jets (rule frozen; for JEDI-linear
    expected S29); per-jet response sign(rho) dS_u* / sd(S_u*) under K (headline H4), and under Kr (control).
  McNemar (exact) on paired call changes: T vs Tr, W vs Wr, K vs Kr, across vs inside; W -> Z calls under T vs Tr.
Statistics: 1,000 jet-level bootstrap resamples (the same resampled jet indices for every model, so model contrasts are paired);
  95 % percentile intervals. Null distribution per family: jedi = 15 permuted copies (a, c, b) (and, reported separately, the jedis-d64
  shuffled-label seeds); dsets / jedis = 5 permuted + 2 shuffled-label models. H0: trained effect <= null effect; H1: trained >
  null (one-sided). Per null model i: paired bootstrap p_i = (1 + #{boot (trained - null_i) <= 0}) / (1 + B); headline p = max_i
  p_i (intersection-union test: trained above every null); also the rank p = (1 + #{null point >= trained point}) / (1 + n_null)
  and the null 95 % band (2.5 / 97.5 % of the pooled bootstrap replicates of the null models). Holm across the 4 headline tests x
  3 families (trained seed 0 for dsets / jedis).
Ablation arm (JEDI-linear trained; dsets / jedis trained seed 0): every unit of P, S, z (dsets: S, h3; jedis: g units, S, h2)
  set to its mean over jets (mean ablation) -> % calls changed; live = >= 1 %. Every live unit then resample-ablated: its value
  taken from a random other jet (20 random derangements, rng 3074) -> mean % calls changed. g -> context: all 64 g units to their
  mean vs g of a random other jet (20 resamples) (JEDI-linear, jedis). Spearman rho of mean vs resample cost over the live units
  (bootstrap CI over jets); H0 per unit: resample cost <= 1 % (one-sided z on the per-jet averages), Benjamini-Hochberg over the
  live units.
Predictions. Thesis (elements are learned function): trained outside the null 95 % band on every headline test (W -> Z flow and
  calibrated read-out ratio near 0 for nulls, across / inside ratio near 1 for nulls); Spearman >= 0.8; g -> resample comparable to
  14.9 %. Alternative (architecture / input artefact): nulls give comparable flow / read-out or across-edge ratios (variant a > 1
  with variant b ~ 1 would place the window effect in the masks), or resample reorders live units / g's effect < 5 %.
Runs (cluster):
  ./job.sh x8-jedi     'python -m interp.x8_nulls --part jedi'          -> $AI_RESULTS/jedi/x8_jedi.json, .npz
  ./job.sh x8-jabl     'python -m interp.x8_nulls --part jedi-abl'      -> x8_jedi_abl.json
  AI_PY=conda ./job.sh x8-dsets 'python -m interp.x8_nulls --part small --family dsets'   -> x8_dsets.json, .npz
  AI_PY=conda ./job.sh x8-jedis 'python -m interp.x8_nulls --part small --family jedis'   -> x8_jedis.json, .npz
  ./job.sh x8-comb     'python -m interp.x8_nulls --part combine'       -> x8_summary.json, x8_nulls.png
  --smoke: 100 jets per class, 2 permuted copies, 3 resamples, 100 bootstraps (outputs *_smoke)
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
SMALL = Path(os.environ.get('AI_SMALL_MODELS', '/j-jepa-vol/aaron-interp/models/small'))
CL = ['g', 'q', 'W', 'Z', 't']; W_, Z_ = 2, 3
T0 = time.time()


def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


ap = argparse.ArgumentParser()
ap.add_argument('--part', required=True, choices=['jedi', 'jedi-abl', 'small', 'combine'])
ap.add_argument('--family', default='dsets', choices=['dsets', 'jedis'])
ap.add_argument('--smoke', action='store_true')
args = ap.parse_args() if __name__ == '__main__' else ap.parse_args(['--part', 'combine'])
SM = args.smoke
PER = 100 if SM else 2000; SEED = 307
NPERM = 2 if SM else 5; NRES = 3 if SM else 20; NB = 100 if SM else 1000
SFX = '_smoke' if SM else ''
BATCH = 5000
PAIRS = (('A13_I5', 13, 5), ('A13_I11', 13, 11), ('A0_I3', 0, 3))
EDITS = ('T', 'Tr', 'W', 'Wr', 'K', 'Kr')


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, np.ndarray): return clean(o.tolist())
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating, float)): return None if np.isnan(o) else (('inf' if o > 0 else '-inf') if np.isinf(o) else float(o))
    return o


def save_json(name, R): (OUT / f'{name}{SFX}.json').write_text(json.dumps(clean(R), indent=1))


# ================================================================================================ sample and edits
def load():
    from interp.jedi_core import sample
    x, y, _ = sample(per_class=PER, seed=SEED)
    x = np.asarray(x, np.float32); J = len(x)
    perm = np.random.default_rng(SEED).permutation(J); fit = np.sort(perm[:J // 2]); te = np.sort(perm[J // 2:])
    nf = len(fit); fit_fit, fit_val = fit[:3 * nf // 4], fit[3 * nf // 4:]
    return x, y, dict(fit=fit, fit_fit=fit_fit, fit_val=fit_val, te=te)


def moments(x):
    from interp.e17_axis_jedi import moments as m_
    return m_(x)


def translate_dir(x, d, th):
    e = x.copy(); rl = (e[..., 0] > 0)
    e[..., 1] = np.where(rl, x[..., 1] + (d * np.cos(th))[:, None].astype(np.float32), x[..., 1])
    e[..., 2] = np.where(rl, x[..., 2] + (d * np.sin(th))[:, None].astype(np.float32), x[..., 2])
    return e


def scale_ang(x, k):
    e = x.copy(); rl = e[..., 0] > 0
    e[..., 1] = np.where(rl, x[..., 1] * np.float32(k), x[..., 1]); e[..., 2] = np.where(rl, x[..., 2] * np.float32(k), x[..., 2])
    return e


def rotate(x, phi):
    rl = x[..., 0] > 0; c, s = np.cos(phi)[:, None], np.sin(phi)[:, None]; e = x.copy()
    e[..., 1] = np.where(rl, c * x[..., 1] - s * x[..., 2], x[..., 1]); e[..., 2] = np.where(rl, s * x[..., 1] + c * x[..., 2], x[..., 2])
    return e.astype(np.float32)


def swap(x, k):
    nreal = (x[..., 0] > 0).sum(1); ok = nreal > k + 1
    xe = x.copy(); xe[ok, k, 1:], xe[ok, k + 1, 1:] = x[ok, k + 1, 1:], x[ok, k, 1:]
    return ok, xe


def make_edits(x):
    from interp.e17_axis_jedi import wta_axis_one, wta_selftest, reaxis
    J = len(x); E = {}; info = {}
    assert wta_selftest()['ok']
    E['T'] = translate_dir(x, np.full(J, 0.06), np.zeros(J))
    E['Tr'] = translate_dir(x, np.full(J, 0.06), np.random.default_rng(3071).uniform(0, 2 * np.pi, J))
    pt = x[..., 0].astype(np.float64); de = x[..., 1].astype(np.float64); dp = x[..., 2].astype(np.float64); real = pt > 0
    t = time.time(); a = np.array([wta_axis_one(pt[j, real[j]], de[j, real[j]], dp[j, real[j]]) for j in range(J)])
    info['wta_seconds'] = time.time() - t
    E['W'] = reaxis(x, a[:, 0], a[:, 1]); dW = np.hypot(a[:, 0], a[:, 1]); info['wta_disp_mean'] = float(dW.mean())
    info['wta_disp_zero_frac'] = float((dW == 0).mean())
    E['Wr'] = translate_dir(x, dW, np.random.default_rng(3072).uniform(0, 2 * np.pi, J))
    E['K'] = scale_ang(x, 1.1)
    E['Kr'] = rotate(x, 2 * np.arcsin(0.05) * np.random.default_rng(3073).choice([-1.0, 1.0], J))
    SW = {}
    for k in sorted({p[1] for p in PAIRS} | {p[2] for p in PAIRS}):
        ok, xe = swap(x, k); SW[k] = (ok, xe)
    M0 = moments(x); P = {}
    for e in ('T', 'W', 'K'):
        Me = moments(E[e]); P[e] = dict(pred=0.5 * np.log(Me['M2'] / M0['M2']), dlogm=Me['logm'] - M0['logm'])
    info['pred_T_mean'] = float(P['T']['pred'].mean()); info['dlogm_T_maxabs'] = float(np.abs(P['T']['dlogm']).max())
    info['dlogm_K_mean'] = float(P['K']['dlogm'].mean()); info['pred_W_mean'] = float(P['W']['pred'].mean())
    info['dlogm_W_maxabs'] = float(np.abs(P['W']['dlogm']).max())
    Q = (pt * (de ** 2 + dp ** 2)).sum(1)
    log(f'edits: WTA {info["wta_seconds"]:.0f}s, mean |a| {dW.mean():.4f} rad; pred_T {info["pred_T_mean"]:.4f}; '
        f'max |dlogm_T| {info["dlogm_T_maxabs"]:.2e}; dlogm_K {info["dlogm_K_mean"]:.4f}; pred_W {info["pred_W_mean"]:.4f}')
    return E, SW, P, Q, info


# ================================================================================================ statistics helpers
def mcnemar(a, b):
    from scipy import stats
    a, b = np.asarray(a, bool), np.asarray(b, bool); n10 = int((a & ~b).sum()); n01 = int((~a & b).sum()); n = n10 + n01
    p = 1.0 if n == 0 else float(min(1.0, 2 * stats.binom.cdf(min(n10, n01), n, 0.5)))
    return dict(edit_only=n10, matched_only=n01, p=p)


def bh(ps):
    ps = np.asarray(ps, float); m = len(ps)
    if m == 0: return []
    o = np.argsort(ps); q = ps[o] * m / np.arange(1, m + 1); q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m); out[o] = np.minimum(q, 1.0); return out.tolist()


def holm(ps):
    idx = [i for i, p in enumerate(ps) if p is not None]; order = sorted(idx, key=lambda i: ps[i]); m = len(order)
    out = [None] * len(ps); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i])); out[i] = run
    return out


def ci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    return [float(np.quantile(v, .025)), float(np.quantile(v, .975))] if len(v) else [None, None]


def boot_idx(n, key):
    return np.random.default_rng(30700 + key).integers(0, n, size=(NB, n))


def safe_ratio(a, b):
    return a / b if (np.isfinite(a) and np.isfinite(b) and b != 0) else np.nan


def win_ratio(a, b):
    """across / inside; +inf when inside == 0 < across (the null's swap inside the window changes nothing), nan when both 0"""
    if b == 0: return np.inf if a > 0 else np.nan
    return a / b


# ================================================================================================ the tests on one model
class Model:
    """run(x, O=None, keep) -> dict of float64 arrays: logits (J, 5), last (J, d), S (J, 64) [, g, P]; O = {stage: (mask (u,),
    value (u,) or (J, u))}; qs1 = (n, 64) zero-bit mask of the S pooling (None without masks)"""


def unit_costs(m, x, stage, width, c0, live_only=None):
    R = m.run(x, keep=(stage,)); v = R[stage]; mean = v.mean(0); ch = np.zeros((width, len(x)), bool); run = []
    for c in range(width):
        if live_only is not None and c not in live_only: continue
        if np.all(v[:, c] == v[0, c]): continue      # constant on these jets: mean ablation is a no-op
        mk = np.zeros(width); mk[c] = 1
        ch[c] = m.run(x, O={stage: (mk, mean)})['logits'].argmax(1) != c0; run.append(c)
    return ch, mean, run


def measure(m, x, y, sp, E, SW, P, Q, label, keep_units=()):
    """all four tests on model m; returns (summary dict, per-jet arrays dict)"""
    t = time.time(); J = len(x); A = {}; S = dict(label=label)
    R0 = m.run(x); c0 = R0['logits'].argmax(1); S['acc'] = float((c0 == y).mean())
    S['call_frac'] = (np.bincount(c0, minlength=5) / J).tolist()
    sd = R0['S'].std(0)
    # live S units (mean ablation on x[::5])
    xs = x[::5]; c0s = c0[::5]
    chS, _, _ = unit_costs(m, xs, 'S', R0['S'].shape[1], c0s); costS = chS.mean(1)
    liveS = [int(c) for c in np.flatnonzero(costS >= 0.01)]; S['liveS'] = liveS; S['costS'] = costS.round(5).tolist()
    # ridge read-out of log m at the last hidden layer
    from interp.e17_axis_jedi import Ridge, r2
    logm = moments(x)['logm']; rd = Ridge(R0['last'], logm, sp['fit_fit'], sp['fit_val'], sp['fit'])
    te = sp['te']; S['readout_r2_test'] = r2(rd(R0['last'])[te], logm[te]); S['ridge_alpha'] = rd.alpha
    r0 = rd(R0['last'])
    Re = {}
    for e in EDITS:
        Re[e] = m.run(E[e]); ce = Re[e]['logits'].argmax(1); A[f'ch_{e}'] = ce != c0
        A[f'Z_{e}'] = (ce == Z_)
        S[e] = dict(calls_changed=float((ce != c0).mean()), acc=float((ce == y).mean()),
                    W_to_Z=float((ce[y == W_] == Z_).mean()))
    S['W_to_Z_unedited'] = float((c0[y == W_] == Z_).mean())
    A['flow_T'] = (A['Z_T'].astype(float) - (c0 == Z_))[y == W_]; A['flow_Tr'] = (A['Z_Tr'].astype(float) - (c0 == Z_))[y == W_]
    S['T']['W_to_Z_flow'] = float(A['flow_T'].mean()); S['Tr']['W_to_Z_flow'] = float(A['flow_Tr'].mean())
    S['T']['mcnemar_calls_vs_Tr'] = mcnemar(A['ch_T'], A['ch_Tr'])
    S['T']['mcnemar_WtoZ_vs_Tr'] = mcnemar(A['Z_T'][y == W_], A['Z_Tr'][y == W_])
    S['W']['mcnemar_calls_vs_Wr'] = mcnemar(A['ch_W'], A['ch_Wr'])
    S['K']['mcnemar_calls_vs_Kr'] = mcnemar(A['ch_K'], A['ch_Kr'])
    S['W']['acc_unedited'] = S['acc']
    A['corr0'] = (c0 == y); A['corr_W'] = Re['W']['logits'].argmax(1) == y
    for e in ('T', 'W', 'K'):
        A[f'dR_{e}'] = (rd(Re[e]['last']) - r0)[te]; A[f'pred_{e}'] = P[e]['pred'][te]; A[f'dlogm_{e}'] = P[e]['dlogm'][te]
    kcal = safe_ratio(A['dR_K'].mean(), A['dlogm_K'].mean()); S['readout_kcal'] = kcal
    for e in ('T', 'W'):
        raw = safe_ratio(A[f'dR_{e}'].mean(), A[f'pred_{e}'].mean())
        S[e]['readout_obs_over_pred'] = raw; S[e]['readout_cal_ratio'] = safe_ratio(raw, kcal)
        S[e]['readout_dR_mean'] = float(A[f'dR_{e}'].mean())
    # windows
    qs1 = getattr(m, 'qs1', None); S['windows'] = {}
    order = sorted(liveS, key=lambda c: -costS[c])
    for name, k, j in PAIRS:
        units = [c for c in liveS if qs1 is not None and qs1[k, c] != qs1[k + 1, c]]; rule = 'live units with a qs1 edge at k|k+1'
        if not units:
            units = order[:10]; rule = '10 live S units of largest cost (no qs1 edge at k|k+1)'
        if not units:
            units = [int(c) for c in np.argsort(-sd)[:10]]; rule = '10 S units of largest sd (no live S unit)'
        units = [c for c in units if sd[c] > 0]
        okA, xa = SW[k]; okI, xi = SW[j]; com = okA & okI
        Sa = m.run(xa[com]); Si = m.run(xi[com])
        if units:
            ma = (np.abs(Sa['S'][:, units] - R0['S'][com][:, units]) / sd[units]).mean(1)
            mi = (np.abs(Si['S'][:, units] - R0['S'][com][:, units]) / sd[units]).mean(1)
        else:
            ma = mi = np.zeros(int(com.sum()))
        A[f'win_a_{name}'] = ma; A[f'win_i_{name}'] = mi
        ca = Sa['logits'].argmax(1) != c0[com]; cI = Si['logits'].argmax(1) != c0[com]
        per_unit = {int(c): dict(across=float((np.abs(Sa['S'][:, c] - R0['S'][com][:, c])).mean() / sd[c]),
                                 inside=float((np.abs(Si['S'][:, c] - R0['S'][com][:, c])).mean() / sd[c]))
                    for c in sorted(set(units) | set(u for u in keep_units if sd[u] > 0))}
        S['windows'][name] = dict(edge=k, inside_pair=j, units=units, rule=rule, N=int(com.sum()), across=float(ma.mean()),
                                  inside=float(mi.mean()), ratio=safe_ratio(ma.mean(), mi.mean()),
                                  calls_across=float(ca.mean()), calls_inside=float(cI.mean()),
                                  mcnemar_calls=mcnemar(ca, cI), per_unit=per_unit)
    # radial
    from scipy.stats import spearmanr
    rho = np.array([spearmanr(R0['S'][:, c], Q).correlation if sd[c] > 0 else 0.0 for c in range(R0['S'].shape[1])])
    rho = np.nan_to_num(rho); u = int(np.argmax(np.abs(rho))); sg = np.sign(rho[u]) if rho[u] != 0 else 1.0
    A['rad_K'] = sg * (Re['K']['S'][:, u] - R0['S'][:, u]) / (sd[u] if sd[u] > 0 else 1.0)
    A['rad_Kr'] = sg * (Re['Kr']['S'][:, u] - R0['S'][:, u]) / (sd[u] if sd[u] > 0 else 1.0)
    dQ = (Q * 0.21) / Q.std()
    sl = float(np.polyfit(dQ, A['rad_K'], 1)[0]) if A['rad_K'].std() > 0 else 0.0
    S['radial'] = dict(unit=u, rho=float(rho[u]), top5=[[int(c), float(rho[c])] for c in np.argsort(-np.abs(rho))[:5]],
                       response_K=float(A['rad_K'].mean()), response_Kr=float(A['rad_Kr'].mean()), slope_vs_dQ=sl,
                       rho_S29=float(rho[29]) if len(rho) > 29 else None)
    if 29 < len(rho) and sd[29] > 0:
        S['radial']['S29_response_K'] = float(((Re['K']['S'][:, 29] - R0['S'][:, 29]) / sd[29]).mean() * np.sign(rho[29] or 1))
    w = S['windows']['A13_I5']
    log(f'{label}: acc {S["acc"]:.4f}; T calls {S["T"]["calls_changed"]:.3f} W->Z {S["W_to_Z_unedited"]:.3f}->{S["T"]["W_to_Z"]:.3f} '
        f'(flow {S["T"]["W_to_Z_flow"]:+.3f}; Tr {S["Tr"]["W_to_Z_flow"]:+.3f}); read-out R2 {S["readout_r2_test"]:.3f} cal ratio T '
        f'{S["T"]["readout_cal_ratio"]}; WTA calls {S["W"]["calls_changed"]:.3f} acc {S["acc"]:.3f}->{S["W"]["acc"]:.3f}; '
        f'A13/I5 {w["across"]:.4f}/{w["inside"]:.4f} = {w["ratio"]} units {w["units"]}; radial S{u} rho {rho[u]:+.3f} '
        f'resp {S["radial"]["response_K"]:+.3f} (Kr {S["radial"]["response_Kr"]:+.3f}); live S {len(liveS)}; {time.time() - t:.0f}s')
    return S, A


def ablation(m, x, y, stages, ctx=None):
    """mean vs resample ablation; stages = [(name, width)]; ctx = name of the context layer (g) or None"""
    t = time.time(); J = len(x); R0 = m.run(x, keep=tuple(s for s, _ in stages) + ((ctx,) if ctx else ()))
    c0 = m.run(x)['logits'].argmax(1); rng = np.random.default_rng(3074); perms = []
    for _ in range(NRES):
        p = rng.permutation(J); fx = np.flatnonzero(p == np.arange(J))
        for i in fx: p[i], p[(i + 1) % J] = p[(i + 1) % J], p[i]
        assert not (p == np.arange(J)).any(); perms.append(p)
    out = dict(N=J, acc=float((c0 == y).mean()), n_resamples=NRES, stages={}); units = []
    for st, wd in stages:
        chm, mean, ran = unit_costs(m, x, st, wd, c0); cm = chm.mean(1); live = [int(c) for c in np.flatnonzero(cm >= 0.01)]
        V = R0[st]; rs = {}
        for c in live:
            mk = np.zeros(wd); mk[c] = 1; acc_ = np.zeros(J)
            for p in perms:
                acc_ += m.run(x, O={st: (mk, V[p])})['logits'].argmax(1) != c0
            rs[c] = acc_ / NRES; units.append((st, c, chm[c], rs[c]))
        out['stages'][st] = dict(width=wd, n_run=len(ran), mean_cost=cm.round(5).tolist(), live=live,
                                 resample_cost={c: float(v.mean()) for c, v in rs.items()})
        log(f'ablation {st}: live {len(live)}: ' + ', '.join(f'{c}:{cm[c] * 100:.1f}/{rs[c].mean() * 100:.1f}%' for c in live)
            + f' ({time.time() - t:.0f}s)')
    from scipy.stats import spearmanr, norm
    if len(units) >= 3:
        Mm = np.array([u[2] for u in units], float); Mr = np.array([u[3] for u in units])
        rho = spearmanr(Mm.mean(1), Mr.mean(1)).correlation; bi = boot_idx(J, 99); rb = []
        for b in bi[:min(NB, 1000)]:
            rb.append(spearmanr(Mm[:, b].mean(1), Mr[:, b].mean(1)).correlation)
        out['spearman_mean_vs_resample'] = dict(rho=float(rho), ci=ci(rb), n_units=len(units))
        per_stage = {}
        for st, _ in stages:
            ii = [k for k, u in enumerate(units) if u[0] == st]
            if len(ii) >= 3: per_stage[st] = float(spearmanr(Mm[ii].mean(1), Mr[ii].mean(1)).correlation)
        out['spearman_per_stage'] = per_stage
        z = (Mr.mean(1) - 0.01) / np.maximum(Mr.std(1, ddof=1) / np.sqrt(J), 1e-12); p = norm.sf(z).tolist(); q = bh(p)
        out['units'] = [dict(stage=u[0], unit=int(u[1]), mean_cost=float(u[2].mean()), mean_ci=ci(u[2][bi].mean(1)),
                             resample_cost=float(u[3].mean()), resample_ci=ci(u[3][bi].mean(1)), p_resample_gt_1pct=p[k],
                             bh_q=q[k]) for k, u in enumerate(units)]
        out['n_live_resample_bh05'] = int(sum(v < 0.05 for v in q))
        log(f'ablation: Spearman mean vs resample over {len(units)} live units {rho:.3f} {out["spearman_mean_vs_resample"]["ci"]}; '
            f'per stage {per_stage}; BH q < 0.05 (resample cost > 1 %): {out["n_live_resample_bh05"]}/{len(units)}')
    if ctx:
        G = R0[ctx]; wd = G.shape[1]; mk = np.ones(wd)
        cm = m.run(x, O={ctx: (mk, G.mean(0))})['logits'].argmax(1) != c0; acc_ = np.zeros(J); first = None
        for p in perms:
            ch = m.run(x, O={ctx: (mk, G[p])})['logits'].argmax(1) != c0; acc_ += ch; first = ch if first is None else first
        acc_ /= NRES; bi = boot_idx(J, 98)
        out['context'] = dict(stage=ctx, mean_cost=float(cm.mean()), mean_ci=ci(cm[bi].mean(1)), resample_cost=float(acc_.mean()),
                              resample_ci=ci(acc_[bi].mean(1)), mcnemar_mean_vs_resample0=mcnemar(cm, first))
        log(f'context {ctx}: mean ablation {cm.mean() * 100:.2f} % calls, resample {acc_.mean() * 100:.2f} % '
            f'{[round(v * 100, 2) for v in out["context"]["resample_ci"]]}')
    out['seconds'] = time.time() - t
    return out


# ================================================================================================ JEDI-linear
LAYER_KEYS = ('l1', 'qs', 'l2', 'dd', 'add', 'l3', 'qs1', 'd4', 'd5', 'd6', 'd7')


def quantizers(L, k):
    iq = L[k].iq; return list(iq.quantizers) if hasattr(iq, 'quantizers') else [iq]


def qbits(q):
    for a in ('b', 'bits'):
        if hasattr(q, a): return np.asarray(getattr(q, a), np.float64)
    k, i, f = (np.asarray(getattr(q, a), np.float64) for a in ('k', 'i', 'f'))
    return np.maximum(k + i + f, 0)


class JediModel(Model):
    def __init__(self, net, KB=None):
        import jax, jax.numpy as jnp
        self.net = net; L = net.L; self.L = L
        KB = KB if KB is not None else net.KB
        self.KB = {k: (jnp.asarray(np.asarray(v[0]), jnp.float32), jnp.asarray(np.asarray(v[1]), jnp.float32)) for k, v in KB.items()}
        b = qbits(quantizers(L, 'qs1')[0].quantizer); b2 = b.reshape(b.shape[-2:]) if b.ndim >= 2 else b[None]
        self.qs1 = np.broadcast_to(b2 > 0, (64, b2.shape[-1])).astype(np.float32)

        def dense(k, x, KB, act=True):
            z = jnp.einsum(L[k].equation, L[k].iq(x, training=False), KB[k][0]) + KB[k][1]
            return L[k].activation(z) if act and L[k].activation is not None else z

        def pool(k, a): return jnp.sum(L[k].iq(a, training=False), axis=L[k].axes, keepdims=L[k].keepdims) * L[k].scale

        def ov(name, t, O):
            if name not in O: return t
            m, v = O[name]
            if v.ndim == 2 and t.ndim == 3: v = v[:, None, :]
            return t * (1 - m) + m * v

        def forward(x, KB, O):
            a = dense('l1', x, KB); P = ov('P', pool('qs', a), O); g = ov('g', dense('dd', P, KB), O); s = dense('l2', a, KB)
            qs_, qd_ = L['add'].iq([s, g], training=False)
            u = dense('l3', qs_ + qd_, KB); S = ov('S', pool('qs1', u), O)
            h4 = dense('d4', S, KB); h5 = dense('d5', h4, KB); z = ov('z', dense('d6', h5, KB, act=False), O)
            lo = dense('d7', jax.nn.relu(z), KB); J = x.shape[0]
            return dict(P=P.reshape(J, -1), g=g.reshape(J, -1), S=S.reshape(J, -1), last=z.reshape(J, -1), logits=lo.reshape(J, -1))
        self._f = jax.jit(forward)

    def run(self, x, O=None, keep=('logits', 'last', 'S')):
        keep = tuple(dict.fromkeys(keep)); out = {k: [] for k in keep}; O = O or {}
        for i in range(0, len(x), BATCH):
            xb = x[i:i + BATCH]; Ob = {}
            for k, (m, v) in O.items():
                v = np.asarray(v, np.float32); v = v[i:i + BATCH] if v.ndim == 2 else v
                Ob[k] = (np.asarray(m, np.float32), v)
            r = self._f(self.net.inputs(xb, 64), self.KB, Ob)
            for k in keep: out[k].append(np.asarray(r['last'] if k == 'z' else r[k], np.float64))
        return {k: np.concatenate(v) for k, v in out.items()}


def permute_KB(KB, s, variant):
    rng = np.random.default_rng(s); out = {}
    for k in sorted(KB):
        K, b = (np.asarray(v, np.float64).copy() for v in KB[k]); nk = K.reshape(-1); nb = b.reshape(-1)
        for arr in (nk, nb):
            pos = np.flatnonzero(arr != 0) if variant == 'a' else np.arange(arr.size)
            arr[pos] = arr[pos][rng.permutation(len(pos))]
        out[k] = (nk.reshape(K.shape), nb.reshape(b.shape))
    return out


def permute_iq(L, s):
    """variant b: every input quantizer's bit tables permuted within the table (one permutation per quantizer). Returns the
    originals for restore_iq and a description."""
    rng = np.random.default_rng(100 + s); orig = []; desc = []
    for k in LAYER_KEYS:
        for qi, q in enumerate(quantizers(L, k)):
            qq = q.quantizer; vs = [v for v in qq.variables]
            sizes = [int(np.prod(v.shape)) for v in vs]; mx = max(sizes) if sizes else 0
            if mx <= 1: continue
            perm = rng.permutation(mx)
            for v in vs:
                a = np.asarray(v.numpy())
                if a.size != mx: continue
                orig.append((v, a.copy())); v.assign(a.reshape(-1)[perm].reshape(a.shape))
            if s == 0: desc.append(f'{k}[{qi}]: ' + ', '.join(f'{getattr(v, "path", v.name)} {tuple(v.shape)}' for v in vs))
    return orig, desc


def permute_slots(L, s):
    """variant c: slot-axis permutation (per channel) of the slot x channel tables of qs, l2, add, l3, qs1"""
    rng = np.random.default_rng(200 + s); orig = []
    for k in ('qs', 'l2', 'add', 'l3', 'qs1'):
        for q in quantizers(L, k):
            vs = list(q.quantizer.variables); arrs = [np.asarray(v.numpy()) for v in vs]
            full = [a.ndim >= 2 and a.shape[-2] == 64 for a in arrs]
            if not any(full): continue
            shp = arrs[full.index(True)].shape; C_ = shp[-1]
            P_ = np.stack([rng.permutation(64) for _ in range(C_)], 1)          # (64 slots, C) source slot per channel
            for v, a, f in zip(vs, arrs, full):
                if not f or a.shape != shp: continue
                a2 = a.reshape(-1, 64, C_); new = np.take_along_axis(a2, np.broadcast_to(P_, a2.shape), axis=1)
                orig.append((v, a.copy())); v.assign(new.reshape(a.shape))
    return orig


def restore_iq(orig):
    for v, a in orig: v.assign(a)


def part_jedi():
    import jax
    from interp.jedi_core import Jedi
    x, y, sp = load(); E, SW, P, Q, info = make_edits(x)
    log(f'sample: {len(x)} jets (per class {PER}, seed {SEED}); fit half {len(sp["fit"])}, test half {len(sp["te"])}')
    net = Jedi(64); R = dict(sample=dict(per_class=PER, seed=SEED, N=len(x)), edits=info, models={})
    base = JediModel(net)
    ref = net.run(x[:2000])['logits']; mine = base.run(x[:2000])['logits']
    R['forward_check_maxabs'] = float(np.abs(ref - mine).max()); log(f'own forward vs Jedi.run: max |d logits| {R["forward_check_maxabs"]:.3g}')
    arrays = {}
    S_, A_ = measure(base, x, y, sp, E, SW, P, Q, 'JEDI-linear trained', keep_units=(43, 52, 63))
    R['models']['jedi_trained'] = S_; arrays.update({f'jedi_trained|{k}': v for k, v in A_.items()}); save_json('x8_jedi', R)
    nz = {k: [int((np.asarray(v[0]) != 0).sum()), int(np.asarray(v[0]).size)] for k, v in net.KB.items()}; R['kernel_nonzero'] = nz
    log('fused kernels non-zero / size: ' + ', '.join(f'{k} {a}/{b}' for k, (a, b) in nz.items()))
    for variant in ('a', 'c', 'b'):
        for s in range(NPERM):
            jax.clear_caches()
            orig = []
            if variant == 'b':
                orig, desc = permute_iq(net.L, s)
                if desc: R['iq_permuted'] = desc; log('variant b quantizer tables: ' + ' | '.join(desc))
            if variant == 'c': orig = permute_slots(net.L, s)
            mdl = JediModel(net, permute_KB(net.KB, s, 'a' if variant == 'c' else variant)); name = f'jedi_perm{variant}{s}'
            S_, A_ = measure(mdl, x, y, sp, E, SW, P, Q, f'JEDI-linear permuted {variant} s{s}')
            S_['qs1_kept_per_channel'] = mdl.qs1.sum(0).astype(int).tolist()
            R['models'][name] = S_; arrays.update({f'{name}|{k}': v for k, v in A_.items()})
            restore_iq(orig); del mdl; save_json('x8_jedi', R)
    chk = JediModel(net).run(x[:2000])['logits']; R['restore_check_maxabs'] = float(np.abs(chk - ref).max())
    log(f'after restore: max |d logits| vs Jedi.run {R["restore_check_maxabs"]:.3g}')
    np.savez_compressed(OUT / f'x8_jedi{SFX}.npz', y=y, te=sp['te'], **arrays)
    R['seconds'] = time.time() - T0; save_json('x8_jedi', R); log('done x8_jedi')


def part_jedi_abl():
    from interp.jedi_core import Jedi
    x, y, sp = load(); net = Jedi(64); m = JediModel(net)
    R = dict(sample=dict(per_class=PER, seed=SEED, N=len(x)), model='JEDI-linear 3-feature n64')
    R['ablation'] = ablation(m, x, y, [('P', 64), ('S', 64), ('z', 16)], ctx='g')
    R['seconds'] = time.time() - T0; save_json('x8_jedi_abl', R); log('done x8_jedi_abl')


# ================================================================================================ small torch models
class TorchModel(Model):
    def __init__(self, net, arch, kind):
        self.net, self.arch, self.kind = net, arch, kind; self.qs1 = None

    def forward(self, x, mk, O):
        import torch, torch.nn.functional as F
        n = self.net; mm = mk.unsqueeze(-1).to(x.dtype); R = {}

        def ov(name, t):
            if name not in O: return t
            m_, v = O[name]; return t * (1 - m_) + m_ * v
        p = n.phi(x) * mm
        if self.kind == 'jedis':
            g = ov('g', p.sum(1) / mm.sum(1).clamp(min=1.0)); R['g'] = g
            u = n.phi2(torch.cat([p, g.unsqueeze(1).expand_as(p)], -1)) * mm; Sv = ov('S', u.sum(1))
            h = F.relu(n.rho[0](Sv)); h = ov('last', F.relu(n.rho[2](h)))
        else:
            Sv = ov('S', p.sum(1)); h = F.relu(n.rho[0](Sv)); h = F.relu(n.rho[2](h)); h = ov('last', F.relu(n.rho[4](h)))
        R['S'] = Sv; R['last'] = h; R['logits'] = n.head(h)
        return R

    def run(self, x, O=None, keep=('logits', 'last', 'S')):
        import torch
        from interp.x2_small_models import normalise
        keep = tuple(dict.fromkeys(keep)); xn, mask = normalise(x, self.arch); out = {k: [] for k in keep}; O = O or {}
        with torch.no_grad():
            for i in range(0, len(xn), BATCH):
                Ob = {}
                for k, (m_, v) in O.items():
                    v = np.asarray(v, np.float32); v = v[i:i + BATCH] if v.ndim == 2 else v
                    Ob[k] = (torch.as_tensor(np.asarray(m_, np.float32)), torch.as_tensor(v))
                r = self.forward(torch.from_numpy(xn[i:i + BATCH]), torch.from_numpy(mask[i:i + BATCH]), Ob)
                for k in keep: out[k].append(r[k].double().numpy())
        return {k: np.concatenate(v) for k, v in out.items()}


def permute_torch(net, s):
    import copy, torch
    rng = np.random.default_rng(s); n2 = copy.deepcopy(net)
    with torch.no_grad():
        for _, mod in n2.named_modules():
            if isinstance(mod, torch.nn.Linear):
                for prm in (mod.weight, mod.bias):
                    a = prm.detach().numpy().reshape(-1).copy(); prm.copy_(torch.from_numpy(a[rng.permutation(a.size)].reshape(prm.shape)))
    return n2


def part_small():
    import torch
    from interp.x2_small_models import load_net, normalise
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
    fam = args.family; mname = 'dsets-d64' if fam == 'dsets' else 'jedis-d64'
    x, y, sp = load(); E, SW, P, Q, info = make_edits(x)
    R = dict(sample=dict(per_class=PER, seed=SEED, N=len(x)), edits=info, family=fam, models={}); arrays = {}

    def one(name, net, arch, label):
        m = TorchModel(net, arch, fam)
        xn, mask = normalise(x[:2000], arch)
        with torch.no_grad(): ref = net(torch.from_numpy(xn), torch.from_numpy(mask)).double().numpy()
        d = float(np.abs(ref - m.run(x[:2000])['logits']).max())
        S_, A_ = measure(m, x, y, sp, E, SW, P, Q, label); S_['forward_check_maxabs'] = d
        R['models'][name] = S_; arrays.update({f'{name}|{k}': v for k, v in A_.items()}); save_json(f'x8_{fam}', R)
        return m
    nets = {}
    for s in (0, 1):
        rd = SMALL / f'{mname}-n64' / f'seed{s}'; net, arch = load_net(rd, 'best.pt'); nets[s] = (net, arch)
        T = json.loads((rd / 'test.json').read_text()); R.setdefault('test_json_acc', {})[f'trained{s}'] = T['accuracy']
        one(f'{fam}_trained{s}', net, arch, f'{mname} trained seed {s}')
    for s in (0, 1):
        rd = SMALL / f'{mname}-n64-shuf' / f'seed{s}'; A = json.loads((rd / 'arch.json').read_text())
        ep = A.get('best', {}).get('epochs_run'); w = f'ckpt/epoch_{ep}.pt'
        net, arch = load_net(rd, w); assert arch['split'].get('shuffle_labels'), 'not a shuffled-label run'
        T = json.loads((rd / 'test.json').read_text()); R.setdefault('test_json_acc', {})[f'shuf{s}_best'] = T['accuracy']
        R.setdefault('shuf_checkpoint', {})[s] = w
        one(f'{fam}_shuf{s}', net, arch, f'{mname} shuffled labels seed {s} ({w})')
    for s in range(NPERM):
        one(f'{fam}_perm{s}', permute_torch(nets[0][0], s), nets[0][1], f'{mname} seed 0 weight-permuted s{s}')
    np.savez_compressed(OUT / f'x8_{fam}{SFX}.npz', y=y, te=sp['te'], **arrays)
    m0 = TorchModel(nets[0][0], nets[0][1], fam)
    stg = [('S', 64), ('last', 16)] if fam == 'dsets' else [('g', 64), ('S', 64), ('last', 16)]
    R['ablation_trained0'] = ablation(m0, x, y, stg, ctx='g' if fam == 'jedis' else None)
    R['seconds'] = time.time() - T0; save_json(f'x8_{fam}', R); log(f'done x8_{fam}')


# ================================================================================================ combine
def stat_fns(te_y):
    """headline and secondary statistics: name -> (subset key, function of a dict of per-jet arrays restricted by idx)"""
    f = {}
    f['H1_translation_WZ_flow'] = ('W', lambda A, i: A['flow_T'][i].mean())
    f['H2_wta_calls_changed'] = ('all', lambda A, i: A['ch_W'][i].mean())
    f['H3_window_ratio_A13_I5'] = ('com_A13_I5', lambda A, i: win_ratio(A['win_a_A13_I5'][i].mean(), A['win_i_A13_I5'][i].mean()))
    f['H4_radial_response'] = ('all', lambda A, i: A['rad_K'][i].mean())
    f['translation_calls_changed'] = ('all', lambda A, i: A['ch_T'][i].mean())
    f['translation_readout_cal_ratio'] = ('te', lambda A, i: safe_ratio(safe_ratio(A['dR_T'][i].mean(), A['pred_T'][i].mean()),
                                                                     safe_ratio(A['dR_K'][i].mean(), A['dlogm_K'][i].mean())))
    f['wta_readout_cal_ratio'] = ('te', lambda A, i: safe_ratio(safe_ratio(A['dR_W'][i].mean(), A['pred_W'][i].mean()),
                                                             safe_ratio(A['dR_K'][i].mean(), A['dlogm_K'][i].mean())))
    f['wta_acc_drop'] = ('all', lambda A, i: A['corr0'][i].mean() - A['corr_W'][i].mean())
    f['window_ratio_A13_I11'] = ('com_A13_I11', lambda A, i: win_ratio(A['win_a_A13_I11'][i].mean(), A['win_i_A13_I11'][i].mean()))
    f['window_ratio_A0_I3'] = ('com_A0_I3', lambda A, i: win_ratio(A['win_a_A0_I3'][i].mean(), A['win_i_A0_I3'][i].mean()))
    f['window_across_A13'] = ('com_A13_I5', lambda A, i: A['win_a_A13_I5'][i].mean())
    f['radial_response_rotation_control'] = ('all', lambda A, i: A['rad_Kr'][i].mean())
    return f


def part_combine():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    D = {}; y = None
    for fam in ('jedi', 'dsets', 'jedis'):
        p = OUT / f'x8_{fam}{SFX}.npz'
        if not p.exists(): log('missing', p); continue
        z = np.load(p); y = z['y']; te = z['te']
        for k in z.files:
            if '|' in k:
                mdl, key = k.split('|'); D.setdefault(mdl, {})[key] = z[k]
    J = len(y); nW = int((y == W_).sum()); F = stat_fns(None)
    ncom = {nm: len(next(iter(D.values()))[f'win_a_{nm}']) for nm, _, _ in PAIRS}
    sizes = dict(all=J, W=nW, te=len(te), **{f'com_{nm}': n for nm, n in ncom.items()})
    BI = {k: boot_idx(n, i) for i, (k, n) in enumerate(sizes.items())}
    full = {k: np.arange(n) for k, n in sizes.items()}
    boot = {}; point = {}
    for mdl, A in D.items():
        for st, (sub, fn) in F.items():
            point[(mdl, st)] = fn(A, full[sub]); boot[(mdl, st)] = np.array([fn(A, b) for b in BI[sub]], float)
    fams = dict(jedi=dict(trained=['jedi_trained'], nulls=[f'jedi_perm{v}{s}' for v in 'acb' for s in range(NPERM)],
                          nulls_c=[f'jedi_permc{s}' for s in range(NPERM)],
                          nulls_a=[f'jedi_perma{s}' for s in range(NPERM)], nulls_b=[f'jedi_permb{s}' for s in range(NPERM)],
                          cross=['jedis_shuf0', 'jedis_shuf1']),
                dsets=dict(trained=['dsets_trained0', 'dsets_trained1'], nulls=[f'dsets_perm{s}' for s in range(NPERM)] + ['dsets_shuf0', 'dsets_shuf1']),
                jedis=dict(trained=['jedis_trained0', 'jedis_trained1'], nulls=[f'jedis_perm{s}' for s in range(NPERM)] + ['jedis_shuf0', 'jedis_shuf1']))
    Rs = dict(sample=dict(per_class=PER, seed=SEED, N=J, N_W=nW, N_test_half=len(te), N_common=ncom), B=NB, tests={})

    def compare(tr, nulls, st):
        nl = [n for n in nulls if (n, st) in point]
        if (tr, st) not in point or not nl: return None
        bt = boot[(tr, st)]; pt_ = point[(tr, st)]; ps = {}
        for n in nl:
            d = bt - boot[(n, st)]; ok = ~np.isnan(d)
            ps[n] = float((1 + int((d[ok] <= 0).sum())) / (1 + int(ok.sum()))) if ok.any() else None
        pool = np.concatenate([boot[(n, st)] for n in nl]); valid = [v for v in ps.values() if v is not None]
        npts = [point[(n, st)] for n in nl]
        rank_p = (1 + sum(1 for v in npts if not np.isnan(v) and not np.isnan(pt_) and v >= pt_)) / (1 + len(nl))
        return dict(trained=pt_, trained_ci=ci(bt), null_points={n: point[(n, st)] for n in nl}, null_band=ci(pool),
                    p_per_null=ps, p_iut=max(valid) if len(valid) == len(nl) else None,
                    p_iut_note=None if len(valid) == len(nl) else 'some null statistics undefined (nan): p over defined nulls only',
                    p_iut_defined=max(valid) if valid else None, rank_p=rank_p, n_null=len(nl))
    for fam, G in fams.items():
        for tr in G['trained']:
            for st in F:
                c = compare(tr, G['nulls'], st)
                if c is None: continue
                if fam == 'jedi':
                    c['vs_variant_a'] = compare(tr, G['nulls_a'], st); c['vs_variant_b'] = compare(tr, G['nulls_b'], st)
                    c['vs_variant_c'] = compare(tr, G['nulls_c'], st)
                    c['vs_jedis_shuffled'] = compare(tr, G['cross'], st)
                Rs['tests'].setdefault(fam, {}).setdefault(tr, {})[st] = c
    heads = [k for k in F if k.startswith('H')]; keys, ps = [], []
    for fam, tr in (('jedi', 'jedi_trained'), ('dsets', 'dsets_trained0'), ('jedis', 'jedis_trained0')):
        for h in heads:
            c = Rs['tests'].get(fam, {}).get(tr, {}).get(h)
            if c is None: continue
            keys.append(f'{fam}:{h}'); ps.append(c['p_iut'] if c['p_iut'] is not None else c['p_iut_defined'])
    hp = holm(ps); Rs['holm'] = {k: dict(p=p, p_holm=h) for k, p, h in zip(keys, ps, hp)}
    for k, v in Rs['holm'].items(): log(f'Holm {k}: p_IUT {v["p"]} -> {v["p_holm"]}')
    for fam, d in Rs['tests'].items():
        for tr, dd in d.items():
            for st, c in dd.items():
                log(f'{fam} {tr} {st}: trained {c["trained"]:.4g} {c["trained_ci"]}; null band {c["null_band"]}; '
                    f'null points ' + ', '.join(f'{n.split("_")[-1]}:{v:.3g}' for n, v in c['null_points'].items() if v is not None)
                    + f'; p_IUT {c["p_iut"]}; rank p {c["rank_p"]:.3f}')
    for nm in ('jedi', 'jedi_abl', 'dsets', 'jedis'):
        p = OUT / f'x8_{nm}{SFX}.json'
        if p.exists(): Rs.setdefault('inputs', []).append(p.name)
    save_json('x8_summary', Rs)
    # figure: four headline tests, trained (point + 95 % CI) vs null models (points) per family
    titles = {'H1_translation_WZ_flow': 'translation 0.06 rad along Δη:\nshare of true W jets moved to a Z call',
              'H2_wta_calls_changed': 'inputs about the WTA axis:\nshare of calls changed',
              'H3_window_ratio_A13_I5': 'angle swap ranks 13|14 vs 5|6:\nacross / inside |ΔS| ratio',
              'H4_radial_response': 'angles × 1.1 at fixed pT: change of the\nS unit most correlated with Σ pT r²  [s.d.]'}
    fig, axs = plt.subplots(1, 4, figsize=(15, 4.4), facecolor='#fcfcfb')
    labels = [('jedi', 'JEDI-linear'), ('dsets', 'Deep Sets'), ('jedis', 'JEDI-style float')]
    for ax, h in zip(axs, heads):
        ax.set_facecolor('#fcfcfb'); ax.grid(True, axis='y', color='#e4e3df'); ax.set_axisbelow(True)
        for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
        for xi, (fam, lab) in enumerate(labels):
            if fam not in Rs['tests']: continue
            tr = fams[fam]['trained'][0]; c = Rs['tests'][fam].get(tr, {}).get(h)
            if c is None: continue
            nv = [v for v in c['null_points'].values() if v is not None and np.isfinite(v)]
            ninf = sum(1 for v in c['null_points'].values() if v is not None and np.isinf(v))
            if ninf: ax.annotate(f'+{ninf} null(s): inside = 0 (ratio ∞)', (xi, 1.0), xycoords=('data', 'axes fraction'),
                                 ha='center', va='top', fontsize=7, color='#5f5e59')
            ax.scatter(xi + 0.15 + np.linspace(-0.06, 0.06, len(nv)), nv, s=22, color='#8a8984', zorder=3,
                       label='no trained function (permuted / shuffled)' if xi == 0 else None)
            lo, hi = c['trained_ci']
            ax.errorbar([xi - 0.15], [c['trained']], yerr=[[c['trained'] - lo], [hi - c['trained']]], fmt='o', color='#2a78d6', ms=7,
                        capsize=3, zorder=4, label='trained (95 % CI)' if xi == 0 else None)
        ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in labels], fontsize=9); ax.set_title(titles[h], fontsize=10)
        if h == 'H3_window_ratio_A13_I5': ax.axhline(1, color='#5f5e59', lw=1, ls=':')
        else: ax.axhline(0, color='#5f5e59', lw=1, ls=':')
    axs[0].legend(fontsize=8, frameon=False, loc='upper right')
    fig.suptitle(f'Headline causal tests on trained networks vs networks with no trained function (N = {J} test jets, seed {SEED})',
                 fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / f'x8_nulls{SFX}.png', dpi=130); log('wrote', OUT / f'x8_nulls{SFX}.png')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    log(f'X8 part {args.part} family {args.family} smoke {SM}')
    dict(jedi=part_jedi, small=part_small, combine=part_combine)[args.part]() if args.part != 'jedi-abl' else part_jedi_abl()
