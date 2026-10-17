"""E20 (claims A1/A3, reviewer objection "objects are a relabelling of sum pooling / EFN filters"): causal test of the
pT-rank windows of JEDI-linear (3-feature, n64) with angle swaps inside vs across a window boundary.

Sample: interp.jedi_core.sample(per_class=2000, seed=41) = 10,000 test jets; network = exact fixed-point Jedi(64).
Windows: zero-bit masks (iq.quantizer.b > 0) of qs (P pooling) and qs1 (S pooling), slot x channel (as interp/e3m_masks.py).
Edits (pT order kept; jets where an edit needs a particle that does not exist are skipped, N reported per edit):
  A13  swap (Δη, Δφ) of the rank-13 and rank-14 particles   (boundary of the u0/u43/u52/u63 window 0-13 and u19 13-63)
  A12  swap (Δη, Δφ) of ranks 12 and 13                     (boundary of the u19 window 13-63, u51 0-12,14; added after the
       qs1 masks showed that u19's lower edge lies between slots 12 and 13, not 13 and 14)
  A0   swap (Δη, Δφ) of ranks 0 and 1                        (boundary of the u1 window 1-63)
  I5   swap (Δη, Δφ) of ranks 5 and 6                        (matched control: no boundary of u0/u43/u19/u1/u29/u30)
  I30  swap (Δη, Δφ) of ranks 30 and 31                      (matched control)
  N13  pT nudge: rank-13 pT -> just below rank 14's (pT_14 - min(0.01 pT_14, (pT_14 - pT_15)/2)), re-sorted: slot 13 =
       old rank 14, slot 14 = old rank 13 with the lowered pT and its own angles (it crosses the 13|14 boundary)
  N5   matched control: the same nudge at ranks 5/6
Per edit and live S unit c (19, E1 >= 1 % calls): dS = S_c(edited) - S_c(unedited) from the network; identity prediction
  pred_win = (1/16) Σ_i m_ic Δu_ic (network's own u per particle, float, before the qs1 quantizer; m = qs1 window);
  pred_all = (1/16) Σ_i Δu_ic (same u, no window: the permutation-invariant-sum prediction);
  pred_dir = window identity over the two edited slots only (context g held at the unedited path's value by construction
  of the sum: the remainder pred_win - pred_dir is the change routed through g);
  frac |dS| > 1 LSB (LSB_c = coarsest qs1 output step over the kept slots), frac dS != 0, R² of dS vs pred_win / pred_all,
  % calls changed, accuracy before -> after (exact McNemar on correctness). 95 % bootstrap CIs over jets (1,000 resamples).
Contrasts (paired, common jets): |dS_A| - |dS_I| per unit, Wilcoxon signed-rank p, Holm over 19 units x 5 contrasts;
  calls changed A vs I: exact McNemar.
Mask controls (float + masks program of E3m, masks on l1 qs l2 dd l3 qs1 d4-d7): qs1 slot masks (and, separately, all four
  slot-mask layers qs, l2, l3, qs1) (a) permuted along the slot axis per channel (10 draws, density kept), (b) shifted by
  +3 / -3 slots (edge value replicated), (c) all ones; accuracy, % calls vs network, and the same six edits.
Untrained controls: Jedi(untrained=True, seed 0) (fixed point, trained number formats and masks) and a He-normal float
  network (kernels N(0, 2/fan_in), biases 0) with the trained masks; |dS|/sd(S) per unit for the six edits.
Writes $AI_RESULTS/jedi/e20_rank_windows.json.       Run: python -m interp.e20_rank_windows_jedi [per_class]
Part 2 (--programs): per-jet dS under the network and the float programs with slot masks set to one (each of qs, l2, l3, qs1,
  and all four) or shifted by +/-3 slots, for the same edits + N12; -> $AI_RESULTS/jedi/e20_programs.json
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'; OUT.mkdir(parents=True, exist_ok=True)
import jax, jax.numpy as jnp
from interp.jedi_core import Jedi, sample
from interp import e19_common as C

T0 = time.time()
def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)

PER = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 2000
SEED = 41; NB = 1000; BATCH = 1000
LIVE_S = [29, 1, 0, 50, 19, 30, 43, 60, 13, 17, 56, 40, 31, 51, 63, 52, 62, 58, 20]   # E1/E13 order (by cost)
LIVE_P = [15, 32, 40, 29, 56, 6, 43, 36, 46, 30, 3, 57]
EDITS = dict(A13=('swap', 13), A12=('swap', 12), A0=('swap', 0), I5=('swap', 5), I30=('swap', 30), N13=('nudge', 13), N5=('nudge', 5))
CONTRASTS = [('A13', 'I5'), ('A13', 'I30'), ('A12', 'I5'), ('A0', 'I5'), ('N13', 'N5')]
NAMES = ('l1', 'qs', 'l2', 'dd', 'l3', 'qs1', 'd4', 'd5', 'd6', 'd7')
SLOT = ('qs', 'l2', 'l3', 'qs1')
rng = np.random.default_rng(SEED)

net = Jedi(64); L = net.L; SC1 = net.pool_scale[1]
MASK0 = {k: (np.asarray(L[k].iq.quantizer.b) > 0).astype(np.float32) for k in NAMES}
M1 = MASK0['qs1'][0]; M0 = MASK0['qs'][0]                       # (64 slots, 64 channels)
q1 = L['qs1'].iq.quantizer
f1 = (np.asarray(q1.b) - np.asarray(q1.i) - np.asarray(q1.k))[0]
LSB = {c: float(SC1 * (2.0 ** -f1[M1[:, c] > 0, c]).max()) if (M1[:, c] > 0).any() else None for c in range(64)}
LSB_FINE = {c: float(SC1 * (2.0 ** -f1[M1[:, c] > 0, c]).min()) if (M1[:, c] > 0).any() else None for c in range(64)}


def describe(col):
    kept = np.flatnonzero(col > 0)
    if len(kept) == 0: return 'none'
    runs, st = [], kept[0]
    for i in range(1, len(kept) + 1):
        if i == len(kept) or kept[i] != kept[i - 1] + 1:
            runs.append(f'{st}-{kept[i - 1]}' if kept[i - 1] > st else f'{st}')
            st = kept[i] if i < len(kept) else None
    return ','.join(runs)


WIN_S = {c: describe(M1[:, c]) for c in LIVE_S}; WIN_P = {c: describe(M0[:, c]) for c in LIVE_P}
log('qs1 windows (live S):', WIN_S); log('qs windows (live P):', WIN_P)
log('S LSB coarse/fine:', {c: (LSB[c], LSB_FINE[c]) for c in LIVE_S})
EDGE = {e: {c: dict(slot_k=bool(M1[k, c] > 0), slot_k1=bool(M1[k + 1, c] > 0)) for c in LIVE_S} for e, (_, k) in EDITS.items()}
for e in EDITS:
    k = EDITS[e][1]
    log(f'edit {e} slots {k},{k + 1}: S units with a qs1 boundary between them',
        [c for c in LIVE_S if M1[k, c] != M1[k + 1, c]], '| P units with a qs boundary', [c for c in LIVE_P if M0[k, c] != M0[k + 1, c]])


# ------------------------------------------------------------------------------------------------ network pass
SL = [0, 1, 5, 6, 12, 13, 14, 30, 31]
SLOTIDX = {s: j for j, s in enumerate(SL)}

def net_pass(N, x):
    """logits, S, P, window / all-slot sums of u (float, before qs1) and the per-slot u at all slots summed with mask"""
    out = dict(logits=[], S=[], P=[], uw=[], ua=[], u_slots=[])
    m1 = jnp.asarray(M1)
    for i in range(0, len(x), BATCH):
        r = N._f(N.inputs(x[i:i + BATCH], 64), {})
        u = r['u']
        out['logits'].append(np.asarray(r['logits'], np.float64)); out['S'].append(np.asarray(r['S'], np.float64))
        out['P'].append(np.asarray(r['P'], np.float64))
        out['uw'].append(np.asarray((u * m1[None]).sum(1), np.float64) * SC1)
        out['ua'].append(np.asarray(u.sum(1), np.float64) * SC1)
        out['u_slots'].append(np.asarray(u, np.float32)[:, SL] * M1[SL][None])
    return {k: np.concatenate(v) for k, v in out.items()}




# ------------------------------------------------------------------------------------------------ edits
def make_edit(x, kind, k):
    pt = x[..., 0]; nreal = (pt > 0).sum(1)
    if kind == 'swap':
        J = np.flatnonzero(nreal > k + 1)
        xe = x[J].copy(); xe[:, [k, k + 1], 1:] = xe[:, [k + 1, k], 1:]
        return J, xe, {}
    nxt = pt[:, k + 2] if k + 2 < 64 else np.zeros(len(x))
    J = np.flatnonzero((nreal > k + 1) & (pt[:, k + 1] > nxt))
    xe = x[J].copy(); p1 = pt[J, k + 1]
    new = p1 - np.minimum(0.01 * p1, 0.5 * (p1 - nxt[J]))
    old = xe[:, k].copy(); xe[:, k] = xe[:, k + 1]; xe[:, k + 1] = old; xe[:, k + 1, 0] = new
    return J, xe, dict(dpt_mean=float((new - pt[J, k]).mean()), dpt_median=float(np.median(new - pt[J, k])),
                       rel_dpt_mean=float(((new - pt[J, k]) / pt[J, k]).mean()))


def boot(fn, n, B=NB):
    idx = rng.integers(0, n, (B, n))
    v = np.array([fn(ii) for ii in idx]); return [float(np.nanquantile(v, .025)), float(np.nanquantile(v, .975))]


def r2(a, p):
    d = ((a - a.mean()) ** 2).sum()
    return float(1 - ((a - p) ** 2).sum() / d) if d > 0 else None


def unit_stats(dS, pw, pa, pd, c, sdS):
    n = len(dS); lsb = LSB[c]
    big = np.abs(dS) > lsb * (1 + 1e-6)
    e = dict(mean_dS=float(dS.mean()), mean_dS_ci=boot(lambda i: dS[i].mean(), n),
             mean_abs_dS=float(np.abs(dS).mean()), mean_abs_dS_ci=boot(lambda i: np.abs(dS[i]).mean(), n),
             mean_abs_dS_over_sd=float(np.abs(dS).mean() / sdS) if sdS > 0 else None,
             frac_gt_lsb=float(big.mean()), frac_gt_lsb_ci=boot(lambda i: big[i].mean(), n),
             frac_nonzero=float((np.abs(dS) > 1e-9).mean()),
             frac_gt_fine_lsb=float((np.abs(dS) > LSB_FINE[c] * (1 + 1e-6)).mean()),
             mean_pred_win=float(pw.mean()), mean_abs_pred_win=float(np.abs(pw).mean()),
             mean_pred_all=float(pa.mean()), mean_abs_pred_all=float(np.abs(pa).mean()),
             mean_abs_pred_direct=float(np.abs(pd).mean()),
             r2_win=r2(dS, pw), r2_all=r2(dS, pa), r2_direct=r2(dS, pd),
             rmse_win_over_lsb=float(np.sqrt(((dS - pw) ** 2).mean()) / lsb))
    if e['r2_win'] is not None and np.abs(dS).sum() > 0:
        e['r2_win_ci'] = boot(lambda i: r2(dS[i], pw[i]) if r2(dS[i], pw[i]) is not None else np.nan, n)
        e['r2_all_ci'] = boot(lambda i: r2(dS[i], pa[i]) if r2(dS[i], pa[i]) is not None else np.nan, n)
        e['slope_win'] = float((pw * dS).sum() / (pw * pw).sum()) if (pw * pw).sum() > 0 else None
    return e


def run_net_edits(N, x, y, base, label, full=True):
    cb = base['logits'].argmax(1); res = {}; per = {}
    sdS = base['S'].std(0)
    for e, (kind, k) in EDITS.items():
        J, xe, info = make_edit(x, kind, k)
        ed = net_pass(N, xe)
        ce = ed['logits'].argmax(1); c0 = cb[J]; n = len(J)
        chg = ce != c0
        rec = dict(N=n, **info, calls_changed=float(chg.mean()), calls_changed_ci=boot(lambda i: chg[i].mean(), n),
                   acc_before=float((c0 == y[J]).mean()), acc_after=float((ce == y[J]).mean()),
                   mcnemar_correct=C.mcnemar(c0 == y[J], ce == y[J]))
        dS_all = ed['S'] - base['S'][J]; pw_all = ed['uw'] - base['uw'][J]; pa_all = ed['ua'] - base['ua'][J]
        du = ed['u_slots'] - base['u_slots'][J]
        pd_all = SC1 * (du[:, SLOTIDX[k]] + du[:, SLOTIDX[k + 1]])
        rec['identity_check_unedited_max_abs_S_minus_uwin'] = float(np.abs(base['S'] - base['uw']).max())
        if full:
            rec['S'] = {c: unit_stats(dS_all[:, c], pw_all[:, c], pa_all[:, c], pd_all[:, c], c, sdS[c]) for c in LIVE_S}
            dP = ed['P'] - base['P'][J]
            rec['P'] = {c: dict(mean_abs_dP=float(np.abs(dP[:, c]).mean()), frac_nonzero=float((np.abs(dP[:, c]) > 1e-9).mean()))
                        for c in LIVE_P}
        else:
            rec['S'] = {c: dict(mean_abs_dS_over_sd=float(np.abs(dS_all[:, c]).mean() / sdS[c]) if sdS[c] > 0 else None,
                                frac_nonzero=float((np.abs(dS_all[:, c]) > 1e-9).mean())) for c in LIVE_S}
        res[e] = rec; per[e] = dict(J=J, dS=dS_all, chg=chg)
        log(f'{label} {e}: N={n} calls changed {100 * chg.mean():.2f} % acc {100 * rec["acc_before"]:.2f} -> {100 * rec["acc_after"]:.2f}; '
            + ' '.join(f'S{c}:{rec["S"][c].get("frac_gt_lsb", rec["S"][c]["frac_nonzero"]):.3f}' for c in (0, 43, 19, 1, 29, 30)))
    return res, per


def contrasts(per):
    out, ps, keys = {}, [], []
    from scipy import stats
    for a, b in CONTRASTS:
        Ja, Jb = per[a]['J'], per[b]['J']; com = np.intersect1d(Ja, Jb)
        ia, ib = np.searchsorted(Ja, com), np.searchsorted(Jb, com)
        ca, cb_ = per[a]['chg'][ia], per[b]['chg'][ib]; n = len(com)
        d = dict(N=n, calls_changed_A=float(ca.mean()), calls_changed_I=float(cb_.mean()),
                 diff=float(ca.mean() - cb_.mean()), diff_ci=boot(lambda i: ca[i].mean() - cb_[i].mean(), n),
                 mcnemar=C.mcnemar(cb_, ca), units={})
        for c in LIVE_S:
            xa, xb = np.abs(per[a]['dS'][ia, c]), np.abs(per[b]['dS'][ib, c]); dd = xa - xb
            ga, gb = xa > LSB[c] * (1 + 1e-6), xb > LSB[c] * (1 + 1e-6)
            try:
                p = float(stats.wilcoxon(xa, xb).pvalue) if (dd != 0).any() else 1.0
            except ValueError:
                p = 1.0
            d['units'][c] = dict(mean_abs_A=float(xa.mean()), mean_abs_I=float(xb.mean()), diff=float(dd.mean()),
                                 diff_ci=boot(lambda i: dd[i].mean(), n), frac_gt_lsb_A=float(ga.mean()),
                                 frac_gt_lsb_I=float(gb.mean()), wilcoxon_p=p)
            ps.append(p); keys.append((f'{a}_vs_{b}', c))
        out[f'{a}_vs_{b}'] = d
    hp = C.holm(ps)
    for (k, c), h in zip(keys, hp): out[k]['units'][c]['holm_p'] = h
    out['holm_m'] = len(ps)
    return out


# ------------------------------------------------------------------------------------------------ float programs
KBT = {k: (jnp.asarray(v[0], jnp.float32), jnp.asarray(v[1], jnp.float32)) for k, v in net.KB.items()}


def prog_fwd(xn, M, KB):
    def q(k, t): return t * M[k]
    def dense(k, t, act=True):
        z = jnp.einsum(L[k].equation, q(k, t), KB[k][0]) + KB[k][1]
        return jax.nn.relu(z) if act and L[k].activation is not None else z
    def pool(k, a): return jnp.sum(q(k, a), axis=L[k].axes, keepdims=L[k].keepdims) * L[k].scale
    a = dense('l1', xn); g = dense('dd', pool('qs', a)); s = dense('l2', a)
    u = dense('l3', s + g); S = pool('qs1', u)
    lo = dense('d7', jax.nn.relu(dense('d6', dense('d5', dense('d4', S)), act=False)))
    J = xn.shape[0]
    return lo.reshape(J, -1), S.reshape(J, -1)


PF = jax.jit(prog_fwd)


def prog_run(x, M, KB=KBT):
    Mj = {k: jnp.asarray(v) for k, v in M.items()}; lo, S = [], []
    for i in range(0, len(x), 5000):
        l_, s_ = PF(jnp.asarray(net.inputs(x[i:i + 5000], 64)), Mj, KB)
        lo.append(np.asarray(l_, np.float64)); S.append(np.asarray(s_, np.float64))
    return np.concatenate(lo), np.concatenate(S)


def prog_edits(x, y, M, cnet, KB=KBT, edits=True):
    lo, S = prog_run(x, M, KB); c = lo.argmax(1); sd = S.std(0)
    rec = dict(acc=float((c == y).mean()), calls_vs_net=float((c != cnet).mean()))
    if not edits: return rec
    rec['edits'] = {}
    for e, (kind, k) in EDITS.items():
        J, xe, _ = make_edit(x, kind, k)
        le, Se = prog_run(xe, M, KB); dS = Se - S[J]
        rec['edits'][e] = dict(N=len(J), calls_changed=float((le.argmax(1) != c[J]).mean()),
                               units={c_: dict(mean_abs_dS=float(np.abs(dS[:, c_]).mean()),
                                               mean_abs_dS_over_sd=float(np.abs(dS[:, c_]).mean() / sd[c_]) if sd[c_] > 0 else None,
                                               frac_gt_lsb=float((np.abs(dS[:, c_]) > LSB[c_] * (1 + 1e-6)).mean()))
                                      for c_ in LIVE_S})
    return rec


def shift_mask(m, d):
    """m (1, 64, C): out[s] = m[clip(s - d)]"""
    idx = np.clip(np.arange(64) - d, 0, 63)
    return m[:, idx, :]


def permute_mask(m, r):
    out = m.copy()
    for c in range(m.shape[2]): out[0, :, c] = m[0, r.permutation(64), c]
    return out


# ------------------------------------------------------------------------------------------------ main
def main():
    x, y, _ = sample(per_class=PER, seed=SEED)
    pt = x[..., 0]; nreal = (pt > 0).sum(1)
    sorted_ok = float((np.diff(pt, axis=1) <= 1e-6).all(1).mean())
    log(f'sample per_class={PER} seed={SEED}: N={len(y)}; jets with pT non-increasing along slots {100 * sorted_ok:.2f} %; '
        f'nreal >= 15: {(nreal >= 15).mean():.3f}, >= 32: {(nreal >= 32).mean():.3f}')
    R = dict(N=int(len(y)), per_class=PER, seed=SEED, n_boot=NB, sorted_frac=sorted_ok, windows_S=WIN_S, windows_P=WIN_P,
             lsb_coarse=LSB, lsb_fine=LSB_FINE, live_S=LIVE_S, live_P=LIVE_P,
             boundary=dict({e: dict(S=[c for c in LIVE_S if M1[k, c] != M1[k + 1, c]],
                                    P=[c for c in LIVE_P if M0[k, c] != M0[k + 1, c]]) for e, (_, k) in EDITS.items()}),
             nreal_hist=np.bincount(nreal, minlength=65).tolist())
    base = net_pass(net, x); cnet = base['logits'].argmax(1)
    R['acc_net'] = float((cnet == y).mean())
    R['identity_unedited_max_abs_S_minus_uwin'] = float(np.abs(base['S'] - base['uw']).max())
    log(f'network accuracy {100 * R["acc_net"]:.2f} %; max |S - (1/16)Σ m u| on unedited jets {R["identity_unedited_max_abs_S_minus_uwin"]:.4f}')
    R['net'], per = run_net_edits(net, x, y, base, 'net')
    R['contrasts'] = contrasts(per)
    for k, v in R['contrasts'].items():
        if k == 'holm_m': continue
        log(f'contrast {k}: N={v["N"]} calls A {100 * v["calls_changed_A"]:.2f} % vs I {100 * v["calls_changed_I"]:.2f} %, '
            f'McNemar p {v["mcnemar"]["p"]:.2e}; holm p ' + ' '.join(f'S{c}:{v["units"][c]["holm_p"]:.1e}' for c in (0, 43, 19, 1, 29, 30)))
    # one real jet for the window diagram: first W jet (y == 2) with >= 32 particles
    j = int(np.flatnonzero((y == 2) & (nreal >= 32))[0])
    R['diagram_jet'] = dict(index=j, label=int(y[j]), nreal=int(nreal[j]), particles=x[j, :nreal[j]].astype(float).round(5).tolist(),
                            u_slots_contrib={c: (base['u_slots'][j, :, c] * SC1).round(4).tolist() for c in (0, 19, 1, 29)})
    del base, per
    json.dump(R, open(OUT / 'e20_rank_windows.json', 'w'), indent=1, default=float); log('wrote partial json (network part)')

    # mask controls on the float + masks program
    MC = {}
    MC['mask'] = prog_edits(x, y, MASK0, cnet); log(f'program mask: acc {100 * MC["mask"]["acc"]:.2f} % calls vs net {100 * MC["mask"]["calls_vs_net"]:.2f} %')
    for scope, layers in (('qs1', ('qs1',)), ('all4', SLOT)):
        for d in (+3, -3):
            M = dict(MASK0); M.update({k: shift_mask(MASK0[k], d) for k in layers})
            MC[f'shift{d:+d}_{scope}'] = prog_edits(x, y, M, cnet)
        M = dict(MASK0); M.update({k: np.ones_like(MASK0[k]) for k in layers})
        MC[f'ones_{scope}'] = prog_edits(x, y, M, cnet)
        perm = []
        for r in range(10):
            rr = np.random.default_rng(1000 + r); M = dict(MASK0); M.update({k: permute_mask(MASK0[k], rr) for k in layers})
            perm.append(prog_edits(x, y, M, cnet, edits=(r < 3)))
        accs = np.array([p['acc'] for p in perm])
        MC[f'perm_{scope}'] = dict(acc_mean=float(accs.mean()), acc_min=float(accs.min()), acc_max=float(accs.max()), acc_all=accs.tolist(),
                                   calls_vs_net_mean=float(np.mean([p['calls_vs_net'] for p in perm])), draws_with_edits=perm[:3])
        for k in [f'shift+3_{scope}', f'shift-3_{scope}', f'ones_{scope}']:
            log(f'program {k}: acc {100 * MC[k]["acc"]:.2f} % calls vs net {100 * MC[k]["calls_vs_net"]:.2f} %; A13/I5 calls changed '
                f'{100 * MC[k]["edits"]["A13"]["calls_changed"]:.2f}/{100 * MC[k]["edits"]["I5"]["calls_changed"]:.2f} %')
        log(f'program perm_{scope} (10 draws): acc mean {100 * accs.mean():.2f} % [{100 * accs.min():.2f}, {100 * accs.max():.2f}]')
    R['mask_controls'] = MC
    json.dump(R, open(OUT / 'e20_rank_windows.json', 'w'), indent=1, default=float); log('wrote partial json (mask controls)')

    # untrained controls
    UN = {}
    un = Jedi(64, untrained=True, seed=0); bu = net_pass(un, x); cu = bu['logits'].argmax(1)
    UN['jedi_untrained'] = dict(acc=float((cu == y).mean()), sd_S={c: float(bu['S'][:, c].std()) for c in LIVE_S})
    UN['jedi_untrained']['edits'], _ = run_net_edits(un, x, y, bu, 'untrained', full=False)
    del bu, un
    hr = np.random.default_rng(7); KBH = {}
    for k, (K, b) in net.KB.items():
        fan_in = int(np.prod(K.shape[:-1])) if K.ndim > 1 else 1
        KBH[k] = (jnp.asarray(hr.normal(0, np.sqrt(2 / fan_in), K.shape), jnp.float32), jnp.zeros(b.shape, jnp.float32))
    UN['he_float_trained_masks'] = prog_edits(x, y, MASK0, cnet, KB=KBH)
    log(f'He-normal float + trained masks: acc {100 * UN["he_float_trained_masks"]["acc"]:.2f} %')
    R['untrained'] = UN
    R['wall_s'] = time.time() - T0
    json.dump(R, open(OUT / 'e20_rank_windows.json', 'w'), indent=1, default=float); log('done; wrote', OUT / 'e20_rank_windows.json')


# ------------------------------------------------------------------------------------------------ part 2: programs
def programs():
    """E20b: per-jet dS under the network and under float programs whose slot masks are removed or moved, for the same
    edits (+ N12, the pT nudge at u19's 12|13 boundary). H0 (window removal changes nothing): |dS| with all four slot masks
    set to one equals |dS| with the trained masks. Writes $AI_RESULTS/jedi/e20_programs.json."""
    from scipy import stats
    EDITS['N12'] = ('nudge', 12)
    x, y, _ = sample(per_class=PER, seed=SEED)
    cnet = net.run(x, keep=('logits',))['logits'].argmax(1)
    PR = {'mask': dict(MASK0)}
    for k in SLOT:
        M = dict(MASK0); M[k] = np.ones_like(MASK0[k]); PR[f'ones_{k}'] = M
    M = dict(MASK0); M.update({k: np.ones_like(MASK0[k]) for k in SLOT}); PR['ones_all4'] = M
    for d in (+3, -3):
        M = dict(MASK0); M.update({k: shift_mask(MASK0[k], d) for k in SLOT}); PR[f'shift{d:+d}_all4'] = M
    out = dict(N=int(len(y)), seed=SEED, per_class=PER, n_boot=NB, acc_net=float((cnet == y).mean()), programs={}, edits={})
    base = {'net': net.run(x, keep=('logits', 'S'))}
    for p, M in PR.items():
        lo, S = prog_run(x, M); base[p] = dict(logits=lo, S=S)
        out['programs'][p] = dict(acc=float((lo.argmax(1) == y).mean()), calls_vs_net=float((lo.argmax(1) != cnet).mean()))
        log(f'program {p}: acc {100 * out["programs"][p]["acc"]:.2f} %, calls vs net {100 * out["programs"][p]["calls_vs_net"]:.2f} %')
    ps, keys = [], []
    for e, (kind, k) in EDITS.items():
        J, xe, info = make_edit(x, kind, k); n = len(J)
        dS, chg = {}, {}
        r = net.run(xe, keep=('logits', 'S')); dS['net'] = r['S'] - base['net']['S'][J]
        chg['net'] = r['logits'].argmax(1) != base['net']['logits'][J].argmax(1)
        for p, M in PR.items():
            lo, S = prog_run(xe, M); dS[p] = S - base[p]['S'][J]; chg[p] = lo.argmax(1) != base[p]['logits'][J].argmax(1)
        rec = dict(N=n, **info, boundary_S=[c for c in LIVE_S if M1[k, c] != M1[k + 1, c]],
                   calls_changed={p: float(v.mean()) for p, v in chg.items()},
                   calls_changed_ci={p: boot(lambda i: v[i].mean(), n) for p, v in chg.items()},
                   mcnemar_mask_vs_ones_all4=C.mcnemar(chg['ones_all4'], chg['mask']), units={})
        for c in LIVE_S:
            u = {}
            for p in dS:
                a = np.abs(dS[p][:, c]); u[p] = dict(mean_abs=float(a.mean()), ci=boot(lambda i: a[i].mean(), n),
                                                     frac_gt_lsb=float((a > LSB[c] * (1 + 1e-6)).mean()))
            u['r2_net_vs_mask'] = r2(dS['net'][:, c], dS['mask'][:, c])
            am, a1 = np.abs(dS['mask'][:, c]), np.abs(dS['ones_all4'][:, c]); dd = am - a1
            try:
                pw = float(stats.wilcoxon(am, a1).pvalue) if (dd != 0).any() else 1.0
            except ValueError:
                pw = 1.0
            u['mask_minus_ones_all4'] = dict(diff=float(dd.mean()), ci=boot(lambda i: dd[i].mean(), n), wilcoxon_p=pw)
            ps.append(pw); keys.append((e, c)); rec['units'][c] = u
        out['edits'][e] = rec
        log(f'{e}: N={n} calls changed net {100 * chg["net"].mean():.2f} / mask {100 * chg["mask"].mean():.2f} / ones_all4 '
            f'{100 * chg["ones_all4"].mean():.2f} %; mean|dS| mask -> ones_all4: '
            + ' '.join(f'S{c}:{rec["units"][c]["mask"]["mean_abs"]:.3f}->{rec["units"][c]["ones_all4"]["mean_abs"]:.3f}'
                       for c in (0, 43, 52, 63, 19, 1, 29, 30, 13, 17)))
    hp = C.holm(ps)
    for (e, c), h in zip(keys, hp): out['edits'][e]['units'][c]['mask_minus_ones_all4']['holm_p'] = h
    out['holm_m'] = len(ps); out['wall_s'] = time.time() - T0
    json.dump(out, open(OUT / 'e20_programs.json', 'w'), indent=1, default=float); log('done; wrote', OUT / 'e20_programs.json')


if __name__ == '__main__':
    if '--programs' in sys.argv:
        programs()
    else:
        main()
