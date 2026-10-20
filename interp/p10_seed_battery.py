"""P10 — ParT retrained from scratch, 3 seeds per feature set, on L-GATr's 2M JetClass subset: interim axis battery.

Question (claim C3, weakness 3 "single checkpoint", confound C6 "data scale"): is the axis-relative (not invariant-mass)
W/Z response of the released ParT_full / ParT_kin a property of training ParT on JetClass, or of that one checkpoint and its
100M-jet training set? The trainings (interp/p10_part_seeds_train.sh, jobs c-p10-full / c-p10-kin) use the authors'
train_JetClass.sh on the 2M subset of P5 (/j-jepa-vol/aaron-interp/data/jetclass2m/selection.json), seeds 0 / 1 / 2, per-epoch
checkpoints in /j-jepa-vol/aaron-interp/models/part_seeds/<feat>_s<seed>/net_epoch-<e>_state.pt.

Sample: 1,000 test jets per class at random rows (numpy default_rng(2026), without replacement) of each class's test file 100
(the file P1 used; P1 took rows 3,000-4,000), 10,000 jets. Baseline = the jets rebuilt unedited with part_core.build.
Models: released ParT_full / ParT_kin (reference on the same jets) and each seed's checkpoint at epoch E (--epoch: 'first' =
the earliest epoch saved by all seeds of that feature set that have saved one, 'last' = the latest common one, or an integer: every seed that has saved that epoch).
Edits (P1 / P1v definitions):
  phi0.06 : every real particle's Δφ + 0.06 rad, stored axis kept, 4-vectors rebuilt (a rigid rotation of the particles:
            invariant mass unchanged; ½Δlog(M0·M2) about the stored axis > 0).
  k1.1    : every particle's (Δη, Δφ) × 1.1 about the stored axis (real mass change, Δlog m ≈ +0.094): positive control.
  wta     : Δη, Δφ, ΔR recomputed about the winner-take-all axis (P1v check 4; 4-vectors unchanged).
  escheme : the same recomputation about the E-scheme axis from the stored particles (control for wta: an axis change of
            the stored-axis convention's own kind).
  rot0.06 : the whole jet rotated by 0.06 rad in φ including the stored axis (inputs equal up to float rounding): null of
            the numerical floor.
Metrics per model and edit: accuracy; fraction of calls changed (Wilson 95 % CI); W→Z = fraction of true-W jets called Z,
before → after, paired difference with 95 % bootstrap CI over W jets (2,000 resamples) and exact McNemar test (H0: the edit
does not change the W→Z indicator; invariant-mass hypothesis H_inv predicts no change for phi0.06, wta and escheme beyond the
rot0.06 floor; the axis hypothesis H_axis predicts W→Z up for phi0.06, as in the released ParT 12 → 31 % (full) and
16 → 37 % (kin) on the P1 rows); mean Δ(logit_W − logit_Z) on true-W jets with bootstrap CI; the logit-ratio
R = mean Δ(lW − lZ)|phi0.06 / mean Δ(lW − lZ)|k1.1 on W jets (bootstrap CI, paired jets) against the ratio the axis-moment
model √(M0·M2) predicts, ½Δlog(M0M2)|phi0.06 / ½Δlog(M0M2)|k1.1 (H_inv predicts R ≈ 0). Holm correction over all McNemar
tests of the run.

Pre-registered (written 2026-10-19, before any final checkpoint exists; tested in a later round on the final checkpoints of
each seed, N = 10,000 random test jets as here, a fresh analysis seed):
  S1 (C3) calibrated read-out ratio: the ridge read-out of log m from the class token (P1 / R1 procedure) moves under
     phi0.06 by 0.7-1.4 of the √(M0M2) prediction after calibration against a matched real change (vs ~0 predicted for an
     invariant-mass read-out), in every seed.
  S2 (C3 route) the per-particle ΔR route (P1v (2b) / P2: per-particle features edited, 4-vectors unedited) carries > 0.8 of the
     phi0.06 read-out change in every full seed (kin: Δφ and ΔR together).
  S3 (block-level route) per seed, the n98-analogue = the class-token unit with the largest |W_W − W_Z| last-layer weight × sd;
     heads of particle block 1 (by resample ablation of the head outputs) carry > 0.9 of its response to k1.1.
  S4 (threshold head) each seed has a class-attention or particle head whose attention to the W-mass pair switches with jet
     mass (logistic fit of head output vs m: midpoint m50 within 60-110 GeV and width < 15 GeV), as b2h8 in the released model.
  Expected under the thesis: every seed shows phi0.06 W→Z up and wta calls changed at the released model's order, and
  S1-S4 hold with different head / unit indices. Alternative: axis dependence varies strongly with the seed or vanishes at 2M
  jets, which would make C3 a property of the released checkpoint / its data scale.
  This interim run (early epochs) only measures the call-level quantities above; S1-S4 are not tested here.

  ./part_job.sh c-p10-run 'python -m interp.p10_seed_battery --epoch first'
  -> $AI_RESULTS/part/p10_battery_<tag>.json (+ p10_battery_<tag>.png)
"""
import argparse, glob, json, os, re, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp.part_core import ParTNetwork, inputs, read_root, fof, kin, build, jmass, moments, cat, OUT, CL
from interp.p1v_verify_part import fwd, axis_set, reaxis, wta_selftest
from interp.e19_common import mcnemar, holm, wilson

t0 = time.time()
log = lambda *a: print(f'[{time.time() - t0:7.1f}s]', *a, flush=True)
iW, iZ = CL.index('Wqq'), CL.index('Zqq')
SEEDS_DIR = '/j-jepa-vol/aaron-interp/models/part_seeds'
REF = dict(full=dict(phi_WZ=(0.12, 0.31), wta_changed=0.117), kin=dict(phi_WZ=(0.16, 0.37), wta_changed=0.341))


def random_sample(per, seed):
    rng = np.random.default_rng(seed); Js, rows = [], {}
    import uproot
    for _ in range(5):              # files.json is rewritten in place by ./sync.sh runs of other loops: read all paths once, retry
        try:
            F = {c: fof(c) for c in CL}; break
        except json.JSONDecodeError:
            time.sleep(10)
    for c in CL:
        f = F[c]; n = uproot.open(f)['tree'].num_entries
        r = np.sort(rng.choice(n, per, replace=False)); J = read_root(f)
        Js.append({k: (v[r] if np.ndim(v) else v) for k, v in J.items()}); rows[c] = dict(file=str(f), n_entries=int(n), first=r[:5].tolist())
        del J; log(f'sample {c}: {per} random rows of {n} ({f})')
    return cat(Js), rows


def boot_diff(a, b, rng, B=2000):
    """paired mean difference b − a with a 95 % bootstrap interval"""
    d = np.asarray(b, np.float64) - np.asarray(a, np.float64); n = len(d); idx = rng.integers(0, n, (B, n)); m = d[idx].mean(1)
    return [float(d.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def boot_ratio_paired(num, den, rng, B=2000):
    num, den = np.asarray(num, np.float64), np.asarray(den, np.float64); n = len(num); idx = rng.integers(0, n, (B, n))
    r = num[idx].mean(1) / den[idx].mean(1)
    return [float(num.mean() / den.mean()), float(np.quantile(r, .025)), float(np.quantile(r, .975))]


def load(n, path):
    """ParTNetwork with the weights of a weaver checkpoint (keys 'mod.*' of ParticleTransformerWrapper)"""
    net = ParTNetwork(n)
    if path is None: return net
    import torch
    sd = torch.load(path, map_location='cpu'); sd = {k[4:] if k.startswith('mod.') else k: v for k, v in sd.items()}
    net.model.load_state_dict(sd, strict=True); net.model.eval()
    W = net.model.fc[0].weight.detach().cpu().double().numpy(); b = net.model.fc[0].bias.detach().cpu().double().numpy()
    net.last = (W.T.copy(), b.copy(), None, None); return net


def pick_epoch(feat, how):
    eps = {}
    for d in sorted(glob.glob(f'{SEEDS_DIR}/{feat}_s*')):
        s = int(d.rsplit('_s', 1)[1])
        eps[s] = sorted(int(re.search(r'epoch-(\d+)_state', f).group(1)) for f in glob.glob(f'{d}/net_epoch-*_state.pt'))
    eps = {s: v for s, v in eps.items() if v}            # seeds still queued / without a saved epoch are left out (reported)
    if how not in ('first', 'last'):                       # an integer: every seed that has saved that epoch
        e = int(how); return e, {s: v for s, v in eps.items() if e in v}
    common = sorted(set.intersection(*[set(v) for v in eps.values()])) if eps else []
    if not common: return None, eps
    return (common[0] if how == 'first' else common[-1]), eps


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--epoch', default='first'); ap.add_argument('--per', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=2026); ap.add_argument('--tag', default=None); a = ap.parse_args()
    rng = np.random.default_rng(a.seed + 1); OUT.mkdir(parents=True, exist_ok=True)
    J0, rows = random_sample(a.per, a.seed); y = J0['label'].astype(int); N = len(y); W = y == iW
    pt, de, dp, ms = kin(J0); real = pt > 0
    Jb = build(J0, pt, de, dp, ms); M0 = moments(*kin(Jb)[:3]); m0 = jmass(Jb)
    E = {'phi0.06': build(J0, pt, de, dp + 0.06 * real, ms), 'k1.1': build(J0, pt, de * 1.1, dp * 1.1, ms)}
    Jr = dict(Jb); Jr['jet_phi'] = (Jb['jet_phi'] + 0.06).astype(np.float32); c, s_ = np.cos(0.06), np.sin(0.06)
    Jr['px'] = (c * Jb['px'] - s_ * Jb['py']).astype(np.float32); Jr['py'] = (s_ * Jb['px'] + c * Jb['py']).astype(np.float32); E['rot0.06'] = Jr
    A, ainfo = axis_set(dict(Jb=Jb)); E['wta'] = reaxis(Jb, *A['wta'])[0]; E['escheme'] = reaxis(Jb, *A['escheme'])[0]
    pred = {k: 0.5 * np.log(moments(*kin(Je)[:3])['M0M2'] / M0['M0M2']) for k, Je in E.items() if k in ('phi0.06', 'k1.1', 'wta', 'escheme')}
    dlogm = {k: np.log(np.maximum(jmass(Je), 1.0)) - np.log(np.maximum(m0, 1.0)) for k, Je in E.items()}
    res = dict(N=N, per_class=a.per, seed=a.seed, rows=rows, wta_selftest=wta_selftest(), axes_info=ainfo, mass_median=float(np.median(m0)),
               edits={k: dict(pred_half_dlogM0M2_W=float(pred[k][W].mean()) if k in pred else None, dlogm_mean=float(dlogm[k].mean()),
                              dlogm_absmax=float(np.abs(dlogm[k]).max())) for k in E},
               reference_released_P1=REF, models={})
    log(f'{N} jets; mass median {np.median(m0):.1f} GeV; WTA = hardest particle in {100 * ainfo["wta_is_hardest"]:.1f} %; edits: '
        + '; '.join(f'{k} pred {v["pred_half_dlogM0M2_W"]} dlogm {v["dlogm_mean"]:+.4f}' for k, v in res['edits'].items()))
    pred_ratio = float(pred['phi0.06'][W].mean() / pred['k1.1'][W].mean()); res['M2_pred_ratio_phi_over_k'] = pred_ratio
    plist, pkeys = [], []
    for feat in ('full', 'kin'):
        e, eps = pick_epoch(feat, a.epoch); res.setdefault('epochs_available', {})[feat] = eps; res.setdefault('epoch_used', {})[feat] = e
        models = [(f'released_{feat}', None)] + ([(f'{feat}_s{s}@e{e}', f'{SEEDS_DIR}/{feat}_s{s}/net_epoch-{e}_state.pt') for s in sorted(eps)] if e is not None else [])
        log(f'{feat}: epochs per seed {eps}; using epoch {e}')
        X = {k: inputs(Je, feat) for k, Je in [('base', Jb)] + list(E.items())}
        for name, path in models:
            net = load(feat, path); t1 = time.time()
            z0, L0 = fwd(net, *X['base']); c0 = L0.argmax(1); g0 = L0[:, iW] - L0[:, iZ]
            row = dict(path=path, acc=float((c0 == y).mean()), own=[float((c0[y == k] == k).mean()) for k in range(10)],
                       WZ0=float((c0[W] == iZ).mean()), edits={})
            dg = {}
            for k in E:
                z, L = fwd(net, *X[k]); c = L.argmax(1); g = L[:, iW] - L[:, iZ]; dg[k] = (g - g0)[W]
                ch = int((c != c0).sum()); a0, a1 = c0[W] == iZ, c[W] == iZ
                mc = mcnemar(a0, a1); plist.append(mc['p']); pkeys.append((name, k))
                row['edits'][k] = dict(acc=float((c == y).mean()), changed=ch / N, changed_ci=wilson(ch, N),
                                       WZ=[float(a0.mean()), float(a1.mean())], dWZ=boot_diff(a0, a1, rng), mcnemar_WZ=mc,
                                       ZW=[float((c0[y == iZ] == iW).mean()), float((c[y == iZ] == iW).mean())],
                                       dlogit_WZ_on_W=boot_diff(np.zeros(W.sum()), dg[k], rng))
            row['logit_ratio_phi_over_k'] = boot_ratio_paired(dg['phi0.06'], dg['k1.1'], rng)
            res['models'][name] = row; del net
            ed = row['edits']
            log(f'  {name}: acc {100 * row["acc"]:.2f} %; W→Z phi0.06 {100 * ed["phi0.06"]["WZ"][0]:.1f} → {100 * ed["phi0.06"]["WZ"][1]:.1f} % '
                f'(Δ {100 * ed["phi0.06"]["dWZ"][0]:+.1f} [{100 * ed["phi0.06"]["dWZ"][1]:+.1f}, {100 * ed["phi0.06"]["dWZ"][2]:+.1f}], McNemar p {ed["phi0.06"]["mcnemar_WZ"]["p"]:.1e}); '
                f'k1.1 W→Z → {100 * ed["k1.1"]["WZ"][1]:.1f} %; wta changed {100 * ed["wta"]["changed"]:.1f} % (escheme {100 * ed["escheme"]["changed"]:.1f} %, rot {100 * ed["rot0.06"]["changed"]:.2f} %); '
                f'logit ratio phi/k {row["logit_ratio_phi_over_k"][0]:+.2f} [{row["logit_ratio_phi_over_k"][1]:+.2f}, {row["logit_ratio_phi_over_k"][2]:+.2f}] '
                f'vs M2 prediction {pred_ratio:.2f} ({time.time() - t1:.0f} s)')
    hp = holm(plist)
    for (name, k), p in zip(pkeys, hp): res['models'][name]['edits'][k]['mcnemar_WZ']['p_holm'] = p
    tag = a.tag or f'e{a.epoch}'
    json.dump(res, open(OUT / f'p10_battery_{tag}.json', 'w'), indent=1); log(f'wrote {OUT}/p10_battery_{tag}.json')
    try:
        plot(res, OUT / f'p10_battery_{tag}.png')
    except Exception as ex:
        log(f'plot failed: {ex!r}')
    log('P10 battery DONE')


def plot(res, path):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, feat in zip(axs, ('full', 'kin')):
        names = [n for n in res['models'] if n.startswith(f'released_{feat}') or n.startswith(f'{feat}_s')]
        for i, n in enumerate(names):
            r = res['models'][n]['edits']
            for j, (k, col) in enumerate((('phi0.06', '#3b6fb6'), ('k1.1', '#d08a2c'), ('rot0.06', '#999999'))):
                d = r[k]['dWZ']; x = i + (j - 1) * 0.25
                ax.bar(x, 100 * d[0], 0.24, color=col, label=k if i == 0 else None)
                ax.errorbar(x, 100 * d[0], yerr=[[100 * (d[0] - d[1])], [100 * (d[2] - d[0])]], color='k', lw=1, capsize=2)
        ax.set_xticks(range(len(names))); ax.set_xticklabels([n.replace(f'released_{feat}', 'released\n(100M jets)').replace(f'{feat}_', '').replace('@', '\n') for n in names], fontsize=8)
        ax.axhline(0, color='k', lw=0.6); ax.set_title(f'ParT_{feat}: change in true W called Z'); ax.grid(axis='y', alpha=0.3)
    axs[0].set_ylabel('Δ(W→Z) [percentage points], 95 % CI'); axs[0].legend(title='edit', fontsize=8)
    fig.text(0.5, -0.02, f'phi0.06: particles rotated 0.06 rad about the stored axis (invariant mass unchanged); k1.1: angles ×1.1 (mass +10 %); '
             f'rot0.06: whole jet incl. axis rotated (null). N = {res["N"]} random test jets.', ha='center', fontsize=7.5)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches='tight')


if __name__ == '__main__':
    main()
