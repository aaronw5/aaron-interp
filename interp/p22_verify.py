"""P22 verification (loop C). Re-measures the P22 axis-response ratio R on a fresh sample and adds three checks:
  1. fresh rows: test_20M/{ZJetsToNuNu,ZToQQ,WToQQ}_105.root (P22 used _104), 5,000 random rows per file (default_rng(3105 + class)),
     N = 15,000; rebuilt unedited through part_core.build.
  2. other rigid shifts about the stored axis: phi - 0.06 and eta + 0.06 (eta = jet_eta + sign(jet_eta) deta), each with its own
     per-jet matched dilation (k_j = sqrt(M0M2(shift) / M0M2(base))); R for phi + 0.06 as in P22; and R as a ratio of means.
  3. R across training epochs of the true-label arms t and t2 (epoch-0, 2, 4, 6, 8 checkpoints and final = epoch 9), to see whether
     R grows with optimisation steps at fixed data (an alternative P22 lists as untested).
Models: P22 arms t, t2, u, a2; P14 arm a; released ParT_kin and ParT_full. Read-out d as P22 (l_W - l_Z; window arms l_in - l_out).
Statistics: per-jet slope through 0 with 2,000-resample jet bootstrap; permutation tests (per-jet swap) for (u, t), (t, a),
(t, released_kin), (t epoch 0, t final), (t2 epoch 0, t2 final), Holm over the family; McNemar on W -> Z under phi + 0.06 (Holm).
Route shares (P2 method) for the phi + 0.06 matched dilation.

  ./part_job.sh c-p22-verify 'python -m interp.p22_verify'        -> $AI_RESULTS/part/p22_verify.json, p22_verify.png
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp.part_core import build, kin, moments, inputs, OUT, CL
from interp.e19_common import mcnemar, holm
from interp.p14_planted_train import masses, MOD as MOD14
from interp.p14_planted_part import slope_boot, boot_weights, perm_test, ci
from interp.p22_planted_true_train import MOD as MOD22
from interp.p22_planted_true_part import sample, Model, iW, iZ, iQ

T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


def load(name, ckpt):
    """Model(name) with its weights replaced by checkpoint `ckpt` (e.g. epoch-0.pt) of the same arm"""
    M = Model(name, False)
    if ckpt != 'final.pt':
        d = (MOD14 if name == 'a' else MOD22) / name; f = d / ckpt
        M.m.load_state_dict(M.t.load(f, map_location=M.dev)); M.path = str(f); M.m.eval()
    return M


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--per', type=int, default=5000); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.smoke: a.per = 300
    fn = OUT / ('p22_verify_smoke.json' if a.smoke else 'p22_verify.json'); OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3222)
    J0 = sample(a.per, seed=3105, idx=105); y10 = J0['label'].astype(int); N = len(y10); isW = y10 == iW
    pt, de, dp, ms = kin(J0); real = pt > 0
    Jb = build(J0, pt, de, dp, ms); Mb = moments(*kin(Jb)[:3])['M0M2']; okk = Mb > 0
    E = {'phi+': build(J0, pt, de, dp + 0.06 * real, ms), 'phi-': build(J0, pt, de, dp - 0.06 * real, ms),
         'eta+': build(J0, pt, de + 0.06 * real, dp, ms)}
    for s in ('phi+', 'phi-', 'eta+'):
        M2 = moments(*kin(E[s])[:3])['M0M2']; kj = np.where(okk, np.sqrt(M2 / np.where(okk, Mb, 1.0)), 1.0)[:, None]
        E['dil_' + s] = build(J0, pt, de * kj, dp * kj, ms)
    Jr = dict(Jb); Jr['jet_phi'] = (Jb['jet_phi'] + 0.06).astype(np.float32); c_, s_ = np.cos(0.06), np.sin(0.06)
    Jr['px'] = (c_ * Jb['px'] - s_ * Jb['py']).astype(np.float32); Jr['py'] = (s_ * Jb['px'] + c_ * Jb['py']).astype(np.float32); E['rot'] = Jr
    mi, mA = masses(Jb); edm = {k: masses(Je) for k, Je in E.items()}
    res = dict(N=N, per_class=a.per, files=[f'test_20M/{c}_105.root' for c in ('ZJetsToNuNu', 'ZToQQ', 'WToQQ')], seed=3105,
               edits={k: dict(dlogminv=float(np.mean(np.log(np.maximum(edm[k][0], 1)) - np.log(np.maximum(mi, 1)))),
                              dlogmA=float(np.mean(np.log(np.maximum(edm[k][1], 1e-6)) - np.log(np.maximum(mA, 1e-6))))) for k in E}, models={})
    log(f'{N} jets; ' + '; '.join(f'{k}: dlog m_inv {v["dlogminv"]:+.4f} dlog m_A {v["dlogmA"]:+.4f}' for k, v in res['edits'].items()))
    X = {kd: {k: inputs(Je, kd) for k, Je in [('base', Jb)] + list(E.items())} for kd in ('kin', 'full')}
    Wb = boot_weights(N, rng); dd = {}; plist, pkeys = [], []
    runs = [(n, 'final.pt') for n in ('t', 't2', 'u', 'a', 'a2', 'released_kin', 'released_full')]
    if not a.smoke: runs += [(n, f'epoch-{e}.pt') for n in ('t', 't2') for e in (0, 2, 4, 6, 8)]
    for name, ck in runs:
        key = name if ck == 'final.pt' else f'{name}@{ck[:-3]}'
        M = load(name, ck); Xk = X[M.kind]; t1 = time.time()
        L0, _ = M.run(*Xk['base']); d0 = M.read(L0); row = dict(path=M.path, outputs=M.K)
        if M.K == 2:
            own = (np.abs(mi - 85.0) < 10.0).astype(int); row['acc_own_label'] = float((L0.argmax(1) == own).mean())
        else:
            ok = M.call3(L0) == y10; row['acc3'] = float(ok.mean())
            if M.K == 10: row['acc10'] = float((L0.argmax(1) == y10).mean())
        D = {k: M.read(M.run(*Xk[k])[0]) - d0 for k in E}
        Ls = {}
        for s in ('phi+', 'phi-', 'eta+'):
            R, Rb = slope_boot(D[s], D['dil_' + s], Wb)
            row[f'R_{s}'] = [R] + ci(Rb)
            row[f'Rmeans_{s}'] = [float(D[s].mean() / D['dil_' + s].mean())] + ci((Wb @ D[s]) / (Wb @ D['dil_' + s]))
        Rk, Rkb = slope_boot(D['rot'], D['dil_phi+'], Wb); row['R_rot_null'] = [Rk] + ci(Rkb)
        row['dd_mean'] = {k: float(v.mean()) for k, v in D.items()}; row['mean_abs_dd_dil'] = float(np.abs(D['dil_phi+']).mean())
        dd[key] = (D['phi+'], D['dil_phi+'])
        if M.K != 2:
            L1, _ = M.run(*Xk['phi+']); p0 = M.call3(L0)[isW] == iZ; p1 = M.call3(L1)[isW] == iZ
            mc = mcnemar(p0, p1); plist.append(mc['p']); pkeys.append(key); Bi = rng.integers(0, isW.sum(), (2000, isW.sum()))
            row['WtoZ_phi+'] = dict(before=float(p0.mean()), after=float(p1.mean()), change=[float(p1.mean() - p0.mean())] + ci(p1[Bi].mean(1) - p0[Bi].mean(1)), mcnemar=mc)
        if ck == 'final.pt':
            k = 'dil_phi+'; Lpp, _ = M.run(Xk[k][0], Xk['base'][1], Xk['base'][2]); Lpr, _ = M.run(Xk['base'][0], Xk[k][1], Xk['base'][2])
            s1, b1 = slope_boot(M.read(Lpp) - d0, D[k], Wb); s2, b2 = slope_boot(M.read(Lpr) - d0, D[k], Wb)
            row['routes_dil_phi+'] = dict(per_particle=[s1] + ci(b1), pair=[s2] + ci(b2))
        res['models'][key] = row; del M
        log(f'{key}: ' + (f'acc own {row["acc_own_label"]:.4f}' if 'acc_own_label' in row else f'acc3 {row["acc3"]:.4f}' + (f' acc10 {row["acc10"]:.4f}' if 'acc10' in row else ''))
            + '; ' + '; '.join(f'R {s} {row["R_" + s][0]:+.3f} [{row["R_" + s][1]:+.3f}, {row["R_" + s][2]:+.3f}] (means {row["Rmeans_" + s][0]:+.3f})' for s in ('phi+', 'phi-', 'eta+'))
            + f'; rot null {Rk:+.4f}; |dd dil| {row["mean_abs_dd_dil"]:.3f}'
            + (f'; W->Z {row["WtoZ_phi+"]["before"]:.3f} -> {row["WtoZ_phi+"]["after"]:.3f}' if 'WtoZ_phi+' in row else '')
            + (f'; route pp {row["routes_dil_phi+"]["per_particle"][0]:.3f} pair {row["routes_dil_phi+"]["pair"][0]:.3f}' if 'routes_dil_phi+' in row else '')
            + f' ({time.time() - t1:.0f} s)')
        json.dump(res, open(fn, 'w'), indent=1, default=float)
    for k, p in zip(pkeys, holm(plist)): res['models'][k]['WtoZ_phi+']['mcnemar']['p_holm'] = p
    res['perm'] = {}; pp = []
    for x1, x2 in (('u', 't'), ('t', 'a'), ('t', 'released_kin'), ('t@epoch-0', 't'), ('t2@epoch-0', 't2')):
        if x1 in dd and x2 in dd:
            res['perm'][f'{x1}_vs_{x2}'] = perm_test(dd[x1][0], dd[x1][1], dd[x2][0], dd[x2][1], rng); pp.append(f'{x1}_vs_{x2}')
    for k, p in zip(pp, holm([res['perm'][k]['p'] for k in pp])):
        res['perm'][k]['p_holm'] = p; log(f'permutation H0 {k}: diff {res["perm"][k]["diff"]:+.3f}, p {res["perm"][k]["p"]:.4f}, Holm {p:.4f}')
    json.dump(res, open(fn, 'w'), indent=1, default=float); log(f'wrote {fn}')
    try:
        plot(res, fn.with_suffix('.png'))
    except Exception as ex:
        log(f'plot failed: {ex!r}')
    log('P22 verify DONE')


def plot(res, path):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1.6, 1]))
    names = [n for n in ('t', 't2', 'u', 'a', 'a2', 'released_kin', 'released_full') if n in res['models']]
    for j, (s, col) in enumerate((('phi+', '#3b6fb6'), ('phi-', '#7fa7d9'), ('eta+', '#d08a2c'))):
        for i, n in enumerate(names):
            r = res['models'][n]['R_' + s]; x = i + (j - 1) * 0.25
            ax.bar(x, r[0], 0.24, color=col, label=s.replace('+', ' + 0.06').replace('-', ' - 0.06') if i == 0 else None)
            ax.errorbar(x, r[0], yerr=[[r[0] - r[1]], [r[2] - r[0]]], color='k', capsize=2, lw=0.8)
    ax.axhline(0, color='k', lw=0.8); ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel('R = response to rigid shift about stored axis\n/ response to matched real dilation'); ax.legend(fontsize=8, title='shift', title_fontsize=8)
    ax.set_title('Fresh sample (test_20M *_105): R by model and shift direction', fontsize=9)
    for n, col in (('t', '#3b6fb6'), ('t2', '#d08a2c')):
        ep = [e for e in (0, 2, 4, 6, 8) if f'{n}@epoch-{e}' in res['models']]
        rr = [res['models'][f'{n}@epoch-{e}']['R_phi+'] for e in ep] + ([res['models'][n]['R_phi+']] if n in res['models'] else [])
        ee = ep + ([9] if n in res['models'] else [])
        if ee: bx.errorbar(ee, [r[0] for r in rr], yerr=[[r[0] - r[1] for r in rr], [r[2] - r[0] for r in rr]], marker='o', color=col, capsize=2, label=f'true labels, arm {n}')
    for n, ls in (('released_kin', '--'), ('released_full', ':')):
        if n in res['models']: bx.axhline(res['models'][n]['R_phi+'][0], color='#777', ls=ls, lw=1, label=n.replace('_', ' '))
    bx.set_xlabel('training epoch (1.5M jets per epoch)'); bx.set_ylabel('R (phi + 0.06)'); bx.legend(fontsize=7); bx.set_ylim(-0.05, 1)
    bx.set_title('R across training of the true-label arms', fontsize=9)
    fig.text(0.5, -0.02, f'N = {res["N"]} W/Z/QCD jets; per-jet slope through 0, 95 % bootstrap over jets (2,000).', ha='center', fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches='tight')


if __name__ == '__main__':
    if '--plot' in sys.argv:
        f = Path(sys.argv[sys.argv.index('--plot') + 1]); plot(json.load(open(f)), f.with_suffix('.png'))
    else:
        main()
