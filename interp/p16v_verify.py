"""P16v: verifier of P16 (planted-label ParT arms), on fresh rows.

Sample: N = 20,000 random rows (numpy default_rng(16)) of the union of JetClass test_20M/{ZJetsToNuNu,ZToQQ,WToQQ}_101.root
(P16 used the _100 files with seed 14). Edits, read-outs and R exactly as interp/p14_planted_part.py (phi = dphi + 0.06 rad about
the stored axis; dil = per-jet dilation with matched Delta log m_A; rot = rigid rotation incl. the axis, null).

Added checks:
  1. training-length scan: R of arm a at epochs 0, 2, 5, 7, 9 and arm b at epochs 2, 5 (does R grow with training?)
  2. released read-out dependence: R of released ParT_full / ParT_kin with (i) logit_W - logit_Z (P16), (ii) the resonance-vs-QCD
     read-out logsumexp(logit_W, logit_Z) - logit_QCD, which is closer to a mass-window read-out like the planted arms'
  3. window-edge accuracy of arms a, b, c in bins of |m_inv - 85 GeV| (unedited and after phi)
  4. route split of the matched dilation (per-particle only / pair only) for a, b, c and the released models
Permutation tests (paired per jet, 2,000 permutations): R_a = R_b, R_a = R_released_kin, R_a(ep 0) = R_a(ep 9).

  ./part_job.sh c-p16-verify 'python -m interp.p16v_verify'      -> $AI_RESULTS/part/p16_verify.json
"""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp.part_core import read_root, build, kin, moments, cat, inputs, OUT, CL
from interp.e19_common import mcnemar, wilson
from interp.p14_planted_train import masses, CLS3, JC, SIDE, M0_, W0_
from interp.p14_planted_part import Model, slope_boot, boot_weights, perm_test, ci

T0 = time.time()
iW, iZ, iQ = CL.index('Wqq'), CL.index('Zqq'), CL.index('QCD')


def log(*a):
    print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


def sample(n, seed=16, part='101'):
    import uproot
    fs = [JC / 'test_20M' / f'{c}_{part}.root' for c in CLS3]; ne = [uproot.open(f)['tree'].num_entries for f in fs]
    rng = np.random.default_rng(seed); g = np.sort(rng.choice(sum(ne), n, replace=False)); off = np.cumsum([0] + ne); Js = []
    for i, f in enumerate(fs):
        r = g[(g >= off[i]) & (g < off[i + 1])] - off[i]; J = read_root(f)
        Js.append({k: (v[r] if np.ndim(v) else v) for k, v in J.items()}); del J; log(f'sample {f.name}: {len(r)} rows of {ne[i]}')
    return cat(Js)


def main():
    rng = np.random.default_rng(1616); OUT.mkdir(parents=True, exist_ok=True)
    J0 = sample(20000); y10 = J0['label'].astype(int); N = len(y10)
    pt, de, dp, ms = kin(J0); real = pt > 0
    Jb = build(J0, pt, de, dp, ms); Mb = moments(*kin(Jb)[:3])
    E = {'phi': build(J0, pt, de, dp + 0.06 * real, ms)}
    M2p = moments(*kin(E['phi'])[:3])['M0M2']; okk = Mb['M0M2'] > 0
    kj = np.where(okk, np.sqrt(M2p / np.where(okk, Mb['M0M2'], 1.0)), 1.0)[:, None]
    E['dil'] = build(J0, pt, de * kj, dp * kj, ms)
    Jr = dict(Jb); Jr['jet_phi'] = (Jb['jet_phi'] + 0.06).astype(np.float32); c_, s_ = np.cos(0.06), np.sin(0.06)
    Jr['px'] = (c_ * Jb['px'] - s_ * Jb['py']).astype(np.float32); Jr['py'] = (s_ * Jb['px'] + c_ * Jb['py']).astype(np.float32); E['rot'] = Jr
    mi, mA = masses(Jb); side = np.load(SIDE / 'labels.npz'); c, w = float(side['c']), float(side['w'])
    lab = dict(a=(np.abs(mi - M0_) < W0_).astype(int), b=(np.abs(mA - c) < w).astype(int)); lab['c'] = lab['a']
    edm = {k: masses(Je) for k, Je in E.items()}
    labE = {k: dict(a=(np.abs(edm[k][0] - M0_) < W0_).astype(int), b=(np.abs(edm[k][1] - c) < w).astype(int)) for k in E}
    for k in E: labE[k]['c'] = labE[k]['a']
    res = dict(N=N, seed=16, files='test_20M/{ZJetsToNuNu,ZToQQ,WToQQ}_101.root', c=c, w=w,
               class_counts={n: int((y10 == CL.index(n)).sum()) for n in ('QCD', 'Zqq', 'Wqq')},
               pos_rate=dict(a=float(lab['a'].mean()), b=float(lab['b'].mean())), ab_agree=float((lab['a'] == lab['b']).mean()),
               corr_minv_mA=float(np.corrcoef(mi, mA)[0, 1]),
               edits={k: dict(dlogminv=float(np.mean(np.log(np.maximum(edm[k][0], 1)) - np.log(np.maximum(mi, 1)))),
                              dlogmA=float(np.mean(np.log(np.maximum(edm[k][1], 1e-6)) - np.log(np.maximum(mA, 1e-6)))),
                              label_flip=dict(a=float((labE[k]['a'] != lab['a']).mean()), b=float((labE[k]['b'] != lab['b']).mean()))) for k in E},
               models={})
    log(json.dumps(res['edits']))
    X = {kd: {k: inputs(Je, kd) for k, Je in [('base', Jb)] + list(E.items())} for kd in ('kin', 'full')}
    Wb = boot_weights(N, rng); dd = {}
    bins = [0, 2, 5, 8, 10, 12, 15, 20, 1e9]; am = np.abs(mi - M0_); bidx = np.digitize(am, bins) - 1
    specs = [('a', None), ('b', None), ('c', None), ('a', 0), ('a', 2), ('a', 5), ('a', 7), ('b', 2), ('b', 5), ('released_full', None), ('released_kin', None)]
    for name, ep in specs:
        key = name if ep is None else f'{name}_ep{ep}'
        M = Model(name, False, ep); Xk = X[M.kind]; t1 = time.time()
        L0, _ = M.run(*Xk['base']); c0 = L0.argmax(1); row = dict(path=getattr(M, 'path', None))
        reads = {'default': M.read}
        if not M.arm:
            reads['res_vs_QCD'] = lambda L: np.logaddexp(L[:, iW], L[:, iZ]) - L[:, iQ]
            row['acc10'] = float((c0 == y10).mean())
        else:
            own = lab[M.arm]; ok0 = c0 == own; row['acc_own'] = [float(ok0.mean())] + wilson(int(ok0.sum()), N)
        Ls = {k: M.run(*Xk[k])[0] for k in E}
        for rn, f in reads.items():
            d0 = f(L0); Dd = {k: f(Ls[k]) - d0 for k in E}
            R, Rb = slope_boot(Dd['phi'], Dd['dil'], Wb); Rk, _ = slope_boot(Dd['rot'], Dd['dil'], Wb)
            row.setdefault('R', {})[rn] = [R] + ci(Rb); row.setdefault('R_rot_null', {})[rn] = Rk
            if rn == 'default': dd[key] = (Dd['phi'], Dd['dil']); D0 = Dd
            if not M.arm:
                for cl, ix in (('W', iW), ('Z', iZ), ('QCD', iQ)):
                    s = y10 == ix; r_, rb_ = slope_boot(Dd['phi'][s], Dd['dil'][s], boot_weights(int(s.sum()), rng, 500))
                    row.setdefault('R_by_class', {}).setdefault(rn, {})[cl] = [r_] + ci(rb_)
        for k in E:
            ch = int((Ls[k].argmax(1) != c0).sum()); row.setdefault('changed', {})[k] = [ch / N] + wilson(ch, N)
        if M.arm:
            okp = Ls['phi'].argmax(1) == own
            row['mcnemar_pos_phi'] = mcnemar(c0 == 1, Ls['phi'].argmax(1) == 1)
            row['acc_own_after_phi'] = float(okp.mean()); row['acc_own_label_after_phi_edited_labels'] = float((Ls['phi'].argmax(1) == labE['phi'][M.arm]).mean())
            row['acc_by_abs_minv_minus_85'] = [dict(bin=[bins[i], bins[i + 1]], n=int((bidx == i).sum()), acc=float(ok0[bidx == i].mean()),
                                                    acc_after_phi=float(okp[bidx == i].mean()), changed_phi=float((Ls['phi'].argmax(1) != c0)[bidx == i].mean()))
                                               for i in range(len(bins) - 1)]
        if ep is None:
            d0 = M.read(L0)
            Lpp, _ = M.run(Xk['dil'][0], Xk['base'][1], Xk['base'][2]); Lpr, _ = M.run(Xk['base'][0], Xk['dil'][1], Xk['base'][2])
            s1, b1 = slope_boot(M.read(Lpp) - d0, D0['dil'], Wb); s2, b2 = slope_boot(M.read(Lpr) - d0, D0['dil'], Wb)
            row['route_dil'] = dict(per_particle=[s1] + ci(b1), pair=[s2] + ci(b2))
        res['models'][key] = row; del M
        log(f'{key}: ' + json.dumps({k: row[k] for k in ('R', 'acc_own', 'acc10', 'route_dil', 'changed') if k in row}) + f' ({time.time() - t1:.0f} s)')
        json.dump(res, open(OUT / 'p16_verify.json', 'w'), indent=1, default=float)
    res['perm'] = {}
    for x1, x2 in (('a', 'b'), ('a', 'released_kin'), ('a', 'released_full'), ('a_ep0', 'a'), ('a_ep5', 'a'), ('a', 'c')):
        res['perm'][f'{x1}_vs_{x2}'] = perm_test(dd[x1][0], dd[x1][1], dd[x2][0], dd[x2][1], rng)
        log(f'perm R_{x1} = R_{x2}: {res["perm"][f"{x1}_vs_{x2}"]}')
    json.dump(res, open(OUT / 'p16_verify.json', 'w'), indent=1, default=float)
    try:
        plot(res, OUT / 'p16_verify.png')
    except Exception as ex:
        log(f'plot failed {ex!r}')
    log('P16v DONE')


def plot(res, path):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    M = res['models']; fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    ax = axs[0]; ks = [k for k in ('a', 'b', 'c', 'released_full', 'released_kin') if k in M]
    for i, k in enumerate(ks):
        for j, rn in enumerate(('default', 'res_vs_QCD')):
            if rn not in M[k]['R']: continue
            r = M[k]['R'][rn]; x = i + (j - 0.5) * 0.35 if 'res_vs_QCD' in M[k]['R'] else i
            ax.bar(x, r[0], 0.33, color=['#3b6fb6', '#d08a2c'][j], label=(['P16 read-out', 'released: W or Z vs QCD read-out'][j] if (i == 3 or j == 0 and i == 0) else None))
            ax.errorbar(x, r[0], yerr=[[r[0] - r[1]], [r[2] - r[0]]], color='k', capsize=2, lw=0.8)
    ax.axhline(1, ls='--', color='grey', lw=0.8); ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(range(len(ks))); ax.set_xticklabels(['a: $m$ window', 'b: $m_A$ window', 'c: pair only', 'released full', 'released kin'], fontsize=8)
    ax.set_ylabel('R (phi shift / matched dilation), 95 % CI'); ax.set_title('Fresh rows (test _101, seed 16)', fontsize=9); ax.legend(fontsize=7)
    ax = axs[1]
    for arm, col in (('a', '#3b6fb6'), ('b', '#d08a2c')):
        eps = sorted([(int(k.split('ep')[1]), M[k]['R']['default']) for k in M if k.startswith(arm + '_ep')] + [(9, M[arm]['R']['default'])])
        ax.errorbar([e for e, _ in eps], [r[0] for _, r in eps], yerr=[[r[0] - r[1] for _, r in eps], [r[2] - r[0] for _, r in eps]], marker='o', color=col, capsize=2, label=f'arm {arm}')
    ax.set_xlabel('training epoch'); ax.set_ylabel('R'); ax.set_title('R of the planted arms during training', fontsize=9); ax.legend(fontsize=7); ax.set_ylim(bottom=0)
    fig.tight_layout(); fig.savefig(path, dpi=130)


if __name__ == '__main__':
    if '--plot' in sys.argv:
        f = Path(sys.argv[sys.argv.index('--plot') + 1]); plot(json.load(open(f)), f.with_suffix('.png'))
    else:
        main()
