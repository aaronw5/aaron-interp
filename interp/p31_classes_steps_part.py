"""P31 analysis — axis response R of the P31 arms (interp/p31_classes_steps_train.py: m10 = 10 classes on 5.0M jets, m10s = 10
classes on 1.5M jets, L = 3 classes for 50 epochs) next to P22 arms t / t2 (3 classes, 1.5M jets, 10 epochs, seeds 14 / 15), the
released ParT_kin / ParT_full and an untrained 3-output ParT_kin (init0).

Sample: fresh rows of JetClass test_20M/{ZJetsToNuNu,ZToQQ,WToQQ}_114.root, 5,000 random rows per file (numpy default_rng(3114 +
class index), i.e. 3114 / 3115 / 3116 for QCD / Z / W, sorted), N = 15,000, rebuilt unedited through part_core.build.
Read-out d = logit_Z - logit_W (3- and 10-output models alike).
Edits (as P22v): rigid shifts of every real particle about the stored axis phi + 0.06, phi - 0.06, eta + 0.06 rad (eta = jet_eta +
sign(jet_eta) deta; invariant mass unchanged up to < 1e-3); for each, a per-jet matched real dilation (deta, dphi) x k_j,
k_j = sqrt(M0M2(shift) / M0M2(base)), which moves the axis moment m_A by the same amount and changes the invariant mass;
null: the whole jet including the axis rotated by 0.06 rad in phi (inputs equal up to rounding).
Metrics per model: 3-class accuracy (argmax over QCD/Z/W outputs) and, for 10-output models, 10-class argmax accuracy on the
sample (and the P1-sample 10-class accuracy for the released models); R_s = per-jet least-squares slope through 0 of
Delta d(shift s) on Delta d(dilation s), 1,000-resample jet bootstrap; R as ratio of means; rotation-null slope; W -> Z = fraction
of true-W jets called Z (argmax over QCD/Z/W) before / after phi + 0.06, McNemar; route shares of the phi + 0.06 matched dilation
(P2 method: per-particle inputs edited with pair four-vectors kept / pair four-vectors edited with per-particle inputs kept;
per-jet slope on the full-edit change).
Tests: permutation (per-jet swap of the two models' (Delta d(shift), Delta d(dil)), 2,000 permutations), H0 R_arm = R_t, for arm in
{m10, m10s, L, t2} x direction in {phi+, phi-, eta+}, Holm over the family; McNemar on W -> Z (Holm).
--epochs: arm L (and m10 / m10s) at saved epoch checkpoints, R in the three directions.

  ./part_job.sh c-p31-run 'python -m interp.p31_classes_steps_part --models m10s,t,t2,released_kin,released_full,init0'
  -> $AI_RESULTS/part/p31_classes_steps_<tag>.json, .png
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp.part_core import build, kin, moments, inputs, OUT
from interp.e19_common import mcnemar, holm
from interp.p14_planted_train import masses
from interp.p14_planted_part import slope_boot, boot_weights, perm_test, ci
from interp.p22_planted_true_part import sample, Model, iW, iZ, iQ
from interp.p22_planted_true_train import build_model
from interp.p31_classes_steps_train import MOD as MOD31, ARMS as ARMS31

T0 = time.time()
DIRS = ('phi+', 'phi-', 'eta+')


def log(*a):
    print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


def load(name, ckpt='final.pt', smoke=False):
    """P31 arms from models/p31/<arm>/<ckpt>; P22 arms, released, init via the P22 Model class"""
    if name in ARMS31:
        M = Model('init0', False); K = ARMS31[name]['ncls']; M.K = K; M.name = name
        M.m = build_model(M.dev, K); f = MOD31 / (name + ('_smoke' if smoke else '')) / ckpt
        M.m.load_state_dict(M.t.load(f, map_location=M.dev)); M.path = str(f); M.m.eval()
        for p in M.m.parameters(): p.requires_grad_(False)
        M.cap = {}; M.m.fc.register_forward_hook(lambda mod, i, o: M.cap.__setitem__('h', i[0].detach()))
        M.wz = [iZ, iW] if K == 10 else [1, 2]
        return M
    M = Model(name, smoke)
    if ckpt != 'final.pt':
        from interp.p22_planted_true_train import MOD as MOD22
        f = MOD22 / name / ckpt; M.m.load_state_dict(M.t.load(f, map_location=M.dev)); M.path = str(f); M.m.eval()
    return M


def read_zw(M, L):
    return L[:, M.wz[0]] - L[:, M.wz[1]]          # logit_Z - logit_W


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--models', default='m10s,t,t2,released_kin,released_full,init0')
    ap.add_argument('--epochs', default='', help='e.g. L:0,4,9,14,19,24,29,34,39,44,49;m10s:0,4,9'); ap.add_argument('--per', type=int, default=5000)
    ap.add_argument('--smoke', action='store_true'); ap.add_argument('--tag', default=None); ap.add_argument('--nboot', type=int, default=1000); a = ap.parse_args()
    if a.smoke: a.per = 300
    tag = a.tag or ('smoke' if a.smoke else 'run'); fn = OUT / f'p31_classes_steps_{tag}.json'; OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3131)
    J0 = sample(a.per, seed=3114, idx=114); y10 = J0['label'].astype(int); N = len(y10); isW = y10 == iW
    pt, de, dp, ms = kin(J0); real = pt > 0
    Jb = build(J0, pt, de, dp, ms); Mb = moments(*kin(Jb)[:3])['M0M2']; okk = Mb > 0
    E = {'phi+': build(J0, pt, de, dp + 0.06 * real, ms), 'phi-': build(J0, pt, de, dp - 0.06 * real, ms),
         'eta+': build(J0, pt, de + 0.06 * real, dp, ms)}
    for s in DIRS:
        M2 = moments(*kin(E[s])[:3])['M0M2']; kj = np.where(okk, np.sqrt(M2 / np.where(okk, Mb, 1.0)), 1.0)[:, None]
        E['dil_' + s] = build(J0, pt, de * kj, dp * kj, ms)
    Jr = dict(Jb); Jr['jet_phi'] = (Jb['jet_phi'] + 0.06).astype(np.float32); c_, s_ = np.cos(0.06), np.sin(0.06)
    Jr['px'] = (c_ * Jb['px'] - s_ * Jb['py']).astype(np.float32); Jr['py'] = (s_ * Jb['px'] + c_ * Jb['py']).astype(np.float32); E['rot'] = Jr
    mi, mA = masses(Jb); edm = {k: masses(Je) for k, Je in E.items()}
    res = dict(N=N, per_class=a.per, files=[f'test_20M/{c}_114.root' for c in ('ZJetsToNuNu', 'ZToQQ', 'WToQQ')], seed='default_rng(3114 + class index)',
               readout='logit_Z - logit_W', nboot=a.nboot,
               edits={k: dict(dlogminv=float(np.mean(np.log(np.maximum(edm[k][0], 1)) - np.log(np.maximum(mi, 1)))),
                              dlogmA=float(np.mean(np.log(np.maximum(edm[k][1], 1e-6)) - np.log(np.maximum(mA, 1e-6))))) for k in E}, models={})
    log(f'{N} jets; ' + '; '.join(f'{k}: dlog m_inv {v["dlogminv"]:+.4f} dlog m_A {v["dlogmA"]:+.4f}' for k, v in res['edits'].items()))
    X = {kd: {k: inputs(Je, kd) for k, Je in [('base', Jb)] + list(E.items())} for kd in ('kin', 'full')}
    Wb = boot_weights(N, rng, a.nboot); dd = {}; plist, pkeys = [], []
    runs = [(n, 'final.pt') for n in a.models.split(',') if n]
    for spec in [s for s in a.epochs.split(';') if s]:
        n, es = spec.split(':'); runs += [(n, f'epoch-{int(e)}.pt') for e in es.split(',')]
    for name, ck in runs:
        key = name if ck == 'final.pt' else f'{name}@{ck[:-3]}'
        try:
            M = load(name, ck, a.smoke)
        except FileNotFoundError as ex:
            log(f'{key}: no checkpoint ({ex}); skipped'); continue
        Xk = X[M.kind]; t1 = time.time()
        L0, _ = M.run(*Xk['base']); d0 = read_zw(M, L0); row = dict(path=M.path, outputs=M.K)
        ok = M.call3(L0) == y10; Bi = rng.integers(0, N, (a.nboot, N)); row['acc3'] = [float(ok.mean())] + ci(ok[Bi].mean(1))
        row['confusion3_rows_true_QCD_Z_W'] = [[int(((y10 == i) & (M.call3(L0) == j)).sum()) for j in (iQ, iZ, iW)] for i in (iQ, iZ, iW)]
        if M.K == 10:
            row['acc10_on_sample'] = float((L0.argmax(1) == y10).mean())
            if name.startswith('released') and not a.smoke and ck == 'final.pt':
                from interp.part_core import sample as p1sample
                if 'P1' not in X:
                    JP = p1sample(1000, 3000); X['P1'] = {kd: inputs(build(JP, *kin(JP)), kd) for kd in ('kin', 'full')}; X['P1y'] = JP['label'].astype(int); del JP
                LP, _ = M.run(*X['P1'][M.kind]); row['acc10_P1_sample'] = float((LP.argmax(1) == X['P1y']).mean())
        D = {k: read_zw(M, M.run(*Xk[k])[0]) - d0 for k in E}
        for s in DIRS:
            R, Rb = slope_boot(D[s], D['dil_' + s], Wb); row[f'R_{s}'] = [R] + ci(Rb)
            row[f'Rmeans_{s}'] = [float(D[s].mean() / D['dil_' + s].mean())] + ci((Wb @ D[s]) / (Wb @ D['dil_' + s]))
        Rk, Rkb = slope_boot(D['rot'], D['dil_phi+'], Wb); row['R_rot_null'] = [Rk] + ci(Rkb)
        row['dd_mean'] = {k: float(v.mean()) for k, v in D.items()}; row['mean_abs_dd_dil_phi+'] = float(np.abs(D['dil_phi+']).mean())
        dd[key] = {s: (D[s], D['dil_' + s]) for s in DIRS}
        L1, _ = M.run(*Xk['phi+']); p0 = M.call3(L0)[isW] == iZ; p1 = M.call3(L1)[isW] == iZ
        mc = mcnemar(p0, p1); plist.append(mc['p']); pkeys.append(key); Bw = rng.integers(0, isW.sum(), (a.nboot, isW.sum()))
        row['WtoZ_phi+'] = dict(before=float(p0.mean()), after=float(p1.mean()), change=[float(p1.mean() - p0.mean())] + ci(p1[Bw].mean(1) - p0[Bw].mean(1)), mcnemar=mc)
        if ck == 'final.pt':
            k = 'dil_phi+'; Lpp, _ = M.run(Xk[k][0], Xk['base'][1], Xk['base'][2]); Lpr, _ = M.run(Xk['base'][0], Xk[k][1], Xk['base'][2])
            dpp, dpr = read_zw(M, Lpp) - d0, read_zw(M, Lpr) - d0
            s1, b1 = slope_boot(dpp, D[k], Wb); s2, b2 = slope_boot(dpr, D[k], Wb)
            row['routes_dil_phi+'] = dict(per_particle=[s1] + ci(b1), pair=[s2] + ci(b2),
                                          additivity_r2=float(1 - ((dpp + dpr - D[k]) ** 2).sum() / ((D[k] - D[k].mean()) ** 2).sum()))
        res['models'][key] = row; del M
        log(f'{key}: acc3 {row["acc3"][0]:.4f}' + (f' acc10(sample) {row["acc10_on_sample"]:.4f}' if 'acc10_on_sample' in row else '')
            + (f' acc10(P1) {row["acc10_P1_sample"]:.4f}' if 'acc10_P1_sample' in row else '') + '; '
            + '; '.join(f'R {s} {row["R_" + s][0]:+.3f} [{row["R_" + s][1]:+.3f}, {row["R_" + s][2]:+.3f}] (means {row["Rmeans_" + s][0]:+.3f})' for s in DIRS)
            + f'; rot null {Rk:+.4f}; |dd dil| {row["mean_abs_dd_dil_phi+"]:.3f}; W->Z {row["WtoZ_phi+"]["before"]:.3f} -> {row["WtoZ_phi+"]["after"]:.3f}'
            + (f'; route pp {row["routes_dil_phi+"]["per_particle"][0]:.3f} pair {row["routes_dil_phi+"]["pair"][0]:.3f}' if 'routes_dil_phi+' in row else '')
            + f' ({time.time() - t1:.0f} s)')
        json.dump(res, open(fn, 'w'), indent=1, default=float)
    for k, p in zip(pkeys, holm(plist)): res['models'][k]['WtoZ_phi+']['mcnemar']['p_holm'] = p
    res['perm'] = {}; pp = []
    for x1 in ('m10', 'm10s', 'L', 't2'):
        for s in DIRS:
            if x1 in dd and 't' in dd:
                kk = f'{x1}_vs_t_{s}'; res['perm'][kk] = perm_test(dd[x1][s][0], dd[x1][s][1], dd['t'][s][0], dd['t'][s][1], rng); pp.append(kk)
    for s in DIRS:
        if 'm10' in dd and 'm10s' in dd:
            kk = f'm10_vs_m10s_{s}'; res['perm'][kk] = perm_test(dd['m10'][s][0], dd['m10'][s][1], dd['m10s'][s][0], dd['m10s'][s][1], rng); pp.append(kk)
    for k, p in zip(pp, holm([res['perm'][k]['p'] for k in pp])):
        res['perm'][k]['p_holm'] = p; log(f'permutation H0 {k}: diff {res["perm"][k]["diff"]:+.3f}, p {res["perm"][k]["p"]:.4f}, Holm {p:.4f}')
    json.dump(res, open(fn, 'w'), indent=1, default=float); log(f'wrote {fn}')
    try:
        plot(res, fn.with_suffix('.png'))
    except Exception as ex:
        log(f'plot failed: {ex!r}')
    log('P31 analysis DONE')


LABEL = dict(t='3 classes\n1.5M jets\n10 ep (P22 t)', t2='3 classes\nseed 15\n(P22 t2)', m10s='10 classes\n1.5M jets\n10 ep',
             m10='10 classes\n5.0M jets\n10 ep', L='3 classes\n1.5M jets\n50 ep', released_kin='released\nParT_kin', released_full='released\nParT_full',
             init0='untrained')


def plot(res, path):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    names = [n for n in LABEL if n in res['models'] and n != 'init0']
    eps = sorted((int(k.split('@epoch-')[1]), k) for k in res['models'] if k.startswith('L@'))
    fig, axs = plt.subplots(1, 2 if eps else 1, figsize=(11 if eps else 8, 4.3), squeeze=False, gridspec_kw=dict(width_ratios=[1.6, 1] if eps else [1]))
    ax = axs[0, 0]
    for j, (s, col) in enumerate((('phi+', '#3b6fb6'), ('phi-', '#7fa7d9'), ('eta+', '#d08a2c'))):
        for i, n in enumerate(names):
            r = res['models'][n]['R_' + s]; x = i + (j - 1) * 0.25
            ax.bar(x, r[0], 0.24, color=col, label={'phi+': 'φ + 0.06 rad', 'phi-': 'φ − 0.06 rad', 'eta+': 'η + 0.06'}[s] if i == 0 else None)
            ax.errorbar(x, r[0], yerr=[[r[0] - r[1]], [r[2] - r[0]]], color='k', capsize=2, lw=0.8)
    ax.axhline(0.5, color='#555', lw=0.8, ls='--'); ax.text(len(names) - 0.5, 0.51, 'pre-registered 0.5', ha='right', va='bottom', fontsize=7, color='#555')
    ax.axhline(0, color='k', lw=0.8); ax.set_xticks(range(len(names))); ax.set_xticklabels([LABEL[n] for n in names], fontsize=7.5)
    ax.set_ylim(-0.05, 1.05); ax.set_ylabel('R = read-out change under rigid shift\n/ change under matched real mass change')
    ax.legend(fontsize=8, title='shift about stored jet axis', title_fontsize=8, loc='upper left')
    ax.set_title('ParT_kin axis response by training: class count and length', fontsize=9.5)
    if eps:
        bx = axs[0, 1]
        for s, col in (('phi+', '#3b6fb6'), ('phi-', '#7fa7d9'), ('eta+', '#d08a2c')):
            rr = [res['models'][k]['R_' + s] for _, k in eps]
            bx.errorbar([e for e, _ in eps], [r[0] for r in rr], yerr=[[r[0] - r[1] for r in rr], [r[2] - r[0] for r in rr]], marker='o', ms=3, color=col, capsize=2, label=s)
        bx.axhline(0.5, color='#555', lw=0.8, ls='--'); bx.set_ylim(-0.15, 1.05); bx.set_xlabel('epoch of the 50-epoch 3-class run (1.47M jets each)')
        bx.axvspan(34.5, 49.5, color='#ddd', lw=0, zorder=0); bx.text(35, 1.0, 'lr decay', fontsize=7, color='#555')
        bx.set_ylabel('R'); bx.legend(fontsize=7); bx.set_title('R over training, arm L (lr flat until epoch 34)', fontsize=9.5)
    fig.text(0.5, -0.02, f'N = {res["N"]} held-out W/Z/QCD jets (test_20M *_114); per-jet slope through 0, 95 % bootstrap over jets ({res["nboot"]}).',
             ha='center', fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches='tight')


if __name__ == '__main__':
    if '--plot' in sys.argv:
        f = Path(sys.argv[sys.argv.index('--plot') + 1]); plot(json.load(open(f)), f.with_suffix('.png'))
    else:
        main()
