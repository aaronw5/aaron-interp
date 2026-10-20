"""P22 — analysis of the ParT_kin arms trained with the P14 code path on true labels (interp/p22_planted_true_train.py): does
training on the true W/Z/QCD labels at the P14 scale produce the released models' axis response, and does random-offset
augmentation remove it?

Sample: fresh rows of JetClass test_20M/{ZJetsToNuNu,ZToQQ,WToQQ}_104.root, N = 3,000 random rows per class (numpy
default_rng(2204) per file, sorted), rebuilt unedited with part_core.build (all edits go through the same build).
Models: t, t2, u (3 outputs: QCD, Z, W), a2 (2 outputs, planted m_inv window) from models/p22/<arm>/final.pt; P14 arm a
(models/p14/a/final.pt); released ParT_full / ParT_kin (10 outputs); untrained controls init0..init3 (3-output ParT_kin,
torch.manual_seed(100 + k), never trained).
Read-out d: 3-output and released: logit_W - logit_Z; 2-output planted: logit_in - logit_out.
Edits (as P14 / P16):
  phi   every real particle's dphi + 0.06 rad about the stored axis (m_inv unchanged; axis moment m_A = sqrt(M0 M2) up)
  dil   per-jet matched real dilation (deta, dphi) x k_j, k_j = sqrt(M0M2(phi) / M0M2(base)): same Delta log m_A as phi
  k1.1  angles x 1.1 (P2 reference real change)
  rot   whole jet incl. the axis rotated 0.06 rad in phi (null: inputs equal up to rounding)
Metrics per model:
  accuracy (3-class arms: 3x3 confusion rows true QCD/Z/W; released: 10-class and W/Z/QCD-restricted; planted: own label);
  R = per-jet least-squares slope of Delta d(phi) on Delta d(dil) (through 0), 95 % bootstrap over jets (2,000); rot null;
  W -> Z = fraction of true-W jets called Z before / after phi (McNemar on the paired Z-call indicator; Holm over all McNemar);
  route shares of dil and k1.1 (P2 method: per-particle inputs only edited / pair four-vectors only edited; per-jet slope on
  the full-edit change, bootstrap CI);
  DAS (P8 method, interp/p14_planted_part.das): one direction at the output of particle block 2 and 8, jet-mean token value
  swapped base <- source; trained so that the intervened call equals the source jet's window on V in {m_inv, m_A}; planted
  arms: the P14 windows; 3-output and released models: W/Z jets only, call = argmax(logit_Z, logit_W), target W-like =
  1[m_inv < 85.8 GeV] (midpoint of m_W, m_Z) or 1[m_A < c_A] with c_A set so both targets have the same W-like rate;
  interchange-intervention accuracy (IIA) overall and on pairs whose two targets differ; 4 random directions as control.
Tests: permutation (per-jet swap of the two models' (Delta d(phi), Delta d(dil))) H0 R_x = R_y for (t, a), (u, t), (t, t2),
  (a, a2), (t, released_kin), (u, init0); McNemar on 3-class correctness t vs u; Holm over the permutation family.
Predictions (pre-registered in running/C.md): thesis R_t in 0.6-0.8, R_u < 0.3 with accuracy within 1 pt of t; alt 1 (scale /
  10 classes needed) R_t ~ R_a ~ 0.2; alt 2 (augmentation leaves it) R_u ~ R_t.

  ./part_job.sh c-p22-run 'python -m interp.p22_planted_true_part'              -> results/part/p22_planted_true.json, .png
  --smoke: 300 jets per class, smoke checkpoints, 20 DAS steps
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp.part_core import read_root, build, kin, moments, cat, inputs, OUT, CL, ParTNetwork
from interp.p1v_verify_part import Ridge, split
from interp.e19_common import mcnemar, holm, wilson
from interp.p14_planted_train import masses, CLS3, JC, SIDE, MOD as MOD14, M0_, W0_
from interp.p14_planted_part import slope_boot, boot_weights, perm_test, das, ci
from interp.p22_planted_true_train import MOD as MOD22, build_model

T0 = time.time()
iW, iZ, iQ = CL.index('Wqq'), CL.index('Zqq'), CL.index('QCD')
MWZ = 0.5 * (80.4 + 91.2)


def log(*a):
    print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


def sample(per, seed=2204, idx=104):
    Js = []
    for ci_, c in enumerate(CLS3):
        f = JC / 'test_20M' / f'{c}_{idx}.root'; J = read_root(f); n = len(J['label'])
        r = np.sort(np.random.default_rng(seed + ci_).choice(n, per, replace=False))
        Js.append({k: (v[r] if np.ndim(v) else v) for k, v in J.items()}); del J; log(f'sample {f.name}: {per} of {n} rows')
    return cat(Js)


class Model:
    """run(x, v, m) on kin or full inputs -> logits (N, K), class token (N, 128); read(L) -> W-vs-Z or in-vs-out logit difference"""

    def __init__(self, name, smoke):
        import torch
        self.t, self.name = torch, name; self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'; self.path = None
        if name.startswith('released_'):
            self.kind = name.split('_')[1]; self.K = 10; self.m = ParTNetwork(self.kind).model.to(self.dev)
        elif name.startswith('init'):
            self.kind, self.K = 'kin', 3; torch.manual_seed(100 + int(name[4:])); self.m = build_model(self.dev, 3)
        else:
            self.kind = 'kin'; self.K = 2 if name in ('a', 'a2') else 3; self.m = build_model(self.dev, self.K)
            d = (MOD14 if name == 'a' else MOD22) / (name + ('_smoke' if smoke else '')); f = d / 'final.pt'
            self.m.load_state_dict(torch.load(f, map_location=self.dev)); self.path = str(f)
        self.m.eval()
        for p in self.m.parameters(): p.requires_grad_(False)
        self.cap = {}; self.m.fc.register_forward_hook(lambda mod, i, o: self.cap.__setitem__('h', i[0].detach()))
        self.wz = [iZ, iW] if self.K == 10 else ([1, 2] if self.K == 3 else None)

    def x(self, X):
        return X

    def read(self, L):
        return L[:, self.wz[1]] - L[:, self.wz[0]] if self.wz else L[:, 1] - L[:, 0]

    def call3(self, L):
        """call among QCD / Z / W as a 10-class index (iQ, iZ, iW)"""
        if self.K == 10: return np.array([iQ, iZ, iW])[L[:, [iQ, iZ, iW]].argmax(1)]
        return np.array([iQ, iZ, iW])[L.argmax(1)]

    def run(self, x, v, m, batch=256):
        t = self.t; N = len(x); order = np.argsort(m[:, 0].sum(1), kind='stable'); L = None; H = np.zeros((N, 128), np.float32)
        T = lambda a: t.from_numpy(np.ascontiguousarray(a)).to(self.dev)
        with t.no_grad():
            for i in range(0, N, batch):
                idx = order[i:i + batch]; P = int(m[idx, 0].sum(1).max()); out = self.m(T(x[idx, :, :P]), T(v[idx, :, :P]), T(m[idx, :, :P]))
                if L is None: L = np.zeros((N, out.shape[1]), np.float32)
                L[idx] = out.float().cpu().numpy(); H[idx] = self.cap['h'].float().cpu().numpy()
        return L.astype(np.float64), H.astype(np.float64)


class WZView:
    """DAS view of a 3- or 10-output model: forward returns (logit_Z, logit_W) so that call 1 = W"""

    def __init__(self, M):
        import torch
        class Net(torch.nn.Module):
            def __init__(s, inner, cols):
                super().__init__(); s.inner = inner; s.blocks = inner.blocks; s.cols = cols

            def forward(s, *a):
                return s.inner(*a)[:, s.cols]
        self.t, self.dev, self.name = M.t, M.dev, M.name; self.m = Net(M.m, M.wz)

    def x(self, X):
        return X


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--models', default='t,t2,u,a2,a,released_kin,released_full,init0,init1,init2,init3')
    ap.add_argument('--per', type=int, default=3000); ap.add_argument('--smoke', action='store_true'); ap.add_argument('--das-steps', type=int, default=400)
    ap.add_argument('--tag', default=None); a = ap.parse_args()
    if a.smoke: a.per, a.das_steps = 300, 20
    tag = a.tag or ('smoke' if a.smoke else 'run'); fn = OUT / f'p22_planted_true_{tag}.json'
    rng = np.random.default_rng(2222); OUT.mkdir(parents=True, exist_ok=True)
    J0 = sample(a.per); y10 = J0['label'].astype(int); N = len(y10); isW = y10 == iW
    pt, de, dp, ms = kin(J0); real = pt > 0
    Jb = build(J0, pt, de, dp, ms); Mb = moments(*kin(Jb)[:3])
    E = {'phi': build(J0, pt, de, dp + 0.06 * real, ms)}
    M2p = moments(*kin(E['phi'])[:3])['M0M2']; okk = Mb['M0M2'] > 0
    kj = np.where(okk, np.sqrt(M2p / np.where(okk, Mb['M0M2'], 1.0)), 1.0)[:, None]
    E['dil'] = build(J0, pt, de * kj, dp * kj, ms); E['k1.1'] = build(J0, pt, de * 1.1, dp * 1.1, ms)
    Jr = dict(Jb); Jr['jet_phi'] = (Jb['jet_phi'] + 0.06).astype(np.float32); c_, s_ = np.cos(0.06), np.sin(0.06)
    Jr['px'] = (c_ * Jb['px'] - s_ * Jb['py']).astype(np.float32); Jr['py'] = (s_ * Jb['px'] + c_ * Jb['py']).astype(np.float32); E['rot'] = Jr
    mi, mA = masses(Jb); side = np.load(SIDE / ('labels_smoke.npz' if a.smoke else 'labels.npz')); c, w = float(side['c']), float(side['w'])
    lab = dict(a=(np.abs(mi - M0_) < W0_).astype(int)); lab['a2'] = lab['a']; labb = (np.abs(mA - c) < w).astype(int)
    wzj = np.isin(y10, [iW, iZ]); tA = (mi < MWZ).astype(int); cA = float(np.quantile(mA[wzj], tA[wzj].mean())); tB = (mA < cA).astype(int)
    edm = {k: masses(Je) for k, Je in E.items()}
    res = dict(N=N, per_class=a.per, files=[f'test_20M/{c_}_104.root' for c_ in CLS3], seed=2204, c=c, w=w, wz_threshold=dict(m_inv=MWZ, m_A=cA),
               edits={k: dict(dlogminv=float(np.mean(np.log(np.maximum(edm[k][0], 1)) - np.log(np.maximum(mi, 1)))),
                              dlogmA=float(np.mean(np.log(np.maximum(edm[k][1], 1e-6)) - np.log(np.maximum(mA, 1e-6))))) for k in E}, models={})
    log(f'{N} jets; edits: ' + '; '.join(f'{k}: dlog m_inv {v["dlogminv"]:+.4f} dlog m_A {v["dlogmA"]:+.4f}' for k, v in res['edits'].items())
        + f'; W/Z DAS thresholds m_inv {MWZ:.1f}, m_A {cA:.2f} GeV; targets differ on {(tA != tB)[wzj].mean():.3f} of W/Z jets')
    X = {kd: {k: inputs(Je, kd) for k, Je in [('base', Jb)] + list(E.items())} for kd in ('kin', 'full')}
    fit, val, tr, te = split(N); Wb = boot_weights(N, rng); dd, correct = {}, {}; plist, pkeys = [], []
    for name in a.models.split(','):
        try:
            M = Model(name, a.smoke)
        except FileNotFoundError as ex:
            log(f'{name}: no checkpoint ({ex}); skipped'); continue
        Xk = X[M.kind]; t1 = time.time()
        L0, H0 = M.run(*Xk['base']); d0 = M.read(L0); row = dict(path=M.path, outputs=M.K)
        if M.K == 2:
            c0 = L0.argmax(1); own = lab[name]; row['acc_own_label'] = [float((c0 == own).mean())] + list(wilson(int((c0 == own).sum()), N))
            row['acc_vs_mA_window'] = float((c0 == labb).mean())
            row['confusion_label_vs_call'] = [[int(((own == i) & (c0 == j)).sum()) for j in (0, 1)] for i in (0, 1)]
            pos = lambda L: L.argmax(1) == 1
        else:
            c3 = M.call3(L0); ok = c3 == y10; correct[name] = ok
            row['acc3'] = [float(ok.mean())] + list(wilson(int(ok.sum()), N))
            row['confusion3_rows_true_QCD_Z_W_cols_call'] = [[int(((y10 == i) & (c3 == j)).sum()) for j in (iQ, iZ, iW)] for i in (iQ, iZ, iW)]
            if M.K == 10:
                c10 = L0.argmax(1); row['acc10'] = float((c10 == y10).mean())
                if not a.smoke:
                    from interp.part_core import sample as p1sample
                    if 'P1' not in X:
                        JP = p1sample(1000, 3000); X['P1'] = {kd: inputs(build(JP, *kin(JP)), kd) for kd in ('kin', 'full')}; X['P1y'] = JP['label'].astype(int); del JP
                    LP, _ = M.run(*X['P1'][M.kind]); row['acc10_P1_sample'] = float((LP.argmax(1) == X['P1y']).mean())
                    log(f'{name}: P1-sample 10-class accuracy {row["acc10_P1_sample"]:.4f} (P1: full 0.859, kin 0.745)')
            pos = lambda L, M=M: M.call3(L) == iZ          # W -> Z: Z call on true-W jets
        rd = Ridge(H0, np.log(np.maximum(mi, 1.0)), fit, val, tr)
        Dd, Dp, Ls = {}, {}, {}
        for k in E:
            L, H = M.run(*Xk[k]); Dd[k] = M.read(L) - d0; Dp[k] = rd(H[te]) - rd(H0[te]); Ls[k] = L
            sel = isW if M.K != 2 else np.ones(N, bool)
            p0, p1 = pos(L0)[sel], pos(L)[sel]; mc = mcnemar(p0, p1); plist.append(mc['p']); pkeys.append((name, k))
            Bi = rng.integers(0, sel.sum(), (2000, sel.sum()))
            row.setdefault('edits', {})[k] = dict(changed=float((L.argmax(1) != L0.argmax(1)).mean()), pos_before=float(p0.mean()), pos_after=float(p1.mean()),
                                                  pos_change=[float(p1.mean() - p0.mean())] + ci(p1[Bi].mean(1) - p0[Bi].mean(1)), mcnemar_pos=mc,
                                                  dd_mean=float(Dd[k].mean()), dprobe_mean=float(Dp[k].mean()))
            if M.K == 2: row['edits'][k]['acc_own_label_after'] = float((L.argmax(1) == (np.abs(edm[k][0] - M0_) < W0_)).mean())
            else: row['edits'][k]['acc3_after'] = float((M.call3(L) == y10).mean())
        R, Rb = slope_boot(Dd['phi'], Dd['dil'], Wb); row['R'] = [R] + ci(Rb)
        Rk, Rkb = slope_boot(Dd['rot'], Dd['dil'], Wb); row['R_rot_null'] = [Rk] + ci(Rkb)
        Wt = boot_weights(len(te), rng, 1000); row['R_probe'] = [float(Dp['phi'].mean() / Dp['dil'].mean())] + ci((Wt @ Dp['phi']) / (Wt @ Dp['dil']))
        if M.K != 2:
            s, sb = slope_boot(Dd['phi'][isW], Dd['dil'][isW], boot_weights(int(isW.sum()), rng, 1000)); row['R_Wjets'] = [s] + ci(sb)
        row['mean_abs_dd_dil'] = float(np.abs(Dd['dil']).mean()); dd[name] = (Dd['phi'], Dd['dil'])
        row['routes'] = {}
        for k in ('dil', 'k1.1'):
            Lpp, _ = M.run(Xk[k][0], Xk['base'][1], Xk['base'][2]); Lpr, _ = M.run(Xk['base'][0], Xk[k][1], Xk['base'][2])
            dpp, dpr = M.read(Lpp) - d0, M.read(Lpr) - d0; s1, b1 = slope_boot(dpp, Dd[k], Wb); s2, b2 = slope_boot(dpr, Dd[k], Wb)
            row['routes'][k] = dict(per_particle=[s1] + ci(b1), pair=[s2] + ci(b2),
                                    additivity_r2=float(1 - ((dpp + dpr - Dd[k]) ** 2).sum() / ((Dd[k] - Dd[k].mean()) ** 2).sum()))
        e = row['edits']['phi']
        log(f'{name}: ' + (f'acc own label {row["acc_own_label"][0]:.4f}' if M.K == 2 else f'acc3 {row["acc3"][0]:.4f} confusion {row["confusion3_rows_true_QCD_Z_W_cols_call"]}')
            + f'; R {R:+.3f} [{row["R"][1]:+.3f}, {row["R"][2]:+.3f}] (rot null {Rk:+.4f}); R_probe {row["R_probe"][0]:+.3f}; '
            + (f'positive call {e["pos_before"]:.3f} -> {e["pos_after"]:.3f}' if M.K == 2 else f'W->Z {e["pos_before"]:.3f} -> {e["pos_after"]:.3f}')
            + '; ' + '; '.join(f'route {k}: pp {v["per_particle"][0]:.3f} pair {v["pair"][0]:.3f}' for k, v in row['routes'].items()) + f' ({time.time() - t1:.0f} s)')
        if not name.startswith('init') and name != 'released_full':
            if M.K == 2:
                row['das'] = das(M, Xk['base'][0], Xk['base'][1], Xk['base'][2], dict(minv=lab['a'], mA=labb), tr, te, rng, sites=(1, 7), steps=a.das_steps)
            else:
                row['das'] = das(WZView(M), Xk['base'][0], Xk['base'][1], Xk['base'][2], dict(minv=tA, mA=tB), tr[wzj[tr]], te[wzj[te]], rng, sites=(1, 7), steps=a.das_steps)
        res['models'][name] = row; del M
        json.dump(res, open(fn, 'w'), indent=1, default=float)
    for (nm, k), p in zip(pkeys, holm(plist)): res['models'][nm]['edits'][k]['mcnemar_pos']['p_holm'] = p
    res['perm'] = {}; pp = []
    for x1, x2 in (('t', 'a'), ('u', 't'), ('t', 't2'), ('a', 'a2'), ('t', 'released_kin'), ('u', 'init0')):
        if x1 in dd and x2 in dd:
            res['perm'][f'{x1}_vs_{x2}'] = perm_test(dd[x1][0], dd[x1][1], dd[x2][0], dd[x2][1], rng); pp.append(f'{x1}_vs_{x2}')
    for k, p in zip(pp, holm([res['perm'][k]['p'] for k in pp])):
        res['perm'][k]['p_holm'] = p; log(f'permutation H0 {k}: diff {res["perm"][k]["diff"]:+.3f}, p {res["perm"][k]["p"]:.4f}, Holm {p:.4f}')
    if 't' in correct and 'u' in correct:
        okt, oku = correct['t'], correct['u']; Bi = rng.integers(0, N, (2000, N))
        res['acc_t_vs_u'] = dict(diff=float(oku.mean() - okt.mean()), ci=ci(oku[Bi].mean(1) - okt[Bi].mean(1)), mcnemar=mcnemar(okt, oku))
        log(f'accuracy u - t {res["acc_t_vs_u"]["diff"]:+.4f} {res["acc_t_vs_u"]["ci"]}; McNemar p {res["acc_t_vs_u"]["mcnemar"]["p"]:.3g}')
    inits = [res['models'][k]['R'][0] for k in res['models'] if k.startswith('init')]
    if inits: res['R_init_range'] = [min(inits), max(inits)]
    json.dump(res, open(fn, 'w'), indent=1, default=float); log(f'wrote {fn}')
    try:
        plot(res, fn.with_suffix('.png'))
    except Exception as ex:
        log(f'plot failed: {ex!r}')
    log('P22 analysis DONE')


LABEL = dict(t='true labels\nseed 14', t2='true labels\nseed 15', u='true labels\n+ random\noffsets', a='mass-window\nlabel seed 14',
             a2='mass-window\nlabel seed 15', released_kin='released\nkin', released_full='released\nfull')


def plot(res, path):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    names = [n for n in LABEL if n in res['models']]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for i, n in enumerate(names):
        r = res['models'][n]['R']; col = '#999999' if n.startswith('released') else ('#3b6fb6' if n in ('t', 't2', 'u') else '#d08a2c')
        ax.bar(i, r[0], 0.6, color=col); ax.errorbar(i, r[0], yerr=[[r[0] - r[1]], [r[2] - r[0]]], color='k', capsize=3, lw=1)
        ax.text(i, max(r[2], 0) + 0.03, f'{r[0]:.2f}', ha='center', fontsize=8)
    if 'R_init_range' in res:
        ax.axhspan(*res['R_init_range'], color='#cccccc', alpha=0.4, label='untrained networks (4 initialisations)')
    ax.axhline(0, color='k', lw=0.8); ax.axhline(1, color='k', lw=0.5, ls=':')
    ax.set_xticks(range(len(names))); ax.set_xticklabels([LABEL[n] for n in names], fontsize=8)
    ax.set_ylabel('R = response to 0.06 rad shift (invariant mass fixed)\n/ response to matched real mass change')
    ax.set_title('Axis response of ParT_kin by training label (1.5M W/Z/QCD jets, 10 epochs)', fontsize=10)
    if 'R_init_range' in res: ax.legend(fontsize=8, loc='upper left')
    fig.text(0.5, -0.02, f'N = {res["N"]} held-out W/Z/QCD jets (test_20M *_104, seed 2204); bars: per-jet slope, bootstrap 95 % CI.', ha='center', fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches='tight')


if __name__ == '__main__':
    if '--plot' in sys.argv:
        f = Path(sys.argv[sys.argv.index('--plot') + 1]); plot(json.load(open(f)), f.with_suffix('.png'))
    else:
        main()
