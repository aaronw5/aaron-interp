"""P14 — planted ground truth for the ParT methods: analysis of the planted-label ParTs (interp/p14_planted_train.py) and of the
released ParT_full / ParT_kin restricted to W/Z/QCD jets (control).

Sample: N = 20,000 random rows (numpy default_rng(14)) of the union of JetClass test_20M/{ZJetsToNuNu,ZToQQ,WToQQ}_100.root (the
held-out sample of the training script; its side file data/p14/labels.npz gives the arm-b window c, w). Baseline = the jets rebuilt
unedited with part_core.build (all edits go through the same build). Labels: a = 1[|m_inv - 85| < 10 GeV], b = 1[|m_A - c| < w]
with m_A = sqrt(M0 M2) about the stored axis, recomputed on the rebuilt jets.
Models: arms a, b, c, s (models/p14/<arm>/final.pt, or epoch-<e>.pt with --epoch), released_full, released_kin.
Read-out d: arms: logit_1 - logit_0 (positive = inside the planted window); released: logit_W - logit_Z (P10 convention).

Edits (part_core.build):
  phi   every real particle's dphi + 0.06 rad about the stored axis (rigid rotation; m_inv unchanged; m_A up)
  dil   per-jet matched real dilation: (deta, dphi) x k_j with k_j = sqrt(M0M2(phi) / M0M2(base)), so Delta log m_A is the same as
        under phi for every jet, and m_inv changes by about the same log amount (real mass change)
  k1.1  angles x 1.1 (P2's reference real change)
  rot   whole jet incl. the axis rotated 0.06 rad in phi (inputs equal up to float rounding; null)
Test 1 (P1 translation, calibrated): R = per-jet least-squares slope of Delta d(phi) on Delta d(dil) (through the origin), 95 %
  bootstrap over jets. H_inv (network reads m_inv): R = 0; H_axis (reads m_A): R = 1. Planted predictions: a ~ 0 if ParT can learn
  invariance from the label, b ~ 1, c = 0 by construction. Probe version: ridge read-out of log m_inv from the 128-dim class token
  (fit on the unedited train half, P1 procedure), R_probe = mean Delta log m-hat(phi) / mean Delta log m-hat(dil) on the test half.
  Permutation test H0: R_a = R_b (and R_a = R_s, R_a = R_c): per-jet swap of the two models' (Delta d(phi), Delta d(dil)) pairs.
  Calls changed (Wilson CI), McNemar on the positive-call indicator before / after (Holm over all McNemar tests).
Test 2 (P2 route split) for dil and k1.1: variants per-particle only (x_edit, v_base) and pair only (x_base, v_edit); share =
  per-jet slope of Delta d(variant) on Delta d(full edit), bootstrap CI. Planted: c pair share ~ 1; b per-particle ~ 1.
Test 3 (P8 DAS, arms only): 1-dim jet-mean direction u at the output of particle block 2 and block 8: the mean over real particles
  of (token . u) of the base jet is set to the source jet's value; u trained (Adam, cross-entropy) so that the intervened call equals
  the planted window of the SOURCE jet's variable V, V in {m_inv, m_A}. Interchange-intervention accuracy (IIA) on held-out pairs,
  on all pairs and on the discriminating pairs (window_minv(src) != window_mA(src)); McNemar of IIA(m_inv) vs IIA(m_A) on the
  discriminating pairs; control: 4 random unit directions.
Test 4 (P9 EAP-IG circuit, interp/p14_eap_part.py): arms and released_kin (W vs Z jets); 1,500 discovery jets (train half), 3,000
  evaluation jets (test half); edges / nodes at F = 0.8, literal node circuit accuracy, confusion and 20 random circuits.

  ./part_job.sh c-p14-run 'python -m interp.p14_planted_part'                 -> results/part/p14_planted.json
  ./part_job.sh c-p14-ctl 'python -m interp.p14_planted_part --models released_full,released_kin --tag released'
  --smoke: 3,000 jets, smoke checkpoints (models/p14/<arm>_smoke), short DAS
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp.part_core import read_root, build, kin, moments, jmass, cat, inputs, OUT, CL, ParTNetwork
from interp.p1v_verify_part import Ridge, split
from interp.e19_common import mcnemar, holm, wilson
from interp.p14_planted_train import masses, feed, build_model, CLS3, JC, SIDE, MOD, M0_, W0_

T0 = time.time()
iW, iZ, iQ = CL.index('Wqq'), CL.index('Zqq'), CL.index('QCD')
NB = 2000


def log(*a):
    print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


def ci(b):
    return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]


def sample(n, seed=14):
    import uproot
    fs = [JC / 'test_20M' / f'{c}_100.root' for c in CLS3]; ne = [uproot.open(f)['tree'].num_entries for f in fs]
    rng = np.random.default_rng(seed); g = np.sort(rng.choice(sum(ne), n, replace=False)); off = np.cumsum([0] + ne); Js = []
    for ci_, f in enumerate(fs):
        r = g[(g >= off[ci_]) & (g < off[ci_ + 1])] - off[ci_]; J = read_root(f)
        Js.append({k: (v[r] if np.ndim(v) else v) for k, v in J.items()}); del J; log(f'sample {f.name}: {len(r)} rows')
    return cat(Js)


class Model:
    """uniform wrapper: run(x, v, m) on precomputed (unfed) inputs -> logits (N, K), class token h (N, 128)"""

    def __init__(self, name, smoke, epoch):
        import torch
        self.t, self.name = torch, name; self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        if name.startswith('released_'):
            self.kind = name.split('_')[1]; self.arm = None; pn = ParTNetwork(self.kind); self.m = pn.model.to(self.dev)
        else:
            self.kind, self.arm = 'kin', name; self.m = build_model(self.dev)
            d = MOD / (name + ('_smoke' if smoke else '')); f = d / ('final.pt' if epoch is None else f'epoch-{epoch}.pt')
            self.m.load_state_dict(torch.load(f, map_location=self.dev)); self.path = str(f)
        self.m.eval()
        for p in self.m.parameters(): p.requires_grad_(False)
        self.cap = {}; self.m.fc.register_forward_hook(lambda mod, i, o: self.cap.__setitem__('h', i[0].detach()))

    def x(self, X):
        return X if self.arm is None else feed(X, self.arm)

    def read(self, L):
        return L[:, 1] - L[:, 0] if self.arm else L[:, iW] - L[:, iZ]

    def run(self, x, v, m, batch=256):
        t = self.t; N = len(x); order = np.argsort(m[:, 0].sum(1), kind='stable'); x = self.x(x); L = None; H = np.zeros((N, 128), np.float32)
        T = lambda a: t.from_numpy(np.ascontiguousarray(a)).to(self.dev)
        with t.no_grad():
            for i in range(0, N, batch):
                idx = order[i:i + batch]; P = int(m[idx, 0].sum(1).max()); out = self.m(T(x[idx, :, :P]), T(v[idx, :, :P]), T(m[idx, :, :P]))
                if L is None: L = np.zeros((N, out.shape[1]), np.float32)
                L[idx] = out.float().cpu().numpy(); H[idx] = self.cap['h'].float().cpu().numpy()
        return L.astype(np.float64), H.astype(np.float64)


def slope_boot(y, x, W):
    s = float((y * x).sum() / (x * x).sum()); b = (W @ (y * x)) / (W @ (x * x)); return s, b


def boot_weights(n, rng, B=NB):
    """multinomial bootstrap counts (B, n) as float32 (resampling jets with replacement)"""
    W = np.zeros((B, n), np.float32)
    for b in range(B): W[b] = np.bincount(rng.integers(0, n, n), minlength=n)
    return W


def perm_test(y1, x1, y2, x2, rng, n=2000):
    """H0: R_1 = R_2 (per-jet slope of y on x, same jets): swap the two models' (y, x) per jet with probability 1/2"""
    f = lambda ya, xa, yb, xb: (ya * xa).sum() / (xa * xa).sum() - (yb * xb).sum() / (xb * xb).sum()
    obs = f(y1, x1, y2, x2); c = 0
    for _ in range(n):
        s = rng.random(len(y1)) < 0.5
        c += abs(f(np.where(s, y2, y1), np.where(s, x2, x1), np.where(s, y1, y2), np.where(s, x1, x2))) >= abs(obs)
    return dict(diff=float(obs), p=float((1 + c) / (1 + n)), n_perm=n)


# ------------------------------------------------------------------------------------------------------------ DAS
def das(M, X, v, m, tgt, idx_tr, idx_te, rng, sites, steps, bs=128):
    """tgt: dict V -> window label per jet (0/1). Returns per site, per V (and random): IIA overall / on discriminating pairs."""
    t = M.t; dev = M.dev; x = M.x(X); cnt = m[:, 0].sum(1).astype(int); C = 128
    T = lambda a: t.from_numpy(np.ascontiguousarray(a)).to(dev)
    out = {}
    for site in sites:
        blk = M.m.blocks[site]; st = {}
        def hook(mod, inp, o):
            if st.get('rec'): st['h'] = ((o * st['tm']).sum(0) / st['tm'].sum(0)).detach()
            if st.get('u') is not None:
                u = st['u']; pb = ((o * st['tm']).sum(0) / st['tm'].sum(0)) @ u
                o = o + ((st['ps'] - pb)[None, :, None] * u[None, None, :]) * st['tm']
            return o
        hk = blk.register_forward_hook(hook)
        try:
            # jet-mean token at the site for every jet (no intervention)
            Hm = np.zeros((len(x), C), np.float32); order = np.argsort(cnt, kind='stable')
            with t.no_grad():
                for i in range(0, len(x), 256):
                    b = order[i:i + 256]; P = int(cnt[b].max()); mb = T(m[b, :, :P]); st.update(rec=True, u=None, tm=mb[:, 0, :].T[..., None])
                    M.m(T(x[b, :, :P]), T(v[b, :, :P]), mb); Hm[b] = st['h'].float().cpu().numpy()
            st['rec'] = False; Hm_t = T(Hm)

            def run_int(b, src, u, grad):
                P = int(cnt[b].max()); mb = T(m[b, :, :P]); st['tm'] = mb[:, 0, :].T[..., None]; st['u'] = u
                st['ps'] = Hm_t[T(src)] @ u
                if grad: return M.m(T(x[b, :, :P]), T(v[b, :, :P]), mb)
                with t.no_grad(): return M.m(T(x[b, :, :P]), T(v[b, :, :P]), mb)

            src_te = rng.permutation(idx_te)      # one source per base jet, shared by every direction (paired comparisons)

            def evaluate(u):
                base = idx_te; call = np.zeros(len(base), int); o = np.argsort(cnt[base], kind='stable')
                for i in range(0, len(base), 256):
                    j = o[i:i + 256]; L = run_int(base[j], src_te[j], u, False); call[j] = L.argmax(1).cpu().numpy()
                return src_te, call

            res = {}
            for V in tgt:
                g = t.Generator().manual_seed(int(rng.integers(1 << 30)))
                w = t.randn(C, generator=g).to(dev).requires_grad_(True); opt = t.optim.Adam([w], lr=0.02); losses = []
                for s in range(steps):
                    b = np.sort(rng.choice(idx_tr, bs, replace=False)); src = rng.choice(idx_tr, bs)
                    u = w / w.norm(); L = run_int(b, src, u, True)
                    loss = t.nn.functional.cross_entropy(L.float(), T(tgt[V][src].astype(np.int64)))
                    opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss))
                u = (w / w.norm()).detach(); res[V] = dict(u=u, loss0=float(np.mean(losses[:10])), loss1=float(np.mean(losses[-20:])))
            ev = {}
            names = list(tgt)
            for V in names:
                ev[V] = evaluate(res[V]['u'])
            for k in range(4):
                g = t.Generator().manual_seed(1000 + k); u = t.randn(C, generator=g).to(dev); u = u / u.norm()
                ev[f'rand{k}'] = evaluate(u)
            ev['none'] = evaluate(t.zeros(C, device=dev))     # no intervention (u = 0): the base calls
            cos = float((res['minv']['u'] * res['mA']['u']).sum())
            st['u'] = None
        finally:
            hk.remove()
        base = idx_te; o = {}
        for key, (src, call) in ev.items():
            disc = tgt['minv'][src] != tgt['mA'][src]
            e = {}
            for V in names:
                ok = call == tgt[V][src]; Wb = rng.integers(0, len(ok), (NB, len(ok)))
                e[f'iia_vs_{V}'] = [float(ok.mean())] + ci(ok[Wb].mean(1))
                okd = ok[disc]; Wd = rng.integers(0, len(okd), (NB, len(okd))) if len(okd) else None
                e[f'iia_disc_vs_{V}'] = [float(okd.mean())] + ci(okd[Wd].mean(1)) if len(okd) else None
            e['n_pairs'] = int(len(base)); e['n_disc'] = int(disc.sum())
            if key in res: e.update(loss_start=res[key]['loss0'], loss_end=res[key]['loss1'])
            o[key] = e
        # paired comparison on the discriminating pairs (same base / source pairs): does the m_inv-trained direction reach the
        # m_inv-window target more often than the m_A-trained direction reaches the m_A-window target?
        src = ev['minv'][0]; disc = tgt['minv'][src] != tgt['mA'][src]
        ok_i = (ev['minv'][1] == tgt['minv'][src])[disc]; ok_a = (ev['mA'][1] == tgt['mA'][src])[disc]
        o['paired_disc'] = dict(own_minv=float(ok_i.mean()) if len(ok_i) else None, own_mA=float(ok_a.mean()) if len(ok_a) else None,
                                mcnemar=mcnemar(ok_i, ok_a), cos_u_minv_u_mA=cos)
        out[f'block{site + 1}'] = o
        log(f'  DAS {M.name} block {site + 1}: ' + '; '.join(f'{k}: IIA vs minv {v["iia_vs_minv"][0]:.3f} vs mA {v["iia_vs_mA"][0]:.3f}, '
                                                          f'disc minv {v["iia_disc_vs_minv"][0] if v["iia_disc_vs_minv"] else float("nan"):.3f} mA {v["iia_disc_vs_mA"][0] if v["iia_disc_vs_mA"] else float("nan"):.3f}'
                                                          for k, v in o.items() if k in ('minv', 'mA', 'rand0', 'none'))
            + f'; cos(u_minv, u_mA) {cos:+.2f}')
    return out


# ------------------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--models', default='a,b,c,s,released_full,released_kin')
    ap.add_argument('--n', type=int, default=20000); ap.add_argument('--epoch', type=int, default=None); ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--tag', default=None); ap.add_argument('--das-steps', type=int, default=400)
    ap.add_argument('--no-eap', dest='eap', action='store_false'); ap.add_argument('--eap-disc', type=int, default=1500); ap.add_argument('--eap-eval', type=int, default=3000)
    a = ap.parse_args()
    if a.smoke: a.n, a.das_steps, a.eap_disc, a.eap_eval = 3000, 30, 60, 120
    rng = np.random.default_rng(1414); OUT.mkdir(parents=True, exist_ok=True)
    J0 = sample(a.n); y10 = J0['label'].astype(int); N = len(y10)
    pt, de, dp, ms = kin(J0); real = pt > 0
    Jb = build(J0, pt, de, dp, ms); Mb = moments(*kin(Jb)[:3])
    E = {'phi': build(J0, pt, de, dp + 0.06 * real, ms)}
    M2p = moments(*kin(E['phi'])[:3])['M0M2']; okk = Mb['M0M2'] > 0      # single-particle jets at the axis: M0M2 = 0 -> k = 1
    kj = np.where(okk, np.sqrt(M2p / np.where(okk, Mb['M0M2'], 1.0)), 1.0)[:, None]
    E['dil'] = build(J0, pt, de * kj, dp * kj, ms); E['k1.1'] = build(J0, pt, de * 1.1, dp * 1.1, ms)
    Jr = dict(Jb); Jr['jet_phi'] = (Jb['jet_phi'] + 0.06).astype(np.float32); c_, s_ = np.cos(0.06), np.sin(0.06)
    Jr['px'] = (c_ * Jb['px'] - s_ * Jb['py']).astype(np.float32); Jr['py'] = (s_ * Jb['px'] + c_ * Jb['py']).astype(np.float32); E['rot'] = Jr
    mi, mA = masses(Jb); side = np.load(SIDE / ('labels_smoke.npz' if a.smoke else 'labels.npz'))
    c, w = float(side['c']), float(side['w']); lab = dict(a=(np.abs(mi - M0_) < W0_).astype(int), b=(np.abs(mA - c) < w).astype(int))
    lab['c'] = lab['s'] = lab['a']
    edm = {k: masses(Je) for k, Je in E.items()}
    labE = {k: dict(a=(np.abs(edm[k][0] - M0_) < W0_).astype(int), b=(np.abs(edm[k][1] - c) < w).astype(int)) for k in E}
    for k in E: labE[k]['c'] = labE[k]['s'] = labE[k]['a']
    res = dict(N=N, seed=14, c=c, w=w, n_M0M2_zero=int((~okk).sum()), window_a=[M0_, W0_], class_counts={cl: int((y10 == CL.index(n)).sum()) for cl, n in (('QCD', 'QCD'), ('Z', 'Zqq'), ('W', 'Wqq'))},
               pos_rate=dict(a=float(lab['a'].mean()), b=float(lab['b'].mean())), ab_agree=float((lab['a'] == lab['b']).mean()),
               side_label_agree=dict(a=float((side['test_a'][:N].astype(int) == lab['a']).mean()) if len(side['test_a']) == N else None),
               edits={k: dict(dlogminv=float(np.mean(np.log(np.maximum(edm[k][0], 1)) - np.log(np.maximum(mi, 1)))),
                              dlogmA=float(np.mean(np.log(np.maximum(edm[k][1], 1e-6)) - np.log(np.maximum(mA, 1e-6)))),
                              label_flip=dict(a=float((labE[k]['a'] != lab['a']).mean()), b=float((labE[k]['b'] != lab['b']).mean()))) for k in E},
               models={})
    log(f'{N} jets; c {c:.2f} w {w:.2f}; pos rate a {lab["a"].mean():.3f} b {lab["b"].mean():.3f}; edits: ' +
        '; '.join(f'{k}: dlog m_inv {v["dlogminv"]:+.4f} dlog m_A {v["dlogmA"]:+.4f} label flips a {v["label_flip"]["a"]:.3f} b {v["label_flip"]["b"]:.3f}' for k, v in res['edits'].items()))
    X = {kd: {k: inputs(Je, kd) for k, Je in [('base', Jb)] + list(E.items())} for kd in ('kin', 'full')}
    fit, val, tr, te = split(N); Wb = boot_weights(N, rng); dd = {}; plist, pkeys = [], []
    for name in a.models.split(','):
        try:
            M = Model(name, a.smoke, a.epoch)
        except FileNotFoundError as ex:
            log(f'{name}: no checkpoint ({ex}); skipped'); continue
        Xk = X[M.kind]; t1 = time.time()
        L0, H0 = M.run(*Xk['base']); c0 = L0.argmax(1); d0 = M.read(L0); row = dict(path=getattr(M, 'path', None))
        if M.arm:
            own = lab[M.arm]; row['acc_own_label'] = float((c0 == own).mean()); row['acc_ci'] = wilson(int((c0 == own).sum()), N)
            row['acc_by_class'] = {cl: float((c0 == own)[y10 == CL.index(n)].mean()) for cl, n in (('QCD', 'QCD'), ('Z', 'Zqq'), ('W', 'Wqq'))}
            row['confusion_label_vs_call'] = [[int(((own == i) & (c0 == j)).sum()) for j in (0, 1)] for i in (0, 1)]
            row['acc_vs_other_label'] = {k: float((c0 == lab[k]).mean()) for k in ('a', 'b')}
        else:
            if not a.smoke:                    # reference: the P1 sample (rows 3,000-4,000 of each class's test file 100)
                from interp.part_core import sample as p1sample
                if 'P1' not in X:
                    JP = p1sample(1000, 3000); X['P1'] = {kd: inputs(build(JP, *kin(JP)), kd) for kd in ('kin', 'full')}; X['P1y'] = JP['label'].astype(int); del JP
                LP, _ = M.run(*X['P1'][M.kind]); row['acc10_P1_sample'] = float((LP.argmax(1) == X['P1y']).mean())
                log(f'{name}: P1-sample 10-class accuracy {row["acc10_P1_sample"]:.4f} (P1: full 0.859, kin 0.745)')
            row['acc10'] = float((c0 == y10).mean()); sub3 = [iQ, iZ, iW]; c3 = np.array(sub3)[L0[:, sub3].argmax(1)]
            row['acc3_restricted'] = float((c3 == y10).mean()); row['WZ'] = float((c0[y10 == iW] == iZ).mean())
            row['confusion10_rows_QCD_Z_W'] = [[int(((y10 == i) & (c0 == j)).sum()) for j in range(10)] for i in sub3]
        rd = Ridge(H0, np.log(np.maximum(mi, 1.0)), fit, val, tr); row['probe_r2_te'] = float(1 - ((rd(H0[te]) - np.log(np.maximum(mi[te], 1))) ** 2).sum() / ((np.log(np.maximum(mi[te], 1)) - np.log(np.maximum(mi[te], 1)).mean()) ** 2).sum())
        Dd, Dp, calls = {}, {}, {}
        for k in E:
            L, H = M.run(*Xk[k]); Dd[k] = M.read(L) - d0; Dp[k] = rd(H[te]) - rd(H0[te]); calls[k] = L.argmax(1)
            pos0, pos1 = (c0 == 1, calls[k] == 1) if M.arm else (c0 == iW, calls[k] == iW)
            mc = mcnemar(pos0, pos1); plist.append(mc['p']); pkeys.append((name, k))
            ch = int((calls[k] != c0).sum())
            row.setdefault('edits', {})[k] = dict(changed=ch / N, changed_ci=wilson(ch, N), pos_call=[float(pos0.mean()), float(pos1.mean())], mcnemar_pos=mc,
                                                  dd_mean=float(Dd[k].mean()), dprobe_mean=float(Dp[k].mean()), max_abs_dd=float(np.abs(Dd[k]).max()))
            if M.arm: row['edits'][k]['acc_own_label_after'] = float((calls[k] == labE[k][M.arm]).mean())
        R, Rb = slope_boot(Dd['phi'], Dd['dil'], Wb); row['R'] = [R] + ci(Rb)
        Rk, Rkb = slope_boot(Dd['rot'], Dd['dil'], Wb); row['R_rot_null'] = [Rk] + ci(Rkb)
        pr = float(Dp['phi'].mean() / Dp['dil'].mean()); Wt = boot_weights(len(te), rng, 1000)
        row['R_probe'] = [pr] + ci((Wt @ Dp['phi']) / (Wt @ Dp['dil']))
        if not M.arm:
            Wm = y10 == iW; s, sb = slope_boot(Dd['phi'][Wm], Dd['dil'][Wm], boot_weights(int(Wm.sum()), rng, 1000)); row['R_Wjets'] = [s] + ci(sb)
        dd[name] = (Dd['phi'], Dd['dil'])
        # test 2: route split
        row['routes'] = {}
        for k in ('dil', 'k1.1', 'phi'):
            Lpp, _ = M.run(Xk[k][0], Xk['base'][1], Xk['base'][2]); Lpr, _ = M.run(Xk['base'][0], Xk[k][1], Xk['base'][2])
            dpp, dpr = M.read(Lpp) - d0, M.read(Lpr) - d0
            if k == 'phi':
                row['routes'][k] = dict(max_abs_pair=float(np.abs(dpr).max()), mean_abs_pair=float(np.abs(dpr).mean()), mean_abs_full=float(np.abs(Dd[k]).mean()),
                                        mean_abs_rot_null=float(np.abs(Dd['rot']).mean()), pp_slope=slope_boot(dpp, Dd[k], Wb)[0]); continue
            s1, b1 = slope_boot(dpp, Dd[k], Wb); s2, b2 = slope_boot(dpr, Dd[k], Wb)
            row['routes'][k] = dict(per_particle=[s1] + ci(b1), pair=[s2] + ci(b2), sum=s1 + s2,
                                    additivity_r2=float(1 - ((dpp + dpr - Dd[k]) ** 2).sum() / ((Dd[k] - Dd[k].mean()) ** 2).sum()))
        log(f'{name}: ' + (f'acc own label {row["acc_own_label"]:.4f} (vs a {row["acc_vs_other_label"]["a"]:.3f}, b {row["acc_vs_other_label"]["b"]:.3f})' if M.arm else
                           f'acc10 {row["acc10"]:.4f}, 3-class restricted {row["acc3_restricted"]:.4f}, W->Z {row["WZ"]:.3f}')
            + f'; R {R:+.3f} [{row["R"][1]:+.3f}, {row["R"][2]:+.3f}] (rot null {Rk:+.4f}); R_probe {pr:+.3f}; calls changed phi {row["edits"]["phi"]["changed"]:.3f} dil {row["edits"]["dil"]["changed"]:.3f} rot {row["edits"]["rot"]["changed"]:.4f}; '
            + '; '.join(f'route {k}: pp {v["per_particle"][0]:.3f} pair {v["pair"][0]:.3f}' for k, v in row['routes'].items() if k != 'phi')
            + f'; phi pair-route mean|dd| {row["routes"]["phi"]["mean_abs_pair"]:.2e} (full {row["routes"]["phi"]["mean_abs_full"]:.2e}, rot null {row["routes"]["phi"]["mean_abs_rot_null"]:.2e}) ({time.time() - t1:.0f} s)')
        if M.arm:
            tgt = dict(minv=lab['a'], mA=lab['b'])
            row['das'] = das(M, Xk['base'][0], Xk['base'][1], Xk['base'][2], tgt, tr, te, rng, sites=(1, 7), steps=a.das_steps)
        if a.eap and (M.arm or M.kind == 'kin'):
            from interp.p14_eap_part import run_eap
            y3 = np.select([y10 == iQ, y10 == iZ, y10 == iW], [0, 1, 2], -1)
            if M.arm:
                own = lab[M.arm]; pool_d, pool_e = tr, te
            else:                              # released_kin: W (1) vs Z (0) jets only
                own = (y10 == iW).astype(int); pool_d, pool_e = tr[np.isin(y10[tr], [iW, iZ])], te[np.isin(y10[te], [iW, iZ])]
            r2 = np.random.default_rng(2014)
            idd = np.sort(r2.choice(pool_d, min(a.eap_disc, len(pool_d)), replace=False)); ide = np.sort(r2.choice(pool_e, min(a.eap_eval, len(pool_e)), replace=False))
            try:
                row['eap'] = run_eap(M, Jb, own, y3, idd, ide, r2, nrand=2 if a.smoke else 20, log=log)
            except Exception as ex:
                import traceback; traceback.print_exc(); row['eap'] = dict(error=repr(ex))
        res['models'][name] = row; del M
        json.dump(res, open(OUT / f'p14_planted_{a.tag or ("smoke" if a.smoke else "run")}.json', 'w'), indent=1, default=float)
    hp = holm(plist)
    for (nm, k), p in zip(pkeys, hp): res['models'][nm]['edits'][k]['mcnemar_pos']['p_holm'] = p
    res['perm'] = {}
    for x1, x2 in (('a', 'b'), ('a', 's'), ('a', 'c'), ('b', 'c'), ('a', 'released_kin'), ('a', 'released_full')):
        if x1 in dd and x2 in dd:
            res['perm'][f'{x1}_vs_{x2}'] = perm_test(dd[x1][0], dd[x1][1], dd[x2][0], dd[x2][1], rng)
            log(f'permutation H0 R_{x1} = R_{x2}: diff {res["perm"][f"{x1}_vs_{x2}"]["diff"]:+.3f}, p {res["perm"][f"{x1}_vs_{x2}"]["p"]:.4f}')
    fn = OUT / f'p14_planted_{a.tag or ("smoke" if a.smoke else "run")}.json'
    json.dump(res, open(fn, 'w'), indent=1, default=float); log(f'wrote {fn}')
    try:
        plot(res, fn.with_suffix('.png'))
    except Exception as ex:
        log(f'plot failed: {ex!r}')
    log('P14 analysis DONE')


LABEL = dict(a='a: $m$\nwindow', b='b: $m_A$\nwindow', c='c: $m$ window,\npair only', s='s: shuffled\n(constant)',
             released_full='released\nfull', released_kin='released\nkin')


def plot(res, path):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    names = [n for n in ('a', 'b', 'c', 's', 'released_full', 'released_kin') if n in res['models']]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw=dict(width_ratios=[1.2, 1]))
    ax = axs[0]
    for i, n in enumerate(names):
        r = res['models'][n]['R']; col = '#3b6fb6' if n in ('a', 'b', 'c', 's') else '#999999'
        ax.bar(i, r[0], 0.6, color=col); ax.errorbar(i, r[0], yerr=[[r[0] - r[1]], [r[2] - r[0]]], color='k', capsize=3, lw=1)
        ax.text(i, max(r[2], 0) + 0.04, f'{r[0]:.2f}', ha='center', fontsize=8)
    ax.axhline(1, color='#d08a2c', ls='--', lw=1, label='reads axis-moment mass $m_A$ (R = 1)')
    ax.axhline(0, color='k', lw=0.8, label='reads invariant mass $m$ (R = 0)')
    ax.set_xticks(range(len(names))); ax.set_xticklabels([LABEL[n] for n in names], fontsize=7.5)
    ax.set_ylabel('R = response to $\\phi$ + 0.06 rad / response to\nmatched real mass change (95 % CI)')
    ax.set_title(f'Rigid rotation about the stored axis ($m$ unchanged, $m_A$ +{100 * (np.exp(res["edits"]["phi"]["dlogmA"]) - 1):.0f} %)', fontsize=9.5); ax.legend(fontsize=7.5, loc='upper right')
    ax = axs[1]; w = 0.38
    for i, n in enumerate(names):
        rt = res['models'][n]['routes']['dil']
        for j, (k, col, lab) in enumerate((('per_particle', '#3b6fb6', 'per-particle inputs only'), ('pair', '#d08a2c', 'pair features only'))):
            v = rt[k]; x = i + (j - 0.5) * w
            ax.bar(x, v[0], w * 0.95, color=col, label=lab if i == 0 else None)
            ax.errorbar(x, v[0], yerr=[[v[0] - v[1]], [v[2] - v[0]]], color='k', capsize=2, lw=0.8)
    ax.axhline(1, color='k', lw=0.5, ls=':'); ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(range(len(names))); ax.set_xticklabels([LABEL[n] for n in names], fontsize=7.5)
    ax.set_ylabel('share of the read-out change'); ax.set_title('Route of a real mass change (matched dilation)', fontsize=9.5); ax.legend(fontsize=7.5)
    fig.text(0.5, -0.03, f'N = {res["N"]} random held-out W/Z/QCD JetClass jets (seed 14). Read-out: logit difference (planted: inside vs outside the '
             f'window; released: W vs Z). Bars: per-jet slope, bootstrap 95 % CI.', ha='center', fontsize=7.5)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches='tight')


if __name__ == '__main__':
    if '--plot' in sys.argv:
        f = Path(sys.argv[sys.argv.index('--plot') + 1]); plot(json.load(open(f)), f.with_suffix('.png'))
    else:
        main()
