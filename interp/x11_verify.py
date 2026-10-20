"""X11 verifier: rerun the X11 atlas's key comparisons on fresh jets and add two missing controls.

Parts (python -m interp.x11_verify --part <p>; writes $AI_RESULTS/jedi/x11_verify_<p>.json):
  main       JEDI-linear S: native vs 200 fresh raw rotations (rng 5000+i) + 20 z-scored rotations (rng 6000+i) + SAE 2x/4x
             top-d; causal re-test (pT or angle x1.1 in the top term's window) of the named native S units on fresh causal
             jets, with the mismatched-formula control. JEDI-linear h4: native vs 40 raw rotations.
  untrained  the untrained JEDI-linear (Jedi(64, untrained=True, seed=0)) S layer: native vs 50 raw + 10 z-scored rotations
             (does the native-basis excess need training?).
  qlib       the generic Q observable library of the test file (383 standard jet observables, not built from JEDI-linear's
             mechanism) in place of the hand-built rank-window library: JEDI-linear S native vs 50 raw rotations, Deep Sets S
             native vs 20 raw rotations (is the excess, and the JEDI vs Deep Sets gap, tied to the hand-built library?).
Jets (all parts): fit = sample(4000, 101) (fixed permutation rng(0): 2/3 fit-train, 1/3 fit-dev); test = sample(4000, 103)
minus every jet also in the fit sample; causal = 3,000 test jets (rng 5) in JEDI-linear's l1 input range under every edit.
Naming rule, edits, slope rule as interp.x11_sae_atlas (imported unchanged). Bootstrap: B = 500 test-jet resamples (rng 11).
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import interp.x11_sae_atlas as X

OUT = Path(os.environ.get('AI_RESULTS', Path(__file__).resolve().parents[1] / 'results')) / 'jedi'; OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
B, BC, N_CAUSAL = 500, 500, 3000


def log(*a): print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


class LeanFitter(X.Fitter):
    """X.Fitter with the candidate matrix preallocated (float64), for the 383-observable Q library within 2 GiB"""

    def __init__(self, Htr, Hdev, keys, n_sel, seed=1):
        from jetdistill import mars
        self.mars = mars; self.keys = keys; self.Htr, self.Hdev = Htr, Hdev; Ntr = len(Htr[keys[0]])
        self.sel = np.sort(np.random.default_rng(seed).choice(Ntr, min(n_sel, Ntr), replace=False))
        self.knots = {k: mars.thresholds(Htr[k], False, False) for k in keys}
        ncol = sum(1 + 2 * len(self.knots[k]) for k in keys)
        self.C = np.empty((len(self.sel), ncol)); desc, self.obs_cols = [], {}; c = 0
        for k in keys:
            v = Htr[k][self.sel]; i0 = c
            self.C[:, c] = v; desc.append(dict(q=k, kind='lin', t=None)); c += 1
            for t in self.knots[k]: self.C[:, c] = np.maximum(0, v - t); desc.append(dict(q=k, kind='gt', t=float(t))); c += 1
            for t in self.knots[k]: self.C[:, c] = np.maximum(0, t - v); desc.append(dict(q=k, kind='lt', t=float(t))); c += 1
            self.obs_cols[k] = (i0 + 1, c)
        self.desc = desc; self.Cn2 = (self.C ** 2).sum(0)
        Z = np.stack([Htr[k][self.sel] for k in keys], 1); self.Zs = (Z - Z.mean(0)) / np.maximum(Z.std(0), 1e-12)
        log(f'lean fitter: {len(keys)} observables, {ncol} candidate columns, {len(self.sel)} selection jets')


def jets():
    from interp.jedi_core import sample
    x, y, Q = sample(4000, 101); xt, yt, Qt = sample(4000, 103)
    i0, i1 = X.indices(4000, 101), X.indices(4000, 103); keep = ~np.isin(i1, i0)
    xt, yt = xt[keep], yt[keep]; Qt = {k: v[keep] for k, v in Qt.items()}
    J = len(x); perm = np.random.default_rng(0).permutation(J); nf = 2 * J // 3
    log(f'jets: fit {J} (train {nf}, dev {J - nf}), test {len(xt)} (removed {int((~keep).sum())} overlapping)')
    return x, y, Q, xt, yt, Qt, perm[:nf], perm[nf:], dict(fit=J, fit_train=nf, fit_dev=J - nf, test=len(xt), overlap_removed=int((~keep).sum()))


def rot(d, seed):
    M = np.random.default_rng(seed).standard_normal((d, d)); Qm, Rr = np.linalg.qr(M); return Qm * np.sign(np.diag(Rr))


def name_instance(fitter, Ytr, Ydv, Yt, Ht, W):
    D = Yt.shape[1]; Ft = np.zeros_like(Yt); dirs = []
    for j in range(D):
        if np.ptp(Ytr[:, j]) < 1e-9: dirs.append(dict(j=j, r2=0.0, K=0, F=None)); Ft[:, j] = Ytr[:, j].mean(); continue
        F = fitter.fit(Ytr[:, j], Ydv[:, j]); Ft[:, j] = fitter.predict(F, Ht)
        r2 = float(1 - ((Yt[:, j] - Ft[:, j]) ** 2).mean() / max(Yt[:, j].var(), 1e-300))
        sp = [abs(c) * float(fitter.tv(t, fitter.Htr).std()) for t, c in zip(F['terms'], F['coef'][1:])]
        top = F['terms'][int(np.argmax(sp))] if sp else None
        dirs.append(dict(j=j, r2=r2, K=F['K'], F=F, top=X.term_str(top) if top else None, top_q=top['q'] if top else None))
    Rb = X.boot_r2(W, Yt, Ft); fb = (Rb >= X.R2_NAMED).mean(1)
    fr = float(np.mean([dd['r2'] >= X.R2_NAMED for dd in dirs]))
    return fr, fb, dirs


def compare(label, fitter, layer_acts, Ht, W, n_raw, n_std, seed_raw, seed_std, extra_bases=None):
    """layer_acts = (h_train, h_dev, h_test) for all units; returns dict with native, rotation fractions, stats"""
    htr, hdv, ht = layer_acts
    live = np.flatnonzero((htr.std(0) > 1e-6) & ((htr > 1e-6).mean(0) >= 0.005)); d = len(live)
    mu, sd = htr[:, live].mean(0), htr[:, live].std(0)
    log(f'--- {label}: {d} live units: {live.tolist()}')
    t0 = time.time()
    fr_n, fb_n, dirs_n = name_instance(fitter, htr[:, live], hdv[:, live], ht[:, live], Ht, W)
    log(f'{label} native: named {fr_n:.3f} ({sum(dd["r2"] >= X.R2_NAMED for dd in dirs_n)}/{d}); {time.time() - t0:.0f} s')
    out = dict(live=live.tolist(), d=d, native=fr_n, native_ci=X.ci(fb_n),
               native_dirs=[dict(unit=int(live[dd['j']]), r2=dd['r2'], K=dd['K'], top=dd.get('top')) for dd in dirs_n])
    for kind, n_, s0 in (('rot_raw', n_raw, seed_raw), ('rot_std', n_std, seed_std)):
        frs, fbs = [], []
        for i in range(n_):
            Qm = rot(d, s0 + i)
            T = (lambda Z: Z[:, live] @ Qm.T) if kind == 'rot_raw' else (lambda Z: ((Z[:, live] - mu) / sd) @ Qm.T)
            fr, fb, _ = name_instance(fitter, T(htr), T(hdv), T(ht), Ht, W); frs.append(fr); fbs.append(fb)
            if i % 10 == 0 or i == n_ - 1: log(f'{label} {kind}{i}: named {fr:.3f}; running max {max(frs):.3f}')
        if not n_: continue
        frs = np.array(frs); dlt = fb_n - np.mean(fbs, 0)
        out[kind] = dict(n=n_, fracs=frs.tolist(), mean=float(frs.mean()), sd=float(frs.std(ddof=1)) if n_ > 1 else None,
                         min=float(frs.min()), max=float(frs.max()), n_ge_native=int((frs >= fr_n).sum()),
                         p_perm=float((1 + (frs >= fr_n).sum()) / (n_ + 1)),
                         z=float((fr_n - frs.mean()) / frs.std(ddof=1)) if n_ > 1 and frs.std() > 0 else None,
                         diff=float(fr_n - frs.mean()), diff_ci=X.ci(dlt))
        log(f'{label} {kind}: native {fr_n:.3f} vs mean {frs.mean():.3f} [{frs.min():.3f}, {frs.max():.3f}], '
            f'{int((frs >= fr_n).sum())}/{n_} >= native, p {out[kind]["p_perm"]:.4f}')
    return out, live, dirs_n, (mu, sd), fb_n


def part_main(a):
    from interp.jedi_core import Jedi
    x, y, Q, xt, yt, Qt, itr, idv, Jinfo = jets()
    R = dict(part='main', jets=Jinfo)
    from interp.e11_mars_all import WINDOWS
    EDITS = [(k, w) for w in WINDOWS for k in ('pt', 'r')]
    net = Jedi(64); rng = np.random.default_rng(5); cand = rng.permutation(len(xt)); ok = X.in_range(net, xt[cand])
    for k, w in EDITS: ok &= X.in_range(net, X.edit(xt[cand], k, w))
    xc = xt[cand[ok][:N_CAUSAL]]; R['jets'].update(causal=len(xc), causal_in_range_fraction=float(ok.mean()))
    xe = {f'{k}:{w}': X.edit(xc, k, w) for k, w in EDITS}
    H = X.library(x); Ht = X.library(xt); Hc = X.library(xc); He = {e: X.library(v) for e, v in xe.items()}
    keys = [k for k in H if np.isfinite(H[k]).all() and np.isfinite(Ht[k]).all() and np.ptp(H[k][itr]) > 0]
    fitter = X.Fitter({k: H[k][itr] for k in keys}, {k: H[k][idv] for k in keys}, keys, X.N_SEL)
    run = lambda z: net.run(z, keep=('S', 'h4', 'logits'))
    A, At, Ac = run(x), run(xt), run(xc); Ae = {e: run(v)['S'] for e, v in xe.items()}
    R['accuracy'] = dict(fit=float((A['logits'].argmax(1) == y).mean()), test=float((At['logits'].argmax(1) == yt).mean()))
    log(f'accuracy {R["accuracy"]}')
    W = np.stack([np.bincount(r_, minlength=len(xt)) for r_ in np.random.default_rng(11).integers(0, len(xt), (B, len(xt)))]).astype(np.float64)
    hS = A['S']
    S, live, dirs_n, (mu, sd), fb_n = compare('jedi S', fitter, (hS[itr], hS[idv], At['S']), Ht, W, 200, 20, 5000, 6000)
    # SAE top-d
    d = len(live); s_rms = float(np.sqrt((sd ** 2).mean())); Xs = (hS[itr][:, live] - mu) / s_rms; Xsd = (hS[idv][:, live] - mu) / s_rms
    S['sae'] = {}
    for mult in (2, 4):
        trial = []
        for lam in X.LAMBDAS:
            P, encn, fvu, l0 = X.train_sae(Xs, Xsd, mult * d, lam, seed=mult + 10); trial.append(dict(lam=lam, fvu=fvu, l0=l0))
            if fvu <= 0.10: break
        enc = lambda Z: encn((Z[:, live] - mu) / s_rms); f = enc(hS[itr])
        alive = set(np.flatnonzero((f > 1e-6).mean(0) >= 0.005).tolist()); topk = [int(i) for i in np.argsort(-(f ** 2).mean(0)) if i in alive][:d]
        fr, fb, _ = name_instance(fitter, enc(hS[itr])[:, topk], enc(hS[idv])[:, topk], enc(At['S'])[:, topk], Ht, W)
        S['sae'][f'{mult}x'] = dict(trials=trial, frac=fr, minus_native=fr - S['native'], ci=X.ci(fb - fb_n), p=X.boot_p2(fb - fb_n))
        log(f'jedi S SAE {mult}x: lambda {trial[-1]["lam"]}, FVU {trial[-1]["fvu"]:.3f}, top-{len(topk)} named {fr:.3f} (native {S["native"]:.3f})')
    # causal re-test of the named native S units
    hc = Ac['S'][:, live]; he = {e: Ae[e][:, live] for e in xe}; sdy = At['S'][:, live].std(0)
    BIc = np.random.default_rng(13).integers(0, len(xc), (BC, len(xc))); named = [dd for dd in dirs_n if dd['r2'] >= X.R2_NAMED]; ca = []
    for dd in named:
        j = dd['j']; w = X.window_of(dd['top_q']); best = None
        for kind in ('pt', 'r'):
            e = f'{kind}:{w}'; dp = fitter.predict(dd['F'], He[e]) - fitter.predict(dd['F'], Hc); sz = float(np.abs(dp).mean() / sdy[j])
            if best is None or sz > best[1]: best = (e, sz, dp)
        e, sz, dp = best; da = he[e][:, j] - hc[:, j]; P_, A_ = dp[BIc], da[BIc]
        slb = ((P_ - P_.mean(1, keepdims=True)) * (A_ - A_.mean(1, keepdims=True))).mean(1) / np.maximum(P_.var(1), 1e-300)
        sl = float(np.cov(dp, da)[0, 1] / dp.var()); c = X.ci(slb)
        ca.append(dict(unit=int(live[j]), top=dd['top'], edit=e, pred_size_sd=sz, slope=sl, slope_ci=c,
                       causal='pass' if (c[0] >= X.SLOPE_LO and c[1] <= X.SLOPE_HI) else 'fail', _dp=dp, _da=da))
        log(f'causal S{live[j]}: {e} slope {sl:.3f} [{c[0]:.3f}, {c[1]:.3f}] {ca[-1]["causal"]}')
    mis = []
    if len(ca) >= 2:
        rr = np.random.default_rng(17)
        while True:
            pm = rr.permutation(len(ca))
            if (pm != np.arange(len(ca))).all(): break
        for k1, k2 in enumerate(pm):
            dd, o = named[k1], named[k2]; e = ca[k1]['edit']
            dp = fitter.predict(o['F'], He[e]) - fitter.predict(o['F'], Hc); da = ca[k1]['_da']
            if dp.var() <= 0: mis.append(False); continue
            P_, A_ = dp[BIc], da[BIc]
            slb = ((P_ - P_.mean(1, keepdims=True)) * (A_ - A_.mean(1, keepdims=True))).mean(1) / np.maximum(P_.var(1), 1e-300)
            c = X.ci(slb); mis.append(bool(c[0] >= X.SLOPE_LO and c[1] <= X.SLOPE_HI))
    for c_ in ca: c_.pop('_dp'); c_.pop('_da')
    S['causal'] = dict(units=ca, n_pass=sum(c_['causal'] == 'pass' for c_ in ca), n_named=len(ca), mismatched_pass=int(sum(mis)), mismatched_n=len(mis))
    log(f'causal: {S["causal"]["n_pass"]}/{len(ca)} pass; mismatched {sum(mis)}/{len(mis)}')
    R['S'] = S; (OUT / 'x11_verify_main.json').write_text(json.dumps(X.clean(R), ensure_ascii=False))
    h4 = A['h4']
    R['h4'], *_ = compare('jedi h4', fitter, (h4[itr], h4[idv], At['h4']), Ht, W, 40, 0, 7000, 0)
    R['seconds'] = time.time() - T0
    (OUT / 'x11_verify_main.json').write_text(json.dumps(X.clean(R), ensure_ascii=False)); log('wrote x11_verify_main.json')


def part_untrained(a):
    from interp.jedi_core import Jedi
    x, y, Q, xt, yt, Qt, itr, idv, Jinfo = jets()
    H = X.library(x); Ht = X.library(xt)
    keys = [k for k in H if np.isfinite(H[k]).all() and np.isfinite(Ht[k]).all() and np.ptp(H[k][itr]) > 0]
    fitter = X.Fitter({k: H[k][itr] for k in keys}, {k: H[k][idv] for k in keys}, keys, X.N_SEL)
    W = np.stack([np.bincount(r_, minlength=len(xt)) for r_ in np.random.default_rng(11).integers(0, len(xt), (B, len(xt)))]).astype(np.float64)
    net = Jedi(64, untrained=True, seed=0); A = net.run(x, keep=('S', 'logits')); At = net.run(xt, keep=('S', 'logits'))
    R = dict(part='untrained', jets=Jinfo, accuracy_test=float((At['logits'].argmax(1) == yt).mean()))
    hS = A['S']; R['S'], *_ = compare('untrained jedi S', fitter, (hS[itr], hS[idv], At['S']), Ht, W, 50, 10, 5000, 6000)
    R['seconds'] = time.time() - T0
    (OUT / 'x11_verify_untrained.json').write_text(json.dumps(X.clean(R), ensure_ascii=False)); log('wrote x11_verify_untrained.json')


def part_qlib(a):
    x, y, Q, xt, yt, Qt, itr, idv, Jinfo = jets()
    keys = [k for k in Q if np.isfinite(Q[k]).all() and np.isfinite(Qt[k]).all() and np.ptp(Q[k][itr]) > 0]
    Qf = {k: np.asarray(Q[k], np.float64) for k in keys}; Qtt = {k: np.asarray(Qt[k], np.float64) for k in keys}; del Q, Qt
    log(f'Q library: {len(keys)} observables; containing "top": {[k for k in keys if "top" in k]}')
    fitter = LeanFitter({k: Qf[k][itr] for k in keys}, {k: Qf[k][idv] for k in keys}, keys, 3000)
    W = np.stack([np.bincount(r_, minlength=len(xt)) for r_ in np.random.default_rng(11).integers(0, len(xt), (B, len(xt)))]).astype(np.float64)
    R = dict(part='qlib', jets=Jinfo, library=dict(n=len(keys), keys=keys))
    from interp.jedi_core import Jedi
    net = Jedi(64); A = net.run(x, keep=('S',))['S']; At = net.run(xt, keep=('S',))['S']; del net
    R['jedi_S'], *_ = compare('Q-lib jedi S', fitter, (A[itr], A[idv], At), Qtt, W, 50, 0, 5000, 0)
    (OUT / 'x11_verify_qlib.json').write_text(json.dumps(X.clean(R), ensure_ascii=False))
    run, _, _ = X.get_model('dsets'); A = run(x)['S']; At = run(xt)['S']
    R['dsets_S'], *_ = compare('Q-lib dsets S', fitter, (A[itr], A[idv], At), Qtt, W, 20, 0, 5000, 0)
    R['seconds'] = time.time() - T0
    (OUT / 'x11_verify_qlib.json').write_text(json.dumps(X.clean(R), ensure_ascii=False)); log('wrote x11_verify_qlib.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--part', choices=['main', 'untrained', 'qlib'], required=True); a = ap.parse_args()
    log('args', vars(a)); dict(main=part_main, untrained=part_untrained, qlib=part_qlib)[a.part](a)
