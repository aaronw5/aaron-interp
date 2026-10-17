"""X4 (follows E18): do the small taggers trained with (deta, dphi) measured about the hardest particle compute the invariant
mass m or sqrt(M0 M2) about their own axis?

Models: /j-jepa-vol/aaron-interp/models/small/{dsets-d64,linf-2b-d32-k8}-n64{-axhard,}/seed{0,1}/best.pt.
Inputs exactly as in training (train/train_small.py): the first 64 particles of the val archive (= the training code's test
set), pT-ordered (re-sorted by pT after any edit, stable), for axis = hardest re-measured about slot 0 with
train_small.recentre_hardest (torch, float32), then (x - shift) / scale from arch.json, padded rows 0, mask = raw pT > 0.
In the stored-axis convention M1 ~ 0, so sqrt(M0 M2) about the axis = m; about the hardest particle M0 M2 = m^2 + |M1|^2.

Stages (cluster, conda python):
  --prep     W and Z jets of the val archive: exclusive two-subjet clustering (kT: d_ij = min(pT_i^2, pT_j^2) dR_ij^2;
             C/A: dR_ij^2; E-scheme four-vectors; pairwise merging until two pseudojets remain, no beam distance) of the 64
             stored particles; z_A = pT fraction of the subjet holding the hardest particle (slot 0). Axis-jump selection:
             kT, z_A in (0.2, 0.8) (both subjets z > 0.2), j = hardest particle of the other subjet, the raise
             pT_j -> (1 + EPS) pT_hardest needs at most FMAX of pT_j; jets of the seed-71 read-out sample excluded. Null:
             a random particle k != j of the same subjet (rng 72) raised by the same dpT, which keeps it below pT_hardest.
             -> $AI_RESULTS/small/x4_cache.npz
  --model M --axis {hardest,stored} --seed s
             test 1: ridge read-outs (E6v Ridge, alpha on the last quarter of the fit half) from the last hidden layer
                     (dsets h3, linf pool) to log m, log sqrt(M0 M2) about the model's axis and their difference; fit / test
                     halves of 20,000 fresh test jets (4,000 per class, seed 71, rng(71) permutation); held-out R^2 with
                     bootstrap CIs (1,000 over test-half jets).
             test 2: axis jump on the selected W / Z jets; read-out change vs dlog m (H0a) and 1/2 dlog(M0 M2) about the
                     model's axis (H0b), raw and through matched calibration edits (offsets about the model's axis scaled by
                     k = exp(prediction) per jet on the unedited jet); null edit; W -> Z, Z -> W with McNemar tests.
             test 3: rigid shift of all real particles (deta + 0.03, 0.06): input change and response.
             test 4: all W, Z jets of the val archive: W -> Z (Z -> W) rate binned by z_A, Wilson intervals,
                     Cochran-Armitage trend test, logistic slope.
             -> $AI_RESULTS/small/x4_<run>_s<seed>.json, x4_<run>_s<seed>.npz (per-jet arrays)
  --combine  cross-model tests (paired bootstrap of test-4 slopes axhard - stored), T15_axhard_mass.png, F8_axis_jump.png,
             x4_summary.json
  AI_PY=conda ./job.sh x4-prep 'python -m interp.x4_axis_models --prep'
  AI_PY=conda ./job.sh x4-ds-h0 'python -m interp.x4_axis_models --model dsets-d64 --axis hardest --seed 0'
  AI_PY=conda ./job.sh x4-combine 'python -m interp.x4_axis_models --combine'
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'train'))
from interp.x2_small_models import (Ridge, DSRunner, TFRunner, run, moments, wilson, mcnemar, ci95, load_net, r2,  # noqa: E402
                                    _table, edit)

DATA = Path(os.environ.get('JETDISTILL_DATA', '/j-jepa-vol/aaron-interp/data'))
SMALL = Path(os.environ.get('AI_SMALL_MODELS', '/j-jepa-vol/aaron-interp/models/small'))
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results'))
OUT = RES / 'small'; SITE = RES / 'site'
CL = ['g', 'q', 'W', 'Z', 't']; W_, Z_ = 2, 3
N = 64; SEED = 71; PER_CLASS = 4000
EPS = 0.005          # pT_j' = (1 + EPS) pT_hardest
FMAX = 0.05          # largest allowed relative raise of pT_j
ZCUT = 0.2
B_EDIT = 2000; B_R2 = 1000
ZBINS = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
MODELS = ('dsets-d64', 'linf-2b-d32-k8'); AXES = ('hardest', 'stored'); SEEDS = (0, 1)


def log(*a):
    print(time.strftime('[%H:%M:%S]'), *a, flush=True)


def run_name(model, axis):
    return f'{model}-n{N}' + ('-axhard' if axis == 'hardest' else '')


# ================================================================================================ data
def archive():
    import h5py
    x = np.load(DATA / 'val_ptetaphi_f16.npy', mmap_mode='r')
    with h5py.File(DATA / 'val_meta.h5', 'r') as f:
        y = f['label'][:].astype(np.int64)
    assert len(x) == len(y)
    return x, y


def rows(xa, idx):
    """raw (pT, deta, dphi) of the first 64 slots of archive rows idx (sorted) as float32 (float16 -> float32, as training)"""
    out = np.empty((len(idx), N, 3), np.float32)
    for i in range(0, len(idx), 20000):
        out[i:i + 20000] = np.asarray(xa[idx[i:i + 20000], :N, :], np.float32)
    return out


def sample_rows(y):
    rng = np.random.default_rng(SEED)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), PER_CLASS, replace=False) for c in range(5)]))


def pt_sort(x):
    o = np.argsort(-x[..., 0], axis=1, kind='stable')
    return np.take_along_axis(x, o[..., None], 1)


def to_inputs(x, axis):
    """raw stored-coordinate particles -> the model's un-normalised inputs: pT-sorted (stable); axis = hardest: offsets about
    slot 0 by train_small.recentre_hardest (torch float32)"""
    import torch
    from train_small import recentre_hardest
    xs = pt_sort(x)
    if axis == 'stored':
        return xs
    t = torch.from_numpy(np.ascontiguousarray(xs))
    return recentre_hardest(t, t[..., 0] > 0).numpy()


def scale_about_axis(x, k, axis):
    """real particles' offsets about the model's axis (slot 0 for hardest, (0, 0) for stored) multiplied by k (per jet)"""
    xs = pt_sort(x).astype(np.float64); real = xs[..., 0] > 0; k = np.asarray(k, np.float64).reshape(-1, 1)
    c = xs[:, :1, 1:3] if axis == 'hardest' else np.zeros((len(xs), 1, 2))
    for f in (1, 2):
        xs[..., f] = np.where(real, c[..., f - 1] + k * (xs[..., f] - c[..., f - 1]), xs[..., f])
    return xs.astype(np.float32)


def axis_moment(xin, chunk=20000):
    """1/2 log(M0 M2) about the origin of the inputs (the model's axis), float64"""
    out = []
    for i in range(0, len(xin), chunk):
        x = np.asarray(xin[i:i + chunk], np.float64); pt = x[..., 0]
        out.append(0.5 * np.log(np.maximum(pt.sum(1) * (pt * (x[..., 1] ** 2 + x[..., 2] ** 2)).sum(1), 1e-300)))
    return np.concatenate(out)


def logm(x, chunk=20000):
    """log of the invariant mass of the (massless) particles, as x2 moments()"""
    return np.concatenate([moments(x[i:i + chunk])['logm'] for i in range(0, len(x), chunk)])


# ================================================================================================ two-subjet clustering
def _kin(P):
    pt = np.hypot(P[..., 0], P[..., 1])
    yy = 0.5 * np.log(np.maximum(P[..., 3] + P[..., 2], 1e-300) / np.maximum(P[..., 3] - P[..., 2], 1e-300))
    return pt, yy, np.arctan2(P[..., 1], P[..., 0])


def _dist(pt_a, y_a, p_a, pt_b, y_b, p_b, alg):
    dp = np.remainder(p_a - p_b + np.pi, 2 * np.pi) - np.pi
    d = (y_a - y_b) ** 2 + dp ** 2
    return d * np.minimum(pt_a, pt_b) ** 2 if alg == 'kt' else d


def cluster2(x, alg='kt', batch=1000):
    """exclusive clustering to two pseudojets. Returns lab (J, 64) int8: 0 = subjet holding slot 0 (hardest particle),
    1 = the other, -1 padding; ptAB (J, 2) subjet pT (A = holding slot 0)."""
    J = len(x); lab = np.full((J, N), -1, np.int8); ptAB = np.zeros((J, 2))
    for s in range(0, J, batch):
        xb = x[s:s + batch].astype(np.float64); B = len(xb)
        pt, eta, phi = xb[..., 0], xb[..., 1], xb[..., 2]; alive = pt > 0
        P = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), pt * np.cosh(eta)], -1)
        rep = np.tile(np.arange(N), (B, 1))
        pc, yc, phc = _kin(P); pc = np.where(alive, pc, 0.0); yc = np.where(alive, yc, 0.0)
        D = _dist(pc[:, :, None], yc[:, :, None], phc[:, :, None], pc[:, None, :], yc[:, None, :], phc[:, None, :], alg)
        D[~(alive[:, :, None] & alive[:, None, :])] = np.inf
        D[:, np.arange(N), np.arange(N)] = np.inf
        while True:
            act = np.flatnonzero(alive.sum(1) > 2)
            if len(act) == 0: break
            f = D[act].reshape(len(act), -1).argmin(1); i, j = np.divmod(f, N)
            P[act, i] += P[act, j]; alive[act, j] = False
            r_ = rep[act]; rep[act] = np.where(r_ == j[:, None], i[:, None], r_)
            D[act, j, :] = np.inf; D[act, :, j] = np.inf
            pi_, yi, phi_i = _kin(P[act, i])
            pa, ya, pha = _kin(P[act])
            d = _dist(pi_[:, None], yi[:, None], phi_i[:, None], pa, ya, pha, alg)
            d[~alive[act]] = np.inf; d[np.arange(len(act)), i] = np.inf
            D[act, i, :] = d; D[act, :, i] = d
        real = xb[..., 0] > 0
        A = rep[:, 0]
        other = np.where(alive & (np.arange(N)[None] != A[:, None]), np.arange(N)[None], N).min(1)
        lb = np.where(rep == A[:, None], 0, 1).astype(np.int8); lb[~real] = -1
        lab[s:s + B] = lb
        pA = _kin(P[np.arange(B), A])[0]
        pB = np.where(other < N, _kin(P[np.arange(B), np.minimum(other, N - 1)])[0], 0.0)
        ptAB[s:s + B] = np.stack([pA, pB], 1)
    return lab, ptAB


def subjet_pt(x, lab):
    """pT of the two subjets (vector sums of the particles labelled 0 / 1)"""
    x = x.astype(np.float64); pt, eta, phi = x[..., 0], x[..., 1], x[..., 2]
    out = []
    for c in (0, 1):
        w = (lab == c).astype(np.float64)
        out.append(np.hypot((w * pt * np.cos(phi)).sum(1), (w * pt * np.sin(phi)).sum(1)))
    return np.stack(out, 1)


def cluster_selftest():
    """two well-separated groups: kT and C/A both return them"""
    x = np.zeros((1, N, 3), np.float32)
    v = [(50, 0.0, 0.0), (30, 0.02, 0.01), (40, 0.4, 0.1), (5, 0.41, 0.12), (3, -0.01, 0.0)]
    for i, r_ in enumerate(sorted(v, key=lambda t: -t[0])): x[0, i] = r_
    ok = {}
    for alg in ('kt', 'ca'):
        lab, ptAB = cluster2(x, alg)
        ok[alg] = dict(lab=lab[0, :5].tolist(), ok=bool(lab[0, :5].tolist() == [0, 1, 0, 1, 0]))
    return ok


def wrap_selftest():
    x = np.zeros((1, N, 3), np.float32); x[0, 0] = (10, 0.1, 0.2); x[0, 1] = (5, -0.1, 0.0)
    xi = to_inputs(x, 'hardest')
    return dict(ok=bool(np.allclose(xi[0, :2], [[10, 0, 0], [5, -0.2, -0.2]], atol=1e-6)), x=xi[0, :2].tolist())


# ================================================================================================ prep
def prep():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    st = dict(cluster=cluster_selftest(), wrap=wrap_selftest()); log('selftests', st)
    assert all(v['ok'] for v in st['cluster'].values()) and st['wrap']['ok']
    xa, y = archive(); samp = sample_rows(y)
    wz = np.flatnonzero((y == W_) | (y == Z_)); X = rows(xa, wz)
    log(f'{len(wz)} W/Z jets loaded [{time.time() - t0:.0f}s]')
    pt = X[..., 0]; unsorted = int((np.diff(pt, axis=1) > 0).any(1).sum())
    labk, ptk = cluster2(X, 'kt'); log(f'kT done [{time.time() - t0:.0f}s]')
    labc, ptc = cluster2(X, 'ca'); log(f'C/A done [{time.time() - t0:.0f}s]')
    zk = ptk[:, 0] / np.maximum(ptk.sum(1), 1e-12); zc = ptc[:, 0] / np.maximum(ptc.sum(1), 1e-12)
    nreal = (pt > 0).sum(1)
    two = (zk > ZCUT) & (zk < 1 - ZCUT) & (nreal >= 2)
    ptB = np.where(labk == 1, pt, -1.0); j = ptB.argmax(1); ptj = ptB[np.arange(len(X)), j]
    has_b = ptj > 0
    new = (1 + EPS) * pt[:, 0]; frac = np.where(has_b, new / np.maximum(ptj, 1e-12) - 1, np.inf)
    in_samp = np.isin(wz, samp)
    cand = two & has_b & (frac <= FMAX) & ~in_samp
    dpt = new - ptj
    rng = np.random.default_rng(72); kk = np.full(len(X), -1)
    for n in np.flatnonzero(cand):
        ok = np.flatnonzero((labk[n] == 1) & (np.arange(N) != j[n]) & (pt[n] > 0) & (pt[n] + dpt[n] < pt[n, 0] * (1 - 1e-3)))
        if len(ok): kk[n] = rng.choice(ok)
    sel = np.flatnonzero(cand & (kk >= 0))
    # same-partition check with C/A: j in the other C/A subjet, C/A z_A in (0.2, 0.8)
    ca_agree = (labc[sel, j[sel]] == 1) & (zc[sel] > ZCUT) & (zc[sel] < 1 - ZCUT)
    Xs = X[sel]; xj = Xs.copy(); xn = Xs.copy(); ar = np.arange(len(sel))
    xj[ar, j[sel], 0] = new[sel].astype(np.float32)
    xn[ar, kk[sel], 0] = (Xs[ar, kk[sel], 0] + dpt[sel]).astype(np.float32)
    zB_new = subjet_pt(xj, labk[sel]); zB_new = zB_new[:, 1] / zB_new.sum(1)
    zA_old = zk[sel]
    dlm_j = logm(xj) - logm(Xs); dlm_n = logm(xn) - logm(Xs)
    info = dict(n_wz=int(len(wz)), unsorted_rows=unsorted, two_prong=int(two.sum()), two_prong_frac=float(two.mean()),
                frac_le_FMAX_among_two=float((frac[two] <= FMAX).mean()), excluded_sample=int((two & (frac <= FMAX) & in_samp).sum()),
                no_null_candidate=int((cand & (kk < 0)).sum()), selected=int(len(sel)),
                selected_W=int((y[wz[sel]] == W_).sum()), selected_Z=int((y[wz[sel]] == Z_).sum()),
                ca_agree=float(ca_agree.mean()), kt_ca_zA_corr=float(np.corrcoef(zk[two], zc[two])[0, 1]),
                raise_frac_quantiles=np.percentile(frac[sel], [0, 10, 50, 90, 100]).tolist(),
                dpt_GeV_quantiles=np.percentile(dpt[sel], [0, 10, 50, 90, 100]).tolist(),
                dlogm_jump_quantiles=np.percentile(dlm_j, [0, 1, 10, 50, 90, 99, 100]).tolist(), dlogm_jump_mean=float(dlm_j.mean()),
                dlogm_null_quantiles=np.percentile(dlm_n, [0, 1, 10, 50, 90, 99, 100]).tolist(), dlogm_null_mean=float(dlm_n.mean()),
                zA_old_quantiles=np.percentile(zA_old, [0, 10, 50, 90, 100]).tolist(), frac_hardest_in_softer=float((zA_old < 0.5).mean()),
                pred_z_mean=float(np.mean(0.5 * np.log(zA_old / zB_new))), eps=EPS, fmax=FMAX, zcut=ZCUT, selftests=st)
    np.savez_compressed(OUT / 'x4_cache.npz', wz=wz, y_wz=y[wz], zA_kt=zk, zA_ca=zc, sample=samp, sel=sel, j=j[sel], k=kk[sel],
                        dpt=dpt[sel].astype(np.float64), new=new[sel].astype(np.float64), frac=frac[sel], lab_sel=labk[sel],
                        zA_old=zA_old, zB_new=zB_new, ca_agree=ca_agree)
    (OUT / 'x4_prep.json').write_text(json.dumps(info, indent=1))
    log('prep', json.dumps({k: v for k, v in info.items() if k != 'selftests'}), f'[{time.time() - t0:.0f}s]')


def edited(Xs, C):
    ar = np.arange(len(Xs)); xj = Xs.copy(); xn = Xs.copy()
    xj[ar, C['j'], 0] = C['new'].astype(np.float32)
    xn[ar, C['k'], 0] = (Xs[ar, C['k'], 0] + C['dpt']).astype(np.float32)
    return xj, xn


# ================================================================================================ statistics
def bstat(a, bi):
    from scipy.stats import norm
    b = a[bi].mean(1); se = float(b.std()); m = float(a.mean())
    return dict(mean=m, ci=ci95(b), se=se, p=float(2 * norm.sf(abs(m) / se)) if se > 0 else None,
                p_boot=float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()))))


def ratio_stat(a, b, bi):
    return dict(ratio=float(a.mean() / b.mean()), ci=ci95(a[bi].mean(1) / b[bi].mean(1)))


def slope(xv, yv):
    w = np.linalg.lstsq(np.c_[xv, np.ones(len(xv))], yv, rcond=None)[0]
    return dict(slope=float(w[0]), intercept=float(w[1]), r2=float(np.corrcoef(xv, yv)[0, 1] ** 2))


def logistic(xv, yv, it=30):
    """unpenalised logistic regression y ~ 1 + x by IRLS -> (b0, b1, se_b1)"""
    X = np.c_[np.ones(len(xv)), xv]; b = np.zeros(2)
    for _ in range(it):
        p = 1 / (1 + np.exp(-X @ b)); w = p * (1 - p) + 1e-12
        H = X.T @ (X * w[:, None]); g = X.T @ (yv - p); step = np.linalg.solve(H, g); b += step
        if np.abs(step).max() < 1e-10: break
    p = 1 / (1 + np.exp(-X @ b)); H = X.T @ (X * (p * (1 - p))[:, None])
    return float(b[0]), float(b[1]), float(np.sqrt(np.linalg.inv(H)[1, 1]))


def cochran_armitage(k, n, t):
    from scipy.stats import norm
    k, n, t = map(np.asarray, (k, n, t)); Nn = n.sum(); pb = k.sum() / Nn
    T = (t * (k - n * pb)).sum(); V = pb * (1 - pb) * ((n * t * t).sum() - (n * t).sum() ** 2 / Nn)
    z = float(T / np.sqrt(V)) if V > 0 else 0.0
    return dict(z=z, p=float(2 * norm.sf(abs(z))))


def binned(flag, zv, edges=ZBINS):
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m_ = (zv >= a) & (zv < b) if b < 1 else (zv >= a) & (zv <= b)
        k, n = int(flag[m_].sum()), int(m_.sum())
        out.append(dict(lo=a, hi=b, n=n, k=k, rate=k / n if n else None, wilson=wilson(k, n)))
    good = [e for e in out if e['n'] > 0]
    ca = cochran_armitage([e['k'] for e in good], [e['n'] for e in good], [(e['lo'] + e['hi']) / 2 for e in good])
    b0, b1, se = logistic(zv, flag.astype(np.float64))
    return dict(bins=out, trend=ca, logistic=dict(b0=b0, b1=b1, se=se, ci=[b1 - 1.96 * se, b1 + 1.96 * se]))


# ================================================================================================ model stage
def model_stage(model, axis, seed):
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
    t0 = time.time(); name = run_name(model, axis); run_dir = SMALL / name / f'seed{seed}'
    net, arch = load_net(run_dir); assert arch['inputs'].get('axis', 'stored') == axis, (arch['inputs'].get('axis'), axis)
    runner = DSRunner(net) if arch['spec']['arch'] == 'dsets' else TFRunner(net); last = runner.last
    want = ('logits', last)
    fwd = lambda xin: run(runner, xin, arch, want=want)
    log(f'X4 {name} seed {seed}: axis {axis}, norm {arch["inputs"].get("norm_file", "norm_stats.json (stored)")}, last layer {last}')
    C = dict(np.load(OUT / 'x4_cache.npz'))
    xa, y = archive(); samp = C['sample']; probs_ref = np.load(run_dir / 'test_probs.npy', mmap_mode='r')
    res = dict(model=model, axis=axis, seed=seed, run=str(run_dir), last=last, eps=EPS, fmax=FMAX)

    # ---------------------------------------------------------------- test 1: read-outs on natural jets
    Xn = rows(xa, samp); yn = y[samp]; Xi = to_inputs(Xn, axis)
    R0 = fwd(Xi); c0 = R0['logits'].argmax(1); H = R0[last]
    e = np.exp(R0['logits'] - R0['logits'].max(1, keepdims=True)); p = e / e.sum(1, keepdims=True)
    P_ = np.asarray(probs_ref[samp])
    res['check'] = dict(acc_sample=float((c0 == yn).mean()), probs_maxabs_diff=float(np.abs(P_ - p).max()),
                        argmax_agree=float((P_.argmax(1) == c0).mean()), n=int(len(samp)),
                        sorted_identity=bool(np.array_equal(pt_sort(Xn), Xn)),
                        slot0_offset_max=float(np.abs(Xi[:, 0, 1:]).max()))
    log('check', res['check'])
    tm = logm(Xn); tb = axis_moment(Xi); rho = tb - tm
    J = len(Xn); perm = np.random.default_rng(SEED).permutation(J); tr, te = np.sort(perm[:J // 2]), np.sort(perm[J // 2:])
    trf, trv = tr[: len(tr) * 3 // 4], tr[len(tr) * 3 // 4:]
    readers = dict(logm=Ridge(H, tm, trf, trv, tr), axmom=Ridge(H, tb, trf, trv, tr), rho=Ridge(H, rho, trf, trv, tr))
    tgt = dict(logm=tm, axmom=tb, rho=rho)
    bi = np.random.default_rng(11).integers(0, len(te), size=(B_R2, len(te)))
    T1 = dict(n_fit=int(len(tr)), n_test=int(len(te)), corr_logm_axmom=float(np.corrcoef(tm, tb)[0, 1]),
              corr_logm_axmom_WZ=float(np.corrcoef(tm[np.isin(yn, [W_, Z_])], tb[np.isin(yn, [W_, Z_])])[0, 1]),
              rho_quantiles=np.percentile(rho, [10, 50, 90, 99]).tolist(), r2={}, alpha={}, se={})
    sse_b = {}
    for kname, rd in readers.items():
        t = tgt[kname][te]; pr = rd(H[te]); err = (pr - t) ** 2
        sst = ((t[bi] - t[bi].mean(1, keepdims=True)) ** 2).sum(1); rb = 1 - err[bi].sum(1) / sst; sse_b[kname] = rb
        T1['r2'][kname] = dict(r2=r2(pr, t), ci=ci95(rb)); T1['alpha'][kname] = rd.alpha
    d = sse_b['logm'] - sse_b['axmom']
    T1['r2_logm_minus_axmom'] = dict(diff=T1['r2']['logm']['r2'] - T1['r2']['axmom']['r2'], ci=ci95(d),
                                     p_boot=float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))))
    res['test1'] = T1
    log('T1', json.dumps(T1))

    # ---------------------------------------------------------------- test 3: rigid shift
    T3 = {}
    for dl in (0.03, 0.06):
        Xsh = edit(Xn, d_eta=dl); Xsi = to_inputs(Xsh, axis); Rs = fwd(Xsi); cs = Rs['logits'].argmax(1)
        obs = readers['logm'](Rs[last][te]) - readers['logm'](H[te]); pb = (axis_moment(Xsi) - tb)[te]
        T3[f'eta{dl}'] = dict(input_maxabs_change=float(np.abs(Xsi - Xi).max()), calls_changed=float((cs != c0).mean()),
                              W_to_Z=float((cs[yn == W_] == Z_).mean()), W_to_Z_unedited=float((c0[yn == W_] == Z_).mean()),
                              dreadout_mean=float(obs.mean()), pred_axmom=float(pb.mean()),
                              dlogm_true_absmax=float(np.abs(logm(Xsh) - tm).max()),
                              logit_maxabs_change=float(np.abs(Rs['logits'] - R0['logits']).max()))
    res['test3'] = T3; log('T3', json.dumps(T3))

    # ---------------------------------------------------------------- test 2: axis jump
    sel_rows = C['wz'][C['sel']]; ys = C['y_wz'][C['sel']]; Xs = rows(xa, sel_rows)
    xj, xn = edited(Xs, C)
    Xs_i, xj_i, xn_i = to_inputs(Xs, axis), to_inputs(xj, axis), to_inputs(xn, axis)
    hard_i = to_inputs(Xs, 'hardest'); hard_j = to_inputs(xj, 'hardest'); hard_n = to_inputs(xn, 'hardest')
    lm0 = logm(Xs)
    pred = dict(jump=dict(a=logm(xj) - lm0, b=axis_moment(xj_i) - axis_moment(Xs_i),
                          b_hardest=axis_moment(hard_j) - axis_moment(hard_i)),
                null=dict(a=logm(xn) - lm0, b=axis_moment(xn_i) - axis_moment(Xs_i),
                          b_hardest=axis_moment(hard_n) - axis_moment(hard_i)))
    pred_z = 0.5 * np.log(C['zA_old'] / C['zB_new'])
    cal_a = scale_about_axis(Xs, np.exp(pred['jump']['a']), axis); cal_b = scale_about_axis(Xs, np.exp(pred['jump']['b']), axis)
    RR = {nm: fwd(to_inputs(v, axis)) for nm, v in (('cal_a', cal_a), ('cal_b', cal_b))}
    RR['nat'] = fwd(Xs_i); RR['jump'] = fwd(xj_i); RR['null'] = fwd(xn_i)
    cal_check = dict(cal_b_axmom_vs_pred_maxabs=float(np.abs(axis_moment(to_inputs(cal_b, axis)) - axis_moment(Xs_i) - pred['jump']['b']).max()),
                     cal_b_dlogm_mean=float((logm(cal_b) - lm0).mean()), cal_b_pred_mean=float(pred['jump']['b'].mean()))
    bi = np.random.default_rng(13).integers(0, len(Xs), size=(B_EDIT, len(Xs)))
    rd = {k: readers[k] for k in ('logm', 'axmom')}
    D = {r_: {nm: rd[r_](RR[nm][last]) - rd[r_](RR['nat'][last]) for nm in ('jump', 'null', 'cal_a', 'cal_b')} for r_ in rd}
    T2 = dict(n=int(len(Xs)), n_W=int((ys == W_).sum()), n_Z=int((ys == Z_).sum()), cal_check=cal_check,
              corr_pred_b_pred_z=float(np.corrcoef(pred['jump']['b'], pred_z)[0, 1]),
              pred=dict(jump_a=bstat(pred['jump']['a'], bi), jump_b=bstat(pred['jump']['b'], bi),
                        jump_b_hardest=bstat(pred['jump']['b_hardest'], bi), jump_z=bstat(pred_z, bi),
                        null_a=bstat(pred['null']['a'], bi), null_b=bstat(pred['null']['b'], bi)), readout={})
    subsets = dict(all=np.ones(len(Xs), bool), ca_agree=C['ca_agree'].astype(bool), frac_le_0p02=C['frac'] <= 0.02,
                   hardest_in_harder=C['zA_old'] >= 0.5, hardest_in_softer=C['zA_old'] < 0.5, W=ys == W_, Z=ys == Z_)
    for r_ in rd:
        out = {}
        for sn, m_ in subsets.items():
            if m_.sum() < 30: continue
            b2 = np.random.default_rng(14).integers(0, int(m_.sum()), size=(B_EDIT, int(m_.sum())))
            o = D[r_]['jump'][m_]; pa = pred['jump']['a'][m_]; pb = pred['jump']['b'][m_]
            ca_, cb_ = D[r_]['cal_a'][m_], D[r_]['cal_b'][m_]
            e_ = dict(n=int(m_.sum()), obs=bstat(o, b2), pred_a=float(pa.mean()), pred_b=float(pb.mean()),
                      cal_a=bstat(ca_, b2), cal_b=bstat(cb_, b2),
                      H0a_raw=bstat(o - pa, b2), H0b_raw=bstat(o - pb, b2), H0a_cal=bstat(o - ca_, b2), H0b_cal=bstat(o - cb_, b2),
                      ratio_obs_pred_b=ratio_stat(o, pb, b2) if abs(pb.mean()) > 1e-4 else None,
                      ratio_obs_cal_b=ratio_stat(o, cb_, b2) if abs(cb_.mean()) > 1e-4 else None,
                      perjet_obs_on_pred_b=slope(pb, o) if pb.std() > 1e-6 else None,
                      perjet_obs_on_pred_b_hardest=slope(pred['jump']['b_hardest'][m_], o),
                      perjet_obs_on_cal_b=slope(cb_, o) if cb_.std() > 1e-6 else None,
                      null_obs=bstat(D[r_]['null'][m_], b2), null_pred_a=float(pred['null']['a'][m_].mean()),
                      null_pred_b=float(pred['null']['b'][m_].mean()))
            out[sn] = e_
        T2['readout'][r_] = out
        a_ = out['all']
        log(f'T2 reader {r_}: obs {a_["obs"]["mean"]:+.4f} {a_["obs"]["ci"]} pred_a {a_["pred_a"]:+.4f} pred_b {a_["pred_b"]:+.4f} '
            f'cal_a {a_["cal_a"]["mean"]:+.4f} cal_b {a_["cal_b"]["mean"]:+.4f} | H0a cal p {a_["H0a_cal"]["p"]:.2e} H0b cal p {a_["H0b_cal"]["p"]:.2e} '
            f'| null {a_["null_obs"]["mean"]:+.4f} | slope {a_["perjet_obs_on_pred_b"]}')
    cN, cJ, cU, cA, cB = (RR[k]['logits'].argmax(1) for k in ('nat', 'jump', 'null', 'cal_a', 'cal_b'))
    calls = {}
    for tru, to, nm in ((W_, Z_, 'W_to_Z'), (Z_, W_, 'Z_to_W')):
        m_ = ys == tru; a0, aj, an = cN[m_] == to, cJ[m_] == to, cU[m_] == to; n_ = int(m_.sum())
        calls[nm] = dict(n=n_, nat=float(a0.mean()), jump=float(aj.mean()), null=float(an.mean()),
                         nat_ci=wilson(int(a0.sum()), n_), jump_ci=wilson(int(aj.sum()), n_), null_ci=wilson(int(an.sum()), n_),
                         mcnemar_jump_vs_null=mcnemar(an, aj), mcnemar_jump_vs_nat=mcnemar(a0, aj), mcnemar_null_vs_nat=mcnemar(a0, an))
        for sn in ('hardest_in_harder', 'hardest_in_softer'):
            mm = m_ & subsets[sn]; b0_, bj_, bn_ = cN[mm] == to, cJ[mm] == to, cU[mm] == to
            bb_ = cB[mm] == to
            calls[nm][sn] = dict(n=int(mm.sum()), nat=float(b0_.mean()), jump=float(bj_.mean()), null=float(bn_.mean()),
                                 cal_a=float((cA[mm] == to).mean()), cal_b=float(bb_.mean()),
                                 mcnemar_jump_vs_null=mcnemar(bn_, bj_), mcnemar_jump_vs_cal_b=mcnemar(bb_, bj_))
        calls[nm].update(cal_a=float((cA[m_] == to).mean()), cal_b=float((cB[m_] == to).mean()))
    calls['changed'] = dict(jump=float((cJ != cN).mean()), null=float((cU != cN).mean()), cal_a=float((cA != cN).mean()), cal_b=float((cB != cN).mean()),
                            mcnemar=mcnemar(cU != cN, cJ != cN))
    T2['calls'] = calls; res['test2'] = T2
    log('T2 calls', json.dumps(calls))

    # ---------------------------------------------------------------- test 4: W -> Z vs z_A on all W / Z jets
    cw, rho_w, dmax, agree = [], [], 0.0, 0
    for i in range(0, len(C['wz']), 15000):       # chunks keep the pod under 2 GiB
        ii = C['wz'][i:i + 15000]; Xw = rows(xa, ii); Rw = fwd(to_inputs(Xw, axis)); c_ = Rw['logits'].argmax(1)
        e = np.exp(Rw['logits'] - Rw['logits'].max(1, keepdims=True)); pw = e / e.sum(1, keepdims=True); Pw = np.asarray(probs_ref[ii])
        dmax = max(dmax, float(np.abs(Pw - pw).max())); agree += int((Pw.argmax(1) == c_).sum()); cw.append(c_)
        rho_w.append(axis_moment(to_inputs(Xw, 'hardest')) - logm(Xw)); del Xw, Rw
    cw = np.concatenate(cw); rho_w = np.concatenate(rho_w)
    res['check']['wz_probs_maxabs_diff'] = dmax; res['check']['wz_argmax_agree'] = agree / len(cw)
    yw = C['y_wz']; zA = C['zA_kt']; T4 = {}
    for tru, to, nm in ((W_, Z_, 'W_to_Z'), (Z_, W_, 'Z_to_W')):
        m_ = yw == tru; flag = cw[m_] == to
        T4[nm] = dict(n=int(m_.sum()), rate=float(flag.mean()), by_zA=binned(flag, zA[m_]),
                      by_zA_two_prong=binned(flag[(zA[m_] > ZCUT) & (zA[m_] < 1 - ZCUT)], zA[m_][(zA[m_] > ZCUT) & (zA[m_] < 1 - ZCUT)]))
        q = np.percentile(rho_w[m_], [0, 25, 50, 75, 100]); q[-1] += 1e-9
        T4[nm]['by_rho_quartile'] = binned(flag, rho_w[m_], edges=tuple(q))
        log(f'T4 {nm}: rate {flag.mean():.4f}; by z_A ' + ' '.join(f'{b["rate"]:.3f}' if b['rate'] is not None else 'n/a' for b in T4[nm]['by_zA']['bins'])
            + f'; trend z {T4[nm]["by_zA"]["trend"]["z"]:+.2f} p {T4[nm]["by_zA"]["trend"]["p"]:.2e}; logistic b1 {T4[nm]["by_zA"]["logistic"]["b1"]:+.3f}')
    res['test4'] = T4
    res['elapsed_s'] = float(time.time() - t0)
    tag = f'x4_{name}_s{seed}'
    (OUT / f'{tag}.json').write_text(json.dumps(res, indent=1))
    np.savez_compressed(OUT / f'{tag}.npz', obs=D['logm']['jump'], obs_null=D['logm']['null'], cal_b=D['logm']['cal_b'],
                        cal_a=D['logm']['cal_a'], obs_axmom=D['axmom']['jump'], pred_a=pred['jump']['a'], pred_b=pred['jump']['b'],
                        pred_b_hardest=pred['jump']['b_hardest'], pred_z=pred_z, calls_wz=cw.astype(np.int8),
                        cN=cN.astype(np.int8), cJ=cJ.astype(np.int8), cU=cU.astype(np.int8), rho_wz=rho_w)
    log(f'wrote {OUT / tag}.json/.npz [{time.time() - t0:.0f}s]')


# ================================================================================================ combine
SHORT = {'dsets-d64': 'Deep Sets', 'linf-2b-d32-k8': 'Linformer'}


def combine():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    C = dict(np.load(OUT / 'x4_cache.npz')); PR = json.loads((OUT / 'x4_prep.json').read_text())
    R, NZ = {}, {}
    for m in MODELS:
        for a in AXES:
            for s in SEEDS:
                tag = f'x4_{run_name(m, a)}_s{s}'
                R[m, a, s] = json.loads((OUT / f'{tag}.json').read_text()); NZ[m, a, s] = dict(np.load(OUT / f'{tag}.npz'))
    # cross-model: test-4 logistic slope axhard - stored (paired bootstrap over jets, same seed index)
    yw, zA = C['y_wz'], C['zA_kt']; cross = {}
    rng = np.random.default_rng(15)
    for m in MODELS:
        for s in SEEDS:
            for tru, to, nm in ((W_, Z_, 'W_to_Z'), (Z_, W_, 'Z_to_W')):
                mm = yw == tru; z = zA[mm]; fh = (NZ[m, 'hardest', s]['calls_wz'][mm] == to).astype(float)
                fs = (NZ[m, 'stored', s]['calls_wz'][mm] == to).astype(float)
                d0 = logistic(z, fh)[1] - logistic(z, fs)[1]; bs = []
                for _ in range(300):
                    ii = rng.integers(0, len(z), len(z)); bs.append(logistic(z[ii], fh[ii])[1] - logistic(z[ii], fs[ii])[1])
                bs = np.array(bs)
                lo = (z < 0.3); hi = (z >= 0.7)
                cross[f'{m}_s{s}_{nm}'] = dict(slope_diff=float(d0), ci=ci95(bs), p=float(2 * min((bs <= 0).mean(), (bs >= 0).mean())),
                                               axhard_lowz=float(fh[lo].mean()), axhard_highz=float(fh[hi].mean()),
                                               stored_lowz=float(fs[lo].mean()), stored_highz=float(fs[hi].mean()),
                                               n_lowz=int(lo.sum()), n_highz=int(hi.sum()))
    # per-jet slopes of the read-out change on 1/2 dlog(M0 M2) about the hardest particle (a jet property, the same x for both
    # families), with the read-out's own response to real changes of that size (matched calibration) for comparison
    from scipy.stats import norm
    SL = {}
    for (m, a, s), z_ in NZ.items():
        n = len(z_['obs']); bi = np.random.default_rng(16).integers(0, n, size=(B_EDIT, n)); xv = z_['pred_b_hardest']
        xb = xv[bi]; xc = xb - xb.mean(1, keepdims=True); sxx = (xc ** 2).sum(1)
        sl = lambda yv: ((xc * (yv[bi] - yv[bi].mean(1, keepdims=True))).sum(1) / sxx)
        pt_ = lambda yv: float(np.polyfit(xv, yv, 1)[0])
        e = {}
        for k, yv in (('obs', z_['obs']), ('obs_axmom', z_['obs_axmom']), ('cal_a', z_['cal_a']), ('cal_b', z_['cal_b']),
                      ('null', z_['obs_null']), ('dlogm', z_['pred_a'])):
            bb = sl(yv); e[k] = dict(slope=pt_(yv), ci=ci95(bb), se=float(bb.std()), r2=float(np.corrcoef(xv, yv)[0, 1] ** 2)); e[k]['_b'] = bb
        def test(k1, k2):
            d = e[k1]['_b'] - e[k2]['_b']; d0 = e[k1]['slope'] - e[k2]['slope']
            return dict(diff=float(d0), ci=ci95(d), p=float(2 * norm.sf(abs(d0) / d.std())))
        e['H0a'] = test('obs', 'cal_a'); e['H0b'] = test('obs', 'cal_b')
        e['p_slope0'] = float(2 * norm.sf(abs(e['obs']['slope']) / e['obs']['se']))
        e['p_slope1'] = float(2 * norm.sf(abs(e['obs']['slope'] - 1) / e['obs']['se']))
        fr = e['obs']['_b'] / e['cal_b']['_b']; e['frac_of_cal_b'] = dict(ratio=e['obs']['slope'] / e['cal_b']['slope'], ci=ci95(fr))
        for k in ('obs', 'obs_axmom', 'cal_a', 'cal_b', 'null', 'dlogm'): e[k].pop('_b')
        SL[f'{m}|{a}|s{s}'] = e
        log(f'slopes {m} {a} s{s}: obs {e["obs"]["slope"]:.3f} {e["obs"]["ci"]} axmom {e["obs_axmom"]["slope"]:.3f} cal_b {e["cal_b"]["slope"]:.3f} '
            f'cal_a {e["cal_a"]["slope"]:.3f} frac {e["frac_of_cal_b"]} H0a p {e["H0a"]["p"]:.2e} H0b p {e["H0b"]["p"]:.2e}')
    out = dict(prep=PR, cross_test4=cross, slopes=SL, models={f'{m}|{a}|s{s}': R[m, a, s] for (m, a, s) in R})
    (OUT / 'x4_summary.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(cross, indent=1))

    # ---------------------------------------------------------------- T15 table
    f3 = lambda v: f'{v:+.3f}'; pc = lambda v: f'{100 * v:.1f}'; f2 = lambda v: f'{v:.2f}'
    def pfmt(p):
        if p is None: return 'n/a'
        return '<1e-300' if p < 1e-300 else (f'{p:.1e}' if p < 0.01 else f'{p:.2f}')
    cols = ['model (seed)', 'axis', 'acc\n(20k)', 'R² read-out\nlog m', 'R² read-out\nlog √(M0M2) axis',
            'slope of Δ log m\nread-out on\n½Δlog(M0M2) [CI]', 'slope of read-out\nresponse to a real\nchange of that size',
            'p H0a\n(inv. m)', 'p H0b\n(√(M0M2) axis)', 'mean Δ read-out\n(pred. ½Δlog M0M2)\nhardest in harder /\nin softer prong',
            'W→Z %, hardest in\nharder prong: nat /\njump / null / real\nchange (matched)', 'McNemar p\njump vs null', 'W→Z % natural\nz_A<0.3 / ≥0.7']
    rows_ = []
    for m in MODELS:
        for a in AXES:
            for s in SEEDS:
                r_ = R[m, a, s]; T1 = r_['test1']; T2 = r_['test2']; Rl = T2['readout']['logm']; cw = T2['calls']['W_to_Z']
                e = SL[f'{m}|{a}|s{s}']; xk = cross[f'{m}_s{s}_W_to_Z']; hh, hs = Rl['hardest_in_harder'], Rl['hardest_in_softer']
                lowz, highz = (xk['axhard_lowz'], xk['axhard_highz']) if a == 'hardest' else (xk['stored_lowz'], xk['stored_highz'])
                if a == 'hardest':
                    cal = f'{f2(e["cal_b"]["slope"])}\n(obs / cal {f2(e["frac_of_cal_b"]["ratio"])}\n[{f2(e["frac_of_cal_b"]["ci"][0])}, {f2(e["frac_of_cal_b"]["ci"][1])}])'
                    pa, pb = pfmt(e['H0a']['p']), pfmt(e['H0b']['p'])
                else:
                    cal = 'both ≈ 0\n(axis does\nnot move)'; pa = pb = f'slope = 0:\n{pfmt(e["p_slope0"])}'
                rows_.append([f'{SHORT[m]} (s{s})', 'hardest\nparticle' if a == 'hardest' else 'stored\n(centroid)',
                              f'{r_["check"]["acc_sample"]:.3f}',
                              f'{T1["r2"]["logm"]["r2"]:.3f}\n[{T1["r2"]["logm"]["ci"][0]:.3f}, {T1["r2"]["logm"]["ci"][1]:.3f}]',
                              f'{T1["r2"]["axmom"]["r2"]:.3f}\n[{T1["r2"]["axmom"]["ci"][0]:.3f}, {T1["r2"]["axmom"]["ci"][1]:.3f}]',
                              f'{f2(e["obs"]["slope"])}\n[{f2(e["obs"]["ci"][0])}, {f2(e["obs"]["ci"][1])}]', cal, pa, pb,
                              f'{f3(hh["obs"]["mean"])} ({f3(hh["pred_b"])})\n{f3(hs["obs"]["mean"])} ({f3(hs["pred_b"])})',
                              f'{pc(cw["hardest_in_harder"]["nat"])} / {pc(cw["hardest_in_harder"]["jump"])}\n/ {pc(cw["hardest_in_harder"]["null"])} / {pc(cw["hardest_in_harder"]["cal_b"])}',
                              pfmt(cw['hardest_in_harder']['mcnemar_jump_vs_null']['p']),
                              f'{pc(lowz)} / {pc(highz)}'])
    n_sel = PR['selected']
    note = (f'Read-outs: ridge from the last hidden layer (Deep Sets h3, Linformer pool), fitted on 10,000 natural test jets (seed 71), R² on the other 10,000 (bootstrap 95% CI). '
            f'Axis jump: {n_sel} W/Z jets with two kT subjets (z > 0.2);\nthe hardest particle of the other subjet raised to 1.005 × the hardest pT (raise ≤ 5% of its pT; '
            f'median Δlog m {PR["dlogm_jump_quantiles"][3]:+.4f}); inputs recomputed about the new hardest particle. Slope: per-jet regression of the change of the log m read-out on '
            '½Δlog(M0M2) about the hardest particle\n(invariant mass predicts 0, √(M0M2) about the axis predicts the slope of the response of the read-out to a real change of the same size: '
            'offsets about the axis scaled by exp(prediction) per jet). p H0a / H0b: bootstrap (2,000 over jets) of observed − calibrated slope.\n'
            'Mean Δ: jets whose hardest particle sits in the harder (z_A ≥ 0.5) or softer prong. Null: a random other particle of the same subjet raised by the same ΔpT; real change: the matched calibration edit (W jets, n = 552). '
            'Last column: all W test jets, W called Z, z_A = pT fraction of the subjet holding the hardest particle.')
    _table('T15_axhard_mass.png', 'X4: invariant mass or √(M0·M2) about the input axis: read-outs and the axis-jump edit', cols, rows_,
           [1.05, 0.75, 0.48, 0.95, 1.15, 1.15, 1.2, 0.75, 0.9, 1.4, 1.2, 0.8, 1.0], note=note, fs=7.2, line_in=0.14,
           vlines=(2, 5, 10))

    # ---------------------------------------------------------------- F8 figure
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 8.6), dpi=150, sharex=True)
    pbh = NZ[MODELS[0], 'hardest', 0]['pred_b_hardest']; lo_, hi_ = np.percentile(pbh, [1, 99]); edges = np.linspace(lo_, hi_, 11)
    col = {0: '#2a6f97', 1: '#c8553d'}
    for ri, m in enumerate(MODELS):
        for ci_, a in enumerate(AXES):
            ax = axs[ri, ci_]
            for s in SEEDS:
                z_ = NZ[m, a, s]; xv = z_['pred_b_hardest']; yv = z_['obs']
                if s == 0: ax.scatter(xv, yv, s=2, alpha=0.12, color='#888', rasterized=True, lw=0)
                mid, mu, se, cb = [], [], [], []
                for e0, e1 in zip(edges[:-1], edges[1:]):
                    mm = (xv >= e0) & (xv < e1)
                    if mm.sum() >= 20:
                        mid.append(0.5 * (e0 + e1)); mu.append(yv[mm].mean()); se.append(yv[mm].std() / np.sqrt(mm.sum())); cb.append(z_['cal_b'][mm].mean())
                ax.errorbar(mid, mu, yerr=1.96 * np.array(se), fmt='o', ms=4, color=col[s], label=f'observed, binned mean (seed {s})', zorder=4)
                if s == 0:
                    ma = [z_['obs_axmom'][(xv >= e0) & (xv < e1)].mean() for e0, e1 in zip(edges[:-1], edges[1:]) if ((xv >= e0) & (xv < e1)).sum() >= 20]
                    ax.plot(mid, ma, '^', mfc='none', ms=6, color='#6a4c93', label='read-out fitted to log √(M0M2) about the axis, binned mean (seed 0)', zorder=4)
                if a == 'hardest':
                    ax.plot(mid, cb, '--', color=col[s], lw=1.2, label=f'read-out response to a real change of ½Δlog(M0M2) (seed {s})', zorder=3)
            xx = np.array([lo_, hi_])
            ax.plot(xx, xx, '-', color='k', lw=1.2, label='√(M0M2) about the hardest particle: Δ = ½Δlog(M0M2)')
            ax.axhline(float(np.mean(NZ[m, a, 0]['pred_a'])), color='#3a7d44', lw=1.6, label='invariant mass: Δ = Δlog m (mean)')
            ax.set_xlim(lo_, hi_); ax.set_ylim(min(lo_, -0.1) * 1.05, max(hi_, 0.1) * 1.05)
            ax.grid(alpha=0.25)
            ax.set_title(f'{SHORT[m]}, trained with axis = {"hardest particle" if a == "hardest" else "stored (centroid)"}', fontsize=10)
            if ri == 1: ax.set_xlabel('½Δlog(M0·M2) about the hardest particle (per jet)')
            if ci_ == 0: ax.set_ylabel('change of the log m read-out')
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f'Axis jump on {n_sel} W/Z jets: the hardest particle moves to the other prong (Δlog m ≈ {PR["dlogm_jump_mean"]:+.3f})', fontsize=11)
    fig.tight_layout(rect=(0, 0.11, 1, 0.96))
    SITE.mkdir(parents=True, exist_ok=True); fig.savefig(SITE / 'F8_axis_jump.png', bbox_inches='tight'); print('wrote', SITE / 'F8_axis_jump.png')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--prep', action='store_true'); ap.add_argument('--combine', action='store_true')
    ap.add_argument('--model'); ap.add_argument('--axis', choices=AXES); ap.add_argument('--seed', type=int)
    a = ap.parse_args()
    if a.prep: prep()
    elif a.combine: combine()
    else: model_stage(a.model, a.axis, a.seed)


if __name__ == '__main__':
    main()
