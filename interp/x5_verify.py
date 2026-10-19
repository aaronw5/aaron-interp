"""X5 verification (re-runs elements (a) and (d) of interp.x5_dsets_sparsity on a fresh sample, adds controls).

    python -m interp.x5_verify --part A        (float s0-2 + HGQ beta 3e-4 s0,1,2b; one subprocess per model)
    python -m interp.x5_verify --part B        (L1 s0-2 + HGQ beta 1e-8 s0-2)
    python -m interp.x5_verify --combine       (reads the per-model files, writes x5_verify.json)

Sample: interp.jedi_core.sample(per_class=2000, seed=4242) (X5 used seed 311).
(a) as X5: live phi1 units (unit -> mean over real particles, calls changed >= 1 %), pure share on live units, P_nz on units with
    any nonzero weight vs the X6v column-permutation null, here 2,000 draws with a new RNG; also at thresholds 0.8 / 0.95, and for
    HGQ an iq-aware version (term spread from sd of the phi1 input quantizer output instead of the raw input); phi1 iq zero-bit
    fraction per feature.
(d) as X5 (translate +0.06 rad deta, rotation with equal mean displacement, random-direction translation, matched angular scale,
    ridge read-out of log m on h3, calibrated ratio), plus: translation by -0.06 rad deta (same predicted M2 change);
    for HGQ, the fraction of quantizer inputs that overflow (|iq(x) - x| > 2^-f, wrap mode) at every quantized layer per edit,
    and W -> Z restricted to true-W jets with no overflow anywhere under the unedited and translated inputs.
Writes $AI_RESULTS/jedi/x5_verify_<label>_s<seed>.json and x5_verify.json.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np

from interp.x5_dsets_sparsity import FloatDS, HgqDS, run, moments, edit, rotate, Ridge, mcnemar, wilson, ci95, holm, clean

OUT = Path(os.environ.get('AI_RESULTS', Path(__file__).resolve().parents[1] / 'results')) / 'jedi'; OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)
MS = '/j-jepa-vol/aaron-interp/models/small/'
MODELS = {
    'float': [(s, MS + f'dsets-d64-n64/seed{s}') for s in (0, 1, 2)],
    'l1': [(s, MS + f'dsets-d64-n64-l1_3e-2/seed{s}') for s in (0, 1, 2)],
    'hgq': [(0, MS + 'dsets-d64-n64-hgq_3e-4/seed0'), (1, MS + 'dsets-d64-n64-hgq_3e-4/seed1'), (2, MS + 'dsets-d64-n64-hgq_3e-4/seed2b')],
    'hgq_acc': [(s, MS + f'dsets-d64-n64-hgq_1e-8/seed{s}') for s in (0, 1, 2)],
}
PARTS = {'A': ('float', 'hgq'), 'B': ('l1', 'hgq_acc')}
W_, Z_ = 2, 3; DELTA = 0.06; NB = 1000; NNULL = 2000; SEED = 4242


def types(al, sd, units, thr):
    w2 = (np.abs(al[:, units]) * sd[:, None]) ** 2; tot = w2.sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        spt = w2[0] / tot; sang = (w2[1] + w2[2]) / tot
    return np.where(tot == 0, 'constant', np.where(spt >= thr, 'ramp', np.where(sang >= thr, 'angular', 'mixed'))), spt


def pure(al, sd, units, thr):
    t, _ = types(al, sd, units, thr); return float(np.isin(t, ['ramp', 'angular']).mean()) if len(units) else np.nan


def null_test(al, sd, thr, rng):
    nzu = np.flatnonzero((al != 0).any(0)); Pnz = pure(al, sd, nzu, thr); Pn = []
    for _ in range(NNULL):
        ap = np.stack([al[f, rng.permutation(al.shape[1])] for f in range(3)]); u = np.flatnonzero((ap != 0).any(0))
        Pn.append(pure(ap, sd, u, thr))
    Pn = np.array(Pn)
    return dict(P_nz=Pnz, n_nonzero=len(nzu), null_q=[float(np.nanquantile(Pn, q)) for q in (.025, .5, .975)],
                p=float((1 + (Pn >= Pnz - 1e-12).sum()) / (1 + NNULL)))


def sq(a, nd):
    a = np.asarray(a, np.float64)
    while a.ndim > nd and a.shape[0] == 1: a = a[0]
    return a


def qinfo(q):
    try:
        return np.asarray(q.k, np.float64), np.asarray(q.i, np.float64), np.asarray(q.f, np.float64)
    except Exception as e:  # noqa: BLE001
        log('quantizer attrs unavailable', repr(e)); return None


def hgq_overflow(net, x, B=1000):
    """per quantized layer: fraction of quantizer input entries with |iq(v) - v| > 2^-f (real particles / all for rho);
    per jet: any overflow anywhere"""
    import jax.numpy as jnp
    L = net.L; KB = net.KB
    F = {k: qinfo(L[k].iq.quantizer) for k in ('phi1', 'phi2', 'pool', 'h1', 'h2', 'h3', 'head')}
    cnt = {k: [0, 0] for k in F}; anyj = []

    def ov(k, v):
        qv = np.asarray(L[k].iq(jnp.asarray(v, jnp.float32), training=False)); fk = F[k]
        tol = (2.0 ** (-np.broadcast_to(sq(fk[2], v.ndim - 1), v.shape[1:])) if fk is not None else np.full(v.shape[1:], 1e-3)) * 1.001 + 1e-6
        return np.abs(qv - v) > tol

    def dense(k, v, act=True):
        z = np.asarray(jnp.einsum(L[k].equation, L[k].iq(jnp.asarray(v, jnp.float32), training=False), KB[k][0]) + KB[k][1])
        return np.maximum(z, 0) if act else z
    for i in range(0, len(x), B):
        from interp.x5_dsets_sparsity import norm
        xn, m = norm(net, x[i:i + B]); rl = m > 0; aj = np.zeros(len(xn), bool)
        o = ov('phi1', xn); cnt['phi1'][0] += int((o & rl[..., None]).sum()); cnt['phi1'][1] += int(rl.sum() * 3); aj |= (o & rl[..., None]).any((1, 2))
        a = dense('phi1', xn); o = ov('phi2', a); cnt['phi2'][0] += int((o & rl[..., None]).sum()); cnt['phi2'][1] += int(rl.sum() * a.shape[-1]); aj |= (o & rl[..., None]).any((1, 2))
        u = dense('phi2', a) * m[..., None]; o = ov('pool', u); cnt['pool'][0] += int(o.sum()); cnt['pool'][1] += o.size; aj |= o.any((1, 2))
        h = np.asarray(net.pool(u))
        for k in ('h1', 'h2', 'h3', 'head'):
            o = ov(k, h); cnt[k][0] += int(o.sum()); cnt[k][1] += o.size; aj |= o.any(1); h = dense(k, h, act=(k != 'head'))
        anyj.append(aj)
    return {k: v[0] / max(v[1], 1) for k, v in cnt.items()}, np.concatenate(anyj)


def analyse(label, seed, d):
    from interp.jedi_core import sample
    R = dict(label=label, seed=seed, dir=d, sample=dict(per_class=2000, seed=SEED))
    JS = OUT / f'x5_verify_{label}_s{seed}.json'; save = lambda: JS.write_text(json.dumps(clean(R), indent=1))
    rd = Path(d)
    if label.startswith('hgq'):
        net = HgqDS(rd)
    else:
        net = FloatDS(rd / ('weights_zeroed.npz' if label == 'l1' else 'weights.npz'))
    x, y, _ = sample(per_class=2000, seed=SEED); x = np.ascontiguousarray(x); J = len(x); real = x[..., 0] > 0
    log(label, seed, 'sample', J)
    r0 = run(net, x, keep=('a', 'h3', 'logits')); c0 = r0['logits'].argmax(1)
    R['acc'] = float((c0 == y).mean()); log('acc', R['acc'])
    # ---- (a)
    amean = r0['a'][real].mean(0); chA = np.zeros(64)
    for c in range(64):
        chA[c] = (run(net, x, ov_a=(c, amean[c]), keep=('logits',))['logits'].argmax(1) != c0).mean()
    liveA = np.flatnonzero(chA >= 0.01)
    alpha = net.K1 / net.scale[:, None]; sdx = x[real].astype(np.float64).std(0)
    A = dict(n_live=len(liveA), phi1_zero_frac=float((net.K1 == 0).mean()), spt_all=types(alpha, sdx, np.arange(64), 0.9)[1],
             type_all=types(alpha, sdx, np.arange(64), 0.9)[0])
    for thr in (0.8, 0.9, 0.95):
        A[f'thr{thr}'] = dict(P_live=pure(alpha, sdx, liveA, thr), **null_test(alpha, sdx, thr, np.random.default_rng(424201)))
    if label.startswith('hgq'):
        import jax.numpy as jnp
        from interp.x5_dsets_sparsity import norm
        xn, m = norm(net, x); qx = np.asarray(net.L['phi1'].iq(jnp.asarray(xn), training=False))[real].astype(np.float64)
        sdq = qx.std(0) * net.scale            # physical units, so that alpha (= K1 / scale) applies
        A['iq_aware'] = dict(sd_raw=sdx, sd_q=sdq, P_live=pure(alpha, sdq, liveA, 0.9), **null_test(alpha, sdq, 0.9, np.random.default_rng(424202)))
        fk = qinfo(net.L['phi1'].iq.quantizer)
        if fk is not None:
            bits = np.broadcast_to(sq(np.maximum(fk[0] + fk[1] + fk[2], 0), 2), (64, 3)); A['phi1_iq_zero_bit_frac_per_feature'] = (bits <= 0).mean(0)
            A['phi1_iq_i_mean'] = np.broadcast_to(sq(fk[1], 2), (64, 3)).mean(0); A['phi1_iq_f_mean'] = np.broadcast_to(sq(fk[2], 2), (64, 3)).mean(0)
            A['phi1_iq_i_slot0_5'] = np.broadcast_to(sq(fk[1], 2), (64, 3))[:6]; A['phi1_iq_k_mean'] = np.broadcast_to(sq(fk[0], 2), (64, 3)).mean(0)
    R['a'] = A; log('(a)', {k: A[k] for k in A if k.startswith('thr')}, 'iq-aware p', A.get('iq_aware', {}).get('p'), A.get('phi1_iq_i_mean'), A.get('phi1_iq_f_mean')); save()
    # ---- (d)
    M = moments(x); perm = np.random.default_rng(424203).permutation(J); tr, te = np.sort(perm[:J // 2]), np.sort(perm[J // 2:])
    rd_ = Ridge(r0['h3'], M['logm'], tr[: 3 * len(tr) // 4], tr[3 * len(tr) // 4:], tr); isW = y == W_
    rng = np.random.default_rng(424204); pt = x[..., 0].astype(np.float64); rr = np.sqrt(x[..., 1].astype(np.float64) ** 2 + x[..., 2] ** 2)
    rbar = (pt * rr * real).sum(1) / np.maximum((pt * real).sum(1), 1e-9)
    th = 2 * np.arcsin(np.minimum(1, DELTA / (2 * np.maximum(rbar, 1e-9)))) * rng.choice([-1, 1], J); ang = rng.uniform(0, 2 * np.pi, J)
    E = {'translate_eta': edit(x, d_eta=DELTA), 'translate_eta_neg': edit(x, d_eta=-DELTA), 'rotation': rotate(x, th),
         'translate_random_dir': edit(x, d_eta=DELTA * np.cos(ang), d_phi=DELTA * np.sin(ang))}
    Me = moments(E['translate_eta']); pred = 0.5 * (np.log(Me['M0'] * Me['M2']) - np.log(M['M0'] * M['M2'])); lk = float(pred[te].mean())
    E['matched_scale'] = edit(x, k=float(np.exp(lk)))
    base = rd_(r0['h3']); D = dict(pred_half_dlogM0M2=lk, matched_k=float(np.exp(lk)), WtoZ_unedited=float((c0[isW] == Z_).mean()),
                                   readout_r2_test=float(1 - ((base[te] - M['logm'][te]) ** 2).sum() / ((M['logm'][te] - M['logm'][te].mean()) ** 2).sum()))
    calls, dm = {}, {}
    for k, xe in E.items():
        r = run(net, xe, keep=('h3', 'logits')); calls[k] = r['logits'].argmax(1); dm[k] = rd_(r['h3']) - base
        zW = calls[k][isW] == Z_; bi = np.random.default_rng(424205).integers(0, isW.sum(), (NB, isW.sum()))
        D[k] = dict(WtoZ=float(zW.mean()), WtoZ_wilson=wilson(int(zW.sum()), int(isW.sum())), WtoZ_boot=ci95(zW[bi].mean(1)),
                    dlogmhat=float(dm[k][te].mean()), calls_changed=float((calls[k] != c0).mean()))
    for k in ('rotation', 'translate_random_dir', 'translate_eta_neg'):
        D[f'mcnemar_translate_vs_{k}'] = mcnemar(calls['translate_eta'][isW] == Z_, calls[k][isW] == Z_)
    bt = np.random.default_rng(424206).integers(0, len(te), (NB, len(te))); ob, mt = dm['translate_eta'][te], dm['matched_scale'][te]
    D['calibrated_ratio'] = float(ob.mean() / mt.mean()); D['calibrated_ratio_ci'] = ci95(ob[bt].mean(1) / mt[bt].mean(1))
    ob2 = dm['rotation'][te]; D['rotation_ratio'] = float(ob2.mean() / mt.mean()); D['rotation_ratio_ci'] = ci95(ob2[bt].mean(1) / mt[bt].mean(1))
    if label.startswith('hgq'):
        OV = {}; anyov = {}
        for k, xe in [('unedited', x)] + list(E.items()):
            OV[k], anyov[k] = hgq_overflow(net, xe)
        D['overflow_frac'] = OV; D['jets_any_overflow'] = {k: float(v.mean()) for k, v in anyov.items()}
        clean_ = isW & ~anyov['unedited'] & ~anyov['translate_eta']
        D['WtoZ_no_overflow_W'] = dict(n=int(clean_.sum()), unedited=float((c0[clean_] == Z_).mean()) if clean_.any() else None,
                                       translate=float((calls['translate_eta'][clean_] == Z_).mean()) if clean_.any() else None,
                                       rotation=float((calls['rotation'][clean_] == Z_).mean()) if clean_.any() else None)
    R['d'] = D; R['wall_s'] = time.time() - T0
    log('(d)', {k: D[k] for k in ('WtoZ_unedited', 'calibrated_ratio', 'calibrated_ratio_ci', 'rotation_ratio')},
        {k: D[k]['WtoZ'] for k in E}, D.get('jets_any_overflow'), D.get('WtoZ_no_overflow_W')); save()


def combine():
    V = {}
    for lab in MODELS:
        for s in (0, 1, 2):
            p = OUT / f'x5_verify_{lab}_s{s}.json'
            if p.exists(): V[f'{lab}_s{s}'] = json.loads(p.read_text())
    C = dict(sample_seed=SEED, models=sorted(V), table={})
    for k, v in V.items():
        a, d = v['a'], v['d']
        C['table'][k] = dict(acc=v['acc'], n_live=a['n_live'], P_live=a['thr0.9']['P_live'], P_nz=a['thr0.9']['P_nz'], null=a['thr0.9']['null_q'],
                             p_a=a['thr0.9']['p'], p_a_holm4=min(1, 4 * a['thr0.9']['p']), p_a_thr08=a['thr0.8']['p'], p_a_thr095=a['thr0.95']['p'],
                             p_a_iq=a.get('iq_aware', {}).get('p'), WtoZ0=d['WtoZ_unedited'], WtoZ_tr=d['translate_eta']['WtoZ'],
                             WtoZ_neg=d['translate_eta_neg']['WtoZ'], WtoZ_rot=d['rotation']['WtoZ'], WtoZ_rand=d['translate_random_dir']['WtoZ'],
                             ratio=d['calibrated_ratio'], ratio_ci=d['calibrated_ratio_ci'], rot_ratio=d['rotation_ratio'],
                             p_rot=d['mcnemar_translate_vs_rotation']['p'], jets_any_overflow=d.get('jets_any_overflow'),
                             WtoZ_no_overflow=d.get('WtoZ_no_overflow_W'))
    # same-seed similarity of the two HGQ arms' phi1 (pT share per unit, unit types)
    sim = {}
    for s in (0, 1, 2):
        a, b = V.get(f'hgq_s{s}'), V.get(f'hgq_acc_s{s}')
        if a and b:
            x_, y_ = np.array(a['a']['spt_all'], float), np.array(b['a']['spt_all'], float)
            sim[s] = dict(pearson_pT_share=float(np.corrcoef(x_, y_)[0, 1]), same_type=float(np.mean(np.array(a['a']['type_all']) == np.array(b['a']['type_all']))))
    C['hgq_arms_same_seed_similarity'] = sim
    (OUT / 'x5_verify.json').write_text(json.dumps(clean(C), indent=1)); print(json.dumps(clean(C), indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--part', choices=list(PARTS)); ap.add_argument('--one', nargs=3)
    ap.add_argument('--combine', action='store_true'); a = ap.parse_args()
    if a.combine: combine()
    elif a.one: analyse(a.one[0], int(a.one[1]), a.one[2])
    else:
        rc = 0
        for lab in PARTS[a.part]:
            for s, d in MODELS[lab]:
                rc |= subprocess.run([sys.executable, '-m', 'interp.x5_verify', '--one', lab, str(s), d]).returncode
        combine(); sys.exit(rc)
