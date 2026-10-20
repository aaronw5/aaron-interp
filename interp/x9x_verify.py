"""X9x verification (X9x_verify): fresh sample, finer T grid around each augmentation range A, and a missing control for
H_recentre (offset-readout removal on the pooled latent).

Sample: interp.jedi_core.sample(per_class=1600, seed=4801) (N 8,000; 1,600 true W); none of the X9x seeds (3801, 3802).
Networks: the 30 trained X9x networks (PFN, EFN, Deep Sets; A in {0, 0.04, 0.08, 0.16, 0.30}; seeds 0, 1), loaded with
  interp.x9x_augrange.load_net (same numpy forward as X9x; accuracy on this sample vs stored test accuracy as a check).
(1) Fine-grid replication of Rc and T*_25.  Per network T in {A/2, A, A+0.01, A+0.02, A+0.04, A+0.08} (A = 0: {0.02, 0.04}),
  translation of all real particles at 4 azimuths (0, 90, 180, 270 deg; per-jet mean), rotation null (X6 rotate_matched,
  rng 4810 + round(1000 T)), matched dilation dilM2 (angles x sqrt(1 + T^2 M0 / M2)).  dc = calls(T) - calls(rot) per jet,
  Rc = mean dc / mean calls(dilM2), 95 % jet-bootstrap (1,000 draws).  T*_25 = first linearly interpolated T where Rc
  reaches 0.25 (as X9x).  Per architecture Pearson r and OLS slope of T*_25 on A; permutation p (20,000 draws).
(2) Offset-readout removal (H_recentre control; not run in X9x).  Networks A in {0, 0.16, 0.30}, seeds 0, 1.  Rows split by a
  random permutation (rng 4820) into F (fit, 4,000) and V (test, 4,000).  Ridge probe latent -> pT-weighted centroid c on F
  jets translated by |t| ~ U(0, 0.4), uniform direction (rng 4821); offset-readout subspace Q = orthonormalised raw-coordinate
  read-out vectors (B / sd) for c_eta and c_phi (2-dim).  On V, translation by T_in = 0.75 A (A = 0: 0.12) at 4 azimuths:
  dL = L(translated) - L(unedited).  Latent interventions (head applied to the edited pooled latent):
    full      L0 + dL                    (= the real translation)
    removed   L0 + dL - Q Q^T dL          (translation with its offset read-out removed)
    only      L0 + Q Q^T dL               (offset read-out moved, nothing else)
    random    L0 + dL - R R^T dL          (100 random 2-dim subspaces R, rng 4830)
  Metric: fraction of jets whose call differs from the unedited call (mean over azimuths); calibration: calls(dilM2 at T_in).
  H0 (H_tolerant / no linear offset read-out used): removed - full ~ random - full.  H1 (H_recentre through a linear 2-dim
  offset read-out): removed - full > 0, above the random-subspace 95th percentile, of the order of calls(dilM2).
  Jet bootstrap CI on removed - full; exact McNemar (removed vs full call-change indicators, azimuth 0).
  ./job.sh x9x-verify 'python -m interp.x9x_verify'  -> $AI_RESULTS/jedi/x9x_verify.json
(3) --ctrl2 (removal2): same split, probe and T_in; 200 random 2-dim subspaces (rng 4831) with the removed (or injected)
  perturbation rescaled per jet to the norm of the Q perturbation (norm-matched null); removal of the top-2 principal subspace of
  dL; direction of call changes (fraction of jets moved to a heavier / lighter mass rank, g = q < W < Z < t); probe response
  slope d(probe c_eta) / dT on +deta translations of V.
  ./job.sh x9x-verify2 'python -m interp.x9x_verify --ctrl2'  -> $AI_RESULTS/jedi/x9x_verify2.json
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp.x9x_augrange import load_net, centroid, ridge_fit, ridge_pred, mcnemar, clean, AUGS, SEEDS  # noqa: E402
from interp.x9_efn_pfn import translate, rotate_matched, scale_angles, moments  # noqa: E402

RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); OUT = RES / 'jedi'
SEED, PER = 4801, 1600; AZ = np.deg2rad([0.0, 90.0, 180.0, 270.0]); NB = 1000; T0 = time.time()


def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def grid(A):
    if A == 0: return [0.02, 0.04]
    return sorted({round(v, 4) for v in (A / 2, A, A + 0.01, A + 0.02, A + 0.04, A + 0.08)})


def rc_scan(net, x, M, Ts, Ib):
    c0 = net.logits(x).argmax(1); out = {}
    for T in Ts:
        ch = np.zeros(len(x))
        for th in AZ: ch += net.logits(translate(x, T, th)).argmax(1) != c0
        ch /= len(AZ)
        cr = net.logits(rotate_matched(x, T, np.random.default_rng(4810 + int(round(1000 * T))))).argmax(1) != c0
        cd = net.logits(scale_angles(x, np.sqrt(1 + T ** 2 * M['M0'] / np.maximum(M['M2'], 1e-30)))).argmax(1) != c0
        dc = ch - cr; md, mD = dc[Ib].mean(1), cd.astype(float)[Ib].mean(1); ok = mD > 0
        rb = md[ok] / mD[ok]
        out[str(T)] = dict(calls_T=float(ch.mean()), calls_rot=float(cr.mean()), calls_dil=float(cd.mean()), dc=float(dc.mean()),
                           dc_ci=[float(np.quantile(md, .025)), float(np.quantile(md, .975))],
                           Rc=float(dc.mean() / cd.mean()) if cd.mean() > 0 else None,
                           Rc_ci=[float(np.quantile(rb, .025)), float(np.quantile(rb, .975))] if ok.mean() > 0.9 else [None, None])
    return dict(acc_ref=None, per_T=out, c0=c0)


def t25(per_T, Ts):
    rr = [per_T[str(T)]['Rc'] or 0.0 for T in Ts]
    for k, (T, r) in enumerate(zip(Ts, rr)):
        if r >= 0.25:
            return T if k == 0 else Ts[k - 1] + (0.25 - rr[k - 1]) / (r - rr[k - 1]) * (T - Ts[k - 1])
    return None


def perm(T, A, draws=20000, seed=4840):
    T, A = np.asarray(T, float), np.asarray(A, float); r0 = np.corrcoef(A, T)[0, 1]; rng = np.random.default_rng(seed)
    c = sum(np.corrcoef(rng.permutation(A), T)[0, 1] >= r0 - 1e-12 for _ in range(draws))
    return float(r0), float((1 + c) / (1 + draws))


def orth(V):
    q, _ = np.linalg.qr(V); return q


def removal(net, x, M, A):
    rng = np.random.default_rng(4820); p = rng.permutation(len(x)); F, V = np.sort(p[:len(x) // 2]), np.sort(p[len(x) // 2:])
    r = np.random.default_rng(4821); xf = x[F].copy(); rl = xf[..., 0] > 0
    rad, th = r.uniform(0, 0.4, len(F)), r.uniform(0, 2 * np.pi, len(F))
    xf[..., 1] += (rad * np.cos(th))[:, None].astype(np.float32) * rl; xf[..., 2] += (rad * np.sin(th))[:, None].astype(np.float32) * rl
    P = ridge_fit(net.latent(xf), centroid(xf)); Q = orth(np.stack([P['B'][:, 0] / P['sd'], P['B'][:, 1] / P['sd']], 1))
    xv = x[V]; L0 = net.latent(xv); c0 = net.head(L0).argmax(1); Tin = 0.75 * A if A > 0 else 0.12
    # probe check on V: translated by +Tin along deta
    pv = ridge_pred(P, net.latent(translate(xv, Tin, 0.0)))[:, 0]; cv = centroid(translate(xv, Tin, 0.0))[:, 0]
    rq = np.random.default_rng(4830); Rs = [orth(rq.standard_normal((net.ell, 2))) for _ in range(100)]
    full = np.zeros(len(V)); rem = np.zeros(len(V)); only = np.zeros(len(V)); rnd = np.zeros((100, len(V))); fq, fr = [], []
    f0 = r0 = None
    for th in AZ:
        dL = net.latent(translate(xv, Tin, th)) - L0; pq = dL @ Q @ Q.T
        cf = net.head(L0 + dL).argmax(1) != c0; cr_ = net.head(L0 + dL - pq).argmax(1) != c0
        full += cf; rem += cr_; only += net.head(L0 + pq).argmax(1) != c0
        if f0 is None: f0, r0 = cf, cr_
        fq.append(float(np.linalg.norm(pq, axis=1).mean() / np.linalg.norm(dL, axis=1).mean()))
        for k, R in enumerate(Rs):
            pr = dL @ R @ R.T; rnd[k] += net.head(L0 + dL - pr).argmax(1) != c0
            if th == AZ[0]: fr.append(float(np.linalg.norm(pr, axis=1).mean() / np.linalg.norm(dL, axis=1).mean()))
    full, rem, only, rnd = full / 4, rem / 4, only / 4, rnd / 4
    mv = dict(M0=M['M0'][V], M2=M['M2'][V])
    cd = net.head(net.latent(scale_angles(xv, np.sqrt(1 + Tin ** 2 * mv['M0'] / np.maximum(mv['M2'], 1e-30))))).argmax(1) != c0
    # outside-range reference: real translation by A + 0.08
    co = np.zeros(len(V))
    for th in AZ: co += net.head(net.latent(translate(xv, A + 0.08, th))).argmax(1) != c0
    co /= 4
    I = np.random.default_rng(4841).integers(0, len(V), (NB, len(V))); d = rem - full; db = d[I].mean(1)
    rm = rnd.mean(1) - full.mean()
    return dict(T_in=Tin, N_fit=len(F), N_test=len(V), probe_R2_deta_on_V=float(1 - ((pv - cv) ** 2).sum() / ((cv - cv.mean()) ** 2).sum()),
                probe_alpha=P['alpha'], calls_full=float(full.mean()), calls_removed=float(rem.mean()), calls_only=float(only.mean()),
                removed_minus_full=float(d.mean()), removed_minus_full_ci=[float(np.quantile(db, .025)), float(np.quantile(db, .975))],
                random_removed_minus_full=dict(mean=float(rm.mean()), q05=float(np.quantile(rm, .05)), q95=float(np.quantile(rm, .95)),
                                               p_one_sided=float((1 + (rm >= d.mean() - 1e-12).sum()) / 101)),
                frac_dL_norm_in_Q=float(np.mean(fq)), frac_dL_norm_in_random_mean=float(np.mean(fr)),
                calls_dilM2_Tin=float(cd.mean()), calls_translation_A_plus_0p08=float(co.mean()),
                removed_minus_full_over_dil=float(d.mean() / cd.mean()) if cd.mean() > 0 else None,
                mcnemar_removed_vs_full_az0=mcnemar(r0, f0))


RANK = np.array([0, 0, 1, 2, 3])   # g, q, W, Z, t ordered by jet mass (g and q share rank 0)


def shift_dir(c0, c1):
    """fraction of all jets whose call moves to a heavier / lighter mass rank"""
    d = RANK[c1] - RANK[c0]; return float((d > 0).mean()), float((d < 0).mean())


def removal2(net, x, M, A, nr=200):
    """(3) norm-matched control for the offset-readout removal: random 2-dim subspaces R with the perturbation rescaled per jet
    to the norm of the Q perturbation; plus the top-2 principal subspace of dL; plus the direction of the call changes."""
    rng = np.random.default_rng(4820); p = rng.permutation(len(x)); F, V = np.sort(p[:len(x) // 2]), np.sort(p[len(x) // 2:])
    r = np.random.default_rng(4821); xf = x[F].copy(); rl = xf[..., 0] > 0
    rad, th = r.uniform(0, 0.4, len(F)), r.uniform(0, 2 * np.pi, len(F))
    xf[..., 1] += (rad * np.cos(th))[:, None].astype(np.float32) * rl; xf[..., 2] += (rad * np.sin(th))[:, None].astype(np.float32) * rl
    P = ridge_fit(net.latent(xf), centroid(xf)); Q = orth(np.stack([P['B'][:, 0] / P['sd'], P['B'][:, 1] / P['sd']], 1))
    xv = x[V]; L0 = net.latent(xv); c0 = net.head(L0).argmax(1); Tin = 0.75 * A if A > 0 else 0.12
    slope = float((ridge_pred(P, net.latent(translate(xv, Tin, 0.0)))[:, 0] - ridge_pred(P, L0)[:, 0]).mean() / Tin)
    dLs = [net.latent(translate(xv, Tin, th)) - L0 for th in AZ]
    _, _, Vt = np.linalg.svd(np.concatenate(dLs) - 0, full_matrices=False); Pc = Vt[:2].T
    rq = np.random.default_rng(4831); Rs = [orth(rq.standard_normal((net.ell, 2))) for _ in range(nr)]
    acc = {k: np.zeros(len(V)) for k in ('full', 'remQ', 'onlyQ', 'remPC')}; rr, ro = np.zeros((nr, len(V))), np.zeros((nr, len(V)))
    dirs = {k: [] for k in ('full', 'remQ', 'onlyQ')}; nPC = []
    for dL in dLs:
        pq = dL @ Q @ Q.T; nq = np.linalg.norm(pq, axis=1)
        cs = dict(full=net.head(L0 + dL).argmax(1), remQ=net.head(L0 + dL - pq).argmax(1), onlyQ=net.head(L0 + pq).argmax(1),
                  remPC=net.head(L0 + dL - dL @ Pc @ Pc.T).argmax(1))
        nPC.append(float(np.linalg.norm(dL @ Pc @ Pc.T, axis=1).mean() / np.linalg.norm(dL, axis=1).mean()))
        for k, c in cs.items():
            acc[k] += c != c0
            if k in dirs: dirs[k].append(shift_dir(c0, c))
        for i, R in enumerate(Rs):
            pr = dL @ R @ R.T; a = (nq / np.maximum(np.linalg.norm(pr, axis=1), 1e-12))[:, None]
            rr[i] += net.head(L0 + dL - a * pr).argmax(1) != c0; ro[i] += net.head(L0 + a * pr).argmax(1) != c0
    for k in acc: acc[k] /= len(AZ)
    rr, ro = rr / len(AZ), ro / len(AZ)
    out = dict(T_in=Tin, N_test=len(V), probe_slope_on_translation=slope, frac_dL_norm_in_PC2=float(np.mean(nPC)),
               calls={k: float(v.mean()) for k, v in acc.items()},
               heavier_lighter={k: [float(np.mean([d[0] for d in v])), float(np.mean([d[1] for d in v]))] for k, v in dirs.items()})
    I = np.random.default_rng(4843).integers(0, len(V), (NB, len(V)))
    for name, obs, rnd in (('remove', acc['remQ'], rr), ('only', acc['onlyQ'], ro)):
        ex = obs - acc['full'] if name == 'remove' else obs; exr = rnd.mean(1) - (acc['full'].mean() if name == 'remove' else 0.0)
        b = ex[I].mean(1)
        out[name] = dict(Q=float(ex.mean()), Q_ci=[float(np.quantile(b, .025)), float(np.quantile(b, .975))],
                         random_normmatched=dict(mean=float(exr.mean()), q95=float(np.quantile(exr, .95)), max=float(exr.max())),
                         p_one_sided=float((1 + (exr >= ex.mean() - 1e-12).sum()) / (nr + 1)))
    return out


def main2():
    from interp.jedi_core import sample
    x, y, _ = sample(PER, SEED); x = np.ascontiguousarray(x, np.float32); M = moments(x)
    R = dict(header='see interp/x9x_verify.py docstring, part (3) removal2', sample=dict(per_class=PER, seed=SEED, N=len(y)), removal2={})
    fo = OUT / 'x9x_verify2.json'
    for arch in ('dsets', 'pfn', 'efn'):
        for A in (0.0, 0.16, 0.30):
            for s in SEEDS:
                key = f'{arch}_A{A}_s{s}'; net = load_net(arch, A, s)[0]; q = removal2(net, x, M, A); R['removal2'][key] = q
                log(key, json.dumps(clean(q))); fo.write_text(json.dumps(clean(R), indent=1))
    log('wrote', fo, 'X9x verify2 DONE')


def main():
    from interp.jedi_core import sample
    x, y, _ = sample(PER, SEED); x = np.ascontiguousarray(x, np.float32); M = moments(x)
    Ib = np.random.default_rng(4842).integers(0, len(x), (NB, len(x)))
    log(f'sample({PER}, {SEED}): N {len(y)}, W {int((y == 2).sum())}')
    R = dict(header='see interp/x9x_verify.py docstring', sample=dict(per_class=PER, seed=SEED, N=len(y)), nets={}, removal={})
    fo = OUT / 'x9x_verify.json'; save = lambda: fo.write_text(json.dumps(clean(R), indent=1))
    for arch in ('dsets', 'pfn', 'efn'):
        for A in AUGS:
            for s in SEEDS:
                key = f'{arch}_A{A}_s{s}'
                try: net, info, _ = load_net(arch, A, s)
                except Exception as e:  # noqa: BLE001
                    log('SKIP', key, repr(e)[:200]); R['nets'][key] = dict(error=repr(e)[:300]); save(); continue
                Ts = grid(A); o = rc_scan(net, x, M, Ts, Ib); c0 = o.pop('c0')
                o.update(arch=arch, A=A, seed=s, Ts=Ts, acc=float((c0 == y).mean()), acc_test=info.get('acc_test'), T_star_25=t25(o['per_T'], Ts))
                R['nets'][key] = o
                log(key, f'acc {o["acc"]:.4f} (test {info.get("acc_test")}); T*_25 {o["T_star_25"]}; Rc ' +
                    ' '.join(f'{T}:{o["per_T"][str(T)]["Rc"]:.3f}' for T in Ts if o['per_T'][str(T)]['Rc'] is not None))
                if A in (0.0, 0.16, 0.30):
                    R['removal'][key] = removal(net, x, M, A); q = R['removal'][key]
                    log(key, 'removal', json.dumps(clean({k: q[k] for k in ('T_in', 'probe_R2_deta_on_V', 'calls_full', 'calls_removed', 'calls_only',
                                                                             'removed_minus_full', 'removed_minus_full_ci', 'random_removed_minus_full',
                                                                             'frac_dL_norm_in_Q', 'frac_dL_norm_in_random_mean', 'calls_dilM2_Tin',
                                                                             'calls_translation_A_plus_0p08')})))
                save()
        rows = [v for k, v in R['nets'].items() if v.get('arch') == arch and v.get('T_star_25') is not None]
        if len(rows) >= 4:
            A_, T_ = [v['A'] for v in rows], [v['T_star_25'] for v in rows]; r, p = perm(T_, A_); sl, ic = np.polyfit(A_, T_, 1)
            aug = [v for v in rows if v['A'] > 0]; sl2, ic2 = np.polyfit([v['A'] for v in aug], [v['T_star_25'] for v in aug], 1)
            R.setdefault('fit', {})[arch] = dict(n=len(rows), pearson_r=r, p_one_sided=p, slope=float(sl), intercept=float(ic),
                                                 slope_aug_only=float(sl2), intercept_aug_only=float(ic2),
                                                 T_minus_A=[float(v['T_star_25'] - v['A']) for v in aug])
            log(arch, 'fit', json.dumps(clean(R['fit'][arch])))
        save()
    log('wrote', fo, 'X9x verify DONE')


if __name__ == '__main__':
    main2() if '--ctrl2' in sys.argv else main()
