"""P4 — ParticleNet (official JetClass weights): is the axis-relative mass response of ParT (C3, D4) also in ParticleNet?

Models: ParticleNet_full, ParticleNet_kin (released weights /j-jepa-vol/aaron-interp/models/particlenet; network code copied into
interp/pn_net.py from weaver-core 0.4.17; settings of particle_transformer/networks/example_ParticleNet.py; inputs per the official
data configs copied to interp/pn_data/JetClass_{kin,full}.yaml, checked against jetdistill.part.network.FEATURES at start).
ParticleNet input: points (Δη, Δφ) for the layer-1 kNN graph (k = 16); per-particle features as ParT's (pt_log, e_log, logptrel,
logerel, deltaR [, charge, PID, d0, dz …], deta, dphi), BatchNorm'd; 3 EdgeConv blocks; layer-1 edge input = [x_i, x_j − x_i];
layers 2 and 3 build their kNN graphs in the learned feature space; mean pooling; FC 256 (ReLU) → 10. Padded to 128 slots, as weaver.
ParT_full / ParT_kin are run on the same jets with the same edits (call metrics only) for a paired comparison.

Samples: P1 rows 3,000–4,000 of each class's first test file (10,000 jets, part_core.sample, rebuilt with part_core.build) and the
fresh rows 6,000–7,000 (P2v). Bootstrap: 2,000 resamples over jets, stratified by class (seed 2), percentile 95 % intervals.
Step 0: accuracy (raw and rebuilt jets) vs the released ParticleNet number (JetClass test 0.844, full) and ParT on the same rows.
Step 1 (P1/P1v mirror): edits phi±0.06 (every real particle Δφ ± 0.06: rigid rotation, invariant mass unchanged), rand0.06 (per jet a
  random direction θ ~ U[0, 2π), seed 3: (Δη, Δφ) + 0.06 (cos θ, sin θ); the η part changes m by the particle masses only),
  angles × 1.05 / 1.1 (real mass change) and, per shift edit, angles × k with k matched (secant) so that the mean true Δlog m on the
  read-out test half equals the shift's mean ½Δlog(M0·M2) (R1's calibration). Controls: re-run of the unedited inputs (same batches;
  other batch size) = float noise floor; per-class calls changed (classes other than W/Z = untouched-class control).
  Metrics: accuracy, calls changed, W→Z and Z→W, Δ(logit_W − logit_Z) on true W, Δlog m̂ of a ridge read-out of log m from the 256-d
  last hidden layer (fit on the unedited jets, seed-1 50/50 split, alpha on the last quarter of the train half; mean over the test half).
  Tests: H0inv "invariant mass" (mean Δlog m̂ = 0), H0M2 "pure axis moment" (mean Δlog m̂ = c·mean ½Δlog(M0M2), c = read-out gain of
  the matched calibration edit; calibrated ratio = obs / (c·pred) = 1); bootstrap p (paired over test-half jets) with Holm over all
  ratio tests; McNemar (exact) for W→Z edit vs unedited and calls changed edit vs the re-run noise control.
Step 2 (route split; edits phi0.06, rand0.06, ang1.1): (a) full edit; kNN graph of layer 1 / of all layers fixed to the unedited graph;
  points only (only the layer-1 graph sees the edit) / points unedited; feature groups deltaR, (deta, dphi), all three, fed unedited
  ("removed") or alone edited ("only"); layer-1 EdgeConv terms: centre x_i (Φ), edge x_j − x_i (Θ), shortcut (1×1 conv on x_i) each
  edited alone (graph points unedited) or each kept unedited (rest edited). Share = variant / (a) of Δ(lW − lZ) on W, ΔW→Z, Δlog m̂,
  with stratified bootstrap intervals; H0 share = 0 and share = 1 (bootstrap p, Holm per model/sample/edit). kNN-graph overlap with
  the unedited graph per layer (fraction of the 16 neighbours kept, real particles).
Step 3 (D4): (Δη, Δφ, ΔR) about the stored axis (identity), the E-scheme axis, the hardest particle and the WTA axis (P1v's code;
  4-vectors unchanged): calls changed with intervals, McNemar vs the noise control; ParticleNet and ParT on the P1 rows.
  ./part_job.sh c-p4-run 'python -m interp.p4_particlenet'            -> $AI_RESULTS/particlenet/p4_particlenet.json
  ./part_job.sh c-p4-quick 'python -m interp.p4_particlenet 30'       (30 jets per class, smoke)
  python -m interp.p4_particlenet --plot                               -> $AI_RESULTS/particlenet/p4_*.png
"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

log = lambda *a: print(*a, flush=True)
RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'particlenet'
iZ, iW = 6, 7
NB = 2000
SHIFT = 0.06
ROUTE_EDITS = ('phi0.06', 'rand0.06', 'ang1.1')
GROUPS = dict(dR=['deltaR'], dedp=['deta', 'dphi'], coords=['deltaR', 'deta', 'dphi'])


def _imports():
    import torch
    from interp import part_core as pc, p1v_verify_part as p1v, p2v_verify_part as p2v, pn_net
    from jetdistill.part.network import FEATURES, inputs
    assert (pc.CL.index('Zqq'), pc.CL.index('Wqq')) == (iZ, iW)
    globals().update(torch=torch, pc=pc, p1v=p1v, p2v=p2v, pn_net=pn_net, FEATURES=FEATURES, inputs=inputs)


def check_config():
    """the copied official data configs vs the feature list used by jetdistill (names, centre, scale, clip; points)"""
    import yaml
    out = {}
    for n in ('kin', 'full'):
        C = yaml.safe_load(open(Path(__file__).parent / 'pn_data' / f'JetClass_{n}.yaml'))
        pts = [v[0] if isinstance(v, list) else v for v in C['inputs']['pf_points']['vars']]
        assert pts == ['part_deta', 'part_dphi'], pts
        fv = C['inputs']['pf_features']['vars']; F = FEATURES[n]; assert len(fv) == len(F), (n, len(fv), len(F))
        for v, (k, c, s, cl) in zip(fv, F):
            v = v if isinstance(v, list) else [v, None]
            assert v[0] == 'part_' + k, (v, k)
            if v[1] is None: assert c is None
            else:
                assert abs(v[1] - c) < 1e-9 and abs(v[2] - s) < 1e-9, (v, c, s)
                lo, hi = (v[3], v[4]) if len(v) > 4 else (-5, 5); assert tuple(cl) == (lo, hi), (v, cl)
        assert C['inputs']['pf_features']['length'] == 128 and C['inputs']['pf_features']['pad_mode'] == 'wrap'
        out[n] = dict(n_features=len(F), points=pts, ok=True)
    return out


# ------------------------------------------------------------------------------------------------ inputs and forward
def pn_inputs(J, n):
    """(x (N, F, 128), points (N, 2, 128), mask (N, 1, 128)) float32 as weaver feeds ParticleNet (length 128)"""
    x, _, m = inputs(J, n); P = x.shape[-1]
    if P < 128:
        x = np.concatenate([x, np.zeros(x.shape[:2] + (128 - P,), np.float32)], -1); m = np.concatenate([m, np.zeros(m.shape[:2] + (128 - P,), np.float32)], -1)
    names = [f[0] for f in FEATURES[n]]
    return dict(x=np.ascontiguousarray(x), p=np.ascontiguousarray(x[:, [names.index('deta'), names.index('dphi')]]), m=m)


def pn_batch(model, p, f, m, fc=None, fe=None, fs=None, kfix=None, kref=None):
    """ParticleNet.forward (use_fusion False) with the layer-1 EdgeConv input split: centre term from fc, edge term x_j − x_i from
    fe, shortcut from fs (each defaults to f); kfix: per-layer kNN indices to use instead of recomputing (None entries recompute);
    kref: per-layer reference indices for the overlap measure. Returns logits, hidden (input of the last Linear), knn list, overlaps."""
    torch = globals()['torch']; knn, ggf = pn_net.knn, pn_net.get_graph_feature_v1
    mask = m; counts = torch.clamp(mask.sum(-1), min=1.0)
    bn = lambda a: model.bn_fts(a * mask) * mask
    fts = bn(f); fc = fts if fc is None else bn(fc); fe = fts if fe is None else bn(fe); fs = fts if fs is None else bn(fs)
    shift = (mask == 0) * 1e9; knns, ovs = [], []
    for li, conv in enumerate(model.edge_convs):
        pts = (p * mask if li == 0 else fts) + shift
        idx = knn(pts, conv.k) if (kfix is None or kfix[li] is None) else kfix[li]
        knns.append(idx)
        if kref is not None:
            ov = (idx[..., :, None] == kref[li][..., None, :]).any(-1).float().mean(-1)
            ovs.append((ov * mask[:, 0]).sum(-1) / counts[:, 0])
        a, e, s_in = (fc, fe, fs) if li == 0 else (fts, fts, fts)
        C = a.shape[1]
        x = torch.cat([ggf(a, conv.k, idx)[:, :C], ggf(e, conv.k, idx)[:, C:]], 1)
        for c2, b2, act in zip(conv.convs, conv.bns, conv.acts): x = act(b2(c2(x)))
        x = x.mean(-1)
        sc = conv.sc_bn(conv.sc(s_in)) if conv.sc is not None else s_in
        fts = conv.sc_act(sc + x) * mask
    pooled = fts.sum(-1) / counts; h = model.fc[0](pooled); L = model.fc[1](h)
    return L, h, knns, ovs


def fwd(model, X, batch=250, fc=None, fe=None, fs=None, p=None, kfix=None, kref=None, capture=False, native=False):
    """batched forward over all jets (fixed contiguous batches). X from pn_inputs; fc/fe/fs/p: replacement arrays (N, ., 128).
    kfix / kref: per-layer (N, 128, 16) int16 arrays or None. native: weaver's own ParticleNet.forward (identity check)."""
    torch = globals()['torch']; dev = next(model.parameters()).device; N = len(X['x'])
    L = np.zeros((N, 10)); H = np.zeros((N, 256)); K = [np.zeros((N, 128, 16), np.int16) for _ in range(3)] if capture else None
    OV = np.zeros((N, 3)) if kref is not None else None
    T = lambda a, s: torch.from_numpy(np.ascontiguousarray(a[s])).to(dev)
    with torch.no_grad():
        for i in range(0, N, batch):
            s = slice(i, min(N, i + batch))
            pp = T(X['p'] if p is None else p, s); ff = T(X['x'], s); mm = T(X['m'], s)
            if native:
                Lb = model(pp.clone(), ff.clone(), mm); L[s] = Lb.double().cpu().numpy(); continue
            kf = None if kfix is None else [None if k is None else T(k, s).long() for k in kfix]
            kr = None if kref is None else [T(k, s).long() for k in kref]
            Lb, hb, kn, ov = pn_batch(model, pp, ff, mm, fc=None if fc is None else T(fc, s), fe=None if fe is None else T(fe, s),
                                      fs=None if fs is None else T(fs, s), kfix=kf, kref=kr)
            L[s] = Lb.double().cpu().numpy(); H[s] = hb.double().cpu().numpy()
            if capture:
                for li in range(3): K[li][s] = kn[li].cpu().numpy().astype(np.int16)
            if kref is not None: OV[s] = torch.stack(ov, 1).double().cpu().numpy()
    return dict(L=L, h=H, knn=K, ov=OV)


# ------------------------------------------------------------------------------------------------ sample and edits
def prep(per, start):
    J0 = pc.sample(per, start); pt, de, dp, ms = pc.kin(J0); real = pt > 0; Jb = pc.build(J0, pt, de, dp, ms); m0 = pc.jmass(Jb)
    y = J0['label'].astype(int); N = len(y)
    S = dict(J0=J0, Jb=Jb, y=y, pt=pt, de=de, dp=dp, ms=ms, real=real, logm0=np.log(np.maximum(m0, 1.0)), M=pc.moments(*pc.kin(Jb)[:3]), start=start)
    th = np.random.default_rng(3).uniform(0, 2 * np.pi, N)[:, None]
    JE = {'phi0.06': pc.build(J0, pt, de, dp + SHIFT * real, ms), 'phi-0.06': pc.build(J0, pt, de, dp - SHIFT * real, ms),
          'rand0.06': pc.build(J0, pt, de + SHIFT * np.cos(th) * real, dp + SHIFT * np.sin(th) * real, ms),
          'ang1.05': pc.build(J0, pt, de * 1.05, dp * 1.05, ms), 'ang1.1': pc.build(J0, pt, de * 1.1, dp * 1.1, ms)}
    S['split'] = p1v.split(N); te = S['split'][3]
    S['pred'] = {k: 0.5 * np.log(pc.moments(*pc.kin(Je)[:3])['M0M2'] / S['M']['M0M2']) for k, Je in JE.items()}
    S['true'] = {k: np.log(np.maximum(pc.jmass(Je), 1.0)) - S['logm0'] for k, Je in JE.items()}
    # matched calibration: angles × k with mean true Δlog m on the test half = mean prediction of each shift edit (secant on log k)
    S['cal'] = {}
    for e in ('phi0.06', 'phi-0.06', 'rand0.06'):
        tgt = float(S['pred'][e][te].mean())
        f = lambda lk: float((np.log(np.maximum(pc.jmass(pc.build(J0, pt, de * math.exp(lk), dp * math.exp(lk), ms)), 1.0)) - S['logm0'])[te].mean()) - tgt
        x0, x1 = 0.0, tgt; f0, f1 = f(x0), f(x1)
        for _ in range(6):
            if abs(f1) < 1e-6 or f1 == f0: break
            x0, x1, f0 = x1, x1 - f1 * (x1 - x0) / (f1 - f0), f1; f1 = f(x1)
        k = math.exp(x1); name = f'cal_{e}'; JE[name] = pc.build(J0, pt, de * k, dp * k, ms)
        S['true'][name] = np.log(np.maximum(pc.jmass(JE[name]), 1.0)) - S['logm0']; S['pred'][name] = S['true'][name]
        S['cal'][e] = dict(k=k, target=tgt, achieved=float(S['true'][name][te].mean()))
    S['JE'] = JE
    rng = np.random.default_rng(2); Wc = np.zeros((NB, N), np.float32)
    for c in range(10):
        ix = np.flatnonzero(y == c)
        for b in range(NB): Wc[b] += np.bincount(ix[rng.integers(0, len(ix), len(ix))], minlength=N)
    S['Wc'] = Wc
    S['edit_checks'] = {k: dict(dlogm_true_mean=float(S['true'][k].mean()), dlogm_true_absmax=float(np.abs(S['true'][k]).max()),
                                pred_mean=float(S['pred'][k].mean()), pred_mean_te=float(S['pred'][k][te].mean()),
                                mask_equal=bool(np.array_equal(Je['mask'], Jb['mask']))) for k, Je in JE.items()}
    return S


# ------------------------------------------------------------------------------------------------ statistics
def pq(S, v, sel):
    p, bs = p2v.bmean(S, np.asarray(v, np.float64), sel); return dict(est=p, ci=p2v.ci(bs)), bs


def call_stats(S, L0, L, c_noise=None):
    y = S['y']; c0, c = L0.argmax(1), L.argmax(1); isW, isZ, al = y == iW, y == iZ, np.ones(len(y), bool); oth = ~(isW | isZ)
    out = {}; B = {}
    for k, v, sel in (('acc', c == y, al), ('changed', c != c0, al), ('changed_other_classes', c != c0, oth), ('W_to_Z', c == iZ, isW),
                      ('W_to_Z_base', c0 == iZ, isW), ('dW_to_Z', (c == iZ).astype(float) - (c0 == iZ), isW), ('Z_to_W', c == iW, isZ),
                      ('Z_to_W_base', c0 == iW, isZ), ('dlogit_WZ_on_W', (L[:, iW] - L[:, iZ]) - (L0[:, iW] - L0[:, iZ]), isW),
                      ('dlogit_WZ_on_Z', (L[:, iW] - L[:, iZ]) - (L0[:, iW] - L0[:, iZ]), isZ)):
        out[k], B[k] = pq(S, v, sel)
    out['changed_by_class'] = [float((c != c0)[y == k].mean()) for k in range(10)]
    out['own_class'] = [float((c[y == k] == k).mean()) for k in range(10)]
    out['calls_from_W'] = [float((c[isW] == k).mean()) for k in range(10)]
    out['mcnemar_W_to_Z'] = p2v.mcnemar(c0[isW] == iZ, c[isW] == iZ)
    if c_noise is not None: out['mcnemar_changed_vs_noise'] = p2v.mcnemar(c_noise != c0, c != c0)
    return out, B


def readout_vec(S, ridge, h0, h):
    te = S['split'][3]; v = np.zeros(len(S['y'])); v[te] = ridge(h[te]) - ridge(h0[te]); sel = np.zeros(len(v), bool); sel[te] = True
    return v, sel


def share_stats(Ba, Bv):
    """share = variant / full edit, bootstrap interval and tests H0 share = 0, share = 1"""
    out = {}
    for k in Ba:
        a, v = Ba[k], Bv[k]; ok = np.abs(a[1]) > 1e-12
        p0 = p2v.pboot(v[1], 0.0); p1 = p2v.pboot(v[1] - a[1], 0.0)
        out[k] = dict(a=a[0], v=v[0], share=(v[0] / a[0] if abs(a[0]) > 1e-12 else None), share_ci=p2v.ci(v[1][ok] / a[1][ok]),
                      p_share0=p0[0], p_share0_norm=p0[1], p_share1=p1[0], p_share1_norm=p1[1])
    return out


def ratio_tests(S, ridge, h0, H, e):
    """H0inv: mean Δlog m̂ = 0; H0M2: mean Δlog m̂ = c·mean pred (c = gain of the matched calibration edit)"""
    te = np.zeros(len(S['y']), bool); te[S['split'][3]] = True
    obs, bo = pq(S, readout_vec(S, ridge, h0, H[e])[0], te); cal, bc = pq(S, readout_vec(S, ridge, h0, H[f'cal_{e}'])[0], te)
    tr, bt = pq(S, S['true'][f'cal_{e}'], te); pr, bp = pq(S, S['pred'][e], te)
    c, cb = cal['est'] / tr['est'], bc / bt; Eh, Eb = c * pr['est'], cb * bp
    p_inv = p2v.pboot(bo, 0.0); p_m2 = p2v.pboot(bo - Eb, 0.0)
    return dict(obs=obs, pred=pr, cal_true=tr, cal_readout=cal, c=c, c_ci=p2v.ci(cb), expected_M2=Eh, expected_M2_ci=p2v.ci(Eb),
                raw_ratio=obs['est'] / pr['est'], calibrated_ratio=obs['est'] / Eh, calibrated_ratio_ci=p2v.ci(bo / Eb),
                p_inv=p_inv[0], p_inv_norm=p_inv[1], p_M2=p_m2[0], p_M2_norm=p_m2[1])


# ------------------------------------------------------------------------------------------------ route variants
def variants(Xb, Xe, n):
    names = [f[0] for f in FEATURES[n]]; ix = {g: [names.index(k) for k in ks] for g, ks in GROUPS.items()}
    def swap(base, other, g): a = base.copy(); a[:, ix[g]] = other[:, ix[g]]; return a
    fb, fe, pb, pe = Xb['x'], Xe['x'], Xb['p'], Xe['p']
    V = {'a_full': dict(), 'knn1_fixed': dict(kfix='l1'), 'knnall_fixed': dict(kfix='all'),
         'points_only': dict(x=fb), 'points_unedited': dict(p=pb)}
    for g in GROUPS:
        V[f'{g}_removed'] = dict(x=swap(fe, fb, g)); V[f'{g}_only'] = dict(x=swap(fb, fe, g), p=pb)
    V.update(ctr_only=dict(p=pb, x=fb, fc=fe), edge_only=dict(p=pb, x=fb, fe=fe), sc_only=dict(p=pb, x=fb, fs=fe),
             ctr_removed=dict(fc=fb), edge_removed=dict(fe=fb), sc_removed=dict(fs=fb))
    return V


def run_variant(model, Xe, spec, K0):
    X = dict(Xe); X['x'] = spec.get('x', Xe['x'])
    kf = None
    if spec.get('kfix') == 'l1': kf = [K0[0], None, None]
    elif spec.get('kfix') == 'all': kf = K0
    return fwd(model, X, p=spec.get('p'), fc=spec.get('fc'), fe=spec.get('fe'), fs=spec.get('fs'), kfix=kf, kref=K0)


# ------------------------------------------------------------------------------------------------ main
def main():
    _imports(); torch = globals()['torch']; t0 = time.time(); RES.mkdir(parents=True, exist_ok=True)
    PER = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    PNDIR = Path(os.environ.get('JETDISTILL_MODELS', '/j-jepa-vol/aaron-interp/models/part')).parent / 'particlenet'
    res = dict(per_class=PER, n_boot=NB, boot_seed=2, rand_dir_seed=3, shift=SHIFT, device=dev, config=check_config(), samples={})
    log(f'step 0: official data configs match the jetdistill feature lists: {res["config"]}')
    res['wta_selftest'] = p1v.wta_selftest(); log(f'WTA self-test {res["wta_selftest"]}')
    SM = {}
    for s in (3000, 6000):
        S = prep(PER, s); SM[s] = S
        res['samples'][str(s)] = dict(rows=[s, s + PER], N=len(S['y']), edit_checks=S['edit_checks'], calibration=S['cal'], models={})
        log(f'sample rows {s}-{s + PER}: {len(S["y"])} jets ({time.time() - t0:.0f} s); edits: ' +
            '; '.join(f'{k} true Δlog m {v["dlogm_true_mean"]:+.5f} (max {v["dlogm_true_absmax"]:.4f}) pred ½Δlog M0M2 {v["pred_mean"]:+.4f}' for k, v in S['edit_checks'].items()))
        log(f'   calibration k: {S["cal"]}')
    # axes for step 3 (P1 rows)
    A, ainfo = p1v.axis_set(SM[3000]); AX = {a: p1v.reaxis(SM[3000]['Jb'], ea, pa)[0] for a, (ea, pa) in A.items()}
    res['axes_info'] = ainfo; log(f'axes ready ({time.time() - t0:.0f} s): {ainfo}')

    # -------------------------------- ParT on the same jets (calls only)
    for n in ('full', 'kin'):
        net = pc.ParTNetwork(n)
        for s, S in SM.items():
            t1 = time.time(); y = S['y']
            Lr = net.run(S['J0'])['logits']; L0 = net.run(S['Jb'])['logits']; Ln = net.run(S['Jb'], batch=128)['logits']; cn = Ln.argmax(1)
            row = dict(acc_raw=float((Lr.argmax(1) == y).mean()), acc=float((L0.argmax(1) == y).mean()),
                       noise_changed=float((cn != L0.argmax(1)).mean()), noise_max_dlogit=float(np.abs(Ln - L0).max()), edits={})
            for e in ('phi0.06', 'phi-0.06', 'rand0.06', 'ang1.1'):
                row['edits'][e] = call_stats(S, L0, net.run(S['JE'][e])['logits'], cn)[0]
            if s == 3000:
                row['axes'] = {a: call_stats(S, L0, net.run(E4)['logits'], cn)[0] for a, E4 in AX.items()}
            res['samples'][str(s)]['models'][f'ParT_{n}'] = row
            log(f'ParT_{n} rows {s}: acc {100 * row["acc"]:.2f} % (raw {100 * row["acc_raw"]:.2f}); noise {100 * row["noise_changed"]:.3f} % | ' +
                '; '.join(f'{e}: changed {100 * v["changed"]["est"]:.2f} %, W→Z {v["W_to_Z_base"]["est"]:.3f} → {v["W_to_Z"]["est"]:.3f} [{v["W_to_Z"]["ci"][0]:.3f}, {v["W_to_Z"]["ci"][1]:.3f}]' for e, v in row['edits'].items())
                + ('' if s != 3000 else ' | axes changed: ' + ', '.join(f'{a} {100 * v["changed"]["est"]:.2f} %' for a, v in row['axes'].items())) + f' ({time.time() - t1:.0f} s)')
        del net

    # -------------------------------- ParticleNet
    tests = []                          # (path, p) for Holm over every ratio test
    for n in ('full', 'kin'):
        model = pn_net.load_particlenet(PNDIR / f'ParticleNet_{n}.pt', len(FEATURES[n]), dev); key = f'ParticleNet_{n}'
        for s, S in SM.items():
            t1 = time.time(); y = S['y']; trf, trv, tr, te = S['split']
            Xr = pn_inputs(S['J0'], n); Xb = pn_inputs(S['Jb'], n)
            Lr = fwd(model, Xr)['L']; O0 = fwd(model, Xb, capture=True); L0, h0, K0 = O0['L'], O0['h'], O0['knn']
            Lnat = fwd(model, Xb, native=True)['L']
            On = fwd(model, Xb, kref=K0); Ln = On['L']; cn = Ln.argmax(1); Ln2 = fwd(model, Xb, batch=97)['L']
            ridge = p1v.Ridge(h0, S['logm0'], trf, trv, tr)
            row = dict(acc_raw=float((Lr.argmax(1) == y).mean()), acc=float((L0.argmax(1) == y).mean()),
                       acc_ci=pq(S, L0.argmax(1) == y, np.ones(len(y), bool))[0]['ci'],
                       own_class=[float((L0.argmax(1)[y == k] == k).mean()) for k in range(10)],
                       calls_from_W=[float((L0.argmax(1)[y == iW] == k).mean()) for k in range(10)],
                       split_vs_native_max_dlogit=float(np.abs(Lnat - L0).max()),
                       noise=dict(rerun_changed=float((cn != L0.argmax(1)).mean()), rerun_max_dlogit=float(np.abs(Ln - L0).max()),
                                  rerun_knn_overlap=On['ov'].mean(0).tolist(), batch97_changed=float((Ln2.argmax(1) != L0.argmax(1)).mean()),
                                  batch97_max_dlogit=float(np.abs(Ln2 - L0).max())),
                       readout=dict(r2=p1v.r2(ridge(h0[te]), S['logm0'][te]), alpha=float(ridge.alpha)), edits={}, ratio_tests={}, routes={})
            log(f'{key} rows {s}: acc {100 * row["acc"]:.2f} % [{100 * row["acc_ci"][0]:.2f}, {100 * row["acc_ci"][1]:.2f}] (raw jets {100 * row["acc_raw"]:.2f} %); '
                f'split forward vs weaver forward max |ΔL| {row["split_vs_native_max_dlogit"]:.2e}; noise {row["noise"]}; read-out R² {row["readout"]["r2"]:.3f}')
            # ---- step 1
            H = {}
            for e, Je in S['JE'].items():
                Xe = pn_inputs(Je, n); assert np.array_equal(Xe['m'], Xb['m'])
                O = fwd(model, Xe, kref=K0); H[e] = O['h']
                st, Bst = call_stats(S, L0, O['L'], cn)
                dv, sel = readout_vec(S, ridge, h0, O['h']); st['dlogm_hat'] = pq(S, dv, sel)[0]
                st['dlogm_hat_W'] = float(dv[sel & (y == iW)].mean()); st['pred_te'] = float(S['pred'][e][te].mean()); st['true_te'] = float(S['true'][e][te].mean())
                st['knn_overlap'] = O['ov'].mean(0).tolist(); st['knn_overlap_W'] = O['ov'][y == iW].mean(0).tolist()
                row['edits'][e] = st
                log(f'   {e:13s} acc {100 * st["acc"]["est"]:5.2f} changed {100 * st["changed"]["est"]:5.2f} % [{100 * st["changed"]["ci"][0]:.2f}, {100 * st["changed"]["ci"][1]:.2f}] '
                    f'(other classes {100 * st["changed_other_classes"]["est"]:.2f}) | W→Z {st["W_to_Z_base"]["est"]:.3f} → {st["W_to_Z"]["est"]:.3f} [{st["W_to_Z"]["ci"][0]:.3f}, {st["W_to_Z"]["ci"][1]:.3f}] '
                    f'McNemar {st["mcnemar_W_to_Z"]} | Z→W {st["Z_to_W_base"]["est"]:.3f} → {st["Z_to_W"]["est"]:.3f} | Δ(lW−lZ)|W {st["dlogit_WZ_on_W"]["est"]:+.3f} '
                    f'[{st["dlogit_WZ_on_W"]["ci"][0]:+.3f}, {st["dlogit_WZ_on_W"]["ci"][1]:+.3f}] | Δlog m̂ {st["dlogm_hat"]["est"]:+.4f} [{st["dlogm_hat"]["ci"][0]:+.4f}, {st["dlogm_hat"]["ci"][1]:+.4f}] '
                    f'pred {st["pred_te"]:+.4f} true {st["true_te"]:+.5f} | kNN overlap per layer {np.round(st["knn_overlap"], 4).tolist()}')
            for e in ('phi0.06', 'phi-0.06', 'rand0.06'):
                rt = ratio_tests(S, ridge, h0, H, e); row['ratio_tests'][e] = rt
                tests += [((key, str(s), e, 'p_inv'), rt['p_inv'] if rt['p_inv'] > 2 / (NB + 1) else rt['p_inv_norm']),
                          ((key, str(s), e, 'p_M2'), rt['p_M2'] if rt['p_M2'] > 2 / (NB + 1) else rt['p_M2_norm'])]
                log(f'   ratio {e}: obs {rt["obs"]["est"]:+.4f} {np.round(rt["obs"]["ci"], 4).tolist()}, pred {rt["pred"]["est"]:+.4f}, gain c {rt["c"]:.3f} {np.round(rt["c_ci"], 3).tolist()}, '
                    f'raw ratio {rt["raw_ratio"]:.3f}, calibrated ratio {rt["calibrated_ratio"]:.3f} {np.round(rt["calibrated_ratio_ci"], 3).tolist()}; '
                    f'p(H0 inv) {rt["p_inv"]:.4f}/{rt["p_inv_norm"]:.1e}, p(H0 M2) {rt["p_M2"]:.4f}/{rt["p_M2_norm"]:.1e}')
            # ---- step 2
            for e in ROUTE_EDITS:
                Xe = pn_inputs(S['JE'][e], n); V = variants(Xb, Xe, n); R = {}; Ba = None
                for vn, spec in V.items():
                    O = run_variant(model, Xe, spec, K0); st, Bst = call_stats(S, L0, O['L'], cn)
                    dv, sel = readout_vec(S, ridge, h0, O['h']); Bst['dlogm_hat'] = pq(S, dv, sel)[1]
                    Bq = {'dlogit_WZ_on_W': (st['dlogit_WZ_on_W']['est'], Bst['dlogit_WZ_on_W']), 'dW_to_Z': (st['dW_to_Z']['est'], Bst['dW_to_Z']),
                          'dlogm_hat': (pq(S, dv, sel)[0]['est'], Bst['dlogm_hat']), 'changed': (st['changed']['est'], Bst['changed'])}
                    if vn == 'a_full': Ba = Bq
                    R[vn] = dict(changed=st['changed'], W_to_Z=st['W_to_Z'], dlogit_WZ_on_W=st['dlogit_WZ_on_W'], mcnemar_W_to_Z=st['mcnemar_W_to_Z'],
                                 mcnemar_changed_vs_noise=st['mcnemar_changed_vs_noise'], knn_overlap=O['ov'].mean(0).tolist(),
                                 share=share_stats(Ba, Bq), max_abs_dlogit=float(np.abs(O['L'] - L0).max()))
                fam = [(vn, q, h) for vn in R if vn != 'a_full' for q in ('dlogit_WZ_on_W', 'dW_to_Z', 'dlogm_hat') for h in ('p_share0', 'p_share1')]
                ps = [R[vn]['share'][q][h] if R[vn]['share'][q][h] > 2 / (NB + 1) else R[vn]['share'][q][h + '_norm'] for vn, q, h in fam]
                for (vn, q, h), a in zip(fam, p2v.holm(ps)): R[vn]['share'][q][h + '_holm'] = a
                row['routes'][e] = R
                sh = lambda vn, q='dlogit_WZ_on_W': R[vn]['share'][q]
                log(f'   routes {e} (Δ(lW−lZ)|W full {R["a_full"]["dlogit_WZ_on_W"]["est"]:+.3f}; W→Z {R["a_full"]["W_to_Z"]["est"]:.3f}): ' + '; '.join(
                    f'{vn} {sh(vn)["share"]:.3f} [{sh(vn)["share_ci"][0]:.3f}, {sh(vn)["share_ci"][1]:.3f}] (m̂ {sh(vn, "dlogm_hat")["share"]:.3f}, W→Z {sh(vn, "dW_to_Z")["share"] if sh(vn, "dW_to_Z")["share"] is None else round(sh(vn, "dW_to_Z")["share"], 3)}; '
                    f'chg {100 * R[vn]["changed"]["est"]:.2f} %; kNN ov {np.round(R[vn]["knn_overlap"], 3).tolist()})' for vn in R if vn != 'a_full'))
            # ---- step 3
            if s == 3000:
                row['axes'] = {}
                for a, E4 in AX.items():
                    X4 = pn_inputs(E4, n); O = fwd(model, X4, kref=K0); st = call_stats(S, L0, O['L'], cn)[0]
                    st['knn_overlap'] = O['ov'].mean(0).tolist(); st['other_features_max_change'] = float(np.abs(np.delete(X4['x'] - Xb['x'], [
                        [f[0] for f in FEATURES[n]].index(k) for k in ('deltaR', 'deta', 'dphi')], 1)).max())
                    row['axes'][a] = st
                    log(f'   axis {a:8s}: changed {100 * st["changed"]["est"]:.2f} % [{100 * st["changed"]["ci"][0]:.2f}, {100 * st["changed"]["ci"][1]:.2f}], '
                        f'acc {100 * st["acc"]["est"]:.2f}, W→Z {st["W_to_Z"]["est"]:.3f}, McNemar changed vs noise {st["mcnemar_changed_vs_noise"]}, '
                        f'kNN overlap {np.round(st["knn_overlap"], 4).tolist()}, other features max Δ {st["other_features_max_change"]:.1e}')
            res['samples'][str(s)]['models'][key] = row
            log(f'{key} rows {s} done ({time.time() - t1:.0f} s)')
            json.dump(res, open(RES / ('p4_particlenet.json' if PER == 1000 else f'p4_particlenet_{PER}.json'), 'w'), indent=1)
        del model
    adj = p2v.holm([p for _, p in tests])
    res['holm_ratio_tests'] = [dict(test='/'.join(k), p=p, p_holm=a) for (k, p), a in zip(tests, adj)]
    log('Holm over ratio tests: ' + '; '.join(f'{d["test"]} {d["p"]:.2e} → {d["p_holm"]:.2e}' for d in res['holm_ratio_tests']))
    fn = RES / ('p4_particlenet.json' if PER == 1000 else f'p4_particlenet_{PER}.json')
    json.dump(res, open(fn, 'w'), indent=1); log(f'wrote {fn}'); log(f'P4 DONE ({time.time() - t0:.0f} s)')


# ------------------------------------------------------------------------------------------------ figures
def plot():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    R = json.load(open(RES / 'p4_particlenet.json')); S = R['samples']['3000']['models']; F = R['samples']['6000']['models']
    INK, INK2 = '#222', '#666'
    models = ['ParticleNet_full', 'ParticleNet_kin', 'ParT_full', 'ParT_kin']
    conds = [('unedited', None, '#bbbbbb'), ('rigid φ + 0.06 rad\n(mass unchanged)', 'phi0.06', '#c0392b'),
             ('random direction 0.06 rad\n(mass unchanged)', 'rand0.06', '#e67e22'), ('angles × 1.1\n(mass +{:.1f} %)'.format(100 * (math.exp(R['samples']['3000']['edit_checks']['ang1.1']['dlogm_true_mean']) - 1)), 'ang1.1', '#2c7fb8')]
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=140); w = 0.2
    for j, (lab, e, col) in enumerate(conds):
        xs, ys, lo, hi = [], [], [], []
        for i, m in enumerate(models):
            d = S[m]['edits']['phi0.06']['W_to_Z_base'] if e is None else S[m]['edits'][e]['W_to_Z']
            xs.append(i + (j - 1.5) * w); ys.append(100 * d['est']); lo.append(100 * (d['est'] - d['ci'][0])); hi.append(100 * (d['ci'][1] - d['est']))
        ax.bar(xs, ys, w * 0.92, color=col, label=lab, yerr=[lo, hi], capsize=2, ecolor=INK2)
        for x, v in zip(xs, ys): ax.text(x, v + 1.2, f'{v:.0f}', ha='center', fontsize=8, color=INK)
    ax.set_xticks(range(len(models))); ax.set_xticklabels([f'{m.replace("_", " ")}\n(acc. {100 * S[m]["acc"]:.1f} %)' for m in models], fontsize=10)
    ax.set_ylabel('true W jets called Z [%]', fontsize=11); ax.spines[['top', 'right']].set_visible(False)
    ax.set_title('Moving every particle 0.06 rad off the jet axis turns W calls into Z in both ParticleNet and ParT', fontsize=11.5, color=INK, loc='left')
    ax.legend(fontsize=8.5, frameon=False, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.17))
    fig.text(0.01, 0.005, '10,000 JetClass test jets (rows 3,000–4,000 of each class\'s first file, 1,000 per class); bars: 1,000 true W jets; 95 % bootstrap intervals (2,000 resamples, class-stratified).',
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(RES / 'p4_wz_shift.png', facecolor='white'); plt.close(fig); log('wrote', RES / 'p4_wz_shift.png')
    # route shares under phi0.06 (and ang1.1) for ParticleNet
    rows = [('kNN graph fixed (layer 1)', 'knn1_fixed'), ('kNN graphs fixed (all 3 layers)', 'knnall_fixed'), ('only kNN points edited', 'points_only'),
            ('only deltaR edited', 'dR_only'), ('only (Δη, Δφ) features edited', 'dedp_only'), ('only centre term x_i edited', 'ctr_only'),
            ('only edge term x_j − x_i edited', 'edge_only'), ('only shortcut (x_i) edited', 'sc_only')]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), dpi=140, sharey=True)
    for ax, m in zip(axs, ('ParticleNet_full', 'ParticleNet_kin')):
        for j, (e, col, lab) in enumerate((('phi0.06', '#c0392b', 'rigid φ + 0.06 (mass unchanged)'), ('ang1.1', '#2c7fb8', 'angles × 1.1 (real mass change)'))):
            Rt = S[m]['routes'][e]; ys = np.arange(len(rows)) + (j - 0.5) * 0.36
            v = [Rt[k]['share']['dlogit_WZ_on_W']['share'] for _, k in rows]; c = [Rt[k]['share']['dlogit_WZ_on_W']['share_ci'] for _, k in rows]
            ax.barh(ys, v, 0.34, color=col, label=lab, xerr=[[a - b[0] for a, b in zip(v, c)], [b[1] - a for a, b in zip(v, c)]], capsize=2, ecolor=INK2)
        ax.axvline(0, color=INK, lw=0.8); ax.axvline(1, color=INK2, lw=0.8, ls='--')
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=9); ax.invert_yaxis()
        ax.set_xlabel('share of the full edit\'s change in logit(W) − logit(Z) on true W jets', fontsize=9.5)
        ax.set_title(m.replace('_', ' '), fontsize=11, color=INK, loc='left'); ax.spines[['top', 'right']].set_visible(False)
    axs[0].legend(fontsize=8.5, frameon=False, loc='lower right')
    fig.suptitle('ParticleNet: which input route carries the W/Z response (1 = all of it; "kNN graph fixed" rows keep the rest edited)', fontsize=11, color=INK, x=0.01, ha='left')
    fig.tight_layout(); fig.savefig(RES / 'p4_routes.png', facecolor='white'); plt.close(fig); log('wrote', RES / 'p4_routes.png')


if __name__ == '__main__':
    if sys.argv[1:] == ['--plot']: plot()
    else: main()
