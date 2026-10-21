"""P36 — on-manifold axis test (claims C3, D4; JEDI E23 analogue): do ParT, ParticleNet and L-GATr change their output when a
jet whose pT centroid naturally sits far from the stored jet axis is recentred, at fixed invariant mass?

Background. Every C3 number so far (P1, P2, P23) moved the axis by an artificial rigid shift of all constituents. Real jets are
already off-centre: the pT-weighted centroid of the constituents, c = (sum pT_i deta_i, sum pT_i dphi_i) / sum pT_i, is not exactly
on the stored (E-scheme, four-momentum-sum) axis. If ParT's mass read-out is tied to the axis moment M2 = sum pT_i r_i^2 about the
stored axis (r_i = (deta_i, dphi_i)), then M2 = M2_c + |c|^2 M0 (M0 = sum pT, M2_c = moment about the centroid), so a jet with a
large natural |c| carries an extra |c|^2 M0 in M2 and should change its output when recentred, by the amount a real mass change
with the same change of 1/2 log M2 produces.

Models (all on the same jets): released ParT_full, ParT_kin (JetClass, 10 classes); released ParticleNet_full, ParticleNet_kin;
untrained ParT_full (weaver initialisation, torch seed 1; zero reference); L-GATr (our own training run anrunw-ai-c-p5-lgatr2,
latest checkpoint in /j-jepa-vol/aaron-interp/models/lgatr/runs/jc2m/jc2m_s0a/models at job start; Lorentz-equivariant
reference: it reads lab four-momenta and computes its angular features about the particle-sum axis it recomputes, so a rigid
phi translation is an exact symmetry for it and an eta translation is close to a longitudinal boost).
Sample: JetClass test_20M <class>_118.root, 10 classes, PER random rows per class (numpy default_rng(3636), drawn class by class in
part_core.CL order; never first-k); jets with < 2 particles dropped. Inputs rebuilt with part_core.build
(eta = jet_eta + sign(jet_eta) deta, phi = jet_phi + dphi); 'base' = unedited particles rebuilt.
Selection: |c| per jet; per class, the top 1 / 2 / 5 % of |c| ('tail1', 'tail2', 'tail5'). Matched low-|c| partner for every
tail5 jet: same class, |c| below the class median, nearest in standardised (log m, log pT_jet, N particles) with weights
10 : 3 : 1 (mass matched most tightly), with replacement.
Edits (applied to the stored (deta, dphi) of the real particles; pT and particle masses kept; four-vectors and per-particle
features rebuilt):
  rec      recentre: every particle translated by -c (pT centroid moved onto the stored axis). Pair distances unchanged;
           M2 -> M2 - |c|^2 M0.
  ref      reflect: translated by +c (offset doubled). M2 -> M2 + 3 |c|^2 M0.
  rot      rotation of (deta, dphi) about the stored axis by a per-jet angle theta ~ U(60, 300) deg. Pair distances, |c|, M0, M2
           unchanged (the centroid moves to another direction at the same distance).
  rs       random rigid shift of length |c| in a per-jet uniformly random direction. M2 -> M2 + 2 |c|^2 M0 cos(a) + |c|^2 M0.
  dil_rec, dil_ref, dil_rs   matched real change: dilation of all angles about the stored axis by k = sqrt(M2_edit / M2), which
           gives the same change of 1/2 log M2 as rec / ref / rs but is a real change of the jet's invariant mass (P1/P23 method).
Read-outs: (a) W-vs-Z log-odds l_W - l_Z (logits); (b) class-representation mass probe: ridge regression of log m on the input
of the final linear layer (ParT: class token; ParticleNet: pooled EdgeConv features), fitted on the unedited jets outside
tail5 (3/4 fit, 1/4 to choose alpha), evaluated on all edits (L-GATr: logits only).
Ratios: direct R(e) = mean d(e) / mean d(dil_e) over a jet set (d = read-out change vs base). Primary (steadier when the
matched change is small): kappa = slope through 0 of d(dilation) on its predicted 1/2 dlog M2, pooled over dil_rec, dil_ref,
dil_rs on the same jets (P23 method); Rk(e) = mean d(e) / (kappa mean 1/2 dlog M2(e)); R_nat = Rk(rec). R = 1: the network
responds to recentring as to a real mass change of the same 1/2 dlog M2; R = 0: no response. Rk(rot) uses the recentring
prediction as its scale (the rotation itself predicts 0).
Controls / tests: calls changed (10-class argmax) under rec vs rot on the same jets (exact McNemar on discordant per-jet call
changes); W->Z (true W called Z) and Z->W flows; H0 mean d(rec) = mean d(rot): sign-flip permutation of per-jet d(rec) - d(rot);
R_nat vs R_rs: bootstrap p; R_nat (trained) vs R_nat (untrained ParT): bootstrap p of the difference; observational slope:
(l_W - l_Z)(tail jet) - (l_W - l_Z)(matched low-|c| jet) on unedited jets vs the edit prediction -d(rec). 95 % bootstrap over jets
(NB resamples); Holm over models x tails within each test family.
Checks: invariant-mass change per edit (relative, all jets), pairwise dR change from rebuilt four-vectors (rec, ref, rot, rs; 400
jets), unedited accuracy on this sample and on the P1 sample (rows 3,000-4,000 of each _100 file; ParT_full 85.9 %, ParT_kin 74.5 %).

Stages (run as separate cluster jobs; selection and edits are deterministic, so the stages agree jet by jet):
  ./sync.sh && ./part_job.sh c-p36-run 'python -m interp.p36_natural_offset --stage nets'
  ./sync.sh && ./part_job.sh c-p36-lg2 '/j-jepa-vol/aaron-interp/env-lgatr/bin/python -m interp.p36_natural_offset --stage lgatr'
       (GPU with xformers block-diagonal attention as in training; a CPU run (dense attention mask) gave chance accuracy)
  ./job.sh c-p36-an 2 6 'python -m interp.p36_natural_offset --stage analyze'
  -> $AI_RESULTS/part/p36_nets.npz, $AI_RESULTS/lgatr/p36_lgatr.npz, $AI_RESULTS/part/p36_natural_offset.json, p36_natural_offset.png
"""
import argparse, glob, json, math, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ap = argparse.ArgumentParser(); ap.add_argument('--stage', required=True, choices=['nets', 'lgatr', 'analyze'])
ap.add_argument('--smoke', action='store_true'); ARGS = ap.parse_args()
SMOKE = ARGS.smoke
SEED, SEED_EDIT = 3636, 3637
PER = 150 if SMOKE else 2000
NB = 300 if SMOKE else 2000
NPERM = 300 if SMOKE else 5000
N_LG_RAND = 300 if SMOKE else 2000
TAILS = {'tail1': 0.01, 'tail2': 0.02, 'tail5': 0.05}
TEST = Path('/j-jepa-vol/JetClass/Pythia/test_20M')
PRE = {'QCD': 'ZJetsToNuNu', 'Hbb': 'HToBB', 'Hcc': 'HToCC', 'Hgg': 'HToGG', 'H4q': 'HToWW4Q', 'Hqql': 'HToWW2Q1L', 'Zqq': 'ZToQQ',
       'Wqq': 'WToQQ', 'Tbqq': 'TTBar', 'Tbl': 'TTBarLep'}
CL = ['QCD', 'Hbb', 'Hcc', 'Hgg', 'H4q', 'Hqql', 'Zqq', 'Wqq', 'Tbqq', 'Tbl']
iW, iZ = CL.index('Wqq'), CL.index('Zqq')
EDITS = ['base', 'rec', 'ref', 'rot', 'rs', 'dil_rec', 'dil_ref', 'dil_rs']
RES = Path(os.environ.get('AI_RESULTS', 'results'))
TAG = '_smoke' if SMOKE else ''
NETS_NPZ = RES / 'part' / f'p36_nets{TAG}.npz'
LG_NPZ = RES / 'lgatr' / f'p36_lgatr{TAG}.npz'
OUT_JSON = RES / 'part' / f'p36_natural_offset{TAG}.json'
OUT_PNG = RES / 'part' / f'p36_natural_offset{TAG}.png'
LG_RUN = '/j-jepa-vol/aaron-interp/models/lgatr/runs/jc2m/jc2m_s0a/models'
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


# ------------------------------------------------------------------------------------------------ sample, selection, edits
def load_sample():
    import uproot
    from interp.part_core import read_root, sub, cat
    rng = np.random.default_rng(SEED); parts, src = [], []
    for k, c in enumerate(CL):
        path = TEST / f'{PRE[c]}_118.root'; N = uproot.open(path)['tree'].num_entries
        if SMOKE: N = min(N, 20000)
        rows = np.sort(rng.choice(N, PER, replace=False)); ch = []
        for s in range(0, N, 25000):
            r = rows[(rows >= s) & (rows < s + 25000)]
            if len(r): ch.append(sub(read_root(path, s, min(s + 25000, N)), r - s))
        J = cat(ch); assert np.all(J['label'] == k), (c, np.unique(J['label'])); parts.append(J); src.append(np.stack([np.full(PER, k), rows], 1))
        log(f'read {c}: {PER} rows of {path.name} ({N} entries)')
    J = cat(parts); src = np.concatenate(src); keep = (J['mask'] > 0).sum(1) >= 2
    from interp.part_core import sub as _sub
    return _sub(J, np.where(keep)[0]), src[keep]


def selection(J):
    """per-jet c, |c|, moments, tails and matched low-|c| partners (deterministic from the jets)"""
    from interp.part_core import kin, jmass
    pt, de, dp, ms = kin(J); pt64 = pt.astype(np.float64)
    M0 = pt64.sum(1); M1e = (pt64 * de).sum(1); M1p = (pt64 * dp).sum(1); M2 = (pt64 * (de.astype(np.float64) ** 2 + dp.astype(np.float64) ** 2)).sum(1)
    ce, cp = M1e / M0, M1p / M0; cabs = np.hypot(ce, cp); y = J['label'].astype(int); n = len(y)
    m = jmass(J).astype(np.float64); ptj = J['jet_pt'].astype(np.float64); npart = (J['mask'] > 0).sum(1)
    S = dict(ce=ce, cp=cp, cabs=cabs, M0=M0, M2=M2, m=m, ptj=ptj, npart=npart, y=y)
    for t, q in TAILS.items():
        f = np.zeros(n, bool)
        for k in range(10):
            ik = np.where(y == k)[0]; nt = max(1, int(round(q * len(ik)))); f[ik[np.argsort(-cabs[ik], kind='stable')[:nt]]] = True
        S[t] = f
    feat = np.stack([np.log(np.maximum(m, 1.0)), np.log(ptj), npart.astype(float)], 1); feat = (feat - feat.mean(0)) / feat.std(0) * np.array([10.0, 3.0, 1.0])   # mass matched most tightly
    match = np.full(n, -1)
    for k in range(10):
        ik = np.where(y == k)[0]; low = ik[cabs[ik] <= np.median(cabs[ik])]; tl = ik[S['tail5'][ik]]
        dd = ((feat[tl][:, None, :] - feat[low][None, :, :]) ** 2).sum(2); match[tl] = low[dd.argmin(1)]
    S['match'] = match
    r = np.random.default_rng(SEED_EDIT); S['theta'] = np.radians(r.uniform(60, 300, n)); S['alpha'] = r.uniform(0, 2 * np.pi, n)
    S['M2_rec'] = M2 - 2 * (ce * M1e + cp * M1p) + cabs ** 2 * M0
    S['M2_ref'] = M2 + 2 * (ce * M1e + cp * M1p) + cabs ** 2 * M0
    rse, rsp = cabs * np.cos(S['alpha']), cabs * np.sin(S['alpha']); S['rse'], S['rsp'] = rse, rsp
    S['M2_rs'] = M2 + 2 * (rse * M1e + rsp * M1p) + cabs ** 2 * M0
    return S, (pt, de, dp, ms)


def make_edit(J, K, S, name, rows=None):
    from interp.part_core import build, sub
    pt, de, dp, ms = K
    if rows is not None:
        J = sub(J, rows); pt, de, dp, ms = pt[rows], de[rows], dp[rows], ms[rows]; S = {k: (v[rows] if np.ndim(v) else v) for k, v in S.items()}
    r = (pt > 0).astype(np.float32); col = lambda a: a[:, None].astype(np.float32)
    if name == 'base': return build(J, pt, de, dp, ms)
    if name == 'rec': return build(J, pt, de - col(S['ce']) * r, dp - col(S['cp']) * r, ms)
    if name == 'ref': return build(J, pt, de + col(S['ce']) * r, dp + col(S['cp']) * r, ms)
    if name == 'rs': return build(J, pt, de + col(S['rse']) * r, dp + col(S['rsp']) * r, ms)
    if name == 'rot':
        co, si = col(np.cos(S['theta'])), col(np.sin(S['theta'])); return build(J, pt, (co * de - si * dp) * r, (si * de + co * dp) * r, ms)
    if name.startswith('dil_'):
        k = col(np.sqrt(np.maximum(S['M2_' + name[4:]], 1e-30) / S['M2'])); return build(J, pt, de * k, dp * k, ms)
    raise ValueError(name)


def geometry_checks(Je, Jb, rows):
    """max |change| of pairwise dR^2 from the rebuilt four-vectors (rows only) and relative invariant-mass change (all jets)"""
    from interp.part_core import jmass
    def etaphi(J):
        m = J['mask'][rows] > 0; pt = np.hypot(J['px'][rows], J['py'][rows]).astype(np.float64)
        eta = np.arcsinh(np.where(m, J['pz'][rows], 0) / np.where(m, pt, 1)); phi = np.arctan2(J['py'][rows], J['px'][rows]).astype(np.float64)
        return eta, phi, m
    e0, p0, m0 = etaphi(Jb); e1, p1, m1 = etaphi(Je)
    # pT order can change only for the dilation (no change of pT anywhere), so slots correspond
    d0 = (e0[:, :, None] - e0[:, None, :]) ** 2 + np.angle(np.exp(1j * (p0[:, :, None] - p0[:, None, :]))) ** 2
    d1 = (e1[:, :, None] - e1[:, None, :]) ** 2 + np.angle(np.exp(1j * (p1[:, :, None] - p1[:, None, :]))) ** 2
    mm = m0[:, :, None] & m0[:, None, :]
    rel = np.abs(jmass(Je).astype(np.float64) / np.maximum(jmass(Jb).astype(np.float64), 1e-6) - 1)
    ad = np.abs(d1 - d0)[mm]; far = (np.abs(Jb['deta'][rows]) > 1) | (np.abs(Jb['dphi'][rows]) > 1)
    return dict(pair_dR2_maxabs=float(ad.max()), pair_dR2_median_abs=float(np.median(ad)), pair_dR2_p999=float(np.percentile(ad, 99.9)),
                pair_frac_gt_1em4=float((ad > 1e-4).mean()), n_pairs=int(mm.sum()), jets_with_particle_beyond_1rad=int(far.any(1).sum()),
                mass_rel_median=float(np.median(rel)), mass_rel_p99=float(np.percentile(rel, 99)), mass_rel_max=float(rel.max()))


# ------------------------------------------------------------------------------------------------ stage: ParT / ParticleNet
def stage_nets():
    import torch
    from interp.part_core import ParTNetwork, inputs, sample, kin, build, jmass
    from interp import pn_net
    from weaver.nn.model.ParticleTransformer import ParticleTransformer
    from jetdistill.part.network import CFG, FEATURES
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; log('device', dev, torch.__version__)
    strip_mod = lambda sd: {k[4:] if k.startswith('mod.') else k: v for k, v in sd.items()}
    M = {}

    def hook(model):
        box = {}; model.fc.register_forward_hook(lambda mod, inp, out: box.__setitem__('h', inp[0].detach())); return box

    def part(name, feat, init=None):
        net = ParTNetwork(feat)
        if init is not None:
            torch.manual_seed(init); fresh = ParticleTransformer(input_dim=len(FEATURES[feat]), **CFG)
            net.model.load_state_dict(fresh.state_dict(), strict=True)
        net.model.to(dev).eval(); M[name] = dict(kind='part', feat=feat, model=net.model, box=hook(net.model))

    part('ParT_full', 'full'); part('ParT_kin', 'kin')
    PND = Path(os.environ.get('JETDISTILL_MODELS', '/j-jepa-vol/aaron-interp/models/part')).parent / 'particlenet'
    for feat in ('full', 'kin'):
        m = pn_net.load_particlenet(PND / f'ParticleNet_{feat}.pt', len(FEATURES[feat]), dev)
        M[f'PN_{feat}'] = dict(kind='pn', feat=feat, model=m, box=hook(m))
    part('ParT_untrained', 'full', init=1)
    names = list(M); log('models', names)

    def run(J, bs=512):
        N = len(J['mask']); order = np.argsort((J['mask'] > 0).sum(1), kind='stable')
        L = {k: np.zeros((N, 10), np.float32) for k in names}; H = {k: None for k in names}
        T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
        for s in range(0, N, bs):
            idx = order[s:s + bs]; Jc = {k: (a[idx] if np.ndim(a) else a) for k, a in J.items()}
            X = {f: inputs(Jc, f) for f in ('full', 'kin')}
            for k in names:
                mm = M[k]; x, v, msk = X[mm['feat']]
                with torch.no_grad():
                    if mm['kind'] == 'part': out = mm['model'](T(x), T(v), T(msk))
                    else:
                        P = x.shape[-1]; pad = (lambda a: np.concatenate([a, np.zeros(a.shape[:2] + (128 - P,), np.float32)], -1)) if P < 128 else (lambda a: a)
                        xp, mp = pad(x), pad(msk); fn = [f[0] for f in FEATURES[mm['feat']]]
                        out = mm['model'](T(xp[:, [fn.index('deta'), fn.index('dphi')]]), T(xp), T(mp))
                L[k][idx] = out.float().cpu().numpy(); h = mm['box']['h'].float().cpu().numpy()
                if H[k] is None: H[k] = np.zeros((N, h.shape[1]), np.float32)
                H[k][idx] = h
        return L, H

    # P1-sample accuracy (known values)
    J1 = sample(100 if SMOKE else 1000, 3000); y1 = J1['label'].astype(int); L1, _ = run(build(J1, *kin(J1)))
    acc_p1 = {k: float((L1[k].argmax(1) == y1).mean()) for k in names}; log('P1-sample accuracy', {k: round(v, 4) for k, v in acc_p1.items()}, 'N', len(y1))
    del J1, L1
    J, src = load_sample(); S, K = selection(J); n = len(S['y']); log(f'sample N={n}; |c| median {np.median(S["cabs"]):.4f}, '
        f'tail5 min {S["cabs"][S["tail5"]].min():.4f}, tail1 min {S["cabs"][S["tail1"]].min():.4f}')
    Lr, _ = run(J); acc_raw = {k: float((Lr[k].argmax(1) == S['y']).mean()) for k in names}; del Lr
    Lall = {k: np.zeros((len(EDITS), n, 10), np.float32) for k in names}; Hb = {}; Hall = {k: [] for k in names}
    geo = {}; grow = np.random.default_rng(1).choice(n, min(400, n), replace=False)
    dlogm = {}; Jb = None
    for e, name in enumerate(EDITS):
        Je = make_edit(J, K, S, name)
        if name == 'base': Jb = Je; mb = jmass(Jb).astype(np.float64)
        else:
            geo[name] = geometry_checks(Je, Jb, grow)
        dlogm[name] = np.log(np.maximum(jmass(Je).astype(np.float64), 1e-3)) - np.log(np.maximum(mb, 1e-3))
        L, H = run(Je)
        for k in names:
            Lall[k][e] = L[k]; Hall[k].append(H[k].astype(np.float16))
        log(f'edit {name:8s} done; ' + ', '.join(f'{k} acc {(L[k].argmax(1) == S["y"]).mean():.4f}' for k in names)
            + (f'; geometry {geo[name]}' if name in geo else ''))
    # mass probe: ridge on the final-layer input, fitted on unedited jets outside tail5
    logm = np.log(np.maximum(mb, 1.0)); out_t = np.where(~S['tail5'])[0]; pr = np.random.default_rng(5).permutation(out_t)
    fit, val = np.sort(pr[: len(pr) * 3 // 4]), np.sort(pr[len(pr) * 3 // 4:])
    probe, probe_info = {}, {}
    for k in names:
        X0 = Hall[k][0].astype(np.float64); mu, sd = X0[out_t].mean(0), X0[out_t].std(0); sd[sd < 1e-9] = np.inf
        Z = (X0 - mu) / sd; t = logm; best = None
        for a in 10.0 ** np.arange(-4, 4.1, 0.5):
            Zc = Z[fit] - Z[fit].mean(0); w = np.linalg.solve(Zc.T @ Zc + a * len(fit) * np.eye(Z.shape[1]), Zc.T @ (t[fit] - t[fit].mean()))
            b0 = t[fit].mean() - Z[fit].mean(0) @ w; err = ((Z[val] @ w + b0 - t[val]) ** 2).mean()
            if best is None or err < best[0]: best = (err, a, w, b0)
        _, a, w, b0 = best; r2v = 1 - best[0] / t[val].var()
        probe[k] = np.stack([((Hall[k][e].astype(np.float64) - mu) / sd) @ w + b0 for e in range(len(EDITS))]).astype(np.float32)
        probe_info[k] = dict(alpha=float(a), r2_val=float(r2v), n_fit=int(len(fit)), n_val=int(len(val)), dim=int(Z.shape[1]))
        log(f'probe {k}: held-out R2 {r2v:.3f} (alpha {a:g})')
    ex = np.where(S['tail1'])[0]; pt, de, dp, _ = K
    NETS_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(NETS_NPZ, names=np.array(names), edits=np.array(EDITS), src=src, y=S['y'],
                        **{f'S_{k}': v for k, v in S.items()}, **{f'L_{k}': v for k, v in Lall.items()}, **{f'P_{k}': v for k, v in probe.items()},
                        **{f'dlogm_{k}': v for k, v in dlogm.items()}, ex_idx=ex, ex_pt=pt[ex], ex_de=de[ex], ex_dp=dp[ex])
    meta = dict(acc_p1=acc_p1, acc_raw=acc_raw, geometry=geo, probe=probe_info, device=dev, N=int(n))
    (NETS_NPZ.parent / f'p36_nets_meta{TAG}.json').write_text(json.dumps(meta, indent=1)); log('wrote', NETS_NPZ)


# ------------------------------------------------------------------------------------------------ stage: L-GATr
def stage_lgatr():
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    from interp.p5_lgatr_launch import load_lgatr, logits
    from interp.part_core import sample, kin, build
    c = sorted(glob.glob(f'{LG_RUN}/*.pt'), key=lambda p: int(p.split('_it')[-1].split('.')[0])); ck = c[-1]
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    log('checkpoint', ck, 'device', dev); model, cfg = load_lgatr(ck, dev); log('loaded L-GATr; force_xformers', model.force_xformers)
    J1 = sample(50 if SMOKE else 300, 3000); y1 = J1['label'].astype(int); J1b = build(J1, *kin(J1)); L1 = logits(model, cfg, J1b, dev)
    acc_p1 = float((L1.argmax(1) == y1).mean()); log(f'P1-sample accuracy (rows 3000..{3000 + (50 if SMOKE else 300)} of _100) {acc_p1:.4f} N {len(y1)}')
    # diagnostic: the dense ('materialized') attention-mask path the authors' wrapper uses without xformers (a CPU run of this
    # stage gave chance accuracy, 0.100 on 3,000 P1 jets)
    fx = model.force_xformers; model.force_xformers = False
    try:
        L1d = logits(model, cfg, J1b, dev); log(f'diagnostic dense-mask path: accuracy {(L1d.argmax(1) == y1).mean():.4f}, max |dlogit| vs xformers {np.abs(L1d - L1).max():.3g}')
    except Exception as ex: log('diagnostic dense-mask path failed:', repr(ex)[:300])
    model.force_xformers = fx
    J, src = load_sample(); S, K = selection(J); n = len(S['y'])
    rnd = np.random.default_rng(7).choice(n, min(N_LG_RAND, n), replace=False)
    rows = np.unique(np.concatenate([np.where(S['tail5'])[0], S['match'][S['tail5']], rnd])); log(f'L-GATr subset {len(rows)} jets')
    Lall = np.full((len(EDITS), n, 10), np.nan, np.float32)
    for e, name in enumerate(EDITS):
        Je = make_edit(J, K, S, name, rows); Lall[e, rows] = logits(model, cfg, Je, dev, bs=128)
        log(f'edit {name:8s} acc on subset {(Lall[e, rows].argmax(1) == S["y"][rows]).mean():.4f}')
    LG_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(LG_NPZ, L=Lall, rows=rows, rand=np.sort(rnd), y=S['y'], cabs=S['cabs'], ckpt=np.array(ck), acc_p1=acc_p1, n_p1=len(y1))
    log('wrote', LG_NPZ)


# ------------------------------------------------------------------------------------------------ stage: analysis
def ci(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) else [None, None]


def p_boot(bs, v0=0.0):
    bs = np.asarray(bs, float); bs = bs[np.isfinite(bs)]
    return float(max(1.0 / len(bs), min(1.0, 2 * min((bs <= v0).mean(), (bs >= v0).mean())))) if len(bs) else None


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.zeros(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj.tolist()


def mcnemar(a, b):
    from scipy.stats import binomtest
    x = int((a & ~b).sum()); z = int((~a & b).sum())
    return dict(only_first=x, only_second=z, p=(float(binomtest(x, x + z, 0.5).pvalue) if x + z else 1.0))


def stage_analyze():
    D = np.load(NETS_NPZ, allow_pickle=False); names = [str(x) for x in D['names']]; E = {str(x): i for i, x in enumerate(D['edits'])}
    meta = json.loads((NETS_NPZ.parent / f'p36_nets_meta{TAG}.json').read_text())
    y = D['y']; n = len(y); S = {k[2:]: D[k] for k in D.files if k.startswith('S_')}
    L = {k: D[f'L_{k}'] for k in names}; P = {k: D[f'P_{k}'] for k in names}
    lg = None
    if LG_NPZ.exists():
        G = np.load(LG_NPZ, allow_pickle=False); assert np.array_equal(G['y'], y) and np.allclose(G['cabs'], S['cabs'])
        L['LGATr'] = G['L']; lg = dict(ckpt=str(G['ckpt']), acc_p1=float(G['acc_p1']), n_p1=int(G['n_p1']), rows=G['rows'], rand=G['rand'])
        names = names + ['LGATr']
    rng = np.random.default_rng(11)
    half = lambda a: 0.5 * np.log(np.maximum(a, 1e-30) / S['M2'])
    pred = {'rec': half(S['M2_rec']), 'ref': half(S['M2_ref']), 'rs': half(S['M2_rs']), 'rot': np.zeros(n)}
    sets = {t: S[t] for t in TAILS}
    lowm = np.zeros(n, bool); lowm[S['match'][S['tail5']]] = True; sets['low_matched'] = lowm; sets['all'] = np.ones(n, bool)
    out = dict(node='P36', N=int(n), per_class=PER, seed=SEED, seed_edit=SEED_EDIT, nb=NB, nperm=NPERM,
               sample='JetClass test_20M <class>_118.root, PER random rows per class (default_rng(3636)); jets with >= 2 particles',
               nets_meta=meta, lgatr=(None if lg is None else {k: v for k, v in lg.items() if k in ('ckpt', 'acc_p1', 'n_p1')}),
               sets={}, models={})
    # set descriptions
    for s, f in sets.items():
        idx = np.where(f)[0]
        out['sets'][s] = dict(n=int(len(idx)), cabs_median=float(np.median(S['cabs'][idx])), cabs_min=float(S['cabs'][idx].min()),
                              cabs_max=float(S['cabs'][idx].max()), mass_median=float(np.median(S['m'][idx])),
                              frac_c2M0_over_M2_median=float(np.median((S['cabs'] ** 2 * S['M0'] / S['M2'])[idx])),
                              pred_half_dlogM2={e: float(pred[e][idx].mean()) for e in ('rec', 'ref', 'rs')},
                              dlogm_mean={e: float(D[f'dlogm_{e}'][idx].mean()) for e in E if e != 'base'},
                              dlogm_absmax={e: float(np.abs(D[f'dlogm_{e}'][idx]).max()) for e in E if e != 'base'})
    fam = {'mcnemar_rec_vs_rot': [], 'perm_rec_vs_rot': [], 'R_vs_untrained': []}
    for k in names:
        Lk = L[k]; mod = {}
        avail = np.all(np.isfinite(Lk[0]), 1)
        for s, f0 in sets.items():
            f = f0 & avail
            if k == 'LGATr' and s == 'all': f = np.zeros(n, bool); f[lg['rand']] = True
            idx = np.where(f)[0]; m_ = len(idx)
            if m_ < 5: continue
            nb = NB if m_ <= 5000 else 500; npm = NPERM if m_ <= 5000 else 500
            Wb = np.stack([rng.integers(0, m_, m_) for _ in range(nb)])
            c0 = Lk[E['base'], idx].argmax(1); yy = y[idx]; r = dict(n=int(m_), acc_base=float((c0 == yy).mean()))
            ch = {}
            for e in ('rec', 'ref', 'rot', 'rs', 'dil_rec', 'dil_ref', 'dil_rs'):
                ce = Lk[E[e], idx].argmax(1); x = (ce != c0); ch[e] = x
                r[f'changed_{e}'] = dict(v=float(x.mean()), ci=ci(x[Wb].mean(1)))
                isW, isZ = yy == iW, yy == iZ
                r[f'flow_{e}'] = dict(W_to_Z=[float((c0[isW] == iZ).mean()) if isW.any() else None, float((ce[isW] == iZ).mean()) if isW.any() else None],
                                      Z_to_W=[float((c0[isZ] == iW).mean()) if isZ.any() else None, float((ce[isZ] == iW).mean()) if isZ.any() else None],
                                      nW=int(isW.sum()), nZ=int(isZ.sum()), acc_after=float((ce == yy).mean()))
            r['mcnemar_rec_vs_rot'] = mcnemar(ch['rec'], ch['rot']); r['mcnemar_ref_vs_rot'] = mcnemar(ch['ref'], ch['rot'])
            # W-vs-Z log-odds on the true W and Z jets of the set (as P23); probe on all jets of the set
            wz = np.isin(yy, [iW, iZ]); ro = {'logodds_WZ': (lambda e, ix: (Lk[E[e], ix, iW] - Lk[E[e], ix, iZ]).astype(np.float64), wz)}
            if k in P: ro['probe_logm'] = (lambda e, ix: P[k][E[e], ix].astype(np.float64), np.ones(m_, bool))
            r['readouts'] = {}
            for rn, (fn0, sel) in ro.items():
                ix = idx[sel]; mr = len(ix)
                if mr < 5: continue
                Wr = np.stack([rng.integers(0, mr, mr) for _ in range(nb)]); fn = lambda e: fn0(e, ix)
                sg = rng.choice([-1.0, 1.0], size=(npm, mr))
                b = fn('base'); d = {e: fn(e) - b for e in ('rec', 'ref', 'rot', 'rs', 'dil_rec', 'dil_ref', 'dil_rs')}
                rr = {f'd_{e}': dict(v=float(d[e].mean()), ci=ci(d[e][Wr].mean(1))) for e in d}
                for e in ('rec', 'ref', 'rs'):
                    num, den = d[e][Wr].mean(1), d['dil_' + e][Wr].mean(1)
                    rr[f'R_{e}'] = dict(v=float(d[e].mean() / d['dil_' + e].mean()), ci=ci(num / den), _b=num / den)
                    # rotation-subtracted
                    rr[f'R_{e}_minus_rot'] = dict(v=float((d[e].mean() - d['rot'].mean()) / d['dil_' + e].mean()),
                                                  ci=ci((num - d['rot'][Wr].mean(1)) / den))
                # kappa calibration (P23): slope through 0 of d(dilation) on its predicted 1/2 dlog M2, pooled over the three matched
                # dilations on the same jets; R_kappa(e) = mean d(e) / (kappa * mean pred(e)). Steadier than the direct ratio when
                # the matched change is small.
                pr = {e: pred[e][ix] for e in ('rec', 'ref', 'rs')}
                kn = sum(d['dil_' + e] * pr[e] for e in pr); kd = sum(pr[e] ** 2 for e in pr)
                kap = kn.sum() / kd.sum(); kap_b = kn[Wr].sum(1) / kd[Wr].sum(1)
                rr['kappa'] = dict(v=float(kap), ci=ci(kap_b))
                for e in ('rec', 'ref', 'rs', 'rot'):
                    pe = pr.get(e, pred['rec'][ix])          # rot: scaled by the recentring prediction for reference
                    num = d[e][Wr].mean(1); den = kap_b * pe[Wr].mean(1)
                    rr[f'Rk_{e}'] = dict(v=float(d[e].mean() / (kap * pe.mean())), ci=ci(num / den), _b=num / den)
                rr['Rk_rec_minus_Rk_rs'] = dict(v=rr['Rk_rec']['v'] - rr['Rk_rs']['v'], ci=ci(rr['Rk_rec']['_b'] - rr['Rk_rs']['_b']),
                                                p=p_boot(rr['Rk_rec']['_b'] - rr['Rk_rs']['_b']))
                rr['Rk_rec_minus_Rk_rot'] = dict(v=rr['Rk_rec']['v'] - rr['Rk_rot']['v'], ci=ci(rr['Rk_rec']['_b'] - rr['Rk_rot']['_b']),
                                                 p=p_boot(rr['Rk_rec']['_b'] - rr['Rk_rot']['_b']))
                rr['R_rec_minus_R_rs'] = dict(v=rr['R_rec']['v'] - rr['R_rs']['v'], ci=ci(rr['R_rec']['_b'] - rr['R_rs']['_b']),
                                              p=p_boot(rr['R_rec']['_b'] - rr['R_rs']['_b']))
                # H0 mean d(rec) = mean d(rot): sign-flip permutation of per-jet differences
                dd = d['rec'] - d['rot']
                rr['perm_rec_vs_rot'] = dict(diff=float(dd.mean()), ci=ci(dd[Wr].mean(1)),
                                             p=float((1 + (np.abs(sg @ dd / mr) >= abs(dd.mean())).sum()) / (1 + npm)))
                d0 = d['rec']; rr['perm_rec_vs_0'] = dict(p=float((1 + (np.abs(sg @ d0 / mr) >= abs(d0.mean())).sum()) / (1 + npm)))
                # calibration size
                rr['kappa_dil_rec_per_unit'] = float(d['dil_rec'].mean() / pred['rec'][ix].mean())
                rr['_Wb'] = Wr; rr['_idx'] = ix; rr['n'] = int(mr)
                r['readouts'][rn] = rr
            # observational pairs (tail sets only): unedited difference tail - matched low partner vs edit prediction -d(rec)
            if s.startswith('tail'):
                mi = S['match'][idx]; ok = (mi >= 0) & np.isin(y[idx], [iW, iZ])          # true W and Z jets, as the read-out
                if k == 'LGATr': ok &= np.isin(mi, lg['rows'])
                ii, jj = idx[ok], mi[ok]; Wo = np.stack([rng.integers(0, len(ii), len(ii)) for _ in range(NB)])
                lo = lambda e, a: (Lk[E[e], a, iW] - Lk[E[e], a, iZ]).astype(np.float64)
                obs = lo('base', ii) - lo('base', jj); edp = -(lo('rec', ii) - lo('base', ii))
                dP = 0.5 * np.log(S['M2'][ii] / S['M2'][jj]); dlm = np.log(S['m'][ii] / np.maximum(S['m'][jj], 1e-3))
                r['observational'] = dict(n=int(len(ii)), obs_diff_logodds=dict(v=float(obs.mean()), ci=ci(obs[Wo].mean(1))),
                                          edit_pred_minus_drec=dict(v=float(edp.mean()), ci=ci(edp[Wo].mean(1))),
                                          obs_minus_pred=dict(v=float((obs - edp).mean()), ci=ci((obs - edp)[Wo].mean(1)), p=p_boot((obs - edp)[Wo].mean(1))),
                                          pair_half_dlogM2=float(dP.mean()), pair_dlogm=float(dlm.mean()),
                                          pair_abs_dlogm_median=float(np.median(np.abs(dlm))),
                                          pair_cabs=[float(S['cabs'][ii].mean()), float(S['cabs'][jj].mean())])
            mod[s] = r
        out['models'][k] = mod
    # tests vs untrained, Holm over models x tails
    ref = 'ParT_untrained'
    for fam_name in fam:
        keys, ps = [], []
        for k in names:
            if k == ref and fam_name == 'R_vs_untrained': continue
            for s in TAILS:
                r = out['models'][k].get(s)
                if r is None: continue
                if fam_name == 'mcnemar_rec_vs_rot': p = r['mcnemar_rec_vs_rot']['p']
                elif fam_name == 'perm_rec_vs_rot':
                    if 'logodds_WZ' not in r['readouts']: continue
                    p = r['readouts']['logodds_WZ']['perm_rec_vs_rot']['p']
                else:
                    # paired bootstrap: the untrained ParT's R on the same jets and the same resamples
                    # on the class-representation mass probe (all jets of the set; L-GATr has no probe -> its W/Z log-odds vs the
                    # untrained ParT's W/Z log-odds on the same W/Z jets)
                    if 'probe_logm' in r['readouts']:
                        a = r['readouts']['probe_logm']; idx = a['_idx']; Wb = a['_Wb']; Pu = P[ref]
                        du = lambda e: (Pu[E[e], idx] - Pu[E['base'], idx]).astype(np.float64)
                    else:
                        a = r['readouts']['logodds_WZ']; idx = a['_idx']; Wb = a['_Wb']; Lu = L[ref]
                        du = lambda e: ((Lu[E[e], idx, iW] - Lu[E[e], idx, iZ]) - (Lu[E['base'], idx, iW] - Lu[E['base'], idx, iZ])).astype(np.float64)
                    prr = {e: pred[e][idx] for e in ('rec', 'ref', 'rs')}
                    kn = sum(du('dil_' + e) * prr[e] for e in prr); kd = sum(prr[e] ** 2 for e in prr)
                    ub = du('rec')[Wb].mean(1) / (kn[Wb].sum(1) / kd[Wb].sum(1) * prr['rec'][Wb].mean(1))
                    u0 = du('rec').mean() / (kn.sum() / kd.sum() * prr['rec'].mean())
                    diff = a['Rk_rec']['_b'] - ub; p = p_boot(diff)
                    a['Rk_rec_minus_untrained'] = dict(v=float(a['Rk_rec']['v'] - u0), ci=ci(diff), p=p, untrained_Rk=dict(v=float(u0), ci=ci(ub)))
                keys.append((k, s)); ps.append(p)
        for (k, s), p, ph in zip(keys, ps, holm(ps)):
            out['models'][k][s].setdefault('holm', {})[fam_name] = dict(p=p, p_holm=ph, m=len(ps))
    # print summary
    for k in names:
        for s in list(TAILS) + ['low_matched', 'all']:
            r = out['models'][k].get(s)
            if r is None: continue
            for rn, a in r['readouts'].items():
                log(f'   {k:15s} {s:11s} {rn:10s} n={a["n"]:5d} kappa {a["kappa"]["v"]:+.2f} Rk_rec {a["Rk_rec"]["v"]:+.2f} {np.round(a["Rk_rec"]["ci"], 2)} '
                    f'Rk_rs {a["Rk_rs"]["v"]:+.2f} Rk_ref {a["Rk_ref"]["v"]:+.2f} Rk_rot {a["Rk_rot"]["v"]:+.2f} R_rec(direct) {a["R_rec"]["v"]:+.2f} '
                    f'd_rec {a["d_rec"]["v"]:+.4f} d_rot {a["d_rot"]["v"]:+.4f} perm(rec=0) p {a["perm_rec_vs_0"]["p"]:.3g}')
            log(f'{k:15s} {s:11s} n={r["n"]:5d} acc {r["acc_base"]:.3f} | changed rec {100 * r["changed_rec"]["v"]:5.2f} rot {100 * r["changed_rot"]["v"]:5.2f} '
                f'ref {100 * r["changed_ref"]["v"]:5.2f} rs {100 * r["changed_rs"]["v"]:5.2f} dil_rec {100 * r["changed_dil_rec"]["v"]:5.2f} | '
                f'McN p {r["mcnemar_rec_vs_rot"]["p"]:.2g}')
    figure(D, S, L, out)
    OUT_JSON.write_text(json.dumps(strip(out), indent=1)); log('wrote', OUT_JSON)


def strip(o):
    if isinstance(o, dict): return {str(k): strip(v) for k, v in o.items() if not str(k).startswith('_')}
    if isinstance(o, (list, tuple)): return [strip(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, float) and not math.isfinite(o): return None
    return o


def figure(D, S, L, out):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    E = {str(x): i for i, x in enumerate(D['edits'])}; ex = D['ex_idx']; y = D['y']; Lf = L['ParT_full']
    sm = lambda z: np.exp(z - z.max()) / np.exp(z - z.max()).sum()
    # example: tail1, true W or Z, largest |change of l_W - l_Z| under recentring by ParT_full
    cand = [i for i, j in enumerate(ex) if y[j] in (iW, iZ)] or list(range(len(ex)))
    dl = [abs((Lf[E['rec'], ex[i], iW] - Lf[E['rec'], ex[i], iZ]) - (Lf[E['base'], ex[i], iW] - Lf[E['base'], ex[i], iZ])) for i in cand]
    i = cand[int(np.argmax(dl))]; j = ex[i]; pt, de, dp = D['ex_pt'][i], D['ex_de'][i], D['ex_dp'][i]; mk = pt > 0
    ce, cp = S['ce'][j], S['cp'][j]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    for ax, (tt, sx, sy, e) in zip(axs[:2], [('as recorded (pT centroid off the axis)', 0, 0, 'base'), ('recentred: all particles moved by $-\\vec c$', ce, cp, 'rec')]):
        ax.scatter(de[mk] - sx, dp[mk] - sy, s=8 + 400 * pt[mk] / pt.max(), c='tab:blue', alpha=0.6, edgecolor='none')
        ax.plot(0, 0, 'k+', ms=14, mew=2, label='stored jet axis')
        ax.plot(ce - sx, cp - sy, 'rx', ms=10, mew=2, label='$p_T$ centroid')
        p = sm(Lf[E[e], j].astype(np.float64)); M2 = S['M2'][j] if e == 'base' else S['M2_rec'][j]
        ax.set_title(f'{tt}\nParT_full: P(W) = {p[iW]:.2f}, P(Z) = {p[iZ]:.2f}; $M_2/M_0$ = {M2 / S["M0"][j]:.4f}', fontsize=9.5)
        ax.set_xlabel('$\\Delta\\eta$ from stored axis'); ax.set_ylabel('$\\Delta\\phi$ from stored axis [rad]')
        ax.set_xlim(-0.8, 0.8); ax.set_ylim(-0.8, 0.8); ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc='lower right')
    axs[0].text(0.02, 0.97, f'true {CL[y[j]]}, m = {S["m"][j]:.1f} GeV, $p_T$ = {S["ptj"][j]:.0f} GeV\n$|\\vec c|$ = {S["cabs"][j]:.3f} rad (top 1 % of its class)\ninvariant mass and all pair distances unchanged by the edit',
                 transform=axs[0].transAxes, va='top', fontsize=8)
    ax = axs[2]; mods = [k for k in ('ParT_full', 'ParT_kin', 'PN_full', 'PN_kin', 'LGATr', 'ParT_untrained') if k in out['models']]
    xs = np.arange(len(mods)); w = 0.2
    for off, e, col, lab in [(-1.5 * w, 'rec', 'tab:red', 'recentre: shift by $-\\vec c$'), (-0.5 * w, 'rs', 'tab:orange', 'shift by $|\\vec c|$, random direction'),
                             (0.5 * w, 'dil_rec', 'tab:blue', 'real mass change, same $\\frac{1}{2}\\Delta\\log M_2$ as recentring'),
                             (1.5 * w, 'rot', 'tab:gray', 'rotation about the axis by 60-300 deg')]:
        v = [100 * out['models'][k]['tail2'][f'changed_{e}']['v'] for k in mods]; c_ = [out['models'][k]['tail2'][f'changed_{e}']['ci'] for k in mods]
        ax.bar(xs + off, v, w, color=col, label=lab, yerr=[[vv - 100 * cc[0] for vv, cc in zip(v, c_)], [100 * cc[1] - vv for vv, cc in zip(v, c_)]], capsize=3)
    ax.set_xticks(xs); ax.set_xticklabels([m.replace('_', '\n').replace('LGATr', 'L-GATr') for m in mods], fontsize=8.5); ax.set_ylabel('10-class calls changed [%]')
    ax.set_title(f'top 2 % natural offset per class (N = {out["models"]["ParT_full"]["tail2"]["n"]})', fontsize=9.5); ax.set_ylim(0, 1.45 * ax.get_ylim()[1]); ax.legend(fontsize=7.5, loc='upper left'); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT_PNG, dpi=130); log('wrote', OUT_PNG)


if __name__ == '__main__':
    os.environ.setdefault('JETDISTILL_TAGGER', 'part')
    {'nets': stage_nets, 'lgatr': stage_lgatr, 'analyze': stage_analyze}[ARGS.stage]()
