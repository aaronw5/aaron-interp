"""P5 — L-GATr (Lorentz-equivariant geometric algebra transformer, Spinner et al. 2024/2025) on JetClass: loader + symmetry
check on the P1 rows. Used now (round C1) on the randomly initialised network (architecture-level check) and later (C2/C3) on
the checkpoints of the training run launched with interp/p5_lgatr_job.sh.

Weights: none released (checked 2026-10-19: heidelberg-hepml/lorentz-gatr README and releases, heidelberg-hepml/lgatr,
tagging-guide, lloca-experiments, tagger-quantization, arXiv 2405.14806 / 2411.00446, Zenodo / HuggingFace search). We train
our own with the authors' code (ext/lorentz-gatr @ 35aca2a, config_paper/jctagging) on a 2M-jet JetClass training subset.

Network inputs (authors' embedding, experiments/tagging/embedding.py, used here unchanged):
  multivector channel: each particle's lab-frame four-momentum (E, px, py, pz) / 20 GeV;
  spurion tokens: beam reference 'xyplane' with two beams + the time direction (1, 0, 0, 0); these break the Lorentz group down
    to rotations about the beam axis (and reflections that keep them), so a rigid azimuthal rotation of all particles is an exact
    symmetry of the network; a longitudinal boost is not (it changes the time direction relative to the particles);
  scalars per particle: 6 PID flags/charge, tanh(d0), d0err, tanh(dz), dzerr, plus (add_scalar_features) log pT, log E,
    log pT/pT_jet, log E/E_jet, Δφ, Δη, ΔR about the jet four-momentum sum recomputed from the particles (not the stored axis).
  So the stored jet axis (jet_eta, jet_phi) never enters: the P1 edit "add δ to every particle's Δφ with the stored axis kept",
  which moves ParT's calls, is for L-GATr a rigid rotation of the whole jet about the beam.

Pre-registered predictions (written before any trained checkpoint exists):
  H_sym (by construction): rigid φ rotation by 0.06 rad (lab-frame px, py rotated; and the P1 build edit) -> calls changed at
    float noise (< 0.1 %), |Δlogit| at the level of the batch-composition noise floor, for the random and the trained network.
  η shift by 0.06 at fixed pT and particle mass (P1 edit) and an exact longitudinal boost of rapidity 0.06: not symmetries
    (time reference, log E and log E/E_jet change); invariant mass unchanged (boost) or nearly (η shift). Under the
    invariant-mass hypothesis (H_inv) the W/Z decision moves little: W→Z change ≪ ParT's +17-21 points at 0.06.
  Non-symmetry, iso-invariant-mass edits that change the axis moment M2 (E23 / R1 family, C2/C3): H_inv predicts read-out
    ratio ≈ 0 against a matched real change; ParT gave 0.7-1.4. Alternative H_axis: the Δη/Δφ/ΔR scalar features about the
    recomputed jet axis (or the reference-vector symmetry breaking) let L-GATr learn a non-zero axis-moment response.
  Control: angles × 1.1 about the stored axis (real mass change, Δlog m ≈ +0.094) -> W→Z rises for the trained network.

  python -m interp.p5_lgatr_launch [--ckpt PATH|auto|random] [--per 1000] [--tag NAME]
  (cluster: AI_PY=conda ./job.sh c-p5-run 2 8 'JETDISTILL_TAGGER=part JETDISTILL_DATA=/j-jepa-vol/aaron-interp/code/cluster/data_part
   /j-jepa-vol/aaron-interp/env-lgatr/bin/python -m interp.p5_lgatr_launch --ckpt random')
-> $AI_RESULTS/lgatr/p5_<tag>.json
"""
import argparse, glob, json, os, sys, time
from pathlib import Path
import numpy as np

V = '/j-jepa-vol/aaron-interp'
LG = os.environ.get('LGATR_SRC', f'{V}/ext/lorentz-gatr')
RUNS = f'{V}/models/lgatr/runs/jc2m'
# lorentz-gatr class order -> part_core CL order (QCD, Hbb, Hcc, Hgg, H4q, Hqql, Zqq, Wqq, Tbqq, Tbl)
LG_CLASSES = ['ZJetsToNuNu', 'HToBB', 'HToCC', 'HToGG', 'HToWW4Q', 'HToWW2Q1L', 'TTBar', 'TTBarLep', 'WToQQ', 'ZToQQ']
TO_CL = [0, 1, 2, 3, 4, 5, 9, 8, 6, 7]
t0 = time.time()
log = lambda *a: print(f'[{time.time() - t0:7.1f}s]', *a, flush=True)


def load_lgatr(ckpt=None, device='cpu'):
    """(model, cfg). ckpt None -> random init (torch seed 0) from config_paper/jctagging; otherwise a .pt from a run directory
    whose config.yaml is next to models/. The authors' experiment class sets the derived channel counts (num_global_tokens,
    out_mv_channels, in_s_channels) and the geometric-algebra flags exactly as in training."""
    import torch
    sys.path.insert(0, LG); cwd = os.getcwd(); os.chdir(LG)
    try:
        from omegaconf import OmegaConf
        from hydra import initialize_config_dir, compose
        from hydra.utils import instantiate
        import experiments.base_experiment  # noqa: F401  (registers base_attention / base_mlp)
        from experiments.tagging.jetclassexperiment import JetClassTaggingExperiment
        if ckpt is None:
            with initialize_config_dir(config_dir=f'{LG}/config_paper', version_base=None):
                cfg = compose(config_name='jctagging')
        else:
            cfg = OmegaConf.load(Path(ckpt).parents[1] / 'config.yaml')
        exp = JetClassTaggingExperiment(cfg, 0, 1)
        exp.init_physics(); exp.init_geometric_algebra()
        torch.manual_seed(0)
        model = instantiate(exp.cfg.model)
        if ckpt is not None:
            sd = torch.load(ckpt, map_location='cpu')['model']; model.load_state_dict(sd)
        model.force_xformers = device == 'cuda' and bool(exp.cfg.training.force_xformers)
        return model.to(device).eval(), exp.cfg
    finally:
        os.chdir(cwd)


def to_lgatr_inputs(J):
    """jet dict (part_core / read_root) -> (pf_vectors (B, 4, 128) = E, px, py, pz; pf_features (B, 10, 128)) as the authors'
    miniweaver/default.yaml builds them: charge, isChargedHadron, isNeutralHadron, isPhoton, isElectron, isMuon, tanh(d0val),
    clip(d0err, 0, 1), tanh(dzval), clip(dzerr, 0, 1); padded slots zero."""
    m = (J['mask'] > 0).astype(np.float32)
    vec = np.stack([J['energy'], J['px'], J['py'], J['pz']], 1) * m[:, None]
    f = [J['charge'], J['isChargedHadron'], J['isNeutralHadron'], J['isPhoton'], J['isElectron'], J['isMuon'],
         np.tanh(J['d0val']), np.clip(J['d0err'], 0, 1), np.tanh(J['dzval']), np.clip(J['dzerr'], 0, 1)]
    return vec.astype(np.float32), (np.stack(f, 1) * m[:, None]).astype(np.float32)


def logits(model, cfg, J, device, bs=128, order=None):
    """(N, 10) logits in part_core CL order; `order` permutes the jets inside the batches (batch-composition noise floor)"""
    import torch
    sys.path.insert(0, LG)
    from experiments.tagging.embedding import dense_to_sparse_jet, embed_tagging_data_into_ga
    vec, feat = to_lgatr_inputs(J); N = len(vec); idx = np.arange(N) if order is None else order; out = np.zeros((N, 10), np.float32)
    with torch.no_grad():
        for s in range(0, N, bs):
            r = idx[s:s + bs]
            fm, sc, ptr = dense_to_sparse_jet(torch.from_numpy(vec[r]).to(device), torch.from_numpy(feat[r]).to(device))
            emb = embed_tagging_data_into_ga(fm, sc, ptr, cfg.data)
            out[r] = model(emb).float().cpu().numpy()
    return out[:, TO_CL]


def rotate_lab(J, a):
    """exact azimuthal rotation of every particle's (px, py) by a (stored axis and deta/dphi untouched; L-GATr does not read them)"""
    E = dict(J); c, s = np.cos(a), np.sin(a)
    E['px'] = (c * J['px'] - s * J['py']).astype(np.float32); E['py'] = (s * J['px'] + c * J['py']).astype(np.float32); return E


def boost_z(J, y):
    """exact longitudinal boost of rapidity y applied to every particle's (E, pz)"""
    E = dict(J); m = J['mask'] > 0; ch, sh = np.cosh(y), np.sinh(y)
    E['energy'] = np.where(m, ch * J['energy'] + sh * J['pz'], 0).astype(np.float32)
    E['pz'] = np.where(m, sh * J['energy'] + ch * J['pz'], 0).astype(np.float32); return E


def boot(x, B=2000, seed=0):
    r = np.random.default_rng(seed); n = len(x); bs = [x[r.integers(0, n, n)].mean() for _ in range(B)]
    return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def compare(L0, L1, y, iW, iZ):
    from scipy.stats import binomtest
    p0, p1 = L0.argmax(1), L1.argmax(1); ch = (p0 != p1).astype(float)
    c0, c1 = p0 == y, p1 == y; b, c = int((c0 & ~c1).sum()), int((~c0 & c1).sum())
    W, Z = y == iW, y == iZ
    d = (L1[:, iW] - L1[:, iZ]) - (L0[:, iW] - L0[:, iZ])
    return dict(calls_changed=boot(ch), acc_before=float(c0.mean()), acc_after=float(c1.mean()),
                mcnemar_b_c=[b, c], mcnemar_p=float(binomtest(b, b + c).pvalue) if b + c else 1.0,
                W_to_Z=[float((p0[W] == iZ).mean()), float((p1[W] == iZ).mean())],
                Z_to_W=[float((p0[Z] == iW).mean()), float((p1[Z] == iW).mean())],
                d_logitWZ_on_W=boot(d[W]) if W.any() else None,
                max_abs_dlogit=float(np.abs(L1 - L0).max()), median_abs_dlogit=float(np.median(np.abs(L1 - L0))))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ckpt', default='auto'); ap.add_argument('--per', type=int, default=1000)
    ap.add_argument('--tag', default=None); a = ap.parse_args()
    import torch
    from interp.part_core import sample, kin, build, jmass, CL
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = a.ckpt
    if ck == 'auto':
        c = sorted(glob.glob(f'{RUNS}/*/models/*.pt'), key=os.path.getmtime); ck = c[-1] if c else 'random'
    ckpt = None if ck == 'random' else ck
    tag = a.tag or ('randinit' if ckpt is None else 'ckpt'); OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'lgatr'
    OUT.mkdir(parents=True, exist_ok=True)
    log('device', dev, 'torch', torch.__version__, 'checkpoint', ckpt or 'random init (seed 0)')
    model, cfg = load_lgatr(ckpt, dev)
    npar = sum(p.numel() for p in model.parameters()); log('model params', npar, 'force_xformers', model.force_xformers)
    J0 = sample(a.per, 3000); y = J0['label'].astype(int); N = len(y); iW, iZ = CL.index('Wqq'), CL.index('Zqq')
    log('sample', N, 'jets (rows 3000 ..', 3000 + a.per, 'of each class test file 100)')
    pt, de, dp, ms = kin(J0); Jb = build(J0, pt, de, dp, ms)
    L_raw = logits(model, cfg, J0, dev); acc_raw = float((L_raw.argmax(1) == y).mean()); log('accuracy stored jets', round(acc_raw, 4))
    L0 = logits(model, cfg, Jb, dev); log('accuracy rebuilt jets', round(float((L0.argmax(1) == y).mean()), 4))
    perm = np.random.default_rng(1).permutation(N)
    edits = {
        'noise_floor_batch_order': (Jb, perm),
        'rebuild_vs_stored': None,
        'rot_lab_phi_0.06': (rotate_lab(Jb, 0.06), None),
        'p1_build_dphi+0.06': (build(J0, pt, de, dp + 0.06 * (pt > 0), ms), None),
        'p1_build_deta+0.06': (build(J0, pt, de + 0.06 * (pt > 0), dp, ms), None),
        'boost_z_y0.06': (boost_z(Jb, 0.06), None),
        'control_angles_x1.1': (build(J0, pt, de * 1.1, dp * 1.1, ms), None),
    }
    res = dict(node='P5', checkpoint=ckpt or 'random-init-seed0', lgatr_commit='35aca2ac4921b8092b5a14f40deed11e7c015e53',
               n_params=int(npar), device=dev, N=int(N), sample='part_core.sample(per, 3000): rows 3000..3000+per of test file 100',
               acc_stored=acc_raw, acc_rebuilt=float((L0.argmax(1) == y).mean()), edits={})
    m0 = jmass(Jb)
    for k, v in edits.items():
        if v is None: r = compare(L_raw, L0, y, iW, iZ); r['dlogm_mean'] = 0.0
        else:
            Je, order = v; L1 = logits(model, cfg, Je, dev, order=order); r = compare(L0, L1, y, iW, iZ)
            r['dlogm_mean'] = float(np.mean(np.log(np.maximum(jmass(Je), 1e-3)) - np.log(np.maximum(m0, 1e-3))))
        res['edits'][k] = r
        log(f'{k:26s} calls changed {100 * r["calls_changed"][0]:.3f}% [{100 * r["calls_changed"][1]:.3f}, {100 * r["calls_changed"][2]:.3f}]'
            f' max|dlogit| {r["max_abs_dlogit"]:.2e} W->Z {r["W_to_Z"][0]:.3f}->{r["W_to_Z"][1]:.3f} dlog m {r["dlogm_mean"]:+.5f}')
    f = OUT / f'p5_{tag}.json'; f.write_text(json.dumps(res, indent=1)); log('wrote', f)


if __name__ == '__main__':
    main()
