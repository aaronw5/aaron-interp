"""P60 (loop C, round C13; claims C6 v2, C3s, C10 v5): L-GATr retrained with ParT's stored-axis inputs. Is ParT's response to
moving particles off the stored jet axis (at fixed invariant mass) set by the input representation or by the architecture?

Model. L-GATr (Lorentz-equivariant geometric algebra transformer), authors' code ext/lorentz-gatr @ 35aca2a, config_paper/jctagging
(12 blocks, 16 multivector / 32 scalar channels, 8 heads, Lion lr 3e-4, cosine, batch 128, 200,000 iterations), the same 2M-jet
random-file JetClass subset and file ranges as P5, seed 0, run jc2m_sa_s0 (job interp/p60_lgatr_storedaxis_job.sh). One change
(interp/p60_lgatr_storedaxis.patch, applied in the private worktree ext/lorentz-gatr-p60, data.features=storedaxis): of the 7
per-particle scalar features the embedding adds (add_scalar_features), Δφ, Δη and ΔR = sqrt(Δη² + Δφ²) are the JetClass stored
part_dphi, part_deta (about the stored jet_eta / jet_phi; exactly ParT's inputs, read by the loader through
miniweaver/storedaxis.yaml) instead of being recomputed about the axis of the particle four-momentum sum. log pT, log E, log pT/pT_jet,
log E/E_jet, the four-vector multivector inputs, the spurions (beam x-y plane bivector, time direction) and the preprocessing
(mean, factor) are unchanged. With stored_axis_features false (every other run, including P5) the patched code is the authors'.

Stages
  unittest  (run inside the training job before launch, and as a CPU job): (T1) on 200 P1-sample jets (20 per class, rows 3000-3019
            of test file 100) the patched embedding's Δφ, Δη, ΔR columns equal ParT's part_dphi, part_deta, part_deltaR inputs
            (jetdistill.part.network.inputs, 'full') to 1e-6; the other 4 added features and the 10 PID / track columns equal those
            of the default path; (T2) with features=default the patched embedding equals the pristine 35aca2a embedding bit for bit
            (P5's inputs reproduced); (T3) the miniweaver loader with storedaxis.yaml yields 12 feature rows and, per jet, phi(px, py)
            − Δφ and η(p) ∓ Δη are constant (the stored branches belong to the same particles, in the same order); (T4) a random
            stored-axis model gives finite logits.  -> $AI_RESULTS/lgatr/p60_unittest.json; exit 1 on any failure.
  lgatr     L-GATr models (stored-axis run at --its, P5 run at the same iterations, P5 final, untrained stored-axis model) on the
            battery.  part: ParT_full, ParT_kin (released) and the P10 retrains ParT_full_s0, ParT_kin_s0 (same 2M jets).
  analyze   statistics and figure.

Samples. (1) P1 sample: part_core.sample(1000, 3000) = rows 3000-3999 of each class's test file 100, N 10,000: 10-class accuracy and
confusion; its W and Z jets (N 2,000) also get the edits. (2) fresh val_5M WToQQ_122 / ZToQQ_122 (the planned "_128" does not exist:
val_5M holds files 120-124; 121 is L-GATr's validation file, 123 / 124 were used by P46 / P51 / P51v), 2,000 random rows per class,
numpy default_rng(6060), sorted (never first-k), jets with < 2 particles dropped.

Edits (W/Z jets; particle pT, masses, PID and track inputs fixed; r = (Δη, Δφ) about the stored axis):
  PHI±0.06   (a) every particle's Δφ += ±0.06 rad, stored axis kept: the lab four-vectors rotate about the beam by 0.06 rad
             (invariant mass and all pair masses unchanged); stored-axis Δφ inputs shift by 0.06.
  E6±0.06    (b) only the stored jet_phi moved by ±0.06 rad, particles fixed: four-vectors identical, Δφ inputs shift by ∓0.06.
  BOOST±0.06 (c) every particle's rapidity += ±0.06 at fixed pT and mass (exact longitudinal boost, P51), stored axis kept.
  E4|edit    (d) matched dilation: every particle's (Δη, Δφ) × k with k² = M2(edit) / M2(unedited) per jet, M2 = Σ pT r² about the
             stored axis: a real change of the jet mass of the size the edit predicts if the network read the axis moment.
Read-out d = logit_W − logit_Z. R = Σ_jets Δd(edit) / Σ_jets Δd(E4|edit): 0 = no response, 1 = the response to a real mass change
of the same predicted size. Pooled families PHI, E6, BOOST (both signs). L-GATr routes: A full edit; D four-vectors edited, scalar
features held at unedited values; F only the scalar features (incl. stored Δη / Δφ / ΔR) edited.
Statistics: 2,000 bootstrap resamples over jets (paired across models and edits), sign-flip permutation test of the mean Δd
(5,000), exact McNemar on per-jet correctness, Holm over the pre-registered tests.
Pre-registered: H0 (architecture sets it): R(PHI), R(E6) of the stored-axis L-GATr inside [−0.1, 0.1] like P5. H1 (input
representation sets it): R(PHI), R(E6) ≥ 0.3, toward the P10 ParT retrains (0.6-0.8), growing with iteration.

  CPU unit test:  AI_PY=conda ./job.sh c-p60-ut 2 6 'LGATR_SRC=/j-jepa-vol/aaron-interp/ext/lorentz-gatr-p60 \
                  /j-jepa-vol/aaron-interp/env-lgatr/bin/python -m interp.p60_axis_battery --stage unittest'
  battery:        ./part_job.sh c-p60-run 'LGATR_SRC=/j-jepa-vol/aaron-interp/ext/lorentz-gatr-p60 \
                  /j-jepa-vol/aaron-interp/env-lgatr/bin/python -m interp.p60_axis_battery --stage lgatr --its it4999 && \
                  python -m interp.p60_axis_battery --stage part && python -m interp.p60_axis_battery --stage analyze --its it4999'
-> $AI_RESULTS/lgatr/p60_*.{json,npz,png}
"""
import argparse, glob, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('JETDISTILL_TAGGER', 'part')
ap = argparse.ArgumentParser(); ap.add_argument('--stage', required=True); ap.add_argument('--its', default='it4999')
ap.add_argument('--smoke', action='store_true'); ARGS = ap.parse_args()
SMOKE = ARGS.smoke; TAG = '_smoke' if SMOKE else ''
SEED, VFILE = 6060, 122
PER_WZ = 150 if SMOKE else 2000
PER_P1 = 100 if SMOKE else 1000
NB = 200 if SMOKE else 2000
NPERM = 500 if SMOKE else 5000
D = 0.06
V = '/j-jepa-vol/aaron-interp'
LGSRC = os.environ.get('LGATR_SRC', f'{V}/ext/lorentz-gatr-p60')
RUN_SA = f'{V}/models/lgatr/runs/jc2m/jc2m_sa_s0'
RUN_P5 = f'{V}/models/lgatr/runs/jc2m/jc2m_s0a'
SEEDS = f'{V}/models/part_seeds'
VAL = Path('/j-jepa-vol/JetClass/Pythia/val_5M')
CL = ['QCD', 'Hbb', 'Hcc', 'Hgg', 'H4q', 'Hqql', 'Zqq', 'Wqq', 'Tbqq', 'Tbl']
PRE = {'Zqq': 'ZToQQ', 'Wqq': 'WToQQ'}
iW, iZ = CL.index('Wqq'), CL.index('Zqq')
TO_CL = [0, 1, 2, 3, 4, 5, 9, 8, 6, 7]       # lorentz-gatr class order -> CL order
ROOT = Path(os.environ.get('AI_RESULTS', 'results')); OUT = ROOT / 'lgatr'
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


# ------------------------------------------------------------------------------------------------ L-GATr wrapper
def lg_inputs(J, stored):
    """dense (vec (B, 4, 128) = E, px, py, pz; feat (B, 10 [+2], 128)) as the authors' miniweaver default.yaml (+ storedaxis.yaml)"""
    m = (J['mask'] > 0).astype(np.float32)
    vec = np.stack([J['energy'], J['px'], J['py'], J['pz']], 1) * m[:, None]
    f = [J['charge'], J['isChargedHadron'], J['isNeutralHadron'], J['isPhoton'], J['isElectron'], J['isMuon'],
         np.tanh(J['d0val']), np.clip(J['d0err'], 0, 1), np.tanh(J['dzval']), np.clip(J['dzerr'], 0, 1)]
    if stored: f += [J['deta'], J['dphi']]
    return vec.astype(np.float32), (np.stack(f, 1) * m[:, None]).astype(np.float32)


def exp_cfg(cfg_or_overrides):
    """JetClassTaggingExperiment cfg (derived channel counts, stored_axis_features flag) from a saved config or compose overrides"""
    sys.path.insert(0, LGSRC); cwd = os.getcwd(); os.chdir(LGSRC)
    try:
        from omegaconf import OmegaConf
        import experiments.base_experiment  # noqa: F401
        from experiments.tagging.jetclassexperiment import JetClassTaggingExperiment
        if isinstance(cfg_or_overrides, list):
            from hydra import initialize_config_dir, compose
            from hydra.core.global_hydra import GlobalHydra
            GlobalHydra.instance().clear()
            with initialize_config_dir(config_dir=f'{LGSRC}/config_paper', version_base=None):
                cfg = compose(config_name='jctagging', overrides=cfg_or_overrides)
        else:
            cfg = OmegaConf.load(cfg_or_overrides)
        exp = JetClassTaggingExperiment(cfg, 0, 1); exp.init_physics(); exp.init_geometric_algebra(); return exp.cfg
    finally:
        os.chdir(cwd)


class LGX:
    """L-GATr from a run directory (config.yaml) and a checkpoint file (None = random init, torch seed `seed`)"""

    def __init__(self, cfg, ckpt, dev, seed=0):
        import torch
        from hydra.utils import instantiate
        sys.path.insert(0, LGSRC)
        self.cfg = cfg; self.dev = dev
        self.stored = bool('stored_axis_features' in cfg.data and cfg.data.stored_axis_features)
        torch.manual_seed(seed); model = instantiate(cfg.model)
        if ckpt is not None: model.load_state_dict(torch.load(ckpt, map_location='cpu')['model'])
        model.force_xformers = dev == 'cuda' and bool(cfg.training.force_xformers)
        self.model = model.to(dev).eval()
        from experiments.tagging.embedding import dense_to_sparse_jet, embed_tagging_data_into_ga
        self._d2s, self._emb = dense_to_sparse_jet, embed_tagging_data_into_ga

    def embed(self, J, sl):
        import torch
        from interp.part_core import sub
        vec, feat = lg_inputs(sub(J, sl), self.stored)
        fm, sc, ptr = self._d2s(torch.from_numpy(vec).to(self.dev), torch.from_numpy(feat).to(self.dev))
        return self._emb(fm, sc, ptr, self.cfg.data)

    def logits(self, J_mv, J_feat=None, bs=128):
        """four-vectors (multivector inputs) from J_mv; scalar inputs (PID / track + the 7 added features) from J_feat if given"""
        import torch
        N = len(J_mv['label']); out = np.zeros((N, 10), np.float32)
        with torch.no_grad():
            for s in range(0, N, bs):
                sl = np.arange(s, min(s + bs, N)); e = self.embed(J_mv, sl)
                if J_feat is not None:
                    f = self.embed(J_feat, sl); assert f['s'].shape == e['s'].shape and torch.equal(f['batch'], e['batch']); e['s'] = f['s']
                out[sl] = self.model(e).float().cpu().numpy()
        return out[:, TO_CL]


# ------------------------------------------------------------------------------------------------ stage: unittest
def stage_unittest():
    import torch, importlib.util, tempfile
    from interp.part_core import sample, inputs
    sys.path.insert(0, LGSRC)
    res = dict(node='P60', lgatr_src=LGSRC, git_head=subprocess.run(['git', '-C', LGSRC, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
               git_diff_stat=subprocess.run(['git', '-C', LGSRC, 'diff', '--stat'], capture_output=True, text=True).stdout.strip(), tests={})
    cfg_sa = exp_cfg(['data.features=storedaxis']); cfg_df = exp_cfg([])
    res['in_s_channels'] = dict(storedaxis=int(cfg_sa.model.net.in_s_channels), default=int(cfg_df.model.net.in_s_channels))
    from experiments.tagging.embedding import dense_to_sparse_jet, embed_tagging_data_into_ga
    J = sample(20, 3000); N = len(J['label']); log('unit-test jets', N)
    vs, fs = lg_inputs(J, True); vd, fd = lg_inputs(J, False)
    T = lambda a: torch.from_numpy(a.copy())

    def emb(fn, v, f, cfg):
        fm, sc, ptr = dense_to_sparse_jet(T(v), T(f)); e = fn(fm.clone(), sc.clone(), ptr.clone(), cfg.data)
        b = e['batch']; nb = int(b.max()) + 1; cnt = torch.bincount(b, minlength=nb); start = torch.cumsum(cnt, 0) - cnt
        pos = torch.arange(len(b)) - start[b]; return e, pos >= 3                     # 0 global, 1 beam, 2 time spurion, >= 3 particles
    es, ps = emb(embed_tagging_data_into_ga, vs, fs, cfg_sa); ed, pd = emb(embed_tagging_data_into_ga, vd, fd, cfg_df)
    S, Sd = es['s'][ps].numpy(), ed['s'][pd].numpy()
    x, _, mk = inputs(J, 'full'); from jetdistill.part.network import FEATURES
    fn_ = [f[0] for f in FEATURES['full']]; mm = mk[:, 0] > 0
    part = {k: x[:, fn_.index(k)][mm] for k in ('dphi', 'deta', 'deltaR')}
    n_part = int(mm.sum()); assert S.shape[0] == n_part == Sd.shape[0], (S.shape, n_part, Sd.shape)
    # columns: 0-9 PID / track, 10 log pT, 11 log E, 12 log pT rel, 13 log E rel, 14 dphi, 15 deta, 16 dR (x 4 after - 0.2), 17 global flag
    t1 = dict(max_abs_dphi=float(np.abs(S[:, 14] - part['dphi']).max()), max_abs_deta=float(np.abs(S[:, 15] - part['deta']).max()),
              max_abs_dR=float(np.abs(S[:, 16] - part['deltaR']).max() / 4.0),
              max_abs_other_cols_vs_default=float(np.abs(np.delete(S, [14, 15, 16], 1) - np.delete(Sd, [14, 15, 16], 1)).max()),
              n_particles=n_part, n_jets=int(N), s_shape=list(S.shape))
    # dR in physical units: L-GATr column 16 un-preprocessed vs hypot of the stored part_deta, part_dphi (ParT's raw deltaR, before
    # ParT's clip to [-5, 5] after scaling)
    m_ = J['mask'] > 0; dr_st = np.hypot(J['deta'], J['dphi'])[m_].astype(np.float32); dr_lg = S[:, 16] / 4.0 + 0.2
    t1['max_abs_dR_raw_vs_stored_hypot'] = float(np.abs(dr_lg - dr_st).max())
    t1['max_abs_dR_vs_hypot_of_own_cols'] = float(np.abs(dr_lg - np.hypot(S[:, 14], S[:, 15])).max())
    bad = np.abs(S[:, 16] - part['deltaR']) > 4e-6
    t1['n_dR_mismatch_vs_ParT_input'] = int(bad.sum()); t1['ParT_deltaR_input_clipped_at_5'] = int((np.abs(part['deltaR']) >= 5 - 1e-6).sum())
    if bad.any(): t1['examples_mismatch'] = dict(lg=S[bad, 16][:5].tolist(), part=part['deltaR'][bad][:5].tolist(), dr_st=dr_st[bad][:5].tolist())
    t1['pass'] = bool(max(t1['max_abs_dphi'], t1['max_abs_deta'], t1['max_abs_dR_raw_vs_stored_hypot']) < 1e-6 and t1['max_abs_other_cols_vs_default'] < 1e-6
                      and t1['n_dR_mismatch_vs_ParT_input'] == t1['ParT_deltaR_input_clipped_at_5'])
    # how different the authors' own-axis features are (for the record)
    t1['own_axis_vs_stored_median_abs'] = dict(dphi=float(np.median(np.abs(Sd[:, 14] - S[:, 14]))), deta_abs=float(np.median(np.abs(np.abs(Sd[:, 15]) - np.abs(S[:, 15])))))
    res['tests']['T1_stored_features_equal_ParT'] = t1; log('T1', t1)
    # T2: default path == pristine embedding
    src = subprocess.run(['git', '-C', LGSRC, 'show', '35aca2ac4921b8092b5a14f40deed11e7c015e53:experiments/tagging/embedding.py'],
                         capture_output=True, text=True, check=True).stdout
    tmp = Path(tempfile.mkdtemp()) / 'embedding_pristine.py'; tmp.write_text(src)
    spec = importlib.util.spec_from_file_location('embedding_pristine', tmp); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    eo, po = emb(mod.embed_tagging_data_into_ga, vd, fd, cfg_df)
    t2 = dict(s_equal=bool(torch.equal(eo['s'], ed['s'])), mv_equal=bool(torch.equal(eo['mv'], ed['mv'])),
              batch_equal=bool(torch.equal(eo['batch'], ed['batch'])), global_equal=bool(torch.equal(eo['is_global'], ed['is_global'])))
    t2['pass'] = all(t2.values()); res['tests']['T2_default_path_equals_pristine'] = t2; log('T2', t2)
    # T3: miniweaver loader with storedaxis.yaml
    cwd = os.getcwd(); os.chdir(LGSRC)
    try:
        from experiments.tagging.miniweaver.dataset import SimpleIterDataset
        from experiments.tagging.miniweaver.loader import to_filelist
        fpath = str(VAL / f'WToQQ_{VFILE}.root'); fdict, _ = to_filelist([fpath])
        ds = SimpleIterDataset(fdict, 'experiments/tagging/miniweaver/storedaxis.yaml', for_training=False, extra_selection=None,
                               remake_weights=True, load_range_and_fraction=((0, 1), 1), file_fraction=1, fetch_by_files=True,
                               fetch_step=1, infinity_mode=False, in_memory=False, name='p60unittest')
        Xs = []
        for i, (X, y, Z) in enumerate(ds):
            Xs.append((X['pf_features'], X['pf_vectors']))
            if i + 1 >= 200: break
    finally:
        os.chdir(cwd)
    nf = Xs[0][0].shape; bad_phi, bad_eta, sgn_used, maxspread_phi, maxspread_eta = 0, 0, [], 0.0, 0.0
    for F_, V_ in Xs:
        m = np.abs(V_).sum(0) > 0
        if m.sum() < 2: continue
        E, px, py, pz = (V_[k][m].astype(np.float64) for k in range(4))
        phi = np.arctan2(py, px); eta = np.arctanh(pz / np.sqrt(px ** 2 + py ** 2 + pz ** 2))
        a = np.angle(np.exp(1j * (phi - F_[11][m]))); sp = float(np.abs(np.angle(np.exp(1j * (a - a[0])))).max()); maxspread_phi = max(maxspread_phi, sp)
        bad_phi += sp > 1e-4
        e1, e2 = eta - F_[10][m], eta + F_[10][m]; s1, s2 = float(np.ptp(e1)), float(np.ptp(e2)); s = min(s1, s2)
        maxspread_eta = max(maxspread_eta, s); bad_eta += s > 1e-4; sgn_used.append(1 if s1 <= s2 else -1)
    t3 = dict(n_jets=len(Xs), pf_features_shape=list(nf), jets_phi_inconsistent=int(bad_phi), jets_eta_inconsistent=int(bad_eta),
              max_spread_phi_minus_dphi=maxspread_phi, max_spread_eta_minus_sdeta=maxspread_eta,
              sign_of_deta_vs_eta=dict(plus=int(sum(s == 1 for s in sgn_used)), minus=int(sum(s == -1 for s in sgn_used))))
    t3['pass'] = bool(nf[0] == 12 and bad_phi == 0 and bad_eta == 0 and len(Xs) == 200); res['tests']['T3_loader_storedaxis'] = t3; log('T3', t3)
    # T4: random stored-axis model forward
    t4 = {}; dev4 = 'cuda' if torch.cuda.is_available() else 'cpu'
    for nm_, c_ in (('storedaxis', cfg_sa), ('default', cfg_df)):
        lg = LGX(c_, None, dev4, seed=0); L = lg.logits(J); e_ = lg.embed(J, np.arange(N))
        t4[nm_] = dict(nonfinite_jets=int((~np.isfinite(L).all(1)).sum()), emb_s_finite=bool(torch.isfinite(e_['s']).all()),
                       emb_mv_finite=bool(torch.isfinite(e_['mv']).all()), force_xformers=bool(lg.model.force_xformers))
        if nm_ == 'storedaxis':      # same embedding through the default-config model class: isolates the input from the weights
            ok_rows = np.isfinite(L).all(1); t4[nm_]['first_nonfinite'] = int(np.where(~ok_rows)[0][0]) if (~ok_rows).any() else None
    # on CPU (no xformers, materialised mask) the random-init forward is non-finite for the default (authors') config as well
    # (c-p60-ut2), so there the test only requires the stored-axis path to behave like the default path; on GPU it must be finite
    t4['device'] = dev4
    t4['pass'] = bool(t4['storedaxis']['nonfinite_jets'] == 0) if dev4 == 'cuda' else bool(t4['storedaxis']['nonfinite_jets'] == t4['default']['nonfinite_jets'])
    res['tests']['T4_forward'] = t4; log('T4', t4)
    res['all_pass'] = all(t['pass'] for t in res['tests'].values())
    OUT.mkdir(parents=True, exist_ok=True); (OUT / 'p60_unittest.json').write_text(json.dumps(res, indent=1)); log('all_pass', res['all_pass'])
    sys.exit(0 if res['all_pass'] else 1)


# ------------------------------------------------------------------------------------------------ samples and edits
def _read_rows(job):
    path, rows = job
    import uproot
    from interp.part_core import read_root, sub, cat
    N = uproot.open(path)['tree'].num_entries; ch = []
    for s in range(0, N, 25000):
        r = rows[(rows >= s) & (rows < s + 25000)]
        if len(r): ch.append(sub(read_root(path, s, min(s + 25000, N)), r - s))
    return cat(ch)


def get_samples():
    """{'p1': 10-class P1 sample, 'val': fresh W/Z sample} (raw jet dicts), cached"""
    from interp.part_core import sample, sub, cat
    f = OUT / f'p60_sample{TAG}.npz'
    if f.exists():
        z = np.load(f); S = {'p1': {}, 'val': {}}
        for k in z.files: S[k.split('__')[0]][k.split('__', 1)[1]] = z[k]
        log('samples from cache', f); return S
    import uproot
    rng = np.random.default_rng(SEED); jobs = []
    for c in ('Wqq', 'Zqq'):
        p = str(VAL / f'{PRE[c]}_{VFILE}.root'); n = uproot.open(p)['tree'].num_entries; jobs.append((p, np.sort(rng.choice(n, PER_WZ, replace=False))))
    WZ = cat([_read_rows(j) for j in jobs]); ok = (WZ['mask'] > 0).sum(1) >= 2; log('val W/Z jets >= 2 particles', int(ok.sum()), 'of', len(ok))
    WZ = sub(WZ, np.where(ok)[0]); assert set(np.unique(WZ['label']).tolist()) == {iW, iZ}
    P1 = sample(PER_P1, 3000)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(f, **{f'p1__{k}': v for k, v in P1.items()}, **{f'val__{k}': v for k, v in WZ.items()}); log('cached', f)
    return {'p1': P1, 'val': WZ}


def wz_of(S):
    from interp.part_core import sub
    y = S['p1']['label'].astype(int); return {'p1wz': sub(S['p1'], np.where((y == iW) | (y == iZ))[0]), 'val': S['val']}


def M2of(J):
    from interp.part_core import kin
    pt, de, dp, _ = kin(J); p = pt.astype(np.float64); return (p * (de.astype(np.float64) ** 2 + dp.astype(np.float64) ** 2)).sum(1)


def edits(J):
    """yields (name, edited jet dict, match name); 'base' first; each edit followed by its matched dilation E4|edit"""
    from interp.part_core import kin, build
    K = kin(J); pt, de, dp, ms = K; rm = pt > 0
    Jb = build(J, *K); M2b = M2of(Jb)

    def e4(nm, Je):
        k = np.sqrt(M2of(Je) / M2b)[:, None]
        return f'E4|{nm}', build(J, pt, (de * k * rm).astype(np.float32), (dp * k * rm).astype(np.float32), ms)
    yield 'base', Jb
    for s in (+1, -1):
        nm = f'PHI{s * D:+.2f}'; Je = build(J, pt, de, np.where(rm, dp + s * D, 0).astype(np.float32), ms); yield nm, Je; yield e4(nm, Je)
    for s in (+1, -1):     # stored jet_phi += s*D, particles fixed: four-vectors identical, every Δφ input -= s*D
        nm = f'E6{s * D:+.2f}'; Je = dict(Jb); Je['jet_phi'] = (Jb['jet_phi'] + s * D).astype(Jb['jet_phi'].dtype)
        Je['dphi'] = np.where(Jb['mask'] > 0, Jb['dphi'] - s * D, 0).astype(np.float32); yield nm, Je; yield e4(nm, Je)
    sg = np.where(J['jet_eta'] < 0, -1.0, 1.0)[:, None]; je = J['jet_eta'][:, None].astype(np.float64)
    eta = je + sg * de.astype(np.float64); p64 = np.maximum(pt.astype(np.float64), 1e-12); mT = np.sqrt(p64 ** 2 + ms.astype(np.float64) ** 2)
    yv = np.arcsinh(p64 * np.sinh(eta) / mT)
    for s in (+1, -1):
        nm = f'BOOST{s * D:+.2f}'; eta2 = np.arcsinh(mT * np.sinh(yv + s * D) / p64); de2 = np.where(rm, sg * (eta2 - je), 0).astype(np.float32)
        Je = build(J, pt, de2, dp, ms); yield nm, Je; yield e4(nm, Je)


def geometry(J):
    from interp.part_core import jmass
    out = {}; Jb = None
    for nm, Je in edits(J):
        if nm == 'base': Jb = Je; m0 = jmass(Jb).astype(np.float64); M0 = M2of(Jb); continue
        out[nm] = dict(dlogM2_mean=float(np.mean(np.log(M2of(Je) / M0))),
                       dlog_minv_mean=float(np.mean(np.log(np.maximum(jmass(Je), 1e-3) / np.maximum(m0, 1e-3)))),
                       max_abs_dfourvec=float(max(np.abs(Je[k] - Jb[k]).max() for k in ('px', 'py', 'pz', 'energy'))))
    return out


def acc_block(L, y):
    p = L.argmax(1); C = np.zeros((10, 10), int); np.add.at(C, (y, p), 1)
    return dict(acc=float((p == y).mean()), per_class_recall={c: float((p[y == i] == i).mean()) for i, c in enumerate(CL)}, confusion=C.tolist())


# ------------------------------------------------------------------------------------------------ stage: L-GATr
def stage_lgatr():
    import torch
    S = get_samples(); W = wz_of(S)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; its = [t for t in ARGS.its.split(',') if t]
    cfg_sa = exp_cfg(f'{RUN_SA}/config.yaml'); cfg_p5 = exp_cfg(f'{RUN_P5}/config.yaml')
    assert cfg_sa.data.stored_axis_features and not cfg_p5.data.stored_axis_features
    models = {}
    for t in its:
        for nm, run, cfg in (('LGsa', RUN_SA, cfg_sa), ('LGp5', RUN_P5, cfg_p5)):
            p = f'{run}/models/model_run0_{t}.pt'
            if os.path.exists(p): models[f'{nm}_{t}'] = LGX(cfg, p, dev); log('loaded', nm, t)
            else: log('missing', p)
    models['LGp5_final'] = LGX(cfg_p5, f'{RUN_P5}/models/model_run0.pt', dev)
    models['LGsa_untrained'] = LGX(cfg_sa, None, dev, seed=SEED)
    out = dict(acc={}, geometry={}, rerun_floor={}); St = {}
    y1 = S['p1']['label'].astype(int)
    from interp.part_core import kin, build
    P1b = build(S['p1'], *kin(S['p1']))
    for mn, lg in models.items():
        L = lg.logits(P1b); out['acc'][mn] = acc_block(L, y1); out['acc'][mn]['acc_stored_inputs'] = float((lg.logits(S['p1']).argmax(1) == y1).mean())
        log(mn, 'P1 10-class accuracy', round(out['acc'][mn]['acc'], 4), 'N', len(y1))
    for sn, J in W.items():
        out['geometry'][sn] = geometry(J); y = J['label'].astype(int); St[f'{sn}|y'] = y; Jb = None
        for nm, Je in edits(J):
            if nm == 'base': Jb = Je
            for mn, lg in models.items():
                L = lg.logits(Je); St[f'{sn}|{mn}|{nm}|A|d'] = L[:, iW] - L[:, iZ]; St[f'{sn}|{mn}|{nm}|A|p'] = L.argmax(1).astype(np.int8)
                if nm == 'base':
                    L2 = lg.logits(Je); out['rerun_floor'][f'{sn}|{mn}'] = float(np.abs((L2[:, iW] - L2[:, iZ]) - St[f'{sn}|{mn}|base|A|d']).max()); continue
                if not nm.startswith('E4|'):
                    for r, (a, b) in (('D', (Je, Jb)), ('F', (Jb, Je))):
                        L = lg.logits(a, b); St[f'{sn}|{mn}|{nm}|{r}|d'] = L[:, iW] - L[:, iZ]; St[f'{sn}|{mn}|{nm}|{r}|p'] = L.argmax(1).astype(np.int8)
            log(sn, nm, {mn: round(float(np.mean(St[f'{sn}|{mn}|{nm}|A|d'] - St[f'{sn}|{mn}|base|A|d'])), 4) for mn in models})
    f = OUT / f'p60_lg_{"_".join(its)}{TAG}'; np.savez(f.with_suffix('.npz'), **St); f.with_suffix('.json').write_text(json.dumps(out, indent=1)); log('wrote', f)


# ------------------------------------------------------------------------------------------------ stage: ParT
def stage_part():
    import torch
    from interp.p39v_verify import runner
    from interp.part_core import ParTNetwork, kin, build
    from jetdistill.part.network import FEATURES
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; Mdl = {}
    strip = lambda sd: {k[4:] if k.startswith('mod.') else k: v for k, v in sd.items()}
    for nm, feat, pth in (('ParT_full', 'full', None), ('ParT_kin', 'kin', None),
                          ('ParT_full_s0', 'full', f'{SEEDS}/full_s0/net_best_epoch_state.pt'), ('ParT_kin_s0', 'kin', f'{SEEDS}/kin_s0/net_best_epoch_state.pt')):
        net = ParTNetwork(feat)
        if pth: net.model.load_state_dict(strip(torch.load(pth, map_location='cpu')), strict=True)
        net.model.to(dev).eval(); Mdl[nm] = dict(kind='part', feat=feat, model=net.model); log('loaded', nm, pth)
    names = list(Mdl); run = runner(Mdl, FEATURES, dev)
    S = get_samples(); W = wz_of(S); out = dict(acc={}); St = {}
    y1 = S['p1']['label'].astype(int); La = run(build(S['p1'], *kin(S['p1'])), names)
    for k in names: out['acc'][k] = acc_block(La[k], y1); log(k, 'P1 10-class accuracy', round(out['acc'][k]['acc'], 4))
    for sn, J in W.items():
        St[f'{sn}|y'] = J['label'].astype(int)
        for nm, Je in edits(J):
            L = run(Je, names)
            for k in names: St[f'{sn}|{k}|{nm}|A|d'] = L[k][:, iW] - L[k][:, iZ]; St[f'{sn}|{k}|{nm}|A|p'] = L[k].argmax(1).astype(np.int8)
            log(sn, nm, {k: round(float(np.mean(St[f'{sn}|{k}|{nm}|A|d'] - St[f'{sn}|{k}|base|A|d'])), 4) for k in names})
    f = OUT / f'p60_part{TAG}'; np.savez(f.with_suffix('.npz'), **St); f.with_suffix('.json').write_text(json.dumps(out, indent=1)); log('wrote', f)


# ------------------------------------------------------------------------------------------------ stage: analyze
def ci(b):
    b = np.asarray(b, float); b = b[np.isfinite(b)]; return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def boot_p(b):
    b = np.asarray(b, float); b = b[np.isfinite(b)]; return float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()) + 1 / len(b)))


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.zeros(m); r_ = 0.0
    for r, i in enumerate(o): r_ = max(r_, min(1.0, (m - r) * p[i])); adj[i] = r_
    return adj.tolist()


def signflip(x, rng):
    x = np.asarray(x, float); t = abs(x.mean()); n = len(x); cnt = 0
    if t == 0: return 1.0
    for s in range(0, NPERM, 250):
        b = min(250, NPERM - s); cnt += int((np.abs(rng.choice([-1.0, 1.0], (b, n)) @ x / n) >= t - 1e-15).sum())
    return float((1 + cnt) / (1 + NPERM))


def mcnemar(a, b):
    from scipy.stats import binomtest
    x, z = int((a & ~b).sum()), int((~a & b).sum()); return dict(lost=x, gained=z, p=float(binomtest(x, x + z).pvalue) if x + z else 1.0)


FAM = {'PHI': ['PHI+0.06', 'PHI-0.06'], 'E6': ['E6+0.06', 'E6-0.06'], 'BOOST': ['BOOST+0.06', 'BOOST-0.06']}


def stage_analyze():
    its = [t for t in ARGS.its.split(',') if t]; St = {}; Js = {}
    for f in [OUT / f'p60_lg_{"_".join(its)}{TAG}.npz', OUT / f'p60_part{TAG}.npz']:
        z = np.load(f); St.update({k: z[k] for k in z.files}); Js[f.stem] = json.loads(f.with_suffix('.json').read_text()); log('loaded', f)
    res = dict(node='P60', readout='d = logit_W - logit_Z', R_def='sum_jets Delta d(edit) / sum_jets Delta d(matched dilation E4|edit)',
               samples=dict(p1=f'part_core.sample({PER_P1}, 3000): rows 3000..{3000 + PER_P1 - 1} of each class test file 100 (10-class); edits on its W/Z jets',
                            val=f'val_5M WToQQ_{VFILE} / ZToQQ_{VFILE}, {PER_WZ}/class random rows, default_rng({SEED}), >= 2 particles'),
               NB=NB, NPERM=NPERM, accuracy={}, per_sample={})
    for v in Js.values():
        for k, a in v['acc'].items(): res['accuracy'][k] = a
        if 'geometry' in v: res['geometry'] = v['geometry']; res['rerun_floor'] = v['rerun_floor']
    for sn in ('p1wz', 'val'):
        y = St[f'{sn}|y']; N = len(y); w = y == iW
        models = sorted({k.split('|')[1] for k in St if k.startswith(sn + '|') and k.count('|') == 4})
        rb = np.random.default_rng(SEED + 3); C = np.zeros((NB, N), np.float32)
        for b in range(NB): C[b] = np.bincount(rb.integers(0, N, N), minlength=N)
        prng = np.random.default_rng(SEED + 4); BO = {}; o = {}
        dd = lambda m, e, r: St[f'{sn}|{m}|{e}|{r}|d'].astype(np.float64) - St[f'{sn}|{m}|base|A|d'].astype(np.float64)
        for m in models:
            om = {}
            for fn, es in FAM.items():
                for r in ('A', 'D', 'F'):
                    if f'{sn}|{m}|{es[0]}|{r}|d' not in St: continue
                    nu = sum(dd(m, e, r) for e in es); de = sum(dd(m, f'E4|{e}', 'A') for e in es)
                    b = (C @ nu) / (C @ de); BO[(m, fn, r)] = b
                    ent = dict(R=float(nu.sum() / de.sum()), R_ci=ci(b), mean_dd=float(nu.mean() / len(es)), mean_dd_E4=float(de.mean() / len(es)))
                    if r == 'A':
                        p0 = np.concatenate([St[f'{sn}|{m}|base|A|p']] * len(es)); p1 = np.concatenate([St[f'{sn}|{m}|{e}|A|p'] for e in es])
                        yy = np.concatenate([y] * len(es)); ww = yy == iW; pE = np.concatenate([St[f'{sn}|{m}|E4|{e}|A|p'] for e in es])
                        ent.update(calls_changed=float((p0 != p1).mean()), calls_changed_E4=float((p0 != pE).mean()),
                                   WtoZ=[float((p0[ww] == iZ).mean()), float((p1[ww] == iZ).mean())], WtoZ_E4=float((pE[ww] == iZ).mean()),
                                   mcnemar=mcnemar(p0 == yy, p1 == yy), p_signflip=signflip(nu, prng),
                                   per_sign={e: dict(R=float(dd(m, e, 'A').sum() / dd(m, f'E4|{e}', 'A').sum()),
                                                     WtoZ_after=float((St[f'{sn}|{m}|{e}|A|p'][w] == iZ).mean())) for e in es},
                                   in_H0_band=bool(ci(b)[0] >= -0.1 and ci(b)[1] <= 0.1), above_H1=bool(ci(b)[0] >= 0.3))
                    om[f'{fn}|{r}'] = ent
            om['acc_WZ_sample'] = float((St[f'{sn}|{m}|base|A|p'] == y).mean()); om['WtoZ_unedited'] = float((St[f'{sn}|{m}|base|A|p'][w] == iZ).mean())
            o[m] = om; log(sn, m, {k: (round(v['R'], 3), [round(x, 3) for x in v['R_ci']]) for k, v in om.items() if isinstance(v, dict) and k.endswith('|A')})
        cmp = {}
        for it in its:
            a = f'LGsa_{it}'
            if a not in models: continue
            for ref in [f'LGp5_{it}', 'LGp5_final', 'LGsa_untrained', 'ParT_full_s0', 'ParT_kin_s0', 'ParT_full', 'ParT_kin']:
                if ref not in models: continue
                for fn in FAM:
                    db = BO[(a, fn, 'A')] - BO[(ref, fn, 'A')]
                    cmp[f'{a}-{ref}|{fn}'] = dict(diff=o[a][f'{fn}|A']['R'] - o[ref][f'{fn}|A']['R'], ci=ci(db), p=boot_p(db))
        ks = list(cmp)
        for k, h in zip(ks, holm([cmp[k]['p'] for k in ks])): cmp[k]['p_holm'] = h
        # pre-registered test p-values (sign-flip, per model and family) Holm-corrected within sample
        pk = [(m, fn) for m in models for fn in FAM if f'{fn}|A' in o[m]]
        for (m, fn), h in zip(pk, holm([o[m][f'{fn}|A']['p_signflip'] for m, fn in pk])): o[m][f'{fn}|A']['p_signflip_holm'] = h
        res['per_sample'][sn] = dict(N=int(N), models=o, compare=cmp)
    # accuracy difference stored-axis vs P5 at matched iteration (P1 sample, McNemar not available across files -> bootstrap of the difference)
    res['acc_diff_vs_P5'] = {}
    for it in its:
        a, b = res['accuracy'].get(f'LGsa_{it}'), res['accuracy'].get(f'LGp5_{it}')
        if a and b: res['acc_diff_vs_P5'][it] = dict(sa=a['acc'], p5=b['acc'], diff=a['acc'] - b['acc'])
    f = OUT / f'p60_axis_battery_{"_".join(its)}{TAG}.json'; f.write_text(json.dumps(res, indent=1, default=float)); log('wrote', f)
    try: plot(res, its)
    except Exception:
        import traceback; traceback.print_exc()


def plot(res, its):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from interp.part_core import kin, build
    S = get_samples(); J = S['val']; K = kin(J); y = J['label'].astype(int)
    j = int(np.where((y == iW) & ((K[0] > 0).sum(1) >= 25))[0][0]); r = K[0][j] > 0
    fig, ax = plt.subplots(1, 2, figsize=(15, 8.2), gridspec_kw=dict(width_ratios=[1, 1.5]))
    a = ax[0]; s = 300 * K[0][j][r] / K[0][j][r].max()
    a.scatter(K[1][j][r], K[2][j][r], s=s, facecolors='none', edgecolors='0.45', label='particles (area ∝ pT), unedited')
    a.scatter(K[1][j][r], K[2][j][r] + D, s=s, facecolors='none', edgecolors='#d95f02', alpha=0.8, label='PHI +0.06: every particle Δφ + 0.06 rad')
    a.plot(0, 0, 'k+', ms=16, mew=2, label='stored jet axis (kept fixed)')
    a.set_xlabel('Δη to stored jet axis'); a.set_ylabel('Δφ to stored jet axis [rad]'); a.set_aspect('equal', 'datalim')
    a.set_title(f'Edit (a) on one W jet (pT {J["jet_pt"][j]:.0f} GeV): particles shifted in φ,\ninvariant mass unchanged; '
                f'E6 moves the axis by −0.06 instead (same Δφ inputs)', fontsize=10)
    a.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False)
    a = ax[1]; o = res['per_sample']['val']['models']
    mods = [m for m in [f'LGsa_{t}' for t in its] + [f'LGp5_{t}' for t in its] + ['LGp5_final', 'LGsa_untrained', 'ParT_kin_s0', 'ParT_full_s0', 'ParT_full'] if m in o]
    lab = lambda m: (m.replace('LGsa_', 'L-GATr stored-axis inputs, ').replace('LGp5_', 'L-GATr own-axis inputs (P5), ').replace('it', 'iteration ')
                     .replace('untrained', 'untrained').replace('final', 'final'))
    cmap = plt.get_cmap('tab10')
    fams = [('PHI', '(a) particles shifted\nin φ by ±0.06 rad'), ('E6', '(b) stored axis moved\nby ±0.06 rad'), ('BOOST', '(c) boost along beam\nrapidity ±0.06')]
    for i, (fn, fl) in enumerate(fams):
        for q, m in enumerate(mods):
            v = o[m].get(f'{fn}|A');
            if v is None: continue
            yy = i + (q - (len(mods) - 1) / 2) * 0.1; lo, hi = v['R_ci']
            un = m.endswith('untrained')
            a.errorbar(v['R'], yy, xerr=[[max(v['R'] - lo, 0)], [max(hi - v['R'], 0)]], fmt='o', color=cmap(q), ms=5, capsize=2, mfc='none' if un else cmap(q),
                       label=(lab(m) + (f' (hollow: its responses are tiny, |Δd| {abs(v["mean_dd_E4"]):.3f} under the real mass change, so R is a ratio of near-zero numbers)' if un else '')) if i == 0 else None)
    a.axvspan(-0.1, 0.1, color='0.88', zorder=0); a.axvline(0, color='k', lw=0.6); a.axvline(1, color='k', lw=0.6, ls=':')
    a.set_yticks(range(len(fams))); a.set_yticklabels([f[1] for f in fams], fontsize=9); a.invert_yaxis()
    a.set_xlabel('R = change of (logit W − logit Z) under the edit ÷ change under a real mass change\n(matched angular dilation) of the same predicted size')
    a.set_title(f'Response to moving particles off the stored axis at fixed invariant mass\n(val W/Z jets N {res["per_sample"]["val"]["N"]}; 95 % bootstrap intervals; grey band |R| < 0.1)', fontsize=10)
    a.legend(fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.2), frameon=False, ncol=1)
    fig.tight_layout(); f = OUT / f'p60_axis_battery_{"_".join(its)}{TAG}.png'; fig.savefig(f, dpi=110); log('wrote', f)


if __name__ == '__main__':
    {'unittest': stage_unittest, 'lgatr': stage_lgatr, 'part': stage_part, 'analyze': stage_analyze, 'sample': get_samples}[ARGS.stage]()
