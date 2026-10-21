"""P46 (loop C, round C11, pre-registered test of claim C6): does our Lorentz-equivariant tagger L-GATr ignore the jet axis at
fixed invariant mass, unlike ParT (C3)?

Background. ParT reads the W-vs-Z mass about the STORED jet axis: a rigid 0.06 rad phi shift of every constituent (invariant mass
unchanged) moves its W-vs-Z read-out by 0.80 of the move under a real mass change of the same predicted size (R 0.80, ParT_full).
L-GATr's inputs are the lab four-momenta (E, px, py, pz) of the particles (Lorentz-equivariant multivector channels), two fixed
reference multivectors ("spurions": the x-y plane bivector = beam reference, and the time direction (1, 0, 0, 0)), PID / track
scalars, and 7 per-particle scalar features computed inside the network's embedding from the four-momenta, about the axis of the
particle four-momentum sum (E-scheme axis), NOT the stored axis: log pT, log E, log pT/pT_jet, log E/E_jet, dphi, deta, dR
(input scalar columns 10-16). L-GATr never reads jet_eta / jet_phi or the stored deta / dphi.
Predictions by construction: (a) a rigid rotation of all particles about the beam leaves the four-vector part invariant, the
spurions invariant and all 7 scalar features unchanged (they are relative to the four-momentum-sum axis or rotation invariant), so
L-GATr's logits change only at float level; (b) E6 (stored axis moved, lab four-vectors fixed) gives L-GATr bit-for-bit (up to
float32 rebuilding) the same input. A rigid rapidity shift (longitudinal boost, extra edit 'BOOST') is not a symmetry of the
inputs: log E, log E/E_jet change and the time spurion is not boosted; this is the frame-dependent route (P44v: F_logE 8.9 % calls).

Sample. JetClass val_5M WToQQ_123.root and ZToQQ_123.root (not used by any loop-C node; not L-GATr's training files; its
validation files for W / Z are _121), 2,000 random rows each, numpy default_rng(4646) (never first-k), jets with < 2 particles
dropped. 10-class accuracy sample: 1,000 random rows per class of <class>_123.root (same rng), _124 for QCD, Hcc, TTBar (whose
_123 file is L-GATr's validation file), and for W / Z 1,000 random jets of the W / Z sample. Inputs rebuilt with part_core.build.
The sample is cached in $AI_RESULTS/lgatr/p46_sample.npz (read once; later stages / checkpoints use the cache).

Models. L-GATr (run anrunw-ai-c-p5-lgatr2, authors' jctagging config, 2M JetClass jets): checkpoints given by --ckpts
(model_run0_<tag>.pt; 'untrained' = same config, random init with torch seed 4646). ParT references: released ParT_full and
ParT_kin (100M jets), and the P10 retrains ParT_kin_s0 and ParT_full_s0 (same 2M-jet subset as L-GATr): architecture vs data.

Edits (all on the W/Z sample; pT, particle masses, PID and track inputs fixed; r = (deta, dphi) about the stored axis):
 PHI+-0.06  every real particle's dphi += +-0.06 rad, stored axis fixed: the lab four-vectors are rotated about the beam by
            0.06 rad (exact rotation; invariant mass and all pair masses unchanged); for ParT the constituents sit 0.06 rad off
            the stored axis, M2 = sum pT r^2 grows by ~ 0.0036 M0.
 BOOST+-0.06 every particle's rapidity y_i += +-0.06 at fixed pT and mass (exact longitudinal boost; jet and pair masses
            unchanged), stored axis fixed; new deta from the boosted pseudorapidity.
 E6_<delta> (P43v) every particle's (deta, dphi) += delta e (e random unit direction per jet, default_rng(4646 + 7)), stored axis
            moved by -delta e: lab four-vectors unchanged; delta in {0.005, 0.01, 0.02, 0.04, 0.06} rad.
 E1_<f>     (P43) spread about the fixed pT-weighted centroid changed by f in {-10, -5, -2, +2, +5, +10} %: two similar-pT
            particles among the 8 hardest moved apart / together along their line, Delta M2 = f M2, centroid fixed. This changes
            the invariant mass (m^2 ~ M0 M2c at small angles). Only jets with an admissible pair (P43 rule) are used.
 E1m_<f>    E1 followed by an invariant-mass compensation: a second pair (the highest-pT-sum pair among the 8 hardest not sharing
            a particle with the E1 pair) is moved along its own line about its pT barycentre (centroid fixed) until the jet
            invariant mass equals the unedited one (bisection, float64). Net: jet mass fixed, centroid fixed, M2 changes only by
            the small-angle residual; the pair invariants of the 4 moved particles change. Normalised by E4|E1_f.
 E5+-       (P43 null) same pair and per-particle displacement as E1 +-10 %, direction chosen so M2 and the centroid are
            unchanged; changes pair invariants with M2 fixed. Pooled as P43: (dd(E5+) + dd(E5-)) / (dd(E4|E1+10) - dd(E4|E1-10)).
 E4|<edit>  matched angular dilation about the stored axis: every particle's (deta, dphi) x k, k^2 = M2(edit) / M2 per jet (the
            same predicted change of the axis-moment mass sqrt(M0 M2) as the edit; a real change of jet and pair masses).
Read-out d = l_W - l_Z (logit difference). R(edit) = sum_jets Delta d(edit) / sum_jets Delta d(matched E4): 0 = no response,
1 = the response to a real mass change of the same predicted size.
L-GATr route split (per edit and per dilation): A = full edit; D = edited four-vectors, the 7 scalar features (cols 10-16) taken
from the unedited jet ("features held"); F = unedited four-vectors, scalar features from the edited jet ("features only");
for BOOST also C = edited four-vectors, features held, time spurion boosted with the jet (exact symmetry check: predicted 0).
The time spurion is held at its unedited value (1, 0, 0, 0) in A, D, F; in C it is co-transformed.
Statistics: NB bootstrap resamples over jets (shared count matrix, paired across edits and models); sign-flip permutation of the
per-jet Delta d (H0: no mean response); exact McNemar on per-jet correctness; Holm over the edit x dose grid per model.

P51 (round C12): relaunch on the final checkpoint (tag 'final' -> model_run0.pt); outputs are prefixed with $P46_NODE (default p51);
the sample cache keeps the name p46_sample.npz (same sample).
  ./sync.sh && ./part_job.sh c-p51-run '/j-jepa-vol/aaron-interp/env-lgatr/bin/python -m interp.p46_lgatr_c6 --stage lgatr --ckpts final,it174999,untrained && python -m interp.p46_lgatr_c6 --stage part && python -m interp.p46_lgatr_c6 --stage analyze'
-> $AI_RESULTS/lgatr/p51_lg_<tags>.npz, $AI_RESULTS/part/p51_part.npz, $AI_RESULTS/lgatr/p51_lgatr_c6.json, p51_lgatr_c6.png
"""
import argparse, glob, json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('JETDISTILL_TAGGER', 'part')
ap = argparse.ArgumentParser(); ap.add_argument('--stage', required=True); ap.add_argument('--ckpts', default='it174999,it154999,untrained')
ap.add_argument('--smoke', action='store_true'); ARGS = ap.parse_args()
SMOKE = ARGS.smoke
SEED, FIDX = 4646, 123
PER_WZ = 150 if SMOKE else 2000
PER_ACC = 30 if SMOKE else 1000
NB = 200 if SMOKE else 2000
NPERM = 500 if SMOKE else 5000
VAL = Path('/j-jepa-vol/JetClass/Pythia/val_5M')
PRE = {'QCD': 'ZJetsToNuNu', 'Hbb': 'HToBB', 'Hcc': 'HToCC', 'Hgg': 'HToGG', 'H4q': 'HToWW4Q', 'Hqql': 'HToWW2Q1L', 'Zqq': 'ZToQQ',
       'Wqq': 'WToQQ', 'Tbqq': 'TTBar', 'Tbl': 'TTBarLep'}
CL = ['QCD', 'Hbb', 'Hcc', 'Hgg', 'H4q', 'Hqql', 'Zqq', 'Wqq', 'Tbqq', 'Tbl']
iW, iZ = CL.index('Wqq'), CL.index('Zqq')
LG_VALF = {'ZJetsToNuNu': 123, 'HToBB': 121, 'HToCC': 123, 'HToGG': 120, 'HToWW4Q': 121, 'HToWW2Q1L': 124, 'TTBar': 123,
           'TTBarLep': 124, 'WToQQ': 121, 'ZToQQ': 121}
AFILE = {c: (124 if LG_VALF[PRE[c]] == FIDX else FIDX) for c in CL}
ROOT = Path(os.environ.get('AI_RESULTS', 'results'))
TAG = '_smoke' if SMOKE else ''
SAMPLE = ROOT / 'lgatr' / f'p46_sample{TAG}.npz'
LGRUN = '/j-jepa-vol/aaron-interp/models/lgatr/runs/jc2m/jc2m_s0a/models'
SEEDS = '/j-jepa-vol/aaron-interp/models/part_seeds'
DPHI, DY = 0.06, 0.06
DL6 = [0.005, 0.01, 0.02, 0.04, 0.06]
FS = [-0.10, -0.05, -0.02, 0.02, 0.05, 0.10]
TITLE = os.environ.get('P46_TITLE', 'L-GATr: R near 0 for every mass-fixed edit; ParT: R 0.75-0.81 under the boost')
T0 = time.time()
NODE = os.environ.get('P46_NODE', 'p51')           # P51 (round C12) = P46 relaunched on the final checkpoint; outputs p51_*
CKFILE = {'final': 'model_run0.pt'}                # tag -> file; other tags -> model_run0_<tag>.pt


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


# ------------------------------------------------------------------------------------------------ sample
def _read_rows(job):
    path, rows = job
    import uproot
    from interp.part_core import read_root, sub, cat
    N = uproot.open(path)['tree'].num_entries; ch = []
    for s in range(0, N, 25000):
        r = rows[(rows >= s) & (rows < s + 25000)]
        if len(r): ch.append(sub(read_root(path, s, min(s + 25000, N)), r - s))
    return cat(ch)


def get_sample():
    """(WZ jet dict, ACC jet dict), raw (not rebuilt)"""
    from interp.part_core import sub, cat
    if SAMPLE.exists():
        z = np.load(SAMPLE); WZ, AC = {}, {}
        for k in z.files:
            (WZ if k.startswith('wz__') else AC)[k[4:]] = z[k]
        log('sample from cache', SAMPLE, len(WZ['label']), len(AC['label'])); return WZ, AC
    import uproot
    from concurrent.futures import ProcessPoolExecutor
    rng = np.random.default_rng(SEED); jobs = []
    for c in ('Wqq', 'Zqq'):
        p = str(VAL / f'{PRE[c]}_{FIDX}.root'); N = uproot.open(p)['tree'].num_entries
        jobs.append((p, np.sort(rng.choice(N, PER_WZ, replace=False))))
    oth = [c for c in CL if c not in ('Wqq', 'Zqq')]
    for c in oth:
        p = str(VAL / f'{PRE[c]}_{AFILE[c]}.root'); N = uproot.open(p)['tree'].num_entries
        jobs.append((p, np.sort(rng.choice(N, PER_ACC, replace=False))))
    log('reading', [(Path(p).name, len(r)) for p, r in jobs])
    with ProcessPoolExecutor(4) as ex: Js = list(ex.map(_read_rows, jobs))
    WZ = cat(Js[:2]); assert (WZ['label'][:len(jobs[0][1])] == iW).all() and (WZ['label'][len(jobs[0][1]):] == iZ).all()
    ok = (WZ['mask'] > 0).sum(1) >= 2; log('W/Z jets with >= 2 particles', int(ok.sum()), 'of', len(ok)); WZ = sub(WZ, np.where(ok)[0])
    for c, J in zip(oth, Js[2:]): assert (J['label'] == CL.index(c)).all(), c
    y = WZ['label'].astype(int)
    aw = np.sort(rng.choice(np.where(y == iW)[0], PER_ACC, replace=False)); az = np.sort(rng.choice(np.where(y == iZ)[0], PER_ACC, replace=False))
    AC = cat(Js[2:] + [sub(WZ, aw), sub(WZ, az)])
    SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(SAMPLE, **{f'wz__{k}': v for k, v in WZ.items()}, **{f'ac__{k}': v for k, v in AC.items()})
    log('cached sample', SAMPLE); return WZ, AC


# ------------------------------------------------------------------------------------------------ edits
def M2of(K):
    pt, de, dp, _ = K; p = pt.astype(np.float64); return (p * (de.astype(np.float64) ** 2 + dp.astype(np.float64) ** 2)).sum(1)


def fourvec(J, pt, de, dp, ms):
    """float64 (E, px, py, pz) per particle from (pt, deta, dphi, mass) about the stored axis of J (part_core.build convention)"""
    sg = np.where(J['jet_eta'] < 0, -1.0, 1.0)[:, None]; m = pt > 0
    eta = J['jet_eta'][:, None].astype(np.float64) + sg * de; phi = J['jet_phi'][:, None].astype(np.float64) + dp
    pt = pt.astype(np.float64); px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    E = np.sqrt((pt * np.cosh(eta)) ** 2 + ms.astype(np.float64) ** 2)
    return [np.where(m, a, 0.0) for a in (E, px, py, pz)]


def minv(P4):
    E, px, py, pz = (a.sum(-1) for a in P4); return np.sqrt(np.maximum(E ** 2 - px ** 2 - py ** 2 - pz ** 2, 0))


def compensate(J, K, P, de2, dp2):
    """E1m: after the E1 move (de2, dp2), move a second pair (c, d) along its line about its pT barycentre until the jet invariant
    mass equals the unedited one. Returns de3, dp3, valid, info."""
    pt, de, dp, ms = K; n = len(pt); r = np.arange(n)
    m0 = minv(fourvec(J, pt, de.astype(np.float64), dp.astype(np.float64), ms))
    prs = [(a, b) for a in range(8) for b in range(a + 1, 8)]; ia = np.array([p[0] for p in prs]); ib = np.array([p[1] for p in prs])
    a, b = P['a'][:, None], P['b'][:, None]
    excl = (ia[None] == a) | (ia[None] == b) | (ib[None] == a) | (ib[None] == b)
    pa, pb = pt[:, ia].astype(np.float64), pt[:, ib].astype(np.float64)
    sep = np.hypot(de2[:, ia] - de2[:, ib], dp2[:, ia] - dp2[:, ib])
    ok = (pa > 0) & (pb > 0) & ~excl & (sep > 1e-4); score = np.where(ok, pa + pb, -np.inf); k = score.argmax(1)
    valid = np.isfinite(score[r, k]); c, d = ia[k], ib[k]
    pc, pd = pt[r, c].astype(np.float64), pt[r, d].astype(np.float64); S = np.maximum(pc + pd, 1e-12)
    xc = np.stack([de2[r, c], dp2[r, c]], 1); xd = np.stack([de2[r, d], dp2[r, d]], 1); dv = xc - xd
    P4 = fourvec(J, pt, de2, dp2, ms)
    keepm = np.ones_like(P4[0], bool); keepm[r, c] = False; keepm[r, d] = False
    rest = [np.where(keepm, q, 0).sum(1) for q in P4]
    one = lambda x, p, m: fourvec({'jet_eta': J['jet_eta'], 'jet_phi': J['jet_phi']}, p[:, None], x[:, :1], x[:, 1:], m[:, None])

    def mass(lam):
        x1 = xc + (lam * pd / S)[:, None] * dv; x2 = xd - (lam * pc / S)[:, None] * dv
        q1 = one(x1, pc, ms[r, c]); q2 = one(x2, pd, ms[r, d])
        tot = [rest[i] + q1[i][:, 0] + q2[i][:, 0] for i in range(4)]
        return np.sqrt(np.maximum(tot[0] ** 2 - tot[1] ** 2 - tot[2] ** 2 - tot[3] ** 2, 0))
    lo, hi = np.full(n, -0.9), np.full(n, 3.0); flo, fhi = mass(lo) - m0, mass(hi) - m0
    valid &= (flo * fhi) < 0
    for _ in range(60):
        mid = 0.5 * (lo + hi); fm = mass(mid) - m0; same = np.sign(fm) == np.sign(flo)
        lo = np.where(same, mid, lo); flo = np.where(same, fm, flo); hi = np.where(same, hi, mid)
    lam = 0.5 * (lo + hi); lam = np.where(valid, lam, 0.0)
    de3, dp3 = de2.copy(), dp2.copy()
    x1 = xc + (lam * pd / S)[:, None] * dv; x2 = xd - (lam * pc / S)[:, None] * dv
    de3[r, c], dp3[r, c], de3[r, d], dp3[r, d] = x1[:, 0], x1[:, 1], x2[:, 0], x2[:, 1]
    resid = np.abs(np.log(np.maximum(mass(lam), 1e-9) / np.maximum(m0, 1e-9)))
    return de3, dp3, valid, dict(lam_median=float(np.median(lam[valid])) if valid.any() else None,
                                 max_abs_dlogm_after=float(resid[valid].max()) if valid.any() else None, n_valid=int(valid.sum()))


def edits(J):
    """generator of (name, rows, jet dict for those rows, info). Order: base, then each edit followed by its matched E4."""
    from interp.part_core import kin, build, jmass, sub
    from interp import p43_spread_vs_centroid as P43
    from interp.p43v_verify import rigid
    K = kin(J); pt, de, dp, ms = K; rm = pt > 0; n = len(pt); ALL = np.arange(n)
    Jb = build(J, *K); M2b = M2of(kin(Jb)); mb = jmass(Jb).astype(np.float64)

    def info(rows, Je, match, extra=None):
        Ke = kin(Je); dl = np.log(M2of(Ke) / M2b[rows]); dm = np.log(np.maximum(jmass(Je).astype(np.float64), 1e-3) / np.maximum(mb[rows], 1e-3))
        I = dict(rows=rows, dlogM2=dl, match=match, geo=dict(n=int(len(rows)), dlogM2_mean=float(dl.mean()), dlogM2_median=float(np.median(dl)),
                                                             dlog_minv_mean=float(dm.mean()), dlog_minv_maxabs=float(np.abs(dm).max())))
        if extra: I['geo'].update(extra)
        return I

    def e4(name, rows, dl):
        Jr = sub(J, rows); Kr = kin(Jr); k = np.sqrt(np.exp(dl))[:, None]; r_ = Kr[0] > 0
        Je = build(Jr, Kr[0], (Kr[1] * k * r_).astype(np.float32), (Kr[2] * k * r_).astype(np.float32), Kr[3])
        return f'E4|{name}', rows, Je, info(rows, Je, f'E4|{name}', dict(k_median=float(np.median(k))))

    yield 'base', ALL, Jb, info(ALL, Jb, 'base')
    for s in (+1, -1):
        nm = f'PHI{s * DPHI:+.2f}'; Je = build(J, pt, de, np.where(rm, dp + s * DPHI, 0).astype(np.float32), ms); I = info(ALL, Je, nm)
        yield nm, ALL, Je, I; yield e4(nm, ALL, I['dlogM2'])
    sg = np.where(J['jet_eta'] < 0, -1.0, 1.0)[:, None]; je = J['jet_eta'][:, None].astype(np.float64)
    eta = je + sg * de.astype(np.float64); p64 = np.maximum(pt.astype(np.float64), 1e-12); mT = np.sqrt(p64 ** 2 + ms.astype(np.float64) ** 2)
    yv = np.arcsinh(p64 * np.sinh(eta) / mT)
    for s in (+1, -1):
        nm = f'BOOST{s * DY:+.2f}'; eta2 = np.arcsinh(mT * np.sinh(yv + s * DY) / p64); de2 = np.where(rm, sg * (eta2 - je), 0).astype(np.float32)
        Je = build(J, pt, de2, dp, ms); I = info(ALL, Je, nm); yield nm, ALL, Je, I; yield e4(nm, ALL, I['dlogM2'])
    rr = np.random.default_rng(SEED + 7); al = rr.uniform(0, 2 * np.pi, n); er = np.stack([np.cos(al), np.sin(al)], 1)
    for dl in DL6:
        nm = f'E6_{dl:.3f}'; sh = dl * er; Je, nflip = rigid(J, K, sh[:, 0], sh[:, 1]); I = info(ALL, Je, nm, dict(n_eta_sign_flip=nflip))
        yield nm, ALL, Je, I; yield e4(nm, ALL, I['dlogM2'])
    Pall = P43.plan_edits(pt, de, dp, np.random.default_rng(SEED + 1)); kept = np.where(Pall['keep'])[0]
    log(f'E1 pair rule: {len(kept)} of {n} jets admissible', Pall['diag'])
    Jk = sub(J, kept); Kk = kin(Jk); Pk = P43.plan_edits(*Kk[:3], np.random.default_rng(SEED + 2)); assert Pk['keep'].all()
    for f in FS:
        nm = f'E1_{f:+.2f}'; Je = P43.make(Jk, Kk, Pk, nm); I = info(kept, Je, nm); yield nm, kept, Je, I; yield e4(nm, kept, I['dlogM2'])
        de2, dp2, _, _ = P43.displaced(Kk, Pk, nm); de3, dp3, ok, ci = compensate(Jk, Kk, Pk, de2, dp2)
        r_ = Kk[0] > 0; Je = build(Jk, Kk[0], np.where(r_, de3, 0).astype(np.float32), np.where(r_, dp3, 0).astype(np.float32), Kk[3])
        v = np.where(ok)[0]; nm2 = f'E1m_{f:+.2f}'; yield nm2, kept[v], sub(Je, v), info(kept[v], sub(Je, v), nm, ci)
    for nm in ('E5+', 'E5-'):
        Je = P43.make(Jk, Kk, Pk, nm); yield nm, kept, Je, info(kept, Je, 'E1_+0.10' if nm == 'E5+' else 'E1_-0.10')


def store_put(S, model, edit, route, rows, L, N):
    d = np.full(N, np.nan, np.float32); p = np.full(N, -1, np.int8)
    d[rows] = L[:, iW] - L[:, iZ]; p[rows] = L.argmax(1); S[f'{model}|{edit}|{route}|d'] = d; S[f'{model}|{edit}|{route}|p'] = p


def meta_put(S, M, name, I, N):
    r = np.zeros(N, bool); r[I['rows']] = True; dl = np.full(N, np.nan); dl[I['rows']] = I['dlogM2']
    S[f'meta|{name}|rows'] = r; S[f'meta|{name}|dlogM2'] = dl; M[name] = dict(match=I['match'], geo=I['geo'])


def acc_block(L, y):
    p = L.argmax(1); C = np.zeros((10, 10), int); np.add.at(C, (y, p), 1)
    return dict(acc=float((p == y).mean()), per_class_recall={c: float((p[y == i] == i).mean()) for i, c in enumerate(CL)}, confusion=C.tolist())


# ------------------------------------------------------------------------------------------------ stage: L-GATr
def stage_lgatr():
    import torch
    sys.argv = [sys.argv[0], '--stage', 'none']
    import interp.p44_lgatr_grades as P
    from interp.part_core import kin, build, sub
    WZ, AC = get_sample()                          # before CUDA init (the reader forks worker processes)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; tags = ARGS.ckpts.split(','); models = {}
    for t in tags:
        if t == 'untrained':
            ms = torch.manual_seed; torch.manual_seed = lambda s: ms(SEED)
            try: models['LGATr_untrained'] = P.LG(None, dev)
            finally: torch.manual_seed = ms
        else:
            pth = f'{LGRUN}/{CKFILE.get(t, f"model_run0_{t}.pt")}'; assert os.path.exists(pth), pth; models[f'LGATr_{t}'] = P.LG(pth, dev)
        log('loaded', t, dev)
    N = len(WZ['label']); S, M = {}, {}; out = dict(ckpts=tags, acc={})
    ACb = build(AC, *kin(AC)); ya = AC['label'].astype(int); yw = WZ['label'].astype(int)
    for mn, lg in models.items():
        out['acc'][mn] = acc_block(lg.logits(ACb), ya); out['acc'][mn]['acc_stored_inputs'] = float((lg.logits(AC).argmax(1) == ya).mean())
        log(mn, '10-class accuracy', out['acc'][mn]['acc'], 'N', len(ya))
    for name, rows, Je, I in edits(WZ):
        meta_put(S, M, name, I, N); Jbr = None
        if name == 'base': Jb = Je
        else: Jbr = sub(Jb, rows)
        for mn, lg in models.items():
            store_put(S, mn, name, 'A', rows, lg.logits(Je), N)
            if name == 'base':
                L2 = lg.logits(Je); out.setdefault('rerun_floor', {})[mn] = float(np.abs((L2[:, iW] - L2[:, iZ]) - S[f'{mn}|base|A|d']).max())
                out.setdefault('acc_WZ', {})[mn] = float((L2.argmax(1) == yw).mean()); continue
            store_put(S, mn, name, 'D', rows, lg.logits(Je, Jbr), N)
            store_put(S, mn, name, 'F', rows, lg.logits(Jbr, Je), N)
            if name.startswith('BOOST'):
                store_put(S, mn, name, 'C', rows, lg.logits(Je, Jbr, spur_y=float(name[5:])), N)
        log(name, I['geo'], {mn: round(float(np.nanmean(S[f'{mn}|{name}|A|d'] - S[f'{mn}|base|A|d'])), 4) for mn in models})
    out['meta'] = M; out['N_WZ'] = int(N); out['N_acc'] = int(len(ya))
    S['meta|y'] = yw; f = ROOT / 'lgatr' / f'{NODE}_lg_{"_".join(tags)}{TAG}'
    np.savez(f.with_suffix('.npz'), **S); f.with_suffix('.json').write_text(json.dumps(out, indent=1, default=float)); log('wrote', f)


# ------------------------------------------------------------------------------------------------ stage: ParT
def stage_part():
    import torch
    from interp.p39v_verify import runner
    from interp.part_core import ParTNetwork, kin, build
    from jetdistill.part.network import FEATURES
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; Mdl = {}
    strip_mod = lambda sd: {k[4:] if k.startswith('mod.') else k: v for k, v in sd.items()}
    for nm, feat, pth in (('ParT_full', 'full', None), ('ParT_kin', 'kin', None),
                          ('ParT_kin_s0', 'kin', f'{SEEDS}/kin_s0/net_best_epoch_state.pt'), ('ParT_full_s0', 'full', f'{SEEDS}/full_s0/net_best_epoch_state.pt')):
        net = ParTNetwork(feat)
        if pth: net.model.load_state_dict(strip_mod(torch.load(pth, map_location='cpu')), strict=True)
        net.model.to(dev).eval(); Mdl[nm] = dict(kind='part', feat=feat, model=net.model); log('loaded', nm, pth)
    names = list(Mdl); run = runner(Mdl, FEATURES, dev)
    WZ, AC = get_sample(); N = len(WZ['label']); S, M = {}, {}; out = dict(acc={})
    ya = AC['label'].astype(int); La = run(build(AC, *kin(AC)), names)
    for k in names: out['acc'][k] = acc_block(La[k], ya); log(k, '10-class accuracy', out['acc'][k]['acc'])
    for name, rows, Je, I in edits(WZ):
        meta_put(S, M, name, I, N); L = run(Je, names)
        for k in names: store_put(S, k, name, 'A', rows, L[k], N)
        if name == 'base': out['acc_WZ'] = {k: float((L[k].argmax(1) == WZ['label']).mean()) for k in names}
        log(name, {k: round(float(np.nanmean(S[f'{k}|{name}|A|d'] - S[f'{k}|base|A|d'])), 4) for k in names})
    out['meta'] = M; S['meta|y'] = WZ['label'].astype(int); f = ROOT / 'part' / f'{NODE}_part{TAG}'
    np.savez(f.with_suffix('.npz'), **S); f.with_suffix('.json').write_text(json.dumps(out, indent=1, default=float)); log('wrote', f)


# ------------------------------------------------------------------------------------------------ stage: analyze
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.zeros(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj.tolist()


def ci(b):
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] if len(b) else [None, None]


def boot_p(b):
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    return float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()) + 1 / len(b))) if len(b) else None


def signflip(x, rng):
    x = np.asarray(x, float); t = abs(x.mean()); n = len(x)
    if t == 0: return 1.0
    cnt = 0
    for s in range(0, NPERM, 250):
        b = min(250, NPERM - s); cnt += int((np.abs(rng.choice([-1.0, 1.0], (b, n)) @ x / n) >= t - 1e-15).sum())
    return float((1 + cnt) / (1 + NPERM))


def mcnemar(a, b):
    from scipy.stats import binomtest
    x, z = int((a & ~b).sum()), int((~a & b).sum()); return dict(lost=x, gained=z, p=float(binomtest(x, x + z).pvalue) if x + z else 1.0)


def stage_analyze():
    S, J = {}, {}
    files = sorted(glob.glob(str(ROOT / 'lgatr' / f'{NODE}_lg_*{TAG}.npz'))) + [str(ROOT / 'part' / f'{NODE}_part{TAG}.npz')]
    if not SMOKE: files = [f for f in files if '_smoke' not in f]
    for f in files:
        if not os.path.exists(f): continue
        z = np.load(f); S.update({k: z[k] for k in z.files}); J[Path(f).stem] = json.loads(Path(f).with_suffix('.json').read_text()); log('loaded', f)
    y = S['meta|y']; N = len(y); isW, isZ = y == iW, y == iZ
    models = sorted({k.split('|')[0] for k in S if not k.startswith('meta|')})
    meta = {}
    for v in J.values(): meta.update(v['meta'])
    edits_ = [e for e in meta if e != 'base' and not e.startswith('E4|')]
    rb = np.random.default_rng(SEED + 3); C = np.zeros((NB, N), np.float32)
    for b in range(NB): C[b] = np.bincount(rb.integers(0, N, N), minlength=N)
    prng = np.random.default_rng(SEED + 4)
    main_lg = 'LGATr_final' if 'LGATr_final' in models else 'LGATr_it199999' if 'LGATr_it199999' in models else ('LGATr_it174999' if 'LGATr_it174999' in models else None)
    res = dict(node=NODE.upper(), sample=f'val_5M {{WToQQ,ZToQQ}}_{FIDX}.root, {PER_WZ}/class, default_rng({SEED}); 10-class: {PER_ACC}/class files {AFILE}',
               N_WZ=int(N), NB=NB, NPERM=NPERM, main_lgatr=main_lg, models=models, readout='d = logit_W - logit_Z',
               R_def='sum_jets Delta d(edit) / sum_jets Delta d(matched dilation E4)', geometry={e: meta[e]['geo'] for e in meta},
               accuracy={}, per_model={}, pooled={}, compare={})
    for v in J.values():
        for k, a in v['acc'].items(): res['accuracy'][k] = dict(acc10=a['acc'], acc10_stored=a.get('acc_stored_inputs'), per_class=a['per_class_recall'],
                                                                 acc_WZ_sample=v.get('acc_WZ', {}).get(k), rerun_floor=v.get('rerun_floor', {}).get(k),
                                                                 confusion=a['confusion'])
    BOOTS = {}
    den_of = lambda e: f"E4|{meta[e]['match']}"

    def delta(m, e, route):
        k = f'{m}|{e}|{route}|d'
        if k not in S: return None
        d = S[k].astype(np.float64) - S[f'{m}|base|A|d'].astype(np.float64); return np.where(np.isfinite(d), d, 0.0)

    for m in models:
        o = {}; pv, pk = [], []
        for e in edits_:
            rows = S[f'meta|{e}|rows']
            for route in ('A', 'D', 'F', 'C'):
                num = delta(m, e, route)
                if num is None: continue
                den = delta(m, den_of(e), 'A') * rows
                v = num.sum() / den.sum(); b = (C @ num) / (C @ den); BOOTS[(m, e, route)] = b
                r = dict(R=float(v), R_ci=ci(b), mean_dd=float(num[rows].mean()), mean_dd_E4=float(den[rows].mean()),
                         abs_ratio=float(np.abs(num[rows]).mean() / max(np.abs(den[rows]).mean(), 1e-12)))
                p0r, p1r = S[f'{m}|base|A|p'][rows], S[f'{m}|{e}|{route}|p'][rows]
                r.update(calls_changed_route=float((p0r != p1r).mean()), mcnemar_route=mcnemar(p0r == y[rows], p1r == y[rows]))
                if route == 'A':
                    p0, p1 = S[f'{m}|base|A|p'][rows], S[f'{m}|{e}|A|p'][rows]; yr = y[rows]; w, z = yr == iW, yr == iZ
                    pE = S[f'{m}|{den_of(e)}|A|p'][rows]
                    r.update(n=int(rows.sum()), calls_changed=float((p0 != p1).mean()), calls_changed_E4=float((p0 != pE).mean()),
                             WtoZ=[float((p0[w] == iZ).mean()), float((p1[w] == iZ).mean())], WtoZ_E4=float((pE[w] == iZ).mean()),
                             ZtoW=[float((p0[z] == iW).mean()), float((p1[z] == iW).mean())], mcnemar=mcnemar(p0 == yr, p1 == yr),
                             p_signflip=signflip(num[rows], prng))
                    pv.append(r['p_signflip']); pk.append(e)
                o[f'{e}|{route}'] = r
            if f'{m}|{e}|D|d' in S:
                a_, d_ = delta(m, e, 'A'), delta(m, e, 'D')
                o[f'{e}|feature_share'] = float((a_ - d_).sum() / a_.sum()) if a_.sum() != 0 else None
                if f'{m}|{e}|C|d' in S:   # BOOST: A - D = scalar-feature route, D - C = time-spurion route, C = residual (exact symmetry -> 0)
                    c_ = delta(m, e, 'C'); o[f'{e}|route_split'] = dict(
                        mean_dd_A=float(a_.mean()), feature_route=float((a_ - d_).mean()), spurion_route=float((d_ - c_).mean()),
                        residual_C=float(c_.mean()), mean_abs_A=float(np.abs(a_).mean()), mean_abs_D=float(np.abs(d_).mean()),
                        mean_abs_C=float(np.abs(c_).mean()), share_feature_absd=float(np.abs(a_ - d_).sum() / max(np.abs(a_ - d_).sum() + np.abs(d_ - c_).sum(), 1e-12)))
        for e, h in zip(pk, holm(pv)): o[f'{e}|A']['p_holm'] = h
        res['per_model'][m] = o
        # pooled families
        fam = {'PHI': ['PHI+0.06', 'PHI-0.06'], 'BOOST': ['BOOST+0.06', 'BOOST-0.06'], 'E6_0.01-0.06': [f'E6_{x:.3f}' for x in DL6[1:]],
               'E1_raise': [f'E1_{f:+.2f}' for f in FS if f > 0], 'E1_lower': [f'E1_{f:+.2f}' for f in FS if f < 0],
               'E1m_raise': [f'E1m_{f:+.2f}' for f in FS if f > 0], 'E1m_lower': [f'E1m_{f:+.2f}' for f in FS if f < 0]}
        po = {}
        for fn, es in fam.items():
            for route in ('A', 'D', 'F', 'C'):
                if f'{m}|{es[0]}|{route}|d' not in S: continue
                nums = [delta(m, e, route) for e in es]; dens = [delta(m, den_of(e), 'A') * S[f'meta|{e}|rows'] for e in es]
                nu, de_ = sum(x.sum() for x in nums), sum(x.sum() for x in dens)
                b = sum(C @ x for x in nums) / sum(C @ x for x in dens); BOOTS[(m, fn, route)] = b
                po[f'{fn}|{route}'] = dict(R=float(nu / de_), R_ci=ci(b), within_pm01=bool(ci(b)[0] is not None and ci(b)[0] >= -0.1 and ci(b)[1] <= 0.1))
        # added in P51 (after the smoke run showed opposite-sign R for E1m raise / lower): E1m pooled over all six doses.
        # antisym = linear part, (sum raise - sum lower) / (E4 raise - E4 lower); sym = even part, (sum raise + sum lower) / same.
        for route in ('A',):
            up, dn = [f'E1m_{f:+.2f}' for f in FS if f > 0], [f'E1m_{f:+.2f}' for f in FS if f < 0]
            nu_u, nu_d = sum(delta(m, e, route) for e in up), sum(delta(m, e, route) for e in dn)
            de_u = sum(delta(m, den_of(e), 'A') * S[f'meta|{e}|rows'] for e in up); de_d = sum(delta(m, den_of(e), 'A') * S[f'meta|{e}|rows'] for e in dn)
            den = de_u - de_d; bd = C @ den
            for nm_, nu in (('E1m_antisym', nu_u - nu_d), ('E1m_sym', nu_u + nu_d)):
                b = (C @ nu) / bd; BOOTS[(m, nm_, route)] = b; po[f'{nm_}|{route}'] = dict(R=float(nu.sum() / den.sum()), R_ci=ci(b))
        # pooled call flow per informative family (route A): share of calls changed, W->Z before/after, McNemar on correctness
        for fn, es in (('BOOST', ['BOOST+0.06', 'BOOST-0.06']), ('E1m_raise', [f'E1m_{f:+.2f}' for f in FS if f > 0]),
                       ('E1m_lower', [f'E1m_{f:+.2f}' for f in FS if f < 0]), ('E5', ['E5+', 'E5-']), ('PHI', ['PHI+0.06', 'PHI-0.06'])):
            p0s, p1s, ys = [], [], []
            for e in es:
                rows = S[f'meta|{e}|rows']; p0s.append(S[f'{m}|base|A|p'][rows]); p1s.append(S[f'{m}|{e}|A|p'][rows]); ys.append(y[rows])
            p0, p1, yy = np.concatenate(p0s), np.concatenate(p1s), np.concatenate(ys); w = yy == iW
            po[f'{fn}|calls'] = dict(n_pairs=int(len(yy)), calls_changed=float((p0 != p1).mean()), WtoZ=[float((p0[w] == iZ).mean()), float((p1[w] == iZ).mean())],
                                    mcnemar=mcnemar(p0 == yy, p1 == yy))
        n5 = delta(m, 'E5+', 'A') + delta(m, 'E5-', 'A'); rows5 = S['meta|E5+|rows']
        d5 = (delta(m, 'E4|E1_+0.10', 'A') - delta(m, 'E4|E1_-0.10', 'A')) * rows5
        b = (C @ n5) / (C @ d5); BOOTS[(m, 'E5', 'A')] = b; po['E5|A'] = dict(R=float(n5.sum() / d5.sum()), R_ci=ci(b))
        res['pooled'][m] = po
        log(m, {k: (round(v['R'], 3), [round(x, 3) for x in v['R_ci']]) for k, v in po.items() if k.endswith('|A')})
    # comparisons L-GATr vs ParT references (paired over the same bootstrap resamples)
    if main_lg:
        for ref in [x for x in models if x != main_lg]:
            cmp = {}
            for fn in ('PHI', 'BOOST', 'E6_0.01-0.06', 'E1_raise', 'E1_lower', 'E1m_raise', 'E1m_lower', 'E1m_antisym', 'E1m_sym', 'E5'):
                b1, b2 = BOOTS.get((main_lg, fn, 'A')), BOOTS.get((ref, fn, 'A'))
                if b1 is None or b2 is None: continue
                k1 = f'{fn}|A'
                cmp[fn] = dict(diff=res['pooled'][main_lg][k1]['R'] - res['pooled'][ref][k1]['R'], ci=ci(b1 - b2), p=boot_p(b1 - b2))
            ks = list(cmp)
            for k, h in zip(ks, holm([cmp[k]['p'] for k in ks])): cmp[k]['p_holm'] = h
            res['compare'][f'{main_lg}_minus_{ref}'] = cmp
    out = ROOT / 'lgatr' / f'{NODE}_lgatr_c6{TAG}.json'; out.write_text(json.dumps(res, indent=1, default=float)); log('wrote', out)
    try: plot(res, S)
    except Exception:
        import traceback; traceback.print_exc()


def plot(res, S):
    """left: one real W jet, the E1m +10 % edit (spread pair moved apart, second pair moved together until the invariant mass is
    restored); right: R per informative edit family and model (PHI / E6 are symmetry / plumbing checks, listed in the json)."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from interp.part_core import kin, build, jmass, sub
    from interp import p43_spread_vs_centroid as P43
    WZ, _ = get_sample(); K = kin(WZ); pt, de, dp, ms = K
    Pall = P43.plan_edits(pt, de, dp, np.random.default_rng(SEED + 1)); kept = np.where(Pall['keep'])[0]
    Jk = sub(WZ, kept); Kk = kin(Jk); Pk = P43.plan_edits(*Kk[:3], np.random.default_rng(SEED + 2))
    de2, dp2, _, _ = P43.displaced(Kk, Pk, 'E1_+0.10'); de3, dp3, ok, _ = compensate(Jk, Kk, Pk, de2, dp2)
    cand = np.where(ok & (Jk['label'] == iW) & ((Kk[0] > 0).sum(1) >= 20))[0]; j = int(cand[0]); r = Kk[0][j] > 0
    m0 = float(jmass(build(Jk, *Kk))[j]); r_ = Kk[0] > 0
    m1 = float(jmass(build(Jk, Kk[0], np.where(r_, de3, 0).astype(np.float32), np.where(r_, dp3, 0).astype(np.float32), Kk[3]))[j])
    fig, ax = plt.subplots(1, 2, figsize=(15, 7.0), gridspec_kw=dict(width_ratios=[1, 1.45]))
    a = ax[0]; s = 400 * Kk[0][j][r] / Kk[0][j][r].max()
    a.scatter(Kk[1][j][r], Kk[2][j][r], s=s, facecolors='none', edgecolors='0.5', label='particles (area ∝ pT)')
    mv = np.where(r & (np.hypot(de3[j] - Kk[1][j], dp3[j] - Kk[2][j]) > 1e-6))[0]
    pa, pb = int(Pk['a'][j]), int(Pk['b'][j])
    X5 = 5.0                                       # arrows drawn at 5x the displacement (stated in the legend)
    for q in mv:
        c = '#d95f02' if q in (pa, pb) else '#1b9e77'; x0, y0 = Kk[1][j][q], Kk[2][j][q]
        a.annotate('', xy=(x0 + X5 * (de3[j, q] - x0), y0 + X5 * (dp3[j, q] - y0)), xytext=(x0, y0),
                   arrowprops=dict(arrowstyle='->', color=c, lw=2.2))
    a.plot([], [], color='#d95f02', lw=2, label='pair 1 moved apart (pT-weighted spread +10 %)')
    a.plot([], [], color='#1b9e77', lw=2, label='pair 2 moved together until the invariant mass is restored')
    a.plot([], [], color='w', label='(arrows drawn at 5x the actual displacement)')
    a.set_xlabel('Δη to jet axis'); a.set_ylabel('Δφ to jet axis [rad]'); a.set_aspect('equal', 'datalim')
    a.set_title(f'E1m on one W jet (pT {Jk["jet_pt"][j]:.0f} GeV): particles rearranged,\n'
                f'invariant mass {m0:.1f} GeV before, {m1:.1f} GeV after', fontsize=10)
    a.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=1, frameon=False)
    a = ax[1]
    fams = [('E1_raise', 'E1 spread +2–10 %\n(real mass increase, reference)'), ('BOOST', 'BOOST rapidity ±0.06\n(mass fixed)'),
            ('E1m_raise', 'E1m, spread +2–10 %\n(mass restored)'), ('E1m_lower', 'E1m, spread −2–10 %\n(mass restored)'),
            ('E1m_antisym', 'E1m pooled\n(raise − lower)'), ('E5', 'E5 null rearrangement\n(mass ~fixed)')]
    mods = [m for m in (res['main_lgatr'], 'LGATr_it174999', 'ParT_kin', 'ParT_full') if m in res['pooled']]
    cols = {'LGATr_final': '#1f4e9c', 'LGATr_it174999': '#6b8fd6', 'LGATr_untrained': '0.6', 'ParT_kin': '#8e44ad', 'ParT_full': '#d35400'}
    labs = {'LGATr_final': 'L-GATr (final)', 'LGATr_it174999': 'L-GATr (iteration 174999)', 'LGATr_untrained': 'L-GATr untrained',
            'ParT_kin': 'ParT_kin', 'ParT_full': 'ParT_full'}
    for i, (fn, lab) in enumerate(fams):
        for q, m in enumerate(mods):
            v = res['pooled'][m].get(f'{fn}|A')
            if v is None or v['R_ci'][0] is None: continue
            yy = i + (q - (len(mods) - 1) / 2) * 0.15; lo, hi = v['R_ci']; x = float(np.clip(v['R'], -2.4, 2.4))
            a.errorbar(x, yy, xerr=[[max(x - max(lo, -2.5), 0)], [max(min(hi, 2.5) - x, 0)]], fmt='o', color=cols.get(m, 'k'), ms=5, capsize=2,
                       label=labs.get(m, m) if i == 0 else None)
    a.axvspan(-0.1, 0.1, color='0.88', zorder=0); a.axvline(0, color='k', lw=0.6); a.axvline(1, color='k', lw=0.6, ls=':')
    a.set_xlim(-2.5, 2.5)
    a.set_yticks(range(len(fams))); a.set_yticklabels([f[1] for f in fams], fontsize=9); a.invert_yaxis()
    a.set_xlabel('R = change of (logit W − logit Z) under the edit\n÷ change under a real mass change (matched angular dilation) of the same predicted size')
    a.set_title(TITLE + '\n(95 % bootstrap intervals; grey band |R| < 0.1)', fontsize=10)
    a.legend(fontsize=8, loc='lower right')
    fig.tight_layout(); f = ROOT / 'lgatr' / f'{NODE}_lgatr_c6{TAG}.png'; fig.savefig(f, dpi=110); log('wrote', f)


if __name__ == '__main__':
    {'lgatr': stage_lgatr, 'part': stage_part, 'analyze': stage_analyze, 'sample': get_sample}[ARGS.stage]()
