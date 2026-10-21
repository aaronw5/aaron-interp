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

  ./sync.sh && ./part_job.sh c-p46-run '/j-jepa-vol/aaron-interp/env-lgatr/bin/python -m interp.p46_lgatr_c6 --stage lgatr --ckpts it174999,it154999,untrained && python -m interp.p46_lgatr_c6 --stage part && python -m interp.p46_lgatr_c6 --stage analyze'
-> $AI_RESULTS/lgatr/p46_lg_<tags>.npz, $AI_RESULTS/part/p46_part.npz, $AI_RESULTS/lgatr/p46_lgatr_c6.json, p46_lgatr_c6.png
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
TITLE = os.environ.get('P46_TITLE', 'W-vs-Z response to mass-fixed edits, relative to a real mass change (grey band: |R| < 0.1)')
T0 = time.time()


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
            pth = f'{LGRUN}/model_run0_{t}.pt'; assert os.path.exists(pth), pth; models[f'LGATr_{t}'] = P.LG(pth, dev)
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
    S['meta|y'] = yw; f = ROOT / 'lgatr' / f'p46_lg_{"_".join(tags)}{TAG}'
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
    out['meta'] = M; S['meta|y'] = WZ['label'].astype(int); f = ROOT / 'part' / f'p46_part{TAG}'
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
    files = sorted(glob.glob(str(ROOT / 'lgatr' / f'p46_lg_*{TAG}.npz'))) + [str(ROOT / 'part' / f'p46_part{TAG}.npz')]
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
    main_lg = 'LGATr_it199999' if 'LGATr_it199999' in models else ('LGATr_it174999' if 'LGATr_it174999' in models else None)
    res = dict(node='P46', sample=f'val_5M {{WToQQ,ZToQQ}}_{FIDX}.root, {PER_WZ}/class, default_rng({SEED}); 10-class: {PER_ACC}/class files {AFILE}',
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
        n5 = delta(m, 'E5+', 'A') + delta(m, 'E5-', 'A'); rows5 = S['meta|E5+|rows']
        d5 = (delta(m, 'E4|E1_+0.10', 'A') - delta(m, 'E4|E1_-0.10', 'A')) * rows5
        b = (C @ n5) / (C @ d5); BOOTS[(m, 'E5', 'A')] = b; po['E5|A'] = dict(R=float(n5.sum() / d5.sum()), R_ci=ci(b))
        res['pooled'][m] = po
        log(m, {k: (round(v['R'], 3), [round(x, 3) for x in v['R_ci']]) for k, v in po.items() if k.endswith('|A')})
    # comparisons L-GATr vs ParT references (paired over the same bootstrap resamples)
    if main_lg:
        for ref in [x for x in models if x != main_lg]:
            cmp = {}
            for fn in ('PHI', 'BOOST', 'E6_0.01-0.06', 'E1_raise', 'E1_lower', 'E1m_raise', 'E1m_lower', 'E5'):
                b1, b2 = BOOTS.get((main_lg, fn, 'A')), BOOTS.get((ref, fn, 'A'))
                if b1 is None or b2 is None: continue
                k1 = 'E5|A' if fn == 'E5' else f'{fn}|A'
                cmp[fn] = dict(diff=res['pooled'][main_lg][k1]['R'] - res['pooled'][ref][k1]['R'], ci=ci(b1 - b2), p=boot_p(b1 - b2))
            ks = list(cmp)
            for k, h in zip(ks, holm([cmp[k]['p'] for k in ks])): cmp[k]['p_holm'] = h
            res['compare'][f'{main_lg}_minus_{ref}'] = cmp
    out = ROOT / 'lgatr' / f'p46_lgatr_c6{TAG}.json'; out.write_text(json.dumps(res, indent=1, default=float)); log('wrote', out)
    try: plot(res, S)
    except Exception:
        import traceback; traceback.print_exc()


def plot(res, S):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from interp.part_core import kin, build
    WZ, _ = get_sample(); K = kin(WZ); pt, de, dp, ms = K
    j = int(np.where((WZ['label'] == iW) & ((pt > 0).sum(1) >= 25))[0][0]); r = pt[j] > 0
    k = np.sqrt(np.exp(S['meta|E4|PHI+0.06|dlogM2'][j]))
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw=dict(width_ratios=[1, 1.5]))
    a = ax[0]; s = 400 * pt[j][r] / pt[j][r].max()
    a.scatter(de[j][r], dp[j][r], s=s, facecolors='none', edgecolors='0.45', label='unedited')
    a.scatter(de[j][r], dp[j][r] + DPHI, s=s, color='#d95f02', alpha=0.6, label='PHI +0.06: every particle phi + 0.06 rad')
    a.scatter(de[j][r] * k, dp[j][r] * k, s=s * 0.5, marker='x', color='#1b9e77', label=f'matched dilation: angles x {k:.3f}')
    a.plot([0], [0], 'k+', ms=16, mew=2, label='stored jet axis')
    a.set_xlabel('Δη to stored axis'); a.set_ylabel('Δφ to stored axis [rad]'); a.set_aspect('equal', 'datalim')
    a.set_title(f'One W jet (pT {WZ["jet_pt"][j]:.0f} GeV): edit vs matched real mass change\n'
                'L-GATr sees PHI +0.06 as a rotation about the beam (same jet)', fontsize=10)
    a.legend(fontsize=8, loc='lower left')
    a = ax[1]
    fams = [('PHI', 'PHI ±0.06 rad\n(rotation about beam)'), ('BOOST', 'BOOST ±0.06\n(rapidity shift)'),
            ('E6_0.01-0.06', 'E6 0.01–0.06 rad\n(stored axis moved)'), ('E1_raise', 'E1 spread +2–10 %\n(mass changes)'),
            ('E1m_raise', 'E1m spread +2–10 %\n(mass restored)'), ('E5', 'E5 null\n(M2 fixed)')]
    mods = [m for m in (res['main_lgatr'], 'LGATr_it154999', 'LGATr_untrained', 'ParT_kin_s0', 'ParT_full_s0', 'ParT_kin', 'ParT_full') if m in res['pooled']]
    cols = {'LGATr_it199999': '#1f4e9c', 'LGATr_it174999': '#1f4e9c', 'LGATr_it154999': '#6b8fd6', 'LGATr_untrained': '0.6',
            'ParT_kin_s0': '#c0392b', 'ParT_full_s0': '#e67e22', 'ParT_kin': '#8e44ad', 'ParT_full': '#d35400'}
    for i, (fn, lab) in enumerate(fams):
        for q, m in enumerate(mods):
            key = 'E5|A' if fn == 'E5' else f'{fn}|A'; v = res['pooled'][m].get(key)
            if v is None: continue
            yy = i + (q - (len(mods) - 1) / 2) * 0.11; lo, hi = v['R_ci']
            a.errorbar(v['R'], yy, xerr=[[v['R'] - lo], [hi - v['R']]], fmt='o', color=cols.get(m, 'k'), ms=5, capsize=2,
                       label=m.replace('_', ' ') if i == 0 else None)
    a.axvspan(-0.1, 0.1, color='0.85', zorder=0); a.axvline(0, color='k', lw=0.6); a.axvline(1, color='k', lw=0.6, ls=':')
    a.set_yticks(range(len(fams))); a.set_yticklabels([f[1] for f in fams], fontsize=9); a.invert_yaxis()
    a.set_xlabel('R = W-vs-Z logit response to the edit / response to a real mass change of the same predicted size')
    a.set_title(TITLE, fontsize=10)
    a.legend(fontsize=8, loc='lower right')
    fig.tight_layout(); f = ROOT / 'lgatr' / f'p46_lgatr_c6{TAG}.png'; fig.savefig(f, dpi=110); log('wrote', f)


if __name__ == '__main__':
    {'lgatr': stage_lgatr, 'part': stage_part, 'analyze': stage_analyze, 'sample': get_sample}[ARGS.stage]()
