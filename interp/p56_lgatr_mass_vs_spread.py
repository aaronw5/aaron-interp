"""P56 (loop C, round C13; claims C6 v2, C22; ParT references C3 / C18 v3): invariant mass vs spread about L-GATr's own momentum
axis.

Question. P51 / P51v showed that L-GATr ignores the stored jet axis. Every fixed-mass edit so far also kept the pT-weighted angular
spread about L-GATr's OWN axis (the direction of the summed particle three-momenta, about which its per-particle Δη, Δφ, ΔR scalar
features are computed), and for nearly massless particles at small angles the jet invariant mass m and that spread are nearly the
same number (m² ≈ M0·M2). Here the two are separated by changing particle masses (energies) at fixed three-momenta.

Sample. JetClass test_20M WToQQ_117 / ZToQQ_117 (W/Z arm) and TTBar_117 (wider-angle top arm; the planned val_5M *_125 files do not
exist, val_5M has _120-_124 only, all used before; test_20M _117 is not used by any earlier node), 2,000 random rows per file, numpy
default_rng(5656) (never first-k; not L-GATr's training or validation files); jets with < 3 particles dropped. Accuracy check on
the P1 sample (part_core.sample(400, 3000), test_20M, 4,000 jets, 10 classes).

Notation. Particle i: pT_i, pseudorapidity η_i, azimuth φ_i, three-momentum p_i (fixed unless stated), mass m_i, energy
E_i = sqrt(|p_i|² + m_i²). Own axis: (η_J, φ_J) of Σ p_i (pseudorapidity, as in the L-GATr embedding). M2_own = Σ pT_i ΔR_i² about
the own axis. m = jet invariant mass of Σ (E_i, p_i). All edits keep the stored axis (jet_eta, jet_phi); four-vectors are rebuilt
in float64 and cast to float32.

Edits (doses f = +2, +5, +10 % in m):
 M_f    particle-mass edit: every particle keeps p_i; m_i'² = m_i² + μ² with one μ per jet (bisection) so that m' = (1 + f) m.
        Directions, pT, Σ p, the own axis and M2_own are unchanged; only E_i (log E, log E/E_jet, pair masses) change.
        (The plan wrote E_i' = sqrt(|p_i|² + μ²); JetClass charged hadrons carry m_i = 0.1396 GeV and neutral hadrons up to
        ~0.5 GeV, so μ is added in quadrature to keep every particle at least as heavy as before.)
 E4_f   matched dilation (real-mass calibration): every particle's (η − η_J, Δφ) about the own axis scaled by k, k per jet
        (bisection) so that m' = (1 + f) m; particle masses fixed. The same Δm as M_f, made by opening angles.
 N_f    energy-shuffle null: ΔE_i of M_f with signs ± chosen per particle (greedy balance, largest |ΔE| first; minus only where
        E_i − |ΔE_i| > 0) and the larger-sum group scaled by c ≤ 1 so that Σ ΔE = 0: m and Σ p unchanged, energy features changed.
 X_f    outward move only: the two hardest particles' offsets from the own axis scaled by (1 + a), a per jet so that m' = (1 + f) m.
 S_f    spread up at fixed m (f = 2, 5 %): X_f, then the masses of all other particles lowered, m_j'² = λ m_j² (λ ∈ [0, 1] per
        jet) until m' = m (float64 residual reported). Only jets where removing those masses can absorb the rise (m(λ=0) ≤ m)
        are used (feasible subset, N reported).
 Srel   all-jet version of S5: d(X5) − d(M5): two jets with the same m ((1.05) m), one with a larger spread (X5), one with heavier
        particles (M5).
 ZERO   real-particle-mass reference: every particle massless (p fixed);  PION: every particle given m_i = 0.1396 GeV (p fixed).
        Each with its own matched dilation E4|ZERO, E4|PION (same Δm).
Read-out d = ℓ_W − ℓ_Z (W/Z arm) and d_t = ℓ_Tbqq − ℓ_QCD (top arm). R(edit) = Σ_jets Δd(edit) / Σ_jets Δd(E4 of the same Δm):
1 = responds as to a real mass change; 0 = no response.
Predictions: H_inv (L-GATr reads invariant mass): R_M ≈ 1, R_S ≈ 0. H_spread (reads the spread about its own axis through its
scalar features): R_M ≈ R_N, R_S ≈ 1, route F carries the response. ParT_full / ParT_kin / PN_full (read the spread about the stored
axis, C3 / C18 v3): R_M ≈ 0, R_S ≈ 1.
Routes (L-GATr final): A full edit; D edited four-vectors, the 7 scalar features (log pT, log E, log pT/pT_J, log E/E_J, Δφ, Δη, ΔR
about the own axis) from the unedited jet; F unedited four-vectors, features from the edited jet.
Statistics: 2,000 bootstrap resamples over jets (shared count matrix, paired across edits / models); exact McNemar on per-jet
correctness (W/Z arm), Holm over all arms × models; bootstrap p for R_M = 1, R_M = R_N, R_S = 0, R_S = 1. Off-manifold: C2ST
(histogram likelihood-ratio classifier, 2D (log|p|, log E) per particle, edited jets vs other unedited jets; AUC 0.5 = same).

  ./sync.sh && ./part_job.sh c-p56-run '/j-jepa-vol/aaron-interp/env-lgatr/bin/python -m interp.p56_lgatr_mass_vs_spread --stage lgatr && python -m interp.p56_lgatr_mass_vs_spread --stage part && python -m interp.p56_lgatr_mass_vs_spread --stage analyze'
-> $AI_RESULTS/lgatr/p56_lg.npz, p56_part.npz, p56_lgatr_mass_vs_spread.json, p56_lgatr_mass_vs_spread.png
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('JETDISTILL_TAGGER', 'part')
ap = argparse.ArgumentParser(); ap.add_argument('--stage', required=True); ap.add_argument('--smoke', action='store_true')
ARGS = ap.parse_args(); SMOKE = ARGS.smoke
SEED, FIDX = 5656, 117
PER = 120 if SMOKE else 2000
NB = 200 if SMOKE else 2000
FS = [0.01, 0.02, 0.05, 0.10]
FS_S = [0.01, 0.02, 0.05]
MPI = 0.1396
VAL = Path('/j-jepa-vol/JetClass/Pythia/test_20M')
CL = ['QCD', 'Hbb', 'Hcc', 'Hgg', 'H4q', 'Hqql', 'Zqq', 'Wqq', 'Tbqq', 'Tbl']
iW, iZ, iT, iQ = CL.index('Wqq'), CL.index('Zqq'), CL.index('Tbqq'), CL.index('QCD')
FILES = {'wz': ['WToQQ', 'ZToQQ'], 'tt': ['TTBar']}
ROOT = Path(os.environ.get('AI_RESULTS', 'results')) / 'lgatr'
TAG = '_smoke' if SMOKE else ''
LGRUN = '/j-jepa-vol/aaron-interp/models/lgatr/runs/jc2m/jc2m_s0a/models'
PNDIR = Path('/j-jepa-vol/aaron-interp/models/particlenet')
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
    from interp.part_core import sub, cat
    f = ROOT / f'p56_sample{TAG}.npz'
    if f.exists():
        z = np.load(f); out = {'wz': {}, 'tt': {}}
        for k in z.files: out[k[:2]][k[4:]] = z[k]
        log('sample from cache', {s: len(v['label']) for s, v in out.items()}); return out
    import uproot
    from concurrent.futures import ProcessPoolExecutor
    rng = np.random.default_rng(SEED); jobs = []
    for pre in ('WToQQ', 'ZToQQ', 'TTBar'):
        p = str(VAL / f'{pre}_{FIDX}.root'); N = uproot.open(p)['tree'].num_entries
        jobs.append((p, np.sort(rng.choice(N, PER, replace=False))))
    log('reading', [(Path(p).name, len(r)) for p, r in jobs])
    with ProcessPoolExecutor(3) as ex: Js = list(ex.map(_read_rows, jobs))
    out = {'wz': cat(Js[:2]), 'tt': Js[2]}
    for s, J in out.items():
        ok = (J['mask'] > 0).sum(1) >= 3; log(s, 'jets with >= 3 particles', int(ok.sum()), 'of', len(ok)); out[s] = sub(J, np.where(ok)[0])
    assert set(np.unique(out['wz']['label'])) == {iW, iZ} and set(np.unique(out['tt']['label'])) == {iT}
    ROOT.mkdir(parents=True, exist_ok=True)
    np.savez(f, **{f'{s}__{k}': v for s, J in out.items() for k, v in J.items()}); log('cached', f); return out


# ------------------------------------------------------------------------------------------------ geometry (float64)
wrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi


def geom(J):
    from interp.part_core import kin
    K = kin(J); pt, de, dp, ms = (a.astype(np.float64) for a in K); rm = pt > 0
    sg = np.where(J['jet_eta'] < 0, -1.0, 1.0)[:, None]
    eta = J['jet_eta'][:, None].astype(np.float64) + sg * de; phi = J['jet_phi'][:, None].astype(np.float64) + dp
    return dict(K=K, pt=pt, eta=np.where(rm, eta, 0), phi=np.where(rm, phi, 0), ms=np.where(rm, ms, 0), rm=rm, sg=sg)


def p4(G, eta, phi, ms=None, E=None):
    pt, rm = G['pt'], G['rm']; px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    if E is None: E = np.sqrt((pt * np.cosh(eta)) ** 2 + ms ** 2)
    return [np.where(rm, a, 0.0) for a in (E, px, py, pz)]


def minv(P):
    E, px, py, pz = (a.sum(-1) for a in P); return np.sqrt(np.maximum(E ** 2 - px ** 2 - py ** 2 - pz ** 2, 0))


def own_axis(G, eta, phi):
    _, px, py, pz = p4(G, eta, phi, G['ms']); Px, Py, Pz = px.sum(1), py.sum(1), pz.sum(1); PT = np.hypot(Px, Py)
    return np.arcsinh(Pz / PT), np.arctan2(Py, Px)


def M2own(G, eta, phi):
    eJ, pJ = own_axis(G, eta, phi)
    return (G['pt'] * ((eta - eJ[:, None]) ** 2 + wrap(phi - pJ[:, None]) ** 2) * G['rm']).sum(1)


def mk(J, G, eta, phi, ms, E=None):
    """jet dict (part_core.build convention, stored axis kept) with particles at (eta, phi), masses ms; energies from float64"""
    from interp.part_core import build
    rm = G['rm']; je, jp = J['jet_eta'][:, None].astype(np.float64), J['jet_phi'][:, None].astype(np.float64)
    de = np.where(rm, G['sg'] * (eta - je), 0); dp = np.where(rm, wrap(phi - jp), 0)
    pt32 = G['K'][0]
    Je = build(J, pt32, de.astype(np.float32), dp.astype(np.float32), np.where(rm, ms, 0).astype(np.float32))
    if E is None: E = p4(G, eta, phi, ms)[0]
    o = np.argsort(-pt32, 1, kind='stable')[:, :128]; Es = np.take_along_axis(E, o, 1)
    Je['energy'] = np.where(Je['mask'] > 0, Es, 0).astype(np.float32); Je['jet_energy'] = Je['energy'].sum(1).astype(np.float32)
    return Je


def solve(fun, lo, hi, tgt, it=90):
    """per-jet bisection for increasing fun: fun(x) = tgt, x in [lo, hi]; returns x, valid"""
    lo, hi = np.broadcast_to(np.asarray(lo, float), tgt.shape).copy(), np.broadcast_to(np.asarray(hi, float), tgt.shape).copy()
    flo, fhi = fun(lo) - tgt, fun(hi) - tgt; ok = (flo <= 0) & (fhi >= 0)
    for _ in range(it):
        mid = 0.5 * (lo + hi); fm = fun(mid) - tgt; up = fm < 0
        lo = np.where(up, mid, lo); hi = np.where(up, hi, mid)
    return 0.5 * (lo + hi), ok


def edits(J, with_plot=False):
    """yields (name, rows (bool), Je for those rows, info dict). E4_f and the edits at dose f share the target m' = (1 + f) m."""
    from interp.part_core import sub, jmass
    G = geom(J); n = len(G['pt']); rm = G['rm']; eta, phi, ms, pt = G['eta'], G['phi'], G['ms'], G['pt']; r = np.arange(n)
    P0 = p4(G, eta, phi, ms); E0 = P0[0]; m0 = minv(P0); Ptot2 = sum(P0[i].sum(1) ** 2 for i in (1, 2, 3))
    pabs2 = np.where(rm, (pt * np.cosh(eta)) ** 2, 0); eJ, pJ = own_axis(G, eta, phi); M20 = M2own(G, eta, phi)
    ALL = np.ones(n, bool)
    Jb = mk(J, G, eta, phi, ms); mb32 = jmass(Jb).astype(np.float64)
    PLOT = {}

    def info(rows, Je, eta_, phi_, m_new, extra=None):
        if not rows.any(): return dict(n=0, **(extra or {}))
        dlm = np.log(m_new[rows] / m0[rows]); dl32 = np.log(np.maximum(jmass(Je).astype(np.float64), 1e-6) / np.maximum(mb32[rows], 1e-6))
        dM2 = np.log(M2own(G, eta_, phi_)[rows] / M20[rows])
        I = dict(n=int(rows.sum()), dlogm64_mean=float(dlm.mean()), dlogm32_mean=float(dl32.mean()),
                 dlogm32_minus64_maxabs=float(np.abs(dl32 - dlm).max()), dlogM2own_mean=float(dM2.mean()),
                 dlogM2own_maxabs=float(np.abs(dM2).max()), dlogM2own_median=float(np.median(dM2)))
        if extra: I.update(extra)
        return I

    def emit(name, rows, eta_, phi_, ms_, E_=None, extra=None):
        rows = rows & ALL; idx = np.where(rows)[0]
        Gs = {k: (v[idx] if isinstance(v, np.ndarray) and v.shape[:1] == (n,) else v) for k, v in G.items()}
        Gs['K'] = tuple(a[idx] for a in G['K']); Gs['sg'] = G['sg'][idx]
        E_ = p4(G, eta_, phi_, ms_)[0] if E_ is None else E_
        Je = mk(sub(J, idx), Gs, eta_[idx], phi_[idx], ms_[idx], E_[idx])
        m_new = minv(p4(G, eta_, phi_, ms_, E_))
        return name, rows, Je, info(rows, Je, eta_, phi_, m_new, extra)

    def dil(k):
        return eJ[:, None] + k[:, None] * (eta - eJ[:, None]), pJ[:, None] + k[:, None] * wrap(phi - pJ[:, None])

    yield 'base', ALL, Jb, dict(n=n, m0_median=float(np.median(m0)), M2own_median=float(np.median(M20)))
    # hardest two particles (slot indices) and their outward move
    o = np.argsort(-G['K'][0], 1, kind='stable'); h1, h2 = o[:, 0], o[:, 1]
    hard = np.zeros_like(rm); hard[r, h1] = True; hard[r, h2] = True

    def outward(a):
        e2, p2 = eta.copy(), phi.copy()
        for h in (h1, h2):
            e2[r, h] = eJ + (1 + a) * (eta[r, h] - eJ); p2[r, h] = pJ + (1 + a) * wrap(phi[r, h] - pJ)
        return e2, p2

    def m_E(u):
        return np.sqrt(np.maximum(np.where(rm, np.sqrt(pabs2 + ms ** 2 + u[:, None]), 0).sum(1) ** 2 - Ptot2, 0))

    def e4(name, tgt):
        k, ok = solve(lambda k: minv(p4(G, *dil(k), ms)), 0.2, 4.0, tgt)
        e_, p_ = dil(k); return emit(f'E4|{name}', ok, e_, p_, ms, extra=dict(k_median=float(np.median(k[ok]))))

    for f in FS:
        tgt = (1 + f) * m0
        yield e4(f'{f:+.2f}', tgt)
        # M_f
        u, okM = solve(m_E, 0.0, 400.0, tgt); mu = np.sqrt(u); msM = np.where(rm, np.sqrt(ms ** 2 + u[:, None]), 0)
        EM = p4(G, eta, phi, msM)[0]
        yield emit(f'M{f:+.2f}', okM, eta, phi, msM, extra=dict(mu_median=float(np.median(mu[okM])), mu_p5_p95=[float(x) for x in np.percentile(mu[okM], [5, 95])],
                                                               frac_mu_gt_mpi=float((mu[okM] > MPI).mean()), dE_jet_median=float(np.median((EM - E0).sum(1)[okM]))))
        if with_plot and f == 0.10: PLOT['M'] = dict(E0=E0, EM=EM, pabs=np.sqrt(pabs2), mu=mu, m0=m0, ok=okM)
        # N_f
        dE = np.where(rm, EM - E0, 0); od = np.argsort(-dE, 1); s = np.zeros(n); sign = np.zeros_like(dE)
        for j in range(dE.shape[1]):
            ix = od[:, j]; d = dE[r, ix]; real = rm[r, ix]; neg = real & (E0[r, ix] - d > 0) & (s > 0)
            sg_ = np.where(real, np.where(neg, -1.0, 1.0), 0.0); sign[r, ix] = sg_; s += sg_ * d
        Sp, Sn = (dE * (sign > 0)).sum(1), (dE * (sign < 0)).sum(1); okN = okM & (Sn > 0)
        cpos = np.where(Sp > Sn, Sn / np.maximum(Sp, 1e-300), 1.0); cneg = np.where(Sn >= Sp, Sp / np.maximum(Sn, 1e-300), 1.0)
        EN = E0 + np.where(sign > 0, cpos[:, None], cneg[:, None]) * sign * dE
        cc = np.minimum(cpos, cneg)[okN]
        yield emit(f'N{f:+.2f}', okN, eta, phi, ms, E_=EN, extra=dict(c_median=float(np.median(cc)), c_p5=float(np.percentile(cc, 5)),
                                                                     frac_particles_E_below_p=float(((EN < np.sqrt(pabs2) - 1e-12) & rm)[okN].sum() / rm[okN].sum()),
                                                                     max_abs_sumdE=float(np.abs((EN - E0).sum(1)[okN]).max())))
        # X_f (outward move of the two hardest particles)
        a, okX = solve(lambda a: minv(p4(G, *outward(a), ms)), 0.0, 40.0, tgt); eX, pX = outward(a)
        yield emit(f'X{f:+.2f}', okX, eX, pX, ms, extra=dict(a_median=float(np.median(a[okX]))))
        # S_f (X_f, then other particles' masses lowered until m is restored)
        if f in FS_S:
            def m_lam(lam):
                mm = np.where(hard, ms, np.sqrt(lam)[:, None] * ms); return minv(p4(G, eX, pX, mm))
            lam, okS = solve(m_lam, 0.0, 1.0, m0); okS &= okX; msS = np.where(hard, ms, np.sqrt(lam)[:, None] * ms)
            resid = np.abs(m_lam(lam) / m0 - 1)[okS]
            yield emit(f'S{f:+.2f}', okS, eX, pX, msS, extra=dict(lam_median=float(np.median(lam[okS])) if okS.any() else None,
                                                                   frac_feasible=float(okS.mean()), max_rel_m_resid64=float(resid.max()) if okS.any() else None))
            if with_plot and f == 0.02: PLOT['S'] = dict(eX=eX, pX=pX, msS=msS, ok=okS, eJ=eJ, pJ=pJ, h1=h1, h2=h2)
    # real-particle-mass references
    for nm, msR in (('ZERO', np.zeros_like(ms)), ('PION', np.where(rm, MPI, 0.0))):
        mR = minv(p4(G, eta, phi, msR)); okR = mR > 0
        yield emit(nm, okR, eta, phi, msR)
        yield e4(nm, mR)
    if with_plot: PLOT.update(G=G, eta=eta, phi=phi, ms=ms); yield 'PLOT', None, None, PLOT


def readouts(L):
    return L[:, iW] - L[:, iZ], L[:, iT] - L[:, iQ]


def put(S, smp, model, edit, route, rows, L):
    N = len(rows); a = np.full(N, np.nan, np.float32); b = a.copy(); p = np.full(N, -1, np.int8)
    dw, dt = readouts(L); a[rows], b[rows], p[rows] = dw, dt, L.argmax(1)
    S[f'{smp}|{model}|{edit}|{route}|dWZ'] = a; S[f'{smp}|{model}|{edit}|{route}|dTQ'] = b; S[f'{smp}|{model}|{edit}|{route}|p'] = p


def acc_p1(fn):
    """10-class accuracy on the P1 sample (part_core.sample(400, 3000), rebuilt with build)"""
    from interp.part_core import sample, build, kin
    J = sample(30 if SMOKE else 400, 3000); Jb = build(J, *kin(J)); L = fn(Jb); return float((L.argmax(1) == J['label']).mean()), int(len(L))


# ------------------------------------------------------------------------------------------------ stage: L-GATr
def stage_lgatr():
    import torch
    sys.argv = [sys.argv[0], '--stage', 'none']
    import interp.p44_lgatr_grades as P
    from interp.part_core import sub
    SM = get_sample()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = {'LGATr_final': P.LG(f'{LGRUN}/model_run0.pt', dev)}
    ms_ = torch.manual_seed; torch.manual_seed = lambda s: ms_(SEED)
    try: models['LGATr_untrained'] = P.LG(None, dev)
    finally: torch.manual_seed = ms_
    log('loaded L-GATr final + untrained on', dev)
    S, out = {}, dict(acc_P1={}, acc_sample={}, rerun_floor={}, meta={})
    for mn, lg in models.items():
        out['acc_P1'][mn] = acc_p1(lg.logits); log(mn, 'P1-sample 10-class accuracy', out['acc_P1'][mn])
    for smp, J in SM.items():
        y = J['label'].astype(int); M = {}
        for name, rows, Je, I in edits(J):
            M[name] = I; S[f'{smp}|meta|{name}|rows'] = rows
            if name == 'base':
                Jb = Je
                for mn, lg in models.items():
                    L = lg.logits(Je); put(S, smp, mn, 'base', 'A', rows, L); L2 = lg.logits(Je)
                    out['rerun_floor'][f'{smp}|{mn}'] = float(np.abs(readouts(L2)[0] - readouts(L)[0]).max())
                    out['acc_sample'][f'{smp}|{mn}'] = float((L.argmax(1) == y).mean())
                log(smp, 'base', I, out['acc_sample']); continue
            Jbr = sub(Jb, np.where(rows)[0])
            Z = np.zeros((0, 10), np.float32)
            for mn, lg in models.items():
                put(S, smp, mn, name, 'A', rows, lg.logits(Je) if rows.any() else Z)
                if mn == 'LGATr_final':
                    put(S, smp, mn, name, 'D', rows, lg.logits(Je, Jbr) if rows.any() else Z)
                    put(S, smp, mn, name, 'F', rows, lg.logits(Jbr, Je) if rows.any() else Z)
            k = 'dWZ' if smp == 'wz' else 'dTQ'
            log(smp, name, {a: (round(b, 5) if isinstance(b, float) else b) for a, b in I.items()},
                {mn: round(float(np.nanmean(S[f'{smp}|{mn}|{name}|A|{k}'] - S[f'{smp}|{mn}|base|A|{k}'])), 4) for mn in models})
        out['meta'][smp] = M; S[f'{smp}|meta|y'] = y
    f = ROOT / f'p56_lg{TAG}'; np.savez(f.with_suffix('.npz'), **S); f.with_suffix('.json').write_text(json.dumps(out, indent=1, default=float)); log('wrote', f)


# ------------------------------------------------------------------------------------------------ stage: ParT / ParticleNet
def stage_part():
    import torch
    from interp.p39v_verify import runner
    from interp.part_core import ParTNetwork
    from interp import pn_net
    from jetdistill.part.network import FEATURES
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; Mdl = {}
    for nm, feat in (('ParT_full', 'full'), ('ParT_kin', 'kin')):
        net = ParTNetwork(feat); net.model.to(dev).eval(); Mdl[nm] = dict(kind='part', feat=feat, model=net.model)
    Mdl['PN_full'] = dict(kind='pn', feat='full', model=pn_net.load_particlenet(PNDIR / 'ParticleNet_full.pt', len(FEATURES['full']), dev).eval())
    names = list(Mdl); run = runner(Mdl, FEATURES, dev); log('loaded', names, dev)
    SM = get_sample(); S, out = {}, dict(acc_P1={}, acc_sample={})
    from interp.part_core import sample, build, kin
    J1 = sample(30 if SMOKE else 400, 3000); L1 = run(build(J1, *kin(J1)), names)
    for k in names: out['acc_P1'][k] = (float((L1[k].argmax(1) == J1['label']).mean()), int(len(J1['label']))); log(k, 'P1 accuracy', out['acc_P1'][k])
    for smp, J in SM.items():
        y = J['label'].astype(int)
        for name, rows, Je, I in edits(J):
            S[f'{smp}|meta|{name}|rows'] = rows; L = run(Je, names) if rows.any() else {k: np.zeros((0, 10), np.float32) for k in names}
            for k in names: put(S, smp, k, name, 'A', rows, L[k])
            if name == 'base':
                for k in names: out['acc_sample'][f'{smp}|{k}'] = float((L[k].argmax(1) == y).mean())
                log(smp, out['acc_sample']); continue
            kk = 'dWZ' if smp == 'wz' else 'dTQ'
            log(smp, name, {k: round(float(np.nanmean(S[f'{smp}|{k}|{name}|A|{kk}'] - S[f'{smp}|{k}|base|A|{kk}'])), 4) for k in names})
    f = ROOT / f'p56_part{TAG}'; np.savez(f.with_suffix('.npz'), **S); f.with_suffix('.json').write_text(json.dumps(out, indent=1, default=float)); log('wrote', f)


# ------------------------------------------------------------------------------------------------ stage: analyze
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.zeros(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj.tolist()


def ci(b):
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] if len(b) else [None, None]


def boot_p(b, v0=0.0):
    b = np.asarray(b, float) - v0; b = b[np.isfinite(b)]
    return float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()) + 1 / len(b))) if len(b) else None


def mcnemar(a, b):
    from scipy.stats import binomtest
    x, z = int((a & ~b).sum()), int((~a & b).sum()); return dict(lost=x, gained=z, p=float(binomtest(x, x + z).pvalue) if x + z else 1.0)


def c2st(Xa, Xb, rng, nb=30):
    """histogram likelihood-ratio classifier on 2D features; train on half, AUC on the other half (0.5 = indistinguishable)"""
    n = min(len(Xa), len(Xb), 40000); Xa = Xa[rng.choice(len(Xa), n, replace=False)]; Xb = Xb[rng.choice(len(Xb), n, replace=False)]
    h = n // 2; X = np.concatenate([Xa, Xb]); edges = [np.quantile(X[:, j], np.linspace(0, 1, nb + 1)) for j in range(2)]
    for e in edges: e[0] -= 1e-9; e[-1] += 1e-9
    def b_(Z): return [np.clip(np.searchsorted(edges[j], Z[:, j]) - 1, 0, nb - 1) for j in range(2)]
    Ha = np.ones((nb, nb)); Hb = np.ones((nb, nb)); np.add.at(Ha, tuple(b_(Xa[:h])), 1); np.add.at(Hb, tuple(b_(Xb[:h])), 1)
    sc = np.log(Hb / Ha); sa, sb = sc[tuple(b_(Xa[h:]))], sc[tuple(b_(Xb[h:]))]
    from scipy.stats import mannwhitneyu
    return float(mannwhitneyu(sb, sa).statistic / (len(sa) * len(sb)))


def stage_analyze():
    S, JS = {}, {}
    for nm in ('p56_lg', 'p56_part'):
        f = ROOT / f'{nm}{TAG}.npz'; z = np.load(f); S.update({k: z[k] for k in z.files}); JS[nm] = json.loads(f.with_suffix('.json').read_text()); log('loaded', f)
    meta = JS['p56_lg']['meta']
    models = ['LGATr_final', 'LGATr_untrained', 'ParT_full', 'ParT_kin', 'PN_full']
    res = dict(node='P56', sample=f'test_20M WToQQ/ZToQQ/TTBar_{FIDX}.root, {PER} random rows per file, default_rng({SEED}); jets with >= 3 particles',
               NB=NB, FS=FS, FS_S=FS_S, readout=dict(wz='d = logit_W - logit_Z', tt='d_t = logit_Tbqq - logit_QCD'),
               R_def='sum_jets Delta d(edit) / sum_jets Delta d(matched own-axis dilation E4 with the same Delta m)', geometry=meta,
               accuracy=dict(P1={**JS['p56_lg']['acc_P1'], **JS['p56_part']['acc_P1']}, sample={**JS['p56_lg']['acc_sample'], **JS['p56_part']['acc_sample']},
                             rerun_floor=JS['p56_lg']['rerun_floor']), arms={}, tests={}, calls={})
    BO = {}
    mcn_p, mcn_k = [], []
    for smp in ('wz', 'tt'):
        y = S[f'{smp}|meta|y']; N = len(y); key = 'dWZ' if smp == 'wz' else 'dTQ'
        rb = np.random.default_rng(SEED + 3 + (smp == 'tt')); C = np.zeros((NB, N), np.float32)
        for b in range(NB): C[b] = np.bincount(rb.integers(0, N, N), minlength=N)
        rows = lambda e: S[f'{smp}|meta|{e}|rows']

        def dd(m, e, route='A'):
            k = f'{smp}|{m}|{e}|{route}|{key}'
            if k not in S: return None
            d = S[k].astype(np.float64) - S[f'{smp}|{m}|base|A|{key}'].astype(np.float64); return np.where(np.isfinite(d), d, 0.0)

        def ratio(m, pairs, route='A', minus=None):
            """pairs: [(edit, dilation)], rows = valid in both; minus: [(edit)] subtracted in the numerator (Srel)"""
            nu = np.zeros(N); de = np.zeros(N)
            for i, (e, d) in enumerate(pairs):
                rr = rows(e) & rows(d); x = dd(m, e, route)
                if x is None: return None
                if minus is not None: x = x - dd(m, minus[i], route); rr = rr & rows(minus[i])
                nu += x * rr; de += dd(m, d) * rr
            b = (C @ nu) / (C @ de); return float(nu.sum() / de.sum()), b, nu, de
        A = {}
        for m in models:
            o = {}
            arms = {f'M{f:+.2f}': [(f'M{f:+.2f}', f'E4|{f:+.2f}')] for f in FS}
            arms.update({f'N{f:+.2f}': [(f'N{f:+.2f}', f'E4|{f:+.2f}')] for f in FS})
            arms.update({f'X{f:+.2f}': [(f'X{f:+.2f}', f'E4|{f:+.2f}')] for f in FS})
            arms.update({f'S{f:+.2f}': [(f'S{f:+.2f}', f'E4|{f:+.2f}')] for f in FS_S})
            arms.update({'M_pooled': [(f'M{f:+.2f}', f'E4|{f:+.2f}') for f in FS], 'N_pooled': [(f'N{f:+.2f}', f'E4|{f:+.2f}') for f in FS],
                         'S_pooled': [(f'S{f:+.2f}', f'E4|{f:+.2f}') for f in FS_S], 'X_poolS': [(f'X{f:+.2f}', f'E4|{f:+.2f}') for f in FS_S],
                         'ZERO': [('ZERO', 'E4|ZERO')], 'PION': [('PION', 'E4|PION')]})
            for an, pr in arms.items():
                for route in ('A', 'D', 'F'):
                    q = ratio(m, pr, route)
                    if q is None: continue
                    v, b, nu, de = q; BO[(smp, m, an, route)] = b
                    o[f'{an}|{route}'] = dict(R=v, R_ci=ci(b), n=int((de != 0).sum()), mean_dd=float(nu.sum() / max((de != 0).sum(), 1)),
                                              mean_dd_E4=float(de.sum() / max((de != 0).sum(), 1)))
            q = ratio(m, [('X+0.05', 'E4|+0.05')], 'A', minus=['M+0.05'])
            v, b, _, _ = q; BO[(smp, m, 'Srel', 'A')] = b; o['Srel|A'] = dict(R=v, R_ci=ci(b))
            if m == 'LGATr_final':
                for route in ('D', 'F'):
                    v, b, _, _ = ratio(m, [('X+0.05', 'E4|+0.05')], route, minus=['M+0.05']); o[f'Srel|{route}'] = dict(R=v, R_ci=ci(b))
                # real-mass route split of the dilation: R_D, R_F = sum dd_route(E4) / sum dd_A(E4)
                for f in FS:
                    e = f'E4|{f:+.2f}'; a_ = dd(m, e).sum()
                    o[f'{e}|route_share'] = dict(D=float(dd(m, e, 'D').sum() / a_), F=float(dd(m, e, 'F').sum() / a_))
            A[m] = o
            # tests (H_inv vs H_spread)
            bM, bN, bS = BO.get((smp, m, 'M_pooled', 'A')), BO.get((smp, m, 'N_pooled', 'A')), BO.get((smp, m, 'S_pooled', 'A'))
            t = dict(p_RM_eq_1=boot_p(bM, 1.0), p_RM_eq_RN=boot_p(bM - bN), RM_minus_RN=dict(v=o['M_pooled|A']['R'] - o['N_pooled|A']['R'], ci=ci(bM - bN)),
                     p_RS_eq_0=boot_p(bS, 0.0), p_RS_eq_1=boot_p(bS, 1.0), p_Srel_eq_0=boot_p(BO[(smp, m, 'Srel', 'A')], 0.0),
                     p_Srel_eq_1=boot_p(BO[(smp, m, 'Srel', 'A')], 1.0))
            res['tests'][f'{smp}|{m}'] = t
            # calls
            cc = {}
            p0 = S[f'{smp}|{m}|base|A|p']
            for e in [x for x in meta[smp] if x not in ('base',)]:
                rr = rows(e)
                if not rr.any(): continue
                p1 = S[f'{smp}|{m}|{e}|A|p'][rr]; pb = p0[rr]; yr = y[rr]; w = yr == iW
                c = dict(n=int(rr.sum()), calls_changed=float((pb != p1).mean()), mcnemar=mcnemar(pb == yr, p1 == yr))
                if smp == 'wz': c['WtoZ'] = [float((pb[w] == iZ).mean()), float((p1[w] == iZ).mean())]
                else: c['T_called_T'] = [float((pb == iT).mean()), float((p1 == iT).mean())]
                cc[e] = c; mcn_p.append(c['mcnemar']['p']); mcn_k.append((smp, m, e))
            res['calls'][f'{smp}|{m}'] = cc
            rd = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 3)
            log(smp, m, {k: (rd(v['R']), [rd(x) for x in v['R_ci']]) for k, v in o.items() if '|' in k and k.split('|')[1] in ('A', 'D', 'F')})
        res['arms'][smp] = A
    for (smp, m, e), h in zip(mcn_k, holm(mcn_p)): res['calls'][f'{smp}|{m}'][e]['mcnemar']['p_holm'] = h
    res['n_mcnemar_tests_holm'] = len(mcn_p)
    # off-manifold check (C2ST) on the W/Z sample, per particle (log|p|, log E); edited jets vs the other half of unedited jets
    SM = get_sample(); J = SM['wz']; rng = np.random.default_rng(SEED + 9); half = rng.random(len(J['label'])) < 0.5
    feats = lambda Je, sel: (lambda m_: np.stack([np.log(np.sqrt(Je['px'] ** 2 + Je['py'] ** 2 + Je['pz'] ** 2)[m_]), np.log(Je['energy'][m_])], 1))((Je['mask'] > 0) & sel[:, None])
    def featm(Je, sel):
        m_ = (Je['mask'] > 0) & sel[:, None]; p2 = (Je['px'].astype(np.float64) ** 2 + Je['py'] ** 2 + Je['pz'] ** 2); q = Je['energy'].astype(np.float64) ** 2 - p2
        return np.stack([0.5 * np.log(p2[m_]), (np.sign(q) * np.sqrt(np.abs(q)))[m_]], 1)
    c2 = {}; c2m = {}; PLOT = None
    for name, rr, Je, I in edits(J, with_plot=True):
        if name == 'PLOT': PLOT = I; continue
        if name == 'base':
            Jb = Je; Xu = feats(Jb, ~half); c2['base_vs_base'] = c2st(feats(Jb, half), Xu, rng)
            Xum = featm(Jb, ~half); c2m['base_vs_base'] = c2st(featm(Jb, half), Xum, rng); continue
        if name.startswith(('M', 'N', 'S', 'ZERO', 'PION', 'E4|+0.05', 'X+0.05')):
            sel = half[rr]; Xe = feats(Je, sel); Xe = Xe[np.isfinite(Xe).all(1)]
            if len(Xe) >= 500: c2[name] = c2st(Xe, Xu, rng); c2m[name] = c2st(featm(Je, sel), Xum, rng)
    res['c2st_auc'] = c2; res['c2st_auc_mass'] = c2m; res['c2st_def'] = dict(c2st_auc='(log|p|, log E) per particle', c2st_auc_mass='(log|p|, signed sqrt|E^2 - p^2|) per particle')
    log('C2ST AUC (log|p|, log E)', {k: round(v, 3) for k, v in c2.items()}); log('C2ST AUC (log|p|, particle mass)', {k: round(v, 3) for k, v in c2m.items()})
    out = ROOT / f'p56_lgatr_mass_vs_spread{TAG}.json'; out.write_text(json.dumps(res, indent=1, default=float)); log('wrote', out)
    try: plot(res, J, PLOT)
    except Exception:
        import traceback; traceback.print_exc()


def plot(res, J, P):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    G = P['G']; y = J['label']; rm = G['rm']
    cand = np.where(P['M']['ok'] & P['S']['ok'] & (y == iW) & (rm.sum(1) >= 25))[0]; j = int(cand[0]); r = rm[j]
    fig, ax = plt.subplots(1, 3, figsize=(19, 7.4), gridspec_kw=dict(width_ratios=[1, 1, 1.35]))
    a = ax[0]; pab = P['M']['pabs'][j][r]
    a.scatter(pab, P['M']['E0'][j][r] - pab + 1e-6, s=14, color='0.55', label='before: E − |p| (original masses; photons ~0)')
    a.scatter(pab, P['M']['EM'][j][r] - pab, s=14, color='#1f4e9c', label=f'after mass edit M +10 %: μ = {P["M"]["mu"][j]:.2f} GeV added')
    a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('particle momentum |p| [GeV] (unchanged)'); a.set_ylabel('E − |p| [GeV]')
    a.set_title(f'(a) one W jet: energies raised at fixed momenta\njet mass {P["M"]["m0"][j]:.1f} → {1.1 * P["M"]["m0"][j]:.1f} GeV; directions unchanged', fontsize=10)
    a.legend(fontsize=8, loc='lower left')
    a = ax[1]; eJ, pJ = P['S']['eJ'][j], P['S']['pJ'][j]
    x0, y0 = (G['eta'][j] - eJ)[r], wrap(G['phi'][j] - pJ)[r]; x1, y1 = (P['S']['eX'][j] - eJ)[r], wrap(P['S']['pX'][j] - pJ)[r]
    s = 300 * G['pt'][j][r] / G['pt'][j][r].max(); lowered = (P['S']['msS'][j] < G['ms'][j] - 1e-9)[r]
    a.scatter(x0, y0, s=s, facecolors='none', edgecolors='0.55', label='particles (area ∝ pT)')
    a.scatter(x0[lowered], y0[lowered], s=10, color='#d95f02', label='mass lowered to keep the jet mass fixed')
    for q in np.where(np.hypot(x1 - x0, y1 - y0) > 1e-9)[0]:
        a.annotate('', xy=(x0[q] + 5 * (x1[q] - x0[q]), y0[q] + 5 * (y1[q] - y0[q])), xytext=(x0[q], y0[q]), arrowprops=dict(arrowstyle='->', color='#1b9e77', lw=2.2))
    a.plot([], [], color='#1b9e77', lw=2, label='two hardest particles moved outward (arrows 5x actual)')
    a.plot(0, 0, 'k+', ms=12, label='own axis (Σ p)')
    a.set_xlabel('Δη to own axis'); a.set_ylabel('Δφ to own axis [rad]'); a.set_aspect('equal', 'datalim')
    a.set_title('(b) same jet, spread-up edit S +2 %:\nspread about the own axis up, jet mass unchanged', fontsize=10)
    a.legend(fontsize=8, loc='best')
    a = ax[2]; A = res['arms']['wz']
    arms = [('M_pooled', 'M: heavier particles, m +2–10 %\n(spread fixed)'), ('N_pooled', 'N: energy shuffle\n(m and spread fixed)'),
            ('S_pooled', 'S: spread up, m fixed\n(feasible jets, doses 1–5 %)'), ('Srel', 'X5 − M5: spread up at\nequal m (all jets)'), ('X+0.05', 'X: two hardest outward\n(spread and m up 5 %)'),
            ('PION', 'all particles given pion mass\n(real-mass reference)')]
    mods = [('LGATr_final', 'L-GATr', '#1f4e9c'), ('LGATr_untrained', 'L-GATr untrained', '0.6'), ('ParT_full', 'ParT_full', '#d35400'),
            ('ParT_kin', 'ParT_kin', '#8e44ad'), ('PN_full', 'ParticleNet_full', '#27ae60')]
    for i, (an, lab) in enumerate(arms):
        for q, (m, ml, c) in enumerate(mods):
            v = A[m].get(f'{an}|A')
            if v is None or v['R_ci'][0] is None: continue
            yy = i + (q - 2) * 0.15; lo, hi = v['R_ci']; x = float(np.clip(v['R'], -1.9, 2.9))
            a.errorbar(x, yy, xerr=[[max(x - max(lo, -2), 0)], [max(min(hi, 3) - x, 0)]], fmt='o', color=c, ms=5, capsize=2, label=ml if i == 0 else None)
    a.axvline(0, color='k', lw=0.6); a.axvline(1, color='k', lw=0.6, ls=':'); a.set_xlim(-2, 3)
    a.set_yticks(range(len(arms))); a.set_yticklabels([x[1] for x in arms], fontsize=9); a.invert_yaxis()
    a.set_xlabel('R = change of (logit W − logit Z) under the edit\n÷ change under an angular dilation with the same jet-mass change')
    a.set_title('(c) R per edit (W/Z jets, 95 % bootstrap intervals)\n1 = responds as to a real mass change, 0 = no response', fontsize=10)
    a.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False)
    fig.suptitle('P56: does L-GATr follow the jet invariant mass or the spread of particles about its own momentum axis?', fontsize=11)
    fig.tight_layout(); f = ROOT / f'p56_lgatr_mass_vs_spread{TAG}.png'; fig.savefig(f, dpi=110); log('wrote', f)


if __name__ == '__main__':
    {'lgatr': stage_lgatr, 'part': stage_part, 'analyze': stage_analyze, 'sample': get_sample}[ARGS.stage]()
