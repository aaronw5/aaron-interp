"""P39 -- axis dependence on unedited jets (claims C18, C3; D4). Does ParT read the jet mass about the stored jet axis in jets
as they occur, or only when an edit moves the momentum direction off the stored axis?

Background (P36, P36v). c = sum_i pT_i (deta_i, dphi_i) / sum_i pT_i is the pT centroid of the constituents about the stored axis.
About the stored axis M2 = sum_i pT_i r_i^2 = M2c + |c|^2 M0 (M0 = sum pT, M2c = the same moment about the centroid), so the
axis-moment mass m_A = sqrt(M0 M2) of an off-centre jet exceeds the centroid-moment mass m_c = sqrt(M0 M2c) (which equals the
small-angle invariant mass for massless particles). In unedited JetClass jets the stored axis is the total-momentum direction to
0.0007 rad (P36v); recentring (shift by -c) moves the momentum direction ~0.026 rad off the stored axis, so P36's R ~1 could come
from that inconsistency rather than from M2.

Sample: JetClass test_20M WToQQ_109.root and ZToQQ_109.root (no _127 file exists: test_20M holds _100-_119, val_5M _120-_124; _109
was used by no loop-C node), numpy default_rng(3939), drawn class by class in part_core.CL order: for W and Z a permutation of all
rows, whose first PER_S (10,000) rows are the pre-registered sample 'S' and the rest the extension; for the 8 other classes PER_O
(1,000) random rows of <class>_109.root (for the 10-class mirror test (c) only). Jets with < 2 particles dropped. Inputs rebuilt
with part_core.build (eta = jet_eta + sign(jet_eta) deta).
Models: released ParT_full, ParT_kin, ParticleNet_full (PN_full), ParticleNet_kin (PN_kin); L-GATr (our run
anrunw-ai-c-p5-lgatr2, latest checkpoint at job start; it recomputes its axis from the particles, stage lgatr).
Read-out d = l_Z - l_W (logit difference). Calibration: per-jet local slope s_j = (d(k=1.03) - d(k=1/1.03)) / (2 log 1.03), where
k scales every particle's (deta, dphi) about the stored axis (a real change of the jet mass; dlog m_A = log k exactly).

(a) natural matched pairs (no edit). Per class (W, Z): top quintile of |c| vs bottom quintile, paired with replacement
    (default_rng(3940)). Primary 'nn': exact cells of 4 quartile bins of log pT_jet and N particles, then the nearest bottom jet
    in standardised (log m_inv, log M2c) with caliper |dlog m_inv| <= 0.01. Secondary 'cem4' (the plan's literal design): 4
    quartile bins each of log m_inv, log pT_jet, N, log M2c, random bottom jet of the cell; in the smoke run it left a mean
    dlog m_inv of +0.04, 7x the treatment. M2c is the moment about the pT centroid (matching on M2 about the stored axis would
    equalise m_A itself). Per pair (top - bottom): y = d_top - d_bot,
    sbar = (s_top + s_bot)/2, P_I = sbar dlog m_inv (H0, invariant-mass read-out), P_A = sbar dlog m_A (H1, axis moment),
    P_C = sbar dlog m_c (centroid moment). R_nat = sum(y - P_I) / sum(P_A - P_I): 0 = only the invariant mass matters, 1 = the
    network follows m_A. Also y/P_A (plain ratio), a version with the global slope kappa = mean s_j, and an OLS version with the
    residual differences of log pT, N, |eta_jet| as covariates. 95 % bootstrap over pairs (NB); sign-flip permutation (NPERM)
    for H0 R_nat = 0 (flip y - P_I) and H1 R_nat = 1 (flip y - P_A).
(b) edits on the S W/Z jets (pT and particle masses kept):
    rec      shift every particle by -c (P36; stored axis kept -> momentum direction ~|c| off the stored axis)
    rec_rb   rec, then stored axis rebuilt: jet_eta/jet_phi := direction of the edited constituents' momentum sum, deta/dphi
             recomputed about it (momentum axis on the stored axis by construction)
    base_rb  the axis rebuild applied to the unedited jet (control for the rebuild itself)
    iso      shift by |c| at +-120 deg from c (random sign): M2 about the stored axis unchanged (2 s.c + |s|^2 = 0), momentum
             direction moved |c| off the stored axis as in rec. H1 (M2) predicts d(iso) = 0; 'momentum-axis displacement'
             predicts d(iso) = d(rec); a response linear in the displacement along c predicts d(iso) = d(rec)/2
    rs       shift by |c| in a uniformly random direction (P36)
    ref      shift by +c (offset doubled)
    dil_rec  angles x sqrt(M2_rec / M2) (matched real mass change)
(c) phi mirror split (all S jets, 10 classes): mirp = dphi -> -dphi; with charge, d0 and dz signs flipped one at a time
    (mirp_q, mirp_d0, mirp_dz), charge and d0 together (mirp_qd0) and all three (mirp_all); flips without the mirror (q, d0, dz).
    Calls changed (10-class argmax) vs unedited, class-balanced; exact McNemar of per-jet 'call changed' under mirp vs under each
    mirror variant; Holm over models x variants.
(d) the pre-registered P31 analysis is run separately with interp.p31_classes_steps_part (command in the P39 report).

Checks: unedited accuracy on the P1 sample (rows 3,000-4,000 of each _100 file; ParT_full 85.9 %, ParT_kin 74.5 %, PN 84.6 / 70.4 %)
and on this sample; momentum-axis offset (angle between the constituents' summed three-momentum and the stored axis) and
centroid offset after each edit; invariant-mass change per edit.

  ./sync.sh && ./part_job.sh c-p39-smoke 'python -m interp.p39_natural_axis --stage nets --smoke && python -m interp.p39_natural_axis --stage analyze --smoke'
  ./sync.sh && ./part_job.sh c-p39-run 'python -m interp.p39_natural_axis --stage nets'
  ./sync.sh && ./part_job.sh c-p39-lg '/j-jepa-vol/aaron-interp/env-lgatr/bin/python -m interp.p39_natural_axis --stage lgatr'
  ./sync.sh && ./job.sh c-p39-an 2 8 'python -m interp.p39_natural_axis --stage analyze'
  -> $AI_RESULTS/part/p39_nets.npz, $AI_RESULTS/lgatr/p39_lgatr.npz, $AI_RESULTS/part/p39_natural_axis.json, p39_natural_axis.png,
     p39_mirror.png
"""
import argparse, glob, json, math, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('JETDISTILL_TAGGER', 'part')
ap = argparse.ArgumentParser(); ap.add_argument('--stage', required=True, choices=['nets', 'lgatr', 'analyze'])
ap.add_argument('--smoke', action='store_true'); ARGS = ap.parse_args()
SMOKE = ARGS.smoke
SEED, FIDX = 3939, 109
PER_S = 300 if SMOKE else 10000
PER_O = 30 if SMOKE else 1000
NMAX = 12000 if SMOKE else None          # smoke: only the first NMAX rows of each file
CHUNK = 6000 if SMOKE else 20000
NB = 200 if SMOKE else 2000
NPERM = 500 if SMOKE else 10000
KDIL = 1.03
TEST = Path('/j-jepa-vol/JetClass/Pythia/test_20M')
PRE = {'QCD': 'ZJetsToNuNu', 'Hbb': 'HToBB', 'Hcc': 'HToCC', 'Hgg': 'HToGG', 'H4q': 'HToWW4Q', 'Hqql': 'HToWW2Q1L', 'Zqq': 'ZToQQ',
       'Wqq': 'WToQQ', 'Tbqq': 'TTBar', 'Tbl': 'TTBarLep'}
CL = ['QCD', 'Hbb', 'Hcc', 'Hgg', 'H4q', 'Hqql', 'Zqq', 'Wqq', 'Tbqq', 'Tbl']
iW, iZ = CL.index('Wqq'), CL.index('Zqq')
A_EDITS = ['base', 'dilp', 'dilm']
B_EDITS = ['rec', 'rec_rb', 'base_rb', 'iso', 'rs', 'ref', 'dil_rec']
C_EDITS = ['mirp', 'mirp_q', 'mirp_d0', 'mirp_dz', 'mirp_qd0', 'mirp_all', 'q', 'd0', 'dz']
S_EDITS = A_EDITS + B_EDITS + C_EDITS
LG_EDITS = A_EDITS + ['rec', 'iso', 'rs', 'ref', 'dil_rec']
FLIPS = {'mirp': (True, ()), 'mirp_q': (True, ('charge',)), 'mirp_d0': (True, ('d0val',)), 'mirp_dz': (True, ('dzval',)),
         'mirp_qd0': (True, ('charge', 'd0val')), 'mirp_all': (True, ('charge', 'd0val', 'dzval')),
         'q': (False, ('charge',)), 'd0': (False, ('d0val',)), 'dz': (False, ('dzval',))}
RES = Path(os.environ.get('AI_RESULTS', 'results'))
TAG = '_smoke' if SMOKE else ''
NETS_NPZ = RES / 'part' / f'p39_nets{TAG}.npz'
LG_NPZ = RES / 'lgatr' / f'p39_lgatr{TAG}.npz'
OUT_JSON = RES / 'part' / f'p39_natural_axis{TAG}.json'
OUT_PNG = RES / 'part' / f'p39_natural_axis{TAG}.png'
OUT_PNG2 = RES / 'part' / f'p39_mirror{TAG}.png'
LG_RUN = '/j-jepa-vol/aaron-interp/models/lgatr/runs/jc2m/jc2m_s0a/models'
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


# ------------------------------------------------------------------------------------------------ sample
def draws():
    """deterministic row draws: {class: (N_entries, rows_S (sorted), is_full_file)}; W/Z: a permutation of all rows, the first
    PER_S are S; other classes: PER_O random rows"""
    import uproot
    rng = np.random.default_rng(SEED); out = {}
    for c in CL:
        path = TEST / f'{PRE[c]}_{FIDX}.root'; N = uproot.open(path)['tree'].num_entries
        if NMAX: N = min(N, NMAX)
        if c in ('Wqq', 'Zqq'):
            perm = rng.permutation(N); out[c] = (N, np.sort(perm[:PER_S]), True)
        else:
            out[c] = (N, np.sort(rng.choice(N, PER_O, replace=False)), False)
    return out


def read_rows(path, rows, N):
    from interp.part_core import read_root, sub, cat
    ch = []
    for s in range(0, N, 25000):
        r = rows[(rows >= s) & (rows < s + 25000)]
        if len(r): ch.append(sub(read_root(path, s, min(s + 25000, N)), r - s))
    return cat(ch)


# ------------------------------------------------------------------------------------------------ geometry and edits
def geom(J):
    """per-jet quantities of a (raw or built) jet about its stored axis"""
    from interp.part_core import kin, jmass
    pt, de, dp, ms = kin(J); p = pt.astype(np.float64); de64, dp64 = de.astype(np.float64), dp.astype(np.float64)
    M0 = p.sum(1); M1e = (p * de64).sum(1); M1p = (p * dp64).sum(1); M2 = (p * (de64 ** 2 + dp64 ** 2)).sum(1)
    ce, cp = M1e / np.maximum(M0, 1e-9), M1p / np.maximum(M0, 1e-9)
    oe, op = momentum_offset(J)
    # shape covariates about the centroid: pT_D = sum pT^2 / M0^2, leading-particle pT fraction, eccentricity of the pT-weighted
    # (deta, dphi) covariance (1 - lambda_min / lambda_max), skewness |sum pT (r - c)^3| / (M0 width^3) along the major axis
    W = np.maximum(M0, 1e-9); xe, xp = de64 - ce[:, None], dp64 - cp[:, None]; m_ = p > 0
    Cee, Cpp, Cep = (p * xe ** 2 * m_).sum(1) / W, (p * xp ** 2 * m_).sum(1) / W, (p * xe * xp * m_).sum(1) / W
    tr, det = Cee + Cpp, Cee * Cpp - Cep ** 2; disc = np.sqrt(np.maximum(tr ** 2 / 4 - det, 0)); l1, l2 = tr / 2 + disc, np.maximum(tr / 2 - disc, 0)
    return dict(ce=ce, cp=cp, cabs=np.hypot(ce, cp), M0=M0, M2=M2, M2c=M2 - (ce ** 2 + cp ** 2) * M0, minv=jmass(J).astype(np.float64),
                ptj=J['jet_pt'].astype(np.float64), npart=(J['mask'] > 0).sum(1).astype(np.int32), etaj=J['jet_eta'].astype(np.float64),
                momoff=np.hypot(oe, op), ptd=(p ** 2).sum(1) / W ** 2, lead=p.max(1) / W, ecc=1 - l2 / np.maximum(l1, 1e-12))


def momentum_offset(J):
    """(deta, dphi) of the constituents' summed three-momentum direction from the stored axis, in stored coordinates"""
    m = J['mask'] > 0; px, py, pz = ((J[k].astype(np.float64) * m).sum(1) for k in ('px', 'py', 'pz'))
    eta = np.arcsinh(pz / np.maximum(np.hypot(px, py), 1e-9)); phi = np.arctan2(py, px)
    sg = np.where(J['jet_eta'] < 0, -1.0, 1.0)
    return sg * (eta - J['jet_eta'].astype(np.float64)), np.angle(np.exp(1j * (phi - J['jet_phi'].astype(np.float64))))


def shifts(G, n, seed):
    r = np.random.default_rng(seed); al = r.uniform(0, 2 * np.pi, n); sg = r.choice([-1.0, 1.0], n); th = sg * 2 * np.pi / 3
    S = dict(rse=G['cabs'] * np.cos(al), rsp=G['cabs'] * np.sin(al),
             ise=np.cos(th) * G['ce'] - np.sin(th) * G['cp'], isp=np.sin(th) * G['ce'] + np.cos(th) * G['cp'])
    M1e, M1p = G['ce'] * G['M0'], G['cp'] * G['M0']
    S['M2_rec'] = G['M2'] - 2 * (G['ce'] * M1e + G['cp'] * M1p) + G['cabs'] ** 2 * G['M0']
    return S


def rebuild_axis(J, pt, de, dp, ms):
    """particles at absolute (eta, phi) given by J's stored axis and (de, dp); new stored axis = direction of their momentum sum;
    (deta, dphi) recomputed about it; the jet dict rebuilt"""
    from interp.part_core import build
    m = pt > 0; sg = np.where(J['jet_eta'] < 0, -1.0, 1.0)[:, None]
    eta = J['jet_eta'][:, None].astype(np.float64) + sg * de; phi = J['jet_phi'][:, None].astype(np.float64) + dp
    p = pt.astype(np.float64)
    px, py, pz = (np.where(m, p * np.cos(phi), 0).sum(1), np.where(m, p * np.sin(phi), 0).sum(1), np.where(m, p * np.sinh(eta), 0).sum(1))
    eN = np.arcsinh(pz / np.hypot(px, py)); fN = np.arctan2(py, px); sN = np.where(eN < 0, -1.0, 1.0)[:, None]
    deN = np.where(m, sN * (eta - eN[:, None]), 0); dpN = np.where(m, np.angle(np.exp(1j * (phi - fN[:, None]))), 0)
    J2 = dict(J); J2['jet_eta'] = eN.astype(np.float32); J2['jet_phi'] = fN.astype(np.float32)
    return build(J2, pt, deN.astype(np.float32), dpN.astype(np.float32), ms)


def make_edit(J, K, G, S, name):
    from interp.part_core import build, PK
    pt, de, dp, ms = K; r = (pt > 0).astype(np.float32); col = lambda a: a[:, None].astype(np.float32)
    if name == 'base': return build(J, pt, de, dp, ms)
    if name == 'dilp': return build(J, pt, de * KDIL, dp * KDIL, ms)
    if name == 'dilm': return build(J, pt, de / KDIL, dp / KDIL, ms)
    if name == 'rec': return build(J, pt, de - col(G['ce']) * r, dp - col(G['cp']) * r, ms)
    if name == 'rec_rb': return rebuild_axis(J, pt, de - col(G['ce']) * r, dp - col(G['cp']) * r, ms)
    if name == 'base_rb': return rebuild_axis(J, pt, de, dp, ms)
    if name == 'iso': return build(J, pt, de + col(S['ise']) * r, dp + col(S['isp']) * r, ms)
    if name == 'rs': return build(J, pt, de + col(S['rse']) * r, dp + col(S['rsp']) * r, ms)
    if name == 'ref': return build(J, pt, de + col(G['ce']) * r, dp + col(G['cp']) * r, ms)
    if name == 'dil_rec':
        k = col(np.sqrt(np.maximum(S['M2_rec'], 1e-30) / G['M2'])); return build(J, pt, de * k, dp * k, ms)
    if name in FLIPS:
        mir, keys = FLIPS[name]; ex = {k: J[k] for k in PK}
        for k in keys: ex[k] = -J[k]
        return build(J, pt, de, -dp if mir else dp, ms, extra=ex)
    raise ValueError(name)


# ------------------------------------------------------------------------------------------------ stage: ParT / ParticleNet
def load_nets(dev):
    from interp.part_core import ParTNetwork
    from interp import pn_net
    from jetdistill.part.network import FEATURES
    M = {}
    for feat in ('full', 'kin'):
        net = ParTNetwork(feat); net.model.to(dev).eval(); M[f'ParT_{feat}'] = dict(kind='part', feat=feat, model=net.model)
    PND = Path(os.environ.get('JETDISTILL_MODELS', '/j-jepa-vol/aaron-interp/models/part')).parent / 'particlenet'
    for feat in ('full', 'kin'):
        M[f'PN_{feat}'] = dict(kind='pn', feat=feat, model=pn_net.load_particlenet(PND / f'ParticleNet_{feat}.pt', len(FEATURES[feat]), dev))
    return M, FEATURES


def runner(M, FEATURES, dev):
    import torch
    from interp.part_core import inputs
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)

    def run(J, names, bs=512):
        N = len(J['mask']); order = np.argsort((J['mask'] > 0).sum(1), kind='stable'); L = {k: np.zeros((N, 10), np.float32) for k in names}
        feats = sorted({M[k]['feat'] for k in names})
        for s in range(0, N, bs):
            idx = order[s:s + bs]; Jc = {k: (a[idx] if np.ndim(a) else a) for k, a in J.items()}
            X = {f: inputs(Jc, f) for f in feats}
            for k in names:
                mm = M[k]; x, v, msk = X[mm['feat']]
                with torch.no_grad():
                    if mm['kind'] == 'part': out = mm['model'](T(x), T(v), T(msk))
                    else:
                        P = x.shape[-1]; pad = (lambda a: np.concatenate([a, np.zeros(a.shape[:2] + (128 - P,), np.float32)], -1)) if P < 128 else (lambda a: a)
                        xp, mp = pad(x), pad(msk); fn = [f[0] for f in FEATURES[mm['feat']]]
                        out = mm['model'](T(xp[:, [fn.index('deta'), fn.index('dphi')]]), T(xp), T(mp))
                L[k][idx] = out.float().cpu().numpy()
        return L
    return run


def edit_models(name, names):
    """models whose inputs change under the edit (the kin models see neither charge nor d0 / dz)"""
    if name in FLIPS and name != 'mirp': return [k for k in names if k.endswith('full')]
    return names


def fill_kin(L, e, k, n_rows):
    """kin-model logits for sign flips: the same inputs as mirp (mirror variants) or as base (no-mirror flips)"""
    src = 'mirp' if FLIPS[e][0] else 'base'
    return src


def stage_nets():
    import torch
    from interp.part_core import sample, kin, build, sub
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; log('device', dev, torch.__version__)
    M, FEATURES = load_nets(dev); names = list(M); run = runner(M, FEATURES, dev); log('models', names)
    J1 = sample(100 if SMOKE else 1000, 3000); y1 = J1['label'].astype(int); L1 = run(build(J1, *kin(J1)), names)
    acc_p1 = {k: float((L1[k].argmax(1) == y1).mean()) for k in names}; log('P1-sample accuracy', {k: round(v, 4) for k, v in acc_p1.items()}, 'N', len(y1))
    del J1, L1
    D = draws(); A = {}; Sx = {}; geoE = {}
    sidx = {e: i for i, e in enumerate(S_EDITS)}

    def do_S(J, cls):
        """S edits on one class's S jets; returns logits per model (len(S_EDITS), n, 10) and edit geometry"""
        from interp.part_core import kin as _kin
        K = _kin(J); G = geom(J); n = len(G['ce']); Sh = shifts(G, n, SEED + 100 + CL.index(cls))
        Lk = {k: np.full((len(S_EDITS), n, 10), np.nan, np.float32) for k in names}; gE = {}
        edits = S_EDITS if cls in ('Wqq', 'Zqq') else ['base'] + C_EDITS
        Jb = None
        for e in edits:
            Je = make_edit(J, K, G, Sh, e); ge = geom(Je)
            if e == 'base': Jb = ge; G['minv'] = ge['minv']
            gE[e] = dict(dlogmA=0.5 * np.log(np.maximum(ge['M2'], 1e-30) / np.maximum(Jb['M2'], 1e-30)) + 0.5 * np.log(ge['M0'] / Jb['M0']),
                         dlogminv=np.log(np.maximum(ge['minv'], 1e-3) / np.maximum(Jb['minv'], 1e-3)), momoff=ge['momoff'], cabs=ge['cabs'])
            em = edit_models(e, names); L = run(Je, em)
            for k in em: Lk[k][sidx[e]] = L[k]
            for k in names:
                if k not in em: Lk[k][sidx[e]] = Lk[k][sidx[fill_kin(L, e, k, n)]]
            del Je
        return Lk, G, gE, Sh, K

    for c in CL:
        N, rowsS, full = D[c]; path = TEST / f'{PRE[c]}_{FIDX}.root'; t1 = time.time()
        if full:
            from interp.part_core import read_root
            parts = []
            for s in range(0, N, CHUNK):
                e_ = min(s + CHUNK, N); J = read_root(path, s, e_); ok = (J['mask'] > 0).sum(1) >= 2; rows = np.arange(s, e_)[ok]; J = sub(J, np.where(ok)[0])
                assert np.all(J['label'] == CL.index(c)), c
                K = kin(J); G = geom(J); n = len(rows); La = {k: np.zeros((len(A_EDITS), n, 10), np.float32) for k in names}
                for ie, e in enumerate(A_EDITS):
                    Je = make_edit(J, K, G, None, e); L = run(Je, names)
                    for k in names: La[k][ie] = L[k]
                    if e == 'base': G['minv'] = geom(Je)['minv']      # invariant mass of the rebuilt unedited jet (as the network reads it)
                    del Je
                parts.append(dict(rows=rows, G=G, L=La)); log(f'{c}: rows {s}-{e_} ({n} jets, >= 2 particles) base/dil done ({time.time() - t1:.0f} s)')
            A[c] = dict(rows=np.concatenate([p['rows'] for p in parts]), G={k: np.concatenate([p['G'][k] for p in parts]) for k in parts[0]['G']},
                        L={k: np.concatenate([p['L'][k] for p in parts], 1) for k in names})
        J = read_rows(path, rowsS, N); ok = (J['mask'] > 0).sum(1) >= 2; J = sub(J, np.where(ok)[0]); rs_ = rowsS[ok]
        Lk, G, gE, Sh, K = do_S(J, c)
        pt, de, dp, _ = K
        Sx[c] = dict(rows=rs_, Lk=Lk, G=G, gE=gE, Sh=Sh, ex=(pt.astype(np.float16), de.astype(np.float16), dp.astype(np.float16)) if c in ('Wqq', 'Zqq') else None)
        acc = {k: float((Lk[k][0].argmax(1) == CL.index(c)).mean()) for k in names}
        log(f'{c}: S {len(rs_)} jets, {len(S_EDITS) if c in ("Wqq", "Zqq") else 1 + len(C_EDITS)} edits ({time.time() - t1:.0f} s); recall {acc}; '
            + ('rec momoff median %.4f, rec_rb momoff %.5f, rec_rb |c| %.4f vs base %.4f, iso dlogmA mean %.2e' % (
                np.median(gE['rec']['momoff']), np.median(gE['rec_rb']['momoff']), np.median(gE['rec_rb']['cabs']), np.median(gE['base']['cabs']),
                gE['iso']['dlogmA'].mean()) if 'rec' in gE else ''))
    # save
    out = {}
    for c in ('Zqq', 'Wqq'):
        a = A[c]; out[f'A_{c}_rows'] = a['rows']
        for k, v in a['G'].items(): out[f'A_{c}_G_{k}'] = v
        for k, v in a['L'].items(): out[f'A_{c}_L_{k}'] = v
    yS = np.concatenate([np.full(len(Sx[c]['rows']), CL.index(c)) for c in CL]); rowsS = np.concatenate([Sx[c]['rows'] for c in CL])
    out['S_y'] = yS; out['S_rows'] = rowsS
    for k in names: out[f'S_L_{k}'] = np.concatenate([Sx[c]['Lk'][k] for c in CL], 1)
    for g in Sx['Wqq']['G']: out[f'S_G_{g}'] = np.concatenate([Sx[c]['G'][g] for c in CL])
    for g in ('rse', 'rsp', 'ise', 'isp', 'M2_rec'): out[f'S_Sh_{g}'] = np.concatenate([Sx[c]['Sh'][g] for c in CL])
    nS = len(yS)
    for e in S_EDITS:
        for q in ('dlogmA', 'dlogminv', 'momoff', 'cabs'):
            out[f'S_E_{e}_{q}'] = np.concatenate([Sx[c]['gE'][e][q] if e in Sx[c]['gE'] else np.full(len(Sx[c]['rows']), np.nan) for c in CL])
    for c in ('Zqq', 'Wqq'):
        pt, de, dp = Sx[c]['ex']; out[f'X_{c}_pt'], out[f'X_{c}_de'], out[f'X_{c}_dp'] = pt, de, dp
    NETS_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(NETS_NPZ, names=np.array(names), s_edits=np.array(S_EDITS), a_edits=np.array(A_EDITS), **out)
    meta = dict(acc_p1=acc_p1, n_p1=int(len(y1)), device=dev, N_S=int(nS), draws={c: dict(N=int(D[c][0]), n_S=int(len(D[c][1]))) for c in CL},
                files=f'test_20M/<class>_{FIDX}.root', seed=SEED)
    (NETS_NPZ.parent / f'p39_nets_meta{TAG}.json').write_text(json.dumps(meta, indent=1)); log('wrote', NETS_NPZ)


# ------------------------------------------------------------------------------------------------ stage: L-GATr
def stage_lgatr():
    import torch
    from interp.p5_lgatr_launch import load_lgatr, logits
    from interp.part_core import sample, kin, build, sub
    c_ = sorted(glob.glob(f'{LG_RUN}/*.pt'), key=lambda p: int(p.split('_it')[-1].split('.')[0])); ck = c_[-1]
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; log('checkpoint', ck, 'device', dev)
    model, cfg = load_lgatr(ck, dev); log('loaded L-GATr; force_xformers', model.force_xformers)
    J1 = sample(50 if SMOKE else 300, 3000); y1 = J1['label'].astype(int); L1 = logits(model, cfg, build(J1, *kin(J1)), dev)
    acc_p1 = float((L1.argmax(1) == y1).mean()); log(f'P1-sample accuracy (rows 3000..{3000 + (50 if SMOKE else 300)} of _100) {acc_p1:.4f} N {len(y1)}')
    D = draws(); out = {}
    for c in ('Zqq', 'Wqq'):
        N, rowsS, _ = D[c]; J = read_rows(TEST / f'{PRE[c]}_{FIDX}.root', rowsS, N); ok = (J['mask'] > 0).sum(1) >= 2; J = sub(J, np.where(ok)[0])
        K = kin(J); G = geom(J); Sh = shifts(G, len(G['ce']), SEED + 100 + CL.index(c)); Lc = np.zeros((len(LG_EDITS), len(G['ce']), 10), np.float32)
        for ie, e in enumerate(LG_EDITS):
            Lc[ie] = logits(model, cfg, make_edit(J, K, G, Sh, e), dev, bs=128)
            log(f'L-GATr {c} {e}: recall {(Lc[ie].argmax(1) == CL.index(c)).mean():.4f}')
        out[f'{c}_rows'] = rowsS[ok]; out[f'{c}_L'] = Lc
    LG_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(LG_NPZ, edits=np.array(LG_EDITS), ckpt=np.array(ck), acc_p1=acc_p1, n_p1=len(y1), **out); log('wrote', LG_NPZ)


# ------------------------------------------------------------------------------------------------ stage: analysis
def ci(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) else [None, None]


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.zeros(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj.tolist()


def signflip_p(x, rng, nperm=None):
    """two-sided sign-flip permutation p for H0 mean(x) = 0"""
    nperm = nperm or NPERM; x = np.asarray(x, float); t = abs(x.mean()); n = len(x); cnt = 0; done = 0
    while done < nperm:
        b = min(500, nperm - done); sg = rng.choice([-1.0, 1.0], size=(b, n)); cnt += int((np.abs(sg @ x / n) >= t).sum()); done += b
    return float((1 + cnt) / (1 + nperm))


def boot_idx(n, rng, nb=None):
    for _ in range(nb or NB): yield rng.integers(0, n, n)


def mcnemar(a, b):
    from scipy.stats import binomtest
    x = int((a & ~b).sum()); z = int((~a & b).sum())
    return dict(only_first=x, only_second=z, p=(float(binomtest(x, x + z, 0.5).pvalue) if x + z else 1.0))


def match_pairs(G, y, rng, method='nn', top_q=0.8, bot_q=0.2, caliper=0.01):
    """top-|c| quintile jets paired with bottom-quintile jets of the same class, with replacement.
    method 'cem4' (the plan's literal design): coarsened exact matching on 4 quartile bins each of log m_inv, log pT, N, log M2c
      (computed on top + bottom jets of the class), random bottom jet of the cell.
    method 'nn' (primary): exact cells of 4 quartile bins of log pT and N, then the nearest bottom jet of the cell in
      standardised (log m_inv, log M2c), kept if |dlog m_inv| <= caliper (the 4-bin mass coarsening left a mass gap 7x the
      treatment in the smoke run). returns (i_top, i_bot, info)"""
    from scipy.spatial import cKDTree
    I, Jb, info = [], [], {}
    for k in (iW, iZ):
        ik = np.where(y == k)[0]; c = G['cabs'][ik]; hi, lo = np.quantile(c, top_q), np.quantile(c, bot_q)
        top, bot = ik[c >= hi], ik[c <= lo]; both = np.concatenate([top, bot])
        lm, l2 = np.log(np.maximum(G['minv'][both], 1e-3)), np.log(np.maximum(G['M2c'][both], 1e-12))
        ae = np.abs(G['etaj'][both])
        V = np.stack([np.log(G['ptj'][both]), G['npart'][both].astype(float)] + ([lm, l2] if method == 'cem4' else []) + ([ae] if method == 'nneta' else []), 1)
        cells = np.zeros(len(both), np.int64)
        for j in range(V.shape[1]):
            e = np.quantile(V[:, j], [0.25, 0.5, 0.75]); cells = cells * 4 + np.searchsorted(e, V[:, j], side='right')
        ct, cb = cells[:len(top)], cells[len(top):]
        F = np.stack([lm / lm.std(), l2 / l2.std()] + ([ae / ae.std()] if method == 'nneta' else []), 1); Ft, Fb = F[:len(top)], F[len(top):]
        mt = 0; ncell = 0
        for cc in np.unique(ct):
            tt = np.where(ct == cc)[0]; bb = np.where(cb == cc)[0]
            if not len(bb): continue
            ncell += 1
            if method == 'cem4':
                pick = bb[rng.integers(0, len(bb), len(tt))]; keep = np.ones(len(tt), bool)
            else:
                _, q = cKDTree(Fb[bb]).query(Ft[tt], k=1); pick = bb[q]
                keep = np.abs(lm[tt] - lm[len(top) + pick]) <= caliper
                if method == 'nneta': keep &= np.abs(ae[tt] - ae[len(top) + pick]) <= 0.05
            I += list(top[tt[keep]]); Jb += list(bot[pick[keep]]); mt += int(keep.sum())
        info[CL[k]] = dict(n_top=int(len(top)), n_bot=int(len(bot)), n_matched=mt, frac_matched=float(mt / max(len(top), 1)),
                           cabs_top_median=float(np.median(G['cabs'][top])), cabs_bot_median=float(np.median(G['cabs'][bot])),
                           q80=float(hi), q20=float(lo), n_cells_used=ncell)
    I, Jb = np.array(I, int), np.array(Jb, int); info['n_pairs'] = int(len(I)); info['n_distinct_bottom'] = int(len(np.unique(Jb))); info['method'] = method
    return I, Jb, info


def analyze_natural(d, s, G, y, rng, label, method='nn'):
    """(a) for one model on one jet set; d, s: per-jet read-out and local slope"""
    I, B, info = match_pairs(G, y, rng, method)
    lmi = np.log(np.maximum(G['minv'], 1e-3)); lmA = 0.5 * np.log(G['M0'] * G['M2']); lmc = 0.5 * np.log(G['M0'] * np.maximum(G['M2c'], 1e-30))
    yv = d[I] - d[B]; sb = 0.5 * (s[I] + s[B]); kap = float(np.mean(s[np.isin(y, [iW, iZ])]))
    dI, dA, dC = lmi[I] - lmi[B], lmA[I] - lmA[B], lmc[I] - lmc[B]
    PI, PA, PC = sb * dI, sb * dA, sb * dC
    res = dict(match=info, kappa_mean_slope=kap, n_pairs=int(len(I)))
    bal = {}
    for nm, v in (('log_minv', lmi), ('log_mA', lmA), ('log_mc', lmc), ('log_ptj', np.log(G['ptj'])), ('npart', G['npart'].astype(float)),
                  ('log_M2c', np.log(np.maximum(G['M2c'], 1e-30))), ('cabs', G['cabs']), ('abs_etaj', np.abs(G['etaj'])), ('momoff', G['momoff']),
                  ('ptd', G['ptd']), ('lead', G['lead']), ('ecc', G['ecc'])):
        dv = v[I] - v[B]; sdp = np.sqrt(0.5 * (v[I].var() + v[B].var())) + 1e-30
        bal[nm] = dict(mean_diff=float(dv.mean()), smd=float(dv.mean() / sdp))
    res['balance'] = bal
    res['means'] = dict(y_obs=float(yv.mean()), pred_H0_minv=float(PI.mean()), pred_H1_mA=float(PA.mean()), pred_centroid_mc=float(PC.mean()),
                        dlog_mA_minus_dlog_minv=float((dA - dI).mean()), dlog_offset_term=float(np.mean(0.5 * np.log1p(G['cabs'][I] ** 2 * G['M0'][I] / np.maximum(G['M2c'][I], 1e-30))
                                                                                                         - 0.5 * np.log1p(G['cabs'][B] ** 2 * G['M0'][B] / np.maximum(G['M2c'][B], 1e-30)))),
                        sbar_mean=float(sb.mean()))
    num, den = yv - PI, PA - PI
    R = float(num.sum() / den.sum()); Rp = float(yv.sum() / PA.sum())
    numk, denk = yv - kap * dI, kap * (dA - dI); Rk = float(numk.sum() / denk.sum())
    Rb, Rpb, Rkb, yb, PAb, PIb = [], [], [], [], [], []
    for b in boot_idx(len(I), rng):
        Rb.append(num[b].sum() / den[b].sum()); Rpb.append(yv[b].sum() / PA[b].sum()); Rkb.append(numk[b].sum() / denk[b].sum())
        yb.append(yv[b].mean()); PAb.append(PA[b].mean()); PIb.append(PI[b].mean())
    res['R_nat'] = dict(v=R, ci=ci(Rb)); res['R_plain_y_over_PA'] = dict(v=Rp, ci=ci(Rpb)); res['R_nat_global_kappa'] = dict(v=Rk, ci=ci(Rkb))
    res['ci_means'] = dict(y_obs=ci(yb), pred_H1_mA=ci(PAb), pred_H0_minv=ci(PIb))
    res['perm_H0_Rnat0'] = signflip_p(yv - PI, rng); res['perm_H1_Rnat1'] = signflip_p(yv - PA, rng)
    # OLS: (y - P_I) = R (P_A - P_I) + b . (dlog pT, dN, d|eta_j|, d pT_D, d lead, d ecc)  (no intercept: pairs are oriented top - bottom)
    X = np.stack([den, np.log(G['ptj'][I] / G['ptj'][B]), (G['npart'][I] - G['npart'][B]).astype(float), np.abs(G['etaj'][I]) - np.abs(G['etaj'][B]),
                  G['ptd'][I] - G['ptd'][B], G['lead'][I] - G['lead'][B], G['ecc'][I] - G['ecc'][B]], 1)
    beta = np.linalg.lstsq(X, num, rcond=None)[0]; Ob = []
    for b in boot_idx(len(I), rng, min(NB, 1000)): Ob.append(np.linalg.lstsq(X[b], num[b], rcond=None)[0][0])
    res['R_nat_ols_cov'] = dict(v=float(beta[0]), ci=ci(Ob))
    # expected noise: per-pair s.d. of y - P_I and the resulting half-width on R
    res['sd_pair_resid'] = float(num.std()); res['R_halfwidth_expected'] = float(1.96 * num.std() / np.sqrt(len(I)) / max(abs(den.mean()), 1e-12))
    label = f'{label} [{method}]'
    log(f'  (a) {label}: pairs {len(I)}; y {yv.mean():+.4f} P_I {PI.mean():+.4f} P_A {PA.mean():+.4f} P_C {PC.mean():+.4f}; '
        f'R_nat {R:+.2f} {np.round(ci(Rb), 2)} (global kappa {Rk:+.2f}; OLS {beta[0]:+.2f}); p(R=0) {res["perm_H0_Rnat0"]:.3g} p(R=1) {res["perm_H1_Rnat1"]:.3g}; '
        f'balance dlog m_inv {bal["log_minv"]["mean_diff"]:+.4f} dlog m_A {bal["log_mA"]["mean_diff"]:+.4f} d|c| {bal["cabs"]["mean_diff"]:+.4f}')
    res['_pairs'] = (I, B)
    return res


def stage_analyze():
    D = np.load(NETS_NPZ, allow_pickle=False); names = [str(x) for x in D['names']]; SE = {str(x): i for i, x in enumerate(D['s_edits'])}
    meta = json.loads((NETS_NPZ.parent / f'p39_nets_meta{TAG}.json').read_text()); rng = np.random.default_rng(3941)
    out = dict(node='P39', seed=SEED, files=meta['files'], nb=NB, nperm=NPERM, kdil=KDIL, nets_meta=meta, readout='d = logit_Z - logit_W')
    # ---- jet sets
    yS = D['S_y'].astype(int); GS = {k[4:]: D[k] for k in D.files if k.startswith('S_G_')}; wzS = np.isin(yS, [iW, iZ])
    GA = {}; yA = []; LA = {k: [] for k in names}; isS_A = []
    for c in ('Zqq', 'Wqq'):
        rows = D[f'A_{c}_rows']; yA.append(np.full(len(rows), CL.index(c)))
        Srows = D['S_rows'][yS == CL.index(c)]; isS_A.append(np.isin(rows, Srows))
        for k in names: LA[k].append(D[f'A_{c}_L_{k}'])
        for g in [f[len(f'A_{c}_G_'):] for f in D.files if f.startswith(f'A_{c}_G_')]: GA.setdefault(g, []).append(D[f'A_{c}_G_{g}'])
    yA = np.concatenate(yA); isS_A = np.concatenate(isS_A); GA = {k: np.concatenate(v) for k, v in GA.items()}; LA = {k: np.concatenate(v, 1) for k, v in LA.items()}
    out['sample'] = dict(N_all_WZ=int(len(yA)), N_S_WZ=int(wzS.sum()), N_S_all=int(len(yS)), N_S_per_class={CL[k]: int((yS == k).sum()) for k in range(10)},
                         cabs_median_WZ=float(np.median(GA['cabs'])), momoff_median_WZ=float(np.median(GA['momoff'])),
                         momoff_p99_WZ=float(np.percentile(GA['momoff'], 99)), cabs_p80=float(np.quantile(GA['cabs'], 0.8)))
    log('sample', out['sample'])
    # accuracy
    out['accuracy'] = dict(P1_sample=meta['acc_p1'], S_10class={k: float((D[f'S_L_{k}'][SE['base']].argmax(1) == yS).mean()) for k in names},
                           S_10class_balanced={k: float(np.mean([(D[f'S_L_{k}'][SE['base']][yS == j].argmax(1) == j).mean() for j in range(10)])) for k in names},
                           WZ_all_recall={k: float((LA[k][0].argmax(1) == yA).mean()) for k in names})
    log('accuracy', out['accuracy'])
    # L-GATr
    LG = None
    if LG_NPZ.exists():
        G_ = np.load(LG_NPZ, allow_pickle=False); LE = {str(x): i for i, x in enumerate(G_['edits'])}
        Lc = np.concatenate([G_['Zqq_L'], G_['Wqq_L']], 1); rows_lg = np.concatenate([G_['Zqq_rows'], G_['Wqq_rows']])
        rows_s = np.concatenate([D['S_rows'][yS == iZ], D['S_rows'][yS == iW]]); assert np.array_equal(rows_lg, rows_s), 'L-GATr rows differ from S'
        LG = dict(L=Lc, E=LE, ckpt=str(G_['ckpt']), acc_p1=float(G_['acc_p1']), n_p1=int(G_['n_p1']))
        out['accuracy']['LGATr'] = dict(ckpt=LG['ckpt'], P1_sample=LG['acc_p1'], n_p1=LG['n_p1'])
    # ---- (a) natural matched pairs
    out['a_natural'] = {}
    dfun = lambda L: (L[..., iZ] - L[..., iW]).astype(np.float64)
    wz_idx = np.where(wzS)[0]; GSwz = {k: v[wz_idx] for k, v in GS.items()}; ySwz = yS[wz_idx]
    pa_all = []
    for k in names:
        r = {}
        LS = D[f'S_L_{k}'][:, wz_idx]; dS = dfun(LS[SE['base']]); sS = (dfun(LS[SE['dilp']]) - dfun(LS[SE['dilm']])) / (2 * np.log(KDIL))
        r['S'] = analyze_natural(dS, sS, GSwz, ySwz, np.random.default_rng(3940), f'{k} S')
        r['S_cem4'] = analyze_natural(dS, sS, GSwz, ySwz, np.random.default_rng(3940), f'{k} S', 'cem4')
        r['S_eta'] = analyze_natural(dS, sS, GSwz, ySwz, np.random.default_rng(3940), f'{k} S', 'nneta')
        dA_ = dfun(LA[k][0]); sA_ = (dfun(LA[k][1]) - dfun(LA[k][2])) / (2 * np.log(KDIL))
        r['all'] = analyze_natural(dA_, sA_, GA, yA, np.random.default_rng(3940), f'{k} all rows (extension)')
        r['all_cem4'] = analyze_natural(dA_, sA_, GA, yA, np.random.default_rng(3940), f'{k} all rows (extension)', 'cem4')
        r['all_eta'] = analyze_natural(dA_, sA_, GA, yA, np.random.default_rng(3940), f'{k} all rows (extension)', 'nneta')
        out['a_natural'][k] = r; pa_all.append((k, r['S']['perm_H0_Rnat0']))
    if LG is not None:
        L_ = LG['L']; E_ = LG['E']; rows_order = np.concatenate([np.where(yS == iZ)[0], np.where(yS == iW)[0]])
        Gl = {k: v[rows_order] for k, v in GS.items()}; yl = yS[rows_order]
        dL = dfun(L_[E_['base']]); sL = (dfun(L_[E_['dilp']]) - dfun(L_[E_['dilm']])) / (2 * np.log(KDIL))
        out['a_natural']['LGATr'] = dict(S=analyze_natural(dL, sL, Gl, yl, np.random.default_rng(3940), 'LGATr S'), S_cem4=analyze_natural(dL, sL, Gl, yl, np.random.default_rng(3940), 'LGATr S', 'cem4'), S_eta=analyze_natural(dL, sL, Gl, yl, np.random.default_rng(3940), 'LGATr S', 'nneta')); pa_all.append(('LGATr', out['a_natural']['LGATr']['S']['perm_H0_Rnat0']))
    for (k, p), ph in zip(pa_all, holm([p for _, p in pa_all])): out['a_natural'][k]['S']['perm_H0_Rnat0_holm'] = ph
    # ---- (b) edits on the S W/Z jets
    out['b_edits'] = {}
    q80 = {k: np.quantile(GSwz['cabs'][ySwz == k], 0.8) for k in (iW, iZ)}; q95 = {k: np.quantile(GSwz['cabs'][ySwz == k], 0.95) for k in (iW, iZ)}
    sets = dict(all=np.ones(len(ySwz), bool), top20=np.array([GSwz['cabs'][i] >= q80[ySwz[i]] for i in range(len(ySwz))]),
                top5=np.array([GSwz['cabs'][i] >= q95[ySwz[i]] for i in range(len(ySwz))]))
    geoB = {}
    for e in ['base'] + B_EDITS:
        g = {q: D[f'S_E_{e}_{q}'][wz_idx] for q in ('dlogmA', 'dlogminv', 'momoff', 'cabs')}
        geoB[e] = {s: dict(momoff_median=float(np.median(g['momoff'][f])), momoff_p90=float(np.percentile(g['momoff'][f], 90)),
                           cabs_median=float(np.median(g['cabs'][f])), dlogmA_mean=float(g['dlogmA'][f].mean()),
                           dlogminv_mean=float(g['dlogminv'][f].mean()), dlogminv_absp99=float(np.percentile(np.abs(g['dlogminv'][f]), 99))) for s, f in sets.items()}
    out['b_geometry'] = geoB
    for s_, f in sets.items(): log(f'  (b) geometry {s_}: ' + '; '.join(f'{e} momoff {geoB[e][s_]["momoff_median"]:.4f} |c| {geoB[e][s_]["cabs_median"]:.4f} dlogmA {geoB[e][s_]["dlogmA_mean"]:+.5f} dlogm {geoB[e][s_]["dlogminv_mean"]:+.5f}' for e in geoB))
    def b_model(Lfull, label):
        d0 = dfun(Lfull[SE['base']] if 'base' in SE else Lfull[0]); sl = (dfun(Lfull[SE['dilp']]) - dfun(Lfull[SE['dilm']])) / (2 * np.log(KDIL))
        r = {}
        for s_, f in sets.items():
            ix = np.where(f)[0]; n = len(ix); rr = dict(n=n)
            dd = {e: dfun(Lfull[SE[e]])[ix] - d0[ix] for e in B_EDITS}
            pa = {e: sl[ix] * D[f'S_E_{e}_dlogmA'][wz_idx][ix] for e in B_EDITS}
            pi = {e: sl[ix] * D[f'S_E_{e}_dlogminv'][wz_idx][ix] for e in B_EDITS}
            Bs = np.stack([rng.integers(0, n, n).astype(np.int32) for _ in range(min(NB, 1000))])
            for e in B_EDITS:
                c0 = Lfull[SE['base']][ix].argmax(1); ce = Lfull[SE[e]][ix].argmax(1); ch = ce != c0
                rr[e] = dict(d=float(dd[e].mean()), d_ci=ci([dd[e][b].mean() for b in Bs]), pred_H1_mA=float(pa[e].mean()), pred_H0_minv=float(pi[e].mean()),
                             absd=float(np.abs(dd[e]).mean()), changed=float(ch.mean()), changed_ci=ci([ch[b].mean() for b in Bs]))
                if abs(pa[e].mean()) > 1e-3:
                    rr[e]['R'] = dict(v=float(dd[e].mean() / pa[e].mean()), ci=ci([dd[e][b].mean() / pa[e][b].mean() for b in Bs]))
                rr[e]['perm_d0'] = signflip_p(dd[e], rng, min(NPERM, 5000))
            rat = [dd['iso'][b].mean() / dd['rec'][b].mean() for b in Bs]
            rr['iso_over_rec'] = dict(v=float(dd['iso'].mean() / dd['rec'].mean()), ci=ci(rat))
            rr['perm_iso_eq_rec'] = signflip_p(dd['iso'] - dd['rec'], rng, min(NPERM, 5000))
            rr['perm_iso_eq_half_rec'] = signflip_p(dd['iso'] - 0.5 * dd['rec'], rng, min(NPERM, 5000))
            rr['recrb_minus_baserb'] = dict(v=float((dd['rec_rb'] - dd['base_rb']).mean()), ci=ci([(dd['rec_rb'] - dd['base_rb'])[b].mean() for b in Bs]),
                                           p=signflip_p(dd['rec_rb'] - dd['base_rb'], rng, min(NPERM, 5000)))
            r[s_] = rr
            log(f'  (b) {label} {s_} n={n}: ' + '; '.join(f'{e} d {rr[e]["d"]:+.4f} (H1 {rr[e]["pred_H1_mA"]:+.4f}) ch {100 * rr[e]["changed"]:.1f}%' for e in B_EDITS)
                + f'; iso/rec {rr["iso_over_rec"]["v"]:+.2f} {np.round(rr["iso_over_rec"]["ci"], 2)} p(iso=rec) {rr["perm_iso_eq_rec"]:.3g} p(iso=0) {rr["iso"]["perm_d0"]:.3g}')
        return r
    for k in names: out['b_edits'][k] = b_model(D[f'S_L_{k}'][:, wz_idx], k)
    if LG is not None:
        L_ = LG['L']; E_ = LG['E']; rows_order = np.concatenate([np.where(yS[wz_idx] == iZ)[0], np.where(yS[wz_idx] == iW)[0]])
        inv = np.argsort(rows_order); full = np.full((len(SE), len(wz_idx), 10), np.nan, np.float32)
        for e, i in E_.items(): full[SE[e]] = L_[i][inv]
        full[SE['rec_rb']] = full[SE['rec']]; full[SE['base_rb']] = full[SE['base']]   # L-GATr ignores the stored axis: rebuilt = unrebuilt four-vectors
        out['b_edits']['LGATr'] = b_model(full, 'LGATr')
    # ---- (c) mirror split (all S jets, 10 classes; class-balanced)
    out['c_mirror'] = {}; famc = []
    wcls = 1.0 / np.bincount(yS, minlength=10)[yS] / 10
    BsC = np.stack([rng.integers(0, len(yS), len(yS)).astype(np.int32) for _ in range(min(NB, 1000))])
    for k in names:
        L_ = D[f'S_L_{k}']; c0 = L_[SE['base']].argmax(1); r = dict(acc_base=float((c0 == yS).mean()))
        ch = {e: L_[SE[e]].argmax(1) != c0 for e in C_EDITS}
        for e in C_EDITS:
            x = ch[e]; r[e] = dict(changed_balanced=float((x * wcls).sum() / wcls.sum()), changed_raw=float(x.mean()),
                                   ci=ci([(x[b] * wcls[b]).sum() / wcls[b].sum() for b in BsC]), acc_after=float((L_[SE[e]].argmax(1) == yS).mean()),
                                   mean_abs_dWZ=float(np.abs(dfun(L_[SE[e]]) - dfun(L_[SE['base']]))[wzS].mean()))
            if e != 'mirp' and FLIPS[e][0] and k.endswith('full'):
                r[e]['mcnemar_vs_mirp'] = mcnemar(ch['mirp'], ch[e]); famc.append((k, e, r[e]['mcnemar_vs_mirp']['p']))
        out['c_mirror'][k] = r
        log(f'  (c) {k}: acc {r["acc_base"]:.4f}; changed (balanced) ' + ' '.join(f'{e} {100 * r[e]["changed_balanced"]:.2f}%' for e in C_EDITS))
    for (k, e, p), ph in zip(famc, holm([f[2] for f in famc])):
        out['c_mirror'][k][e]['mcnemar_vs_mirp']['p_holm'] = ph; out['c_mirror'][k][e]['mcnemar_vs_mirp']['m'] = len(famc)
        log(f'  (c) McNemar {k} mirp vs {e}: {out["c_mirror"][k][e]["mcnemar_vs_mirp"]}')
    # ---- figure and json
    try:
        figure(D, out, GSwz, ySwz, names)
    except Exception as ex:
        import traceback; traceback.print_exc(); log('figure failed', repr(ex))
    OUT_JSON.write_text(json.dumps(strip(out), indent=1)); log('wrote', OUT_JSON)


def strip(o):
    if isinstance(o, dict): return {str(k): strip(v) for k, v in o.items() if not str(k).startswith('_')}
    if isinstance(o, (list, tuple)): return [strip(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, float) and not math.isfinite(o): return None
    return o


def figure(D, out, G, y, names):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    mods = [k for k in ('ParT_full', 'ParT_kin', 'PN_full', 'PN_kin', 'LGATr') if k in out['a_natural']]
    lab = {'ParT_full': 'ParT\nfull', 'ParT_kin': 'ParT\nkin', 'PN_full': 'ParticleNet\nfull', 'PN_kin': 'ParticleNet\nkin', 'LGATr': 'L-GATr'}
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.8), gridspec_kw=dict(width_ratios=[1, 1.15, 1.25]))
    # (left) a matched natural pair, ParT_full pairs on S
    I, B = out['a_natural']['ParT_full']['S_eta']['_pairs']
    gap = G['cabs'][I] - G['cabs'][B]; j = int(np.argsort(-gap)[min(5, len(gap) - 1)]); it, ib = I[j], B[j]
    wz_cls = {iZ: 'Zqq', iW: 'Wqq'}; ax = axs[0]
    for (ii, col, nm, off) in ((it, 'tab:red', 'off-centre', 0), (ib, 'tab:blue', 'centred (matched)', 0)):
        c = wz_cls[y[ii]]; sel = np.where(y == y[ii])[0]; pos = int(np.searchsorted(sel, ii))
        pt = D[f'X_{c}_pt'][pos].astype(float); de = D[f'X_{c}_de'][pos].astype(float); dp = D[f'X_{c}_dp'][pos].astype(float); mk = pt > 0
        ax.scatter(de[mk], dp[mk], s=6 + 300 * pt[mk] / pt.max(), color=col, alpha=0.45, edgecolor='none',
                   label=f'{nm}: m = {G["minv"][ii]:.0f} GeV, $p_T$ = {G["ptj"][ii]:.0f} GeV, |c| = {G["cabs"][ii]:.3f}')
        ax.plot(G['ce'][ii], G['cp'][ii], marker='x', color=col, ms=11, mew=2.5)
    ax.plot(0, 0, 'k+', ms=16, mew=2, label='stored jet axis (= momentum direction)')
    ax.set_xlim(-0.8, 0.8); ax.set_ylim(-0.8, 0.8); ax.set_aspect('equal'); ax.grid(alpha=0.3)
    ax.set_xlabel('$\\Delta\\eta$ from stored axis'); ax.set_ylabel('$\\Delta\\phi$ from stored axis [rad]')
    ax.legend(fontsize=6.8, loc='lower left'); ax.set_title('(a) a matched natural pair, unedited (x = $p_T$ centroid)', fontsize=9.5)
    # (centre) R_nat per model
    ax = axs[1]; xs = np.arange(len(mods))
    for off, key, rk, col, nm in ((-0.27, 'S', 'R_nat', 'tab:gray', 'matched in m, $p_T$, N, $M_{2c}$ (plan variables), pre-registered 20k jets'),
                                  (-0.09, 'all', 'R_nat', 'k', 'same matching, all 200k jets'),
                                  (0.09, 'all_eta', 'R_nat', 'tab:orange', 'also matched in |$\\eta_{jet}$| (200k jets; few pairs survive)'),
                                  (0.27, 'all_eta', 'R_nat_ols_cov', 'tab:purple', 'same, also adjusted for jet shape')):
        for i, m in enumerate(mods):
            r = out['a_natural'][m].get(key)
            if r is None: continue
            v, c = r[rk]['v'], r[rk]['ci']
            ax.errorbar(i + off, v, yerr=[[max(v - c[0], 0)], [max(c[1] - v, 0)]], fmt='o', color=col, capsize=3, label=nm if i == 0 else None)
            if rk == 'R_nat': ax.text(i + off + 0.03, v, f' {r["n_pairs"]}', fontsize=6, ha='left', va='center', color=col)
    ax.axhline(0, color='k', lw=0.9); ax.axhline(1, color='tab:green', lw=1.2, ls='--')
    ax.text(-0.45, 1.6, 'H1 = 1: follows axis-moment mass $m_A$', ha='left', fontsize=7.5, color='tab:green')
    ax.text(-0.45, -1.8, 'H0 = 0: invariant mass only', ha='left', fontsize=7.5)
    ax.set_xticks(xs); ax.set_xticklabels([lab[m] for m in mods], fontsize=8)
    ax.set_ylabel('$R_{nat}$ = (score gap $-$ $m_{inv}$ prediction)\n/ ($m_A$ prediction $-$ $m_{inv}$ prediction)')
    ax.legend(fontsize=6.3, loc='lower right'); ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(-30, 9); ax.set_title('(a) off-centre vs matched centred W/Z jets, unedited (pair counts at points)', fontsize=9.5)
    # (right) edits on the top-20 % off-centre jets: d and H1 prediction
    ax = axs[2]; eds = ['rec', 'iso', 'rs', 'rec_rb', 'base_rb', 'dil_rec']; w = 0.8 / len(mods)
    for i, m in enumerate(mods):
        rr = out['b_edits'][m]['top20']; v = [rr[e]['d'] for e in eds]; c = [rr[e]['d_ci'] for e in eds]
        ax.bar(np.arange(len(eds)) + (i - (len(mods) - 1) / 2) * w, v, w, yerr=[[max(a - b[0], 0) for a, b in zip(v, c)], [max(b[1] - a, 0) for a, b in zip(v, c)]],
               capsize=2, label=lab[m].replace('\n', ' '))
        ax.scatter(np.arange(len(eds)) + (i - (len(mods) - 1) / 2) * w, [rr[e]['pred_H1_mA'] for e in eds], marker='_', s=90, color='k', zorder=5,
                   label='H1 ($m_A$) prediction' if i == 0 else None)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(range(len(eds))); ax.set_xticklabels(['recentre\n(axis kept)', 'shift |c| at 120°\n($M_2$ fixed)', 'shift |c|\nrandom dir.', 'recentre +\naxis rebuilt',
                                                       'axis rebuilt\nonly', 'matched real\nmass change'], fontsize=7)
    ax.set_ylabel('mean change of $\\ell_Z - \\ell_W$'); ax.legend(fontsize=6.5, ncol=2); ax.grid(axis='y', alpha=0.3)
    ax.set_title(f'(b) edits on the 20 % most off-centre W/Z jets (N = {out["b_edits"]["ParT_full"]["top20"]["n"]})', fontsize=9.5)
    fig.subplots_adjust(left=0.045, right=0.99, bottom=0.16, top=0.92, wspace=0.42); fig.savefig(OUT_PNG, dpi=130); log('wrote', OUT_PNG)
    # mirror figure
    fig, ax = plt.subplots(figsize=(9, 4.2)); eds = C_EDITS; mm = [k for k in names]
    lbl = {'mirp': 'mirror φ', 'mirp_q': '+ charge', 'mirp_d0': '+ d0 sign', 'mirp_dz': '+ dz sign', 'mirp_qd0': '+ charge, d0', 'mirp_all': '+ charge, d0, dz',
           'q': 'charge only\n(no mirror)', 'd0': 'd0 only\n(no mirror)', 'dz': 'dz only\n(no mirror)'}
    w = 0.8 / len(mm)
    for i, k in enumerate(mm):
        r = out['c_mirror'][k]; v = [100 * r[e]['changed_balanced'] for e in eds]; c = [r[e]['ci'] for e in eds]
        ax.bar(np.arange(len(eds)) + (i - (len(mm) - 1) / 2) * w, v, w, yerr=[[a - 100 * b[0] for a, b in zip(v, c)], [100 * b[1] - a for a, b in zip(v, c)]], capsize=2,
               label=k.replace('PN', 'ParticleNet').replace('_', ' '))
    ax.set_xticks(range(len(eds))); ax.set_xticklabels([lbl[e] for e in eds], fontsize=7.5); ax.set_ylabel('10-class calls changed [%] (class-balanced)')
    ax.set_title(f'(c) φ mirror about the jet axis, with signed track inputs flipped (N = {len(D["S_y"])} jets, 10 classes)', fontsize=9.5)
    ax.legend(fontsize=7.5); ax.grid(axis='y', alpha=0.3); fig.tight_layout(); fig.savefig(OUT_PNG2, dpi=130); log('wrote', OUT_PNG2)


if __name__ == '__main__':
    {'nets': stage_nets, 'lgatr': stage_lgatr, 'analyze': stage_analyze}[ARGS.stage]()
