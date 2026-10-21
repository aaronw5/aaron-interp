"""H48 (loop D, new claim D20, exploratory): do the JetClass taggers use the kinematic endpoint of the visible lepton + b system
in leptonic top jets (m_bl <= sqrt(m_t^2 - m_W^2) = 152.6 GeV) and the visible-mass edge of H -> W W* -> l nu q q' (m_vis <= m_H =
125 GeV), beyond a smooth preference for jet mass and the presence / isolation of the lepton (D2)?

Sample (random rows, never first-k)
  JetClass test_20M TTBarLep_100, HToWW2Q1L_100, ZJetsToNuNu_100 (QCD), TTBar_100 (hadronic top), PER random rows each
  (np.random.default_rng(4801 + k), k = 0..3). Every variant, the unedited jet included, is rebuilt from its particle four-vectors
  about its own E-scheme axis (axis = direction of the summed four-momentum; eta = jet_eta + sign(jet_eta) * deta convention of
  part_core.build).
Definitions
  lepton l   the hardest real particle with isElectron or isMuon set. b system B = every other real particle of the jet (the neutrino
             is invisible, so B holds the b quark's hadrons plus radiation and, for H jets, the q q' pair).
  m_vis      invariant mass of B + l = the invariant mass of the whole jet (exactly, since B is all other particles). For t -> b l nu
             without radiation m_vis = m_bl <= sqrt(m_t^2 - m_W^2) = 152.6 GeV (m_t 172.5, m_W 80.4); for H -> l nu q q' m_vis <= 125 GeV.
  theta      3-D opening angle between the lepton momentum and the B momentum (lab frame).
Edits (only the lepton moves; its energy, |p|, mass and all its other inputs are kept; B is untouched)
  E(dm)  rotate the lepton momentum in the plane spanned by p_B and p_l, changing theta to theta' with
         m_vis'^2 = m_B^2 + m_l^2 + 2 (E_B E_l - |p_B| |p_l| cos theta') = (m_vis + dm)^2 (solved exactly; dm = +20 GeV).
  C      control: rotate the lepton about the p_B direction (theta and so m_vis fixed) by the azimuth phi that moves its direction by
         the same 3-D angle delta = |theta' - theta| as E: cos delta = cos^2 theta + sin^2 theta cos phi; sign of phi random (rng 4810).
  TN     translation control: the unedited particles rebuilt about the E-edited jet's axis (the input shift that E causes through
         the axis alone).
  A jet enters an arm only if E and C exist and the lepton stays within Delta R < 0.8 of the edited jet axis in both.
Arms (equal dm = +20 GeV; windows on unedited m_vis; edge X = 152.6 GeV for TTBarLep, 125 GeV for HToWW2Q1L)
  below    m_vis in [X - 43, X - 23]  -> [X - 23, X - 3]   (stays below the edge)
  across   m_vis in [X - 13, X - 3]   -> [X + 7,  X + 17]  (crosses the edge)
  reverse  m_vis in [X + 7,  X + 27]  with dm = -20 GeV  (crosses back to below; exploratory)
Models: ParT_full, PN_full (see the isElectron / isMuon flags), ParT_kin, PN_kin (four-momenta only), ParT_perm1..6 (ParT_full with
  every weight tensor permuted, torch seeds 5..10: the null band).
Scores (10-class logits L; QCD 0, Hqql 5, Tbqq 8, Tbl 9): s_Tbl = L_Tbl - logsumexp(L_QCD, L_Tbqq) = log p(Tbl)/(p(QCD)+p(Tbqq));
  s_Hqql likewise. TTBarLep jets are scored with s_Tbl, HToWW2Q1L jets with s_Hqql.
References (sklearn, trained on unedited jets of train_100M <pre>_000, NREF random rows per class, rng 4821..4824; signal vs
  QCD + TTBar): LR(m) logistic on m_vis; MLP(m) MLP (32, 32) on m_vis; MLP(m, lep) on (m_vis, f_l = pT_l / pT_jet, relative
  isolation I_l = sum pT of other particles within Delta R < 0.2 of the lepton / pT_l, has_lepton). Each is evaluated on the same
  edited jets; its change is the response a smooth mass (or mass + lepton) dependence predicts.
Statistics: per (model, class, arm) E - C per jet: mean, 95 % bootstrap CI over jets (B), two-sided sign-flip permutation p;
  across - below: difference of arm means, bootstrap CI, arm-label permutation p; across net - MLP(m): sign-flip p. Holm over
  (4 trained nets x 2 classes x {E-C below, E-C across, E-C reverse, across-below, across net-MLP(m)}). McNemar on calls at the
  50 % signal-efficiency threshold (median s over unedited signal jets), E vs C. Data check: fraction of TTBarLep (HToWW2Q1L) jets
  with a lepton and m_vis above the edge; binned mean score vs m_vis on unedited jets; slope change at the edge (bootstrap).
Run: ./sync.sh && AI_PY=conda ./job.sh d-h48-run 4 8 'export PATH=/j-jepa-vol/aaron-interp/env-part/bin:$PATH JETDISTILL_TAGGER=part
     JETDISTILL_DEVICE=cpu JETDISTILL_DATA=/j-jepa-vol/aaron-interp/code/cluster/data_part JETDISTILL_MODELS=/j-jepa-vol/aaron-interp/models/part;
     python -m interp.h48_lepton_endpoint'      (H48_SMOKE=1: 300 per class, 2 permuted nets, 2,000 reference rows)
Outputs: $AI_RESULTS/physics/h48_lepton_endpoint.json, h48_perjet.npz, h48_jet_edit.png, h48_response.png
"""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
SMOKE = bool(os.environ.get('H48_SMOKE'))
RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'physics'
TAG = 'h48_smoke' if SMOKE else 'h48'
PER = 300 if SMOKE else 3000
NREF = 2000 if SMOKE else 15000
B = 200 if SMOKE else 1000
NPERM = 500 if SMOKE else 5000
NPN = 2 if SMOKE else 6
QCD, HQQL, TBQQ, TBL = 0, 5, 8, 9
MT, MW, MH = 172.5, 80.4, 125.0
EDGE = dict(Tbl=float(np.sqrt(MT ** 2 - MW ** 2)), Hqql=MH)
CLS = (('Tbl', 'TTBarLep', TBL), ('Hqql', 'HToWW2Q1L', HQQL), ('QCD', 'ZJetsToNuNu', QCD), ('Tbqq', 'TTBar', TBQQ))
SIGC = dict(Tbl=TBL, Hqql=HQQL)
TEST = Path('/j-jepa-vol/JetClass/Pythia/test_20M')
TRAIN = Path('/j-jepa-vol/JetClass/Pythia/train_100M')
ARMS = dict(below=(-43.0, -23.0, 20.0), across=(-13.0, -3.0, 20.0), reverse=(7.0, 27.0, -20.0))
TESTED_NETS = ('ParT_full', 'PN_full', 'ParT_kin', 'PN_kin')
PERM_NETS = tuple(f'ParT_perm{i}' for i in range(1, NPN + 1))
NETS = TESTED_NETS + PERM_NETS
REFS = ('LR_m', 'MLP_m', 'MLP_mlep')
VARS = ('E', 'C', 'TN')
RISO, RJET = 0.2, 0.8


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ================================================================================================================ data
def load_rows(path, n, seed, chunk=20000):
    import uproot
    from interp.part_core import read_root, sub, cat
    N = uproot.open(path)['tree'].num_entries
    if SMOKE: N = min(N, 40000)
    rows = np.sort(np.random.default_rng(seed).choice(N, n, replace=False)); parts = []
    for s in range(0, N, chunk):
        r = rows[(rows >= s) & (rows < s + chunk)]
        if len(r): parts.append(sub(read_root(path, start=s, stop=min(s + chunk, N)), r - s))
    log(f'{path.name}: {n} random rows of {N} (rng {seed})')
    return cat(parts), dict(file=path.name, rows_in_file=int(N), drawn=int(n), seed=seed)


def p4(J):
    m = J['mask'] > 0
    return (np.stack([J['px'], J['py'], J['pz'], J['energy']], -1).astype(np.float64) * m[..., None]), m


def lepton_index(J):
    """slot of the hardest real particle flagged electron or muon (-1 if none); pT order computed, not assumed"""
    m = J['mask'] > 0; isl = m & ((J['isElectron'] > 0) | (J['isMuon'] > 0)); pt = np.hypot(J['px'], J['py'])
    key = np.where(isl, pt, -1.0); i = key.argmax(1); return np.where(isl.any(1), i, -1)


def m4(v):
    return np.sqrt(np.maximum(v[..., 3] ** 2 - (v[..., :3] ** 2).sum(-1), 0))


def axis_of(P, m):
    S = (P * m[..., None]).sum(1); pt = np.hypot(S[:, 0], S[:, 1])
    return np.arcsinh(S[:, 2] / np.maximum(pt, 1e-12)), np.arctan2(S[:, 1], S[:, 0])


def rebuild(J, P, m, axis=None):
    """jet dict (part_core.build) from particle four-vectors P (N, 128, 4) about the E-scheme axis of P (or `axis`)"""
    from interp.part_core import build
    jeta, jphi = axis_of(P, m) if axis is None else axis
    pt = np.hypot(P[..., 0], P[..., 1]) * m; ptc = np.where(m, pt, 1.0)
    eta = np.where(m, np.arcsinh(P[..., 2] / ptc), 0); phi = np.where(m, np.arctan2(P[..., 1], P[..., 0]), 0)
    mass = m4(P) * m; sgn = np.where(jeta < 0, -1.0, 1.0)[:, None]
    deta = np.where(m, sgn * (eta - jeta[:, None]), 0); dphi = np.where(m, wrap(phi - jphi[:, None]), 0)
    J2 = dict(J); J2['jet_eta'] = jeta.astype(np.float32); J2['jet_phi'] = jphi.astype(np.float32)
    return build(J2, pt, deta, dphi, mass)


def obs(Jd):
    """per-jet observables of a (built) jet dict: m_vis, lepton pT fraction f_l, relative isolation I_l, has_lepton, lepton Delta R
    to the axis, theta (lepton - B opening angle)"""
    from interp.part_core import jmass, kin
    pt, de, dp, _ = kin(Jd); li = lepton_index(Jd); has = li >= 0; r = np.arange(len(li)); lj = np.where(has, li, 0)
    ptl = pt[r, lj]; dR = np.hypot(de - de[r, lj][:, None], wrap(dp - dp[r, lj][:, None]))
    m = Jd['mask'] > 0; oth = m.copy(); oth[r, lj] = False
    iso = np.where(has, (pt * (oth & (dR < RISO))).sum(1) / np.maximum(ptl, 1e-9), 0.0)
    P, _ = p4(Jd); Pl = P[r, lj]; PB = P.sum(1) - Pl
    cth = (Pl[:, :3] * PB[:, :3]).sum(1) / np.maximum(np.linalg.norm(Pl[:, :3], axis=1) * np.linalg.norm(PB[:, :3], axis=1), 1e-12)
    return dict(m=jmass(Jd).astype(np.float64), f=np.where(has, ptl / Jd['jet_pt'], 0.0), iso=iso, has=has.astype(np.float64),
                dRl=np.where(has, np.hypot(de[r, lj], dp[r, lj]), np.nan), theta=np.where(has, np.arccos(np.clip(cth, -1, 1)), np.nan))


# ================================================================================================================ edits
def edit_arm(J, rows, dm, rng):
    """E, C, TN four-vectors for the jets `rows` (all with a lepton); returns dict of P arrays, masks, ok, bookkeeping"""
    from interp.part_core import sub
    Ja = sub(J, rows); P, m = p4(Ja); li = lepton_index(Ja); r = np.arange(len(rows)); assert (li >= 0).all()
    Pl = P[r, li].copy(); PB = P.sum(1) - Pl
    pl = np.linalg.norm(Pl[:, :3], axis=1); pb = np.linalg.norm(PB[:, :3], axis=1)
    bh = PB[:, :3] / pb[:, None]; lh = Pl[:, :3] / pl[:, None]
    cth = np.clip((lh * bh).sum(1), -1, 1); th = np.arccos(cth)
    mB2 = PB[:, 3] ** 2 - pb ** 2; ml2 = Pl[:, 3] ** 2 - pl ** 2; m0 = m4(P.sum(1)); m1 = m0 + dm
    cth1 = (mB2 + ml2 + 2 * PB[:, 3] * Pl[:, 3] - m1 ** 2) / (2 * pb * pl)
    okE = (np.abs(cth1) <= 1) & (m1 > 0) & (th > 1e-6); th1 = np.arccos(np.clip(cth1, -1, 1))
    eh = lh - cth[:, None] * bh; eh /= np.maximum(np.linalg.norm(eh, axis=1), 1e-12)[:, None]
    lE = np.cos(th1)[:, None] * bh + np.sin(th1)[:, None] * eh
    delta = np.abs(th1 - th); cphi = (np.cos(delta) - cth ** 2) / np.maximum(np.sin(th) ** 2, 1e-12)
    okC = okE & (cphi >= -1) & (cphi <= 1); phi = np.arccos(np.clip(cphi, -1, 1)) * rng.choice([-1.0, 1.0], len(rows))
    fh = np.cross(bh, eh)
    lC = cth[:, None] * bh + np.sin(th)[:, None] * (np.cos(phi)[:, None] * eh + np.sin(phi)[:, None] * fh)
    out = {}
    for v, lv in (('E', lE), ('C', lC)):
        Q = P.copy(); Q[r, li, :3] = lv * pl[:, None]; Q[r, li, 3] = Pl[:, 3]; out[v] = Q
    axE = axis_of(out['E'], m); out['TN'] = (P, axE)
    # checks
    mE = m4(out['E'].sum(1)); mC = m4(out['C'].sum(1))
    ok = okC.copy()
    for v in ('E', 'C'):
        ae, ap = axis_of(out[v], m); Q = out[v][r, li]; pt_ = np.hypot(Q[:, 0], Q[:, 1])
        dR = np.hypot(np.arcsinh(Q[:, 2] / pt_) - ae, wrap(np.arctan2(Q[:, 1], Q[:, 0]) - ap)); ok &= dR < RJET
    ang = lambda a, b_: np.arccos(np.clip((a * b_).sum(1), -1, 1))
    bk = dict(m0=m0, mE=mE, mC=mC, theta=th, theta_E=th1, delta_E=ang(lh, lE), delta_C=ang(lh, lC), okE=okE, okC=okC, ok=ok,
              El=Pl[:, 3], ptl=np.hypot(Pl[:, 0], Pl[:, 1]), ptlE=np.hypot(out['E'][r, li, 0], out['E'][r, li, 1]),
              ptlC=np.hypot(out['C'][r, li, 0], out['C'][r, li, 1]))
    if ok.any():
        assert np.allclose(mE[ok], m1[ok], atol=1e-4 * np.maximum(m1[ok], 1)), 'E mass target missed'
        assert np.allclose(mC[ok], m0[ok], atol=1e-4 * np.maximum(m0[ok], 1)), 'C changed the mass'
        assert np.allclose(bk['delta_C'][ok], bk['delta_E'][ok], atol=1e-6), 'C displacement != E displacement'
    return Ja, m, out, bk


# ================================================================================================================ models
def load_net(name):
    from interp.h44_bc_lifetime import load_net as L44
    return L44(name)


def logits(M, Jd):
    from interp.h44_bc_lifetime import logits as G44
    return G44(M, Jd)


def score(L, c):
    return L[:, SIGC[c]] - np.logaddexp(L[:, QCD], L[:, TBQQ])


def lo(p):
    p = np.clip(p, 1e-7, 1 - 1e-7); return np.log(p / (1 - p))


# ================================================================================================================ stats
def boot(x, rng, b=B):
    x = np.asarray(x, float)
    if len(x) == 0: return float('nan'), [float('nan')] * 2
    mm = x[rng.integers(0, len(x), (b, len(x)))].mean(1); return float(x.mean()), np.percentile(mm, [2.5, 97.5]).tolist()


def perm2(x, rng, n=NPERM):
    x = np.asarray(x, float)
    if len(x) == 0 or np.all(x == 0): return 1.0
    obs_ = abs(x.mean()); s = rng.choice([-1.0, 1.0], (n, len(x))); return float((np.sum(np.abs((s * x).mean(1)) >= obs_ - 1e-15) + 1) / (n + 1))


def diff2(a, b_, rng):
    """across - below: mean difference, bootstrap CI (independent resampling), arm-label permutation p"""
    a, b_ = np.asarray(a, float), np.asarray(b_, float)
    if len(a) == 0 or len(b_) == 0: return dict(mean=float('nan'), ci=[float('nan')] * 2, p_perm=1.0, n=[len(a), len(b_)])
    d = a.mean() - b_.mean(); bs = a[rng.integers(0, len(a), (B, len(a)))].mean(1) - b_[rng.integers(0, len(b_), (B, len(b_)))].mean(1)
    z = np.concatenate([a, b_]); cnt = 0
    for _ in range(NPERM // 5):
        q = rng.permutation(z); cnt += abs(q[:len(a)].mean() - q[len(a):].mean()) >= abs(d) - 1e-15
    return dict(mean=float(d), ci=np.percentile(bs, [2.5, 97.5]).tolist(), p_perm=float((cnt + 1) / (NPERM // 5 + 1)), n=[len(a), len(b_)])


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); adj = np.empty_like(ps); run = 0.0
    for r, i in enumerate(o): run = max(run, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run
    return adj


def mcnemar(a, b_):
    from scipy.stats import binomtest
    n01 = int((a & ~b_).sum()); n10 = int((~a & b_).sum())
    return dict(pass_to_fail=n01, fail_to_pass=n10, p=float(binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0)


def auc(s, y):
    from scipy.stats import rankdata
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.ndarray): return clean(o.tolist())
    return o


def slope_change(m, s, X, rng, w=40.0):
    """least-squares slope of s vs m on [X, X + w] minus on [X - w, X]; bootstrap CI over jets"""
    def one(mm, ss):
        a = (mm >= X - w) & (mm < X); b_ = (mm >= X) & (mm < X + w)
        if a.sum() < 5 or b_.sum() < 5: return np.nan, np.nan, np.nan
        sa = np.polyfit(mm[a], ss[a], 1)[0]; sb = np.polyfit(mm[b_], ss[b_], 1)[0]; return sb - sa, sa, sb
    d, sa, sb = one(m, s); bs = []
    for _ in range(min(B, 500)):
        i = rng.integers(0, len(m), len(m)); bs.append(one(m[i], s[i])[0])
    bs = np.array(bs, float); bs = bs[np.isfinite(bs)]
    return dict(slope_below=sa, slope_above=sb, change=d, ci=np.percentile(bs, [2.5, 97.5]).tolist() if len(bs) else [np.nan] * 2,
                n_below=int(((m >= X - w) & (m < X)).sum()), n_above=int(((m >= X) & (m < X + w)).sum()))


# ================================================================================================================ references
REF_COLS = dict(LR_m=('m',), MLP_m=('m',), MLP_mlep=('m', 'f', 'iso', 'has'))


def fit_refs(out):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    F = {}; out['ref_rows'] = {}
    for k, (c, pre, lab) in enumerate(CLS):
        Jr, inf = load_rows(TRAIN / f'{pre}_000.root', NREF, 4821 + k); out['ref_rows'][pre] = inf
        F[c] = obs(Jr); F[c]['y'] = np.full(len(Jr['label']), lab); del Jr
    REF = {}; out['ref_info'] = {}
    for c in SIGC:
        Fc = {k: np.concatenate([F[x][k] for x in (c, 'QCD', 'Tbqq')]) for k in ('m', 'f', 'iso', 'has')}
        y = np.concatenate([np.ones(len(F[c]['m'])), np.zeros(len(F['QCD']['m']) + len(F['Tbqq']['m']))])
        tr = np.random.default_rng(4825).random(len(y)) < 0.8
        for r in REFS:
            X = np.column_stack([Fc[k] for k in REF_COLS[r]])
            mdl = (make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)) if r == 'LR_m' else
                   make_pipeline(StandardScaler(), MLPClassifier((32, 32), max_iter=400, early_stopping=True, random_state=0)))
            mdl.fit(X[tr], y[tr]); REF[(c, r)] = mdl
            out['ref_info'][f'{c}|{r}'] = dict(features=REF_COLS[r], heldout_acc=float(mdl.score(X[~tr], y[~tr])),
                                               heldout_auc=auc(mdl.predict_proba(X[~tr])[:, 1], y[~tr].astype(int)))
            log(f'ref {c} {r}: ' + json.dumps(out['ref_info'][f'{c}|{r}']))
        # ref curve: log-odds vs m (lepton features at the signal-class medians)
        grid = np.arange(40.0, 230.0, 5.0); med = {k: float(np.median(F[c][k][F[c]['has'] > 0])) for k in ('f', 'iso')}
        out['ref_info'][f'{c}|curve'] = dict(m=grid, LR_m=lo(REF[(c, 'LR_m')].predict_proba(grid[:, None])[:, 1]),
                                              MLP_m=lo(REF[(c, 'MLP_m')].predict_proba(grid[:, None])[:, 1]),
                                              MLP_mlep=lo(REF[(c, 'MLP_mlep')].predict_proba(np.column_stack([grid, np.full_like(grid, med['f']), np.full_like(grid, med['iso']), np.ones_like(grid)]))[:, 1]),
                                              lepton_medians=med)
        # training-sample endpoint check
        hs = F[c]['has'] > 0
        out['ref_info'][f'{c}|train_frac_above_edge'] = float((F[c]['m'][hs] > EDGE[c]).mean())
    return REF


def ref_lo(REF, c, r, O):
    return lo(REF[(c, r)].predict_proba(np.column_stack([O[k] for k in REF_COLS[r]]))[:, 1])


# ================================================================================================================ main
def main():
    from interp.part_core import cat
    RES.mkdir(parents=True, exist_ok=True)
    out = dict(node='H48', smoke=SMOKE, per_class=PER, n_ref=NREF, B=B, n_perm=NPERM, edges=EDGE, arms=ARMS, r_iso=RISO)
    parts = []; out['sample'] = {}
    for k, (c, pre, lab) in enumerate(CLS):
        Jc, inf = load_rows(TEST / f'{pre}_100.root', PER, 4801 + k); parts.append(Jc); out['sample'][c] = inf
        assert np.all(Jc['label'] == lab), f'{c} labels'
    J = cat(parts); y = J['label'].astype(int); N = len(y)
    P0, M0 = p4(J); Jid = rebuild(J, P0, M0); O0 = obs(Jid); log(f'sample {N} jets; unedited rebuilt about own E-scheme axis')
    # ---- data check
    dc = {}
    for c, _, lab in CLS:
        s_ = y == lab; hs = s_ & (O0['has'] > 0)
        dc[c] = dict(n=int(s_.sum()), frac_with_lepton=float(hs.sum() / s_.sum()),
                     frac_m_above_top_edge=float((O0['m'][hs] > EDGE['Tbl']).mean()) if hs.any() else None,
                     frac_m_above_H_edge=float((O0['m'][hs] > EDGE['Hqql']).mean()) if hs.any() else None,
                     m_quantiles=np.percentile(O0['m'][hs], [5, 25, 50, 75, 95]).tolist() if hs.any() else None,
                     median_f=float(np.median(O0['f'][hs])) if hs.any() else None, median_iso=float(np.median(O0['iso'][hs])) if hs.any() else None,
                     m_hist=np.histogram(O0['m'][hs], bins=np.arange(0, 305, 5))[0].tolist() if hs.any() else None)
        log(f'data {c}: ' + json.dumps({k: v for k, v in dc[c].items() if k != 'm_hist'}))
    out['data_check'] = dc
    # ---- arms
    rng = np.random.default_rng(4810); ARM = {}
    for c in SIGC:
        X = EDGE[c]
        for a, (lo_, hi_, dm) in ARMS.items():
            sel = np.flatnonzero((y == SIGC[c]) & (O0['has'] > 0) & (O0['m'] >= X + lo_) & (O0['m'] < X + hi_))
            if len(sel) == 0: continue
            Ja, ma, V, bk = edit_arm(J, sel, dm, rng); ok = bk['ok']; rows = sel[ok]
            Jv = {}
            for v in ('E', 'C'): Jv[v] = rebuild(_sub(Ja, ok), V[v][ok], ma[ok])
            Pt, ax = V['TN']; Jv['TN'] = rebuild(_sub(Ja, ok), Pt[ok], ma[ok], axis=(ax[0][ok], ax[1][ok]))
            Ov = {v: obs(Jv[v]) for v in VARS}
            info = dict(n_window=int(len(sel)), n_okE=int(bk['okE'].sum()), n_okC=int(bk['okC'].sum()), n=int(ok.sum()), dm=dm,
                        window=[X + lo_, X + hi_])
            for k in ('m0', 'mE', 'mC', 'theta', 'theta_E', 'delta_E', 'El', 'ptl', 'ptlE', 'ptlC'):
                info[f'mean_{k}'] = float(np.mean(bk[k][ok])) if ok.any() else None
            for v in VARS:
                for k in ('m', 'f', 'iso', 'dRl'):
                    info[f'{v}_mean_d{k}'] = float(np.mean(Ov[v][k] - O0[k][rows])) if ok.any() else None
            ARM[(c, a)] = dict(rows=rows, J=Jv, O=Ov, bk={k: v_[ok] for k, v_ in bk.items()}, info=info)
            log(f'arm {c} {a}: ' + json.dumps(info))
    out['arms_info'] = {f'{c}|{a}': A['info'] for (c, a), A in ARM.items()}
    # ---- networks
    SC = {}; out['unedited'] = {}
    for name in NETS:
        Mdl = load_net(name); Lid = logits(Mdl, Jid); SC[(name, 'id')] = Lid
        bg = np.isin(y, [QCD, TBQQ])
        u = dict(acc_10class=float((Lid.argmax(1) == y).mean()),
                 acc_10class_by_class={c: float((Lid.argmax(1) == lab)[y == lab].mean()) for c, _, lab in CLS},
                 auc_Tbl_vs_QCD_Tbqq=auc(score(Lid, 'Tbl')[(y == TBL) | bg], (y[(y == TBL) | bg] == TBL).astype(int)),
                 auc_Hqql_vs_QCD_Tbqq=auc(score(Lid, 'Hqql')[(y == HQQL) | bg], (y[(y == HQQL) | bg] == HQQL).astype(int)),
                 acc_4class=float((np.array([QCD, HQQL, TBQQ, TBL])[Lid[:, [QCD, HQQL, TBQQ, TBL]].argmax(1)] == y).mean()))
        out['unedited'][name] = u; log(f'{name}: ' + json.dumps(u))
        if name == 'ParT_full':
            Lraw = logits(Mdl, J); dl = np.abs(Lraw - Lid)
            out['input_path_check'] = dict(max_dlogit=float(dl.max()), mean_dlogit=float(dl.mean()), p99_dlogit=float(np.percentile(dl.max(1), 99)),
                                           frac_10class_calls_changed=float((Lraw.argmax(1) != Lid.argmax(1)).mean()),
                                           acc_raw_stored_axis=float((Lraw.argmax(1) == y).mean()))
            log('stored-axis raw vs rebuilt about own E-scheme axis: ' + json.dumps(out['input_path_check']))
        for (c, a), A in ARM.items():
            for v in VARS: SC[(name, c, a, v)] = logits(Mdl, A['J'][v])
        del Mdl
    # ---- references
    REF = fit_refs(out)
    RL = {}
    for c in SIGC:
        for r in REFS: RL[(r, c, 'id')] = ref_lo(REF, c, r, O0)
    for (c, a), A in ARM.items():
        for r in REFS:
            for v in VARS: RL[(r, c, a, v)] = ref_lo(REF, c, r, A['O'][v])
    # ---- statistics
    rs = np.random.default_rng(4899); res = {}; cells = []
    thr = {(n, c): float(np.median(score(SC[(n, 'id')], c)[y == SIGC[c]])) for n in NETS for c in SIGC}

    def resp(mname, c, a, v):
        rows = ARM[(c, a)]['rows']
        if mname in REFS: return RL[(mname, c, a, v)] - RL[(mname, c, 'id')][rows]
        return score(SC[(mname, c, a, v)], c) - score(SC[(mname, 'id')], c)[rows]

    EC = {}
    for (c, a), A in ARM.items():
        for mname in NETS + REFS:
            d = {v: resp(mname, c, a, v) for v in VARS}; EC[(mname, c, a)] = d['E'] - d['C']
            if mname not in REFS:   # secondary score: signal vs all nine other classes
                rows = ARM[(c, a)]['rows']; sr = lambda L: L[:, SIGC[c]] - np.log(np.exp(L - L.max(1, keepdims=True)).sum(1) - np.exp(L[:, SIGC[c]] - L.max(1))) - L.max(1)
                x = (sr(SC[(mname, c, a, 'E')]) - sr(SC[(mname, c, a, 'C')])); mu, ci = boot(x, rs)
                res[f'{c}|{a}|E-C_vs_rest|{mname}'] = dict(mean=mu, ci=ci, n=len(x))
                res[f'{c}|{a}|frac_called_signal_10class|{mname}'] = dict(id=float((SC[(mname, 'id')][rows].argmax(1) == SIGC[c]).mean()),
                    **{v: float((SC[(mname, c, a, v)].argmax(1) == SIGC[c]).mean()) for v in VARS})
            for k, x in (('E', d['E']), ('C', d['C']), ('TN', d['TN']), ('E-C', d['E'] - d['C'])):
                mu, ci = boot(x, rs); cell = dict(mean=mu, ci=ci, n=len(x))
                if k == 'E-C' and mname in TESTED_NETS and len(x) >= 30:
                    cell['p_perm'] = perm2(x, rs); cells.append(f'{c}|{a}|E-C|{mname}')
                res[f'{c}|{a}|{k}|{mname}'] = cell
            if mname in TESTED_NETS + PERM_NETS:
                callf = lambda v: score(SC[(mname, c, a, v)], c) > thr[(mname, c)]
                res[f'{c}|{a}|mcnemar_E_vs_C|{mname}'] = dict(**mcnemar(callf('C'), callf('E')), pass_rate_C=float(callf('C').mean()),
                                                              pass_rate_E=float(callf('E').mean()))
        for k in ('E', 'C', 'TN', 'E-C'):
            pm = [res[f'{c}|{a}|{k}|{n}']['mean'] for n in PERM_NETS]
            res[f'{c}|{a}|{k}|perm_band'] = dict(min=float(np.min(pm)), max=float(np.max(pm)), mean=float(np.mean(pm)))
        for mname in TESTED_NETS:
            x = EC[(mname, c, a)] - EC[('MLP_m', c, a)]; mu, ci = boot(x, rs); cell = dict(mean=mu, ci=ci, n=len(x))
            if a == 'across' and len(x) >= 30: cell['p_perm'] = perm2(x, rs); cells.append(f'{c}|{a}|net-MLP_m|{mname}')
            res[f'{c}|{a}|net-MLP_m|{mname}'] = cell
            x = EC[(mname, c, a)] - EC[('MLP_mlep', c, a)]; mu, ci = boot(x, rs); res[f'{c}|{a}|net-MLP_mlep|{mname}'] = dict(mean=mu, ci=ci, n=len(x))
    for c in SIGC:
        if (c, 'across') in ARM and (c, 'below') in ARM:
            for mname in NETS + REFS:
                d = diff2(EC[(mname, c, 'across')], EC[(mname, c, 'below')], rs); res[f'{c}|across-below|E-C|{mname}'] = d
                if mname in TESTED_NETS and min(d['n']) >= 30: cells.append(f'{c}|across-below|E-C|{mname}')
    adj = holm([res[k]['p_perm'] for k in cells])
    for k, a_ in zip(cells, adj): res[k]['p_holm'] = float(a_)
    out['holm_family'] = dict(n_tests=len(cells), cells=cells); out['results'] = res; out['call_thresholds_eps50'] = {f'{n}|{c}': t for (n, c), t in thr.items()}
    # ---- data curve: mean score vs m on unedited signal jets with a lepton; slope change at the edge
    curve = {}
    for c in SIGC:
        hs = (y == SIGC[c]) & (O0['has'] > 0); bins = np.arange(40.0, 235.0, 10.0); mm = O0['m'][hs]; ib = np.digitize(mm, bins) - 1
        cc = dict(bins=bins, n=[int((ib == i).sum()) for i in range(len(bins) - 1)])
        for mname in TESTED_NETS + ('ParT_perm1',):
            s = score(SC[(mname, 'id')], c)[hs]
            cc[mname] = [float(s[ib == i].mean()) if (ib == i).sum() >= 5 else None for i in range(len(bins) - 1)]
            cc[f'{mname}|slope_change'] = slope_change(mm, s, EDGE[c], rs)
        for r in REFS:
            s = RL[(r, c, 'id')][hs]; cc[r] = [float(s[ib == i].mean()) if (ib == i).sum() >= 5 else None for i in range(len(bins) - 1)]
            cc[f'{r}|slope_change'] = slope_change(mm, s, EDGE[c], rs)
        curve[c] = cc
        log(f'curve {c}: ' + json.dumps({k: (round(v['change'], 4) if isinstance(v, dict) else None) for k, v in cc.items() if 'slope' in k}))
    out['score_vs_m'] = curve
    for (c, a) in ARM:
        s = '  '.join(f'{n} {res[f"{c}|{a}|E-C|{n}"]["mean"]:+.3f}' for n in TESTED_NETS + REFS)
        pb = res[f'{c}|{a}|E-C|perm_band']; log(f'{c} {a} E-C: {s}  perm [{pb["min"]:+.3f}, {pb["max"]:+.3f}]')
    for c in SIGC:
        if f'{c}|across-below|E-C|ParT_full' in res:
            log(f'{c} across-below: ' + '  '.join(f'{n} {res[f"{c}|across-below|E-C|{n}"]["mean"]:+.3f}' for n in TESTED_NETS + REFS))
    (RES / f'{TAG}_lepton_endpoint.json').write_text(json.dumps(clean(out), indent=1))
    keep = dict(y=y, m0=O0['m'], has=O0['has'])
    for (n, *rest), L in SC.items():
        if n in TESTED_NETS or n == 'ParT_perm1': keep['L|' + '|'.join([n] + list(rest))] = L.astype(np.float32)
    for (c, a), A in ARM.items():
        keep[f'rows|{c}|{a}'] = A['rows']
        for k, v in A['bk'].items(): keep[f'bk|{c}|{a}|{k}'] = np.asarray(v, np.float64)
    np.savez_compressed(RES / f'{TAG}_perjet.npz', **keep); log('saved')
    try:
        fig_jet(J, ARM, O0)
    except Exception as e: log('jet figure failed', repr(e))
    try:
        fig_resp(out)
    except Exception as e: log('response figure failed', repr(e))


def _sub(J, ok):
    return {k: (v[ok] if np.ndim(v) else v) for k, v in J.items()}


# ================================================================================================================ figures
def fig_jet(J, ARM, O0):
    import matplotlib
    matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from interp.part_core import kin
    A = ARM[('Tbl', 'across')]; bk = A['bk']; i = int(np.argmin(np.abs(bk['m0'] - np.median(bk['m0']))))
    j = A['rows'][i]
    from interp.part_core import sub
    P, m = p4(sub(J, np.array([j]))); J0 = rebuild(sub(J, np.array([j])), P, m)
    panels = [('unedited', J0, None), (f'E: lepton rotated away from the b system\n(opening angle {np.degrees(bk["theta"][i]):.1f}° → {np.degrees(bk["theta_E"][i]):.1f}°)', _sub(A['J']['E'], np.array([i])), 'E'),
              (f'C: lepton rotated around the b-system direction\n(opening angle fixed, moved by the same {np.degrees(bk["delta_E"][i]):.1f}°)', _sub(A['J']['C'], np.array([i])), 'C')]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.4), sharex=True, sharey=True)
    for ax, (tt, Jd, v) in zip(axs, panels):
        from interp.part_core import jmass
        pt, de, dp, _ = kin(Jd); mk = Jd['mask'][0] > 0; li = lepton_index(Jd)[0]
        oth = mk.copy(); oth[li] = False
        ax.scatter(de[0, oth], dp[0, oth], s=4 + 300 * pt[0, oth] / pt[0].max(), c='#7f8c9a', alpha=0.7, label='other particles (b system), area ∝ pT')
        ax.scatter([de[0, li]], [dp[0, li]], s=4 + 300 * pt[0, li] / pt[0].max(), marker='*', c='#c0392b', edgecolors='k', label='lepton')
        if v is not None:
            ax.scatter([O_l[0]], [O_l[1]], marker='x', c='#c0392b', s=60, label='lepton before the edit (unedited-axis coordinates)')
        else:
            O_l = (de[0, li], dp[0, li])
        ax.add_patch(plt.Circle((0, 0), 0.8, fill=False, ls=':', color='k', lw=0.8))
        ax.set_title(f'{tt}\njet mass m_vis = {float(jmass(Jd)[0]):.1f} GeV', fontsize=9); ax.set_xlabel('Δη to the jet axis')
        ax.set_aspect('equal')
    axs[0].set_ylabel('Δφ to the jet axis'); axs[0].legend(fontsize=7, loc='lower left', frameon=False)
    fig.suptitle(f'One JetClass t→bℓν jet (pT {J["jet_pt"][j]:.0f} GeV): only the lepton moves, its energy fixed; '
                 f'E raises the jet mass above the 152.6 GeV endpoint, C keeps it', fontsize=11, y=1.02)
    fig.savefig(RES / f'{TAG}_jet_edit.png', dpi=120, bbox_inches='tight'); plt.close(fig); log('jet figure saved')


def fig_resp(out):
    import matplotlib
    matplotlib.use('Agg'); import matplotlib.pyplot as plt
    R = out['results']; fig, axs = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw=dict(width_ratios=[1.3, 1]))
    ax = axs[0]; arms = [(c, a) for c in ('Tbl', 'Hqql') for a in ('below', 'across', 'reverse') if f'{c}|{a}|E-C|ParT_full' in R]
    ms = [('ParT_full', 'ParT_full', '#2f5d8a'), ('PN_full', 'ParticleNet_full', '#c0603a'), ('ParT_kin', 'ParT_kin', '#7fa7cc'),
          ('PN_kin', 'ParticleNet_kin', '#e0a080'), ('MLP_m', 'MLP(m) reference', '#8a8a8a'), ('MLP_mlep', 'MLP(m, lepton) reference', '#5a9a5a')]
    w = 0.13
    for k, (mname, lab, col) in enumerate(ms):
        mu = np.array([R[f'{c}|{a}|E-C|{mname}']['mean'] for c, a in arms]); ci = np.array([R[f'{c}|{a}|E-C|{mname}']['ci'] for c, a in arms])
        ax.bar(np.arange(len(arms)) + (k - 2.5) * w, mu, w, color=col, label=lab, yerr=[mu - ci[:, 0], ci[:, 1] - mu], capsize=2)
    for k, (c, a) in enumerate(arms):
        pb = R[f'{c}|{a}|E-C|perm_band']; ax.fill_between([k - 0.45, k + 0.45], pb['min'], pb['max'], color='k', alpha=0.12, lw=0)
    nm = {'Tbl': 't→bℓν', 'Hqql': 'H→ℓνqq'}; dmw = {'below': '+20 GeV,\nstays below edge', 'across': '+20 GeV,\ncrosses edge', 'reverse': '−20 GeV,\nback below edge'}
    ax.set_xticks(range(len(arms))); ax.set_xticklabels([f'{nm[c]}\n{dmw[a]}\n(n={R[f"{c}|{a}|E-C|ParT_full"]["n"]})' for c, a in arms], fontsize=8)
    ax.axhline(0, color='k', lw=0.6); ax.set_ylabel('E − C: change in signal log-odds\n(signal vs QCD + hadronic top)')
    ax.set_title('Lepton opening-angle edit minus azimuthal control (grey band: ' + str(NPN) + ' permuted ParT)', fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc='lower left')
    ax = axs[1]; C = out['score_vs_m']['Tbl']; b = np.array(C['bins']); x = 0.5 * (b[1:] + b[:-1])
    for mname, lab, col in ms[:5]:
        yv = np.array([np.nan if v is None else v for v in C[mname]], float); ax.plot(x, yv, '-o', ms=3, color=col, label=lab)
    ax.axvline(EDGE['Tbl'], color='k', ls='--', lw=0.8); ax.text(EDGE['Tbl'] + 2, ax.get_ylim()[0] + 0.5, 'm_bl endpoint\n152.6 GeV', fontsize=8)
    ax.set_xlabel('jet mass m_vis of unedited t→bℓν jets with a lepton [GeV]'); ax.set_ylabel('mean log p(t→bℓν)/(p(QCD)+p(t→bqq))')
    ax.set_title('Data check: mean score vs jet mass (unedited jets; compare slopes, not levels)', fontsize=10); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(RES / f'{TAG}_response.png', dpi=130); plt.close(fig); log('response figure saved')


if __name__ == '__main__':
    main()
