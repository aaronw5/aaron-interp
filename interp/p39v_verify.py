"""P39v -- skeptic's re-run of P39 (axis dependence on unedited jets) on a fresh sample, with three controls P39 lacked.

Fresh sample: JetClass test_20M <class>_116.root (P39 used _109), numpy default_rng(3995) (P39: 3939), random rows (never first-k):
PER_WZ = 30,000 Wqq and 30,000 Zqq rows; for the mirror test 800 random rows of each of the 8 other classes' _116 files plus 800
random jets of each of the W and Z samples. Jets with < 2 particles dropped. Inputs rebuilt with part_core.build
(eta = jet_eta + sign(jet_eta) deta). Models: released ParT_full, ParT_kin, ParticleNet_full (PN_full), ParticleNet_kin (PN_kin).
Read-out d = l_Z - l_W; local slope s_j = (d(k=1.03) - d(k=1/1.03)) / (2 log 1.03), k = angular dilation about the stored axis.

Re-runs (same definitions as P39):
 (a) natural pairs, P39's primary 'nn' design: top vs bottom quintile of |c| (c = pT centroid about the stored axis) per class,
     exact quartile cells of log pT and N, nearest bottom jet in standardised (log m_inv, log M2c), caliper |dlog m_inv| <= 0.01.
     y = d_top - d_bot, P_A = sbar dlog m_A, P_I = sbar dlog m_inv, R_nat = sum(y - P_I) / sum(P_A - P_I).
 (b) edits on all W/Z jets and on the per-class top 20 % of |c|: rec (shift by -c), iso (shift |c| at +-120 deg from c; M2 fixed),
     rs (|c| random direction), ref (+c), dil_rec (angles x sqrt(M2_rec/M2)).
 (c) phi mirror: mirp (dphi -> -dphi), mirp_q, mirp_d0, mirp_qd0 (charge / d0 signs flipped with the mirror); 10 classes.
New controls:
 perp   shift by |c| at +-90 deg from c (random sign): the momentum direction moves |c| off the stored axis as in rec, but M2 rises
        by |c|^2 M0 (dlog m_A = -dlog m_A(rec) to first order). Predictions of perp/rec: M2 (axis moment) reading -1; 'momentum-
        axis displacement magnitude' +1; 'linear in displacement along c' 0.
 eta_tr the top (off-centre) jet of each natural pair moved to its partner's jet_eta (deta, dphi, pT, masses kept; only the
        particle energies and the stored axis eta change). If P39's natural score gap y comes from |eta_jet| (via the energy
        inputs), y_tr = d(top moved) - d(bottom) collapses toward P_A; if it comes from the centroid offset, y_tr ~ y.
        eta_trb: the bottom jet moved to the top jet's jet_eta (the same test from the other side).
 eta law: second-order expansion of sum_i pT_i sinh(eta_i) = pT_jet sinh(eta_jet) gives c_eta ~ -tanh|eta_jet| M2 / (2 M0) (in the
        stored signed deta coordinates; c_phi is third order). Fraction of variance of c_eta and of |c| explained by it.
Statistics: bootstrap over jets / pairs (NB), sign-flip permutation (NPERM), exact McNemar, Holm within each family.
  ./sync.sh && ./part_job.sh c-p39-verify 'python -m interp.p39v_verify'
  -> $AI_RESULTS/part/p39_verify.json
"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('JETDISTILL_TAGGER', 'part')
SMOKE = '--smoke' in sys.argv
SEED, FIDX = 3995, 116
PER_WZ = 600 if SMOKE else 30000
PER_O = 40 if SMOKE else 800
NB = 200 if SMOKE else 2000
NPERM = 300 if SMOKE else 5000
KDIL = 1.03
TEST = Path('/j-jepa-vol/JetClass/Pythia/test_20M')
PRE = {'QCD': 'ZJetsToNuNu', 'Hbb': 'HToBB', 'Hcc': 'HToCC', 'Hgg': 'HToGG', 'H4q': 'HToWW4Q', 'Hqql': 'HToWW2Q1L', 'Zqq': 'ZToQQ',
       'Wqq': 'WToQQ', 'Tbqq': 'TTBar', 'Tbl': 'TTBarLep'}
CL = ['QCD', 'Hbb', 'Hcc', 'Hgg', 'H4q', 'Hqql', 'Zqq', 'Wqq', 'Tbqq', 'Tbl']
iW, iZ = CL.index('Wqq'), CL.index('Zqq')
B_EDITS = ['base', 'dilp', 'dilm', 'rec', 'iso', 'perp', 'rs', 'ref', 'dil_rec']
C_EDITS = ['mirp', 'mirp_q', 'mirp_d0', 'mirp_qd0']
FLIPS = {'mirp': (), 'mirp_q': ('charge',), 'mirp_d0': ('d0val',), 'mirp_qd0': ('charge', 'd0val')}
RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'
OUT = RES / f'p39_verify{"_smoke" if SMOKE else ""}.json'
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def read_rand(path, n, rng):
    import uproot
    from interp.part_core import read_root, sub, cat
    N = uproot.open(path)['tree'].num_entries; rows = np.sort(rng.choice(N, min(n, N), replace=False)); ch = []
    for s in range(0, N, 25000):
        r = rows[(rows >= s) & (rows < s + 25000)]
        if len(r): ch.append(sub(read_root(path, s, min(s + 25000, N)), r - s))
    J = cat(ch); ok = (J['mask'] > 0).sum(1) >= 2
    return sub(J, np.where(ok)[0]), rows[ok]


def momentum_offset(J):
    m = J['mask'] > 0; px, py, pz = ((J[k].astype(np.float64) * m).sum(1) for k in ('px', 'py', 'pz'))
    eta = np.arcsinh(pz / np.maximum(np.hypot(px, py), 1e-9)); phi = np.arctan2(py, px)
    sg = np.where(J['jet_eta'] < 0, -1.0, 1.0)
    return sg * (eta - J['jet_eta'].astype(np.float64)), np.angle(np.exp(1j * (phi - J['jet_phi'].astype(np.float64))))


def geom(J):
    from interp.part_core import kin, jmass
    pt, de, dp, ms = kin(J); p = pt.astype(np.float64); de64, dp64 = de.astype(np.float64), dp.astype(np.float64)
    M0 = p.sum(1); M2 = (p * (de64 ** 2 + dp64 ** 2)).sum(1); M2e = (p * de64 ** 2).sum(1)
    ce, cp = (p * de64).sum(1) / M0, (p * dp64).sum(1) / M0; oe, op = momentum_offset(J)
    return dict(ce=ce, cp=cp, cabs=np.hypot(ce, cp), M0=M0, M2=M2, M2e=M2e, M2c=M2 - (ce ** 2 + cp ** 2) * M0, minv=jmass(J).astype(np.float64),
                ptj=J['jet_pt'].astype(np.float64), npart=(J['mask'] > 0).sum(1).astype(np.int32), etaj=J['jet_eta'].astype(np.float64),
                momoff=np.hypot(oe, op))


def make_edit(J, K, G, Sh, name):
    from interp.part_core import build, PK
    pt, de, dp, ms = K; r = (pt > 0).astype(np.float32); col = lambda a: a[:, None].astype(np.float32)
    if name == 'base': return build(J, pt, de, dp, ms)
    if name == 'dilp': return build(J, pt, de * KDIL, dp * KDIL, ms)
    if name == 'dilm': return build(J, pt, de / KDIL, dp / KDIL, ms)
    if name == 'rec': return build(J, pt, de - col(G['ce']) * r, dp - col(G['cp']) * r, ms)
    if name == 'ref': return build(J, pt, de + col(G['ce']) * r, dp + col(G['cp']) * r, ms)
    if name in ('iso', 'perp', 'rs'):
        a, b = Sh[name]; return build(J, pt, de + col(a) * r, dp + col(b) * r, ms)
    if name == 'dil_rec':
        k = col(np.sqrt(np.maximum(G['M2c'], 1e-30) / G['M2'])); return build(J, pt, de * k, dp * k, ms)
    if name in FLIPS:
        ex = {k: J[k] for k in PK}
        for k in FLIPS[name]: ex[k] = -J[k]
        return build(J, pt, de, -dp, ms, extra=ex)
    raise ValueError(name)


def shifts(G, rng):
    n = len(G['ce']); ce, cp, ca = G['ce'], G['cp'], G['cabs']; S = {}
    for nm, ang in (('iso', 2 * np.pi / 3), ('perp', np.pi / 2)):
        th = rng.choice([-1.0, 1.0], n) * ang; S[nm] = (np.cos(th) * ce - np.sin(th) * cp, np.sin(th) * ce + np.cos(th) * cp)
    al = rng.uniform(0, 2 * np.pi, n); S['rs'] = (ca * np.cos(al), ca * np.sin(al))
    return S


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


# ------------------------------------------------------------------------------------------------ statistics
def ci(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) else [None, None]


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.zeros(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj.tolist()


def signflip_p(x, rng, nperm=None):
    nperm = nperm or NPERM; x = np.asarray(x, float); t = abs(x.mean()); n = len(x); cnt = 0; done = 0
    while done < nperm:
        b = min(200, nperm - done); sg = rng.choice([-1.0, 1.0], size=(b, n)); cnt += int((np.abs(sg @ x / n) >= t).sum()); done += b
    return float((1 + cnt) / (1 + nperm))


def mcnemar(a, b):
    from scipy.stats import binomtest
    x = int((a & ~b).sum()); z = int((~a & b).sum())
    return dict(only_first=x, only_second=z, p=(float(binomtest(x, x + z, 0.5).pvalue) if x + z else 1.0))


def match_nn(G, y, caliper=0.01):
    from scipy.spatial import cKDTree
    I, B = [], []
    for k in (iW, iZ):
        ik = np.where(y == k)[0]; c = G['cabs'][ik]; hi, lo = np.quantile(c, 0.8), np.quantile(c, 0.2)
        top, bot = ik[c >= hi], ik[c <= lo]; both = np.concatenate([top, bot])
        lm, l2 = np.log(np.maximum(G['minv'][both], 1e-3)), np.log(np.maximum(G['M2c'][both], 1e-12))
        V = np.stack([np.log(G['ptj'][both]), G['npart'][both].astype(float)], 1); cells = np.zeros(len(both), np.int64)
        for j in range(V.shape[1]):
            e = np.quantile(V[:, j], [0.25, 0.5, 0.75]); cells = cells * 4 + np.searchsorted(e, V[:, j], side='right')
        ct, cb = cells[:len(top)], cells[len(top):]; F = np.stack([lm / lm.std(), l2 / l2.std()], 1); Ft, Fb = F[:len(top)], F[len(top):]
        for cc in np.unique(ct):
            tt = np.where(ct == cc)[0]; bb = np.where(cb == cc)[0]
            if not len(bb): continue
            _, q = cKDTree(Fb[bb]).query(Ft[tt], k=1); pick = bb[q]; keep = np.abs(lm[tt] - lm[len(top) + pick]) <= caliper
            I += list(top[tt[keep]]); B += list(bot[pick[keep]])
    return np.array(I, int), np.array(B, int)


# ------------------------------------------------------------------------------------------------ main
def main():
    import torch
    from interp.part_core import sample, kin, build, sub, cat
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; log('device', dev)
    M, FEATURES = load_nets(dev); names = list(M); run = runner(M, FEATURES, dev)
    J1 = sample(100 if SMOKE else 1000, 3000); y1 = J1['label'].astype(int); L1 = run(build(J1, *kin(J1)), names)
    R = dict(node='P39v', seed=SEED, file=f'test_20M <class>_{FIDX}.root', per_wz=PER_WZ, per_other=PER_O, nb=NB, nperm=NPERM,
             readout='d = logit_Z - logit_W', acc_P1={k: float((L1[k].argmax(1) == y1).mean()) for k in names}, n_P1=int(len(y1)))
    log('P1 accuracy', R['acc_P1']); del J1, L1
    rng = np.random.default_rng(SEED)
    # ---- W/Z sample: base, dilations, edits
    Js, Gs, Ls, ys, Ks, Shs = [], [], [], [], [], []
    for c in ('Wqq', 'Zqq'):
        J, rows = read_rand(TEST / f'{PRE[c]}_{FIDX}.root', PER_WZ, rng); assert np.all(J['label'] == CL.index(c))
        K = kin(J); G = geom(J); Sh = shifts(G, rng); L = {k: np.zeros((len(B_EDITS), len(rows), 10), np.float32) for k in names}; gE = {}
        for ie, e in enumerate(B_EDITS):
            Je = make_edit(J, K, G, Sh, e); ge = geom(Je)
            if e == 'base': G['minv'] = ge['minv']; Gb = ge
            gE[e] = dict(dlogmA=0.5 * np.log(ge['M2'] / Gb['M2']), dlogminv=np.log(np.maximum(ge['minv'], 1e-3) / np.maximum(Gb['minv'], 1e-3)),
                         momoff=ge['momoff'], cabs=ge['cabs'])
            Lr = run(Je, names)
            for k in names: L[k][ie] = Lr[k]
            del Je
        log(f'{c}: {len(rows)} jets; recall ' + str({k: round(float((L[k][0].argmax(1) == CL.index(c)).mean()), 4) for k in names}))
        G['gE'] = gE; Js.append(J); Gs.append(G); Ls.append(L); ys.append(np.full(len(rows), CL.index(c))); Ks.append(K); Shs.append(Sh)
    y = np.concatenate(ys); n = len(y)
    G = {k: np.concatenate([g[k] for g in Gs]) for k in Gs[0] if k != 'gE'}
    gE = {e: {q: np.concatenate([g['gE'][e][q] for g in Gs]) for q in Gs[0]['gE'][e]} for e in B_EDITS}
    L = {k: np.concatenate([l[k] for l in Ls], 1) for k in names}
    Jall = cat(Js); Kall = tuple(np.concatenate([k_[i] for k_ in Ks]) for i in range(4)); del Js
    E = {e: i for i, e in enumerate(B_EDITS)}; dfun = lambda A: (A[..., iZ] - A[..., iW]).astype(np.float64)
    # ---- eta law for the natural offset
    sgn_ce_pred = -np.tanh(np.abs(G['etaj'])) * G['M2'] / (2 * G['M0'])
    r2 = lambda a, b: float(1 - np.var(a - b) / np.var(a))
    R['eta_law'] = dict(formula='c_eta ~ -tanh|eta_jet| M2/(2 M0); c_phi third order', r2_ceta=r2(G['ce'], sgn_ce_pred),
                        r2_cabs=r2(G['cabs'], np.abs(sgn_ce_pred)), slope_ceta=float(np.polyfit(sgn_ce_pred, G['ce'], 1)[0]),
                        median_cabs=float(np.median(G['cabs'])), median_abs_ceta=float(np.median(np.abs(G['ce']))), median_abs_cphi=float(np.median(np.abs(G['cp']))),
                        corr_cabs_tanh_abseta=float(np.corrcoef(G['cabs'], np.tanh(np.abs(G['etaj'])))[0, 1]),
                        median_momoff=float(np.median(G['momoff'])), offset_term_median=float(np.median(0.5 * np.log1p(G['cabs'] ** 2 * G['M0'] / G['M2c']))))
    log('eta law', R['eta_law'])
    # ---- (a) natural pairs + eta transport
    I, Bp = match_nn(G, y); npair = len(I); log('pairs', npair)
    Jt = sub(Jall, I); Kt = tuple(k_[I] for k_ in Kall); Jt['jet_eta'] = Jall['jet_eta'][Bp].copy()
    Jb = sub(Jall, Bp); Kb = tuple(k_[Bp] for k_ in Kall); Jb['jet_eta'] = Jall['jet_eta'][I].copy()
    Jt_e = build(Jt, *Kt); Jb_e = build(Jb, *Kb); Lt = run(Jt_e, names); Lb = run(Jb_e, names)
    R['eta_tr_geometry'] = dict(momoff_top_after_median=float(np.median(geom(Jt_e)['momoff'])), momoff_top_before_median=float(np.median(G['momoff'][I])))
    del Jt, Jb, Jt_e, Jb_e
    lmi = np.log(np.maximum(G['minv'], 1e-3)); lmA = 0.5 * np.log(G['M0'] * G['M2'])
    R['a_natural'] = {}; fam_a = []
    rng2 = np.random.default_rng(SEED + 1)
    for k in names:
        d = dfun(L[k][E['base']]); s = (dfun(L[k][E['dilp']]) - dfun(L[k][E['dilm']])) / (2 * np.log(KDIL))
        yv = d[I] - d[Bp]; sb = 0.5 * (s[I] + s[Bp]); PI = sb * (lmi[I] - lmi[Bp]); PA = sb * (lmA[I] - lmA[Bp])
        ytr = dfun(Lt[k]) - d[Bp]; ytrb = d[I] - dfun(Lb[k])
        bs = [rng2.integers(0, npair, npair, dtype=np.int32) for _ in range(NB)]
        num, den = yv - PI, PA - PI
        r = dict(n_pairs=npair, y=float(yv.mean()), y_ci=ci([yv[b].mean() for b in bs]), P_A=float(PA.mean()), P_I=float(PI.mean()),
                 R_nat=float(num.sum() / den.sum()), R_nat_ci=ci([num[b].sum() / den[b].sum() for b in bs]),
                 y_eta_tr_top=float(ytr.mean()), y_eta_tr_top_ci=ci([ytr[b].mean() for b in bs]),
                 y_eta_tr_bot=float(ytrb.mean()), y_eta_tr_bot_ci=ci([ytrb[b].mean() for b in bs]),
                 frac_gap_removed_top=float(1 - ytr.mean() / yv.mean()), frac_gap_removed_top_ci=ci([1 - ytr[b].mean() / yv[b].mean() for b in bs]),
                 R_nat_after_eta_tr=float((ytr - PI).sum() / den.sum()), R_nat_after_eta_tr_ci=ci([(ytr - PI)[b].sum() / den[b].sum() for b in bs]),
                 p_y_eq_0=signflip_p(yv, rng2), p_ytr_eq_y=signflip_p(ytr - yv, rng2), p_ytr_eq_PA=signflip_p(ytr - PA, rng2),
                 balance=dict(abs_eta_smd=float((np.abs(G['etaj'][I]) - np.abs(G['etaj'][Bp])).mean() / np.sqrt(0.5 * (np.abs(G['etaj'][I]).var() + np.abs(G['etaj'][Bp]).var()))),
                              dlog_minv=float((lmi[I] - lmi[Bp]).mean()), dcabs=float((G['cabs'][I] - G['cabs'][Bp]).mean())))
        R['a_natural'][k] = r; fam_a += [(k, 'p_y_eq_0', r['p_y_eq_0']), (k, 'p_ytr_eq_y', r['p_ytr_eq_y'])]
        log(f'(a) {k}: pairs {npair} y {r["y"]:+.4f} {np.round(r["y_ci"], 4)} P_A {r["P_A"]:+.4f} R_nat {r["R_nat"]:+.2f} {np.round(r["R_nat_ci"], 2)}; '
            f'eta-transported top: y {r["y_eta_tr_top"]:+.4f} {np.round(r["y_eta_tr_top_ci"], 4)} (removed {r["frac_gap_removed_top"]:.2f}); bottom: {r["y_eta_tr_bot"]:+.4f}; '
            f'R_nat after {r["R_nat_after_eta_tr"]:+.2f} {np.round(r["R_nat_after_eta_tr_ci"], 2)}; |eta| smd {r["balance"]["abs_eta_smd"]:+.2f}')
    for (k, nm, p), ph in zip(fam_a, holm([f[2] for f in fam_a])): R['a_natural'][k][nm + '_holm'] = ph
    # ---- (b) edits
    q80 = {kk: np.quantile(G['cabs'][y == kk], 0.8) for kk in (iW, iZ)}; top = np.array([G['cabs'][i] >= q80[y[i]] for i in range(n)])
    sets = dict(all=np.ones(n, bool), top20=top); R['b_geometry'] = {}
    for e in B_EDITS[3:]:
        R['b_geometry'][e] = {s_: dict(momoff_median=float(np.median(gE[e]['momoff'][f])), cabs_median=float(np.median(gE[e]['cabs'][f])),
                                      dlogmA_mean=float(gE[e]['dlogmA'][f].mean()), dlogminv_mean=float(gE[e]['dlogminv'][f].mean())) for s_, f in sets.items()}
    log('(b) geometry', json.dumps(R['b_geometry']))
    R['b_edits'] = {}; fam_b = []
    for k in names:
        d0 = dfun(L[k][E['base']]); sl = (dfun(L[k][E['dilp']]) - dfun(L[k][E['dilm']])) / (2 * np.log(KDIL)); R['b_edits'][k] = {}
        for s_, f in sets.items():
            ix = np.where(f)[0]; m_ = len(ix); bs = [rng2.integers(0, m_, m_, dtype=np.int32) for _ in range(NB)]; rr = dict(n=m_)
            dd = {e: dfun(L[k][E[e]])[ix] - d0[ix] for e in B_EDITS[3:]}; pa = {e: sl[ix] * gE[e]['dlogmA'][ix] for e in B_EDITS[3:]}
            c0 = L[k][E['base']][ix].argmax(1)
            for e in B_EDITS[3:]:
                rr[e] = dict(d=float(dd[e].mean()), d_ci=ci([dd[e][b].mean() for b in bs]), pred_mA=float(pa[e].mean()),
                             R=float(dd[e].mean() / pa[e].mean()) if abs(pa[e].mean()) > 1e-4 else None,
                             R_ci=ci([dd[e][b].mean() / pa[e][b].mean() for b in bs]) if abs(pa[e].mean()) > 1e-4 else None,
                             calls_changed=float((L[k][E[e]][ix].argmax(1) != c0).mean()))
            for e in ('iso', 'perp', 'rs', 'ref'):
                rr[f'{e}_over_rec'] = dict(v=float(dd[e].mean() / dd['rec'].mean()), ci=ci([dd[e][b].mean() / dd['rec'][b].mean() for b in bs]))
            rr['p_iso_eq_0'] = signflip_p(dd['iso'], rng2); rr['p_iso_eq_rec'] = signflip_p(dd['iso'] - dd['rec'], rng2)
            rr['p_iso_eq_half_rec'] = signflip_p(dd['iso'] - 0.5 * dd['rec'], rng2)
            rr['p_perp_eq_rec'] = signflip_p(dd['perp'] - dd['rec'], rng2); rr['p_perp_eq_0'] = signflip_p(dd['perp'], rng2)
            rr['p_perp_eq_minus_rec'] = signflip_p(dd['perp'] + dd['rec'], rng2)
            # joint fit over rec, iso, perp, ref per jet: d_e = a * P_A(e) + b * s * u_e, u_e = signed momentum displacement along -c / |c|
            u = {'rec': 1.0, 'iso': 0.5, 'perp': 0.0, 'ref': -1.0}; X, Y = [], []
            for e in u:
                X.append(np.stack([pa[e], sl[ix] * u[e] * G['cabs'][ix]], 1)); Y.append(dd[e])
            X, Y = np.concatenate(X), np.concatenate(Y); beta = np.linalg.lstsq(X, Y, rcond=None)[0]
            bb = []
            for b in bs[:500]:
                bi = np.concatenate([b + j * m_ for j in range(4)]); bb.append(np.linalg.lstsq(X[bi], Y[bi], rcond=None)[0])
            bb = np.array(bb); rr['joint_fit'] = dict(a_mA=float(beta[0]), a_ci=ci(bb[:, 0]), b_disp_per_rad=float(beta[1]), b_ci=ci(bb[:, 1]))
            R['b_edits'][k][s_] = rr
            if s_ == 'top20': fam_b += [(k, nm, rr[nm]) for nm in ('p_iso_eq_rec', 'p_iso_eq_half_rec', 'p_perp_eq_rec', 'p_perp_eq_0')]
            log(f'(b) {k} {s_} n={m_}: ' + '; '.join(f'{e} d {rr[e]["d"]:+.4f} (mA {rr[e]["pred_mA"]:+.4f})' for e in B_EDITS[3:])
                + f'; iso/rec {rr["iso_over_rec"]["v"]:+.2f} {np.round(rr["iso_over_rec"]["ci"], 2)}; perp/rec {rr["perp_over_rec"]["v"]:+.2f} {np.round(rr["perp_over_rec"]["ci"], 2)}'
                + f'; joint a {beta[0]:+.2f} {np.round(rr["joint_fit"]["a_ci"], 2)} b {beta[1]:+.2f}')
    for (k, nm, p), ph in zip(fam_b, holm([f[2] for f in fam_b])): R['b_edits'][k]['top20'][nm + '_holm'] = ph
    # ---- (c) mirror on 10 classes
    Jc, yc = [], []
    for c in CL:
        if c in ('Wqq', 'Zqq'):
            ii = np.where(y == CL.index(c))[0]; pick = np.sort(rng.choice(ii, min(PER_O, len(ii)), replace=False)); Jc.append(sub(Jall, pick))
        else:
            J, _ = read_rand(TEST / f'{PRE[c]}_{FIDX}.root', PER_O, rng); Jc.append(J)
        yc.append(np.full(len(Jc[-1]['mask']), CL.index(c)))
    Jc = cat(Jc); yc = np.concatenate(yc); Kc = kin(Jc); del Jall
    Lc = {e: run(make_edit(Jc, Kc, None, None, e), names) for e in ['base'] + C_EDITS}
    w = 1.0 / np.bincount(yc, minlength=10)[yc]; bsC = [rng2.integers(0, len(yc), len(yc)) for _ in range(min(NB, 1000))]
    R['c_mirror'] = {}; fam_c = []
    for k in names:
        c0 = Lc['base'][k].argmax(1); ch = {e: Lc[e][k].argmax(1) != c0 for e in C_EDITS}; r = dict(acc_base_balanced=float(np.mean([(c0[yc == j] == j).mean() for j in range(10)])))
        for e in C_EDITS:
            r[e] = dict(changed_balanced=float((ch[e] * w).sum() / w.sum()), ci=ci([(ch[e][b] * w[b]).sum() / w[b].sum() for b in bsC]))
            if e != 'mirp' and k.endswith('full'): r[e]['mcnemar_vs_mirp'] = mcnemar(ch['mirp'], ch[e]); fam_c.append((k, e, r[e]['mcnemar_vs_mirp']['p']))
        R['c_mirror'][k] = r; log(f'(c) {k}: ' + ' '.join(f'{e} {100 * r[e]["changed_balanced"]:.2f}%' for e in C_EDITS))
    for (k, e, p), ph in zip(fam_c, holm([f[2] for f in fam_c])): R['c_mirror'][k][e]['mcnemar_vs_mirp']['p_holm'] = ph
    R['N_mirror'] = int(len(yc))
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(R, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    log('wrote', OUT)


if __name__ == '__main__':
    main()
