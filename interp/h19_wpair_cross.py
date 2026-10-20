"""H19 (loop D; claims D6 vs D1 / H3v, open question c): W-candidate pair mass inside top jets, JetClass-trained vs
Top-Landscape-trained models, on matched jets and the same edit.

Question. H4 / H4v: Top Landscape ParT / PN / LorentzNet respond to the W-pair mass at fixed m and tau32 at 2-4x a
side-effect reference (MLPx). H3v: JetClass ParT shows no W-pair response beyond a W-fixed control. The two used different
edits, controls and jets. Here the H4 edit and references are applied to both samples and both model families.
  H1 (edit / sample difference): with H4's edit, JetClass ParT / PN also respond beyond MLPx (2-4x MLPx).
  H0 (training difference): JetClass models stay at the MLPx level, Top-trained models respond at the MLPxW level on the same
      JetClass jets. If Top models also lose the response on JetClass jets, the sample (pT window / detector) is the factor.

Samples (random rows, never first-k):
  jc   JetClass test files TTBar_100.root (Tbqq) and ZJetsToNuNu_100.root (QCD), rows with stored 550 < jet_pt < 650 GeV and
       |jet_eta| < 2, 5,000 per class drawn with default_rng(4191) among the eligible rows.
  top  Top Landscape test split, 5,000 top + up to 5,000 QCD with default_rng(4192), excluding the H4 rows (seeds 601 / 602)
       and the H4v rows (seeds 611 / 612).
Edit (unchanged from interp.h4_top_wmass via interp.h4_verify.edit_set): three exclusive-kT subjets; W pair = the pair whose
  mass is closest to 80.4 GeV; subjets moved rigidly (pT, constituent masses kept) so that the W-pair mass -> t at fixed jet mass
  and tau32 (|dm| < 0.5 GeV, |dtau32| < 0.02, |mW - t| < tol). Sets: T60, T100 (tops with 65 < mW < 95, tol 5), T80 (tops with
  6 < |mW - 80.4| < 15 moved onto 80.4, tol 2), Q80 (QCD in 105 < mSD < 210 with mW in 50-70 or 90-115 -> 80.4, tol 5).
  Controls on the same jets: N (non-W pair given the same relative mass change, W pair fixed, m and tau32 fixed), R (rotation
  about the jet axis with the same pT-weighted mean displacement). Analysis rows: W and N both accepted.
Models
  JetClass (released weights): ParT_full, ParT_kin, PN_full, PN_kin; score = L(Tbqq) - L(QCD) of the 10-class logits =
  log p(Tbqq)/p(QCD). On top jets only the kin models are run (Top Landscape has no track / PID inputs): the Top four-vectors
  (128 hardest) are rebuilt as a JetClass jet dict about the constituent-sum axis with all non-kinematic features zero.
  Top Landscape: ParT_kin, PN_kin (H4 checkpoint snapshots), LorentzNet (released), PFN, EFN (H2); inputs are the
  four-vectors only, so JetClass jets are passed as their (<= 128) particle four-vectors.
  References, retrained per dataset (training jets disjoint from the samples): LR2 = binned LR(mSD, tau32) (negative control);
  MLPx = MLP(64, 64) on mSD, tau1-3, tau32, tau21, multiplicity, sorted subjet pT fractions, sorted subjet masses (side effects of
  the edit, no pair mass); MLPxW = MLPx + mW. jc refs: TTBar_101 / ZJetsToNuNu_101 in-window rows (up to 25,000 per class,
  rng 4193); top refs: Top val split, 25,000 per class (rng 4194). Every reference is evaluated on both samples.
Metrics per sample x set x model: d_k = s_k - s_id (k = W, N, R; log-odds), W - N mean with 95 % bootstrap (B = 2000 over jets);
  sign-flip permutation p (H0: W - N = 0; one-sided in the mW-window direction: T60 / T100 down, T80 / Q80 up);
  ratio (W - N)_net / (W - N)_ref for ref = MLPx and MLPxW trained on the model's training dataset, with bootstrap CI;
  H0_side: (W - N)_net = (W - N)_MLPx per jet (sign-flip permutation, one-sided: |net| larger in the mW direction);
  Holm over networks x samples x sets for each family; McNemar W vs N calls at eps_S 0.3 / 0.5 (thresholds from the sample's
  unedited tops); background rejection 1/eps_B at those thresholds on the sample's QCD jets.
Checks: unedited AUC, binary accuracy (score > 0), 10-class accuracy (JetClass nets); input-path check (JetClass id rebuild vs
  read_root jets); C2ST between the samples (HGB on the 17 substructure features, 2-fold AUC) for top and QCD jets.
Run: ./part_job.sh d-h19-run 'python -m interp.h19_wpair_cross'   (H19_SMOKE=1: 200 per class, 2,000 ref jets, h19_smoke*)
Outputs: $AI_RESULTS/physics/h19_wpair_cross.json, h19_perjet.npz, h19_wpair.png
"""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import h4_top_wmass as H          # noqa: E402
from interp import h4_verify as HV            # noqa: E402
from interp import data_top as dt             # noqa: E402

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
SMOKE = bool(os.environ.get('H19_SMOKE'))
RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'physics'
TAG = 'h19_smoke' if SMOKE else 'h19'
PER = 200 if SMOKE else 5000
NREF = 2000 if SMOKE else 25000
B = 500 if SMOKE else 2000
NPERM = 500 if SMOKE else 2000
L_JC = 128
DIRECTION = {'T60': -1, 'T100': -1, 'T80': 1, 'Q80': 1}
TBQQ, QCD = 8, 0
NETS_JC = ['ParT_full', 'ParT_kin', 'PN_full', 'PN_kin']
NETS_TOP = ['ParT_top', 'PN_top', 'LNet_top', 'PFN_top', 'EFN_top']
REFS = ['LR2', 'MLPx', 'MLPxW']


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ======================================================================================================= JetClass data
def jc_file(c, k):
    from interp.part_core import files, PRE, DATA
    f = next(x['file'] for x in files() if x['file'].split('/')[-1] == f'{PRE[c]}_{k}.root')
    for p in (Path(f), DATA.parent / f, DATA / 'raw' / 'test' / Path(f).name):
        if p.exists(): return p
    raise FileNotFoundError(f)


def jc_read(c, k, per, seed):
    """per random rows of <class>_<k>.root among 550 < jet_pt < 650, |jet_eta| < 2 (stored jet), >= 3 particles"""
    import uproot
    from interp.part_core import read_root, cat, sub
    f = jc_file(c, k); t = uproot.open(f)['tree']
    a = t.arrays(['jet_pt', 'jet_eta', 'jet_nparticles'], library='np')
    ok = (a['jet_pt'] > 550) & (a['jet_pt'] < 650) & (np.abs(a['jet_eta']) < 2) & (a['jet_nparticles'] >= 3)
    el = np.flatnonzero(ok); n = min(per, len(el))
    idx = np.sort(np.random.default_rng(seed).choice(el, n, replace=False)); parts = []; N = t.num_entries
    for s in range(0, N, 20000):
        loc = idx[(idx >= s) & (idx < s + 20000)] - s
        if len(loc): parts.append(sub(read_root(f, start=s, stop=min(s + 20000, N)), loc))
    J = cat(parts); log(f'jc {c} file {f.name}: {len(el)} eligible of {N}; drew {n} (rng {seed})')
    return J, idx, dict(file=f.name, eligible=int(len(el)), rows_in_file=int(N), drawn=int(n))


def jc_P4(J):
    m = J['mask'] > 0
    P4 = np.stack([J['px'], J['py'], J['pz'], J['energy']], -1) * m[..., None]
    return P4.astype(np.float32), m.sum(1).astype(np.int64)


def jc_dict(base, P4, nn):
    """JetClass jet dict from particle four-vectors (same particle order as base, pT unchanged by the edits) about base's stored
    axis: eta = jet_eta + sign(jet_eta) deta (part_core.build)"""
    from interp.part_core import build
    P = P4.astype(np.float64); L = P.shape[1]; real = (np.arange(L)[None, :] < nn[:, None])
    pt = np.hypot(P[..., 0], P[..., 1]); real &= pt > 0; ptc = np.where(real, pt, 1.0)
    eta = np.arcsinh(np.where(real, P[..., 2], 0) / ptc); phi = np.arctan2(P[..., 1], P[..., 0])
    m = base['pmass'][:, :L].astype(np.float64)   # constituent masses kept by every edit: taken exactly from the stored particles
    je = base['jet_eta'].astype(np.float64)[:, None]; jp = base['jet_phi'].astype(np.float64)[:, None]; sg = np.where(je < 0, -1.0, 1.0)
    deta = np.where(real, sg * (eta - je), 0); dphi = np.where(real, wrap(phi - jp), 0)
    return build(base, np.where(real, pt, 0), deta, dphi, np.where(real, m, 0))


def top_base(P4, nn):
    """JetClass-style base dict for Top Landscape jets: axis = constituent sum, non-kinematic particle features zero"""
    from interp.part_core import PK
    P = P4.astype(np.float64); L = P.shape[1]; real = np.arange(L)[None, :] < nn[:, None]
    S = (P * real[..., None]).sum(1); jpt = np.hypot(S[:, 0], S[:, 1])
    b = {k: np.zeros(P.shape[:2], np.float32) for k in PK}
    b.update(jet_eta=np.arcsinh(S[:, 2] / jpt).astype(np.float32), jet_phi=np.arctan2(S[:, 1], S[:, 0]).astype(np.float32),
             label=np.zeros(len(nn), np.int8), pmass=np.zeros(P.shape[:2], np.float32))   # Top Landscape constituents are massless
    return b


def subd(Jd, rows):
    return {k: (v[rows] if np.ndim(v) else v) for k, v in Jd.items()}


# ======================================================================================================= models
class Models:
    def __init__(self):
        import torch
        from interp.part_core import ParTNetwork
        from interp.pn_net import load_particlenet
        from jetdistill.config import MODELS
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'; self.dev = dev
        self.jc = {}
        for n in ('full', 'kin'):
            self.jc[f'ParT_{n}'] = ('part', n, ParTNetwork(n))
        for n, dims in (('full', 17), ('kin', 7)):
            self.jc[f'PN_{n}'] = ('pn', n, load_particlenet(Path(MODELS).parent / 'particlenet' / f'ParticleNet_{n}.pt', dims, dev))
        self.top = {'ParT_top': H.WeaverNet('ParT', RES / 'h4_ckpt_part_kin.pt', dev), 'PN_top': H.WeaverNet('PN', RES / 'h4_ckpt_pnet_kin.pt', dev),
                    'LNet_top': H.LNet(dev, mass64=True)}
        self.flows = {'PFN_top': H.Flow('pfn'), 'EFN_top': H.Flow('efn')}; self.massless = True
        log(f'models loaded on {dev}')

    def jc_logits(self, name, J):
        from interp.part_core import inputs
        from interp.p1v_verify_part import fwd
        from interp.h1_irc_panel import pn_logits
        kind, n, net = self.jc[name]
        return fwd(net, *inputs(J, n))[1] if kind == 'part' else pn_logits(net, J, n)

    def score(self, ds, P4, nn, base, order, want_logits=False):
        """all models applicable to sample ds on the jets (P4, nn); base = the JetClass-style dict of the same rows"""
        r = {}; lg = {}
        Jd = jc_dict(base, P4, nn)
        for name in self.jc:
            if ds == 'top' and name.endswith('full'): continue
            L = self.jc_logits(name, Jd); r[name] = L[:, TBQQ] - L[:, QCD]; lg[name] = L
        if ds == 'jc' and self.massless:   # Top Landscape constituents are massless (checked in main): E = |p| for the Top models
            P4 = P4.copy(); P4[..., 3] = np.sqrt((P4[..., :3].astype(np.float64) ** 2).sum(-1)) * (P4[..., 3] != 0)
        for name, net in self.top.items():
            r[name] = net.score(P4, nn, order) if name == 'LNet_top' else net.score(P4, nn)
        k3 = np.concatenate([H.kin3_of(P4[s:s + 5000], nn[s:s + 5000]) for s in range(0, len(nn), 5000)])
        for name, f in self.flows.items(): r[name] = f.score(k3)
        return (r, lg) if want_logits else r


def xsel(X, O, withW):
    c = [X[:, 0], X[:, 1], X[:, 2], X[:, 3], O['tau32'], X[:, 4], X[:, 7], X[:, 11], X[:, 12], X[:, 13], X[:, 14], X[:, 15], X[:, 16]]
    return np.column_stack(c + ([O['mW']] if withW else []))


E2 = [np.linspace(0, 300, 31), np.linspace(0, 1, 26)]
lo = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))


def fit_refs(X, O, y, tag):
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    R = {'LR2': H._lr_fit(np.stack([O['mSD'], O['tau32']], 1), y, E2)}; info = {}
    for nm, w in (('MLPx', False), ('MLPxW', True)):
        Xf = xsel(X, O, w); sc = StandardScaler().fit(Xf)
        m = MLPClassifier(hidden_layer_sizes=(64, 64), early_stopping=True, random_state=0, max_iter=200).fit(sc.transform(Xf), y)
        R[nm] = (m, sc, w); info[nm] = dict(iterations=int(m.n_iter_), val_acc=float(m.best_validation_score_))
        log(f'ref {nm}_{tag}: {m.n_iter_} iterations, internal val acc {m.best_validation_score_:.4f} on {len(y)} jets')
    return R, info


def eval_refs(R, X, O):
    r = {'LR2': H._lr_eval(R['LR2'], np.stack([O['mSD'], O['tau32']], 1), E2)}
    for nm in ('MLPx', 'MLPxW'):
        m, sc, w = R[nm]; r[nm] = lo(m.predict_proba(sc.transform(xsel(X, O, w)))[:, 1])
    return r


def all_refs(REF, X, O):
    out = {}
    for tr, R in REF.items():
        for k, v in eval_refs(R, X, O).items(): out[f'{k}_{tr}'] = v
    return out


# ======================================================================================================= samples
def load_jc():
    Jt, it, it_info = jc_read('Tbqq', 100, PER, 4191); Jq, iq, iq_info = jc_read('QCD', 100, PER, 4191 + 1)
    from interp.part_core import cat
    J = cat([Jt, Jq]); P4raw, nn = jc_P4(J); P4 = H._identity(P4raw, nn).astype(np.float32)
    y = (J['label'] == TBQQ).astype(int)
    from interp.part_core import kin
    J['pmass'] = kin(J)[3].astype(np.float32)
    return dict(J=J, P4=P4, nn=nn, y=y, base=J, info=dict(Tbqq=it_info, QCD=iq_info, label_check=dict(Tbqq=int((Jt['label'] == TBQQ).sum()), QCD=int((Jq['label'] == QCD).sum()))))


def load_top():
    yt = np.load(dt.DST / 'test' / 'label.npy'); rng0 = np.random.default_rng(601)
    oldT = np.concatenate([rng0.choice(np.flatnonzero(yt == c), 5000, replace=False) for c in (0, 1)])
    oldQ = np.random.default_rng(602).choice(np.flatnonzero(yt == 0), 100000, replace=False)
    excl = np.concatenate([oldT, oldQ])
    idxTv, _ = HV.fresh('test', 5000, 611, (0, 1), excl); idxQv, _ = HV.fresh('test', 90000, 612, (0,), np.concatenate([excl, idxTv]))
    ok = np.ones(len(yt), bool); ok[np.concatenate([excl, idxTv, idxQv])] = False
    rng = np.random.default_rng(4192); idx = []; avail = {}
    for c in (1, 0):
        el = np.flatnonzero((yt == c) & ok); avail[int(c)] = int(len(el)); idx.append(rng.choice(el, min(PER, len(el)), replace=False))
    idx = np.sort(np.concatenate(idx)); full = H._full_test(); jet = np.load(dt.DST / 'test' / 'jet.npy')
    nn = np.minimum(jet[idx, 5].astype(np.int64), full.shape[1]); P4 = H._identity(full[idx][..., :4].astype(np.float32), nn).astype(np.float32)
    y = yt[idx].astype(int); log(f'top sample: {len(idx)} rows ({int(y.sum())} top), fresh rows available per class {avail}')
    return dict(P4=P4, nn=nn, y=y, base=top_base(P4, nn), idx=idx, info=dict(fresh_available=avail, n=int(len(idx)), n_top=int(y.sum())))


def ref_train_jc():
    Xs, Os, ys = [], [], []
    for c, lab in (('Tbqq', 1), ('QCD', 0)):
        J, _, inf = jc_read(c, 101, NREF, 4193 + lab); P4, nn = jc_P4(J); X, O = H._features(P4, nn); Xs.append(X); Os.append(O); ys.append(np.full(len(nn), lab))
    O = {k: np.concatenate([o[k] for o in Os]) for k in Os[0]}
    return np.concatenate(Xs), O, np.concatenate(ys)


def ref_train_top():
    yv = np.load(dt.DST / 'val' / 'label.npy'); rv = np.random.default_rng(4194)
    iv = np.sort(np.concatenate([rv.choice(np.flatnonzero(yv == c), NREF, replace=False) for c in (0, 1)]))
    t = time.time(); fv = np.load(dt.DST / 'val' / 'full.npy'); Pv = np.asarray(fv[iv][..., :4], np.float32); del fv   # sequential read (random memmap rows are slow on CephFS)
    nv = np.minimum(np.load(dt.DST / 'val' / 'jet.npy')[iv, 5].astype(np.int64), Pv.shape[1]); log(f'top val rows read in {time.time() - t:.0f}s')
    X, O = H._features(Pv, nv); return X, O, yv[iv].astype(int)


# ======================================================================================================= stats helpers
def perm_p(x, sign, rng, n=NPERM):
    """sign-flip permutation p for H0 mean(x) = 0, one-sided toward sign * mean > 0"""
    x = np.asarray(x, float); obs = sign * x.mean()
    s = rng.choice([-1.0, 1.0], (n, len(x))); null = sign * (s * x).mean(1)
    return float((np.sum(null >= obs) + 1) / (n + 1))


def c2st(Xa, Xb, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    X = np.concatenate([Xa, Xb]); y = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]; p = np.zeros(len(y))
    for tr, te in StratifiedKFold(2, shuffle=True, random_state=seed).split(X, y):
        p[te] = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return float(H._auc(p, y.astype(int)))


# ======================================================================================================= main
def run_sample(ds, S, M, REF):
    """edits, scores, references for one sample; returns dict of per-set arrays"""
    nn, P4, y = S['nn'], S['P4'], S['y']; O = H._obs(P4, nn); S['O'] = O; log(f'[{ds}] observables done on {len(nn)} jets')
    win = (O['mSD'] > 105) & (O['mSD'] < 210) & (nn >= 3)
    elig = {'T60': win & (y == 1) & (O['mW'] > 65) & (O['mW'] < 95)}; elig['T100'] = elig['T60']
    d80 = np.abs(O['mW'] - H.MWREF); elig['T80'] = win & (y == 1) & (d80 > 6) & (d80 < 15)
    elig['Q80'] = win & (y == 0) & (((O['mW'] > 50) & (O['mW'] < 70)) | ((O['mW'] > 90) & (O['mW'] < 115)))
    tgt = {'T60': (60.0, 5.0), 'T100': (100.0, 5.0), 'T80': (H.MWREF, 2.0), 'Q80': (H.MWREF, 5.0)}
    E = {}
    for key in ('T60', 'T100', 'T80', 'Q80'):
        t, tol = tgt[key]; rows = np.flatnonzero(elig[key]); ed = HV.edit_set(P4, nn, rows, t); acc = {}
        for k in 'WN':
            e = ed[k]; rr = e['rows']; Oe = H._obs(e['P'], nn[rr]) if len(rr) else {kk: np.zeros(0) for kk in O}
            dm = np.abs(Oe['m'] - O['m'][rr]); dtau = np.abs(Oe['tau32'] - O['tau32'][rr])
            okW = np.abs(Oe['mW'] - t) < tol if k == 'W' else np.abs(Oe['mW'] - O['mW'][rr]) < 2
            acc[k] = set(rr[(dm < 0.5) & (dtau < 0.02) & okW].tolist())
        ar = np.array(sorted(acc['W'] & acc['N']), np.int64)
        E[key] = dict(t=t, rows=ar, edit=ed, pos={k: np.searchsorted(ed[k]['rows'], ar) for k in 'WNR'},
                      n=dict(eligible=int(len(rows)), W_conv=int(len(ed['W']['rows'])), N_conv=int(len(ed['N']['rows'])),
                             W_acc=len(acc['W']), N_acc=len(acc['N']), analysis=int(len(ar))))
        log(f'[{ds}] {key} edited: {E[key]["n"]}')
    # unedited scores on the whole sample
    order = np.random.default_rng(605).permutation(len(nn))
    s_id, lg = M.score(ds, P4, nn, S['base'], order, want_logits=True)
    X_id, _ = H._features(P4, nn); s_id.update(all_refs(REF, X_id, O)); S['X'] = X_id
    if ds == 'jc':   # input-path check: read_root jets as stored (build of kin) vs the identity-rebuilt four-vectors
        from interp.part_core import build, kin
        Jst = build(S['J'], *kin(S['J'])); chk = {}
        for name in M.jc:
            L0 = M.jc_logits(name, Jst); s0 = L0[:, TBQQ] - L0[:, QCD]
            ad = np.abs(s0 - s_id[name])
            chk[name] = dict(max_abs_dscore=float(ad.max()), p99_abs_dscore=float(np.quantile(ad, 0.99)), median_abs_dscore=float(np.median(ad)),
                             calls_changed=float(np.mean((s0 > 0) != (s_id[name] > 0))))
        S['input_check'] = chk; log(f'[{ds}] input-path check (stored vs identity rebuild): {chk}')
    perf = {}
    for k, s in s_id.items():
        perf[k] = dict(AUC=float(H._auc(s, y)), acc_binary=float(((s > 0) == (y == 1)).mean()) if not k.startswith('LR2') else None)
        if k in lg:
            truth = np.where(y == 1, TBQQ, QCD); perf[k]['acc_10class'] = float((lg[k].argmax(1) == truth).mean())
            perf[k]['frac_argmax_Tbqq_or_QCD'] = float(np.isin(lg[k].argmax(1), [TBQQ, QCD]).mean())
        st, sb = s[y == 1], s[y == 0]; perf[k]['wp'] = {}
        for nm, q in (('eS30', 0.7), ('eS50', 0.5)):
            thr = float(np.quantile(st, q)); eb = float((sb >= thr).mean())
            perf[k]['wp'][nm] = dict(thr=thr, eps_B=eb, n_bkg_pass=int((sb >= thr).sum()), rejection=float(1 / eb) if eb > 0 else None)
    log(f'[{ds}] unedited: ' + ', '.join(f"{k} AUC {v['AUC']:.4f} acc {v['acc_binary']}" for k, v in perf.items()))
    S['s_id'] = s_id; S['perf'] = perf
    # edited scores on the analysis rows (identity scored on the same rows in the same order, so LorentzNet batches match)
    for key, e in E.items():
        ar = e['rows']; e['s'] = {}
        if len(ar) == 0: continue
        ordr = np.random.default_rng(606).permutation(len(ar)); base = subd(S['base'], ar)
        e['s']['id'] = M.score(ds, P4[ar], nn[ar], base, ordr); e['s']['id'].update({k: v[ar] for k, v in s_id.items() if k not in e['s']['id']})
        e['X'] = {'id': (X_id[ar], {k: v[ar] for k, v in O.items()})}
        for k in 'WNR':
            Pk = e['edit'][k]['P'][e['pos'][k]]; s = M.score(ds, Pk, nn[ar], base, ordr)
            Xk, Ok = H._features(Pk, nn[ar]); s.update(all_refs(REF, Xk, Ok)); e['s'][k] = s; e['X'][k] = (Xk, Ok)
        log(f'[{ds}] {key} scored on {len(ar)} analysis jets')
    return E


def stats(ds, S, E, models):
    rng = np.random.default_rng(7); out = {}
    for key, e in E.items():
        ar = e['rows']; n = len(ar); r = dict(counts=e['n'], models={})
        if n < 5: out[key] = r; continue
        X0, O0 = e['X']['id']; XW, OW = e['X']['W']; XN, ON = e['X']['N']
        r['observables_median'] = {nm: {v: float(np.median(o[nm])) for v, o in (('id', O0), ('W', OW), ('N', ON))} for nm in ('mW', 'm', 'tau32', 'tau21')}
        r['observables_median']['subjet_mass_sum'] = {v: float(np.median(x[:, 14:17].sum(1))) for v, x in (('id', X0), ('W', XW), ('N', XN))}
        bi = rng.integers(0, n, (B, n)); sg = DIRECTION[key]
        d = {m: {k: e['s'][k][m] - e['s']['id'][m] for k in 'WNR'} for m in models}
        WN = {m: d[m]['W'] - d[m]['N'] for m in models}
        for m in models:
            x = WN[m]; M_ = dict(dW=float(d[m]['W'].mean()), dW_ci95=np.percentile(d[m]['W'][bi].mean(1), [2.5, 97.5]).tolist(),
                                dN=float(d[m]['N'].mean()), dN_ci95=np.percentile(d[m]['N'][bi].mean(1), [2.5, 97.5]).tolist(),
                                dR=float(d[m]['R'].mean()), dR_ci95=np.percentile(d[m]['R'][bi].mean(1), [2.5, 97.5]).tolist(),
                                W_minus_N=float(x.mean()), ci95=np.percentile(x[bi].mean(1), [2.5, 97.5]).tolist(),
                                p_perm_WN=perm_p(x, sg, rng))
            if not m.startswith(('LR2', 'MLPx')):
                tr = 'top' if m.endswith('_top') else 'jc'
                for ref in ('MLPx', 'MLPxW'):
                    rx = WN[f'{ref}_{tr}']; bm = rx[bi].mean(1)
                    M_[f'ratio_{ref}'] = float(x.mean() / rx.mean()) if abs(rx.mean()) > 1e-9 else None
                    M_[f'ratio_{ref}_ci95'] = np.percentile(x[bi].mean(1) / np.where(np.abs(bm) > 1e-9, bm, np.nan), [2.5, 97.5]).tolist()
                    M_[f'ref_{ref}_{tr}_WN'] = float(rx.mean())
                M_['p_perm_vs_MLPx'] = perm_p(x - WN[f'MLPx_{tr}'], sg, rng)
                M_['p_perm_vs_MLPxW_two_sided'] = float(min(1.0, 2 * min(perm_p(x - WN[f'MLPxW_{tr}'], 1, rng), perm_p(x - WN[f'MLPxW_{tr}'], -1, rng))))
            wp = S['perf'][m]['wp']; M_['calls'] = {}
            for nm in ('eS30', 'eS50'):
                thr = wp[nm]['thr']; s0 = e['s']['id'][m]; c0 = s0 >= thr; cW = e['s']['W'][m] >= thr; cN = e['s']['N'][m] >= thr
                M_['calls'][nm] = dict(pass_id=float(c0.mean()), pass_W=float(cW.mean()), pass_N=float(cN.mean()), mcnemar_W_vs_N=H._mcnemar(cN, cW))
            r['models'][m] = M_
        out[key] = r
    return out


def main():
    RES.mkdir(parents=True, exist_ok=True)
    z = np.array([[1.0, 0, 5, 6], [0, 1.0, 2, 3], [0.5, 0.5, 1, 2]]); H._tau32(z[:, 0].copy(), z[:, 1].copy(), z[:, 2].copy(), z[:, 3].copy(), 3)
    log(f'H19 start; smoke={SMOKE}, per class {PER}, ref jets per class {NREF}')
    SAMP = {'jc': load_jc(), 'top': load_top()}
    log('samples loaded: ' + json.dumps({k: v['info'] for k, v in SAMP.items()}, default=int))
    # references (fitted before the networks so that the GPU is not idle long: CPU work first, then GPU)
    REF = {}; ref_info = {}
    Xr, Or, yr = ref_train_jc(); REF['jc'], ref_info['jc'] = fit_refs(Xr, Or, yr, 'jc'); ref_info['jc']['n'] = int(len(yr)); del Xr, Or
    Xr, Or, yr = ref_train_top(); REF['top'], ref_info['top'] = fit_refs(Xr, Or, yr, 'top'); ref_info['top']['n'] = int(len(yr)); del Xr, Or
    M = Models()
    pm = {}
    for ds, S in SAMP.items():   # constituent masses per sample (decides the massless conversion of JetClass jets for the Top models)
        P = S['P4'].astype(np.float64); real = np.arange(P.shape[1])[None, :] < S['nn'][:, None]
        m2 = (P[..., 3] ** 2 - (P[..., :3] ** 2).sum(-1))[real]; Ep = P[..., 3][real]
        pm[ds] = dict(median_abs_m=float(np.median(np.sqrt(np.abs(m2)))), p90_abs_m=float(np.quantile(np.sqrt(np.abs(m2)), 0.9)),
                      frac_m_above_0p1GeV=float(np.mean(np.sqrt(np.maximum(m2, 0)) > 0.1)), median_E=float(np.median(Ep)))
    M.massless = pm['top']['frac_m_above_0p1GeV'] < 0.01; log(f'constituent masses {pm}; JetClass jets passed massless to Top models: {M.massless}')
    E = {ds: run_sample(ds, S, M, REF) for ds, S in SAMP.items()}
    # domain-shift check between the two samples
    c2 = {}
    for cls, lab in (('top', 1), ('QCD', 0)):
        Xa = SAMP['jc']['X'][SAMP['jc']['y'] == lab]; Xb = SAMP['top']['X'][SAMP['top']['y'] == lab]
        c2[cls] = dict(AUC_all17=c2st(Xa, Xb), AUC_no_multiplicity=c2st(np.delete(Xa, 7, 1), np.delete(Xb, 7, 1)),
                       median_jc={nm: float(np.median(SAMP['jc']['O'][nm][SAMP['jc']['y'] == lab])) for nm in ('m', 'mSD', 'tau32', 'mW', 'n')},
                       median_top={nm: float(np.median(SAMP['top']['O'][nm][SAMP['top']['y'] == lab])) for nm in ('m', 'mSD', 'tau32', 'mW', 'n')})
    log(f'C2ST jc vs top: {c2}')
    R = dict(node='H19', smoke=SMOKE, constituent_masses=pm, jc_to_top_massless=bool(M.massless), samples={ds: S['info'] for ds, S in SAMP.items()}, references=ref_info, c2st_jc_vs_top=c2,
             input_check_jc=SAMP['jc'].get('input_check'), unedited={ds: S['perf'] for ds, S in SAMP.items()}, sets={})
    nets = {}
    for ds in SAMP:
        models = [m for m in NETS_JC + NETS_TOP if m in SAMP[ds]['s_id']] + [f'{r}_{t}' for r in REFS for t in ('jc', 'top')]
        nets[ds] = models; R['sets'][ds] = stats(ds, SAMP[ds], E[ds], models)
    for fam, key in (('vs_zero', 'p_perm_WN'), ('vs_MLPx', 'p_perm_vs_MLPx')):
        tests = [(ds, k, m) for ds in R['sets'] for k, r in R['sets'][ds].items() for m in r['models'] if m in NETS_JC + NETS_TOP]
        adj = H._holm([R['sets'][ds][k]['models'][m][key] for ds, k, m in tests]) if tests else []
        for (ds, k, m), pa in zip(tests, adj): R['sets'][ds][k]['models'][m][f'holm_{fam}'] = float(pa)
        R[f'holm_family_{fam}'] = f'{len(tests)} tests: networks x samples x sets (T60, T100, T80, Q80), one-sided in the mW-window direction'
    (RES / f'{TAG}_wpair_cross.json').write_text(json.dumps(R, indent=1, default=float)); log(f'wrote {TAG}_wpair_cross.json')
    np.savez_compressed(RES / f'{TAG}_perjet.npz', **{f'{ds}|{k}|{v}|{m}': np.asarray(e['s'][v][m], np.float32)
                                                    for ds in E for k, e in E[ds].items() for v in e['s'] for m in e['s'][v]})
    for ds, sets in R['sets'].items():
        for key, r in sets.items():
            log(f'== {ds} {key} n {r["counts"]["analysis"]}')
            for m, Mm in r['models'].items():
                log(f"   {m:12s} dW {Mm['dW']:+.3f} dN {Mm['dN']:+.3f} dR {Mm['dR']:+.3f} W-N {Mm['W_minus_N']:+.3f} {np.round(Mm['ci95'], 3)} "
                    f"p {Mm['p_perm_WN']:.3g} " + (f"xMLPx {Mm['ratio_MLPx']} xMLPxW {Mm['ratio_MLPxW']} pMLPx {Mm['p_perm_vs_MLPx']:.3g} holm {Mm.get('holm_vs_MLPx')}"
                                                  if 'ratio_MLPx' in Mm else ''))
    try: fig(R)
    except Exception as ex: log('figure failed:', repr(ex))


def fig(R):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    nets = ['ParT_full', 'ParT_kin', 'PN_full', 'PN_kin', 'ParT_top', 'PN_top', 'LNet_top', 'PFN_top', 'EFN_top']
    lab = {'ParT_full': 'JetClass\nParT full', 'ParT_kin': 'JetClass\nParT kin', 'PN_full': 'JetClass\nPN full', 'PN_kin': 'JetClass\nPN kin',
           'ParT_top': 'Top\nParT', 'PN_top': 'Top\nPN', 'LNet_top': 'Top\nLorentzNet', 'PFN_top': 'Top\nPFN', 'EFN_top': 'Top\nEFN'}
    fig_, axes = plt.subplots(2, 1, figsize=(10, 7.4), sharex=True); x = np.arange(len(nets))
    for ax, key, ttl in ((axes[0], 'T60', 'W-pair mass 80 -> 60 GeV'), (axes[1], 'T100', 'W-pair mass 80 -> 100 GeV')):
        for j, (ds, c, mk, nm) in enumerate((('jc', '#2b6cb0', 'o', 'JetClass jets (550-650 GeV)'), ('top', '#c05621', 's', 'Top Landscape jets'))):
            r = R['sets'][ds].get(key, {}).get('models', {})
            for i, m in enumerate(nets):
                if m not in r or r[m].get('ratio_MLPx') is None: continue
                xx = i + (j - 0.5) * 0.3; v = r[m]['ratio_MLPx']; lo_, hi_ = r[m]['ratio_MLPx_ci95']
                ax.errorbar(xx, v, yerr=[[max(v - lo_, 0)], [max(hi_ - v, 0)]], fmt=mk, color=c, ms=7, capsize=3, label=nm)
                tr = 'top' if m.endswith('_top') else 'jc'; rx, rw = r[m].get(f'ref_MLPx_{tr}_WN'), r[m].get(f'ref_MLPxW_{tr}_WN')
                if rx and rw is not None: ax.plot(xx, rw / rx, marker='_', color=c, ms=18, mew=2.5, alpha=0.5)
        h, l = ax.get_legend_handles_labels(); seen = {}
        for hh, ll in zip(h, l): seen.setdefault(ll, hh)
        ax.axhline(1, color='k', lw=1, ls='--'); ax.axhline(0, color='0.6', lw=0.8); ax.axvline(3.5, color='0.8', lw=0.8)
        ax.set_ylabel('(W - N) net / (W - N) MLPx'); ax.set_title(f'{ttl} at fixed m and tau32', fontsize=10, loc='left')
        ax.legend(list(seen.values()), list(seen.keys()), fontsize=8.5, loc='upper left')
    axes[1].set_xticks(x); axes[1].set_xticklabels([lab[m] for m in nets], fontsize=9)
    fig_.suptitle('W-pair response relative to the side-effect reference MLPx (dashed line = 1; short bars = MLPxW, the reference that sees mW)', fontsize=10)
    fig_.tight_layout(); fig_.savefig(RES / f'{TAG}_wpair.png', dpi=130); log(f'wrote {TAG}_wpair.png')


if __name__ == '__main__':
    if '--fig' in sys.argv: fig(json.loads((RES / f'{TAG}_wpair_cross.json').read_text()))
    else: main()
