"""P26 — Sophon (ParT architecture trained on JetClass-II, 188 classes, released weights): is the axis-relative mass response
(C3: ParT's W/Z read-out follows the axis moment m_A = sqrt(M0 M2) about the stored axis, not the invariant mass) still
present after training on a larger and more varied, mass-decorrelated dataset?

Model: jet-universe/sophon models/JetClassII_Sophon/model.pt (state dict; also on github), network code interp/sophon_part.py
(weaver-core dev/custom_train_eval ParticleTransformer, 8 particle blocks + 2 class blocks, fc 128 -> 512 -> 188), inputs as
data/JetClassII/JetClassII_full.yaml (17 per-particle features incl. deltaR, deta, dphi; four-vectors scaled by 500 / jet_pt).
Cross-check: the released ONNX export (models/JetClassII_Sophon/model.onnx) on the same inputs (if onnxruntime installs).
Reference models: released ParT_full / ParT_kin (JetClass, 10 classes). Null: Sophon architecture, untrained (seeds 100, 101).

Class map (Sophon): Z-like = X -> bb, cc, ss (labels 0-2); W-like = X -> bc, cs, bq, cq, sq (4-8); X -> qq (u/d, label 3) is
shared by W -> ud and Z -> uu/dd and is left out of both; QCD = labels 161-187. Group score = log sum of softmax over the group.
Read-out d = log p(Z group) - log p(W group) (ParT: logit_Z - logit_W). Mapped call = argmax over (QCD, Z, W) group scores.
Second read-out d_res = log p(two-prong resonance, labels 0-8) - log p(QCD) (ParT: log(p_Z + p_W) - log p_QCD).
Probe: ridge read-out of log m_inv from the 128-d class token (input of fc), fitted on the unedited jets (P1v split, seed 1).

Samples:
  A  JetClass test_20M/{ZJetsToNuNu,ZToQQ,WToQQ}_112.root, 5,000 random rows per class (numpy default_rng(2601)), rebuilt
     unedited with part_core.build. jet_sdmass read for the Sophon training selection (200 < pT < 2500, 20 < m_SD < 500 GeV).
  B  JetClass-II test shards (read in place, /j-jepa-vol/JetClass2/Pythia/data/Res2P_0250..0253, QCD_0350): jets with
     500 < pT < 1000 GeV and 20 < m_SD < 500 GeV; Z-like truth X -> bb/cc/ss and W-like truth X -> cs/cq/sq with resonance
     mass 70 < m_X < 100 GeV; QCD any label 161-187; up to 2,000 random jets per group (default_rng(2602)).
Edits (as P1 / P1v / P22):
  phi       every particle's dphi + 0.06 rad about the stored axis (m_inv unchanged; m_A up)
  dil       per-jet matched dilation (deta, dphi) x k_j, k_j = sqrt(M0M2(phi) / M0M2(base)) (same Delta log m_A, real mass change)
  rot       whole jet and axis rotated 0.06 rad in phi (null: inputs equal up to rounding)
  phi_dR0   phi with the deltaR feature fed unedited (route: dphi only); phi_dphi0: dphi fed unedited (route: deltaR only)
  wta       deta, dphi, deltaR recomputed about the WTA axis (C/A, winner-take-all; P1v code); 4-vectors unchanged
  rotc      deltaR-keeping control: every particle rotated about the stored axis by a random angle per jet (deltaR unchanged)
  trc       matched translation control (X8v): inputs recomputed about an axis displaced by the WTA offset length in a random direction
Metrics: R = per-jet LS slope (through 0) of Delta d(phi) on Delta d(dil), 95 % bootstrap over jets (2,000); rot null;
  R_res, R_probe (calibrated against dil); W -> Z = fraction of true-W jets with mapped call Z before / after (McNemar);
  dilation response relative to the Z - W separation of d; route slopes; WTA / control call-change fractions (Wilson CI;
  McNemar WTA vs control on the paired changed indicator); permutation tests R_Sophon vs R_ParT_full / kin / untrained; Holm.
Pre-registered (running/C.md, task): thesis R 0.5-1.0 and WTA changes >= ~10 %; alternative R < 0.3 with the dilation
  response kept and WTA changes ~ the rotation control. Sample-A accuracy far below sample B -> domain-shift caveat.

  ./part_job.sh c-p26-run 'bash interp/p26_fetch.sh && python -m interp.p26_sophon_axis'   -> results/part/p26_sophon_axis.json, .png
  --smoke: 300 jets per class (A), 200 per group (B)
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp.part_core import read_root, build, kin, moments, cat, inputs, OUT, CL, ParTNetwork, sample as p1sample
from interp.p1v_verify_part import Ridge, split, wta_axes, reaxis
from interp.e19_common import mcnemar, holm, wilson
from interp.p14_planted_part import slope_boot, boot_weights, perm_test, ci
from interp.p14_planted_train import masses, JC

T0 = time.time()
iW, iZ, iQ = CL.index('Wqq'), CL.index('Zqq'), CL.index('QCD')
SOPHON = Path(os.environ.get('SOPHON_DIR', '/j-jepa-vol/aaron-interp/models/sophon'))
JC2 = Path('/j-jepa-vol/JetClass2/Pythia/data')
CLS3 = ['ZJetsToNuNu', 'ZToQQ', 'WToQQ']            # truth index 0 QCD, 1 Z, 2 W
DPHI = 0.06


def log(*a):
    print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


# ------------------------------------------------------------------------------------------------ samples
def sample_A(per, seed=2601, idx=112):
    import uproot
    rng = np.random.default_rng(seed); Js, sd = [], []
    for c in CLS3:
        f = JC / 'test_20M' / f'{c}_{idx}.root'; J = read_root(f); n = len(J['label'])
        r = np.sort(rng.choice(n, per, replace=False)); Js.append({k: (v[r] if np.ndim(v) else v) for k, v in J.items()}); del J
        s = uproot.open(f)['tree'].arrays(['jet_sdmass'], library='np')['jet_sdmass']; sd.append(s[r].astype(np.float32))
        log(f'sample A {f.name}: {per} of {n} rows')
    J = cat(Js); J['sdmass'] = np.concatenate(sd)
    y3 = np.select([J['label'] == iQ, J['label'] == iZ, J['label'] == iW], [0, 1, 2], -1); assert (y3 >= 0).all()
    return J, y3


def jc2_to_dict(tab):
    """JetClass-II arrow table -> jet dict as data.read_root (particles stored pT-ordered, checked; padded / cut to 128)"""
    import awkward as ak
    a = ak.from_arrow(tab); P = 128
    pt = np.hypot(a['part_px'], a['part_py']); order = ak.argsort(pt, ascending=False, stable=True)
    pad = lambda x: ak.to_numpy(ak.fill_none(ak.pad_none(x[order], P, clip=True), 0)).astype(np.float32)
    J = {k[5:]: pad(a[k]) for k in ['part_px', 'part_py', 'part_pz', 'part_energy', 'part_deta', 'part_dphi', 'part_d0val', 'part_d0err',
                                    'part_dzval', 'part_dzerr', 'part_charge', 'part_isChargedHadron', 'part_isNeutralHadron',
                                    'part_isPhoton', 'part_isElectron', 'part_isMuon']}
    J['mask'] = pad(ak.ones_like(a['part_px']))
    for k in ('jet_pt', 'jet_eta', 'jet_phi', 'jet_energy'): J[k] = ak.to_numpy(a[k]).astype(np.float32)
    J['sdmass'] = ak.to_numpy(a['jet_sdmass']).astype(np.float32); J['jc2_label'] = ak.to_numpy(a['jet_label']).astype(np.int32)
    return J


def sample_B(per, seed=2602, smoke=False):
    import pyarrow.parquet as pq, awkward as ak
    rng = np.random.default_rng(seed); want = {0: per, 1: per, 2: per}; got = {0: [], 1: [], 2: []}; info = []
    files = [JC2 / f'Res2P_{i:04d}.parquet' for i in range(250, 254)] + [JC2 / 'QCD_0350.parquet']
    for f in files:
        isqcd = f.name.startswith('QCD'); grp = [0] if isqcd else [1, 2]
        if all(sum(len(x['jet_pt']) for x in got[g]) >= want[g] for g in grp): continue
        cols = ['jet_pt', 'jet_sdmass', 'jet_label'] + ([] if isqcd else ['aux_genpart_mass', 'aux_genpart_isResX'])
        t = pq.read_table(f, columns=cols); a = ak.from_arrow(t)
        pt, sd, lab = (ak.to_numpy(a[k]) for k in ('jet_pt', 'jet_sdmass', 'jet_label'))
        sel = (pt > 500) & (pt < 1000) & (sd > 20) & (sd < 500)
        if isqcd:
            G = {0: sel & (lab >= 161)}
        else:
            mx = ak.to_numpy(ak.fill_none(ak.firsts(a['aux_genpart_mass'][a['aux_genpart_isResX']]), -1)); win = (mx > 70) & (mx < 100)
            G = {1: sel & win & np.isin(lab, [0, 1, 2]), 2: sel & win & np.isin(lab, [5, 7, 8])}
        rows = {}
        for g, s in G.items():
            have = sum(len(x['jet_pt']) for x in got[g]); need = want[g] - have; cand = np.flatnonzero(s)
            if need > 0 and len(cand): rows[g] = np.sort(rng.choice(cand, min(need, len(cand)), replace=False))
        info.append(dict(file=f.name, n=int(len(pt)), candidates={int(g): int(s.sum()) for g, s in G.items()}, taken={int(g): int(len(r)) for g, r in rows.items()}))
        if rows:
            allr = np.sort(np.concatenate(list(rows.values())))
            full = pq.read_table(f, columns=[c for c in pq.ParquetFile(f).schema_arrow.names if c.startswith('part_')] +
                                 ['jet_pt', 'jet_eta', 'jet_phi', 'jet_energy', 'jet_sdmass', 'jet_label']).take(allr)
            J = jc2_to_dict(full); pos = {r: i for i, r in enumerate(allr)}
            for g, r in rows.items():
                got[g].append({k: v[[pos[x] for x in r]] for k, v in J.items()})
        log(f'sample B {f.name}: {info[-1]}')
    Js, y3 = [], []
    for g in (0, 1, 2):
        Jg = cat(got[g]); Js.append(Jg); y3.append(np.full(len(Jg['jet_pt']), g))
    J = cat(Js); y3 = np.concatenate(y3); J['label'] = np.array([iQ, iZ, iW])[y3].astype(np.int8)
    return J, y3, info


# ------------------------------------------------------------------------------------------------ edits
def edits(J0, rng):
    pt, de, dp, ms = kin(J0); real = pt > 0
    Jb = build(J0, pt, de, dp, ms); Mb = moments(*kin(Jb)[:3])
    E = {'phi': build(J0, pt, de, dp + DPHI * real, ms)}
    M2p = moments(*kin(E['phi'])[:3])['M0M2']; okk = Mb['M0M2'] > 0
    kj = np.where(okk, np.sqrt(M2p / np.where(okk, Mb['M0M2'], 1.0)), 1.0)[:, None]
    E['dil'] = build(J0, pt, de * kj, dp * kj, ms)
    Jr = dict(Jb); Jr['jet_phi'] = (Jb['jet_phi'] + DPHI).astype(np.float32); c_, s_ = np.cos(DPHI), np.sin(DPHI)
    Jr['px'] = (c_ * Jb['px'] - s_ * Jb['py']).astype(np.float32); Jr['py'] = (s_ * Jb['px'] + c_ * Jb['py']).astype(np.float32); E['rot'] = Jr
    # WTA axis and its controls
    t = time.time(); wd, wp = wta_axes(Jb); twta = time.time() - t
    sgn = np.where(Jb['jet_eta'] < 0, -1.0, 1.0); je, jp = Jb['jet_eta'].astype(np.float64), Jb['jet_phi'].astype(np.float64)
    E['wta'], _ = reaxis(Jb, je + sgn * wd, jp + wp)
    off = np.hypot(wd, wp); ang = rng.uniform(0, 2 * np.pi, len(off))
    E['trc'], _ = reaxis(Jb, je + sgn * off * np.cos(ang), jp + off * np.sin(ang))
    al = rng.uniform(0, 2 * np.pi, len(off))[:, None]; ca, sa = np.cos(al), np.sin(al)
    E['rotc'] = build(J0, pt, (ca * de - sa * dp) * real, (sa * de + ca * dp) * real, ms)
    mi, mA = masses(Jb); em = {k: masses(Je) for k, Je in E.items()}
    dR = lambda J: np.hypot(J['deta'], J['dphi'])
    info = dict(wta_seconds=twta, wta_offset_median=float(np.median(off)), wta_offset_mean=float(off.mean()),
                wta_is_hardest=float(np.mean((np.abs(wd - Jb['deta'][:, 0]) < 1e-7) & (np.abs(wp - Jb['dphi'][:, 0]) < 1e-7))),
                rotc_max_abs_dDeltaR=float(np.abs(dR(E['rotc']) - dR(Jb))[Jb['mask'] > 0].max()),
                edits={k: dict(dlogminv=float(np.mean(np.log(np.maximum(em[k][0], 1)) - np.log(np.maximum(mi, 1)))),
                               dlogmA=float(np.mean(np.log(np.maximum(em[k][1], 1e-6)) - np.log(np.maximum(mA, 1e-6))))) for k in E})
    return Jb, E, mi, info


# ------------------------------------------------------------------------------------------------ models
def lse(L, idx):
    a = L[:, idx]; mx = a.max(1, keepdims=True); return (mx + np.log(np.exp(a - mx).sum(1, keepdims=True)))[:, 0]


class Model:
    def __init__(self, name, sd=None):
        import torch
        self.t, self.name = torch, name; self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        if name.startswith('part_'):
            self.kind = name[5:]; self.K = 10; self.m = ParTNetwork(self.kind).model.to(self.dev).eval()
            self.cap = {}; self.m.fc.register_forward_hook(lambda mod, i, o: self.cap.__setitem__('h', i[0].detach()))
            self.dR, self.dphi = 4, (16 if self.kind == 'full' else 6)
        else:
            from interp.sophon_part import SophonNet, SOPHON_IDX
            self.kind = 'sophon'; self.K = 188
            self.net = SophonNet(sd, seed=None if sd is not None else 100 + int(name[-1])).to(self.dev).eval(); self.m = self.net
            self.dR, self.dphi = SOPHON_IDX['deltaR'], SOPHON_IDX['dphi']
        for p in self.m.parameters(): p.requires_grad_(False)

    def feats(self, J):
        if self.kind == 'sophon':
            from interp.sophon_part import sophon_inputs
            _, x, v, m = sophon_inputs(J); return x, v, m
        return inputs(J, self.kind)

    def run(self, J, sub=None, batch=256):
        """J -> logits (N, K), class token (N, 128); sub = (J_base, [feature channels]) feeds those channels from J_base"""
        t = self.t; N = len(J['mask']); order = np.argsort(J['mask'].sum(1), kind='stable')
        L = np.zeros((N, self.K), np.float32); H = np.zeros((N, 128), np.float32)
        T = lambda a: t.from_numpy(np.ascontiguousarray(a)).to(self.dev)
        with t.no_grad():
            for i in range(0, N, batch):
                idx = order[i:i + batch]; Jb = {k: (v[idx] if np.ndim(v) else v) for k, v in J.items()}
                x, v, m = self.feats(Jb)
                if sub is not None:
                    xb, _, mb = self.feats({k: (v_[idx] if np.ndim(v_) else v_) for k, v_ in sub[0].items()}); assert np.array_equal(m, mb)
                    for c in sub[1]: x[:, c] = xb[:, c]
                out = self.m(T(x), T(v), T(m))
                L[idx] = out.float().cpu().numpy()
                H[idx] = (self.net.cls if self.kind == 'sophon' else self.cap['h']).float().cpu().numpy()
        return L.astype(np.float64), H.astype(np.float64)

    def read(self, L):
        """(d = log pZ - log pW, d_res = log p(res) - log pQCD, mapped call 0 QCD / 1 Z / 2 W, top-1 class)"""
        if self.kind == 'sophon':
            from interp.sophon_part import GROUP_Z, GROUP_W, GROUP_QCD, GROUP_RES2
            gZ, gW, gQ, gR = lse(L, GROUP_Z), lse(L, GROUP_W), lse(L, GROUP_QCD), lse(L, GROUP_RES2)
        else:
            gZ, gW, gQ = L[:, iZ], L[:, iW], L[:, iQ]; gR = lse(L, [iZ, iW])
        return gZ - gW, gR - gQ, np.stack([gQ, gZ, gW], 1).argmax(1), L.argmax(1)


def onnx_check(sd_dir, J, net_out, n=200):
    """the released ONNX export on the same inputs: max |softmax difference| vs our PyTorch run"""
    try:
        try:
            import onnxruntime as ort
        except ImportError:
            import subprocess
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-cache-dir', '--no-deps', '--target', '/tmp/ort', 'onnxruntime==1.16.3', 'flatbuffers', 'coloredlogs', 'humanfriendly'], check=True)
            sys.path.append('/tmp/ort'); import onnxruntime as ort      # appended: the job's numpy / protobuf stay first
        from interp.sophon_part import sophon_inputs
        so = ort.SessionOptions(); so.intra_op_num_threads = 4; so.inter_op_num_threads = 1
        s = ort.InferenceSession(str(sd_dir / 'model.onnx'), so, providers=['CPUExecutionProvider'])
        names = [i.name for i in s.get_inputs()]; rows = np.arange(min(n, len(J['mask']))); out = []
        for r in rows:          # the export has the batch size fixed to 1 (reshape to (P, 8, 16) per head): one jet per call
            Jn = {k: (v[[r]] if np.ndim(v) else v) for k, v in J.items()}
            pts, x, v, m = sophon_inputs(Jn); feed = dict(pf_points=pts, pf_features=x, pf_vectors=v, pf_mask=m)
            out.append(s.run(None, {k: feed[k] for k in names})[0][0])
        out = np.array(out); res_ = dict(onnx_output_width=int(out.shape[1])); out = out[:, :188]     # export_embed: softmax (188) + class token (128)
        L = net_out[rows]; p = np.exp(L - L.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        o = out if np.allclose(out.sum(1), 1, atol=1e-3) else np.exp(out - out.max(1, keepdims=True)) / np.exp(out - out.max(1, keepdims=True)).sum(1, keepdims=True)
        return dict(ok=True, inputs=names, n=int(len(rows)), **res_, max_abs_softmax_diff=float(np.abs(o - p).max()),
                    top1_agree=float((o.argmax(1) == p.argmax(1)).mean()))
    except Exception as ex:
        return dict(ok=False, error=repr(ex)[:400])


# ------------------------------------------------------------------------------------------------ analysis of one model on one sample
def analyse(M, Jb, E, y3, mi, rng, NBt=2000):
    N = len(y3); isW, isZ = y3 == 2, y3 == 1; t1 = time.time()
    L0, H0 = M.run(Jb); d0, r0, c0, k0 = M.read(L0)
    fit, val, tr, te = split(N); rd = Ridge(H0, np.log(np.maximum(mi, 1.0)), fit, val, tr)
    row = dict(acc3=[float((c0 == y3).mean())] + wilson(int((c0 == y3).sum()), N),
               confusion3_rows_true_QCD_Z_W=[[int(((y3 == i) & (c0 == j)).sum()) for j in range(3)] for i in range(3)],
               acc_res_vs_qcd=float(((r0 > 0) == (y3 > 0)).mean()), acc_WZ_only=float((np.where(d0 > 0, 1, 2) == y3)[y3 > 0].mean()),
               d_mean_by_class=[float(d0[y3 == g].mean()) for g in range(3)], probe_r2_test=float(1 - ((rd(H0[te]) - np.log(np.maximum(mi[te], 1))) ** 2).sum() / ((np.log(np.maximum(mi[te], 1)) - np.log(np.maximum(mi[te], 1)).mean()) ** 2).sum()))
    Bz = rng.integers(0, isZ.sum(), (1000, isZ.sum())); Bw = rng.integers(0, isW.sum(), (1000, isW.sum()))
    sep = d0[isZ][Bz].mean(1) - d0[isW][Bw].mean(1); row['sep_ZminusW'] = [float(d0[isZ].mean() - d0[isW].mean())] + ci(sep)
    if M.kind == 'sophon':
        from interp.sophon_part import GROUP_QQ, GROUP_RES2, GROUP_QCD
        p = np.exp(L0 - L0.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        row['top1_in_res2'] = float(np.isin(k0, GROUP_RES2).mean()); row['top1_in_qcd'] = float(np.isin(k0, GROUP_QCD).mean())
        row['mean_p_qq_by_class'] = [float(p[y3 == g][:, GROUP_QQ].sum(1).mean()) for g in range(3)]
        row['top1_hist'] = {int(k): int(v) for k, v in zip(*np.unique(k0, return_counts=True)) if v >= max(5, N // 200)}
    Dd, Dr, Dp, C, K = {}, {}, {}, {}, {}
    variants = list(E.items()) + [('phi_dR0', (E['phi'], (Jb, [M.dR]))), ('phi_dphi0', (E['phi'], (Jb, [M.dphi])))]
    for k, Je in variants:
        if isinstance(Je, tuple): L, H = M.run(Je[0], sub=Je[1])
        else: L, H = M.run(Je)
        d, r, c, kk = M.read(L); Dd[k], Dr[k], C[k], K[k] = d - d0, r - r0, c, kk; Dp[k] = rd(H[te]) - rd(H0[te])
    Wb = boot_weights(N, rng, NBt); out = {}
    for k in Dd:
        p0, p1 = (c0 == 1)[isW], (C[k] == 1)[isW]; Bi = rng.integers(0, isW.sum(), (1000, isW.sum()))
        ch = K[k] != k0; ch3 = C[k] != c0
        out[k] = dict(changed_top1=[float(ch.mean())] + wilson(int(ch.sum()), N), changed_mapped3=[float(ch3.mean())] + wilson(int(ch3.sum()), N),
                      acc3_after=float((C[k] == y3).mean()), W_to_Z=[float(p0.mean()), float(p1.mean())], W_to_Z_change=[float(p1.mean() - p0.mean())] + ci(p1[Bi].mean(1) - p0[Bi].mean(1)),
                      mcnemar_W_to_Z=mcnemar(p0, p1), dd_mean=float(Dd[k].mean()), dd_mean_W=float(Dd[k][isW].mean()),
                      dd_mean_W_ci=ci(Dd[k][isW][Bi].mean(1)), dres_mean=float(Dr[k].mean()), dprobe_mean=float(Dp[k].mean()))
    row['edits'] = out
    R, Rb = slope_boot(Dd['phi'], Dd['dil'], Wb); row['R'] = [R] + ci(Rb); row['R_p_le0'] = float((1 + (Rb <= 0).sum()) / (1 + len(Rb)))
    Rk, Rkb = slope_boot(Dd['rot'], Dd['dil'], Wb); row['R_rot_null'] = [Rk] + ci(Rkb)
    Rr, Rrb = slope_boot(Dr['phi'], Dr['dil'], Wb); row['R_res'] = [Rr] + ci(Rrb)
    Wt = boot_weights(len(te), rng, 1000); row['R_probe'] = [float(Dp['phi'].mean() / Dp['dil'].mean())] + ci((Wt @ Dp['phi']) / (Wt @ Dp['dil']))
    row['R_probe_rot_null'] = float(Dp['rot'].mean() / Dp['dil'].mean())
    s, sb = slope_boot(Dd['phi'][isW], Dd['dil'][isW], boot_weights(int(isW.sum()), rng, 1000)); row['R_Wjets'] = [s] + ci(sb)
    row['R_ratio_of_means'] = float(Dd['phi'].mean() / Dd['dil'].mean())
    row['dil_over_sep'] = float(Dd['dil'][isW].mean() / row['sep_ZminusW'][0]) if abs(row['sep_ZminusW'][0]) > 1e-9 else None
    row['mean_abs_dd_dil'] = float(np.abs(Dd['dil']).mean()); row['sd_d0'] = float(d0.std())
    row['routes_phi'] = {}
    for k in ('phi_dR0', 'phi_dphi0'):
        s, sb = slope_boot(Dd[k], Dd['phi'], Wb); row['routes_phi'][k] = [s] + ci(sb)
    # WTA vs controls: paired changed indicators
    row['wta_vs_controls'] = {}
    for ctl in ('rotc', 'trc'):
        for lev, KK, base in (('top1', K, k0), ('mapped3', C, c0)):
            a, b = KK['wta'] != base, KK[ctl] != base
            row['wta_vs_controls'][f'{ctl}_{lev}'] = dict(wta=float(a.mean()), control=float(b.mean()), mcnemar=mcnemar(b, a))
    log(f'{M.name}: acc3 {row["acc3"][0]:.4f} (W/Z-only {row["acc_WZ_only"]:.4f}, res-vs-QCD {row["acc_res_vs_qcd"]:.4f}); '
        f'R {R:+.3f} [{row["R"][1]:+.3f}, {row["R"][2]:+.3f}] rot null {Rk:+.4f}; R_res {Rr:+.3f}; R_probe {row["R_probe"][0]:+.3f} '
        f'(probe R2 {row["probe_r2_test"]:.3f}); sep Z-W {row["sep_ZminusW"][0]:+.3f}, dil/sep {row["dil_over_sep"]}; '
        f'W->Z phi {out["phi"]["W_to_Z"]} dil {out["dil"]["W_to_Z"]}; changed top1 wta {out["wta"]["changed_top1"][0]:.4f} '
        f'rotc {out["rotc"]["changed_top1"][0]:.4f} trc {out["trc"]["changed_top1"][0]:.4f}; mapped3 wta {out["wta"]["changed_mapped3"][0]:.4f} '
        f'rotc {out["rotc"]["changed_mapped3"][0]:.4f} trc {out["trc"]["changed_mapped3"][0]:.4f}; routes {row["routes_phi"]} ({time.time() - t1:.0f} s)')
    return row, (Dd['phi'], Dd['dil'])


def verdict(row):
    R = row['R'][0]; w = row['edits']['wta']['changed_mapped3'][0]; rc = row['edits']['rotc']['changed_mapped3'][0]
    return dict(thesis=bool(0.5 <= R <= 1.0 and row['edits']['wta']['changed_top1'][0] >= 0.10),
                alternative=bool(R < 0.3 and (row['dil_over_sep'] or 0) > 0.05 and abs(w - rc) < 0.03))


def plot(res, fn):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    keys = [k for k in res['rows']]; lab = [k.replace('__', '\n') for k in keys]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    R = np.array([res['rows'][k]['R'] for k in keys]); x = np.arange(len(keys))
    ax[0].bar(x, R[:, 0], color=['C0' if 'sophon' in k and 'init' not in k else ('C7' if 'init' in k else 'C1') for k in keys])
    ax[0].errorbar(x, R[:, 0], yerr=[R[:, 0] - R[:, 1], R[:, 2] - R[:, 0]], fmt='none', ecolor='k', capsize=3)
    ax[0].axhspan(0.5, 1.0, color='C2', alpha=0.12, label='thesis range 0.5-1.0'); ax[0].axhline(0.3, color='C3', ls='--', lw=1, label='alternative: R < 0.3')
    ax[0].set_xticks(x); ax[0].set_xticklabels(lab, fontsize=7); ax[0].set_ylabel('R = Δd(φ + 0.06 rad) / Δd(matched dilation)')
    ax[0].set_title('Response to a 0.06 rad axis offset per unit of real size change'); ax[0].legend(fontsize=7)
    W = np.array([[res['rows'][k]['edits'][e]['changed_top1'][0] for e in ('wta', 'trc', 'rotc')] for k in keys]) * 100
    for j, (e, c) in enumerate(zip(('WTA axis', 'random offset, same length', 'ΔR-keeping rotation'), ('C0', 'C8', 'C7'))):
        ax[1].bar(x + (j - 1) * 0.27, W[:, j], 0.27, label=e, color=c)
    ax[1].set_xticks(x); ax[1].set_xticklabels(lab, fontsize=7); ax[1].set_ylabel('top-1 calls changed [%]')
    ax[1].set_title('Inputs recomputed about another axis'); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(fn, dpi=130); plt.close(fig)


def main():
    import torch
    ap = argparse.ArgumentParser(); ap.add_argument('--per', type=int, default=5000); ap.add_argument('--perB', type=int, default=2000)
    ap.add_argument('--smoke', action='store_true'); a = ap.parse_args()
    if a.smoke: a.per, a.perB = 300, 200
    tag = 'smoke' if a.smoke else 'run'; OUT.mkdir(parents=True, exist_ok=True); fn = OUT / f'p26_sophon_axis_{tag}.json'
    rng = np.random.default_rng(2626)
    sd = torch.load(SOPHON / 'model.pt', map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd: sd = sd['state_dict']
    if not all(k.startswith('mod.') for k in sd): sd = {('mod.' + k if not k.startswith('mod.') else k): v for k, v in sd.items()}
    log(f'Sophon state dict: {len(sd)} tensors, {sum(v.numel() for v in sd.values() if hasattr(v, "numel")) / 1e6:.2f} M numbers')
    res = dict(per_A=a.per, per_B=a.perB, seeds=dict(A=2601, B=2602, analysis=2626), rows={}, samples={})
    models = {'sophon': lambda: Model('sophon', sd), 'part_full': lambda: Model('part_full'), 'part_kin': lambda: Model('part_kin'),
              'sophon_init0': lambda: Model('sophon_init0'), 'sophon_init1': lambda: Model('sophon_init1')}
    npar = sum(p.numel() for p in Model('sophon', sd).m.parameters()); res['sophon_parameters'] = int(npar); log(f'Sophon parameters: {npar / 1e6:.3f} M')
    # reference accuracies of the released ParT models on the P1 sample
    if not a.smoke:
        JP = p1sample(1000, 3000); JPb = build(JP, *kin(JP)); yP = JP['label'].astype(int)
        for nm in ('part_full', 'part_kin'):
            M = models[nm](); L, _ = M.run(JPb); res[f'{nm}_acc10_P1_sample'] = float((L.argmax(1) == yP).mean()); log(f'{nm}: P1-sample 10-class accuracy {res[f"{nm}_acc10_P1_sample"]:.4f} (P1: full 0.859, kin 0.745)')
        del JP, JPb
    dd = {}
    for sname in ('A', 'B'):
        if sname == 'A':
            J0, y3 = sample_A(a.per); binfo = None
        else:
            J0, y3, binfo = sample_B(a.perB, smoke=a.smoke)
        insel = (J0['jet_pt'] > 200) & (J0['jet_pt'] < 2500) & (J0['sdmass'] > 20) & (J0['sdmass'] < 500)
        Jb, E, mi, info = edits(J0, rng); info.update(N=int(len(y3)), per_class=[int((y3 == g).sum()) for g in range(3)], sophon_selection_frac=float(insel.mean()),
                                                      mass_median_by_class=[float(np.median(mi[y3 == g])) for g in range(3)], files_B=binfo)
        res['samples'][sname] = info
        log(f'sample {sname}: N {len(y3)}, per class {info["per_class"]}, in Sophon selection {insel.mean():.3f}, m_inv median by class '
            f'{info["mass_median_by_class"]}; WTA offset median {info["wta_offset_median"]:.4f} ({info["wta_seconds"]:.0f} s); edits {info["edits"]}')
        for nm in (['sophon', 'part_full', 'part_kin', 'sophon_init0', 'sophon_init1'] if sname == 'A' else ['sophon', 'part_full', 'part_kin']):
            M = models[nm]()
            if nm == 'sophon' and sname == 'A':
                L0, _ = M.run(Jb); res['onnx_check'] = onnx_check(SOPHON, Jb, L0); log(f'ONNX cross-check: {res["onnx_check"]}')
                _, _, c0, _ = M.read(L0); res['sophon_acc3_A_in_selection'] = float((c0 == y3)[insel].mean())
            row, dd[(nm, sname)] = analyse(M, Jb, E, y3, mi, rng); row['verdict'] = verdict(row); res['rows'][f'{nm}__{sname}'] = row
            if nm == 'sophon' and sname == 'B':        # in-distribution check: 188-class top-1 against the JetClass-II label
                from interp.sophon_part import GROUP_QCD
                L0, _ = M.run(Jb); k0 = L0.argmax(1); lab = J0['jc2_label']; qcd = np.isin(lab, GROUP_QCD)
                row['top1_acc188'] = float((k0 == lab).mean()); row['top1_acc188_by_group'] = [float((k0 == lab)[y3 == g].mean()) for g in range(3)]
                row['top1_in_true_major_group'] = float(np.where(qcd, np.isin(k0, GROUP_QCD), np.isin(k0, list(range(9)))).mean())
                log(f'sophon on B: 188-class top-1 {row["top1_acc188"]:.4f} by group QCD/Z/W {row["top1_acc188_by_group"]}; top-1 in the true major group {row["top1_in_true_major_group"]:.4f}')
            del M; torch.cuda.empty_cache() if torch.cuda.is_available() else None
            json.dump(res, open(fn, 'w'), indent=1)
        del J0, Jb, E
    # permutation tests (same jets) and Holm over the test family
    tests, ps = {}, []
    for x, y in (('sophon', 'part_full'), ('sophon', 'part_kin'), ('sophon', 'sophon_init0')):
        t = perm_test(*dd[(x, 'A')], *dd[(y, 'A')], rng); tests[f'R_{x}_vs_{y}_A'] = t; ps.append(t['p'])
    for nm in ('sophon', 'part_full', 'part_kin'):
        for s in ('A', 'B'):
            e = res['rows'][f'{nm}__{s}']['edits']; tests[f'mcnemar_WtoZ_phi_{nm}_{s}'] = e['phi']['mcnemar_W_to_Z']; ps.append(e['phi']['mcnemar_W_to_Z']['p'])
            for ctl in ('rotc', 'trc'):
                m_ = res['rows'][f'{nm}__{s}']['wta_vs_controls'][f'{ctl}_top1']['mcnemar']; tests[f'mcnemar_wta_vs_{ctl}_{nm}_{s}'] = m_; ps.append(m_['p'])
    for k, h in zip(tests, holm(ps)): tests[k]['p_holm'] = h
    res['tests'] = tests
    rs, rb = res['rows']['sophon__A'], res['rows']['sophon__B']
    res['domain_shift_caveat'] = dict(acc_res_vs_qcd_A=rs['acc_res_vs_qcd'], acc_res_vs_qcd_B=rb['acc_res_vs_qcd'], acc3_A=rs['acc3'][0], acc3_B=rb['acc3'][0],
                                      flag=bool(rs['acc_res_vs_qcd'] < rb['acc_res_vs_qcd'] - 0.10))
    res['seconds'] = time.time() - T0; json.dump(res, open(fn, 'w'), indent=1)
    plot(res, OUT / f'p26_sophon_axis_{tag}.png'); log(f'wrote {fn}; domain caveat {res["domain_shift_caveat"]}; tests ' +
                                                       '; '.join(f'{k} p {v.get("p"):.2e} holm {v["p_holm"]:.2e}' for k, v in tests.items()))


if __name__ == '__main__':
    main()
