"""P40 — class-token MLP neuron atlas of ParT (loop C round C9, exploratory; new claim C19).

Question. Are the 2 x 512 hidden neurons of ParT's two class-attention-block MLPs (cls_blocks[0..1].fc1 -> GELU -> LayerNorm
-> fc2) individually readable (one physics observable or one class per neuron), more often than the directions of a random
rotation of the same hidden space, and does steering a labelled neuron move the class calls the way its label predicts?

Models: ParT_full, ParT_kin (released). Atlas controls: an untrained ParT of each feature set (ParTNetwork(untrained=True,
  seed 0): every weight tensor redrawn from a Gaussian with the trained tensor's mean and s.d.), a weight-permuted ParT of
  each (every weight tensor's entries permuted, seed 5, as P13), and 10 random orthogonal rotations Q (Haar, seeds 0-9) of
  each 512-d hidden layer, applied (i) to the raw activations ('rot_raw') and (ii) to the activations standardised on the
  labelling half ('rot_std'; P15v showed that the comparison can depend on this choice). A rotation that is undone before
  fc2 leaves the network output unchanged, so rotated coordinates are an equally valid basis of the same computation.
Sample: JetClass test_20M <sample>_110.root (the file named in the plan, _126, does not exist in test_20M, which has
  _100-_119; _110-_113 are the least used), NPC random rows per class (numpy default_rng(4040), never first-k), every jet
  rebuilt through part_core.build. Within each class a random half is the labelling half, the other the held-out half.
  Reference check: unedited accuracy on the P1 sample (part_core.sample(per_class=1000): rows 3,000-4,000 of each _100 file).
Sites: per block b1, b2 the pre-GELU (fc1 output) and the post-GELU (GELU output, before the 512-d LayerNorm and fc2)
  activation of the class token, (J, 512) each.
(1) Read. 22 labels: 12 observables (N, log pT, log m (invariant), log m_A (m_A = sqrt(M0 M2), M0 = sum pT, M2 = sum pT r^2
  about the stored axis), tau21, tau32 (file N-subjettiness), D2 (beta = 1, all particles), leading-particle pT fraction,
  number of charged / neutral particles, number of leptons, max |d0| of charged particles) and 10 one-vs-rest classes.
  Score S = Spearman rho for an observable, 2 (AUC - 0.5) for a class (both in [-1, 1]). Label of a unit = argmax |S| on the
  labelling half; its held-out score v = sign_lab x S_heldout[label]. p-values on the held-out half (t for rho, normal
  approximation of Mann-Whitney for AUC), Benjamini-Hochberg over 512 x 22 per site. Clean = v >= thr and BH q < 0.05.
  thr = 95th percentile of v over the untrained network's 1,024 units of the same site type (pre / post), so the untrained
  network passes <= 5 %. Also reported at fixed thr 0.2 / 0.3 / 0.5.
(2) Write (trained models, post-GELU clean units, at most NSTEER per model by v), on the held-out half: the class-token path
  is re-run from cached class-block inputs (checked against the full forward). Each unit gets a target class c* and a
  predicted sign s: class label -> c* = that class, s = sign of its score; observable label o -> c* = argmax_c sign(rho) x
  (AUC_c(o) - 0.5) (the class in which o is most shifted in the direction the unit follows), s = +1. Steering: add k x sd
  (k = +-1, 2, 4; sd = the unit's s.d. on the labelling half) to the unit; mean ablation: set it to its labelling-half mean.
  Metrics: Delta l_c* (l_c = log p_c - log(1 - p_c), per jet vs the unedited pass), specificity Delta l_c* - mean of the
  other 9 Delta l, 95 % bootstrap over jets (NB), exact McNemar on 'called c*' (steering) or 'called correctly' among class-c*
  jets (ablation), BH over units. Controls per unit: (a) the same-index column of rotation Q_0 (raw) in the same block,
  steered by k x its own projection s.d.; (b) the unit of the same block with the closest s.d. whose label and target class
  differ ('matched'), steered by k x its s.d., read on the first unit's c*. Direction-correct = s Delta l_c*(+2 sd) > 0 with
  its interval excluding 0 and s Delta l_c*(-2 sd) < 0.
(3) Calibration (mass-labelled units: label log m or log m_A): real mass change = all particle angles (deta, dphi) x 1.05 at
  fixed pT and particle mass (the P1 matched dilation; m and m_A x ~1.05). Patch the unit's activation to its value on the
  dilated jet (everything else from the unedited jet): R = mean Delta l_c*(patch) / mean Delta l_c*(dilation) = the share of
  the network's response to the real mass change carried by this unit. Joint patch of all mass units vs the same number of
  random rotated directions (10 draws).
(4) Neuron circuits per class c (post-GELU units labelled with class c): ablate them (others intact) / keep only them (the
  other post-GELU units of both class blocks at their mean; attention and particle blocks intact): 10-class accuracy,
  recall, confusion; NRSET random unit sets of the same per-block size.
  ./sync.sh && ./part_job.sh c-p40-run 'python -m interp.p40_neuron_atlas --smoke && python -m interp.p40_neuron_atlas'
  -> $AI_RESULTS/part/p40_neuron_atlas[_smoke].json, results/part/p40_neuron_atlas.png
"""
import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import interp.p32_lepton_route as P      # noqa: E402  (parses --smoke / --plot; helpers log, boot stats, load_rows, rebuild)

SMOKE = P.SMOKE
SEED = 4040
FILE = '110'
NPC = 100 if SMOKE else 2000
NB = 100 if SMOKE else 1000
NROT = 2 if SMOKE else 10
NSTEER = 6 if SMOKE else 300
NRSET = 2 if SMOKE else 20
NRCAL = 2 if SMOKE else 10
BS = 500
KS = (-4, -2, -1, 1, 2, 4)
DIL = 1.05
FN = P.OUTD / f'p40_neuron_atlas{P.TAG}.json'
FIG = P.OUTD / f'p40_neuron_atlas{P.TAG}.png'
log = P.log
OBS = ['N', 'log_pt', 'log_m', 'log_mA', 'tau21', 'tau32', 'D2', 'lead_frac', 'n_charged', 'n_neutral', 'n_lepton', 'max_abs_d0']
SITES = [(0, 'pre'), (0, 'post'), (1, 'pre'), (1, 'post')]
MASSLAB = ('log_m', 'log_mA')
FAMS = [('count', ['N', 'n_charged', 'n_neutral']), ('mass', ['log_m', 'log_mA']), ('pt', ['log_pt']), ('shape', ['tau21', 'tau32', 'D2']),
        ('lead', ['lead_frac']), ('lepton', ['n_lepton']), ('d0', ['max_abs_d0'])]
FAMIX = [[OBS.index(k) for k in ks] for _, ks in FAMS] + [[len(OBS) + c] for c in range(10)]
FAMOF = {j: f for f, ix in enumerate(FAMIX) for j in ix}


# ------------------------------------------------------------------------------------------------ data
def load(rng):
    import uproot
    from interp.part_core import PRE, CL, sub, cat
    lab, ho, taus = [], [], []
    for c in CL:
        path = P.TEST / f'{PRE[c]}_{FILE}.root'
        J, rows = P.load_rows(path, NPC, rng)
        t = uproot.open(path)['tree'].arrays([f'jet_tau{k}' for k in (1, 2, 3)], library='np')
        T = np.stack([t[f'jet_tau{k}'][rows] for k in (1, 2, 3)], 1).astype(np.float64)
        perm = rng.permutation(len(rows)); a, b = np.sort(perm[:NPC // 2]), np.sort(perm[NPC // 2:])
        lab.append(sub(J, a)); ho.append(sub(J, b)); taus.append((T[a], T[b]))
    JL, JH = cat(lab), cat(ho); TL = np.concatenate([x[0] for x in taus]); TH = np.concatenate([x[1] for x in taus])
    # shuffle the jet order (arrays are no longer class-ordered: no tie-break artefacts in ranks)
    pl, ph = rng.permutation(len(TL)), rng.permutation(len(TH))
    return P.rebuild(sub(JL, pl)), TL[pl], P.rebuild(sub(JH, ph)), TH[ph]


def d2(J, dev):
    """D2 (beta = 1) from all particles: e2 = sum_{i<j} z_i z_j R_ij, e3 = sum_{i<j<k} z_i z_j z_k R_ij R_ik R_jk, D2 = e3 / e2^3"""
    import torch
    from interp.part_core import kin
    pt, de, dp, _ = kin(J); out = np.zeros(len(pt))
    for s in range(0, len(pt), 500):
        p = torch.from_numpy(pt[s:s + 500]).double().to(dev); e = torch.from_numpy(de[s:s + 500]).double().to(dev)
        f = torch.from_numpy(dp[s:s + 500]).double().to(dev)
        z = p / p.sum(1, keepdim=True)
        dphi = torch.remainder(f[:, :, None] - f[:, None, :] + np.pi, 2 * np.pi) - np.pi
        R = torch.sqrt((e[:, :, None] - e[:, None, :]) ** 2 + dphi ** 2) * (z[:, :, None] > 0) * (z[:, None, :] > 0)
        e2 = 0.5 * torch.einsum('ni,nij,nj->n', z, R, z)
        A = z[:, None, :] * R                                                                          # A_ij = z_j R_ij
        e3 = torch.einsum('ni,nij,nij->n', z, A @ R, A) / 6.0                                          # sum_i z_i sum_jk A_ij R_jk A_ik
        out[s:s + 500] = (e3 / e2.clamp_min(1e-12) ** 3).cpu().numpy()
    return out


def observables(J, T, dev):
    from interp.part_core import kin, jmass, moments
    pt, de, dp, _ = kin(J); m = J['mask'] > 0; mo = moments(pt, de, dp)
    raw = dict(N=m.sum(1).astype(float), pt=J['jet_pt'].astype(float), m=jmass(J).astype(float), mA=np.sqrt(np.maximum(mo['M0'] * mo['M2'], 0)))
    ch = (J['charge'] != 0) & m
    O = dict(N=raw['N'], log_pt=np.log(raw['pt']), log_m=np.log(raw['m'] + 1e-3), log_mA=np.log(raw['mA'] + 1e-9),
             tau21=T[:, 1] / np.maximum(T[:, 0], 1e-9), tau32=T[:, 2] / np.maximum(T[:, 1], 1e-9), D2=d2(J, dev),
             lead_frac=pt.max(1) / pt.sum(1), n_charged=ch.sum(1).astype(float), n_neutral=((J['charge'] == 0) & m).sum(1).astype(float),
             n_lepton=(((J['isElectron'] > 0.5) | (J['isMuon'] > 0.5)) & m).sum(1).astype(float),
             max_abs_d0=np.where(ch, np.abs(J['d0val']), 0).max(1).astype(float))
    return np.stack([O[k] for k in OBS], 1), raw


# ------------------------------------------------------------------------------------------------ network
class Net:
    """ParT with hooks on the two class blocks: records pre-GELU (fc1 output) and post-GELU (GELU output) of the class token,
    applies ops to the post-GELU vector h (n, 512) per block: ('mean', mask (512,) bool, mu (512,)), ('add', vec (512,)),
    ('patch', U (512, k) orthonormal, T (n, k)): h += (T - h U) U^T. tail() re-runs the class blocks from cached inputs."""

    def __init__(self, feat, kind='trained', seed=0):
        import torch
        from interp.part_core import ParTNetwork
        self.t, self.feat, self.kind = torch, feat, kind
        self.net = ParTNetwork(feat, untrained=(kind == 'untrained'), seed=seed); self.m = self.net.model; self.dev = self.net.device
        self.m.eval()
        if kind == 'permuted':
            g = torch.Generator().manual_seed(seed)
            with torch.no_grad():
                for p in self.m.parameters():
                    if p.numel() > 1: p.copy_(p.flatten()[torch.randperm(p.numel(), generator=g).to(p.device)].view_as(p))
        self.ctl = None; self.rec = False; self.capon = False; self.buf = {}
        for b, blk in enumerate(self.m.cls_blocks):
            blk.fc1.register_forward_hook(lambda mod, i, o, b=b: self._pre(b, o))
            blk.act.register_forward_hook(lambda mod, i, o, b=b: self._post(b, o))
        b0 = self.m.cls_blocks[0]; orig = b0.forward             # torch 1.13: no kwargs in pre-hooks, so wrap forward

        def fwd0(x, x_cls=None, padding_mask=None, attn_mask=None):
            if self.capon: self.cx = x.detach(); self.cpm = padding_mask.detach()
            return orig(x, x_cls=x_cls, padding_mask=padding_mask, attn_mask=attn_mask)
        b0.forward = fwd0

    def _pre(self, b, o):
        if self.rec: self.buf[('pre', b)] = o[0].detach().float()

    def _post(self, b, o):
        t = self.t
        if self.rec: self.buf[('post', b)] = o[0].detach().float()
        ops = (self.ctl or {}).get(b)
        if not ops: return None
        h = o[0]
        for op in ops:
            if op[0] == 'mean': h = t.where(op[1][None, :], op[2][None, :], h)
            elif op[0] == 'add': h = h + op[1]
            elif op[0] == 'patch': h = h + (op[2] - h @ op[1]) @ op[1].T
        return h[None]

    def run(self, J, cache=False, bs=BS):
        """full forward: logits (N, 10), activations {(site, b): (N, 512)}, cache [(idx tensor, x, padding mask)]"""
        t = self.t
        from interp.part_core import sub, inputs
        N = len(J['mask']); order = np.argsort(J['mask'].sum(1), kind='stable'); L = np.zeros((N, 10), np.float32)
        A = {(s, b): np.zeros((N, 512), np.float32) for b, s in SITES}; C = []
        for s0 in range(0, N, bs):
            idx = order[s0:s0 + bs]; x, v, m = inputs(sub(J, idx), self.feat)
            self.rec, self.capon, self.ctl = True, cache, None
            with t.no_grad(): out = self.m(*(t.from_numpy(a).to(self.dev) for a in (x, v, m)))
            L[idx] = out.float().cpu().numpy()
            for k in A: A[k][idx] = self.buf[k].cpu().numpy()
            if cache: C.append((t.from_numpy(idx).to(self.dev), self.cx, self.cpm))
        self.rec = self.capon = False; return L, A, C

    def tail(self, C, N, mk=None):
        t = self.t; Lt = t.zeros(N, 10, device=self.dev)
        with t.no_grad():
            for it, x, pm in C:
                self.ctl = mk(it) if mk else None; n = x.shape[1]
                cls = self.m.cls_token.expand(1, n, -1)
                for blk in self.m.cls_blocks: cls = blk(x, x_cls=cls, padding_mask=pm)
                Lt[it] = self.m.fc(self.m.norm(cls).squeeze(0)).float()
        self.ctl = None; return Lt


# ------------------------------------------------------------------------------------------------ atlas statistics
def bh(p):
    p = np.asarray(p, float).ravel(); m = len(p); o = np.argsort(p); q = np.empty(m)
    q[o] = np.minimum.accumulate((p[o] * m / np.arange(1, m + 1))[::-1])[::-1]; return np.minimum(q, 1)


class Scorer:
    """Spearman with the 12 observables and one-vs-rest AUC (as 2 (AUC - 0.5)) for the 10 classes, on one half"""

    def __init__(self, O, y):
        from scipy.stats import rankdata
        self.n = len(y); R = rankdata(O, axis=0); self.ORz = (R - R.mean(0)) / R.std(0)
        self.Y = np.eye(10)[y]; self.ny = self.Y.sum(0)

    def __call__(self, A):
        from scipy.stats import rankdata, norm, t as tdist
        n = self.n; A = np.asarray(A, np.float64); const = A.std(0) < 1e-7 * (1 + np.abs(A).mean(0))
        R = rankdata(A, axis=0); sd = R.std(0); Rz = (R - R.mean(0)) / np.where(sd > 0, sd, 1)
        rho = np.clip(Rz.T @ self.ORz / n, -0.999999, 0.999999)
        auc = (R.T @ self.Y - self.ny * (self.ny + 1) / 2) / (self.ny * (n - self.ny))
        S = np.concatenate([rho, 2 * (auc - 0.5)], 1); S[const] = 0
        pr = 2 * tdist.sf(np.abs(rho) * np.sqrt((n - 2) / (1 - rho ** 2)), n - 2)
        pa = 2 * norm.sf(np.abs(auc - 0.5) / np.sqrt((n + 1) / (12 * self.ny * (n - self.ny))))
        Pv = np.concatenate([pr, pa], 1); Pv[const] = 1; return S, Pv


def label_units(SL, SH, PH):
    """label on the labelling half, held-out signed score v and BH q of that label (BH over all units x 22)"""
    lab = np.abs(SL).argmax(1); sg = np.sign(SL[np.arange(len(lab)), lab]); sg[sg == 0] = 1
    v = sg * SH[np.arange(len(lab)), lab]; q = bh(PH).reshape(PH.shape)[np.arange(len(lab)), lab]
    # family scores on the held-out half: max |S| within a family of related labels (count, mass, pT, shape, leading fraction,
    # lepton, d0, and each class alone); top1 / top2 = best and second-best family, famok = the label's family is the best one
    F = np.stack([np.abs(SH[:, ix]).max(1) for ix in FAMIX], 1); srt = np.sort(F, 1)
    famok = F.argmax(1) == np.array([FAMOF[j] for j in lab])
    return dict(lab=lab, sign=sg, v=v, q=q, top1=srt[:, -1], top2=srt[:, -2], famok=famok, labscore=SL[np.arange(len(lab)), lab])


def wilson(k, n):
    from scipy.stats import norm
    z = norm.ppf(0.975); p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def haar(n, seed):
    r = np.random.default_rng(seed); Z = r.standard_normal((n, n)); Qm, Rm = np.linalg.qr(Z); return Qm * np.sign(np.diag(Rm))[None, :]


def atlas_model(AL, AH, scL, scH, rot=True):
    """per variant ('neuron', 'rot_raw<s>', 'rot_std<s>') and site: label_units output"""
    out = {}
    bases = [('neuron', None, None)]
    if rot: bases += [(f'rot_{kind}{s}', kind, s) for s in range(NROT) for kind in ('raw', 'std')]
    Qs = {s: haar(512, s) for s in range(NROT)} if rot else {}
    for name, kind, s in bases:
        out[name] = {}
        for b, site in SITES:
            a, h = AL[(site, b)].astype(np.float64), AH[(site, b)].astype(np.float64)
            if kind == 'std':
                mu, sd = a.mean(0), a.std(0); sd = np.where(sd > 1e-8, sd, 1); a, h = (a - mu) / sd, (h - mu) / sd
            if kind is not None: a, h = a @ Qs[s], h @ Qs[s]
            SL, _ = scL(a); SH, PH = scH(h)
            out[name][f'b{b + 1}_{site}'] = label_units(SL, SH, PH)
    return out


def clean_stats(U, thr):
    """U: {site: label_units}; pooled over b1, b2 per site type"""
    res = {}
    for st in ('pre', 'post'):
        v = np.concatenate([U[f'b{b}_{st}']['v'] for b in (1, 2)]); q = np.concatenate([U[f'b{b}_{st}']['q'] for b in (1, 2)])
        lab = np.concatenate([U[f'b{b}_{st}']['lab'] for b in (1, 2)])
        t1 = np.concatenate([U[f'b{b}_{st}']['top1'] for b in (1, 2)]); t2 = np.concatenate([U[f'b{b}_{st}']['top2'] for b in (1, 2)])
        fo = np.concatenate([U[f'b{b}_{st}']['famok'] for b in (1, 2)])
        r = {}
        for tn, tv in (('thr_untrained', thr[st]), ('0.2', 0.2), ('0.3', 0.3), ('0.5', 0.5)):
            c = (v >= tv) & (q < 0.05); k = int(c.sum())
            r[tn] = dict(thr=float(tv), n_clean=k, frac=k / len(v), ci=wilson(k, len(v)),
                         frac_selective=float((c & fo & (t1 - t2 >= 0.1)).mean()), ci_selective=wilson(int((c & fo & (t1 - t2 >= 0.1)).sum()), len(v)),
                         frac_selective_02=float((c & fo & (t1 - t2 >= 0.2)).mean()),
                         per_block=[float(((U[f'b{b}_{st}']['v'] >= tv) & (U[f'b{b}_{st}']['q'] < 0.05)).mean()) for b in (1, 2)],
                         label_counts={(OBS + ['cls_' + x for x in CLN])[j]: int((c & (lab == j)).sum()) for j in range(22) if (c & (lab == j)).any()})
        r['v_median'] = float(np.median(v)); r['v_q90'] = float(np.quantile(v, 0.9)); res[st] = r
    return res


# ------------------------------------------------------------------------------------------------ steering statistics (torch)
def lodds_all(L):
    t = __import__('torch'); n = L.shape[0]; E = L[:, None, :].expand(n, 10, 10).clone()
    E[:, t.arange(10), t.arange(10)] = -float('inf'); return L - t.logsumexp(E, -1)


class Boot:
    def __init__(self, N, nb, rng, dev, y):
        import torch
        self.t = torch; self.W = torch.from_numpy(P.boot_W(N, nb, rng)).to(dev)
        self.Wc = {c: torch.from_numpy(P.boot_W(int((y == c).sum()), nb, rng)).to(dev) for c in range(10)}

    def ms(self, x, c=None):
        W = self.W if c is None else self.Wc[c]; bs = (W @ x.float()).cpu().numpy(); v = float(x.float().mean())
        return dict(v=v, ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], p_boot=P.p_two(bs))

    def ratio(self, x, y):
        bx = (self.W @ x.float()).cpu().numpy(); by = (self.W @ y.float()).cpu().numpy(); r = bx / by
        return dict(v=float(x.float().mean() / y.float().mean()), ci=[float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))])


def main():
    import torch
    from interp.part_core import CL, sample
    global CLN; CLN = list(CL)
    P.OUTD.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(SEED); torch.manual_seed(SEED)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    R = dict(node='P40', seed=SEED, smoke=SMOKE, file=f'test_20M <sample>_{FILE}.root', NPC=NPC, NB=NB, NROT=NROT, NSTEER=NSTEER, KS=KS, DIL=DIL)
    JL, TL, JH, TH = load(rng); yL, yH = JL['label'].astype(int), JH['label'].astype(int)
    OL, rawL = observables(JL, TL, dev); OH, rawH = observables(JH, TH, dev)
    R['sample'] = dict(n_label=len(yL), n_heldout=len(yH), per_class_heldout=np.bincount(yH, minlength=10).tolist(),
                       obs_median_heldout={k: float(np.median(OH[:, j])) for j, k in enumerate(OBS)},
                       obs_corr_spearman_heldout=np.round(__import__('scipy.stats', fromlist=['spearmanr']).spearmanr(OH).correlation, 3).tolist())
    log('sample', R['sample']['n_label'], R['sample']['n_heldout'], R['sample']['obs_median_heldout'])
    scL, scH = Scorer(OL, yL), Scorer(OH, yH)
    J1 = P.rebuild(sample(per_class=1000)); y1 = J1['label'].astype(int)
    # dilated held-out jets (the P1 matched dilation: every particle's deta, dphi x 1.05 at fixed pT and mass)
    from interp.part_core import kin, build, jmass
    pt, de, dp, ms_ = kin(JH); JD = build(JH, pt, de * DIL, dp * DIL, ms_)
    R['dilation'] = dict(dlog_m_median=float(np.median(np.log(jmass(JD) + 1e-3) - np.log(jmass(JH) + 1e-3))))
    ATL, models = {}, {}
    # ---------------- controls first (thresholds)
    for feat in ('full', 'kin'):
        for kind, seed in (('untrained', 0), ('permuted', 5)):
            name = f'ParT_{feat}_{kind}'; M = Net(feat, kind, seed)
            LL, AL, _ = M.run(JL); LH, AH, _ = M.run(JH)
            U = atlas_model(AL, AH, scL, scH, rot=False)['neuron']; ATL[name] = U
            models[name] = dict(acc_heldout=float((LH.argmax(1) == yH).mean()))
            log(f'{name}: acc {models[name]["acc_heldout"]:.4f}; v median post {np.median(np.concatenate([U["b1_post"]["v"], U["b2_post"]["v"]])):.3f}')
            del M; torch.cuda.empty_cache()
    thr = {feat: {st: float(np.quantile(np.concatenate([ATL[f'ParT_{feat}_untrained'][f'b{b}_{st}']['v'] for b in (1, 2)]), 0.95)) for st in ('pre', 'post')}
           for feat in ('full', 'kin')}
    R['thresholds'] = thr; log('thresholds (95th pct of untrained v)', thr)
    for feat in ('full', 'kin'):
        for kind in ('untrained', 'permuted'):
            a_ = models[f'ParT_{feat}_{kind}']['atlas'] = clean_stats(ATL[f'ParT_{feat}_{kind}'], thr[feat])
            log(f'  ParT_{feat}_{kind}: clean pre {a_["pre"]["thr_untrained"]["frac"]:.3f} post {a_["post"]["thr_untrained"]["frac"]:.3f}; '
                f'selective pre {a_["pre"]["thr_untrained"]["frac_selective"]:.3f} post {a_["post"]["thr_untrained"]["frac_selective"]:.3f}')
    # ---------------- trained models
    for feat in ('full', 'kin'):
        name = f'ParT_{feat}'; t0 = time.time(); M = Net(feat); res = {}
        L1, _, _ = M.run(J1); res['acc_P1_sample'] = float((L1.argmax(1) == y1).mean())
        LL, AL, _ = M.run(JL); LH, AH, C = M.run(JH, cache=True)
        res['acc_label'] = float((LL.argmax(1) == yL).mean()); res['acc_heldout'] = float((LH.argmax(1) == yH).mean())
        N = len(yH); L0 = M.tail(C, N); res['tail_max_abs_diff'] = float(np.abs(L0.cpu().numpy() - LH).max())
        log(f'==== {name}: acc P1 sample {res["acc_P1_sample"]:.4f} labelling {res["acc_label"]:.4f} held-out {res["acc_heldout"]:.4f}; tail |dL| {res["tail_max_abs_diff"]:.2e}')
        A = atlas_model(AL, AH, scL, scH, rot=True); ATL[name] = A['neuron']
        at = {k: clean_stats(U, thr[feat]) for k, U in A.items()}
        res['atlas'] = summarize_atlas(at)
        for st in ('pre', 'post'):
            s = res['atlas'][st]['thr_untrained']
            log(f'  atlas {st}: neuron {s["neuron"]["frac"]:.3f} {np.round(s["neuron"]["ci"], 3).tolist()} | rot_raw mean {s["rot_raw"]["mean"]:.3f} '
                f'[{s["rot_raw"]["min"]:.3f}, {s["rot_raw"]["max"]:.3f}] p {s["rot_raw"]["p_emp"]:.3f} | rot_std mean {s["rot_std"]["mean"]:.3f} '
                f'[{s["rot_std"]["min"]:.3f}, {s["rot_std"]["max"]:.3f}] p {s["rot_std"]["p_emp"]:.3f}; labels {s["neuron"]["label_counts"]}')
            log(f'  selective {st}: neuron {s["neuron"]["frac_selective"]:.3f} {np.round(s["neuron"]["ci_selective"], 3).tolist()} | rot_raw {s["rot_raw"]["frac_selective_mean"]:.3f} '
                f'(max {max(s["rot_raw"]["frac_selective_all"]):.3f}, p {s["rot_raw"]["p_emp_selective"]:.3f}) | rot_std {s["rot_std"]["frac_selective_mean"]:.3f} '
                f'(max {max(s["rot_std"]["frac_selective_all"]):.3f}, p {s["rot_std"]["p_emp_selective"]:.3f}); gap>=0.2: neuron {s["neuron"]["frac_selective_02"]:.3f} '
                f'rot_raw {s["rot_raw"]["frac_selective_02_mean"]:.3f} rot_std {s["rot_std"]["frac_selective_02_mean"]:.3f}')
        res['top_units'] = top_units(A['neuron'], AH, OH, rawH, yH, thr[feat]['post'])
        for u in res['top_units'][:12]: log('  top', u['unit'], u['label'], f'v {u["v"]:.3f}', 'class means', np.round(u['class_mean'], 2).tolist())
        # ---------------- steering, ablation, calibration, circuits
        Q0 = haar(512, 0)
        res.update(write(M, C, N, L0, A['neuron'], AL, AH, OL, yL, yH, thr[feat]['post'], Q0, rng, dev))
        LD, AD, _ = M.run(JD)
        res['calibration'] = calibrate(M, C, N, L0, LD, A['neuron'], AL, AH, AD, OL, yL, yH, thr[feat]['post'], rng, dev)
        res['circuits'] = circuits(M, C, N, L0, A['neuron'], AL, yH, thr[feat]['post'], rng, dev)
        res['runtime_s'] = time.time() - t0; models[name] = res; R['models'] = models
        FN.write_text(json.dumps(P.strip(R), indent=1)); log(f'  wrote {FN} ({res["runtime_s"]:.0f} s)')
        del M, C; torch.cuda.empty_cache()
    R['models'] = models; R['runtime_s'] = time.time() - P.T0
    FN.write_text(json.dumps(P.strip(R), indent=1)); log(f'done; wrote {FN}')
    try:
        plot(R, ATL, FIG)
    except Exception as e:                                                     # the figure is optional; the JSON is the result
        log('plot failed', repr(e))


def summarize_atlas(at):
    out = {}
    for st in ('pre', 'post'):
        out[st] = {}
        for tn in ('thr_untrained', '0.2', '0.3', '0.5'):
            nf = at['neuron'][st][tn]['frac']; d = dict(neuron=at['neuron'][st][tn])
            for kind in ('raw', 'std'):
                fr = np.array([at[f'rot_{kind}{s}'][st][tn]['frac'] for s in range(NROT)])
                d[f'rot_{kind}'] = dict(fracs=fr.tolist(), mean=float(fr.mean()), min=float(fr.min()), max=float(fr.max()),
                                        ratio=float(nf / fr.mean()) if fr.mean() > 0 else None,
                                        p_emp=float((1 + (fr >= nf).sum()) / (1 + len(fr))),
                                        frac_selective_mean=float(np.mean([at[f'rot_{kind}{s}'][st][tn]['frac_selective'] for s in range(NROT)])),
                                        frac_selective_all=[at[f'rot_{kind}{s}'][st][tn]['frac_selective'] for s in range(NROT)],
                                        p_emp_selective=float((1 + sum(at[f'rot_{kind}{s}'][st][tn]['frac_selective'] >= at['neuron'][st][tn]['frac_selective'] for s in range(NROT))) / (1 + NROT)),
                                        frac_selective_02_mean=float(np.mean([at[f'rot_{kind}{s}'][st][tn]['frac_selective_02'] for s in range(NROT)])))
            out[st][tn] = d
        out[st]['v_median'] = dict(neuron=at['neuron'][st]['v_median'], rot_raw=float(np.mean([at[f'rot_raw{s}'][st]['v_median'] for s in range(NROT)])),
                                   rot_std=float(np.mean([at[f'rot_std{s}'][st]['v_median'] for s in range(NROT)])))
    return out


def labname(j):
    return (OBS + ['cls_' + x for x in CLN])[j]


def top_units(U, AH, OH, rawH, yH, thr, k=20):
    rows = []
    for b in (1, 2):
        u = U[f'b{b}_post']
        for i in range(512): rows.append((u['v'][i], b, i))
    rows.sort(key=lambda r: -r[0]); out = []
    for v, b, i in rows[:k]:
        u = U[f'b{b}_post']; j = int(u['lab'][i]); a = AH[('post', b - 1)][:, i]
        e = dict(unit=f'b{b}n{i}', block=b, index=i, label=labname(j), sign=float(u['sign'][i]), v=float(v), q=float(u['q'][i]),
                 clean=bool(v >= thr and u['q'][i] < 0.05), top2_gap=float(u['top1'][i] - u['top2'][i]),
                 act_mean=float(a.mean()), act_sd=float(a.std()), class_mean=[float(a[yH == c].mean()) for c in range(10)])
        # profile against the jet mass (GeV) and against the label quantity (deciles, in physical units)
        mb = np.arange(0, 301, 20); dig = np.digitize(rawH['m'], mb)
        e['mass_profile'] = dict(bin_lo_GeV=mb[:-1].tolist(), mean=[float(a[dig == q].mean()) if (dig == q).sum() >= 20 else None for q in range(1, len(mb))])
        if j < len(OBS):
            o = OH[:, j]; qe = np.quantile(o, np.linspace(0, 1, 11)); dq = np.clip(np.digitize(o, qe[1:-1]), 0, 9)
            phys = {'log_pt': np.exp, 'log_m': np.exp, 'log_mA': np.exp}.get(OBS[j], lambda x: x)
            e['label_profile'] = dict(decile_centre=[float(phys(np.median(o[dq == q]))) for q in range(10)], mean=[float(a[dq == q].mean()) for q in range(10)])
        out.append(e)
    return out


def target_of(U, OL, yL, b, i):
    """target class and predicted sign for unit i of block b (see docstring)"""
    u = U[f'b{b}_post']; j = int(u['lab'][i]); sg = float(u['sign'][i])
    if j >= len(OBS): return j - len(OBS), sg
    o = OL[:, j]; from scipy.stats import rankdata
    r = rankdata(o); n = len(o); aucs = np.array([(r[yL == c].sum() - (yL == c).sum() * ((yL == c).sum() + 1) / 2) / ((yL == c).sum() * (n - (yL == c).sum())) for c in range(10)])
    return int(np.argmax(sg * (aucs - 0.5))), 1.0


def write(M, C, N, L0, U, AL, AH, OL, yL, yH, thr, Q0, rng, dev):
    import torch
    t = torch; l0 = lodds_all(L0); c0 = L0.argmax(1); B = Boot(N, NB, rng, dev, yH); yt = t.from_numpy(yH).to(dev)
    mu = {b: AL[('post', b - 1)].mean(0) for b in (1, 2)}; sd = {b: AL[('post', b - 1)].std(0) for b in (1, 2)}
    Qt = t.from_numpy(Q0.astype(np.float32)).to(dev)
    tgt = {(b, i): target_of(U, OL, yL, b, i) for b in (1, 2) for i in range(512)}
    clean = [(float(U[f'b{b}_post']['v'][i]), b, i) for b in (1, 2) for i in range(512) if U[f'b{b}_post']['v'][i] >= thr and U[f'b{b}_post']['q'][i] < 0.05]
    pick = np.sort(rng.choice(len(clean), min(NSTEER, len(clean)), replace=False)); sel = [clean[k] for k in pick]   # random subset, not top-v
    log(f'  write: {len(clean)} clean post-GELU units, steering {len(sel)}')

    def run_ops(b, ops):
        return M.tail(C, N, lambda it: {b - 1: ops(it)})

    def effect(Lx, c, s):
        d = lodds_all(Lx) - l0; dc = d[:, c]; spec = dc - (d.sum(1) - dc) / 9; c1 = Lx.argmax(1)
        return d, dc, spec, c1

    def steer_dir(b, vec, sdv, c, s):
        """vec (512,) unit-norm direction in post-GELU space; sdv its activation s.d.; returns per-level stats"""
        out = {}
        for k in KS:
            add = (k * sdv) * vec
            Lx = run_ops(b, lambda it: [('add', add)]); _, dc, spec, c1 = effect(Lx, c, s)
            e = dict(dl=float(dc.mean()), spec=float(spec.mean()), called_c_change=float(((c1 == c).float() - (c0 == c).float()).mean()),
                     calls_changed=float((c1 != c0).float().mean()))
            if abs(k) == 2:
                e['dl_boot'] = B.ms(dc); e['spec_boot'] = B.ms(spec)
                e['mcnemar'] = P.mcnemar((c0 == c).cpu().numpy(), (c1 == c).cpu().numpy())
            out[str(k)] = e
        return out

    def ablate_dir(b, vec, muv, c):
        U_ = vec[:, None]
        Lx = run_ops(b, lambda it: [('patch', U_, t.full((len(it), 1), muv, device=dev))]); _, dc, spec, c1 = effect(Lx, c, 1.0)
        m = yt == c; dcc = dc[m]
        return dict(dl_all=B.ms(dc), dl_class=B.ms(dcc, c), spec=B.ms(spec), calls_changed=float((c1 != c0).float().mean()),
                    mcnemar_correct_class=P.mcnemar((c0[m] == c).cpu().numpy(), (c1[m] == c).cpu().numpy()))

    rows = []
    for v, b, i in sel:
        c, s = tgt[(b, i)]; e_i = t.zeros(512, device=dev); e_i[i] = 1.0
        r = dict(unit=f'b{b}n{i}', block=b, index=i, label=labname(int(U[f'b{b}_post']['lab'][i])), v=v, target=CLN[c], c=c, s=s, sd=float(sd[b][i]))
        r['steer'] = steer_dir(b, e_i, float(sd[b][i]), c, s); r['ablate'] = ablate_dir(b, e_i, float(mu[b][i]), c)
        # control (a): rotated direction (same index column of Q0)
        q = Qt[:, i]; pa = AL[('post', b - 1)] @ Q0[:, i]
        r['rot'] = dict(steer=steer_dir(b, q, float(pa.std()), c, s), ablate=ablate_dir(b, q, float(pa.mean()), c), sd=float(pa.std()))
        # control (b): matched-s.d. unit with another label and another target class
        cand = [j for j in range(512) if j != i and sd[b][j] > 1e-6 and tgt[(b, j)][0] != c and U[f'b{b}_post']['lab'][j] != U[f'b{b}_post']['lab'][i]]
        j = min(cand, key=lambda j: abs(np.log(sd[b][j] / max(sd[b][i], 1e-9))))
        e_j = t.zeros(512, device=dev); e_j[j] = 1.0
        r['matched'] = dict(unit=f'b{b}n{j}', label=labname(int(U[f'b{b}_post']['lab'][j])), target=CLN[tgt[(b, j)][0]], sd=float(sd[b][j]),
                            steer=steer_dir(b, e_j, float(sd[b][j]), c, s), ablate=ablate_dir(b, e_j, float(mu[b][j]), c))
        rows.append(r)
        if len(rows) <= 15 or len(rows) % 25 == 0:
            st = r['steer']['2']; log(f'  {r["unit"]} {r["label"]} -> {r["target"]} s {s:+.0f}: +2sd dl {st["dl"]:+.3f} {np.round(st["dl_boot"]["ci"], 3).tolist()} '
                                       f'-2sd {r["steer"]["-2"]["dl"]:+.3f}; rot {r["rot"]["steer"]["2"]["dl"]:+.3f}; matched {r["matched"]["steer"]["2"]["dl"]:+.3f}; '
                                       f'ablate class dl {r["ablate"]["dl_class"]["v"]:+.3f}')
    return dict(write=summarize_write(rows), write_units=rows)


def summarize_write(rows):
    from scipy.stats import wilcoxon, binomtest
    if not rows: return dict(n=0)
    out = dict(n=len(rows))

    def dirok(st, s):
        e = st['2']; lo, hi = e['dl_boot']['ci']
        return (s * e['dl'] > 0) and ((lo > 0) if s > 0 else (hi < 0)) and (s * st['-2']['dl'] < 0)
    for key, get in (('neuron', lambda r: r['steer']), ('rot', lambda r: r['rot']['steer']), ('matched', lambda r: r['matched']['steer'])):
        ok = np.array([dirok(get(r), r['s']) for r in rows]); sgn = np.array([r['s'] * get(r)['2']['dl'] for r in rows])
        sp = np.array([r['s'] * get(r)['2']['spec'] for r in rows]); mag = np.array([abs(get(r)['2']['dl']) for r in rows])
        pm = np.array([get(r)['2']['mcnemar']['p'] for r in rows])
        out[key] = dict(frac_direction_correct=float(ok.mean()), ci=wilson(int(ok.sum()), len(ok)), signed_dl2_median=float(np.median(sgn)),
                        signed_dl2_mean=float(sgn.mean()), signed_spec2_median=float(np.median(sp)), abs_dl2_median=float(np.median(mag)),
                        frac_mcnemar_bh=float((bh(pm) < 0.05).mean()),
                        by_label_type={lt: float(ok[[(r['label'].startswith('cls_')) == (lt == 'class') for r in rows]].mean()) if any((r['label'].startswith('cls_')) == (lt == 'class') for r in rows) else None for lt in ('class', 'observable')},
                        signed_dl_by_k={str(k): float(np.median([r['s'] * get(r)[str(k)]['dl'] for r in rows])) for k in KS})
        out[key]['_sgn'] = sgn
    for ctl in ('rot', 'matched'):
        d = out['neuron']['_sgn'] - out[ctl]['_sgn']
        out[f'neuron_vs_{ctl}'] = dict(median_diff=float(np.median(d)), frac_neuron_larger=float((d > 0).mean()),
                                       wilcoxon_p=float(wilcoxon(out['neuron']['_sgn'], out[ctl]['_sgn']).pvalue) if len(d) > 1 and np.any(d != 0) else None,
                                       sign_test_p=float(binomtest(int((d > 0).sum()), int((d != 0).sum()), 0.5).pvalue) if np.any(d != 0) else None)
    ab = np.array([r['ablate']['dl_class']['v'] * r['s'] for r in rows]); abr = np.array([r['rot']['ablate']['dl_class']['v'] * r['s'] for r in rows])
    out['ablation'] = dict(signed_dl_class_median=float(np.median(ab)), frac_lowers_target=float((ab < 0).mean()),
                           rot_signed_dl_class_median=float(np.median(abr)), rot_frac_lowers=float((abr < 0).mean()),
                           frac_mcnemar_bh=float((bh([r['ablate']['mcnemar_correct_class']['p'] for r in rows]) < 0.05).mean()),
                           max_calls_changed=float(max(r['ablate']['calls_changed'] for r in rows)))
    return out


def calibrate(M, C, N, L0, LD, U, AL, AH, AD, OL, yL, yH, thr, rng, dev):
    import torch
    t = torch; l0 = lodds_all(L0); ld = lodds_all(t.from_numpy(LD).to(dev)); dD = ld - l0; B = Boot(N, NB, rng, dev, yH)
    mass = [(b, i) for b in (1, 2) for i in range(512) if U[f'b{b}_post']['v'][i] >= thr and U[f'b{b}_post']['q'][i] < 0.05
            and labname(int(U[f'b{b}_post']['lab'][i])) in MASSLAB]
    out = dict(n_mass_units=len(mass), dilation_dl={CLN[c]: B.ms(dD[:, c]) for c in range(10)},
               dilation_calls_changed=float((ld.argmax(1) != l0.argmax(1)).float().mean()))
    log(f'  calibration: {len(mass)} mass-labelled units; dilation dl ' + ', '.join(f'{CLN[c]} {dD[:, c].mean():+.3f}' for c in range(10)))
    AHt = {b: t.from_numpy(AH[('post', b - 1)]).to(dev) for b in (1, 2)}; ADt = {b: t.from_numpy(AD[('post', b - 1)]).to(dev) for b in (1, 2)}
    units = []
    for b, i in mass:
        c, s = target_of(U, OL, yL, b, i); sdv = float(AL[('post', b - 1)][:, i].std())
        e = t.zeros(512, 1, device=dev); e[i, 0] = 1
        Lx = M.tail(C, N, lambda it: {b - 1: [('patch', e, ADt[b][it, i:i + 1])]}); dP = lodds_all(Lx) - l0
        units.append(dict(unit=f'b{b}n{i}', label=labname(int(U[f'b{b}_post']['lab'][i])), target=CLN[c], s=s,
                          shift_sd=float((AD[('post', b - 1)][:, i] - AH[('post', b - 1)][:, i]).mean() / sdv),
                          dl_dil=float(dD[:, c].mean()), dl_patch=B.ms(dP[:, c]), R=B.ratio(dP[:, c], dD[:, c])))
    out['units'] = units
    for u in units[:20]: log(f'    {u["unit"]} {u["label"]} -> {u["target"]}: shift {u["shift_sd"]:+.3f} sd; dl dil {u["dl_dil"]:+.3f} patch {u["dl_patch"]["v"]:+.4f} R {u["R"]["v"]:+.3f} {np.round(u["R"]["ci"], 3).tolist()}')

    def joint(sets):
        ops = {}
        for b in (1, 2):
            ix = [i for bb, i in sets if bb == b]
            if not ix: continue
            Ub = t.zeros(512, len(ix), device=dev)
            for k, i in enumerate(ix): Ub[i, k] = 1
            ops[b] = Ub
        return ops

    def joint_dirs(Ud):     # Ud {b: (512, k) orthonormal}
        Lx = M.tail(C, N, lambda it: {b - 1: [('patch', Ud[b], ADt[b][it] @ Ud[b])] for b in Ud}); return lodds_all(Lx) - l0
    if mass:
        dJ = joint_dirs(joint(mass)); cls = [CLN.index(x) for x in ('Wqq', 'Zqq', 'QCD', 'Tbqq')] + [int(dD.mean(0).abs().argmax())]
        out['joint'] = {CLN[c]: dict(dl_patch=B.ms(dJ[:, c]), R=B.ratio(dJ[:, c], dD[:, c])) for c in sorted(set(cls))}
        rnd = []
        for r_ in range(NRCAL):
            Qr = haar(512, 100 + r_); Ud = {}
            for b in (1, 2):
                k = sum(1 for bb, _ in mass if bb == b)
                if k: Ud[b] = t.from_numpy(Qr[:, rng.choice(512, k, replace=False)].astype(np.float32)).to(dev)
            dR = joint_dirs(Ud); rnd.append({CLN[c]: float(dR[:, c].mean() / dD[:, c].mean()) for c in sorted(set(cls))})
        out['joint_random_dirs'] = {k: dict(mean=float(np.mean([x[k] for x in rnd])), min=float(np.min([x[k] for x in rnd])), max=float(np.max([x[k] for x in rnd]))) for k in rnd[0]}
        log('  joint mass units R: ' + ', '.join(f'{k} {v["R"]["v"]:+.3f} {np.round(v["R"]["ci"], 3).tolist()}' for k, v in out['joint'].items())
            + ' | random dirs: ' + ', '.join(f'{k} {v["mean"]:+.3f} [{v["min"]:+.3f}, {v["max"]:+.3f}]' for k, v in out['joint_random_dirs'].items()))
    return out


def circuits(M, C, N, L0, U, AL, yH, thr, rng, dev):
    import torch
    t = torch; mu = {b: t.from_numpy(AL[('post', b - 1)].mean(0)).to(dev) for b in (1, 2)}; yt = t.from_numpy(yH).to(dev); B = Boot(N, NB, rng, dev, yH)

    def ev(sets, keep):
        ops = {}
        for b in (1, 2):
            m = t.zeros(512, dtype=t.bool, device=dev)
            for bb, i in sets:
                if bb == b: m[i] = True
            if keep: m = ~m
            if m.any(): ops[b - 1] = [('mean', m, mu[b])]
        Lx = M.tail(C, N, lambda it: ops); c1 = Lx.argmax(1); ok = (c1 == yt).float()
        conf = np.zeros((10, 10))
        for a, p in zip(yH, c1.cpu().numpy()): conf[a, p] += 1
        conf = conf / conf.sum(1, keepdims=True)
        return dict(acc10=B.ms(ok), recall=np.diag(conf).tolist(), confusion=np.round(conf, 4).tolist())
    out = dict(intact=ev([], False), all_at_mean=ev([], True))
    log(f'  circuits: intact acc {out["intact"]["acc10"]["v"]:.4f}; all 1,024 post-GELU units at mean {out["all_at_mean"]["acc10"]["v"]:.4f}')
    per = {}
    for c in range(10):
        sets = [(b, i) for b in (1, 2) for i in range(512) if U[f'b{b}_post']['v'][i] >= thr and U[f'b{b}_post']['q'][i] < 0.05
                and int(U[f'b{b}_post']['lab'][i]) == len(OBS) + c]
        if not sets: continue
        e = dict(units=[f'b{b}n{i}' for b, i in sets], n=len(sets), ablate=ev(sets, False), keep_only=ev(sets, True))
        ra, rk = [], []
        for _ in range(NRSET):
            rs = [(b, int(i)) for b in (1, 2) for i in rng.choice(512, sum(1 for bb, _ in sets if bb == b), replace=False)]
            a, k = ev(rs, False), ev(rs, True); ra.append((a['acc10']['v'], a['recall'][c])); rk.append((k['acc10']['v'], k['recall'][c]))
        ra, rk = np.array(ra), np.array(rk)
        e['random_ablate'] = dict(acc_mean=float(ra[:, 0].mean()), acc_min=float(ra[:, 0].min()), recall_mean=float(ra[:, 1].mean()), recall_min=float(ra[:, 1].min()),
                                  p_recall=float((1 + (ra[:, 1] <= e['ablate']['recall'][c]).sum()) / (1 + len(ra))))
        e['random_keep'] = dict(acc_mean=float(rk[:, 0].mean()), acc_max=float(rk[:, 0].max()), recall_mean=float(rk[:, 1].mean()), recall_max=float(rk[:, 1].max()),
                                p_recall=float((1 + (rk[:, 1] >= e['keep_only']['recall'][c]).sum()) / (1 + len(rk))))
        per[CLN[c]] = e
        log(f'  class {CLN[c]} ({len(sets)} units): ablate acc {e["ablate"]["acc10"]["v"]:.4f} recall {out["intact"]["recall"][c]:.3f} -> {e["ablate"]["recall"][c]:.3f} '
            f'(random {e["random_ablate"]["recall_mean"]:.3f}, p {e["random_ablate"]["p_recall"]:.3f}); keep-only acc {e["keep_only"]["acc10"]["v"]:.4f} '
            f'recall {e["keep_only"]["recall"][c]:.3f} (random {e["random_keep"]["recall_mean"]:.3f}, max {e["random_keep"]["recall_max"]:.3f}, p {e["random_keep"]["p_recall"]:.3f})')
    out['per_class'] = per
    return out


# ------------------------------------------------------------------------------------------------ figure
def plot(R, ATL, fig):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    f, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    a = ax[0]; xs = 0
    for feat in ('full', 'kin'):
        m = R['models'][f'ParT_{feat}']['atlas']['post']['thr_untrained']
        a.bar(xs, m['neuron']['frac'], color='C0', width=0.6); a.errorbar(xs, m['neuron']['frac'], yerr=[[m['neuron']['frac'] - m['neuron']['ci'][0]], [m['neuron']['ci'][1] - m['neuron']['frac']]], color='k', capsize=3)
        a.scatter(np.full(NROT, xs + 1), m['rot_raw']['fracs'], color='C1', s=14); a.scatter(np.full(NROT, xs + 2), m['rot_std']['fracs'], color='C2', s=14)
        a.bar(xs + 3, R['models'][f'ParT_{feat}_untrained']['atlas']['post']['thr_untrained']['frac'], color='C7', width=0.6)
        a.bar(xs + 4, R['models'][f'ParT_{feat}_permuted']['atlas']['post']['thr_untrained']['frac'], color='C5', width=0.6)
        xs += 6
    a.set_xticks([0, 1, 2, 3, 4, 6, 7, 8, 9, 10]); a.set_xticklabels(['full\nneurons', 'rot\nraw', 'rot\nstd', 'untr.', 'perm.'] * 2, fontsize=7)
    a.set_ylabel('fraction of 1,024 post-GELU units that are clean'); a.set_title('(a) readable units: neurons vs rotated directions\n(left ParT_full, right ParT_kin)', fontsize=9)
    a = ax[1]; data, labs = [], []
    for feat in ('full', 'kin'):
        rows = R['models'][f'ParT_{feat}'].get('write_units', [])
        for key, nm in (('steer', 'neuron'), ('rot', 'rotated'), ('matched', 'matched')):
            g = (lambda r: r['steer']) if key == 'steer' else (lambda r, k=key: r[k]['steer'])
            data.append([r['s'] * g(r)['2']['dl'] for r in rows] or [0]); labs.append(f'{feat}\n{nm}')
    a.boxplot(data, showfliers=False); a.axhline(0, color='k', lw=0.6); a.set_xticks(range(1, len(labs) + 1)); a.set_xticklabels(labs, fontsize=7)
    a.set_ylabel('signed change of target-class log-odds at +2 s.d.'); a.set_title('(b) steering one unit by +2 s.d.: target class moves\nin the predicted direction (> 0)', fontsize=9)
    a = ax[2]; tu = [u for u in R['models']['ParT_full']['top_units'] if u['label'] in MASSLAB]
    if tu:
        u = tu[0]; lo = np.array(u['mass_profile']['bin_lo_GeV']) + 10; mv = np.array([np.nan if x is None else x for x in u['mass_profile']['mean']], float)
        a.plot(lo, mv, 'o-', label=f'{u["unit"]} (label {u["label"]}, v = {u["v"]:.2f})')
        a.set_xlabel('jet invariant mass [GeV]'); a.set_ylabel('mean post-GELU activation'); a.legend(fontsize=7)
        a.set_title('(c) ParT_full: most readable mass-labelled unit\n(held-out jets, all classes)', fontsize=9)
    f.tight_layout(); f.savefig(fig, dpi=130); log(f'wrote {fig}')


if __name__ == '__main__':
    main()
