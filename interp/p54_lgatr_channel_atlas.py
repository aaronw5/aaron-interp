"""P54 (loop C, round C12, exploratory; new claim C25): L-GATr equivariant-MLP channel atlas and literal channel circuits.

Question. In every L-GATr block the MLP is   x -> GeometricBilinear -> gated GELU -> EquiLinear.  The gated GELU acts on each of
the 32 hidden multivector channels separately: channel c (a 16-component multivector x_c) becomes GELU(x_c,0) * x_c, the gate
being that channel's own scalar (grade-0) part. Its 64 hidden auxiliary scalars get a plain GELU. Does this per-channel gating
make the channel basis privileged, i.e. do single channels read as one physics observable or one class more often than the
directions of a random grade-preserving rotation of the same hidden space (an equally valid basis: a rotation undone before the
next EquiLinear leaves the network unchanged), and do small literal channel circuits reproduce the class calls?
ParT's class-token MLP neurons were no cleaner than random rotations (P40: selective fraction 0.488 vs 0.49-0.50).

Terms. Multivector components: 0 scalar (grade 0), 1-4 four-vector (t, x, y, z; grade 1), 5-10 bivector, 11-14 axial vector,
15 pseudoscalar. Tokens of a jet: position 0 global token (the classifier reads its output), 1-2 beam / time reference
("spurion") tokens, 3.. particles (pT order).

Model: L-GATr final checkpoint model_run0.pt (our run anrunw-ai-c-p5-lgatr2, authors' jctagging config, 12 blocks).
Controls: untrained L-GATr (trained run's config, torch seed 5454, no weights loaded); channel-permuted L-GATr (trained weights,
  in every block the 32 hidden multivector channels and the 64 hidden scalars are permuted (seed 5454 + block) between the gated
  GELU and the read-out EquiLinear, so each channel is read out by another channel's weights); 10 random rotations per block.
Rotations ('grade-preserving'): per block, an independent Haar orthogonal 32 x 32 matrix R_g for each grade g mixes the 32
  channels' grade-g components (the same R_1 acts on t, x, y, z, so Lorentz structure is kept); a Haar 64 x 64 matrix mixes the
  auxiliary scalars. 'raw' = applied to the activations as they are; 'std' = applied after scaling each channel to unit size on
  sample 1 (P15v / P40 showed the comparison can depend on this choice).
Sample: JetClass test_20M <class>_106.root (val_5M has only _120-_124, all used in loop C; test_20M _106 was used only for
  ParT's W/Z in P13), 2,000 random rows per class (numpy default_rng(5454), never first-k), split at random within each class
  into sample 1 (labelling / ranking, 1,000 per class) and sample 2 (held-out evaluation, 1,000 per class); jet order shuffled.
  Reference check: unedited accuracy on the P1 sample (rows 3,000-3,300 of each _100 test file, N 3,000; expected 83.73 %).

Steps
 atlas   Per block b, per hidden multivector channel c (after the gated GELU), per jet: f0 = mean over particle tokens of the
         scalar part x_c,0; f1 = mean over particle tokens of the Minkowski square of the grade-1 part, x_t^2 - x_x^2 - x_y^2 -
         x_z^2. Per auxiliary scalar unit: fs = mean over particle tokens. Each feature mapped to normal scores with sample 1's
         distribution (monotone; R^2 then behaves like a rank correlation). Labels (17): observables (Gaussianised the same way)
         log m (invariant mass of the particles), log pT (jet pT), N (number of particles), tau21, tau32 (N-subjettiness ratios
         tau2/tau1, tau3/tau2, file values), D2 (energy-correlation ratio e3/e2^3, beta = 1, all particles), log M2/M0 (M2 =
         sum pT r^2 about the stored jet axis, M0 = sum pT; the pT-weighted mean squared angular distance) and the 10 classes
         (one vs rest). Per unit and label a least-squares fit on sample 1 (unit features + intercept), evaluated on sample 2:
         observable score = sqrt(max(R^2, 0)), class score = 2 AUC - 1 (AUC of the fitted score for that class vs the rest).
         Clean = best score >= sqrt(0.5) (held-out R^2 >= 0.5; for a class 2 AUC - 1 >= 0.707); also at score >= 0.3 and at the
         untrained network's 95th percentile. Selective = clean and the best label's family (mass; pT; count; shape = tau21,
         tau32, D2; width = M2/M0; each class alone) beats the second-best family by >= 0.1. Effect = trained fraction minus
         rotated fraction; 95 % bootstrap over sample-2 jets (fitted maps fixed, NB draws, the same draws for all bases);
         p (one-sided, H0: channel units are no cleaner than rotated directions) from (i) the rank among the 10 rotations and
         (ii) a permutation of unit membership between the trained basis and the pooled rotated bases (10,000 permutations).
 steer   For each label, the channel whose scalar part f0 scores highest on sample 1 (all 12 x 32 channels), and as control the
         best-scoring direction for the same label in each of 3 raw grade-0 rotations (all blocks). Edit: add k x sd to the
         channel's scalar part (or k x sd_u x u to the 32 scalar parts for a direction u) at every particle token, k = +-1, 2, 4,
         sd = s.d. of the pooled feature on sample 1 (so the jet-level feature moves by exactly k s.d.). Target class c*: the
         class itself (class label; predicted sign = sign of the unit's class score) or, for an observable o, the class in which
         o is most shifted in the direction the unit follows (argmax_c sigma (AUC_c(o) - 0.5), sigma = sign of the unit's
         correlation with o; predicted sign +). Read-out l_c* = log p_c* - log(1 - p_c*). Direction-correct step: predicted sign
         x sign(k) x Delta l_c* > 0. Per label: share of correct (jet, step) pairs, channel minus mean of the 3 directions,
         bootstrap CI over jets, sign-flip permutation p over jets, McNemar on (jet, step) pairs vs direction 1, Holm over 17.
         Sample: 200 random jets per class of sample 2 (N 2,000).
 ablate  Each of the 384 channels mean-ablated in turn (its whole multivector set to the sample-1 mean of its token type: global,
         reference 1, reference 2, particle) on the same 2,000 jets: accuracy change, McNemar, Holm over 384; null: 30 random
         raw-rotation directions ablated the same way (component along u set to its mean).
 circuit For class k: channels ranked on sample 1 (5,000 jets) by mean |attribution| a = sum_tokens,components (h_donor - h) .
         d l_k / d h (first-order effect of resampling the channel on l_k), frozen. On sample 2 (N 10,000): keep the top K
         channels (K = 8, 16, 32, 64 of 384); every other hidden multivector channel and every hidden auxiliary scalar of every
         block is resample-ablated (replaced by its value in the clean run of a pT-matched donor jet: the batch neighbour in jet
         pT, token k <- donor token k mod n_donor). Attention, layer norms and residual stream run live. Reported: 10-class
         accuracy, class-k recall, class-k vs rest AUC of l_k, 10-class confusion; size-matched random channel sets (3 per K);
         floor (no channel kept); a variant with the auxiliary scalars live (K = 32); one all-class ranking (sum over k).
 calib   E4: real 5 % mass dilation (every particle's deta, dphi x 1.05 about the stored axis at fixed pT and particle mass;
         m x ~1.05) on the W and Z jets of sample 2. Patch the chosen channels' activations (all tokens) to their values in the
         dilated jet's run, everything else from the unedited jet. Read-out d = l_W - l_Z (logits).
         R_patch = sum_jets Delta d(patch) / sum_jets Delta d(dilation) = share of the network's response to the real mass change
         carried by the patched channels. Sets: top K log-m channels by held-out R^2 (K = 1, 8, 32), all 384, 10 random K-sets,
         top-K log-m directions of raw rotation 0.

  ./sync.sh && ./part_job.sh c-p54-run '/j-jepa-vol/aaron-interp/env-lgatr/bin/python -m interp.p54_lgatr_channel_atlas'
  (runs the whole pipeline at smoke size first in the same process, then the full size; --no_smoke skips that)
-> $AI_RESULTS/lgatr/p54_lgatr_channel_atlas[_smoke].json, p54_arrays.npz, p54_lgatr_channel_atlas.png
"""
import argparse, json, os, sys, time, traceback
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('JETDISTILL_TAGGER', 'part')
ap = argparse.ArgumentParser(); ap.add_argument('--no_smoke', action='store_true'); ap.add_argument('--smoke_only', action='store_true')
ap.add_argument('--sections', default='atlas,steer,ablate,circuit,calib'); ap.add_argument('--figure_only', action='store_true')
ARGS = ap.parse_args()
SEED, FIDX = 5454, 106
TEST = Path('/j-jepa-vol/JetClass/Pythia/test_20M')
CKPT = '/j-jepa-vol/aaron-interp/models/lgatr/runs/jc2m/jc2m_s0a/models/model_run0.pt'
PRE = {'QCD': 'ZJetsToNuNu', 'Hbb': 'HToBB', 'Hcc': 'HToCC', 'Hgg': 'HToGG', 'H4q': 'HToWW4Q', 'Hqql': 'HToWW2Q1L', 'Zqq': 'ZToQQ',
       'Wqq': 'WToQQ', 'Tbqq': 'TTBar', 'Tbl': 'TTBarLep'}
CL = ['QCD', 'Hbb', 'Hcc', 'Hgg', 'H4q', 'Hqql', 'Zqq', 'Wqq', 'Tbqq', 'Tbl']
iW, iZ = CL.index('Wqq'), CL.index('Zqq')
OBS = ['log_m', 'log_pt', 'N', 'tau21', 'tau32', 'D2', 'log_M2_over_M0']
LABELS = OBS + ['cls_' + c for c in CL]
NL = len(LABELS); NO = len(OBS)
FAMS = [[0], [1], [2], [3, 4, 5], [6]] + [[NO + c] for c in range(10)]
FAMNAMES = ['mass', 'pT', 'count', 'shape', 'width'] + CL
FAMOF = np.array([next(f for f, ix in enumerate(FAMS) if j in ix) for j in range(NL)])
NBLK, NC, NS = 12, 32, 64
KS = (-4, -2, -1, 1, 2, 4)
ETA = [1.0, -1.0, -1.0, -1.0]
RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'lgatr'
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def jdump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(obj, indent=1, default=float)); log('wrote', path)


def params(smoke):
    return dict(smoke=smoke, TAG='_smoke' if smoke else '', PER=30 if smoke else 2000, NB=20 if smoke else 200, NROT=2 if smoke else 10,
                NSTEER=6 if smoke else 200, NRDIR=2 if smoke else 3, NABL=8 if smoke else NBLK * NC, NNULL=4 if smoke else 30,
                NRANK=10 if smoke else 500, NRSET=1 if smoke else 3, NPERM=200 if smoke else 10000, BS=125 if smoke else 250,
                CSIZES=(8, 16) if smoke else (8, 16, 32, 64), P1=20 if smoke else 300)


# ------------------------------------------------------------------------------------------------ data
def read_sample(per, rng):
    import uproot
    from interp.part_core import read_root, sub, cat
    s1, s2, t1, t2 = [], [], [], []
    for c in CL:
        path = TEST / f'{PRE[c]}_{FIDX}.root'; tr = uproot.open(path)['tree']; N = tr.num_entries
        rows = np.sort(rng.choice(N, per, replace=False)); ch = []
        for s in range(0, N, 25000):
            r = rows[(rows >= s) & (rows < s + 25000)]
            if len(r): ch.append(sub(read_root(path, s, min(s + 25000, N)), r - s))
        J = cat(ch); assert (J['label'] == CL.index(c)).all(), c
        t = tr.arrays([f'jet_tau{k}' for k in (1, 2, 3)], library='np'); T = np.stack([t[f'jet_tau{k}'][rows] for k in (1, 2, 3)], 1).astype(np.float64)
        p = rng.permutation(per); a, b = np.sort(p[:per // 2]), np.sort(p[per // 2:])
        s1.append(sub(J, a)); s2.append(sub(J, b)); t1.append(T[a]); t2.append(T[b])
    J1, J2, T1, T2 = cat(s1), cat(s2), np.concatenate(t1), np.concatenate(t2)
    p1, p2 = rng.permutation(len(T1)), rng.permutation(len(T2))
    return sub(J1, p1), T1[p1], sub(J2, p2), T2[p2]


def d2(J, dev):
    """D2 (beta = 1) from all particles (copied from P40): e2 = sum_{i<j} z_i z_j R_ij, e3 = sum_{i<j<k} z_i z_j z_k R_ij R_ik R_jk"""
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
        A = z[:, None, :] * R
        e3 = torch.einsum('ni,nij,nij->n', z, A @ R, A) / 6.0
        out[s:s + 500] = (e3 / e2.clamp_min(1e-12) ** 3).cpu().numpy()
    return out


def observables(J, T, dev):
    from interp.part_core import kin, jmass, moments
    pt, de, dp, _ = kin(J); m = J['mask'] > 0; mo = moments(pt, de, dp)
    O = dict(log_m=np.log(jmass(J).astype(float) + 1e-3), log_pt=np.log(J['jet_pt'].astype(float)), N=m.sum(1).astype(float),
             tau21=T[:, 1] / np.maximum(T[:, 0], 1e-9), tau32=T[:, 2] / np.maximum(T[:, 1], 1e-9), D2=d2(J, dev),
             log_M2_over_M0=np.log(np.maximum(mo['M2'] / np.maximum(mo['M0'], 1e-9), 1e-12)))
    return np.stack([O[k] for k in OBS], 1)


# ------------------------------------------------------------------------------------------------ network
def load_untrained(dev, seed):
    import torch
    from interp.p5_lgatr_launch import LG
    sys.path.insert(0, LG); cwd = os.getcwd(); os.chdir(LG)
    try:
        from omegaconf import OmegaConf
        from hydra.utils import instantiate
        import experiments.base_experiment  # noqa: F401
        from experiments.tagging.jetclassexperiment import JetClassTaggingExperiment
        cfg = OmegaConf.load(Path(CKPT).parents[1] / 'config.yaml'); exp = JetClassTaggingExperiment(cfg, 0, 1)
        exp.init_physics(); exp.init_geometric_algebra(); torch.manual_seed(seed); model = instantiate(exp.cfg.model)
        model.force_xformers = dev == 'cuda' and bool(exp.cfg.training.force_xformers)
        return model.to(dev).eval(), exp.cfg
    finally:
        os.chdir(cwd)


class Net:
    """L-GATr with a forward hook on every block's gated nonlinearity (output = hidden multivectors (1, T, 32, 16) and hidden
    auxiliary scalars (1, T, 64)). self.rec(b, mv, s) records, self.op(b, mv, s) -> (mv, s) edits, self.perm permutes channels
    after both (channel-permuted control)."""

    def __init__(self, kind, dev):
        import torch
        from interp.p5_lgatr_launch import load_lgatr, LG
        sys.path.insert(0, LG)
        if kind == 'untrained': self.model, self.cfg = load_untrained(dev, SEED)
        else: self.model, self.cfg = load_lgatr(CKPT, dev)
        for p in self.model.parameters(): p.requires_grad_(False)
        from experiments.tagging.embedding import dense_to_sparse_jet, embed_tagging_data_into_ga
        self._d2s, self._emb = dense_to_sparse_jet, embed_tagging_data_into_ga
        self.dev, self.kind = dev, kind; self.rec = None; self.op = None; self.perm = None
        mods = [m for blk in self.model.net.blocks for m in blk.mlp.layers if type(m).__name__ == 'ScalarGatedNonlinearity']
        assert len(mods) == NBLK, len(mods)
        for b, m in enumerate(mods): m.register_forward_hook(self._hook(b))
        self.meta = None

    def set_perm(self, on):
        import torch
        if not on: self.perm = None; return
        self.perm = [(torch.from_numpy(np.random.default_rng(SEED + b).permutation(NC)).to(self.dev),
                      torch.from_numpy(np.random.default_rng(SEED + 100 + b).permutation(NS)).to(self.dev)) for b in range(NBLK)]

    def _hook(self, b):
        def h(mod, args, out):
            mv, s = out
            assert mv.shape[0] == 1 and mv.shape[-2] == NC and s.shape[-1] == NS, (mv.shape, s.shape)
            if self.rec is not None: self.rec(b, mv, s)
            if self.op is not None: mv, s = self.op(b, mv, s)
            if self.perm is not None: mv, s = mv[..., self.perm[b][0], :], s[..., self.perm[b][1]]
            return (mv, s)
        return h

    def embed(self, J, sl):
        import torch
        from interp.p5_lgatr_launch import to_lgatr_inputs
        from interp.part_core import sub
        Js = sub(J, sl); vec, feat = to_lgatr_inputs(Js)
        fm, sc, ptr = self._d2s(torch.from_numpy(vec).to(self.dev), torch.from_numpy(feat).to(self.dev))
        e = self._emb(fm, sc, ptr, self.cfg.data)
        bt = e['batch']; nb = int(bt.max()) + 1; cnt = torch.bincount(bt, minlength=nb); start = torch.cumsum(cnt, 0) - cnt
        pos = torch.arange(len(bt), device=bt.device) - start[bt]
        assert torch.equal(e['is_global'], pos == 0), 'global token not first'
        npart = cnt - 3; nreal = torch.from_numpy((Js['mask'] > 0).sum(1)).to(self.dev)
        assert torch.equal(npart, nreal), 'token count != particle count + 3'
        jpt = torch.from_numpy(Js['jet_pt'].astype(np.float64)).to(self.dev); o = torch.argsort(jpt); rk = torch.empty_like(o)
        rk[o] = torch.arange(nb, device=self.dev); nbr = torch.where(rk + 1 < nb, rk + 1, rk - 1).clamp(min=0); donor = o[nbr]
        d = donor[bt]; dpos = torch.where(pos < 3, pos, 3 + (pos - 3) % npart[d].clamp(min=1))
        ptok = pos >= 3; ttype = torch.clamp(pos, max=3)
        self.meta = dict(bt=bt, nb=nb, npart=npart, ptok=ptok, pidx=ptok.nonzero(as_tuple=True)[0], ttype=ttype,
                         donor_tok=start[d] + dpos, T=len(bt))
        return {k: e[k] for k in ('mv', 's', 'is_global', 'batch')}

    def logits(self, e):
        from interp.p5_lgatr_launch import TO_CL
        return self.model(e)[:, TO_CL]

    def run(self, J, P, fn=None, rows=None):
        """unedited logits for J (rows), calling fn(e, sl, L0) per batch (it may run more forwards)"""
        import torch
        rows = np.arange(len(J['label'])) if rows is None else rows; out = np.zeros((len(rows), 10), np.float32)
        with torch.no_grad():
            for s in range(0, len(rows), P['BS']):
                sl = rows[s:s + P['BS']]; e = self.embed(J, sl); L0 = self.logits(e); out[s:s + len(sl)] = L0.float().cpu().numpy()
                if fn is not None: fn(e, slice(s, s + len(sl)), L0)
        return out


def pool(meta, x):
    """mean over particle tokens per jet: x (T, ...) -> (nb, ...)"""
    import torch
    pi = meta['pidx']; out = torch.zeros((meta['nb'],) + x.shape[1:], device=x.device, dtype=x.dtype).index_add_(0, meta['bt'][pi], x[pi])
    return out / meta['npart'].clamp(min=1).reshape((-1,) + (1,) * (x.dim() - 1)).to(x.dtype)


def lodds(L):
    import torch
    n = L.shape[0]; E = L[:, None, :].expand(n, 10, 10).clone(); E[:, torch.arange(10), torch.arange(10)] = -float('inf')
    return L - torch.logsumexp(E, -1)


# ------------------------------------------------------------------------------------------------ statistics
def boot_W(n, nb, seed):
    r = np.random.default_rng(seed); return np.stack([np.bincount(r.integers(0, n, n), minlength=n) for _ in range(nb)]).astype(np.float32)


def boot_mean(x, W):
    x = np.asarray(x, float); bs = W @ x / W.sum(1); return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def mcnemar(c0, c1):
    from scipy.stats import binomtest
    b, c = int((c0 & ~c1).sum()), int((~c0 & c1).sum())
    return dict(lost=b, gained=c, p=float(binomtest(b, b + c).pvalue) if b + c else 1.0)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.zeros(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj


def confusion(y, p):
    M = np.zeros((10, 10), int); np.add.at(M, (y, p), 1); return M


def auc_np(s, pos):
    from scipy.stats import rankdata
    r = rankdata(s); n1 = pos.sum(); n0 = len(s) - n1
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ------------------------------------------------------------------------------------------------ atlas scoring (torch, GPU)
def gauss(X1, X2):
    """columnwise normal scores; X1 (N1, U) by its own ranks, X2 (N2, U) through X1's empirical CDF; constant columns -> 0"""
    import torch
    N1 = X1.shape[0]
    const = (X1.std(0) <= 1e-7 * (1 + X1.abs().mean(0)))
    r1 = torch.argsort(torch.argsort(X1, 0), 0).float() + 1
    Z1 = torch.special.ndtri(r1 / (N1 + 1))
    S = torch.sort(X1, 0).values.T.contiguous()
    u = (torch.searchsorted(S, X2.T.contiguous(), right=True).float().T + 0.5) / (N1 + 1)
    Z2 = torch.special.ndtri(u.clamp(0.5 / (N1 + 1), 1 - 0.5 / (N1 + 1)))
    Z1[:, const] = 0; Z2[:, const] = 0
    return Z1, Z2, const


def auc_w(S, pos, W):
    """weighted AUC: S (V, N) scores, pos (V, N) bool, W (B, N) weights -> (V, B)"""
    import torch
    o = torch.argsort(S, 1); ps = torch.gather(pos.float(), 1, o); ng = 1 - ps
    Wo = W[:, o].permute(1, 0, 2)                                   # (V, B, N)
    cn = torch.cumsum(Wo * ng[:, None], 2)
    num = (Wo * ps[:, None] * cn).sum(2); den = (Wo * ps[:, None]).sum(2) * (Wo * ng[:, None]).sum(2)
    return num / den.clamp(min=1e-12)


def score_units(F1, F2, Y1z, Y2z, y1, y2, W=None, chunk=64):
    """F1 (N1, U, d), F2 (N2, U, d) unit features; Y1z, Y2z (N, NO) Gaussianised observables; y class labels (torch long).
    Returns held-out scores (U, NL) and, with W (B, N2), bootstrap scores (B, U, NL)."""
    import torch
    N1, U, d = F1.shape; dev = F1.device
    Z1, Z2, const = gauss(F1.reshape(N1, U * d), F2.reshape(F2.shape[0], U * d))
    Z1 = Z1.reshape(N1, U, d); Z2 = Z2.reshape(-1, U, d); dead = const.reshape(U, d).all(1)
    T1 = torch.cat([Y1z, torch.nn.functional.one_hot(y1, 10).float()], 1)      # (N1, NL)
    pos2 = torch.nn.functional.one_hot(y2, 10).bool().T                         # (10, N2)
    Y2c = Y2z - Y2z.mean(0); sst = (Y2c ** 2).sum(0)
    out = torch.zeros(U, NL, device=dev); outb = None if W is None else torch.zeros(W.shape[0], U, NL, device=dev)
    if W is not None:
        Wn = W.sum(1); sy = W @ Y2z; syy = W @ (Y2z ** 2); sstb = syy - sy ** 2 / Wn[:, None]           # (B, NO)
    for u0 in range(0, U, chunk):
        us = slice(u0, min(U, u0 + chunk)); n = us.stop - us.start
        X1 = torch.cat([torch.ones(n, N1, 1, device=dev), Z1[:, us].permute(1, 0, 2)], 2)            # (n, N1, d+1)
        X2 = torch.cat([torch.ones(n, Z2.shape[0], 1, device=dev), Z2[:, us].permute(1, 0, 2)], 2)
        A = X1.transpose(1, 2) @ X1 + 1e-4 * torch.eye(d + 1, device=dev)
        C = torch.linalg.solve(A, X1.transpose(1, 2) @ T1)                                             # (n, d+1, NL)
        Pr = X2 @ C                                                                                    # (n, N2, NL)
        R = (Y2z[None] - Pr[..., :NO]) ** 2                                                            # (n, N2, NO)
        r2 = 1 - R.sum(1) / sst
        out[us, :NO] = r2.clamp(min=0).sqrt()
        S = Pr[..., NO:].permute(0, 2, 1).reshape(n * 10, -1)                                          # (n*10, N2)
        P_ = pos2[None].expand(n, 10, -1).reshape(n * 10, -1)
        a = auc_w(S, P_, torch.ones(1, S.shape[1], device=dev))[:, 0].reshape(n, 10)
        out[us, NO:] = (2 * a - 1).clamp(min=0)
        if W is not None:
            sse = torch.einsum('bn,unk->buk', W, R)
            outb[:, us, :NO] = (1 - sse / sstb[:, None]).clamp(min=0).sqrt()
            for v0 in range(0, n * 10, 16):
                vs = slice(v0, min(n * 10, v0 + 16)); ab = auc_w(S[vs], P_[vs], W)                     # (v, B)
                idx = torch.arange(vs.start, vs.stop, device=dev)
                outb[:, us.start + idx // 10, NO + idx % 10] = (2 * ab.T - 1).clamp(min=0)
    out[dead] = 0
    if outb is not None: outb[:, dead] = 0
    return out, outb


def flags(S, thr):
    """S (..., U, NL) -> clean, selective (..., U) bool"""
    import torch
    best = S.max(-1).values; lab = S.argmax(-1)
    F = torch.stack([S[..., ix].max(-1).values for ix in FAMS], -1); srt = torch.sort(F, -1).values
    famok = F.argmax(-1) == torch.as_tensor(FAMOF, device=S.device)[lab]
    clean = best >= thr
    return clean, clean & famok & (srt[..., -1] - srt[..., -2] >= 0.1)


def haar(n, seed):
    r = np.random.default_rng(seed); Z = r.standard_normal((n, n)); Qm, Rm = np.linalg.qr(Z); return Qm * np.sign(np.diag(Rm))[None, :]


# ------------------------------------------------------------------------------------------------ sections
def capture(net, J, P, rows=None, means=False, gram=True):
    """pooled features: f0 (N, 12, 32), Gram G (N, 12, 32, 32) of the grade-1 parts (Minkowski), fs (N, 12, 64); unedited
    logits; optionally token-type means (4, 12, 32, 16) / (4, 12, 64)"""
    import torch
    dev = net.dev; rows = np.arange(len(J['label'])) if rows is None else rows; N = len(rows)
    f0 = torch.zeros(N, NBLK, NC, device=dev); G = torch.zeros((N, NBLK, NC, NC) if gram else (N, NBLK, NC), device=dev); fs = torch.zeros(N, NBLK, NS, device=dev)
    eta = torch.tensor(ETA, device=dev)
    msum = torch.zeros(4, NBLK, NC, 16, device=dev, dtype=torch.float64); ssum = torch.zeros(4, NBLK, NS, device=dev, dtype=torch.float64)
    mcnt = torch.zeros(4, device=dev, dtype=torch.float64); cur = {}

    def rec(b, mv, s):
        m = net.meta; x = mv[0].float(); sl = cur['sl']
        f0[sl, b] = pool(m, x[:, :, 0]); fs[sl, b] = pool(m, s[0].float())
        v = x[:, :, 1:5]
        G[sl, b] = pool(m, torch.einsum('tam,tbm->tab', v * eta, v)) if gram else pool(m, (v * v * eta).sum(-1))
        if means:
            tt = m['ttype']; msum[:, b] += torch.zeros(4, NC, 16, device=dev, dtype=torch.float64).index_add_(0, tt, x.double())
            ssum[:, b] += torch.zeros(4, NS, device=dev, dtype=torch.float64).index_add_(0, tt, s[0].double())
            if b == 0: mcnt.add_(torch.bincount(tt, minlength=4).double())

    net.rec = rec
    try:
        L = np.zeros((N, 10), np.float32)
        with torch.no_grad():
            for s0 in range(0, N, P['BS']):
                sl = slice(s0, min(N, s0 + P['BS'])); cur['sl'] = sl; e = net.embed(J, rows[sl]); L[sl] = net.logits(e).float().cpu().numpy()
    finally:
        net.rec = None
    r = dict(f0=f0, G=G, fs=fs, L=L)
    if means: r['mean_mv'] = (msum / mcnt[:, None, None, None]).float(); r['mean_s'] = (ssum / mcnt[:, None, None]).float()
    return r


def unit_feats(C, basis, rots, b_std):
    """per basis: MV features (N, 384, 2) = (f0, f1) and aux (N, 768, 1). basis = ('channel',) | ('raw'|'std', r)"""
    import torch
    f0, G, fs = C['f0'], C['G'], C['fs']
    if basis[0] == 'channel':
        g = torch.diagonal(G, dim1=-2, dim2=-1) if G.dim() == 4 else G; a = fs
    else:
        kind, r = basis; R0, R1, RS = rots[r]
        if kind == 'std':
            s0, s1, ss = b_std; f0 = f0 / s0; G = G / (s1[..., :, None] * s1[..., None, :]); fs = fs / ss
        f0 = torch.einsum('bca,nba->nbc', R0, f0)
        g = torch.einsum('bca,nbad,bcd->nbc', R1, G, R1)
        a = torch.einsum('bca,nba->nbc', RS, fs)
    N = f0.shape[0]
    return torch.stack([f0, g], -1).reshape(N, NBLK * NC, 2), a.reshape(N, NBLK * NS, 1)


def sec_atlas(nets, D, P, out, arr):
    import torch
    dev = nets['trained'].dev
    J1, J2, O1, O2, y1, y2 = D['J1'], D['J2'], D['O1'], D['O2'], D['y1'], D['y2']
    Ot1 = torch.from_numpy(O1).float().to(dev); Ot2 = torch.from_numpy(O2).float().to(dev)
    Y1z, Y2z, _ = gauss(Ot1, Ot2)
    yt1 = torch.from_numpy(y1).long().to(dev); yt2 = torch.from_numpy(y2).long().to(dev)
    W = torch.from_numpy(boot_W(len(y2), P['NB'], SEED)).to(dev)
    rots = {r: tuple(torch.from_numpy(np.stack([haar(n, 100000 * r + 1000 * g + b) for b in range(NBLK)])).float().to(dev)
                     for g, n in ((0, NC), (1, NC), (2, NS))) for r in range(P['NROT'])}
    caps = {}; res = dict(units=dict(mv=NBLK * NC, aux=NBLK * NS), bases={})
    for kind in ('trained', 'untrained', 'permuted'):
        net = nets['untrained' if kind == 'untrained' else 'trained']; net.set_perm(kind == 'permuted')
        try:
            c1 = capture(net, J1, P, means=(kind == 'trained'), gram=(kind == 'trained')); c2 = capture(net, J2, P, gram=(kind == 'trained'))
        finally:
            net.set_perm(False)
        acc1 = float((c1['L'].argmax(1) == y1).mean()); acc2 = float((c2['L'].argmax(1) == y2).mean())
        res[f'acc_{kind}'] = dict(sample1=acc1, sample2=boot_mean(c2['L'].argmax(1) == y2, boot_W(len(y2), 1000, 1)))
        log(f'atlas capture {kind}: acc sample 1 {100 * acc1:.2f} %, sample 2 {100 * acc2:.2f} %')
        caps[kind] = (c1, c2)
    c1, c2 = caps['trained']
    D['L2'] = c2['L']; D['mean_mv'] = c1['mean_mv']; D['mean_s'] = c1['mean_s']; D['C1'] = c1
    s0 = c1['f0'].std(0).clamp(min=1e-6); s1 = torch.diagonal(c1['G'], dim1=-2, dim2=-1).abs().mean(0).sqrt().clamp(min=1e-6)
    ss = c1['fs'].std(0).clamp(min=1e-6); bstd = (s0, s1, ss)
    bases = [('trained', ('channel',)), ('untrained', ('channel',)), ('permuted', ('channel',))] + \
            [('trained', ('raw', r)) for r in range(P['NROT'])] + [('trained', ('std', r)) for r in range(P['NROT'])]
    SC = {}
    for kind, bs in bases:
        name = kind if bs[0] == 'channel' else f'rot_{bs[0]}{bs[1]}'
        a1, b1 = unit_feats(caps[kind][0], bs, rots, bstd); a2, b2 = unit_feats(caps[kind][1], bs, rots, bstd)
        Smv, Smvb = score_units(a1, a2, Y1z, Y2z, yt1, yt2, W=W)
        Sax, _ = score_units(b1, b2, Y1z, Y2z, yt1, yt2)
        SC[name] = dict(mv=Smv, mvb=Smvb, aux=Sax)
        log(f'atlas scored {name}: mv median best score {float(Smv.max(1).values.median()):.3f}, aux {float(Sax.max(1).values.median()):.3f}')
    thr = dict(R2_0p5=float(np.sqrt(0.5)), s0p3=0.3,
               untrained95_mv=float(torch.quantile(SC['untrained']['mv'].max(1).values, 0.95)),
               untrained95_aux=float(torch.quantile(SC['untrained']['aux'].max(1).values, 0.95)))
    res['thresholds'] = thr; res['threshold_defs'] = dict(R2_0p5='best score >= sqrt(0.5): held-out R^2 >= 0.5 (class: 2AUC-1 >= 0.707)',
                                                          s0p3='best score >= 0.3', untrained95='95th percentile of the untrained network units of the same type')
    rawn = [f'rot_raw{r}' for r in range(P['NROT'])]; stdn = [f'rot_std{r}' for r in range(P['NROT'])]
    summ = {}
    for ut in ('mv', 'aux'):
        summ[ut] = {}
        for tn in ('R2_0p5', 's0p3', 'untrained95'):
            tv = thr[f'untrained95_{ut}'] if tn == 'untrained95' else thr[tn]; r = {}
            fl = {n: flags(SC[n][ut], tv) for n in SC}
            for n in SC:
                cl, se = fl[n]
                r[n] = dict(clean=float(cl.float().mean()), selective=float(se.float().mean()),
                            per_block_clean=[float(x) for x in cl.float().reshape(NBLK, -1).mean(1)],
                            label_counts={LABELS[j]: int((cl & (SC[n][ut].argmax(1) == j)).sum()) for j in range(NL) if bool((cl & (SC[n][ut].argmax(1) == j)).any())})
            for which in ('clean', 'selective'):
                k = 0 if which == 'clean' else 1
                for rn, names in (('raw', rawn), ('std', stdn)):
                    rv = [r[n][which] for n in names]; tr = r['trained'][which]
                    d = dict(trained=tr, rot_mean=float(np.mean(rv)), rot_min=float(np.min(rv)), rot_max=float(np.max(rv)),
                             p_rank=float((1 + sum(v >= tr for v in rv)) / (1 + len(rv))))
                    # permutation of unit membership: trained units vs pooled rotated units
                    ft = fl['trained'][k].float().cpu().numpy(); fr = np.concatenate([fl[n][k].float().cpu().numpy() for n in names])
                    pool_ = np.concatenate([ft, fr]); obs = ft.mean() - fr.mean(); g = np.random.default_rng(SEED); cnt = 0
                    for _ in range(P['NPERM']):
                        pp = g.permutation(pool_); cnt += (pp[:len(ft)].mean() - pp[len(ft):].mean()) >= obs - 1e-12
                    d['p_perm_units'] = float((1 + cnt) / (1 + P['NPERM']))
                    if ut == 'mv':
                        bt_ = flags(SC['trained']['mvb'], tv)[k].float().mean(1).cpu().numpy()
                        brot = np.mean([flags(SC[n]['mvb'], tv)[k].float().mean(1).cpu().numpy() for n in names], 0)
                        bun = flags(SC['untrained']['mvb'], tv)[k].float().mean(1).cpu().numpy()
                        d['trained_ci'] = [float(np.percentile(bt_, 2.5)), float(np.percentile(bt_, 97.5))]
                        d['rot_mean_ci'] = [float(np.percentile(brot, 2.5)), float(np.percentile(brot, 97.5))]
                        dd = bt_ - brot; d['diff'] = float(tr - np.mean(rv)); d['diff_ci'] = [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]
                        d['untrained_ci'] = [float(np.percentile(bun, 2.5)), float(np.percentile(bun, 97.5))]
                    r[f'{which}_vs_rot_{rn}'] = d
            r['untrained'] = r['untrained']; summ[ut][tn] = r
            log(f'atlas {ut} {tn} (thr {tv:.3f}): clean trained {r["trained"]["clean"]:.3f} raw rot {r["clean_vs_rot_raw"]["rot_mean"]:.3f} '
                f'[{r["clean_vs_rot_raw"]["rot_min"]:.3f}, {r["clean_vs_rot_raw"]["rot_max"]:.3f}] std rot {r["clean_vs_rot_std"]["rot_mean"]:.3f} '
                f'untrained {r["untrained"]["clean"]:.3f} permuted {r["permuted"]["clean"]:.3f} | selective trained {r["trained"]["selective"]:.3f} '
                f'raw {r["selective_vs_rot_raw"]["rot_mean"]:.3f} [{r["selective_vs_rot_raw"]["rot_min"]:.3f}, {r["selective_vs_rot_raw"]["rot_max"]:.3f}] '
                f'p_perm {r["selective_vs_rot_raw"]["p_perm_units"]:.4f}')
    res['summary'] = summ
    # top channels per label (trained, MV) with held-out score
    St = SC['trained']['mv'].cpu().numpy()
    res['top_channels'] = {LABELS[j]: [dict(block=int(u // NC), channel=int(u % NC), score=float(St[u, j]), best_label=LABELS[int(St[u].argmax())])
                                       for u in np.argsort(-St[:, j])[:5]] for j in range(NL)}
    res['best_score_quantiles'] = {n: {ut: [float(x) for x in np.quantile(SC[n][ut].max(1).values.cpu().numpy(), [0.5, 0.9, 0.99])] for ut in ('mv', 'aux')}
                                   for n in ('trained', 'untrained', 'permuted', 'rot_raw0', 'rot_std0') if n in SC}
    out['atlas'] = res
    for n in ('trained', 'untrained', 'permuted', 'rot_raw0', 'rot_std0'):
        if n in SC: arr[f'S_mv_{n}'] = SC[n]['mv'].cpu().numpy(); arr[f'S_aux_{n}'] = SC[n]['aux'].cpu().numpy()
    D['S_mv'] = St; D['rots'] = rots; D['Y1z'] = Y1z
    del caps


def corr_scores(X, Y1z, y1):
    """in-sample (sample 1) scores of single features X (N1, V): |corr| with Gaussianised observables (signed corr kept) and
    2AUC - 1 per class (signed). -> (V, NL) signed"""
    import torch
    Z, _, _ = gauss(X, X[:2]); Zc = Z - Z.mean(0); Yc = Y1z - Y1z.mean(0)
    rho = (Zc.T @ Yc) / (Zc.norm(dim=0)[:, None] * Yc.norm(dim=0)[None] + 1e-12)
    pos = torch.nn.functional.one_hot(y1, 10).bool().T
    V = Z.shape[1]; S = Z.T.repeat_interleave(10, 0); Pm = pos.repeat(V, 1)
    a = auc_w(S, Pm, torch.ones(1, Z.shape[0], device=Z.device))[:, 0].reshape(V, 10)
    return torch.cat([rho, 2 * a - 1], 1)


def sec_steer(net, D, P, out, arr):
    import torch
    dev = net.dev; C1 = D['C1']; y1 = D['y1']; yt1 = torch.from_numpy(y1).long().to(dev); Y1z = D['Y1z']
    f0 = C1['f0'].reshape(len(y1), NBLK * NC)
    Sch = corr_scores(f0, Y1z, yt1)                                                       # (384, NL) signed
    # class shift of each observable: AUC_c(o) - 0.5 on sample 1
    O1 = D['O1']; AO = np.array([[auc_np(O1[:, o], y1 == c) - 0.5 for c in range(10)] for o in range(NO)])
    units = []
    for j in range(NL):
        cand = [('channel', None)] + [('rot', r) for r in range(P['NRDIR'])]
        for kind, r in cand:
            if kind == 'channel':
                sc = Sch[:, j]; u = int(sc.abs().argmax()); b, c = u // NC, u % NC
                vec = torch.zeros(NC, device=dev); vec[c] = 1.0; feat = f0[:, u]
            else:
                R0 = D['rots'][r][0]                                                          # (12, 32, 32): rows = directions
                proj = torch.einsum('bca,nba->nbc', R0, C1['f0']).reshape(len(y1), NBLK * NC)
                sc = corr_scores(proj, Y1z, yt1)[:, j]; u = int(sc.abs().argmax()); b, c = u // NC, u % NC
                vec = R0[b, c].clone(); feat = proj[:, u]
            sg = float(np.sign(float(sc[u]))) or 1.0
            if j < NO: cstar = int(np.argmax(sg * AO[j])); spred = 1.0
            else: cstar = j - NO; spred = sg
            units.append(dict(label=LABELS[j], kind=kind, rot=r, block=int(b), index=int(c), score_s1=float(sc[u]), cstar=CL[cstar],
                              cstar_i=cstar, spred=spred, sd=float(feat.std()), vec=vec))
    log(f'steer: {len(units)} units ({NL} labels x {1 + P["NRDIR"]})')
    rng = np.random.default_rng(SEED + 1)
    rows = np.sort(np.concatenate([rng.choice(np.where(D['y2'] == c)[0], P['NSTEER'], replace=False) for c in range(10)]))
    y = D['y2'][rows]; N = len(rows)
    dl = np.zeros((len(units), len(KS), N), np.float32)

    def fn(e, sl, L0):
        l0 = lodds(L0.float())
        pi = net.meta['pidx']
        for ui, U in enumerate(units):
            for ki, k in enumerate(KS):
                a = k * U['sd']; vb = U['vec']; bb = U['block']

                def op(b, mv, s, a=a, vb=vb, bb=bb):
                    if b != bb: return mv, s
                    mv = mv.clone(); mv[0, pi, :, 0] += a * vb; return mv, s
                net.op = op
                try:
                    l1 = lodds(net.logits(e).float())
                finally:
                    net.op = None
                dl[ui, ki, sl] = (l1[:, U['cstar_i']] - l0[:, U['cstar_i']]).cpu().numpy()
    net.run(D['J2'], P, fn, rows=rows)
    W = boot_W(N, 1000, SEED + 2); sgnk = np.sign(np.array(KS))[:, None]
    res = []; pr = []
    for j in range(NL):
        grp = [i for i, U in enumerate(units) if U['label'] == LABELS[j]]; ch = grp[0]; rt = grp[1:]
        corr = lambda i: (units[i]['spred'] * sgnk * dl[i]) > 0                         # (6, N)
        cc = corr(ch); cr = np.stack([corr(i) for i in rt])                               # (R, 6, N)
        sh_ch = cc.mean(0); sh_rt = cr.mean((0, 1)); dj = sh_ch - sh_rt
        g = np.random.default_rng(j); null = np.array([(dj * g.choice([-1, 1], N)).mean() for _ in range(5000)])
        p = float((1 + (null >= dj.mean()).sum()) / 5001)
        mc = mcnemar(cr[0].ravel(), cc.ravel())
        U = units[ch]
        r = dict(label=LABELS[j], channel=dict(block=U['block'], channel=U['index'], score_s1=U['score_s1'], cstar=U['cstar'], spred=U['spred'], sd=U['sd']),
                 rot=[dict(rot=units[i]['rot'], block=units[i]['block'], dir=units[i]['index'], score_s1=units[i]['score_s1'], cstar=units[i]['cstar']) for i in rt],
                 share_channel=boot_mean(sh_ch, W), share_rot=boot_mean(sh_rt, W), diff=boot_mean(dj, W), p_perm=p, mcnemar_vs_rot0=mc,
                 dl_plus2_channel=boot_mean(U['spred'] * dl[ch, KS.index(2)], W),
                 dl_plus2_rot=boot_mean(np.mean([units[i]['spred'] * dl[i, KS.index(2)] for i in rt], 0), W),
                 dl_by_k_channel=[float((U['spred'] * dl[ch, ki]).mean()) for ki in range(len(KS))])
        res.append(r); pr.append(p)
    ph = holm(pr)
    for r, h in zip(res, ph): r['p_holm'] = float(h)
    for r in res:
        log(f"steer {r['label']:16s} ch b{r['channel']['block']}c{r['channel']['channel']} (s1 {r['channel']['score_s1']:+.2f}, c* {r['channel']['cstar']}) "
            f"correct {r['share_channel'][0]:.3f} vs rot {r['share_rot'][0]:.3f}; diff {r['diff'][0]:+.3f} [{r['diff'][1]:+.3f}, {r['diff'][2]:+.3f}] "
            f"p_holm {r['p_holm']:.3g}; dl(+2sd) {r['dl_plus2_channel'][0]:+.3f} vs {r['dl_plus2_rot'][0]:+.3f}")
    out['steer'] = dict(N=int(N), rows='200 random sample-2 jets per class (default_rng(5455))', KS=list(KS), labels=res,
                        n_holm_lt_0p01=int((ph < 0.01).sum()), n_diff_pos=int(sum(r['diff'][0] > 0 for r in res)))
    arr['steer_dl'] = dl


def mean_op(net, D, b0, chans=None, dirvec=None):
    import torch
    Mm = D['mean_mv']

    def op(b, mv, s):
        if b != b0: return mv, s
        tt = net.meta['ttype']; h = mv[0].clone(); mu = Mm[:, b][tt]                           # (T, 32, 16)
        if chans is not None: h[:, chans] = mu[:, chans]
        else:
            u = dirvec
            pr = torch.einsum('c,tcm->tm', u, h) - torch.einsum('c,tcm->tm', u, mu); h = h - u[None, :, None] * pr[:, None, :]
        return h[None], s
    return op


def sec_ablate(net, D, P, out, arr):
    import torch
    dev = net.dev; rng = np.random.default_rng(SEED + 1)
    rows = np.sort(np.concatenate([rng.choice(np.where(D['y2'] == c)[0], P['NSTEER'], replace=False) for c in range(10)]))
    y = D['y2'][rows]; N = len(rows)
    chans = [(u // NC, u % NC) for u in range(P['NABL'])]
    g = np.random.default_rng(SEED + 3)
    nulls = [(int(g.integers(NBLK)), int(g.integers(NC)), int(g.integers(P['NROT']))) for _ in range(P['NNULL'])]
    Lc = np.zeros((len(chans), N), np.int16); Ln = np.zeros((len(nulls), N), np.int16)
    dls = np.zeros((len(chans), N, 10), np.float32); p0 = np.zeros(N, np.int16)

    def fn(e, sl, L0):
        l0 = lodds(L0.float()); p0[sl] = L0.argmax(1).cpu().numpy()
        for i, (b, c) in enumerate(chans):
            net.op = mean_op(net, D, b, chans=[c])
            try: L = net.logits(e)
            finally: net.op = None
            Lc[i, sl] = L.argmax(1).cpu().numpy(); dls[i, sl] = (lodds(L.float()) - l0).cpu().numpy()
        for i, (b, c, r) in enumerate(nulls):
            net.op = mean_op(net, D, b, dirvec=D['rots'][r][0][b, c])
            try: L = net.logits(e)
            finally: net.op = None
            Ln[i, sl] = L.argmax(1).cpu().numpy()
    net.run(D['J2'], P, fn, rows=rows)
    c0 = p0 == y; W = boot_W(N, 1000, SEED + 4)
    rows_out = []; ps = []
    for i, (b, c) in enumerate(chans):
        c1 = Lc[i] == y; m = mcnemar(c0, c1); ps.append(m['p'])
        u = b * NC + c; j = int(D['S_mv'][u].argmax())
        rows_out.append(dict(block=b, channel=c, dacc=float(c1.mean() - c0.mean()), calls_changed=float((Lc[i] != p0).mean()), mcnemar=m,
                             best_label=LABELS[j], best_score=float(D['S_mv'][u, j]),
                             own_class_dl=float(dls[i][y == (j - NO)][:, j - NO].mean()) if j >= NO else None))
    ph = holm(ps)
    for r, h in zip(rows_out, ph): r['p_holm'] = float(h)
    dn = np.array([float((Ln[i] == y).mean() - c0.mean()) for i in range(len(nulls))]); cn = np.array([float((Ln[i] != p0).mean()) for i in range(len(nulls))])
    da = np.array([r['dacc'] for r in rows_out]); cc = np.array([r['calls_changed'] for r in rows_out])
    top = sorted(rows_out, key=lambda r: r['dacc'])[:10]
    out['ablate'] = dict(N=int(N), acc_unedited=boot_mean(c0, W), n_holm_lt_0p05=int((ph < 0.05).sum()),
                         dacc_quantiles=[float(x) for x in np.quantile(da, [0, 0.01, 0.1, 0.5, 0.9, 1])],
                         calls_changed_median=float(np.median(cc)), calls_changed_q90=float(np.quantile(cc, 0.9)),
                         null_dirs=dict(dacc_quantiles=[float(x) for x in np.quantile(dn, [0, 0.1, 0.5, 0.9, 1])], calls_changed_median=float(np.median(cn)),
                                        calls_changed_q90=float(np.quantile(cn, 0.9))),
                         frac_channels_above_null_max_calls=float((cc > cn.max()).mean()),
                         per_block_mean_calls_changed=[float(cc[[k for k, (b, c) in enumerate(chans) if b == bb]].mean()) if any(b == bb for b, c in chans) else None for bb in range(NBLK)],
                         top10=top, channels=rows_out)
    log(f"ablate: unedited {100 * c0.mean():.2f} %, Holm<0.05 {int((ph < 0.05).sum())}/{len(chans)}; dacc min {da.min():+.4f} median {np.median(da):+.4f}; "
        f"calls changed median {np.median(cc):.4f} q90 {np.quantile(cc, 0.9):.4f} vs rotated-direction null median {np.median(cn):.4f} max {cn.max():.4f}")
    arr['ablate_calls'] = Lc


def attributions(net, D, P):
    """mean |first-order effect of resampling channel c on l_k| per (k, block, channel) on NRANK jets per class of sample 1"""
    import torch
    rng = np.random.default_rng(SEED + 5); y1 = D['y1']
    rows = np.sort(np.concatenate([rng.choice(np.where(y1 == c)[0], min(P['NRANK'], int((y1 == c).sum())), replace=False) for c in range(10)]))
    A = torch.zeros(10, NBLK, NC, device=net.dev, dtype=torch.float64); n = 0
    fx = net.model.force_xformers; net.model.force_xformers = False      # dense masked attention for the backward pass
    try:
        _attr_loop(net, D, rows, A)
    finally:
        net.model.force_xformers = fx
    return (A / len(rows)).float().cpu().numpy()


def _attr_loop(net, D, rows, A):
    import torch
    for s in range(0, len(rows), 50):
        sl = rows[s:s + 50]
        with torch.no_grad():
            e = net.embed(D['J1'], sl); cache = {}
            net.rec = lambda b, mv, s_: cache.__setitem__(b, mv[0].detach().clone())
            try: net.logits(e)
            finally: net.rec = None
        dl = {}

        def op(b, mv, s_):
            d = torch.zeros_like(mv, requires_grad=True); dl[b] = d; return mv + d, s_
        net.op = op
        try:
            with torch.enable_grad():
                L = lodds(net.logits(e).float())
                for k in range(10):
                    g = torch.autograd.grad(L[:, k].sum(), [dl[b] for b in range(NBLK)], retain_graph=k < 9)
                    for b in range(NBLK):
                        h = cache[b]; diff = h[net.meta['donor_tok']] - h                     # (T, 32, 16)
                        at = (diff * g[b][0]).sum(-1)                                           # (T, 32)
                        pj = torch.zeros(net.meta['nb'], NC, device=net.dev).index_add_(0, net.meta['bt'], at)
                        A[k, b] += pj.abs().sum(0).double()
        finally:
            net.op = None


def sec_circuit(net, D, P, out, arr):
    import torch
    from scipy.special import logsumexp
    dev = net.dev; A = attributions(net, D, P); log('circuit: attributions done')
    flat = A.reshape(10, NBLK * NC); rank = {k: np.argsort(-flat[k]) for k in range(10)}; rank['all'] = np.argsort(-flat.sum(0))
    g = np.random.default_rng(SEED + 6)
    variants = [('floor', None, 0, False)]
    for K in P['CSIZES']:
        for k in list(range(10)) + ['all']: variants.append((f'cls_{CL[k] if k != "all" else "all"}_K{K}', rank[k][:K], K, False))
        for r in range(P['NRSET']): variants.append((f'random{r}_K{K}', g.choice(NBLK * NC, K, replace=False), K, False))
    Kx = 32 if 32 in P['CSIZES'] else P['CSIZES'][-1]
    variants.append(('floor_auxlive', None, 0, True))
    for k in range(10): variants.append((f'cls_{CL[k]}_K{Kx}_auxlive', rank[k][:Kx], Kx, True))
    for r in range(P['NRSET']): variants.append((f'random{r}_K{Kx}_auxlive', g.choice(NBLK * NC, Kx, replace=False), Kx, True))
    N = len(D['y2']); Ls = {v[0]: np.zeros((N, 10), np.float32) for v in variants}
    log(f'circuit: {len(variants)} variants on N {N}')

    def fn(e, sl, L0):
        cache = {}
        net.rec = lambda b, mv, s_: cache.__setitem__(b, (mv[0].clone(), s_[0].clone()))
        try: L0b = net.logits(e)
        finally: net.rec = None
        dt = net.meta['donor_tok']
        for name, keep, K, auxlive in variants:
            km = torch.zeros(NBLK, NC, dtype=torch.bool, device=dev)
            if keep is not None:
                kt = torch.as_tensor(np.asarray(keep), device=dev); km[kt // NC, kt % NC] = True

            def op(b, mv, s_, km=km, auxlive=auxlive):
                cm, cs = cache[b]
                h = torch.where(km[b][None, :, None], mv[0], cm[dt]); s2 = s_[0] if auxlive else cs[dt]
                return h[None], s2[None]
            net.op = op
            try: Ls[name][sl] = net.logits(e).float().cpu().numpy()
            finally: net.op = None
    net.run(D['J2'], P, fn)
    y = D['y2']; L0 = D['L2']; p0 = L0.argmax(1); c0 = p0 == y; W = boot_W(N, 1000, SEED + 7)
    res = dict(N=int(N), acc_unedited=boot_mean(c0, W), confusion_unedited=confusion(y, p0).tolist(),
               recall_unedited={c: float((p0[y == i] == i).mean()) for i, c in enumerate(CL)},
               auc_unedited={c: auc_np(L0[:, i] - logsumexp(np.delete(L0, i, 1), 1), y == i) for i, c in enumerate(CL)},
               ranking='mean |attribution| on l_k over NRANK jets per class of sample 1', variants={})
    for name, keep, K, auxlive in variants:
        L = Ls[name]; p = L.argmax(1); c1 = p == y
        r = dict(K=K, aux_live=auxlive, acc=boot_mean(c1, W), mcnemar_vs_unedited=mcnemar(c0, c1), same_call=float((p == p0).mean()),
                 recall={c: float((p[y == i] == i).mean()) for i, c in enumerate(CL)}, confusion=confusion(y, p).tolist(),
                 auc={c: auc_np(L[:, i] - logsumexp(np.delete(L, i, 1), 1), y == i) for i, c in enumerate(CL)})
        if keep is not None: r['kept'] = [f'b{int(u) // NC}c{int(u) % NC}' for u in keep]
        res['variants'][name] = r
    # summary per K: class circuit's own-class recall / AUC vs random sets
    summ = {}
    for K in P['CSIZES']:
        own_rec = [res['variants'][f'cls_{c}_K{K}']['recall'][c] for c in CL]; own_auc = [res['variants'][f'cls_{c}_K{K}']['auc'][c] for c in CL]
        rnd_rec = [np.mean([res['variants'][f'random{r}_K{K}']['recall'][c] for r in range(P['NRSET'])]) for c in CL]
        rnd_auc = [np.mean([res['variants'][f'random{r}_K{K}']['auc'][c] for r in range(P['NRSET'])]) for c in CL]
        summ[f'K{K}'] = dict(own_recall_mean=float(np.mean(own_rec)), random_recall_mean=float(np.mean(rnd_rec)),
                             own_auc_mean=float(np.mean(own_auc)), random_auc_mean=float(np.mean(rnd_auc)),
                             own_recall=dict(zip(CL, map(float, own_rec))), own_auc=dict(zip(CL, map(float, own_auc))),
                             all_class_acc=res['variants'][f'cls_all_K{K}']['acc'], random_acc=[res['variants'][f'random{r}_K{K}']['acc'][0] for r in range(P['NRSET'])],
                             n_classes_own_recall_above_all_random=int(sum(own_rec[i] > max(res['variants'][f'random{r}_K{K}']['recall'][CL[i]] for r in range(P['NRSET'])) for i in range(10))))
        log(f"circuit K{K}: own-class recall mean {np.mean(own_rec):.3f} (random sets {np.mean(rnd_rec):.3f}, unedited {np.mean(list(res['recall_unedited'].values())):.3f}); "
            f"own AUC {np.mean(own_auc):.3f} vs random {np.mean(rnd_auc):.3f}; all-class circuit acc {100 * res['variants'][f'cls_all_K{K}']['acc'][0]:.2f} %, "
            f"random acc {[round(100 * x, 2) for x in summ[f'K{K}']['random_acc']]}")
    fl = res['variants']['floor']; log(f"circuit floor: acc {100 * fl['acc'][0]:.2f} %, mean recall {np.mean(list(fl['recall'].values())):.3f}; "
                                       f"aux-live floor acc {100 * res['variants']['floor_auxlive']['acc'][0]:.2f} %")
    res['summary'] = summ; res['attribution_top5'] = {(CL[k] if k != 'all' else 'all'): [f'b{int(u) // NC}c{int(u) % NC}' for u in rank[k][:5]] for k in rank}
    out['circuit'] = res; arr['circuit_attr'] = A


def NRCAL(P):
    return 1 if P['smoke'] else 10


def sec_calib(net, D, P, out, arr):
    import torch
    from interp.part_core import kin, build, jmass, sub
    dev = net.dev; y2 = D['y2']; rows = np.where((y2 == iW) | (y2 == iZ))[0]
    J = sub(D['J2'], rows); pt, de, dp, ms = kin(J); Jb = build(J, pt, de, dp, ms); Jd = build(J, pt, de * 1.05, dp * 1.05, ms)
    dlogm = float(np.mean(np.log(jmass(Jd) + 1e-3) - np.log(jmass(Jb) + 1e-3)))
    S = D['S_mv'][:, OBS.index('log_m')]; top = np.argsort(-S)
    g = np.random.default_rng(SEED + 8)
    Ks = (1, 8, 32) if not P['smoke'] else (1, 8)
    R0 = D['rots'][0][0]
    # rotation-0 directions ranked by their own held-out log-m score (from the atlas arrays if present)
    Srot = out.get('_S_rot_raw0')
    sets = [('all', np.arange(NBLK * NC), 'chan')]
    for K in Ks:
        sets.append((f'top{K}_logm', top[:K], 'chan'))
        for r in range(NRCAL(P)): sets.append((f'random{r}_K{K}', g.choice(NBLK * NC, K, replace=False), 'chan'))
        if Srot is not None: sets.append((f'rot0_top{K}_logm', np.argsort(-Srot)[:K], 'dir'))
    N = len(rows); dd = {k: np.zeros(N) for k in ['dil'] + [s[0] for s in sets]}
    BS = P['BS']
    with torch.no_grad():
        for s0 in range(0, N, BS):
            sl = np.arange(s0, min(N, s0 + BS))
            ed = net.embed(Jd, sl); cache = {}
            net.rec = lambda b, mv, s_: cache.__setitem__(b, mv[0].clone())
            try: Ld = net.logits(ed).float()
            finally: net.rec = None
            Td = net.meta['T']
            eb = net.embed(Jb, sl); assert net.meta['T'] == Td, 'token layout differs between unedited and dilated jets'
            Lb = net.logits(eb).float(); d0 = (Lb[:, iW] - Lb[:, iZ])
            dd['dil'][sl] = ((Ld[:, iW] - Ld[:, iZ]) - d0).cpu().numpy()
            for name, idx, kind in sets:
                if kind == 'chan':
                    km = torch.zeros(NBLK, NC, dtype=torch.bool, device=dev); it = torch.as_tensor(idx, device=dev); km[it // NC, it % NC] = True

                    def op(b, mv, s_, km=km):
                        return torch.where(km[b][None, :, None], cache[b], mv[0])[None], s_
                else:
                    Ub = {}
                    for u in idx:
                        Ub.setdefault(int(u) // NC, []).append(R0[int(u) // NC, int(u) % NC])
                    Ub = {b: torch.stack(v, 1) for b, v in Ub.items()}                          # (32, k) orthonormal columns

                    def op(b, mv, s_, Ub=Ub):
                        if b not in Ub: return mv, s_
                        U = Ub[b]; h = mv[0]; dlt = cache[b] - h
                        return (h + torch.einsum('ck,dk,tdm->tcm', U, U, dlt))[None], s_
                net.op = op
                try: Lp = net.logits(eb).float()
                finally: net.op = None
                dd[name][sl] = ((Lp[:, iW] - Lp[:, iZ]) - d0).cpu().numpy()
    W = boot_W(N, 1000, SEED + 9); den = dd['dil']; bden = W @ den
    res = dict(N=int(N), jets='true W and Z jets of sample 2', edit='every particle deta, dphi x 1.05 about the stored axis (fixed pT, mass)',
               dlogm_mean=dlogm, readout='d = l_W - l_Z (logits)', dd_dilation=boot_mean(den, W), sets={})
    for name, idx, kind in sets:
        num = dd[name]; bn = W @ num; Rb = bn / bden
        res['sets'][name] = dict(R=float(num.sum() / den.sum()), R_ci=[float(np.percentile(Rb, 2.5)), float(np.percentile(Rb, 97.5))], K=int(len(idx)),
                                 units=[f'b{int(u) // NC}c{int(u) % NC}' for u in idx[:32]] if name != 'all' else 'all 384')
    for K in Ks:
        rr = [res['sets'][f'random{r}_K{K}']['R'] for r in range(NRCAL(P))]
        res[f'K{K}'] = dict(top=res['sets'][f'top{K}_logm']['R'], top_ci=res['sets'][f'top{K}_logm']['R_ci'], random_mean=float(np.mean(rr)),
                            random_min=float(np.min(rr)), random_max=float(np.max(rr)),
                            rot0=res['sets'].get(f'rot0_top{K}_logm', {}).get('R'))
        log(f"calib K{K}: R_patch top log-m channels {res[f'K{K}']['top']:.3f} {np.round(res[f'K{K}']['top_ci'], 3).tolist()} vs random {np.mean(rr):.3f} "
            f"[{np.min(rr):.3f}, {np.max(rr):.3f}], rotated top dirs {res[f'K{K}']['rot0']}")
    log(f"calib: all 384 channels R_patch {res['sets']['all']['R']:.3f}; dilation dd {res['dd_dilation']}; dlog m {dlogm:.4f}")
    out['calib'] = res


# ------------------------------------------------------------------------------------------------ figure
def figure(out, path):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    matplotlib.rcParams['text.usetex'] = False
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    a = ax[0]; s = out['atlas']['summary']['mv']['R2_0p5']; s3 = out['atlas']['summary']['mv']['s0p3']
    names = ['trained\nchannels', 'rotated\n(raw)', 'rotated\n(std)', 'untrained', 'channel-\npermuted']
    for i, (ss, lab, off) in enumerate(((s, 'clean (R$^2\\geq$0.5)', -0.2), (s3, 'clean (score$\\geq$0.3)', 0.2))):
        v = [ss['trained']['clean'], ss['clean_vs_rot_raw']['rot_mean'], ss['clean_vs_rot_std']['rot_mean'], ss['untrained']['clean'], ss['permuted']['clean']]
        lo = [0, v[1] - ss['clean_vs_rot_raw']['rot_min'], v[2] - ss['clean_vs_rot_std']['rot_min'], 0, 0]
        hi = [0, ss['clean_vs_rot_raw']['rot_max'] - v[1], ss['clean_vs_rot_std']['rot_max'] - v[2], 0, 0]
        a.bar(np.arange(5) + off, v, 0.38, yerr=[lo, hi], label=lab, color=['#3465a4', '#999999', '#bbbbbb', '#e9b96e', '#ad7fa8'], alpha=1 if i == 0 else 0.55, capsize=3)
    a.set_xticks(range(5)); a.set_xticklabels(names, fontsize=8); a.set_ylabel('fraction of the 384 hidden multivector channels')
    a.set_title('(a) readable channels vs rotated bases\n(error bar: range over 10 rotations; dark R$^2\\geq$0.5, light score$\\geq$0.3)', fontsize=9)
    a = ax[1]; st = out.get('steer')
    if st:
        labs = [r['label'].replace('cls_', '') for r in st['labels']]; x = np.arange(len(labs))
        a.bar(x - 0.2, [r['share_channel'][0] for r in st['labels']], 0.4, label='top channel', color='#3465a4')
        a.bar(x + 0.2, [r['share_rot'][0] for r in st['labels']], 0.4, label='top rotated direction', color='#999999')
        for i, r in enumerate(st['labels']):
            if r['p_holm'] < 0.01: a.text(i - 0.2, r['share_channel'][0] + 0.01, '*', ha='center')
        a.axhline(0.5, color='k', lw=0.6, ls=':'); a.set_xticks(x); a.set_xticklabels(labs, rotation=70, fontsize=7); a.set_ylim(0, 1.05)
        a.set_ylabel('share of steering steps that move the\ntarget class log-odds the predicted way'); a.legend(fontsize=7)
        a.set_title('(b) steering the top channel per label by $\\pm$1, 2, 4 s.d.\n(* channel > rotated, Holm p < 0.01; N 2,000 jets)', fontsize=9)
    a = ax[2]; ci = out.get('circuit')
    if ci:
        Ks = sorted(int(k[1:]) for k in ci['summary']); sm = ci['summary']
        a.plot(Ks, [sm[f'K{K}']['own_recall_mean'] for K in Ks], 'o-', color='#3465a4', label='class-k circuit: class-k recall (mean over 10)')
        a.plot(Ks, [sm[f'K{K}']['random_recall_mean'] for K in Ks], 's--', color='#999999', label='random channel set: class-k recall')
        a.plot(Ks, [sm[f'K{K}']['all_class_acc'][0] for K in Ks], '^-', color='#4e9a06', label='all-class circuit: 10-class accuracy')
        a.axhline(np.mean(list(ci['recall_unedited'].values())), color='k', lw=0.8, label='unedited (mean recall)')
        a.axhline(ci['variants']['floor']['acc'][0], color='r', lw=0.8, ls=':', label='floor: no channel kept (accuracy)')
        a.set_xscale('log', base=2); a.set_xticks(Ks); a.set_xticklabels(Ks); a.set_xlabel('channels kept (of 384); all others and all hidden scalars resampled')
        a.set_ylim(0, 1); a.legend(fontsize=7); a.set_title(f"(c) literal channel circuits (sample 2, N {ci['N']:,})", fontsize=9)
    fig.suptitle('P54: L-GATr hidden MLP channels (final checkpoint, JetClass test_20M _106)', fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); log('wrote', path)


# ------------------------------------------------------------------------------------------------ main
def main_one(smoke, nets, full):
    P = params(smoke); TAG = P['TAG']
    out = dict(node='P54', params={k: v for k, v in P.items()}, checkpoint=CKPT,
               sample=f'JetClass test_20M <class>_{FIDX}.root, 2,000 random rows per class (default_rng({SEED})), split at random within class '
                      f'into sample 1 / sample 2 (1,000 per class each)' + (f'; smoke: {P["PER"] // 2} random jets per class of each' if smoke else ''))
    arr = {}; fn = RES / f'p54_lgatr_channel_atlas{TAG}.json'
    D = dict(full)
    if smoke:
        g = np.random.default_rng(1)
        for k in ('1', '2'):
            y = full['y' + k]; idx = np.sort(np.concatenate([g.choice(np.where(y == c)[0], P['PER'] // 2, replace=False) for c in range(10)]))
            from interp.part_core import sub
            D['J' + k] = sub(full['J' + k], idx); D['y' + k] = y[idx]; D['O' + k] = full['O' + k][idx]
    log(f'{"smoke" if smoke else "full"}: sample 1 N {len(D["y1"])}, sample 2 N {len(D["y2"])}')
    L = nets['trained'].run(full['J0'], P, rows=None if not smoke else np.arange(0, len(full['J0']['label']), 50))
    y0 = full['J0']['label'].astype(int) if not smoke else full['J0']['label'].astype(int)[::50]
    out['P1_sample'] = dict(N=int(len(y0)), acc=float((L.argmax(1) == y0).mean()))
    log(f"P1 sample accuracy {100 * out['P1_sample']['acc']:.2f} % (N {out['P1_sample']['N']}; expected 83.73 % at N 3,000)")
    secs = ARGS.sections.split(',')
    for name, f in (('atlas', sec_atlas), ('steer', sec_steer), ('ablate', sec_ablate), ('circuit', sec_circuit), ('calib', sec_calib)):
        if name not in secs: continue
        if name != 'atlas' and 'S_mv' not in D: log(f'skip {name}: atlas missing'); continue
        t = time.time()
        try:
            if name == 'atlas':
                f(nets, D, P, out, arr)
                if 'S_mv_rot_raw0' in arr: out['_S_rot_raw0'] = arr['S_mv_rot_raw0'][:, OBS.index('log_m')]
            else: f(nets['trained'], D, P, out, arr)
        except Exception:
            out[f'{name}_error'] = traceback.format_exc(); log(f'ERROR in {name}:\n' + out[f'{name}_error'])
        log(f'section {name} {time.time() - t:.0f}s')
        jdump({k: v for k, v in out.items() if not k.startswith('_')}, fn)
    np.savez_compressed(RES / f'p54_arrays{TAG}.npz', **arr)
    try: figure(out, RES / f'p54_lgatr_channel_atlas{TAG}.png')
    except Exception: log('figure failed:\n' + traceback.format_exc())
    return out


def main():
    import torch
    if ARGS.figure_only:
        figure(json.loads((RES / 'p54_lgatr_channel_atlas.json').read_text()), RES / 'p54_lgatr_channel_atlas.png'); return
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    log('device', dev, torch.__version__)
    J1, T1, J2, T2 = read_sample(2000, np.random.default_rng(SEED))
    from interp.part_core import sample
    full = dict(J1=J1, J2=J2, y1=J1['label'].astype(int), y2=J2['label'].astype(int),
                O1=np.nan_to_num(observables(J1, T1, dev)), O2=np.nan_to_num(observables(J2, T2, dev)), J0=sample(300, 3000))
    log(f'samples read: {len(full["y1"])} + {len(full["y2"])} jets; P1 sample {len(full["J0"]["label"])}')
    nets = dict(trained=Net('trained', dev)); log('loaded trained L-GATr', CKPT)
    nets['untrained'] = Net('untrained', dev); log('loaded untrained L-GATr (seed 5454)')
    if not ARGS.no_smoke:
        o = main_one(True, nets, full); bad = [k for k in o if k.endswith('_error')]
        if bad: log('smoke errors in', bad, '-> stop'); sys.exit(1)
        log('smoke ok')
    if not ARGS.smoke_only: main_one(False, nets, full)
    log('done')


if __name__ == '__main__':
    main()
