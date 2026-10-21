"""P44 (loop C, round C10, exploratory): L-GATr internals. Which multivector grades and which symmetry-breaking reference
multivectors (spurions) carry the class decision and the jet mass in our L-GATr (run anrunw-ai-c-p5-lgatr2, authors' jctagging
config, 2M random-file JetClass jets; latest checkpoint at job start, or the final one if present).

Terms. A multivector of the space-time algebra has 16 components, grouped by grade: grade 0 = 1 scalar (index 0), grade 1 = 4
vector components (t, x, y, z; 1-4), grade 2 = 6 bivector components (tx, ty, tz, xy, xz, yz; 5-10), grade 3 = 4 axial-vector
(trivector) components (11-14), grade 4 = 1 pseudoscalar (15). Every L-GATr token (one per particle, one global token that is
read out, one token per spurion) carries 16 hidden multivector channels (16 x 16 = 256 numbers) and 32 auxiliary scalar channels
(s, Lorentz-invariant by construction). Grade 0 and the s channels are Lorentz invariant ("invariant channels", 48 per token);
grades 1-4 transform under boosts and rotations ("covariant channels", 240 per token). Equivariance: transforming every input
multivector (particles and spurions) by the same Lorentz transformation transforms every hidden multivector the same way and
leaves invariant channels and the logits unchanged. Spurions: fixed reference multivectors given as extra tokens that break the
symmetry to the subgroup that keeps them: here the x-y plane bivector (beam reference 'xyplane', invariant under rotations about
the beam and boosts along it) and the time direction (1, 0, 0, 0) (invariant under rotations only). The authors' per-particle
scalar inputs (log pT, log E, log pT/pT_jet, log E/E_jet, dphi, deta, dR about the particle-sum axis) are a second route that
breaks the symmetry: log E and log E/E_jet change under a boost along the beam.

Sample: JetClass val_5M <class>_122.root (not in L-GATr's training or validation files, data/jetclass2m/selection.json), 1,000
random rows per class (np.random.default_rng(4444), N 10,000), shuffled once (same rng) so batches mix classes. Inputs rebuilt
with part_core.build (as P5/P39). Unedited accuracy and 10-class confusion first; P1 sample (rows 3000-3300 of _100, N 3,000)
for the known 84.27 % of it154999.

(a) grade ablations at the output of each of the 12 blocks: one group of components of every token set to its sample mean
    (per token type: global, beam spurion, time spurion, particle; means over this sample) or resampled from another jet (donor =
    a random other jet of the batch; particle k takes the donor's particle k mod n_donor). Groups: s (32), g0 (16), g1 (64),
    g2 (96), g3 (64), g4 (16), inv = g0 + s (48), cov = g1-g4 (240). Control (stage ctl, 2,000-jet subset = first 2,000 of the
    shuffled sample): a random set of the same number of the 288 components per token, mean-ablated, 50 draws per block x count.
(b) spurion route, on all 10,000 jets: rigid azimuthal rotation 0.06 and 0.3 rad (exact symmetry), and a longitudinal boost
    of rapidity y in {0.1, 0.3, 0.5} (every particle's rapidity + y, pT and mass fixed, so jet mass and all pair masses m_ij are
    unchanged) split into routes (mv = particle four-momenta, spur = time spurion, feat = scalar features):
      A  mv boosted, spurions fixed, feat recomputed            the physical edit
      B  mv boosted, time spurion boosted by +y, feat recomputed  (= F by equivariance)
      C  mv boosted, time spurion boosted, feat held            exact symmetry: predicted 0 change (float)
      D  mv boosted, spurions fixed, feat held                  spurion route
      E  mv unchanged, time spurion boosted by -y, feat held    spurion swap (= D by equivariance)
      F  mv unchanged, spurions fixed, feat recomputed from the boosted jet   scalar-feature route
    plus the ParT-style P1 build edits deta + 0.1 (pseudorapidity shift at fixed pT, mass) and dphi + 0.06.
(c) mass read-out: ridge regression of log m_inv (GeV) from the invariant channels (global token 48 + particle-token mean 48)
    per block, fit on 5,000 jets, tested on the other 5,000; calibrated by a real angular dilation (all deta, dphi x 1.1 about
    the stored axis) and mass-fixed edits (rigid phi + 0.06, boost y 0.3 routes A and D).
    DAS (stage das): 1-d distributed alignment search in the 48 invariant channels of every token at block k: base = jet,
    source = the same jet dilated by 1.1 or 1/1.1 (random per jet); the base's projection on a learned unit direction u is
    replaced by the source's; trained (KL to the source logits) on 3,000 jets, tested on 3,000 others. IIA (interchange
    intervention accuracy) = fraction of test jets whose call changes under the dilation for which the intervened call equals
    the source call. Controls: 20 random unit directions; all 48 invariant channels swapped (upper bound of the subspace); the
    untrained L-GATr (random init, torch seed 0, same config) with the same procedure.

  ./sync.sh && ./part_job.sh c-p44-run '/j-jepa-vol/aaron-interp/env-lgatr/bin/python -m interp.p44_lgatr_grades --stage main'
  ./sync.sh && ./part_job.sh c-p44-ctl '/j-jepa-vol/aaron-interp/env-lgatr/bin/python -m interp.p44_lgatr_grades --stage ctl'
  ./sync.sh && ./part_job.sh c-p44-das '/j-jepa-vol/aaron-interp/env-lgatr/bin/python -m interp.p44_lgatr_grades --stage das'
  ./sync.sh && ./job.sh c-p44-an 1 2 'python -m interp.p44_lgatr_grades --stage analyze'
  (--smoke: 40 rows per class, 2 control draws, few DAS steps; stages can be joined with commas)
-> $AI_RESULTS/lgatr/p44_{main,ctl,das}.npz/json, p44_lgatr_grades.json, p44_lgatr_grades.png
"""
import argparse, glob, json, os, sys, time, traceback
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('JETDISTILL_TAGGER', 'part')
ap = argparse.ArgumentParser(); ap.add_argument('--stage', required=True); ap.add_argument('--smoke', action='store_true')
ap.add_argument('--sections', default='base,grades,spurion,probe'); ARGS = ap.parse_args()
SMOKE = ARGS.smoke
SEED, FIDX = 4444, 122
PER = 40 if SMOKE else 1000
NC = 100 if SMOKE else 2000            # control subset
NDRAW = 2 if SMOKE else 50
NDAS = 150 if SMOKE else 3000           # DAS train / test jets each
DAS_EPOCHS = 1 if SMOKE else 6
DAS_BLOCKS = [0, 5, 11] if SMOKE else [0, 2, 4, 6, 8, 10, 11]
NRAND = 3 if SMOKE else 20
NB = 200 if SMOKE else 2000
BS = 500
VAL = Path('/j-jepa-vol/JetClass/Pythia/val_5M')
PRE = {'QCD': 'ZJetsToNuNu', 'Hbb': 'HToBB', 'Hcc': 'HToCC', 'Hgg': 'HToGG', 'H4q': 'HToWW4Q', 'Hqql': 'HToWW2Q1L', 'Zqq': 'ZToQQ',
       'Wqq': 'WToQQ', 'Tbqq': 'TTBar', 'Tbl': 'TTBarLep'}
CL = ['QCD', 'Hbb', 'Hcc', 'Hgg', 'H4q', 'Hqql', 'Zqq', 'Wqq', 'Tbqq', 'Tbl']
iW, iZ = CL.index('Wqq'), CL.index('Zqq')
RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'lgatr'
TAG = '_smoke' if SMOKE else ''
RUN = '/j-jepa-vol/aaron-interp/models/lgatr/runs/jc2m/jc2m_s0a/models'
NBLK = 12
# flattened per-token components: mv channel c, blade i -> c * 16 + i (0..255); s channel j -> 256 + j
BLADES = {'g0': [0], 'g1': [1, 2, 3, 4], 'g2': [5, 6, 7, 8, 9, 10], 'g3': [11, 12, 13, 14], 'g4': [15]}
GROUPS = {g: [c * 16 + i for c in range(16) for i in b] for g, b in BLADES.items()}
GROUPS = {'s': list(range(256, 288)), **GROUPS}
GROUPS['inv'] = sorted(GROUPS['g0'] + GROUPS['s'])
GROUPS['cov'] = sorted(GROUPS['g1'] + GROUPS['g2'] + GROUPS['g3'] + GROUPS['g4'])
GNAMES = ['s', 'g0', 'g1', 'g2', 'g3', 'g4', 'inv', 'cov']
COUNTS = sorted({len(GROUPS[g]) for g in GNAMES})          # 16, 32, 48, 64, 96, 240
INV = GROUPS['inv']
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def jdump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(obj, indent=1, default=float)); log('wrote', path)


# ------------------------------------------------------------------------------------------------ sample
def read_sample():
    import uproot
    from interp.part_core import read_root, sub, cat
    rng = np.random.default_rng(SEED); parts = []; rows_all = []
    for c in CL:
        path = VAL / f'{PRE[c]}_{FIDX}.root'; N = uproot.open(path)['tree'].num_entries
        rows = np.sort(rng.choice(N, PER, replace=False)); ch = []
        for s in range(0, N, 25000):
            r = rows[(rows >= s) & (rows < s + 25000)]
            if len(r): ch.append(sub(read_root(path, s, min(s + 25000, N)), r - s))
        J = cat(ch); assert (J['label'] == CL.index(c)).all(), c
        parts.append(J); rows_all.append(np.stack([np.full(len(rows), CL.index(c)), rows], 1))
    J = cat(parts); rows = np.concatenate(rows_all); perm = rng.permutation(len(rows))
    return sub(J, perm), rows[perm]


# ------------------------------------------------------------------------------------------------ model plumbing
class LG:
    def __init__(self, ckpt, dev):
        from interp.p5_lgatr_launch import load_lgatr, LG as SRC
        sys.path.insert(0, SRC)
        self.model, self.cfg = load_lgatr(ckpt, dev); self.dev = dev
        for p in self.model.parameters(): p.requires_grad_(False)
        from experiments.tagging.embedding import dense_to_sparse_jet, embed_tagging_data_into_ga
        from experiments.tagging.wrappers import xformers_sa_mask
        from gatr.interface import embed_vector
        self._d2s, self._emb, self._mask, self._ev = dense_to_sparse_jet, embed_tagging_data_into_ga, xformers_sa_mask, embed_vector

    def _embed1(self, J, sl):
        import torch
        from interp.p5_lgatr_launch import to_lgatr_inputs
        from interp.part_core import sub
        vec, feat = to_lgatr_inputs(sub(J, sl))
        fm, sc, ptr = self._d2s(torch.from_numpy(vec).to(self.dev), torch.from_numpy(feat).to(self.dev))
        return self._emb(fm, sc, ptr, self.cfg.data)

    def embed(self, J_mv, sl, J_feat=None, spur_y=0.0):
        """embedding with particle four-momenta from J_mv; scalar features from J_feat (if given, else J_mv); time spurion
        boosted by rapidity spur_y. Adds tt = token type (0 global, 1 beam spurion, 2 time spurion, 3 particle), pos, start."""
        import torch
        e = self._embed1(J_mv, sl)
        b = e['batch']; nb = int(b.max()) + 1; cnt = torch.bincount(b, minlength=nb); start = torch.cumsum(cnt, 0) - cnt
        pos = torch.arange(len(b), device=b.device) - start[b]; tt = torch.clamp(pos, max=3)
        assert torch.equal(e['is_global'], pos == 0), 'global token not first'
        mvz = e['mv'][:, 0]
        assert torch.allclose(mvz[tt == 1], torch.eye(16, device=b.device)[8].expand((tt == 1).sum(), 16)), 'beam spurion not xy bivector at pos 1'
        assert torch.allclose(mvz[tt == 2], torch.eye(16, device=b.device)[1].expand((tt == 2).sum(), 16)), 'time spurion not at pos 2'
        if J_feat is not None:
            f = self._embed1(J_feat, sl); assert f['s'].shape == e['s'].shape and torch.equal(f['batch'], b), 'feature jet structure differs'
            e['s'] = f['s']
        if spur_y:
            t = torch.zeros(16, device=b.device, dtype=e['mv'].dtype); t[1] = float(np.cosh(spur_y)); t[4] = float(np.sinh(spur_y))
            e['mv'] = e['mv'].clone(); e['mv'][tt == 2, 0] = t      # xy-plane bivector is invariant under a boost along z
        e.update(tt=tt, pos=pos, start=start, cnt=cnt, nb=nb)
        return e

    def upto(self, e, k, cache=False):
        """run linear_in and blocks 0..k; returns (h_mv, h_s, mask[, list of per-block outputs])"""
        m = self.model; net = m.net
        mask = self._mask(e['batch'], materialize=not m.force_xformers)
        if not m.force_xformers and mask.dtype != e['s'].dtype: mask = mask.to(e['s'].dtype)
        h_mv, h_s = net.linear_in(e['mv'].unsqueeze(0), scalars=e['s'].unsqueeze(0)); outs = []
        for j in range(k + 1):
            h_mv, h_s = net.blocks[j](h_mv, scalars=h_s, attention_mask=mask)
            if cache: outs.append((h_mv, h_s))
        return h_mv, h_s, mask, outs

    def rest(self, e, mask, k, h_mv, h_s):
        """blocks k+1..11 then linear_out and the global-token read-out; logits in CL order"""
        from interp.p5_lgatr_launch import TO_CL
        m = self.model; net = m.net
        for j in range(k + 1, NBLK):
            h_mv, h_s = net.blocks[j](h_mv, scalars=h_s, attention_mask=mask)
        o_mv, o_s = net.linear_out(h_mv, scalars=h_s)
        return m.extract_from_ga(o_mv, o_s, e['batch'], e['is_global'])[:, TO_CL]

    def logits(self, J_mv, J_feat=None, spur_y=0.0, bs=BS):
        import torch
        N = len(J_mv['label']); out = np.zeros((N, 10), np.float32)
        with torch.no_grad():
            for s in range(0, N, bs):
                sl = np.arange(s, min(s + bs, N)); e = self.embed(J_mv, sl, J_feat, spur_y)
                h_mv, h_s, mask, _ = self.upto(e, NBLK - 1); out[sl] = self.rest(e, mask, NBLK - 1, h_mv, h_s).float().cpu().numpy()
        return out


def flat(h_mv, h_s):
    import torch
    T = h_mv.shape[1]; return torch.cat([h_mv[0].reshape(T, 256), h_s[0]], 1)


def unflat(X):
    T = X.shape[0]; return X[:, :256].reshape(1, T, 16, 16), X[:, 256:].unsqueeze(0)


def latest_ckpt():
    fin = sorted(p for p in glob.glob(f'{RUN}/*.pt') if '_it' not in Path(p).name)
    if fin: return fin[-1]
    c = sorted(glob.glob(f'{RUN}/*_it*.pt'), key=lambda p: int(p.split('_it')[-1].split('.')[0])); return c[-1]


# ------------------------------------------------------------------------------------------------ statistics helpers
def boot(x, rng=None, nb=None):
    rng = rng or np.random.default_rng(0); x = np.asarray(x, float); n = len(x)
    bs = np.array([x[rng.integers(0, n, n)].mean() for _ in range(nb or NB)])
    return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def mcnemar(c0, c1):
    from scipy.stats import binomtest
    b, c = int((c0 & ~c1).sum()), int((~c0 & c1).sum())
    return dict(lost=b, gained=c, p=float(binomtest(b, b + c).pvalue) if b + c else 1.0)


def confusion(y, p):
    M = np.zeros((10, 10), int); np.add.at(M, (y, p), 1); return M


def compare(L0, L1, y, rng=None):
    p0, p1 = L0.argmax(1), L1.argmax(1); W, Z = y == iW, y == iZ
    return dict(acc=float((p1 == y).mean()), acc_before=float((p0 == y).mean()), calls_changed=boot((p0 != p1).astype(float), rng),
                mcnemar=mcnemar(p0 == y, p1 == y), W_to_Z=[float((p0[W] == iZ).mean()), float((p1[W] == iZ).mean())],
                Z_to_W=[float((p0[Z] == iW).mean()), float((p1[Z] == iW).mean())],
                max_abs_dlogit=float(np.abs(L1 - L0).max()), mean_abs_dlogit=float(np.abs(L1 - L0).mean()))


def pair_masses(J, n=200):
    m = J['mask'][:n] > 0; P = np.stack([J['energy'][:n], J['px'][:n], J['py'][:n], J['pz'][:n]], -1).astype(np.float64)
    S = P[:, :, None, :] + P[:, None, :, :]; m2 = S[..., 0] ** 2 - (S[..., 1:] ** 2).sum(-1)
    return np.sqrt(np.maximum(m2, 0)), m[:, :, None] & m[:, None, :]


# ------------------------------------------------------------------------------------------------ stage main
def sample_means(lg, J):
    """per block, per token type (global, beam, time, particle): mean of the 288 flattened components over this sample"""
    import torch
    S = torch.zeros(NBLK, 4, 288, dtype=torch.float64, device=lg.dev); C = torch.zeros(4, dtype=torch.float64, device=lg.dev)
    N = len(J['label'])
    with torch.no_grad():
        for s in range(0, N, BS):
            sl = np.arange(s, min(s + BS, N)); e = lg.embed(J, sl); _, _, _, outs = lg.upto(e, NBLK - 1, cache=True)
            oh = torch.nn.functional.one_hot(e['tt'], 4).double()
            C += oh.sum(0)
            for k, (hm, hs) in enumerate(outs): S[k] += oh.T @ flat(hm, hs).double()
    return (S / C[None, :, None]).float()


def donor_index(e, rng):
    """token -> donor token: jet i takes jet d(i) = (i + shift) mod B (shift random in 1..B-1); global / spurion tokens from the
    donor's same token; particle q from the donor's particle q mod n_donor"""
    import torch
    B = e['nb']; sh = int(rng.integers(1, B)) if B > 1 else 0
    b = e['batch']; d = (b + sh) % B; pos = e['pos']; npart = e['cnt'] - 3
    q = torch.where(pos >= 3, 3 + (pos - 3) % npart[d], pos)
    return e['start'][d] + q


def sec_base(lg, J, Jb, y, out, npz):
    from interp.part_core import sample, kin, build
    L_raw = lg.logits(J); L0 = lg.logits(Jb)
    # forward through the split plumbing vs the wrapper's own forward (one batch)
    import torch
    with torch.no_grad():
        sl = np.arange(0, min(BS, len(y))); e = lg.embed(Jb, sl); Lw = lg.model({k: e[k] for k in ('mv', 's', 'is_global', 'batch')})
        from interp.p5_lgatr_launch import TO_CL
        dsplit = float((Lw[:, TO_CL].float().cpu().numpy() - L0[sl]).__abs__().max())
    perm = np.random.default_rng(1).permutation(len(y)); Lp = lg.logits(sub_(Jb, perm))
    floor = float(np.abs(Lp - L0[perm]).max())
    J1 = sample(50 if SMOKE else 300, 3000); y1 = J1['label'].astype(int); L1 = lg.logits(build(J1, *kin(J1)))
    p0 = L0.argmax(1)
    out['base'] = dict(N=int(len(y)), acc_stored=float((L_raw.argmax(1) == y).mean()), acc=float((p0 == y).mean()),
                       acc_ci=boot((p0 == y).astype(float)), per_class_recall={c: float((p0[y == i] == i).mean()) for i, c in enumerate(CL)},
                       confusion=confusion(y, p0).tolist(), split_vs_wrapper_max_abs_dlogit=dsplit, batch_order_max_abs_dlogit=floor,
                       P1_sample=dict(N=int(len(y1)), acc=float((L1.argmax(1) == y1).mean())))
    npz['L0'] = L0; npz['L_raw'] = L_raw
    log(f"unedited acc {out['base']['acc']:.4f} (stored inputs {out['base']['acc_stored']:.4f}); P1 sample {out['base']['P1_sample']['acc']:.4f} "
        f"N {len(y1)}; split-vs-wrapper max|dL| {dsplit:.2e}; batch-order max|dL| {floor:.2e}")
    return L0


def sub_(J, rows):
    from interp.part_core import sub
    return sub(J, rows)


def sec_grades(lg, Jb, y, L0, out, npz):
    import torch
    M = sample_means(lg, Jb); npz['means'] = M.cpu().numpy(); log('sample means done')
    N = len(y); Lg = np.zeros((NBLK, len(GNAMES), 2, N, 10), np.float16); rng = np.random.default_rng(SEED + 1)
    idx = {g: torch.tensor(GROUPS[g], device=lg.dev) for g in GNAMES}
    with torch.no_grad():
        for s in range(0, N, BS):
            sl = np.arange(s, min(s + BS, N)); e = lg.embed(Jb, sl); _, _, mask, outs = lg.upto(e, NBLK - 1, cache=True)
            src = donor_index(e, rng)
            for k in range(NBLK):
                X = flat(*outs[k]); Rm = M[k][e['tt']]; Rr = X[src]
                for gi, g in enumerate(GNAMES):
                    for mi, R in enumerate((Rm, Rr)):
                        X2 = X.clone(); X2[:, idx[g]] = R[:, idx[g]]
                        Lg[k, gi, mi, sl] = lg.rest(e, mask, k, *unflat(X2)).float().cpu().numpy()
            log(f'grades batch {s // BS + 1}/{(N + BS - 1) // BS}')
    npz['L_grades'] = Lg
    res = {}
    for k in range(NBLK):
        for gi, g in enumerate(GNAMES):
            for mi, mode in enumerate(('mean', 'resample')):
                L1 = Lg[k, gi, mi].astype(np.float32); r = compare(L0, L1, y); r['confusion'] = confusion(y, L1.argmax(1)).tolist()
                r['per_class_recall'] = {c: float((L1.argmax(1)[y == i] == i).mean()) for i, c in enumerate(CL)}
                res[f'b{k}_{g}_{mode}'] = r
        log(f'block {k}: ' + ' '.join(f"{g} {100 * res[f'b{k}_{g}_mean']['acc']:.1f}/{100 * res[f'b{k}_{g}_resample']['acc']:.1f}" for g in GNAMES))
    out['grades'] = res


def sec_spurion(lg, Jb, y, L0, out, npz):
    from interp.p5_lgatr_launch import rotate_lab, boost_z
    from interp.part_core import kin, build, jmass
    res = {}; m0 = jmass(Jb)
    def rec(name, L1, Je=None, note=''):
        r = compare(L0, L1, y)
        if Je is not None: r['dlogm_mean'] = float(np.mean(np.log(np.maximum(jmass(Je), 1e-3)) - np.log(np.maximum(m0, 1e-3))))
        r['note'] = note; res[name] = r; npz[f'Lsp_{name}'] = L1.astype(np.float16)
        log(f"{name:22s} calls {100 * r['calls_changed'][0]:.3f}% [{100 * r['calls_changed'][1]:.3f}, {100 * r['calls_changed'][2]:.3f}] "
            f"acc {100 * r['acc']:.2f} max|dL| {r['max_abs_dlogit']:.2e} W->Z {r['W_to_Z'][0]:.3f}->{r['W_to_Z'][1]:.3f}")
        return L1
    rec('identity_heldpath', lg.logits(Jb, Jb, 0.0), Jb, 'features from the same jet: plumbing check')
    for a in (0.06, 0.3): rec(f'rot_phi_{a}', lg.logits(rotate_lab(Jb, a)), rotate_lab(Jb, a), 'rigid azimuthal rotation of every particle')
    pt, de, dp, ms = kin(Jb)
    rec('p1_deta+0.1', lg.logits(build(Jb, pt, de + 0.1 * (pt > 0), dp, ms)), build(Jb, pt, de + 0.1 * (pt > 0), dp, ms), 'pseudorapidity +0.1 at fixed pT, mass')
    rec('p1_dphi+0.06', lg.logits(build(Jb, pt, de, dp + 0.06 * (pt > 0), ms)), build(Jb, pt, de, dp + 0.06 * (pt > 0), ms), 'phi +0.06 via build')
    eq = {}
    for yv in (0.1, 0.3, 0.5):
        Jy = boost_z(Jb, yv)
        mp0, mk = pair_masses(Jb); mp1, _ = pair_masses(Jy); big = mk & (mp0 > 1.0)
        res[f'boost{yv}_mass_check'] = dict(jet_mass_max_rel=float(np.max(np.abs(jmass(Jy) - m0) / np.maximum(m0, 1.0))),
                                            pair_mass_max_rel=float(np.max(np.abs(mp1 - mp0)[big] / mp0[big])))
        L = {}
        L['A'] = rec(f'boost{yv}_A', lg.logits(Jy), Jy, 'mv boosted, spurions fixed, features recomputed (physical edit)')
        L['B'] = rec(f'boost{yv}_B', lg.logits(Jy, None, yv), Jy, 'mv + time spurion boosted, features recomputed')
        L['C'] = rec(f'boost{yv}_C', lg.logits(Jy, Jb, yv), Jy, 'mv + time spurion boosted, features held: exact symmetry')
        L['D'] = rec(f'boost{yv}_D', lg.logits(Jy, Jb, 0.0), Jy, 'mv boosted, spurions fixed, features held: spurion route')
        L['E'] = rec(f'boost{yv}_E', lg.logits(Jb, Jb, -yv), Jb, 'spurion swap: time spurion boosted by -y, rest unchanged')
        L['F'] = rec(f'boost{yv}_F', lg.logits(Jb, Jy, 0.0), Jb, 'feature route: only the scalar features from the boosted jet')
        eq[str(yv)] = dict(C_vs_base=float(np.abs(L['C'] - L0).max()), B_vs_F=float(np.abs(L['B'] - L['F']).max()),
                           D_vs_E=float(np.abs(L['D'] - L['E']).max()))
        log(f'boost {yv}: equivariance checks max|dL| C-base {eq[str(yv)]["C_vs_base"]:.2e} B-F {eq[str(yv)]["B_vs_F"]:.2e} D-E {eq[str(yv)]["D_vs_E"]:.2e}; '
            f'mass check {res[f"boost{yv}_mass_check"]}')
    res['equivariance_checks'] = eq; out['spurion'] = res


def probe_features(lg, J_mv, J_feat=None, spur_y=0.0):
    """(12, N, 96): per block, the global token's 48 invariant channels and the mean of the particle tokens' 48"""
    import torch
    N = len(J_mv['label']); F = np.zeros((NBLK, N, 96), np.float32); inv = torch.tensor(INV, device=lg.dev)
    with torch.no_grad():
        for s in range(0, N, BS):
            sl = np.arange(s, min(s + BS, N)); e = lg.embed(J_mv, sl, J_feat, spur_y); _, _, _, outs = lg.upto(e, NBLK - 1, cache=True)
            g = e['tt'] == 0; p = e['tt'] == 3; bp = e['batch'][p]
            for k, (hm, hs) in enumerate(outs):
                U = flat(hm, hs)[:, inv]
                sm = torch.zeros(e['nb'], 48, device=lg.dev).index_add_(0, bp, U[p]); sm /= (e['cnt'] - 3).clamp(min=1)[:, None].float()
                F[k, sl] = torch.cat([U[g], sm], 1).cpu().numpy()
    return F


def ridge_fit(X, t, rng):
    mu, sd = X.mean(0), X.std(0) + 1e-6; Z = (X - mu) / sd; n = len(t); o = rng.permutation(n); tr, va = o[: int(0.8 * n)], o[int(0.8 * n):]
    def fit(Zt, tt, a):
        A = np.c_[Zt, np.ones(len(Zt))]; R = a * np.eye(A.shape[1]); R[-1, -1] = 0; return np.linalg.solve(A.T @ A + R, A.T @ tt)
    best = None
    for a in (1e-2, 1e-1, 1, 10, 100, 1000):
        w = fit(Z[tr], t[tr], a); r = np.mean((np.c_[Z[va], np.ones(len(va))] @ w - t[va]) ** 2)
        if best is None or r < best[0]: best = (r, a)
    w = fit(Z, t, best[1]); return lambda Xn: np.c_[(Xn - mu) / sd, np.ones(len(Xn))] @ w, best[1]


def probe_analysis(Fs, logm, edits_dlogm, rng):
    """Fs: {edit: (12, N, 96)}, 'base' and 'dil' required. ridge per block on the first half, test on the second"""
    N = len(logm); h = N // 2; tr, te = np.arange(h), np.arange(h, N); res = {}
    for k in range(NBLK):
        f, a = ridge_fit(Fs['base'][k][tr], logm[tr], rng); pb = f(Fs['base'][k][te])
        r2 = 1 - np.mean((pb - logm[te]) ** 2) / np.var(logm[te]); d = {}
        dprobe = {e: f(Fs[e][k][te]) - pb for e in Fs if e != 'base'}
        dil_true = edits_dlogm['dil'][te]
        for e, dp in dprobe.items():
            bs_g, bs_r = [], []
            for _ in range(NB):
                i = rng.integers(0, len(te), len(te)); bs_g.append(dp[i].mean() / dil_true[i].mean()); bs_r.append(dp[i].mean() / dprobe['dil'][i].mean())
            d[e] = dict(dprobe_mean=float(dp.mean()), dlogm_true_mean=float(edits_dlogm[e][te].mean()),
                        gain_vs_true_dil=[float(dp.mean() / dil_true.mean()), float(np.percentile(bs_g, 2.5)), float(np.percentile(bs_g, 97.5))],
                        ratio_to_probe_dil=[float(dp.mean() / dprobe['dil'].mean()), float(np.percentile(bs_r, 2.5)), float(np.percentile(bs_r, 97.5))])
        res[f'b{k}'] = dict(R2_test=float(r2), alpha=a, edits=d)
    return res


def sec_probe(lg, Jb, y, out, npz, name='probe'):
    from interp.p5_lgatr_launch import boost_z
    from interp.part_core import kin, build, jmass
    pt, de, dp, ms = kin(Jb); Jd = build(Jb, pt, de * 1.1, dp * 1.1, ms); Jp = build(Jb, pt, de, dp + 0.06 * (pt > 0), ms); Jy = boost_z(Jb, 0.3)
    lm = lambda J: np.log(np.maximum(jmass(J), 1.0))
    ok = jmass(Jb) > 1.0; logm = lm(Jb)
    Fs = {'base': probe_features(lg, Jb), 'dil': probe_features(lg, Jd), 'phi0.06': probe_features(lg, Jp),
          'boost0.3_A': probe_features(lg, Jy), 'boost0.3_D': probe_features(lg, Jy, Jb, 0.0)}
    dl = {'dil': lm(Jd) - logm, 'phi0.06': lm(Jp) - logm, 'boost0.3_A': lm(Jy) - logm, 'boost0.3_D': lm(Jy) - logm}
    w = np.where(ok)[0]
    res = probe_analysis({k: v[:, w] for k, v in Fs.items()}, logm[w], {k: v[w] for k, v in dl.items()}, np.random.default_rng(SEED + 2))
    res['N_fit'] = int(len(w) // 2); res['N_test'] = int(len(w) - len(w) // 2); res['dlogm_dil_mean'] = float(dl['dil'][w].mean())
    out[name] = res
    log(f'{name}: R2 per block ' + ' '.join(f"{res[f'b{k}']['R2_test']:.3f}" for k in range(NBLK)))
    log(f'{name}: dil gain per block ' + ' '.join(f"{res[f'b{k}']['edits']['dil']['gain_vs_true_dil'][0]:.2f}" for k in range(NBLK)))
    log(f'{name}: boost0.3_A / probe dil per block ' + ' '.join(f"{res[f'b{k}']['edits']['boost0.3_A']['ratio_to_probe_dil'][0]:+.3f}" for k in range(NBLK)))


def stage_main(lg, J, Jb, y, rows):
    out = dict(node='P44', ckpt=lg.ckpt, sample=f'val_5M <class>_{FIDX}.root, {PER} random rows per class, default_rng({SEED}), shuffled',
               N=int(len(y))); npz = dict(rows=rows, y=y)
    secs = ARGS.sections.split(','); L0 = None
    for sname in ['base'] + [s for s in secs if s != 'base']:
        try:
            if sname == 'base': L0 = sec_base(lg, J, Jb, y, out, npz)
            elif sname == 'grades': sec_grades(lg, Jb, y, L0, out, npz)
            elif sname == 'spurion': sec_spurion(lg, Jb, y, L0, out, npz)
            elif sname == 'probe': sec_probe(lg, Jb, y, out, npz)
        except Exception:
            log(f'SECTION {sname} FAILED'); traceback.print_exc(); out.setdefault('failed', []).append(sname)
        jdump(out, RES / f'p44_main{TAG}.json')
    np.savez_compressed(RES / f'p44_main{TAG}.npz', **npz); log('wrote npz')


# ------------------------------------------------------------------------------------------------ stage ctl
def stage_ctl(lg, J, Jb, y, rows):
    import torch
    M = sample_means(lg, Jb); Js = sub_(Jb, np.arange(NC)); ys = y[:NC]
    rng = np.random.default_rng(SEED + 3)
    sets = {(k, c, d): torch.tensor(np.sort(rng.choice(288, c, replace=False)), device=lg.dev) for k in range(NBLK) for c in COUNTS for d in range(NDRAW)}
    P = np.zeros((NBLK, len(COUNTS), NDRAW, NC), np.int8); P0 = np.zeros(NC, np.int8)
    with torch.no_grad():
        for s in range(0, NC, BS):
            sl = np.arange(s, min(s + BS, NC)); e = lg.embed(Js, sl); hm, hs, mask, outs = lg.upto(e, NBLK - 1, cache=True)
            P0[sl] = lg.rest(e, mask, NBLK - 1, hm, hs).argmax(1).cpu().numpy()
            for k in range(NBLK):
                X = flat(*outs[k]); Rm = M[k][e['tt']]
                for ci, c in enumerate(COUNTS):
                    for d in range(NDRAW):
                        ix = sets[(k, c, d)]; X2 = X.clone(); X2[:, ix] = Rm[:, ix]
                        P[k, ci, d, sl] = lg.rest(e, mask, k, *unflat(X2)).argmax(1).cpu().numpy()
                log(f'ctl batch {s // BS + 1}, block {k}: acc per count ' + ' '.join(f'{c}:{100 * (P[k, ci][:, sl] == ys[sl][None]).mean():.1f}' for ci, c in enumerate(COUNTS)))
    np.savez_compressed(RES / f'p44_ctl{TAG}.npz', P=P, P0=P0, y=ys, rows=rows[:NC], counts=np.array(COUNTS), ckpt=np.array(lg.ckpt))
    log('wrote ctl npz; unedited acc on subset', float((P0 == ys).mean()))


# ------------------------------------------------------------------------------------------------ stage das
def das_caches(lg, Jb, Js, k, bs):
    import torch
    C = []; N = len(Jb['label'])
    with torch.no_grad():
        for s in range(0, N, bs):
            sl = np.arange(s, min(s + bs, N)); eb = lg.embed(Jb, sl); es = lg.embed(Js, sl)
            assert torch.equal(eb['batch'], es['batch'])
            hb, sb, mask, _ = lg.upto(eb, k); hs_, ss_, _, _ = lg.upto(es, k)
            Lb = lg.rest(eb, mask, k, hb, sb); Ls = lg.rest(es, mask, k, hs_, ss_)
            C.append(dict(e=eb, mask=mask, Xb=flat(hb, sb), Xs=flat(hs_, ss_), Lb=Lb, Ls=Ls))
    return C


def das_apply(X_b, X_s, u, inv):
    import torch
    Ub, Us = X_b[:, inv], X_s[:, inv]; X2 = X_b.clone()
    X2[:, inv] = Ub + ((Us - Ub) @ u)[:, None] * u[None] if u is not None else Us
    return X2


def das_eval(lg, C, k, u, inv):
    import torch
    hit = tot = 0; rec = []
    with torch.no_grad():
        for c in C:
            Li = lg.rest(c['e'], c['mask'], k, *unflat(das_apply(c['Xb'], c['Xs'], u, inv)))
            pb, ps, pi = c['Lb'].argmax(1), c['Ls'].argmax(1), Li.argmax(1); fl = pb != ps
            hit += int((pi[fl] == ps[fl]).sum()); tot += int(fl.sum())
            den = ((c['Ls'] - c['Lb']) ** 2).sum(1); num = ((c['Ls'] - Li) ** 2).sum(1); ok = den > 1e-4
            rec.append((1 - num[ok] / den[ok]).cpu().numpy())
    rec = np.concatenate(rec)
    return dict(IIA=hit / max(tot, 1), n_flipped=tot, hits=hit, logit_recovery_median=float(np.median(rec)))


def das_model(lg, Jtr_b, Jtr_s, Jte_b, Jte_s, label):
    import torch
    inv = torch.tensor(INV, device=lg.dev); res = {}; g = torch.Generator(device='cpu').manual_seed(SEED)
    for k in DAS_BLOCKS:
        Ctr = das_caches(lg, Jtr_b, Jtr_s, k, 128); Cte = das_caches(lg, Jte_b, Jte_s, k, BS)
        v = torch.randn(48, generator=g).to(lg.dev).requires_grad_(True); opt = torch.optim.Adam([v], lr=0.02); losses = []
        for ep in range(DAS_EPOCHS):
            for c in Ctr:
                u = v / v.norm(); Li = lg.rest(c['e'], c['mask'], k, *unflat(das_apply(c['Xb'], c['Xs'], u, inv)))
                loss = torch.nn.functional.kl_div(torch.log_softmax(Li, 1), torch.softmax(c['Ls'], 1), reduction='batchmean')
                opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss))
        u = (v / v.norm()).detach()
        r = dict(learned=das_eval(lg, Cte, k, u, inv), full_invariant_swap=das_eval(lg, Cte, k, None, inv),
                 loss_first_last=[float(np.mean(losses[:5])), float(np.mean(losses[-5:]))])
        rnd = [das_eval(lg, Cte, k, (lambda w: w / w.norm())(torch.randn(48, generator=g).to(lg.dev)), inv)['IIA'] for _ in range(NRAND)]
        r['random_IIA'] = dict(mean=float(np.mean(rnd)), max=float(np.max(rnd)), all=rnd)
        r['u_weight_on_g0'] = float((u[torch.tensor([i for i, c in enumerate(INV) if c < 256], device=lg.dev)] ** 2).sum())
        res[f'b{k}'] = r
        log(f"{label} DAS block {k}: IIA learned {r['learned']['IIA']:.3f} (n {r['learned']['n_flipped']}), random {r['random_IIA']['mean']:.3f} "
            f"(max {r['random_IIA']['max']:.3f}), all-48 swap {r['full_invariant_swap']['IIA']:.3f}; loss {r['loss_first_last']}; |u_g0|^2 {r['u_weight_on_g0']:.2f}")
        del Ctr, Cte; torch.cuda.empty_cache()
    return res


def stage_das(lg, J, Jb, y, rows, lg_rand):
    from interp.part_core import kin, build
    pt, de, dp, ms = kin(Jb); f = np.where(np.random.default_rng(SEED + 4).random(len(y)) < 0.5, 1.1, 1 / 1.1)[:, None]
    Js = build(Jb, pt, de * f, dp * f, ms)
    tr, te = np.arange(NDAS), np.arange(NDAS, 2 * NDAS)
    out = dict(node='P44', ckpt=lg.ckpt, N_train=NDAS, N_test=NDAS, blocks=DAS_BLOCKS, epochs=DAS_EPOCHS, factors='1.1 or 1/1.1 per jet')
    for name, m in (('trained', lg), ('untrained', lg_rand)):
        try:
            out[name] = das_model(m, sub_(Jb, tr), sub_(Js, tr), sub_(Jb, te), sub_(Js, te), name)
        except Exception:
            log(f'DAS {name} FAILED'); traceback.print_exc(); out.setdefault('failed', []).append(name)
        jdump(out, RES / f'p44_das{TAG}.json')
    try:
        L = lg_rand.logits(Jb); out['untrained_acc'] = float((L.argmax(1) == y).mean()); log('untrained acc', out['untrained_acc'])
        sec_probe(lg_rand, Jb, y, out, {}, name='probe_untrained')
    except Exception:
        log('untrained probe FAILED'); traceback.print_exc(); out.setdefault('failed', []).append('probe_untrained')
    jdump(out, RES / f'p44_das{TAG}.json')


# ------------------------------------------------------------------------------------------------ stage analyze
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.zeros(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj


def stage_analyze():
    main = json.loads((RES / f'p44_main{TAG}.json').read_text()); Z = np.load(RES / f'p44_main{TAG}.npz')
    C = np.load(RES / f'p44_ctl{TAG}.npz'); das = json.loads((RES / f'p44_das{TAG}.json').read_text())
    y = Z['y']; L0 = Z['L0']; Lg = Z['L_grades']; P = C['P']; nc = len(C['y'])
    assert np.array_equal(C['rows'], Z['rows'][:nc]) and np.array_equal(C['y'], y[:nc]), 'control subset rows differ'
    p0 = L0[:nc].argmax(1); ys = y[:nc]; c0 = (p0 == ys)
    log('unedited acc on control subset main / ctl', c0.mean(), (C['P0'] == ys).mean())
    rng = np.random.default_rng(SEED + 5); tests = []; A = {}
    for k in range(NBLK):
        for gi, g in enumerate(GNAMES):
            ci = list(C['counts']).index(len(GROUPS[g]))
            cg = (Lg[k, gi, 0, :nc].astype(np.float32).argmax(1) == ys).astype(float)
            cc = (P[k, ci] == ys[None]).astype(float)                        # (draws, nc)
            d = cc.mean(0) - cg                                              # per-jet: control accuracy minus grade accuracy
            bs = [d[rng.integers(0, nc, nc)].mean() for _ in range(NB)]
            t = abs(d.mean()); sg = rng.choice([-1.0, 1.0], size=(2000, nc)); pperm = float((1 + (np.abs(sg @ d / nc) >= t).sum()) / 2001)
            acc_draws = cc.mean(1)
            from scipy.stats import norm
            se = d.std(ddof=1) / np.sqrt(nc); pz = float(2 * norm.sf(abs(d.mean()) / se)) if se > 0 else 1.0
            A[f'b{k}_{g}'] = dict(acc_grade_subset=float(cg.mean()), acc_ctrl_mean=float(acc_draws.mean()),
                                  acc_ctrl_2p5_97p5=[float(np.percentile(acc_draws, 2.5)), float(np.percentile(acc_draws, 97.5))],
                                  extra_drop_vs_ctrl=[float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                  rank_p_drop=float((1 + (acc_draws <= cg.mean()).sum()) / (1 + len(acc_draws))), p_signflip=pperm, p_z=pz,
                                  n_components=len(GROUPS[g]))
            tests.append(f'b{k}_{g}')
    ph = holm([A[t]['p_signflip'] for t in tests])
    for t, p in zip(tests, ph): A[t]['p_holm'] = float(p)
    for t, p in zip(tests, holm([A[t]['p_z'] for t in tests])): A[t]['p_z_holm'] = float(p)
    from scipy.stats import binomtest
    for m in ('trained', 'untrained'):
        for k, r in das.get(m, {}).items():
            for key in ('learned', 'full_invariant_swap'):
                h, n = r[key]['hits'], r[key]['n_flipped']
                if n: ci_ = binomtest(h, n).proportion_ci(0.95); r[key]['IIA_ci95'] = [float(ci_.low), float(ci_.high)]
                rm = r['random_IIA']['mean']; r[key]['p_vs_random_mean'] = float(binomtest(h, n, max(rm, 1e-3), alternative='greater').pvalue) if n else 1.0
    ck_ctl = str(C['ckpt']); log('checkpoints main / ctl / das:', main.get('ckpt'), ck_ctl, das.get('ckpt'))
    summ = dict(node='P44', ckpt=main.get('ckpt'), ckpt_ctl=ck_ctl, ckpt_das=das.get('ckpt'), base=main.get('base'), grades_vs_control=A, n_tests_holm=len(tests),
                grades_full=main.get('grades'), spurion=main.get('spurion'), probe=main.get('probe'), das=das)
    jdump(summ, RES / f'p44_lgatr_grades{TAG}.json')
    try:
        figure(main, A, das)
    except Exception:
        traceback.print_exc()


def figure(main, A, das):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw=dict(width_ratios=[1.35, 1, 1]))
    rows = ['s', 'g0', 'g1', 'g2', 'g3', 'g4', 'inv', 'cov']
    lab = {'s': 'aux. scalars s (32)', 'g0': 'grade 0 scalar (16)', 'g1': 'grade 1 vector (64)', 'g2': 'grade 2 bivector (96)',
           'g3': 'grade 3 axial vec. (64)', 'g4': 'grade 4 pseudoscalar (16)', 'inv': 'invariant: g0 + s (48)', 'cov': 'covariant: g1-g4 (240)'}
    G = main['grades']; acc0 = main['base']['acc']
    H = np.array([[100 * G[f'b{k}_{g}_mean']['acc'] for k in range(NBLK)] for g in rows])
    Cm = np.array([[100 * A[f'b{k}_{g}']['acc_ctrl_mean'] for k in range(NBLK)] for g in rows])
    im = ax[0].imshow(H, cmap='viridis', vmin=10, vmax=100 * acc0, aspect='auto')
    for i in range(len(rows)):
        for k in range(NBLK):
            ax[0].text(k, i, f'{H[i, k]:.0f}\n({Cm[i, k]:.0f})', ha='center', va='center', fontsize=6.5, color='w' if H[i, k] < 60 else 'k')
    ax[0].set_yticks(range(len(rows))); ax[0].set_yticklabels([lab[r] for r in rows], fontsize=8); ax[0].set_xticks(range(NBLK))
    ax[0].set_xlabel('block whose output is mean-ablated'); plt.colorbar(im, ax=ax[0], fraction=0.03, label='10-class accuracy [%]')
    ax[0].set_title(f'(a) accuracy after mean ablation of one channel group\n(random-component control of equal size in brackets; unedited {100 * acc0:.1f} %)', fontsize=9)
    S = main['spurion']; ys = [0.1, 0.3, 0.5]
    rl = {'A': 'A: full boost (physical edit)', 'D': 'D: spurion route only', 'E': 'E: spurion swap', 'F': 'F: scalar-feature route only',
          'B': 'B: particles + spurion, features recomputed', 'C': 'C: particles + spurion, features held'}
    for r, mk in zip('ADEFBC', 'os^vDx'):
        v = [100 * S[f'boost{yv}_{r}']['calls_changed'][0] for yv in ys]
        lo = [100 * S[f'boost{yv}_{r}']['calls_changed'][1] for yv in ys]; hi = [100 * S[f'boost{yv}_{r}']['calls_changed'][2] for yv in ys]
        ax[1].errorbar(ys, v, yerr=[np.subtract(v, lo), np.subtract(hi, v)], marker=mk, capsize=2, label=rl[r])
    ax[1].axhline(100 * S['rot_phi_0.3']['calls_changed'][0], color='grey', ls=':', label='rigid phi rotation 0.3 rad')
    ax[1].set_yscale('log'); ax[1].set_ylim(0.02, 15); ax[1].set_xlabel('longitudinal boost: rapidity shift y of every particle')
    ax[1].set_ylabel('calls changed [% of 10,000 jets]'); ax[1].legend(fontsize=7); ax[1].set_title('(b) which input route carries the boost response\n(jet mass and all pair masses unchanged)', fontsize=9)
    Pr = main['probe']; ks = range(NBLK)
    ax[2].plot(ks, [Pr[f'b{k}']['R2_test'] for k in ks], 'k-o', label='$R^2$ of log $m$ probe (test jets)')
    ax[2].plot(ks, [Pr[f'b{k}']['edits']['dil']['gain_vs_true_dil'][0] for k in ks], 'C0-s', label='probe change / true change, angles x 1.1')
    for e, cc, nm in (('phi0.06', 'C1', 'rigid phi + 0.06'), ('boost0.3_A', 'C2', 'boost y 0.3'), ('boost0.3_D', 'C3', 'boost y 0.3, spurion route')):
        ax[2].plot(ks, [Pr[f'b{k}']['edits'][e]['ratio_to_probe_dil'][0] for k in ks], '-^', color=cc, label=f'{nm}: probe change / dilation probe change')
    if 'probe_untrained' in das:
        ax[2].plot(ks, [das['probe_untrained'][f'b{k}']['R2_test'] for k in ks], 'k:', label='$R^2$, untrained L-GATr')
    ax[2].axhline(0, color='grey', lw=0.5); ax[2].set_xlabel('block'); ax[2].legend(fontsize=7)
    ax[2].set_title('(c) jet-mass read-out from the invariant channels\n(m fixed under the phi shift and the boost)', fontsize=9)
    fig.tight_layout(); p = RES / f'p44_lgatr_grades{TAG}.png'; fig.savefig(p, dpi=130); log('wrote', p)


# ------------------------------------------------------------------------------------------------ main
if __name__ == '__main__':
    stages = ARGS.stage.split(',')
    if stages == ['analyze']:
        stage_analyze(); sys.exit(0)
    import torch
    from interp.part_core import kin, build
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; ck = latest_ckpt(); log('checkpoint', ck, 'device', dev)
    lg = LG(ck, dev); lg.ckpt = ck; log('loaded L-GATr; force_xformers', lg.model.force_xformers)
    J, rows = read_sample(); y = J['label'].astype(int); Jb = build(J, *kin(J)); log('sample', len(y), 'jets')
    RES.mkdir(parents=True, exist_ok=True)
    for st in stages:
        if st == 'main': stage_main(lg, J, Jb, y, rows)
        elif st == 'ctl': stage_ctl(lg, J, Jb, y, rows)
        elif st == 'das':
            lr = LG(None, dev); lr.ckpt = 'random init (torch seed 0)'; log('loaded untrained L-GATr')
            stage_das(lg, J, Jb, y, rows, lr)
        elif st == 'analyze': stage_analyze()
    log('done')
