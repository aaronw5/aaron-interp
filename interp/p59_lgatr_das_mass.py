"""P59 (loop C, round C13, exploratory; claims C22, C25, C6 v2): distributed alignment search (DAS) for the jet mass inside L-GATr.

Question. Is there a low-dimensional subspace of L-GATr's hidden state that, when copied from a source jet s into a base jet b,
changes the network's output on b the way a real change of b's invariant mass to m_s does; in which block and in which grade of
the equivariant representation does it appear; and does the stored variable follow the invariant mass or the spread of the
particles about the jet's own momentum axis?

Terms.
 Residual stream: after block k (k = 0..11) every token carries 16 multivector channels x_c (16 components each: 0 scalar = grade 0;
   1-4 four-vector t, x, y, z = grade 1; 5-10 bivector = grade 2; 11-14 axial vector = grade 3; 15 pseudoscalar = grade 4) and 32
   auxiliary scalars. Tokens of a jet: 0 = global token (the classifier reads its output: logits = scalar parts of the 10 output
   channels of the final EquiLinear applied to this token; mean_aggregation is off, so this token IS the read-out aggregation),
   1-2 = beam / time reference tokens, 3.. = particles in pT order.
 Grade-preserving subspace: an orthonormal n x d matrix U acting on the channel index of one grade group (n = 16 for each
   multivector grade, 32 for the auxiliary scalars); for grade 1 the same U acts on t, x, y, z, so the map commutes with Lorentz
   transformations (equivalently: rotate the channels within the grade by an orthogonal map and take its first d coordinates).
 Interchange intervention at block k: h_b <- h_b + U U^T (h_s - h_b) on the chosen tokens, then blocks k+1..11 run on the edited
   state. Tokens: 'aligned' = global, the two reference tokens and particle ranks 0..min(n_b, n_s)-1 (pT rank alignment);
   'global' = the global token only.
 Variants: g0 = U on the grade-0 (scalar) parts of the 16 channels; g1 = U on the grade-1 four-vectors; aux = U on the 32 auxiliary
   scalars; all = one d-dim U in every one of the six groups (grades 0-4 and aux) at once (6 d channels in all).
 DAS: U = QR(A) with A trained (Adam) on fit pairs to minimise KL(softmax(l_cf) || softmax(l_patched)).
 Counterfactual (cf): b with its particles' (deta, dphi) about the stored axis scaled by one factor lambda (fixed pT, particle
   mass) chosen by bisection so that m(b_lambda) = m_s exactly ('real dilation'). The cf output = what the network does when only
   b's mass changes to m_s.
 IIA (interchange intervention accuracy): fraction of pairs whose patched 10-class call (argmax) equals the cf call. IIA_ch: the
   same on the pairs where the cf call differs from b's own call (the informative pairs). F (recovered fraction): least-squares
   slope through 0 of the patched change on the counterfactual change of d = l_W - l_Z (logit difference),
   F = sum (d_p - d_b)(d_cf - d_b) / sum (d_cf - d_b)^2 (a plain ratio of sums is unstable: mass increases and decreases cancel).
 p-values against the random-subspace null: empirical (1 + #null >= DAS) / (1 + draws) and a Gaussian tail p from the null mean
   and s.d. (z = (DAS - mean) / sd); Holm is applied to the Gaussian p (100 draws cannot resolve p below 0.01).
Controls: (i) 100 random grade-preserving subspaces of the same d (Haar) on 200 held-out pairs (30 for global-token patching, d 1 and 4);
 (ii) DAS on untrained L-GATr (authors' config, torch seed 5959, no weights); (iii) decoy variable: donors with pT different by
 5-30 % (log) at mass matched within 5 %, target = output on b after a real pT rescale (pT x pT_s/pT_b, angles rescaled so that m
 stays m_b); (iv) shuffled pairs (grade-0 and all-grade variants): target = the cf output of a random other pair.
 DAS fit: 40 Adam steps (lr 0.08) on batches of 100 fit pairs. Global-token patching at d = 1 and 4 only.
Edits (on held-out W and Z jets, projections on the d = 1 subspace): M: every particle mass raised at fixed three-momentum,
 m_i' = sqrt(m_i^2 + (kappa |p_i|)^2), kappa by bisection so that m' = 1.05 m (angles and the spread about the momentum axis
 unchanged); DM: dilation with m' = 1.05 m (the same mass change made with angles); S: reference R0 = particle masses raised
 (kappa0) so that m = 1.10 m_b; S = R0's particles with angles x 1.05 and particle masses lowered (kappa1 < kappa0) so that m
 equals m(R0) (spread up 5 %, m fixed); DS = R0 with angles x 1.05 at R0's particle masses (spread up 5 % and m up).
 ratio_M = sum dproj(M) / sum dproj(DM): 1 if the coordinate follows m, 0 if it follows the spread.
 ratio_S = sum dproj(S - R0) / sum dproj(DS - R0): 0 if it follows m, 1 if it follows the spread.
Sample: JetClass test_20M W (WToQQ), Z (ZToQQ), QCD (ZJetsToNuNu) files _112 (fit) and _104 (evaluate): the planned _126/_127 do
 not exist (test_20M has _100-_119); _112 and _104 were used only by ParT nodes (P22, P8-era). 5,000 random rows per class per
 split (numpy default_rng(5959), never first-k). Pairs: base and source from the same split, |pT_s/pT_b - 1| <= 0.05,
 0.05 <= |log m_s - log m_b| <= 0.3, m_b, m_s >= 10 GeV; one random eligible source per base; 1,000 bases per class per split.
 Random generators: rows of the fit files default_rng(5959), of the evaluation files default_rng(5960); pairs, edits default_rng(5961).
 Reference: unedited accuracy on the P1 sample (rows 3,000-3,300 of each _100 test file, N 3,000; expected 83.73 %).

  ./sync.sh && ./part_job.sh c-p59-b01 '/j-jepa-vol/aaron-interp/env-lgatr/bin/python -m interp.p59_lgatr_das_mass --blocks 0,1 --tag b01'
  ... --merge (CPU) combines the per-job files, applies Holm, writes p59_lgatr_das_mass.json and the figure.
-> $AI_RESULTS/lgatr/p59_<tag>[_smoke].json, p59_U_<tag>.npz; merged p59_lgatr_das_mass.json, p59_lgatr_das_mass.png
"""
import argparse, json, os, sys, time, traceback
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('JETDISTILL_TAGGER', 'part')
ap = argparse.ArgumentParser()
ap.add_argument('--blocks', default='0'); ap.add_argument('--tag', default='b0'); ap.add_argument('--smoke', action='store_true')
ap.add_argument('--merge', action='store_true'); ap.add_argument('--ctl', action='store_true'); ap.add_argument('--tags', default='b01,b23,b456,b7to11')
ARGS = ap.parse_args()
SEED, FFIT, FEVAL = 5959, 112, 104
TEST = Path('/j-jepa-vol/JetClass/Pythia/test_20M')
CKPT = '/j-jepa-vol/aaron-interp/models/lgatr/runs/jc2m/jc2m_s0a/models/model_run0.pt'
PRE = {'QCD': 'ZJetsToNuNu', 'Zqq': 'ZToQQ', 'Wqq': 'WToQQ'}
CL = ['QCD', 'Hbb', 'Hcc', 'Hgg', 'H4q', 'Hqql', 'Zqq', 'Wqq', 'Tbqq', 'Tbl']
C3 = ['Wqq', 'Zqq', 'QCD']
iW, iZ = CL.index('Wqq'), CL.index('Zqq')
NBLK, NC, NS = 12, 16, 32
GR = {'g0': slice(0, 1), 'g1': slice(1, 5), 'g2': slice(5, 11), 'g3': slice(11, 15), 'g4': slice(15, 16)}
VARIANTS = {'g0': ['g0'], 'g1': ['g1'], 'aux': ['aux'], 'all': ['g0', 'g1', 'g2', 'g3', 'g4', 'aux']}
DS_ = (1, 2, 4)
RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'lgatr'
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def jdump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(obj, indent=1, default=float)); log('wrote', path)


def params(smoke):
    return dict(smoke=smoke, PER=5000 if not smoke else 300, NPAIR=1000 if not smoke else 40, NED=1000 if not smoke else 30,
                BSF=100 if not smoke else 40, BSE=200 if not smoke else 60, STEPS=40 if not smoke else 4, LR=0.08,
                NNULL=200 if not smoke else 60, NDRAW=100 if not smoke else 4, NDRAWG=30 if not smoke else 2, NB=1000 if not smoke else 50,
                P1=300 if not smoke else 20)


# ------------------------------------------------------------------------------------------------ data
def read_split(fidx, per, rng):
    import uproot
    from interp.part_core import read_root, sub, cat
    out = []
    for c in C3:
        path = TEST / f'{PRE[c]}_{fidx}.root'; N = uproot.open(path)['tree'].num_entries
        rows = np.sort(rng.choice(N, per, replace=False)); ch = []
        for s in range(0, N, 25000):
            r = rows[(rows >= s) & (rows < s + 25000)]
            if len(r): ch.append(sub(read_root(path, s, min(s + 25000, N)), r - s))
        J = cat(ch); assert (J['label'] == CL.index(c)).all(), c; out.append(J)
    return cat(out)


def load_split(fidx, per, seed):
    """read once and cache (npz under $AI_RESULTS/lgatr/cache); other jobs wait for the cache (lock file) instead of re-reading"""
    cdir = RES / 'cache'; cdir.mkdir(parents=True, exist_ok=True); f = cdir / f'p59_{fidx}_{per}_{seed}.npz'; lk = cdir / (f.name + '.lock')
    if not f.exists():
        try:
            fd = os.open(lk, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd); mine = True
        except FileExistsError:
            mine = False
        if mine:
            J = read_split(fidx, per, np.random.default_rng(seed)); tmp = cdir / (f.name + '.tmp.npz'); np.savez(tmp, **J); os.replace(tmp, f)
            log('cached', f); return J
        t = time.time()
        while not f.exists():
            if time.time() - t > 5400: raise RuntimeError('cache wait timeout ' + str(f))
            time.sleep(20)
        time.sleep(5); log('waited for cache', f, f'{time.time() - t:.0f}s')
    z = np.load(f); return {k: z[k] for k in z.files}


def rebuild(J):
    from interp.part_core import kin, build
    pt, de, dp, ms = kin(J); return build(J, pt, de, dp, ms)


def bisect(fun, lo, hi, target, it=50):
    """vectorised bisection for an increasing per-jet function fun(x) -> value; returns x, final relative error"""
    lo = np.full(len(target), lo, float); hi = np.full(len(target), hi, float)
    for _ in range(it):
        mid = 0.5 * (lo + hi); v = fun(mid); up = v < target; lo = np.where(up, mid, lo); hi = np.where(up, hi, mid)
    x = 0.5 * (lo + hi); return x, np.abs(fun(x) / target - 1)


def dilate(J, lam, ptscale=None, kappa=None):
    """angles (deta, dphi about the stored axis) x lam per jet, optional pT x ptscale, optional particle masses
    m_i' = sqrt(m_i^2 + (kappa |p_i|)^2) with |p_i| taken after the angle change"""
    from interp.part_core import kin, build
    pt, de, dp, ms = kin(J); lam = np.asarray(lam, float)[:, None]
    pt2 = pt * (1.0 if ptscale is None else np.asarray(ptscale, float)[:, None])
    de2, dp2 = de * lam, dp * lam
    if kappa is not None:
        sgn = np.where(J['jet_eta'] < 0, -1.0, 1.0)[:, None]; eta = J['jet_eta'][:, None] + sgn * de2
        p = pt2 * np.cosh(eta); ms = np.sqrt(ms.astype(float) ** 2 + (np.asarray(kappa, float)[:, None] * p) ** 2) * (pt > 0)
    return build(J, pt2, de2, dp2, ms)


def own_spread(J):
    """|p|-weighted mean squared angle of the particles about the jet three-momentum direction"""
    m = J['mask'] > 0; P = np.stack([(J[k] * m).sum(1) for k in ('px', 'py', 'pz')], 1)
    p = np.stack([J[k] for k in ('px', 'py', 'pz')], -1).astype(float); pn = np.linalg.norm(p, axis=-1)
    c = np.clip((p * P[:, None]).sum(-1) / np.maximum(pn * np.linalg.norm(P, axis=1)[:, None], 1e-9), -1, 1)
    th = np.arccos(c) * m; return (pn * th ** 2 * m).sum(1) / np.maximum((pn * m).sum(1), 1e-9)


def make_pairs(m, pt, y, kind, nper, rng):
    """base rows (nper per class, random) and one random eligible source each (see docstring)"""
    lm, lp = np.log(np.maximum(m, 1e-3)), np.log(pt); ok_m = m >= 10; n = len(m)
    src = np.full(n, -1)
    for s0 in range(0, n, 1000):
        i = np.arange(s0, min(n, s0 + 1000))
        dpt = np.abs(lp[None, :] - lp[i, None]); dm = np.abs(lm[None, :] - lm[i, None])
        if kind == 'mass': E = (np.abs(pt[None, :] / pt[i, None] - 1) <= 0.05) & (dm >= 0.05) & (dm <= 0.3)
        else: E = (np.abs(m[None, :] / m[i, None] - 1) <= 0.05) & (dpt >= 0.05) & (dpt <= 0.3)
        E &= ok_m[None, :] & ok_m[i, None]; E[np.arange(len(i)), i] = False
        for a, r in enumerate(i):
            c = np.flatnonzero(E[a])
            if len(c): src[r] = rng.choice(c)
    base = np.concatenate([np.sort(rng.choice(np.flatnonzero((y == CL.index(c)) & (src >= 0)), nper, replace=False)) for c in C3])
    base = base[rng.permutation(len(base))]
    return base, src[base]


def pair_set(J, base, src, kind):
    """(Jb, Js, Jcf, info): Jcf = b dilated to m_s (mass) or pT-rescaled to pT_s at fixed m (decoy)"""
    from interp.part_core import sub, jmass
    Jb, Js = sub(J, base), sub(J, src); mb, ms_ = jmass(Jb).astype(float), jmass(Js).astype(float)
    if kind == 'mass':
        lam, err = bisect(lambda x: jmass(dilate(Jb, x)).astype(float), 0.4, 2.5, ms_); Jcf = dilate(Jb, lam); extra = dict(lam=lam)
    else:
        c = Js['jet_pt'].astype(float) / Jb['jet_pt'].astype(float)
        lam, err = bisect(lambda x: jmass(dilate(Jb, x, ptscale=c)).astype(float), 0.3, 3.0, mb); Jcf = dilate(Jb, lam, ptscale=c)
        extra = dict(lam=lam, ptscale=c)
    info = dict(N=int(len(base)), max_rel_mass_err=float(err.max()), median_abs_dlogm=float(np.median(np.abs(np.log(ms_ / mb)))),
                median_abs_dlogpt=float(np.median(np.abs(np.log(Js['jet_pt'] / Jb['jet_pt'])))),
                lam_range=[float(extra['lam'].min()), float(extra['lam'].max())],
                base_classes={c: int((Jb['label'] == CL.index(c)).sum()) for c in C3})
    return Jb, Js, Jcf, info


# ------------------------------------------------------------------------------------------------ network
class Stop(Exception):
    pass


class Net:
    def __init__(self, kind, dev):
        import torch
        from interp.p5_lgatr_launch import load_lgatr, LG
        sys.path.insert(0, LG)
        if kind == 'untrained': self.model, self.cfg = self._untrained(dev)
        else: self.model, self.cfg = load_lgatr(CKPT, dev)
        for p in self.model.parameters(): p.requires_grad_(False)
        from experiments.tagging.embedding import dense_to_sparse_jet, embed_tagging_data_into_ga
        from experiments.tagging.wrappers import xformers_sa_mask
        self._d2s, self._emb, self._mask = dense_to_sparse_jet, embed_tagging_data_into_ga, xformers_sa_mask
        self.dev, self.kind = dev, kind; self.blocks = self.model.net.blocks; assert len(self.blocks) == NBLK
        self.cap_k = None; self.cap = None
        for b, blk in enumerate(self.blocks): blk.register_forward_hook(self._hook(b))

    def _untrained(self, dev):
        import torch
        from interp.p5_lgatr_launch import LG
        cwd = os.getcwd(); os.chdir(LG)
        try:
            from omegaconf import OmegaConf
            from hydra.utils import instantiate
            import experiments.base_experiment  # noqa: F401
            from experiments.tagging.jetclassexperiment import JetClassTaggingExperiment
            cfg = OmegaConf.load(Path(CKPT).parents[1] / 'config.yaml'); exp = JetClassTaggingExperiment(cfg, 0, 1)
            exp.init_physics(); exp.init_geometric_algebra(); torch.manual_seed(SEED); model = instantiate(exp.cfg.model)
            model.force_xformers = dev == 'cuda' and bool(exp.cfg.training.force_xformers)
            return model.to(dev).eval(), exp.cfg
        finally:
            os.chdir(cwd)

    def _hook(self, b):
        def h(mod, args, out):
            if self.cap_k == b:
                self.cap = (out[0][0].detach(), out[1][0].detach()); raise Stop()
            return out
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
        assert torch.equal(cnt - 3, torch.from_numpy((Js['mask'] > 0).sum(1)).to(self.dev)), 'token count != particles + 3'
        e = {k: e[k] for k in ('mv', 's', 'is_global', 'batch')}
        e.update(pos=pos, start=start, cnt=cnt, nb=nb, amask=self._mask(bt, materialize=not self.model.force_xformers))
        return e

    def logits(self, e):
        from interp.p5_lgatr_launch import TO_CL
        return self.model({k: e[k] for k in ('mv', 's', 'is_global', 'batch')})[:, TO_CL]

    def state(self, e, k):
        self.cap_k = k
        try: self.model({kk: e[kk] for kk in ('mv', 's', 'is_global', 'batch')}); raise RuntimeError('hook did not fire')
        except Stop: pass
        finally: self.cap_k = None
        return self.cap

    def step(self, b, hm, hs, e):
        """run block b alone on a state (T, 16, 16), (T, 32)"""
        self.cap_k = None
        o = self.blocks[b](hm[None], scalars=hs[None], additional_qk_features_mv=None, additional_qk_features_s=None, attention_mask=e['amask'])
        return o[0][0], o[1][0]

    def tail(self, k, hm, hs, e):
        """blocks k+1..11 and the read-out from the state after block k"""
        from interp.p5_lgatr_launch import TO_CL
        x, s = hm[None], hs[None]
        for blk in self.blocks[k + 1:]:
            x, s = blk(x, scalars=s, additional_qk_features_mv=None, additional_qk_features_s=None, attention_mask=e['amask'])
        om, _ = self.model.net.linear_out(x, scalars=s)
        return om[0, e['is_global'], :, 0][:, TO_CL]


def source_map(eb, es):
    """for each base token the source token at the same position (global, references) or pT rank; -1 if the source has fewer"""
    import torch
    bt = eb['batch']; pos = eb['pos']; ns = es['cnt'] - 3
    r = pos - 3; ok = (pos < 3) | (r < ns[bt])
    m = torch.where(ok, es['start'][bt] + pos, torch.full_like(pos, -1)); return m


class PairSet:
    """batches of (base, source) pairs with embeddings, source map, targets and states advanced block by block"""

    def __init__(self, net, Jb, Js, Jt, bs, name):
        import torch
        self.net, self.name = net, name; self.n = len(Jb['label']); self.B = []
        with torch.no_grad():
            for s0 in range(0, self.n, bs):
                sl = np.arange(s0, min(self.n, s0 + bs)); eb = net.embed(Jb, sl); es = net.embed(Js, sl) if Js is not None else None
                d = dict(sl=sl, eb=eb, es=es, Lb=net.logits(eb).float())
                if Js is not None: d['Ls'] = net.logits(es).float(); d['map'] = source_map(eb, es)
                if Jt is not None:
                    et = net.embed(Jt, sl); assert len(et['batch']) == len(eb['batch']), 'token layout differs b vs target'; d['Lt'] = net.logits(et).float()
                d['tok_aligned'] = (d['map'] >= 0).float() if Js is not None else None
                d['tok_global'] = (eb['pos'] == 0).float()
                self.B.append(d)
        self.k = None
        for nm in ('Lb', 'Ls', 'Lt'):
            if nm in self.B[0]: setattr(self, nm, torch.cat([d[nm] for d in self.B]).cpu().numpy())
        if Js is not None:
            fr = [float((d['map'][d['eb']['pos'] >= 3] >= 0).float().mean()) for d in self.B]
            self.frac_particles_patched = float(np.mean(fr))
            self.frac_tokens_patched = float(np.mean([float(d['tok_aligned'].mean()) for d in self.B]))

    def goto(self, k):
        import torch
        with torch.no_grad():
            for d in self.B:
                if self.k is None or self.k > k:
                    d['hb'] = self.net.state(d['eb'], k)
                    if d['es'] is not None: d['hs'] = self.net.state(d['es'], k)
                else:
                    for b in range(self.k + 1, k + 1):
                        d['hb'] = self.net.step(b, *d['hb'], d['eb'])
                        if d['es'] is not None: d['hs'] = self.net.step(b, *d['hs'], d['es'])
        self.k = k


def patch(hb, hs, mp, tok, Us):
    import torch
    hm, hsc = hb; sm, ssc = hs[0][mp.clamp(min=0)], hs[1][mp.clamp(min=0)]
    dm = (sm - hm) * tok[:, None, None]; parts = []
    for g in ('g0', 'g1', 'g2', 'g3', 'g4'):
        x = hm[:, :, GR[g]]
        if g in Us: U = Us[g]; x = x + torch.einsum('cd,ed,tem->tcm', U, U, dm[:, :, GR[g]])
        parts.append(x)
    hm2 = torch.cat(parts, -1); hs2 = hsc
    if 'aux' in Us: U = Us['aux']; hs2 = hsc + ((ssc - hsc) * tok[:, None]) @ U @ U.T
    return hm2, hs2


def groups_n(g):
    return NS if g == 'aux' else NC


def rand_U(variant, d, gen, dev):
    import torch
    out = {}
    for g in VARIANTS[variant]:
        A = torch.from_numpy(gen.standard_normal((groups_n(g), d))).float().to(dev); out[g] = torch.linalg.qr(A).Q
    return out


def full_U(variant, dev):
    import torch
    return {g: torch.eye(groups_n(g), device=dev) for g in VARIANTS[variant]}


def patched_logits(ps, k, Us, tokmode, batches=None):
    import torch
    out = []
    with torch.no_grad():
        for i, d in enumerate(ps.B):
            if batches is not None and i not in batches: continue
            hm, hs = patch(d['hb'], d['hs'], d['map'], d['tok_' + tokmode], Us); out.append(ps.net.tail(k, hm, hs, d['eb']).float())
    return torch.cat(out).cpu().numpy()


def train_das(ps, k, variant, d, tokmode, target, P, seed):
    """target: (n, 10) numpy logits per pair of ps (cf or shuffled cf)"""
    import torch
    g = torch.Generator(device='cpu').manual_seed(seed)
    A = {gg: torch.randn(groups_n(gg), d, generator=g).to(ps.net.dev).requires_grad_(True) for gg in VARIANTS[variant]}
    opt = torch.optim.Adam(list(A.values()), lr=P['LR']); T = torch.from_numpy(target).float().to(ps.net.dev)
    rng = np.random.default_rng(seed); losses = []
    order = np.concatenate([rng.permutation(len(ps.B)) for _ in range(P['STEPS'] // len(ps.B) + 1)])[:P['STEPS']]
    for bi in order:
        dd = ps.B[bi]; Us = {gg: torch.linalg.qr(a).Q for gg, a in A.items()}
        with torch.enable_grad():
            hm, hs = patch(dd['hb'], dd['hs'], dd['map'], dd['tok_' + tokmode], Us)
            L = ps.net.tail(k, hm, hs, dd['eb']).float()
            pt = torch.softmax(T[torch.as_tensor(dd['sl'], device=T.device)], -1); loss = (pt * (torch.log(pt + 1e-12) - torch.log_softmax(L, -1))).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss))
    q = max(1, len(losses) // 4)
    with torch.no_grad():
        return {gg: torch.linalg.qr(a).Q.detach() for gg, a in A.items()}, [float(np.mean(losses[:q])), float(np.mean(losses[-q:]))]


# ------------------------------------------------------------------------------------------------ metrics
def boot_W(n, nb, seed):
    r = np.random.default_rng(seed); return np.stack([np.bincount(r.integers(0, n, n), minlength=n) for _ in range(nb)]).astype(np.float32)


def metrics(Lp, Lt, Lb, W):
    cp, ct, cb = Lp.argmax(1), Lt.argmax(1), Lb.argmax(1); hit = (cp == ct).astype(float); ch = ct != cb
    dp, dt, db = (L[:, iW] - L[:, iZ] for L in (Lp, Lt, Lb)); num, den = dp - db, dt - db
    bi = W @ hit / W.sum(1); bf = (W @ (num * den)) / (W @ (den * den))
    bch = (W[:, ch] @ hit[ch]) / np.maximum(W[:, ch].sum(1), 1)
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return dict(IIA=float(hit.mean()), IIA_ci=q(bi), IIA_ch=float(hit[ch].mean()) if ch.any() else None, IIA_ch_ci=q(bch),
                n_changed=int(ch.sum()), F=float((num * den).sum() / (den * den).sum()), F_ci=q(bf))


def gp(x, nul):
    import math
    nul = nul[np.isfinite(nul)]
    if not np.isfinite(x) or len(nul) < 2: return None
    sd = nul.std(ddof=1)
    if sd <= 0: return 0.0 if x > nul.mean() else 1.0
    return float(0.5 * math.erfc((x - nul.mean()) / sd / math.sqrt(2)))


def quick(Lp, Lt, Lb):
    hit = Lp.argmax(1) == Lt.argmax(1); ch = Lt.argmax(1) != Lb.argmax(1)
    dp, dt, db = (L[:, iW] - L[:, iZ] for L in (Lp, Lt, Lb))
    return float(hit.mean()), float(hit[ch].mean()) if ch.any() else np.nan, float(((dp - db) * (dt - db)).sum() / ((dt - db) ** 2).sum())


# ------------------------------------------------------------------------------------------------ edits
def make_edits(J, P, rng):
    from interp.part_core import sub, jmass
    y = J['label']; rows = np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), P['NED'], replace=False) for c in (iW, iZ)]))
    Jb = sub(J, rows); mb = jmass(Jb).astype(float); n = len(rows); one = np.ones(n)
    kM, eM = bisect(lambda x: jmass(dilate(Jb, one, kappa=x)).astype(float), 0.0, 1.0, 1.05 * mb); JM = dilate(Jb, one, kappa=kM)
    lD, eD = bisect(lambda x: jmass(dilate(Jb, x)).astype(float), 0.8, 1.3, 1.05 * mb); JDM = dilate(Jb, lD)
    k0, e0 = bisect(lambda x: jmass(dilate(Jb, one, kappa=x)).astype(float), 0.0, 1.0, 1.10 * mb); JR0 = dilate(Jb, one, kappa=k0)
    mR0 = jmass(JR0).astype(float)
    k1, e1 = bisect(lambda x: jmass(dilate(Jb, 1.05 * one, kappa=x)).astype(float), 0.0, 1.0, mR0); JS = dilate(Jb, 1.05 * one, kappa=k1)
    JDS = dilate(Jb, 1.05 * one, kappa=k0)
    sp = {nm: own_spread(JJ) for nm, JJ in (('b', Jb), ('M', JM), ('DM', JDM), ('R0', JR0), ('S', JS), ('DS', JDS))}
    ms = {nm: jmass(JJ).astype(float) for nm, JJ in (('b', Jb), ('M', JM), ('DM', JDM), ('R0', JR0), ('S', JS), ('DS', JDS))}
    info = dict(N=int(n), rows='1,000 random true W and 1,000 random true Z jets of the evaluation split', max_rel_mass_err=dict(M=float(eM.max()), DM=float(eD.max()), R0=float(e0.max()), S=float(e1.max())),
                kappa_M_median=float(np.median(kM)), kappa0_median=float(np.median(k0)), kappa1_median=float(np.median(k1)), lam_DM_median=float(np.median(lD)),
                frac_kappa1_lt_kappa0=float((k1 < k0).mean()),
                dlogm={a: float(np.median(np.log(ms[a] / ms[b]))) for a, b in (('M', 'b'), ('DM', 'b'), ('S', 'R0'), ('DS', 'R0'))},
                dlog_own_spread={a: float(np.median(np.log(sp[a] / sp[b]))) for a, b in (('M', 'b'), ('DM', 'b'), ('S', 'R0'), ('DS', 'R0'))})
    return dict(b=Jb, M=JM, DM=JDM, R0=JR0, S=JS, DS=JDS), info


def proj(hb, e, variant, U):
    """d = 1 coordinate per token: g0, aux = scalar; g1 = Minkowski square of the projected four-vector. -> (global, particle mean)"""
    import torch
    hm, hs = hb
    if variant == 'g0': v = hm[:, :, 0] @ U[:, 0]
    elif variant == 'aux': v = hs @ U[:, 0]
    else:
        w = torch.einsum('tcm,c->tm', hm[:, :, 1:5], U[:, 0]); v = w[:, 0] ** 2 - (w[:, 1:] ** 2).sum(-1)
    bt = e['batch']; pt_ = e['pos'] >= 3; nb = e['nb']
    gl = v[e['is_global']]
    pm = torch.zeros(nb, device=v.device).index_add_(0, bt[pt_], v[pt_]) / (e['cnt'] - 3).clamp(min=1).float()
    return gl, pm


# ------------------------------------------------------------------------------------------------ main (per block range)
def run(blocks, P, tag):
    import torch
    from interp.part_core import sub, jmass, sample
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; log('device', dev, torch.__version__)
    Jf = rebuild(load_split(FFIT, P['PER'], SEED)); Je = rebuild(load_split(FEVAL, P['PER'], SEED + 1)); rng = np.random.default_rng(SEED + 2)
    log(f'read fit _{FFIT} {len(Jf["label"])} jets, eval _{FEVAL} {len(Je["label"])} jets (rebuilt from per-particle pT, deta, dphi, mass)')
    out = dict(node='P59', tag=tag, blocks=blocks, params=P, checkpoint=CKPT, seed=SEED, files=dict(fit=FFIT, eval=FEVAL), sets={}, block={})
    S = {}
    for split, J in (('fit', Jf), ('eval', Je)):
        m, pt, y = jmass(J).astype(float), J['jet_pt'].astype(float), J['label'].astype(int)
        for kind in ('mass', 'pt'):
            b, s = make_pairs(m, pt, y, kind, P['NPAIR'], rng); Jb, Js, Jt, info = pair_set(J, b, s, kind)
            S[(split, kind)] = (Jb, Js, Jt); out['sets'][f'{split}_{kind}'] = info; log(f'pairs {split} {kind}: {info}')
    ED, edinfo = make_edits(Je, P, rng); out['edits_info'] = edinfo; log('edits:', edinfo)
    # example pair for the figure
    Jb, Js, Jt = S[('eval', 'mass')]; ex = {}
    from interp.part_core import kin
    for nm, JJ in (('base', Jb), ('source', Js), ('counterfactual', Jt)):
        pt_, de, dp, _ = kin(sub(JJ, [0])); mk = pt_[0] > 0
        ex[nm] = dict(pt=pt_[0][mk].tolist(), deta=de[0][mk].tolist(), dphi=dp[0][mk].tolist(), m=float(jmass(sub(JJ, [0]))[0]),
                      jet_pt=float(JJ['jet_pt'][0]), label=CL[int(JJ['label'][0])])
    out['example_pair'] = ex
    nets = dict(trained=Net('trained', dev), untrained=Net('untrained', dev)); log('nets loaded')
    J0 = sample(P['P1'], 3000); L0 = np.concatenate([nets['trained'].logits(nets['trained'].embed(J0, np.arange(s, min(len(J0['label']), s + 250)))).float().cpu().detach().numpy()
                                                   for s in range(0, len(J0['label']), 250)])
    out['P1_sample'] = dict(N=int(len(L0)), acc=float((L0.argmax(1) == J0['label']).mean()), expected=0.8373)
    log(f"P1 sample accuracy {100 * out['P1_sample']['acc']:.2f} % (N {len(L0)}; expected 83.73 % at N 3,000)")
    PS = {}
    PS['fit'] = PairSet(nets['trained'], *S[('fit', 'mass')], P['BSF'], 'fit')
    PS['eval'] = PairSet(nets['trained'], *S[('eval', 'mass')], P['BSE'], 'eval')
    if ARGS.ctl: return run_ctl(blocks, P, tag, out, PS, nets, dev)
    PS['dfit'] = PairSet(nets['trained'], *S[('fit', 'pt')], P['BSF'], 'dfit')
    PS['deval'] = PairSet(nets['trained'], *S[('eval', 'pt')], P['BSE'], 'deval')
    PS['ufit'] = PairSet(nets['untrained'], *S[('fit', 'mass')], P['BSF'], 'ufit')
    PS['ueval'] = PairSet(nets['untrained'], *S[('eval', 'mass')], P['BSE'], 'ueval')
    PE = {nm: PairSet(nets['trained'], JJ, None, None, P['BSE'], 'ed_' + nm) for nm, JJ in ED.items()}
    log('pair sets embedded')
    ev = PS['eval']; yb = S[('eval', 'mass')][0]['label'].astype(int)
    out['unedited'] = dict(acc_eval_bases_10class=float((ev.Lb.argmax(1) == yb).mean()),
                           acc_eval_bases_untrained=float((PS['ueval'].Lb.argmax(1) == yb).mean()),
                           frac_cf_call_changed=float((ev.Lt.argmax(1) != ev.Lb.argmax(1)).mean()),
                           frac_cf_call_changed_untrained=float((PS['ueval'].Lt.argmax(1) != PS['ueval'].Lb.argmax(1)).mean()),
                           frac_decoy_call_changed=float((PS['deval'].Lt.argmax(1) != PS['deval'].Lb.argmax(1)).mean()),
                           frac_particles_patched_aligned=ev.frac_particles_patched, frac_tokens_patched_aligned=ev.frac_tokens_patched,
                           mean_dd_cf=float(np.mean((ev.Lt[:, iW] - ev.Lt[:, iZ]) - (ev.Lb[:, iW] - ev.Lb[:, iZ]))))
    log('unedited:', out['unedited'])
    # edit network responses (read-out d = l_W - l_Z)
    dd = {nm: PE[nm].Lb[:, iW] - PE[nm].Lb[:, iZ] for nm in PE}; Wd = boot_W(len(dd['b']), P['NB'], SEED + 3)
    def ratio(a, b0, c, c0):
        num, den = dd[a] - dd[b0], dd[c] - dd[c0]; bs = (Wd @ num) / (Wd @ den)
        return [float(num.sum() / den.sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    out['edits_network'] = dict(R_M=ratio('M', 'b', 'DM', 'b'), R_S=ratio('S', 'R0', 'DS', 'R0'),
                                dd_DM=float(np.mean(dd['DM'] - dd['b'])), dd_M=float(np.mean(dd['M'] - dd['b'])),
                                dd_S=float(np.mean(dd['S'] - dd['R0'])), dd_DS=float(np.mean(dd['DS'] - dd['R0'])),
                                calls_changed=dict(M=float((PE['M'].Lb.argmax(1) != PE['b'].Lb.argmax(1)).mean()),
                                                   DM=float((PE['DM'].Lb.argmax(1) != PE['b'].Lb.argmax(1)).mean()),
                                                   S=float((PE['S'].Lb.argmax(1) != PE['R0'].Lb.argmax(1)).mean()),
                                                   DS=float((PE['DS'].Lb.argmax(1) != PE['R0'].Lb.argmax(1)).mean())))
    log('edit network responses:', out['edits_network'])
    jdump(out, RES / f'p59_{tag}{"_smoke" if P["smoke"] else ""}.json')
    W = boot_W(ev.n, P['NB'], SEED + 1); Wd_ = boot_W(PS['deval'].n, P['NB'], SEED + 2); Wu = boot_W(PS['ueval'].n, P['NB'], SEED + 4)
    nb_null = list(range(int(np.ceil(P['NNULL'] / P['BSE'])))); nn = min(ev.n, len(nb_null) * P['BSE'])
    rng_shuf = np.random.default_rng(SEED + 5)
    perm_f = rng_shuf.permutation(PS['fit'].n); perm_e = rng_shuf.permutation(ev.n)
    Usave = {}
    for k in blocks:
        tb = time.time(); R = {}
        for nm in PS: PS[nm].goto(k)
        for nm in PE: PE[nm].goto(k)
        # check: tail from the cached state reproduces the unedited logits
        d0 = ev.B[0]
        with torch.no_grad(): chk = float((ev.net.tail(k, *d0['hb'], d0['eb']).float() - d0['Lb']).abs().max())
        R['tail_check_max_abs_dlogit'] = chk; log(f'block {k}: states ready ({time.time() - tb:.0f}s); tail check max |dlogit| {chk:.2e}')
        R['no_patch'] = metrics(ev.Lb, ev.Lt, ev.Lb, W)
        for tm in ('aligned', 'global'):
            R[f'full_swap_{tm}'] = metrics(patched_logits(ev, k, full_U('all', dev), tm), ev.Lt, ev.Lb, W)
        log(f"block {k}: no-patch IIA {R['no_patch']['IIA']:.3f}; full swap aligned IIA {R['full_swap_aligned']['IIA']:.3f} F {R['full_swap_aligned']['F']:.3f}; "
            f"global IIA {R['full_swap_global']['IIA']:.3f} F {R['full_swap_global']['F']:.3f}")
        R['das'] = {}
        for tm in ('aligned', 'global'):
            for v in VARIANTS:
                for d in (DS_ if tm == 'aligned' else (1, 4)):
                    t1 = time.time(); key = f'{tm}_{v}_d{d}'
                    U, loss = train_das(PS['fit'], k, v, d, tm, PS['fit'].Lt, P, SEED + 1000 * k + 10 * d + len(v) + (500 if tm == 'global' else 0))
                    Lp = patched_logits(ev, k, U, tm); r = metrics(Lp, ev.Lt, ev.Lb, W); r['loss_first_last_epoch'] = loss
                    # null: random grade-preserving subspaces of the same d on the first nn eval pairs
                    gen = np.random.default_rng(SEED + 7 + 100 * k + 10 * d + list(VARIANTS).index(v))
                    nd = P['NDRAW'] if tm == 'aligned' else P['NDRAWG']; Lt_n, Lb_n = ev.Lt[:nn], ev.Lb[:nn]
                    das_n = quick(Lp[:nn], Lt_n, Lb_n); nul = np.array([quick(patched_logits(ev, k, rand_U(v, d, gen, dev), tm, nb_null), Lt_n, Lb_n) for _ in range(nd)])
                    r['null'] = dict(n_pairs=int(nn), draws=int(nd), das_on_subset=list(das_n),
                                     IIA_q=[float(x) for x in np.nanpercentile(nul[:, 0], [2.5, 50, 97.5])],
                                     IIA_ch_q=[float(x) for x in np.nanpercentile(nul[:, 1], [2.5, 50, 97.5])],
                                     F_q=[float(x) for x in np.nanpercentile(nul[:, 2], [2.5, 50, 97.5])],
                                     p_IIA=float((1 + np.sum(nul[:, 0] >= das_n[0])) / (1 + nd)),
                                     p_F=float((1 + np.sum(nul[:, 2] >= das_n[2])) / (1 + nd)),
                                     pz_IIA=gp(das_n[0], nul[:, 0]), pz_F=gp(das_n[2], nul[:, 2]), pz_IIA_ch=gp(das_n[1], nul[:, 1]),
                                     IIA_minus_null_median=float(das_n[0] - np.nanmedian(nul[:, 0])))
                    R['das'][key] = r; Usave[f'k{k}_{key}'] = {g: u.cpu().numpy() for g, u in U.items()}
                    log(f"block {k} {key}: IIA {r['IIA']:.3f} {np.round(r['IIA_ci'], 3).tolist()} IIA_ch {r['IIA_ch']} F {r['F']:.3f} "
                        f"{np.round(r['F_ci'], 3).tolist()} | null IIA median {r['null']['IIA_q'][1]:.3f} F median {r['null']['F_q'][1]:.3f} "
                        f"p_IIA {r['null']['p_IIA']:.4f} p_F {r['null']['p_F']:.4f} | loss {np.round(loss, 4).tolist()} ({time.time() - t1:.0f}s)")
        # controls at d = 4, aligned tokens
        R['controls'] = {}
        for v in VARIANTS:
            c = {}
            Uu, lu = train_das(PS['ufit'], k, v, 4, 'aligned', PS['ufit'].Lt, P, SEED + 50000 + 1000 * k + len(v))
            c['untrained'] = metrics(patched_logits(PS['ueval'], k, Uu, 'aligned'), PS['ueval'].Lt, PS['ueval'].Lb, Wu)
            c['untrained_no_patch'] = metrics(PS['ueval'].Lb, PS['ueval'].Lt, PS['ueval'].Lb, Wu)
            Ud, ld = train_das(PS['dfit'], k, v, 4, 'aligned', PS['dfit'].Lt, P, SEED + 60000 + 1000 * k + len(v))
            c['decoy_on_decoy'] = metrics(patched_logits(PS['deval'], k, Ud, 'aligned'), PS['deval'].Lt, PS['deval'].Lb, Wd_)
            c['decoy_on_mass'] = metrics(patched_logits(ev, k, Ud, 'aligned'), ev.Lt, ev.Lb, W)
            Um = {g: torch.from_numpy(u).to(dev) for g, u in Usave[f'k{k}_aligned_{v}_d4'].items()}
            c['mass_on_decoy'] = metrics(patched_logits(PS['deval'], k, Um, 'aligned'), PS['deval'].Lt, PS['deval'].Lb, Wd_)
            c['decoy_no_patch'] = metrics(PS['deval'].Lb, PS['deval'].Lt, PS['deval'].Lb, Wd_)
            c['overlap_mass_decoy'] = {g: float((Um[g].T @ Ud[g]).pow(2).sum() / 4) for g in Um}
            if v in ('all', 'g0'):
                Us_, ls_ = train_das(PS['fit'], k, v, 4, 'aligned', PS['fit'].Lt[perm_f], P, SEED + 70000 + 1000 * k + len(v))
                Lp = patched_logits(ev, k, Us_, 'aligned')
                c['shuffled_vs_shuffled_target'] = metrics(Lp, ev.Lt[perm_e], ev.Lb, W); c['shuffled_vs_true_cf'] = metrics(Lp, ev.Lt, ev.Lb, W)
            else:
                c['shuffled_vs_shuffled_target'] = c['shuffled_vs_true_cf'] = dict(IIA=np.nan, F=np.nan, IIA_ch=None)
            c['shuffled_no_patch'] = metrics(ev.Lb, ev.Lt[perm_e], ev.Lb, W)
            R['controls'][v] = c
            log(f"block {k} controls {v}: untrained IIA {c['untrained']['IIA']:.3f} IIA_ch {c['untrained']['IIA_ch']} F {c['untrained']['F']:.3f} "
                f"(no patch {c['untrained_no_patch']['IIA']:.3f}); decoy IIA {c['decoy_on_decoy']['IIA']:.3f} F {c['decoy_on_decoy']['F']:.3f} "
                f"(no patch {c['decoy_no_patch']['IIA']:.3f}); decoy-subspace on mass F {c['decoy_on_mass']['F']:.3f}; mass-subspace on decoy F "
                f"{c['mass_on_decoy']['F']:.3f}; shuffled IIA {c['shuffled_vs_shuffled_target']['IIA']:.3f} (no patch {c['shuffled_no_patch']['IIA']:.3f}) "
                f"F vs true cf {c['shuffled_vs_true_cf']['F']:.3f}; overlap {c['overlap_mass_decoy']}")
        # edits: projections on the d = 1 subspaces (aligned training) of g0, aux, g1
        R['edits'] = {}
        ne = PE['b'].n; We = boot_W(ne, P['NB'], SEED + 6); lm = np.log(jmass(ED['b']).astype(float)); lsp = np.log(own_spread(ED['b']))
        for v in ('g0', 'aux', 'g1'):
            for tm in ('aligned', 'global'):
                U = torch.from_numpy(Usave[f'k{k}_{tm}_{v}_d1'][v]).to(dev); pr = {}
                with torch.no_grad():
                    for nm in PE:
                        a, b_ = zip(*[proj(dd_['hb'], dd_['eb'], v, U) for dd_ in PE[nm].B]); pr[nm] = (torch.cat(a).cpu().numpy(), torch.cat(b_).cpu().numpy())
                e = {}
                for j, where in enumerate(('global_token', 'particle_mean')):
                    def rr(a, b0, c, c0):
                        num, den = pr[a][j] - pr[b0][j], pr[c][j] - pr[c0][j]; bs = (We @ num) / (We @ den)
                        return [float(num.sum() / den.sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
                    x = pr['b'][j]
                    sd_ = x.std() + 1e-12
                    e[where] = dict(ratio_M=rr('M', 'b', 'DM', 'b'), ratio_S=rr('S', 'R0', 'DS', 'R0'),
                                    dproj_over_sd={a: float(np.mean(pr[a][j] - pr[b0][j]) / sd_) for a, b0 in (('M', 'b'), ('DM', 'b'), ('S', 'R0'), ('DS', 'R0'))},
                                    corr_logm=float(np.corrcoef(x, lm)[0, 1]), corr_log_spread=float(np.corrcoef(x, lsp)[0, 1]),
                                    dproj_DM_over_sd=float(np.mean(pr['DM'][j] - x) / (x.std() + 1e-12)))
                R['edits'][f'{tm}_{v}'] = e
                log(f"block {k} edits {tm} {v} d1: global ratio_M {np.round(e['global_token']['ratio_M'], 3).tolist()} ratio_S "
                    f"{np.round(e['global_token']['ratio_S'], 3).tolist()} corr log m {e['global_token']['corr_logm']:.3f} | particle-mean ratio_M "
                    f"{np.round(e['particle_mean']['ratio_M'], 3).tolist()} ratio_S {np.round(e['particle_mean']['ratio_S'], 3).tolist()}")
        out['block'][str(k)] = R; log(f'block {k} done in {time.time() - tb:.0f}s')
        jdump(out, RES / f'p59_{tag}{"_smoke" if P["smoke"] else ""}.json')
        np.savez_compressed(RES / f'p59_U_{tag}{"_smoke" if P["smoke"] else ""}.npz', **{f'{a}__{g}': u for a, gg in Usave.items() for g, u in gg.items()})
    log('done')


def joint(Lp, Lt, Lb, Ls, W):
    """least squares of the patched change on (cf change, source-output change) jointly, d = l_W - l_Z; plus call agreement
    with the source on pairs where the source call differs from the cf call"""
    dp, dt, db, ds = (L[:, iW] - L[:, iZ] for L in (Lp, Lt, Lb, Ls)); y = dp - db; X = np.stack([dt - db, ds - db], 1)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]; bb = []
    for w in W:
        A = (X * w[:, None]).T @ X; bb.append(np.linalg.solve(A, (X * w[:, None]).T @ y))
    bb = np.array(bb); cp, ct, cs = Lp.argmax(1), Lt.argmax(1), Ls.argmax(1); m = cs != ct
    return dict(beta_cf=float(beta[0]), beta_cf_ci=[float(x) for x in np.percentile(bb[:, 0], [2.5, 97.5])], beta_src=float(beta[1]),
                beta_src_ci=[float(x) for x in np.percentile(bb[:, 1], [2.5, 97.5])], corr_cf_src=float(np.corrcoef(X[:, 0], X[:, 1])[0, 1]),
                n_src_ne_cf=int(m.sum()), frac_eq_src=float((cp[m] == cs[m]).mean()), frac_eq_cf=float((cp[m] == ct[m]).mean()))


def run_ctl(blocks, P, tag, out, PS, nets, dev):
    """follow-up: shuffled-pair control for every variant at d = 1 and 4 (aligned tokens), and a joint regression of the patched
    change on the cf change and on the source jet's own output change, for DAS and shuffled subspaces"""
    ev = PS['eval']; W = boot_W(ev.n, P['NB'], SEED + 1); rng_shuf = np.random.default_rng(SEED + 5)
    perm_f = rng_shuf.permutation(PS['fit'].n); perm_e = rng_shuf.permutation(ev.n)
    out['ctl'] = {}; out['full_swap'] = {}
    for k in blocks:
        tb = time.time(); PS['fit'].goto(k); ev.goto(k); R = {}
        Lfull = patched_logits(ev, k, full_U('all', dev), 'aligned'); R['full_swap'] = dict(metrics(Lfull, ev.Lt, ev.Lb, W), **joint(Lfull, ev.Lt, ev.Lb, ev.Ls, W))
        R['source_output_as_patch'] = dict(metrics(ev.Ls, ev.Lt, ev.Lb, W), **joint(ev.Ls, ev.Lt, ev.Lb, ev.Ls, W))
        for v in VARIANTS:
            for d in (1, 4):
                if k == 11 and v == 'g1': continue
                U, _ = train_das(PS['fit'], k, v, d, 'aligned', PS['fit'].Lt, P, SEED + 1000 * k + 10 * d + len(v))
                Lp = patched_logits(ev, k, U, 'aligned'); r = dict(das=dict(metrics(Lp, ev.Lt, ev.Lb, W), **joint(Lp, ev.Lt, ev.Lb, ev.Ls, W)))
                Us_, _ = train_das(PS['fit'], k, v, d, 'aligned', PS['fit'].Lt[perm_f], P, SEED + 70000 + 1000 * k + 10 * d + len(v))
                Lq = patched_logits(ev, k, Us_, 'aligned')
                r['shuffled'] = dict(metrics(Lq, ev.Lt, ev.Lb, W), **joint(Lq, ev.Lt, ev.Lb, ev.Ls, W))
                r['shuffled_vs_own_target'] = metrics(Lq, ev.Lt[perm_e], ev.Lb, W)
                R[f'{v}_d{d}'] = r
                log(f"ctl block {k} {v} d{d}: DAS F {r['das']['F']:.3f} beta_cf {r['das']['beta_cf']:.3f} {np.round(r['das']['beta_cf_ci'], 3).tolist()} "
                    f"beta_src {r['das']['beta_src']:.3f} IIA_ch {r['das']['IIA_ch']:.3f} | shuffled F {r['shuffled']['F']:.3f} beta_cf {r['shuffled']['beta_cf']:.3f} "
                    f"{np.round(r['shuffled']['beta_cf_ci'], 3).tolist()} beta_src {r['shuffled']['beta_src']:.3f} IIA_ch {r['shuffled']['IIA_ch']:.3f}")
        out['ctl'][str(k)] = R; log(f'ctl block {k} done in {time.time() - tb:.0f}s; full swap beta_cf {R["full_swap"]["beta_cf"]:.3f} beta_src {R["full_swap"]["beta_src"]:.3f}')
        jdump(out, RES / f'p59_ctl_{tag}{"_smoke" if P["smoke"] else ""}.json')
    log('done')


# ------------------------------------------------------------------------------------------------ merge, Holm, figure
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.zeros(m); run_ = 0.0
    for r, i in enumerate(o):
        run_ = max(run_, min(1.0, (m - r) * p[i])); adj[i] = run_
    return adj


def merge():
    tags = ARGS.tags.split(','); parts = [json.loads((RES / f'p59_{t}.json').read_text()) for t in tags]
    out = {k: v for k, v in parts[0].items() if k not in ('block', 'tag', 'blocks')}; out['block'] = {}
    for p in parts: out['block'].update(p['block'])
    out['P1_sample_per_job'] = [p['P1_sample']['acc'] for p in parts]
    out['ctl'] = {}
    for t in ('a', 'b', 'c'):
        f = RES / f'p59_ctl_{t}.json'
        if f.exists(): out['ctl'].update(json.loads(f.read_text()).get('ctl', {}))
    keys = [(k, f'aligned_{v}_d{d}') for k in sorted(out['block'], key=int) for v in VARIANTS for d in DS_]
    for stat in ('p_IIA', 'p_F', 'pz_IIA', 'pz_F', 'pz_IIA_ch'):
        ph = holm([1.0 if out['block'][k]['das'][key]['null'][stat] is None else out['block'][k]['das'][key]['null'][stat] for k, key in keys])
        for (k, key), h in zip(keys, ph): out['block'][k]['das'][key]['null'][stat + '_holm'] = float(h)
    keysg = [(k, f'global_{v}_d{d}') for k in sorted(out['block'], key=int) for v in VARIANTS for d in (1, 4)]
    for stat in ('p_IIA', 'p_F', 'pz_IIA', 'pz_F', 'pz_IIA_ch'):
        ph = holm([1.0 if out['block'][k]['das'][key]['null'][stat] is None else out['block'][k]['das'][key]['null'][stat] for k, key in keysg])
        for (k, key), h in zip(keysg, ph): out['block'][k]['das'][key]['null'][stat + '_holm'] = float(h)
    best = max(keys, key=lambda kk: out['block'][kk[0]]['das'][kk[1]]['IIA'])
    out['summary'] = dict(holm_family=f'{len(keys)} tests (12 blocks x 4 variants x 3 d, aligned tokens); global-token family separate',
                          best_aligned=dict(block=best[0], key=best[1], **{x: out['block'][best[0]]['das'][best[1]][x] for x in ('IIA', 'IIA_ci', 'IIA_ch', 'F', 'F_ci')}),
                          H1_IIA_ge_0p8=[f'b{k} {key}' for k, key in keys if out['block'][k]['das'][key]['IIA'] >= 0.8],
                          H1_IIAch_ge_0p8=[f'b{k} {key}' for k, key in keys if (out['block'][k]['das'][key]['IIA_ch'] or 0) >= 0.8])
    jdump(out, RES / 'p59_lgatr_das_mass.json')
    figure(out, RES / 'p59_lgatr_das_mass.png')


def figure(out, path):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ks = sorted(out['block'], key=int); x = np.array([int(k) for k in ks]); B = out['block']
    col = dict(g0='#3465a4', g1='#cc0000', aux='#4e9a06', all='#75507b')
    nz = lambda v: np.nan if v is None else v
    lab = dict(g0='grade 0 (scalar parts)', g1='grade 1 (four-vectors)', aux='auxiliary scalars', all='all grades')
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    a = ax[0, 0]
    for v in VARIANTS:
        y = [nz(B[k]['das'][f'aligned_{v}_d4']['IIA_ch']) for k in ks]
        lo = [B[k]['das'][f'aligned_{v}_d4']['null']['IIA_ch_q'][0] for k in ks]; hi = [B[k]['das'][f'aligned_{v}_d4']['null']['IIA_ch_q'][2] for k in ks]
        a.plot(x, y, 'o-', color=col[v], label=f'DAS, {lab[v]}'); a.fill_between(x, lo, hi, color=col[v], alpha=0.12)
    a.plot(x, [nz(B[k]['no_patch']['IIA_ch']) for k in ks], 'k:', label='no patch')
    a.plot(x, [nz(B[k]['full_swap_aligned']['IIA_ch']) for k in ks], 'k--', label='whole state swapped')
    a.axhline(0.8, color='grey', lw=0.6)
    a.set_xlabel('block k (patch after block k)'); a.set_ylabel('IIA on pairs whose call the mass change flips')
    a.set_title('(a) d = 4, aligned tokens; bands: 95 % of 100 random subspaces', fontsize=10); a.legend(fontsize=7); a.set_ylim(-0.02, 1.02)
    a = ax[0, 1]; C = out.get('ctl', {}); kc = [k for k in ks if k in C]; xc = [int(k) for k in kc]
    for v, d in (('aux', 1), ('all', 4), ('g0', 4), ('g1', 4)):
        kk = [k for k in kc if f'{v}_d{d}' in C[k]]
        if not kk: continue
        r = np.array([C[k][f'{v}_d{d}']['das']['beta_cf_ci'] for k in kk])
        a.plot([int(k) for k in kk], [C[k][f'{v}_d{d}']['das']['beta_cf'] for k in kk], '-', marker='o', ms=4, color=col[v], label=f'DAS {lab[v]}, d = {d}')
        a.fill_between([int(k) for k in kk], r[:, 0], r[:, 1], color=col[v], alpha=0.15)
        a.plot([int(k) for k in kk], [C[k][f'{v}_d{d}']['shuffled']['beta_cf'] for k in kk], ':', marker='x', ms=5, color=col[v], label=f'shuffled-pair control, {lab[v]}, d = {d}')
    if kc: a.plot(xc, [C[k]['full_swap']['beta_cf'] for k in kc], 'k--', label='whole state swapped')
    a.axhline(0, color='k', lw=0.5)
    a.set_xlabel('block k'); a.set_ylabel(r'mass-specific share $\beta_{cf}$ of the change in $l_W - l_Z$')
    a.set_title(r'(b) $\Delta d_{patch} = \beta_{cf}\,\Delta d_{cf} + \beta_{src}\,\Delta d_{source}$ (joint fit); bands: 95 % bootstrap', fontsize=9); a.legend(fontsize=6)
    a = ax[1, 0]; ex = out['example_pair']
    for nm, c, mk in (('base', '#3465a4', 'o'), ('source', '#cc0000', 's'), ('counterfactual', '#4e9a06', '^')):
        e = ex[nm]; pt = np.array(e['pt'])
        a.scatter(e['deta'], e['dphi'], s=300 * pt / pt.max() + 3, facecolors='none' if nm != 'base' else c, edgecolors=c, marker=mk, alpha=0.7,
                  label=f"{nm}: {e['label']}, m = {e['m']:.1f} GeV, pT = {e['jet_pt']:.0f} GeV")
    a.set_xlabel('Δη to the stored jet axis'); a.set_ylabel('Δφ to the stored jet axis'); a.legend(fontsize=7)
    a.set_title('(c) one pair: counterfactual = base with angles scaled until m = m_source', fontsize=10)
    a = ax[1, 1]
    for v, mk in (('aux', 'o'), ('g0', 's')):
        dp = {e: np.array([B[k]['edits'][f'global_{v}']['global_token']['dproj_over_sd'][e] for k in ks]) for e in ('M', 'DM', 'S', 'DS')}
        sg = np.sign(dp['DM']); sg[sg == 0] = 1
        for e, c, lbl in (('DM', '#3465a4', 'DM: angles scaled, m +5 %'), ('M', '#cc0000', 'M: particle masses raised, m +5 %, angles fixed'),
                          ('S', '#4e9a06', 'S: angles +5 %, particle masses lowered, m fixed'), ('DS', '#c4a000', 'DS: angles +5 % from R0, m +4 %')):
            a.plot(x, sg * dp[e], marker=mk, ls='-' if v == 'aux' else ':', color=c, ms=4, label=f'{lab[v]}: {lbl}' if v == 'aux' else f'{lab[v]}: {e}')
    a.axhline(0, color='k', lw=0.5)
    a.set_xlabel('block k'); a.set_ylabel('change of the d = 1 coordinate at the global token (s.d.)')
    a.set_title('(d) one trained direction under the edits (sign: DM > 0); invariant-mass reader: M = DM, S = 0', fontsize=9); a.legend(fontsize=6)
    fig.suptitle('P59: one auxiliary-scalar direction of L-GATr carries about 40 % of the mass-specific response to a real mass change; no subspace reaches IIA 0.8 (3,000 held-out pairs)', fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=110); log('figure', path)


if __name__ == '__main__':
    if ARGS.merge: merge()
    else:
        P = params(ARGS.smoke)
        try: run([int(b) for b in ARGS.blocks.split(',')], P, ARGS.tag)
        except Exception: log('ERROR\n' + traceback.format_exc()); sys.exit(1)
