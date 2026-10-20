"""P8 — where ParT's axis-relative mass forms: per-block residual patching, an M2-direction search (DAS) and substitution into n98.

Claim C3 / C2b (thesis C v2 step 3; absorbs proposed P7). C3: ParT's mass response enters through the per-particle deltaR input
and follows the axis moment, not the invariant mass (phi + 0.06 rad: invariant mass fixed, W -> Z 12 -> 31 %). C2b: n98 (z[97],
the W/Z class-token coordinate after the final LayerNorm) reads it through the eight block-1 class-attention heads (b1h3 about
half). P8 asks at which particle block the deltaR input becomes a jet-level M2-like variable, and whether that variable is a
low-dimensional linear subspace that, written into a jet, moves n98 and the W/Z call as the phi shift does.

Model ParT_full (released JetClass weights, weaver-core 0.4.17); ParT_kin as replication arm; untrained ParT_full (every weight
tensor redrawn from a Gaussian with the trained tensor's mean and s.d., ParTNetwork(untrained=True)) as the network control.
Samples (JetClass test_20M, random rows, numpy default_rng(2026), never first-k; build = part_core.build, eta = jet_eta +
sign(jet_eta) * deta):
  WZ : 10,000 random rows of WToQQ_101.root + 10,000 of ZToQQ_101.root (20,000 jets).  Fit/test halves (permutation seed 2026).
  T10: 500 random rows of each class's <sample>_100.root (5,000 jets, 10 classes) -> 10-class read.
  P1 : rows 3,000-4,000 of each class's first test file (part_core.sample) -> unedited-accuracy check against P1 (85.9 / 74.5 %).
Observables (about the stored jet axis; deta, dphi as stored, dR = hypot(deta, dphi)): m = invariant mass of the particles
[GeV]; 'M0' in the task text = invariant mass squared, log m^2; M2 = sum pT dR^2 / sum pT (axis moment); mom = sqrt(sum pT * sum
pT dR^2) (the axis-moment mass estimate, E6); girth = sum pT dR / sum pT.
Edits (every real particle; pT and particle mass kept unless stated; px, py, pz, E, jet_pt, jet_energy rebuilt):
  phi d   : dphi + d (d = 0.03, 0.06, 0.09): invariant mass unchanged, M2 increases by about d^2;
  eta d   : deta + d (d = 0.06): rigid shift in eta (longitudinal boost; invariant mass unchanged up to particle masses);
  ang k   : (deta, dphi) x k (k = 1.1, and k_match = exp(mean 1/2 dlog M2 of phi 0.06) = the matched-calibration real-mass edit
            with the same predicted (axis-moment) change as phi 0.06);
  pt c    : pT and particle mass x c (c = 0.8, 1.25): invariant mass x c, M2 unchanged.
(a) Residual patching (WZ, N_PATCH random jets; ParT_full: edits phi 0.06, ang 1.1, ang k_match with the edited 4-vectors).
    Sites per particle block l = 1..8: attention sublayer output (post_attn_norm output, before the residual add), MLP output
    (fc2 output, before the residual add), and the block output (whole residual stream). All particles, position by position
    (the edits keep the pT order, so slot p is the same particle).
      denoise : edited run, site set to its clean value; restored share = 1 - mean(dP) / mean(dE)
      noise   : clean run, site set to its edited value; carried share = mean(dP) / mean(dE)
      resample: edited run, site set to the clean value of a donor jet in the same multiplicity batch, same class, |dm| < 5 GeV
                (positions beyond the donor's multiplicity wrap, as weaver's padding)  [control]
      subset  : edited run, clean value at a random 50 % of the real particles  [control]
      cumulative (added after the smoke run showed each single sublayer restores 1-2 %): edited run with the clean outputs of all
                attention sublayers (allA), all MLPs (allM), all 16 sublayers (allsub: only the direct residual path from the edited
                embedding still carries the edit), and the sublayers of blocks 1..l (upto l); noise versions of allA / allM / allsub.
    d = per-jet change vs the clean run of n98 [s.d. of n98 over WZ] and of logit_W - logit_Z; W -> Z rate on true W jets.
    95 % bootstrap over jets (NB resamples); H0: restored share = 0, two-sided bootstrap p, Holm over 16 sublayers x 2 edits
    (phi 0.06, ang 1.1); also H0': share <= 0.5 (one-sided).
    ParT_kin: denoise / noise / block output for phi 0.06 and ang 1.1 on N_PATCH_KIN jets (logit_W - logit_Z; z[97] is not a W/Z
    coordinate in ParT_kin).
(b) Representation search at sites emb (embedding output), b1..b8 (particle block outputs), cls1 (class token after class block 1).
    Probes: ridge (part p1v Ridge; alpha chosen on the last quarter of the fit half) on the masked particle mean of each particle
    site (cls1: the token) for log m^2, log M2, log mom, log girth; R^2 on the test half; probe read-out change under phi 0.06,
    eta 0.06, ang 1.1, ang k_match, pt 1.25 vs the true change, and the ratio read-out(phi 0.06) / read-out(ang k_match) (1 =
    reads the two the same, as an axis-moment variable; 0 = reads the invariant mass).
    DAS: orthonormal Q (128 x k, k = 1, 2, 4; QR of a free matrix, Adam) trained on the fit half so that the interchanged run
    reproduces the counterfactual run's 10 logits (MSE). Counterfactual = the same jet with the edit applied to the per-particle
    inputs (4-vectors, hence pair features, kept from the unedited jet, so the whole edit flows through the residual stream).
      variable 'M2': counterfactual edits phi 0.03 / 0.06 / 0.09 (M2 up, invariant mass fixed);
      variable 'M0': counterfactual edits pt 0.8 / 1.25 (invariant mass x c, M2 fixed; the per-particle route sees pt_log, e_log).
    Fitted: 'M2' jm k = 1, 2, 4 at every particle site, tok k = 1, 2, 4 at cls1, pp k = 1, 4; 'M0' jm / tok k = 4 (STEPS Adam
    steps of BS jets; lr 0.02). Untrained ParT_full and ParT_kin: 'M2' jm / tok k = 4 at every site.
    Interchange modes: 'pp' h_p <- h_p + (h'_p - h_p) Q Q^T per particle; 'jm' h_p <- h_p + (mean h' - mean h) Q Q^T (jet-level
    variable broadcast to every particle; masked mean over real particles); 'tok' at cls1.
    Evaluation on N_EVAL test-half jets for counterfactuals phi 0.06 (trained edit), eta 0.06 (held-out edit, same M2 increase),
    ang k_match (real mass change), pt 1.25 (M2 fixed): logit recovery = 1 - sum ||L_int - L_cf||^2 / sum ||L_base - L_cf||^2
    ("IIA" here); call IIA = share of jets whose 10-class call the edit changes that get the counterfactual call; n98 recovery =
    mean dz97_int / mean dz97_cf. Controls: NRAND random orthonormal Q of equal k (same site / mode), DAS on the untrained
    ParT_full, M0-DAS evaluated on the M2 counterfactuals. Paired bootstrap over jets, Holm over the 57 trained 'M2' subspaces.
(c) Substitution at the earliest particle block l* where the 'jm' M2 subspace (k = 4) reaches >= 0.8 x the best block's phi 0.06
    recovery; k* = the smallest k with >= 0.9 x the k = 4 recovery there. On N_SUB test-half W/Z jets: recipient's masked-mean
    projection replaced by a natural donor's (same multiplicity batch, same class, |dm| < 3 GeV, |dlog M2| >= 0.15): h_p <- h_p +
    (mean_d - mean_r) Q Q^T. Variants: DAS Q*, NRAND random Q, M0-DAS Q (same site / k), all 128 dims (Q = I). Metrics: slope (through
    0) of dz97 and d(lW - lZ) on dlog M2 (donor - recipient), vs the same slope under phi 0.06 (per-jet dlog M2 of the edit);
    W -> Z on W recipients with dlog M2 > 0; McNemar on 10-class correctness. Path patching (P7) with P3's exact class-stage
    re-implementation (Stage): substituted particle embeddings reach n98 only through (i) block-1 head h (its context vector
    from the substituted run, everything else from the base run; block-2 keys/values from the base run), (ii) b1h3 + b1h6, (iii)
    all block-1 heads, (iv) block-2 heads directly (block-1 contexts from the base run); share = slope via path / slope of the
    full substitution (substitution) or mean via path / mean full (phi 0.06 edit on the same jets).
    10-class read on T10: self-donor phi 0.06 interchange through Q* vs the full edit vs random Q: calls and confusion.

  ./sync.sh && ./part_job.sh c-p8-smoke 'python -m interp.p8_mass_formation_part --smoke'
  ./sync.sh && ./part_job.sh c-p8-run 'python -m interp.p8_mass_formation_part'   -> results/part/p8_mass_formation.json + .png
"""
import contextlib, json, math, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SMOKE = '--smoke' in sys.argv
SEED = 2026
TEST = Path(os.environ.get('P8_TEST', '/j-jepa-vol/JetClass/Pythia/test_20M'))
OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'
FN = OUT / ('p8_mass_formation_smoke.json' if SMOKE else 'p8_mass_formation.json')
PER_WZ = 300 if SMOKE else 10000
PER_10 = 30 if SMOKE else 500
MAXROW = 25000 if SMOKE else None
N_PATCH = 300 if SMOKE else 6000
N_PATCH_KIN = 200 if SMOKE else 4000
NB = 100 if SMOKE else 1000
STEPS = 12 if SMOKE else 150
BS = 64 if SMOKE else 128
N_EVAL = 150 if SMOKE else 3000
N_SUB = 200 if SMOKE else 6000
NRAND = 2 if SMOKE else 3
DIMS = (1, 2, 4)
BATCH = 256
C, H, HD = 128, 8, 16
SITES = ['emb'] + [f'b{l}' for l in range(1, 9)] + ['cls1']
SKEY = {'emb': ('e',), 'cls1': ('c1',), **{f'b{l}': ('b', l - 1) for l in range(1, 9)}}
M2_CF = ['phi0.03', 'phi0.06', 'phi0.09']
M0_CF = ['pt0.8', 'pt1.25']
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:6.0f} s]', *a, flush=True)


from interp.part_core import ParTNetwork, inputs, read_root, sample, kin, build, jmass, sub, cat, CL, PRE, N98  # noqa: E402
from interp.p3_wz_circuit_part import boot_W, ci, p_two, holm, mcnemar, Stage  # noqa: E402
from interp.p1v_verify_part import Ridge, r2  # noqa: E402

iW, iZ = CL.index('Wqq'), CL.index('Zqq')


# ------------------------------------------------------------------------------------------------------------ data
def load_rows(path, n, rng, chunk=25000, maxrow=None):
    import uproot
    N = uproot.open(path)['tree'].num_entries
    for mr in (MAXROW, maxrow):
        if mr: N = min(N, mr)
    rows = np.sort(rng.choice(N, n, replace=False)); parts = []
    for s in range(0, N, chunk):
        r = rows[(rows >= s) & (rows < s + chunk)]
        if len(r): parts.append(sub(read_root(path, s, min(s + chunk, N)), r - s))
    return cat(parts), rows


def obs(J):
    pt, de, dp, _ = kin(J); pt = pt.astype(np.float64); dr2 = de.astype(np.float64) ** 2 + dp.astype(np.float64) ** 2
    S0 = pt.sum(1); S2 = (pt * dr2).sum(1); S1 = (pt * np.sqrt(dr2)).sum(1); m = np.maximum(jmass(J).astype(np.float64), 1.0)
    return dict(m=m, log_m2=2 * np.log(m), log_M2=np.log(np.maximum(S2, 1e-12) / S0), log_mom=0.5 * np.log(np.maximum(S0 * S2, 1e-12)),
                log_girth=np.log(np.maximum(S1, 1e-12) / S0))


TARGETS = ['log_m2', 'log_M2', 'log_mom', 'log_girth']


def make_edit(S, name):
    pt, de, dp, ms, r = S['pt'], S['de'], S['dp'], S['ms'], S['real']
    if name == 'base': return build(S['J0'], pt, de, dp, ms)
    if name == 'angk': k = S['k_match']; return build(S['J0'], pt, de * k, dp * k, ms)
    kind, val = name[:3] if not name.startswith('pt') else 'pt', float(name[3:] if not name.startswith('pt') else name[2:])
    if kind == 'phi': return build(S['J0'], pt, de, dp + val * r, ms)
    if kind == 'eta': return build(S['J0'], pt, de + val * r, dp, ms)
    if kind == 'ang': return build(S['J0'], pt, de * val, dp * val, ms)
    if kind == 'pt': return build(S['J0'], pt * val, de, dp, ms * val)
    raise ValueError(name)


def prepare(J0, names, models, vnames, k_match=None):
    """jet sample -> D: per model the fed inputs x[name] (N, F, P), 4-vectors v[name], mask, counts, labels, observables"""
    pt, de, dp, ms = kin(J0)
    S = dict(J0=J0, pt=pt, de=de, dp=dp, ms=ms, real=(pt > 0).astype(np.float32))
    D = dict(y=J0['label'].astype(int), x={n: {} for n in models}, v={}, obs={})
    for name in ['base'] + [n for n in names if n != 'angk'] + (['angk'] if 'angk' in names else []):
        if name == 'angk':
            if k_match is None: k_match = float(np.exp(np.mean(0.5 * (D['obs']['phi0.06']['log_M2'] - D['obs']['base']['log_M2']))))
            S['k_match'] = k_match
        Je = make_edit(S, name); D['obs'][name] = obs(Je)
        for n in models:
            x, v, m = inputs(Je, n); D['x'][n][name] = x
            if name == 'base': D['m'] = m
            else: assert np.array_equal(m, D['m'])
        if name in vnames or name == 'base': D['v'][name] = v
        del Je
    D['cnt'] = D['m'][:, 0].sum(1).astype(int); D['k_match'] = k_match
    return D


# ------------------------------------------------------------------------------------------------------------ model with hooks
class Net:
    """forward hooks on the embedding, every particle block's attention-sublayer output (post_attn_norm), MLP output (fc2) and
    block output, and class block 1's output. patch[key] = (replacement (P, n, C), position mask or None); fn[key] = f(out);
    want = keys to record (cloned, after any patch / fn)."""

    def __init__(self, pn):
        import torch
        self.t, self.pn, self.m, self.dev = torch, pn, pn.model, pn.device
        for p in self.m.parameters(): p.requires_grad_(False)
        self.patch, self.fn, self.want, self.rec = {}, {}, set(), {}
        mods = {('e',): self.m.embed, ('c1',): self.m.cls_blocks[0]}
        for i, b in enumerate(self.m.blocks):
            mods[('a', i)] = b.post_attn_norm; mods[('m', i)] = b.fc2; mods[('b', i)] = b
        self.hooks = [mod.register_forward_hook(self._h(k)) for k, mod in mods.items()]

    def _h(self, key):
        def f(mod, inp, out):
            if key in self.patch:
                rep, pm = self.patch[key]; out = rep.clone() if pm is None else self.t.where(pm, rep, out)
            if key in self.fn: out = self.fn[key](out)
            if key in self.want: self.rec[key] = out.detach().clone()
            return out
        return f

    def clear(self):
        self.patch, self.fn, self.want, self.rec = {}, {}, set(), {}

    def run(self, x, v, m, grad=False):
        with (contextlib.nullcontext() if grad else self.t.no_grad()):
            L = self.m(x, v, m)
        return L, self.pn._cls['h'].clone()


def T(net, a):
    return net.t.from_numpy(np.ascontiguousarray(a)).to(net.dev)


def mbatches(idx, cnt, size=BATCH):
    idx = np.asarray(idx); o = idx[np.argsort(cnt[idx], kind='stable')]
    return [o[i:i + size] for i in range(0, len(o), size)]


def tens(net, D, model, name, idx, P, vname='base'):
    return T(net, D['x'][model][name][idx][:, :, :P]), T(net, D['v'][vname][idx][:, :, :P]), T(net, D['m'][idx][:, :, :P])


def mpn_of(mb):
    return mb[:, 0, :].T[..., None]                       # (P, n, 1) float


def make_fn(mode, Q, hd, mpn, donor=None):
    """interchange: replace the Q-subspace component of the site's output by the donor activation's"""
    if mode in ('pp', 'tok'):
        return lambda out: out + ((hd - out) @ Q) @ Q.T
    cnt = mpn.sum(0)                                      # (n, 1)
    md = (hd * mpn).sum(0) / cnt
    if donor is not None: md = md[donor]
    return lambda out: out + ((md - (out * mpn).sum(0) / cnt) @ Q) @ Q.T


# ------------------------------------------------------------------------------------------------------------ statistics
def share(Wb, dE, dP, kind):
    v = float(dP.sum() / dE.sum()); bs = (Wb @ dP) / (Wb @ dE)
    if kind == 'den': v, bs = 1 - v, 1 - bs
    return dict(v=v, ci=ci(bs), p0=p_two(bs, 0.0), p_le_half=float(max(1 / len(bs), (bs <= 0.5).mean()))), bs


def slope(Wb, y, x):
    s = float((y * x).sum() / (x * x).sum()); bs = (Wb @ (y * x)) / (Wb @ (x * x)); return s, bs


# ------------------------------------------------------------------------------------------------------------ (a) patching
def patching(net, D, model, idx_all, edits, rng, full, sd97):
    t = net.t; n = len(idx_all); pos = np.full(len(D['y']), -1); pos[idx_all] = np.arange(n)
    KS = [('a', i) for i in range(8)] + [('m', i) for i in range(8)]; KB = [('b', i) for i in range(8)]
    R = dict(base=dict(z=np.zeros(n), L=np.zeros((n, 10))))
    for e in edits: R[e] = {}

    def save(e, var, rows, L, z):
        d = R[e].setdefault(var, dict(z=np.zeros(n), L=np.zeros((n, 10))))
        d['z'][rows] = z[:, N98].double().cpu().numpy(); d['L'][rows] = L.double().cpu().numpy()

    y, mass = D['y'], D['obs']['base']['m']
    for bi, idx in enumerate(mbatches(idx_all, D['cnt'])):
        rows = pos[idx]; P = int(D['cnt'][idx].max())
        xb, vb, mb = tens(net, D, model, 'base', idx, P)
        net.clear(); net.want = set(KS + KB); Lb, zb = net.run(xb, vb, mb); Rc = dict(net.rec)
        R['base']['z'][rows] = zb[:, N98].double().cpu().numpy(); R['base']['L'][rows] = Lb.double().cpu().numpy()
        nb_ = len(idx); donor = np.zeros(nb_, int)
        for j in range(nb_):
            same = np.where((y[idx] == y[idx[j]]) & (np.arange(nb_) != j))[0]
            if not len(same): same = np.where(np.arange(nb_) != j)[0]
            close = same[np.abs(mass[idx[same]] - mass[idx[j]]) < 5.0]
            donor[j] = rng.choice(close) if len(close) else same[np.argmin(np.abs(mass[idx[same]] - mass[idx[j]]))]
        cntb = D['cnt'][idx]
        posidx = T(net, (np.arange(P)[:, None] % np.maximum(cntb[donor], 1)[None, :]).astype(np.int64))
        dt = T(net, donor.astype(np.int64))[None, :].expand(P, nb_)
        pm = T(net, ((rng.random((P, nb_)) < 0.5) & (np.arange(P)[:, None] < cntb[None, :]))[..., None])
        for e in edits:
            xe, _, _ = tens(net, D, model, e, idx, P); ve = T(net, D['v'][e][idx][:, :, :P])
            net.clear(); net.want = set(KS + KB); Le, ze = net.run(xe, ve, mb); Re = dict(net.rec); net.want = set()
            save(e, 'edit', rows, Le, ze)
            for k in KS + KB:
                lab = f'{k[0]}{k[1] + 1}'
                net.patch = {k: (Rc[k], None)}; save(e, 'den_' + lab, rows, *net.run(xe, ve, mb))
                net.patch = {k: (Re[k], None)}; save(e, 'noi_' + lab, rows, *net.run(xb, vb, mb))
                if full and k in KS:
                    net.patch = {k: (Rc[k][posidx, dt], None)}; save(e, 'res_' + lab, rows, *net.run(xe, ve, mb))
                    net.patch = {k: (Rc[k], pm)}; save(e, 'sub_' + lab, rows, *net.run(xe, ve, mb))
            A_ = [('a', i) for i in range(8)]; M_ = [('m', i) for i in range(8)]
            groups = dict(allA=A_, allM=M_, allsub=A_ + M_, **{f'upto{l}': [k for k in A_ + M_ if k[1] < l] for l in range(1, 8)})
            for g, ks in groups.items():
                net.patch = {k: (Rc[k], None) for k in ks}; save(e, 'den_' + g, rows, *net.run(xe, ve, mb))
                if g in ('allA', 'allM', 'allsub'):
                    net.patch = {k: (Re[k], None) for k in ks}; save(e, 'noi_' + g, rows, *net.run(xb, vb, mb))
            net.clear()
        if bi % 10 == 0: log(f'  patching {model}: batch {bi + 1} of {math.ceil(n / BATCH)}')
    # ---- statistics
    yy = y[idx_all]; Wb = boot_W(n, NB, np.random.default_rng(SEED + 1)).astype(np.float64)
    zb, Lb = R['base']['z'], R['base']['L']; wz = lambda L: L[:, iW] - L[:, iZ]
    out = dict(N=n, N_W=int((yy == iW).sum()), N_Z=int((yy == iZ).sum()), edits={})
    base_WZ = float((Lb.argmax(1)[yy == iW] == iZ).mean())
    for e in edits:
        E = R[e]; dzE = (E['edit']['z'] - zb) / sd97; dwE = wz(E['edit']['L']) - wz(Lb)
        rWZ_E = float((E['edit']['L'].argmax(1)[yy == iW] == iZ).mean())
        o = dict(edit=dict(dz97=float(dzE.mean()), dz97_ci=ci((Wb @ dzE) / Wb.sum(1)), dz97_W=float(dzE[yy == iW].mean()),
                           dz97_Z=float(dzE[yy == iZ].mean()), dlWZ=float(dwE.mean()), dlWZ_ci=ci((Wb @ dwE) / Wb.sum(1)),
                           W_to_Z_base=base_WZ, W_to_Z=rWZ_E, changed=float((E['edit']['L'].argmax(1) != Lb.argmax(1)).mean())),
                 variants={})
        for var, d in E.items():
            if var == 'edit': continue
            kind = var[:3]; dz = (d['z'] - zb) / sd97 if var[:3] != 'noi' else (d['z'] - zb) / sd97
            dw = wz(d['L']) - wz(Lb)
            sk = 'noi' if kind == 'noi' else 'den'
            sz, _ = share(Wb, dzE, dz, sk) if model == 'full' else (None, None)
            sw, _ = share(Wb, dwE, dw, sk)
            rWZ = float((d['L'].argmax(1)[yy == iW] == iZ).mean())
            calls = (rWZ_E - rWZ) / (rWZ_E - base_WZ) if kind != 'noi' else (rWZ - base_WZ) / (rWZ_E - base_WZ)
            o['variants'][var] = dict(n98=sz, lWZ=sw, W_to_Z=rWZ, call_share=float(calls) if abs(rWZ_E - base_WZ) > 1e-9 else None)
        out['edits'][e] = o
    # Holm over 16 sublayers x 2 edits (phi 0.06, ang 1.1), denoise share, primary metric (n98 for full, lW - lZ for kin)
    met = 'n98' if model == 'full' else 'lWZ'; keys, ps = [], []
    for e in ('phi0.06', 'ang1.1'):
        if e not in out['edits']: continue
        for var, d in out['edits'][e]['variants'].items():
            if var[:5] in ('den_a', 'den_m') and var[5:].isdigit(): keys.append((e, var)); ps.append(d[met]['p0'])
    for (e, var), ph in zip(keys, holm(ps) if ps else []): out['edits'][e]['variants'][var][met]['p0_holm'] = ph
    out['holm_n_tests'] = len(ps)
    return out


# ------------------------------------------------------------------------------------------------------------ (b) probes
def capture(net, D, model, names, idx_all):
    """masked particle mean at emb, b1..b8 and the cls1 token -> {name: (n, 10, C)}; also z97 and logits"""
    t = net.t; n = len(idx_all); pos = np.full(len(D['y']), -1); pos[idx_all] = np.arange(n)
    F = {nm: np.zeros((n, len(SITES), C), np.float32) for nm in names}; Z = {nm: np.zeros(n) for nm in names}; L = {nm: np.zeros((n, 10)) for nm in names}
    for idx in mbatches(idx_all, D['cnt']):
        rows = pos[idx]; P = int(D['cnt'][idx].max())
        for nm in names:
            x, v, mb = tens(net, D, model, nm, idx, P, nm if nm in D['v'] else 'base'); mp = mpn_of(mb); cntt = mp.sum(0)
            net.clear(); net.want = {SKEY[s] for s in SITES}; Lx, zx = net.run(x, v, mb)
            for si, s in enumerate(SITES):
                h = net.rec[SKEY[s]]
                F[nm][rows, si] = (h[0] if s == 'cls1' else (h * mp).sum(0) / cntt).float().cpu().numpy()
            Z[nm][rows] = zx[:, N98].double().cpu().numpy(); L[nm][rows] = Lx.double().cpu().numpy()
    net.clear()
    return F, Z, L


def probes(F, D, idx_all, fit, val, tr, te, edits):
    o = {}; Wb = boot_W(len(te), NB, np.random.default_rng(SEED + 2)).astype(np.float64)
    ob = {nm: {k: D['obs'][nm][k][idx_all] for k in TARGETS} for nm in ['base'] + edits}
    for si, s in enumerate(SITES):
        o[s] = {}
        for tg in TARGETS:
            X0 = F['base'][:, si].astype(np.float64); rd = Ridge(X0, ob['base'][tg], fit, val, tr); p0 = rd(X0[te])
            e = dict(r2_test=r2(p0, ob['base'][tg][te]), alpha=float(rd.alpha), edits={})
            for nm in edits:
                dp = rd(F[nm][te, si].astype(np.float64)) - p0; dt = ob[nm][tg][te] - ob['base'][tg][te]
                e['edits'][nm] = dict(readout=float(dp.mean()), readout_ci=ci((Wb @ dp) / Wb.sum(1)), true=float(dt.mean()),
                                      ratio_to_true=(float(dp.mean() / dt.mean()) if abs(dt.mean()) > 1e-6 else None))
            if 'phi0.06' in edits and 'angk' in edits:
                dpp = rd(F['phi0.06'][te, si].astype(np.float64)) - p0; dpk = rd(F['angk'][te, si].astype(np.float64)) - p0
                bs = (Wb @ dpp) / (Wb @ dpk)
                e['phi_over_matched'] = dict(v=float(dpp.sum() / dpk.sum()), ci=ci(bs))
            o[s][tg] = e
    return o


# ------------------------------------------------------------------------------------------------------------ (b) DAS
def das_fit(net, D, model, site, mode, k, cfs, train_idx, rng):
    t = net.t; key = SKEY[site]
    g = t.Generator().manual_seed(int(rng.integers(1 << 30)))
    Wp = t.randn(C, k, generator=g).to(net.dev).requires_grad_(True); opt = t.optim.Adam([Wp], lr=0.02); losses = []
    for s in range(STEPS):
        idx = np.sort(rng.choice(train_idx, BS, replace=False)); P = int(D['cnt'][idx].max()); cf = cfs[int(rng.integers(len(cfs)))]
        xb, vb, mb = tens(net, D, model, 'base', idx, P); xd = T(net, D['x'][model][cf][idx][:, :, :P]); mp = mpn_of(mb)
        net.clear(); net.want = {key}; Lcf, _ = net.run(xd, vb, mb); hd = net.rec[key]; net.want = set()
        Q = t.linalg.qr(Wp)[0]; net.fn = {key: make_fn(mode, Q, hd, mp)}
        L, _ = net.run(xb, vb, mb, grad=True); loss = ((L - Lcf) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss))
    net.clear()
    return t.linalg.qr(Wp.detach())[0], float(np.mean(losses[:10])), float(np.mean(losses[-20:]))


def rand_Q(net, k, rng):
    t = net.t; return t.linalg.qr(t.from_numpy(rng.standard_normal((C, k)).astype(np.float32)).to(net.dev))[0]


def das_eval(net, D, model, subs, cfs, eval_idx):
    """subs: list of dict(name, site, mode, Q) -> per (name, cf) per-jet arrays + reference arrays"""
    n = len(eval_idx); pos = np.full(len(D['y']), -1); pos[eval_idx] = np.arange(n)
    ref = {cf: dict(e0=np.zeros(n), zb=np.zeros(n), zcf=np.zeros(n), cb=np.zeros(n, int), ccf=np.zeros(n, int)) for cf in cfs}
    A = {(s['name'], cf): dict(e=np.zeros(n), z=np.zeros(n), c=np.zeros(n, int)) for s in subs for cf in cfs}
    need = {SKEY[s['site']] for s in subs}
    for idx in mbatches(eval_idx, D['cnt']):
        rows = pos[idx]; P = int(D['cnt'][idx].max()); xb, vb, mb = tens(net, D, model, 'base', idx, P); mp = mpn_of(mb)
        net.clear(); Lb, zb = net.run(xb, vb, mb)
        for cf in cfs:
            xd = T(net, D['x'][model][cf][idx][:, :, :P])
            net.clear(); net.want = set(need); Lcf, zcf = net.run(xd, vb, mb); Rd = dict(net.rec); net.want = set()
            r = ref[cf]; r['e0'][rows] = ((Lb - Lcf) ** 2).sum(1).double().cpu().numpy(); r['zb'][rows] = zb[:, N98].double().cpu().numpy()
            r['zcf'][rows] = zcf[:, N98].double().cpu().numpy(); r['cb'][rows] = Lb.argmax(1).cpu().numpy(); r['ccf'][rows] = Lcf.argmax(1).cpu().numpy()
            for s in subs:
                key = SKEY[s['site']]; net.fn = {key: make_fn(s['mode'], s['Q'], Rd[key], mp)}; L, z = net.run(xb, vb, mb)
                a = A[(s['name'], cf)]; a['e'][rows] = ((L - Lcf) ** 2).sum(1).double().cpu().numpy()
                a['z'][rows] = z[:, N98].double().cpu().numpy(); a['c'][rows] = L.argmax(1).cpu().numpy()
            net.fn = {}
    net.clear()
    return ref, A


def das_summary(ref, A, names, cfs, Wb):
    o = {}
    for nm in names:
        o[nm] = {}
        for cf in cfs:
            r = ref[cf]; a = A[(nm, cf)]; ch = r['ccf'] != r['cb']
            bs = 1 - (Wb @ a['e']) / (Wb @ r['e0']); dzc = r['zcf'] - r['zb']; dzi = a['z'] - r['zb']
            o[nm][cf] = dict(rec=float(1 - a['e'].sum() / r['e0'].sum()), rec_ci=ci(bs), call_iia=(float((a['c'][ch] == r['ccf'][ch]).mean()) if ch.any() else None),
                             n_changed=int(ch.sum()), call_same_all=float((a['c'] == r['ccf']).mean()),
                             n98_rec=(float(dzi.sum() / dzc.sum()) if abs(dzc.sum()) > 1e-9 else None))
            o[nm][cf]['_bs'] = bs
    return o


def compare(o, a, b, cf):
    """paired bootstrap of rec(a) - rec(b) (same jets, same resamples)"""
    d = o[a][cf]['_bs'] - o[b][cf]['_bs']
    return dict(diff=float(o[a][cf]['rec'] - o[b][cf]['rec']), ci=ci(d), p=p_two(d, 0.0))


def strip(o):
    if isinstance(o, dict): return {str(k): strip(v) for k, v in o.items() if k != '_bs'}
    if isinstance(o, (list, tuple)): return [strip(v) for v in o]
    if isinstance(o, np.integer): return o.item()
    if isinstance(o, np.floating): o = float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, float) and not math.isfinite(o): return None
    return o


def das_block(net, D, model, plan, cfs_eval, train_idx, eval_idx, rng, tag):
    """plan: list of (var, site, mode, k). Fits, random controls, evaluation -> summary dict"""
    subs, fits = [], {}
    for var, site, mode, k in plan:
        t1 = time.time(); Q, l0, l1 = das_fit(net, D, model, site, mode, k, M2_CF if var == 'M2' else M0_CF, train_idx, rng)
        nm = f'{var}|{site}|{mode}|{k}'; subs.append(dict(name=nm, site=site, mode=mode, Q=Q)); fits[nm] = dict(loss_first10=l0, loss_last20=l1, Q=Q)
        log(f'  DAS {tag} {nm}: loss {l0:.4f} -> {l1:.4f} ({time.time() - t1:.0f} s)')
    rnd = []
    for (site, mode, k) in sorted({(s, m, k) for v, s, m, k in plan if v == 'M2'}):
        for r in range(NRAND): rnd.append(dict(name=f'rand{r}|{site}|{mode}|{k}', site=site, mode=mode, Q=rand_Q(net, k, rng)))
    t1 = time.time(); ref, A = das_eval(net, D, model, subs + rnd, cfs_eval, eval_idx); log(f'  DAS {tag} evaluation: {len(subs) + len(rnd)} subspaces x {len(cfs_eval)} counterfactuals ({time.time() - t1:.0f} s)')
    Wb = boot_W(len(eval_idx), NB, np.random.default_rng(SEED + 3)).astype(np.float64)
    o = das_summary(ref, A, [s['name'] for s in subs + rnd], cfs_eval, Wb)
    # random mean per (site, mode, k): average per-jet errors over draws
    for (site, mode, k) in sorted({(s, m, k) for v, s, m, k in plan if v == 'M2'}):
        nm = f'rand|{site}|{mode}|{k}'; o[nm] = {}
        for cf in cfs_eval:
            e = np.mean([A[(f'rand{r}|{site}|{mode}|{k}', cf)]['e'] for r in range(NRAND)], 0); r_ = ref[cf]
            o[nm][cf] = dict(rec=float(1 - e.sum() / r_['e0'].sum()), rec_ci=ci(1 - (Wb @ e) / (Wb @ r_['e0'])),
                             rec_draws=[o[f'rand{r}|{site}|{mode}|{k}'][cf]['rec'] for r in range(NRAND)])
            o[nm][cf]['_bs'] = 1 - (Wb @ e) / (Wb @ r_['e0'])
    tests = {}
    for var, site, mode, k in plan:
        if var != 'M2': continue
        nm = f'M2|{site}|{mode}|{k}'; tests[nm] = {cf: compare(o, nm, f'rand|{site}|{mode}|{k}', cf) for cf in cfs_eval}
        m0 = f'M0|{site}|{mode}|{k}'
        if m0 in o: tests[nm]['vs_M0_subspace'] = {cf: compare(o, nm, m0, cf) for cf in cfs_eval}
    keys = list(tests); ps = [tests[k_]['phi0.06']['p'] for k_ in keys]
    for k_, ph in zip(keys, holm(ps) if ps else []): tests[k_]['phi0.06']['p_holm'] = ph
    return dict(N_eval=len(eval_idx), steps=STEPS, batch=BS, summary=o, tests_vs_random=tests, holm_n_tests=len(ps),
                fits={k_: dict(loss_first10=v['loss_first10'], loss_last20=v['loss_last20']) for k_, v in fits.items()}), {k_: v['Q'] for k_, v in fits.items()}


# ------------------------------------------------------------------------------------------------------------ (c) substitution
def substitution(st, net, D, Qs, lstar, kstar, sub_idx, rng, sd97):
    t = net.t; site = lstar; key = SKEY[site]; n = len(sub_idx); pos = np.full(len(D['y']), -1); pos[sub_idx] = np.arange(n)
    y = D['y']; mass = D['obs']['base']['m']; lM2 = D['obs']['base']['log_M2']; lM2phi = D['obs']['phi0.06']['log_M2']
    Qd = Qs[f'M2|{site}|jm|{kstar}']; Qm = Qs.get(f'M0|{site}|jm|4'); Qr = [rand_Q(net, kstar, rng) for _ in range(NRAND)]
    Qi = t.eye(C, device=net.dev)
    VARS = ['das'] + [f'rand{r}' for r in range(NRAND)] + (['m0'] if Qm is not None else []) + ['all128']
    QV = dict(das=Qd, m0=Qm, all128=Qi, **{f'rand{r}': Qr[r] for r in range(NRAND)})
    PATHS = [f'b1h{h + 1}' for h in range(H)] + ['b1h3+b1h6', 'block1', 'block2_direct']
    R = {v: dict(z=np.full(n, np.nan), L=np.full((n, 10), np.nan)) for v in VARS + ['base', 'phi']}
    PZ = {f'{w}:{p}': np.full(n, np.nan) for w in ('das', 'phi') for p in PATHS}
    donor_g = np.full(n, -1); recon = 0.0
    for idx in mbatches(sub_idx, D['cnt']):
        rows = pos[idx]; P = int(D['cnt'][idx].max()); xb, vb, mb = tens(net, D, 'full', 'base', idx, P); mp = mpn_of(mb); nb_ = len(idx)
        donor = np.full(nb_, -1)
        for j in range(nb_):
            c = np.where((y[idx] == y[idx[j]]) & (np.abs(mass[idx] - mass[idx[j]]) < 3.0) & (np.abs(lM2[idx] - lM2[idx[j]]) >= 0.15))[0]
            if len(c): donor[j] = rng.choice(c)
        ok = donor >= 0
        if not ok.any(): continue
        donor_g[rows[ok]] = idx[donor[ok]]
        net.clear(); net.want = {key}; Lb, zb = net.run(xb, vb, mb); hb = net.rec[key]; net.want = set()
        Xb = st.cap['X'].permute(1, 0, 2).contiguous().float().clone(); pad = (mb[:, 0] == 0)
        okt = T(net, ok); dn = T(net, np.where(ok, donor, np.arange(nb_)).astype(np.int64))
        zz, Ll, R_ = st.cls(Xb, pad, rec=True); recon = max(recon, float((Ll - Lb).abs().max())); ctx_b = R_['ctx']
        R['base']['z'][rows] = zb[:, N98].double().cpu().numpy(); R['base']['L'][rows] = Lb.double().cpu().numpy()
        Xs = {}
        for v in VARS:
            net.fn = {key: make_fn('jm', QV[v], hb, mp, donor=dn)}; L, z = net.run(xb, vb, mb); net.fn = {}
            R[v]['z'][rows[ok]] = z[ok][:, N98].double().cpu().numpy(); R[v]['L'][rows[ok]] = L[ok].double().cpu().numpy()
            if v == 'das': Xs['das'] = st.cap['X'].permute(1, 0, 2).contiguous().float().clone()
        xe = T(net, D['x']['full']['phi0.06'][idx][:, :, :P]); ve = T(net, D['v']['phi0.06'][idx][:, :, :P])
        L, z = net.run(xe, ve, mb); R['phi']['z'][rows] = z[:, N98].double().cpu().numpy(); R['phi']['L'][rows] = L.double().cpu().numpy()
        Xs['phi'] = st.cap['X'].permute(1, 0, 2).contiguous().float().clone()
        with t.no_grad():
            for w, Xw in Xs.items():
                _, _, Rw = st.cls(Xw, pad, rec=True); cs = Rw['ctx']
                for p in PATHS:
                    if p.startswith('b1h') and '+' not in p: sel = [int(p[3:]) - 1]
                    elif p == 'b1h3+b1h6': sel = [2, 5]
                    else: sel = list(range(H))
                    s_ = t.tensor([h in sel for h in range(H)], device=net.dev)
                    if p == 'block2_direct': zp, _, _ = st.cls(Xw, pad, {('ctx', 0): (ctx_b[0], s_)})
                    else: zp, _, _ = st.cls(Xb, pad, {('ctx', 0): (cs[0], s_)})
                    keep = rows[ok] if w == 'das' else rows; msk = ok if w == 'das' else np.ones(nb_, bool)
                    PZ[f'{w}:{p}'][keep] = zp[t.from_numpy(msk).to(net.dev)][:, N98].double().cpu().numpy()
    net.clear()
    has = donor_g >= 0; rid = sub_idx[has]; did = donor_g[has]; nn = int(has.sum())
    dl = lM2[did] - lM2[rid]; dm = mass[did] - mass[rid]; yy = y[rid]
    Wb = boot_W(nn, NB, np.random.default_rng(SEED + 4)).astype(np.float64)
    zb = R['base']['z'][has]; Lb = R['base']['L'][has]; wz = lambda L: L[:, iW] - L[:, iZ]
    out = dict(site=site, k=kstar, N_recipients=nn, N_candidates=n, stage_reimpl_max_dlogit=recon,
               donor=dict(dlogM2_mean_abs=float(np.abs(dl).mean()), dlogM2_pos_share=float((dl > 0).mean()), dm_mean_abs=float(np.abs(dm).mean())),
               variants={})
    dlphi = (lM2phi - lM2)[rid]; dzphi = (R['phi']['z'][has] - zb) / sd97; dwphi = wz(R['phi']['L'][has]) - wz(Lb)
    sphi, bphi = slope(Wb, dzphi, dlphi); swphi, bwphi = slope(Wb, dwphi, dlphi)
    out['phi0.06'] = dict(slope_n98_per_dlogM2=sphi, ci=ci(bphi), slope_lWZ_per_dlogM2=swphi, ci_lWZ=ci(bwphi), dlogM2_mean=float(dlphi.mean()))
    BS_ = {}
    pos_ = dl > 0
    for v in VARS:
        dz = (R[v]['z'][has] - zb) / sd97; dw = wz(R[v]['L'][has]) - wz(Lb); s, b = slope(Wb, dz, dl); sw, bw = slope(Wb, dw, dl); BS_[v] = (b, bw)
        c0 = Lb.argmax(1); c1 = R[v]['L'][has].argmax(1); Wp = (yy == iW) & pos_
        out['variants'][v] = dict(slope_n98_per_dlogM2=s, ci=ci(b), slope_lWZ_per_dlogM2=sw, ci_lWZ=ci(bw), ratio_to_phi=s / sphi, ratio_to_phi_ci=ci(b / bphi),
                                  W_to_Z_base_dM2pos=float((c0[Wp] == iZ).mean()), W_to_Z_dM2pos=float((c1[Wp] == iZ).mean()), N_W_dM2pos=int(Wp.sum()),
                                  mcnemar_dM2pos=mcnemar(c0[pos_] == yy[pos_], c1[pos_] == yy[pos_]),
                                  mcnemar_dM2neg=mcnemar(c0[~pos_] == yy[~pos_], c1[~pos_] == yy[~pos_]),
                                  changed=float((c0 != c1).mean()))
    rb = np.mean([BS_[f'rand{r}'][0] for r in range(NRAND)], 0)
    out['rand_mean'] = dict(slope_n98_per_dlogM2=float(np.mean([out['variants'][f'rand{r}']['slope_n98_per_dlogM2'] for r in range(NRAND)])), ci=ci(rb))
    d = BS_['das'][0] - rb; out['das_vs_rand'] = dict(diff=float(out['variants']['das']['slope_n98_per_dlogM2'] - out['rand_mean']['slope_n98_per_dlogM2']), ci=ci(d), p=p_two(d, 0.0))
    if 'm0' in BS_:
        d = BS_['das'][0] - BS_['m0'][0]; out['das_vs_m0'] = dict(ci=ci(d), p=p_two(d, 0.0))
    # path patching shares
    sd_all, bd_all = slope(Wb, (R['das']['z'][has] - zb) / sd97, dl)
    out['paths'] = {}
    for p in PATHS:
        dzp = (PZ[f'das:{p}'][has] - zb) / sd97; sp, bp = slope(Wb, dzp, dl)
        dzq = (PZ[f'phi:{p}'][has] - zb) / sd97; mq = (Wb @ dzq) / (Wb @ dzphi)
        out['paths'][p] = dict(substitution_share=sp / sd_all, substitution_share_ci=ci(bp / bd_all), phi_share=float(dzq.sum() / dzphi.sum()), phi_share_ci=ci(mq))
    return out


def ten_class(net, D10, Qd, site, kstar, rng):
    """T10: phi 0.06 full edit vs the self-donor interchange through Qd (jm) vs random Q: calls, confusion, accuracy"""
    key = SKEY[site]; idx_all = np.arange(len(D10['y'])); n = len(idx_all); y = D10['y']
    Qr = rand_Q(net, kstar, rng); C_ = {v: np.zeros(n, int) for v in ('base', 'edit', 'das', 'rand')}
    for idx in mbatches(idx_all, D10['cnt']):
        P = int(D10['cnt'][idx].max()); xb, vb, mb = tens(net, D10, 'full', 'base', idx, P); mp = mpn_of(mb)
        xe = T(net, D10['x']['full']['phi0.06'][idx][:, :, :P]); ve = T(net, D10['v']['phi0.06'][idx][:, :, :P])
        net.clear(); L, _ = net.run(xb, vb, mb); C_['base'][idx] = L.argmax(1).cpu().numpy()
        net.want = {key}; L, _ = net.run(xe, ve, mb); hd = net.rec[key]; net.want = set(); C_['edit'][idx] = L.argmax(1).cpu().numpy()
        for v, Q in (('das', Qd), ('rand', Qr)):
            net.fn = {key: make_fn('jm', Q, hd, mp)}; L, _ = net.run(xb, vb, mb); net.fn = {}; C_[v][idx] = L.argmax(1).cpu().numpy()
    net.clear()
    conf = lambda c: [[int(((y == a) & (c == b)).sum()) for b in range(10)] for a in range(10)]
    ch = C_['edit'] != C_['base']
    return dict(N=n, acc={v: float((c == y).mean()) for v, c in C_.items()}, confusion={v: conf(c) for v, c in C_.items()},
                W_to_Z={v: float((c[y == iW] == iZ).mean()) for v, c in C_.items()},
                edit_changed=float(ch.mean()), das_matches_edit_on_changed=float((C_['das'][ch] == C_['edit'][ch]).mean()) if ch.any() else None,
                rand_matches_edit_on_changed=float((C_['rand'][ch] == C_['edit'][ch]).mean()) if ch.any() else None,
                das_changed=float((C_['das'] != C_['base']).mean()), rand_changed=float((C_['rand'] != C_['base']).mean()),
                mcnemar_das=mcnemar(C_['base'] == y, C_['das'] == y), mcnemar_edit=mcnemar(C_['base'] == y, C_['edit'] == y))


# ------------------------------------------------------------------------------------------------------------ main
def main():
    import torch
    OUT.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(SEED)
    res = dict(node='P8', seed=SEED, smoke=SMOKE, sizes=dict(PER_WZ=PER_WZ, PER_10=PER_10, N_PATCH=N_PATCH, N_PATCH_KIN=N_PATCH_KIN, NB=NB, STEPS=STEPS,
               BS=BS, N_EVAL=N_EVAL, N_SUB=N_SUB, NRAND=NRAND), files={})
    # ---- samples
    JW, rW = load_rows(TEST / 'WToQQ_101.root', PER_WZ, rng); JZ, rZ = load_rows(TEST / 'ZToQQ_101.root', PER_WZ, rng)
    J0 = cat([JW, JZ]); res['files']['WZ'] = ['WToQQ_101.root', 'ZToQQ_101.root']; res['rows_head'] = dict(W=rW[:5].tolist(), Z=rZ[:5].tolist())
    log(f'WZ sample: {len(J0["label"])} jets, labels {np.bincount(J0["label"].astype(int), minlength=10).tolist()}')
    names = ['phi0.03', 'phi0.06', 'phi0.09', 'eta0.06', 'ang1.1', 'pt0.8', 'pt1.25', 'angk']
    D = prepare(J0, names, ['full', 'kin'], ['phi0.06', 'eta0.06', 'ang1.1', 'angk', 'pt1.25']); del J0, JW, JZ
    N = len(D['y']); y = D['y']
    res['k_match'] = D['k_match']
    res['truth'] = {nm: dict(dlog_m=float(np.mean(np.log(D['obs'][nm]['m'] / D['obs']['base']['m']))), dlog_M2=float(np.mean(D['obs'][nm]['log_M2'] - D['obs']['base']['log_M2'])),
                             half_dlog_M2_W=float(np.mean(0.5 * (D['obs'][nm]['log_M2'] - D['obs']['base']['log_M2'])[y == iW])),
                             dlog_mom=float(np.mean(D['obs'][nm]['log_mom'] - D['obs']['base']['log_mom'])))
                    for nm in names}
    for nm, tr_ in res['truth'].items(): log(f'edit {nm}: mean dlog m {tr_["dlog_m"]:+.4f}, dlog M2 {tr_["dlog_M2"]:+.4f}, dlog mom {tr_["dlog_mom"]:+.4f}')
    log(f'matched real-mass edit: angles x {D["k_match"]:.4f}')
    perm = np.random.default_rng(SEED).permutation(N); tr, te = np.sort(perm[:N // 2]), np.sort(perm[N // 2:])
    fit, val = tr[: len(tr) * 3 // 4], tr[len(tr) * 3 // 4:]
    # ---- models
    st = Stage('full'); net = Net(st.net); sd = {}
    # unedited accuracy: P1 sample and WZ sample
    P1 = sample(30 if SMOKE else 1000, 3000); pt_, de_, dp_, ms_ = kin(P1); P1b = build(P1, pt_, de_, dp_, ms_)
    acc = {}
    for nmod in ('full', 'kin'):
        pn = st.net if nmod == 'full' else ParTNetwork('kin'); r_ = pn.run(P1b)
        acc[f'P1_{nmod}'] = float((r_['logits'].argmax(1) == P1b['label'].astype(int)).mean())
        if nmod == 'kin': kin_pn = pn
    del P1, P1b
    log(f'unedited accuracy on the P1 sample (10 classes): full {100 * acc["P1_full"]:.2f} %, kin {100 * acc["P1_kin"]:.2f} % (P1: 85.9 / 74.5 %)')
    allidx = np.arange(N)
    F, Zc, Lc = capture(net, D, 'full', ['base', 'phi0.06', 'eta0.06', 'ang1.1', 'angk', 'pt1.25'], allidx)
    sd97 = float(Zc['base'].std()); sd['full'] = sd97
    acc['WZ_full_10class'] = float((Lc['base'].argmax(1) == y).mean())
    acc['WZ_full_WZbinary'] = float((np.where(Lc['base'][:, iW] > Lc['base'][:, iZ], iW, iZ) == y).mean())
    acc['WZ_full_W_to_Z'] = float((Lc['base'].argmax(1)[y == iW] == iZ).mean())
    for nm in ('phi0.06', 'eta0.06', 'ang1.1', 'angk', 'pt1.25'):
        acc[f'WZ_full_W_to_Z_{nm}'] = float((Lc[nm].argmax(1)[y == iW] == iZ).mean())
        acc[f'WZ_full_dz97_{nm}'] = float(((Zc[nm] - Zc['base']) / sd97).mean())
    res['accuracy'] = acc; res['n98_sd'] = sd97
    log(f'WZ sample, ParT_full: 10-class acc {100 * acc["WZ_full_10class"]:.2f} %, W/Z binary {100 * acc["WZ_full_WZbinary"]:.2f} %, W->Z {acc["WZ_full_W_to_Z"]:.3f}; '
        + ', '.join(f'{nm}: W->Z {acc[f"WZ_full_W_to_Z_{nm}"]:.3f} dz97 {acc[f"WZ_full_dz97_{nm}"]:+.3f}' for nm in ('phi0.06', 'eta0.06', 'ang1.1', 'angk', 'pt1.25')))
    with open(FN, 'w') as f: json.dump(strip(res), f, indent=1)
    # ---- (b) probes
    t1 = time.time(); res['probes_full'] = probes({k: v for k, v in F.items()}, D, allidx, fit, val, tr, te, ['phi0.06', 'eta0.06', 'ang1.1', 'angk', 'pt1.25'])
    for s in SITES:
        p = res['probes_full'][s]
        log(f'  probe {s:5s}: R2 ' + ', '.join(f'{tg} {p[tg]["r2_test"]:.3f}' for tg in TARGETS) + ' | phi/matched ' + ', '.join(f'{tg} {p[tg]["phi_over_matched"]["v"]:.2f}' for tg in TARGETS))
    log(f'(b) probes done ({time.time() - t1:.0f} s)'); del F
    with open(FN, 'w') as f: json.dump(strip(res), f, indent=1)
    # ---- (a) patching
    t1 = time.time(); pidx = np.sort(rng.choice(N, min(N_PATCH, N), replace=False))
    res['patching_full'] = patching(net, D, 'full', pidx, ['phi0.06', 'ang1.1', 'angk'], rng, True, sd97)
    for e, o in res['patching_full']['edits'].items():
        log(f'  patching full {e}: edit dz97 {o["edit"]["dz97"]:+.3f}, W->Z {o["edit"]["W_to_Z_base"]:.3f} -> {o["edit"]["W_to_Z"]:.3f}')
        for l in range(1, 9):
            v = o['variants']; g = lambda nm: v[nm]['n98']['v']
            log(f'    block {l}: denoise attn {g(f"den_a{l}"):+.3f} mlp {g(f"den_m{l}"):+.3f} stream {g(f"den_b{l}"):+.3f} | noise attn {g(f"noi_a{l}"):+.3f} '
                f'mlp {g(f"noi_m{l}"):+.3f} stream {g(f"noi_b{l}"):+.3f} | resample attn {g(f"res_a{l}"):+.3f} mlp {g(f"res_m{l}"):+.3f} | subset attn {g(f"sub_a{l}"):+.3f} mlp {g(f"sub_m{l}"):+.3f}')
    log(f'(a) patching full done ({time.time() - t1:.0f} s)')
    with open(FN, 'w') as f: json.dump(strip(res), f, indent=1)
    # ---- (b) DAS on ParT_full
    t1 = time.time(); ev = np.sort(rng.choice(te, min(N_EVAL, len(te)), replace=False)); CFE = ['phi0.06', 'eta0.06', 'angk', 'pt1.25']
    plan = [('M2', s, 'tok' if s == 'cls1' else 'jm', k) for s in SITES for k in DIMS] + \
           [('M2', s, 'pp', k) for s in SITES if s != 'cls1' for k in (1, 4)] + \
           [('M0', s, 'tok' if s == 'cls1' else 'jm', 4) for s in SITES]
    res['das_full'], Qs = das_block(net, D, 'full', plan, CFE, tr, ev, rng, 'full')
    log(f'(b) DAS full done ({time.time() - t1:.0f} s)')
    S_ = res['das_full']['summary']
    for s in SITES:
        mode = 'tok' if s == 'cls1' else 'jm'
        log(f'  {s:5s} {mode}: ' + ' | '.join(f'k{k} M2 phi {S_[f"M2|{s}|{mode}|{k}"]["phi0.06"]["rec"]:.3f} eta {S_[f"M2|{s}|{mode}|{k}"]["eta0.06"]["rec"]:.3f} '
                                             f'angk {S_[f"M2|{s}|{mode}|{k}"]["angk"]["rec"]:.3f} pt {S_[f"M2|{s}|{mode}|{k}"]["pt1.25"]["rec"]:.3f} rand {S_[f"rand|{s}|{mode}|{k}"]["phi0.06"]["rec"]:.3f} '
                                             f'M0(k4) on phi {S_[f"M0|{s}|{mode}|4"]["phi0.06"]["rec"]:.3f}' for k in DIMS))
        if s != 'cls1':
            log(f'  {s:5s} pp : ' + ' | '.join(f'k{k} phi {S_[f"M2|{s}|pp|{k}"]["phi0.06"]["rec"]:.3f} eta {S_[f"M2|{s}|pp|{k}"]["eta0.06"]["rec"]:.3f} rand {S_[f"rand|{s}|pp|{k}"]["phi0.06"]["rec"]:.3f}' for k in (1, 4)))
    with open(FN, 'w') as f: json.dump(strip(res), f, indent=1)
    # ---- choose l*, k*
    blocks = [f'b{l}' for l in range(1, 9)]; rec4 = {s: S_[f'M2|{s}|jm|4']['phi0.06']['rec'] for s in blocks}
    best = max(rec4.values()); lstar = next(s for s in blocks if rec4[s] >= 0.8 * best)
    kstar = next(k for k in DIMS if S_[f'M2|{lstar}|jm|{k}']['phi0.06']['rec'] >= 0.9 * rec4[lstar])
    res['choice'] = dict(rule='earliest particle block with jm M2 k=4 phi0.06 recovery >= 0.8 x best block; smallest k with >= 0.9 x k=4 there',
                         rec_k4=rec4, lstar=lstar, kstar=kstar)
    log(f'chosen site {lstar}, k = {kstar} (k=4 recovery by block: ' + ', '.join(f'{s} {v:.3f}' for s, v in rec4.items()) + ')')
    # ---- (c) substitution + path patching
    t1 = time.time(); sidx = np.sort(rng.choice(te, min(N_SUB, len(te)), replace=False))
    res['substitution'] = substitution(st, net, D, Qs, lstar, kstar, sidx, rng, sd97)
    sb = res['substitution']
    log(f'(c) substitution at {lstar} k={kstar}: {sb["N_recipients"]} recipients; phi0.06 slope dz97/dlogM2 {sb["phi0.06"]["slope_n98_per_dlogM2"]:+.3f} '
        + ', '.join(f'{v} {d["slope_n98_per_dlogM2"]:+.3f} (ratio {d["ratio_to_phi"]:.2f}, W->Z {d["W_to_Z_base_dM2pos"]:.3f}->{d["W_to_Z_dM2pos"]:.3f})' for v, d in sb['variants'].items())
        + f' ({time.time() - t1:.0f} s)')
    log('  paths: ' + ', '.join(f'{p} sub {d["substitution_share"]:.3f} phi {d["phi_share"]:.3f}' for p, d in sb['paths'].items()))
    with open(FN, 'w') as f: json.dump(strip(res), f, indent=1)
    # ---- 10-class read
    t1 = time.time(); rng10 = np.random.default_rng(SEED + 10)
    J10 = cat([load_rows(TEST / f'{PRE[c]}_100.root', PER_10, rng10, maxrow=50000)[0] for c in CL]); D10 = prepare(J10, ['phi0.06'], ['full'], ['phi0.06']); del J10
    res['ten_class'] = ten_class(net, D10, Qs[f'M2|{lstar}|jm|{kstar}'], lstar, kstar, rng); tc = res['ten_class']
    log(f'10-class read (T10, {tc["N"]} jets): acc ' + ', '.join(f'{v} {100 * a:.2f}' for v, a in tc['acc'].items()) + f' %; edit changed {100 * tc["edit_changed"]:.2f} %, '
        f'DAS matches edit on changed {tc["das_matches_edit_on_changed"]}, random {tc["rand_matches_edit_on_changed"]} ({time.time() - t1:.0f} s)')
    del D10
    with open(FN, 'w') as f: json.dump(strip(res), f, indent=1)
    # ---- untrained ParT_full: DAS M2 (jm, tok; k 1, 4)
    t1 = time.time(); un = Net(ParTNetwork('full', untrained=True, seed=SEED))
    plan_u = [('M2', s, 'tok' if s == 'cls1' else 'jm', 4) for s in SITES]
    res['das_untrained'], _ = das_block(un, D, 'full', plan_u, ['phi0.06', 'eta0.06'], tr, ev, rng, 'untrained')
    Su = res['das_untrained']['summary']; cmp = {}
    for s in SITES:
        mode = 'tok' if s == 'cls1' else 'jm'
        for k in (4,):
            a, b = S_[f'M2|{s}|{mode}|{k}']['phi0.06'], Su[f'M2|{s}|{mode}|{k}']['phi0.06']
            d = a['_bs'] - b['_bs']; cmp[f'{s}|{mode}|{k}'] = dict(trained=a['rec'], untrained=b['rec'], diff_ci=ci(d), p=p_two(d, 0.0))
    res['trained_vs_untrained'] = cmp
    md_ = lambda s: 'tok' if s == 'cls1' else 'jm'
    log('  untrained vs trained (phi0.06 rec, k=4): ' + ', '.join('%s %.3f/%.3f' % (s, cmp[s + '|' + md_(s) + '|4']['untrained'], cmp[s + '|' + md_(s) + '|4']['trained']) for s in SITES)
        + f' ({time.time() - t1:.0f} s)')
    del un; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    with open(FN, 'w') as f: json.dump(strip(res), f, indent=1)
    # ---- ParT_kin replication
    t1 = time.time(); kn = Net(kin_pn)
    kidx = np.sort(rng.choice(N, min(N_PATCH_KIN, N), replace=False))
    res['patching_kin'] = patching(kn, D, 'kin', kidx, ['phi0.06', 'ang1.1'], rng, False, 1.0)
    for e, o in res['patching_kin']['edits'].items():
        v = o['variants']
        log(f'  patching kin {e}: dlWZ {o["edit"]["dlWZ"]:+.3f}; denoise attn/mlp/stream by block: ' + ' | '.join(
            f'{l}: {v[f"den_a{l}"]["lWZ"]["v"]:+.2f}/{v[f"den_m{l}"]["lWZ"]["v"]:+.2f}/{v[f"den_b{l}"]["lWZ"]["v"]:+.2f}' for l in range(1, 9)))
    plan_k = [('M2', s, 'tok' if s == 'cls1' else 'jm', 4) for s in SITES]
    res['das_kin'], _ = das_block(kn, D, 'kin', plan_k, ['phi0.06', 'eta0.06', 'ang1.1', 'pt1.25'], tr, ev, rng, 'kin')
    Sk = res['das_kin']['summary']
    log('  kin DAS M2 k=4 phi0.06 rec (vs random): ' + ', '.join('%s %.3f (%.3f)' % (s, Sk['M2|' + s + '|' + md_(s) + '|4']['phi0.06']['rec'], Sk['rand|' + s + '|' + md_(s) + '|4']['phi0.06']['rec']) for s in SITES)
        + f' ({time.time() - t1:.0f} s)')
    res['elapsed_s'] = time.time() - T0
    with open(FN, 'w') as f: json.dump(strip(res), f, indent=1)
    log(f'wrote {FN}')
    try: plot(res)
    except Exception as ex: log(f'plot failed: {ex!r}')
    log('P8 DONE')


# ------------------------------------------------------------------------------------------------------------ figure
def plot(res=None):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    if res is None: res = json.load(open(FN))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), dpi=150)
    V = res['patching_full']['edits']['phi0.06']['variants']; xs = np.arange(1, 9)
    nmf = lambda key, l: (f'den_upto{l}' if l < 8 else 'den_allsub') if key == 'den_upto' else f'{key}{l}'
    for key, lab, col, mk in (('den_a', 'attention output of block l only', '#2a6fb0', 'o'), ('den_m', 'MLP output of block l only', '#d2691e', 's'),
                              ('den_upto', 'all attention + MLP outputs of blocks 1..l', '#333333', '^')):
        v = np.array([V[nmf(key, l)]['n98']['v'] for l in xs]); lo = np.array([V[nmf(key, l)]['n98']['ci'][0] for l in xs]); hi = np.array([V[nmf(key, l)]['n98']['ci'][1] for l in xs])
        ax[0].errorbar(xs, v, yerr=[v - lo, hi - v], color=col, marker=mk, capsize=3, label=lab)
    v = np.array([V[f'res_a{l}']['n98']['v'] for l in xs]); ax[0].plot(xs, v, color='#2a6fb0', ls=':', marker='o', mfc='none', label='attention, donor-jet control')
    ax[0].axhline(0, color='k', lw=0.6); ax[0].set_xlabel('particle block'); ax[0].set_ylabel('share of n98 change restored')
    ax[0].set_title('Patch unedited sublayer outputs into the φ + 0.06 rad run', fontsize=10); ax[0].legend(fontsize=8, frameon=False)
    S = res['das_full']['summary']; Su = res['das_untrained']['summary']; xs2 = np.arange(len(SITES)); md = lambda s: 'tok' if s == 'cls1' else 'jm'
    for nm, lab, col, src, mk in (('M2|{s}|{m}|1', '1-dim subspace (trained on φ shifts)', '#2a6fb0', S, 'o'), ('M2|{s}|{m}|4', '4-dim subspace', '#174a7c', S, 's'),
                                  ('rand|{s}|{m}|4', 'random 4-dim subspace', '#999999', S, 'x'), ('M0|{s}|{m}|4', '4-dim subspace trained on pT scaling', '#d2691e', S, 'v'),
                                  ('M2|{s}|{m}|4', '4-dim, untrained network', '#7a9a3a', Su, 'D')):
        v = np.array([src[nm.format(s=s, m=md(s))]['phi0.06']['rec'] for s in SITES])
        ax[1].plot(xs2, v, color=col, marker=mk, label=lab)
    v = np.array([S[f'M2|{s}|{md(s)}|4']['eta0.06']['rec'] for s in SITES]); ax[1].plot(xs2, v, color='#174a7c', ls='--', marker='s', mfc='none', label='4-dim, tested on η + 0.06 (not trained on)')
    ax[1].set_xticks(xs2); ax[1].set_xticklabels(['embed'] + [f'block {l}' for l in range(1, 9)] + ['class blk 1'], rotation=40, fontsize=8)
    ax[1].axhline(0, color='k', lw=0.6); ax[1].set_ylabel('share of the φ + 0.06 logit change reproduced'); ax[1].set_ylim(-0.05, 1.08)
    ax[1].set_title('Write only a k-dim jet-mean subspace from the shifted jet', fontsize=10); ax[1].legend(fontsize=7.5, frameon=False, ncol=2, loc='upper left')
    fig.suptitle(f'ParT_full ({res["patching_full"]["N"]:,} / {res["das_full"]["N_eval"]:,} W and Z test jets): the response to a φ shift at fixed invariant mass builds up over all '
                 'blocks, and one jet-mean direction carries most of it at every depth', fontsize=10.5)
    fig.tight_layout(); fn = OUT / ('p8_mass_formation_smoke.png' if SMOKE else 'p8_mass_formation.png'); fig.savefig(fn, bbox_inches='tight'); log(f'wrote {fn}')


if __name__ == '__main__':
    if '--plot' in sys.argv: plot()
    else: main()
