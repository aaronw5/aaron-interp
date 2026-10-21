"""X15 (claims A1, A3, A10; exploratory): does a Linformer learn pT-rank windows like JEDI-linear's zero-bit masks?

What E is: in a Linformer block (train/models.py, linf-2b-d32-k8) the keys K_n and values V_n of the N = 64 pT-sorted particle
tokens are summarised into k = 8 "slots" by one learned N x k matrix E, k_j = sum_n E[n, j] K_n (the same E for keys and values
and both heads; padded tokens are zeroed before the sum). Row E[n] is therefore a learned weighting of the particle of pT rank n
(rank 0 = hardest) in each of the 8 summaries: a direct rank weighting that can be read from the weights.

Networks (existing checkpoints; no training)
  L-st-s0/s1   linf-2b-d32-k8-n64 seeds 0, 1 (inputs about the stored jet axis)
  L-ax-s0/s1   linf-2b-d32-k8-n64-axhard seeds 0, 1 (inputs about the hardest particle, train_small.recentre_hardest)
  D-s0/s1      dsets-d64-n64 seeds 0, 1 (Deep Sets: a sum over particles, so the output cannot depend on rank labels)
  U0..U4       untrained Linformers (default PyTorch init under torch.manual_seed(5000 + k); E ~ N(0, 1/64))
  JEDI         official JEDI-linear 3-feature n64 (interp.jedi_core.Jedi; jax env, --mode jedi), reference edge 13|14
  E_init       the trained Linformers' own initial E (ckpt/epoch_0.pt), to separate learned change from init noise.
Sample: interp.jedi_core.sample(per_class=1000, seed=1501) = 5,000 test jets; particles pT-sorted (stable); each network gets
  its own input convention ((x - shift) / scale from arch.json, padded rows 0).
Step 1 (weights): edge score at boundary i|i+1 (block b): d_b(i) = ||E_b[i+1] - E_b[i]||_2 over the 8 slots, divided by its
  median over the candidate boundaries; score(i) = mean over the 2 blocks.  Candidate boundaries C: 1 <= i <= 62 with >= 500
  jets having a real particle at rank i + 1 (rank 0 excluded so the hardest-axis recentring is never touched).  Top-5 per
  network; recurrence = overlap of top-5 sets (exact and within +-1 rank) vs the chance overlap of random 5-subsets of C
  (hypergeometric p; 20,000 random subsets for +-1); Spearman of the full profiles with a rank-permutation p.
Step 2 (causal, angle swap): at boundary i|i+1 the particles of rank i and i+1 exchange (Delta eta, Delta phi) and keep their pT.
  The pT ordering, the pT multiset and the set of occupied positions are unchanged; only which angle sits at which rank changes.
  Applied only to jets with >= i + 2 real particles (others excluded from that pair's mean).  Scan of all boundaries in C.
  Flat pairs: from boundaries with score <= the median and >= 2 ranks from any top-5 edge, one per edge, minimising
  |i - j| / 4 + |log(gap_i / gap_j)| + |log(dR_i / dR_j)| (gap = median relative pT gap (pT_i - pT_i+1) / pT_i, dR = median
  angular distance of the two particles; both from the sample).  Metric per pair: mean over affected jets of
  mean_c |Delta logit_c| (5 classes) and % calls changed.  Ratio = mean over edges / mean over flats; 95 % CI from 1,000
  jet-bootstrap resamples; exact permutation test over the 252 edge / flat labellings of the 10 pairs; per-edge paired test vs
  its flat (bootstrap p for |Delta logit|, exact McNemar for calls), Holm over the 5 edges.
  Also: Spearman over all boundaries of the swap effect vs the edge score (raw and partial given rank: residuals of both after
  a quadratic in rank), permutation p (2,000).
Step 3 (window shuffle): for each top-5 edge e, 'across' block = ranks e-2..e+3 (3 on each side of the edge), 'inside' block =
  the nearest 6 consecutive ranks with none of the 5 top edges between them; the angles inside the block are permuted by the
  same random derangement per jet (rng 150100 + e) in both, pT kept per rank.  6 particles moved in both.  Ratio across / inside
  with bootstrap CI, per-edge paired bootstrap p, Holm.
Controls: untrained Linformers with their own top-5 edges (same procedure), Deep Sets at L-st-s0's pairs and blocks, trained
  Linformers at L-st-s0's pairs (transfer).  JEDI-linear: edge 13|14 (end of S0 = sum pT of ranks 0-13; start of S19 = ranks
  13-63) vs flats 15|16, 19|20, 20|21, 22|23, 23|24 (boundaries of no S or P zero-bit window in E20).
Run:
  AI_PY=conda ./job.sh x15-run 1 2 'python -m interp.x15_linformer_ranks --mode small'   -> $AI_RESULTS/jedi/x15_small.json
  ./job.sh x15-jedi 'python -m interp.x15_linformer_ranks --mode jedi'                 -> $AI_RESULTS/jedi/x15_jedi.json
  ./job.sh x15-comb 'python -m interp.x15_linformer_ranks --mode combine'              -> x15_summary.json, x15_linformer_ranks.png
"""
import argparse, itertools, json, math, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
SMALL = Path(os.environ.get('AI_SMALL_MODELS', '/j-jepa-vol/aaron-interp/models/small'))
PER, SEED = 1000, 1501; B = 1000; NTOP = 5; MINJ = 500; BLK = 6
JEDI_EDGE = 13; JEDI_FLATS = [15, 19, 20, 22, 23]
T0 = time.time()


def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, np.ndarray): return clean(o.tolist())
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    return o


# ================================================================================================ data and edits
def load():
    from interp.jedi_core import sample
    x, y, _ = sample(per_class=PER, seed=SEED)
    x = np.asarray(x, np.float32)
    o = np.argsort(-x[..., 0], axis=1, kind='stable'); xs = np.take_along_axis(x, o[..., None], 1)
    return xs, y, bool(np.array_equal(xs, x))


def sample_idx():
    from jetdistill.config import RESULTS
    yy = np.asarray(np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r')['y'])
    rng = np.random.default_rng(SEED)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(yy == c), PER, replace=False) for c in range(5)])), len(yy)


def recentre_np(x):
    x = np.asarray(x, np.float32); d = x[..., 1:3] - x[:, :1, 1:3]
    dp = np.remainder(d[..., 1] + np.float32(np.pi), np.float32(2 * np.pi)) - np.float32(np.pi)
    d = np.stack([d[..., 0], dp], -1) * (x[..., :1] > 0).astype(np.float32)
    return np.concatenate([x[..., :1], d], -1).astype(np.float32)


def nreal(x): return (x[..., 0] > 0).sum(1)


def candidates(x):
    nr = nreal(x)
    return [i for i in range(1, 63) if (nr >= i + 2).sum() >= MINJ]


def swap(x, i):
    nr = nreal(x); el = nr >= i + 2; e = x.copy()
    e[el, i, 1:3] = x[el, i + 1, 1:3]; e[el, i + 1, 1:3] = x[el, i, 1:3]
    return e, el


def derangements(J, m, seed):
    rng = np.random.default_rng(seed); P = np.empty((J, m), int); idx = np.arange(m)
    for j in range(J):
        while True:
            p = rng.permutation(m)
            if not np.any(p == idx): break
        P[j] = p
    return P


def shuffle_block(x, start, P):
    nr = nreal(x); el = nr >= start + BLK; e = x.copy(); blk = np.arange(start, start + BLK)
    src = blk[P]                                              # (J, m) source ranks
    ang = np.take_along_axis(x[..., 1:3], src[..., None], 1)
    for k in range(BLK):
        e[el, start + k, 1:3] = ang[el, k]
    return e, el


def pair_features(x, C):
    nr = nreal(x); F = {}
    for i in C:
        el = nr >= i + 2; p0, p1 = x[el, i, 0].astype(np.float64), x[el, i + 1, 0].astype(np.float64)
        gap = np.median((p0 - p1) / np.maximum(p0, 1e-9))
        dR = np.median(np.hypot(x[el, i, 1] - x[el, i + 1, 1], x[el, i, 2] - x[el, i + 1, 2]))
        F[i] = dict(gap=float(max(gap, 1e-4)), dR=float(max(dR, 1e-5)), n=int(el.sum()))
    return F


# ================================================================================================ statistics
def boot_w(J, seed=SEED):
    return np.random.default_rng(seed).multinomial(J, np.full(J, 1.0 / J), size=B).astype(np.float64)


def pair_means(D, EL, W=None):
    """D, EL: (P, J) -> per-pair mean over eligible jets; with W (B, J): (B, P)"""
    if W is None: return (D * EL).sum(1) / np.maximum(EL.sum(1), 1)
    return (W @ (D * EL).T) / np.maximum(W @ EL.T, 1)


def ci(a): return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); out = np.empty(m); run = 0
    for r, k in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[k])); out[k] = run
    return out.tolist()


def mcnemar(a, b):
    from math import comb
    n10 = int((a & ~b).sum()); n01 = int((~a & b).sum()); n = n10 + n01
    if n == 0: return dict(n10=0, n01=0, p=1.0)
    k = min(n10, n01); p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return dict(n10=n10, n01=n01, p=p)


def ratio_test(Dd, Dc, EL, pidx, edges, flats, W):
    """Dd = |Delta logit| (P, J), Dc = call changed (P, J); pidx maps boundary -> row; edges, flats lists of boundaries"""
    ie = [pidx[i] for i in edges]; iF = [pidx[i] for i in flats]
    res = {}
    for nm, D in (('dlogit', Dd), ('calls', Dc)):
        m = pair_means(D, EL); mb = pair_means(D, EL, W)
        r = m[ie].mean() / max(m[iF].mean(), 1e-12); rb = mb[:, ie].mean(1) / np.maximum(mb[:, iF].mean(1), 1e-12)
        allp = ie + iF; perm = []
        for S in itertools.combinations(range(len(allp)), len(ie)):
            a = [allp[k] for k in S]; b = [allp[k] for k in range(len(allp)) if k not in S]
            perm.append(m[a].mean() / max(m[b].mean(), 1e-12))
        perm = np.array(perm)
        res[nm] = dict(edge_mean=float(m[ie].mean()), flat_mean=float(m[iF].mean()), ratio=float(r), ratio_ci=ci(rb),
                       p_boot_le1=float(np.mean(rb <= 1.0)), p_perm=float(np.mean(perm >= r - 1e-12)), n_perm=len(perm),
                       per_pair_edge=m[ie].tolist(), per_pair_flat=m[iF].tolist())
    per = []
    for e, f in zip(edges, flats):
        a, b = pidx[e], pidx[f]; both = (EL[a] > 0) & (EL[b] > 0); d = (Dd[a] - Dd[b])[both]
        Wb = W[:, both]; db = (Wb @ d) / np.maximum(Wb.sum(1), 1)
        pb = float(min(1.0, 2 * min(np.mean(db <= 0), np.mean(db >= 0))))
        mc = mcnemar(Dc[a][both] > 0, Dc[b][both] > 0)
        per.append(dict(edge=e, flat=f, n=int(both.sum()), dlogit_edge=float(Dd[a][both].mean()), dlogit_flat=float(Dd[b][both].mean()),
                        diff=float(d.mean()), diff_ci=ci(db), p_boot=pb, calls_edge=float(Dc[a][both].mean()),
                        calls_flat=float(Dc[b][both].mean()), mcnemar=mc))
    ph = holm([q['p_boot'] for q in per]); pm = holm([q['mcnemar']['p'] for q in per])
    for q, h, h2 in zip(per, ph, pm): q['p_boot_holm'] = h; q['mcnemar_p_holm'] = h2
    res['per_edge'] = per
    return res


def spearman(a, b):
    from scipy.stats import rankdata
    ra, rb = rankdata(a), rankdata(b); return float(np.corrcoef(ra, rb)[0, 1])


def resid_rank(v, r):
    from scipy.stats import rankdata
    X = np.stack([np.ones_like(r, dtype=float), r, r ** 2], 1); y = rankdata(v)
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def effect_vs_score(m, score, C, seed=7):
    rng = np.random.default_rng(seed); r = np.asarray(C, float); s = np.asarray(score, float); m = np.asarray(m, float)
    rho = spearman(m, s); a, b = resid_rank(m, r), resid_rank(s, r); pr = float(np.corrcoef(a, b)[0, 1])
    null = np.array([spearman(m, rng.permutation(s)) for _ in range(2000)])
    nullp = []
    for _ in range(2000):
        nullp.append(float(np.corrcoef(a, rng.permutation(b))[0, 1]))
    nullp = np.array(nullp)
    return dict(spearman=rho, p_perm=float(np.mean(np.abs(null) >= abs(rho))), partial_given_rank=pr,
                p_perm_partial=float(np.mean(np.abs(nullp) >= abs(pr))))


# ================================================================================================ E reading
def edge_profile(Es, C):
    """Es: list of (64, k) arrays (one per block) -> normalised score over C, per-block raw d"""
    raw = [np.linalg.norm(E[1:] - E[:-1], axis=1) for E in Es]          # (63,) boundary i|i+1 at index i
    sc = np.mean([d[C] / np.median(d[C]) for d in raw], 0)
    return sc, [d.tolist() for d in raw]


def top_edges(score, C, n=NTOP): return [C[k] for k in np.argsort(-np.asarray(score))[:n]]


def match_flats(score, C, edges, F):
    med = np.median(score); pool = [i for i, s in zip(C, score) if s <= med and min(abs(i - e) for e in edges) >= 2]
    used, out = set(), []
    for e in edges:
        best = min((j for j in pool if j not in used),
                   key=lambda j: abs(e - j) / 4 + abs(math.log(F[e]['gap'] / F[j]['gap'])) + abs(math.log(F[e]['dR'] / F[j]['dR'])))
        used.add(best); out.append(best)
    return out


def blocks_for(edges, C, nr):
    out = []
    for e in edges:
        a = e - 2
        if a < 1 or a + BLK > 64 or (nr >= a + BLK).sum() < MINJ: continue
        best = None
        for s in range(1, 64 - BLK + 1):
            if (nr >= s + BLK).sum() < MINJ: continue
            if any(s <= t <= s + BLK - 2 for t in edges): continue
            dist = abs((s + (BLK - 1) / 2) - (e + 0.5))
            if best is None or dist < best[0]: best = (dist, s)
        if best is not None: out.append(dict(edge=e, across=a, inside=best[1]))
    return out


def overlap_stats(A, Bs, C, seed=11):
    from scipy.stats import hypergeom
    A, Bs = list(A), list(Bs); ex = len(set(A) & set(Bs)); n = len(C)
    tol = sum(any(abs(a - b) <= 1 for b in Bs) for a in A)
    rng = np.random.default_rng(seed); null = []
    for _ in range(20000):
        R = rng.choice(C, len(A), replace=False); null.append(sum(any(abs(a - b) <= 1 for b in Bs) for a in R))
    null = np.array(null)
    return dict(exact=ex, exact_chance=len(A) * len(Bs) / n, p_exact=float(hypergeom.sf(ex - 1, n, len(Bs), len(A))),
                within1=int(tol), within1_chance=float(null.mean()), p_within1=float(np.mean(null >= tol)))


def profile_corr(a, b, seed=12):
    rng = np.random.default_rng(seed); r = spearman(a, b)
    null = np.array([spearman(a, rng.permutation(b)) for _ in range(5000)])
    return dict(spearman=r, p_perm=float(np.mean(np.abs(null) >= abs(r))))


# ================================================================================================ small mode (torch)
def small_mode():
    import importlib.util, torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
    spec = importlib.util.spec_from_file_location('small_models', ROOT / 'train' / 'models.py')
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    OUT.mkdir(parents=True, exist_ok=True)
    x, y, srt = load(); J = len(y); nr = nreal(x); C = candidates(x); pidx = {i: k for k, i in enumerate(C)}
    F = pair_features(x, C); W = boot_w(J)
    log(f'step 0: sample(per_class={PER}, seed={SEED}) N {J}; already pT-sorted {srt}; candidate boundaries {C[0]}..{C[-1]} ({len(C)})')
    # recentre check against the training code
    sys.path.insert(0, str(ROOT / 'train'))
    try:
        from train_small import recentre_hardest
        t = torch.from_numpy(np.ascontiguousarray(x)); rc = float(np.abs(recentre_hardest(t, t[..., 0] > 0).numpy() - recentre_np(x)).max())
    except Exception as ex:  # noqa: BLE001
        rc = f'not checked: {ex!r}'
    try:
        idx, nfull = sample_idx()
    except Exception as ex:  # noqa: BLE001
        idx, nfull = None, None; log('sample_idx failed', repr(ex))
    log('recentre numpy vs torch max |diff|', rc)

    nets = {}
    def add(name, run, seed, kind):
        d = SMALL / run / f'seed{0 if kind == "untrained" else seed}'; arch = json.loads((d / 'arch.json').read_text())
        if kind == 'untrained':
            torch.manual_seed(5000 + seed)
            net = M.build_model(arch['spec'], arch['n_particles'], cls_token=False, final_ln=arch['final_ln'])
        else:
            net = M.build_model(arch['spec'], arch['n_particles'], cls_token=False, final_ln=arch['final_ln'])
            net.load_state_dict(torch.load(d / 'best.pt', map_location='cpu'))
        net.eval(); nets[name] = dict(net=net, arch=arch, axis=arch['inputs'].get('axis', 'stored'), kind=kind, dir=d, seed=seed, run=run)
    add('L-st-s0', 'linf-2b-d32-k8-n64', 0, 'trained'); add('L-st-s1', 'linf-2b-d32-k8-n64', 1, 'trained')
    add('L-ax-s0', 'linf-2b-d32-k8-n64-axhard', 0, 'trained'); add('L-ax-s1', 'linf-2b-d32-k8-n64-axhard', 1, 'trained')
    add('D-s0', 'dsets-d64-n64', 0, 'deepsets'); add('D-s1', 'dsets-d64-n64', 1, 'deepsets')
    for k in range(5): add(f'U{k}', 'linf-2b-d32-k8-n64', k, 'untrained')

    def fwd(nm, xr):
        N = nets[nm]; a = N['arch']; xi = recentre_np(xr) if N['axis'] == 'hardest' else xr
        sh = np.asarray(a['inputs']['shift'], np.float32); sc = np.asarray(a['inputs']['scale'], np.float32)
        m = xi[..., 0] > 0; xn = (((xi - sh) / sc) * m[..., None]).astype(np.float32); out = []
        with torch.no_grad():
            for i in range(0, len(xn), 1000):
                out.append(N['net'](torch.from_numpy(xn[i:i + 1000]), torch.from_numpy(m[i:i + 1000])).double().numpy())
        return np.concatenate(out)

    res = dict(N=J, per_class=PER, seed=SEED, sorted_identity=srt, candidates=C, pair_features=F, recentre_check=rc,
               jedi_windows=dict(S0='ranks 0-13', S19='ranks 13-63'), nets={})
    # ---------------- step 1: read E
    for nm, N in nets.items():
        L0 = fwd(nm, x); c0 = L0.argmax(1); R = dict(kind=N['kind'], axis=N['axis'], run=N['run'], seed=N['seed'], acc=float((c0 == y).mean()))
        if N['kind'] != 'untrained':
            try:
                R['ref_acc_test_json'] = json.loads((N['dir'] / 'test.json').read_text()).get('accuracy')
                if idx is not None:
                    P = np.load(N['dir'] / 'test_probs.npy', mmap_mode='r')
                    if len(P) == nfull:
                        P = np.asarray(P[idx]); e_ = np.exp(L0 - L0.max(1, keepdims=True)); p_ = e_ / e_.sum(1, keepdims=True)
                        R['probs_maxabs_vs_test_probs'] = float(np.abs(P - p_).max()); R['argmax_agree_test_probs'] = float((P.argmax(1) == c0).mean())
            except Exception as ex:  # noqa: BLE001
                R['ref_check_error'] = repr(ex)
        N['L0'] = L0; N['c0'] = c0
        if N['run'].startswith('linf'):
            Es = [b.attn.E.detach().double().numpy() for b in N['net'].blocks]
            sc, raw = edge_profile(Es, C); R['E'] = [E.tolist() for E in Es]; R['edge_score'] = sc.tolist(); R['edge_raw'] = raw
            R['row_norm'] = [np.linalg.norm(E, axis=1).tolist() for E in Es]
            R['top_edges'] = top_edges(sc, C)
            if N['kind'] == 'trained':
                try:
                    ck = torch.load(N['dir'] / 'ckpt' / 'epoch_0.pt', map_location='cpu')
                    sd = ck.get('model', ck.get('state_dict', ck)) if isinstance(ck, dict) else ck
                    Ei = [sd[f'blocks.{b}.attn.E'].double().numpy() for b in range(len(Es))]
                    torch.manual_seed(N['seed'])
                    Mi = M.build_model(N['arch']['spec'], 64, cls_token=False, final_ln=N['arch']['final_ln'])
                    Er = [b.attn.E.detach().double().numpy() for b in Mi.blocks]
                    dE = [E - E0 for E, E0 in zip(Es, Ei)]
                    scd, _ = edge_profile(dE, C)
                    R['init'] = dict(reconstructed_seed_matches_ckpt=float(max(np.abs(a - b).max() for a, b in zip(Ei, Er))),
                                     corr_E_Einit=[float(np.corrcoef(E.ravel(), E0.ravel())[0, 1]) for E, E0 in zip(Es, Ei)],
                                     rel_change=[float(np.linalg.norm(d) / np.linalg.norm(E0)) for d, E0 in zip(dE, Ei)],
                                     edge_score_init=edge_profile(Ei, C)[0].tolist(), edge_score_dE=scd.tolist(),
                                     top_edges_dE=top_edges(scd, C), dE=[d.tolist() for d in dE])
                except Exception as ex:  # noqa: BLE001
                    R['init'] = dict(error=repr(ex))
            log(f'step 1 {nm}: acc {100 * R["acc"]:.2f} % (ref {R.get("ref_acc_test_json")}, probs |d| {R.get("probs_maxabs_vs_test_probs")}); '
                f'top-5 E edges {R["top_edges"]}; max score {max(sc):.2f}; init {json.dumps(R.get("init", {}).get("corr_E_Einit"))} '
                f'rel change {json.dumps(R.get("init", {}).get("rel_change"))} top dE {R.get("init", {}).get("top_edges_dE")}')
        else:
            log(f'step 1 {nm}: acc {100 * R["acc"]:.2f} % (ref {R.get("ref_acc_test_json")}, probs |d| {R.get("probs_maxabs_vs_test_probs")})')
        res['nets'][nm] = R
    # recurrence
    T = {nm: res['nets'][nm]['top_edges'] for nm in nets if 'top_edges' in res['nets'][nm]}
    S = {nm: res['nets'][nm]['edge_score'] for nm in T}
    pairs = [('L-st-s0', 'L-st-s1'), ('L-ax-s0', 'L-ax-s1'), ('L-st-s0', 'L-ax-s0'), ('L-st-s1', 'L-ax-s1'),
             ('L-st-s0', 'L-ax-s1'), ('L-st-s1', 'L-ax-s0')] + [(f'U{a}', f'U{b}') for a, b in itertools.combinations(range(5), 2)]
    rec = {}
    for a, b in pairs:
        rec[f'{a}|{b}'] = dict(overlap_stats(T[a], T[b], C), profile=profile_corr(S[a], S[b]))
        if a.startswith('L') and 'top_edges_dE' in res['nets'][a].get('init', {}) and 'top_edges_dE' in res['nets'][b].get('init', {}):
            ia, ib = res['nets'][a]['init'], res['nets'][b]['init']
            rec[f'{a}|{b}']['dE'] = dict(overlap_stats(ia['top_edges_dE'], ib['top_edges_dE'], C), profile=profile_corr(ia['edge_score_dE'], ib['edge_score_dE']))
    for nm in ('L-st-s0', 'L-st-s1', 'L-ax-s0', 'L-ax-s1'):
        it = res['nets'][nm].get('init', {})
        if 'edge_score_init' in it: rec[f'{nm}|own_init'] = dict(overlap_stats(T[nm], top_edges(it['edge_score_init'], C), C),
                                                                 profile=profile_corr(S[nm], it['edge_score_init']))
    res['recurrence'] = rec
    for k, v in rec.items():
        log(f'step 1 recurrence {k}: exact {v["exact"]} (chance {v["exact_chance"]:.2f}, p {v["p_exact"]:.3f}); +-1 {v["within1"]} '
            f'(chance {v["within1_chance"]:.2f}, p {v["p_within1"]:.3f}); profile rho {v["profile"]["spearman"]:.3f} p {v["profile"]["p_perm"]:.4f}'
            + (f'; dE exact {v["dE"]["exact"]} rho {v["dE"]["profile"]["spearman"]:.3f} p {v["dE"]["profile"]["p_perm"]:.4f}' if 'dE' in v else ''))
    # ---------------- step 2: swap scan
    for nm, N in nets.items():
        Dd = np.zeros((len(C), J), np.float32); Dc = np.zeros((len(C), J), np.float32); EL = np.zeros((len(C), J), np.float32)
        for k, i in enumerate(C):
            xe, el = swap(x, i); L1 = fwd(nm, xe)
            Dd[k] = np.abs(L1 - N['L0']).mean(1); Dc[k] = (L1.argmax(1) != N['c0']); EL[k] = el
        N['scan'] = (Dd, Dc, EL); R = res['nets'][nm]
        R['scan_dlogit'] = pair_means(Dd, EL).tolist(); R['scan_calls'] = pair_means(Dc, EL).tolist(); R['scan_n'] = EL.sum(1).tolist()
        R['logit_scale'] = float(np.abs(N['L0'] - N['L0'].mean(1, keepdims=True)).mean())
        log(f'step 2 scan {nm}: mean |dlogit| over boundaries {np.mean(R["scan_dlogit"]):.4g}, calls {100 * np.mean(R["scan_calls"]):.2f} %')
    ref = 'L-st-s0'
    for nm, N in nets.items():
        R = res['nets'][nm]; Dd, Dc, EL = N['scan']; tests = {}
        if 'top_edges' in R:
            own_e = R['top_edges']; own_f = match_flats(R['edge_score'], C, own_e, F); R['flats'] = own_f
            tests['own'] = dict(edges=own_e, flats=own_f, **ratio_test(Dd, Dc, EL, pidx, own_e, own_f, W))
            tests['effect_vs_score_own'] = effect_vs_score(R['scan_dlogit'], R['edge_score'], C)
            if 'init' in R and 'top_edges_dE' in R['init']:
                de = R['init']['top_edges_dE']; df = match_flats(R['init']['edge_score_dE'], C, de, F)
                tests['own_dE'] = dict(edges=de, flats=df, **ratio_test(Dd, Dc, EL, pidx, de, df, W))
                tests['effect_vs_score_dE'] = effect_vs_score(R['scan_dlogit'], R['init']['edge_score_dE'], C)
        if nm != ref:
            re_, rf = res['nets'][ref]['top_edges'], res['nets'][ref]['flats']
            tests['at_ref'] = dict(edges=re_, flats=rf, **ratio_test(Dd, Dc, EL, pidx, re_, rf, W))
            tests['effect_vs_score_ref'] = effect_vs_score(R['scan_dlogit'], res['nets'][ref]['edge_score'], C)
        R['step2'] = tests
        for k, t in tests.items():
            if 'dlogit' in t:
                log(f'step 2 {nm} [{k}] edges {t["edges"]} flats {t["flats"]}: |dlogit| ratio {t["dlogit"]["ratio"]:.3f} {np.round(t["dlogit"]["ratio_ci"], 3).tolist()} '
                    f'perm p {t["dlogit"]["p_perm"]:.3f}; calls {100 * t["calls"]["edge_mean"]:.2f} vs {100 * t["calls"]["flat_mean"]:.2f} % ratio {t["calls"]["ratio"]:.3f} '
                    f'{np.round(t["calls"]["ratio_ci"], 3).tolist()} perm p {t["calls"]["p_perm"]:.3f}; Holm p (boot) {np.round([q["p_boot_holm"] for q in t["per_edge"]], 4).tolist()}')
            else:
                log(f'step 2 {nm} [{k}] {json.dumps(t)}')
    U = [res['nets'][f'U{k}']['step2']['own']['dlogit']['ratio'] for k in range(5)]
    Uc = [res['nets'][f'U{k}']['step2']['own']['calls']['ratio'] for k in range(5)]
    res['untrained_ratio'] = dict(dlogit=U, dlogit_q95=float(np.quantile(U, 0.95)), calls=Uc, calls_q95=float(np.nanquantile(Uc, 0.95)))
    log('step 2 untrained own-edge ratios', res['untrained_ratio'])
    # ---------------- step 3: window shuffle
    for nm, N in nets.items():
        R = res['nets'][nm]; src = nm if 'top_edges' in R else ref
        edges = res['nets'][src]['top_edges']; BL = blocks_for(edges, C, nr); rows = []
        for b in BL:
            P = derangements(J, BLK, 150100 + b['edge'])
            xa, ea = shuffle_block(x, b['across'], P); xi, ei = shuffle_block(x, b['inside'], P)
            La, Li = fwd(nm, xa), fwd(nm, xi); both = ea & ei
            da = np.abs(La - N['L0']).mean(1)[both]; di = np.abs(Li - N['L0']).mean(1)[both]
            ca = (La.argmax(1) != N['c0'])[both]; cc = (Li.argmax(1) != N['c0'])[both]
            Wb = W[:, both]; s = np.maximum(Wb.sum(1), 1)
            rb = (Wb @ da) / s / np.maximum((Wb @ di) / s, 1e-12); db = (Wb @ (da - di)) / s
            rows.append(dict(b, n=int(both.sum()), dlogit_across=float(da.mean()), dlogit_inside=float(di.mean()),
                             ratio=float(da.mean() / max(di.mean(), 1e-12)), ratio_ci=ci(rb),
                             p_boot=float(min(1.0, 2 * min(np.mean(db <= 0), np.mean(db >= 0)))),
                             calls_across=float(ca.mean()), calls_inside=float(cc.mean()), mcnemar=mcnemar(ca, cc)))
        if rows:
            for q, h, h2 in zip(rows, holm([q['p_boot'] for q in rows]), holm([q['mcnemar']['p'] for q in rows])):
                q['p_boot_holm'] = h; q['mcnemar_p_holm'] = h2
        R['step3'] = dict(edges_from=src, rows=rows,
                          mean_ratio=float(np.mean([q['ratio'] for q in rows])) if rows else None,
                          calls_across=float(np.mean([q['calls_across'] for q in rows])) if rows else None,
                          calls_inside=float(np.mean([q['calls_inside'] for q in rows])) if rows else None)
        log(f'step 3 {nm} (edges of {src}): ' + '; '.join(f'e{q["edge"]} across {q["across"]}-{q["across"] + 5} vs inside {q["inside"]}-{q["inside"] + 5}: '
                                                          f'|dl| ratio {q["ratio"]:.2f} {np.round(q["ratio_ci"], 2).tolist()} Holm p {q["p_boot_holm"]:.3g}; calls '
                                                          f'{100 * q["calls_across"]:.2f}/{100 * q["calls_inside"]:.2f} % McN Holm {q["mcnemar_p_holm"]:.3g}' for q in rows))
    for N in nets.values(): N.pop('scan', None)
    fo = OUT / 'x15_small.json'; fo.write_text(json.dumps(clean(res))); log('wrote', fo)


# ================================================================================================ jedi mode (jax)
def jedi_mode():
    from interp.jedi_core import Jedi
    OUT.mkdir(parents=True, exist_ok=True)
    x, y, srt = load(); J = len(y); C = candidates(x); pidx = {i: k for k, i in enumerate(C)}; W = boot_w(J); F = pair_features(x, C)
    net = Jedi(64); L0 = net.run(x)['logits']; c0 = L0.argmax(1); acc = float((c0 == y).mean())
    log(f'jedi: sample N {J}, sorted {srt}, acc {100 * acc:.2f} %')
    Dd = np.zeros((len(C), J), np.float32); Dc = np.zeros((len(C), J), np.float32); EL = np.zeros((len(C), J), np.float32)
    for k, i in enumerate(C):
        xe, el = swap(x, i); L1 = net.run(xe)['logits']; Dd[k] = np.abs(L1 - L0).mean(1); Dc[k] = L1.argmax(1) != c0; EL[k] = el
    log('jedi scan done')
    t = ratio_test(Dd, Dc, EL, pidx, [JEDI_EDGE] * len(JEDI_FLATS), JEDI_FLATS, W)
    # single-edge ratio vs the mean of the 5 flats (the repeated-edge labelling above gives the per-edge paired tests)
    m = pair_means(Dd, EL); mb = pair_means(Dd, EL, W); mc = pair_means(Dc, EL); mcb = pair_means(Dc, EL, W)
    ie = pidx[JEDI_EDGE]; iF = [pidx[i] for i in JEDI_FLATS]
    r = m[ie] / m[iF].mean(); rb = mb[:, ie] / mb[:, iF].mean(1); rc = mc[ie] / max(mc[iF].mean(), 1e-12); rcb = mcb[:, ie] / np.maximum(mcb[:, iF].mean(1), 1e-12)
    out = dict(N=J, seed=SEED, acc=acc, sorted_identity=srt, candidates=C, edge=JEDI_EDGE, flats=JEDI_FLATS, pair_features=F,
               scan_dlogit=m.tolist(), scan_calls=mc.tolist(), scan_n=EL.sum(1).tolist(),
               dlogit=dict(edge=float(m[ie]), flat=float(m[iF].mean()), ratio=float(r), ratio_ci=ci(rb),
                           rank_of_edge_among_6=int(1 + np.sum(m[iF] > m[ie]))),
               calls=dict(edge=float(mc[ie]), flat=float(mc[iF].mean()), ratio=float(rc), ratio_ci=ci(rcb)),
               per_flat=t['per_edge'])
    log(f'jedi 13|14 vs flats {JEDI_FLATS}: |dlogit| {m[ie]:.4g} vs {m[iF].mean():.4g} ratio {r:.3f} {np.round(ci(rb), 3).tolist()}; '
        f'calls {100 * mc[ie]:.2f} vs {100 * mc[iF].mean():.2f} % ratio {rc:.3f} {np.round(ci(rcb), 3).tolist()}')
    fo = OUT / 'x15_jedi.json'; fo.write_text(json.dumps(clean(out))); log('wrote', fo)


# ================================================================================================ combine
def combine():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    S = json.loads((OUT / 'x15_small.json').read_text()); Jd = json.loads((OUT / 'x15_jedi.json').read_text())
    C = S['candidates']; nets = S['nets']
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(width_ratios=[1.5, 1, 1.1]))
    a = ax[0]
    U = np.array([nets[f'U{k}']['edge_score'] for k in range(5)])
    a.fill_between(C, U.min(0), U.max(0), color='0.85', label='untrained Linformers (5, min-max)')
    for nm, c in (('L-st-s0', 'C0'), ('L-st-s1', 'C1'), ('L-ax-s0', 'C2'), ('L-ax-s1', 'C3')):
        a.plot(C, nets[nm]['edge_score'], color=c, lw=1.2, label=f'{nm} (top-5: {nets[nm]["top_edges"]})')
    lo = 0.25
    a.plot([0, 13.5], [lo + 0.08] * 2, color='k', lw=4, solid_capstyle='butt'); a.text(0.5, lo + 0.12, 'JEDI S0 mask: ranks 0-13', fontsize=7.5)
    a.plot([12.5, 63], [lo] * 2, color='0.4', lw=4, solid_capstyle='butt'); a.text(30, lo + 0.04, 'JEDI S19 mask: ranks 13-63', fontsize=7.5)
    a.set_ylim(lo - 0.08, 2.6)
    a.axvline(13.5, color='k', ls=':', lw=0.8)
    a.set_xlabel('boundary i | i+1 between pT ranks (0 = hardest particle)'); a.set_ylabel(r'edge score $\|E_{i+1}-E_i\|$ / median')
    a.set_title('(a) Linformer rank-projection E: jumps between adjacent pT ranks', fontsize=9); a.legend(fontsize=6.5, loc='upper right', bbox_to_anchor=(1, 0.93))
    a = ax[1]; rows = []
    for nm in ('L-st-s0', 'L-st-s1', 'L-ax-s0', 'L-ax-s1', 'U0', 'U1', 'U2', 'U3', 'U4'):
        t = nets[nm]['step2']['own']['dlogit']; rows.append((nm + ' (own E edges)', t['ratio'], t['ratio_ci']))
    for nm in ('D-s0', 'D-s1'):
        t = nets[nm]['step2']['at_ref']['dlogit']; rows.append((nm + ' Deep Sets (L-st-s0 pairs)', t['ratio'], t['ratio_ci']))
    rows.append(('JEDI-linear 13|14 vs 5 flats', Jd['dlogit']['ratio'], Jd['dlogit']['ratio_ci']))
    for k, (lab, r, c) in enumerate(rows):
        col = 'C0' if lab.startswith('L') else ('0.5' if lab.startswith('U') else ('C4' if lab.startswith('D') else 'k'))
        a.errorbar(r, -k, xerr=[[r - c[0]], [c[1] - r]], fmt='o', color=col, ms=4, capsize=2)
    a.set_yticks(-np.arange(len(rows))); a.set_yticklabels([r[0] for r in rows], fontsize=7); a.axvline(1, color='k', lw=0.7, ls=':')
    a.set_xlabel('angle-swap response, edge pairs / matched flat pairs\n(mean |Δ logit|, 95 % jet-bootstrap CI)')
    a.set_title('(b) swapping angles across an E edge vs a flat pair', fontsize=9)
    a = ax[2]; R = nets['L-st-s0']
    a.scatter(R['edge_score'], R['scan_dlogit'], c=C, cmap='viridis', s=14)
    for e in R['top_edges']: a.annotate(str(e), (R['edge_score'][C.index(e)], R['scan_dlogit'][C.index(e)]), fontsize=7)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(min(C), max(C))); fig.colorbar(sm, ax=a, label='boundary rank i')
    ev = R['step2']['effect_vs_score_own']
    a.set_xlabel('E edge score at boundary i|i+1'); a.set_ylabel('mean |Δ logit| after swapping ranks i, i+1')
    a.set_title(f'(c) L-st-s0, all {len(C)} boundaries: Spearman {ev["spearman"]:.2f}, given rank {ev["partial_given_rank"]:.2f}', fontsize=9)
    fig.tight_layout(); fo = OUT / 'x15_linformer_ranks.png'; fig.savefig(fo, dpi=130); log('wrote', fo)
    summ = dict(untrained=S['untrained_ratio'], recurrence=S['recurrence'], jedi=dict((k, Jd[k]) for k in ('acc', 'dlogit', 'calls')),
                nets={nm: dict(acc=v['acc'], top_edges=v.get('top_edges'), flats=v.get('flats'),
                               step2={k: (dict(edges=t.get('edges'), flats=t.get('flats'), dlogit=t.get('dlogit'), calls=t.get('calls')) if 'dlogit' in t else t)
                                      for k, t in v['step2'].items()}, step3=v['step3']) for nm, v in nets.items()})
    (OUT / 'x15_summary.json').write_text(json.dumps(clean(summ), indent=1)); log('wrote', OUT / 'x15_summary.json')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--mode', choices=('small', 'jedi', 'combine'), required=True)
    a = ap.parse_args()
    dict(small=small_mode, jedi=jedi_mode, combine=combine)[a.mode]()


if __name__ == '__main__':
    main()
