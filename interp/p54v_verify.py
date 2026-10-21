"""P54v (loop C, round C12): verification of P54 (L-GATr equivariant-MLP channel atlas) on a fresh sample, with added controls.

Sample. JetClass test_20M <class>_107.root (P54 used _106), 2,000 random rows per class, numpy default_rng(5455) (never
first-k), split at random within class into sample 1 (labelling / ranking, N 10,000) and sample 2 (held out, N 10,000).
Model: L-GATr final checkpoint model_run0.pt (P54's). All P54 definitions are reused by importing interp.p54_lgatr_channel_atlas.

What is re-measured / added
 1 atlas     P54's sec_atlas unchanged (trained channel basis vs 10 raw + 10 std grade-preserving rotations, untrained, permuted).
 2 extra     Added rotation controls, point estimates on sample 2:
             (a) 20 further untied raw rotations (fresh seeds) -> rank / z of the trained clean and selective fractions;
             (b) 10 'tied' raw rotations and 10 tied std rotations: ONE Haar matrix per block acts on every grade, so a rotated
                 unit's scalar part and four-vector part come from the same direction (in P54 the scalar and four-vector parts of
                 a rotated unit came from two unrelated directions R_0, R_1, while a channel's f0 and f1 belong to one channel);
             (c) pre-gate atlas: the same features taken at the INPUT of the gated GELU (before gating) for the channel basis and
                 10 untied + 10 tied raw rotations. If the channel basis is equally privileged before the gate, the read-out
                 privilege is not produced by the gating itself.
 3 steer     P54's sec_steer at N 1,000 (100 random sample-2 jets per class).
 4 ablate    Block-matched null for single-unit mean ablation: in every block all 32 channels and 8 tied-rotation directions
             (whole multivector set to its token-type mean) on N 1,000; calls changed per unit, per block channel vs direction
             (Mann-Whitney), share of channels above the block's direction maximum. (P54's null drew 30 directions from random
             blocks and compared each channel with the pooled maximum.)
 5 circuit   Ceiling of the literal-circuit test: P54's frozen all-class ranking (its circuit_attr, ranked on _106 sample 1)
             evaluated on N 5,000 sample-2 jets: keep K = 32 ... 384 channels with the hidden auxiliary scalars resampled
             (P54's protocol) or live; random K-sets; floor. 'all384 / aux resampled' = every hidden multivector channel kept,
             only the 768 hidden scalars resampled: if this is near chance, P54's circuit failure is set by the scalars.
 6 calib     R_patch for the top-K log-m channels against (i) uniformly random K-sets (P54), (ii) block-matched random K-sets
             (same number of channels from each block as the top set), (iii) top-K log-m directions of 3 tied rotations
             (patched direction == scored direction), (iv) P54's untied rotation-0 directions (scored with R_1 on the
             four-vector part but patched along R_0 on all grades).

  ./sync.sh && ./part_job.sh c-p54-verify '/j-jepa-vol/aaron-interp/env-lgatr/bin/python -m interp.p54v_verify'
  second pass (after a fix in compare()): ./part_job.sh c-p54-verify2 'P54V_SECTIONS=atlas,extra,calib,ablate P54V_TAG=_b /j-jepa-vol/aaron-interp/env-lgatr/bin/python -m interp.p54v_verify'
-> $AI_RESULTS/part/p54_verify.json
"""
import json, os, sys, time, traceback
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('JETDISTILL_TAGGER', 'part')
_argv = sys.argv; sys.argv = [sys.argv[0]]
import interp.p54_lgatr_channel_atlas as M
sys.argv = _argv
M.FIDX = 107
SEED = 5455
NBLK, NC, NS, NO = M.NBLK, M.NC, M.NS, M.NO
SECS = os.environ.get('P54V_SECTIONS', 'atlas,extra,calib,circuit,steer,ablate').split(',')
OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'part' / f"p54_verify{os.environ.get('P54V_TAG', '')}.json"
ATTR = Path(os.environ.get('AI_RESULTS', 'results')) / 'lgatr' / 'p54_arrays.npz'
log = M.log


def dump(out):
    M.jdump({k: v for k, v in out.items() if not k.startswith('_')}, OUT)


def haar_set(seed, tied):
    """(R0, R1, RS) per block as torch tensors; tied -> R1 = R0"""
    import torch
    R0 = np.stack([M.haar(NC, seed + 10 * b) for b in range(NBLK)])
    R1 = R0 if tied else np.stack([M.haar(NC, seed + 10 * b + 1) for b in range(NBLK)])
    RS = np.stack([M.haar(NS, seed + 10 * b + 2) for b in range(NBLK)])
    return tuple(torch.from_numpy(x).float().cuda() if torch.cuda.is_available() else torch.from_numpy(x).float() for x in (R0, R1, RS))


def stds(c1):
    import torch
    return (c1['f0'].std(0).clamp(min=1e-6), torch.diagonal(c1['G'], dim1=-2, dim2=-1).abs().mean(0).sqrt().clamp(min=1e-6),
            c1['fs'].std(0).clamp(min=1e-6))


def score(c1, c2, bs, rots, bstd, D, W=None):
    a1, b1 = M.unit_feats(c1, bs, rots, bstd); a2, b2 = M.unit_feats(c2, bs, rots, bstd)
    Smv, Smvb = M.score_units(a1, a2, D['Y1z'], D['Y2z'], D['yt1'], D['yt2'], W=W)
    Sax, _ = M.score_units(b1, b2, D['Y1z'], D['Y2z'], D['yt1'], D['yt2'])
    return Smv, Smvb, Sax


def fr(S, thr):
    cl, se = M.flags(S, thr); return float(cl.float().mean()), float(se.float().mean())


def perm_p(ft, fr_, n=10000, seed=SEED):
    pool_ = np.concatenate([ft, fr_]); obs = ft.mean() - fr_.mean(); g = np.random.default_rng(seed); cnt = 0
    for _ in range(n):
        pp = g.permutation(pool_); cnt += (pp[:len(ft)].mean() - pp[len(ft):].mean()) >= obs - 1e-12
    return float((1 + cnt) / (1 + n))


def compare(name_tr, S_tr, S_rots, Sb_tr=None, Sb_rots=None, thr=float(np.sqrt(0.5))):
    """trained-basis fractions vs a family of rotated bases"""
    r = {}
    for k, which in enumerate(('clean', 'selective')):
        tr = fr(S_tr, thr)[k]; rv = np.array([fr(S, thr)[k] for S in S_rots])
        ft = M.flags(S_tr, thr)[k].float().cpu().numpy(); fp = np.concatenate([M.flags(S, thr)[k].float().cpu().numpy() for S in S_rots])
        d = dict(trained=tr, rot_mean=float(rv.mean()), rot_sd=float(rv.std(ddof=1)), rot_min=float(rv.min()), rot_max=float(rv.max()),
                 n_rot=int(len(rv)), p_rank=float((1 + (rv >= tr).sum()) / (1 + len(rv))),
                 z=float((tr - rv.mean()) / max(rv.std(ddof=1), 1e-9)), p_perm_units=perm_p(ft, fp), diff=float(tr - rv.mean()))
        if Sb_tr is not None:
            bt = M.flags(Sb_tr, thr)[k].float().mean(1).cpu().numpy(); br = np.mean([M.flags(S, thr)[k].float().mean(1).cpu().numpy() for S in Sb_rots], 0)
            dd = bt - br; d['diff_ci'] = [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]
            d['p_boot_le0'] = float((1 + (dd <= 0).sum()) / (1 + len(dd)))
        r[which] = d
    log(f'extra {name_tr}: clean {r["clean"]["trained"]:.3f} vs {r["clean"]["rot_mean"]:.3f} [{r["clean"]["rot_min"]:.3f}, {r["clean"]["rot_max"]:.3f}] '
        f'z {r["clean"]["z"]:.1f} p_perm {r["clean"]["p_perm_units"]:.4f} | selective {r["selective"]["trained"]:.3f} vs {r["selective"]["rot_mean"]:.3f} '
        f'[{r["selective"]["rot_min"]:.3f}, {r["selective"]["rot_max"]:.3f}] z {r["selective"]["z"]:.1f} p_perm {r["selective"]["p_perm_units"]:.4f}')
    return r


# --------------------------------------------------------------------------------------- pre-gate capture
def add_pregate_hooks(net):
    net._pre = None
    mods = [m for blk in net.model.net.blocks for m in blk.mlp.layers if type(m).__name__ == 'ScalarGatedNonlinearity']
    assert len(mods) == NBLK

    def mk(b):
        def h(mod, args, kwargs):
            if net._pre is not None:
                mv = args[0]; s = kwargs['scalars'] if 'scalars' in kwargs else args[1]
                net._pre(b, mv, s)
            return None
        return h
    for b, m in enumerate(mods): m.register_forward_pre_hook(mk(b), with_kwargs=True)


def capture_pre(net, J, P):
    import torch
    dev = net.dev; N = len(J['label']); eta = torch.tensor(M.ETA, device=dev)
    f0 = torch.zeros(N, NBLK, NC, device=dev); G = torch.zeros(N, NBLK, NC, NC, device=dev); fs = torch.zeros(N, NBLK, NS, device=dev); cur = {}

    def rec(b, mv, s):
        m = net.meta; x = mv[0].float(); sl = cur['sl']
        f0[sl, b] = M.pool(m, x[:, :, 0]); fs[sl, b] = M.pool(m, s[0].float()); v = x[:, :, 1:5]
        G[sl, b] = M.pool(m, torch.einsum('tam,tbm->tab', v * eta, v))
    net._pre = rec; rows = np.arange(N)
    try:
        with torch.no_grad():
            for s0 in range(0, N, P['BS']):
                sl = slice(s0, min(N, s0 + P['BS'])); cur['sl'] = sl; net.logits(net.embed(J, rows[sl]))
    finally:
        net._pre = None
    return dict(f0=f0, G=G, fs=fs)


def sec_extra(net, D, P, out):
    import torch
    dev = net.dev; thr = float(np.sqrt(0.5))
    W = torch.from_numpy(M.boot_W(len(D['y2']), P['NB'], SEED)).to(dev)
    c1 = M.capture(net, D['J1'], P); c2 = M.capture(net, D['J2'], P); bstd = stds(c1)
    res = dict(threshold='held-out R^2 >= 0.5 (class: 2AUC-1 >= 0.707)')
    rots = {}
    for i in range(20): rots[f'u{i}'] = haar_set(7_000_000 + 1000 * i, tied=False)
    for i in range(10): rots[f't{i}'] = haar_set(8_000_000 + 1000 * i, tied=True)
    S = {}
    S['ch'] = score(c1, c2, ('channel',), rots, bstd, D, W=W)
    for k in rots:
        S[f'raw_{k}'] = score(c1, c2, ('raw', k), rots, bstd, D, W=W if k.startswith('t') else None)
        if k.startswith('t'): S[f'std_{k}'] = score(c1, c2, ('std', k), rots, bstd, D)
    log('extra: post-gate bases scored')
    un = [S[f'raw_u{i}'][0] for i in range(20)]; ti = [S[f'raw_t{i}'][0] for i in range(10)]; ts = [S[f'std_t{i}'][0] for i in range(10)]
    res['post_gate'] = dict(
        untied_raw_20=compare('post untied raw x20', S['ch'][0], un),
        tied_raw_10=compare('post tied raw x10', S['ch'][0], ti, S['ch'][1], [S[f'raw_t{i}'][1] for i in range(10)]),
        tied_std_10=compare('post tied std x10', S['ch'][0], ts),
        aux_untied_raw_20=compare('post aux untied raw x20', S['ch'][2], [S[f'raw_u{i}'][2] for i in range(20)]))
    # labels of clean trained channels and of tied rotation 0
    for nm, Sx in (('channel', S['ch'][0]), ('tied_raw0', S['raw_t0'][0])):
        cl, _ = M.flags(Sx, thr); lab = Sx.argmax(1)
        res['post_gate'][f'label_counts_{nm}'] = {M.LABELS[j]: int((cl & (lab == j)).sum()) for j in range(M.NL) if bool((cl & (lab == j)).any())}
        res['post_gate'][f'per_block_clean_{nm}'] = [float(x) for x in cl.float().reshape(NBLK, -1).mean(1)]
    # tied-rotation log-m scores for calibration
    D['_tied'] = {i: (rots[f't{i}'][0], S[f'raw_t{i}'][0][:, M.OBS.index('log_m')].cpu().numpy()) for i in range(3)}
    del c1, c2; torch.cuda.empty_cache()
    # pre-gate
    p1 = capture_pre(net, D['J1'], P); p2 = capture_pre(net, D['J2'], P); pstd = stds(p1)
    Sp = {'ch': score(p1, p2, ('channel',), rots, pstd, D)}
    for i in range(10):
        Sp[f'u{i}'] = score(p1, p2, ('raw', f'u{i}'), rots, pstd, D); Sp[f't{i}'] = score(p1, p2, ('raw', f't{i}'), rots, pstd, D)
    res['pre_gate'] = dict(untied_raw_10=compare('pre untied raw x10', Sp['ch'][0], [Sp[f'u{i}'][0] for i in range(10)]),
                           tied_raw_10=compare('pre tied raw x10', Sp['ch'][0], [Sp[f't{i}'][0] for i in range(10)]),
                           aux_untied_raw_10=compare('pre aux untied raw x10', Sp['ch'][2], [Sp[f'u{i}'][2] for i in range(10)]))
    cl, _ = M.flags(Sp['ch'][0], thr); lab = Sp['ch'][0].argmax(1)
    res['pre_gate']['label_counts_channel'] = {M.LABELS[j]: int((cl & (lab == j)).sum()) for j in range(M.NL) if bool((cl & (lab == j)).any())}
    del p1, p2; torch.cuda.empty_cache()
    out['extra'] = res


# --------------------------------------------------------------------------------------- block-matched ablation null
def sec_ablate_block(net, D, P, out, NDIR=8, PERC=100):
    import torch
    from scipy.stats import mannwhitneyu
    rng = np.random.default_rng(SEED + 11)
    rows = np.sort(np.concatenate([rng.choice(np.where(D['y2'] == c)[0], PERC, replace=False) for c in range(10)]))
    y = D['y2'][rows]; N = len(rows); Rt = D['_tied'][0][0]
    units = [('ch', b, c) for b in range(NBLK) for c in range(NC)] + [('dir', b, j) for b in range(NBLK) for j in range(NDIR)]
    Lp = np.zeros((len(units), N), np.int16); p0 = np.zeros(N, np.int16)

    def fn(e, sl, L0):
        p0[sl] = L0.argmax(1).cpu().numpy()
        for i, (k, b, c) in enumerate(units):
            net.op = M.mean_op(net, D, b, chans=[c]) if k == 'ch' else M.mean_op(net, D, b, dirvec=Rt[b, c])
            try: Lp[i, sl] = net.logits(e).argmax(1).cpu().numpy()
            finally: net.op = None
    net.run(D['J2'], P, fn, rows=rows)
    cc = (Lp != p0[None]).mean(1); acc = (Lp == y[None]).mean(1) - (p0 == y).mean()
    blocks = []
    for b in range(NBLK):
        ich = [i for i, u in enumerate(units) if u[0] == 'ch' and u[1] == b]; idr = [i for i, u in enumerate(units) if u[0] == 'dir' and u[1] == b]
        a, d = cc[ich], cc[idr]
        blocks.append(dict(block=b, ch_mean=float(a.mean()), ch_median=float(np.median(a)), ch_max=float(a.max()), dir_mean=float(d.mean()),
                           dir_median=float(np.median(d)), dir_max=float(d.max()), frac_ch_above_dir_max=float((a > d.max()).mean()),
                           p_mwu_ch_gt_dir=float(mannwhitneyu(a, d, alternative='greater').pvalue),
                           p_mwu_ch_ne_dir=float(mannwhitneyu(a, d, alternative='two-sided').pvalue)))
        log(f"ablate block {b}: channels calls changed mean {a.mean():.4f} (max {a.max():.3f}) vs tied directions {d.mean():.4f} (max {d.max():.3f}); "
            f"MWU p(ch>dir) {blocks[-1]['p_mwu_ch_gt_dir']:.3g}")
    ch = cc[:NBLK * NC]; dr = cc[NBLK * NC:]
    # block-stratified permutation: within each block shuffle channel/direction labels; statistic = mean over blocks of (ch mean - dir mean)
    g = np.random.default_rng(SEED + 12); obs = np.mean([bb['ch_mean'] - bb['dir_mean'] for bb in blocks]); cnt = 0
    for _ in range(10000):
        st = []
        for b in range(NBLK):
            v = np.concatenate([ch[b * NC:(b + 1) * NC], dr[b * NDIR:(b + 1) * NDIR]]); v = g.permutation(v); st.append(v[:NC].mean() - v[NC:].mean())
        cnt += abs(np.mean(st)) >= abs(obs) - 1e-12
    out['ablate_block'] = dict(N=int(N), rows=f'{PERC} random sample-2 jets per class (default_rng({SEED + 11}))', ndir_per_block=NDIR,
                               edit='whole multivector of the unit set to its sample-1 token-type mean (direction: component along u, all 16 components)',
                               acc_unedited=float((p0 == y).mean()), blocks=blocks, mean_block_diff_ch_minus_dir=float(obs),
                               p_perm_block_stratified_two_sided=float((1 + cnt) / 10001),
                               dacc_ch_quantiles=[float(x) for x in np.quantile(acc[:NBLK * NC], [0, 0.1, 0.5, 0.9, 1])],
                               dacc_dir_quantiles=[float(x) for x in np.quantile(acc[NBLK * NC:], [0, 0.1, 0.5, 0.9, 1])])
    log(f'ablate block-matched: mean over blocks (channel - direction) calls changed {obs:+.4f}, p_perm {(1 + cnt) / 10001:.4f}')


# --------------------------------------------------------------------------------------- circuit ceiling
def sec_circuit_ceiling(net, D, P, out, PERC=500):
    import torch
    A = np.load(ATTR)['circuit_attr']; rank_all = np.argsort(-A.reshape(10, NBLK * NC).sum(0))
    rng = np.random.default_rng(SEED + 13)
    rows = np.sort(np.concatenate([rng.choice(np.where(D['y2'] == c)[0], PERC, replace=False) for c in range(10)]))
    y = D['y2'][rows]; N = len(rows); g = np.random.default_rng(SEED + 14)
    allc = np.arange(NBLK * NC)
    variants = [('floor', None, False), ('floor_auxlive', None, True), ('all384', allc, False), ('all384_auxlive', allc, True)]
    for K in (32, 64, 128, 192, 256, 320):
        rs = g.choice(NBLK * NC, K, replace=False)
        for al in (False, True):
            variants.append((f'ranked_K{K}' + ('_auxlive' if al else ''), rank_all[:K], al))
            variants.append((f'random_K{K}' + ('_auxlive' if al else ''), rs, al))
    # block 0 alone resampled / everything but block 0 resampled (aux of the same blocks treated alike)
    Ls = {v[0]: np.zeros((N, 10), np.float32) for v in variants}; L0s = np.zeros((N, 10), np.float32)
    blk_variants = [('only_block0_resampled', [0]), ('blocks1to11_resampled', list(range(1, NBLK)))]
    for nm, _ in blk_variants: Ls[nm] = np.zeros((N, 10), np.float32)
    dev = net.dev

    def fn(e, sl, L0):
        L0s[sl] = L0.float().cpu().numpy(); cache = {}
        net.rec = lambda b, mv, s_: cache.__setitem__(b, (mv[0].clone(), s_[0].clone()))
        try: net.logits(e)
        finally: net.rec = None
        dt = net.meta['donor_tok']
        for name, keep, auxlive in variants:
            km = torch.zeros(NBLK, NC, dtype=torch.bool, device=dev)
            if keep is not None:
                kt = torch.as_tensor(np.asarray(keep), device=dev); km[kt // NC, kt % NC] = True

            def op(b, mv, s_, km=km, auxlive=auxlive):
                cm, cs = cache[b]
                return torch.where(km[b][None, :, None], mv[0], cm[dt])[None], (s_[0] if auxlive else cs[dt])[None]
            net.op = op
            try: Ls[name][sl] = net.logits(e).float().cpu().numpy()
            finally: net.op = None
        for name, bl in blk_variants:
            def op(b, mv, s_, bl=bl):
                if b not in bl: return mv, s_
                cm, cs = cache[b]; return cm[dt][None], cs[dt][None]
            net.op = op
            try: Ls[name][sl] = net.logits(e).float().cpu().numpy()
            finally: net.op = None
    net.run(D['J2'], P, fn, rows=rows)
    p0 = L0s.argmax(1); c0 = p0 == y; W = M.boot_W(N, 1000, SEED + 15)
    res = dict(N=int(N), rows=f'{PERC} random sample-2 jets per class (default_rng({SEED + 13}))',
               ranking="P54's frozen all-class ranking (sum over k of mean |attribution|, ranked on test_20M _106 sample 1)",
               resample='donor = batch neighbour in jet pT (P54 protocol); kept channels live, others from the donor clean run',
               acc_unedited=M.boot_mean(c0, W), variants={})
    for name in Ls:
        p = Ls[name].argmax(1); c1 = p == y
        r = dict(acc=M.boot_mean(c1, W), same_call=float((p == p0).mean()), mcnemar_vs_unedited=M.mcnemar(c0, c1),
                 recall={c: float((p[y == i] == i).mean()) for i, c in enumerate(M.CL)})
        if name.startswith('all384') or name.startswith('only_block0') or name.startswith('blocks1'): r['confusion'] = M.confusion(y, p).tolist()
        res['variants'][name] = r
        log(f"circuit ceiling {name:28s}: acc {100 * r['acc'][0]:.2f} % [{100 * r['acc'][1]:.2f}, {100 * r['acc'][2]:.2f}], same call {r['same_call']:.3f}")
    out['circuit_ceiling'] = res


# --------------------------------------------------------------------------------------- calibration with matched controls
def sec_calib_v(net, D, P, out):
    import torch
    from interp.part_core import kin, build, jmass, sub
    dev = net.dev; y2 = D['y2']; rows = np.where((y2 == M.iW) | (y2 == M.iZ))[0]
    J = sub(D['J2'], rows); pt, de, dp, ms = kin(J); Jb = build(J, pt, de, dp, ms); Jd = build(J, pt, de * 1.05, dp * 1.05, ms)
    dlogm = float(np.mean(np.log(jmass(Jd) + 1e-3) - np.log(jmass(Jb) + 1e-3)))
    S = D['S_mv'][:, M.OBS.index('log_m')]; top = np.argsort(-S); g = np.random.default_rng(SEED + 16)
    sets = [('all', np.arange(NBLK * NC), 'chan', None)]
    R0u = D['rots'][0][0]; Srot = out.get('_S_rot_raw0')
    for K in (8, 32):
        tk = top[:K]; sets.append((f'top{K}_logm', tk, 'chan', None))
        for r in range(10): sets.append((f'random{r}_K{K}', g.choice(NBLK * NC, K, replace=False), 'chan', None))
        blk = tk // NC
        for r in range(10):
            sel = []
            for b in np.unique(blk):
                n = int((blk == b).sum()); sel += list(b * NC + g.choice(NC, n, replace=False))
            sets.append((f'blockmatched{r}_K{K}', np.array(sel), 'chan', None))
        for i, (Rt, St) in D['_tied'].items(): sets.append((f'tied{i}_top{K}_logm', np.argsort(-St)[:K], 'dir', Rt))
        if Srot is not None: sets.append((f'untied_rot0_top{K}_logm', np.argsort(-Srot)[:K], 'dir', R0u))
    N = len(rows); dd = {k: np.zeros(N) for k in ['dil'] + [s[0] for s in sets]}
    with torch.no_grad():
        for s0 in range(0, N, P['BS']):
            sl = np.arange(s0, min(N, s0 + P['BS'])); ed = net.embed(Jd, sl); cache = {}
            net.rec = lambda b, mv, s_: cache.__setitem__(b, mv[0].clone())
            try: Ld = net.logits(ed).float()
            finally: net.rec = None
            Td = net.meta['T']; eb = net.embed(Jb, sl); assert net.meta['T'] == Td
            Lb = net.logits(eb).float(); d0 = Lb[:, M.iW] - Lb[:, M.iZ]; dd['dil'][sl] = ((Ld[:, M.iW] - Ld[:, M.iZ]) - d0).cpu().numpy()
            for name, idx, kind, Rm in sets:
                if kind == 'chan':
                    km = torch.zeros(NBLK, NC, dtype=torch.bool, device=dev); it = torch.as_tensor(idx, device=dev); km[it // NC, it % NC] = True

                    def op(b, mv, s_, km=km):
                        return torch.where(km[b][None, :, None], cache[b], mv[0])[None], s_
                else:
                    Ub = {}
                    for u in idx: Ub.setdefault(int(u) // NC, []).append(Rm[int(u) // NC, int(u) % NC])
                    Ub = {b: torch.stack(v, 1) for b, v in Ub.items()}

                    def op(b, mv, s_, Ub=Ub):
                        if b not in Ub: return mv, s_
                        U = Ub[b]; h = mv[0]; return (h + torch.einsum('ck,dk,tdm->tcm', U, U, cache[b] - h))[None], s_
                net.op = op
                try: Lp = net.logits(eb).float()
                finally: net.op = None
                dd[name][sl] = ((Lp[:, M.iW] - Lp[:, M.iZ]) - d0).cpu().numpy()
    W = M.boot_W(N, 1000, SEED + 17); den = dd['dil']; bden = W @ den
    res = dict(N=int(N), jets='true W and Z jets of sample 2 (test_20M _107)', edit="E4: every particle's deta, dphi x 1.05 about the stored axis",
               dlogm_mean=dlogm, dd_dilation=M.boot_mean(den, W), sets={})
    Rb = {}
    for name, idx, kind, _ in sets:
        num = dd[name]; Rb[name] = (W @ num) / bden
        res['sets'][name] = dict(R=float(num.sum() / den.sum()), R_ci=[float(np.percentile(Rb[name], 2.5)), float(np.percentile(Rb[name], 97.5))],
                                 units=[f'b{int(u) // NC}c{int(u) % NC}' for u in idx[:32]] if name != 'all' else 'all 384')
    for K in (8, 32):
        t = res['sets'][f'top{K}_logm']['R']; summ = dict(top=t, top_ci=res['sets'][f'top{K}_logm']['R_ci'], top_blocks=np.bincount(top[:K] // NC, minlength=NBLK).tolist())
        for fam, keys in (('random', [f'random{r}_K{K}' for r in range(10)]), ('blockmatched', [f'blockmatched{r}_K{K}' for r in range(10)]),
                          ('tied_dirs', [f'tied{i}_top{K}_logm' for i in range(3)])):
            v = [res['sets'][k]['R'] for k in keys]; diff = Rb[f'top{K}_logm'] - np.mean([Rb[k] for k in keys], 0)
            summ[fam] = dict(mean=float(np.mean(v)), min=float(np.min(v)), max=float(np.max(v)), n_ge_top=int(sum(x >= t for x in v)),
                             diff_top_minus_mean=float(t - np.mean(v)), diff_ci=[float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))])
        summ['untied_rot0'] = res['sets'].get(f'untied_rot0_top{K}_logm', {}).get('R')
        res[f'K{K}'] = summ
        log(f"calib K{K}: top {t:.3f} | random {summ['random']['mean']:.3f} [{summ['random']['min']:.3f},{summ['random']['max']:.3f}] | "
            f"block-matched {summ['blockmatched']['mean']:.3f} [{summ['blockmatched']['min']:.3f},{summ['blockmatched']['max']:.3f}] | "
            f"tied dirs {summ['tied_dirs']['mean']:.3f} [{summ['tied_dirs']['min']:.3f},{summ['tied_dirs']['max']:.3f}] | untied rot0 {summ['untied_rot0']}")
    log(f"calib all 384: {res['sets']['all']['R']:.3f}")
    out['calib_v'] = res


def main():
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; log('device', dev, torch.__version__)
    P = M.params(False); P['NSTEER'] = 100
    out = dict(node='P54v', sample=f'JetClass test_20M <class>_107.root, 2,000 random rows per class (default_rng({SEED})), split at random '
                                   f'within class into sample 1 / sample 2 (1,000 per class each)', checkpoint=M.CKPT, params={k: v for k, v in P.items()})
    J1, T1, J2, T2 = M.read_sample(2000, np.random.default_rng(SEED))
    from interp.part_core import sample
    D = dict(J1=J1, J2=J2, y1=J1['label'].astype(int), y2=J2['label'].astype(int),
             O1=np.nan_to_num(M.observables(J1, T1, dev)), O2=np.nan_to_num(M.observables(J2, T2, dev)))
    J0 = sample(300, 3000); log(f'samples read: {len(D["y1"])} + {len(D["y2"])}')
    nets = dict(trained=M.Net('trained', dev)); nets['untrained'] = M.Net('untrained', dev); net = nets['trained']; add_pregate_hooks(net)
    L = net.run(J0, P); out['P1_sample'] = dict(N=int(len(L)), acc=float((L.argmax(1) == J0['label'].astype(int)).mean()))
    log(f"P1 sample accuracy {100 * out['P1_sample']['acc']:.2f} % (expected 83.73 %)")
    Ot1 = torch.from_numpy(D['O1']).float().to(dev); Ot2 = torch.from_numpy(D['O2']).float().to(dev)
    D['Y1z'], D['Y2z'], _ = M.gauss(Ot1, Ot2); D['yt1'] = torch.from_numpy(D['y1']).long().to(dev); D['yt2'] = torch.from_numpy(D['y2']).long().to(dev)
    arr = {}
    for name, f in (('atlas', lambda: M.sec_atlas(nets, D, P, out, arr)), ('extra', lambda: sec_extra(net, D, P, out)),
                    ('calib', lambda: sec_calib_v(net, D, P, out)), ('circuit', lambda: sec_circuit_ceiling(net, D, P, out)),
                    ('steer', lambda: M.sec_steer(net, D, P, out, arr)), ('ablate', lambda: sec_ablate_block(net, D, P, out))):
        if name not in SECS: continue
        t = time.time()
        try:
            f()
            if name == 'atlas':
                out['_S_rot_raw0'] = arr['S_mv_rot_raw0'][:, M.OBS.index('log_m')]
                Ot1 = torch.from_numpy(D['O1']).float().to(dev); D['Y1z'], D['Y2z'], _ = M.gauss(Ot1, Ot2)
        except Exception:
            out[f'{name}_error'] = traceback.format_exc(); log(f'ERROR in {name}:\n' + out[f'{name}_error'])
        log(f'section {name} {time.time() - t:.0f}s'); dump(out)
    log('done')


if __name__ == '__main__':
    main()
