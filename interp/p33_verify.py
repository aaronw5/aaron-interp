"""P33 verification (loop C, round C7): fresh sample, linearised-network control, tighter neighbour-holding move, head
sufficiency, float32 call flips.

Sample: JetClass test_20M WToQQ_117 + ZToQQ_117 (P33 used _116), PER random rows each (numpy default_rng(7117), sorted indices,
never first-k). Build as P33 (part_core.build, eta = jet_eta + sign(jet_eta) deta, pT kept). Pair features always in float64
(P33 'pair64'), cast to float32 before the pair embedding. Per jet, for each of the 10 leading stored particles:
  s+/s-   radial move by +-0.01 rad;  s3+/s3- by +-0.003 rad
  c+/c-   particle + every particle within dR < 0.05 moved by the same 0.01 rad radial vector (P33 cluster move)
  k+/k-   same with radius 0.01 (only the neighbours closer than the step are carried)
Conditions: 'pair' = unedited per-particle inputs, edited float64 pair features (P33 pair64);
  'lin' = first-order (linearised) network: d_lin = sum_{c,i,j} dS/dF_cij|unedited * (F_cij(edited) - F_cij(unedited)), the gradient
          taken once per jet at the unedited jet. Its even part is produced only by the nonlinearity of the pair features
          (ln dR, ln kT, ln m2) in the particle coordinates; the network enters only through its fixed gradient.
  on s+/s- rows: KOh (head h's pair bias reset to unedited), ONLY (only the P33 top heads get the edited bias; full: 0, 2; kin:
          2, 3, 4), KOTOP (those heads reset).
Input alone: E_in = sum_j even(Delta ln dR_ij)^2 (P33b) for s, s3, c, k moves.
Float32 calls: 10-class argmax with float32 pair features (as deployed) vs float64 pair features, on the W/Z sample and on a
10-class sample (NFLIP random rows per class, files _117, default_rng(7118)); McNemar exact test on correctness changes.
Read-out: d = change of logit_Z - logit_W vs the unedited jet in the same condition. Even (d+ + d-)/2, odd (d+ - d-)/2.
  ./sync.sh && ./part_job.sh c-p33-verify 'python -m interp.p33_verify'
  -> $AI_RESULTS/part/p33_verify.json
"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_argv = sys.argv; sys.argv = [_argv[0]]
import interp.p33_pair_channels as P            # noqa: E402  (functions only; its argparse sees no flags)
sys.argv = _argv
SMOKE = '--smoke' in sys.argv
SEED = 7117; PER = 6 if SMOKE else int(os.environ.get('P33V_PER', 800)); NFLIP = 10 if SMOKE else 150; NB = 100 if SMOKE else 1000
DR, DR3, R1, R2, NTOP, BS, JCH = 0.01, 0.003, 0.05, 0.01, 10, 128, 16
TEST = Path('/j-jepa-vol/JetClass/Pythia/test_20M')
OUTD = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'; FN = OUTD / f"p33_verify{'_smoke' if SMOKE else os.environ.get('P33V_TAG', '')}.json"
ET = ['s+', 's-', 's3+', 's3-', 'c+', 'c-', 'k+', 'k-']
TOP = dict(ParT_full=[0, 2], ParT_kin=[2, 3, 4])
T0 = time.time()


def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def load(fn, n, rng, limit=None):
    path = TEST / fn
    if SMOKE or limit:
        limit = 25000 if SMOKE else limit
        import uproot
        from interp.part_core import read_root, sub
        N = uproot.open(path)['tree'].num_entries; rows = np.sort(rng.choice(min(N, limit), n, replace=False))
        return sub(read_root(path, 0, min(N, limit)), rows)
    return P.load_rows(path, n, rng)[0]


def plan(pt, de, dp):
    """rows: (jet, etype, particle, nn, cs05, cs01) and edited (deta, dphi)"""
    meta, DE, DP = [], [], []
    for j in range(len(pt)):
        m = pt[j] > 0; n = int(m.sum()); e0, p0 = de[j].astype(np.float64), dp[j].astype(np.float64)
        for i in range(min(NTOP, n)):
            ue, up = P.radial_unit(e0[i], p0[i]); dist = np.hypot(e0 - e0[i], p0 - p0[i]); dist[~m] = np.inf
            c1, c2 = np.where(dist < R1)[0], np.where(dist < R2)[0]; d2 = dist.copy(); d2[i] = np.inf; nn = float(d2.min())
            for t, d, sel in (('s+', DR, [i]), ('s-', -DR, [i]), ('s3+', DR3, [i]), ('s3-', -DR3, [i]),
                              ('c+', DR, c1), ('c-', -DR, c1), ('k+', DR, c2), ('k-', -DR, c2)):
                e, p = e0.copy(), p0.copy(); e[sel] += d * ue; p[sel] += d * up
                meta.append((j, ET.index(t), i, nn, len(c1), len(c2))); DE.append(e); DP.append(p)
    return np.array(meta, np.float64), np.stack(DE), np.stack(DP)


def input_terms(pt, de, dp, meta):
    """E_in, O_in per (s-row) particle for s, s3, c, k moves: sum over changed pairs of even/odd(Delta ln dR)^2"""
    I = np.where(meta[:, 1] == 0)[0]; out = np.zeros((len(I), 8))
    for r, (j, i) in enumerate(meta[I][:, [0, 2]].astype(int)):
        m = pt[j] > 0; e0, p0 = de[j][m].astype(np.float64), dp[j][m].astype(np.float64)
        rr = math.hypot(e0[i], p0[i]); ue, up = (e0[i] / rr, p0[i] / rr) if rr > 1e-6 else (1.0, 0.0)
        dist = np.hypot(e0 - e0[i], p0 - p0[i]); res = []
        for step, rad in ((DR, 0), (DR3, 0), (DR, R1), (DR, R2)):
            cl = np.where(dist < rad)[0] if rad else np.array([i]); ot = np.setdiff1d(np.arange(len(e0)), cl)
            if not len(ot): res += [0.0, 0.0]; continue
            d0 = np.hypot(e0[cl, None] - e0[None, ot], p0[cl, None] - p0[None, ot]); l0 = np.log(np.maximum(d0, 1e-8))
            Dm = lambda s: np.log(np.maximum(np.hypot(e0[cl, None] + s * ue - e0[None, ot], p0[cl, None] + s * up - p0[None, ot]), 1e-8)) - l0
            a, b = Dm(step), Dm(-step); res += [(((a + b) / 2) ** 2).sum(), (((a - b) / 2) ** 2).sum()]
        out[r] = res
    return out


def collect(name, J, iW, iZ):
    import torch
    from interp.part_core import kin, build, sub
    mod, feat, dev = P.load_model(name); R = P.Runner(mod, dev); top = TOP[name]
    N = len(J['label']); pt, de, dp, ms = kin(J); f8 = lambda a: np.asarray(a, np.float64)
    J0 = build(J, f8(pt), f8(de), f8(dp), f8(ms)); CONDS = ['pair', 'lin'] + [f'KO{h}' for h in range(8)] + ['ONLY', 'KOTOP']
    META, D, L32, L64 = [], [], np.zeros((N, 10), np.float32), np.zeros((N, 10), np.float32)
    for c0 in range(0, N, JCH):
        jj = np.arange(c0, min(c0 + JCH, N)); P_ = int(J0['mask'][jj].sum(1).max())
        meta, DE, DP = plan(pt[jj], de[jj], dp[jj]); jl = meta[:, 0].astype(int)
        xb, vb, mb = (R.T(a) for a in P.inputs_fixed(sub(J0, jj), feat, P_))
        F64b = R.feats(R.T(P.v64(sub(J, jj), f8(pt[jj]), f8(de[jj]), f8(dp[jj]), f8(ms[jj]), P_)))     # float64
        Fb = F64b.float().clone().requires_grad_(True)
        with torch.enable_grad():
            Lb = R.fwd(xb, R.embed(Fb)[0], mb); sb = Lb[:, iZ] - Lb[:, iW]; g = torch.autograd.grad(sb.sum(), Fb)[0].double()
        with torch.no_grad():
            Ub = R.embed(F64b.float())[0]; sb = sb.detach()
            L64[jj] = Lb.detach().cpu().numpy(); L32[jj] = R.fwd(xb, R.embed(R.feats(vb))[0], mb).cpu().numpy()
            Jr = sub(J, jj[jl]); d = np.full((len(jl), len(CONDS)), np.nan)
            for b0 in range(0, len(jl), BS):
                sl = np.arange(len(jl))[b0:b0 + BS]; Jb = sub(Jr, sl); p_, _, _, m_ = kin(Jb); gi = R.T(jl[sl]).long()
                E = build(Jb, f8(p_), DE[sl], DP[sl], f8(m_))
                assert np.array_equal(E['mask'], J0['mask'][jj[jl[sl]]])
                F64e = R.feats(R.T(P.v64(Jb, f8(p_), DE[sl], DP[sl], f8(m_), P_)))
                Ue = R.embed(F64e.float())[0]; base = sb[gi]
                d[sl, 0] = ((lambda L: L[:, iZ] - L[:, iW])(R.fwd(xb[gi], Ue, mb[gi])) - base).cpu().numpy()
                d[sl, 1] = (g[gi] * (F64e - F64b[gi])).sum((1, 2, 3)).cpu().numpy()
                et = meta[sl, 1].astype(int); sel = np.where(et <= 1)[0]
                if len(sel):
                    t = R.T(sel).long(); Ubs = Ub[gi[t]]; Ues = Ue[t]
                    for k, (kind, hs) in enumerate([('KO', [h]) for h in range(8)] + [('ONLY', top), ('KO', top)]):
                        Uk = Ues.clone()
                        if kind == 'KO': Uk[:, hs] = Ubs[:, hs]
                        else: Uk = Ubs.clone(); Uk[:, hs] = Ues[:, hs]
                        L = R.fwd(xb[gi[t]], Uk, mb[gi[t]]); d[sl[sel], 2 + k] = (L[:, iZ] - L[:, iW] - base[t]).cpu().numpy()
            meta[:, 0] = jj[jl]; META.append(meta); D.append(d)
        if (c0 // JCH) % 10 == 0: log(f'{name}: {c0 + len(jj)} / {N} jets')
    del mod; torch.cuda.empty_cache()
    return np.concatenate(META), np.concatenate(D), CONDS, L32, L64, (pt, de, dp)


def mcnemar(a, b):
    """exact two-sided McNemar on paired correctness a, b (bool arrays)"""
    from scipy.stats import binomtest
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    return dict(correct32_only=n01, correct64_only=n10, p=float(binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0)


def analyse(name, meta, d, CONDS, kinp, y):
    pt, de, dp = kinp; N = len(y); C = {c: k for k, c in enumerate(CONDS)}
    et = meta[:, 1].astype(int); idx = {t: np.where(et == k)[0] for k, t in enumerate(ET)}
    I = idx['s+']; jr = meta[I, 0].astype(int); nn, cs1, cs2 = meta[I, 3], meta[I, 4], meta[I, 5]
    for t in ET: assert np.array_equal(meta[idx[t], 0], meta[I, 0]) and np.array_equal(meta[idx[t], 2], meta[I, 2])
    G = input_terms(pt, de, dp, meta); log(f'{name}: input terms done')
    rng = np.random.default_rng(SEED + 5); Wb = np.zeros((NB, N))
    for b in range(NB): Wb[b] = np.bincount(rng.integers(0, N, N), minlength=N)
    S = lambda v, m=None: np.bincount(jr if m is None else jr[m], v if m is None else v[m], minlength=N)
    def rat(num, den, m=None, H0=None):
        a, b = S(num, m), S(den, m); bs = (Wb @ a) / (Wb @ b); o = dict(v=float(a.sum() / b.sum()), ci=[float(q) for q in np.nanpercentile(bs, [2.5, 97.5])])
        if H0 is not None: o['H0'] = H0; o['p_boot'] = float(max(1 / NB, min(1, 2 * min(np.nanmean(bs <= H0), np.nanmean(bs >= H0)))))
        return o, bs
    ev = lambda t, k: (d[idx[t + '+'], k] + d[idx[t + '-'], k]) / 2
    od = lambda t, k: (d[idx[t + '+'], k] - d[idx[t + '-'], k]) / 2
    R = dict(n_jets=int(N), n_particles=int(len(I)), frac_nn_lt_0_01=float((nn < 0.01).mean()), median_nn=float(np.median(nn)))
    def smooth(e1, o1, e3, o3, m=None):
        a1, b1, a3, b3 = (S(v, m) for v in (e1, o1, e3, o3)); v = (a3.sum() / b3.sum()) / (a1.sum() / b1.sum())
        bs = ((Wb @ a3) / (Wb @ b3)) / ((Wb @ a1) / (Wb @ b1)); return dict(v=float(v), ci=[float(q) for q in np.nanpercentile(bs, [2.5, 97.5])]), bs
    bins = [('all', None)] + [(f'nn_{lo}_{hi}', (nn >= lo) & (nn < hi)) for lo, hi in zip([0, 0.003, 0.01, 0.03], [0.003, 0.01, 0.03, np.inf])]
    sm = {}
    for lab, m in bins:
        o = {}
        net, bn = smooth(*(f(t, C['pair']) ** 2 for t, f in (('s', ev), ('s', od), ('s3', ev), ('s3', od))), m=m)
        lin, bl = smooth(*(f(t, C['lin']) ** 2 for t, f in (('s', ev), ('s', od), ('s3', ev), ('s3', od))), m=m)
        inp, _ = smooth(G[:, 0], G[:, 1], G[:, 2], G[:, 3], m=m)
        dd = bn - bl; o = dict(n=int(len(I) if m is None else m.sum()), net=net, lin=lin, inp=inp,
                               net_minus_lin=dict(v=net['v'] - lin['v'], ci=[float(q) for q in np.nanpercentile(dd, [2.5, 97.5])],
                                                  p_boot_two_sided=float(max(1 / NB, 2 * min(np.nanmean(dd <= 0), np.nanmean(dd >= 0))))))
        e, el = ev('s', C['pair']), ev('s', C['lin'])
        o['lin_even_ms_over_net'] = rat(el ** 2, e ** 2, m)[0]
        o['lin_even_R2'] = float(1 - ((e - el)[m if m is not None else slice(None)] ** 2).sum() / (e[m if m is not None else slice(None)] ** 2).sum())
        oo, ol = od('s', C['pair']), od('s', C['lin']); mm = m if m is not None else slice(None)
        o['lin_odd_R2'] = float(1 - ((oo - ol)[mm] ** 2).sum() / (oo[mm] ** 2).sum()); o['lin_odd_ms_over_net'] = float((ol[mm] ** 2).sum() / (oo[mm] ** 2).sum())
        o3, o3l = od('s3', C['pair']), od('s3', C['lin']); o['lin_odd_R2_0.003'] = float(1 - ((o3 - o3l)[mm] ** 2).sum() / (o3[mm] ** 2).sum())
        o['net_even_ms_share'] = float((e[m] ** 2).sum() / (e ** 2).sum()) if m is not None else 1.0
        sm[lab] = o
    R['smoothness'] = sm
    e = ev('s', C['pair']); R['even_ms_share_nn_lt_0_01'] = rat(e ** 2 * (nn < 0.01), e ** 2)[0]
    cl = {}
    for tag, m, gi in (('R0.05_cs>=2', cs1 >= 2, 4), ('R0.01_cs>=2', cs2 >= 2, 6)):
        t = 'c' if tag.startswith('R0.05') else 'k'
        cl[tag] = dict(n=int(m.sum()), net=rat(ev(t, C['pair']) ** 2, ev('s', C['pair']) ** 2, m, 1.0)[0],
                       lin=rat(ev(t, C['lin']) ** 2, ev('s', C['lin']) ** 2, m)[0], inp=rat(G[:, gi], G[:, 0], m)[0])
    R['cluster_over_single_even'] = cl
    ep = e; Spp = S(ep ** 2); hd = {}
    for h in range(8):
        k = C[f'KO{h}']; a = S(ep * (ep - ev('s', k))); bs = (Wb @ a) / (Wb @ Spp)
        hd[h] = dict(proj_share=dict(v=float(a.sum() / Spp.sum()), ci=[float(q) for q in np.percentile(bs, [2.5, 97.5])]),
                     p_one_sided_le_1_8=float(max(1 / NB, (bs <= 1 / 8).mean())))
    from interp.p33_pair_channels import holm
    for h, p in zip(range(8), holm([hd[h]['p_one_sided_le_1_8'] for h in range(8)])): hd[h]['p_holm8'] = p
    R['heads'] = {str(h): v for h, v in hd.items()}
    for c in ('ONLY', 'KOTOP'):
        R[f'top_heads_{c}_even_proj_share'] = rat(ep * ev('s', C[c]), ep ** 2)[0]
        R[f'top_heads_{c}_even_ms_over_pair'] = rat(ev('s', C[c]) ** 2, ep ** 2)[0]
    o = od('s', C['pair']); R['top_heads_ONLY_odd_proj_share'] = rat(o * od('s', C['ONLY']), o ** 2)[0]
    R['top_heads'] = TOP[name]
    log(f"{name}: smooth net {sm['all']['net']['v']:.3f} lin {sm['all']['lin']['v']:.3f} inp {sm['all']['inp']['v']:.3f}; cluster net "
        f"{cl['R0.05_cs>=2']['net']['v']:.3f} R0.01 {cl['R0.01_cs>=2']['net']['v']:.3f}; ONLY {R['top_heads_ONLY_even_proj_share']['v']:.3f}")
    return R


def main():
    from interp.part_core import kin, sub, cat
    from jetdistill.config import CLASSES as CL
    OUTD.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(SEED)
    J = cat([load(f'{p}_117.root', PER, rng) for p in ('WToQQ', 'ZToQQ')]); J = sub(J, np.where((kin(J)[0] > 0).sum(1) >= 2)[0])
    y = J['label'].astype(int); iW, iZ = CL.index('Wqq'), CL.index('Zqq'); log(f'W/Z sample {len(y)} jets')
    rng2 = np.random.default_rng(SEED + 1)
    from interp.part_core import PRE
    JF = cat([load(f'{PRE[c]}_117.root', NFLIP, rng2, limit=25000) for c in CL]); yF = JF['label'].astype(int); log(f'10-class sample {len(yF)} jets')
    res = dict(settings=dict(seed=SEED, per_class=PER, files='W/ZToQQ_117', nflip_per_class=NFLIP, dr=DR, dr3=DR3, radii=[R1, R2], NB=NB,
                             top_heads_from_P33=TOP), models={})
    for name in ('ParT_full', 'ParT_kin'):
        meta, d, CONDS, L32, L64, kinp = collect(name, J, iW, iZ)
        r = analyse(name, meta, d, CONDS, kinp, y)
        r['acc_WZ_float32'] = float((L32.argmax(1) == y).mean()); r['acc_WZ_float64'] = float((L64.argmax(1) == y).mean())
        r['flips_WZ'] = dict(n=int((L32.argmax(1) != L64.argmax(1)).sum()), N=int(len(y)), **mcnemar(L32.argmax(1) == y, L64.argmax(1) == y))
        # 10-class flips, float32 vs float64 pair features
        from interp.part_core import build
        mod, feat, dev = P.load_model(name); Rn = P.Runner(mod, dev); pt, de, dp, ms = kin(JF); f8 = lambda a: np.asarray(a, np.float64)
        J0 = build(JF, f8(pt), f8(de), f8(dp), f8(ms)); a32, a64 = np.zeros(len(yF), int), np.zeros(len(yF), int)
        import torch
        with torch.no_grad():
            for c0 in range(0, len(yF), 64):
                jj = np.arange(c0, min(c0 + 64, len(yF))); P_ = int(J0['mask'][jj].sum(1).max())
                xb, vb, mb = (Rn.T(a) for a in P.inputs_fixed(sub(J0, jj), feat, P_))
                F64 = Rn.feats(Rn.T(P.v64(sub(JF, jj), f8(pt[jj]), f8(de[jj]), f8(dp[jj]), f8(ms[jj]), P_))).float()
                a32[jj] = Rn.fwd(xb, Rn.embed(Rn.feats(vb))[0], mb).argmax(1).cpu().numpy()
                a64[jj] = Rn.fwd(xb, Rn.embed(F64)[0], mb).argmax(1).cpu().numpy()
        del mod; torch.cuda.empty_cache()
        r['flips_10class'] = dict(acc32=float((a32 == yF).mean()), acc64=float((a64 == yF).mean()), n=int((a32 != a64).sum()), N=int(len(yF)),
                                  **mcnemar(a32 == yF, a64 == yF))
        log(f"{name}: acc W/Z {r['acc_WZ_float32']:.4f} (f64 {r['acc_WZ_float64']:.4f}); flips W/Z {r['flips_WZ']['n']}, 10-class {r['flips_10class']}")
        res['models'][name] = r
    FN.write_text(json.dumps(P.clean(res), indent=1)); log(f'wrote {FN}')


if __name__ == '__main__':
    main()
