"""P45v - verification of P45 (model stitching ParticleNet_full -> ParT_full), loop C.

What it re-runs (fresh data, different seeds):
  training jets: train_100M, 2 random files per class, 15,000 random rows each, default_rng(4747) (P45: 4545); batch order seed 7.
  evaluation: val_5M *_122 (P45: *_121), 2,000 random rows per class, default_rng(4748), N 20,000, rebuilt with part_core.build;
  plus the P1 sample (test *_100 rows 3000-4000 per class, N 10,000) for the reference accuracy.
  forward stitches (ParticleNet -> ParT, as P45): pn1/pn2/pn3 -> block 1 and block 8, u2 (untrained PN EdgeConv-2) -> b1, b8,
  raw (ParT's 17 scaled inputs) -> b1, b8, pnpool -> ParT class token, id -> b1 (code check).
Missing control added here (reverse stitches, ParT -> ParticleNet): ParT tokens after block 1 (pt1) or after block 8 (pt8) are
  mapped by one affine map into the input of PN EdgeConv-2 (64 per particle; PN's EdgeConv-1 skipped) or of EdgeConv-3 (128;
  EdgeConv-1 and -2 skipped); ParT's class-token vector (ptcls, 128) -> PN's pooled vector (256, the input of PN's fc). The
  map is fitted to PN's own activation (least squares) and trained on KL(p_PN || p_stitched). If the recipient sets R, these
  give a PN-sized R (~0.5); if the donor's features set it, a ParT-sized R (~0.8).
Training length check: every map is trained for one pass with P45's schedule (Adam 5e-4, 3 % warm-up, cosine), evaluated, then
  trained for a second pass (Adam 2.5e-4, cosine, other batch order) and evaluated again.
Edits and metrics as in P45 (phi + 0.06 rad on every particle; matched dilation k_j = sqrt(M0M2 shifted / M0M2 base); R = slope
  through 0 of the per-jet change of d = l_W - l_Z under the shift against its change under the dilation, true W/Z/QCD jets);
  NB = 2,000 bootstrap resamples over jets (p resolution 5e-4); Holm over the stitches in each test family.
  ./sync.sh && ./part_job.sh c-p45-verify 'python -m interp.p45v_verify --smoke && python -m interp.p45v_verify'
  -> $AI_RESULTS/part/p45_verify[_smoke].json, p45_verify[_smoke].png
"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp import p45_stitch_pn_part as P   # noqa: E402  (reuses data loading, networks, statistics helpers)

SMOKE = P.SMOKE
TAG = '_smoke' if SMOKE else ''
RES = P.RES
FN, FIG = RES / f'p45_verify{TAG}.json', RES / f'p45_verify{TAG}.png'
NTR_FILE = 300 if SMOKE else 15000
NEV = 100 if SMOKE else 2000
NP1 = 100 if SMOKE else 1000
NLS = 1000 if SMOKE else 20000
NB = 200 if SMOKE else 2000
BS = 256
iZ, iW, iQ = 6, 7, 0
log = P.log

FWD = [('pn1', 1), ('pn2', 1), ('pn3', 1), ('pn1', 8), ('pn2', 8), ('pn3', 8), ('u2', 1), ('u2', 8), ('raw', 1), ('raw', 8),
       ('pnpool', 'cls'), ('id', 1)]
REV = [('pt1', 'ec2'), ('pt8', 'ec2'), ('pt1', 'ec3'), ('pt8', 'ec3'), ('ptcls', 'fc')]
DIN = dict(pn1=64, pn2=128, pn3=256, u2=128, raw=17, pnpool=256, id=128, pt1=128, pt8=128, ptcls=128)
DOUT = dict(ec2=64, ec3=128, fc=256)


def stitches():
    L = [dict(name=f'{s}>b{k}' if k != 'cls' else f'{s}>cls', src=s, k=k, rec='ParT') for s, k in FWD]
    L += [dict(name=f'{s}>{k}', src=s, k=k, rec='PN') for s, k in REV]
    return L


def train_jets():
    from interp.part_core import PRE, CL, cat
    rng = np.random.default_rng(4747); parts, info = [], []
    for c in CL:
        for fi in rng.choice(100, 2, replace=False):
            J, inf = P.load_rows(P.TRAIN / f'{PRE[c]}_{fi:03d}.root', NTR_FILE, rng); parts.append(J); info.append(inf)
            log(f'  train {Path(inf["file"]).name}: {inf["drawn"]} random rows of {inf["rows_in_file"]}')
    return cat(parts), info


def eval_samples():
    from interp.part_core import PRE, CL, cat, sample, kin, build, moments, jmass
    S = {}
    J1 = sample(NP1, 3000); S['P1'] = dict(J=P.rebuild({k: J1[k] for k in P.KEEP}), info=f'test_20M *_100 rows 3000-{3000 + NP1} per class (P1 sample)')
    rng = np.random.default_rng(4748); parts = []
    for c in CL:
        J, _ = P.load_rows(P.VAL / f'{PRE[c]}_122.root', NEV, rng); parts.append(J)
    S['v122'] = dict(J=P.rebuild(cat(parts)), info=f'val_5M *_122, {NEV} random rows per class, default_rng(4748)')
    for nm, s in S.items():
        J = s['J']; pt, de, dp, ms = kin(J); real = pt > 0
        Jp = build(J, pt, de, dp + P.SHIFT * real, ms)
        Mb = moments(*kin(J)[:3])['M0M2']; Mp = moments(*kin(Jp)[:3])['M0M2']; ok = Mb > 0
        kj = np.where(ok, np.sqrt(Mp / np.where(ok, Mb, 1.0)), 1.0)[:, None]
        Jd = build(J, pt, de * kj, dp * kj, ms)
        s['cond'] = dict(base=J, phi=Jp, dil=Jd)
        m0, mp, md = (np.log(np.maximum(jmass(x), 1.0)) for x in (J, Jp, Jd))
        # pairwise check: the phi shift leaves every pairwise (Delta y, Delta phi) unchanged -> the pair-bias input is unchanged
        s['edit_check'] = dict(phi_dlogm_absmax=float(np.abs(mp - m0).max()), dil_dlogm_mean=float((md - m0).mean()), k_mean=float(kj.mean()),
                               mask_equal=bool(np.array_equal(Jp['mask'], J['mask']) and np.array_equal(Jd['mask'], J['mask'])))
        log(f'sample {nm}: N {len(J["label"])}; edit check {s["edit_check"]}')
    return S


def pn_from(N, B, fts, start):
    """ParticleNet from the input of EdgeConv `start` (0-based index 1 = EdgeConv-2, 2 = EdgeConv-3) to the logits"""
    t = N.t; md = N.pn; mask = B['mp']; shift = (mask == 0) * 1e9; fts = fts * mask
    for i in range(start, len(md.edge_convs)): fts = md.edge_convs[i](fts + shift, fts) * mask
    counts = t.clamp(mask.sum(-1), min=1.0); return md.fc(fts.sum(-1) / counts)


class Ctx:
    def __init__(self, N, B):
        t = N.t
        with t.no_grad():
            self.B = B; self.P = B['P']; self.h0, self.am, self.pad = N.part_pre(B)
            o, pool, self.pnL = N.pn_feats(N.pn, B); self.pno = o
            self.src = dict(pn1=o[0], pn2=o[1], pn3=o[2], pnpool=pool)
            uo, _, _ = N.pn_feats(N.upn, B); self.src['u2'] = uo[1]
            self.L, self.xc, self.hs = N.part_from(self.h0, self.am, self.pad, 1, collect=True)   # hs[k-1] = input of block k; hs[8] = output of block 8
            self.src.update(pt1=self.hs[1], pt8=self.hs[8], ptcls=self.xc)

    def source(self, s, k):
        """(P, N, d) per-particle or (N, d) vector"""
        if s in ('pnpool', 'ptcls'): return self.src[s]
        if s == 'id': return self.hs[k - 1]
        if s == 'raw': return self.B['x'].permute(2, 0, 1)
        if s in ('pt1', 'pt8'): return self.src[s]
        return self.src[s][..., :self.P].permute(2, 0, 1)

    def target(self, st):
        k = st['k']
        if k == 'cls': return self.xc
        if k == 'fc': return self.src['pnpool']
        if k == 'ec2': return self.pno[0][..., :self.P].permute(2, 0, 1)
        if k == 'ec3': return self.pno[1][..., :self.P].permute(2, 0, 1)
        return self.hs[k - 1]

    def run(self, N, st, A):
        t = N.t; k = st['k']; X = self.source(st['src'], k)
        if k == 'cls': return N.part.fc(A(X))
        if k == 'fc': return N.pn.fc(A(X))
        if k in ('ec2', 'ec3'):
            f = A(X).permute(1, 2, 0); f = t.nn.functional.pad(f, (0, 128 - self.P))
            return pn_from(N, self.B, f, 1 if k == 'ec2' else 2)
        tok = A(X).masked_fill(self.pad.t().unsqueeze(-1), 0)
        return N.part_from(tok, self.am, self.pad, k)[0]


def fit_init(N, JT, sts, order):
    from interp.part_core import sub
    t = N.t; G, C = {}, {}; n = 0
    with t.no_grad():
        for idx in order:
            if n >= NLS: break
            cx = Ctx(N, N.tensors(sub(JT, idx))); n += len(idx); real = ~cx.pad.t()
            for st in sts:
                if st['src'] == 'id': continue
                X = cx.source(st['src'], st['k']); Y = cx.target(st)
                if X.dim() == 3: X, Y = X[real], Y[real]
                X = t.cat([X.double(), t.ones(len(X), 1, dtype=t.float64, device=X.device)], 1)
                G[st['name']] = G.get(st['name'], 0) + X.T @ X; C[st['name']] = C.get(st['name'], 0) + X.T @ Y.double()
    W = {}
    for k, g in G.items():
        lam = 1e-4 * g.diagonal().mean(); W[k] = t.linalg.solve(g + lam * t.eye(len(g), dtype=g.dtype, device=g.device), C[k])
    return W, n


def make_affine(N, st, Winit):
    t = N.t; din = DIN[st['src']]; dout = DOUT.get(st['k'], 128); A = t.nn.Linear(din, dout).to(N.dev)
    with t.no_grad():
        if st['src'] == 'id': A.weight.copy_(t.eye(128)); A.bias.zero_()
        else: w = Winit[st['name']]; A.weight.copy_(w[:-1].T.float()); A.bias.copy_(w[-1].float())
    return A


def teacher(N, JT):
    from interp.part_core import sub
    t = N.t; mult = JT['mask'].sum(1).astype(int); o = np.argsort(mult, kind='stable'); TL = np.zeros((len(o), 10), np.float32); TP = TL.copy()
    with t.no_grad():
        for i in range(0, len(o), 512):
            idx = o[i:i + 512]; B = N.tensors(sub(JT, idx)); h, am, pad = N.part_pre(B)
            TL[idx] = N.part_from(h, am, pad, 1)[0].float().cpu().numpy(); TP[idx] = N.pn_feats(N.pn, B)[2].float().cpu().numpy()
    return TL, TP


def train_pass(N, JT, TL, TP, sts, As, seed, lr, warmfrac):
    from interp.part_core import sub
    t = N.t; F = t.nn.functional; mult = JT['mask'].sum(1).astype(int); order = P.batches(mult, seed)
    opts = {k: t.optim.Adam(A.parameters(), lr=lr) for k, A in As.items()}
    nst = len(order); warm = max(1, int(warmfrac * nst))
    lrf = lambda s: ((s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, nst - warm)))) if warmfrac > 0 else 0.5 * (1 + math.cos(math.pi * s / nst))
    sch = {k: t.optim.lr_scheduler.LambdaLR(o, lrf) for k, o in opts.items()}; run = {k: [] for k in As}; hist = {k: [] for k in As}; t1 = time.time()
    for A in As.values(): A.train()
    for s, idx in enumerate(order):
        cx = Ctx(N, N.tensors(sub(JT, idx)))
        tl = {'ParT': F.log_softmax(t.from_numpy(TL[idx]).to(N.dev), 1), 'PN': F.log_softmax(t.from_numpy(TP[idx]).to(N.dev), 1)}
        for st in sts:
            k = st['name']; L = cx.run(N, st, As[k]); loss = F.kl_div(F.log_softmax(L, 1), tl[st['rec']], log_target=True, reduction='batchmean')
            opts[k].zero_grad(set_to_none=True); loss.backward(); opts[k].step(); sch[k].step(); run[k].append(float(loss))
        if (s + 1) % max(1, nst // 10) == 0 or s == nst - 1:
            for k in As: hist[k].append([s + 1, float(np.mean(run[k]))]); run[k] = []
            el = time.time() - t1
            log(f'pass seed {seed}: step {s + 1}/{nst} ({el:.0f} s, eta {el / (s + 1) * (nst - s - 1):.0f} s); KL ' + ', '.join(f'{k} {hist[k][-1][1]:.3f}' for k in As))
    for A in As.values(): A.eval()
    return hist


def evaluate(N, S, sts, As):
    from interp.part_core import sub
    t = N.t; Jb = S['cond']['base']; n = len(Jb['label']); o = np.argsort(Jb['mask'].sum(1).astype(int), kind='stable')
    names = ['ParT', 'PN'] + [st['name'] for st in sts]; out = {c: {k: np.zeros((n, 10)) for k in names} for c in S['cond']}
    with t.no_grad():
        for i in range(0, n, 256):
            idx = o[i:i + 256]
            for c, J in S['cond'].items():
                cx = Ctx(N, N.tensors(sub(J, idx)))
                out[c]['ParT'][idx] = cx.L.double().cpu().numpy(); out[c]['PN'][idx] = cx.pnL.double().cpu().numpy()
                for st in sts: out[c][st['name']][idx] = cx.run(N, st, As[st['name']]).double().cpu().numpy()
    return out


PAIRS = [('pn2>b8', 'u2>b8'), ('pn2>b1', 'u2>b1'), ('pn2>b1', 'pn2>b8'), ('pn3>b1', 'pn3>b8'), ('pn1>b8', 'pn2>b8'), ('raw>b1', 'pn2>b1'),
         ('pt1>ec2', 'pt8>ec2'), ('pt1>ec3', 'pt8>ec3')]


def metrics(S, out, sts):
    y = S['cond']['base']['label'].astype(int); n = len(y); rng = np.random.default_rng(47480)
    Wb = np.zeros((NB, n), np.float32)
    for b in range(NB): Wb[b] = np.bincount(rng.integers(0, n, n), minlength=n)
    isW = y == iW; wz = (y == iW) | (y == iZ) | (y == iQ); wsum = Wb.sum(1)
    sl = lambda yv, xv: (float((yv * xv).sum() / (xv * xv).sum()), (Wb[:, wz] @ (yv * xv)) / (Wb[:, wz] @ (xv * xv)))
    d = lambda L: L[:, iW] - L[:, iZ]; base, R = {}, {}
    for k in ['ParT', 'PN'] + [st['name'] for st in sts]:
        L0, Lp, Ld = (out[c][k] for c in ('base', 'phi', 'dil')); ok = (L0.argmax(1) == y).astype(np.float64)
        dp, dd = (d(Lp) - d(L0))[wz], (d(Ld) - d(L0))[wz]; r, rb = sl(dp, dd); c0, c1 = L0.argmax(1)[isW] == iZ, Lp.argmax(1)[isW] == iZ
        cf = np.zeros((10, 10))
        for a, b in zip(y, L0.argmax(1)): cf[a, b] += 1
        cf = cf / np.maximum(cf.sum(1, keepdims=True), 1)
        R[k] = dict(acc=float(ok.mean()), acc_ci=P.ci(Wb @ ok / wsum), R=r, R_ci=P.ci(rb), dd_phi=float(dp.mean()), dd_dil=float(dd.mean()),
                    WtoZ_before=float(c0.mean()), WtoZ_after=float(c1.mean()), WtoZ_mcnemar=P.mcnemar(c0, c1), own_class=np.diag(cf).tolist(), confusion=cf.round(4).tolist())
        base[k] = dict(ok=ok, L0=L0, rb=rb)
    for st in sts:
        k = st['name']; rec = base[st['rec']]; b = base[k]; row = R[k]; g = (Wb @ (rec['ok'] - b['ok'])) / wsum
        row.update(recipient=st['rec'], gap=float(rec['ok'].mean() - b['ok'].mean()), gap_ci=P.ci(g), mcnemar_vs_recipient=P.mcnemar(rec['ok'] > 0, b['ok'] > 0),
                   agree_recipient=float((b['L0'].argmax(1) == rec['L0'].argmax(1)).mean()),
                   KL_recipient=float(np.mean(np.sum(P._sm(rec['L0']) * (P._ls(rec['L0']) - P._ls(b['L0'])), 1))),
                   R_minus_PN=[float(row['R'] - R['PN']['R'])] + P.ci(b['rb'] - base['PN']['rb']), p_R_vs_PN=P.bp(b['rb'] - base['PN']['rb']),
                   R_minus_ParT=[float(row['R'] - R['ParT']['R'])] + P.ci(b['rb'] - base['ParT']['rb']), p_R_vs_ParT=P.bp(b['rb'] - base['ParT']['rb']))
    fam = [st['name'] for st in sts if st['src'] != 'id']
    for key, get in (('holm_gap', lambda r: r['mcnemar_vs_recipient']['p']), ('holm_R_vs_PN', lambda r: r['p_R_vs_PN']), ('holm_R_vs_ParT', lambda r: r['p_R_vs_ParT'])):
        for k, p in zip(fam, P.holm([get(R[k]) for k in fam])): R[k][key] = p
    pairs = {}
    for a, b in PAIRS:
        dr = base[a]['rb'] - base[b]['rb']; pairs[f'{a} - {b}'] = dict(dR=float(R[a]['R'] - R[b]['R']), ci=P.ci(dr), p=P.bp(dr))
    for k, p in zip(list(pairs), P.holm([v['p'] for v in pairs.values()])): pairs[k]['holm'] = p
    R['_pairs'] = pairs
    return R


def plot(res):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for j, ph in enumerate(('pass1', 'pass2')):
        M = res[ph]['v122']['models']; ks = [k for k in M if not k.startswith('_')]; a = ax[j]
        xs = np.arange(len(ks)); Rv = [M[k]['R'] for k in ks]; lo = [M[k]['R'] - M[k]['R_ci'][0] for k in ks]; hi = [M[k]['R_ci'][1] - M[k]['R'] for k in ks]
        col = ['purple' if k == 'ParT' else 'brown' if k == 'PN' else ('#1f77b4' if M[k].get('recipient') == 'ParT' else '#d62728') for k in ks]
        a.bar(xs, Rv, color=col, alpha=0.75); a.errorbar(xs, Rv, yerr=[lo, hi], fmt='none', c='k', lw=0.8)
        for k, x in zip(ks, xs): a.text(x, 0.02, f"{M[k]['acc'] * 100:.1f}%", rotation=90, ha='center', va='bottom', fontsize=6, color='w')
        a.axhline(M['ParT']['R'], c='purple', ls='-.', lw=1); a.axhline(M['PN']['R'], c='brown', ls='-.', lw=1)
        a.set_xticks(xs); a.set_xticklabels(ks, rotation=70, fontsize=7); a.set_ylabel('R (W-Z logit response to phi+0.06 / to matched dilation)')
        a.set_title(f'{ph}: blue = into ParT, red = ParT features into ParticleNet; bar text = 10-class accuracy', fontsize=8)
    fig.suptitle(f'P45 verify: val_5M _122, N {res["pass1"]["v122"]["N"]}; dashed lines: ParT_full (purple), ParticleNet_full (brown)', fontsize=9)
    fig.tight_layout(); fig.savefig(FIG, dpi=130); log('figure', FIG)


def main():
    import torch
    from interp.part_core import sub
    RES.mkdir(parents=True, exist_ok=True); torch.manual_seed(4545)
    N = P.Nets(); log(f'device {N.dev}, {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}')
    res = dict(node='P45v', smoke=SMOKE, NB=NB, NTR_FILE=NTR_FILE, NEV=NEV, NLS=NLS, stitches=stitches())
    S = eval_samples(); sts = stitches()
    J = sub(S['P1']['cond']['base'], np.arange(0, len(S['P1']['cond']['base']['label']), max(1, len(S['P1']['cond']['base']['label']) // 300)))
    B = N.tensors(J)
    with torch.no_grad():
        cx = Ctx(N, B); L2 = N.part(B['x'], B['v'], B['m']); P2 = N.pn(B['pp'].clone(), B['xp'].clone(), B['mp'])
        r2 = pn_from(N, B, cx.pno[0], 1); r3 = pn_from(N, B, cx.pno[1], 2)
    res['identity_check'] = dict(part=float((cx.L - L2).abs().max()), pn=float((cx.pnL - P2).abs().max()), pn_from_ec2=float((r2 - P2).abs().max()),
                                 pn_from_ec3=float((r3 - P2).abs().max()), pn_fc=float((N.pn.fc(cx.src['pnpool']) - P2).abs().max()))
    log('identity check', res['identity_check'])
    JT, info = train_jets(); res['train'] = dict(files=info, N=int(len(JT['label'])))
    TL, TP = teacher(N, JT); res['train']['ParT_acc'] = float((TL.argmax(1) == JT['label']).mean()); res['train']['PN_acc'] = float((TP.argmax(1) == JT['label']).mean())
    log(f'teachers: ParT {res["train"]["ParT_acc"]:.4f} PN {res["train"]["PN_acc"]:.4f} on {len(JT["label"])} training jets')
    mult = JT['mask'].sum(1).astype(int); Winit, nls = fit_init(N, JT, sts, P.batches(mult, 7)); log(f'least squares on {nls} jets')
    As = {st['name']: make_affine(N, st, Winit) for st in sts}
    for ph, (seed, lr, wf) in (('pass1', (7, 5e-4, 0.03)), ('pass2', (8, 2.5e-4, 0.0))):
        res['train'][ph] = train_pass(N, JT, TL, TP, sts, As, seed, lr, wf); res[ph] = {}
        for nm, s in S.items():
            M = metrics(s, evaluate(N, s, sts, As), sts)
            res[ph][nm] = dict(info=s['info'], N=int(len(s['cond']['base']['label'])), edit_check=s['edit_check'], models=M)
            log(f'== {ph} {nm}: ParT acc {M["ParT"]["acc"]:.4f} R {M["ParT"]["R"]:.3f} [{M["ParT"]["R_ci"][0]:.3f}, {M["ParT"]["R_ci"][1]:.3f}] '
                f'W->Z {M["ParT"]["WtoZ_before"]:.3f}->{M["ParT"]["WtoZ_after"]:.3f}; PN acc {M["PN"]["acc"]:.4f} R {M["PN"]["R"]:.3f} [{M["PN"]["R_ci"][0]:.3f}, {M["PN"]["R_ci"][1]:.3f}] '
                f'W->Z {M["PN"]["WtoZ_before"]:.3f}->{M["PN"]["WtoZ_after"]:.3f}')
            for st in sts:
                r = M[st['name']]
                log(f'  {st["name"]:10s} acc {r["acc"]:.4f} gap({r["recipient"]}) {r["gap"] * 100:+.2f} [{r["gap_ci"][0] * 100:+.2f}, {r["gap_ci"][1] * 100:+.2f}] '
                    f'KL {r["KL_recipient"]:.3f} | R {r["R"]:.3f} [{r["R_ci"][0]:.3f}, {r["R_ci"][1]:.3f}] dphi {r["dd_phi"]:+.3f} ddil {r["dd_dil"]:+.3f} '
                    f'W->Z {r["WtoZ_before"]:.3f}->{r["WtoZ_after"]:.3f} | Holm R-PN {r.get("holm_R_vs_PN", float("nan")):.2g} R-ParT {r.get("holm_R_vs_ParT", float("nan")):.2g}')
            for k, v in M['_pairs'].items(): log(f'  pair {k}: dR {v["dR"]:+.3f} [{v["ci"][0]:+.3f}, {v["ci"][1]:+.3f}] p {v["p"]:.2g} Holm {v["holm"]:.2g}')
            json.dump(res, open(FN, 'w'), indent=1, default=float)
    plot(res); log('done', FN)


if __name__ == '__main__':
    main()
