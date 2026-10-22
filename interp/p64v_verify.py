"""P64 verifier (loop C): fresh-sample rerun of P64's purity and graph-rewiring results for ParticleNet, plus one missing control.

Fresh sample: val_5M *_123 (P64 used *_124) of QCD / Z / W / top, 2,500 random rows per class, numpy default_rng(6490), N 10,000.
Prong labels as P64 (exclusive-kT, E-scheme, no beam distance; N = 2 subjets for W/Z, 3 for top/QCD; p64.excl_kt).

Reruns (same code paths as P64: p64.run, p64.purity, p64.Graph): unedited accuracy; EC1/EC2/EC3 edge purity for PN_full,
PN_kin and untrained PN_full (torch seed 6464); arms R1, R2, R3r1, R3r2, R3n2 on EC2, EC3 and both.

New control (rotated partition): every particle gets the subjet label of the original particle nearest to its position rotated
by -90 degrees about the jet's pT centroid in (deta, dphi), i.e. the same exclusive-kT partition turned by 90 degrees. This is a
spatially coherent partition with prong-like region sizes that does not follow the prongs. R1rot / R2rot / R3r1rot / R3r2rot are
the P64 arms run with these labels (R3r*rot count-matched to R*rot). If the R2 > R3r2 excess for top/QCD comes from cutting a
graph along prong boundaries, R2/R3r2 should exceed R2rot/R3r2rot; if it comes from cutting along any spatial boundary, the two
ratios should be equal. The rotated partition is not size-matched (its regions are often unequal), so a second fake
partition 'sec' is size-matched: particles are ordered by their angle about the pT centroid, starting at the direction of the
highest-pT subjet's centroid, and cut into consecutive angular sectors with exactly the true subjet sizes (for W/Z this is the
half-plane split along the prong axis, which cuts each prong through its centre). Arms R1sec / R2sec / R3r1sec / R3r2sec.
Also the purity of each layer's graph with respect to the rotated partition (trained vs untrained).

Distance diagnostic: mean feature-space rank (1 = nearest other particle) of the neighbours each arm puts into the graph, to show
how far R2 / R3r2 / R3n2 / R2rot move the neighbour sets in the layer's own feature distance.

  ./sync.sh && ./part_job.sh c-p64-verify 'python -m interp.p64v_verify --smoke && python -m interp.p64v_verify'
  -> $AI_RESULTS/part/p64_verify[_smoke].json
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp import p64_pn_knn_prongs as p64

SMOKE = '--smoke' in sys.argv
TAG = '_smoke' if SMOKE else ''
RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'
FN = RES / f'p64_verify{TAG}.json'
NPC = 50 if SMOKE else 2500
FILE = 123
FAKES = ('rot', 'sec')
ARMS = ('R1', 'R2', 'R3r1', 'R3r2', 'R3n2') + tuple(a + f for f in FAKES for a in ('R1', 'R2', 'R3r1', 'R3r2'))
LAYERSETS = dict(ec2=(1,), ec3=(2,), both=(1, 2))
iQ, iZ, iW, iT = 0, 6, 7, 8
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


def load_sample():
    from interp.part_core import read_root, PRE, cat, build, kin
    import uproot
    rng = np.random.default_rng(6490); parts_all, info = [], []
    for c, lab, _ in p64.CLS:
        path = p64.VAL / f'{PRE[c]}_{FILE}.root'; N = uproot.open(path)['tree'].num_entries
        if SMOKE: N = min(N, 20000)
        rows = np.sort(rng.choice(N, NPC, replace=False)); parts = []
        for s in range(0, N, 25000):
            r = rows[(rows >= s) & (rows < s + 25000)]
            if len(r):
                J = read_root(path, start=s, stop=min(s + 25000, N)); parts.append({k: J[k][r - s] for k in p64.KEEP})
        J = cat(parts); assert (J['label'] == lab).all(); parts_all.append(J)
        info.append(dict(file=path.name, rows_in_file=int(N), n=int(len(rows)))); log(f'  {path.name}: {len(rows)} random rows of {N}')
    J = cat(parts_all); return build(J, *kin(J)), info


def rot_labels(J, lab):
    """label of the original particle nearest to each particle's position rotated by -90 deg about the pT centroid"""
    from interp.part_core import kin
    pt, de, dp, _ = kin(J); real = pt > 0; out = np.full_like(lab, -1)
    w = pt / np.maximum(pt.sum(1, keepdims=True), 1e-12); ce, cp = (w * de).sum(1, keepdims=True), (w * dp).sum(1, keepdims=True)
    for s in range(0, len(lab), 500):
        sl = slice(s, s + 500); x, y = de[sl] - ce[sl], dp[sl] - cp[sl]; qx, qy = y, -x
        d2 = (qx[:, :, None] - x[:, None, :]) ** 2 + (qy[:, :, None] - y[:, None, :]) ** 2
        d2 = np.where(real[sl][:, None, :], d2, np.inf); j = d2.argmin(-1)
        o = np.take_along_axis(lab[sl], j, 1); o = np.where(real[sl] & (lab[sl] >= 0).any(1, keepdims=True), o, -1); out[sl] = o
    return out


def sec_labels(J, lab):
    """size-matched angular sectors about the pT centroid, starting at the highest-pT subjet's direction"""
    from interp.part_core import kin
    pt, de, dp, _ = kin(J); real = pt > 0; out = np.full_like(lab, -1)
    for b in range(len(lab)):
        r = real[b] & (lab[b] >= 0)
        if not r.any(): continue
        p, x, y, l = pt[b, r], de[b, r], dp[b, r], lab[b, r]; w = p / p.sum(); x, y = x - (w * x).sum(), y - (w * y).sum()
        labs = np.unique(l); lead = labs[np.argmax([p[l == a].sum() for a in labs])]; sl = l == lead
        t0 = np.arctan2((p[sl] * y[sl]).sum(), (p[sl] * x[sl]).sum()); ph = (np.arctan2(y, x) - t0) % (2 * np.pi)
        o = np.argsort(ph, kind='stable'); new = np.empty(len(l), np.int64); st = 0
        for a in labs:
            na = int((l == a).sum()); new[o[st:st + na]] = a; st += na
        out[b, np.where(r)[0]] = new
    return out


STATS = {'cur': None}
_ORIG_EDIT = p64.edit_graph


def edit_graph_rec(arm, pts, idx, idx1, B, gen):
    import torch
    new, nr = _ORIG_EDIT(arm, pts, idx, idx1, B, gen)
    if STATS['cur'] is not None:
        x = pts.double(); xx = (x ** 2).sum(1); dist = xx[:, :, None] + xx[:, None, :] - 2 * torch.einsum('ncp,ncq->npq', x, x)
        rank = p64.rank_rows(dist); N, P, k = new.shape; real = B['m'][:, 0] > 0
        rn = real.gather(1, new.reshape(N, -1)).view(N, P, k) & real[:, :, None]
        ro = real.gather(1, idx.reshape(N, -1)).view(N, P, k) & real[:, :, None]
        big = (real.sum(1) >= 40)[:, None, None]   # jets with >= 40 particles, where every arm has enough candidates
        STATS.setdefault(STATS['cur'], [0.0, 0, 0.0, 0])
        a = STATS[STATS['cur']]
        a[0] += float((rank.gather(-1, new).double() * (rn & big)).sum()); a[1] += int((rn & big).sum())
        a[2] += float((rank.gather(-1, idx).double() * (ro & big)).sum()); a[3] += int((ro & big).sum())
    return new, nr


p64.edit_graph = edit_graph_rec


def boot_ratio_of_ratios(I, fa, fb, fc, fd, sel):
    v = []
    for ii in I:
        s = sel[ii]; A, Bm, C, D = (f[ii][s].mean() for f in (fa, fb, fc, fd)); v.append((A / max(Bm, 1e-12)) / max(C / max(D, 1e-12), 1e-12))
    v = np.array(v); s = sel
    pt = (fa[s].mean() / fb[s].mean()) / max(fc[s].mean() / max(fd[s].mean(), 1e-12), 1e-12)
    lv = np.log(np.maximum(v, 1e-12)); p = float(min(1.0, 2 * min((lv <= 0).mean(), (lv >= 0).mean()) + 1.0 / len(v)))
    return [float(pt), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), p]


def main():
    import torch
    from interp import pn_net
    from jetdistill.part.network import FEATURES
    RES.mkdir(parents=True, exist_ok=True); dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f'device {dev} torch {torch.__version__}')
    J, info = load_sample(); y = J['label'].astype(int); n = len(y)
    lab2, lab3 = p64.excl_kt(J, dev); prim = np.where(np.isin(y, (iW, iZ))[:, None], lab2, lab3); LABS = dict(prong=prim, rot=rot_labels(J, prim), sec=sec_labels(J, prim)); real = J['mask'] > 0
    res = dict(node='P64v', smoke=SMOKE, N=n, files=info, NB=p64.NB, models={}, fake_partitions={})
    for fk in FAKES:
        F = LABS[fk]; agree = ((F == prim) & real).sum(1) / np.maximum(real.sum(1), 1)
        ngr = np.array([len(np.unique(r[(r >= 0)])) for r in F])
        big = np.array([max(((F[b] == a) & real[b]).sum() for a in range(3)) for b in range(n)]) / np.maximum(real.sum(1), 1)
        bigt = np.array([max(((prim[b] == a) & real[b]).sum() for a in range(3)) for b in range(n)]) / np.maximum(real.sum(1), 1)
        res['fake_partitions'][fk] = dict(mean_agreement_with_prongs=float(agree.mean()),
                                          agreement_by_class={c: float(agree[y == l].mean()) for c, l, _ in p64.CLS},
                                          frac_jets_one_group=float((ngr <= 1).mean()), largest_group_share=float(big.mean()),
                                          largest_group_share_prongs=float(bigt.mean()))
        log(f'partition {fk}', res['fake_partitions'][fk])
    bt = p64.Boot(n); sel = p64.pair_sel(y); groups = dict(WZ=np.isin(y, (iW, iZ)), top=y == iT, QCD=y == iQ)
    nets = {}
    for mname, nn_ in (('PN_full', 'full'), ('PN_kin', 'kin')):
        nf = len(FEATURES[nn_]); net = pn_net.load_particlenet(p64.PNDIR / f'ParticleNet_{nn_}.pt', nf, dev)
        for p in net.parameters(): p.requires_grad_(False)
        nets[mname] = (net, nn_)
    torch.manual_seed(6464); un = pn_net.ParticleNet(input_dims=len(FEATURES['full']), **pn_net.PN_CFG).to(dev).eval()
    for p in un.parameters(): p.requires_grad_(False)
    nets['untrained_s6464'] = (un, 'full')
    PUR = {}
    for mname, (net, nn_) in nets.items():
        TB = p64.batches(J, nn_, dev, prim); Ls = []; PP = {k: [dict() for _ in range(3)] for k in LABS}; off = 0
        with torch.no_grad():
            for B in TB:
                L, idxs, _ = p64.run(net, B, rec=True); Ls.append(L.double().cpu().numpy()); nb = len(L)
                for lk, LA in LABS.items():
                    rl = torch.from_numpy(LA[off:off + nb]).to(dev)
                    for li in range(3):
                        d, _ = p64.purity(idxs[li], B, lab=rl)
                        for a, v in d.items(): PP[lk][li].setdefault(a, []).append(v)
                off += nb
        L0 = np.concatenate(Ls); PP = {lk: [{a: np.concatenate(v) for a, v in d.items()} for d in P] for lk, P in PP.items()}
        PUR[mname] = PP
        R = dict(acc_sample=float((L0.argmax(1) == y).mean()), acc_per_class={c: float((L0[y == l].argmax(1) == l).mean()) for c, l, _ in p64.CLS})
        pur = {}
        for lname, P in PP.items():
            for li in range(3):
                for g, s in groups.items():
                    s = s & P[li]['valid']; m, _ = bt.mean(P[li]['u'], s); r_, _ = bt.mean(P[li]['ru'], s)
                    row = dict(u=m, rand=r_[0])
                    if li > 0:
                        d1 = bt.mean(P[li]['u'] - P[0]['u'], s); row['minus_ec1'] = d1[0]; row['p_minus_ec1'] = p64.pboot(d1[1])
                    pur[f'{lname}_ec{li + 1}_{g}'] = row
        R['purity'] = pur
        log(f'{mname}: acc {R["acc_sample"]:.4f}; purity prong EC1/2/3 WZ ' + '/'.join(f'{pur[f"prong_ec{l}_WZ"]["u"][0]:.3f}' for l in (1, 2, 3)) +
            ' top ' + '/'.join(f'{pur[f"prong_ec{l}_top"]["u"][0]:.3f}' for l in (1, 2, 3)) +
            f' rand WZ {pur["prong_ec1_WZ"]["rand"]:.3f} top {pur["prong_ec1_top"]["rand"]:.3f}' + ''.join(
            f'; {fk} EC1/2/3 WZ ' + '/'.join(f'{pur[f"{fk}_ec{l}_WZ"]["u"][0]:.3f}' for l in (1, 2, 3)) + ' top ' +
            '/'.join(f'{pur[f"{fk}_ec{l}_top"]["u"][0]:.3f}' for l in (1, 2, 3)) + f' rand WZ {pur[f"{fk}_ec1_WZ"]["rand"]:.3f} top {pur[f"{fk}_ec1_top"]["rand"]:.3f}'
            for fk in FAKES))
        if mname.startswith('untrained'):
            res['models'][mname] = R; del TB; continue
        FL, arms = {}, {}
        for ls, layers in LAYERSETS.items():
            for arm in ARMS:
                fk = next((f for f in FAKES if arm.endswith(f)), None); base = arm[:-3] if fk else arm; g = torch.Generator(device=dev).manual_seed(6467); Ls, Rr = [], [[] for _ in layers]
                STATS['cur'] = f'{mname}|{arm}-{ls}'; off = 0
                with torch.no_grad():
                    for B in TB:
                        nb = B['m'].shape[0]; Bx = dict(B)
                        if fk: Bx['lab'] = torch.from_numpy(LABS[fk][off:off + nb]).to(dev)
                        L, _, reps = p64.run(net, Bx, arm=base, layers=layers, gen=g); Ls.append(L.double().cpu().numpy()); off += nb
                        rr = (B['m'][:, 0] > 0)
                        for q, nr in enumerate(reps): Rr[q].append((nr.float() * rr).sum().item() / rr.sum().item() / 16.0)
                STATS['cur'] = None
                L = np.concatenate(Ls); f = p64.flips(L0, L, y); FL[f'{arm}-{ls}'] = f
                st = STATS[f'{mname}|{arm}-{ls}']
                r = dict(acc=float((L.argmax(1) == y).mean()), frac_edges_replaced={f'ec{li + 1}': float(np.mean(v)) for li, v in zip(layers, Rr)},
                         mean_rank_new_neighbours=st[0] / max(st[1], 1), mean_rank_original_neighbours=st[2] / max(st[3], 1))
                for c, sl in sel.items(): r[f'changed_{c}'] = bt.mean(f[c], sl)[0]
                arms[f'{arm}-{ls}'] = r
                log(f'  {mname} {arm:8s}-{ls:4s} acc {r["acc"]:.4f} repl {r["frac_edges_replaced"]} rank new {r["mean_rank_new_neighbours"]:.1f} '
                    f'(orig {r["mean_rank_original_neighbours"]:.1f}) chg argmax {r["changed_argmax"][0]:.4f} WZ {r["changed_WZ"][0]:.4f} tQ {r["changed_tQ"][0]:.4f}')
        tests = {}
        for ls in LAYERSETS:
            for a, c in (('R2', 'R3r2'), ('R1', 'R3r1'), ('R2', 'R3n2'), ('R2rot', 'R3r2rot'), ('R1rot', 'R3r1rot'), ('R2sec', 'R3r2sec'), ('R1sec', 'R3r1sec')):
                for t, sl in sel.items():
                    mc = p64.mcnemar(FL[f'{a}-{ls}'][t][sl], FL[f'{c}-{ls}'][t][sl]); mc['ratio'] = bt.ratio(FL[f'{a}-{ls}'][t], FL[f'{c}-{ls}'][t], sl)
                    tests[f'{a}-{ls} vs {c}-{ls} [{t}]'] = mc
            for t, sl in sel.items():   # (R2 / R3r2) / (R2fake / R3r2fake): prong partition vs fake partition
                for fk in FAKES:
                    tests[f'ratio-of-ratios R2 vs R2{fk} {ls} [{t}]'] = dict(rr=boot_ratio_of_ratios(bt.I, *(FL[f'{x}-{ls}'][t].astype(float) for x in ('R2', 'R3r2', 'R2' + fk, 'R3r2' + fk)), sl))
                    log(f'  {mname} {ls} [{t}] R2/R3r2 {np.round(tests[f"R2-{ls} vs R3r2-{ls} [{t}]"]["ratio"], 3)}  R2{fk}/R3r2{fk} {np.round(tests[f"R2{fk}-{ls} vs R3r2{fk}-{ls} [{t}]"]["ratio"], 3)}  '
                        f'ratio of ratios {np.round(tests[f"ratio-of-ratios R2 vs R2{fk} {ls} [{t}]"]["rr"], 3)}')
        R['arms'], R['tests'] = arms, tests; res['models'][mname] = R
        json.dump(res, open(FN, 'w'), indent=1, default=float); del TB; torch.cuda.empty_cache()
    keys = [(m, k) for m in res['models'] for k in res['models'][m].get('tests', {}) if 'p' in res['models'][m]['tests'][k]]
    for (m, k), h in zip(keys, p64.holm([res['models'][m]['tests'][k]['p'] for m, k in keys])): res['models'][m]['tests'][k]['holm'] = h
    keys = [(m, k) for m in res['models'] for k in res['models'][m].get('tests', {}) if 'rr' in res['models'][m]['tests'][k]]
    for (m, k), h in zip(keys, p64.holm([res['models'][m]['tests'][k]['rr'][3] for m, k in keys])): res['models'][m]['tests'][k]['holm'] = h
    # trained - untrained EC3 purity (prong and rotated labels)
    tu = {}
    for mname in ('PN_full', 'PN_kin'):
        for lname in LABS:
            for g, s in groups.items():
                a, b = PUR[mname][lname][2], PUR['untrained_s6464'][lname][2]; ss = s & a['valid'] & b['valid']
                d = bt.mean(a['u'] - b['u'], ss); tu[f'{mname}_{lname}_ec3_{g}'] = d[0] + [p64.pboot(d[1])]
    res['trained_minus_untrained_ec3'] = tu; log('trained - untrained EC3', tu)
    json.dump(res, open(FN, 'w'), indent=1, default=float); log('done', FN)


if __name__ == '__main__':
    main()
