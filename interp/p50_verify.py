"""P50 verify (loop C): re-run of P50's EdgeConv-1 substitution on a fresh sample, plus two missing controls.

Fresh sample: val_5M *_124 (P50 used *_123), evaluation 1,000 random rows per class (default_rng(6060)), fit 500 random rows per
class from the remaining rows (default_rng(6061)), 80/20 split by jet (default_rng(6061)), 35 % of real particles kept as centres
(record()'s torch seed 5052, on the new jets). Models PN_kin (main) and PN_full. EC1 only (P50's claim is about EC1).

Arms (all EC1 messages replaced by an MLP fit, rest of the network intact):
  main fits (width 256, 12,000 steps, as P50's primary run): A, B, C, F (P50's definitions) and
  D = A + (BN'd deltaR_j - deltaR_i): the neighbour's distance to the jet axis relative to i's, without the signed 2-d direction.
  ladder (matched-fit-quality control): B and F (which contain the orientation) fitted with small MLPs / few steps so that their
  held-out R2 spans A's; A also on the ladder. If R depends only on fit quality, A lies on the B/F ladder's R-vs-R2 line; if
  removing the orientation specifically lowers R, A lies below it.
Edits and R as in P50 (phi + 0.06 rigid shift vs matched dilation, d = l_W - l_Z, W and Z jets).
  ./sync.sh && ./part_job.sh c-p50-verify 'python -m interp.p50_verify'  -> $AI_RESULTS/part/p50_verify.json
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import interp.p50_pn_edge_function as P

RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'
FN = RES / 'p50_verify.json'
FILE_ID = '124'
MAIN = [('A', 256, 12000), ('B', 256, 12000), ('C', 256, 12000), ('D', 256, 12000), ('F', 256, 12000)]
LADDER = [(k, w, s) for k in ('B', 'F') for (w, s) in ((8, 150), (16, 300), (16, 1500), (32, 1500), (64, 3000))] + \
         [('A', 16, 300), ('A', 64, 3000)]
T0 = time.time()


def log(*a): print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


_comp0 = P.components


def components2(M, cin, idx, B, xb, gen=None):
    out = _comp0(M, cin, idx, B, xb, gen=gen)
    if cin.shape[1] == xb.shape[1]:      # layer 1 only: neighbour's deltaR minus centre's (BN'd)
        r = xb[:, [M.ir]]; out['dr'] = P.gather(r, idx) - r.unsqueeze(-1)
    return out


P.components = components2
P.COMPOSE['D'] = ('cen', 'inv', 'dr')


def load_samples():
    from interp.part_core import read_root, PRE, CL, cat, build, kin
    import uproot
    re, rf = np.random.default_rng(6060), np.random.default_rng(6061); E, F, info = [], [], []
    for c in CL:
        path = P.VAL / f'{PRE[c]}_{FILE_ID}.root'; N = uproot.open(path)['tree'].num_entries
        ev = np.sort(re.choice(N, P.NEV, replace=False)); rest = np.setdiff1d(np.arange(N), ev)
        ft = np.sort(rf.choice(rest, P.NFIT, replace=False)); rows = np.union1d(ev, ft); parts = []
        for s in range(0, N, 25000):
            r = rows[(rows >= s) & (rows < s + 25000)]
            if len(r):
                J = read_root(path, start=s, stop=min(s + 25000, N)); parts.append({k: J[k][r - s] for k in P.KEEP})
        J = cat(parts); ie = np.isin(rows, ev)
        E.append({k: v[ie] for k, v in J.items()}); F.append({k: v[~ie] for k, v in J.items()})
        info.append(dict(file=path.name, rows_in_file=int(N), eval=int(ie.sum()), fit=int((~ie).sum())))
        log(f'  {path.name}: {ie.sum()} eval + {(~ie).sum()} fit')
    reb = lambda J: build(J, *kin(J))
    return reb(cat(E)), reb(cat(F)), info


def fit_mlp(Dl, kind, width, steps, seed):
    import torch
    X = torch.cat([Dl[c] for c in P.COMPOSE[kind]], 1); Y = Dl['y']; tr = Dl['train']
    Xtr, Ytr, Xte, Yte = P.flat(X[tr]), P.flat(Y[tr]), P.flat(X[~tr]), P.flat(Y[~tr])
    mu = Xtr.mean(0); sd = Xtr.std(0).clamp(min=1e-3); torch.manual_seed(seed); d, c = X.shape[1], Y.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(d, width), torch.nn.ReLU(), torch.nn.Linear(width, width), torch.nn.ReLU(),
                              torch.nn.Linear(width, c)).to(X.device)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    Ztr = (Xtr - mu) / sd; g = torch.Generator(device='cpu').manual_seed(seed)
    for st in range(steps):
        b = torch.randint(0, len(Ztr), (4096,), generator=g).to(X.device)
        loss = ((net(Ztr[b]) - Ytr[b]) ** 2).mean(); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
    net.eval()
    for p in net.parameters(): p.requires_grad_(False)
    f = P.Fit(kind, 'mlp', mu, sd, mlp=net)
    with torch.no_grad(): Pr = f.predict(Xte)
    rr = P.r2(Yte, Pr)
    for lev in rr: rr[lev].pop('per_channel')
    return f, rr


def main():
    import torch
    from interp import pn_net
    from interp.part_core import sample, build, kin
    from jetdistill.part.network import FEATURES
    RES.mkdir(parents=True, exist_ok=True); dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    log('device', dev, torch.cuda.get_device_name(0) if dev == 'cuda' else '')
    JE, JF, info = load_samples(); C, chk = P.edits(JE); log('edit check', chk)
    rng = np.random.default_rng(6061); perm = rng.permutation(len(JF['label'])); train_jets = np.sort(perm[: int(0.8 * len(perm))])
    res = dict(node='P50v', files=info, edit_check=chk, N_eval=int(len(JE['label'])), N_fit=int(len(JF['label'])), models={})
    st = P.Stats(JE['label'].astype(int))
    for mname, n in (('PN_kin', 'kin'), ('PN_full', 'full')):
        nf = len(FEATURES[n]); net = pn_net.load_particlenet(P.PNDIR / f'ParticleNet_{n}.pt', nf, dev)
        for p in net.parameters(): p.requires_grad_(False)
        M = P.Model(mname, n, net, dev); names = [f[0] for f in FEATURES[n]]; M.ir = names.index('deltaR')
        TB = P.batches_of(C, n, dev); L0 = P.logits(M, TB); R = dict(unedited=st.row(L0))
        u = R['unedited']; log(f'{mname}: acc {u["acc"]:.4f} R_phi {u["R_phi_WZ"]:.3f} {u["R_phi_WZ_ci"]} num {u["dd_phi_mean"]:.3f} den {u["dd_dil_phi_mean"]:.3f}')
        D = P.record(M, JF, train_jets)  # record() seeds its centre-choice generator with 5052; the jets are new
        log(f'{mname}: recorded EC1 {tuple(D[0]["y"].shape)}')
        R['arms'] = {}; ARML = {}; boots = {}
        for q, (kd, w, s) in enumerate(MAIN + LADDER):
            nm = f'{kd}-w{w}-s{s}'; t1 = time.time()
            f, rr = fit_mlp(D[0], kd, w, s, 6100 + q)
            La = P.logits(M, TB, {0: f}); r = st.row(La, ref=L0); r['fit'] = rr; r['kind'] = kd; r['width'] = w; r['steps'] = s
            ARML[nm] = La; boots[nm] = st.R(La, 'phi', 'WZ')[1]; R['arms'][nm] = r
            log(f'  {mname} {nm:14s} R2 edge {rr["edge"]["mean"]:.3f} agg {rr["agg"]["mean"]:.3f} | acc {r["acc"]:.4f} ({r["dacc_vs_unedited"][0]*100:+.2f}) '
                f'R_phi {r["R_phi_WZ"]:.3f} {np.round(r["R_phi_WZ_ci"], 3).tolist()} num {r["dd_phi_mean"]:.3f} den {r["dd_dil_phi_mean"]:.3f} '
                f'dR p {r["p_dR_phi_WZ_vs_unedited"]:.3g} ({time.time() - t1:.0f} s)')
            res['models'][mname] = R; json.dump(res, open(FN, 'w'), indent=1, default=float)
        del D; torch.cuda.empty_cache()
        # between-arm tests
        R['between'] = {}
        for a, b in (('A-w256-s12000', 'B-w256-s12000'), ('A-w256-s12000', 'D-w256-s12000'), ('A-w256-s12000', 'C-w256-s12000'),
                     ('D-w256-s12000', 'B-w256-s12000')):
            dR, p = st.perm_dR(ARML[b], ARML[a], 'phi', 'WZ')
            R['between'][f'{b} - {a}'] = dict(dR=[dR] + P.ci(boots[b] - boots[a]), p=p, dacc=R['arms'][b]['acc'] - R['arms'][a]['acc'])
        # matched-fit-quality control: line R ~ R2(agg) over the B/F arms (orientation present); A's excess over the line
        ori = [k for k, v in R['arms'].items() if v['kind'] in ('B', 'F')]
        for lev in ('edge', 'agg'):
            x = np.array([R['arms'][k]['fit'][lev]['mean'] for k in ori]); Y = np.stack([boots[k] for k in ori], 1)
            y0 = np.array([R['arms'][k]['R_phi_WZ'] for k in ori])
            out = {}
            for a in [k for k, v in R['arms'].items() if v['kind'] in ('A', 'C', 'D')]:
                xa = R['arms'][a]['fit'][lev]['mean']
                c0 = np.polyfit(x, y0, 1); exc0 = R['arms'][a]['R_phi_WZ'] - np.polyval(c0, xa)
                exb = np.array([boots[a][i] - np.polyval(np.polyfit(x, Y[i], 1), xa) for i in range(len(Y))])
                out[a] = dict(R2=xa, R=R['arms'][a]['R_phi_WZ'], R_pred_from_oriented_line=float(np.polyval(c0, xa)),
                              excess=[float(exc0)] + P.ci(exb), p_boot_two_sided=float(min(1.0, 2 * min((exb >= 0).mean(), (exb <= 0).mean()))))
            R[f'matched_quality_{lev}'] = dict(line=np.polyfit(x, y0, 1).tolist(), oriented_points=[(k, float(xx), float(yy)) for k, xx, yy in zip(ori, x, y0)], arms=out)
            log(f'{mname} matched-quality ({lev}): line {np.round(np.polyfit(x, y0, 1), 3).tolist()}; ' +
                '; '.join(f'{a}: R {v["R"]:.3f} pred {v["R_pred_from_oriented_line"]:.3f} excess {v["excess"][0]:+.3f} [{v["excess"][1]:+.3f}, {v["excess"][2]:+.3f}] p {v["p_boot_two_sided"]:.3g}' for a, v in out.items()))
        log(f'{mname} between', {k: (round(v['dR'][0], 3), round(v['p'], 4), round(v['dacc'], 4)) for k, v in R['between'].items()})
        res['models'][mname] = R; json.dump(res, open(FN, 'w'), indent=1, default=float)
        del TB, net, M; torch.cuda.empty_cache()
    log('done', FN)


if __name__ == '__main__':
    main()
