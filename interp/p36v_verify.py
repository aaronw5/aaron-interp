"""P36v — verification of P36 (natural pT-centroid offset, recentring at fixed invariant mass) on a fresh sample, with two
symmetry nulls that P36 lacked and an on-manifold check of the recentred jets.

Fresh sample: JetClass test_20M <class>_119.root (P36 used _118), numpy default_rng(3690) (P36: 3636), random rows (never
first-k): 8,000 Wqq and 8,000 Zqq jets (4x P36's W/Z count, so tail5 holds 400 W + 400 Z) and 1,000 jets of each other class.
Models: ParT_full, ParT_kin, ParticleNet_full, ParticleNet_kin (released weights). L-GATr and the untrained ParT not rerun.
Selection and edits as in P36 (same code): c = pT-weighted mean (deta, dphi) about the stored axis; per-class tails top 1/2/5 %
of |c|; rec = shift by -c; ref = shift by +c; rs = shift by |c| in a random direction; rot = rotation about the axis by
U(60, 300) deg; dil_e = angles scaled by sqrt(M2_e / M2) (matched real mass change).
New nulls (both keep pT, M0, M2, |c|, all pair dR and the invariant mass):
  mirp  dphi -> -dphi (reflection in phi about the jet axis; a symmetry of the physics; also leaves every particle energy fixed)
  flip  (deta, dphi) -> (-deta, -dphi) (180 deg rotation about the axis; invariant mass exact for massless particles)
On-manifold check: offset of the constituent four-momentum-sum direction (E-scheme axis) from the stored axis, in the stored
(deta, dphi) coordinates, before and after each edit.
Read-outs and statistics as P36: W-vs-Z log-odds on true W/Z jets, kappa (pooled matched dilations), Rk(e) = mean d(e) /
(kappa mean 1/2 dlog M2(e)), bootstrap over jets (2,000), sign-flip permutation (5,000), McNemar on call changes, Holm over models.
  ./sync.sh && ./part_job.sh c-p36-verify 'python -m interp.p36v_verify'
  -> $AI_RESULTS/part/p36_verify.json (+ p36_verify_jets.npz)
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('JETDISTILL_TAGGER', 'part')
SMOKE = '--smoke' in sys.argv
SEED, SEED_EDIT = 3690, 3691
PER = {c: (8000 if c in ('Wqq', 'Zqq') else 1000) for c in ['QCD', 'Hbb', 'Hcc', 'Hgg', 'H4q', 'Hqql', 'Zqq', 'Wqq', 'Tbqq', 'Tbl']}
if SMOKE: PER = {c: 200 for c in PER}
NB, NPERM = (300, 300) if SMOKE else (2000, 5000)
TAILS = {'tail1': 0.01, 'tail2': 0.02, 'tail5': 0.05}
TEST = Path('/j-jepa-vol/JetClass/Pythia/test_20M')
PRE = {'QCD': 'ZJetsToNuNu', 'Hbb': 'HToBB', 'Hcc': 'HToCC', 'Hgg': 'HToGG', 'H4q': 'HToWW4Q', 'Hqql': 'HToWW2Q1L', 'Zqq': 'ZToQQ',
       'Wqq': 'WToQQ', 'Tbqq': 'TTBar', 'Tbl': 'TTBarLep'}
CL = ['QCD', 'Hbb', 'Hcc', 'Hgg', 'H4q', 'Hqql', 'Zqq', 'Wqq', 'Tbqq', 'Tbl']
iW, iZ = CL.index('Wqq'), CL.index('Zqq')
EDITS = ['base', 'rec', 'ref', 'rs', 'rot', 'mirp', 'flip', 'dil_rec', 'dil_ref', 'dil_rs']
RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'
TAG = '_smoke' if SMOKE else ''
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def load_sample():
    import uproot
    from interp.part_core import read_root, sub, cat
    rng = np.random.default_rng(SEED); parts = []
    for k, c in enumerate(CL):
        path = TEST / f'{PRE[c]}_119.root'; N = uproot.open(path)['tree'].num_entries
        rows = np.sort(rng.choice(N, PER[c], replace=False)); ch = []
        for s in range(0, N, 25000):
            r = rows[(rows >= s) & (rows < s + 25000)]
            if len(r): ch.append(sub(read_root(path, s, min(s + 25000, N)), r - s))
        J = cat(ch); assert np.all(J['label'] == k); parts.append(J); log(f'read {c}: {PER[c]} random rows of {path.name} ({N})')
    J = cat(parts); keep = (J['mask'] > 0).sum(1) >= 2
    return sub(J, np.where(keep)[0])


def selection(J):
    from interp.part_core import kin, jmass
    pt, de, dp, ms = kin(J); p = pt.astype(np.float64)
    M0 = p.sum(1); M1e = (p * de).sum(1); M1p = (p * dp).sum(1); M2 = (p * (de.astype(np.float64) ** 2 + dp.astype(np.float64) ** 2)).sum(1)
    ce, cp = M1e / M0, M1p / M0; cabs = np.hypot(ce, cp); y = J['label'].astype(int); n = len(y)
    S = dict(ce=ce, cp=cp, cabs=cabs, M0=M0, M2=M2, y=y, m=jmass(J).astype(np.float64))
    for t, q in TAILS.items():
        f = np.zeros(n, bool)
        for k in range(10):
            ik = np.where(y == k)[0]; nt = max(1, int(round(q * len(ik)))); f[ik[np.argsort(-cabs[ik], kind='stable')[:nt]]] = True
        S[t] = f
    r = np.random.default_rng(SEED_EDIT); S['theta'] = np.radians(r.uniform(60, 300, n)); S['alpha'] = r.uniform(0, 2 * np.pi, n)
    S['M2_rec'] = M2 - 2 * (ce * M1e + cp * M1p) + cabs ** 2 * M0
    S['M2_ref'] = M2 + 2 * (ce * M1e + cp * M1p) + cabs ** 2 * M0
    S['rse'], S['rsp'] = cabs * np.cos(S['alpha']), cabs * np.sin(S['alpha'])
    S['M2_rs'] = M2 + 2 * (S['rse'] * M1e + S['rsp'] * M1p) + cabs ** 2 * M0
    return S, (pt, de, dp, ms)


def make_edit(J, K, S, name):
    from interp.part_core import build
    pt, de, dp, ms = K; r = (pt > 0).astype(np.float32); col = lambda a: a[:, None].astype(np.float32)
    if name == 'base': return build(J, pt, de, dp, ms)
    if name == 'rec': return build(J, pt, de - col(S['ce']) * r, dp - col(S['cp']) * r, ms)
    if name == 'ref': return build(J, pt, de + col(S['ce']) * r, dp + col(S['cp']) * r, ms)
    if name == 'rs': return build(J, pt, de + col(S['rse']) * r, dp + col(S['rsp']) * r, ms)
    if name == 'rot':
        co, si = col(np.cos(S['theta'])), col(np.sin(S['theta'])); return build(J, pt, (co * de - si * dp) * r, (si * de + co * dp) * r, ms)
    if name == 'mirp': return build(J, pt, de, -dp, ms)
    if name == 'flip': return build(J, pt, -de, -dp, ms)
    if name.startswith('dil_'):
        k = col(np.sqrt(np.maximum(S['M2_' + name[4:]], 1e-30) / S['M2'])); return build(J, pt, de * k, dp * k, ms)
    raise ValueError(name)


def escheme_offset(J):
    """offset of the four-momentum-sum direction from the stored axis, in stored (deta, dphi) coordinates"""
    m = J['mask'] > 0; px, py, pz = ((J[k].astype(np.float64) * m).sum(1) for k in ('px', 'py', 'pz'))
    eta = np.arcsinh(pz / np.hypot(px, py)); phi = np.arctan2(py, px)
    sg = np.where(J['jet_eta'] < 0, -1.0, 1.0)
    return sg * (eta - J['jet_eta'].astype(np.float64)), np.angle(np.exp(1j * (phi - J['jet_phi'].astype(np.float64))))


def main():
    import torch
    from interp.part_core import ParTNetwork, inputs, sample, kin, build, jmass
    from interp import pn_net
    from jetdistill.part.network import FEATURES
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; log('device', dev)
    M = {}
    for feat in ('full', 'kin'):
        net = ParTNetwork(feat); net.model.to(dev).eval(); M[f'ParT_{feat}'] = dict(kind='part', feat=feat, model=net.model)
    PND = Path(os.environ.get('JETDISTILL_MODELS', '/j-jepa-vol/aaron-interp/models/part')).parent / 'particlenet'
    for feat in ('full', 'kin'):
        M[f'PN_{feat}'] = dict(kind='pn', feat=feat, model=pn_net.load_particlenet(PND / f'ParticleNet_{feat}.pt', len(FEATURES[feat]), dev))
    names = list(M)

    def run(J, bs=512):
        N = len(J['mask']); order = np.argsort((J['mask'] > 0).sum(1), kind='stable'); L = {k: np.zeros((N, 10), np.float32) for k in names}
        T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
        for s in range(0, N, bs):
            idx = order[s:s + bs]; Jc = {k: (a[idx] if np.ndim(a) else a) for k, a in J.items()}
            X = {f: inputs(Jc, f) for f in ('full', 'kin')}
            for k in names:
                mm = M[k]; x, v, msk = X[mm['feat']]
                with torch.no_grad():
                    if mm['kind'] == 'part': out = mm['model'](T(x), T(v), T(msk))
                    else:
                        P = x.shape[-1]; pad = (lambda a: np.concatenate([a, np.zeros(a.shape[:2] + (128 - P,), np.float32)], -1)) if P < 128 else (lambda a: a)
                        xp, mp = pad(x), pad(msk); fn = [f[0] for f in FEATURES[mm['feat']]]
                        out = mm['model'](T(xp[:, [fn.index('deta'), fn.index('dphi')]]), T(xp), T(mp))
                L[k][idx] = out.float().cpu().numpy()
        return L

    J1 = sample(100 if SMOKE else 1000, 3000); y1 = J1['label'].astype(int); L1 = run(build(J1, *kin(J1)))
    acc_p1 = {k: float((L1[k].argmax(1) == y1).mean()) for k in names}; log('P1-sample accuracy', acc_p1, 'N', len(y1)); del J1, L1
    J = load_sample(); S, K = selection(J); n = len(S['y']); y = S['y']; log('N', n)
    # on-manifold check on the raw (unrebuilt) jets
    oe0, op0 = escheme_offset(J)
    Lall = {k: np.zeros((len(EDITS), n, 10), np.float32) for k in names}; geo = {}; mb = None
    for e, name in enumerate(EDITS):
        Je = make_edit(J, K, S, name); me = jmass(Je).astype(np.float64)
        if name == 'base': mb = me
        oe, op = escheme_offset(Je); off = np.hypot(oe, op)
        rel = np.abs(me / np.maximum(mb, 1e-6) - 1)
        geo[name] = dict(mass_rel_median=float(np.median(rel)), mass_rel_p99=float(np.percentile(rel, 99)), mass_rel_max=float(rel.max()),
                         escheme_offset_median_all=float(np.median(off)), escheme_offset_median_tail5=float(np.median(off[S['tail5']])),
                         escheme_offset_p90_tail5=float(np.percentile(off[S['tail5']], 90)))
        L = run(Je)
        for k in names: Lall[k][e] = L[k]
        log(f'edit {name:8s}', geo[name], ' '.join(f'{k} acc {(L[k].argmax(1) == y).mean():.4f}' for k in names))
    raw_off = np.hypot(oe0, op0)
    onm = dict(raw_escheme_offset_median_all=float(np.median(raw_off)), raw_escheme_offset_median_tail5=float(np.median(raw_off[S['tail5']])),
               cabs_median_all=float(np.median(S['cabs'])), cabs_median_tail5=float(np.median(S['cabs'][S['tail5']])),
               corr_c_vs_escheme_offset_tail5=float(np.corrcoef(np.r_[S['ce'][S['tail5']], S['cp'][S['tail5']]], np.r_[oe0[S['tail5']], op0[S['tail5']]])[0, 1]))
    log('on-manifold', onm)

    # ---------------------------------------------------------------- analysis
    E = {e: i for i, e in enumerate(EDITS)}; rng = np.random.default_rng(11)
    half = lambda a: 0.5 * np.log(np.maximum(a, 1e-30) / S['M2'])
    pred = {'rec': half(S['M2_rec']), 'ref': half(S['M2_ref']), 'rs': half(S['M2_rs'])}
    ci = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]
    pb = lambda b: float(max(1 / len(b), min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()))))
    from scipy.stats import binomtest
    def mcn(a, b):
        x, z = int((a & ~b).sum()), int((~a & b).sum()); return dict(only_first=x, only_second=z, p=float(binomtest(x, x + z, 0.5).pvalue) if x + z else 1.0)
    out = dict(node='P36v', N=n, per_class=PER, seed=SEED, files='_119', acc_p1=acc_p1, geometry=geo, on_manifold=onm, models={},
               sets={s: dict(n=int(S[s].sum()), nWZ=int((S[s] & np.isin(y, [iW, iZ])).sum()), cabs_median=float(np.median(S['cabs'][S[s]])),
                             pred_rec=float(pred['rec'][S[s]].mean())) for s in TAILS})
    out['acc_sample'] = {k: float((Lall[k][0].argmax(1) == y).mean()) for k in names}
    sets = dict(TAILS); sets = {s: S[s] for s in TAILS}; sets['all'] = np.ones(n, bool)
    holm_in = []
    for k in names:
        Lk = Lall[k]; out['models'][k] = {}
        for s, f in sets.items():
            r = {}; idx = np.where(f)[0]; c0 = Lk[0, idx].argmax(1); yy = y[idx]
            # class-balanced calls changed (mean over classes of the per-class rate), bootstrap within classes
            Wb = np.stack([rng.integers(0, len(idx), len(idx)) for _ in range(NB)]); wcls = 1.0 / np.bincount(yy, minlength=10)[yy] / 10
            ch = {}
            for e in EDITS[1:]:
                x = Lk[E[e], idx].argmax(1) != c0; ch[e] = x
                bw = (x[Wb] * wcls[Wb]).sum(1) / wcls[Wb].sum(1)
                r[f'changed_{e}'] = dict(v_balanced=float((x * wcls).sum() / wcls.sum()), ci=ci(bw), v_raw=float(x.mean()))
            r['mcnemar_rec_vs_mirp'] = mcn(ch['rec'], ch['mirp']); r['mcnemar_rec_vs_flip'] = mcn(ch['rec'], ch['flip'])
            r['mcnemar_rec_vs_dil_rec'] = mcn(ch['rec'], ch['dil_rec'])
            ix = idx[np.isin(yy, [iW, iZ])]; mr = len(ix); Wr = np.stack([rng.integers(0, mr, mr) for _ in range(NB)])
            lo = lambda e: (Lk[E[e], ix, iW] - Lk[E[e], ix, iZ]).astype(np.float64); b = lo('base')
            d = {e: lo(e) - b for e in EDITS[1:]}
            rr = {f'd_{e}': dict(v=float(d[e].mean()), ci=ci(d[e][Wr].mean(1)), absmean=float(np.abs(d[e]).mean()),
                                 absmean_ci=ci(np.abs(d[e])[Wr].mean(1))) for e in d}
            pr = {e: pred[e][ix] for e in ('rec', 'ref', 'rs')}
            kn = sum(d['dil_' + e] * pr[e] for e in pr); kd = sum(pr[e] ** 2 for e in pr)
            kap = kn.sum() / kd.sum(); kb = kn[Wr].sum(1) / kd[Wr].sum(1); rr['kappa'] = dict(v=float(kap), ci=ci(kb))
            Rb = {}
            for e in ('rec', 'ref', 'rs', 'mirp', 'flip', 'rot'):
                pe = pr.get(e, pr['rec']); Rb[e] = d[e][Wr].mean(1) / (kb * pe[Wr].mean(1))
                rr[f'Rk_{e}'] = dict(v=float(d[e].mean() / (kap * pe.mean())), ci=ci(Rb[e]))
            rr['Rk_rec_minus_Rk_rs'] = dict(v=rr['Rk_rec']['v'] - rr['Rk_rs']['v'], ci=ci(Rb['rec'] - Rb['rs']), p=pb(Rb['rec'] - Rb['rs']))
            rr['R_direct_rec'] = float(d['rec'].mean() / d['dil_rec'].mean())
            sg = rng.choice([-1.0, 1.0], size=(NPERM, mr))
            for e in ('rec', 'mirp', 'flip'):
                rr[f'perm_{e}_vs_0'] = float((1 + (np.abs(sg @ d[e] / mr) >= abs(d[e].mean())).sum()) / (1 + NPERM))
            dd = d['rec'] - d['mirp']; rr['perm_rec_vs_mirp'] = dict(diff=float(dd.mean()), ci=ci(dd[Wr].mean(1)),
                                                                     p=float((1 + (np.abs(sg @ dd / mr) >= abs(dd.mean())).sum()) / (1 + NPERM)))
            rr['n'] = mr; r['logodds_WZ'] = rr
            if s == 'tail5': holm_in.append((k, rr['perm_rec_vs_0'], r['mcnemar_rec_vs_mirp']['p']))
            out['models'][k][s] = r
            log(f'{k:9s} {s:6s} nWZ {mr:5d} kappa {kap:+.2f} Rk_rec {rr["Rk_rec"]["v"]:+.2f} {np.round(rr["Rk_rec"]["ci"], 2)} Rk_rs {rr["Rk_rs"]["v"]:+.2f} '
                f'{np.round(rr["Rk_rs"]["ci"], 2)} Rk_mirp {rr["Rk_mirp"]["v"]:+.2f} Rk_flip {rr["Rk_flip"]["v"]:+.2f} p(rec=0) {rr["perm_rec_vs_0"]:.3g} | '
                f'|d| rec {rr["d_rec"]["absmean"]:.3f} mirp {rr["d_mirp"]["absmean"]:.3f} dil_rec {rr["d_dil_rec"]["absmean"]:.3f} | changed% '
                + ' '.join(f'{e} {100 * r["changed_" + e]["v_balanced"]:.1f}' for e in EDITS[1:]) + f' | McN rec/mirp {r["mcnemar_rec_vs_mirp"]}')
    # Holm over the 4 models (tail5)
    for j, key in ((1, 'perm_rec_vs_0'), (2, 'mcnemar_rec_vs_mirp')):
        ps = np.array([h[j] for h in holm_in]); o = np.argsort(ps); adj = np.zeros(len(ps)); run_ = 0
        for rk, i in enumerate(o): run_ = max(run_, min(1, (len(ps) - rk) * ps[i])); adj[i] = run_
        for h, a in zip(holm_in, adj): out['models'][h[0]]['tail5'].setdefault('holm', {})[key] = dict(p=float(ps[holm_in.index(h)]), p_holm=float(a), m=len(ps))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f'p36_verify{TAG}.json').write_text(json.dumps(out, indent=1)); log('wrote', RES / f'p36_verify{TAG}.json')
    np.savez_compressed(RES / f'p36_verify_jets{TAG}.npz', y=y, cabs=S['cabs'], **{f'L_{k}': v for k, v in Lall.items()}, oe0=oe0, op0=op0)


if __name__ == '__main__':
    main()
