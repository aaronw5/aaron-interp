"""P43v -- verification of P43 (spread about the stored axis vs centroid offset) on a fresh sample, plus the missing control.

Fresh sample: JetClass val_5M WToQQ_121.root and ZToQQ_121.root (P43 used _120), 10,000 random rows each, numpy default_rng(4344).
Pair rule and edits E1, E2, E3, E4, E5 are imported unchanged from interp/p43_spread_vs_centroid.py.

Missing control (new, E6): P43's offset estimate E2 - E3 is a rigid translation of all axis-relative coordinates by delta, but it is
applied to the E3 jet, whose two hardest similar-pT particles have already been moved 0.03-0.13 rad (E3 alone changes ~46 % of
ParT_full calls). E6 applies the same rigid translation to the UNEDITED jet: every real particle's (deta, dphi) += delta * e and the
stored axis (jet_eta, jet_phi) moves by -delta * e, so the lab-frame particles (and invariant mass, pair features) are exactly
unchanged; M2c is unchanged and M2 changes by M0 (|c + delta e|^2 - |c|^2). delta in {0.005, 0.01, 0.02, 0.04, 0.06} rad;
direction e = the P43 E2 direction of the jet ('E6e', same direction as E2 - E3) and a uniform random direction ('E6r').
Jets whose jet_eta would change sign under the axis move keep jet_eta (their particles shift rigidly in eta instead; counted).
Each E6 gets a per-jet matched E4 dilation (same Delta log M2). E6r and its E4 are also run on the jets P43 dropped (no pair).
Read-out d = logit_Z - logit_W; R = sum Delta d(edit) / sum Delta d(matched E4); 2,000 paired bootstraps; sign-flip permutation;
Holm over the 4 trained models.
  ./sync.sh && ./part_job.sh c-p43-verify 'python -m interp.p43v_verify'
  -> $AI_RESULTS/part/p43_verify.json
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('JETDISTILL_TAGGER', 'part')
from interp import p43_spread_vs_centroid as P43  # noqa: E402

SMOKE = '--smoke' in sys.argv
SEED, FIDX = 4344, 121
PER = 400 if SMOKE else 10000
NB = 200 if SMOKE else 2000
NPERM = 300 if SMOKE else 5000
DL6 = [0.005, 0.01, 0.02, 0.04, 0.06]
RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'
OUT = RES / f'p43_verify{"_smoke" if SMOKE else ""}.json'
T0 = time.time()
P43.NB, P43.NPERM = NB, NPERM


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def rigid(J, K, de_shift, dp_shift):
    """all real particles' (deta, dphi) += shift; stored axis moved by -shift so the lab-frame particles stay fixed."""
    from interp.part_core import build
    pt, de, dp, ms = K; r = (pt > 0)
    sg = np.where(J['jet_eta'] < 0, -1.0, 1.0); new_eta = J['jet_eta'] - sg * de_shift
    flip = np.sign(new_eta) != np.sign(J['jet_eta'])
    J2 = dict(J); J2['jet_eta'] = np.where(flip, J['jet_eta'], new_eta).astype(np.float32)
    J2['jet_phi'] = (J['jet_phi'] - dp_shift).astype(np.float32)
    E = build(J2, pt, np.where(r, de + de_shift[:, None], 0).astype(np.float32), np.where(r, dp + dp_shift[:, None], 0).astype(np.float32), ms)
    return E, int(flip.sum())


def main():
    import torch
    from interp.p39v_verify import read_rand, load_nets, runner
    from interp.part_core import kin, build, jmass, cat, sub
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; log('device', dev)
    M, FEATURES = load_nets(dev); names = list(M); run = runner(M, FEATURES, dev)
    CL = ['QCD', 'Hbb', 'Hcc', 'Hgg', 'H4q', 'Hqql', 'Zqq', 'Wqq', 'Tbqq', 'Tbl']; iW, iZ = 7, 6
    rng = np.random.default_rng(SEED); Js = []
    for c, pre in (('Wqq', 'WToQQ'), ('Zqq', 'ZToQQ')):
        J, rows = read_rand(P43.VAL / f'{pre}_{FIDX}.root', PER, rng); assert np.all(J['label'] == CL.index(c)); Js.append(J)
    J = cat(Js); del Js; n0 = len(J['label']); K = kin(J)
    Pall = P43.plan_edits(*K[:3], rng); keep = Pall['keep']
    R = dict(node='P43v', seed=SEED, files=f'val_5M {{WToQQ,ZToQQ}}_{FIDX}.root', per_class=PER, nb=NB, n_read=int(n0), n_kept=int(keep.sum()),
             pair_constraint_pass=Pall['diag'])
    Lall = run(build(J, *K), names); yall = J['label'].astype(int)
    R['acc_all'] = {k: float((Lall[k].argmax(1) == yall).mean()) for k in names}
    R['acc_kept'] = {k: float((Lall[k].argmax(1) == yall)[keep].mean()) for k in names}
    R['acc_dropped'] = {k: float((Lall[k].argmax(1) == yall)[~keep].mean()) for k in names}
    log('n_read', n0, 'kept', int(keep.sum()), 'acc all/kept/dropped', R['acc_all'], R['acc_kept'], R['acc_dropped'])
    out_sets = {}
    for setname, rows in (('kept', np.where(keep)[0]), ('dropped', np.where(~keep)[0])):
        Js = sub(J, rows); Ks = kin(Js); n = len(rows); y = Js['label'].astype(int); isW = y == iW
        M0, ce, cp, M2, M2c = P43.moments(*Ks[:3])
        rr = np.random.default_rng(SEED + 7); al = rr.uniform(0, 2 * np.pi, n); er = np.stack([np.cos(al), np.sin(al)], 1)
        L, dlog, match, geo = {}, {}, {}, {}
        L['base'] = run(build(Js, *Ks), names)
        if setname == 'kept':
            P = P43.plan_edits(*Ks[:3], np.random.default_rng(SEED + 1)); assert P['keep'].all()
            E1n = [f'E1_{f:+.2f}' for f in P43.FS]; E2n = [f'E2_{x:.2f}' for x in P43.DELTAS]; E3n = [e.replace('E2', 'E3') for e in E2n]
            for e, f in zip(E1n, P43.FS): dlog[e] = np.full(n, np.log1p(f)); match[e] = e
            for e2, e3 in zip(E2n, E3n): dlog[e2] = np.log(P[(e2, 'M2new')] / P['M2']); match[e2] = e2; match[e3] = e2
            match['E5+'], match['E5-'] = 'E1_+0.10', 'E1_-0.10'
            for e in E1n + E2n + E3n + ['E5+', 'E5-']:
                L[e] = run(P43.make(Js, Ks, P, e), names)
            ee = P[('E2_0.04', 'dc')] / 0.04
        else:
            E1n, E2n, E3n = [], [], []; ee = None
        flips = {}
        for dl in DL6:
            for tag, e_ in (('E6e', ee), ('E6r', er)):
                if e_ is None: continue
                nm = f'{tag}_{dl:.3f}'; sh = dl * e_
                Je, flips[nm] = rigid(Js, Ks, sh[:, 0], sh[:, 1]); L[nm] = run(Je, names)
                Ke = kin(Je); _, cee, cpe, M2e, M2ce = P43.moments(*Ke[:3])
                dlog[nm] = np.log((M2c + M0 * ((ce + sh[:, 0]) ** 2 + (cp + sh[:, 1]) ** 2)) / M2); match[nm] = nm
                geo[nm] = dict(dlogM2_planned=float(dlog[nm].mean()), dlogM2_realised=float(np.mean(np.log(M2e / M2))),
                               dlogM2c=float(np.mean(np.log(M2ce / M2c))), max_abs_dlog_minv=float(np.max(np.abs(np.log(np.maximum(jmass(Je), 1e-3) / np.maximum(jmass(build(Js, *Ks)), 1e-3))))),
                               n_eta_sign_flip_jets=flips[nm]); del Je
                log(setname, nm, geo[nm])
        for e in [m for m in match.values() if m in dlog]:
            if f'E4|{e}' in L: continue
            k_ = np.sqrt(np.exp(dlog[e])); L[f'E4|{e}'] = run(P43.make(Js, Ks, None, 'E4', k=k_), names)
        # statistics
        rb = np.random.default_rng(SEED + 2); C = np.zeros((NB, n), np.float32)
        for b in range(NB): C[b] = np.bincount(rb.integers(0, n, n), minlength=n)
        prs = np.random.default_rng(SEED + 3); res = {}
        for k in names:
            fn = lambda A: (A[:, iZ] - A[:, iW]).astype(np.float64)
            base = fn(L['base'][k]); dlt = {e: fn(L[e][k]) - base for e in L if e != 'base'}; S = {e: (v.sum(), C @ v) for e, v in dlt.items()}
            def ratio(num_es, den_es, sub_es=()):
                nu = sum(S[e][0] for e in num_es) - sum(S[e][0] for e in sub_es); nb = sum(S[e][1] for e in num_es) - sum(S[e][1] for e in sub_es)
                de = sum(S[f'E4|{match[e]}'][0] for e in den_es); db = sum(S[f'E4|{match[e]}'][1] for e in den_es); return nu / de, nb / db
            o = {}
            for nm in [x for x in L if x.startswith('E6')]:
                v, b = ratio([nm], [nm]); o[nm] = dict(R=float(v), R_ci=P43.ci(b), dmean=float(dlt[nm].mean()), e4_dmean=float(dlt[f'E4|{nm}'].mean()),
                                                      calls_changed=float((L[nm][k].argmax(1) != L['base'][k].argmax(1)).mean()),
                                                      WtoZ_edit=float((L[nm][k].argmax(1) == iZ)[isW].mean()), WtoZ_E4=float((L[f'E4|{nm}'][k].argmax(1) == iZ)[isW].mean()))
            o['WtoZ_base'] = float((L['base'][k].argmax(1) == iZ)[isW].mean())
            for tag in ('E6e', 'E6r'):
                es = [f'{tag}_{d:.3f}' for d in (0.01, 0.02, 0.04)]
                if es[0] in L:
                    v, b = ratio(es, es); o[f'{tag}_pooled_0.01-0.04'] = dict(R=float(v), R_ci=P43.ci(b)); o[f'_{tag}b'] = b
            if setname == 'kept':
                for nm, es in (('E1_raise', [e for e in E1n if '+' in e]), ('E1_lower', [e for e in E1n if '-' in e])):
                    v, b = ratio(es, es); o[nm] = dict(R=float(v), R_ci=P43.ci(b)); o[f'_{nm}b'] = b
                v, b = ratio(E2n, E2n, E3n); o['E2_minus_E3'] = dict(R=float(v), R_ci=P43.ci(b)); o['_e23b'] = b
                for d in P43.DELTAS:
                    e2, e3, e6 = f'E2_{d:.2f}', f'E3_{d:.2f}', f'E6e_{d:.3f}'
                    v, b = ratio([e2], [e2], [e3]); v6, b6 = ratio([e6], [e6])
                    o[f'E2mE3_{d:.2f}'] = dict(R=float(v), R_ci=P43.ci(b)); o[f'diff_E2mE3_minus_E6e_{d:.2f}'] = dict(v=float(v - v6), ci=P43.ci(b - b6), p=P43.boot_p(b - b6, None))
                    o[f'E6e_over_E4_dlog_check_{d:.2f}'] = float(np.max(np.abs(dlog[e2] - dlog[e6])))
                dd = o['_e23b'] - o['_E6eb']; o['diff_E2mE3_minus_E6e_pooled'] = dict(v=o['E2_minus_E3']['R'] - o['E6e_pooled_0.01-0.04']['R'], ci=P43.ci(dd), p=P43.boot_p(dd, None))
                d1 = o['_E1_raiseb'] - o['_E6eb']; o['E1raise_minus_E6e'] = dict(v=o['E1_raise']['R'] - o['E6e_pooled_0.01-0.04']['R'], ci=P43.ci(d1), p=P43.boot_p(d1, None))
                nu = S['E5+'][0] + S['E5-'][0]; de_ = S['E4|E1_+0.10'][0] - S['E4|E1_-0.10'][0]
                o['E5'] = dict(R=float(nu / de_), R_ci=P43.ci((S['E5+'][1] + S['E5-'][1]) / (S['E4|E1_+0.10'][1] - S['E4|E1_-0.10'][1])))
                o['p_E6e_0.04_eq_0'] = P43.signflip_p(dlt['E6e_0.040'], prs)
            res[k] = {kk: vv for kk, vv in o.items() if not kk.startswith('_')}
            log(setname, k, {kk: (round(vv['R'], 3), np.round(vv['R_ci'], 3).tolist()) for kk, vv in res[k].items() if isinstance(vv, dict) and 'R' in vv})
        if setname == 'kept':
            for key in ('E1raise_minus_E6e', 'diff_E2mE3_minus_E6e_pooled'):
                ks = [k for k in names]; hp = P43.holm([res[k][key]['p'] for k in ks])
                for k, h in zip(ks, hp): res[k][key]['holm'] = h
        out_sets[setname] = dict(n=int(n), geometry=geo, models=res, median_cabs=float(np.median(np.hypot(ce, cp))), median_M2_over_M0=float(np.median(M2 / M0)))
    R['sets'] = out_sets
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(R, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    log('wrote', OUT)


if __name__ == '__main__':
    main()
