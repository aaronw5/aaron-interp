"""H23 verification (loop D): rerun of the H23 JetClass H4 and H3v arms on fresh rows, plus a dose-response control.

Fresh sample: TTBar_100 / ZJetsToNuNu_100, 550 < jet_pt < 650 GeV, |jet_eta| < 2, >= 3 particles; 2,000 Tbqq (default_rng(6623))
and 2,000 QCD (default_rng(6624)), random rows, excluding the H19 / H19v draws (as H23) AND the H23 draws (rng 9523 / 9524 from the
H19-excluded pool, reconstructed with the same rng calls).
Arms: H4 W-pair edit (h4_verify.edit_set) T60, T100 and two added targets T70, T90 (same eligible tops, 65 < mW < 95; dose-response
control: a mass-window reading predicts |W - N| ordered 60 > 70 and 100 > 90, a generic-disruption reading predicts no ordering in
|mW - 80|), each built on the fixed stored axis (st) and recentred about each edited jet's own E-scheme axis (rc), with the
translation-only control TN; H3v k-means W-pair edit at 60 / 100 GeV. JetClass models only (ParT/PN full and kin); MLPx_jc,
MLPxW_jc, LR2_jc references retrained exactly as in H23 (deterministic).
Run: ./part_job.sh d-h23-verify 'python -m interp.h23_verify'
Output: $AI_RESULTS/physics/h23_verify_jc.json, h23_verify_jc_perjet.npz
"""
import numpy as np
from interp import h23_axis_audit as A

SEED_T, SEED_Q = 6623, 6624


def window_fresh(c, k, per, seed, excl_old, prev):
    """random rows; excl_old = [(seed, n)] drawn from the eligible pool (H19 / H19v); prev = (seed, n) the H23 draw, made from
    the pool minus excl_old"""
    import uproot
    from interp.part_core import read_root, cat, sub
    from interp.h19_wpair_cross import jc_file
    f = jc_file(c, k); t = uproot.open(f)['tree']
    a = t.arrays(['jet_pt', 'jet_eta', 'jet_nparticles'], library='np')
    ok = (a['jet_pt'] > 550) & (a['jet_pt'] < 650) & (np.abs(a['jet_eta']) < 2) & (a['jet_nparticles'] >= 3)
    el = np.flatnonzero(ok); old = np.concatenate([np.random.default_rng(s).choice(el, min(n, len(el)), replace=False) for s, n in excl_old])
    el2 = np.setdiff1d(el, old); h23 = np.random.default_rng(prev[0]).choice(el2, prev[1], replace=False)
    el3 = np.setdiff1d(el2, h23); idx = np.sort(np.random.default_rng(seed).choice(el3, per, replace=False)); parts = []; N = t.num_entries
    for s in range(0, N, 20000):
        loc = idx[(idx >= s) & (idx < s + 20000)] - s
        if len(loc): parts.append(sub(read_root(f, start=s, stop=min(s + 20000, N)), loc))
    info = dict(file=f.name, eligible=int(len(el)), excluded_h19=int(len(np.unique(old))), excluded_h23=int(len(h23)), drawn=int(len(idx)),
                overlap_with_excluded=int(len(np.intersect1d(idx, np.concatenate([old, h23])))), seed=seed)
    A.log(f'jc {c}: {info}')
    return cat(parts), info


def load_jc_fresh():
    from interp.part_core import cat, kin
    from interp import h4_top_wmass as H
    from interp.h19_wpair_cross import jc_P4
    Jt, it = window_fresh('Tbqq', 100, A.PER, SEED_T, [(4191, 5000), (4391, 5000)], (9523, A.PER))
    Jq, iq = window_fresh('QCD', 100, A.PER, SEED_Q, [(4192, 5000), (4392, 3000)], (9524, A.PER))
    J = cat([Jt, Jq]); P4raw, nn = jc_P4(J); P4 = H._identity(P4raw, nn).astype(np.float32); y = (J['label'] == A.TBQQ).astype(int)
    J['pmass'] = kin(J)[3].astype(np.float32)
    return dict(J=J, base=J, P4=P4, nn=nn, y=y, ax=(J['jet_eta'].astype(np.float64), J['jet_phi'].astype(np.float64)), info=dict(Tbqq=it, QCD=iq))


def edits_dose(S, sets):
    """A.edits_h4 with the added targets T70 / T90 (same eligibility as T60 / T100, tolerance 5 GeV)"""
    from interp import h4_top_wmass as H
    from interp import h4_verify as HV
    nn, P4, y = S['nn'], S['P4'], S['y']; O = H._obs(P4, nn); S['O'] = O
    win = (O['mSD'] > 105) & (O['mSD'] < 210) & (nn >= 3)
    e6 = win & (y == 1) & (O['mW'] > 65) & (O['mW'] < 95)
    tgt = {'T60': 60.0, 'T70': 70.0, 'T90': 90.0, 'T100': 100.0}; E = {}
    for key in sets:
        t = tgt[key]; tol = 5.0; rows = np.flatnonzero(e6)
        ed = HV.edit_set(P4, nn, rows, t); acc = {}
        for k in 'WN':
            e = ed[k]; rr = e['rows']
            if len(rr) == 0: acc[k] = set(); continue
            Oe = H._obs(e['P'], nn[rr]); dm = np.abs(Oe['m'] - O['m'][rr]); dtau = np.abs(Oe['tau32'] - O['tau32'][rr])
            okW = np.abs(Oe['mW'] - t) < tol if k == 'W' else np.abs(Oe['mW'] - O['mW'][rr]) < 2
            acc[k] = set(rr[(dm < 0.5) & (dtau < 0.02) & okW].tolist())
        ar = np.array(sorted(acc['W'] & acc['N'] & set(ed['R']['rows'].tolist())), np.int64)
        E[key] = dict(rows=ar, P={k: ed[k]['P'][np.searchsorted(ed[k]['rows'], ar)] for k in 'WNR'},
                      counts=dict(eligible=int(len(rows)), W_acc=len(acc['W']), N_acc=len(acc['N']), analysis=int(len(ar))))
        A.log(f'H4 {key}: {E[key]["counts"]}')
    return E


def main():
    A.TAG = 'h23_verify'
    A.SETS = ('T60', 'T70', 'T90', 'T100')
    A.DIRECTION.update({'T70': -1, 'T90': -1})
    A.edits_h4 = edits_dose
    rng = np.random.default_rng(123); tests = []; PJ = {}
    S = load_jc_fresh(); N = A.Nets(top=False); REF = A.refs(['jc'])
    order = np.random.default_rng(605).permutation(len(S['nn']))
    lg_st, lg_rc = {}, {}
    s_st = N.score('jc', S['P4'], S['nn'], S['base'], S['ax'], order, logits=lg_st); s_st.update(A.ref_scores(REF, S['P4'], S['nn']))
    s_rc = N.score('jc', S['P4'], S['nn'], S['base'], A.axis(S['P4'], S['nn']), order, logits=lg_rc)
    perf = {f'{k}|st': v for k, v in A.perf_of(s_st, S['y'], lg_st).items()}; perf.update({f'{k}|rc': v for k, v in A.perf_of(s_rc, S['y'], lg_rc).items()})
    A.log('unedited (fresh jc sample) ' + ', '.join(f"{k} AUC {v['AUC']:.4f} acc {v['acc_binary']:.4f}" + (f" 10c {v['acc_10class']:.4f}" if 'acc_10class' in v else '') for k, v in perf.items()))
    out = dict(node='H23_verify', part='jc', sample=S['info'], unedited=perf)
    out['H4'] = A.run_h4('jc', S, N, REF, perf, PJ, tests, rng)
    out['H3v'] = A.run_h3v(S, N, REF, perf, PJ, tests, rng)
    A.finish(out, tests, PJ, 'jc')


if __name__ == '__main__':
    main()
    A.log('done')
