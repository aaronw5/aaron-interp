"""H50 verification (loop D): rerun the JetClass part of interp/h50_z_vs_dr.py on a fresh sample, with two added controls.

Fresh sample: JetClass test files *_102 (H50 used *_101), 2,000 random rows per class (rng 6001-6003), edit rng 6010; reference
training rows from train_100M *_003 (H50: *_002), rng 6016-6018. Everything else (edits, acceptance, models, statistics) is the
H50 code, imported unchanged.

Added controls:
  Brot(lam)  both prongs translated rigidly so that their joining vector in (y, phi) is ROTATED about the pT-weighted centre by
             theta = sign(lam - 1) * 2 arcsin(|lam - 1| / 2). Each prong is displaced by the same distance as under B(lam), but
             Delta R, z, prong masses and constituent pT stay exactly fixed (jet mass changes only at second order). Brot - idc
             measures the response to moving the prongs without changing their separation (generic translation sensitivity);
             B - Brot is the separation change at equal prong displacement.
  idc(lam)   the unedited jet (so that Brot - idc goes through the same bootstrap / acceptance machinery).
  references MLP0+dR+pm (m, pT, N, dR, m_hi, m_lo: z-blind and D2-blind, but carries every quantity that fixes z through
             m^2 ~ m_a^2/z + m_b^2/(1-z) + z(1-z) pT^2 dR^2) and MLP0+z+dR.
Outputs: $AI_RESULTS/physics/h50_verify_* (TAG h50_verify).
Parts: --part jc (env-part, CPU) | nets (GPU) | stats
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import h50_z_vs_dr as H  # noqa: E402
from interp import h45_prong_sharing as H45  # noqa: E402

H.TAG = 'h50_verify'
H.PER_JC = 2000
TEST = Path('/j-jepa-vol/JetClass/Pythia/test_20M')
H.VARIANTS = H.VARIANTS + [('Brot', H.LAMS), ('idc', H.LAMS)]
H.VNAMES = [f'{k}|{p}' for k, ps in H.VARIANTS for p in ps]
H.ARMS = dict(H.ARMS); H.ARMS['B'] = H.ARMS['B'] + [('Brot', 'idc'), ('B', 'Brot')]
H.REFS = dict(H.REFS); H.REFS['MLP0+dR+pm'] = ['m', 'pt', 'N', 'dR', 'msub_hi', 'msub_lo']; H.REFS['MLP0+z+dR'] = ['m', 'pt', 'N', 'z', 'dR']
_orig_jet_edits = H.jet_edits


def rotate_pair(pt, y, phi, m, lab, theta):
    a = [H.paxis(pt, y, phi, m, lab == k) for k in (0, 1)]; w = np.array([np.hypot(v[2][0], v[2][1]) for v in a])
    c0 = np.array([a[0][0], a[0][1]]); d = np.array([a[1][0] - a[0][0], H.wrap(a[1][1] - a[0][1])]); C = c0 + d * w[1] / w.sum()
    Rm = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    y2 = y.copy(); p2 = phi.copy()
    for k, ck in ((0, c0), (1, c0 + d)):
        sh = C + Rm @ (ck - C) - ck; s = lab == k; y2[s] += sh[0]; p2[s] += sh[1]
    return y2, p2


def jet_edits_v(args):
    j, out, info = _orig_jet_edits(args)
    _, P, lab, _ = args
    pt, y, phi, m = H.kin(P)
    for lam in H.LAMS:
        th = np.sign(lam - 1.0) * 2 * np.arcsin(abs(lam - 1.0) / 2)
        yr, pr = rotate_pair(pt, y, phi, m, lab, th); out[f'Brot|{lam}'] = H.build(pt, yr, pr, m)
        info[f'dm|Brot|{lam}'] = H.jm(pt, yr, pr, m) - H.jm(pt, y, phi, m)
        out[f'idc|{lam}'] = P.copy()
    return j, out, info


H.jet_edits = jet_edits_v
_orig_restore_job = H.restore_job


def restore_job_v(args):
    """H50 restore_job, but a jet that fails the 1e-6 E invariant assertion is rejected instead of stopping the run (seen once:
    a one-particle prong whose target mass is 8e-4 GeV, which restore_pm's 1e-3 GeV tolerance lets through)"""
    try:
        return _orig_restore_job(args)
    except AssertionError:
        return args[0], None, dict(reason='assert_fail')


H.restore_job = restore_job_v


def part_jc():
    from interp import h4_top_wmass as H4
    from interp import h19_wpair_cross as X
    from interp.h41_colour_flow import read_window
    import interp.h41_colour_flow as H41
    from interp.part_core import cat, kin as pkin, PK
    H41.log = H.log
    Js, meta = [], dict(node='H50_verify', part='jc', sample={})
    for k, c in enumerate(H45.JC_CLASSES):
        J, inf = read_window(TEST / f'{H45.JC_PRE[c]}_102.root', H.PER_JC, 6001 + k); Js.append(J); meta['sample'][c] = inf
    J = cat(Js); y = J['label'].astype(int); P4raw, nn = X.jc_P4(J); P4 = H4._identity(P4raw, nn).astype(np.float32)
    J['pmass'] = pkin(J)[3].astype(np.float32)
    H.log(f'jc sample {len(nn)} jets; labels {np.bincount(y, minlength=10).tolist()}; median pT {np.median(J["jet_pt"]):.0f} GeV')
    rows = np.flatnonzero(nn >= 3)
    VAR, ACC, OBS, INFO, LAB, emeta = H.run_edits(P4, nn, rows, 6010, 'jc')
    D = {'nn': nn, 'y': y, 'LAB': LAB.astype(np.int16), 'P|id': P4}
    for v in H.VNAMES: D[f'P|{v}'] = VAR[v]
    for k in PK + ['pmass', 'jet_eta', 'jet_phi', 'jet_pt', 'label']: D[f'base|{k}'] = J[k]
    H.RES.mkdir(parents=True, exist_ok=True); np.savez_compressed(H.RES / f'{H.TAG}_jc_inputs.npz', **D); H.log('wrote jc inputs')
    Os, ys, meta['ref_rows'] = [], [], {}
    for k, c in enumerate(H45.JC_CLASSES):
        Jr, inf = read_window(H45.TRAIN / f'{H45.JC_PRE[c]}_003.root', H.NREF, 6016 + k); meta['ref_rows'][c] = inf
        Pr, nr = X.jc_P4(Jr); Pr = H4._identity(Pr, nr).astype(np.float32); Os.append(H45.observables(Pr, nr)); ys.append(Jr['label'].astype(int))
    Op = {q: np.concatenate([o[q] for o in Os]) for q in H45.OBSK}; yp = np.concatenate(ys)
    R, meta['ref_auc'] = H.fit_refs(Op, yp, H45.JC_SCORES, H45.JC_CLASSES)
    H.save_obs(H.RES / f'{H.TAG}_jc_obs.npz', OBS, ACC, INFO, y, {nm: H.ref_scores(R, O) for nm, O in OBS.items()}, {})
    meta['edit'] = emeta; (H.RES / f'{H.TAG}_jc_meta.json').write_text(json.dumps(H.jsonable(meta), indent=1)); H.log('jc part written')


def selftest_rot():
    rng = np.random.default_rng(0); P = []
    for (e, p, pt, nk) in ((0.0, 0.0, 400.0, 12), (0.20, 0.10, 150.0, 9)):
        for _ in range(nk):
            ee, pp, q = e + rng.normal(0, 0.025), p + rng.normal(0, 0.025), pt / nk * rng.uniform(0.5, 1.5)
            P.append([q * np.cos(pp), q * np.sin(pp), q * np.sinh(ee), np.sqrt((q * np.cosh(ee)) ** 2 + 0.14 ** 2)])
    P = np.array(P); lab = np.repeat([0, 1], [12, 9]); nn = np.array([len(P)])
    j, out, info = jet_edits_v((0, P, lab, 1)); v0 = H45.prong_vars(P[None], nn, lab[None])
    pt, y, phi, m = H.kin(P); M0 = [H.pmass(pt, y, phi, m, lab == k) for k in (0, 1)]
    for lam in H.LAMS:
        Q = out[f'Brot|{lam}']; q = H45.prong_vars(Q[None], nn, lab[None]); mq = [H.pmass(*H.kin(Q), lab == k) for k in (0, 1)]
        assert abs(q['dR'][0] - v0['dR'][0]) < 1e-9 and abs(q['z'][0] - v0['z'][0]) < 1e-12 and max(abs(mq[0] - M0[0]), abs(mq[1] - M0[1])) < 1e-9
        H.log(f'selftest Brot|{lam}: dR {v0["dR"][0]:.4f}->{q["dR"][0]:.4f}; dm {info[f"dm|Brot|{lam}"]:+.3f} GeV (B: {H.mass4(out[f"B|{lam}"].sum(0)) - H.jm(pt, y, phi, m):+.3f})')
    H.log('selftest_rot passed')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--part', default=None); a = ap.parse_args()
    H.log(f'H50 verify part {a.part}; per class {H.PER_JC}; variants {H.VNAMES}')
    if a.part == 'jc':
        H.selftest(); selftest_rot(); part_jc()
    elif a.part == 'nets':
        H.part_nets()
    elif a.part == 'stats':
        H.part_stats()
    elif a.part == 'selftest':
        H.selftest(); selftest_rot()
    H.log('done')
