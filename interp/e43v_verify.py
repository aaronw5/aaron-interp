"""E43 verifier: runs interp.e43_azimuth_route_jedi on a fresh seed with ONE patch applied in memory (the worker's file is not
changed): line 497 builds the pair-slope bootstrap weights with len(rows) where rows is a boolean mask over all N jets, so the
weights have N columns instead of rows.sum() and step 3 crashes (ValueError (N/2, 6) vs (N, 1); seen in job e43-vsmoke).
Output directory redirected to a pod-local scratch directory; results copied to $AI_RESULTS/jedi/e43_verify*.{json,npz}
(also after a crash).  Adds independent checks recomputed here from the edited jets:
  (i) invariants: pT, r of every particle, padding, which particles moved, m, vector M1, Q2 and complex C2;
  (ii) which jets the Q2 solver keeps (class counts, natural Q2 and multiplicity of kept vs dropped jets);
  (iii) side effects: change of axis-free shape observables (e2 at beta 0.5 / 1 / 2, hardest-pair dR01, pT^D, leading
        pT fraction) under each edit vs its matched control, with a paired bootstrap over jets.
  python -m interp.e43v_verify [--seed 2243] [--pc 1500] [--nrand 200] [--nhold 600] [--smoke]
"""
import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); OUT = RES / 'jedi'
BUG = "poisson(1.0, (NB, len(rows)))"; FIX = "poisson(1.0, (NB, int(np.sum(rows))))"


def load_patched():
    src = (ROOT / 'interp' / 'e43_azimuth_route_jedi.py').read_text()
    assert src.count(BUG) == 1, 'patch target not found'
    spec = importlib.util.spec_from_loader('interp.e43_azimuth_route_jedi', loader=None)
    mod = importlib.util.module_from_spec(spec); mod.__file__ = str(ROOT / 'interp' / 'e43_azimuth_route_jedi.py')
    exec(compile(src.replace(BUG, FIX), mod.__file__, 'exec'), mod.__dict__); sys.modules['interp.e43_azimuth_route_jedi'] = mod
    return mod


def qci(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]; return np.quantile(v, [.025, .975]).tolist() if len(v) else [None, None]


def indep_checks(E, seed, pc):
    from interp.jedi_core import sample
    from interp.x10_das_jedi import mass
    import interp.e41_true_mass_layers_jedi as E41
    x, y, _ = sample(pc, seed); Ed, info, sub = E.make_edits(x, seed + 7); o = {'N': len(x)}
    pt = x[..., 0].astype(np.float64); real = pt > 0; r0 = np.hypot(x[..., 1], x[..., 2]); th0 = np.arctan2(x[..., 2], x[..., 1])
    m0 = mass(x); M10 = np.stack([(pt * x[..., 1]).sum(1), (pt * x[..., 2]).sum(1)], 1)
    w = pt * r0 ** 2; C0 = np.stack([(w * np.cos(2 * th0)).sum(1), (w * np.sin(2 * th0)).sum(1)], 1); Q20 = np.hypot(*C0.T) / w.sum(1)
    inv = {}
    for k, (xe, ok) in Ed.items():
        if not ok.any(): inv[k] = dict(n=0); continue
        xe = xe.astype(np.float64); r1 = np.hypot(xe[..., 1], xe[..., 2]); th1 = np.arctan2(xe[..., 2], xe[..., 1])
        moved = (np.abs(np.angle(np.exp(1j * (th1 - th0)))) > 1e-5) & real
        M11 = np.stack([(pt * xe[..., 1]).sum(1), (pt * xe[..., 2]).sum(1)], 1)
        w1 = pt * r1 ** 2; C1 = np.stack([(w1 * np.cos(2 * th1)).sum(1), (w1 * np.sin(2 * th1)).sum(1)], 1)
        inv[k] = dict(n=int(ok.sum()), pt_changed=bool((xe[..., 0] != x[..., 0]).any()), padded_changed=bool((xe[~real] != x[~real]).any()),
                      max_abs_dr=float(np.abs(r1 - r0)[ok].max()), moved_outside_hard14=int((moved & ~sub)[ok].sum()),
                      max_moved_per_jet=int(moved[ok].sum(1).max()), max_rel_dm=float(np.abs(mass(xe)[ok] / m0[ok] - 1).max()),
                      max_dM1_over_M0=float((np.hypot(*(M11 - M10)[ok].T) / pt[ok].sum(1)).max()),
                      median_rel_dQ2=float(np.median(np.hypot(*C1[ok].T) / np.hypot(*C0[ok].T) - 1)),
                      max_rel_dC2_complex=float((np.hypot(*(C1 - C0)[ok].T) / np.hypot(*C0[ok].T)).max()))
    o['invariants'] = inv
    # (ii) selection of kept jets
    nreal = real.sum(1); sel = {}
    for k in ('turn10', 'ctrl10', 'Q2p', 'Q2cp', 'Q2m', 'Q2cm'):
        ok = Ed[k][1]
        sel[k] = dict(kept=int(ok.sum()), kept_frac_by_class={c: float(ok[y == c].mean()) for c in range(5)},
                      median_Q2_kept=float(np.median(Q20[ok])) if ok.any() else None, median_Q2_dropped=float(np.median(Q20[~ok])),
                      median_N_kept=float(np.median(nreal[ok])) if ok.any() else None, median_N_dropped=float(np.median(nreal[~ok])))
    o['selection'] = sel; o['class_order'] = 'labels 0..4 as interp.jedi_core.sample (W = 2, Z = 3 in the E-series scripts)'
    # (iii) side effects on axis-free shape observables, edit vs matched control (paired over jets kept in both)
    names = ['logm', 'logptJ', 'N', 'ptD', 'e2_b0.5', 'e2_b1', 'e2_b2', 'lead_frac', 'dR01']
    F0, _ = E41.c2st_feats(x); side = {}
    for ke, kc in (('Q2p', 'Q2cp'), ('Q2m', 'Q2cm'), ('turn10', 'ctrl10')):
        ok = Ed[kc][1] & Ed[ke][1]; n = int(ok.sum())
        if n < 20: side[ke] = dict(n=n); continue
        Fe, _ = E41.c2st_feats(Ed[ke][0][ok]); Fc, _ = E41.c2st_feats(Ed[kc][0][ok]); f0 = F0[ok]
        rng = np.random.default_rng(seed + 11); Wb = rng.poisson(1.0, (1000, n)).astype(float); s = {}
        for j, nm in enumerate(names):
            sd = max(F0[:, j].std(), 1e-12); de = (Fe[:, j] - f0[:, j]) / sd; dc = (Fc[:, j] - f0[:, j]) / sd; diff = np.abs(de) - np.abs(dc)
            bd = (Wb @ diff) / Wb.sum(1)
            s[nm] = dict(mean_d_edit_over_sd=float(de.mean()), mean_d_ctrl_over_sd=float(dc.mean()), mean_abs_edit=float(np.abs(de).mean()),
                         mean_abs_ctrl=float(np.abs(dc).mean()), abs_diff_ci=qci(bd), p_two_sided=float(min(1, 2 * min((bd <= 0).mean(), (bd >= 0).mean()))))
        side[ke] = dict(n=n, control=kc, features=s)
    o['side_effects'] = side
    return o


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, default=2243); ap.add_argument('--pc', type=int, default=1500)
    ap.add_argument('--nrand', type=int, default=200); ap.add_argument('--nhold', type=int, default=600); ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--checks-only', action='store_true')
    a = ap.parse_args()
    if a.smoke: a.pc = min(a.pc, 150); a.nrand = min(a.nrand, 10); a.nhold = min(a.nhold, 150)
    E = load_patched()
    tmp = Path('/tmp/e43v'); tmp.mkdir(parents=True, exist_ok=True); E.OUT = tmp; OUT.mkdir(parents=True, exist_ok=True)
    sfx = '_smoke' if a.smoke else f'_s{a.seed}'
    try:
        chk = indep_checks(E, a.seed, a.pc); (OUT / f'e43_verify_checks{sfx}.json').write_text(json.dumps(E.clean(chk) if hasattr(E, 'clean') else chk, indent=1, default=float))
        E.log('independent checks:', json.dumps(chk, default=float)[:6000])
        if not a.checks_only: E.main(a)
    finally:
        for f in tmp.iterdir():
            dst = f.name.replace('e43_main', 'e43_verify_main').replace('e43_example', 'e43_verify_example')
            if a.smoke: dst = dst.replace('_smoke', '_smoke2')
            shutil.copy(f, OUT / dst); E.log(f'copied {f.name} -> {OUT / dst}')


if __name__ == '__main__':
    main()
