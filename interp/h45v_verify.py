"""H45v (loop D skeptic): fresh-sample rerun of H45 (prong momentum-sharing edit D at fixed jet four-momentum) plus a new control.

Fresh samples
  jc   JetClass test files WToQQ_101, ZToQQ_101, ZJetsToNuNu_101 (H45 used the _100 files, so no row overlap): stored
       500 < jet_pt < 1000 GeV, |jet_eta| < 2, >= 3 particles; PER random rows per class (rng 4601 / 4602 / 4603).
  hls  hls4ml test jets (data/_jets/n64/full_test.npz), PER random rows per class (rng 4614) drawn from the rows NOT used by
       H45 (H45's rng 4504 draw of 2000 per class is reproduced and excluded).
Edits (same code path as H45, re-implemented and cross-checked against h45_prong_sharing.edit_one on every jet):
  D(dc)  rotation by alpha = arccos c0 - arccos c1 about n+ x beta_hat in the jet rest frame (|cos theta*| c0 -> c1)
  cphi   same |alpha| about beta_hat (= lab rotation about the jet axis; z, prong energies, Delta R fixed)
  cn     NEW control: same |alpha| about n+ itself (the prong axis) in the rest frame, random sign. Both prong four-momenta (so z,
         Delta R, prong masses, jet four-momentum) are exactly fixed; the particles inside each prong are rotated about the prong
         axis. A same-angle rest-frame rotation that is not a lab-frame symmetry of the whole jet; computed at dc = +-0.3.
References: MLPx (m, pT, tau21, D2, N, Delta R) and MLPz (+ z_g), 3 retrains, fresh train rows (JetClass train *_001 files,
  rng 4616-4618; hls4ml train archive without dev, rng 4619), NREF per class.
Models: jc ParT_full/kin, PN_full/kin, 3 weight-permuted ParT_full (torch seeds 15-17); hls JEDI-linear official n64 3-feature,
  dsets-d64-n64 seed0, nulls 2 redrawn JEDI (seeds 10, 11), 2 permuted dsets (seeds 17, 18).
Parts: --part jc (env-part python, CPU) | hls (conda) | hjedi (jax env) | nets (GPU) | stats (conda) -> $AI_RESULTS/physics/h45_verify*
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import h45_prong_sharing as H45   # noqa: E402

T00 = time.time()
SMOKE = int(os.environ.get('H45V_SMOKE', '0'))
RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'physics'
V = Path('/j-jepa-vol/aaron-interp')
TAG = 'h45_verify_smoke' if SMOKE else 'h45_verify'
PER = SMOKE or 1200
NREF = 1500 if SMOKE else 10000
B = 300 if SMOKE else 2000
DCS = (-0.3, -0.15, 0.15, 0.3)
CN_DCS = (-0.3, 0.3)
H45.NSEED = 2 if SMOKE else 3
H45.REFS = ('MLPx', 'MLPz')
H45.NPERM = 500 if SMOKE else 2000
VARS = ['D', 'cphi', 'cn']


def log(*a):
    print(f'[{time.time() - T00:7.0f}s]', *a, flush=True)


H45.log = log
jsonable = H45.jsonable


def edits_v(P, lab, dc, rng):
    s0 = lab == 0
    if s0.sum() < 1 or (~s0).sum() < 1: return None, None
    Va = P[s0].sum(0); T = P.sum(0); be = T[:3] / T[3]; bn = np.linalg.norm(be)
    nh = be / bn; na = H45.boost(Va, be)[:3]; na = na / np.linalg.norm(na); cs = float(na @ nh); npl = na if cs >= 0 else -na
    c0 = abs(cs); c1 = float(np.clip(c0 + dc, 0.0, H45.CMAX)); al = float(np.arccos(min(c0, 1.0)) - np.arccos(c1))
    k = np.cross(npl, nh)
    if np.linalg.norm(k) < 1e-9: return None, None
    Pr = H45.boost(P, be)

    def apply(R):
        Q = Pr.copy(); Q[:, :3] = Q[:, :3] @ R.T; return H45.boost(Q, -be)
    out = {'D': apply(H45.rodrigues(k, al)), 'cphi': apply(H45.rodrigues(nh, rng.choice([-1.0, 1.0]) * abs(al))),
           'cn': apply(H45.rodrigues(npl, rng.choice([-1.0, 1.0]) * abs(al)))}
    return out, dict(c0=c0, c1=c1, alpha=al)


def disp(P, Q):
    pt0, e0, p0, _ = H45.ptetaphim(P); _, e1, p1, _ = H45.ptetaphim(Q)
    return float((pt0 * np.hypot(e1 - e0, H45.wrap(p1 - p0))).sum() / pt0.sum())


def edit_set(P4, nn, rows, seed, tag):
    LAB = H45.kt2_labels(P4, nn); O0 = H45.observables(P4, nn); J = len(nn)
    E = {(dc, k): dict(P=P4.copy(), acc=np.zeros(J, bool)) for dc in DCS for k in VARS}
    INFO = {dc: dict(c0=np.full(J, np.nan), c1=np.full(J, np.nan), alpha=np.full(J, np.nan), **{f'disp_{k}': np.full(J, np.nan) for k in VARS}) for dc in DCS}
    chk = dict(max_dD_vs_h45=0.0, max_djet_rel=0.0, max_dpipj_rel=0.0, max_dprong_cn_rel=0.0, n_checked_pairs=0)
    for jj, j in enumerate(rows):
        n = nn[j]; P = P4[j, :n].astype(np.float64); lab = LAB[j, :n]
        for i, dc in enumerate(DCS):
            r, inf = edits_v(P, lab, dc, np.random.default_rng([seed, int(j), i]))
            if r is None: continue
            r0, _ = H45.edit_one(P, lab, dc, np.random.default_rng(0))
            chk['max_dD_vs_h45'] = max(chk['max_dD_vs_h45'], float(np.abs(r0['D'] - r['D']).max()))
            I = INFO[dc]
            for k in ('c0', 'c1', 'alpha'): I[k][j] = inf[k]
            Esum = P[:, 3].sum()
            for kind in VARS:
                if kind == 'cn' and dc not in CN_DCS: continue
                Q = r[kind]; I[f'disp_{kind}'][j] = disp(P, Q)
                chk['max_djet_rel'] = max(chk['max_djet_rel'], float(np.abs(Q.sum(0) - P.sum(0)).max() / Esum))
                if jj < 300:
                    pp = lambda X: X[:, 3][:, None] * X[:, 3][None] - X[:, :3] @ X[:, :3].T
                    chk['max_dpipj_rel'] = max(chk['max_dpipj_rel'], float(np.abs(pp(Q) - pp(P)).max() / Esum ** 2)); chk['n_checked_pairs'] += 1
                if kind == 'cn':
                    for s in (lab == 0, lab == 1):
                        chk['max_dprong_cn_rel'] = max(chk['max_dprong_cn_rel'], float(np.abs(Q[s].sum(0) - P[s].sum(0)).max() / Esum))
                E[(dc, kind)]['P'][j, :n] = Q.astype(np.float32); E[(dc, kind)]['acc'][j] = H45.prong_offsets_ok(Q, lab)
    log(f'[{tag}] edits done on {len(rows)} jets; checks {chk}')
    OBS = {'id': O0}
    for dc in DCS:
        for kind in VARS:
            if kind == 'cn' and dc not in CN_DCS: continue
            OBS[(dc, kind)] = H45.observables(E[(dc, kind)]['P'], nn)
    log(f'[{tag}] observables done')
    return E, OBS, INFO, LAB, chk


def save_obs(path, OBS, E, INFO, y, extra, chk):
    D = {'y': y}
    for key, O in OBS.items():
        nm = 'id' if key == 'id' else f'{key[0]}|{key[1]}'
        for q in H45.OBSK: D[f'O|{nm}|{q}'] = np.asarray(O[q], np.float32)
    for (dc, kind), e in E.items(): D[f'acc|{dc}|{kind}'] = e['acc']
    for dc, I in INFO.items():
        for k, v in I.items(): D[f'info|{dc}|{k}'] = v.astype(np.float32)
    D['checks'] = np.array(json.dumps(chk)); D.update(extra); np.savez_compressed(path, **D)


def refs_and_save(OBS, E, INFO, y, Op, yp, scores, cls_of, path, chk):
    R, fit = H45.fit_refs(Op, yp, scores, cls_of)
    extra = {}
    for key, O in OBS.items():
        nm = 'id' if key == 'id' else f'{key[0]}|{key[1]}'
        for k, v in H45.ref_scores(R, O).items(): extra[f'R|{nm}|{k}'] = v
    save_obs(path, OBS, E, INFO, y, extra, chk); return fit


def part_jc():
    from interp import h4_top_wmass as H
    from interp import h19_wpair_cross as X
    from interp.h41_colour_flow import read_window
    import interp.h41_colour_flow as H41
    from interp.part_core import cat, kin, PK
    H41.log = log
    Js, meta = [], dict(sample={})
    for k, c in enumerate(H45.JC_CLASSES):
        J, inf = read_window(X.jc_file(c, 101), PER, 4601 + k); Js.append(J); meta['sample'][c] = inf
    J = cat(Js); y = J['label'].astype(int); P4raw, nn = X.jc_P4(J); P4 = H._identity(P4raw, nn).astype(np.float32)
    J['pmass'] = kin(J)[3].astype(np.float32)
    rows = np.flatnonzero(nn >= 3)
    E, OBS, INFO, LAB, chk = edit_set(P4, nn, rows, 4610, 'jc')
    D = {'nn': nn, 'y': y, 'id': P4}
    for (dc, kind), e in E.items():
        if kind == 'cn' and dc not in CN_DCS: continue
        D[f'P|{dc}|{kind}'] = e['P']
    for k in PK + ['pmass', 'jet_eta', 'jet_phi', 'jet_pt', 'label']: D[f'base|{k}'] = J[k]
    RES.mkdir(parents=True, exist_ok=True); np.savez_compressed(RES / f'{TAG}_jc_inputs.npz', **D)
    Os, ys = [], []
    for k, c in enumerate(H45.JC_CLASSES):
        Jr, inf = read_window(H45.TRAIN / f'{H45.JC_PRE[c]}_001.root', NREF, 4616 + k); meta.setdefault('ref_rows', {})[c] = inf
        Pr, nr = X.jc_P4(Jr); Pr = H._identity(Pr, nr).astype(np.float32); Os.append(H45.observables(Pr, nr)); ys.append(Jr['label'].astype(int))
    Op = {q: np.concatenate([o[q] for o in Os]) for q in H45.OBSK}; yp = np.concatenate(ys)
    meta['ref_fit'] = refs_and_save(OBS, E, INFO, y, Op, yp, H45.JC_SCORES, H45.JC_CLASSES, RES / f'{TAG}_jc_obs.npz', chk)
    meta['checks'] = chk; (RES / f'{TAG}_jc_meta.json').write_text(json.dumps(jsonable(meta), indent=1)); log('jc part written')


def part_nets():
    import torch
    from interp.h23_axis_audit import Nets
    from interp.part_core import ParTNetwork
    import interp.h23_axis_audit as H23
    H23.log = log
    Z = np.load(RES / f'{TAG}_jc_inputs.npz'); nn = Z['nn']; base = {k.split('|', 1)[1]: Z[k] for k in Z.files if k.startswith('base|')}
    N = Nets(top=False); names = list(H45.JC_NETS)
    for s in range(3):
        netp = ParTNetwork('full'); g = torch.Generator().manual_seed(15 + s)
        with torch.no_grad():
            for p in netp.model.parameters():
                if p.numel() > 1: p.copy_(p.flatten()[torch.randperm(p.numel(), generator=g).to(p.device)].view_as(p))
        N.jc[f'ParT_perm{s}'] = ('part', 'full', netp); names.append(f'ParT_perm{s}')
    out = {}
    for k in ['id'] + [k for k in Z.files if k.startswith('P|')]:
        P = Z[k]; lg = {}
        N.score('jc', P, nn, base, H45.axis(P, nn), None, jc_names=names, top=False, logits=lg)
        nm = 'id' if k == 'id' else k[2:]
        for m, L in lg.items(): out[f'L|{nm}|{m}'] = L.astype(np.float32)
        log(f'scored {nm}')
    y = Z['y']
    for m in names: log(f'{m}: 10-class accuracy on the fresh sample {float((out[f"L|id|{m}"].argmax(1) == y).mean()):.4f}')
    np.savez_compressed(RES / f'{TAG}_jc_logits.npz', **out)


def hls_draw_v():
    from jetdistill.config import RESULTS
    d = np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r'); y = np.asarray(d['y']).astype(int)
    r0 = np.random.default_rng(4504); orig = np.concatenate([r0.choice(np.flatnonzero(y == c), 2000, replace=False) for c in range(5)])
    rng = np.random.default_rng(4614)
    idx = np.sort(np.concatenate([rng.choice(np.setdiff1d(np.flatnonzero(y == c), orig), PER, replace=False) for c in range(5)]))
    return np.asarray(d['x'][idx], np.float32), y[idx], idx, int(len(np.intersect1d(idx, orig)))


def part_hls():
    import h5py
    import torch
    from interp import h4_top_wmass as H
    from interp.h1_irc_panel import load_small, torch_logits
    x, y, idx, ov = hls_draw_v(); real = x[..., 0] > 0; nn = real.sum(1).astype(np.int64)
    P4 = H._identity(H45.p4(x[..., 0].astype(np.float64), x[..., 1], x[..., 2], 0.0) * real[..., None], nn).astype(np.float32)
    log(f'hls fresh sample {len(y)} (rng 4614), overlap with H45 rows {ov}')
    rows = np.flatnonzero((nn >= 3) & (y != 4))
    E, OBS, INFO, LAB, chk = edit_set(P4, nn, rows, 4611, 'hls')
    D = {'y': y, 'idx': idx, 'nn': nn, 'X|id': H45.model_x(P4, nn, H45.axis(P4, nn))}
    for (dc, kind), e in E.items():
        if kind == 'cn' and dc not in CN_DCS: continue
        D[f'X|{dc}|{kind}'] = H45.model_x(e['P'], nn, H45.axis(e['P'], nn))
    RES.mkdir(parents=True, exist_ok=True); np.savez_compressed(RES / f'{TAG}_hls_inputs.npz', **D)
    with h5py.File(V / 'data' / 'train_meta.h5', 'r') as f: ya = f['label'][:].astype(np.int64)
    dev = np.asarray(np.load(V / 'data' / 'splits.npz')['dev'], np.int64); isdev = np.zeros(len(ya), bool); isdev[dev] = True
    rr = np.random.default_rng(4619); it = np.sort(np.concatenate([rr.choice(np.flatnonzero(~isdev & (ya == c)), NREF, replace=False) for c in range(4)]))
    xr = np.asarray(np.load(V / 'data' / 'train_ptetaphi_f16.npy', mmap_mode='r')[it, :64], np.float32); yr = ya[it]
    rl = xr[..., 0] > 0; nr = rl.sum(1).astype(np.int64)
    Pr = H._identity((H45.p4(xr[..., 0].astype(np.float64), xr[..., 1], xr[..., 2], 0.0) * rl[..., None]).astype(np.float32), nr).astype(np.float32)
    Op = H45.observables(Pr, nr)
    fit = refs_and_save(OBS, E, INFO, y, Op, yr, H45.HLS_SCORES, H45.HLS_CLASSES, RES / f'{TAG}_hls_obs.npz', chk)
    (RES / f'{TAG}_hls_meta.json').write_text(json.dumps(jsonable(dict(overlap_with_h45=ov, ref_fit=fit, checks=chk)), indent=1))
    # small nets
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
    d = V / 'models' / 'small' / 'dsets-d64-n64' / 'seed0'; models = {}
    net, sh, sc = load_small(d); models['dsets'] = (net, sh, sc)
    for s in range(2):
        netp, sh, sc = load_small(d); g = torch.Generator().manual_seed(17 + s)
        with torch.no_grad():
            for p in netp.parameters():
                f = p.data.flatten(); p.data.copy_(f[torch.randperm(f.numel(), generator=g)].view_as(p))
        models[f'dsets_perm{s}'] = (netp, sh, sc)
    Zi = np.load(RES / f'{TAG}_hls_inputs.npz'); out = {}
    for name, (n_, sh, sc) in models.items():
        for k in [k for k in Zi.files if k.startswith('X|')]: out[f'L|{k[2:]}|{name}'] = torch_logits(n_, np.asarray(Zi[k], np.float32).copy(), sh, sc).astype(np.float32)
        log(f'{name}: 5-class accuracy {float((out[f"L|id|{name}"].argmax(1) == y).mean()):.4f}')
    np.savez_compressed(RES / f'{TAG}_hls_logits_small.npz', **out)


def part_hjedi():
    from interp.jedi_core import Jedi
    nets = {'jedi': Jedi(64, '3-feature')}
    for s in (10, 11): nets[f'jedi_redrawn{s - 10}'] = Jedi(64, '3-feature', untrained=True, seed=s)
    Zi = np.load(RES / f'{TAG}_hls_inputs.npz'); y = Zi['y']; out = {}
    for name, n_ in nets.items():
        for k in [k for k in Zi.files if k.startswith('X|')]: out[f'L|{k[2:]}|{name}'] = np.asarray(n_.run(np.asarray(Zi[k], np.float32))['logits'], np.float32)
        log(f'{name}: 5-class accuracy {float((out[f"L|id|{name}"].argmax(1) == y).mean()):.4f}')
    np.savez_compressed(RES / f'{TAG}_hls_logits_jedi.npz', **out)


# ------------------------------------------------------------------------------------------------------------- statistics
def stats_ds(ds, orig):
    from scipy.stats import spearmanr
    Zo = dict(np.load(RES / f'{TAG}_{ds}_obs.npz')); y = Zo['y']
    if ds == 'jc':
        Zl = dict(np.load(RES / f'{TAG}_jc_logits.npz')); cls_of, scores, cells, nets = H45.JC_CLASSES, H45.JC_SCORES, H45.JC_CELLS, H45.JC_NETS
        nulls = [f'ParT_perm{s}' for s in range(3)]
    else:
        Zl = dict(np.load(RES / f'{TAG}_hls_logits_small.npz')); Zl.update(dict(np.load(RES / f'{TAG}_hls_logits_jedi.npz')))
        cls_of, scores, cells, nets = H45.HLS_CLASSES, H45.HLS_SCORES, H45.HLS_CELLS, ['jedi', 'dsets']
        nulls = ['jedi_redrawn0', 'jedi_redrawn1', 'dsets_perm0', 'dsets_perm1']
    names = {v: k for k, v in cls_of.items()}; cname = np.array([names.get(int(v), str(v)) for v in y])
    refk = sorted({k.split('|')[2] for k in Zo if k.startswith('R|id|')})

    def S(m, v, sc):
        if '@' in m: return Zo[f'R|{v}|{m}|{sc}'].astype(np.float64)
        return H45.score_of(ds, Zl[f'L|{v}|{m}'].astype(np.float64), sc)
    out = dict(checks=json.loads(str(Zo['checks'])), unedited={}, cells={}, obs={}, tests=[])
    for m in nets + nulls:
        L = Zl[f'L|id|{m}']; out['unedited'][m] = dict(acc=float((L.argmax(1) == y).mean()),
                                                         **{f'eff_{c}': float((L[cname == c].argmax(1) == cls_of[c]).mean()) for c in cls_of if (cname == c).any()})
    rng = np.random.default_rng(4650)
    for c in cls_of:
        if c == 't' or not (cname == c).any(): continue
        for dc in CN_DCS:
            rr_ = np.flatnonzero(Zo[f'acc|{dc}|D'] & Zo[f'acc|{dc}|cphi'] & Zo[f'acc|{dc}|cn'] & (cname == c)); ob = dict(n=int(len(rr_)))
            for v in VARS:
                ob[v] = {q: float((Zo[f'O|{dc}|{v}|{q}'][rr_] - Zo[f'O|id|{q}'][rr_]).mean()) for q in ('z', 'cos', 'zg', 'dR', 'tau21', 'D2', 'm', 'msub_hi', 'msub_lo')}
                ob[v]['disp_median'] = float(np.nanmedian(Zo[f'info|{dc}|disp_{v}'][rr_]))
            out['obs'][f'{dc}|{c}'] = ob
    thr = {(m, sc): float(np.median(S(m, 'id', sc)[cname == scores[sc][0]])) for m in nets for sc in scores}
    for (c, sc) in cells:
        for dc in DCS:
            acc = {k: Zo[f'acc|{dc}|{k}'] for k in VARS}; rp = np.flatnonzero(acc['D'] & acc['cphi'] & (cname == c))
            key = f'{c}|{sc}|{dc}'; R = dict(n=int(len(rp)), models={})
            if len(rp) < 20: out['cells'][key] = R; continue
            bi = rng.integers(0, len(rp), (B, len(rp)))
            hascn = dc in CN_DCS
            if hascn:
                rn = np.flatnonzero(acc['D'] & acc['cphi'] & acc['cn'] & (cname == c)); bin_ = rng.integers(0, len(rn), (B, len(rn))); R['n_cn'] = int(len(rn))
            dl = {}
            for m in nets + nulls + refk:
                s0 = S(m, 'id', sc); d = {v: S(m, f'{dc}|{v}', sc) - s0 for v in VARS if (v != 'cn' or hascn)}; dl[m] = d; Mm = {}
                Mm['D_minus_cphi'], Mm['D_minus_cphi_ci95'] = H45.boot((d['D'] - d['cphi'])[rp], bi); Mm['dcphi'] = float(d['cphi'][rp].mean())
                if hascn:
                    Mm['cn_minus_cphi'], Mm['cn_minus_cphi_ci95'] = H45.boot((d['cn'] - d['cphi'])[rn], bin_)
                    Mm['D_minus_cn'], Mm['D_minus_cn_ci95'] = H45.boot((d['D'] - d['cn'])[rn], bin_)
                if m in nets:
                    x = (d['D'] - d['cphi'])[rp]; Mm['D_minus_cphi_tp'] = H45.tp(x); Mm['D_minus_cphi_p'] = H45.perm2(x, rng)
                    out['tests'].append((key, m, 'D_minus_cphi_tp'))
                    if hascn:
                        x = (d['cn'] - d['cphi'])[rn]; Mm['cn_minus_cphi_tp'] = H45.tp(x); out['tests'].append((key, m, 'cn_minus_cphi_tp'))
                    t_ = thr[(m, sc)]; Mm['mcnemar_D_vs_cphi'] = H45.mcnemar(S(m, f'{dc}|cphi', sc)[rp] >= t_, S(m, f'{dc}|D', sc)[rp] >= t_)
                    o = orig.get('cells', {}).get(key, {}).get('models', {}).get(m, {}).get('D_minus_cphi')
                    Mm['h45_value'] = o
                R['models'][m] = Mm
            mx = np.mean([R['models'][r]['D_minus_cphi'] for r in refk if r.startswith('MLPx@')])
            R['MLPx_mean'] = float(mx)
            for m in nets:
                if abs(mx) >= 0.2: R['models'][m]['ratio_to_MLPx'] = float(R['models'][m]['D_minus_cphi'] / mx)
                x = (dl[m]['D'] - dl[m]['cphi'] - (dl['MLPx@0']['D'] - dl['MLPx@0']['cphi']))[rp]
                R['models'][m]['net_minus_MLPx0'], R['models'][m]['net_minus_MLPx0_ci95'] = H45.boot(x, bi)
            for con in ('D_minus_cphi', 'cn_minus_cphi'):
                v_ = [R['models'][m][con] for m in nulls if con in R['models'][m]]
                if v_: R[f'null_band_{con}'] = [min(v_), max(v_)]
                for b_ in ('MLPx', 'MLPz'):
                    v_ = [R['models'][r][con] for r in refk if r.startswith(b_ + '@') and con in R['models'][r]]
                    if v_: R[f'ref_range_{b_}_{con}'] = [min(v_), max(v_)]
            out['cells'][key] = R
            log(f'[{ds}] {key} n {len(rp)}: ' + ' | '.join(f"{m} D-cphi {R['models'][m]['D_minus_cphi']:+.2f} (H45 {R['models'][m]['h45_value']})"
                                                       + (f" cn-cphi {R['models'][m]['cn_minus_cphi']:+.2f}" if hascn else '') for m in nets)
                + f" | MLPx {R.get('ref_range_MLPx_D_minus_cphi')} null {R.get('null_band_D_minus_cphi')}")
    out['dose'] = {}
    for (c, sc) in cells:
        for m in nets:
            ys_ = [out['cells'][f'{c}|{sc}|{dc}']['models'][m]['D_minus_cphi'] for dc in DCS if m in out['cells'][f'{c}|{sc}|{dc}'].get('models', {})]
            if len(ys_) == 4: out['dose'][f'{c}|{sc}|{m}'] = dict(spearman=float(spearmanr(DCS, ys_)[0]), means=ys_)
    if ds == 'hls':
        Zi = np.load(RES / f'{TAG}_hls_inputs.npz')

        def mom(x):
            x = x.astype(np.float64); pt = x[..., 0]; M0 = pt.sum(1); M2 = (pt * (x[..., 1] ** 2 + x[..., 2] ** 2)).sum(1); return M0, np.sqrt(M0 * M2)
        M0i, mAi = mom(Zi['X|id']); out['moments'] = {}
        for dc in CN_DCS:
            for v in ('D', 'cn'):
                M0v, mAv = mom(Zi[f'X|{dc}|{v}'])
                for c in ('W', 'Z', 'q', 'g'):
                    rr_ = np.flatnonzero(Zo[f'acc|{dc}|D'] & Zo[f'acc|{dc}|cphi'] & (cname == c)); dlg = np.log(mAv[rr_] / mAi[rr_])
                    out['moments'][f'{dc}|{v}|{c}'] = dict(dlog_mA_mean=float(dlg.mean()), dlog_mA_p90abs=float(np.quantile(np.abs(dlg), 0.9)),
                                                           dM0_rel_p90abs=float(np.quantile(np.abs(M0v[rr_] / M0i[rr_] - 1), 0.9)))
    return out


def part_stats():
    op = RES / 'h45_prong_sharing.json'; orig = json.loads(op.read_text())['sets'] if op.exists() else {}
    R = dict(node='H45v', per_class=PER, n_ref=NREF, n_retrains=H45.NSEED, bootstrap=B, sets={})
    for ds in ('jc', 'hls'): R['sets'][ds] = stats_ds(ds, orig.get(ds, {}))
    lst = [(ds, k, m, p) for ds in R['sets'] for (k, m, p) in R['sets'][ds]['tests']]
    adj = H45.holm([R['sets'][ds]['cells'][k]['models'][m][p] for ds, k, m, p in lst])
    for (ds, k, m, p), a in zip(lst, adj): R['sets'][ds]['cells'][k]['models'][m][f'{p}_holm'] = float(a)
    R['holm'] = dict(n_tests=len(lst), n_D_cphi_lt05=int(sum(a < 0.05 for (ds, k, m, p), a in zip(lst, adj) if p.startswith('D_'))),
                     n_D_cphi=int(sum(p.startswith('D_') for *_r, p in lst)),
                     n_cn_cphi_lt05=int(sum(a < 0.05 for (ds, k, m, p), a in zip(lst, adj) if p.startswith('cn_'))),
                     n_cn_cphi=int(sum(p.startswith('cn_') for *_r, p in lst)))
    log(f'Holm {R["holm"]}')
    (RES / f'{TAG}.json').write_text(json.dumps(jsonable(R), indent=1)); log(f'wrote {TAG}.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--part', required=True); a = ap.parse_args()
    log(f'H45v part {a.part}; per class {PER}; smoke {SMOKE}')
    dict(jc=part_jc, nets=part_nets, hls=part_hls, hjedi=part_hjedi, stats=part_stats)[a.part](); log('done')
