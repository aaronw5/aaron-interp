"""P26v — verification of P26 (Sophon axis response) on fresh rows, plus controls P26 did not run.

Fresh samples (no row overlap with P26):
  A'  JetClass test_20M/{ZJetsToNuNu,ZToQQ,WToQQ}_113.root (P26 used _112), 5,000 random rows per class, default_rng(2699).
  B'  JetClass-II Res2P_0254..0259 + QCD_0351 (P26 used Res2P_0250..0253 + QCD_0350), same selection as P26, 2,000 per group,
      default_rng(2698).
Re-run of the P26 analysis (interp.p26_sophon_axis.analyse, unchanged) for sophon / part_full / part_kin / sophon_init0 on A'
and sophon / part_full on B'.

Added controls (per model, class token h = input of fc, 128-d):
  1. probe calibration: mean Δprobe(dil) / mean Δlog m_inv(dil) (1 = probe tracks the real mass change in full).
  2. probe-free ratio: u = unit vector of mean Δh(dil); R_h = mean(Δh(φ)·u) / mean(Δh(dil)·u); cos(mean Δh(φ), mean Δh(dil)).
  3. route split of the dilation response: 'dil_vec' = dilated four-vectors with the per-particle angular features (Δη, Δφ, ΔR)
     fed unedited (pair features + pT/E features move, axis-relative inputs do not); 'dil_ang' = unedited four-vectors with
     the dilated Δη, Δφ, ΔR (axis-relative inputs only). Shares = Δprobe(variant) / Δprobe(dil), also for the output d.
     An invariant-mass reader takes its dilation response through dil_vec; an axis reader through dil_ang (φ edit = ang only).
  4. paired bootstrap test (same jets, same weights) of R_probe(Sophon) − R_probe(other model); H0: difference = 0.
  ./part_job.sh c-p26-verify 'python -m interp.p26v_verify'
"""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp.part_core import build, kin, cat, OUT, CL
from interp.p1v_verify_part import Ridge, split
from interp.p14_planted_part import boot_weights, ci, slope_boot
from interp.e19_common import holm
from interp import p26_sophon_axis as P

T0 = time.time()
ANG = {'sophon': [4, 15, 16], 'full': [4, 15, 16], 'kin': [4, 5, 6]}   # deltaR, deta, dphi channels


def log(*a):
    print(f'[{time.time() - T0:7.0f} s]', *a, flush=True)


def sample_B(per, seed, files):
    """P26 sample_B with the shard list as an argument (body otherwise identical)"""
    import pyarrow.parquet as pq, awkward as ak
    rng = np.random.default_rng(seed); want = {0: per, 1: per, 2: per}; got = {0: [], 1: [], 2: []}; info = []
    for f in files:
        isqcd = f.name.startswith('QCD'); grp = [0] if isqcd else [1, 2]
        if all(sum(len(x['jet_pt']) for x in got[g]) >= want[g] for g in grp): continue
        cols = ['jet_pt', 'jet_sdmass', 'jet_label'] + ([] if isqcd else ['aux_genpart_mass', 'aux_genpart_isResX'])
        a = ak.from_arrow(pq.read_table(f, columns=cols))
        pt, sd, lab = (ak.to_numpy(a[k]) for k in ('jet_pt', 'jet_sdmass', 'jet_label'))
        sel = (pt > 500) & (pt < 1000) & (sd > 20) & (sd < 500)
        if isqcd:
            G = {0: sel & (lab >= 161)}
        else:
            mx = ak.to_numpy(ak.fill_none(ak.firsts(a['aux_genpart_mass'][a['aux_genpart_isResX']]), -1)); win = (mx > 70) & (mx < 100)
            G = {1: sel & win & np.isin(lab, [0, 1, 2]), 2: sel & win & np.isin(lab, [5, 7, 8])}
        rows = {}
        for g, s in G.items():
            need = want[g] - sum(len(x['jet_pt']) for x in got[g]); cand = np.flatnonzero(s)
            if need > 0 and len(cand): rows[g] = np.sort(rng.choice(cand, min(need, len(cand)), replace=False))
        info.append(dict(file=f.name, taken={int(g): int(len(r)) for g, r in rows.items()}))
        if rows:
            allr = np.sort(np.concatenate(list(rows.values())))
            full = pq.read_table(f, columns=[c for c in pq.ParquetFile(f).schema_arrow.names if c.startswith('part_')] +
                                 ['jet_pt', 'jet_eta', 'jet_phi', 'jet_energy', 'jet_sdmass', 'jet_label']).take(allr)
            J = P.jc2_to_dict(full); pos = {r: i for i, r in enumerate(allr)}
            for g, r in rows.items(): got[g].append({k: v[[pos[x] for x in r]] for k, v in J.items()})
        log(f"sample B' {info[-1]}")
    Js, y3 = [], []
    for g in (0, 1, 2):
        Jg = cat(got[g]); Js.append(Jg); y3.append(np.full(len(Jg['jet_pt']), g))
    J = cat(Js); y3 = np.concatenate(y3); J['label'] = np.array([P.iQ, P.iZ, P.iW])[y3].astype(np.int8)
    return J, y3, info


def extras(M, Jb, E, mi, y3, Wt_te, Wt_all):
    """controls 1-3; returns dict and the per-jet probe deltas on the test half (for the paired test)"""
    N = len(mi); fit, val, tr, te = split(N); logm = np.log(np.maximum(mi, 1.0))
    L0, H0 = M.run(Jb); rd = Ridge(H0, logm, fit, val, tr); d0 = M.read(L0)[0]
    ang = ANG[M.kind]; V = {}
    V['phi'] = M.run(E['phi']); V['dil'] = M.run(E['dil'])
    V['dil_vec'] = M.run(E['dil'], sub=(Jb, ang)); V['dil_ang'] = M.run(Jb, sub=(E['dil'], ang))
    V['rotc'] = M.run(E['rotc']); V['wta'] = M.run(E['wta']); V['trc'] = M.run(E['trc'])
    mi_dil = P.masses(E['dil'])[0]; dlogm_dil = np.log(np.maximum(mi_dil, 1.0)) - logm
    Dp = {k: rd(H[te]) - rd(H0[te]) for k, (L, H) in V.items()}
    Dd = {k: M.read(L)[0] - d0 for k, (L, H) in V.items()}
    Dh = {k: H - H0 for k, (L, H) in V.items()}
    u = Dh['dil'].mean(0); u /= np.linalg.norm(u) + 1e-12
    proj = {k: Dh[k] @ u for k in Dh}
    rr = lambda a, b, W: [float(a.mean() / b.mean())] + ci((W @ a) / (W @ b))
    out = dict(probe_r2_test=float(1 - ((rd(H0[te]) - logm[te]) ** 2).sum() / ((logm[te] - logm[te].mean()) ** 2).sum()),
               probe_calibration_dil=rr(Dp['dil'], dlogm_dil[te], Wt_te),
               dprobe_mean={k: float(v.mean()) for k, v in Dp.items()},
               R_probe=rr(Dp['phi'], Dp['dil'], Wt_te),
               R_h_proj=rr(proj['phi'], proj['dil'], Wt_all),
               cos_mean_dh_phi_dil=float(Dh['phi'].mean(0) @ Dh['dil'].mean(0) / (np.linalg.norm(Dh['phi'].mean(0)) * np.linalg.norm(Dh['dil'].mean(0)) + 1e-12)),
               norm_mean_dh={k: float(np.linalg.norm(v.mean(0))) for k, v in Dh.items()},
               rms_dh={k: float(np.sqrt((v ** 2).sum(1).mean())) for k, v in Dh.items()},
               R_h_proj_wta=rr(proj['wta'], proj['dil'], Wt_all), R_h_proj_trc=rr(proj['trc'], proj['dil'], Wt_all), R_h_proj_rotc=rr(proj['rotc'], proj['dil'], Wt_all),
               route_dil_probe={k: rr(Dp[k], Dp['dil'], Wt_te) for k in ('dil_vec', 'dil_ang')},
               route_dil_proj={k: rr(proj[k], proj['dil'], Wt_all) for k in ('dil_vec', 'dil_ang')},
               dd_mean={k: float(v.mean()) for k, v in Dd.items()},
               dd_mean_W={k: float(v[y3 == 2].mean()) for k, v in Dd.items()})
    log(f'{M.name} extras: calib {out["probe_calibration_dil"]}, R_probe {out["R_probe"]}, R_h_proj {out["R_h_proj"]}, cos {out["cos_mean_dh_phi_dil"]:.3f}, '
        f'route probe {out["route_dil_probe"]}, route proj {out["route_dil_proj"]}, proj wta/trc/rotc {out["R_h_proj_wta"][0]:.3f}/{out["R_h_proj_trc"][0]:.3f}/{out["R_h_proj_rotc"][0]:.3f}')
    return out, (Dp['phi'], Dp['dil']), (proj['phi'], proj['dil'])


def main():
    import torch
    OUT.mkdir(parents=True, exist_ok=True); fn = OUT / 'p26_verify_run.json'
    rng = np.random.default_rng(2699)
    sd = torch.load(P.SOPHON / 'model.pt', map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd: sd = sd['state_dict']
    if not all(k.startswith('mod.') for k in sd): sd = {'mod.' + k: v for k, v in sd.items()}
    mk = {'sophon': lambda: P.Model('sophon', sd), 'part_full': lambda: P.Model('part_full'), 'part_kin': lambda: P.Model('part_kin'),
          'sophon_init0': lambda: P.Model('sophon_init0')}
    res = dict(seeds=dict(A=2699, B=2698, analysis=2699), files=dict(A='test_20M/*_113.root', B='Res2P_0254..0259, QCD_0351'), rows={}, extras={}, samples={}, tests={})
    for sname in ('A', 'B'):
        if sname == 'A':
            J0, y3 = P.sample_A(5000, seed=2699, idx=113)
        else:
            files = [P.JC2 / f'Res2P_{i:04d}.parquet' for i in range(254, 260)] + [P.JC2 / 'QCD_0351.parquet']
            J0, y3, binfo = sample_B(2000, 2698, files)
        Jb, E, mi, info = P.edits(J0, rng); info.update(N=int(len(y3)), per_class=[int((y3 == g).sum()) for g in range(3)])
        res['samples'][sname] = info; log(f"sample {sname}': N {len(y3)} per class {info['per_class']}; edits {info['edits']}")
        N = len(y3); te = split(N)[3]; Wt_te = boot_weights(len(te), rng, 2000); Wt_all = boot_weights(N, rng, 2000); DP, PJ = {}, {}
        for nm in (['sophon', 'part_full', 'part_kin', 'sophon_init0'] if sname == 'A' else ['sophon', 'part_full']):
            M = mk[nm]()
            row, _ = P.analyse(M, Jb, E, y3, mi, rng); row['verdict'] = P.verdict(row); res['rows'][f'{nm}__{sname}'] = row
            ex, DP[nm], PJ[nm] = extras(M, Jb, E, mi, y3, Wt_te, Wt_all); res['extras'][f'{nm}__{sname}'] = ex
            del M; torch.cuda.empty_cache() if torch.cuda.is_available() else None
            json.dump(res, open(fn, 'w'), indent=1)
        for other in [k for k in DP if k != 'sophon']:
            for lab, D, W in (('R_probe', DP, Wt_te), ('R_h_proj', PJ, Wt_all)):
                a, b = D['sophon'], D[other]
                diff_b = (W @ a[0]) / (W @ a[1]) - (W @ b[0]) / (W @ b[1]); obs = a[0].mean() / a[1].mean() - b[0].mean() / b[1].mean()
                p = float(min(1.0, 2 * (1 + min((diff_b >= 0).sum(), (diff_b <= 0).sum())) / (1 + len(diff_b))))
                res['tests'][f'{lab}_sophon_minus_{other}_{sname}'] = dict(diff=float(obs), ci=ci(diff_b), p=p)
        del J0, Jb, E
    ks = list(res['tests']); hp = holm([res['tests'][k]['p'] for k in ks])
    for k, h in zip(ks, hp): res['tests'][k]['p_holm'] = h
    res['seconds'] = time.time() - T0; json.dump(res, open(fn, 'w'), indent=1)
    log(f'wrote {fn}; tests ' + '; '.join(f'{k} {v["diff"]:+.3f} {v["ci"]} holm {v["p_holm"]:.1e}' for k, v in res['tests'].items()))


if __name__ == '__main__':
    main()
