"""E14 verification (skeptic rerun of interp/e14_halfplanes_jedi.py on a fresh sample, plus controls E14 did not run).

Sample: sample(per_class=2000, seed=31) = 10,000 test jets (index overlap with the E14 sample, seed 8, and E6v, seed 17, reported).
Reference: the fixed-point network (jedi_core.Jedi).  Program: E14's float + zero-bit-mask Program (imported, unchanged).
R1  fresh-sample rerun of E14 Part 1 (variance shares, HP via g, s paths), Part 2 contours + polygon + controls (with this
    sample's mean g), Part 3 (rigid translation δ 0.03 / 0.06 along ±η, ±φ, ±diag: measured g free / g frozen vs maps; program
    counterfactuals point-symmetrized, even in η, even in φ, 8-rotation average) and Part 4 (one half-plane statement → 0).
V1  the symmetrized counterfactuals computed on the fixed-point network instead of the program: with g frozen at the unedited
    jet's g, S is a sum of per-particle terms, so ½[S(x') + S(−x')] (point), ½[S(x') + S(M_η x')] (η), ½[S(x') + S(M_φ x')] (φ)
    are the S of the symmetrized per-particle map; ΔS goes through the fixed-point head (S override): Δlog m̂ (h5 ridge), W→Z.
    Identity check: the S override with the unsymmetrized ΔS reproduces the g-frozen run.
V2  map vs sample: odd part of the fixed-point network (g free; g frozen at g(x)) on the point-reflected sample −x.  With
    odd(x) = A + B and odd(−x) = A − B, A is the part carried by the network's per-particle response and B the part from the
    sample's asymmetry (B equals the point-symmetrized counterfactual's odd part).
V3  baselines for the Part-2 contour controls: best circle, Fourier m ≤ 1, m ≤ 2, m ≤ 4 fits of r_c(θ) (rms relative error),
    and the number of half-plane directions with nonzero Part-1 weight per unit.
V4  (--head) why the point-symmetrized counterfactual depends on the head: see head_main.
Run:  python -m interp.e14_verify_jedi [--quick] [--head | --ci]   (--ci: 95 % bootstrap intervals, e14_verify_ci.json)
Out:  $AI_RESULTS/jedi/e14_verify.json (e14_verify_quick.json with --quick); --head: e14_verify_head.json
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
QUICK = '--quick' in sys.argv
SEED, PER = 31, (40 if QUICK else 2000)
JSON_PATH = OUT / ('e14_verify_quick.json' if QUICK else 'e14_verify.json')
import interp.e14_halfplanes_jedi as E   # noqa: E402

if QUICK:
    E.NPERM = 20; E.NCTRL = 20
DIRS, AXES, DELTAS, HP, MAIN = E.DIRS, E.AXES, E.DELTAS, E.HP, E.MAIN
OPP = {'+eta': '-eta', '-eta': '+eta', '+phi': '-phi', '-phi': '+phi', '+diag': '-diag', '-diag': '+diag'}
W_, Z_ = 2, 3
_LOG = []


def log(*a):
    s = ' '.join(str(v) for v in a); print(s, flush=True); _LOG.append(s)


def preflect(x):
    """x -> -x for real particles (Δη, Δφ negated; padded slots untouched)"""
    e = x.copy(); rl = e[..., 0] > 0
    e[..., 1] = np.where(rl, -x[..., 1], x[..., 1]); e[..., 2] = np.where(rl, -x[..., 2], x[..., 2]); return e.astype(np.float32)


def sample_idx(per, seed):
    """the row indices jedi_core.sample(per, seed) draws (same code)"""
    from jetdistill.config import RESULTS
    d = np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r'); y = np.asarray(d['y']); rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), per, replace=False) for c in range(5)]))


def eo(v):
    out = {}
    for ax in AXES:
        ev = 0.5 * (v[f'+{ax}'] + v[f'-{ax}']); od = 0.5 * (v[f'+{ax}'] - v[f'-{ax}'])
        out[ax] = dict(even=float(ev), odd=float(od), odd_over_even=float(od / ev) if abs(ev) > 1e-12 else None)
    return out


def head_fixed(net, x, S_in, batch=2500):
    """fixed-point head (d4 input quantizer onward) on a given S (J, 64): S override of all 64 units -> h5, logits, S"""
    kd = bool(getattr(net.L['qs1'], 'keepdims', False))
    one = np.ones(64, np.float32); out = dict(h5=[], logits=[], S=[])
    for i in range(0, len(x), batch):
        v = np.asarray(S_in[i:i + batch], np.float32)
        if kd: v = v[:, None, :]
        r = net._f(net.inputs(x[i:i + batch], net.n), {'S': (one, v)})
        for k in out: out[k].append(np.asarray(r[k], np.float32).reshape(len(v), -1))
    return {k: np.concatenate(v) for k, v in out.items()}


def baselines(rc):
    """rms relative error of least-squares fits of r_c(θ) with Fourier modes m ≤ M (M = 0: best circle about the axis)"""
    th = np.radians(E.TH_DEG); ok = np.isfinite(rc); r = rc[ok]; t = th[ok]; w = 1.0 / r; out = {}
    for M in (0, 1, 2, 4):
        A = np.stack([np.ones_like(t)] + [f(m * t) for m in range(1, M + 1) for f in (np.cos, np.sin)], 1)
        coef = np.linalg.lstsq(A * w[:, None], r * w, rcond=None)[0]; p = A @ coef
        out[f'fourier_m_le_{M}'] = float(np.sqrt(np.mean(((p - r) / r) ** 2)))
    return out


def main():
    from interp.jedi_core import Jedi, sample
    from interp.e4_e5_jedi import masks, make_masked
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True); np.seterr(all='ignore')
    x, y, Q = sample(PER, seed=SEED); x = np.asarray(x, np.float32); J = len(x); del Q
    i_new = sample_idx(PER, SEED); ov = {f'seed{s}': int(np.intersect1d(i_new, sample_idx(2000, s)).size) for s in (8, 17)}
    res = dict(N=J, per_class=PER, seed=SEED, quick=QUICK, overlap_rows=ov)
    log(f'E14 verify: {J} jets ({PER}/class, seed {SEED}); rows shared with seed 8 (E14): {ov["seed8"]}, seed 17 (E6v): {ov["seed17"]}  ({time.time()-t0:.0f}s)')
    real = x[..., 0] > 0; th_deg = np.degrees(np.arctan2(x[..., 2], x[..., 1])) % 360; sector = (th_deg // (360 / E.NSEC)).astype(int)
    net = Jedi(64); M3, KEEP = masks(net); LO, HI = E.range_tables(net)
    K1, b1 = net.KB['l1']
    nvec = np.stack([K1[1, HP] / net.scale[1], K1[2, HP] / net.scale[2]], 1)
    beta = np.array([b1[j] - (K1[:, j] * net.shift / net.scale).sum() for j in HP])
    ang = np.degrees(np.arctan2(nvec[:, 1], nvec[:, 0])) % 360
    res['qs1_keepdims'] = bool(getattr(net.L['qs1'], 'keepdims', False))

    # ---------------- reference
    R0 = E.fixed(net, x, keep=('g', 'S', 'logits', 'h5'))
    call0 = R0['logits'].argmax(1); sdS = R0['S'].std(0); live = [int(c) for c in np.flatnonzero(sdS > 1e-9)]; nl = len(live)
    res['acc_fixed'] = float((call0 == y).mean()); res['live_u'] = live
    prog = E.Program(net)
    Mf = make_masked(net, live)(x)
    Pf = prog.run(x, keep=('P', 'g', 'u', 'S', 'logits', 'h5'), live=live)
    res['program'] = dict(acc=float((Pf['logits'].argmax(1) == y).mean()), calls_changed_vs_fixed=float((Pf['logits'].argmax(1) != call0).mean()),
                          max_abs_diff_vs_make_masked_u=float(np.abs(Pf['u'] - Mf['u']).max()))
    del Mf
    gmf = Pf['g'].mean(0); Pmean = Pf['P'].mean(0); del Pf['P'], Pf['u']
    Pmu = prog.run(x, gov=gmf, keep=('u', 'S', 'logits'), live=live)
    res['program']['acc_g_mean'] = float((Pmu['logits'].argmax(1) == y).mean())
    log(f"fixed point acc {res['acc_fixed']:.4f}; live {live}; program acc {res['program']['acc']:.4f}, calls changed {res['program']['calls_changed_vs_fixed']*100:.2f}%, "
        f"max|u - make_masked| {res['program']['max_abs_diff_vs_make_masked_u']:.1e}; program g at mean {res['program']['acc_g_mean']:.4f}; qs1 keepdims {res['qs1_keepdims']}  ({time.time()-t0:.0f}s)")

    # ---------------- R1 Part 1
    P1 = E.part1(prog, x, live, KEEP, real, Pmean)
    hpk = [f'a{j}' for j in HP]; P1s = {}
    for c in live:
        d = P1[c]; g_ = d['groups']
        paths = [dict(s=p['s'], W3=round(p['W3'], 3), hp={k: v for k, v in p['W2'].items() if k in hpk}) for p in d['paths']]
        P1s[c] = dict(HP=g_['HP']['share'], PT=g_['PT']['share'], CONST=g_['CONST']['share'], bias=g_['bias']['share'], ctx=g_['ctx']['share'],
                      ctx_hp=g_['ctx_ang']['share'], frac=d['hp_over_hp_plus_ctx_ang'], err=d['decomposition_max_err'],
                      hp_paths=[p for p in paths if p['hp']], n_dirs=int(sum(abs(d['eff_weight_active'][k]) > 1e-9 for k in hpk)),
                      eff_hp={k: d['eff_weight_active'][k] for k in hpk})
        log(f"P1 u{c:<3d} HP {100*P1s[c]['HP']:6.1f} PT {100*P1s[c]['PT']:6.1f} bias {100*P1s[c]['bias']:6.1f} ctx {100*P1s[c]['ctx']:6.1f} ctx_HP {100*P1s[c]['ctx_hp']:6.1f} "
            f"frac {E.fmt(P1s[c]['frac'])} n_dirs {P1s[c]['n_dirs']} paths " + '; '.join(f"s{p['s']}({p['W3']:+.2f}):{','.join(p['hp'])}" for p in P1s[c]['hp_paths']) + f" err {P1s[c]['err']:.1e}")
    res['part1'] = P1s
    log(f'   ({time.time()-t0:.0f}s)')

    # ---------------- R1 Part 2 + V3: contours, polygon, controls, baselines
    th = np.radians(E.TH_DEG); RRf, TTf = np.meshgrid(E.R_FINE, th, indexing='ij'); EF, PF = (RRf * np.cos(TTf)).ravel(), (RRf * np.sin(TTf)).ravel()
    rng = np.random.default_rng(SEED); V3 = []
    for slot in E.SLOTS:
        for p in E.PT_GRID:
            pf = prog.particle(slot, np.full(EF.size, p), EF, PF, gmf)[:, live].reshape(len(E.R_FINE), len(th), nl)
            for q, c in enumerate(live):
                rc, nx = E.first_cross(pf[..., q], E.R_FINE); ok = np.isfinite(rc)
                if ok.mean() < 0.9: continue
                st = E.contour_stats(rc)
                wg = np.array([P1[c]['eff_weight_active'][k] for k in hpk]); ws = np.array([P1[c]['eff_weight_slot'][str(slot)][k] for k in hpk])
                row = dict(slot=slot, pt=p, unit=c, ratio=st.get('ratio'), theta_min=st.get('theta_min_deg'), mean=st.get('mean'), n_cross_max=int(nx.max()),
                           n_dirs=int((np.abs(wg) > 1e-9).sum()), baselines=baselines(rc))
                if np.abs(wg).sum() > 0 or np.abs(ws).sum() > 0:
                    Gl = prog.grad_a(slot, p, rc[ok] * np.cos(th[ok]), rc[ok] * np.sin(th[ok]), c)[:, HP].mean(0)
                    sg = np.sign(wg.sum()) if wg.sum() != 0 else 1.0
                    Wd = {'global': wg, 'slot': ws, 'local': Gl, 'equal_all9': sg * np.ones(len(HP))}
                    pc = E.polygon_compare(rc, Wd, nvec, beta, rng, controls=(slot == 5 and p in E.CTRL_PT))
                    for k in ('global', 'slot', 'local', 'fitted', 'equal_all9'):
                        row[f'{k}_rms'] = (pc.get(k) or {}).get('rms_rel')
                    for k in ('perm_global', 'random_dirs', 'rotated_dirs'):
                        if k in pc: row[k] = dict(median=pc[k]['rms_rel_median'], p_le=pc[k]['p_le'])
                V3.append(row)
    res['part2'] = V3
    ctrl = [r for r in V3 if 'random_dirs' in r]
    log(f'P2/V3: {len(V3)} complete contours; max/min {min(r["ratio"] for r in V3 if r["ratio"]):.3f}–{max(r["ratio"] for r in V3 if r["ratio"]):.3f}; '
        f'global-weight polygon rms median {np.nanmedian([r.get("global_rms", np.nan) if r.get("global_rms") is not None else np.nan for r in V3]):.1e}, '
        f'max {np.nanmax([r.get("global_rms") if r.get("global_rms") is not None else np.nan for r in V3]):.3f}')
    for r in ctrl:
        b = r['baselines']
        log(f"  ctrl slot {r['slot']} {r['pt']:.0f} GeV u{r['unit']:<3d} n_dirs {r['n_dirs']} max/min {r['ratio']:.3f} | polygon global {E.fmt(r.get('global_rms'), '{:.4f}')} fitted {E.fmt(r.get('fitted_rms'), '{:.4f}')} | "
            f"random dirs median {E.fmt(r['random_dirs']['median'], '{:.4f}')} p {E.fmt(r['random_dirs']['p_le'], '{:.3f}')} rotated {E.fmt(r['rotated_dirs']['median'], '{:.4f}')} p {E.fmt(r['rotated_dirs']['p_le'], '{:.3f}')} "
            f"perm {E.fmt(r['perm_global']['median'], '{:.4f}')} | circle {b['fourier_m_le_0']:.4f} m≤1 {b['fourier_m_le_1']:.4f} m≤2 {b['fourier_m_le_2']:.4f} m≤4 {b['fourier_m_le_4']:.4f}")
    log(f'   ({time.time()-t0:.0f}s)')

    # ---------------- R1 Part 3 + V1 + V2
    lm = np.log(np.maximum(E.inv_mass(x), 1e-3)); half = np.random.default_rng(SEED + 1).permutation(J) < J // 2
    rd = E.ridge(R0['h5'][half], lm[half]); lm0 = rd(R0['h5']); lm0_prog = rd(Pf['h5'])
    res['mass_readout_r2_test'] = float(1 - np.mean((lm0[~half] - lm[~half]) ** 2) / np.var(lm[~half]))
    log(f"\nP3: ridge log m on fixed h5: held-out R² {res['mass_readout_r2_test']:.3f}")
    g0 = R0['g']; xR = preflect(x); xMe = E.mirror(x, 1); xMp = E.mirror(x, 2)
    Bz = {k: E.fixed(net, xx, keep=('S', 'logits', 'h5'), g_jet=g0) for k, xx in (('R', xR), ('Me', xMe), ('Mp', xMp))}
    BfR = E.fixed(net, xR, keep=('S', 'logits', 'h5'))
    lmR_free0 = rd(BfR['h5']); lmR_froz0 = rd(Bz['R']['h5'])

    def prog_cf(xe):
        Sm = prog.run(xe, gov=gmf, keep=('S',))['S']
        rots = [Sm] + [prog.run(preflect(xe) if k == 4 else E.rotate(xe, 45.0 * k), gov=gmf, keep=('S',))['S'] for k in range(1, 8)]
        Se = prog.run(E.mirror(xe, 1), gov=gmf, keep=('S',))['S']; Sp = prog.run(E.mirror(xe, 2), gov=gmf, keep=('S',))['S']
        return dict(maps=Sm, maps_point_even=0.5 * (Sm + rots[4]), maps_even_in_eta=0.5 * (Sm + Se), maps_even_in_phi=0.5 * (Sm + Sp), maps_radial=np.mean(rots, 0))
    CF0 = prog_cf(x); call0_prog = Pf['logits'].argmax(1)
    P3 = {}; id_err = 0.0
    for dd in DELTAS:
        for name, (de, dp) in DIRS.items():
            key = f'{dd}_{name}'; xe = E.shift(x, de * dd, dp * dd); xeR = preflect(xe)
            ok = E.in_range(net, xe, LO, HI) & E.in_range(net, xeR, LO, HI); okW = ok & (y == W_)
            Rf = E.fixed(net, xe, keep=('S', 'logits', 'h5')); Rz = E.fixed(net, xe, keep=('S', 'logits', 'h5'), g_jet=g0)
            FzR = E.fixed(net, xeR, keep=('S', 'logits', 'h5'), g_jet=g0)
            FzMe = E.fixed(net, E.mirror(xe, 1), keep=('S',), g_jet=g0)['S']; FzMp = E.fixed(net, E.mirror(xe, 2), keep=('S',), g_jet=g0)['S']
            FfR = E.fixed(net, xeR, keep=('S', 'logits', 'h5'))
            CFe = prog_cf(xe)
            dS = dict(fixed_free=Rf['S'] - R0['S'], fixed_frozen=Rz['S'] - R0['S'],
                      fx_point_even=0.5 * (Rz['S'] + FzR['S']) - 0.5 * (R0['S'] + Bz['R']['S']),
                      fx_even_in_eta=0.5 * (Rz['S'] + FzMe) - 0.5 * (R0['S'] + Bz['Me']['S']),
                      fx_even_in_phi=0.5 * (Rz['S'] + FzMp) - 0.5 * (R0['S'] + Bz['Mp']['S']))
            for v in E.CF: dS['prog_' + v] = CFe[v] - CF0[v]
            mass = dict(fixed_free=rd(Rf['h5']) - lm0, fixed_frozen=rd(Rz['h5']) - lm0)
            calls = dict(fixed_free=Rf['logits'].argmax(1), fixed_frozen=Rz['logits'].argmax(1))
            for v in ('fixed_frozen', 'fx_point_even', 'fx_even_in_eta', 'fx_even_in_phi'):
                H = head_fixed(net, x, R0['S'] + dS[v]); nm = 'fx_identity' if v == 'fixed_frozen' else v
                mass[nm] = rd(H['h5']) - lm0; calls[nm] = H['logits'].argmax(1)
                if v == 'fixed_frozen': id_err = max(id_err, float(np.abs(H['h5'] - Rz['h5']).max()), float(np.abs(H['logits'] - Rz['logits']).max()))
            for v in E.CF:
                _, h5c, _, lo_ = prog.head(Pf['S'] + dS['prog_' + v]); mass['prog_' + v] = rd(h5c) - lm0_prog; calls['prog_' + v] = lo_.argmax(1)
            row = dict(n=int(ok.sum()), mass={k: dict(mean=float(v[ok].mean()), sem=float(v[ok].std() / np.sqrt(ok.sum()))) for k, v in mass.items()},
                       WZ={k: float((c_[okW] == Z_).mean()) for k, c_ in calls.items()},
                       calls_changed={k: float((c_ != (call0_prog if k.startswith('prog') else call0))[ok].mean()) for k, c_ in calls.items()},
                       units={}, refl={})
            # V2: the reflected sample −x translated along OPP(name)
            row['refl'] = dict(dir=OPP[name], mass_free=float((rd(FfR['h5']) - lmR_free0)[ok].mean()), mass_frozen=float((rd(FzR['h5']) - lmR_froz0)[ok].mean()),
                               WZ_free=float((FfR['logits'].argmax(1)[okW] == Z_).mean()),
                               units={c: dict(free=float((FfR['S'][:, c] - BfR['S'][:, c])[ok].mean() / sdS[c]),
                                              frozen=float((FzR['S'][:, c] - Bz['R']['S'][:, c])[ok].mean() / sdS[c])) for c in MAIN})
            for c in MAIN:
                d = {k: float(v[ok, c].mean() / sdS[c]) for k, v in dS.items()}
                a_, b_ = dS['prog_maps'][ok, c], dS['fixed_frozen'][ok, c]
                d['corr_maps_frozen'] = float(np.corrcoef(a_, b_)[0, 1]) if a_.std() > 0 and b_.std() > 0 else None
                row['units'][c] = d
            P3[key] = row
            m_ = row['mass']
            log(f"P3 {key:11s} n {row['n']} | Δlog m̂ free {m_['fixed_free']['mean']:+.4f} frozen {m_['fixed_frozen']['mean']:+.4f} (S-override identity {m_['fx_identity']['mean']:+.4f}) "
                f"maps {m_['prog_maps']['mean']:+.4f} | fx point {m_['fx_point_even']['mean']:+.4f} η {m_['fx_even_in_eta']['mean']:+.4f} φ {m_['fx_even_in_phi']['mean']:+.4f} | "
                f"prog point {m_['prog_maps_point_even']['mean']:+.4f} radial {m_['prog_maps_radial']['mean']:+.4f} | −x along {OPP[name]}: free {row['refl']['mass_free']:+.4f} frozen {row['refl']['mass_frozen']:+.4f} | "
                f"W→Z free {row['WZ']['fixed_free']:.3f} maps {row['WZ']['prog_maps']:.3f} fx point {row['WZ']['fx_point_even']:.3f}, −x {row['refl']['WZ_free']:.3f}  ({time.time()-t0:.0f}s)")
    res['part3'] = P3; res['s_override_identity_max_err'] = id_err
    log(f'S-override identity: max |h5, logits diff| {id_err:.2e}')
    # summary
    variants = list(P3[f'{DELTAS[0]}_+eta']['mass'])
    summ = {}
    for dd in DELTAS:
        S_ = dict(mass={}, WZ={}, units={}, ratios={})
        for v in variants:
            by = {n: P3[f'{dd}_{n}']['mass'][v]['mean'] for n in DIRS}; S_['mass'][v] = dict(by_dir=by, eo=eo(by))
            S_['WZ'][v] = {n: P3[f'{dd}_{n}']['WZ'][v] for n in DIRS}
        for v in ('free', 'frozen'):
            by = {P3[f'{dd}_{n}']['refl']['dir']: P3[f'{dd}_{n}']['refl'][f'mass_{v}'] for n in DIRS}; S_['mass'][f'refl_{v}'] = dict(by_dir=by, eo=eo(by))
        S_['WZ']['refl_free'] = {P3[f'{dd}_{n}']['refl']['dir']: P3[f'{dd}_{n}']['refl']['WZ_free'] for n in DIRS}
        for c in MAIN:
            u = {}
            for v in P3[f'{dd}_+eta']['units'][c]:
                if v.startswith('corr'): u[v] = {n: P3[f'{dd}_{n}']['units'][c][v] for n in DIRS}; continue
                by = {n: P3[f'{dd}_{n}']['units'][c][v] for n in DIRS}; u[v] = dict(by_dir=by, eo=eo(by))
            for v in ('free', 'frozen'):
                by = {P3[f'{dd}_{n}']['refl']['dir']: P3[f'{dd}_{n}']['refl']['units'][c][v] for n in DIRS}; u[f'refl_{v}'] = dict(by_dir=by, eo=eo(by))
            pm = np.array([u['prog_maps']['by_dir'][n] for n in DIRS]); mf = np.array([u['fixed_frozen']['by_dir'][n] for n in DIRS])
            u['slope_frozen_on_maps'] = float(pm @ mf / (pm @ pm)); u['spearman_maps_frozen'] = E.spearman(pm, mf)
            S_['units'][c] = u
        M_ = S_['mass']
        for ax in AXES:
            od = {v: M_[v]['eo'][ax]['odd'] for v in M_}
            S_['ratios'][ax] = dict(maps_over_frozen=od['prog_maps'] / od['fixed_frozen'], maps_over_free=od['prog_maps'] / od['fixed_free'],
                                    frozen_over_free=od['fixed_frozen'] / od['fixed_free'], fx_point_over_frozen=od['fx_point_even'] / od['fixed_frozen'],
                                    A_free=0.5 * (od['fixed_free'] + od['refl_free']), B_free=0.5 * (od['fixed_free'] - od['refl_free']),
                                    A_frozen=0.5 * (od['fixed_frozen'] + od['refl_frozen']), B_frozen=0.5 * (od['fixed_frozen'] - od['refl_frozen']))
        summ[str(dd)] = S_
        log(f'\nP3 summary δ = {dd}: Δlog m̂ (h5) by direction ' + ' '.join(DIRS) + ' | odd/even η φ diag | odd η φ diag')
        for v in M_:
            e_ = M_[v]['eo']
            s = (f'  {v:22s} ' + ' '.join(f"{M_[v]['by_dir'][n]:+.4f}" for n in DIRS) + ' | ' + ' '.join(E.fmt(e_[a]['odd_over_even'], '{:+.3f}') for a in AXES) +
                 ' | ' + ' '.join(f"{e_[a]['odd']:+.4f}" for a in AXES))
            if v in S_['WZ']: s += ' | W→Z ' + ' '.join(f"{S_['WZ'][v][n]:.2f}" for n in DIRS)
            log(s)
        log('  ratios of odd parts: ' + '; '.join(f"{ax}: maps/frozen {r['maps_over_frozen']:.2f}, maps/free {r['maps_over_free']:.2f}, frozen/free {r['frozen_over_free']:.2f}, "
                                                  f"fx point/frozen {r['fx_point_over_frozen']:+.3f}, A/B free {r['A_free']:+.4f}/{r['B_free']:+.4f}, A/B frozen {r['A_frozen']:+.4f}/{r['B_frozen']:+.4f}"
                                                  for ax, r in S_['ratios'].items()))
        for c in MAIN:
            u = S_['units'][c]
            log(f"  S{c}: slope frozen/maps {u['slope_frozen_on_maps']:.3f}, Spearman {u['spearman_maps_frozen']:.2f}; odd/even frozen " +
                ' '.join(E.fmt(u['fixed_frozen']['eo'][a]['odd_over_even'], '{:+.3f}') for a in AXES) + ' maps ' + ' '.join(E.fmt(u['prog_maps']['eo'][a]['odd_over_even'], '{:+.3f}') for a in AXES) +
                ' fx point ' + ' '.join(E.fmt(u['fx_point_even']['eo'][a]['odd_over_even'], '{:+.3f}') for a in AXES) + ' −x frozen ' + ' '.join(E.fmt(u['refl_frozen']['eo'][a]['odd_over_even'], '{:+.3f}') for a in AXES) +
                ' | per-jet corr maps/frozen ' + ' '.join(E.fmt(u['corr_maps_frozen'][n]) for n in DIRS))
    res['part3_summary'] = summ
    log(f'   ({time.time()-t0:.0f}s)')

    # ---------------- R1 Part 4
    U0 = E.fixed(net, x, keep=('u',), live=live)['u']
    dead_c = [j for j in range(64) if np.all(K1[:, j] == 0) and b1[j] <= 0]; dead = int(dead_c[0]) if dead_c else None
    P4 = {}
    for j, kind in [(j, 'halfplane') for j in HP] + ([(dead, 'dead')] if dead is not None else []) + [(56, 'pt_ramp')]:
        Rf = E.fixed(net, x, keep=('u', 'S', 'logits'), a_zero=j, live=live); Rz = E.fixed(net, x, keep=('u', 'S', 'logits'), a_zero=j, live=live, g_jet=g0)
        Qm = prog.run(x, gov=gmf, azero=j, keep=('u', 'S', 'logits'), live=live)
        row = dict(kind=kind, calls_free=float((Rf['logits'].argmax(1) != call0).mean()), calls_frozen=float((Rz['logits'].argmax(1) != call0).mean()),
                   calls_prog=float((Qm['logits'].argmax(1) != Pmu['logits'].argmax(1)).mean()))
        if kind == 'halfplane':
            a0 = float(ang[HP.index(j)]); own = int(a0 // (360 / E.NSEC)); opp = (own + E.NSEC // 2) % E.NSEC
            hf = [k for k in range(E.NSEC) if E.circ_diff((k + 0.5) * 360 / E.NSEC, a0) < 90]; row['angle_deg'] = a0
        tot = {k: np.zeros(E.NSEC) for k in ('free', 'frozen', 'prog')}; corrs = []
        for q, c in enumerate(live):
            wmask = real & KEEP[None, :, c]; shs = {}
            for k, dU in (('free', Rf['u'][..., q] - U0[..., q]), ('frozen', Rz['u'][..., q] - U0[..., q]), ('prog', Qm['u'][..., q] - Pmu['u'][..., q])):
                _, sh = E.sector_stats(dU, wmask, sector, sdS[c], J); tot[k] += np.abs(sh); shs[k] = np.array(sh)
            if shs['prog'].std() > 0 and shs['frozen'].std() > 0: corrs.append(float(np.corrcoef(shs['prog'], shs['frozen'])[0, 1]))
        row['median_corr_sectors_prog_vs_frozen'] = float(np.median(corrs)) if corrs else None
        for k in tot:
            row[k] = E.sel_metrics(tot[k], own, opp, hf) if kind == 'halfplane' else dict(total_abs=float(tot[k].sum()),
                                                                                      sector_cv=float(tot[k].std() / tot[k].mean()) if tot[k].mean() > 0 else None)
        P4[f'a{j}'] = row
        s = f"P4 a{j:<2d} {kind:9s} calls free {row['calls_free']*100:5.2f}% frozen {row['calls_frozen']*100:5.2f}% prog {row['calls_prog']*100:5.2f}%"
        if kind == 'halfplane':
            s += (f" | own-half frac frozen {E.fmt(row['frozen']['own_half_frac'], '{:.3f}')} free {E.fmt(row['free']['own_half_frac'], '{:.3f}')} | own/opp free {E.fmt(row['free']['own_over_opp'], '{:.1f}')} "
                  f"| opp sector frozen {row['frozen']['opp']:+.2e} | median corr sectors prog/frozen {E.fmt(row['median_corr_sectors_prog_vs_frozen'], '{:.4f}')}")
        else:
            s += f" | Σ|share| frozen {row['frozen']['total_abs']:.4f} sector CV {E.fmt(row['frozen']['sector_cv'])}"
        log(s + f'  ({time.time()-t0:.0f}s)')
    res['part4'] = P4
    res['elapsed_s'] = float(time.time() - t0); res['log'] = _LOG
    JSON_PATH.write_text(json.dumps(E.clean(res)))
    log(f'wrote {JSON_PATH} ({time.time()-t0:.0f}s)')


def head_main():
    """--head: why the point-symmetrized counterfactual depends on the head (follow-up to V1/V2).  Same sample (seed 31).
    All with g frozen at the unedited jet's g.  S_x' = fixed-point S of x', ΔS_x = S(x+δd) − S(x), ΔS_−x = S(−x−δd) − S(−x),
    S_sym(x') = ½[S(x') + S(−x')].  Mass read-out = the same ridge on fixed h5.  Variants (head: fx = fixed-point via S override,
    pg = float + mask program head):
      hybrid_fx / hybrid_pg   head(S_x + ½[ΔS_x + ΔS_−x]) − head(S_x)            (E14's construction)
      transplant_fx / _pg     head(S_x + ΔS_−x) − head(S_x)                       (reflected jet's ΔS on the original base)
      symnet_fx / symnet_pg   head(S_sym(x+δd)) − head(S_sym(x))                  (network with point-symmetrized per-particle maps)
      prog_symnet             program, g at mean: head(S_sym(x+δd)) − head(S_sym(x)) with the program's S
    plus the S-level odd/even of ½[ΔS_x + ΔS_−x] for all live units."""
    from interp.jedi_core import Jedi, sample
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True); np.seterr(all='ignore')
    x, y, Q = sample(PER, seed=SEED); x = np.asarray(x, np.float32); J = len(x); del Q
    net = Jedi(64); LO, HI = E.range_tables(net); prog = E.Program(net)
    R0 = E.fixed(net, x, keep=('g', 'S', 'logits', 'h5')); g0 = R0['g']; sdS = R0['S'].std(0); live = [int(c) for c in np.flatnonzero(sdS > 1e-9)]
    lm = np.log(np.maximum(E.inv_mass(x), 1e-3)); half = np.random.default_rng(SEED + 1).permutation(J) < J // 2
    rd = E.ridge(R0['h5'][half], lm[half])
    gmf = prog.run(x, keep=('g',))['g'].mean(0)
    xR = preflect(x); BR = E.fixed(net, xR, keep=('S',), g_jet=g0)['S']; S0 = R0['S']
    Ssym0 = 0.5 * (S0 + BR)
    QS0 = prog.run(x, gov=gmf, keep=('S',))['S']; QSR0 = prog.run(xR, gov=gmf, keep=('S',))['S']; Qsym0 = 0.5 * (QS0 + QSR0)

    def mfx(S): return rd(head_fixed(net, x, S)['h5'])
    def mpg(S): return rd(prog.head(S)[1])
    base = dict(hybrid_fx=mfx(S0), hybrid_pg=mpg(S0), symnet_fx=mfx(Ssym0), symnet_pg=mpg(Ssym0), prog_symnet=mpg(Qsym0))
    log(f'E14 verify --head: {J} jets (seed {SEED}); fixed head on S0 vs program head on S0: mean |Δ log m̂| {np.abs(base["hybrid_fx"] - base["hybrid_pg"]).mean():.4f} ({time.time()-t0:.0f}s)')
    rec = {}
    for dd in DELTAS:
        for name, (de, dp) in DIRS.items():
            key = f'{dd}_{name}'; xe = E.shift(x, de * dd, dp * dd); xeR = preflect(xe)
            ok = E.in_range(net, xe, LO, HI) & E.in_range(net, xeR, LO, HI)
            Se = E.fixed(net, xe, keep=('S',), g_jet=g0)['S']; SeR = E.fixed(net, xeR, keep=('S',), g_jet=g0)['S']
            dSx = Se - S0; dSmx = SeR - BR; dcf = 0.5 * (dSx + dSmx)
            QSe = prog.run(xe, gov=gmf, keep=('S',))['S']; QSeR = prog.run(xeR, gov=gmf, keep=('S',))['S']
            m = dict(frozen_fx=mfx(S0 + dSx) - base['hybrid_fx'], frozen_pg=mpg(S0 + dSx) - base['hybrid_pg'],
                     hybrid_fx=mfx(S0 + dcf) - base['hybrid_fx'], hybrid_pg=mpg(S0 + dcf) - base['hybrid_pg'],
                     transplant_fx=mfx(S0 + dSmx) - base['hybrid_fx'], transplant_pg=mpg(S0 + dSmx) - base['hybrid_pg'],
                     symnet_fx=mfx(0.5 * (Se + SeR)) - base['symnet_fx'], symnet_pg=mpg(0.5 * (Se + SeR)) - base['symnet_pg'],
                     prog_symnet=mpg(0.5 * (QSe + QSeR)) - base['prog_symnet'])
            rec[key] = dict(dir_minus_x=OPP[name], mass={k: float(v[ok].mean()) for k, v in m.items()},
                            S_cf_sd={c: float(dcf[ok, c].mean() / sdS[c]) for c in live})
            log(f'{key:11s} ' + ' '.join(f'{k} {v:+.4f}' for k, v in rec[key]['mass'].items()) + f'  ({time.time()-t0:.0f}s)')
    summ = {}
    for dd in DELTAS:
        s_ = dict(mass={}, S_cf={})
        for k in rec[f'{dd}_+eta']['mass']:
            by = {n: rec[f'{dd}_{n}']['mass'][k] for n in DIRS}; s_['mass'][k] = dict(by_dir=by, eo=eo(by))
        # transplant: the response of −x moved along OPP(name), placed on x's base; label by the −x direction
        for k in ('transplant_fx', 'transplant_pg'):
            by = {OPP[n]: rec[f'{dd}_{n}']['mass'][k] for n in DIRS}; s_['mass'][k + '_by_minus_x_dir'] = dict(by_dir=by, eo=eo(by))
        for c in live:
            by = {n: rec[f'{dd}_{n}']['S_cf_sd'][c] for n in DIRS}; s_['S_cf'][c] = eo(by)
        summ[str(dd)] = s_
        log(f'\nδ = {dd}: Δlog m̂ by direction ' + ' '.join(DIRS) + ' | odd/even η φ diag | odd η φ diag')
        for k, v in s_['mass'].items():
            log(f'  {k:28s} ' + ' '.join(f"{v['by_dir'][n]:+.4f}" for n in DIRS) + ' | ' + ' '.join(E.fmt(v['eo'][a]['odd_over_even'], '{:+.3f}') for a in AXES) +
                ' | ' + ' '.join(f"{v['eo'][a]['odd']:+.4f}" for a in AXES))
        log('  S-level ½[ΔS_x + ΔS_−x], odd part / sd(S) (η, φ, diag) and even part: ' +
            '; '.join(f"S{c} " + ' '.join(f"{s_['S_cf'][c][a]['odd']:+.3f}" for a in AXES) + f" (even {s_['S_cf'][c]['eta']['even']:+.3f})" for c in live))
    out = OUT / ('e14_verify_head_quick.json' if QUICK else 'e14_verify_head.json')
    out.write_text(json.dumps(E.clean(dict(N=J, seed=SEED, per_class=PER, records=rec, summary=summ, log=_LOG, elapsed_s=time.time() - t0))))
    log(f'wrote {out} ({time.time()-t0:.0f}s)')


def ci_main():
    """--ci: 95 % bootstrap intervals (1000 resamples over jets, paired) for the odd part of Δlog m̂ (h5 ridge) per axis:
    measured g free / g frozen (fixed point), maps (program, g at mean, E14 construction), E14's point-symmetrized hybrid
    (program), the reflected-sample responses (V2: A, B), and the symmetrized network (fixed head and program head);
    ratios maps/frozen, frozen/free, maps/free; H0 for B: B = 0 (two-sided bootstrap p)."""
    from interp.jedi_core import Jedi, sample
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True); np.seterr(all='ignore')
    x, y, Q = sample(PER, seed=SEED); x = np.asarray(x, np.float32); J = len(x); del Q
    net = Jedi(64); LO, HI = E.range_tables(net); prog = E.Program(net)
    R0 = E.fixed(net, x, keep=('g', 'S', 'logits', 'h5')); g0 = R0['g']
    lm = np.log(np.maximum(E.inv_mass(x), 1e-3)); half = np.random.default_rng(SEED + 1).permutation(J) < J // 2
    rd = E.ridge(R0['h5'][half], lm[half]); lm0 = rd(R0['h5'])
    Pf = prog.run(x, keep=('g', 'S', 'h5')); gmf = Pf['g'].mean(0); lmp0 = rd(Pf['h5'])
    xR = preflect(x); BfR = E.fixed(net, xR, keep=('S', 'h5')); BzR = E.fixed(net, xR, keep=('S', 'h5'), g_jet=g0)
    QS0 = prog.run(x, gov=gmf, keep=('S',))['S']; QSR0 = prog.run(xR, gov=gmf, keep=('S',))['S']
    Ssym0 = 0.5 * (R0['S'] + BzR['S']); mfx_sym0 = rd(head_fixed(net, x, Ssym0)['h5']); mpg_sym0 = rd(prog.head(Ssym0)[1])
    V = ('free', 'frozen', 'maps', 'hybrid_point_prog', 'refl_free', 'refl_frozen', 'symnet_fx', 'symnet_pg')
    D = {dd: {v: {} for v in V} for dd in DELTAS}; okall = np.ones(J, bool)
    for dd in DELTAS:
        for name, (de, dp) in DIRS.items():
            xe = E.shift(x, de * dd, dp * dd); xeR = preflect(xe)
            okall &= E.in_range(net, xe, LO, HI) & E.in_range(net, xeR, LO, HI)
            Rf = E.fixed(net, xe, keep=('h5',)); Rz = E.fixed(net, xe, keep=('S', 'h5'), g_jet=g0)
            FfR = E.fixed(net, xeR, keep=('h5',)); FzR = E.fixed(net, xeR, keep=('S', 'h5'), g_jet=g0)
            QSe = prog.run(xe, gov=gmf, keep=('S',))['S']; QSeR = prog.run(xeR, gov=gmf, keep=('S',))['S']
            d = D[dd]
            d['free'][name] = rd(Rf['h5']) - lm0; d['frozen'][name] = rd(Rz['h5']) - lm0
            d['maps'][name] = rd(prog.head(Pf['S'] + QSe - QS0)[1]) - lmp0
            d['hybrid_point_prog'][name] = rd(prog.head(Pf['S'] + 0.5 * (QSe + QSeR) - 0.5 * (QS0 + QSR0))[1]) - lmp0
            d['refl_free'][OPP[name]] = rd(FfR['h5']) - rd(BfR['h5']); d['refl_frozen'][OPP[name]] = rd(FzR['h5']) - rd(BzR['h5'])
            Ssym = 0.5 * (Rz['S'] + FzR['S'])
            d['symnet_fx'][name] = rd(head_fixed(net, x, Ssym)['h5']) - mfx_sym0; d['symnet_pg'][name] = rd(prog.head(Ssym)[1]) - mpg_sym0
        log(f'δ = {dd}: per-jet responses done ({time.time()-t0:.0f}s)')
    idx_ok = np.flatnonzero(okall); n = len(idx_ok); rng = np.random.default_rng(SEED + 7); NB = 50 if QUICK else 1000
    BI = rng.integers(0, n, size=(NB, n))
    res = dict(N=J, n_used=int(n), seed=SEED, n_boot=NB, per_axis={})
    for dd in DELTAS:
        res['per_axis'][str(dd)] = {}
        for ax in AXES:
            od = {v: (0.5 * (D[dd][v][f'+{ax}'] - D[dd][v][f'-{ax}']))[idx_ok] for v in V}
            ev = {v: (0.5 * (D[dd][v][f'+{ax}'] + D[dd][v][f'-{ax}']))[idx_ok] for v in V}
            q = dict(od, **{f'even_{v}': ev[v] for v in V})
            boot = {k: np.array([a[b].mean() for b in BI]) for k, a in q.items()}
            pt = {k: float(a.mean()) for k, a in q.items()}
            der = dict(A_free=lambda s: 0.5 * (s['free'] + s['refl_free']), B_free=lambda s: 0.5 * (s['free'] - s['refl_free']),
                       A_frozen=lambda s: 0.5 * (s['frozen'] + s['refl_frozen']), B_frozen=lambda s: 0.5 * (s['frozen'] - s['refl_frozen']),
                       maps_over_frozen=lambda s: s['maps'] / s['frozen'], maps_over_free=lambda s: s['maps'] / s['free'],
                       frozen_over_free=lambda s: s['frozen'] / s['free'], odd_over_even_free=lambda s: s['free'] / s['even_free'],
                       odd_over_even_maps=lambda s: s['maps'] / s['even_maps'])
            out = {}
            for k in q:
                out[f'odd_{k}' if not k.startswith('even_') else k] = dict(value=pt[k], ci95=[float(np.percentile(boot[k], 2.5)), float(np.percentile(boot[k], 97.5))])
            for k, f in der.items():
                b = f(boot); v = f(pt)
                out[k] = dict(value=float(v), ci95=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))])
                if k.startswith('B_'): out[k]['p_two_sided_H0_zero'] = float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean())))
            res['per_axis'][str(dd)][ax] = out
            log(f'δ = {dd} {ax}: ' + '; '.join(f"{k} {v['value']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]" + (f" p {v['p_two_sided_H0_zero']:.3f}" if 'p_two_sided_H0_zero' in v else '')
                                               for k, v in out.items() if not k.startswith('even_')))
    res['log'] = _LOG; res['elapsed_s'] = time.time() - t0
    out_p = OUT / ('e14_verify_ci_quick.json' if QUICK else 'e14_verify_ci.json'); out_p.write_text(json.dumps(E.clean(res)))
    log(f'wrote {out_p} ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    if '--head' in sys.argv: head_main()
    elif '--ci' in sys.argv: ci_main()
    else: main()
