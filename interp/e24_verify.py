"""E24 verification (fresh sample; independent edit code; added causal controls).

Sample: interp.jedi_core.sample(per_class=2000, seed=2501), N = 10,000 test jets (E24 used seed 2401).
Edits (re-implemented here, not imported from e24_azimuth_prong):
  rot, rot2  two independent rigid rotations about the stored axis (rot2 = null for rot-vs-rot McNemar)
  A          θ_i ~ U(0, 2π) per real particle (pT, r exact)
  A'         A + translate / rescale about the pT centroid so that M0, vec M1, M2 equal the original
  Bx         exact half-plane-cell scramble: at each particle's own radius r the nine official half-plane lines
             (α·(pT, Δη, Δφ) + β = 0, physical units) cut the circle into arcs; θ_i is redrawn uniformly inside the arc that
             contains it (all nine on/off states kept exactly, offsets included)
  Brx        control: the same construction with the arc boundaries rotated by φ_J ~ U(0, 2π) per jet
JEDI-only causal decomposition of the A' - rot call excess:
  gfreeze    g (dd output, the context vector) held at the jet's unedited value
  S58S31frz  S58 and S31 (functions of g only, E5) held at unedited values
  only58_31  unedited jet with only S58, S31 set to their edited values
Models: JEDI-linear official n64, Deep Sets d64 s0/s1, PFN 64-64-32 s0/s1 (loaders from e24_azimuth_prong.get_model).
Real-jet calibration (JEDI, Deep Sets s0): pool sample(20000, 2502) minus main jets; e24 calibration_pairs.
  ./job.sh e24-verify 'python -m interp.e24_verify'  ->  $AI_RESULTS/jedi/e24_verify.json
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
T0 = time.time(); NB = 1000
CL = ['g', 'q', 'W', 'Z', 't']
HP = [29, 6, 40, 57, 3, 43, 30, 36, 46]
RADIAL_S = [29, 1, 0, 19, 30]


def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.ndarray): return clean(o.tolist())
    return o


_B = {}


def bidx(n):
    if n not in _B: _B[n] = np.random.default_rng(2501).integers(0, n, (NB, n)).astype(np.int32)
    return _B[n]


def ci(v):
    v = np.asarray(v, np.float64); b = v[bidx(len(v))].mean(1)
    return dict(mean=float(v.mean()), ci=[float(np.quantile(b, .025)), float(np.quantile(b, .975))])


def ci_ratio(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64); I = bidx(len(a))
    r = a[I].mean(1) / b[I].mean(1)
    return dict(value=float(a.mean() / b.mean()), ci=[float(np.quantile(r, .025)), float(np.quantile(r, .975))])


def mcnemar(a, b):
    from math import lgamma, log as ln, exp
    a = np.asarray(a, bool); b = np.asarray(b, bool); n10 = int((a & ~b).sum()); n01 = int((~a & b).sum()); n = n10 + n01
    if n == 0: return dict(n10=0, n01=0, p=1.0)
    k = min(n10, n01); lp = [lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1) - n * ln(2) for i in range(k + 1)]
    mx = max(lp); return dict(n10=n10, n01=n01, p=float(min(1.0, 2 * exp(mx) * sum(exp(x - mx) for x in lp))))


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o): run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


# ------------------------------------------------------------------ edits (independent implementation)
def to_polar(x):
    x = np.asarray(x, np.float64); return np.hypot(x[..., 1], x[..., 2]), np.arctan2(x[..., 2], x[..., 1]), x[..., 0] > 0


def set_polar(x, r, th, real):
    e = np.asarray(x, np.float64).copy()
    e[..., 1] = np.where(real, r * np.cos(th), e[..., 1]); e[..., 2] = np.where(real, r * np.sin(th), e[..., 2]); return e


def mom(x):
    x = np.asarray(x, np.float64); pt = x[..., 0]; p = x[..., 1:3]
    return pt.sum(1), (pt[..., None] * p).sum(1), (pt * (p ** 2).sum(-1)).sum(1)


def rotate(x, seed):
    r, th, real = to_polar(x); phi = np.random.default_rng(seed).uniform(0, 2 * np.pi, (len(x), 1))
    return set_polar(x, r, th + phi, real)


def scramble(x, seed):
    r, th, real = to_polar(x); return set_polar(x, r, np.random.default_rng(seed).uniform(0, 2 * np.pi, th.shape), real)


def moment_hold(e, x):
    """affine map of the real particles of e about their pT centroid: centroid -> x's centroid; second central moment ->
    x's (M0 is untouched, so M0, vec M1, M2 all equal x's)"""
    M0, M1, M2 = mom(x); m0, m1, m2 = mom(e); c = M1 / M0[:, None]; c2 = m1 / m0[:, None]
    var, var2 = M2 / M0 - (c ** 2).sum(1), m2 / m0 - (c2 ** 2).sum(1)
    s = np.sqrt(var / var2)
    out = np.asarray(e, np.float64).copy(); real = out[..., 0] > 0
    q = c[:, None] + s[:, None, None] * (out[..., 1:3] - c2[:, None])
    out[..., 1:3] = np.where(real[..., None], q, out[..., 1:3]); return out


def cell_scramble(x, alpha, beta, seed, rot_seed=None):
    """exact arc of the circle of radius r_i bounded by the crossings of the half-plane lines (α_e Δη + α_p Δφ + α_0 pT
    + β = 0); θ_i redrawn uniformly in its arc. rot_seed: crossings rotated by a per-jet angle (control)."""
    x64 = np.asarray(x, np.float64); r, th, real = to_polar(x64); pt = x64[..., 0]
    off = np.zeros((len(x), 1)) if rot_seed is None else np.random.default_rng(rot_seed).uniform(0, 2 * np.pi, (len(x), 1))
    cr = []
    for (a0, ae, ap), b in zip(alpha, beta):
        A = np.hypot(ae, ap); psi = np.arctan2(ap, ae); rhs = -(b + a0 * pt)                 # A r cos(θ-ψ) = rhs
        with np.errstate(divide='ignore', invalid='ignore'):
            cosv = rhs / (A * r)
        ok = np.abs(cosv) < 1; d = np.arccos(np.clip(cosv, -1, 1))
        cr += [np.where(ok, psi + d + off, np.nan), np.where(ok, psi - d + off, np.nan)]
    C = np.stack(cr, -1)                                                                      # (J, n, 18)
    fwd = np.mod(C - th[..., None], 2 * np.pi); bwd = np.mod(th[..., None] - C, 2 * np.pi)
    fwd = np.where(np.isnan(C), np.inf, fwd); bwd = np.where(np.isnan(C), np.inf, bwd)
    up = fwd.min(-1); lo = -bwd.min(-1)
    full = ~np.isfinite(up); up = np.where(full, np.pi, up); lo = np.where(full, -np.pi, lo)
    u = np.random.default_rng(seed).uniform(0, 1, th.shape)
    return set_polar(x64, r, th + lo + u * (up - lo), real), np.abs(np.angle(np.exp(1j * (lo + u * (up - lo)))))[real]


def checks(x, E):
    M0, M1, M2 = mom(x); r0, _, real = to_polar(x); out = {}
    for k, e in E.items():
        m0, m1, m2 = mom(e); r, _, _ = to_polar(e)
        out[k] = dict(max_rel_dM0=float(np.max(np.abs(m0 / M0 - 1))), max_rel_dM2=float(np.max(np.abs(m2 / M2 - 1))),
                      med_dM1_over_M0_mrad=float(1e3 * np.median(np.linalg.norm(m1 - M1, axis=1) / M0)),
                      med_dr_mrad=float(1e3 * np.median(np.abs(r - r0)[real])),
                      padding_changed=int((np.asarray(e)[~real] != np.asarray(x, np.float64)[~real]).sum()),
                      max_dpt=float(np.max(np.abs(np.asarray(e)[..., 0] - np.asarray(x)[..., 0]))))
    return out


# ------------------------------------------------------------------ main
def main():
    import interp.e24_azimuth_prong as E24
    from interp.jedi_core import sample, Jedi
    from jetdistill.config import RESULTS
    R = dict(node='E24-verify', sample='interp.jedi_core.sample(per_class=2000, seed=2501)', models={})
    save = lambda: (OUT / 'e24_verify.json').write_text(json.dumps(clean(R), indent=1))
    x, y, Q = sample(2000, 2501); N = len(x); R['N'] = N
    d = np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r'); y_all = np.asarray(d['y'])
    im_old = E24.sample_idx(y_all, 2000, 2401); im = E24.sample_idx(y_all, 2000, 2501)
    R['overlap_with_E24_main'] = int(np.isin(im, im_old).sum()); log('overlap with E24 sample', R['overlap_with_E24_main'])
    j = Jedi(64); E24._JEDI['net'] = j
    K, b = j.KB['l1']; K = K.reshape(3, 64); b = b.reshape(-1); alpha = K / j.scale[:, None]; beta = b - (K * (j.shift / j.scale)[:, None]).sum(0)
    al = [alpha[:, c] for c in HP]; be = [beta[c] for c in HP]
    R['hp_w_pt'] = [float(alpha[0, c]) for c in HP]
    Ed = {'rot': rotate(x, 2510), 'rot2': rotate(x, 2511), 'A': scramble(x, 2512)}
    Ed["A'"] = moment_hold(Ed['A'], x)
    def cells(rs):
        parts = [cell_scramble(x[i:i + 1000], al, be, 2513 + i, None if rs is None else rs + i) for i in range(0, N, 1000)]
        return np.concatenate([p_[0] for p_ in parts]), np.concatenate([p_[1] for p_ in parts])
    Ed['Bx'], dth_b = cells(None); Ed['Brx'], dth_br = cells(30000)
    R['cell_dtheta_median_rad'] = dict(Bx=float(np.median(dth_b)), Brx=float(np.median(dth_br)),
                                       Bx_p90=float(np.quantile(dth_b, .9)), Brx_p90=float(np.quantile(dth_br, .9)))
    Ed = {k: v.astype(np.float32) for k, v in Ed.items()}
    R['edit_checks'] = checks(x, Ed); log('edit checks', json.dumps(clean(R['edit_checks']))); log('cell dθ', R['cell_dtheta_median_rad'])
    # substructure
    t0, d0 = E24.substructure(x)
    R['tau21_check_vs_Q'] = float(np.max(np.abs(t0 - Q['tau21']))) if 'tau21' in Q else None
    R['substructure'] = {}
    TT = {}
    for k in ('rot', "A'", 'Bx', 'Brx'):
        t, dd = E24.substructure(Ed[k]); TT[k] = t
        R['substructure'][k] = {CL[c]: dict(tau21=[float(np.median(t0[y == c])), float(np.median(t[y == c]))],
                                            D2=[float(np.median(d0[y == c])), float(np.median(dd[y == c]))],
                                            dtau21=ci(t[y == c] - t0[y == c])) for c in range(5)}
    log('substructure', json.dumps(clean({k: {c: v['tau21'] for c, v in s.items()} for k, s in R['substructure'].items()})))
    save()
    # half-plane on/off flips through the real network (l1 input quantizer included)
    real = x[..., 0] > 0; flips = {}
    a0 = np.concatenate([j.run(x[i:i + 1000], keep=('a',))['a'][..., HP] > 0 for i in range(0, N, 1000)])
    for k in ('rot', "A'", 'Bx', 'Brx'):
        a1 = np.concatenate([j.run(Ed[k][i:i + 1000], keep=('a',))['a'][..., HP] > 0 for i in range(0, N, 1000)])
        flips[k] = {f'a{c}': float(((a0[..., n_] != a1[..., n_]) & real).sum() / real.sum()) for n_, c in enumerate(HP)}
    del a0, a1; R['hp_flip_frac'] = flips; log('half-plane flips', json.dumps(clean(flips))); save()

    for name in ['jedi', 'dsets_s0', 'dsets_s1', 'pfn_s0', 'pfn_s1']:
        log('=====', name)
        run, acc_ref, kind, net = E24.get_model(name)
        base = run(x); c0 = base['logits'].argmax(1); M = dict(acc=float((c0 == y).mean()), acc_ref=acc_ref, edits={})
        R['models'][name] = M; outs = {k: run(e) for k, e in Ed.items()}
        ch = {k: o['logits'].argmax(1) != c0 for k, o in outs.items()}
        heavy = np.isin(c0, [2, 3, 4])
        Lc = np.linalg.norm(base['L'] - base['L'].mean(0), axis=1)
        for k, o in outs.items():
            c1 = o['logits'].argmax(1)
            M['edits'][k] = dict(calls_changed=ci(ch[k]), heavy_to_light=ci(np.isin(c1, [0, 1])[heavy]),
                                 latent_change_over_spread=ci_ratio(np.linalg.norm(o['L'] - base['L'], axis=1), Lc))
            if k not in ('rot',):
                M['edits'][k]['vs_rot'] = dict(diff=ci(ch[k].astype(float) - ch['rot']), mcnemar=mcnemar(ch[k], ch['rot']))
        M['Bx_vs_Brx'] = dict(diff=ci(ch['Bx'].astype(float) - ch['Brx']), mcnemar=mcnemar(ch['Bx'], ch['Brx']))
        M['Aprime_vs_A'] = dict(diff=ci(ch["A'"].astype(float) - ch['A']), mcnemar=mcnemar(ch["A'"], ch['A']))
        log(name, f'acc {M["acc"]:.4f}', ', '.join(f'{k} {v["calls_changed"]["mean"]:.4f} L {v["latent_change_over_spread"]["value"]:.3f}' for k, v in M['edits'].items()))
        # calibration
        if name in ('jedi', 'dsets_s0'):
            if 'cal' not in R:
                xp, yp, _ = sample(20000, 2502); ip = E24.sample_idx(y_all, 20000, 2502); keep = ~np.isin(ip, im); xp, yp = xp[keep], yp[keep]
                t0p, _ = E24.substructure(xp, with_d2=False); dt = TT["A'"] - t0
                sh, sa, _ = E24.calibration_pairs(x, y, xp, yp, dt, t0, t0p); ok = (sh >= 0) & (sa >= 0)
                R['cal'] = dict(n_both=int(ok.sum()), pool=int(len(xp)), dtau21_shift=ci(t0p[sh[ok]] - t0[ok]))
                CAL = (xp, sh, sa, ok); log('calibration pairs', R['cal']); save()
            xp, sh, sa, ok = CAL
            cs = run(xp[sh[ok]])['logits'].argmax(1); cz = run(xp[sa[ok]])['logits'].argmax(1); cj = c0[ok]
            att = (cs != cj).astype(float) - (cz != cj)
            M['calibration'] = dict(n=int(ok.sum()), tau21_attributable=ci(att), Aprime_same_jets=ci(ch["A'"][ok]),
                                    Aprime_minus_rot_same_jets=ci(ch["A'"][ok].astype(float) - ch['rot'][ok]))
            log(name, 'calibration', json.dumps(clean(M['calibration'])))
        if kind == 'jedi':
            M['internals'] = jedi_part(j, x, Ed, base, c0, ch)
        save()
    save(); log('wrote', OUT / 'e24_verify.json')


def jedi_part(j, x, Ed, base, c0, ch):
    keep = ('g', 'S', 'logits'); I0 = j.run(x, keep=keep); out = {}
    Ssd = I0['S'].std(0)
    live = [0, 1, 13, 17, 19, 20, 29, 30, 31, 40, 43, 50, 51, 52, 56, 58, 60, 62, 63]   # E24 live S list
    # S override shape: find the one that reproduces the base logits exactly
    m64 = np.ones(64)
    shp = None
    for cand in ((len(x), 1, 64), (len(x), 64)):
        try:
            r = j.run_jetwise(x, {'S': (m64, I0['S'].reshape(cand))}, keep=('logits',))['logits']
            if r.shape == I0['logits'].shape and np.max(np.abs(r - I0['logits'])) < 1e-4: shp = cand; break
        except Exception as ex:
            log('S override shape', cand, 'failed', repr(ex))
    out['S_override_shape'] = list(shp) if shp else None; log('S override shape', shp)
    rg = j.run_jetwise(x, {'g': (m64, I0['g'])}, keep=('logits',))['logits']
    out['g_identity_maxdiff'] = float(np.max(np.abs(rg - I0['logits'])))
    m58 = np.zeros(64); m58[[58, 31]] = 1
    for k in ('rot', 'rot2', 'A', "A'", 'Bx', 'Brx'):
        I1 = j.run(Ed[k], keep=keep); pe = {}
        pe['dS_over_sd'] = {f'S{c}': ci(np.abs(I1['S'][:, c] - I0['S'][:, c]) / Ssd[c]) for c in live}
        pe['_dS_raw'] = {c: np.abs(I1['S'][:, c] - I0['S'][:, c]) / Ssd[c] for c in live}
        gf = j.run_jetwise(Ed[k], {'g': (m64, I0['g'])}, keep=('S', 'logits'))
        pe['gfreeze_calls_changed'] = gf['logits'].argmax(1) != c0
        pe['gfreeze_S31_S58_maxdS'] = float(np.max(np.abs(gf['S'][:, [31, 58]] - I0['S'][:, [31, 58]])))
        if shp:
            fz = j.run_jetwise(Ed[k], {'S': (m58, I0['S'].reshape(shp))}, keep=('logits',))['logits']
            pe['S58S31frz_calls_changed'] = fz.argmax(1) != c0
            on = j.run_jetwise(x, {'S': (m58, I1['S'].reshape(shp))}, keep=('logits',))['logits']
            pe['only58_31_calls_changed'] = on.argmax(1) != c0
            if k == "A'":
                full = j.run_jetwise(x, {'S': (m64, I1['S'].reshape(shp))}, keep=('logits',))['logits']
                out['S_substitution_exact_maxdiff'] = float(np.max(np.abs(full - I1['logits'])))
        out[k] = pe
        log(f'jedi {k}: calls {ch[k].mean():.4f}; g frozen {pe["gfreeze_calls_changed"].mean():.4f} (S31/S58 max dS {pe["gfreeze_S31_S58_maxdS"]:.2e})'
            + (f'; S58,S31 frozen {pe["S58S31frz_calls_changed"].mean():.4f}; only S58,S31 edited {pe["only58_31_calls_changed"].mean():.4f}' if shp else '')
            + '; S58 ' + f'{pe["dS_over_sd"]["S58"]["mean"]:.3f} S31 {pe["dS_over_sd"]["S31"]["mean"]:.3f}')
    res = dict(identity=dict(S_override_shape=out['S_override_shape'], g_identity_maxdiff=out['g_identity_maxdiff'],
                             S_substitution_exact_maxdiff=out.get('S_substitution_exact_maxdiff')), per_edit={}, tests={})
    for k in ('rot', 'rot2', 'A', "A'", 'Bx', 'Brx'):
        pe = out[k]; r_ = dict(dS_over_sd=pe['dS_over_sd'], calls_changed=ci(ch[k]), gfreeze_calls_changed=ci(pe['gfreeze_calls_changed']),
                               gfreeze_S31_S58_maxdS=pe['gfreeze_S31_S58_maxdS'])
        if shp:
            r_['S58S31frz_calls_changed'] = ci(pe['S58S31frz_calls_changed']); r_['only58_31_calls_changed'] = ci(pe['only58_31_calls_changed'])
        res['per_edit'][k] = r_
    # excess over rot, with each intervention (paired, same jets)
    T = {}
    for lab, get in (('none', lambda k: ch[k]), ('gfreeze', lambda k: out[k]['gfreeze_calls_changed']),
                     ('S58S31frz', lambda k: out[k].get('S58S31frz_calls_changed')), ('only58_31', lambda k: out[k].get('only58_31_calls_changed'))):
        if get("A'") is None: continue
        T[lab] = dict(Aprime_minus_rot=ci(get("A'").astype(float) - get('rot')), mcnemar=mcnemar(get("A'"), get('rot')),
                      rot2_minus_rot=ci(get('rot2').astype(float) - get('rot')), mcnemar_rot2=mcnemar(get('rot2'), get('rot')))
    res['tests']['call_excess'] = T
    # per S unit: A' - rot, paired bootstrap p, Holm
    ps, units, diffs = [], [], {}
    for c in live:
        dlt = out["A'"]['_dS_raw'][c] - out['rot']['_dS_raw'][c]; b = dlt[bidx(len(dlt))].mean(1)
        p = float(min(1.0, ((np.abs(b - b.mean()) >= abs(dlt.mean())).sum() + 1) / (NB + 1))); ps.append(p); units.append(c); diffs[c] = ci(dlt)
    hp = holm(ps)
    res['tests']['S_Aprime_minus_rot'] = {f'S{c}': dict(diff=diffs[c], p=p, p_holm=float(h), radial=c in RADIAL_S) for c, p, h in zip(units, ps, hp)}
    return res


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True); main()
