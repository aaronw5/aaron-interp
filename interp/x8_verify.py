"""X8 verification (verifier node): rerun of the JEDI-linear part of X8 on a fresh sample, plus two controls.

Sample: interp.jedi_core.sample(per_class=2000, seed=7307) = 10,000 test jets (X8 used seed 307).
Reuses interp.x8_nulls (edits, measure(), permutations, ablation()) unchanged; only the sample seed, the resample count
(NRES = 10 instead of 20) and the null set differ.
Models: JEDI-linear trained; variant a s0..4 (non-zero kernel entries permuted, masks in place); variant c s0..4 (as a, plus
  slot tables permuted per channel); variant b s0 (all entries + quantizer tables permuted); and an added null:
  untrained redraw s0..2 (jetdistill.network.redraw_untrained: every kernel / bias / BN gamma redrawn from a Gaussian with the
  trained mean and s.d.; trained number formats, incl. zero-bit masks, kept) = a null that keeps per-layer scale.
Added controls:
  (1) H3 unit control: per-unit across (13|14) / inside (5|6) |dS|/sd for every live S unit, split by whether the unit's own
      qs1 mask changes between slots 13 and 14 (edge) or not (no edge); same split on every non-degenerate null.
  (2) H4 normalisation: response / (|rho| * E_Q), E_Q = mean(0.21 Q) / sd(Q), Q = sum pT r^2 (the response of a unit that
      is a linear function of Q, scaled by its correlation).
Statistics: paired jet bootstrap (1,000; same indices for every model); per-null one-sided p; p_IUT = max over nulls, over
  all nulls and over non-degenerate nulls (max call share < 0.99); Holm over the 4 headline tests.
Run: ./job.sh x8-verify 'python -m interp.x8_verify'  -> $AI_RESULTS/jedi/x8_verify.json
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import interp.x8_nulls as X  # noqa: E402  (import parses ['--part', 'combine']: no side effects)

X.SEED = 7307; X.PER = 2000; X.NRES = 10; X.NB = 1000; X.SFX = '_verify_unused'
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
T0 = time.time()
KEEP = ('flow_T', 'flow_Tr', 'ch_T', 'ch_W', 'ch_Wr', 'corr0', 'corr_W', 'win_a_A13_I5', 'win_i_A13_I5', 'rad_K', 'rad_Kr')


def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def unit_control(m, x, SW, liveS, sd, k=13, j=5):
    okA, xa = SW[k]; okI, xi = SW[j]; com = okA & okI
    R0 = m.run(x[com], keep=('S',))['S']; Sa = m.run(xa[com], keep=('S',))['S']; Si = m.run(xi[com], keep=('S',))['S']
    qs1 = m.qs1; out = {}
    for name, sel in (('edge', lambda c: qs1[k, c] != qs1[k + 1, c]), ('no_edge', lambda c: qs1[k, c] == qs1[k + 1, c])):
        us = [c for c in liveS if sel(c) and sd[c] > 0]
        if not us: out[name] = dict(units=[], across=None, inside=None, ratio=None); continue
        a = (np.abs(Sa[:, us] - R0[:, us]) / sd[us]).mean(1); i = (np.abs(Si[:, us] - R0[:, us]) / sd[us]).mean(1)
        out[name] = dict(units=us, across=float(a.mean()), inside=float(i.mean()),
                         ratio=float(a.mean() / i.mean()) if i.mean() > 0 else ('inf' if a.mean() > 0 else None),
                         inside_edge_at_5_6=[int(c) for c in us if qs1[j, c] != qs1[j + 1, c]],
                         per_unit={int(c): [float((np.abs(Sa[:, c] - R0[:, c]) / sd[c]).mean()),
                                            float((np.abs(Si[:, c] - R0[:, c]) / sd[c]).mean())] for c in us})
    return out


def main():
    import jax
    from interp.jedi_core import Jedi
    OUT.mkdir(parents=True, exist_ok=True)
    x, y, sp = X.load(); E, SW, P, Q, info = X.make_edits(x); J = len(x)
    EQ = float((0.21 * Q).mean() / Q.std())
    R = dict(sample=dict(per_class=X.PER, seed=X.SEED, N=J, N_W=int((y == X.W_).sum())), edits=info, E_Q=EQ, models={},
             note='verification of X8 (seed 307) on seed 7307; JEDI-linear part only; NRES 10')
    log(f'sample {J} jets seed {X.SEED}; E_Q {EQ:.4f}')
    net = Jedi(64); base = X.JediModel(net)
    R['forward_check_maxabs'] = float(np.abs(net.run(x[:2000])['logits'] - base.run(x[:2000])['logits']).max())
    arrays = {}

    def one(name, mdl, label):
        S_, A_ = X.measure(mdl, x, y, sp, E, SW, P, Q, label)
        R0 = mdl.run(x, keep=('S',))['S']; sd = R0.std(0)
        S_['unit_control_A13_I5'] = unit_control(mdl, x, SW, S_['liveS'], sd)
        S_['degenerate'] = bool(max(S_['call_frac']) >= 0.99)
        rho = abs(S_['radial']['rho'])
        S_['radial']['response_over_rho_EQ'] = float(S_['radial']['response_K'] / (rho * EQ)) if rho > 0 else None
        R['models'][name] = S_; arrays[name] = {k: np.asarray(A_[k], float) for k in KEEP}
        uc = S_['unit_control_A13_I5']
        log(f'  {name}: degenerate {S_["degenerate"]}; unit control edge {uc["edge"].get("ratio")} {uc["edge"].get("units")} | '
            f'no edge {uc["no_edge"].get("ratio")} {uc["no_edge"].get("units")}; radial resp/(rho EQ) '
            f'{S_["radial"]["response_over_rho_EQ"]}')
        (OUT / 'x8_verify.json').write_text(json.dumps(X.clean(R), indent=1))

    one('jedi_trained', base, 'JEDI-linear trained')
    for variant, seeds in (('a', range(5)), ('c', range(5)), ('b', range(1))):
        for s in seeds:
            jax.clear_caches(); orig = []
            if variant == 'b': orig, _ = X.permute_iq(net.L, s)
            if variant == 'c': orig = X.permute_slots(net.L, s)
            mdl = X.JediModel(net, X.permute_KB(net.KB, s, 'a' if variant == 'c' else variant))
            one(f'jedi_perm{variant}{s}', mdl, f'JEDI-linear permuted {variant} s{s}')
            X.restore_iq(orig); del mdl
    R['restore_check_maxabs'] = float(np.abs(X.JediModel(net).run(x[:2000])['logits'] - net.run(x[:2000])['logits']).max())
    for s in range(3):
        jax.clear_caches(); nu = Jedi(64, untrained=True, seed=s)
        one(f'jedi_untrained{s}', X.JediModel(nu), f'JEDI-linear untrained redraw s{s}'); del nu

    # ---------------------------------------------------------------- statistics
    nW = int((y == X.W_).sum()); ncom = len(arrays['jedi_trained']['win_a_A13_I5'])
    BI = dict(W=X.boot_idx(nW, 1), all=X.boot_idx(J, 0), com=X.boot_idx(ncom, 3))
    F = {'H1_translation_WZ_flow': ('W', lambda A, i: A['flow_T'][i].mean()),
         'H2_wta_calls_changed': ('all', lambda A, i: A['ch_W'][i].mean()),
         'H3_window_ratio_A13_I5': ('com', lambda A, i: X.win_ratio(A['win_a_A13_I5'][i].mean(), A['win_i_A13_I5'][i].mean())),
         'H4_radial_response': ('all', lambda A, i: A['rad_K'][i].mean()),
         'wta_acc_drop': ('all', lambda A, i: A['corr0'][i].mean() - A['corr_W'][i].mean()),
         'translation_calls_changed': ('all', lambda A, i: A['ch_T'][i].mean()),
         'radial_rotation_control': ('all', lambda A, i: A['rad_Kr'][i].mean())}
    full = dict(W=np.arange(nW), all=np.arange(J), com=np.arange(ncom))
    nulls = [n for n in R['models'] if n != 'jedi_trained']
    nondeg = [n for n in nulls if not R['models'][n]['degenerate']]
    R['nulls'] = nulls; R['nulls_nondegenerate'] = nondeg; R['tests'] = {}
    for st, (sub, fn) in F.items():
        pt = {n: float(fn(arrays[n], full[sub])) for n in R['models']}
        bt = {n: np.array([fn(arrays[n], b) for b in BI[sub]], float) for n in R['models']}
        ps = {}
        for n in nulls:
            d = bt['jedi_trained'] - bt[n]; ok = ~np.isnan(d)
            ps[n] = float((1 + int((d[ok] <= 0).sum())) / (1 + int(ok.sum()))) if ok.any() and np.isfinite(pt[n]) else None

        def iut(ns):
            v = [ps[n] for n in ns if ps[n] is not None]; return max(v) if v else None

        def rng_(ns):
            v = [pt[n] for n in ns if np.isfinite(pt[n])]; return [min(v), max(v)] if v else None
        R['tests'][st] = dict(trained=pt['jedi_trained'], trained_ci=X.ci(bt['jedi_trained']), null_points=pt, p_per_null=ps,
                              p_iut_all=iut(nulls), p_iut_nondegenerate=iut(nondeg), null_range_all=rng_(nulls),
                              null_range_nondegenerate=rng_(nondeg),
                              null_range_untrained=rng_([n for n in nulls if 'untrained' in n]),
                              p_iut_untrained=iut([n for n in nulls if 'untrained' in n]),
                              n_inf=[n for n in nulls if np.isinf(pt[n])])
        log(f'{st}: trained {pt["jedi_trained"]:.4f} {R["tests"][st]["trained_ci"]}; nulls all {R["tests"][st]["null_range_all"]} '
            f'nondeg {R["tests"][st]["null_range_nondegenerate"]} untrained {R["tests"][st]["null_range_untrained"]}; '
            f'p_IUT all {R["tests"][st]["p_iut_all"]} nondeg {R["tests"][st]["p_iut_nondegenerate"]}')
    heads = [k for k in F if k.startswith('H')]
    hp = X.holm([R['tests'][h]['p_iut_nondegenerate'] for h in heads])
    R['holm_nondegenerate'] = {h: dict(p=R['tests'][h]['p_iut_nondegenerate'], p_holm=p) for h, p in zip(heads, hp)}
    log('Holm (non-degenerate nulls): ' + str(R['holm_nondegenerate']))
    (OUT / 'x8_verify.json').write_text(json.dumps(X.clean(R), indent=1))

    # ---------------------------------------------------------------- ablation arm (trained), NRES 10
    jax.clear_caches()
    R['ablation'] = X.ablation(X.JediModel(net), x, y, [('P', 64), ('S', 64), ('z', 16)], ctx='g')
    R['seconds'] = time.time() - T0
    (OUT / 'x8_verify.json').write_text(json.dumps(X.clean(R), indent=1)); log('done x8_verify')


if __name__ == '__main__':
    main()
