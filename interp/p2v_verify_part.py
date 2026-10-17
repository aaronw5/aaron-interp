"""P2v — verification of P2 (inbox/P2_routes.md) with intervals and tests.

Samples: rows 6,000–7,000 of each class's first test file (1,000 per class; primary, fresh) and P2's rows 3,000–4,000
(reproduction). Edits as P2 (part_core.build): angles × {0.9, 1.1, 1.25}, pT × {0.8, 1.25}; extra pT × 0.9 (dose point for the
ParT_kin pT × 0.8 row); controls angles × 1.0, pT × 1.0, and 'rot0.06' (every particle φ + 0.06: pair features invariant in exact
arithmetic, so variant (b) of it is the float32 noise floor of the pair route, P1v's variant (d)).
Variants as P2: (a) (x_edit, v_edit), (b) pair route only (x_base, v_edit), (c) per-particle route only (x_edit, v_base).
Read-out: P2's ridge of log m on the unedited class token, refitted per sample (seed-1 split, train half); on the fresh sample
also the read-out fitted on rows 3,000–4,000 applied unchanged.
Statistics: 2,000 bootstrap resamples over jets, stratified by class (rng seed 2); every share (b)/(a), (c)/(a) and every
calls-changed fraction as estimate [2.5 %, 97.5 %]. Tests: H0 share_pair = 0 (mean_b = 0) and H0' share_pair = 1 (mean_b − mean_a = 0),
two-sided bootstrap percentile p ((k + 1)/(B + 1) floor) and normal p from the bootstrap s.e.; McNemar (exact binomial) of (b) vs
the unedited network on correctness, on W→Z (W jets), and of 'call changed' in (b) vs in the rot0.06 (b) noise control; Holm
over each family.
ParT_kin pT × 0.8 checks: pair-feature distributions (real–real pairs, i ≠ j) of edited vs unedited jets, the pair BatchNorm's
running statistics (training data), jet pT vs the JetClass range [500, 1000] GeV, W→H→bb by original jet pT, and a causal split of
(b): unedited 4-vectors with only ln kT += ln c, only ln m² += 2 ln c, or both (checked against (b)).
  ./part_job.sh p2v-run 'python -m interp.p2v_verify_part'          -> results/part/p2v_verify.json
  ./part_job.sh p2v-quick 'python -m interp.p2v_verify_part 50'     (50 jets per class, smoke)
  python -m interp.p2v_verify_part --table                          -> results/site/T12b_part_routes_stats.png
"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

log = lambda *a: print(*a, flush=True)
OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'
iHbb, iZ, iW = 1, 6, 7          # part_core.CL order: QCD, Hbb, Hcc, Hgg, H4q, Hqql, Zqq, Wqq, Tbqq, Tbl (checked in main)


def _imports():
    """the ParT stack (torch, weaver, uproot) only for the run, so that --table needs matplotlib alone"""
    from interp import part_core as pc, p1v_verify_part as p1v
    assert (pc.CL.index('Hbb'), pc.CL.index('Zqq'), pc.CL.index('Wqq')) == (iHbb, iZ, iW)
    globals().update(ParTNetwork=pc.ParTNetwork, inputs=pc.inputs, sample=pc.sample, kin=pc.kin, build=pc.build, jmass=pc.jmass,
                     Ridge=p1v.Ridge, r2=p1v.r2, split=p1v.split, fwd=p1v.fwd)
EDITS = [('ang', 0.9), ('ang', 1.1), ('ang', 1.25), ('pt', 0.8), ('pt', 1.25), ('pt', 0.9), ('ang', 1.0), ('pt', 1.0), ('rot', 0.06)]
MAIN = ['ang0.9', 'ang1.1', 'ang1.25', 'pt0.8', 'pt1.25']
VAR = ('a', 'b', 'c')
NB = 2000
PAIR_NAMES = ['lnkT', 'lnz', 'lnDeltaR', 'lnm2']
LNEPS = math.log(1e-8)
ename = lambda kind, f: f'{kind}{f:g}'


def make_edit(S, kind, f):
    J0, pt, de, dp, ms, real = S['J0'], S['pt'], S['de'], S['dp'], S['ms'], S['real']
    if kind == 'ang': return build(J0, pt, de * f, dp * f, ms)
    if kind == 'pt': return build(J0, pt * f, de, dp, ms * f)
    return build(J0, pt, de, dp + f * real, ms)


def prep(per, start):
    J0 = sample(per, start); pt, de, dp, ms = kin(J0); Jb = build(J0, pt, de, dp, ms); m0 = jmass(Jb)
    S = dict(J0=J0, y=J0['label'].astype(int), pt=pt, de=de, dp=dp, ms=ms, real=pt > 0, Jb=Jb, logm0=np.log(np.maximum(m0, 1.0)), start=start)
    S['JE'] = {ename(k, f): make_edit(S, k, f) for k, f in EDITS}
    S['dlogm_true'] = {k: float((np.log(np.maximum(jmass(Je), 1.0)) - S['logm0']).mean()) for k, Je in S['JE'].items()}
    S['split'] = split(len(S['y']))
    rng = np.random.default_rng(2); N = len(S['y']); Wc = np.zeros((NB, N), np.float32)
    for k in range(10):
        ix = np.flatnonzero(S['y'] == k)
        for b in range(NB): Wc[b] += np.bincount(ix[rng.integers(0, len(ix), len(ix))], minlength=N)
    S['Wc'] = Wc
    return S


# ------------------------------------------------------------------------------------------------ statistics
def bmean(S, v, sel):
    """point mean and bootstrap means of v over the jets sel (bool, length N)"""
    W = S['Wc'][:, sel]; x = v[sel].astype(np.float64)
    return float(x.mean()), (W @ x) / W.sum(1)


def ci(bs):
    bs = np.asarray(bs)[np.isfinite(bs)]
    if len(bs) == 0: return [None, None]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def pboot(bs, h=0.0):
    """two-sided bootstrap percentile p of H0: statistic = h; and normal p from the bootstrap s.e."""
    k = min((bs <= h).sum(), (bs >= h).sum()); pb = min(1.0, 2 * (k + 1) / (len(bs) + 1))
    se = float(bs.std()); z = (bs.mean() - h) / se if se > 0 else float('inf')
    return float(pb), float(math.erfc(abs(z) / math.sqrt(2)))


def binom_two_sided(n01, n10):
    n = n01 + n10
    if n == 0: return 1.0
    k = min(n01, n10)
    try:
        from scipy.stats import binom
        return float(min(1.0, 2 * binom.cdf(k, n, 0.5)))
    except Exception:
        s = sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2)) for i in range(k + 1))
        return float(min(1.0, 2 * s))


def mcnemar(u, v):
    """paired binary outcomes u (reference), v (test): n01 = u false & v true, n10 = u true & v false, exact p"""
    n01, n10 = int((~u & v).sum()), int((u & ~v).sum())
    return dict(n01=n01, n10=n10, p=binom_two_sided(n01, n10))


def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.zeros(m); run = 0.0
    for r, i in enumerate(o): run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj.tolist()


def quantities(S, L0, L, dl):
    """per-jet vectors and the jets they are averaged over"""
    y = S['y']; c0 = L0.argmax(1); c = L.argmax(1); te = np.zeros(len(y), bool); te[S['split'][3]] = True
    dlf = np.zeros(len(y)); dlf[S['split'][3]] = dl
    isW, isZ = y == iW, y == iZ; all_ = np.ones(len(y), bool)
    return dict(changed=((c != c0).astype(float), all_), dW_to_Z=((c == iZ).astype(float) - (c0 == iZ), isW),
                dZ_to_W=((c == iW).astype(float) - (c0 == iW), isZ), dW_to_Hbb=((c == iHbb).astype(float) - (c0 == iHbb), isW),
                dlogit_WZ_on_W=((L[:, iW] - L[:, iZ]) - (L0[:, iW] - L0[:, iZ]), isW), dlogm_hat=(dlf, te))


SHARE_Q = ('dlogit_WZ_on_W', 'dW_to_Z', 'dZ_to_W', 'dlogm_hat', 'dlogm_hat_P2readout')


def edit_stats(S, L0, R, dlP2=None):
    """R[var] = (L, dl, dlp): shares with intervals and tests, calls changed with intervals"""
    Q = {v: quantities(S, L0, R[v][0], R[v][1]) for v in VAR}
    if R['a'][2] is not None:
        for v in VAR:
            q = quantities(S, L0, R[v][0], R[v][2]); Q[v]['dlogm_hat_P2readout'] = q['dlogm_hat']
    out = dict(mean={}, ci={}, share={})
    M = {}
    for v in VAR:
        for k, (x, sel) in Q[v].items():
            p, bs = bmean(S, x, sel); M[(v, k)] = (p, bs)
            out['mean'].setdefault(v, {})[k] = p; out['ci'].setdefault(v, {})[k] = ci(bs)
    for k in SHARE_Q:
        if ('a', k) not in M: continue
        pa, ba = M[('a', k)]; pb, bb = M[('b', k)]; pc, bc = M[('c', k)]
        okb = np.abs(ba) > 1e-12
        sb, sc = bb[okb] / ba[okb], bc[okb] / ba[okb]
        p0 = pboot(bb, 0.0); p1 = pboot(bb - ba, 0.0); pc1 = pboot(bc - ba, 0.0); pc0 = pboot(bc, 0.0)
        out['share'][k] = dict(a=pa, b=pb, c=pc, pair=(pb / pa if abs(pa) > 1e-12 else None), pair_ci=ci(sb),
                               perparticle=(pc / pa if abs(pa) > 1e-12 else None), perparticle_ci=ci(sc),
                               sum=((pb + pc) / pa if abs(pa) > 1e-12 else None), sum_ci=ci(sb + sc),
                               a_ci=ci(ba), a_sign_frac_boot=float(np.mean(np.sign(ba) == np.sign(pa))),
                               p_pair_eq0_boot=p0[0], p_pair_eq0_norm=p0[1], p_pair_eq1_boot=p1[0], p_pair_eq1_norm=p1[1],
                               p_pp_eq0_boot=pc0[0], p_pp_eq1_boot=pc1[0], p_pp_eq1_norm=pc1[1])
    return out


# ------------------------------------------------------------------------------------------------ pair features
def pair_values(v, m, batch=48):
    """the 4 pair features (weaver's pairwise_lv_fts as PairEmbed calls it) on every real–real pair i ≠ j, float32 (n, 4)"""
    import torch
    from functools import partial
    from weaver.nn.model.ParticleTransformer import pairwise_lv_fts
    f = partial(pairwise_lv_fts, num_outputs=4, eps=1e-8, for_onnx=False)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; out = []
    order = np.argsort(m[:, 0].sum(1), kind='stable')
    with torch.no_grad():
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]; P = int(m[idx, 0].sum(1).max())
            t = torch.from_numpy(np.ascontiguousarray(v[idx, :, :P])).to(dev)
            mm = torch.from_numpy(np.ascontiguousarray(m[idx, 0, :P])).to(dev) > 0
            off = mm[:, :, None] & mm[:, None, :] & ~torch.eye(P, dtype=torch.bool, device=dev)[None]
            D = f(t.unsqueeze(-1), t.unsqueeze(-2))
            out.append(torch.stack([D[:, k][off] for k in range(4)], 1).cpu().numpy())
    return np.concatenate(out)


QS = [0.0, 0.001, 0.01, 0.5, 0.99, 0.999, 1.0]


def pair_range(S, n='kin'):
    """distributions of the pair features of the edited jets against those of the unedited jets"""
    xb, vb, mb = inputs(S['Jb'], n); base = pair_values(vb, mb)
    qb = np.quantile(base, QS, axis=0)
    res = dict(unedited=dict(pairs=len(base), quantiles={PAIR_NAMES[k]: qb[:, k].tolist() for k in range(4)},
                             mean={PAIR_NAMES[k]: float(base[:, k].mean()) for k in range(4)}, sd={PAIR_NAMES[k]: float(base[:, k].std()) for k in range(4)},
                             frac_at_clamp={PAIR_NAMES[k]: float((base[:, k] <= LNEPS + 1e-3).mean()) for k in range(4)}))
    for name in ('pt0.8', 'pt0.9', 'pt1.25', 'ang0.9', 'ang1.25'):
        _, ve, _ = inputs(S['JE'][name], n); e = pair_values(ve, mb)
        res[name] = dict(quantiles={PAIR_NAMES[k]: np.quantile(e[:, k], QS).tolist() for k in range(4)},
                         mean_shift={PAIR_NAMES[k]: float((e[:, k] - base[:, k]).mean()) for k in range(4)},
                         frac_below_unedited_min={PAIR_NAMES[k]: float((e[:, k] < qb[0, k]).mean()) for k in range(4)},
                         frac_below_unedited_q001={PAIR_NAMES[k]: float((e[:, k] < qb[1, k]).mean()) for k in range(4)},
                         frac_above_unedited_q999={PAIR_NAMES[k]: float((e[:, k] > qb[5, k]).mean()) for k in range(4)},
                         frac_above_unedited_max={PAIR_NAMES[k]: float((e[:, k] > qb[6, k]).mean()) for k in range(4)},
                         frac_at_clamp={PAIR_NAMES[k]: float((e[:, k] <= LNEPS + 1e-3).mean()) for k in range(4)})
        del e
    return res


def shifted_forward(net, x, v, m, shift):
    """forward with the pair features of v shifted by `shift` (4,) where they are above the eps clamp"""
    torch = net.torch; pe = net.model.pair_embed; orig = pe.pairwise_lv_fts
    s = torch.tensor(shift, dtype=torch.float32, device=net.device)

    def patched(xi, xj):
        o = orig(xi, xj); sv = s.view(1, 4, *([1] * (o.dim() - 2)))
        return torch.where(o > LNEPS + 1e-3, o + sv, o)
    pe.pairwise_lv_fts = patched
    try: return fwd(net, x, v, m)
    finally: pe.pairwise_lv_fts = orig


# ------------------------------------------------------------------------------------------------ main
def main():
    _imports(); t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    PER = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    SM = {start: prep(PER, start) for start in (3000, 6000)}
    log(f'samples ready ({time.time() - t0:.0f} s); true Δlog m: ' + '; '.join(f'{s}: ' + ', '.join(f'{k} {v:+.4f}' for k, v in SM[s]['dlogm_true'].items()) for s in SM))
    res = dict(N=len(SM[6000]['y']), per_class=PER, n_boot=NB, boot_seed=2, samples={}, edits=[ename(*e) for e in EDITS])
    for s, S in SM.items():
        Jb = S['Jb']; jp = Jb['jet_pt']; yW = S['y'] == iW
        res['samples'][str(s)] = dict(rows=[s, s + PER], dlogm_true=S['dlogm_true'], models={},
                                      jet_pt=dict(quantiles=np.quantile(jp, QS).tolist(), frac_x0p8_below500=float((0.8 * jp < 500).mean()),
                                                  frac_x0p9_below500=float((0.9 * jp < 500).mean()), frac_x1p25_above1000=float((1.25 * jp > 1000).mean())))
    res['pair_range'] = pair_range(SM[6000]); log(f'pair ranges done ({time.time() - t0:.0f} s): {json.dumps(res["pair_range"])[:3000]}')

    for n in ('full', 'kin'):
        net = ParTNetwork(n); ridgeP2 = None
        try: res.setdefault('pair_batchnorm', {})[n] = dict(running_mean=net.model.pair_embed.embed[0].running_mean.cpu().tolist(),
                                                            running_sd=net.model.pair_embed.embed[0].running_var.sqrt().cpu().tolist())
        except Exception as ex: log('batchnorm stats:', ex)
        for s in (3000, 6000):
            S = SM[s]; t1 = time.time(); y = S['y']; trf, trv, tr, te = S['split']
            xb, vb, mb = inputs(S['Jb'], n); z0, L0 = fwd(net, xb, vb, mb); c0 = L0.argmax(1)
            ridge = Ridge(z0, S['logm0'], trf, trv, tr); rz0 = ridge(z0[te])
            if s == 3000: ridgeP2 = ridge
            rP = ridgeP2 if s == 6000 else None; rzP = rP(z0[te]) if rP is not None else None
            base = dict(acc=float((c0 == y).mean()), W_to_Z=float((c0[y == iW] == iZ).mean()), Z_to_W=float((c0[y == iZ] == iW).mean()),
                        W_to_Hbb=float((c0[y == iW] == iHbb).mean()), ridge_r2=r2(rz0, S['logm0'][te]),
                        ridgeP2_r2=(r2(rzP, S['logm0'][te]) if rP is not None else None))
            row = dict(baseline=base, edits={}, mcnemar={})
            log(f'ParT_{n} rows {s}: baseline {base}')
            calls = {}
            for name, Je in S['JE'].items():
                xe, ve, me = inputs(Je, n); assert np.array_equal(me, mb)
                R = {}
                for v_, (x, v) in zip(VAR, ((xe, ve), (xb, ve), (xe, vb))):
                    z, L = fwd(net, x, v, mb); R[v_] = (L, ridge(z[te]) - rz0, (rP(z[te]) - rzP) if rP is not None else None)
                    calls[(name, v_)] = L.argmax(1)
                E = edit_stats(S, L0, R)
                cfw = lambda v_: [float((calls[(name, v_)][y == iW] == k).mean()) for k in range(10)]
                E['calls_from_W'] = {v_: cfw(v_) for v_ in VAR}
                E['own_class'] = {v_: [float((calls[(name, v_)][y == k] == k).mean()) for k in range(10)] for v_ in VAR}
                E['max_abs_dlogit'] = {v_: float(np.abs(R[v_][0] - L0).max()) for v_ in VAR}
                if name.startswith('pt') and name != 'pt1':
                    f = float(name[2:]); orig = S['Jb']['jet_pt']
                    lo = (f * orig < 500) if f < 1 else (f * orig > 1000)
                    E['by_jet_pt'] = {}
                    for g in (False, True):
                        sel = (y == iW) & (lo == g)
                        E['by_jet_pt']['out_of_range' if g else 'in_range'] = dict(
                            n_W=int(sel.sum()), W_to_Hbb={v_: float((calls[(name, v_)][sel] == iHbb).mean()) for v_ in VAR},
                            W_to_Z={v_: float((calls[(name, v_)][sel] == iZ).mean()) for v_ in VAR},
                            W_to_W={v_: float((calls[(name, v_)][sel] == iW).mean()) for v_ in VAR},
                            base_W_to_Hbb=float((c0[sel] == iHbb).mean()), base_W_to_W=float((c0[sel] == iW).mean()))
                row['edits'][name] = E
                sh = E['share']
                log(f'   {name:7s} changed % a/b/c ' + ' / '.join(f'{100 * E["mean"][v]["changed"]:.2f} [{100 * E["ci"][v]["changed"][0]:.2f}, {100 * E["ci"][v]["changed"][1]:.2f}]' for v in VAR)
                    + ' | pair share: ' + ', '.join(f'{k} {sh[k]["pair"]} {sh[k]["pair_ci"]} p0 {sh[k]["p_pair_eq0_boot"]:.4f}/{sh[k]["p_pair_eq0_norm"]:.1e} p1 {sh[k]["p_pair_eq1_boot"]:.4f}/{sh[k]["p_pair_eq1_norm"]:.1e}' for k in sh))
                del xe, ve, R
            # McNemar: (b) vs unedited; and 'changed' in (b) vs in the rot0.06 (b) noise control
            ok0 = c0 == y; isW = y == iW; ch_noise = calls[('rot0.06', 'b')] != c0
            for name in MAIN + ['pt0.9', 'ang1', 'pt1', 'rot0.06']:
                cb = calls[(name, 'b')]
                row['mcnemar'][name] = dict(correct=mcnemar(ok0, cb == y), W_to_Z=mcnemar(c0[isW] == iZ, cb[isW] == iZ),
                                            changed_vs_noise=mcnemar(ch_noise, cb != c0), noise_changed=float(ch_noise.mean()))
            fam = [(nm, t) for nm in MAIN for t in ('correct', 'W_to_Z', 'changed_vs_noise')]
            adj = holm([row['mcnemar'][nm][t]['p'] for nm, t in fam])
            for (nm, t), a in zip(fam, adj): row['mcnemar'][nm][t]['p_holm'] = a
            row['mcnemar_family'] = dict(n_tests=len(fam), tests=[f'{nm}:{t}' for nm, t in fam])
            fam2 = [(nm, k, h) for nm in MAIN for k in ('dlogit_WZ_on_W', 'dW_to_Z', 'dlogm_hat') for h in ('p_pair_eq0_boot', 'p_pair_eq1_boot')]
            adj2 = holm([row['edits'][nm]['share'][k][h] for nm, k, h in fam2])
            for (nm, k, h), a in zip(fam2, adj2): row['edits'][nm]['share'][k][h + '_holm'] = a
            row['share_test_family'] = dict(n_tests=len(fam2))
            log(f'   McNemar: ' + '; '.join(f'{nm} ' + ', '.join(f'{t} {row["mcnemar"][nm][t]["n01"]}/{row["mcnemar"][nm][t]["n10"]} p {row["mcnemar"][nm][t]["p"]:.2e}' for t in ('correct', 'W_to_Z', 'changed_vs_noise')) for nm in row['mcnemar']))
            # causal split of the pair route at pT × c (fresh sample only): unedited 4-vectors with ln kT / ln m² shifted
            if s == 6000:
                row['pt_shift_split'] = {}
                for f in (0.8, 0.9, 1.25):
                    lc = math.log(f); name = ename('pt', f); cb = calls[(name, 'b')]; Lb_ref = None
                    _, ve, _ = inputs(S['JE'][name], n); _, Lb_ref = fwd(net, xb, ve, mb); del ve
                    out = {}
                    for lab, sh in (('both', [lc, 0, 0, 2 * lc]), ('lnkT_only', [lc, 0, 0, 0]), ('lnm2_only', [0, 0, 0, 2 * lc])):
                        _, L = shifted_forward(net, xb, vb, mb, sh); c = L.argmax(1)
                        out[lab] = dict(changed=float((c != c0).mean()), W_to_Hbb=float((c[y == iW] == iHbb).mean()), W_to_Z=float((c[y == iW] == iZ).mean()),
                                        W_to_W=float((c[y == iW] == iW).mean()), own_class=[float((c[y == k] == k).mean()) for k in range(10)],
                                        dlogit_WZ_on_W=float(((L[:, iW] - L[:, iZ]) - (L0[:, iW] - L0[:, iZ]))[y == iW].mean()),
                                        agree_with_b=float((c == cb).mean()), max_abs_dlogit_vs_b=float(np.abs(L - Lb_ref).max()))
                    row['pt_shift_split'][name] = out
                    log(f'   pT × {f} shift split: ' + '; '.join(f'{k}: changed {100 * o["changed"]:.2f} %, W→Hbb {o["W_to_Hbb"]:.3f}, W→Z {o["W_to_Z"]:.3f}, Δ(lW−lZ) W {o["dlogit_WZ_on_W"]:+.3f}, agree with (b) {o["agree_with_b"]:.4f}' for k, o in out.items()))
            res['samples'][str(s)]['models'][n] = row
            log(f'ParT_{n} rows {s} done ({time.time() - t1:.0f} s)')
        del net
    fn = OUT / ('p2v_verify.json' if PER == 1000 else f'p2v_verify_{PER}.json')
    json.dump(res, open(fn, 'w'), indent=1); log(f'wrote {fn}'); log(f'P2v DONE ({time.time() - t0:.0f} s)')




# ------------------------------------------------------------------------------------------------ table (site/tables.py style)
def table():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    res = json.load(open(OUT / 'p2v_verify.json')); SITE = OUT.parent / 'site'
    lab = {'ang0.9': 'angles × 0.9', 'ang1.1': 'angles × 1.1', 'ang1.25': 'angles × 1.25', 'pt0.8': 'pT × 0.8', 'pt0.9': 'pT × 0.9', 'pt1.25': 'pT × 1.25'}
    F = res['samples']['6000']; P = res['samples']['3000']
    z = lambda x: f'{x:.2f}'.replace('-0.00', '0.00')
    sci = lambda s, key: f'{z(s[key])} [{z(s[key + "_ci"][0])}, {z(s[key + "_ci"][1])}]'
    pp = lambda p: '≤ 0.001' if p <= 0.0011 else f'{p:.3f}'
    rows = []
    for n in ('full', 'kin'):
        for e in lab:
            E = F['models'][n]['edits'][e]; S = E['share']; Mc = F['models'][n]['mcnemar'][e]['W_to_Z']; L = S['dlogit_WZ_on_W']
            ch = f'{100 * E["mean"]["b"]["changed"]:.1f} [{100 * E["ci"]["b"]["changed"][0]:.1f}, {100 * E["ci"]["b"]["changed"][1]:.1f}]'
            nec = f'{z(1 - L["perparticle"])} [{z(1 - L["perparticle_ci"][1])}, {z(1 - L["perparticle_ci"][0])}]'
            rows.append([f'ParT_{n}', f'{lab[e]}  ({F["dlogm_true"][e]:+.3f})', f'{L["a"]:+.2f}', sci(L, 'pair'), nec, f'{L["sum"]:.2f}',
                         f'{pp(L["p_pair_eq0_boot"])} / {pp(L["p_pair_eq1_boot"])}', sci(S['dW_to_Z'], 'pair'),
                         f'{sci(S["dlogm_hat"], "pair")}', sci(S['dlogm_hat'], 'perparticle'), ch,
                         f'{Mc["n01"]} / {Mc["n10"]}, p {Mc["p"]:.0e}', f'{P["models"][n]["edits"][e]["share"]["dlogit_WZ_on_W"]["pair"]:.2f}'])
    cols = ['model', 'edit  (true Δlog m)', 'Δ(logit W − Z)\non W, (a)', 'pair share (b)/(a)\nof Δ(logit W − Z), W jets', '1 − (c)/(a): drop when\npair route removed',
            '[(b)+(c)]/(a)', 'bootstrap p, H0: pair\nshare = 0 / share = 1', 'pair share (b)/(a)\nof ΔW→Z', 'read-out Δlog m:\npair share (b)/(a)',
            'read-out Δlog m:\nper-particle (c)/(a)', '(b) calls\nchanged [%]', 'McNemar (b) vs unedited,\nW→Z: gained / lost, p', 'P2 rows 3000–4000:\npair share Δlogit']
    widths = [0.85, 1.75, 1.2, 1.8, 1.8, 0.95, 1.8, 1.6, 1.65, 1.65, 1.2, 2.0, 1.65]
    nf = F['models']['full']['edits']['rot0.06']['mean']['b']['changed']; nk = F['models']['kin']['edits']['rot0.06']['mean']['b']['changed']
    nkc = F['models']['kin']['edits']['rot0.06']['ci']['b']['changed']
    kb = F['models']['kin']['edits']['pt0.8']; sp = F['models']['kin']['pt_shift_split']['pt0.8']; bj = kb['by_jet_pt']
    note = ('Sample: rows 6,000–7,000 of each class\'s first JetClass test file (10,000 jets; fresh); last column: P2\'s rows 3,000–4,000 through the same code. Intervals: 2,000 bootstrap '
            'resamples over jets, stratified by class, [2.5 %, 97.5 %].\n(a) full edit; (b) pair route only (4-vectors edited, every per-particle input unedited); (c) per-particle route only. '
            'Logit columns are the primary measure; read-out columns use P2\'s ridge of log m refitted on the unedited class token (secondary).\n'
            f'Noise floor of the pair route ((b) of a rigid φ + 0.06 shift, pair features invariant): ParT_full {100 * nf:.2f} %, ParT_kin {100 * nk:.2f} % [{100 * nkc[0]:.2f}, {100 * nkc[1]:.2f}] of calls changed; '
            'factor-1.0 controls: 0.00 % (both). McNemar p values unadjusted; all W→Z tests survive Holm over 15 tests per model.\n'
            f'ParT_kin pT × 0.8, (b): W→H→bb {kb["calls_from_W"]["b"][1]:.2f} of W jets ({bj["in_range"]["W_to_Hbb"]["b"]:.2f} for jets whose scaled pT stays in [500, 1000] GeV, '
            f'{bj["out_of_range"]["W_to_Hbb"]["b"]:.2f} below 500 GeV); unedited 4-vectors with ln m² − 0.446 only: {sp["lnm2_only"]["W_to_Hbb"]:.2f}, with ln kT − 0.223 only: {sp["lnkT_only"]["W_to_Hbb"]:.3f}. '
            'No edited pair feature lies outside the unedited range.')
    SITE.mkdir(parents=True, exist_ok=True)
    Wd = sum(widths); Ht, Hn = 0.34 * (len(rows) + 1.8), 1.05; H = Ht + 0.45 + Hn
    fig = plt.figure(figsize=(Wd, H), dpi=160); ax = fig.add_axes([0.003, Hn / H, 0.994, Ht / H]); ax.axis('off')
    t = ax.table(cellText=rows, colLabels=cols, colWidths=[w / Wd for w in widths], bbox=[0, 0, 1, 1], cellLoc='left', colLoc='left')
    t.auto_set_font_size(False); t.set_fontsize(8.5)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor('#d8d4ca'); cell.PAD = 0.03
        if r == 0: cell.set_facecolor('#eeebe3'); cell.set_text_props(weight='bold'); cell.set_height(cell.get_height() * 1.8)
        elif r % 2 == 0: cell.set_facecolor('#f8f7f3')
    fig.text(0.003, 1 - 0.12 / H, 'ParT: input routes of the response to a real mass change, with 95 % intervals and tests (P2v)', va='top', fontsize=11, weight='bold')
    fig.text(0.003, (Hn - 0.08) / H, note, va='top', fontsize=7.8, color='#555', linespacing=1.45)
    fig.savefig(SITE / 'T12b_part_routes_stats.png', bbox_inches='tight', pad_inches=0.1); log(f'wrote {SITE / "T12b_part_routes_stats.png"}')


if __name__ == '__main__':
    if sys.argv[1:] == ['--table']: table()
    else: main()
