"""X9x verification, part 3 (direction control for 'beyond A the network responds as to a real increase in mass spread').

X9x calibrates the translation response by Rc = (calls changed by translation - by matched rotation) / calls changed by the
matched M2 dilation dilM2.  Rc compares counts only; it does not say whether translation moves the calls the same way as an
increase of M2.  This part tests the direction.
Sample: interp.jedi_core.sample(per_class=1600, seed=4901) (N 8,000), seed not used by X9x (3801, 3802) or X9x_verify (4801).
Networks: the 30 trained X9x networks (PFN, EFN, Deep Sets; A in {0, 0.04, 0.08, 0.16, 0.30}; seeds 0, 1), interp.x9x_augrange.load_net.
Offsets per network: inside T = A / 2 (A > 0); beyond T = A + 0.04 and A + 0.08 (A = 0: 0.04, 0.08).
Edits (all real particles only):
  trans   rigid translation by T at 4 azimuths (0, 90, 180, 270 deg), per-jet quantities averaged over the azimuths
  dil     angles x s, s = sqrt(1 + T^2 M0 / M2) (X9x dilM2: same azimuth-mean Delta log M2 as the translation)
  con     angles x 1 / s (matched M2 decrease, same |Delta log M2|; direction control)
  rot     X6 rotate_matched with mean displacement T (rng 4910 + round(1000 T)) (null used by X9x)
Metrics (per network and T), mass rank g = q (0) < W (1) < Z (2) < t (3):
  changed  fraction of jets whose call differs from the unedited call
  hl       fraction moved to a heavier / lighter rank
  drank    mean rank change over all jets; Rrank = drank(trans) / drank(dil) (signed calibrated response; 1 = translation moves
           calls up in mass as far as the matched M2 increase; 0 = direction-neutral; < 0 = like a decrease)
  agree    among jets changed by both trans (azimuth k) and dil: fraction whose new call is the same; chance level = the same
           fraction expected if the new calls were independent given the unedited call (sum over c0 strata of sum_c pT(c) pD(c));
           agree_con = the same against con.
  overlap  P(trans changes | dil changes) - P(trans changes | dil does not change)
Statistics: 1,000-draw jet bootstrap (rng 4942) for drank(trans), Rrank, agree - chance; H0 (direction-neutral generic
  perturbation): drank(trans) = 0 and agree = chance; H1 (acts as an M2 increase): drank(trans) > 0 with Rrank near 1 and
  agree > chance > agree_con.  p (two-sided bootstrap) for drank(trans) = 0; Holm over the 60 beyond-A (network, T) cells.
  ./job.sh x9x-verify3 1 2 'python -m interp.x9x_verify3'  -> $AI_RESULTS/jedi/x9x_verify3.json
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp.x9x_augrange import load_net, clean, holm, AUGS, SEEDS  # noqa: E402
from interp.x9_efn_pfn import translate, rotate_matched, scale_angles, moments  # noqa: E402

RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results')); OUT = RES / 'jedi'
SEED, PER = 4901, 1600; AZ = np.deg2rad([0.0, 90.0, 180.0, 270.0]); NB = 1000; T0 = time.time()
RANK = np.array([0, 0, 1, 2, 3])


def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def offsets(A): return [('beyond', 0.04), ('beyond', 0.08)] if A == 0 else [('inside', A / 2), ('beyond', A + 0.04), ('beyond', A + 0.08)]


def chance_agree(c0, ct, cd, m):
    """expected agreement of new calls among jets m if ct and cd were independent given c0"""
    tot, n = 0.0, int(m.sum())
    if n == 0: return np.nan
    for k in np.unique(c0[m]):
        s = m & (c0 == k); w = s.sum() / n
        pt = np.bincount(ct[s], minlength=5) / s.sum(); pd = np.bincount(cd[s], minlength=5) / s.sum(); tot += w * (pt * pd).sum()
    return tot


def cell(net, x, M, T, Ib):
    c0 = net.logits(x).argmax(1); s = np.sqrt(1 + T ** 2 * M['M0'] / np.maximum(M['M2'], 1e-30))
    cd = net.logits(scale_angles(x, s)).argmax(1); cc = net.logits(scale_angles(x, 1 / s)).argmax(1)
    cr = net.logits(rotate_matched(x, T, np.random.default_rng(4910 + int(round(1000 * T))))).argmax(1)
    cts = [net.logits(translate(x, T, th)).argmax(1) for th in AZ]
    r0 = RANK[c0]
    dr_t = np.mean([RANK[c] - r0 for c in cts], 0); dr_d = (RANK[cd] - r0).astype(float); dr_c = (RANK[cc] - r0).astype(float)
    ch_t = np.mean([c != c0 for c in cts], 0); ch_d = (cd != c0).astype(float); ch_c = (cc != c0).astype(float); ch_r = (cr != c0).astype(float)
    hl = lambda cs: [float(np.mean([(RANK[c] > r0).mean() for c in cs])), float(np.mean([(RANK[c] < r0).mean() for c in cs]))]
    ag, ch, agc, chc, ov = [], [], [], [], []
    for c in cts:
        m = (c != c0) & (cd != c0); ag.append((c[m] == cd[m]).mean() if m.any() else np.nan); ch.append(chance_agree(c0, c, cd, m))
        mc = (c != c0) & (cc != c0); agc.append((c[mc] == cc[mc]).mean() if mc.any() else np.nan); chc.append(chance_agree(c0, c, cc, mc))
        t = c != c0; dm = cd != c0
        ov.append(t[dm].mean() - t[~dm].mean() if dm.any() and (~dm).any() else np.nan)
    # bootstrap over jets (azimuth-averaged per-jet quantities; agreement recomputed per draw on azimuth 0 only for speed)
    bt, bR = dr_t[Ib].mean(1), dr_t[Ib].mean(1) / np.where(np.abs(dr_d[Ib].mean(1)) > 1e-12, dr_d[Ib].mean(1), np.nan)
    c = cts[0]; m = (c != c0) & (cd != c0); ba = []
    for I in Ib[:300]:
        mm = m[I]
        if mm.sum() < 5: continue
        ba.append((c[I][mm] == cd[I][mm]).mean() - chance_agree(c0[I], c[I], cd[I], mm))
    ba = np.asarray(ba); pt = float(min(1.0, 2 * min((bt <= 0).mean(), (bt >= 0).mean())))
    q = lambda v: [float(np.nanquantile(v, .025)), float(np.nanquantile(v, .975))]
    return dict(T=T, changed=dict(trans=float(ch_t.mean()), dil=float(ch_d.mean()), con=float(ch_c.mean()), rot=float(ch_r.mean())),
                Rc=float((ch_t - ch_r).mean() / ch_d.mean()) if ch_d.mean() > 0 else None,
                heavier_lighter=dict(trans=hl(cts), dil=hl([cd]), con=hl([cc]), rot=hl([cr])),
                drank=dict(trans=float(dr_t.mean()), trans_ci=q(bt), dil=float(dr_d.mean()), con=float(dr_c.mean()),
                           rot=float((RANK[cr] - r0).mean()), p_trans_eq0=max(pt, 1 / NB)),
                Rrank=float(dr_t.mean() / dr_d.mean()) if abs(dr_d.mean()) > 1e-12 else None, Rrank_ci=q(bR),
                agree_dil=float(np.nanmean(ag)), chance_dil=float(np.nanmean(ch)), agree_minus_chance_az0_ci=q(ba) if len(ba) else [None, None],
                agree_con=float(np.nanmean(agc)), chance_con=float(np.nanmean(chc)), overlap_dil=float(np.nanmean(ov)))


def main():
    from interp.jedi_core import sample
    x, y, _ = sample(PER, SEED); x = np.ascontiguousarray(x, np.float32); M = moments(x)
    Ib = np.random.default_rng(4942).integers(0, len(x), (NB, len(x)))
    log(f'sample({PER}, {SEED}): N {len(y)}')
    R = dict(header='see interp/x9x_verify3.py docstring', sample=dict(per_class=PER, seed=SEED, N=len(y)), nets={})
    fo = OUT / 'x9x_verify3.json'; save = lambda: fo.write_text(json.dumps(clean(R), indent=1))
    for arch in ('dsets', 'pfn', 'efn'):
        for A in AUGS:
            for s in SEEDS:
                key = f'{arch}_A{A}_s{s}'; net = load_net(arch, A, s)[0]; o = dict(arch=arch, A=A, seed=s, cells={})
                for where, T in offsets(A):
                    e = cell(net, x, M, round(T, 4), Ib); e['where'] = where; o['cells'][str(round(T, 4))] = e
                    log(key, where, f'T {T:.3f} changed {json.dumps(e["changed"])} Rc {e["Rc"]} drank t {e["drank"]["trans"]:.3f} '
                        f'{e["drank"]["trans_ci"]} d {e["drank"]["dil"]:.3f} c {e["drank"]["con"]:.3f} Rrank {e["Rrank"]} '
                        f'agree {e["agree_dil"]:.3f} (chance {e["chance_dil"]:.3f}) con {e["agree_con"]:.3f} (chance {e["chance_con"]:.3f}) '
                        f'overlap {e["overlap_dil"]:.3f} HL {json.dumps(e["heavier_lighter"])}')
                R['nets'][key] = o; save()
    cells = [(k, T) for k, v in R['nets'].items() for T, e in v['cells'].items() if e['where'] == 'beyond']
    h = holm([R['nets'][k]['cells'][T]['drank']['p_trans_eq0'] for k, T in cells])
    for (k, T), hh in zip(cells, h): R['nets'][k]['cells'][T]['drank']['p_holm'] = hh
    R['holm_m'] = len(cells); save(); log('wrote', fo, 'X9x verify3 DONE')


if __name__ == '__main__':
    main()
