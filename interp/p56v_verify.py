"""P56v (loop C verifier): rerun of P56's L-GATr W/Z arm on fresh rows, plus two missing controls.

Sample. JetClass test_20M WToQQ_117 / ZToQQ_117, 2,000 rows per file drawn with numpy default_rng(5657) from the rows NOT drawn
by P56 (P56's default_rng(5656) draw is regenerated and excluded; overlap asserted 0). Never first-k. Jets with >= 3 particles.

Model. L-GATr final (model_run0.pt). Edits: P56's own generator (interp.p56_lgatr_mass_vs_spread.edits) unchanged, so the
rerun tests the same code on new jets: E4 (own-axis dilation), M (particle masses raised in quadrature at fixed p), N (energy
shuffle at fixed m), X (two hardest outward), S (spread up, other masses lowered, m fixed), ZERO / PION.
Routes A (full), D (edited four-vectors, unedited scalar features), F (unedited four-vectors, edited features).

Controls added here.
 NF_f  sign-flipped energy shuffle: E_i(NF) = E0_i - (E_i(N) - E0_i). Same |dE_i| as N and as M, Sum dE = 0, m and Sum p fixed.
       If the four-vector route's response to N is linear in the signed dE_i, d(NF) = -d(N); if it is a generic response to
       off-shell particles (sign-symmetric), d(NF) = d(N). sym = (d(N) + d(NF)) / 2 estimates the sign-symmetric part.
 rerun L-GATr base run twice on identical input: per-jet nondeterminism (mean, sd, max) and its size as an R contribution.

  ./sync.sh && ./part_job.sh c-p56-verify '/j-jepa-vol/aaron-interp/env-lgatr/bin/python -m interp.p56v_verify'
-> $AI_RESULTS/part/p56_verify.json
"""
import sys, os, json, time
from pathlib import Path
sys.argv = [sys.argv[0], '--stage', 'none']
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import interp.p56_lgatr_mass_vs_spread as P56

SEED_NEW, NB = 5657, 2000
OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'
T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def sample():
    import uproot
    from interp.part_core import sub, cat
    r0, r1 = np.random.default_rng(P56.SEED), np.random.default_rng(SEED_NEW); jobs = []
    for pre in ('WToQQ', 'ZToQQ'):
        p = str(P56.VAL / f'{pre}_{P56.FIDX}.root'); N = uproot.open(p)['tree'].num_entries
        old = r0.choice(N, 2000, replace=False); pool = np.setdiff1d(np.arange(N), old)
        new = np.sort(r1.choice(pool, 2000, replace=False)); assert len(np.intersect1d(new, old)) == 0
        jobs.append((p, new))
    J = cat([P56._read_rows(j) for j in jobs]); ok = (J['mask'] > 0).sum(1) >= 3
    log('jets with >= 3 particles', int(ok.sum()), 'of', len(ok)); return sub(J, np.where(ok)[0])


def main():
    import torch
    import interp.p44_lgatr_grades as P44
    from interp.part_core import sub, jmass
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    lg = P44.LG(f'{P56.LGRUN}/model_run0.pt', dev); log('loaded L-GATr on', dev)
    J = sample(); y = J['label'].astype(int); N = len(y)
    dfun = lambda L: (L[:, P56.iW] - L[:, P56.iZ]).astype(np.float64)
    D, P, ROWS, META = {}, {}, {}, {}
    for name, rows, Je, I in P56.edits(J):
        META[name] = I; ROWS[name] = rows
        if name == 'base':
            Jb = Je; L = lg.logits(Je); L2 = lg.logits(Je); L3 = lg.logits(Je)
            D['base'] = dfun(L); D['rerun'] = dfun(L2); D['rerun2'] = dfun(L3); P['base'] = L.argmax(1); P['rerun'] = L2.argmax(1)
            log('base acc', float((L.argmax(1) == y).mean())); continue
        idx = np.where(rows)[0]; Jbr = sub(Jb, idx)
        variants = [(name, Je)]
        if name.startswith('N'):
            JF = dict(Je); m = Je['mask'] > 0
            JF['energy'] = np.where(m, 2 * Jbr['energy'].astype(np.float64) - Je['energy'].astype(np.float64), 0).astype(np.float32)
            JF['jet_energy'] = JF['energy'].sum(1).astype(np.float32)
            p = np.sqrt(Je['px'].astype(np.float64) ** 2 + Je['py'] ** 2 + Je['pz'] ** 2)
            META['NF' + name[1:]] = dict(n=int(len(idx)), frac_E_below_p=float(((JF['energy'] < p - 1e-6) & m).sum() / m.sum()),
                                         n_E_nonpos=int(((JF['energy'] <= 0) & m).sum()),
                                         dlogm_maxabs=float(np.abs(np.log(np.maximum(jmass(JF), 1e-6) / np.maximum(jmass(Jbr), 1e-6))).max()))
            ROWS['NF' + name[1:]] = rows; variants.append(('NF' + name[1:], JF))
        for nm, Jv in variants:
            for route in ('A', 'D', 'F'):
                if nm.startswith(('S', 'ZERO', 'PION', 'E4|ZERO', 'E4|PION')) and route != 'A' and not nm.startswith('S'): continue
                if not rows.any(): continue
                L = lg.logits(Jv) if route == 'A' else (lg.logits(Jv, Jbr) if route == 'D' else lg.logits(Jbr, Jv))
                a = np.full(N, np.nan); a[idx] = dfun(L); D[f'{nm}|{route}'] = a
                if route == 'A': pp = np.full(N, -1); pp[idx] = L.argmax(1); P[nm] = pp
            log(nm, 'mean dd A', round(float(np.nanmean(D[f'{nm}|A'] - D['base'])), 4))
    # ---------------------------------------------------------------- analysis
    rb = np.random.default_rng(SEED_NEW + 3); C = np.zeros((NB, N), np.float32)
    for b in range(NB): C[b] = np.bincount(rb.integers(0, N, N), minlength=N)
    dd = lambda k: np.where(np.isfinite(D[k] - D['base']), D[k] - D['base'], 0.0)
    ci = lambda b: [float(np.percentile(b[np.isfinite(b)], 2.5)), float(np.percentile(b[np.isfinite(b)], 97.5))]

    def bp(b, v0=0.0):
        b = b[np.isfinite(b)] - v0; return float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()) + 1 / len(b)))

    def R(pairs, route='A', minus=None, combine=None):
        nu, de = np.zeros(N), np.zeros(N)
        for i, (e, d) in enumerate(pairs):
            rr = ROWS[e] & ROWS[d]; x = dd(f'{e}|{route}')
            if minus: x = x - dd(f'{minus[i]}|{route}'); rr = rr & ROWS[minus[i]]
            if combine: x = 0.5 * (x + combine[1] * dd(f'{combine[0][i]}|{route}'))
            nu += x * rr; de += dd(f'{d}|A') * rr
        b = (C @ nu) / (C @ de); return float(nu.sum() / de.sum()), b
    FS, res, BO = P56.FS, {}, {}
    arms = {f'{e}{f:+.2f}': [(f'{e}{f:+.2f}', f'E4|{f:+.2f}')] for e in ('M', 'N', 'NF', 'X') for f in FS}
    arms.update({f'{e}_pooled': [(f'{e}{f:+.2f}', f'E4|{f:+.2f}') for f in FS] for e in ('M', 'N', 'NF', 'X')})
    arms.update({'S_pooled': [(f'S{f:+.2f}', f'E4|{f:+.2f}') for f in P56.FS_S], 'ZERO': [('ZERO', 'E4|ZERO')], 'PION': [('PION', 'E4|PION')]})
    for an, pr in arms.items():
        for route in ('A', 'D', 'F'):
            if any(f'{e}|{route}' not in D for e, _ in pr): continue
            v, b = R(pr, route); BO[(an, route)] = b; res[f'{an}|{route}'] = dict(R=v, ci=ci(b))
    for route in ('A', 'D', 'F'):
        v, b = R([('X+0.05', 'E4|+0.05')], route, minus=['M+0.05']); BO[('Srel', route)] = b; res[f'Srel|{route}'] = dict(R=v, ci=ci(b), p0=bp(b))
        prN = [(f'N{f:+.2f}', f'E4|{f:+.2f}') for f in FS]; nf = [f'NF{f:+.2f}' for f in FS]
        v, b = R(prN, route, combine=(nf, +1)); res[f'Nsym_pooled|{route}'] = dict(R=v, ci=ci(b), p0=bp(b))
        v, b = R(prN, route, combine=(nf, -1)); res[f'Nanti_pooled|{route}'] = dict(R=v, ci=ci(b), p0=bp(b))
    for f in FS:
        e = f'E4|{f:+.2f}'; a = dd(f'{e}|A').sum(); res[f'{e}|route_share'] = dict(D=float(dd(f'{e}|D').sum() / a), F=float(dd(f'{e}|F').sum() / a))
    bM, bN = BO[('M_pooled', 'A')], BO[('N_pooled', 'A')]
    tests = dict(p_RM_eq_1=bp(bM, 1.0), p_RM_eq_0=bp(bM), RM_minus_RN=dict(v=res['M_pooled|A']['R'] - res['N_pooled|A']['R'], ci=ci(bM - bN)),
                 p_RS_eq_0=bp(BO[('S_pooled', 'A')]), p_RS_eq_1=bp(BO[('S_pooled', 'A')], 1.0), p_N_eq_NF=bp(BO[('N_pooled', 'A')] - BO[('NF_pooled', 'A')]))
    # rerun noise
    rr = D['rerun'] - D['base']; rr2 = D['rerun2'] - D['base']
    noise = dict(mean=float(rr.mean()), sd=float(rr.std()), maxabs=float(np.abs(rr).max()), mean2=float(rr2.mean()), maxabs2=float(np.abs(rr2).max()),
                 frac_jets_gt_0p01=float((np.abs(rr) > 0.01).mean()), calls_changed=float((P['rerun'] != P['base']).mean()),
                 as_R_per_dose={f'{f:+.2f}': float(rr.sum() / dd(f'E4|{f:+.2f}|A').sum()) for f in FS})
    # calls
    from scipy.stats import binomtest
    calls = {}
    for e in ('M+0.05', 'N+0.05', 'NF+0.05', 'X+0.05', 'E4|+0.05', 'M+0.10', 'E4|+0.10'):
        r_ = ROWS[e]; p0, p1 = P['base'][r_], P[e][r_]; yr = y[r_]; w = yr == P56.iW; c0, c1 = p0 == yr, p1 == yr
        x, z = int((c0 & ~c1).sum()), int((~c0 & c1).sum())
        calls[e] = dict(changed=float((p0 != p1).mean()), WtoZ=[float((p0[w] == P56.iZ).mean()), float((p1[w] == P56.iZ).mean())],
                        lost=x, gained=z, p=float(binomtest(x, x + z).pvalue) if x + z else 1.0)
    out = dict(node='P56v', sample=f'test_20M WToQQ/ZToQQ_{P56.FIDX}, 2000 rows/file, default_rng({SEED_NEW}) excluding P56 rows', N=N,
               acc_base=float((P['base'] == y).mean()), meta=META, R=res, tests=tests, rerun=noise, calls=calls)
    OUT.mkdir(parents=True, exist_ok=True); f = OUT / 'p56_verify.json'; f.write_text(json.dumps(out, indent=1, default=float)); log('wrote', f)
    for k, v in res.items(): log(k, v)
    log(tests); log(noise); log(calls)


if __name__ == '__main__':
    main()
