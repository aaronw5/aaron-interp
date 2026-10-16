"""P0 — smoke test of the ParT package inside aaron-interp (claims A8/A10; stage 1).
400 test jets per class (rows 3,000–3,400 of each class's first test file, as G192): ParT_full must give ≈ 86 % (G192: 86.2 %),
ParT_kin ≈ 74 % (74.4 %) on the same jets. Also: the rebuild identity (build(J, kin(J)) with no edit: calls changed, max input
differences), the attention hooks (capture on, mask None: same logits), and the moment approximation m² ≈ M0·M2 − |M1|².
  ./run_part.sh interp/p0_smoke_part.py > logs/p0_smoke.log 2>&1"""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp.part_core import ParTNetwork, Attention, sample, kin, build, jmass, moments, OUT, CL

t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
log = lambda *a: print(*a, flush=True)
J0 = sample(400, 3000); y = J0['label'].astype(int); N = len(y)
log(f'{N} jets read ({time.time() - t0:.0f} s); multiplicity median {int(np.median(J0["mask"].sum(1)))}, max {int(J0["mask"].sum(1).max())}')
res = dict(N=N, per_class=400, rows=[3000, 3400])
pt, de, dp, ms = kin(J0); M = moments(pt, de, dp); m = jmass(J0)
ok = m > 1
res['approx'] = dict(corr_m=float(np.corrcoef(np.sqrt(np.maximum(M['m2a'], 0))[ok], m[ok])[0, 1]),
                     median_ratio=float(np.median(np.sqrt(np.maximum(M['m2a'], 0))[ok] / m[ok])),
                     M1sq_over_M0M2_median=float(np.median((M['M1'] ** 2 / M['M0M2'])[ok])),
                     particle_mass_median=float(np.median(ms[pt > 0])), particle_mass_p95=float(np.percentile(ms[pt > 0], 95)))
log('moments: corr(sqrt(M0M2-|M1|^2), m) {corr_m:.5f}, median ratio {median_ratio:.4f}, |M1|^2/(M0M2) median {M1sq_over_M0M2_median:.2e}; '
    'particle mass median {particle_mass_median:.3f} GeV p95 {particle_mass_p95:.3f}'.format(**res['approx']))
J1 = build(J0, pt, de, dp, ms)
res['rebuild'] = {k: float(np.abs(J1[k] - J0[k]).max()) for k in ('px', 'py', 'pz', 'energy', 'deta', 'dphi', 'mask')}
res['rebuild'].update(jet_pt_rel=float(np.abs(J1['jet_pt'] / J0['jet_pt'] - 1).max()), jet_energy_rel=float(np.abs(J1['jet_energy'] / J0['jet_energy'] - 1).max()),
                      mass_rel=float(np.abs(jmass(J1) / np.maximum(m, 1e-3) - 1)[ok].max()))
log('rebuild (no edit) max |diff|: ' + ', '.join(f'{k} {v:.3g}' for k, v in res['rebuild'].items()))
for n, want in (('full', 86.2), ('kin', 74.4)):
    net = ParTNetwork(n); t1 = time.time(); R0 = net.run(J0); c0 = R0['logits'].argmax(1); acc = float((c0 == y).mean())
    R1 = net.run(J1); c1 = R1['logits'].argmax(1)
    try:                      # the attention hooks need torch >= 2.0 (register_forward_pre_hook(..., with_kwargs=True))
        att = Attention(net); L2 = np.concatenate([att.forward({k: v[i:i + 200] for k, v in J0.items()}, capture=True)[0] for i in range(0, N, 200)]); att.remove()
        hook_diff = float(np.abs(L2 - R0['logits']).max())
    except TypeError as e:
        hook_diff = f'hooks unavailable ({e})'
    row = dict(acc=acc, G192=want, s_per_1000=float((time.time() - t1) / 3 / N * 1000), calls_changed_by_rebuild=float((c1 != c0).mean()),
               max_logit_diff_rebuild=float(np.abs(R1['logits'] - R0['logits']).max()), max_logit_diff_hooks=hook_diff,
               own_class=[float((c0[y == k] == k).mean()) for k in range(10)], z_sd_n98=float(R0['z'][:, 98].std()),
               z_active=int((R0['z'].std(0) > 0.01).sum()))
    res[n] = row
    log(f'ParT_{n}: accuracy {100 * acc:.1f} % (G192 {want} %); {row["s_per_1000"]:.1f} s per 1,000 jets; rebuild changes {100 * row["calls_changed_by_rebuild"]:.2f} % of calls '
        f'(max logit diff {row["max_logit_diff_rebuild"]:.3g}); hooks on: max logit diff {hook_diff if isinstance(hook_diff, str) else f"{hook_diff:.3g}"}; active neurons {row["z_active"]}, sd(z[98]) {row["z_sd_n98"]:.3f}')
    log('   own-class rate: ' + ' '.join(f'{c} {100 * v:.0f}' for c, v in zip(CL, row['own_class'])))
    del net
json.dump(res, open(OUT / 'p0_smoke.json', 'w'), indent=1); log(f'P0 DONE ({time.time() - t0:.0f} s)')
