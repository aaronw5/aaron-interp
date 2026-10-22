"""H80 diagnostic: coincident / lattice positions of constituents in Top Landscape and QG test jets (why edits are detectable)."""
import numpy as np
from pathlib import Path

for name, d in (('top', Path('/j-jepa-vol/aaron-interp/data/top/test')), ('qg', Path('/j-jepa-vol/aaron-interp/data/qg/test'))):
    F = np.load(d / 'full.npy', mmap_mode='r'); jet = np.load(d / 'jet.npy'); rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(len(jet), 1000, replace=False)); P = np.asarray(F[rows], np.float64); nn = np.minimum(jet[rows, 5].astype(int), P.shape[1])
    mind, ncoin, npair, etas, phis, ms, pts = [], 0, 0, [], [], [], []
    for j in range(len(rows)):
        p = P[j, :nn[j], :4]; pt = np.hypot(p[:, 0], p[:, 1]); eta = np.arcsinh(p[:, 2] / pt); phi = np.arctan2(p[:, 1], p[:, 0])
        D = np.hypot(eta[:, None] - eta[None], (phi[:, None] - phi[None] + np.pi) % (2 * np.pi) - np.pi); iu = np.triu_indices(len(pt), 1)
        d = D[iu]; mind.append(d.min()); ncoin += (d < 1e-4).sum(); npair += len(d); etas.append(eta); phis.append(phi)
        ms.append(np.sqrt(np.maximum(p[:, 3] ** 2 - (p[:, :3] ** 2).sum(1), 0)) / p[:, 3]); pts.append(pt)
    mind = np.array(mind); eta = np.concatenate(etas); phi = np.concatenate(phis); m = np.concatenate(ms); pt = np.concatenate(pts)
    print(name, 'jets', len(rows), 'median N', np.median(nn), 'frac jets with a pair dR<1e-4', (mind < 1e-4).mean(), 'pairs<1e-4 frac', ncoin / npair,
          'min pair dR quantiles', np.quantile(mind, [0.1, 0.5, 0.9]))
    for g in (0.0174, 0.0175, 0.025, 0.05, 0.087, 0.0873, 0.1):
        fe = np.abs((eta / g) - np.round(eta / g)); fp = np.abs((phi / g) - np.round(phi / g))
        print(f'  grid {g}: frac |eta/g - round| < 0.01: {(fe < 0.01).mean():.3f}; phi: {(fp < 0.01).mean():.3f} (uniform expectation 0.02)')
    u, c = np.unique(np.round(eta, 6), return_counts=True); print('  distinct eta values', len(u), 'of', len(eta), 'max multiplicity', c.max())
    print('  m/E quantiles', np.quantile(m, [0.5, 0.9, 0.99]), 'pT min quantiles', np.quantile(pt, [0.001, 0.01, 0.1]))
    print('  pT float32 roundoff check: unique pT', len(np.unique(np.round(pt, 4))), 'of', len(pt))
