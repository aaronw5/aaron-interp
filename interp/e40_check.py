"""E40 diagnostic: the jet four-momentum check of the edits (max relative change), worst jets."""
import numpy as np
from interp.jedi_core import sample
from interp.e40_qg_count_jedi import jet4, mass, null_resort, nreal

X, y, _ = sample(2000, 4002); m0 = jet4(X); Y = null_resort(X, np.random.default_rng(1)); m = jet4(Y)
r = np.abs(m - m0).max(1) / m0[:, 0]; j = int(r.argmax())
print('worst', j, r[j], m0[j], m[j], nreal(X)[j])
print('pt min/max', X[j, :, 0].min(), X[j, :, 0].max(), 'max |angle|', np.abs(X[j, :, 1:]).max())
print('top r', np.sort(r)[-5:], 'neg pT', int((X[:, :, 0] < 0).sum()), 'E<=0', int((m0[:, 0] <= 0).sum()), 'smallest E', np.sort(m0[:, 0])[:5])
print('max |angle| overall', np.abs(X[:, :, 1:]).max(), 'jets with |angle|>1', int((np.abs(X[:, :, 1:]).max((1, 2)) > 1).sum()))
