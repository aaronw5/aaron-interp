"""X9 environment check (run with the venv python /j-jepa-vol/aaron-interp/venv-ef/bin/python): versions, a tiny EFN and PFN
built, run on random inputs, and their Dense weights listed (the order x9_train.export relies on)."""
import numpy as np
import tensorflow as tf
import energyflow as ef
from energyflow.archs import EFN, PFN

print('tensorflow', tf.__version__, 'keras', getattr(tf.keras, '__version__', '?'), 'energyflow', ef.__version__, 'numpy', np.__version__, flush=True)
e = EFN(input_dim=2, Phi_sizes=(8, 4), F_sizes=(6,), output_dim=5, summary=False)
p = PFN(input_dim=3, Phi_sizes=(8, 4), F_sizes=(6,), output_dim=5, summary=False)
z = np.random.rand(4, 7).astype('float32'); z[:, 5:] = 0; z /= z.sum(1, keepdims=True); a = np.random.randn(4, 7, 2).astype('float32')
print('EFN out', e.predict([z, a], verbose=0).shape, flush=True)
xp = np.random.randn(4, 7, 3).astype('float32'); xp[:, 5:] = 0
print('PFN out', p.predict(xp, verbose=0).shape, flush=True)
for name, m in (('EFN', e), ('PFN', p)):
    for L in m.model.layers:
        print(name, L.__class__.__name__, L.name, [w.shape for w in L.get_weights()], getattr(L, 'activation', None).__name__ if hasattr(L, 'activation') else '')
print('ENVCHECK OK')
