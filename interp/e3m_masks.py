"""float program with HGQ's zero-bit entries kept as masks (no rounding, no wrap) vs the fixed-point network"""
import sys, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '.')
from interp.jedi_core import Jedi, sample
x, y, Q = sample(per_class=2000)
net = Jedi(64); L = net.L
KB = {k: (jnp.asarray(v[0], jnp.float32), jnp.asarray(v[1], jnp.float32)) for k, v in net.KB.items()}
def bits(name):
    q = L[name].iq
    qs = q.quantizers if hasattr(q, 'quantizers') else [q]
    out = []
    for qq in qs:
        qq = qq.quantizer if hasattr(qq, 'quantizer') else qq
        out.append(np.asarray(qq.b) if hasattr(qq, 'b') else None)
    return out
MASK = {}
for name in ('l1', 'qs', 'l2', 'dd', 'l3', 'qs1', 'd4', 'd5', 'd6', 'd7'):
    b = bits(name)[0]
    if b is not None:
        MASK[name] = jnp.asarray(b > 0, jnp.float32)
        print(f'{name:4s} bit shape {b.shape}, zero-bit entries {(b == 0).mean()*100:.1f}%')
badd = bits('add'); print('add', [None if v is None else (v.shape, float((v == 0).mean())) for v in badd])
def mk(mode):
    def q(name, t):
        if mode == 'fixed': return L[name].iq(t, training=False)
        if mode == 'mask' and name in MASK: return t * MASK[name]
        return t
    def dense(k, t, act=True):
        z = jnp.einsum(L[k].equation, q(k, t), KB[k][0]) + KB[k][1]
        return jax.nn.relu(z) if act and L[k].activation is not None else z
    def pool(k, a):
        return jnp.sum(q(k, a), axis=L[k].axes, keepdims=L[k].keepdims) * L[k].scale
    def f(xn):
        a = dense('l1', xn); g = dense('dd', pool('qs', a)); s = dense('l2', a)
        if mode == 'fixed':
            qs_, qd_ = L['add'].iq([s, g], training=False); t = qs_ + qd_
        else: t = s + g
        u = dense('l3', t); h = dense('d5', dense('d4', pool('qs1', u)))
        return dense('d7', jax.nn.relu(dense('d6', h, act=False))).reshape(len(xn), -1)
    return jax.jit(f)
xn = net.inputs(x, 64)
base = np.asarray(mk('fixed')(xn)).argmax(1)
for mode in ('fixed', 'mask', 'float'):
    c = np.asarray(mk(mode)(xn)).argmax(1)
    print(f'{mode:6s} acc {(c == y).mean():.4f} calls changed vs fixed {(c != base).mean()*100:.2f}%')
b = bits('qs1')[0][0]   # (64 slots, 64 channels)
live = [29, 0, 1, 19, 30, 43, 50]
for c in live:
    print(f'qs1 u{c}: slots kept {int((b[:, c] > 0).sum())}/64; first zero-bit slot {int(np.argmax(b[:, c] == 0)) if (b[:, c]==0).any() else None}; kept slots {np.flatnonzero(b[:, c] > 0)[:20].tolist()}')
