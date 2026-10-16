"""does the decision depend on the pooling quantizers' rounding? forward with each quantizer optionally skipped"""
import sys, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '.')
from interp.jedi_core import Jedi, sample
x, y, Q = sample(per_class=2000)
net = Jedi(64); L = net.L
KB = {k: (jnp.asarray(v[0], jnp.float32), jnp.asarray(v[1], jnp.float32)) for k, v in net.KB.items()}
def mk(skip):
    def q(name, t):
        return t if name in skip else L[name].iq(t, training=False)
    def dense(k, t, act=True):
        z = jnp.einsum(L[k].equation, q(k, t), KB[k][0]) + KB[k][1]
        return jax.nn.relu(z) if act and L[k].activation is not None else z
    def pool(k, a):
        return jnp.sum(q(k, a), axis=L[k].axes, keepdims=L[k].keepdims) * L[k].scale
    def f(xn):
        a = dense('l1', xn); g = dense('dd', pool('qs', a)); s = dense('l2', a)
        if 'add' in skip: t = s + g
        else:
            qs_, qd_ = L['add'].iq([s, g], training=False); t = qs_ + qd_
        u = dense('l3', t); h = dense('d5', dense('d4', pool('qs1', u)))
        z = dense('d6', h, act=False)
        return dense('d7', jax.nn.relu(z)).reshape(len(xn), -1)
    return jax.jit(f)
xn = net.inputs(x, 64)
base = np.asarray(mk(())(xn)).argmax(1)
print('base acc', (base == y).mean())
ALL = ("l1","qs","l2","dd","add","l3","qs1","d4","d5","d6","d7")
for skip in [tuple(k for k in ALL if k != keep) for keep in ALL] + [("d7",), ("d6",), ("d6", "d7")]:
    c = np.asarray(mk(skip)(xn)).argmax(1)
    lab = ("all float except " + [k for k in ALL if k not in skip][0]) if len(skip) == 10 else "float " + "+".join(skip)
    print(f"{lab:32s} acc {(c == y).mean():.4f}  calls changed {(c != base).mean()*100:.2f}%")
