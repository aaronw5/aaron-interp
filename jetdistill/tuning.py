"""Steps 2 and 3: all coefficients tuned together, then pruning with retraining.

Step 2 keeps the terms and thresholds of step 1 (the first K terms of each neuron, coefficients refitted by least squares)
and trains every coefficient and intercept at once with Adam, full batch, on 100,000 fitting jets. Targets:
  probabilities  soft cross-entropy to the network's class probabilities, through the network's own last layer
                 (formula neurons rounded to the network's fixed-point grid, straight-through gradient)
  decisions      cross-entropy to the network's class (one-hot)
  labels         cross-entropy to the true class
  neurons        no class scores: only R (below)
plus λ·R for the first three, R = mean over neurons of (h_j − h_j,network)² / Var(h_j,network) (keeps each formula neuron
close to the network's; the 16 neurons reach the probabilities only through 5 class scores, so without R the tuning can
move them in directions the probabilities do not see).
The kept step (checked every 50 of 800) is the one with the best validation score: agreement with the network's class
(probabilities, decisions), accuracy (labels) or mean neuron R² (neurons).
Step 3 removes the weakest terms (|coefficient| × spread of the term), 15 % at a time, retraining after each cut; a cut is
kept while the validation score stays within TOL of step 2's (0.1 point; 0.005 of R² for 'neurons'), else the cut size
is halved, down to 2 %."""
import numpy as np
from . import formula as F

from .config import SMOKE


def log(*a):
    print(*a, flush=True)
STEPS, LR = (800 if not SMOKE else 100), 3e-3
STEPS_BY_TARGET = dict(labels=500) if not SMOKE else {}     # as in the original runs
TOL = dict(probabilities=0.001, decisions=0.001, labels=0.001, neurons=0.005)


def refit(neurons, Q, Z, ridge=1e-6):
    """least-squares coefficients of the given terms to each neuron's pre-activation Z (clipped below as in step 1)"""
    out = []
    for nr, B in zip(neurons, F.bases(neurons, Q)):
        z = Z[:, nr['neuron']]; zp = z[z > 0]; tgt = np.maximum(z, -0.2 * (zp.std() if len(zp) > 50 else z.std()))
        if not nr['terms']: out.append(dict(nr, intercept=float(tgt.mean()))); continue
        sd = B.std(0) + 1e-12; Bn = (B - B.mean(0)) / sd
        c = np.linalg.solve(Bn.T @ Bn + ridge * len(Bn) * np.eye(Bn.shape[1]), Bn.T @ (tgt - tgt.mean())) / sd
        out.append(dict(nr, intercept=float(tgt.mean() - (B.mean(0) * c).sum()), terms=[dict(t, coef=float(a)) for t, a in zip(nr['terms'], c)]))
    return out


def neuron_r2(H, Hn):
    """mean over neurons of 1 − MSE / Var (neurons constant in the network left out)"""
    v = Hn.var(0); ok = v > 1e-9
    return float(np.mean(1 - ((H - Hn) ** 2).mean(0)[ok] / v[ok]))


def train(formula, Qf, Qd, target, fit, dev, last, lam=0.0, steps=None, lr=LR, history=None):
    """fit/dev: dict(P=network probabilities, y=true class, net=network class, H=network neurons (ReLU)) of those jets.
    Returns (formula with trained coefficients, validation score of the kept step). history: a list that receives, every 50
    steps, the loss and the score on the training and on the validation jets (analysis/convergence.py)."""
    import jax, jax.numpy as jnp
    steps = steps or STEPS_BY_TARGET.get(target, STEPS)
    K7, b7, i7, f7 = (jnp.asarray(x, jnp.float32) for x in last)
    B = [jnp.asarray(b, jnp.float32) for b in F.bases(formula, Qf)]; Bd = [jnp.asarray(b, jnp.float32) for b in F.bases(formula, Qd)]
    sd = [jnp.asarray(np.asarray(b).std(0) + 1e-9) for b in B]
    th = [(jnp.asarray(c, jnp.float32) * s, jnp.asarray(c0, jnp.float32)) for (c, c0), s in zip(F.coefs(formula), sd)]   # scaled coefficients

    def hidden(th, Bs):
        return jnp.stack([jax.nn.relu(b @ (c / s) + c0) for (c, c0), b, s in zip(th, Bs, sd)], 1)

    def logits(th, Bs):
        h = hidden(th, Bs); sc = 2.0 ** f7; hq = (jnp.floor(h * sc + 0.5) / sc) % (2.0 ** i7)
        return (h + jax.lax.stop_gradient(hq - h)) @ K7 + b7

    Hn = jnp.asarray(fit['H'], jnp.float32); vn = jnp.asarray(np.asarray(fit['H']).var(0) + 1e-6, jnp.float32)
    R = lambda th: jnp.mean((hidden(th, B) - Hn) ** 2 / vn)
    T = {'probabilities': fit['P'], 'decisions': np.eye(5)[fit['net']], 'labels': np.eye(5)[fit['y']]}.get(target)
    if target == 'neurons':
        loss = R
    else:
        Tj = jnp.asarray(T, jnp.float32)
        loss = (lambda th: -jnp.mean(jnp.sum(Tj * jax.nn.log_softmax(logits(th, B)), 1)) + lam * R(th)) if lam > 0 else \
               (lambda th: -jnp.mean(jnp.sum(Tj * jax.nn.log_softmax(logits(th, B)), 1)))
    if target == 'neurons':
        ev = jax.jit(lambda th: hidden(th, Bd)); score = lambda th: neuron_r2(np.asarray(ev(th)), dev['H'])
    else:
        ev = jax.jit(lambda th: logits(th, Bd)); ref = dev['y'] if target == 'labels' else dev['net']
        score = lambda th: float((np.asarray(ev(th)).argmax(1) == ref).mean())
    if history is not None:
        evf = jax.jit(lambda th: hidden(th, B) if target == 'neurons' else logits(th, B)); ref_f = fit['y'] if target == 'labels' else fit['net']
        score_fit = (lambda th: neuron_r2(np.asarray(evf(th)), fit['H'])) if target == 'neurons' else (lambda th: float((np.asarray(evf(th)).argmax(1) == ref_f).mean()))
    g = jax.jit(jax.value_and_grad(loss)); tm = jax.tree_util.tree_map
    m_, v_ = tm(jnp.zeros_like, th), tm(jnp.zeros_like, th); best = (-np.inf, th)
    for i in range(steps):
        _, gr = g(th); m_ = tm(lambda a, b: .9 * a + .1 * b, m_, gr); v_ = tm(lambda a, b: .999 * a + .001 * b * b, v_, gr)
        th = tm(lambda p, a, b: p - lr * (a / (1 - .9 ** (i + 1))) / (jnp.sqrt(b / (1 - .999 ** (i + 1))) + 1e-8), th, m_, v_)
        if i % 50 == 49 or i == steps - 1:
            s = score(th)
            if history is not None: history.append(dict(step=i + 1, loss=float(loss(th)), train=score_fit(th), val=s))
            if s > best[0]: best = (s, th)
    return F.with_coefs(formula, [(np.asarray(c / s, np.float64), float(c0)) for (c, c0), s in zip(best[1], sd)]), best[0]


def prune(formula, Qf, Qd, target, fit, dev, last, lam=0.0, frac=0.15, min_frac=0.02, log=log):
    """step 3; returns (pruned formula, path of (terms, validation score))"""
    cur, ref = train(formula, Qf, Qd, target, fit, dev, last, lam); path = [(F.n_terms(cur), ref)]
    log(f'step 2: {path[-1][0]} terms, validation {ref:.4f}')
    while frac >= min_frac:
        imp = sorted((abs(t['coef']) * float(b[:, i].std()), j, i) for j, (nr, b) in enumerate(zip(cur, F.bases(cur, Qf))) for i, t in enumerate(nr['terms']))
        if not imp: break
        cut = {(j, i) for _, j, i in imp[:max(1, int(len(imp) * frac))]}
        trial, s = train([dict(nr, terms=[t for i, t in enumerate(nr['terms']) if (j, i) not in cut]) for j, nr in enumerate(cur)], Qf, Qd, target, fit, dev, last, lam)
        if s >= ref - TOL[target]:
            cur = trial; path.append((F.n_terms(cur), s)); log(f'  kept a cut of {len(cut)}: {path[-1][0]} terms, validation {s:.4f}')
        else:
            frac /= 2
    return cur, path


def targets(net_run, y):
    """the tuning targets of a set of jets from the network's outputs"""
    L = net_run['logits']; P = np.exp(L - L.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    return dict(P=P, net=L.argmax(1), y=y, H=np.maximum(net_run['z'], 0))
