"""The class fit and the attention fit on the test jets: the paper's metrics (accuracy, AUC, Rej) and agreement with ParT.
Checked first on the validation jets, where it must give the tuning's best validation agreement.

  python -m jetdistill.part.clseval cls|attn [which]      which: dev (the check) or full_test (default)"""
import json, sys, time
import numpy as np
from ..config import RESULTS
from .clsfit import particle_features, basis, cls_forward, log
from .attnfit import start_scores


class Ctx:
    """the jet quantities the fits use, each read once for all the jets (the test jets' BlockQ is lazy)"""
    def __init__(self, Q, keys): self.c = {k: np.asarray(Q[k], np.float32) for k in keys}
    def __getitem__(self, k): return self.c[k]


def logits(kind, which='full_test', net='full', device='mps', chunk=20000, sub=2000, log=log):
    import torch
    from ..pipeline import jets
    from .network import ParTNetwork
    t0 = time.time(); d = RESULTS / f'{kind}_fit' / net; meta = json.loads((d / f'{kind}_fit.json').read_text()); Pz = np.load(d / f'{kind}_fit.npz')
    knots = [np.asarray(k, np.float32) for k in meta['knots']]; J = jets(net, which); n = len(J['y'])
    J2 = dict(x=J['x'], ext=J['ext'], jet=J['jet'], Q=Ctx(J['Q'], meta['ctx']))
    T = lambda a: torch.from_numpy(np.asarray(a, np.float32)).to(device)
    model = ParTNetwork(net).model; model.eval()
    if kind == 'cls':
        W, V, mu = T(Pz['W']), T(Pz['V']), T(Pz['mu'])
        def fwd(F, m):
            xr = (basis(F, knots) @ W @ V + mu) * m[..., None]; return cls_forward(model, xr, m)[1]
    else:
        last = ParTNetwork(net).last; Kw, bw = T(last[0]), T(last[1]); M = T(Pz['M']); p = {k: T(Pz[k]) for k in Pz.files if k != 'M'}
        def fwd(F, m):                                                        # as attnfit.tune's run (pooled, pooled2)
            Phi = basis(F, knots) @ M
            sc = (start_scores(F) + Phi @ p['A']).masked_fill(~m[..., None], -1e9); sc = torch.cat([p['b_cls'].expand(len(m), 1, -1), sc], 1); al = torch.softmax(sc, 1)
            X = torch.cat([torch.einsum('nph,npk->nhk', al[:, 1:], Phi), al[:, 0, :, None]], 2).reshape(len(m), -1)
            if meta['layers'] == 2:
                z1 = X @ p['W1'] + p['c1']; z1 = torch.cat([z1, torch.ones_like(z1[:, :1])], 1); Q2 = torch.einsum('khr,nr->nkh', p['A2'], z1)
                sc = (start_scores(F) + torch.einsum('npk,nkh->nph', Phi, Q2)).masked_fill(~m[..., None], -1e9)
                sc = torch.cat([(z1 @ p['b2_cls'].T)[:, None], sc], 1); al = torch.softmax(sc, 1)
                X = torch.cat([X, torch.cat([torch.einsum('nph,npk->nhk', al[:, 1:], Phi), al[:, 0, :, None]], 2).reshape(len(m), -1)], 1)
            return (X @ p['W'] + p['c']) @ Kw + bw
    out = np.empty((n, 10), np.float32)
    with torch.no_grad():
        for a in range(0, n, chunk):
            r = np.arange(a, min(a + chunk, n)); F, ok = particle_features(J2, r); o = np.argsort(ok.sum(1), kind='stable')
            for b in range(0, len(r), sub):
                rr = o[b:b + sub]; P = int(ok[rr].sum(1).max())
                Ft = torch.from_numpy(F[rr, :P]).to(device, torch.float16).float(); mt = torch.from_numpy(ok[rr, :P]).to(device)   # float16 as in the tuning
                out[r[rr]] = fwd(Ft, mt).cpu().numpy()
            if device == 'mps': torch.mps.empty_cache()
            if a % 200000 == 0: log(f'  {kind} fit on {which}: {a + len(r)} of {n} jets, {time.time() - t0:.0f} s')
    return out, J


def evaluate(kind, which='full_test', net='full', log=log):
    from ..metrics import softmax, paper_metrics
    L, J = logits(kind, which, net, log=log); P = softmax(L.astype(np.float64)); pred = P.argmax(1)
    r = dict(kind=kind, which=which, n=len(pred), agreement=float((pred == J['net']).mean()), **paper_metrics(P, J['y']))
    if which == 'full_test':
        r['part'] = paper_metrics(J['P'], J['y'])
        r['per_class'] = [dict(c=int(c), agreement=float((pred[J['net'] == c] == c).mean()), accuracy=float((pred[J['y'] == c] == c).mean())) for c in range(10)]
        np.save(RESULTS / f'{kind}_fit' / net / 'test_logits.npy', L.astype(np.float16))
    (RESULTS / f'{kind}_fit' / net / f'metrics_{which}.json').write_text(json.dumps(r, indent=1))
    log(f'{kind} fit on {which} ({len(pred)} jets): same class as ParT {100 * r["agreement"]:.2f}%, accuracy {100 * r["accuracy"]:.2f}%, AUC {r["auc"]:.4f}')
    return r


if __name__ == '__main__':
    evaluate(sys.argv[1], *(sys.argv[2:3] or ['full_test']))
