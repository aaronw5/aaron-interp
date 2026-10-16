"""The Particle Transformer (ParT, Qu, Li & Qian 2022), official JetClass models (github.com/jet-universe/particle_transformer,
models/ParT_kin.pt and ParT_full.pt), run exactly as the weaver framework feeds it.

ParTNetwork(n).run(jets) gives, for jets from data.read_root (particles as the files store them):
  z       (J, 128)  the class token after the last LayerNorm ("the neurons" the formulas describe; no activation follows)
  h       (J, 128)  the same numbers (the last layer reads them as they are)
  logits  (J, 10)   the class scores (QCD, H→bb, H→cc, H→gg, H→4q, H→ℓνqq′, Z→qq, W→qq, t→bqq′, t→bℓν)
ParTNetwork.last = (W (128, 10), b (10,), None, None): the network's own last layer, logits = h·W + b (the two None
stand where the JEDI tagger keeps its fixed-point formats: no rounding here).

The model code is weaver-core 0.4.x (the version the released weights were trained with; 0.5 computes differently).
Inputs follow data/JetClass/JetClass_<kin|full>.yaml of the ParT repository: each scaled feature is
clip((x - center) * scale, -5, 5), padded slots are filled by repeating the jet's own particles (weaver's 'wrap'
padding; they are masked out, only their values must be finite), the pairwise features are built by the model from the
particles' (px, py, pz, E)."""
import os
import numpy as np
from ..config import MODELS

# (name, center, scale, clip): the pf_features of the two yaml files, in their order
KIN = [('pt_log', 1.7, 0.7, (-5, 5)), ('e_log', 2.0, 0.7, (-5, 5)), ('logptrel', -4.7, 0.7, (-5, 5)), ('logerel', -4.7, 0.7, (-5, 5)),
       ('deltaR', 0.2, 4.0, (-5, 5))]
FEATURES = dict(
    kin=KIN + [('deta', None, 1, None), ('dphi', None, 1, None)],
    full=KIN + [(k, None, 1, None) for k in ('charge', 'isChargedHadron', 'isNeutralHadron', 'isPhoton', 'isElectron', 'isMuon', 'd0')]
         + [('d0err', 0, 1, (0, 1)), ('dz', None, 1, None), ('dzerr', 0, 1, (0, 1)), ('deta', None, 1, None), ('dphi', None, 1, None)])
CFG = dict(num_classes=10, pair_input_dim=4, use_pre_activation_pair=False, embed_dims=[128, 512, 128], pair_embed_dims=[64, 64, 64],
           num_heads=8, num_layers=8, num_cls_layers=2, block_params=None, cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
           fc_params=[], activation='gelu', trim=True, for_inference=False)


def raw_features(J, k):
    """one per-particle feature before scaling, (J, P) float32, from the jet dict of data.read_root"""
    pt = np.hypot(J['px'], J['py']); E = J['energy']; ok = J['mask'] > 0; one = lambda a: np.where(ok, a, 1.0)
    f = dict(pt_log=lambda: np.log(one(pt)), e_log=lambda: np.log(one(E)), logptrel=lambda: np.log(one(pt) / J['jet_pt'][:, None]),
             logerel=lambda: np.log(one(E) / J['jet_energy'][:, None]), deltaR=lambda: np.hypot(J['deta'], J['dphi']),
             d0=lambda: np.tanh(J['d0val']), dz=lambda: np.tanh(J['dzval']))
    return np.asarray(f[k]() if k in f else J[k], np.float32)


def inputs(J, n):
    """(features (J, F, P), vectors (J, 4, P), mask (J, 1, P)) as float32, P = the largest multiplicity among the jets"""
    P = int(J['mask'].sum(1).max()); m = J['mask'][:, :P] > 0; cnt = m.sum(1)
    wrap = np.arange(P)[None, :] % np.maximum(cnt, 1)[:, None]                  # padded slot -> a real particle of the same jet
    fill = lambda a: np.where(m, a[:, :P], np.take_along_axis(a[:, :P], wrap, 1))
    feats = []
    for k, c, s, cl in FEATURES[n]:
        x = raw_features(J, k)
        if c is not None: x = np.clip((x - np.float32(c)) * np.float32(s), *cl)
        feats.append(fill(np.where(J['mask'] > 0, x, 0)))
    vec = np.stack([fill(J[k]) for k in ('px', 'py', 'pz', 'energy')], 1)
    return np.stack(feats, 1).astype(np.float32), vec.astype(np.float32), m[:, None, :].astype(np.float32)


class ParTNetwork:
    def __init__(self, n, untrained=False, seed=0):
        import torch
        from weaver.nn.model.ParticleTransformer import ParticleTransformer
        self.n = n; self.torch = torch
        self.model = ParticleTransformer(input_dim=len(FEATURES[n]), **CFG)
        sd = torch.load(MODELS / f'ParT_{n}.pt', map_location='cpu'); sd = {k[4:] if k.startswith('mod.') else k: v for k, v in sd.items()}
        if untrained:          # the control: every weight redrawn from a Gaussian with the trained tensor's mean and spread
            g = torch.Generator().manual_seed(seed)
            sd = {k: (torch.randn(v.shape, generator=g) * v.float().std() + v.float().mean()).to(v.dtype) if v.is_floating_point() and v.numel() > 1 and not k.endswith(('running_mean', 'running_var')) else v
                  for k, v in sd.items()}
        self.model.load_state_dict(sd, strict=True); self.model.eval()
        self.device = os.environ.get('JETDISTILL_DEVICE', 'cpu')           # 'mps' (Apple GPU) or 'cuda' for faster inference
        self.model.to(self.device)
        W = self.model.fc[0].weight.detach().cpu().double().numpy(); b = self.model.fc[0].bias.detach().cpu().double().numpy()
        self.last = (W.T.copy(), b.copy(), None, None)
        self._cls = {}
        self.model.fc.register_forward_hook(lambda mod, inp, out: self._cls.__setitem__('h', inp[0].detach()))

    def run(self, J, batch=256, log=None):
        """J: jet dict of data.read_root -> dict(z, h, logits); jets are batched by multiplicity (fewer padded slots)"""
        torch = self.torch; N = len(J['mask']); order = np.argsort(J['mask'].sum(1), kind='stable')
        Z = np.zeros((N, 128), np.float32); L = np.zeros((N, 10), np.float32)
        with torch.no_grad():
            for i in range(0, N, batch):
                idx = order[i:i + batch]; x, v, m = inputs({k: a[idx] for k, a in J.items()}, self.n)
                dev = lambda a: torch.from_numpy(a).to(self.device)
                out = self.model(dev(x), dev(v), dev(m))
                L[idx] = out.cpu().numpy(); Z[idx] = self._cls['h'].cpu().numpy()
                if log and (i // batch) % 200 == 0: log(f'  network {self.n}: {i + len(idx)} / {N} jets')
        return dict(z=Z.astype(np.float64), h=Z.astype(np.float64), logits=L.astype(np.float64))

    def run_internals(self, J, batch=256, log=None):
        """run() plus the input of the class-attention blocks: dict(z, logits, x (J, 128, 128) float16 — each particle's
        embedding after the 8 particle-attention blocks, in the jet dict's particle order, 0 in empty slots — mask (J, 128))"""
        torch = self.torch; N = len(J['mask']); order = np.argsort(J['mask'].sum(1), kind='stable'); P = J['mask'].shape[1]
        X = np.zeros((N, P, 128), np.float16); Z = np.zeros((N, 128), np.float32); L = np.zeros((N, 10), np.float32); cap = {}
        hk = self.model.cls_blocks[0].register_forward_pre_hook(lambda mod, args, kwargs: cap.__setitem__('x', args[0].detach()), with_kwargs=True)
        try:
            with torch.no_grad():
                for i in range(0, N, batch):
                    idx = order[i:i + batch]; x, v, m = inputs({k: a[idx] for k, a in J.items()}, self.n)
                    dev = lambda a: torch.from_numpy(a).to(self.device)
                    out = self.model(dev(x), dev(v), dev(m)); L[idx] = out.cpu().numpy(); Z[idx] = self._cls['h'].cpu().numpy()
                    xe = cap['x'].permute(1, 0, 2).cpu().numpy()            # (batch, P_trimmed, 128)
                    X[idx, :xe.shape[1]] = xe.astype(np.float16)
                    if log and (i // batch) % 200 == 0: log(f'  network {self.n} (internals): {i + len(idx)} / {N} jets')
        finally:
            hk.remove()
        X *= (J['mask'] > 0)[..., None]
        return dict(z=Z.astype(np.float64), logits=L.astype(np.float64), x=X, mask=J['mask'] > 0)

    def logits_from_h(self, H):
        return np.asarray(H, np.float64) @ self.last[0] + self.last[1]
