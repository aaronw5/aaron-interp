"""P11 check: are the P10 seeds distinct networks? Pearson correlation of flattened weight tensors between every pair of
checkpoints (released, s0, s1, s2; epoch 0 and best) per feature set. Distinct random initialisations give |r| near 0 for
weight matrices whose rows/heads are permutation-symmetric; a shared initialisation gives r well above 0.
  AI_PY=conda ./job.sh c-p11-wcheck 'python -m interp.p11_weight_check'"""
import itertools, torch
S = '/j-jepa-vol/aaron-interp/models/part_seeds'; M = '/j-jepa-vol/aaron-interp/models/part'
KEYS = ['embed.embed.1.weight', 'pair_embed.embed.1.weight', 'blocks.0.attn.in_proj_weight', 'cls_blocks.0.attn.in_proj_weight', 'fc.0.weight']
ld = lambda p: {k[4:] if k.startswith('mod.') else k: v.float() for k, v in torch.load(p, map_location='cpu').items()}
for feat in ('full', 'kin'):
    C = {'released': ld(f'{M}/ParT_{feat}.pt')}
    for s in (0, 1, 2):
        C[f's{s}_e0'] = ld(f'{S}/{feat}_s{s}/net_epoch-0_state.pt'); C[f's{s}'] = ld(f'{S}/{feat}_s{s}/net_best_epoch_state.pt')
    keys = [k for k in KEYS if k in C['released']] or list(C['released'])[:5]
    print(feat, 'keys', keys, flush=True)
    for a, b in itertools.combinations(C, 2):
        r = []
        for k in keys:
            x, y = C[a][k].flatten(), C[b][k].flatten(); r.append(float(torch.corrcoef(torch.stack([x, y]))[0, 1]))
        print(f'  {feat} {a:9s} vs {b:9s}: r = ' + ', '.join(f'{v:+.3f}' for v in r), flush=True)
