"""H39: weaver network config that seeds torch before building the authors' example network (weaver-core 0.4.17 has no --seed).
Env: H39_BASE_NET = path of the authors' network file (ext/particle_transformer/networks/example_*.py), H39_SEED = seed."""
import importlib.util
import os
import torch

_spec = importlib.util.spec_from_file_location('_h39_base_net', os.environ['H39_BASE_NET'])
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


def get_model(data_config, **kwargs):
    torch.manual_seed(int(os.environ.get('H39_SEED', '1')))
    return _base.get_model(data_config, **kwargs)


def get_loss(data_config, **kwargs):
    return _base.get_loss(data_config, **kwargs)
