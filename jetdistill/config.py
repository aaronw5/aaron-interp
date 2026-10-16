"""Paths, the tagger and the setups (experiments).

Tagger (environment variable JETDISTILL_TAGGER; unset = jedi, nothing below changes for JEDI):
  jedi  (default)  JEDI-linear on the hls4ml jet dataset: networks 8 and 64 (particles), 5 classes, 16 ReLU neurons
  part             Particle Transformer (ParT) on JetClass (jetdistill/part/, copied from distill-part/jld-features): networks
                   'kin' and 'full', 10 classes, the 128 numbers of its class token after the last LayerNorm, a linear last layer
Paths (environment variables, defaults relative to the working directory):
  JETDISTILL_DATA     data/ (jedi), data_part/ (part)        the dataset files made by `python -m jetdistill.data` (see README)
  JETDISTILL_MODELS   models/ (jedi), models_part/ (part)    the official models (JEDI: github.com/calad0i/JEDI-linear, official_models/)
  JETDISTILL_RESULTS  results/ (jedi), results_part/ (part)  everything the pipeline writes: results/<setup>/n<N>/...
"""
import os
from pathlib import Path

TAGGER = os.environ.get('JETDISTILL_TAGGER', 'jedi')
assert TAGGER in ('jedi', 'part'), TAGGER
_sfx = '' if TAGGER == 'jedi' else '_part'
DATA = Path(os.environ.get('JETDISTILL_DATA', 'data' + _sfx))
MODELS = Path(os.environ.get('JETDISTILL_MODELS', 'models' + _sfx))
RESULTS = Path(os.environ.get('JETDISTILL_RESULTS', 'results' + _sfx))
SMOKE = os.environ.get('JETDISTILL_SMOKE') == '1'                    # a quick run of every stage on a few thousand jets (tests)

if TAGGER == 'jedi':
    CLASSES = ['g', 'q', 'W', 'Z', 't']
    NETWORKS = (8, 64)
    RELU = True                  # the neurons are max(0, z); the last layer rounds them to a fixed-point grid
    # jets used by each step (all from the TRAIN archive except the test jets, which come from the VAL archive)
    N_STEP1_FIT, N_TUNE_FIT, N_DEV, N_TEST_SPLIT = 40000, 100000, 25000, 50000
    N_STEP4_FIT, N_EXPLAIN, N_FULL_TEST = 150000, 60000, None          # None: the whole test file
else:
    CLASSES = ['QCD', 'Hbb', 'Hcc', 'Hgg', 'H4q', 'Hqql', 'Zqq', 'Wqq', 'Tbqq', 'Tbl']
    NETWORKS = ('kin', 'full')
    RELU = False                 # the neurons are the LayerNorm outputs themselves; the last layer is linear
    # jets used by each step (training / validation jets from JetClass's training-side files, test jets from its test set)
    N_STEP1_FIT, N_TUNE_FIT, N_DEV, N_TEST_SPLIT = 30000, 50000, 20000, 25000
    N_STEP4_FIT, N_EXPLAIN, N_FULL_TEST = 80000, 40000, None
if SMOKE:
    N_STEP1_FIT, N_TUNE_FIT, N_DEV, N_TEST_SPLIT, N_STEP4_FIT, N_EXPLAIN, N_FULL_TEST = 3000, 3000, 2000, 2000, 3000, 3000, 4000
NC = len(CLASSES)


def net_dir(n):
    """the folder name of a network: n8, n64 (jedi); kin, full (part)"""
    return f'n{n}' if isinstance(n, int) else str(n)


def parse_net(s):
    return int(s) if str(s).isdigit() else str(s)


def n_particles(n):
    """particles a network sees (and the observables are computed from)"""
    return n if isinstance(n, int) else 128

MASSES = {'m_W': 80.4, 'm_Z': 91.19, 'm_H': 125.1, 'm_t': 172.8}   # offered as candidate thresholds of mass observables
MASS_MULT = (.5, 1., 1.5, 2.)


class Setup(dict):
    __getattr__ = dict.get


SETUPS = {
    'all': Setup(label='All observables', page_desc='all jet quantities, including the jet mass and masses of particle subsets; candidate thresholds: the 5–95% quantiles of each quantity, and for mass quantities also 0.5, 1, 1.5 and 2 × m_W, m_Z, m_H, m_t',
                 masses_as_thresholds=True),
    'nophys': Setup(label='No W, Z or top mass as a candidate threshold', page_desc='all jet quantities, including the jet mass; candidate thresholds: only the 5–95% quantiles of each quantity (no W, Z, H or top mass values offered)',
                    masses_as_thresholds=False),
    'nomass': Setup(label='No mass observables', page_desc='no mass quantities (jet mass, masses of particle subsets, subjet and soft-drop masses, m/pT are excluded); candidate thresholds: the 5–95% quantiles of each quantity',
                    masses_as_thresholds=False, no_mass=True),
    'nomass_strict': Setup(label='No mass observables or exact equivalents', page_desc='no mass quantities, and also no quantity equal to (m/ΣpT)² in disguise (|correlation| > 0.98: Σ zᵢzⱼΔRᵢⱼ², Σ zΔR², λ₁ + λ₂, …); candidate thresholds: the 5–95% quantiles of each quantity',
                           masses_as_thresholds=False, no_mass=True, strict=True),
    'all_agree': Setup(label='All observables, tuned for agreement', page_desc='as “All observables”; step 2 is tuned toward the network’s decision for each jet (its class) instead of its probabilities',
                       masses_as_thresholds=True, step1_from='all', target='decisions', truth_family=False),
}
FIDELITY_LAMBDA = 0.01    # weight of the neuron-closeness term R in steps 2 and 3
