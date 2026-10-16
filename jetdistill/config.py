"""Paths and the setups (experiments).

Paths (environment variables, defaults relative to the working directory):
  JETDISTILL_DATA     data/     the dataset files made by `python -m jetdistill.data` (see README)
  JETDISTILL_MODELS   models/   the official JEDI-linear models (github.com/calad0i/JEDI-linear, official_models/)
  JETDISTILL_RESULTS  results/  everything the pipeline writes: results/<setup>/n<N>/...
"""
import os
from pathlib import Path

DATA = Path(os.environ.get('JETDISTILL_DATA', 'data'))
MODELS = Path(os.environ.get('JETDISTILL_MODELS', 'models'))
RESULTS = Path(os.environ.get('JETDISTILL_RESULTS', 'results'))
CLASSES = ['g', 'q', 'W', 'Z', 't']

# jets used by each step (all from the TRAIN archive except the test jets, which come from the VAL archive)
N_STEP1_FIT, N_TUNE_FIT, N_DEV, N_TEST_SPLIT = 40000, 100000, 25000, 50000
N_STEP4_FIT, N_EXPLAIN, N_FULL_TEST = 150000, 60000, None          # None: the whole test file
SMOKE = os.environ.get('JETDISTILL_SMOKE') == '1'                    # a quick run of every stage on a few thousand jets (tests)
if SMOKE:
    N_STEP1_FIT, N_TUNE_FIT, N_DEV, N_TEST_SPLIT, N_STEP4_FIT, N_EXPLAIN, N_FULL_TEST = 3000, 3000, 2000, 2000, 3000, 3000, 4000

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
