#!/bin/bash
# B6w setup (cluster job): venv /j-jepa-vol/aaron-interp/envs/symreg from the jax job env's python (no system site packages; the jax env's
# site-packages added by a .pth file), pip install gplearn (+ a smoke fit with the sklearn>=1.6 validate_data shim used by
# interp.b6w_symreg), then try PySR with a Julia depot under /j-jepa-vol/aaron-interp/envs/julia (juliacall downloads Julia
# itself; juliaup not needed). Writes /j-jepa-vol/aaron-interp/envs/symreg/STATUS with one line per component.
set -u
E=/j-jepa-vol/aaron-interp/envs; V=$E/symreg; mkdir -p $E
export PIP_CACHE_DIR=/tmp/pipcache JULIA_DEPOT_PATH=$E/julia/depot PYTHON_JULIAPKG_PROJECT=$E/julia/project JULIA_NUM_THREADS=1
echo "base python: $(which python) $(python -V 2>&1)"
[ -x $V/bin/python ] || python -m venv $V
# the jax env is itself a venv over /opt/conda: --system-site-packages would expose /opt/conda (old scikit-learn, a pyarrow built
# for another numpy), so the venv has no system packages and sees the jax env's site-packages (scikit-learn 1.7, scipy, pandas,
# jax) through a .pth file
echo $(python -c 'import site; print(site.getsitepackages()[0])') > $V/lib/python3.10/site-packages/zz_jaxenv.pth
$V/bin/python -m pip install -q --upgrade pip 2>&1 | tail -2
$V/bin/python -m pip install -q gplearn 2>&1 | tail -3
: > $V/STATUS
$V/bin/python - <<'PY' && echo "gplearn ok" >> $V/STATUS || echo "gplearn FAILED" >> $V/STATUS
import numpy as np, sklearn, gplearn
from gplearn.genetic import SymbolicRegressor
if not hasattr(SymbolicRegressor, '_validate_data'):
    from sklearn.utils.validation import validate_data
    SymbolicRegressor._validate_data = lambda self, *a, **k: validate_data(self, *a, **k)
X = np.random.default_rng(0).normal(size=(500, 3)); y = X[:, 0] * X[:, 1] + 2 * X[:, 2]
m = SymbolicRegressor(population_size=200, generations=5, random_state=0).fit(X, y)
print('sklearn', sklearn.__version__, 'gplearn', gplearn.__version__, m._program, 'R2', m.score(X, y))
PY
mkdir -p $E/julia
timeout 2400 bash -c "$V/bin/python -m pip install -q pysr 2>&1 | tail -3; $V/bin/python - <<'PY'
import time, numpy as np
t = time.time()
from pysr import PySRRegressor
X = np.random.default_rng(0).normal(size=(300, 3)); y = X[:, 0] * X[:, 1] + 2 * X[:, 2]
m = PySRRegressor(niterations=3, populations=4, binary_operators=['+', '*', '-'], unary_operators=[], procs=0, multithreading=False,
                  progress=False, verbosity=0, deterministic=True, random_state=0, temp_equation_file=True)
m.fit(X, y); print('pysr ok', m.sympy(), f'{time.time() - t:.0f} s')
PY" && echo "pysr ok" >> $V/STATUS || echo "pysr FAILED (see log)" >> $V/STATUS
cat $V/STATUS; du -sh $E/* 2>/dev/null
