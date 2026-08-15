# Make heavy / optional HDO baselines lazy so HCBO can run without their
# transitive dependencies (e.g. ax-platform for ALEBO). When `run.py` is
# invoked with ``--HCBO_only``, none of these are actually called, but they
# would otherwise fail at import time.
def _make_missing(name, err):
    def _missing(*args, **kwargs):
        raise ImportError(
            f"Optional HDO baseline '{name}' is unavailable in this environment: {err}"
        )
    return _missing


try:
    from .REMBO.rembo import optimize_with_REMBO
except Exception as _e:
    optimize_with_REMBO = _make_missing("optimize_with_REMBO", _e)

try:
    from .MCTSVS.mcts_vs import optimize_with_MCTSVS
except Exception as _e:
    optimize_with_MCTSVS = _make_missing("optimize_with_MCTSVS", _e)

try:
    from .ALEBO.alebo import optimize_with_ALEBO
except Exception as _e:
    optimize_with_ALEBO = _make_missing("optimize_with_ALEBO", _e)

try:
    from .TuRBO.run import optimize_with_TuRBO
except Exception as _e:
    optimize_with_TuRBO = _make_missing("optimize_with_TuRBO", _e)

try:
    from .dropoutBO.run_drop import optimize_with_DropoutBO
except Exception as _e:
    optimize_with_DropoutBO = _make_missing("optimize_with_DropoutBO", _e)

try:
    from .core.CMAES import optimize_with_CMAES
except Exception as _e:
    optimize_with_CMAES = _make_missing("optimize_with_CMAES", _e)

try:
    from .optimization_problem import OptimizationProblem
except Exception as _e:
    OptimizationProblem = _make_missing("OptimizationProblem", _e)

try:
    from .test_optimization_problem import get_Hartmann6_problem, get_Ackly_problem
except Exception as _e:
    get_Hartmann6_problem = _make_missing("get_Hartmann6_problem", _e)
    get_Ackly_problem = _make_missing("get_Ackly_problem", _e)

try:
    from .BO.bo import optimize_with_BO
except Exception as _e:
    optimize_with_BO = _make_missing("optimize_with_BO", _e)
