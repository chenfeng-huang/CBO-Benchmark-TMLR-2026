"""Per-dataset pyro `FunctionalCausalModel` builders for the CBO_Benchmark suite.

Each builder returns ``(fcm, optimization_domain, intervention_vars, target, task)``.

The FCMs are hand-translated from ``CBO_Benchmark/sem/<dataset>_sem_equations.json``
(and cross-checked with ``CBO_Benchmark/dataset_generators/generate_<dataset>_*.py``).
We do not auto-parse the SEM JSON because several datasets encode their non-linear
forms only in the free-text ``function`` field (e.g. ``exp(-X)``, ``cos(Z)``,
``sin(-L + 2*R)``); hand-translation keeps the math obvious and reviewable.

Empty contextual sets are used because none of the CBO_Benchmark configs declare
contextual variables. With an empty contextual set, POMPSExperiment effectively
runs UCB-MAB over the action-relevant intervention scopes (POMIS-style), with
HEBO as the inner BO subroutine.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

import pyro
import pyro.distributions as dist
import torch
import yaml

from pomis.scm import Domain, RealDomain
from pomps.fcm import FunctionalCausalModel, Functor


# CoCaBO's SharedFunctor returns Python floats (popped from a pandas dict),
# not tensors -- so any torch ops in our SEM functors need to coerce inputs
# first. We expose tiny aliases to keep the SEM lambdas readable.
_t = lambda v: torch.as_tensor(v, dtype=torch.float32)
_exp = lambda v: torch.exp(_t(v))
_cos = lambda v: torch.cos(_t(v))
_sin = lambda v: torch.sin(_t(v))
_sig = lambda v: torch.sigmoid(_t(v))
_abs = lambda v: torch.abs(_t(v))


@dataclass
class FCMBundle:
    fcm: FunctionalCausalModel
    domain: tp.List[Domain]
    intervention_vars: tp.Set[str]
    target: str
    task: str


def _domain_from_config(config: dict) -> tp.List[RealDomain]:
    out = []
    for var, (lo, hi) in config["interventional_domain"].items():
        out.append(RealDomain(var, float(lo), float(hi)))
    return out


def _no_latents() -> tp.Dict[str, tp.Any]:
    return {}


# ---------------------------------------------------------------------------
# chain
# ---------------------------------------------------------------------------
def _build_chain(config: dict) -> FCMBundle:
    """SEM:
        X ~ N(0,1),  W ~ N(0,1)
        Z = -0.5*X + N(0,1)
        Y = -W - 3*Z*X + N(0,1)
    """
    functors = {
        Functor(lambda: pyro.sample("X", dist.Normal(0.0, 1.0)), "X"),
        Functor(lambda: pyro.sample("W", dist.Normal(0.0, 1.0)), "W"),
        Functor(
            lambda X: pyro.sample("Z", dist.Normal(-0.5 * X, 1.0)), "Z"
        ),
        Functor(
            lambda W, X, Z: pyro.sample(
                "Y", dist.Normal(-W - 3.0 * Z * X, 1.0)
            ),
            "Y",
        ),
    }
    fcm = FunctionalCausalModel(functors, _no_latents)
    return FCMBundle(fcm, _domain_from_config(config),
                     set(config["intervention"]), config["target"], config["task"])


# ---------------------------------------------------------------------------
# ecology  (linear SCM with normal exogenous)
# ---------------------------------------------------------------------------
def _build_ecology(config: dict) -> FCMBundle:
    functors = {
        # Exogenous
        Functor(lambda: pyro.sample("E", dist.Normal(24.184130434782606,
                                                     3.2204054790158585)), "E"),
        Functor(lambda: pyro.sample("S", dist.Normal(36.591623913043485,
                                                     0.14919708693132147)), "S"),
        Functor(lambda: pyro.sample("N", dist.Normal(0.4920652173913041,
                                                     1.5924077981635778)), "N"),
        Functor(lambda: pyro.sample("T", dist.Normal(2357.893695652174,
                                                     27.609355287148222)), "T"),
        # Endogenous (linear)
        Functor(lambda E: pyro.sample(
            "X", dist.Normal(18.79817390034509 + 15.797383660309157 * E,
                             28.896638707243493)), "X"),
        Functor(lambda N: pyro.sample(
            "C", dist.Normal(0.37342021355081695 - 0.0024002572713753005 * N,
                             0.039295137153226196)), "C"),
        Functor(lambda C: pyro.sample(
            "L", dist.Normal(6665.081995591907 - 10737.462582329481 * C,
                             1546.9130344410867)), "L"),
        Functor(lambda X: pyro.sample(
            "P", dist.Normal(8.427705924233592 - 0.0009655314986363629 * X,
                             0.005258257571036228)), "P"),
        Functor(lambda X: pyro.sample(
            "D", dist.Normal(2131.6721072369096 - 0.2165604666734924 * X,
                             18.412099847093625)), "D"),
        Functor(lambda E, S, X: pyro.sample(
            "O", dist.Normal(3.2452480541052733 + 0.09433222790709161 * E
                             + 0.006753920720864577 * S
                             - 0.005737087151230129 * X,
                             0.03695797753398314)), "O"),
        # Note: T has no children in the ecology SEM; we still wire a 0*T term
        # into Y so that T appears as a node in the induced DiGraph (CoCaBO's
        # MPSReductor.simplify expects every interventional var to be present
        # in the graph). The 0 coefficient preserves the original SEM exactly.
        Functor(lambda L, N, P, O, T: pyro.sample(
            "Y", dist.Normal(211.4225551648116
                             - 3.012576882356712e-05 * L
                             + 0.01667959146973974 * N
                             - 25.71927722049691 * P
                             - 0.5004034826468742 * O
                             + 0.0 * T,
                             1.0733401618242278)), "Y"),
    }
    fcm = FunctionalCausalModel(functors, _no_latents)
    return FCMBundle(fcm, _domain_from_config(config),
                     set(config["intervention"]), config["target"], config["task"])


# ---------------------------------------------------------------------------
# epidemiology
# ---------------------------------------------------------------------------
def _build_epidemiology(config: dict) -> FCMBundle:
    """SEM:
        B ~ U(-1, 1),  T ~ U(4, 8)
        L = exp(0.5*T + N(0,1))
        R = 4 + L*T
        Y = 0.5 + cos(4*T) + sin(-L + 2*R) + B + N(0,1)
    """
    functors = {
        Functor(lambda: pyro.sample("B", dist.Uniform(-1.0, 1.0)), "B"),
        Functor(lambda: pyro.sample("T", dist.Uniform(4.0, 8.0)), "T"),
        Functor(
            lambda T: pyro.sample(
                "L",
                dist.LogNormal(0.5 * T, 1.0),  # exp(N(0.5T, 1)) = LogNormal
            ),
            "L",
        ),
        Functor(
            lambda L, T: pyro.sample(
                "R", dist.Delta(4.0 + L * T)
            ),
            "R",
        ),
        Functor(
            lambda T, L, R, B: pyro.sample(
                "Y",
                dist.Normal(
                    0.5 + _cos(4.0 * T) + _sin(-L + 2.0 * R) + B,
                    1.0,
                ),
            ),
            "Y",
        ),
    }
    fcm = FunctionalCausalModel(functors, _no_latents)
    return FCMBundle(fcm, _domain_from_config(config),
                     set(config["intervention"]), config["target"], config["task"])


# ---------------------------------------------------------------------------
# healthcare
# ---------------------------------------------------------------------------
def _build_healthcare(config: dict) -> FCMBundle:
    """SEM (sigmoid relations have noise_std = 0):
        Age ~ U(55, 75)
        BMI = 27 - 0.01*Age + N(0, 0.7)
        Aspirin = sigmoid(-8 + 0.1*Age + 0.03*BMI)
        Statin  = sigmoid(-13 + 0.1*Age + 0.2*BMI)
        cancer  = sigmoid(2.2 - 0.05*Age + 0.01*BMI - 0.04*Statin + 0.02*Aspirin)
        PSA = 6.8 + 0.04*Age - 0.15*BMI - 0.6*Statin + 0.55*Aspirin + cancer + N(0, 0.4)
    """
    functors = {
        Functor(lambda: pyro.sample("Age", dist.Uniform(55.0, 75.0)), "Age"),
        Functor(lambda Age: pyro.sample(
            "BMI", dist.Normal(27.0 - 0.01 * Age, 0.7)), "BMI"),
        Functor(lambda Age, BMI: pyro.sample(
            "Aspirin", dist.Delta(_sig(-8.0 + 0.1 * Age + 0.03 * BMI))),
            "Aspirin"),
        Functor(lambda Age, BMI: pyro.sample(
            "Statin", dist.Delta(_sig(-13.0 + 0.1 * Age + 0.2 * BMI))),
            "Statin"),
        Functor(lambda Age, BMI, Statin, Aspirin: pyro.sample(
            "cancer", dist.Delta(_sig(2.2 - 0.05 * Age + 0.01 * BMI
                                      - 0.04 * Statin + 0.02 * Aspirin))),
            "cancer"),
        Functor(lambda Age, BMI, Statin, Aspirin, cancer: pyro.sample(
            "PSA", dist.Normal(6.8 + 0.04 * Age - 0.15 * BMI
                               - 0.6 * Statin + 0.55 * Aspirin + cancer,
                               0.4)),
            "PSA"),
    }
    fcm = FunctionalCausalModel(functors, _no_latents)
    return FCMBundle(fcm, _domain_from_config(config),
                     set(config["intervention"]), config["target"], config["task"])


# ---------------------------------------------------------------------------
# protein  (purely linear SCM, fitted from real data)
# ---------------------------------------------------------------------------
def _build_protein(config: dict) -> FCMBundle:
    functors = {
        Functor(lambda: pyro.sample("PKC", dist.Uniform(1.0, 106.0)), "PKC"),
        Functor(lambda PKC: pyro.sample(
            "PKA", dist.Normal(554.3907305751273 + 0.8411525573301047 * PKC,
                               427.43799602174334)), "PKA"),
        Functor(lambda PKC, PKA: pyro.sample(
            "Raf", dist.Normal(62.199045961323414
                               - 0.17774509086121948 * PKC
                               - 0.0003792075216956362 * PKA,
                               41.76878172507802)), "Raf"),
        Functor(lambda PKC, PKA, Raf: pyro.sample(
            "Mek", dist.Normal(-1.0902746220577555
                               + 0.03966245723060245 * PKC
                               - 0.0006520266851554879 * PKA
                               + 0.5208447051064984 * Raf,
                               16.695865110803133)), "Mek"),
        Functor(lambda PKC, PKA: pyro.sample(
            "P38", dist.Normal(15.144327731160235
                               + 1.2347832903144826 * PKC
                               + 0.000590540235751414 * PKA,
                               13.111163757152335)), "P38"),
        Functor(lambda PKC, PKA: pyro.sample(
            "Jnk", dist.Normal(52.9536028509298
                               - 0.7648011883501407 * PKC
                               - 0.005306096821872689 * PKA,
                               42.074714667865486)), "Jnk"),
        Functor(lambda PKA: pyro.sample(
            "Akt", dist.Normal(-31.110747106386974
                               + 0.12890532877586242 * PKA,
                               113.95850408157932)), "Akt"),
        Functor(lambda PKA, Mek: pyro.sample(
            "Erk", dist.Normal(-23.248743333459377
                               + 0.08170673252473967 * PKA
                               - 0.029885814321525 * Mek,
                               82.76019809593954)), "Erk"),
    }
    fcm = FunctionalCausalModel(functors, _no_latents)
    return FCMBundle(fcm, _domain_from_config(config),
                     set(config["intervention"]), config["target"], config["task"])


# ---------------------------------------------------------------------------
# synthetic  (with two latent confounders U1, U2)
# ---------------------------------------------------------------------------
def _build_synthetic(config: dict) -> FCMBundle:
    """SEM (with confounders U1, U2):
        F ~ N(0, 1)
        A = F^2 + U1
        B = U2
        C = exp(-B)
        D = exp(-C) / 10
        E = cos(A) + C/10
        Y = cos(D) + sin(E) + U1 + U2 * eps_y,  eps_y ~ N(0, 0.1)
        (eps_y std matches sem/synthetic_sem_equations.json noise_std = 0.1)
    """
    def latents():
        # Standard-normal confounders; unspecified in the JSON, so we use N(0,1)
        # to match the conventional choice for "unobserved" in CBO benchmarks.
        return {
            "U1": pyro.sample("U1", dist.Normal(0.0, 1.0)),
            "U2": pyro.sample("U2", dist.Normal(0.0, 1.0)),
        }

    functors = {
        Functor(lambda: pyro.sample("F", dist.Normal(0.0, 1.0)), "F"),
        Functor(lambda F, U1: pyro.sample("A", dist.Delta(_t(F * F) + U1)), "A"),
        Functor(lambda U2: pyro.sample("B", dist.Delta(U2)), "B"),
        Functor(lambda B: pyro.sample("C", dist.Delta(_exp(-_t(B)))), "C"),
        Functor(lambda C: pyro.sample("D", dist.Delta(_exp(-_t(C)) / 10.0)), "D"),
        Functor(lambda A, C: pyro.sample(
            "E", dist.Delta(_cos(A) + _t(C) / 10.0)), "E"),
        Functor(lambda D, E, U1, U2: pyro.sample(
            "Y", dist.Normal(_cos(D) + _sin(E) + U1, _abs(U2) * 0.1 + 1e-6)),
                "Y"),
    }
    fcm = FunctionalCausalModel(functors, latents)
    return FCMBundle(fcm, _domain_from_config(config),
                     set(config["intervention"]), config["target"], config["task"])


# ---------------------------------------------------------------------------
# synthetic_2
# ---------------------------------------------------------------------------
def _build_synthetic_2(config: dict) -> FCMBundle:
    """SEM:
        X ~ N(0,1)
        Z = exp(-X) + N(0,1)
        Y = cos(Z) - exp(-Z/20) + N(0,1)
    """
    functors = {
        Functor(lambda: pyro.sample("X", dist.Normal(0.0, 1.0)), "X"),
        Functor(lambda X: pyro.sample(
            "Z", dist.Normal(_exp(-_t(X)), 1.0)), "Z"),
        Functor(lambda Z: pyro.sample(
            "Y", dist.Normal(_cos(Z) - _exp(-_t(Z) / 20.0), 1.0)), "Y"),
    }
    fcm = FunctionalCausalModel(functors, _no_latents)
    return FCMBundle(fcm, _domain_from_config(config),
                     set(config["intervention"]), config["target"], config["task"])


# ---------------------------------------------------------------------------
# toyGraph  (deterministic chain)
# ---------------------------------------------------------------------------
def _build_toy_graph(config: dict) -> FCMBundle:
    """SEM (deterministic, noise_std = 0):
        X ~ Delta(0)               (acts as exogenous root; intervention overrides it)
        Z = exp(-X)
        Y = cos(Z) - exp(-Z/20)
    """
    # Tiny noise so HEBO's surrogate doesn't see a perfectly noiseless target.
    eps = 1e-4
    functors = {
        Functor(lambda: pyro.sample("X", dist.Normal(0.0, 1.0)), "X"),
        Functor(lambda X: pyro.sample(
            "Z", dist.Normal(_exp(-_t(X)), eps)), "Z"),
        Functor(lambda Z: pyro.sample(
            "Y", dist.Normal(_cos(Z) - _exp(-_t(Z) / 20.0), eps)), "Y"),
    }
    fcm = FunctionalCausalModel(functors, _no_latents)
    return FCMBundle(fcm, _domain_from_config(config),
                     set(config["intervention"]), config["target"], config["task"])


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
_BUILDERS: tp.Dict[str, tp.Callable[[dict], FCMBundle]] = {
    "chain": _build_chain,
    "ecology": _build_ecology,
    "epidemiology": _build_epidemiology,
    "healthcare": _build_healthcare,
    "protein": _build_protein,
    "synthetic": _build_synthetic,
    "synthetic_2": _build_synthetic_2,
    "toyGraph": _build_toy_graph,
}


def build_for_dataset(dataset: str, config_path: str = None) -> FCMBundle:
    if dataset not in _BUILDERS:
        raise KeyError(
            f"No CoCaBO FCM builder registered for dataset '{dataset}'. "
            f"Known datasets: {sorted(_BUILDERS.keys())}"
        )
    config_path = config_path or f"configs/{dataset}.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return _BUILDERS[dataset](config)


def supported_datasets() -> tp.List[str]:
    return sorted(_BUILDERS.keys())
