import torch
from botorch.fit import fit_gpytorch_mll as fit_gpytorch_model
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood


def fit_gp_model(X, Y, Yvar=None):
    if Y.ndim == 1:
        Y = Y.unsqueeze(dim=-1)
    model = SingleTaskGP(
        X, Y, outcome_transform=Standardize(m=Y.shape[-1])
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model
