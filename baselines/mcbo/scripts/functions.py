r"""
Defines the environments that BO agents take actions in. 

References:

.. [astudilloBayesianOptimizationFunction2021]
    Astudillo, Raul, and Peter Frazier. "Bayesian optimization of function networks." 
    Advances in Neural Information Processing Systems 34 (2021): 14463-14475.
.. [agliettiCausalBayesianOptimization2020]
    Aglietti, Virginia, et al. "Causal bayesian optimization." International Conference
    on Artificial Intelligence and Statistics. PMLR, 2020.
"""

import torch
import math
import numpy as np

from mcbo.utils.dag import DAG, FNActionInput
from mcbo.utils import functions_utils
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal

class Env:
    """
    Abstract class for environments.
    """

    def evaluate(self, X):
        r"""
        For action input X returns the observation of all network nodes.
        """
        raise NotImplementedError

    def check_input(self, X):
        """
        Checks if the input to evaluate matches the correct input dim. 
        """
        if X.shape[-1] != self.input_dim:
            raise ValueError("Input dimension does not fit for this function")

    def get_causal_quantities(self):
        """
        Outputs a dict of environment details that are computed differently for 
        CausalEnv and FNEnv. 
        """
        raise NotImplementedError

    def get_base_profile(self):
        """
        Outputs a dict of environment details that can be computed similarly for all
        types of environments. 
        """
        return {
            "additive_noise_dists": self.additive_noise_dists,
            "interventional": self.interventional,
            "dag": self.dag,
            "input_dim": self.input_dim,
        }

    def get_env_profile(self):
        """
        Outputs a dictionary containing all details of the environment needed for 
        BO experiments. 
        """
        return {**self.get_base_profile(), **self.get_causal_quantities()}


class FNEnv(Env):
    """
    Abstract class for function network environments. All function network environments
    implemented as subclasses first appeared in 
    [astudilloBayesianOptimizationFunction2021].
    """

    def __init__(self, dag: DAG, action_input: FNActionInput):
        self.dag = dag
        self.input_dim = action_input.get_input_dim()
        self.action_input = action_input
        self.interventional = False

    def get_causal_quantities(self):
        # no targets because in FNEnv actions are added nodes, not do-interventions.
        valid_targets = [torch.zeros(self.input_dim)]
        do_map = None
        active_input_indices = self.action_input.get_active_input_indices()
        return {
            "valid_targets": valid_targets,
            "do_map": do_map,
            "active_input_indices": active_input_indices,
        }


class CausalEnv(Env):
    """
    Abstract class for environments based upon do() interventions. All CausalEnvs 
    implemented as subclasses first appeared in 
    [agliettiCausalBayesianOptimization2020].
    """

    def __init__(self, dag: DAG):
        self.dag = dag
        # Every node has a binary (whether to intervene) and an intervention value.
        self.input_dim = dag.get_n_nodes() * 2
        self.interventional = True

    def do_int(self, unintervened_value_i, do_x_i, x_i):
        r"""
        Given a batch of unintervened_value_i, do_x_i and x_i computes a batch of 
        observed variable values. 
        Parameters
        ----------
        unintervened_value_i : Tensor
            Node value if not intervened upon.
        do_x_i : Tensor
            For each element in the batch, 1 if the node is intervened upon, 0 
            otherwise.
        x_i : Tensor
            For each element in the batch, the node value if it was intervened upon. 
        
        Return
        ------
        t : Tensor
            The batch of node values after considering if interventions were performed. 
        """
        return unintervened_value_i * (1 - do_x_i) + do_x_i * x_i

    def do_map(self, X):
        r"""Map the inputs from [0,1] to the space of intervention inputs. Required 
        because all our BO algorithms always select actions in [0,1]."""
        raise NotImplementedError

    def get_causal_quantities(self):
        # for CausalEnv we use no active input indices on any nodes
        active_input_indices = [list() for _ in range(self.input_dim)]
        valid_targets = self.valid_targets
        do_map = self.do_map
        return {
            "valid_targets": valid_targets,
            "do_map": do_map,
            "active_input_indices": active_input_indices,
        }


class Dropwave(FNEnv):
    """
    A modification of the classic Drop-Wave test function to the Function Networks
    setting.
    """
    def __init__(self, noise_scales=0.0):
        parent_nodes = [[], [0]]
        dag = DAG(parent_nodes)
        active_input_indices = [[0, 1], []]
        action_input = FNActionInput(active_input_indices)
        super(Dropwave, self).__init__(dag, action_input)
        self.additive_noise_dists = functions_utils.noise_scales_to_normals(
            noise_scales, self.dag.get_n_nodes()
        )
        # Theoretical maximum for dropwave function (approximately 1.0 at origin)
        self.theoretical_optimum = 1.0

    def evaluate(self, X):
        self.check_input(X)
        X_scaled = 10.24 * X - 5.12
        input_shape = X_scaled.shape
        output = torch.empty(input_shape[:-1] + torch.Size([self.dag.get_n_nodes()]))
        norm_X = torch.norm(X_scaled, dim=-1)
        noise_0 = self.additive_noise_dists[0].rsample(
            sample_shape=output[..., 0].shape
        )
        output[..., 0] = norm_X + noise_0
        noise_1 = self.additive_noise_dists[1].rsample(
            sample_shape=output[..., 1].shape
        )
        output[..., 1] = (1.0 + torch.cos(12.0 * norm_X)) / (
            2.0 + 0.5 * (norm_X**2)
        ) + noise_1
        return output


class Alpine2(FNEnv):
    """
    A modification of the classic Alpine test function to the Function Networks
    setting.
    """
    def __init__(self, noise_scales=0.0):
        parent_nodes = [[], [0], [1], [2], [3], [4]]
        dag = DAG(parent_nodes)
        active_input_indices = [[0], [1], [2], [3], [4], [5]]
        action_input = FNActionInput(active_input_indices)
        super(Alpine2, self).__init__(dag, action_input)
        self.additive_noise_dists = functions_utils.noise_scales_to_normals(
            noise_scales, self.dag.get_n_nodes()
        )
        self.theoretical_optimum = 400.0

    def evaluate(self, X):
        self.check_input(X)
        X_scaled = 10.0 * X
        output = torch.empty(X_scaled.shape[:-1] + torch.Size([self.dag.get_n_nodes()]))
        for i in range(self.dag.get_n_nodes()):
            x_i = X_scaled[..., i]
            if i == 0:
                noise_0 = self.additive_noise_dists[0].rsample(
                    sample_shape=output[..., 0].shape
                )
                output[..., i] = -torch.sqrt(x_i) * torch.sin(x_i) + noise_0
            else:
                noise_i = self.additive_noise_dists[i].rsample(
                    sample_shape=output[..., i].shape
                )
                output[..., i] = (
                    torch.sqrt(x_i) * torch.sin(x_i) * output[..., i - 1] + noise_i
                )
        return output


class Ackley(FNEnv):
    """
    A modification of the classic Ackley test function to the Function Networks
    setting.
    """
    def __init__(self, noise_scales=0.0):
        parent_nodes = [[], [], [0, 1]]
        dag = DAG(parent_nodes)
        active_input_indices = [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], []]
        action_input = FNActionInput(active_input_indices)
        super(Ackley, self).__init__(dag, action_input)
        self.additive_noise_dists = functions_utils.noise_scales_to_normals(
            noise_scales, self.dag.get_n_nodes()
        )
        # Maximum of the negated Ackley objective (0 at the origin)
        self.theoretical_optimum = 0.0

    def evaluate(self, X):
        self.check_input(X)
        X_scaled = 4.0 * (X - 0.5)
        output = torch.empty(X_scaled.shape[:-1] + (self.dag.get_n_nodes(),))
        noise_0 = self.additive_noise_dists[0].rsample(
            sample_shape=output[..., 0].shape
        )
        output[..., 0] = (
            1 / self.input_dim * torch.sum(torch.square(X_scaled[..., :]), dim=-1)
            + noise_0
        )
        noise_1 = self.additive_noise_dists[1].rsample(
            sample_shape=output[..., 1].shape
        )
        output[..., 1] = (
            1
            / self.input_dim
            * torch.sum(torch.cos(2 * math.pi * X_scaled[..., :]), dim=-1)
            + noise_1
        )
        noise_2 = self.additive_noise_dists[2].rsample(
            sample_shape=output[..., 2].shape
        )
        output[..., 2] = (
            20 * torch.exp(-0.2 * torch.sqrt(output[..., 0]))
            + torch.exp(output[..., 1])
            - 20
            - math.e
            + noise_2
        )
        return output


class Rosenbrock(FNEnv):
    """
    A modification of the classic Rosenbrock test function to the Function Networks
    setting.
    """
    def __init__(self, noise_scales=0.0, D=5):
        # D action variables a_0..a_{D-1} and a chain of D-1 computed nodes
        # (the benchmark default config uses D=5).
        if D < 2:
            raise ValueError("Rosenbrock requires D >= 2")
        parent_nodes = [[]] + [[i] for i in range(D - 2)]
        dag = DAG(parent_nodes)
        active_input_indices = [[i, i + 1] for i in range(D - 1)]
        action_input = FNActionInput(active_input_indices)
        super(Rosenbrock, self).__init__(dag, action_input)
        self.additive_noise_dists = functions_utils.noise_scales_to_normals(
            noise_scales, self.dag.get_n_nodes()
        )
        # Theoretical maximum for Rosenbrock function (minimum is 0, so maximum when negated is 0)
        self.theoretical_optimum = 0.0

    def evaluate(self, X):
        self.check_input(X)
        X_scaled = 4.0 * (X - 0.5)
        output = torch.empty(X_scaled.shape[:-1] + torch.Size([self.dag.get_n_nodes()]))
        noise_0 = self.additive_noise_dists[0].rsample(
            sample_shape=output[..., 0].shape
        )
        output[..., 0] = (
            -100 * torch.square(X_scaled[..., 1] - torch.square(X_scaled[..., 0]))
            - torch.square(1 - X_scaled[..., 0])
            + noise_0
        )
        for i in range(1, self.dag.get_n_nodes()):
            x_i = X_scaled[..., i]
            j = i + 1
            x_j = X_scaled[..., j]
            noise_i = self.additive_noise_dists[i].rsample(
                sample_shape=output[..., i].shape
            )
            output[..., i] = (
                -100 * torch.square(x_j - torch.square(x_i))
                - torch.square(1 - x_i)
                + output[..., i - 1]
                + noise_i
            )

        return output


class ChainSoft(FNEnv):
    r"""
    Soft-intervention Chain SCM with observed nodes X, W, Z, Y.

    Base SCM:
      X = U_X, W = U_W, Z = -0.5 X + U_Z, Y = -W - 3 Z X + U_Y.

    The function-network action is a compact parameterization of the admissible
    soft intervention family:
      action[0] selects pure functional Z intervention vs mixed W+Z,
      action[1] maps to the optional hard intervention W := w in [-1, 1],
      action[2:4] scale/phase a 10 sine + 10 cosine policy basis for Z := pi_0(X).
    """
    def __init__(self, noise_scales=0.0, grid_size=10, n_alpha=10, n_beta=10, coeff_bound=0.27):
        parent_nodes = [[], [], [0], [0, 1, 2]]
        dag = DAG(parent_nodes)
        active_input_indices = [[], [0, 1], [2, 3], []]
        action_input = FNActionInput(active_input_indices)
        super(ChainSoft, self).__init__(dag, action_input)
        self.additive_noise_dists = functions_utils.noise_scales_to_normals(
            noise_scales, self.dag.get_n_nodes()
        )
        # Fixed sin/cos policy basis with coefficients linspace'd over
        # [-coeff_bound, coeff_bound], modulated by an optimized scale and phase.
        # These constants come from configs/chain_soft.yaml soft_intervention.
        self.grid_size = int(grid_size)
        self.n_alpha = int(n_alpha)
        self.n_beta = int(n_beta)
        self.coeff_bound = float(coeff_bound)
        self.theoretical_optimum = 1.0 + 3.0 * math.sqrt(2.0 / math.pi)

    def _policy(self, context, policy_scale_action, policy_phase_action):
        scale = 2.0 * policy_scale_action - 1.0
        phase = 2.0 * policy_phase_action - 1.0
        dtype = context.dtype
        device = context.device
        freqs = torch.arange(1, self.grid_size + 1, dtype=dtype, device=device)
        alpha_base = torch.linspace(
            -self.coeff_bound, self.coeff_bound, self.n_alpha,
            dtype=dtype, device=device
        )
        beta_base = torch.linspace(
            self.coeff_bound, -self.coeff_bound, self.n_beta,
            dtype=dtype, device=device
        )
        context_expanded = context.unsqueeze(-1) * freqs + phase.unsqueeze(-1)
        policy = scale * (
            torch.sum(alpha_base * torch.sin(context_expanded), dim=-1)
            + torch.sum(beta_base * torch.cos(context_expanded), dim=-1)
        ) / math.sqrt(self.n_alpha + self.n_beta)
        return torch.clamp(policy, -1.0, 1.0)

    def evaluate(self, X):
        self.check_input(X)
        output = torch.empty(X.shape[:-1] + torch.Size([self.dag.get_n_nodes()]))
        batch_shape = output[..., 0].shape

        output[..., 0] = torch.normal(0, 1, batch_shape)
        w_unintervened = torch.normal(0, 1, batch_shape)
        use_w = (X[..., 0] > 0.5).to(dtype=X.dtype)
        w_value = 2.0 * X[..., 1] - 1.0
        output[..., 1] = use_w * w_value + (1.0 - use_w) * w_unintervened
        output[..., 2] = self._policy(output[..., 0], X[..., 2], X[..., 3])

        noise_y = self.additive_noise_dists[3].rsample(sample_shape=batch_shape)
        y_raw = -output[..., 1] - 3.0 * output[..., 2] * output[..., 0] + noise_y
        output[..., 3] = -y_raw
        return output


class ToyGraph(CausalEnv):
    r""" The ToyGraph environment from [agliettiCausalBayesianOptimization2020].
    """
    def __init__(self, noise_scales=1.0):
        parent_nodes = [[], [0], [1]]
        dag = DAG(parent_nodes)
        super(ToyGraph, self).__init__(dag)
        self.valid_targets = [
            torch.tensor([0, 1, 0]),
            torch.tensor([1, 0, 0]),
            torch.tensor([0, 0, 0]),
        ]

        self.additive_noise_dists = functions_utils.noise_scales_to_normals(
            noise_scales, self.dag.get_n_nodes()
        )
        self.theoretical_optimum = 2.5718

    def do_map(self, X):
        r"""Maps an input in [0,1] to the range used in this environment"""
        B = X.clone()
        B[..., 0] = 10.0 * X[..., 0] - 5.0
        B[..., 1] = 25 * X[..., 1] - 5.0
        return B

    def evaluate(self, X):
        self.check_input(X)
        X_do = X[
            ..., : self.dag.get_n_nodes()
        ]  # seperate the binary part from the intervention part
        X = X[..., self.dag.get_n_nodes() :]
        X = self.do_map(X)
        output = torch.empty(X.shape[:-1] + torch.Size([self.dag.get_n_nodes()]))
        noise0 = self.additive_noise_dists[0].rsample(sample_shape=output[..., 0].shape)
        output[..., 0] = self.do_int(noise0, X_do[..., 0], X[..., 0])

        noise1 = self.additive_noise_dists[1].rsample(sample_shape=output[..., 1].shape)
        output[..., 1] = self.do_int(
            (torch.exp(-output[..., 0]) + noise1), X_do[..., 1], X[..., 1]
        )

        noise2 = self.additive_noise_dists[2].rsample(sample_shape=output[..., 2].shape)
        unintervened_output_2 = (
            torch.cos(output[..., 1]) - torch.exp(-output[..., 1] / 20.0) + noise2
        )
        r'''
        [agliettiCausalBayesianOptimization2020] always minimizes outcomes whilst we 
        maximize, so I take the negative of the final output. 
        '''
        output[..., 2] = -self.do_int(unintervened_output_2, X_do[..., 2], X[..., 2])
        return output


class PSAGraph(CausalEnv):
    r""" The PSAGraph environment from [agliettiCausalBayesianOptimization2020].
    """
    def __init__(self):
        # ordering: age, bmi, A, S, cancer, Y
        parent_nodes = [[], [0], [0, 1], [0, 1], [0, 1, 2, 3], [0, 1, 2, 3, 4]]
        dag = DAG(parent_nodes)
        super(PSAGraph, self).__init__(dag)
        self.valid_targets = [
            torch.tensor([0, 0, 1, 0, 0, 0]),
            torch.tensor([0, 0, 0, 1, 0, 0]),
            torch.tensor([0, 0, 1, 1, 0, 0]),
        ]
        self.additive_noise_dists = [
            Uniform(-10, 10),
            Normal(0, 0.7),
            Normal(0, 0.001),
            Normal(0, 0.001),
            Normal(0, 0.001),
            Normal(0, 0.4),
        ]
        # Estimated theoretical optimum for PSAGraph (since it's negated, this is positive)
        self.theoretical_optimum = -4.13

    def do_map(self, X):
        return X

    def f_age(self, shape):
        # easier to just sample implicitely. Only use the additive_noise_dists
        # for passing to the experiments module
        return functions_utils.uniform(shape, 55, 75)

    def f_bmi(self, age):
        return torch.normal(27.0 - 0.01 * age, 0.7)

    def f_A(self, age, bmi):
        sigmoid_result = torch.sigmoid(-8.0 + 0.10 * age + 0.03 * bmi)
        threshold = 0.34  # mean of sigmoid_result
        return (sigmoid_result > threshold).float()

    def f_S(self, age, bmi):
        sigmoid_result = torch.sigmoid(-13.0 + 0.10 * age + 0.20 * bmi)
        threshold = 0.24  # mean of sigmoid_result
        return (sigmoid_result > threshold).float()

    def f_cancer(self, age, bmi, A, S):
        sigmoid_result = torch.sigmoid(2.2 - 0.05 * age + 0.01 * bmi - 0.04 * S + 0.02 * A)
        threshold = 0.31  # mean of sigmoid_result
        return (sigmoid_result > threshold).float()

    def f_Y(self, age, bmi, A, S, cancer):
        return torch.normal(
            6.8 + 0.04 * age - 0.15 * bmi - 0.60 * S + 0.55 * A + 1.00 * cancer, 0.4
        )

    def evaluate(self, X):
        self.check_input(X)
        X_do = X[
            ..., : self.dag.get_n_nodes()
        ]  # seperate the binary part from the intervention part
        X = X[..., self.dag.get_n_nodes() :]
        X = self.do_map(X)
        output = torch.empty(X.shape[:-1] + torch.Size([self.dag.get_n_nodes()]))
        output[..., 0] = self.do_int(
            self.f_age(output[..., 0].shape), X_do[..., 0], X[..., 0]
        )
        output[..., 1] = self.do_int(
            self.f_bmi(output[..., 0]), X_do[..., 1], X[..., 1]
        )
        output[..., 2] = self.do_int(
            self.f_A(output[..., 0], output[..., 1]), X_do[..., 2], X[..., 2]
        )

        output[..., 3] = self.do_int(
            self.f_S(output[..., 0], output[..., 1]), X_do[..., 3], X[..., 3]
        )

        unintervened_output_4 = self.f_cancer(
            output[..., 0], output[..., 1], output[..., 2], output[..., 3]
        )
        output[..., 4] = self.do_int(unintervened_output_4, X_do[..., 4], X[..., 4])
        r'''
        [agliettiCausalBayesianOptimization2020] always minimizes outcomes whilst we 
        maximize, so I take the negative of the final output. 
        '''
        output[..., 5] = -self.f_Y(
            output[..., 0],
            output[..., 1],
            output[..., 2],
            output[..., 3],
            output[..., 4],
        )
        return output


class CompleteGraph(CausalEnv):
    r""" 
    The CompleteGraph environment implementing a deterministic (noise-free) SCM with variables A, B, C, D, E, Y.
    Based on the structural equations (without noise terms):
    - A = F² + U₁
    - B = U₂  
    - C = exp(-B)
    - D = exp(-C)/10
    - E = cos(A) + C/10
    - Y = cos(D) + sin(E) + U₁ + U₂
    
    Only interventions on B, D, and E are allowed.
    """
    def __init__(self, noise_scales=None):
        # Node ordering: A, B, C, D, E, Y (indices 0, 1, 2, 3, 4, 5)
        # Dependencies:
        # A (0): no endogenous parents
        # B (1): no endogenous parents  
        # C (2): parents = [B] = [1]
        # D (3): parents = [C] = [2]
        # E (4): parents = [A, C] = [0, 2]
        # Y (5): parents = [D, E] = [3, 4]
        parent_nodes = [[], [], [1], [2], [0, 2], [3, 4]]
        dag = DAG(parent_nodes)
        super(CompleteGraph, self).__init__(dag)
        
        # Define valid intervention targets
        # Only allow interventions on B, D, and E
        self.valid_targets = [
            torch.tensor([0, 1, 0, 0, 0, 0]),  # intervene on B
            torch.tensor([0, 0, 0, 1, 0, 0]),  # intervene on D
            torch.tensor([0, 0, 0, 0, 1, 0]),  # intervene on E
            torch.tensor([0, 1, 0, 1, 0, 0]),  # intervene on B and D
            torch.tensor([0, 1, 0, 0, 1, 0]),  # intervene on B and E
            torch.tensor([0, 0, 0, 1, 1, 0]),  # intervene on D and E
            torch.tensor([0, 1, 0, 1, 1, 0]),  # intervene on B, D, and E
            torch.tensor([0, 0, 0, 0, 0, 0]),  # no interventions
        ]
        
        # No noise - deterministic environment (noise_scales parameter ignored)
        self.additive_noise_dists = functions_utils.noise_scales_to_normals(
            0.0, self.dag.get_n_nodes()
        )

        self.theoretical_optimum = 4.4474

    def do_map(self, X):
        r"""Maps an input in [0,1] to the intervention domains
        (B: [-5, 4], D: [-5, 5], E: [-6, 3]); other nodes are not interventional."""
        B = X.clone()
        B[..., 1] = -5.0 + (4.0 - (-5.0)) * X[..., 1]   # B -> [-5, 4]
        B[..., 3] = -5.0 + (5.0 - (-5.0)) * X[..., 3]   # D -> [-5, 5]
        B[..., 4] = -6.0 + (3.0 - (-6.0)) * X[..., 4]   # E -> [-6, 3]
        return B
    
    def generate_exogenous_variables(self, shape):
        """Generate exogenous variables U1, U2, F for each sample"""
        U1 = torch.normal(0, 1, shape)
        U2 = torch.normal(0, 1, shape) 
        F = torch.normal(0, 1, shape)
        return U1, U2, F
    
    def f_A(self, F, U1, noise):
        """A = F² + U₁ + ε_A"""
        return F**2 + U1 + noise
    
    def f_B(self, U2, noise):
        """B = U₂ + ε_B"""
        return U2 + noise
    
    def f_C(self, B, noise):
        """C = exp(-B) + ε_C"""
        return torch.exp(-B) + noise
    
    def f_D(self, C, noise):
        """D = exp(-C)/10 + ε_D"""
        return torch.exp(-C) / 10 + noise
    
    def f_E(self, A, C, noise):
        """E = cos(A) + C/10 + ε_E"""
        return torch.cos(A) + C/10 + noise
    
    def f_Y(self, D, E, U1, U2, noise):
        """Y = cos(D) + sin(E) + U₁ + U₂·ε_y"""
        return torch.cos(D) + torch.sin(E) + U1 + U2 * noise

    def evaluate(self, X):
        self.check_input(X)
        X_do = X[..., : self.dag.get_n_nodes()]  # binary intervention flags
        X = X[..., self.dag.get_n_nodes() :]     # intervention values
        X = self.do_map(X)
        
        output = torch.empty(X.shape[:-1] + torch.Size([self.dag.get_n_nodes()]))
        
        # Generate exogenous variables for this batch
        batch_shape = output[..., 0].shape
        U1, U2, F = self.generate_exogenous_variables(batch_shape)
        
        # Node 0: A (no noise)
        output[..., 0] = self.do_int(
            self.f_A(F, U1, torch.zeros_like(F)), X_do[..., 0], X[..., 0]
        )
        
        # Node 1: B (no noise)
        output[..., 1] = self.do_int(
            self.f_B(U2, torch.zeros_like(U2)), X_do[..., 1], X[..., 1]
        )
        
        # Node 2: C (depends on B, no noise)
        output[..., 2] = self.do_int(
            self.f_C(output[..., 1], torch.zeros_like(output[..., 1])), X_do[..., 2], X[..., 2]
        )
        
        # Node 3: D (depends on C, no noise)
        output[..., 3] = self.do_int(
            self.f_D(output[..., 2], torch.zeros_like(output[..., 2])), X_do[..., 3], X[..., 3]
        )
        
        # Node 4: E (depends on A, C, no noise)
        output[..., 4] = self.do_int(
            self.f_E(output[..., 0], output[..., 2], torch.zeros_like(output[..., 0])), X_do[..., 4], X[..., 4]
        )
        
        # Node 5: Y (depends on D, E, and exogenous U1, U2; eps_y ~ N(0, 0.1))
        # Negate Y to convert from minimization to maximization problem
        output[..., 5] = -self.f_Y(
            output[..., 3], output[..., 4], U1, U2, 0.1 * torch.randn_like(output[..., 3])
        )
        
        return output


class Synthetic_2(CausalEnv):
    r"""
    The Synthetic_2 environment implementing a synthetic causal graph with variables X, Z, Y.
    Based on the structural equations:
    
    Exogenous Variables:
    - X ~ N(0,1) (exogenous)
    
    Endogenous Variables:
    - Z = exp(-X) + U_Z, where U_Z ~ N(0,1)
    - Y = cos(Z) - exp(-Z/20) + U_Y, where U_Y ~ N(0,1)
    
    Interventional domains:
    - X: [-3, 2]
    - Z: [-1, 1]
    
    Only interventions on X and Z are allowed.
    Target variable: Y (minimize)
    
    Node ordering: X(0), Z(1), Y(2)
    """
    def __init__(self, noise_scales=1.0):
        # Node ordering and dependencies:
        # 0: X (exogenous) - parents: []
        # 1: Z - parents: [0] (X)
        # 2: Y - parents: [1] (Z)
        parent_nodes = [
            [],    # 0: X
            [0],   # 1: Z
            [1]    # 2: Y
        ]
        dag = DAG(parent_nodes)
        super(Synthetic_2, self).__init__(dag)
        
        # Define valid intervention targets
        # Allow interventions on X and Z
        self.valid_targets = [
            torch.tensor([1, 0, 0]),  # intervene on X
            torch.tensor([0, 1, 0]),  # intervene on Z
            torch.tensor([1, 1, 0]),  # intervene on X and Z
            torch.tensor([0, 0, 0]),  # no interventions
        ]
        
        # Set up noise distributions
        self.additive_noise_dists = functions_utils.noise_scales_to_normals(
            noise_scales, self.dag.get_n_nodes()
        )
        
        self.theoretical_optimum = 3.9690
    
    def do_map(self, X):
        r"""Maps an input in [0,1] to the intervention ranges for X and Z"""
        B = X.clone()
        # Map X intervention values to [-3, 2]
        B[..., 0] = 5.0 * X[..., 0] - 3.0  # [0,1] -> [-3, 2]
        # Map Z intervention values to [-1, 1]  
        B[..., 1] = 2.0 * X[..., 1] - 1.0  # [0,1] -> [-1, 1]
        # Y is not interventional, but we need to handle the dimension
        B[..., 2] = X[..., 2]  # Keep as is (not used for interventions)
        return B
    
    def f_X(self, batch_shape):
        """X ~ N(0,1) (exogenous)"""
        return torch.normal(0, 1, batch_shape)
    
    def f_Z(self, X, noise):
        """Z = exp(-X) + U_Z, where U_Z ~ N(0,1)"""
        return torch.exp(-X) + noise
    
    def f_Y(self, Z, noise):
        """Y = cos(Z) - exp(-Z/20) + U_Y, where U_Y ~ N(0,1)"""
        return torch.cos(Z) - torch.exp(-Z/20) + noise

    def evaluate(self, X):
        self.check_input(X)
        X_do = X[..., : self.dag.get_n_nodes()]  # binary intervention flags
        X = X[..., self.dag.get_n_nodes() :]     # intervention values
        X = self.do_map(X)
        
        output = torch.empty(X.shape[:-1] + torch.Size([self.dag.get_n_nodes()]))
        batch_shape = output[..., 0].shape
        
        # Node 0: X (exogenous, normally distributed)
        noise_0 = self.additive_noise_dists[0].rsample(sample_shape=batch_shape)
        output[..., 0] = self.do_int(
            self.f_X(batch_shape), X_do[..., 0], X[..., 0]
        )
        
        # Node 1: Z (depends on X, with noise)
        noise_1 = self.additive_noise_dists[1].rsample(sample_shape=batch_shape)
        output[..., 1] = self.do_int(
            self.f_Z(output[..., 0], noise_1), X_do[..., 1], X[..., 1]
        )
        
        # Node 2: Y (depends on Z, with noise) - TARGET VARIABLE
        # Since we want to minimize Y, we negate it for maximization
        noise_2 = self.additive_noise_dists[2].rsample(sample_shape=batch_shape)
        output[..., 2] = -self.f_Y(output[..., 1], noise_2)
        
        return output


class Chain(CausalEnv):
    r"""
    The Chain environment implementing a causal graph with variables X, W, Z, Y.
    Based on the structural equations:
    
    Exogenous Variables:
    - X ~ N(0,1) (exogenous)
    - W ~ N(0,1) (exogenous)
    
    Endogenous Variables:
    - Z = -0.5*X + U_Z, where U_Z ~ N(0,1)
    - Y = -W - 3*Z*X + U_Y, where U_Y ~ N(0,1)
    
    Interventional domains:
    - Z: [-1, 1]
    - W: [-1, 1]
    
    Only interventions on Z and W are allowed.
    Target variable: Y (minimize)
    
    Node ordering: X(0), W(1), Z(2), Y(3)
    """
    def __init__(self, noise_scales=1.0):
        # Node ordering and dependencies:
        # 0: X (exogenous) - parents: []
        # 1: W (exogenous) - parents: []
        # 2: Z - parents: [0] (X)
        # 3: Y - parents: [0, 1, 2] (X, W, Z)
        parent_nodes = [
            [],       # 0: X
            [],       # 1: W
            [0],      # 2: Z
            [0, 1, 2] # 3: Y
        ]
        dag = DAG(parent_nodes)
        super(Chain, self).__init__(dag)
        
        # Define valid intervention targets
        # Allow interventions on Z and W
        self.valid_targets = [
            torch.tensor([0, 0, 1, 0]),  # intervene on Z
            torch.tensor([0, 1, 0, 0]),  # intervene on W
            torch.tensor([0, 1, 1, 0]),  # intervene on W and Z
            torch.tensor([0, 0, 0, 0]),  # no interventions
        ]
        
        # Set up noise distributions
        self.additive_noise_dists = functions_utils.noise_scales_to_normals(
            noise_scales, self.dag.get_n_nodes()
        )
        
        self.theoretical_optimum = 15.8060
    
    def do_map(self, X):
        r"""Maps an input in [0,1] to the intervention ranges for Z and W"""
        B = X.clone()
        # Map X intervention values (not interventional, but keep for consistency)
        B[..., 0] = X[..., 0]  # X is not interventional
        # Map W intervention values to [-1, 1]
        B[..., 1] = 2.0 * X[..., 1] - 1.0  # [0,1] -> [-1, 1]
        # Map Z intervention values to [-1, 1]  
        B[..., 2] = 2.0 * X[..., 2] - 1.0  # [0,1] -> [-1, 1]
        # Y is not interventional
        B[..., 3] = X[..., 3]
        return B
    
    def f_X(self, batch_shape):
        """X ~ N(0,1) (exogenous)"""
        return torch.normal(0, 1, batch_shape)
    
    def f_W(self, batch_shape):
        """W ~ N(0,1) (exogenous)"""
        return torch.normal(0, 1, batch_shape)
    
    def f_Z(self, X, noise):
        """Z = -0.5*X + U_Z, where U_Z ~ N(0,1)"""
        return -0.5 * X + noise
    
    def f_Y(self, X, W, Z, noise):
        """Y = -W - 3*Z*X + U_Y, where U_Y ~ N(0,1)"""
        return -W - 3.0 * Z * X + noise

    def evaluate(self, X):
        self.check_input(X)
        X_do = X[..., : self.dag.get_n_nodes()]  # binary intervention flags
        X = X[..., self.dag.get_n_nodes() :]     # intervention values
        X = self.do_map(X)
        
        output = torch.empty(X.shape[:-1] + torch.Size([self.dag.get_n_nodes()]))
        batch_shape = output[..., 0].shape
        
        # Node 0: X (exogenous, normally distributed)
        output[..., 0] = self.f_X(batch_shape)
        
        # Node 1: W (exogenous, normally distributed) - INTERVENTIONABLE
        noise_1 = self.additive_noise_dists[1].rsample(sample_shape=batch_shape)
        output[..., 1] = self.do_int(
            self.f_W(batch_shape), X_do[..., 1], X[..., 1]
        )
        
        # Node 2: Z (depends on X, with noise) - INTERVENTIONABLE
        noise_2 = self.additive_noise_dists[2].rsample(sample_shape=batch_shape)
        output[..., 2] = self.do_int(
            self.f_Z(output[..., 0], noise_2), X_do[..., 2], X[..., 2]
        )
        
        # Node 3: Y (depends on X, W, Z, with noise) - TARGET VARIABLE
        # Since we want to minimize Y, we negate it for maximization
        noise_3 = self.additive_noise_dists[3].rsample(sample_shape=batch_shape)
        output[..., 3] = -self.f_Y(output[..., 0], output[..., 1], output[..., 2], noise_3)
        
        return output


class EcologyH(CausalEnv):
    r"""
    The EcologyH environment: Bermuda reef calcification SCM matching
    sem/ecology_sem_equations.json.

    Variables: E (Tem), S (Sal), N (Nut), T (TA) exogenous;
               X (pCO2) <- E; C (Chl-a) <- N; L (Light) <- C; P (pHsw) <- X;
               D (DIC) <- X; O (Omega_A) <- E, S, X; Y (NEC) <- L, N, P, O.

    Interventional variables: N, O, C, T, D with domains
    N: [-2, 5], O: [2, 4], C: [0, 1], T: [2200, 2500], D: [1950, 2100].

    Target variable: Y = NEC (MAXIMIZED; the output is not negated).

    Node ordering: E(0), S(1), N(2), T(3), X(4), C(5), L(6), P(7), D(8), O(9), Y(10)
    """
    def __init__(self, noise_scales=1.0):
        # Parent sets (indices in node ordering above)
        parent_nodes = [
            [],           # 0: E (exogenous)
            [],           # 1: S (exogenous)
            [],           # 2: N (exogenous)
            [],           # 3: T (exogenous)
            [0],          # 4: X <- E
            [2],          # 5: C <- N
            [5],          # 6: L <- C
            [4],          # 7: P <- X
            [4],          # 8: D <- X
            [0, 1, 4],    # 9: O <- E, S, X
            [6, 2, 7, 9], # 10: Y <- L, N, P, O
        ]
        dag = DAG(parent_nodes)
        super(EcologyH, self).__init__(dag)

        # Valid intervention targets: N(2), O(9), C(5), T(3), D(8)
        self.valid_targets = [
            torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),  # intervene on N
            torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),  # intervene on O
            torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),  # intervene on C
            torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),  # intervene on T
            torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),  # intervene on D
            torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # no interventions
        ]

        # Unit-normal noise; the fitted residual stds are embedded in the f_* below
        self.additive_noise_dists = functions_utils.noise_scales_to_normals(
            noise_scales, self.dag.get_n_nodes()
        )

        self.theoretical_optimum = 9.2652

    def do_map(self, X):
        r"""Maps an input in [0,1] to the intervention ranges."""
        B = X.clone()
        # E, S are not interventional
        B[..., 0] = X[..., 0]
        B[..., 1] = X[..., 1]
        # N -> [-2, 5]
        B[..., 2] = -2.0 + (5.0 - (-2.0)) * X[..., 2]
        # T -> [2200, 2500]
        B[..., 3] = 2200.0 + (2500.0 - 2200.0) * X[..., 3]
        # X is not interventional
        B[..., 4] = X[..., 4]
        # C -> [0, 1]
        B[..., 5] = X[..., 5]
        # L is not interventional
        B[..., 6] = X[..., 6]
        # P is not interventional
        B[..., 7] = X[..., 7]
        # D -> [1950, 2100]
        B[..., 8] = 1950.0 + (2100.0 - 1950.0) * X[..., 8]
        # O -> [2, 4]
        B[..., 9] = 2.0 + (4.0 - 2.0) * X[..., 9]
        # Y is not interventional
        B[..., 10] = X[..., 10]
        return B

    def f_E(self, batch_shape):
        """E ~ N(24.184130, 3.220405)"""
        return torch.normal(24.184130434782606, 3.2204054790158585, batch_shape)

    def f_S(self, batch_shape):
        """S ~ N(36.591624, 0.149197)"""
        return torch.normal(36.591623913043485, 0.14919708693132147, batch_shape)

    def f_N(self, batch_shape):
        """N ~ N(0.492065, 1.592408)"""
        return torch.normal(0.4920652173913041, 1.5924077981635778, batch_shape)

    def f_T(self, batch_shape):
        """T ~ N(2357.893696, 27.609355)"""
        return torch.normal(2357.893695652174, 27.609355287148222, batch_shape)

    def f_X(self, E, noise):
        """X = 18.798174 + 15.797384*E + U_X, U_X ~ N(0, 28.896639)"""
        return 18.79817390034509 + 15.797383660309157 * E + 28.896638707243493 * noise

    def f_C(self, N, noise):
        """C = 0.373420 - 0.002400*N + U_C, U_C ~ N(0, 0.039295)"""
        return 0.37342021355081695 - 0.0024002572713753005 * N + 0.039295137153226196 * noise

    def f_L(self, C, noise):
        """L = 6665.081996 - 10737.462582*C + U_L, U_L ~ N(0, 1546.913034)"""
        return 6665.081995591907 - 10737.462582329481 * C + 1546.9130344410867 * noise

    def f_P(self, X, noise):
        """P = 8.427706 - 0.000966*X + U_P, U_P ~ N(0, 0.005258)"""
        return 8.427705924233592 - 0.0009655314986363629 * X + 0.005258257571036228 * noise

    def f_D(self, X, noise):
        """D = 2131.672107 - 0.216560*X + U_D, U_D ~ N(0, 18.412100)"""
        return 2131.6721072369096 - 0.2165604666734924 * X + 18.412099847093625 * noise

    def f_O(self, E, S, X, noise):
        """O = 3.245248 + 0.094332*E + 0.006754*S - 0.005737*X + U_O, U_O ~ N(0, 0.036958)"""
        return (3.2452480541052733 + 0.09433222790709161 * E + 0.006753920720864577 * S
                - 0.005737087151230129 * X + 0.03695797753398314 * noise)

    def f_Y(self, L, N, P, O, noise):
        """Y = 211.422555 - 0.000030*L + 0.016680*N - 25.719277*P - 0.500403*O + U_Y, U_Y ~ N(0, 1.073340)"""
        return (211.4225551648116 - 3.012576882356712e-05 * L + 0.01667959146973974 * N
                - 25.71927722049691 * P - 0.5004034826468742 * O + 1.0733401618242278 * noise)

    def evaluate(self, X):
        self.check_input(X)
        X_do = X[..., : self.dag.get_n_nodes()]  # binary intervention flags
        X = X[..., self.dag.get_n_nodes() :]     # intervention values
        X = self.do_map(X)

        output = torch.empty(X.shape[:-1] + torch.Size([self.dag.get_n_nodes()]))
        batch_shape = output[..., 0].shape

        # Node 0: E (exogenous)
        output[..., 0] = self.f_E(batch_shape)

        # Node 1: S (exogenous)
        output[..., 1] = self.f_S(batch_shape)

        # Node 2: N (exogenous) - INTERVENTIONABLE
        output[..., 2] = self.do_int(self.f_N(batch_shape), X_do[..., 2], X[..., 2])

        # Node 3: T (exogenous) - INTERVENTIONABLE
        output[..., 3] = self.do_int(self.f_T(batch_shape), X_do[..., 3], X[..., 3])

        # Node 4: X <- E
        noise_4 = self.additive_noise_dists[4].rsample(sample_shape=batch_shape)
        output[..., 4] = self.f_X(output[..., 0], noise_4)

        # Node 5: C <- N - INTERVENTIONABLE
        noise_5 = self.additive_noise_dists[5].rsample(sample_shape=batch_shape)
        output[..., 5] = self.do_int(
            self.f_C(output[..., 2], noise_5), X_do[..., 5], X[..., 5]
        )

        # Node 6: L <- C
        noise_6 = self.additive_noise_dists[6].rsample(sample_shape=batch_shape)
        output[..., 6] = self.f_L(output[..., 5], noise_6)

        # Node 7: P <- X
        noise_7 = self.additive_noise_dists[7].rsample(sample_shape=batch_shape)
        output[..., 7] = self.f_P(output[..., 4], noise_7)

        # Node 8: D <- X - INTERVENTIONABLE
        noise_8 = self.additive_noise_dists[8].rsample(sample_shape=batch_shape)
        output[..., 8] = self.do_int(
            self.f_D(output[..., 4], noise_8), X_do[..., 8], X[..., 8]
        )

        # Node 9: O <- E, S, X - INTERVENTIONABLE
        noise_9 = self.additive_noise_dists[9].rsample(sample_shape=batch_shape)
        output[..., 9] = self.do_int(
            self.f_O(output[..., 0], output[..., 1], output[..., 4], noise_9),
            X_do[..., 9], X[..., 9]
        )

        # Node 10: Y = NEC (TARGET, maximized; NOT negated)
        noise_10 = self.additive_noise_dists[10].rsample(sample_shape=batch_shape)
        output[..., 10] = self.f_Y(
            output[..., 6], output[..., 2], output[..., 7], output[..., 9], noise_10
        )

        return output


class Diabetes(CausalEnv):
    r"""
    The Diabetes environment implementing a diabetes prediction SCM based on medical data.
    Based on the structural equations (noise-free for endogenous variables):
    - DiabetesPedigreeFunction ~ Uniform(0.078, 2.42) (exogenous)
    - Age ~ Uniform(21, 81) as integer (exogenous)
    - Pregnancies = -1.3394 + 0.16·Age
    - BloodPressure = 56.2742 + 0.39·Age
    - SkinThickness = 10.6921 + 0.20·BloodPressure + 8.59·DiabetesPedigreeFunction - 0.24·Age
    - Insulin = 3.2963 + 3.00·SkinThickness + 31.48·DiabetesPedigreeFunction
    - BMI = 23.2497 + 0.08·BloodPressure + 0.17·SkinThickness
    - Glucose = 74.4170 - 0.21·SkinThickness + 0.10·Insulin + 0.66·BMI + 0.66·Age
    - Outcome = 1.7908 - 0.02·Pregnancies - 0.01·Glucose + 0.00·BloodPressure - 0.01·BMI - 0.10·DiabetesPedigreeFunction
    
    Only interventions on BloodPressure [0, 122] and Insulin [0, 846] are allowed.
    
    Node ordering: DiabetesPedigreeFunction(0), Age(1), Pregnancies(2), BloodPressure(3), 
                  SkinThickness(4), Insulin(5), BMI(6), Glucose(7), Outcome(8)
    """
    def __init__(self, noise_scales=None):
        # Node ordering and dependencies:
        # 0: DiabetesPedigreeFunction (exogenous) - parents: []
        # 1: Age (exogenous) - parents: []
        # 2: Pregnancies - parents: [1] (Age)
        # 3: BloodPressure - parents: [1] (Age)
        # 4: SkinThickness - parents: [3, 0, 1] (BloodPressure, DiabetesPedigreeFunction, Age)
        # 5: Insulin - parents: [4, 0] (SkinThickness, DiabetesPedigreeFunction)
        # 6: BMI - parents: [3, 4] (BloodPressure, SkinThickness)
        # 7: Glucose - parents: [4, 5, 6, 1] (SkinThickness, Insulin, BMI, Age)
        # 8: Outcome - parents: [2, 7, 3, 6, 0] (Pregnancies, Glucose, BloodPressure, BMI, DiabetesPedigreeFunction)
        parent_nodes = [
            [],           # 0: DiabetesPedigreeFunction
            [],           # 1: Age
            [1],          # 2: Pregnancies
            [1],          # 3: BloodPressure
            [3, 0, 1],    # 4: SkinThickness
            [4, 0],       # 5: Insulin
            [3, 4],       # 6: BMI
            [4, 5, 6, 1], # 7: Glucose
            [2, 7, 3, 6, 0] # 8: Outcome
        ]
        dag = DAG(parent_nodes)
        super(Diabetes, self).__init__(dag)
        
        # Define valid intervention targets
        # Only allow interventions on Insulin and BloodPressure
        self.valid_targets = [
            torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0]),  # BloodPressure
            torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0]),  # Insulin
            torch.tensor([0, 0, 0, 1, 0, 1, 0, 0, 0]),  # BloodPressure + Insulin
        ]
        
        # No noise - deterministic environment (noise_scales parameter ignored)
        self.additive_noise_dists = functions_utils.noise_scales_to_normals(
            0.0, self.dag.get_n_nodes()
        )
        
        # Estimated theoretical optimum (maximize outcome - diabetes prediction score)
        self.theoretical_optimum = 0.8368  
    
    def do_map(self, X):
        return X
    
    def f_DiabetesPedigreeFunction(self, batch_shape):
        """DiabetesPedigreeFunction ~ Uniform(0.078, 2.42)"""
        return torch.rand(batch_shape) * (2.42 - 0.078) + 0.078
    
    def f_Age(self, batch_shape):
        """Age ~ Uniform(21, 81) as integer"""
        # Generate uniform float in [21, 81] then round to integer
        age_float = torch.rand(batch_shape) * (81.0 - 21.0) + 21.0
        return torch.round(age_float)
    
    def f_Pregnancies(self, Age, noise):
        """Pregnancies = -1.3394 + 0.16·Age + ε"""
        return -1.3394 + 0.16 * Age + noise
    
    def f_BloodPressure(self, Age, noise):
        """BloodPressure = 56.2742 + 0.39·Age + ε"""
        return 56.2742 + 0.39 * Age + noise
    
    def f_SkinThickness(self, BloodPressure, DiabetesPedigreeFunction, Age, noise):
        """SkinThickness = 10.6921 + 0.20·BloodPressure + 8.59·DiabetesPedigreeFunction - 0.24·Age + ε"""
        return 10.6921 + 0.20 * BloodPressure + 8.59 * DiabetesPedigreeFunction - 0.24 * Age + noise
    
    def f_Insulin(self, SkinThickness, DiabetesPedigreeFunction, noise):
        """Insulin = 3.2963 + 3.00·SkinThickness + 31.48·DiabetesPedigreeFunction + ε"""
        return 3.2963 + 3.00 * SkinThickness + 31.48 * DiabetesPedigreeFunction + noise
    
    def f_BMI(self, BloodPressure, SkinThickness, noise):
        """BMI = 23.2497 + 0.08·BloodPressure + 0.17·SkinThickness + ε"""
        return 23.2497 + 0.08 * BloodPressure + 0.17 * SkinThickness + noise
    
    def f_Glucose(self, SkinThickness, Insulin, BMI, Age, noise):
        """Glucose = 74.4170 - 0.21·SkinThickness + 0.10·Insulin + 0.66·BMI + 0.66·Age + ε"""
        return 74.4170 - 0.21 * SkinThickness + 0.10 * Insulin + 0.66 * BMI + 0.66 * Age + noise
    
    def f_Outcome(self, Pregnancies, Glucose, BloodPressure, BMI, DiabetesPedigreeFunction, noise):
        """Outcome = 1.7908 - 0.02·Pregnancies - 0.01·Glucose + 0.00·BloodPressure - 0.01·BMI - 0.10·DiabetesPedigreeFunction + ε"""
        return 1.7908 - 0.02 * Pregnancies - 0.01 * Glucose + 0.00 * BloodPressure - 0.01 * BMI - 0.10 * DiabetesPedigreeFunction + noise

    def evaluate(self, X):
        self.check_input(X)
        X_do = X[..., : self.dag.get_n_nodes()]  # binary intervention flags
        X = X[..., self.dag.get_n_nodes() :]     # intervention values
        X = self.do_map(X)
        
        output = torch.empty(X.shape[:-1] + torch.Size([self.dag.get_n_nodes()]))
        batch_shape = output[..., 0].shape
        
        # Node 0: DiabetesPedigreeFunction (exogenous, uniformly distributed)
        output[..., 0] = self.f_DiabetesPedigreeFunction(batch_shape)
        
        
        # Node 1: Age (exogenous, uniformly distributed)
        output[..., 1] = self.f_Age(batch_shape)
        
        
        # Node 2: Pregnancies (depends on Age, no noise)
        output[..., 2] = self.f_Pregnancies(output[..., 1], torch.zeros_like(output[..., 1]))
        
        
        # Node 3: BloodPressure (depends on Age, no noise) - INTERVENTIONABLE
        output[..., 3] = self.do_int(
            self.f_BloodPressure(output[..., 1], torch.zeros_like(output[..., 1])), X_do[..., 3], X[..., 3]
        )
        
        # Node 4: SkinThickness (depends on BloodPressure, DiabetesPedigreeFunction, Age, no noise)
        output[..., 4] = self.do_int(
            self.f_SkinThickness(output[..., 3], output[..., 0], output[..., 1], torch.zeros_like(output[..., 3])), X_do[..., 4], X[..., 4]
        )
        
        # Node 5: Insulin (depends on SkinThickness, DiabetesPedigreeFunction, no noise) - INTERVENTIONABLE
        output[..., 5] = self.do_int(
            self.f_Insulin(output[..., 4], output[..., 0], torch.zeros_like(output[..., 4])), X_do[..., 5], X[..., 5]
        )
        
        
        # Node 6: BMI (depends on BloodPressure, SkinThickness, no noise)
        output[..., 6] = self.do_int(
            self.f_BMI(output[..., 3], output[..., 4], torch.zeros_like(output[..., 3])), X_do[..., 6], X[..., 6]
        )
        
        # Node 7: Glucose (depends on SkinThickness, Insulin, BMI, Age, no noise)
        output[..., 7] = self.do_int(
            self.f_Glucose(output[..., 4], output[..., 5], output[..., 6], output[..., 1], torch.zeros_like(output[..., 4])), X_do[..., 7], X[..., 7]
        )
        
        # Node 8: Outcome (depends on Pregnancies, Glucose, BloodPressure, BMI, DiabetesPedigreeFunction, no noise)
        # This is our target variable - maximize outcome (negate for maximization)
        output[..., 8] = -self.do_int(
            self.f_Outcome(
                output[..., 2], output[..., 7], output[..., 3], output[..., 6], output[..., 0], torch.zeros_like(output[..., 2])
            ), X_do[..., 8], X[..., 8]
        )
        
        return output


class PSA_cdc(CausalEnv):
    r"""
    The PSA_cdc environment implementing a PSA screening SCM based on CDC data.
    Based on the structural equations:
    
    Exogenous Variables:
    - Age ~ Uniform(40.0, 80.0)
    
    Endogenous Variables:
    - Aspirin = -0.11150676125977188 + 0.0028351700976795165·Age
    - cancer = -0.03219056548415463 + 0.000624055746190663·Age
    - Statin = -0.4301116250265201 + 0.011962926923648507·Age + 0.22879470732551127·Aspirin
    - BMI = 31.909557845884837 - 0.057206596823079305·Age + 1.8024468168006789·Statin
    - PSA = -1.7212497864635612 + 0.07057053193905057·Age + 4.93154199589294·cancer - 0.02146297604133429·BMI
    
    Only interventions on Aspirin and Statin are allowed.
    
    Node ordering: Age(0), Aspirin(1), cancer(2), Statin(3), BMI(4), PSA(5)
    """
    def __init__(self, noise_scales=None):
        # Node ordering and dependencies:
        # 0: Age (exogenous) - parents: []
        # 1: Aspirin - parents: [0] (Age)
        # 2: cancer - parents: [0] (Age)
        # 3: Statin - parents: [0, 1] (Age, Aspirin)
        # 4: BMI - parents: [0, 3] (Age, Statin)
        # 5: PSA - parents: [0, 2, 4] (Age, cancer, BMI)
        parent_nodes = [
            [],        # 0: Age
            [0],       # 1: Aspirin
            [0],       # 2: cancer
            [0, 1],    # 3: Statin
            [0, 3],    # 4: BMI
            [0, 2, 4]  # 5: PSA
        ]
        dag = DAG(parent_nodes)
        super(PSA_cdc, self).__init__(dag)
        
        # Define valid intervention targets
        # Only allow interventions on Aspirin and Statin (medical treatments)
        self.valid_targets = [
            torch.tensor([0, 1, 0, 0, 0, 0]),  # Aspirin
            torch.tensor([0, 0, 0, 1, 0, 0]),  # Statin
            torch.tensor([0, 1, 0, 1, 0, 0]),  # Aspirin + Statin

        ]
        
        # No noise - deterministic environment (noise_scales parameter ignored)
        self.additive_noise_dists = functions_utils.noise_scales_to_normals(
            0.0, self.dag.get_n_nodes()
        )
        
        # Estimated theoretical optimum (maximize PSA)
        self.theoretical_optimum = -0.07  # Updated estimate
    
    def do_map(self, X):
        r"""Maps an input in [0,1] to appropriate intervention ranges for medical variables"""
        B = X.clone()
        # Map intervention values to medical ranges
        # BloodPressure: [0, 122] (node 3)
        B[..., 3] = 122.0 * X[..., 3]  
        # Insulin: [0, 846] (node 5)
        B[..., 5] = 846.0 * X[..., 5]
        return B
    
    def f_Age(self, batch_shape):
        """Age ~ Uniform(40.0, 80.0)"""
        return torch.rand(batch_shape) * (80.0 - 40.0) + 40.0
    
    def f_Aspirin(self, Age, noise):
        """Aspirin = -0.11150676125977188 + 0.0028351700976795165·Age + ε"""
        return -0.11150676125977188 + 0.0028351700976795165 * Age + noise
    
    def f_cancer(self, Age, noise):
        """cancer = -0.03219056548415463 + 0.000624055746190663·Age + ε"""
        return -0.03219056548415463 + 0.000624055746190663 * Age + noise
    
    def f_Statin(self, Age, Aspirin, noise):
        """Statin = -0.4301116250265201 + 0.011962926923648507·Age + 0.22879470732551127·Aspirin + ε"""
        return -0.4301116250265201 + 0.011962926923648507 * Age + 0.22879470732551127 * Aspirin + noise
    
    def f_BMI(self, Age, Statin, noise):
        """BMI = 31.909557845884837 - 0.057206596823079305·Age + 1.8024468168006789·Statin + ε"""
        return 31.909557845884837 - 0.057206596823079305 * Age + 1.8024468168006789 * Statin + noise
    
    def f_PSA(self, Age, cancer, BMI, noise):
        """PSA = -1.7212497864635612 + 0.07057053193905057·Age + 4.93154199589294·cancer - 0.02146297604133429·BMI + ε"""
        return -1.7212497864635612 + 0.07057053193905057 * Age + 4.93154199589294 * cancer - 0.02146297604133429 * BMI + noise

    def evaluate(self, X):
        self.check_input(X)
        X_do = X[..., : self.dag.get_n_nodes()]  # binary intervention flags
        X = X[..., self.dag.get_n_nodes() :]     # intervention values
        X = self.do_map(X)
        
        output = torch.empty(X.shape[:-1] + torch.Size([self.dag.get_n_nodes()]))
        batch_shape = output[..., 0].shape
        
        # Node 0: Age (exogenous, uniformly distributed)
        output[..., 0] = self.f_Age(batch_shape)
        
        
        # Node 1: Aspirin (depends on Age, no noise)
        output[..., 1] = self.f_Aspirin(output[..., 0], torch.zeros_like(output[..., 0]))
        
        # Node 2: cancer (depends on Age, no noise)
        output[..., 2] = self.f_cancer(output[..., 0], torch.zeros_like(output[..., 0]))
        
        # Node 3: Statin (depends on Age, Aspirin, no noise)
        output[..., 3] = self.do_int(
            self.f_Statin(output[..., 0], output[..., 1], torch.zeros_like(output[..., 0])), X_do[..., 3], X[..., 3]
        )
        
        # Node 4: BMI (depends on Age, Statin, no noise)
        output[..., 4] = self.do_int(
            self.f_BMI(output[..., 0], output[..., 3], torch.zeros_like(output[..., 0])), X_do[..., 4], X[..., 4]
        )
        
        # Node 5: PSA (depends on Age, cancer, BMI, no noise)
        # This is our target variable - maximize PSA (negate for maximization)
        output[..., 5] = -self.f_PSA(
            output[..., 0], output[..., 2], output[..., 4], torch.zeros_like(output[..., 0])
        )
        
        return output


class Epidemiology(CausalEnv):
    r"""
    The Epidemiology environment implementing a complex causal model with variables B, T, L, R, Y.
    Based on the structural equations:
    
    Exogenous Variables:
    - B ~ Uniform(-1, 1)
    - T ~ Uniform(4, 8)
    
    Endogenous Variables:
    - L = exp(0.5 * T + U) where U ~ N(0,1)
    - R = 4 + L * T (deterministic)
    - Y = 0.5 + cos(4*T) + sin(-L + 2*R) + B + ε where ε ~ N(0,1)
    
    Interventional domains:
    - L: [0.479, 801.787]
    - B: [-0.994, 0.999]
    
    Only interventions on L and B are allowed.
    Target variable: Y (minimize)
    
    Node ordering: B(0), T(1), L(2), R(3), Y(4)
    """
    def __init__(self, noise_scales=1.0):
        # Node ordering and dependencies based on causal_order: B, T, L, R, Y
        # 0: B (exogenous) - parents: []
        # 1: T (exogenous) - parents: []
        # 2: L - parents: [1] (T)
        # 3: R - parents: [2, 1] (L, T)
        # 4: Y - parents: [1, 2, 3, 0] (T, L, R, B)
        parent_nodes = [
            [],        # 0: B
            [],        # 1: T
            [1],       # 2: L
            [2, 1],    # 3: R
            [1, 2, 3, 0] # 4: Y
        ]
        dag = DAG(parent_nodes)
        super(Epidemiology, self).__init__(dag)
        
        # Define valid intervention targets (L and B)
        self.valid_targets = [
            torch.tensor([1, 0, 0, 0, 0]),  # intervene on B
            torch.tensor([0, 0, 1, 0, 0]),  # intervene on L
            torch.tensor([1, 0, 1, 0, 0]),  # intervene on B and L
            torch.tensor([0, 0, 0, 0, 0]),  # no interventions
        ]
        
        # Set up noise distributions
        self.additive_noise_dists = functions_utils.noise_scales_to_normals(
            noise_scales, self.dag.get_n_nodes()
        )
        
        # Theoretical optimum (estimated based on the complex trigonometric function)
        self.theoretical_optimum = 2.5  # Placeholder - would need optimization to find exact value
    
    def do_map(self, X):
        r"""Maps an input in [0,1] to the intervention ranges"""
        B = X.clone()
        # Map B intervention values to [-0.994, 0.999]
        B[..., 0] = -0.994 + (0.999 - (-0.994)) * X[..., 0]
        # T is not interventional
        B[..., 1] = X[..., 1]
        # Map L intervention values to [0.479, 801.787]
        B[..., 2] = 0.479 + (801.787 - 0.479) * X[..., 2]
        # R and Y are not interventional
        B[..., 3] = X[..., 3]
        B[..., 4] = X[..., 4]
        return B
    
    def f_B(self, batch_shape):
        """B ~ Uniform(-1, 1)"""
        return torch.rand(batch_shape) * 2.0 - 1.0
    
    def f_T(self, batch_shape):
        """T ~ Uniform(4, 8)"""
        return torch.rand(batch_shape) * 4.0 + 4.0
    
    def f_L(self, T, noise):
        """L = exp(0.5 * T + U) where U ~ N(0,1)"""
        return torch.exp(0.5 * T + noise)
    
    def f_R(self, L, T, noise):
        """R = 4 + L * T (deterministic, noise ignored)"""
        return 4.0 + L * T
    
    def f_Y(self, T, L, R, B, noise):
        """Y = 0.5 + cos(4*T) + sin(-L + 2*R) + B + ε where ε ~ N(0,1)"""
        return 0.5 + torch.cos(4.0 * T) + torch.sin(-L + 2.0 * R) + B + noise

    def evaluate(self, X):
        self.check_input(X)
        X_do = X[..., : self.dag.get_n_nodes()]  # binary intervention flags
        X = X[..., self.dag.get_n_nodes() :]     # intervention values
        X = self.do_map(X)
        
        output = torch.empty(X.shape[:-1] + torch.Size([self.dag.get_n_nodes()]))
        batch_shape = output[..., 0].shape
        
        # Node 0: B (exogenous, uniformly distributed) - INTERVENTIONABLE
        output[..., 0] = self.do_int(
            self.f_B(batch_shape), X_do[..., 0], X[..., 0]
        )
        
        # Node 1: T (exogenous, uniformly distributed)
        output[..., 1] = self.f_T(batch_shape)
        
        # Node 2: L (depends on T, with noise) - INTERVENTIONABLE
        noise_2 = self.additive_noise_dists[2].rsample(sample_shape=batch_shape)
        output[..., 2] = self.do_int(
            self.f_L(output[..., 1], noise_2), X_do[..., 2], X[..., 2]
        )
        
        # Node 3: R (depends on L, T, deterministic)
        output[..., 3] = self.f_R(output[..., 2], output[..., 1], torch.zeros_like(output[..., 2]))
        
        # Node 4: Y (depends on T, L, R, B, with noise) - TARGET VARIABLE
        # Since we want to minimize Y, we negate it for maximization
        noise_4 = self.additive_noise_dists[4].rsample(sample_shape=batch_shape)
        output[..., 4] = -self.f_Y(output[..., 1], output[..., 2], output[..., 3], output[..., 0], noise_4)
        
        return output


class Protein(CausalEnv):
    r"""Protein signalling DAG matching CBO_Benchmark/sem/protein_sem_equations.json.

    Variables (causal order, 8 nodes):
      0: PKC (exogenous, INTERVENTIONABLE)
      1: PKA (depends on PKC, INTERVENTIONABLE)
      2: Raf (depends on PKC, PKA)
      3: Mek (depends on PKC, PKA, Raf)
      4: P38 (depends on PKC, PKA)
      5: Jnk (depends on PKC, PKA)
      6: Akt (depends on PKA)
      7: Erk (depends on PKA, Mek) -- TARGET (minimize, so we negate it)

    Intervention domains (from CBO_Benchmark/configs/protein.yaml):
      PKC in [0.5, 106.5]
      PKA in [1.45, 4491.5]
    """

    def __init__(self, noise_scales=1.0):
        parent_nodes = [
            [],            # 0: PKC
            [0],           # 1: PKA
            [0, 1],        # 2: Raf
            [0, 1, 2],     # 3: Mek
            [0, 1],        # 4: P38
            [0, 1],        # 5: Jnk
            [1],           # 6: Akt
            [1, 3],        # 7: Erk
        ]
        dag = DAG(parent_nodes)
        super(Protein, self).__init__(dag)

        # Allow MCBO to choose between intervening on PKC, PKA, or both.
        # The "no interventions" target is also included for the MCBO loop.
        self.valid_targets = [
            torch.tensor([1, 0, 0, 0, 0, 0, 0, 0]),  # do(PKC)
            torch.tensor([0, 1, 0, 0, 0, 0, 0, 0]),  # do(PKA)
            torch.tensor([1, 1, 0, 0, 0, 0, 0, 0]),  # do(PKC, PKA)
            torch.tensor([0, 0, 0, 0, 0, 0, 0, 0]),  # observational
        ]

        self.additive_noise_dists = functions_utils.noise_scales_to_normals(
            noise_scales, self.dag.get_n_nodes()
        )

        # Theoretical optimum (max-space). Erk = -23.249 + 0.0817*PKA - 0.0299*Mek
        # is min-Erk-near the analytic argmin (PKC=0.5, PKA=1.45) which gives
        # E[Erk] approx -24.07. Negated for maximization.
        self.theoretical_optimum = 24.07

    def do_map(self, X):
        B = X.clone()
        # Map PKC intervention values from [0,1] to [0.5, 106.5]
        B[..., 0] = 0.5 + (106.5 - 0.5) * X[..., 0]
        # Map PKA intervention values from [0,1] to [1.45, 4491.5]
        B[..., 1] = 1.45 + (4491.5 - 1.45) * X[..., 1]
        # Other nodes are not interventional
        for i in range(2, 8):
            B[..., i] = X[..., i]
        return B

    # --- structural equations (linear-Gaussian, taken from the SEM JSON) ---
    def f_PKC(self, batch_shape):
        # Exogenous uniform draw on the intervention domain (used only when not
        # intervened upon). Uses distributions.uniform for sampling shape.
        u = Uniform(torch.tensor(1.0), torch.tensor(106.0))
        return u.sample(batch_shape)

    def f_PKA(self, PKC, noise):
        return 554.3907305751273 + 0.8411525573301047 * PKC + noise

    def f_Raf(self, PKC, PKA, noise):
        return (62.199045961323414
                - 0.17774509086121948 * PKC
                - 0.0003792075216956362 * PKA + noise)

    def f_Mek(self, PKC, PKA, Raf, noise):
        return (-1.0902746220577555
                + 0.03966245723060245 * PKC
                - 0.0006520266851554879 * PKA
                + 0.5208447051064984 * Raf + noise)

    def f_P38(self, PKC, PKA, noise):
        return (15.144327731160235
                + 1.2347832903144826 * PKC
                + 0.000590540235751414 * PKA + noise)

    def f_Jnk(self, PKC, PKA, noise):
        return (52.9536028509298
                - 0.7648011883501407 * PKC
                - 0.005306096821872689 * PKA + noise)

    def f_Akt(self, PKA, noise):
        return -31.110747106386974 + 0.12890532877586242 * PKA + noise

    def f_Erk(self, PKA, Mek, noise):
        return (-23.248743333459377
                + 0.08170673252473967 * PKA
                - 0.029885814321525 * Mek + noise)

    def evaluate(self, X):
        self.check_input(X)
        X_do = X[..., : self.dag.get_n_nodes()]
        X = X[..., self.dag.get_n_nodes():]
        X = self.do_map(X)

        output = torch.empty(X.shape[:-1] + torch.Size([self.dag.get_n_nodes()]))
        batch_shape = output[..., 0].shape

        # Node 0: PKC (exogenous, INTERVENTIONABLE)
        output[..., 0] = self.do_int(
            self.f_PKC(batch_shape), X_do[..., 0], X[..., 0]
        )

        # Node 1: PKA (INTERVENTIONABLE)
        noise_1 = self.additive_noise_dists[1].rsample(sample_shape=batch_shape)
        output[..., 1] = self.do_int(
            self.f_PKA(output[..., 0], noise_1), X_do[..., 1], X[..., 1]
        )

        # Node 2: Raf
        noise_2 = self.additive_noise_dists[2].rsample(sample_shape=batch_shape)
        output[..., 2] = self.f_Raf(output[..., 0], output[..., 1], noise_2)

        # Node 3: Mek
        noise_3 = self.additive_noise_dists[3].rsample(sample_shape=batch_shape)
        output[..., 3] = self.f_Mek(output[..., 0], output[..., 1], output[..., 2], noise_3)

        # Node 4: P38
        noise_4 = self.additive_noise_dists[4].rsample(sample_shape=batch_shape)
        output[..., 4] = self.f_P38(output[..., 0], output[..., 1], noise_4)

        # Node 5: Jnk
        noise_5 = self.additive_noise_dists[5].rsample(sample_shape=batch_shape)
        output[..., 5] = self.f_Jnk(output[..., 0], output[..., 1], noise_5)

        # Node 6: Akt
        noise_6 = self.additive_noise_dists[6].rsample(sample_shape=batch_shape)
        output[..., 6] = self.f_Akt(output[..., 1], noise_6)

        # Node 7: Erk -- TARGET (minimize), so negate for the MCBO max loop.
        noise_7 = self.additive_noise_dists[7].rsample(sample_shape=batch_shape)
        output[..., 7] = -self.f_Erk(output[..., 1], output[..., 3], noise_7)

        return output
