# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Implementation of SCM examples that we run experiments on."""

from __future__ import annotations

import abc
import collections
from typing import Any, Optional, Tuple

import graphviz
from networkx.classes import multidigraph
from networkx.drawing import nx_agraph
import numpy as np
try:
    import pygraphviz
    HAS_PYGRAPHVIZ = True
except ImportError:
    print("Warning: pygraphviz not available. Some graph functionalities may be limited.")
    HAS_PYGRAPHVIZ = False
from scipy import stats

from ccbo.scm_examples import scm
from ccbo.utils import utilities
import networkx as nx


class BaseExample(abc.ABC):
  """Abstract class for experiment examples."""

  def __init__(self):
    self._variables = None
    self._constraints = None

  @property
  def constraints(self) -> Any:
    return self._constraints

  @property
  def variables(self) -> Any:
    """Returns the variables dictionary."""
    return self._variables

  @abc.abstractproperty   # pylint: disable=deprecated-decorator
  def scm_funcs(self) -> collections.OrderedDict[str, Any]:
    """Returns the functions of the structural causal model."""
    raise NotImplementedError("scm_funcs should be implemented")

  @abc.abstractmethod
  def structural_causal_model(self, variables: Optional[Tuple[str, ...]],
                              lambdas: Optional[Tuple[float, ...]]) -> Any:
    """Returns the scm with fncs, variables and constraints."""
    raise NotImplementedError("structural_causal_model should be implemented")


class SyntheticExample1(BaseExample):
  """Synthetic example #1 - corresponds to DAG 1(c) in the cCBO paper."""

  @property
  def scm_funcs(self) -> collections.OrderedDict[str, Any]:
    """Define functions in SCM."""
    x = lambda noise, sample: noise
    z = lambda noise, sample: np.exp(-sample["X"]) + noise
    y = lambda noise, sample: np.cos(sample["Z"]) - np.exp(-sample["Z"] / 20.0  # pylint: disable=g-long-lambda
                                                          ) + noise
    return collections.OrderedDict([("X", x), ("Z", z), ("Y", y)])

  def structural_causal_model(self,
                              variables: Optional[Tuple[str, ...]],
                              lambdas: Optional[Tuple[float, ...]]) -> Any:
    self._variables = {
        "X": ["m", [-3, 2]],
        "Z": ["m", [-1, 1]],
        "Y": ["t"],
    }
    if variables is not None and lambdas is not None:
      self._constraints = {
          var: [utilities.Direction.LOWER, val]
          for var, val in zip(variables, lambdas)
      }

    return scm.Scm(
        constraints=self.constraints,
        scm_funcs=self.scm_funcs,
        variables=self.variables)


class SyntheticExample2(BaseExample):
  """Synthetic example #2 - corresponds to DAG 1(d) in the cCBO paper."""

  @property
  def scm_funcs(self) -> collections.OrderedDict[str, Any]:
    """Define functions in SCM."""
    a = lambda noise, sample: noise
    b = lambda noise, sample: noise
    c = lambda noise, sample: np.exp(-sample["A"]) / 5. + noise
    d = lambda noise, sample: np.cos(sample["B"]) + sample["C"] / 10. + noise
    e = lambda noise, sample: np.exp(-sample["C"]) / 10. + noise
    y = lambda noise, sample: np.cos(sample["D"]) - sample["D"] / 5. + np.sin(  # pylint: disable=g-long-lambda
        sample["E"]) - sample["E"] / 4. + noise
    return collections.OrderedDict([("A", a), ("B", b), ("C", c), ("D", d),
                                    ("E", e), ("Y", y)])

  def graph(self) -> multidigraph.MultiDiGraph:
    """Define causal graph structure."""
    ranking = []
    nodes = ["A", "B", "C", "D", "E", "Y"]
    myedges = ["A -> C; C -> E; B -> D; D -> Y; C -> D; E -> Y"]
    ranking.append("{{ rank=same; {} }} ".format(" ".join(nodes)))
    ranking = "".join(ranking)
    edges = "".join(myedges)
    graph = "digraph {{ rankdir=LR; {} {} }}".format(edges, ranking)
    if HAS_PYGRAPHVIZ:
        dag = nx_agraph.from_agraph(
            pygraphviz.AGraph(graphviz.Source(graph).source))
    else:
        # Fallback: create a simple networkx graph
        import networkx as nx
        dag = nx.DiGraph()
        # Add nodes
        for node in nodes:
            dag.add_node(node)
        # Add edges based on the structure
        dag.add_edges_from([("A", "C"), ("C", "E"), ("B", "D"), ("D", "Y"), ("C", "D"), ("E", "Y")])
    return dag

  def structural_causal_model(self,
                              variables: Optional[Tuple[str, ...]],
                              lambdas: Optional[Tuple[float, ...]]) -> Any:
    self._variables = {
        "A": ["m", [-5, 5]],
        "B": ["nm", [-4, 4]],
        "C": ["nm", [0, 10]],
        "D": ["m", [-1, 1]],
        "E": ["m", [-1, 1]],
        "Y": ["t"],
    }

    if variables is not None and lambdas is not None:
      self._constraints = {
          var: [utilities.Direction.LOWER, val]
          for var, val in zip(variables, lambdas)
      }

    return scm.Scm(
        constraints=self.constraints,
        scm_funcs=self.scm_funcs,
        variables=self.variables,
        graph=self.graph())


class HealthExample(BaseExample):
  """Real example #1 - corresponds to Fig 1(a) in the cCBO paper."""

  @property
  def scm_funcs(self) -> collections.OrderedDict[str, Any]:
    """Define equations in SCM."""
    a = lambda noise, sample: np.random.uniform(low=55, high=75)  # age
    # bmr - base metabolic rate
    b = lambda noise, sample: stats.truncnorm.rvs(-1, 2) * 10 + 1500.
    c = lambda noise, sample: np.random.uniform(low=-100, high=100)  # calories
    # height
    d = lambda noise, sample: stats.truncnorm.rvs(-0.5, 0.5) * 10 + 175.

    e = lambda noise, sample: (sample["B"] + 6.8 * sample["A"] - 5 * sample["D"]  # pylint: disable=g-long-lambda
                              ) / 13.7 + sample["C"] * 150. / 7716.  # weight

    f = lambda noise, sample: sample["E"] / ((sample["D"] / 100)**2)  # bmi

    g = lambda noise, sample: np.random.uniform(low=0, high=1)  # statin
    h = lambda noise, sample: utilities.sigmoid(-8.0 + 0.10 * sample["A"] + 0.03  # pylint: disable=g-long-lambda
                                                * sample["F"])  # aspirin
    i = lambda noise, sample: utilities.sigmoid(2.2 - 0.05 * sample[  # pylint: disable=g-long-lambda
        "A"] + 0.01 * sample["F"] - 0.04 * sample["G"] + 0.02 * sample["H"]
                                               )  # cancer

    y = lambda noise, sample: np.random.normal(  # pylint: disable=g-long-lambda
        6.8 + 0.04 * sample["A"] - 0.15 * sample["F"] - 0.60 * sample["G"] +
        0.55 * sample["H"] + 1.00 * sample["I"], 0.4)  # psa
    return collections.OrderedDict([("A", a), ("B", b), ("C", c), ("D", d),
                                    ("E", e), ("F", f), ("G", g), ("H", h),
                                    ("I", i), ("Y", y)])

  def graph(self) -> multidigraph.MultiDiGraph:
    """Define causal graph structure."""
    ranking = []
    nodes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "Y"]
    myedges = [
        "A -> F; A -> G;  A -> I; A -> H; A -> E; B -> E; C -> E; D -> E; D ->"  # pylint: disable=implicit-str-concat
        " F; E -> F; F -> H;  F -> I; G -> I; G -> Y; H -> Y; H -> I; I -> Y"
    ]

    ranking.append("{{ rank=same; {} }} ".format(" ".join(nodes)))
    ranking = "".join(ranking)
    edges = "".join(myedges)
    graph = "digraph {{ rankdir=LR; {} {} }}".format(edges, ranking)
    if HAS_PYGRAPHVIZ:
        dag = nx_agraph.from_agraph(
            pygraphviz.AGraph(graphviz.Source(graph).source))
    else:
        # Fallback: create a simple networkx graph
        import networkx as nx
        dag = nx.DiGraph()
        # Add nodes
        for node in nodes:
            dag.add_node(node)
        # Add edges based on the structure - parsing the edge description
        edge_pairs = [("A", "F"), ("A", "G"), ("A", "I"), ("A", "H"), ("A", "E"), 
                      ("B", "E"), ("C", "E"), ("D", "E"), ("D", "F"), ("E", "F"), 
                      ("F", "H"), ("F", "I"), ("G", "I"), ("G", "Y"), ("H", "Y"), 
                      ("H", "I"), ("I", "Y")]
        dag.add_edges_from(edge_pairs)
    return dag

  def structural_causal_model(self,
                              variables: Optional[Tuple[str, ...]],
                              lambdas: Optional[Tuple[float, ...]]) -> Any:
    self._variables = {
        "A": ["nm", [55, 75]],
        "B": ["nm", [1450, 1550]],
        "C": ["m", [-400, +400]],
        "D": ["nm", [169, 180]],
        "E": ["nm", [68, 86]],
        "F": ["nm", [19, 25]],
        "G": ["m", [0, 1]],
        "H": ["m", [0, 1]],
        "I": ["nm", [0.2, 0.5]],
        "Y": ["t"],
    }

    if variables is not None and lambdas is not None:
      self._constraints = {
          var: [utilities.Direction.LOWER, val]
          for var, val in zip(variables, lambdas)
      }

    return scm.Scm(
        constraints=self.constraints,
        scm_funcs=self.scm_funcs,
        variables=self.variables,
        graph=self.graph())


class CompleteGraphExample(BaseExample):
  """CompleteGraph example - complex causal model with latent confounders."""

  @property
  def scm_funcs(self) -> collections.OrderedDict[str, Any]:
    """Define functions in SCM based on CompleteGraph model."""
    # Store latent variables as class attributes for access across functions
    self._U1 = None
    self._U2 = None
    
    def f(noise, sample):
      return noise  # F ~ N(0,1)
    
    def a(noise, sample):
      # A = F² + U₁ + ε_A
      # Generate U1 if not exists
      if self._U1 is None:
        self._U1 = np.random.normal(0, 1)
      return sample["F"]**2 + self._U1 + noise
    
    def b(noise, sample):
      # B = U₂ + ε_B
      # Generate U2 if not exists
      if self._U2 is None:
        self._U2 = np.random.normal(0, 1)
      return self._U2 + noise
    
    def c(noise, sample):
      # C = exp(-B) + ε_C
      return np.exp(-sample["B"]) + noise
    
    def d(noise, sample):
      # D = exp(-C)/10 + ε_D
      return np.exp(-sample["C"]) / 10 + noise
    
    def e(noise, sample):
      # E = cos(A) + C/10 + ε_E
      return np.cos(sample["A"]) + sample["C"]/10 + noise
    
    def y(noise, sample):
      # Y = cos(D) + sin(E) + U₁ + U₂·ε_y
      # Use the same latent variables
      if self._U1 is None:
        self._U1 = np.random.normal(0, 1)
      if self._U2 is None:
        self._U2 = np.random.normal(0, 1)
      return (np.cos(sample["D"]) + np.sin(sample["E"]) + 
              self._U1 + self._U2 * noise)
    
    return collections.OrderedDict([("F", f), ("A", a), ("B", b), ("C", c),
                                    ("D", d), ("E", e), ("Y", y)])

  def graph(self) -> multidigraph.MultiDiGraph:
    """Define causal graph structure for CompleteGraph."""
    ranking = []
    nodes = ["F", "A", "B", "C", "D", "E", "Y"]
    # Causal structure: F -> A -> E -> Y and B -> C -> D -> Y
    myedges = ["F -> A; A -> E; B -> C; C -> D; C -> E; D -> Y; E -> Y"]
    
    ranking.append("{{ rank=same; {} }} ".format(" ".join(nodes)))
    ranking = "".join(ranking)
    edges = "".join(myedges)
    graph = "digraph {{ rankdir=LR; {} {} }}".format(edges, ranking)
    
    if HAS_PYGRAPHVIZ:
        dag = nx_agraph.from_agraph(
            pygraphviz.AGraph(graphviz.Source(graph).source))
    else:
        # Fallback: create a simple networkx graph
        import networkx as nx
        dag = nx.DiGraph()
        # Add nodes
        for node in nodes:
            dag.add_node(node)
        # Add edges based on the structure
        edge_pairs = [("F", "A"), ("A", "E"), ("B", "C"), ("C", "D"), 
                      ("C", "E"), ("D", "Y"), ("E", "Y")]
        dag.add_edges_from(edge_pairs)
    return dag

  def structural_causal_model(self,
                              variables: Optional[Tuple[str, ...]],
                              lambdas: Optional[Tuple[float, ...]]) -> Any:
    # Reset latent variables for each new SCM instance
    self._U1 = None
    self._U2 = None
    
    self._variables = {
        "F": ["m", [-3, 3]],      # F ~ N(0,1), allow range [-3,3]
        "A": ["m", [-2, 12]],     # A = F² + U₁, range roughly [-2,12]
        "B": ["m", [-3, 3]],      # B = U₂, range [-3,3]
        "C": ["m", [0, 20]],      # C = exp(-B), range [0,20]
        "D": ["m", [0, 2]],       # D = exp(-C)/10, range [0,2]
        "E": ["m", [-1, 3]],      # E = cos(A) + C/10, range [-1,3]
        "Y": ["t"],               # Y is target
    }

    if variables is not None and lambdas is not None:
      self._constraints = {
          var: [utilities.Direction.LOWER, val]
          for var, val in zip(variables, lambdas)
      }

    return scm.Scm(
        constraints=self.constraints,
        scm_funcs=self.scm_funcs,
        variables=self.variables,
        graph=self.graph())


class PSACDCExample(BaseExample):
  """PSA-CDC example - real-world medical data with causal interventions."""

  @property
  def scm_funcs(self) -> collections.OrderedDict[str, Any]:
    """Define functions in SCM based on PSA-CDC structural equations."""
    
    def age(noise, sample):
      # Age ~ Uniform(40.0, 80.0) - Exogenous variable
      return np.random.uniform(40.0, 80.0)
    
    def aspirin(noise, sample):
      # Aspirin = -0.11150676125977188 + 0.0028351700976795165·Age + ε
      return (-0.11150676125977188 + 
              0.0028351700976795165 * sample["Age"] + noise)
    
    def cancer(noise, sample):
      # cancer = -0.03219056548415463 + 0.000624055746190663·Age + ε
      return (-0.03219056548415463 + 
              0.000624055746190663 * sample["Age"] + noise)
    
    def statin(noise, sample):
      # Statin = -0.4301116250265201 + 0.011962926923648507·Age + 0.22879470732551127·Aspirin + ε
      return (-0.4301116250265201 + 
              0.011962926923648507 * sample["Age"] + 
              0.22879470732551127 * sample["Aspirin"] + noise)
    
    def bmi(noise, sample):
      # BMI = 31.909557845884837 - 0.057206596823079305·Age + 1.8024468168006789·Statin + ε
      return (31.909557845884837 - 
              0.057206596823079305 * sample["Age"] + 
              1.8024468168006789 * sample["Statin"] + noise)
    
    def psa(noise, sample):
      # PSA = -1.7212497864635612 + 0.07057053193905057·Age + 4.93154199589294·cancer - 0.02146297604133429·BMI + ε
      return (-1.7212497864635612 + 
              0.07057053193905057 * sample["Age"] + 
              4.93154199589294 * sample["cancer"] - 
              0.02146297604133429 * sample["BMI"] + noise)
    
    return collections.OrderedDict([("Age", age), ("Aspirin", aspirin), ("cancer", cancer), 
                                    ("Statin", statin), ("BMI", bmi), ("PSA", psa)])

  def graph(self) -> multidigraph.MultiDiGraph:
    """Define causal graph structure for PSA-CDC."""
    ranking = []
    nodes = ["Age", "Aspirin", "cancer", "Statin", "BMI", "PSA"]
    # Causal structure based on the equations:
    # Age -> Aspirin, cancer, Statin, BMI, PSA
    # Aspirin -> Statin
    # Statin -> BMI
    # cancer -> PSA
    # BMI -> PSA
    myedges = ["Age -> Aspirin; Age -> cancer; Age -> Statin; Age -> BMI; Age -> PSA; " +
               "Aspirin -> Statin; Statin -> BMI; cancer -> PSA; BMI -> PSA"]
    
    ranking.append("{{ rank=same; {} }} ".format(" ".join(nodes)))
    ranking = "".join(ranking)
    edges = "".join(myedges)
    graph = "digraph {{ rankdir=LR; {} {} }}".format(edges, ranking)
    
    if HAS_PYGRAPHVIZ:
        dag = nx_agraph.from_agraph(
            pygraphviz.AGraph(graphviz.Source(graph).source))
    else:
        # Fallback: create a simple networkx graph
        import networkx as nx
        dag = nx.DiGraph()
        # Add nodes
        for node in nodes:
            dag.add_node(node)
        # Add edges based on the structure
        edge_pairs = [("Age", "Aspirin"), ("Age", "cancer"), ("Age", "Statin"), 
                      ("Age", "BMI"), ("Age", "PSA"), ("Aspirin", "Statin"), 
                      ("Statin", "BMI"), ("cancer", "PSA"), ("BMI", "PSA")]
        dag.add_edges_from(edge_pairs)
    return dag

  def structural_causal_model(self,
                              variables: Optional[Tuple[str, ...]],
                              lambdas: Optional[Tuple[float, ...]]) -> Any:
    self._variables = {
        "Age": ["nm", [40, 80]],        # Exogenous: Uniform(40, 80)
        "Aspirin": ["m", [0, 1]],       # Interventional: binary
        "cancer": ["nm", [0, 1]],       # Cancer probability
        "Statin": ["m", [0, 1]],        # Interventional: binary
        "BMI": ["nm", [15, 50]],        # BMI range
        "PSA": ["t"],                   # Target variable
    }

    if variables is not None and lambdas is not None:
      self._constraints = {
          var: [utilities.Direction.LOWER, val]
          for var, val in zip(variables, lambdas)
      }

    return scm.Scm(
        constraints=self.constraints,
        scm_funcs=self.scm_funcs,
        variables=self.variables,
        graph=self.graph())


class DiabetesExample(BaseExample):
  """Diabetes example - diabetes prediction with medical interventions."""

  @property
  def scm_funcs(self) -> collections.OrderedDict[str, Any]:
    """Define functions in SCM based on Diabetes structural equations."""
    
    def diabetes_pedigree_function(noise, sample):
      # DiabetesPedigreeFunction ~ Uniform(0.078, 2.42) - Exogenous variable
      return np.random.uniform(0.078, 2.42)
    
    def age(noise, sample):
      # Age ~ Uniform(21.0, 81.0) - Exogenous variable
      return np.random.uniform(21.0, 81.0)
    
    def pregnancies(noise, sample):
      # Pregnancies = -1.3394 + 0.16·Age + ε
      return -1.3394 + 0.16 * sample["Age"] + noise
    
    def blood_pressure(noise, sample):
      # BloodPressure = 56.2742 + 0.39·Age + ε
      return 56.2742 + 0.39 * sample["Age"] + noise
    
    def skin_thickness(noise, sample):
      # SkinThickness = 10.6921 + 0.20·BloodPressure + 8.59·DiabetesPedigreeFunction - 0.24·Age + ε
      return (10.6921 + 0.20 * sample["BloodPressure"] + 
              8.59 * sample["DiabetesPedigreeFunction"] - 0.24 * sample["Age"] + noise)
    
    def insulin(noise, sample):
      # Insulin = 3.2963 + 3.00·SkinThickness + 31.48·DiabetesPedigreeFunction + ε
      return (3.2963 + 3.00 * sample["SkinThickness"] + 
              31.48 * sample["DiabetesPedigreeFunction"] + noise)
    
    def bmi(noise, sample):
      # BMI = 23.2497 + 0.08·BloodPressure + 0.17·SkinThickness + ε
      return (23.2497 + 0.08 * sample["BloodPressure"] + 
              0.17 * sample["SkinThickness"] + noise)
    
    def glucose(noise, sample):
      # Glucose = 74.4170 - 0.21·SkinThickness + 0.10·Insulin + 0.66·BMI + 0.66·Age + ε
      return (74.4170 - 0.21 * sample["SkinThickness"] + 0.10 * sample["Insulin"] + 
              0.66 * sample["BMI"] + 0.66 * sample["Age"] + noise)
    
    def outcome(noise, sample):
      # Outcome = 1.7908 - 0.02·Pregnancies - 0.01·Glucose + 0.00·BloodPressure - 0.01·BMI - 0.10·DiabetesPedigreeFunction + ε
      return (1.7908 - 0.02 * sample["Pregnancies"] - 0.01 * sample["Glucose"] + 
              0.00 * sample["BloodPressure"] - 0.01 * sample["BMI"] - 
              0.10 * sample["DiabetesPedigreeFunction"] + noise)
    
    return collections.OrderedDict([
        ("DiabetesPedigreeFunction", diabetes_pedigree_function), ("Age", age), 
        ("Pregnancies", pregnancies), ("BloodPressure", blood_pressure), 
        ("SkinThickness", skin_thickness), ("Insulin", insulin), ("BMI", bmi), 
        ("Glucose", glucose), ("Outcome", outcome)
    ])

  def graph(self) -> multidigraph.MultiDiGraph:
    """Define causal graph structure for Diabetes."""
    ranking = []
    nodes = ["DiabetesPedigreeFunction", "Age", "Pregnancies", "BloodPressure", 
             "SkinThickness", "Insulin", "BMI", "Glucose", "Outcome"]
    
    # Causal structure based on the equations:
    # DiabetesPedigreeFunction -> SkinThickness, Insulin, Outcome
    # Age -> Pregnancies, BloodPressure, SkinThickness, Glucose
    # BloodPressure -> SkinThickness, BMI
    # SkinThickness -> Insulin, BMI, Glucose
    # Insulin -> Glucose
    # BMI -> Glucose
    # Pregnancies, Glucose, BloodPressure, BMI, DiabetesPedigreeFunction -> Outcome
    myedges = [
        "DiabetesPedigreeFunction -> SkinThickness; DiabetesPedigreeFunction -> Insulin; DiabetesPedigreeFunction -> Outcome; " +
        "Age -> Pregnancies; Age -> BloodPressure; Age -> SkinThickness; Age -> Glucose; " +
        "BloodPressure -> SkinThickness; BloodPressure -> BMI; " +
        "SkinThickness -> Insulin; SkinThickness -> BMI; SkinThickness -> Glucose; " +
        "Insulin -> Glucose; BMI -> Glucose; " +
        "Pregnancies -> Outcome; Glucose -> Outcome; BMI -> Outcome"
    ]
    
    ranking.append("{{ rank=same; {} }} ".format(" ".join(nodes)))
    ranking = "".join(ranking)
    edges = "".join(myedges)
    graph = "digraph {{ rankdir=LR; {} {} }}".format(edges, ranking)
    
    if HAS_PYGRAPHVIZ:
        dag = nx_agraph.from_agraph(
            pygraphviz.AGraph(graphviz.Source(graph).source))
    else:
        # Fallback: create a simple networkx graph
        import networkx as nx
        dag = nx.DiGraph()
        # Add nodes
        for node in nodes:
            dag.add_node(node)
        # Add edges based on the structure
        edge_pairs = [
            ("DiabetesPedigreeFunction", "SkinThickness"), ("DiabetesPedigreeFunction", "Insulin"), 
            ("DiabetesPedigreeFunction", "Outcome"), ("Age", "Pregnancies"), ("Age", "BloodPressure"), 
            ("Age", "SkinThickness"), ("Age", "Glucose"), ("BloodPressure", "SkinThickness"), 
            ("BloodPressure", "BMI"), ("SkinThickness", "Insulin"), ("SkinThickness", "BMI"), 
            ("SkinThickness", "Glucose"), ("Insulin", "Glucose"), ("BMI", "Glucose"), 
            ("Pregnancies", "Outcome"), ("Glucose", "Outcome"), ("BMI", "Outcome")
        ]
        dag.add_edges_from(edge_pairs)
    return dag

  def structural_causal_model(self,
                              variables: Optional[Tuple[str, ...]],
                              lambdas: Optional[Tuple[float, ...]]) -> Any:
    self._variables = {
        "DiabetesPedigreeFunction": ["nm", [0.078, 2.42]],  # Exogenous: Uniform(0.078, 2.42)
        "Age": ["nm", [21, 81]],                            # Exogenous: Uniform(21, 81)
        "Pregnancies": ["nm", [0, 20]],                     # Pregnancies range
        "BloodPressure": ["m", [0, 122]],                   # Interventional: [0, 122]
        "SkinThickness": ["nm", [0, 100]],                  # Skin thickness range
        "Insulin": ["m", [0, 846]],                         # Interventional: [0, 846]
        "BMI": ["nm", [10, 70]],                            # BMI range
        "Glucose": ["nm", [0, 200]],                        # Glucose range
        "Outcome": ["t"],                                   # Target variable
    }

    if variables is not None and lambdas is not None:
      self._constraints = {
          var: [utilities.Direction.LOWER, val]
          for var, val in zip(variables, lambdas)
      }

    return scm.Scm(
        constraints=self.constraints,
        scm_funcs=self.scm_funcs,
        variables=self.variables,
        graph=self.graph())


class ChainExample(BaseExample):
  """Chain example based on user-provided SCM for X, W, Z, Y.

  Relationships:
    - X: exogenous, U_X ~ N(0, 1)
    - W: exogenous, U_W ~ N(0, 1) and manipulable in [-1, 1]
    - Z: endogenous linear, Z = -0.5 * X + U_Z and manipulable in [-1, 1]
    - Y: interaction linear, Y = -W - 3 * Z * X + U_Y (task: min)
  """

  @property
  def scm_funcs(self) -> collections.OrderedDict[str, Any]:
    x = lambda noise, sample: noise
    w = lambda noise, sample: noise
    z = lambda noise, sample: -0.5 * float(sample["X"]) + noise
    y = lambda noise, sample: (
        -1.0 * float(sample["W"]) + (-3.0) * float(sample["Z"]) * float(sample["X"]) + noise
    )
    return collections.OrderedDict([("X", x), ("W", w), ("Z", z), ("Y", y)])

  def graph(self) -> multidigraph.MultiDiGraph:
    """Define causal graph for the chain example."""
    ranking = []
    nodes = ["X", "W", "Z", "Y"]
    myedges = ["X -> Z; X -> Y; W -> Y; Z -> Y"]
    ranking.append("{{ rank=same; {} }} ".format(" ".join(nodes)))
    ranking = "".join(ranking)
    edges = "".join(myedges)
    graph = "digraph {{ rankdir=LR; {} {} }}".format(edges, ranking)
    if HAS_PYGRAPHVIZ:
      dag = nx_agraph.from_agraph(
          pygraphviz.AGraph(graphviz.Source(graph).source))
    else:
      # Fallback: simple networkx graph
      import networkx as nx
      dag = nx.DiGraph()
      for node in nodes:
        dag.add_node(node)
      dag.add_edges_from([("X", "Z"), ("X", "Y"), ("W", "Y"), ("Z", "Y")])
    return dag

  def structural_causal_model(self,
                              variables: Optional[Tuple[str, ...]],
                              lambdas: Optional[Tuple[float, ...]]) -> Any:
    self._variables = {
        # X is exogenous, not directly manipulable in this setup
        "X": ["nm", [-3, 3]],
        # W and Z are manipulable in [-1, 1]
        "W": ["m", [-1, 1]],
        "Z": ["m", [-1, 1]],
        # Y is the optimization target
        "Y": ["t"],
    }

    if variables is not None and lambdas is not None:
      self._constraints = {
          var: [utilities.Direction.LOWER, val]
          for var, val in zip(variables, lambdas)
      }

    return scm.Scm(
        constraints=self.constraints,
        scm_funcs=self.scm_funcs,
        variables=self.variables,
        graph=self.graph())

EXAMPLES_DICT = {
    "synthetic1": SyntheticExample1,
    # Map 'synthetic2' to the X–Z–Y SCM per user specification
    "synthetic2": SyntheticExample1,
    "health": HealthExample,
    "completegraph": CompleteGraphExample,
    "psa_cdc": PSACDCExample,
    "diabetes": DiabetesExample,
    "chain": ChainExample,
}


def build_scm_from_numeric_spec(spec: dict) -> scm.Scm:
  """Build an SCM from a numeric JSON-like specification."""
  variables = spec["variables"]
  interventions = set(map(str, spec.get("intervention", [])))
  interventional_domain = spec.get("interventional_domain", {})
  target = str(spec.get("target"))

  # Order variables numerically if possible
  try:
    ordered = sorted(variables.keys(), key=lambda k: int(k))
  except Exception:
    ordered = list(variables.keys())

  # Build functions
  fn_map = []
  for v in ordered:
    info = variables[v]
    vtype = info.get("type")
    rtype = info.get("relationship_type")
    params = info.get("relationship_params", {})
    if vtype == "exogenous" and rtype == "uniform":
      low = float(params.get("min_val", -1.0))
      high = float(params.get("max_val", 1.0))
      fn_map.append((v, (lambda lo, hi: (lambda noise, sample: np.random.uniform(lo, hi)))(low, high)))
    elif vtype == "exogenous" and rtype == "normal":
      mean = float(info.get("intercept", 0.0))
      std = float(params.get("noise_std", 1.0))
      def make_normal(mean, std):
        def f(noise, sample):
          # noise is standard normal; apply scale and shift
          return mean + std * float(noise)
        return f
      fn_map.append((v, make_normal(mean, std)))
    elif vtype == "endogenous" and rtype in ("linear", "interaction_linear"):
      deps = [str(d) for d in info.get("dependencies", [])]
      coeffs = {str(k): float(val) for k, val in info.get("coefficients", {}).items()}
      intercept = float(info.get("intercept", 0.0))
      noise_std = float(params.get("noise_std", 0.0))
      interactions = params.get("interactions", []) if rtype == "interaction_linear" else []
      def make_linear_intercept(deps, coeffs, intercept, noise_std, interactions):
        def f(noise, sample):
          total = float(intercept)
          for p in deps:
            total += coeffs.get(p, 0.0) * float(sample[p])
          # Add interaction terms if provided: product of listed variables times coefficient
          for inter in interactions:
            vars_list = [str(var) for var in inter.get("variables", [])]
            coef = float(inter.get("coefficient", 0.0))
            prod = 1.0
            for var in vars_list:
              prod *= float(sample[var])
            total += coef * prod
          return total + noise_std * float(noise)
        return f
      fn_map.append((v, make_linear_intercept(deps, coeffs, intercept, noise_std, interactions)))
    elif vtype == "endogenous" and rtype == "exponential_linear":
      # L = exp(sum of coefficients * dependencies + noise)
      # e.g., L = exp(0.5 * T + U) where U ~ N(0,1)
      deps = [str(d) for d in info.get("dependencies", [])]
      coeffs = {str(k): float(val) for k, val in info.get("coefficients", {}).items()}
      noise_std = float(params.get("noise_std", 1.0))
      def make_exponential_linear(deps, coeffs, noise_std):
        def f(noise, sample):
          linear_term = 0.0
          for p in deps:
            linear_term += coeffs.get(p, 0.0) * float(sample[p])
          # noise is standard normal N(0,1), scale by noise_std
          return np.exp(linear_term + noise_std * float(noise))
        return f
      fn_map.append((v, make_exponential_linear(deps, coeffs, noise_std)))
    elif vtype == "endogenous" and rtype == "complex_trigonometric":
      # Y = intercept + cos(coeff_T * T) + sin(coeff_L * L + coeff_R * R) + B + noise
      # e.g., Y = 0.5 + cos(4*T) + sin(-L + 2*R) + B + noise
      deps = [str(d) for d in info.get("dependencies", [])]
      coeffs = {str(k): float(val) for k, val in info.get("coefficients", {}).items()}
      intercept = float(info.get("intercept", 0.0))
      noise_std = float(params.get("noise_std", 1.414))
      def make_complex_trigonometric(deps, coeffs, intercept, noise_std):
        def f(noise, sample):
          # Implementation for Y = 0.5 + cos(4*T) + sin(-L + 2*R) + B + noise
          result = float(intercept)
          # T goes into cos term
          if "T" in coeffs:
            result += np.cos(coeffs["T"] * float(sample.get("T", 0.0)))
          # L and R go into sin term: -L + 2*R
          sin_term = 0.0
          if "L" in coeffs:
            sin_term += coeffs["L"] * float(sample.get("L", 0.0))
          if "R" in coeffs:
            sin_term += coeffs["R"] * float(sample.get("R", 0.0))
          result += np.sin(sin_term)
          # Other variables (like B) are added linearly
          for dep in deps:
            if dep not in ["T", "L", "R"] and dep in coeffs:
              result += coeffs[dep] * float(sample.get(dep, 0.0))
          # Add noise
          return result + noise_std * float(noise)
        return f
      fn_map.append((v, make_complex_trigonometric(deps, coeffs, intercept, noise_std)))
    else:
      fn_map.append((v, lambda noise, sample: float(noise)))

  scm_funcs = collections.OrderedDict(fn_map)

  # Variables metadata
  variables_meta = {}
  for v in ordered:
    if v == target:
      variables_meta[v] = ["t"]
    elif v in interventions:
      dom = interventional_domain.get(v, [-1.0, 1.0])
      variables_meta[v] = ["m", [float(dom[0]), float(dom[1])]]
    else:
      info = variables[v]
      if info.get("type") == "exogenous" and info.get("relationship_type") == "uniform":
        low = float(info["relationship_params"].get("min_val", -1.0))
        high = float(info["relationship_params"].get("max_val", 1.0))
        variables_meta[v] = ["nm", [low, high]]
      elif info.get("relationship_type") == "exponential_linear":
        # For exponential functions, use a wide positive range
        variables_meta[v] = ["nm", [0.0, 10000.0]]
      elif info.get("relationship_type") == "interaction_linear":
        # For interaction terms, use a wide range
        variables_meta[v] = ["nm", [-10000.0, 10000.0]]
      else:
        variables_meta[v] = ["nm", [-100.0, 100.0]]

  # Graph
  dag = nx.DiGraph()
  for v in ordered:
    dag.add_node(v)
  for v in ordered:
    for p in map(str, variables[v].get("dependencies", [])):
      dag.add_edge(p, v)

  return scm.Scm(constraints=None, scm_funcs=scm_funcs, variables=variables_meta, graph=dag)
