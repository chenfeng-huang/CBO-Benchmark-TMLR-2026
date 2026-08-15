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

"""Define a toy graph.

Define the variables and the scm for a toy graph example.
"""
from __future__ import annotations

import collections
from typing import Any, Dict, List, Optional

from networkx.classes import multidigraph
from networkx.drawing import nx_agraph
try:
    import pygraphviz
    HAS_PYGRAPHVIZ = True
except ImportError:
    print("Warning: pygraphviz not available in scm.py. Some graph functionalities may be limited.")
    HAS_PYGRAPHVIZ = False

from ccbo.scm_examples import base
from ccbo.utils import graph_functions
from ccbo.utils import utilities


class Scm(base.ScmExample):
  """Basic SCM."""

  def __init__(
      self,
      variables: Optional[Dict[str, List[Any]]] = None,
      constraints: Optional[Dict[str, List[Any]]] = None,
      scm_funcs: Optional[collections.OrderedDict[str, Any]] = None,
      graph: Optional[multidigraph.MultiDiGraph] = None):

    if variables is None:
      variables = {
          "X": ["m", [-4, 1]],
          "Z": ["m", [-3, 3]],
          "Y": ["t"],
      }

    if constraints is None:
      constraints = {
          "X": [utilities.Direction.LOWER, 1],
          "Z": [utilities.Direction.HIGHER, 1]
      }

    if scm_funcs is None:
      scm_funcs = self.scm_funcs()

    if graph is None:
      graph = self.graph()

    args = {
        "variables": variables,
        "constraints": constraints,
        "scm_funcs": scm_funcs,
        "graph": graph
    }
    super().__init__(**args)

  def scm_funcs(self) -> collections.OrderedDict[str, Any]:
    x = lambda noise, sample: noise
    z = lambda noise, sample: 2. * sample["X"] + noise
    y = lambda noise, sample: -1 * sample["Z"] + noise
    return collections.OrderedDict([("X", x), ("Z", z), ("Y", y)])

  def graph(self) -> multidigraph.MultiDiGraph:
    """Define graph topology."""
    dag_view = graph_functions.make_graphical_model(
        topology="dependent", nodes=["X", "Z", "Y"], verbose=True)
    if HAS_PYGRAPHVIZ:
        dag = nx_agraph.from_agraph(pygraphviz.AGraph(dag_view.source))
    else:
        # Fallback: create a simple networkx graph with a basic topology
        import networkx as nx
        dag = nx.DiGraph()
        # Add nodes
        for node in ["X", "Z", "Y"]:
            dag.add_node(node)
        # Add edges for a dependent topology: X -> Z -> Y
        dag.add_edges_from([("X", "Z"), ("Z", "Y")])
    return dag
