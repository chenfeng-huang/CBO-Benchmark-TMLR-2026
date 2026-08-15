from typing import List
from networkx import MultiDiGraph


def make_graphical_model(
    start_time: int, stop_time: int, topology: str, nodes: List[str], target_node: str = None, verbose: bool = False
) -> MultiDiGraph:
    """
    Build a temporal causal graph as a NetworkX MultiDiGraph without external graphviz dependencies.

    Parameters
    ----------
    start_time : int
        Index of first time-step
    stop_time : int
        Index of the last time-step (inclusive)
    topology : str
        Either "dependent" (chain within-slice) or "independent" (all manipulatives -> target)
    nodes : list[str]
        Nodes in a single time-slice. If topology=="independent", include the target in this list or pass via target_node
    target_node : str, optional
        Required for topology=="independent"; the within-slice target
    verbose : bool
        Ignored (kept for backward compatibility)
    """

    assert start_time <= stop_time
    assert topology in ["dependent", "independent"]
    assert nodes

    # Normalize nodes/target for independent topology
    if topology == "independent":
        assert target_node is not None
        node_list = [n for n in nodes if n != target_node]
        all_slice_nodes = node_list + [target_node]
    else:
        node_list = list(nodes)
        all_slice_nodes = node_list

    G = MultiDiGraph()

    # Add nodes for each time index
    for t in range(start_time, stop_time + 1):
        for n in all_slice_nodes:
            G.add_node(f"{n}_{t}")

    # Add within-slice (spatial) edges
    for t in range(start_time, stop_time + 1):
        if topology == "dependent":
            for a, b in zip(node_list, node_list[1:]):
                G.add_edge(f"{a}_{t}", f"{b}_{t}")
        else:
            for a in node_list:
                G.add_edge(f"{a}_{t}", f"{target_node}_{t}")

    # Add temporal (transition) edges
    for t in range(start_time, stop_time):
        for n in all_slice_nodes:
            G.add_edge(f"{n}_{t}", f"{n}_{t+1}")

    return G

