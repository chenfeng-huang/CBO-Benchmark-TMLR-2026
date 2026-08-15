from dcbo.experimental.experiments import optimal_sequence_of_interventions
from dcbo.utils.dag_utils.graph_functions import make_graphical_model
from dcbo.utils.sem_utils.toy_sems import (
    PISHCAT_SEM,
    LinearMultipleChildrenSEM,
    NonStationaryDependentSEM,
    StationaryDependentMultipleChildrenSEM,
    StationaryDependentSEM,
    StationaryIndependentSEM,
    CompleteGraphSEM,
    PSA_SEM,
)
from dcbo.utils.sequential_intervention_functions import get_interventional_grids
from dcbo.utils.utilities import powerset
from networkx import MultiDiGraph


def setup_PISHCAT(T: int = 3):
    # Setup used in `Developing Optimal Causal Cyber-Defence Agents via Cyber Security Simulation`.

    SEM = PISHCAT_SEM()
    init_sem = SEM.static()
    sem = SEM.dynamic()

    slice_node_set = ["P", "I", "S", "H", "C", "A", "T"]
    G = make_graphical_model(0, T - 1, topology="dependent", nodes=slice_node_set, verbose=True)
    dag_view = G

    # Modulate base structure to fit PISHCAT framework

    # Transitions
    for t in range(T - 1):
        # Add
        G.add_edge("S_{}".format(t), "H_{}".format(t + 1))
        # Remove
        G.remove_edge("P_{}".format(t), "P_{}".format(t + 1))
        G.remove_edge("I_{}".format(t), "I_{}".format(t + 1))
        G.remove_edge("H_{}".format(t), "H_{}".format(t + 1))
        G.remove_edge("C_{}".format(t), "C_{}".format(t + 1))
        G.remove_edge("A_{}".format(t), "A_{}".format(t + 1))
        G.remove_edge("T_{}".format(t), "T_{}".format(t + 1))

    # Emissions
    for t in range(T):
        # Remove
        G.remove_edge("P_{}".format(t), "I_{}".format(t))
        G.remove_edge("S_{}".format(t), "H_{}".format(t))
        G.remove_edge("C_{}".format(t), "A_{}".format(t))
        # Add
        G.add_edge("P_{}".format(t), "H_{}".format(t))
        G.add_edge("P_{}".format(t), "A_{}".format(t))
        G.add_edge("I_{}".format(t), "A_{}".format(t))
        G.add_edge("C_{}".format(t), "T_{}".format(t))

    #  Specifiy all the exploration sets based on the manipulative variables in the DAG
    exploration_sets = list(powerset(["P", "I"]))
    # Specify the intervention domain for each variable
    intervention_domain = {"P": [0, 1], "I": [0, 3]}
    # Specify a grid over each exploration and use the grid to find the best intevention value for that ES
    interventional_grids = get_interventional_grids(exploration_sets, intervention_domain, size_intervention_grid=100)

    _, optimal_interventions, true_objective_values, _, _, all_causal_effects = optimal_sequence_of_interventions(
        exploration_sets=exploration_sets,
        interventional_grids=interventional_grids,
        initial_structural_equation_model=init_sem,
        structural_equation_model=sem,
        G=G,
        T=T,
        model_variables=slice_node_set,
        target_variable="T",
    )

    return (
        init_sem,
        sem,
        dag_view,
        G,
        exploration_sets,
        intervention_domain,
        true_objective_values,
        optimal_interventions,
        all_causal_effects,
    )


def setup_linear_multiple_children_scm(T: int = 3):

    #  Load SEM from figure 1 in paper (upper left quadrant)
    SEM = LinearMultipleChildrenSEM()
    init_sem = SEM.static()
    sem = SEM.dynamic()

    dag_view = make_graphical_model(
        0, T - 1, topology="dependent", nodes=["X", "Z", "Y"], verbose=True
    )  # Base topology that we build on
    G = dag_view
    G.add_edges_from([("X_0", "Y_0"), ("X_1", "Y_1"), ("X_2", "Y_2")])

    #  Specifiy all the exploration sets based on the manipulative variables in the DAG
    exploration_sets = list(powerset(["X", "Z"]))
    # Specify the intervention domain for each variable
    intervention_domain = {"X": [-4, 1], "Z": [-3, 3]}
    # Specify a grid over each exploration and use the grid to find the best intevention value for that ES
    interventional_grids = get_interventional_grids(exploration_sets, intervention_domain, size_intervention_grid=100)

    _, optimal_interventions, true_objective_values, _, _, all_causal_effects = optimal_sequence_of_interventions(
        exploration_sets=exploration_sets,
        interventional_grids=interventional_grids,
        initial_structural_equation_model=init_sem,
        structural_equation_model=sem,
        G=G,
        T=T,
        model_variables=["X", "Z", "Y"],
        target_variable="Y",
    )

    return (
        init_sem,
        sem,
        dag_view,
        G,
        exploration_sets,
        intervention_domain,
        true_objective_values,
        optimal_interventions,
        all_causal_effects,
    )


def setup_stat_multiple_children_scm(T: int = 3):

    #  Load SEM from figure 1 in paper (upper left quadrant)
    SEM = StationaryDependentMultipleChildrenSEM()
    init_sem = SEM.static()
    sem = SEM.dynamic()

    dag_view = make_graphical_model(
        0, T - 1, topology="dependent", nodes=["X", "Z", "Y"], verbose=True
    )  # Base topology that we build on
    G = dag_view
    G.add_edges_from([("X_0", "Y_0"), ("X_1", "Y_1"), ("X_2", "Y_2")])

    #  Specifiy all the exploration sets based on the manipulative variables in the DAG
    exploration_sets = list(powerset(["X", "Z"]))
    # Specify the intervention domain for each variable
    intervention_domain = {"X": [-4, 1], "Z": [-3, 3]}
    # Specify a grid over each exploration and use the grid to find the best intevention value for that ES
    interventional_grids = get_interventional_grids(exploration_sets, intervention_domain, size_intervention_grid=100)

    _, optimal_interventions, true_objective_values, _, _, all_causal_effects = optimal_sequence_of_interventions(
        exploration_sets=exploration_sets,
        interventional_grids=interventional_grids,
        initial_structural_equation_model=init_sem,
        structural_equation_model=sem,
        G=G,
        T=T,
        model_variables=["X", "Z", "Y"],
        target_variable="Y",
    )

    return (
        init_sem,
        sem,
        dag_view,
        G,
        exploration_sets,
        intervention_domain,
        true_objective_values,
        optimal_interventions,
        all_causal_effects,
    )


def setup_stat_scm(T: int = 3):

    #  Load SEM from figure 1 in paper (upper left quadrant)
    SEM = StationaryDependentSEM()
    init_sem = SEM.static()
    sem = SEM.dynamic()

    G_view = make_graphical_model(
        0, T - 1, topology="dependent", nodes=["X", "Z", "Y"], verbose=True
    )  # Base topology that we build on
    G = G_view

    #  Specifiy all the exploration sets based on the manipulative variables in the DAG
    exploration_sets = list(powerset(["X", "Z"]))
    # Specify the intervention domain for each variable
    intervention_domain = {"X": [-4, 1], "Z": [-3, 3]}
    # Specify a grid over each exploration and use the grid to find the best intevention value for that ES
    interventional_grids = get_interventional_grids(exploration_sets, intervention_domain, size_intervention_grid=100)

    _, optimal_interventions, true_objective_values, _, _, all_causal_effects = optimal_sequence_of_interventions(
        exploration_sets=exploration_sets,
        interventional_grids=interventional_grids,
        initial_structural_equation_model=init_sem,
        structural_equation_model=sem,
        G=G,
        T=T,
        model_variables=["X", "Z", "Y"],
        target_variable="Y",
    )

    return (
        init_sem,
        sem,
        G_view,
        G,
        exploration_sets,
        intervention_domain,
        true_objective_values,
        optimal_interventions,
        all_causal_effects,
    )


def setup_ind_scm(T: int = 3):

    #  Load SEM from figure 1 in paper (upper left quadrant)
    SEM = StationaryIndependentSEM()
    init_sem = SEM.static()
    sem = SEM.dynamic()

    G_view = make_graphical_model(
        0, T - 1, topology="independent", nodes=["X", "Z", "Y"], target_node="Y", verbose=True
    )  # Base topology that we build on
    G = G_view

    #  Specifiy all the exploration sets based on the manipulative variables in the DAG
    exploration_sets = list(powerset(["X", "Z"]))
    # Specify the intervention domain for each variable
    intervention_domain = {"X": [-4, 1], "Z": [-3, 3]}
    # Specify a grid over each exploration and use the grid to find the best intevention value for that ES
    interventional_grids = get_interventional_grids(exploration_sets, intervention_domain, size_intervention_grid=100)

    _, _, true_objective_values, _, _, _ = optimal_sequence_of_interventions(
        exploration_sets=exploration_sets,
        interventional_grids=interventional_grids,
        initial_structural_equation_model=init_sem,
        structural_equation_model=sem,
        G=G,
        T=T,
        model_variables=["X", "Z", "Y"],
        target_variable="Y",
    )

    return init_sem, sem, G_view, G, exploration_sets, intervention_domain, true_objective_values


def setup_nonstat_scm(T: int = 3):

    #  Load SEM from figure 3(c) in paper (upper left quadrant)
    SEM = NonStationaryDependentSEM(change_point=1)  #  Explicitly tell SEM to change at t=1
    init_sem = SEM.static()
    sem = SEM.dynamic()

    dag_view = make_graphical_model(
        0, T - 1, topology="dependent", nodes=["X", "Z", "Y"], verbose=True
    )  # Base topology that we build on
    dag = dag_view

    #  Specifiy all the exploration sets based on the manipulative variables in the DAG
    exploration_sets = list(powerset(["X", "Z"]))
    # Specify the intervention domain for each variable
    intervention_domain = {"X": [-4, 1], "Z": [-3, 3]}
    # Specify a grid over each exploration and use the grid to find the best intevention value for that ES
    interventional_grids = get_interventional_grids(exploration_sets, intervention_domain, size_intervention_grid=100)

    _, _, true_objective_values, _, _, all_causal_effects = optimal_sequence_of_interventions(
        exploration_sets=exploration_sets,
        interventional_grids=interventional_grids,
        initial_structural_equation_model=init_sem,
        structural_equation_model=sem,
        G=dag,
        T=T,
        model_variables=["X", "Z", "Y"],
        target_variable="Y",
    )

    return (
        init_sem,
        sem,
        dag_view,
        dag,
        exploration_sets,
        intervention_domain,
        true_objective_values,
        all_causal_effects,
    )


def setup_complete_graph(T: int = 1):
    SEM = CompleteGraphSEM()
    init_sem = SEM.static()
    sem = SEM.dynamic()

    # Build graph explicitly to avoid unintended chain edges.
    # U1/U2 are latent confounders (U1 -> A, Y; U2 -> B, Y), modelled as
    # explicit exogenous nodes.
    slice_nodes = ["U1", "U2", "F", "A", "B", "C", "D", "E", "Y"]
    G = MultiDiGraph()
    G.T = T
    for t in range(T):
        for n in slice_nodes:
            G.add_node(f"{n}_{t}")
    # Temporal identity (if T>1)
    if T > 1:
        for t in range(T - 1):
            for n in slice_nodes:
                G.add_edge(f"{n}_{t}", f"{n}_{t+1}")
    # Desired within-slice emissions
    for t in range(T):
        G.add_edge(f"U1_{t}", f"A_{t}")
        G.add_edge(f"U1_{t}", f"Y_{t}")
        G.add_edge(f"U2_{t}", f"B_{t}")
        G.add_edge(f"U2_{t}", f"Y_{t}")
        G.add_edge(f"F_{t}", f"A_{t}")
        G.add_edge(f"B_{t}", f"C_{t}")
        G.add_edge(f"C_{t}", f"D_{t}")
        G.add_edge(f"A_{t}", f"E_{t}")
        G.add_edge(f"C_{t}", f"E_{t}")
        G.add_edge(f"D_{t}", f"Y_{t}")
        G.add_edge(f"E_{t}", f"Y_{t}")

    # Default exploration sets and domain can be overridden by config
    exploration_sets = [("B",), ("D",), ("E",)]
    intervention_domain = {"B": [-5, 4], "D": [-5, 5], "E": [-6, 3]}

    # No precomputed ground truth by default
    true_objective_values = None
    all_causal_effects = None

    return (
        init_sem,
        sem,
        G,
        G,
        exploration_sets,
        intervention_domain,
        true_objective_values,
        all_causal_effects,
    )


def setup_psa(T: int = 1):
    SEM = PSA_SEM()
    init_sem = SEM.static()
    sem = SEM.dynamic()

    slice_nodes = ["Age", "BMI", "Aspirin", "Statin", "cancer", "PSA"]
    G = MultiDiGraph()
    G.T = T
    for t in range(T):
        for n in slice_nodes:
            G.add_node(f"{n}_{t}")
    if T > 1:
        for t in range(T - 1):
            for n in slice_nodes:
                G.add_edge(f"{n}_{t}", f"{n}_{t+1}")
    for t in range(T):
        G.add_edge(f"Age_{t}", f"BMI_{t}")
        G.add_edge(f"Age_{t}", f"Aspirin_{t}")
        G.add_edge(f"BMI_{t}", f"Aspirin_{t}")
        G.add_edge(f"Age_{t}", f"Statin_{t}")
        G.add_edge(f"BMI_{t}", f"Statin_{t}")
        G.add_edge(f"Statin_{t}", f"cancer_{t}")
        G.add_edge(f"Aspirin_{t}", f"cancer_{t}")
        G.add_edge(f"Age_{t}", f"cancer_{t}")
        G.add_edge(f"BMI_{t}", f"cancer_{t}")
        G.add_edge(f"cancer_{t}", f"PSA_{t}")
        G.add_edge(f"Age_{t}", f"PSA_{t}")
        G.add_edge(f"BMI_{t}", f"PSA_{t}")
        G.add_edge(f"Statin_{t}", f"PSA_{t}")
        G.add_edge(f"Aspirin_{t}", f"PSA_{t}")

    exploration_sets = [("Aspirin",), ("Statin",)]
    intervention_domain = {"Aspirin": [0, 1], "Statin": [0, 1]}
    true_objective_values = None
    all_causal_effects = None

    return (
        init_sem,
        sem,
        G,
        G,
        exploration_sets,
        intervention_domain,
        true_objective_values,
        all_causal_effects,
    )


def setup_psa_cdc(T: int = 1):
    from dcbo.utils.sem_utils.toy_sems import PSA_CDC_SEM

    SEM = PSA_CDC_SEM()
    init_sem = SEM.static()
    sem = SEM.dynamic()

    slice_nodes = ["Age", "BMI", "Aspirin", "Statin", "cancer", "PSA"]
    G = MultiDiGraph()
    G.T = T
    for t in range(T):
        for n in slice_nodes:
            G.add_node(f"{n}_{t}")
    if T > 1:
        for t in range(T - 1):
            for n in slice_nodes:
                G.add_edge(f"{n}_{t}", f"{n}_{t+1}")
    for t in range(T):
        # Relations per PSA-cdc equations
        G.add_edge(f"Age_{t}", f"BMI_{t}")
        G.add_edge(f"Statin_{t}", f"BMI_{t}")
        G.add_edge(f"Age_{t}", f"Aspirin_{t}")
        G.add_edge(f"Age_{t}", f"Statin_{t}")
        G.add_edge(f"Aspirin_{t}", f"Statin_{t}")
        G.add_edge(f"Age_{t}", f"cancer_{t}")
        G.add_edge(f"cancer_{t}", f"PSA_{t}")
        G.add_edge(f"Age_{t}", f"PSA_{t}")
        G.add_edge(f"BMI_{t}", f"PSA_{t}")

    exploration_sets = [("Aspirin",), ("Statin",)]
    intervention_domain = {"Aspirin": [0, 1], "Statin": [0, 1]}
    true_objective_values = None
    all_causal_effects = None

    return (
        init_sem,
        sem,
        G,
        G,
        exploration_sets,
        intervention_domain,
        true_objective_values,
        all_causal_effects,
    )


def setup_diabetes(T: int = 1):
    from dcbo.utils.sem_utils.toy_sems import DIABETES_SEM

    SEM = DIABETES_SEM()
    init_sem = SEM.static()
    sem = SEM.dynamic()

    slice_nodes = [
        "DiabetesPedigreeFunction",
        "Age",
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "Outcome",
    ]
    G = MultiDiGraph()
    G.T = T
    for t in range(T):
        for n in slice_nodes:
            G.add_node(f"{n}_{t}")
    if T > 1:
        for t in range(T - 1):
            for n in slice_nodes:
                G.add_edge(f"{n}_{t}", f"{n}_{t+1}")
    for t in range(T):
        # Edges per SEM equations
        G.add_edge(f"Age_{t}", f"Pregnancies_{t}")
        G.add_edge(f"Age_{t}", f"BloodPressure_{t}")
        G.add_edge(f"BloodPressure_{t}", f"SkinThickness_{t}")
        G.add_edge(f"DiabetesPedigreeFunction_{t}", f"SkinThickness_{t}")
        G.add_edge(f"Age_{t}", f"SkinThickness_{t}")
        G.add_edge(f"SkinThickness_{t}", f"Insulin_{t}")
        G.add_edge(f"DiabetesPedigreeFunction_{t}", f"Insulin_{t}")
        G.add_edge(f"BloodPressure_{t}", f"BMI_{t}")
        G.add_edge(f"SkinThickness_{t}", f"BMI_{t}")
        G.add_edge(f"SkinThickness_{t}", f"Glucose_{t}")
        G.add_edge(f"Insulin_{t}", f"Glucose_{t}")
        G.add_edge(f"BMI_{t}", f"Glucose_{t}")
        G.add_edge(f"Age_{t}", f"Glucose_{t}")
        G.add_edge(f"Pregnancies_{t}", f"Outcome_{t}")
        G.add_edge(f"Glucose_{t}", f"Outcome_{t}")
        G.add_edge(f"BloodPressure_{t}", f"Outcome_{t}")
        G.add_edge(f"BMI_{t}", f"Outcome_{t}")
        G.add_edge(f"DiabetesPedigreeFunction_{t}", f"Outcome_{t}")

    # Exploration sets and domains from diabetes.yaml
    exploration_sets = [("Insulin",), ("BloodPressure",)]
    intervention_domain = {"Insulin": [0, 846], "BloodPressure": [0, 122]}
    true_objective_values = None
    all_causal_effects = None

    return (
        init_sem,
        sem,
        G,
        G,
        exploration_sets,
        intervention_domain,
        true_objective_values,
        all_causal_effects,
    )


def setup_toygraph(T: int = 1):
    # Uses StationaryDependentSEM matching the toy equations X->Z->Y
    SEM = StationaryDependentSEM()
    init_sem = SEM.static()
    sem = SEM.dynamic()

    # Build simple X,Z,Y graph over T
    G_view = make_graphical_model(0, T - 1, topology="dependent", nodes=["X", "Z", "Y"], verbose=True)
    G = G_view

    # Exploration sets and domain from toyGraph.yaml
    exploration_sets = list(powerset(["X", "Z"]))
    intervention_domain = {"X": [-5, 5], "Z": [-5, 20]}

    # Precompute true causal effects if desired (optional here)
    interventional_grids = get_interventional_grids(exploration_sets, intervention_domain, size_intervention_grid=100)
    _, _, true_objective_values, _, _, all_causal_effects = optimal_sequence_of_interventions(
        exploration_sets=exploration_sets,
        interventional_grids=interventional_grids,
        initial_structural_equation_model=init_sem,
        structural_equation_model=sem,
        G=G,
        T=T,
        model_variables=["X", "Z", "Y"],
        target_variable="Y",
    )

    return (
        init_sem,
        sem,
        G_view,
        G,
        exploration_sets,
        intervention_domain,
        true_objective_values,
        all_causal_effects,
    )


def setup_chain(T: int = 1):
    from dcbo.utils.sem_utils.toy_sems import ChainSEM

    SEM = ChainSEM()
    init_sem = SEM.static()
    sem = SEM.dynamic()

    # Build custom graph for chain structure: X->Z, {X,W,Z}->Y
    slice_nodes = ["X", "W", "Z", "Y"]
    G = MultiDiGraph()
    G.T = T
    
    # Add nodes for each time index
    for t in range(T):
        for n in slice_nodes:
            G.add_node(f"{n}_{t}")
    
    # Add within-slice (emission) edges per chain SEM: X->Z, {X,W,Z}->Y
    for t in range(T):
        G.add_edge(f"X_{t}", f"Z_{t}")  # Z depends on X
        G.add_edge(f"X_{t}", f"Y_{t}")  # Y depends on X
        G.add_edge(f"W_{t}", f"Y_{t}")  # Y depends on W
        G.add_edge(f"Z_{t}", f"Y_{t}")  # Y depends on Z
    
    # Add temporal (transition) edges
    if T > 1:
        for t in range(T - 1):
            for n in slice_nodes:
                G.add_edge(f"{n}_{t}", f"{n}_{t+1}")

    exploration_sets = [("Z",), ("W",)]
    intervention_domain = {"Z": [-1, 1], "W": [-1, 1]}
    # Precomputing ground truth is optional; skip by default
    true_objective_values = None
    all_causal_effects = None

    return (
        init_sem,
        sem,
        G,
        G,
        exploration_sets,
        intervention_domain,
        true_objective_values,
        all_causal_effects,
    )


def setup_ecology(T: int = 1):
    from dcbo.utils.sem_utils.toy_sems import EcologySEM

    SEM = EcologySEM()
    init_sem = SEM.static()
    sem = SEM.dynamic()

    # Build ecology DAG (sem/ecology_sem_equations.json):
    # E,S,N,T exogenous; X<-E; C<-N; L<-C; P<-X; D<-X; O<-E,S,X; Y<-L,N,P,O
    slice_nodes = ["E", "S", "N", "T", "X", "C", "L", "P", "D", "O", "Y"]
    G = MultiDiGraph()
    G.T = T

    # Add nodes for each time index
    for t in range(T):
        for n in slice_nodes:
            G.add_node(f"{n}_{t}")

    # Add within-slice (emission) edges per ecology SEM
    for t in range(T):
        # X (pCO2) depends on E (Tem)
        G.add_edge(f"E_{t}", f"X_{t}")
        # C (Chl-a) depends on N (Nut)
        G.add_edge(f"N_{t}", f"C_{t}")
        # L (Light) depends on C
        G.add_edge(f"C_{t}", f"L_{t}")
        # P (pHsw) depends on X
        G.add_edge(f"X_{t}", f"P_{t}")
        # D (DIC) depends on X
        G.add_edge(f"X_{t}", f"D_{t}")
        # O (Omega_A) depends on E, S, X
        G.add_edge(f"E_{t}", f"O_{t}")
        G.add_edge(f"S_{t}", f"O_{t}")
        G.add_edge(f"X_{t}", f"O_{t}")
        # Y (NEC) depends on L, N, P, O
        G.add_edge(f"L_{t}", f"Y_{t}")
        G.add_edge(f"N_{t}", f"Y_{t}")
        G.add_edge(f"P_{t}", f"Y_{t}")
        G.add_edge(f"O_{t}", f"Y_{t}")

    # Add temporal (transition) edges
    if T > 1:
        for t in range(T - 1):
            for n in slice_nodes:
                G.add_edge(f"{n}_{t}", f"{n}_{t+1}")

    # Manipulable variables and domains from configs/ecology.yaml
    exploration_sets = [("N",), ("O",), ("C",), ("T",), ("D",)]
    intervention_domain = {
        "N": [-2.0, 5.0],
        "O": [2.0, 4.0],
        "C": [0.0, 1.0],
        "T": [2200.0, 2500.0],
        "D": [1950.0, 2100.0],
    }
    true_objective_values = None
    all_causal_effects = None

    return (
        init_sem,
        sem,
        G,
        G,
        exploration_sets,
        intervention_domain,
        true_objective_values,
        all_causal_effects,
    )


def setup_protein(T: int = 1):
    """Build the protein DAG matching CBO_Benchmark/sem/protein_sem_equations.json.

    Topology (per-slice): PKC (exogenous) -> PKA -> {Raf, Mek, P38, Jnk, Akt} -> Erk.
    Intervention vars per CBO_Benchmark/configs/protein.yaml: PKC, PKA, Mek, Akt.
    """
    from dcbo.utils.sem_utils.toy_sems import ProteinSEM

    SEM = ProteinSEM()
    init_sem = SEM.static()
    sem = SEM.dynamic()

    slice_nodes = ["PKC", "PKA", "Raf", "Mek", "P38", "Jnk", "Akt", "Erk"]
    G = MultiDiGraph()
    G.T = T

    for t in range(T):
        for n in slice_nodes:
            G.add_node(f"{n}_{t}")

    # Within-slice edges (per the protein SEM dependencies).
    for t in range(T):
        # PKA depends on PKC
        G.add_edge(f"PKC_{t}", f"PKA_{t}")
        # Raf depends on PKC, PKA
        G.add_edge(f"PKC_{t}", f"Raf_{t}")
        G.add_edge(f"PKA_{t}", f"Raf_{t}")
        # Mek depends on PKC, PKA, Raf
        G.add_edge(f"PKC_{t}", f"Mek_{t}")
        G.add_edge(f"PKA_{t}", f"Mek_{t}")
        G.add_edge(f"Raf_{t}", f"Mek_{t}")
        # P38 depends on PKC, PKA
        G.add_edge(f"PKC_{t}", f"P38_{t}")
        G.add_edge(f"PKA_{t}", f"P38_{t}")
        # Jnk depends on PKC, PKA
        G.add_edge(f"PKC_{t}", f"Jnk_{t}")
        G.add_edge(f"PKA_{t}", f"Jnk_{t}")
        # Akt depends on PKA
        G.add_edge(f"PKA_{t}", f"Akt_{t}")
        # Erk depends on PKA, Mek
        G.add_edge(f"PKA_{t}", f"Erk_{t}")
        G.add_edge(f"Mek_{t}", f"Erk_{t}")

    # Temporal (transition) edges
    if T > 1:
        for t in range(T - 1):
            for n in slice_nodes:
                G.add_edge(f"{n}_{t}", f"{n}_{t+1}")

    # Default exploration sets / domains from CBO_Benchmark/configs/protein.yaml
    exploration_sets = [("PKC",), ("PKA",), ("Mek",), ("Akt",)]
    intervention_domain = {
        "PKC": [0.5, 106.5],
        "PKA": [1.45, 4491.5],
        "Mek": [0.5, 389.5],
        "Akt": [1.2, 3555.5],
    }
    true_objective_values = None
    all_causal_effects = None

    return (
        init_sem,
        sem,
        G,
        G,
        exploration_sets,
        intervention_domain,
        true_objective_values,
        all_causal_effects,
    )


def setup_epidemiology(T: int = 1):
    from dcbo.utils.sem_utils.toy_sems import EpidemiologySEM

    SEM = EpidemiologySEM()
    init_sem = SEM.static()
    sem = SEM.dynamic()

    # Build epidemiology DAG: B, T (exogenous) -> L -> R -> Y
    slice_nodes = ["B", "T", "L", "R", "Y"]
    G = MultiDiGraph()
    G.T = T
    
    # Add nodes for each time index
    for t in range(T):
        for n in slice_nodes:
            G.add_node(f"{n}_{t}")
    
    # Add within-slice (emission) edges per epidemiology SEM
    for t in range(T):
        # L depends on T
        G.add_edge(f"T_{t}", f"L_{t}")
        # R depends on L and T
        G.add_edge(f"L_{t}", f"R_{t}")
        G.add_edge(f"T_{t}", f"R_{t}")
        # Y depends on T, L, R, B
        G.add_edge(f"T_{t}", f"Y_{t}")
        G.add_edge(f"L_{t}", f"Y_{t}")
        G.add_edge(f"R_{t}", f"Y_{t}")
        G.add_edge(f"B_{t}", f"Y_{t}")
    
    # Add temporal (transition) edges
    if T > 1:
        for t in range(T - 1):
            for n in slice_nodes:
                G.add_edge(f"{n}_{t}", f"{n}_{t+1}")

    # Default exploration sets from epidemiology.yaml
    exploration_sets = [("L",), ("B",)]
    intervention_domain = {
        "L": [0.479, 801.787],
        "B": [-0.994, 0.999],
    }
    true_objective_values = None
    all_causal_effects = None

    return (
        init_sem,
        sem,
        G,
        G,
        exploration_sets,
        intervention_domain,
        true_objective_values,
        all_causal_effects,
    )

