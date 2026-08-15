import argparse
import warnings
import os
from typing import Tuple, Optional, List, Dict
# Suppress benign GPy kernel RuntimeWarnings (overflow/invalid during intermediate steps)
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"GPy\.kern\..*")
from itertools import combinations

from dcbo.experimental.experiments import run_methods_replicates
from dcbo.utils.sem_utils.sem_estimate import build_sem_hat
from dcbo.utils.sem_utils.toy_sems import (
    StationaryDependentSEM,
    StationaryIndependentSEM,
    LinearMultipleChildrenSEM,
    StationaryDependentMultipleChildrenSEM,
    NonStationaryDependentSEM,
    PISHCAT_SEM,
)
from dcbo.examples.example_setups import (
    setup_stat_scm,
    setup_ind_scm,
    setup_linear_multiple_children_scm,
    setup_stat_multiple_children_scm,
    setup_nonstat_scm,
    setup_PISHCAT,
    setup_complete_graph,
    setup_psa,
    setup_psa_cdc,
    setup_diabetes,
    setup_toygraph,
    setup_chain,
    setup_ecology,
    setup_epidemiology,
    setup_protein,
)


def select_setup(name: str, T: int):
    name = name.lower()
    if name == "stat":
        # (init_sem, sem, G_view, G, exploration_sets, intervention_domain, true_objective_values, optimal_interventions, all_causal_effects)
        _, _, _, G, exploration_sets, intervention_domain, true_objective_values, *_ = setup_stat_scm(T=T)
        sem_class = StationaryDependentSEM
        base_target = "Y"
        return sem_class, G, exploration_sets, intervention_domain, true_objective_values, base_target
    elif name == "synthetic_2":
        # Alias of toygraph/stat with StationaryDependentSEM; domain/ES can be overridden via --config
        _, _, _, G, exploration_sets, intervention_domain, true_objective_values, *_ = setup_toygraph(T=T)
        from dcbo.utils.sem_utils.toy_sems import StationaryDependentSEM
        sem_class = StationaryDependentSEM
        base_target = "Y"
        return sem_class, G, exploration_sets, intervention_domain, true_objective_values, base_target
    elif name == "ind":
        # (init_sem, sem, G_view, G, exploration_sets, intervention_domain, true_objective_values)
        _, _, _, G, exploration_sets, intervention_domain, true_objective_values = setup_ind_scm(T=T)
        sem_class = StationaryIndependentSEM
        base_target = "Y"
        return sem_class, G, exploration_sets, intervention_domain, true_objective_values, base_target
    elif name == "linmc":
        # (init_sem, sem, dag_view, G, exploration_sets, intervention_domain, true_objective_values, optimal_interventions, all_causal_effects)
        _, _, _, G, exploration_sets, intervention_domain, true_objective_values, *_ = (
            setup_linear_multiple_children_scm(T=T)
        )
        sem_class = LinearMultipleChildrenSEM
        base_target = "Y"
        return sem_class, G, exploration_sets, intervention_domain, true_objective_values, base_target
    elif name == "statmc":
        # (init_sem, sem, dag_view, G, exploration_sets, intervention_domain, true_objective_values, optimal_interventions, all_causal_effects)
        _, _, _, G, exploration_sets, intervention_domain, true_objective_values, *_ = (
            setup_stat_multiple_children_scm(T=T)
        )
        sem_class = StationaryDependentMultipleChildrenSEM
        base_target = "Y"
        return sem_class, G, exploration_sets, intervention_domain, true_objective_values, base_target
    elif name == "nonstat":
        # (init_sem, sem, dag_view, dag, exploration_sets, intervention_domain, true_objective_values, all_causal_effects)
        _, _, _, G, exploration_sets, intervention_domain, true_objective_values, *_ = setup_nonstat_scm(T=T)
        sem_class = NonStationaryDependentSEM
        base_target = "Y"
        return sem_class, G, exploration_sets, intervention_domain, true_objective_values, base_target
    elif name == "pishcat":
        # (init_sem, sem, dag_view, G, exploration_sets, intervention_domain, true_objective_values, optimal_interventions, all_causal_effects)
        _, _, _, G, exploration_sets, intervention_domain, true_objective_values, *_ = setup_PISHCAT(T=T)
        sem_class = PISHCAT_SEM
        base_target = "T"
        return sem_class, G, exploration_sets, intervention_domain, true_objective_values, base_target
    elif name == "complete":
        # (init_sem, sem, G_view, G, exploration_sets, intervention_domain, true_objective_values, all_causal_effects)
        _, _, _, G, exploration_sets, intervention_domain, true_objective_values, *_ = setup_complete_graph(T=T)
        # SEM provided directly by setup
        from dcbo.utils.sem_utils.toy_sems import CompleteGraphSEM
        sem_class = CompleteGraphSEM
        base_target = "Y"
        return sem_class, G, exploration_sets, intervention_domain, true_objective_values, base_target
    elif name == "psa":
        _, _, _, G, exploration_sets, intervention_domain, true_objective_values, *_ = setup_psa(T=T)
        from dcbo.utils.sem_utils.toy_sems import PSA_SEM
        sem_class = PSA_SEM
        base_target = "PSA"
        return sem_class, G, exploration_sets, intervention_domain, true_objective_values, base_target
    elif name == "psa-cdc":
        _, _, _, G, exploration_sets, intervention_domain, true_objective_values, *_ = setup_psa_cdc(T=T)
        from dcbo.utils.sem_utils.toy_sems import PSA_CDC_SEM
        sem_class = PSA_CDC_SEM
        base_target = "PSA"
        return sem_class, G, exploration_sets, intervention_domain, true_objective_values, base_target
    elif name == "diabetes":
        _, _, _, G, exploration_sets, intervention_domain, true_objective_values, *_ = setup_diabetes(T=T)
        from dcbo.utils.sem_utils.toy_sems import DIABETES_SEM
        sem_class = DIABETES_SEM
        base_target = "Outcome"
        return sem_class, G, exploration_sets, intervention_domain, true_objective_values, base_target
    elif name == "toygraph":
        _, _, _, G, exploration_sets, intervention_domain, true_objective_values, *_ = setup_toygraph(T=T)
        from dcbo.utils.sem_utils.toy_sems import StationaryDependentSEM
        sem_class = StationaryDependentSEM
        base_target = "Y"
        return sem_class, G, exploration_sets, intervention_domain, true_objective_values, base_target
    elif name == "chain":
        _, _, _, G, exploration_sets, intervention_domain, true_objective_values, *_ = setup_chain(T=T)
        from dcbo.utils.sem_utils.toy_sems import ChainSEM
        sem_class = ChainSEM
        base_target = "Y"
        return sem_class, G, exploration_sets, intervention_domain, true_objective_values, base_target
    elif name == "ecology":
        _, _, _, G, exploration_sets, intervention_domain, true_objective_values, *_ = setup_ecology(T=T)
        from dcbo.utils.sem_utils.toy_sems import EcologySEM
        sem_class = EcologySEM
        base_target = "Y"
        return sem_class, G, exploration_sets, intervention_domain, true_objective_values, base_target
    elif name == "epidemiology":
        _, _, _, G, exploration_sets, intervention_domain, true_objective_values, *_ = setup_epidemiology(T=T)
        from dcbo.utils.sem_utils.toy_sems import EpidemiologySEM
        sem_class = EpidemiologySEM
        base_target = "Y"
        return sem_class, G, exploration_sets, intervention_domain, true_objective_values, base_target
    elif name == "protein":
        _, _, _, G, exploration_sets, intervention_domain, true_objective_values, *_ = setup_protein(T=T)
        from dcbo.utils.sem_utils.toy_sems import ProteinSEM
        sem_class = ProteinSEM
        base_target = "Erk"
        return sem_class, G, exploration_sets, intervention_domain, true_objective_values, base_target
    else:
        raise ValueError(f"Unknown setup '{name}'.")


def default_change_points(T: int) -> Optional[list]:
    # Default single change point at t=1 if valid
    if T >= 2:
        return [False] + [True] + [False] * (T - 2)
    return None


def parse_args():
    p = argparse.ArgumentParser(description="Run DCBO-only experiments")
    p.add_argument("--setup", type=str, default="stat", choices=["stat", "nonstat", "ind", "linmc", "statmc", "pishcat", "complete", "psa", "psa-cdc", "diabetes", "toygraph", "synthetic_2", "chain", "ecology", "epidemiology", "protein"], help="Example setup")
    p.add_argument("--T", type=int, default=3, help="Number of time steps")
    p.add_argument("--n-obs", type=int, default=100, help="Observational samples to draw")
    p.add_argument("--trials", type=int, default=3, help="Trials per time step")
    p.add_argument("--reps", type=int, default=1, help="Number of replicates")
    p.add_argument("--n-restart", type=int, default=1, help="GP optimization restarts")
    p.add_argument("--cost-structure", type=int, default=1, help="Intervention cost structure id")
    p.add_argument("--online", action="store_true", help="Enable online mode (resample obs over time)")
    p.add_argument("--use-di", action="store_true", help="Use data integration (forward propagation)")
    p.add_argument("--transfer-hp-o", action="store_true", help="Transfer obs GP hyperparameters across t")
    p.add_argument("--transfer-hp-i", action="store_true", help="Transfer interventional GP hyperparameters across t")
    p.add_argument("--no-hp-i-prior", action="store_true", help="Disable Gamma prior on interventional GP variance")
    p.add_argument("--mc", action="store_true", help="Use Monte Carlo in target evaluation")
    p.add_argument("--debug", action="store_true", help="Enable debug plotting")
    p.add_argument("--save-data", action="store_true", help="Persist results to ../data/<folder>/...")
    p.add_argument("--folder", type=str, default=None, help="Subfolder for saving results (when --save-data)")
    p.add_argument("--num-anchor-points", type=int, default=100, help="Anchor points for acquisition eval")
    p.add_argument("--sample-anchor-points", action="store_true", help="Sample anchor points stochastically")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--target", type=str, default=None, help="Base target variable (default per setup)")
    p.add_argument("--config", type=str, default=None, help="Path to YAML/JSON config to override setup (interventions, domains, n_obs, etc.)")
    return p.parse_args()


def _build_es_from_config(interventions: List[str], num_intervention: Optional[int]) -> List[tuple]:
    if not interventions:
        return []
    if num_intervention is None:
        # default: singles and full tuple
        es = [(v,) for v in interventions]
        if len(interventions) > 1:
            es.append(tuple(interventions))
        return es
    else:
        return [tuple(c) for c in combinations(interventions, num_intervention)]


def main():
    args = parse_args()

    sem_class, G, exploration_sets, intervention_domain, ground_truth, default_target = select_setup(
        args.setup, args.T
    )

    base_target_variable = args.target if args.target else default_target

    # Some methods require change-points (non-stationary)
    change_points = default_change_points(args.T) if args.setup == "nonstat" else None

    # Optional config override (YAML/JSON)
    manipulative_variables_override: Optional[List[str]] = None
    if args.config:
        cfg: Dict = {}
        ext = os.path.splitext(args.config)[1].lower()
        if ext in (".yml", ".yaml"):
            try:
                import yaml  # type: ignore
            except ModuleNotFoundError as e:
                raise SystemExit(
                    "Config file appears to be YAML, but PyYAML is not installed. Install with 'conda install -n DCBO pyyaml' or 'pip install pyyaml', or provide a JSON config."
                )
            with open(args.config, "r") as f:
                cfg = yaml.safe_load(f)
        else:
            import json
            with open(args.config, "r") as f:
                cfg = json.load(f)

        # Override domain
        if isinstance(cfg.get("interventional_domain"), dict):
            # Expect shape: {"X": [-5,5], "Z": [-5,20]}
            intervention_domain = {k: list(v) for k, v in cfg["interventional_domain"].items()}

        # Override exploration sets based on interventions and num_intervention
        interventions = cfg.get("intervention") or cfg.get("interventions")
        num_intervention = cfg.get("num_intervention")
        if interventions:
            es = _build_es_from_config(list(interventions), num_intervention)
            if es:
                exploration_sets = es
                ground_truth = None  # avoid mismatch with precomputed ground truth
            manipulative_variables_override = list(interventions)

            # Reorder intervention_domain to match interventions order (Root asserts equality of lists)
            if isinstance(intervention_domain, dict):
                ordered_domain = {}
                for var in manipulative_variables_override:
                    if var in intervention_domain:
                        ordered_domain[var] = list(intervention_domain[var])
                # Include any stray keys (shouldn't happen) preserving their order
                for k in intervention_domain.keys():
                    if k not in ordered_domain:
                        ordered_domain[k] = list(intervention_domain[k])
                intervention_domain = ordered_domain

        # Override target and observations
        if cfg.get("target"):
            base_target_variable = str(cfg["target"]).strip()
        if cfg.get("num_observations"):
            args.n_obs = int(cfg["num_observations"])  # override observational samples

        # Optionally override task
        if cfg.get("task") in ("min", "max"):
            task_override = cfg["task"]
        else:
            task_override = "min"
    else:
        task_override = "min"

    results = run_methods_replicates(
        G=G,
        sem=sem_class,
        make_sem_estimator=build_sem_hat,
        intervention_domain=intervention_domain,
        methods_list=["DCBO"],
        obs_samples=None,
        exploration_sets=exploration_sets,
        base_target_variable=base_target_variable,
        ground_truth=ground_truth,
        total_timesteps=args.T,
        reps=args.reps,
        number_of_trials=args.trials,
        number_of_trials_BO_ABO=None,
        n_restart=args.n_restart,
        save_data=args.save_data,
        n_obs=args.n_obs,
        cost_structure=args.cost_structure,
        optimal_assigned_blankets=None,
        debug_mode=args.debug,
        use_mc=args.mc,
        online=args.online,
        concat=False,
        use_di=args.use_di,
        transfer_hp_o=args.transfer_hp_o,
        transfer_hp_i=args.transfer_hp_i,
        hp_i_prior=not args.no_hp_i_prior,
        estimate_sem=True,
        folder=args.folder,
        num_anchor_points=args.num_anchor_points,
        n_obs_t=None,
        sample_anchor_points=args.sample_anchor_points,
        seed=args.seed,
        controlled_experiment=True,
        noise_experiment=False,
        args_sem=None,
        manipulative_variables=manipulative_variables_override,
        change_points=change_points,
        task=task_override,
    )

    # Minimal textual summary
    dcbo_models = results.get("DCBO", [])
    print(f"Completed DCBO with {len(dcbo_models)} replicate(s).")
    if dcbo_models:
        m0 = dcbo_models[0]
        print("Per-trial cost (t=0):", m0.per_trial_cost[0])
        print("Optimal target values during trials (t=0):", m0.optimal_outcome_values_during_trials[0])

    # Save best-so-far target value per trial to CSV
    try:
        import csv
        from pathlib import Path

        if args.save_data and args.folder:
            out_dir = Path(__file__).resolve().parent.parent / "data" / args.folder
        else:
            out_dir = Path(__file__).resolve().parent / "results"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_csv = out_dir / (
            f"dcbo_best_so_far_seed{args.seed}_T{args.T}_trials{args.trials}_reps{args.reps}.csv"
        )

        with out_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["method", "replicate", "time_index", "trial_index", "best_so_far_value", "y"]) 
                
            for rep_idx, model in enumerate(dcbo_models):
                for t_idx in range(model.T):
                    bsf_list = model.optimal_outcome_values_during_trials[t_idx]
                    for trial_idx, val in enumerate(bsf_list):
                        # y at each trial: first trial (trial_idx==0) is observation; write empty or NaN
                        try:
                            if trial_idx == 0:
                                y_val = ""
                            else:
                                # outcome_values[t] contains [initial, observed, y(trial1), y(trial2), ...]
                                y_seq = model.outcome_values[t_idx]
                                y_val = y_seq[trial_idx + 1] if len(y_seq) > trial_idx + 1 else ""
                        except Exception:
                            y_val = ""
                        writer.writerow(["DCBO", rep_idx, t_idx, trial_idx, val, y_val])

        print(f"Saved best-so-far CSV: {out_csv}")
    except Exception as e:
        print(f"Warning: failed to write best-so-far CSV due to: {e}")


if __name__ == "__main__":
    main()


