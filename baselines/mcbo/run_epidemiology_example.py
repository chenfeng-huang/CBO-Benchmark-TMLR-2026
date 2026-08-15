#!/usr/bin/env python3
"""
Example script to run MCBO experiments with the Epidemiology environment.

Usage:
    python run_epidemiology_example.py

This script demonstrates how to run the Epidemiology causal environment with MCBO.
The epidemiology model includes:
- Target: Y (minimize)
- Task: min
- Interventions: L, B
- Interventional domains:
  - L: [0.479, 801.787]
  - B: [-0.994, 0.999]
- Number of observations: 200

You can also use the runner.py script with command line arguments:
    python scripts/runner.py -e epidemiology -a MCBO -s 42
"""

import subprocess
import sys
import os

def run_epidemiology_experiment():
    """Run an Epidemiology experiment using the runner script."""
    
    print("🧬 Running Epidemiology MCBO Experiment")
    print("=" * 50)
    
    # Parameters based on user specification
    algorithm = "MCBO"  # Can be: Random, UCB, EI, MCBO, etc.
    seed = 42
    environment = "epidemiology"
    beta = 10.0  # Exploration parameter for UCB-based algorithms
    noise_scale = 1.0  # As specified in the causal model (Y has noise)
    
    print(f"Environment: {environment}")
    print(f"Algorithm: {algorithm}")
    print(f"Seed: {seed}")
    print(f"Beta (exploration): {beta}")
    print(f"Noise scale: {noise_scale}")
    print()
    
    print("Causal Model Details:")
    print("- Target variable: Y (minimize)")
    print("- Interventional variables: L, B")
    print("- Interventional domains:")
    print("  - L: [0.479, 801.787]")
    print("  - B: [-0.994, 0.999]")
    print("- Number of observations: 200")
    print()
    
    # Build the command
    cmd = [
        sys.executable, "scripts/runner.py",
        "-e", environment,
        "-a", algorithm,
        "-s", str(seed),
        "-b", str(beta),
        "--initial_obs_samples", "20",  # More initial samples for complex environment
        "--initial_int_samples", "5",   # More interventional samples per node
        "-n", str(noise_scale)  # Noise scale as specified
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print()
    
    try:
        # Run the experiment
        result = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Experiment completed successfully!")
            print("\nOutput:")
            print(result.stdout)
        else:
            print("❌ Experiment failed!")
            print("\nError output:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Failed to run experiment: {e}")
        return False
    
    # Check for results file
    results_file = f"trial_results_{algorithm}_{environment}_{seed}.csv"
    if os.path.exists(results_file):
        print(f"📊 Results saved to: {results_file}")
    
    return True

def run_comparison_experiments():
    """Run comparison experiments with different algorithms."""
    
    print("\n🔬 Running Comparison Experiments")
    print("=" * 50)
    
    algorithms = ["Random", "MCBO"]
    seeds = [42, 68, 123]
    
    for algo in algorithms:
        for seed in seeds:
            print(f"\nRunning {algo} with seed {seed}...")
            
            cmd = [
                sys.executable, "scripts/runner.py",
                "-e", "epidemiology",
                "-a", algo,
                "-s", str(seed),
                "-b", "10.0",
                "--initial_obs_samples", "20",
                "--initial_int_samples", "5",
                "-n", "1.0"
            ]
            
            try:
                result = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✅ {algo} (seed {seed}) completed successfully!")
                else:
                    print(f"❌ {algo} (seed {seed}) failed!")
                    print(result.stderr)
            except Exception as e:
                print(f"❌ Failed to run {algo} (seed {seed}): {e}")

if __name__ == "__main__":
    print("Epidemiology Environment Setup Complete! 🎉")
    print("\nThis environment implements a complex causal model with:")
    print("- Exogenous variables: B ~ U(-1,1), T ~ U(4,8)")
    print("- Endogenous variables: L, R, Y with complex relationships")
    print("- Target: Minimize Y")
    print("- Interventions allowed on: L, B")
    print()
    
    # Run single experiment
    success = run_epidemiology_experiment()
    
    if success:
        print("\n" + "="*50)
        print("Would you like to run comparison experiments? (y/n)")
        response = input().lower().strip()
        
        if response in ['y', 'yes']:
            run_comparison_experiments()
            print("\n🎯 All experiments completed!")
        else:
            print("\n✨ Single experiment completed!")
    
    print("\nTo run additional experiments manually, use:")
    print("python scripts/runner.py -e epidemiology -a MCBO -s <seed>")





