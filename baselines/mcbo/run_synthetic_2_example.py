#!/usr/bin/env python3
"""
Example script to run MCBO experiments with the Synthetic_2 environment.

Usage:
    python run_synthetic_2_example.py

This script demonstrates how to run the Synthetic_2 causal environment with MCBO.
You can also use the runner.py script with command line arguments:
    python scripts/runner.py -e Synthetic_2 -a Random -s 42
"""

import subprocess
import sys
import os

def run_synthetic_2_experiment():
    """Run a Synthetic_2 experiment using the runner script."""
    
    print("🚀 Running Synthetic_2 MCBO Experiment")
    print("=" * 50)
    
    # Example parameters
    algorithm = "Random"  # Can be: Random, UCB, EI, etc.
    seed = 42
    environment = "Synthetic_2"
    
    print(f"Environment: {environment}")
    print(f"Algorithm: {algorithm}")
    print(f"Seed: {seed}")
    print()
    
    # Build the command
    cmd = [
        sys.executable, "scripts/runner.py",
        "-e", environment,
        "-a", algorithm,
        "-s", str(seed),
        "--initial_obs_samples", "10",
        "--initial_int_samples", "2"
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
    results_file = f"trial_results_MCBO_{environment}_{seed}.csv"
    if os.path.exists(results_file):
        print(f"📊 Results saved to: {results_file}")
    
    return True

if __name__ == "__main__":
    print("Synthetic_2 Environment Setup Complete! 🎉")
    print()
    print("The Synthetic_2 causal environment has been successfully configured with:")
    print("• Variables: X (exogenous), Z (endogenous), Y (target)")
    print("• Causal relationships:")
    print("  - X ~ N(0,1)")
    print("  - Z = exp(-X) + U_Z, where U_Z ~ N(0,1)")
    print("  - Y = cos(Z) - exp(-Z/20) + U_Y, where U_Y ~ N(0,1)")
    print("• Interventional domains: X ∈ [-3, 2], Z ∈ [-1, 1]")
    print("• Target: Minimize Y")
    print()
    
    # Ask user if they want to run an example
    response = input("Would you like to run a quick example experiment? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        success = run_synthetic_2_experiment()
        if success:
            print("\n🎉 All done! You can now run Synthetic_2 experiments using:")
            print("    python scripts/runner.py -e Synthetic_2 -a [ALGORITHM] -s [SEED]")
    else:
        print("\n✅ Setup complete! You can run Synthetic_2 experiments using:")
        print("    python scripts/runner.py -e Synthetic_2 -a [ALGORITHM] -s [SEED]")
        print("\nAvailable algorithms: Random, UCB, EI, qEI, qKG, etc.")
