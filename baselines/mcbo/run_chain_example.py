#!/usr/bin/env python3
"""
Example script to run MCBO experiments with the Chain environment.

Usage:
    python run_chain_example.py

This script demonstrates how to run the Chain causal environment with MCBO.
You can also use the runner.py script with command line arguments:
    python scripts/runner.py -e chain -a MCBO -s 42
"""

import subprocess
import sys
import os

def run_chain_experiment():
    """Run a Chain experiment using the runner script."""
    
    print("🔗 Running Chain MCBO Experiment")
    print("=" * 50)
    
    # Example parameters
    algorithm = "MCBO"  # Can be: Random, UCB, EI, MCBO, etc.
    seed = 42
    environment = "chain"
    beta = 10.0  # Higher beta for more exploration
    
    print(f"Environment: {environment}")
    print(f"Algorithm: {algorithm}")
    print(f"Seed: {seed}")
    print(f"Beta (exploration): {beta}")
    print()
    
    # Build the command
    cmd = [
        sys.executable, "scripts/runner.py",
        "-e", environment,
        "-a", algorithm,
        "-s", str(seed),
        "-b", str(beta),
        "--initial_obs_samples", "10",
        "--initial_int_samples", "2",
        "-n", "0.0"  # No noise for cleaner results
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print()
    
    try:
        # Run the experiment
        result = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Experiment completed successfully!")
            # Extract final results from output
            lines = result.stdout.split('\n')
            for line in lines:
                if "Final Best Value:" in line or "FINAL AVERAGE GAP SCORE:" in line:
                    print(f"📊 {line.strip()}")
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

if __name__ == "__main__":
    print("Chain Environment Setup Complete! 🎉")
    print()
    print("The Chain causal environment has been successfully configured with:")
    print("• Variables: X (exog), W (exog), Z (endog), Y (target)")
    print("• Causal relationships:")
    print("  - X ~ N(0,1) (not interventional)")
    print("  - W ~ N(0,1) (interventional)")
    print("  - Z = -0.5*X + U_Z (interventional)")
    print("  - Y = -W - 3*Z*X + U_Y (target to minimize)")
    print("• Interventional domains: W ∈ [-1,1], Z ∈ [-1,1]")
    print("• Optimal strategy: W=1, Z=sign(X) to minimize Y")
    print("• Theoretical optimum: ~3.44 (in maximization terms)")
    print()
    
    # Ask user if they want to run an example
    response = input("Would you like to run a Chain experiment? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        success = run_chain_experiment()
        if success:
            print("\n🎉 All done! You can now run Chain experiments using:")
            print("    python scripts/runner.py -e chain -a [ALGORITHM] -s [SEED]")
            print("    ./run_chain.sh")
    else:
        print("\n✅ Setup complete! You can run Chain experiments using:")
        print("    python scripts/runner.py -e chain -a [ALGORITHM] -s [SEED]")
        print("    ./run_chain.sh")
        print("\nAvailable algorithms: Random, UCB, EI, qEI, MCBO, etc.")
