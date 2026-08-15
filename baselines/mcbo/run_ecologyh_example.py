#!/usr/bin/env python3
"""
Example script to run MCBO experiments with the EcologyH environment.

Usage:
    python run_ecologyh_example.py

This script demonstrates how to run the EcologyH causal environment with MCBO.
You can also use the runner.py script with command line arguments:
    python scripts/runner.py -e ecologyh -a MCBO -s 42
"""

import subprocess
import sys
import os

def run_ecologyh_experiment():
    """Run an EcologyH experiment using the runner script."""
    
    print("🌿 Running EcologyH MCBO Experiment")
    print("=" * 50)
    
    # Example parameters
    algorithm = "MCBO"  # Can be: Random, UCB, EI, MCBO, etc.
    seed = 42
    environment = "ecologyh"
    beta = 15.0  # Higher beta for more exploration in complex space
    
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
        "--initial_obs_samples", "15",  # More initial samples for complex environment
        "--initial_int_samples", "3",
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
    print("EcologyH Environment Setup Complete! 🎉")
    print()
    print("The EcologyH causal environment has been successfully configured with:")
    print("• Variables: 11 total (L, E, N, X, S, P, T, D, O, C, Y)")
    print("• Causal structure: L → E → {X,S,P} → T → D → O → C → Y")
    print("                                      ↑")
    print("                                      N")
    print("• Interventional variables: L, E, X, S, P (5 variables)")
    print("• Interventional domains:")
    print("  - L: [378.73, 6109.3]")
    print("  - E: [18.28, 29.5]")
    print("  - X: [306.9025, 535.2337]")
    print("  - S: [36.169, 36.8806]")
    print("  - P: [7.9263, 8.1414]")
    print("• Target: Y (minimize)")
    print("• Optimal strategy: Set all interventional variables to minimum values")
    print("• Theoretical optimum: ~-1.397 (in maximization terms)")
    print()
    
    # Ask user if they want to run an example
    response = input("Would you like to run an EcologyH experiment? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        success = run_ecologyh_experiment()
        if success:
            print("\n🎉 All done! You can now run EcologyH experiments using:")
            print("    python scripts/runner.py -e ecologyh -a [ALGORITHM] -s [SEED]")
            print("    ./run_ecologyh.sh")
    else:
        print("\n✅ Setup complete! You can run EcologyH experiments using:")
        print("    python scripts/runner.py -e ecologyh -a [ALGORITHM] -s [SEED]")
        print("    ./run_ecologyh.sh")
        print("\nAvailable algorithms: Random, UCB, EI, qEI, MCBO, etc.")
        print("\nNote: EcologyH is a complex environment with 11 variables and 5 interventional")
        print("      variables. Consider using higher exploration (beta=15+) and more initial")
        print("      samples for better performance.")
