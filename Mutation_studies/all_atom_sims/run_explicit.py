"""
Run explicit solvent simulations.
Start with a test subset of 10 mutants spanning the stability range,
then expand if results look good.

Usage:
  python run_explicit.py --mode test    # 10 mutants (WT + 4 stable + 4 unstable + 1 mid)
  python run_explicit.py --mode full    # all 50 mutants
"""

import os
import subprocess
import argparse
import pickle
import pandas as pd
import numpy as np

PKL_PATH = "mutant_dataset.pkl"
SEL_CSV  = "selected_mutants.csv"

def get_test_subset():
    """
    Select 10 mutants that maximally span the experimental
    stability range — gives the best signal for validation.
    """
    sel = pd.read_csv(SEL_CSV)
    sel_sorted = sel.sort_values('y')

    subset = pd.concat([
        sel_sorted.head(4),                          # 4 most unstable
        sel_sorted.iloc[[len(sel_sorted)//2 - 1,
                         len(sel_sorted)//2]],       # 2 middle
        sel_sorted.tail(4),                          # 4 most stable
    ]).drop_duplicates()

    names = ["1PGA"] + subset['name'].tolist()  # always include WT
    print("Test subset selected:")
    for n in names:
        row = sel[sel['name']==n]
        y = row['y'].values[0] if len(row) else 'WT'
        print(f"  {n:<8}  y={y}")
    return names


def run_simulation(mut, rep=1, timeout_hours=6):
    """Run one explicit solvent simulation."""
    print(f"\n{'='*50}")
    print(f"Starting {mut} rep {rep}")
    print(f"{'='*50}")

    try:
        result = subprocess.run(
            ["python", "simulate_explicit.py", mut, str(rep)],
            check=True,
            timeout=timeout_hours * 3600
        )
        print(f"Done: {mut} rep {rep}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAILED: {mut} rep {rep} — {e}")
        return False
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {mut} rep {rep} exceeded {timeout_hours}h")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["pilot", "test", "full"],
        default="pilot"
    )

    parser.add_argument(
        "--reps",
        type=int,
        default=3,
        help="Number of replicas per mutant"
    )

    args = parser.parse_args()

    if args.mode == "pilot":

        mutant_list = [
            "1PGA",
            "Y45S",
            "E27L"
        ]

        print("\nRunning PILOT set")
        print(f"Mutants : {mutant_list}")
        print(f"Replicas: {args.reps}")

    elif args.mode == "test":

        mutant_list = get_test_subset()

        print(f"\nRunning TEST subset")
        print(f"Mutants : {len(mutant_list)}")
        print(f"Replicas: {args.reps}")

    else:

        with open(PKL_PATH, "rb") as f:
            pkl_df = pickle.load(f)

        mutant_list = ["1PGA"] + pkl_df["name"].tolist()

        print(f"\nRunning FULL set")
        print(f"Mutants : {len(mutant_list)}")
        print(f"Replicas: {args.reps}")

    failed = []

    total_jobs = len(mutant_list) * args.reps
    current_job = 0

    for mut in mutant_list:

        for rep in range(1, args.reps + 1):

            current_job += 1

            print(
                f"\n[{current_job}/{total_jobs}] "
                f"{mut} rep{rep}"
            )

            success = run_simulation(
                mut,
                rep=rep
            )

            if not success:
                failed.append(
                    f"{mut}_rep{rep}"
                )

    print("\n" + "="*60)

    if failed:

        print(f"Failed jobs: {len(failed)}")

        for item in failed:
            print("  ", item)

    else:
        print("All jobs completed successfully.")