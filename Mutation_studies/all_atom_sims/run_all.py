import os
import subprocess

PDB_DIR = "/home/feynman/projects/codes/Mutation_studies/mutant_pdbs"

pdbs = sorted([
    f for f in os.listdir(PDB_DIR)
    if f.endswith(".pdb")
])

N_REPLICAS = 1

failed = []

for pdb in pdbs:
    mut = pdb.replace(".pdb", "")
    for rep in range(1, N_REPLICAS + 1):
        print(f"Starting {mut} replica {rep}")
        try:
            subprocess.run(
                ["python", "simulate_mutant.py", mut, str(rep)],
                check=True
            )
            print(f"Done: {mut} rep {rep}")
        except subprocess.CalledProcessError as e:
            print(f"FAILED: {mut} replica {rep} — {e}")
            failed.append((mut, rep))

print("\nAll jobs attempted.")
if failed:
    print("Failed runs:")
    for mut, rep in failed:
        print(f"  {mut} rep {rep}")
else:
    print("All completed successfully.")