import subprocess
import os

input_dir = "/home/feynman/projects/codes/MC Simulations/MC_NVT/argon_sim"
run_dir = "/data/argon_sim/run1"

os.makedirs(run_dir, exist_ok=True)

with open(f"{run_dir}/run_output.txt", "w") as f:
    subprocess.run([
        "/home/feynman/software/lammps/build/lmp",
        "-sf", "gpu",
        "-pk", "gpu", "1",
        "-in", f"{input_dir}/input.Argon",
        "-log", "argon.log"
    ], 
    cwd=run_dir, 
    stdout=f, 
    stderr=subprocess.STDOUT, 
    check=True)