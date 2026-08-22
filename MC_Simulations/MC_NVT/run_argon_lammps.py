import subprocess
import os

input_dir = "/home/feynman/projects/codes/MC_Simulations/MC_NVT/argon_sim"
run_dir = "/data/argon_sim/run2"

nprocs = 10

os.makedirs(run_dir, exist_ok=True)

with open(f"{run_dir}/run_output.txt", "w") as f:
        process = subprocess.Popen(
        [
            "mpirun",
            "-np", str(nprocs),
            "/home/feynman/software/lammps/build/lmp",
            "-sf", "gpu",
            "-pk", "gpu", "1", "neigh", "yes",
            "-in", f"{input_dir}/input.argon2",
            "-log", "argon.log"
        ], 
        cwd=run_dir, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True
    )
        for line in process.stdout:
            print(line, end=" ")
            f.write(line)
        process.wait()