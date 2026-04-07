import subprocess

work_dir = "/home3/kelvin/argon_sim"

with open(f"{work_dir}/run_output.txt", "w") as f:
    subprocess.run([
        "/home/kelvin/lammps/build_gpu/lmp",
        "-sf", "gpu",
        "-pk", "gpu", "1",
        "-in", "input.Argon",
        "-log", "argon.log"
    ], cwd=work_dir, stdout=f, stderr=subprocess.STDOUT, check=True)