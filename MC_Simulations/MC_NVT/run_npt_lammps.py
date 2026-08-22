import subprocess

p = 0.05
pressures = [p + 0.01*i for i in range(20)]
# pressures = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]

for p in pressures:
    print(f"Running for P = {p:.2f}")
    

    subprocess.run([
    "lmp",
    "-var", "P", str(p),
    "-in", "npt_input.lj",
    "-log", f"/data/lammps_log/log_{p:.2f}.lammps"
    ])