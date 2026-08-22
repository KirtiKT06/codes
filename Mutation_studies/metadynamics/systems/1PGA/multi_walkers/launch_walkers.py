"""
launch_walkers.py  –  spawn all walkers as parallel subprocesses.

Usage:
    python launch_walkers.py --walkers 4

Run this script from BASE_DIR (the code/data folder).
Walker stdout logs go to OUT_DIR/walker_N/stdout.log.
"""

import argparse
import os
import signal
import subprocess
import sys

# ==========================================
# CLI
# ==========================================

parser = argparse.ArgumentParser()
parser.add_argument("--walkers", type=int, default=4,
                    help="Number of walkers to launch")
parser.add_argument("--script",  type=str, default="run_meta.py",
                    help="Walker script to run (default: run_meta.py)")
args = parser.parse_args()

N_WALKERS = args.walkers

# CODE + INPUT FILES live here (where you run this script from)
BASE_DIR = "/home/feynman/projects/codes/Mutation_studies/metadynamics/WT/multi_walkers"

# ALL OUTPUT (traj, logs, COLVAR, HILLS, checkpoints) goes here
OUT_DIR  = "/data/mutation_study/metaD/WT/multi_walker"
os.makedirs(OUT_DIR, exist_ok=True)

# ==========================================
# GENERATE PLUMED TEMPLATE FIRST
# make_plumed.py lives in BASE_DIR and reads
# wt_solvated.pdb + ca_contacts_with_dist.dat
# from the same folder → cwd must be BASE_DIR
# ==========================================

print(f"Generating plumed_template.dat for {N_WALKERS} walkers …")
ret = subprocess.run(
    [sys.executable, "make_plumed.py", "--walkers", str(N_WALKERS)],
    cwd=BASE_DIR      # FIXED: was OUT_DIR — script and data files are in BASE_DIR
)
if ret.returncode != 0:
    sys.exit("make_plumed.py failed – aborting.")

# ==========================================
# LAUNCH WALKERS
# ==========================================

processes = []

for wid in range(N_WALKERS):

    # stdout.log goes to OUT_DIR/walker_N/ alongside traj, COLVAR, etc.
    walker_out_dir = os.path.join(OUT_DIR, f"walker_{wid}")
    os.makedirs(walker_out_dir, exist_ok=True)           # FIXED: was BASE_DIR

    log_path = os.path.join(walker_out_dir, "stdout.log")
    log_fh   = open(log_path, "w")

    cmd = [
        sys.executable, os.path.join(BASE_DIR, args.script),  # absolute script path
        "--walker",        str(wid),
        "--total-walkers", str(N_WALKERS),
    ]

    print(f"  Launching walker {wid}  →  {log_path}")

    proc = subprocess.Popen(
        cmd,
        cwd=BASE_DIR,          # run_meta.py reads template + PDB from here
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )

    # Stagger launches so walkers don't all hit CUDA init at the same moment.
    # 30 s is enough for one walker to finish loading the force field + PDB
    # before the next one requests GPU memory.
    if wid < N_WALKERS - 1:
        import time
        print(f"  (waiting 30 s before next walker …)")
        time.sleep(30)

    processes.append((wid, proc, log_fh))

print(f"\nAll {N_WALKERS} walkers running.  Waiting …\n")
print(f"  Logs   → {OUT_DIR}/walker_N/stdout.log")
print(f"  Traj   → {OUT_DIR}/walker_N/traj.dcd")
print(f"  COLVAR → {OUT_DIR}/walker_N/COLVAR")
print(f"  HILLS  → {OUT_DIR}/HILLS.<id>\n")

# ==========================================
# HANDLE Ctrl-C GRACEFULLY
# ==========================================

def _kill_all(sig, frame):
    print("\nInterrupt received – killing all walkers …")
    for wid, proc, fh in processes:
        proc.terminate()
    sys.exit(1)

signal.signal(signal.SIGINT,  _kill_all)
signal.signal(signal.SIGTERM, _kill_all)

# ==========================================
# WAIT FOR ALL TO FINISH
# ==========================================

failed = []

for wid, proc, fh in processes:
    rc = proc.wait()
    fh.close()
    status = "OK" if rc == 0 else f"FAILED (rc={rc})"
    print(f"  Walker {wid}: {status}")
    if rc != 0:
        failed.append(wid)

if failed:
    print(f"\nFailed walkers: {failed}")
    sys.exit(1)
else:
    print("\nAll walkers finished successfully.")