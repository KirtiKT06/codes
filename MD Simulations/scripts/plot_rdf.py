import torch
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import glob

from MD_core.analysis import radial_distribution
from MD_core.system import System
import numpy as np

base_dir = Path("/home3/kelvin/md_simulations")
run_dir = max(base_dir.glob("lj_run_*"), key=lambda p: p.stat().st_mtime)
files = sorted((run_dir / "trajectory").glob("frame_*.npy"))

print(f"Using run directory: {run_dir}")
print(f"Number of frames found: {len(files)}")

g_total = torch.zeros(100)
count = 0
for f in files[50:]:
    frame = np.load(f)
    positions = torch.tensor(frame, dtype=torch.float32)
    velocities = torch.zeros_like(positions)
    masses = torch.ones(positions.shape[0])

    box = torch.tensor([6.8, 6.8, 6.8])

    system = System(positions, velocities, masses, box, device="cpu")

    r, g = radial_distribution(system, bins=100)
    g_total += g
    count += 1
g_avg = g_total / count

plt.plot(r, g_avg)
plt.xlabel("r")
plt.ylabel("g(r)")
plt.title("Radial Distribution Function")
plt.show()