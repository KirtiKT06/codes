# Standard imports
import torch
from pathlib import Path
import datetime
import math
import sys

# Adding parent directory to the import path to import from MD_core modules
sys.path.append(str(Path(__file__).resolve().parents[1]))

# importing MD_core modules
from MD_core.system import System
from MD_core.potentials import LennardJones
from MD_core.integrator import VelocityVerlet
from MD_core.simulation import Simulation
from MD_core.initialization import maxwell_boltzmann
from MD_core.initialization import remove_com_drift
from MD_core.initialization import rescale_temperature
from MD_core.logger import ThermoLogger
from MD_core.utils import get_device
from MD_core.trajectory import TrajectoryWriter
from MD_core.thermo import ThermoPrinter
from MD_core.thermostats import LangevinThermostat

device = get_device()                                           # calls MD_core/utils.py to get information of the device
print(f"Running on {device}")
thermo = ThermoPrinter(interval=50)                             # calls MD_core/thermo.py

# System parameters
N = 256                                                         # number of particles       
box = 6.8                                                       # box length
T = 1.0                                                         # reduced temperature in LJ units

# creating result directory
RESULTS_ROOT = Path("/home3/kelvin/md_simulations")
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = RESULTS_ROOT / f"lj_run_{timestamp}"
run_dir.mkdir(parents=True, exist_ok=True)
print("Results will be stored in:", run_dir)

# Generate initial positions. Simple Cubic lattice here, with some random noise so particles don't start perfectly symmetric.
n_side = int(math.ceil(N ** (1/3)))
spacing = box / n_side
coords = torch.linspace(0, box - spacing, n_side, device=device)
grid = torch.stack(torch.meshgrid(coords, coords, coords, indexing="ij"), dim=-1)
positions = grid.reshape(-1,3)[:N]
positions += 0.1 * (torch.rand_like(positions) - 0.5)

# initializing masses and velocities of particles. Calls MD_core/initialization.py
masses = torch.ones(N, device=device)
velocities = maxwell_boltzmann(masses, T, device)               # initializes velocities from Maxwell-Boltzmann distribution
velocities = remove_com_drift(velocities, masses)               # ensures \sum{mv} = 0
velocities = rescale_temperature(velocities, masses, T)         # readjusts velocity so that T_exact = target_T

system = System(positions, velocities, masses, box, device)     # create system objects; calls MD_core/system.py

potential = LennardJones()                                      # create potentials; calls MD_core/potentials.py
integrator = VelocityVerlet(dt=0.0005)                          # create integrator; calls MD_core/integrator.py

thermostat = LangevinThermostat(                                # creates a thermostat; valls MD_core/thermostats.py
    temperature=1.0,
    gamma=1.0,
    dt=0.0005)

traj = TrajectoryWriter(run_dir, interval=50)                   # creates trajectory writer; calls MD_core/trajectory.py
log_file = run_dir / "thermo.csv"
logger = ThermoLogger(log_file)

sim = Simulation(system,                                        # create simulation object; calls MD_core/simultion.py
                 potential, 
                 integrator, 
                 logger=logger, 
                 trajectory=traj,
                 thermo=thermo,
                 thermostat=thermostat
                 )

# -------------------------
# Equilibration phase
# -------------------------

equil_steps = 2000
for step in range(equil_steps):
    sim.step()
print("Equilibration complete")
# switch to NVE
sim.thermostat = None
print("Switching to NVE ensemble")

# -------------------------
# Production run
# -------------------------

sim.run(steps=20000)