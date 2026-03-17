import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from MD_core.utils import get_device
from MD_core.system import System
from MD_core.potentials import LennardJones
from MD_core.integrator import VelocityVerlet
from MD_core.simulation import Simulation
from MD_core.initialization import maxwell_boltzmann, remove_com_drift, rescale_temperature

device = get_device()                               # calls MD_core/utils.py to get information of the device
print(f"Running on {device}")

N = 100
box = [10.0,10.0,10.0]

positions = torch.rand(N,3, device=device)*10
masses = torch.ones(N, device=device)

T = 1.0

velocities = maxwell_boltzmann(masses,T,device)
velocities = remove_com_drift(velocities,masses)
velocities = rescale_temperature(velocities,masses,T)

system = System(positions,velocities,masses,box,device)

potential = LennardJones(cutoff=2.5)
integrator = VelocityVerlet(dt=0.005)

sim = Simulation(system,potential,integrator)

fig, ax = plt.subplots()

scatter = ax.scatter([],[], s=20)

ax.set_xlim(0,10)
ax.set_ylim(0,10)

def update(frame):

    sim.run(5)

    pos = system.positions.detach().cpu().numpy()

    scatter.set_offsets(pos[:,:2])

    return scatter,

ani = FuncAnimation(fig, update, frames=500, interval=30)

plt.show()