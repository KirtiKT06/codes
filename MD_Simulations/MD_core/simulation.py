from MD_core.neighbour_list import NeighborList
from MD_core.observables import temperature

class Simulation:

    def __init__(self, system, potential, integrator, box=6.8,
                 cutoff=2.5, skin=0.3, logger=None, trajectory=None, thermo=None,
                 thermostat=None):
        
        # create objects of the class
        self.system = system
        self.potential = potential
        self.integrator = integrator
        self.logger = logger
        self.trajectory = trajectory
        self.thermo = thermo
        self.thermostat = thermostat
        self.current_step = 0

        # create neighbour list, calls MD_core/neighbour_list.py
        self.neighbour_list = NeighborList(cutoff, skin, system.box)

        # build initial list
        self.neighbour_list.build(system)

        # compute initial force; calls MD_core/potential.py
        energy, self.forces, _ = self.potential.compute(system,
                                                self.neighbour_list)


    def step(self):

        if self.thermostat is not None:
            self.thermostat.apply(self.system)

        # first half velocity update + position update
        self.integrator.first_half(self.system, self.forces)

        # rebuild neighbour list if needed
        if self.neighbour_list.needs_update(self.system):
            self.neighbour_list.build(self.system)

        # compute new forces
        energy, new_forces, virial = self.potential.compute(self.system,
                                            self.neighbour_list)
        # tail corrections
        energy += self.potential.tail_correction_energy(self.system)

        N = self.system.n_particles
        V = self.system.volume
        T = temperature(self.system)

        rho = N / V
        pressure = rho * T + virial / (3 * V)
        
        if self.logger is not None:
            self.logger.log(self.current_step, self.system, energy)
        
        if self.trajectory:
            self.trajectory.write(self.current_step, self.system)
        
        if self.thermo:
            self.thermo.print(self.current_step, self.system, energy, pressure)
        

        # second half velocity update
        self.integrator.second_half(self.system, new_forces)

        self.forces = new_forces

        self.current_step += 1

    def run(self, steps):

        for _ in range(steps):
            self.step()