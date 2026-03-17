import torch

def kinetic_energy(system):

    v2 = torch.sum(system.velocities**2, dim=1)
    
    return 0.5*torch.sum(system.masses * v2)

def temperature(system, kB=1.0):

    N = system.n_particles
    KE = kinetic_energy(system)

    return 2*KE/(3*N*kB)