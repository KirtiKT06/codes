import torch

def maxwell_boltzmann(masses, temperature, device, kB=1.0):

    N = masses.shape[0]

    std = torch.sqrt(kB * temperature / masses)

    velocities = torch.randn(N,3, device=device) * std[:,None]

    return velocities

def remove_com_drift(velocities, masses):

    total_momentum = torch.sum(masses[:,None] * velocities, dim=0)

    total_mass = torch.sum(masses)

    v_com = total_momentum / total_mass

    velocities = velocities - v_com

    return velocities

def rescale_temperature(velocities, masses, target_T, kB=1.0):

    KE = 0.5 * torch.sum(masses[:,None] * velocities**2)

    N = masses.shape[0]

    current_T = 2*KE/(3*N*kB)

    scale = torch.sqrt(target_T/current_T)

    velocities = velocities * scale

    return velocities