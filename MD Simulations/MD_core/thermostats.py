import torch

class LangevinThermostat:

    def __init__(self, temperature, gamma, dt, kB=1.0):
        self.temperature = temperature
        self.gamma = gamma
        self.dt = dt
        self.kB = kB

    def apply(self, system):

        v = system.velocities
        m = system.masses[:, None]

        # friction term
        friction = -self.gamma * v

        # random noise
        sigma = torch.sqrt(2 * self.gamma * self.kB * self.temperature / m)
        noise = sigma * torch.randn_like(v) * (self.dt ** 0.5)
        
        # update velocities
        system.velocities += (friction / m) * self.dt + noise