import torch

class System:

    def __init__(self, positions, velocities, masses, box, device):

        self.device = device
        self.positions = positions.to(device)
        self.velocities = velocities.to(device)
        self.masses = masses.to(device)
        self.box = torch.tensor(box, device=device)
        self.forces = torch.zeros_like(self.positions)

    @property
    def n_particles(self):
        return self.positions.shape[0]
    
    @property
    def volume(self):
        return torch.prod(self.box)