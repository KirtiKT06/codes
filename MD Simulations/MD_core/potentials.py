import torch
import math
from MD_core.boundary import minimum_image

class LennardJones:

    def __init__(self, epsilon=1.0, sigma=1.0, cutoff=2.5):

        # create and initialize class objects
        self.epsilon = epsilon
        self.sigma = sigma
        self.cutoff = cutoff * self.sigma

    def compute(self, system, neighbour_list):

        #initialize the system
        pos = system.positions
        box = system.box

        mask = neighbour_list.neighbors         # (N, N)
        i, j = torch.where(mask)                # get neighbour pairs

        rij = pos[i] - pos[j]
        rij = minimum_image(rij, box)

        r = torch.norm(rij, dim=-1)
        r = torch.clamp(r, min=1e-6)

        # LJ calculations
        inv_r = self.sigma / r
        inv_r6 = inv_r**6
        inv_r12 = inv_r6**2

        pair_energy = 4 * self.epsilon * (inv_r12 - inv_r6)
        
        force_scalar = 24*self.epsilon*(2*inv_r12 - inv_r6)/r
        
        fij = force_scalar[:, None] * rij

        # initialize forces
        forces = torch.zeros_like(pos)

        #accumulate forces
        forces.index_add_(0, i, fij)
        forces.index_add_(0, j, -fij)

        total_energy = torch.sum(pair_energy) * 0.5
        virial = torch.sum(rij * fij) * 0.5

        return total_energy, forces, virial
    
    def tail_correction_energy(self, system):

        N = system.n_particles
        V = system.volume

        rho = N / V

        rc3 = (self.sigma / self.cutoff)**3
        rc9 = rc3**3

        U_tail = (8*math.pi * rho * self.epsilon * self.sigma**3/3) * (rc9/3 - rc3)

        return N * U_tail