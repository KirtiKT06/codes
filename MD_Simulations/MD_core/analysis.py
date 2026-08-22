import torch
import numpy as np
from MD_core.boundary import minimum_image

def radial_distribution(system, bins=100, r_max=None):

    pos = system.positions.detach().cpu()
    box = system.box.detach().cpu()

    N = system.n_particles
    V = system.volume

    if r_max is None:
        r_max = torch.min(box).item()/2

    diff = pos[:,None,:] - pos[None,:,:]
    diff = minimum_image(diff, box)
    r = torch.norm(diff, dim=-1)

    # take only upper triangle (avoid double counting)
    mask = torch.triu(torch.ones_like(r), diagonal=1).bool()
    r = r[mask]

    hist, edges = np.histogram(r.cpu().numpy(), bins=bins, range=(0,r_max))

    dr = edges[1] - edges[0]
    r_centers = 0.5*(edges[1:] + edges[:-1])
    rho = N / V.cpu().item()
    shell_volume = 4*np.pi*(r_centers**2)*dr
    g_r = hist / (rho * N * shell_volume)
    g_r /= g_r[-1]

    return r_centers, g_r