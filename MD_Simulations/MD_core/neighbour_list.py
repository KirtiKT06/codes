import torch
from MD_core.boundary import minimum_image
from MD_core.cell_list import CellList

class NeighborList:

    def __init__(self, cutoff, skin, box):

        self.cutoff = cutoff
        self.skin = skin
        self.list_cutoff = cutoff + skin
        self.cell_list = CellList(box, self.list_cutoff)
        self.neighbors = None
        self.last_positions = None


    def build(self, system):

        pos = system.positions
        box = system.box
        cell_ids = self.cell_list.build(pos)
        diff = pos[:,None,:] - pos[None,:,:]
        diff = minimum_image(diff, box)

        r = torch.norm(diff, dim=-1)

        mask = (r < self.list_cutoff)
        mask.fill_diagonal_(False)

        self.neighbors = mask
        self.last_positions = pos.clone()


    def needs_update(self, system):

        disp = system.positions - self.last_positions

        max_disp = torch.max(torch.norm(disp, dim=1))

        return max_disp > self.skin * 0.5