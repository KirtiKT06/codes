import torch

class CellList:

    def __init__(self, box, cutoff):

        self.box = box
        self.cutoff = cutoff
        self.n_cells = torch.floor(box / cutoff).int()

    def build(self, positions):

        cell_indices = torch.floor(
            positions / self.cutoff
        ).int()
        cell_indices = torch.clamp(cell_indices, min=0)
        return cell_indices