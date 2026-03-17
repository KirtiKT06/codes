import numpy as np
from pathlib import Path


class TrajectoryWriter:

    def __init__(self, directory, interval=100):

        self.directory = Path(directory)
        self.interval = interval

        self.traj_dir = self.directory / "trajectory"
        self.traj_dir.mkdir(parents=True, exist_ok=True)

    def write(self, step, system):

        if step % self.interval != 0:
            return

        positions = system.positions.detach().cpu().numpy()
        filename = self.traj_dir / f"frame_{step:06d}.npy"
        np.save(filename, positions)