import csv
from pathlib import Path
from MD_core.observables import kinetic_energy, temperature

class ThermoLogger:

    def __init__(self, filepath):

        self.filepath = Path(filepath)

        self.file = open(self.filepath, "w", newline="")
        self.writer = csv.writer(self.file)

        self.writer.writerow([
            "step",
            "temperature",
            "kinetic_energy",
            "potential_energy",
            "total_energy"
        ])

    def log(self, step, system, potential_energy):

        KE = kinetic_energy(system)
        T = temperature(system)

        total = KE + potential_energy

        self.writer.writerow([
            step,
            T.item(),
            KE.item(),
            potential_energy.item(),
            total.item()
        ])

    def close(self):
        self.file.close()