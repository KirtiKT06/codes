from MD_core.observables import kinetic_energy, temperature

class ThermoPrinter:

    def __init__(self, interval=100):

        self.interval = interval
        print(f"{'step':>8} {'Temp':>10} {'KE':>12} {'PE':>12} {'TotalE':>12} {'Pressure':>12}")

    def print(self, step, system, potential_energy, pressure):

        if step % self.interval != 0:
            return

        KE = kinetic_energy(system)
        T = temperature(system)

        total = KE + potential_energy

        print(
            f"{step:8d} "
            f"{T.item():10.4f} "
            f"{KE.item():12.4f} "
            f"{potential_energy.item():12.4f} "
            f"{total.item():12.4f}"
            f"{pressure.item():12.4f}"
        )