from MD_core.boundary import wrap_positions

class VelocityVerlet:

    def __init__(self, dt):
        self.dt = dt

    def first_half(self, system, forces):

        dt = self.dt
        system.velocities += 0.5 * forces / system.masses[:,None] * dt
        system.positions += system.velocities * dt
        system.positions = wrap_positions(system.positions, system.box)

    def second_half(self, system, forces):

        dt = self.dt
        system.velocities += 0.5 * forces / system.masses[:,None] * dt