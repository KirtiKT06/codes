import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm
import os
"""
=============================
Directory Setup
=============================
"""
os.makedirs("results/tables", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

class LJ_potential:
    r"""
    Lennard–Jones potential energy calculator for a system of particles
    under periodic boundary conditions.

    The total potential energy is written as:

    .. math::

        U = \sum_{i<j} V_{\mathrm{LJ}}(r_{ij})

    where the Lennard–Jones pair potential is

    .. math::

        V_{\mathrm{LJ}}(r) =
        4\epsilon \left[
            \left(\frac{\sigma}{r}\right)^{12}
            -
            \left(\frac{\sigma}{r}\right)^6
        \right]

    To improve computational efficiency, the interaction is split into:
    - an explicit short-range contribution up to a cut-off radius r_c
    - a long-range tail correction assuming g(r) = 1 for r > r_c
    """
    def __init__(self, positions, epsilon, sigma, box_len):
        r"""
        Initialize the Lennard–Jones system.

        Parameters
        ----------
        positions : ndarray of shape (N, 3)
            Cartesian coordinates of N particles.
        epsilon : float
            Depth of the LJ potential well.
        sigma : float
            Finite distance at which the LJ potential is zero.
        box_len : float
            Length of the cubic simulation box.

        Notes
        -----
        The reduced number density is defined as:

        .. math::

            \rho = \frac{N}{L^3}

        where L is the box length.
        """
        self.positions = positions
        self.epsilon = epsilon
        self.sigma = sigma
        self.box_len = box_len
        self.r_cuts = np.linspace(1.0, 4.0, 25)
        self.U_short_list = []
        self.U_long_list = []
        self.U_total_list = []
        self.U_per_particle = []
        self.results = []
    
    def LJ_pot(self, r):
        r"""
        Lennard–Jones pair potential.

        Parameters
        ----------
        r : float
            Inter-particle distance.

        Returns
        -------
        float
            Lennard–Jones potential energy at distance r.
        Equation
        --------
        .. math::

            V_{\mathrm{LJ}}(r) =
            4\epsilon \left[
                \left(\frac{\sigma}{r}\right)^{12}
                -
                \left(\frac{\sigma}{r}\right)^6
            \right]
        """
        term = (self.sigma/r)**6
        potential = 4*self.epsilon*((term)**2 - term)
        return potential
    
    def pot_cal(self):
        r"""
        Compute the Lennard–Jones potential energy as a function
        of the cut-off radius.

        The total energy is decomposed as:

        .. math::

            U(r_c) = U_{\mathrm{short}}(r_c) + U_{\mathrm{tail}}(r_c)

        Short-range contribution
        ------------------------
        Explicitly computed using pairwise summation:

        .. math::

            U_{\mathrm{short}} =
            \sum_{i<j}^{r_{ij}<r_c} V_{\mathrm{LJ}}(r_{ij})

        Periodic boundary conditions are applied using the
        Minimum Image Convention (MIC):

        .. math::

            \mathbf{r}_{ij} =
            \mathbf{r}_i - \mathbf{r}_j
            - L \cdot \mathrm{round}\!\left(
                \frac{\mathbf{r}_i - \mathbf{r}_j}{L}
            \right)

        Long-range (tail) correction
        ----------------------------
        Assumes a uniform fluid beyond r_c, i.e.,

        .. math::

            g(r) = 1 \quad \text{for } r > r_c

        The tail correction is then:

        .. math::

            U_{\mathrm{tail}} =
            \frac{8\pi}{3}
            N \rho \epsilon \sigma^3
            \left[
                \frac{1}{3}\left(\frac{\sigma}{r_c}\right)^9
                -
                \left(\frac{\sigma}{r_c}\right)^3
            \right]
        """
        n = self.positions.shape[0]
        for r_cut in tqdm(self.r_cuts):
            u_short = 0.0
            for i in range(n):
                for j in range(i+1, n):
                    r_ij = self.positions[i] - self.positions[j]
                    r_ij -= self.box_len*np.round(r_ij/self.box_len)
                    r = np.linalg.norm(r_ij)
                    if r < r_cut:
                        u_short += self.LJ_pot(r)
            rho = n/self.box_len**3
            u_long = (8*np.pi/3)*n*rho*self.epsilon*self.sigma**3*((1/3)*(self.sigma/r_cut)**9 - (self.sigma/r_cut)**3)
            u_total = u_short + u_long
            self.U_short_list.append(u_short)
            self.U_long_list.append(u_long)
            self.U_total_list.append(u_total)
        self.U_per_particle = np.array(self.U_total_list) / n
        df = pd.DataFrame({r"$r_c$": self.r_cuts,
                        r"$U_lr$": self.U_long_list,
                        r"$U_sr$": self.U_short_list,
                        r"$U_total$": self.U_total_list,
                        r"$U/N$": self.U_per_particle,
                          })
        filename = f"results/tables/{self.__class__.__name__}_stats.csv"
        df.to_csv(filename, float_format="%.6f")
        latex_file = filename.replace(".csv", ".tex")
        df.to_latex(
        latex_file,
        float_format="%.6f",
        caption="Lennard–Jones energy components as a function of cut-off radius",
        label="tab:lj_cutoff_convergence"
        )


    def plotting(self):    
        r"""
        Generate plots illustrating cut-off convergence.

        Plots
        -----
        1. Total energy, short-range energy, and tail correction vs r_c
        2. Energy per particle:

        .. math::

            \frac{U}{N}

        These plots demonstrate:
        - dominance of short-range repulsion at moderate density
        - decay of long-range attractive contributions
        - convergence of total energy for r_c ≳ 2.5σ
        """
        N = self.positions.shape[0]
        filename1 = f"results/figures/LJ_energy_components_N{N}.pdf"
        filename2 = f"results/figures/LJ_energy_per_particle_N{N}.pdf"
        plt.figure(figsize=(7, 5))   
        plt.xlim(0.9, 4.1)
        plt.ylim(-800, 1700)
        plt.plot(self.r_cuts, self.U_total_list, linewidth=2.5, color='steelblue', marker='o', markersize=4, label='Total Energy')
        plt.plot(self.r_cuts, self.U_short_list, linewidth=2.5, color='red', marker='^', markersize=4, label='Short-range Energy (Repulsive)')
        plt.plot(self.r_cuts, self.U_long_list, linewidth=2.5, color='gold', marker='*', markersize=4, label='Long-range Energy (Attractive)')
        plt.tick_params(axis='both', labelsize=11)
        plt.grid(alpha=0.2)
        plt.xlabel("Cut-off radius $r_c \\rightarrow$", fontsize=13)
        plt.ylabel("Total Energy $\\rightarrow$", fontsize=13)
        plt.title(rf"Lennard-Jones Potential for a system of {self.positions.shape[0]} particles", fontsize=14)
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(filename1.replace(".pdf", ".png"), dpi=600, bbox_inches="tight")
        plt.show()
        plt.figure(figsize=(7, 5))
        plt.plot(self.r_cuts, self.U_per_particle, linewidth=2.5, color='steelblue', marker='o', markersize=4)
        plt.xlabel("Cut-off radius $r_c \\rightarrow$", fontsize=13)
        plt.ylabel("Energy per particle $(U/N) \\rightarrow$", fontsize=13)
        plt.title("Figure: Energy per particle vs. $r_c$", fontsize=14)
        plt.grid(alpha=0.2)
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(filename2.replace(".pdf", ".png"), dpi=600, bbox_inches="tight")
        plt.show()

def main():
    positions = np.loadtxt("coords_LJ.dat", skiprows=2)
    epsilon = 1.0
    sigma = 1.0
    box_len = 8.0
    lj_system = LJ_potential(
        positions=positions,
        epsilon=epsilon,
        sigma=sigma,
        box_len=box_len)
    lj_system.pot_cal()
    lj_system.plotting()
if __name__ == "__main__":
    main()