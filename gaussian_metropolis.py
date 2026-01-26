"""
This code implements a Metropolis Monte Carlo sampler for a 1D Gaussian
distribution.

Key features:
- Samples from N(0, sigma^2) using the Metropolis algorithm
- Automatically tunes the proposal step size (delta_xm) using
  the dual averaging (Robbins-Monro type) method
- Computes acceptance rate, integrated autocorrelation time,
  and effective sample size
- Produces publication-quality plots

Overall time complexity:
O(n_tunes * n_rounds + n_samples)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from tqdm import tqdm
from config import TABLES_DIR, FIGURES_DIR
import os

class GaussianMetropolisSampler():
    """
    A class implementing Metropolis Monte Carlo sampling for a
    one-dimensional Gaussian target distribution.

    Parameters
    ----------
    sigma : float
        Standard deviation of the target Gaussian distribution
    target_accep : float
        Desired acceptance rate for tuning delta_xm (default 0.5)
    seed : int
        Random seed for reproducibility
    """
    def __init__(self, sigma, target_accep = 0.5, seed = 42):
        self.sigma = sigma                                  # Store the target Gaussian width
        self.target_accep = target_accep                    # Target acceptance rate for proposal tuning
        self.rng = np.random.default_rng(seed)              # Random number generator (Numpy's modern RNG)
        self.delta_xm = None                                # Step size (To be optimized later)
    
    def log_target(self, x):
        """
        Logarithm of the target probability density function.

        For a Gaussian distribution:
        p(x) ∝ exp(-x^2 / (2 * sigma^2))

        Using log-probabilities avoids numerical underflow and
        simplifies Metropolis acceptance ratios.
        """
        y = x**2/(2*self.sigma**2)
        return -y
    
    def optimize_delta_xm(self, epsilon, gamma=0.1, burnin_frac=0.3):
        """
        Optimize the proposal step size (delta_xm) using dual averaging.

        The algorithm adapts delta_xm so that the observed acceptance
        rate approaches the target acceptance rate.

        Parameters
        ----------
        epsilon : float
            Desired statistical accuracy for acceptance estimation
        gamma : float
            Learning-rate parameter for dual averaging
        burnin_frac : float
            Fraction of initial rounds to discard when averaging theta

        Returns
        -------
        delta_xm : float
            Optimized proposal step size
        """
        n_tunes = int(np.ceil(1/(4*epsilon**2)))                        # Number of metropolis steps per tuning round derived from a Bernoulli estimator
        n_rounds = 100                                                  # Number of dual-averaging update rounds
        mu = np.log(self.sigma)                                         # Reference log step size (mu in dual averaging)
        burnin = int(burnin_frac * n_rounds)                            # Burn-in rounds for step-size averaging
        theta = np.log(self.sigma)                                      # Current value of log(delta_xm)
        g_bar = 0.0
        x = 0.0
        theta_samples = []
        for n in tqdm(range(1, n_rounds + 1)):                                # Dual averaging loop
            delta_xm = np.exp(theta)
            accepted = 0
            # Metropolis sub-chain
            for _ in range(n_tunes):                                    # Short Metropolis sub-chain to estimate acceptance rate
                x_prop = x + self.rng.normal(0, delta_xm)      
                log_acc = self.log_target(x_prop) - self.log_target(x)  # Log acceptance ratio
                if np.log(self.rng.uniform()) < log_acc:
                    x = x_prop
                    accepted += 1
            acc_rate = accepted / n_tunes                               # Acceptance rate
            g = acc_rate - self.target_accep                            # Acceptance error relative to target
            g_bar = (n - 1) / n * g_bar + g / n                         # Running Average of the error
            theta = mu + (np.sqrt(n) / gamma) * g_bar                   # Dual averaging update
            if n > burnin:
                theta_samples.append(theta)
        theta_opt = np.mean(theta_samples)                              # Optimization of step size
        self.delta_xm = np.exp(theta_opt)
        return self.delta_xm
    
    def sample(self, n):
        """
        Generate samples from the target Gaussian using the Metropolis algorithm.

        Parameters
        ----------
        n : int
            Number of Monte Carlo samples to generate

        Returns
        -------
        x : ndarray
            Array of sampled values
        """
        if self.delta_xm is None:
            raise RuntimeError("Run optimize_delta_xm() first")
        x = np.zeros(n)
        x[0] = 0.0                                                                          # initialization of Markov Chain
        accepted = 0
        for i in range(1,n):                                                                # Metropolis sampling loop
            x_current = x[i-1]
            x_proposal = x_current + self.rng.normal(0,self.delta_xm)                       # Propose a new position
            log_accp_ratio = self.log_target(x_proposal) - self.log_target(x_current)       # Log acceptance ratio
            rand = np.log(self.rng.uniform())
            if rand < log_accp_ratio:                                                       # Accept or reject the move
                x[i] = x_proposal
                accepted += 1
            else:
                x[i] = x_current
        acceptance_rate = accepted / (n-1)                                                  # Acceptance rate (stored as a diagnostic)
        self.last_acceptance_rate = acceptance_rate
        return x 
          
    def plot(self, samples):
        """
        Plot a histogram of Metropolis samples together with the
        analytical Gaussian probability density.
        """
        xx = np.linspace(-4*self.sigma, 4*self.sigma, 400)
        gaussian = np.exp(-xx**2 / (2*self.sigma**2)) / ((np.sqrt(2*np.pi)*self.sigma))
        plt.figure(figsize=(7, 5))
        plt.hist(samples, bins=100, density=True, color="steelblue",alpha=0.6, edgecolor="white", label="Metropolis samples")
        plt.plot(xx, gaussian, color="black", lw=2.5, label="Theory")                        # Analytical gaussian
        plt.xlabel("RANDOM VARIABLE X $\\rightarrow$", fontsize=13)
        plt.ylabel("PROBABILITY DENSITY $\\rightarrow$", fontsize=13)
        plt.tick_params(axis='both', labelsize=11)
        plt.grid(alpha=0.2)
        plt.legend(frameon=False)
        x_max = 0.0                                                                          # Maximum of the Gaussian
        y_max = 1 / (np.sqrt(2*np.pi) * self.sigma)
        plt.scatter(x_max, y_max, color="red", zorder=5)
        plt.annotate(
            rf"$(x_{{\max}}=0,\; p_{{\max}}={y_max:.3f})$",
            xy=(x_max, y_max),
            xytext=(x_max + 0.5*self.sigma, y_max),
            arrowprops=dict(arrowstyle="->", lw=1),
            fontsize=11
            )
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.title(rf"Gaussian Metropolis sampling ($\sigma={self.sigma}$)", fontsize=14)
        plt.tight_layout()
        plt.savefig(
        os.path.join(
        FIGURES_DIR,
        f"Gaussian_Metropolis_sigma_{self.sigma}.png"
          ),
        dpi=600,
        bbox_inches="tight")
        plt.show()

    def autocorrelation_time(self, samples, max_lag=None):
        """
        Compute the integrated autocorrelation time and effective
        sample size for a given observable.

        Parameters
        ----------
        samples : ndarray
            Monte Carlo samples
        max_lag : int or None
            Maximum lag for autocorrelation calculation

        Returns
        -------
        tau_int : float
            Integrated autocorrelation time
        N_eff : float
            Effective number of independent samples
        """
        def autocorrelation(x, max_lag):
            x = np.asarray(x)
            x = x - np.mean(x)
            var = np.var(x)
            acf = np.zeros(max_lag)
            for k in range(max_lag):
                acf[k] = np.mean(x[:len(x)-k] * x[k:]) / var
            return acf
        max_lag = min(200, len(samples)//10)
        acf = autocorrelation(samples, max_lag)
        positive_acf = acf[1:][acf[1:] > 0]
        tau_int = 1 + 2*np.sum(positive_acf)
        N_eff = len(samples) / (2 * tau_int)
        return tau_int, N_eff
    
    def estimate_pdf(self, samples, bins=100):
        """
        Estimate probability density from samples using a normalized histogram.

        Parameters
        ----------
        samples : array_like
            Monte Carlo samples
        bins : int
            Number of histogram bins

        Returns
        -------
        bin_centers : ndarray
            Centers of histogram bins
        pdf : ndarray
            Estimated probability density
        """
        pdf, bin_edges = np.histogram(samples, bins=bins, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        return bin_centers, pdf
    
    def analytic_pdf(self, x):
        """
        Analytical Gaussian probability density.
        """
        return (
            np.exp(-x**2 / (2 * self.sigma**2))
            / (np.sqrt(2*np.pi) * self.sigma)
        )

def main():
    """
    Main driver function.

    - Initialises the sampler
    - Optimises the proposal step size
    - Generates samples
    - Prints diagnostics
    - Produces plots
    - Stores result in tabular form
    """
    sigmas = [0.1, 1.0, 10.0]
    results = []
    for sigma in sigmas:
        print(f"\nRunning Metropolis for sigma = {sigma}")
        sampler = GaussianMetropolisSampler(sigma=sigma)                                            # Initialization
        delta_xm = sampler.optimize_delta_xm(epsilon=0.02)                                          # Tuning proposal
        samples = sampler.sample(1000000)
        x_hist, pdf_hist = sampler.estimate_pdf(samples)                                            # PDF estimation from the SAME data
        pdf_exact = sampler.analytic_pdf(x_hist)
        max_error = np.max(np.abs(pdf_hist - pdf_exact)/pdf_exact)                                  # Quantitative validation
        tau, N_eff = sampler.autocorrelation_time(samples[:200000])                                 # Diagnostics
        results.append({
            "σ": sigma,
            "Δxₘ": delta_xm,
            "Acceptance": sampler.last_acceptance_rate,
            "τ_int": tau,
            "N_eff": N_eff,
            "PDF_error": max_error})
        sampler.plot(samples)                                                                       # Plot from same data
    df = pd.DataFrame(results)
    print(tabulate(df, headers="keys", tablefmt="github", floatfmt=".4f"))                          # Print data in table form
    df.to_csv(
    os.path.join(TABLES_DIR, "Gaussian_Metropolis_summary.csv"),
    float_format="%.6f",
    index=False
    )


if __name__ == "__main__":
    main()