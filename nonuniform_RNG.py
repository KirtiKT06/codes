import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
class ProbabilityDistribution:
    # ============================================================
    # BASE CLASS: ProbabilityDistribution
    # ------------------------------------------------------------
    # This is an abstract base class (a "contract").
    # It defines WHAT every probability distribution must provide,
    # but does NOT implement any actual formula.
    #
    # Polymorphism:
    # Any subclass must implement these methods, so samplers
    # can use them without knowing which distribution they have.
    # ============================================================
    def pdf(self, x):
        """Return the value of the probability density function f(x)."""
        raise NotImplementedError("pdf(x) not implemented")
    
    def inverse_cdf(self, u):
        """Return x such that F(x) = u (inverse transform sampling)."""
        raise NotImplementedError("inverse_cdf(u) not implemented")
    
    def support(self):
        """Return the lower and upper bounds of the distribution."""
        raise NotImplementedError("support() not implemented")
    
    def theoretical_moments(self):
        """Return the theoretical mean and variance."""
        raise NotImplementedError("theoretical_moments() not implemented")


class LinearPDF(ProbabilityDistribution):
    # ============================================================
    # DISTRIBUTION 1: Linear PDF f(x) = 2x on [0, 1]
    # ============================================================
    def __init__(self):
        self.name = "Linear Probabilty Distribution of function f(x)=2x on [0, 1]"
    
    def pdf(self, x):
        if 0 <= x <= 1:
            return 2 * x
        return 0.0
    
    def inverse_cdf(self, u):
        return np.sqrt(u)
    
    def support(self):
        return (0.0, 1.0)
    
    def theoretical_moments(self):
        mean = 2 / 3
        variance = 1 / 18
        return mean, variance

class TriangularPDF(ProbabilityDistribution):
    # ============================================================
    # DISTRIBUTION 2: Symmetric triangular PDF on [-1, 1]
    # ============================================================
    def __init__(self):
        self.name = "Triangular Probability distribution for given function on [-1, 1]"
    
    def pdf(self, x):
        if -1 <= x < 0:
            return 1 + x
        elif 0 <= x < 1:
            return 1-x
        else: 
            return 0.0
    
    def inverse_cdf(self, u):
        if u < 0.5:
            return (-1 + np.sqrt(2*u))
        else:
            return (1 - np.sqrt(2*(1 - u)))
    
    def support(self):
        return (-1.0, 1.0)
    
    def theoretical_moments(self):
        mean = 0
        variance = 1 / 6
        return mean, variance

class InversionSampler:
    # ============================================================
    # SAMPLER 1: InversionSampler
    # ------------------------------------------------------------
    # Implements inverse transform sampling.
    # This sampler works for ANY distribution that provides
    # an inverse_cdf(u) method.
    # ============================================================
    def __init__(self, distribution, rng=None):
        self.name1 = "Inversion Sampling"
        self.distribution = distribution
        self.rng = rng if rng is not None else np.random.default_rng(seed=42)
    
    def sample(self, n):
        samples = []
        for _ in range(n):
            u = self.rng.uniform(0.0, 1.0)
            x = self.distribution.inverse_cdf(u)
            samples.append(x)
        return np.array(samples)

class RejectionSampler:
    # ============================================================
    # SAMPLER 2: RejectionSampler
    # ------------------------------------------------------------
    # Implements acceptance–rejection sampling using
    # a uniform proposal distribution over the support.
    # ============================================================
    def __init__(self, distribution, M, rng=None):
        self.name1 = "Acceptance-Rejection Sampling"
        self.distribution = distribution
        self.M = M
        self.rng = rng if rng is not None else np.random.default_rng(seed=42)
        self.trials = 0
        self.accepted = 0
    
    def sample(self, n):
        """Generate n samples using rejection sampling."""
        self.trials = 0
        self.accepted = 0
        samples = []
        xmin, xmax = self.distribution.support()
        while len(samples) < n:
            self.trials += 1
            x = self.rng.uniform(xmin, xmax)
            u = self.rng.uniform(0.0, 1.0)
            # uniform proposal: g(x) = 1 / (xmax - xmin)
            g = 1.0 / (xmax - xmin)
            if u <= self.distribution.pdf(x) / (self.M * g):
                samples.append(x)
                self.accepted += 1
        return np.array(samples)
    
    def acceptance_rate(self):
        """Return the acceptance rate of the sampler."""
        return self.accepted / self.trials

class Experiment:
    # ============================================================
    # EXPERIMENT CLASS
    # ------------------------------------------------------------
    # This class runs the Monte Carlo experiment, computes
    # statistics, and produces plots.
    # ============================================================
    def __init__(self, distribution, sampler, n):
        self.distribution = distribution
        self.sampler = sampler
        self.n = n
        self.samples = None
    
    def run(self):
        """Generate samples."""
        self.samples = self.sampler.sample(self.n)
    
    def compute_statistics(self):
        """Compute Distribution and Theoretical statistics."""
        mean_mc = np.mean(self.samples)
        var_mc = np.var(self.samples)
        mean_th, var_th = self.distribution.theoretical_moments()
        return {
            "mean_mc": mean_mc,
            "var_mc": var_mc,
            "mean_th": mean_th,
            "var_th": var_th,
        }
    
    def estimate_pdf(self, bins=100):
        hist, bin_edges = np.histogram(self.samples, bins=bins, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_width = bin_edges[1] - bin_edges[0]
        pdf_true = np.array([self.distribution.pdf(x) for x in bin_centers])
        mae = np.mean(np.abs(hist - pdf_true))
        mse = np.mean((hist - pdf_true)**2)
        ise = np.sum((hist - pdf_true)**2) * bin_width                          # Approximation to ∫(f̂(x) − f(x))² dx using histogram bins
        return mae, mse, ise  
    
    def report(self):
        # Distribution statistics
        mean_mc = np.mean(self.samples)
        var_mc = np.var(self.samples)
        # Theoretical statistics
        mean_th, var_th = self.distribution.theoretical_moments()
        # PDF comparison errors
        mae, mse, ise = self.estimate_pdf()
        # Create table
        data1 = {
            "Obtained from distribution": [mean_mc, var_mc],
            "Analytical":  [mean_th, var_th],
            "Absolute Error": [abs(mean_mc - mean_th), abs(var_mc - var_th)]}
        index = ["Mean", "Variance"]
        df = pd.DataFrame(data1, index=index)
        print(f"\nDistribution: {self.distribution.name}")
        print(f"Sampling Technique:{self.sampler.name1}")
        print(f"Number of samples: {self.n}\n")
        if hasattr(self.sampler, "acceptance_rate"):
            print(f"Acceptance rate is: {self.sampler.acceptance_rate()}\n")
        print(tabulate(df, headers="keys", tablefmt="github", floatfmt=".6f"))
        data2 = {"Mean Average Error": [mae],
                 "Mean Squared Error": [mse],
                 "Integrated Squared error": [ise]}
        df = pd.DataFrame(data2, index=["value"])
        print("\nComparison of Probability Densities from Generated histogram vs. True function\n")
        print(tabulate(df, headers="keys", tablefmt="github", floatfmt=".6f"))       
    
    def plot(self, bins=100):
        """Plot histogram and analytical PDF."""
        plt.figure(figsize=(7,5))
        xmin, xmax = self.distribution.support()
        x = np.linspace(xmin, xmax, 1000)
        y = [self.distribution.pdf(xi) for xi in x]
        plt.plot(x, y, color='red', label="Analytical PDF", linewidth=2.5)
        plt.hist(self.samples, bins=bins, density=True, color="steelblue",alpha=0.6, edgecolor="white", label="Estimated PDF")
        plt.xlabel("RANDOM VARIABLE X $\\rightarrow$", fontsize=13)
        plt.ylabel("PROBABILITY DENSITY P(X) $\\rightarrow$", fontsize=13)
        plt.title(self.distribution.name, fontsize=14)
        plt.tick_params(axis='both', labelsize=11)
        plt.grid(alpha=0.2)
        plt.legend(frameon=False)
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.show()

def main():
    # ============================================================
    # MAIN FUNCTION
    # ------------------------------------------------------------
    # This is the "driver" of the program.
    # It wires together distributions, samplers, and experiments.
    # ============================================================
    n = 1000000
    # Choose distribution by uncommenting
    # dist = TriangularPDF()
    dist = LinearPDF()   # <- switch here if needed
    # Choose sampler bu uncommenting
    sampler = InversionSampler(dist)
    # sampler = RejectionSampler(dist, M=2)
    # Run experiment
    exp = Experiment(dist, sampler, n)
    exp.run()
    exp.report()
    exp.plot()

if __name__=="__main__":
    main()