"""
This code intends to generate random numbers for a Gaussian distribution. 
The value of delta_xm is optimized for a given sigma internally using dual averaging method. Time complexity is O(n_tunes*n_rounds)
"""
import numpy as np
import matplotlib.pyplot as plt
class GaussianMetropolisSampler():
    def __init__(self, sigma, target_accep = 0.5, seed = 42):
        self.sigma = sigma
        self.target_accep = target_accep
        self.rng = np.random.default_rng(seed)
        self.delta_xm = None
    
    def log_target(self, x):
        y = x**2/(2*self.sigma**2)
        return -y
    
    def optimize_delta_xm(self, epsilon, gamma=0.1, burnin_frac=0.3):
        n_tunes = int(np.ceil(1/(4*epsilon**2)))
        n_rounds = 100
        mu = np.log(self.sigma)
        burnin = int(burnin_frac * n_rounds)
        theta = np.log(self.sigma)     
        g_bar = 0.0
        x = 0.0
        theta_samples = []
        for n in range(1, n_rounds + 1):
            delta_xm = np.exp(theta)
            accepted = 0
            # Metropolis sub-chain
            for _ in range(n_tunes):
                x_prop = x + self.rng.normal(0, delta_xm)
                log_acc = -(x_prop**2 - x**2) / (2 * self.sigma**2)
                if np.log(self.rng.uniform()) < log_acc:
                    x = x_prop
                    accepted += 1
            acc_rate = accepted / n_tunes
            # Dual averaging update
            g = acc_rate - self.target_accep
            g_bar = (n - 1) / n * g_bar + g / n
            theta = mu + (np.sqrt(n) / gamma) * g_bar
            # Collect only after burn-in
            if n > burnin:
                theta_samples.append(theta)
        theta_opt = np.mean(theta_samples)
        self.delta_xm = np.exp(theta_opt)
        return self.delta_xm
    
    def sample(self, n):
        if self.delta_xm is None:
            raise RuntimeError("Run optimize_delta_xm() first")
        x = np.zeros(n)
        x[0] = 0.0
        for i in range(1,n):
            x_current = x[i-1]
            x_proposal = x_current + self.rng.normal(0,self.delta_xm)
            log_accp_ratio = self.log_target(x_proposal) - self.log_target(x_current)
            rand = np.log(self.rng.uniform())
            if rand < log_accp_ratio:
                x[i] = x_proposal
            else:
                x[i] = x_current
        return x
    
    def plot(self, n):
        samples = self.sample(n)
        xx = np.linspace(-4, 4, 400)
        gaussian = np.exp(-xx**2 / 2) / np.sqrt(2*np.pi)
        plt.hist(samples, bins=100, density=True, alpha=0.6)
        plt.plot(xx, gaussian, 'r', lw=2)
        plt.xlabel("RANDOM VARIABLE X")
        plt.ylabel("PROBABILITY DENSITY")
        plt.show()
    
def main():
    sigma = 1
    sampler = GaussianMetropolisSampler(sigma=sigma)
    delta_xm = sampler.optimize_delta_xm(epsilon=0.02)
    print(f"Optimzed value of delta_xm for sigma value {sigma:.2f} is {delta_xm:.6f}")
    sampler.plot(n=1000000)

if __name__ == "__main__":
    main()