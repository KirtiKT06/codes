import numpy as np
from scipy.stats import t
import glob
import os
from collections import defaultdict

# Base directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Output directory: simulations/data
OUT_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(OUT_DIR, exist_ok=True)

class gcmc_analysis():
    """
    Block averaging with confidence intervals for correlated MC data.
    """
    def __init__(self, data, n_blocks=20, confidence=0.90):
        """
        Parameters
        ----------
        data : array-like
            Time series data (1D).
        n_blocks : int
            Number of blocks for block averaging.
        confidence : float
            Confidence level (e.g., 0.90 for 90% CI).
        """
        self.data = np.asarray(data)
        self.n_blocks = n_blocks
        self.confidence = confidence

        if self.data.ndim != 1:
            raise ValueError("Data must be 1D")

        self._block_means = None 

    def block_means(self):
        """
        Compute and return block means.
        """
        n = len(self.data)
        block_size = n // self.n_blocks

        if block_size < 1:
            raise ValueError("Too many blocks for data length")

        trimmed = self.data[:self.n_blocks * block_size]
        blocks = trimmed.reshape(self.n_blocks, block_size)

        self._block_means = blocks.mean(axis=1)
        return self._block_means


    def mean_and_ci(self):
        """
        Compute mean and confidence interval using block means.
        """
        if self._block_means is None:
            self.block_means()

        b = self._block_means
        nb = len(b)
        mean = np.mean(b)

        if nb < 2:
            raise ValueError(f"Not enough blocks ({nb}) to estimate error."
        "Reduce n_blocks or increase data length.")

        # standard error of the mean (block-based)
        std_err = np.sqrt(
            np.sum((b - mean) ** 2) / (nb * (nb - 1))
        )

        # Student-t factor
        alpha = 1.0 - self.confidence
        tval = t.ppf(1.0 - alpha / 2.0, df=nb - 1)

        ci = tval * std_err
        return mean, ci

# -------------------------------------------------
# Helper: analyze multiple replicas
# -------------------------------------------------
def analyze_replicas(files, column, n_blocks=20, confidence=0.90):
    """
    Perform block averaging for multiple replicas and average results.

    Parameters
    ----------
    files : list of str
        Paths to replica data files.
    column : int
        Column index of observable (0-based).
    """

    replica_means = []
    replica_vars = []

    for fname in files:
        data = np.loadtxt(fname, comments="#")
        obs = data[:, column]

        gcmc = gcmc_analysis(obs, n_blocks=n_blocks, confidence=confidence)
        mean, ci = gcmc.mean_and_ci()

        replica_means.append(mean)
        replica_vars.append(ci**2)

    replica_means = np.array(replica_means)
    final_mean = replica_means.mean()
    final_error = replica_means.std(ddof=1) / np.sqrt(len(replica_means))

    return final_mean, final_error    

# -------------------------------------------------
# Helper: parse state from filename
# -------------------------------------------------
def parse_state_from_filename(fname):
    """
    Extract T and f from filename like:
    gcmc_T1.0_f0.0365_rep0.dat
    """
    base = os.path.basename(fname)
    parts = base.replace(".dat", "").split("_")
    T = float(parts[1][1:])    # remove 'T'
    f = float(parts[2][1:])    # remove 'f'
    return T, f

files = sorted(
        glob.glob(os.path.join(BASE_DIR, "gcmc_T*_f*_rep*.dat"))
    )
groups = defaultdict(list)

for fname in files:
    T, f = parse_state_from_filename(fname)
    groups[(T, f)].append(fname)

print("================================")
print("GCMC RESULTS (90% CI)")
print("================================")

for (T, f), flist in groups.items():
    rho_mean, rho_err = analyze_replicas(
        flist,
        column=2,
        n_blocks=20,
        confidence=0.90
    )
    print("--------------------------------")
    print(f"T* = {T}, f = {f}")
    print(f"<rho> = {rho_mean:.6f} ± {rho_err:.6f}")