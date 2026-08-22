"""
Check explicit solvent log files for simulation health.
Run: python check_logs.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from scipy.stats import linregress

OUT_ROOT = "/data/openmm_out/1PGA_mutations/explicit_solvent"

# Find all log files
log_files = glob.glob(f"{OUT_ROOT}/*/*_log.csv")
print(f"Found {len(log_files)} log files")

for log_path in sorted(log_files):
    mut = os.path.basename(log_path).split("_rep")[0]
    print(f"\nChecking {mut}...")

    try:
        df = pd.read_csv(log_path)
        if df.empty:
            print(f"  [EMPTY] Log file is empty")
            continue

        print(f"  Columns: {df.columns.tolist()}")
        print(f"  Frames:  {len(df)}")

        # Find column names (they vary slightly)
        pe_col   = [c for c in df.columns if "Potential" in c][0]
        temp_col = [c for c in df.columns if "Temperature" in c][0]
        den_col  = [c for c in df.columns if "Density" in c][0]
        te_col   = [c for c in df.columns if "Total" in c][0]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"{mut} — Simulation Health Check", fontsize=14)

        df[pe_col].plot(ax=axes[0,0], title="Potential Energy (kJ/mol)",
                       color="steelblue")
        df[temp_col].plot(ax=axes[0,1], title="Temperature (K)",
                         color="darkorange")
        axes[0,1].axhline(350, color="red", linestyle="--",
                          linewidth=1, label="Target 350K")
        axes[0,1].legend()

        df[den_col].plot(ax=axes[1,0], title="Density (g/mL)",
                        color="green")
        axes[1,0].axhline(0.95, color="red", linestyle="--",
                          linewidth=1, label="Expected ~0.95")
        axes[1,0].legend()

        df[te_col].plot(ax=axes[1,1], title="Total Energy (kJ/mol)",
                       color="purple")

        plt.tight_layout()
        save_path = f"log_check_{mut}.png"
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")

        # Quick health summary
        temp_mean = df[temp_col].mean()
        temp_std  = df[temp_col].std()
        den_final = df[den_col].iloc[-100:].mean()
        pe_vals = df[pe_col].values
        slope, _, _, _, _ = linregress(range(len(pe_vals)), pe_vals)
        slope_per_ns = slope * len(pe_vals) / 50  # kJ/mol per ns

        print(f"  Temperature: {temp_mean:.1f} ± {temp_std:.1f} K "
              f"{'✓' if abs(temp_mean-350)<5 else '✗ PROBLEM'}")
        print(f"  Final density: {den_final:.3f} g/mL "
              f"{'✓' if 0.85 < den_final < 1.05 else '✗ PROBLEM'}")
        print(f"  Energy slope: {slope_per_ns:.1f} kJ/mol/ns "
              f"{'✓' if abs(slope_per_ns) < 50 else '✗ CHECK THIS'}")

    except Exception as e:
        print(f"  [ERROR] {e}")

print("\nDone. Check the PNG files for visual inspection.")