"""
Sanity check: print prot_water_hbonds vs experimental y
to verify the trend makes physical sense.
Run: python check_hbonds.py
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

feat   = pd.read_csv("explicit_features.csv")
sel    = pd.read_csv("selected_mutants.csv")

merged = feat[feat["mutant"] != "1PGA"].merge(
    sel[["name", "y"]], left_on="mutant", right_on="name"
).sort_values("y")

print("Mutant    y (exp)   prot_water_hbonds")
print("-" * 42)
for _, row in merged.iterrows():
    bar = "█" * int(row["prot_water_hbonds"] / 10)
    print(f"{row['mutant']:<8}  {row['y']:>+6.3f}    "
          f"{row['prot_water_hbonds']:>6.0f}  {bar}")

rho, p = spearmanr(merged["prot_water_hbonds"], merged["y"])
print(f"\nSpearman ρ = {rho:.3f}  p = {p:.4f}")
print(f"n = {len(merged)}")

# Scatter plot
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(merged["prot_water_hbonds"], merged["y"],
           s=80, color="steelblue", edgecolors="k",
           linewidths=0.5, zorder=3)

for _, row in merged.iterrows():
    ax.annotate(row["mutant"],
                (row["prot_water_hbonds"], row["y"]),
                fontsize=7, color="gray",
                xytext=(4, 4), textcoords="offset points")

z  = np.polyfit(merged["prot_water_hbonds"], merged["y"], 1)
xs = np.linspace(merged["prot_water_hbonds"].min(),
                 merged["prot_water_hbonds"].max(), 100)
ax.plot(xs, np.poly1d(z)(xs), color="tomato",
        linewidth=1.5, linestyle="--")

ax.set_xlabel("Protein-Water H-bonds", fontsize=11)
ax.set_ylabel("Experimental y (−ΔG)", fontsize=11)
ax.set_title(f"ρ = {rho:.3f}  p = {p:.4f}  n = {len(merged)}",
             fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("hbond_vs_y.png", dpi=150)
print("\nSaved: hbond_vs_y.png")

# from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
# import numpy as np

# def compute_protein_hbonds(u):

#     h = HydrogenBondAnalysis(
#         universe=u,
#         donors_sel="protein",
#         acceptors_sel="protein",
#         d_a_cutoff=3.5,
#         d_h_a_angle_cutoff=150
#     )

#     h.run(step=50)

#     hbonds = h.results.hbonds

#     if len(hbonds) == 0:
#         return 0.0

#     frames = np.unique(hbonds[:,0])

#     counts = []

#     for frame in frames:
#         counts.append(
#             np.sum(hbonds[:,0] == frame)
#         )

#     return np.mean(counts)