# import pandas as pd

# bad = set(
#     open("linearized_fallback_structures.txt")
#     .read()
#     .splitlines()
# )

# df = pd.read_csv(
#     "/data/rna_electrostats/phi_res/all_residue_potentials.csv",
#     low_memory=False
# )

# clean = df[~df["pdb_id"].isin(bad)]

# clean.to_csv(
#     "/data/rna_electrostats/phi_res/all_residue_potentials_nonlinear_only.csv",
#     index=False
# )

import pandas as pd
import numpy as np

df = pd.read_csv(
    "/data/rna_electrostats/phi_res/all_residue_potentials_nonlinear_only.csv",
    low_memory=False
)

bad = (
    df.groupby("pdb_id")["potential_mean_kT_e_8A"]
      .apply(lambda x: np.nanmax(np.abs(x)))
)

bad_ids = set(
    bad[bad > 1e3].index
)

print("Removing", len(bad_ids), "structures")

clean = df[~df["pdb_id"].isin(bad_ids)]

clean.to_csv(
    "/data/rna_electrostats/phi_res/all_residue_potentials_clean.csv",
    index=False
)