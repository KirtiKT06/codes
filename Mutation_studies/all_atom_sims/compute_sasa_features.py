import os
import numpy as np
import pandas as pd
import MDAnalysis as mda
import freesasa

OUT_ROOT = "/data/openmm_out/1PGA_mutations/explicit_solvent"


def load_trajectory(mut):

    rep_dir = os.path.join(
        OUT_ROOT,
        f"{mut}_rep1"
    )

    dcd = os.path.join(
        rep_dir,
        f"{mut}_rep1_trajectory.dcd"
    )

    pdb = os.path.join(
        rep_dir,
        f"{mut}_rep1_final.pdb"
    )

    if not os.path.exists(dcd):
        return None

    return mda.Universe(pdb, dcd)


def frame_sasa(universe):

    tmp_pdb = "_tmp_frame.pdb"

    values = []

    for ts in universe.trajectory[::100]:

        universe.select_atoms("protein").write(tmp_pdb)

        structure = freesasa.Structure(tmp_pdb)

        result = freesasa.calc(structure)

        values.append(result.totalArea())

    if os.path.exists(tmp_pdb):
        os.remove(tmp_pdb)

    return np.array(values)


def compute_features(mut):

    u = load_trajectory(mut)

    if u is None:
        return None

    sasa = frame_sasa(u)

    return {
        "mutant": mut,
        "mean_sasa": np.mean(sasa),
        "std_sasa": np.std(sasa),
    }


def main():

    wt = compute_features("1PGA")

    records = []

    for d in sorted(os.listdir(OUT_ROOT)):

        if not d.endswith("_rep1"):
            continue

        mut = d.replace("_rep1", "")

        if mut == "1PGA":
            continue

        row = compute_features(mut)

        if row is None:
            continue

        row["delta_mean_sasa"] = (
            row["mean_sasa"] - wt["mean_sasa"]
        )

        row["delta_std_sasa"] = (
            row["std_sasa"] - wt["std_sasa"]
        )

        row["rel_mean_sasa"] = (
            row["mean_sasa"] - wt["mean_sasa"]
        ) / wt["mean_sasa"]

        records.append(row)

        print(
            mut,
            row["rel_mean_sasa"]
        )

    pd.DataFrame(records).to_csv(
        "sasa_features.csv",
        index=False
    )


if __name__ == "__main__":
    main()