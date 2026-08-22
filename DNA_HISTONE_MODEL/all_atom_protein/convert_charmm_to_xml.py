"""
Run this ONCE after downloading CHARMM-GUI's PDB Reader & Manipulator ->
OpenMM output. It converts the CHARMM-format toppar files into a real
OpenMM XML force field file, so build_system.py (which loads
app.ForceField(*forcefield.files)) needs no changes at all.

Modeller.addSolvent() specifically requires an app.ForceField object, not a
CharmmParameterSet — that's why this conversion is worth doing once up
front rather than restructuring the rest of the pipeline to juggle two
different loading mechanisms.

Usage:
    python3 convert_charmm_to_xml.py /path/to/toppar_dir output.xml

Point the toppar_dir at wherever you unzipped CHARMM-GUI's download — it
should contain files like top_all36_prot.rtf, par_all36m_prot.prm,
toppar_water_ions.str, and possibly toppar_all36_prot_*.str patches
(disulfides, etc. — include whichever ones CHARMM-GUI bundled for your
system).
"""
import glob
import os
import sys

from parmed.charmm import CharmmParameterSet
from parmed.openmm import OpenMMParameterSet


def main():
    if len(sys.argv) != 3:
        print(f"Usage: python3 {sys.argv[0]} /path/to/toppar_dir output.xml")
        sys.exit(1)

    toppar_dir = sys.argv[1]
    out_xml = sys.argv[2]

    # Pick up every topology (.rtf), parameter (.prm), and stream (.str)
    # file CHARMM-GUI included. Order matters for CHARMM parameter files
    # (later files can override earlier ones) — CharmmParameterSet handles
    # this the same way the CHARMM program itself would via a toppar.str
    # "master" list if one exists; otherwise this glob-based order is fine
    # for the standard protein+water+ions bundle CHARMM-GUI ships.
    patterns = ["*.rtf", "*.prm", "*.str"]
    files = []
    for pattern in patterns:
        files.extend(sorted(glob.glob(os.path.join(toppar_dir, pattern))))

    if not files:
        raise FileNotFoundError(
            f"No .rtf/.prm/.str files found in {toppar_dir} — check the path."
        )

    print(f"Loading {len(files)} CHARMM parameter/topology files:")
    for f in files:
        print(f"  {f}")

    charmm_params = CharmmParameterSet(*files)
    omm_params = OpenMMParameterSet.from_parameterset(charmm_params)
    omm_params.write(out_xml)

    print(f"\nOpenMM XML force field written to: {out_xml}")
    print(
        "Point input.json's forcefield.files at this file (plus a TIP3P "
        "water xml if it wasn't bundled into the conversion — check "
        "whether HOH/water residues appear in the output XML; if not, "
        "keep using OpenMM's bundled 'charmm36/water.xml' for water only, "
        "since standard TIP3P water/ion parameters are identical between "
        "plain CHARMM36 and CHARMM36m)."
    )


if __name__ == "__main__":
    main()