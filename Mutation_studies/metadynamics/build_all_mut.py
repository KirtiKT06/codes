from openmm.app import *
from openmm import Platform
from openmm.unit import nanometer

from pathlib import Path

# ==========================================
# PATHS
# ==========================================

input_dir = Path("/home/feynman/projects/codes/Mutation_studies/mutant_pdbs")
output_dir = Path("systems")

output_dir.mkdir(exist_ok=True)

# ==========================================
# FORCE FIELD
# ==========================================

forcefield = ForceField(
    "amber19/protein.ff19SB.xml",
    "amber19/tip3pfb.xml"
)

cpu_platform = Platform.getPlatformByName("CPU")

# ==========================================
# LOOP OVER MUTANTS
# ==========================================

pdb_files = sorted(input_dir.glob("*.pdb"))

print(f"\nFound {len(pdb_files)} PDB files\n")

for pdb_file in pdb_files:

    mutant_name = pdb_file.stem

    mutant_folder = output_dir / mutant_name
    mutant_folder.mkdir(exist_ok=True)

    output_pdb = mutant_folder / f"{mutant_name}_solvated.pdb"

    print("=" * 60)
    print(f"Processing {mutant_name}")
    print("=" * 60)

    try:

        pdb = PDBFile(str(pdb_file))

        modeller = Modeller(
            pdb.topology,
            pdb.positions
        )

        print("Adding hydrogens...")

        modeller.addHydrogens(
            forcefield,
            platform=cpu_platform
        )

        print("Adding solvent...")

        modeller.addSolvent(
            forcefield,
            model="tip3p",
            padding=1.0 * nanometer
        )

        print(
            f"Total atoms: "
            f"{modeller.topology.getNumAtoms()}"
        )

        with open(output_pdb, "w") as f:

            PDBFile.writeFile(
                modeller.topology,
                modeller.positions,
                f
            )

        print(f"Saved: {output_pdb}")

    except Exception as e:

        print(
            f"FAILED: {mutant_name}"
        )

        print(e)

print("\nFinished.\n")