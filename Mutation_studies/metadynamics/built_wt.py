from openmm.app import *
from openmm import *
from openmm.unit import nanometer
from openmm import Platform

pdb = PDBFile("/home/feynman/projects/codes/Mutation_studies/mutant_pdbs/1PGA.pdb")

forcefield = ForceField(
    "amber19/protein.ff19SB.xml",
    "amber19/tip3pfb.xml"
)

modeller = Modeller(
    pdb.topology,
    pdb.positions
)

modeller.addHydrogens(forcefield, platform=Platform.getPlatformByName("CPU"))

modeller.addSolvent(
    forcefield,
    model='tip3p',
    padding=1.0*nanometer
)

print("Atoms:", modeller.topology.getNumAtoms())

with open("wt_solvated.pdb","w") as f:
    PDBFile.writeFile(
        modeller.topology,
        modeller.positions,
        f
    )