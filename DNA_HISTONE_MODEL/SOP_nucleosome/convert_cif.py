from openmm.app import PDBxFile, PDBFile

cif = PDBxFile("/data/cg_dna_histone/sop_single_bead/run3/hpc_final.cif")

# avoid the CRYST1 bug
cif.topology.setPeriodicBoxVectors(None)

with open("/data/cg_dna_histone/sop_single_bead/run3/hpc_final_converted.pdb", "w") as f:
    PDBFile.writeFile(
        cif.topology,
        cif.positions,
        f
    )

print("Done.")