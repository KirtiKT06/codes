from openmmplumed import PlumedForce

with open("/home/feynman/projects/codes/Mutation_studies/metadynamics/plumed.dat") as f:
    plumed_script = f.read()

force = PlumedForce(plumed_script)

print("PLUMED force created successfully")
