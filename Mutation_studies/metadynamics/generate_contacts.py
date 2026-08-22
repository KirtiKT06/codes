import mdtraj as md

traj = md.load("/home/feynman/projects/codes/Mutation_studies/mutant_pdbs/1PGA.pdb")

ca_atoms = traj.topology.select("name CA")

contacts = []

for i in range(len(ca_atoms)):
    for j in range(i+4, len(ca_atoms)):

        pair = [ca_atoms[i], ca_atoms[j]]

        d = md.compute_distances(
            traj,
            [pair]
        )[0][0]

        if d < 0.8:

            contacts.append(
                (ca_atoms[i]+1,
                 ca_atoms[j]+1,
                 d)
            )

print("Contacts:", len(contacts))

with open("ca_contacts_with_dist.dat","w") as f:

    for a,b,d in contacts:

        f.write(
            f"{a} {b} {d:.8f}\n"
        )