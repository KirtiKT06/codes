mass = {
    'ALA':71.0788, 'ARG':156.1875, 'ASN':114.1038,
    'ASP':115.0886, 'CYS':103.1388, 'GLN':128.1307,
    'GLU':129.1155, 'GLY':57.0519, 'HIS':137.1411,
    'ILE':113.1594, 'LEU':113.1594, 'LYS':128.1741,
    'MET':131.1926, 'PHE':147.1766, 'PRO':97.1167,
    'SER':87.0782, 'THR':101.1051, 'TRP':186.2132,
    'TYR':163.1760, 'VAL':99.1326
}

with open("histone_cg_with_tails.pdb") as pdb, open("masses_proteins.inp", "w") as out:

    out.write("Type,resid,chain,mass(amu)\n")

    for line in pdb:
        if line.startswith("ATOM"):

            resname = line[17:20].strip()
            chain   = line[21].strip()
            resid   = line[22:26].strip()

            out.write(
                f"{resname},{resid},{chain},{mass[resname]:10.4f}\n"
            )