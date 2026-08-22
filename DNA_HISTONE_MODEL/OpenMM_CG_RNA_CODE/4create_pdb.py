"""
converts xyz file to pdb file
"""
import numpy as np
import pandas as pd

cMg = 18

# dt = np.dtype({'names': ['A1','A2','A3', 'A4'],
#                'formats': ['U7', '<f8', '<f8', '<f8' ]})
# xyz = np.loadtxt("../Add_ion/6ufm_neutral.xyz", dtype=dt, unpack=True, skiprows=2)
# xyz = pd.read_csv("../Add_ion/All/6ufm_Mg%d_full.xyz"%(cMg), header=None, delimiter='\t', skiprows=2)
xyz = pd.read_csv("../Add_Ion/MgScan/6dmc_Mg%d.xyz"%(cMg), header=None, delimiter='\t', skiprows=2)
seq = np.loadtxt("4_sequence.inp",dtype='U3')

print(xyz[0][1])
# file = open("6c27_L0.xyz", "w")
# fout = open("4rum_Mg0mM_K%dmM.pdb"%(cMg), "w")
# fout = open("../DATA/Full_MgScan/Mg%d/6ufm_Mg%d_full.pdb"%(cMg, cMg), "w")
fout = open("./MgScan/6dmc_Mg%d.pdb"%(cMg), "w")
nop= len(xyz[0])
nbead= 3*len(seq)
L_half = 100.0
# f = "%s\n\n"%(str(nop))
# file.write(f)
# file.write("\n")

nMg = list(xyz[0]).count("Mg")
nCo = list(xyz[0]).count("Co")
nK  = list(xyz[0]).count("K")
nCl = list(xyz[0]).count("Cl")
print("nMg: ", nMg, nK, nCl) 
nion = nMg+nK+nCl
rna_resid = [i.upper() for i in seq for j in range(3)]
ion_resid = [xyz[0][i].upper() for i in range(nbead,nop)]
chainID_rnaA = ["A" for i in range(3*102)]
# chainID_rnaB = ["B" for i in range(3*77)]
chainID_mg  = ["B" for i in range(nMg)]
chainID_co  = ["C" for i in range(nCo)]
chainID_k   = ["D" for i in range(nK)]
chainID_cl  = ["E" for i in range(nCl)]

residue_sequence_number_rnaA = [i+1 for i in range(102) for j in range(3)]
# residue_sequence_number_rnaB = [i for i in range(3*77) for j in range(3)]
residue_sequence_number_Mg= [ i+1 for i in range(nMg)]
residue_sequence_number_Co= [ i+1 for i in range(nCo)]
residue_sequence_number_K = [ i+1 for i in range(nK)]
residue_sequence_number_Cl= [ i+1 for i in range(nCl)]

residue_sequence_number_ions = residue_sequence_number_Mg+residue_sequence_number_Co+residue_sequence_number_K+residue_sequence_number_Cl

# print("residue_sequence_number_rna: ", residue_sequence_number_rnaA, )


print(rna_resid)
print(len(rna_resid))
atomS = ["ATOM" for i in range(nbead)]
hetatmS=["HETATM" for i in range(nbead, nop)]

atomsring = atomS + hetatmS
space = ' '
atomSerialNumber=[int(i) for i in range(len(xyz[0]))]
atomName = [i.upper() for i in xyz[0]]
alternate_location_indicator = space
residue_name = rna_resid + ion_resid
chainIDlist = chainID_rnaA + chainID_mg + chainID_co + chainID_k + chainID_cl
# print("chainIDlist ", chainIDlist)
residue_sequence_number = residue_sequence_number_rnaA + residue_sequence_number_ions
code_for_insertion_of_residues = space
X,Y,Z = xyz[1]+L_half, xyz[2]+L_half, xyz[3]+L_half
occupancy = 1.0
temp_factor = 0.0
eleSym = list(xyz[0])
# list_elem = [xyz[0][i][0]+str(i) for i in range(len(xyz[0])) ]
# eleSym = list(list_elem)


for i in range(nop):

    # f = "%s\t%f\t%f\t%f\n"%(xyz[0][i],xyz[1][i]+L_half, xyz[2][i]+L_half,xyz[3][i]+L_half,)
    # file.write(f)
    # fout.write("{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}\n".format(atomsring[i],
    #                                 atomSerialNumber[i],
    #                                 atomName[i],
    #                                 alternate_location_indicator,
    #                                 residue_name[i],
    #                                 chainIDlist[i],
    #                                 residue_sequence_number[i],
    #                                 code_for_insertion_of_residues,
    #                                 X[i], Y[i], Z[i],
    #                                 occupancy,
    #                                 temp_factor,
    #                                 eleSym[i]))
    
    fout.write("{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}\n".format(atomsring[i],
                                                                                                                            atomSerialNumber[i],
                                                                                                                            atomName[i],
                                                                                                                            alternate_location_indicator,
                                                                                                                            residue_name[i],
                                                                                                                            chainIDlist[i],
                                                                                                                            residue_sequence_number[i],
                                                                                                                            code_for_insertion_of_residues,
                                                                                                                            X[i], Y[i], Z[i],
                                                                                                                            occupancy,temp_factor,eleSym[i]))
    
fout.close()
# file.close()




