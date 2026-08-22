"""
This script generates the harmonic angle input file for the coarse-grained RNA model. 
It reads the RNA sequence from a file and calculates the number of residues, beads, and angles. 
It then writes the harmonic angle parameters to an output file based on the sequence."""
import numpy as np
seq = np.loadtxt("4_sequence.inp", dtype= 'U2')
print (seq)
no_of_residues = len(seq)
no_of_beads = 3*no_of_residues
no_of_angles = 4*no_of_residues - 3

print("Nummber of residues = %d \n"%no_of_residues)
print("Nummber of beads = %d \n"%no_of_beads)
print("Nummber of angles = %d \n"%no_of_angles)
outfile = open("2_harmonic_angles.inp", "w")

# Force constant(k_theta in kcal/(mol.rad^2) )
k_P1S1B = 5
k_P2S1B = 5
k_P1S1P2 = 20
k_S1P2S2 = 20

# Equilibrium distance theta0 in rad
a_P1_S1_P2 = 1.5116
a_S1_P2_S2 = 1.5892
a_P1_S1_G  = 1.7298
a_P2_S1_G  = 1.8552
a_P1_S1_C  = 1.5198
a_P2_S1_C  = 1.9004
a_P1_S1_U  = 1.5126
a_P2_S1_U  = 1.9012
a_P1_S1_A  = 1.6595
a_P2_S1_A  = 1.8659

for i_ang in range(1,no_of_beads+1,3):

    if i_ang == no_of_beads-2:
        print("i = ", i_ang)
        print(i_ang-1, i_ang, i_ang+1, "P1 - S1 -", seq[i_ang//3])

        if   seq[i_ang//3] == 'G': a_P1_S1_B = a_P1_S1_G
        elif seq[i_ang//3] == 'C': a_P1_S1_B = a_P1_S1_C
        elif seq[i_ang//3] == 'A': a_P1_S1_B = a_P1_S1_A
        elif seq[i_ang//3] == 'U': a_P1_S1_B = a_P1_S1_U
        # p = ("%d,%d,%d,%.2lf,%.4lf\n"%(i_ang-1, i_ang, i_ang+1, k_P1S1B, a_P1_S1_B))
        p = ("%d,%d,%d,%.2lf,%.4lf\n"%(i_ang-1, i_ang, i_ang+1, k_P1S1B, a_P1_S1_B))
        outfile.write(p)
  
    else:    
        print("i = ", i_ang)
        print(i_ang-1, i_ang, i_ang+1, "P1 - S1 -", seq[i_ang//3])
        print(i_ang+1, i_ang, i_ang+2, "P2 - S1 -", seq[i_ang//3])
        print(i_ang-1, i_ang, i_ang+2, "P1 - S1 - P2")
        print(i_ang, i_ang+2, i_ang+3, "S1 - P2 - S2" )

        if seq[i_ang//3] == 'G': a_P1_S1_B = a_P1_S1_G
        elif seq[i_ang//3] == 'C': a_P1_S1_B = a_P1_S1_C
        elif seq[i_ang//3] == 'A': a_P1_S1_B = a_P1_S1_A
        elif seq[i_ang//3] == 'U': a_P1_S1_B = a_P1_S1_U
        # p = ("%d,%d,%d,%.2lf,%.4lf\n"%(i_ang-1, i_ang, i_ang+1, k_P1S1B, a_P1_S1_B))
        p = ("%d,%d,%d,%.2lf,%.4lf\n"%(i_ang-1, i_ang, i_ang+1, k_P1S1B, a_P1_S1_B))
        outfile.write(p)

        if   seq[i_ang//3] == 'G': a_P2_S1_B = a_P2_S1_G
        elif seq[i_ang//3] == 'C': a_P2_S1_B = a_P2_S1_C
        elif seq[i_ang//3] == 'A': a_P2_S1_B = a_P2_S1_A
        elif seq[i_ang//3] == 'U': a_P2_S1_B = a_P2_S1_U
        # p = ("%d,%d,%d,%.2lf,%.4lf\n"%(i_ang+1, i_ang, i_ang+2, k_P2S1B, a_P2_S1_B))
        p = ("%d,%d,%d,%.2lf,%.4lf\n"%(i_ang+1, i_ang, i_ang+2, k_P2S1B, a_P2_S1_B))
        outfile.write(p)

        # p = ("%d,%d,%d,%.2lf,%.4lf\n"%(i_ang-1, i_ang, i_ang+2, k_P1S1P2, a_P1_S1_P2))
        p = ("%d,%d,%d,%.2lf,%.4lf\n"%(i_ang-1, i_ang, i_ang+2, k_P1S1P2, a_P1_S1_P2))
        outfile.write(p)

        # p = ("%d,%d,%d,%.2lf,%.4lf\n"%(i_ang, i_ang+2, i_ang+3, k_S1P2S2, a_S1_P2_S2))
        p = ("%d,%d,%d,%.2lf,%.4lf\n"%(i_ang, i_ang+2, i_ang+3, k_S1P2S2, a_S1_P2_S2))
        outfile.write(p)

outfile.close()