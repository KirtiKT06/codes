import numpy as np
seq = np.loadtxt("4_sequence.inp", dtype= 'U2')
print (seq)
no_of_residues = len(seq)
no_of_beads = 3*no_of_residues
no_of_bonds = no_of_beads - 1
print("Nummber of residues = %d \n"%no_of_residues)
print("Nummber of beads = %d \n"%no_of_beads)
print("Nummber of bonds = %d \n"%no_of_bonds)
outfile = open("1_harmonic_bonds.inp", "w")

# Force constant(k_r in kcal/(mol.A^2) )
k_PS = 23
k_SP = 64
k_SB = 10

# Equilibrium distance r0 in A
d_PS = 4.5318
d_SP = 3.6708
d_SG = 4.7667
d_SC = 4.1272
d_SU = 4.1288
d_SA = 4.6712

for i_bond in range(0,no_of_bonds+1,3):

    if i_bond == no_of_bonds-2:       
        print("i = ", i_bond)
        print(i_bond, i_bond+1, 23, "P -- S")
        print(i_bond+1, i_bond+2, 10, "S --", seq[i_bond//3])

        # p = ("%d,%d,%.2lf,%.5lf\n"%(i_bond, i_bond+1, k_PS, d_PS))
        p = ("%d,%d,%.2lf,%.5lf\n"%(i_bond, i_bond+1, k_PS, d_PS))
        outfile.write(p)
        if seq[i_bond//3] == 'G': d_SB = d_SG
        elif seq[i_bond//3] == 'C': d_SB = d_SC
        elif seq[i_bond//3] == 'A': d_SB = d_SA
        elif seq[i_bond//3] == 'U': d_SB = d_SU
        # p = ("%d,%d,%.2lf,%.5lf\n"%(i_bond+1, i_bond+2, k_SB, d_SB,))
        p = ("%d,%d,%.2lf,%.5lf\n"%(i_bond+1, i_bond+2, k_SB, d_SB,))
        outfile.write(p)
  
    else:    
        print("i = ", i_bond)
        print(i_bond, i_bond+1, 23, "P -- S")
        print(i_bond+1, i_bond+2, 10, "S --", seq[i_bond//3])
        print(i_bond+1, i_bond+3, 64, "S -- P")

        # p = ("%d,%d,%.2lf,%5lf\n"%(i_bond, i_bond+1, k_PS, d_PS))
        p = ("%d,%d,%.2lf,%5lf\n"%(i_bond, i_bond+1, k_PS, d_PS))
        outfile.write(p)
        if seq[i_bond//3] == 'G': d_SB = d_SG
        elif seq[i_bond//3] == 'C': d_SB = d_SC
        elif seq[i_bond//3] == 'A': d_SB = d_SA
        elif seq[i_bond//3] == 'U': d_SB = d_SU
        # p = ("%d,%d,%.2lf,%.5lf\n"%(i_bond+1, i_bond+2, k_SB, d_SB))
        p = ("%d,%d,%.2lf,%.5lf\n"%(i_bond+1, i_bond+2, k_SB, d_SB))
        outfile.write(p)
        # p = ("%d,%d,%.2lf,%.5lf\n"%(i_bond+1, i_bond+3, k_SP, d_SP))
        p = ("%d,%d,%.2lf,%.5lf\n"%(i_bond+1, i_bond+3, k_SP, d_SP))
        outfile.write(p)

outfile.close()