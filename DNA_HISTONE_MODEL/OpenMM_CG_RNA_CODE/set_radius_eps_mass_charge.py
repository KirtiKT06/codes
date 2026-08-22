
import numpy as np

dt = np.dtype({'names': ['A1','A2','A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12'],
               'formats': ['U7','<i4' ,'U4', 'U4', 'U4', '<i4', '<f8','<f8','<f8','<f8','<f8','U3' ]})
a = np.loadtxt("../CreatePDB/6dmc_only_cg_com.pdb", dtype=dt, unpack=True, skiprows=0)
b = np.loadtxt('4_sequence.inp', dtype="<U1")
print(len(a[2]))

# file = open("../Production/K30mM/Mg20mM_Co8mM/8_sigma_eps_mass_charge.inp", 'w')
file = open("./3_sigma_eps_mass_charge.inp", 'w')
f = ("Type, radius(A), eps(kcal/mol), mass(amu), charge(unit)\n")
file.write(f)
j=0
for i in a[2]:
    if i == 'P':
        f = ("%s,%lf,%lf,%lf,%lf\n"%(i, 2.1, 0.2, 62.9714, -1))
        file.write(f)
    elif i=='S':
        f = ("%s,%lf,%lf,%lf,%lf\n"%(i, 2.9, 0.2, 131.1083, 0))
        file.write(f)
    elif i=='C' or i=='B':
        if b[j] == 'A':
            f = ("%s,%lf,%lf,%lf,%lf\n"%(b[j], 2.8, 0.2, 134.11876, 0))
            file.write(f)
        elif b[j] == 'G':
            f = ("%s,%lf,%lf,%lf,%lf\n"%(b[j], 3.0, 0.2, 150.11816, 0))
            file.write(f)
        elif b[j] == 'C':
            f = ("%s,%lf,%lf,%lf,%lf\n"%(b[j], 2.7, 0.2, 110.09406, 0))
            file.write(f)
        elif b[j] == 'U':
            f = ("%s,%lf,%lf,%lf,%lf\n"%(b[j], 2.7, 0.2, 111.07882, 0))
            file.write(f)
        j +=1
    elif i=='MG': ## Ion parameters are from Li paper paper page-2741 (dx.doi.org/10.1021/ct400146w|J.  Chem.  Theory  Comput.2013, 9, 2733−2748)
        f = ("%s,%lf,%lf,%lf,%lf\n"%(i, 1.353, 0.00941798, 24.305, 2))
        file.write(f)
    elif i=='CO':
        f = ("%s,%lf,%lf,%lf,%lf\n"%(i, 1.288, 0.00417787, 58.9332, 2))
        file.write(f)
    elif i=='NI':
        f = ("%s,%lf,%lf,%lf,%lf\n"%(i, 1.221, 0.00155814, 58.9332, 2))
        file.write(f)
    elif i=='ZN':
        f = ("%s,%lf,%lf,%lf,%lf\n"%(i, 1.252, 0.00250973, 58.9332, 2))
        file.write(f)
    elif i=='FE':
        f = ("%s,%lf,%lf,%lf,%lf\n"%(i, 1.343, 0.00838052, 58.9332, 2))
        file.write(f)
    elif i=='K':
        f = ("%s,%lf,%lf,%lf,%lf\n"%(i, 1.590, 0.2794651, 39.0983, 1))
        file.write(f)
    elif i=='CL':
        f = ("%s,%lf,%lf,%lf,%lf\n"%(i, 2.760, 0.0116615, 35.453, -1))
        file.write(f)

file.close()

