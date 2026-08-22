
# import numpy as np

# def consecutive_stack(Temperature, n_beads):

#     bead_type = np.loadtxt("./Inputfiles/3_sigma_eps_mass_charge.inp", dtype="U5", usecols=0, unpack=True, skiprows=1, delimiter=',')
#     # A =["GG","GA","AG","GC","GU","UG","CG","AC","CA","AU","UA","AA","CU","UC","CC","UU"]
#     Stack_type = ['UU',   'CC',  'UC',  'CU',  'AA',  'UA',  'AU',  'CA',  'AC', 'CG',  'UG',  'GU',  'GC',  'AG',  'GA',  'GG']
#     h_list     = [ 3.00,  3.66,  3.62,  3.63,  4.00,  3.92,  3.96,  3.92,  3.96, 4.23,  4.68,  4.62,  4.72,  4.76,  4.72,  5.20]
#     s_list     = [-3.56, -1.57, -1.57, -1.57, -0.32, -0.32, -0.32, -0.32, -0.32, 0.77,  2.92,  2.92,  4.37,  5.30,  5.30,  7.35]
#     Tm_list    = [-21,    13,    13,    13,    26,    26,    26,    26,    26,   42,    65,    65,    70,    68,    68,    93  ]
#     r0_list    = [ 4.25,  4.25,  4.27,  4.23,  4.18,  4.70,  3.83,  4.70,  3.83, 4.98,  5.00,  3.66,  3.68,  4.43,  4.01,  4.24]

#     # print(bead_type)

#     Temp = float(Temperature)
#     nbeads = int(n_beads)

#     phi10 = -2.58684
#     phi20 = 3.07135

#     U0, R0, Phi10, Phi20 = [],[],[],[]

#     for i in range(len(bead_type)):
#         #if i in [497,500]: ### Skipping residue 167 of RNA and 1 of tRNA chain which are not consecutive stack pairs
#         #    U0.append(0)
#         #    R0.append(0)
#         #    Phi10.append(0)
#         #    Phi20.append(0)
#         # if i > 2 and i <(nbeads-5):
#         #elif i <(nbeads-5):
#         if i <(nbeads-5):
#             # print((bead_type[i]+bead_type[i+3]))
#             if (bead_type[i]+bead_type[i+3] in Stack_type):
#                 j = Stack_type.index(bead_type[i]+bead_type[i+3])
#                 # print("j=",j)
#                 u0 = -h_list[j] + 0.001987*((Temp-273.15) - Tm_list[j])*s_list[j]

#                 U0.append(u0)
#                 R0.append(r0_list[j])
#                 Phi10.append(phi10)
#                 Phi20.append(phi20)

#     return U0, R0, Phi10, Phi20

import numpy as np

bead_type = np.loadtxt("3_sigma_eps_mass_charge.inp", dtype="U5", usecols=0, unpack=True, skiprows=1, delimiter=',')
# A =["GG","GA","AG","GC","GU","UG","CG","AC","CA","AU","UA","AA","CU","UC","CC","UU"]
Stack_type = ['AA', 'GG', 'CC', 'TT', 'TC', 'CT', 'TA', 'AT', 'CA', 'AC', 'CG', 'GC', 'TG', 'GT', 'AG', 'GA']
h_list = [4.67, 5.13, 4.15, 4.17, 4.13, 4.18, 4.18, 4.28, 4.21, 4.24, 4.73, 4.81, 4.70, 4.86, 4.82, 4.83 ]
s_list = [0.94, -0.29, 0.98, 0.89, 0.71, 0.94, 0.87, 0.65, 0.92, 0.79, 2.41, 2.33, 1.69, 1.73, 0.98, 1.06 ]
Tm_list = [322.0, 353.9, 288.3, 288.3, 288.3, 288.3, 293.0, 293.0, 293.0, 293.0, 331.9, 331.9, 332.6, 332.6, 333.6, 333.6]
r0_list= [3.74, 3.67, 4.04, 4.02, 3.79, 3.79, 4.10, 4.10, 4.06, 4.06, 3.98, 3.98, 3.83, 3.83, 3.79, 3.79 ]
# print(bead_type)

Temp = 273.15 + 25

nbeads = 78
phi10 = -2.6943
phi20 = 3.0866

file=open("5_stack_T%d.inp"%Temp,"w")

for i in range(len(bead_type)):
    if i > 2 and i <(nbeads-5):
        print((bead_type[i]+bead_type[i+3]))
        if (bead_type[i]+bead_type[i+3] in Stack_type):
            j = Stack_type.index(bead_type[i]+bead_type[i+3])
            print("j=",j)
            u0 = -h_list[j] + 0.001987*((Temp) - Tm_list[j])*s_list[j]

            f = "%s,%f,%f,%f,%f\n"%(bead_type[i],u0, r0_list[j], phi10, phi20)
            # f = "%s,%f,%f,%f\n"%(bead_type[i],float(0),float(0),float(0),r0_list[j], phi10, phi20)
            file.write(f)
        else:
            f = "%s,%f,%f,%f,%f\n"%(bead_type[i],float(0),float(0),float(0),float(0))
            # file.write(f)
    else:
        f = "%s,%f,%f,%f,%f\n"%(bead_type[i],float(0),float(0),float(0),float(0))
        # file.write(f)

file.close()
print("Temperature: ", Temp)