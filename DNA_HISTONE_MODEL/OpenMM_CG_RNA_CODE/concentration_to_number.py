"""
This script calculates the number of ions to be added to a box of given dimensions to achieve a desired 
concentration of ions in mM.
"""

desired_concentration_of_ion_in_mM = 18 # in mM or mmol/L
box_lengthx = 200 # in Angstrom
box_lengthy = 200 # in Angstrom
box_lengthz = 200 # in Angstrom
N_Avogradro = 6.023e23

desired_concentration_of_ion_in_M = desired_concentration_of_ion_in_mM/1000
box_length_in_cm_x = box_lengthx*(1e-8)
box_length_in_cm_y = box_lengthy*(1e-8)
box_length_in_cm_z = box_lengthz*(1e-8)

box_volume_in_cm3 = box_length_in_cm_x*box_length_in_cm_y*box_length_in_cm_z
print(box_volume_in_cm3)

number_of_ions_in_the_box = (1/1000)*(desired_concentration_of_ion_in_M*box_volume_in_cm3*N_Avogradro)
print("For a box dimension of %lf,%lf,%lf (Angstrom) and concentration of %lf mM,\n the number_of_ions_in_the_box will be %ld"%(box_lengthx,box_lengthy,box_length_in_cm_z ,desired_concentration_of_ion_in_mM,int(round(number_of_ions_in_the_box))))


"""
2 mM = 10
3 mM = 14
3.5mM= 17
4 mM = 19
6 mM = 29
8 mM = 39
10 mM= 48
12 mM= 58
16 mM= 77
20 mM= 96
25 mM= 120
30 mM= 145
40 mM= 193
50 mM= 241
60 mM= 289
75 mM= 361
80 mM= 385
100mM= 482
"""
