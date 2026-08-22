"""
Moves COM of the molecule to the origin
"""
import numpy as np

file = "./6dmc_cg.xyz"
x,y,z = np.loadtxt(file, unpack=True, usecols=(1,2,3), skiprows=2)

sumx = sum(x)
sumy = sum(y)
sumz = sum(z)
n_beads = len(x)

comx, comy, comz = sumx/n_beads, sumy/n_beads, sumz/n_beads
print(comx,comy,comz)
f = open("6dmc_cg_com.dat", "w")
g = open("6dmc_cg_com.xyz", "w")
p0 = ("%d\n\n"%(n_beads))
g.write(p0)
for i in range(len(x)):
        p = ("%lf\t%lf\t%lf\n"%(x[i]-comx, y[i]-comy, z[i]-comz))
        f.write(p)
for j in range(0,len(x),3):        
        p1 = ("P\t%lf\t%lf\t%lf\n"%(x[j]-comx, y[j]-comy, z[j]-comz))
        g.write(p1)
        p1 = ("S\t%lf\t%lf\t%lf\n"%(x[j+1]-comx, y[j+1]-comy, z[j+1]-comz))
        g.write(p1)
        p1 = ("B\t%lf\t%lf\t%lf\n"%(x[j+2]-comx, y[j+2]-comy, z[j+2]-comz))
        g.write(p1)
f.close()
# g.close()
