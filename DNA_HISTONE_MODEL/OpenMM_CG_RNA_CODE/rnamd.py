from __future__ import print_function
import openmm as mm
import openmm.app as app
import openmm.unit as u
# from app import ReducedStateDataReporter
import numpy as np
import os, sys
#import parmed as pmd
import json
from sys import platform
import random
import time
from datetime import datetime                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
from set_stack_input import consecutive_stack


from datetime import datetime
start_time = datetime.now()

print("|---------------------------------------|")
print("         OpenMM Version: ",mm.__version__)
print("|_______________________________________|\n")
# do your work here



##**************************************##
###    Last updated on Jul 15, 2024    ###
##   Integrated Stacking with the code  ##
#----------------------------------------#  
###        Updated on Feb 21, 2024     ###
##      Corrected SS-Hbond index        ##
#----------------------------------------#
###        Updated on Nov 3,  2023      ###
## removing ions from the               ##    
## 8_radius_eps_mass_charge.inp file    ##
#----------------------------------------#
###        Updated on July 6, 2023     ###
## added nCO                            ##
#----------------------------------------#
###       Updated on June 13, 2023     ###
## Added CMMotionRemover()              ##
## Added angle cusp constraint          ##
##**************************************##

# random.seed(100)
# np.random.seed(100)

### INPUT READING ###
#-------------------#
configdic = json.load(open('input.json'))

### Parameters that set file names
platform_type    = configdic["platform_type"   ] # CPU, Reference, GPU
numsteps         = configdic['numsteps'        ] # Total time step
data_interval    = configdic['data_interval'   ] # Data saving frequency
snap_interval    = configdic['snap_interval'   ] # Snap saving frequency
pdb_prefix       = configdic['pdb_prefix'      ] # Prefix to initial coordinate PDB file
dcd_prefix       = configdic['dcd_prefix'      ] # name of the DCD file excluding the extension ".dcd"
data_name        = configdic['data_name'       ] # name of the data file including the desired extension name (e.g. ".dat")
N_nucleic        = configdic['N_nucleic'       ]
nMg              = configdic['nMg'             ]
nCo              = configdic['nCo'             ]
nK               = configdic['nK'              ]
nCl              = configdic['nCl'             ]
rMg              = configdic['rMg'             ] # in Angstrom
rCo              = configdic['rCo'             ] # in Angstrom
rK               = configdic['rK'              ] # in Angstrom
rCl              = configdic['rCl'             ] # in Angstrom
epsMg            = configdic['epsMg'           ] # in Kcal/mol
epsCo            = configdic['epsCo'           ] # in Kcal/mol
epsK             = configdic['epsK'            ] # in Kcal/mol
epsCl            = configdic['epsCl'           ] # in Kcal/mol
mMg              = configdic['mMg'             ] # in amu
mCo              = configdic['mCo'             ] # in amu
mK               = configdic['mK'              ] # in amu
mCl              = configdic['mCl'             ] # in amu
qMg              = configdic['qMg'             ] # in proton charge
qCo              = configdic['qCo'             ] # in proton charge
qK               = configdic['qK'              ] # in proton charge
qCl              = configdic['qCl'             ] # in proton charge
u0HBond          = configdic['u0HBond'         ] # in Kcal/mol
u0TStack         = configdic['u0TStack'        ] # in Kcal/mol
LJ_cutoff        = configdic['LJ_cutoff'       ] # in Angstrom
L                = configdic['box_length'      ] # in Angstrom
Temp             = configdic['Temp'            ] # in Kelvin
Boltz_Const      = configdic['Boltz_Const'     ]
s_rna_rna        = configdic['s_rna_rna'       ] # in Angstrom (Only specific for RNA-RNA interactions)
rSmallestIon     = configdic['rSmallestIon'    ] # in Angstrom (Needed for the modified LJ interactions)
friction_coeff   = configdic['friction_coeff'  ] # Low Friction 1/ps, Water Friction~ 91/ps
time_step        = configdic['time_step'       ] # Integration time step (dt)
final_state      = configdic["final_state"     ] # if True then only final snap will be saved
minimization     = configdic['minimization'    ] # if True then only minimization takes place
restart          = configdic['restart'         ] # if True then only minimization takes place
ComMotionRemover = configdic['ComMotionRemover'] # if True then only Center of mass motion gets rescaled
CMMR_frequency   = configdic['CMMR_frequency'  ] # frequency at which ComMotionRemover works
# ----------------------------------------------- #
pi_cut     = 3.124139361
nbeads     = 3*int(N_nucleic)
nIon      = nMg + nCo + nK + nCl
nop        = nbeads+ nIon
box_length = L*u.angstrom
T          = Temp*u.kelvin
kB         = Boltz_Const*(u.kilocalorie_per_mole/u.kelvin)
##***********************************************##


#### READING PDB FOR INITIAL CONFIGURATION ####

pdb = app.PDBFile('%s.pdb'%pdb_prefix)
positions = pdb.positions
# print(positions[0])

####****************************###
####  READING Parameter Files  ####


BondList = np.loadtxt("./Inputfiles/1_harmonic_bonds.inp", unpack=True, delimiter=',')
AngleList= np.loadtxt("./Inputfiles/2_harmonic_angles.inp",unpack=True, delimiter=',')
radius, epsilon, mass, charge = np.loadtxt("./Inputfiles/3_sigma_eps_mass_charge.inp", usecols=(1,2,3,4), unpack=True, skiprows=1, delimiter=',')
# u0ST, rST, pST1, pST2=np.loadtxt("5_stack.inp", usecols=(1,2,3,4), unpack=True, delimiter=',')
u0ST, rST, pST1, pST2= consecutive_stack(Temp, nbeads)

stack_input = open("Stack_input.out", 'w')
for i in range(len(u0ST)):
     f = "%f,%f,%f,%f\n"%(u0ST[i], rST[i], pST1[i], pST2[i])
     stack_input.write(f)
stack_input.close()

BB_hbond = np.loadtxt("./HbondParameterFile/BB_param_5tpy.inp", unpack=True, delimiter=',')
BS_hbond = np.loadtxt("./HbondParameterFile/BS_param_5tpy.inp", unpack=True, delimiter=',')
PS_hbond = np.loadtxt("./HbondParameterFile/PS_param_5tpy.inp", unpack=True, delimiter=',')
SS_hbond = np.loadtxt("./HbondParameterFile/SS_param_5tpy.inp", unpack=True, delimiter=',')
TStack   = np.loadtxt("./HbondParameterFile/TS_param_5tpy.inp", unpack=True, delimiter=',')

print("BB_hbond: ",len(BB_hbond[0]))
print("BS_hbond: ",len(BS_hbond[0]))
print("PS_hbond: ",len(PS_hbond[0]))
print("SS_hbond: ",len(SS_hbond[0]))
print("TStack  : ",len(TStack[0]))

# print(hST)
# print(BondList)
##*************************************************##
############ Building OpenMM simulation #############
print('\n       Constructing OpenMM simulation')
print("========================================")
system = mm.System()
system.setDefaultPeriodicBoxVectors(mm.Vec3(box_length, 0, 0), mm.Vec3(0, box_length, 0), mm.Vec3(0, 0, box_length))

for i in range(nbeads):
    system.addParticle(float(mass[i])*u.amu)
for i in range(nMg):
    system.addParticle(float(mMg)*u.amu)
for i in range(nCo):
    system.addParticle(float(mCo)*u.amu)
for i in range(nK):
    system.addParticle(float(mK)*u.amu)
for i in range(nCl):
    system.addParticle(float(mCl)*u.amu)

HarmonicBondForce = mm.HarmonicBondForce()
HarmonicAngleForce= mm.HarmonicAngleForce()

for i in range(len(BondList[0])):
    # print(int(BondList[0][i]), int(BondList[1][i]), float(BondList[3][i]), 2.0*float(BondList[2][i]))
    HarmonicBondForce.addBond(int(BondList[0][i]), 
                              int(BondList[1][i]), 
                              float(BondList[3][i])*u.angstrom, 
                              2.0*float(BondList[2][i])*u.kilocalorie_per_mole/(u.angstrom)**2)

for i in range(len(AngleList[0])):
    # print(int(AngleList[0][i]), int(AngleList[1][i]), float(AngleList[2][i]), 
    #                                               float(AngleList[4][i]), 2.0*float(AngleList[3][i]))
    HarmonicAngleForce.addAngle(int(AngleList[0][i]), 
                                int(AngleList[1][i]), 
                                int(AngleList[2][i]), 
                                float(AngleList[4][i])*u.radians, 
                                2.0*float(AngleList[3][i])*(u.kilocalories_per_mole/(u.radians)**2))


diamg=2.0*rSmallestIon*u.angstroms      ##
# diamg=2.0*0.8*u.angstroms    ## Natalia
# diamg=2.0*1.3530*u.angstroms ## Habib

#*****************************************************************#

######LJ for rna and rna######
#****************************#
r_rna_rna = s_rna_rna*u.angstroms
LJForce_rna_rna = mm.CustomNonbondedForce("select(step(radii_rna_rna-r)*eps_rna_rna, eps_rna_rna*(((dia_mg/(r+dia_mg-radii_rna_rna))^12)-(2.0*((dia_mg/(r+dia_mg-radii_rna_rna))^6))+1), 0); \
                                           eps_rna_rna = sqrt(eps_rna_rna1*eps_rna_rna2);")

LJForce_rna_rna.addPerParticleParameter('eps_rna_rna')
LJForce_rna_rna.addGlobalParameter('dia_mg', diamg)
LJForce_rna_rna.addGlobalParameter('radii_rna_rna', r_rna_rna)
LJForce_rna_rna.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
LJForce_rna_rna.setCutoffDistance(float(LJ_cutoff)*u.angstrom)

for i in range(nbeads):
    LJForce_rna_rna.addParticle([float(epsilon[i])*u.kilocalorie_per_mole])
for i in range(nbeads,nop):
    LJForce_rna_rna.addParticle([float(0.0)*u.kilocalorie_per_mole])

for i in range(len(BondList[0])):
    LJForce_rna_rna.addExclusion(BondList[0][i],BondList[1][i])

# for i in range(LJForce_rna_rna.getNumExclusions()):
#     print(LJForce_rna_rna.getExclusionParticles(i))  

#******************************************************************#

######LJ for ion and ion######
#****************************#
LJForce_ion_ion = mm.CustomNonbondedForce("select(step(radii_ion_ion - r)*eps_ion_ion, eps_ion_ion*(((dia_mg/(r+dia_mg-radii_ion_ion))^12)-(2.0*((dia_mg/(r+dia_mg-radii_ion_ion))^6))+1), 0); \
                                    eps_ion_ion = sqrt(eps_ion_ion1*eps_ion_ion2); \
                                    radii_ion_ion = (radii_ion_ion1+radii_ion_ion2);" )

LJForce_ion_ion.addPerParticleParameter('radii_ion_ion')
LJForce_ion_ion.addPerParticleParameter('eps_ion_ion')
LJForce_ion_ion.addGlobalParameter('dia_mg', diamg)
LJForce_ion_ion.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
LJForce_ion_ion.setCutoffDistance(float(LJ_cutoff)*u.angstrom)


for i in range(0,nbeads):
    LJForce_ion_ion.addParticle([float(radius[i])*u.angstroms, float(0)*u.kilocalorie_per_mole])
for j in range(0,nMg):
    LJForce_ion_ion.addParticle([float(rMg)*u.angstroms, float(epsMg)*u.kilocalorie_per_mole])
for j in range(0,nCo):
    LJForce_ion_ion.addParticle([float(rCo)*u.angstroms, float(epsCo)*u.kilocalorie_per_mole])
for j in range(0,nK):
    LJForce_ion_ion.addParticle([float(rK)*u.angstroms, float(epsK)*u.kilocalorie_per_mole])
for j in range(0,nCl):
    LJForce_ion_ion.addParticle([float(rCl)*u.angstroms, float(epsCl)*u.kilocalorie_per_mole])

for i in range(len(BondList[0])):
    LJForce_ion_ion.addExclusion(BondList[0][i],BondList[1][i])

# for i in range(LJForce_ion_ion.getNumExclusions()):
#     LJForce_ion_ion.getExclusionParticles(i) 

#*****************************************************************#

######LJ for RNA and ion######
#****************************#
LJForce_rna_ion = mm.CustomNonbondedForce("select(factor*step(radii_rna_ion - r)*eps_rna_ion, factor*eps_rna_ion*(((dia_mg/(r+dia_mg-radii_rna_ion))^12)-(2.0*((dia_mg/(r+dia_mg-radii_rna_ion))^6))+1), 0); \
                                           eps_rna_ion = sqrt(eps_rna_ion1*eps_rna_ion2); \
                                           radii_rna_ion = (radii_rna_ion1 + radii_rna_ion2);\
                                           factor = (abs(factor1-factor2))/2" )

LJForce_rna_ion.addPerParticleParameter('radii_rna_ion')
LJForce_rna_ion.addPerParticleParameter('eps_rna_ion')
LJForce_rna_ion.addPerParticleParameter('factor')
LJForce_rna_ion.addGlobalParameter('dia_mg', diamg)
LJForce_rna_ion.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
LJForce_rna_ion.setCutoffDistance(float(LJ_cutoff)*u.angstrom)

type_index = [1, -1] ## rna_index: 1, ion_index: -1
for i in range(0,nbeads):
    LJForce_rna_ion.addParticle([float(radius[i])*u.angstroms, float(epsilon[i])*u.kilocalorie_per_mole, float(type_index[0])])
for i in range(nMg):
    LJForce_rna_ion.addParticle([float(rMg)*u.angstroms, float(epsMg)*u.kilocalorie_per_mole, float(type_index[1])])

for i in range(nCo):
    LJForce_rna_ion.addParticle([float(rCo)*u.angstroms, float(epsCo)*u.kilocalorie_per_mole, float(type_index[1])])

for i in range(nK):
    LJForce_rna_ion.addParticle([float(rK)*u.angstroms,  float(epsK)*u.kilocalorie_per_mole,  float(type_index[1])])

for i in range(nCl):
    LJForce_rna_ion.addParticle([float(rCl)*u.angstroms, float(epsCl)*u.kilocalorie_per_mole, float(type_index[1])])

for i in range(len(BondList[0])):
    LJForce_rna_ion.addExclusion(BondList[0][i],BondList[1][i])

# for i in range(LJForce_rna_ion.getNumExclusions()):
#     LJForce_rna_ion.getExclusionParticles(i)

#*****************************************************************#
####STACKING INTERACTION POETNTIAL####

krST= 1.4/(u.angstrom)**2
kphi1ST = 4.0/(u.radian)**2
kphi2ST = 4.0/(u.radian)**2
pi = np.pi

StackEnergyExpression = "u0ST /(1.0 + kr*delRsq + kphi1*delphi1sq + kphi2*delphi2sq);"
StackEnergyExpression+= "delRsq = delR*delR; delR = (RST-R0ST); RST = distance(p3,p6);"

StackEnergyExpression+= "delphi1sq = delphi1*delphi1;"
"""
#The Following expression is wrong because of the first term in the right hand side expresion. The step function of
#any abs function is always one. It does not produce zero. You can modify the expression using one of these two 
#expression given below.
StackEnergyExpression+= "delphi1 = dphi1*step(abs(pi-dphi1)) +(dphi1-(2.0*pi))*step(dphi1-pi) +(dphi1+(2.0*pi))*step(-dphi1-pi);"
#You can choose one of these two expressions for delphi1. Both expressions give the same value
"""
StackEnergyExpression+= "delphi1 = dphi1 - 2.0*pi*step(dphi1-pi) + 2.0*pi*step(-dphi1-pi);"
"""Alternative expression"""
# StackEnergyExpression+= "delphi1 = dphi1 - 2.0*pi*(dphi1/(abs(dphi1)))*step(abs(dphi1)-pi);"
StackEnergyExpression+= "dphi1 =(phi1ST-phi10ST); phi1ST= dihedral(p1,p2,p4,p5);"



StackEnergyExpression+= "delphi2sq = delphi2*delphi2;"
#Same as delphi1. Use one of these two expressions
StackEnergyExpression+= "delphi2 = dphi2 - 2.0*pi*step(dphi2-pi) + 2.0*pi*step(-dphi2-pi);"
"""Alternative expression"""
# StackEnergyExpression+= "delphi2 = dphi2 - 2.0*pi*(dphi2/(abs(dphi2)))*step(abs(dphi2)-pi);
StackEnergyExpression+= "dphi2 =(phi2ST-phi20ST); phi2ST= dihedral(p2,p4,p5,p7);"

StackForce = mm.CustomCompoundBondForce(7, StackEnergyExpression)
StackForce.addPerBondParameter("u0ST")
StackForce.addPerBondParameter("R0ST")
StackForce.addPerBondParameter("phi10ST")
StackForce.addPerBondParameter("phi20ST")
StackForce.addGlobalParameter("kr", krST)
StackForce.addGlobalParameter("kphi1", kphi1ST)
StackForce.addGlobalParameter("kphi2", kphi2ST)
StackForce.addGlobalParameter("pi", pi)

for j in range(1,(1+len(u0ST))):
    i = 3*j+2
    StackForce.addBond([i-2,i-1,i,i+1,i+2,i+3,i+4],    [float(u0ST[j-1])*u.kilocalorie_per_mole,
                                                        float( rST[j-1])*u.angstroms,
                                                        float(pST1[j-1])*u.radian,
                                                        float(pST2[j-1])*u.radian])
    
#----------------------------------------------------#
#### HBOND INTERACTION POTENTIAL ####

u0HB    = float(u0HBond)*u.kilocalorie_per_mole
krHB    = 5.0/(u.angstrom)**2
ktheta1 = 1.5/(u.radian)**2
ktheta2 = 1.5/(u.radian)**2
kphi0HB = 0.15/(u.radian)**2
kphi1HB = 0.15/(u.radian)**2
kphi2HB = 0.15/(u.radian)**2
pi = np.pi


HBEnergyExpression = "u0HB*nHB*step(pi_cut-theta1HB)*step(pi_cut-theta2HB)*exp(-u1);"
HBEnergyExpression += "u1 = (kr_hb*delRsq + ktheta1_hb*deltheta1sq + ktheta2_hb*deltheta2sq + kphi0_hb*delphi0sq + kphi1_hb*delphi1sq + kphi2_hb*delphi2sq);"

HBEnergyExpression+= "delRsq = delR*delR; delR = (RHB-R0HB); RHB = distance(p3,p4);"
# ------------------------------------------ #
HBEnergyExpression+= "deltheta1sq = deltheta1*deltheta1;"
HBEnergyExpression+= "deltheta1 = dtheta1;"
# HBEnergyExpression+= "deltheta1 = dtheta1 - 2.0*pi*step(dtheta1-pi) + 2.0*pi*step(-dtheta1-pi);"
# HBEnergyExpression+= "deltheta1 = dtheta1 - 2.0*pi*(dtheta1/(abs(dtheta1)))*step(abs(dtheta1)-pi);"
HBEnergyExpression+= "dtheta1 =(theta1HB-theta10HB); theta1HB= angle(p2,p3,p4);"

HBEnergyExpression+= "deltheta2sq = deltheta2*deltheta2;"
HBEnergyExpression+= "deltheta2 = dtheta2;"
# HBEnergyExpression+= "deltheta2 = dtheta2 - 2.0*pi*step(dtheta2-pi) + 2.0*pi*step(-dtheta2-pi);"
# HBEnergyExpression+= "deltheta1 = dtheta1 - 2.0*pi*(dtheta1/(abs(dtheta1)))*step(abs(dtheta1)-pi);"
HBEnergyExpression+= "dtheta2 =(theta2HB-theta20HB); theta2HB= angle(p3,p4,p5);"
# ------------------------------------------ #

HBEnergyExpression+= "delphi0sq = delphi0*delphi0;"
HBEnergyExpression+= "delphi0 = dphi0 - 2.0*pi*step(dphi0-pi) + 2.0*pi*step(-dphi0-pi);"
# HBEnergyExpression+= "delphi0 = dphi0 - 2.0*pi*(dphi0/(abs(dphi0)))*step(abs(dphi0)-pi);"
HBEnergyExpression+= "dphi0 =(phi0HB-phi00HB); phi0HB= dihedral(p2,p3,p4,p5);"

HBEnergyExpression+= "delphi1sq = delphi1*delphi1;"
HBEnergyExpression+= "delphi1 = dphi1 - 2.0*pi*step(dphi1-pi) + 2.0*pi*step(-dphi1-pi);"
# HBEnergyExpression+= "delphi1 = dphi1 - 2.0*pi*(dphi1/(abs(dphi1)))*step(abs(dphi1)-pi);"
HBEnergyExpression+= "dphi1 =(phi1HB-phi10HB); phi1HB= dihedral(p1,p2,p3,p4);"


HBEnergyExpression+= "delphi2sq = delphi2*delphi2;"
HBEnergyExpression+= "delphi2 = dphi2 - 2.0*pi*step(dphi2-pi) + 2.0*pi*step(-dphi2-pi);"
# HBEnergyExpression+= "delphi2 = dphi2 - 2.0*pi*(dphi2/(abs(dphi2)))*step(abs(dphi2)-pi);
HBEnergyExpression+= "dphi2 =(phi2HB-phi20HB); phi2HB= dihedral(p3,p4,p5,p6);"
# ------------------------------------------ #

## holo Cannonical
HBForce_BB = mm.CustomCompoundBondForce(6, HBEnergyExpression)
HBForce_BB.addPerBondParameter("R0HB")
HBForce_BB.addPerBondParameter("theta10HB")
HBForce_BB.addPerBondParameter("theta20HB")
HBForce_BB.addPerBondParameter("phi00HB")
HBForce_BB.addPerBondParameter("phi10HB")
HBForce_BB.addPerBondParameter("phi20HB")
HBForce_BB.addPerBondParameter("nHB")
HBForce_BB.addGlobalParameter("u0HB", u0HB)
HBForce_BB.addGlobalParameter("kr_hb", krHB)
HBForce_BB.addGlobalParameter("ktheta1_hb", ktheta1)
HBForce_BB.addGlobalParameter("ktheta2_hb", ktheta2)
HBForce_BB.addGlobalParameter("kphi0_hb", kphi0HB)
HBForce_BB.addGlobalParameter("kphi1_hb", kphi1HB)
HBForce_BB.addGlobalParameter("kphi2_hb", kphi2HB)
HBForce_BB.addGlobalParameter("pi", pi)
HBForce_BB.addGlobalParameter("pi_cut", pi_cut)

# for j in range(0,len(BB_hbond[0])):
#     a,b = int(BB_hbond[0][j]),int(BB_hbond[1][j])
#     HBForce_BB.addBond([a+1,a-1,a,b,b-1,b+1],[float( BB_hbond[2][j])*u.angstroms,
#                                               float( BB_hbond[3][j])*u.radian,
#                                               float( BB_hbond[4][j])*u.radian,
#                                               float( BB_hbond[5][j])*u.radian,
#                                               float( BB_hbond[6][j])*u.radian,
#                                               float( BB_hbond[7][j])*u.radian,
#                                               int(   BB_hbond[8][j])])
    
# --- MODIFIED LOOP FOR ALL-TO-ALL AND TERMINAL SAFETY ---
for j in range(0,len(BB_hbond[0])):
    a = int(BB_hbond[0][j])
    b = int(BB_hbond[1][j])
    
    # SAFEGUARD: Prevent 'a+1' or 'b+1' from bleeding into the ion arrays.
    # nbeads is defined earlier in your script as 3 * int(N_nucleic)
    if (a + 1) >= nbeads or (b + 1) >= nbeads:
        continue # Skip this bond to avoid crashing or calculating dihedrals with ions
        
    HBForce_BB.addBond([a+1, a-1, a, b, b-1, b+1], [
        float(BB_hbond[2][j])*u.angstroms,
        float(BB_hbond[3][j])*u.radian,
        float(BB_hbond[4][j])*u.radian,
        float(BB_hbond[5][j])*u.radian,
        float(BB_hbond[6][j])*u.radian,
        float(BB_hbond[7][j])*u.radian,
        int(BB_hbond[8][j])
    ])


## holo base-sugar Hbond
HBForce_BS = mm.CustomCompoundBondForce(6, HBEnergyExpression)
HBForce_BS.addPerBondParameter("R0HB")
HBForce_BS.addPerBondParameter("theta10HB")
HBForce_BS.addPerBondParameter("theta20HB")
HBForce_BS.addPerBondParameter("phi00HB")
HBForce_BS.addPerBondParameter("phi10HB")
HBForce_BS.addPerBondParameter("phi20HB")
HBForce_BS.addPerBondParameter("nHB")
HBForce_BS.addGlobalParameter("u0HB", u0HB)
HBForce_BS.addGlobalParameter("kr_hb", krHB)
HBForce_BS.addGlobalParameter("ktheta1_hb", ktheta1)
HBForce_BS.addGlobalParameter("ktheta2_hb", ktheta2)
HBForce_BS.addGlobalParameter("kphi0_hb", kphi0HB)
HBForce_BS.addGlobalParameter("kphi1_hb", kphi1HB)
HBForce_BS.addGlobalParameter("kphi2_hb", kphi2HB)
HBForce_BS.addGlobalParameter("pi", pi)
HBForce_BS.addGlobalParameter("pi_cut", pi_cut)

for j in range(0,len(BS_hbond[0])):
    b = int(BS_hbond[0][j])
    s = int(BS_hbond[1][j])
    HBForce_BS.addBond([b+1,b-1,b,s,s+2,s+3],[float( BS_hbond[2][j])*u.angstroms,
                                              float( BS_hbond[3][j])*u.radian,
                                              float( BS_hbond[4][j])*u.radian,
                                              float( BS_hbond[5][j])*u.radian,
                                              float( BS_hbond[6][j])*u.radian,
                                              float( BS_hbond[7][j])*u.radian,
                                              int(   BS_hbond[8][j])])

# ## holo phosphate-sugar Hbond
HBForce_PS = mm.CustomCompoundBondForce(6, HBEnergyExpression)
HBForce_PS.addPerBondParameter("R0HB")
HBForce_PS.addPerBondParameter("theta10HB")
HBForce_PS.addPerBondParameter("theta20HB")
HBForce_PS.addPerBondParameter("phi00HB")
HBForce_PS.addPerBondParameter("phi10HB")
HBForce_PS.addPerBondParameter("phi20HB")
HBForce_PS.addPerBondParameter("nHB")
HBForce_PS.addGlobalParameter("u0HB", u0HB)
HBForce_PS.addGlobalParameter("kr_hb", krHB)
HBForce_PS.addGlobalParameter("ktheta1_hb", ktheta1)
HBForce_PS.addGlobalParameter("ktheta2_hb", ktheta2)
HBForce_PS.addGlobalParameter("kphi0_hb", kphi0HB)
HBForce_PS.addGlobalParameter("kphi1_hb", kphi1HB)
HBForce_PS.addGlobalParameter("kphi2_hb", kphi2HB)
HBForce_PS.addGlobalParameter("pi", pi)
HBForce_PS.addGlobalParameter("pi_cut", pi_cut)


for j in range(0,len(PS_hbond[0])):
    p = int(PS_hbond[0][j])
    s = int(PS_hbond[1][j])
    HBForce_PS.addBond([p+3,p+1,p,s,s+2,s+3], [float(PS_hbond[2][j])*u.angstroms,
                                               float(PS_hbond[3][j])*u.radian,
                                               float(PS_hbond[4][j])*u.radian,
                                               float(PS_hbond[5][j])*u.radian,
                                               float(PS_hbond[6][j])*u.radian,
                                               float(PS_hbond[7][j])*u.radian,
                                               int(  PS_hbond[8][j])])    
## Sugar-sugar Hbond
HBForce_SS = mm.CustomCompoundBondForce(6, HBEnergyExpression)
HBForce_SS.addPerBondParameter("R0HB")
HBForce_SS.addPerBondParameter("theta10HB")
HBForce_SS.addPerBondParameter("theta20HB")
HBForce_SS.addPerBondParameter("phi00HB")
HBForce_SS.addPerBondParameter("phi10HB")
HBForce_SS.addPerBondParameter("phi20HB")
HBForce_SS.addPerBondParameter("nHB")
HBForce_SS.addGlobalParameter("u0HB", u0HB)
HBForce_SS.addGlobalParameter("kr_hb", krHB)
HBForce_SS.addGlobalParameter("ktheta1_hb", ktheta1)
HBForce_SS.addGlobalParameter("ktheta2_hb", ktheta2)
HBForce_SS.addGlobalParameter("kphi0_hb", kphi0HB)
HBForce_SS.addGlobalParameter("kphi1_hb", kphi1HB)
HBForce_SS.addGlobalParameter("kphi2_hb", kphi2HB)
HBForce_SS.addGlobalParameter("pi", pi)
HBForce_SS.addGlobalParameter("pi_cut", pi_cut)

for j in range(0,len(SS_hbond[0])):
    s1 = int(SS_hbond[0][j])
    s2 = int(SS_hbond[1][j])
    HBForce_SS.addBond([s1+3,s1+2,s1,s2,s2+2,s2+3],[float( SS_hbond[2][j])*u.angstroms,
                                              float( SS_hbond[3][j])*u.radian,
                                              float( SS_hbond[4][j])*u.radian,
                                              float( SS_hbond[5][j])*u.radian,
                                              float( SS_hbond[6][j])*u.radian,
                                              float( SS_hbond[7][j])*u.radian,
                                              int(   SS_hbond[8][j])])

# # --------------------------------- #
# #### Tert-stack INTERACTION POTENTIAL ####

u0TS      = u0TStack*u.kilocalorie_per_mole
krTS      = 5.0/(u.angstrom)**2
ktheta1TS = 1.5/(u.radian)**2
ktheta2TS = 1.5/(u.radian)**2
kphi0TS   = 0.15/(u.radian)**2
kphi1TS   = 0.15/(u.radian)**2
kphi2TS   = 0.15/(u.radian)**2
pi        = np.pi

TSEnergyExpression = "nTS*step(pi_cut-theta1TS)*step(pi_cut-theta2TS)*u0TS/(1+u2);"
TSEnergyExpression += "u2 = (kr_ts*delRsq + ktheta1_ts*deltheta1sq + ktheta2_ts*deltheta2sq + kphi0_ts*delphi0sq + kphi1_ts*delphi1sq + kphi2_ts*delphi2sq);"

TSEnergyExpression+= "delRsq = delR*delR; delR = (RTS-R0TS); RTS = distance(p3,p4);"
# ------------------------------------------ #
TSEnergyExpression+= "deltheta1sq = deltheta1*deltheta1;"
TSEnergyExpression+= "deltheta2sq = deltheta2*deltheta2;"
TSEnergyExpression+= "deltheta1   = (theta1TS-theta10TS); theta1TS= angle(p2,p3,p4);"
TSEnergyExpression+= "deltheta2   = (theta2TS-theta20TS); theta2TS= angle(p3,p4,p5);"
# ------------------------------------------ #

TSEnergyExpression+= "delphi0sq = delphi0*delphi0;"
TSEnergyExpression+= "delphi0 = dphi0 - 2.0*pi*step(dphi0-pi) + 2.0*pi*step(-dphi0-pi);"
# TSEnergyExpression+= "delphi0 = dphi0 - 2.0*pi*(dphi0/(abs(dphi0)))*step(abs(dphi0)-pi);"
TSEnergyExpression+= "dphi0 =(phi0TS-phi00TS); phi0TS= dihedral(p2,p3,p4,p5);"

TSEnergyExpression+= "delphi1sq = delphi1*delphi1;"
TSEnergyExpression+= "delphi1 = dphi1 - 2.0*pi*step(dphi1-pi) + 2.0*pi*step(-dphi1-pi);"
# TSEnergyExpression+= "delphi1 = dphi1 - 2.0*pi*(dphi1/(abs(dphi1)))*step(abs(dphi1)-pi);"
TSEnergyExpression+= "dphi1 =(phi1TS-phi10TS); phi1TS = dihedral(p1,p2,p3,p4);"


TSEnergyExpression+= "delphi2sq = delphi2*delphi2;"
TSEnergyExpression+= "delphi2 = dphi2 - 2.0*pi*step(dphi2-pi) + 2.0*pi*step(-dphi2-pi);"
# TSEnergyExpression+= "delphi2 = dphi2 - 2.0*pi*(dphi2/(abs(dphi2)))*step(abs(dphi2)-pi);"
TSEnergyExpression+= "dphi2 =(phi2TS-phi20TS); phi2TS= dihedral(p3,p4,p5,p6);"
# ------------------------------------------ #


## holo tert_stack
TSForce = mm.CustomCompoundBondForce(6, TSEnergyExpression)
TSForce.addPerBondParameter("R0TS")
TSForce.addPerBondParameter("theta10TS")
TSForce.addPerBondParameter("theta20TS")
TSForce.addPerBondParameter("phi00TS")
TSForce.addPerBondParameter("phi10TS")
TSForce.addPerBondParameter("phi20TS")
TSForce.addPerBondParameter("nTS")
TSForce.addGlobalParameter("u0TS", u0TS)
TSForce.addGlobalParameter("kr_ts", krTS)
TSForce.addGlobalParameter("ktheta1_ts", ktheta1TS)
TSForce.addGlobalParameter("ktheta2_ts", ktheta2TS)
TSForce.addGlobalParameter("kphi0_ts", kphi0TS)
TSForce.addGlobalParameter("kphi1_ts", kphi1TS)
TSForce.addGlobalParameter("kphi2_ts", kphi2TS)
TSForce.addGlobalParameter("pi", pi)
TSForce.addGlobalParameter("pi_cut", pi_cut)

for j in range(0,len(TStack[0])):
    a = int(         TStack[0][j])
    b = int(         TStack[1][j])
    TSForce.addBond([a+1,a-1,a,b,b-1,b+1], [float( TStack[2][j])*u.angstroms,
                                            float( TStack[3][j])*u.radian,
                                            float( TStack[4][j])*u.radian,
                                            float( TStack[5][j])*u.radian,
                                            float( TStack[6][j])*u.radian,
                                            float( TStack[7][j])*u.radian,
                                            int(   TStack[8][j])])
    
# -------------------------------------------#
# Nonbonded electrostatic: using PME
# -------------------------------------------#
tcent = T/u.kelvin - 273.15
dielectric_sqrt = np.sqrt(87.74-(0.4008*tcent)+(0.0009398*tcent*tcent)-(1.41*tcent*tcent*tcent/1000000))

print("dielectric_constant: ", dielectric_sqrt)
ESForce = mm.NonbondedForce()
# ESForce.setNonbondedMethod(mm.NonbondedForce.PME)
# ESForce.setEwaldErrorTolerance( 5e-3 )
# ESForce.setCutoffDistance(0.499*box_length)
ESForce.setNonbondedMethod(mm.NonbondedForce.PME)
ESForce.setEwaldErrorTolerance( 1e-4 )
ESForce.setCutoffDistance(25.0*u.angstrom) ## UPDATED CUTOFF FOR PME
# print("charge")
for i in charge:
    ESForce.addParticle((float(i/dielectric_sqrt))*u.elementary_charge, 0*u.angstrom, 0*u.kilocalorie_per_mole)
for i in range(nMg):
    ESForce.addParticle((float(qMg/dielectric_sqrt))*u.elementary_charge, 0*u.angstrom, 0*u.kilocalorie_per_mole)
for i in range(nCo):
    ESForce.addParticle((float(qCo/dielectric_sqrt))*u.elementary_charge, 0*u.angstrom, 0*u.kilocalorie_per_mole)
for i in range(nK):
    ESForce.addParticle((float(qK/dielectric_sqrt))*u.elementary_charge, 0*u.angstrom, 0*u.kilocalorie_per_mole)
for i in range(nCl):
    ESForce.addParticle((float(qCl/dielectric_sqrt))*u.elementary_charge, 0*u.angstrom, 0*u.kilocalorie_per_mole)


for i in range(len(BondList[0])):
        ESForce.addException(BondList[0][i],BondList[1][i], 0*u.elementary_charge*u.elementary_charge,0*u.angstrom, 0*u.kilocalorie_per_mole)
#***********************************************#
###### Electro Static: Custom, With Cutoff ######
#***********************************************#
# tcent = T/u.kelvin - 273.15
# dielectric = 87.74-(0.4008*tcent)+(0.0009398*tcent*tcent)-(1.41*tcent*tcent*tcent/1000000)
# lb_kt = (332.0637090/dielectric)*u.angstrom*u.kilocalorie_per_mole

# ESForce= mm.CustomNonbondedForce("q_sq*lb_kt*(1/r); q_sq = q1*q2;")
# ESForce.addPerParticleParameter('q')
# ESForce.addGlobalParameter('lb_kt', lb_kt)
# ESForce.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
# ESForce.setCutoffDistance(0.499*box_length)

# for i in range(0,nbeads):
#     # print(float(charge[i]))
#     ESForce.addParticle([float(charge[i])])
# for i in range(nMg):
#     ESForce.addParticle([float(qMg)])
# for i in range(nCo):
#     ESForce.addParticle([float(qCo)])
# for i in range(nK):
#     ESForce.addParticle([float(qK)])
# for i in range(nCl):
#     ESForce.addParticle([float(qCl)])

# for i in range(len(BondList[0])):
#         ESForce.addExclusion(BondList[0][i],BondList[1][i])
#*****************************************************************#
system.addForce(HarmonicBondForce  ) # 1
system.addForce(HarmonicAngleForce ) # 2
system.addForce(LJForce_rna_rna    ) # 3
system.addForce(LJForce_ion_ion    ) # 4
system.addForce(LJForce_rna_ion    ) # 5
system.addForce(StackForce         ) # 6
system.addForce(HBForce_BB         ) # 7
system.addForce(HBForce_BS         ) # 8
system.addForce(HBForce_PS         ) # 9
system.addForce(HBForce_SS         ) # 10
system.addForce(TSForce            ) # 11  
system.addForce(ESForce            ) # 12

if ComMotionRemover:
    motion_remover = mm.CMMotionRemover(CMMR_frequency)
    system.addForce(motion_remover)

for i in range(system.getNumForces()):
    force = system.getForce(i)
    force.setForceGroup(i)


#########################################################################################################################
##### Energy Reporter ###############
totalforcegroup = system.getNumForces()

class EnergyReporter(object):
        def __init__ (self, file, reportInterval):
                self._out = open(file, 'w')
                self._reportInterval = reportInterval

        def __del__ (self):
                self._out.close()

        def describeNextReport(self, simulation):
                step = self._reportInterval - simulation.currentStep % self._reportInterval
                return (step, False, False, False, True)

        def report(self, simulation, state):
                self._out.write(str(simulation.currentStep))
                for i in range(totalforcegroup):
                        state = simulation.context.getState(getEnergy=True, groups={i})
                        energy = state.getPotentialEnergy() / u.kilocalorie_per_mole
                        self._out.write("," + str(energy))
                self._out.write("\n")
##########################################################################################################################

integrator = mm.LangevinMiddleIntegrator(
    T,
    friction_coeff/u.picosecond,
    time_step*u.picoseconds
)

##################@dm

platform = mm.Platform.getPlatformByName(platform_type)

if platform_type == 'CPU':
    simulation = app.Simulation(pdb.topology, system, integrator, platform)

elif platform_type == 'CUDA':
    properties = {'CudaPrecision': 'double'}
    simulation = app.Simulation(pdb.topology, system, integrator, platform, properties)

else:
    simulation = app.Simulation(pdb.topology, system, integrator, platform)

# ---------------- INIT ----------------
if restart == False:
        simulation.context.setPositions(positions)
        simulation.context.setVelocitiesToTemperature(T)
        print("restart: False\n")
else:
        print("restart: True\n")
        simulation.loadCheckpoint('chkin.chk')


############ Running OpenMM Energy Minimization##########

if minimization == True:
    print('Performing minimization')
    print("=========================")
    simulation.minimizeEnergy(tolerance=1e-5*u.kilojoule_per_mole/u.angstroms)
    print('Minimized energy:', simulation.context.getState(getEnergy=True).getPotentialEnergy())
    positions = simulation.context.getState(getPositions=True).getPositions()
    app.PDBFile.writeFile(pdb.topology, positions, open('%s_minimized.pdb'%pdb_prefix , 'w'))


############## Running OpenMM Simulation ################

print("\n PERFORMING SIMULATION")
print("========================")
print('Simulating with T =',T)
print('Initiating MD simulation for %i steps'%numsteps)

simulation.reporters.append(app.DCDReporter('%s.dcd'%dcd_prefix, snap_interval))

simulation.reporters.append(app.StateDataReporter(
    file=data_name,
    reportInterval=data_interval,
    step=True,
    time=True, 
    potentialEnergy=True, 
    kineticEnergy=True, 
    temperature=True, 
    progress=True, 
    remainingTime=True,                                        
    speed=True, 
    totalSteps=numsteps, 
    separator=','
))

simulation.reporters.append(app.CheckpointReporter('chkout.chk', data_interval))

fe = [
      "1.HarmonicBondForce",
      "2.HarmonicAngleForce",
      "3.LJForce_rna_rna",
      "4.LJForce_ion_ion",
      "5.LJForce_rna_ion",
      "6.StackForce",
      "7.HBForce_BB",
      "8.HBForce_BS",
      "9.HBForce_PS",
      "10.HBForce_SS",
      "11.TSForce",
      "12.ESForce_PME"
]

if ComMotionRemover:
    fe.append("CMMotionRemover")

simulation.reporters.append(EnergyReporter('energy.dat', data_interval))

print("\nEnergy components of initial position\n")
for i in range(system.getNumForces()):
    print(fe[i], simulation.context.getState(getEnergy=True, groups={i}).getPotentialEnergy())

simulation.step(numsteps)

###########################################################
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

if final_state == True:

    print('Saving final step pdb')
    positions = simulation.context.getState(getPositions=True).getPositions()
    app.PDBFile.writeFile(pdb.topology, positions, open('%s_final.pdb'%pdb_prefix , 'w'))

    print("\nEnergy components of final position\n")

    for i in range(system.getNumForces()):
        print(fe[i], simulation.context.getState(getEnergy=True, groups={i}).getPotentialEnergy())


