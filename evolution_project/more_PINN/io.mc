##############################################################################
# MC-shell I/O capture file.
# Creation Date and Time:  Mon Aug 10 10:38:19 2026

##############################################################################
Hello world from PE 0
Vnm_tstart: starting timer 26 (APBS WALL CLOCK)..
NOsh_parseInput:  Starting file parsing...
NOsh: Parsing READ section
NOsh: Storing molecule 0 path 4tna_out.pqr
NOsh: Done parsing READ section
NOsh: Done parsing READ section (nmol=1, ndiel=0, nkappa=0, ncharge=0, npot=0)
NOsh: Parsing ELEC section
NOsh_parseMG: Parsing parameters for MG calculation
NOsh_parseMG:  Parsing dime...
PBEparm_parseToken:  trying dime...
MGparm_parseToken:  trying dime...
NOsh_parseMG:  Parsing cglen...
PBEparm_parseToken:  trying cglen...
MGparm_parseToken:  trying cglen...
NOsh_parseMG:  Parsing fglen...
PBEparm_parseToken:  trying fglen...
MGparm_parseToken:  trying fglen...
NOsh_parseMG:  Parsing cgcent...
PBEparm_parseToken:  trying cgcent...
MGparm_parseToken:  trying cgcent...
NOsh_parseMG:  Parsing fgcent...
PBEparm_parseToken:  trying fgcent...
MGparm_parseToken:  trying fgcent...
NOsh_parseMG:  Parsing mol...
PBEparm_parseToken:  trying mol...
NOsh_parseMG:  Parsing npbe...
PBEparm_parseToken:  trying npbe...
NOsh: parsed npbe
NOsh_parseMG:  Parsing bcfl...
PBEparm_parseToken:  trying bcfl...
NOsh_parseMG:  Parsing pdie...
PBEparm_parseToken:  trying pdie...
NOsh_parseMG:  Parsing sdie...
PBEparm_parseToken:  trying sdie...
NOsh_parseMG:  Parsing ion...
PBEparm_parseToken:  trying ion...
NOsh_parseMG:  Parsing ion...
PBEparm_parseToken:  trying ion...
NOsh_parseMG:  Parsing srfm...
PBEparm_parseToken:  trying srfm...
NOsh_parseMG:  Parsing chgm...
PBEparm_parseToken:  trying chgm...
MGparm_parseToken:  trying chgm...
NOsh_parseMG:  Parsing srad...
PBEparm_parseToken:  trying srad...
NOsh_parseMG:  Parsing swin...
PBEparm_parseToken:  trying swin...
NOsh_parseMG:  Parsing sdens...
PBEparm_parseToken:  trying sdens...
NOsh_parseMG:  Parsing temp...
PBEparm_parseToken:  trying temp...
NOsh_parseMG:  Parsing calcenergy...
PBEparm_parseToken:  trying calcenergy...
NOsh_parseMG:  Parsing calcforce...
PBEparm_parseToken:  trying calcforce...
NOsh_parseMG:  Parsing write...
PBEparm_parseToken:  trying write...
NOsh_parseMG:  Parsing end...
MGparm_check:  checking MGparm object of type 1.
NOsh:  nlev = 4, dime = (161, 161, 225)
NOsh: Done parsing ELEC section (nelec = 1)
NOsh: Done parsing file (got QUIT)
Valist_readPQR: Counted 1997 atoms
Valist_getStatistics:  Max atom coordinate:  (58.806, 30.04, 67.304)
Valist_getStatistics:  Min atom coordinate:  (5.357, -16.989, -10.11)
Valist_getStatistics:  Molecule center:  (32.0815, 6.5255, 28.597)
NOsh_setupCalcMGAUTO(./src/generic/nosh.c, 1868):  coarse grid center = 39.801 4.588 30.962
NOsh_setupCalcMGAUTO(./src/generic/nosh.c, 1873):  fine grid center = 39.801 4.588 30.962
NOsh_setupCalcMGAUTO (./src/generic/nosh.c, 1885):  Coarse grid spacing = 0.896562, 0.856437, 0.747366
NOsh_setupCalcMGAUTO (./src/generic/nosh.c, 1887):  Fine grid spacing = 0.459063, 0.418938, 0.434866
NOsh_setupCalcMGAUTO (./src/generic/nosh.c, 1889):  Displacement between fine and coarse grids = 0, 0, 0
NOsh:  2 levels of focusing with 0.512025, 0.489163, 0.581865 reductions
NOsh_setupMGAUTO:  Resetting boundary flags
NOsh_setupCalcMGAUTO (./src/generic/nosh.c, 1983):  starting mesh repositioning.
NOsh_setupCalcMGAUTO (./src/generic/nosh.c, 1985):  coarse mesh center = 39.801 4.588 30.962
NOsh_setupCalcMGAUTO (./src/generic/nosh.c, 1990):  coarse mesh upper corner = 111.526 73.103 114.667
NOsh_setupCalcMGAUTO (./src/generic/nosh.c, 1995):  coarse mesh lower corner = -31.924 -63.927 -52.743
NOsh_setupCalcMGAUTO (./src/generic/nosh.c, 2000):  initial fine mesh upper corner = 76.526 38.103 79.667
NOsh_setupCalcMGAUTO (./src/generic/nosh.c, 2005):  initial fine mesh lower corner = 3.076 -28.927 -17.743
NOsh_setupCalcMGAUTO (./src/generic/nosh.c, 2066):  final fine mesh upper corner = 76.526 38.103 79.667
NOsh_setupCalcMGAUTO (./src/generic/nosh.c, 2071):  final fine mesh lower corner = 3.076 -28.927 -17.743
NOsh_setupMGAUTO:  Resetting boundary flags
NOsh_setupCalc:  Mapping ELEC statement 0 (1) to calculation 1 (2)
Vnm_tstart: starting timer 27 (Setup timer)..
Setting up PBE object...
Vpbe_ctor2:  solute radius = 49.4535
Vpbe_ctor2:  solute dimensions = 56.4692 x 49.775 x 79.8842
Vpbe_ctor2:  solute charge = -61
Vpbe_ctor2:  bulk ionic strength = 0.145235
Vpbe_ctor2:  xkappa = 0.124096
Vpbe_ctor2:  Debye length = 8.0583
Vpbe_ctor2:  zkappa2 = 1.23198
Vpbe_ctor2:  zmagic = 7042.98
Vpbe_ctor2:  Constructing Vclist with 75 x 75 x 75 table
Vclist_ctor2:  Using 75 x 75 x 75 hash table
Vclist_ctor2:  automatic domain setup.
Vclist_ctor2:  Using 2.5 max radius
Vclist_setupGrid:  Grid lengths = (66.513, 60.093, 90.478)
Vclist_setupGrid:  Grid lower corner = (-1.175, -23.521, -16.642)
Vclist_assignAtoms:  Have 2366749 atom entries
Vacc_storeParms:  Surf. density = 10
Vacc_storeParms:  Max area = 265.904
Vacc_storeParms:  Using 2696-point reference sphere
Setting up PDE object...
Vpmp_ctor2:  Using meth = 1, mgsolv = 0
Setting PDE center to local center...
Vpmg_fillco:  filling in source term.
fillcoCharge:  Calling fillcoChargeSpline2...
Vpmg_fillco:  filling in source term.
Vpmg_fillco:  marking ion and solvent accessibility.
fillcoCoef:  Calling fillcoCoefMol...
Vacc_SASA: Time elapsed: 0.288272
Vpmg_fillco:  done filling coefficient arrays
Vpmg_fillco:  filling boundary arrays
Vpmg_fillco:  done filling boundary arrays
Vnm_tstop: stopping timer 27 (Setup timer).  CPU TIME = 5.443049e+00
Vnm_tstart: starting timer 28 (Solver timer)..
Vnm_tstart: starting timer 30 (Vnewdrv2: fine problem setup)..
Vbuildops: Fine: (161, 161, 225)
Vbuildops: Operator stencil (lev, numdia) = (1, 4)
Vnm_tstop: stopping timer 30 (Vnewdrv2: fine problem setup).  CPU TIME = 1.215120e-01
Vnm_tstart: starting timer 30 (Vnewdrv2: coarse problem setup)..
Vbuildops: Galer: (081, 081, 113)
Vbuildops: Galer: (041, 041, 057)
Vbuildops: Galer: (021, 021, 029)
Vnm_tstop: stopping timer 30 (Vnewdrv2: coarse problem setup).  CPU TIME = 7.837570e-01
Vnm_tstart: starting timer 30 (Vnewdrv2: solve)..
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 6.517559e+00
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vnewton: Damping enabled
Vnewton: Using errtol_s: 3739057.465881
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 8.681743e+00
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vprtstp: iteration = 1
Vprtstp: relative residual = 3.482895e+05
Vprtstp: contraction number = 3.482895e+05
Vnewton: Attempting damping, relres = 398.400289
Vnewton: Attempting damping, relres = 0.751886
Vnewton: Attempting damping, relres = 0.774530
Vnewton: Damping accepted, relres = 0.751886
Vprtstp: iteration = 1
Vprtstp: relative residual = 7.518857e-01
Vprtstp: contraction number = 7.518857e-01
Vnewton: Using errtol_s: 2811343.708343
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 1.601973e+01
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vprtstp: iteration = 1
Vprtstp: relative residual = 1.913928e+05
Vprtstp: contraction number = 1.913928e+05
Vnewton: Attempting damping, relres = 0.106574
Vnewton: Attempting damping, relres = 0.413202
Vnewton: Damping accepted, relres = 0.106574
Vnewton: Damping disabled
Vprtstp: iteration = 2
Vprtstp: relative residual = 1.065742e-01
Vprtstp: contraction number = 1.417425e-01
Vnewton: Using errtol_s: 398486.946292
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 2.406738e+01
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vprtstp: iteration = 1
Vprtstp: relative residual = 3.997084e+04
Vprtstp: contraction number = 3.997084e+04
Vprtstp: iteration = 3
Vprtstp: relative residual = 2.650368e-02
Vprtstp: contraction number = 2.486877e-01
Vnewton: Using errtol_s: 99098.786667
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 3.011470e+01
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vprtstp: iteration = 1
Vprtstp: relative residual = 1.170561e+04
Vprtstp: contraction number = 1.170561e+04
Vprtstp: iteration = 4
Vprtstp: relative residual = 6.934250e-03
Vprtstp: contraction number = 2.616335e-01
Vnewton: Using errtol_s: 25927.560830
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 3.516909e+01
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vprtstp: iteration = 1
Vprtstp: relative residual = 3.343063e+03
Vprtstp: contraction number = 3.343063e+03
Vprtstp: iteration = 5
Vprtstp: relative residual = 1.471505e-03
Vprtstp: contraction number = 2.122082e-01
Vnewton: Using errtol_s: 5502.041762
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 3.998694e+01
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vprtstp: iteration = 1
Vprtstp: relative residual = 8.333150e+02
Vprtstp: contraction number = 8.333150e+02
Vprtstp: iteration = 6
Vprtstp: relative residual = 2.413885e-04
Vprtstp: contraction number = 1.640419e-01
Vnewton: Using errtol_s: 902.565353
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 4.389544e+01
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vprtstp: iteration = 1
Vprtstp: relative residual = 1.743089e+02
Vprtstp: contraction number = 1.743089e+02
Vprtstp: iteration = 7
Vprtstp: relative residual = 4.253640e-05
Vprtstp: contraction number = 1.762155e-01
Vnewton: Using errtol_s: 159.046046
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 4.880643e+01
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vprtstp: iteration = 1
Vprtstp: relative residual = 3.531543e+01
Vprtstp: contraction number = 3.531543e+01
Vprtstp: iteration = 8
Vprtstp: relative residual = 8.508608e-06
Vprtstp: contraction number = 2.000312e-01
Vnewton: Using errtol_s: 31.814174
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 5.302602e+01
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vprtstp: iteration = 1
Vprtstp: relative residual = 7.360084e+00
Vprtstp: contraction number = 7.360084e+00
Vprtstp: iteration = 9
Vprtstp: relative residual = 1.771784e-06
Vprtstp: contraction number = 2.082343e-01
Vnewton: Using errtol_s: 6.624803
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 5.770696e+01
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vprtstp: iteration = 1
Vprtstp: relative residual = 1.612160e+00
Vprtstp: contraction number = 1.612160e+00
Vprtstp: iteration = 10
Vprtstp: relative residual = 3.880526e-07
Vprtstp: contraction number = 2.190180e-01
Vnm_tstop: stopping timer 30 (Vnewdrv2: solve).  CPU TIME = 5.448853e+01
Vnm_tstop: stopping timer 28 (Solver timer).  CPU TIME = 5.555159e+01
Vpmg_setPart:  lower corner = (-31.924, -63.927, -52.743)
Vpmg_setPart:  upper corner = (111.526, 73.103, 114.667)
Vpmg_setPart:  actual minima = (-31.924, -63.927, -52.743)
Vpmg_setPart:  actual maxima = (111.526, 73.103, 114.667)
Vpmg_setPart:  bflag[FRONT] = 0
Vpmg_setPart:  bflag[BACK] = 0
Vpmg_setPart:  bflag[LEFT] = 0
Vpmg_setPart:  bflag[RIGHT] = 0
Vpmg_setPart:  bflag[UP] = 0
Vpmg_setPart:  bflag[DOWN] = 0
Vnm_tstart: starting timer 29 (Energy timer)..
Vpmg_energy:  calculating full PBE energy
Vpmg_qmEnergy:  Calculating nonlinear energy
Vpmg_energy:  qmEnergy = 3.802162285796E+01 kT
Vpmg_qfEnergyVolume:  Calculating energy
Vpmg_energy:  qfEnergy = 6.655983135404E+04 kT
Vpmg_energy:  dielEnergy = 3.321028462487E+04 kT
Vnm_tstop: stopping timer 29 (Energy timer).  CPU TIME = 6.717600e-02
Vnm_tstart: starting timer 30 (Force timer)..
Vnm_tstop: stopping timer 30 (Force timer).  CPU TIME = 1.000000e-06
Vnm_tstart: starting timer 27 (Setup timer)..
Setting up PBE object...
Vpbe_ctor2:  solute radius = 49.4535
Vpbe_ctor2:  solute dimensions = 56.4692 x 49.775 x 79.8842
Vpbe_ctor2:  solute charge = -61
Vpbe_ctor2:  bulk ionic strength = 0.145235
Vpbe_ctor2:  xkappa = 0.124096
Vpbe_ctor2:  Debye length = 8.0583
Vpbe_ctor2:  zkappa2 = 1.23198
Vpbe_ctor2:  zmagic = 7042.98
Vpbe_ctor2:  Constructing Vclist with 75 x 75 x 75 table
Vclist_ctor2:  Using 75 x 75 x 75 hash table
Vclist_ctor2:  automatic domain setup.
Vclist_ctor2:  Using 2.5 max radius
Vclist_setupGrid:  Grid lengths = (66.513, 60.093, 90.478)
Vclist_setupGrid:  Grid lower corner = (-1.175, -23.521, -16.642)
Vclist_assignAtoms:  Have 2366749 atom entries
Vacc_storeParms:  Surf. density = 10
Vacc_storeParms:  Max area = 265.904
Vacc_storeParms:  Using 2696-point reference sphere
Setting up PDE object...
Vpmp_ctor2:  Using meth = 1, mgsolv = 0
Setting PDE center to local center...
Vpmg_ctor2:  Filling boundary with old solution!
VPMG::focusFillBound -- New mesh mins = 3.076, -28.927, -17.743
VPMG::focusFillBound -- New mesh maxs = 76.526, 38.103, 79.667
VPMG::focusFillBound -- Old mesh mins = -31.924, -63.927, -52.743
VPMG::focusFillBound -- Old mesh maxs = 111.526, 73.103, 114.667
VPMG::extEnergy:  energy flag = 1
Vpmg_setPart:  lower corner = (3.076, -28.927, -17.743)
Vpmg_setPart:  upper corner = (76.526, 38.103, 79.667)
Vpmg_setPart:  actual minima = (-31.924, -63.927, -52.743)
Vpmg_setPart:  actual maxima = (111.526, 73.103, 114.667)
Vpmg_setPart:  bflag[FRONT] = 0
Vpmg_setPart:  bflag[BACK] = 0
Vpmg_setPart:  bflag[LEFT] = 0
Vpmg_setPart:  bflag[RIGHT] = 0
Vpmg_setPart:  bflag[UP] = 0
Vpmg_setPart:  bflag[DOWN] = 0
VPMG::extEnergy:   Finding extEnergy dimensions...
VPMG::extEnergy    Disj part lower corner = (3.076, -28.927, -17.743)
VPMG::extEnergy    Disj part upper corner = (76.526, 38.103, 79.667)
VPMG::extEnergy    Old lower corner = (-31.924, -63.927, -52.743)
VPMG::extEnergy    Old upper corner = (111.526, 73.103, 114.667)
Vpmg_qmEnergy:  Calculating nonlinear energy
VPMG::extEnergy: extQmEnergy = 0.165879 kT
Vpmg_qfEnergyVolume:  Calculating energy
VPMG::extEnergy: extQfEnergy = 0 kT
VPMG::extEnergy: extDiEnergy = 0.28346 kT
Vpmg_fillco:  filling in source term.
fillcoCharge:  Calling fillcoChargeSpline2...
Vpmg_fillco:  filling in source term.
Vpmg_fillco:  marking ion and solvent accessibility.
fillcoCoef:  Calling fillcoCoefMol...
Vacc_SASA: Time elapsed: 0.255946
Vpmg_fillco:  done filling coefficient arrays
Vnm_tstop: stopping timer 27 (Setup timer).  CPU TIME = 1.539490e+00
Vnm_tstart: starting timer 28 (Solver timer)..
Vnm_tstart: starting timer 30 (Vnewdrv2: fine problem setup)..
Vbuildops: Fine: (161, 161, 225)
Vbuildops: Operator stencil (lev, numdia) = (1, 4)
Vnm_tstop: stopping timer 30 (Vnewdrv2: fine problem setup).  CPU TIME = 1.253700e-01
Vnm_tstart: starting timer 30 (Vnewdrv2: coarse problem setup)..
Vbuildops: Galer: (081, 081, 113)
Vbuildops: Galer: (041, 041, 057)
Vbuildops: Galer: (021, 021, 029)
Vnm_tstop: stopping timer 30 (Vnewdrv2: coarse problem setup).  CPU TIME = 8.970490e-01
Vnm_tstart: starting timer 30 (Vnewdrv2: solve)..
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 6.379797e+01
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vnewton: Damping enabled
Vnewton: Using errtol_s: 5061502.636294
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 6.580756e+01
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vprtstp: iteration = 1
Vprtstp: relative residual = 9.771408e+05
Vprtstp: contraction number = 9.771408e+05
Vnewton: Attempting damping, relres = 237.321809
Vnewton: Attempting damping, relres = 0.734053
Vnewton: Attempting damping, relres = 0.797506
Vnewton: Damping accepted, relres = 0.734053
Vprtstp: iteration = 1
Vprtstp: relative residual = 7.340532e-01
Vprtstp: contraction number = 7.340532e-01
Vnewton: Using errtol_s: 3715412.289836
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 7.333817e+01
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vprtstp: iteration = 1
Vprtstp: relative residual = 5.745831e+05
Vprtstp: contraction number = 5.745831e+05
Vnewton: Attempting damping, relres = 0.142919
Vnewton: Attempting damping, relres = 0.428964
Vnewton: Damping accepted, relres = 0.142919
Vnewton: Damping disabled
Vprtstp: iteration = 2
Vprtstp: relative residual = 1.429191e-01
Vprtstp: contraction number = 1.946985e-01
Vnewton: Using errtol_s: 723385.290210
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 7.917179e+01
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vprtstp: iteration = 1
Vprtstp: relative residual = 9.557786e+04
Vprtstp: contraction number = 9.557786e+04
Vprtstp: iteration = 3
Vprtstp: relative residual = 2.869097e-02
Vprtstp: contraction number = 2.007498e-01
Vnewton: Using errtol_s: 145219.431597
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 8.395139e+01
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vprtstp: iteration = 1
Vprtstp: relative residual = 1.896222e+04
Vprtstp: contraction number = 1.896222e+04
Vprtstp: iteration = 4
Vprtstp: relative residual = 6.217970e-03
Vprtstp: contraction number = 2.167222e-01
Vnewton: Using errtol_s: 31472.273604
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 9.062381e+01
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vprtstp: iteration = 1
Vprtstp: relative residual = 4.392044e+03
Vprtstp: contraction number = 4.392044e+03
Vprtstp: iteration = 5
Vprtstp: relative residual = 1.152065e-03
Vprtstp: contraction number = 1.852798e-01
Vnewton: Using errtol_s: 5831.177574
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 9.618687e+01
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vprtstp: iteration = 1
Vprtstp: relative residual = 9.758950e+02
Vprtstp: contraction number = 9.758950e+02
Vprtstp: iteration = 6
Vprtstp: relative residual = 1.825851e-04
Vprtstp: contraction number = 1.584852e-01
Vnewton: Using errtol_s: 924.155162
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 1.007464e+02
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vprtstp: iteration = 1
Vprtstp: relative residual = 1.862063e+02
Vprtstp: contraction number = 1.862063e+02
Vprtstp: iteration = 7
Vprtstp: relative residual = 3.312212e-05
Vprtstp: contraction number = 1.814064e-01
Vnewton: Using errtol_s: 167.647675
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 1.050054e+02
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vprtstp: iteration = 1
Vprtstp: relative residual = 3.603347e+01
Vprtstp: contraction number = 3.603347e+01
Vprtstp: iteration = 8
Vprtstp: relative residual = 6.407282e-06
Vprtstp: contraction number = 1.934442e-01
Vnewton: Using errtol_s: 32.430474
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 1.092478e+02
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vprtstp: iteration = 1
Vprtstp: relative residual = 7.004992e+00
Vprtstp: contraction number = 7.004992e+00
Vprtstp: iteration = 9
Vprtstp: relative residual = 1.245578e-06
Vprtstp: contraction number = 1.944003e-01
Vnewton: Using errtol_s: 6.304494
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 1.147725e+02
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vprtstp: iteration = 1
Vprtstp: relative residual = 1.421783e+00
Vprtstp: contraction number = 1.421783e+00
Vprtstp: iteration = 10
Vprtstp: relative residual = 2.528125e-07
Vprtstp: contraction number = 2.029681e-01
Vnm_tstop: stopping timer 30 (Vnewdrv2: solve).  CPU TIME = 5.612619e+01
Vnm_tstop: stopping timer 28 (Solver timer).  CPU TIME = 5.729735e+01
Vpmg_setPart:  lower corner = (3.076, -28.927, -17.743)
Vpmg_setPart:  upper corner = (76.526, 38.103, 79.667)
Vpmg_setPart:  actual minima = (3.076, -28.927, -17.743)
Vpmg_setPart:  actual maxima = (76.526, 38.103, 79.667)
Vpmg_setPart:  bflag[FRONT] = 0
Vpmg_setPart:  bflag[BACK] = 0
Vpmg_setPart:  bflag[LEFT] = 0
Vpmg_setPart:  bflag[RIGHT] = 0
Vpmg_setPart:  bflag[UP] = 0
Vpmg_setPart:  bflag[DOWN] = 0
Vnm_tstart: starting timer 29 (Energy timer)..
Vpmg_energy:  calculating full PBE energy
Vpmg_qmEnergy:  Calculating nonlinear energy
Vpmg_energy:  qmEnergy = 3.783069351551E+01 kT
Vpmg_qfEnergyVolume:  Calculating energy
Vpmg_energy:  qfEnergy = 2.302661310266E+05 kT
Vpmg_energy:  dielEnergy = 1.150641533274E+05 kT
Vnm_tstop: stopping timer 29 (Energy timer).  CPU TIME = 6.381000e-02
Vnm_tstart: starting timer 30 (Force timer)..
Vnm_tstop: stopping timer 30 (Force timer).  CPU TIME = 1.000000e-06
Vgrid_writeDX:  Opening virtual socket...
Vgrid_writeDX:  Writing to virtual socket...
Vgrid_writeDX:  Writing comments for ASC format.
Vnm_tstop: stopping timer 26 (APBS WALL CLOCK).  CPU TIME = 1.208174e+02
##############################################################################
# MC-shell I/O capture file.
# Creation Date and Time:  Mon Aug 10 10:38:30 2026

##############################################################################
Hello world from PE 0
Vnm_tstart: starting timer 26 (APBS WALL CLOCK)..
NOsh_parseInput:  Starting file parsing...
NOsh: Parsing READ section
NOsh: Storing molecule 0 path 4tna_out.pqr
NOsh: Done parsing READ section
NOsh: Done parsing READ section (nmol=1, ndiel=0, nkappa=0, ncharge=0, npot=0)
NOsh: Parsing ELEC section
NOsh_parseMG: Parsing parameters for MG calculation
NOsh_parseMG:  Parsing dime...
PBEparm_parseToken:  trying dime...
MGparm_parseToken:  trying dime...
NOsh_parseMG:  Parsing cglen...
PBEparm_parseToken:  trying cglen...
MGparm_parseToken:  trying cglen...
NOsh_parseMG:  Parsing fglen...
PBEparm_parseToken:  trying fglen...
MGparm_parseToken:  trying fglen...
NOsh_parseMG:  Parsing cgcent...
PBEparm_parseToken:  trying cgcent...
MGparm_parseToken:  trying cgcent...
NOsh_parseMG:  Parsing fgcent...
PBEparm_parseToken:  trying fgcent...
MGparm_parseToken:  trying fgcent...
NOsh_parseMG:  Parsing mol...
PBEparm_parseToken:  trying mol...
NOsh_parseMG:  Parsing npbe...
PBEparm_parseToken:  trying npbe...
NOsh: parsed npbe
NOsh_parseMG:  Parsing bcfl...
PBEparm_parseToken:  trying bcfl...
NOsh_parseMG:  Parsing pdie...
PBEparm_parseToken:  trying pdie...
NOsh_parseMG:  Parsing sdie...
PBEparm_parseToken:  trying sdie...
NOsh_parseMG:  Parsing srfm...
PBEparm_parseToken:  trying srfm...
NOsh_parseMG:  Parsing chgm...
PBEparm_parseToken:  trying chgm...
MGparm_parseToken:  trying chgm...
NOsh_parseMG:  Parsing srad...
PBEparm_parseToken:  trying srad...
NOsh_parseMG:  Parsing swin...
PBEparm_parseToken:  trying swin...
NOsh_parseMG:  Parsing sdens...
PBEparm_parseToken:  trying sdens...
NOsh_parseMG:  Parsing temp...
PBEparm_parseToken:  trying temp...
NOsh_parseMG:  Parsing calcenergy...
PBEparm_parseToken:  trying calcenergy...
NOsh_parseMG:  Parsing calcforce...
PBEparm_parseToken:  trying calcforce...
NOsh_parseMG:  Parsing write...
PBEparm_parseToken:  trying write...
NOsh_parseMG:  Parsing end...
MGparm_check:  checking MGparm object of type 1.
NOsh:  nlev = 4, dime = (161, 161, 225)
NOsh: Done parsing ELEC section (nelec = 1)
NOsh: Done parsing file (got QUIT)
Valist_readPQR: Counted 1997 atoms
Valist_getStatistics:  Max atom coordinate:  (58.806, 30.04, 67.304)
Valist_getStatistics:  Min atom coordinate:  (5.357, -16.989, -10.11)
Valist_getStatistics:  Molecule center:  (32.0815, 6.5255, 28.597)
NOsh_setupCalcMGAUTO(./src/generic/nosh.c, 1868):  coarse grid center = 39.801 4.588 30.962
NOsh_setupCalcMGAUTO(./src/generic/nosh.c, 1873):  fine grid center = 39.801 4.588 30.962
NOsh_setupCalcMGAUTO (./src/generic/nosh.c, 1885):  Coarse grid spacing = 0.896562, 0.856437, 0.747366
NOsh_setupCalcMGAUTO (./src/generic/nosh.c, 1887):  Fine grid spacing = 0.459063, 0.418938, 0.434866
NOsh_setupCalcMGAUTO (./src/generic/nosh.c, 1889):  Displacement between fine and coarse grids = 0, 0, 0
NOsh:  2 levels of focusing with 0.512025, 0.489163, 0.581865 reductions
NOsh_setupMGAUTO:  Resetting boundary flags
NOsh_setupCalcMGAUTO (./src/generic/nosh.c, 1983):  starting mesh repositioning.
NOsh_setupCalcMGAUTO (./src/generic/nosh.c, 1985):  coarse mesh center = 39.801 4.588 30.962
NOsh_setupCalcMGAUTO (./src/generic/nosh.c, 1990):  coarse mesh upper corner = 111.526 73.103 114.667
NOsh_setupCalcMGAUTO (./src/generic/nosh.c, 1995):  coarse mesh lower corner = -31.924 -63.927 -52.743
NOsh_setupCalcMGAUTO (./src/generic/nosh.c, 2000):  initial fine mesh upper corner = 76.526 38.103 79.667
NOsh_setupCalcMGAUTO (./src/generic/nosh.c, 2005):  initial fine mesh lower corner = 3.076 -28.927 -17.743
NOsh_setupCalcMGAUTO (./src/generic/nosh.c, 2066):  final fine mesh upper corner = 76.526 38.103 79.667
NOsh_setupCalcMGAUTO (./src/generic/nosh.c, 2071):  final fine mesh lower corner = 3.076 -28.927 -17.743
NOsh_setupMGAUTO:  Resetting boundary flags
NOsh_setupCalc:  Mapping ELEC statement 0 (1) to calculation 1 (2)
Vnm_tstart: starting timer 27 (Setup timer)..
Setting up PBE object...
Vpbe_ctor2:  solute radius = 49.4535
Vpbe_ctor2:  solute dimensions = 56.4692 x 49.775 x 79.8842
Vpbe_ctor2:  solute charge = -61
Vpbe_ctor2:  bulk ionic strength = 0
Vpbe_ctor2:  xkappa = 0
Vpbe_ctor2:  Debye length = 0
Vpbe_ctor2:  zkappa2 = 0
Vpbe_ctor2:  zmagic = 7042.98
Vpbe_ctor2:  Constructing Vclist with 75 x 75 x 75 table
Vclist_ctor2:  Using 75 x 75 x 75 hash table
Vclist_ctor2:  automatic domain setup.
Vclist_ctor2:  Using 1.9 max radius
Vclist_setupGrid:  Grid lengths = (64.809, 58.389, 88.774)
Vclist_setupGrid:  Grid lower corner = (-0.323, -22.669, -15.79)
Vclist_assignAtoms:  Have 1731660 atom entries
Vacc_storeParms:  Surf. density = 10
Vacc_storeParms:  Max area = 201.062
Vacc_storeParms:  Using 2040-point reference sphere
Setting up PDE object...
Vpmp_ctor2:  Using meth = 1, mgsolv = 0
Setting PDE center to local center...
Vpmg_fillco:  filling in source term.
fillcoCharge:  Calling fillcoChargeSpline2...
Vpmg_fillco:  filling in source term.
Vpmg_fillco:  filling boundary arrays
Vpmg_fillco:  done filling boundary arrays
Vnm_tstop: stopping timer 27 (Setup timer).  CPU TIME = 2.225896e+00
Vnm_tstart: starting timer 28 (Solver timer)..
Vnm_tstart: starting timer 30 (Vnewdrv2: fine problem setup)..
Vbuildops: Fine: (161, 161, 225)
Vbuildops: Operator stencil (lev, numdia) = (1, 4)
Vnm_tstop: stopping timer 30 (Vnewdrv2: fine problem setup).  CPU TIME = 1.257980e-01
Vnm_tstart: starting timer 30 (Vnewdrv2: coarse problem setup)..
Vbuildops: Galer: (081, 081, 113)
Vbuildops: Galer: (041, 041, 057)
Vbuildops: Galer: (021, 021, 029)
Vnm_tstop: stopping timer 30 (Vnewdrv2: coarse problem setup).  CPU TIME = 7.285420e-01
Vnm_tstart: starting timer 30 (Vnewdrv2: solve)..
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 3.288842e+00
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vnewton: Damping enabled
Vnewton: Using errtol_s: 55562354.657850
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 4.672993e+00
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vprtstp: iteration = 1
Vprtstp: relative residual = 6.982463e+06
Vprtstp: contraction number = 6.982463e+06
Vnewton: Attempting damping, relres = 0.113102
Vnewton: Attempting damping, relres = 0.556390
Vnewton: Damping accepted, relres = 0.113102
Vnewton: Damping disabled
Vprtstp: iteration = 1
Vprtstp: relative residual = 1.131021e-01
Vprtstp: contraction number = 1.131021e-01
Vnewton: Using errtol_s: 6284217.083145
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 1.061524e+01
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vprtstp: iteration = 1
Vprtstp: relative residual = 8.448379e+05
Vprtstp: contraction number = 8.448379e+05
Vprtstp: iteration = 2
Vprtstp: relative residual = 1.368470e-02
Vprtstp: contraction number = 1.209942e-01
Vnewton: Using errtol_s: 760354.119785
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 1.608226e+01
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vprtstp: iteration = 1
Vprtstp: relative residual = 1.053740e+05
Vprtstp: contraction number = 1.053740e+05
Vprtstp: iteration = 3
Vprtstp: relative residual = 1.706851e-03
Vprtstp: contraction number = 1.247269e-01
Vnewton: Using errtol_s: 94836.633425
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 2.071442e+01
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vprtstp: iteration = 1
Vprtstp: relative residual = 1.342890e+04
Vprtstp: contraction number = 1.342890e+04
Vprtstp: iteration = 4
Vprtstp: relative residual = 2.175215e-04
Vprtstp: contraction number = 1.274403e-01
Vnewton: Using errtol_s: 12086.006917
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 2.591762e+01
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vprtstp: iteration = 1
Vprtstp: relative residual = 1.740464e+03
Vprtstp: contraction number = 1.740464e+03
Vprtstp: iteration = 5
Vprtstp: relative residual = 2.819207e-05
Vprtstp: contraction number = 1.296059e-01
Vnewton: Using errtol_s: 1566.417936
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 2.940401e+01
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vprtstp: iteration = 1
Vprtstp: relative residual = 2.285787e+02
Vprtstp: contraction number = 2.285787e+02
Vprtstp: iteration = 6
Vprtstp: relative residual = 3.702522e-06
Vprtstp: contraction number = 1.313320e-01
Vnewton: Using errtol_s: 205.720842
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 3.852524e+01
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vprtstp: iteration = 1
Vprtstp: relative residual = 3.032925e+01
Vprtstp: contraction number = 3.032925e+01
Vprtstp: iteration = 7
Vprtstp: relative residual = 4.912738e-07
Vprtstp: contraction number = 1.326863e-01
Vnm_tstop: stopping timer 30 (Vnewdrv2: solve).  CPU TIME = 3.937798e+01
Vnm_tstop: stopping timer 28 (Solver timer).  CPU TIME = 4.043353e+01
Vpmg_setPart:  lower corner = (-31.924, -63.927, -52.743)
Vpmg_setPart:  upper corner = (111.526, 73.103, 114.667)
Vpmg_setPart:  actual minima = (-31.924, -63.927, -52.743)
Vpmg_setPart:  actual maxima = (111.526, 73.103, 114.667)
Vpmg_setPart:  bflag[FRONT] = 0
Vpmg_setPart:  bflag[BACK] = 0
Vpmg_setPart:  bflag[LEFT] = 0
Vpmg_setPart:  bflag[RIGHT] = 0
Vpmg_setPart:  bflag[UP] = 0
Vpmg_setPart:  bflag[DOWN] = 0
Vnm_tstart: starting timer 29 (Energy timer)..
Vpmg_energy:  calculating only q-phi energy
Vpmg_qfEnergyVolume:  Calculating energy
Vpmg_energy:  qfEnergy = 1.203365317473E+05 kT
Vnm_tstop: stopping timer 29 (Energy timer).  CPU TIME = 4.171000e-03
Vnm_tstart: starting timer 30 (Force timer)..
Vnm_tstop: stopping timer 30 (Force timer).  CPU TIME = 1.000000e-06
Vnm_tstart: starting timer 27 (Setup timer)..
Setting up PBE object...
Vpbe_ctor2:  solute radius = 49.4535
Vpbe_ctor2:  solute dimensions = 56.4692 x 49.775 x 79.8842
Vpbe_ctor2:  solute charge = -61
Vpbe_ctor2:  bulk ionic strength = 0
Vpbe_ctor2:  xkappa = 0
Vpbe_ctor2:  Debye length = 0
Vpbe_ctor2:  zkappa2 = 0
Vpbe_ctor2:  zmagic = 7042.98
Vpbe_ctor2:  Constructing Vclist with 75 x 75 x 75 table
Vclist_ctor2:  Using 75 x 75 x 75 hash table
Vclist_ctor2:  automatic domain setup.
Vclist_ctor2:  Using 1.9 max radius
Vclist_setupGrid:  Grid lengths = (64.809, 58.389, 88.774)
Vclist_setupGrid:  Grid lower corner = (-0.323, -22.669, -15.79)
Vclist_assignAtoms:  Have 1731660 atom entries
Vacc_storeParms:  Surf. density = 10
Vacc_storeParms:  Max area = 201.062
Vacc_storeParms:  Using 2040-point reference sphere
Setting up PDE object...
Vpmp_ctor2:  Using meth = 1, mgsolv = 0
Setting PDE center to local center...
Vpmg_ctor2:  Filling boundary with old solution!
VPMG::focusFillBound -- New mesh mins = 3.076, -28.927, -17.743
VPMG::focusFillBound -- New mesh maxs = 76.526, 38.103, 79.667
VPMG::focusFillBound -- Old mesh mins = -31.924, -63.927, -52.743
VPMG::focusFillBound -- Old mesh maxs = 111.526, 73.103, 114.667
VPMG::extEnergy:  energy flag = 1
Vpmg_setPart:  lower corner = (3.076, -28.927, -17.743)
Vpmg_setPart:  upper corner = (76.526, 38.103, 79.667)
Vpmg_setPart:  actual minima = (-31.924, -63.927, -52.743)
Vpmg_setPart:  actual maxima = (111.526, 73.103, 114.667)
Vpmg_setPart:  bflag[FRONT] = 0
Vpmg_setPart:  bflag[BACK] = 0
Vpmg_setPart:  bflag[LEFT] = 0
Vpmg_setPart:  bflag[RIGHT] = 0
Vpmg_setPart:  bflag[UP] = 0
Vpmg_setPart:  bflag[DOWN] = 0
VPMG::extEnergy:   Finding extEnergy dimensions...
VPMG::extEnergy    Disj part lower corner = (3.076, -28.927, -17.743)
VPMG::extEnergy    Disj part upper corner = (76.526, 38.103, 79.667)
VPMG::extEnergy    Old lower corner = (-31.924, -63.927, -52.743)
VPMG::extEnergy    Old upper corner = (111.526, 73.103, 114.667)
Vpmg_qmEnergy:  Zero energy for zero ionic strength!
VPMG::extEnergy: extQmEnergy = 0 kT
Vpmg_qfEnergyVolume:  Calculating energy
VPMG::extEnergy: extQfEnergy = 0 kT
VPMG::extEnergy: extDiEnergy = 5177.38 kT
Vpmg_fillco:  filling in source term.
fillcoCharge:  Calling fillcoChargeSpline2...
Vpmg_fillco:  filling in source term.
Vnm_tstop: stopping timer 27 (Setup timer).  CPU TIME = 2.826400e-01
Vnm_tstart: starting timer 28 (Solver timer)..
Vnm_tstart: starting timer 30 (Vnewdrv2: fine problem setup)..
Vbuildops: Fine: (161, 161, 225)
Vbuildops: Operator stencil (lev, numdia) = (1, 4)
Vnm_tstop: stopping timer 30 (Vnewdrv2: fine problem setup).  CPU TIME = 1.280160e-01
Vnm_tstart: starting timer 30 (Vnewdrv2: coarse problem setup)..
Vbuildops: Galer: (081, 081, 113)
Vbuildops: Galer: (041, 041, 057)
Vbuildops: Galer: (021, 021, 029)
Vnm_tstop: stopping timer 30 (Vnewdrv2: coarse problem setup).  CPU TIME = 1.142471e+00
Vnm_tstart: starting timer 30 (Vnewdrv2: solve)..
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 4.442461e+01
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vnewton: Damping enabled
Vnewton: Using errtol_s: 57023491.196358
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 4.509832e+01
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vprtstp: iteration = 1
Vprtstp: relative residual = 7.127063e+06
Vprtstp: contraction number = 7.127063e+06
Vnewton: Attempting damping, relres = 0.112486
Vnewton: Attempting damping, relres = 0.556183
Vnewton: Damping accepted, relres = 0.112486
Vnewton: Damping disabled
Vprtstp: iteration = 1
Vprtstp: relative residual = 1.124862e-01
Vprtstp: contraction number = 1.124862e-01
Vnewton: Using errtol_s: 6414357.068171
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 5.157291e+01
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vprtstp: iteration = 1
Vprtstp: relative residual = 7.701203e+05
Vprtstp: contraction number = 7.701203e+05
Vprtstp: iteration = 2
Vprtstp: relative residual = 1.215478e-02
Vprtstp: contraction number = 1.080558e-01
Vnewton: Using errtol_s: 693108.262070
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 5.853123e+01
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vprtstp: iteration = 1
Vprtstp: relative residual = 8.577808e+04
Vprtstp: contraction number = 8.577808e+04
Vprtstp: iteration = 3
Vprtstp: relative residual = 1.353833e-03
Vprtstp: contraction number = 1.113827e-01
Vnewton: Using errtol_s: 77200.271258
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 6.274285e+01
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vprtstp: iteration = 1
Vprtstp: relative residual = 9.815785e+03
Vprtstp: contraction number = 9.815785e+03
Vprtstp: iteration = 4
Vprtstp: relative residual = 1.549222e-04
Vprtstp: contraction number = 1.144323e-01
Vnewton: Using errtol_s: 8834.206803
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 6.728905e+01
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vprtstp: iteration = 1
Vprtstp: relative residual = 1.138813e+03
Vprtstp: contraction number = 1.138813e+03
Vprtstp: iteration = 5
Vprtstp: relative residual = 1.797385e-05
Vprtstp: contraction number = 1.160185e-01
Vnewton: Using errtol_s: 1024.931634
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 7.249342e+01
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vprtstp: iteration = 1
Vprtstp: relative residual = 1.337604e+02
Vprtstp: contraction number = 1.337604e+02
Vprtstp: iteration = 6
Vprtstp: relative residual = 2.111136e-06
Vprtstp: contraction number = 1.174560e-01
Vnewton: Using errtol_s: 120.384334
Vnm_tstop: stopping timer 40 (MG iteration).  CPU TIME = 7.766102e+01
Vprtstp: iteration = 0
Vprtstp: relative residual = 1.000000e+00
Vprtstp: contraction number = 1.000000e+00
Vprtstp: iteration = 1
Vprtstp: relative residual = 1.589159e+01
Vprtstp: contraction number = 1.589159e+01
Vprtstp: iteration = 7
Vprtstp: relative residual = 2.508165e-07
Vprtstp: contraction number = 1.188064e-01
Vnm_tstop: stopping timer 30 (Vnewdrv2: solve).  CPU TIME = 3.841275e+01
Vnm_tstop: stopping timer 28 (Solver timer).  CPU TIME = 3.986977e+01
Vpmg_setPart:  lower corner = (3.076, -28.927, -17.743)
Vpmg_setPart:  upper corner = (76.526, 38.103, 79.667)
Vpmg_setPart:  actual minima = (3.076, -28.927, -17.743)
Vpmg_setPart:  actual maxima = (76.526, 38.103, 79.667)
Vpmg_setPart:  bflag[FRONT] = 0
Vpmg_setPart:  bflag[BACK] = 0
Vpmg_setPart:  bflag[LEFT] = 0
Vpmg_setPart:  bflag[RIGHT] = 0
Vpmg_setPart:  bflag[UP] = 0
Vpmg_setPart:  bflag[DOWN] = 0
Vnm_tstart: starting timer 29 (Energy timer)..
Vpmg_energy:  calculating only q-phi energy
Vpmg_qfEnergyVolume:  Calculating energy
Vpmg_energy:  qfEnergy = 2.836721593312E+05 kT
Vnm_tstop: stopping timer 29 (Energy timer).  CPU TIME = 4.081000e-03
Vnm_tstart: starting timer 30 (Force timer)..
Vnm_tstop: stopping timer 30 (Force timer).  CPU TIME = 1.000000e-06
Vgrid_writeDX:  Opening virtual socket...
Vgrid_writeDX:  Writing to virtual socket...
Vgrid_writeDX:  Writing comments for ASC format.
Vnm_tstop: stopping timer 26 (APBS WALL CLOCK).  CPU TIME = 8.385559e+01
