#### This script should be run with 1 based index #####

set fp [open "./ps.txt" "r"]

# file_type = 1 (Base base non canonical, Tstack)
# file_type = 2 (Base sugar)
# file_type = 3 (Sugar sugar)
# file_type = 4 (Phosphate sugar)

##vvvvvvvvvvvv##
set file_type 4
##^^^^^^^^^^^^##

# Degree to Radian 1 degree = pi/180 rad
set factor [expr (3.14159265359 / 180.00)]
puts $factor

set file_data [read $fp]
puts $file_data
close $fp

set nhbond [expr [llength $file_data]]

set Rna_cg [atomselect top all]

# nresidue here is actually number of CG beads/atoms
set nresidue [$Rna_cg num]
puts $nresidue


# ============================================================
# SAFE MEASURE ROUTINES
# These avoid:
#   - index out of range
#   - one atom appears twice in list
# ============================================================

proc safe_bond {a b natoms} {
    if {$a < 0 || $b < 0 || $a >= $natoms || $b >= $natoms} {
        return 0.0
    }

    if {$a == $b} {
        return 0.0
    }

    return [measure bond [list $a $b]]
}


proc safe_angle {a b c natoms factor} {
    if {$a < 0 || $b < 0 || $c < 0 ||
        $a >= $natoms || $b >= $natoms || $c >= $natoms} {
        return 0.0
    }

    if {$a == $b || $a == $c || $b == $c} {
        return 0.0
    }

    set angle_deg [measure angle [list $a $b $c]]
    return [expr ($angle_deg * $factor)]
}


proc safe_dihed {a b c d natoms factor} {
    if {$a < 0 || $b < 0 || $c < 0 || $d < 0 ||
        $a >= $natoms || $b >= $natoms ||
        $c >= $natoms || $d >= $natoms} {
        return 0.0
    }

    if {$a == $b || $a == $c || $a == $d ||
        $b == $c || $b == $d ||
        $c == $d} {
        return 0.0
    }

    set dihed_deg [measure dihed [list $a $b $c $d]]
    return [expr ($dihed_deg * $factor)]
}


# ============================================================
# TYPE 1: BASE-BASE / TSTACK
# ============================================================

if {$file_type == 1} {

    set outfile_base_base [open 6ufh_Base_base_noncan.inp w]
    set outfile_bead_index_BL_bb [open 6ufh_bead_index_BL_bb_noncan.txt w]
    set outfile_bead_index_BA_bb [open 6ufh_bead_index_BA_bb_noncan.txt w]
    set outfile_bead_index_TA_bb [open 6ufh_bead_index_TA_bb_noncan.txt w]

    for { set i 0 } { $i < $nhbond } { incr i 5} {

        # Base-base non canonical hbond parameters
        set donor_name [lindex $file_data $i]
        set donor_index [lindex $file_data [expr $i+2]]
        set donor_residue [expr $donor_index - 1]

        set BDI [expr int(3 * $donor_residue + 2)]
        puts $BDI

        set acceptor_name [lindex $file_data [expr $i+1]]
        set acceptor_index [lindex $file_data [expr $i+3]]
        set acceptor_residue [expr $acceptor_index - 1]

        set BAI [expr int(3 * $acceptor_residue + 2)]
        puts $BAI

        # Extract number of hbonds from 5th column
        set num_hbonds [lindex $file_data [expr $i+4]]

        ## Bond distance
        set bond_distance [safe_bond $BAI $BDI $nresidue]

        ## Bond angles
        set bond_angle_rad_10 [safe_angle $BAI $BDI [expr $BDI - 1] $nresidue $factor]
        set bond_angle_rad_20 [safe_angle $BDI $BAI [expr $BAI - 1] $nresidue $factor]

        ## Dihedral angles with terminal fallback
        if { [expr $BDI + 1] < $nresidue } {
            set P_D_next [expr $BDI + 1]
        } else {
            set P_D_next [expr $BDI - 2]
        }

        if { [expr $BAI + 1] < $nresidue } {
            set P_A_next [expr $BAI + 1]
        } else {
            set P_A_next [expr $BAI - 2]
        }

        set torsion_angle_rad_00 [safe_dihed [expr $BDI - 1] $BDI $BAI [expr $BAI - 1] $nresidue $factor]
        set torsion_angle_rad_10 [safe_dihed $P_D_next [expr $BDI - 1] $BDI $BAI $nresidue $factor]
        set torsion_angle_rad_20 [safe_dihed $BDI $BAI [expr $BAI - 1] $P_A_next $nresidue $factor]

        puts $outfile_bead_index_BL_bb "r0\t$BAI\t$BDI"
        puts $outfile_bead_index_BA_bb "theta_10\t$BAI\t$BDI\t[expr $BDI - 1]"
        puts $outfile_bead_index_BA_bb "theta_20\t$BDI\t$BAI\t[expr $BAI - 1]"
        puts $outfile_bead_index_TA_bb "psi_00\t[expr $BDI - 1]\t$BDI\t$BAI\t[expr $BAI - 1]"
        puts $outfile_bead_index_TA_bb "psi_10\t$P_D_next\t[expr $BDI - 1]\t$BDI\t$BAI"
        puts $outfile_bead_index_TA_bb "psi_20\t$BDI\t$BAI\t[expr $BAI - 1]\t$P_A_next"

        puts $bond_distance

        puts $outfile_base_base "$BDI,$BAI,$bond_distance,$bond_angle_rad_10,$bond_angle_rad_20,$torsion_angle_rad_00,$torsion_angle_rad_10,$torsion_angle_rad_20,$num_hbonds"
    }

    close $outfile_base_base
    close $outfile_bead_index_BL_bb
    close $outfile_bead_index_BA_bb
    close $outfile_bead_index_TA_bb


# ============================================================
# TYPE 2: BASE-SUGAR
# ============================================================

} elseif {$file_type == 2} {

    set outfile_base_sugr [open 6ufh_Base_sugr.inp w]
    set outfile_bead_index_BL_bs [open 6ufh_bead_index_BL_bs.txt w]
    set outfile_bead_index_BA_bs [open 6ufh_bead_index_BA_bs.txt w]
    set outfile_bead_index_TA_bs [open 6ufh_bead_index_TA_bs.txt w]

    for { set i 0 } { $i < $nhbond } { incr i 5} {

        # Base-sugar hbond parameters
        set base_name [lindex $file_data $i]
        set base_index [lindex $file_data [expr $i+2]]
        set base_residue [expr $base_index - 1]

        set BI [expr int(3 * $base_residue + 2)]
        puts $BI

        set sugr_name [lindex $file_data [expr $i+1]]
        set sugr_index [lindex $file_data [expr $i+3]]
        set sugr_residue [expr $sugr_index - 1]

        set SI [expr int(3 * $sugr_residue + 1)]
        puts $SI

        # Extract number of hbonds from 5th column
        set num_hbonds [lindex $file_data [expr $i+4]]

        ## Bond distance
        set bond_distance [safe_bond $BI $SI $nresidue]

        ## Dynamic indexing for terminal residues
        if { [expr $BI + 1] < $nresidue } {
            set P_B_next [expr $BI + 1]
        } else {
            set P_B_next [expr $BI - 2]
        }

        if { [expr $SI + 3] < $nresidue } {
            set P_S_next [expr $SI + 2]
            set S_S_next [expr $SI + 3]
        } else {
            # fallback perspective
            set P_S_next [expr $SI - 1]
            set S_S_next [expr $SI - 3]

            if { $S_S_next < 0 } {
                set S_S_next [expr $SI + 1]
            }
        }

        ## Bond angles
        set bond_angle_rad_10 [safe_angle [expr $BI - 1] $BI $SI $nresidue $factor]
        set bond_angle_rad_20 [safe_angle $P_S_next $SI $BI $nresidue $factor]

        ## Dihedral angles
        set torsion_angle_rad_00 [safe_dihed $P_S_next $SI $BI [expr $BI - 1] $nresidue $factor]
        set torsion_angle_rad_10 [safe_dihed $SI $BI [expr $BI - 1] $P_B_next $nresidue $factor]
        set torsion_angle_rad_20 [safe_dihed $BI $SI $P_S_next $S_S_next $nresidue $factor]

        puts $outfile_bead_index_BL_bs "r0\t$BI\t$SI"
        puts $outfile_bead_index_BA_bs "theta_10\t[expr $BI - 1]\t$BI\t$SI"
        puts $outfile_bead_index_BA_bs "theta_20\t$P_S_next\t$SI\t$BI"
        puts $outfile_bead_index_TA_bs "psi_00\t$P_S_next\t$SI\t$BI\t[expr $BI - 1]"
        puts $outfile_bead_index_TA_bs "psi_10\t$SI\t$BI\t[expr $BI - 1]\t$P_B_next"
        puts $outfile_bead_index_TA_bs "psi_20\t$BI\t$SI\t$P_S_next\t$S_S_next"

        puts $bond_distance

        puts $outfile_base_sugr "$BI,$SI,$bond_distance,$bond_angle_rad_10,$bond_angle_rad_20,$torsion_angle_rad_00,$torsion_angle_rad_10,$torsion_angle_rad_20,$num_hbonds"
    }

    close $outfile_base_sugr
    close $outfile_bead_index_BL_bs
    close $outfile_bead_index_BA_bs
    close $outfile_bead_index_TA_bs


# ============================================================
# TYPE 3: SUGAR-SUGAR
# ============================================================

} elseif {$file_type == 3} {

    set outfile_sugr_sugr [open 6ufh_Sugr_sugr.inp w]
    set outfile_bead_index_BL_ss [open 6ufh_bead_index_BL_ss.txt w]
    set outfile_bead_index_BA_ss [open 6ufh_bead_index_BA_ss.txt w]
    set outfile_bead_index_TA_ss [open 6ufh_bead_index_TA_ss.txt w]

    for { set i 0 } { $i < $nhbond } { incr i 5} {

        set donor_name [lindex $file_data $i]
        set donor_index [lindex $file_data [expr $i+2]]
        set donor_residue [expr $donor_index - 1]

        set BDI [expr int(3 * $donor_residue + 1)]
        puts $BDI

        set acceptor_name [lindex $file_data [expr $i+1]]
        set acceptor_index [lindex $file_data [expr $i+3]]
        set acceptor_residue [expr $acceptor_index - 1]

        set BAI [expr int(3 * $acceptor_residue + 1)]
        puts $BAI

        # Extract number of hbonds from 5th column
        set num_hbonds [lindex $file_data [expr $i+4]]

        ## Bond distance
        set bond_distance [safe_bond $BAI $BDI $nresidue]

        ## Terminal fallback indices
        if { [expr $BDI + 2] < $nresidue } {
            set P_D_next [expr $BDI + 2]
        } else {
            set P_D_next [expr $BDI - 1]
        }

        if { [expr $BAI + 2] < $nresidue } {
            set P_A_next [expr $BAI + 2]
        } else {
            set P_A_next [expr $BAI - 1]
        }

        if { [expr $BDI + 3] < $nresidue } {
            set S_D_next [expr $BDI + 3]
        } else {
            set S_D_next [expr $BDI - 3]
        }

        if { [expr $BAI + 3] < $nresidue } {
            set S_A_next [expr $BAI + 3]
        } else {
            set S_A_next [expr $BAI - 3]
        }

        ## Bond angles
        set bond_angle_rad_10 [safe_angle $P_D_next $BDI $BAI $nresidue $factor]
        set bond_angle_rad_20 [safe_angle $P_A_next $BAI $BDI $nresidue $factor]

        ## Dihedral angles
        set torsion_angle_rad_00 [safe_dihed $P_D_next $BDI $BAI $P_A_next $nresidue $factor]
        set torsion_angle_rad_10 [safe_dihed $BAI $BDI $P_D_next $S_D_next $nresidue $factor]
        set torsion_angle_rad_20 [safe_dihed $BDI $BAI $P_A_next $S_A_next $nresidue $factor]

        puts $outfile_bead_index_BL_ss "r0\t$BAI\t$BDI"
        puts $outfile_bead_index_BA_ss "theta_10\t$P_D_next\t$BDI\t$BAI"
        puts $outfile_bead_index_BA_ss "theta_20\t$P_A_next\t$BAI\t$BDI"
        puts $outfile_bead_index_TA_ss "psi_00\t$P_D_next\t$BDI\t$BAI\t$P_A_next"
        puts $outfile_bead_index_TA_ss "psi_10\t$BAI\t$BDI\t$P_D_next\t$S_D_next"
        puts $outfile_bead_index_TA_ss "psi_20\t$BDI\t$BAI\t$P_A_next\t$S_A_next"

        puts $bond_distance

        puts $outfile_sugr_sugr "$BDI,$BAI,$bond_distance,$bond_angle_rad_10,$bond_angle_rad_20,$torsion_angle_rad_00,$torsion_angle_rad_10,$torsion_angle_rad_20,$num_hbonds"
    }

    close $outfile_sugr_sugr
    close $outfile_bead_index_BL_ss
    close $outfile_bead_index_BA_ss
    close $outfile_bead_index_TA_ss


# ============================================================
# TYPE 4: PHOSPHATE-SUGAR
# ============================================================

} elseif {$file_type == 4} {

    set outfile_phos_sugr [open 6ufh_Phos_sugr.inp w]
    set outfile_bead_index_BL_ps [open 6ufh_bead_index_BL_ps.txt w]
    set outfile_bead_index_BA_ps [open 6ufh_bead_index_BA_ps.txt w]
    set outfile_bead_index_TA_ps [open 6ufh_bead_index_TA_ps.txt w]

    for { set i 0 } { $i < $nhbond } { incr i 5} {

        set phos_name [lindex $file_data $i]
        set phos_index [lindex $file_data [expr $i+2]]
        set phos_residue [expr $phos_index - 1]

        set PI [expr int(3 * $phos_residue)]
        puts $PI

        set sugr_name [lindex $file_data [expr $i+1]]
        set sugr_index [lindex $file_data [expr $i+3]]
        set sugr_residue [expr $sugr_index - 1]

        set SI [expr int(3 * $sugr_residue + 1)]
        puts $SI

        # Extract number of hbonds from 5th column
        set num_hbonds [lindex $file_data [expr $i+4]]

        ## Bond distance
        set bond_distance [safe_bond $PI $SI $nresidue]

        ## Terminal fallback indices
        if { [expr $PI + 1] < $nresidue } {
            set S_P_next [expr $PI + 1]
        } else {
            set S_P_next [expr $PI - 2]
        }

        if { [expr $PI + 3] < $nresidue } {
            set P_P_next [expr $PI + 3]
        } else {
            set P_P_next [expr $PI - 3]
        }

        if { [expr $SI + 2] < $nresidue } {
            set P_S_next [expr $SI + 2]
        } else {
            set P_S_next [expr $SI - 1]
        }

        if { [expr $SI + 3] < $nresidue } {
            set S_S_next [expr $SI + 3]
        } else {
            set S_S_next [expr $SI - 3]
        }

        ## Bond angles
        set bond_angle_rad_10 [safe_angle $S_P_next $PI $SI $nresidue $factor]
        set bond_angle_rad_20 [safe_angle $P_S_next $SI $PI $nresidue $factor]

        ## Dihedral angles
        set torsion_angle_rad_00 [safe_dihed $P_S_next $SI $PI $S_P_next $nresidue $factor]
        set torsion_angle_rad_10 [safe_dihed $SI $PI $S_P_next $P_P_next $nresidue $factor]
        set torsion_angle_rad_20 [safe_dihed $PI $SI $P_S_next $S_S_next $nresidue $factor]

        puts $outfile_bead_index_BL_ps "r0\t$PI\t$SI"
        puts $outfile_bead_index_BA_ps "theta_10\t$S_P_next\t$PI\t$SI"
        puts $outfile_bead_index_BA_ps "theta_20\t$P_S_next\t$SI\t$PI"
        puts $outfile_bead_index_TA_ps "psi_00\t$P_S_next\t$SI\t$PI\t$S_P_next"
        puts $outfile_bead_index_TA_ps "psi_10\t$SI\t$PI\t$S_P_next\t$P_P_next"
        puts $outfile_bead_index_TA_ps "psi_20\t$PI\t$SI\t$P_S_next\t$S_S_next"

        puts $bond_distance

        puts $outfile_phos_sugr "$PI,$SI,$bond_distance,$bond_angle_rad_10,$bond_angle_rad_20,$torsion_angle_rad_00,$torsion_angle_rad_10,$torsion_angle_rad_20,$num_hbonds"
    }

    close $outfile_phos_sugr
    close $outfile_bead_index_BL_ps
    close $outfile_bead_index_BA_ps
    close $outfile_bead_index_TA_ps
}