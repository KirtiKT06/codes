#puts "hello world!"
##_Load a new molecule_##
#mol new chainX2.pdb

set sel [atomselect top all]
set sel1 [$sel get residue]
set nresidue [lindex $sel1 [expr [llength $sel1]-1] ]
puts $nresidue
set outfile [open 1i9x_cg.xyz w]
set outfile1 [open 1i9x_cg.dat w]
set startIndex 0 

set beads [expr 3* ($nresidue + 1)]
puts $outfile "$beads\n"
puts $beads
for { set i $startIndex } { $i <= $nresidue } { incr i } {

        set list_P { P OP1 OP2 OP1H OP2H }
        # set list_S { O5' C5' C4' O4' C3' O3' C2' O2' C1' }
        set list_S { O5' C5' H5'1 H5'2 C4' H4' O4' C1' H1'  C3' H3' C2' H2'1 O2' HO'2 O3' }
        # set list_B { N9 C8 N7 C5 C6 O6 N1 C2 N2 N3 C4 N6 O2 O4 }
        set list_B { N9 C8 N7 C5 C6 O6 N1 C2 N2 N3 C4 N6 O2 O4 H1 H2 H21 H22 H3 H41 H42 H5 H6 H61 H62 H8 }
        
        #set list_G { N9 C8 N7 C5 C6 O6 N1 C2 N2 N3 C4 }
        #set list_G { N9 C8 N7 C5 C6 O6 N1 C2 N2 N3 C4, H8 H1 H21 H22 }
        #set list_A { N9 C8 N7 C5 C6 N6 N1 C2 N3 C4 }
        #set list_A { N9 C8 N7 C5 C6 N6 N1 C2 N3 C4 H8 H61 H62 H2 }
        #set list_U { N1 C2 O2 N3 C4 O4 C5 C6 }
        #set list_U { N1 C2 O2 N3 C4 O4 C5 C6 H6 H5 H3 }
        #set list_C { N1 C2 O2 N3 C4 N4 C5 C6 }
        #set list_C { N1 C2 O2 N3 C4 N4 C5 C6 H6 H5 H41 H42}
        set phos [atomselect top "(residue $i) and name $list_P"]
        set sugr [atomselect top "(residue $i) and name $list_S"]
        set base [atomselect top "(residue $i) and name $list_B"]

        flush $outfile
        set print_P0 [measure center $phos weight mass]
        set print_S0 [measure center $sugr weight mass]
        set print_B0 [measure center $base weight mass]
        set print_P [format "P,%.5f,%0.5f,%0.5f,%d" [lindex $print_P0 0] [lindex $print_P0 1] [lindex $print_P0 2] -1]
        set print_S [format "S,%.5f,%0.5f,%0.5f,%d" [lindex $print_S0 0] [lindex $print_S0 1] [lindex $print_S0 2] 0]
        set print_B [format "B,%.5f,%0.5f,%0.5f,%d" [lindex $print_B0 0] [lindex $print_B0 1] [lindex $print_B0 2] 0]

        puts $outfile "P\t[lindex $print_P0 0]\t[lindex $print_P0 1]\t[lindex $print_P0 2]"       
        puts $outfile "S\t[lindex $print_S0 0]\t[lindex $print_S0 1]\t[lindex $print_S0 2]"                
        puts $outfile "B\t[lindex $print_B0 0]\t[lindex $print_B0 1]\t[lindex $print_B0 2]"

        puts $outfile1 $print_P  
        puts $outfile1 $print_S 
        puts $outfile1 $print_B 
                
        $phos delete
        $sugr delete
        $base delete
                
}
close $outfile
close $outfile1
