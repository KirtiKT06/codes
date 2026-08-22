"""
Inspect the actual Bond/Angle force constants ParmEd wrote into
charmm36m.xml for the atom types used by GLU (carboxylate) and ARG
(guanidinium) residues — the two residue types flagged by run_md.py's
force diagnostic. A catastrophic force with completely normal geometry
(see find_clash.py output) means a bad PARAMETER, not a bad position —
this checks whether that parameter is visibly broken (k=0, negative,
NaN, or wildly outside normal CHARMM36 ranges) in the XML itself.

Usage:
    python3 inspect_ff_params.py charmm36m.xml
"""
import sys
import xml.etree.ElementTree as ET


# Normal CHARMM36 bond force constants are roughly 200-600 kcal/mol/A^2
# (~400,000-1,200,000 kJ/mol/nm^2 in OpenMM's kJ/mol/nm^2 units) for heavy
# atom bonds, and equilibrium lengths 0.09-0.16 nm. Flag anything wildly
# outside that.
K_MIN_SANE = 1000.0
K_MAX_SANE = 5_000_000.0
LEN_MIN_SANE = 0.05
LEN_MAX_SANE = 0.25


def check_residue(root, resname, watch_atoms):
    print(f"=== {resname} ===")
    for residue in root.iter("Residue"):
        if residue.get("name") != resname:
            continue

        atom_types = {}
        for atom in residue.findall("Atom"):
            atom_types[atom.get("name")] = atom.get("type")

        print(f"Atom -> type mapping for watched atoms:")
        for a in watch_atoms:
            print(f"  {a}: {atom_types.get(a, '!!! NOT FOUND IN TEMPLATE !!!')}")

        print("Bonds within this residue involving watched atoms:")
        for bond in residue.findall("Bond"):
            a1, a2 = bond.get("atomName1"), bond.get("atomName2")
            if a1 in watch_atoms or a2 in watch_atoms:
                print(f"  bond {a1}-{a2}")
        print()


def check_bond_parameters(root, atom_types_of_interest):
    print("=== HarmonicBondForce parameters for watched atom types ===")
    for force in root.iter("HarmonicBondForce"):
        for bond in force.findall("Bond"):
            t1, t2 = bond.get("type1") or bond.get("class1"), bond.get("type2") or bond.get("class2")
            if t1 in atom_types_of_interest or t2 in atom_types_of_interest:
                k = float(bond.get("k"))
                length = float(bond.get("length"))
                flags = []
                if not (K_MIN_SANE <= k <= K_MAX_SANE):
                    flags.append(f"K OUT OF SANE RANGE ({k})")
                if not (LEN_MIN_SANE <= length <= LEN_MAX_SANE):
                    flags.append(f"LENGTH OUT OF SANE RANGE ({length})")
                flag_str = "  <<<< " + "; ".join(flags) if flags else ""
                print(f"  {t1}-{t2}: k={k:.1f} length={length:.4f}{flag_str}")
    print()


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} charmm36m.xml")
        sys.exit(1)

    xml_path = sys.argv[1]
    tree = ET.parse(xml_path)
    root = tree.getroot()

    check_residue(root, "GLU", ["CG", "CD", "OE1", "OE2"])
    check_residue(root, "ARG", ["NE", "CZ", "NH1", "NH2"])

    # collect the actual atom types used by GLU/ARG's flagged atoms so we
    # can look up their bond parameters directly
    types_of_interest = set()
    for resname, watch_atoms in [("GLU", ["CG", "CD", "OE1", "OE2"]), ("ARG", ["NE", "CZ", "NH1", "NH2"])]:
        for residue in root.iter("Residue"):
            if residue.get("name") != resname:
                continue
            for atom in residue.findall("Atom"):
                if atom.get("name") in watch_atoms:
                    types_of_interest.add(atom.get("type"))

    print(f"Atom types being checked: {types_of_interest}\n")
    check_bond_parameters(root, types_of_interest)


if __name__ == "__main__":
    main()