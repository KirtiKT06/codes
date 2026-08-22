"""
Check HarmonicAngleForce entries for GLU (carboxylate) and ARG
(guanidinium) atom types in charmm36m.xml for a degrees-vs-radians unit
bug. OpenMM XML angle values must be in radians (max possible value is
pi ~= 3.14159); anything larger is almost certainly a raw degrees value
that never got converted (a real angle of "120 degrees" mistakenly
written as literal 120.0 instead of 2.094 radians is a >36x larger
target angle than intended, and produces a correspondingly enormous
restoring force at any real geometry).

Usage:
    python check_angle_units.py charmm36m.xml
"""
import math
import sys
import xml.etree.ElementTree as ET


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} charmm36m.xml")
        sys.exit(1)

    xml_path = sys.argv[1]
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # get the atom types used by GLU's carboxylate and ARG's guanidinium
    watch = {
        "GLU": ["CG", "CD", "OE1", "OE2"],
        "ARG": ["NE", "CZ", "NH1", "NH2"],
    }
    types_of_interest = set()
    for resname, atom_names in watch.items():
        for residue in root.iter("Residue"):
            if residue.get("name") != resname:
                continue
            for atom in residue.findall("Atom"):
                if atom.get("name") in atom_names:
                    types_of_interest.add(atom.get("type"))

    print(f"Watched atom types: {types_of_interest}\n")

    print("=== HarmonicAngleForce entries involving watched types ===")
    any_bad = False
    for force in root.iter("HarmonicAngleForce"):
        for angle_el in force.findall("Angle"):
            t1 = angle_el.get("type1") or angle_el.get("class1")
            t2 = angle_el.get("type2") or angle_el.get("class2")
            t3 = angle_el.get("type3") or angle_el.get("class3")
            if t1 in types_of_interest or t2 in types_of_interest or t3 in types_of_interest:
                angle_val = float(angle_el.get("angle"))
                k_val = float(angle_el.get("k"))
                flag = ""
                if angle_val > math.pi:
                    flag = f"  <<<< IMPOSSIBLE AS RADIANS (> pi = {math.pi:.4f}) — looks like raw degrees, e.g. {math.degrees(0)} check: if this is meant to be {angle_val} degrees, correct radian value = {math.radians(angle_val):.4f}"
                    any_bad = True
                print(f"  {t1}-{t2}-{t3}: angle={angle_val} k={k_val}{flag}")

    print()
    if any_bad:
        print("FOUND unit bug: at least one angle value exceeds pi radians — this is almost certainly the cause.")
    else:
        print("No angle values exceed pi radians — this specific hypothesis does not hold for angle terms.")

    # also check impropers, same unit convention applies and these were
    # explicitly flagged as "compressed"/merged during your conversion
    print("\n=== CustomTorsionForce / improper entries involving watched types (if present) ===")
    for force in root.iter("CustomTorsionForce"):
        for improper_el in force.findall("Improper"):
            t1 = improper_el.get("type1") or improper_el.get("class1")
            t2 = improper_el.get("type2") or improper_el.get("class2")
            t3 = improper_el.get("type3") or improper_el.get("class3")
            t4 = improper_el.get("type4") or improper_el.get("class4")
            if any(t in types_of_interest for t in (t1, t2, t3, t4)):
                print(f"  {t1}-{t2}-{t3}-{t4}: {improper_el.attrib}")


if __name__ == "__main__":
    main()