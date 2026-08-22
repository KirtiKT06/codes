"""
Check whether the flagged atoms' directly-bonded neighbors (1-2 pairs) and
next-nearest neighbors (1-3 pairs) are properly EXCLUDED from the
NonbondedForce in the actual built System. Normal geometry + normal bond
parameters + a catastrophic force is the signature of a missing exclusion:
if a bonded pair isn't excluded, OpenMM evaluates a full LJ+Coulomb
interaction at the bond distance (~0.12-0.15 nm), which is astronomically
repulsive/attractive — easily large enough to explain a 34,000,000
kJ/mol/nm force with completely unremarkable coordinates.

Usage:
    python3 check_exclusions.py
"""
from openmm import app, XmlSerializer, NonbondedForce

from config import load_config, outpath


def main():
    cfg = load_config()
    system_xml = outpath(cfg, "system_xml")
    solvated_pdb = outpath(cfg, "solvated_pdb")

    with open(system_xml) as f:
        system = XmlSerializer.deserialize(f.read())

    pdb = app.PDBFile(solvated_pdb)
    atoms = list(pdb.topology.atoms())

    nonbonded = None
    for force in system.getForces():
        if isinstance(force, NonbondedForce):
            nonbonded = force
            break
    if nonbonded is None:
        raise RuntimeError("No NonbondedForce found in the System.")

    # Build the set of atom-index pairs that ARE marked as exceptions
    # (exclusions are just exceptions with chargeProd=0, epsilon=0 in
    # OpenMM's API — scaled 1-4 interactions are also "exceptions" but
    # with nonzero values, so we print the actual values rather than
    # just presence/absence).
    exceptions = {}
    for i in range(nonbonded.getNumExceptions()):
        p1, p2, chargeProd, sigma, epsilon = nonbonded.getExceptionParameters(i)
        exceptions[frozenset((p1, p2))] = (chargeProd, sigma, epsilon)

    # pairs to check: the flagged atom + its bonded neighbors, straight
    # from find_clash.py's output (GLU59 chain E CD/OE1/OE2/CG, ARG NE
    # neighbors)
    pairs_to_check = [
        (8956, 8957, "GLU59-E CD-OE1 (1-2, directly bonded)"),
        (8956, 8958, "GLU59-E CD-OE2 (1-2, directly bonded)"),
        (8956, 8953, "GLU59-E CD-CG  (1-2, directly bonded)"),
        (8958, 8953, "GLU59-E OE2-CG (1-3, two bonds apart)"),
        (8957, 8958, "GLU59-E OE1-OE2 (1-3, two bonds apart)"),
        (1405, 1406, "ARG83-A NE-HE (1-2, directly bonded)"),
        (1405, 1407, "ARG83-A NE-CZ (1-2, directly bonded)"),
    ]

    print(f"{'pair':45s} {'in exceptions?':15s} chargeProd  sigma    epsilon")
    for i1, i2, label in pairs_to_check:
        key = frozenset((i1, i2))
        if key in exceptions:
            chargeProd, sigma, epsilon = exceptions[key]
            print(f"{label:45s} {'YES':15s} {chargeProd} {sigma} {epsilon}")
        else:
            print(f"{label:45s} {'*** MISSING ***':15s} -- full nonbonded interaction applies at bond distance --")


if __name__ == "__main__":
    main()