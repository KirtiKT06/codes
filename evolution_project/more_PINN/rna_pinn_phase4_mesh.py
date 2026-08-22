"""
Phase 4, part A: load a real molecule from a pdb2pqr-generated PQR file
(atom coordinates, AMBER charges, radii) and build an implicit union-of-spheres
surface -- same architectural idea as Phase 3's DumbbellGeometry, but now for
~1600 atoms instead of 2, which requires a KD-tree instead of naive
all-pairs distance evaluation (a 700k-grid-point x 1656-atom dense distance
matrix would be ~9 GB in float64 -- not viable).

NOTE ON FIDELITY: this uses the van der Waals surface (union of atomic radii)
as an approximation of the true solvent-excluded surface (SES) that Achondo
et al. use (built via NanoShaper, which we don't have here). The vdW surface
has pockets/crevices the true SES (rolling a solvent probe sphere over the
vdW surface) would smooth over. This is a known, flagged simplification --
not silently swept under the rug.
"""

import numpy as np
from scipy.spatial import cKDTree
from skimage import measure
import trimesh
import tetgen


def parse_pqr(path):
    """Return (coords (n,3), charges (n,), radii (n,))."""
    coords, charges, radii = [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                parts = line.split()
                # PQR columns: recordName, atomNum, atomName, resName, resNum, x, y, z, charge, radius
                # (some PQR writers include chain ID, shifting columns -- handle both)
                try:
                    x, y, z, q, r = map(float, parts[-5:])
                except ValueError:
                    continue
                coords.append((x, y, z))
                charges.append(q)
                radii.append(r)
    return np.array(coords), np.array(charges), np.array(radii)


class MoleculeGeometry:
    def __init__(self, coords, radii, grid_spacing=1.0, k_neighbors=12, pad=3.0,
                 min_radius=1.0):
        """
        min_radius: floor applied to radii used for SURFACE/MESH GENERATION
            ONLY (never touches the actual point charges in RealChargeSystem).
            Force-field LJ radii (e.g. AMBER) can legitimately be zero for
            polar hydrogens (H bonded to O/N) -- that's correct for MD energy
            terms, but wrong for defining a dielectric boundary: it lets the
            meshed surface pass within <1A of a real point charge (see chat:
            diagnostic found exactly this, at the exact distance this
            geometric argument predicts). Standard practice in continuum
            electrostatics codes is a nonzero radius floor for this reason.
        """
        self.coords = coords
        self.radii_raw = radii  # kept for reference/debugging
        self.radii = np.maximum(radii, min_radius)
        self.min_radius = min_radius
        n_floored = int((radii < min_radius).sum())
        if n_floored:
            print(f"Applied min_radius={min_radius}A floor to {n_floored}/{len(radii)} "
                  f"atoms (mostly polar H) for surface generation only")

        self.tree = cKDTree(coords)
        self.k = min(k_neighbors, len(coords))

        lo = coords.min(axis=0) - radii.max() - pad
        hi = coords.max(axis=0) + radii.max() + pad
        self.grid_lo, self.grid_hi = lo, hi
        self.grid_spacing = grid_spacing

        self.surf_mesh = None
        self.tet_nodes = None
        self.tet_elems = None

    def sdf(self, pts):
        """Approximate signed distance to the union-of-spheres surface via a
        KD-tree k-nearest-neighbor query (exact all-pairs is infeasible at
        this atom count x grid-point count)."""
        pts = np.atleast_2d(pts)
        dists, idx = self.tree.query(pts, k=self.k)
        if self.k == 1:
            dists = dists[:, None]
            idx = idx[:, None]
        sdf_candidates = dists - self.radii[idx]
        return sdf_candidates.min(axis=1)

    def build_surface(self):
        nx = int((self.grid_hi[0] - self.grid_lo[0]) / self.grid_spacing)
        ny = int((self.grid_hi[1] - self.grid_lo[1]) / self.grid_spacing)
        nz = int((self.grid_hi[2] - self.grid_lo[2]) / self.grid_spacing)
        print(f"Grid: {nx} x {ny} x {nz} = {nx*ny*nz:,} points, spacing={self.grid_spacing}A")

        xs = np.linspace(self.grid_lo[0], self.grid_hi[0], nx)
        ys = np.linspace(self.grid_lo[1], self.grid_hi[1], ny)
        zs = np.linspace(self.grid_lo[2], self.grid_hi[2], nz)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

        field = self.sdf(pts).reshape(nx, ny, nz)

        spacing = ((self.grid_hi - self.grid_lo) / (np.array([nx, ny, nz]) - 1))
        verts, faces, normals, _ = measure.marching_cubes(field, level=0.0, spacing=spacing)
        verts = verts + self.grid_lo

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
        mesh.fix_normals()
        # IMPORTANT: do NOT keep only the single largest connected component.
        # For an elongated/looped molecule (e.g. tRNA's anticodon loop), real
        # structural features can legitimately mesh as separate closed
        # surfaces from the main body at finite grid resolution -- verified
        # by checking which real atoms are nearest each component's centroid
        # (see chat: components #2/#3 here were real anticodon-loop residues,
        # not noise). Only discard components below a small noise-volume
        # threshold (a handful of grid_spacing^3 -- genuine marching-cubes
        # aliasing, not real solute volume).
        components = mesh.split(only_watertight=False)
        noise_threshold = 3.0 * self.grid_spacing ** 3
        kept = [c for c in components if c.volume > noise_threshold]
        n_discarded = len(components) - len(kept)
        if n_discarded:
            print(f"Discarded {n_discarded} sub-noise-threshold fragments "
                  f"(volume < {noise_threshold:.2f} A^3 each)")
        mesh = trimesh.util.concatenate(kept) if len(kept) > 1 else kept[0]
        self.surf_mesh = mesh
        return mesh

    def build_volume(self, mindihedral=8, minratio=2.0):
        tg = tetgen.TetGen(self.surf_mesh.vertices, self.surf_mesh.faces)
        nodes, elems, _, _ = tg.tetrahedralize(order=1, mindihedral=mindihedral, minratio=minratio)
        self.tet_nodes, self.tet_elems = nodes, elems
        return nodes, elems

    def is_inside(self, pts):
        return self.sdf(pts) < 0


if __name__ == "__main__":
    coords, charges, radii = parse_pqr("4tna_out.pqr")
    print(f"Parsed {len(coords)} atoms, net charge {charges.sum():.3f}, "
          f"radius range [{radii.min():.2f}, {radii.max():.2f}]")

    geom = MoleculeGeometry(coords, radii, grid_spacing=1.0)
    geom.build_surface()
    print(f"Surface mesh: {len(geom.surf_mesh.vertices)} verts, "
          f"{len(geom.surf_mesh.faces)} faces, watertight={geom.surf_mesh.is_watertight}")