# isolate_analytic.py
import torch
from structure import load_rna_structure
from pb_pinn import PBFields, DEVICE

phosphate_xyz, phosphate_q, allatom_xyz = load_rna_structure("1EHZ.pdb", chain_id="A")
fields = PBFields(phosphate_xyz, phosphate_q, allatom_xyz, ionic_strength_M=0.15)

center = phosphate_xyz[0]
for d in [2.0, 3.0, 5.0, 10.0, 20.0, 40.0]:
    pt = torch.tensor([center + [d, 0, 0]], dtype=torch.float32, device=DEVICE)
    phi_analytic_only = fields.phi_analytic(pt).item()
    print(f"d={d:5.1f}  phi_analytic_only={phi_analytic_only:+.4f}")