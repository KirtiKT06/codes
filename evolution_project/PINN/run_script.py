"""
Run it as: python run_script.py 2>&1 | tee training_log.txt"""
from run_demo import run_on_pdb
import torch
net, scaler, fields, gammas = run_on_pdb(
    "1EHZ.pdb",
    chain_id="A",          # set to "A" explicitly if grep above shows multiple chains
    ionic_strength_M=0.15,  # physiological monovalent
    n_epochs=80000,          
)
torch.save(net.state_dict(), "pinn_1ehz.pt")
print("Gamma_i range:", gammas.min(), gammas.max())
print("Gamma_i mean:", gammas.mean())