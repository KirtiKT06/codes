"""
Feature-extraction hook for RhoFold+ (ml4bio/RhoFold), for use as
Category B (MSA-conditioned, pre-fold) and Category C (3D-structure-aware,
post-fold) representations in the layer-wise electrostatics probe.

STATUS: scoped starting point, NOT verified against the real RhoFold
codebase -- this sandbox has no network access to clone the repo, download
the gated checkpoint, or run a forward pass. Every attribute path below
(rhofold.msa_stack, rhofold.pairwise2heavyatom, etc.) is inferred from the
architecture description in the paper (RNA-FM embed -> Rhoformer stack of
MSA+pair updates over 10 recycling cycles -> IPA structure module) and from
the GitHub README's CLI usage, NOT from reading RhoFold's actual module
tree. Treat every `rhofold.<attr>` reference here as "the shape of the hook
you need to write," not working code -- clone
https://github.com/ml4bio/RhoFold, inspect rhofold/model/rhofold.py's
forward() method, and adjust names/shapes to match before running anything.

Setup (real steps, from the RhoFold README):
    git clone https://github.com/ml4bio/RhoFold
    cd RhoFold && pip install -e .
    # request access at the HF form linked from the README, then:
    wget https://huggingface.co/cuhkaih/rhofold/resolve/main/rhofold_pretrained_params.pt \
        -O pretrained/RhoFold_pretrained.pt

Usage sketch (see main() at the bottom):
    python rhofold_feature_hook.py --fasta 1EHZ_A.fasta --ckpt pretrained/RhoFold_pretrained.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_rhofold(ckpt_path: str, device: str = "cuda"):
    """Load a RhoFold model from its checkpoint.

    UNVERIFIED: RhoFold's actual loader is in rhofold/rhofold.py or
    inference.py in the real repo -- this assumes a similar shape to a
    standard PyTorch checkpoint load, but the real entry point may wrap
    this differently (config object, separate MSA-generation step object,
    etc.). Check inference.py's `main()` in the real repo for the actual
    load sequence before trusting this function.
    """
    import torch
    from rhofold.rhofold import RhoFold  # path per GitHub repo structure

    model = RhoFold()  # may need a config argument -- check repo
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.to(device)
    model.eval()
    return model


def extract_rhofold_features(model, fasta_path: str, a3m_path: str | None,
                              device: str = "cuda") -> dict[str, np.ndarray]:
    """Run RhoFold on one sequence and pull out per-residue representations
    at two points in the pipeline:

      - 'msa_conditioned'  (Category B): the single-sequence-track
        representation coming OUT of the Rhoformer stack, after MSA+pair
        co-evolutionary refinement but BEFORE the IPA structure module
        commits to explicit 3D coordinates.
      - 'structure_aware'  (Category C): the per-residue representation
        coming OUT of the IPA structure module, after 3D geometry has been
        resolved.

    If a3m_path is None, RhoFold generates its own MSA automatically (this
    is the whole point -- it replaces the Infernal/Rfam pipeline you'd
    otherwise have to build). That auto-search needs its sequence
    databases configured per the RhoFold README; if that setup is
    unavailable, RhoFold also supports single-sequence-only input as a
    lower-accuracy fallback (per --input_fas alone in the CLI) -- useful
    for a first smoke test of this hook even before the MSA databases are
    set up, though at that point 'msa_conditioned' isn't really
    MSA-conditioned and shouldn't be reported as Category B.

    UNVERIFIED: the actual attribute names for intermediate activations
    (`model.rhoformer`, `model.structure_module`, whatever the real repo
    calls them) need confirming against rhofold/rhofold.py's forward()
    method. The clean way to get these without editing RhoFold's source is
    a PyTorch forward hook (see below) registered on the two boundary
    submodules once you've identified their real names by printing
    `model` after loading.
    """
    import torch

    sequence = Path(fasta_path).read_text().splitlines()[1].strip()

    captured = {}

    def _capture(name):
        def _hook(_module, _input, output):
            captured[name] = output
        return _hook

    # UNVERIFIED submodule names -- replace with whatever `print(model)`
    # shows for (a) the last Rhoformer/recycling block and (b) the
    # structure module's final per-residue output.
    handle_b = model.rhoformer.register_forward_hook(_capture("msa_conditioned"))
    handle_c = model.structure_module.register_forward_hook(_capture("structure_aware"))

    try:
        with torch.no_grad():
            # UNVERIFIED call signature -- check inference.py for how it
            # actually invokes the model (likely takes tokenized MSA +
            # sequence features, not a raw string).
            model(fasta=fasta_path, a3m=a3m_path, device=device)
    finally:
        handle_b.remove()
        handle_c.remove()

    return {
        "msa_conditioned": captured["msa_conditioned"].squeeze(0).cpu().numpy(),
        "structure_aware": captured["structure_aware"].squeeze(0).cpu().numpy(),
        "sequence": sequence,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta", required=True, help="Single-sequence FASTA (e.g. from the label pipeline).")
    ap.add_argument("--a3m", default=None, help="Optional pre-built MSA; omit to let RhoFold auto-search.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, help="Path to save .npz of extracted features.")
    args = ap.parse_args()

    model = load_rhofold(args.ckpt, args.device)
    feats = extract_rhofold_features(model, args.fasta, args.a3m, args.device)

    print(f"sequence length: {len(feats['sequence'])}")
    print(f"msa_conditioned shape: {feats['msa_conditioned'].shape}")
    print(f"structure_aware shape: {feats['structure_aware'].shape}")

    if args.out:
        np.savez(args.out, msa_conditioned=feats["msa_conditioned"],
                  structure_aware=feats["structure_aware"], sequence=feats["sequence"])


if __name__ == "__main__":
    main()