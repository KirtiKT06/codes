"""
embeddings.py

ESM-2 embedding extraction via HuggingFace `transformers`. This module
requires network access to huggingface.co and a torch install with (ideally)
CUDA -- it will NOT run inside this sandbox, which is why it's kept separate
from the tested pure-numpy modules (intrinsic_dim, robustness_evolvability,
cv_discovery, geometry_analysis). Run this part on your own machine/cluster,
same place you already run ESM-2 for the GB1 project.

Extracts embeddings from MULTIPLE layers, not just the last -- Valeriani et
al. (2022) found homology/structural information peaks at an intermediate
"plateau" layer, often before the final layer, so defaulting to
last-hidden-state (the common practice in most downstream-prediction code)
may not be the right layer for THIS question.

Supports both mean-pooling (whole-sequence) and position-specific pooling
(embedding of, or centered on, the mutated residue) -- the AAV/Spike
embedding-limits work found global mean pooling dilutes localized mutational
signal, so for interface-localized proteins (GB1, RBD) position-specific
extraction is worth comparing against mean pooling, not assuming mean
pooling is sufficient.
"""

from __future__ import annotations
import numpy as np
from typing import Sequence


MODEL_SIZES = {
    "esm2_t12_35M":  "facebook/esm2_t12_35M_UR50D",
    "esm2_t30_150M": "facebook/esm2_t30_150M_UR50D",
    "esm2_t33_650M": "facebook/esm2_t33_650M_UR50D",
    "esm2_t36_3B":   "facebook/esm2_t36_3B_UR50D",
}


def extract_esm2_embeddings(
    sequences: Sequence[str],
    model_name: str = "esm2_t33_650M",
    layers: Sequence[int] | None = None,
    device: str = "cuda",
    batch_size: int = 8,
    pooling: str = "mean",
    mutated_positions: Sequence[int] | None = None,
    position_window: int = 0,
) -> dict[int, np.ndarray]:
    """
    Returns {layer_index: (n_sequences, hidden_dim) array}.

    layers: which hidden-state layers to keep (0 = embedding layer,
        1..n_layers = transformer blocks). If None, keeps ALL layers -- do
        this at least once per protein to locate the plateau layer
        empirically (per Valeriani et al.), rather than assuming it.
    pooling: 'mean' (average over all non-special tokens) or 'position'
        (average over `mutated_positions[i] +/- position_window`, i.e. a
        window centered on the mutation for sequence i). 'position' requires
        `mutated_positions` to be provided, one integer per sequence
        (0-indexed into the sequence, not including the BOS token that ESM
        tokenizers add).
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    hf_name = MODEL_SIZES.get(model_name, model_name)
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModelForMaskedLM.from_pretrained(hf_name, output_hidden_states=True)
    model.eval().to(device)

    if pooling == "position" and mutated_positions is None:
        raise ValueError("pooling='position' requires mutated_positions")

    all_hidden: dict[int, list[np.ndarray]] = {}

    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch_seqs = list(sequences[start:start + batch_size])
            enc = tokenizer(batch_seqs, return_tensors="pt", padding=True,
                             truncation=True).to(device)
            out = model(**enc)
            hidden_states = out.hidden_states  # tuple: (n_layers+1) x (B, L, H)

            n_layers = len(hidden_states)
            layer_ids = layers if layers is not None else list(range(n_layers))

            attn_mask = enc["attention_mask"]  # (B, L)
            # ESM tokenizers: token 0 = <cls>/BOS, last valid token = <eos>.
            # Exclude both special tokens from pooling.
            special_mask = attn_mask.clone()
            special_mask[:, 0] = 0
            seq_lens = attn_mask.sum(dim=1)
            for b, L in enumerate(seq_lens.tolist()):
                special_mask[b, int(L) - 1] = 0  # drop EOS

            for li in layer_ids:
                h = hidden_states[li]  # (B, L, H)
                if pooling == "mean":
                    m = special_mask.unsqueeze(-1).float()
                    pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
                elif pooling == "position":
                    pooled_list = []
                    for b in range(h.shape[0]):
                        seq_idx = start + b
                        p = mutated_positions[seq_idx] + 1  # +1 for BOS offset
                        lo = max(p - position_window, 1)
                        hi = min(p + position_window + 1, h.shape[1] - 1)
                        pooled_list.append(h[b, lo:hi, :].mean(dim=0))
                    pooled = torch.stack(pooled_list, dim=0)
                else:
                    raise ValueError(f"Unknown pooling mode: {pooling}")
                all_hidden.setdefault(li, []).append(pooled.cpu().numpy())

    return {li: np.concatenate(chunks, axis=0) for li, chunks in all_hidden.items()}


def save_embeddings(path: str, embeddings: dict[int, np.ndarray], sequences: Sequence[str]):
    np.savez_compressed(path, sequences=np.array(sequences, dtype=object),
                         **{f"layer_{k}": v for k, v in embeddings.items()})


def load_embeddings(path: str) -> tuple[list[str], dict[int, np.ndarray]]:
    data = np.load(path, allow_pickle=True)
    sequences = list(data["sequences"])
    embeddings = {int(k.split("_")[1]): data[k] for k in data.files if k.startswith("layer_")}
    return sequences, embeddings
