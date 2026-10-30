"""
Two controls for centered_latent_manifold.py, kept in one small module.

(1) RANDOM-INITIALISED MODEL  (--random-init)
    Same architecture, same tokenizer, same extraction code, but the weights are random, never
    pretrained. If a randomly initialised network "explains" as much of the electrostatic labels as
    the pretrained one, the pretrained model's signal is just letters + position + architecture,
    not something learned from RNA data. Run it with the same --layer as the pretrained run for a
    like-for-like comparison.

(2) LEAVE-FAMILY-OUT SPLIT  (--family-table, --family-folds K, --family-fold i)
    The default train/val/test split is by structure, so a tRNA in the test set still has hundreds of
    close cousins in training. Here every RNA FAMILY goes entirely to train or entirely to test, so
    the test score measures generalisation to families the probe has never seen.
      * families are packed into K folds greedily by size (largest first, lightest fold) so folds are
        balanced and deterministic;
      * fold i is the test set; everything else is train (+ the original 10% val structures);
      * structures whose chains belong to families on BOTH sides are dropped (default) so no family
        leaks across the split; --family-straddle longest keeps them, assigned by their longest chain.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("latent_controls")


# ----------------------------------------------------------------------
# (1) random-initialised model
# ----------------------------------------------------------------------

def install_random_init_loader(seed: int) -> int:
    """Replace `load_model` (defined in rna_embedding_probe_v2) by a version that builds the SAME
    architecture from its config with random weights. Patches every already-imported project module
    that imported `load_model` by name. Returns the number of modules patched."""
    import rna_embedding_probe_v2 as rep

    orig = rep.load_model

    def load_model_random(model_key: str, device: str):
        import torch
        from multimolecule import AutoModel, RnaTokenizer
        try:
            from multimolecule import AutoConfig
        except ImportError:                      # older multimolecule: use transformers' registry
            from transformers import AutoConfig

        hf_id = rep.MODEL_REGISTRY[model_key]
        tokenizer = RnaTokenizer.from_pretrained(hf_id)
        config = AutoConfig.from_pretrained(hf_id)
        config.output_hidden_states = True
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = AutoModel.from_config(config)    # random weights -- nothing pretrained is loaded
        model.to(device)
        model.eval()
        n_layers = model.config.num_hidden_layers + 1
        tok_max = getattr(tokenizer, "model_max_length", None)
        cfg_max = getattr(model.config, "max_position_embeddings", None)
        if tok_max is None or tok_max > 100_000:
            tok_max = 1_000_000
        if cfg_max is None or cfg_max > 100_000:
            cfg_max = 1_000_000
        log.warning("RANDOM-INIT %s (seed %d): weights are NOT pretrained", model_key, seed)
        return tokenizer, model, n_layers, min(tok_max, cfg_max)

    patched = 0
    for name, mod in list(sys.modules.items()):
        if not name.startswith(("rna_", "electrostatic", "centered", "__main__")):
            continue
        try:
            if getattr(mod, "load_model", None) is orig:
                mod.load_model = load_model_random
                patched += 1
        except Exception:
            continue
    log.warning("random-init loader installed (seed %d), patched %d module(s)", seed, patched)
    if patched == 0:
        raise RuntimeError("could not find any module using load_model; random-init NOT active")
    return patched


# ----------------------------------------------------------------------
# (2) leave-family-out split
# ----------------------------------------------------------------------

def load_family_table(path: Path) -> dict:
    """CSV with columns pdb_id, chain, family  ->  {'PDB::chain': family}. `chain` is the chain
    letter used in your labels / FASTA headers (the internal one)."""
    t = pd.read_csv(path, dtype=str)
    need = {"pdb_id", "chain", "family"}
    if not need <= set(t.columns):
        raise ValueError(f"{path} must have columns {sorted(need)}; has {list(t.columns)}")
    t = t.dropna(subset=["pdb_id", "chain", "family"])
    return {f"{p.upper()}::{c}": f for p, c, f in zip(t["pdb_id"], t["chain"], t["family"])}


def build_family_split(pdb_groups, chain_groups, splits, table: dict, n_folds: int, fold: int,
                       straddle: str = "drop"):
    """Returns (new_splits, keep_mask, summary_df, info_dict) for the rows given.
    pdb_groups / chain_groups are the per-residue arrays used by centered_latent_manifold
    (chain_groups look like 'PDB::chain')."""
    if not 0 <= fold < n_folds:
        raise ValueError(f"--family-fold must be in [0, {n_folds - 1}]")
    if straddle not in ("drop", "longest"):
        raise ValueError(straddle)
    pdb_groups, chain_groups, splits = np.asarray(pdb_groups), np.asarray(chain_groups), np.asarray(splits)

    chains, inv, counts = np.unique(chain_groups, return_inverse=True, return_counts=True)
    chain_pdb = np.array([c.split("::", 1)[0] for c in chains])
    fam_of_chain = np.array([table.get(f"{c.split('::', 1)[0].upper()}::{c.split('::', 1)[1]}",
                                       f"unassigned::{c}") for c in chains], dtype=object)
    n_unassigned = int(np.sum([str(f).startswith("unassigned::") for f in fam_of_chain]))
    if n_unassigned:
        log.warning("%d of %d chains have no family; each is treated as its own family "
                    "(this may leak real families across the split -- fix the table if many)",
                    n_unassigned, len(chains))

    fam_size = pd.Series(counts, index=fam_of_chain).groupby(level=0).sum()
    order = sorted(fam_size.index, key=lambda f: (-int(fam_size[f]), str(f)))
    load = np.zeros(n_folds)
    fold_of_family = {}
    for f in order:
        k = int(np.argmin(load))
        fold_of_family[f] = k
        load[k] += fam_size[f]
    chain_fold = np.array([fold_of_family[f] for f in fam_of_chain])
    chain_is_test = chain_fold == fold

    df = pd.DataFrame({"pdb": chain_pdb, "test": chain_is_test, "n": counts})
    g = df.groupby("pdb").agg(any_t=("test", "any"), all_t=("test", "all"))
    longest = df.sort_values("n", ascending=False).drop_duplicates("pdb").set_index("pdb")["test"]
    if straddle == "drop":
        pdb_test = g["all_t"]
        pdb_drop = g["any_t"] & ~g["all_t"]
    else:
        pdb_test = longest.reindex(g.index)
        pdb_drop = pd.Series(False, index=g.index)

    row_pdb_test = pd.Series(pdb_groups).map(pdb_test).to_numpy(dtype=bool)
    row_drop = pd.Series(pdb_groups).map(pdb_drop).to_numpy(dtype=bool)
    keep = ~row_drop
    new_splits = np.where(row_pdb_test, "test", np.where(splits == "val", "val", "train")).astype(object)

    # leakage check on the kept rows: no family on both sides
    fam_row = np.asarray(fam_of_chain)[inv]
    fam_test = set(fam_row[keep & (new_splits == "test")])
    fam_train = set(fam_row[keep & (new_splits != "test")])
    overlap = fam_test & fam_train
    if straddle == "drop" and overlap:
        raise RuntimeError(f"family leakage across split: {sorted(map(str, overlap))[:5]}")

    summary = pd.DataFrame({
        "family": fam_size.index.astype(str), "n_residues": fam_size.to_numpy(),
        "fold": [fold_of_family[f] for f in fam_size.index],
    })
    summary["n_chains"] = [int((fam_of_chain == f).sum()) for f in fam_size.index]
    summary["is_test_family"] = summary["fold"] == fold
    summary = summary.sort_values("n_residues", ascending=False)

    info = {
        "n_folds": n_folds, "fold": fold, "straddle": straddle,
        "n_chains": int(len(chains)), "n_unassigned_chains": n_unassigned,
        "n_families": int(len(fam_size)), "n_test_families": int(len(fam_test)),
        "n_structures": int(len(g)), "n_structures_dropped": int(pdb_drop.sum()),
        "n_structures_test": int((pdb_test & ~pdb_drop).sum()),
        "residues_total": int(len(pdb_groups)), "residues_dropped": int(row_drop.sum()),
        "residues_test": int((keep & (new_splits == "test")).sum()),
        "residues_train": int((keep & (new_splits == "train")).sum()),
        "residues_val": int((keep & (new_splits == "val")).sum()),
        "family_leakage": sorted(map(str, overlap))[:10],
    }
    log.info("FAMILY SPLIT fold %d/%d: %d test families, %d test structures, %d test residues; "
             "dropped %d straddling structures (%d residues)", fold, n_folds, info["n_test_families"],
             info["n_structures_test"], info["residues_test"], info["n_structures_dropped"],
             info["residues_dropped"])
    return new_splits, keep, summary, info


def save_family_split_report(outdir: Path, summary: pd.DataFrame, info: dict):
    outdir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(outdir / "family_split_summary.csv", index=False)
    (outdir / "family_split_info.json").write_text(json.dumps(info, indent=2))
