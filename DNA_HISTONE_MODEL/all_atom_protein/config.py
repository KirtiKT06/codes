"""
Shared config loader for the nucleosome-octamer MD pipeline.
Every stage script (01_strip_dna.py, 02_fix_structure.py, ...) imports
load_config() from here so there is exactly one place that knows how
to read input.json.
"""
import json
import os


def load_config(path="input.json"):
    with open(path) as f:
        cfg = json.load(f)

    # Make every path in "paths" absolute w.r.t. output_dir where relevant,
    # and make sure output_dir exists.
    out_dir = cfg["paths"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    return cfg


def outpath(cfg, key):
    """Convenience: cfg['paths'][key], joined with output_dir if not already absolute."""
    p = cfg["paths"][key]
    if os.path.isabs(p):
        return p
    return os.path.join(cfg["paths"]["output_dir"], p)
