#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Predict interaction class probabilities for ligand and protein pairs.

The ensemble returns, for each pair, the probability that the complex forms at least one
hydrogen bond and the probability that it forms seven or more hydrophobic contacts, plus
a spread term that says how far the ensemble members disagree.

Examples
    python predict_dti.py --input examples/ligands.csv --output preds.csv

    python predict_dti.py \
        --input my_ligands.csv \
        --ensemble_dir checkpoints/dti_ensemble \
        --params checkpoints/dti_ensemble/model_params.json \
        --device cuda --batch_size 32 --output preds.csv

The input is a CSV with a `smiles` or `ligand_smiles` column. A `protein_sequence`
column is optional. When it is absent the wild-type HIV-1 protease homodimer is used,
which is the target the released checkpoints were trained on.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from clinch.targets import WT_HIV1_PROTEASE                   # noqa: E402


def _load_predictor_class():
    """Imported late so that --help works before the heavy stack is installed."""
    try:
        from clinch.dti.predict import EnsembleDTIPredictor
    except ImportError as exc:
        raise SystemExit(
            "a dependency for interaction prediction is missing, %s\n"
            "Install the inference stack with  pip install -r requirements.txt" % exc)
    return EnsembleDTIPredictor


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path,
                    help="CSV with a smiles or ligand_smiles column")
    ap.add_argument("--output", required=True, type=Path, help="where to write the CSV")
    ap.add_argument("--ensemble_dir", type=Path,
                    default=HERE / "checkpoints" / "dti_ensemble",
                    help="directory holding model_0/best_model.pt and the rest")
    ap.add_argument("--params", type=Path, default=None,
                    help="best_params.json, found inside the ensemble dir by default")
    ap.add_argument("--num_models", type=int, default=10,
                    help="members to load, the released deployment ensemble has 10")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default=None, help="cuda, cuda:0 or cpu")
    ap.add_argument("--protein", default=None,
                    help="override the protein sequence for every row")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="[%(levelname)s] %(message)s")

    if not args.input.is_file():
        print("input file not found, %s" % args.input, file=sys.stderr)
        return 1
    df = pd.read_csv(args.input)
    if "ligand_smiles" not in df.columns:
        if "smiles" in df.columns:
            df = df.rename(columns={"smiles": "ligand_smiles"})
        else:
            print("the input needs a smiles or ligand_smiles column, found %s"
                  % list(df.columns), file=sys.stderr)
            return 1
    if args.protein:
        df["protein_sequence"] = args.protein
    elif "protein_sequence" not in df.columns:
        df["protein_sequence"] = WT_HIV1_PROTEASE
        print("no protein_sequence column, using the wild-type HIV-1 protease dimer")

    df = df[df.ligand_smiles.notna()].reset_index(drop=True)
    if df.empty:
        print("no usable rows in the input", file=sys.stderr)
        return 1

    EnsembleDTIPredictor = _load_predictor_class()
    try:
        predictor = EnsembleDTIPredictor(
            ensemble_dir=args.ensemble_dir, params_path=args.params,
            num_models=args.num_models, device=args.device)
    except FileNotFoundError as exc:
        print("%s\n\nSee checkpoints/README.md for where to put the weights."
              % exc, file=sys.stderr)
        return 1

    print("scoring %d pairs with %d ensemble members"
          % (len(df), len(predictor.members)))
    out = predictor.predict(df, batch_size=args.batch_size)
    if out.empty:
        print("the ensemble returned nothing", file=sys.stderr)
        return 1

    out = out.rename(columns={"ligand_smiles": "smiles"})
    out = out.drop(columns=["protein_sequence"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, float_format="%.6f")
    print("wrote %s, %d rows" % (args.output, len(out)))
    if "prob_hbond_1" in out:
        print("  mean P(at least one hydrogen bond)   %.3f" % out.prob_hbond_1.mean())
    if "prob_hydrophobic_1" in out:
        print("  mean P(seven or more hydrophobic)    %.3f"
              % out.prob_hydrophobic_1.mean())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
