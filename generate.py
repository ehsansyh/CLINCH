#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sample molecules from the trained CLINCH generator.

The policy was optimised by PPO against the interaction predictor while being held
inside the chemistry of the known actives. Sampling it reproduces the pool the
candidate library was selected from.

Examples
    python generate.py --n 1000 --output generated.csv

    python generate.py \
        --checkpoint checkpoints/generator/best_model.pt \
        --vocab checkpoints/generator/vocab.json \
        --n 10000 --batch_size 500 --temperature 1.0 \
        --device cuda --output pool.csv --score

With --score the sampled molecules are passed through the interaction ensemble, which
is what the selection stage in the paper used. Without it the script only samples and
canonicalises, which needs no GPU and no protein model.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from clinch.generator.sample import canonicalise, load_generator, sample  # noqa: E402
from clinch.targets import WT_HIV1_PROTEASE                                # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path,
                    default=HERE / "checkpoints" / "generator" / "best_model.pt")
    ap.add_argument("--vocab", type=Path,
                    default=HERE / "checkpoints" / "generator" / "vocab.json")
    ap.add_argument("--n", type=int, default=1000, help="how many molecules to sample")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="1.0 is the deployment setting, lower is greedier")
    ap.add_argument("--max_len", type=int, default=150)
    ap.add_argument("--seed", type=int, default=74)
    ap.add_argument("--device", default=None, help="cuda, cuda:0 or cpu")
    ap.add_argument("--keep_duplicates", action="store_true")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--report", type=Path, default=None,
                    help="optional JSON file for the validity and uniqueness numbers")
    ap.add_argument("--score", action="store_true",
                    help="also run the interaction ensemble on what was sampled")
    ap.add_argument("--ensemble_dir", type=Path,
                    default=HERE / "checkpoints" / "dti_ensemble")
    ap.add_argument("--params", type=Path, default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="[%(levelname)s] %(message)s")

    import torch
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if not args.checkpoint.is_file():
        print("generator checkpoint not found, %s\n\nSee checkpoints/README.md."
              % args.checkpoint, file=sys.stderr)
        return 1
    if not args.vocab.is_file():
        print("vocabulary not found, %s" % args.vocab, file=sys.stderr)
        return 1

    agent, tokenizer, cfg = load_generator(args.checkpoint, args.vocab, device=device)
    print("policy loaded on %s, %d tokens, %d layers, width %d"
          % (device, len(tokenizer.vocab), cfg["num_layers"], cfg["embed_size"]))

    raw = sample(agent, tokenizer, args.n, batch_size=args.batch_size,
                 max_len=min(args.max_len, cfg["max_len"]),
                 temperature=args.temperature, device=device)
    kept, report = canonicalise(raw, dedupe=not args.keep_duplicates)
    print("sampled %(sampled)d, valid %(valid)d (%(validity_pct).2f%%), "
          "unique %(unique)d (%(uniqueness_pct).2f%% of valid)" % report)

    out = pd.DataFrame({"smiles": kept})

    if args.score:
        from clinch.dti.predict import EnsembleDTIPredictor
        params = args.params or (args.ensemble_dir / "model_params.json")
        try:
            predictor = EnsembleDTIPredictor(ensemble_dir=args.ensemble_dir,
                                             params_path=params, device=device)
        except FileNotFoundError as exc:
            print("cannot score, %s" % exc, file=sys.stderr)
            return 1
        scored = predictor.predict(
            pd.DataFrame({"ligand_smiles": kept,
                          "protein_sequence": WT_HIV1_PROTEASE}))
        scored = scored.rename(columns={"ligand_smiles": "smiles"})
        scored = scored.drop(columns=["protein_sequence"])
        out = out.merge(scored, on="smiles", how="left")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, float_format="%.6f")
    print("wrote %s, %d molecules" % (args.output, len(out)))

    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("wrote %s" % args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
