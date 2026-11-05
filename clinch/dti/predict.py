# -*- coding: utf-8 -*-
"""Ensemble inference for the CLINCH interaction predictor.

Each member of the ensemble is loaded in turn, run over the whole input, then released,
which keeps peak memory at one model rather than five. For every ligand and protein pair
the ensemble returns the median class probability and a spread term.

The spread is the interquartile range of the positive class probability across members.
With five models the interquartile range is steadier than the standard deviation, which a
single outlying member can dominate. This is the same quantity the reinforcement learning
reward used as its uncertainty discount, so numbers here line up with the training runs.
"""
from __future__ import annotations

import gc
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .model import DTI_Model, DtiDataset, collate_fn, smiles_to_graph

log = logging.getLogger(__name__)

# two-class formulation of the deployed model
#   hbond        class 0 is no hydrogen bond, class 1 is one or more
#   hydrophobic  class 0 is six or fewer contacts, class 1 is seven or more
TARGET_COLS = {
    "hbond": ["prob_hbond_0", "prob_hbond_1"],
    "hydrophobic": ["prob_hydrophobic_0", "prob_hydrophobic_1"],
}


class EnsembleDTIPredictor:
    def __init__(self, ensemble_dir, params_path, num_models=5, device=None):
        self.ensemble_dir = Path(ensemble_dir)
        self.params_path = Path(params_path)
        if not self.ensemble_dir.is_dir():
            raise FileNotFoundError("no ensemble directory at %s" % self.ensemble_dir)
        if not self.params_path.is_file():
            raise FileNotFoundError("no parameter file at %s" % self.params_path)

        with open(self.params_path, encoding="utf-8") as fh:
            self.cfg = json.load(fh)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.num_models = num_models
        self.interaction_types = self.cfg.get("interaction_types",
                                              ["hbond", "hydrophobic"])
        self.target_cols = {t: TARGET_COLS[t] for t in self.interaction_types}
        self.n_outputs = {t: len(c) for t, c in self.target_cols.items()}
        self.in_dim = smiles_to_graph("C").x.shape[1]

        found = [i for i in range(num_models) if self._path(i).is_file()]
        if not found:
            raise FileNotFoundError(
                "no member checkpoints under %s, expected model_0/best_model.pt "
                "through model_%d/best_model.pt" % (self.ensemble_dir, num_models - 1))
        if len(found) < num_models:
            log.warning("only %d of %d members present, continuing with %s",
                        len(found), num_models, found)
        self.members = found
        log.info("ensemble ready on %s, %d members, atom feature width %d",
                 self.device, len(found), self.in_dim)

    def _path(self, i):
        return self.ensemble_dir / ("model_%d" % i) / "best_model.pt"

    def predict(self, df: pd.DataFrame, batch_size: int = 16) -> pd.DataFrame:
        required = {"ligand_smiles", "protein_sequence"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError("input is missing the columns %s" % sorted(missing))
        if df.empty:
            return pd.DataFrame()

        work = df.copy()
        for cols in self.target_cols.values():
            for col in cols:
                if col not in work:
                    work[col] = 0.0

        collected = {}
        for i in self.members:
            self._run_member(i, work, collected, batch_size)
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        rows = []
        for (smi, seq), per_type in collected.items():
            row = {"ligand_smiles": smi, "protein_sequence": seq}
            for itype in self.interaction_types:
                preds = per_type.get(itype) or []
                if not preds:
                    for col in self.target_cols[itype]:
                        row[col] = float("nan")
                    row["unc_%s" % itype] = float("nan")
                    row["n_models_%s" % itype] = 0
                    continue
                stack = np.stack(preds, axis=0)
                median = np.median(stack, axis=0)
                for k, col in enumerate(self.target_cols[itype]):
                    row[col] = float(median[k])
                positive = stack[:, 1]
                if len(positive) >= 3:
                    spread = float(np.percentile(positive, 75)
                                   - np.percentile(positive, 25))
                else:
                    spread = float(positive.std())
                row["unc_%s" % itype] = spread
                row["std_%s" % itype] = float(positive.std())
                row["n_models_%s" % itype] = int(len(positive))
            rows.append(row)
        return pd.DataFrame(rows)

    def _run_member(self, idx, df, collected, batch_size):
        path = self._path(idx)
        model = None
        try:
            model = DTI_Model(self.cfg, self.in_dim, self.n_outputs, self.device,
                              self.cfg["protein_max_len"])
            model.load_state_dict(torch.load(path, map_location=self.device))
            model.to(self.device).eval()

            dataset = DtiDataset(df, self.target_cols, self.cfg["protein_max_len"],
                                 self.cfg.get("attention_supervision", False))
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=collate_fn, num_workers=0)

            use_amp = self.device.type == "cuda"
            with torch.no_grad():
                for batch in loader:
                    if batch is None:
                        continue
                    batch = batch.to(self.device)
                    if use_amp:
                        with torch.amp.autocast("cuda"):
                            global_preds, _ = model(batch)
                    else:
                        global_preds, _ = model(batch)
                    for j, smi in enumerate(batch.smiles):
                        key = (smi, batch.protein_sequence[j])
                        slot = collected.setdefault(
                            key, {t: [] for t in self.interaction_types})
                        for itype in self.interaction_types:
                            # heads emit log probabilities
                            p = global_preds[itype][j].exp().float().cpu().numpy()
                            slot[itype].append(p)
            log.info("member %d done", idx)
        except Exception as exc:
            log.error("member %d failed, %s", idx, exc)
            raise
        finally:
            del model
