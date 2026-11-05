# -*- coding: utf-8 -*-
"""Sampling from a trained CLINCH generator checkpoint.

The policy is an autoregressive transformer over SMILES tokens. Sampling starts from the
start token and draws one token at a time until the end token appears or the length limit
is reached. Sequences that have already finished are padded so that the batch keeps its
shape, and the padding is stripped when the tokens are decoded.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from rdkit import Chem, RDLogger

from .models import ActorCritic
from .vocabulary import SMILESTokenizer

RDLogger.DisableLog("rdApp.*")

# architecture of the released policy, from the training config
DEFAULT_ARCH = dict(embed_size=384, hidden_size=1536, num_layers=6,
                    nhead=8, dropout=0.2, max_len=150)


def load_generator(checkpoint, vocab_file, device="cpu", arch=None):
    """Build the policy and load a checkpoint into it.

    The architecture is read from the checkpoint itself where possible, so a checkpoint
    trained with a different width or depth still loads without extra arguments.
    """
    device = torch.device(device)
    tokenizer = SMILESTokenizer(vocab_file=str(vocab_file))

    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "embedding.weight" not in state:
        for key in ("model_state_dict", "state_dict", "agent", "policy"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if "embedding.weight" not in state:
        raise ValueError(
            "the checkpoint does not look like a generator policy, "
            "no embedding.weight was found among %d entries" % len(state))

    cfg = dict(DEFAULT_ARCH)
    if arch:
        cfg.update(arch)
    vocab_size, embed_size = state["embedding.weight"].shape
    cfg["embed_size"] = int(embed_size)
    n_layers = len({k.split(".")[2] for k in state if k.startswith("transformer.layers.")})
    if n_layers:
        cfg["num_layers"] = n_layers
    ff = "transformer.layers.0.linear1.weight"
    if ff in state:
        cfg["hidden_size"] = int(state[ff].shape[0])
    if "pos_encoder.pe" in state:
        cfg["max_len"] = int(state["pos_encoder.pe"].shape[1])

    if vocab_size != len(tokenizer.vocab):
        raise ValueError(
            "vocabulary mismatch, the checkpoint expects %d tokens but %s holds %d. "
            "Use the vocab.json that was saved with this checkpoint."
            % (vocab_size, vocab_file, len(tokenizer.vocab)))

    agent = ActorCritic(vocab_size=vocab_size, embed_size=cfg["embed_size"],
                        hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"],
                        dropout=cfg["dropout"], nhead=cfg["nhead"],
                        max_len=cfg["max_len"])
    missing, unexpected = agent.load_state_dict(state, strict=False)
    if missing:
        raise ValueError("checkpoint is missing %d tensors, first few %s"
                         % (len(missing), missing[:4]))
    agent.to(device).eval()
    return agent, tokenizer, cfg


@torch.no_grad()
def sample_batch(agent, tokenizer, n, max_len=150, temperature=1.0, device="cpu"):
    """Draw n raw SMILES strings. Some of them will not parse."""
    device = torch.device(device)
    seq = torch.full((n, 1), tokenizer.start_idx, dtype=torch.long, device=device)
    running = torch.ones(n, dtype=torch.bool, device=device)
    for _ in range(max_len):
        action, _, _, _ = agent.get_action_and_value(seq, temperature=temperature)
        masked = action * running.long() + tokenizer.pad_idx * (~running).long()
        seq = torch.cat([seq, masked.unsqueeze(1)], dim=1)
        running = running & (masked != tokenizer.end_idx)
        if not running.any():
            break
    return tokenizer.decode(seq[:, 1:])


@torch.no_grad()
def sample(agent, tokenizer, total, batch_size=256, max_len=150, temperature=1.0,
           device="cpu", progress=True):
    """Draw `total` raw SMILES in chunks so a large request fits in memory."""
    out = []
    bar = None
    if progress:
        try:
            from tqdm import tqdm
            bar = tqdm(total=total, desc="sampling", unit="mol")
        except ImportError:
            bar = None
    while len(out) < total:
        n = min(batch_size, total - len(out))
        out.extend(sample_batch(agent, tokenizer, n, max_len=max_len,
                                temperature=temperature, device=device))
        if bar:
            bar.update(n)
    if bar:
        bar.close()
    return out[:total]


def canonicalise(raw, dedupe=True):
    """Drop unparseable strings, canonicalise the rest, optionally drop duplicates.

    Returns the kept molecules and a small report, so validity and uniqueness can be
    quoted without recomputing them.
    """
    kept, seen = [], set()
    n_invalid = 0
    n_dup = 0
    for s in raw:
        if not s:
            n_invalid += 1
            continue
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            n_invalid += 1
            continue
        try:
            canon = Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            n_invalid += 1
            continue
        if dedupe and canon in seen:
            n_dup += 1
            continue
        seen.add(canon)
        kept.append(canon)
    n = len(raw)
    report = {
        "sampled": n,
        "valid": n - n_invalid,
        "validity_pct": round(100.0 * (n - n_invalid) / n, 2) if n else 0.0,
        "unique": len(kept),
        "uniqueness_pct": (round(100.0 * len(kept) / (n - n_invalid), 2)
                           if n - n_invalid else 0.0),
        "duplicates_dropped": n_dup,
    }
    return kept, report
