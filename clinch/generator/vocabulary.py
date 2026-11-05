"""
SMILES tokenizer with regex-based multi-character atom support.

Old (character) mode: each char a token. "Cl" -> ['C', 'l'].
New (regex) mode: multi-char atoms and bracket expressions are single tokens.
    "Cl"       -> ['Cl']
    "[C@@H]"   -> ['[C@@H]']
    "Brc1ccc"  -> ['Br', 'c', '1', 'c', 'c', 'c']

This is the same scheme used by ChemBERTa / MoLFormer.
Backward compatibility: if `use_regex=False` is passed, falls back to the
old per-character behaviour.
"""

import json
import os
import re
import torch


# Standard SMILES tokenization regex (Schwaller et al. 2019 / RXNMapper style).
# Order matters: bracketed atoms first, then 2-char atoms, then 1-char atoms,
# then bonds / parentheses / ring closures.
SMILES_REGEX_PATTERN = (
    r"(\[[^\]]+\]"            # [N+], [C@@H], [nH], [Si], etc.
    r"|Br|Cl|Si|Se|Na|Li|Mg|Ca|Al|As|Sb|Te"  # 2-char atoms
    r"|\%\d{2}"               # ring closures > 9, e.g. %10
    r"|[BCNOPSFIcnops]"       # 1-char atoms / aromatic
    r"|[0-9]"                 # ring closures 0-9
    r"|=|#|/|\\|@|\+|-|\.|\(|\))"
)
SMILES_REGEX = re.compile(SMILES_REGEX_PATTERN)


class SMILESTokenizer:
    SPECIAL_TOKENS = ['<pad>', '<start>', '<end>', '<unk>']

    def __init__(self, vocab_file="vocab.json", use_regex=True):
        self.vocab_file = vocab_file
        self.use_regex = use_regex

        if os.path.exists(vocab_file):
            with open(vocab_file, 'r') as f:
                data = json.load(f)
            # Support both old (list) and new (dict) formats
            if isinstance(data, dict):
                self.vocab = data.get('vocab', [])
                # Saved tokenizer carries its own mode flag — honour it
                self.use_regex = data.get('use_regex', use_regex)
            else:
                self.vocab = data
            print(f"Vocab loaded from {vocab_file} (size={len(self.vocab)}, "
                  f"use_regex={self.use_regex})")
        else:
            self._load_default_vocab()

        self._build_maps()

    # --------- vocab management ---------
    def _load_default_vocab(self):
        """Sensible default. Regex mode: include common multi-char atoms.
        Will be overwritten by build_from_smiles() on first pretrain run."""
        atoms_2char = ['Br', 'Cl', 'Si', 'Se']
        atoms_1char = ['B', 'C', 'N', 'O', 'P', 'S', 'F', 'I', 'H',
                       'c', 'n', 'o', 'p', 's']
        digits = [str(d) for d in range(10)]
        ring_pct = [f'%{i:02d}' for i in range(10, 30)]
        bonds = ['=', '#', '/', '\\', '@', '+', '-', '.', '(', ')']
        # Common bracketed atoms seen in drug-like SMILES
        brackets = ['[C@H]', '[C@@H]', '[N+]', '[O-]', '[N-]', '[NH]',
                    '[nH]', '[NH+]', '[NH2+]', '[NH3+]', '[n+]',
                    '[C@]', '[C@@]', '[S+]', '[O+]', '[S-]', '[Si]',
                    '[Cl-]', '[Br-]', '[Na+]', '[K+]']

        self.vocab = (list(self.SPECIAL_TOKENS)
                      + atoms_2char + atoms_1char
                      + digits + ring_pct + bonds + brackets)

    def _build_maps(self):
        self.char_to_int = {c: i for i, c in enumerate(self.vocab)}
        self.int_to_char = {i: c for i, c in enumerate(self.vocab)}
        self.pad_idx = self.char_to_int.get('<pad>', 0)
        self.start_idx = self.char_to_int.get('<start>', 1)
        self.end_idx = self.char_to_int.get('<end>', 2)
        self.unk_idx = self.char_to_int.get('<unk>', self.pad_idx)

    def build_from_smiles(self, smiles_iter):
        """Scan a corpus of SMILES strings and rebuild the vocab to include
        every token actually seen. Call this once before pretraining if you
        change tokenizer mode or want to support a larger chemical space."""
        token_set = set()
        for smi in smiles_iter:
            for tok in self.tokenize(smi):
                token_set.add(tok)
        # Keep special tokens first
        regular = sorted(token_set - set(self.SPECIAL_TOKENS))
        self.vocab = list(self.SPECIAL_TOKENS) + regular
        self._build_maps()
        print(f"Rebuilt vocab from corpus: {len(self.vocab)} tokens, "
              f"use_regex={self.use_regex}")
        return self

    @property
    def size(self):
        return len(self.vocab)

    # --------- tokenize / detokenize ---------
    def tokenize(self, smi):
        """Split a SMILES string into tokens."""
        if self.use_regex:
            return SMILES_REGEX.findall(smi)
        return list(smi)

    def detokenize(self, tokens):
        """Inverse of tokenize — multi-char tokens just concatenate cleanly."""
        return ''.join(tokens)

    # --------- batch encode / decode ---------
    def encode(self, smiles_list, max_len=150, device='cpu'):
        batch_tensor = []
        for smi in smiles_list:
            tokens = self.tokenize(smi)
            ids = ([self.start_idx]
                   + [self.char_to_int.get(t, self.unk_idx) for t in tokens]
                   + [self.end_idx])
            if len(ids) < max_len:
                ids += [self.pad_idx] * (max_len - len(ids))
            batch_tensor.append(ids[:max_len])
        return torch.tensor(batch_tensor, dtype=torch.long, device=device)

    def decode(self, token_ids):
        smiles = []
        for seq in token_ids:
            toks = []
            for idx in seq:
                idx_item = idx.item() if hasattr(idx, 'item') else idx
                if idx_item == self.end_idx:
                    break
                if idx_item in (self.start_idx, self.pad_idx, self.unk_idx):
                    continue
                toks.append(self.int_to_char.get(idx_item, ""))
            smiles.append(self.detokenize(toks))
        return smiles

    def save_vocab(self, filepath):
        payload = {'vocab': self.vocab, 'use_regex': self.use_regex}
        with open(filepath, 'w') as f:
            json.dump(payload, f, indent=2)
