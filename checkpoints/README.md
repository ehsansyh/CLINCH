# Weights

Model weights are not tracked in git. Download the release archive and unpack it here so
that the layout matches what the scripts expect.

```
checkpoints/
├── dti_ensemble/
│   ├── model_params.json
│   ├── model_0/best_model.pt
│   ├── model_1/best_model.pt
│   ├── ...
│   ├── model_8/best_model.pt
│   └── model_9/best_model.pt
└── generator/
    ├── best_model.pt
    └── vocab.json
```

`vocab.json` ships with the repository. Everything else has to be downloaded.

## Interaction ensemble

Ten models, loaded one at a time by `predict_dti.py`. The directory names matter.
`model_0` through `model_9` is the pattern the loader looks for, and a missing member is
skipped with a warning rather than treated as an error, so check the printed member count
if the numbers look off.

This is the deployment ensemble of the paper, trained on all wild-type data and used as
the reward proxy during reinforcement learning and to score the candidate pool. It is a
different object from the twenty-five model configuration, five scaffold-grouped folds by
five seeds, which exists only to measure generalisation. Do not mix the two. A
twenty-five model directory loaded with `--num_models 10` would quietly take the first
ten folds' models and report numbers that mean nothing.

Each member holds roughly one million trainable parameters, a few megabytes on disk. The
protein encoder is frozen and is not stored inside them, which is why they are small.

`model_params.json` is the configuration the ensemble was trained with. The architecture
is rebuilt from it, so it has to be the file that was saved alongside these weights.
The keys the loader reads are

| key | what it does |
| --- | --- |
| `hidden_size`, `num_layers`, `heads`, `dropout` | network shape |
| `protein_max_len` | truncation length for the protein encoder |
| `interaction_types` | which heads exist, normally `hbond` and `hydrophobic` |
| `attention_supervision` | whether the dataset emits attention targets |

One caution. A `best_params.json` left over from hyperparameter search is not the same
file. The search configuration carries the defaults that were explored, not the settings
the final models were fitted with, and `attention_supervision` in particular reads
`false` there while supervision was enabled in the deployed models. Use the config that
was saved with the checkpoints.

## Generator

One file, `best_model.pt`, a state dictionary for the actor-critic policy. The loader
reads the width, depth, feed-forward size and maximum length straight out of the tensor
shapes, so a checkpoint from a different configuration still loads.

The released policy is 64 vocabulary tokens, 384 embedding width, 6 transformer layers,
1536 feed-forward width, 8 heads and a 150 token limit.

`vocab.json` has to be the vocabulary the checkpoint was trained with. The loader
compares the vocabulary size against the embedding matrix and refuses to continue when
they disagree, because a silent mismatch would decode into nonsense rather than fail.

## Protein encoder

Not stored here. On first use the code pulls `facebook/esm2_t33_650M_UR50D` from the
Hugging Face hub, roughly 2.6 GB. Set `HF_HOME` to control where it lands, and
`HF_HUB_OFFLINE=1` once it is cached if the machine has no network.

## Checking an install

```bash
python -c "import torch, torch_geometric, transformers, rdkit; print('imports fine')"
python generate.py --n 20 --device cpu --output /tmp/smoke.csv
```

The second command only needs the generator checkpoint. It is the quickest way to tell
whether the policy and the vocabulary belong together.
