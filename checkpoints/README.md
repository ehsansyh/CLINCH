# Weights

Model weights are not tracked in git. Download the release archive and unpack it here so
that the layout matches what the scripts expect.

```
checkpoints/
├── dti_ensemble/
│   ├── best_params.json
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

`best_params.json` is the configuration the ensemble was trained with. The architecture
is rebuilt from it, so it has to be the file that was saved alongside these weights.
The loader looks for `best_params.json` inside the ensemble directory, falls back to
`model_params.json` for older runs, and `--params` overrides both. The keys it reads are

| key | what it does |
| --- | --- |
| `hidden_size`, `num_layers`, `heads`, `dropout` | network shape |
| `protein_max_len` | truncation length for the protein encoder |
| `interaction_types` | which heads exist, normally `hbond` and `hydrophobic` |
| `attention_supervision` | whether the dataset emits attention targets |

One caution, and it is a real trap. More than one `best_params.json` can exist in a
project, written by different tuning runs, and they disagree on things that change the
architecture. Two files seen here differ on `hidden_size`, 256 against 128. Loading
weights against the wrong one fails, and the raw PyTorch error is not obvious, so the
loader catches it and says which parameter file it was building from. Always take the
file that sits with the weights you are loading.

A second note on `attention_supervision`. It reads `false` in the deployment
configuration, which looks wrong given that attention was supervised during training.
At inference the flag only controls whether the dataset emits attention target tensors,
which nothing needs when you are reading predictions, so `false` is correct here and
says nothing about how the models were fitted.

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
