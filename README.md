# VERA-MEL

Official release code for the VERA multimodal entity-linking system evaluated on WikiMEL and WikiDiverse. This repository is prepared for the ICASSP code-release track and contains the runnable source, configurations, and training/evaluation scripts.

The public model implementation is named **VERA** throughout this repository. The model combines frozen multimodal encoders with text evidence selection, semantic mining, bidirectional reading, visual grounding, and reliability-aware score aggregation.

## Repository layout

```text
VERA/
├── codes/
│   ├── main.py
│   ├── lit_model.py
│   ├── model/modeling.py
│   └── utils/functions.py
├── config/
│   ├── wikimel.yaml
│   └── wikidiverse.yaml
├── scripts/
│   ├── train.sh
│   └── test.sh
├── requirements.txt
├── run.sh
└── .gitignore
```

No dataset, pretrained backbone, checkpoint, training log, or evaluation result is bundled. These files are intentionally kept outside the repository because of size, licensing, and reproducibility considerations.

## Dependencies

The reference environment is Linux, Python 3.10, CUDA 12.1, PyTorch 2.5.1, and Lightning 2.0.7. Create an isolated environment and install a CUDA-compatible PyTorch build first:

```bash
conda create -n vera python=3.10
conda activate vera
# Install the PyTorch/torchvision pair appropriate for your CUDA driver.
# For the reference CUDA 12.1 environment, use the official PyTorch index.
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

`mamba-ssm` may require a working CUDA compiler/toolchain matching the installed PyTorch build. If you use a different CUDA version, install compatible PyTorch, torchvision, and `mamba-ssm` versions together.

## Data and pretrained models

Download WikiMEL and WikiDiverse using the dataset instructions from the [FissFuse dataset repository](https://github.com/pengfei-luo/FissFuse). The link is provided only for dataset provenance; VERA is the model released here. The expected files are:

```text
<data-root>/WikiMEL/
├── WIKIMEL_train.jsonl
├── WIKIMEL_valid.jsonl
├── WIKIMEL_test.jsonl
├── entities.jsonl
└── wikimel_image.h5

<data-root>/WikiDiverse/
├── WIKIDIVERSE_train.jsonl
├── WIKIDIVERSE_valid.jsonl
├── WIKIDIVERSE_test.jsonl
├── entities.jsonl
└── image.h5py
```

The default configs use `./data/WikiMEL` and `./data/WikiDiverse`. To keep data elsewhere, export:

```bash
export VERA_WIKIMEL_ROOT=/path/to/WikiMEL
export VERA_WIKIDIVERSE_ROOT=/path/to/WikiDiverse
```

The CLIP configuration expects `openai/clip-vit-base-patch32`; the optional BERT/ResNet path expects `bert-base-uncased` and `microsoft/resnet-101`. To use local cached directories, set the model variables and leave `local_files: true`:

```bash
export VERA_CLIP_MODEL=/path/to/clip-vit-base-patch32
export VERA_BERT_MODEL=/path/to/bert_base_uncased
export VERA_RESNET_MODEL=/path/to/resnet-101
```

For a first online download, edit the selected YAML and set `local_files: false`. Review the model and dataset licenses before redistribution.

## Running

The wrappers are location-independent and can be launched from any working directory. `bash run.sh` starts the default WikiMEL VERA training run. `GPU` selects visible GPUs, `PYTHON` selects the Python executable, `DATASET` selects WikiMEL or WikiDiverse, and `CONFIG` overrides the selected YAML file.

### One-command training

After installing the dependencies, preparing both external assets, and exporting any required paths, run:

```bash
bash run.sh
```

This command uses `config/wikimel.yaml` by default, trains VERA, validates after each epoch, restores the best validation checkpoint, and runs the test split. It requires a visible CUDA GPU. To run WikiDiverse instead:

```bash
DATASET=wikidiverse bash run.sh
```

The repository does not bundle the data or pretrained models, so `bash run.sh` cannot work in a fresh clone until those external assets are installed and configured.

### Smoke test

The smoke commands use one percent of the training data for one epoch. They require the external dataset, pretrained models, and at least one CUDA device:

```bash
GPU=0 CONFIG=./config/wikimel.yaml \
  ./scripts/train.sh --percentage 0.01 --epoch 1 --run_name smoke_wikimel

GPU=0 CONFIG=./config/wikidiverse.yaml \
  ./scripts/train.sh --percentage 0.01 --epoch 1 --run_name smoke_wikidiverse
```

Training creates `checkpoints/`, `runs/`, and `rank_save/` locally. These generated directories are ignored by Git and are not part of the release.

### Training

```bash
GPU=0 CONFIG=./config/wikimel.yaml ./scripts/train.sh
GPU=0 CONFIG=./config/wikidiverse.yaml ./scripts/train.sh
```

The default configs use the paper experiment settings. Override selected arguments without editing YAML:

```bash
./scripts/train.sh \
  --config ./config/wikimel.yaml \
  --seed 43 --percentage 1 --epoch 100 --run_name reproduction
```

### Evaluation from a checkpoint

```bash
CHECKPOINT=/path/to/model.ckpt \
CONFIG=./config/wikimel.yaml \
  ./scripts/test.sh
```

The checkpoint path may be absolute or relative to the invocation directory. Evaluation still writes generated metrics/ranks under the repository’s ignored output directories.

## Reproducibility notes

- The reference seed is `43`; the YAML files also fix batch sizes, candidate counts, optimizer settings, and Lightning trainer settings.
- The implementation requires CUDA because the training entry point explicitly validates a visible GPU.
- Dataset files and pretrained backbones are external inputs. Their exact versions, local paths, and licenses should be recorded in an experiment log.
- The release does not include trained weights. A checkpoint can be released separately after checking storage, dataset, backbone, and model-parameter licensing requirements.
- Results produced by a local run should not be committed unless the paper’s artifact policy explicitly requires them.

## Relation to the AVM-EDR repository format

This release follows the useful parts of the AVM-EDR presentation style—paper-oriented README, dependency setup, dataset provenance, a compact source tree, and simple run commands—but it is not a copy. VERA does not use AVM-EDR’s LLM-rescoring module or `.env` credentials, so those files are not included. A framework figure, citation metadata, and an open-source license should be added only after the authors supply the approved figure, bibliographic information, and licensing decision. This repository currently has no `LICENSE` file; absence of a license does not grant permission to reuse the code.
