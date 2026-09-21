# VERA-MEL
<p align="center">
  <img src="framework.png"
       alt="VERA Framework"
       width="900">
</p>

Official release code for the VERA multimodal entity-linking system evaluated on WikiMEL and WikiDiverse. This repository is prepared for the ICASSP code-release track and contains the runnable source, configurations, and training/evaluation scripts.

The public model implementation is named **VERA** throughout this repository. The model combines frozen multimodal encoders with text evidence selection, semantic mining, bidirectional reading, visual grounding, and reliability-aware score aggregation.


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


## Dataset
Download the datasets from [FissFuse dataset repository](https://github.com/pengfei-luo/FissFuse).
Create a data root directory at ./data/ and place the downloaded datasets under this directory.


## Running the code

The wrappers are location-independent and can be launched from any working directory. `bash run.sh` starts the default WikiMEL VERA training run. `GPU` selects visible GPUs, `PYTHON` selects the Python executable, `DATASET` selects WikiMEL or WikiDiverse, and `CONFIG` overrides the selected YAML file.




