# RMViT

A two-stage Transformer for radio map estimation from sparse measurements and building maps, without transmitter-location inputs.

RMViT uses separate CNN branches to extract radio and building features, a Transformer encoder and query decoder to model global dependencies, and multi-scale skip fusion to retain spatial details. Stage I reconstructs a coarse radio map. Stage II takes the coarse map together with the original inputs and predicts a residual correction, which is added to the coarse map to obtain the final estimate. The two stages are trained sequentially, with Stage I frozen during Stage II training.

## Dataset

This work uses the **RadioMapSeer** dataset. Download the dataset and find its citation information on the official website:

[RadioMapSeer official website](https://radiomapseer.github.io/)

Extract the dataset under `./data`, preserving its original directory structure. The current training scripts use IRT2 radio maps, complete building maps, and the antenna images read by the loader:

```text
data/
├── gain/IRT2/
└── png/
    ├── buildings_complete/
    └── antennas/
```

## Environment

Use a dedicated Python environment. Install compatible PyTorch and torchvision builds for your CPU or CUDA environment, then install the project dependencies:

```bash
pip install -r requirement.txt
```

The project also depends on NumPy, scikit-image, pandas, matplotlib, and Pillow. A CUDA-capable GPU is recommended for training; the scripts fall back to CPU when CUDA is unavailable.

## Running

Run the commands from the project root.

First, train Stage I to obtain the coarse reconstruction model:

```bash
python train_vit_128_skip.py
```

After Stage I finishes, train the Stage II refinement model:

```bash
python train_vit_128_skip_refine.py
```

The entry points use `module_vit_128_skip.py` and `module_vit_128_skip_refine.py` for the two models, and `loaders.py` for data loading. Each training script validates during training, saves the best checkpoint according to its validation loss, and evaluates that checkpoint after training. Checkpoints and test summaries are written to the configured output directory.
