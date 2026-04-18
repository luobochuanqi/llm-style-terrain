# AGENTS.md

## Project Overview

LLM-style terrain generation: Perlin noise → SDXL image-to-image refinement → Residual U-Net cVAE learned generation.

## Quick Reference

```bash
cd python-src
uv sync                    # Install deps (uv, not pip)
uv run python main.py      # Main workflow: Perlin + SDXL
```

## Architecture

```
python-src/
├── main.py                    # Perlin + SDXL refinement
├── train_unetcvae.py          # Residual U-Net cVAE training
├── infer_unetcvae.py          # cVAE inference (generate/interpolate/batch)
├── src/
│   ├── config.py              # GeneratorConfig, DiffusionConfig, ControlNetConfig
│   ├── generators/            # Perlin noise
│   ├── diffusion/             # SDXL inference
│   ├── train_unetcvae/        # cVAE model & training
│   ├── dataprocess/           # 256×256 preprocessing
│   └── dataprocess_512/       # 512×512 preprocessing
├── tools/                     # PNG→RAW conversion utilities
└── outputs/                   # All outputs (gitignored)

../data/training-dataset/      # External data directory (gitignored)
```

## Essential Commands

### Main Workflow
```bash
cd python-src
uv run python main.py              # Perlin → SDXL refinement
```

### cVAE Pipeline (Residual U-Net) ⭐
```bash
# Training (debug/fast/full modes)
uv run python train_unetcvae.py --fast

# Inference: generate from condition vector
uv run python infer_unetcvae.py generate \
    --checkpoint outputs/unetcvae/checkpoints/model_best.pth \
    --condition "4.0 5.0 1.5" \
    --output outputs/generated/danxia.png

# Interpolation between styles
uv run python infer_unetcvae.py interpolate \
    --start "2.0 3.0 1.0" --end "5.0 6.0 2.5" --steps 10
```

### Data Preprocessing
```bash
# 256×256 pipeline
uv run python -m src.dataprocess.preprocess
uv run python -m src.dataprocess.extract_features
uv run python -m src.dataprocess.visualize_features

# 512×512 pipeline
uv run python -m src.dataprocess_512.preprocess
uv run python -m src.dataprocess_512.extract_features
uv run python -m src.dataprocess_512.visualize_features
```

### Demo Scripts (no formal test framework)
```bash
uv run python test.py                # Brownian bridge demo
uv run python mapgen_demo.py         # DiT generation
uv run python heightmapstyle_demo.py # Style transfer
uv run python gamelandscape_demo.py  # Game landscape generation
uv run python controlnet_demo.py     # ControlNet
```

## Configuration

Edit `src/config.py`:
- `GeneratorConfig`: Perlin params (n=2^10, scale=300, octaves=6)
- `DiffusionConfig`: SDXL model, steps=25, strength=0.4 (low to preserve structure)
- `ControlNetConfig`: Canny thresholds, conditioning_scale=0.5

cVAE training config in `src/train_unetcvae/config.py` (modes: debug/fast/full).

## Key Requirements

- **Python 3.11+**
- **GPU required** for SDXL/cVAE (CUDA)
- **VRAM**: ≥8GB for SDXL, ≥4-5GB for cVAE (with AMP)
- Uses `uv` for dependency management

## Critical Gotchas

- **SDXL downloads ~7GB on first run** - enable `enable_cpu_offload=True` to reduce VRAM
- **Heightmap format**: grayscale PNG/RAW, black=valleys(0), white=peaks(255/65535)
- **cVAE condition vector**: "S R C" = Sharpness(1-6), Ruggedness(1-7), Complexity(1-4)
- **Danxia landform**: S=3.5-5.0, R=4.0-6.0, C=1.0-2.0
- **喀斯特地貌 (Karst)**: S=2.0-4.0, R=4.0-6.5, C=1.5-3.0
- **Data directory**: `../data/training-dataset/` (external, not in repo)
- **All outputs in `outputs/` are gitignored**

## Code Style

- Imports: stdlib → third-party → local (absolute within src/)
- 4 spaces, 88 char max, f-strings, type hints required
- Chinese comments acceptable

## No Formal Tooling

- No pytest/unittest - demo scripts serve as tests
- No linting/typecheck configured
- No pre-commit hooks
