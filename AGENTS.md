# AGENTS.md

## Project Overview

Python project for generating terrain heightmaps: Perlin noise → SDXL image-to-image refinement.

## Structure

```
├── python-src/          # All Python code
│   ├── main.py          # Entry point
│   ├── src/
│   │   ├── config.py    # Dataclass config (GeneratorConfig, DiffusionConfig, OutputConfig)
│   │   ├── generators/  # Perlin noise generation (perlin.py)
│   │   ├── diffusion/   # SDXL inference (sdxl_inference.py)
│   │   └── mapgen/      # DiT-based heightmap generation (experimental)
│   ├── outputs/         # Generated files (gitignored)
│   └── .venv/           # Virtual env (gitignored)
```

## Commands

All commands run from `python-src/`:

```bash
cd python-src

# Run the terrain generator
python main.py

# Install dependencies (uv is the package manager)
uv sync
```

## Key Requirements

- **Python 3.11+** (specified in `.python-version`)
- **GPU required** for SDXL inference (uses `torch` with CUDA)
- Uses `uv` for dependency management, not pip
- **No test suite yet** - tests directory does not exist

## Configuration

Edit `src/config.py` dataclasses:
- `GeneratorConfig`: Perlin noise params (size=2^n, scale, octaves, seed)
- `DiffusionConfig`: SDXL model, inference steps, prompts for Danxia landform
- `OutputConfig`: output directory, filenames

## Outputs

- `outputs/heightmap_perlin.raw` - Raw uint8 heightmap from Perlin noise
- `outputs/heightmap_diffusion.png` - SDXL-refined heightmap as PNG

## Gotchas

- SDXL model downloads on first run (~7GB)
- Enable `enable_cpu_offload=True` in config to reduce VRAM usage
- Heightmaps are grayscale: black=valleys, white=peaks
- Output format is `raw` binary for Perlin, `png` for diffusion result
