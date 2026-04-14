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
│   │   ├── diffusion/   # SDXL inference (sdxl_inference.py, controlnet_inference.py)
│   │   ├── mapgen/      # DiT-based heightmap generation (experimental)
│   │   ├── heightmapstyle/  # Heightmap style transfer
│   │   └── gamelandscape/   # Game landscape generation
│   ├── tools/           # Utility scripts (png2raw.py)
│   ├── test.py          # Demo/test script (Brownian bridge simulation)
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

# Run a single test script
python test.py

# Run demo scripts
python mapgen_demo.py          # DiT map generation demo
python heightmapstyle_demo.py  # Heightmap style transfer demo
python gamelandscape_demo.py   # Game landscape demo
python controlnet_demo.py      # ControlNet inference demo
```

## Key Requirements

- **Python 3.11+** (specified in `.python-version`)
- **GPU required** for SDXL inference (uses `torch` with CUDA)
- Uses `uv` for dependency management, not pip
- No formal test suite - demo scripts serve as manual tests

## Configuration

Edit `src/config.py` dataclasses:
- `GeneratorConfig`: Perlin noise params (size=2^n, scale, octaves, seed)
- `DiffusionConfig`: SDXL model, inference steps, prompts for Danxia landform
- `ControlNetConfig`: ControlNet model, conditioning scale, canny thresholds
- `OutputConfig`: output directories, filenames

## Code Style Guidelines

### Imports
- Standard library imports first (sys, pathlib, numpy)
- Third-party imports second (torch, diffusers, PIL)
- Local imports third (from ..config import ...)
- Use absolute imports within src/ package

### Formatting
- 4 spaces for indentation (no tabs)
- Maximum line length: 88 characters (follow PEP 8)
- Use f-strings for string formatting
- Type hints required for function signatures

### Naming Conventions
- snake_case for functions, variables, modules
- PascalCase for classes (e.g., PerlinHeightmapGenerator)
- UPPER_CASE for constants
- Descriptive names in English; Chinese comments allowed

### Types
- Use `Optional[T]` for nullable values
- Use `Union` for multiple types
- Dataclasses for configuration (see `src/config.py`)
- Explicit return type annotations

### Error Handling
- Catch specific exceptions (e.g., `ValueError`, `KeyboardInterrupt`)
- Use try/except blocks around model loading and inference
- Print user-friendly error messages with emoji indicators:
  - ✅ Success, ⚠️ Warning, ❌ Error, ⏭️ Skipped

### Documentation
- Module docstrings at top of each file (triple-quoted, Chinese allowed)
- Docstrings for all public functions and classes
- Include Args, Returns, Raises sections for complex functions

### Comments
- Chinese comments acceptable for internal documentation
- Keep comments concise and meaningful
- Use inline comments sparingly

## Outputs

- `outputs/perlin/heightmap.raw` - Raw uint8 heightmap from Perlin noise
- `outputs/diffusion/heightmap.png` - SDXL-refined heightmap as PNG
- `outputs/controlnet/` - ControlNet output directory
- `outputs/heightmapstyle/` - Style transfer outputs
- `outputs/gamelandscape/` - Game landscape outputs
- `outputs/mapgen/` - DiT map generation outputs

## Gotchas

- SDXL model downloads on first run (~7GB)
- Enable `enable_cpu_offload=True` in config to reduce VRAM usage
- Heightmaps are grayscale: black=valleys, white=peaks
- Output format is `raw` binary for Perlin, `png` for diffusion result
- All output directories are gitignored

## Notes for Agents

- No Cursor rules (.cursor/rules/ or .cursorrules) exist
- No GitHub Copilot instructions (.github/copilot-instructions.md) exist
- No formal linting or type checking configured yet
- Code follows PEP 8 conventions with Chinese comments allowed
- No pytest or unittest framework - use demo scripts for testing
