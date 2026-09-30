"""ESM-2 model registry for Project 1.

To plug in a new model, add one entry to ESM2_MODELS:
  key   : short tag used in filenames, CLI args, and plot labels
  hf_id : HuggingFace model identifier (facebook/esm2_*)
  hidden: embedding dimension (hidden_size in config.json)
  layers: number of transformer layers
  params: approximate parameter count (for display)

The hidden dimension is also confirmed at runtime from the actual checkpoint,
so the registry entry is informational — you can add a model without knowing
its exact hidden size and it will still work.
"""

ESM2_MODELS: dict[str, dict] = {
    # ── Available on HuggingFace (facebook/esm2_*) ────────────────────────────
    "esm2_8m": {
        "hf_id":  "facebook/esm2_t6_8M_UR50D",
        "hidden": 320,
        "layers": 6,
        "params": "8M",
        "note":   "Smallest / fastest; good for rapid iteration",
    },
    "esm2_35m": {
        "hf_id":  "facebook/esm2_t12_35M_UR50D",
        "hidden": 480,
        "layers": 12,
        "params": "35M",
        "note":   "Good balance of speed and representation quality",
    },
    "esm2_150m": {
        "hf_id":  "facebook/esm2_t30_150M_UR50D",
        "hidden": 640,
        "layers": 30,
        "params": "150M",
        "note":   "Mid-size; strong representations for most tasks",
    },
    "esm2_650m": {
        "hf_id":  "facebook/esm2_t33_650M_UR50D",
        "hidden": 1280,
        "layers": 33,
        "params": "650M",
        "note":   "Large; richer latent space, needs more GPU memory",
    },
    "esm2_3b": {
        "hf_id":  "facebook/esm2_t36_3B_UR50D",
        "hidden": 2560,
        "layers": 36,
        "params": "3B",
        "note":   "Very large; ~10 GB VRAM recommended",
    },
    "esm2_15b": {
        "hf_id":  "facebook/esm2_t48_15B_UR50D",
        "hidden": 5120,
        "layers": 48,
        "params": "15B",
        "note":   "Largest ESM-2; requires multiple GPUs or CPU offload",
    },
}

# Default cache root shared by all models in this project
DEFAULT_CACHE_ROOT = None   # None → each model caches in outputs/<tag>/cache/


def get_model(tag: str) -> dict:
    """Return the registry entry for a model tag, with a helpful error."""
    if tag not in ESM2_MODELS:
        available = ", ".join(ESM2_MODELS)
        raise ValueError(
            f"Unknown ESM-2 model tag '{tag}'. "
            f"Available: {available}\n"
            f"To add a new model, edit models.py and add an entry to ESM2_MODELS."
        )
    return ESM2_MODELS[tag]


def list_models() -> str:
    """Pretty-print the registry."""
    lines = ["Available ESM-2 models:", ""]
    for tag, info in ESM2_MODELS.items():
        lines.append(f"  {tag:<14} {info['params']:<6}  layers={info['layers']:<3}  "
                     f"hidden={info['hidden']:<5}  {info['note']}")
    return "\n".join(lines)
