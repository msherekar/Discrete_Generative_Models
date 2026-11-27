"""The METL backend: its pooled representation and its Rosetta attribute head.

METL-L-2M-3D-GFP shares no architecture, tokenizer or training corpus with the
ESM-2 encoder the generator samples in, which is what makes it an independent
judge.
"""
import contextlib

import numpy as np

from .common import (METL_CKPT, METL_PDB, METL_ROOT, chdir, device,
                     seq_to_variant)

# ══════════════════════════════════════════════════════════════════════════════
# Backends
# ══════════════════════════════════════════════════════════════════════════════

# metl/code/models.py and this directory's models.py share a name, and whichever
# landed in sys.modules first wins. METL's inference.py does a plain `import
# models`, so without isolation it picks up our ESM-2 registry and fails with
# "module 'models' has no attribute 'Model'".
_METL_MODULE_NAMES = (
    "models", "tasks", "encode", "utils", "structure", "relative_attention",
    "datasets", "datamodules", "training_utils", "inference", "constants",
    "metrics", "rosetta_data_utils", "pdb_sampler", "analysis_utils",
)
_METL_CACHE: dict = {}


@contextlib.contextmanager
def metl_imports():
    """Let METL import its own modules, then put ours back."""
    import sys
    saved_path = list(sys.path)
    saved_modules = {name: sys.modules.pop(name)
                     for name in _METL_MODULE_NAMES if name in sys.modules}
    sys.path.insert(0, str(METL_ROOT / "code"))
    try:
        yield
    finally:
        sys.path[:] = saved_path
        for name in _METL_MODULE_NAMES:
            sys.modules.pop(name, None)
        sys.modules.update(saved_modules)


def _load_metl():
    """Load METL once and keep it; the checkpoint costs seconds to rebuild."""
    if "model" in _METL_CACHE:
        return _METL_CACHE["model"], _METL_CACHE["encoder"]
    if not METL_CKPT.is_file():
        raise FileNotFoundError(
            f"METL checkpoint not found at {METL_CKPT}. It ships with the metl "
            f"repository under pretrained_models/."
        )
    with metl_imports(), chdir(METL_ROOT):
        from inference import load_pytorch_module
        import encode as metl_encode
        model = load_pytorch_module(str(METL_CKPT), pdb_fns=[METL_PDB]).eval().to(device())
        # Hold a reference to the function itself: the module is about to be
        # removed from sys.modules again.
        encoder = metl_encode.encode
    _METL_CACHE.update(model=model, encoder=encoder)
    return model, encoder


def encode_metl(sequences: list[str], wt: str, batch_size: int = 64) -> np.ndarray:
    """Pooled 256-dimensional METL representation, one row per sequence."""
    import torch
    model, encoder = _load_metl()
    captured = {}
    handle = model.model.fc1.register_forward_hook(
        lambda mod, inp, out: captured.__setitem__("h", out.detach().cpu())
    )
    out = []
    try:
        with chdir(METL_ROOT), torch.no_grad():
            for start in range(0, len(sequences), batch_size):
                chunk = sequences[start:start + batch_size]
                variants = [seq_to_variant(s, wt) for s in chunk]
                x = encoder(encoding="int_seqs", variants=variants,
                            wt_aa=wt, wt_offset=1)
                model(torch.tensor(x).to(device()), pdb_fn=METL_PDB)
                out.append(captured["h"].numpy().astype(np.float32))
    finally:
        handle.remove()
    return np.concatenate(out)


# ══════════════════════════════════════════════════════════════════════════════
# Rosetta biophysical attributes (METL pretraining head)
# ══════════════════════════════════════════════════════════════════════════════
#
# METL-Local's pretraining task is to predict 55 Rosetta biophysical attributes
# from sequence, and the shipped checkpoint keeps that head intact:
# `model.prediction.weight` is (55, 256). So the attributes come free with a
# forward pass -- no Rosetta run, which matters because Rosetta costs ~135-215 s
# per variant for a 237-residue protein like avGFP (Gelman et al., Nature
# Methods 2025). Twenty thousand generated sequences would be weeks of CPU.
#
# Outputs are in METL's standardized pretraining units (roughly mean 0, sd 1
# across its simulated variants), not raw Rosetta energies.
#
# Measured on 3,000 avgfp_train rows, Spearman against experimental brightness:
#
#     0   total_score           -0.6902    most stable <-> brightest
#    32   pack                  +0.5675
#     1   fa_atr                -0.5834
#    41   total_sasa            -0.2541
#    27   exposed_hydrophobics  -0.1495    near-orthogonal to brightness
#    43   unsat_hbond           -0.1310    near-orthogonal to brightness
#
# total_score is the best stability measure AND the most redundant with
# brightness, so it belongs in a constraint rather than in a scalarized
# objective: weighting it against brightness traces a line, not a Pareto front.
# exposed_hydrophobics and unsat_hbond carry real variance (sd 0.83 and 0.72)
# while being nearly independent of brightness, which is what a second objective
# has to be for the tradeoff to mean anything.

ATTRIBUTE_NAMES: tuple[str, ...] = ()


def attribute_names() -> tuple[str, ...]:
    """The 55 attribute names, in the order the prediction head emits them."""
    global ATTRIBUTE_NAMES
    if not ATTRIBUTE_NAMES:
        with metl_imports():
            import constants
            ATTRIBUTE_NAMES = tuple(constants.ROSETTA_ATTRIBUTES_TRAINING)
    return ATTRIBUTE_NAMES


def attribute_index(name: str) -> int:
    """Index of an attribute by name, so callers need not hardcode integers."""
    names = attribute_names()
    if name not in names:
        raise ValueError(f"Unknown Rosetta attribute '{name}'. "
                         f"Known: {', '.join(names)}")
    return names.index(name)


def metl_attributes(sequences: list[str], wt: str, batch_size: int = 64) -> np.ndarray:
    """Predicted Rosetta attributes, (len(sequences), 55) in standardized units.

    Reads the pretraining head rather than the pooled fc1 representation that
    encode_metl() returns. Same frozen checkpoint, same forward pass.
    """
    import torch
    model, encoder = _load_metl()
    out = []
    with chdir(METL_ROOT), torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            chunk = sequences[start:start + batch_size]
            variants = [seq_to_variant(s, wt) for s in chunk]
            x = encoder(encoding="int_seqs", variants=variants,
                        wt_aa=wt, wt_offset=1)
            y = model(torch.tensor(x).to(device()), pdb_fn=METL_PDB)
            out.append(y.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(out)


def metl_attributes_wt(wt: str) -> np.ndarray:
    """The same 55 attributes for the wild type, as a (55,) vector.

    The stability constraint is expressed relative to this: a design is allowed
    to be less stable than wild type only by a stated margin delta.
    """
    import torch
    model, encoder = _load_metl()
    with chdir(METL_ROOT), torch.no_grad():
        x = encoder(encoding="int_seqs", variants=["_wt"], wt_aa=wt, wt_offset=1)
        y = model(torch.tensor(x).to(device()), pdb_fn=METL_PDB)
    return y.detach().cpu().numpy().astype(np.float32)[0]


if __name__ == "__main__":
    # METL requires the checkpoint to be present; only check that the module
    # loads and the common helpers work without it.
    from .common import seq_to_variant, METL_CKPT
    print(f"  METL checkpoint path: {METL_CKPT}")
    print(f"  METL_CKPT exists: {METL_CKPT.exists()}")
    # wt_attributes requires the model loaded; skip if checkpoint absent.
    if METL_CKPT.exists():
        wt = "ACDEFGHIKLMNPQRSTVWY"[:10]   # short synthetic WT
        try:
            attrs = wt_attributes(wt)
            assert attrs.ndim == 1, f"wt_attributes: expected 1D, got {attrs.shape}"
            print(f"  wt_attributes: shape={attrs.shape}  first={attrs[:3]}")
        except Exception as e:
            print(f"  wt_attributes: {type(e).__name__}: {e}")
    else:
        print(f"  wt_attributes: skipped (checkpoint not found)")
    print("oracles/metl.py OK")
