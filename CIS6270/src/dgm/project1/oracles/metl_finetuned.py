"""The released finetuned METL GFP models, which predict brightness directly.

These are Gelman et al.'s own target models from Zenodo record 14908509, loaded
through the lightweight `metl-pretrained` package rather than the full repo, so
they can serve as a cross-check oracle and reproduce the paper's design setting.
"""
import numpy as np

from .common import AMINO_ACIDS, METL_TARGET_UUIDS, ROOT, device, seq_to_variant

# Loaded models, keyed by UUID; each costs seconds to rebuild from disk.
_CACHE: dict = {}

# The 3D models need the structure file; the 1D models do not. Gelman et al.
# report that 3D relative position embeddings help METL-Global but "do not make
# a difference for METL-Local models", so the 1D model is the one to prefer --
# equally accurate and with no staging dependency on an OSG node.
NEEDS_PDB = {"ft-3d", "ft-3d-64"}

# Where metl-pretrained caches its downloads. Pointed at the project cache so
# the weights ride along with the ESM-2 weights in one staged directory.
CACHE_DIR = ROOT / "cache"


def available() -> bool:
    """Whether metl-pretrained is importable in this environment.

    Checked rather than assumed because osg/README.md records that the full
    metl repo needs pytorch-lightning plus five more dependencies absent from
    the container image. metl-pretrained is packaged "with minimal
    dependencies", so it may be installable where the full repo is not -- and
    if it is, on-node brightness scoring becomes viable instead of being
    deferred to a local post-hoc pass.
    """
    try:
        import metl  # noqa: F401
    except ImportError:
        return False
    return True


def load(which: str = "ft-1d"):
    """Load a finetuned target model and its encoder by short name.

    `which` is a key of METL_TARGET_UUIDS: "ft-1d" and "ft-3d" are finetuned on
    80% of the GFP DMS, "ft-1d-64" and "ft-3d-64" on only 64 examples and are
    the exact models behind the manuscript's design experiment.
    """
    if which not in METL_TARGET_UUIDS:
        raise ValueError(f"unknown METL target model '{which}', "
                         f"expected one of {sorted(METL_TARGET_UUIDS)}")
    if which in _CACHE:
        return _CACHE[which]
    import torch
    import metl
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # metl-pretrained downloads through torch.hub, which defaults to
    # ~/.cache/torch. Point it at the project cache instead: that is the one
    # directory every run shares and every OSG job already stages, so the
    # 9.4 MB checkpoint rides along with the ESM-2 weights rather than being
    # re-downloaded on each node -- or failing on a node with no network.
    torch.hub.set_dir(str(CACHE_DIR / "torch_hub"))
    model, encoder = metl.get_from_uuid(uuid=METL_TARGET_UUIDS[which])
    model = model.eval().to(device())
    _CACHE[which] = (model, encoder)
    return model, encoder


def predict(sequences: list[str], wt: str, which: str = "ft-1d",
            batch_size: int = 64, pdb_path=None) -> np.ndarray:
    """Predicted functional score, one value per sequence.

    Unlike the source model in metl.py -- whose head emits 55 Rosetta
    attributes, so "METL predicts brightness" was only ever true through a
    ridge fitted on top of it -- these output the DMS functional score itself.

    The variant notation is 0-indexed, which is exactly what seq_to_variant
    already emits, and the wild-type sequence shipped with metl-pretrained is
    byte-identical to data/avgfp_wt.txt, so no realignment is needed.

    IMPORTANT, and it must be disclosed wherever these numbers appear: the
    "ft-1d" and "ft-3d" models were trained on 80% of the same Sarkisyan GFP
    data this project's splits are carved from. They have almost certainly seen
    rows in avgfp_val.csv and avgfp_test.csv, so no clean held-out Spearman can
    be quoted for them from these splits. They are valid as a second opinion on
    GENERATED sequences, which are novel, and invalid as a headline accuracy
    claim.
    """
    import torch
    model, encoder = load(which)
    variants = [seq_to_variant(s, wt) for s in sequences]
    # A sequence identical to the wild type has no substitutions; METL spells
    # that case "_wt" rather than the empty string.
    variants = [v if v else "_wt" for v in variants]
    extra = {}
    if which in NEEDS_PDB:
        if pdb_path is None:
            raise ValueError(f"METL target model '{which}' uses 3D relative "
                             f"position embeddings and needs the 1gfl_cm.pdb "
                             f"structure; pass pdb_path, or use 'ft-1d' which "
                             f"the paper reports as equally accurate for "
                             f"METL-Local.")
        extra["pdb_fn"] = str(pdb_path)
    out = []
    with torch.no_grad():
        for start in range(0, len(variants), batch_size):
            chunk = variants[start:start + batch_size]
            encoded = encoder.encode_variants(wt, chunk)
            y = model(torch.tensor(encoded).to(device()), **extra)
            out.append(y.detach().cpu().numpy().astype(np.float32).reshape(-1))
    return np.concatenate(out)


def valid_sequences(sequences: list[str], wt: str) -> np.ndarray:
    """Mask of sequences METL can score: right length, canonical residues.

    METL's encoder indexes into a fixed amino-acid alphabet and assumes the
    design is an aligned variant of the wild type, so a sequence of the wrong
    length or carrying a non-canonical residue would raise rather than return a
    low score. Filtering first keeps a decoding artifact from stopping a whole
    evaluation pass.
    """
    allowed = set(AMINO_ACIDS)
    return np.array([len(s) == len(wt) and set(s) <= allowed
                     for s in sequences], dtype=bool)


def add_arguments(parser):
    """Flags selecting which METL judge scores the generated designs."""
    group = parser.add_argument_group("METL oracle")
    group.add_argument("--metl-target", default=None,
                       choices=sorted(METL_TARGET_UUIDS),
                       help="Score designs with a released finetuned METL GFP "
                            "model, which outputs the functional score "
                            "directly (default: none, use the ridge fitted on "
                            "the source model's pooled representation). "
                            "'ft-1d' needs no PDB file. Disclose that these "
                            "trained on 80%% of the same DMS; 'ft-1d-64' and "
                            "'ft-3d-64' are the 64-example design-experiment "
                            "models and do not have that overlap problem to "
                            "anything like the same degree.")
    group.add_argument("--metl-stability", action="store_true",
                       help="Report METL's predicted total_score (stability) "
                            "as the orthogonal fidelity axis, replacing "
                            "amino-acid composition total variation. It has "
                            "Spearman -0.69 with experimental brightness, so "
                            "'does guidance toward brightness destroy "
                            "predicted stability?' is a real Pareto question "
                            "rather than a proxy one, and it comes free with "
                            "the same forward pass -- no Rosetta run, which "
                            "would be 135-215 s per variant.")
    return parser
