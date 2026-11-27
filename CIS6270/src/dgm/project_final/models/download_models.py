"""
Cache-aware verifier for existing `fair-esm` PyTorch Hub checkpoints (`*.pt`)
and downloader/verifier for Track A DPLM checkpoints (`Synthyra/DPLM-150M`, `650M`).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

from dgm.project_final.config import DATA_DIR, ensure_project_dirs
from dgm.project_final.eval.evaluate import compute_esm2_pseudo_perplexity

MANIFEST_PATH: Path = DATA_DIR / "models" / "manifest.json"

LOCAL_TORCH_HUB_ESM: Tuple[str, ...] = (
    "esm2_t30_150M_UR50D",
    "esm2_t33_650M_UR50D",
)

DPLM_REGISTRY: Dict[str, str] = {
    "dplm_150m": "Synthyra/DPLM-150M",
    "dplm_650m": "Synthyra/DPLM-650M",
}


def verify_local_torch_hub_esm(device: torch.device) -> Dict[str, Any]:
    """Verify existing `~/.cache/torch/hub/checkpoints/esm2_*.pt` models on GB10 GPU."""
    torch_ckpts = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch")) / "hub" / "checkpoints"
    probe_seq = "EEEKDKEKEDEHRQHQEHQEHDENNDRRPQHTGPVFRPSPFVRVFFVPFPGVRRNEQEHHDEE"
    results: Dict[str, Any] = {}

    for stem in LOCAL_TORCH_HUB_ESM:
        ckpt_file = torch_ckpts / f"{stem}.pt"
        if not ckpt_file.exists():
            results[stem] = {"status": "missing", "expected_path": str(ckpt_file)}
            continue

        t0 = time.perf_counter()
        ppl = compute_esm2_pseudo_perplexity([probe_seq], model_name=stem, device=device)[0]
        dt = time.perf_counter() - t0
        print(f"[TorchHub ESM] {stem} -> OK | PPL={ppl:.2f} | path={ckpt_file} ({dt:.1f}s)")
        results[stem] = {
            "status": "verified_from_torch_hub_cache",
            "path": str(ckpt_file),
            "probe_ppl": ppl,
        }
    return results


def resolve_and_verify_dplm(
    key: str, repo_id: str, device: torch.device, cache_only: bool = False
) -> Dict[str, Any]:
    """Verify `Synthyra/DPLM-*` from `~/.cache/huggingface` or download if missing."""
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

    t0 = time.perf_counter()
    try:
        cached_path = snapshot_download(repo_id=repo_id, local_files_only=True)
        source_tag = "reused_from_hf_cache"
        print(f"[{key}] Found in ~/.cache/huggingface -> {cached_path}")
    except Exception:
        if cache_only:
            print(f"[{key}] Not in ~/.cache/huggingface yet (run without --cache-only to fetch).")
            return {"repo_id": repo_id, "status": "not_downloaded_yet"}
        print(f"[{key}] Downloading `{repo_id}` into ~/.cache/huggingface ...")
        cached_path = snapshot_download(
            repo_id=repo_id, ignore_patterns=["*.msgpack", "*.h5", "*.ot"]
        )
        source_tag = "downloaded_to_hf_cache"

    cfg = AutoConfig.from_pretrained(cached_path, trust_remote_code=True, local_files_only=True)
    if hasattr(cfg, "attn_backend"):
        cfg.attn_backend = "sdpa"

    # Keep FP32-resident parameters and run BF16 autocast on CUDA per DPLM's contract
    model = AutoModelForMaskedLM.from_pretrained(
        cached_path, config=cfg, trust_remote_code=True, torch_dtype=torch.float32, local_files_only=True
    ).to(device).eval()

    try:
        tokenizer = AutoTokenizer.from_pretrained(cached_path, trust_remote_code=True, local_files_only=True)
    except Exception:
        tokenizer = model.tokenizer

    mask_tok = tokenizer.mask_token or "<mask>"
    enc = tokenizer(f"EKEH{mask_tok}GSP{mask_tok}YPQHKDE", return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

    with torch.no_grad():
        if device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=input_ids, attention_mask=attn_mask)
        else:
            out = model(input_ids=input_ids, attention_mask=attn_mask)
        logits = out.logits if hasattr(out, "logits") else out[0]
        if torch.isnan(logits).any():
            raise RuntimeError(f"NaN logits detected for {repo_id}")

    n_params = sum(p.numel() for p in model.parameters())
    dt = time.perf_counter() - t0
    print(f"[{key}] OK ({source_tag}) | params={n_params / 1e6:.1f}M | logits={tuple(logits.shape)} ({dt:.1f}s)")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "repo_id": repo_id,
        "cached_path": str(cached_path),
        "source": source_tag,
        "num_parameters": n_params,
        "status": "verified",
    }


def check_bio_packages() -> Dict[str, str]:
    """Check status of optional Python biophysical evaluators (`metapredict`, `localcider`)."""
    probe = "EEEKDKEKEDEHRQHQEHQEHDENNDRRPQHTGPVFRPSPFVRVFFVPFPGVRRNEQEHHDEE"
    out: Dict[str, str] = {}
    try:
        import metapredict as meta

        dis = meta.predict_disorder(probe)
        out["metapredict"] = f"OK (mean_disorder={sum(dis)/len(dis):.3f})"
    except ImportError:
        out["metapredict"] = "NOT_INSTALLED (optional: uv pip install --no-deps metapredict)"

    try:
        from localcider.sequenceParameters import SequenceParameters

        out["localcider"] = f"OK (kappa={SequenceParameters(probe).get_kappa():.3f})"
    except ImportError:
        out["localcider"] = "NOT_INSTALLED (optional: uv pip install --no-deps localcider)"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify cached `.pt` ESM-2 and download Track A DPLM.")
    parser.add_argument(
        "--dplm",
        nargs="*",
        choices=list(DPLM_REGISTRY.keys()),
        default=["dplm_150m"],
        help="DPLM checkpoints to verify/download (`dplm_150m` default; add `dplm_650m` for scale test).",
    )
    parser.add_argument("--cache-only", action="store_true", help="Verify local caches without downloading.")
    args = parser.parse_args()

    ensure_project_dirs()
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== 1. Verifying Existing PyTorch Hub ESM-2 Checkpoints (*.pt) ===")
    esm_status = verify_local_torch_hub_esm(device)

    print("\n=== 2. Verifying / Downloading Track A DPLM Checkpoints ===")
    dplm_status: Dict[str, Any] = {}
    for key in args.dplm:
        dplm_status[key] = resolve_and_verify_dplm(
            key, DPLM_REGISTRY[key], device=device, cache_only=args.cache_only
        )

    print("\n=== 3. Biophysical Oracle Packages ===")
    bio_status = check_bio_packages()
    for k, v in bio_status.items():
        print(f"  - {k:14s}: {v}")

    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump({"esm_torch_hub": esm_status, "dplm_hf": dplm_status, "bio_packages": bio_status}, f, indent=2)
    print(f"\nSaved model manifest -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()