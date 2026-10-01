#!/usr/bin/env python3
"""Download one ESM-2 model into a cache directory, for staging to OSG.

The worker node runs offline: HF_HUB_OFFLINE is set in the image so a missing
weight fails loudly instead of each of a hundred jobs pulling the same files
from HuggingFace. That means the cache has to be populated once, here, and
shipped with the job.

Run it on the access point inside the image, so the files land in the layout
the same transformers version will look for:

  apptainer exec dgm-gpu-v1.sif \\
      env PYTHONPATH=$PWD/src python3 osg/bin/fetch_esm_weights.py --model esm2_8m

Or run it anywhere the environment already works, then stage with
osg/bin/stage.sh. Sizes on disk: esm2_8m 30 MB, esm2_35m 130 MB,
esm2_150m 568 MB, esm2_650m 2.5 GB, esm2_3b 11 GB. Above about a gigabyte,
stage through OSDF rather than transfer_input_files.
"""
import argparse
from pathlib import Path

from dgm.common.esm_models import get_model, list_models


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="esm2_8m",
                        help="ESM-2 model tag (default: esm2_8m)")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Where to write the cache. Default: the project's "
                             "own cache directory.")
    parser.add_argument("--list-models", action="store_true",
                        help="Print the known model tags and exit.")
    args = parser.parse_args()

    if args.list_models:
        print(list_models())
        return

    # Imported here so --list-models works without torch present.
    from transformers import AutoTokenizer, EsmForMaskedLM

    from dgm.common.paths import cache_dir as project_cache

    info = get_model(args.model)
    cache = (args.cache_dir or project_cache(create=True)).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)

    print(f"model     : {args.model}  ({info['hf_id']})")
    print(f"cache dir : {cache}")
    AutoTokenizer.from_pretrained(info["hf_id"], cache_dir=cache)
    EsmForMaskedLM.from_pretrained(info["hf_id"], cache_dir=cache,
                                   use_safetensors=True)

    subdir = cache / ("models--" + info["hf_id"].replace("/", "--"))
    total = sum(f.stat().st_size for f in subdir.rglob("*") if f.is_file())
    print(f"fetched   : {subdir.name}  ({total / 1e6:.0f} MB)")
    if total > 1e9:
        print("  this is over 1 GB; stage it through OSDF rather than "
              "transfer_input_files")


if __name__ == "__main__":
    main()
