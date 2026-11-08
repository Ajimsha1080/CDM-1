#!/usr/bin/env python3
"""One-time offline cache utility for ModernBERT backbone.
Pre-downloads and caches the pinned ModernBERT backbone and tokenizer into the local
Hugging Face cache directory so CDM-1 can operate completely offline with local_files_only=True.
"""

import sys
import os
from pathlib import Path

PINNED_MODEL_NAME = "answerdotai/ModernBERT-base"
PINNED_REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"


def cache_backbone(model_name: str = PINNED_MODEL_NAME, revision: str = PINNED_REVISION) -> None:
    print(f"[*] Pre-caching ModernBERT backbone '{model_name}' (revision: {revision[:8]})...")
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        print("[-] Error: 'transformers' package is required. Run 'pip install transformers'.", file=sys.stderr)
        sys.exit(1)

    print("    [1/2] Fetching tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    print(f"          Tokenizer cached ({len(tokenizer)} tokens).")

    print("    [2/2] Fetching model weights...")
    model = AutoModel.from_pretrained(model_name, revision=revision)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"          Model cached successfully ({total_params:,} parameters).")
    print(f"\n[+] SUCCESS: ModernBERT backbone cached. CDM-1 can now start with local_files_only=True in offline environments.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pre-cache ModernBERT backbone for offline CDM-1 operation.")
    parser.add_argument("--model", default=PINNED_MODEL_NAME, help="Hugging Face model ID")
    parser.add_argument("--revision", default=PINNED_REVISION, help="Git commit hash revision")
    args = parser.parse_args()

    cache_backbone(model_name=args.model, revision=args.revision)


if __name__ == "__main__":
    main()
