"""Build, partition, and validate CDM-1 datasets with split manifests, distribution checks, and leakage audits."""

import os
import sys
import json
import hashlib
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cdm.data.curated_corpus import (
    build_curated_decision_corpus,
    save_dataset_to_disk,
)
from cdm.data.dataset_loader import (
    split_dataset,
    validate_split_distribution,
)


def compute_file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=== Rebuilding CDM-1 Datasets with Stratified Group Temporal Splits ===")
    in_domain_examples, heldout_domain_examples = build_curated_decision_corpus()
    print(f"Total in-domain examples: {len(in_domain_examples)}")
    print(f"Total held-out domain examples: {len(heldout_domain_examples)}")

    train_data, val_data, test_data = split_dataset(
        in_domain_examples, split_type="stratified_group_temporal", train_ratio=0.70, val_ratio=0.15
    )

    # Automated split-distribution and leakage validation (fails if coverage is insufficient)
    print("Validating split distributions and auditing for leakage...")
    audit_summary = validate_split_distribution(
        train_data,
        val_data,
        test_data,
        min_class_count=5,
    )
    print("Split validation passed successfully!")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, "train.jsonl")
    val_path = os.path.join(base_dir, "val.jsonl")
    test_path = os.path.join(base_dir, "test.jsonl")
    heldout_path = os.path.join(base_dir, "heldout_domain_test.jsonl")

    save_dataset_to_disk(train_data, train_path)
    save_dataset_to_disk(val_data, val_path)
    save_dataset_to_disk(test_data, test_path)
    save_dataset_to_disk(heldout_domain_examples, heldout_path)

    # Build comprehensive manifest
    manifest = {
        "dataset_version": "2.0.0",
        "total_in_domain_examples": len(in_domain_examples),
        "total_heldout_domain_examples": len(heldout_domain_examples),
        "split_counts": {
            "train": len(train_data),
            "validation": len(val_data),
            "test": len(test_data),
            "heldout_domain_test": len(heldout_domain_examples),
        },
        "file_checksums": {
            "train.jsonl": compute_file_sha256(train_path),
            "val.jsonl": compute_file_sha256(val_path),
            "test.jsonl": compute_file_sha256(test_path),
            "heldout_domain_test.jsonl": compute_file_sha256(heldout_path),
        },
        "provenance_breakdown": {
            "externally_sourced_real": sum(1 for e in in_domain_examples if not e.label_provenance.synthetic),
            "curated_synthetic": sum(1 for e in in_domain_examples if e.label_provenance.synthetic),
        },
        "split_audit": audit_summary,
        "leakage_audit_passed": True,
    }

    manifest_path = os.path.join(base_dir, "split_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest exported to {manifest_path}")
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)} | Heldout: {len(heldout_domain_examples)}")
    for s_name, stats in audit_summary.items():
        print(f"  [{s_name.upper()}] Domains: {stats['domains']} | Binary: {stats['binary_distribution']} | Risks: {stats['risk_distribution']}")


if __name__ == "__main__":
    main()
