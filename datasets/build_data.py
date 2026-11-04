"""Build, partition, and validate CDM-1 datasets with split manifests and leakage checks."""

import os
import sys
import json
import hashlib
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cdm.data.curated_corpus import build_curated_decision_corpus, save_dataset_to_disk
from cdm.data.dataset_loader import split_dataset


def compute_file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("Building curated multi-domain decision corpus...")
    examples = build_curated_decision_corpus()
    print(f"Total deduplicated examples generated: {len(examples)}")

    train_data, val_data, test_data = split_dataset(
        examples, split_type="group_temporal", train_ratio=0.70, val_ratio=0.15
    )

    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, "train.jsonl")
    val_path = os.path.join(base_dir, "val.jsonl")
    test_path = os.path.join(base_dir, "test.jsonl")

    save_dataset_to_disk(train_data, train_path)
    save_dataset_to_disk(val_data, val_path)
    save_dataset_to_disk(test_data, test_path)

    # Audit for overlap between splits
    train_ids = set(e.example_id for e in train_data)
    val_ids = set(e.example_id for e in val_data)
    test_ids = set(e.example_id for e in test_data)

    overlap_train_val = train_ids.intersection(val_ids)
    overlap_train_test = train_ids.intersection(test_ids)
    overlap_val_test = val_ids.intersection(test_ids)

    assert len(overlap_train_val) == 0, f"Leakage detected between train and val: {overlap_train_val}"
    assert len(overlap_train_test) == 0, f"Leakage detected between train and test: {overlap_train_test}"
    assert len(overlap_val_test) == 0, f"Leakage detected between val and test: {overlap_val_test}"

    # Build manifest
    manifest = {
        "dataset_version": "1.0.0",
        "total_examples": len(examples),
        "split_counts": {
            "train": len(train_data),
            "validation": len(val_data),
            "test": len(test_data),
        },
        "file_checksums": {
            "train.jsonl": compute_file_sha256(train_path),
            "val.jsonl": compute_file_sha256(val_path),
            "test.jsonl": compute_file_sha256(test_path),
        },
        "provenance_breakdown": {
            "verified_real_expert": sum(1 for e in examples if not e.label_provenance.synthetic),
            "curated_synthetic": sum(1 for e in examples if e.label_provenance.synthetic),
        },
        "domain_distribution": dict(Counter(e.domain for e in examples)),
        "binary_label_distribution": dict(Counter(e.target_decision.binary_decision for e in examples)),
        "priority_label_distribution": dict(Counter(e.target_decision.priority_level for e in examples)),
        "risk_label_distribution": dict(Counter(e.target_decision.risk_level for e in examples)),
        "leakage_audit_passed": True,
    }

    manifest_path = os.path.join(base_dir, "split_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest exported to {manifest_path}")
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    print(f"Zero-leakage audit verified: {manifest['leakage_audit_passed']}")


if __name__ == "__main__":
    main()
