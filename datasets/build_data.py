"""Build and partition the CDM-1 datasets into train, val, and test splits."""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cdm.data.curated_corpus import build_curated_decision_corpus, save_dataset_to_disk
from cdm.data.dataset_loader import split_dataset


def main():
    print("Building curated multi-domain decision corpus...")
    examples = build_curated_decision_corpus()
    print(f"Total curated examples generated: {len(examples)}")

    train_data, val_data, test_data = split_dataset(
        examples, split_type="temporal", train_ratio=0.7, val_ratio=0.15
    )

    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, "train.jsonl")
    val_path = os.path.join(base_dir, "val.jsonl")
    test_path = os.path.join(base_dir, "test.jsonl")

    save_dataset_to_disk(train_data, train_path)
    save_dataset_to_disk(val_data, val_path)
    save_dataset_to_disk(test_data, test_path)

    print(f"Saved Train set ({len(train_data)} examples) to {train_path}")
    print(f"Saved Val set ({len(val_data)} examples) to {val_path}")
    print(f"Saved Test set ({len(test_data)} examples) to {test_path}")


if __name__ == "__main__":
    main()
