"""Unit tests for CDM-1 Dataset Ingestion, Preprocessing, and Splitting Pipeline.
"""

import os
import tempfile
import pytest
import torch
from cdm.data.curated_corpus import (
    build_curated_decision_corpus,
    save_dataset_to_disk,
    load_dataset_from_disk,
)
from cdm.data.preprocessor import DecisionPreprocessor
from cdm.data.dataset_loader import DecisionDataset, collate_decision_batch, split_dataset
from cdm.config.feature_schema import FeatureSchema


def test_curated_corpus_generation_and_serialization():
    examples = build_curated_decision_corpus()
    assert len(examples) > 100

    # Test serialization to disk
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "corpus.jsonl")
        save_dataset_to_disk(examples, file_path)
        assert os.path.exists(file_path)

        loaded = load_dataset_from_disk(file_path)
        assert len(loaded) == len(examples)
        assert loaded[0].example_id == examples[0].example_id
        assert loaded[0].domain == examples[0].domain


def test_dataset_preprocessing_and_collation():
    examples = build_curated_decision_corpus()[:8]
    preprocessor = DecisionPreprocessor()
    dataset = DecisionDataset(examples, preprocessor)
    assert len(dataset) == 8

    # Collate batch
    batch = [dataset[i] for i in range(4)]
    collated = collate_decision_batch(batch)

    assert "input_ids" in collated
    assert collated["input_ids"].shape[0] == 4
    assert "numeric_values" in collated
    assert collated["numeric_values"].shape == (4, 6)
    assert "targets" in collated
    assert "binary_target" in collated["targets"]
    assert "priority_target" in collated["targets"]
    assert "score_target" in collated["targets"]


def test_temporal_dataset_splitting():
    examples = build_curated_decision_corpus()
    train_ex, val_ex, test_ex = split_dataset(examples, split_type="temporal", train_ratio=0.7, val_ratio=0.15)

    assert len(train_ex) > 0
    assert len(val_ex) > 0
    assert len(test_ex) > 0
    assert len(train_ex) + len(val_ex) + len(test_ex) == len(examples)

    # Verify no temporal leakage: train max timestamp <= test min timestamp
    train_max_time = max(e.temporal_timestamp for e in train_ex if e.temporal_timestamp)
    test_min_time = min(e.temporal_timestamp for e in test_ex if e.temporal_timestamp)
    assert train_max_time <= test_min_time
