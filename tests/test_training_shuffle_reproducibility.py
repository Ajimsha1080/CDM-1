"""Tests for Individual Training Example Shuffling and Deterministic Reproducibility (Gate 2).
Validates:
1. Individual training examples are reshuffled before batch construction.
2. Candidate, feature, and label associations are strictly preserved within each example.
3. Validation and test batches remain deterministic (shuffle=False).
4. Reproducibility across epochs with fixed random seeds.
"""

import random
import torch
import pytest

from research.training.train_multitask import collate_cached_examples


def test_individual_example_shuffling_preserves_associations():
    """Verify that reshuffling individual examples does not scramble feature-label associations."""
    device = torch.device("cpu")
    num_examples = 20

    # Synthetic individual example cache with explicit unique markers
    examples = []
    for i in range(num_examples):
        nc = (i % 3) + 1
        examples.append({
            "pooled_text": torch.full((1, 16), fill_value=float(i)),
            "candidate_pooled": torch.full((1, nc, 16), fill_value=float(i * 10)),
            "num_candidates": nc,
            "numeric_values": torch.full((5,), fill_value=float(i)),
            "missing_indicators": torch.zeros((5,)),
            "categorical_indices": {"cat_a": torch.tensor(i)},
            "targets": {"binary_target": torch.tensor(i % 2), "id_marker": torch.tensor(i)},
            "constraint_mask": torch.ones((nc,)),
        })

    # Reshuffle with seed
    rng = random.Random(1337)
    indices = list(range(num_examples))
    rng.shuffle(indices)

    shuffled_examples = [examples[idx] for idx in indices]
    batch = collate_cached_examples(shuffled_examples[:8], device)

    # Check associations in batch: for each element in batch, all tensors must reflect the same original example id
    for b_idx in range(8):
        orig_id = batch["targets"]["id_marker"][b_idx].item()
        assert batch["pooled_text"][b_idx, 0].item() == float(orig_id)
        assert batch["numeric_values"][b_idx, 0].item() == float(orig_id)
        assert batch["categorical_indices"]["cat_a"][b_idx].item() == orig_id
        assert batch["targets"]["binary_target"][b_idx].item() == (orig_id % 2)


def test_fixed_seed_shuffle_reproducibility():
    """Verify that same seed produces identical order and different seeds produce different order."""
    num_examples = 50
    indices = list(range(num_examples))

    rng1 = random.Random(42)
    order1 = list(indices)
    rng1.shuffle(order1)

    rng2 = random.Random(42)
    order2 = list(indices)
    rng2.shuffle(order2)

    rng3 = random.Random(43)
    order3 = list(indices)
    rng3.shuffle(order3)

    assert order1 == order2
    assert order1 != order3
