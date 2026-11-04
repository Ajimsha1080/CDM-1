"""STAGE 2: Structured Tabular Feature Encoder Training.
Trains the tabular neural network on continuous variables, missingness flags, and categorical embeddings.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from cdm.config.model_config import CDMModelConfig
from cdm.models.structured_encoder import StructuredFeatureEncoder
from cdm.data.dataset_loader import DecisionDataset, collate_decision_batch
from cdm.data.curated_corpus import load_dataset_from_disk
from cdm.data.preprocessor import DecisionPreprocessor
from cdm.utils.training_utils import set_seed


def train_structured_encoder(epochs: int = 10, lr: float = 1e-3, batch_size: int = 16):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== STAGE 2: Training Structured Feature Encoder on {device} ===")

    train_data = load_dataset_from_disk("datasets/train.jsonl")
    val_data = load_dataset_from_disk("datasets/val.jsonl")

    preprocessor = DecisionPreprocessor()
    train_ds = DecisionDataset(train_data, preprocessor)
    val_ds = DecisionDataset(val_data, preprocessor)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_decision_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_decision_batch)

    cfg = CDMModelConfig().structured_encoder
    encoder = StructuredFeatureEncoder(cfg).to(device)
    probe = nn.Linear(cfg.projection_dim, 4).to(device)  # probe priority classes

    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(probe.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        encoder.train()
        probe.train()
        total_loss = 0.0

        for batch in train_loader:
            num_vals = batch["numeric_values"].to(device)
            missing = batch["missing_indicators"].to(device)
            cat_idx = {k: v.to(device) for k, v in batch["categorical_indices"].items()}
            targets = batch["targets"]["priority_target"].to(device)

            optimizer.zero_grad()
            emb = encoder(num_vals, missing, cat_idx)
            logits = probe(emb)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Structured Loss: {avg_loss:.4f}")

    print("Stage 2 structured feature encoder trained successfully.")
    return encoder


if __name__ == "__main__":
    train_structured_encoder()
