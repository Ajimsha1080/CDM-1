"""STAGE 1: Pretrained Language Encoder Baseline Training.
Extracts semantic text representations and trains text projection head on decision tasks.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from cdm.config.model_config import CDMModelConfig
from cdm.models.language_encoder import LanguageEncoder
from cdm.data.dataset_loader import DecisionDataset, collate_decision_batch
from cdm.data.curated_corpus import load_dataset_from_disk
from cdm.data.preprocessor import DecisionPreprocessor
from cdm.utils.training_utils import set_seed, EarlyStopping


def train_language_encoder(epochs: int = 5, lr: float = 1e-3, batch_size: int = 8):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== STAGE 1: Training Language Encoder Projection on {device} ===")

    train_data = load_dataset_from_disk("datasets/train.jsonl")
    val_data = load_dataset_from_disk("datasets/val.jsonl")

    preprocessor = DecisionPreprocessor()
    train_ds = DecisionDataset(train_data, preprocessor)
    val_ds = DecisionDataset(val_data, preprocessor)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_decision_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_decision_batch)

    config = CDMModelConfig().language_encoder
    encoder = LanguageEncoder(config).to(device)

    # Classification probe for baseline evaluation
    classifier = nn.Linear(config.projection_dim, 2).to(device)
    optimizer = torch.optim.AdamW(list(encoder.projection.parameters()) + list(classifier.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    for epoch in range(epochs):
        encoder.train()
        classifier.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"]["binary_target"].to(device)

            optimizer.zero_grad()
            emb = encoder(input_ids, attention_mask)
            logits = classifier(emb)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        encoder.eval()
        classifier.eval()
        val_correct = 0
        total_val = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["targets"]["binary_target"].to(device)
                emb = encoder(input_ids, attention_mask)
                preds = torch.argmax(classifier(emb), dim=-1)
                val_correct += (preds == targets).sum().item()
                total_val += targets.size(0)

        val_acc = val_correct / max(total_val, 1)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Val Accuracy: {val_acc:.4f}")

    print("Stage 1 language encoder baseline established successfully.")
    return encoder


if __name__ == "__main__":
    train_language_encoder()
