"""STAGE 6: Predictive Network Training.
Trains binary outcome, multiclass outcome, resolution time regression, and risk estimation
heads using actual verified outcome data.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from cdm.config.model_config import CDMModelConfig
from cdm.models.cdm_model import CDMModel
from cdm.data.dataset_loader import DecisionDataset, collate_decision_batch
from cdm.data.curated_corpus import load_dataset_from_disk
from cdm.data.preprocessor import DecisionPreprocessor
from cdm.utils.training_utils import set_seed


def train_predictive_network(epochs: int = 10, lr: float = 1e-3, batch_size: int = 16):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== STAGE 6: Training Predictive Network on {device} ===")

    train_data = load_dataset_from_disk("datasets/train.jsonl")
    val_data = load_dataset_from_disk("datasets/val.jsonl")

    preprocessor = DecisionPreprocessor()
    train_ds = DecisionDataset(train_data, preprocessor)
    val_ds = DecisionDataset(val_data, preprocessor)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_decision_batch)

    model = CDMModel().to(device)
    model.language_encoder.freeze_backbone()

    optimizer = torch.optim.AdamW(
        list(model.structured_encoder.parameters())
        + list(model.fusion.parameters())
        + list(model.predictive_network.parameters()),
        lr=lr,
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            num_vals = batch["numeric_values"].to(device)
            missing = batch["missing_indicators"].to(device)
            cat_idx = {k: v.to(device) for k, v in batch["categorical_indices"].items()}
            targets = {k: v.to(device) for k, v in batch["targets"].items()}

            cand_ids = batch.get("candidate_input_ids")
            cand_att = batch.get("candidate_attention_mask")
            cand_masks = batch.get("candidate_masks")
            if cand_ids is not None:
                cand_ids = cand_ids.to(device)
            if cand_att is not None:
                cand_att = cand_att.to(device)
            if cand_masks is not None:
                cand_masks = cand_masks.to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                numeric_values=num_vals,
                missing_indicators=missing,
                categorical_indices=cat_idx,
                candidate_input_ids=cand_ids,
                candidate_attention_mask=cand_att,
                candidate_masks=cand_masks,
            )
            loss, loss_dict = model.compute_multitask_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Predictive Loss: {avg_loss:.4f}")

    print("Stage 6 predictive network trained successfully.")
    return model


if __name__ == "__main__":
    train_predictive_network()
