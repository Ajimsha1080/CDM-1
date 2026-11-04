"""STAGE 7: Multi-Task Joint Training Pipeline for CDM-1.
Performs end-to-end multi-task optimization across language, structured features,
predictive outcomes, and multi-head decisions with text embedding caching for high-speed CPU training.
"""

import os
import sys
import json
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from cdm.config.model_config import CDMModelConfig
from cdm.models.cdm_model import CDMModel
from cdm.data.dataset_loader import DecisionDataset, collate_decision_batch
from cdm.data.curated_corpus import load_dataset_from_disk
from cdm.data.preprocessor import DecisionPreprocessor
from cdm.utils.training_utils import set_seed, EarlyStopping


def cache_text_embeddings(model: CDMModel, loader: DataLoader, device: torch.device):
    """Pre-extracts text embeddings through frozen language encoder to eliminate repeated forward passes."""
    model.eval()
    cached_batches = []
    print("Pre-computing text embeddings with frozen backbone...", flush=True)
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            text_emb = model.language_encoder(input_ids, attention_mask)
            cached_batches.append({
                "text_emb": text_emb.detach(),
                "numeric_values": batch["numeric_values"].to(device),
                "missing_indicators": batch["missing_indicators"].to(device),
                "categorical_indices": {k: v.to(device) for k, v in batch["categorical_indices"].items()},
                "targets": {k: v.to(device) for k, v in batch["targets"].items()},
                "constraint_mask": batch["constraint_mask"].to(device) if batch["constraint_mask"] is not None else None,
            })
    return cached_batches


def train_multitask(
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 16,
    checkpoint_dir: str = "experiments/checkpoints/cdm-1-general",
):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== STAGE 7: Joint Multi-Task Optimization on {device} ===", flush=True)

    train_data = load_dataset_from_disk("datasets/train.jsonl")
    val_data = load_dataset_from_disk("datasets/val.jsonl")

    preprocessor = DecisionPreprocessor()
    train_ds = DecisionDataset(train_data, preprocessor)
    val_ds = DecisionDataset(val_data, preprocessor)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_decision_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_decision_batch)

    cfg = CDMModelConfig()
    model = CDMModel(cfg).to(device)
    model.language_encoder.freeze_backbone()

    param_stats = model.count_parameters()
    print(f"Trainable Parameters: {param_stats['trainable_parameters']:,}", flush=True)
    print(f"Frozen Parameters:    {param_stats['frozen_parameters']:,}", flush=True)
    print(f"Total Parameters:     {param_stats['total_parameters']:,}", flush=True)

    # Precompute frozen text embeddings
    cached_train = cache_text_embeddings(model, train_loader, device)
    cached_val = cache_text_embeddings(model, val_loader, device)
    print(f"Cached {len(cached_train)} train batches and {len(cached_val)} val batches.", flush=True)

    # Train all custom neural components
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    early_stopping = EarlyStopping(patience=5, delta=1e-3, mode="min")

    history = []
    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in cached_train:
            text_emb = batch["text_emb"]
            num_vals = batch["numeric_values"]
            missing = batch["missing_indicators"]
            cat_idx = batch["categorical_indices"]
            targets = batch["targets"]

            optimizer.zero_grad()

            # Fused representation & head forward passes
            struct_emb = model.structured_encoder(num_vals, missing, cat_idx)
            cognitive_state = model.fusion(text_emb, struct_emb)
            predictive_out = model.predictive_network(cognitive_state)
            decision_out = model.decision_network(cognitive_state)

            outputs = {
                "cognitive_state": cognitive_state,
                "decision": decision_out,
                "predictive": predictive_out,
                "candidates": None,
            }

            loss, loss_dict = model.compute_multitask_loss(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        avg_train_loss = train_loss / len(cached_train)

        # Validation step
        model.eval()
        val_loss = 0.0
        val_correct_bin = 0
        val_total_bin = 0

        with torch.no_grad():
            for batch in cached_val:
                text_emb = batch["text_emb"]
                num_vals = batch["numeric_values"]
                missing = batch["missing_indicators"]
                cat_idx = batch["categorical_indices"]
                targets = batch["targets"]

                struct_emb = model.structured_encoder(num_vals, missing, cat_idx)
                cognitive_state = model.fusion(text_emb, struct_emb)
                predictive_out = model.predictive_network(cognitive_state)
                decision_out = model.decision_network(cognitive_state)

                outputs = {
                    "cognitive_state": cognitive_state,
                    "decision": decision_out,
                    "predictive": predictive_out,
                    "candidates": None,
                }

                v_loss, _ = model.compute_multitask_loss(outputs, targets)
                val_loss += v_loss.item()

                bin_preds = decision_out["binary_prediction"]
                val_correct_bin += (bin_preds == targets["binary_target"]).sum().item()
                val_total_bin += targets["binary_target"].size(0)

        avg_val_loss = val_loss / len(cached_val)
        val_acc = val_correct_bin / max(val_total_bin, 1)

        print(
            f"Epoch {epoch+1:02d}/{epochs:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Bin Acc: {val_acc:.4f}",
            flush=True,
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_binary_accuracy": val_acc,
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"--> Val loss improved to {best_val_loss:.4f}. Saving checkpoint to {checkpoint_dir}", flush=True)
            model.save_pretrained(checkpoint_dir)

        if early_stopping(avg_val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}.", flush=True)
            break

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.1f} seconds. Best Val Loss: {best_val_loss:.4f}", flush=True)

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/training_history.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": "CDM-1",
            "parameters": param_stats,
            "history": history,
            "best_val_loss": best_val_loss,
            "duration_seconds": elapsed,
        }, f, indent=2)

    return model


if __name__ == "__main__":
    train_multitask()
