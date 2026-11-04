"""Training utility functions: reproducible seeding, early stopping,

metric calculation, and learning rate scheduling for CDM-1.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional


def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class EarlyStopping:
    """Early stopping to halt training when validation metric stops improving."""

    def __init__(self, patience: int = 5, delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, val_score: float) -> bool:
        score = -val_score if self.mode == "min" else val_score

        if self.best_score is None:
            self.best_score = score
            return False

        if score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = score
            self.counter = 0

        return False
