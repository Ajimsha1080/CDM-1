"""STAGE 10: Model Export and Checkpoint Validation for CDM-1.
Exports trained model weights in SafeTensors format, serializes tokenizer files,
writes feature schema, and verifies standalone reloading.
"""

import os
import sys
import json
import shutil
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoTokenizer
from cdm.models.cdm_model import CDMModel
from cdm.config.feature_schema import FeatureSchema


def export_model_package(
    source_checkpoint_dir: str = "experiments/checkpoints/cdm-1-general",
    export_dir: str = "experiments/release/cdm-1-general",
):
    print(f"=== STAGE 10: Exporting Model Checkpoint to {export_dir} ===")
    os.makedirs(export_dir, exist_ok=True)

    # 1. Load model and save weights & config
    print("Exporting SafeTensors weights and config.json...")
    model = CDMModel.from_pretrained(source_checkpoint_dir)
    model.save_pretrained(export_dir)

    # 2. Export tokenizer
    print("Exporting tokenizer vocabulary and configs...")
    tokenizer = AutoTokenizer.from_pretrained(model.config.language_encoder.model_name_or_path)
    tokenizer.save_pretrained(export_dir)

    # 3. Export feature schema
    schema = FeatureSchema.get_default_schema()
    with open(os.path.join(export_dir, "feature_schema.json"), "w", encoding="utf-8") as f:
        f.write(schema.model_dump_json(indent=2))

    # 4. Verify round-trip load
    print("Verifying standalone reload from exported package...")
    reloaded_model = CDMModel.from_pretrained(export_dir)
    params = reloaded_model.count_parameters()
    print(f"Reload successful! Verified parameter counts: {params}")

    print(f"Export completed. Ready for local inference and SDK deployment.")


if __name__ == "__main__":
    export_model_package()
