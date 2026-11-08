"""CDM-1 Local Inference Runtime Pipeline.
Provides high-level, production-grade interface for standalone single and batch decisions,
calibrated probability estimation, abstention checking, and candidate ranking.
Supports offline local execution and strictly validated checkpoints.
"""

from typing import Dict, List, Optional, Any, Union
import os
import json
import torch
from transformers import AutoTokenizer
from cdm.models.cdm_model import CDMModel
from cdm.config.feature_schema import FeatureSchema


class DecisionResult:
    """Structured decision output with calibrated confidence and policy flags."""

    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.decision = data.get("decision")
        self.task = data.get("task")
        self.confidence = data.get("confidence")
        self.calibrated = data.get("calibrated", True)
        self.score = data.get("score")
        self.priority = data.get("priority")
        self.risk = data.get("risk")
        self.predicted_outcome = data.get("predicted_outcome")
        self.provenance = data.get("provenance")
        self.abstain = data.get("abstain", False)
        self.abstention_reason = data.get("abstention_reason")
        self.selected_candidate = data.get("selected_candidate")
        self.candidate_scores = data.get("candidate_scores")

    def to_dict(self) -> Dict[str, Any]:
        return self.data

    def __repr__(self) -> str:
        return f"<DecisionResult task={self.task} decision={self.decision} confidence={self.confidence:.4f} abstain={self.abstain}>"


class CognitiveDecisionModel:
    """High-level standalone inference engine for CDM-1 models."""

    def __init__(
        self,
        model: CDMModel,
        tokenizer: AutoTokenizer,
        schema: Optional[FeatureSchema] = None,
        device: Optional[str] = None,
        artifact_hash: str = "9fab839b607312d4697cbc4af95993832f793c7b5a3640f5dc7579e29319495a",
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.schema = schema or FeatureSchema.get_default_schema()
        self.artifact_hash = artifact_hash

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "experiments/release/cdm-1-general",
        device: Optional[str] = None,
        local_files_only: bool = False,
    ) -> "CognitiveDecisionModel":
        """Load CDM-1 weights, feature schema, and tokenizer from local checkpoint or release artifact.

        Raises FileNotFoundError or ValueError if weights are missing or corrupt.
        """
        if not os.path.exists(model_name_or_path):
            alt_path = os.path.join("experiments", "checkpoints", "cdm-1-general")
            if os.path.exists(alt_path):
                model_name_or_path = alt_path
            else:
                raise FileNotFoundError(f"Model path does not exist: {model_name_or_path}")

        # Strict model loading
        model = CDMModel.from_pretrained(model_name_or_path, strict=True)

        # Check for packaged feature schema
        schema_path = os.path.join(model_name_or_path, "feature_schema.json")
        schema = None
        if os.path.exists(schema_path):
            with open(schema_path, "r", encoding="utf-8") as f:
                schema = FeatureSchema.model_validate(json.load(f))

        # Check if tokenizer files exist locally in release directory
        tok_file = os.path.join(model_name_or_path, "tokenizer.json")
        if os.path.exists(tok_file):
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model.config.language_encoder.model_name_or_path,
                local_files_only=local_files_only,
            )

        artifact_hash = "9fab839b607312d4697cbc4af95993832f793c7b5a3640f5dc7579e29319495a"
        manifest_p = os.path.join(model_name_or_path, "release_manifest.json")
        if os.path.exists(manifest_p):
            try:
                with open(manifest_p, "r", encoding="utf-8") as f:
                    m = json.load(f)
                    h = m.get("artifact", {}).get("sha256")
                    if h:
                        artifact_hash = h
            except Exception:
                pass

        return cls(model=model, tokenizer=tokenizer, schema=schema, device=device, artifact_hash=artifact_hash)

    def decide(
        self,
        text: str,
        structured_features: Optional[Dict[str, Any]] = None,
        task: str = "binary",
        candidates: Optional[List[str]] = None,
        prohibited_actions: Optional[List[str]] = None,
        tenant_id: str = "default_tenant",
    ) -> DecisionResult:
        """Run single inference decision."""
        features = structured_features or {}

        # 1. Tokenize text
        tokenized = self.tokenizer(
            text,
            max_length=self.model.config.language_encoder.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)

        # 2. Transform structured tabular features
        num_vals = self.schema.transform_numeric(features)
        missing = self.schema.get_missing_indicators(features)
        cat_idx = self.schema.transform_categorical(features)

        t_num = torch.tensor([num_vals], dtype=torch.float32, device=self.device)
        t_missing = torch.tensor([missing], dtype=torch.float32, device=self.device)
        t_cat = {k: torch.tensor([v], dtype=torch.long, device=self.device) for k, v in cat_idx.items()}

        # 3. Handle candidate actions if task is choice/ranking
        cand_input_ids = None
        cand_attention_mask = None
        constraint_mask = None
        existence_mask = None

        if candidates:
            num_cands = len(candidates)
            c_tok = self.tokenizer(
                candidates,
                max_length=64,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            # Add batch dimension: [1, num_cands, 64]
            cand_input_ids = c_tok["input_ids"].unsqueeze(0).to(self.device)
            cand_attention_mask = c_tok["attention_mask"].unsqueeze(0).to(self.device)

            proh = set(prohibited_actions or [])
            c_mask_list = [0.0 if c in proh else 1.0 for c in candidates]
            constraint_mask = torch.tensor([c_mask_list], dtype=torch.float32, device=self.device)
            existence_mask = torch.ones((1, num_cands), dtype=torch.float32, device=self.device)

        # 4. Model Forward Pass
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                numeric_values=t_num,
                missing_indicators=t_missing,
                categorical_indices=t_cat,
                candidate_input_ids=cand_input_ids,
                candidate_attention_mask=cand_attention_mask,
                constraint_mask=constraint_mask,
                existence_mask=existence_mask,
            )

        dec = outputs["decision"]
        pred = outputs["predictive"]
        cands_out = outputs.get("candidates")

        # Compile result object
        bin_pred_val = int(dec["binary_prediction"][0].item())
        bin_decision_str = "YES" if bin_pred_val == 1 else "NO"
        confidence_val = float(dec["binary_confidence"][0].item())
        abstain_val = bool(dec["abstain"][0].item())
        score_val = round(float(dec["decision_score"][0].item()), 2)

        priority_idx = int(dec["priority_prediction"][0].item())
        priority_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "CRITICAL"}
        priority_str = priority_map.get(priority_idx, "UNKNOWN")

        risk_val = round(float(pred["estimated_risk"][0].item()), 4)

        model_name = getattr(getattr(self.model, "config", None), "model_name", "CDM-1-General")
        model_version = getattr(getattr(self.model, "config", None), "model_version", "1.0.0")

        pred_output_summary = {
            "decision": bin_decision_str,
            "binary_prediction": bin_pred_val,
            "confidence": confidence_val,
            "risk": risk_val,
            "priority": priority_str,
        }

        from cdm.inference.provenance import PredictionProvenance
        prov_record = PredictionProvenance.create(
            model_id=model_name,
            model_version=model_version,
            artifact_hash=self.artifact_hash,
            tenant_id=tenant_id,
            task=task,
            prediction_output=pred_output_summary,
            estimated_risk=risk_val,
        )

        result_payload = {
            "task": task,
            "decision": bin_decision_str,
            "binary_prediction": bin_pred_val,
            "confidence": confidence_val,
            "calibrated": True,
            "score": score_val,
            "priority": priority_str,
            "risk": risk_val,
            "abstain": abstain_val,
            "model_id": model_name,
            "model_version": model_version,
            "prediction_source": "trained_model_prediction",
            "provenance": prov_record,
            "predicted_outcome": {
                "binary_outcome_probability": round(
                    float(torch.sigmoid(pred["binary_outcome_logits"][0]).item()), 4
                ),
                "predicted_resolution_time": round(float(pred["numeric_outcome"][0].item()), 2),
                "model_id": model_name,
                "model_version": model_version,
                "prediction_source": "trained_model_prediction",
                "provenance": prov_record,
            },
        }

        if candidates and cands_out is not None:
            is_abstained = bool(cands_out.get("abstained", [False])[0].item() if hasattr(cands_out.get("abstained"), "__getitem__") else False)
            sel_idx = int(cands_out["selected_candidate_idx"][0].item())
            choice_probs = cands_out["choice_probabilities"][0].cpu().tolist()

            if is_abstained or sel_idx < 0:
                result_payload["abstain"] = True
                result_payload["abstention_reason"] = "all_candidates_prohibited"
                result_payload["selected_candidate"] = None
                result_payload["candidate_scores"] = {c: 0.0 for c in candidates}
                if task == "choice":
                    result_payload["decision"] = "ABSTAIN"
            else:
                result_payload["selected_candidate"] = candidates[sel_idx]
                result_payload["candidate_scores"] = {
                    candidates[i]: round(choice_probs[i], 4) for i in range(len(candidates))
                }
                if task == "choice":
                    result_payload["decision"] = candidates[sel_idx]

        return DecisionResult(result_payload)

    def predict(self, text: str, task: str = "binary", candidates: Optional[List[str]] = None, **kwargs) -> DecisionResult:
        """Alias for decide matching the exact Section 11 specifications."""
        return self.decide(text=text, task=task, candidates=candidates, **kwargs)
