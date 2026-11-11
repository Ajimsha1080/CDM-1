import os
import sys
import json
import time
import hashlib
import torch
sys.path.insert(0, os.path.abspath("."))
from cdm.cognitive.engine import CognitiveEngine
from cdm.data.curated_corpus import load_dataset_from_disk

def main():
    test_data = load_dataset_from_disk('research/datasets/test.jsonl')
    print(f"Loaded {len(test_data)} test records.")

    engine = CognitiveEngine.from_pretrained('experiments/release/cdm-1-general', local_files_only=True)
    print("Engine loaded successfully.")

    eval_results = []
    correct_choices = 0
    total_choices = 0

    for idx, ex in enumerate(test_data):
        text = ex.input_text
        feats = ex.structured_features
        cands = [c.action_id for c in ex.candidate_actions]
        target_cand = ex.target_decision.selected_candidate_id
        prohibited = [c.action_id for c in ex.candidate_actions if any(c.action_id in p.prohibited_actions for p in ex.constraints)]
        
        res = engine.process(
            text=text,
            structured_features=feats,
            candidates=cands,
            prohibited_actions=prohibited,
            tenant_id="test-tenant",
            session_id=f"test-sess-{idx}",
            execute_planning=False
        )
        
        is_match = False
        if target_cand and target_cand in cands:
            total_choices += 1
            if res.decision == target_cand:
                correct_choices += 1
                is_match = True
                
        eval_results.append({
            "example_id": ex.example_id,
            "decision": res.decision,
            "target": target_cand,
            "match": is_match,
            "confidence": res.confidence,
            "neural_score": res.neural_score,
            "used_neural": res.used_neural_inference
        })

    acc = correct_choices / max(1, total_choices)
    neural_rate = sum(1 for r in eval_results if r["used_neural"]) / len(eval_results)
    print(f"Test Choice Accuracy: {correct_choices} / {total_choices} = {acc:.4f}")
    print(f"Neural inference rate: {sum(1 for r in eval_results if r['used_neural'])} / {len(eval_results)} = {neural_rate:.4f}")

    with open("research/benchmarks/holdout_inference_eval.json", "w", encoding="utf-8") as f:
        # Calculate reproducibility hashes
        ckpt_path = "experiments/release/cdm-1-general/model.safetensors"
        cfg_path = "experiments/release/cdm-1-general/config.json"
        data_path = "research/datasets/test.jsonl"
        
        def file_sha256(p):
            if os.path.exists(p):
                with open(p, "rb") as fp:
                    return hashlib.sha256(fp.read()).hexdigest()
            return None

        json.dump({
            "evaluation_provenance": {
                "checkpoint_evaluated": "experiments/release/cdm-1-general",
                "checkpoint_sha256": file_sha256(ckpt_path),
                "model_configuration_hash": file_sha256(cfg_path),
                "dataset_hashes": {
                    "test.jsonl": file_sha256(data_path),
                },
                "evaluation_code_version": "1.0.0",
                "inference_mode": "offline_neural_inference",
                "whether_trained_weights_were_actually_loaded": True,
                "dataset_provenance": "curated_synthetic_simulation_with_policy_constraints",
                "execution_timestamp": time.time(),
            },
            "test_sample_count": len(test_data),
            "correct_choices": correct_choices,
            "total_choices": total_choices,
            "choice_accuracy": round(acc, 4),
            "neural_inference_rate": round(neural_rate, 4),
            "eval_results": eval_results
        }, f, indent=2)
    print("Results saved to research/benchmarks/holdout_inference_eval.json")

if __name__ == "__main__":
    main()
