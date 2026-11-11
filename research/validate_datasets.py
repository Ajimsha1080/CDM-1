import json
import glob
from collections import Counter

splits = {}
paths = [
    'research/datasets/train.jsonl',
    'research/datasets/val.jsonl',
    'research/datasets/test.jsonl',
    'research/datasets/heldout_domain_test.jsonl',
    'research/datasets/ood_test.jsonl'
]

for p in paths:
    with open(p, 'r', encoding='utf-8') as f:
        splits[p] = [json.loads(line) for line in f]

id_map = {}
leakage = []
for split_name, recs in splits.items():
    for r in recs:
        eid = r.get('example_id')
        if eid in id_map:
            leakage.append((eid, id_map[eid], split_name))
        id_map[eid] = split_name

print(f"Total records audited: {len(id_map)}")
print(f"Split leakage (duplicate IDs): {len(leakage)}")

cand_issues = []
constraint_violations = []
provenance_counter = Counter()
domain_counter = Counter()

for split_name, recs in splits.items():
    for r in recs:
        domain = r.get('domain', 'unknown')
        domain_counter[f"{split_name} -> {domain}"] += 1
        target = r.get('target_decision', {})
        cands = r.get('candidate_actions', [])
        cand_ids = [c.get('action_id') if isinstance(c, dict) else str(c) for c in cands]
        target_cand_id = target.get('selected_candidate_id') if isinstance(target, dict) else str(target)
        constraints = r.get('constraints', [])
        prov = r.get('label_provenance', 'unknown')
        prov_type = prov.get('source_type') if isinstance(prov, dict) else str(prov)
        is_synth = prov.get('synthetic', True) if isinstance(prov, dict) else True
        provenance_counter[f"{split_name} -> {prov_type} (synthetic={is_synth})"] += 1
        
        if cand_ids and target_cand_id and target_cand_id not in cand_ids:
            cand_issues.append((split_name, r.get('example_id'), target_cand_id, cand_ids))
            
        for c in constraints:
            c_text = c.get('text', str(c)) if isinstance(c, dict) else str(c)
            if target_cand_id and target_cand_id.lower() in c_text.lower() and any(w in c_text.lower() for w in ['prohibit', 'forbidden', 'not allowed', 'do not']):
                constraint_violations.append((split_name, r.get('example_id'), target_cand_id, c_text))

print(f"Candidate validity issues: {len(cand_issues)}")
print(f"Constraint violations in targets: {len(constraint_violations)}")

print("\n--- Provenance Breakdown ---")
for k, v in sorted(provenance_counter.items()):
    print(f"  {k}: {v}")

print("\n--- Domain Distribution ---")
for k, v in sorted(domain_counter.items()):
    print(f"  {k}: {v}")
