import json
from itertools import combinations
import os
from tqdm import tqdm # 引入tqdm来显示进度条

def build_cooccurrence_graph(data_file_path, output_dir):
    all_entities = set()
    all_triplets = []

    with open(data_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Processing documents")):
            if not line.strip():
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed JSON on line {line_num + 1}")
                continue
            entities_in_doc = {
                entity['linked_id'] 
                for entity in data.get('linked_entities', []) 
                if entity and 'linked_id' in entity and entity['linked_id']
            }

            if len(entities_in_doc) < 2:
                continue 

            for entity1, entity2 in combinations(entities_in_doc, 2):
                all_triplets.append((entity1, "co_occurrence", entity2))
            all_entities.update(entities_in_doc)

    print(f"\nProcessing finished.")
    print(f"Total unique entities found: {len(all_entities)}")
    print(f"Total co-occurrence triplets created: {len(all_triplets)}")

    entity_list = sorted(list(all_entities))
    entity_to_id = {name: i for i, name in enumerate(entity_list)}
    relation_to_id = {"co_occurrence": 0}
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Writing output files to: {output_dir}")
    with open(os.path.join(output_dir, 'entity2id.txt'), 'w', encoding='utf-8') as f:
        f.write(f"{len(entity_to_id)}\n")
        for name, eid in entity_to_id.items():
            f.write(f"{name}\t{eid}\n")
    with open(os.path.join(output_dir, 'relation2id.txt'), 'w', encoding='utf-8') as f:
        f.write(f"{len(relation_to_id)}\n")
        for name, rid in relation_to_id.items():
            f.write(f"{name}\t{rid}\n")
    with open(os.path.join(output_dir, 'train2id.txt'), 'w', encoding='utf-8') as f:
        f.write(f"{len(all_triplets)}\n")
        for h, r, t in tqdm(all_triplets, desc="Writing triplets"):
            hid = entity_to_id[h]
            rid = relation_to_id[r]
            tid = entity_to_id[t]
            f.write(f"{hid}\t{tid}\t{rid}\n")

    print(f"Successfully created all files in {output_dir}")

if __name__ == "__main__":
    input_file = "data.json" 
    output_folder = "cooccurrence_kg"
    build_cooccurrence_graph(input_file, output_folder)

