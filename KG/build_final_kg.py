import json
from tqdm import tqdm
import os
from itertools import combinations

def create_kg_files_from_neighbor_data(input_path, output_dir):
    print("--- Step 1: Scanning file to collect all entities and relations ---")
    
    all_entities = set()
    all_relations = set()
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Scanning entities and relations"):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            seed_entities_in_doc = set()
            for entity in item.get('linked_entities', []):
                seed_entities_in_doc.add(entity['linked_id'])
                for neighbor_info in entity.get('neighbors', []):
                    all_entities.add(neighbor_info['neighbor_id'])
                    all_relations.add(neighbor_info['relation_id'])
            
            all_entities.update(seed_entities_in_doc)
            if len(seed_entities_in_doc) >= 2:
                all_relations.add("co_occurrence")

    print(f"Found {len(all_entities)} unique entities and {len(all_relations)} unique relations.")
    print("\n--- Step 2: Creating ID maps for entities and relations ---")
    
    entity_list = sorted(list(all_entities))
    relation_list = sorted(list(all_relations))
    
    entity_to_id = {name: i for i, name in enumerate(entity_list)}
    relation_to_id = {name: i for i, name in enumerate(relation_list)}

    print("ID maps created.")

    print("\n--- Step 3: Generating all triplets from the data ---")
    
    all_triplets = set()
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Generating triplets"):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            seed_entities_in_doc = {
                entity['linked_id'] 
                for entity in item.get('linked_entities', []) 
                if entity.get('link_source') != 'Fallback'
            }
            
            explicit_relations_in_doc = set()
            for entity in item.get('linked_entities', []):
                head_id = entity['linked_id']
                for neighbor_info in entity.get('neighbors', []):
                    rel_id = neighbor_info['relation_id']
                    tail_id = neighbor_info['neighbor_id']
                    
                    if head_id in entity_to_id and rel_id in relation_to_id and tail_id in entity_to_id:
                        all_triplets.add((head_id, rel_id, tail_id))
                        pair = tuple(sorted((head_id, tail_id)))
                        explicit_relations_in_doc.add(pair)
            if len(seed_entities_in_doc) >= 2:
                for e1, e2 in combinations(seed_entities_in_doc, 2):
                    pair = tuple(sorted((e1, e2)))
                    if pair not in explicit_relations_in_doc:
                        all_triplets.add((e1, "co_occurrence", e2))

    print(f"Generated {len(all_triplets)} unique triplets.")
    print("\n--- Step 4: Writing graph files to the output directory ---")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, 'entity2id.txt'), 'w', encoding='utf-8') as f:
        f.write(f"{len(entity_to_id)}\n")
        for name, eid in entity_to_id.items(): f.write(f"{name}\t{eid}\n")
    with open(os.path.join(output_dir, 'relation2id.txt'), 'w', encoding='utf-8') as f:
        f.write(f"{len(relation_to_id)}\n")
        for name, rid in relation_to_id.items(): f.write(f"{name}\t{rid}\n")
    with open(os.path.join(output_dir, 'train2id.txt'), 'w', encoding='utf-8') as f:
        f.write(f"{len(all_triplets)}\n")
        for h, r, t in tqdm(all_triplets, desc="Writing train2id.txt"):
            hid = entity_to_id[h]
            rid = relation_to_id[r]
            tid = entity_to_id[t]
            f.write(f"{hid}\t{tid}\t{rid}\n") 

    print(f"\nAll files successfully created in '{output_dir}'. You can now use this directory for KGE model training.")


if __name__ == "__main__":
    
    INPUT_NEIGHBOR_FILE = "data_with_neighbors.jsonl"
    OUTPUT_KG_FOLDER = "final_enriched_kg"
    create_kg_files_from_neighbor_data(
        input_path=INPUT_NEIGHBOR_FILE,
        output_dir=OUTPUT_KG_FOLDER
    )