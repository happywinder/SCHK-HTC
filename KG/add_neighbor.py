import json
from tqdm import tqdm
import os
from SPARQLWrapper import SPARQLWrapper, JSON
import time

def get_wikidata_neighbors(entity_id, limit=10):

    if entity_id.startswith('Q') and entity_id.endswith('_'):
        entity_id=entity_id[:-1]
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setReturnFormat(JSON)

    query = f"""
    SELECT ?prop ?propLabel ?neighbor ?neighborLabel WHERE {{
      {{
        wd:{entity_id} ?prop ?neighbor.
        FILTER(isIRI(?neighbor) && STRSTARTS(STR(?neighbor), "http://www.wikidata.org/entity/Q"))
      }}
      UNION
      {{
        ?neighbor ?prop wd:{entity_id}.
        FILTER(isIRI(?neighbor) && STRSTARTS(STR(?neighbor), "http://www.wikidata.org/entity/Q"))
      }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}
    LIMIT {limit}
    """
    
    sparql.setQuery(query)
    
    neighbors = []
    try:
        results = sparql.query().convert()
        for result in results["results"]["bindings"]:
            prop_url = result.get("prop", {}).get("value", "")
            neighbor_url = result.get("neighbor", {}).get("value", "")
            
            neighbors.append({
                "relation_id": prop_url.split('/')[-1],
                "relation_name": result.get("propLabel", {}).get("value", "N/A"),
                "neighbor_id": neighbor_url.split('/')[-1],
                "neighbor_name": result.get("neighborLabel", {}).get("value", "N/A"),
            })
    except Exception as e:
        print(f"Warning: Wikidata SPARQL query failed for {entity_id}. Error: {e}")
        time.sleep(2)
        
    return neighbors


def load_umls_relations(mrrel_path):
    umls_kb = {}
    if not os.path.exists(mrrel_path):
        print(f"Warning: MRREL.RRF not found at '{mrrel_path}'. UMLS neighbor lookup will be skipped.")
        return umls_kb

    with open(mrrel_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading MRREL.RRF"):
            parts = line.strip().split('|')
            if len(parts) > 4:
                cui1, rel, cui2 = parts[0], parts[3], parts[4]
                if cui1 not in umls_kb: umls_kb[cui1] = []
                umls_kb[cui1].append({"relation_name": rel, "neighbor_id": cui2})
                if cui2 not in umls_kb: umls_kb[cui2] = []
                umls_kb[cui2].append({"relation_name": f"inverse_{rel}", "neighbor_id": cui1})
    print("UMLS relationships loaded.")
    return umls_kb

def get_umls_neighbors(entity_id, umls_kb, limit=10):
    if entity_id in umls_kb:
        all_neighbors = umls_kb[entity_id]
        if len(all_neighbors) > limit:
            import random
            return random.sample(all_neighbors, limit)
        return all_neighbors
    return []


def main():
    input_filepath = "data.json" 
    output_filepath = "data_with_neighbors.jsonl" 
    mrrel_file_path = "./MRREL.RRF" 
    max_neighbors_per_entity = 10 
    umls_kb = load_umls_relations(mrrel_file_path)
    with open(input_filepath, 'r', encoding='utf-8') as f_in, \
         open(output_filepath, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, desc="Processing documents and finding neighbors"):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            enhanced_entities = []
            for entity in item.get('linked_entities', []):
                entity_id = entity.get("linked_id")
                source = entity.get("link_source")
                neighbors = []
                if source == "MeSH/UMLS" and entity_id and umls_kb:
                    raw_neighbors = get_umls_neighbors(entity_id, umls_kb, limit=max_neighbors_per_entity)
                    for n in raw_neighbors:
                         neighbors.append({
                             "relation_id": n["relation_name"],
                             "relation_name": n["relation_name"],
                             "neighbor_id": n["neighbor_id"],
                             "neighbor_name": "N/A (Lookup required)"
                         })
                entity['neighbors'] = neighbors
                enhanced_entities.append(entity)
            item['linked_entities'] = enhanced_entities
            json.dump(item, f_out, ensure_ascii=False)
            f_out.write('\n')
            
    print(f"\nAll tasks complete! Data with neighbors has been saved to '{output_filepath}'.")

if __name__ == "__main__":
    main()