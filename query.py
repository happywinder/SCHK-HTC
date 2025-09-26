import requests

def query_conceptnet(entity, language="en", max_results=10):
    entity_norm = "_".join(entity.lower().strip().split())  # 变成 conceptnet API 格式
    url = f"http://api.conceptnet.io/c/{language}/increased demand"

    try:
        obj = requests.get(url).json()
        edges = obj.get('edges', [])[:max_results]
        triples = []

        for edge in edges:
            start = edge['start']['label']
            rel = edge['rel']['label']
            end = edge['end']['label']
            triples.append((start, rel, end))

        return triples

    except Exception as e:
        print(f"Error fetching {entity}: {e}")
        return []

from tqdm import tqdm
entity_knowledge = {}
with open("output_entities.txt",'r') as f:
    data = f.readlines()
    all_entities=[item.strip() for item in data]

for entity in tqdm(sorted(all_entities)):
    triples = query_conceptnet(entity)
    print(triples)
    if triples:
        entity_knowledge[entity] = triples

import json
with open("entity_conceptnet_knowledge.json", "w", encoding="utf-8") as fw:
    json.dump(entity_knowledge, fw, ensure_ascii=False, indent=2)
