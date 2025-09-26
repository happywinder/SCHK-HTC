import spacy
import scispacy
from datasets import load_dataset
from tqdm.auto import tqdm
import json
nlp = spacy.load("en_core_sci_lg")

def extract_entities_from_text(text):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append((ent.text, ent.label_))
    return entities
sample_index = 10 
with open("dataset/WebOfScience/wos_total.json",'r') as f:
    data=f.readlines()
    data=[json.loads(item) for item in data]

all_extracted_data = []

for i in tqdm(range(len(data))):
    text = data[i]['doc_token']
    
    entities = extract_entities_from_text(text)
    
    all_extracted_data.append({
        'text': text,
        'label_level_1': data[i]['doc_label'][0],
        'label_level_2': data[i]['doc_label'][1],
        'entities': entities
    })

print("\n批量提取完成！")

with open("entity.json",'w') as f:
    json.dump(all_extracted_data,f,ensure_ascii=False,indent=2)