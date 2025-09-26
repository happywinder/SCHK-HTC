import spacy
import scispacy
import json
from typing import List, Dict
from scispacy.abbreviation import AbbreviationDetector
from tqdm import tqdm

nlp_general = spacy.load("en_core_web_trf")    
nlp_sci = spacy.load("en_core_sci_scibert")   
nlp_sci.add_pipe("abbreviation_detector")
ENTITY_TYPE_MAPPING = {
    "Person": ["professor", "scientist", "student", "author", "researcher"],
    "Organization": ["university", "institute", "company", "lab", "department"],
    "Location": ["china", "germany", "hospital", "center", "beijing"],
    "Technology": ["deep learning", "cnn", "transformer", "wireless sensor", "mri", "algorithm", "blockchain"],
    "Medical": ["diabetes", "cancer", "brain", "liver", "antibody", "virus", "therapy", "alzheimer", "epilepsy"],
    "Biological": ["bacteria", "protein", "enzyme", "mouse", "dna", "gene"],
    "Abstract": ["intelligence", "emotion", "consciousness", "freedom"],
    "Product": ["iphone", "mri machine", "vaccine"],
    "Event": ["covid-19", "earthquake", "war"],
}

def guess_entity_type(entity_text: str) -> str:
    text_lower = entity_text.lower()
    for entity_type, keywords in ENTITY_TYPE_MAPPING.items():
        for keyword in keywords:
            if keyword in text_lower:
                return entity_type
    return "Unknown"

def extract_entities_combined(text: str) -> List[Dict]:
    results = {}

    doc_general = nlp_general(text)
    doc_sci = nlp_sci(text)
    for ent in doc_general.ents:
        if ent.label_ not in {"CARDINAL", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY"}:
            key = ent.text.strip().lower()
            if key not in results:
                results[key] = {
                    "entity": ent.text.strip(),
                    "source": "spaCy-NER",
                    "spacy_type": ent.label_,
                    "mapped_type": guess_entity_type(ent.text)
                }
    for chunk in doc_general.noun_chunks:
        phrase = chunk.text.strip()
        if 2 <= len(phrase.split()) <= 6:
            key = phrase.lower()
            if key not in results:
                results[key] = {
                    "entity": phrase,
                    "source": "spaCy-Chunk",
                    "spacy_type": "NOUN_PHRASE",
                    "mapped_type": guess_entity_type(phrase)
                }

    for ent in doc_sci.ents:
        key = ent.text.strip().lower()
        if key not in results:
            results[key] = {
                "entity": ent.text.strip(),
                "source": "scispaCy",
                "spacy_type": ent.label_,
                "mapped_type": guess_entity_type(ent.text)
            }

    return list(results.values())

if __name__ == "__main__":
    input_path = "dataset/WebOfScience/wos_total.json"
    output_path = "dataset/WebOfScience/wos_entities.json"

    with open(input_path, 'r') as f:
        data = [json.loads(line) for line in f]

    output = []

    for idx, item in enumerate(tqdm(data)):
        doc_id = item.get("doc_id", f"doc_{idx}")
        text = item["doc_token"]
        entities = extract_entities_combined(text)

        output.append({
            "doc_id": doc_id,
            "entities": entities
        })

    with open(output_path, 'w') as fout:
        for record in output:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

