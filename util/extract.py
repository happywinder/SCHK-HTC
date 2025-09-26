import json
import spacy

nlp = spacy.load("en_core_web_sm")  # 可换成更强的模型如 'en_core_web_trf'

label_entities = set()
text_entities = set()

with open("dataset/WebOfScience/wos_train.json", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)

        # 1. 从 doc_label 中提取实体
        labels = obj.get("doc_label", [])
        label_entities.update(label.lower() for label in labels)

        # 2. 从 doc_token 中抽取名词短语
        text = obj.get("doc_token", "")
        doc = nlp(text)
        for chunk in doc.noun_chunks:
            phrase = chunk.text.lower().strip()
            if len(phrase.split()) <= 6 and len(phrase) > 2:
                text_entities.add(phrase)

# 合并所有实体
all_entities = label_entities.union(text_entities)

# 输出结果
print(f"共提取实体数：{len(all_entities)}")
with open("dataset/WebOfScience/extracted_entities.txt", "w", encoding="utf-8") as fw:
    for e in sorted(all_entities):
        fw.write(e + "\n")
