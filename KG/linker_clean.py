import json
with open("output_linked_data.json",'r') as f:
    data=[json.loads(item) for item in f.readlines()]

for index,item in enumerate(data):
    clean_list=[]
    for d in item['linked_entities']:
        if d['link_source']=='MeSH/UMLS':
            clean_list.append(d)
    data[index]['linked_entities']=clean_list

with open("data.json",'a') as f:
    for _,item in enumerate(data):
        item['doc_token']=item['text']
        item.pop('text')
        item['doc_label']=[item['label_level_1'],item['label_level_2']]
        item.pop('label_level_1')
        item.pop('label_level_2')
        json.dump(item,f,ensure_ascii=False)
        f.write("\n")
