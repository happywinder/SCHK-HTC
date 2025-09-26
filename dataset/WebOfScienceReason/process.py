import json

with open("wos_train.json",'r')as f:
    data= f.readlines()
    with open("wos_train_revise.json",'w') as p:
        for line in data:
            t=json.loads(line)
            t['doc_token']=t['doc_token'].split('\nReasons for categorization at the first level:')[0]
            p.write(json.dumps(t,ensure_ascii=False)+"\n")
