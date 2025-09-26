# SCHK-HTC

Source code for our paper:SCHK-HTC: Sibling Contrastive Learning with Hierarchical Knowledge-aware Prompt Tuning For Hierarchy Text Classification

## Data Preparation

### HTC datasets

Please download the original dataset and then use these scripts.

#### WebOfScience

Please download wos dataset [link](https://drive.google.com/file/d/1UuVDd3uEVVFcuy6i-LdZUo6SHJSMQ1-b/view?usp=share_link)

Excute the scripts

```shell
cd ./dataset/WebOfScience
python preprocess_wos.py
```

#### DBpedia

The original dataset wiki_data.csv can be acquired [link](https://drive.google.com/file/d/1eAHuIvasAd3g2oA-meWEeksIIfqycRWq/view?usp=drive_link)

```
cd ./dataset/WebOfScience
python preprocess_dbp.py
```

#### Rcv1-v2

The original dataset can be acquired [here](https://trec.nist.gov/data/reuters/reuters.html) by signing an agreement.

```
cd ./dataset/rcv1
python preprocess_rcv1.py
python data_rcv1.py
```

### Knowledge Graph

We adopt the advanced knowledge graph named Wikidata[1]

### Data Process

1. Following the strategy proposed by KagNet[2]. Please refer to the original codes: https://github.com/INK-USC/KagNet.
2. After recognizing the entities, adopt the Node2Vec[3] to train concept embedding.

# Citation

`[1]Vrandečić, Denny, and Markus Krötzsch. "Wikidata: a free collaborative knowledgebase." *Communications of the ACM* 57.10 (2014): 78-85.`

`[2]Lin, Bill Yuchen, et al. "Kagnet: Knowledge-aware graph networks for commonsense reasoning." arXiv preprint arXiv:1909.02151 (2019).`

`[3]Grover, Aditya, and Jure Leskovec. "node2vec: Scalable feature learning for networks." Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining. 2016.`



