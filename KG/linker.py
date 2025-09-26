import spacy
import scispacy
from scispacy.linking import EntityLinker
import spacy_entity_linker
import json
from tqdm import tqdm # 用于显示漂亮的进度条

# --- 1. 模型加载 (这个过程比较慢，所以只在脚本开始时执行一次) ---

def load_models():
    """加载所有需要的NLP模型，返回两个nlp对象"""
    print("开始加载模型，请稍候...")
    

    print("加载模型1: Standard SpaCy + Wikidata Linker")
    nlp_wikidata = spacy.load("en_core_web_lg")
    try:
        nlp_wikidata.add_pipe("entityLinker",last=True)
    except ValueError:
        print("警告: Wikidata 链接器已存在或加载失败，跳过添加。")
    # 加载模型 1: SciSpacy + MeSH/UMLS 链接器
    print("加载模型2: SciSpacy + umls Linker")
    nlp_scispacy = spacy.load("en_core_sci_lg")
    try:
        nlp_scispacy.add_pipe("scispacy_linker", config={"linker_name": "umls"})
    except ValueError:
        print("警告: umls 链接器已存在或加载失败，跳过添加。")

    print("所有模型加载完毕！")
    return nlp_scispacy, nlp_wikidata

# --- 2. 核心的混合链接函数 (我们之前的逻辑) ---

def get_hybrid_entity_info(entity_text, nlp_wiki, nlp_sci):
    """
    对单个实体文本进行混合链接，优先尝试Wikidata，然后是SciSpacy。
    """
    # 尝试使用Wikidata进行链接 (对通用和CS术语更好)
    doc_wiki = nlp_wiki(entity_text)
    if doc_wiki.ents and doc_wiki.ents[0]._.kb_ents:
        entity = doc_wiki.ents[0]
        return {
            "id": entity._.entity_id_,
            "name": entity._.entity_label_,
            "description": entity._.entity_desc_,
            "source": "Wikidata"
        }
    
    # 如果Wikidata失败，尝试使用SciSpacy
    doc_sci = nlp_sci(entity_text)
    if doc_sci.ents and doc_sci.ents[0]._.kb_ents:
        entity = doc_sci.ents[0]
        cui = entity._.kb_ents[0][0]
        kb_entity = nlp_sci.get_pipe('scispacy_linker').kb.cui_to_entity[cui]
        return {
            "id": cui,
            "name": kb_entity.canonical_name,
            "description": kb_entity.definition,
            "source": "MeSH/UMLS"
        }
    
    # 如果全部失败，返回规范化文本
    return {
        "id": entity_text.lower().replace(" ", "_"),
        "name": entity_text,
        "description": "N/A",
        "source": "Fallback"
    }


def main():
    input_filepath = "entity.json"
    output_filepath = "output_linked_data.json"

    nlp_scispacy, nlp_wikidata = load_models()

    with open(input_filepath, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    print(f"共找到 {len(all_data)} 条数据待处理。")

    processed_results = []
    
    for item in tqdm(all_data, desc="正在链接实体"):
        original_entities = item.get('entities', [])
        
        linked_entities_list = []
        
        for entity_text, entity_type in original_entities:
            linked_info = get_hybrid_entity_info(entity_text, nlp_wikidata, nlp_scispacy)
            
            final_entity_data = {
                "original_text": entity_text,
                "original_type": entity_type,
                "linked_id": linked_info["id"],
                "linked_name": linked_info["name"],
                "linked_description": linked_info["description"],
                "link_source": linked_info["source"]
            }
            # print(final_entity_data)
            linked_entities_list.append(final_entity_data)
            
        # 创建一个新的item字典，包含所有原始信息和新链接的实体列表
        processed_item = {
            "text": item["text"],
            "label_level_1": item["label_level_1"],
            "label_level_2": item["label_level_2"],
            "linked_entities": linked_entities_list # 使用新的键来存储
        }

        with open(output_filepath, 'a', encoding='utf-8') as f:
            # indent=2 让JSON文件格式化，更易读
            json.dump(processed_item, f, ensure_ascii=False)
            f.write("\n")
        
    print("所有任务完成！")

if __name__ == "__main__":
    main()