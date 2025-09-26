import json
from itertools import combinations
import os
from tqdm import tqdm

def build_graph_from_linked_data(linked_data_path, mrrel_path, output_kg_dir):
    """
    从已经完成实体链接的数据中提取关系，构建知识图谱。

    Args:
        linked_data_path (str): 包含 'linked_entities' 的 JSON Lines 文件路径。
        mrrel_path (str): UMLS 的 MRREL.RRF 文件路径。
        output_kg_dir (str): 输出 KG 文件的目录。
    """
    
    # --- 1. 预加载 UMLS 关系数据 ---
    print("Step 1: Pre-loading UMLS relationships from MRREL.RRF...")
    # 创建一个字典来高效地存储和查询关系: {cui1: {cui2: [rel1, rel2, ...]}}
    relations_kb = {}
    if os.path.exists(mrrel_path):
        with open(mrrel_path, 'r', encoding='utf-8') as f:
            # 使用tqdm来显示加载进度
            for line in tqdm(f, desc="Loading MRREL.RRF"):
                # MRREL.RRF 文件列由'|'分隔
                parts = line.strip().split('|')
                if len(parts) > 4:
                    cui1, rel, cui2 = parts[0], parts[3], parts[4]
                    
                    # 初始化内部字典
                    if cui1 not in relations_kb:
                        relations_kb[cui1] = {}
                    if cui2 not in relations_kb[cui1]:
                        relations_kb[cui1][cui2] = set() # 使用set避免重复关系
                    
                    relations_kb[cui1][cui2].add(rel)
        print("UMLS relationships loaded successfully.")
    else:
        print(f"Warning: MRREL.RRF not found at '{mrrel_path}'. The script will only generate co-occurrence relationships.")
        
    # --- 2. 遍历链接好的数据，提取三元组 ---
    print("\nStep 2: Extracting triplets from linked data...")
    all_triplets = set()  # 使用 set 自动处理重复的三元组
    all_entities = set()  # 收集所有出现过的实体

    with open(linked_data_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing linked documents"):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # 提取当前文档中所有唯一的、非回退的实体ID
            entity_ids_in_doc = list(set(
                entity['linked_id'] 
                for entity in item.get('linked_entities', [])
                if entity.get('link_source') != 'Fallback' # 过滤掉链接失败的实体
            ))
            
            # 更新全局实体集合
            all_entities.update(entity_ids_in_doc)
            
            if len(entity_ids_in_doc) < 2:
                continue

            # 遍历文档中所有的实体对 (cui1, cui2)
            for cui1, cui2 in combinations(entity_ids_in_doc, 2):
                found_explicit_relation = False
                
                # 检查正向关系: cui1 -> cui2
                if cui1 in relations_kb and cui2 in relations_kb.get(cui1, {}):
                    for rel in relations_kb[cui1][cui2]:
                        all_triplets.add((cui1, rel, cui2))
                    found_explicit_relation = True
                
                # 检查反向关系: cui2 -> cui1
                if cui2 in relations_kb and cui1 in relations_kb.get(cui2, {}):
                    for rel in relations_kb[cui2][cui1]:
                        all_triplets.add((cui2, rel, cui1))
                    found_explicit_relation = True
                
                # 如果没有找到任何显式关系，则添加共现关系
                if not found_explicit_relation:
                    all_triplets.add((cui1, "co_occurrence", cui2))
                    
    print(f"\nExtraction complete. Found {len(all_entities)} unique entities and {len(all_triplets)} unique triplets.")

    # --- 3. 将图谱数据写入文件 ---
    print("\nStep 3: Writing knowledge graph files...")
    if not os.path.exists(output_kg_dir):
        os.makedirs(output_kg_dir)
        
    # 为所有收集到的实体和关系创建ID映射
    # 确保实体和关系列表是排序的，以保证每次运行生成的ID一致
    entity_list = sorted(list(all_entities))
    relation_list = sorted(list(set(t[1] for t in all_triplets)))
    entity_to_id = {name: i for i, name in enumerate(entity_list)}
    relation_to_id = {name: i for i, name in enumerate(relation_list)}
    
    # 写入 entity2id.txt
    with open(os.path.join(output_kg_dir, 'entity2id.txt'), 'w', encoding='utf-8') as f:
        f.write(f"{len(entity_to_id)}\n")
        for name, eid in entity_to_id.items():
            f.write(f"{name}\t{eid}\n")

    # 写入 relation2id.txt
    with open(os.path.join(output_kg_dir, 'relation2id.txt'), 'w', encoding='utf-8') as f:
        f.write(f"{len(relation_to_id)}\n")
        for name, rid in relation_to_id.items():
            f.write(f"{name}\t{rid}\n")

    # 写入 train2id.txt
    with open(os.path.join(output_kg_dir, 'train2id.txt'), 'w', encoding='utf-8') as f:
        f.write(f"{len(all_triplets)}\n")
        for h, r, t in tqdm(all_triplets, desc="Writing triplets"):
            # 确保三元组中的实体和关系都在映射表中
            if h in entity_to_id and t in entity_to_id and r in relation_to_id:
                hid = entity_to_id[h]
                rid = relation_to_id[r]
                tid = entity_to_id[t]
                # PyKE/PyG 格式: head_id, tail_id, relation_id
                f.write(f"{hid}\t{tid}\t{rid}\n")

    print(f"\nKnowledge graph files have been successfully saved to the '{output_kg_dir}' directory.")
    print("You can now use this directory to train your KGE model (e.g., RotatE).")

if __name__ == "__main__":
    LINKED_DATA_FILE = "data.json"

    # 2. UMLS关系文件: MRREL.RRF 的完整路径
    #    示例: "D:/UMLS_data/2023AA/META/MRREL.RRF"
    #    如果此文件不存在，脚本只会生成共现关系。
    MRREL_FILE_PATH = "./MRREL.RRF" 

    # 3. 输出目录: 存放最终图谱文件的文件夹名称
    OUTPUT_KG_FOLDER = "final_enriched_kg"
    
    # --- 执行主函数 ---
    build_graph_from_linked_data(
        linked_data_path=LINKED_DATA_FILE,
        mrrel_path=MRREL_FILE_PATH,
        output_kg_dir=OUTPUT_KG_FOLDER
    )