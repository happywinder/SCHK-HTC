import torch
import torch.optim as optim
from torch_geometric.nn.kge import RotatE # 1. 导入 RotatE
from tqdm import tqdm
import os
import numpy as np

# 设置使用的GPU
os.environ['CUDA_VISIBLE_DEVICES']='5' 

def load_kg_data(data_dir):
    with open(os.path.join(data_dir, 'entity2id.txt'), 'r') as f:
        num_nodes = int(f.readline().strip())
    with open(os.path.join(data_dir, 'relation2id.txt'), 'r') as f:
        num_relations = int(f.readline().strip())
        
    # 加载三元组，假设是 (h, t, r) 格式
    triplets_htr = np.loadtxt(os.path.join(data_dir, 'train2id.txt'), dtype=np.int64, skiprows=1)
    # 转换为 PyG 需要的 (h, r, t) 格式
    triplets_hrt = triplets_htr[:, [0, 2, 1]] 
    
    head_index = torch.from_numpy(triplets_hrt[:, 0])
    relation_type = torch.from_numpy(triplets_hrt[:, 1])
    tail_index = torch.from_numpy(triplets_hrt[:, 2])

    print(f"Loaded {num_nodes} entities, {num_relations} relations, and {len(head_index)} triplets.")
    
    return num_nodes, num_relations, head_index, relation_type, tail_index


def train_embeddings_with_pyg(data_dir, embedding_dim=768, epochs=500, lr=0.001, batch_size=2048, checkpoint_interval=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    num_nodes, num_relations, head_index, relation_type, tail_index = load_kg_data(data_dir)
    model = RotatE(
        num_nodes=num_nodes,
        num_relations=num_relations,
        hidden_channels=embedding_dim // 2, 
    ).to(device)
    print(f"Using RotatE model with embedding dimension: {embedding_dim} (hidden_channels={embedding_dim//2})")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        loader = model.loader(
            head_index=head_index.to(device),
            rel_type=relation_type.to(device),
            tail_index=tail_index.to(device),
            batch_size=batch_size,
            shuffle=True,
        )
        
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        for head, rel, tail in pbar:
            optimizer.zero_grad()
            loss = model.loss(head, rel, tail)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{epochs}, Average Loss: {avg_loss:.4f}, Best Loss: {best_loss:.4f}")
        
        if epoch % checkpoint_interval == 0:
            if avg_loss < best_loss:
                print(f"Loss improved from {best_loss:.4f} to {avg_loss:.4f}. Saving model...")
                best_loss = avg_loss
                
                # 4. 修改保存的文件名
                torch.save(model.state_dict(), "best_model_rotate.pt")
                print("Model state_dict saved to best_model_rotate.pt")
            else:
                print(f"Loss did not improve from {best_loss:.4f}. Not saving.")
    best_model_path = "best_model_rotate.pt"
    print(f"\nTraining finished. Loading best model from {best_model_path} to save final embeddings...")
    
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print("Best model loaded successfully.")
    else:
        print("No best model was saved during training. Saving the final model instead.")
        
    model.eval()
    entity_embeddings = model.node_emb.weight.data.cpu()
    torch.save(entity_embeddings, "entity_embeddings_rotate.pt")
    print(f"Entity embeddings from the best model saved to entity_embeddings_rotate.pt. Shape: {entity_embeddings.shape}")

    relation_embeddings = model.rel_emb.weight.data.cpu()
    torch.save(relation_embeddings, "relation_embeddings_rotate.pt")
    
if __name__ == "__main__":
    kg_data_directory = "./final_enriched_kg"
    
    train_embeddings_with_pyg(
        data_dir=kg_data_directory,
        embedding_dim=768, 
        epochs=500,
        lr=0.0005,             
        batch_size=1024,   
        checkpoint_interval=10
    )