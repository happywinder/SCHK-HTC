import os
os.environ['CUDA_VISIBLE_DEVICES']='5' 
import torch

from torch_geometric.nn import Node2Vec
from torch_geometric.data import Data
import numpy as np
from tqdm import tqdm

def load_edge_index(data_dir):

    with open(os.path.join(data_dir, 'entity2id.txt'), 'r') as f:
        num_nodes = int(f.readline().strip())
        
    print(f"Total number of nodes (entities): {num_nodes}")
    
    triplets = np.loadtxt(os.path.join(data_dir, 'train2id.txt'), dtype=np.int64, skiprows=1)
    
    head_nodes = triplets[:, 0]
    tail_nodes = triplets[:, 1]

    edge_index_forward = torch.tensor([head_nodes, tail_nodes], dtype=torch.long)
    edge_index_backward = torch.tensor([tail_nodes, head_nodes], dtype=torch.long)
    edge_index = torch.cat([edge_index_forward, edge_index_backward], dim=1)
    
    print(f"Created edge_index with {edge_index.shape[1]} edges (bidirectional).")
    
    return num_nodes, edge_index

def train_with_deepwalk(data_dir, embedding_dim=768, walk_length=80, context_size=10, walks_per_node=10, epochs=5, batch_size=128, lr=0.01):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    num_nodes, edge_index = load_edge_index(data_dir)

    model = Node2Vec(
        num_nodes=num_nodes,
        edge_index=edge_index,
        embedding_dim=embedding_dim,
        walk_length=walk_length,       
        context_size=context_size,     
        walks_per_node=walks_per_node, 
        p=1.0,                         
        q=1.0,                         
        num_negative_samples=1,        
        sparse=True,                 
    ).to(device)
    
    print("DeepWalk (Node2Vec p=1, q=1) model created.")
    loader = model.loader(batch_size=batch_size, shuffle=True, num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=lr)

    def train_epoch():
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc="Training Epoch")
        for pos_rw, neg_rw in pbar:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        return total_loss / len(loader)

    print("\nStarting training...")
    for epoch in range(1, epochs + 1):
        loss = train_epoch()
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
    print("\nTraining finished. Saving embeddings...")
    model.eval()
    all_embeddings = model().data.cpu()
    
    output_filename = "deepwalk.pt"
    torch.save(all_embeddings, output_filename)
    print(f"Node embeddings saved to {output_filename}. Shape: {all_embeddings.shape}")


if __name__ == "__main__":
    kg_data_directory = "./final_enriched_kg" 
    train_with_deepwalk(
        data_dir=kg_data_directory,
        embedding_dim=768,      
        walk_length=80,         
        context_size=10,       
        walks_per_node=100,     
        epochs=5,
        batch_size=8,
        lr=0.005
    )