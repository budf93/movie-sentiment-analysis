import torch
import pandas as pd
import pickle
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from collections import Counter

# --- CONFIGURATION ---
dataset_path = "IMDB Dataset.csv"
NUM_SAMPLES = 2000  # Limit for demonstration speed (Use 50000 for full training)
HIDDEN_DIM = 64
EMBED_DIM = 64
EPOCHS = 10
BATCH_SIZE = 32

print("1. Loading Dataset...")
df = pd.read_csv(dataset_path)
# Convert labels: positive -> 1, negative -> 0
df['label'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
df = df.head(NUM_SAMPLES)  # Take a subset

# --- PREPROCESSING ---
print("2. Building Vocabulary...")
word_counts = Counter()
for text in df['review']:
    for word in text.lower().split():
        word_counts[word] += 1

# 1. Filter words first
filtered_words = [word for word, count in word_counts.items() if count > 1]

# 2. Create mapping with continuous indices
vocab = {word: i + 1 for i, word in enumerate(filtered_words)}

# 3. Add UNK token
vocab['<UNK>'] = 0 

vocab_size = len(vocab)
print(f"Vocabulary Size: {vocab_size}")

def create_graph_from_text(text, label, window_size=3):
    words = text.lower().split()
    node_indices = [vocab.get(w, vocab['<UNK>']) for w in words]
    
    if not node_indices: # Handle empty review edge case
        return None

    x = torch.tensor(node_indices, dtype=torch.long).unsqueeze(1)
    
    # Sliding window edges
    edge_source, edge_target = [], []
    for i in range(len(node_indices)):
        for j in range(1, window_size + 1):
            if i + j < len(node_indices):
                edge_source.extend([i, i+j])
                edge_target.extend([i+j, i])
                
    if not edge_source: # Handle single word reviews
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([edge_source, edge_target], dtype=torch.long)
    
    y = torch.tensor([label], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, y=y)

print("3. Converting Text to Graphs (this may take a moment)...")
data_list = []
for _, row in df.iterrows():
    graph = create_graph_from_text(row['review'], row['label'])
    if graph:
        data_list.append(graph)

# --- MODEL DEFINITION ---
class SentimentGNN(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.conv1 = GCNConv(embed_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = torch.nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.embedding(x.squeeze())
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return torch.sigmoid(x)

# --- TRAINING ---
print("4. Training Model...")
loader = DataLoader(data_list, batch_size=BATCH_SIZE, shuffle=True)
model = SentimentGNN(vocab_size, EMBED_DIM, HIDDEN_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCELoss()

model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out.squeeze(), batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.4f}")

# --- SAVING ARTIFACTS ---
print("5. Saving Model and Vocab...")
torch.save(model.state_dict(), "gnn_sentiment_model.pth")
with open("vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)
    
print("Done! You can now run app.py")