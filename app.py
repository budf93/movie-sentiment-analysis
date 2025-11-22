import streamlit as st
import torch
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

# --- CONFIGURATION ---
# Must match training config
HIDDEN_DIM = 64
EMBED_DIM = 64

# --- MODEL CLASS (Must be identical to training) ---
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
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return torch.sigmoid(x)

# --- HELPER FUNCTIONS ---
@st.cache_resource
def load_artifacts():
    # Load Vocab
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    
    # Load Model
    vocab_size = len(vocab)
    model = SentimentGNN(vocab_size, EMBED_DIM, HIDDEN_DIM)
    model.load_state_dict(torch.load("gnn_sentiment_model.pth"))
    model.eval()
    
    return model, vocab

def process_input(text, vocab, window_size=3):
    words = text.lower().split()
    node_indices = [vocab.get(w, vocab['<UNK>']) for w in words]
    
    if not node_indices:
        return None, None

    x = torch.tensor(node_indices, dtype=torch.long).unsqueeze(1)
    
    edge_source, edge_target = [], []
    for i in range(len(node_indices)):
        for j in range(1, window_size + 1):
            if i + j < len(node_indices):
                edge_source.extend([i, i+j])
                edge_target.extend([i+j, i])
                
    if not edge_source:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([edge_source, edge_target], dtype=torch.long)
        
    batch = torch.zeros(x.size(0), dtype=torch.long) # Batch vector of zeros for single sample
    
    data = Data(x=x, edge_index=edge_index, batch=batch)
    return data, words

def draw_graph(data, words):
    G = nx.Graph()
    for i, word in enumerate(words):
        G.add_node(i, label=word)
    
    edge_index = data.edge_index.numpy()
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        G.add_edge(src, dst)

    fig, ax = plt.subplots(figsize=(10, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, 
            labels=nx.get_node_attributes(G, 'label'),
            node_color='#A0CBE2', edge_color='#BBBBBB', 
            node_size=2000, font_size=9, ax=ax)
    return fig

# --- STREAMLIT UI ---
st.title("🎬 GNN Sentiment Analysis")
st.markdown("This app uses a **Graph Neural Network** to analyze the sentiment of movie reviews. It constructs a graph where words are nodes and nearby words are connected.")

# Load Model
try:
    model, vocab = load_artifacts()
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Model files not found. Please run 'train.py' first.")
    st.stop()

# User Input
user_text = st.text_area("Enter a Movie Review:", "The cinematography was beautiful but the story was dull and boring.")

if st.button("Analyze Sentiment"):
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        # 1. Process
        data, words = process_input(user_text, vocab)
        
        if data:
            # 2. Predict
            with torch.no_grad():
                prediction = model(data).item()
            
            # 3. Display Results
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Result")
                sentiment = "Positive" if prediction > 0.5 else "Negative"
                color = "green" if sentiment == "Positive" else "red"
                st.markdown(f"### :{color}[{sentiment}]")
                st.markdown(f"**Confidence:** {prediction:.4f}")
            
            with col2:
                st.subheader("Graph Structure")
                st.markdown("How the GNN sees your text:")
                fig = draw_graph(data, words)
                st.pyplot(fig)
        else:
            st.error("Could not process text (words might be unknown).")