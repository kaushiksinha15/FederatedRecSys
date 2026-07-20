import torch
import torch.nn as nn

class TransformerRecommender(nn.Module):
    """
    A lightweight Transformer-based model for sequential recommendation.
    Simulates the 'Personalized Transformer Aggregation' aspect.
    """
    def __init__(self, num_items=2000, embed_dim=64, num_heads=4, num_layers=2, dropout=0.1):
        super(TransformerRecommender, self).__init__()
        self.num_items = num_items
        self.embed_dim = embed_dim
        
        # Item embeddings (0 is usually padding)
        self.item_embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        
        # Positional encoding for sequence (assuming max sequence length of 50)
        self.position_embedding = nn.Embedding(50, embed_dim)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer to predict the next item
        self.fc_out = nn.Linear(embed_dim, num_items + 1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand_as(x)
        
        # Embeddings + Positional
        out = self.item_embedding(x) + self.position_embedding(positions)
        
        # Pass through Transformer
        out = self.transformer(out)
        
        # We take the representation of the last item in the sequence to predict the next one
        last_hidden_state = out[:, -1, :]
        
        # Project to item vocabulary size
        logits = self.fc_out(last_hidden_state)
        return logits
