import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor

class TemporalAttentionPool(nn.Module):
    def __init__(self, hidden_size: int, attn_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1, bias=False),
        )
    def forward(self, hidden_states):
        scores  = self.attn(hidden_states)           # [B, T, 1]
        weights = torch.softmax(scores, dim=1)       # [B, T, 1]
        pooled  = (hidden_states * weights).sum(dim=1)
        return pooled, weights.squeeze(-1)

class Wav2Vec2Classifier(nn.Module):
    def __init__(self, wav2vec2_name, hidden_dim, attn_dim, dropout,
                 freeze_encoder=False):
        super().__init__()
        self.encoder   = Wav2Vec2Model.from_pretrained(wav2vec2_name)
        encoder_dim    = self.encoder.config.hidden_size  # 768
        self.attn_pool = TemporalAttentionPool(encoder_dim, attn_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1),
        )
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, input_values):
        outputs      = self.encoder(input_values)
        hidden       = outputs.last_hidden_state
        pooled, attn = self.attn_pool(hidden)
        logit        = self.classifier(pooled)
        return logit.squeeze(-1), attn   # returns (score [B], attn_weights [B,T])
