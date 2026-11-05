import torch
import torch.nn as nn
from torch.distributions import Categorical
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class ActorCritic(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout, nhead=8, max_len=200):
        super().__init__()
        
        # 1. Embeddings
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, max_len)
        
        # 2. Transformer Encoder (GPT Architecture essentially)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size, 
            nhead=nhead, 
            dim_feedforward=hidden_size, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Heads
        self.actor_head = nn.Linear(embed_size, vocab_size)
        self.critic_head = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.embed_size = embed_size
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _generate_square_subsequent_mask(self, sz, device):
        # ماسک برای اینکه مدل آینده را نبیند
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(device)

    def forward(self, x):
        # x: [Batch, Seq_Len]
        mask = self._generate_square_subsequent_mask(x.size(1), x.device)
        emb = self.embedding(x) * math.sqrt(self.embed_size)
        emb = self.pos_encoder(emb)
        output = self.transformer(emb, mask=mask, is_causal=True)
        return output

    def get_action_and_value(self, x, action=None, temperature=1.0):
        output = self.forward(x) # [Batch, Seq, Embed]
        logits = self.actor_head(output)
        values = self.critic_head(output)
        
        dist = Categorical(logits=logits)
        
        if action is None:
            # حالت تولید (Generation): فقط آخرین توکن
            last_logits = logits[:, -1, :] / temperature
            dist_last = Categorical(logits=last_logits)
            raw_action = dist_last.sample()
            log_prob = dist_last.log_prob(raw_action)
            return raw_action, log_prob, None, values[:, -1, :]
        else:
            # حالت آموزش (Training): کل sequence
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            return action, log_prob, entropy, values
