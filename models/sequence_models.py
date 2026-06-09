"""
序列模型 — 统一输入 [bs, seq_len, feature_dim], Backbone+Classifier 分离
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MLPClassifier(nn.Module):
    """统一分类头: d_model → d_model/2 → num_class"""
    def __init__(self, d_model, num_class, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, num_class)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [bs, d_model]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ========================== LSTM ==========================

class LSTMModel(nn.Module):
    """LSTM 序列模型 — 统一输入 [bs, seq_len, feature_dim]"""
    def __init__(self, feature_dim=4, seq_len=30, num_class=4,
                 hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_class = num_class

        self.embed = nn.Linear(feature_dim, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.classifier = MLPClassifier(hidden_size, num_class, dropout)

    def forward(self, x):
        # x: [bs, seq_len, feature_dim]
        x = self.embed(x)                       # [bs, seq_len, hidden]
        x, _ = self.lstm(x)                     # [bs, seq_len, hidden]
        x = x[:, -1, :]                          # [bs, hidden]
        x = self.classifier(x)                   # [bs, num_class]
        return x

    def backbone_state_dict(self):
        full = self.state_dict()
        return {k: v for k, v in full.items() if not k.startswith('classifier.')}

    def load_backbone_state_dict(self, backbone_state):
        own = self.state_dict()
        for k, v in backbone_state.items():
            if k in own and not k.startswith('classifier.'): own[k] = v
        self.load_state_dict(own, strict=False)

    def freeze_backbone(self):
        for name, p in self.named_parameters():
            if not name.startswith('classifier.'): p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.parameters(): p.requires_grad = True


# ========================== Transformer Encoder ==========================

class EncoderModel(nn.Module):
    """Transformer Encoder — 统一输入 [bs, seq_len, feature_dim]"""
    def __init__(self, feature_dim=4, seq_len=30, num_class=4,
                 d_model=128, nhead=4, num_layers=3, dropout=0.3):
        super().__init__()
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_class = num_class

        self.embedding = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, seq_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = MLPClassifier(d_model, num_class, dropout)

    def forward(self, x):
        # x: [bs, seq_len, feature_dim]
        x = self.embedding(x)                    # [bs, seq_len, d_model]
        x = self.pos_encoder(x)                  # [bs, seq_len, d_model]
        x = self.transformer(x)                  # [bs, seq_len, d_model]
        x = x.mean(dim=1)                        # [bs, d_model]
        x = self.classifier(x)                   # [bs, num_class]
        return x

    def backbone_state_dict(self):
        full = self.state_dict()
        return {k: v for k, v in full.items() if not k.startswith('classifier.')}

    def load_backbone_state_dict(self, backbone_state):
        own = self.state_dict()
        for k, v in backbone_state.items():
            if k in own and not k.startswith('classifier.'): own[k] = v
        self.load_state_dict(own, strict=False)

    def freeze_backbone(self):
        for name, p in self.named_parameters():
            if not name.startswith('classifier.'): p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.parameters(): p.requires_grad = True


# ========================== Decoder Transformer ==========================

class DecoderTransformerModel(nn.Module):
    """Decoder-only Transformer — 统一输入 [bs, seq_len, feature_dim]"""
    def __init__(self, feature_dim=4, seq_len=30, num_class=4,
                 d_model=128, nhead=4, num_layers=3, dropout=0.3):
        super().__init__()
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_class = num_class

        self.embedding = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, seq_len, dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.classifier = MLPClassifier(d_model, num_class, dropout)

    def forward(self, x):
        # x: [bs, seq_len, feature_dim]
        x = self.embedding(x)
        x = self.pos_encoder(x)
        causal_mask = torch.tril(torch.ones(self.seq_len, self.seq_len, device=x.device))
        x = self.transformer(x, x, tgt_mask=causal_mask)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x

    def backbone_state_dict(self):
        full = self.state_dict()
        return {k: v for k, v in full.items() if not k.startswith('classifier.')}

    def load_backbone_state_dict(self, backbone_state):
        own = self.state_dict()
        for k, v in backbone_state.items():
            if k in own and not k.startswith('classifier.'): own[k] = v
        self.load_state_dict(own, strict=False)

    def freeze_backbone(self):
        for name, p in self.named_parameters():
            if not name.startswith('classifier.'): p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.parameters(): p.requires_grad = True


# ========================== MoE Decoder ==========================

class MoEFeedForward(nn.Module):
    """MoE FFN with load-balancing aux loss"""
    def __init__(self, d_model, dim_feedforward, num_experts=4, top_k=1, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = nn.Linear(d_model, num_experts)
        expert_dim = dim_feedforward // num_experts * 2
        self.W1 = nn.Linear(d_model, expert_dim)
        self.W2 = nn.Linear(expert_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.aux_loss = 0.0

    def forward(self, x):
        N, seq_len, D = x.shape
        flat = x.reshape(-1, D)
        router_logits = self.router(flat)
        router_probs = F.softmax(router_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        # Aux loss
        ones_hot = F.one_hot(topk_indices.view(-1), num_classes=self.num_experts).float()
        self.aux_loss = self.num_experts * (ones_hot.mean(dim=0) * router_probs.mean(dim=0)).sum()

        out = torch.zeros_like(flat)
        for exp_i in range(self.num_experts):
            mask = (topk_indices == exp_i)
            idx = mask.nonzero(as_tuple=True)[0]
            if idx.numel() == 0: continue
            w = topk_probs[idx]
            if w.dim() > 1: w = w[:, 0] if self.top_k > 1 else w.squeeze(-1)
            expert_in = flat[idx]
            h = F.relu(self.W1(expert_in))
            h = self.dropout(h)
            expert_out = self.dropout(self.W2(h))
            out[idx] += expert_out * w.unsqueeze(-1)

        return out.reshape(N, seq_len, D)


class MoEDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_experts=4, top_k=1, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.moe = MoEFeedForward(d_model, dim_feedforward, num_experts, top_k, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask=None):
        attn_out, _ = self.self_attn(x, x, x, attn_mask=causal_mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.moe(x)))
        return x

    @property
    def aux_loss(self):
        return self.moe.aux_loss


class DecoderTransformerModelMoE(nn.Module):
    """MoE Decoder Transformer — 统一输入 [bs, seq_len, feature_dim]"""
    def __init__(self, feature_dim=4, seq_len=30, num_class=4,
                 d_model=128, nhead=4, num_layers=3, num_experts=4, top_k=1, dropout=0.3):
        super().__init__()
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_class = num_class

        self.embedding = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, seq_len, dropout)

        dim_feedforward = d_model * 4
        self.layers = nn.ModuleList([
            MoEDecoderLayer(d_model, nhead, dim_feedforward, num_experts, top_k, dropout)
            for _ in range(num_layers)
        ])

        self.classifier = MLPClassifier(d_model, num_class, dropout)

    def forward(self, x):
        # x: [bs, seq_len, feature_dim]
        x = self.embedding(x)
        x = self.pos_encoder(x)
        causal_mask = torch.triu(torch.ones(self.seq_len, self.seq_len, device=x.device) * float('-inf'), diagonal=1)
        total_aux = 0.0
        for layer in self.layers:
            x = layer(x, causal_mask=causal_mask)
            total_aux += layer.aux_loss
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x, total_aux

    def backbone_state_dict(self):
        full = self.state_dict()
        return {k: v for k, v in full.items() if not k.startswith('classifier.')}

    def load_backbone_state_dict(self, backbone_state):
        own = self.state_dict()
        for k, v in backbone_state.items():
            if k in own and not k.startswith('classifier.'): own[k] = v
        self.load_state_dict(own, strict=False)

    def freeze_backbone(self):
        for name, p in self.named_parameters():
            if not name.startswith('classifier.'): p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.parameters(): p.requires_grad = True
