"""
序列模型 (LSTM, Transformer) 用于时间序列预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMModel(nn.Module):
    """LSTM 序列模型"""
    def __init__(self, input_size=2, seq_len=15, hidden_size=256, num_classes=5, num_layers=2, dropout=0.3):
        """
        Args:
            input_size: 每个时间步的特征数 (应该 = 2)
            seq_len: 序列长度 (15个历史步)
            hidden_size: LSTM隐层大小
            num_classes: 分类数 (5)
            num_layers: LSTM层数
            dropout: dropout比例
        """
        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, 30) - 30维向量
                reshape为 (batch_size, 15, 2)
                维度0-14: pct (% change)
                维度15-29: change (change_rate)
        """
        # Reshape: (batch, 30) -> (batch, 15, 2)
        batch_size = x.size(0)
        x = x.reshape(batch_size, self.seq_len, self.input_size)
        
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 使用最后一个时间步的输出
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        # 全连接层
        x = F.relu(self.fc1(last_hidden))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class TransformerModel(nn.Module):
    """Transformer 序列模型"""
    def __init__(self, input_size=2, seq_len=15, d_model=128, nhead=4, num_layers=3, 
                 dropout=0.3, num_classes=5):
        """
        Args:
            input_size: 每个时间步的特征数 (应该 = 2)
            seq_len: 序列长度 (15个历史步)
            d_model: Transformer内部维度
            nhead: 多头注意力的头数
            num_layers: Transformer编码器层数
            dropout: dropout比例
            num_classes: 分类数 (5)
        """
        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.d_model = d_model
        
        # 嵌入层 (把输入投影到d_model维)
        self.embedding = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, seq_len, dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 全连接层
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, 30) - 30维向量
                reshape为 (batch_size, 15, 2)
                维度0-14: pct (% change)
                维度15-29: change (change_rate)
        """
        # Reshape: (batch, 30) -> (batch, 15, 2)
        batch_size = x.size(0)
        x = x.reshape(batch_size, self.seq_len, self.input_size)
        
        # 嵌入层: (batch, 15, 2) -> (batch, 15, d_model)
        x = self.embedding(x)
        
        # 位置编码: (batch, 15, d_model)
        x = self.pos_encoder(x)
        
        # Transformer编码: (batch, 15, d_model)
        x = self.transformer_encoder(x)
        
        # 使用平均池化: (batch, 15, d_model) -> (batch, d_model)
        x = x.mean(dim=1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
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
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# 为了向后兼容，创建一个包装类
class LSTMSequenceModel(nn.Module):
    """接收(batch, 30)的输入，自动拆分成(batch, 15, 2)"""
    def __init__(self, num_classes=5, hidden_size=256, num_layers=2):
        super().__init__()
        self.model = LSTMModel(
            input_size=2,
            seq_len=15,
            hidden_size=hidden_size,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout=0.3
        )
    
    def forward(self, x):
        return self.model(x)


class TransformerSequenceModel(nn.Module):
    """接收(batch, 30)的输入，自动拆分成(batch, 15, 2)"""
    def __init__(self, num_classes=5, d_model=128, num_layers=3):
        super().__init__()
        self.model = TransformerModel(
            input_size=2,
            seq_len=15,
            d_model=d_model,
            nhead=4,
            num_layers=num_layers,
            dropout=0.3,
            num_classes=num_classes
        )
    
    def forward(self, x):
        return self.model(x)
