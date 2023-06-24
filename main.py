import torch
from torch import nn


class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_heads, num_layers, dropout=0.5):
        super(Transformer, self).__init__()

        # 位置编码层
        self.pos_encoder = PositionalEncoding(input_dim, dropout)
        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(input_dim, num_heads, hidden_dim, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        # 全连接层
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, src):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.fc(output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# 参数设置
INPUT_DIM = 512
OUTPUT_DIM = 512
HIDDEN_DIM = 2048
NUM_HEADS = 8
NUM_LAYERS = 6
DROPOUT = 0.1

model = Transformer(INPUT_DIM, OUTPUT_DIM, HIDDEN_DIM, NUM_HEADS, NUM_LAYERS, DROPOUT)
print(model)
