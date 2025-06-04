import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CrossFormerBlock1D(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, drop=0.1, attn_drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, 
                                          dropout=attn_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        self.drop_path = nn.Identity()  # có thể dùng DropPath nếu muốn

    def forward(self, x):
        # x: (B, T, D)
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = residual + self.drop_path(x)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class CrossFormerClassifier1D(nn.Module):
    def __init__(self, input_len=256, num_segments=4, dim=64, num_classes=11, num_blocks=3):
        super().__init__()
        self.embedding = nn.Linear(input_len, dim)
        self.blocks = nn.Sequential(
            *[CrossFormerBlock1D(dim=dim) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Flatten(),
            nn.Linear(dim * num_segments, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (B, num_segments, segment_length)
        x = self.embedding(x)  # → (B, num_segments, dim)
        x = self.blocks(x)     # → (B, num_segments, dim)
        return self.head(x)    # → (B, num_classes)
