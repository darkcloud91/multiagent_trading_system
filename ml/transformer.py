# trading_system/ml/transformer.py

import torch
import torch.nn as nn

class PositionalEmbeddingLearnable(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.pe.weight, mean=0.0, std=0.02)

    def forward(self, x):
        B, T, D = x.size()
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        return self.pe(pos_ids)

class TimeSeriesTransformer(nn.Module):
    """
    Encoder Transformer con:
      - Proyección de entrada + dropout
      - Positional Embedding aprendible
      - Token [CLS]
      - Fusión CLS + mean-pooling
      - (Nuevo) Embedding por activo (asset_ids)
    """
    def __init__(
        self,
        input_size: int,
        num_classes: int = 3,          # 3 clases por los umbrales
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_ff: int = 512,
        dropout: float = 0.2,
        max_len: int = 1024,
        num_assets: int | None = None,  # NUEVO
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes

        self.input_proj = nn.Linear(input_size, d_model)
        self.inp_dropout = nn.Dropout(dropout)

        self.pos_emb = PositionalEmbeddingLearnable(max_len=max_len+1, d_model=d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # NUEVO: embedding de activo (mismo D para poder sumar)
        self.asset_emb = nn.Embedding(num_assets, d_model) if num_assets is not None else None
        if self.asset_emb is not None:
            nn.init.normal_(self.asset_emb.weight, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x, asset_ids: torch.Tensor | None = None):
        """
        x: (B, T, F)
        asset_ids: (B,) o None
        """
        B, T, F = x.size()
        h = self.input_proj(x)         # (B, T, D)
        h = self.inp_dropout(h)

        if self.asset_emb is not None and asset_ids is not None:
            a = self.asset_emb(asset_ids)        # (B, D)
            h = h + a.unsqueeze(1)               # broadcast por tiempo

        pe = self.pos_emb(h)                     # (B, T, D)

        cls = self.cls_token.expand(B, 1, -1)    # (B, 1, D)
        pe0 = self.pos_emb.pe.weight[0].unsqueeze(0).unsqueeze(0)
        h = torch.cat([cls, h], dim=1)           # (B, T+1, D)
        pe = torch.cat([pe0.expand(B, 1, -1), pe], dim=1)

        h = h + pe
        h = self.encoder(h)

        cls_vec = h[:, 0, :]
        mean_vec = h[:, 1:, :].mean(dim=1)
        fused = 0.5 * cls_vec + 0.5 * mean_vec

        fused = self.norm(fused)
        logits = self.head(fused)
        return logits