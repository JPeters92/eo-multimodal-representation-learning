# gpp_lightning_temporal_transformer.py
import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
from typing import Optional

# ---------------------------------------------------------------------
# Temporal Positional Embedding (same logic as your previous models)
# ---------------------------------------------------------------------
class TemporalPositionalEmbedding(nn.Module):
    """
    Learned temporal embedding based on cumulative time gaps.
    Matches your TransformerAE's implementation.
    """
    def __init__(self, d_model: int, max_position: int = 90):
        super().__init__()
        self.d_model = d_model
        self.max_position = max_position
        self.embedding = nn.Embedding(max_position, d_model)

    def forward(self, cumulative_positions: torch.Tensor) -> torch.Tensor:
        """
        cumulative_positions: (B, T) with integer day offsets or cumulative gaps
        returns: (B, T, D)
        """
        positions = torch.clamp(cumulative_positions, 0, self.max_position - 1)
        return self.embedding(positions)


# ---------------------------------------------------------------------
# GPP Transformer Regressor (no custom blocks)
# ---------------------------------------------------------------------
class GPPTemporalTransformer(pl.LightningModule):
    """
    Transformer-based regressor for 90-day GPP prediction.
    Uses learned TemporalPositionalEmbedding instead of sinusoidal PE.
    """

    def __init__(
        self,
        num_features: int = 7,
        seq_len: int = 90,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        pool: str = "last",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_steps: int = 1000,
        warmup_factor:float = 1e-3,
        reduce_patience: int = 5,
        reduce_factor: float = 0.2,
        min_lr: float = 1e-7,
        max_position: int = 512,
        time_first: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_features = num_features
        self.seq_len = seq_len
        self.time_first = time_first
        self.pool = pool
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.warmup_factor = warmup_factor
        self.reduce_patience = reduce_patience
        self.reduce_factor = reduce_factor
        self.min_lr = min_lr

        # --- input projection ---
        self.input_proj = nn.Linear(num_features, d_model)
        self.temporal_embedding = TemporalPositionalEmbedding(d_model, max_position=seq_len)

        # --- transformer encoder ---
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # --- regression head ---
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # --- loss ---
        self.criterion = nn.L1Loss()
        #self.val_criterion = nn.L1Loss()

    # ---------------------------------------------------------
    # Helper: cumulative day embedding from time gaps
    # ---------------------------------------------------------
    def compute_cumulative_positions(self, time_gaps: torch.Tensor) -> torch.Tensor:
        """
        Compute cumulative positions from per-frame gaps.
        time_gaps: (B, T-1)
        returns: (B, T)
        """
        B = time_gaps.size(0)
        cum = torch.zeros((B, self.seq_len), dtype=torch.long, device=time_gaps.device)
        cum[:, 1:] = torch.cumsum(time_gaps, dim=1)
        return cum

    # ---------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------
    def forward(self, X: torch.Tensor, time_gaps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        X: (B, T, F) if time_first=True, else (B, F, T)
        time_gaps: (B, T-1) integer or long tensor
        Returns standardized GPP predictions (B,)
        """
        if not self.time_first:
            X = X.transpose(1, 2)  # (B, T, F)
        X = self.input_proj(X)  # (B, T, D)

        if time_gaps is not None:
            cum_pos = self.compute_cumulative_positions(time_gaps)
            pos_emb = self.temporal_embedding(cum_pos)
            X = X + pos_emb / torch.sqrt(torch.tensor(X.size(-1), dtype=torch.float, device=X.device))

        x_enc = self.encoder(X)

        if self.pool == "last":
            pooled = x_enc[:, -1, :]
        else:
            pooled = x_enc.mean(dim=1)

        y_hat = self.head(pooled).squeeze(-1)
        return y_hat

    # ---------------------------------------------------------
    # Lightning training/validation
    # ---------------------------------------------------------
    def training_step(self, batch, batch_idx):
        # expected batch: (X, y) or (X, y, meta)
        X, y = batch[:2]
        y_hat = self(X)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss,
                 on_step=True, on_epoch=True, prog_bar=True,
                 batch_size=X.size(0))  # <= tell Lightning the batch size here
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch[:2]
        y_hat = self(X)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss,
                 on_epoch=True, prog_bar=True,
                 batch_size=X.size(0))  # <= and here
        return loss

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        warmup = {
            "scheduler": LinearLR(opt, start_factor=self.warmup_factor, total_iters=self.warmup_steps),  # shorter warmup helps small datasets
            "interval": "step",
            "frequency": 1,
            "name": "linear_warmup",
        }
        plateau = {
            "scheduler": ReduceLROnPlateau(
                opt, mode="min", factor=self.reduce_factor, patience=self.reduce_patience,
                min_lr=self.min_lr, verbose=False
            ),
            "monitor": "val_loss",   # <- monitor val_loss, not val_mse
            "interval": "epoch",
            "name": "plateau",
        }
        return [opt], [warmup, plateau]


