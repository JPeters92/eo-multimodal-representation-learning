from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
import torch.optim as optim
import lightning.pytorch as pl
try:
    from model.attention import TemporalPositionalEmbedding
    from model.loss import WeightedMaskedLoss
    from model.model_blocks import *
except:
    from attention import TemporalPositionalEmbedding
    from loss import WeightedMaskedLoss
    from model_blocks import *

import torch
import torch.nn as nn


def compute_cumulative_positions(time_gaps):
    # Initialize cumulative positions with zeros for each sample in the batch
    batch_size, frames_minus_one = time_gaps.size()
    cumulative_positions = torch.zeros((batch_size, frames_minus_one + 1), dtype=torch.long, device=time_gaps.device)

    # Fill cumulative positions by computing the cumulative sum for each batch sample
    cumulative_positions[:, 1:] = torch.cumsum(time_gaps, dim=1)

    return cumulative_positions

class ModalityEncoder(nn.Module):
    def __init__(self, in_channels=12, dbottleneck=6, positional_embedding=None, transformer_emc=None, frames=11, d_model=64*4, num_reduced_tokens=4):
        super().__init__()
        self.frames=frames

        self.dim_reducer = MultiScaleDimensionalityReducer_12(in_channels=in_channels, out_channels=int(
            d_model / 16))  # in_channels=149, reduction_ratio=8)

        # shared positional embedding (can be None for ablation)
        self.positional_embedding_shared = positional_embedding
        self.transformer_enc = transformer_emc

        self.encoder_linear = nn.Sequential(
            nn.Linear(d_model, 16),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True) ,  # Activation after linear layer
            nn.SELU()
        )

        self.token_reducer = nn.Sequential(
            nn.Linear(self.frames, num_reduced_tokens),
            # nn.LeakyReLU(negative_slope=0.1, inplace=True),  # Activation for token reduction
            nn.SELU()
        )

        self.bottleneck_reducer = nn.Sequential(
            nn.Linear(16 * num_reduced_tokens, dbottleneck),
            nn.Softplus(beta=1, threshold=20)
        )

    def forward(self, x, time_gaps):
        x = self.dim_reducer(x)  # Expected shape: (batch_size, frames, 15, 15, 64)
        x = x.reshape(x.size(0), self.frames, -1)  # Expected shape: (batch_size, frames, din)
        # Add Positional embeddings & Pass through Transformer encoder
        cumulative_positions = compute_cumulative_positions(time_gaps)  # Shape: (batch_size, frames)
        pos_emb = self.positional_embedding_shared(cumulative_positions)  # Shape: (batch_size, frames, d2)
        pos_emb = pos_emb / torch.sqrt(torch.tensor(pos_emb.size(-1), dtype=torch.float))
        x = x + pos_emb  # Shape should match (frames, batch_size, d2)
        x = self.transformer_enc(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.encoder_linear(x)

        x = self.token_reducer(x.permute(0, 2, 1))

        x = x.reshape(x.size(0), -1)
        x = self.bottleneck_reducer(x)
        return x


class ModalityDecoder(nn.Module):
    def __init__(self, out_channels=12, dbottleneck=6, positional_embedding=None, transformer_dec=None, frames=11, d_model=64*4, num_reduced_tokens=4):
        super().__init__()
        self.frames = frames
        self.d_model = d_model
        self.num_reduced_tokens = num_reduced_tokens

        self.upscaler = MultiScaleAttentionUpscaler_12(out_channels=out_channels, in_channels=int(
            d_model / 16))  # in_channels=8, out_channels=149)

        # shared positional embedding (can be None for ablation)
        self.positional_embedding_shared = positional_embedding
        self.transformer_dec = transformer_dec #3 # Number of encoder layers

        self.decoder_linear = nn.Sequential(
            nn.Linear(16, d_model),
            #nn.LeakyReLU(negative_slope=0.1, inplace=True) ,  # Activation after linear layer
            nn.SELU()
        )

        self.token_upsampler = nn.Sequential(
            nn.Linear(num_reduced_tokens, self.frames),
            #nn.LeakyReLU(negative_slope=0.01, inplace=True),   # Activation for token upsampling
            nn.SELU()
        )

        self.bottleneck_expander = nn.Sequential(
            nn.Linear(dbottleneck, 16 * num_reduced_tokens),
            nn.Softplus(beta=1, threshold=20)
        )

    def forward(self, x, time_gaps):
        x = self.bottleneck_expander(x)
        x = x.reshape(x.size(0), self.num_reduced_tokens, 16)
        x = self.token_upsampler(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.decoder_linear(x)
        cumulative_positions = compute_cumulative_positions(time_gaps)  # Shape: (batch_size, frames)
        pos_emb = self.positional_embedding_shared(cumulative_positions)  # Shape: (frames, batch_size, d2)
        pos_emb = pos_emb/ torch.sqrt(torch.tensor(pos_emb.size(-1), dtype=torch.float))
        x = x + pos_emb
        x = self.transformer_dec(x)
        x = x.reshape(x.size(0) * self.frames, self.d_model // 16, 4, 4)
        x = self.upscaler(x)  # Shape: (batch_size, frames, 15, 15, 209)
        return x


class TransformerAE(pl.LightningModule):
    """
    Autoencoder LightningModule for the num_dims=10 (12-channel) path.

    Expects batches like:
        x:         (B, T, 12, 15, 15)
        time_gaps: (B, T-1)  integer gaps (days)
        mask:      same shape as x (bool) or broadcastable

    Forward returns:
        decoded: (B, T, 12, 15, 15)
        encoded: (B, dbottleneck)
    """
    def __init__(
        self,
        dbottleneck: int = 6,
        channels: int = 10,
        frames: int = 11,
        max_position: int = 200,
        d_model: int = 64 * 4,          # 256
        num_reduced_tokens: int = 4,
        dim_feedforward: int = 8,   # encoder FFN size (matches your num_dims=10 branch)
        learning_rate: float = 1e-4,
        loss_fn: nn.Module = None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['loss_fn'])  # keeps config in checkpoints

        self.frames = frames
        self.d_model = d_model
        self.num_reduced_tokens = num_reduced_tokens
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn if loss_fn is not None else WeightedMaskedLoss(lambda_sam=0., lambda_ssim=0., lambda_mae=1.)

        # Shared temporal positional embedding
        self.positional_embedding_shared = TemporalPositionalEmbedding(
            d_model=self.d_model, max_position=max_position
        )

        transformer_shared = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=dim_feedforward,
                dropout=0.25,
                batch_first=True  # Ensure batch-first data layout
            ),
            num_layers=3  # 3 # Number of encoder layers
        )

        # Encoder / Decoder (num_dims=10 path)
        self.encoder = ModalityEncoder(
            dbottleneck=dbottleneck,
            in_channels=channels,
            positional_embedding=self.positional_embedding_shared,
            transformer_emc=transformer_shared,
            frames=frames,
            d_model=d_model,
            num_reduced_tokens=num_reduced_tokens

        )

        self.decoder = ModalityDecoder(
            dbottleneck=dbottleneck,
            out_channels=channels,
            frames=frames,
            positional_embedding=self.positional_embedding_shared,
            transformer_dec=transformer_shared,
            d_model=d_model,
            num_reduced_tokens=num_reduced_tokens
        )

    def forward(self, x: torch.Tensor, time_gaps: torch.Tensor):
        """
        x:         (B, T, 12, 15, 15)
        time_gaps: (B, T-1)
        """
        z = self.encoder(x, time_gaps)              # (B, dbottleneck)
        decoded = self.decoder(z, time_gaps)        # (B, T, 12, 15, 15)
        return decoded, z

    # --------- Training / Validation / Test steps ---------
    def training_step(self, batch, batch_idx):
        x, time_gaps, mask = batch
        decoded, z = self(x, time_gaps)

        total_loss, mse_loss, ssim_loss, sam_loss, center_mae = self.loss_fn(decoded, x, mask)

        self.log_dict(
            {
                "train_total": total_loss,
                "train_mae": mse_loss,
                "train_ssim": ssim_loss,
                "train_sam": sam_loss,
                "train_center": center_mae,
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )

        # optional: log LR safely
        try:
            opt = self.optimizers()
            if opt is not None and len(opt.param_groups) > 0:
                self.log("lr", opt.param_groups[0]["lr"], prog_bar=True, on_step=True)
        except Exception:
            pass

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, time_gaps, mask = batch
        decoded, z = self(x, time_gaps)
        total_loss, mse_loss, ssim_loss, sam_loss, center_mae = self.loss_fn(decoded, x, mask, val=True)

        # monitor "val_loss" for ReduceLROnPlateau (kept same naming as your previous class)
        self.log_dict(
            {
                "val_total": total_loss,
                "val_mae": mse_loss,
                "val_ssim": ssim_loss,
                "val_sam": sam_loss,
                "val_loss": center_mae,
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )
        return center_mae

    def test_step(self, batch, batch_idx):
        x, time_gaps, mask = batch
        decoded, z = self(x, time_gaps)
        total_loss, mse_loss, ssim_loss, sam_loss, center_mae = self.loss_fn(decoded, x, mask, val=True)
        self.log_dict(
            {"test_total": total_loss, "test_mae": mse_loss, "test_ssim": ssim_loss, "test_sam": sam_loss, "test_center": center_mae},
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )
        return total_loss

    # --------- Optimizers / Schedulers ---------
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1.5e-4)

        warmup_scheduler = {
            "scheduler": LinearLR(optimizer, start_factor=0.001, total_iters=35000),
            "interval": "step",
            "frequency": 1,
        }

        plateau_scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.35,
                patience=5,
                verbose=False,
                min_lr=1.5e-6,
                threshold=1.5e-6,
                threshold_mode="abs",
            ),
            "interval": "epoch",
            "monitor": "val_loss",
        }

        return [optimizer], [warmup_scheduler, plateau_scheduler]



def main():
    # Define model parameters
    frames = 11  # Number of frames we want to consider
    batch_size = 2  # Batch size for dummy data
    channels = 10

    # Instantiate the model
    model = TransformerAE(dbottleneck=7, channels=channels)

    # Generate dummy input data
    x = torch.randn(batch_size, frames, channels, 15, 15)

    # Generate dummy time gaps for each sample in the batch (shape: (batch_size, frames - 1))
    time_gaps = torch.tensor([
        [2, 5, 3, 2, 4, 2, 2, 4, 5, 3],  # Time gaps for the first sample in the batch
        [1, 3, 2, 4, 3, 5, 1, 2, 4, 3]  # Time gaps for the second sample in the batch
    ], dtype=torch.long)

    mask = torch.ones_like(x, dtype=torch.bool)
    # Ensure the model is in evaluation mode
    model.eval()

    # Forward pass
    with torch.no_grad():
        decoded, _ = model(x, time_gaps)
        print('==============')
        #print(output.shape)

    # Output result
    print("Input shape:", x.shape)
    print("Time gaps shape:", time_gaps.shape)
    print("Output shape:", decoded.shape)

    print(model.loss_fn(decoded, x, mask))



if __name__ == "__main__":
    main()