import torch
from torch import nn as nn
from torch.nn import functional as F
from torchmetrics.image import SpectralAngleMapper


def gaussian_kernel(window_size: int, sigma: float):
    x = torch.arange(window_size).float() - window_size // 2
    gauss = torch.exp(-0.5 * (x / sigma) ** 2)
    return gauss / gauss.sum()

def create_window(window_size: int, channel: int, device):
    _1D_window = gaussian_kernel(window_size, sigma=1.5).unsqueeze(1)
    _2D_window = _1D_window @ _1D_window.T
    window = _2D_window.expand(channel, 1, window_size, window_size).to(device)
    return window


class WeightedMaskedLoss(nn.Module):
    def __init__(self, spatial_size=(15, 15), frames=11, sparsity_weight=0.01, sparsity_target=0.05,
                 lambda_mae=0.33, lambda_ssim=0.02, lambda_sam=0.65, center_mae=1.): #max epoch 65 lambda_mse=0.23, lambda_ssim=0.03, lambda_sam=0.74): #
        super(WeightedMaskedLoss, self).__init__()
        self.spatial_size = spatial_size
        self.sparsity_weight = sparsity_weight
        self.sparsity_target = sparsity_target
        self.frames = frames
        self.lambda_mae = lambda_mae
        self.lambda_ssim = lambda_ssim
        self.lambda_sam = lambda_sam
        self.center_mae = center_mae

        self.weight_map = self.create_weight_map()
        self.sam_metric = SpectralAngleMapper()

    def create_weight_map(self):
        """
        Create a spatial weight map for the spatial dimensions.
        The most central pixel gets a weight of 1, the surrounding pixels get
        progressively smaller weights based on their distance from the center.
        """
        h, w = self.spatial_size
        center = (h // 2, w // 2)

        weight_map = torch.zeros(h, w)
        for i in range(h):
            for j in range(w):
                dist = max(abs(i - center[0]), abs(j - center[1]))
                weight = 1.0 if dist == 0 else 0.01 * (0.1 ** (dist - 1))
                weight_map[i, j] = weight

        return weight_map

    def create_temporal_weights(self, frames):
        """
        Create temporal weights for each frame based on their distance from the central frame.
        """
        center_frame = frames // 2
        temporal_weights = torch.zeros(frames)

        for i in range(frames):
            dist = abs(i - center_frame)
            weight = 1.0 if dist == 0 else 0.1 ** dist
            temporal_weights[i] = weight

        return temporal_weights

    def sparsity_penalty(self, activations):
        activation_mean = torch.mean(activations, dim=0)
        kl_divergence = self.sparsity_target * torch.log(self.sparsity_target / (activation_mean + 1e-10)) + \
                        (1 - self.sparsity_target) * torch.log((1 - self.sparsity_target) / (1 - activation_mean + 1e-10))
        return torch.sum(kl_divergence)


    def ssim_loss(self, img1, img2, window_size=7):
        device = img1.device
        batch, frames, channels, height, width = img1.shape
        img1 = img1.view(batch * frames, channels, height, width)
        img2 = img2.view(batch * frames, channels, height, width)
        #self.ssim_metric = self.ssim_metric.to(img1.device)

        # Compute C1 and C2 dynamically
        L = 1  # Assuming images are normalized between [0,1]
        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2
        window = create_window(window_size, channels, device)

        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channels)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channels)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channels) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        #return self.ssim_metric(img1, img2)
        return 1 - ssim_map.mean()


    def spectral_angle_mapper_loss(self, y_true, y_pred, eps=1e-8):
        dot_product = torch.sum(y_true * y_pred, dim=1)
        norm_true = torch.norm(y_true, p=2, dim=1)
        norm_pred = torch.norm(y_pred, p=2, dim=1)

        cos_theta = torch.clamp(dot_product / (norm_true * norm_pred + eps), -1.0, 1.0)
        return torch.acos(cos_theta).mean()

    def spectral_angle_mapper_loss(self, y_true, y_pred, validity_mask, weight_mask, eps=1e-8):
        """
        Spectral Angle Mapper (SAM) loss
        """
        # Compute dot product & norms over channels
        dot_product = torch.sum(y_true * y_pred, dim=2)  # (B, T, H, W)
        norm_true = torch.norm(y_true, p=2, dim=2)
        norm_pred = torch.norm(y_pred, p=2, dim=2)

        # Compute cosine similarity and angle
        cos_theta = dot_product / (norm_true * norm_pred + eps)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        angle = torch.acos(cos_theta)  # (B, T, H, W)

        # Reduce validity mask over channels (all channels valid at a position)
        valid_mask = validity_mask.all(dim=2)  # (B, T, H, W)

        # Reduce weight mask (all channels share the same weight)
        reduced_weights = weight_mask[:, :, 0, :, :]  # (B, T, H, W)

        # Apply validity mask
        valid_angles = angle[valid_mask]
        valid_weights = reduced_weights[valid_mask]

        # Avoid division by zero
        if valid_weights.sum() == 0:
            return torch.tensor(0.0, device=y_true.device)

        # Compute weighted mean SAM loss
        weighted_loss = (valid_angles * valid_weights).sum() / valid_weights.sum()

        return weighted_loss

    def forward(self, output, target, mask, latent_activations=None, val=False):

        if torch.isnan(output).any():
            print("Output contains NaN values")
            return torch.tensor(0.0, requires_grad=True, device=output.device)
        if torch.isnan(target).any():
            print("Target contains NaN values")
        if not mask.any():
            print("No valid values in the batch")
            return torch.tensor(0.0, requires_grad=True, device=output.device)

        batch_size, frames, indices, h, w = output.size()

        spatial_weight_map = self.weight_map.to(output.device).unsqueeze(0).unsqueeze(0).unsqueeze(2).expand(batch_size, frames, indices, h, w)
        temporal_weights = self.create_temporal_weights(frames).to(output.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        temporal_weight_map = temporal_weights.expand(batch_size, frames, indices, h, w)

        combined_weight_map = spatial_weight_map * temporal_weight_map
        combined_weight_map = combined_weight_map.view(batch_size, frames, indices, h, w)
        combined_weight_map[combined_weight_map == 1.0] = self.center_mae
        combined_weight_map_norm = combined_weight_map / torch.sum(combined_weight_map, dim=(1, 2, 3, 4), keepdim=True)


        masked_output = output[mask]
        masked_target = target[mask]
        masked_weights = combined_weight_map_norm[mask]
        central_time = 5
        central_x = central_y = 7
        central_output = output[:, central_time, :, central_x, central_y]  # Shape: (batch_size, channels)
        central_target = target[:, central_time, :, central_x, central_y]  # Shape: (batch_size, channels)

        center_mae = torch.mean(torch.abs(central_output - central_target))

        if self.lambda_mae <= 0: mse_loss = 0
        else: mse_loss = torch.sum(masked_weights * torch.abs(masked_output - masked_target)) / batch_size
        #print(f'MSE loss: {mse_loss}')
        #mape = torch.sum(masked_weights * torch.abs(masked_output - masked_target)) / (torch.abs(masked_target) + 1e-8) /batch_size

        if self.lambda_ssim <= 0: ssim_loss = 0
        else: ssim_loss = self.ssim_loss(output, target)

        if self.lambda_sam <= 0: sam_loss = 0
        else: sam_loss = self.spectral_angle_mapper_loss(output, target, mask, combined_weight_map)
#
        total_loss = (self.lambda_mae * mse_loss) + (self.lambda_ssim * ssim_loss) + (self.lambda_sam * sam_loss)

        if latent_activations is not None:
            sparsity_loss = self.sparsity_penalty(latent_activations)
            total_loss += self.sparsity_weight * sparsity_loss



        return total_loss, mse_loss, ssim_loss, sam_loss, center_mae
