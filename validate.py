import os
import re
import torch
import pickle
from tqdm import tqdm
from model.model import TransformerAE
from dataset.dataloader import HDF5Dataset
from torch.utils.data import DataLoader

def reverse_transformation(x, recon, log=False):
    epsilon = 160

    # Clip to [0, 1] just like during training
    x = torch.clamp(x, 0, 1)
    recon = torch.clamp(recon, 0, 1)

    # Load min_max_dict
    if log:
        with open("./all_ranges_no_clouds_log.pkl", "rb") as f:
            min_max_dict = pickle.load(f)
    else:
        with open("./all_ranges_no_clouds_rel.pkl", "rb") as f:
            min_max_dict = pickle.load(f)

    # Create normalization tensors (assumes feature order is preserved)
    min_vals = torch.tensor([v[0] for v in min_max_dict.values()], device=x.device).view(1, 1, -1, 1, 1)
    max_vals = torch.tensor([v[1] for v in min_max_dict.values()], device=x.device).view(1, 1, -1, 1, 1)

    # Unnormalize both input and reconstruction (log-space)
    x = x * (max_vals - min_vals) + min_vals
    recon = recon * (max_vals - min_vals) + min_vals

    # === Step 2: Undo log transform ===
    if log:
        x = torch.exp(x) - epsilon
        recon = torch.exp(recon) - epsilon


    return x, recon

def center_mae(x, recon):
    # Select central coordinate only: day=5, x=y=7
    x_center = x[:, 5, :, 7, 7]  # shape: [B, C]
    recon_center = recon[:, 5, :, 7, 7]  # shape: [B, C]

    # Compute MAE only on valid (masked) elements
    mae_per_element = torch.abs(recon_center - x_center)
    return mae_per_element.mean()


# === SETTINGS ===
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
checkpoint_dir = "/scratch/jpeters/DeepFeatures/checkpoints/149_002_018_080"
#checkpoint_dir = "/scratch/jpeters/DeepFeatures/checkpoints/149_002_018_080_log"
batch_size = 24

# === Load Test Dataset ===
test_dataset = HDF5Dataset("./dataset/test_149.h5")
#test_dataset = HDF5Dataset("/net/data_ssd/deepfeatures/log_test_149.h5")
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

# === Match checkpoint filenames ===
pattern = re.compile(r"ae-7-epoch=(\d+)-val_loss=([0-9.]+e[+-]?\d+)\.ckpt")

# === Track Results ===
results = []

# === Iterate and Evaluate Checkpoints ===
for fname in os.listdir(checkpoint_dir):
    match = pattern.match(fname)
    if match:
        epoch = int(match.group(1))
        if 171 <= epoch <= 171:
            path = os.path.join(checkpoint_dir, fname)

            # Load model
            model = TransformerAE(dbottleneck=7)
            checkpoint = torch.load(path, map_location=device)
            model.load_state_dict(checkpoint['state_dict'])
            model.to(device)
            model.eval()

            # Evaluate on test set using MAE
            total_loss = 0
            count = 0
            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f"Evaluating {fname}", leave=False):
                    x, time_gaps, mask = batch
                    x = x.to(device)
                    time_gaps = time_gaps.to(device)
                    mask = mask.to(device)
                    recon, _ = model(x, time_gaps=time_gaps)

                    x, recon = reverse_transformation(x, recon)

                    #total_loss, mse_loss, ssim_loss, sam_loss, loss = model.loss_fn(recon, x, mask, val=True)  # total MAE
                    #_, _, _, _, loss = model.loss_fn(recon, x, mask, val=True)  # total MAE
                    loss = center_mae(x, recon)

                    num_elements = mask.sum().item()  # or x.numel() if unmasked

                    total_loss += loss.item() * num_elements  # Scale the mean loss back to total error
                    count += num_elements  # Correctly count contributing elements

            mae = total_loss / count
            results.append((fname, mae))
            print(f"{fname} → Test MAE: {mae:.6f}")

# === Report Best ===
best_ckpt, best_mae = min(results, key=lambda x: x[1])
print(f"\n✅ Best checkpoint on test set: {best_ckpt} with MAE: {best_mae:.6f}")
