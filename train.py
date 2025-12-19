import torch
import random
import lightning.pytorch as L
from model.model_s1_s2 import TransformerAE
from dataset.dataloader import HDF5Dataset
from torch.utils.data import DataLoader
from model.model_fusion import FusedS1S2, load_enc_dec_from_ae_ckpt
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import EarlyStopping

torch.set_float32_matmul_precision('medium')  # For better performance


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # Define the parameters
    batch_size = 16

    train_dataset = HDF5Dataset("./train_s2.h5")
    #train_dataset = HDF5Dataset("./train_s1_s2.h5", s1_s2=True)

    val_dataset = HDF5Dataset("./val_s2.h5")
    #val_dataset = HDF5Dataset("./val_s1_s2.h5", s1_s2=True)

    # Create DataLoaders
    val_iterator   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=24)
    #test_iterator  = DataLoader(test_split, batch_size=batch_size, shuffle=False, num_workers=32)
    train_iterator = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=24)

    # Initialize the autoencoder model
    dbottleneck = 9
    #dbottleneck = 2
    #dbottleneck = 7
    channels = 10
    #channels = 2
    #channels = 12
    #channels = 149

    # Instantiate the model
    model = TransformerAE(dbottleneck=dbottleneck, channels=channels, num_reduced_tokens=7)
    #model = TransformerAE(dbottleneck=dbottleneck, channels=channels, num_reduced_tokens=5)
    #model.apply(hierarchical_initialize_weights)
    #model.to(device)  # Move the model to GPU 3
    #model.reset_parameters()
    #print('====================')

    #batch_size = 2
    #frames_s1 = 11  # Anzahl Zeitpunkte Sentinel-1
    #frames_s2 = 11  # Anzahl Zeitpunkte Sentinel-2
    #chans_s1 = 2
    #chans_s2 = 10
    #dbottleneck_s1 = 2
    #dbottleneck_s2 = 9
#
    #s2_ckpt = './checkpoints/003_025_072_test/s2_3/ae-9-epoch=154-val_loss=3.686e-03.ckpt'
    ##s1_ckpt = './checkpoints/003_025_072_test/s1/ae-2-epoch=68-val_loss=6.832e-04.ckpt'
    #s1_ckpt = './checkpoints/s1_3/ae-2-epoch=65-val_loss=6.460e-04.ckpt'
#
    ##s1_d2_ckpt = './checkpoints/003_025_072_test/ss1_s2/ae-9-epoch=133-val_loss=2.622e-03.ckpt'
    #s1_d2_ckpt = './checkpoints/s2/ae-7-epoch=113-val_loss=2.161e-03.ckpt'
    #s1_d2_ckpt = './checkpoints/s1_s2_1/ae-7-epoch=72-val_loss=3.417e-03.ckpt'
    ##s1_d2_ckpt = './checkpoints/003_025_072_test/s1_s2_2/ae-9-epoch=93-val_loss=2.245e-03.ckpt'
#
    #enc_s1, dec_s1 = load_enc_dec_from_ae_ckpt(
    #    device=device,
    #    ckpt_path=s1_ckpt,
    #    #ckpt_path=None,
    #    channels=chans_s1,
    #    dbottleneck=dbottleneck_s1,
    #    num_reduced_tokens=7
    #)
    #enc_s2, dec_s2 = load_enc_dec_from_ae_ckpt(
    #    device=device,
    #    ckpt_path=s2_ckpt,
    #    #ckpt_path=None,
    #    channels=chans_s2,
    #    dbottleneck=dbottleneck_s2,
    #    num_reduced_tokens=6
    #)
#
    ##print('====================')
#
    ## Modell instanziieren
    #model = FusedS1S2(
    #    enc_s1=enc_s1, dec_s1=dec_s1,
    #    enc_s2=enc_s2, dec_s2=dec_s2,
    #    dbottleneck_s1=dbottleneck_s1,
    #    dbottleneck_s2=dbottleneck_s2,
    #    freeze_encoders=False,
    #    dbottleneck=7 #dbottleneck
    #)


    #checkpoint = torch.load(s1_d2_ckpt, map_location=device)
    #model.load_state_dict(checkpoint['state_dict'])
    model.to(device)#"""

    # Define checkpointing (optional)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        #dirpath='checkpoints_norm_rel_SI/148_000_100_000_2/',
        #dirpath='checkpoints_norm_rel_SI/148_w_sam/002_095_039/',
        #dirpath='checkpoints/149_002_018_080_vc/',
        #dirpath='checkpoints/149_002_018_080_test/3_3/',
        #dirpath='checkpoints/002_018_080_test/2/',
        dirpath='checkpoints/s2/',
        #filename='foundation-ae-{epoch:02d}-{val_loss:.3e}',
        filename=f'ae-{7}-{{epoch:02d}}-{{val_loss:.3e}}',
        save_top_k=5,
        mode='min'
    )

    # Define early stopping
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',  # Metric to monitor
        patience=16,  # Number of epochs to wait for improvement
        mode='min',  # Stop if the metric stops decreasing
        verbose=True,  # Print early stopping message
    )

    # Initialize the PyTorch Lightning trainer
    trainer = L.Trainer(
        max_epochs=500,
        gradient_clip_val=1.0,
        #log_every_n_steps=10,
        enable_progress_bar=True,
        check_val_every_n_epoch=1,  # Ensure validation is run every epoch
        devices=[1],
        accelerator="gpu",
        callbacks=[checkpoint_callback, early_stopping_callback],
    )


    # Train the model using your custom train and validation iterators
    with torch.autograd.set_detect_anomaly(True):
        trainer.fit(model, train_dataloaders=train_iterator, val_dataloaders=val_iterator)

if __name__ == "__main__":
    main()
# nohup python train.py > train3_3.log 2>&1 &