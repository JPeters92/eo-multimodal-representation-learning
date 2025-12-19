import torch.nn as nn
try:
    from model.attention import MultiScaleAttentionBlock, ConvAttentionBlock
except:
    from attention import MultiScaleAttentionBlock, ConvAttentionBlock



class MultiScaleDimensionalityReducer_221(nn.Module):
    def __init__(self, in_channels=221, reduction_ratio=8, dropout_prob=0.07):
        super(MultiScaleDimensionalityReducer_221, self).__init__()

        self.reduced_channels=4
        # Block 1: Multi-scale attention with CBAM and spatial downsampling
        self.block1 = MultiScaleAttentionBlock(
            in_channels, in_channels, kernel_sizes=[3, 7], reduction_ratio=reduction_ratio
        )
        #self.conv_res = nn.Conv3d(in_channels=in_channels, out_channels=4, kernel_size=(1, 4, 4), stride=(1, 3, 3))
        self.pool =nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=2, stride=2),
            nn.SiLU()  # Add SiLU here
        )

        self.dropout1 = nn.Dropout2d(p=dropout_prob)

        # Block 2: Further reduction
        #self.block2 = MultiScaleAttentionBlock(
        #    64, 64, kernel_sizes=[3], reduction_ratio=reduction_ratio
        #)
        self.block2 = ConvAttentionBlock(
            128, 128, kernel_size=3, reduction_ratio=reduction_ratio
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=2, stride=2, padding=1),  # Downsample to ~5x5
            nn.SiLU()  # Add SiLU here
        )

        self.dropout2 = nn.Dropout2d(p=dropout_prob)

        # Final block for dimensionality reduction
        #self.block3 = MultiScaleAttentionBlock(
        #    4, 4, kernel_sizes=[3], reduction_ratio=2
        #)
        self.block3 = ConvAttentionBlock(
            64, self.reduced_channels, kernel_size=3, reduction_ratio=2, attention_kernel=3
        )



    def forward(self, x):
        # Input: (batch_size, frames, height=15, width=15, spectral_indices=209)
        x = x.view(-1, 221, 15, 15)

        y = x
        # Block 1
        x = self.block1(x)
        #print(x.shape)
        x = self.pool(x)
        x = x + y
        #z = self.conv_res(x)

        x = self.conv1(x)
        #print(x.shape)

        x = self.dropout1(x)

        # Block 2
        y = x
        x = self.block2(x)
        x = self.pool(x)
        x = x + y
        #print(x.shape)

        x = self.conv2(x)
        #print(x.shape)

        x = self.dropout2(x)

        # Block 3
        x = self.block3(x)
        #print(x.shape)

        #x = x + z
        #x = self.conv3(x)
        x = x.view(-1, 11, self.reduced_channels, 4, 4)
        return x


class MultiScaleAttentionUpscaler_221(nn.Module):
    def __init__(self, in_channels=4, out_channels=221, reduction_ratio=8, dropout_prob=0.1):
        super(MultiScaleAttentionUpscaler_221, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1, stride=1),
            nn.SiLU()  # Add SiLU here
        )
        # Block 1: Multi-scale attention with CBAM and upsampling
        #self.block1 = MultiScaleAttentionBlock(
        #    in_channels, in_channels, kernel_sizes=[3], reduction_ratio=reduction_ratio
        #)
        self.block1 = ConvAttentionBlock(
            16, 16, kernel_size=3, reduction_ratio=reduction_ratio
        )
        self.pool =nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        #self.conv_res = nn.ConvTranspose3d(in_channels=in_channels, out_channels=128, kernel_size=(1, 6, 6), stride=(1, 3, 3))

        #self.conv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=64, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=64, kernel_size=2, stride=2),
            nn.SiLU()  # Add SiLU here
        )
        #self.upsample1 = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=True)
        self.dropout1 = nn.Dropout2d(p=dropout_prob)

        # Block 2: Multi-scale attention with CBAM and further upsampling
        #self.block2 = MultiScaleAttentionBlock(
        #    64, 64, kernel_sizes=[3], reduction_ratio=reduction_ratio
        #)
        self.block2 = ConvAttentionBlock(
            64, 64, kernel_size=3, reduction_ratio=reduction_ratio
        )
        #self.upsample2 = nn.Upsample(scale_factor=(1, 1.9, 1.9), mode="trilinear", align_corners=True)
        self.upsample2 = nn.ConvTranspose2d(64, 128, kernel_size=2, stride=2, padding=1, output_padding=1)
        self.dropout2 = nn.Dropout2d(p=dropout_prob)

        # Final Block: Full reconstruction
        self.block3 = MultiScaleAttentionBlock(
            128, out_channels, kernel_sizes=[3, 7], reduction_ratio=reduction_ratio
        )

        #self.final_refinement = nn.Sequential(
        #    nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=1),
        #    nn.SiLU(),
        #    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        #)
        #self.final_refinement = nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        # Input: (batch_size, reduced_channels=12, frames, height=5, width=5)

        # Block 1
        #print('_____________________')

        x = self.conv0(x)
        #print(x.shape)
        y=x
        x = self.block1(x)
        x = self.pool(x)
        x = x + y
        #z = self.conv_res(x)
        #print(x.shape)
        #print(z.shape)
        x = self.conv1(x)  # Upsample to ~10x10
        #print(x.shape)
        x = self.dropout1(x)

        # Block 2
        y=x
        x = self.block2(x)
        x = self.pool(x)
        x = x + y
        x = self.upsample2(x)  # Upsample to ~15x15
        x = self.dropout2(x)
        #x = x+z

        # Block 3
        #y=x
        x = self.block3(x)
        #x=self.pool(x)
        #x = x + y
        #print(x.shape)
        #x = self.final_refinement(x)

        # Change back to original format
        x = x.view(-1, 11, 221, 15, 15)

        # Output shape: (batch_size, frames, height=15, width=15, out_channels=209)
        return x



class MultiScaleDimensionalityReducer_149(nn.Module):
    def __init__(self, in_channels=149, out_channels = 4*4, reduction_ratio=8, dropout_prob=0.07):
        super(MultiScaleDimensionalityReducer_149, self).__init__()

        self.reduced_channels=out_channels
        # Block 1: Multi-scale attention with CBAM and spatial downsampling
        self.block1 = MultiScaleAttentionBlock(
            in_channels, in_channels, kernel_sizes=[3, 7], reduction_ratio=reduction_ratio
        )
        #self.conv_res = nn.Conv3d(in_channels=in_channels, out_channels=4, kernel_size=(1, 4, 4), stride=(1, 3, 3))
        self.pool =nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=2, stride=2),
            nn.SiLU()  # Add SiLU here
        )

        self.dropout1 = nn.Dropout2d(p=dropout_prob)

        # Block 2: Further reduction
        #self.block2 = MultiScaleAttentionBlock(
        #    64, 64, kernel_sizes=[3], reduction_ratio=reduction_ratio
        #)
        self.block2 = ConvAttentionBlock(
            128, 128, kernel_size=3, reduction_ratio=reduction_ratio
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=2, stride=2, padding=1),  # Downsample to ~5x5
            nn.SiLU()  # Add SiLU here
        )

        self.dropout2 = nn.Dropout2d(p=dropout_prob)

        # Final block for dimensionality reduction
        #self.block3 = MultiScaleAttentionBlock(
        #    4, 4, kernel_sizes=[3], reduction_ratio=2
        #)
        self.block3 = ConvAttentionBlock(
            64, self.reduced_channels, kernel_size=3, reduction_ratio=2, attention_kernel=3
        )



    def forward(self, x):
        # Input: (batch_size, frames, height=15, width=15, spectral_indices=209)
        #x = x.view(-1, 149, 15, 15)
        x = x.reshape(-1, 149, 15, 15)

        y = x
        # Block 1
        x = self.block1(x)
        #print(x.shape)
        x = self.pool(x)
        x = x + y
        #z = self.conv_res(x)

        x = self.conv1(x)
        #print(x.shape)

        x = self.dropout1(x)

        # Block 2
        y = x
        x = self.block2(x)
        x = self.pool(x)
        x = x + y
        #print(x.shape)

        x = self.conv2(x)
        #print(x.shape)

        x = self.dropout2(x)

        # Block 3
        x = self.block3(x)
        #print(x.shape)

        #x = x + z
        #x = self.conv3(x)
        x = x.view(-1, 11, self.reduced_channels, 4, 4)
        return x


class MultiScaleAttentionUpscaler_149(nn.Module):
    def __init__(self, in_channels=16, out_channels=149, reduction_ratio=8, dropout_prob=0.1):
        super(MultiScaleAttentionUpscaler_149, self).__init__()

        #self.conv0 = nn.Sequential(
        #    nn.Conv2d(4, 16, kernel_size=1, stride=1),
        #    nn.SiLU()  # Add SiLU here
        #)
        # Block 1: Multi-scale attention with CBAM and upsampling
        #self.block1 = MultiScaleAttentionBlock(
        #    in_channels, in_channels, kernel_sizes=[3], reduction_ratio=reduction_ratio
        #)
        self.block1 = ConvAttentionBlock(
            in_channels=in_channels, out_channels=in_channels, kernel_size=3, reduction_ratio=reduction_ratio
        )
        self.pool =nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        #self.conv_res = nn.ConvTranspose3d(in_channels=in_channels, out_channels=128, kernel_size=(1, 6, 6), stride=(1, 3, 3))

        #self.conv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=64, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=64, kernel_size=2, stride=2),
            nn.SiLU()  # Add SiLU here
        )
        #self.upsample1 = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=True)
        self.dropout1 = nn.Dropout2d(p=dropout_prob)

        # Block 2: Multi-scale attention with CBAM and further upsampling
        #self.block2 = MultiScaleAttentionBlock(
        #    64, 64, kernel_sizes=[3], reduction_ratio=reduction_ratio
        #)
        self.block2 = ConvAttentionBlock(
            64, 64, kernel_size=3, reduction_ratio=reduction_ratio
        )
        #self.upsample2 = nn.Upsample(scale_factor=(1, 1.9, 1.9), mode="trilinear", align_corners=True)
        self.upsample2 = nn.ConvTranspose2d(64, 128, kernel_size=2, stride=2, padding=1, output_padding=1)
        self.dropout2 = nn.Dropout2d(p=dropout_prob)

        # Final Block: Full reconstruction
        self.block3 = MultiScaleAttentionBlock(
            128, out_channels, kernel_sizes=[3, 7], reduction_ratio=reduction_ratio
        )

        #self.final_refinement = nn.Sequential(
        #    nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=1),
        #    nn.SiLU(),
        #    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        #)
        #self.final_refinement = nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        # Input: (batch_size, reduced_channels=12, frames, height=5, width=5)

        # Block 1
        #print('_____________________')
        #print(x.shape)

        #x = self.conv0(x)
        #print(x.shape)
        y=x
        x = self.block1(x)
        x = self.pool(x)
        x = x + y
        #z = self.conv_res(x)
        #print(x.shape)
        #print(z.shape)
        x = self.conv1(x)  # Upsample to ~10x10
        #print(x.shape)
        x = self.dropout1(x)

        # Block 2
        y=x
        x = self.block2(x)
        x = self.pool(x)
        x = x + y
        x = self.upsample2(x)  # Upsample to ~15x15
        x = self.dropout2(x)
        #x = x+z

        # Block 3
        #y=x
        x = self.block3(x)
        #x=self.pool(x)
        #x = x + y
        #print(x.shape)
        #x = self.final_refinement(x)

        # Change back to original format
        x = x.view(-1, 11, 149, 15, 15)

        # Output shape: (batch_size, frames, height=15, width=15, out_channels=209)
        return x



class MultiScaleAttentionUpscaler_12(nn.Module):
    def __init__(self, in_channels=16, out_channels=12, reduction_ratio=8, dropout_prob=0.1):
        super(MultiScaleAttentionUpscaler_12, self).__init__()

        #self.conv0 = nn.Sequential(
        #    nn.Conv2d(4, 16, kernel_size=1, stride=1),
        #    nn.SiLU()  # Add SiLU here
        #)
        # Block 1: Multi-scale attention with CBAM and upsampling
        self.block1 = ConvAttentionBlock(
            in_channels=in_channels, out_channels=in_channels, kernel_size=3, reduction_ratio=reduction_ratio
        )
        self.pool =nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        #self.conv_res = nn.ConvTranspose3d(in_channels=in_channels, out_channels=128, kernel_size=(1, 6, 6), stride=(1, 3, 3))

        #self.conv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=64, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=64, kernel_size=2, stride=2),
            nn.SiLU()  # Add SiLU here
        )
        #self.upsample1 = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=True)
        self.dropout1 = nn.Dropout2d(p=dropout_prob)

        # Block 2: Multi-scale attention with CBAM and further upsampling
        #self.block2 = MultiScaleAttentionBlock(
        #    64, 64, kernel_sizes=[3], reduction_ratio=reduction_ratio
        #)
        self.block2 = ConvAttentionBlock(
            64, 64, kernel_size=3, reduction_ratio=reduction_ratio
        )
        #self.upsample2 = nn.Upsample(scale_factor=(1, 1.9, 1.9), mode="trilinear", align_corners=True)
        self.upsample2 = nn.ConvTranspose2d(64, 128, kernel_size=2, stride=2, padding=1, output_padding=1)
        self.dropout2 = nn.Dropout2d(p=dropout_prob)

        # Final Block: Full reconstruction
        self.block3 = MultiScaleAttentionBlock(
            128, out_channels, kernel_sizes=[1, 3, 7], reduction_ratio=reduction_ratio
        )

        #self.final_refinement = nn.Sequential(
        #    nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=1),
        #    nn.SiLU(),
        #    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        #)
        self.final_refinement = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        # Input: (batch_size, reduced_channels=12, frames, height=5, width=5)

        # Block 1
        #print('_____________________')
        #print(x.shape)

        #x = self.conv0(x)
        #print(x.shape)
        y=x
        x = self.block1(x)
        x = self.pool(x)
        x = x + y
        #z = self.conv_res(x)
        #print(x.shape)
        #print(z.shape)
        x = self.conv1(x)  # Upsample to ~10x10
        #print(x.shape)
        x = self.dropout1(x)

        # Block 2
        y=x
        x = self.block2(x)
        x = self.pool(x)
        x = x + y
        x = self.upsample2(x)  # Upsample to ~15x15
        x = self.dropout2(x)
        #x = x+z

        # Block 3
        #y=x
        x = self.block3(x)
        #x=self.pool(x)
        #x = x + y
        #print(x.shape)
        #x = self.final_refinement(x)

        # Change back to original format
        x = x.view(-1, 11, 12, 15, 15)

        # Output shape: (batch_size, frames, height=15, width=15, out_channels=209)
        return x


class MultiScaleDimensionalityReducer_12(nn.Module):
    def __init__(self, in_channels=12, out_channels=4 * 4, reduction_ratio=8, dropout_prob=0.07):
        super(MultiScaleDimensionalityReducer_12, self).__init__()

        self.reduced_channels = out_channels
        # Block 1: Multi-scale attention with CBAM and spatial downsampling
        self.block1 = MultiScaleAttentionBlock(
            in_channels, in_channels, kernel_sizes=[1, 3, 7], reduction_ratio=reduction_ratio
        )
        # self.conv_res = nn.Conv3d(in_channels=in_channels, out_channels=4, kernel_size=(1, 4, 4), stride=(1, 3, 3))
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=2, stride=2),
            nn.SiLU()  # Add SiLU here
        )

        self.dropout1 = nn.Dropout2d(p=dropout_prob)

        # Block 2: Further reduction
        # self.block2 = MultiScaleAttentionBlock(
        #    64, 64, kernel_sizes=[3], reduction_ratio=reduction_ratio
        # )
        self.block2 = ConvAttentionBlock(
            64, 64, kernel_size=3, reduction_ratio=reduction_ratio
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),  # Downsample to ~5x5
            nn.SiLU()  # Add SiLU here
        )

        self.block3 = ConvAttentionBlock(
            128, 128, kernel_size=3, reduction_ratio=reduction_ratio
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=2, stride=2, padding=1),  # Downsample to ~5x5
            nn.SiLU()  # Add SiLU here
        )

        self.dropout2 = nn.Dropout2d(p=dropout_prob)

        # Final block for dimensionality reduction
        # self.block3 = MultiScaleAttentionBlock(
        #    4, 4, kernel_sizes=[3], reduction_ratio=2
        # )
        self.block3 = ConvAttentionBlock(
            64, 64, kernel_size=3, reduction_ratio=2, attention_kernel=3
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, self.reduced_channels, kernel_size=1, stride=1, padding=0),  # Downsample to ~5x5
            nn.SiLU()  # Add SiLU here
        )

    def forward(self, x):
        # Input: (batch_size, frames, height=15, width=15, spectral_indices=209)

        x = x.view(-1, 12, 15, 15)
        # x = x.reshape(-1, 149, 15, 15)

        y = x
        # Block 1
        x = self.block1(x)
        # print(x.shape)
        x = self.pool(x)
        x = x + y
        # z = self.conv_res(x)

        x = self.conv1(x)
        # print(x.shape)

        x = self.dropout1(x)

        # Block 2
        y = x
        x = self.block2(x)
        x = self.pool(x)
        x = x + y
        # print(x.shape)

        x = self.conv2(x)
        # print(x.shape)

        x = self.dropout2(x)

        # Block 3
        y = x
        x = self.block3(x)# torch.Size([176, 16, 4, 4])
        x = self.pool(x)
        x = x + y


        x = self.conv3(x)

        print(x.shape)


        # x = x + z
        # x = self.conv3(x)
        x = x.view(-1, 11, self.reduced_channels, 4, 4)
        return x


class MultiScaleDimensionalityReducer_12(nn.Module):
    def __init__(self, in_channels=12, out_channels = 4*4, reduction_ratio=8, dropout_prob=0.07):
        super(MultiScaleDimensionalityReducer_12, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, 149, kernel_size=1, stride=1, padding=0),  # Downsample to ~5x5
            nn.SiLU()  # Add SiLU here
        )

        self.reduced_channels=out_channels
        # Block 1: Multi-scale attention with CBAM and spatial downsampling
        self.block1 = MultiScaleAttentionBlock(
            149, 149, kernel_sizes=[3, 7], reduction_ratio=reduction_ratio
        )
        #self.conv_res = nn.Conv3d(in_channels=in_channels, out_channels=4, kernel_size=(1, 4, 4), stride=(1, 3, 3))
        self.pool =nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=149, out_channels=128, kernel_size=2, stride=2),
            nn.SiLU()  # Add SiLU here
        )

        self.dropout1 = nn.Dropout2d(p=dropout_prob)

        # Block 2: Further reduction
        #self.block2 = MultiScaleAttentionBlock(
        #    64, 64, kernel_sizes=[3], reduction_ratio=reduction_ratio
        #)
        self.block2 = ConvAttentionBlock(
            128, 128, kernel_size=3, reduction_ratio=reduction_ratio
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=2, stride=2, padding=1),  # Downsample to ~5x5
            nn.SiLU()  # Add SiLU here
        )

        self.dropout2 = nn.Dropout2d(p=dropout_prob)

        # Final block for dimensionality reduction
        #self.block3 = MultiScaleAttentionBlock(
        #    4, 4, kernel_sizes=[3], reduction_ratio=2
        #)
        self.block3 = ConvAttentionBlock(
            64, self.reduced_channels, kernel_size=3, reduction_ratio=2, attention_kernel=3
        )



    def forward(self, x):
        # Input: (batch_size, frames, channel, height=15, width=15)
        #x = x.reshape(-1, 12, 15, 15)
        #x = x.reshape(-1, 10, 15, 15)

        x = x.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1])

        x = self.conv0(x)

        y = x
        # Block 1
        x = self.block1(x)
        #print(x.shape)
        x = self.pool(x)
        x = x + y
        #z = self.conv_res(x)

        x = self.conv1(x)
        #print(x.shape)

        x = self.dropout1(x)

        # Block 2
        y = x
        x = self.block2(x)
        x = self.pool(x)
        x = x + y
        #print(x.shape)

        x = self.conv2(x)
        #print(x.shape)

        x = self.dropout2(x)

        # Block 3
        x = self.block3(x)
        #print(x.shape)

        #x = x + z
        #x = self.conv3(x)
        x = x.view(-1, 11, self.reduced_channels, 4, 4)
        return x


class MultiScaleAttentionUpscaler_12(nn.Module):
    def __init__(self, in_channels=16, out_channels=12, reduction_ratio=8, dropout_prob=0.1):
        super(MultiScaleAttentionUpscaler_12, self).__init__()

        #self.conv0 = nn.Sequential(
        #    nn.Conv2d(4, 16, kernel_size=1, stride=1),
        #    nn.SiLU()  # Add SiLU here
        #)
        # Block 1: Multi-scale attention with CBAM and upsampling
        #self.block1 = MultiScaleAttentionBlock(
        #    in_channels, in_channels, kernel_sizes=[3], reduction_ratio=reduction_ratio
        #)
        self.block1 = ConvAttentionBlock(
            in_channels=in_channels, out_channels=in_channels, kernel_size=3, reduction_ratio=reduction_ratio
        )
        self.pool =nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        #self.conv_res = nn.ConvTranspose3d(in_channels=in_channels, out_channels=128, kernel_size=(1, 6, 6), stride=(1, 3, 3))

        #self.conv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=64, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=64, kernel_size=2, stride=2),
            nn.SiLU()  # Add SiLU here
        )
        #self.upsample1 = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=True)
        self.dropout1 = nn.Dropout2d(p=dropout_prob)

        # Block 2: Multi-scale attention with CBAM and further upsampling
        #self.block2 = MultiScaleAttentionBlock(
        #    64, 64, kernel_sizes=[3], reduction_ratio=reduction_ratio
        #)
        self.block2 = ConvAttentionBlock(
            64, 64, kernel_size=3, reduction_ratio=reduction_ratio
        )
        #self.upsample2 = nn.Upsample(scale_factor=(1, 1.9, 1.9), mode="trilinear", align_corners=True)
        self.upsample2 = nn.ConvTranspose2d(64, 128, kernel_size=2, stride=2, padding=1, output_padding=1)
        self.dropout2 = nn.Dropout2d(p=dropout_prob)

        # Final Block: Full reconstruction
        self.block3 = MultiScaleAttentionBlock(
            128, 149, kernel_sizes=[3, 7], reduction_ratio=reduction_ratio
        )

        self.final_refinement = nn.Sequential(
            nn.Conv2d(149, out_channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        #self.final_refinement = nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        # Input: (batch_size, reduced_channels=12, frames, height=5, width=5)

        # Block 1
        #print('_____________________')
        #print(x.shape)

        #x = self.conv0(x)
        #print(x.shape)
        y=x
        x = self.block1(x)
        x = self.pool(x)
        x = x + y
        #z = self.conv_res(x)
        #print(x.shape)
        #print(z.shape)
        x = self.conv1(x)  # Upsample to ~10x10
        #print(x.shape)
        x = self.dropout1(x)

        # Block 2
        y=x
        x = self.block2(x)
        x = self.pool(x)
        x = x + y
        x = self.upsample2(x)  # Upsample to ~15x15
        x = self.dropout2(x)
        #x = x+z

        # Block 3
        #y=x
        x = self.block3(x)
        #x=self.pool(x)
        #x = x + y
        #print(x.shape)
        x = self.final_refinement(x)
        #print('~~~~~~~~~~~~~~~~~')
        # Change back to original format
        x = x.view(-1, 11, x.shape[-3], x.shape[-2], x.shape[-1])
        #x = x.view(-1, 11, 10, 15, 15)

        # Output shape: (batch_size, frames, height=15, width=15, out_channels=209)
        return x


class MultiScaleDimensionalityReducer_68(nn.Module):
    def __init__(self, in_channels=67, reduction_ratio=8, dropout_prob=0.07):
        super(MultiScaleDimensionalityReducer_68, self).__init__()

        self.reduced_channels=15
        # Block 1: Multi-scale attention with CBAM and spatial downsampling
        self.block1 = MultiScaleAttentionBlock(
            in_channels, in_channels, kernel_sizes=[3, 5, 7], reduction_ratio=reduction_ratio
        )
        #self.conv_res = nn.Conv3d(in_channels=in_channels, out_channels=4, kernel_size=(1, 4, 4), stride=(1, 3, 3))
        self.pool =nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=2, stride=2),
            nn.SiLU()  # Add SiLU here
        )

        self.dropout1 = nn.Dropout2d(p=dropout_prob)

        # Block 2: Further reduction
        #self.block2 = MultiScaleAttentionBlock(
        #    64, 64, kernel_sizes=[3], reduction_ratio=reduction_ratio
        #)
        self.block2 = ConvAttentionBlock(
            in_channels, in_channels, kernel_size=3, reduction_ratio=reduction_ratio
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, padding=1),  # Downsample to ~5x5
            nn.SiLU()  # Add SiLU here
        )

        self.dropout2 = nn.Dropout2d(p=dropout_prob)

        # Final block for dimensionality reduction
        #self.block3 = MultiScaleAttentionBlock(
        #    4, 4, kernel_sizes=[3], reduction_ratio=2
        #)
        self.block3 = ConvAttentionBlock(
            in_channels, self.reduced_channels, kernel_size=3, reduction_ratio=2, attention_kernel=3
        )




    def forward(self, x):
        # Input: (batch_size, frames, height=15, width=15, spectral_indices=209)
        x = x.view(-1, 67, 15, 15)

        y = x
        # Block 1
        x = self.block1(x)
        #print(x.shape)
        x = self.pool(x)
        x = x + y
        #z = self.conv_res(x)

        x = self.conv1(x)
        #print(x.shape)

        x = self.dropout1(x)

        # Block 2
        y = x
        x = self.block2(x)
        x = self.pool(x)
        x = x + y
        #print(x.shape)

        x = self.conv2(x)
        #print(x.shape)

        x = self.dropout2(x)

        # Block 3
        x = self.block3(x)
        #print(x.shape)

        #x = x + z
        #x = self.conv3(x)
        x = x.view(-1, 11, self.reduced_channels, 4, 4)
        #print(x.shape)
        return x


class MultiScaleAttentionUpscaler_68(nn.Module):
    def __init__(self, in_channels=24, out_channels=67, reduction_ratio=8, dropout_prob=0.1):
        super(MultiScaleAttentionUpscaler_68, self).__init__()

        #self.conv0 = nn.Sequential(
        #    nn.Conv2d(4, 16, kernel_size=1, stride=1),
        #    nn.SiLU()  # Add SiLU here
        #)
        # Block 1: Multi-scale attention with CBAM and upsampling
        #self.block1 = MultiScaleAttentionBlock(
        #    in_channels, in_channels, kernel_sizes=[3], reduction_ratio=reduction_ratio
        #)
        self.block1 = ConvAttentionBlock(
            in_channels, in_channels, kernel_size=3, reduction_ratio=reduction_ratio
        )
        self.pool =nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        #self.conv_res = nn.ConvTranspose3d(in_channels=in_channels, out_channels=128, kernel_size=(1, 6, 6), stride=(1, 3, 3))

        #self.conv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=64, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2),
            nn.SiLU()  # Add SiLU here
        )
        #self.upsample1 = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=True)
        self.dropout1 = nn.Dropout2d(p=dropout_prob)

        # Block 2: Multi-scale attention with CBAM and further upsampling
        #self.block2 = MultiScaleAttentionBlock(
        #    64, 64, kernel_sizes=[3], reduction_ratio=reduction_ratio
        #)
        self.block2 = ConvAttentionBlock(
            out_channels, out_channels, kernel_size=3, reduction_ratio=reduction_ratio
        )
        #self.upsample2 = nn.Upsample(scale_factor=(1, 1.9, 1.9), mode="trilinear", align_corners=True)
        self.upsample2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2, padding=1, output_padding=1)
        self.dropout2 = nn.Dropout2d(p=dropout_prob)

        # Final Block: Full reconstruction
        self.block3 = MultiScaleAttentionBlock(
            out_channels, out_channels, kernel_sizes=[3, 5, 7], reduction_ratio=reduction_ratio
        )

        #self.final_refinement = nn.Sequential(
        #    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        #    nn.SiLU(),
        #    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        #)
        self.final_refinement = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        # Input: (batch_size, reduced_channels=12, frames, height=5, width=5)

        # Block 1
        #print('_____________________')
        #print(x.shape)

        #x = self.conv0(x)
        #print(x.shape)
        y=x
        x = self.block1(x)
        x = self.pool(x)
        x = x + y
        #z = self.conv_res(x)
        #print(x.shape)
        #print(z.shape)
        x = self.conv1(x)  # Upsample to ~10x10
        #print(x.shape)
        x = self.dropout1(x)

        # Block 2
        y=x
        x = self.block2(x)
        x = self.pool(x)
        x = x + y
        x = self.upsample2(x)  # Upsample to ~15x15
        x = self.dropout2(x)
        #x = x+z

        # Block 3
        y=x
        x = self.block3(x)
        x=self.pool(x)
        x = x + y
        #print(x.shape)
        x = self.final_refinement(x)

        # Change back to original format
        x = x.view(-1, 11, 67, 15, 15)

        # Output shape: (batch_size, frames, height=15, width=15, out_channels=209)
        return x

class MultiScaleDimensionalityReducerSmall(nn.Module):
    def __init__(self, channels=12, reduction_ratio=8, dropout_prob=0.07):
        super(MultiScaleDimensionalityReducerSmall, self).__init__()

        # Block 1: Multi-scale attention with CBAM and spatial downsampling
        self.block1 = MultiScaleAttentionBlock(
            channels, channels, kernel_sizes=[3, 5, 7], reduction_ratio=2
        )
        #self.conv_res = nn.Conv3d(in_channels=in_channels, out_channels=4, kernel_size=(1, 4, 4), stride=(1, 3, 3))
        self.pool =nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=128, kernel_size=2, stride=2),
            nn.SiLU()  # Add SiLU here
        )

        self.dropout1 = nn.Dropout2d(p=dropout_prob)

        # Block 2: Further reduction
        #self.block2 = MultiScaleAttentionBlock(
        #    64, 64, kernel_sizes=[3], reduction_ratio=reduction_ratio
        #)
        self.block2 = ConvAttentionBlock(
            128, 128, kernel_size=3, reduction_ratio=reduction_ratio
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=2, stride=2, padding=1),  # Downsample to ~5x5
            nn.SiLU()  # Add SiLU here
        )

        self.dropout2 = nn.Dropout2d(p=dropout_prob)

        # Final block for dimensionality reduction
        #self.block3 = MultiScaleAttentionBlock(
        #    4, 4, kernel_sizes=[3], reduction_ratio=2
        #)
        self.block3 = ConvAttentionBlock(
            64, 4, kernel_size=3, reduction_ratio=2, attention_kernel=3
        )



    def forward(self, x):
        # Input: (batch_size, frames, height=15, width=15, spectral_indices=209)
        x = x.view(-1, 12, 15, 15)

        y = x
        # Block 1
        x = self.block1(x)
        #print(x.shape)
        x = self.pool(x)
        x = x + y
        #z = self.conv_res(x)

        x = self.conv1(x)
        #print(x.shape)

        x = self.dropout1(x)

        # Block 2
        y = x
        x = self.block2(x)
        x = self.pool(x)
        x = x + y
        #print(x.shape)

        x = self.conv2(x)
        #print(x.shape)

        x = self.dropout2(x)

        # Block 3
        #y = x
        x = self.block3(x)
        #x = x + y
        #print(x.shape)

        #x = x + z
        #x = self.conv3(x)
        x = x.view(-1, 11, 4, 4, 4)
        return x


class MultiScaleAttentionUpscalerSmall(nn.Module):
    def __init__(self, channels = 12, reduction_ratio=8, dropout_prob=0.1):
        super(MultiScaleAttentionUpscalerSmall, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=1, stride=1),
            nn.SiLU()  # Add SiLU here
        )
        # Block 1: Multi-scale attention with CBAM and upsampling
        #self.block1 = MultiScaleAttentionBlock(
        #    in_channels, in_channels, kernel_sizes=[3], reduction_ratio=reduction_ratio
        #)
        self.block1 = ConvAttentionBlock(
            16, 16, kernel_size=3, reduction_ratio=reduction_ratio
        )
        self.pool =nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        #self.conv_res = nn.ConvTranspose3d(in_channels=in_channels, out_channels=128, kernel_size=(1, 6, 6), stride=(1, 3, 3))

        #self.conv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=64, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=64, kernel_size=2, stride=2),
            nn.SiLU()  # Add SiLU here
        )
        #self.upsample1 = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=True)
        self.dropout1 = nn.Dropout2d(p=dropout_prob)

        # Block 2: Multi-scale attention with CBAM and further upsampling
        #self.block2 = MultiScaleAttentionBlock(
        #    64, 64, kernel_sizes=[3], reduction_ratio=reduction_ratio
        #)
        self.block2 = ConvAttentionBlock(
            64, 64, kernel_size=3, reduction_ratio=reduction_ratio
        )
        #self.upsample2 = nn.Upsample(scale_factor=(1, 1.9, 1.9), mode="trilinear", align_corners=True)
        self.upsample2 = nn.ConvTranspose2d(64, 128, kernel_size=2, stride=2, padding=1, output_padding=1)
        self.dropout2 = nn.Dropout2d(p=dropout_prob)

        # Final Block: Full reconstruction
        self.block3 = MultiScaleAttentionBlock(
            128, 128, kernel_sizes=[3, 7], reduction_ratio=reduction_ratio
        )

        # Final Block: Full reconstruction
        #self.block3 = MultiScaleAttentionBlock(
        #    channels, channels, kernel_sizes=[3, 5, 7], reduction_ratio=reduction_ratio
        #)
        self.final_refinement = nn.Sequential(
            nn.Conv2d(128, channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        )


    def forward(self, x):
        # Input: (batch_size, reduced_channels=12, frames, height=5, width=5)

        # Block 1
        #print('_____________________')
        #print(x.shape)

        x = self.conv0(x)
        #print(x.shape)
        y=x
        x = self.block1(x)
        x = self.pool(x)
        x = x + y
        #z = self.conv_res(x)
        #print(x.shape)
        #print(z.shape)
        x = self.conv1(x)  # Upsample to ~10x10
        #print(x.shape) #torch.Size([22, 12, 8, 8])
        x = self.dropout1(x)

        # Block 2
        y=x
        x = self.block2(x)
        x = self.pool(x)
        x = x + y
        x = self.upsample2(x)  # Upsample to ~15x15
        x = self.dropout2(x)
        #x = x+z

        # Block 3
        y = x
        x = self.block3(x)
        #print(x.shape)
        x = x + y
        x = self.final_refinement(x)

        # Change back to original format
        x = x.view(-1, 11, 12, 15, 15)

        # Output shape: (batch_size, frames, height=15, width=15, out_channels=209)
        return x




class MultiScaleDimensionalityReducerSmall2(nn.Module):
    def __init__(self, channels=12, reduction_ratio=8, dropout_prob=0.07):
        super(MultiScaleDimensionalityReducerSmall2, self).__init__()

        # Block 1: Multi-scale attention with CBAM and spatial downsampling
        self.block1 = MultiScaleAttentionBlock(
            channels, channels, kernel_sizes=[3, 7], reduction_ratio=2
        )
        #self.conv_res = nn.Conv3d(in_channels=in_channels, out_channels=4, kernel_size=(1, 4, 4), stride=(1, 3, 3))
        self.pool =nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=128, kernel_size=2, stride=2),
            nn.SiLU()  # Add SiLU here
        )

        self.dropout1 = nn.Dropout2d(p=dropout_prob)

        # Block 2: Further reduction
        #self.block2 = MultiScaleAttentionBlock(
        #    64, 64, kernel_sizes=[3], reduction_ratio=reduction_ratio
        #)
        self.block2 = ConvAttentionBlock(
            128, 128, kernel_size=3, reduction_ratio=reduction_ratio
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=2, stride=2, padding=1),  # Downsample to ~5x5
            nn.SiLU()  # Add SiLU here
        )

        self.dropout2 = nn.Dropout2d(p=dropout_prob)

        # Final block for dimensionality reduction
        #self.block3 = MultiScaleAttentionBlock(
        #    4, 4, kernel_sizes=[3], reduction_ratio=2
        #)
        self.block3 = ConvAttentionBlock(
            64, 12, kernel_size=3, reduction_ratio=2, attention_kernel=3
        )



    def forward(self, x):
        # Input: (batch_size, frames, height=15, width=15, spectral_indices=209)
        x = x.view(-1, 12, 15, 15)

        y = x
        # Block 1
        x = self.block1(x)
        #print(x.shape)
        x = self.pool(x)
        x = x + y
        #z = self.conv_res(x)

        x = self.conv1(x)
        #print(x.shape)

        x = self.dropout1(x)

        # Block 2
        y = x
        x = self.block2(x)
        x = self.pool(x)
        x = x + y
        #print(x.shape)

        x = self.conv2(x)
        #print(x.shape)

        x = self.dropout2(x)

        # Block 3
        #y = x
        x = self.block3(x)
        #x = x + y
        #print(x.shape)

        #x = x + z
        #x = self.conv3(x)
        x = x.view(-1, 11, 12, 4, 4)
        return x


class MultiScaleAttentionUpscalerSmall2(nn.Module):
    def __init__(self, channels = 12, reduction_ratio=8, dropout_prob=0.1):
        super(MultiScaleAttentionUpscalerSmall2, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(12, 16, kernel_size=1, stride=1),
            nn.SiLU()  # Add SiLU here
        )
        # Block 1: Multi-scale attention with CBAM and upsampling
        #self.block1 = MultiScaleAttentionBlock(
        #    in_channels, in_channels, kernel_sizes=[3], reduction_ratio=reduction_ratio
        #)
        self.block1 = ConvAttentionBlock(
            16, 16, kernel_size=3, reduction_ratio=reduction_ratio
        )
        self.pool =nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        #self.conv_res = nn.ConvTranspose3d(in_channels=in_channels, out_channels=128, kernel_size=(1, 6, 6), stride=(1, 3, 3))

        #self.conv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=64, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=64, kernel_size=2, stride=2),
            nn.SiLU()  # Add SiLU here
        )
        #self.upsample1 = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=True)
        self.dropout1 = nn.Dropout2d(p=dropout_prob)

        # Block 2: Multi-scale attention with CBAM and further upsampling
        #self.block2 = MultiScaleAttentionBlock(
        #    64, 64, kernel_sizes=[3], reduction_ratio=reduction_ratio
        #)
        self.block2 = ConvAttentionBlock(
            64, 64, kernel_size=3, reduction_ratio=reduction_ratio
        )
        #self.upsample2 = nn.Upsample(scale_factor=(1, 1.9, 1.9), mode="trilinear", align_corners=True)
        self.upsample2 = nn.ConvTranspose2d(64, 128, kernel_size=2, stride=2, padding=1, output_padding=1)
        self.dropout2 = nn.Dropout2d(p=dropout_prob)

        # Final Block: Full reconstruction
        self.block3 = MultiScaleAttentionBlock(
            128, 128, kernel_sizes=[3, 7], reduction_ratio=reduction_ratio
        )

        # Final Block: Full reconstruction
        #self.block3 = MultiScaleAttentionBlock(
        #    channels, channels, kernel_sizes=[3, 5, 7], reduction_ratio=reduction_ratio
        #)
        self.final_refinement = nn.Sequential(
            nn.Conv2d(128, channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        )


    def forward(self, x):
        # Input: (batch_size, reduced_channels=12, frames, height=5, width=5)

        # Block 1
        #print('_____________________')
        #print(x.shape)

        x = self.conv0(x)
        #print(x.shape)
        y=x
        x = self.block1(x)
        x = self.pool(x)
        x = x + y
        #z = self.conv_res(x)
        #print(x.shape)
        #print(z.shape)
        x = self.conv1(x)  # Upsample to ~10x10
        #print(x.shape) #torch.Size([22, 12, 8, 8])
        x = self.dropout1(x)

        # Block 2
        y=x
        x = self.block2(x)
        x = self.pool(x)
        x = x + y
        x = self.upsample2(x)  # Upsample to ~15x15
        x = self.dropout2(x)
        #x = x+z

        # Block 3
        y = x
        x = self.block3(x)
        #print(x.shape)
        x = x + y
        x = self.final_refinement(x)

        # Change back to original format
        x = x.view(-1, 11, 12, 15, 15)

        # Output shape: (batch_size, frames, height=15, width=15, out_channels=209)
        return x