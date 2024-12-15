import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.selu = nn.SELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.residual(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.selu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = self.selu(x)
        return x

class BayesNuSeg(nn.Module):
    def __init__(self, in_channels=1, seed_out_channels=2, instance_out_channels=2, dropout_rate=0.5):
        super(BayesNuSeg, self).__init__()
        # Encoder
        self.encoder1 = ResidualConvBlock(in_channels, 64)
        self.encoder2 = ResidualConvBlock(64, 128)
        self.encoder3 = ResidualConvBlock(128, 256)
        self.encoder4 = ResidualConvBlock(256, 512)
        self.encoder5 = ResidualConvBlock(512, 1024)

        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Seed Branch Decoder
        self.seed_upconv5 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.seed_decoder5 = ResidualConvBlock(1024, 512)
        self.seed_upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.seed_decoder4 = ResidualConvBlock(512, 256)
        self.seed_upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.seed_decoder3 = ResidualConvBlock(256, 128)
        self.seed_upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.seed_decoder2 = ResidualConvBlock(128, 64)
        self.seed_upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.seed_decoder1 = ResidualConvBlock(128, 64)
        self.seed_output = nn.Conv2d(64, seed_out_channels, kernel_size=1)

        # Instance Branch Decoder
        self.instance_upconv5 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.instance_decoder5 = ResidualConvBlock(1024, 512)
        self.instance_upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.instance_decoder4 = ResidualConvBlock(512, 256)
        self.instance_upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.instance_decoder3 = ResidualConvBlock(256, 128)
        self.instance_upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.instance_decoder2 = ResidualConvBlock(128, 64)
        self.instance_upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.instance_decoder1 = ResidualConvBlock(128, 64)
        self.instance_output = nn.Conv2d(64, instance_out_channels, kernel_size=1)

        # Dropout layer for Bayesian approximation
        self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        enc5 = self.encoder5(self.pool(enc4))

        # Seed Branch Decoder
        seed_up5 = self.seed_upconv5(self.dropout(enc5))
        seed_dec5 = self.seed_decoder5(torch.cat([seed_up5, enc4], dim=1))
        seed_up4 = self.seed_upconv4(seed_dec5)
        seed_dec4 = self.seed_decoder4(torch.cat([seed_up4, enc3], dim=1))
        seed_up3 = self.seed_upconv3(seed_dec4)
        seed_dec3 = self.seed_decoder3(torch.cat([seed_up3, enc2], dim=1))
        seed_up2 = self.seed_upconv2(seed_dec3)
        seed_dec2 = self.seed_decoder2(torch.cat([seed_up2, enc1], dim=1))
        seed_up1 = self.seed_upconv1(seed_dec2)
        seed_dec1 = self.seed_decoder1(torch.cat([seed_up1, x], dim=1))
        seed_output = self.seed_output(seed_dec1)

        # Instance Branch Decoder
        instance_up5 = self.instance_upconv5(self.dropout(enc5))
        instance_dec5 = self.instance_decoder5(torch.cat([instance_up5, enc4], dim=1))
        instance_up4 = self.instance_upconv4(instance_dec5)
        instance_dec4 = self.instance_decoder4(torch.cat([instance_up4, enc3], dim=1))
        instance_up3 = self.instance_upconv3(instance_dec4)
        instance_dec3 = self.instance_decoder3(torch.cat([instance_up3, enc2], dim=1))
        instance_up2 = self.instance_upconv2(instance_dec3)
        instance_dec2 = self.instance_decoder2(torch.cat([instance_up2, enc1], dim=1))
        instance_up1 = self.instance_upconv1(instance_dec2)
        instance_dec1 = self.instance_decoder1(torch.cat([instance_up1, x], dim=1))
        instance_output = self.instance_output(instance_dec1)

        return seed_output, instance_output
