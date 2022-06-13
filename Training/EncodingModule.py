import torch
from torch.nn import Module
import torch.nn.functional as F
import torch.nn as nn

class Encoding(Module):
    
    def __init__(self, D, K):  ## D, K the input and output channels
        super(Encoding, self).__init__()
        self.D, self.K = D, K 

        ## encoding conv layers
        self.encodinglayers = nn.Sequential(
            torch.nn.Conv3d(in_channels= self.D, out_channels = self.D, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(num_features=self.D), 
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(kernel_size=2, stride=2),
            torch.nn.Conv3d(in_channels= self.D, out_channels = self.K, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(num_features=self.K), 
            torch.nn.ReLU(),
        )

    def forward(self, X):
        E = self.encodinglayers(X)
        return E

class EncModule(nn.Module):
    def __init__(self, in_channels, out_channels=32, se_loss=True):
        super(EncModule, self).__init__()
        self.out_channels = out_channels
        self.se_loss = se_loss
        self.encoding = Encoding(in_channels, out_channels)

        self.fc = nn.Sequential(
            nn.Linear(out_channels * 12 * 12 * 12, in_channels),
            nn.Sigmoid())
        if self.se_loss:
            self.selayer = nn.Sequential(
                nn.ConvTranspose3d(in_channels=out_channels, out_channels=12, kernel_size=3, padding=1, stride=2, output_padding=1),
                torch.nn.BatchNorm3d(num_features=12), 
                torch.nn.Tanh()
            )

    def forward(self, x):
        en = self.encoding(x) 
        b, c, _, _, _ = x.size()
        flatEn = en.view(-1, self.out_channels * 12 * 12 * 12 )
        gamma = self.fc(flatEn)

        y = gamma.view(b, c, 1, 1, 1)

        outputs = [F.relu_(x + x * y)]
        if self.se_loss:
            outputs.append(self.selayer(en))
        return tuple(outputs)
