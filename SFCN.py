import torch
import torch.nn as nn
import torch.nn.functional as F

from math import floor


def get_Bb_dims(b):
    """
    Return the dimension array of the network corresponding to the model Bb
    """ 
    D = [
        [2, 2, 2, 3, 3, 2],         #0
        [4, 4, 4, 6, 6, 4],         #1
        [4, 6, 6, 8, 8, 6],         #2
        [4, 8, 8, 16, 16, 8],       #3
        [4, 8, 16, 32, 32, 8],      #4
        [8, 16, 32, 64, 64, 16],    #5
        [16, 32, 64, 128, 128, 32], #6
        [32, 64, 128, 256, 256, 64] #7
        ]
    return D[b]

class Block(nn.Module):
    """
    3D convolutional layer with batch normalization and ReLU activation.
    """
    def __init__(self, in_channels, out_channels, max_pool=False):
        super(Block, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = max_pool
        self.max_pool_layer = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.max_pool:
            x = self.max_pool_layer(x)
        return x

class FcBlock(nn.Module):
    """
    1x1x1 3D convolutional layer with batch normalization and ReLU activation.
    """
    def __init__(self, in_channels, out_channels):
        super(FcBlock, self).__init__()
        self.fc = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SFCN(nn.Module):
    """ 
    A conventional convolutional network 
    based on SFCN: https://doi.org/10.1016/j.media.2020.101871,
    """
    def __init__(self, dims, out_dim):
        super(SFCN, self).__init__()
        self.blocks = nn.ModuleList()

        dims = [1] + dims
        for i in range(len(dims)-2):
            self.blocks.append(Block(dims[i], dims[i+1], max_pool=True))
        
        self.blocks.append(FcBlock(dims[-2], dims[-1],))

        # handle output dimension
        if type(out_dim) == int:
            self.out_shape = out_dim
            self.out_dim = out_dim
        else:
            self.out_shape = out_dim
            self.out_dim = torch.prod(torch.tensor(out_dim)).item()

        self.fc = nn.Linear(dims[-1], self.out_dim, bias=True)

    def forward(self, x):
        """
        x : (batch_size, 1, n*32, m*32, l*32)
        """
        for block in self.blocks:
            x = block(x)
        
        # Global average pooling
        x = F.adaptive_avg_pool3d(x, 1).view(x.size(0), -1)
        #x = torch.mean(x, dim=(-3, -2, -1))
        
        x = self.fc(x)

        # If the output dimension is not a scalar, reshape it
        if self.out_shape != self.out_dim:
            x = x.view(-1, *self.out_shape)

        return x

    def get_feature_maps(self, x):
        feature_maps = []
        for block in self.blocks:
            x = block(x)
            feature_maps.append(x)
        return feature_maps

