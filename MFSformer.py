import torch
import numpy as np
import torch.nn as nn
import scipy.io as sio
from torch.utils.data import Dataset,DataLoader
from sklearn.preprocessing import maxabs_scale
import torch
from torch import nn,optim
from torch import DoubleTensor
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report



# Multiscale Convolution Block (MSC)
class MSCBlock( nn.Module ):
    def __init__(self, in_channels):
        super( MSCBlock, self ).__init__()
        self.conv1 = nn.Conv1d( in_channels, in_channels, kernel_size=3, padding=1 )
        self.conv2 = nn.Conv1d( in_channels, in_channels, kernel_size=5, padding=2 )
        self.conv3 = nn.Conv1d( in_channels, in_channels, kernel_size=7, padding=3 )
        self.bn = nn.BatchNorm1d( in_channels * 3 )
        self.gelu = nn.GELU()

    def forward(self, x):
        out1 = self.conv1( x )
        out2 = self.conv2( x )
        out3 = self.conv3( x )
        out = torch.cat( [out1, out2, out3], dim=1 )  # Concatenate along the channel dimension
        out = self.bn( out )
        out = self.gelu( out )
        return out


# Fuse-Shuffle Attention (FSA) Block
class FSA( nn.Module ):
    def __init__(self, in_channels):
        super( FSA, self ).__init__()
        # Split into two branches for channel and spatial attention
        self.gap = nn.AdaptiveAvgPool1d( 1 )
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv1d( in_channels // 2, in_channels // 2, kernel_size=1 )
        self.conv2 = nn.Conv1d( in_channels // 2, in_channels // 2, kernel_size=1 )
        self.gn = nn.GroupNorm(1, in_channels // 2)

    def forward(self, x):
        # Split the input into two branches along the channel dimension
        x1, x2 = torch.chunk( x, 2, dim=1 )
        # print('x', x.shape)
        # print( 'x1', x1.shape )
        # print( 'x2', x2.shape )
        # Channel attention
        s1 = self.gap( x1 )
        #s2 = self.gap( x2 )
        # print( 's', s.shape )
        #
        # print( 's1', s1.shape )
        attn1 = self.sigmoid( self.conv1( self.gn( s1 ) ) ) * x1

        # Spatial attention
        attn2 = self.sigmoid( self.conv2( self.gn( x2 ) ) ) * x2

        # Merge and shuffle channels
        out = torch.cat( [attn1, attn2], dim=1 )
        batch_size, channels, length = out.size()
        out = out.view( batch_size, -1, 2, length ).permute( 0, 2, 1, 3 ).contiguous().view( batch_size, channels,
                                                                                             length )

        return out

# Input Layer
class InputLayer(nn.Module):
    def __init__(self):
        super(InputLayer, self).__init__()
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.conv = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.bn_gelu = nn.Sequential(
            nn.BatchNorm1d(16),
            nn.GELU()
        )

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.bn_gelu(x)
        return x

# Feature Extraction Layer with MSC and FSA
class FeatureExtractionLayer(nn.Module):
    def __init__(self, in_channels):
        super(FeatureExtractionLayer, self).__init__()
        self.msc = MSCBlock(in_channels)
        self.fsa = FSA(in_channels * 3)

    def forward(self, x):
        x = self.msc(x)
        x = self.fsa(x)
        return x

# MFSFormer Model
class MFSFormer(nn.Module):
    def __init__(self, num_classes=5):
        super(MFSFormer, self).__init__()
        self.input_layer = InputLayer()
        self.feature_extraction_layers = nn.Sequential(
            FeatureExtractionLayer(16),
            FeatureExtractionLayer(48),
            FeatureExtractionLayer(144)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(432, num_classes)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.feature_extraction_layers(x)
        x = self.pool(x).squeeze(-1)
        x= self.fc(x)
        return x


