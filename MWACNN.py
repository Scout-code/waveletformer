import torch
import numpy as np
import torch.nn as nn
import scipy.io as sio
from torch.utils.data import Dataset,DataLoader, WeightedRandomSampler
from sklearn.preprocessing import maxabs_scale
import torch
from torch import DoubleTensor
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
from pytorch_wavelets import DWT1DForward, DWT1DInverse, DWT1D  # or simply DWT1D, IDWT1D
import pywt


#train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, shuffle=False, num_workers=0
#train_loader = torch.utils.data.TensorDataset((X_train, y_train))

class A_cSE(nn.Module):
      def __init__(self, in_ch):
        super(A_cSE, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(in_ch),
            nn.ReLU(inplace=True),
           )
        self.conv1 = nn.Sequential(
           nn.Conv1d(in_ch, int(in_ch/2), kernel_size=1, padding=0),
           nn.BatchNorm1d(int(in_ch/2)),
           nn.ReLU(inplace=True),
            )
        self.conv2 = nn.Sequential(
            nn.Conv1d(int(in_ch/2), in_ch, kernel_size=1, padding=0),
            nn.BatchNorm1d(in_ch)
           )
      def forward(self, in_x):

        x = self.conv0(in_x)
        x = nn.AvgPool1d(x.size()[2:])(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.sigmoid(x)
        return in_x * x + in_x

class SConv_1D(nn.Module):
     def __init__(self, in_ch, out_ch, kernel, pad):
      super(SConv_1D, self).__init__()
      self.conv = nn.Sequential(
          nn.Conv1d(in_ch, out_ch, kernel, padding=pad),
          nn.GroupNorm(6, out_ch),
          nn.ReLU(inplace=True),
         )

     def forward(self, x):

            x = self.conv(x)
            return x
#input = 1
numf =12

class MWA_CNN( nn.Module ):
    def __init__(self, numf=12, channel=1):
        super( MWA_CNN, self ).__init__()
        self.DWT0 = DWT1DForward( J=1, wave='db1' ).cuda()
        self.SConv1 = SConv_1D( channel * 2, numf, 3, 1 )
        self.DWT1 = DWT1DForward( J=1, wave='db1' ).cuda()
        self.dropout1 = nn.Dropout( p=0.1 )
        self.cSE1 = A_cSE( numf * 2 )
        self.SConv2 = SConv_1D( numf * 2, numf * 2, 3, 1 )
        self.DWT2 = DWT1DForward( J=1, wave='db1' ).cuda()
        self.dropout2 = nn.Dropout( p=0.1 )
        self.cSE2 = A_cSE( numf * 4 ),
        self.SConv3 = SConv_1D( numf * 4, numf * 4, 3, 1 )
        self.DWT3 = DWT1DForward( J=1, wave='db1' ).cuda()
        self.dropout3 = nn.Dropout( p=0.1 )
        self.cSE3 = A_cSE( numf * 8 )
        self.SConv4 = SConv_1D( numf * 8, numf * 8, 3, 1 )
        self.DWT4 = DWT1DForward( J=1, wave='db1' ).cuda()
        self.dropout4 = nn.Dropout( p=0.1 )
        self.cSE4 = A_cSE( numf * 16 )
        self.SConv5 = SConv_1D( numf * 16, numf * 16, 3, 1 )
        self.DWT5 = DWT1DForward( J=1, wave='db1' ).cuda()
        self.dropout5 = nn.Dropout( p=0.1 )
        self.cSE5 = A_cSE( numf * 32 )
        self.SConv6 = SConv_1D( numf * 32, numf * 32, 3, 1 )
        self.avg_pool = nn.AdaptiveAvgPool1d( (1) )
        self.fc = nn.Linear( numf * 32, 5 )

    def forward(self, input):
        #print(input.shape)
        DMT_yl, DMT_yh = self.DWT0( input )
        output = torch.cat( [DMT_yl, DMT_yh[0]], dim=1 )
        #print( output.shape )
        output = self.SConv1( output )
        #print( output.shape )
        DMT_yl, DMT_yh = self.DWT1( output )
        #print( DMT_yl.shape )
        #print( DMT_yh[0].shape )
        output = torch.cat( [DMT_yl, DMT_yh[0]], dim=1 )
        #print( output.shape )
        output = self.dropout1( output )
        output = self.cSE1( output )
        output = self.SConv2( output )
        #print( output.shape )
        DMT_yl, DMT_yh = self.DWT2( output )
        output = torch.cat( [DMT_yl, DMT_yh[0]], dim=1 )
        #print( output.shape )
        output = self.dropout2( output )
        #print( output.shape )
        #output = self.cSE2( output )
        #print( output.shape )
        #output = self.SConv3( output )
        DMT_yl, DMT_yh = self.DWT3( output )
        output = torch.cat( [DMT_yl, DMT_yh[0]], dim=1 )
        output = self.dropout3( output )
        output = self.cSE3( output )
        output = self.SConv4( output )
        DMT_yl, DMT_yh = self.DWT4( output )
        output = torch.cat( [DMT_yl, DMT_yh[0]], dim=1 )
        output = self.dropout4( output )
        output = self.cSE4( output )
        output = self.SConv5( output )
        DMT_yl, DMT_yh = self.DWT5( output )
        output = torch.cat( [DMT_yl, DMT_yh[0]], dim=1 )
        output = self.dropout5( output )
        output = self.cSE5( output )
        output = self.SConv6( output )
        output = self.avg_pool( output )
        output = output.view( output.size( 0 ), -1 )
        output = self.fc( output )
        return  output

cnn = MWA_CNN()

