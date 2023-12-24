import torch
import torch.nn as nn
import torch.nn.functional as F



class VGG(nn.Module):
    def __init__(self,vgg_name, dropout, num_class):
        super().__init__()
        # 1. define multiple convolution and downsampling layers
        # 3. define full-connected layer to classify
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )

        self.conv = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096,num_class)
            # if we use crossentropy then we needn't softmax() here.
        )

    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # extract features
        out = self.conv(x)
        # classification
        out= self.fc(out)
        return out


class ResBlock(nn.Module):
    ''' residual block'''
    def __init__(self, in_channel, out_channel, stride):
        super().__init__()
        '''
        in_channel: number of channels in the input image.
        out_channel: number of channels produced by the convolution.
        stride: stride of the convolution.
        '''
        # 1. define double convolution
             # convolution
             # batch normalization
             # activate function
             # ......
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel,out_channel,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(out_channel)
        )

        self.shortcut = nn.Sequential()
        # 2. if in_channel != out_channel or stride != 1, deifine 1x1 convolution layer to change the channel or size.
        if in_channel!=out_channel or stride!=1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel,out_channel,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channel)
            )
        # Note: we are going to implement 'Basic residual block' by above steps, you can also implement 'Bottleneck Residual block'
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # 1. convolve the input
        # 2. if in_channel != out_channel or stride != 1, change the channel or size of 'x' using 1x1 convolution.
        # 3. Add the output of the convolution and the original data (or from 2.)
        # 4. relu
        out = self.relu(self.residual_function(x)+self.shortcut(x))
        return out


class ResNet(nn.Module):
    '''residual network'''
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()

        # 1. define convolution layer to process raw RGB image
        # 2. define multiple residual blocks
        # 3. define full-connected layer to classify
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2_x = self._make_layer(block,64,num_blocks[0],1)
        self.conv3_x = self._make_layer(block,128,num_blocks[1],2)
        self.conv4_x = self._make_layer(block,256,num_blocks[2],2)
        self.conv5_x = self._make_layer(block,512,num_blocks[3],2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512,num_classes)
        self.feature = nn.Sequential(
            self.conv1,
            self.conv2_x,
            self.conv3_x,
            self.conv4_x,
            self.conv5_x
        )

    def _make_layer(self,block,out_channels,num_blocks,stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels,out_channels,stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # extract features
        out = self.feature(x)
        # classification
        out = self.avgpool(out)
        out = self.fc(out.view(out.size(0),-1))
        return out
    

class ResNextBlock(nn.Module):
    '''ResNext block'''
    def __init__(self, in_channel, out_channel, bottle_neck, group, stride):
        super().__init__()
        # in_channel: number of channels in the input image
        # out_channel: number of channels produced by the convolution
        # bottle_neck: int, bottleneck= out_channel / hidden_channel 
        # group: number of blocked connections from input channels to output channels
        # stride: stride of the convolution.

        # 1. define convolution
             # 1x1 convolution
             # batch normalization
             # activate function
             # 3x3 convolution
             # ......
             # 1x1 convolution
             # ......

        # 2. if in_channel != out_channel or stride != 1, deifine 1x1 convolution layer to change the channel or size.

    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # 1. convolve the input
        # 2. if in_channel != out_channel or stride != 1, change the channel or size of 'x' using 1x1 convolution.
        # 3. Add the output of the convolution and the original data (or from 2.)
        # 4. relu
        return out


class ResNext(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. define convolution layer to process raw RGB image
        # 2. define multiple residual blocks
        # 3. define full-connected layer to classify

    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # extract features
        # classification
        return out

