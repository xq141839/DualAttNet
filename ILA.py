from torch.nn import functional as F
import torch.nn as nn
import torch

class ILA(nn.Module):
    def __init__(self, channel, num_classes):
        super(ILA, self).__init__()

        self.conv1 = nn.Conv2d(channel, num_classes, kernel_size=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(channel, channel // 16, kernel_size=1, stride=1, bias=False)
        self.conv3 = nn.Conv2d(channel // 16, num_classes, kernel_size=1, stride=1, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(num_classes)


    def forward(self, x):

        gap = F.adaptive_avg_pool2d(x, 1)

        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)

        gap = self.conv2(gap)
        gap = self.relu(gap)

        atten = self.conv3(gap)
        atten = torch.sigmoid(atten)
        out = x * atten
      
        return out