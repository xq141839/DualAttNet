from torch.nn import functional as F
import torch.nn as nn
import torch
from ILA import ILA
from FGDA import FGDA

class DualAttNet(nn.Module):
    def __init__(self, channel, num_classes):
        super(ILA, self).__init__()

        self.conv1 = nn.Conv2d(channel, num_classes, kernel_size=1, stride=1, bias=False)
        self.ila = ILA(channel, num_classes)


    def forward(self, fpn_feature, classification_feature):

        x = self.conv1(fpn_feature)
        global_att_map = self.ila(fpn_feature)
        local_att = FGDA(classification_feature, x.shape[-2:])
        dual_att_map = x * local_att + global_att_map

        out = dual_att_map.sum(dim=(2, 3))

        return out