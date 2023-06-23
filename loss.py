from torch.nn import functional as F
import torch.nn as nn
import torch


class DualAttNetLoss(nn.Module):

    def __init__(self, pos_weight=None, with_logit=True):
        super(DualAttNetLoss, self).__init__()
        self.pos_weight = pos_weight
        if pos_weight is not None:
            self.pos_weight = torch.FloatTensor(self.pos_weight).cuda()
        self.with_logit = with_logit

    def forward(self, att_maps, annotations):
        batch_size  = att_maps[0].shape[0]
        one_hot = torch.zeros(att_maps[0].shape)
            
        for i in range(batch_size):
            target = annotations[i, :, 4].long()
            for j in range(len(target)):
                if target[j] != -1:
                    one_hot[i, target[j]] = 1
                    
        if not self.with_logit:
            att_maps = torch.clamp(att_maps, 1e-10, 1.-1e-10)

        if torch.cuda.is_available():
            one_hot = one_hot.cuda()
        
        
        att_maps = torch.stack(att_maps, dim = 0)
        outputs = torch.mean(att_maps, dim=0)

        if self.with_logit:

            return F.binary_cross_entropy_with_logits(outputs, one_hot, pos_weight=self.pos_weight)
        else:
            return F.binary_cross_entropy(outputs, one_hot).mean()