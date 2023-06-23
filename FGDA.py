from torch.nn import functional as F
import torch.nn as nn
import torch

def FGDA(classification_maps, src_shape):
        batch_size = classification_maps.shape[0]
        batch_att = []

        for i in range(batch_size):
            batch_map = classification_maps[i] # W x H x Anchor X Class

            # for each class
            att = []
            for j in range(batch_map[0].shape[-1]):
                target_index = j

                max_anchor = torch.max(batch_map, 2)[0]
                class_map = max_anchor[..., target_index]

                class_map = F.interpolate(class_map.unsqueeze(0).unsqueeze(0), size=src_shape,
                                     mode='bilinear', align_corners=True)
                class_map = class_map.squeeze().squeeze()

                norm_map = class_map / class_map.sum()
                    
                att.append(norm_map)

            batch_att.append(torch.stack(att, dim=0))

        batch_att = torch.stack(batch_att, dim=0)

        return batch_att