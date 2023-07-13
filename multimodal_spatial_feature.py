import math
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.ops.boxes as box_ops

from torch import nn, Tensor
from typing import Dict, List, Optional, Tuple
from text_features import SPATIAL_TEXT_FEATURES

def senmatic_spatial_feature(    
    boxes_1: List[Tensor], boxes_2: List[Tensor],
    labels: List[Tensor], eps: float = 1e-10
):

    features=[]
    
    for b1,b2,label in zip(boxes_1,boxes_2,labels):
        
        c1_x = (b1[:, 0] + b1[:, 2]) / 2; c1_y = (b1[:, 1] + b1[:, 3]) / 2   
        c2_x = (b2[:, 0] + b2[:, 2]) / 2; c2_y = (b2[:, 1] + b2[:, 3]) / 2

        human_to_object_vector_x=c2_x-c1_x      
        human_to_object_vector_y=c2_y-c1_y

        center_to_box_vector_x=b1[:,2]-c1_x;center_to_box_vector_y=b1[:,3]-c1_y  

        vector1_lenth=torch.sqrt(torch.mul(human_to_object_vector_x,human_to_object_vector_x)+torch.mul(human_to_object_vector_y,human_to_object_vector_y))
        vector2_lenth=torch.sqrt(torch.mul(center_to_box_vector_x,center_to_box_vector_x)+torch.mul(center_to_box_vector_y,center_to_box_vector_y))

        sin_value=torch.div(human_to_object_vector_y, vector1_lenth)
        pi=math.pi
        
        for index in range(len(human_to_object_vector_x)):
            lenth1=vector1_lenth[index]
            lenth2=vector2_lenth[index]
            
            sin=sin_value[index]
            spatial_number=0

            if lenth1<=lenth2/4:
                spatial_number=0
                features.append(SPATIAL_TEXT_FEATURES[str((label[index].tolist(),spatial_number))])
                continue
            if math.sin(345/180*pi)<sin<=math.sin(15/180*pi):
                spatial_number=1
            elif math.sin(15/180*pi)<sin<=math.sin(75/180*pi):
                spatial_number=2
            elif math.sin(75/180*pi)<sin<=1:
                spatial_number=3
            elif math.sin(165/180*pi)<sin<=math.sin(105/180*pi):
                spatial_number=4
            elif math.sin(195/180*pi)<sin<=math.sin(165/180*pi):
                spatial_number=5
            elif math.sin(255/180*pi)<sin<=math.sin(195/180*pi):
                spatial_number=6
            elif -1<=sin<=math.sin(255/180*pi):
                spatial_number=7
            elif math.sin(285/180*pi)<sin<=math.sin(345/180*pi):
                spatial_number=8
            else:
                print("there is something wrong in senmatic spatial feature")
            features.append(SPATIAL_TEXT_FEATURES[str((label[index].tolist(),spatial_number))]) 

    features=torch.Tensor(features).to('cuda')
    features=features.float()

    return features


def visual_spatial_feature(
    boxes_1: List[Tensor], boxes_2: List[Tensor],
    shapes: List[Tuple[int, int]], eps: float = 1e-10
) -> Tensor:
    features = []
    for b1, b2, shape in zip(boxes_1, boxes_2, shapes):
        h, w = shape
        c1_x = (b1[:, 0] + b1[:, 2]) / 2; c1_y = (b1[:, 1] + b1[:, 3]) / 2
        c2_x = (b2[:, 0] + b2[:, 2]) / 2; c2_y = (b2[:, 1] + b2[:, 3]) / 2
        b1_w = b1[:, 2] - b1[:, 0]; b1_h = b1[:, 3] - b1[:, 1]
        b2_w = b2[:, 2] - b2[:, 0]; b2_h = b2[:, 3] - b2[:, 1]

        d_x = torch.abs(c2_x - c1_x) / (b1_w + eps)
        d_y = torch.abs(c2_y - c1_y) / (b1_h + eps)
        iou = torch.diag(box_ops.box_iou(b1, b2))
        f = torch.stack([
            c1_x / w, c1_y / h, c2_x / w, c2_y / h,
            b1_w / w, b1_h / h, b2_w / w, b2_h / h,
            b1_w * b1_h / (h * w), b2_w * b2_h / (h * w),
            b2_w * b2_h / (b1_w * b1_h + eps),
            b1_w / (b1_h + eps), b2_w / (b2_h + eps),
            iou,
            (c2_x > c1_x).float() * d_x,
            (c2_x < c1_x).float() * d_x,
            (c2_y > c1_y).float() * d_y,
            (c2_y < c1_y).float() * d_y,
        ], 1)

        features.append(
            torch.cat([f, torch.log(f + eps)], 1)
        )
    return torch.cat(features)