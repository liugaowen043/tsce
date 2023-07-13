import torch
import torch.nn.functional as F

from torch import nn, Tensor
from typing import List, Optional, Tuple
from collections import OrderedDict

import pocket

from ops import compute_spatial_encodings
from multimodal_spatial_feature import senmatic_spatial_feature
from modified_encoder import ModifiedEncoder,MultiBranchFusion

class InteractionHead(nn.Module):
    """
    Interaction head that constructs and classifies box pairs  
    Parameters:
    -----------
    box_pair_predictor: nn.Module     
        Module that classifies box pairs
    hidden_state_size: int   
        Size of the object features
    representation_size: int  
        Size of the human-object pair features
    num_channels: int    
        Number of channels in the global image features
    num_classes: int
        Number of target classes
    human_idx: int   
        The index of human/person class
    object_class_to_target_class: List[list]
        The set of valid action classes for each object type
    """
    def __init__(self,
        box_pair_predictor: nn.Module,
        hidden_state_size: int, representation_size: int,
        num_channels: int, num_classes: int, human_idx: int,
        object_class_to_target_class: List[list]
    ) -> None:
        super().__init__()

        self.box_pair_predictor = box_pair_predictor

        self.hidden_state_size = hidden_state_size
        self.representation_size = representation_size

        self.num_classes = num_classes
        self.human_idx = human_idx     
        self.object_class_to_target_class = object_class_to_target_class

        self.spatial_head = nn.Sequential(
            nn.Linear(36, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, representation_size),
            nn.ReLU(),
        )

        self.coop_layer = ModifiedEncoder(
            hidden_size=hidden_state_size,
            representation_size=representation_size,
            num_layers=2,
            return_weights=True
        )
        
        self.comp_layer = pocket.models.TransformerEncoderLayer(
            hidden_size=representation_size * 2,
            return_weights=True
        )
        
        self.mbf = MultiBranchFusion(
            hidden_state_size * 2,
            representation_size, representation_size,
            cardinality=16
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.mbf_g = MultiBranchFusion(
            num_channels, representation_size,
            representation_size, cardinality=16
        )

        self.mix_feature_layer= nn.Sequential(
            nn.Linear(representation_size*2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, representation_size),
            nn.ReLU(),
        )

    def compute_prior_scores(self,
        x: Tensor, y: Tensor, scores: Tensor, object_class: Tensor
    ) -> Tensor:
        prior_h = torch.zeros(len(x), self.num_classes, device=scores.device)
        prior_o = torch.zeros_like(prior_h)

        # Raise the power of object detection scores during inference
        p = 1.0 if self.training else 2.8
        s_h = scores[x].pow(p)
        s_o = scores[y].pow(p)

        # Map object class index to target class index
        # Object class index to target class index is a one-to-many mapping
        target_cls_idx = [self.object_class_to_target_class[obj.item()]
            for obj in object_class[y]]
        # Duplicate box pair indices for each target class
        pair_idx = [i for i, tar in enumerate(target_cls_idx) for _ in tar]
        # Flatten mapped target indices
        flat_target_idx = [t for tar in target_cls_idx for t in tar]

        prior_h[pair_idx, flat_target_idx] = s_h[pair_idx]
        prior_o[pair_idx, flat_target_idx] = s_o[pair_idx]

        return torch.stack([prior_h, prior_o])

    def forward(self, features: OrderedDict, image_shapes: Tensor, region_props: List[dict]):
        """
        Parameters:
        -----------
        features: OrderedDict
            Feature maps returned by FPN
        image_shapes: Tensor
            (B, 2) Image shapes, heights followed by widths
        region_props: List[dict]
            Region proposals with the following keys
            `boxes`: Tensor
                (N, 4) Bounding boxes
            `scores`: Tensor
                (N,) Object confidence scores
            `labels`: Tensor
                (N,) Object class indices
            `hidden_states`: Tensor
                (N, 256) Object features
        """
        
        device = features.device
        global_features = self.avg_pool(features).flatten(start_dim=1)

        boxes_h_collated = []; boxes_o_collated = []
        prior_collated = []; object_class_collated = []   
        pairwise_tokens_collated = []
        attn_maps_collated = []
       

        for b_idx, props in enumerate(region_props):             
           
            boxes = props['boxes']
            scores = props['scores']
            labels = props['labels']
            unary_tokens = props['hidden_states']    
            is_human = labels == self.human_idx
            n_h = torch.sum(is_human); n = len(boxes)  

            if not torch.all(labels[:n_h]==self.human_idx): 
                h_idx = torch.nonzero(is_human).squeeze(1)
                o_idx = torch.nonzero(is_human == 0).squeeze(1)
                perm = torch.cat([h_idx, o_idx])
              
                boxes = boxes[perm]; scores = scores[perm]
                labels = labels[perm]; unary_tokens = unary_tokens[perm]
                
            if n_h == 0 or n <= 1:
                pairwise_tokens_collated.append(torch.zeros(
                    0, 2 * self.representation_size,
                    device=device)
                )
                boxes_h_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                boxes_o_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                object_class_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                prior_collated.append(torch.zeros(2, 0, self.num_classes, device=device))
                continue
           
            x, y = torch.meshgrid(
                torch.arange(n, device=device),
                torch.arange(n, device=device)
            )
            # Valid human-object pairs
            x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < n_h)).unbind(1)
            if len(x_keep) == 0:
                raise ValueError("There are no valid human-object pairs")
            x = x.flatten(); y = y.flatten()
           
            # Compute visual spatial features
            box_pair_spatial = compute_spatial_encodings(
                [boxes[x]], [boxes[y]], [image_shapes[b_idx]]
            )
            box_pair_spatial = self.spatial_head(box_pair_spatial)        
            box_pair_spatial_reshaped = box_pair_spatial.reshape(n, n, -1) 
            unary_tokens, unary_attn = self.coop_layer(unary_tokens, box_pair_spatial_reshaped)

            # Compute text spatial features
            text_spatial_features=senmatic_spatial_feature([boxes[x_keep]],[boxes[y_keep]],[labels[y_keep]])
            # Spatial Feature Fusion
            mixed_feature=text_spatial_features+box_pair_spatial_reshaped[x_keep,y_keep]
            pairwise_tokens = torch.cat([
                self.mbf(
                    torch.cat([unary_tokens[x_keep], unary_tokens[y_keep]], 1),          
                    mixed_feature
                ), self.mbf_g(
                    global_features[b_idx, None],
                    mixed_feature
            )], dim=1)

            pairwise_tokens, pairwise_attn = self.comp_layer(pairwise_tokens)
            pairwise_tokens_collated.append(pairwise_tokens)
            boxes_h_collated.append(x_keep)
            boxes_o_collated.append(y_keep)
            object_class_collated.append(labels[y_keep])
           
            prior_collated.append(self.compute_prior_scores(
                x_keep, y_keep, scores, labels)
            )
            attn_maps_collated.append((unary_attn, pairwise_attn))  
        
        pairwise_tokens_collated = torch.cat(pairwise_tokens_collated)
        logits = self.box_pair_predictor(pairwise_tokens_collated)
        return logits, prior_collated, \
            boxes_h_collated, boxes_o_collated, object_class_collated, attn_maps_collated
