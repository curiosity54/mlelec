import numpy as np
import torch
import torch.nn as nn
from metatensor import TensorMap, TensorBlock, Labels

class model(nn.Module):
    def __init__(self,
                 features,
                 device=None,
                 **kwargs):

        super().__init__()
        self.features = features 
        self.device = 'cpu'
        self.bias = False
        self.model = nn.Sequential(
                nn.Linear(self.features[0].values.shape[-1],100, bias=self.bias),
                nn.Linear(100,50, bias=self.bias),
                nn.Linear(50,100, bias=self.bias),
                nn.Linear(100,self.features[0].values.shape[-1],bias=self.bias)
        ).to(self.device)

    def forward(self, features = None):
        pred_blocks = []
        if features is None:
            features = self.features
        keys = features.keys
        for k, feat in features.items():
            pred = self.model(feat.values)
            pred_blocks.append(
                    TensorBlock(
                        values = pred,
                        samples = feat.samples,
                        components = feat.components,
                        properties = feat.properties,
                    )
                )
            pred = None
        
        return TensorMap(keys, pred_blocks)

    def forward_nomts(self, features = None):
        pred_blocks = []
        if features is None:
            features = self.features
        keys = features.keys
        for k, feat in features.items():
            pred = self.model(feat.values)
            pred_blocks.append(pred)
        return pred_blocks