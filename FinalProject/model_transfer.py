import torch 
import torch.nn as nn 
from torchvision.models import (ResNet18_Weights, resnet18)

class TransferSiameseNet(nn.Module) : 
    def __init__(self, embed_dim = 128, dropout = 0.3, pretrained = True) : 
        super().__init__()
        weights = (ResNet18_Weights.DEFAULT if pretrained else None)
        self.backbone = resnet18(weights = weights)
        self.backbone.fc = nn.Identity()
        self.projection = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(256, embed_dim), nn.ReLU(inplace=True))
        self.compare = nn.Sequential(nn.Linear(embed_dim * 3, 128), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))


    def freeze_backbone(self) : 
        for p in self.backbone.parameters() : 
            p.requires_grad = False 

    def unfreeze_layer4(self) : 
        self.freeze_backbone()
        for p in self.backbone.layer4.parameters() : 
            p.requires_grad = True 

    def unfreeze_layer3_layer4(self) : 
        self.freeze_backbone()
        for p in self.backbone.layer3.parameters() : 
            p.requires_grad = True
        for p in self.backbone.layer4.parameters() : 
            p.requires_grad = True

    def head_parameters(self) : 
        return (list(self.projection.parameters()) + list(self.compare.parameters()))

    def backbone_trainable_parameters(self) : 
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def encode(self, x) : 
        x = self.backbone(x)
        return self.projection(x)

    def forward(self, img_a, img_b) :
        ea = self.encode(img_a)
        eb = self.encode(img_b)
        pair_features = torch.cat((torch.abs(ea - eb), ea * eb, ea + eb), dim=1)
        return self.compare(pair_features).squeeze(1)