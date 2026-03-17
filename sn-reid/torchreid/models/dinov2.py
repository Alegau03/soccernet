from __future__ import absolute_import
import torch
from torch import nn
from torch.nn import functional as F

try:
    from transformers import Dinov2Model
    from peft import LoraConfig, get_peft_model
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

__all__ = ['dinov2_vitb14_lora', 'dinov2_vits14_lora']

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class DINOv2ReID(nn.Module):
    def __init__(self, num_classes, loss='softmax', pretrained=True, use_gpu=True, model_name='facebook/dinov2-small', use_lora=True):
        super(DINOv2ReID, self).__init__()
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers and peft are required for DINOv2 models. Install them with: pip install transformers peft")
            
        self.loss = loss
        self.use_gpu = use_gpu
        self.in_planes = 768 if 'base' in model_name else 384
        
        # Load Backbone
        self.backbone = Dinov2Model.from_pretrained(model_name)
        
        if use_lora:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=["query", "value", "key", "dense"], 
                lora_dropout=0.1,
                bias="none",
                modules_to_save=[],
            )
            self.backbone = get_peft_model(self.backbone, lora_config)

        # Classifier Head
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.in_planes, num_classes, bias=False)
        
        # OCR Head (Compatibility)
        self.classifier_ocr = nn.Linear(self.in_planes, 101, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.classifier_ocr.apply(weights_init_classifier)

    def forward(self, x):
        outputs = self.backbone(pixel_values=x)
        features = outputs.last_hidden_state[:, 0, :] # [CLS] token
        
        v = features
        if self.bottleneck:
            v_bn = self.bottleneck(v)
        else:
            v_bn = v

        if not self.training:
            return v_bn

        y = self.classifier(v_bn)
        # y_ocr = self.classifier_ocr(v_bn)

        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v_bn
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

def dinov2_vitb14_lora(num_classes, loss='softmax', pretrained=True, use_gpu=True, **kwargs):
    return DINOv2ReID(
        num_classes=num_classes, 
        loss=loss, 
        pretrained=pretrained, 
        use_gpu=use_gpu, 
        model_name='facebook/dinov2-base', 
        use_lora=True
    )

def dinov2_vits14_lora(num_classes, loss='softmax', pretrained=True, use_gpu=True, **kwargs):
    return DINOv2ReID(
        num_classes=num_classes, 
        loss=loss, 
        pretrained=pretrained, 
        use_gpu=use_gpu, 
        model_name='facebook/dinov2-small', 
        use_lora=True
    )
